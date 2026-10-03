"""Pre-policy snapshot capture and deterministic Full replay.

The recorder writes an ``input`` event *before* calling ``TheoryRuntime.decide``
and a separate ``decision`` event afterwards.  A third ``tool_gate`` event is
captured when the production gate processes the resulting certificate.  This
ordering prevents an aggregate result from being mistaken for a pre-policy
snapshot.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from obligate.theory.gate import ToolGate
from obligate.theory.model import (
    ActionCandidate,
    ConfirmationGrant,
    EvidenceAssertion,
    GraphEdge,
    GraphNode,
    SignedAtom,
    TaskContract,
    TheoryDecision,
    canonical_json,
    sha256_json,
)
from obligate.theory.eoc import HazardSignal
from obligate.theory.policy import build_default_policy
from obligate.theory.runtime import RuntimeEvidence, TheoryRuntime, _replay_checked_policy

from .snapshot_index import SnapshotEventIndex

SNAPSHOT_SCHEMA_VERSION = "obligate-pre-policy-snapshot-v1"
DETERMINISTIC_FIELDS_VERSION = "obligate-deterministic-replay-v1"
_CASE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("obligate_snapshot_case_id", default="unknown")
_CURRENT_SNAPSHOT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "obligate_current_snapshot", default=None
)


@contextlib.contextmanager
def snapshot_case(case_id: str) -> Iterator[None]:
    token = _CASE_ID.set(str(case_id))
    try:
        yield
    finally:
        _CASE_ID.reset(token)


class SnapshotRecorder:
    """Thread-safe experiment-only monkeypatch around unchanged production code."""

    def __init__(self, path: Path, *, benchmark: str, variant: str = "full") -> None:
        self.path = path.resolve()
        self.benchmark = benchmark
        self.variant = variant
        self._lock = threading.RLock()
        self._counter = 0
        self._original_decide: Callable[..., Any] | None = None
        self._original_gate_process: Callable[..., Any] | None = None

    def __enter__(self) -> "SnapshotRecorder":
        if self._original_decide is not None:
            raise RuntimeError("snapshot recorder is already installed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._original_decide = TheoryRuntime.decide
        self._original_gate_process = ToolGate.process
        recorder = self

        def decide_wrapper(
            runtime: TheoryRuntime,
            action: ActionCandidate,
            evidence: RuntimeEvidence,
            *,
            nonce: str | None = None,
        ) -> TheoryDecision:
            resolved_case_id = _resolve_case_id(evidence)
            snapshot_id = recorder._next_id(action, case_id=resolved_case_id)
            recorder._append(
                {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "event": "input",
                    "snapshot_id": snapshot_id,
                    "captured_before_decision": True,
                    "captured_at": _utc_now(),
                    "benchmark": recorder.benchmark,
                    "case_id": resolved_case_id,
                    "variant": recorder.variant,
                    "action": serialize_action(action),
                    "evidence": serialize_evidence(evidence),
                    "runtime": serialize_runtime(runtime),
                    "nonce_argument": nonce,
                }
            )
            _CURRENT_SNAPSHOT.set(snapshot_id)
            try:
                assert recorder._original_decide is not None
                decision = recorder._original_decide(runtime, action, evidence, nonce=nonce)
            except BaseException as exc:
                recorder._append(
                    {
                        "schema_version": SNAPSHOT_SCHEMA_VERSION,
                        "event": "decision_error",
                        "snapshot_id": snapshot_id,
                        "captured_at": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                )
                raise
            recorder._append(
                {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "event": "decision",
                    "snapshot_id": snapshot_id,
                    "captured_at": _utc_now(),
                    "deterministic_fields_version": DETERMINISTIC_FIELDS_VERSION,
                    "deterministic": deterministic_projection(decision),
                }
            )
            return decision

        def gate_wrapper(gate: ToolGate, **kwargs: Any) -> Any:
            assert recorder._original_gate_process is not None
            result = recorder._original_gate_process(gate, **kwargs)
            snapshot_id = _CURRENT_SNAPSHOT.get()
            if snapshot_id is not None:
                certificate = kwargs.get("certificate")
                certificate_digest = str(getattr(certificate, "digest", "") or "")
                recorder._append(
                    {
                        "schema_version": SNAPSHOT_SCHEMA_VERSION,
                        "event": "tool_gate",
                        "snapshot_id": snapshot_id,
                        "captured_at": _utc_now(),
                        # Keep the digest inside the compact ``gate`` mapping so
                        # snapshot consolidation preserves the exact
                        # certificate that production ToolGate authorized.
                        "gate": gate_projection(
                            result,
                            certificate_digest=certificate_digest,
                        ),
                        "dispatch_intent": bool(kwargs.get("dispatcher"))
                        and str(getattr(kwargs.get("certificate"), "outcome", ""))
                        in {"allow", "execute_with_constraints"},
                        "idempotency_key": kwargs.get("idempotency_key"),
                    }
                )
            return result

        TheoryRuntime.decide = decide_wrapper  # type: ignore[method-assign]
        ToolGate.process = gate_wrapper  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._original_decide is not None
        assert self._original_gate_process is not None
        TheoryRuntime.decide = self._original_decide  # type: ignore[method-assign]
        ToolGate.process = self._original_gate_process  # type: ignore[method-assign]
        self._original_decide = None
        self._original_gate_process = None

    def _next_id(self, action: ActionCandidate, *, case_id: str) -> str:
        with self._lock:
            self._counter += 1
            material = f"{self.benchmark}\0{case_id}\0{self._counter}\0{action.digest}"
            return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _append(self, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def serialize_action(action: ActionCandidate) -> dict[str, Any]:
    return action.normalized_dict()


def deserialize_action(value: Mapping[str, Any]) -> ActionCandidate:
    return ActionCandidate(
        tool_name=str(value["tool_name"]),
        arguments=dict(value.get("arguments") or {}),
        actor=str(value.get("actor") or "benchmark_agent"),
        tenant=str(value.get("tenant") or "local"),
        session=str(value.get("session") or "default"),
        side_effect=bool(value.get("side_effect")),
        external_sink=bool(value.get("external_sink")),
        state_mutation=bool(value.get("state_mutation")),
        sensitive_read=bool(value.get("sensitive_read")),
        privileged_access=bool(value.get("privileged_access")),
        schema_known=bool(value.get("schema_known", True)),
        payload_digest=str(value.get("payload_digest") or ""),
        asset_digests=tuple(str(item) for item in value.get("asset_digests") or ()),
    )


def serialize_evidence(evidence: RuntimeEvidence) -> dict[str, Any]:
    return {
        "contract": evidence.contract.to_dict(),
        "assertions": [item.to_dict() for item in evidence.assertions],
        "hazards": [_hazard_to_dict(item) for item in evidence.hazards],
        "explicit_denies": [_hazard_to_dict(item) for item in evidence.explicit_denies],
        "graph_nodes": [item.to_dict() for item in evidence.graph_nodes],
        "graph_edges": [item.to_dict() for item in evidence.graph_edges],
        "facts": dict(evidence.facts or {}),
        "confirmation": evidence.confirmation.to_dict() if evidence.confirmation else None,
    }


def deserialize_evidence(value: Mapping[str, Any]) -> RuntimeEvidence:
    contract = value.get("contract") or {}
    confirmation = value.get("confirmation")
    return RuntimeEvidence(
        contract=TaskContract(
            permit=frozenset(str(item) for item in contract.get("permit") or ()),
            deny=frozenset(str(item) for item in contract.get("deny") or ()),
            unresolved=frozenset(str(item) for item in contract.get("unresolved") or ()),
            source_digest=str(contract.get("source_digest") or ""),
        ),
        assertions=tuple(_assertion_from_dict(item) for item in value.get("assertions") or ()),
        hazards=tuple(_hazard_from_dict(item) for item in value.get("hazards") or ()),
        explicit_denies=tuple(_hazard_from_dict(item) for item in value.get("explicit_denies") or ()),
        graph_nodes=tuple(_node_from_dict(item) for item in value.get("graph_nodes") or ()),
        graph_edges=tuple(_edge_from_dict(item) for item in value.get("graph_edges") or ()),
        facts=dict(value.get("facts") or {}),
        confirmation=(
            ConfirmationGrant(
                nonce=str(confirmation["nonce"]),
                confirmation_binding_digest=str(confirmation["confirmation_binding_digest"]),
                initial_certificate_digest=str(confirmation["initial_certificate_digest"]),
            )
            if isinstance(confirmation, Mapping)
            else None
        ),
    )


def serialize_runtime(runtime: TheoryRuntime) -> dict[str, Any]:
    return {
        "policy_digest": runtime.policy.digest,
        "policy_acceptance_digest": runtime.acceptance.digest,
        "loader_replay_digest": runtime.loader_replay_digest,
        "witness_limit": runtime.witness_limit,
        "max_candidates": runtime.max_candidates,
        "capabilities": sorted(runtime.capabilities),
    }


def deterministic_projection(decision: TheoryDecision) -> dict[str, Any]:
    trace = dict(decision.trace)
    return {
        "public_decision": decision.public_decision,
        "execute": decision.execute,
        "triggers": [item.to_dict() for item in decision.triggers],
        "semantic_obligations": [item.to_dict() for item in decision.semantic_obligations],
        "implementation_constraints": [item.to_dict() for item in decision.implementation_constraints],
        "ideal": [item.to_dict() for item in decision.ideal],
        "safe_candidates": [item.to_dict() for item in decision.safe_candidates],
        "feasible_realizations": [item.to_dict() for item in decision.feasible_realizations],
        "frontier": [item.to_dict() for item in decision.frontier],
        "selected": decision.selected.to_dict() if decision.selected else None,
        "certificate": decision.certificate.to_dict(),
        "certificate_digest": decision.certificate.digest,
        "trace": {
            key: trace.get(key)
            for key in (
                "semantic_version",
                "policy_digest",
                "policy_acceptance_digest",
                "loader_replay_digest",
                "guarded",
                "approval_satisfied",
                "action_graph",
                "witness_closure",
                "rejected_enabling_assertions",
                "gap",
                "hazard",
                "overflow",
                "valid_certificate_evidence",
                "eoc",
                "lattice",
            )
        },
    }


def gate_projection(
    result: Any,
    *,
    certificate_digest: str | None = None,
) -> dict[str, Any]:
    outbox = getattr(result, "outbox", None)
    state = getattr(result, "state", None)
    dispatched = bool(getattr(result, "dispatched", False))
    return {
        "decision": getattr(result, "decision", None),
        "state": state,
        "dispatched": dispatched,
        # Additive/backward-compatible field.  Older snapshots remain
        # replayable, while integrity experiments can require this binding for
        # newly captured dispatched actions.
        "certificate_digest": certificate_digest,
        "nonce": getattr(result, "nonce", None),
        "reason": getattr(result, "reason", ""),
        "outbox_status": getattr(outbox, "status", None),
        "outbox_attempts": getattr(outbox, "attempts", None),
        "authorization_result": (
            "authorized" if dispatched else "pending" if state == "pending" else "not_authorized"
        ),
        "simulated_state_transitions": _state_transitions(state, dispatched=dispatched),
    }


def replay_snapshot_file(
    snapshot_path: Path,
    output_path: Path,
    *,
    event_index_path: Path | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if event_index_path is None:
        event_index_path = output_path.with_suffix(".events.sqlite3")
    total = 0
    matched = 0
    gate_total = 0
    gate_matched = 0
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    policy = build_default_policy()
    acceptance = _replay_checked_policy(policy)
    with SnapshotEventIndex.build(snapshot_path, event_index_path) as index, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in index.records():
            snapshot_id = record.snapshot_id
            source = index.load(record.input_offset, record.input_length)
            expected_event = index.load(record.decision_offset, record.decision_length)
            if source is None or expected_event is None:
                missing.append(snapshot_id)
                continue
            total += 1
            runtime_info = source["runtime"]
            if runtime_info["policy_digest"] != policy.digest or runtime_info["policy_acceptance_digest"] != acceptance.digest:
                raise ValueError(f"snapshot {snapshot_id} does not bind the current replay-checked Full policy")
            runtime = TheoryRuntime(
                acceptance=acceptance,
                policy=policy,
                witness_limit=runtime_info.get("witness_limit"),
                max_candidates=runtime_info.get("max_candidates"),
                capabilities=runtime_info.get("capabilities") or (),
            )
            action = deserialize_action(source["action"])
            evidence = deserialize_evidence(source["evidence"])
            actual_decision = runtime.decide(action, evidence, nonce=source.get("nonce_argument"))
            actual = deterministic_projection(actual_decision)
            expected = expected_event["deterministic"]
            actual = _bind_online_issuer_projection(actual, expected)
            equal = canonical_json(actual) == canonical_json(expected)
            if equal:
                matched += 1
            else:
                mismatches.append(
                    {
                        "snapshot_id": snapshot_id,
                        "case_id": source.get("case_id"),
                        "differing_paths": _diff_paths(expected, actual),
                    }
                )
            gate_expected_event = index.load(record.tool_gate_offset, record.tool_gate_length)
            gate_equal: bool | None = None
            gate_differing_paths: list[str] = []
            if gate_expected_event is not None:
                gate_total += 1
                gate_expected = _normalized_gate_projection(gate_expected_event.get("gate") or {})
                gate_actual = _replay_gate_projection(
                    snapshot_id=snapshot_id,
                    expected=gate_expected,
                    action=action,
                    decision=actual_decision,
                    policy=policy,
                    acceptance=acceptance,
                    idempotency_key=gate_expected_event.get("idempotency_key"),
                )
                gate_differing_paths = _diff_paths(gate_expected, gate_actual)
                gate_equal = not gate_differing_paths
                if gate_equal:
                    gate_matched += 1
            row = {
                "snapshot_id": snapshot_id,
                "case_id": source.get("case_id"),
                "decision_match": equal,
                "gate_observed_online": gate_expected_event is not None,
                "tool_gate_replay_match": gate_equal,
                "tool_gate_differing_paths": gate_differing_paths,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": DETERMINISTIC_FIELDS_VERSION,
        "snapshot_file": str(snapshot_path.resolve()),
        "snapshot_file_sha256": _sha256_file(snapshot_path),
        "snapshot_event_index": str(event_index_path.resolve()),
        "complete_snapshot_count": total,
        "decision_match_count": matched,
        "decision_match_rate": 0.0 if total == 0 else matched / total,
        "decision_fields_100_percent_match": total > 0 and matched == total,
        "certificate_digest_100_percent_match": total > 0 and matched == total,
        "online_process_issuer_fields_reused_for_digest_recalculation": True,
        "issuer_secret_or_proof_generation_replayed": False,
        "tool_gate_replay_count": gate_total,
        "tool_gate_replay_match_count": gate_matched,
        "tool_gate_replay_match_rate": 0.0 if gate_total == 0 else gate_matched / gate_total,
        "mock_toolgate_authorization_100_percent_match": gate_total > 0 and gate_matched == gate_total,
        "simulated_nonce_outbox_transitions_100_percent_match": gate_total > 0 and gate_matched == gate_total,
        "incomplete_snapshot_ids": missing,
        "mismatches": mismatches[:100],
        "external_side_effects_replayed": False,
        "scope_note": "Remote tool responses and external side effects are intentionally not replayed.",
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _bind_online_issuer_projection(actual: dict[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Bind opaque per-process issuer fields while replaying every unsigned field.

    Formal AgentDojo cases run in separate processes whose in-memory HMAC keys
    are intentionally not logged.  The stored issuer id/proof are therefore
    treated as opaque online evidence.  Recomputing the certificate digest over
    the replayed unsigned fields plus those stored opaque fields detects any
    deterministic certificate-payload drift without persisting the issuer key.
    """

    actual_certificate = actual.get("certificate")
    expected_certificate = expected.get("certificate")
    if not isinstance(actual_certificate, dict) or not isinstance(expected_certificate, Mapping):
        return actual
    bound = dict(actual_certificate)
    bound["issuer_id"] = expected_certificate.get("issuer_id", "")
    bound["issuer_proof"] = expected_certificate.get("issuer_proof", "")
    value = dict(actual)
    value["certificate"] = bound
    value["certificate_digest"] = sha256_json(bound)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_gate_projection(
    *,
    snapshot_id: str,
    expected: Mapping[str, Any],
    action: ActionCandidate,
    decision: TheoryDecision,
    policy: Any,
    acceptance: Any,
    idempotency_key: str | None,
) -> dict[str, Any]:
    if decision.selected is None:
        return _normalized_gate_projection(
            {
                "decision": "block",
                "state": "blocked",
                "dispatched": False,
                "nonce": None,
                "reason": "replay decision has no realization",
                "outbox_status": None,
                "outbox_attempts": None,
            }
        )
    gate = ToolGate(policy=policy, acceptance=acceptance)

    def dispatcher(action_value: Any, realization_value: Any, key: str | None) -> Any:
        del action_value, realization_value, key
        if expected.get("state") == "uncertain":
            raise RuntimeError("simulated replay delivery uncertainty")
        return {"simulated": True, "snapshot_id": snapshot_id}

    certificate = decision.certificate
    result = gate.process(
        action=action,
        certificate=certificate,
        policy=acceptance,
        realization=decision.selected,
        fact_digest=certificate.fact_digest,
        evidence_digest=certificate.evidence_digest,
        plan_digest=certificate.plan_digest,
        dispatcher=dispatcher if expected.get("dispatched") else None,
        idempotency_key=idempotency_key,
        nonce=(str(expected["nonce"]) if expected.get("nonce") else None),
    )
    return _normalized_gate_projection(gate_projection(result))


def _normalized_gate_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    state = value.get("state")
    dispatched = bool(value.get("dispatched"))
    return {
        "decision": value.get("decision"),
        "state": state,
        "dispatched": dispatched,
        "nonce": value.get("nonce"),
        # Error strings may contain provider-specific exception messages and
        # are not deterministic authorization state.
        "reason_present": bool(value.get("reason")),
        "outbox_status": value.get("outbox_status"),
        "outbox_attempts": value.get("outbox_attempts"),
        "authorization_result": value.get("authorization_result")
        or ("authorized" if dispatched else "pending" if state == "pending" else "not_authorized"),
        "simulated_state_transitions": value.get("simulated_state_transitions")
        or _state_transitions(state, dispatched=dispatched),
    }


def _state_transitions(state: Any, *, dispatched: bool) -> list[str]:
    if state == "pending":
        return ["pending"]
    if state == "done" and dispatched:
        return ["claimed", "sending", "done"]
    if state == "uncertain" and dispatched:
        return ["claimed", "sending", "uncertain"]
    if state == "blocked":
        return ["blocked"]
    return [str(state)] if state is not None else []


def _assertion_from_dict(value: Mapping[str, Any]) -> EvidenceAssertion:
    return EvidenceAssertion(
        atom=_atom_from_dict(value["atom"]),
        evidence_ref=str(value["evidence_ref"]),
        source_kind=str(value["source_kind"]),
        trusted_for_authorization=bool(value.get("trusted_for_authorization")),
    )


def _atom_from_dict(value: Mapping[str, Any]) -> SignedAtom:
    return SignedAtom(
        predicate=str(value["predicate"]),
        arguments=tuple(str(item) for item in value.get("arguments") or ()),
        polarity=str(value.get("polarity") or "+"),  # type: ignore[arg-type]
    )


def _node_from_dict(value: Mapping[str, Any]) -> GraphNode:
    return GraphNode(
        node_id=str(value["node_id"]),
        node_type=str(value["node_type"]),
        label=str(value["label"]),
        trust=str(value.get("trust") or "unknown"),  # type: ignore[arg-type]
        metadata=dict(value.get("metadata") or {}),
    )


def _edge_from_dict(value: Mapping[str, Any]) -> GraphEdge:
    return GraphEdge(
        edge_id=str(value["edge_id"]),
        source=str(value["source"]),
        target=str(value["target"]),
        relation=str(value["relation"]),
        evidence_refs=tuple(str(item) for item in value.get("evidence_refs") or ()),
    )


def _hazard_to_dict(value: HazardSignal) -> dict[str, Any]:
    return {
        "signal_id": value.signal_id,
        "subject": value.subject,
        "evidence_refs": list(value.evidence_refs),
        "hard": value.hard,
        "preferred_control": value.preferred_control,
        "kind": value.kind,
    }


def _hazard_from_dict(value: Mapping[str, Any]) -> HazardSignal:
    return HazardSignal(
        signal_id=str(value["signal_id"]),
        subject=str(value["subject"]),
        evidence_refs=tuple(str(item) for item in value.get("evidence_refs") or ()),
        hard=bool(value.get("hard")),
        preferred_control=str(value.get("preferred_control") or "block"),
        kind=str(value.get("kind") or "hazard"),
    )


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "$"]
    if isinstance(left, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths[:100]
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{prefix}.length"]
        paths: list[str] = []
        for index, (a, b) in enumerate(zip(left, right)):
            paths.extend(_diff_paths(a, b, f"{prefix}[{index}]"))
        return paths[:100]
    return [] if left == right else [prefix or "$"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_case_id(evidence: RuntimeEvidence) -> str:
    explicit = _CASE_ID.get()
    if explicit != "unknown":
        return explicit
    facts = evidence.facts or {}
    for key in (
        "agent_safetybench.case_id",
        "asb.case_id",
        "agent_security_bench.case_id",
        "case_id",
    ):
        value = facts.get(key)
        if value is not None and str(value):
            return str(value)
    return "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Full pre-policy snapshots")
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-index", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = replay_snapshot_file(args.snapshots, args.output, event_index_path=args.event_index)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["decision_fields_100_percent_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SnapshotRecorder",
    "deserialize_action",
    "deserialize_evidence",
    "deterministic_projection",
    "replay_snapshot_file",
    "snapshot_case",
]
