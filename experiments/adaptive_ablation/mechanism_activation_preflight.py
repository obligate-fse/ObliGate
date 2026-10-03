"""Hard preflight for real conflict and lifted-join mechanism activation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from obligate.theory.checker import check_policy
from obligate.theory.lattice import realization_satisfies, satisfies_all
from obligate.theory.model import (
    ActionCandidate,
    EvidenceAssertion,
    Realization,
    SignedAtom,
    TaskContract,
    canonical_json,
)
from obligate.theory.eoc import HazardSignal
from obligate.theory.policy import PolicySpec, build_default_policy
from obligate.theory.runtime import RuntimeEvidence, TheoryRuntime

from .mechanism_activation import (
    ACTIVATION_CAPABILITIES,
    activate_runtime_evidence,
    activation_manifest,
)
from .research_harness import ResearchRuntime, ResearchVariant
from .snapshots import deserialize_action, deserialize_evidence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = Path(
    os.environ.get("OBLIGATE_PREFLIGHT_SNAPSHOT_DIR")
    or ROOT / "reproduced" / "adaptive_inputs" / "agentdojo_snapshots"
).resolve()
DEFAULT_OUTPUT = Path(
    os.environ.get("OBLIGATE_MECHANISM_PREFLIGHT")
    or ROOT / "reproduced" / "adaptive_inputs" / "mechanism_activation_preflight.json"
).resolve()
MIN_ACTIVATED_CASES = 10
MAX_EVALUATED_CANDIDATES_PER_MECHANISM = 1_000


def _assertion(predicate: str, polarity: str, ref: str) -> EvidenceAssertion:
    return EvidenceAssertion(
        atom=SignedAtom(predicate, ("mock_send",), polarity),  # type: ignore[arg-type]
        evidence_ref=ref,
        source_kind="trusted_contract" if polarity == "+" else "observable_low_trust_path",
        trusted_for_authorization=polarity == "+",
    )


def _mock_action() -> ActionCandidate:
    return ActionCandidate(
        tool_name="mock_send",
        arguments={"recipient": "virtual-recipient"},
        side_effect=True,
        external_sink=True,
    )


def _base_mock_assertions() -> tuple[EvidenceAssertion, ...]:
    return tuple(
        _assertion(predicate, "+", f"mock:{predicate}")
        for predicate in (
            "schema_conforms",
            "action_authorized",
            "target_authorized",
            "flow_authorized",
        )
    )


def _runtimes() -> tuple[PolicySpec, TheoryRuntime, ResearchRuntime, ResearchRuntime]:
    policy = build_default_policy()
    acceptance = check_policy(policy).acceptance
    full = TheoryRuntime(
        acceptance=acceptance,
        policy=policy,
        capabilities=ACTIVATION_CAPABILITIES,
    )
    conflict = ResearchRuntime(
        ResearchVariant.CONFLICT_COLLAPSE,
        enabled=True,
        policy=policy,
        capabilities=ACTIVATION_CAPABILITIES,
    )
    no_join = ResearchRuntime(
        ResearchVariant.NO_LIFTED_JOIN,
        enabled=True,
        policy=policy,
        capabilities=ACTIVATION_CAPABILITIES,
    )
    return policy, full, conflict, no_join


def run_synthetic_probes() -> dict[str, Any]:
    """Prove that the named semantic changes are live before calibration."""

    policy, full_runtime, conflict_runtime, no_join_runtime = _runtimes()
    action = _mock_action()

    conflict_evidence = RuntimeEvidence(
        contract=TaskContract(),
        assertions=(*_base_mock_assertions(), _assertion("action_authorized", "-", "mock:refute")),
    )
    conflict_full = full_runtime.decide(action, conflict_evidence)
    conflict_variant = conflict_runtime.decide(action, conflict_evidence)
    original_witness = conflict_variant.trace["witness_closure_original"]["witnesses"][
        "action_authorized(mock_send)"
    ]
    effective_witness = conflict_variant.trace["witness_closure_effective"]["witnesses"][
        "action_authorized(mock_send)"
    ]
    conflict_checks = {
        "full_has_real_bipolar_conflict": original_witness["value"] == "conflict",
        "variant_collapses_only_effective_view": effective_witness["value"] == "support-only",
        "full_gap_contains_conflicted_claim": "action_authorized(mock_send)" in conflict_full.trace["gap"],
        "variant_gap_removes_collapsed_claim": "action_authorized(mock_send)"
        not in conflict_variant.trace["gap"],
        "public_or_execution_decision_diverges": _decision_signature(conflict_full)
        != _decision_signature(conflict_variant),
    }

    joined_evidence = RuntimeEvidence(
        contract=TaskContract(),
        assertions=_base_mock_assertions(),
        hazards=(
            HazardSignal("mock:sandbox", "sandbox-local control", ("mock:sandbox",), preferred_control="sandbox"),
            HazardSignal("mock:redact", "redaction-local control", ("mock:redact",), preferred_control="redact"),
        ),
    )
    joined_full = full_runtime.decide(action, joined_evidence)
    joined_variant = no_join_runtime.decide(action, joined_evidence)
    local = _local_remedy_audit(joined_full, policy)
    local_digests = set(local["feasible_local_plan_digests"])
    joined_checks = {
        "multiple_distinct_feasible_local_remedies": len(local_digests) >= 2,
        "no_single_local_remedy_satisfies_all_active_duties": local[
            "no_single_feasible_local_satisfies_all"
        ],
        "full_selects_nonlocal_joined_plan": bool(joined_full.selected)
        and joined_full.selected.plan.digest not in local_digests,
        "full_join_is_dispatchable": joined_full.execute,
        "no_lifted_join_fails_closed": not joined_variant.execute
        and joined_variant.public_decision in {"block", "block_with_compliance_error"},
        "public_or_execution_decision_diverges": _decision_signature(joined_full)
        != _decision_signature(joined_variant),
    }
    return {
        "conflict": {
            "checks": conflict_checks,
            "pass": all(conflict_checks.values()),
            "full": _decision_signature(conflict_full),
            "conflict_collapse": _decision_signature(conflict_variant),
            "original_witness_value": original_witness["value"],
            "effective_witness_value": effective_witness["value"],
        },
        "lifted_join": {
            "checks": joined_checks,
            "pass": all(joined_checks.values()),
            "full": _decision_signature(joined_full),
            "no_lifted_join": _decision_signature(joined_variant),
            "local_remedy_audit": local,
        },
        "pass": all(conflict_checks.values()) and all(joined_checks.values()),
    }


def audit_real_snapshots(snapshot_dir: Path) -> dict[str, Any]:
    """Replay real pre-policy observations through the activation profile."""

    files = tuple(sorted(snapshot_dir.glob("*.jsonl"), key=lambda item: item.name))
    if not files:
        return {
            "available": False,
            "pass": False,
            "reason": "no_agentdojo_full_snapshot_files",
            "snapshot_dir": str(snapshot_dir.resolve()),
        }
    policy, full_runtime, conflict_runtime, no_join_runtime = _runtimes()
    population_digest = hashlib.sha256()
    input_count = 0
    transformed_hazard_count = 0
    conflict_candidate_count = 0
    lifted_candidate_count = 0
    conflict_witnesses: list[dict[str, Any]] = []
    lifted_witnesses: list[dict[str, Any]] = []
    conflict_cases: set[str] = set()
    lifted_cases: set[str] = set()
    conflict_evaluated = 0
    lifted_evaluated = 0

    for path in files:
        population_digest.update(path.name.encode("utf-8"))
        population_digest.update(b"\0")
        with path.open("rb") as handle:
            for raw_line in handle:
                population_digest.update(raw_line)
                if b'"event": "input"' not in raw_line:
                    continue
                source = json.loads(raw_line)
                input_count += 1
                action = deserialize_action(source["action"])
                evidence = activate_runtime_evidence(
                    action.tool_name,
                    deserialize_evidence(source["evidence"]),
                )
                transformed_hazard_count += sum(
                    1
                    for item in (evidence.facts or {}).get(
                        "adaptive_ablation.mechanism_activation_transformations", []
                    )
                )
                case_id = str(source.get("case_id") or "unknown")
                conflict_candidate = _direct_conflict_keys(evidence)
                controls = {
                    item.preferred_control
                    for item in evidence.hazards
                    if not item.hard and item.preferred_control in {"sandbox", "redact", "require_confirmation"}
                }
                lifted_candidate = len(controls) >= 2

                if conflict_candidate:
                    conflict_candidate_count += 1
                    if conflict_evaluated < MAX_EVALUATED_CANDIDATES_PER_MECHANISM:
                        conflict_evaluated += 1
                        full = full_runtime.decide(action, evidence)
                        variant = conflict_runtime.decide(action, evidence)
                        original_values = _witness_values(
                            variant.trace["witness_closure_original"], conflict_candidate
                        )
                        effective_values = _witness_values(
                            variant.trace["witness_closure_effective"], conflict_candidate
                        )
                        real = any(value == "conflict" for value in original_values.values())
                        collapsed = any(
                            original_values.get(key) == "conflict" and effective_values.get(key) == "support-only"
                            for key in original_values
                        )
                        divergent = _decision_signature(full) != _decision_signature(variant)
                        if real and collapsed and divergent:
                            conflict_cases.add(case_id)
                            if len(conflict_witnesses) < 20:
                                conflict_witnesses.append(
                                    {
                                        "snapshot_id": source.get("snapshot_id"),
                                        "case_id": case_id,
                                        "conflict_keys": sorted(conflict_candidate),
                                        "original_values": original_values,
                                        "effective_values": effective_values,
                                        "full": _decision_signature(full),
                                        "conflict_collapse": _decision_signature(variant),
                                    }
                                )

                if lifted_candidate:
                    lifted_candidate_count += 1
                    if lifted_evaluated < MAX_EVALUATED_CANDIDATES_PER_MECHANISM:
                        lifted_evaluated += 1
                        full = full_runtime.decide(action, evidence)
                        local = _local_remedy_audit(full, policy)
                        local_digests = set(local["feasible_local_plan_digests"])
                        selected_is_joined = bool(full.selected) and full.selected.plan.digest not in local_digests
                        if len(local_digests) >= 2 and selected_is_joined:
                            variant = no_join_runtime.decide(action, evidence)
                            divergent = _decision_signature(full) != _decision_signature(variant)
                            if divergent:
                                lifted_cases.add(case_id)
                                if len(lifted_witnesses) < 20:
                                    lifted_witnesses.append(
                                        {
                                            "snapshot_id": source.get("snapshot_id"),
                                            "case_id": case_id,
                                            "active_local_controls": sorted(controls),
                                            "full": _decision_signature(full),
                                            "no_lifted_join": _decision_signature(variant),
                                            "local_remedy_audit": local,
                                        }
                                    )

    checks = {
        "real_input_population_present": input_count > 0,
        "observable_hazards_transformed": transformed_hazard_count > 0,
        "conflict_candidates_present": conflict_candidate_count > 0,
        "real_conflict_divergence_in_at_least_ten_cases": len(conflict_cases) >= MIN_ACTIVATED_CASES,
        "multiple_local_control_candidates_present": lifted_candidate_count > 0,
        "real_lifted_join_divergence_in_at_least_ten_cases": len(lifted_cases) >= MIN_ACTIVATED_CASES,
    }
    return {
        "available": True,
        "snapshot_dir": str(snapshot_dir.resolve()),
        "snapshot_file_count": len(files),
        "population_sha256": population_digest.hexdigest(),
        "input_snapshot_count": input_count,
        "transformed_hazard_count": transformed_hazard_count,
        "conflict_candidate_snapshot_count": conflict_candidate_count,
        "conflict_evaluated_count": conflict_evaluated,
        "conflict_activated_unique_case_count_lower_bound": len(conflict_cases),
        "lifted_join_candidate_snapshot_count": lifted_candidate_count,
        "lifted_join_evaluated_count": lifted_evaluated,
        "lifted_join_activated_unique_case_count_lower_bound": len(lifted_cases),
        "evaluation_cap_per_mechanism": MAX_EVALUATED_CANDIDATES_PER_MECHANISM,
        "minimum_unique_cases_required": MIN_ACTIVATED_CASES,
        "checks": checks,
        "conflict_witnesses": conflict_witnesses,
        "lifted_join_witnesses": lifted_witnesses,
        "pass": all(checks.values()),
        "uses_benchmark_labels_or_scorer_fields": False,
        "external_actions_executed": False,
        "victim_model_calls": 0,
    }


def run_preflight(snapshot_dir: Path, output: Path) -> dict[str, Any]:
    synthetic = run_synthetic_probes()
    calibration = audit_real_snapshots(snapshot_dir)
    allowed = bool(synthetic["pass"] and calibration["pass"])
    value = {
        "schema_version": "obligate-mechanism-activation-preflight-v1",
        "activation_profile": activation_manifest(),
        "synthetic_semantic_probes": synthetic,
        "real_snapshot_calibration": calibration,
        "formal_run_allowed": allowed,
        "gate_semantics": (
            "Formal execution must stop before any victim-model call unless formal_run_allowed is true."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def _direct_conflict_keys(evidence: RuntimeEvidence) -> set[str]:
    positive = {item.atom.unsigned_key for item in evidence.assertions if item.atom.polarity == "+"}
    negative = {item.atom.unsigned_key for item in evidence.assertions if item.atom.polarity == "-"}
    return positive & negative


def _witness_values(closure: Mapping[str, Any], keys: Iterable[str]) -> dict[str, str]:
    witnesses = closure.get("witnesses") or {}
    return {
        key: str((witnesses.get(key) or {}).get("value") or "unknown")
        for key in sorted(set(keys))
    }


def _materialize(template: Realization) -> Realization:
    required = set(template.capabilities)
    return replace(
        template,
        realization_id=f"preflight:{template.realization_id}",
        capabilities=frozenset(required & ACTIVATION_CAPABILITIES),
        actual_audit=template.actual_audit if "audit" in ACTIVATION_CAPABILITIES else frozenset(),
        available=template.available and required <= ACTIVATION_CAPABILITIES,
    )


def _local_remedy_audit(decision: Any, policy: PolicySpec) -> dict[str, Any]:
    plans_by_json = {canonical_json(item.to_dict()): item for item in policy.behavior_domain}
    triggers = {item.trigger_id: item for item in decision.triggers}
    feasible: dict[str, Any] = {}
    for trigger_id, remedies in (decision.trace["eoc"].get("remedies") or {}).items():
        trigger = triggers[trigger_id]
        for remedy in remedies:
            plan = plans_by_json.get(canonical_json(remedy["plan"]))
            if plan is None:
                continue
            own_semantics = satisfies_all(plan, trigger.semantic_obligations)
            own_realization = any(
                realization_satisfies(
                    _materialize(template),
                    plan,
                    trigger.implementation_constraints,
                )
                for template in policy.realizations
            )
            if own_semantics and own_realization:
                feasible[plan.digest] = {
                    "trigger_id": trigger_id,
                    "plan": {
                        "execution_env": plan.execution_env,
                        "network_scope": plan.network_scope,
                        "data_scope": plan.data_scope,
                        "human_gate": plan.human_gate,
                        "audit_must_count": len(plan.audit.must),
                    },
                    "satisfies_all_active_duties": satisfies_all(
                        plan, decision.semantic_obligations
                    ),
                }
    return {
        "feasible_local_plan_digests": sorted(feasible),
        "feasible_local_remedies": [feasible[key] for key in sorted(feasible)],
        "no_single_feasible_local_satisfies_all": bool(feasible)
        and not any(item["satisfies_all_active_duties"] for item in feasible.values()),
    }


def _decision_signature(decision: Any) -> dict[str, Any]:
    return {
        "public_decision": decision.public_decision,
        "execute": bool(decision.execute),
        "selected_plan_digest": decision.selected.plan.digest if decision.selected else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentdojo-snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value = run_preflight(args.agentdojo_snapshot_dir, args.output)
    print(
        json.dumps(
            {
                "formal_run_allowed": value["formal_run_allowed"],
                "synthetic_pass": value["synthetic_semantic_probes"]["pass"],
                "real_snapshot_calibration_pass": value["real_snapshot_calibration"]["pass"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if value["formal_run_allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_real_snapshots", "run_preflight", "run_synthetic_probes"]
