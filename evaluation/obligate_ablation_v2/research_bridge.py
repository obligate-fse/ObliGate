"""AgentDojo simulator bridge for ObliGate ablation-v2 research variants.

The bridge is installed explicitly in a case worker, labels every artifact as
research-only, and restores every monkey patch on uninstall. It may execute a
counterfactual decision inside AgentDojo, but it never creates or claims a
production ActionCertificate or policy acceptance record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from obligate.theory.gate import GateResult
from obligate.theory.model import (
    ActionCandidate,
    Realization,
    canonical_json,
    sha256_json,
)
from obligate.theory.runtime import RuntimeEvidence

from .variant_runtime import (
    BOUND_RECORD_FACT_KEY,
    BoundEvidenceRecord,
    ResearchDecision,
    ResearchRuntime,
    ResearchVariant,
    ScalarThresholds,
    bound_records_for_action,
)


def _markers() -> dict[str, Any]:
    return {
        "research_only": True,
        "verified": False,
        "research_counterfactual_mode": True,
        "acceptance_record": None,
        "action_certificate": None,
        "main_theorem_guarantee": False,
    }


class DecisionTraceWriter:
    """Thread-safe append-only JSONL sink for input/decision/cache events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def __call__(self, event: Mapping[str, Any]) -> None:
        payload = {**dict(event), **_markers()}
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()


@dataclass(frozen=True, slots=True)
class ResearchDispatchBinding:
    """Unsigned simulator binding; intentionally not an ActionCertificate."""

    policy_definition_digest: str
    action_digest: str
    fact_digest: str
    evidence_digest: str
    plan_digest: str
    realization_digest: str
    outcome: str
    issuer_id: str = "research-only-unverified"
    issuer_proof: str = ""
    nonce: None = None
    prior_certificate_digest: None = None
    confirmation_binding_digest: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_type": "research-dispatch-binding-not-certificate",
            "policy_definition_digest": self.policy_definition_digest,
            "policy_acceptance_digest": None,
            "action_digest": self.action_digest,
            "fact_digest": self.fact_digest,
            "evidence_digest": self.evidence_digest,
            "plan_digest": self.plan_digest,
            "realization_digest": self.realization_digest,
            "outcome": self.outcome,
            "issuer_id": self.issuer_id,
            "issuer_proof": "",
            "nonce": None,
            "prior_certificate_digest": None,
            "confirmation_binding_digest": None,
            **_markers(),
        }

    @property
    def policy_acceptance_digest(self) -> None:
        # The compatibility projection may read this attribute, but research
        # artifacts must never contain a value that resembles Accept(kappa).
        return None

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ResearchDecisionAdapter:
    """Shape adapter for the existing firewall compatibility projection."""

    core: ResearchDecision
    certificate: ResearchDispatchBinding

    @property
    def public_decision(self) -> str:
        return self.core.public_decision

    @property
    def execute(self) -> bool:
        return self.core.execute

    @property
    def triggers(self):
        return self.core.triggers

    @property
    def semantic_obligations(self):
        return self.core.semantic_obligations

    @property
    def implementation_constraints(self):
        return self.core.implementation_constraints

    @property
    def ideal(self):
        return self.core.ideal

    @property
    def safe_candidates(self):
        return self.core.safe_candidates

    @property
    def feasible_realizations(self):
        return self.core.feasible_realizations

    @property
    def frontier(self):
        return self.core.frontier

    @property
    def selected(self):
        return self.core.selected

    @property
    def trace(self) -> Mapping[str, Any]:
        source = dict(self.core.trace)
        return {
            **source,
            "semantic_version": "research-only/unverified",
            "policy_acceptance_digest": None,
            "loader_replay_digest": None,
            "witness_closure": source["witness_closure_effective"],
            "valid_certificate_evidence": False,
            "research_dispatch_binding_digest": self.certificate.digest,
            **_markers(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.core.to_dict(),
            "certificate": None,
            "research_dispatch_binding": self.certificate.to_dict(),
            "research_dispatch_binding_digest": self.certificate.digest,
            **_markers(),
        }


class ResearchRuntimeAdapter:
    def __init__(
        self,
        variant: ResearchVariant | str,
        capabilities: Sequence[str],
        *,
        case_id: str,
        event_sink: DecisionTraceWriter,
        scalar_thresholds: ScalarThresholds | Mapping[str, Any] | Sequence[float] | None,
    ) -> None:
        self.core = ResearchRuntime(
            variant,
            research_counterfactual_mode=True,
            capabilities=tuple(capabilities),
            case_id=case_id,
            event_sink=event_sink,
            scalar_thresholds=scalar_thresholds,  # type: ignore[arg-type]
        )
        self.policy = self.core.policy
        self.acceptance = None
        self.certificate_issuer = None
        self.variant = self.core.variant

    def decide(self, action: ActionCandidate, evidence: RuntimeEvidence) -> ResearchDecisionAdapter:
        core = self.core.decide(action, evidence)
        selected = core.selected
        binding = ResearchDispatchBinding(
            policy_definition_digest=self.core.policy.digest,
            action_digest=action.digest,
            fact_digest=str(core.trace["fact_digest"]),
            evidence_digest=str(core.trace["evidence_digest"]),
            plan_digest=(selected.plan.digest if selected else sha256_json({"plan": "none"})),
            realization_digest=(selected.digest if selected else sha256_json({"realization": "none"})),
            outcome=core.public_decision,
        )
        return ResearchDecisionAdapter(core, binding)


@dataclass(frozen=True, slots=True)
class ResearchOutbox:
    status: str
    result_digest: str = ""
    attempts: int = 1
    transitions: tuple[str, ...] = ("fresh", "claimed", "sending", "done")
    research_only: bool = True
    verified: bool = False
    research_counterfactual_mode: bool = True


class ResearchBindingGate:
    """Atomic simulator gate for unsigned research dispatch bindings."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._lock = RLock()
        self._operations: dict[str, str] = {}
        self.research_only = True
        self.verified = False
        self.research_counterfactual_mode = True
        self.acceptance_record = None

    def process(
        self,
        *,
        action: ActionCandidate,
        certificate: ResearchDispatchBinding,
        policy: Any,
        realization: Realization,
        fact_digest: str,
        evidence_digest: str,
        plan_digest: str,
        dispatcher: Any = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> GateResult:
        del policy, kwargs
        if not isinstance(certificate, ResearchDispatchBinding):
            return GateResult(
                "block",
                "blocked",
                False,
                reason="formal certificate rejected by research bridge",
            )
        if certificate.action_digest != action.digest:
            return GateResult("block", "blocked", False, reason="research action binding mismatch")
        if certificate.fact_digest != fact_digest or certificate.evidence_digest != evidence_digest:
            return GateResult("block", "blocked", False, reason="research evidence binding mismatch")
        if certificate.plan_digest != plan_digest or certificate.realization_digest != realization.digest:
            return GateResult("block", "blocked", False, reason="research realization binding mismatch")
        if certificate.outcome in {"block", "block_with_compliance_error"}:
            return GateResult(
                certificate.outcome,  # type: ignore[arg-type]
                "blocked",
                False,
                reason="research non-dispatch outcome",
            )
        if certificate.outcome == "require_confirmation":
            return GateResult(
                "require_confirmation",
                "pending",
                False,
                reason="research confirmation pause; no production nonce issued",
            )
        if dispatcher is None:
            return GateResult("block", "blocked", False, reason="research dispatch requires dispatcher")

        operation = sha256_json(
            {
                "action": action.normalized_dict(),
                "binding": certificate.to_dict(),
                "idempotency_key": idempotency_key,
            }
        )
        with self._lock:
            if operation in self._operations:
                return GateResult(
                    "block",
                    "blocked",
                    False,
                    reason="research duplicate dispatch rejected",
                )
            self._operations[operation] = "sending"
        try:
            result = dispatcher(action, realization, idempotency_key)
        except BaseException as exc:
            with self._lock:
                self._operations[operation] = "uncertain"
            outbox = ResearchOutbox(
                "uncertain",
                transitions=("fresh", "claimed", "sending", "uncertain"),
            )
            return GateResult(
                certificate.outcome,  # type: ignore[arg-type]
                "uncertain",
                True,
                reason=f"research delivery uncertain:{type(exc).__name__}",
                outbox=outbox,  # type: ignore[arg-type]
            )
        with self._lock:
            self._operations[operation] = "done"
        outbox = ResearchOutbox("done", result_digest=sha256_json(_result_digest_material(result)))
        return GateResult(
            certificate.outcome,  # type: ignore[arg-type]
            "done",
            True,
            result=result,
            outbox=outbox,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class BridgeOriginals:
    tool_gate: Any
    runtime_builder: Any
    boundary_adapter: Any
    realization_validator_descriptor: Any


def install_research_bridge(
    firewall_module: Any,
    variant: ResearchVariant | str,
    *,
    case_id: str,
    decision_trace_path: str | Path,
    scalar_thresholds: ScalarThresholds | Mapping[str, Any] | Sequence[float] | None = None,
) -> BridgeOriginals:
    """Install one case-scoped research bridge and return restoration handles."""

    selected = ResearchVariant(variant)
    writer = DecisionTraceWriter(decision_trace_path)
    originals = BridgeOriginals(
        tool_gate=firewall_module.ToolGate,
        runtime_builder=firewall_module._build_default_theory_runtime,
        boundary_adapter=firewall_module.from_tool_boundary,
        realization_validator_descriptor=firewall_module.AgentDojoToolFirewall.__dict__["_validate_builtin_host_realization"],
    )

    def runtime_builder(capabilities: Sequence[str]) -> ResearchRuntimeAdapter:
        return ResearchRuntimeAdapter(
            selected,
            tuple(sorted(set(capabilities))),
            case_id=str(case_id),
            event_sink=writer,
            scalar_thresholds=scalar_thresholds,
        )

    def boundary_adapter(*args: Any, **kwargs: Any):
        canonical = originals.boundary_adapter(*args, **kwargs)
        context = kwargs.get("context")
        spec = kwargs.get("spec")
        if context is None or spec is None:
            raise TypeError("research boundary adapter requires keyword context and spec")
        records = _bound_records_from_canonical(
            canonical,
            context=context,
            spec=spec,
            case_id=str(case_id),
        )
        facts = dict(canonical.evidence.facts or {})
        facts[BOUND_RECORD_FACT_KEY] = [item.to_dict() for item in records]
        return replace(
            canonical,
            evidence=replace(canonical.evidence, facts=facts),
        )

    firewall_module.ToolGate = ResearchBindingGate
    firewall_module._build_default_theory_runtime = runtime_builder
    firewall_module.from_tool_boundary = boundary_adapter

    # AgentDojo is a virtual simulator. Constrained counterfactual decisions may
    # execute there for utility measurement, but no backend-realizability claim
    # follows from this case-scoped research override.
    firewall_module.AgentDojoToolFirewall._validate_builtin_host_realization = staticmethod(lambda realization: None)
    return originals


def uninstall_research_bridge(firewall_module: Any, originals: BridgeOriginals) -> None:
    firewall_module.ToolGate = originals.tool_gate
    firewall_module._build_default_theory_runtime = originals.runtime_builder
    firewall_module.from_tool_boundary = originals.boundary_adapter
    firewall_module.AgentDojoToolFirewall._validate_builtin_host_realization = originals.realization_validator_descriptor


def _bound_records_from_canonical(
    canonical: Any,
    *,
    context: Any,
    spec: Any,
    case_id: str,
) -> tuple[BoundEvidenceRecord, ...]:
    del context
    action = canonical.action
    evidence = canonical.evidence
    facts = evidence.facts or {}
    sequence = facts.get("graph.sequence_depth", 0)
    source_step = f"{case_id}:step:{sequence}"
    family = str(getattr(spec, "group", "") or action.tool_name)
    return bound_records_for_action(
        action,
        evidence,
        tool_family=family,
        source_step=source_step,
    )


def _result_digest_material(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _result_digest_material(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256_json(list(value)), "length": len(value)}
    if hasattr(value, "model_dump"):
        return _result_digest_material(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _result_digest_material(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_result_digest_material(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_result_digest_material(item) for item in value), key=canonical_json)
    if hasattr(value, "__dict__"):
        return {
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": _result_digest_material(vars(value)),
        }
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "string_value": str(value),
    }


__all__ = [
    "BridgeOriginals",
    "DecisionTraceWriter",
    "ResearchBindingGate",
    "ResearchDecisionAdapter",
    "ResearchDispatchBinding",
    "ResearchRuntimeAdapter",
    "install_research_bridge",
    "uninstall_research_bridge",
]
