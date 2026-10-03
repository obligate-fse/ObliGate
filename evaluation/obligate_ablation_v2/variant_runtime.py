"""Isolated, research-only ObliGate component-ablation runtime.

The production theory core is imported as a library but is never patched here.
This module deliberately does not run the finite checker, does not receive an
``Accept(kappa)`` record, and never issues an :class:`ActionCertificate`.

End-to-end variants are counterfactual policies for the AgentDojo simulator.
The two conflict variants are intended for frozen-snapshot replay, although the
same deterministic ``decide`` API is useful for synthetic preflight probes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence

from obligate.theory.lattice import (
    EnforceableControlPlanner,
    ControlPlanSelection,
    realization_satisfies,
    satisfies_all,
)
from obligate.theory.model import (
    ActionCandidate,
    BehaviorPlan,
    EvidenceAssertion,
    Realization,
    SignedAtom,
    Trigger,
    canonical_json,
    is_authorization_predicate,
    sha256_json,
)
from obligate.theory.eoc import HazardSignal, EOCCompiler, EOCResult
from obligate.theory.policy import PolicySpec, build_default_policy
from obligate.theory.runtime import (
    RuntimeEvidence,
    build_runtime_action_graph,
    requirements_for_action,
)
from obligate.theory.witness import BipolarWitnessEngine, WitnessClosure

from .scalar_scoring import (
    aggregate_score,
    plan_score,
    scalar_decision,
    trigger_scores,
)

RESEARCH_HARNESS_VERSION = "obligate-ablation-v2-research-runtime/2"
BOUND_RECORD_FACT_KEY = "obligate_ablation_v2.bound_evidence_records"
PARAMETER_SOURCE_PREDICATE = "parameter_source_trusted"
AUTHORIZATION_EVIDENCE_PREDICATES = (
    "action_authorized",
    "target_authorized",
    "flow_authorized",
    "privilege_authorized",
)
TRUSTED_PARAMETER_SOURCE_TYPES = frozenset(
    {
        "user_task",
        "trusted_tool_result",
        "trusted_account_record",
        "trusted_bill_metadata",
    }
)
DESTINATION_OBJECT_PARAMETER_KEYS = frozenset(
    {
        "account",
        "account_id",
        "bill",
        "bill_id",
        "calendar",
        "calendar_id",
        "channel",
        "channel_id",
        "contact",
        "contact_id",
        "destination",
        "destination_id",
        "email",
        "email_address",
        "event",
        "event_id",
        "file",
        "file_id",
        "file_path",
        "folder",
        "folder_id",
        "group",
        "group_id",
        "hotel",
        "iban",
        "member",
        "member_id",
        "path",
        "recipient",
        "restaurant",
        "target",
        "target_id",
        "to",
        "transaction",
        "transaction_id",
        "user",
        "user_id",
    }
)


class ResearchVariant(str, Enum):
    UNBOUND_EVIDENCE = "unbound-evidence"
    GAP_BLIND = "gap-blind"
    SCALAR_AVERAGE = "scalar-average"
    SINGLE_REMEDY = "single-remedy"
    BLOCK_ALL_GUARDED = "block-all-guarded"
    CONFLICT_TO_SUPPORT = "conflict-to-support"
    CONFLICT_TO_REFUTE = "conflict-to-refute"


E2E_VARIANTS = frozenset(
    {
        ResearchVariant.UNBOUND_EVIDENCE,
        ResearchVariant.GAP_BLIND,
        ResearchVariant.SCALAR_AVERAGE,
        ResearchVariant.SINGLE_REMEDY,
        ResearchVariant.BLOCK_ALL_GUARDED,
    }
)
SNAPSHOT_VARIANTS = frozenset({ResearchVariant.CONFLICT_TO_SUPPORT, ResearchVariant.CONFLICT_TO_REFUTE})


@dataclass(frozen=True, slots=True)
class BoundEvidenceRecord:
    """An explicit action/value/object/step-bound authorization observation.

    ``supports=False`` represents the current lookup relation without claiming
    that it is authorized.  Only trusted support records are committed to the
    historical evidence store and can be incorrectly reused by the ablation.
    """

    action_digest: str
    tool_family: str
    parameter_key: str
    value: str
    object_id: str
    source_step: str
    predicate: str
    evidence_ref: str = ""
    supports: bool = True
    trusted_for_authorization: bool = True
    evidence_kind: str = "authorization_assertion"
    source_type: str = "unknown"
    source_evidence_refs: tuple[str, ...] = ()
    object_role: str = "parameter"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_evidence_refs", tuple(str(item) for item in self.source_evidence_refs))
        required = (
            self.action_digest,
            self.tool_family,
            self.parameter_key,
            self.object_id,
            self.source_step,
            self.predicate,
        )
        if any(not str(item) for item in required):
            raise ValueError("bound evidence identity fields must be non-empty")

    @property
    def projection(self) -> tuple[str, str, str]:
        return (self.tool_family, self.parameter_key, self.predicate)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "action_digest": self.action_digest,
                "tool_family": self.tool_family,
                "parameter_key": self.parameter_key,
                "value": self.value,
                "object_id": self.object_id,
                "source_step": self.source_step,
                "predicate": self.predicate,
                "evidence_ref": self.evidence_ref,
                "supports": self.supports,
                "trusted_for_authorization": self.trusted_for_authorization,
                "evidence_kind": self.evidence_kind,
                "source_type": self.source_type,
                "source_evidence_refs": list(self.source_evidence_refs),
                "object_role": self.object_role,
            }
        )

    def mismatches(self, other: "BoundEvidenceRecord") -> bool:
        return (
            self.action_digest != other.action_digest
            or self.value != other.value
            or self.object_id != other.object_id
            or self.source_step != other.source_step
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_digest": self.action_digest,
            "tool_family": self.tool_family,
            "parameter_key": self.parameter_key,
            "value": self.value,
            "object_id": self.object_id,
            "source_step": self.source_step,
            "predicate": self.predicate,
            "evidence_ref": self.evidence_ref or f"bound:{self.digest[:20]}",
            "supports": self.supports,
            "trusted_for_authorization": self.trusted_for_authorization,
            "evidence_kind": self.evidence_kind,
            "source_type": self.source_type,
            "source_evidence_refs": list(self.source_evidence_refs),
            "object_role": self.object_role,
        }

    def claim_for_action(self, tool_name: str) -> SignedAtom:
        """Return the current claim reconstructed after the unbound lookup.

        Authorization assertions retain their production action-level shape.
        Trusted-parameter provenance is represented separately at parameter
        granularity so the research trace does not mislabel provenance as an
        action authorization assertion.
        """

        arguments = (
            (tool_name, self.parameter_key)
            if self.predicate == PARAMETER_SOURCE_PREDICATE
            else (tool_name,)
        )
        return SignedAtom(self.predicate, arguments, "+")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundEvidenceRecord":
        return cls(
            action_digest=str(value["action_digest"]),
            tool_family=str(value["tool_family"]),
            parameter_key=str(value["parameter_key"]),
            value=str(value.get("value", "")),
            object_id=str(value["object_id"]),
            source_step=str(value["source_step"]),
            predicate=str(value["predicate"]),
            evidence_ref=str(value.get("evidence_ref", "")),
            supports=bool(value.get("supports", True)),
            trusted_for_authorization=bool(value.get("trusted_for_authorization", True)),
            evidence_kind=str(value.get("evidence_kind", "authorization_assertion")),
            source_type=str(value.get("source_type", "unknown")),
            source_evidence_refs=tuple(str(item) for item in value.get("source_evidence_refs", ())),
            object_role=str(value.get("object_role", "parameter")),
        )


class BoundEvidenceHistory:
    """Case-local support history used only by ``unbound-evidence``."""

    def __init__(self) -> None:
        self._records: list[BoundEvidenceRecord] = []
        self._digests: set[str] = set()
        self._lock = RLock()

    def snapshot(self) -> tuple[BoundEvidenceRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def commit(self, records: Iterable[BoundEvidenceRecord]) -> None:
        with self._lock:
            for record in records:
                if not record.supports or not record.trusted_for_authorization:
                    continue
                if record.digest in self._digests:
                    continue
                self._digests.add(record.digest)
                self._records.append(record)

    def projected_matches(
        self,
        current: BoundEvidenceRecord,
    ) -> tuple[BoundEvidenceRecord, ...]:
        with self._lock:
            values = [
                item
                for item in self._records
                if item.projection == current.projection
                and item.supports
                and item.trusted_for_authorization
                and item.mismatches(current)
            ]
        return tuple(
            sorted(
                values,
                key=lambda item: (item.evidence_kind != current.evidence_kind, item.digest),
            )
        )


@dataclass(frozen=True, slots=True)
class ScalarThresholds:
    t_allow: float
    t_confirm: float
    t_block: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.t_allow < self.t_confirm < self.t_block <= 1.0):
            raise ValueError("scalar thresholds must satisfy 0 <= t_allow < t_confirm < t_block <= 1")

    @classmethod
    def from_value(
        cls,
        value: "ScalarThresholds | Mapping[str, Any] | Sequence[float] | None",
    ) -> "ScalarThresholds | None":
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            if len(value) != 3:
                raise ValueError("scalar threshold sequence must contain three values")
            return cls(*(float(item) for item in value))
        return cls(
            float(value.get("t_allow", value.get("allow"))),
            float(value.get("t_confirm", value.get("confirm"))),
            float(value.get("t_block", value.get("block"))),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "t_allow": self.t_allow,
            "t_confirm": self.t_confirm,
            "t_block": self.t_block,
        }


@dataclass(frozen=True, slots=True)
class ResearchDecision:
    variant: ResearchVariant
    public_decision: str
    execute: bool
    triggers: tuple[Trigger, ...]
    semantic_obligations: tuple[Any, ...]
    implementation_constraints: tuple[Any, ...]
    ideal: tuple[BehaviorPlan, ...]
    safe_candidates: tuple[BehaviorPlan, ...]
    feasible_realizations: tuple[Realization, ...]
    frontier: tuple[Realization, ...]
    selected: Realization | None
    trace: Mapping[str, Any]
    research_only: bool = True
    verified: bool = False
    research_counterfactual_mode: bool = True
    acceptance_record: None = None
    certificate: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_version": RESEARCH_HARNESS_VERSION,
            "variant": self.variant.value,
            "research_only": True,
            "verified": False,
            "research_counterfactual_mode": True,
            "acceptance_record": None,
            "certificate": None,
            "main_theorem_guarantee": False,
            "public_decision": self.public_decision,
            "execute": self.execute,
            "triggers": [item.to_dict() for item in self.triggers],
            "semantic_obligations": [item.to_dict() for item in self.semantic_obligations],
            "implementation_constraints": [item.to_dict() for item in self.implementation_constraints],
            "ideal": [item.to_dict() for item in self.ideal],
            "safe_candidates": [item.to_dict() for item in self.safe_candidates],
            "feasible_realizations": [item.to_dict() for item in self.feasible_realizations],
            "frontier": [item.to_dict() for item in self.frontier],
            "selected": self.selected.to_dict() if self.selected else None,
            "trace": dict(self.trace),
        }


EventSink = Callable[[Mapping[str, Any]], None]


class ResearchRuntime:
    """Unverified counterfactual runtime with a production-compatible shape."""

    def __init__(
        self,
        variant: ResearchVariant | str,
        *,
        research_counterfactual_mode: bool = False,
        policy: PolicySpec | None = None,
        witness_limit: int | None = None,
        max_candidates: int | None = None,
        capabilities: Iterable[str] = (
            "host",
            "sandbox",
            "network",
            "redaction",
            "confirmation",
            "audit",
        ),
        bound_history: BoundEvidenceHistory | None = None,
        scalar_thresholds: ScalarThresholds | Mapping[str, Any] | Sequence[float] | None = None,
        scalar_scores: Mapping[str, float] | None = None,
        case_id: str = "",
        event_sink: EventSink | None = None,
    ) -> None:
        if not research_counterfactual_mode:
            raise RuntimeError("research counterfactual runtime is disabled unless explicitly enabled")
        self.variant = ResearchVariant(variant)
        self.policy = policy or build_default_policy()
        self.witness_limit = witness_limit
        self.max_candidates = max_candidates
        self.capabilities = frozenset(capabilities)
        self.bound_history = bound_history or BoundEvidenceHistory()
        self.scalar_thresholds = ScalarThresholds.from_value(scalar_thresholds)
        # Kept only as a compatibility/audit field.  Online scores themselves
        # always use scalar_scoring.py, the same frozen implementation used by
        # cross-fit calibration; callers cannot substitute outcome-tuned scores.
        self.scalar_scores = dict(scalar_scores or {})
        if self.variant is ResearchVariant.SCALAR_AVERAGE:
            if self.scalar_thresholds is None:
                raise ValueError("scalar-average requires preregistered scalar thresholds")
        self.case_id = str(case_id)
        self.event_sink = event_sink
        self.compiler = EOCCompiler(policy=self.policy)
        self.control_planner = EnforceableControlPlanner()
        self._decision_cache: dict[tuple[str, str], ResearchDecision] = {}
        self._lock = RLock()

    @property
    def variant_definition_digest(self) -> str:
        return sha256_json(
            {
                "harness_version": RESEARCH_HARNESS_VERSION,
                "variant": self.variant.value,
                "policy_digest": self.policy.digest,
                "witness_limit": self.witness_limit,
                "max_candidates": self.max_candidates,
                "capabilities": sorted(self.capabilities),
                "scalar_thresholds": (self.scalar_thresholds.to_dict() if self.scalar_thresholds else None),
                "scalar_scores": dict(sorted(self.scalar_scores.items())),
            }
        )

    def decide(self, action: ActionCandidate, evidence: RuntimeEvidence) -> ResearchDecision:
        serialized_evidence = serialize_runtime_evidence(evidence)
        evidence_digest = sha256_json(serialized_evidence)
        base_evidence_digest = predecision_base_evidence_digest(serialized_evidence)
        cache_key = (action.digest, evidence_digest)
        step = (evidence.facts or {}).get("graph.sequence_depth", 0)
        evidence_summary = {
            "assertion_count": len(evidence.assertions),
            "hazard_count": len(evidence.hazards),
            "explicit_deny_count": len(evidence.explicit_denies),
            "graph_node_count": len(evidence.graph_nodes),
            "graph_edge_count": len(evidence.graph_edges),
        }
        with self._lock:
            cached = self._decision_cache.get(cache_key)
            if cached is not None:
                self._emit(
                    {
                        "event": "fresh_re_adjudication",
                        "event_type": "fresh_re_adjudication",
                        "case_id": self.case_id,
                        "sequence": step,
                        "step": step,
                        "variant": self.variant.value,
                        "action_digest": action.digest,
                        "action": action.normalized_dict(),
                        "evidence_digest": evidence_digest,
                        "predecision_base_evidence_digest": base_evidence_digest,
                        "evidence_summary": evidence_summary,
                        "evidence_input_digest": evidence_digest,
                        "cache_reused": True,
                        "cache_reused_for_fresh_rebind": True,
                        "public_decision": cached.public_decision,
                        "decision_digest": sha256_json(cached.to_dict()),
                        **_research_markers(),
                    }
                )
                return cached

            current_records = extract_bound_records(evidence, action_digest=action.digest)
            self._emit(
                {
                    "event": "input",
                    "event_type": "input",
                    "case_id": self.case_id,
                    "sequence": step,
                    "step": step,
                    "variant": self.variant.value,
                    "action_digest": action.digest,
                    "action": action.normalized_dict(),
                    "evidence_digest": evidence_digest,
                    "predecision_base_evidence_digest": base_evidence_digest,
                    "evidence_summary": evidence_summary,
                    "evidence_input_digest": evidence_digest,
                    "bound_evidence_records": [item.to_dict() for item in current_records],
                    "cache_reused": False,
                    "cache_reused_for_fresh_rebind": False,
                    **_research_markers(),
                }
            )
            decision = self._decide_uncached(action, evidence, current_records)
            self._decision_cache[cache_key] = decision
            # Commit only after the first decision.  This makes the immediately
            # repeated pre-send adjudication a cache hit and prevents the current
            # action from becoming its own historical authorization witness.
            self.bound_history.commit(current_records)
            self._emit(
                {
                    "event": "decision",
                    "event_type": "decision",
                    "case_id": self.case_id,
                    "sequence": step,
                    "step": step,
                    "variant": self.variant.value,
                    "action_digest": action.digest,
                    "action": action.normalized_dict(),
                    "evidence_digest": evidence_digest,
                    "predecision_base_evidence_digest": base_evidence_digest,
                    "evidence_summary": evidence_summary,
                    "evidence_input_digest": evidence_digest,
                    "decision_digest": sha256_json(decision.to_dict()),
                    "public_decision": decision.public_decision,
                    "triggers": [item.to_dict() for item in decision.triggers],
                    "selected": (decision.selected.to_dict() if decision.selected else None),
                    "plan": (decision.selected.plan.to_dict() if decision.selected else None),
                    "trace": dict(decision.trace),
                    "decision": decision.to_dict(),
                    "cache_reused": False,
                    "cache_reused_for_fresh_rebind": False,
                    **_research_markers(),
                }
            )
            return decision

    def _decide_uncached(
        self,
        action: ActionCandidate,
        evidence: RuntimeEvidence,
        current_records: tuple[BoundEvidenceRecord, ...],
    ) -> ResearchDecision:
        graph = build_runtime_action_graph(action, evidence)
        requirements = requirements_for_action(action)
        assertions = tuple(
            item
            for item in evidence.assertions
            if not (item.atom.polarity == "+" and is_authorization_predicate(item.atom.predicate) and not item.trusted_for_authorization)
        )
        rejected = tuple(item for item in evidence.assertions if item not in assertions)
        misuse: list[dict[str, Any]] = []
        if self.variant is ResearchVariant.UNBOUND_EVIDENCE:
            assertions, misuse = self._unbound_assertions(action, assertions, current_records)

        explicit_denies = list(evidence.explicit_denies)
        for requirement in requirements:
            if requirement.unsigned_key in evidence.contract.deny:
                explicit_denies.append(
                    HazardSignal(
                        signal_id=f"task_contract_deny:{requirement.unsigned_key}",
                        subject=f"TaskContract explicitly denies {requirement.unsigned_key}",
                        evidence_refs=(evidence.contract.source_digest or "task_contract:deny",),
                        hard=True,
                        kind="hazard",
                    )
                )

        original_closure = BipolarWitnessEngine(witness_limit=self.witness_limit).close(assertions, self.policy.rules)
        effective_closure = original_closure
        if self.variant is ResearchVariant.CONFLICT_TO_SUPPORT:
            effective_closure = collapse_conflicts(original_closure, keep="support")
        elif self.variant is ResearchVariant.CONFLICT_TO_REFUTE:
            effective_closure = collapse_conflicts(original_closure, keep="refute")

        raw_eoc: EOCResult | None = None
        if self.variant is ResearchVariant.GAP_BLIND:
            raw_eoc = self.compiler.compile(
                requirements=requirements,
                closure=effective_closure,
                hazards=evidence.hazards,
                explicit_denies=explicit_denies,
            )
            masked = self.compiler.compile(
                requirements=(),
                closure=effective_closure,
                hazards=evidence.hazards,
                explicit_denies=explicit_denies,
            )
            eoc = replace(masked, requirements=requirements)
        else:
            eoc = self.compiler.compile(
                requirements=requirements,
                closure=effective_closure,
                hazards=evidence.hazards,
                explicit_denies=explicit_denies,
            )

        realizations = tuple(self._materialize(item) for item in self.policy.realizations)
        if self.variant is ResearchVariant.SCALAR_AVERAGE:
            selection, variant_details = self._select_scalar(eoc, realizations)
        elif self.variant is ResearchVariant.SINGLE_REMEDY:
            selection = self._select_single_remedy(eoc, realizations, evidence)
            variant_details = {"single_remedy_checks_all_active_obligations": True}
        elif self.variant is ResearchVariant.BLOCK_ALL_GUARDED and action.guarded:
            selection = self._forced_block(eoc, realizations)
            variant_details = {"all_guarded_actions_forced_to_block": True}
        else:
            selection = self._select_standard(eoc, realizations, evidence)
            variant_details = {}

        fact_digest = sha256_json(
            {
                "research_variant": self.variant.value,
                "policy_definition_digest": self.policy.digest,
                "contract": evidence.contract.to_dict(),
                "runtime_facts": dict(evidence.facts or {}),
                "witness": effective_closure.to_dict(),
                "eoc": eoc.to_dict(),
            }
        )
        trace = {
            **_research_markers(),
            "harness_version": RESEARCH_HARNESS_VERSION,
            "variant_definition_digest": self.variant_definition_digest,
            "policy_definition_digest": self.policy.digest,
            "policy_acceptance_digest": None,
            "loader_replay_digest": None,
            "case_id": self.case_id,
            "action_digest": action.digest,
            "fact_digest": fact_digest,
            "evidence_digest": graph.digest,
            "action_graph": graph.to_dict(),
            "witness_closure_original": original_closure.to_dict(),
            "witness_closure_effective": effective_closure.to_dict(),
            "rejected_enabling_assertions": [item.to_dict() for item in rejected],
            "gap_before_mask": list(raw_eoc.gaps) if raw_eoc is not None else list(eoc.gaps),
            "gap": list(eoc.gaps),
            "hazard": list(eoc.hazards),
            "overflow": list(eoc.overflow),
            "triggers": [item.to_dict() for item in eoc.triggers],
            "valid_certificate_evidence": False,
            "certificate": None,
            "unbound_evidence_misuse": misuse,
            "eoc": eoc.to_dict(),
            "lattice": selection.to_dict(),
            **variant_details,
        }
        return ResearchDecision(
            variant=self.variant,
            public_decision=selection.outcome,
            execute=selection.execute,
            triggers=eoc.triggers,
            semantic_obligations=eoc.semantic_obligations,
            implementation_constraints=eoc.implementation_constraints,
            ideal=selection.ideal,
            safe_candidates=selection.safe_candidates,
            feasible_realizations=selection.feasible_realizations,
            frontier=selection.frontier,
            selected=selection.selected,
            trace=trace,
        )

    def _unbound_assertions(
        self,
        action: ActionCandidate,
        assertions: tuple[EvidenceAssertion, ...],
        current_records: tuple[BoundEvidenceRecord, ...],
    ) -> tuple[tuple[EvidenceAssertion, ...], list[dict[str, Any]]]:
        values = list(assertions)
        existing_support = {
            item.atom.unsigned_key
            for item in assertions
            if item.atom.polarity == "+" and item.trusted_for_authorization
        }
        injected_atoms: set[SignedAtom] = set()
        misuse: list[dict[str, Any]] = []
        ordered_records = sorted(
            current_records,
            key=lambda item: (
                item.evidence_kind != "parameter_provenance_requirement",
                item.digest,
            ),
        )
        for current in ordered_records:
            if current.action_digest != action.digest:
                continue
            if current.supports and current.trusted_for_authorization:
                continue
            atom = current.claim_for_action(action.tool_name)
            if atom.unsigned_key in existing_support:
                continue
            matches = self.bound_history.projected_matches(current)
            if not matches:
                continue
            historical = matches[0]
            if atom in injected_atoms:
                continue
            reference = f"unbound-reuse:{historical.digest[:20]}"
            values.append(
                EvidenceAssertion(
                    atom=atom,
                    evidence_ref=reference,
                    source_kind="research_unbound_historical_projection",
                    trusted_for_authorization=True,
                )
            )
            injected_atoms.add(atom)
            misuse.append(
                {
                    "projection": list(current.projection),
                    "current": current.to_dict(),
                    "historical": historical.to_dict(),
                    "evidence_kind": historical.evidence_kind,
                    "current_evidence_kind": current.evidence_kind,
                    "historical_evidence_kind": historical.evidence_kind,
                    "mismatch_fields": [
                        name
                        for name in ("action_digest", "value", "object_id", "source_step")
                        if getattr(current, name) != getattr(historical, name)
                    ],
                    "injected_evidence_ref": reference,
                }
            )
        return tuple(values), misuse

    def _select_standard(
        self,
        eoc: EOCResult,
        realizations: tuple[Realization, ...],
        evidence: RuntimeEvidence,
    ) -> ControlPlanSelection:
        trigger_remedies = [tuple(remedy.plan for remedy in eoc.remedies[trigger.trigger_id]) for trigger in eoc.triggers]
        return self.control_planner.select(
            base_remedies=(self.policy.default_plan,),
            trigger_remedies=trigger_remedies,
            obligations=eoc.semantic_obligations,
            implementation_constraints=eoc.implementation_constraints,
            plan_library=self.policy.behavior_domain,
            realizations=realizations,
            default_plan=self.policy.default_plan,
            max_candidates=self.max_candidates,
            approval_satisfied=evidence.confirmation is not None,
        )

    def _select_single_remedy(
        self,
        eoc: EOCResult,
        realizations: tuple[Realization, ...],
        evidence: RuntimeEvidence,
    ) -> ControlPlanSelection:
        local_plans = [self.policy.default_plan]
        for trigger in eoc.triggers:
            local_plans.extend(remedy.plan for remedy in eoc.remedies[trigger.trigger_id])
        library = tuple({plan.digest: plan for plan in local_plans}.values())
        # No lifted join is supplied.  The unchanged selector still checks every
        # candidate against the conjunction of all active obligations.
        return self.control_planner.select(
            base_remedies=(self.policy.default_plan,),
            trigger_remedies=(),
            obligations=eoc.semantic_obligations,
            implementation_constraints=eoc.implementation_constraints,
            plan_library=library,
            realizations=realizations,
            default_plan=self.policy.default_plan,
            max_candidates=self.max_candidates,
            approval_satisfied=evidence.confirmation is not None,
        )

    def _select_scalar(
        self,
        eoc: EOCResult,
        realizations: tuple[Realization, ...],
    ) -> tuple[ControlPlanSelection, dict[str, Any]]:
        assert self.scalar_thresholds is not None
        hard = tuple(item for item in eoc.triggers if item.hard)
        eoc_payload = eoc.to_dict()
        per_trigger = trigger_scores(eoc_payload)
        aggregate = aggregate_score(eoc_payload)
        mapped = scalar_decision(
            aggregate,
            (
                self.scalar_thresholds.t_allow,
                self.scalar_thresholds.t_confirm,
                self.scalar_thresholds.t_block,
            ),
            hard=bool(hard),
        )

        if mapped == "block":
            selection = self._forced_block(eoc, realizations)
            branch = "hard_or_explicit_deny_block" if hard else "scalar_block"
        else:
            if mapped == "allow":
                selected = self._least_cost_realization(realizations, plans=(self.policy.default_plan,), constraints=())
                selection = self._scalar_selection(
                    "allow" if selected is not None else "block_with_compliance_error",
                    selected,
                    ideal=(self.policy.default_plan,),
                )
                branch = "allow"
            elif mapped == "execute_with_constraints":
                highest = max(
                    eoc.triggers,
                    key=lambda item: (per_trigger[item.trigger_id], item.trigger_id),
                )
                local_plans = tuple(remedy.plan for remedy in eoc.remedies[highest.trigger_id])
                local_constraints = highest.implementation_constraints
                selected = self._least_cost_realization(
                    realizations,
                    plans=local_plans,
                    constraints=local_constraints,
                    require_dispatch=True,
                    reject_confirmation=True,
                )
                if selected is None:
                    selection = self._forced_block(eoc, realizations)
                    branch = "highest_trigger_local_realization_unavailable"
                else:
                    selection = self._scalar_selection(
                        "execute_with_constraints",
                        selected,
                        ideal=local_plans,
                    )
                    branch = "execute_with_constraints_highest_trigger_only"
            elif mapped == "require_confirmation":
                confirmation_plans = tuple(item for item in self.policy.behavior_domain if item.confirmation)
                selected = self._least_cost_realization(realizations, plans=confirmation_plans, constraints=())
                if selected is None:
                    selection = self._forced_block(eoc, realizations)
                    branch = "confirmation_realization_unavailable"
                else:
                    selection = self._scalar_selection("require_confirmation", selected, ideal=confirmation_plans)
                    branch = "require_confirmation"
            else:
                raise AssertionError(f"unexpected scalar decision {mapped!r}")

        return selection, {
            "scalar_average": {
                "thresholds": self.scalar_thresholds.to_dict(),
                "trigger_scores": per_trigger,
                "aggregate": aggregate,
                "mapped_decision": mapped,
                "branch": branch,
                "hard_trigger_ids": [item.trigger_id for item in hard],
                "hard_and_explicit_deny_non_compensatory": True,
                "selected_plan_score": (plan_score(selection.selected.plan.to_dict()) if selection.selected is not None else None),
            }
        }

    def _least_cost_realization(
        self,
        realizations: Sequence[Realization],
        *,
        plans: Sequence[BehaviorPlan],
        constraints: Sequence[Any],
        require_dispatch: bool = False,
        reject_confirmation: bool = False,
    ) -> Realization | None:
        plan_digests = {item.digest for item in plans}
        candidates = []
        for realization in realizations:
            if realization.plan.digest not in plan_digests:
                continue
            if require_dispatch and realization.plan.block_like:
                continue
            if reject_confirmation and realization.plan.confirmation:
                continue
            if not realization_satisfies(realization, realization.plan, constraints):
                continue
            candidates.append(realization)
        return min(candidates, key=lambda item: (*item.utility_vector, item.realization_id)) if candidates else None

    @staticmethod
    def _scalar_selection(
        outcome: str,
        selected: Realization | None,
        *,
        ideal: Sequence[BehaviorPlan],
    ) -> ControlPlanSelection:
        values = (selected,) if selected is not None else ()
        plans = (selected.plan,) if selected is not None else ()
        non_blocking = values if selected is not None and not selected.plan.block_like else ()
        return ControlPlanSelection(
            outcome=outcome,  # type: ignore[arg-type]
            ideal=tuple(ideal),
            safe_candidates=plans,
            feasible_realizations=values,
            non_blocking=non_blocking,
            frontier=non_blocking,
            selected=selected,
            compliant_block=selected if selected is not None and selected.plan.block_like else None,
        )

    def _forced_block(
        self,
        eoc: EOCResult,
        realizations: Sequence[Realization],
    ) -> ControlPlanSelection:
        blockers = [
            item
            for item in realizations
            if item.plan.block_like
            and satisfies_all(item.plan, eoc.semantic_obligations)
            and realization_satisfies(item, item.plan, eoc.implementation_constraints)
        ]
        selected = min(blockers, key=lambda item: (*item.utility_vector, item.realization_id)) if blockers else None
        return ControlPlanSelection(
            outcome="block" if selected is not None else "block_with_compliance_error",
            ideal=(selected.plan,) if selected is not None else (),
            safe_candidates=(selected.plan,) if selected is not None else (),
            feasible_realizations=(selected,) if selected is not None else (),
            non_blocking=(),
            frontier=(),
            selected=selected,
            compliant_block=selected,
        )

    def _materialize(self, template: Realization) -> Realization:
        required = set(template.capabilities)
        return replace(
            template,
            realization_id=f"runtime:{template.realization_id}",
            capabilities=frozenset(required & self.capabilities),
            actual_audit=(template.actual_audit if "audit" in self.capabilities else frozenset()),
            available=template.available and required <= self.capabilities,
        )

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(dict(event))


def collapse_conflicts(closure: WitnessClosure, *, keep: str) -> WitnessClosure:
    """Collapse only true bipolar conflicts to one selected polarity."""

    if keep not in {"support", "refute"}:
        raise ValueError("keep must be 'support' or 'refute'")
    witnesses = dict(closure.witnesses)
    unsigned = {key[1:] for key in witnesses}
    for key in unsigned:
        positive = tuple(witnesses.get(f"+{key}", ()))
        negative = tuple(witnesses.get(f"-{key}", ()))
        if not positive or not negative:
            continue
        if keep == "support":
            witnesses[f"-{key}"] = ()
        else:
            witnesses[f"+{key}"] = ()
    return WitnessClosure(
        witnesses=witnesses,
        overflow=closure.overflow,
        overflow_paths=closure.overflow_paths,
        rule_firings=closure.rule_firings,
        exact=closure.exact,
        witness_limit=closure.witness_limit,
    )


def extract_bound_records(evidence: RuntimeEvidence, *, action_digest: str) -> tuple[BoundEvidenceRecord, ...]:
    facts = evidence.facts or {}
    raw = facts.get(BOUND_RECORD_FACT_KEY, ())
    if not isinstance(raw, (list, tuple)):
        return ()
    records: list[BoundEvidenceRecord] = []
    for item in raw:
        if isinstance(item, BoundEvidenceRecord):
            record = item
        elif isinstance(item, Mapping):
            record = BoundEvidenceRecord.from_dict(item)
        else:
            raise TypeError("bound evidence records must be mappings or BoundEvidenceRecord")
        if record.action_digest != action_digest:
            raise ValueError("current bound evidence record is bound to another action")
        records.append(record)
    return tuple(sorted(records, key=lambda item: item.digest))


def bound_records_for_action(
    action: ActionCandidate,
    evidence: RuntimeEvidence,
    *,
    tool_family: str,
    source_step: str,
) -> tuple[BoundEvidenceRecord, ...]:
    """Build exact action/value/object/step records from production evidence.

    Two evidence channels stay distinguishable:

    * trusted authorization assertions for the current action requirement;
    * trusted parameter provenance from ``agentdojo.arg_provenance``.

    Parameter provenance keeps a generic observation claim for relation-set
    accounting and a separate record for each authorization evidence
    predicate.  The latter is what the deliberately unsound unbound lookup can
    reuse for a different value/object/step; production Full never consumes
    these research records.
    """

    positive = {
        assertion.atom.unsigned_key: assertion.evidence_ref
        for assertion in evidence.assertions
        if assertion.atom.polarity == "+" and assertion.trusted_for_authorization
    }
    requirements = tuple(
        item
        for item in requirements_for_action(action)
        if is_authorization_predicate(item.predicate)
    )
    facts = evidence.facts or {}
    provenance_by_key: dict[str, list[Mapping[str, Any]]] = {}
    for item in facts.get("agentdojo.arg_provenance") or ():
        if not isinstance(item, Mapping):
            continue
        provenance_by_key.setdefault(str(item.get("arg_name") or ""), []).append(item)

    arguments = list(sorted(action.arguments.items())) or [("__action__", action.tool_name)]
    explicit_target_keys: set[str] = set()
    for fact_name in (
        "agentdojo.target_entity_value_match",
        "agentdojo.recipient_value_match",
    ):
        target_match = facts.get(fact_name)
        if isinstance(target_match, Mapping):
            target_key = target_match.get("arg_name") or target_match.get("field_name")
            if target_key:
                explicit_target_keys.add(str(target_key).casefold())
    if facts.get("agentdojo.target_arg_entities") and len(arguments) == 1:
        explicit_target_keys.add(str(arguments[0][0]).casefold())
    records: list[BoundEvidenceRecord] = []
    for parameter_key, raw_value in arguments:
        key = str(parameter_key)
        value = canonical_json(raw_value)
        object_id = f"{key}:{sha256_json(raw_value)[:20]}"
        matching_provenance = [
            item
            for item in provenance_by_key.get(key, ())
            if canonical_json(item.get("value")) == value
        ]
        trusted_provenance = [
            item
            for item in matching_provenance
            if str(item.get("source_type") or "unknown") in TRUSTED_PARAMETER_SOURCE_TYPES
        ]
        source_types = sorted({str(item.get("source_type") or "unknown") for item in matching_provenance})
        source_refs = tuple(
            sorted(
                {
                    str(ref)
                    for item in matching_provenance
                    for ref in (item.get("evidence_refs") or ())
                    if ref
                }
            )
        )
        trusted_refs = tuple(
            sorted(
                {
                    str(ref)
                    for item in trusted_provenance
                    for ref in (item.get("evidence_refs") or ())
                    if ref
                }
            )
        )
        source_type = "+".join(source_types) if source_types else "unknown"
        object_role = (
            "destination_or_object"
            if key.casefold() in DESTINATION_OBJECT_PARAMETER_KEYS
            or key.casefold() in explicit_target_keys
            else "parameter"
        )

        for requirement in requirements:
            support = requirement.unsigned_key in positive
            records.append(
                BoundEvidenceRecord(
                    action_digest=action.digest,
                    tool_family=str(tool_family),
                    parameter_key=key,
                    value=value,
                    object_id=object_id,
                    source_step=str(source_step),
                    predicate=requirement.predicate,
                    evidence_ref=positive.get(requirement.unsigned_key, ""),
                    supports=support,
                    trusted_for_authorization=support,
                    evidence_kind="authorization_assertion",
                    source_type=source_type,
                    source_evidence_refs=source_refs,
                    object_role=object_role,
                )
            )

        trusted_source = bool(trusted_provenance)
        provenance_ref = (
            f"arg-provenance:{sha256_json({'key': key, 'source_types': source_types, 'refs': trusted_refs})[:20]}"
            if trusted_source
            else ""
        )
        records.append(
            BoundEvidenceRecord(
                action_digest=action.digest,
                tool_family=str(tool_family),
                parameter_key=key,
                value=value,
                object_id=object_id,
                source_step=str(source_step),
                predicate=PARAMETER_SOURCE_PREDICATE,
                evidence_ref=provenance_ref,
                supports=trusted_source,
                trusted_for_authorization=trusted_source,
                evidence_kind="parameter_provenance",
                source_type=source_type,
                source_evidence_refs=trusted_refs or source_refs,
                object_role=object_role,
            )
        )
        for predicate in AUTHORIZATION_EVIDENCE_PREDICATES:
            records.append(
                BoundEvidenceRecord(
                    action_digest=action.digest,
                    tool_family=str(tool_family),
                    parameter_key=key,
                    value=value,
                    object_id=object_id,
                    source_step=str(source_step),
                    predicate=predicate,
                    evidence_ref=provenance_ref,
                    supports=trusted_source,
                    trusted_for_authorization=trusted_source,
                    evidence_kind="parameter_provenance_requirement",
                    source_type=source_type,
                    source_evidence_refs=trusted_refs or source_refs,
                    object_role=object_role,
                )
            )
    return tuple(sorted(records, key=lambda item: item.digest))


def serialize_runtime_evidence(evidence: RuntimeEvidence) -> dict[str, Any]:
    return {
        "contract": evidence.contract.to_dict(),
        "assertions": [item.to_dict() for item in evidence.assertions],
        "hazards": [
            {
                "signal_id": item.signal_id,
                "subject": item.subject,
                "evidence_refs": list(item.evidence_refs),
                "hard": item.hard,
                "preferred_control": item.preferred_control,
                "kind": item.kind,
            }
            for item in evidence.hazards
        ],
        "explicit_denies": [
            {
                "signal_id": item.signal_id,
                "subject": item.subject,
                "evidence_refs": list(item.evidence_refs),
                "hard": item.hard,
                "preferred_control": item.preferred_control,
                "kind": item.kind,
            }
            for item in evidence.explicit_denies
        ],
        "graph_nodes": [item.to_dict() for item in evidence.graph_nodes],
        "graph_edges": [item.to_dict() for item in evidence.graph_edges],
        "facts": dict(evidence.facts or {}),
        "confirmation": evidence.confirmation.to_dict() if evidence.confirmation else None,
    }


def predecision_base_evidence_digest(serialized_evidence: Mapping[str, Any]) -> str:
    """Hash the shared policy input, excluding research-only derived records.

    The bridge injects ``BOUND_RECORD_FACT_KEY`` after the production boundary
    adapter has created RuntimeEvidence.  Those records are a deterministic
    research projection of the same action/evidence, not an additional input.
    Keeping them in the comparison would make every Full-vs-variant decision
    appear downstream even while the actual pre-policy input is identical.
    """

    projected = dict(serialized_evidence)
    facts = dict(projected.get("facts") or {})
    facts.pop(BOUND_RECORD_FACT_KEY, None)
    projected["facts"] = facts
    return sha256_json(projected)


def _research_markers() -> dict[str, Any]:
    return {
        "research_only": True,
        "verified": False,
        "research_counterfactual_mode": True,
        "acceptance_record": None,
        "action_certificate": None,
        "main_theorem_guarantee": False,
    }


__all__ = [
    "AUTHORIZATION_EVIDENCE_PREDICATES",
    "BOUND_RECORD_FACT_KEY",
    "DESTINATION_OBJECT_PARAMETER_KEYS",
    "E2E_VARIANTS",
    "PARAMETER_SOURCE_PREDICATE",
    "RESEARCH_HARNESS_VERSION",
    "SNAPSHOT_VARIANTS",
    "TRUSTED_PARAMETER_SOURCE_TYPES",
    "BoundEvidenceHistory",
    "BoundEvidenceRecord",
    "ResearchDecision",
    "ResearchRuntime",
    "ResearchVariant",
    "ScalarThresholds",
    "bound_records_for_action",
    "collapse_conflicts",
    "extract_bound_records",
    "predecision_base_evidence_digest",
    "serialize_runtime_evidence",
]
