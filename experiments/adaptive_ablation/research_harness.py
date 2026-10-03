"""Isolated research-only mechanism ablations.

This module is not imported by the production loader.  It deliberately does
not run the checker, does not receive ``Accept(kappa)``, and never issues an
ActionCertificate.  Its outputs are diagnostics without main-theorem claims.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from obligate.theory.lattice import EnforceableControlPlanner, ControlPlanSelection
from obligate.theory.model import (
    ActionCandidate,
    GraphEdge,
    GraphNode,
    Realization,
    SignedAtom,
    Trigger,
    canonical_json,
    is_authorization_predicate,
)
from obligate.theory.eoc import HazardSignal, EOCCompiler, EOCResult
from obligate.theory.policy import PolicySpec, build_default_policy
from obligate.theory.runtime import RuntimeEvidence, build_runtime_action_graph, requirements_for_action
from obligate.theory.witness import BipolarWitnessEngine, WitnessClosure

RESEARCH_HARNESS_VERSION = "obligate-research-only-ablation-v1"


class ResearchVariant(str, Enum):
    CONFLICT_COLLAPSE = "conflict-collapse"
    GAP_BLIND = "gap-blind"
    NO_LIFTED_JOIN = "no-lifted-join"


@dataclass(frozen=True, slots=True)
class ResearchDecision:
    variant: ResearchVariant
    public_decision: str
    execute: bool
    triggers: tuple[Trigger, ...]
    semantic_obligations: tuple[Any, ...]
    implementation_constraints: tuple[Any, ...]
    ideal: tuple[Any, ...]
    safe_candidates: tuple[Any, ...]
    feasible_realizations: tuple[Realization, ...]
    frontier: tuple[Realization, ...]
    selected: Realization | None
    trace: Mapping[str, Any]
    research_only: bool = True
    verified: bool = False
    acceptance_record: None = None
    certificate: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_version": RESEARCH_HARNESS_VERSION,
            "variant": self.variant.value,
            "research_only": True,
            "verified": False,
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


class ResearchRuntime:
    """Unverified counterfactual runtime, disabled unless explicitly constructed."""

    def __init__(
        self,
        variant: ResearchVariant | str,
        *,
        enabled: bool = False,
        policy: PolicySpec | None = None,
        witness_limit: int | None = None,
        max_candidates: int | None = None,
        capabilities: Iterable[str] = ("host", "sandbox", "redaction", "confirmation", "audit"),
    ) -> None:
        if not enabled:
            raise RuntimeError("research-only ablation harness is disabled by default")
        self.variant = ResearchVariant(variant)
        self.policy = policy or build_default_policy()
        self.witness_limit = witness_limit
        self.max_candidates = max_candidates
        self.capabilities = frozenset(capabilities)
        self.compiler = EOCCompiler(policy=self.policy)
        self.control_planner = EnforceableControlPlanner()

    @property
    def variant_sha256(self) -> str:
        path = Path(__file__).resolve()
        material = {
            "harness_version": RESEARCH_HARNESS_VERSION,
            "variant": self.variant.value,
            "policy_digest": self.policy.digest,
            "witness_limit": self.witness_limit,
            "max_candidates": self.max_candidates,
            "capabilities": sorted(self.capabilities),
            "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()

    def decide(self, action: ActionCandidate, evidence: RuntimeEvidence) -> ResearchDecision:
        graph = build_runtime_action_graph(action, evidence)
        requirements = requirements_for_action(action)
        assertions = tuple(
            item
            for item in evidence.assertions
            if not (
                item.atom.polarity == "+"
                and is_authorization_predicate(item.atom.predicate)
                and not item.trusted_for_authorization
            )
        )
        rejected_enabling_assertions = tuple(item for item in evidence.assertions if item not in assertions)
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
        if self.variant is ResearchVariant.CONFLICT_COLLAPSE:
            effective_closure = _collapse_conflicts(original_closure)

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
            raw_eoc = None
            eoc = self.compiler.compile(
                requirements=requirements,
                closure=effective_closure,
                hazards=evidence.hazards,
                explicit_denies=explicit_denies,
            )

        default_plan = self.policy.default_plan
        trigger_remedies = [
            tuple(remedy.plan for remedy in eoc.remedies[trigger.trigger_id]) for trigger in eoc.triggers
        ]
        realizations = tuple(self._materialize(item) for item in self.policy.realizations)
        if self.variant is ResearchVariant.NO_LIFTED_JOIN:
            # Only Base and individual Trigger-local remedies are candidates.
            # Every candidate is still checked against *all* active duties by
            # the unchanged lattice selector; if none is feasible it falls back.
            local_library = tuple(
                {plan.digest: plan for plan in (default_plan, *(p for group in trigger_remedies for p in group))}.values()
            )
            selection = self.control_planner.select(
                base_remedies=(default_plan,),
                trigger_remedies=(),
                obligations=eoc.semantic_obligations,
                implementation_constraints=eoc.implementation_constraints,
                plan_library=local_library,
                realizations=realizations,
                default_plan=default_plan,
                max_candidates=self.max_candidates,
                approval_satisfied=evidence.confirmation is not None,
            )
        else:
            selection = self.control_planner.select(
                base_remedies=(default_plan,),
                trigger_remedies=trigger_remedies,
                obligations=eoc.semantic_obligations,
                implementation_constraints=eoc.implementation_constraints,
                plan_library=self.policy.behavior_domain,
                realizations=realizations,
                default_plan=default_plan,
                max_candidates=self.max_candidates,
                approval_satisfied=evidence.confirmation is not None,
            )
        return self._decision(
            action=action,
            graph=graph.to_dict(),
            original_closure=original_closure,
            effective_closure=effective_closure,
            rejected_enabling_assertions=rejected_enabling_assertions,
            eoc=eoc,
            raw_eoc=raw_eoc,
            selection=selection,
        )

    def _materialize(self, template: Realization) -> Realization:
        required = set(template.capabilities)
        return replace(
            template,
            # Preserve the production realization identity so offline deltas
            # reflect only the named mechanism change, not a logging prefix.
            realization_id=f"runtime:{template.realization_id}",
            capabilities=frozenset(required & self.capabilities),
            actual_audit=template.actual_audit if "audit" in self.capabilities else frozenset(),
            available=template.available and required <= self.capabilities,
        )

    def _decision(
        self,
        *,
        action: ActionCandidate,
        graph: Mapping[str, Any],
        original_closure: WitnessClosure,
        effective_closure: WitnessClosure,
        rejected_enabling_assertions: tuple[Any, ...],
        eoc: EOCResult,
        raw_eoc: EOCResult | None,
        selection: ControlPlanSelection,
    ) -> ResearchDecision:
        trace = {
            "harness_version": RESEARCH_HARNESS_VERSION,
            "variant_sha256": self.variant_sha256,
            "research_only": True,
            "verified": False,
            "acceptance_record": None,
            "certificate": None,
            "main_theorem_guarantee": False,
            "policy_definition_digest": self.policy.digest,
            "action_digest": action.digest,
            "action_graph": graph,
            "witness_closure_original": original_closure.to_dict(),
            "witness_closure_effective": effective_closure.to_dict(),
            "rejected_enabling_assertions": [item.to_dict() for item in rejected_enabling_assertions],
            "gap_before_mask": list(raw_eoc.gaps) if raw_eoc is not None else list(eoc.gaps),
            "gap": list(eoc.gaps),
            "hazard": list(eoc.hazards),
            "overflow": list(eoc.overflow),
            "triggers": [item.to_dict() for item in eoc.triggers],
            "valid_certificate_preconditions_only": eoc.valid_certificate_evidence,
            "eoc": eoc.to_dict(),
            "lattice": selection.to_dict(),
            "local_candidate_only": self.variant is ResearchVariant.NO_LIFTED_JOIN,
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


def _collapse_conflicts(closure: WitnessClosure) -> WitnessClosure:
    witnesses = dict(closure.witnesses)
    unsigned = {key[1:] for key in witnesses}
    for key in unsigned:
        positive = tuple(witnesses.get(f"+{key}", ()))
        negative = tuple(witnesses.get(f"-{key}", ()))
        if positive and negative:
            # The sole semantic change: a positive antichain collapses a
            # conflict to support-only. Refute-only and unknown are untouched.
            witnesses[f"-{key}"] = ()
    return WitnessClosure(
        witnesses=witnesses,
        overflow=closure.overflow,
        overflow_paths=closure.overflow_paths,
        rule_firings=closure.rule_firings,
        exact=closure.exact,
        witness_limit=closure.witness_limit,
    )


def variant_hashes() -> dict[str, str]:
    return {
        variant.value: ResearchRuntime(variant, enabled=True).variant_sha256
        for variant in ResearchVariant
    }


__all__ = [
    "RESEARCH_HARNESS_VERSION",
    "ResearchDecision",
    "ResearchRuntime",
    "ResearchVariant",
    "variant_hashes",
]
