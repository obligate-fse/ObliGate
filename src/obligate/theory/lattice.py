"""Five-dimensional lifted ECP control domain and selector."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from .model import (
    BehaviorPlan,
    ImplementationConstraint,
    PublicDecision,
    Realization,
    SemanticObligation,
    exec_semantics,
)

ENV_ORDER = ("host", "sandbox", "no_execute")
NETWORK_ORDER = ("allow", "allowlist", "deny")
DATA_ORDER = ("raw", "redact", "no_sensitive")
HUMAN_ORDER = ("none", "approval_required")


def _rank(value: str, order: tuple[str, ...]) -> int:
    try:
        return order.index(value)
    except ValueError as exc:  # defensive: dataclass Literal is not runtime enforcement
        raise ValueError(f"unknown lattice value {value!r}") from exc


def plan_join(left: BehaviorPlan, right: BehaviorPlan) -> BehaviorPlan:
    """Coordinate join plus Must-union/Allowed-intersection audit join."""

    return BehaviorPlan(
        execution_env=ENV_ORDER[max(_rank(left.execution_env, ENV_ORDER), _rank(right.execution_env, ENV_ORDER))],  # type: ignore[arg-type]
        network_scope=NETWORK_ORDER[
            max(_rank(left.network_scope, NETWORK_ORDER), _rank(right.network_scope, NETWORK_ORDER))
        ],  # type: ignore[arg-type]
        data_scope=DATA_ORDER[max(_rank(left.data_scope, DATA_ORDER), _rank(right.data_scope, DATA_ORDER))],  # type: ignore[arg-type]
        human_gate=HUMAN_ORDER[max(_rank(left.human_gate, HUMAN_ORDER), _rank(right.human_gate, HUMAN_ORDER))],  # type: ignore[arg-type]
        audit=left.audit.join(right.audit),
    )


def safety_no_weaker(candidate: BehaviorPlan, baseline: BehaviorPlan) -> bool:
    """Return ``baseline <=_s candidate`` (candidate is at least as strong)."""

    return (
        exec_semantics(candidate) <= exec_semantics(baseline)
        and candidate.audit.no_weaker_than(baseline.audit)
    )


def safety_stronger(candidate: BehaviorPlan, baseline: BehaviorPlan) -> bool:
    return candidate != baseline and safety_no_weaker(candidate, baseline)


def normalize_plans(plans: Iterable[BehaviorPlan]) -> tuple[BehaviorPlan, ...]:
    unique = {item.digest: item for item in plans}
    return tuple(unique[key] for key in sorted(unique))


def minimal_plans(plans: Iterable[BehaviorPlan]) -> tuple[BehaviorPlan, ...]:
    """Return Min_{<=s}: the least restrictive non-dominated plans."""

    values = normalize_plans(plans)
    return tuple(
        candidate
        for candidate in values
        if not any(safety_stronger(candidate, other) for other in values if other != candidate)
    )


def lifted_join(left: Sequence[BehaviorPlan], right: Sequence[BehaviorPlan]) -> tuple[BehaviorPlan, ...]:
    if not left or not right:
        return ()
    return minimal_plans(plan_join(a, b) for a, b in product(left, right))


def lifted_join_all(remedy_antichains: Sequence[Sequence[BehaviorPlan]]) -> tuple[BehaviorPlan, ...]:
    if not remedy_antichains:
        return ()
    current = tuple(remedy_antichains[0])
    for antichain in remedy_antichains[1:]:
        current = lifted_join(current, tuple(antichain))
        if not current:
            break
    return minimal_plans(current)


def satisfies_obligation(plan: BehaviorPlan, obligation: SemanticObligation) -> bool:
    if not plan.audit.consistent:
        return False
    if obligation.forbid_confirmation and plan.confirmation:
        return False
    if obligation.forbid_dispatch and not plan.block_like:
        return False
    if obligation.min_execution_env is not None and _rank(plan.execution_env, ENV_ORDER) < _rank(
        obligation.min_execution_env, ENV_ORDER
    ):
        return False
    if obligation.min_network_scope is not None and _rank(plan.network_scope, NETWORK_ORDER) < _rank(
        obligation.min_network_scope, NETWORK_ORDER
    ):
        return False
    if obligation.min_data_scope is not None and _rank(plan.data_scope, DATA_ORDER) < _rank(
        obligation.min_data_scope, DATA_ORDER
    ):
        return False
    if obligation.require_approval and plan.human_gate != "approval_required" and not plan.block_like:
        return False
    if not obligation.audit_must <= plan.audit.must:
        return False
    return True


def satisfies_all(plan: BehaviorPlan, obligations: Sequence[SemanticObligation]) -> bool:
    return all(satisfies_obligation(plan, item) for item in obligations)


def realization_satisfies(
    realization: Realization,
    plan: BehaviorPlan,
    implementation_constraints: Sequence[ImplementationConstraint],
) -> bool:
    if not realization.available or realization.plan != plan:
        return False
    required = {item.capability for item in implementation_constraints}
    if plan.block_like:
        # Execution controls are alternative implementations of a Trigger,
        # not prerequisites for the fail-stop remedy.  A compliant block
        # still has to realize every audit/state capability explicitly.
        required = {item for item in required if item in {"audit", "state_store"}}
    if not required <= realization.capabilities:
        return False
    if not plan.audit.must <= realization.actual_audit <= plan.audit.allowed:
        return False
    if any(
        not root.startswith(("trigger:", "backend:", "baseline:"))
        for _, root in realization.control_roots
    ):
        return False
    return True


def utility_dominates(left: Realization, right: Realization) -> bool:
    no_worse = all(a <= b for a, b in zip(left.utility_vector, right.utility_vector))
    strictly_better = any(a < b for a, b in zip(left.utility_vector, right.utility_vector))
    return no_worse and strictly_better


def utility_frontier(realizations: Iterable[Realization]) -> tuple[Realization, ...]:
    candidates = tuple(sorted(realizations, key=lambda item: item.realization_id))
    return tuple(
        item
        for item in candidates
        if not any(utility_dominates(other, item) for other in candidates if other != item)
    )


@dataclass(frozen=True, slots=True)
class ControlPlanSelection:
    outcome: PublicDecision
    ideal: tuple[BehaviorPlan, ...]
    safe_candidates: tuple[BehaviorPlan, ...]
    feasible_realizations: tuple[Realization, ...]
    non_blocking: tuple[Realization, ...]
    frontier: tuple[Realization, ...]
    selected: Realization | None
    compliant_block: Realization | None
    search_overflow: bool = False

    @property
    def execute(self) -> bool:
        return self.outcome in {"allow", "execute_with_constraints"}

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "ideal": [item.to_dict() for item in self.ideal],
            "safe_candidates": [item.to_dict() for item in self.safe_candidates],
            "feasible_realizations": [item.to_dict() for item in self.feasible_realizations],
            "non_blocking": [item.to_dict() for item in self.non_blocking],
            "frontier": [item.to_dict() for item in self.frontier],
            "selected": self.selected.to_dict() if self.selected else None,
            "compliant_block": self.compliant_block.to_dict() if self.compliant_block else None,
            "search_overflow": self.search_overflow,
        }


class EnforceableControlPlanner:
    """Lifted join, realization filtering, Pareto selection and fallback."""

    def select(
        self,
        *,
        base_remedies: Sequence[BehaviorPlan],
        trigger_remedies: Sequence[Sequence[BehaviorPlan]],
        obligations: Sequence[SemanticObligation],
        implementation_constraints: Sequence[ImplementationConstraint],
        plan_library: Sequence[BehaviorPlan],
        realizations: Sequence[Realization],
        default_plan: BehaviorPlan,
        max_candidates: int | None = None,
        approval_satisfied: bool = False,
    ) -> ControlPlanSelection:
        antichains: list[Sequence[BehaviorPlan]] = [tuple(base_remedies)]
        antichains.extend(tuple(item) for item in trigger_remedies)
        ideal = lifted_join_all(antichains)
        if not ideal:
            ideal = ()

        library = normalize_plans((*plan_library, *ideal))
        search_overflow = max_candidates is not None and len(library) > max_candidates
        if search_overflow:
            # A bounded search may omit a non-blocking plan.  It therefore
            # cannot return a normal result; membership must fail closed.
            library = library[: max_candidates]

        safe = tuple(
            plan
            for plan in library
            if plan.audit.consistent
            and satisfies_all(plan, obligations)
            and any(safety_no_weaker(plan, lower_bound) for lower_bound in ideal)
        )
        feasible: list[Realization] = []
        for plan in safe:
            feasible.extend(
                item
                for item in realizations
                if realization_satisfies(item, plan, implementation_constraints)
            )
        feasible_tuple = tuple(sorted({item.realization_id: item for item in feasible}.values(), key=lambda item: item.realization_id))
        non_blocking = tuple(item for item in feasible_tuple if not item.plan.block_like)

        if search_overflow:
            return self._fallback(
                ideal=ideal,
                safe=safe,
                feasible=feasible_tuple,
                realizations=realizations,
                obligations=obligations,
                implementation_constraints=implementation_constraints,
                search_overflow=True,
            )

        if non_blocking:
            frontier = utility_frontier(non_blocking)
            selected = min(frontier, key=lambda item: (*item.utility_vector, item.realization_id))
            outcome = project_public(
                selected,
                default_plan,
                approval_satisfied=approval_satisfied,
            )
            return ControlPlanSelection(
                outcome=outcome,
                ideal=ideal,
                safe_candidates=safe,
                feasible_realizations=feasible_tuple,
                non_blocking=non_blocking,
                frontier=frontier,
                selected=selected,
                compliant_block=None,
            )

        return self._fallback(
            ideal=ideal,
            safe=safe,
            feasible=feasible_tuple,
            realizations=realizations,
            obligations=obligations,
            implementation_constraints=implementation_constraints,
            search_overflow=False,
        )

    def _fallback(
        self,
        *,
        ideal: tuple[BehaviorPlan, ...],
        safe: tuple[BehaviorPlan, ...],
        feasible: tuple[Realization, ...],
        realizations: Sequence[Realization],
        obligations: Sequence[SemanticObligation],
        implementation_constraints: Sequence[ImplementationConstraint],
        search_overflow: bool,
    ) -> ControlPlanSelection:
        compliant_blocks = [
            item
            for item in realizations
            if item.plan.block_like
            and satisfies_all(item.plan, obligations)
            and realization_satisfies(item, item.plan, implementation_constraints)
        ]
        blocker = min(compliant_blocks, key=lambda item: (*item.utility_vector, item.realization_id)) if compliant_blocks else None
        outcome: PublicDecision = "block" if blocker is not None else "block_with_compliance_error"
        return ControlPlanSelection(
            outcome=outcome,
            ideal=ideal,
            safe_candidates=safe,
            feasible_realizations=feasible,
            non_blocking=(),
            frontier=(),
            selected=blocker,
            compliant_block=blocker,
            search_overflow=search_overflow,
        )


def project_public(
    realization: Realization,
    default_plan: BehaviorPlan,
    *,
    approval_satisfied: bool = False,
) -> PublicDecision:
    plan = realization.plan
    if plan.block_like:
        return "block"
    if plan.confirmation and not approval_satisfied:
        return "require_confirmation"
    non_default_concrete = any(
        root != "baseline" and not root.startswith("baseline:")
        for _, root in realization.control_roots
    )
    if plan == default_plan and not non_default_concrete:
        return "allow"
    return "execute_with_constraints"


__all__ = [
    "EnforceableControlPlanner",
    "DATA_ORDER",
    "ENV_ORDER",
    "HUMAN_ORDER",
    "ControlPlanSelection",
    "NETWORK_ORDER",
    "lifted_join",
    "lifted_join_all",
    "minimal_plans",
    "plan_join",
    "project_public",
    "realization_satisfies",
    "safety_no_weaker",
    "satisfies_all",
    "satisfies_obligation",
    "utility_frontier",
]
