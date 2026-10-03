"""Finite policy specification for the theory-aligned checker.

The policy objects in this module are intentionally declarative.  A policy
contains only finite domains, grounded signed rules, finite transition
relations, named Trigger compilations, remedies and concrete realizations.
Consequently :mod:`obligate.theory.checker` can enumerate every legal fact
valuation and every abstract behavior instead of trusting policy-supplied
``passed`` flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Mapping

from .model import (
    THEORY_SEMANTIC_VERSION,
    AuditContract,
    AuditUnit,
    BehaviorPlan,
    ImplementationConstraint,
    Realization,
    Remedy,
    SemanticObligation,
    SignedAtom,
    SignedRule,
    TriggerKind,
    exec_semantics,
    sha256_json,
)

EvidenceValue = Literal["support-only", "refute-only", "conflict", "unknown", "overflow"]

EVIDENCE_VALUES: tuple[EvidenceValue, ...] = (
    "support-only",
    "refute-only",
    "conflict",
    "unknown",
    "overflow",
)

EXECUTION_ORDER = {"host": 0, "sandbox": 1, "no_execute": 2}
NETWORK_ORDER = {"allow": 0, "allowlist": 1, "deny": 2}
DATA_ORDER = {"raw": 0, "redact": 1, "no_sensitive": 2}
HUMAN_ORDER = {"none": 0, "approval_required": 1}


@dataclass(frozen=True, slots=True)
class AtomDomain:
    """Finite domain for one grounded unsigned atom."""

    atom: str
    values: tuple[EvidenceValue, ...] = EVIDENCE_VALUES
    description: str = ""

    def __post_init__(self) -> None:
        if not self.atom or self.atom[0] in {"+", "-"}:
            raise ValueError("AtomDomain.atom must be a non-empty unsigned atom key")
        values = tuple(self.values)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"atom domain {self.atom!r} must be non-empty and duplicate-free")
        invalid = set(values) - set(EVIDENCE_VALUES)
        if invalid:
            raise ValueError(f"atom domain {self.atom!r} has unsupported values: {sorted(invalid)}")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {"atom": self.atom, "values": list(self.values), "description": self.description}


@dataclass(frozen=True, slots=True)
class StatusLiteral:
    atom: str
    values: frozenset[EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", frozenset(self.values))
        if not self.atom or not self.values:
            raise ValueError("StatusLiteral requires an atom and at least one value")
        invalid = set(self.values) - set(EVIDENCE_VALUES)
        if invalid:
            raise ValueError(f"StatusLiteral has unsupported values: {sorted(invalid)}")

    def matches(self, valuation: Mapping[str, EvidenceValue]) -> bool:
        return valuation.get(self.atom) in self.values

    def to_dict(self) -> dict[str, Any]:
        return {"atom": self.atom, "values": sorted(self.values)}


@dataclass(frozen=True, slots=True)
class FiniteCondition:
    """Small condition language used by invariants and finite transitions.

    ``all_of`` is conjunctive, ``any_of`` is an optional disjunction and
    ``none_of`` excludes matching literals.  Empty conditions denote True.
    """

    all_of: tuple[StatusLiteral, ...] = ()
    any_of: tuple[StatusLiteral, ...] = ()
    none_of: tuple[StatusLiteral, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "all_of", tuple(self.all_of))
        object.__setattr__(self, "any_of", tuple(self.any_of))
        object.__setattr__(self, "none_of", tuple(self.none_of))

    def matches(self, valuation: Mapping[str, EvidenceValue]) -> bool:
        return (
            all(item.matches(valuation) for item in self.all_of)
            and (not self.any_of or any(item.matches(valuation) for item in self.any_of))
            and not any(item.matches(valuation) for item in self.none_of)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_of": [item.to_dict() for item in self.all_of],
            "any_of": [item.to_dict() for item in self.any_of],
            "none_of": [item.to_dict() for item in self.none_of],
        }


TRUE_CONDITION = FiniteCondition()


@dataclass(frozen=True, slots=True)
class DomainInvariant:
    invariant_id: str
    antecedent: FiniteCondition
    consequent: FiniteCondition
    description: str = ""

    def allows(self, valuation: Mapping[str, EvidenceValue]) -> bool:
        return not self.antecedent.matches(valuation) or self.consequent.matches(valuation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "antecedent": self.antecedent.to_dict(),
            "consequent": self.consequent.to_dict(),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class StateInvariant:
    invariant_id: str
    condition: FiniteCondition
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "condition": self.condition.to_dict(),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class PlanCondition:
    execution_env: frozenset[str] | None = None
    network_scope: frozenset[str] | None = None
    data_scope: frozenset[str] | None = None
    human_gate: frozenset[str] | None = None
    dispatch: bool | None = None
    confirmation: bool | None = None
    audit_must_contains: frozenset[AuditUnit] = frozenset()

    def __post_init__(self) -> None:
        for name in ("execution_env", "network_scope", "data_scope", "human_gate"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, frozenset(value))
        object.__setattr__(self, "audit_must_contains", frozenset(self.audit_must_contains))

    def matches(self, plan: BehaviorPlan) -> bool:
        if self.execution_env is not None and plan.execution_env not in self.execution_env:
            return False
        if self.network_scope is not None and plan.network_scope not in self.network_scope:
            return False
        if self.data_scope is not None and plan.data_scope not in self.data_scope:
            return False
        if self.human_gate is not None and plan.human_gate not in self.human_gate:
            return False
        if self.dispatch is not None and (not plan.block_like) != self.dispatch:
            return False
        if self.confirmation is not None and plan.confirmation != self.confirmation:
            return False
        return self.audit_must_contains <= plan.audit.must

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_env": sorted(self.execution_env) if self.execution_env is not None else None,
            "network_scope": sorted(self.network_scope) if self.network_scope is not None else None,
            "data_scope": sorted(self.data_scope) if self.data_scope is not None else None,
            "human_gate": sorted(self.human_gate) if self.human_gate is not None else None,
            "dispatch": self.dispatch,
            "confirmation": self.confirmation,
            "audit_must_contains": [item.to_dict() for item in sorted(self.audit_must_contains)],
        }


ANY_PLAN = PlanCondition()


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    transition_id: str
    when: FiniteCondition = TRUE_CONDITION
    plan_when: PlanCondition = ANY_PLAN
    updates: tuple[tuple[str, EvidenceValue], ...] = ()

    def __post_init__(self) -> None:
        updates = tuple(sorted((str(key), value) for key, value in self.updates))
        if len({key for key, _ in updates}) != len(updates):
            raise ValueError(f"transition {self.transition_id!r} has duplicate updates")
        object.__setattr__(self, "updates", updates)

    def applies(self, valuation: Mapping[str, EvidenceValue], plan: BehaviorPlan) -> bool:
        return self.when.matches(valuation) and self.plan_when.matches(plan)

    def successor(self, valuation: Mapping[str, EvidenceValue]) -> dict[str, EvidenceValue]:
        result = dict(valuation)
        result.update(self.updates)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "when": self.when.to_dict(),
            "plan_when": self.plan_when.to_dict(),
            "updates": [{"atom": key, "value": value} for key, value in self.updates],
        }


@dataclass(frozen=True, slots=True)
class GuardedActionSpec:
    action_type: str
    tool_schemas: frozenset[str] = frozenset()
    side_effects: frozenset[str] = frozenset()
    external_sinks: frozenset[str] = frozenset()
    state_mutations: frozenset[str] = frozenset()
    sensitive_reads: frozenset[str] = frozenset()
    privileged_accesses: frozenset[str] = frozenset()
    critical_parameters: frozenset[str] = frozenset()
    targets: frozenset[str] = frozenset()
    flows: frozenset[str] = frozenset()
    requirement_claims: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "tool_schemas",
            "side_effects",
            "external_sinks",
            "state_mutations",
            "sensitive_reads",
            "privileged_accesses",
            "critical_parameters",
            "targets",
            "flows",
        ):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        claims = tuple(sorted((str(key), str(value)) for key, value in self.requirement_claims))
        if len({key for key, _ in claims}) != len(claims):
            raise ValueError(f"action {self.action_type!r} has duplicate requirement objects")
        object.__setattr__(self, "requirement_claims", claims)

    @property
    def guarded(self) -> bool:
        return any(
            (
                self.side_effects,
                self.external_sinks,
                self.state_mutations,
                self.sensitive_reads,
                self.privileged_accesses,
            )
        )

    @property
    def required_objects(self) -> frozenset[str]:
        groups = {
            "tool_schema": self.tool_schemas,
            "side_effect": self.side_effects,
            "external_sink": self.external_sinks,
            "state_mutation": self.state_mutations,
            "sensitive_read": self.sensitive_reads,
            "privileged_access": self.privileged_accesses,
            "parameter": self.critical_parameters,
            "target": self.targets,
            "flow": self.flows,
        }
        return frozenset(f"{kind}:{value}" for kind, values in groups.items() for value in values)

    @property
    def claim_map(self) -> dict[str, str]:
        return dict(self.requirement_claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "guarded": self.guarded,
            "required_objects": sorted(self.required_objects),
            "requirement_claims": dict(self.requirement_claims),
        }


@dataclass(frozen=True, slots=True)
class TriggerSpec:
    trigger_id: str
    kind: TriggerKind
    subject: str
    hard: bool
    semantic_obligations: tuple[SemanticObligation, ...]
    implementation_constraints: tuple[ImplementationConstraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_obligations", tuple(self.semantic_obligations))
        object.__setattr__(self, "implementation_constraints", tuple(self.implementation_constraints))

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.subject)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "kind": self.kind,
            "subject": self.subject,
            "hard": self.hard,
            "semantic_obligations": [item.to_dict() for item in self.semantic_obligations],
            "implementation_constraints": [item.to_dict() for item in self.implementation_constraints],
        }


@dataclass(frozen=True, slots=True)
class BadEffectSpec:
    effect_id: str
    facts: FiniteCondition
    plan: PlanCondition
    description: str = ""

    def occurs(self, valuation: Mapping[str, EvidenceValue], plan: BehaviorPlan) -> bool:
        return self.facts.matches(valuation) and self.plan.matches(plan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "facts": self.facts.to_dict(),
            "plan": self.plan.to_dict(),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class SelectorSpec:
    forced_realization_id: str | None = None
    utility_order: tuple[str, ...] = ("task_loss", "confirmation_burden", "runtime_cost")
    tie_break: str = "realization_id"

    def to_dict(self) -> dict[str, Any]:
        return {
            "forced_realization_id": self.forced_realization_id,
            "utility_order": list(self.utility_order),
            "tie_break": self.tie_break,
        }


REQUIRED_CONFIRMATION_BINDINGS = frozenset(
    {
        "nonce",
        "actor",
        "tenant",
        "session",
        "tool",
        "audience",
        "action",
        "parameters",
        "payload",
        "assets",
        "evidence",
        "policy_digest",
        "plan_digest",
        "initial_certificate_digest",
        "confirmation_binding_digest",
        "expiry",
    }
)


@dataclass(frozen=True, slots=True)
class ConfirmationSpec:
    bound_fields: frozenset[str]
    one_time: bool = True
    re_adjudicate: bool = True
    atomic_claim_and_outbox: bool = True
    immutable_outbox: bool = True
    no_retry_without_idempotency: bool = True
    transitions: tuple[tuple[str, str], ...] = (
        ("none", "pending"),
        ("pending", "fresh"),
        ("pending", "closed"),
        ("fresh", "claimed"),
        ("claimed", "sending"),
        ("sending", "done"),
        ("sending", "uncertain"),
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "bound_fields", frozenset(self.bound_fields))
        object.__setattr__(self, "transitions", tuple(sorted(set(self.transitions))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bound_fields": sorted(self.bound_fields),
            "one_time": self.one_time,
            "re_adjudicate": self.re_adjudicate,
            "atomic_claim_and_outbox": self.atomic_claim_and_outbox,
            "immutable_outbox": self.immutable_outbox,
            "no_retry_without_idempotency": self.no_retry_without_idempotency,
            "transitions": [list(item) for item in self.transitions],
        }


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    atom_domains: tuple[AtomDomain, ...]
    domain_invariants: tuple[DomainInvariant, ...]
    state_invariants: tuple[StateInvariant, ...]
    transitions: tuple[TransitionSpec, ...]
    rules: tuple[SignedRule, ...]
    guarded_actions: tuple[GuardedActionSpec, ...]
    hazard_atoms: frozenset[str]
    overflow_atoms: frozenset[str]
    explicit_deny_atoms: frozenset[str]
    triggers: tuple[TriggerSpec, ...]
    remedies: tuple[Remedy, ...]
    fallback_remedies: tuple[Remedy, ...]
    behavior_domain: tuple[BehaviorPlan, ...]
    default_plan: BehaviorPlan
    realizations: tuple[Realization, ...]
    bad_effects: tuple[BadEffectSpec, ...]
    selector: SelectorSpec
    confirmation: ConfirmationSpec
    declared_public_decisions: tuple[tuple[str, str], ...]
    runtime_claim_templates: tuple[tuple[str, str], ...]
    runtime_hazard_templates: tuple[tuple[str, str], ...]
    semantic_version: str = THEORY_SEMANTIC_VERSION
    max_cartesian_product: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "atom_domains",
            "domain_invariants",
            "state_invariants",
            "transitions",
            "rules",
            "guarded_actions",
            "triggers",
            "remedies",
            "fallback_remedies",
            "behavior_domain",
            "realizations",
            "bad_effects",
            "declared_public_decisions",
            "runtime_claim_templates",
            "runtime_hazard_templates",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("hazard_atoms", "overflow_atoms", "explicit_deny_atoms"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))
        if not self.name:
            raise ValueError("policy name must be non-empty")
        if self.max_cartesian_product < 1:
            raise ValueError("max_cartesian_product must be positive")

    @property
    def atom_domain_map(self) -> dict[str, AtomDomain]:
        return {item.atom: item for item in self.atom_domains}

    @property
    def trigger_map(self) -> dict[tuple[str, str], TriggerSpec]:
        return {item.key: item for item in self.triggers}

    @property
    def bad_effect_map(self) -> dict[str, BadEffectSpec]:
        return {item.effect_id: item for item in self.bad_effects}

    @property
    def runtime_claim_map(self) -> dict[str, str]:
        return dict(self.runtime_claim_templates)

    @property
    def runtime_hazard_map(self) -> dict[str, str]:
        return dict(self.runtime_hazard_templates)

    @property
    def required_claims(self) -> frozenset[str]:
        return frozenset(claim for action in self.guarded_actions for _, claim in action.requirement_claims)

    @property
    def theoretical_cartesian_product(self) -> int:
        count = 1
        for domain in self.atom_domains:
            count *= len(domain.values)
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_version": self.semantic_version,
            "atom_domains": [item.to_dict() for item in self.atom_domains],
            "domain_invariants": [item.to_dict() for item in self.domain_invariants],
            "state_invariants": [item.to_dict() for item in self.state_invariants],
            "transitions": [item.to_dict() for item in self.transitions],
            "rules": [item.to_dict() for item in self.rules],
            "guarded_actions": [item.to_dict() for item in self.guarded_actions],
            "hazard_atoms": sorted(self.hazard_atoms),
            "overflow_atoms": sorted(self.overflow_atoms),
            "explicit_deny_atoms": sorted(self.explicit_deny_atoms),
            "triggers": [item.to_dict() for item in self.triggers],
            "remedies": [item.to_dict() for item in self.remedies],
            "fallback_remedies": [item.to_dict() for item in self.fallback_remedies],
            "behavior_domain": [item.to_dict() for item in self.behavior_domain],
            "default_plan": self.default_plan.to_dict(),
            "realizations": [item.to_dict() for item in self.realizations],
            "bad_effects": [item.to_dict() for item in self.bad_effects],
            "selector": self.selector.to_dict(),
            "confirmation": self.confirmation.to_dict(),
            "declared_public_decisions": dict(self.declared_public_decisions),
            "runtime_claim_templates": dict(self.runtime_claim_templates),
            "runtime_hazard_templates": dict(self.runtime_hazard_templates),
            "max_cartesian_product": self.max_cartesian_product,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def positive_present(value: EvidenceValue | None) -> bool:
    return value in {"support-only", "conflict"}


def signed_atom_present(valuation: Mapping[str, EvidenceValue], atom: SignedAtom) -> bool:
    value = valuation.get(atom.unsigned_key)
    if atom.polarity == "+":
        return positive_present(value)
    return value in {"refute-only", "conflict"}


def invariant_holds(policy: PolicySpec, valuation: Mapping[str, EvidenceValue]) -> bool:
    return all(item.condition.matches(valuation) for item in policy.state_invariants)


def obligation_satisfied(plan: BehaviorPlan, obligation: SemanticObligation) -> bool:
    if obligation.forbid_dispatch and not plan.block_like:
        return False
    if obligation.forbid_confirmation and plan.confirmation:
        return False
    if obligation.min_execution_env is not None and EXECUTION_ORDER[plan.execution_env] < EXECUTION_ORDER[obligation.min_execution_env]:
        return False
    if obligation.min_network_scope is not None and NETWORK_ORDER[plan.network_scope] < NETWORK_ORDER[obligation.min_network_scope]:
        return False
    if obligation.min_data_scope is not None and DATA_ORDER[plan.data_scope] < DATA_ORDER[obligation.min_data_scope]:
        return False
    # A safe blocking remedy does not need to obtain an approval merely to
    # refrain from dispatching the protected action.
    if obligation.require_approval and not plan.block_like and plan.human_gate != "approval_required":
        return False
    return obligation.audit_must <= plan.audit.must


def plan_no_weaker_than(left: BehaviorPlan, right: BehaviorPlan) -> bool:
    """Return ``right <=_s left`` (``left`` is at least as strong)."""

    return (
        exec_semantics(left) <= exec_semantics(right)
        and left.audit.no_weaker_than(right.audit)
    )


def join_behavior_plans(left: BehaviorPlan, right: BehaviorPlan) -> BehaviorPlan:
    def stronger(first: str, second: str, order: Mapping[str, int]) -> str:
        return first if order[first] >= order[second] else second

    return BehaviorPlan(
        execution_env=stronger(left.execution_env, right.execution_env, EXECUTION_ORDER),  # type: ignore[arg-type]
        network_scope=stronger(left.network_scope, right.network_scope, NETWORK_ORDER),  # type: ignore[arg-type]
        data_scope=stronger(left.data_scope, right.data_scope, DATA_ORDER),  # type: ignore[arg-type]
        human_gate=stronger(left.human_gate, right.human_gate, HUMAN_ORDER),  # type: ignore[arg-type]
        audit=left.audit.join(right.audit),
    )


def expected_trigger_keys(policy: PolicySpec, valuation: Mapping[str, EvidenceValue]) -> frozenset[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for claim in policy.required_claims:
        if valuation.get(claim) != "support-only":
            result.add(("gap", claim))
    for atom in policy.hazard_atoms:
        if positive_present(valuation.get(atom)):
            result.add(("hazard", atom))
    for atom in policy.overflow_atoms:
        if valuation.get(atom) == "overflow":
            result.add(("overflow", atom))
    for atom in policy.explicit_deny_atoms:
        if positive_present(valuation.get(atom)):
            result.add(("hazard", atom))
    return frozenset(result)


def active_trigger_specs(policy: PolicySpec, valuation: Mapping[str, EvidenceValue]) -> tuple[TriggerSpec, ...]:
    expected = expected_trigger_keys(policy, valuation)
    return tuple(item for item in policy.triggers if item.key in expected)


def realization_feasible(realization: Realization, triggers: tuple[TriggerSpec, ...]) -> bool:
    if not realization.available:
        return False
    obligations = tuple(obligation for trigger in triggers for obligation in trigger.semantic_obligations)
    if not all(obligation_satisfied(realization.plan, obligation) for obligation in obligations):
        return False
    capabilities = set(realization.capabilities)
    constraints = tuple(
        constraint
        for trigger in triggers
        for constraint in trigger.implementation_constraints
    )
    if realization.plan.block_like:
        constraints = tuple(
            item for item in constraints if item.capability in {"audit", "state_store"}
        )
    return all(
        constraint.capability in capabilities
        for constraint in constraints
    )


def utility_frontier(realizations: tuple[Realization, ...]) -> tuple[Realization, ...]:
    def dominates(left: Realization, right: Realization) -> bool:
        return all(a <= b for a, b in zip(left.utility_vector, right.utility_vector)) and any(
            a < b for a, b in zip(left.utility_vector, right.utility_vector)
        )

    return tuple(
        item
        for item in sorted(realizations, key=lambda value: value.realization_id)
        if not any(dominates(other, item) for other in realizations if other is not item)
    )


def select_realization(policy: PolicySpec, feasible: tuple[Realization, ...]) -> Realization | None:
    if policy.selector.forced_realization_id is not None:
        return next((item for item in feasible if item.realization_id == policy.selector.forced_realization_id), None)
    non_blocking = tuple(item for item in feasible if not item.plan.block_like)
    candidates = non_blocking or tuple(item for item in feasible if item.plan.block_like)
    frontier = utility_frontier(candidates)
    return min(frontier, key=lambda item: (*item.utility_vector, item.realization_id), default=None)


def strict_public_projection(policy: PolicySpec, realization: Realization) -> str:
    plan = realization.plan
    if plan.block_like:
        return "block"
    if plan.confirmation:
        return "require_confirmation"
    if plan == policy.default_plan and not realization.control_roots:
        return "allow"
    return "execute_with_constraints"


def enumerate_behavior_domain(audits: tuple[AuditContract, ...]) -> tuple[BehaviorPlan, ...]:
    plans = []
    for env, network, data, human, audit in product(
        EXECUTION_ORDER,
        NETWORK_ORDER,
        DATA_ORDER,
        HUMAN_ORDER,
        audits,
    ):
        plans.append(BehaviorPlan(env, network, data, human, audit))  # type: ignore[arg-type]
    unique = {plan.digest: plan for plan in plans}
    return tuple(unique[key] for key in sorted(unique))


def default_benchmark_policy() -> PolicySpec:
    """Return a small exhaustive policy shared by all benchmark adapters.

    The policy deliberately contains no benchmark oracle facts.  Its finite
    domain models the same claim and hazard templates consumed by the online
    runtime.  Concrete tool/target/flow atoms are grounded instances of these
    templates; the mapping is part of the policy digest.  Source occurrence
    remains distinct from authorization so RuleSound can reject
    ``OccursIn -> AuthorizesUse``.
    """

    support = frozenset({"support-only"})
    positive = frozenset({"support-only", "conflict"})
    not_supported = frozenset({"refute-only", "conflict", "unknown", "overflow"})

    atom_domains = (
        AtomDomain("schema_conforms", EVIDENCE_VALUES),
        AtomDomain("authorized", EVIDENCE_VALUES),
        AtomDomain("hazard_hard", ("support-only", "refute-only")),
        AtomDomain("hazard_confirmation", ("support-only", "refute-only")),
        AtomDomain("hazard_sandbox", ("support-only", "refute-only")),
        AtomDomain("hazard_redaction", ("support-only", "refute-only")),
        AtomDomain("explicit_deny", ("support-only", "refute-only")),
        AtomDomain("occurs_in", ("support-only", "refute-only")),
        AtomDomain("authorizes_use", ("support-only", "refute-only")),
        AtomDomain("invariant_ok", ("support-only", "refute-only")),
    )
    domain_invariants = (
        DomainInvariant(
            "deny_implies_hard_hazard",
            FiniteCondition(all_of=(StatusLiteral("explicit_deny", support),)),
            FiniteCondition(all_of=(StatusLiteral("hazard_hard", support),)),
        ),
        DomainInvariant(
            "reachable_states_preserve_invariant",
            TRUE_CONDITION,
            FiniteCondition(all_of=(StatusLiteral("invariant_ok", support),)),
        ),
    )
    rules = (
        SignedRule(
            "explicit_deny_implies_hazard",
            (SignedAtom("explicit_deny", polarity="+"),),
            SignedAtom("hazard_hard", polarity="+"),
        ),
    )

    required_objects = {
        "tool_schema:registered_tool",
        "side_effect:transfer",
        "external_sink:recipient",
        "state_mutation:ledger",
        "parameter:recipient",
        "parameter:amount",
        "target:recipient",
        "flow:funds_to_recipient",
    }
    action = GuardedActionSpec(
        "benchmark_guarded_action",
        tool_schemas=frozenset({"registered_tool"}),
        side_effects=frozenset({"transfer"}),
        external_sinks=frozenset({"recipient"}),
        state_mutations=frozenset({"ledger"}),
        critical_parameters=frozenset({"recipient", "amount"}),
        targets=frozenset({"recipient"}),
        flows=frozenset({"funds_to_recipient"}),
        requirement_claims=tuple(
            (item, "schema_conforms" if item.startswith("tool_schema:") else "authorized")
            for item in sorted(required_objects)
        ),
    )

    def audit_unit(field: str, value_constraint: str, transform: str = "none") -> AuditUnit:
        return AuditUnit(field, value_constraint, transform, "obligate_audit")

    decision_unit = audit_unit("decision", "public_decision_enum")
    action_unit = audit_unit("action_digest", "sha256", "hash")
    evidence_unit = audit_unit("evidence_digest", "sha256", "hash")
    plan_unit = audit_unit("plan_digest", "sha256", "hash")
    essential_audit = frozenset({decision_unit, action_unit, evidence_unit, plan_unit})
    # Every persisted field is enumerated here.  Raw prompts, arguments,
    # action graphs, scorer labels, evidence payloads, and tool results are
    # intentionally absent; only identifiers, booleans, counters, timings and
    # cryptographic digests are admissible.
    allowed = frozenset(
        {
            *essential_audit,
            audit_unit("event_type", "audit_event_enum"),
            audit_unit("execute", "boolean"),
            audit_unit("reason_codes", "identifier_list", "allowlist"),
            audit_unit("realization_digest", "sha256_or_empty", "hash"),
            audit_unit("policy_acceptance_digest", "sha256_or_empty", "hash"),
            audit_unit("certificate_digest", "sha256_or_empty", "hash"),
            audit_unit("result_digest", "sha256_or_empty", "hash"),
            audit_unit("reason_digest", "sha256_or_empty", "hash"),
            audit_unit("theory_public_decision", "public_decision_or_empty"),
            audit_unit("trigger_ids", "identifier_list", "allowlist"),
            audit_unit("valid_certificate_evidence", "boolean"),
            audit_unit("gap_count", "nonnegative_integer"),
            audit_unit("hazard_count", "nonnegative_integer"),
            audit_unit("overflow_count", "nonnegative_integer"),
            audit_unit("confirmation_required", "boolean"),
            audit_unit("confirmation_executed", "boolean"),
            audit_unit("gateway_confirmation_counted_separately", "boolean"),
            audit_unit("repeated_unsafe_action", "boolean"),
            audit_unit("modules_executed", "identifier_list", "allowlist"),
            audit_unit("modules_skipped", "identifier_list", "allowlist"),
            audit_unit("matched_rules", "identifier_list", "allowlist"),
            audit_unit("matched_invariants", "identifier_list", "allowlist"),
            audit_unit("policy_engines_executed", "identifier_list", "allowlist"),
            audit_unit("policy_engine_finding_ids", "identifier_list", "allowlist"),
            audit_unit("ablation_profile", "identifier", "allowlist"),
            audit_unit("ablation_config", "boolean_map", "allowlist"),
            audit_unit("decision_metadata", "safe_metadata_map", "allowlist"),
            audit_unit("tool_registered", "boolean"),
            audit_unit("policy_ms", "nonnegative_number"),
            audit_unit("gate_state", "gate_state_enum"),
            audit_unit("gate_dispatched", "boolean"),
            audit_unit("outbox_status", "outbox_state_or_empty"),
            audit_unit("sanitized", "boolean"),
            audit_unit("observe_ms", "nonnegative_number"),
        }
    )
    baseline_audit = AuditContract(frozenset({decision_unit}), allowed)
    full_audit = AuditContract(essential_audit, allowed)
    behavior_domain = enumerate_behavior_domain((baseline_audit, full_audit))
    default_plan = BehaviorPlan("host", "allow", "raw", "none", baseline_audit)
    confirm_plan = BehaviorPlan("host", "allow", "raw", "approval_required", full_audit)
    sandbox_plan = BehaviorPlan("sandbox", "allowlist", "redact", "none", full_audit)
    redaction_plan = BehaviorPlan("host", "allow", "no_sensitive", "none", full_audit)
    block_plan = BehaviorPlan("no_execute", "deny", "no_sensitive", "none", full_audit)

    gap_obligation = SemanticObligation(
        "obligation_gap_authorized",
        "gap_authorized",
        "An unresolved authorization claim requires a bound approval.",
        require_approval=True,
        audit_must=essential_audit,
        bad_effect="unauthorized_unconfirmed_dispatch",
    )
    schema_obligation = SemanticObligation(
        "obligation_gap_schema",
        "gap_schema_conforms",
        "An unknown or conflicting tool schema forbids dispatch and confirmation.",
        forbid_dispatch=True,
        forbid_confirmation=True,
        audit_must=essential_audit,
        bad_effect="schema_gap_dispatch",
    )
    hazard_obligation = SemanticObligation(
        "obligation_hard_hazard",
        "hazard_hard",
        "A hard hazard forbids dispatch and confirmation.",
        forbid_dispatch=True,
        forbid_confirmation=True,
        audit_must=essential_audit,
        bad_effect="hard_hazard_dispatch",
    )
    confirmation_obligation = SemanticObligation(
        "obligation_confirmation_hazard",
        "hazard_confirmation",
        "A soft hazard requires a bound approval before dispatch.",
        require_approval=True,
        audit_must=essential_audit,
        bad_effect="confirmation_hazard_unapproved_dispatch",
    )
    sandbox_obligation = SemanticObligation(
        "obligation_sandbox_hazard",
        "hazard_sandbox",
        "The action requires sandbox, network allowlisting, and redaction.",
        min_execution_env="sandbox",
        min_network_scope="allowlist",
        min_data_scope="redact",
        audit_must=essential_audit,
        bad_effect="sandbox_hazard_host_dispatch",
    )
    redaction_obligation = SemanticObligation(
        "obligation_redaction_hazard",
        "hazard_redaction",
        "The action may dispatch only without sensitive data.",
        min_data_scope="no_sensitive",
        audit_must=essential_audit,
        bad_effect="redaction_hazard_sensitive_dispatch",
    )
    overflow_authorized_obligation = SemanticObligation(
        "obligation_overflow_authorized",
        "overflow_authorized",
        "Hard-risk witness overflow fails closed.",
        forbid_dispatch=True,
        forbid_confirmation=True,
        audit_must=essential_audit,
        bad_effect="overflow_dispatch",
    )
    overflow_schema_obligation = SemanticObligation(
        "obligation_overflow_schema",
        "overflow_schema_conforms",
        "Schema witness overflow fails closed.",
        forbid_dispatch=True,
        forbid_confirmation=True,
        audit_must=essential_audit,
        bad_effect="overflow_dispatch",
    )
    deny_obligation = SemanticObligation(
        "obligation_explicit_deny",
        "explicit_deny",
        "Explicit denial forbids dispatch and confirmation.",
        forbid_dispatch=True,
        forbid_confirmation=True,
        audit_must=essential_audit,
        bad_effect="explicit_deny_dispatch",
    )
    triggers = (
        TriggerSpec("gap_authorized", "gap", "authorized", False, (gap_obligation,)),
        TriggerSpec("gap_schema_conforms", "gap", "schema_conforms", True, (schema_obligation,)),
        TriggerSpec("hazard_hard", "hazard", "hazard_hard", True, (hazard_obligation,)),
        TriggerSpec(
            "hazard_confirmation",
            "hazard",
            "hazard_confirmation",
            False,
            (confirmation_obligation,),
            (
                ImplementationConstraint("impl_confirmation", "hazard_confirmation", "confirmation"),
                ImplementationConstraint("impl_confirmation_audit", "hazard_confirmation", "audit"),
            ),
        ),
        TriggerSpec(
            "hazard_sandbox",
            "hazard",
            "hazard_sandbox",
            False,
            (sandbox_obligation,),
            (
                ImplementationConstraint("impl_sandbox", "hazard_sandbox", "sandbox"),
                ImplementationConstraint("impl_sandbox_network", "hazard_sandbox", "network"),
                ImplementationConstraint("impl_sandbox_redaction", "hazard_sandbox", "redaction"),
                ImplementationConstraint("impl_sandbox_audit", "hazard_sandbox", "audit"),
            ),
        ),
        TriggerSpec(
            "hazard_redaction",
            "hazard",
            "hazard_redaction",
            False,
            (redaction_obligation,),
            (
                ImplementationConstraint("impl_redaction", "hazard_redaction", "redaction"),
                ImplementationConstraint("impl_redaction_audit", "hazard_redaction", "audit"),
            ),
        ),
        TriggerSpec("overflow_authorized", "overflow", "authorized", True, (overflow_authorized_obligation,)),
        TriggerSpec("overflow_schema_conforms", "overflow", "schema_conforms", True, (overflow_schema_obligation,)),
        # Explicit deny is a hard Hazard; it is not a fourth Trigger kind.
        TriggerSpec("explicit_deny", "hazard", "explicit_deny", True, (deny_obligation,)),
    )

    remedies = (
        Remedy("remedy_gap_confirm", "gap_authorized", confirm_plan, "trigger:gap_authorized"),
        Remedy("remedy_schema_block", "gap_schema_conforms", block_plan, "trigger:gap_schema_conforms"),
        Remedy("remedy_hazard_block", "hazard_hard", block_plan, "trigger:hazard_hard"),
        Remedy("remedy_confirmation", "hazard_confirmation", confirm_plan, "trigger:hazard_confirmation"),
        Remedy("remedy_sandbox", "hazard_sandbox", sandbox_plan, "trigger:hazard_sandbox"),
        Remedy("remedy_redaction", "hazard_redaction", redaction_plan, "trigger:hazard_redaction"),
        Remedy("remedy_overflow_block", "overflow_authorized", block_plan, "trigger:overflow_authorized"),
        Remedy("remedy_schema_overflow_block", "overflow_schema_conforms", block_plan, "trigger:overflow_schema_conforms"),
        Remedy("remedy_deny_block", "explicit_deny", block_plan, "trigger:explicit_deny"),
    )
    # b_Pi is checked separately from Rem(x), so the declared remedy family
    # remains an antichain while every Trigger still has a fail-stop fallback.
    fallback_remedies = tuple(
        Remedy(
            f"fallback_{trigger.trigger_id}",
            trigger.trigger_id,
            block_plan,
            f"trigger:{trigger.trigger_id}",
        )
        for trigger in triggers
    )

    def realization_for(plan: BehaviorPlan) -> Realization:
        capabilities = {"audit"}
        roots: list[tuple[str, str]] = []
        task_loss = 0
        confirmation_burden = 0
        runtime_cost = 0
        if plan.block_like:
            roots.extend(
                (control, "baseline:fail_stop")
                for control in ("execution_env", "network_scope", "data_scope")
                if getattr(plan, control) != getattr(default_plan, control)
            )
            task_loss += 10
        else:
            capabilities.add("host" if plan.execution_env == "host" else "sandbox")
            if plan.execution_env != default_plan.execution_env:
                roots.append(("execution_env", f"backend:{plan.execution_env}"))
                task_loss += 1
                runtime_cost += 2
            if plan.network_scope != default_plan.network_scope:
                capabilities.add("network")
                roots.append(("network_scope", "backend:network"))
                runtime_cost += 1
            if plan.data_scope != default_plan.data_scope:
                capabilities.add("redaction")
                roots.append(("data_scope", "backend:redaction"))
                task_loss += 1
                runtime_cost += 1
        if plan.human_gate != default_plan.human_gate:
            capabilities.add("confirmation")
            roots.append(("human_gate", "backend:confirmation"))
            confirmation_burden += 1
            runtime_cost += 1
        if plan.audit != default_plan.audit:
            roots.append(("audit", "backend:audit"))
            runtime_cost += 1
        realization_id = (
            "realize_default"
            if plan == default_plan
            else "realize_confirmation"
            if plan == confirm_plan
            else "realize_fail_stop"
            if plan == block_plan
            else f"realize:{plan.digest[:20]}"
        )
        return Realization(
            realization_id=realization_id,
            plan=plan,
            capabilities=frozenset(capabilities),
            actual_audit=allowed,
            control_roots=tuple(roots),
            task_loss=task_loss,
            confirmation_burden=confirmation_burden,
            runtime_cost=runtime_cost,
        )

    realizations = tuple(realization_for(plan) for plan in behavior_domain)

    bad_effects = (
        BadEffectSpec(
            "schema_gap_dispatch",
            FiniteCondition(all_of=(StatusLiteral("schema_conforms", not_supported),)),
            PlanCondition(dispatch=True),
        ),
        BadEffectSpec(
            "unauthorized_unconfirmed_dispatch",
            FiniteCondition(all_of=(StatusLiteral("authorized", not_supported),)),
            PlanCondition(dispatch=True, human_gate=frozenset({"none"})),
        ),
        BadEffectSpec(
            "hard_hazard_dispatch",
            FiniteCondition(all_of=(StatusLiteral("hazard_hard", positive),)),
            PlanCondition(dispatch=True),
        ),
        BadEffectSpec(
            "confirmation_hazard_unapproved_dispatch",
            FiniteCondition(all_of=(StatusLiteral("hazard_confirmation", positive),)),
            PlanCondition(dispatch=True, human_gate=frozenset({"none"})),
        ),
        BadEffectSpec(
            "sandbox_hazard_host_dispatch",
            FiniteCondition(all_of=(StatusLiteral("hazard_sandbox", positive),)),
            PlanCondition(dispatch=True, execution_env=frozenset({"host"})),
        ),
        BadEffectSpec(
            "redaction_hazard_sensitive_dispatch",
            FiniteCondition(all_of=(StatusLiteral("hazard_redaction", positive),)),
            PlanCondition(dispatch=True, data_scope=frozenset({"raw", "redact"})),
        ),
        BadEffectSpec(
            "overflow_dispatch",
            FiniteCondition(
                any_of=(
                    StatusLiteral("authorized", frozenset({"overflow"})),
                    StatusLiteral("schema_conforms", frozenset({"overflow"})),
                )
            ),
            PlanCondition(dispatch=True),
        ),
        BadEffectSpec(
            "explicit_deny_dispatch",
            FiniteCondition(all_of=(StatusLiteral("explicit_deny", positive),)),
            PlanCondition(dispatch=True),
        ),
    )
    confirmation = ConfirmationSpec(REQUIRED_CONFIRMATION_BINDINGS)
    return PolicySpec(
        name="obligate_default_benchmark_policy",
        atom_domains=atom_domains,
        domain_invariants=domain_invariants,
        state_invariants=(
            StateInvariant(
                "policy_state_invariant",
                FiniteCondition(all_of=(StatusLiteral("invariant_ok", support),)),
            ),
        ),
        transitions=(TransitionSpec("identity_step"),),
        rules=rules,
        guarded_actions=(action,),
        hazard_atoms=frozenset({"hazard_hard", "hazard_confirmation", "hazard_sandbox", "hazard_redaction"}),
        overflow_atoms=frozenset({"authorized", "schema_conforms"}),
        explicit_deny_atoms=frozenset({"explicit_deny"}),
        triggers=triggers,
        remedies=remedies,
        fallback_remedies=fallback_remedies,
        behavior_domain=behavior_domain,
        default_plan=default_plan,
        realizations=realizations,
        bad_effects=bad_effects,
        selector=SelectorSpec(),
        confirmation=confirmation,
        declared_public_decisions=tuple(
            (
                item.realization_id,
                "block"
                if item.plan.block_like
                else "require_confirmation"
                if item.plan.confirmation
                else "allow"
                if item.plan == default_plan and not item.control_roots
                else "execute_with_constraints",
            )
            for item in realizations
        ),
        runtime_claim_templates=(
            ("schema_conforms", "schema_conforms"),
            ("action_authorized", "authorized"),
            ("target_authorized", "authorized"),
            ("flow_authorized", "authorized"),
            ("privilege_authorized", "authorized"),
        ),
        runtime_hazard_templates=(
            ("hard", "hazard_hard"),
            ("require_confirmation", "hazard_confirmation"),
            ("sandbox", "hazard_sandbox"),
            ("execute_with_constraints", "hazard_sandbox"),
            ("allow_in_sandbox", "hazard_sandbox"),
            ("redact", "hazard_redaction"),
            ("no_sensitive", "hazard_redaction"),
            ("fallback", "hazard_hard"),
        ),
    )


build_default_benchmark_policy = default_benchmark_policy
build_default_policy = default_benchmark_policy


__all__ = [
    "ANY_PLAN",
    "DATA_ORDER",
    "EVIDENCE_VALUES",
    "EXECUTION_ORDER",
    "EvidenceValue",
    "HUMAN_ORDER",
    "NETWORK_ORDER",
    "REQUIRED_CONFIRMATION_BINDINGS",
    "TRUE_CONDITION",
    "AtomDomain",
    "BadEffectSpec",
    "ConfirmationSpec",
    "DomainInvariant",
    "FiniteCondition",
    "GuardedActionSpec",
    "PlanCondition",
    "PolicySpec",
    "SelectorSpec",
    "StateInvariant",
    "StatusLiteral",
    "TransitionSpec",
    "TriggerSpec",
    "active_trigger_specs",
    "build_default_benchmark_policy",
    "build_default_policy",
    "default_benchmark_policy",
    "enumerate_behavior_domain",
    "expected_trigger_keys",
    "invariant_holds",
    "join_behavior_plans",
    "obligation_satisfied",
    "plan_no_weaker_than",
    "positive_present",
    "realization_feasible",
    "select_realization",
    "signed_atom_present",
    "strict_public_projection",
    "utility_frontier",
]
