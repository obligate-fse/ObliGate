"""Exhaustive finite checker for an ObliGate :class:`PolicySpec`.

Every result is computed from the policy's finite fact and behavior domains.
The checker never consumes a policy-supplied acceptance bit.  Rejection
records contain replayable valuations, plans, rules, Trigger compilations or
realizations, and acceptance records bind all eighteen verification
conditions plus the complete enumeration statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Mapping

from .model import (
    EvidenceAssertion,
    PolicyAcceptanceRecord,
    SignedAtom,
    canonical_json,
    exec_semantics,
    is_authorization_predicate,
    sha256_json,
)
from .policy import (
    REQUIRED_CONFIRMATION_BINDINGS,
    EvidenceValue,
    FiniteCondition,
    PolicySpec,
    TriggerSpec,
    active_trigger_specs,
    build_default_policy,
    expected_trigger_keys,
    invariant_holds,
    join_behavior_plans,
    obligation_satisfied,
    plan_no_weaker_than,
    realization_feasible,
    select_realization,
    signed_atom_present,
    strict_public_projection,
)
from .witness import BipolarWitnessEngine

CHECKER_VERSION = "obligate-theory-finite-checker/1"

VC_NAMES: tuple[tuple[str, str], ...] = (
    ("VC-1", "DomainComplete/Closed"),
    ("VC-2", "RuleSound"),
    ("VC-3", "ReqCoverage"),
    ("VC-4", "CertificateAdequacy"),
    ("VC-5", "TriggerExactness"),
    ("VC-6", "Compile/RemedyTotal"),
    ("VC-7", "NonTriviality"),
    ("VC-8", "BadEffectNonEmpty"),
    ("VC-9", "LocalRemedySound"),
    ("VC-10", "CompositionPreserve"),
    ("VC-11", "HardRiskBlockOnly"),
    ("VC-12", "JoinLUB"),
    ("VC-13", "AuditBounds"),
    ("VC-14", "SelectorNonBlocking"),
    ("VC-15", "Fallback"),
    ("VC-16", "ProjectionStrict"),
    ("VC-17", "ConfirmationBinding"),
    ("VC-18", "ExplanationRoots"),
)


@dataclass(frozen=True, slots=True)
class ValuationRecord:
    valuation_id: str
    values: Mapping[str, EvidenceValue]

    def to_dict(self) -> dict[str, Any]:
        return {"valuation_id": self.valuation_id, "values": dict(self.values)}


@dataclass(frozen=True, slots=True)
class VCResult:
    vc_id: str
    name: str
    passed: bool
    checked_cases: int
    counterexamples: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.vc_id,
            "name": self.name,
            "passed": self.passed,
            "checked_cases": self.checked_cases,
            "counterexamples": [dict(item) for item in self.counterexamples],
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class CheckReport:
    policy_name: str
    policy_digest: str
    checker_version: str
    semantic_version: str
    checks: tuple[VCResult, ...]
    domain_stats: Mapping[str, Any]
    acceptance_record: PolicyAcceptanceRecord

    @property
    def accepted(self) -> bool:
        return all(item.passed for item in self.checks) and len(self.checks) == len(VC_NAMES)

    @property
    def acceptance(self) -> PolicyAcceptanceRecord:
        """Stable shorthand used by ToolGate/ToolFirewall integrations."""

        return self.acceptance_record

    @property
    def status(self) -> str:
        return "accepted" if self.accepted else "rejected"

    @property
    def failures(self) -> tuple[VCResult, ...]:
        return tuple(item for item in self.checks if not item.passed)

    def check(self, vc_id: str) -> VCResult:
        return next(item for item in self.checks if item.vc_id == vc_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "policy_name": self.policy_name,
            "policy_digest": self.policy_digest,
            "checker_version": self.checker_version,
            "semantic_version": self.semantic_version,
            "checks": [item.to_dict() for item in self.checks],
            "domain_stats": dict(self.domain_stats),
            "acceptance_record": self.acceptance_record.to_dict(),
            "acceptance_record_digest": self.acceptance_record.digest,
        }


class FinitePolicyChecker:
    """Discharge VC-1 through VC-18 by complete finite enumeration."""

    def __init__(self, *, max_counterexamples: int = 20) -> None:
        if max_counterexamples < 1:
            raise ValueError("max_counterexamples must be positive")
        self.max_counterexamples = max_counterexamples

    def check(self, policy: PolicySpec) -> CheckReport:
        valuations, domain_stats, enumeration_errors = self._enumerate(policy)
        checks = (
            self._vc1_domain_complete_closed(policy, valuations, domain_stats, enumeration_errors),
            self._vc2_rule_sound(policy, valuations),
            self._vc3_req_coverage(policy),
            self._vc4_certificate_adequacy(policy, valuations),
            self._vc5_trigger_exactness(policy, valuations),
            self._vc6_compile_remedy_total(policy),
            self._vc7_non_triviality(policy),
            self._vc8_bad_effect_non_empty(policy, valuations),
            self._vc9_local_remedy_sound(policy, valuations),
            self._vc10_composition_preserve(policy, valuations),
            self._vc11_hard_risk_block_only(policy),
            self._vc12_join_lub(policy),
            self._vc13_audit_bounds(policy),
            self._vc14_selector_non_blocking(policy, valuations),
            self._vc15_fallback(policy, valuations),
            self._vc16_projection_strict(policy),
            self._vc17_confirmation_binding(policy),
            self._vc18_explanation_roots(policy),
        )
        counterexamples = {
            item.vc_id: [dict(value) for value in item.counterexamples]
            for item in checks
            if item.counterexamples
        }
        record_stats = {**domain_stats, "counterexamples": counterexamples}
        record = PolicyAcceptanceRecord(
            policy_digest=policy.digest,
            checker_version=CHECKER_VERSION,
            semantic_version=policy.semantic_version,
            vc_results=tuple((item.vc_id, item.passed) for item in checks),
            domain_stats=record_stats,
        )
        return CheckReport(
            policy_name=policy.name,
            policy_digest=policy.digest,
            checker_version=CHECKER_VERSION,
            semantic_version=policy.semantic_version,
            checks=checks,
            domain_stats=domain_stats,
            acceptance_record=record,
        )

    def _enumerate(
        self, policy: PolicySpec
    ) -> tuple[tuple[ValuationRecord, ...], dict[str, Any], tuple[dict[str, Any], ...]]:
        errors: list[dict[str, Any]] = []
        domains = tuple(sorted(policy.atom_domains, key=lambda item: item.atom))
        atom_names = [item.atom for item in domains]
        duplicates = sorted({name for name in atom_names if atom_names.count(name) > 1})
        if duplicates:
            errors.append({"reason": "duplicate atom domains", "atoms": duplicates})

        theoretical = policy.theoretical_cartesian_product
        complete = theoretical <= policy.max_cartesian_product and not duplicates
        legal: list[ValuationRecord] = []
        excluded = 0
        if complete:
            for values in product(*(item.values for item in domains)):
                valuation = dict(zip(atom_names, values))
                if not all(item.allows(valuation) for item in policy.domain_invariants):
                    excluded += 1
                    continue
                valuation_id = "valuation:" + sha256_json(valuation)
                legal.append(ValuationRecord(valuation_id, valuation))
        else:
            errors.append(
                {
                    "reason": "cartesian product cannot be exhaustively generated under the policy limit",
                    "theoretical_cartesian_product": theoretical,
                    "max_cartesian_product": policy.max_cartesian_product,
                }
            )

        legal.sort(key=lambda item: item.valuation_id)
        valuation_ids = [item.valuation_id for item in legal]
        enumerator_payload = [item.to_dict() for item in legal]
        enumerator_hash = sha256_json(enumerator_payload)
        behavior_ids = [sha256_json(item.to_dict()) for item in policy.behavior_domain]
        stats: dict[str, Any] = {
            "grounded_atom_count": len(domains),
            "theoretical_cartesian_product": theoretical,
            "excluded_by_domain_invariants": excluded,
            "legal_valuation_count": len(legal),
            "valuation_ids": valuation_ids,
            "enumerator_hash": enumerator_hash,
            "enumerator_version": "obligate-theory-lexicographic-enumerator/1",
            "complete": complete,
            "behavior_domain_count": len(policy.behavior_domain),
            "behavior_domain_hash": sha256_json(behavior_ids),
            "realization_count": len(policy.realizations),
            "trigger_count": len(policy.triggers),
            "rule_count": len(policy.rules),
        }
        return tuple(legal), stats, tuple(errors)

    def _vc1_domain_complete_closed(
        self,
        policy: PolicySpec,
        valuations: tuple[ValuationRecord, ...],
        stats: Mapping[str, Any],
        enumeration_errors: tuple[dict[str, Any], ...],
    ) -> VCResult:
        counterexamples = list(enumeration_errors)
        checked = int(stats["theoretical_cartesian_product"])
        domain_map = policy.atom_domain_map
        if not valuations:
            self._append(counterexamples, {"reason": "legal valuation set is empty"})

        referenced = self._referenced_atoms(policy)
        for atom in sorted(referenced - set(domain_map)):
            self._append(counterexamples, {"reason": "referenced atom has no finite domain", "atom": atom})

        legal_values = {canonical_json(item.values) for item in valuations}
        behavior_values = {canonical_json(item.to_dict()) for item in policy.behavior_domain}
        if len(behavior_values) != len(policy.behavior_domain):
            self._append(counterexamples, {"reason": "behavior domain contains duplicate points"})
        if canonical_json(policy.default_plan.to_dict()) not in behavior_values:
            self._append(counterexamples, {"reason": "default plan is outside the declared behavior domain"})

        for transition in policy.transitions:
            for atom, value in transition.updates:
                if atom not in domain_map or value not in domain_map.get(atom, _EMPTY_ATOM_DOMAIN).values:
                    self._append(
                        counterexamples,
                        {
                            "reason": "transition update is outside the finite domain",
                            "transition": transition.transition_id,
                            "atom": atom,
                            "value": value,
                        },
                    )
            for record in valuations:
                for plan in policy.behavior_domain:
                    if not transition.applies(record.values, plan):
                        continue
                    checked += 1
                    successor = transition.successor(record.values)
                    if canonical_json(successor) not in legal_values:
                        self._append(
                            counterexamples,
                            {
                                "reason": "abstract successor is outside the complete legal domain",
                                "transition": transition.transition_id,
                                "valuation_id": record.valuation_id,
                                "valuation": dict(record.values),
                                "plan": plan.to_dict(),
                                "successor": successor,
                            },
                        )

        return self._result(
            "VC-1",
            checked,
            counterexamples,
            details={
                "complete": stats["complete"],
                "valuation_ids": stats["valuation_ids"],
                "enumerator_hash": stats["enumerator_hash"],
            },
        )

    def _vc2_rule_sound(self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        engine = BipolarWitnessEngine()
        for rule in policy.rules:
            if (
                rule.head.polarity == "+"
                and is_authorization_predicate(rule.head.predicate)
                and not rule.trusted_for_authorization
            ):
                self._append(
                    counterexamples,
                    {"reason": "authorization-producing rule is not marked trusted", "rule": rule.to_dict()},
                )
        for record in valuations:
            assertions = self._assertions(record)
            closure = engine.close(assertions, policy.rules)
            for rule in policy.rules:
                checked += 1
                if all(signed_atom_present(record.values, item) for item in rule.premises) and not signed_atom_present(
                    record.values, rule.head
                ):
                    self._append(
                        counterexamples,
                        {
                            "reason": "rule premises hold in the independent semantic valuation but the head does not",
                            "valuation_id": record.valuation_id,
                            "valuation": dict(record.values),
                            "rule": rule.to_dict(),
                        },
                    )
            for signed_key, witnesses in closure.witnesses.items():
                if not witnesses:
                    continue
                atom = self._parse_signed_key(signed_key)
                if not signed_atom_present(record.values, atom):
                    self._append(
                        counterexamples,
                        {
                            "reason": "signed-rule closure derives a fact absent from the independent semantic table",
                            "valuation_id": record.valuation_id,
                            "valuation": dict(record.values),
                            "derived_atom": signed_key,
                            "witnesses": [sorted(item) for item in witnesses],
                        },
                    )
        return self._result(
            "VC-2",
            checked,
            counterexamples,
            details={"rules": len(policy.rules), "witness_engine": "exact"},
        )

    def _vc3_req_coverage(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        domain_atoms = set(policy.atom_domain_map)
        guarded_seen = False
        for action in policy.guarded_actions:
            checked += 1
            if not action.guarded:
                continue
            guarded_seen = True
            missing = action.required_objects - set(action.claim_map)
            if missing:
                self._append(
                    counterexamples,
                    {"action_type": action.action_type, "reason": "security-relevant objects lack certificate claims", "missing": sorted(missing)},
                )
            for obj, claim in action.requirement_claims:
                if claim not in domain_atoms:
                    self._append(
                        counterexamples,
                        {"action_type": action.action_type, "reason": "requirement claim has no finite atom domain", "object": obj, "claim": claim},
                    )
        if not guarded_seen:
            self._append(counterexamples, {"reason": "policy declares no Guarded action schema"})
        runtime_claims = policy.runtime_claim_map
        required_runtime_predicates = {
            "schema_conforms",
            "action_authorized",
            "target_authorized",
            "flow_authorized",
            "privilege_authorized",
        }
        missing_runtime = required_runtime_predicates - set(runtime_claims)
        if missing_runtime:
            self._append(
                counterexamples,
                {"reason": "runtime requirement predicate lacks a checked finite template", "missing": sorted(missing_runtime)},
            )
        for runtime_predicate, template in sorted(runtime_claims.items()):
            checked += 1
            if template not in policy.required_claims:
                self._append(
                    counterexamples,
                    {
                        "reason": "runtime requirement maps outside checked Req templates",
                        "runtime_predicate": runtime_predicate,
                        "template": template,
                    },
                )
        return self._result("VC-3", checked, counterexamples)

    def _vc4_certificate_adequacy(
        self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]
    ) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        valid_witnesses = 0
        transition_witnesses = 0
        legal_values = {canonical_json(item.values) for item in valuations}
        for record in valuations:
            valid_certificate = not expected_trigger_keys(policy, record.values)
            if not valid_certificate or not invariant_holds(policy, record.values):
                continue
            valid_witnesses += 1
            for transition in policy.transitions:
                if not transition.applies(record.values, policy.default_plan):
                    continue
                transition_witnesses += 1
                checked += 1
                successor = transition.successor(record.values)
                if canonical_json(successor) not in legal_values or not invariant_holds(policy, successor):
                    self._append(
                        counterexamples,
                        {
                            "reason": "a default abstract step under a valid certificate violates I_Pi",
                            "valuation_id": record.valuation_id,
                            "valuation": dict(record.values),
                            "transition": transition.transition_id,
                            "successor": successor,
                        },
                    )
        if valid_witnesses == 0:
            self._append(counterexamples, {"reason": "ValidCert has no witness in the complete finite domain"})
        if transition_witnesses == 0:
            self._append(counterexamples, {"reason": "ExecSem has no default-step witness under ValidCert"})
        return self._result(
            "VC-4",
            checked,
            counterexamples,
            details={"valid_certificate_valuations": valid_witnesses, "default_transition_witnesses": transition_witnesses},
        )

    def _vc5_trigger_exactness(self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        declared_keys = [item.key for item in policy.triggers]
        duplicates = sorted({key for key in declared_keys if declared_keys.count(key) > 1})
        if duplicates:
            self._append(counterexamples, {"reason": "duplicate Trigger keys", "keys": [list(item) for item in duplicates]})
        universe = self._trigger_universe(policy)
        for extra in sorted(set(declared_keys) - universe):
            self._append(counterexamples, {"reason": "Trigger is outside Gap/Hazard/Overflow universe", "trigger_key": list(extra)})
        for record in valuations:
            checked += 1
            expected = expected_trigger_keys(policy, record.values)
            actual = frozenset(item.key for item in active_trigger_specs(policy, record.values))
            if actual != expected:
                self._append(
                    counterexamples,
                    {
                        "reason": "Trig != Gap union Hazard union Overflow",
                        "valuation_id": record.valuation_id,
                        "valuation": dict(record.values),
                        "expected": [list(item) for item in sorted(expected)],
                        "actual": [list(item) for item in sorted(actual)],
                    },
                )
        return self._result("VC-5", checked, counterexamples, details={"trigger_universe": [list(item) for item in sorted(universe)]})

    def _vc6_compile_remedy_total(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        trigger_by_key: dict[tuple[str, str], list[TriggerSpec]] = {}
        for trigger in policy.triggers:
            trigger_by_key.setdefault(trigger.key, []).append(trigger)
        remedies_by_trigger: dict[str, list[Any]] = {}
        for remedy in policy.remedies:
            remedies_by_trigger.setdefault(remedy.trigger_id, []).append(remedy)
        fallbacks_by_trigger: dict[str, list[Any]] = {}
        for remedy in policy.fallback_remedies:
            fallbacks_by_trigger.setdefault(remedy.trigger_id, []).append(remedy)
        for key in sorted(self._trigger_universe(policy)):
            checked += 1
            matches = trigger_by_key.get(key, [])
            if len(matches) != 1:
                self._append(
                    counterexamples,
                    {"reason": "Trigger compilation is missing or ambiguous", "trigger_key": list(key), "count": len(matches)},
                )
                continue
            trigger = matches[0]
            if not trigger.semantic_obligations:
                self._append(counterexamples, {"reason": "Compile^sem is empty", "trigger": trigger.trigger_id})
            for obligation in trigger.semantic_obligations:
                if obligation.trigger_id != trigger.trigger_id:
                    self._append(
                        counterexamples,
                        {"reason": "semantic obligation is bound to a different Trigger", "trigger": trigger.trigger_id, "obligation": obligation.to_dict()},
                    )
            for constraint in trigger.implementation_constraints:
                if constraint.trigger_id != trigger.trigger_id:
                    self._append(
                        counterexamples,
                        {"reason": "implementation constraint is bound to a different Trigger", "trigger": trigger.trigger_id, "constraint": constraint.to_dict()},
                    )
            remedies = remedies_by_trigger.get(trigger.trigger_id, [])
            if not remedies:
                self._append(counterexamples, {"reason": "Rem(x) is empty", "trigger": trigger.trigger_id})
            for index, left in enumerate(remedies):
                for right in remedies[index + 1 :]:
                    checked += 1
                    if plan_no_weaker_than(left.plan, right.plan) or plan_no_weaker_than(right.plan, left.plan):
                        self._append(
                            counterexamples,
                            {
                                "reason": "Rem(x) is not an antichain",
                                "trigger": trigger.trigger_id,
                                "left": left.to_dict(),
                                "right": right.to_dict(),
                            },
                        )
            fallbacks = fallbacks_by_trigger.get(trigger.trigger_id, [])
            if not fallbacks:
                self._append(counterexamples, {"reason": "b_Pi fallback is missing", "trigger": trigger.trigger_id})
            elif any(not item.plan.block_like or item.plan.confirmation for item in fallbacks):
                self._append(counterexamples, {"reason": "b_Pi fallback is not fail-stop", "trigger": trigger.trigger_id})
        known_ids = {item.trigger_id for item in policy.triggers}
        for remedy in (*policy.remedies, *policy.fallback_remedies):
            if remedy.trigger_id not in known_ids:
                self._append(counterexamples, {"reason": "remedy references an unknown Trigger", "remedy": remedy.to_dict()})
        required_controls = {"hard", "require_confirmation", "sandbox", "redact", "fallback"}
        hazard_templates = policy.runtime_hazard_map
        missing_controls = required_controls - set(hazard_templates)
        if missing_controls:
            self._append(
                counterexamples,
                {"reason": "runtime hazard control lacks a checked finite template", "missing": sorted(missing_controls)},
            )
        for control, template in sorted(hazard_templates.items()):
            checked += 1
            if ("hazard", template) not in trigger_by_key:
                self._append(
                    counterexamples,
                    {"reason": "runtime hazard control maps outside checked Trigger templates", "control": control, "template": template},
                )
        return self._result("VC-6", checked, counterexamples)

    def _vc7_non_triviality(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        for trigger in policy.triggers:
            checked += 1
            non_trivial = [item.obligation_id for item in trigger.semantic_obligations if not obligation_satisfied(policy.default_plan, item)]
            if not non_trivial:
                self._append(
                    counterexamples,
                    {
                        "reason": "Trigger does not compile any semantic obligation that excludes the default plan",
                        "trigger": trigger.trigger_id,
                        "default_plan": policy.default_plan.to_dict(),
                    },
                )
        return self._result("VC-7", checked, counterexamples)

    def _vc8_bad_effect_non_empty(
        self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]
    ) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        effects = policy.bad_effect_map
        for trigger in policy.triggers:
            for obligation in trigger.semantic_obligations:
                effect = effects.get(obligation.bad_effect)
                if effect is None:
                    self._append(
                        counterexamples,
                        {"reason": "obligation references an undefined bad effect", "trigger": trigger.trigger_id, "obligation": obligation.to_dict()},
                    )
                    continue
                witness = None
                for record in valuations:
                    for plan in policy.behavior_domain:
                        checked += 1
                        if effect.occurs(record.values, plan):
                            witness = {"valuation_id": record.valuation_id, "valuation": dict(record.values), "plan": plan.to_dict()}
                            break
                    if witness:
                        break
                if witness is None:
                    self._append(
                        counterexamples,
                        {"reason": "bad effect has no reachable witness in the complete fact x behavior domain", "effect": effect.effect_id},
                    )
        return self._result("VC-8", checked, counterexamples)

    def _vc9_local_remedy_sound(
        self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]
    ) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        effects = policy.bad_effect_map
        for record in valuations:
            expected = expected_trigger_keys(policy, record.values)
            for trigger in policy.triggers:
                if trigger.key not in expected:
                    continue
                satisfying = 0
                for plan in policy.behavior_domain:
                    if not all(obligation_satisfied(plan, item) for item in trigger.semantic_obligations):
                        continue
                    satisfying += 1
                    checked += 1
                    violated = [
                        item.bad_effect
                        for item in trigger.semantic_obligations
                        if (effect := effects.get(item.bad_effect)) is not None and effect.occurs(record.values, plan)
                    ]
                    if violated:
                        self._append(
                            counterexamples,
                            {
                                "reason": "a plan satisfies Compile^sem(x) but retains Bad_x",
                                "valuation_id": record.valuation_id,
                                "valuation": dict(record.values),
                                "trigger": trigger.trigger_id,
                                "plan": plan.to_dict(),
                                "bad_effects": violated,
                            },
                        )
                if satisfying == 0:
                    self._append(
                        counterexamples,
                        {"reason": "Trigger semantic obligations are unsatisfiable in the behavior domain", "trigger": trigger.trigger_id, "valuation_id": record.valuation_id},
                    )
        return self._result("VC-9", checked, counterexamples)

    def _vc10_composition_preserve(
        self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]
    ) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        transition_witnesses = 0
        for record in valuations:
            if not invariant_holds(policy, record.values):
                continue
            triggers = active_trigger_specs(policy, record.values)
            obligations = tuple(item for trigger in triggers for item in trigger.semantic_obligations)
            for plan in policy.behavior_domain:
                if not all(obligation_satisfied(plan, item) for item in obligations):
                    continue
                for transition in policy.transitions:
                    if not transition.applies(record.values, plan):
                        continue
                    transition_witnesses += 1
                    checked += 1
                    successor = transition.successor(record.values)
                    if not invariant_holds(policy, successor):
                        self._append(
                            counterexamples,
                            {
                                "reason": "the conjunction of active semantic obligations does not preserve I_Pi",
                                "valuation_id": record.valuation_id,
                                "valuation": dict(record.values),
                                "active_triggers": [item.trigger_id for item in triggers],
                                "plan": plan.to_dict(),
                                "transition": transition.transition_id,
                                "successor": successor,
                            },
                        )
        if transition_witnesses == 0:
            self._append(counterexamples, {"reason": "CompositionPreserve is vacuous: no abstract transition was enumerated"})
        return self._result("VC-10", checked, counterexamples, details={"transition_witnesses": transition_witnesses})

    def _vc11_hard_risk_block_only(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        for trigger in policy.triggers:
            if not trigger.hard:
                continue
            satisfying = []
            for plan in policy.behavior_domain:
                if all(obligation_satisfied(plan, item) for item in trigger.semantic_obligations):
                    checked += 1
                    satisfying.append(plan)
                    if not plan.block_like or plan.confirmation:
                        self._append(
                            counterexamples,
                            {"reason": "hard Trigger admits dispatch or confirmation", "trigger": trigger.trigger_id, "plan": plan.to_dict()},
                        )
            for remedy in (*policy.remedies, *policy.fallback_remedies):
                if remedy.trigger_id == trigger.trigger_id and (not remedy.plan.block_like or remedy.plan.confirmation):
                    self._append(
                        counterexamples,
                        {"reason": "hard Trigger remedy is not block-only", "trigger": trigger.trigger_id, "remedy": remedy.to_dict()},
                    )
            if not satisfying:
                self._append(counterexamples, {"reason": "hard Trigger has no satisfiable block plan", "trigger": trigger.trigger_id})
        return self._result("VC-11", checked, counterexamples)

    def _vc12_join_lub(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        domain = tuple(policy.behavior_domain)
        domain_values = {canonical_json(item.to_dict()): item for item in domain}
        for left in domain:
            for right in domain:
                checked += 1
                joined = join_behavior_plans(left, right)
                key = canonical_json(joined.to_dict())
                if key not in domain_values:
                    self._append(
                        counterexamples,
                        {"reason": "coordinate join is not closed in the finite behavior domain", "left": left.to_dict(), "right": right.to_dict(), "join": joined.to_dict()},
                    )
                    continue
                if not (
                    exec_semantics(joined) <= exec_semantics(left)
                    and exec_semantics(joined) <= exec_semantics(right)
                    and plan_no_weaker_than(joined, left)
                    and plan_no_weaker_than(joined, right)
                ):
                    self._append(
                        counterexamples,
                        {
                            "reason": "join is not a common strengthening under ExecSem trace inclusion",
                            "left": left.to_dict(),
                            "right": right.to_dict(),
                            "join": joined.to_dict(),
                        },
                    )
                for candidate in domain:
                    if plan_no_weaker_than(candidate, left) and plan_no_weaker_than(candidate, right) and not plan_no_weaker_than(candidate, joined):
                        self._append(
                            counterexamples,
                            {
                                "reason": "coordinate join is not the least common strengthening",
                                "left": left.to_dict(),
                                "right": right.to_dict(),
                                "join": joined.to_dict(),
                                "smaller_upper_bound": candidate.to_dict(),
                            },
                        )
                        break
        return self._result(
            "VC-12",
            checked,
            counterexamples,
            details={"safety_order": "ExecSem(candidate) subseteq ExecSem(baseline) plus AuditContract order"},
        )

    def _vc13_audit_bounds(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        plans = [
            *policy.behavior_domain,
            *(item.plan for item in policy.remedies),
            *(item.plan for item in policy.fallback_remedies),
            *(item.plan for item in policy.realizations),
        ]
        for plan in plans:
            checked += 1
            if not plan.audit.consistent:
                self._append(counterexamples, {"reason": "AuditContract violates Must subset Allowed", "plan": plan.to_dict()})
        audits = tuple({canonical_json(item.audit.to_dict()): item.audit for item in policy.behavior_domain}.values())
        for left in audits:
            for right in audits:
                checked += 1
                joined = left.join(right)
                if not joined.consistent:
                    self._append(
                        counterexamples,
                        {"reason": "audit join creates Must/Allowed conflict", "left": left.to_dict(), "right": right.to_dict(), "join": joined.to_dict()},
                    )
        for realization in policy.realizations:
            checked += 1
            missing = realization.plan.audit.must - realization.actual_audit
            excess = realization.actual_audit - realization.plan.audit.allowed
            if missing or excess:
                self._append(
                    counterexamples,
                    {
                        "reason": "realization audit is outside MustAudit/AllowedAudit bounds",
                        "realization": realization.to_dict(),
                        "missing": [item.to_dict() for item in sorted(missing)],
                        "excess": [item.to_dict() for item in sorted(excess)],
                    },
                )
        return self._result("VC-13", checked, counterexamples)

    def _vc14_selector_non_blocking(
        self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]
    ) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        for record in valuations:
            checked += 1
            triggers = active_trigger_specs(policy, record.values)
            feasible = tuple(item for item in policy.realizations if realization_feasible(item, triggers))
            non_blocking = tuple(item for item in feasible if not item.plan.block_like)
            selected = select_realization(policy, feasible)
            if non_blocking and (selected is None or selected.plan.block_like):
                self._append(
                    counterexamples,
                    {
                        "reason": "selector chose block/undefined while a safe feasible non-blocking realization exists",
                        "valuation_id": record.valuation_id,
                        "valuation": dict(record.values),
                        "active_triggers": [item.trigger_id for item in triggers],
                        "non_blocking": [item.realization_id for item in non_blocking],
                        "selected": selected.realization_id if selected else None,
                    },
                )
        return self._result("VC-14", checked, counterexamples)

    def _vc15_fallback(self, policy: PolicySpec, valuations: tuple[ValuationRecord, ...]) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        branch_counts = {"non_blocking": 0, "compliant_block": 0, "compliance_error": 0}
        for record in valuations:
            checked += 1
            triggers = active_trigger_specs(policy, record.values)
            feasible = tuple(item for item in policy.realizations if realization_feasible(item, triggers))
            non_blocking = tuple(item for item in feasible if not item.plan.block_like)
            blockers = tuple(item for item in feasible if item.plan.block_like)
            selected = select_realization(policy, feasible)
            if non_blocking:
                branch_counts["non_blocking"] += 1
                if selected not in non_blocking:
                    self._append(counterexamples, {"reason": "fallback bypassed a non-blocking frontier", "valuation_id": record.valuation_id})
            elif blockers:
                branch_counts["compliant_block"] += 1
                if selected not in blockers:
                    self._append(
                        counterexamples,
                        {
                            "reason": "a compliant blocker exists but fallback did not select it",
                            "valuation_id": record.valuation_id,
                            "blockers": [item.realization_id for item in blockers],
                            "selected": selected.realization_id if selected else None,
                        },
                    )
            else:
                branch_counts["compliance_error"] += 1
                if selected is not None and not selected.plan.block_like:
                    self._append(
                        counterexamples,
                        {"reason": "compliance_error branch dispatches the protected tool", "valuation_id": record.valuation_id, "selected": selected.to_dict()},
                    )
        return self._result("VC-15", checked, counterexamples, details=branch_counts)

    def _vc16_projection_strict(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        declared = dict(policy.declared_public_decisions)
        if len(declared) != len(policy.declared_public_decisions):
            self._append(counterexamples, {"reason": "duplicate declared public-decision entries"})
        for realization in policy.realizations:
            checked += 1
            expected = strict_public_projection(policy, realization)
            actual = declared.get(realization.realization_id)
            if actual != expected:
                self._append(
                    counterexamples,
                    {"reason": "public projection hides or mislabels a control", "realization": realization.to_dict(), "expected": expected, "actual": actual},
                )
            if actual == "allow" and (realization.plan != policy.default_plan or realization.control_roots):
                self._append(
                    counterexamples,
                    {"reason": "ordinary allow contains a non-default abstract or concrete control", "realization": realization.to_dict()},
                )
        return self._result("VC-16", checked, counterexamples)

    def _vc17_confirmation_binding(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        confirmation = policy.confirmation
        missing = REQUIRED_CONFIRMATION_BINDINGS - confirmation.bound_fields
        if missing:
            self._append(counterexamples, {"reason": "confirmation token omits required binding fields", "missing": sorted(missing)})
        for name in (
            "one_time",
            "re_adjudicate",
            "atomic_claim_and_outbox",
            "immutable_outbox",
            "no_retry_without_idempotency",
        ):
            if not getattr(confirmation, name):
                self._append(counterexamples, {"reason": "confirmation atomicity property is disabled", "property": name})

        required_edges = {
            ("none", "pending"),
            ("pending", "fresh"),
            ("pending", "closed"),
            ("fresh", "claimed"),
            ("claimed", "sending"),
            ("sending", "done"),
            ("sending", "uncertain"),
        }
        allowed_edges = required_edges
        transitions = set(confirmation.transitions)
        for edge in sorted(required_edges - transitions):
            self._append(counterexamples, {"reason": "confirmation state machine is missing a required transition", "transition": list(edge)})
        for edge in sorted(transitions - allowed_edges):
            self._append(counterexamples, {"reason": "confirmation state machine permits a forbidden transition", "transition": list(edge)})
        if ("claimed", "fresh") in transitions:
            self._append(counterexamples, {"reason": "claimed nonce can return to fresh"})
        terminal = {"done", "closed", "uncertain"}
        for edge in transitions:
            if edge[0] in terminal:
                self._append(counterexamples, {"reason": "terminal confirmation state has an outgoing transition", "transition": list(edge)})
        return self._result("VC-17", len(transitions) + len(confirmation.bound_fields), counterexamples)

    def _vc18_explanation_roots(self, policy: PolicySpec) -> VCResult:
        counterexamples: list[dict[str, Any]] = []
        checked = 0
        trigger_ids = {item.trigger_id for item in policy.triggers}
        for realization in policy.realizations:
            checked += 1
            roots = dict(realization.control_roots)
            required_controls = self._non_default_controls(policy, realization.plan)
            missing = required_controls - set(roots)
            if missing:
                self._append(
                    counterexamples,
                    {"reason": "non-default concrete controls lack explanation roots", "realization": realization.realization_id, "missing": sorted(missing)},
                )
            for control, root in realization.control_roots:
                if root.startswith("trigger:"):
                    trigger_id = root.split(":", 1)[1]
                    if trigger_id not in trigger_ids:
                        self._append(counterexamples, {"reason": "explanation references an unknown Trigger", "realization": realization.realization_id, "control": control, "root": root})
                elif root.startswith("backend:"):
                    capability = root.split(":", 1)[1]
                    if capability not in realization.capabilities:
                        self._append(counterexamples, {"reason": "backend explanation root is not a realization capability", "realization": realization.realization_id, "control": control, "root": root})
                elif root.startswith("baseline:"):
                    if not root.split(":", 1)[1]:
                        self._append(counterexamples, {"reason": "empty baseline explanation root", "realization": realization.realization_id, "control": control})
                else:
                    self._append(counterexamples, {"reason": "explanation root is not Trigger, backend coupling, or Base", "realization": realization.realization_id, "control": control, "root": root})
        for remedy in (*policy.remedies, *policy.fallback_remedies):
            checked += 1
            root = remedy.explanation_root
            if root.startswith("trigger:") and root.split(":", 1)[1] in trigger_ids:
                continue
            if root.startswith("baseline:") and root.split(":", 1)[1]:
                continue
            if root.startswith("backend:") and root.split(":", 1)[1]:
                continue
            self._append(counterexamples, {"reason": "remedy explanation root is invalid", "remedy": remedy.to_dict()})
        return self._result("VC-18", checked, counterexamples)

    def _result(
        self,
        vc_id: str,
        checked_cases: int,
        counterexamples: Iterable[Mapping[str, Any]],
        *,
        details: Mapping[str, Any] | None = None,
    ) -> VCResult:
        name = dict(VC_NAMES)[vc_id]
        values = tuple(dict(item) for item in counterexamples)[: self.max_counterexamples]
        return VCResult(vc_id, name, not values, checked_cases, values, details or {})

    def _append(self, values: list[dict[str, Any]], value: Mapping[str, Any]) -> None:
        if len(values) < self.max_counterexamples:
            values.append(dict(value))

    @staticmethod
    def _trigger_universe(policy: PolicySpec) -> frozenset[tuple[str, str]]:
        return frozenset(
            [("gap", item) for item in policy.required_claims]
            + [("hazard", item) for item in policy.hazard_atoms]
            + [("overflow", item) for item in policy.overflow_atoms]
            + [("hazard", item) for item in policy.explicit_deny_atoms]
        )

    @staticmethod
    def _condition_atoms(condition: FiniteCondition) -> set[str]:
        return {item.atom for item in (*condition.all_of, *condition.any_of, *condition.none_of)}

    def _referenced_atoms(self, policy: PolicySpec) -> set[str]:
        result = set(policy.required_claims) | set(policy.hazard_atoms) | set(policy.overflow_atoms) | set(policy.explicit_deny_atoms)
        for invariant in policy.domain_invariants:
            result.update(self._condition_atoms(invariant.antecedent))
            result.update(self._condition_atoms(invariant.consequent))
        for invariant in policy.state_invariants:
            result.update(self._condition_atoms(invariant.condition))
        for transition in policy.transitions:
            result.update(self._condition_atoms(transition.when))
            result.update(key for key, _ in transition.updates)
        for rule in policy.rules:
            result.add(rule.head.unsigned_key)
            result.update(item.unsigned_key for item in rule.premises)
        for effect in policy.bad_effects:
            result.update(self._condition_atoms(effect.facts))
        return result

    @staticmethod
    def _assertions(record: ValuationRecord) -> tuple[EvidenceAssertion, ...]:
        result = []
        for atom, value in sorted(record.values.items()):
            if value in {"support-only", "conflict"}:
                result.append(EvidenceAssertion(SignedAtom(atom, polarity="+"), f"{record.valuation_id}:{atom}:+", "independent_semantics", True))
            if value in {"refute-only", "conflict"}:
                result.append(EvidenceAssertion(SignedAtom(atom, polarity="-"), f"{record.valuation_id}:{atom}:-", "independent_semantics", True))
        return tuple(result)

    @staticmethod
    def _parse_signed_key(key: str) -> SignedAtom:
        polarity = key[0]
        unsigned = key[1:]
        if "(" not in unsigned:
            return SignedAtom(unsigned, polarity=polarity)  # type: ignore[arg-type]
        predicate, raw = unsigned.split("(", 1)
        arguments = tuple(item for item in raw[:-1].split(",") if item)
        return SignedAtom(predicate, arguments, polarity)  # type: ignore[arg-type]

    @staticmethod
    def _non_default_controls(policy: PolicySpec, plan: Any) -> set[str]:
        result = set()
        for name in ("execution_env", "network_scope", "data_scope", "human_gate"):
            if getattr(plan, name) != getattr(policy.default_plan, name):
                result.add(name)
        if plan.audit != policy.default_plan.audit:
            result.add("audit")
        return result


class PolicyChecker(FinitePolicyChecker):
    """Compatibility alias with the concise public name used in the paper."""


def check_policy(policy: PolicySpec, *, max_counterexamples: int = 20) -> CheckReport:
    return FinitePolicyChecker(max_counterexamples=max_counterexamples).check(policy)


def check_default_policy(*, max_counterexamples: int = 20) -> CheckReport:
    """Build and exhaustively check the benchmark-neutral default policy."""

    return check_policy(build_default_policy(), max_counterexamples=max_counterexamples)


# Used only to avoid a branch when reporting an update to an unknown atom.
@dataclass(frozen=True, slots=True)
class _EmptyAtomDomain:
    values: tuple[EvidenceValue, ...] = ()


_EMPTY_ATOM_DOMAIN = _EmptyAtomDomain()


__all__ = [
    "CHECKER_VERSION",
    "VC_NAMES",
    "CheckReport",
    "FinitePolicyChecker",
    "PolicyChecker",
    "VCResult",
    "ValuationRecord",
    "build_default_policy",
    "check_default_policy",
    "check_policy",
]
