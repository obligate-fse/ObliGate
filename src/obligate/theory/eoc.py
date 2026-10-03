"""EOC Trigger total compilation into semantic and implementation duties."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from .lattice import minimal_plans
from .model import (
    ImplementationConstraint,
    Remedy,
    SemanticObligation,
    SignedAtom,
    Trigger,
)
from .policy import PolicySpec, TriggerSpec
from .witness import WitnessClosure


@dataclass(frozen=True, slots=True)
class HazardSignal:
    signal_id: str
    subject: str
    evidence_refs: tuple[str, ...]
    hard: bool = False
    preferred_control: str = "block"
    kind: str = "hazard"


@dataclass(frozen=True, slots=True)
class EOCResult:
    requirements: tuple[SignedAtom, ...]
    gaps: tuple[str, ...]
    hazards: tuple[str, ...]
    overflow: tuple[str, ...]
    triggers: tuple[Trigger, ...]
    semantic_obligations: tuple[SemanticObligation, ...]
    implementation_constraints: tuple[ImplementationConstraint, ...]
    safe_remedies: Mapping[str, tuple[Remedy, ...]]
    remedies: Mapping[str, tuple[Remedy, ...]]

    @property
    def valid_certificate_evidence(self) -> bool:
        return not self.gaps and not self.hazards and not self.overflow

    def to_dict(self) -> dict[str, object]:
        return {
            "requirements": [item.to_dict() for item in self.requirements],
            "gaps": list(self.gaps),
            "hazards": list(self.hazards),
            "overflow": list(self.overflow),
            "triggers": [item.to_dict() for item in self.triggers],
            "semantic_obligations": [item.to_dict() for item in self.semantic_obligations],
            "implementation_constraints": [item.to_dict() for item in self.implementation_constraints],
            "safe_remedies": {key: [item.to_dict() for item in value] for key, value in sorted(self.safe_remedies.items())},
            "remedies": {key: [item.to_dict() for item in value] for key, value in sorted(self.remedies.items())},
            "valid_certificate_evidence": self.valid_certificate_evidence,
        }


class EOCCompiler:
    """Instantiate only Trigger/remedy templates from a checked PolicySpec."""

    def __init__(self, *, policy: PolicySpec) -> None:
        self.policy = policy
        self._trigger_templates = policy.trigger_map
        self._remedies_by_trigger: dict[str, tuple[Remedy, ...]] = {}
        self._fallbacks_by_trigger: dict[str, tuple[Remedy, ...]] = {}
        for template in policy.triggers:
            self._remedies_by_trigger[template.trigger_id] = tuple(
                item for item in policy.remedies if item.trigger_id == template.trigger_id
            )
            self._fallbacks_by_trigger[template.trigger_id] = tuple(
                item for item in policy.fallback_remedies if item.trigger_id == template.trigger_id
            )

    def compile(
        self,
        *,
        requirements: Sequence[SignedAtom],
        closure: WitnessClosure,
        hazards: Iterable[HazardSignal] = (),
        explicit_denies: Iterable[HazardSignal] = (),
    ) -> EOCResult:
        requirement_tuple = tuple(requirements)
        gaps = tuple(
            item.unsigned_key
            for item in requirement_tuple
            if closure.bipolar(item.predicate, item.arguments).value != "support-only"
        )
        overflow = tuple(
            sorted(
                {
                    item.unsigned_key
                    for item in requirement_tuple
                    if closure.overflow_affects(item)
                }
                | {key[1:] for key in closure.overflow}
            )
        )
        hazard_list = tuple(sorted(hazards, key=lambda item: item.signal_id))
        deny_list = tuple(sorted(explicit_denies, key=lambda item: item.signal_id))

        triggers: list[Trigger] = []
        remedy_map: dict[str, tuple[Remedy, ...]] = {}
        safe_remedy_map: dict[str, tuple[Remedy, ...]] = {}
        for subject in gaps:
            predicate = _predicate_from_subject(subject)
            template_subject = self.policy.runtime_claim_map.get(predicate)
            template_key = (
                ("gap", template_subject)
                if template_subject is not None
                else ("hazard", self.policy.runtime_hazard_map["fallback"])
            )
            trigger, remedies, safe_remedies = self._instantiate(
                template_key,
                subject=subject,
                evidence_refs=_refs_for_subject(subject, closure),
            )
            triggers.append(trigger)
            remedy_map[trigger.trigger_id] = remedies
            safe_remedy_map[trigger.trigger_id] = safe_remedies
        for signal in hazard_list:
            control = "hard" if signal.hard else signal.preferred_control
            template_subject = self.policy.runtime_hazard_map.get(
                control,
                self.policy.runtime_hazard_map["fallback"],
            )
            trigger, remedies, safe_remedies = self._instantiate(
                ("hazard", template_subject),
                subject=signal.subject,
                evidence_refs=signal.evidence_refs,
                identity=signal.signal_id,
            )
            triggers.append(trigger)
            remedy_map[trigger.trigger_id] = remedies
            safe_remedy_map[trigger.trigger_id] = safe_remedies
        for subject in overflow:
            predicate = _predicate_from_subject(subject)
            template_subject = self.policy.runtime_claim_map.get(predicate, "authorized")
            trigger, remedies, safe_remedies = self._instantiate(
                ("overflow", template_subject),
                subject=subject,
                evidence_refs=(
                    tuple(closure.overflow_paths.get(f"+{subject}", ()))
                    + tuple(closure.overflow_paths.get(f"-{subject}", ()))
                    or (f"overflow:{subject}",)
                ),
            )
            triggers.append(trigger)
            remedy_map[trigger.trigger_id] = remedies
            safe_remedy_map[trigger.trigger_id] = safe_remedies
        for signal in deny_list:
            trigger, remedies, safe_remedies = self._instantiate(
                ("hazard", "explicit_deny"),
                subject=signal.subject,
                evidence_refs=signal.evidence_refs,
                identity=signal.signal_id,
            )
            triggers.append(trigger)
            remedy_map[trigger.trigger_id] = remedies
            safe_remedy_map[trigger.trigger_id] = safe_remedies

        # ExplicitDeny is compiled as a hard Hazard, so Trigger remains
        # definitionally Gap U Hazard U Overflow.
        trigger_by_id = {item.trigger_id: item for item in triggers}
        compiled = tuple(trigger_by_id[key] for key in sorted(trigger_by_id))
        obligations = tuple(
            item for trigger in compiled for item in trigger.semantic_obligations
        )
        implementation = tuple(
            item for trigger in compiled for item in trigger.implementation_constraints
        )
        return EOCResult(
            requirements=requirement_tuple,
            gaps=tuple(sorted(set(gaps))),
            hazards=tuple(item.subject for item in hazard_list) + tuple(item.subject for item in deny_list),
            overflow=overflow,
            triggers=compiled,
            semantic_obligations=obligations,
            implementation_constraints=implementation,
            safe_remedies={key: safe_remedy_map[key] for key in sorted(safe_remedy_map)},
            remedies={key: remedy_map[key] for key in sorted(remedy_map)},
        )

    def _instantiate(
        self,
        template_key: tuple[str, str],
        *,
        subject: str,
        evidence_refs: tuple[str, ...],
        identity: str | None = None,
    ) -> tuple[Trigger, tuple[Remedy, ...], tuple[Remedy, ...]]:
        try:
            template: TriggerSpec = self._trigger_templates[template_key]  # type: ignore[index]
        except KeyError as exc:  # loader/checker mismatch must fail closed
            raise RuntimeError(f"checked policy has no Trigger template {template_key!r}") from exc
        trigger_id = _stable_id(template.kind, identity or subject)
        obligations = tuple(
            replace(
                item,
                obligation_id=f"{item.obligation_id}:{trigger_id.split(':')[-1]}",
                trigger_id=trigger_id,
            )
            for item in template.semantic_obligations
        )
        implementation = tuple(
            replace(
                item,
                constraint_id=f"{item.constraint_id}:{trigger_id.split(':')[-1]}",
                trigger_id=trigger_id,
            )
            for item in template.implementation_constraints
        )
        trigger = Trigger(
            trigger_id=trigger_id,
            kind=template.kind,
            subject=subject,
            hard=template.hard,
            evidence_refs=evidence_refs,
            semantic_obligations=obligations,
            implementation_constraints=implementation,
        )
        policy_remedies = self._remedies_by_trigger.get(template.trigger_id, ())
        # Rem(x) is a checker-verified antichain.  b_Pi is represented by the
        # separate PolicySpec fallback_remedies relation.
        frontier_plans = minimal_plans(item.plan for item in policy_remedies)
        if {item.digest for item in frontier_plans} != {
            item.plan.digest for item in policy_remedies
        }:
            raise RuntimeError(
                f"checked policy remedies for {template.trigger_id!r} are not an antichain"
            )
        by_digest = {item.plan.digest: item for item in policy_remedies}
        remedies = tuple(
            Remedy(
                remedy_id=f"{by_digest[plan.digest].remedy_id}:{trigger_id.split(':')[-1]}",
                trigger_id=trigger_id,
                plan=plan,
                explanation_root=by_digest[plan.digest].explanation_root,
            )
            for plan in frontier_plans
        )
        if not remedies:
            raise RuntimeError(f"checked policy Trigger {template.trigger_id!r} has no online remedy")
        policy_fallbacks = self._fallbacks_by_trigger.get(template.trigger_id, ())
        suffix = trigger_id.split(":")[-1]
        fallbacks = tuple(
            Remedy(
                remedy_id=f"{item.remedy_id}:{suffix}",
                trigger_id=trigger_id,
                plan=item.plan,
                explanation_root=item.explanation_root,
            )
            for item in policy_fallbacks
        )
        safe_remedies = tuple(
            {item.plan.digest: item for item in (*remedies, *fallbacks)}[key]
            for key in sorted({item.plan.digest for item in (*remedies, *fallbacks)})
        )
        if not any(item.plan.block_like for item in safe_remedies):
            raise RuntimeError(
                f"checked policy Trigger {template.trigger_id!r} has no safe block in Rem_safe"
            )
        return trigger, remedies, safe_remedies


def _stable_id(kind: str, subject: str) -> str:
    digest = hashlib.sha256(f"{kind}\x00{subject}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


def _predicate_from_subject(subject: str) -> str:
    return subject.split("(", 1)[0]


def _refs_for_subject(subject: str, closure: WitnessClosure) -> tuple[str, ...]:
    witness = closure.bipolar_from_key(subject)
    refs: set[str] = set()
    for item in (*witness.support, *witness.refute):
        refs.update(item)
    return tuple(sorted(refs)) or (f"missing:{subject}",)


__all__ = ["HazardSignal", "EOCCompiler", "EOCResult"]
