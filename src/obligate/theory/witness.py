"""Bipolar minimal-witness closure for the finite signed-rule fragment."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal, Mapping

from .model import EvidenceAssertion, SignedAtom, SignedRule

BipolarValue = Literal["support-only", "refute-only", "conflict", "unknown"]
Witness = frozenset[str]


def _witness_sort_key(value: Witness) -> tuple[int, tuple[str, ...]]:
    return (len(value), tuple(sorted(value)))


def minimal_antichain(values: Iterable[Witness]) -> tuple[Witness, ...]:
    """Return Min_subset(values) in deterministic order."""

    result: list[Witness] = []
    for candidate in sorted(set(values), key=_witness_sort_key):
        if any(existing <= candidate for existing in result):
            continue
        result = [existing for existing in result if not candidate < existing]
        result.append(candidate)
    return tuple(sorted(result, key=_witness_sort_key))


@dataclass(frozen=True, slots=True)
class BipolarWitness:
    support: tuple[Witness, ...]
    refute: tuple[Witness, ...]

    @property
    def value(self) -> BipolarValue:
        if self.support and self.refute:
            return "conflict"
        if self.support:
            return "support-only"
        if self.refute:
            return "refute-only"
        return "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "support": [sorted(item) for item in self.support],
            "refute": [sorted(item) for item in self.refute],
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class WitnessClosure:
    witnesses: Mapping[str, tuple[Witness, ...]]
    overflow: frozenset[str]
    overflow_paths: Mapping[str, tuple[str, ...]]
    rule_firings: tuple[dict[str, object], ...]
    exact: bool
    witness_limit: int | None

    def for_atom(self, atom: SignedAtom) -> tuple[Witness, ...]:
        return tuple(self.witnesses.get(atom.key, ()))

    def bipolar(self, predicate: str, arguments: tuple[str, ...] = ()) -> BipolarWitness:
        positive = SignedAtom(predicate, arguments, "+")
        negative = SignedAtom(predicate, arguments, "-")
        return BipolarWitness(self.for_atom(positive), self.for_atom(negative))

    def value(self, predicate: str, arguments: tuple[str, ...] = ()) -> BipolarValue:
        return self.bipolar(predicate, arguments).value

    def overflow_affects(self, atom: SignedAtom | str) -> bool:
        if isinstance(atom, SignedAtom):
            return atom.key in self.overflow or atom.unsigned_key in {item[1:] for item in self.overflow}
        return any(item == atom or item[1:] == atom for item in self.overflow)

    def to_dict(self) -> dict[str, object]:
        grouped: dict[str, dict[str, object]] = {}
        unsigned = sorted({key[1:] for key in self.witnesses} | {key[1:] for key in self.overflow})
        for key in unsigned:
            grouped[key] = self.bipolar_from_key(key).to_dict()
        return {
            "witnesses": grouped,
            "signed_antichains": {key: [sorted(item) for item in value] for key, value in sorted(self.witnesses.items())},
            "overflow": sorted(self.overflow),
            "overflow_paths": {key: list(value) for key, value in sorted(self.overflow_paths.items())},
            "rule_firings": list(self.rule_firings),
            "exact": self.exact,
            "witness_limit": self.witness_limit,
        }

    def bipolar_from_key(self, unsigned_key: str) -> BipolarWitness:
        return BipolarWitness(
            tuple(self.witnesses.get(f"+{unsigned_key}", ())),
            tuple(self.witnesses.get(f"-{unsigned_key}", ())),
        )


class BipolarWitnessEngine:
    """Fair work-list closure over grounded signed Horn rules.

    Exact mode is selected with ``witness_limit=None`` and no emergency firing
    budget (the default).  Bounded mode retains at most K non-dominated
    witnesses per signed atom.  Every budget deletion marks Overflow
    immediately and propagates it through the signed rule dependency graph
    before a decision can be returned.
    """

    def __init__(self, *, witness_limit: int | None = None, max_rule_firings: int | None = None) -> None:
        if witness_limit is not None and witness_limit < 1:
            raise ValueError("witness_limit must be positive or None")
        if max_rule_firings is not None and max_rule_firings < 1:
            raise ValueError("max_rule_firings must be positive")
        self.witness_limit = witness_limit
        self.max_rule_firings = max_rule_firings

    def close(self, assertions: Iterable[EvidenceAssertion], rules: Iterable[SignedRule]) -> WitnessClosure:
        rule_list = tuple(sorted(rules, key=lambda item: item.rule_id))
        by_premise: dict[str, list[SignedRule]] = defaultdict(list)
        dependencies: dict[str, set[str]] = defaultdict(set)
        for rule in rule_list:
            for premise in rule.premises:
                by_premise[premise.key].append(rule)
                dependencies[premise.key].add(rule.head.key)

        antichains: dict[str, tuple[Witness, ...]] = {}
        overflow_roots: set[str] = set()
        overflow_paths: dict[str, tuple[str, ...]] = {}
        worklist: deque[str] = deque()
        queued: set[str] = set()

        def enqueue(key: str) -> None:
            if key not in queued:
                queued.add(key)
                worklist.append(key)

        for assertion in sorted(assertions, key=lambda item: (item.atom.key, item.evidence_ref)):
            changed, dropped = self._insert(antichains, assertion.atom.key, frozenset({assertion.evidence_ref}))
            if dropped:
                overflow_roots.add(assertion.atom.key)
            if changed:
                enqueue(assertion.atom.key)

        firings: list[dict[str, object]] = []
        firing_count = 0
        firing_budget_exhausted = False
        while worklist:
            changed_key = worklist.popleft()
            queued.discard(changed_key)
            for rule in by_premise.get(changed_key, ()):  # fair FIFO over every dependent rule
                premise_sets = [antichains.get(item.key, ()) for item in rule.premises]
                if any(not values for values in premise_sets):
                    continue
                for parts in product(*premise_sets):
                    firing_count += 1
                    if self.max_rule_firings is not None and firing_count > self.max_rule_firings:
                        # The emergency budget makes every rule head whose
                        # closure may still be incomplete unsafe to query.
                        # Mark them all instead of pretending unrelated
                        # work-list branches were closed exactly.
                        overflow_roots.update(item.head.key for item in rule_list)
                        firing_budget_exhausted = True
                        worklist.clear()
                        break
                    candidate = frozenset().union(*parts)
                    changed, dropped = self._insert(antichains, rule.head.key, candidate)
                    if dropped:
                        overflow_roots.add(rule.head.key)
                    if changed:
                        enqueue(rule.head.key)
                        firings.append(
                            {
                                "rule_id": rule.rule_id,
                                "head": rule.head.key,
                                "witness": sorted(candidate),
                                "premises": [item.key for item in rule.premises],
                            }
                        )
                if firing_budget_exhausted:
                    break

        overflow = set(overflow_roots)
        queue: deque[str] = deque(sorted(overflow_roots))
        for root in sorted(overflow_roots):
            overflow_paths[root] = (root,)
        while queue:
            current = queue.popleft()
            for downstream in sorted(dependencies.get(current, ())):
                if downstream in overflow:
                    continue
                overflow.add(downstream)
                overflow_paths[downstream] = (*overflow_paths[current], downstream)
                queue.append(downstream)

        all_keys = {item.atom.key for item in assertions}
        for rule in rule_list:
            all_keys.add(rule.head.key)
            all_keys.update(item.key for item in rule.premises)
        frozen = {key: tuple(antichains.get(key, ())) for key in sorted(all_keys)}
        return WitnessClosure(
            witnesses=frozen,
            overflow=frozenset(overflow),
            overflow_paths=overflow_paths,
            rule_firings=tuple(firings),
            exact=self.witness_limit is None and not firing_budget_exhausted,
            witness_limit=self.witness_limit,
        )

    def _insert(self, antichains: dict[str, tuple[Witness, ...]], key: str, candidate: Witness) -> tuple[bool, bool]:
        before = tuple(antichains.get(key, ()))
        combined = minimal_antichain((*before, candidate))
        if combined == before:
            return False, False
        dropped_for_budget = False
        if self.witness_limit is not None and len(combined) > self.witness_limit:
            dropped_for_budget = True
            combined = combined[: self.witness_limit]
        antichains[key] = combined
        return combined != before, dropped_for_budget


__all__ = [
    "BipolarValue",
    "BipolarWitness",
    "BipolarWitnessEngine",
    "Witness",
    "WitnessClosure",
    "minimal_antichain",
]
