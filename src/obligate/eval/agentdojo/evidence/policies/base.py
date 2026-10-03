from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

PolicyDecision = Literal["allow", "require_confirmation", "block", "quarantine", "allow_in_sandbox"]


@dataclass(frozen=True, slots=True)
class PolicyFinding:
    engine: str
    decision: PolicyDecision
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    unsafe_parameters: list[str] = field(default_factory=list)
    trusted_parameters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "unsafe_parameters": list(self.unsafe_parameters),
            "trusted_parameters": list(self.trusted_parameters),
            "metadata": dict(self.metadata),
        }


class AgentDojoPolicyEngine(Protocol):
    name: str
    suites: set[str]

    def evaluate(self, *, facts: dict[str, Any], graph: dict[str, Any], context: Any) -> list[PolicyFinding]:
        ...
