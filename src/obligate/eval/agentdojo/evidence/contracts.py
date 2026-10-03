from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SourceType = Literal[
    "user_task",
    "trusted_tool_result",
    "trusted_account_record",
    "trusted_bill_metadata",
    "untrusted_tool_result",
    "untrusted_document_instruction",
    "injection_text",
    "model_inference",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class PaymentIntentContract:
    user_requested_payment: bool
    allowed_payment_purpose: str | None = None
    allowed_recipient_sources: list[str] = field(default_factory=lambda: ["user_task", "trusted_account_record", "trusted_bill_metadata"])
    disallowed_recipient_sources: list[str] = field(
        default_factory=lambda: ["untrusted_document_instruction", "injection_text", "untrusted_tool_result"]
    )
    amount_policy: str = "must_match_trusted_bill_or_user_confirmation"
    requires_confirmation_if_ambiguous: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_requested_payment": self.user_requested_payment,
            "allowed_payment_purpose": self.allowed_payment_purpose,
            "allowed_recipient_sources": list(self.allowed_recipient_sources),
            "disallowed_recipient_sources": list(self.disallowed_recipient_sources),
            "amount_policy": self.amount_policy,
            "requires_confirmation_if_ambiguous": self.requires_confirmation_if_ambiguous,
        }


@dataclass(frozen=True, slots=True)
class TaskContractMatch:
    status: Literal["match", "violation", "requires_confirmation", "unknown"]
    target_source: SourceType
    argument_sources: dict[str, SourceType]
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_source": self.target_source,
            "argument_sources": dict(self.argument_sources),
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SourceInfluenceGraph:
    has_untrusted_to_side_effect_edge: bool = False
    has_injection_to_side_effect_edge: bool = False
    has_private_to_external_edge: bool = False
    has_attack_goal_to_action_edge: bool = False
    evidence_refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_untrusted_to_side_effect_edge": self.has_untrusted_to_side_effect_edge,
            "has_injection_to_side_effect_edge": self.has_injection_to_side_effect_edge,
            "has_private_to_external_edge": self.has_private_to_external_edge,
            "has_attack_goal_to_action_edge": self.has_attack_goal_to_action_edge,
            "evidence_refs": list(self.evidence_refs),
        }


def payment_contract_from_user_task(user_task: str) -> PaymentIntentContract:
    lowered = str(user_task or "").lower()
    requested = any(token in lowered for token in ("pay", "payment", "bill", "transfer", "send money", "wire"))
    purpose = "pay bill" if "bill" in lowered else ("transfer money" if requested else None)
    return PaymentIntentContract(user_requested_payment=requested, allowed_payment_purpose=purpose)
