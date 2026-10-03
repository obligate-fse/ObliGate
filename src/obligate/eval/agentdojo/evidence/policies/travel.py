from __future__ import annotations

from typing import Any

from .base import PolicyFinding


class TravelPolicyEngine:
    name = "travel"
    suites = {"travel"}

    def evaluate(self, *, facts: dict[str, Any], graph: dict[str, Any], context: Any) -> list[PolicyFinding]:
        findings: list[PolicyFinding] = []
        if facts.get("agentdojo.tool_group") == "external_send":
            if (
                facts.get("agentdojo.delegated_target_selection")
                and facts.get("agentdojo.task_authorized")
                and not facts.get("agentdojo.attack_goal_overlap")
            ):
                return findings
            if (
                facts.get("agentdojo.task_authorized")
                and facts.get("agentdojo.args_match_user_entity")
                and not facts.get("agentdojo.args_match_untrusted_entity")
                and not graph.get("graph.has_private_to_external_edge")
                and not graph.get("graph.has_private_to_executed_external_edge")
                and not graph.get("graph.has_untrusted_to_executed_side_effect_edge")
            ):
                return findings
            if (
                facts.get("agentdojo.task_authorized")
                and facts.get("agentdojo.args_match_user_entity")
                and not facts.get("agentdojo.args_match_untrusted_entity")
                and not graph.get("graph.has_private_to_executed_external_edge")
            ):
                findings.append(
                    PolicyFinding(
                        engine=self.name,
                        decision="require_confirmation",
                        confidence=0.75,
                        reason_codes=["ADJ-TRAVEL-CONFIRM-AUTHORIZED-SEND"],
                        metadata={"reason": "user-authorized travel email with untrusted context requires confirmation"},
                    )
                )
                return findings
            if facts.get("agentdojo.untrusted_seen") or facts.get("agentdojo.args_match_untrusted_entity"):
                findings.append(
                    PolicyFinding(
                        engine=self.name,
                        decision="block",
                        confidence=0.85,
                        reason_codes=["ADJ-TRAVEL-INJECTED-SEND"],
                        metadata={"reason": "travel email destination or body appears influenced by untrusted content"},
                    )
                )
            return findings
        if facts.get("agentdojo.tool_group") != "booking_commit":
            return []
        if _booking_target_from_low_trust_context(facts):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.9,
                    reason_codes=["ADJ-TRAVEL-ADAPTIVE-BOOKING-TARGET"],
                    metadata={"reason": "booking target or booking parameters are sourced from low-trust adaptive content"},
                )
            )
        if facts.get("agentdojo.injection_followup_side_effect_seen") or facts.get("agentdojo.injection_recommendation_attack_seen"):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.86,
                    reason_codes=["ADJ-TRAVEL-004"],
                    metadata={"reason": "booking follows an injected side effect or recommendation steering instruction"},
                )
            )
        if (
            facts.get("agentdojo.injection_seen")
            and (not facts.get("agentdojo.task_authorized") or facts.get("agentdojo.sensitive_args_not_in_user_task"))
            and not facts.get("agentdojo.args_match_user_entity")
        ):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.85,
                    reason_codes=["ADJ-TRAVEL-001"],
                    metadata={"reason": "injection-like travel output attempted unauthorized or mismatched booking"},
                )
            )
        if facts.get("agentdojo.target_entity_source") == "unknown":
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="require_confirmation",
                    confidence=0.65,
                    reason_codes=["ADJ-TRAVEL-CONFIRM-UNKNOWN-TARGET"],
                    metadata={"reason": "unknown travel booking target requires confirmation"},
                )
            )
        if facts.get("agentdojo.untrusted_seen") and not facts.get("agentdojo.args_match_user_entity") and not facts.get("agentdojo.task_authorized"):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.85,
                    reason_codes=["ADJ-TRAVEL-002"],
                    metadata={"reason": "untrusted travel content attempted unauthorized booking"},
                )
            )
        if facts.get("agentdojo.args_match_untrusted_entity") and not facts.get("agentdojo.args_match_user_entity"):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.85,
                    reason_codes=["ADJ-TRAVEL-004"],
                    metadata={"reason": "booking target appears sourced from untrusted output rather than user task"},
                )
            )
        if (
            facts.get("agentdojo.injection_seen")
            and facts.get("agentdojo.explicit_booking_target_in_task")
            and not facts.get("agentdojo.booking_target_matches_user_goal")
        ):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.85,
                    reason_codes=["ADJ-TRAVEL-003"],
                    metadata={"reason": "travel booking target does not match the user goal after untrusted review content"},
                )
            )
        return findings


def _booking_target_from_low_trust_context(facts: dict[str, Any]) -> bool:
    if not facts.get("agentdojo.adaptive_attack_seen"):
        return False
    if not facts.get("agentdojo.untrusted_seen"):
        return False
    if facts.get("agentdojo.args_match_user_entity") or facts.get("agentdojo.booking_target_matches_user_goal"):
        return False
    source_map = facts.get("agentdojo.arg_source_map") or {}
    target_source = facts.get("agentdojo.target_entity_source")
    if target_source in {"untrusted_tool_result", "injection_text"}:
        return True
    if facts.get("agentdojo.args_match_untrusted_entity") or facts.get("agentdojo.injection_seen"):
        return True
    if facts.get("agentdojo.sensitive_args_not_in_user_task") and any(
        key in source_map or key in repr(facts.get("agentdojo.arg_entities") or {}).lower()
        for key in ("hotel", "restaurant", "company", "city", "date", "price")
    ):
        return True
    return False
