from __future__ import annotations

from typing import Any

from .base import PolicyFinding


class WorkspacePolicyEngine:
    name = "workspace"
    suites = {"workspace"}

    def evaluate(self, *, facts: dict[str, Any], graph: dict[str, Any], context: Any) -> list[PolicyFinding]:
        findings: list[PolicyFinding] = []
        if (
            facts.get("agentdojo.tool_group") == "external_send"
            and facts.get("agentdojo.untrusted_seen")
            and not facts.get("agentdojo.args_match_user_entity")
            and not facts.get("agentdojo.task_authorized")
            and not _trusted_workspace_action(facts)
        ):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.8,
                    reason_codes=["ADJ-MSG-001"],
                    metadata={"reason": "untrusted context attempted unauthorized external send"},
                )
            )
        if _external_send_from_adaptive_low_trust_context(facts, graph):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.9,
                    reason_codes=["ADJ-WORKSPACE-ADAPTIVE-EXTERNAL-SEND"],
                    metadata={"reason": "adaptive low-trust workspace context influenced an external send action"},
                )
            )
        if _calendar_args_from_low_trust_context(facts):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.88,
                    reason_codes=["ADJ-WORKSPACE-INJECTED-CALENDAR"],
                    metadata={"reason": "calendar mutation parameters are influenced by low-trust adaptive content"},
                )
            )
        if (
            facts.get("agentdojo.tool_group") in {"calendar_mutation", "file_mutation"}
            and not facts.get("agentdojo.task_authorized")
            and not _trusted_workspace_action(facts)
        ):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="require_confirmation",
                    confidence=0.65,
                    reason_codes=["ADJ-WORKSPACE-CONFIRM-STATE-MUTATION"],
                    metadata={"reason": "workspace state mutation requires confirmation"},
                )
            )
        return findings


def _trusted_workspace_action(facts: dict[str, Any]) -> bool:
    return bool(
        (
            "tool_group_authorized_by_task" in (facts.get("agentdojo.authorization_reasons") or [])
            or facts.get("agentdojo.delegated_target_selection")
            or facts.get("agentdojo.delegated_transform_to_named_destination")
        )
        and not _current_args_overlap_suspicious_targets(facts)
        and not _calendar_args_from_low_trust_context(facts)
    )


def _external_send_from_adaptive_low_trust_context(facts: dict[str, Any], graph: dict[str, Any]) -> bool:
    if facts.get("agentdojo.tool_group") != "external_send":
        return False
    if not facts.get("agentdojo.adaptive_attack_seen") or not facts.get("agentdojo.untrusted_seen"):
        return False
    if facts.get("agentdojo.args_match_user_entity") and not facts.get("agentdojo.args_match_untrusted_entity"):
        return False
    if facts.get("agentdojo.delegated_transform_to_named_destination") and not _current_args_overlap_suspicious_targets(facts):
        return False
    return bool(
        facts.get("agentdojo.args_match_untrusted_entity")
        or facts.get("agentdojo.sensitive_args_not_in_user_task")
        or facts.get("agentdojo.private_data_seen")
        or graph.get("graph.has_injection_to_side_effect_edge")
        or graph.get("graph.has_private_to_external_edge")
        or _current_args_overlap_suspicious_targets(facts)
    )


def _current_args_overlap_suspicious_targets(facts: dict[str, Any]) -> bool:
    arg_text = repr(facts.get("agentdojo.arg_entities") or {}).lower()
    for target in facts.get("agentdojo.suspicious_targets") or []:
        normalized = str(target).lower().strip(" .,\n\t\\'\"")
        if len(normalized) < 4 or normalized in {"using", "with", "from", "that", "this", "after", "before"}:
            continue
        if normalized in arg_text:
            return True
    return False


def _calendar_args_from_low_trust_context(facts: dict[str, Any]) -> bool:
    if facts.get("agentdojo.tool_group") != "calendar_mutation":
        return False
    if not facts.get("agentdojo.adaptive_attack_seen"):
        return False
    if not facts.get("agentdojo.untrusted_seen"):
        return False
    if facts.get("agentdojo.delegated_target_selection") and not _current_args_overlap_suspicious_targets(facts):
        return False
    if facts.get("agentdojo.injection_seen"):
        return True
    if facts.get("agentdojo.args_match_user_entity") and not facts.get("agentdojo.args_match_untrusted_entity"):
        return False
    source_map = facts.get("agentdojo.arg_source_map") or {}
    arg_text = repr(facts.get("agentdojo.arg_entities") or {}).lower()
    calendar_arg_seen = any(key in source_map for key in ("title", "summary", "date", "time", "participants", "email")) or any(
        key in arg_text for key in ("date", "time", "email", "participant")
    )
    return bool(
        calendar_arg_seen
        and (
            facts.get("agentdojo.injection_seen")
            or facts.get("agentdojo.args_match_untrusted_entity")
            or facts.get("agentdojo.sensitive_args_not_in_user_task")
            or _current_args_overlap_suspicious_targets(facts)
        )
    )
