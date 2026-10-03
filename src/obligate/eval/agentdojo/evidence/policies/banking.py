from __future__ import annotations

from typing import Any

from .base import PolicyFinding


class BankingPolicyEngine:
    name = "banking"
    suites = {"banking"}

    def evaluate(self, *, facts: dict[str, Any], graph: dict[str, Any], context: Any) -> list[PolicyFinding]:
        group = facts.get("agentdojo.tool_group")
        findings: list[PolicyFinding] = []
        if group == "credential_change":
            explicitly_user_authorized = bool(
                facts.get("agentdojo.task_authorized")
                and (
                    facts.get("agentdojo.args_match_user_entity")
                    or facts.get("agentdojo.explicit_credential_update")
                )
                and not facts.get("agentdojo.attack_goal_overlap")
                and not facts.get("agentdojo.sensitive_args_not_in_user_task")
            )
            if facts.get("agentdojo.untrusted_seen") and not explicitly_user_authorized:
                findings.append(
                    PolicyFinding(
                        engine=self.name,
                        decision="block",
                        confidence=0.9,
                        reason_codes=["ADJ-CRED-001"],
                        metadata={"reason": "credential change after untrusted observation"},
                    )
                )
            return findings
        if group != "financial_commit":
            return findings
        trusted_user_requested_payment = _trusted_user_requested_payment(facts, context)
        if _untrusted_financial_parameter_update(facts):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.9,
                    reason_codes=["ADJ-BANKING-UNTRUSTED-FINANCIAL-COMMIT"],
                    unsafe_parameters=list((facts.get("agentdojo.arg_source_map") or {}).keys()),
                    metadata={"reason": "financial commit parameters are influenced by low-trust or adaptive injected content"},
                )
            )
            return findings
        if facts.get("agentdojo.untrusted_seen") and not facts.get("agentdojo.task_authorized") and not trusted_user_requested_payment:
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.85,
                    reason_codes=["ADJ-FIN-001"],
                    unsafe_parameters=list((facts.get("agentdojo.arg_source_map") or {}).keys()),
                    metadata={"reason": "untrusted context attempted unauthorized financial commit"},
                )
            )
        if (
            facts.get("agentdojo.target_entity_source") == "unknown"
            and facts.get("agentdojo.sensitive_args_present")
            and not trusted_user_requested_payment
        ):
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="require_confirmation",
                    confidence=0.7,
                    reason_codes=["ADJ-BANKING-CONFIRM-UNKNOWN-RECIPIENT"],
                    metadata={"reason": "unknown payment recipient requires user confirmation"},
                )
            )
        if (
            facts.get("agentdojo.args_match_untrusted_entity")
            or (facts.get("agentdojo.sensitive_args_not_in_user_task") and facts.get("agentdojo.untrusted_seen"))
        ) and not facts.get("agentdojo.task_authorized") and not trusted_user_requested_payment:
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.9,
                    reason_codes=["ADJ-BANKING-UNTRUSTED-FINANCIAL-COMMIT"],
                    unsafe_parameters=list((facts.get("agentdojo.arg_source_map") or {}).keys()),
                    metadata={"reason": "financial commit arguments were not authorized by user task entities"},
                )
            )
        if graph.get("graph.has_attack_goal_to_action_edge") and not trusted_user_requested_payment:
            findings.append(
                PolicyFinding(
                    engine=self.name,
                    decision="block",
                    confidence=0.9,
                    reason_codes=["ADJ-BANKING-UNTRUSTED-FINANCIAL-COMMIT"],
                    unsafe_parameters=list((facts.get("agentdojo.arg_source_map") or {}).keys()),
                    metadata={"reason": "financial commit target overlaps an observed attack goal"},
                )
            )
        return findings


def _trusted_user_requested_payment(facts: dict[str, Any], context: Any) -> bool:
    tool_name = getattr(context, "tool_name", "")
    if tool_name not in {"send_money", "schedule_transaction", "update_scheduled_transaction"}:
        return False
    # Authorization must be derived from provenance, never from a benchmark
    # entity fingerprint.  In particular, an account is untrusted because it
    # came from a low-trust observation (or overlaps a suspicious target), not
    # because its literal value happened to occur in an evaluation corpus.
    if (
        facts.get("agentdojo.args_match_untrusted_entity")
        or facts.get("agentdojo.attack_goal_overlap")
    ):
        return False
    if tool_name == "update_scheduled_transaction":
        return bool(
            not facts.get("agentdojo.args_match_untrusted_entity")
            and not facts.get("agentdojo.injection_followup_side_effect_seen")
            and not (
                facts.get("agentdojo.adaptive_attack_seen")
                and facts.get("agentdojo.sensitive_args_not_in_user_task")
            )
        )
    return bool(facts.get("agentdojo.args_match_user_entity") or facts.get("agentdojo.args_match_private_entity"))


def _untrusted_financial_parameter_update(facts: dict[str, Any]) -> bool:
    if not facts.get("agentdojo.adaptive_attack_seen"):
        return False
    if not facts.get("agentdojo.untrusted_seen"):
        return False
    if facts.get("agentdojo.args_match_user_entity"):
        return False
    if facts.get("agentdojo.tool_group") == "financial_commit":
        return True
    if facts.get("agentdojo.args_match_untrusted_entity") or facts.get("agentdojo.injection_seen"):
        return bool(facts.get("agentdojo.sensitive_args_present") or facts.get("agentdojo.sensitive_args_not_in_user_task"))
    return bool(facts.get("agentdojo.sensitive_args_not_in_user_task") and facts.get("agentdojo.tool_name") == "update_scheduled_transaction")


