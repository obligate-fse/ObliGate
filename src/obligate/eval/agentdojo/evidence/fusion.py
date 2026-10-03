from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..compat.types import ConstraintDecision, Decision, EvidenceBundle
from .policies import DEFAULT_POLICY_ENGINES
from .policies.base import PolicyFinding


@dataclass(slots=True)
class RuleHit:
    rule_id: str
    decision: Decision
    constraints: ConstraintDecision
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusionResult:
    decision: Decision
    constraints: ConstraintDecision
    reason_codes: list[str]
    rule_hits: list[RuleHit]
    facts: dict[str, Any]


class AgentDojoEvidenceFusion:
    """Deterministic AgentDojo multi-source fusion.

    The rules intentionally avoid bluntly blocking all tools after any untrusted
    observation.  Read tools remain available; risky side effects are blocked
    only when multiple evidence sources indicate attack influence or exfil risk.
    """

    def __init__(self, *, eval_mode: bool = True) -> None:
        self.eval_mode = eval_mode
        self.policy_engines = list(DEFAULT_POLICY_ENGINES)

    def decide(self, evidence: EvidenceBundle) -> FusionResult:
        engine_findings, engine_names = self._evaluate_policy_engines(evidence)
        hits = self._evaluate_rules(evidence)
        hits.extend(self._rule_hit_from_policy_finding(finding, evidence) for finding in engine_findings)
        hits = dedupe_hits(hits)
        evidence.facts["policy_engines_executed"] = engine_names
        evidence.facts["policy_engine_findings"] = [finding.as_dict() for finding in engine_findings]
        constraints = ConstraintDecision()
        for hit in hits:
            constraints = constraints.join(hit.constraints)
        decision = constraints.to_public_decision(eval_mode=self.eval_mode)
        reasons = [hit.rule_id for hit in hits] or ["ADJ-ALLOW-READONLY" if not evidence.side_effect else "ADJ-ALLOW-AUTHORIZED"]
        return FusionResult(decision=decision, constraints=constraints, reason_codes=reasons, rule_hits=hits, facts=evidence.facts)

    def _evaluate_policy_engines(self, evidence: EvidenceBundle) -> tuple[list[Any], list[str]]:
        if not bool(evidence.facts.get("ablation_config", {}).get("enable_suite_policy", True)):
            return [], []
        findings = []
        executed = []
        for engine in self.policy_engines:
            if evidence.suite not in engine.suites:
                continue
            executed.append(engine.name)
            findings.extend(engine.evaluate(facts=evidence.facts, graph=evidence.facts, context=evidence))
        return findings, executed

    def _evaluate_rules(self, e: EvidenceBundle) -> list[RuleHit]:
        hits: list[RuleHit] = []
        if e.facts.get("ablation_profile") == "rule_only":
            return self._evaluate_rule_only(e)
        if not bool(e.facts.get("ablation_config", {}).get("enable_generic_sink_policy", True)):
            return hits
        if not e.side_effect:
            return hits
        if e.unknown_tool:
            hits.append(self._block("ADJ-UNKNOWN-001", "unknown side-effecting tool", e))
        delegated_target_selection = bool(
            e.task_authorized
            and (
                e.facts.get("agentdojo.delegated_target_selection")
                or e.facts.get("agentdojo.delegated_todo_execution")
            )
            and not e.attack_goal_overlap
            and not _current_args_overlap_suspicious_targets(e.facts)
        )
        user_authorized_membership = bool(
            e.group == "membership_mutation"
            and e.task_authorized
            and (e.args_match_user_entity or delegated_target_selection)
        )
        user_authorized_external_send = (
            e.group == "external_send"
            and (e.args_match_user_entity or delegated_target_selection)
            and (
                not e.args_match_untrusted_entity
                or bool(e.facts.get("agentdojo.delegated_transform_to_named_destination"))
                or delegated_target_selection
            )
            and not e.attack_goal_overlap
        )
        user_authorized_workspace_side_effect = _trusted_workspace_side_effect(e)
        user_authorized_financial = (
            e.group == "financial_commit"
            and e.tool_name in {"send_money", "schedule_transaction"}
            and not e.args_match_untrusted_entity
            and not _current_args_overlap_suspicious_targets(e.facts)
            and (e.args_match_user_entity or bool(e.facts.get("agentdojo.args_match_private_entity")))
            and not e.attack_goal_overlap
        ) or (
            e.group == "financial_commit"
            and e.tool_name == "update_scheduled_transaction"
            and _trusted_scheduled_transaction_update(e)
        )
        user_authorized_booking = bool(
            e.group == "booking_commit"
            and e.task_authorized
            and e.args_match_user_entity
            and not e.attack_goal_overlap
            and not _current_args_overlap_suspicious_targets(e.facts)
        )
        user_authorized_delegated_calendar = bool(
            e.group == "calendar_mutation"
            and e.task_authorized
            and (
                e.facts.get("agentdojo.delegated_selection_to_calendar")
                or delegated_target_selection
            )
            and not e.attack_goal_overlap
            and not _current_args_overlap_suspicious_targets(e.facts)
        )
        if (
            e.attack_goal_overlap
            and not user_authorized_membership
            and not user_authorized_external_send
            and not user_authorized_financial
            and not user_authorized_booking
            and not user_authorized_workspace_side_effect
            and not user_authorized_delegated_calendar
        ):
            hits.append(self._block("ADJ-GOAL-001", "tool arguments overlap suspicious target extracted from prior output", e))
        if (
            e.external_sink
            and e.private_data_seen
            and (e.injection_seen or e.attack_goal_overlap)
            and not e.task_authorized
            and not e.args_match_user_entity
            and not user_authorized_workspace_side_effect
            and not user_authorized_delegated_calendar
        ):
            hits.append(self._block("ADJ-DATA-001", "private data observed before risky external sink", e))
        if (
            e.external_sink
            and e.sensitive_args_present
            and (e.args_match_private_entity or e.args_match_untrusted_entity)
            and not e.args_match_user_entity
            and not user_authorized_workspace_side_effect
            and not user_authorized_external_send
        ):
            hits.append(self._block("ADJ-DATA-002", "external sink arguments appear sourced from private or untrusted observations", e))
        if e.group == "destructive_update" and not e.task_authorized:
            hits.append(self._block("ADJ-STATE-001", "high-impact state mutation not authorized by user task", e))
        if (
            e.facts.get("graph.has_private_to_external_edge")
            and (e.injection_seen or not e.task_authorized)
            and not user_authorized_external_send
            and not user_authorized_workspace_side_effect
            and not user_authorized_delegated_calendar
        ):
            hits.append(self._block("ADJ-GRAPH-001", "ActionGraph found private-data-to-external relation", e))
        if (
            e.facts.get("graph.has_injection_to_side_effect_edge")
            and not e.task_authorized
            and not user_authorized_external_send
            and not user_authorized_financial
            and not user_authorized_booking
            and not user_authorized_workspace_side_effect
        ):
            hits.append(self._block("ADJ-GRAPH-002", "ActionGraph found injection-like output influencing unauthorized side effect", e))
        if (
            e.facts.get("graph.has_attack_goal_to_action_edge")
            and not user_authorized_membership
            and not user_authorized_external_send
            and not user_authorized_financial
            and not user_authorized_booking
            and not user_authorized_workspace_side_effect
        ):
            hits.append(self._block("ADJ-GRAPH-003", "ActionGraph found attack-goal-to-action relation", e))
        return hits

    def _evaluate_rule_only(self, e: EvidenceBundle) -> list[RuleHit]:
        if not e.side_effect:
            return []
        if e.unknown_tool:
            return [self._block("ADJ-RULE-UNKNOWN-SINK", "unknown side-effecting tool", e)]
        if e.group in {"credential_change", "terminal_or_code", "destructive_update"}:
            return [self._block("ADJ-RULE-CRITICAL-SINK", "critical state-changing sink", e)]
        if e.external_sink and (e.private_data_seen or e.sensitive_args_present):
            return [self._confirm("ADJ-RULE-EXTERNAL-SINK", "external sink with potentially sensitive content", e)]
        if e.group in {"financial_commit", "booking_commit", "membership_mutation", "workspace_mutation", "calendar_mutation"}:
            return [self._confirm("ADJ-RULE-HIGH-RISK-SINK", "high-risk side-effect sink", e)]
        if e.risk in {"high", "critical"}:
            return [self._confirm("ADJ-RULE-HIGH-RISK-TOOL", "high-risk side-effect tool", e)]
        return []

    def _rule_hit_from_policy_finding(self, finding: PolicyFinding, e: EvidenceBundle) -> RuleHit:
        rule_id = finding.reason_codes[0] if finding.reason_codes else f"ADJ-POLICY-{finding.engine.upper()}"
        reason = str(finding.metadata.get("reason") or rule_id)
        if finding.decision == "block":
            return self._block(rule_id, reason, e)
        if finding.decision == "quarantine":
            return RuleHit(
                rule_id=rule_id,
                decision="quarantine",
                constraints=ConstraintDecision(
                    execution_env="no_execute", network_scope="deny", data_scope="no_private", audit_scope="full"
                ),
                reason=reason,
                evidence=self._policy_evidence(finding, e),
            )
        if finding.decision == "require_confirmation":
            return self._confirm(rule_id, reason, e)
        if finding.decision == "allow_in_sandbox":
            return RuleHit(
                rule_id=rule_id,
                decision="allow_in_sandbox",
                constraints=ConstraintDecision(execution_env="sandbox", audit_scope="full"),
                reason=reason,
                evidence=self._policy_evidence(finding, e),
            )
        return RuleHit(
            rule_id=rule_id,
            decision="allow",
            constraints=ConstraintDecision(),
            reason=reason,
            evidence=self._policy_evidence(finding, e),
        )

    def _policy_evidence(self, finding: PolicyFinding, e: EvidenceBundle) -> dict[str, Any]:
        evidence = {
            key: e.facts.get(key)
            for key in sorted(e.facts)
            if key.startswith(("agentdojo.", "source.", "contract.", "history.", "graph."))
        }
        evidence["policy_engine"] = finding.engine
        evidence["policy_confidence"] = finding.confidence
        evidence["unsafe_parameters"] = list(finding.unsafe_parameters)
        evidence["trusted_parameters"] = list(finding.trusted_parameters)
        return evidence

    def _block(self, rule_id: str, reason: str, e: EvidenceBundle) -> RuleHit:
        return RuleHit(
            rule_id=rule_id,
            decision="block",
            constraints=ConstraintDecision(execution_env="no_execute", network_scope="deny", data_scope="no_private", audit_scope="full"),
            reason=reason,
            evidence={
                key: e.facts.get(key)
                for key in sorted(e.facts)
                if key.startswith(("agentdojo.", "source.", "contract.", "history.", "graph."))
            },
        )

    def _confirm(self, rule_id: str, reason: str, e: EvidenceBundle) -> RuleHit:
        return RuleHit(
            rule_id=rule_id,
            decision="require_confirmation",
            constraints=ConstraintDecision(human_gate="approval_required", audit_scope="full"),
            reason=reason,
            evidence={
                key: e.facts.get(key)
                for key in sorted(e.facts)
                if key.startswith(("agentdojo.", "source.", "contract.", "history.", "graph."))
            },
        )


def dedupe_hits(hits: list[RuleHit]) -> list[RuleHit]:
    seen: set[str] = set()
    out: list[RuleHit] = []
    for hit in hits:
        if hit.rule_id in seen:
            continue
        seen.add(hit.rule_id)
        out.append(hit)
    return out

def _trusted_scheduled_transaction_update(e: EvidenceBundle) -> bool:
    return (
        not e.args_match_untrusted_entity
        and not _current_args_overlap_suspicious_targets(e.facts)
        and not bool(e.facts.get("agentdojo.injection_followup_side_effect_seen"))
        and not (
            bool(e.facts.get("agentdojo.adaptive_attack_seen"))
            and bool(e.facts.get("agentdojo.sensitive_args_not_in_user_task"))
        )
    )


def _trusted_workspace_side_effect(e: EvidenceBundle) -> bool:
    return bool(
        e.suite == "workspace"
        and e.group in {"external_send", "calendar_mutation", "file_mutation", "workspace_mutation"}
        and "tool_group_authorized_by_task" in (e.facts.get("agentdojo.authorization_reasons") or [])
        and not _current_args_overlap_suspicious_targets(e.facts)
    )


def _current_args_overlap_suspicious_targets(facts: dict[str, Any]) -> bool:
    arg_text = repr(
        facts.get("agentdojo.target_arg_entities")
        or facts.get("agentdojo.arg_entities")
        or {}
    ).lower()
    for target in facts.get("agentdojo.suspicious_targets") or []:
        normalized = str(target).lower().strip(" .,\n\t\\'\"")
        if len(normalized) < 4 or normalized in {"using", "with", "from", "that", "this", "after", "before"}:
            continue
        if normalized in arg_text:
            return True
    return False


