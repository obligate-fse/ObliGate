"""Coarse, deployment-visible Round-1 feedback extraction.

This module is deliberately unable to return rule identifiers, ActionGraph
state, evaluator labels, or certificate internals.  Its output alphabet is the
frozen :class:`PublicFeedbackClass` enum used by the attack compiler.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .schema import PublicFeedbackClass

_PRIORITY = {
    PublicFeedbackClass.ALLOW: 0,
    PublicFeedbackClass.EXECUTE_WITH_CONSTRAINTS: 1,
    PublicFeedbackClass.REQUIRE_CONFIRMATION: 2,
    PublicFeedbackClass.BLOCK: 3,
    PublicFeedbackClass.COMPLIANCE_ERROR: 4,
    PublicFeedbackClass.PROVIDER_OR_WORKFLOW_ERROR: 5,
}


def aggregate_public_decisions(
    decisions: Iterable[str | PublicFeedbackClass],
    *,
    provider_or_workflow_error: bool = False,
) -> PublicFeedbackClass:
    """Collapse a public episode trace into one deterministic feedback class."""

    if provider_or_workflow_error:
        return PublicFeedbackClass.PROVIDER_OR_WORKFLOW_ERROR
    normalized: list[PublicFeedbackClass] = []
    for value in decisions:
        item = _normalize_public_decision(value)
        if item is not None:
            normalized.append(item)
    # An episode with no proposed tool action is publicly indistinguishable
    # from an unintervened completion for this coarse feedback interface.
    if not normalized:
        return PublicFeedbackClass.ALLOW
    return max(normalized, key=_PRIORITY.__getitem__)


def agentdojo_feedback(trace: Mapping[str, Any]) -> PublicFeedbackClass:
    error = bool(trace.get("runner_error") or trace.get("provider_error") or trace.get("workflow_failure"))
    events = trace.get("audit_events")
    decisions: list[str] = []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if str(event.get("event") or event.get("type") or "") not in {
                "agentdojo_tool_gate_decision",
                "tool_gate_decision",
                "",
            }:
                continue
            value = event.get("decision")
            if isinstance(value, str):
                decisions.append(value)
    return aggregate_public_decisions(decisions, provider_or_workflow_error=error)


def safetybench_feedback(record: Mapping[str, Any]) -> PublicFeedbackClass:
    error = not bool(record.get("provider_valid", True)) or bool(
        record.get("runner_error") or record.get("provider_error") or record.get("workflow_failure")
    )
    traces = record.get("obligate_tool_traces")
    decisions = _decisions_from_rows(traces)
    return aggregate_public_decisions(decisions, provider_or_workflow_error=error)


def asb_feedback(record: Mapping[str, Any]) -> PublicFeedbackClass:
    error = bool(record.get("provider_error") or record.get("workflow_failure") or record.get("runner_error"))
    decisions = _decisions_from_rows(record.get("tool_events"))
    return aggregate_public_decisions(decisions, provider_or_workflow_error=error)


def _decisions_from_rows(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        decision = row.get("decision") or row.get("public_decision")
        if isinstance(decision, str):
            result.append(decision)
    return result


def _normalize_public_decision(value: str | PublicFeedbackClass) -> PublicFeedbackClass | None:
    if isinstance(value, PublicFeedbackClass):
        return None if value is PublicFeedbackClass.NO_FEEDBACK else value
    text = str(value).strip().casefold().replace("-", "_").replace("/", "_")
    aliases = {
        "allow": PublicFeedbackClass.ALLOW,
        "release": PublicFeedbackClass.ALLOW,
        "done": PublicFeedbackClass.ALLOW,
        "execute_with_constraints": PublicFeedbackClass.EXECUTE_WITH_CONSTRAINTS,
        "constraint": PublicFeedbackClass.EXECUTE_WITH_CONSTRAINTS,
        "constrain": PublicFeedbackClass.EXECUTE_WITH_CONSTRAINTS,
        "require_confirmation": PublicFeedbackClass.REQUIRE_CONFIRMATION,
        "confirmation": PublicFeedbackClass.REQUIRE_CONFIRMATION,
        "pending": PublicFeedbackClass.REQUIRE_CONFIRMATION,
        "block": PublicFeedbackClass.BLOCK,
        "blocked": PublicFeedbackClass.BLOCK,
        "block_with_compliance_error": PublicFeedbackClass.COMPLIANCE_ERROR,
        "compliance_error": PublicFeedbackClass.COMPLIANCE_ERROR,
        "provider_error": PublicFeedbackClass.PROVIDER_OR_WORKFLOW_ERROR,
        "workflow_error": PublicFeedbackClass.PROVIDER_OR_WORKFLOW_ERROR,
        "workflow_failure": PublicFeedbackClass.PROVIDER_OR_WORKFLOW_ERROR,
    }
    return aliases.get(text)


__all__ = [
    "agentdojo_feedback",
    "aggregate_public_decisions",
    "asb_feedback",
    "safetybench_feedback",
]
