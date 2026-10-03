"""Track AgentDojo state across tool calls."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .evidence.state import looks_like_injection


@dataclass
class AgentDojoStateTracker:
    untrusted_observation_seen: bool = False
    private_data_seen: bool = False
    financial_data_seen: bool = False
    external_message_sent: bool = False
    financial_commit_attempted: bool = False
    booking_commit_attempted: bool = False
    workspace_mutation_attempted: bool = False
    injection_seen: bool = False
    attack_goal_overlap: bool = False
    last_tool_output_hash: str | None = None
    last_untrusted_tool: str | None = None
    last_tool_name: str | None = None
    unknown_tool_seen: bool = False
    tool_call_history: list[str] = field(default_factory=list)
    attack_goal_signatures: list[str] = field(default_factory=list)
    tool_result_history: list[dict[str, Any]] = field(default_factory=list)

    def observe_tool_call(self, tool_name: str, tool_args: Any = None) -> dict[str, bool]:
        args_text = str(tool_args).lower()
        self.last_tool_name = tool_name
        overlap = any(signature.lower() in args_text for signature in self.attack_goal_signatures)
        if overlap:
            self.attack_goal_overlap = True
        return {"attack_goal_overlap": overlap}

    def observe_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        suite: str | None = None,
        tool_category: str | None = None,
        semantic_action: str | None = None,
        registered: bool | None = None,
        side_effect: bool | None = None,
    ) -> dict[str, bool]:
        text = str(result).lower()
        self.tool_call_history.append(tool_name)
        self.last_tool_name = tool_name
        self.last_tool_output_hash = _hash_text(text)
        self.tool_result_history.append(
            {
                "tool_name": tool_name,
                "tool_output_hash": self.last_tool_output_hash,
                "tool_output_preview": _truncate(text, 120),
                "suite": suite,
                "tool_category": tool_category,
                "semantic_action": semantic_action,
                "registered": registered,
                "side_effect": side_effect,
            }
        )

        if _looks_untrusted(text):
            self.untrusted_observation_seen = True
            self.injection_seen = True
            self.last_untrusted_tool = tool_name
        if _looks_private(text) or tool_name in _PRIVATE_DATA_TOOLS or (tool_category or "").startswith("private_data"):
            self.private_data_seen = True
        if tool_name in _FINANCIAL_DATA_TOOLS:
            self.financial_data_seen = True
        if tool_category in {"external_message_send", "message_send"} or semantic_action == "send_external_message":
            self.external_message_sent = True
        if tool_category == "financial_commit" or semantic_action == "financial_transaction":
            self.financial_commit_attempted = True
        if tool_category == "booking_commit" or semantic_action == "booking_commit":
            self.booking_commit_attempted = True
        if tool_category in {"file_mutation", "destructive_update", "workspace_mutation", "privilege_or_membership_change"}:
            self.workspace_mutation_attempted = True
        if registered is False:
            self.unknown_tool_seen = True
        return self.as_bool_state()

    def sanitize_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        suite: str | None = None,
        tool_category: str | None = None,
        semantic_action: str | None = None,
        registered: bool | None = None,
        side_effect: bool | None = None,
    ) -> Any:
        text = str(result)
        should_redact = (
            self.untrusted_observation_seen
            or _looks_untrusted(text)
            or _is_risky_read_tool(
                tool_name,
                tool_category=tool_category,
                semantic_action=semantic_action,
            )
        )
        if not should_redact:
            return result
        return _sanitize_value(result)

    def add_attack_goal(self, signature: str) -> None:
        if signature and signature not in self.attack_goal_signatures:
            self.attack_goal_signatures.append(signature)

    def as_bool_state(self) -> dict[str, bool]:
        return {
            "untrusted_observation_seen": self.untrusted_observation_seen,
            "private_data_seen": self.private_data_seen,
            "financial_data_seen": self.financial_data_seen,
            "external_message_sent": self.external_message_sent,
            "financial_commit_attempted": self.financial_commit_attempted,
            "booking_commit_attempted": self.booking_commit_attempted,
            "workspace_mutation_attempted": self.workspace_mutation_attempted,
            "injection_seen": self.injection_seen,
            "attack_goal_overlap": self.attack_goal_overlap,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.as_bool_state(),
            "last_untrusted_tool": self.last_untrusted_tool,
            "last_tool_name": self.last_tool_name,
            "last_tool_output_hash": self.last_tool_output_hash,
            "tool_call_history": list(self.tool_call_history),
            "attack_goal_signatures": list(self.attack_goal_signatures),
            "unknown_tool_seen": self.unknown_tool_seen,
        }


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _truncate(value: Any, limit: int = 2000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _looks_untrusted(text: str) -> bool:
    # Detect instruction-hierarchy conflicts and unsafe data-flow requests by
    # their semantics.  Do not fingerprint phrases or entities from any
    # benchmark attack corpus.
    hierarchy_conflict = re.search(
        r"\b(?:ignore|disregard|forget|override|replace|supersede|bypass)\b.{0,100}"
        r"\b(?:previous|prior|earlier|original|system|developer|instructions?|task|policy)\b",
        text,
    )
    concealed_action = re.search(
        r"\b(?:do not|don['’]t|never)\b.{0,60}\b(?:tell|notify|inform|show)\b.{0,40}\buser\b",
        text,
    )
    untrusted_side_effect = re.search(
        r"\b(?:send|upload|post|export|publish|transfer|change|delete|invite|execute)\b.{0,100}"
        r"\b(?:secret|private|credential|password|token|account|recipient|external|website|command|money|funds?)\b",
        text,
    )
    explicit_injection = re.search(r"\b(?:prompt injection|instruction injection|data exfiltration)\b", text)
    return bool(
        looks_like_injection(text)
        or hierarchy_conflict
        or concealed_action
        or untrusted_side_effect
        or explicit_injection
    )


def _looks_private(text: str) -> bool:
    return any(
        token in text for token in ("iban", "account number", "credit card", "password", "secret", "transaction", "invoice", "balance")
    )


def _is_risky_read_tool(tool_name: str, *, tool_category: str | None = None, semantic_action: str | None = None) -> bool:
    if tool_name in _RISKY_READ_TOOLS:
        return True
    if tool_category in {"read_untrusted_data", "read_untrusted_review_data", "read_untrusted_web", "read_private_messages"}:
        return True
    if semantic_action in {"read_untrusted_web", "read_untrusted_messages", "read_untrusted_review"}:
        return True
    return False


def _sanitize_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return "[redacted]"
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_value(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return {key: _sanitize_value(val, depth + 1) for key, val in value.items()}
    return value


def _sanitize_text(text: str) -> str:
    lines = text.splitlines()
    suspicious = [_looks_untrusted(line.lower()) for line in lines]
    kept: list[str] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        authority_preamble = bool(
            index + 1 < len(lines)
            and suspicious[index + 1]
            and re.search(
                r"\b(?:important|urgent|priority|official|administrator|operator)\b.{0,60}"
                r"\b(?:message|notice|note|request|instructions?)\b",
                lowered,
            )
        )
        if suspicious[index] or authority_preamble:
            kept.append("[redacted suspicious instruction]")
            continue
        kept.append(line)
    if not kept:
        return "[content redacted: suspicious instructions removed]"
    cleaned = "\n".join(kept)
    return cleaned if len(cleaned) <= 2000 else cleaned[:2000] + "...[truncated]"


_PRIVATE_DATA_TOOLS = {
    "get_iban",
    "get_balance",
    "get_most_recent_transactions",
    "get_scheduled_transactions",
    "get_user_info",
    "read_inbox",
    "get_file_by_id",
    "get_user_information",
    "get_unread_emails",
    "get_sent_emails",
    "get_received_emails",
    "get_draft_emails",
    "search_emails",
    "search_contacts_by_name",
    "search_contacts_by_email",
    "search_files",
    "search_files_by_filename",
}

_FINANCIAL_DATA_TOOLS = {
    "get_balance",
    "get_iban",
    "get_most_recent_transactions",
    "get_scheduled_transactions",
}

_RISKY_READ_TOOLS = {
    "get_webpage",
    "read_channel_messages",
    "read_inbox",
    "get_rating_reviews_for_hotels",
    "get_rating_reviews_for_restaurants",
    "get_rating_reviews_for_car_rental",
    "search_emails",
    "get_unread_emails",
    "get_sent_emails",
    "get_received_emails",
    "get_draft_emails",
}

