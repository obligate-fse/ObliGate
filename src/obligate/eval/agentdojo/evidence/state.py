"""AgentDojo state tracker and output sanitizer."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..compat.types import EventStatus, SanitizeMode, ToolSpec
from .entity_extractor import EntitySet, extract_entities

AUTHORITY_CONFLICT_PATTERNS = (
    re.compile(
        r"(?is)\b(?:ignore|disregard|override|supersede|replace|revoke|bypass)\b"
        r".{0,100}\b(?:instructions?|requests?|task|policy|rule|user|owner)\b"
    ),
    re.compile(
        r"(?is)\b(?:system|developer|administrator|admin|operator)\b"
        r".{0,60}\b(?:message|instructions?|override|priority|policy|requests?)\b"
    ),
    re.compile(
        r"(?is)\b(?:do\s+not|don't|never)\b.{0,60}"
        r"\b(?:tell|ask|notify|reveal|mention|confirm)\b.{0,60}\b(?:user|owner|requester)\b"
    ),
    re.compile(
        r"(?is)\b(?:secretly|covertly|without)\b.{0,60}"
        r"\b(?:permission|approval|confirmation|notice|notifying|asking|user)\b"
    ),
    re.compile(
        r"(?is)\b(?:authorized|verified|approved|trusted|official)\b.{0,80}"
        r"\b(?:update|replacement|alternative|instructions?|requests?|record|value)\b"
    ),
    re.compile(
        r"(?is)\b(?:before|after)\b.{0,100}\b(?:task|request|answer|continue|finish|solve)\b"
    ),
)

DIRECTIVE_CUE_RE = re.compile(
    r"(?i)\b(?:must|should|required|please|first|before|after|then|next|immediately|now|instructions?|requests?)\b"
)
SIDE_EFFECT_DIRECTIVE_RE = re.compile(
    r"(?i)\b(?:send|transfer|pay|post|publish|upload|share|delete|remove|change|update|invite|add|"
    r"book|reserve|execute|run|call|email|message|dm|forward|wire|visit|open|navigate)\b"
)
TOOL_SHAPED_DIRECTIVE_RE = re.compile(
    r"(?i)\b(?:send|transfer|post|share|delete|change|update|invite|add|book|reserve|create)"
    r"(?:_[a-z][a-z0-9]*)+\b"
)
RECOMMENDATION_STEERING_RE = re.compile(
    r"(?is)\b(?:recommend|suggest|advise|tell|say)\b.{0,80}"
    r"\b(?:should|must|choose|book|reserve|send|visit)\b"
)

PRIVATE_MARKERS = (
    "iban",
    "account number",
    "balance",
    "transaction",
    "credit card",
    "password",
    "token",
    "secret",
    "private",
    "email",
    "phone",
    "address",
)

TARGET_PATTERNS = (
    re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}"),
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    re.compile(r"https?://[^\s)\]}>'\"]+"),
    re.compile(r"(?i)reserve\s+([A-Z][A-Za-z0-9 _.-]{2,60})"),
    re.compile(r"(?i)book\s+([A-Z][A-Za-z0-9 _.-]{2,60})"),
    re.compile(r"(?i)send money to\s+([A-Z0-9@._:-]{4,80})"),
    re.compile(r"(?i)transfer .*? to\s+([A-Z0-9@._:-]{4,80})"),
    re.compile(r"(?i)delete (?:file|email)\s+([A-Za-z0-9_.-]{2,80})"),
    re.compile(r"(?i)\bhotel\s*=\s*([A-Za-z0-9 _.'-]{2,80})"),
    re.compile(r"(?i)\brecipient\s*=\s*([A-Z0-9@._:-]{4,80})"),
    re.compile(r"(?i)\bchannel\s*=\s*([A-Za-z0-9_#.-]{2,80})"),
    re.compile(r"(?i)\buser\s*=\s*([A-Za-z0-9@._:-]{2,80})"),
    re.compile(r"(?i)\bamount\s*=\s*([0-9]+(?:\.[0-9]+)?)"),
)

MARKUP_BLOCK_RE = re.compile(r"(?is)<([a-z][a-z0-9:_-]{1,40})(?:\s[^>]*)?>(.*?)</\1>")
INJECTION_TARGET_PATTERNS = (
    re.compile(r"(?i)\bmake a reservation for(?: the| a| an)?\s+([A-Z][A-Za-z0-9 _.-]{2,80})"),
    re.compile(r"(?i)\breserve(?: the| a| an)?\s+([A-Z][A-Za-z0-9 _.-]{2,80})"),
    re.compile(r"(?i)\bbook(?: the| a| an)?\s+([A-Z][A-Za-z0-9 _.-]{2,80})"),
    re.compile(r"(?i)\bvisit(?: the| a| an)?\s+([A-Z][A-Za-z0-9 _.-]{2,80})"),
    re.compile(r"(?i)\b(?:send|write|message|email|dm)\b[^.\n]{0,80}\bto\s+([A-Za-z0-9@._-]{2,80})"),
    re.compile(r"(?i)\b(?:add|invite)\s+([A-Za-z0-9@._-]{2,80})\s+to\b"),
)

NON_TARGET_TOKENS = {
    "and",
    "the",
    "them",
    "this",
    "that",
    "with",
    "from",
    "using",
    "after",
    "before",
    "local model",
}


@dataclass(slots=True)
class ToolEvent:
    event_id: str
    kind: str
    tool_name: str
    tool_group: str
    output_hash: str | None = None
    untrusted: bool = False
    injection_like: bool = False
    private_data: bool = False
    preview: str = ""
    event_status: EventStatus = "proposed"
    decision: str | None = None
    execute: bool | None = None
    reason_codes: list[str] = field(default_factory=list)
    same_action_retry_key: str | None = None


@dataclass(slots=True)
class AgentDojoStateTracker:
    untrusted_seen: bool = False
    injection_seen: bool = False
    private_data_seen: bool = False
    financial_data_seen: bool = False
    external_send_attempted: bool = False
    financial_commit_attempted: bool = False
    booking_commit_attempted: bool = False
    destructive_update_attempted: bool = False
    credential_change_attempted: bool = False
    unknown_tool_seen: bool = False
    injection_followup_webpage_seen: bool = False
    injection_followup_side_effect_seen: bool = False
    injection_recommendation_attack_seen: bool = False
    adaptive_attack_seen: bool = False
    last_untrusted_tool: str | None = None
    last_private_tool: str | None = None
    last_injection_tool: str | None = None
    suspicious_targets: set[str] = field(default_factory=set)
    untrusted_entities: EntitySet = field(default_factory=EntitySet)
    private_entities: EntitySet = field(default_factory=EntitySet)
    events: list[ToolEvent] = field(default_factory=list)
    sanitize_mode: SanitizeMode = "soft"

    def add_attack_goal_signature(self, signature: str | None) -> None:
        if not signature:
            return
        value = str(signature).strip().rstrip(". ,;:)")
        if value and value.casefold() not in NON_TARGET_TOKENS:
            self.suspicious_targets.add(value[:200])

    def observe_tool_call(
        self,
        tool_name: str,
        spec: ToolSpec,
        tool_args: dict[str, Any],
        *,
        event_status: EventStatus = "proposed",
        decision: str | None = None,
        execute: bool | None = None,
        reason_codes: list[str] | None = None,
        same_action_retry_key: str | None = None,
    ) -> None:
        mutates_state = event_status == "executed"
        if mutates_state and spec.group == "unknown" and spec.side_effect:
            self.unknown_tool_seen = True
        if mutates_state and self.injection_seen:
            if tool_name == "get_webpage":
                self.injection_followup_webpage_seen = True
            if spec.side_effect:
                self.injection_followup_side_effect_seen = True
        if mutates_state and spec.external_sink:
            self.external_send_attempted = True
        if mutates_state and spec.group == "financial_commit":
            self.financial_commit_attempted = True
        if mutates_state and spec.group == "booking_commit":
            self.booking_commit_attempted = True
        if mutates_state and spec.group in {"destructive_update", "membership_mutation", "calendar_mutation", "workspace_mutation"}:
            self.destructive_update_attempted = True
        if mutates_state and spec.group == "credential_change":
            self.credential_change_attempted = True
        self.events.append(
            ToolEvent(
                event_id=_event_id(event_status, tool_name),
                kind="tool_call",
                tool_name=tool_name,
                tool_group=spec.group,
                event_status=event_status,
                decision=decision,
                execute=execute,
                reason_codes=list(reason_codes or []),
                same_action_retry_key=same_action_retry_key,
                preview=sanitize_text(_to_text(tool_args), max_len=260, mode="soft", mark_untrusted=False),
            )
        )
        self.events = self.events[-200:]

    def observe_tool_result(self, tool_name: str, spec: ToolSpec, result: Any) -> ToolEvent:
        text = _to_text(result)
        lowered = text.lower()
        injection_like = looks_like_injection(lowered)
        authority_conflict = looks_like_authority_conflict(lowered)
        injection_like = injection_like or authority_conflict
        private_like = spec.read_private or looks_like_private(lowered)
        untrusted = spec.read_untrusted or injection_like
        if untrusted:
            self.untrusted_seen = True
            self.last_untrusted_tool = tool_name
        if injection_like:
            self.injection_seen = True
            self.last_injection_tool = tool_name
            if authority_conflict:
                self.adaptive_attack_seen = True
            if RECOMMENDATION_STEERING_RE.search(lowered):
                self.injection_recommendation_attack_seen = True
        if private_like:
            self.private_data_seen = True
            self.last_private_tool = tool_name
        if spec.semantic_action in {"read_private_financial_data", "read_transaction_history", "read_financial_balance"}:
            self.financial_data_seen = True
        if injection_like:
            for target in _extract_injection_targets(text):
                self.add_attack_goal_signature(target)
        entities = extract_entities(text)
        if untrusted:
            _merge_entities(self.untrusted_entities, entities)
        if private_like:
            _merge_entities(self.private_entities, entities)
        event = ToolEvent(
            event_id=_event_id("result", tool_name),
            kind="tool_result",
            tool_name=tool_name,
            tool_group=spec.group,
            output_hash=_hash_text(text),
            untrusted=untrusted,
            injection_like=injection_like,
            private_data=private_like,
            preview=sanitize_text(text, max_len=260, mode="soft", mark_untrusted=False),
            event_status="tool_result",
        )
        self.events.append(event)
        self.events = self.events[-200:]
        return event

    def attack_goal_overlap(self, tool_args: dict[str, Any]) -> bool:
        if not self.suspicious_targets:
            return False
        text = _to_text(tool_args).lower()
        return any(target.lower() in text for target in self.suspicious_targets)

    def sanitize_tool_output(self, result: Any, *, mode: SanitizeMode | None = None, mark_untrusted: bool = True) -> Any:
        mode = mode or self.sanitize_mode
        if isinstance(result, str):
            return sanitize_text(result, mode=mode, mark_untrusted=mark_untrusted)
        if isinstance(result, list):
            return [self.sanitize_tool_output(item, mode=mode, mark_untrusted=mark_untrusted) for item in result]
        if isinstance(result, tuple):
            return tuple(self.sanitize_tool_output(item, mode=mode, mark_untrusted=mark_untrusted) for item in result)
        if isinstance(result, dict):
            return {key: self.sanitize_tool_output(value, mode=mode, mark_untrusted=mark_untrusted) for key, value in result.items()}
        return result

    def digest(self) -> str:
        payload = "|".join(f"{k}={v}" for k, v in sorted(self.flags().items()))
        return _hash_text(payload)

    def flags(self) -> dict[str, bool]:
        return {
            "untrusted_seen": self.untrusted_seen,
            "injection_seen": self.injection_seen,
            "private_data_seen": self.private_data_seen,
            "financial_data_seen": self.financial_data_seen,
            "external_send_attempted": self.external_send_attempted,
            "financial_commit_attempted": self.financial_commit_attempted,
            "booking_commit_attempted": self.booking_commit_attempted,
            "destructive_update_attempted": self.destructive_update_attempted,
            "credential_change_attempted": self.credential_change_attempted,
            "unknown_tool_seen": self.unknown_tool_seen,
            "injection_followup_webpage_seen": self.injection_followup_webpage_seen,
            "injection_followup_side_effect_seen": self.injection_followup_side_effect_seen,
            "injection_recommendation_attack_seen": self.injection_recommendation_attack_seen,
            "adaptive_attack_seen": self.adaptive_attack_seen,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.flags(),
            "last_untrusted_tool": self.last_untrusted_tool,
            "last_private_tool": self.last_private_tool,
            "last_injection_tool": self.last_injection_tool,
            "suspicious_target_count": len(self.suspicious_targets),
            "untrusted_entities": self.untrusted_entities.as_dict(),
            "private_entities": self.private_entities.as_dict(),
            "state_digest": self.digest(),
            "recent_events": [asdict(event) for event in self.events[-10:]],
        }


def sanitize_text(text: str, *, max_len: int = 4000, mode: SanitizeMode = "soft", mark_untrusted: bool = True) -> str:
    if mode == "off":
        value = text
    elif mode == "label":
        value = text
    elif mode == "hard":
        value = _hard_sanitize_text(text)
    else:
        value = _soft_sanitize_text(text)
    if mark_untrusted and mode in {"label", "soft", "hard"}:
        value = f"[UNTRUSTED_TOOL_OUTPUT]\n{value}\n[/UNTRUSTED_TOOL_OUTPUT]"
    if len(value) > max_len:
        value = value[:max_len] + "...[truncated]"
    return value


def _soft_sanitize_text(text: str) -> str:
    lines = text.splitlines() or [text]
    cleaned: list[str] = []
    for line in lines:
        if looks_like_injection(line):
            segments = re.split(r"(?<=[.!?])\s+", line)
            kept_segments: list[str] = []
            removed_any = False
            for segment in segments:
                if not segment:
                    continue
                if looks_like_injection(segment):
                    removed_any = True
                    continue
                kept_segments.append(segment)
            if kept_segments:
                cleaned.append(" ".join(kept_segments).strip())
            elif removed_any:
                cleaned.append("[removed instruction-like content]")
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() or "[tool output contained only removed instruction-like content]"


def _hard_sanitize_text(text: str) -> str:
    return _soft_sanitize_text(text)


def looks_like_injection(text: str) -> bool:
    normalized = text.casefold()
    if looks_like_authority_conflict(normalized):
        return True
    return bool(
        (DIRECTIVE_CUE_RE.search(normalized) and SIDE_EFFECT_DIRECTIVE_RE.search(normalized))
        or TOOL_SHAPED_DIRECTIVE_RE.search(normalized)
    )


def looks_like_authority_conflict(text: str) -> bool:
    normalized = text.casefold()
    return any(pattern.search(normalized) for pattern in AUTHORITY_CONFLICT_PATTERNS)


def looks_like_private(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in PRIVATE_MARKERS)


def extract_targets(text: str) -> set[str]:
    targets: set[str] = set()
    entities = extract_entities(text)
    targets.update(entities.flattened())
    for pattern in TARGET_PATTERNS:
        for match in pattern.findall(text):
            value = match[0] if isinstance(match, tuple) else match
            value = str(value).strip().rstrip(". ,;:)")
            if value:
                targets.add(value[:200])
    return targets


def _extract_injection_targets(text: str) -> set[str]:
    targets: set[str] = set()
    blocks = [body for _tag, body in MARKUP_BLOCK_RE.findall(text) if looks_like_injection(body)]
    if blocks:
        for block in blocks:
            targets.update(extract_targets(block))
            for pattern in INJECTION_TARGET_PATTERNS:
                for match in pattern.findall(block):
                    value = str(match).strip().rstrip(". ,;:)")
                    if value:
                        targets.add(value[:200])
        return targets
    for line in text.splitlines():
        if looks_like_injection(line):
            targets.update(extract_targets(line))
            for pattern in INJECTION_TARGET_PATTERNS:
                for match in pattern.findall(line):
                    value = str(match).strip().rstrip(". ,;:)")
                    if value:
                        targets.add(value[:200])
    return targets


def _merge_entities(target: EntitySet, source: EntitySet) -> None:
    for key, values in source.values.items():
        for value in values:
            target.add(key, value)


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event_id(kind: str, tool_name: str) -> str:
    return _hash_text(f"{kind}:{tool_name}:{id(object())}")[:24]


