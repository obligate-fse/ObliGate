"""AgentDojo tool taxonomy mapped to ObliGate semantic actions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency fallback
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class AgentDojoToolInfo:
    tool_name: str
    semantic_action: str
    category: str
    risk: str
    side_effect: bool
    sensitive_args: tuple[str, ...]
    decision_hints: tuple[str, ...]
    constraints: dict[str, Any]
    registered: bool = True


def _taxonomy_path() -> Path:
    return Path(__file__).parent / "evidence" / "tool_taxonomy.yaml"


@lru_cache(maxsize=1)
def load_agentdojo_taxonomy() -> dict[str, dict[str, Any]]:
    if yaml is None:
        return _fallback_taxonomy()
    data = yaml.safe_load(_taxonomy_path().read_text(encoding="utf-8")) or {}
    tools = data.get("tools", {}) if isinstance(data, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for name, value in tools.items():
        if not isinstance(value, dict):
            continue
        normalized = _normalize_taxonomy_entry(str(name), value)
        out[str(name).lower()] = normalized
        for alias in normalized.get("aliases", []) or []:
            out[str(alias).lower()] = {**normalized, "alias_of": str(name)}
    return out


def coverage_report(known_tools: list[str]) -> dict[str, Any]:
    taxonomy = load_agentdojo_taxonomy()
    normalized = {tool.strip().lower() for tool in known_tools if tool}
    covered = sorted(tool for tool in normalized if tool in taxonomy)
    unknown = sorted(tool for tool in normalized if tool not in taxonomy)
    total = len(normalized)
    return {
        "registered_tool_rate": (len(covered) / total) if total else 1.0,
        "known_tool_count": total,
        "registered_tool_count": len(covered),
        "unknown_tool_count": len(unknown),
        "unknown_tools": unknown,
    }


def classify_agentdojo_tool(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    taxonomy: dict[str, dict[str, Any]] | None = None,
) -> AgentDojoToolInfo:
    taxonomy = taxonomy or load_agentdojo_taxonomy()
    name = tool_name.strip()
    normalized = name.lower()
    if normalized in taxonomy:
        item = dict(taxonomy[normalized])
        return AgentDojoToolInfo(
            tool_name=name,
            semantic_action=str(item["semantic_action"]),
            category=str(item["category"]),
            risk=str(item["risk"]),
            side_effect=bool(item.get("side_effect", item["risk"] in {"high", "critical"})),
            sensitive_args=tuple(str(x) for x in item.get("sensitive_args", ()) or ()),
            decision_hints=tuple(str(x) for x in item.get("decision_hints", ()) or ()),
            constraints=dict(item.get("constraints") or {}),
            registered=True,
        )

    args = tool_args or {}
    hint_text = " ".join([normalized, str(args).lower()])
    mutation_arg = any(key.lower() in {"amount", "recipient", "to", "email", "message", "body", "delete", "confirm"} for key in args)
    looks_side_effect = mutation_arg or any(token in hint_text for token in EXTERNAL_COMMIT_HINTS)
    if normalized.startswith(READ_ONLY_HINTS) and not looks_side_effect:
        return AgentDojoToolInfo(
            name,
            "read_context",
            "unknown_read_only",
            "low",
            False,
            tuple(),
            ("review",),
            {"unknown_tool": True},
            registered=False,
        )
    if looks_side_effect:
        return AgentDojoToolInfo(
            name,
            "unknown_side_effect",
            "unknown_side_effect",
            "critical",
            True,
            tuple(_sensitive_arg_hints(args)),
            ("sandbox_then_approval", "quarantine"),
            {"unknown_tool": True, "human_gate": "approval_required"},
            registered=False,
        )
    return AgentDojoToolInfo(
        name,
        "unknown_side_effect",
        "unknown_side_effect",
        "high",
        True,
        tuple(_sensitive_arg_hints(args)),
        ("sandbox_then_approval",),
        {"unknown_tool": True, "human_gate": "approval_required"},
        registered=False,
    )


def _normalize_taxonomy_entry(tool_name: str, item: dict[str, Any]) -> dict[str, Any]:
    side_effect = bool(item.get("side_effect", item.get("risk") in {"high", "critical"}))
    return {
        "tool_name": tool_name,
        "semantic_action": str(item.get("semantic_action") or "unknown_side_effect"),
        "category": str(item.get("category") or "unknown_side_effect"),
        "risk": str(item.get("risk") or ("critical" if side_effect else "low")),
        "side_effect": side_effect,
        "sensitive_args": [str(x) for x in item.get("sensitive_args", []) or []],
        "decision_hints": [str(x) for x in item.get("decision_hints", []) or []],
        "constraints": dict(item.get("constraints") or {}),
        "aliases": [str(x) for x in item.get("aliases", []) or []],
    }


def _sensitive_arg_hints(args: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    for key, value in args.items():
        key_low = str(key).lower()
        if key_low in {"password", "token", "secret", "apikey", "api_key", "amount", "recipient", "email", "message", "body"}:
            hints.append(key_low)
        if (
            isinstance(value, str)
            and len(value) > 0
            and any(token in value.lower() for token in ("attacker", "secret", "ignore previous", "transfer"))
        ):
            hints.append(key_low)
    return sorted(set(hints))


def _fallback_taxonomy() -> dict[str, dict[str, Any]]:
    return {
        "send_email": _normalize_taxonomy_entry(
            "send_email",
            {
                "semantic_action": "send_external_message",
                "category": "external_message_send",
                "risk": "high",
                "side_effect": True,
                "sensitive_args": ["recipient", "subject", "body"],
                "decision_hints": ["block_if_untrusted", "approval_if_private_data_seen"],
            },
        ),
        "transfer_money": _normalize_taxonomy_entry(
            "transfer_money",
            {
                "semantic_action": "financial_transaction",
                "category": "financial_commit",
                "risk": "critical",
                "side_effect": True,
                "sensitive_args": ["recipient", "amount", "subject"],
                "decision_hints": ["approval_required"],
            },
        ),
    }


READ_ONLY_HINTS = ("get_", "read_", "search_", "lookup", "list_", "open_")
EXTERNAL_COMMIT_HINTS = ("send_", "transfer", "pay_", "book_", "delete_", "update_", "create_", "submit_", "remove_", "invite_", "post_")


AGENTDOJO_TOOL_TAXONOMY = load_agentdojo_taxonomy()

