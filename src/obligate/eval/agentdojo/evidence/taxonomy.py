"""AgentDojo tool taxonomy.

This mapping is based on AgentDojo default-suite tools.  It classifies tools by
security effect, not by attack name.  This is important: the firewall should not
memorize particular injection prompts; it should understand whether a tool is a
read, an external send, a transaction, a booking, or a destructive mutation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..compat.types import ToolGroup, ToolSpec

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class AgentDojoToolTaxonomy:
    def __init__(self, specs: dict[str, ToolSpec] | None = None) -> None:
        self.specs = specs or default_specs()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentDojoToolTaxonomy":
        if yaml is None:
            raise RuntimeError("PyYAML is required for loading taxonomy YAML")
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        specs = default_specs()
        for name, item in (data.get("tools") or data).items():
            specs[str(name)] = ToolSpec(
                name=str(name),
                semantic_action=str(item.get("semantic_action", name)),
                group=str(item.get("group", "unknown")),  # type: ignore[arg-type]
                risk=str(item.get("risk", "low")),  # type: ignore[arg-type]
                side_effect=bool(item.get("side_effect", False)),
                read_private=bool(item.get("read_private", False)),
                read_untrusted=bool(item.get("read_untrusted", False)),
                external_sink=bool(item.get("external_sink", False)),
                state_mutation=bool(item.get("state_mutation", False)),
                sensitive_args=list(item.get("sensitive_args", [])),
                decision_hints=list(item.get("decision_hints", [])),
                description=str(item.get("description", "")),
            )
        return cls(specs)

    def classify(self, tool_name: str, *, suite: str | None = None) -> ToolSpec:
        suite_key = f"{suite}.{tool_name}" if suite else None
        if suite_key and suite_key in self.specs:
            return self.specs[suite_key]
        if tool_name in self.specs:
            return self.specs[tool_name]
        return infer_unknown_tool(tool_name)

    def coverage(self, tool_names: list[str], *, suite: str | None = None) -> dict[str, Any]:
        unknown: list[str] = []
        for name in tool_names:
            if self.classify(name, suite=suite).group == "unknown":
                unknown.append(name)
        total = len(tool_names)
        return {
            "total": total,
            "registered": total - len(unknown),
            "unknown": len(unknown),
            "registered_rate": 0.0 if total == 0 else (total - len(unknown)) / total,
            "unknown_rate": 0.0 if total == 0 else len(unknown) / total,
            "unknown_tools": sorted(set(unknown)),
        }


def spec(
    name: str,
    semantic_action: str,
    group: ToolGroup,
    risk: str = "low",
    *,
    side_effect: bool = False,
    read_private: bool = False,
    read_untrusted: bool = False,
    external_sink: bool = False,
    state_mutation: bool = False,
    sensitive_args: list[str] | None = None,
    decision_hints: list[str] | None = None,
    description: str = "",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        semantic_action=semantic_action,
        group=group,
        risk=risk,  # type: ignore[arg-type]
        side_effect=side_effect,
        read_private=read_private,
        read_untrusted=read_untrusted,
        external_sink=external_sink,
        state_mutation=state_mutation,
        sensitive_args=sensitive_args or [],
        decision_hints=decision_hints or [],
        description=description,
    )


def default_specs() -> dict[str, ToolSpec]:
    tools: dict[str, ToolSpec] = {}

    def add(item: ToolSpec) -> None:
        tools[item.name] = item

    # Banking.
    for name in ["get_iban", "get_balance", "get_most_recent_transactions", "get_scheduled_transactions", "get_user_info"]:
        add(spec(name, "read_private_financial_data", "private_read", "medium", read_private=True))
    add(spec("read_file", "read_private_file", "private_read", "medium", read_private=True, read_untrusted=True))
    for name in ["send_money", "schedule_transaction", "update_scheduled_transaction"]:
        add(
            spec(
                name,
                "financial_transaction",
                "financial_commit",
                "critical",
                side_effect=True,
                state_mutation=True,
                sensitive_args=["recipient", "amount", "iban", "account", "message"],
                decision_hints=["approval_required"],
            )
        )
    add(
        spec(
            "update_password",
            "credential_change",
            "credential_change",
            "critical",
            side_effect=True,
            state_mutation=True,
            sensitive_args=["password", "token"],
        )
    )
    add(
        spec(
            "update_user_info",
            "profile_mutation",
            "workspace_mutation",
            "high",
            side_effect=True,
            state_mutation=True,
            sensitive_args=["email", "phone", "address"],
        )
    )

    # Workspace.
    for name in ["get_unread_emails", "get_sent_emails", "get_received_emails", "get_draft_emails", "search_emails"]:
        add(spec(name, "read_private_messages", "private_read", "low", read_private=True, read_untrusted=True))
    for name in ["search_contacts_by_name", "search_contacts_by_email", "get_file_by_id", "search_files", "search_files_by_filename"]:
        add(spec(name, "read_private_workspace_data", "private_read", "low", read_private=True))
    add(
        spec(
            "send_email",
            "send_external_message",
            "external_send",
            "high",
            side_effect=True,
            external_sink=True,
            sensitive_args=["to", "recipient", "body", "message", "subject"],
        )
    )
    add(spec("delete_email", "delete_email", "destructive_update", "high", side_effect=True, state_mutation=True))
    for name in ["get_current_day", "search_calendar_events", "get_day_calendar_events", "list_files"]:
        add(spec(name, "read_workspace_metadata", "read_only", "low"))
    for name in ["create_calendar_event", "cancel_calendar_event", "reschedule_calendar_event", "add_calendar_event_participants"]:
        add(
            spec(
                name,
                "calendar_mutation",
                "calendar_mutation",
                "high",
                side_effect=True,
                state_mutation=True,
                sensitive_args=["participants", "email", "date", "time"],
            )
        )
    for name in ["create_file", "append_to_file"]:
        add(spec(name, "write_workspace_file", "workspace_mutation", "medium", side_effect=True, state_mutation=True))
    add(
        spec(
            "share_file",
            "share_workspace_file",
            "external_send",
            "high",
            side_effect=True,
            external_sink=True,
            read_private=True,
            sensitive_args=["email", "recipient", "file_id"],
        )
    )
    add(spec("delete_file", "delete_workspace_file", "destructive_update", "high", side_effect=True, state_mutation=True))

    # Slack.
    for name in ["get_channels", "get_users_in_channel"]:
        add(spec(name, "read_slack_metadata", "read_only", "low"))
    add(spec("read_channel_messages", "read_untrusted_messages", "untrusted_read", "low", read_untrusted=True, read_private=True))
    add(spec("read_inbox", "read_private_messages", "private_read", "low", read_private=True, read_untrusted=True))
    for name in ["send_direct_message", "send_channel_message", "send_slack_message"]:
        add(
            spec(
                name,
                "send_external_message",
                "external_send",
                "high",
                side_effect=True,
                external_sink=True,
                sensitive_args=["recipient", "channel", "message", "body"],
            )
        )
    for name in ["add_user_to_channel", "invite_user_to_slack"]:
        add(spec(name, "modify_slack_membership", "membership_mutation", "high", side_effect=True, state_mutation=True))
    add(spec("remove_user_from_slack", "remove_slack_user", "destructive_update", "critical", side_effect=True, state_mutation=True))
    add(
        spec(
            "get_webpage",
            "network_fetch_untrusted_web",
            "untrusted_read",
            "medium",
            read_untrusted=True,
            external_sink=True,
            sensitive_args=["url"],
        )
    )
    add(
        spec(
            "post_webpage",
            "post_external_webpage",
            "external_send",
            "high",
            side_effect=True,
            external_sink=True,
            sensitive_args=["url", "content", "message"],
        )
    )

    # Travel.
    add(spec("get_user_information", "read_user_booking_profile", "private_read", "medium", read_private=True))
    for name in [
        "get_all_hotels_in_city",
        "get_hotels_prices",
        "get_hotels_address",
        "get_all_restaurants_in_city",
        "get_cuisine_type_for_restaurants",
        "get_restaurants_address",
        "get_dietary_restrictions_for_all_restaurants",
        "get_contact_information_for_restaurants",
        "get_price_for_restaurants",
        "check_restaurant_opening_hours",
        "get_all_car_rental_companies_in_city",
        "get_car_types_available",
        "get_car_fuel_options",
        "get_car_rental_address",
        "get_car_price_per_day",
        "get_flight_information",
    ]:
        add(spec(name, "read_travel_option", "read_only", "low"))
    for name in ["get_rating_reviews_for_hotels", "get_rating_reviews_for_restaurants", "get_rating_reviews_for_car_rental"]:
        add(spec(name, "read_untrusted_review", "untrusted_read", "low", read_untrusted=True))
    add(
        spec(
            "reserve_hotel",
            "booking_commit",
            "booking_commit",
            "critical",
            side_effect=True,
            state_mutation=True,
            sensitive_args=["hotel", "city", "date", "price"],
        )
    )
    add(
        spec(
            "reserve_car_rental",
            "booking_commit",
            "booking_commit",
            "critical",
            side_effect=True,
            state_mutation=True,
            sensitive_args=["company", "city", "date", "price"],
        )
    )
    add(
        spec(
            "reserve_restaurant",
            "booking_commit",
            "booking_commit",
            "high",
            side_effect=True,
            state_mutation=True,
            sensitive_args=["restaurant", "city", "date", "time"],
        )
    )

    # Workspace plus-like tools.
    for name in ["run_bash_command", "terminal", "run_command", "shell_command", "execute_command"]:
        add(
            spec(
                name,
                "shell_command",
                "terminal_or_code",
                "critical",
                side_effect=True,
                state_mutation=True,
                sensitive_args=["command", "cmd", "script"],
            )
        )

    return tools


def infer_unknown_tool(tool_name: str) -> ToolSpec:
    lowered = tool_name.lower()
    side_words = (
        "send",
        "post",
        "delete",
        "remove",
        "update",
        "create",
        "reserve",
        "book",
        "schedule",
        "share",
        "invite",
        "add",
        "transfer",
        "pay",
        "run",
        "execute",
    )
    read_words = ("get", "read", "search", "list", "check", "lookup", "find")
    if lowered.startswith(read_words):
        return spec(tool_name, "unknown_read", "unknown", "low", description="inferred read-only unknown tool")
    if any(word in lowered for word in side_words):
        return spec(
            tool_name,
            "unknown_side_effect",
            "unknown",
            "high",
            side_effect=True,
            state_mutation=True,
            description="inferred side-effect unknown tool",
        )
    return spec(tool_name, "unknown_tool", "unknown", "medium", description="unclassified AgentDojo tool")


