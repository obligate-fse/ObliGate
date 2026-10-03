"""Non-oracle task authorization for AgentDojo firewall decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compat.types import ToolCallContext, ToolSpec
from .entity_extractor import EntitySet, extract_entities

KEYWORDS_BY_GROUP = {
    "financial_commit": ("pay", "bill", "transfer", "send money", "payment", "refund", "schedule transaction"),
    "external_send": (
        "send",
        "post",
        "email",
        "message",
        "share",
        "notify",
        "reply",
        "dm",
        "direct message",
        "channel message",
        "slack",
    ),
    "booking_commit": ("book", "reserve", "reservation", "reserv", "booking", "hotel", "restaurant", "car rental", "flight"),
    "calendar_mutation": ("calendar", "event", "meeting", "schedule", "reschedule", "cancel", "remind", "reminder"),
    "workspace_mutation": ("file", "document", "cloud drive", "create", "append", "write", "update", "add"),
    "destructive_update": ("delete", "remove", "cancel"),
    "membership_mutation": ("invite", "add", "remove", "channel", "slack"),
    "credential_change": ("password", "credential"),
    "untrusted_read": (
        "webpage",
        "website",
        "link",
        "browse",
        "read content",
        "find information",
        "send information",
        "what kind",
        "details",
    ),
}

HIGH_RISK_GROUPS = {
    "financial_commit",
    "booking_commit",
    "external_send",
    "calendar_mutation",
    "workspace_mutation",
    "destructive_update",
    "membership_mutation",
    "credential_change",
}


@dataclass(slots=True)
class AuthorizationEvidence:
    task_authorized: bool
    group_authorized: bool
    args_match_user_entity: bool
    args_match_untrusted_entity: bool
    args_match_private_entity: bool
    sensitive_args_not_in_user_task: bool
    entity_authorization_confidence: float
    user_entities: EntitySet
    arg_entities: EntitySet
    reasons: list[str]


def authorize_tool(
    context: ToolCallContext,
    spec: ToolSpec,
    *,
    untrusted_entities: EntitySet | None = None,
    private_entities: EntitySet | None = None,
    low_trust_seen: bool = False,
) -> AuthorizationEvidence:
    # Only ordinary/public or untrusted-content reads take the simplified
    # path.  Private reads and unknown/version-drifted tools must establish
    # task and flow authority below, even in an otherwise clean session.
    if (
        not spec.side_effect
        and not spec.external_sink
        and spec.group != "unknown"
        and (not spec.read_private or spec.group == "untrusted_read")
    ):
        user_entities = extract_entities(context.user_task)
        arg_entities = extract_entities(context.tool_args)
        reason = "trusted_untrusted_content_read" if spec.group == "untrusted_read" else "public_read_only"
        return AuthorizationEvidence(
            True,
            True,
            True,
            False,
            False,
            False,
            1.0,
            user_entities,
            arg_entities,
            [reason],
        )

    task = _trusted_task_text(str(context.user_task or "")).lower()
    user_entities = extract_entities(task)
    arg_entities = extract_entities(context.tool_args)
    untrusted_entities = untrusted_entities or EntitySet()
    private_entities = private_entities or EntitySet()
    target_entities = _target_entities_for_group(spec.group, arg_entities)
    target_in_user_task = _target_entities_for_group(spec.group, user_entities)

    group_authorized = context.tool_name in context.allowed_tools or spec.group in context.allowed_groups
    if not group_authorized:
        group_authorized = any(keyword in task for keyword in KEYWORDS_BY_GROUP.get(spec.group, ()))
    private_read_task_match = bool(
        spec.group == "private_read"
        and _private_read_matches_task(context, spec, task)
    )
    if private_read_task_match:
        group_authorized = True

    args_match_user = _args_match_text(arg_entities, task)
    args_match_untrusted = _args_match_entities(arg_entities, untrusted_entities)
    args_match_private = _args_match_entities(arg_entities, private_entities)
    target_match_user = bool(
        target_entities.flattened()
        and (
            target_entities.flattened() & target_in_user_task.flattened()
            or _args_match_text(target_entities, task)
        )
    )
    target_mismatch = bool(target_entities.flattened() and target_in_user_task.flattened() and not target_match_user)
    mismatch = sensitive_args_not_in_user_task(context.tool_args, spec, task)

    if spec.group in {"booking_commit", "external_send", "financial_commit"} and not group_authorized:
        if args_match_user or target_match_user:
            group_authorized = True
    if spec.group == "external_send" and _external_send_target_authorized(spec, task, user_entities, arg_entities):
        group_authorized = True
    if spec.group == "booking_commit" and _booking_target_authorized(spec, task, user_entities, arg_entities):
        group_authorized = True
    if spec.group == "calendar_mutation" and _calendar_target_authorized(task, user_entities, arg_entities):
        group_authorized = True
    if spec.group == "untrusted_read" and spec.external_sink and args_match_user:
        group_authorized = True

    reasons: list[str] = []
    if group_authorized:
        reasons.append("tool_group_authorized_by_task")
    if private_read_task_match:
        reasons.append("private_read_object_matches_trusted_task")
    if args_match_user:
        reasons.append("arguments_match_user_task_entity")
    if target_match_user:
        reasons.append("tool_target_matches_user_task_entity")
    if args_match_untrusted:
        reasons.append("arguments_match_untrusted_entity")
    if mismatch:
        reasons.append("sensitive_arguments_absent_from_user_task")
    if target_mismatch:
        reasons.append("tool_target_mismatch_user_task_entity")

    high_risk = spec.group in HIGH_RISK_GROUPS
    if high_risk and spec.sensitive_args:
        if spec.group == "external_send":
            task_authorized = (
                group_authorized and (target_match_user or args_match_user) and not args_match_untrusted and not args_match_private
            )
        elif spec.group == "booking_commit":
            task_authorized = (
                group_authorized
                and (target_match_user or args_match_user or not arg_entities.flattened())
                and (target_match_user or not args_match_untrusted)
            )
        elif spec.group == "calendar_mutation":
            task_authorized = group_authorized and (
                _calendar_target_authorized(task, user_entities, arg_entities)
                or target_match_user
                or args_match_user
                or not arg_entities.flattened()
            ) and not args_match_untrusted
        else:
            task_authorized = group_authorized and (args_match_user or not arg_entities.flattened()) and not args_match_untrusted
    else:
        task_authorized = group_authorized
    if context.tool_name in context.allowed_tools or spec.group in context.allowed_groups:
        task_authorized = True
    if spec.group in {"booking_commit", "external_send"} and target_mismatch and context.tool_name not in context.allowed_tools:
        task_authorized = False

    confidence = 0.2
    if group_authorized:
        confidence += 0.35
    if private_read_task_match:
        confidence += 0.4
    if args_match_user:
        confidence += 0.35
    if target_match_user:
        confidence += 0.2
    if mismatch or args_match_untrusted:
        confidence -= 0.35
    if target_mismatch:
        confidence -= 0.2
    confidence = max(0.0, min(1.0, confidence))

    return AuthorizationEvidence(
        task_authorized=task_authorized,
        group_authorized=group_authorized,
        args_match_user_entity=args_match_user,
        args_match_untrusted_entity=args_match_untrusted,
        args_match_private_entity=args_match_private,
        sensitive_args_not_in_user_task=mismatch,
        entity_authorization_confidence=confidence,
        user_entities=user_entities,
        arg_entities=arg_entities,
        reasons=reasons,
    )


def task_authorizes_tool(context: ToolCallContext, spec: ToolSpec) -> bool:
    return authorize_tool(context, spec).task_authorized


def _trusted_task_text(text: str) -> str:
    lowered = text.casefold()
    markers = (
        "ignore previous instructions",
        "strictly adhere",
        "do not use other tools",
        "new instruction",
        "override",
        "forget previous",
    )
    positions = [lowered.find(marker) for marker in markers if marker in lowered]
    return text[: min(positions)] if positions else text


def _private_read_matches_task(
    context: ToolCallContext,
    spec: ToolSpec,
    trusted_task: str,
) -> bool:
    task_tokens = _semantic_tokens(trusted_task)
    tool_tokens = _semantic_tokens(
        f"{context.tool_name} {spec.semantic_action} {spec.description}"
    )
    generic = {
        "read",
        "search",
        "get",
        "list",
        "retrieve",
        "review",
        "private",
        "information",
        "data",
        "tool",
    }
    domain_tokens = tool_tokens - generic
    aliases_by_tool = {
        "search_emails": {"email", "inbox", "invoice", "message", "mail"},
        "read_file": {"file", "document", "bill", "attachment"},
        "read_inbox": {"email", "inbox", "message", "mail"},
        "get_file_by_id": {"file", "document", "recipe", "grocery", "list", "cloud", "drive"},
        # Financial reads are local context operations.  Bind each one to the
        # object named by the trusted task instead of treating every banking
        # read as interchangeable.  Taint is still carried forward and is
        # enforced if the result later reaches an external sink.
        "get_balance": {"balance", "funds"},
        "get_iban": {"iban", "account", "identifier"},
        "get_most_recent_transactions": {
            "transaction",
            "history",
            "recent",
            "spend",
            "spending",
            "spent",
            "purchase",
            "gift",
            "refund",
            "sent",
            "received",
        },
        "get_scheduled_transactions": {
            "scheduled",
            "transaction",
            "payment",
            "rent",
            "standing",
            "order",
        },
        "get_user_info": {"profile", "user", "personal", "account", "address"},
    }
    domain_tokens.update(aliases_by_tool.get(context.tool_name, set()))
    if domain_tokens & task_tokens:
        return True
    argument_tokens = _semantic_tokens(" ".join(_iter_text_values(context.tool_args)))
    return bool((argument_tokens - generic) & task_tokens)


def _semantic_tokens(text: str) -> set[str]:
    normalized = "".join(character if character.isalnum() else " " for character in text.casefold())
    tokens = set()
    for token in normalized.split():
        if len(token) < 3:
            continue
        tokens.add(token)
        if token.endswith("ies") and len(token) > 4:
            tokens.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _iter_text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _iter_text_values(nested)]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _iter_text_values(nested)]
    return [str(value)] if value is not None else []


def sensitive_args_not_in_user_task(args: dict[str, Any], spec: ToolSpec, user_task: str) -> bool:
    arg_entities = extract_entities({key: args.get(key) for key in spec.sensitive_args if key in args})
    if not arg_entities.flattened():
        return False
    return not _args_match_text(arg_entities, user_task)


def _args_match_text(arg_entities: EntitySet, text: str) -> bool:
    if not arg_entities.flattened():
        return False
    return arg_entities.matches_text(text)


def _args_match_entities(arg_entities: EntitySet, haystack: EntitySet) -> bool:
    if not arg_entities.flattened() or not haystack.flattened():
        return False
    return any(normalized in haystack.flattened() for normalized in arg_entities.flattened())


def _target_entities_for_group(group: str, entities: EntitySet) -> EntitySet:
    target = EntitySet()
    if group == "booking_commit":
        for kind in ("hotel", "restaurant", "company", "flight", "city"):
            for value in entities.values_for(kind):
                target.add(kind, value)
        return target
    if group == "external_send":
        for kind in ("recipient", "email", "channel", "user"):
            for value in entities.values_for(kind):
                target.add(kind, value)
        return target
    for kind in ("recipient", "email", "channel", "hotel", "restaurant", "company", "flight", "user"):
        for value in entities.values_for(kind):
            target.add(kind, value)
    return target


def _booking_target_authorized(spec: ToolSpec, user_task: str, user_entities: EntitySet, arg_entities: EntitySet) -> bool:
    if spec.group != "booking_commit":
        return False
    user_targets = _target_entities_for_group("booking_commit", user_entities)
    arg_targets = _target_entities_for_group("booking_commit", arg_entities)
    if arg_targets.flattened() and user_targets.flattened() and arg_targets.flattened() & user_targets.flattened():
        return True
    if user_targets.flattened() and _args_match_text(arg_targets, user_task):
        return True
    return False


def _external_send_target_authorized(spec: ToolSpec, user_task: str, user_entities: EntitySet, arg_entities: EntitySet) -> bool:
    if spec.group != "external_send":
        return False
    user_targets = _target_entities_for_group("external_send", user_entities)
    arg_targets = _target_entities_for_group("external_send", arg_entities)
    if arg_targets.flattened() and user_targets.flattened() and arg_targets.flattened() & user_targets.flattened():
        return True
    if arg_targets.flattened() and _args_match_text(arg_targets, user_task):
        return True
    if spec.name == "send_channel_message":
        channel_values = arg_entities.values_for("channel")
        if channel_values and any(value in user_task.lower() for value in channel_values):
            return True
    if spec.name in {"send_direct_message", "send_email", "send_slack_message"}:
        recipient_values = arg_entities.values_for("recipient") | arg_entities.values_for("email") | arg_entities.values_for("user")
        if recipient_values and any(value in user_task.lower() for value in recipient_values):
            return True
    return False


def _calendar_target_authorized(user_task: str, user_entities: EntitySet, arg_entities: EntitySet) -> bool:
    user_targets = _target_entities_for_group("calendar_mutation", user_entities)
    arg_targets = _target_entities_for_group("calendar_mutation", arg_entities)
    if arg_targets.flattened() and user_targets.flattened() and arg_targets.flattened() & user_targets.flattened():
        return True
    if arg_targets.flattened() and _args_match_text(arg_targets, user_task):
        return True
    if any(word in user_task.lower() for word in ("remind", "reminder", "calendar", "event", "schedule")):
        return True
    return False


