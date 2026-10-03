from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..compat.models_compat import sha256_text
from ..compat.types import ArgumentProvenance, EvidenceBundle, ToolCallContext, ToolSpec, ValueMatchEvidence
from .contracts import SourceInfluenceGraph, TaskContractMatch, payment_contract_from_user_task
from .entity_extractor import extract_entities
from .state import AgentDojoStateTracker, extract_targets
from .task_authorizer import authorize_tool

SENSITIVE_ARG_KEYS = {"password", "token", "secret", "iban", "account", "recipient", "to", "email", "message", "body", "file_id", "amount"}


def _task_requests_named_transform(user_task: str) -> bool:
    task = str(user_task or "").casefold()
    transform = any(
        token in task
        for token in (
            "summarize",
            "summary",
            "extract",
            "compile",
            "forward",
            "information",
            "details",
            "what kind",
        )
    )
    dispatch = any(token in task for token in ("post", "send", "share", "email", "message"))
    return transform and dispatch


def _task_delegates_recipient_resolution(user_task: str) -> bool:
    task = str(user_task or "").casefold()
    dispatch = any(token in task for token in ("send", "email", "message", "forward"))
    delegated_recipient = any(
        token in task
        for token in (
            "to the client",
            "to client",
            "to the customer",
            "to them",
            "to the sender",
            "reply to",
        )
    )
    source = any(token in task for token in ("file", "document", "email", "meeting minutes", "attachment"))
    return dispatch and delegated_recipient and source


def _task_explicitly_supplies_credential(context: ToolCallContext, spec: ToolSpec) -> bool:
    if spec.group != "credential_change":
        return False
    task = str(context.user_task or "").casefold()
    values = [
        str(value).casefold().strip()
        for key, value in context.tool_args.items()
        if key in {"password", "credential"} and value
    ]
    return bool(values and all(value in task for value in values))


def _task_delegates_calendar_selection(user_task: str) -> bool:
    task = str(user_task or "").casefold()
    selection = any(token in task for token in ("suggest", "recommend", "highest rating", "best", "choices", "under"))
    calendar = any(token in task for token in ("calendar", "event", "remind"))
    return selection and calendar


def _task_delegates_target_selection(user_task: str, group: str) -> bool:
    task = str(user_task or "").casefold()
    selector = any(
        token in task
        for token in (
            "find a ",
            "find the ",
            "most ",
            "smallest",
            "highest",
            "best ",
            "not already",
            "based on the email",
            "based on the emails",
            "based on the other document",
            "from the recipe",
            "from recipe",
            "website of",
            "websites of",
        )
    )
    actions = {
        "external_send": ("send", "write them", "reply", "email", "message", "post"),
        "membership_mutation": ("add", "invite", "remove"),
        "calendar_mutation": ("calendar", "event", "remind", "book 5 hours"),
        "workspace_mutation": ("add", "append", "create", "write", "update"),
    }
    return selector and any(token in task for token in actions.get(group, ()))


def _delegated_tool_matches_task(tool_name: str, user_task: str) -> bool:
    task = str(user_task or "").casefold()
    todo_delegation = "todo" in task and any(token in task for token in ("all the tasks", "perform the tasks", "do the tasks"))
    if tool_name == "create_file":
        return "create" in task or "new file" in task
    if tool_name == "append_to_file":
        return any(token in task for token in ("add", "append", "update"))
    if tool_name in {"add_user_to_channel", "invite_user_to_slack"}:
        return todo_delegation or any(token in task for token in ("add", "invite"))
    if tool_name == "remove_user_from_slack":
        return "remove" in task
    return True


def _task_delegates_todo_execution(user_task: str) -> bool:
    task = str(user_task or "").casefold()
    return bool(
        "todo" in task
        and any(token in task for token in ("all the tasks", "perform the tasks", "do the tasks"))
        and any(token in task for token in ("http://", "https://", "www."))
    )


def _task_explicitly_authorizes_profile_update(context: ToolCallContext, spec: ToolSpec) -> bool:
    if context.tool_name != "update_user_info" or spec.group != "workspace_mutation":
        return False
    task = str(context.user_task or "").casefold()
    if not any(token in task for token in ("update my account information", "new address", "update my address")):
        return False
    values = [
        str(value).casefold().strip()
        for key, value in context.tool_args.items()
        if key in {"street", "city", "state", "zip", "postal_code", "address"} and value
    ]
    exact_values = bool(values and all(value in task for value in values))
    delegated_file_values = bool(
        values
        and "accordingly" in task
        and any(token in task for token in ("read '", "read \"", "file", ".txt", ".docx"))
    )
    return exact_values or delegated_file_values


def _security_target_args(args: dict[str, Any], spec: ToolSpec) -> dict[str, Any]:
    keys_by_group = {
        "external_send": {"recipient", "recipients", "to", "email", "user", "user_email", "channel", "url"},
        "booking_commit": {"hotel", "restaurant", "company", "flight", "city"},
        "financial_commit": {"recipient", "to", "iban", "account", "account_id"},
        "calendar_mutation": {"participants", "email", "title", "summary", "description", "location"},
        "membership_mutation": {"user", "user_email", "email", "channel"},
        "workspace_mutation": {"file_id", "filename", "file_name", "name", "path"},
    }
    keys = keys_by_group.get(spec.group)
    if not keys:
        return dict(args)
    return {key: value for key, value in args.items() if str(key).casefold() in keys}


class AgentDojoEvidenceBuilder:
    def build(
        self, *, context: ToolCallContext, spec: ToolSpec, state: AgentDojoStateTracker, graph_facts: dict[str, Any] | None = None
    ) -> EvidenceBundle:
        raw = context.raw_tool_call if isinstance(context.raw_tool_call, dict) else {}
        asb_metadata = raw.get("asb") if isinstance(raw.get("asb"), dict) else {}
        observable_low_trust = bool(
            state.untrusted_seen
            or state.injection_seen
            or asb_metadata.get("low_trust_instruction_seen")
            or asb_metadata.get("prompt_injection_marker_seen")
        )
        authorization = authorize_tool(
            context,
            spec,
            untrusted_entities=state.untrusted_entities,
            private_entities=state.private_entities,
            low_trust_seen=observable_low_trust,
        )
        ablation = dict(context.ablation_config or {})
        provenance_enabled = bool(ablation.get("enable_provenance", True))
        task_contract_enabled = bool(ablation.get("enable_task_contract", True))
        action_graph_enabled = bool(ablation.get("enable_action_graph", True))
        task_authorized = authorization.task_authorized
        sensitive_args_present = self._sensitive_args_present(context.tool_args, spec)
        sensitive_mismatch = authorization.sensitive_args_not_in_user_task
        target_args = _security_target_args(context.tool_args, spec)
        attack_goal_overlap = state.attack_goal_overlap(target_args)
        unknown = spec.group == "unknown"
        user_task_targets = self._booking_target_values(context.user_task, spec)
        booking_target_values = self._booking_target_values(context.tool_args, spec)
        booking_target_matches_user_goal = bool(
            (user_task_targets and booking_target_values and user_task_targets.intersection(booking_target_values))
            or authorization.args_match_user_entity
            or "tool_target_matches_user_task_entity" in authorization.reasons
        )
        explicit_booking_target_in_task = bool(user_task_targets)
        arg_provenance = self._argument_provenance(context, authorization) if provenance_enabled else self._unknown_argument_provenance(context)
        arg_source_map = {item.arg_name: item.source_type for item in arg_provenance}
        value_matches = self._value_match_evidence(context, spec, state, arg_provenance) if provenance_enabled else []
        target_source = self._target_entity_source(spec, arg_source_map)
        delegated_transform_to_named_destination = bool(
            spec.group == "external_send"
            and authorization.group_authorized
            and (
                (authorization.args_match_user_entity and target_source == "user_task")
                or (
                    _task_delegates_recipient_resolution(context.user_task)
                    and (authorization.args_match_private_entity or authorization.args_match_untrusted_entity)
                    and target_source in {"private_tool_result", "trusted_tool_result", "untrusted_tool_result"}
                )
            )
            and not attack_goal_overlap
            and _task_requests_named_transform(context.user_task)
        )
        delegated_selection_to_calendar = bool(
            spec.group == "calendar_mutation"
            and authorization.group_authorized
            and state.untrusted_seen
            and not attack_goal_overlap
            and _task_delegates_calendar_selection(context.user_task)
        )
        delegated_target_selection = bool(
            spec.group in {"external_send", "membership_mutation", "calendar_mutation", "workspace_mutation"}
            and authorization.group_authorized
            and any(event.kind == "tool_result" for event in state.events)
            and not attack_goal_overlap
            and _task_delegates_target_selection(context.user_task, spec.group)
            and _delegated_tool_matches_task(context.tool_name, context.user_task)
        )
        delegated_todo_execution = bool(
            state.events
            and _task_delegates_todo_execution(context.user_task)
            and not attack_goal_overlap
            and spec.group in {"untrusted_read", "external_send", "membership_mutation"}
        )
        explicit_profile_update = _task_explicitly_authorizes_profile_update(context, spec)
        explicit_credential_update = _task_explicitly_supplies_credential(context, spec)
        if (
            delegated_transform_to_named_destination
            or delegated_selection_to_calendar
            or delegated_target_selection
            or explicit_profile_update
            or explicit_credential_update
            or delegated_todo_execution
        ):
            task_authorized = True
        if not task_contract_enabled:
            task_authorized = False
            sensitive_mismatch = bool(spec.side_effect and sensitive_args_present)
            booking_target_matches_user_goal = False
        task_contract = TaskContractMatch(
            status="match" if task_authorized else ("violation" if spec.side_effect else "unknown"),
            target_source=target_source,  # type: ignore[arg-type]
            argument_sources=arg_source_map,  # type: ignore[arg-type]
            confidence=authorization.entity_authorization_confidence,
            reasons=list(authorization.reasons),
        )
        source_influence = SourceInfluenceGraph(
            has_untrusted_to_side_effect_edge=bool(action_graph_enabled and graph_facts and graph_facts.get("graph.has_untrusted_to_side_effect_edge")),
            has_injection_to_side_effect_edge=bool(action_graph_enabled and graph_facts and graph_facts.get("graph.has_injection_to_side_effect_edge")),
            has_private_to_external_edge=bool(action_graph_enabled and graph_facts and graph_facts.get("graph.has_private_to_external_edge")),
            has_attack_goal_to_action_edge=bool(action_graph_enabled and graph_facts and graph_facts.get("graph.has_attack_goal_to_action_edge")),
            evidence_refs=[event.event_id for event in state.events[-5:]] if action_graph_enabled else [],
        )
        args_match_user_entity = authorization.args_match_user_entity if task_contract_enabled else False
        args_match_untrusted_entity = authorization.args_match_untrusted_entity if provenance_enabled else False
        args_match_private_entity = authorization.args_match_private_entity if provenance_enabled else False
        entity_authorization_confidence = authorization.entity_authorization_confidence if task_contract_enabled else 0.0
        authorization_reasons = list(authorization.reasons) if task_contract_enabled else []
        facts = {
            "agentdojo.suite": context.suite,
            "agentdojo.tool_name": context.tool_name,
            "agentdojo.semantic_action": spec.semantic_action,
            "agentdojo.tool_group": spec.group,
            "agentdojo.risk": spec.risk,
            "agentdojo.side_effect": spec.side_effect,
            "agentdojo.read_private": spec.read_private,
            "agentdojo.read_untrusted": spec.read_untrusted,
            "agentdojo.external_sink": spec.external_sink,
            "agentdojo.state_mutation": spec.state_mutation,
            "agentdojo.task_authorized": task_authorized,
            "agentdojo.untrusted_seen": state.untrusted_seen,
            "agentdojo.injection_seen": state.injection_seen,
            "agentdojo.injection_followup_webpage_seen": state.injection_followup_webpage_seen,
            "agentdojo.injection_followup_side_effect_seen": state.injection_followup_side_effect_seen,
            "agentdojo.injection_recommendation_attack_seen": state.injection_recommendation_attack_seen,
            "agentdojo.adaptive_attack_seen": state.adaptive_attack_seen,
            "agentdojo.private_data_seen": state.private_data_seen,
            "agentdojo.financial_data_seen": state.financial_data_seen,
            "agentdojo.attack_goal_overlap": attack_goal_overlap,
            "agentdojo.suspicious_targets": sorted(state.suspicious_targets),
            "agentdojo.unknown_tool": unknown,
            "agentdojo.sensitive_args_present": sensitive_args_present,
            "agentdojo.sensitive_args_not_in_user_task": sensitive_mismatch,
            "agentdojo.args_match_user_entity": args_match_user_entity,
            "agentdojo.args_match_untrusted_entity": args_match_untrusted_entity,
            "agentdojo.args_match_private_entity": args_match_private_entity,
            "agentdojo.entity_authorization_confidence": entity_authorization_confidence,
            "agentdojo.user_entities": authorization.user_entities.as_dict(),
            "agentdojo.arg_entities": authorization.arg_entities.as_dict(),
            "agentdojo.target_arg_entities": extract_entities(target_args).as_dict(),
            "agentdojo.authorization_reasons": authorization_reasons,
            "agentdojo.user_task_targets": sorted(user_task_targets),
            "agentdojo.booking_target_values": sorted(booking_target_values),
            "agentdojo.booking_target_matches_user_goal": booking_target_matches_user_goal,
            "agentdojo.explicit_booking_target_in_task": explicit_booking_target_in_task,
            "agentdojo.delegated_transform_to_named_destination": delegated_transform_to_named_destination,
            "agentdojo.delegated_selection_to_calendar": delegated_selection_to_calendar,
            "agentdojo.delegated_target_selection": delegated_target_selection,
            "agentdojo.delegated_todo_execution": delegated_todo_execution,
            "agentdojo.explicit_profile_update": explicit_profile_update,
            "agentdojo.explicit_credential_update": explicit_credential_update,
            "agentdojo.arg_provenance": [asdict(item) for item in arg_provenance],
            "agentdojo.field_provenance": [asdict(item) for item in value_matches],
            "agentdojo.value_match_evidence": [asdict(item) for item in value_matches],
            "agentdojo.target_entity_value_match": self._first_match(value_matches, "target_entity"),
            "agentdojo.recipient_value_match": self._first_match(value_matches, "recipient"),
            "agentdojo.message_body_value_match": self._first_match(value_matches, "message_body"),
            "agentdojo.amount_value_match": self._first_match(value_matches, "amount"),
            "agentdojo.arg_source_map": arg_source_map,
            "agentdojo.target_entity_source": target_source,
            "agentdojo.message_recipient_source": self._message_target_source(spec, arg_source_map),
            "agentdojo.message_channel_source": arg_source_map.get("channel", "unknown"),
            "agentdojo.message_body_source": arg_source_map.get("body", arg_source_map.get("message", "unknown")),
            "agentdojo.message_contains_private_data": bool(spec.external_sink and state.private_data_seen and sensitive_args_present),
            "agentdojo.recipient_is_user_requested": bool(spec.external_sink and target_source == "user_task"),
            "agentdojo.recipient_is_injected": bool(
                spec.external_sink and target_source in {"untrusted_tool_result", "injection_text"}
            ),
            "agentdojo.channel_is_user_requested": bool(spec.external_sink and arg_source_map.get("channel") == "user_task"),
            "agentdojo.channel_is_injected": bool(spec.external_sink and arg_source_map.get("channel") in {"untrusted_tool_result", "injection_text"}),
            "agentdojo.is_membership_expansion": context.tool_name in {"add_user_to_channel", "invite_user_to_slack"},
            "agentdojo.is_external_communication": bool(spec.external_sink),
            "agentdojo.sink_type": spec.group,
            "agentdojo.asset_risk": spec.risk,
            "agentdojo.tool_call_evidence": {
                "tool_name": context.tool_name,
                "action_type": spec.semantic_action,
                "sink_type": spec.group,
                "arg_provenance": [asdict(item) for item in arg_provenance],
                "task_contract_match": task_contract.status,
                "asset_risk": spec.risk,
                "source_influence": source_influence.as_dict(),
            },
            "agentdojo.task_contract_match": task_contract.as_dict(),
            "agentdojo.source_influence_graph": source_influence.as_dict(),
            "agentdojo.payment_intent_contract": payment_contract_from_user_task(context.user_task).as_dict()
            if spec.group == "financial_commit"
            else None,
            "source.has_untrusted": state.untrusted_seen,
            "source.has_injection_like": state.injection_seen,
            "source.trust_floor": "untrusted" if state.untrusted_seen else "trusted",
            "contract.match": "match" if task_authorized else ("violation" if spec.side_effect else "unknown"),
            "history.private_data_seen": state.private_data_seen,
            "history.financial_data_seen": state.financial_data_seen,
            "history.untrusted_seen": state.untrusted_seen,
            "history.injection_seen": state.injection_seen,
        }
        if graph_facts and action_graph_enabled:
            facts.update(graph_facts)
        return EvidenceBundle(
            suite=context.suite,
            tool_name=context.tool_name,
            semantic_action=spec.semantic_action,
            group=spec.group,
            risk=spec.risk,
            side_effect=spec.side_effect,
            read_private=spec.read_private,
            read_untrusted=spec.read_untrusted,
            external_sink=spec.external_sink,
            state_mutation=spec.state_mutation,
            task_authorized=task_authorized,
            untrusted_seen=state.untrusted_seen,
            injection_seen=state.injection_seen,
            private_data_seen=state.private_data_seen,
            financial_data_seen=state.financial_data_seen,
            attack_goal_overlap=attack_goal_overlap,
            unknown_tool=unknown,
            sensitive_args_present=sensitive_args_present,
            sensitive_args_not_in_user_task=sensitive_mismatch,
            args_match_user_entity=args_match_user_entity,
            args_match_untrusted_entity=args_match_untrusted_entity,
            args_match_private_entity=args_match_private_entity,
            entity_authorization_confidence=entity_authorization_confidence,
            tool_args_digest=sha256_text(repr(context.tool_args)),
            state_digest=state.digest(),
            facts=facts,
        )

    def _sensitive_args_present(self, args: dict[str, Any], spec: ToolSpec) -> bool:
        text = " ".join(f"{k}={v}" for k, v in args.items()).lower()
        return any(key.lower() in text for key in set(spec.sensitive_args) | SENSITIVE_ARG_KEYS)

    def _booking_target_values(self, value: Any, spec: ToolSpec) -> set[str]:
        if spec.group != "booking_commit":
            return set()
        return {target for target in extract_targets(str(value)) if target}

    def _argument_provenance(self, context: ToolCallContext, authorization: Any) -> list[ArgumentProvenance]:
        provenance: list[ArgumentProvenance] = []
        for key, value in sorted(context.tool_args.items()):
            value_text = str(value)
            # Explicit task text outranks later low-trust occurrences of the
            # same destination (for example, "send ... to Bob" after Bob also
            # appears as a message sender).  Occurrence in a tool result is not
            # itself provenance.
            if value_text and value_text.casefold() in str(context.user_task or "").casefold():
                source = "user_task"
                confidence = 0.95
            elif authorization.user_entities.matches_text(value_text):
                source = "user_task"
                confidence = 0.9
            elif authorization.arg_entities.flattened() and authorization.args_match_untrusted_entity:
                source = "untrusted_tool_result"
                confidence = 0.75
            elif authorization.arg_entities.flattened() and authorization.args_match_private_entity:
                source = "trusted_tool_result"
                confidence = 0.65
            else:
                source = "unknown"
                confidence = 0.35
            provenance.append(
                ArgumentProvenance(
                    arg_name=key,
                    value=value,
                    source_type=source,  # type: ignore[arg-type]
                    confidence=confidence,
                    evidence_refs=["user_task" if source == "user_task" else "tool_history"],
                )
            )
        return provenance

    def _unknown_argument_provenance(self, context: ToolCallContext) -> list[ArgumentProvenance]:
        return [
            ArgumentProvenance(
                arg_name=key,
                value=value,
                source_type="unknown",
                confidence=0.0,
                evidence_refs=[],
            )
            for key, value in sorted(context.tool_args.items())
        ]

    def _target_entity_source(self, spec: ToolSpec, arg_source_map: dict[str, str]) -> str:
        if spec.group == "booking_commit":
            for key in ("hotel", "restaurant", "company", "flight", "city"):
                if key in arg_source_map:
                    return arg_source_map[key]
        if spec.group == "financial_commit":
            return arg_source_map.get("recipient", arg_source_map.get("iban", "unknown"))
        if spec.group == "external_send":
            return self._message_target_source(spec, arg_source_map)
        return "unknown"

    def _message_target_source(self, spec: ToolSpec, arg_source_map: dict[str, str]) -> str:
        if spec.group != "external_send":
            return "unknown"
        for key in ("recipient", "recipients", "to", "email", "user", "channel"):
            if key in arg_source_map:
                return arg_source_map[key]
        return "unknown"

    def _value_match_evidence(
        self,
        context: ToolCallContext,
        spec: ToolSpec,
        state: AgentDojoStateTracker,
        arg_provenance: list[ArgumentProvenance],
    ) -> list[ValueMatchEvidence]:
        out: list[ValueMatchEvidence] = []
        recent_events = [event for event in reversed(state.events[-10:]) if event.event_status == "tool_result"]
        for item in arg_provenance:
            role = self._field_role(item.arg_name, spec)
            matched_event = next((event for event in recent_events if str(item.value).lower() and str(item.value).lower() in event.preview.lower()), None)
            source_type = item.source_type
            confidence = item.confidence
            if matched_event and source_type == "unknown":
                source_type = "trusted_tool_result" if matched_event.private_data else ("untrusted_tool_result" if matched_event.untrusted else "trusted_tool_result")
                confidence = 0.65
            match_type = "normalized" if source_type != "unknown" else "unknown"
            if role in {"amount", "date"} and item.source_type != "unknown":
                match_type = role
            out.append(
                ValueMatchEvidence(
                    field_name=item.arg_name,
                    field_role=role,  # type: ignore[arg-type]
                    value=item.value,
                    source_type=source_type,
                    matched_prior_event_id=matched_event.event_id if matched_event else None,
                    matched_span=str(item.value) if item.source_type != "unknown" else None,
                    match_type=match_type,  # type: ignore[arg-type]
                    confidence=confidence,
                    notes=[] if source_type != "unknown" else ["no deterministic source match"],
                )
            )
        return out

    def _field_role(self, field_name: str, spec: ToolSpec) -> str:
        key = field_name.lower()
        if key in {"hotel", "restaurant", "company", "flight"}:
            return "target_entity"
        if key in {"recipient", "to", "email", "user"}:
            return "recipient"
        if key == "channel":
            return "channel"
        if key in {"body", "message", "content"}:
            return "message_body"
        if key == "amount":
            return "amount"
        if "date" in key or "day" in key or "time" in key:
            return "date"
        if key in {"path", "file", "file_id"}:
            return "file_path"
        return "unknown"

    def _first_match(self, matches: list[ValueMatchEvidence], role: str) -> dict[str, Any] | None:
        for match in matches:
            if match.field_role == role:
                return asdict(match)
        return None


