"""Label-blind benchmark adapters for the adaptive-attack compiler."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import (
    ATTACKER_VISIBLE_FIELDS,
    AdaptedAttackInput,
    AttackerChannel,
    Benchmark,
    CanonicalAttackInput,
    NormalizedToolSchema,
)

_GOAL_LINE = re.compile(r"(?im)^\s*(?:goal|objective|target)\s*[:\-]\s*(.+?)\s*$")


def adapt_agentdojo(raw: Mapping[str, Any]) -> AdaptedAttackInput:
    visible = _project(raw, Benchmark.AGENTDOJO)
    original = _clean_text(visible.get("injection_text"))
    schemas = _normalize_tool_schemas(visible.get("tool_schemas"))
    return _adapted(
        benchmark=Benchmark.AGENTDOJO,
        visible=visible,
        channel=_channel(visible.get("attack_channel")),
        original=original,
        schemas=schemas,
    )


def adapt_agent_safetybench(raw: Mapping[str, Any]) -> AdaptedAttackInput:
    visible = _project(raw, Benchmark.AGENT_SAFETYBENCH)
    original = _clean_text(visible.get("instruction"))
    schemas = _normalize_tool_schemas(visible.get("environment_tool_schemas"))
    return _adapted(
        benchmark=Benchmark.AGENT_SAFETYBENCH,
        visible=visible,
        channel=_channel(visible.get("attack_channel")),
        original=original,
        schemas=schemas,
    )


def adapt_asb(raw: Mapping[str, Any]) -> AdaptedAttackInput:
    visible = _project(raw, Benchmark.ASB)
    exposed_schema = _normalize_one_tool_schema(visible.get("attacker_tool_schema"))
    explicit_name = _clean_text(visible.get("attacker_tool_name"))
    explicit_description = _clean_text(visible.get("attacker_tool_description"))
    if exposed_schema is None and (explicit_name or explicit_description):
        exposed_schema = NormalizedToolSchema(name=explicit_name, description=explicit_description, parameters={})
    elif exposed_schema is not None and explicit_description:
        exposed_schema = NormalizedToolSchema(
            name=explicit_name or exposed_schema.name,
            description=explicit_description,
            parameters=exposed_schema.parameters,
        )
    schemas = (exposed_schema,) if exposed_schema is not None else ()
    # ASB's public injection is the rendered attack prompt, not the hidden
    # scorer goal.  It is frozen before the case identifier is associated.
    original = _clean_text(visible.get("attack_payload"))
    if not original:
        original = explicit_description or (exposed_schema.description if exposed_schema is not None else "")
    return _adapted(
        benchmark=Benchmark.ASB,
        visible=visible,
        channel=_channel(visible.get("attack_channel"), default=AttackerChannel.DYNAMIC_TOOL_DESCRIPTION),
        original=original,
        schemas=schemas,
    )


def adapt_visible_input(benchmark: Benchmark | str, raw: Mapping[str, Any]) -> AdaptedAttackInput:
    selected = Benchmark(benchmark)
    if selected is Benchmark.AGENTDOJO:
        return adapt_agentdojo(raw)
    if selected is Benchmark.AGENT_SAFETYBENCH:
        return adapt_agent_safetybench(raw)
    if selected is Benchmark.ASB:
        return adapt_asb(raw)
    raise AssertionError(f"unhandled benchmark: {selected}")


def _adapted(
    *,
    benchmark: Benchmark,
    visible: Mapping[str, Any],
    channel: AttackerChannel,
    original: str,
    schemas: tuple[NormalizedToolSchema, ...],
) -> AdaptedAttackInput:
    goal = _extract_public_goal(original)
    return AdaptedAttackInput(
        benchmark=benchmark,
        source_fields=tuple(name for name in ATTACKER_VISIBLE_FIELDS[benchmark] if name in visible),
        canonical=CanonicalAttackInput(
            attacker_channel=channel,
            original_payload=original,
            public_goal=goal,
            tool_schemas=schemas,
            editable=bool(original.strip()),
        ),
        target_status="parsed_from_visible_text" if goal else "target-unavailable",
    )


def _project(raw: Mapping[str, Any], benchmark: Benchmark) -> dict[str, Any]:
    """Read exactly the explicit attacker-visible source-field allowlist."""

    if not isinstance(raw, Mapping):
        raise TypeError("benchmark adapter input must be a mapping")
    return {name: raw[name] for name in ATTACKER_VISIBLE_FIELDS[benchmark] if name in raw}


def _channel(value: Any, *, default: AttackerChannel = AttackerChannel.UNKNOWN) -> AttackerChannel:
    text = _clean_text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "direct": AttackerChannel.DIRECT_REQUEST,
        "direct_request": AttackerChannel.DIRECT_REQUEST,
        "tool": AttackerChannel.TOOL_OUTPUT,
        "tool_output": AttackerChannel.TOOL_OUTPUT,
        "indirect_tool_output": AttackerChannel.TOOL_OUTPUT,
        "document": AttackerChannel.DOCUMENT,
        "file": AttackerChannel.DOCUMENT,
        "message": AttackerChannel.MESSAGE,
        "email": AttackerChannel.MESSAGE,
        "memory": AttackerChannel.MEMORY,
        "dynamic_tool": AttackerChannel.DYNAMIC_TOOL_DESCRIPTION,
        "dynamic_tool_description": AttackerChannel.DYNAMIC_TOOL_DESCRIPTION,
    }
    return aliases.get(text, default)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_public_goal(text: str) -> str | None:
    match = _GOAL_LINE.search(text)
    if not match:
        return None
    goal = match.group(1).strip()
    return goal[:1000] if goal else None


def _normalize_tool_schemas(value: Any) -> tuple[NormalizedToolSchema, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        return ()
    schemas = [schema for item in values if (schema := _normalize_one_tool_schema(item)) is not None]
    schemas.sort(key=lambda item: (item.name.casefold(), item.description.casefold()))
    return tuple(schemas)


def _normalize_one_tool_schema(value: Any) -> NormalizedToolSchema | None:
    if isinstance(value, str):
        name = _clean_text(value)
        return NormalizedToolSchema(name=name, description="", parameters={}) if name else None
    if not isinstance(value, Mapping):
        return None
    public = value.get("function") if isinstance(value.get("function"), Mapping) else value
    name = _clean_text(public.get("name"))
    description = _clean_text(public.get("description"))
    parameters = public.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    # Re-materialize through public JSON fields only.  Top-level fields such as
    # is_attacker or attack_goal are deliberately discarded.
    safe_parameters = _public_json(parameters)
    if not name and not description:
        return None
    return NormalizedToolSchema(name=name, description=description, parameters=safe_parameters)


def _public_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Opaque objects never enter serialized compiler input.
    return None
