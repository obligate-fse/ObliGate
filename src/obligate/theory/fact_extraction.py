"""Oracle-free extraction at the canonical decision boundary.

Only fields in ``POLICY_FACT_KEYS`` may affect the theory policy.  Benchmark
ground truth and scorer records remain on the evaluation side of the boundary.
The excluded input is neither copied into the fact set nor represented by a
value-dependent digest, so changing hidden labels cannot change a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from obligate.eval.agentdojo.compat.types import ToolCallContext

from .model import canonical_json, sha256_json

FACT_EXTRACTION_SCHEMA = "obligate.observable-facts/v1"

# This is an exact allowlist, not a namespace allowlist.  Adding a policy input
# therefore requires a code review at this boundary and a label-invariance test.
POLICY_FACT_KEYS = frozenset(
    {
        "agentdojo.authorization_reasons",
        "agentdojo.delegated_selection_to_calendar",
        "agentdojo.delegated_target_selection",
        "agentdojo.delegated_todo_execution",
        "agentdojo.delegated_transform_to_named_destination",
        "agentdojo.explicit_credential_update",
        "agentdojo.explicit_profile_update",
        "agentdojo.named_transform_payload_safe",
        "agentdojo.target_entity_source",
        "graph.attack_goal_evidence_source",
        "graph.has_injection_to_side_effect_edge",
        "graph.has_private_to_external_edge",
    }
)

_AUTHORIZATION_REASONS = frozenset(
    {
        "arguments_match_user_task_entity",
        "private_read_object_matches_trusted_task",
        "tool_group_authorized_by_task",
        "tool_target_matches_user_task_entity",
    }
)

_ASB_VISIBLE_KEYS = frozenset(
    {
        "low_trust_instruction_seen",
        "prompt_injection_marker_seen",
        "tool_description",
        "tool_named_in_low_trust_text",
        "tool_named_in_trusted_task",
        "visible_task_tool_grounded",
    }
)
_AGENT_SAFETYBENCH_VISIBLE_KEYS = frozenset({"env_name", "instruction", "tool_description"})


@dataclass(frozen=True, slots=True)
class ExtractedFacts:
    facts: Mapping[str, Any]
    content_hash: str

    def to_trace(self) -> dict[str, object]:
        return {
            "schema": FACT_EXTRACTION_SCHEMA,
            "content_hash": self.content_hash,
            "fact_count": len(self.facts),
            "oracle_inputs": "excluded_by_construction",
        }


class ObservableFactExtractor:
    """Project untyped adapter facts into the reviewed policy fact schema."""

    def extract(self, raw_facts: Mapping[str, Any] | None) -> ExtractedFacts:
        source = raw_facts or {}
        selected: dict[str, Any] = {}
        for key in sorted(POLICY_FACT_KEYS):
            if key not in source:
                continue
            value = source[key]
            if key == "agentdojo.authorization_reasons":
                if not isinstance(value, (list, tuple)):
                    continue
                value = sorted({str(item) for item in value if str(item) in _AUTHORIZATION_REASONS})
            elif key == "graph.attack_goal_evidence_source":
                # ``oracle_upper_bound`` and arbitrary source tags are not facts
                # in the deployable policy domain.
                if value != "observed_tool_output":
                    continue
            else:
                # Canonicalization rejects unserializable or non-finite values.
                canonical_json(value)
            selected[key] = value
        frozen = MappingProxyType(selected)
        return ExtractedFacts(facts=frozen, content_hash=sha256_json(selected))


def oracle_free_tool_context(context: ToolCallContext) -> ToolCallContext:
    """Return the only ``ToolCallContext`` allowed to reach fact extraction.

    IDs, benchmark-authorized tool lists, attack signatures, and oracle modes
    are intentionally erased.  The visible user task, candidate call, public
    tool metadata, and observable low-trust markers remain.
    """

    return replace(
        context,
        user_task_id=None,
        injection_task_id=None,
        allowed_tools=set(),
        allowed_groups=set(),
        attack_goal_signatures=[],
        run_id=f"observable:{sha256_json({'suite': context.suite, 'task': context.user_task})[:20]}",
        sample_id=None,
        raw_tool_call=_visible_raw_tool_call(context.raw_tool_call),
        defense_mode="fair",
    )


def _visible_raw_tool_call(raw_tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(raw_tool_call, Mapping):
        return None
    visible: dict[str, Any] = {}
    asb = raw_tool_call.get("asb")
    if isinstance(asb, Mapping):
        visible["asb"] = {key: asb[key] for key in sorted(_ASB_VISIBLE_KEYS) if key in asb}
    agent_safetybench = raw_tool_call.get("agent_safetybench")
    if isinstance(agent_safetybench, Mapping):
        visible["agent_safetybench"] = {
            key: agent_safetybench[key]
            for key in sorted(_AGENT_SAFETYBENCH_VISIBLE_KEYS)
            if key in agent_safetybench
        }
    return visible or None


__all__ = [
    "ExtractedFacts",
    "FACT_EXTRACTION_SCHEMA",
    "ObservableFactExtractor",
    "POLICY_FACT_KEYS",
    "oracle_free_tool_context",
]
