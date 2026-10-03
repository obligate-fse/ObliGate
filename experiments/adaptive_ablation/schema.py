"""Typed schemas and information-boundary constants for adaptive attacks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = "obligate-adaptive-attack-v2"
ATTACKER_VISIBLE_FIELDS_VERSION = "attacker-visible-fields-v2"
EXPERIMENT_MODE = "one-step-feedback-conditioned"
FORMAL_EXPERIMENT_NAME = (
    "post-freeze, public-full-set, one-step feedback-conditioned "
    "mechanism-aware stress test"
)


class Benchmark(str, Enum):
    AGENTDOJO = "agentdojo"
    AGENT_SAFETYBENCH = "agent_safetybench"
    ASB = "agent_security_bench"


class AttackerChannel(str, Enum):
    DIRECT_REQUEST = "direct_request"
    TOOL_OUTPUT = "tool_output"
    DOCUMENT = "document"
    MESSAGE = "message"
    MEMORY = "memory"
    DYNAMIC_TOOL_DESCRIPTION = "dynamic_tool_description"
    UNKNOWN = "unknown"


class PublicFeedbackClass(str, Enum):
    """Coarse feedback classes allowed by the experiment protocol.

    Only these deployment-visible coarse classes may condition Round 2.  In
    particular, ActionGraph, Gap/Trigger, certificate state, evaluator labels,
    and fine-grained rule identifiers never enter the compiler.
    """

    NO_FEEDBACK = "no_feedback"
    ALLOW = "allow"
    EXECUTE_WITH_CONSTRAINTS = "execute_with_constraints"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"
    COMPLIANCE_ERROR = "compliance_error"
    PROVIDER_OR_WORKFLOW_ERROR = "provider_or_workflow_error"


class ToolSchemaClass(str, Enum):
    NO_TOOL = "no_tool"
    READ_ONLY = "read_only"
    STATE_MUTATION = "state_mutation"
    EXTERNAL_SINK = "external_sink"
    FINANCIAL = "financial"
    CREDENTIAL = "credential"
    MIXED = "mixed"
    GENERIC = "generic"


class MechanismOperator(str, Enum):
    """The five protocol-approved abstract perturbation families.

    The historical class name is retained to avoid breaking frozen-analysis
    imports, but its value set is intentionally identical to the experiment
    protocol.  These values describe benchmark-text transformations; they are
    not executable attack primitives.
    """

    SOURCE_SEMANTIC_AMBIGUITY = "source_semantic_ambiguity"
    CONTRADICTORY_FACT_DESCRIPTIONS = "contradictory_fact_descriptions"
    MISSING_OR_INSUFFICIENT_AUTHORIZATION_EVIDENCE = (
        "missing_or_insufficient_authorization_evidence"
    )
    MULTI_POLICY_OBLIGATION_COUPLING = "multi_policy_obligation_coupling"
    SEMANTICALLY_EQUIVALENT_REWRITE = "semantically_equivalent_rewrite"


DEFAULT_MECHANISM_OPERATORS: tuple[MechanismOperator, ...] = tuple(MechanismOperator)


class CompileStatus(str, Enum):
    COMPILED = "compiled"
    FALLBACK_ORIGINAL = "fallback_original"
    ORIGINAL_NOOP = "original_noop"


# These are source-field allowlists.  Adapters project the raw benchmark row
# before reading values, so extra keys can be present without reaching the
# compiler.  In particular, no case identifier or evaluator field is listed.
PERTURBATION_VISIBLE_FIELDS: Mapping[Benchmark, tuple[str, ...]] = {
    Benchmark.AGENTDOJO: (
        "attack_channel",
        "injection_text",
        "tool_schemas",
    ),
    Benchmark.AGENT_SAFETYBENCH: (
        "attack_channel",
        "instruction",
        "environment_tool_schemas",
    ),
    Benchmark.ASB: (
        "attack_channel",
        "attack_payload",
        "attacker_tool_name",
        "attacker_tool_description",
        "attacker_tool_schema",
    ),
}

# Backwards-compatible import alias.  New manifests and reports use the exact
# protocol term ``perturbation_visible_fields``.
ATTACKER_VISIBLE_FIELDS = PERTURBATION_VISIBLE_FIELDS


FORBIDDEN_ATTACK_COMPILER_FIELDS = frozenset(
    {
        "case_id",
        "task_id",
        "user_task_id",
        "injection_task_id",
        "evaluator",
        "evaluator_label",
        "safety_label",
        "official_answer",
        "ground_truth",
        "hidden_attack_target",
        "attack_target",
        "attack_goal",
        "is_attacker",
        "normal_or_attacker_partition",
        "scorer",
        "scorer_state",
        "risks",
        "failure_modes",
        "fulfillable",
        "action_graph",
        "witnesses",
        "gap",
        "hazard",
        "overflow",
        "trigger",
        "certificate",
        "defense_decision",
        "reason_codes",
    }
)


CANONICAL_COMPILER_FIELDS = frozenset(
    {
        "schema_version",
        "attacker_channel",
        "original_payload",
        "public_goal",
        "tool_schemas",
        "editable",
    }
)


@dataclass(frozen=True)
class NormalizedToolSchema:
    """Only the deploy-time public portion of a tool schema."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _json_safe(self.parameters),
        }


@dataclass(frozen=True)
class CanonicalAttackInput:
    """The complete input accepted by :class:`AttackCompiler`.

    It deliberately has no benchmark name, case identifier, evaluator object,
    labels, hidden target, or ObliGate internal state.
    """

    attacker_channel: AttackerChannel
    original_payload: str
    tool_schemas: tuple[NormalizedToolSchema, ...]
    public_goal: str | None = None
    editable: bool = True
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "attacker_channel": self.attacker_channel.value,
            "original_payload": self.original_payload,
            "public_goal": self.public_goal,
            "tool_schemas": [tool.as_dict() for tool in self.tool_schemas],
            "editable": self.editable,
        }
        if set(value) != CANONICAL_COMPILER_FIELDS:
            raise AssertionError("canonical compiler schema drifted outside its frozen field set")
        return value

    def serialized(self) -> str:
        return canonical_json(self.as_dict())

    @property
    def input_sha256(self) -> str:
        return sha256_text(self.serialized())


@dataclass(frozen=True)
class PublicRound1Feedback:
    """The complete, coarse Round-1 feedback permitted at the attack boundary."""

    decision_class: PublicFeedbackClass
    source: str = "round1_public_trace"

    def __post_init__(self) -> None:
        if self.decision_class is PublicFeedbackClass.NO_FEEDBACK:
            raise ValueError("Round-1 feedback cannot be no_feedback")
        if self.source != "round1_public_trace":
            raise ValueError("feedback source is not protocol approved")

    def as_dict(self) -> dict[str, str]:
        return {
            "decision_class": self.decision_class.value,
            "source": self.source,
        }


@dataclass(frozen=True)
class AdaptedAttackInput:
    """Auditable adapter result; only ``canonical`` enters the compiler."""

    benchmark: Benchmark
    source_fields: tuple[str, ...]
    canonical: CanonicalAttackInput
    target_status: str

    def __post_init__(self) -> None:
        allowed = set(ATTACKER_VISIBLE_FIELDS[self.benchmark])
        if not set(self.source_fields).issubset(allowed):
            raise ValueError("adapter source fields exceed attacker_visible_fields")


@dataclass(frozen=True)
class CompiledPayload:
    schema_version: str
    experiment_mode: str
    status: CompileStatus
    payload: str
    payload_sha256: str
    original_payload_sha256: str
    input_sha256: str
    attacker_channel: AttackerChannel
    tool_schema_class: ToolSchemaClass
    public_feedback_class: PublicFeedbackClass
    template_key: str
    template_sha256: str
    requested_operators: tuple[MechanismOperator, ...]
    operators: tuple[MechanismOperator, ...]
    generation_invalid: bool
    fallback_reason: str | None
    target_status: str

    def as_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "experiment_mode": self.experiment_mode,
            "status": self.status.value,
            "payload_sha256": self.payload_sha256,
            "original_payload_sha256": self.original_payload_sha256,
            "input_sha256": self.input_sha256,
            "attacker_channel": self.attacker_channel.value,
            "tool_schema_class": self.tool_schema_class.value,
            "public_feedback_class": self.public_feedback_class.value,
            "template_key": self.template_key,
            "template_sha256": self.template_sha256,
            "requested_operators": [operator.value for operator in self.requested_operators],
            "operators": [operator.value for operator in self.operators],
            "generation_invalid": self.generation_invalid,
            "fallback_reason": self.fallback_reason,
            "target_status": self.target_status,
        }
        if include_payload:
            value["payload"] = self.payload
        return value


@dataclass(frozen=True)
class CasePayloadBinding:
    """Case association added only after compilation has completed."""

    case_id: str
    benchmark: Benchmark
    compiled: CompiledPayload


def canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Public JSON schemas should already be JSON-compatible.  Rejecting opaque
    # objects prevents evaluator handles or other authority-bearing objects
    # from being stringified into compiler input.
    raise TypeError(f"compiler input contains a non-JSON value: {type(value).__name__}")
