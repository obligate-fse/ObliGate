"""Isolated research harness for adaptive-attack and ablation experiments.

This package is intentionally outside the production ObliGate loader.  Its
attack compiler accepts only the canonical, attacker-visible projection
created by :mod:`experiments.adaptive_ablation.adapters`.
"""

from .adapters import (
    adapt_agent_safetybench,
    adapt_agentdojo,
    adapt_asb,
    adapt_visible_input,
)
from .compiler import AttackCompiler, template_registry_sha256
from .schema import (
    ATTACKER_VISIBLE_FIELDS,
    ATTACKER_VISIBLE_FIELDS_VERSION,
    DEFAULT_MECHANISM_OPERATORS,
    AdaptedAttackInput,
    AttackerChannel,
    Benchmark,
    CanonicalAttackInput,
    CompiledPayload,
    CompileStatus,
    MechanismOperator,
    PublicFeedbackClass,
    ToolSchemaClass,
)

__all__ = [
    "ATTACKER_VISIBLE_FIELDS",
    "ATTACKER_VISIBLE_FIELDS_VERSION",
    "DEFAULT_MECHANISM_OPERATORS",
    "AdaptedAttackInput",
    "AttackCompiler",
    "AttackerChannel",
    "Benchmark",
    "CanonicalAttackInput",
    "CompileStatus",
    "CompiledPayload",
    "MechanismOperator",
    "PublicFeedbackClass",
    "ToolSchemaClass",
    "adapt_agent_safetybench",
    "adapt_agentdojo",
    "adapt_asb",
    "adapt_visible_input",
    "template_registry_sha256",
]
