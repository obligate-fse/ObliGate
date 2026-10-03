"""Configuration for low-overhead evaluation runs."""

from __future__ import annotations

import os
from dataclasses import dataclass

TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EvalFastModeConfig:
    enabled: bool
    disable_studio_events: bool
    audit_buffered: bool
    evidence_graph_mode: str
    disable_preflight: bool
    policy_trace_mode: str
    session_cache: bool


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def load_eval_fast_mode_config() -> EvalFastModeConfig:
    enabled = env_flag("OBLIGATE_EVAL_FAST_MODE")
    return EvalFastModeConfig(
        enabled=enabled,
        disable_studio_events=env_flag("OBLIGATE_DISABLE_STUDIO_EVENTS", default=enabled),
        audit_buffered=env_flag("OBLIGATE_AUDIT_BUFFERED", default=enabled),
        evidence_graph_mode=os.getenv("OBLIGATE_EVIDENCE_GRAPH_MODE", "summary" if enabled else "full"),
        disable_preflight=env_flag("OBLIGATE_DISABLE_PREFLIGHT", default=enabled),
        policy_trace_mode=os.getenv("OBLIGATE_POLICY_TRACE_MODE", "summary" if enabled else "full"),
        session_cache=env_flag("OBLIGATE_SESSION_CACHE", default=enabled),
    )
