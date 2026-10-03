"""Frozen scalar-average scoring shared by calibration and online research runtime."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

EXECUTION_RANK = {"host": 0.0, "sandbox": 0.5, "no_execute": 1.0}
NETWORK_RANK = {"allow": 0.0, "allowlist": 0.5, "deny": 1.0}
DATA_RANK = {"raw": 0.0, "redact": 0.5, "no_sensitive": 1.0}
HUMAN_RANK = {"none": 0.0, "approval_required": 1.0}


def plan_score(plan: Mapping[str, Any]) -> float:
    """Mean normalized strength over env/net/data/human/audit coordinates."""

    audit = plan.get("audit") or {}
    audit_rank = 1.0 if (audit.get("must") or ()) else 0.0
    values = (
        EXECUTION_RANK[str(plan.get("execution_env") or "host")],
        NETWORK_RANK[str(plan.get("network_scope") or "allow")],
        DATA_RANK[str(plan.get("data_scope") or "raw")],
        HUMAN_RANK[str(plan.get("human_gate") or "none")],
        audit_rank,
    )
    return sum(values) / len(values)


def trigger_scores(eoc: Mapping[str, Any]) -> dict[str, float]:
    """Use the weakest (minimal) declared local remedy for every Trigger."""

    values: dict[str, float] = {}
    for trigger_id, remedies in (eoc.get("remedies") or {}).items():
        scores = [plan_score(item["plan"]) for item in remedies if isinstance(item, Mapping) and item.get("plan")]
        values[str(trigger_id)] = min(scores) if scores else 1.0
    for trigger in eoc.get("triggers") or ():
        values.setdefault(str(trigger["trigger_id"]), 1.0)
    return values


def aggregate_score(eoc: Mapping[str, Any]) -> float:
    values = trigger_scores(eoc)
    return 0.0 if not values else sum(values.values()) / len(values)


def has_hard_trigger(eoc: Mapping[str, Any]) -> bool:
    return any(bool(item.get("hard")) for item in eoc.get("triggers") or ())


def scalar_decision(
    score: float,
    thresholds: Sequence[float],
    *,
    hard: bool = False,
) -> str:
    if hard:
        return "block"
    t_allow, t_confirm, t_block = (float(item) for item in thresholds)
    if not 0.0 <= t_allow < t_confirm < t_block <= 1.0:
        raise ValueError("scalar thresholds must satisfy 0 <= allow < confirm < block <= 1")
    if score < t_allow:
        return "allow"
    if score < t_confirm:
        return "execute_with_constraints"
    if score < t_block:
        return "require_confirmation"
    return "block"


__all__ = [
    "DATA_RANK",
    "EXECUTION_RANK",
    "HUMAN_RANK",
    "NETWORK_RANK",
    "aggregate_score",
    "has_hard_trigger",
    "plan_score",
    "scalar_decision",
    "trigger_scores",
]
