"""Summarize ObliGate audit events for AgentDojo experiment reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..tool_taxonomy import coverage_report


def summarize_agentdojo_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [event.get("payload", {}) for event in events if event.get("event_type") == "policy_decision"]
    gate_events = [event.get("payload", {}) for event in events if event.get("event_type") == "agentdojo_tool_gate_decision"]
    fact_events = [event.get("payload", {}) for event in events if event.get("event_type") == "policy_fact_set"]
    policy_latencies = _extract_latencies(events, "policy.total_ms")
    audit_latencies = _extract_latencies(events, "audit.write_ms")
    reason_counts = Counter(reason for decision in decisions for reason in decision.get("reason_codes", []) or [])
    unknown_gate_events = [payload for payload in gate_events if payload.get("unknown_tool") or payload.get("registered") is False]
    registered_gate_events = [payload for payload in gate_events if payload.get("registered") is True]
    decision_counts = Counter(str(item.get("decision")) for item in gate_events)
    fact_count = sum(int((payload.get("namespace_counts") or {}).get("agentdojo", 0)) for payload in fact_events)
    tool_names = [str(item.get("tool_name")) for item in gate_events if item.get("tool_name")]
    coverage = coverage_report(tool_names)
    return {
        "obligate_checked_calls": len(gate_events),
        "obligate_policy_decisions": dict(decision_counts),
        "obligate_blocks": sum(1 for item in gate_events if item.get("decision") in {"block", "quarantine"}),
        "obligate_unknown_tools": len(unknown_gate_events),
        "obligate_registered_tools": len(registered_gate_events),
        "obligate_unknown_tool_rate": (len(unknown_gate_events) / max(len(gate_events), 1)) if gate_events else 0.0,
        "obligate_registered_tool_rate": (len(registered_gate_events) / max(len(gate_events), 1)) if gate_events else 0.0,
        "obligate_top_reason_codes": reason_counts.most_common(10),
        "obligate_avg_policy_latency_ms": _avg(policy_latencies),
        "obligate_p50_policy_latency_ms": _p50(policy_latencies),
        "obligate_p95_policy_latency_ms": _p95(policy_latencies),
        "obligate_p95_audit_latency_ms": _p95(audit_latencies),
        "obligate_fact_count": fact_count,
        "taxonomy_coverage": coverage,
    }


def _extract_latencies(events: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for event in events:
        payload = event.get("payload") or {}
        timings = payload.get("timings_ms") or {}
        value = timings.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
        if key == "audit.write_ms":
            extra = payload.get("audit_write_ms")
            if isinstance(extra, (int, float)):
                values.append(float(extra))
    return values


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _p50(values: list[float]) -> float:
    return _percentile(values, 0.5)


def _p95(values: list[float]) -> float:
    return _percentile(values, 0.95)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return float(ordered[idx])



