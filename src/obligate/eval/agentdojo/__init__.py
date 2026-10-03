"""Minimal, lazily loaded AgentDojo adapter for ObliGate."""

__all__ = [
    "AGENTDOJO_TOOL_TAXONOMY",
    "AgentDojoStateTracker",
    "AgentDojoToolFirewall",
    "AgentDojoToolTaxonomy",
    "ToolExecutionDecision",
    "classify_agentdojo_tool",
    "coverage_report",
    "load_agentdojo_taxonomy",
    "require_agentdojo",
    "summarize_agentdojo_audit",
    "summarize_obligate_audit",
]


def require_agentdojo() -> None:
    try:
        import agentdojo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("AgentDojo evaluation requires: pip install -e '.[agentdojo]'") from exc


def __getattr__(name: str):
    if name in {"AgentDojoStateTracker", "AgentDojoToolTaxonomy"}:
        if name == "AgentDojoStateTracker":
            from .evidence.state import AgentDojoStateTracker

            return AgentDojoStateTracker
        from .evidence.taxonomy import AgentDojoToolTaxonomy

        return AgentDojoToolTaxonomy
    if name in {"AgentDojoToolFirewall", "ToolExecutionDecision", "summarize_obligate_audit"}:
        from .gate.tool_firewall import AgentDojoToolFirewall, ToolExecutionDecision, summarize_obligate_audit

        return {
            "AgentDojoToolFirewall": AgentDojoToolFirewall,
            "ToolExecutionDecision": ToolExecutionDecision,
            "summarize_obligate_audit": summarize_obligate_audit,
        }[name]
    if name in {"AGENTDOJO_TOOL_TAXONOMY", "classify_agentdojo_tool", "coverage_report", "load_agentdojo_taxonomy"}:
        from .tool_taxonomy import AGENTDOJO_TOOL_TAXONOMY, classify_agentdojo_tool, coverage_report, load_agentdojo_taxonomy

        return {
            "AGENTDOJO_TOOL_TAXONOMY": AGENTDOJO_TOOL_TAXONOMY,
            "classify_agentdojo_tool": classify_agentdojo_tool,
            "coverage_report": coverage_report,
            "load_agentdojo_taxonomy": load_agentdojo_taxonomy,
        }[name]
    if name == "summarize_agentdojo_audit":
        from .runner.result_exporter import summarize_agentdojo_audit

        return summarize_agentdojo_audit
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"summarize_agentdojo_audit"})

