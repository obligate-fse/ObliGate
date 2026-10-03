from __future__ import annotations


def require_agentdojo() -> None:
    try:
        import agentdojo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("AgentDojo evaluation requires: pip install -e '.[agentdojo]'") from exc
