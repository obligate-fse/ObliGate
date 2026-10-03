"""Compatibility layer for ActionGraph objects.

When installed inside ObliGate, this module imports the real ObliGate model
classes.  The fallback dataclasses make the adapter testable as a standalone
upgrade bundle before Codex applies it to the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any
from uuid import uuid4

try:  # pragma: no cover
    from obligate.models import ActionEdge, ActionGraph, ActionNode, new_id, sha256_text  # type: ignore
except Exception:  # pragma: no cover

    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:12]}"

    def sha256_text(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    @dataclass(slots=True)
    class ActionNode:
        node_id: str
        action_id: str
        semantic_action: str
        tool: str | None = None
        target: str | None = None
        affected_assets: list[str] = field(default_factory=list)
        source_ids: list[str] = field(default_factory=list)
        side_effect: bool = False
        confidence: float = 1.0
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass(slots=True)
    class ActionEdge:
        edge_id: str
        src_node_id: str
        dst_node_id: str
        relation: str
        evidence_refs: list[str] = field(default_factory=list)
        confidence: float = 1.0
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass(slots=True)
    class ActionGraph:
        graph_id: str
        run_id: str | None = None
        root_action_id: str | None = None
        raw_action_hash: str | None = None
        nodes: list[ActionNode] = field(default_factory=list)
        edges: list[ActionEdge] = field(default_factory=list)
        parser_version: str = "agentdojo-tool-relation-graph-v3"
        complete: bool = True
        metadata: dict[str, Any] = field(default_factory=dict)

