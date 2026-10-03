"""AgentDojo-adapted ActionGraph.

This is not a shell or program DFG.  It is a tool relation graph that exposes
AgentDojo-specific security relations:

- untrusted output -> later side-effecting tool call
- injection-like output -> later high-risk tool call
- private data read -> later external send/share
- private financial data -> later financial commit
- attack-goal signature -> current tool arguments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..compat.models_compat import ActionEdge, ActionGraph, ActionNode, new_id, sha256_text
from ..compat.types import EvidenceBundle, ToolCallContext, ToolSpec
from .state import AgentDojoStateTracker, ToolEvent


@dataclass(slots=True)
class GraphBuildResult:
    graph: ActionGraph
    facts: dict[str, Any]


class AgentDojoActionGraphBuilder:
    parser_version = "agentdojo-tool-relation-graph-v3"

    def build(
        self, *, context: ToolCallContext, spec: ToolSpec, state: AgentDojoStateTracker, evidence: EvidenceBundle | None = None
    ) -> GraphBuildResult:
        ablation = dict(context.ablation_config or {})
        structure_edges = bool(ablation.get("enable_actiongraph_structure_edges", True))
        provenance_edges = bool(ablation.get("enable_actiongraph_provenance_edges", True)) and structure_edges
        dataflow_edges = bool(ablation.get("enable_actiongraph_dataflow_edges", True)) and structure_edges
        history_edges = bool(ablation.get("enable_actiongraph_history_edges", True)) and structure_edges
        graph_id = new_id("adj_graph")
        root_action_id = new_id("adj_action")
        current = self._tool_call_node(root_action_id, context, spec)
        nodes: list[ActionNode] = [current]
        edges: list[ActionEdge] = []
        facts: dict[str, Any] = {
            "graph.kind": "agentdojo_tool_relation_graph",
            "graph.version": self.parser_version,
            "graph.current_tool_node": current.node_id,
            "graph.sequence_depth": len(state.events),
            "graph.has_untrusted_to_side_effect_edge": False,
            "graph.has_untrusted_to_proposed_side_effect_edge": False,
            "graph.has_untrusted_to_executed_side_effect_edge": False,
            "graph.has_injection_to_side_effect_edge": False,
            "graph.has_private_to_external_edge": False,
            "graph.has_private_to_proposed_external_edge": False,
            "graph.has_private_to_executed_external_edge": False,
            "graph.has_private_to_financial_edge": False,
            "graph.has_attack_goal_to_action_edge": False,
            "graph.has_blocked_attempt_edge": bool(history_edges and any(event.event_status == "blocked" for event in state.events)),
            "graph.has_unknown_side_effect_node": spec.group == "unknown" and spec.side_effect,
            "graph.attack_goal_evidence_source": "none",
            "graph.ablation.structure_edges_enabled": structure_edges,
            "graph.ablation.provenance_edges_enabled": provenance_edges,
            "graph.ablation.dataflow_edges_enabled": dataflow_edges,
            "graph.ablation.history_edges_enabled": history_edges,
        }

        previous = self._last_event(state, kind="tool_call", statuses={"executed", "blocked"}) if history_edges else None
        if previous is not None:
            prev_node = self._event_node(previous, context, risk="low")
            nodes.append(prev_node)
            edges.append(
                self._edge(
                    prev_node.node_id, current.node_id, "sequence", 0.6, "previous AgentDojo tool call occurred before current candidate"
                )
            )

        for event in self._recent_events(state, predicate=lambda ev: ev.untrusted or ev.injection_like, limit=3) if provenance_edges else []:
            ev_node = self._event_node(event, context, risk="high" if event.injection_like else "medium")
            nodes.append(ev_node)
            if spec.side_effect:
                edges.append(
                    self._edge(
                        ev_node.node_id,
                        current.node_id,
                        "controlflow",
                        0.82 if event.injection_like else 0.72,
                        "prior untrusted or injection-like tool output may influence current side-effecting tool",
                    )
                )
                facts["graph.has_untrusted_to_side_effect_edge"] = True
                facts["graph.has_untrusted_to_proposed_side_effect_edge"] = True
                if event.injection_like:
                    facts["graph.has_injection_to_side_effect_edge"] = True

        for event in self._recent_events(state, predicate=lambda ev: ev.private_data, limit=3) if dataflow_edges else []:
            ev_node = self._event_node(event, context, risk="high")
            nodes.append(ev_node)
            if spec.external_sink:
                edges.append(self._edge(ev_node.node_id, current.node_id, "dataflow", 0.78, "possible private data to external sink"))
                facts["graph.has_private_to_external_edge"] = True
                facts["graph.has_private_to_proposed_external_edge"] = True
            if spec.group == "financial_commit":
                edges.append(
                    self._edge(ev_node.node_id, current.node_id, "dataflow", 0.7, "private financial data may influence financial commit")
                )
                facts["graph.has_private_to_financial_edge"] = True

        if provenance_edges and state.attack_goal_overlap(context.tool_args):
            attack_node = self._state_node("attack_goal_overlap", context, risk="critical")
            nodes.append(attack_node)
            edges.append(
                self._edge(
                    attack_node.node_id,
                    current.node_id,
                    "controlflow",
                    0.92,
                    "current tool arguments overlap suspicious target extracted from prior output",
                )
            )
            facts["graph.has_attack_goal_to_action_edge"] = True
            facts["graph.attack_goal_evidence_source"] = (
                "oracle_upper_bound" if context.defense_mode == "oracle_full" else "observed_tool_output"
            )

        graph = ActionGraph(
            graph_id=graph_id,
            run_id=context.run_id,
            root_action_id=root_action_id,
            raw_action_hash=sha256_text(f"{context.suite}:{context.tool_name}:{repr(context.tool_args)}"),
            nodes=nodes,
            edges=edges,
            parser_version=self.parser_version,
            complete=True,
            metadata={
                "adapter": "obligate",
                "suite": context.suite,
                "tool_name": context.tool_name,
                "sample_id": context.sample_id,
                "state_digest": state.digest(),
                "edge_count": len(edges),
                "node_count": len(nodes),
                "relation_semantics": "conservative_tool_relation_graph_not_exact_program_dfg",
            },
        )
        facts.update(
            {
                "graph.edge_count": len(edges),
                "graph.node_count": len(nodes),
                "graph.confidence_min": min((edge.confidence for edge in edges), default=1.0),
            }
        )
        return GraphBuildResult(graph=graph, facts=facts)

    def _tool_call_node(self, action_id: str, context: ToolCallContext, spec: ToolSpec) -> ActionNode:
        return ActionNode(
            node_id=new_id("adj_node"),
            action_id=action_id,
            semantic_action=spec.semantic_action,
            tool="AgentDojo",
            target=context.tool_name,
            affected_assets=[f"agentdojo:{context.suite}:{context.tool_name}"],
            source_ids=[f"agentdojo:{context.suite}:tool_call"],
            side_effect=spec.side_effect,
            confidence=0.96 if spec.group != "unknown" else 0.55,
            metadata={
                "suite": context.suite,
                "tool_group": spec.group,
                "risk": spec.risk,
                "tool_args_digest": sha256_text(repr(context.tool_args)),
            },
        )

    def _event_node(self, event: ToolEvent, context: ToolCallContext, *, risk: str) -> ActionNode:
        return ActionNode(
            node_id=new_id("adj_node"),
            action_id=event.event_id,
            semantic_action=f"prior_{event.kind}",
            tool="AgentDojo",
            target=event.tool_name,
            affected_assets=[f"agentdojo:{context.suite}:history"],
            source_ids=[f"agentdojo:{context.suite}:state"],
            side_effect=False,
            confidence=0.85,
            metadata={
                "event_kind": event.kind,
                "event_status": event.event_status,
                "tool_group": event.tool_group,
                "output_hash": event.output_hash,
                "untrusted": event.untrusted,
                "injection_like": event.injection_like,
                "private_data": event.private_data,
                "risk": risk,
            },
        )

    def _state_node(self, semantic: str, context: ToolCallContext, *, risk: str) -> ActionNode:
        return ActionNode(
            node_id=new_id("adj_node"),
            action_id=new_id("adj_state"),
            semantic_action=semantic,
            tool="AgentDojoStateTracker",
            target="state",
            affected_assets=[f"agentdojo:{context.suite}:state"],
            source_ids=[f"agentdojo:{context.suite}:state"],
            side_effect=False,
            confidence=0.9,
            metadata={"risk": risk},
        )

    def _edge(self, src: str, dst: str, relation: str, confidence: float, reason: str) -> ActionEdge:
        return ActionEdge(
            edge_id=new_id("adj_edge"),
            src_node_id=src,
            dst_node_id=dst,
            relation=relation,
            evidence_refs=[reason],
            confidence=confidence,
            metadata={"reason": reason, "graph_kind": "agentdojo_tool_relation"},
        )

    def _recent_events(self, state: AgentDojoStateTracker, *, predicate: Callable[[ToolEvent], bool], limit: int) -> list[ToolEvent]:
        out: list[ToolEvent] = []
        for event in reversed(state.events):
            if predicate(event):
                out.append(event)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def _last_event(self, state: AgentDojoStateTracker, *, kind: str, statuses: set[str] | None = None) -> ToolEvent | None:
        for event in reversed(state.events):
            if event.kind == kind and (statuses is None or event.event_status in statuses):
                return event
        return None


