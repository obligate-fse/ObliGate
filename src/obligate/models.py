"""Core data models used by obligate.

The models are intentionally small and serialisable so they can be logged,
replayed and consumed by other agent adapters.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Risk = Literal["low", "medium", "high", "critical"]
TrustLevel = Literal["admin", "trusted", "semi_trusted", "untrusted", "tool_untrusted", "tainted", "unknown"]
Decision = Literal["allow", "allow_in_sandbox", "require_confirmation", "sandbox_then_approval", "block", "quarantine"]
ContractMatch = Literal["match", "partial_match", "violation", "unknown"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_text(stable_json(data))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def to_dict(obj: Any) -> dict[str, Any]:
    """Dataclass-aware serialiser."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported serialisation type: {type(obj)!r}")


@dataclass(slots=True)
class AssetRecord:
    asset_id: str
    path: str
    canonical_path: str
    asset_type: str
    risk: Risk
    owner: str = "repo"
    discovered_by: str = "asset_scanner"
    confidence: float = 1.0
    last_seen: str = field(default_factory=utc_now)
    protection_policy: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepoAssetGraph:
    repo_root: str
    graph_id: str = field(default_factory=lambda: new_id("asset_graph"))
    generated_at: str = field(default_factory=utc_now)
    assets: list[AssetRecord] = field(default_factory=list)
    sensitive_assets: list[AssetRecord] = field(default_factory=list)
    critical_files: list[AssetRecord] = field(default_factory=list)
    external_sinks: list[dict[str, Any]] = field(default_factory=list)
    agent_capabilities: list[dict[str, Any]] = field(default_factory=list)
    visibility_gaps: list[dict[str, Any]] = field(default_factory=list)
    version_hash: str = ""

    def finalise(self) -> "RepoAssetGraph":
        payload = {
            "repo_root": self.repo_root,
            "assets": [to_dict(a) for a in self.assets],
            "sensitive_assets": [a.asset_id for a in self.sensitive_assets],
            "critical_files": [a.asset_id for a in self.critical_files],
            "external_sinks": self.external_sinks,
            "agent_capabilities": self.agent_capabilities,
            "visibility_gaps": self.visibility_gaps,
        }
        self.version_hash = sha256_json(payload)
        return self

    def asset_for_path(self, path: str | Path) -> AssetRecord | None:
        p = str(path)
        p_norm = p.replace("\\", "/")
        for asset in self.assets:
            ap = asset.path.replace("\\", "/")
            if p_norm == ap or p_norm.endswith("/" + ap) or ap.endswith(p_norm):
                return asset
        return None


@dataclass(slots=True)
class RiskSurfaceReport:
    critical_count: int
    high_count: int
    publish_assets: int
    package_lifecycle_risks: int
    visibility_gap_count: int
    summary: str


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    source_type: str
    trust_level: TrustLevel
    content_hash: str
    retrieved_at: str
    retrieval_path: str
    taint: list[str] = field(default_factory=list)
    allowed_use: list[str] = field(default_factory=list)
    forbidden_use: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    content_preview: str = ""


@dataclass(slots=True)
class ContextGraph:
    graph_id: str = field(default_factory=lambda: new_id("ctx_graph"))
    nodes: list[SourceRecord] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)

    def get(self, source_id: str) -> SourceRecord | None:
        for node in self.nodes:
            if node.source_id == source_id:
                return node
        return None

    def trust_for(self, source_ids: list[str]) -> list[TrustLevel]:
        levels: list[TrustLevel] = []
        for source_id in source_ids:
            node = self.get(source_id)
            levels.append(node.trust_level if node else "unknown")
        return levels

    def has_untrusted(self, source_ids: list[str]) -> bool:
        return any(t in {"untrusted", "tool_untrusted", "tainted", "unknown"} for t in self.trust_for(source_ids))


@dataclass(slots=True)
class TaskContract:
    task_id: str
    goal: str
    user_prompt: str
    allowed_files: list[str]
    forbidden_files: list[str]
    allowed_actions: list[str]
    conditionally_allowed_actions: list[dict[str, str]]
    forbidden_actions: list[str]
    allowed_network: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    allowed_recipients: list[str] = field(default_factory=list)
    allowed_package_sources: list[str] = field(default_factory=list)
    confirmation_required: bool = False
    confirmation_summary: str = ""
    max_risk_without_approval: Risk = "medium"
    confidence: float = 0.75
    revision: int = 1
    expires_at: str | None = None


@dataclass(slots=True)
class IntentDiff:
    action_id: str
    semantic_action: str
    contract_match: ContractMatch
    violation_reason: list[str] = field(default_factory=list)
    decision_hint: str = "allow"


@dataclass(slots=True)
class ActionIR:
    action_id: str
    raw_action: str
    tool: str
    cwd: str
    semantic_action: str
    risk: Risk
    risk_tags: list[str]
    affected_assets: list[str]
    requires: list[str]
    source_ids: list[str] = field(default_factory=list)
    parser_confidence: float = 1.0
    side_effect: bool = False
    command_parts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    graph_id: str | None = None
    sequence_no: int = 0
    parent_action_id: str | None = None


@dataclass(slots=True)
class ActionNode:
    node_id: str
    action_id: str
    semantic_action: str
    tool: str
    target: str | None
    affected_assets: list[str]
    source_ids: list[str]
    side_effect: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionEdge:
    edge_id: str
    src_node_id: str
    dst_node_id: str
    relation: Literal["pipe", "sequence", "redirect", "dataflow", "controlflow", "memoryflow"]
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionGraph:
    graph_id: str
    run_id: str
    root_action_id: str
    raw_action_hash: str
    nodes: list[ActionNode]
    edges: list[ActionEdge]
    parser_version: str
    complete: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionState:
    session_state_id: str
    run_id: str
    task_id: str | None
    secret_taint: bool = False
    touched_secret_assets: list[str] = field(default_factory=list)
    untrusted_source_seen: bool = False
    package_taint: bool = False
    ci_taint: bool = False
    prior_external_sinks: list[str] = field(default_factory=list)
    approval_scope: dict[str, Any] = field(default_factory=dict)
    last_decisions: list[str] = field(default_factory=list)
    state_hash: str = ""
    attempted_secret_taint: bool = False
    confirmed_secret_taint: bool = False
    attempted_secret_assets: list[str] = field(default_factory=list)
    confirmed_secret_assets: list[str] = field(default_factory=list)
    taint_confidence: str = "none"


@dataclass(slots=True)
class PackageEvent:
    package_event_id: str
    action_id: str
    event_type: str
    package: str | None
    source: str
    lifecycle_scripts: list[str]
    registry: str
    risk: Risk
    decision: str
    reason_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecTrace:
    exec_trace_id: str
    action_id: str
    command: str
    sandbox_profile: str
    process_tree: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    network_attempts: list[dict[str, Any]] = field(default_factory=list)
    env_access: list[str] = field(default_factory=list)
    package_scripts: list[str] = field(default_factory=list)
    diff_summary: list[str] = field(default_factory=list)
    exit_code: int | None = None
    risk_observed: list[str] = field(default_factory=list)
    recommended_decision: Decision = "allow"
    trace_complete: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SecretTaintEvent:
    secret_event_id: str
    event: str
    asset: str
    actor: str
    decision: str
    followed_by: str | None = None
    egress_target: str | None = None
    explanation: str = ""
    secret_hash: str | None = None


@dataclass(slots=True)
class PolicyDecision:
    decision_id: str
    action_id: str
    decision: Decision
    risk_score: int
    reason_codes: list[str]
    required_controls: list[str]
    explanation: str
    intent_diff: IntentDiff | None = None
    package_event_id: str | None = None
    exec_trace_id: str | None = None
    matched_rules: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    policy_version: str = ""
    rule_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApprovalRequest:
    approval_request_id: str
    task_id: str
    action_id: str
    plan_hash: str
    action_hash: str
    human_readable_summary: str
    source_influence: list[dict[str, Any]]
    affected_assets: list[str]
    observed_sandbox_risks: list[str]
    recommended_decision: Decision
    available_grants: list[str]
    action_hash_v1: str = ""
    action_hash_v2: str = ""


@dataclass(slots=True)
class ApprovalGrant:
    approval_id: str
    task_id: str
    action_id: str
    approved_plan_hash: str
    approved_action_hash: str
    constraints: list[str]
    expires_at: str
    granted_by: str = "local_user"
    approved_action_hash_v1: str = ""
    approved_action_hash_v2: str = ""


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    content_hash: str
    summary: str
    source_ids: list[str]
    source_trust_floor: TrustLevel
    memory_trust: TrustLevel
    allowed_use: list[str]
    forbidden_use: list[str]
    ttl_seconds: int
    created_at: str
    created_by: str
    write_decision_id: str | None = None


@dataclass(slots=True)
class MCPServerManifest:
    mcp_server_id: str
    launch_command: str
    config_source: str
    declared_tools: list[str]
    declared_capabilities: list[str]
    auth_required: bool
    token_policy: str
    trust_level: TrustLevel = "tool_untrusted"


@dataclass(slots=True)
class MCPInvocation:
    invocation_id: str
    server_id: str
    tool_name: str
    args_hash: str
    declared_capability: str
    observed_capability: str
    decision: str
    reason_codes: list[str]
    output_source_id: str | None = None


@dataclass(slots=True)
class AuditEvent:
    event_id: str
    schema_version: str
    prev_hash: str
    event_hash: str
    timestamp: str
    session_id: str
    task_id: str | None
    actor: str
    event_type: str
    payload: dict[str, Any]
    source_ids: list[str] = field(default_factory=list)
    action_id: str | None = None
    decision_id: str | None = None
    redaction: dict[str, Any] = field(default_factory=lambda: {"secret_values": "redacted", "stored_secret_hashes": True})
