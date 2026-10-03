"""Typed objects for the theory-aligned ObliGate execution path.

The objects in this module deliberately mirror the notation used by the
theory specification.  They are independent from benchmark-specific models
so that the same decision certificate and audit trace can be used by
AgentDojo, Agent-SafetyBench, and Agent Security Bench (ASB).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

THEORY_SEMANTIC_VERSION = "obligate-theory/1"

Polarity = Literal["+", "-"]
TriggerKind = Literal["gap", "hazard", "overflow"]
ExecutionEnv = Literal["host", "sandbox", "no_execute"]
NetworkScope = Literal["allow", "allowlist", "deny"]
DataScope = Literal["raw", "redact", "no_sensitive"]
HumanGate = Literal["none", "approval_required"]
PublicDecision = Literal[
    "allow",
    "execute_with_constraints",
    "require_confirmation",
    "block",
    "block_with_compliance_error",
]


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all bindings."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class SignedAtom:
    """A pre-grounded signed atom; no negation-as-failure is supported."""

    predicate: str
    arguments: tuple[str, ...] = ()
    polarity: Polarity = "+"

    def __post_init__(self) -> None:
        if not self.predicate or not self.predicate.strip():
            raise ValueError("predicate must be non-empty")
        if self.polarity not in {"+", "-"}:
            raise ValueError(f"unsupported polarity: {self.polarity!r}")
        object.__setattr__(self, "arguments", tuple(str(item) for item in self.arguments))

    @property
    def unsigned_key(self) -> str:
        if not self.arguments:
            return self.predicate
        return f"{self.predicate}({','.join(self.arguments)})"

    @property
    def key(self) -> str:
        return f"{self.polarity}{self.unsigned_key}"

    def opposite(self) -> "SignedAtom":
        return SignedAtom(self.predicate, self.arguments, "-" if self.polarity == "+" else "+")

    def to_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "arguments": list(self.arguments), "polarity": self.polarity}


@dataclass(frozen=True, slots=True)
class SignedRule:
    """A finite, function-free, range-restricted and already grounded rule."""

    rule_id: str
    premises: tuple[SignedAtom, ...]
    head: SignedAtom
    trusted_for_authorization: bool = False

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.premises:
            raise ValueError(f"rule {self.rule_id!r} must have at least one premise")
        object.__setattr__(self, "premises", tuple(self.premises))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "premises": [item.to_dict() for item in self.premises],
            "head": self.head.to_dict(),
            "trusted_for_authorization": self.trusted_for_authorization,
        }


def is_authorization_predicate(predicate: str) -> bool:
    return predicate in {"authorized", "authorizes_use"} or predicate.endswith("_authorized")


@dataclass(frozen=True, slots=True)
class EvidenceAssertion:
    atom: SignedAtom
    evidence_ref: str
    source_kind: str
    trusted_for_authorization: bool = False

    def __post_init__(self) -> None:
        if not self.evidence_ref:
            raise ValueError("evidence_ref must be non-empty")
        if not self.source_kind:
            raise ValueError("source_kind must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom": self.atom.to_dict(),
            "evidence_ref": self.evidence_ref,
            "source_kind": self.source_kind,
            "trusted_for_authorization": self.trusted_for_authorization,
        }


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    trust: Literal["trusted", "untrusted", "unknown"] = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "trust": self.trust,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class ActionGraph:
    graph_id: str
    root_action_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    assertions: tuple[EvidenceAssertion, ...]
    complete: bool = True
    parser_version: str = THEORY_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "assertions", tuple(self.assertions))
        node_ids = {item.node_id for item in self.nodes}
        if self.root_action_id not in node_ids:
            raise ValueError("root_action_id is not present in nodes")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"edge {edge.edge_id!r} references an unknown node")

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "root_action_id": self.root_action_id,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
            "assertions": [item.to_dict() for item in self.assertions],
            "complete": self.complete,
            "parser_version": self.parser_version,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Underspecified task contract C_U=(Permit, Deny, Unresolved)."""

    permit: frozenset[str] = frozenset()
    deny: frozenset[str] = frozenset()
    unresolved: frozenset[str] = frozenset()
    source_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "permit", frozenset(self.permit))
        object.__setattr__(self, "deny", frozenset(self.deny))
        object.__setattr__(self, "unresolved", frozenset(self.unresolved))
        if self.permit & self.deny or self.permit & self.unresolved or self.deny & self.unresolved:
            raise ValueError("TaskContract Permit, Deny, and Unresolved must be pairwise disjoint")

    def status(self, claim: str) -> Literal["permit", "deny", "unresolved"]:
        if claim in self.deny:
            return "deny"
        if claim in self.permit:
            return "permit"
        return "unresolved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "permit": sorted(self.permit),
            "deny": sorted(self.deny),
            "unresolved": sorted(self.unresolved),
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    tool_name: str
    arguments: Mapping[str, Any]
    actor: str = "benchmark_agent"
    tenant: str = "local"
    session: str = "default"
    side_effect: bool = False
    external_sink: bool = False
    state_mutation: bool = False
    sensitive_read: bool = False
    privileged_access: bool = False
    schema_known: bool = True
    payload_digest: str = ""
    asset_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty")
        object.__setattr__(self, "arguments", json.loads(canonical_json(dict(self.arguments))))
        object.__setattr__(self, "asset_digests", tuple(sorted(self.asset_digests)))

    @property
    def guarded(self) -> bool:
        return any(
            (
                self.side_effect,
                self.external_sink,
                self.state_mutation,
                self.sensitive_read,
                self.privileged_access,
                not self.schema_known,
            )
        )

    def normalized_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "actor": self.actor,
            "tenant": self.tenant,
            "session": self.session,
            "side_effect": self.side_effect,
            "external_sink": self.external_sink,
            "state_mutation": self.state_mutation,
            "sensitive_read": self.sensitive_read,
            "privileged_access": self.privileged_access,
            "schema_known": self.schema_known,
            "payload_digest": self.payload_digest,
            "asset_digests": list(self.asset_digests),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.normalized_dict())


@dataclass(frozen=True, order=True, slots=True)
class AuditUnit:
    field: str
    value_constraint: str
    transform: str
    sink: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "value_constraint": self.value_constraint,
            "transform": self.transform,
            "sink": self.sink,
        }


@dataclass(frozen=True, slots=True)
class AuditContract:
    must: frozenset[AuditUnit] = frozenset()
    allowed: frozenset[AuditUnit] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "must", frozenset(self.must))
        object.__setattr__(self, "allowed", frozenset(self.allowed))

    @property
    def consistent(self) -> bool:
        return self.must <= self.allowed

    def join(self, other: "AuditContract") -> "AuditContract":
        """Must uses union and Allowed uses intersection, as in the theory."""

        return AuditContract(must=self.must | other.must, allowed=self.allowed & other.allowed)

    def no_weaker_than(self, other: "AuditContract") -> bool:
        """Return True when this contract is at least as strong as ``other``."""

        return other.must <= self.must and self.allowed <= other.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "must": [item.to_dict() for item in sorted(self.must)],
            "allowed": [item.to_dict() for item in sorted(self.allowed)],
            "consistent": self.consistent,
        }


@dataclass(frozen=True, slots=True)
class BehaviorPlan:
    """The five-dimensional abstract control plan L_a."""

    execution_env: ExecutionEnv = "host"
    network_scope: NetworkScope = "allow"
    data_scope: DataScope = "raw"
    human_gate: HumanGate = "none"
    audit: AuditContract = AuditContract()

    def __post_init__(self) -> None:
        # ExecSem(no_execute, ...) has no dispatched trace.  Network, data and
        # approval coordinates are therefore semantically absorbed by block;
        # canonicalization makes the quotient explicit and prevents redundant
        # no_execute+approval_required plans.
        if self.execution_env == "no_execute":
            object.__setattr__(self, "network_scope", "deny")
            object.__setattr__(self, "data_scope", "no_sensitive")
            object.__setattr__(self, "human_gate", "none")

    @property
    def block_like(self) -> bool:
        return self.execution_env == "no_execute"

    @property
    def confirmation(self) -> bool:
        return self.human_gate == "approval_required" and not self.block_like

    @property
    def default_controls(self) -> bool:
        return (
            self.execution_env == "host"
            and self.network_scope == "allow"
            and self.data_scope == "raw"
            and self.human_gate == "none"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_env": self.execution_env,
            "network_scope": self.network_scope,
            "data_scope": self.data_scope,
            "human_gate": self.human_gate,
            "audit": self.audit.to_dict(),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def exec_semantics(plan: BehaviorPlan) -> frozenset[tuple[str, str, str, str]]:
    """Return the finite abstract execution traces admitted by ``plan``.

    Stronger controls admit a subset of traces: block admits none, approval
    admits only approved traces, and default execution admits both approved
    and unapproved traces.
    """

    if plan.block_like:
        return frozenset()
    env_order = ("host", "sandbox")
    network_order = ("allow", "allowlist", "deny")
    data_order = ("raw", "redact", "no_sensitive")
    human_order = ("none", "approval_required")
    environments = env_order[env_order.index(plan.execution_env) :]
    networks = network_order[network_order.index(plan.network_scope) :]
    data_scopes = data_order[data_order.index(plan.data_scope) :]
    human_states = human_order[human_order.index(plan.human_gate) :]
    return frozenset(
        (environment, network, data, human)
        for environment in environments
        for network in networks
        for data in data_scopes
        for human in human_states
    )


@dataclass(frozen=True, slots=True)
class SemanticObligation:
    obligation_id: str
    trigger_id: str
    description: str
    forbid_dispatch: bool = False
    forbid_confirmation: bool = False
    min_execution_env: ExecutionEnv | None = None
    min_network_scope: NetworkScope | None = None
    min_data_scope: DataScope | None = None
    require_approval: bool = False
    audit_must: frozenset[AuditUnit] = frozenset()
    bad_effect: str = "unsafe_default_dispatch"

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_must", frozenset(self.audit_must))

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "trigger_id": self.trigger_id,
            "description": self.description,
            "forbid_dispatch": self.forbid_dispatch,
            "forbid_confirmation": self.forbid_confirmation,
            "min_execution_env": self.min_execution_env,
            "min_network_scope": self.min_network_scope,
            "min_data_scope": self.min_data_scope,
            "require_approval": self.require_approval,
            "audit_must": [item.to_dict() for item in sorted(self.audit_must)],
            "bad_effect": self.bad_effect,
        }


@dataclass(frozen=True, slots=True)
class ImplementationConstraint:
    constraint_id: str
    trigger_id: str
    capability: str

    def to_dict(self) -> dict[str, str]:
        return {
            "constraint_id": self.constraint_id,
            "trigger_id": self.trigger_id,
            "capability": self.capability,
        }


@dataclass(frozen=True, slots=True)
class Trigger:
    trigger_id: str
    kind: TriggerKind
    subject: str
    hard: bool
    evidence_refs: tuple[str, ...]
    semantic_obligations: tuple[SemanticObligation, ...]
    implementation_constraints: tuple[ImplementationConstraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(sorted(set(self.evidence_refs))))
        object.__setattr__(self, "semantic_obligations", tuple(self.semantic_obligations))
        object.__setattr__(self, "implementation_constraints", tuple(self.implementation_constraints))
        if not self.semantic_obligations:
            raise ValueError(f"Trigger {self.trigger_id!r} has no semantic obligation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "kind": self.kind,
            "subject": self.subject,
            "hard": self.hard,
            "evidence_refs": list(self.evidence_refs),
            "semantic_obligations": [item.to_dict() for item in self.semantic_obligations],
            "implementation_constraints": [item.to_dict() for item in self.implementation_constraints],
        }


@dataclass(frozen=True, slots=True)
class Remedy:
    remedy_id: str
    trigger_id: str
    plan: BehaviorPlan
    explanation_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "remedy_id": self.remedy_id,
            "trigger_id": self.trigger_id,
            "plan": self.plan.to_dict(),
            "explanation_root": self.explanation_root,
        }


@dataclass(frozen=True, slots=True)
class Realization:
    realization_id: str
    plan: BehaviorPlan
    capabilities: frozenset[str]
    actual_audit: frozenset[AuditUnit]
    control_roots: tuple[tuple[str, str], ...]
    task_loss: int = 0
    confirmation_burden: int = 0
    runtime_cost: int = 0
    available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "actual_audit", frozenset(self.actual_audit))
        object.__setattr__(self, "control_roots", tuple(sorted(self.control_roots)))

    @property
    def utility_vector(self) -> tuple[int, int, int]:
        return (self.task_loss, self.confirmation_burden, self.runtime_cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "realization_id": self.realization_id,
            "plan": self.plan.to_dict(),
            "capabilities": sorted(self.capabilities),
            "actual_audit": [item.to_dict() for item in sorted(self.actual_audit)],
            "control_roots": [{"control": key, "root": value} for key, value in self.control_roots],
            "utility_vector": list(self.utility_vector),
            "available": self.available,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class PolicyAcceptanceRecord:
    policy_digest: str
    checker_version: str
    semantic_version: str
    vc_results: tuple[tuple[str, bool], ...]
    domain_stats: Mapping[str, Any]

    @property
    def accepted(self) -> bool:
        vc_ids = tuple(vc_id for vc_id, _ in self.vc_results)
        expected = frozenset(f"VC-{index}" for index in range(1, 19))
        return (
            self.semantic_version == THEORY_SEMANTIC_VERSION
            and len(vc_ids) == len(expected)
            and frozenset(vc_ids) == expected
            and all(passed for _, passed in self.vc_results)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_digest": self.policy_digest,
            "checker_version": self.checker_version,
            "semantic_version": self.semantic_version,
            "vc_results": {key: value for key, value in self.vc_results},
            "domain_stats": dict(self.domain_stats),
            "accepted": self.accepted,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ConfirmationGrant:
    """Trusted-state proof returned only after pending -> fresh."""

    nonce: str
    confirmation_binding_digest: str
    initial_certificate_digest: str

    def __post_init__(self) -> None:
        if not self.nonce or not self.confirmation_binding_digest or not self.initial_certificate_digest:
            raise ValueError("confirmation grant fields must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "nonce": self.nonce,
            "confirmation_binding_digest": self.confirmation_binding_digest,
            "initial_certificate_digest": self.initial_certificate_digest,
        }


@dataclass(frozen=True, slots=True)
class ActionCertificate:
    policy_acceptance_digest: str
    action_digest: str
    fact_digest: str
    evidence_digest: str
    plan_digest: str
    realization_digest: str
    outcome: PublicDecision
    nonce: str | None = None
    prior_certificate_digest: str | None = None
    confirmation_binding_digest: str | None = None
    issuer_id: str = ""
    issuer_proof: str = ""

    def signing_dict(self) -> dict[str, Any]:
        return {
            "policy_acceptance_digest": self.policy_acceptance_digest,
            "action_digest": self.action_digest,
            "fact_digest": self.fact_digest,
            "evidence_digest": self.evidence_digest,
            "plan_digest": self.plan_digest,
            "realization_digest": self.realization_digest,
            "outcome": self.outcome,
            "nonce": self.nonce,
            "prior_certificate_digest": self.prior_certificate_digest,
            "confirmation_binding_digest": self.confirmation_binding_digest,
            "issuer_id": self.issuer_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.signing_dict(), "issuer_proof": self.issuer_proof}

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TheoryDecision:
    public_decision: PublicDecision
    execute: bool
    triggers: tuple[Trigger, ...]
    semantic_obligations: tuple[SemanticObligation, ...]
    implementation_constraints: tuple[ImplementationConstraint, ...]
    ideal: tuple[BehaviorPlan, ...]
    safe_candidates: tuple[BehaviorPlan, ...]
    feasible_realizations: tuple[Realization, ...]
    frontier: tuple[Realization, ...]
    selected: Realization | None
    certificate: ActionCertificate
    trace: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_decision": self.public_decision,
            "execute": self.execute,
            "triggers": [item.to_dict() for item in self.triggers],
            "semantic_obligations": [item.to_dict() for item in self.semantic_obligations],
            "implementation_constraints": [item.to_dict() for item in self.implementation_constraints],
            "ideal": [item.to_dict() for item in self.ideal],
            "safe_candidates": [item.to_dict() for item in self.safe_candidates],
            "feasible_realizations": [item.to_dict() for item in self.feasible_realizations],
            "frontier": [item.to_dict() for item in self.frontier],
            "selected": self.selected.to_dict() if self.selected else None,
            "certificate": self.certificate.to_dict(),
            "certificate_digest": self.certificate.digest,
            "trace": dict(self.trace),
        }


__all__ = [
    "THEORY_SEMANTIC_VERSION",
    "ActionCandidate",
    "ActionCertificate",
    "ActionGraph",
    "AuditContract",
    "AuditUnit",
    "BehaviorPlan",
    "ConfirmationGrant",
    "DataScope",
    "EvidenceAssertion",
    "ExecutionEnv",
    "GraphEdge",
    "GraphNode",
    "HumanGate",
    "ImplementationConstraint",
    "NetworkScope",
    "PolicyAcceptanceRecord",
    "PublicDecision",
    "Realization",
    "Remedy",
    "SemanticObligation",
    "SignedAtom",
    "SignedRule",
    "TaskContract",
    "TheoryDecision",
    "Trigger",
    "canonical_json",
    "exec_semantics",
    "is_authorization_predicate",
    "sha256_json",
]
