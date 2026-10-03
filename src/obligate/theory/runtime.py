"""End-to-end theory-aligned decision engine.

This module is the benchmark-independent execution path:

ActionGraph -> bipolar witness closure -> EOC Trigger compilation -> lifted
ECP -> realization/frontier selection -> per-action certificate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import RLock
from typing import Iterable, Mapping, Sequence

from .issuer import CertificateIssuer, default_certificate_issuer
from .lattice import EnforceableControlPlanner, lifted_join_all
from .model import (
    ActionCandidate,
    ActionCertificate,
    ActionGraph,
    ConfirmationGrant,
    EvidenceAssertion,
    GraphEdge,
    GraphNode,
    PolicyAcceptanceRecord,
    Realization,
    SignedAtom,
    SignedRule,
    TaskContract,
    TheoryDecision,
    canonical_json,
    is_authorization_predicate,
    sha256_json,
)
from .eoc import HazardSignal, EOCCompiler
from .policy import PolicySpec, build_default_policy
from .realizer import BoundRealizer, Realizer
from .witness import BipolarWitnessEngine

_POLICY_REPLAY_LOCK = RLock()
_POLICY_REPLAY_CACHE: dict[str, tuple[str, PolicyAcceptanceRecord]] = {}


def _replay_checked_policy(policy: PolicySpec) -> PolicyAcceptanceRecord:
    """Replay an artifact once per process and bind cache hits to canonical bytes."""

    payload = canonical_json(policy.to_dict())
    with _POLICY_REPLAY_LOCK:
        cached = _POLICY_REPLAY_CACHE.get(policy.digest)
        if cached is not None:
            cached_payload, acceptance = cached
            if cached_payload != payload:  # cryptographic collision or corrupted registry
                raise ValueError("policy digest collision in verified-policy registry")
            return acceptance

        from .checker import check_policy

        replay = check_policy(policy)
        if not replay.accepted:
            failures = [item.vc_id for item in replay.failures]
            raise ValueError(f"runtime policy is not accepted: {failures}")
        _POLICY_REPLAY_CACHE[policy.digest] = (payload, replay.acceptance)
        return replay.acceptance


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    contract: TaskContract
    assertions: tuple[EvidenceAssertion, ...]
    hazards: tuple[HazardSignal, ...] = ()
    explicit_denies: tuple[HazardSignal, ...] = ()
    graph_nodes: tuple[GraphNode, ...] = ()
    graph_edges: tuple[GraphEdge, ...] = ()
    facts: Mapping[str, object] | None = None
    confirmation: ConfirmationGrant | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assertions", tuple(self.assertions))
        object.__setattr__(self, "hazards", tuple(self.hazards))
        object.__setattr__(self, "explicit_denies", tuple(self.explicit_denies))
        object.__setattr__(self, "graph_nodes", tuple(self.graph_nodes))
        object.__setattr__(self, "graph_edges", tuple(self.graph_edges))


class TheoryRuntime:
    """Deterministic implementation of the specification's eight online steps."""

    def __init__(
        self,
        *,
        acceptance: PolicyAcceptanceRecord,
        policy: PolicySpec | None = None,
        rules: Sequence[SignedRule] | None = None,
        witness_limit: int | None = None,
        max_candidates: int | None = None,
        capabilities: Iterable[str] | None = None,
        realizer: Realizer | None = None,
        certificate_issuer: CertificateIssuer | None = None,
    ) -> None:
        checked_policy = policy or build_default_policy()
        replay_acceptance = _replay_checked_policy(checked_policy)
        if replay_acceptance.digest != acceptance.digest:
            raise ValueError("acceptance record does not bind the replay-checked runtime policy")
        if rules is not None and tuple(rules) != checked_policy.rules:
            raise ValueError("runtime rules differ from the replay-checked PolicySpec")
        self.policy = checked_policy
        self.acceptance = replay_acceptance
        self.rules = checked_policy.rules
        self.loader_replay_digest = replay_acceptance.digest
        self.witness_limit = witness_limit
        self.max_candidates = max_candidates
        if capabilities is not None and realizer is not None:
            raise ValueError("configure either capabilities or a concrete realizer, not both")
        if capabilities is not None:
            declared_capabilities = frozenset(capabilities)
        elif realizer is not None:
            declared_capabilities = frozenset()
        else:
            declared_capabilities = frozenset(
                ("host", "sandbox", "redaction", "confirmation", "audit")
            )
        self.realizer = realizer or BoundRealizer.from_capabilities(declared_capabilities)
        self.capabilities = declared_capabilities
        self.certificate_issuer = certificate_issuer or default_certificate_issuer()
        self.compiler = EOCCompiler(policy=checked_policy)
        self.control_planner = EnforceableControlPlanner()

    def decide(self, action: ActionCandidate, evidence: RuntimeEvidence, *, nonce: str | None = None) -> TheoryDecision:
        if nonce is not None and evidence.confirmation is None:
            raise ValueError("a certificate nonce requires a trusted confirmation grant")
        if evidence.confirmation is not None and nonce not in {None, evidence.confirmation.nonce}:
            raise ValueError("confirmation grant nonce mismatch")
        graph = build_runtime_action_graph(action, evidence)
        requirements = requirements_for_action(action)
        assertions = tuple(
            item
            for item in evidence.assertions
            if not (
                item.atom.polarity == "+"
                and is_authorization_predicate(item.atom.predicate)
                and not item.trusted_for_authorization
            )
        )
        rejected_enabling_assertions = tuple(
            item for item in evidence.assertions if item not in assertions
        )
        hazards = list(evidence.hazards)
        explicit_denies = list(evidence.explicit_denies)
        for requirement in requirements:
            if requirement.unsigned_key in evidence.contract.deny:
                explicit_denies.append(
                    HazardSignal(
                        signal_id=f"task_contract_deny:{requirement.unsigned_key}",
                        subject=f"TaskContract explicitly denies {requirement.unsigned_key}",
                        evidence_refs=(
                            evidence.contract.source_digest or "task_contract:deny",
                        ),
                        hard=True,
                        kind="hazard",
                    )
                )
        if action.guarded and not self.acceptance.accepted:
            explicit_denies.append(
                HazardSignal(
                    signal_id="policy_not_accepted",
                    subject="policy acceptance record is missing or rejected",
                    evidence_refs=(self.acceptance.digest,),
                    hard=True,
                    kind="hazard",
                )
            )

        closure = BipolarWitnessEngine(witness_limit=self.witness_limit).close(assertions, self.rules)
        eoc = self.compiler.compile(
            requirements=requirements,
            closure=closure,
            hazards=hazards,
            explicit_denies=explicit_denies,
        )
        default_plan = self.policy.default_plan
        trigger_remedies = [tuple(remedy.plan for remedy in eoc.remedies[trigger.trigger_id]) for trigger in eoc.triggers]
        ideal_preview = lifted_join_all(((default_plan,), *trigger_remedies))
        policy_plan_digests = {item.digest for item in self.policy.behavior_domain}
        if any(item.digest not in policy_plan_digests for item in ideal_preview):
            raise RuntimeError("checked behavior domain is not closed under the online lifted join")
        plan_library = self.policy.behavior_domain
        realizations = tuple(self._materialize(item) for item in self.policy.realizations)
        selection = self.control_planner.select(
            base_remedies=(default_plan,),
            trigger_remedies=trigger_remedies,
            obligations=eoc.semantic_obligations,
            implementation_constraints=eoc.implementation_constraints,
            plan_library=plan_library,
            realizations=realizations,
            default_plan=default_plan,
            max_candidates=self.max_candidates,
            approval_satisfied=evidence.confirmation is not None,
        )
        if selection.search_overflow and selection.outcome not in {"block", "block_with_compliance_error"}:
            raise AssertionError("bounded ECP search returned a non-fail-stop result")

        selected = selection.selected
        plan_digest = selected.plan.digest if selected else sha256_json({"plan": "compliance_error"})
        realization_digest = selected.digest if selected else sha256_json({"realization": "none"})
        fact_payload = {
            "policy_digest": self.policy.digest,
            "contract": evidence.contract.to_dict(),
            "runtime_facts": dict(evidence.facts or {}),
            "confirmation": evidence.confirmation.to_dict() if evidence.confirmation else None,
            "witness": closure.to_dict(),
            "eoc": eoc.to_dict(),
        }
        certificate = self.certificate_issuer.issue(
            ActionCertificate(
                policy_acceptance_digest=self.acceptance.digest,
                action_digest=action.digest,
                fact_digest=sha256_json(fact_payload),
                evidence_digest=graph.digest,
                plan_digest=plan_digest,
                realization_digest=realization_digest,
                outcome=selection.outcome,
                nonce=evidence.confirmation.nonce if evidence.confirmation else None,
                prior_certificate_digest=(
                    evidence.confirmation.initial_certificate_digest
                    if evidence.confirmation
                    else None
                ),
                confirmation_binding_digest=(
                    evidence.confirmation.confirmation_binding_digest
                    if evidence.confirmation
                    else None
                ),
            )
        )
        trace = {
            "semantic_version": self.acceptance.semantic_version,
            "policy_digest": self.policy.digest,
            "policy_acceptance_digest": self.acceptance.digest,
            "loader_replay_digest": self.loader_replay_digest,
            "guarded": action.guarded,
            "approval_satisfied": evidence.confirmation is not None,
            "action_graph": graph.to_dict(),
            "witness_closure": closure.to_dict(),
            "rejected_enabling_assertions": [item.to_dict() for item in rejected_enabling_assertions],
            "gap": list(eoc.gaps),
            "hazard": list(eoc.hazards),
            "overflow": list(eoc.overflow),
            "valid_certificate_evidence": eoc.valid_certificate_evidence,
            "eoc": eoc.to_dict(),
            "lattice": selection.to_dict(),
            "realizer": dict(self.realizer.manifest()),
            "external_tcb": [
                "TaskContractSound",
                "FactRefines",
                "ToolConforms",
                "CompleteMediation",
                "RealizerRefines",
                "TrustedMonitor",
                "TrustedStateStore",
            ],
        }
        return TheoryDecision(
            public_decision=selection.outcome,
            execute=selection.execute,
            triggers=eoc.triggers,
            semantic_obligations=eoc.semantic_obligations,
            implementation_constraints=eoc.implementation_constraints,
            ideal=selection.ideal,
            safe_candidates=selection.safe_candidates,
            feasible_realizations=selection.feasible_realizations,
            frontier=selection.frontier,
            selected=selected,
            certificate=certificate,
            trace=trace,
        )

    def _materialize(self, template: Realization) -> Realization:
        return self.realizer.materialize(template)


def requirements_for_action(action: ActionCandidate) -> tuple[SignedAtom, ...]:
    if not action.guarded:
        return ()
    action_arg = (action.tool_name,)
    requirements = [
        SignedAtom("schema_conforms", action_arg, "+"),
        SignedAtom("action_authorized", action_arg, "+"),
    ]
    if action.side_effect or action.external_sink or action.state_mutation:
        requirements.append(SignedAtom("target_authorized", action_arg, "+"))
    if action.external_sink or action.sensitive_read:
        requirements.append(SignedAtom("flow_authorized", action_arg, "+"))
    if action.privileged_access:
        requirements.append(SignedAtom("privilege_authorized", action_arg, "+"))
    return tuple(requirements)


def build_runtime_action_graph(action: ActionCandidate, evidence: RuntimeEvidence) -> ActionGraph:
    digest = action.digest[:16]
    root = f"action:{digest}"
    contract_id = f"contract:{hashlib.sha256(str(evidence.contract.to_dict()).encode()).hexdigest()[:16]}"
    nodes: list[GraphNode] = [
        GraphNode(root, "CandidateAction", action.tool_name, "unknown", {"action_digest": action.digest}),
        GraphNode(contract_id, "TaskContract", "task contract", "trusted", evidence.contract.to_dict()),
    ]
    edges: list[GraphEdge] = []
    for index, (key, value) in enumerate(sorted(action.arguments.items())):
        node_id = f"param:{digest}:{index}"
        nodes.append(GraphNode(node_id, "Param", key, "unknown", {"value_digest": sha256_json(value)}))
        edges.append(GraphEdge(f"edge:param:{index}", node_id, root, "occurs_in", (f"argument:{key}",)))
    nodes.extend(evidence.graph_nodes)
    edges.extend(evidence.graph_edges)
    if evidence.confirmation is not None:
        confirmation_id = f"confirmation:{evidence.confirmation.confirmation_binding_digest[:20]}"
        nodes.append(
            GraphNode(
                confirmation_id,
                "ConfirmationGrant",
                "bound approval",
                "trusted",
                evidence.confirmation.to_dict(),
            )
        )
        edges.append(
            GraphEdge(
                "edge:confirmation-action",
                confirmation_id,
                root,
                "supports",
                (f"confirmation:{evidence.confirmation.nonce}",),
            )
        )
    # Task contracts authorize claims; mere occurrence edges never do.
    edges.append(GraphEdge("edge:contract-action", contract_id, root, "supports", ("task_contract",)))
    graph_id = f"graph:{sha256_json({'action': action.digest, 'assertions': [x.to_dict() for x in evidence.assertions], 'nodes': [x.to_dict() for x in evidence.graph_nodes], 'edges': [x.to_dict() for x in evidence.graph_edges], 'confirmation': evidence.confirmation.to_dict() if evidence.confirmation else None})[:20]}"
    return ActionGraph(
        graph_id=graph_id,
        root_action_id=root,
        nodes=tuple(nodes),
        edges=tuple(edges),
        assertions=evidence.assertions,
        complete=True,
    )


def default_policy_acceptance(*, policy_digest: str = "development-policy") -> PolicyAcceptanceRecord:
    """A rejected sentinel used until the finite checker supplies a real kappa."""

    return PolicyAcceptanceRecord(
        policy_digest=policy_digest,
        checker_version="unverified",
        semantic_version="obligate-theory/1",
        vc_results=(("unverified", False),),
        domain_stats={},
    )


__all__ = [
    "RuntimeEvidence",
    "TheoryRuntime",
    "build_runtime_action_graph",
    "default_policy_acceptance",
    "requirements_for_action",
]
