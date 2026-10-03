from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Iterable

from obligate.theory.benchmark_adapter import (
    compatibility_public_decision,
    from_tool_boundary,
    to_adjudication_result,
)
from obligate.theory.fact_extraction import oracle_free_tool_context
from obligate.theory.gate import GateResult, ToolGate
from obligate.theory.model import ActionCandidate, Realization, TheoryDecision, sha256_json
from obligate.theory.eoc import HazardSignal
from obligate.theory.runtime import RuntimeEvidence, TheoryRuntime

from ..compat.types import ConfirmationMode, SanitizeMode, ToolCallContext, ablation_config_from_profile
from ..evidence.action_graph import AgentDojoActionGraphBuilder
from ..evidence.evidence import AgentDojoEvidenceBuilder
from ..evidence.fusion import AgentDojoEvidenceFusion, FusionResult
from ..evidence.state import AgentDojoStateTracker
from ..evidence.taxonomy import AgentDojoToolTaxonomy, infer_unknown_tool
from .blocked_result import BlockedActionTracker, build_blocked_tool_result, build_confirmation_required_result

_ABLATION_MODULES = (
    "provenance",
    "task_contract",
    "action_graph",
    "suite_policy",
    "recovery_guidance",
    "generic_sink_policy",
    "actiongraph_structure_edges",
    "actiongraph_provenance_edges",
    "actiongraph_dataflow_edges",
    "actiongraph_history_edges",
)


def _digest_value(value: Any) -> str:
    try:
        return sha256_json(value)
    except (TypeError, ValueError):
        return sha256_json(
            {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr_digest": sha256_json(repr(value))}
        )


def _safe_identifier(value: str) -> str:
    candidate = str(value)
    if (
        0 < len(candidate) <= 128
        and candidate.isascii()
        and all(character.isalnum() or character in "._:-/" for character in candidate)
    ):
        return candidate
    return f"sha256:{sha256_json(candidate)}"


def _safe_identifier_list(values: Iterable[Any]) -> list[str]:
    return sorted({_safe_identifier(str(value)) for value in values})


@dataclass(slots=True)
class ToolExecutionDecision:
    execute: bool
    decision: str
    reason_codes: list[str]
    safe_result: Any | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    action_graph_id: str | None = None
    action_graph_facts: dict[str, Any] = field(default_factory=dict)
    fusion_result: FusionResult | None = None
    repeated_unsafe_action: bool = False
    decision_metadata: dict[str, Any] = field(default_factory=dict)
    theory_decision: TheoryDecision | None = None
    theory_action: ActionCandidate | None = field(default=None, repr=False)
    theory_evidence: RuntimeEvidence | None = field(default=None, repr=False)

    def to_audit_event(self) -> dict[str, Any]:
        theory = self.theory_decision
        trace = theory.trace if theory is not None else {}
        certificate = theory.certificate if theory is not None else None
        findings = self.evidence.get("policy_engine_findings", [])
        finding_ids: list[str] = []
        for finding in findings if isinstance(findings, list) else []:
            if not isinstance(finding, dict):
                continue
            finding_ids.extend(
                _safe_identifier(value)
                for value in finding.get("reason_codes", [])
                if isinstance(value, str)
            )
        safe_metadata = {
            key: value
            for key, value in self.decision_metadata.items()
            if key
            in {
                "confirmation_mode",
                "confirmation_required",
                "confirmation_executed",
                "gateway_confirmation_counted_separately",
                "ablation_profile",
                "theory_public_decision",
                "theory_semantic_version",
                "certificate_digest",
            }
            and isinstance(value, (str, bool, int, float, type(None)))
        }
        ablation_config = self.evidence.get("ablation_config", {})
        safe_ablation = {
            _safe_identifier(str(key)): bool(value)
            for key, value in ablation_config.items()
            if isinstance(key, str) and key.startswith("enable_")
        } if isinstance(ablation_config, dict) else {}
        return {
            "event_type": "agentdojo_tool_gate_decision",
            "execute": self.execute,
            "decision": self.decision,
            "reason_codes": _safe_identifier_list(self.reason_codes),
            "action_digest": certificate.action_digest if certificate else _digest_value({"decision": self.decision}),
            "evidence_digest": certificate.evidence_digest if certificate else _digest_value({"reason_codes": self.reason_codes}),
            "plan_digest": certificate.plan_digest if certificate else _digest_value({"execute": self.execute}),
            "realization_digest": certificate.realization_digest if certificate else "",
            "policy_acceptance_digest": certificate.policy_acceptance_digest if certificate else "",
            "certificate_digest": certificate.digest if certificate else "",
            "result_digest": "",
            "reason_digest": "",
            "theory_public_decision": theory.public_decision if theory else "",
            "trigger_ids": _safe_identifier_list(item.trigger_id for item in theory.triggers) if theory else [],
            "valid_certificate_evidence": bool(trace.get("valid_certificate_evidence", False)),
            "gap_count": len(trace.get("gap", [])) if isinstance(trace.get("gap", []), list) else 0,
            "hazard_count": len(trace.get("hazard", [])) if isinstance(trace.get("hazard", []), list) else 0,
            "overflow_count": len(trace.get("overflow", [])) if isinstance(trace.get("overflow", []), list) else 0,
            "repeated_unsafe_action": self.repeated_unsafe_action,
            "modules_executed": _safe_identifier_list(self.evidence.get("modules_executed", [])),
            "modules_skipped": _safe_identifier_list(self.evidence.get("modules_skipped", [])),
            "matched_rules": _safe_identifier_list(self.reason_codes),
            "matched_invariants": _safe_identifier_list(self.evidence.get("matched_invariants", [])),
            "confirmation_required": bool(self.decision_metadata.get("confirmation_required", False)),
            "confirmation_executed": bool(self.decision_metadata.get("confirmation_executed", False)),
            "gateway_confirmation_counted_separately": bool(
                self.decision_metadata.get("gateway_confirmation_counted_separately", False)
            ),
            "decision_metadata": safe_metadata,
            "policy_engines_executed": _safe_identifier_list(self.evidence.get("policy_engines_executed", [])),
            "policy_engine_finding_ids": sorted(set(finding_ids)),
            "ablation_profile": _safe_identifier(str(self.evidence.get("ablation_profile") or "default")),
            "ablation_config": safe_ablation,
            "tool_registered": not bool(self.evidence.get("agentdojo.unknown_tool", False)),
            "gate_state": "not_dispatched",
            "gate_dispatched": False,
            "outbox_status": "",
            "sanitized": False,
            "observe_ms": 0.0,
        }


class AgentDojoToolFirewall:
    """Tool firewall for AgentDojo benchmark runs."""

    def __init__(
        self,
        *,
        taxonomy: AgentDojoToolTaxonomy | None = None,
        state: AgentDojoStateTracker | None = None,
        graph_builder: AgentDojoActionGraphBuilder | None = None,
        evidence_builder: AgentDojoEvidenceBuilder | None = None,
        fusion: AgentDojoEvidenceFusion | None = None,
        sanitize_outputs: bool = True,
        sanitize_mode: SanitizeMode = "soft",
        eval_mode: bool = True,
        enable_taxonomy: bool = True,
        enable_state_tracker: bool = True,
        enable_action_graph: bool = True,
        enable_task_contract: bool = True,
        enable_invariants: bool = True,
        enable_recovery_guidance: bool = True,
        confirmation_mode: ConfirmationMode = "strict_eval",
        ablation_profile: str = "full",
        enable_theory_core: bool = True,
        theory_runtime: TheoryRuntime | None = None,
        theory_gate: ToolGate | None = None,
        theory_capabilities: Iterable[str] = ("host", "confirmation", "audit"),
    ) -> None:
        self.taxonomy = taxonomy or AgentDojoToolTaxonomy()
        self.state = state or AgentDojoStateTracker()
        self.state.sanitize_mode = sanitize_mode
        self.graph_builder = graph_builder or AgentDojoActionGraphBuilder()
        self.evidence_builder = evidence_builder or AgentDojoEvidenceBuilder()
        self.fusion = fusion or AgentDojoEvidenceFusion(eval_mode=eval_mode)
        self.sanitize_outputs = sanitize_outputs
        self.sanitize_mode = sanitize_mode
        self.eval_mode = eval_mode
        self.audit_events: list[dict[str, Any]] = []
        self.base_ablation_config = {
            "enable_taxonomy": enable_taxonomy,
            "enable_state_tracker": enable_state_tracker,
            "enable_action_graph": enable_action_graph,
            "enable_task_contract": enable_task_contract,
            "enable_invariants": enable_invariants,
            "enable_recovery_guidance": enable_recovery_guidance,
        }
        self._blocked_tracker = BlockedActionTracker()
        self.confirmation_mode = confirmation_mode
        self.ablation_profile = ablation_profile
        self.enable_theory_core = enable_theory_core
        self.theory_runtime = theory_runtime or (_build_default_theory_runtime(theory_capabilities) if enable_theory_core else None)
        self.theory_gate = (
            theory_gate
            if theory_gate is not None
            else ToolGate(
                policy=self.theory_runtime.policy,
                acceptance=self.theory_runtime.acceptance,
                realizer=self.theory_runtime.realizer,
                certificate_issuer=self.theory_runtime.certificate_issuer,
            )
            if self.theory_runtime is not None
            else None
        )

    def reset_case_state(self) -> None:
        """Start a clean evidence history without discarding accumulated audit."""

        self.state = AgentDojoStateTracker()
        self.state.sanitize_mode = self.sanitize_mode
        self._blocked_tracker = BlockedActionTracker()
        self.theory_gate = (
            ToolGate(
                policy=self.theory_runtime.policy,
                acceptance=self.theory_runtime.acceptance,
                realizer=self.theory_runtime.realizer,
                certificate_issuer=self.theory_runtime.certificate_issuer,
            )
            if self.theory_runtime is not None
            else None
        )

    def fork_for_case(self) -> "AgentDojoToolFirewall":
        """Return an isolated firewall suitable for concurrent benchmark cases."""

        return AgentDojoToolFirewall(
            taxonomy=self.taxonomy,
            state=AgentDojoStateTracker(),
            graph_builder=self.graph_builder,
            evidence_builder=self.evidence_builder,
            fusion=self.fusion,
            sanitize_outputs=self.sanitize_outputs,
            sanitize_mode=self.sanitize_mode,
            eval_mode=self.eval_mode,
            enable_taxonomy=bool(self.base_ablation_config.get("enable_taxonomy", True)),
            enable_state_tracker=bool(self.base_ablation_config.get("enable_state_tracker", True)),
            enable_action_graph=bool(self.base_ablation_config.get("enable_action_graph", True)),
            enable_task_contract=bool(self.base_ablation_config.get("enable_task_contract", True)),
            enable_invariants=bool(self.base_ablation_config.get("enable_invariants", True)),
            enable_recovery_guidance=bool(self.base_ablation_config.get("enable_recovery_guidance", True)),
            confirmation_mode=self.confirmation_mode,
            ablation_profile=self.ablation_profile,
            enable_theory_core=self.enable_theory_core,
            theory_runtime=self.theory_runtime,
        )

    def guard_before_tool(self, context: ToolCallContext) -> ToolExecutionDecision:
        started = time.perf_counter()
        if self.enable_theory_core:
            # Construct the deployable decision view before taxonomy, state,
            # contract, graph, or fact extraction.  Diagnostic benchmark labels
            # remain available to the scorer but cannot reach the theory core.
            context = oracle_free_tool_context(context)
        modules_executed: list[str] = []
        modules_skipped: list[str] = []
        profile = str(context.ablation_config.get("profile") or self.ablation_profile or "full")
        profile_config = ablation_config_from_profile(profile)
        ablation_config = {**profile_config.as_dict(), **context.ablation_config}
        for key, enabled in self.base_ablation_config.items():
            if key in {"enable_action_graph", "enable_task_contract", "enable_recovery_guidance"}:
                ablation_config[key] = bool(ablation_config.get(key, True)) and bool(enabled)
            else:
                ablation_config.setdefault(key, enabled)
        if "profile" not in ablation_config:
            ablation_config["profile"] = profile

        def module_enabled(name: str) -> bool:
            key = f"enable_{name}"
            return bool(ablation_config.get(key, True))

        if module_enabled("taxonomy"):
            modules_executed.append("taxonomy")
            spec = self.taxonomy.classify(context.tool_name, suite=context.suite)
        else:
            modules_skipped.append("taxonomy")
            spec = infer_unknown_tool(context.tool_name)
        if not self.enable_theory_core and context.defense_mode == "oracle_full":
            for signature in context.attack_goal_signatures:
                self.state.add_attack_goal_signature(signature)
        if module_enabled("state_tracker"):
            modules_executed.append("state_tracker")
            self.state.observe_tool_call(context.tool_name, spec, context.tool_args)
        else:
            modules_skipped.append("state_tracker")
        initial = self.evidence_builder.build(context=context, spec=spec, state=self.state)
        relation_graph = None
        if module_enabled("action_graph"):
            modules_executed.append("action_graph")
            graph_result = self.graph_builder.build(context=context, spec=spec, state=self.state, evidence=initial)
            relation_graph = graph_result.graph
            graph_facts = graph_result.facts
            action_graph_id = graph_result.graph.graph_id
        else:
            modules_skipped.append("action_graph")
            graph_facts = {}
            action_graph_id = None
        evidence = self.evidence_builder.build(context=context, spec=spec, state=self.state, graph_facts=graph_facts)
        evidence.action_graph_id = action_graph_id
        for name in ("task_contract", "invariants"):
            if module_enabled(name):
                modules_executed.append(name)
            else:
                modules_skipped.append(name)
        evidence.facts["ablation_config"] = ablation_config
        evidence.facts["ablation_profile"] = str(ablation_config.get("profile") or profile)
        evidence.facts["modules_executed"] = modules_executed
        evidence.facts["modules_skipped"] = modules_skipped
        evidence.facts["modules_enabled"] = [name for name in _ABLATION_MODULES if bool(ablation_config.get(f"enable_{name}", True))]
        evidence.facts["modules_disabled"] = [name for name in _ABLATION_MODULES if not bool(ablation_config.get(f"enable_{name}", True))]
        evidence.facts["matched_invariants"] = [] if not module_enabled("invariants") else evidence.facts.get("matched_invariants", [])
        theory_decision: TheoryDecision | None = None
        if self.enable_theory_core:
            if self.theory_runtime is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("theory core is enabled without a verified runtime")
            modules_executed.extend(("bipolar_witness", "trigger_compiler", "lifted_ecp", "action_certificate"))
            canonical = from_tool_boundary(
                context=context,
                spec=spec,
                evidence=evidence,
                relation_graph=relation_graph,
            )
            if (
                self.confirmation_mode in {"oracle_user_eval", "gateway_eval"}
                and spec.group == "membership_mutation"
            ):
                confirmation_signal = HazardSignal(
                    "ADJ-MEMBERSHIP-CONFIRM",
                    "membership expansion requires a bound confirmation in this runtime profile",
                    ("baseline:membership_confirmation",),
                    preferred_control="require_confirmation",
                )
                canonical = replace(
                    canonical,
                    evidence=replace(
                        canonical.evidence,
                        hazards=(*canonical.evidence.hazards, confirmation_signal),
                    ),
                )
            if (
                self.confirmation_mode in {"oracle_user_eval", "gateway_eval"}
                and context.suite == "travel"
                and spec.group == "external_send"
                and evidence.facts.get("agentdojo.delegated_target_selection")
            ):
                confirmation_signal = HazardSignal(
                    "ADJ-TRAVEL-CONFIRM-AUTHORIZED-SEND",
                    "derived travel content sent externally requires confirmation in this runtime profile",
                    ("baseline:travel_external_confirmation",),
                    preferred_control="require_confirmation",
                )
                canonical = replace(
                    canonical,
                    evidence=replace(
                        canonical.evidence,
                        hazards=(*canonical.evidence.hazards, confirmation_signal),
                    ),
                )
            theory_decision = self.theory_runtime.decide(canonical.action, canonical.evidence)
            reason_aliases = [
                signal.signal_id
                for signal in (*canonical.evidence.hazards, *canonical.evidence.explicit_denies)
                if signal.signal_id.startswith(("ASB-", "ADJ-"))
            ]
            fusion = to_adjudication_result(
                theory_decision,
                base_facts=dict(canonical.evidence.facts or {}),
                reason_aliases=reason_aliases,
            )
            execute = theory_decision.execute
            public_decision = compatibility_public_decision(theory_decision)
            decision_metadata = {
                "confirmation_mode": self.confirmation_mode,
                "confirmation_required": theory_decision.public_decision == "require_confirmation",
                # Confirmation is a pause.  A later, newly adjudicated action
                # may dispatch through the theory ToolGate; this step never can.
                "confirmation_executed": False,
                "ablation_profile": evidence.facts.get("ablation_profile"),
                "theory_public_decision": theory_decision.public_decision,
                "theory_semantic_version": theory_decision.trace.get("semantic_version"),
                "certificate_digest": theory_decision.certificate.digest,
            }
            if self.confirmation_mode == "gateway_eval" and theory_decision.public_decision == "require_confirmation":
                decision_metadata["gateway_confirmation_counted_separately"] = True
            if module_enabled("action_graph"):
                action_graph_id = str(theory_decision.trace["action_graph"]["graph_id"])
            evidence.facts.clear()
            evidence.facts.update(fusion.facts)
            evidence.facts["modules_executed"] = modules_executed
        else:
            fusion = self.fusion.decide(evidence)
            execute, public_decision, decision_metadata = self._resolve_execution_for_decision(
                fusion=fusion,
                context=context,
                evidence=evidence.facts,
                action_graph_facts=graph_facts,
            )
        repeated = False
        safe = None
        if not execute:
            if public_decision == "require_confirmation":
                safe = build_confirmation_required_result(
                    context,
                    fusion,
                    recovery_guidance_enabled=module_enabled("recovery_guidance"),
                )
            else:
                safe = build_blocked_tool_result(
                    context,
                    fusion,
                    recovery_guidance_enabled=module_enabled("recovery_guidance"),
                )
            retry_count = self._blocked_tracker.record(str(safe["same_action_retry_key"]))
            repeated = retry_count > 1
            if repeated and public_decision != "require_confirmation":
                safe = build_blocked_tool_result(
                    context,
                    fusion,
                    repeated_unsafe_action=True,
                    recovery_guidance_enabled=module_enabled("recovery_guidance"),
                )
            self.state.observe_tool_call(
                context.tool_name,
                spec,
                context.tool_args,
                event_status="blocked",
                decision=public_decision,
                execute=False,
                reason_codes=fusion.reason_codes,
                same_action_retry_key=str(safe["same_action_retry_key"]),
            )
        decision = ToolExecutionDecision(
            execute=execute,
            decision=public_decision,
            reason_codes=fusion.reason_codes,
            safe_result=safe,
            evidence=evidence.facts,
            action_graph_id=action_graph_id,
            action_graph_facts=graph_facts,
            fusion_result=fusion,
            repeated_unsafe_action=repeated,
            decision_metadata=decision_metadata,
            theory_decision=theory_decision,
            theory_action=canonical.action if theory_decision is not None else None,
            theory_evidence=canonical.evidence if theory_decision is not None else None,
        )
        event = decision.to_audit_event()
        event["policy_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        self.audit_events.append(event)
        return decision

    def _resolve_execution_for_decision(
        self,
        *,
        fusion: FusionResult,
        context: ToolCallContext,
        evidence: dict[str, Any],
        action_graph_facts: dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any]]:
        metadata: dict[str, Any] = {
            "confirmation_mode": self.confirmation_mode,
            "confirmation_required": fusion.decision == "require_confirmation",
            "confirmation_executed": False,
            "ablation_profile": evidence.get("ablation_profile"),
        }
        if fusion.decision in {"allow", "allow_in_sandbox"}:
            if self.confirmation_mode in {"oracle_user_eval", "gateway_eval"} and _membership_confirmation_allowed(evidence, action_graph_facts):
                metadata["confirmation_required"] = True
                # Even compatibility/ablation paths obey ToolGate atomicity:
                # this candidate pauses and can only run after a fresh,
                # confirmation-bound adjudication.
                metadata["confirmation_executed"] = False
                if self.confirmation_mode == "gateway_eval":
                    metadata["gateway_confirmation_counted_separately"] = True
                    metadata["gateway_user_confirmation_allowed"] = True
                else:
                    metadata["oracle_user_confirmation_allowed"] = True
                return False, "require_confirmation", metadata
            return True, fusion.decision, metadata
        if fusion.decision == "require_confirmation":
            public_decision = "require_confirmation"
            membership_confirmation = (
                evidence.get("agentdojo.tool_group") == "membership_mutation"
                and bool(evidence.get("agentdojo.task_authorized"))
                and bool(evidence.get("agentdojo.args_match_user_entity"))
                and not bool(action_graph_facts.get("graph.has_private_to_external_edge"))
                and not bool(action_graph_facts.get("graph.has_private_to_executed_external_edge"))
            )
            external_send_confirmation = (
                evidence.get("agentdojo.tool_group") == "external_send"
                and bool(evidence.get("agentdojo.task_authorized"))
                and bool(evidence.get("agentdojo.args_match_user_entity"))
                and not bool(evidence.get("agentdojo.args_match_untrusted_entity"))
                and not bool(evidence.get("agentdojo.attack_goal_overlap"))
                and not bool(action_graph_facts.get("graph.has_private_to_executed_external_edge"))
                and not bool(action_graph_facts.get("graph.has_untrusted_to_executed_side_effect_edge"))
            )
            allow_user_confirmation = membership_confirmation or external_send_confirmation or (
                bool(evidence.get("agentdojo.task_authorized"))
                and not bool(evidence.get("agentdojo.args_match_untrusted_entity"))
                and not bool(evidence.get("agentdojo.attack_goal_overlap"))
                and not bool(action_graph_facts.get("graph.has_private_to_external_edge"))
                and not bool(action_graph_facts.get("graph.has_private_to_executed_external_edge"))
                and not bool(action_graph_facts.get("graph.has_untrusted_to_executed_side_effect_edge"))
            )
            if self.confirmation_mode == "oracle_user_eval":
                metadata["oracle_user_confirmation_allowed"] = allow_user_confirmation
                return False, public_decision, metadata
            metadata["gateway_confirmation_counted_separately"] = self.confirmation_mode == "gateway_eval"
            if self.confirmation_mode == "gateway_eval":
                metadata["gateway_user_confirmation_allowed"] = allow_user_confirmation
                return False, public_decision, metadata
            return False, public_decision, metadata
        return False, fusion.decision, metadata

    def observe_after_tool(self, context: ToolCallContext, raw_result: Any) -> Any:
        started = time.perf_counter()
        spec = self.taxonomy.classify(context.tool_name, suite=context.suite)
        self.state.observe_tool_call(context.tool_name, spec, context.tool_args, event_status="executed", decision="allow", execute=True)
        self.state.observe_tool_result(context.tool_name, spec, raw_result)
        sanitized = self.state.sanitize_tool_output(raw_result, mode=self.sanitize_mode) if self.sanitize_outputs else raw_result
        self.audit_events.append(
            {
                "event_type": "agentdojo_tool_result_observed",
                "result_digest": _digest_value(raw_result),
                "sanitized": sanitized != raw_result,
                "observe_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        return sanitized

    def dispatch_through_theory_gate(
        self,
        decision: ToolExecutionDecision,
        dispatcher: Callable[[ActionCandidate, Realization, str | None], Any],
        *,
        idempotency_key: str | None = None,
        realization_validator: Callable[[Realization], None] | None = None,
    ) -> GateResult:
        """Re-adjudicate and cross the real tool boundary only through ToolGate."""

        if (
            self.theory_runtime is None
            or self.theory_gate is None
            or decision.theory_action is None
            or decision.theory_evidence is None
            or decision.theory_decision is None
        ):
            return GateResult(
                decision="block",
                state="blocked",
                dispatched=False,
                reason="theory gate bindings are unavailable",
            )
        fresh = self.theory_runtime.decide(
            decision.theory_action,
            decision.theory_evidence,
        )
        if fresh.certificate.digest != decision.theory_decision.certificate.digest:
            return GateResult(
                decision="block",
                state="blocked",
                dispatched=False,
                reason="pre-send re-adjudication changed the action certificate",
            )
        if fresh.selected is None:
            result = GateResult(
                decision=fresh.public_decision,
                state="blocked",
                dispatched=False,
                reason="no policy-declared realization is available",
            )
        elif fresh.certificate.outcome in {"allow", "execute_with_constraints"}:
            try:
                (realization_validator or self._validate_builtin_host_realization)(fresh.selected)
            except (RuntimeError, ValueError) as exc:
                result = GateResult(
                    decision="block_with_compliance_error",
                    state="blocked",
                    dispatched=False,
                    reason=f"realization backend unavailable: {exc}",
                )
            else:
                result = self.theory_gate.process(
                    action=decision.theory_action,
                    certificate=fresh.certificate,
                    policy=self.theory_runtime.acceptance,
                    realization=fresh.selected,
                    fact_digest=fresh.certificate.fact_digest,
                    evidence_digest=fresh.certificate.evidence_digest,
                    plan_digest=fresh.selected.plan.digest,
                    # The dispatcher must consume these gate-validated values.  In
                    # particular it may not close over the model's mutable raw
                    # argument object, which would break the certificate binding.
                    dispatcher=dispatcher,
                    idempotency_key=idempotency_key,
                )
        else:
            result = self.theory_gate.process(
                action=decision.theory_action,
                certificate=fresh.certificate,
                policy=self.theory_runtime.acceptance,
                realization=fresh.selected,
                fact_digest=fresh.certificate.fact_digest,
                evidence_digest=fresh.certificate.evidence_digest,
                plan_digest=fresh.selected.plan.digest,
                dispatcher=dispatcher,
                idempotency_key=idempotency_key,
            )
        decision.decision_metadata["tool_gate_state"] = result.state
        decision.decision_metadata["tool_gate_dispatched"] = result.dispatched
        if result.nonce is not None:
            decision.decision_metadata["confirmation_nonce"] = result.nonce
            if isinstance(decision.safe_result, dict):
                decision.safe_result["confirmation_nonce"] = result.nonce
        self.audit_events.append(
            {
                "event_type": "obligate_theory_tool_gate",
                "decision": result.decision,
                "execute": result.dispatched,
                "gate_state": result.state,
                "gate_dispatched": result.dispatched,
                "reason_digest": _digest_value(result.reason) if result.reason else "",
                "action_digest": fresh.certificate.action_digest,
                "evidence_digest": fresh.certificate.evidence_digest,
                "plan_digest": fresh.certificate.plan_digest,
                "realization_digest": fresh.certificate.realization_digest,
                "policy_acceptance_digest": fresh.certificate.policy_acceptance_digest,
                "certificate_digest": fresh.certificate.digest,
                "result_digest": result.outbox.result_digest if result.outbox else "",
                "outbox_status": result.outbox.status if result.outbox else "",
            }
        )
        return result

    def run_guarded_tool(self, context: ToolCallContext, original_tool: Callable[..., Any]) -> tuple[Any, ToolExecutionDecision]:
        decision = self.guard_before_tool(context)

        def dispatch(action: ActionCandidate, realization: Realization, _key: str | None) -> Any:
            if action.tool_name != context.tool_name:
                raise ValueError("gate-validated tool name differs from the guarded boundary")
            self._validate_builtin_host_realization(realization)
            return original_tool(**deepcopy(dict(action.arguments)))

        gate_result = self.dispatch_through_theory_gate(
            decision,
            dispatch,
        )
        if gate_result.state != "done":
            if gate_result.state == "uncertain":
                return {
                    "error": "delivery_outcome_uncertain",
                    "reason": gate_result.reason,
                    "automatic_retry": False,
                }, decision
            return decision.safe_result, decision
        return self.observe_after_tool(context, gate_result.result), decision

    @staticmethod
    def _validate_builtin_host_realization(realization: Realization) -> None:
        """Validate controls implemented by the built-in benchmark boundary."""

        plan = realization.plan
        if plan.block_like:
            return
        unsupported: list[str] = []
        if plan.execution_env != "host":
            unsupported.append(f"execution_env={plan.execution_env}")
        if plan.network_scope != "allow":
            unsupported.append(f"network_scope={plan.network_scope}")
        if plan.data_scope != "raw":
            unsupported.append(f"data_scope={plan.data_scope}")
        if unsupported:
            raise RuntimeError(
                "built-in dispatcher cannot realize " + ", ".join(unsupported)
            )


def summarize_obligate_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    decision_events = [event for event in events if event.get("event_type") == "agentdojo_tool_gate_decision"]
    gate_events = [event for event in events if event.get("event_type") == "obligate_theory_tool_gate"]
    result_events = [event for event in events if event.get("event_type") == "agentdojo_tool_result_observed"]
    policy_latencies = [float(event.get("policy_ms", 0.0)) for event in decision_events if isinstance(event.get("policy_ms"), (int, float))]
    gates_by_certificate: dict[str, list[dict[str, Any]]] = {}
    for gate_event in gate_events:
        gates_by_certificate.setdefault(str(gate_event.get("certificate_digest") or ""), []).append(gate_event)
    registered = 0
    unknown = 0
    blocked = 0
    proposed_blocked = 0
    unmatched_proposed_allow = 0
    actual_done = 0
    actual_dispatched = 0
    actual_blocked = 0
    delivery_uncertain = 0
    confirmation_pending = 0
    matched_gate_events = 0
    repeated_blocked = 0
    theory_decisions = 0
    silent_dispatch = 0
    ordinary_allow_invalid_cert = 0
    overflow_decisions = 0
    compliance_errors = 0
    non_blocking_selected = 0
    confirmations = 0
    rule_hits: dict[str, int] = {}
    for event in decision_events:
        if bool(event.get("tool_registered", False)):
            registered += 1
        else:
            unknown += 1
        proposed_is_blocked = not bool(event.get("execute", True))
        if proposed_is_blocked:
            proposed_blocked += 1
        if event.get("repeated_unsafe_action"):
            repeated_blocked += 1
        public = str(event.get("theory_public_decision") or "")
        certificate_digest = str(event.get("certificate_digest") or "")
        matching = gates_by_certificate.get(certificate_digest, [])
        gate_event = matching.pop(0) if matching else None
        if gate_event is None:
            if proposed_is_blocked:
                blocked += 1
            else:
                unmatched_proposed_allow += 1
        else:
            matched_gate_events += 1
            gate_state = str(gate_event.get("gate_state") or "")
            gate_dispatched = bool(gate_event.get("gate_dispatched", False))
            actual_dispatched += int(gate_dispatched)
            if gate_state == "done":
                actual_done += 1
            elif gate_state == "uncertain":
                delivery_uncertain += 1
            else:
                blocked += 1
                actual_blocked += 1
                if gate_state == "pending":
                    confirmation_pending += 1
        if public:
            theory_decisions += 1
            if gate_event is not None and bool(gate_event.get("gate_dispatched")) and (
                not certificate_digest or public not in {"allow", "execute_with_constraints"}
            ):
                silent_dispatch += 1
            if public == "allow" and (
                not bool(event.get("valid_certificate_evidence"))
                or int(event.get("gap_count") or 0) > 0
                or int(event.get("hazard_count") or 0) > 0
                or int(event.get("overflow_count") or 0) > 0
            ):
                ordinary_allow_invalid_cert += 1
            if int(event.get("overflow_count") or 0) > 0:
                overflow_decisions += 1
            if public == "block_with_compliance_error" or (
                gate_event is not None
                and gate_event.get("decision") == "block_with_compliance_error"
            ):
                compliance_errors += 1
            if public in {"allow", "execute_with_constraints"}:
                non_blocking_selected += 1
            if public == "require_confirmation":
                confirmations += 1
        for hit in event.get("matched_rules") or []:
            rule_id = str(hit)
            if rule_id:
                rule_hits[rule_id] = rule_hits.get(rule_id, 0) + 1
    unmatched_gate_events = [
        event
        for values in gates_by_certificate.values()
        for event in values
    ]
    silent_dispatch += sum(
        1 for event in unmatched_gate_events if bool(event.get("gate_dispatched"))
    )
    total = len(decision_events)
    return {
        "registered_tool_rate": 0.0 if total == 0 else registered / total,
        "unknown_tool_rate": 0.0 if total == 0 else unknown / total,
        "total_tool_calls_gated": total,
        "tool_gate_decision_count": total,
        "blocked_tool_calls": blocked,
        "proposed_blocked_tool_calls": proposed_blocked,
        "matched_tool_gate_event_count": matched_gate_events,
        "unmatched_tool_gate_event_count": len(unmatched_gate_events),
        "actual_gate_done_count": actual_done,
        "actual_gate_dispatched_count": actual_dispatched,
        "actual_gate_blocked_or_paused_count": actual_blocked,
        "delivery_uncertain_count": delivery_uncertain,
        "confirmation_pending_gate_count": confirmation_pending,
        "repeated_block_count": repeated_blocked,
        "theory_decision_count": theory_decisions,
        "silent_dispatch_count": silent_dispatch,
        "ordinary_allow_invalid_certificate_count": ordinary_allow_invalid_cert,
        "overflow_decision_count": overflow_decisions,
        "overflow_rate": 0.0 if theory_decisions == 0 else overflow_decisions / theory_decisions,
        "compliance_error_count": compliance_errors,
        "non_blocking_selected_count": non_blocking_selected,
        "non_blocking_selection_rate": 0.0 if theory_decisions == 0 else non_blocking_selected / theory_decisions,
        "confirmation_count": confirmations,
        "confirmation_burden_rate": 0.0 if theory_decisions == 0 else confirmations / theory_decisions,
        "safe_blocked_result": blocked,
        "allow": actual_done + unmatched_proposed_allow,
        "block": blocked,
        "policy_p50_ms": _percentile(policy_latencies, 0.5),
        "policy_p95_ms": _percentile(policy_latencies, 0.95),
        "rule_hit_counts": dict(sorted(rule_hits.items())),
        "result_events": len(result_events),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return float(ordered[idx])


def _membership_confirmation_allowed(evidence: dict[str, Any], action_graph_facts: dict[str, Any]) -> bool:
    return (
        evidence.get("agentdojo.tool_group") == "membership_mutation"
        and bool(evidence.get("agentdojo.task_authorized"))
        and (bool(evidence.get("agentdojo.args_match_user_entity")) or not bool(evidence.get("agentdojo.attack_goal_overlap")))
        and not bool(action_graph_facts.get("graph.has_private_to_external_edge"))
        and not bool(action_graph_facts.get("graph.has_private_to_executed_external_edge"))
        and not bool(action_graph_facts.get("graph.has_untrusted_to_executed_side_effect_edge"))
    )


@lru_cache(maxsize=1)
def _verified_default_acceptance():
    from obligate.theory.checker import check_default_policy

    report = check_default_policy()
    if not report.accepted:  # fail closed; never trust a declared accepted flag
        failures = [item.vc_id for item in report.failures]
        raise RuntimeError(f"theory policy failed finite checking: {failures}")
    return report.acceptance


def _build_default_theory_runtime(capabilities: Iterable[str]) -> TheoryRuntime:
    return _cached_default_theory_runtime(tuple(sorted(set(capabilities))))


@lru_cache(maxsize=8)
def _cached_default_theory_runtime(capabilities: tuple[str, ...]) -> TheoryRuntime:
    from obligate.theory.policy import build_default_policy

    return TheoryRuntime(
        acceptance=_verified_default_acceptance(),
        policy=build_default_policy(),
        capabilities=capabilities,
    )


