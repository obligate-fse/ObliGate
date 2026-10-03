"""Offline regression checks for the two Table 5 policy controls."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.obligate_ablation_v2.flat_control import FlatGate
from evaluation.obligate_ablation_v2.policy_controls import PolicyControlFirewall, policy_prompt
from evaluation.obligate_ablation_v2.protocol import E2E_VARIANTS, execution_contract
from evaluation.obligate_ablation_v2 import run_case
from evaluation.obligate_ablation_v2.variant_runtime import serialize_runtime_evidence
from obligate.eval.agentdojo.gate import tool_firewall as firewall_module
from obligate.eval.agentdojo.compat.types import ToolCallContext
from obligate.theory.eoc import HazardSignal
from obligate.theory.model import ActionCandidate, EvidenceAssertion, SignedAtom, TaskContract
from obligate.theory.policy import build_default_policy
from obligate.theory.realizer import BoundRealizer
from obligate.theory.runtime import RuntimeEvidence, requirements_for_action


class PolicyControlsTests(unittest.TestCase):
    def setUp(self):
        self.action = ActionCandidate("send_email", {"recipient": "alice@example.test"}, side_effect=True, external_sink=True)
        self.supported = RuntimeEvidence(TaskContract(source_digest="offline"), tuple(
            EvidenceAssertion(req, "user-task:" + req.predicate, "test", True)
            for req in requirements_for_action(self.action)))
        self.control = FlatGate(build_default_policy(), BoundRealizer.from_capabilities(("host", "confirmation", "audit")), "single")

    def test_gap_and_untrusted_support_pause_while_trusted_action_executes(self):
        self.assertTrue(self.control.decide(self.action, self.supported)["execute"])
        untrusted = replace(self.supported, assertions=tuple(
            replace(a, trusted_for_authorization=False) for a in self.supported.assertions))
        gap = self.control.decide(self.action, untrusted)
        self.assertFalse(gap["execute"])
        self.assertIn(gap["decision"], {"require_confirmation", "block", "block_with_compliance_error"})
        self.assertIsNone(gap["certificate"])

    def test_explicit_deny_and_hard_hazard_override_trusted_support(self):
        for evidence in (
            replace(self.supported, explicit_denies=(HazardSignal("deny", "explicit-denial", ("trusted:deny",), hard=True),)),
            replace(self.supported, hazards=(HazardSignal("hard", "hard-hazard", ("trusted:hazard",), hard=True),)),
        ):
            result = self.control.decide(self.action, evidence)
            self.assertFalse(result["execute"])
            self.assertIn(result["decision"], {"block", "block_with_compliance_error"})

    def test_catalogue_has_only_existing_single_remedies_and_blocks(self):
        policy = self.control.policy
        atomic = {policy.default_plan.digest} | {r.plan.digest for r in policy.remedies} | {r.plan.digest for r in policy.fallback_remedies}
        self.assertTrue(all(r.plan.block_like or r.plan.digest in atomic for r in self.control.controls))
        # A missing host-only backend cannot silently execute a sandbox remedy.
        constrained = replace(self.supported, hazards=(HazardSignal("sandbox", "sandbox", ("test:sandbox",), preferred_control="allow_in_sandbox"),))
        self.assertFalse(self.control.decide(self.action, constrained)["execute"])

    def test_signed_conflict_is_not_authorization(self):
        conflict = EvidenceAssertion(SignedAtom("action_authorized", (self.action.tool_name,), "-"), "negative", "test", False)
        result = self.control.decide(self.action, replace(self.supported, assertions=(*self.supported.assertions, conflict)))
        self.assertFalse(result["execute"])

    def test_prompt_only_dispatch_has_no_theory_or_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            firewall = PolicyControlFirewall("policy-prompt-only", case_id="offline", trace_path=trace)
            self.assertIsNone(firewall.theory_runtime)
            self.assertIsNone(firewall.theory_gate)
            context = ToolCallContext(suite="workspace", tool_name="send_email", tool_args={"recipient": "unknown@example.test", "subject": "test", "body": "test"}, user_task="Read my mail.")
            decision = firewall.guard_before_tool(context)
            self.assertTrue(decision.execute)
            self.assertIsNone(decision.theory_decision)
            calls = []
            result = firewall.dispatch_through_theory_gate(decision, lambda a, r, key: calls.append(a) or {"sent": True})
            self.assertEqual(result.state, "done")
            self.assertEqual(len(calls), 1)
            row = json.loads(trace.read_text(encoding="utf-8"))
            self.assertIs(row["research_counterfactual_mode"], False)
            self.assertIs(row["eoc_ecp_used"], False)
            self.assertIs(row["witness_used"], False)
            self.assertIsNone(row["action_certificate"])
            self.assertNotIn("research_dispatch_binding", row["decision"])

    def test_dag_single_boundary_blocks_untrusted_target(self):
        with tempfile.TemporaryDirectory() as directory:
            firewall = PolicyControlFirewall("dag-single", case_id="offline", trace_path=Path(directory)/"trace.jsonl")
            context = ToolCallContext(suite="workspace", tool_name="send_email", tool_args={"recipient": "unknown@example.test", "subject": "test", "body": "test"}, user_task="Read my mail.")
            decision = firewall.guard_before_tool(context)
            self.assertFalse(decision.execute)
            called = []
            result = firewall.dispatch_through_theory_gate(decision, lambda *args: called.append(args))
            self.assertFalse(result.dispatched)
            self.assertEqual(called, [])

    def test_controls_never_use_research_certificate_bridge(self):
        for variant in {"dag-single", "policy-prompt-only"}:
            contract = execution_contract(variant)
            self.assertTrue(contract["policy_control"])
            self.assertFalse(contract["research_counterfactual_mode"])
            self.assertFalse(contract["formal_action_certificate"])
        self.assertEqual(len(E2E_VARIANTS), 8)
        self.assertEqual(E2E_VARIANTS[1], "scalar-average")

    def test_prompt_contains_exact_policy_and_predicates(self):
        prompt = policy_prompt()
        self.assertIn("def default_benchmark_policy", prompt)
        self.assertIn("build_default_policy = default_benchmark_policy", prompt)
        self.assertIn("def from_tool_boundary", prompt)
        self.assertIn("No external guard will enforce", prompt)

    def test_shared_extraction_matches_full_and_strips_benchmark_oracles(self):
        context = ToolCallContext(suite="workspace", tool_name="send_email", tool_args={"recipient": "alice@example.test", "subject": "test", "body": "test"},
            user_task="Send alice@example.test an email.", injection_task_id="secret-scorer-id",
            user_task_id="secret-task-id", attack_goal_signatures=["hidden attack goal"], defense_mode="oracle_full")
        full = firewall_module.AgentDojoToolFirewall(enable_theory_core=False)
        full.enable_theory_core = True
        captured = {}
        class StopBeforeDecision(Exception):
            pass
        class CaptureRuntime:
            def decide(self, action, evidence):
                captured.update(action=action, evidence=evidence)
                raise StopBeforeDecision()
        full.theory_runtime = CaptureRuntime()
        with self.assertRaises(StopBeforeDecision):
            full.guard_before_tool(context)
        with tempfile.TemporaryDirectory() as directory:
            control = PolicyControlFirewall("dag-single", case_id="offline", trace_path=Path(directory)/"trace.jsonl")
            decision = control.guard_before_tool(context)
            self.assertEqual(captured["action"].normalized_dict(), decision.theory_action.normalized_dict())
            self.assertEqual(serialize_runtime_evidence(captured["evidence"]), serialize_runtime_evidence(decision.theory_evidence))
            encoded = json.dumps(serialize_runtime_evidence(decision.theory_evidence))
            self.assertNotIn("secret-scorer-id", encoded)
            self.assertNotIn("hidden attack goal", encoded)

    def test_case_runner_routes_controls_and_restores_pipeline(self):
        from obligate.eval.agentdojo.gate import runtime_wrapper
        for variant in ("dag-single", "policy-prompt-only"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest = folder / "manifest.json"
                manifest.write_text('{"cases": []}', encoding="utf-8")
                args = SimpleNamespace(variant=variant, case_id="offline", model="offline-model", scalar_thresholds=None,
                    model_cache=folder/"cache", model_cache_audit=folder/"cache-audit", decision_trace=folder/"decisions.jsonl",
                    snapshots=folder/"snapshots", output=folder/"out.json", trace_dir=folder/"traces", run_name="offline",
                    suite="workspace", user_task_id="u", injection_task_id="i", bundle_sha256="0"*64,
                    seed=20260716, attempt_id=1, case_manifest=manifest)
                captured = {}
                original_pipeline = run_case.runner.build_pipeline
                def fake_pipeline(llm, **kwargs):
                    captured.update(kwargs)
                    return object()
                def fake_suite(*positional, **kwargs):
                    run_case.runner.build_pipeline("obligate", None, None,
                        system_message=kwargs.get("system_message", "You are a helpful assistant."), max_iters=24)
                    return {"normalized_cases": [{"raw_agentdojo_user_task_success": True, "raw_agentdojo_injection_task_success": False}]}
                with patch.object(run_case.runner, "run_suite", fake_suite), patch.object(runtime_wrapper, "build_obligate_pipeline", fake_pipeline), patch.object(run_case, "install_research_bridge", side_effect=AssertionError("research bridge must not run")):
                    summary = run_case.run(args)
                self.assertIs(run_case.runner.build_pipeline, original_pipeline)
                self.assertEqual(captured["firewall"].variant, variant)
                self.assertFalse(summary["obligate_ablation_v2"]["research_counterfactual_mode"])
                if variant == "policy-prompt-only":
                    self.assertIn("Exact bundled policy definitions", captured["system_message"])


if __name__ == "__main__":
    unittest.main()
