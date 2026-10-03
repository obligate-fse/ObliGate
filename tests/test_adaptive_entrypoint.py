"""Provider-free regression checks for adaptive phases without stress inputs.

Run in the documented AgentDojo/control environment with the project root and
src directory on PYTHONPATH. Public manifest preparation and execution are
isolated so the tests need no upstream data or provider credentials.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.adaptive_closed_loop_v2 import pipeline


class AdaptiveEntrypointTests(unittest.TestCase):
    def _run_without_stress(self, phase: str, expected_rounds: int) -> None:
        config = deepcopy(pipeline.load_config())
        cases = [
            {"benchmark": "agentdojo", "case_id": "fixture-agentdojo"},
            {"benchmark": "agent_safetybench", "case_id": "fixture-safetybench"},
            {"benchmark": "agent_security_bench", "case_id": "fixture-asb"},
        ]
        # build_manifests intentionally omits the stress key in these phases.
        manifests = {"main": {"case_count": len(cases), "cases": cases}, "audit": {}}
        args = pipeline.parse_args(
            ["--phase", phase, "--models", "qwen-plus", "--max-rounds", "5"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(pipeline, "RESULT_ROOT", Path(temporary)),
                patch.object(pipeline, "load_config", return_value=config),
                patch.object(pipeline, "_preflight", return_value={"validated": True}),
                patch.object(pipeline, "build_manifests", return_value=manifests) as build,
                patch.object(pipeline, "_run_segment") as segment,
                patch.object(pipeline, "_write_status") as status,
                patch.object(
                    pipeline, "generate_many", side_effect=AssertionError("provider call")
                ) as generate,
                patch.object(
                    pipeline, "_run_command", side_effect=AssertionError("external run")
                ) as external,
            ):
                pipeline.run(args)

        build.assert_called_once_with(
            seed=int(config.get("sampling_seed", 20260716)), include_stress=False
        )
        segment.assert_called_once()
        call = segment.call_args
        self.assertEqual(call.args[0], phase)
        self.assertEqual(call.args[1], cases)
        self.assertEqual(call.kwargs["max_rounds"], expected_rounds)
        self.assertEqual(call.kwargs["models"], ["qwen-plus"])
        self.assertEqual(call.kwargs["seeds"], [config["main_attacker_seeds"][0]])
        status.assert_any_call(
            "preflight",
            "completed",
            preflight={"validated": True},
            main_cases=3,
            stress_cases=0,
        )
        status.assert_any_call("formal_complete", "completed", phase=phase)
        generate.assert_not_called()
        external.assert_not_called()

    def test_main_phase_runs_without_a_stress_manifest(self) -> None:
        self._run_without_stress("main", expected_rounds=5)

    def test_dry_phase_runs_without_a_stress_manifest_and_caps_rounds(self) -> None:
        self._run_without_stress("dry", expected_rounds=2)


if __name__ == "__main__":
    unittest.main()
