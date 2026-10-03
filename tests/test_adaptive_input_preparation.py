"""Provider-free coverage for run normalization and fresh stress prerequisites."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_adaptive_inputs import normalize_runs, prepare_full_events, validate_full_reference
from evaluation.adaptive_closed_loop_v2 import manifests


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class AdaptiveInputPreparationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        model = "qwen-plus"
        runs = {name: root / name for name in ("agentdojo", "agent_safetybench", "asb_iclr2025")}
        for benchmark, directory in runs.items():
            commands = {
                "agentdojo": ["python", "runner", "--defense", "obligate"],
                "agent_safetybench": ["python", "runner", "--defense", "obligate_visible_fair"],
                "asb_iclr2025": ["python", "runner", "--modes", "obligate_registry_blind"],
            }
            write(directory / "manifest.json", {
                "run_id": "run-fixture", "benchmark_spec": {"benchmark": benchmark},
                "launch": {"model": model, "command": commands[benchmark], "cwd": str(root)},
            })
            write(directory / "result.json", {"status": "succeeded", "exit_code": 0})
        trace = root / "original_trace.json"
        write(trace, {"fixture": "trace"})
        write(runs["agentdojo"] / "run-fixture.json", {
            "model": model, "defense": "obligate", "attack": "important_instructions",
            "benchmark_version": "v1.2.2", "suite": "banking",
            "per_run": [{"suite": "banking", "user_task_id": "user_task_0",
                         "injection_task_id": "injection_task_0", "trace_file": str(trace),
                         "injections": {"fixture": "public payload"}, "utility": False,
                         "security": True, "confirmation_required_count": 1}],
        })
        write(runs["agent_safetybench"] / "gen_res.json", [{"id": 7, "error": "fixture failure"}])
        asb = {"case_id": "asb-1", "mode": "obligate_registry_blind", "provider_error": "fixture failure"}
        (runs["asb_iclr2025"] / "records.jsonl").write_text(json.dumps(asb) + "\n", encoding="utf-8")
        ad_plan = root / "agentdojo_plan.json"
        write(ad_plan, {"cases": [{"case_id": "full_banking_user_task_0_injection_task_0",
              "suite": "banking", "user_task_id": "user_task_0", "injection_task_id": "injection_task_0"}]})
        asb_plan = root / "asb_plan.json"
        write(asb_plan, {"case_ids": ["asb-1"]})
        return {"model": model, "agentdojo_dirs": [runs["agentdojo"]],
                "safetybench_dir": runs["agent_safetybench"], "asb_dir": runs["asb_iclr2025"],
                "output": root / "new-inputs", "agentdojo_plan": ad_plan, "asb_plan": asb_plan,
                "safetybench_count": 1}

    def test_normalizer_preserves_failures_and_creates_readable_per_case_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            exports = normalize_runs(**args)
            output = args["output"]
            raw_path = next((output / "round1/agentdojo/raw_runs").glob("*.json"))
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["run_name"], "full_banking_user_task_0_injection_task_0_obligate_strict")
            self.assertEqual(len(raw["per_run"]), 1)
            self.assertFalse(raw["per_run"][0]["utility"])
            self.assertTrue(raw["per_run"][0]["security"])
            self.assertEqual(json.loads(Path(raw["per_run"][0]["trace_file"]).read_text()), {"fixture": "trace"})
            self.assertEqual((output / "round1/agent_security_bench/records.jsonl").read_bytes(),
                             (args["asb_dir"] / "records.jsonl").read_bytes())
            self.assertEqual(Path(exports["OBLIGATE_AGENTDOJO_R1_ROOT"]), output / "round1/agentdojo")

    def test_mixed_model_rejected_before_output_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            args["model"] = "deepseek-v4-flash"
            with self.assertRaisesRegex(ValueError, "model differs"):
                normalize_runs(**args)
            self.assertFalse(args["output"].exists())

    def test_duplicate_population_rejected_before_output_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            args["agentdojo_dirs"] *= 2
            with self.assertRaisesRegex(ValueError, "duplicate AgentDojo"):
                normalize_runs(**args)
            self.assertFalse(args["output"].exists())

    def test_missing_confirmation_reference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(manifests, "CONFIRMATION_RAW_DIR", Path(temporary) / "missing"):
                with self.assertRaisesRegex(RuntimeError, "missing Full confirmation raw runs"):
                    manifests._confirmation_pressure_cases()

    def test_full_reference_requires_the_complete_selected_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            (snapshots / "case.jsonl").write_text("{}\n")
            raw_dir = root / "raw_runs"
            write(raw_dir / "case.json", {"obligate_ablation_v2": {
                "variant": "full", "model": "qwen-plus", "case_id": "case"}, "per_run": [{}]})
            with self.assertRaisesRegex(ValueError, "do not cover"):
                validate_full_reference(snapshots, raw_dir, "qwen-plus", {"case", "missing"})
            result = prepare_full_events(snapshots, raw_dir, root, "qwen-plus", {"case"})
            self.assertEqual(Path(result["OBLIGATE_STRESS_SNAPSHOT_STREAM"]).read_text(), "{}\n")
            self.assertEqual(Path(result["OBLIGATE_STRESS_CONFIRMATION_RAW_DIR"]), raw_dir)

    def test_current_full_metadata_yields_correct_confirmation_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "case.json", {"run_name": "case_obligate_ablation_v2_full",
                  "obligate_ablation_v2": {"variant": "full", "case_id": "case"},
                  "per_run": [{"confirmation_required_count": 1, "recovery_success": False}]})
            with patch.object(manifests, "CONFIRMATION_RAW_DIR", root):
                self.assertEqual(manifests._confirmation_pressure_cases(), ["case"])

    def test_stress_detector_builds_a_fresh_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "events.jsonl"
            source.write_text(json.dumps({"event": "decision", "snapshot_id": "fixture"}) + "\n")
            preflight = root / "preflight.json"
            write(preflight, {"real_snapshot_calibration": {"population_sha256": "fixture"}})
            index = root / "new-index.sqlite3"
            with (patch.object(manifests, "SNAPSHOT_STREAM", source),
                  patch.object(manifests, "SNAPSHOT_INDEX", index),
                  patch.object(manifests, "PREFLIGHT", preflight),
                  patch.object(manifests, "_runtimes", return_value=(None, None, None, None))):
                result = manifests._detect_snapshot_mechanisms()
            self.assertTrue(index.is_file())
            self.assertEqual(result["audit"]["input_snapshots_evaluated"], 0)
            self.assertEqual(result["conflict"], [])


if __name__ == "__main__":
    unittest.main()
