"""Run one frozen AgentDojo task pair for ObliGate Ablation v2.

Every invocation owns one fresh AgentDojo simulator and one research bridge.
No benchmark outcome is exposed to the online policy or model cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from obligate.eval.agentdojo.gate import tool_firewall as firewall_module
from obligate.eval.agentdojo.runner import run_tool_firewall_eval as runner
from experiments.adaptive_ablation.safety import force_simulator_env
from experiments.adaptive_ablation.snapshots import SnapshotRecorder, snapshot_case

from .common import atomic_write_json, sha256_file, utc_now
from .model_cache import instrument_openai
from .research_bridge import install_research_bridge, uninstall_research_bridge
from .protocol import E2E_VARIANTS, POLICY_CONTROLS, COUNTERFACTUAL_VARIANTS, execution_contract


def _thresholds_for_case(path: Path | None, case_id: str) -> tuple[float, float, float] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    assignments = value.get("case_to_fold") or value.get("fold_assignment") or {}
    fold = str(assignments.get(case_id))
    folds = value.get("folds") or {}
    row = folds.get(fold) or folds.get(f"fold_{fold}")
    if not isinstance(row, dict):
        raise ValueError(f"scalar threshold file has no held-out threshold for {case_id!r}")
    thresholds = row.get("thresholds") or row
    triple = (
        float(thresholds["t_allow"]),
        float(thresholds["t_confirm"]),
        float(thresholds["t_block"]),
    )
    if not 0.0 <= triple[0] < triple[1] < triple[2] <= 1.0:
        raise ValueError(f"invalid scalar thresholds for {case_id}: {triple}")
    return triple


def run(args: argparse.Namespace) -> dict[str, Any]:
    force_simulator_env(os.environ)
    if re.fullmatch(r"[0-9a-f]{64}", str(args.bundle_sha256)) is None:
        raise ValueError("--bundle-sha256 must be a lowercase SHA256 digest")
    if args.variant not in E2E_VARIANTS:
        raise ValueError(f"unsupported E2E variant: {args.variant}")

    original_loader = runner._load_agentdojo_deps
    original_pipeline = runner.build_pipeline

    def patched_loader() -> dict[str, Any]:
        deps = dict(original_loader())
        deps["OpenAI"] = instrument_openai(
            deps["OpenAI"],
            cache_path=args.model_cache,
            audit_path=args.model_cache_audit,
            model_label=args.model,
            variant=args.variant,
            case_id=args.case_id,
            temperature=0.0,
        )
        return deps

    runner._load_agentdojo_deps = patched_loader  # type: ignore[assignment]
    bridge_originals: Any = None
    thresholds = _thresholds_for_case(args.scalar_thresholds, args.case_id)
    if args.variant == "scalar-average" and thresholds is None:
        raise ValueError("scalar-average requires held-out fold thresholds")
    if args.variant in COUNTERFACTUAL_VARIANTS:
        bridge_originals = install_research_bridge(
            firewall_module,
            args.variant,
            case_id=args.case_id,
            decision_trace_path=args.decision_trace,
            scalar_thresholds=thresholds,
        )
    if args.variant in POLICY_CONTROLS:
        from .policy_controls import PolicyControlFirewall
        from obligate.eval.agentdojo.gate.runtime_wrapper import build_obligate_pipeline
        control = PolicyControlFirewall(args.variant, case_id=args.case_id, trace_path=args.decision_trace)
        def control_pipeline(defense, llm, control_plane, *, system_message, max_iters):
            return build_obligate_pipeline(llm, firewall=control, system_message=system_message,
                                          max_iters=max_iters, sanitize_mode=control.sanitize_mode)
        runner.build_pipeline = control_pipeline

    recorder: Any
    if args.variant == "full":
        recorder = SnapshotRecorder(args.snapshots, benchmark="agentdojo", variant="full")
    else:
        recorder = _NullContext()

    started_at = utc_now()
    try:
        with recorder:
            with snapshot_case(args.case_id):
                summary = runner.run_suite(
                    args.suite,
                    args.model,
                    "obligate",
                    benchmark_version="v1.2.2",
                    attack="important_instructions",
                    report_dir=args.output.parent,
                    run_name=args.run_name,
                    user_task_ids=[args.user_task_id],
                    injection_task_ids=[args.injection_task_id],
                    confirmation_mode="strict_eval",
                    save_full_trace=True,
                    trace_dir=args.trace_dir,
                    max_iters=24,
                    mode="fair",
                    **({"system_message": "You are a helpful assistant." + _policy_prompt()}
                       if args.variant == "policy-prompt-only" else {}),
                )
    finally:
        runner._load_agentdojo_deps = original_loader  # type: ignore[assignment]
        runner.build_pipeline = original_pipeline
        if bridge_originals is not None:
            uninstall_research_bridge(firewall_module, bridge_originals)

    normalized = summary.get("normalized_cases")
    if not isinstance(normalized, list) or len(normalized) != 1:
        raise ValueError(f"expected exactly one normalized AgentDojo case, got {type(normalized).__name__}")
    summary["obligate_ablation_v2"] = {
        "schema_version": "obligate-agentdojo-ablation-v2-case-v1",
        "case_id": args.case_id,
        "model": args.model,
        "variant": args.variant,
        **execution_contract(args.variant),
        "simulator_only": True,
        "temperature": 0.0,
        "reasoning_effort": None,
        "max_iters": 24,
        "orchestration_seed": args.seed,
        "provider_request_seed": None,
        "bundle_aggregate_sha256": args.bundle_sha256,
        "fresh_process_attempt": args.attempt_id,
        "case_manifest_sha256": sha256_file(args.case_manifest),
        "scalar_thresholds_sha256": (
            sha256_file(args.scalar_thresholds) if args.scalar_thresholds is not None else None
        ),
        "started_at": started_at,
        "completed_at": utc_now(),
    }
    atomic_write_json(args.output, summary)
    return summary


def _policy_prompt() -> str:
    from .policy_controls import policy_prompt
    return policy_prompt()


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--user-task-id", required=True)
    parser.add_argument("--injection-task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=E2E_VARIANTS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--decision-trace", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--model-cache-audit", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--scalar-thresholds", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--attempt-id", type=int, default=1)
    parser.add_argument("--bundle-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["E2E_VARIANTS", "run"]
