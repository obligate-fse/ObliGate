"""Materialize frozen Ablation v2 manifests and configuration without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .common import atomic_write_json, sha256_file, utc_now
from .scalar_crossfit import stable_fold

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PLAN = (
    ROOT
    / "data"
    / "agentdojo_v1.2.2_949_case_plan.json"
)
SOURCE_PLAN = Path(
    os.environ.get("OBLIGATE_AGENTDOJO_CASE_PLAN") or DEFAULT_SOURCE_PLAN
).resolve()
EXPECTED_SOURCE_SHA256 = os.environ.get(
    "OBLIGATE_AGENTDOJO_CASE_PLAN_SHA256",
    "4637e3a436ee5b8be6550871954cab654ca93ecf3a1956a85d3a0a28bb588eb7",
)
MODELS = {
    "deepseek-v4-flash": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen-plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _dry_cases(cases: list[dict[str, Any]], seed: int) -> list[str]:
    quotas = {"banking": 2, "slack": 2, "travel": 2, "workspace": 4}
    selected: list[str] = []
    for suite, quota in quotas.items():
        population = [item for item in cases if str(item["suite"]) == suite]
        population.sort(
            key=lambda item: hashlib.sha256(
                f"dry\0{seed}\0{item['case_id']}".encode("utf-8")
            ).hexdigest()
        )
        selected.extend(str(item["case_id"]) for item in population[:quota])
    return selected


def prepare(
    output_root: Path,
    *,
    seed: int = 20260716,
    source_plan: Path = SOURCE_PLAN,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
) -> dict[str, Any]:
    source_plan = source_plan.resolve()
    if not source_plan.is_file():
        raise RuntimeError(
            "missing AgentDojo 949-case plan; pass --source-plan or set "
            "OBLIGATE_AGENTDOJO_CASE_PLAN"
        )
    source_sha = sha256_file(source_plan)
    if source_sha != expected_source_sha256:
        raise RuntimeError(f"frozen AgentDojo source plan drift: {source_sha}")
    source = json.loads(source_plan.read_text(encoding="utf-8"))
    cases = [dict(item) for item in source["cases"]]
    ids = [str(item["case_id"]) for item in cases]
    if len(cases) != 949 or len(set(ids)) != 949:
        raise RuntimeError("AgentDojo manifest must contain exactly 949 unique task pairs")

    manifest = {
        **source,
        "schema_version": "obligate-agentdojo-949-manifest-v1",
        "experiment_name": "ObliGate AgentDojo component ablation v2",
        "source_plan": str(source_plan),
        "source_plan_sha256": source_sha,
        "selected_at": utc_now(),
        "selection_frozen": True,
    }
    manifest_path = output_root / "manifests" / "agentdojo_949.json"
    atomic_write_json(manifest_path, manifest)
    assignment = {case_id: stable_fold(case_id, seed=seed) for case_id in sorted(ids)}
    atomic_write_json(
        output_root / "manifests" / "fold_assignment.json",
        {
            "schema_version": "obligate-five-fold-assignment-v1",
            "seed": seed,
            "fold_count": 5,
            "case_to_fold": assignment,
            "fold_counts": {str(fold): sum(value == fold for value in assignment.values()) for fold in range(5)},
            "uses_outcome_labels": False,
        },
    )
    audit = {
        "schema_version": "obligate-agentdojo-sampling-audit-v1",
        "chosen_manifest": str(manifest_path.resolve()),
        "chosen_manifest_sha256": sha256_file(manifest_path),
        "source_plan": str(source_plan),
        "source_plan_sha256_recomputed": source_sha,
        "source_plan_verification": "sha256_exact",
        "case_count": len(cases),
        "unique_case_count": len(set(ids)),
        "by_suite": dict(sorted(Counter(str(item["suite"]) for item in cases).items())),
        "selection_reason": "complete public Cartesian population for the four AgentDojo v1.2.2 suites",
        "benchmark_package": "agentdojo==0.1.35",
        "benchmark_version": "v1.2.2",
        "attack": "important_instructions",
        "forbidden_benchmarks": ["Agent-SafetyBench", "Agent Security Bench"],
    }
    atomic_write_json(output_root / "manifests" / "sampling_audit.json", audit)

    env = dict(os.environ)
    model_configs: dict[str, Any] = {}
    for model, route in MODELS.items():
        secret = env.get(route["api_key_env"], "")
        if not secret:
            raise RuntimeError(f"missing credential for {model}: {route['api_key_env']}")
        model_configs[model] = {
            **route,
            "credential_sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            "credential_length": len(secret),
            "temperature": 0.0,
            "reasoning_effort": None,
            "max_iters": 24,
            "timeout_seconds": 300,
            "task_retry_max_attempts": 4,
            "task_retry_initial_backoff_seconds": 2,
            "runner_internal_retry_max_attempts": 1,
            "fresh_process_case_attempts": 4,
            "orchestration_seed": seed,
            "provider_request_seed": None,
            "workers": 8,
        }
    atomic_write_json(output_root / "config" / "model_configs.json", model_configs)

    experiment_yaml = f"""schema_version: obligate-ablation-v2-experiment-v1
method_name: ObliGate
benchmark: AgentDojo
agentdojo_package: 0.1.35
benchmark_version: v1.2.2
case_count: 949
case_manifest: {manifest_path.resolve()}
case_manifest_sha256: {sha256_file(manifest_path)}
source_plan_sha256: {source_sha}
models:
  - deepseek-v4-flash
  - qwen-plus
attack: important_instructions
defense: obligate
mode: fair
confirmation_mode: strict_eval
temperature: 0.0
reasoning_effort: null
max_iters: 24
runner_internal_retry_max_attempts: 1
fresh_process_case_attempts: 4
retry_initial_backoff_seconds: 2
seed: {seed}
provider_request_seed: null
formal_episode_count: 15184
bootstrap_iterations: 10000
bootstrap_seed: {seed}
old_results_read_only: true
research_variants_deployable: false
"""
    _atomic_text(output_root / "config" / "experiment.yaml", experiment_yaml)
    variants_yaml = (ROOT / "configs" / "ablation_variants.yaml").read_text(encoding="utf-8")
    variants_yaml += """snapshot_only:
  - conflict-to-support
  - conflict-to-refute
fault_injection_only:
  - no-feasibility-filter
  - no-rebind
"""
    _atomic_text(output_root / "config" / "variants.yaml", variants_yaml)

    dry = _dry_cases(cases, seed)
    _atomic_text(output_root / "preflight" / "dry_run_case_ids.txt", "\n".join(dry) + "\n")
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "case_count": len(cases),
        "dry_case_count": len(dry),
        "models": model_configs,
    }
    atomic_write_json(output_root / "pipeline_status.json", {"stage": "prepared", **result, "updated_at": utc_now()})
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--source-plan",
        type=Path,
        default=SOURCE_PLAN,
        help="Frozen 949-case AgentDojo plan (the public plan is bundled under data/)",
    )
    parser.add_argument(
        "--source-plan-sha256",
        default=EXPECTED_SOURCE_SHA256,
        help="Expected SHA256 for --source-plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value = prepare(
        args.output_root,
        seed=args.seed,
        source_plan=args.source_plan,
        expected_source_sha256=args.source_plan_sha256,
    )
    print(json.dumps({key: value[key] for key in ("manifest", "manifest_sha256", "case_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_SOURCE_PLAN", "EXPECTED_SOURCE_SHA256", "MODELS", "SOURCE_PLAN", "prepare"]
