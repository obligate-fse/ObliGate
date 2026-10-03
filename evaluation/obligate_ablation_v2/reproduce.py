"""Staged reproduction driver for paper RQ3 / AgentDojo Table 5 ablation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import atomic_write_json
from .freeze import bundle_fingerprint
from .pipeline import run as run_formal
from .preflight import audit_dry, run_synthetic
from .prepare import EXPECTED_SOURCE_SHA256, MODELS, SOURCE_PLAN, prepare
from .run_case import E2E_VARIANTS
from .scalar_crossfit import crossfit

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(
    os.environ.get("OBLIGATE_AGENTDOJO_PYTHON")
    or os.environ.get("OBLIGATE_PYTHON")
    or sys.executable
).resolve()


def _require_python() -> None:
    if not PYTHON.is_file():
        raise RuntimeError(
            "AgentDojo Python executable does not exist; set "
            "OBLIGATE_AGENTDOJO_PYTHON"
        )


def _run_batch(
    output_root: Path,
    *,
    model: str,
    variant: str,
    thresholds: Path | None,
    resume: bool,
    workers: int,
) -> None:
    model_configs = json.loads(
        (output_root / "config" / "model_configs.json").read_text(encoding="utf-8")
    )
    route = model_configs[model]
    frozen_bundle = bundle_fingerprint(output_root)
    dry_root = output_root / "preflight" / "dry_run"
    log_dir = dry_root / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "-m",
        "evaluation.obligate_ablation_v2.run_batch",
        "--case-manifest",
        str((output_root / "manifests" / "agentdojo_949.json").resolve()),
        "--case-ids-file",
        str((output_root / "preflight" / "dry_run_case_ids.txt").resolve()),
        "--output-root",
        str(dry_root.resolve()),
        "--model",
        model,
        "--variant",
        variant,
        "--base-url",
        str(route["base_url"]),
        "--api-key-env",
        str(route["api_key_env"]),
        "--workers",
        str(workers),
        "--case-attempts",
        "4",
        "--retry-initial-backoff",
        "2",
        "--seed",
        "20260716",
        "--bundle-sha256",
        str(frozen_bundle["aggregate_sha256"]),
        "--credential-sha256",
        str(route["credential_sha256"]),
    ]
    if thresholds is not None:
        command.extend(["--scalar-thresholds", str(thresholds.resolve())])
    if resume:
        command.append("--resume")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    (log_dir / f"{model}.{variant}.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (log_dir / f"{model}.{variant}.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ablation dry preflight failed for {model}/{variant}; "
            f"see {log_dir}"
        )


def run_paid_preflight(
    output_root: Path,
    *,
    resume: bool,
    workers: int,
) -> dict[str, Any]:
    _require_python()
    synthetic = run_synthetic(output_root)
    if not synthetic.get("pass"):
        raise RuntimeError("ablation synthetic preflight failed")

    for model in MODELS:
        _run_batch(
            output_root,
            model=model,
            variant="full",
            thresholds=None,
            resume=resume,
            workers=workers,
        )

    dry_root = output_root / "preflight" / "dry_run"
    thresholds_path = dry_root / "config" / "scalar_crossfit_thresholds.json"
    thresholds = crossfit(
        {
            model: dry_root / "e2e" / model / "full" / "snapshots"
            for model in MODELS
        },
        seed=20260716,
        case_to_fold=json.loads(
            (output_root / "manifests" / "fold_assignment.json").read_text(
                encoding="utf-8"
            )
        )["case_to_fold"],
    )
    atomic_write_json(thresholds_path, thresholds)

    for variant in E2E_VARIANTS:
        if variant == "full":
            continue
        for model in MODELS:
            _run_batch(
                output_root,
                model=model,
                variant=variant,
                thresholds=thresholds_path if variant == "scalar-average" else None,
                resume=resume,
                workers=workers,
            )

    result = audit_dry(output_root)
    if not result.get("formal_run_allowed"):
        raise RuntimeError(
            "ablation paid preflight completed but the formal validity gate did not pass"
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "preflight", "formal", "all"),
        help="Run one stage or the complete staged workflow",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, default=SOURCE_PLAN)
    parser.add_argument(
        "--source-plan-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if args.stage in {"prepare", "all"}:
        prepare(
            output_root,
            source_plan=args.source_plan,
            expected_source_sha256=args.source_plan_sha256,
        )
    if args.stage in {"preflight", "all"}:
        run_paid_preflight(
            output_root,
            resume=args.resume,
            workers=max(1, args.workers),
        )
    if args.stage in {"formal", "all"}:
        return int(run_formal(output_root, resume=args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_paid_preflight"]
