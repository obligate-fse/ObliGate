"""Concurrent AgentDojo runner for ObliGate closed-loop rounds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiments.adaptive_ablation.safety import force_simulator_env

from .common import ROOT, provider_settings, read_json, redact_error, safe_slug, sha256_file, write_json


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    plan = read_json(args.case_plan)
    cases = list(plan.get("cases") or [])
    python_bin = _agentdojo_python()
    commands: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        run_name = f"{safe_slug(args.run_label)}_{safe_slug(case_id)}"
        output = args.out_dir / "raw_runs" / f"{run_name}.json"
        commands.append(
            {
                "case_id": case_id,
                "output": str(output),
                "command": [
                    str(python_bin),
                    "-m",
                    "evaluation.adaptive_closed_loop_v2.run_agentdojo_case",
                    "--case-id",
                    case_id,
                    "--suite",
                    str(case["suite"]),
                    "--user-task-id",
                    str(case["user_task_id"]),
                    "--injection-task-id",
                    str(case["injection_task_id"]),
                    "--payload-manifest",
                    str(args.payload_manifest.resolve()),
                    "--model",
                    args.model,
                    "--run-name",
                    run_name,
                    "--output",
                    str(output.resolve()),
                    "--trace-dir",
                    str((args.out_dir / "full_traces").resolve()),
                    "--usage-ledger",
                    str((args.out_dir / "usage" / f"{safe_slug(case_id)}.jsonl").resolve()),
                    "--run-label",
                    args.run_label,
                    "--defense",
                    args.defense,
                ],
            }
        )
    return commands


def run(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    key_env, base_url = provider_settings(args.model)
    python_bin = _agentdojo_python()
    commands = build_commands(args)
    execution = {
        "schema_version": "obligate-agentdojo-batch-v2",
        "case_count": len(commands),
        "model": args.model,
        "temperature": 0.0,
        "workers": args.workers,
        "case_plan": str(args.case_plan.resolve()),
        "case_plan_sha256": sha256_file(args.case_plan),
        "payload_manifest": str(args.payload_manifest.resolve()),
        "payload_manifest_sha256": sha256_file(args.payload_manifest),
        "run_label": args.run_label,
        "defense": args.defense,
        "python_executable": str(python_bin),
        "commands": commands,
        "simulator_only": True,
    }
    write_json(args.out_dir / "execution_plan.json", execution)
    if args.dry_run:
        return 0
    pending = [item for item in commands if not (args.resume and _valid_output(Path(item["output"]), item["case_id"]))]
    failures: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["OBLIGATE_LLM_PROVIDER"] = "openai_compatible"
    env["OBLIGATE_LLM_BASE_URL"] = base_url
    env["OBLIGATE_OPENAI_COMPAT_SYSTEM_ROLE"] = "1"
    env["PYTHONHASHSEED"] = "20260716"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("OBLIGATE_LLM_API_KEY", env[key_env])
    force_simulator_env(env)
    done = len(commands) - len(pending)
    _write_progress(args.out_dir, done, len(commands), failures)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                subprocess.run,
                item["command"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            done += 1
            try:
                completed = future.result()
            except Exception as exc:  # noqa: BLE001 - keep per-case runner errors in denominator.
                failures.append(
                    {
                        "case_id": item["case_id"],
                        "returncode": None,
                        "exception": type(exc).__name__,
                        "stderr_tail": redact_error(exc, limit=4000),
                    }
                )
                _write_progress(args.out_dir, done, len(commands), failures)
                continue
            if completed.returncode != 0 or not _valid_output(Path(item["output"]), item["case_id"]):
                failures.append(
                    {
                        "case_id": item["case_id"],
                        "returncode": completed.returncode,
                        "stdout_tail": completed.stdout[-2000:],
                        "stderr_tail": completed.stderr[-4000:],
                    }
                )
            _write_progress(args.out_dir, done, len(commands), failures)
    return 1 if failures and args.fail_on_case_failure else 0


def _write_progress(out_dir: Path, completed: int, total: int, failures: list[dict[str, Any]]) -> None:
    write_json(out_dir / "progress.json", {"completed": completed, "total": total, "failure_count": len(failures)})
    write_json(out_dir / "failures.json", failures)


def _valid_output(path: Path, case_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    normalized = value.get("normalized_cases")
    return isinstance(normalized, list) and len(normalized) == 1 and case_id in str(value.get("run_name"))


def _agentdojo_python() -> Path:
    configured = os.environ.get("OBLIGATE_AGENTDOJO_PYTHON") or os.environ.get("OBLIGATE_PYTHON") or sys.executable
    python_bin = Path(configured).resolve()
    if not python_bin.exists():
        raise RuntimeError(f"AgentDojo Python executable does not exist: {python_bin}")
    return python_bin


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--defense", choices=["none", "obligate"], default="obligate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-case-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
