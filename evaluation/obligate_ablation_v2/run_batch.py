"""Concurrent, resumable AgentDojo batch runner for one model/variant cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from .common import atomic_write_json, sha256_file, utc_now
from .protocol import E2E_VARIANTS, execution_contract

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(
    os.environ.get("OBLIGATE_AGENTDOJO_PYTHON")
    or os.environ.get("OBLIGATE_PYTHON")
    or sys.executable
).resolve()


def _selected_cases(manifest: Mapping[str, Any], case_ids_file: Path | None) -> list[dict[str, Any]]:
    cases = [dict(item) for item in manifest["cases"]]
    if case_ids_file is None:
        return cases
    requested = {
        line.strip()
        for line in case_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    selected = [item for item in cases if str(item["case_id"]) in requested]
    missing = requested - {str(item["case_id"]) for item in selected}
    if missing:
        raise ValueError(f"case-id subset contains {len(missing)} unknown cases")
    return selected


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    cases = _selected_cases(manifest, args.case_ids_file)
    out_dir = args.output_root / "e2e" / args.model / args.variant
    cache_path = args.output_root / "model_cache" / f"{args.model}.sqlite3"
    manifest_sha256 = sha256_file(args.case_manifest)
    scalar_sha256 = (
        sha256_file(args.scalar_thresholds) if args.scalar_thresholds is not None else None
    )
    plan: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        run_name = f"{case_id}_obligate_ablation_v2_{args.variant.replace('-', '_')}"
        output = out_dir / "raw_runs" / f"{run_name}.json"
        command = [
            str(PYTHON),
            "-m",
            "evaluation.obligate_ablation_v2.run_case",
            "--case-id",
            case_id,
            "--suite",
            str(case["suite"]),
            "--user-task-id",
            str(case["user_task_id"]),
            "--injection-task-id",
            str(case["injection_task_id"]),
            "--model",
            args.model,
            "--variant",
            args.variant,
            "--run-name",
            run_name,
            "--output",
            str(output.resolve()),
            "--trace-dir",
            str((out_dir / "full_traces").resolve()),
            "--snapshots",
            str((out_dir / "snapshots" / f"{case_id}.jsonl").resolve()),
            "--decision-trace",
            str((out_dir / "decision_traces" / f"{case_id}.jsonl").resolve()),
            "--model-cache",
            str(cache_path.resolve()),
            "--model-cache-audit",
            str((out_dir / "model_cache_audit" / f"{case_id}.jsonl").resolve()),
            "--case-manifest",
            str(args.case_manifest.resolve()),
            "--seed",
            str(args.seed),
            "--bundle-sha256",
            args.bundle_sha256,
        ]
        if args.scalar_thresholds is not None:
            command.extend(["--scalar-thresholds", str(args.scalar_thresholds.resolve())])
        plan.append(
            {
                "case_id": case_id,
                "output": str(output.resolve()),
                "trace_dir": str((out_dir / "full_traces").resolve()),
                "snapshots": str((out_dir / "snapshots" / f"{case_id}.jsonl").resolve()),
                "decision_trace": str(
                    (out_dir / "decision_traces" / f"{case_id}.jsonl").resolve()
                ),
                "model_cache_audit": str(
                    (out_dir / "model_cache_audit" / f"{case_id}.jsonl").resolve()
                ),
                "retry_trace": str((out_dir / "retry_traces" / f"{case_id}.json").resolve()),
                "retry_attempts": str((out_dir / "retry_attempts" / case_id).resolve()),
                "bundle_sha256": args.bundle_sha256,
                "manifest_sha256": manifest_sha256,
                "scalar_thresholds_sha256": scalar_sha256,
                "command": command,
            }
        )
    return plan


def _valid_output(
    path: Path,
    *,
    case_id: str,
    model: str,
    variant: str,
    bundle_sha256: str,
    manifest_sha256: str,
    scalar_thresholds_sha256: str | None,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        meta = value["obligate_ablation_v2"]
        rows = value["normalized_cases"]
    except Exception:
        return False
    return (
        len(rows) == 1
        and str(meta.get("case_id")) == case_id
        and str(meta.get("model")) == model
        and str(meta.get("variant")) == variant
        and meta.get("bundle_aggregate_sha256") == bundle_sha256
        and meta.get("case_manifest_sha256") == manifest_sha256
        and meta.get("scalar_thresholds_sha256") == scalar_thresholds_sha256
    )


def _retry_history_state(
    path: Path,
    output: Path | None = None,
    *,
    expected_case_id: str | None = None,
) -> str | None:
    if not path.is_file():
        return None
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
        attempts = trace.get("attempts") or []
        latest = attempts[-1]
        numbers = [int(row.get("attempt") or 0) for row in attempts]
        failures_valid = all(
            (
                row.get("status") == "failed"
                and isinstance(row.get("returncode"), int)
                and "stderr_tail" in row
            )
            or str(row.get("status") or "").startswith("interrupted_")
            for row in attempts
            if row.get("status") != "succeeded"
        )
        history_valid = bool(attempts) and (
            numbers == list(range(1, len(attempts) + 1))
            and len(attempts) <= 4
            and all(row.get("status") != "succeeded" for row in attempts[:-1])
            and failures_valid
        )
        if (
            not history_valid
            or trace.get("runner_internal_attempts") != 1
            or trace.get("max_fresh_process_attempts") != 4
            or (expected_case_id is not None and trace.get("case_id") != expected_case_id)
        ):
            return None
        if latest.get("status") == "succeeded":
            if output is None or not output.is_file():
                return None
            meta = json.loads(output.read_text(encoding="utf-8"))["obligate_ablation_v2"]
            return (
                "succeeded"
                if trace.get("case_id") == meta.get("case_id")
                and int(latest.get("attempt") or 0)
                == int(meta.get("fresh_process_attempt") or -1)
                else None
            )
        if (
            len(attempts) == 4
            and all(row.get("status") != "succeeded" for row in attempts)
        ):
            return "exhausted"
        return None
    except Exception:
        return None


def _valid_retry_trace(path: Path, output: Path) -> bool:
    return _retry_history_state(path, output) == "succeeded"


def _redact(value: str, secret: str) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def _replace_option(command: list[str], option: str, value: Path | str) -> None:
    index = command.index(option)
    command[index + 1] = str(value)


def _attempt_command(item: Mapping[str, Any], attempt_root: Path, attempt: int) -> tuple[list[str], dict[str, Path]]:
    paths = {
        "output": attempt_root / "raw_run.json",
        "trace_dir": attempt_root / "full_traces",
        "snapshots": attempt_root / "snapshots.jsonl",
        "decision_trace": attempt_root / "decision_trace.jsonl",
        "model_cache_audit": attempt_root / "model_cache_audit.jsonl",
    }
    command = list(item["command"])
    for option, key in (
        ("--output", "output"),
        ("--trace-dir", "trace_dir"),
        ("--snapshots", "snapshots"),
        ("--decision-trace", "decision_trace"),
        ("--model-cache-audit", "model_cache_audit"),
    ):
        _replace_option(command, option, paths[key])
    command.extend(["--attempt-id", str(attempt)])
    return command, paths


def _move_file(source: Path, target: Path) -> None:
    if not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _promote_success(item: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    _move_file(paths["snapshots"], Path(str(item["snapshots"])))
    _move_file(paths["decision_trace"], Path(str(item["decision_trace"])))
    _move_file(paths["model_cache_audit"], Path(str(item["model_cache_audit"])))
    trace_source = paths["trace_dir"]
    if trace_source.is_dir():
        trace_target = Path(str(item["trace_dir"]))
        for source in sorted(trace_source.rglob("*"), key=lambda value: value.as_posix()):
            if source.is_file():
                _move_file(source, trace_target / source.relative_to(trace_source))
    # The normalized case output is the commit marker and is promoted last.
    _move_file(paths["output"], Path(str(item["output"])))


def _archive_orphaned_stable(item: Mapping[str, Any], attempt_root: Path) -> None:
    for key in ("output", "snapshots", "decision_trace", "model_cache_audit"):
        source = Path(str(item[key]))
        if source.is_file():
            target = attempt_root / "orphaned_stable" / f"{key}{source.suffix}"
            _move_file(source, target)


def _load_retry_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("attempts") or []
        result = [dict(row) for row in rows if isinstance(row, Mapping)]
        for row in result:
            if row.get("status") == "produced":
                row["status"] = "interrupted_before_commit"
                row["recovered_from_status"] = "produced"
        return result
    except Exception:
        return []


def _write_retry_trace(
    path: Path,
    item: Mapping[str, Any],
    attempts: int,
    history: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": "obligate-ablation-v2-retry-trace-v1",
            "case_id": item["case_id"],
            "max_fresh_process_attempts": attempts,
            "runner_internal_attempts": 1,
            "attempts": history,
        },
    )


def _run_one(
    item: Mapping[str, Any],
    env: Mapping[str, str],
    secret: str,
    attempts: int,
    initial_backoff: float,
) -> dict[str, Any] | None:
    retry_trace = Path(str(item["retry_trace"]))
    history = _load_retry_history(retry_trace)
    attempts_root = Path(str(item["retry_attempts"]))
    existing = [
        int(path.name.split("-", 1)[1])
        for path in attempts_root.glob("attempt-*")
        if path.is_dir() and path.name.split("-", 1)[1].isdigit()
    ]
    recorded = {int(row.get("attempt") or 0) for row in history}
    for attempt in sorted(set(existing) - recorded):
        history.append(
            {
                "attempt": attempt,
                "status": "interrupted_before_attempt_record",
                "artifact_root": str((attempts_root / f"attempt-{attempt:02d}").resolve()),
            }
        )
    history.sort(key=lambda row: int(row.get("attempt") or 0))
    if history:
        _write_retry_trace(retry_trace, item, attempts, history)
    next_attempt = max([0, *existing, *(int(row.get("attempt") or 0) for row in history)]) + 1
    _archive_orphaned_stable(
        item, attempts_root / f"orphaned-before-attempt-{next_attempt:02d}"
    )
    if len(history) >= attempts:
        return {
            "case_id": item["case_id"],
            "reason": "frozen fresh-process retry budget exhausted",
            "retry_trace": str(retry_trace),
            "attempt_count": len(history),
        }

    while len(history) < attempts:
        attempt = next_attempt
        attempt_root = attempts_root / f"attempt-{attempt:02d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        command, paths = _attempt_command(item, attempt_root, attempt)
        started_at = utc_now()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(env),
            text=True,
            capture_output=True,
            check=False,
        )
        valid = completed.returncode == 0 and _valid_output(
            paths["output"],
            case_id=str(item["case_id"]),
            model=str(item["model"]),
            variant=str(item["variant"]),
            bundle_sha256=str(item["bundle_sha256"]),
            manifest_sha256=str(item["manifest_sha256"]),
            scalar_thresholds_sha256=(
                str(item["scalar_thresholds_sha256"])
                if item.get("scalar_thresholds_sha256") is not None
                else None
            ),
        )
        row = {
            "attempt": attempt,
            "status": "produced" if valid else "failed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "returncode": completed.returncode,
            "stdout_tail": _redact(completed.stdout[-2000:], secret),
            "stderr_tail": _redact(completed.stderr[-4000:], secret),
            "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            "artifact_root": str(attempt_root.resolve()),
        }
        history.append(row)
        _write_retry_trace(retry_trace, item, attempts, history)
        if valid:
            try:
                _promote_success(item, paths)
            except BaseException as exc:
                row["status"] = "failed"
                row["commit_error_type"] = type(exc).__name__
                row["commit_error"] = _redact(str(exc), secret)[:1000]
                _write_retry_trace(retry_trace, item, attempts, history)
            else:
                row["status"] = "succeeded"
                row["committed_at"] = utc_now()
                _write_retry_trace(retry_trace, item, attempts, history)
                return None
        next_attempt += 1
        if len(history) < attempts:
            time.sleep(min(initial_backoff * (2 ** (len(history) - 1)), 30.0))
    return {
        "case_id": item["case_id"],
        "reason": "all frozen fresh-process attempts failed",
        "retry_trace": str(retry_trace),
        "attempt_count": len(history),
    }


def run(args: argparse.Namespace) -> int:
    if args.workers < 1 or args.case_attempts < 1 or args.retry_initial_backoff < 0:
        raise ValueError("workers/case-attempts must be positive and retry backoff nonnegative")
    if args.variant == "scalar-average" and args.scalar_thresholds is None:
        raise ValueError("scalar-average requires --scalar-thresholds")
    plan = build_plan(args)
    for item in plan:
        item["model"] = args.model
        item["variant"] = args.variant
    out_dir = args.output_root / "e2e" / args.model / args.variant
    if not args.resume and out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("non-resume batch rejected because the cell already contains artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    secret = env.get(args.api_key_env, "")
    if not secret:
        raise RuntimeError(f"missing provider credential environment: {args.api_key_env}")
    credential_sha256 = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    if credential_sha256 != args.credential_sha256:
        raise RuntimeError(
            f"credential fingerprint drift for {args.model}/{args.api_key_env}; refusing formal call"
        )
    env["OBLIGATE_LLM_API_KEY"] = secret
    env["OBLIGATE_LLM_BASE_URL"] = args.base_url
    env["OBLIGATE_LLM_PROVIDER"] = "openai_compatible"
    env["OBLIGATE_OPENAI_COMPAT_SYSTEM_ROLE"] = "1"
    env["OBLIGATE_LLM_TIMEOUT"] = "300"
    env["PYTHONHASHSEED"] = str(args.seed)
    env["OBLIGATE_EVAL_SIMULATOR_ONLY"] = "1"
    # Runner-level retries reuse a partially mutated AgentDojo pipeline.  Keep
    # them disabled and retry only by spawning a fresh per-case process below.
    env["AGENTDOJO_LLM_RETRY_ATTEMPTS"] = "1"
    env["AGENTDOJO_LLM_RETRY_INITIAL_SEC"] = str(args.retry_initial_backoff)

    config = {
        "schema_version": "obligate-ablation-v2-batch-v1",
        "formal": bool(args.formal),
        "model": args.model,
        "variant": args.variant,
        "case_count": len(plan),
        "case_manifest": str(args.case_manifest.resolve()),
        "case_manifest_sha256": sha256_file(args.case_manifest),
        "case_subset": str(args.case_ids_file.resolve()) if args.case_ids_file else None,
        "case_subset_sha256": sha256_file(args.case_ids_file) if args.case_ids_file else None,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "credential_sha256": credential_sha256,
        "temperature": 0.0,
        "reasoning_effort": None,
        "max_iters": 24,
        "timeout_seconds": 300,
        "task_retry_max_attempts": args.case_attempts,
        "runner_internal_retry_max_attempts": 1,
        "fresh_process_case_attempts": args.case_attempts,
        "task_retry_initial_backoff_seconds": args.retry_initial_backoff,
        "case_attempts": args.case_attempts,
        "seed": args.seed,
        "bundle_aggregate_sha256": args.bundle_sha256,
        "provider_request_seed": None,
        "workers": args.workers,
        "scalar_thresholds": str(args.scalar_thresholds.resolve()) if args.scalar_thresholds else None,
        "scalar_thresholds_sha256": sha256_file(args.scalar_thresholds) if args.scalar_thresholds else None,
        **execution_contract(args.variant),
        "started_at": utc_now(),
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists() and args.resume:
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        stable_keys = set(config) - {"started_at"}
        if any(prior.get(key) != config.get(key) for key in stable_keys):
            raise RuntimeError("resume rejected because frozen batch configuration changed")
    else:
        atomic_write_json(config_path, config)
    atomic_write_json(out_dir / "execution_plan.json", {"config": config, "commands": plan})

    pending = [
        item
        for item in plan
        if not (
            args.resume
            and _valid_output(
                Path(str(item["output"])),
                case_id=str(item["case_id"]),
                model=args.model,
                variant=args.variant,
                bundle_sha256=args.bundle_sha256,
                manifest_sha256=str(item["manifest_sha256"]),
                scalar_thresholds_sha256=(
                    str(item["scalar_thresholds_sha256"])
                    if item.get("scalar_thresholds_sha256") is not None
                    else None
                ),
            )
            and _valid_retry_trace(
                Path(str(item["retry_trace"])), Path(str(item["output"]))
            )
        )
    ]
    completed_count = len(plan) - len(pending)
    failures: list[dict[str, Any]] = []
    atomic_write_json(
        out_dir / "progress.json",
        {"completed": completed_count, "total": len(plan), "failure_count": 0, "workers": args.workers},
    )
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix=f"{args.model}-{args.variant}") as pool:
        futures = {
            pool.submit(
                _run_one,
                item,
                env,
                secret,
                args.case_attempts,
                args.retry_initial_backoff,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            failure = future.result()
            completed_count += 1
            if failure is not None or not _valid_output(
                Path(str(item["output"])),
                case_id=str(item["case_id"]),
                model=args.model,
                variant=args.variant,
                bundle_sha256=args.bundle_sha256,
                manifest_sha256=str(item["manifest_sha256"]),
                scalar_thresholds_sha256=(
                    str(item["scalar_thresholds_sha256"])
                    if item.get("scalar_thresholds_sha256") is not None
                    else None
                ),
            ) or not _valid_retry_trace(
                Path(str(item["retry_trace"])), Path(str(item["output"]))
            ):
                failures.append(failure or {"case_id": item["case_id"], "attempts": []})
            atomic_write_json(out_dir / "failures.json", failures)
            atomic_write_json(
                out_dir / "progress.json",
                {
                    "completed": completed_count,
                    "total": len(plan),
                    "failure_count": len(failures),
                    "workers": args.workers,
                    "updated_at": utc_now(),
                },
            )
    config["completed_at"] = utc_now()
    config["failure_count"] = len(failures)
    config["completed_case_count"] = len(plan) - len(failures)
    atomic_write_json(out_dir / "run_config.json", config)
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=E2E_VARIANTS, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--case-attempts", type=int, default=4)
    parser.add_argument("--retry-initial-backoff", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--scalar-thresholds", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--credential-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_plan", "run"]
