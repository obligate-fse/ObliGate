"""Strict, resumable formal pipeline for ObliGate AgentDojo Ablation v2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .collect import collect_all
from .common import atomic_write_json, sha256_file, utc_now
from .counterfactual import replay as replay_counterfactual
from .fault_injection import run_fault_injection
from .freeze import bundle_fingerprint, freeze_before, verify_after
from .prepare import MODELS
from .scalar_crossfit import crossfit
from .snapshot_analysis import consolidate
from .statistics import write_statistics_outputs

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(
    os.environ.get("OBLIGATE_AGENTDOJO_PYTHON")
    or os.environ.get("OBLIGATE_PYTHON")
    or sys.executable
).resolve()
VARIANT_CONFIG = Path(
    os.environ.get("OBLIGATE_ABLATION_CONFIG") or os.environ.get("OBLIGATE_RQ4_CONFIG")
    or ROOT / "configs" / "ablation_variants.yaml"
).resolve()


def _variant_order() -> tuple[str, ...]:
    if not VARIANT_CONFIG.is_file():
        raise RuntimeError(f"missing ablation variant configuration: {VARIANT_CONFIG}")
    value = yaml.safe_load(VARIANT_CONFIG.read_text(encoding="utf-8"))
    variants = tuple(str(item) for item in (value or {}).get("e2e", ()))
    if not variants or variants[0] != "full":
        raise RuntimeError("ablation variant configuration must begin with full")
    from .protocol import E2E_VARIANTS
    expected = set(E2E_VARIANTS)
    if set(variants) != expected or len(variants) != len(expected):
        raise RuntimeError(f"unexpected ablation variant set: {variants}")
    return variants[1:]


VARIANT_ORDER = _variant_order()
VALIDITY_PASS_STATUSES = frozenset({"valid", "complete"})
FORMAL_INVALID_EXIT_CODE = 4
FORMAL_ARTIFACT_NAMES = (
    "e2e",
    "fault_injection",
    "hashes",
    "metrics",
    "model_cache",
    "paper_tables",
    "pipeline_logs",
    "reports",
    "snapshots",
    "statistics",
    "traces",
)


def _status(root: Path, stage: str, **extra: Any) -> None:
    prior: dict[str, Any] = {}
    path = root / "pipeline_status.json"
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
    if stage != "formal_complete":
        for key in (
            "completed_at",
            "invalidated_at",
            "validity_audit",
            "validity_audit_sha256",
            "validity_gate_error",
            "validity_overall_status",
            "validity_status_counts",
        ):
            prior.pop(key, None)
        prior["formal_complete"] = False
        prior["validity_gate_passed"] = False
    atomic_write_json(
        path,
        {
            **prior,
            "schema_version": "obligate-ablation-v2-pipeline-status-v1",
            "stage": stage,
            "updated_at": utc_now(),
            **extra,
        },
    )


def _run_cell_pair(
    root: Path,
    *,
    variant: str,
    manifest: Path,
    thresholds: Path | None,
    resume: bool,
) -> None:
    log_dir = root / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[str], Any, Any]] = []
    frozen_bundle = json.loads(
        (root / "hashes" / "bundle_fingerprint.json").read_text(encoding="utf-8")
    )
    frozen_models = json.loads(
        (root / "config" / "model_configs.json").read_text(encoding="utf-8")
    )
    for model, route in MODELS.items():
        frozen_route = frozen_models.get(model) or {}
        if (
            frozen_route.get("base_url") != route["base_url"]
            or frozen_route.get("api_key_env") != route["api_key_env"]
        ):
            raise RuntimeError(f"frozen route drift for {model}")
        command = [
            str(PYTHON),
            "-m",
            "evaluation.obligate_ablation_v2.run_batch",
            "--case-manifest",
            str(manifest.resolve()),
            "--output-root",
            str(root.resolve()),
            "--model",
            model,
            "--variant",
            variant,
            "--base-url",
            str(route["base_url"]),
            "--api-key-env",
            str(route["api_key_env"]),
            "--workers",
            "8",
            "--case-attempts",
            "4",
            "--retry-initial-backoff",
            "2",
            "--seed",
            "20260716",
            "--formal",
            "--bundle-sha256",
            str(frozen_bundle["aggregate_sha256"]),
            "--credential-sha256",
            str(frozen_route["credential_sha256"]),
        ]
        if resume:
            command.append("--resume")
        if variant == "scalar-average":
            if thresholds is None or not thresholds.is_file():
                raise RuntimeError("scalar-average stage has no frozen cross-fit thresholds")
            command.extend(["--scalar-thresholds", str(thresholds.resolve())])
        log_mode = "a" if resume else "x"
        stdout = (log_dir / f"{model}.{variant}.stdout.log").open(log_mode, encoding="utf-8")
        stderr = (log_dir / f"{model}.{variant}.stderr.log").open(log_mode, encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        processes.append((model, process, stdout, stderr))
    _status(
        root,
        f"e2e:{variant}",
        active_cells=[{"model": model, "variant": variant, "pid": process.pid} for model, process, _, _ in processes],
    )
    process_failures: list[dict[str, Any]] = []
    try:
        while processes:
            remaining = []
            for model, process, stdout, stderr in processes:
                code = process.poll()
                if code is None:
                    remaining.append((model, process, stdout, stderr))
                    continue
                stdout.close()
                stderr.close()
                if code != 0:
                    process_failures.append({"model": model, "variant": variant, "returncode": code})
            processes = remaining
            if processes:
                time.sleep(2.0)
    finally:
        for _, process, stdout, stderr in processes:
            if process.poll() is None:
                process.terminate()
            stdout.close()
            stderr.close()
    invalid_counts: dict[str, int] = {}
    for model in MODELS:
        progress = json.loads((root / "e2e" / model / variant / "progress.json").read_text(encoding="utf-8"))
        if progress.get("completed") != 949:
            raise RuntimeError(f"{model}/{variant} incomplete: {progress}")
        invalid_counts[model] = int(progress.get("failure_count") or 0)
    # Provider/workflow/parser failures are valid conservative-denominator
    # observations once every case has been attempted.  They remain in
    # failures.json and later become explicit per-case invalid rows; a worker
    # return code is fatal only when the 949-case attempt set is incomplete.
    _status(
        root,
        f"e2e:{variant}:complete",
        active_cells=[],
        process_returncode_warnings=process_failures,
        invalid_case_counts=invalid_counts,
    )


def _combine_cache_audit(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    audit_paths = {
        *root.glob("e2e/*/*/model_cache_audit/*.jsonl"),
        *root.glob("e2e/*/*/retry_attempts/*/attempt-*/model_cache_audit.jsonl"),
    }
    for path in sorted(audit_paths, key=lambda item: item.as_posix()):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    state_sources: Counter[str] = Counter()
    logical_input_tokens = logical_output_tokens = cached_tokens = 0
    provider_input_tokens = provider_output_tokens = 0
    for row in rows:
        by_prompt[str(row.get("prompt_hash"))].append(row)
        by_cell[f"{row.get('model')}/{row.get('variant')}"][str(row.get("cache_status"))] += 1
        state_sources[str(row.get("visible_tool_state_source") or "missing")] += 1
        usage = row.get("token_usage") or {}
        row_input = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        row_output = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        logical_input_tokens += row_input
        logical_output_tokens += row_output
        if row.get("provider_called") and str(row.get("cache_status") or "") != "error":
            provider_input_tokens += row_input
            provider_output_tokens += row_output
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens += int(details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
    integrity = {
        "same_hash_multiple_request_hash": [
            prompt
            for prompt, values in by_prompt.items()
            if len({row.get("request_hash") for row in values if row.get("request_hash")}) > 1
        ],
        "same_hash_multiple_response_hash": [
            prompt
            for prompt, values in by_prompt.items()
            if len({row.get("response_hash") for row in values if row.get("response_hash")}) > 1
        ],
        "same_hash_multiple_provider_calls": [
            prompt
            for prompt, values in by_prompt.items()
            if sum(
                bool(row.get("provider_called"))
                and str(row.get("cache_status") or "") != "error"
                and bool(row.get("response_hash"))
                for row in values
            )
            > 1
        ],
        "tool_state_fallback_rows": [
            index
            for index, row in enumerate(rows)
            if row.get("visible_tool_state_source") != "explicit_case_reset_state"
            or row.get("visible_tool_state_fallback_used") is not False
        ],
    }
    variants_with_hits = sorted(
        {f"{row.get('model')}/{row.get('variant')}" for row in rows if row.get("cache_hit") and row.get("variant") != "full"}
    )
    raw_path = root / "traces" / "model_cache_audit.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, raw_path)
    value = {
        "schema_version": "obligate-model-cache-audit-summary-v1",
        "created_at": utc_now(),
        "audit_row_count": len(rows),
        "unique_prompt_hash_count": len(by_prompt),
        "provider_call_count": sum(bool(row.get("provider_called")) for row in rows),
        "cache_hit_count": sum(bool(row.get("cache_hit")) for row in rows),
        "cache_miss_count": sum(row.get("cache_status") == "miss" for row in rows),
        "input_tokens": provider_input_tokens,
        "output_tokens": provider_output_tokens,
        "provider_billed_input_tokens": provider_input_tokens,
        "provider_billed_output_tokens": provider_output_tokens,
        "logical_input_tokens_including_cache_reuse": logical_input_tokens,
        "logical_output_tokens_including_cache_reuse": logical_output_tokens,
        "provider_error_call_count_with_unknown_billing": sum(
            bool(row.get("provider_called")) and str(row.get("cache_status") or "") == "error"
            for row in rows
        ),
        "provider_reported_cached_input_tokens": cached_tokens,
        "by_cell": {key: dict(value) for key, value in sorted(by_cell.items())},
        "visible_tool_state_sources": dict(sorted(state_sources.items())),
        "variant_cells_with_common_prefix_hits": variants_with_hits,
        "integrity": integrity,
        "integrity_pass": not integrity["same_hash_multiple_request_hash"]
        and not integrity["same_hash_multiple_response_hash"]
        and not integrity["same_hash_multiple_provider_calls"]
        and not integrity["tool_state_fallback_rows"],
        "raw_api_keys_recorded": False,
        "raw_audit_jsonl": str(raw_path.resolve()),
        "raw_audit_sha256": sha256_file(raw_path),
    }
    atomic_write_json(root / "traces" / "model_cache_summary.json", value)
    return value


def _run_reports(root: Path) -> tuple[dict[str, str], dict[str, Any], Path]:
    from .report import generate_reports

    outputs = generate_reports(root)
    audit_path = Path(outputs.get("validity") or root / "reports" / "validity_audit.json").resolve()
    if not audit_path.is_file():
        return (
            outputs,
            {
                "overall_status": "missing",
                "status_counts": {},
                "gate_error": f"generated validity audit is missing: {audit_path}",
            },
            audit_path,
        )
    try:
        value = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (
            outputs,
            {
                "overall_status": "unreadable",
                "status_counts": {},
                "gate_error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            },
            audit_path,
        )
    if not isinstance(value, dict):
        return (
            outputs,
            {
                "overall_status": "malformed",
                "status_counts": {},
                "gate_error": "generated validity audit top-level value is not an object",
            },
            audit_path,
        )
    return outputs, value, audit_path


def _finalize_reporting_gate(
    root: Path,
    *,
    snapshot_manifest: dict[str, Any],
    threshold_seal: dict[str, Any],
) -> int:
    outputs, audit, audit_path = _run_reports(root)
    overall_status = str(audit.get("overall_status") or "missing").strip().casefold()
    gate_passed = overall_status in VALIDITY_PASS_STATUSES
    common = {
        "formal_complete": gate_passed,
        "validity_gate_passed": gate_passed,
        "validity_overall_status": overall_status,
        "validity_status_counts": audit.get("status_counts") or {},
        "validity_audit": str(audit_path),
        "validity_audit_sha256": sha256_file(audit_path) if audit_path.is_file() else None,
        "report_outputs": outputs,
        "expected_episode_count": 11_388,
        "snapshot_manifest": snapshot_manifest,
        "protected_results_unchanged": True,
        "frozen_code_unchanged": True,
        "frozen_bundle_unchanged": True,
        "scalar_threshold_seal_verified": True,
        "scalar_threshold_seal": threshold_seal,
    }
    if not gate_passed:
        _status(
            root,
            "formal_invalid",
            **common,
            validity_gate_error=audit.get("gate_error"),
            completed_at=None,
            invalidated_at=utc_now(),
        )
        return FORMAL_INVALID_EXIT_CODE
    _status(
        root,
        "formal_complete",
        **common,
        validity_gate_error=None,
        invalidated_at=None,
        completed_at=utc_now(),
    )
    return 0


def _assert_clean_formal_start(root: Path) -> None:
    existing = [name for name in FORMAL_ARTIFACT_NAMES if (root / name).exists()]
    dynamic_threshold = root / "config" / "scalar_crossfit_thresholds.json"
    if dynamic_threshold.exists():
        existing.append(str(dynamic_threshold.relative_to(root)))
    if existing:
        raise RuntimeError(
            "non-resume formal start rejected because formal artifacts already exist: "
            + ", ".join(sorted(existing))
        )


def _frozen_bundle(root: Path) -> dict[str, Any]:
    path = root / "hashes" / "bundle_fingerprint.json"
    if not path.is_file():
        raise RuntimeError("resume requires an existing frozen bundle fingerprint")
    value = json.loads(path.read_text(encoding="utf-8"))
    current = bundle_fingerprint(root)
    if current["aggregate_sha256"] != value.get("aggregate_sha256"):
        raise RuntimeError("resume rejected because frozen code/config/manifest bundle changed")
    preflight = root / "preflight" / "preflight.json"
    if not preflight.is_file() or sha256_file(preflight) != value.get("preflight_gate_sha256"):
        raise RuntimeError("resume rejected because the preflight gate artifact changed")
    return value


def _write_or_verify_threshold_seal(
    root: Path,
    thresholds: Path,
    value: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    seal_path = root / "hashes" / "scalar_crossfit_seal.json"
    if resume and seal_path.is_file():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if not thresholds.is_file() or sha256_file(thresholds) != seal.get("thresholds_sha256"):
            raise RuntimeError("resume rejected because scalar cross-fit thresholds drifted")
        frozen_value = json.loads(thresholds.read_text(encoding="utf-8"))
        frozen_semantic = {key: item for key, item in frozen_value.items() if key != "created_at"}
        recomputed_semantic = {key: item for key, item in value.items() if key != "created_at"}
        if frozen_semantic != recomputed_semantic:
            raise RuntimeError("resume rejected because recomputed scalar thresholds changed")
        if sha256_file(root / "snapshots" / "full_guarded_actions.jsonl") != seal.get(
            "full_guarded_actions_sha256"
        ):
            raise RuntimeError("resume rejected because the sealed Full snapshots changed")
        return seal
    if resume and thresholds.exists() and not seal_path.is_file():
        raise RuntimeError("resume rejected because scalar thresholds have no phase seal")
    atomic_write_json(thresholds, value)
    seal = {
        "schema_version": "obligate-scalar-crossfit-phase-seal-v1",
        "created_at": utc_now(),
        "thresholds": str(thresholds.resolve()),
        "thresholds_sha256": sha256_file(thresholds),
        "full_guarded_actions_sha256": sha256_file(
            root / "snapshots" / "full_guarded_actions.jsonl"
        ),
        "outcome_or_scorer_labels_read": False,
        "shared_across_models": True,
    }
    atomic_write_json(seal_path, seal)
    return seal


def _verify_threshold_seal(root: Path, thresholds: Path) -> dict[str, Any]:
    seal_path = root / "hashes" / "scalar_crossfit_seal.json"
    if not seal_path.is_file() or not thresholds.is_file():
        raise RuntimeError("scalar threshold phase seal is missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    digest = sha256_file(thresholds)
    if digest != seal.get("thresholds_sha256"):
        raise RuntimeError("scalar cross-fit thresholds changed after phase seal")
    for model in MODELS:
        config_path = root / "e2e" / model / "scalar-average" / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("scalar_thresholds_sha256") != digest:
            raise RuntimeError(f"{model} scalar cell did not use the phase-sealed thresholds")
    return seal


def run(root: Path, *, resume: bool = False) -> int:
    manifest = root / "manifests" / "agentdojo_949.json"
    preflight = root / "preflight" / "preflight.json"
    if not manifest.is_file() or not preflight.is_file():
        raise RuntimeError("formal pipeline requires prepared manifest and completed preflight")
    gate = json.loads(preflight.read_text(encoding="utf-8"))
    if not gate.get("formal_run_allowed"):
        raise RuntimeError("formal pipeline blocked because preflight did not pass")

    current_bundle = bundle_fingerprint(root)
    preflight_bundle = gate.get("bundle_fingerprint") or {}
    if preflight_bundle.get("aggregate_sha256") != current_bundle["aggregate_sha256"]:
        raise RuntimeError("formal pipeline blocked because preflight bundle no longer matches current code/config")

    _status(root, "freezing", formal_run_allowed=True)
    before = root / "hashes" / "protected_before.json"
    if resume:
        if not before.is_file():
            raise RuntimeError("resume requested but protected-results freeze is missing")
        frozen_bundle = _frozen_bundle(root)
    else:
        _assert_clean_formal_start(root)
        freeze_before(root)
        frozen_bundle = _frozen_bundle(root)
    if frozen_bundle["aggregate_sha256"] != preflight_bundle.get("aggregate_sha256"):
        raise RuntimeError("frozen formal bundle was not the bundle exercised by preflight")
    frozen_code = json.loads((root / "hashes" / "code_hashes.json").read_text(encoding="utf-8"))
    _status(
        root,
        "frozen",
        code_aggregate_sha256=frozen_code["aggregate_sha256"],
        bundle_aggregate_sha256=frozen_bundle["aggregate_sha256"],
        preflight_bundle_match=True,
    )

    _run_cell_pair(root, variant="full", manifest=manifest, thresholds=None, resume=resume)
    _status(root, "full_postprocessing")
    snapshot_manifest = consolidate(
        {
            "deepseek-v4-flash": root / "e2e" / "deepseek-v4-flash" / "full" / "snapshots",
            "qwen-plus": root / "e2e" / "qwen-plus" / "full" / "snapshots",
        },
        root,
    )
    thresholds_value = crossfit(
        {
            "deepseek-v4-flash": root / "e2e" / "deepseek-v4-flash" / "full" / "snapshots",
            "qwen-plus": root / "e2e" / "qwen-plus" / "full" / "snapshots",
        },
        seed=20260716,
        case_to_fold=json.loads(
            (root / "manifests" / "fold_assignment.json").read_text(encoding="utf-8")
        )["case_to_fold"],
    )
    thresholds = root / "config" / "scalar_crossfit_thresholds.json"
    threshold_seal = _write_or_verify_threshold_seal(
        root,
        thresholds,
        thresholds_value,
        resume=resume,
    )

    for variant in VARIANT_ORDER:
        _run_cell_pair(root, variant=variant, manifest=manifest, thresholds=thresholds, resume=resume)

    _status(root, "snapshot_counterfactual")
    replay_counterfactual(
        root / "snapshots" / "full_guarded_actions.jsonl",
        root / "snapshots" / "counterfactual_results.jsonl",
        root / "snapshots" / "counterfactual_summary.json",
    )
    _status(root, "fault_injection")
    run_fault_injection(
        root / "snapshots" / "full_guarded_actions.jsonl",
        feasibility_output=root / "fault_injection" / "feasibility_results.jsonl",
        certificate_output=root / "fault_injection" / "certificate_mutation_results.jsonl",
        summary_output=root / "fault_injection" / "summary.json",
        strict=True,
    )

    _status(root, "statistics")
    collect_all(root, manifest)
    write_statistics_outputs(root, iterations=10_000, seed=20260716)
    cache_audit = _combine_cache_audit(root)
    if not cache_audit["integrity_pass"]:
        raise RuntimeError("common-prefix cache integrity audit failed")
    threshold_seal = _verify_threshold_seal(root, thresholds)

    _status(root, "reporting")
    after = verify_after(root)
    if (
        not after["all_protected_results_unchanged"]
        or not after["formal_code_hash_unchanged"]
        or not after["formal_bundle_hash_unchanged"]
        or not after["preflight_gate_hash_unchanged"]
    ):
        raise RuntimeError("protected old result or frozen formal code/config/manifest bundle changed")
    return _finalize_reporting_gate(
        root,
        snapshot_manifest=snapshot_manifest,
        threshold_seal=threshold_seal,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args.output_root.resolve(), resume=args.resume)
    except BaseException as exc:
        _status(
            args.output_root.resolve(),
            "failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_INVALID_EXIT_CODE",
    "VALIDITY_PASS_STATUSES",
    "VARIANT_ORDER",
    "run",
]
