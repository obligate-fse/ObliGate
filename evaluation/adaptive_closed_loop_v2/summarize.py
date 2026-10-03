"""Summarize ObliGate adaptive closed-loop experiment artifacts.

The summarizer is intentionally read-only with respect to experiment outputs:
it reads score/usage/stop-reason files and writes derived summaries under
``results/adaptive_closed_loop_v2/reports``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .common import RESULT_ROOT, ROOT, read_json, read_jsonl, sha256_file, utc_now, write_json
from .statistics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    hierarchical_task_cluster_bootstrap_rate,
    holm_adjust,
    paired_binary_task_test,
    seed_rate_range,
)

REPORT_DIR = RESULT_ROOT / "reports"
FORMAL_SEGMENTS = {"main", "mechanism_stress", "agent_safetybench_main", "agent_safetybench_full"}
HISTORICAL_REFERENCE_ROOT = RESULT_ROOT.parent / "adaptive_ablation" / "formal_activation_v1"


def wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    if total <= 0:
        return {"rate": None, "lo": None, "hi": None}
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return {"rate": phat, "lo": (centre - margin) / denom, "hi": (centre + margin) / denom}


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def load_scores() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_path in sorted(RESULT_ROOT.glob("*/**/round_*/scores.jsonl")):
        if not score_path.is_file():
            continue
        meta = _path_meta(score_path)
        if meta.get("segment") not in FORMAL_SEGMENTS:
            continue
        rows.extend(read_jsonl(score_path))
    return rows


def load_expected_counts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for segment, name in (
        ("main", "main_manifest.json"),
        ("mechanism_stress", "mechanism_stress_manifest.json"),
        ("agent_safetybench_main", "agent_safetybench_main_manifest.json"),
        ("agent_safetybench_full", "agent_safetybench_full_manifest.json"),
    ):
        path = RESULT_ROOT / "manifests" / name
        if path.exists():
            manifest = read_json(path)
            out[segment] = {
                "case_count": int(manifest.get("case_count") or 0),
                "benchmark_counts": manifest.get("benchmark_counts") or {},
                "manifest_sha256": manifest.get("manifest_sha256"),
                "name": manifest.get("name"),
            }
    audit_path = RESULT_ROOT / "manifests" / "sampling_audit.json"
    if audit_path.exists():
        out["sampling_audit"] = read_json(audit_path)
    safetybench_audit_path = RESULT_ROOT / "manifests" / "agent_safetybench_full_audit.json"
    if safetybench_audit_path.exists():
        out["agent_safetybench_full_audit"] = read_json(safetybench_audit_path)
    return out


def manifest_case_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for name in ("main_manifest.json", "mechanism_stress_manifest.json", "agent_safetybench_main_manifest.json", "agent_safetybench_full_manifest.json"):
        path = RESULT_ROOT / "manifests" / name
        if not path.exists():
            continue
        manifest = read_json(path)
        for case in manifest.get("cases") or []:
            index[str(case.get("case_id"))] = case
    return index


def iter_group_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def basic_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    mechanism_totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    for row in rows:
        for key, value in (row.get("decision_counts") or {}).items():
            try:
                decision_counts[str(key)] += int(value)
            except Exception:  # noqa: BLE001
                decision_counts[str(key)] += 1
        for key, value in (row.get("mechanism_metrics") or {}).items():
            if isinstance(value, bool):
                mechanism_totals[str(key)] += int(value)
            elif isinstance(value, (int, float)):
                mechanism_totals[str(key)] += int(value)
        if row.get("reason_category"):
            reasons[str(row["reason_category"])] += 1
        if row.get("primary_strategy"):
            strategies[str(row["primary_strategy"])] += 1
    total = len(rows)
    attack_success = sum(1 for row in rows if row.get("attack_success"))
    observed = sum(1 for row in rows if row.get("attack_success_observed"))
    dangerous = sum(1 for row in rows if row.get("dangerous_candidate"))
    success_given_dangerous = sum(1 for row in rows if row.get("dangerous_candidate") and row.get("attack_success"))
    observed_given_dangerous = sum(1 for row in rows if row.get("dangerous_candidate") and row.get("attack_success_observed"))
    provider_valid = sum(1 for row in rows if row.get("provider_valid"))
    workflow_failure = sum(1 for row in rows if row.get("workflow_failure"))
    task_known = [row for row in rows if row.get("task_success") is not None]
    task_success = sum(1 for row in task_known if row.get("task_success"))
    return {
        "cases": total,
        "attack_success": attack_success,
        "attack_success_rate": wilson(attack_success, total),
        "attack_success_observed": observed,
        "attack_success_observed_rate": wilson(observed, total),
        "dangerous_candidate": dangerous,
        "dangerous_candidate_rate": wilson(dangerous, total),
        "conditional_attack_success_given_dangerous": wilson(success_given_dangerous, dangerous),
        "conditional_observed_bypass_given_dangerous": wilson(observed_given_dangerous, dangerous),
        "provider_valid": provider_valid,
        "provider_valid_rate": wilson(provider_valid, total),
        "workflow_failure": workflow_failure,
        "workflow_failure_rate": wilson(workflow_failure, total),
        "task_success_known_cases": len(task_known),
        "task_success": task_success,
        "task_success_rate": wilson(task_success, len(task_known)),
        "decision_counts": dict(sorted(decision_counts.items())),
        "mechanism_metric_totals": dict(sorted(mechanism_totals.items())),
        "reason_categories": dict(sorted(reasons.items())),
        "primary_strategies": dict(strategies.most_common()),
    }


def round_tables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group in sorted(iter_group_rows(rows, ("segment", "victim_model", "attacker_seed", "round")).items()):
        segment, model, seed, round_index = key
        item = {"segment": segment, "victim_model": model, "attacker_seed": seed, "round": round_index}
        item.update(basic_counts(group))
        out.append(item)
    return out


def benchmark_round_tables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group in sorted(iter_group_rows(rows, ("segment", "victim_model", "attacker_seed", "benchmark", "round")).items()):
        segment, model, seed, benchmark, round_index = key
        item = {"segment": segment, "victim_model": model, "attacker_seed": seed, "benchmark": benchmark, "round": round_index}
        item.update(basic_counts(group))
        out.append(item)
    return out


def cumulative_tables(rows: list[dict[str, Any]], horizons: tuple[int, ...] = (1, 2, 3, 4, 5)) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_keys = ("segment", "victim_model", "attacker_seed")
    for key, group in sorted(iter_group_rows(rows, base_keys).items()):
        out.extend(_cumulative_for_group(key, group, horizons, benchmark=None))
    bench_keys = ("segment", "victim_model", "attacker_seed", "benchmark")
    for key, group in sorted(iter_group_rows(rows, bench_keys).items()):
        out.extend(_cumulative_for_group(key[:3], group, horizons, benchmark=key[3]))
    return out


def adaptive_curve_tables(rows: list[dict[str, Any]], horizons: tuple[int, ...] = (1, 2, 3, 4, 5)) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_keys = ("segment", "victim_model")
    for key, group in sorted(iter_group_rows(rows, base_keys).items()):
        out.extend(_hierarchical_curve_for_group(key, group, horizons, benchmark=None))
    bench_keys = ("segment", "victim_model", "benchmark")
    for key, group in sorted(iter_group_rows(rows, bench_keys).items()):
        out.extend(_hierarchical_curve_for_group(key[:2], group, horizons, benchmark=key[2]))
    return out


def mechanism_tables(rows: list[dict[str, Any]], case_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        if row.get("segment") != "mechanism_stress":
            continue
        case = case_index.get(str(row.get("case_id"))) or {}
        item = dict(row)
        item["mechanism_category"] = case.get("mechanism_category") or "unknown"
        item["attack_mode"] = case.get("attack_mode") or "unknown"
        enriched.append(item)
    out: list[dict[str, Any]] = []
    for key, group in sorted(iter_group_rows(enriched, ("victim_model", "attacker_seed", "mechanism_category", "round")).items()):
        model, seed, category, round_index = key
        item = {
            "segment": "mechanism_stress",
            "victim_model": model,
            "attacker_seed": seed,
            "mechanism_category": category,
            "round": round_index,
        }
        item.update(basic_counts(group))
        out.append(item)
    for key, group in sorted(iter_group_rows(enriched, ("victim_model", "attacker_seed", "mechanism_category")).items()):
        model, seed, category = key
        for item in _cumulative_for_group(("mechanism_stress", model, seed), group, (1, 2, 3, 4, 5), benchmark=f"mechanism:{category}"):
            item["mechanism_category"] = category
            out.append(item)
    return out


def _cumulative_for_group(
    key: tuple[Any, Any, Any],
    group: list[dict[str, Any]],
    horizons: tuple[int, ...],
    *,
    benchmark: Any,
) -> list[dict[str, Any]]:
    segment, model, seed = key
    case_ids = {str(row.get("case_id")) for row in group}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        by_case[str(row.get("case_id"))].append(row)
    out: list[dict[str, Any]] = []
    for horizon in horizons:
        success_values = _task_seed_outcomes(group, horizon, "attack_success")
        observed_values = _task_seed_outcomes(group, horizon, "attack_success_observed")
        dangerous_values = _task_seed_outcomes(group, horizon, "dangerous_candidate")
        provider_error_values = _task_seed_outcomes(group, horizon, "provider_invalid_or_error", invert_provider_valid=True)
        workflow_error_values = _task_seed_outcomes(group, horizon, "workflow_failure")
        success_stat = hierarchical_task_cluster_bootstrap_rate(
            success_values,
            iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_BOOTSTRAP_SEED + int(horizon),
        )
        observed_stat = hierarchical_task_cluster_bootstrap_rate(
            observed_values,
            iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_BOOTSTRAP_SEED + 101 + int(horizon),
        )
        dangerous_stat = hierarchical_task_cluster_bootstrap_rate(
            dangerous_values,
            iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_BOOTSTRAP_SEED + 201 + int(horizon),
        )
        conditional_success_stat = hierarchical_task_cluster_bootstrap_rate(
            _conditional_task_seed_outcomes(success_values, dangerous_values),
            iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_BOOTSTRAP_SEED + 301 + int(horizon),
        )
        conditional_observed_stat = hierarchical_task_cluster_bootstrap_rate(
            _conditional_task_seed_outcomes(observed_values, dangerous_values),
            iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_BOOTSTRAP_SEED + 401 + int(horizon),
        )
        success_rounds: list[int] = []
        observed_rounds: list[int] = []
        for case_id, case_rows in by_case.items():
            eligible = [row for row in case_rows if int(row.get("round") or 0) < horizon]
            if not eligible:
                continue
            succ = [int(row.get("round") or 0) + 1 for row in eligible if row.get("attack_success")]
            obs = [int(row.get("round") or 0) + 1 for row in eligible if row.get("attack_success_observed")]
            if succ:
                success_rounds.append(min(succ))
            if obs:
                observed_rounds.append(min(obs))
        denom = len(case_ids)
        item = {
            "segment": segment,
            "victim_model": model,
            "attacker_seed": seed,
            "benchmark": benchmark or "ALL",
            "horizon": horizon,
            "cases": denom,
            "cumulative_attack_success": _task_mean_success_count(success_values),
            "CASR": success_stat,
            "cumulative_observed_bypass": _task_mean_success_count(observed_values),
            "observed_bypass_rate": observed_stat,
            "dangerous_candidate_cases": _task_mean_success_count(dangerous_values),
            "dangerous_candidate_rate": dangerous_stat,
            "conditional_attack_success_given_dangerous": conditional_success_stat,
            "conditional_observed_bypass_given_dangerous": conditional_observed_stat,
            "provider_or_workflow_error_cases": _case_union_success_count(provider_error_values, workflow_error_values),
            "median_round_to_success": median(success_rounds) if success_rounds else None,
            "median_round_to_observed_bypass": median(observed_rounds) if observed_rounds else None,
            "statistical_unit": "original_task/case_id",
            "repeated_observations": ["round", "tool_call", "attacker_seed"],
            "primary_interval": "task-cluster bootstrap",
        }
        out.append(item)
    return out


def _hierarchical_curve_for_group(
    key: tuple[Any, Any],
    group: list[dict[str, Any]],
    horizons: tuple[int, ...],
    *,
    benchmark: Any,
) -> list[dict[str, Any]]:
    segment, model = key
    out: list[dict[str, Any]] = []
    for horizon in horizons:
        success_values = _task_seed_outcomes(group, horizon, "attack_success")
        observed_values = _task_seed_outcomes(group, horizon, "attack_success_observed")
        dangerous_values = _task_seed_outcomes(group, horizon, "dangerous_candidate")
        success_stat = hierarchical_task_cluster_bootstrap_rate(
            success_values,
            iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            seed=DEFAULT_BOOTSTRAP_SEED + 1_000 + int(horizon),
        )
        out.append(
            {
                "segment": segment,
                "victim_model": model,
                "benchmark": benchmark or "ALL",
                "horizon": horizon,
                "tasks": len(success_values),
                "CASR": success_stat,
                "observed_bypass_rate": hierarchical_task_cluster_bootstrap_rate(
                    observed_values,
                    iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
                    seed=DEFAULT_BOOTSTRAP_SEED + 1_100 + int(horizon),
                ),
                "dangerous_candidate_rate": hierarchical_task_cluster_bootstrap_rate(
                    dangerous_values,
                    iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
                    seed=DEFAULT_BOOTSTRAP_SEED + 1_200 + int(horizon),
                ),
                "seed_rate_range": seed_rate_range(success_values),
                "statistical_unit": "original_task/case_id",
                "confidence_band": "95% task-cluster bootstrap",
            }
        )
    return out


def _task_seed_outcomes(
    rows: list[dict[str, Any]],
    horizon: int,
    field: str,
    *,
    invert_provider_valid: bool = False,
) -> dict[str, dict[str, bool]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("case_id")), str(row.get("attacker_seed")))].append(row)
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for (case_id, seed), case_rows in grouped.items():
        eligible = [row for row in case_rows if int(row.get("round") or 0) < horizon]
        if not eligible:
            continue
        if invert_provider_valid:
            value = any(not row.get("provider_valid") for row in eligible)
        else:
            value = any(bool(row.get(field)) for row in eligible)
        out[case_id][seed] = value
    return {case_id: dict(values) for case_id, values in out.items()}


def _conditional_task_seed_outcomes(
    numerator: dict[str, dict[str, bool]],
    denominator: dict[str, dict[str, bool]],
) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for case_id, seed_values in denominator.items():
        for seed, include in seed_values.items():
            if include:
                out[case_id][seed] = bool((numerator.get(case_id) or {}).get(seed))
    return {case_id: dict(values) for case_id, values in out.items()}


def _task_mean_success_count(task_seed_values: dict[str, dict[str, bool]]) -> int | float:
    total = 0.0
    for seed_values in task_seed_values.values():
        if seed_values:
            total += sum(float(value) for value in seed_values.values()) / len(seed_values)
    return int(total) if total.is_integer() else round(total, 6)


def _case_union_success_count(*task_seed_value_sets: dict[str, dict[str, bool]]) -> int:
    cases: set[str] = set()
    for values in task_seed_value_sets:
        for case_id, seed_values in values.items():
            if any(bool(value) for value in seed_values.values()):
                cases.add(case_id)
    return len(cases)


def attacker_generation_summary() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(RESULT_ROOT.glob("*/**/round_*/attacker_round_*.jsonl")):
        meta = _path_meta(path)
        if meta.get("segment") not in FORMAL_SEGMENTS:
            continue
        rows = read_jsonl(path)
        generated = len(rows)
        invalid = sum(1 for item in rows if item.get("generation_invalid"))
        strategies = Counter(str(item.get("primary_strategy")) for item in rows if not item.get("generation_invalid"))
        failures = Counter()
        for item in rows:
            if not item.get("generation_invalid"):
                continue
            details = item.get("validation_failures") or item.get("generation_error") or ["unknown"]
            if isinstance(details, list):
                for detail in details:
                    failures[str(detail)] += 1
            else:
                failures[str(details)] += 1
        out.append(
            {
                **meta,
                "path": str(path),
                "next_round": _round_number(path.name),
                "generated": generated,
                "valid": generated - invalid,
                "invalid": invalid,
                "invalid_rate": wilson(invalid, generated),
                "primary_strategies": dict(strategies.most_common()),
                "validation_failures": dict(failures.most_common()),
            }
        )
    return out


def stop_reason_summary() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(RESULT_ROOT.glob("*/**/seed_*/stop_reasons.json")):
        meta = _seed_meta(path)
        if meta.get("segment") not in FORMAL_SEGMENTS:
            continue
        data = read_json(path)
        reasons = Counter()
        rounds = Counter()
        for item in data.values():
            reasons[str(item.get("reason") or "unknown")] += 1
            rounds[str(item.get("round") if item.get("round") is not None else "unknown")] += 1
        out.append({**meta, "path": str(path), "stopped_cases": len(data), "reasons": dict(reasons.most_common()), "rounds": dict(sorted(rounds.items()))})
    return out


def usage_summary() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    files: Counter[tuple[str, str, str, str]] = Counter()
    for path in sorted(RESULT_ROOT.glob("*/**/*usage*.jsonl")):
        if not path.is_file():
            continue
        meta = _path_meta(path)
        if meta.get("segment") not in FORMAL_SEGMENTS:
            continue
        role = "attacker" if path.name == "attacker_usage.jsonl" else "victim"
        key = (str(meta.get("segment")), str(meta.get("victim_model")), str(meta.get("attacker_seed")), role)
        files[key] += 1
        for row in read_jsonl(path):
            grouped[key]["calls"] += 1
            grouped[key]["input_tokens"] += int(row.get("input_tokens") or 0)
            grouped[key]["output_tokens"] += int(row.get("output_tokens") or 0)
            grouped[key]["total_tokens"] += int(row.get("total_tokens") or 0)
            grouped[key]["token_usage_available_calls"] += int(bool(row.get("token_usage_available")))
            if row.get("status") and row.get("status") != "success":
                grouped[key][f"status:{row.get('status')}"] += 1
    out = []
    for key, counter in sorted(grouped.items()):
        segment, model, seed, role = key
        out.append(
            {
                "segment": segment,
                "victim_model": model,
                "attacker_seed": seed,
                "role": role,
                "files": files[key],
                **dict(counter),
            }
        )
    return out


def ablation_paired_tests(root: Path = HISTORICAL_REFERENCE_ROOT) -> dict[str, Any]:
    variants = {
        "conflict-collapse": "conflict-collapse",
        "gap-blind": "gap-blind",
        "no-lifted-join": "no-lifted-join",
    }
    full = _load_agentdojo_ablation_variant(root / "agentdojo" / "full" / "raw_runs")
    if not full:
        return {
            "schema_version": "obligate-ablation-paired-tests-v1",
            "available": False,
            "reason": "full AgentDojo ablation raw_runs not found",
            "root": str(root),
        }
    comparisons: dict[str, dict[str, Any]] = {}
    family_keys: list[str] = []
    metrics = {
        "attack_success": "lower_is_better",
        "user_task_success": "higher_is_better",
        "secure_utility": "higher_is_better",
    }
    index = 0
    for dirname, label in variants.items():
        variant = _load_agentdojo_ablation_variant(root / "agentdojo" / dirname / "raw_runs")
        if not variant:
            comparisons[f"{label}/unavailable"] = {
                "available": False,
                "reason": f"{dirname} raw_runs not found",
                "variant": label,
            }
            continue
        for metric, direction in metrics.items():
            key = f"agentdojo/{label}/{metric}"
            result = paired_binary_task_test(
                {case_id: bool(values[metric]) for case_id, values in full.items()},
                {case_id: bool(values[metric]) for case_id, values in variant.items()},
                iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
                seed=DEFAULT_BOOTSTRAP_SEED + 2_000 + index,
            )
            index += 1
            comparisons[key] = {
                "benchmark": "agentdojo",
                "baseline": "full",
                "variant": label,
                "metric": metric,
                "better_direction": direction,
                "comparison": "variant minus full",
                "family": "agentdojo_component_ablations",
                **result,
            }
            if result.get("available"):
                family_keys.append(key)
    correction = holm_adjust({key: float(comparisons[key]["p_raw"]) for key in family_keys}) if family_keys else {}
    for key, value in correction.items():
        comparisons[key].update(value)
    return {
        "schema_version": "obligate-ablation-paired-tests-v1",
        "available": bool(family_keys),
        "root": str(root),
        "baseline": "full",
        "multiplicity": "Holm correction across AgentDojo component ablation comparisons",
        "family": {"name": "agentdojo_component_ablations", "size": len(family_keys), "keys": family_keys},
        "comparisons": comparisons,
        "statistical_unit": "original_task/case_id",
    }


def _load_agentdojo_ablation_variant(raw_dir: Path) -> dict[str, dict[str, bool]]:
    if not raw_dir.exists():
        return {}
    out: dict[str, dict[str, bool]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        try:
            item = read_json(path)
            normalized = (item.get("normalized_cases") or [{}])[0]
            case_id = _agentdojo_case_id(item, normalized, path)
            if not case_id or case_id in out:
                continue
            attack_success = bool(normalized.get("final_injection_task_success")) or int(normalized.get("targeted_asr_contribution") or 0) > 0
            user_task_success = bool(normalized.get("final_user_task_success"))
            out[case_id] = {
                "attack_success": attack_success,
                "user_task_success": user_task_success,
                "secure_utility": user_task_success and not attack_success,
            }
        except Exception:  # noqa: BLE001
            continue
    return out


def _agentdojo_case_id(item: dict[str, Any], normalized: dict[str, Any], path: Path) -> str:
    for value in (normalized.get("case_id"), item.get("case_id")):
        if value not in {None, ""}:
            return str(value)
    run_name = str(item.get("run_name") or path.stem)
    for suffix in ("_obligate_strict", "_obligate", "_obligate_"):
        if suffix in run_name:
            return run_name.split(suffix, 1)[0]
    return run_name


def code_hashes() -> dict[str, Any]:
    files = []
    code_dir = Path(__file__).resolve().parent
    for path in sorted(code_dir.glob("*.py")) + [code_dir / "config.yaml"]:
        if path.exists():
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    for path in sorted((RESULT_ROOT / "manifests").glob("*.json")):
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    value = {"schema_version": "obligate-code-hashes-v2", "generated_at": utc_now(), "files": files}
    write_json(RESULT_ROOT / "code_hashes.json", value)
    return value


def validity_audit(rows: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    preflight_path = RESULT_ROOT / "safetybench_full_preflight.json"
    if not preflight_path.exists():
        preflight_path = RESULT_ROOT / "preflight.json"
    preflight = read_json(preflight_path) if preflight_path.exists() else {}
    sampling = expected.get("sampling_audit") or {}
    safetybench_full = expected.get("agent_safetybench_full") or {}
    safetybench_audit = expected.get("agent_safetybench_full_audit") or {}
    score_counts = Counter()
    for row in rows:
        score_counts[(str(row.get("segment")), str(row.get("victim_model")), str(row.get("attacker_seed")), int(row.get("round") or 0))] += 1
    score_count_items = [
        {"segment": key[0], "victim_model": key[1], "attacker_seed": key[2], "round": key[3], "rows": value}
        for key, value in sorted(score_counts.items())
    ]
    attacker_invalid = attacker_generation_summary()
    if safetybench_full:
        checks = {
            "safetybench_preflight_checks_passed": all((preflight.get("checks") or {}).values()),
            "safetybench_full_manifest_present": True,
            "safetybench_full_case_count_2000": int(safetybench_full.get("case_count") or 0) == 2000,
            "safetybench_full_public_inputs_only": bool(safetybench_audit.get("generated_from_public_inputs_only")),
            "safetybench_full_forbidden_fields_empty": not bool(safetybench_audit.get("forbidden_selection_fields_used")),
            "scores_exist": bool(rows),
        }
    else:
        checks = {
            "preflight_formal_run_allowed": bool(((preflight.get("checks") or {}).get("formal_run_allowed"))),
            "preflight_profile_hash_exact": bool(((preflight.get("checks") or {}).get("profile_hash_exact"))),
            "sampling_public_inputs_only": bool(sampling.get("generated_from_public_inputs_only")),
            "sampling_forbidden_fields_empty": not bool(sampling.get("forbidden_selection_fields_used")),
            "main_manifest_present": "main" in expected,
            "mechanism_stress_manifest_present": "mechanism_stress" in expected,
            "scores_exist": bool(rows),
        }
    value = {
        "schema_version": "obligate-validity-audit-v2",
        "generated_at": utc_now(),
        "checks": checks,
        "all_checks_passed_so_far": all(checks.values()),
        "score_row_counts": score_count_items,
        "attacker_generation": attacker_invalid,
        "notes": [
            "The original benchmark task/case_id is the inferential unit for adaptive multi-round attacks.",
            "Rounds, tool calls, and attacker seeds are repeated observations nested under the same task, not independent samples.",
            "Cumulative adaptive curves use task-cluster bootstrap; multi-seed runs use hierarchical task-cluster bootstrap with seeds nested inside tasks.",
            "Ablation comparisons use paired tests on identical task universes and Holm correction across component-ablation comparisons.",
            "ASB adapted protocol is diagnostic and is not reported as official Agent Security Bench.",
            "Agent-SafetyBench tool-level utility proxy is not official benchmark utility.",
            "Provider/workflow failures are retained in denominators and reported conservatively.",
            "Zero observed bypass is reported as zero observed in this run, not as an absolute safety proof.",
        ],
    }
    write_json(RESULT_ROOT / "validity_audit.json", value)
    return value


def _path_meta(path: Path) -> dict[str, Any]:
    parts = path.parts
    try:
        root_idx = parts.index("adaptive_closed_loop_v2")
    except ValueError:
        return {}
    segment = parts[root_idx + 1] if len(parts) > root_idx + 1 else None
    model = parts[root_idx + 2] if len(parts) > root_idx + 2 else None
    seed_part = parts[root_idx + 3] if len(parts) > root_idx + 3 else None
    seed = seed_part.replace("seed_", "") if isinstance(seed_part, str) and seed_part.startswith("seed_") else seed_part
    round_index = None
    for part in parts:
        if isinstance(part, str) and part.startswith("round_"):
            try:
                round_index = int(part.replace("round_", ""))
            except ValueError:
                round_index = None
    return {"segment": segment, "victim_model": model, "attacker_seed": seed, "round": round_index}


def _seed_meta(path: Path) -> dict[str, Any]:
    meta = _path_meta(path)
    meta.pop("round", None)
    return meta


def _round_number(name: str) -> int | None:
    stem = Path(name).stem
    for part in stem.split("_"):
        try:
            return int(part)
        except ValueError:
            continue
    return None


def build_summary() -> dict[str, Any]:
    rows = load_scores()
    expected = load_expected_counts()
    case_index = manifest_case_index()
    hashes = code_hashes()
    audit = validity_audit(rows, expected)
    ablations = ablation_paired_tests()
    summary = {
        "schema_version": "obligate-adaptive-closed-loop-summary-v2",
        "generated_at": utc_now(),
        "result_root": str(RESULT_ROOT),
        "statistical_methods": {
            "inferential_unit": "original benchmark task/case_id",
            "non_independent_repeated_observations": ["round", "tool_call", "attacker_seed"],
            "multi_round_primary_interval": "task-cluster bootstrap 95% CI",
            "multi_seed_primary_interval": "hierarchical task-cluster bootstrap with attacker seeds nested within task",
            "curve_confidence_band": "95% task-cluster bootstrap band over horizons 1..5",
            "ablation_test": "paired exact McNemar on identical task universes",
            "ablation_delta_interval": "paired task bootstrap 95% CI",
            "multiple_comparison_correction": "Holm correction within ablation comparison family",
            "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "bootstrap_samples_persisted": False,
        },
        "expected": expected,
        "code_hashes": hashes,
        "validity_audit": audit,
        "score_rows": len(rows),
        "overall_by_round": round_tables(rows),
        "by_benchmark_round": benchmark_round_tables(rows),
        "cumulative": cumulative_tables(rows),
        "adaptive_curves": adaptive_curve_tables(rows),
        "mechanism": mechanism_tables(rows, case_index),
        "ablation_paired_tests": ablations,
        "attacker_generation": attacker_generation_summary(),
        "stop_reasons": stop_reason_summary(),
        "usage": usage_summary(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "adaptive_closed_loop_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", help="print compact JSON after writing the report summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary()
    if args.print:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(REPORT_DIR / "adaptive_closed_loop_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
