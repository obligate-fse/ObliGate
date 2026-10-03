"""Build paper-facing statistical tables for adaptive and ablation experiments.

This command consumes an existing adaptive_closed_loop_summary.json. If the
summary was produced by the updated task-clustered summarizer, the dossier uses
task-cluster bootstrap intervals and paired ablation tests. If only an older
aggregate-only summary is available, the dossier keeps those legacy aggregate
counts but marks them as descriptive rather than inferential.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import RESULT_ROOT, read_json, utc_now, write_json

DEFAULT_SUMMARY = RESULT_ROOT / "reports" / "adaptive_closed_loop_summary.json"
DEFAULT_OUT_DIR = RESULT_ROOT / "reports" / "statistical_dossier"
DEFAULT_ABLATION_ROOT = RESULT_ROOT.parent / "obligate_ablation_v2"


def build_dossier(
    summary: Mapping[str, Any], *, obligate_ablation_root: Path | None = DEFAULT_ABLATION_ROOT
) -> dict[str, Any]:
    has_clustered = bool(summary.get("adaptive_curves"))
    methods = summary.get("statistical_methods") or _legacy_methods()
    adaptive_curves = _adaptive_curve_rows(summary)
    cumulative = _cumulative_rows(summary, has_clustered=has_clustered)
    round_rows = _round_rows(summary)
    seed_rows = _seed_range_rows(summary)
    ablations = _ablation_rows(summary, obligate_ablation_root=obligate_ablation_root)
    has_ablation_tests = bool(ablations)
    availability = {
        "raw_task_level_scores_available_in_summary": has_clustered,
        "adaptive_task_cluster_bootstrap_available": has_clustered,
        "ablation_paired_tests_available": has_ablation_tests,
        "legacy_aggregate_only": not has_clustered,
        "obligate_ablation_root": str(obligate_ablation_root) if obligate_ablation_root else None,
        "notes": [],
    }
    if not has_clustered:
        availability["notes"].append(
            "Current summary lacks per-case task-level adaptive_curves; legacy Wilson-style aggregate intervals are descriptive only."
        )
        availability["notes"].append(
            "Regenerate summary from raw scores.jsonl with the updated summarizer before making significance claims."
        )
    if not has_ablation_tests:
        availability["notes"].append(
            "Ablation paired tests are unavailable until matching full/variant statistical outputs are present."
        )
    return {
        "schema_version": "obligate-statistical-dossier-v1",
        "generated_at": utc_now(),
        "source_summary_generated_at": summary.get("generated_at"),
        "source_score_rows": summary.get("score_rows"),
        "availability": availability,
        "methods": methods,
        "adaptive_curves": adaptive_curves,
        "cumulative": cumulative,
        "round_descriptives": round_rows,
        "seed_ranges": seed_rows,
        "ablation_paired_tests": ablations,
    }


def write_dossier(dossier: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "statistical_dossier.json", dossier)
    _write_csv(out_dir / "adaptive_curves.csv", _adaptive_curve_csv_rows(dossier))
    _write_csv(out_dir / "cumulative_task_level.csv", dossier.get("cumulative") or [])
    _write_csv(out_dir / "round_descriptives.csv", dossier.get("round_descriptives") or [])
    _write_csv(out_dir / "seed_ranges.csv", dossier.get("seed_ranges") or [])
    _write_csv(out_dir / "ablation_paired_tests.csv", dossier.get("ablation_paired_tests") or [])
    (out_dir / "statistical_dossier.md").write_text(render_markdown(dossier), encoding="utf-8", newline="\n")


def _adaptive_curve_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if summary.get("adaptive_curves"):
        for row in summary.get("adaptive_curves") or []:
            casr = row.get("CASR") or {}
            observed = row.get("observed_bypass_rate") or {}
            dangerous = row.get("dangerous_candidate_rate") or {}
            rows.append(
                {
                    "segment": row.get("segment"),
                    "victim_model": row.get("victim_model"),
                    "benchmark": row.get("benchmark"),
                    "horizon": row.get("horizon"),
                    "tasks": row.get("tasks"),
                    "casr_rate": casr.get("rate"),
                    "casr_ci95_low": _ci_low(casr),
                    "casr_ci95_high": _ci_high(casr),
                    "observed_bypass_rate": observed.get("rate"),
                    "observed_bypass_ci95_low": _ci_low(observed),
                    "observed_bypass_ci95_high": _ci_high(observed),
                    "dangerous_candidate_rate": dangerous.get("rate"),
                    "dangerous_candidate_ci95_low": _ci_low(dangerous),
                    "dangerous_candidate_ci95_high": _ci_high(dangerous),
                    "attacker_seed_count": ((row.get("seed_rate_range") or {}).get("attacker_seed_count")),
                    "seed_min_rate": ((row.get("seed_rate_range") or {}).get("min_rate")),
                    "seed_max_rate": ((row.get("seed_rate_range") or {}).get("max_rate")),
                    "interval": "task-cluster bootstrap 95%",
                    "inferential": True,
                }
            )
        return rows
    for row in summary.get("cumulative") or []:
        if row.get("benchmark") != "ALL" or row.get("segment") not in {"main", "agent_safetybench_full"}:
            continue
        casr = row.get("CASR") or {}
        observed = row.get("observed_bypass_rate") or {}
        dangerous = row.get("dangerous_candidate_rate") or {}
        rows.append(
            {
                "segment": row.get("segment"),
                "victim_model": row.get("victim_model"),
                "benchmark": row.get("benchmark"),
                "horizon": row.get("horizon"),
                "tasks": row.get("cases"),
                "casr_rate": casr.get("rate"),
                "casr_ci95_low": _ci_low(casr),
                "casr_ci95_high": _ci_high(casr),
                "observed_bypass_rate": observed.get("rate"),
                "observed_bypass_ci95_low": _ci_low(observed),
                "observed_bypass_ci95_high": _ci_high(observed),
                "dangerous_candidate_rate": dangerous.get("rate"),
                "dangerous_candidate_ci95_low": _ci_low(dangerous),
                "dangerous_candidate_ci95_high": _ci_high(dangerous),
                "attacker_seed_count": 1,
                "seed_min_rate": None,
                "seed_max_rate": None,
                "interval": "legacy aggregate Wilson/descriptive",
                "inferential": False,
            }
        )
    return rows


def _cumulative_rows(summary: Mapping[str, Any], *, has_clustered: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("cumulative") or []:
        casr = row.get("CASR") or {}
        observed = row.get("observed_bypass_rate") or {}
        dangerous = row.get("dangerous_candidate_rate") or {}
        rows.append(
            {
                "segment": row.get("segment"),
                "victim_model": row.get("victim_model"),
                "attacker_seed": row.get("attacker_seed"),
                "benchmark": row.get("benchmark"),
                "horizon": row.get("horizon"),
                "tasks": row.get("cases"),
                "cumulative_attack_success": row.get("cumulative_attack_success"),
                "casr_rate": casr.get("rate"),
                "casr_ci95_low": _ci_low(casr),
                "casr_ci95_high": _ci_high(casr),
                "cumulative_observed_bypass": row.get("cumulative_observed_bypass"),
                "observed_bypass_rate": observed.get("rate"),
                "observed_bypass_ci95_low": _ci_low(observed),
                "observed_bypass_ci95_high": _ci_high(observed),
                "dangerous_candidate_cases": row.get("dangerous_candidate_cases"),
                "dangerous_candidate_rate": dangerous.get("rate"),
                "dangerous_candidate_ci95_low": _ci_low(dangerous),
                "dangerous_candidate_ci95_high": _ci_high(dangerous),
                "median_round_to_success": row.get("median_round_to_success"),
                "interval": "task-cluster bootstrap 95%" if has_clustered else "legacy aggregate Wilson/descriptive",
                "inferential": has_clustered,
            }
        )
    return rows


def _round_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("overall_by_round") or []:
        rows.append(
            {
                "segment": row.get("segment"),
                "victim_model": row.get("victim_model"),
                "attacker_seed": row.get("attacker_seed"),
                "round": row.get("round"),
                "rows": row.get("cases"),
                "attack_success": row.get("attack_success"),
                "attack_success_observed": row.get("attack_success_observed"),
                "dangerous_candidate": row.get("dangerous_candidate"),
                "provider_valid": row.get("provider_valid"),
                "workflow_failure": row.get("workflow_failure"),
                "decision_counts": json.dumps(row.get("decision_counts") or {}, ensure_ascii=False, sort_keys=True),
                "inferential": False,
                "note": "round rows are descriptive repeated observations, not independent samples",
            }
        )
    return rows


def _seed_range_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("adaptive_curves") or []:
        seed_range = row.get("seed_rate_range") or {}
        if not seed_range.get("available"):
            continue
        rows.append(
            {
                "segment": row.get("segment"),
                "victim_model": row.get("victim_model"),
                "benchmark": row.get("benchmark"),
                "horizon": row.get("horizon"),
                "attacker_seed_count": seed_range.get("attacker_seed_count"),
                "attacker_seeds": ",".join(str(item) for item in seed_range.get("attacker_seeds") or []),
                "min_rate": seed_range.get("min_rate"),
                "max_rate": seed_range.get("max_rate"),
                "rates_by_seed": json.dumps(seed_range.get("rates_by_seed") or {}, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _ablation_rows(
    summary: Mapping[str, Any], *, obligate_ablation_root: Path | None = DEFAULT_ABLATION_ROOT
) -> list[dict[str, Any]]:
    tests = (summary.get("ablation_paired_tests") or {}).get("comparisons") or {}
    rows: list[dict[str, Any]] = []
    for key, row in sorted(tests.items()):
        if not row.get("available"):
            rows.append({"key": key, "available": False, "reason": row.get("reason")})
            continue
        rows.append(
            {
                "key": key,
                "available": True,
                "benchmark": row.get("benchmark"),
                "baseline": row.get("baseline"),
                "variant": row.get("variant"),
                "metric": row.get("metric"),
                "paired_tasks": row.get("paired_tasks"),
                "baseline_successes": row.get("baseline_successes"),
                "variant_successes": row.get("variant_successes"),
                "variant_minus_baseline": row.get("variant_minus_baseline"),
                "delta_ci95_low": (row.get("variant_minus_baseline_ci95") or [None, None])[0],
                "delta_ci95_high": (row.get("variant_minus_baseline_ci95") or [None, None])[1],
                "p_raw": row.get("p_raw"),
                "p_holm": row.get("p_holm"),
                "significant_at_0.05": row.get("significant_at_0.05"),
                "paired_table": json.dumps(row.get("paired_table") or {}, ensure_ascii=False, sort_keys=True),
                "test": row.get("test"),
                "family": row.get("family"),
            }
        )
    rows.extend(_obligate_ablation_rows(obligate_ablation_root))
    return rows


def _obligate_ablation_rows(root: Path | None) -> list[dict[str, Any]]:
    if root is None:
        return []
    stats_dir = root / "statistics"
    paired = _read_optional_json(stats_dir / "paired_metrics.json")
    bootstrap = _read_optional_json(stats_dir / "bootstrap_results.json")
    mcnemar = _read_optional_json(stats_dir / "mcnemar_results.json")
    holm = _read_optional_json(stats_dir / "holm_adjusted.json")
    if not paired:
        return []

    rows: list[dict[str, Any]] = []
    for model, model_doc in sorted(((paired.get("models") or {}).items())):
        variants = (model_doc or {}).get("variants") or {}
        for variant, variant_doc in sorted(variants.items()):
            paired_metrics = (variant_doc or {}).get("metrics") or {}
            for metric, paired_metric in sorted(paired_metrics.items()):
                boot_metric = _nested_metric(bootstrap, model, variant, metric)
                mcnemar_metric = _nested_metric(mcnemar, model, variant, metric)
                holm_metric = (((holm.get("models") or {}).get(model) or {}).get("hypotheses") or {}).get(
                    f"{variant}/{metric}"
                )
                full_absolute = (paired_metric or {}).get("full_absolute") or {}
                variant_absolute = (paired_metric or {}).get("variant_absolute") or {}
                delta = _first_present(
                    (paired_metric or {}).get("variant_minus_full"),
                    (boot_metric or {}).get("delta"),
                    ((holm_metric or {}).get("variant_minus_full") if isinstance(holm_metric, Mapping) else None),
                )
                ci95 = _first_present(
                    (paired_metric or {}).get("ci95"),
                    (boot_metric or {}).get("ci95"),
                    ((holm_metric or {}).get("ci95") if isinstance(holm_metric, Mapping) else None),
                )
                rows.append(
                    {
                        "key": f"obligate_ablation_v2/{model}/{variant}/{metric}",
                        "source": "obligate_ablation_v2",
                        "available": True,
                        "benchmark": "AgentDojo",
                        "model": model,
                        "baseline": "full",
                        "variant": variant,
                        "metric": metric,
                        "paired_tasks": _first_present(
                            (paired_metric or {}).get("paired_case_count"),
                            (boot_metric or {}).get("case_count"),
                            (mcnemar_metric or {}).get("paired_case_count"),
                        ),
                        "active_case_denominator": _first_present(
                            (paired_metric or {}).get("active_case_denominator"),
                            (boot_metric or {}).get("active_case_denominator"),
                            (mcnemar_metric or {}).get("active_case_denominator"),
                        ),
                        "missing_pair_case_count": _first_present(
                            (paired_metric or {}).get("missing_pair_case_count"),
                            (boot_metric or {}).get("missing_pair_case_count"),
                        ),
                        "baseline_successes": full_absolute.get("count"),
                        "baseline_total": full_absolute.get("n"),
                        "baseline_rate": full_absolute.get("rate"),
                        "baseline_ci95_low": _ci_low(full_absolute),
                        "baseline_ci95_high": _ci_high(full_absolute),
                        "variant_successes": variant_absolute.get("count"),
                        "variant_total": variant_absolute.get("n"),
                        "variant_rate": variant_absolute.get("rate"),
                        "variant_ci95_low": _ci_low(variant_absolute),
                        "variant_ci95_high": _ci_high(variant_absolute),
                        "variant_minus_baseline": delta,
                        "delta_ci95_low": (ci95 or [None, None])[0],
                        "delta_ci95_high": (ci95 or [None, None])[1],
                        "p_raw": _first_present(
                            (mcnemar_metric or {}).get("p_value"),
                            ((holm_metric or {}).get("raw_p") if isinstance(holm_metric, Mapping) else None),
                        ),
                        "p_holm": ((holm_metric or {}).get("adjusted_p") if isinstance(holm_metric, Mapping) else None),
                        "significant_at_0.05": (
                            (holm_metric or {}).get("reject") if isinstance(holm_metric, Mapping) else None
                        ),
                        "holm_rank": ((holm_metric or {}).get("holm_rank") if isinstance(holm_metric, Mapping) else None),
                        "holm_threshold": (
                            (holm_metric or {}).get("holm_threshold") if isinstance(holm_metric, Mapping) else None
                        ),
                        "paired_table": json.dumps(
                            _first_present(
                                (paired_metric or {}).get("paired_flips"),
                                (mcnemar_metric or {}).get("paired_flips"),
                            )
                            or {},
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "test": (
                            (mcnemar_metric or {}).get("method")
                            if mcnemar_metric
                            else "case-clustered bootstrap delta only"
                        ),
                        "family": ((holm.get("models") or {}).get(model) or {}).get("family_definition"),
                        "bootstrap_iterations": (boot_metric or {}).get("iterations"),
                        "bootstrap_seed": (boot_metric or {}).get("seed"),
                    }
                )
    return rows


def _nested_metric(document: Mapping[str, Any] | None, model: str, variant: str, metric: str) -> Mapping[str, Any]:
    if not document:
        return {}
    return (
        (((document.get("models") or {}).get(model) or {}).get("variants") or {})
        .get(variant, {})
        .get("metrics", {})
        .get(metric, {})
    )


def _read_optional_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _adaptive_curve_csv_rows(dossier: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(dossier.get("adaptive_curves") or [])


def render_markdown(dossier: Mapping[str, Any]) -> str:
    availability = dossier.get("availability") or {}
    methods = dossier.get("methods") or {}
    lines = [
        "# Adaptive Attack and Ablation Statistical Dossier",
        "",
        f"Generated at: `{dossier.get('generated_at')}`",
        f"Source summary generated at: `{dossier.get('source_summary_generated_at')}`",
        f"Source score rows: `{dossier.get('source_score_rows')}`",
        "",
        "## Inference Unit",
        "",
        "- Primary unit: original benchmark task / `case_id`.",
        "- Rounds, tool calls, and attacker seeds are repeated observations nested under the same task.",
        "- Adaptive curves should use task-cluster bootstrap; multi-seed runs use hierarchical task-cluster bootstrap.",
        "- Ablations use paired tests on identical task universes and Holm correction across component comparisons.",
        "",
        "## Availability",
        "",
        f"- Task-cluster bootstrap available: `{availability.get('adaptive_task_cluster_bootstrap_available')}`",
        f"- Ablation paired tests available: `{availability.get('ablation_paired_tests_available')}`",
        f"- Legacy aggregate only: `{availability.get('legacy_aggregate_only')}`",
    ]
    for note in availability.get("notes") or []:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Methods",
            "",
            f"- Multi-round interval: `{methods.get('multi_round_primary_interval') or methods.get('multi_round_interval')}`",
            f"- Multi-seed interval: `{methods.get('multi_seed_primary_interval') or methods.get('multi_seed_interval')}`",
            f"- Ablation test: `{methods.get('ablation_test')}`",
            f"- Multiple comparisons: `{methods.get('multiple_comparison_correction')}`",
            "",
            "## Adaptive Curves",
            "",
            _md_table(
                ["segment", "model", "benchmark", "horizon", "tasks", "CASR 95%", "observed 95%", "dangerous 95%", "inferential"],
                [
                    [
                        row.get("segment"),
                        row.get("victim_model"),
                        row.get("benchmark"),
                        row.get("horizon"),
                        row.get("tasks"),
                        _fmt_ci(row.get("casr_rate"), row.get("casr_ci95_low"), row.get("casr_ci95_high")),
                        _fmt_ci(row.get("observed_bypass_rate"), row.get("observed_bypass_ci95_low"), row.get("observed_bypass_ci95_high")),
                        _fmt_ci(row.get("dangerous_candidate_rate"), row.get("dangerous_candidate_ci95_low"), row.get("dangerous_candidate_ci95_high")),
                        row.get("inferential"),
                    ]
                    for row in (dossier.get("adaptive_curves") or [])
                    if row.get("benchmark") == "ALL"
                ],
            ),
            "",
            "## Ablation Paired Tests",
            "",
            _md_table(
                ["key", "tasks", "delta", "delta 95%", "p raw", "p Holm", "sig"],
                [
                    [
                        row.get("key"),
                        row.get("paired_tasks"),
                        _fmt_pct(row.get("variant_minus_baseline")),
                        _fmt_ci(row.get("variant_minus_baseline"), row.get("delta_ci95_low"), row.get("delta_ci95_high")),
                        _fmt_p(row.get("p_raw")),
                        _fmt_p(row.get("p_holm")),
                        row.get("significant_at_0.05"),
                    ]
                    for row in (dossier.get("ablation_paired_tests") or [])
                    if row.get("available")
                    and row.get("metric") in {"targeted_asr", "user_utility", "secure_utility"}
                ],
            ),
            "",
            "## Output Files",
            "",
            "- `statistical_dossier.json`: canonical machine-readable summary.",
            "- `adaptive_curves.csv`: curve points and confidence bands.",
            "- `cumulative_task_level.csv`: cumulative task-level metrics.",
            "- `round_descriptives.csv`: per-round descriptive rows only.",
            "- `seed_ranges.csv`: seed range table for multi-seed runs.",
            "- `ablation_paired_tests.csv`: paired tests with Holm correction.",
        ]
    )
    return "\n".join(lines) + "\n"


def _legacy_methods() -> dict[str, Any]:
    return {
        "inferential_unit": "original benchmark task/case_id",
        "non_independent_repeated_observations": ["round", "tool_call", "attacker_seed"],
        "multi_round_interval": "unavailable from aggregate-only legacy summary",
        "multi_seed_interval": "unavailable from aggregate-only legacy summary",
        "ablation_test": "paired exact McNemar on identical task universes when raw variants are present",
        "multiple_comparison_correction": "Holm correction within ablation comparison family",
    }


def _ci_low(metric: Mapping[str, Any]) -> Any:
    if "lo" in metric:
        return metric.get("lo")
    return (metric.get("ci95") or [None, None])[0]


def _ci_high(metric: Mapping[str, Any]) -> Any:
    if "hi" in metric:
        return metric.get("hi")
    return (metric.get("ci95") or [None, None])[1]


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def _fmt_ci(rate: Any, low: Any, high: Any) -> str:
    if rate is None:
        return "NA"
    if low is None or high is None:
        return _fmt_pct(rate)
    return f"{_fmt_pct(rate)} [{_fmt_pct(low)}, {_fmt_pct(high)}]"


def _fmt_p(value: Any) -> str:
    if value is None:
        return "NA"
    number = float(value)
    return "<0.0001" if number < 0.0001 else f"{number:.4f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No available rows._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = read_json(args.summary)
    dossier = build_dossier(summary, obligate_ablation_root=args.ablation_root)
    write_dossier(dossier, args.out_dir)
    print(args.out_dir / "statistical_dossier.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
