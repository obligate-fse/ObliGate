"""Paired, case-level statistics for ObliGate Ablation v2.

The functions in this module deliberately do not pool victim models.  Every
public analysis entry point returns a mapping keyed by model and treats
``case_id`` as the resampling and hypothesis-testing unit.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import atomic_write_json, utc_now
from .fault_injection import MUTATION_NAMES

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260716
DEFAULT_MODELS = ("deepseek-v4-flash", "qwen-plus")
from .protocol import E2E_VARIANTS
DEFAULT_VARIANTS = E2E_VARIANTS[1:]
MECHANISM_VARIANTS = (
    "unbound-evidence",
    "conflict-to-support",
    "conflict-to-refute",
    "gap-blind",
    "single-remedy",
    "no-feasibility-filter",
    "no-rebind",
)
COUNTERFACTUAL_METRICS = (
    "unsafe_flip",
    "over_conservative_flip",
    "constraint_downgrade",
    "non_blocking_plan_lost",
)
INTEGRITY_METRIC = "tamper_or_unrealizable_acceptance"
CERTIFICATE_MUTATION_METRICS = ("acceptance", "silent_dispatch")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count_values(values: Iterable[str]) -> Counter[str]:
    return Counter(values)

# All rates are computed within case before case-level resampling.  This avoids
# treating a case with many tool decisions as many independent observations.
DEFAULT_METRICS: dict[str, bool] = {
    "targeted_asr_observed": True,
    "user_utility_observed": True,
    "secure_utility_observed": True,
    # Unqualified paper-primary metrics use the conservative denominator:
    # provider/workflow invalids count unsafe for ASR and unsuccessful for
    # both utility metrics.
    "targeted_asr": True,
    "user_utility": True,
    "secure_utility": True,
    "block_rate": False,
    "require_confirmation_rate": False,
    "execute_with_constraints_rate": False,
    "ordinary_allow_rate": False,
    "intervention_case_rate": True,
    "post_intervention_task_recovery": True,
    "workflow_provider_invalid": True,
    "average_tool_decisions": False,
}


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided Wilson 95% interval for a binomial proportion.

    An empty active set is a valid experimental result.  In that case the
    interval is represented as ``(nan, nan)`` instead of inventing a
    denominator or raising during report generation.
    """

    if total < 0:
        raise ValueError("total must be non-negative")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total]")
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def wilson95(successes: int, total: int) -> dict[str, Any]:
    """JSON-friendly Wilson summary, including an honest empty-set result."""

    low, high = wilson_interval(successes, total)
    return {
        "count": successes,
        "n": total,
        "rate": successes / total if total else None,
        "ci95": [low, high] if total else [None, None],
    }


def mcnemar_exact(baseline_only: int, variant_only: int) -> dict[str, Any]:
    """Two-sided exact McNemar test over discordant paired cases.

    ``baseline_only`` counts pairs ``(baseline=1, variant=0)`` and
    ``variant_only`` counts ``(baseline=0, variant=1)``.  The p-value is the
    exact two-sided binomial tail under ``p=0.5``.
    """

    if baseline_only < 0 or variant_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = baseline_only + variant_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, variant_only) + 1))
        p_value = min(1.0, 2.0 * (tail / (1 << discordant)))
    return {
        "baseline_only": baseline_only,
        "variant_only": variant_only,
        "discordant": discordant,
        "p_value": p_value,
        "method": "two-sided exact McNemar (binomial p=0.5)",
    }


def clustered_bootstrap_delta(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_field: str,
    variant_field: str,
    case_field: str = "case_id",
    active_field: str | None = None,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap ``Variant - Full`` while resampling whole cases.

    Multiple action rows belonging to one case are first reduced to the mean
    within-case delta.  Cases, rather than actions, are then sampled with
    replacement.  Rows missing either paired value are retained in the audit
    counts but excluded from the estimate.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    seen_cases: set[str] = set()
    active_cases: set[str] = set()
    for row in rows:
        case_id = str(row.get(case_field) or "")
        if not case_id:
            raise ValueError(f"row is missing {case_field!r}")
        seen_cases.add(case_id)
        if active_field is not None and not bool(row.get(active_field)):
            continue
        active_cases.add(case_id)
        baseline = row.get(baseline_field)
        variant = row.get(variant_field)
        if baseline is None or variant is None:
            continue
        grouped[case_id].append(float(variant) - float(baseline))

    case_deltas = {
        case_id: sum(values) / len(values)
        for case_id, values in grouped.items()
        if values
    }
    case_ids = sorted(case_deltas)
    missing_pair_cases = active_cases - set(case_deltas)
    if not case_ids:
        return {
            "available": False,
            "reason": "no paired active cases",
            "delta": None,
            "ci95": [None, None],
            "case_count": 0,
            "active_case_denominator": len(active_cases),
            "seen_case_count": len(seen_cases),
            "missing_pair_case_count": len(missing_pair_cases),
            "iterations": iterations,
            "seed": seed,
            "unit": "case_id",
            "row_count": 0,
        }

    point = sum(case_deltas.values()) / len(case_deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        chosen = [rng.choice(case_ids) for _ in case_ids]
        samples.append(sum(case_deltas[case_id] for case_id in chosen) / len(chosen))
    samples.sort()
    return {
        "available": True,
        "delta": point,
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
        "case_count": len(case_ids),
        "active_case_denominator": len(active_cases),
        "seen_case_count": len(seen_cases),
        "missing_pair_case_count": len(missing_pair_cases),
        "row_count": sum(len(values) for values in grouped.values()),
        "iterations": iterations,
        "seed": seed,
        "unit": "case_id",
    }


def paired_binary_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_field: str,
    variant_field: str,
    case_field: str = "case_id",
    active_field: str | None = None,
) -> dict[str, Any]:
    """Summarize paired binary flips and exact McNemar at case level.

    If a case contains multiple action snapshots, a binary outcome is reduced
    with ``any``.  This makes the estimand "did this case exhibit at least one
    such event" and prevents action-rich cases from receiving extra weight.
    """

    grouped: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"baseline": [], "variant": [], "active": []}
    )
    for row in rows:
        case_id = str(row.get(case_field) or "")
        if not case_id:
            raise ValueError(f"row is missing {case_field!r}")
        grouped[case_id]["active"].append(
            True if active_field is None else bool(row.get(active_field))
        )
        if row.get(baseline_field) is not None and row.get(variant_field) is not None:
            grouped[case_id]["baseline"].append(bool(row[baseline_field]))
            grouped[case_id]["variant"].append(bool(row[variant_field]))

    active_ids = {
        case_id for case_id, value in grouped.items() if any(value["active"])
    }
    paired: list[tuple[bool, bool]] = []
    missing = 0
    for case_id in sorted(active_ids):
        value = grouped[case_id]
        if not value["baseline"] or not value["variant"]:
            missing += 1
            continue
        paired.append((any(value["baseline"]), any(value["variant"])))

    both_zero = sum(not baseline and not variant for baseline, variant in paired)
    baseline_only = sum(baseline and not variant for baseline, variant in paired)
    variant_only = sum(not baseline and variant for baseline, variant in paired)
    both_one = sum(baseline and variant for baseline, variant in paired)
    mcnemar = mcnemar_exact(baseline_only, variant_only)
    baseline_count = baseline_only + both_one
    variant_count = variant_only + both_one
    denominator = len(paired)
    return {
        "active_case_denominator": len(active_ids),
        "paired_case_count": denominator,
        "missing_pair_case_count": missing,
        "paired_flips": {
            "both_zero": both_zero,
            "full_only": baseline_only,
            "variant_only": variant_only,
            "both_one": both_one,
        },
        "full": wilson95(baseline_count, denominator),
        "variant": wilson95(variant_count, denominator),
        "variant_minus_full": (
            (variant_count - baseline_count) / denominator if denominator else None
        ),
        "mcnemar_exact": mcnemar,
        "unit": "case_id",
    }


def _reduce_binary_rows_by_case(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_field: str,
    variant_field: str,
    active_field: str | None = None,
) -> list[dict[str, Any]]:
    """Reduce action/snapshot binary rows to one paired row per active case.

    Mechanism endpoints are defined as "did this case exhibit at least one
    event".  Reducing before both the point estimate *and* bootstrap keeps the
    estimand aligned with the exact McNemar table; otherwise the bootstrap
    would estimate a within-case action-average while the reported absolute
    rates used ``any``.
    """

    grouped: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {"baseline": [], "variant": [], "active": []}
    )
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("row is missing 'case_id'")
        active = True if active_field is None else bool(row.get(active_field))
        grouped[case_id]["active"].append(active)
        if not active:
            continue
        if row.get(baseline_field) is not None:
            grouped[case_id]["baseline"].append(bool(row[baseline_field]))
        if row.get(variant_field) is not None:
            grouped[case_id]["variant"].append(bool(row[variant_field]))

    reduced: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        value = grouped[case_id]
        if not any(value["active"]):
            continue
        reduced.append(
            {
                "case_id": case_id,
                "active": True,
                baseline_field: (
                    any(value["baseline"]) if value["baseline"] else None
                ),
                variant_field: (
                    any(value["variant"]) if value["variant"] else None
                ),
            }
        )
    return reduced


def holm_adjust(
    p_values: Mapping[str, float] | Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Holm step-down family-wise correction with monotone adjusted p-values."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    is_mapping = isinstance(p_values, Mapping)
    items = list(p_values.items()) if is_mapping else list(enumerate(p_values))
    for _, p_value in items:
        if not 0.0 <= float(p_value) <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(items, key=lambda item: (float(item[1]), str(item[0])))
    count = len(ordered)
    adjusted: dict[Any, dict[str, Any]] = {}
    running = 0.0
    still_rejecting = True
    for rank, (key, raw) in enumerate(ordered, start=1):
        raw_value = float(raw)
        running = max(running, min(1.0, (count - rank + 1) * raw_value))
        threshold = alpha / (count - rank + 1) if count else alpha
        reject = still_rejecting and raw_value <= threshold
        if not reject:
            still_rejecting = False
        adjusted[key] = {
            "raw_p": raw_value,
            "adjusted_p": running,
            "holm_rank": rank,
            "holm_threshold": threshold,
            "reject": reject,
        }
    if is_mapping:
        return {str(key): adjusted[key] for key, _ in items}
    return [adjusted[index] for index, _ in items]


def analyze_paired_models(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_fields: Mapping[str, tuple[str, str]],
    binary_metrics: Iterable[str] = (),
    model_field: str = "model",
    case_field: str = "case_id",
    active_fields: Mapping[str, str] | None = None,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze paired metrics separately for every victim model.

    ``metric_fields`` maps a report metric name to
    ``(full_field, variant_field)``.  Binary metrics additionally receive
    paired flip counts, exact McNemar tests, and within-model Holm correction.
    """

    rows_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get(model_field) or "")
        if not model:
            raise ValueError(f"row is missing {model_field!r}; cross-model pooling is forbidden")
        rows_by_model[model].append(row)

    binary = set(binary_metrics)
    result: dict[str, Any] = {
        "schema_version": "obligate-ablation-v2-paired-statistics-v1",
        "model_pooling": False,
        "models": {},
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }
    for model in sorted(rows_by_model):
        model_rows = rows_by_model[model]
        metrics: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for index, (name, (full_field, variant_field)) in enumerate(metric_fields.items()):
            active_field = (active_fields or {}).get(name)
            metric_seed = seed + index
            value: dict[str, Any] = {
                "bootstrap": clustered_bootstrap_delta(
                    model_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    case_field=case_field,
                    active_field=active_field,
                    iterations=iterations,
                    seed=metric_seed,
                )
            }
            if name in binary:
                paired = paired_binary_summary(
                    model_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    case_field=case_field,
                    active_field=active_field,
                )
                value["paired_binary"] = paired
                raw_p[name] = float(paired["mcnemar_exact"]["p_value"])
            metrics[name] = value
        adjusted = holm_adjust(raw_p) if raw_p else {}
        for name, correction in adjusted.items():
            metrics[name]["holm"] = correction
        result["models"][model] = {
            "case_count": len({str(row[case_field]) for row in model_rows}),
            "metrics": metrics,
            "holm_family_size": len(raw_p),
        }
    return result


def write_statistics_outputs(
    output_root: str | Path,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Read E2E per-case rows and write the four required statistics files.

    Holm correction uses one family per victim model over every binary
    ``variant/metric`` comparison.  No inferential result pools models.
    """

    root = Path(output_root)
    paired_document = _document("obligate-ablation-v2-paired-metrics-v1", iterations, seed)
    bootstrap_document = _document("obligate-ablation-v2-bootstrap-v1", iterations, seed)
    mcnemar_document = _document("obligate-ablation-v2-mcnemar-v1", iterations, seed)
    holm_document = _document("obligate-ablation-v2-holm-v1", iterations, seed)
    model_p_values: dict[str, dict[str, float]] = defaultdict(dict)

    for model in models:
        full_rows = _load_per_case(root / "e2e" / model / "full" / "per_case_results.jsonl")
        full_by_case = {str(row["case_id"]): row for row in full_rows}
        paired_document["models"][model] = {"variants": {}}
        bootstrap_document["models"][model] = {"variants": {}}
        mcnemar_document["models"][model] = {"variants": {}}
        for variant in variants:
            variant_rows = _load_per_case(
                root / "e2e" / model / variant / "per_case_results.jsonl"
            )
            variant_by_case = {str(row["case_id"]): row for row in variant_rows}
            case_ids = sorted(set(full_by_case) | set(variant_by_case))
            paired_rows = [
                _paired_case_row(
                    model,
                    case_id,
                    full_by_case.get(case_id),
                    variant_by_case.get(case_id),
                )
                for case_id in case_ids
            ]
            paired_metrics: dict[str, Any] = {}
            bootstrap_metrics: dict[str, Any] = {}
            mcnemar_metrics: dict[str, Any] = {}
            for metric, binary in DEFAULT_METRICS.items():
                full_field = f"full_{metric}"
                variant_field = f"variant_{metric}"
                active_field = f"active_{metric}"
                bootstrap = clustered_bootstrap_delta(
                    paired_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field=active_field,
                    iterations=iterations,
                    seed=seed,
                )
                bootstrap_metrics[metric] = bootstrap
                if binary:
                    paired = paired_binary_summary(
                        paired_rows,
                        baseline_field=full_field,
                        variant_field=variant_field,
                        active_field=active_field,
                    )
                    paired_metrics[metric] = {
                        "full_absolute": paired["full"],
                        "variant_absolute": paired["variant"],
                        "variant_minus_full": paired["variant_minus_full"],
                        "ci95": bootstrap["ci95"],
                        "active_case_denominator": paired["active_case_denominator"],
                        "paired_case_count": paired["paired_case_count"],
                        "missing_pair_case_count": paired["missing_pair_case_count"],
                        "paired_flips": paired["paired_flips"],
                    }
                    exact = {
                        **paired["mcnemar_exact"],
                        "active_case_denominator": paired["active_case_denominator"],
                        "paired_case_count": paired["paired_case_count"],
                        "paired_flips": paired["paired_flips"],
                    }
                    mcnemar_metrics[metric] = exact
                    model_p_values[model][f"{variant}/{metric}"] = float(exact["p_value"])
                else:
                    full_values = _case_values(paired_rows, full_field, active_field)
                    variant_values = _case_values(paired_rows, variant_field, active_field)
                    paired_metrics[metric] = {
                        "full_absolute": _mean_summary(full_values),
                        "variant_absolute": _mean_summary(variant_values),
                        "variant_minus_full": bootstrap["delta"],
                        "ci95": bootstrap["ci95"],
                        "active_case_denominator": bootstrap["active_case_denominator"],
                        "paired_case_count": bootstrap["case_count"],
                        "missing_pair_case_count": bootstrap["missing_pair_case_count"],
                        "paired_flips": None,
                    }
            paired_document["models"][model]["variants"][variant] = {
                "case_count": len(case_ids),
                "metrics": paired_metrics,
            }
            bootstrap_document["models"][model]["variants"][variant] = {
                "case_count": len(case_ids),
                "metrics": bootstrap_metrics,
            }
            mcnemar_document["models"][model]["variants"][variant] = {
                "case_count": len(case_ids),
                "metrics": mcnemar_metrics,
            }

    mechanism_rows = _mechanism_rows(root, models=models)
    for document in (paired_document, bootstrap_document, mcnemar_document):
        document["mechanism"] = {
            "statistical_unit": "case_id",
            "action_rows_reduced_within_case": "any",
            "models": {},
        }
    for model in models:
        for document in (paired_document, bootstrap_document, mcnemar_document):
            document["mechanism"]["models"][model] = {"variants": {}}
        for variant in MECHANISM_VARIANTS:
            rows = mechanism_rows[model][variant]
            metric_names = (
                COUNTERFACTUAL_METRICS
                if variant not in {"no-feasibility-filter", "no-rebind"}
                else (INTEGRITY_METRIC,)
            )
            paired_metrics: dict[str, Any] = {}
            bootstrap_metrics: dict[str, Any] = {}
            mcnemar_metrics: dict[str, Any] = {}
            for metric_index, metric in enumerate(metric_names):
                full_field = f"full_{metric}"
                variant_field = f"variant_{metric}"
                reduced_rows = _reduce_binary_rows_by_case(
                    rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field="active",
                )
                bootstrap = clustered_bootstrap_delta(
                    reduced_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field="active",
                    iterations=iterations,
                    seed=seed + 10_000 + metric_index,
                )
                paired = paired_binary_summary(
                    reduced_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field="active",
                )
                paired_metrics[metric] = {
                    "full_absolute": paired["full"],
                    "variant_absolute": paired["variant"],
                    "variant_minus_full": paired["variant_minus_full"],
                    "ci95": bootstrap["ci95"],
                    "active_case_denominator": paired["active_case_denominator"],
                    "paired_case_count": paired["paired_case_count"],
                    "missing_pair_case_count": paired["missing_pair_case_count"],
                    "paired_flips": paired["paired_flips"],
                    "action_row_count": len(rows),
                }
                bootstrap_metrics[metric] = bootstrap
                exact = {
                    **paired["mcnemar_exact"],
                    "active_case_denominator": paired["active_case_denominator"],
                    "paired_case_count": paired["paired_case_count"],
                    "paired_flips": paired["paired_flips"],
                }
                mcnemar_metrics[metric] = exact
                if paired["paired_case_count"]:
                    model_p_values[model][f"mechanism/{variant}/{metric}"] = float(
                        exact["p_value"]
                    )
            cell = {
                "active_case_count": len({str(row["case_id"]) for row in rows}),
                "action_row_count": len(rows),
            }
            paired_document["mechanism"]["models"][model]["variants"][variant] = {
                **cell,
                "metrics": paired_metrics,
            }
            bootstrap_document["mechanism"]["models"][model]["variants"][variant] = {
                **cell,
                "metrics": bootstrap_metrics,
            }
            mcnemar_document["mechanism"]["models"][model]["variants"][variant] = {
                **cell,
                "metrics": mcnemar_metrics,
            }

    certificate_rows = _load_jsonl_if_present(
        root / "fault_injection" / "certificate_mutation_results.jsonl"
    )
    invalid_path = root / "fault_injection" / "certificate_mutation_invalid.jsonl"
    certificate_invalid_rows = _load_jsonl_if_present(invalid_path)
    invalid_source_available = invalid_path.is_file()
    unknown_mutations = sorted(
        {
            str(row.get("mutation") or "unknown")
            for row in certificate_rows
            if str(row.get("mutation") or "unknown") not in MUTATION_NAMES
        }
    )
    invalid_disclosure = {
        "path": str(invalid_path.resolve()),
        "source_available": invalid_source_available,
        "excluded_snapshot_count": len(certificate_invalid_rows),
        "excluded_case_count": len(
            {str(row.get("case_id") or "") for row in certificate_invalid_rows}
        ),
        "by_model": {
            model: {
                "excluded_snapshot_count": sum(
                    str(row.get("model") or "") == model
                    for row in certificate_invalid_rows
                ),
                "excluded_case_count": len(
                    {
                        str(row.get("case_id") or "")
                        for row in certificate_invalid_rows
                        if str(row.get("model") or "") == model
                    }
                ),
            }
            for model in models
        },
        "reason_counts": dict(
            sorted(
                _count_values(
                    str(row.get("error_type") or row.get("reason") or "unknown")
                    for row in certificate_invalid_rows
                ).items()
            )
        ),
    }
    for document in (paired_document, bootstrap_document, mcnemar_document):
        document["certificate_mutations"] = {
            "statistical_unit": "case_id",
            "snapshot_rows_reduced_within_case": "any",
            "applicability_rule": (
                "eligible includes every recorded mutation row; inferential active cases "
                "contain at least one applicable row; N/A-only cases are excluded from "
                "acceptance and silent-dispatch inference and disclosed separately"
            ),
            "expected_mutations": list(MUTATION_NAMES),
            "unknown_mutations": unknown_mutations,
            "invalid_snapshot_exclusions": invalid_disclosure,
            "models": {},
        }

    for model in models:
        model_rows = [
            row for row in certificate_rows if str(row.get("model") or "") == model
        ]
        for document in (paired_document, bootstrap_document, mcnemar_document):
            document["certificate_mutations"]["models"][model] = {"mutations": {}}
        for mutation_index, mutation in enumerate(MUTATION_NAMES):
            eligible = [
                row
                for row in model_rows
                if str(row.get("mutation") or "") == mutation
            ]
            applicable = [row for row in eligible if row.get("applicable") is True]
            not_applicable = [row for row in eligible if row.get("applicable") is not True]
            eligible_cases = {str(row.get("case_id") or "") for row in eligible}
            applicable_cases = {str(row.get("case_id") or "") for row in applicable}
            not_applicable_cases = {
                str(row.get("case_id") or "") for row in not_applicable
            }
            rows = []
            for row in applicable:
                full_gate = _as_mapping(row.get("full_toolgate"))
                no_rebind = _as_mapping(row.get("no_rebind"))
                rows.append(
                    {
                        "model": model,
                        "case_id": str(row.get("case_id") or ""),
                        "active": True,
                        "full_acceptance": bool(full_gate.get("accepted")),
                        "variant_acceptance": bool(no_rebind.get("accepted")),
                        "full_silent_dispatch": bool(
                            int(full_gate.get("simulated_dispatch_count") or 0)
                        ),
                        "variant_silent_dispatch": bool(
                            no_rebind.get("silent_dispatch")
                        ),
                    }
                )

            paired_metrics: dict[str, Any] = {}
            bootstrap_metrics: dict[str, Any] = {}
            mcnemar_metrics: dict[str, Any] = {}
            for metric_index, metric in enumerate(CERTIFICATE_MUTATION_METRICS):
                full_field = f"full_{metric}"
                variant_field = f"variant_{metric}"
                reduced_rows = _reduce_binary_rows_by_case(
                    rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field="active",
                )
                bootstrap = clustered_bootstrap_delta(
                    reduced_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field="active",
                    iterations=iterations,
                    seed=seed + 20_000 + mutation_index * 10 + metric_index,
                )
                paired = paired_binary_summary(
                    reduced_rows,
                    baseline_field=full_field,
                    variant_field=variant_field,
                    active_field="active",
                )
                paired_metrics[metric] = {
                    "full_absolute": paired["full"],
                    "variant_absolute": paired["variant"],
                    "variant_minus_full": paired["variant_minus_full"],
                    "ci95": bootstrap["ci95"],
                    "active_case_denominator": paired["active_case_denominator"],
                    "paired_case_count": paired["paired_case_count"],
                    "missing_pair_case_count": paired["missing_pair_case_count"],
                    "paired_flips": paired["paired_flips"],
                }
                bootstrap_metrics[metric] = bootstrap
                exact = {
                    **paired["mcnemar_exact"],
                    "active_case_denominator": paired["active_case_denominator"],
                    "paired_case_count": paired["paired_case_count"],
                    "paired_flips": paired["paired_flips"],
                }
                mcnemar_metrics[metric] = exact
                if paired["paired_case_count"]:
                    model_p_values[model][
                        f"certificate_mutation/{mutation}/{metric}"
                    ] = float(exact["p_value"])

            disclosure = {
                "eligible_snapshot_count": len(eligible),
                "eligible_case_count": len(eligible_cases),
                "applicable_snapshot_count": len(applicable),
                "applicable_case_count": len(applicable_cases),
                "not_applicable_snapshot_count": len(not_applicable),
                "not_applicable_case_count": len(not_applicable_cases),
                "not_applicable_only_case_count": len(
                    eligible_cases - applicable_cases
                ),
            }
            paired_document["certificate_mutations"]["models"][model]["mutations"][
                mutation
            ] = {**disclosure, "metrics": paired_metrics}
            bootstrap_document["certificate_mutations"]["models"][model][
                "mutations"
            ][mutation] = {**disclosure, "metrics": bootstrap_metrics}
            mcnemar_document["certificate_mutations"]["models"][model][
                "mutations"
            ][mutation] = {**disclosure, "metrics": mcnemar_metrics}

    for model in models:
        raw = model_p_values.get(model, {})
        holm_document["models"][model] = {
            "family_size": len(raw),
            "family_definition": "all binary variant/metric comparisons within one victim model",
            "hypotheses": holm_adjust(raw),
        }

    statistics_dir = root / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_metrics": paired_document,
        "bootstrap_results": bootstrap_document,
        "mcnemar_results": mcnemar_document,
        "holm_adjusted": holm_document,
    }
    for name, value in outputs.items():
        atomic_write_json(statistics_dir / f"{name}.json", value)
    return outputs


def _mechanism_rows(
    root: Path,
    *,
    models: Sequence[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {variant: [] for variant in MECHANISM_VARIANTS} for model in models
    }

    for row in _load_jsonl_if_present(root / "snapshots" / "counterfactual_results.jsonl"):
        model = str(row.get("model") or "")
        variant = str(row.get("variant") or "")
        case_id = str(row.get("case_id") or "")
        if model not in result or variant not in result[model] or not case_id:
            continue
        converted: dict[str, Any] = {
            "model": model,
            "case_id": case_id,
            "active": True,
        }
        for metric in COUNTERFACTUAL_METRICS:
            converted[f"full_{metric}"] = False
            converted[f"variant_{metric}"] = bool(row.get(metric))
        result[model][variant].append(converted)

    for row in _load_jsonl_if_present(root / "fault_injection" / "feasibility_results.jsonl"):
        model = str(row.get("model") or "")
        case_id = str(row.get("case_id") or "")
        if model not in result or not case_id:
            continue
        no_feas = row.get("no_feasibility_filter")
        no_feas = no_feas if isinstance(no_feas, Mapping) else {}
        result[model]["no-feasibility-filter"].append(
            {
                "model": model,
                "case_id": case_id,
                "active": True,
                f"full_{INTEGRITY_METRIC}": bool(row.get("full_selected_unrealizable")),
                f"variant_{INTEGRITY_METRIC}": bool(
                    no_feas.get("unrealizable_plan_selected")
                    or no_feas.get("attempted_dispatch_under_unavailable_capability")
                ),
            }
        )

    for row in _load_jsonl_if_present(
        root / "fault_injection" / "certificate_mutation_results.jsonl"
    ):
        model = str(row.get("model") or "")
        case_id = str(row.get("case_id") or "")
        if model not in result or not case_id or row.get("applicable") is False:
            continue
        full_gate = row.get("full_toolgate")
        full_gate = full_gate if isinstance(full_gate, Mapping) else {}
        no_rebind = row.get("no_rebind")
        no_rebind = no_rebind if isinstance(no_rebind, Mapping) else {}
        result[model]["no-rebind"].append(
            {
                "model": model,
                "case_id": case_id,
                "active": True,
                f"full_{INTEGRITY_METRIC}": bool(full_gate.get("accepted")),
                f"variant_{INTEGRITY_METRIC}": bool(no_rebind.get("accepted")),
            }
        )
    return result


def _load_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSONL object at {path}:{line_number}")
            rows.append(value)
    return rows


def _document(schema_version: str, iterations: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "created_at": utc_now(),
        "model_pooling": False,
        "statistical_unit": "case_id",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "models": {},
    }


def _load_per_case(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("case_id"):
                raise ValueError(f"invalid per-case row at {path}:{line_number}")
            rows.append(value)
    return rows


def _paired_case_row(
    model: str,
    case_id: str,
    full: Mapping[str, Any] | None,
    variant: Mapping[str, Any] | None,
) -> dict[str, Any]:
    full_metrics = _derive_metrics(full)
    variant_metrics = _derive_metrics(variant)
    value: dict[str, Any] = {"model": model, "case_id": case_id}
    for metric in DEFAULT_METRICS:
        value[f"full_{metric}"] = full_metrics.get(metric)
        value[f"variant_{metric}"] = variant_metrics.get(metric)
        if metric.endswith("_observed"):
            value[f"active_{metric}"] = bool(
                full_metrics.get(metric) is not None
                and variant_metrics.get(metric) is not None
            )
        elif metric == "post_intervention_task_recovery":
            value[f"active_{metric}"] = bool(
                full_metrics.get("intervention_case_rate")
                and variant_metrics.get("intervention_case_rate")
                and full_metrics.get(metric) is not None
                and variant_metrics.get(metric) is not None
            )
        else:
            value[f"active_{metric}"] = True
    return value


def _derive_metrics(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {metric: None for metric in DEFAULT_METRICS}
    decisions = int(row.get("tool_decision_count") or 0)

    def decision_rate(field: str) -> float:
        return float(row.get(field) or 0) / decisions if decisions else 0.0

    return {
        "targeted_asr_observed": _optional_bool(row.get("targeted_asr_observed")),
        "user_utility_observed": _optional_bool(row.get("user_utility_observed")),
        "secure_utility_observed": _optional_bool(row.get("secure_utility_observed")),
        "targeted_asr": _optional_bool(row.get("targeted_asr_conservative")),
        "user_utility": _optional_bool(row.get("user_utility_conservative")),
        "secure_utility": _optional_bool(row.get("secure_utility_conservative")),
        "block_rate": decision_rate("block_count"),
        "require_confirmation_rate": decision_rate("require_confirmation_count"),
        "execute_with_constraints_rate": decision_rate("execute_with_constraints_count"),
        "ordinary_allow_rate": decision_rate("allow_count"),
        "intervention_case_rate": bool(row.get("intervention_case")),
        "post_intervention_task_recovery": _optional_bool(
            row.get("post_intervention_task_recovery")
        ),
        "workflow_provider_invalid": bool(row.get("invalid")),
        "average_tool_decisions": float(decisions),
    }


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _case_values(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    active_field: str,
) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if bool(row.get(active_field)) and row.get(field) is not None
    ]


def _mean_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if probability <= 0:
        return float(sorted_values[0])
    if probability >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


# Explicit aliases used by report/pipeline code and reviewers' terminology.
exact_mcnemar = mcnemar_exact
holm_bonferroni = holm_adjust
analyze_model_variant = analyze_paired_models


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_METRICS",
    "DEFAULT_MODELS",
    "DEFAULT_VARIANTS",
    "analyze_model_variant",
    "analyze_paired_models",
    "clustered_bootstrap_delta",
    "exact_mcnemar",
    "holm_adjust",
    "holm_bonferroni",
    "mcnemar_exact",
    "paired_binary_summary",
    "wilson95",
    "wilson_interval",
    "write_statistics_outputs",
]
