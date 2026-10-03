"""Task-clustered inference for adaptive closed-loop attack runs.

The experimental unit is the original benchmark task. Rounds, tool calls, and
attacker seeds are repeated observations nested under that task, not independent
samples.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260716


def hierarchical_task_cluster_bootstrap_rate(
    task_seed_values: Mapping[str, Mapping[str, bool]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Estimate a rate with tasks as the outer cluster and seeds nested within.

    Point estimate: mean over tasks of the task's mean across observed attacker
    seeds. Bootstrap: resample tasks with replacement, then resample attacker
    seeds within each sampled task. With one seed this reduces to an ordinary
    task-level cluster bootstrap.
    """

    grouped = {
        str(task_id): {str(seed_id): bool(value) for seed_id, value in seeds.items()}
        for task_id, seeds in task_seed_values.items()
        if seeds
    }
    task_ids = sorted(grouped)
    if not task_ids:
        return {"available": False, "reason": "no task-level outcomes"}

    def task_mean(task_id: str, seed_ids: list[str] | None = None) -> float:
        seeds = seed_ids if seed_ids is not None else sorted(grouped[task_id])
        return sum(float(grouped[task_id][seed_id]) for seed_id in seeds) / len(seeds)

    def estimate(selected_tasks: list[str], rng: random.Random | None = None) -> float:
        values = []
        for task_id in selected_tasks:
            seed_ids = sorted(grouped[task_id])
            if rng is not None:
                seed_ids = [rng.choice(seed_ids) for _ in seed_ids]
            values.append(task_mean(task_id, seed_ids))
        return sum(values) / len(values)

    point = estimate(task_ids)
    rng = random.Random(seed)
    samples = sorted(estimate([rng.choice(task_ids) for _ in task_ids], rng) for _ in range(iterations))
    seed_values = sorted({seed_id for seeds in grouped.values() for seed_id in seeds})
    return {
        "available": True,
        "rate": point,
        "lo": _quantile(samples, 0.025),
        "hi": _quantile(samples, 0.975),
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
        "task_count": len(task_ids),
        "attacker_seed_count": len(seed_values),
        "attacker_seeds": seed_values,
        "task_seed_observation_count": sum(len(seeds) for seeds in grouped.values()),
        "iterations": iterations,
        "seed": seed,
        "unit": "original_task/case_id",
        "method": "hierarchical task-cluster bootstrap; attacker seeds nested within task",
        "bootstrap_samples_persisted": False,
    }


def seed_rate_range(task_seed_values: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    grouped = {
        str(task_id): {str(seed_id): bool(value) for seed_id, value in seeds.items()}
        for task_id, seeds in task_seed_values.items()
        if seeds
    }
    by_seed: dict[str, list[bool]] = {}
    for seeds in grouped.values():
        for seed_id, value in seeds.items():
            by_seed.setdefault(seed_id, []).append(value)
    if not by_seed:
        return {"available": False, "reason": "no attacker seed outcomes"}
    seed_rates = {
        seed_id: {
            "tasks": len(values),
            "successes": sum(bool(value) for value in values),
            "rate": sum(bool(value) for value in values) / len(values),
        }
        for seed_id, values in sorted(by_seed.items())
    }
    rates = [float(item["rate"]) for item in seed_rates.values()]
    return {
        "available": True,
        "attacker_seed_count": len(seed_rates),
        "attacker_seeds": sorted(seed_rates),
        "rates_by_seed": seed_rates,
        "min_rate": min(rates),
        "max_rate": max(rates),
        "range": [min(rates), max(rates)],
    }


def paired_binary_task_test(
    baseline: Mapping[str, bool],
    variant: Mapping[str, bool],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if set(baseline) != set(variant):
        return {
            "available": False,
            "reason": "paired task universes differ",
            "baseline_only_tasks": len(set(baseline) - set(variant)),
            "variant_only_tasks": len(set(variant) - set(baseline)),
        }
    task_ids = sorted(str(item) for item in baseline)
    if not task_ids:
        return {"available": False, "reason": "no paired tasks"}
    both_zero = sum(not baseline[task_id] and not variant[task_id] for task_id in task_ids)
    baseline_only = sum(baseline[task_id] and not variant[task_id] for task_id in task_ids)
    variant_only = sum(not baseline[task_id] and variant[task_id] for task_id in task_ids)
    both_one = sum(baseline[task_id] and variant[task_id] for task_id in task_ids)
    n = len(task_ids)
    baseline_successes = baseline_only + both_one
    variant_successes = variant_only + both_one
    delta = (variant_successes - baseline_successes) / n
    rng = random.Random(seed)
    deltas = []
    per_task_delta = [float(variant[task_id]) - float(baseline[task_id]) for task_id in task_ids]
    for _ in range(iterations):
        selected = [rng.choice(per_task_delta) for _ in task_ids]
        deltas.append(sum(selected) / len(selected))
    deltas.sort()
    return {
        "available": True,
        "paired_tasks": n,
        "baseline_successes": baseline_successes,
        "variant_successes": variant_successes,
        "variant_minus_baseline": delta,
        "variant_minus_baseline_ci95": [_quantile(deltas, 0.025), _quantile(deltas, 0.975)],
        "paired_table": {
            "both_zero": both_zero,
            "baseline_only": baseline_only,
            "variant_only": variant_only,
            "both_one": both_one,
        },
        **mcnemar_exact(baseline_only, variant_only),
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "unit": "original_task/case_id",
            "method": "paired nonparametric task bootstrap",
            "samples_persisted": False,
        },
    }


def mcnemar_exact(left_only: int, right_only: int) -> dict[str, Any]:
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (1 << discordant))
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "p_raw": p_value,
        "test": "two-sided exact McNemar on paired tasks",
    }


def holm_adjust(values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    ordered = sorted(values.items(), key=lambda item: (float(item[1]), item[0]))
    running = 0.0
    still_rejecting = True
    adjusted: dict[str, dict[str, Any]] = {}
    for rank, (key, raw) in enumerate(ordered, start=1):
        remaining = len(ordered) - rank + 1
        running = max(running, min(1.0, remaining * float(raw)))
        threshold = alpha / remaining
        reject = still_rejecting and float(raw) <= threshold
        if not reject:
            still_rejecting = False
        adjusted[key] = {
            "p_holm": running,
            "holm_rank": rank,
            "holm_threshold": threshold,
            "significant_at_0.05": reject,
        }
    return adjusted


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
