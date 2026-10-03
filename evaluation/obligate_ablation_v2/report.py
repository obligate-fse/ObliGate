"""Generate bilingual, evidence-backed reports for ObliGate Ablation v2.

All values come from frozen result artefacts below ``output_root``.  Missing or
malformed inputs are rendered as ``N/A`` / ``不完整``; this module never fills
gaps with estimates or historical experiment values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .common import utc_now
from .fault_injection import MUTATION_NAMES
from .validity import (
    E2E_VARIANTS,
    MODELS,
    RESEARCH_VARIANTS,
    build_validity_audit,
    write_validity_audit,
)

MODEL_SHORT = {"deepseek-v4-flash": "DeepSeek", "qwen-plus": "Qwen"}
METRICS = (
    "targeted_asr_observed",
    "user_utility_observed",
    "secure_utility_observed",
    "targeted_asr",
    "user_utility",
    "secure_utility",
    "block_rate",
    "require_confirmation_rate",
    "execute_with_constraints_rate",
    "ordinary_allow_rate",
    "intervention_case_rate",
    "post_intervention_task_recovery",
    "workflow_provider_invalid",
    "average_tool_decisions",
)
METRIC_NAMES_ZH = {
    "targeted_asr_observed": "Targeted ASR（observed）",
    "user_utility_observed": "User Utility（observed）",
    "secure_utility_observed": "Secure Utility（observed）",
    "targeted_asr": "Targeted ASR（conservative）",
    "user_utility": "User Utility（conservative）",
    "secure_utility": "Secure Utility（conservative）",
    "block_rate": "Block rate",
    "require_confirmation_rate": "Confirmation rate",
    "execute_with_constraints_rate": "Constrained-execution rate",
    "ordinary_allow_rate": "Ordinary-allow rate",
    "intervention_case_rate": "Intervention-case rate",
    "post_intervention_task_recovery": "Post-intervention recovery",
    "workflow_provider_invalid": "Workflow/provider invalid",
    "average_tool_decisions": "Average tool decisions",
}
METRIC_NAMES_EN = {key: value.replace("（", " (").replace("）", ")") for key, value in METRIC_NAMES_ZH.items()}
SNAPSHOT_VARIANTS = (
    "unbound-evidence",
    "conflict-to-support",
    "conflict-to-refute",
    "gap-blind",
    "single-remedy",
)
TABLE_B_VARIANTS = (*SNAPSHOT_VARIANTS, "no-feasibility-filter", "no-rebind")


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        return None
                    rows.append(value)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return rows


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    last_error: OSError | None = None
    for attempt in range(80):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    if last_error is not None:
        raise last_error


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cell(e2e: Any, model: str, variant: str) -> Mapping[str, Any]:
    cells = _mapping(_mapping(e2e).get("cells"))
    value = cells.get(f"{model}/{variant}")
    if value is None:
        value = _mapping(cells.get(model)).get(variant)
    return _mapping(value)


def _metric_summary(cell: Mapping[str, Any], scope: str, metric: str) -> Mapping[str, Any]:
    return _mapping(_mapping(cell.get(scope)).get(metric))


def _stats_metric(document: Any, model: str, variant: str, metric: str) -> Mapping[str, Any]:
    model_value = _mapping(_mapping(_mapping(document).get("models")).get(model))
    variant_value = _mapping(_mapping(model_value.get("variants")).get(variant))
    return _mapping(_mapping(variant_value.get("metrics")).get(metric))


def _holm_metric(document: Any, model: str, variant: str, metric: str) -> Mapping[str, Any]:
    model_value = _mapping(_mapping(_mapping(document).get("models")).get(model))
    return _mapping(_mapping(model_value.get("hypotheses")).get(f"{variant}/{metric}"))


def _mechanism_stats_metric(
    document: Any, model: str, variant: str, metric: str
) -> Mapping[str, Any]:
    mechanism = _mapping(_mapping(document).get("mechanism"))
    model_value = _mapping(_mapping(mechanism.get("models")).get(model))
    variant_value = _mapping(_mapping(model_value.get("variants")).get(variant))
    return _mapping(_mapping(variant_value.get("metrics")).get(metric))


def _mechanism_holm_metric(
    document: Any, model: str, variant: str, metric: str
) -> Mapping[str, Any]:
    model_value = _mapping(_mapping(_mapping(document).get("models")).get(model))
    return _mapping(
        _mapping(model_value.get("hypotheses")).get(
            f"mechanism/{variant}/{metric}"
        )
    )


def _certificate_mutation_cell(
    document: Any, model: str, mutation: str
) -> Mapping[str, Any]:
    certificate = _mapping(_mapping(document).get("certificate_mutations"))
    model_value = _mapping(_mapping(certificate.get("models")).get(model))
    return _mapping(_mapping(model_value.get("mutations")).get(mutation))


def _certificate_mutation_metric(
    document: Any, model: str, mutation: str, metric: str
) -> Mapping[str, Any]:
    cell = _certificate_mutation_cell(document, model, mutation)
    return _mapping(_mapping(cell.get("metrics")).get(metric))


def _certificate_mutation_holm_metric(
    document: Any, model: str, mutation: str, metric: str
) -> Mapping[str, Any]:
    model_value = _mapping(_mapping(_mapping(document).get("models")).get(model))
    return _mapping(
        _mapping(model_value.get("hypotheses")).get(
            f"certificate_mutation/{mutation}/{metric}"
        )
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_percent(value: Any, *, signed: bool = False, digits: int = 2) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number * 100:.{digits}f}%"


def _fmt_number(value: Any, *, signed: bool = False, digits: int = 3) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}f}"


def _fmt_ci(value: Any, *, percent: bool = True) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "N/A"
    if percent:
        return f"[{_fmt_percent(value[0])}, {_fmt_percent(value[1])}]"
    return f"[{_fmt_number(value[0])}, {_fmt_number(value[1])}]"


def _fmt_p(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    return "<0.001" if number < 0.001 else f"{number:.3f}"


def _wilson95(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _fmt_case_rate(successes: int, total: int) -> str:
    if total <= 0:
        return "N/A (0 active cases)"
    return f"{successes}/{total} ({_fmt_percent(successes / total)}; {_fmt_ci(_wilson95(successes, total))})"


def _fmt_absolute(value: Any, metric: str) -> str:
    item = _mapping(value)
    rate = item.get("rate") if "rate" in item else item.get("mean")
    n = _integer(item.get("n"))
    if rate is None:
        return "N/A"
    displayed = _fmt_number(rate) if metric == "average_tool_decisions" else _fmt_percent(rate)
    return f"{displayed} (n={n})" if n is not None else displayed


def _fmt_summary(value: Mapping[str, Any]) -> str:
    rate = value.get("rate")
    n = _integer(value.get("n"))
    count = _integer(value.get("count"))
    ci = value.get("wilson95") or value.get("ci95")
    if rate is None:
        return "N/A"
    tail = f"{count}/{n}" if count is not None and n is not None else (f"n={n}" if n is not None else "")
    return f"{_fmt_percent(rate)} {_fmt_ci(ci)} {tail}".strip()


def _fmt_delta(value: Mapping[str, Any], metric: str) -> str:
    delta = value.get("variant_minus_full")
    if delta is None:
        delta = value.get("delta")
    if metric == "average_tool_decisions":
        return f"{_fmt_number(delta, signed=True)} {_fmt_ci(value.get('ci95'), percent=False)}"
    return f"{_fmt_percent(delta, signed=True)} {_fmt_ci(value.get('ci95'))}"


def _fmt_flips(value: Any) -> str:
    flips = _mapping(value)
    if not flips:
        return "N/A"
    return f"F-only={flips.get('full_only', 'N/A')}; V-only={flips.get('variant_only', 'N/A')}; both1={flips.get('both_one', 'N/A')}"


def _normalise_counterfactuals(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if rows is None:
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        variant = row.get("variant") or row.get("ablation")
        if variant:
            result.append(dict(row))
            continue
        for container_name in ("counterfactuals", "variants", "results"):
            container = row.get(container_name)
            if not isinstance(container, Mapping):
                continue
            for name, value in container.items():
                if isinstance(value, Mapping):
                    result.append({**row, **value, "variant": str(name)})
            break
    return result


def _flag(row: Mapping[str, Any], name: str) -> bool:
    aliases = {
        "unsafe_flip": ("unsafe_flip", "dangerous_flip"),
        "over_conservative_flip": ("over_conservative_flip", "overconservative_flip"),
        "constraint_downgrade": ("constraint_downgrade", "constraint_downgrade_flip"),
        "non_blocking_plan_lost": (
            "non_blocking_plan_lost",
            "non_blocking_plan_loss",
            "nonblock_plan_lost",
        ),
    }
    for alias in aliases[name]:
        if alias in row:
            return bool(row.get(alias))
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping) and alias in metrics:
            return bool(metrics.get(alias))
    return False


def _counterfactual_summary(rows: list[dict[str, Any]], *, source_available: bool) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODELS:
        for variant in SNAPSHOT_VARIANTS:
            group = [row for row in rows if str(row.get("model")) == model and str(row.get("variant") or row.get("ablation")) == variant]
            unsafe_cases = {str(row.get("case_id")) for row in group if _flag(row, "unsafe_flip")}
            over_cases = {str(row.get("case_id")) for row in group if _flag(row, "over_conservative_flip")}
            downgrade_cases = {str(row.get("case_id")) for row in group if _flag(row, "constraint_downgrade")}
            plan_loss_cases = {str(row.get("case_id")) for row in group if _flag(row, "non_blocking_plan_lost")}
            result[model][variant] = {
                "available": source_available,
                "snapshot_count": len(group),
                "active_cases": len({str(row.get("case_id")) for row in group}),
                "unsafe_flips": sum(_flag(row, "unsafe_flip") for row in group),
                "unsafe_flip_cases": len(unsafe_cases),
                "over_conservative_flips": sum(_flag(row, "over_conservative_flip") for row in group),
                "over_conservative_flip_cases": len(over_cases),
                "constraint_downgrades": sum(_flag(row, "constraint_downgrade") for row in group),
                "constraint_downgrade_cases": len(downgrade_cases),
                "non_blocking_plans_lost": sum(_flag(row, "non_blocking_plan_lost") for row in group),
                "non_blocking_plan_loss_cases": len(plan_loss_cases),
                "first_decision_divergences": sum(bool(row.get("first_decision_divergence")) for row in group),
                "unbound_evidence_misuse_records": sum(bool(row.get("unbound_evidence_misuse")) for row in group),
            }
    return result


def _fault_summaries(
    feasibility: list[dict[str, Any]] | None,
    certificate: list[dict[str, Any]] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODELS:
        feas = [row for row in (feasibility or []) if str(row.get("model")) == model]
        cert = [row for row in (certificate or []) if str(row.get("model")) == model]
        applicable_cert = [row for row in cert if row.get("applicable") is True]
        result[model]["no-feasibility-filter"] = {
            "available": feasibility is not None,
            "snapshot_count": len(feas),
            "active_cases": len({str(row.get("case_id")) for row in feas}),
            "acceptance": sum(bool(_mapping(row.get("no_feasibility_filter")).get("unrealizable_plan_selected")) for row in feas),
            "full_acceptance": sum(bool(row.get("full_selected_unrealizable")) for row in feas),
        }
        result[model]["no-rebind"] = {
            "available": certificate is not None,
            "snapshot_count": len(cert),
            "applicable_snapshot_count": len(applicable_cert),
            "not_applicable_snapshot_count": len(cert) - len(applicable_cert),
            "active_cases": len(
                {str(row.get("case_id")) for row in applicable_cert}
            ),
            "eligible_cases": len({str(row.get("case_id")) for row in cert}),
            "acceptance": sum(
                bool(_mapping(row.get("no_rebind")).get("accepted"))
                for row in applicable_cert
            ),
            "silent_dispatch": sum(
                bool(_mapping(row.get("no_rebind")).get("silent_dispatch"))
                for row in applicable_cert
            ),
            "full_acceptance": sum(
                bool(_mapping(row.get("full_toolgate")).get("accepted"))
                for row in applicable_cert
            ),
        }
    return result


def _activation_lines(snapshot_manifest: Any, language: str) -> list[str]:
    files = _mapping(_mapping(snapshot_manifest).get("files"))
    labels = (
        ("relation", "relation-active"),
        ("conflict", "conflict-active"),
        ("pure_gap", "pure-gap-active"),
        ("multi", "multi-obligation-active"),
        ("constrained", "constrained-realization-active"),
    )
    lines = [
        "| Activation set | Snapshots | Active cases | Status |",
        "|---|---:|---:|---|",
    ]
    for key, label in labels:
        value = _mapping(files.get(key))
        count = value.get("snapshot_count")
        cases = value.get("unique_case_count")
        if count is None:
            status = "不完整 / incomplete" if language == "zh" else "incomplete"
        elif count == 0:
            status = "真实空集（未造样本）" if language == "zh" else "observed empty set (not fabricated)"
        else:
            status = "已观测" if language == "zh" else "observed"
        lines.append(f"| {label} | {count if count is not None else 'N/A'} | {cases if cases is not None else 'N/A'} | {status} |")
    return lines


def _table_a_lines(paired: Any) -> list[str]:
    lines = [
        "| Variant | DS ΔASR | DS ΔUser Utility | DS ΔSecure Utility | DS ΔBlock | DS ΔConfirm | Qwen ΔASR | Qwen ΔUser Utility | Qwen ΔSecure Utility | Qwen ΔBlock | Qwen ΔConfirm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = (
        "targeted_asr",
        "user_utility",
        "secure_utility",
        "block_rate",
        "require_confirmation_rate",
    )
    for variant in RESEARCH_VARIANTS:
        cells = [variant]
        for model in MODELS:
            for metric in metrics:
                cells.append(_fmt_delta(_stats_metric(paired, model, variant, metric), metric))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _table_b_lines(
    counterfactual: dict[str, dict[str, dict[str, Any]]],
    faults: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    lines = [
        "| Model | Ablation | Active cases | Unsafe flips | Over-conservative flips | Constraint downgrades | Non-blocking plans lost | Tamper / unrealizable acceptance |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for variant in TABLE_B_VARIANTS:
            if variant in SNAPSHOT_VARIANTS:
                item = counterfactual[model][variant]
                if not item["available"]:
                    values = ["N/A"] * 6
                else:
                    values = [
                        str(item["active_cases"]),
                        str(item["unsafe_flips"]),
                        str(item["over_conservative_flips"]),
                        str(item["constraint_downgrades"]),
                        str(item["non_blocking_plans_lost"]),
                        "—",
                    ]
            else:
                item = faults[model][variant]
                if not item["available"]:
                    values = ["N/A"] * 6
                else:
                    values = [str(item["active_cases"]), "—", "—", "—", "—", str(item["acceptance"])]
            lines.append(f"| {MODEL_SHORT[model]} | {variant} | " + " | ".join(values) + " |")
    return lines


def _mechanism_case_rate_lines(counterfactual: dict[str, dict[str, dict[str, Any]]], language: str) -> list[str]:
    lines = [
        "### Case-clustered mechanism rates" if language == "en" else "### 以 case_id 聚类的机制发生率",
        "",
        "| Model | Variant | Active cases | Unsafe flip cases (Wilson 95% CI) | Over-conservative cases (Wilson 95% CI) | Constraint-downgrade cases (Wilson 95% CI) | Non-blocking-plan-loss cases (Wilson 95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for variant in SNAPSHOT_VARIANTS:
            item = counterfactual[model][variant]
            if not item["available"]:
                values = ["N/A"] * 5
            else:
                active = int(item["active_cases"])
                values = [
                    str(active),
                    _fmt_case_rate(int(item["unsafe_flip_cases"]), active),
                    _fmt_case_rate(int(item["over_conservative_flip_cases"]), active),
                    _fmt_case_rate(int(item["constraint_downgrade_cases"]), active),
                    _fmt_case_rate(int(item["non_blocking_plan_loss_cases"]), active),
                ]
            lines.append(f"| {MODEL_SHORT[model]} | {variant} | " + " | ".join(values) + " |")
    lines.append(
        "\n"
        + (
            "同一 case 中多个快照先按 any 聚合，再计算 Wilson 区间，避免动作丰富的任务获得额外权重。active n=0 显示 N/A，而不是伪装成 0% 效应。"
            if language == "zh"
            else "Multiple snapshots within a case are reduced with any before Wilson intervals, preventing action-rich cases from receiving extra weight. active n=0 is N/A, not a fabricated 0% effect."
        )
    )
    return lines


def _mechanism_inference_lines(paired: Any, holm: Any, language: str) -> list[str]:
    zh = language == "zh"
    variants = (
        "unbound-evidence",
        "conflict-to-support",
        "conflict-to-refute",
        "gap-blind",
        "single-remedy",
        "no-feasibility-filter",
        "no-rebind",
    )
    flip_metrics = (
        "unsafe_flip",
        "over_conservative_flip",
        "constraint_downgrade",
        "non_blocking_plan_lost",
    )
    labels_zh = {
        "unsafe_flip": "危险放行 case",
        "over_conservative_flip": "过度保守 case",
        "constraint_downgrade": "约束降级 case",
        "non_blocking_plan_lost": "非阻断方案丢失 case",
        "tamper_or_unrealizable_acceptance": "篡改/不可实现接受 case",
    }
    labels_en = {
        "unsafe_flip": "unsafe-flip case",
        "over_conservative_flip": "over-conservative case",
        "constraint_downgrade": "constraint-downgrade case",
        "non_blocking_plan_lost": "non-blocking-plan-loss case",
        "tamper_or_unrealizable_acceptance": "tamper/unrealizable-acceptance case",
    }
    labels = labels_zh if zh else labels_en
    lines = [
        "### 机制级配对推断（case_id 聚类）" if zh else "### Mechanism-level paired inference (clustered by case_id)",
        "",
        "| Model | Variant | Endpoint | Full absolute | Variant absolute | Variant − Full (10k bootstrap 95% CI) | exact McNemar Holm-adjusted p | Paired flips | Active n |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for model in MODELS:
        for variant in variants:
            metrics = (
                ("tamper_or_unrealizable_acceptance",)
                if variant in {"no-feasibility-filter", "no-rebind"}
                else flip_metrics
            )
            for metric in metrics:
                value = _mechanism_stats_metric(paired, model, variant, metric)
                adjusted = _mechanism_holm_metric(holm, model, variant, metric)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            MODEL_SHORT[model],
                            variant,
                            labels[metric],
                            _fmt_absolute(value.get("full_absolute"), metric),
                            _fmt_absolute(value.get("variant_absolute"), metric),
                            _fmt_delta(value, metric),
                            _fmt_p(adjusted.get("adjusted_p")),
                            _fmt_flips(value.get("paired_flips")),
                            str(value.get("active_case_denominator", "N/A")),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            (
                "每个 case 内的多个动作先以 any 聚合；因此 CI、McNemar 与 Holm 校正均不会让动作较多的任务获得额外权重。active n=0 时效应和 p-value 显示 N/A。"
                if zh
                else "Multiple actions are reduced with any within each case; bootstrap CIs, McNemar tests, and Holm correction therefore do not overweight action-rich tasks. Effects and p-values are N/A when active n=0."
            ),
        ]
    )
    return lines


def _certificate_mutation_inference_lines(
    paired: Any, holm: Any, language: str
) -> list[str]:
    """Render all 13 mutation-specific, case-clustered integrity endpoints."""

    zh = language == "zh"
    certificate = _mapping(_mapping(paired).get("certificate_mutations"))
    invalid = _mapping(certificate.get("invalid_snapshot_exclusions"))
    lines = [
        (
            "### 13 类证书变异：适用性分母与接受率"
            if zh
            else "### Thirteen certificate mutations: applicability and acceptance"
        ),
        "",
        (
            "| Model | Mutation | Eligible snapshots / cases | Applicable snapshots / cases | N/A snapshots / N/A-only cases | Full acceptance | no-rebind acceptance | no-rebind − Full (10k bootstrap 95% CI) | Holm-adjusted p | Paired flips |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        for mutation in MUTATION_NAMES:
            cell = _certificate_mutation_cell(paired, model, mutation)
            metric = _certificate_mutation_metric(
                paired, model, mutation, "acceptance"
            )
            adjusted = _certificate_mutation_holm_metric(
                holm, model, mutation, "acceptance"
            )
            eligible = (
                f"{cell.get('eligible_snapshot_count', 'N/A')} / "
                f"{cell.get('eligible_case_count', 'N/A')}"
            )
            applicable = (
                f"{cell.get('applicable_snapshot_count', 'N/A')} / "
                f"{cell.get('applicable_case_count', 'N/A')}"
            )
            not_applicable = (
                f"{cell.get('not_applicable_snapshot_count', 'N/A')} / "
                f"{cell.get('not_applicable_only_case_count', 'N/A')}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_SHORT[model],
                        mutation,
                        eligible,
                        applicable,
                        not_applicable,
                        _fmt_absolute(metric.get("full_absolute"), "acceptance"),
                        _fmt_absolute(metric.get("variant_absolute"), "acceptance"),
                        _fmt_delta(metric, "acceptance"),
                        _fmt_p(adjusted.get("adjusted_p")),
                        _fmt_flips(metric.get("paired_flips")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            (
                "### 13 类证书变异：静默派发"
                if zh
                else "### Thirteen certificate mutations: silent dispatch"
            ),
            "",
            "| Model | Mutation | Applicable cases | Full silent dispatch | no-rebind silent dispatch | no-rebind − Full (10k bootstrap 95% CI) | Holm-adjusted p | Paired flips |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model in MODELS:
        for mutation in MUTATION_NAMES:
            cell = _certificate_mutation_cell(paired, model, mutation)
            metric = _certificate_mutation_metric(
                paired, model, mutation, "silent_dispatch"
            )
            adjusted = _certificate_mutation_holm_metric(
                holm, model, mutation, "silent_dispatch"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        MODEL_SHORT[model],
                        mutation,
                        str(cell.get("applicable_case_count", "N/A")),
                        _fmt_absolute(
                            metric.get("full_absolute"), "silent_dispatch"
                        ),
                        _fmt_absolute(
                            metric.get("variant_absolute"), "silent_dispatch"
                        ),
                        _fmt_delta(metric, "silent_dispatch"),
                        _fmt_p(adjusted.get("adjusted_p")),
                        _fmt_flips(metric.get("paired_flips")),
                    ]
                )
                + " |"
            )

    excluded = invalid.get("excluded_snapshot_count", "N/A")
    excluded_cases = invalid.get("excluded_case_count", "N/A")
    source_available = invalid.get("source_available", False)
    lines.extend(
        [
            "",
            (
                f"无效 Full 快照在进入 13 类变异分母前排除：snapshots={excluded}，"
                f"cases={excluded_cases}，逐条原因文件可用={source_available}。"
                "每个 case 内多个快照先按 any 聚合；只有 applicable case 进入接受率、"
                "静默派发率、bootstrap 与 McNemar。N/A-only case 仅作分母披露，"
                "不会被当作拒绝或接受。"
                if zh
                else f"Invalid Full snapshots excluded before the 13 mutation denominators: "
                f"snapshots={excluded}, cases={excluded_cases}, row-level reason file "
                f"available={source_available}. Multiple snapshots are reduced with any "
                "within case. Only applicable cases enter acceptance/silent-dispatch rates, "
                "bootstrap, and McNemar; N/A-only cases are disclosed but are counted as "
                "neither rejection nor acceptance."
            ),
        ]
    )
    return lines


def _integrity_detail_lines(
    feasibility: list[dict[str, Any]] | None,
    certificate: list[dict[str, Any]] | None,
    language: str,
) -> list[str]:
    zh = language == "zh"
    lines = [
        "### 后端可实现性明细" if zh else "### Backend-realizability details",
        "",
        "| Model | Disabled capability | Injections | Active cases | Full unrealizable | no-feas unrealizable | no-feas attempted dispatch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if feasibility is None:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    else:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in feasibility:
            groups[(str(row.get("model") or "unknown"), str(row.get("disabled_capability") or "unknown"))].append(row)
        if not groups:
            lines.append(
                "| — | observed empty active set | 0 | 0 | 0 | 0 | 0 |"
            )
        for (model, capability), rows in sorted(groups.items()):
            lines.append(
                f"| {model} | {capability} | {len(rows)} | "
                f"{len({str(row.get('case_id')) for row in rows})} | "
                f"{sum(bool(row.get('full_selected_unrealizable')) for row in rows)} | "
                f"{sum(bool(_mapping(row.get('no_feasibility_filter')).get('unrealizable_plan_selected')) for row in rows)} | "
                f"{sum(bool(_mapping(row.get('no_feasibility_filter')).get('attempted_dispatch_under_unavailable_capability')) for row in rows)} |"
            )
    lines.extend(
        [
            "",
            "### 证书变异重绑定明细" if zh else "### Certificate-mutation rebinding details",
            "",
            "| Model | Mutation | Eligible rows | Applicable rows | N/A rows | Eligible cases | Applicable cases | Full accepts | no-rebind accepts | no-rebind silent dispatch |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if certificate is None:
        lines.append(
            "| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        )
    else:
        groups = defaultdict(list)
        for row in certificate:
            groups[(str(row.get("model") or "unknown"), str(row.get("mutation") or "unknown"))].append(row)
        if not groups:
            lines.append(
                "| — | observed empty active set | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
            )
        for (model, mutation), rows in sorted(groups.items()):
            applicable = [row for row in rows if row.get("applicable") is True]
            lines.append(
                f"| {model} | {mutation} | {len(rows)} | "
                f"{len(applicable)} | {len(rows) - len(applicable)} | "
                f"{len({str(row.get('case_id')) for row in rows})} | "
                f"{len({str(row.get('case_id')) for row in applicable})} | "
                f"{sum(bool(_mapping(row.get('full_toolgate')).get('accepted')) for row in applicable)} | "
                f"{sum(bool(_mapping(row.get('no_rebind')).get('accepted')) for row in applicable)} | "
                f"{sum(bool(_mapping(row.get('no_rebind')).get('silent_dispatch')) for row in applicable)} |"
            )
    if feasibility is not None or certificate is not None:
        all_rows = [*(feasibility or []), *(certificate or [])]
        external_calls = sum((_integer(row.get("external_tool_calls")) or 0) for row in all_rows)
        simulator_only = all(bool(row.get("simulator_only")) for row in all_rows)
        lines.append(
            "\n"
            + (
                f"故障注入 simulator_only={simulator_only}；external_tool_calls={external_calls}。"
                if zh
                else f"Fault injection simulator_only={simulator_only}; external_tool_calls={external_calls}."
            )
        )
    return lines


def _settings_lines(root: Path, model_configs: Any, language: str) -> list[str]:
    config_path = root / "config" / "experiment.yaml"
    try:
        config_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    except (OSError, UnicodeError):
        config_text = ""
    seed = "N/A"
    for raw in config_text.splitlines():
        if raw.strip().startswith("seed:"):
            seed = raw.split(":", 1)[1].strip() or "N/A"
            break
    zh = language == "zh"
    lines = [
        f"- {'方法' if zh else 'Method'}: **ObliGate**",
        f"- {'基准' if zh else 'Benchmark'}: AgentDojo v1.2.2 / `agentdojo==0.1.35`; 949 task pairs; `important_instructions` attack.",
        f"- {'受测模型' if zh else 'Victim models'}: `deepseek-v4-flash`, `qwen-plus`; {'分开估计，不做跨模型简单平均' if zh else 'estimated separately; no simple cross-model average'}.",
        f"- {'运行参数' if zh else 'Runtime'}: temperature=0.0; seed={seed}; max_iters=24; {'每个 case/variant 从相同初始状态重置' if zh else 'identical reset for every case/variant'}.",
        f"- {'端到端规模' if zh else 'E2E scale'}: 8 configurations × 949 cases × 2 models = 15,184 episodes.",
        f"- {'推断' if zh else 'Inference'}: Wilson 95% CI; case-clustered bootstrap (10,000 draws); exact McNemar; within-model Holm correction.",
    ]
    routes = _mapping(model_configs)
    lines.extend(
        [
            "",
            "| Model | API base URL | Key env | Key SHA256 (prefix) | Temperature |",
            "|---|---|---|---|---:|",
        ]
    )
    for model in MODELS:
        value = _mapping(routes.get(model))
        fingerprint = str(value.get("credential_sha256") or "")
        lines.append(
            f"| {model} | {value.get('base_url', 'N/A')} | {value.get('api_key_env', 'N/A')} | "
            f"{fingerprint[:12] if fingerprint else 'N/A'} | {value.get('temperature', 'N/A')} |"
        )
    lines.append(
        "\n"
        + (
            "API key 原文从未写入报告；仅记录环境变量名、长度和 SHA256 指纹。"
            if zh
            else "Raw API keys are never written to the report; only environment names, lengths, and SHA256 fingerprints are audited."
        )
    )
    return lines


def _e2e_lines(e2e: Any, model: str, language: str) -> list[str]:
    lines = [
        "| Variant | Obs. ASR | Obs. User | Obs. Secure | Cons. ASR | Cons. User | Cons. Secure | Invalid | Block | Confirm | Constrained | Allow | Intervention | Recovery | Avg decisions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in E2E_VARIANTS:
        cell = _cell(e2e, model, variant)
        behavior = _mapping(cell.get("behavior"))
        invalid = _mapping(cell.get("invalid"))
        row = [
            variant,
            _fmt_summary(_metric_summary(cell, "observed", "targeted_asr")),
            _fmt_summary(_metric_summary(cell, "observed", "user_utility")),
            _fmt_summary(_metric_summary(cell, "observed", "secure_utility")),
            _fmt_summary(_metric_summary(cell, "conservative", "targeted_asr")),
            _fmt_summary(_metric_summary(cell, "conservative", "user_utility")),
            _fmt_summary(_metric_summary(cell, "conservative", "secure_utility")),
            _fmt_summary(invalid),
            _fmt_percent(behavior.get("block_rate")),
            _fmt_percent(behavior.get("require_confirmation_rate")),
            _fmt_percent(behavior.get("execute_with_constraints_rate")),
            _fmt_percent(behavior.get("ordinary_allow_rate")),
            _fmt_percent(behavior.get("intervention_case_rate")),
            _fmt_summary(_mapping(behavior.get("post_intervention_task_recovery"))),
            _fmt_number(behavior.get("average_tool_decisions_per_case")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    if language == "zh":
        lines.append(
            "\nObserved 仅使用有效返回；conservative 将 provider/workflow/parser failure 保留在 949 分母中（ASR 计为失败防御，效用计为失败任务）。"
        )
    else:
        lines.append(
            "\nObserved uses valid returns only. Conservative retains provider/workflow/parser failures in the 949-case denominator (unsafe for ASR and unsuccessful for utility)."
        )
    return lines


def _detailed_statistics_lines(paired: Any, holm: Any, model: str, language: str) -> list[str]:
    names = METRIC_NAMES_ZH if language == "zh" else METRIC_NAMES_EN
    lines = [
        "| Variant | Metric | Full absolute | Variant absolute | Variant − Full (95% CI) | Holm-adjusted p | Paired flips | Active n |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for variant in RESEARCH_VARIANTS:
        for metric in METRICS:
            value = _stats_metric(paired, model, variant, metric)
            adjusted = _holm_metric(holm, model, variant, metric)
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        names[metric],
                        _fmt_absolute(value.get("full_absolute"), metric),
                        _fmt_absolute(value.get("variant_absolute"), metric),
                        _fmt_delta(value, metric),
                        _fmt_p(adjusted.get("adjusted_p")),
                        _fmt_flips(value.get("paired_flips")),
                        str(value.get("active_case_denominator", "N/A")),
                    ]
                )
                + " |"
            )
    return lines


def _examples_lines(
    e2e: Any,
    cf_rows: list[dict[str, Any]],
    feasibility: list[dict[str, Any]] | None,
    certificate: list[dict[str, Any]] | None,
    language: str,
) -> list[str]:
    zh = language == "zh"
    lines: list[str] = []
    expected_flag = {
        "unbound-evidence": "unsafe_flip",
        "conflict-to-support": "unsafe_flip",
        "conflict-to-refute": "over_conservative_flip",
        "gap-blind": "unsafe_flip",
        "single-remedy": "non_blocking_plan_lost",
    }
    lines.append("### 1. 正向机制见证" if zh else "### 1. Positive mechanism witnesses")
    found = False
    for variant, flag in expected_flag.items():
        selected = [
            row
            for row in cf_rows
            if str(row.get("variant")) == variant
            and (_flag(row, flag) or (variant == "unbound-evidence" and bool(row.get("unbound_evidence_misuse"))))
        ][:3]
        if not selected:
            continue
        found = True
        ids = ", ".join(f"{row.get('model', 'N/A')}:{row.get('case_id', 'N/A')}:{row.get('snapshot_id', 'N/A')}" for row in selected)
        lines.append(f"- `{variant}` — {flag}: {ids}")
    if not found:
        lines.append(
            "- N/A：未提供可解析的正向快照见证，不能构造示例。"
            if zh
            else "- N/A: no parseable positive snapshot witnesses were supplied; none are fabricated."
        )

    lines.append("\n### 2. 反事实/零效应结构" if zh else "\n### 2. Counter-evidence and null effects")
    no_effect = [
        row
        for row in cf_rows
        if not any(
            _flag(row, name)
            for name in (
                "unsafe_flip",
                "over_conservative_flip",
                "constraint_downgrade",
                "non_blocking_plan_lost",
            )
        )
    ]
    if no_effect:
        grouped = Counter(str(row.get("variant") or "unknown") for row in no_effect)
        lines.append(
            "- "
            + ("激活后仍无裁决级损伤的记录数：" if zh else "Active records with no decision-level harm: ")
            + ", ".join(f"{key}={value}" for key, value in sorted(grouped.items()))
            + "."
        )
        examples = no_effect[:3]
        lines.append(
            "- "
            + ("零效应示例：" if zh else "Null-effect examples: ")
            + ", ".join(f"{row.get('model', 'N/A')}:{row.get('case_id', 'N/A')}:{row.get('snapshot_id', 'N/A')}" for row in examples)
            + "."
        )
    else:
        lines.append(
            "- N/A：未提供零效应记录；不能据此声称每次激活都造成损伤。"
            if zh
            else "- N/A: no null-effect rows were supplied; this does not establish that every activation causes harm."
        )

    lines.append("\n### 3. 失败案例与边界" if zh else "\n### 3. Failure cases and boundaries")
    invalid_cells: list[str] = []
    for model in MODELS:
        for variant in E2E_VARIANTS:
            cell = _cell(e2e, model, variant)
            n = _integer(cell.get("n"))
            valid_n = _integer(cell.get("valid_n"))
            if n is not None and valid_n is not None and n > valid_n:
                invalid_cells.append(f"{model}/{variant}={n - valid_n}")
    lines.append(
        "- "
        + (
            ("端到端 provider/workflow/parser failures：" if zh else "E2E provider/workflow/parser failures: ")
            + (", ".join(invalid_cells) if invalid_cells else "0")
        )
        + "."
    )
    full_unrealizable = sum(bool(row.get("full_selected_unrealizable")) for row in (feasibility or []))
    full_tamper = sum(bool(_mapping(row.get("full_toolgate")).get("accepted")) for row in (certificate or []))
    lines.append(
        f"- Full unrealizable selections={full_unrealizable if feasibility is not None else 'N/A'}; "
        f"Full mutated-certificate acceptance={full_tamper if certificate is not None else 'N/A'}."
    )
    lines.append(
        "- "
        + (
            "即使上述计数为 0，也只约束本次 AgentDojo、激活集合、后端能力注入和 13 类证书变异；不代表开放世界中的绝对安全。"
            if zh
            else "Zeros above are scoped to AgentDojo, observed active sets, injected backend failures, and the 13 tested certificate mutations; they are not an open-world safety guarantee."
        )
    )
    return lines


def _analysis_lines(
    paired: Any,
    cf: dict[str, dict[str, dict[str, Any]]],
    faults: dict[str, dict[str, dict[str, Any]]],
    language: str,
) -> list[str]:
    zh = language == "zh"
    lines: list[str] = []
    interpretations = {
        "unbound-evidence": (
            "动作级绑定主要由危险放行、跨对象/跨步骤证据复用来检验。",
            "Action-level binding is tested through unsafe release and cross-object/cross-step evidence reuse.",
        ),
        "gap-blind": (
            "Gap 编译的价值不能只看干预下降；还要同时看 ASR、Secure Utility 与错误任务轨迹。",
            "Gap compilation cannot be judged from lower intervention alone; ASR, Secure Utility, and trajectory errors matter jointly.",
        ),
        "scalar-average": (
            "标量替代在干预负担相近时，应重点检查强约束稀释和 Secure Utility，而非只看平均风险分数。",
            "The scalar surrogate is evaluated for hard-constraint dilution and Secure Utility at comparable intervention burden.",
        ),
        "single-remedy": (
            "Single-remedy 可能保持安全，却因无法 lifted join 而增加 block 或丢失非阻断方案。",
            "Single-remedy may retain safety yet increase blocking or lose non-blocking plans without lifted join.",
        ),
        "dag-single": (
            "DAG-Single 保留有符号证据闭包，但只对已有单规则控制逐条判定；不使用 EOC、lifted ECP 或证书。",
            "DAG-Single retains the signed witness closure and tests existing individual-rule controls independently; it uses no EOC, lifted ECP, or certificate.",
        ),
        "policy-prompt-only": (
            "PolicyPromptOnly 将完整打包策略及可观测谓词放入系统提示，不在外部边界强制执行；提示不同，因此不要求和 Full 命中公共前缀缓存。",
            "PolicyPromptOnly includes the bundled policy and observable predicates in the system prompt, without external boundary enforcement. Its distinct prompt is exempt from the Full common-prefix cache-hit requirement.",
        ),
        "block-all-guarded": (
            "该 sanity reference 只说明低 ASR 可由普遍阻断获得；若效用下降，它不能作为组件必要性的证明。",
            "This sanity reference shows that low ASR can be bought by blanket blocking; utility loss prevents treating it as evidence of component necessity.",
        ),
    }
    for variant in RESEARCH_VARIANTS:
        lines.append(f"### `{variant}`")
        model_fragments: list[str] = []
        for model in MODELS:
            asr = _stats_metric(paired, model, variant, "targeted_asr")
            utility = _stats_metric(paired, model, variant, "user_utility")
            secure = _stats_metric(paired, model, variant, "secure_utility")
            model_fragments.append(
                f"{MODEL_SHORT[model]}: ΔASR={_fmt_delta(asr, 'targeted_asr')}, "
                f"ΔUser={_fmt_delta(utility, 'user_utility')}, "
                f"ΔSecure={_fmt_delta(secure, 'secure_utility')}"
            )
        lines.append("- " + "; ".join(model_fragments) + ".")
        if variant in cf[MODELS[0]]:
            fragments = []
            for model in MODELS:
                value = cf[model][variant]
                fragments.append(
                    f"{MODEL_SHORT[model]} active={value['active_cases'] if value['available'] else 'N/A'}, "
                    f"unsafe={value['unsafe_flips'] if value['available'] else 'N/A'}, "
                    f"over={value['over_conservative_flips'] if value['available'] else 'N/A'}, "
                    f"downgrade={value['constraint_downgrades'] if value['available'] else 'N/A'}, "
                    f"plan-loss={value['non_blocking_plans_lost'] if value['available'] else 'N/A'}, "
                    f"first-divergence={value['first_decision_divergences'] if value['available'] else 'N/A'}, "
                    f"unbound-misuse={value['unbound_evidence_misuse_records'] if value['available'] else 'N/A'}"
                )
            lines.append("- " + "; ".join(fragments) + ".")
        lines.append("- " + interpretations[variant][0 if zh else 1])
        lines.append(
            "- "
            + (
                "若区间跨 0 或 active n=0，应解释为证据不足/局部无效应，而不是把预期方向写成已证实事实。"
                if zh
                else "If the interval crosses zero or active n=0, the honest conclusion is insufficient/local-null evidence, not a confirmed expected direction."
            )
        )
    lines.append("### Snapshot-only bipolar conflict variants")
    for variant in ("conflict-to-support", "conflict-to-refute"):
        fragments = []
        for model in MODELS:
            value = cf[model][variant]
            fragments.append(
                f"{MODEL_SHORT[model]} active={value['active_cases'] if value['available'] else 'N/A'}, "
                f"unsafe={value['unsafe_flips'] if value['available'] else 'N/A'}, "
                f"over={value['over_conservative_flips'] if value['available'] else 'N/A'}"
            )
        lines.append(f"- `{variant}`: " + "; ".join(fragments) + ".")
    lines.append("### Backend feasibility and certificate rebinding")
    for variant in ("no-feasibility-filter", "no-rebind"):
        fragments = []
        for model in MODELS:
            value = faults[model][variant]
            fragments.append(
                f"{MODEL_SHORT[model]} active={value['active_cases'] if value['available'] else 'N/A'}, "
                f"research acceptance={value['acceptance'] if value['available'] else 'N/A'}, "
                f"Full acceptance={value['full_acceptance'] if value['available'] else 'N/A'}"
            )
        lines.append(f"- `{variant}`: " + "; ".join(fragments) + ".")
    return lines


def _audit_evidence_lines(audit: Mapping[str, Any], language: str) -> list[str]:
    zh = language == "zh"
    cache = _mapping(audit.get("cache_boundary_audit"))
    labels = _mapping(audit.get("label_boundary_audit"))
    protected = _mapping(audit.get("protected_results_audit"))
    frozen_hash = str(protected.get("frozen_code_aggregate_sha256") or "")
    current_hash = str(protected.get("current_code_aggregate_sha256") or "")
    return [
        "",
        "| Boundary | Evidence | Result |",
        "|---|---|---|",
        (
            "| Common-prefix cache | "
            f"rows={cache.get('row_count', 'N/A')}; prompts={cache.get('unique_prompt_hashes', 'N/A')}; "
            f"shared-cross-variant={cache.get('shared_cross_variant_prompt_hash_count', 'N/A')}; "
            f"cross-variant-hits={cache.get('cross_variant_cache_hit_count', 'N/A')}; "
            f"provider-calls={cache.get('provider_call_count', 'N/A')}; "
            f"input/output/cached-tokens={cache.get('input_tokens', 'N/A')}/"
            f"{cache.get('output_tokens', 'N/A')}/"
            f"{cache.get('provider_reported_cached_input_tokens', 'N/A')}; "
            f"request-conflicts={len(cache.get('prompt_hash_request_conflicts') or [])}; "
            f"response-conflicts={len(cache.get('prompt_hash_response_conflicts') or [])} | "
            f"{cache.get('status', 'incomplete')} |"
        ),
        (
            "| Label boundary / 5-fold cross-fit | "
            f"labels-read={labels.get('outcome_or_scorer_labels_read', 'N/A')}; "
            f"fields={labels.get('calibration_fields_read', 'N/A')}; "
            f"folds={labels.get('fold_count', 'N/A')}; "
            f"held-out-used={labels.get('held_out_cases_used_in_fit', 'N/A')} | "
            f"{'pass' if labels.get('outcome_or_scorer_labels_read') == [] and labels.get('held_out_cases_used_in_fit') is False else 'incomplete/fail'} |"
        ),
        (
            "| Protected old-result hashes | "
            f"trees-unchanged={protected.get('all_protected_results_unchanged', 'N/A')}; "
            f"formal-code-unchanged={protected.get('formal_code_hash_unchanged', 'N/A')}; "
            f"frozen={frozen_hash[:12] if frozen_hash else 'N/A'}; "
            f"current={current_hash[:12] if current_hash else 'N/A'} | "
            f"{protected.get('status', 'incomplete')} |"
        ),
        "",
        (
            "缓存 hash 绑定 model、规范化 system prompt、完整 history、tool schema、可见 tool state、temperature 与模型参数；仅同一 hash 可复用。"
            if zh
            else "The cache hash binds model, normalized system prompt, complete history, tool schema, visible tool state, temperature, and model parameters; reuse is exact-hash only."
        ),
    ]


def build_report(output_root: str | Path, *, language: str = "zh") -> str:
    if language not in {"zh", "en"}:
        raise ValueError("language must be 'zh' or 'en'")
    root = Path(output_root).resolve()
    e2e = _load_json(root / "reports" / "e2e_summary.json")
    paired = _load_json(root / "statistics" / "paired_metrics.json")
    holm = _load_json(root / "statistics" / "holm_adjusted.json")
    snapshot_manifest = _load_json(root / "manifests" / "snapshot_manifest.json")
    model_configs = _load_json(root / "config" / "model_configs.json")
    cf_source = _load_jsonl(root / "snapshots" / "counterfactual_results.jsonl")
    cf_rows = _normalise_counterfactuals(cf_source)
    feasibility = _load_jsonl(root / "fault_injection" / "feasibility_results.jsonl")
    certificate = _load_jsonl(root / "fault_injection" / "certificate_mutation_results.jsonl")
    audit = _load_json(root / "reports" / "validity_audit.json") or build_validity_audit(root)
    cf = _counterfactual_summary(cf_rows, source_available=cf_source is not None)
    faults = _fault_summaries(feasibility, certificate)
    zh = language == "zh"

    lines = [
        "# ObliGate AgentDojo 双模型组件消融实验报告（Ablation v2）"
        if zh
        else "# ObliGate AgentDojo Dual-Model Component Ablation Report (Ablation v2)",
        "",
        f"> {'生成时间' if zh else 'Generated'}: {utc_now()}",
        f"> {'有效性审计状态' if zh else 'Validity audit status'}: **{audit.get('overall_status', 'incomplete')}**",
        "",
        (
            "**部署边界：只有 Full ObliGate 可部署并通过正式 Checker 与 ToolGate。所有 variants 均为仅用于研究的 AgentDojo 反事实策略，不进入部署路径，也不签发正式 ActionCertificate。**"
            if zh
            else "**Deployment boundary: only Full ObliGate is deployable and passes the formal Checker and ToolGate. Every variant is an AgentDojo counterfactual research policy outside the deployment path and does not issue formal ActionCertificates.**"
        ),
        "",
        "## 一、实验目的与研究问题" if zh else "## 1. Objective and research question",
        "",
        (
            "本实验回答 RQ3：哪些机制分别支撑证据精度、义务完整性、非补偿式组合、后端可实现性与绑定派发。证据链由三部分组成：双模型端到端主消融、冻结动作快照反事实、后端/证书完整性故障注入。"
            if zh
            else "RQ3 asks which mechanisms support evidence precision, obligation completeness, non-compensatory composition, backend realizability, and bound dispatch. Evidence combines dual-model E2E ablation, frozen-action counterfactual replay, and backend/certificate fault injection."
        ),
        "",
        "## 二、实验设置与信息边界" if zh else "## 2. Experimental setup and information boundary",
        "",
        *_settings_lines(root, model_configs, language),
        "",
        (
            "消融与 scalar 校准不得读取 attack/user-success label、scorer state、case answer、normal/attacker tool partition 或 case-specific oracle。Scalar-average 只匹配 Full 的干预/block/confirmation 决策分布。"
            if zh
            else "Ablations and scalar calibration may not read attack/user-success labels, scorer state, case answers, normal/attacker tool partitions, or case-specific oracles. Scalar-average only matches Full intervention/block/confirmation distributions."
        ),
        "",
        "## 三、端到端绝对结果（observed 与 conservative）" if zh else "## 3. E2E absolute results (observed and conservative)",
    ]
    for model in MODELS:
        lines.extend(["", f"### {MODEL_SHORT[model]} (`{model}`)", "", *_e2e_lines(e2e, model, language)])
    lines.extend(
        [
            "",
            "## 四、Table A：端到端主消融" if zh else "## 4. Table A: Main E2E ablation",
            "",
            (
                "下表使用 conservative 主口径；每格为 Variant − Full 及 case-clustered bootstrap 95% CI。DS 与 Qwen 的行为差值也分开列出，没有跨模型平均。"
                if zh
                else "The table uses the conservative primary estimand. Each cell is Variant − Full with a case-clustered bootstrap 95% CI. Behavioral deltas are also model-specific; no cross-model average is used."
            ),
            "",
            *_table_a_lines(paired),
            "",
            "### 完整配对统计" if zh else "### Complete paired statistics",
        ]
    )
    for model in MODELS:
        lines.extend(["", f"#### {MODEL_SHORT[model]}", "", *_detailed_statistics_lines(paired, holm, model, language)])
    lines.append(
        "\n"
        + (
            "Holm-adjusted p 仅适用于冻结统计契约中的配对二元终点（exact McNemar）。动作率与平均决策数只冻结了 case-clustered bootstrap CI，故其 p-value 如实标为 N/A。"
            if zh
            else "Holm-adjusted p-values apply only to paired binary endpoints under exact McNemar. Action-rate and average-decision endpoints have frozen case-clustered bootstrap CIs but no pre-specified p-value, so p is reported as N/A."
        )
    )
    lines.extend(
        [
            "",
            "## 五、机制激活集合与空集限制" if zh else "## 5. Mechanism-active sets and empty-set limitations",
            "",
            *_activation_lines(snapshot_manifest, language),
            "",
            (
                "任何 active n=0 都是本次真实轨迹的结果，不补造正例；因此对应组件的局部因果效应为不可识别，而不是 0 效应。U/H/O 字段采用生产 RuntimeEvidence 中真实可见的 TaskContract 与事实/来源投影，并非 AgentDojo 原始 prompt/history 的第二份副本。"
                if zh
                else "Any active n=0 is an observed property of these trajectories and is not synthetically repaired; the corresponding local causal effect is unidentified, not zero. U/H/O fields are the exact TaskContract and observable fact/provenance projections in production RuntimeEvidence, not a second raw copy of the AgentDojo prompt/history."
            ),
            "",
            "## 六、Table B：机制级反事实与完整性" if zh else "## 6. Table B: Mechanism counterfactuals and integrity",
            "",
            *_table_b_lines(cf, faults),
            "",
            (
                "Table B 以 active `case_id` 为分母并按模型分开。故障注入不调用 victim model，也不触发外部工具；`no-feasibility-filter` 与 `no-rebind` 的 acceptance 仅表示研究模拟器暴露的预期失败模式。"
                if zh
                else "Table B uses active case_id denominators and separates models. Fault injection invokes neither victim models nor external tools; acceptance under no-feasibility-filter/no-rebind denotes the expected research-simulator failure mode only."
            ),
            "",
            *_mechanism_case_rate_lines(cf, language),
            "",
            *_mechanism_inference_lines(paired, holm, language),
            "",
            *_integrity_detail_lines(feasibility, certificate, language),
            "",
            *_certificate_mutation_inference_lines(paired, holm, language),
            "",
            "## 七、逐组件数据分析" if zh else "## 7. Component-wise analysis",
            "",
            *_analysis_lines(paired, cf, faults, language),
            "",
            "## 八、正向见证、反事实结构与失败案例" if zh else "## 8. Positive witnesses, counter-evidence, and failures",
            "",
            *_examples_lines(e2e, cf_rows, feasibility, certificate, language),
            "",
            "## 九、API、缓存、标签边界与旧结果 hash 审计" if zh else "## 9. API, cache, label-boundary, and old-result hash audit",
            "",
        ]
    )
    for check_id in (
        "api_route_and_credential_fingerprint",
        "common_prefix_cache_boundary",
        "label_boundary_and_crossfit",
        "protected_old_results_unchanged",
        "statistics_case_clustered_no_model_pooling",
    ):
        check = next((item for item in audit.get("checks", []) if item.get("id") == check_id), None)
        if check is None:
            lines.append(f"- `{check_id}`: **N/A / incomplete**")
        else:
            lines.append(f"- `{check_id}`: **{check.get('status')}** — {check.get('summary')}")
    lines.extend(_audit_evidence_lines(_mapping(audit), language))
    lines.extend(
        [
            "",
            "## 十、有效性结论与可复现性" if zh else "## 10. Validity conclusion and reproducibility",
            "",
            f"- {'总体状态' if zh else 'Overall status'}: **{audit.get('overall_status', 'incomplete')}**",
        ]
    )
    for check in audit.get("checks", []):
        lines.append(f"- `{check.get('id')}`: **{check.get('status')}** — {check.get('summary')}")
    lines.extend(
        [
            "",
            (
                "只有当所有 required checks 为 pass 时，才能把该轮写成完成的 Ablation v2。`incomplete` 表示输入尚缺或无法验证；`fail` 表示已有证据违反验收条件。任何 N/A 都不得用旧实验或理论预期回填。"
                if zh
                else "Ablation v2 is complete only when every required check passes. `incomplete` means missing/unverifiable evidence; `fail` means observed evidence violates an acceptance condition. No N/A is backfilled from historical runs or theoretical expectations."
            ),
            "",
            "## 十一、论文结论边界" if zh else "## 11. Scope of paper claims",
            "",
            (
                "若 Table A 的跨 case 差值、Table B 的 active-case flips 和完整性故障注入同时呈现预期结构，可支持如下受限结论：动作级绑定抑制跨对象/跨步骤证据复用；双极冲突避免乐观危险放行与悲观过度阻断；Gap 编译阻止无证据动作静默推进；非补偿式组合保留硬约束；lifted join 恢复可实现的非阻断方案；feasibility filter 与证书重绑定把抽象治理落实到派发。该结论只覆盖本次冻结 AgentDojo 分布、两个 victim models 与实际 active denominators；不扩张为开放世界绝对保证。"
                if zh
                else "When Table A case-level deltas, Table B active-case flips, and integrity fault injection jointly show the expected structure, the evidence supports a scoped claim: action-level binding prevents cross-object/step evidence reuse; bipolar conflicts avoid optimistic unsafe release and pessimistic over-blocking; gap compilation prevents unsupported actions from silently proceeding; non-compensatory composition preserves hard constraints; lifted join recovers realizable non-blocking plans; feasibility filtering and certificate rebinding enforce abstract governance at dispatch. The claim is limited to the frozen AgentDojo distribution, two victim models, and observed active denominators—not an open-world absolute guarantee."
            ),
        ]
    )
    return "\n".join(lines)


def build_paper_tables(output_root: str | Path) -> str:
    root = Path(output_root).resolve()
    paired = _load_json(root / "statistics" / "paired_metrics.json")
    holm = _load_json(root / "statistics" / "holm_adjusted.json")
    cf_source = _load_jsonl(root / "snapshots" / "counterfactual_results.jsonl")
    cf_rows = _normalise_counterfactuals(cf_source)
    feasibility = _load_jsonl(root / "fault_injection" / "feasibility_results.jsonl")
    certificate = _load_jsonl(root / "fault_injection" / "certificate_mutation_results.jsonl")
    lines = [
        "# ObliGate Ablation v2 — Paper Tables",
        "",
        "Only Full ObliGate is deployable. All ablation variants are research-only counterfactual policies outside the deployment path.",
        "",
        "## Table A. Main E2E ablation (conservative Variant − Full, 95% case-clustered bootstrap CI)",
        "",
        *_table_a_lines(paired),
        "",
        "No victim-model pooling is used; DS and Qwen behavior columns are separate, with no cross-model average.",
        "",
        "## Table B. Mechanism-level counterfactual and integrity results",
        "",
        *_table_b_lines(
            _counterfactual_summary(cf_rows, source_available=cf_source is not None),
            _fault_summaries(feasibility, certificate),
        ),
        "",
        *_mechanism_inference_lines(paired, holm, "en"),
        "",
        *_certificate_mutation_inference_lines(paired, holm, "en"),
        "",
        "N/A means the required input or active set was unavailable; no historical or synthetic value was substituted.",
    ]
    return "\n".join(lines)


def generate_reports(output_root: str | Path) -> dict[str, str]:
    root = Path(output_root).resolve()
    write_validity_audit(root)
    outputs = {
        "zh": root / "reports" / "ablation_report_zh.md",
        "en": root / "reports" / "ablation_report_en.md",
        "tables": root / "reports" / "paper_tables.md",
        "validity": root / "reports" / "validity_audit.json",
    }
    _atomic_write_text(outputs["zh"], build_report(root, language="zh"))
    _atomic_write_text(outputs["en"], build_report(root, language="en"))
    _atomic_write_text(outputs["tables"], build_paper_tables(root))
    return {key: str(path) for key, path in outputs.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate_reports(args.output_root)
    print(json.dumps(outputs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_paper_tables", "build_report", "generate_reports"]
