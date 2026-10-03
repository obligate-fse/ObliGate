"""Collect conservative case metrics, action behavior, and first divergences."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from obligate.theory.model import sha256_json

from .common import atomic_write_json, sha256_file, utc_now
from .protocol import E2E_VARIANTS, execution_contract
from .snapshot_analysis import iter_logical_snapshots
from .variant_runtime import predecision_base_evidence_digest


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _metric(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    return {
        "count": successes,
        "n": len(values),
        "rate": successes / len(values) if values else None,
        "wilson95": _wilson(successes, len(values)),
    }


def _raw_path(cell: Path, case_id: str, variant: str) -> Path:
    return cell / "raw_runs" / f"{case_id}_obligate_ablation_v2_{variant.replace('-', '_')}.json"


def _read_research_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            source = json.loads(raw)
            if str(source.get("event") or "decision") != "decision":
                continue
            decision = source.get("decision") if isinstance(source.get("decision"), Mapping) else source
            action = source.get("action") or decision.get("action") or {}
            action_digest = str(source.get("action_digest") or decision.get("action_digest") or sha256_json(action))
            raw_evidence_digest = str(
                source.get("evidence_input_digest")
                or source.get("evidence_digest")
                or decision.get("evidence_digest")
                or ""
            )
            evidence_digest = str(
                source.get("predecision_base_evidence_digest") or raw_evidence_digest
            )
            rows.append(
                {
                    "case_id": str(source.get("case_id") or path.stem),
                    "step": int(source.get("step") or source.get("sequence") or len(rows) + 1),
                    "action": action,
                    "action_digest": action_digest,
                    "evidence_digest": evidence_digest,
                    "raw_evidence_digest": raw_evidence_digest,
                    "predecision_digest": (
                        sha256_json(
                            {
                                "action_digest": action_digest,
                                "evidence_digest": evidence_digest,
                            }
                        )
                        if evidence_digest
                        else None
                    ),
                    "evidence_summary": source.get("evidence_summary") or {},
                    "public_decision": str(
                        decision.get("public_decision") or decision.get("decision") or "unknown"
                    ),
                    "triggers": decision.get("triggers") or source.get("triggers") or [],
                    "selected": decision.get("selected") or source.get("selected"),
                    "trace": decision.get("trace") or source.get("trace") or {},
                }
            )
    # The bridge emits exactly one ``decision`` row plus a separately typed
    # ``fresh_re_adjudication`` row for a dispatch.  Filtering by event above is
    # therefore sufficient.  Never deduplicate equal adjacent decisions: they
    # can be genuine repeated blocked actions and are part of the trajectory.
    return rows


def _full_trace_map(snapshot_dir: Path, model: str) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_logical_snapshots(snapshot_dir, model=model):
        decision = record["decision"]
        source = record["input"]
        action = source["action"]
        evidence = source["evidence"]
        action_digest = sha256_json(action)
        raw_evidence_digest = sha256_json(evidence)
        evidence_digest = predecision_base_evidence_digest(evidence)
        values[record["case_id"]].append(
            {
                "case_id": record["case_id"],
                "step": len(values[record["case_id"]]) + 1,
                "action": action,
                "action_digest": action_digest,
                "evidence_digest": evidence_digest,
                "raw_evidence_digest": raw_evidence_digest,
                "predecision_digest": sha256_json(
                    {
                        "action_digest": action_digest,
                        "evidence_digest": evidence_digest,
                    }
                ),
                "evidence_summary": {
                    "assertion_count": len(evidence.get("assertions") or ()),
                    "hazard_count": len(evidence.get("hazards") or ()),
                    "explicit_deny_count": len(evidence.get("explicit_denies") or ()),
                    "graph_node_count": len(evidence.get("graph_nodes") or ()),
                    "graph_edge_count": len(evidence.get("graph_edges") or ()),
                },
                "public_decision": decision.get("public_decision"),
                "triggers": decision.get("triggers") or [],
                "selected": decision.get("selected"),
                "trace": decision.get("trace") or {},
            }
        )
    return values


def _behavior(decisions: list[Mapping[str, Any]]) -> dict[str, Any]:
    names = [str(item.get("public_decision") or "unknown") for item in decisions]
    counts = Counter(names)
    total = len(names)
    return {
        "tool_decision_count": total,
        "allow_count": counts["allow"],
        "execute_with_constraints_count": counts["execute_with_constraints"],
        "require_confirmation_count": counts["require_confirmation"],
        "block_count": counts["block"] + counts["block_with_compliance_error"],
        "intervention_case": any(name != "allow" for name in names),
    }


def collect_cell(
    output_root: Path,
    manifest: Mapping[str, Any],
    *,
    model: str,
    variant: str,
    full_traces: Mapping[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cell = output_root / "e2e" / model / variant
    traces: dict[str, list[dict[str, Any]]] = {}
    if variant == "full":
        traces = dict(full_traces or _full_trace_map(cell / "snapshots", model))
    else:
        for case in manifest["cases"]:
            case_id = str(case["case_id"])
            traces[case_id] = _read_research_trace(cell / "decision_traces" / f"{case_id}.jsonl")

    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        path = _raw_path(cell, case_id, variant)
        invalid = False
        invalid_reason = None
        normalized: Mapping[str, Any] = {}
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                candidates = value.get("normalized_cases") or []
                if len(candidates) != 1:
                    raise ValueError(f"normalized case count={len(candidates)}")
                normalized = candidates[0]
            except Exception as exc:
                invalid = True
                invalid_reason = f"{type(exc).__name__}:{str(exc)[:200]}"
        else:
            invalid = True
            invalid_reason = "missing_output"
        observed_attack = None if invalid else bool(normalized.get("raw_agentdojo_injection_task_success"))
        observed_utility = None if invalid else bool(normalized.get("raw_agentdojo_user_task_success"))
        behavior = _behavior(traces.get(case_id, []))
        rows.append(
            {
                "schema_version": "obligate-ablation-v2-per-case-v1",
                "case_id": case_id,
                "suite": str(case["suite"]),
                "model": model,
                "variant": variant,
                **execution_contract(variant),
                "invalid": invalid,
                "invalid_reason": invalid_reason,
                "targeted_asr_observed": observed_attack,
                "user_utility_observed": observed_utility,
                "secure_utility_observed": (
                    None if invalid else bool(observed_utility and not observed_attack)
                ),
                "targeted_asr_conservative": True if invalid else observed_attack,
                "user_utility_conservative": False if invalid else observed_utility,
                "secure_utility_conservative": (
                    False if invalid else bool(observed_utility and not observed_attack)
                ),
                "post_intervention_task_recovery": (
                    None if invalid else bool(normalized.get("recovery_success"))
                ),
                **behavior,
                "raw_output": str(path.resolve()),
            }
        )

    per_case_path = cell / "per_case_results.jsonl"
    per_case_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = per_case_path.with_name(f".{per_case_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, per_case_path)

    valid = [row for row in rows if not row["invalid"]]
    action_totals = Counter()
    for row in valid:
        for key in (
            "tool_decision_count",
            "allow_count",
            "execute_with_constraints_count",
            "require_confirmation_count",
            "block_count",
        ):
            action_totals[key] += int(row[key])
    total_actions = action_totals["tool_decision_count"]
    summary = {
        "schema_version": "obligate-ablation-v2-cell-summary-v1",
        "created_at": utc_now(),
        "model": model,
        "variant": variant,
        "n": len(rows),
        "valid_n": len(valid),
        "invalid": _metric([bool(row["invalid"]) for row in rows]),
        "observed": {
            "targeted_asr": _metric([bool(row["targeted_asr_observed"]) for row in valid]),
            "user_utility": _metric([bool(row["user_utility_observed"]) for row in valid]),
            "secure_utility": _metric([bool(row["secure_utility_observed"]) for row in valid]),
        },
        "conservative": {
            "targeted_asr": _metric([bool(row["targeted_asr_conservative"]) for row in rows]),
            "user_utility": _metric([bool(row["user_utility_conservative"]) for row in rows]),
            "secure_utility": _metric([bool(row["secure_utility_conservative"]) for row in rows]),
        },
        "behavior": {
            "tool_decision_count": total_actions,
            "average_tool_decisions_per_case": total_actions / len(rows),
            "ordinary_allow_rate": action_totals["allow_count"] / total_actions if total_actions else None,
            "execute_with_constraints_rate": action_totals["execute_with_constraints_count"] / total_actions if total_actions else None,
            "require_confirmation_rate": action_totals["require_confirmation_count"] / total_actions if total_actions else None,
            "block_rate": action_totals["block_count"] / total_actions if total_actions else None,
            "intervention_case_rate": sum(bool(row["intervention_case"]) for row in rows) / len(rows),
            "post_intervention_task_recovery": _metric(
                [
                    bool(row["post_intervention_task_recovery"])
                    for row in valid
                    if row["intervention_case"]
                ]
            ),
        },
        "per_case_results": str(per_case_path.resolve()),
        "per_case_results_sha256": sha256_file(per_case_path),
    }
    atomic_write_json(cell / "summary.json", summary)
    return rows, traces, summary


def _first_divergence(
    case_id: str,
    model: str,
    variant: str,
    full: list[Mapping[str, Any]],
    other: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    def _trajectory_signature(row: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(row.get("action_digest") or sha256_json(row.get("action") or {})),
            str(row.get("evidence_digest") or ""),
        )

    def _suffix_branched(index: int) -> bool:
        return [
            _trajectory_signature(item) for item in full[index + 1 :]
        ] != [
            _trajectory_signature(item) for item in other[index + 1 :]
        ]

    for index in range(max(len(full), len(other))):
        left = full[index] if index < len(full) else None
        right = other[index] if index < len(other) else None
        if left is None or right is None:
            return {
                "case_id": case_id,
                "model": model,
                "variant": variant,
                "step": index + 1,
                "divergence_type": "downstream_trajectory_effect",
                "full": left,
                "variant_value": right,
                "subsequent_trajectory_branched": True,
            }
        same_action = bool(
            left.get("action_digest")
            and left.get("action_digest") == right.get("action_digest")
            and left.get("action") == right.get("action")
        )
        # A direct component effect is identifiable only when the complete
        # decision input is identical.  Same-index/same-action is insufficient:
        # a prior tool result may already have changed the evidence and state.
        same_predecision_input = bool(
            same_action
            and left.get("predecision_digest")
            and left.get("predecision_digest") == right.get("predecision_digest")
            and left.get("evidence_digest") == right.get("evidence_digest")
        )
        same_decision = left.get("public_decision") == right.get("public_decision")
        same_plan = (left.get("selected") or {}).get("plan") == (right.get("selected") or {}).get("plan")
        if not (same_predecision_input and same_decision and same_plan):
            direct = same_predecision_input
            return {
                "case_id": case_id,
                "model": model,
                "variant": variant,
                "step": index + 1,
                "divergence_type": "direct_decision_effect" if direct else "downstream_trajectory_effect",
                "action_a_t": left.get("action") if direct else {"full": left.get("action"), "variant": right.get("action")},
                "full_decision": left.get("public_decision"),
                "variant_decision": right.get("public_decision"),
                "full_triggers": left.get("triggers") or [],
                "variant_triggers": right.get("triggers") or [],
                "full_plan": (left.get("selected") or {}).get("plan"),
                "variant_plan": (right.get("selected") or {}).get("plan"),
                "evidence_summary": {
                    "same_predecision_input": same_predecision_input,
                    "full": left.get("evidence_summary") or {},
                    "variant": right.get("evidence_summary") or {},
                    "full_evidence_digest": left.get("evidence_digest"),
                    "variant_evidence_digest": right.get("evidence_digest"),
                    "full_gap": (left.get("trace") or {}).get("gap") or [],
                    "variant_gap": (right.get("trace") or {}).get("gap") or [],
                },
                "subsequent_trajectory_branched": (True if not direct else _suffix_branched(index)),
            }
    return {
        "case_id": case_id,
        "model": model,
        "variant": variant,
        "step": None,
        "divergence_type": "no_divergence",
        "full_decision": None,
        "variant_decision": None,
        "subsequent_trajectory_branched": False,
        "compared_step_count": len(full),
    }


def collect_all(output_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = ("deepseek-v4-flash", "qwen-plus")
    summaries: dict[str, Any] = {}
    trace_maps: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for model in models:
        full_rows, full_map, full_summary = collect_cell(
            output_root,
            manifest,
            model=model,
            variant="full",
        )
        summaries[f"{model}/full"] = full_summary
        trace_maps[(model, "full")] = full_map
        for variant in E2E_VARIANTS[1:]:
            _, variant_map, summary = collect_cell(
                output_root,
                manifest,
                model=model,
                variant=variant,
            )
            summaries[f"{model}/{variant}"] = summary
            trace_maps[(model, variant)] = variant_map

    divergence_path = output_root / "traces" / "first_divergence.jsonl"
    divergence_path.parent.mkdir(parents=True, exist_ok=True)
    divergence_counts: dict[str, int] = Counter()
    with divergence_path.open("w", encoding="utf-8", newline="\n") as handle:
        for model in models:
            for variant in E2E_VARIANTS[1:]:
                for case in manifest["cases"]:
                    case_id = str(case["case_id"])
                    row = _first_divergence(
                        case_id,
                        model,
                        variant,
                        trace_maps[(model, "full")].get(case_id, []),
                        trace_maps[(model, variant)].get(case_id, []),
                    )
                    if row is not None:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    if row is not None and row.get("divergence_type") != "no_divergence":
                        divergence_counts[f"{model}/{variant}"] += 1
        handle.flush()
        os.fsync(handle.fileno())
    value = {
        "schema_version": "obligate-ablation-v2-collection-v1",
        "created_at": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "cells": summaries,
        "first_divergence_case_counts": dict(sorted(divergence_counts.items())),
        "first_divergence": str(divergence_path.resolve()),
        "first_divergence_sha256": sha256_file(divergence_path),
    }
    atomic_write_json(output_root / "reports" / "e2e_summary.json", value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value = collect_all(args.output_root, args.case_manifest)
    print(json.dumps({"cells": len(value["cells"]), "first_divergence": value["first_divergence_case_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_all", "collect_cell"]
