"""Five-fold label-free threshold calibration for the scalar-average baseline."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from obligate.theory.model import canonical_json

from .common import atomic_write_json, utc_now
from .scalar_scoring import aggregate_score, has_hard_trigger, scalar_decision


def stable_fold(case_id: str, *, seed: int = 20260716, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _complete_records(path: Path) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            sid = str(row.get("snapshot_id") or "")
            if not sid:
                continue
            bucket = events.setdefault(sid, {})
            if sid not in order:
                order.append(sid)
            bucket[str(row.get("event"))] = row
    result: list[dict[str, Any]] = []
    for sid in order:
        bucket = events[sid]
        source = bucket.get("input")
        decision = bucket.get("decision")
        if source is None or decision is None:
            continue
        key = hashlib.sha256(
            canonical_json({"action": source["action"], "evidence": source["evidence"]}).encode("utf-8")
        ).hexdigest()
        gate = bucket.get("tool_gate")
        # Only the explicit no-gate -> gate pair is a production fresh rebind.
        # Equality alone is insufficient because an agent may retry an
        # identical blocked action on the next trajectory step.
        if (
            result
            and result[-1]["input_digest"] == key
            and result[-1].get("gate") is None
            and gate is not None
        ):
            result[-1]["gate"] = gate
            continue
        result.append(
            {
                "snapshot_id": sid,
                "case_id": str(source.get("case_id") or path.stem),
                "input": source,
                "decision": decision["deterministic"],
                "input_digest": key,
                "gate": gate,
            }
        )
    return result


def load_full_rows(model_snapshot_dirs: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, directory in sorted(model_snapshot_dirs.items()):
        for path in sorted(directory.glob("*.jsonl"), key=lambda item: item.name):
            for record in _complete_records(path):
                deterministic = record["decision"]
                eoc = (deterministic.get("trace") or {}).get("eoc") or {}
                rows.append(
                    {
                        "model": model,
                        "case_id": record["case_id"],
                        "snapshot_id": record["snapshot_id"],
                        "score": aggregate_score(eoc),
                        "hard": has_hard_trigger(eoc),
                        "full_decision": str(deterministic["public_decision"]),
                        "trigger_count": len(eoc.get("triggers") or ()),
                    }
                )
    if not rows:
        raise ValueError("no complete Full snapshot decisions found for scalar cross-fitting")
    return rows


def _rates(decisions: Iterable[str]) -> dict[str, float]:
    values = list(decisions)
    if not values:
        raise ValueError("cannot compute decision distribution from zero rows")
    return {
        "intervention_rate": sum(item != "allow" for item in values) / len(values),
        "block_rate": sum(item in {"block", "block_with_compliance_error"} for item in values) / len(values),
        "confirmation_rate": sum(item == "require_confirmation" for item in values) / len(values),
    }


def fit_thresholds(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    target = _rates(str(row["full_decision"]) for row in rows)
    grid = tuple(round(0.025 + index * 0.05, 3) for index in range(20)) + (1.0,)
    best: tuple[tuple[float, float, float, float, float, float], tuple[float, float, float], dict[str, float]] | None = None
    for thresholds in itertools.combinations(grid, 3):
        mapped = [
            scalar_decision(float(row["score"]), thresholds, hard=bool(row["hard"]))
            for row in rows
        ]
        achieved = _rates(mapped)
        squared = sum((achieved[key] - target[key]) ** 2 for key in target)
        absolute = sum(abs(achieved[key] - target[key]) for key in target)
        tie = (
            squared,
            absolute,
            abs(achieved["block_rate"] - target["block_rate"]),
            thresholds[0],
            thresholds[1],
            thresholds[2],
        )
        if best is None or tie < best[0]:
            best = (tie, thresholds, achieved)
    assert best is not None
    return {
        "thresholds": {
            "t_allow": best[1][0],
            "t_confirm": best[1][1],
            "t_block": best[1][2],
        },
        "training_row_count": len(rows),
        "training_case_count": len({str(row["case_id"]) for row in rows}),
        "training_model_counts": {
            model: sum(str(row["model"]) == model for row in rows)
            for model in sorted({str(row["model"]) for row in rows})
        },
        "full_target_distribution": target,
        "scalar_achieved_distribution": best[2],
        "objective_squared_error": best[0][0],
        "objective_absolute_error": best[0][1],
    }


def crossfit(
    model_snapshot_dirs: Mapping[str, Path],
    *,
    seed: int = 20260716,
    fold_count: int = 5,
    case_to_fold: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rows = load_full_rows(model_snapshot_dirs)
    observed_case_ids = sorted({str(row["case_id"]) for row in rows})
    if case_to_fold is None:
        assignment = {
            case_id: stable_fold(case_id, seed=seed, folds=fold_count)
            for case_id in observed_case_ids
        }
        assignment_source = "snapshot_cases"
    else:
        assignment = {str(case_id): int(fold) for case_id, fold in case_to_fold.items()}
        missing = sorted(set(observed_case_ids) - set(assignment))
        invalid = sorted(case_id for case_id, fold in assignment.items() if fold not in range(fold_count))
        drift = sorted(
            case_id
            for case_id, fold in assignment.items()
            if stable_fold(case_id, seed=seed, folds=fold_count) != fold
        )
        if missing or invalid or drift:
            raise ValueError(
                "invalid frozen fold assignment: "
                f"missing={missing[:5]}, invalid={invalid[:5]}, stable_hash_drift={drift[:5]}"
            )
        assignment_source = "frozen_full_case_manifest"
    folds: dict[str, Any] = {}
    for held_out in range(fold_count):
        train = [row for row in rows if assignment[str(row["case_id"])] != held_out]
        test = [row for row in rows if assignment[str(row["case_id"])] == held_out]
        fitted = fit_thresholds(train)
        fitted.update(
            {
                "held_out_fold": held_out,
                "held_out_case_count": sum(fold == held_out for fold in assignment.values()),
                "held_out_case_with_action_rows_count": len(
                    {str(row["case_id"]) for row in test}
                ),
                "held_out_row_count": len(test),
                "held_out_cases_used_in_fit": False,
            }
        )
        folds[str(held_out)] = fitted
    return {
        "schema_version": "obligate-scalar-crossfit-v1",
        "created_at": utc_now(),
        "seed": seed,
        "fold_count": fold_count,
        "case_to_fold": assignment,
        "case_assignment_count": len(assignment),
        "case_assignment_source": assignment_source,
        "folds": folds,
        "models_pooled_for_shared_thresholds": sorted(model_snapshot_dirs),
        "full_snapshot_sources": {
            model: {
                "path": str(path.resolve()),
                "file_count": len(tuple(path.glob("*.jsonl"))),
            }
            for model, path in sorted(model_snapshot_dirs.items())
        },
        "calibration_fields_read": ["case_id", "model", "full_decision", "triggers", "remedies"],
        "outcome_or_scorer_labels_read": [],
        "matches_only": ["overall intervention rate", "block rate", "confirmation rate"],
        "rank_definition": {
            "execution_env": {"host": 0.0, "sandbox": 0.5, "no_execute": 1.0},
            "network_scope": {"allow": 0.0, "allowlist": 0.5, "deny": 1.0},
            "data_scope": {"raw": 0.0, "redact": 0.5, "no_sensitive": 1.0},
            "human_gate": {"none": 0.0, "approval_required": 1.0},
            "audit": {"no_must": 0.0, "one_or_more_must": 1.0},
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepseek-snapshots", type=Path, required=True)
    parser.add_argument("--qwen-snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--fold-assignment", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frozen_assignment = None
    if args.fold_assignment is not None:
        frozen_assignment = json.loads(args.fold_assignment.read_text(encoding="utf-8"))[
            "case_to_fold"
        ]
    value = crossfit(
        {"deepseek-v4-flash": args.deepseek_snapshots, "qwen-plus": args.qwen_snapshots},
        seed=args.seed,
        case_to_fold=frozen_assignment,
    )
    atomic_write_json(args.output, value)
    print(json.dumps({"output": str(args.output), "folds": len(value["folds"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["crossfit", "fit_thresholds", "load_full_rows", "stable_fold"]
