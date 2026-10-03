"""Cross-model replay for successful ObliGate adaptive payloads.

For each victim model, collect the first *observed* successful payload per case
from the main closed-loop run and replay it once from a clean environment on
the opposite victim model. This diagnostic is kept separate from the main CASR
tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import RESULT_ROOT, read_json, read_jsonl, safe_slug, utc_now, write_json
from .manifests import build_manifests
from .pipeline import _materialize_and_run_round, _round_summary, load_config
from .scoring import score_round


def _main_cases_by_id() -> dict[str, dict[str, Any]]:
    manifest_path = RESULT_ROOT / "manifests" / "main_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = build_manifests(seed=20260716)["main"]
    return {str(case["case_id"]): case for case in manifest.get("cases") or []}


def _payloads_for_round(model: str, seed: int, round_index: int) -> dict[str, dict[str, Any]]:
    path = RESULT_ROOT / "main" / safe_slug(model) / f"seed_{seed}" / f"round_{round_index}" / "materialized" / "payload_manifest.json"
    if not path.exists():
        return {}
    manifest = read_json(path)
    return {str(row["case_id"]): row for row in manifest.get("cases") or []}


def collect_first_observed_successes(model: str, seed: int) -> dict[str, dict[str, Any]]:
    root = RESULT_ROOT / "main" / safe_slug(model) / f"seed_{seed}"
    successes: dict[str, dict[str, Any]] = {}
    payload_cache: dict[int, dict[str, dict[str, Any]]] = {}
    for score_path in sorted(root.glob("round_*/scores.jsonl")):
        round_index = int(score_path.parent.name.replace("round_", ""))
        if round_index not in payload_cache:
            payload_cache[round_index] = _payloads_for_round(model, seed, round_index)
        payloads = payload_cache[round_index]
        for row in read_jsonl(score_path):
            case_id = str(row.get("case_id"))
            if case_id in successes:
                continue
            if not row.get("attack_success_observed"):
                continue
            payload_row = payloads.get(case_id)
            if not payload_row:
                continue
            successes[case_id] = {
                **payload_row,
                "source_victim_model": model,
                "source_success_round": round_index,
                "source_score_public_feedback": row.get("public_feedback"),
                "source_benchmark": row.get("benchmark"),
            }
    return successes


def run_direction(source_model: str, target_model: str, seed: int, *, resume: bool) -> dict[str, Any]:
    cfg = load_config()
    all_cases = _main_cases_by_id()
    successes = collect_first_observed_successes(source_model, seed)
    cases = [all_cases[case_id] for case_id in successes if case_id in all_cases]
    out_root = RESULT_ROOT / "cross_model_replay" / f"{safe_slug(source_model)}_to_{safe_slug(target_model)}" / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    source_manifest = {
        "schema_version": "obligate-cross-model-replay-source-v1",
        "generated_at": utc_now(),
        "source_victim_model": source_model,
        "target_victim_model": target_model,
        "attacker_seed": seed,
        "observed_success_payload_count": len(successes),
        "replay_case_count": len(cases),
        "cases": list(successes.values()),
        "selection_rule": "first attack_success_observed payload per case from main run",
    }
    write_json(out_root / "source_success_payloads.json", source_manifest)
    if not cases:
        value = {"source_victim_model": source_model, "target_victim_model": target_model, "attacker_seed": seed, "cases": 0, "status": "no_observed_success_payloads"}
        write_json(out_root / "summary.json", value)
        return value

    payload_rows = {case_id: dict(payload) for case_id, payload in successes.items()}
    score_path = out_root / "round_0" / "scores.jsonl"
    run_label = f"cross_model_replay_{safe_slug(source_model)}_to_{safe_slug(target_model)}_seed{seed}"
    if not (resume and score_path.exists()):
        _materialize_and_run_round(
            cases,
            payload_rows,
            out_root / "round_0",
            cfg,
            model=target_model,
            seed=seed,
            round_index=0,
            segment="cross_model_replay",
            run_label=run_label,
            resume=resume,
        )
        rows = score_round(
            cases=cases,
            payload_rows=payload_rows,
            out_dir=out_root / "round_0",
            score_path=score_path,
            model=target_model,
            attacker_seed=seed,
            round_index=0,
            segment="cross_model_replay",
            run_label=run_label,
        )
    else:
        rows = read_jsonl(score_path)
    summary = {
        "schema_version": "obligate-cross-model-replay-summary-v1",
        "generated_at": utc_now(),
        "source_victim_model": source_model,
        "target_victim_model": target_model,
        "attacker_seed": seed,
        "source_observed_success_payloads": len(successes),
        "replayed_cases": len(cases),
        "round_summary": _round_summary(rows),
    }
    write_json(out_root / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = args.models or ["deepseek-v4-flash", "qwen-plus"]
    if len(models) != 2:
        raise ValueError("cross-model replay expects exactly two models")
    a, b = models
    summaries = [
        run_direction(a, b, args.seed, resume=args.resume),
        run_direction(b, a, args.seed, resume=args.resume),
    ]
    value = {"schema_version": "obligate-cross-model-replay-v1", "generated_at": utc_now(), "seed": args.seed, "directions": summaries}
    write_json(RESULT_ROOT / "cross_model_replay" / "cross_model_replay.json", value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--models", nargs=2, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    value = run(parse_args(argv))
    print(RESULT_ROOT / "cross_model_replay" / "cross_model_replay.json")
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
