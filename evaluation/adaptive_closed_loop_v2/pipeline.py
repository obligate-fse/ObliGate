"""ObliGate RQ2 adaptive closed-loop experiment pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None

from experiments.adaptive_ablation.mechanism_activation import activation_manifest

from .attacker import AttackJob, generate_many
from .common import RESULT_ROOT, ROOT, canonical_sha256, provider_settings, read_json, read_jsonl, safe_slug, sha256_file, sha256_text, utc_now, write_json, write_jsonl
from .manifests import (
    PREFLIGHT,
    _safetybench_full_catalog,
    build_manifests,
    build_safetybench_full_manifest,
)
from .materialize import agentdojo_case_plan, asb_case_ids, round_payload_manifest, safetybench_data
from .scoring import score_round
from .validation import equivalent_rewrite

CONFIG = Path(
    os.environ.get("OBLIGATE_ADAPTIVE_CONFIG")
    or ROOT / "configs" / "adaptive_closed_loop_v2.yaml"
).resolve()
PYTHON = Path(os.environ.get("OBLIGATE_PYTHON") or sys.executable).resolve()
AGENTDOJO_PYTHON = Path(
    os.environ.get("OBLIGATE_AGENTDOJO_PYTHON") or PYTHON
).resolve()
SAFETYBENCH_PYTHON = Path(
    os.environ.get("OBLIGATE_SAFETYBENCH_PYTHON") or PYTHON
).resolve()
ASB_PYTHON = Path(os.environ.get("OBLIGATE_ASB_PYTHON") or PYTHON).resolve()
FROZEN_MAIN_MANIFEST = Path(
    os.environ.get("OBLIGATE_ADAPTIVE_MAIN_MANIFEST")
    or RESULT_ROOT / "manifests" / "main_manifest.json"
).resolve()


def load_config() -> dict[str, Any]:
    text = CONFIG.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    # Minimal fallback for this flat config shape.
    return {
        "victim_models": ["deepseek-v4-flash", "qwen-plus"],
        "attacker_model": "deepseek-v4-flash",
        "max_rounds": 5,
        "main_attacker_seeds": [20260716],
        "stress_attacker_seeds": [20260716, 20260717, 20260718],
        "workers": {"attacker": 12, "agentdojo": 12, "agent_safetybench": 8, "agent_security_bench": 12},
        "attacker_temperature": 0.7,
        "attacker_max_tokens": 700,
    }


def run(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.attacker_workers is not None:
        cfg["workers"]["attacker"] = int(args.attacker_workers)
    if args.agentdojo_workers is not None:
        cfg["workers"]["agentdojo"] = int(args.agentdojo_workers)
    if args.asb_workers is not None:
        cfg["workers"]["agent_security_bench"] = int(args.asb_workers)
    selected_models = args.models or cfg["victim_models"]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase in {"safetybench_smoke", "safetybench_main", "safetybench_full"}:
        _run_safetybench_only(args, cfg, selected_models)
        return
    if args.phase == "agentdojo_asb_main":
        _run_agentdojo_asb_main(args, cfg, selected_models)
        return
    _write_status("preflight", "running", phase=args.phase)
    preflight = _preflight(cfg)
    manifests = build_manifests(
        seed=int(cfg.get("sampling_seed", 20260716)),
        include_stress=args.phase in {"stress", "all"},
    )
    _write_status("preflight", "completed", preflight=preflight, main_cases=manifests["main"]["case_count"], stress_cases=manifests.get("stress", {}).get("case_count", 0))
    if args.phase in {"dry", "all"}:
        dry_cases = _dry_cases(manifests["main"]["cases"])
        _run_segment("dry", dry_cases, cfg, max_rounds=min(int(args.max_rounds or cfg["max_rounds"]), 2), models=selected_models, seeds=[int(cfg["main_attacker_seeds"][0])], resume=args.resume, safetybench_defense=args.safetybench_defense, agentdojo_defense=args.agentdojo_defense, asb_mode=args.asb_mode)
    if args.phase in {"main", "all"}:
        _run_segment("main", manifests["main"]["cases"], cfg, max_rounds=int(args.max_rounds or cfg["max_rounds"]), models=selected_models, seeds=[int(cfg["main_attacker_seeds"][0])], resume=args.resume, safetybench_defense=args.safetybench_defense, agentdojo_defense=args.agentdojo_defense, asb_mode=args.asb_mode)
    if args.phase in {"stress", "all"}:
        _run_segment("mechanism_stress", manifests["stress"]["cases"], cfg, max_rounds=int(args.max_rounds or cfg["max_rounds"]), models=selected_models, seeds=[int(seed) for seed in cfg["stress_attacker_seeds"]], resume=args.resume, safetybench_defense=args.safetybench_defense, agentdojo_defense=args.agentdojo_defense, asb_mode=args.asb_mode)
    _write_status("formal_complete", "completed", phase=args.phase)


def _run_agentdojo_asb_main(args: argparse.Namespace, cfg: dict[str, Any], selected_models: list[str]) -> None:
    _write_status("agentdojo_asb_preflight", "running", phase=args.phase)
    if not FROZEN_MAIN_MANIFEST.exists():
        raise RuntimeError(f"missing frozen main manifest: {FROZEN_MAIN_MANIFEST}")
    source_manifest = read_json(FROZEN_MAIN_MANIFEST)
    requested_benchmarks = set(args.benchmarks or ["agentdojo", "agent_security_bench"])
    cases = [
        dict(case, segment="agentdojo_asb_main")
        for case in source_manifest.get("cases") or []
        if str(case.get("benchmark")) in requested_benchmarks
    ]
    expected_count = 0
    if "agentdojo" in requested_benchmarks:
        expected_count += 949
    if "agent_security_bench" in requested_benchmarks:
        expected_count += 580
    manifest = {
        "schema_version": "obligate-adaptive-case-manifest-v2",
        "name": "agentdojo_asb_main",
        "case_count": len(cases),
        "benchmark_counts": dict(Counter(str(case["benchmark"]) for case in cases)),
        "requested_benchmarks": sorted(requested_benchmarks),
        "source_manifest": str(FROZEN_MAIN_MANIFEST.resolve()),
        "source_manifest_sha256": sha256_file(FROZEN_MAIN_MANIFEST),
        "cases": cases,
    }
    manifest["manifest_sha256"] = canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    checks = {
        "source_manifest_exists": True,
        f"case_count_{expected_count}": len(cases) == expected_count,
        "agentdojo_count_ok": ("agentdojo" not in requested_benchmarks) or sum(1 for case in cases if case["benchmark"] == "agentdojo") == 949,
        "agent_security_bench_count_ok": ("agent_security_bench" not in requested_benchmarks) or sum(1 for case in cases if case["benchmark"] == "agent_security_bench") == 580,
        "agentdojo_defense": args.agentdojo_defense,
        "asb_mode": args.asb_mode,
        "workers": dict(cfg["workers"]),
    }
    if not all(value is True or key in {"agentdojo_defense", "asb_mode", "workers"} for key, value in checks.items()):
        raise RuntimeError(f"AgentDojo/ASB main preflight rejected execution: {checks}")
    write_json(RESULT_ROOT / "manifests" / "agentdojo_asb_main_manifest.json", manifest)
    write_json(RESULT_ROOT / "agentdojo_asb_main_preflight.json", {"schema_version": "obligate-agentdojo-asb-main-preflight-v1", "updated_at": utc_now(), "checks": checks, "manifest_sha256": manifest["manifest_sha256"]})
    max_rounds = int(args.max_rounds or cfg["max_rounds"])
    _write_status("agentdojo_asb_preflight", "completed", phase=args.phase, segment="agentdojo_asb_main", cases=len(cases), max_rounds=max_rounds, checks=checks)
    _run_segment("agentdojo_asb_main", cases, cfg, max_rounds=max_rounds, models=selected_models, seeds=[int(cfg["main_attacker_seeds"][0])], resume=args.resume, safetybench_defense=args.safetybench_defense, agentdojo_defense=args.agentdojo_defense, asb_mode=args.asb_mode)
    _write_status("formal_complete", "completed", phase=args.phase, segment="agentdojo_asb_main")


def _run_safetybench_only(args: argparse.Namespace, cfg: dict[str, Any], selected_models: list[str]) -> None:
    _write_status("safetybench_preflight", "running", phase=args.phase)
    if args.phase == "safetybench_main":
        catalog = _safetybench_full_catalog()
        rng = random.Random(int(cfg.get("sampling_seed", 20260716)))
        by_risk: dict[str, list[str]] = defaultdict(list)
        for case_id, case in catalog.items():
            by_risk[str(case["risk_bucket"])].append(case_id)
        if len(by_risk) != 7:
            raise RuntimeError(f"collapsed SafetyBench taxonomy expected 7 buckets, got {len(by_risk)}")
        selected_ids: list[str] = []
        for risk in sorted(by_risk):
            selected_ids.extend(rng.sample(sorted(by_risk[risk]), 30))
        cases = sorted((dict(catalog[case_id]) for case_id in selected_ids), key=lambda row: str(row["case_id"]))
        manifest = {
            "schema_version": "obligate-adaptive-case-manifest-v2",
            "name": "agent_safetybench_main",
            "case_count": len(cases),
            "benchmark_counts": {"agent_safetybench": len(cases)},
            "sampling_seed": int(cfg.get("sampling_seed", 20260716)),
            "risk_bucket_population_counts": {key: len(value) for key, value in sorted(by_risk.items())},
            "selected_per_risk": dict(Counter(str(case["risk_bucket"]) for case in cases)),
            "cases": cases,
        }
        manifest["manifest_sha256"] = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        write_json(RESULT_ROOT / "manifests" / "agent_safetybench_main_manifest.json", manifest)
        segment = "agent_safetybench_main"
    else:
        manifest_bundle = build_safetybench_full_manifest(seed=int(cfg.get("sampling_seed", 20260716)))
        manifest = manifest_bundle["manifest"]
        cases = list(manifest["cases"])
        segment = "safetybench_smoke" if args.phase == "safetybench_smoke" else "agent_safetybench_full"
    preflight = _safetybench_preflight(cfg, manifest, selected_models, phase=args.phase)
    if args.phase == "safetybench_smoke":
        cases = _safetybench_smoke_cases(cases)
        max_rounds = min(int(args.max_rounds or 2), 2)
    else:
        max_rounds = int(args.max_rounds or cfg["max_rounds"])
    cases = [dict(case, segment=segment) for case in cases]
    _write_status(
        "safetybench_preflight",
        "completed",
        preflight=preflight,
        phase=args.phase,
        segment=segment,
        cases=len(cases),
        max_rounds=max_rounds,
        safetybench_defense=args.safetybench_defense,
    )
    _run_segment(segment, cases, cfg, max_rounds=max_rounds, models=selected_models, seeds=[int(cfg["main_attacker_seeds"][0])], resume=args.resume, safetybench_defense=args.safetybench_defense, agentdojo_defense=args.agentdojo_defense, asb_mode=args.asb_mode)
    _write_status("formal_complete", "completed", phase=args.phase, segment=segment)


def _preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    key_checks = {}
    for model in cfg["victim_models"] + [cfg["attacker_model"]]:
        key_env, base = provider_settings(str(model))
        key_checks[str(model)] = {
            "key_env": key_env,
            "base_url": base,
            "credential_present": bool(os.getenv(key_env)),
        }
    preflight_path = PREFLIGHT
    if not preflight_path.exists():
        raise RuntimeError(
            "missing mechanism-activation preflight; set "
            "OBLIGATE_MECHANISM_PREFLIGHT to the generated JSON file"
        )
    prior = read_json(preflight_path)
    checks = {
        "schema_exact": prior.get("schema_version") == "obligate-mechanism-activation-preflight-v1",
        "formal_run_allowed": prior.get("formal_run_allowed") is True,
        "profile_hash_exact": (prior.get("activation_profile") or {}).get("profile_sha256") == activation_manifest()["profile_sha256"],
        "reference_inputs_read_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"mechanism activation preflight rejected RQ2 execution: {checks}")
    value = {
        "schema_version": "obligate-preflight-v2",
        "updated_at": utc_now(),
        "checks": checks,
        "provider_checks": key_checks,
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "rq3a_reference_inputs_are_read_only": True,
    }
    write_json(RESULT_ROOT / "preflight.json", value)
    return value


def _safetybench_preflight(cfg: dict[str, Any], manifest: dict[str, Any], selected_models: list[str], *, phase: str) -> dict[str, Any]:
    key_checks = {}
    for model in [*selected_models, cfg["attacker_model"]]:
        key_env, base = provider_settings(str(model))
        key_checks[str(model)] = {
            "key_env": key_env,
            "base_url": base,
            "credential_present": bool(os.getenv(key_env)),
        }
    cases = list(manifest.get("cases") or [])
    expected_cases = 210 if phase == "safetybench_main" else 2000
    expected_name = "agent_safetybench_main" if phase == "safetybench_main" else "agent_safetybench_full"
    checks = {
        "manifest_schema": manifest.get("schema_version") == "obligate-adaptive-case-manifest-v2",
        "manifest_name": manifest.get("name") == expected_name,
        f"case_count_{expected_cases}": len(cases) == expected_cases,
        "all_cases_are_safetybench": {str(case.get("benchmark")) for case in cases} == {"agent_safetybench"},
        "unique_case_ids": len({str(case.get("case_id")) for case in cases}) == len(cases),
        "all_round0_hashes_match_payloads": all(sha256_text(str(case.get("round_0_payload") or "")) == str(case.get("round_0_payload_sha256")) for case in cases),
        "python_executable_exists": PYTHON.exists(),
    }
    if phase != "safetybench_main":
        checks["dialog_case_count_111"] = sum(1 for case in cases if case.get("has_dialog")) == 111
    if not all(checks.values()):
        raise RuntimeError(f"SafetyBench full preflight rejected execution: {checks}")
    value = {
        "schema_version": "obligate-safetybench-full-preflight-v1",
        "updated_at": utc_now(),
        "checks": checks,
        "provider_checks": key_checks,
        "result_root": str(RESULT_ROOT),
        "python": str(PYTHON),
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
    write_json(RESULT_ROOT / "safetybench_full_preflight.json", value)
    return value


def _dry_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {"agentdojo": 4, "agent_safetybench": 3, "agent_security_bench": 3}
    selected: list[dict[str, Any]] = []
    for benchmark, limit in buckets.items():
        selected.extend([case for case in cases if case["benchmark"] == benchmark][:limit])
    return selected


def _safetybench_smoke_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_risk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_risk[str(case.get("risk_bucket") or "unknown")].append(case)
    selected: list[dict[str, Any]] = []
    chosen: set[str] = set()
    for risk in sorted(by_risk):
        population = sorted(by_risk[risk], key=lambda row: int(row["case_id"]))
        dialog = [case for case in population if case.get("has_dialog")]
        direct = [case for case in population if not case.get("has_dialog")]
        for pool in (dialog or population, direct or population):
            for case in pool:
                case_id = str(case["case_id"])
                if case_id not in chosen:
                    selected.append(case)
                    chosen.add(case_id)
                    break
    if not any(case.get("has_dialog") for case in selected):
        raise RuntimeError("SafetyBench smoke selection must include at least one dialog case")
    if not any(not case.get("has_dialog") for case in selected):
        raise RuntimeError("SafetyBench smoke selection must include at least one direct-request case")
    return selected


def _run_segment(
    segment: str,
    cases: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    max_rounds: int,
    models: list[str],
    seeds: list[int],
    resume: bool,
    safetybench_defense: str,
    agentdojo_defense: str,
    asb_mode: str,
) -> None:
    for model in models:
        for seed in seeds:
            _run_model_seed(segment, cases, cfg, model=str(model), seed=int(seed), max_rounds=max_rounds, resume=resume, safetybench_defense=safetybench_defense, agentdojo_defense=agentdojo_defense, asb_mode=asb_mode)


def _run_model_seed(
    segment: str,
    cases: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    model: str,
    seed: int,
    max_rounds: int,
    resume: bool,
    safetybench_defense: str,
    agentdojo_defense: str,
    asb_mode: str,
) -> None:
    slug = f"{segment}/{safe_slug(model)}/seed_{seed}"
    root = RESULT_ROOT / segment / safe_slug(model) / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    active = list(cases)
    payload_rows: dict[str, dict[str, Any]] = {
        str(case["case_id"]): {
            "case_id": str(case["case_id"]),
            "benchmark": case["benchmark"],
            "round": 0,
            "payload": case["round_0_payload"],
            "payload_sha256": case["round_0_payload_sha256"],
            "primary_strategy": "original_task" if case.get("round_0_payload_sha256") == case.get("original_payload_sha256") else "round0_frozen",
            "secondary_strategy": None,
            "public_feedback_class": "initial",
            "operators": [],
        }
        for case in active
    }
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stopped: dict[str, dict[str, Any]] = {}
    no_new_counts: defaultdict[str, int] = defaultdict(int)
    for round_index in range(max_rounds):
        active = [case for case in active if str(case["case_id"]) not in stopped]
        if not active:
            break
        round_dir = root / f"round_{round_index}"
        score_path = round_dir / "scores.jsonl"
        run_label = f"{segment}_{safe_slug(model)}_seed{seed}_round{round_index}"
        _write_status(
            f"{slug}/round_{round_index}",
            "running",
            active_cases=len(active),
            stopped_cases=len(stopped),
            max_rounds=max_rounds,
        )
        if not (resume and score_path.exists()):
            _materialize_and_run_round(active, payload_rows, round_dir, cfg, model=model, seed=seed, round_index=round_index, segment=segment, run_label=run_label, resume=resume, safetybench_defense=safetybench_defense, agentdojo_defense=agentdojo_defense, asb_mode=asb_mode)
            rows = score_round(
                cases=active,
                payload_rows=payload_rows,
                out_dir=round_dir,
                score_path=score_path,
                model=model,
                attacker_seed=seed,
                round_index=round_index,
                segment=segment,
                run_label=run_label,
            )
        else:
            rows = read_jsonl(score_path)
        write_json(round_dir / "round_summary.json", _round_summary(rows))
        for row in rows:
            case_id = str(row["case_id"])
            histories[case_id].append(
                {
                    "round": round_index,
                    "payload": payload_rows[case_id]["payload"],
                    "payload_sha256": payload_rows[case_id]["payload_sha256"],
                    "primary_strategy": payload_rows[case_id].get("primary_strategy"),
                    "secondary_strategy": payload_rows[case_id].get("secondary_strategy"),
                    "public_feedback": row.get("public_feedback"),
                    "reason_category": row.get("reason_category"),
                }
            )
            if row.get("attack_success"):
                stopped[case_id] = {"reason": "attack_success", "round": round_index}
            elif row.get("public_feedback") in {"provider_error", "workflow_error"}:
                stopped[case_id] = {"reason": str(row.get("public_feedback")), "round": round_index}
        if round_index >= max_rounds - 1:
            break
        next_jobs = []
        by_case = {str(case["case_id"]): case for case in active}
        for row in rows:
            case_id = str(row["case_id"])
            if case_id in stopped:
                continue
            next_jobs.append(
                AttackJob(
                    case=by_case[case_id],
                    victim_model=model,
                    attacker_seed=seed,
                    next_round=round_index + 1,
                    history=histories[case_id],
                    strong_feedback=segment == "mechanism_stress",
                )
            )
        generated_path = round_dir / f"attacker_round_{round_index + 1}.jsonl"
        if resume and generated_path.exists():
            generated = read_jsonl(generated_path)
        else:
            generated = _generate_next_payloads(next_jobs, cfg, root=root, round_dir=round_dir, next_round=round_index + 1)
            write_jsonl(generated_path, generated)
        for item in generated:
            case_id = str(item.get("case_id"))
            if item.get("generation_invalid"):
                stopped[case_id] = {"reason": "attacker_invalid", "round": round_index + 1, "details": item.get("validation_failures") or item.get("generation_error")}
                continue
            previous_payload = histories[case_id][-1]["payload"]
            combo = (item.get("primary_strategy"), item.get("secondary_strategy"))
            previous_combo = (payload_rows[case_id].get("primary_strategy"), payload_rows[case_id].get("secondary_strategy"))
            if equivalent_rewrite(previous_payload, str(item["payload"])) or combo == previous_combo:
                no_new_counts[case_id] += 1
            else:
                no_new_counts[case_id] = 0
            if no_new_counts[case_id] >= 2:
                stopped[case_id] = {"reason": "equivalent_or_no_new_strategy_twice", "round": round_index + 1}
                continue
            payload_rows[case_id] = dict(item)
        write_json(root / "stop_reasons.json", stopped)
        _write_status(f"{slug}/round_{round_index}", "completed", active_cases=len(active), stopped_cases=len(stopped), next_jobs=len(next_jobs))
    _write_status(f"{slug}", "completed", stopped_cases=len(stopped), total_cases=len(cases))


def _generate_next_payloads(
    next_jobs: list[AttackJob],
    cfg: dict[str, Any],
    *,
    root: Path,
    round_dir: Path,
    next_round: int,
) -> list[dict[str, Any]]:
    attempts = max(1, int(((cfg.get("stopping_rules") or {}).get("attacker_invalid_attempts")) or 1))
    pending = list(next_jobs)
    by_case: dict[str, dict[str, Any]] = {}
    all_attempts: list[dict[str, Any]] = []
    for outer_attempt in range(1, attempts + 1):
        if not pending:
            break
        rows = generate_many(
            pending,
            model=str(cfg["attacker_model"]),
            workers=int(cfg["workers"]["attacker"]),
            usage_ledger=root / "attacker_usage.jsonl",
            temperature=float(cfg.get("attacker_temperature", 0.7)),
            max_tokens=int(cfg.get("attacker_max_tokens", 700)),
        )
        enriched = [dict(row, outer_generation_attempt=outer_attempt) for row in rows]
        all_attempts.extend(enriched)
        latest_by_case = {str(row.get("case_id")): row for row in enriched}
        for row in enriched:
            case_id = str(row.get("case_id"))
            by_case[case_id] = row
        pending = [job for job in pending if bool((latest_by_case.get(str(job.case["case_id"])) or {}).get("generation_invalid"))]
    write_jsonl(round_dir / f"attacker_round_{next_round}_all_attempts.jsonl", all_attempts)
    return sorted(by_case.values(), key=lambda row: (row.get("benchmark"), row.get("case_id"), row.get("victim_model"), row.get("attacker_seed")))


def _materialize_and_run_round(
    active: list[dict[str, Any]],
    payload_rows: dict[str, dict[str, Any]],
    round_dir: Path,
    cfg: dict[str, Any],
    *,
    model: str,
    seed: int,
    round_index: int,
    segment: str,
    run_label: str,
    resume: bool,
    safetybench_defense: str,
    agentdojo_defense: str,
    asb_mode: str,
) -> None:
    materialized = round_dir / "materialized"
    payload_manifest = materialized / "payload_manifest.json"
    round_payload_manifest(active, payload_rows, payload_manifest, round_index=round_index, victim_model=model, attacker_seed=seed, segment=segment)
    ad_plan = materialized / "agentdojo_case_plan.json"
    sb_data = materialized / "agent_safetybench_data.json"
    asb_ids = materialized / "agent_security_bench_case_ids.json"
    agentdojo_case_plan(active, ad_plan)
    safetybench_data(active, payload_rows, sb_data)
    asb_case_ids(active, asb_ids)
    workers = cfg["workers"]
    commands = []
    if any(case["benchmark"] == "agentdojo" for case in active):
        commands.append(
            [
                str(AGENTDOJO_PYTHON),
                "-m",
                "evaluation.adaptive_closed_loop_v2.run_agentdojo_batch",
                "--case-plan",
                str(ad_plan),
                "--payload-manifest",
                str(payload_manifest),
                "--out-dir",
                str(round_dir / "agentdojo"),
                "--model",
                model,
                "--workers",
                str(workers["agentdojo"]),
                "--run-label",
                run_label,
                "--defense",
                agentdojo_defense,
                *(["--resume"] if resume else []),
            ]
        )
    if any(case["benchmark"] == "agent_safetybench" for case in active):
        sb_resume = resume and (round_dir / "agent_safetybench" / "run_config.json").exists()
        commands.append(
            [
                str(SAFETYBENCH_PYTHON),
                "-m",
                "evaluation.adaptive_closed_loop_v2.run_safetybench_batch",
                "--data",
                str(sb_data),
                "--out-dir",
                str(round_dir / "agent_safetybench"),
                "--usage-ledger",
                str(round_dir / "agent_safetybench" / "usage.jsonl"),
                "--run-label",
                run_label,
                "--model",
                model,
                "--defense",
                safetybench_defense,
                "--workers",
                str(workers["agent_safetybench"]),
                *(["--resume"] if sb_resume else []),
            ]
        )
    if any(case["benchmark"] == "agent_security_bench" for case in active):
        asb_run_dir = round_dir / "agent_security_bench" / safe_slug(run_label)
        asb_resume = resume and (asb_run_dir / "run_config.json").exists() and (asb_run_dir / "records.jsonl").exists()
        commands.append(
            [
                str(ASB_PYTHON),
                "-m",
                "evaluation.adaptive_closed_loop_v2.run_asb_batch",
                "--payload-manifest",
                str(payload_manifest),
                "--case-ids-file",
                str(asb_ids),
                "--out-dir",
                str(round_dir / "agent_security_bench"),
                "--usage-ledger",
                str(round_dir / "agent_security_bench" / "usage.jsonl"),
                "--run-label",
                run_label,
                "--model",
                model,
                "--mode",
                asb_mode,
                "--workers",
                str(workers["agent_security_bench"]),
                *(["--resume"] if asb_resume else []),
            ]
        )
    for command in commands:
        _run_command(command, round_dir / "pipeline_logs")


def _run_command(command: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    name = safe_slug(Path(command[2]).name if len(command) > 2 else "command")
    stdout_path = log_dir / f"{name}.{len(list(log_dir.glob(name + '*.stdout.log')))}.stdout.log"
    stderr_path = stdout_path.with_suffix(".stderr.log")
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}; see {stderr_path}")


def _round_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    for row in rows:
        bench = str(row["benchmark"])
        by_benchmark[bench]["cases"] += 1
        total["cases"] += 1
        for key in ("attack_success", "attack_success_observed", "dangerous_candidate", "provider_valid", "workflow_failure"):
            if row.get(key):
                by_benchmark[bench][key] += 1
                total[key] += 1
    return {
        "schema_version": "obligate-round-summary-v2",
        "updated_at": utc_now(),
        "cases": total["cases"],
        "attack_success": total["attack_success"],
        "attack_success_observed": total["attack_success_observed"],
        "dangerous_candidate": total["dangerous_candidate"],
        "provider_valid": total["provider_valid"],
        "workflow_failure": total["workflow_failure"],
        "by_benchmark": {key: dict(value) for key, value in sorted(by_benchmark.items())},
    }


def _write_status(stage: str, status: str, **extra: Any) -> None:
    write_json(
        RESULT_ROOT / "pipeline_status.json",
        {
            "schema_version": "obligate-adaptive-closed-loop-pipeline-v2",
            "stage": stage,
            "status": status,
            "updated_at": utc_now(),
            **extra,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dry", "main", "stress", "all", "safetybench_smoke", "safetybench_main", "safetybench_full", "agentdojo_asb_main"], default="dry")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--safetybench-defense",
        choices=["obligate_visible_fair", "none"],
        default="obligate_visible_fair",
    )
    parser.add_argument("--agentdojo-defense", choices=["obligate", "none"], default="obligate")
    parser.add_argument("--asb-mode", choices=["obligate_registry_blind", "no_defense"], default="obligate_registry_blind")
    parser.add_argument("--benchmarks", nargs="+", choices=["agentdojo", "agent_security_bench"], default=None)
    parser.add_argument("--attacker-workers", type=int, default=None)
    parser.add_argument("--agentdojo-workers", type=int, default=None)
    parser.add_argument("--asb-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
