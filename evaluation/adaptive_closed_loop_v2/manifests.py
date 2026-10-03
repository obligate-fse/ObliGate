from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.adaptive_ablation.mechanism_activation import activate_runtime_evidence
from experiments.adaptive_ablation.mechanism_activation_preflight import (
    _direct_conflict_keys,
    _local_remedy_audit,
    _runtimes,
)
from experiments.adaptive_ablation.snapshot_index import SnapshotEventIndex
from experiments.adaptive_ablation.snapshots import deserialize_action, deserialize_evidence

from .common import ROOT, RESULT_ROOT, canonical_sha256, read_json, read_jsonl, sha256_file, sha256_text, write_json


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).resolve()


FROZEN_INPUT_DIR = _env_path(
    "OBLIGATE_ADAPTIVE_INPUT_ROOT",
    ROOT / "reproduced" / "adaptive_inputs",
)
SAFETYBENCH_UPSTREAM = _env_path(
    "SAFETYBENCH_UPSTREAM",
    ROOT / "third_party" / "Agent-SafetyBench",
)
ASB_UPSTREAM = _env_path(
    "ASB_UPSTREAM",
    ROOT / "third_party" / "ASB",
)
AD_PLAN = _env_path(
    "OBLIGATE_AGENTDOJO_CASE_PLAN",
    ROOT / "data" / "agentdojo_v1.2.2_949_case_plan.json",
)
AD_SOURCES = FROZEN_INPUT_DIR / "agentdojo_public_sources.jsonl"
SB_SOURCES = FROZEN_INPUT_DIR / "agent_safetybench_public_sources.jsonl"
ASB_SOURCES = FROZEN_INPUT_DIR / "agent_security_bench_public_sources.jsonl"
AD_R0 = FROZEN_INPUT_DIR / "agentdojo_round2_payload_manifest.json"
SB_R0 = FROZEN_INPUT_DIR / "agent_safetybench_round2_payload_manifest.json"
ASB_R0 = FROZEN_INPUT_DIR / "agent_security_bench_round2_payload_manifest.json"
SB_RELEASED = SAFETYBENCH_UPSTREAM / "data" / "released_data.json"
SB_ENV_DIR = SAFETYBENCH_UPSTREAM / "environments"
ASB_PLAN = _env_path(
    "OBLIGATE_ASB_CASE_PLAN",
    ROOT / "data" / "asb_iclr2025_8160_case_plan.json",
)
PREFLIGHT = _env_path(
    "OBLIGATE_MECHANISM_PREFLIGHT",
    FROZEN_INPUT_DIR / "mechanism_activation_preflight.json",
)
SNAPSHOT_STREAM = _env_path(
    "OBLIGATE_STRESS_SNAPSHOT_STREAM",
    FROZEN_INPUT_DIR / "agentdojo_snapshots.jsonl",
)
SNAPSHOT_INDEX = _env_path(
    "OBLIGATE_STRESS_SNAPSHOT_INDEX",
    FROZEN_INPUT_DIR / "agentdojo_snapshot_index.sqlite3",
)
OLD_FORMAL = _env_path(
    "OBLIGATE_ADAPTIVE_REFERENCE_ROOT",
    FROZEN_INPUT_DIR / "reference_ablation",
)
CONFIRMATION_RAW_DIR = _env_path(
    "OBLIGATE_STRESS_CONFIRMATION_RAW_DIR",
    OLD_FORMAL / "agentdojo" / "full" / "raw_runs",
)


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_manifests(
    *,
    seed: int = 20260716,
    force: bool = False,
    include_stress: bool = True,
) -> dict[str, Any]:
    manifest_dir = RESULT_ROOT / "manifests"
    main_path = manifest_dir / "main_manifest.json"
    stress_path = manifest_dir / "mechanism_stress_manifest.json"
    audit_path = manifest_dir / "sampling_audit.json"
    if (
        not force
        and main_path.exists()
        and audit_path.exists()
        and (not include_stress or stress_path.exists())
    ):
        cached = {"main": read_json(main_path), "audit": read_json(audit_path)}
        if include_stress:
            cached["stress"] = read_json(stress_path)
        return cached

    if include_stress:
        _write_old_artifact_inventory()
    catalog = build_catalog()
    main_ids, sampling = _sample_main(catalog, seed)
    main_cases = [dict(catalog[benchmark][case_id], segment="main") for benchmark, case_id in main_ids]
    main = _manifest("main", main_cases, seed=seed)
    audit = {
        "schema_version": "obligate-sampling-audit-v2",
        "sampling_seed": seed,
        "generated_from_public_inputs_only": True,
        "forbidden_selection_fields": [
            "scorer_label",
            "attack_success",
            "historical_decision",
            "answer_template",
            "normal_or_attacker_partition",
            "known_bypass",
        ],
        "forbidden_selection_fields_used": [],
        "main_sampling": sampling,
        "source_hashes": {
            _path_label(path): sha256_file(path)
            for path in (
                AD_PLAN,
                AD_SOURCES,
                SB_SOURCES,
                ASB_SOURCES,
                AD_R0,
                SB_R0,
                ASB_R0,
                SB_RELEASED,
                ASB_PLAN,
            )
        },
        "main_manifest_content_sha256": main["manifest_sha256"],
    }
    stress = None
    if include_stress:
        stress_cases, stress_audit = _sample_stress(catalog, seed)
        stress = _manifest("mechanism_stress", stress_cases, seed=seed)
        audit["mechanism_stress_sampling"] = stress_audit
        audit["mechanism_stress_manifest_content_sha256"] = stress["manifest_sha256"]
        audit["source_hashes"][_path_label(PREFLIGHT)] = sha256_file(PREFLIGHT)
    write_json(main_path, main)
    if stress is not None:
        write_json(stress_path, stress)
    write_json(audit_path, audit)
    value = {"main": main, "audit": audit}
    if stress is not None:
        value["stress"] = stress
    return value


def build_safetybench_full_manifest(*, seed: int = 20260716, force: bool = False) -> dict[str, Any]:
    manifest_dir = RESULT_ROOT / "manifests"
    manifest_path = manifest_dir / "agent_safetybench_full_manifest.json"
    audit_path = manifest_dir / "agent_safetybench_full_audit.json"
    if not force and manifest_path.exists() and audit_path.exists():
        return {"manifest": read_json(manifest_path), "audit": read_json(audit_path)}

    catalog = _safetybench_full_catalog()
    cases = [dict(row, segment="agent_safetybench_full") for row in catalog.values()]
    manifest = _manifest("agent_safetybench_full", cases, seed=seed)
    env_hashes = _safetybench_environment_hashes()
    audit = {
        "schema_version": "obligate-safetybench-full-audit-v1",
        "sampling_seed": seed,
        "selection": "complete Agent-SafetyBench released_data.json population; no outcome fields",
        "case_count": len(cases),
        "generated_from_public_inputs_only": True,
        "dialog_case_count": sum(1 for item in cases if item.get("has_dialog")),
        "direct_case_count": sum(1 for item in cases if not item.get("has_dialog")),
        "risk_bucket_population_counts": dict(Counter(str(item["risk_bucket"]) for item in cases)),
        "source_hashes": {
            _path_label(SB_RELEASED): sha256_file(SB_RELEASED),
            "agent_safetybench_environment_manifest_sha256": canonical_sha256(env_hashes),
        },
        "environment_file_count": len(env_hashes),
        "manifest_content_sha256": manifest["manifest_sha256"],
        "forbidden_selection_fields_used": [],
    }
    write_json(manifest_path, manifest)
    write_json(audit_path, audit)
    return {"manifest": manifest, "audit": audit}


def build_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "agentdojo": _agentdojo_catalog(),
        "agent_safetybench": _safetybench_catalog(),
        "agent_security_bench": _asb_catalog(),
    }


def _payload_map(path: Path) -> dict[str, dict[str, Any]]:
    value = read_json(path)
    return {str(item["case_id"]): item for item in value["cases"]}


def _agentdojo_catalog() -> dict[str, dict[str, Any]]:
    from obligate.eval.agentdojo.runner import run_tool_firewall_eval as ad_runner

    plan = read_json(AD_PLAN)["cases"]
    sources = {str(item["case_id"]): item for item in read_jsonl(AD_SOURCES)}
    payloads = _payload_map(AD_R0)
    suites: dict[str, Any] = {}
    out: dict[str, dict[str, Any]] = {}
    deps = ad_runner._load_agentdojo_deps()
    for row in plan:
        case_id = str(row["case_id"])
        suite_name = str(row["suite"])
        suite = suites.setdefault(suite_name, deps["get_suite"]("v1.2.2", suite_name))
        user = suite.user_tasks[str(row["user_task_id"])]
        injection = suite.injection_tasks[str(row["injection_task_id"])]
        source = sources[case_id]
        frozen = payloads[case_id]
        out[case_id] = {
            "benchmark": "agentdojo",
            "case_id": case_id,
            "suite": suite_name,
            "user_task_id": str(row["user_task_id"]),
            "injection_task_id": str(row["injection_task_id"]),
            "user_task": str(getattr(user, "PROMPT", "")),
            "abstract_objective": str(getattr(injection, "GOAL", "")),
            "attack_channel": str(source["attack_channel"]),
            "visible_tools": list(source.get("tool_schemas") or []),
            "original_payload": str(source["injection_text"]).replace("\r\n", "\n").replace("\r", "\n").strip(),
            "round_0_payload": str(frozen["payload"]),
            "round_0_payload_sha256": str(frozen["payload_sha256"]),
            "original_payload_sha256": str(frozen["original_payload_sha256"]),
        }
    if len(out) != 949:
        raise RuntimeError(f"AgentDojo catalog expected 949 cases, got {len(out)}")
    return out


def _risk_bucket(case: dict[str, Any]) -> str:
    risks = [str(item) for item in case.get("risks") or []]
    if any("unsafe information / misinformation" in item.casefold() for item in risks):
        return "Unsafe information / misinformation (produce+spread combined)"
    return risks[0] if risks else "unknown"


def _safetybench_catalog() -> dict[str, dict[str, Any]]:
    released = {str(item["id"]): item for item in read_json(SB_RELEASED)}
    sources = {str(item["case_id"]): item for item in read_jsonl(SB_SOURCES)}
    payloads = _payload_map(SB_R0)
    out: dict[str, dict[str, Any]] = {}
    for case_id, case in released.items():
        source = sources[case_id]
        frozen = payloads[case_id]
        visible = list(source.get("environment_tool_schemas") or [])
        out[case_id] = {
            "benchmark": "agent_safetybench",
            "case_id": case_id,
            "risk_bucket": _risk_bucket(case),
            "risk_labels": list(case.get("risks") or []),
            "user_task": str(case.get("instruction") or ""),
            "abstract_objective": str(case.get("instruction") or ""),
            "attack_channel": str(source["attack_channel"]),
            "visible_tools": visible,
            "original_payload": str(source["instruction"]).strip(),
            "round_0_payload": str(frozen["payload"]),
            "round_0_payload_sha256": str(frozen["payload_sha256"]),
            "original_payload_sha256": str(frozen["original_payload_sha256"]),
        }
    if len(out) != 2000:
        raise RuntimeError(f"SafetyBench catalog expected 2000 cases, got {len(out)}")
    return out


def _safetybench_full_catalog() -> dict[str, dict[str, Any]]:
    released_items = read_json(SB_RELEASED)
    if len(released_items) != 2000:
        raise RuntimeError(f"SafetyBench released data expected 2000 cases, got {len(released_items)}")
    env_tools = _load_safetybench_environment_tools()
    out: dict[str, dict[str, Any]] = {}
    for raw_case in released_items:
        case = dict(raw_case)
        case_id = str(case.get("id"))
        if not case_id or case_id in out:
            raise RuntimeError(f"SafetyBench case id is missing or duplicated: {case_id!r}")
        runtime_payload, channel, dialog_meta = _safetybench_runtime_payload(case)
        visible_tools = _safetybench_visible_tools(case, env_tools)
        instruction = str(case.get("instruction") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        out[case_id] = {
            "benchmark": "agent_safetybench",
            "case_id": case_id,
            "risk_bucket": _risk_bucket(case),
            "risk_labels": list(case.get("risks") or []),
            "user_task": instruction,
            "abstract_objective": instruction,
            "attack_channel": channel,
            "visible_tools": visible_tools,
            "visible_tool_count": len(visible_tools),
            "environment_names": [str(item.get("name") or "") for item in case.get("environments") or []],
            "original_payload": runtime_payload,
            "round_0_payload": runtime_payload,
            "round_0_payload_sha256": sha256_text(runtime_payload),
            "original_payload_sha256": sha256_text(runtime_payload),
            **dialog_meta,
        }
    if len(out) != 2000:
        raise RuntimeError(f"SafetyBench full catalog expected 2000 cases, got {len(out)}")
    return out


def _safetybench_runtime_payload(case: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    dialog = case.get("dialog")
    if dialog is not None:
        if not isinstance(dialog, list):
            raise RuntimeError(f"SafetyBench case {case.get('id')} has non-list dialog")
        user_indexes = [
            index
            for index, message in enumerate(dialog)
            if isinstance(message, dict) and str(message.get("role") or "").casefold() == "user" and isinstance(message.get("content"), str)
        ]
        if len(user_indexes) != 1:
            raise RuntimeError(f"SafetyBench dialog case {case.get('id')} expected exactly one string user message, got {len(user_indexes)}")
        payload = str(dialog[user_indexes[0]]["content"]).replace("\r\n", "\n").replace("\r", "\n").strip()
        return (
            payload,
            "dialog_continuation",
            {
                "has_dialog": True,
                "dialog_user_message_count": len(user_indexes),
                "dialog_user_index": user_indexes[0],
                "dialog_role_sequence": [str(message.get("role") or "") if isinstance(message, dict) else "" for message in dialog],
            },
        )
    payload = str(case.get("instruction") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return (
        payload,
        "direct_request",
        {
            "has_dialog": False,
            "dialog_user_message_count": 0,
            "dialog_user_index": None,
            "dialog_role_sequence": [],
        },
    )


def _load_safetybench_environment_tools() -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(SB_ENV_DIR.glob("*.json")):
        tools = read_json(path)
        env_name = path.stem
        if not isinstance(tools, list):
            raise RuntimeError(f"SafetyBench environment file is not a tool list: {path}")
        by_name: dict[str, dict[str, Any]] = {}
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                raise RuntimeError(f"SafetyBench environment has malformed tool entry: {path}")
            by_name[str(tool["name"])] = tool
        out[env_name] = by_name
    if not out:
        raise RuntimeError(f"SafetyBench environment directory is empty: {SB_ENV_DIR}")
    return out


def _safetybench_visible_tools(case: dict[str, Any], env_tools: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for env in case.get("environments") or []:
        env_name = str(env.get("name") or "")
        tool_names = [str(name) for name in env.get("tools") or []]
        if not env_name and not tool_names:
            continue
        if env_name not in env_tools:
            raise RuntimeError(f"SafetyBench case {case.get('id')} references missing environment {env_name!r}")
        for tool_name in tool_names:
            key = (env_name, tool_name)
            if key in seen:
                continue
            if tool_name not in env_tools[env_name]:
                raise RuntimeError(f"SafetyBench case {case.get('id')} references missing tool {env_name}.{tool_name}")
            tool = deepcopy(env_tools[env_name][tool_name])
            tool["environment"] = env_name
            visible.append(tool)
            seen.add(key)
    return visible


def _safetybench_environment_hashes() -> dict[str, str]:
    return {
        _path_label(path): sha256_file(path)
        for path in sorted(SB_ENV_DIR.glob("*.json"))
    }


def _asb_catalog() -> dict[str, dict[str, Any]]:
    from experiments.agent_security_bench import asb_obligate_runner as asb

    sources = {str(item["case_id"]): item for item in read_jsonl(ASB_SOURCES)}
    payloads = _payload_map(ASB_R0)
    cases = asb.make_cases(
        upstream=ASB_UPSTREAM,
        attack_modes=["dpi", "opi", "mixed", "memory"],
        attack_types=["naive", "context_ignoring", "combined_attack"],
        task_num="all",
        attack_tool_type="all",
        limit=None,
        seed=20260714,
    )
    by_id = {case.case_id: case for case in cases}
    plan_ids = list(read_json(ASB_PLAN)["case_ids"])
    out: dict[str, dict[str, Any]] = {}
    for case_id in plan_ids:
        case = by_id[case_id]
        source = sources[case_id]
        frozen = payloads[case_id]
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.tool_schema()["function"]["parameters"],
            }
            for tool in asb._case_exposed_tools(case)
        ]
        out[case_id] = {
            "benchmark": "agent_security_bench",
            "case_id": case_id,
            "agent_name": case.agent_name,
            "attack_mode": case.attack_mode,
            "attack_type": case.attack_type,
            "user_task": case.task,
            "abstract_objective": str(case.attacker_tool.attacker_instruction),
            "attacker_tool_name": case.attacker_tool.name,
            "attack_channel": str(source["attack_channel"]),
            "visible_tools": tools,
            "original_payload": str(source["attack_payload"]).strip(),
            "round_0_payload": str(frozen["payload"]),
            "round_0_payload_sha256": str(frozen["payload_sha256"]),
            "original_payload_sha256": str(frozen["original_payload_sha256"]),
        }
    if len(out) != 8160:
        raise RuntimeError(f"ASB catalog expected 8160 cases, got {len(out)}")
    return out


def _sample_main(catalog: dict[str, dict[str, dict[str, Any]]], seed: int) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = [("agentdojo", case_id) for case_id in sorted(catalog["agentdojo"])]
    audit: dict[str, Any] = {"agentdojo": {"selection": "complete frozen 949-case plan", "case_count": 949}}

    by_risk: dict[str, list[str]] = defaultdict(list)
    for case_id, case in catalog["agent_safetybench"].items():
        by_risk[str(case["risk_bucket"])].append(case_id)
    if len(by_risk) != 7:
        raise RuntimeError(f"collapsed SafetyBench taxonomy expected 7 buckets, got {len(by_risk)}")
    sb_selected: list[str] = []
    for risk in sorted(by_risk):
        population = sorted(by_risk[risk])
        chosen = rng.sample(population, 30)
        sb_selected.extend(chosen)
    selected.extend(("agent_safetybench", case_id) for case_id in sb_selected)
    audit["agent_safetybench"] = {
        "selection": "seeded random sample over seven collapsed public risk buckets; no outcome fields",
        "risk_bucket_population_counts": {key: len(value) for key, value in sorted(by_risk.items())},
        "selected_per_risk": dict(Counter(catalog["agent_safetybench"][case_id]["risk_bucket"] for case_id in sb_selected)),
        "case_count": len(sb_selected),
    }

    asb = catalog["agent_security_bench"]
    agents = sorted({str(case["agent_name"]) for case in asb.values()})
    if len(agents) != 10:
        raise RuntimeError(f"ASB expected 10 agents, got {len(agents)}")
    opi: list[str] = []
    other: list[str] = []
    for agent in agents:
        population = sorted(case_id for case_id, case in asb.items() if case["agent_name"] == agent and case["attack_mode"] == "opi")
        opi.extend(rng.sample(population, 40))
    for mode in ("dpi", "memory", "mixed"):
        for agent in agents:
            population = sorted(case_id for case_id, case in asb.items() if case["agent_name"] == agent and case["attack_mode"] == mode)
            other.extend(rng.sample(population, 6))
    selected.extend(("agent_security_bench", case_id) for case_id in [*opi, *other])
    audit["agent_security_bench"] = {
        "selection": "seeded balanced random sample by public attack_mode and agent name; no scorer or outcome fields",
        "opi_selected_per_agent": dict(Counter(asb[case_id]["agent_name"] for case_id in opi)),
        "other_selected_per_mode_agent": dict(Counter(f"{asb[c]['attack_mode']}::{asb[c]['agent_name']}" for c in other)),
        "opi_case_count": len(opi),
        "other_case_count": len(other),
        "case_count": len(opi) + len(other),
    }
    if len(selected) != 1739:
        raise RuntimeError(f"main manifest expected 1739 cases, got {len(selected)}")
    return selected, audit


def _sample_stress(catalog: dict[str, dict[str, dict[str, Any]]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Check the confirmation input before indexing a potentially large stream.
    confirmation = _confirmation_pressure_cases()
    preflight = read_json(PREFLIGHT)["real_snapshot_calibration"]
    confirmed_conflict = sorted({str(item["case_id"]) for item in preflight.get("conflict_witnesses") or []})
    detected = _detect_snapshot_mechanisms()
    conflict = sorted(set(confirmed_conflict) | set(detected["conflict"]))
    # "All confirmed" refers to every case confirmed by the complete detector,
    # not only the first twenty witness rows retained by the earlier preflight report.
    rng = random.Random(seed + 91)
    assigned: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    def add(benchmark: str, ids: list[str], mechanism: str, cap: int | None = None) -> list[str]:
        population = [item for item in ids if (benchmark, item) not in assigned]
        if cap is not None and len(population) > cap:
            population = rng.sample(sorted(population), cap)
        for case_id in sorted(population):
            assigned.add((benchmark, case_id))
            row = dict(catalog[benchmark][case_id], segment="mechanism_stress", mechanism_category=mechanism)
            rows.append(row)
        return sorted(population)

    chosen_conflict = add("agentdojo", conflict, "conflict_divergence")
    chosen_gap = add("agentdojo", detected["pure_gap"], "pure_gap", 40)
    chosen_lifted = add("agentdojo", detected["lifted_join"], "lifted_join_divergence", 40)

    asb = catalog["agent_security_bench"]
    agents = sorted({str(case["agent_name"]) for case in asb.values()})
    opi_pool: list[str] = []
    for agent in agents:
        population = sorted(case_id for case_id, case in asb.items() if case["attack_mode"] == "opi" and case["agent_name"] == agent)
        opi_pool.extend(rng.sample(population, 8))
    chosen_opi = add("agent_security_bench", opi_pool, "opi_authorization_observability")

    chosen_confirmation = add("agentdojo", confirmation, "confirmation_pressure_unrecovered", 40)
    audit = {
        "selection_uses_preflight_or_prior_diagnostic_outputs": True,
        "selection_scope": "diagnostic mechanism stress only; excluded from main population CASR",
        "deduplication_priority": [
            "conflict_divergence",
            "pure_gap",
            "lifted_join_divergence",
            "opi_authorization_observability",
            "confirmation_pressure_unrecovered",
        ],
        "confirmed_conflict_case_count": len(chosen_conflict),
        "pure_gap_case_count": len(chosen_gap),
        "lifted_join_case_count": len(chosen_lifted),
        "opi_authorization_observability_case_count": len(chosen_opi),
        "confirmation_pressure_unrecovered_case_count": len(chosen_confirmation),
        "deduplicated_total_case_count": len(rows),
        "detector": detected["audit"],
    }
    return rows, audit


def _detect_snapshot_mechanisms() -> dict[str, Any]:
    policy, full_runtime, conflict_runtime, no_join_runtime = _runtimes()
    conflict: set[str] = set()
    pure_gap: set[str] = set()
    lifted: set[str] = set()
    evaluated = 0
    if not SNAPSHOT_STREAM.is_file():
        raise RuntimeError(
            "missing Full event snapshot stream; prepare adaptive inputs or set "
            "OBLIGATE_STRESS_SNAPSHOT_STREAM"
        )
    with SnapshotEventIndex.build(SNAPSHOT_STREAM, SNAPSHOT_INDEX) as index:
        for record in index.records():
            source = index.load(record.input_offset, record.input_length)
            if not source:
                continue
            evaluated += 1
            action = deserialize_action(source["action"])
            evidence = activate_runtime_evidence(action.tool_name, deserialize_evidence(source["evidence"]))
            case_id = str(source.get("case_id") or "unknown")
            direct_conflicts = _direct_conflict_keys(evidence)
            full = full_runtime.decide(action, evidence)
            gap = bool((full.trace or {}).get("gap"))
            if gap and not direct_conflicts:
                pure_gap.add(case_id)
            if direct_conflicts:
                variant = conflict_runtime.decide(action, evidence)
                if (full.public_decision, full.execute) != (variant.public_decision, variant.execute):
                    conflict.add(case_id)
            controls = {
                item.preferred_control
                for item in evidence.hazards
                if not item.hard and item.preferred_control in {"sandbox", "redact", "require_confirmation"}
            }
            if len(controls) >= 2:
                local = _local_remedy_audit(full, policy)
                local_digests = set(local["feasible_local_plan_digests"])
                selected_is_joined = bool(full.selected) and full.selected.plan.digest not in local_digests
                if len(local_digests) >= 2 and selected_is_joined:
                    variant = no_join_runtime.decide(action, evidence)
                    if (full.public_decision, full.execute) != (variant.public_decision, variant.execute):
                        lifted.add(case_id)
    return {
        "conflict": sorted(conflict),
        "pure_gap": sorted(pure_gap),
        "lifted_join": sorted(lifted),
        "audit": {
            "input_snapshots_evaluated": evaluated,
            "conflict_unique_cases": len(conflict),
            "pure_gap_unique_cases": len(pure_gap),
            "lifted_join_unique_cases": len(lifted),
            "snapshot_stream_population_sha256": read_json(PREFLIGHT)["real_snapshot_calibration"].get("population_sha256"),
            "snapshot_stream_size": SNAPSHOT_STREAM.stat().st_size,
            "snapshot_stream_mtime_ns": SNAPSHOT_STREAM.stat().st_mtime_ns,
            "snapshot_index_sha256": sha256_file(SNAPSHOT_INDEX),
            "uses_benchmark_labels_or_scorer_fields": False,
        },
    }


def _confirmation_pressure_cases() -> list[str]:
    raw_dir = CONFIRMATION_RAW_DIR
    paths = sorted(raw_dir.glob("*.json")) if raw_dir.is_dir() else []
    if not paths:
        raise RuntimeError(
            "missing Full confirmation raw runs; set OBLIGATE_STRESS_CONFIRMATION_RAW_DIR "
            "to a completed RQ3 Full cell's raw_runs directory"
        )
    selected: list[str] = []
    for path in paths:
        value = read_json(path)
        rows = value.get("per_run") or []
        if len(rows) != 1:
            raise RuntimeError(f"Full confirmation raw run must contain one case: {path}")
        metadata = value.get("obligate_ablation_v2") or {}
        if metadata and metadata.get("variant") != "full":
            raise RuntimeError(f"confirmation reference must use Full, not another variant: {path}")
        case_id = str(metadata.get("case_id") or value.get("case_id") or "")
        if not case_id:
            name = str(value.get("run_name") or "")
            if "_obligate_strict" not in name:
                raise RuntimeError(f"Full confirmation raw run lacks a recognized case identifier: {path}")
            case_id = name.split("_obligate_strict", 1)[0]
        run = rows[0]
        confirmation = int(run.get("confirmation_required_count") or 0)
        if confirmation > 0 and not bool(run.get("recovery_success")):
            selected.append(case_id)
    return sorted(set(selected))


def _manifest(name: str, cases: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    clean_cases = sorted(cases, key=lambda row: (row["benchmark"], row["case_id"], row.get("mechanism_category", "")))
    value = {
        "schema_version": "obligate-adaptive-case-manifest-v2",
        "name": name,
        "sampling_seed": seed,
        "case_count": len(clean_cases),
        "benchmark_counts": dict(Counter(item["benchmark"] for item in clean_cases)),
        "cases": clean_cases,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _write_old_artifact_inventory() -> None:
    output = RESULT_ROOT / "old_artifact_inventory_before.json"
    if output.exists():
        return
    protected = [
        PREFLIGHT,
        AD_R0,
        SB_R0,
        ASB_R0,
    ]
    trees = []
    for root in (OLD_FORMAL, FROZEN_INPUT_DIR):
        files = [path for path in root.rglob("*") if path.is_file()]
        trees.append(
            {
                "path": str(root.resolve()),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "latest_mtime_ns": max((path.stat().st_mtime_ns for path in files), default=0),
            }
        )
    value = {
        "schema_version": "obligate-reference-input-read-only-inventory-v1",
        "protected_trees": trees,
        "protected_control_file_hashes": {str(path.resolve()): sha256_file(path) for path in protected if path.exists()},
        "missing_control_files": [str(path.resolve()) for path in protected if not path.exists()],
        "mutation_authorized": False,
    }
    write_json(output, value)


if __name__ == "__main__":
    result = build_manifests(force=False)
    print(json.dumps({key: value.get("case_count") for key, value in result.items() if isinstance(value, dict)}, ensure_ascii=False))
