from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_sha256, read_json, sha256_text, write_json

SAFETYBENCH_RELEASED = (
    Path(
        os.environ.get("SAFETYBENCH_UPSTREAM")
        or ROOT / "third_party" / "Agent-SafetyBench"
    ).resolve()
    / "data"
    / "released_data.json"
)


def round_payload_manifest(
    cases: list[dict[str, Any]],
    payload_rows: dict[str, dict[str, Any]],
    output: Path,
    *,
    round_index: int,
    victim_model: str,
    attacker_seed: int,
    segment: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        payload_row = payload_rows[case_id]
        payload = str(payload_row["payload"])
        rows.append(
            {
                "case_id": case_id,
                "benchmark": case["benchmark"],
                "segment": segment,
                "round": round_index,
                "victim_model": victim_model,
                "attacker_seed": attacker_seed,
                "payload": payload,
                "payload_sha256": sha256_text(payload),
                "original_payload_sha256": case["original_payload_sha256"],
                "round_0_payload_sha256": case["round_0_payload_sha256"],
                "primary_strategy": payload_row.get("primary_strategy"),
                "secondary_strategy": payload_row.get("secondary_strategy"),
                "public_feedback_class": payload_row.get("public_feedback_class", "round_0_frozen"),
                "operators": payload_row.get("operators", []),
                "source_payload_round": payload_row.get("round", round_index),
            }
        )
    manifest = {
        "schema_version": "obligate-round-payload-manifest-v2",
        "experiment_name": "ObliGate adaptive_closed_loop_v2",
        "segment": segment,
        "round": round_index,
        "victim_model": victim_model,
        "attacker_seed": attacker_seed,
        "case_count": len(rows),
        "cases": rows,
    }
    manifest["manifest_sha256"] = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    write_json(output, manifest)
    return manifest


def agentdojo_case_plan(cases: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows = [
        {
            "case_id": str(case["case_id"]),
            "suite": str(case["suite"]),
            "user_task_id": str(case["user_task_id"]),
            "injection_task_id": str(case["injection_task_id"]),
        }
        for case in cases
        if case["benchmark"] == "agentdojo"
    ]
    value = {
        "schema_version": "obligate-agentdojo-case-plan-v2",
        "case_count": len(rows),
        "cases": rows,
    }
    write_json(output, value)
    return value


def safetybench_data(cases: list[dict[str, Any]], payload_rows: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
    selected = {str(case["case_id"]): case for case in cases if case["benchmark"] == "agent_safetybench"}
    released = read_json(SAFETYBENCH_RELEASED)
    by_id = {str(item.get("id")): item for item in released}
    rows: list[dict[str, Any]] = []
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise ValueError(f"SafetyBench selected IDs missing from released data: {missing[:5]}")
    for case_id in selected:
        item = copy.deepcopy(by_id[case_id])
        payload = str(payload_rows[case_id]["payload"])
        payload_location = inject_safetybench_payload(item, payload, case_id=case_id)
        item["_obligate_case_id"] = case_id
        item["_adaptive_payload_sha256"] = sha256_text(payload)
        item["_obligate_round"] = payload_rows[case_id].get("round")
        item["_obligate_payload_location"] = payload_location["location"]
        item["_obligate_payload_user_index"] = payload_location.get("user_index")
        item["_obligate_materialized_payload_sha256"] = sha256_text(_materialized_payload_text(item, payload_location))
        rows.append(item)
    write_json(output, rows)
    return {"case_count": len(rows), "data": str(output)}


def inject_safetybench_payload(item: dict[str, Any], payload: str, *, case_id: str) -> dict[str, Any]:
    if "dialog" not in item:
        item["instruction"] = payload
        return {"location": "instruction"}
    dialog = item.get("dialog")
    if not isinstance(dialog, list):
        raise ValueError(f"SafetyBench dialog case {case_id} has non-list dialog")
    user_indexes = [
        index
        for index, message in enumerate(dialog)
        if isinstance(message, dict) and str(message.get("role") or "").casefold() == "user" and isinstance(message.get("content"), str)
    ]
    if len(user_indexes) != 1:
        raise ValueError(f"SafetyBench dialog case {case_id} expected exactly one string user message, got {len(user_indexes)}")
    user_index = user_indexes[0]
    dialog[user_index]["content"] = payload
    item["instruction"] = payload
    return {"location": "dialog_user", "user_index": user_index}


def _materialized_payload_text(item: dict[str, Any], payload_location: dict[str, Any]) -> str:
    if payload_location["location"] == "dialog_user":
        return str((item.get("dialog") or [])[int(payload_location["user_index"])]["content"])
    return str(item.get("instruction") or "")


def asb_case_ids(cases: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    case_ids = [str(case["case_id"]) for case in cases if case["benchmark"] == "agent_security_bench"]
    value = {
        "schema_version": "obligate-asb-case-selection-v2",
        "case_count": len(case_ids),
        "case_ids": case_ids,
    }
    write_json(output, value)
    return value
