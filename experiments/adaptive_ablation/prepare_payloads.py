"""Build and freeze the complete public Round-2 payload population.

The generated source JSONL contains only the case identifier used for the
post-compilation join and fields explicitly listed in ``ATTACKER_VISIBLE_FIELDS``.
Round-1 feedback is stored separately as one coarse public class per case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .feedback import agentdojo_feedback, asb_feedback, safetybench_feedback
from .freeze_payloads import freeze_records, write_manifest
from .schema import Benchmark, PublicFeedbackClass

ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).resolve()


AD_R1 = _env_path(
    "OBLIGATE_AGENTDOJO_R1_ROOT",
    ROOT / "reproduced" / "main_agentdojo",
)
SB_R1 = _env_path(
    "OBLIGATE_SAFETYBENCH_R1_ROOT",
    ROOT / "reproduced" / "main_cross_benchmark" / "agent_safetybench",
)
ASB_R1 = _env_path(
    "OBLIGATE_ASB_R1_ROOT",
    ROOT / "reproduced" / "main_cross_benchmark" / "asb_iclr2025",
)
AD_PLAN = _env_path(
    "OBLIGATE_AGENTDOJO_CASE_PLAN",
    ROOT / "data" / "agentdojo_v1.2.2_949_case_plan.json",
)
SAFETYBENCH_UPSTREAM = _env_path(
    "SAFETYBENCH_UPSTREAM",
    ROOT / "third_party" / "Agent-SafetyBench",
)
ASB_UPSTREAM = _env_path(
    "ASB_UPSTREAM",
    ROOT / "third_party" / "ASB",
)
ASB_PLAN = _env_path(
    "OBLIGATE_ASB_CASE_PLAN",
    ROOT / "data" / "asb_iclr2025_8160_case_plan.json",
)


def prepare_all(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        Benchmark.AGENTDOJO: build_agentdojo_sources,
        Benchmark.AGENT_SAFETYBENCH: build_safetybench_sources,
        Benchmark.ASB: build_asb_sources,
    }
    expected = {
        Benchmark.AGENTDOJO: 949,
        Benchmark.AGENT_SAFETYBENCH: 2_000,
        Benchmark.ASB: 8_160,
    }
    index: dict[str, Any] = {"schema_version": 1, "benchmarks": {}}
    total = 0
    for benchmark, builder in builders.items():
        rows, feedback = builder()
        _validate_population(rows, feedback, expected=expected[benchmark], benchmark=benchmark)
        source_path = output_dir / f"{benchmark.value}_public_sources.jsonl"
        feedback_path = output_dir / f"{benchmark.value}_round1_public_feedback.jsonl"
        manifest_path = output_dir / f"{benchmark.value}_round2_payload_manifest.json"
        _write_jsonl(source_path, rows)
        _write_jsonl(
            feedback_path,
            (
                {"case_id": case_id, "public_feedback_class": value.value}
                for case_id, value in feedback.items()
            ),
        )
        manifest = freeze_records(
            benchmark=benchmark,
            records=rows,
            feedback_by_case=feedback,
        )
        write_manifest(manifest_path, manifest)
        total += len(rows)
        index["benchmarks"][benchmark.value] = {
            "case_count": len(rows),
            "public_sources": str(source_path),
            "public_sources_sha256": _sha256_file(source_path),
            "round1_public_feedback": str(feedback_path),
            "round1_public_feedback_sha256": _sha256_file(feedback_path),
            "payload_manifest": str(manifest_path),
            "payload_manifest_sha256": _sha256_file(manifest_path),
            "payload_manifest_content_sha256": manifest["manifest_sha256"],
            "feedback_counts": _counts(value.value for value in feedback.values()),
        }
    if total != 11_109:
        raise ValueError(f"formal population must contain 11,109 cases, got {total}")
    index["case_count"] = total
    index["new_victim_episode_budget"] = {
        "round2_full": 11_109,
        "agentdojo_e2e_ablations": 3 * 949,
        "total": 13_956,
        "optional_qwen_enabled": False,
        "optional_qwen_env_gate": "RUN_OPTIONAL_QWEN_TRANSFER=1",
    }
    index_path = output_dir / "payload_freeze_index.json"
    index["index_sha256"] = _sha256_json(index)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def build_agentdojo_sources() -> tuple[list[dict[str, Any]], dict[str, PublicFeedbackClass]]:
    try:
        from agentdojo.benchmark import get_suite
    except ImportError as exc:  # pragma: no cover - exercised by the documented venv command.
        raise RuntimeError(
            "run AgentDojo preparation with OBLIGATE_AGENTDOJO_PYTHON"
        ) from exc

    plan = json.loads(
        AD_PLAN.read_text(encoding="utf-8")
    )
    raw_by_name: dict[str, Mapping[str, Any]] = {}
    for path in (AD_R1 / "raw_runs").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        raw_by_name[str(value.get("run_name"))] = value
    schema_by_suite: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    feedback: dict[str, PublicFeedbackClass] = {}
    for item in plan["cases"]:
        case_id = str(item["case_id"])
        suite_name = str(item["suite"])
        run_name = f"{case_id}_obligate_strict"
        raw = raw_by_name.get(run_name)
        if raw is None:
            raise ValueError(f"missing AgentDojo Round-1 raw run: {run_name}")
        per_run = raw.get("per_run")
        if not isinstance(per_run, list) or len(per_run) != 1:
            raise ValueError(f"AgentDojo run does not contain exactly one case: {run_name}")
        run = per_run[0]
        injections = run.get("injections")
        if not isinstance(injections, Mapping) or not injections:
            raise ValueError(f"AgentDojo run lacks public injection text: {run_name}")
        rendered = sorted({str(value) for value in injections.values()})
        injection_text = "\n\n".join(rendered)
        if suite_name not in schema_by_suite:
            suite = get_suite("v1.2.2", suite_name)
            schema_by_suite[suite_name] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_json_schema(),
                }
                for tool in suite.tools
            ]
        rows.append(
            {
                "case_id": case_id,
                "attack_channel": "tool_output",
                "injection_text": injection_text,
                "tool_schemas": schema_by_suite[suite_name],
            }
        )
        trace_path = Path(str(run.get("trace_file") or ""))
        if not trace_path.is_file():
            raise ValueError(f"missing AgentDojo Round-1 trace: {trace_path}")
        feedback[case_id] = agentdojo_feedback(json.loads(trace_path.read_text(encoding="utf-8")))
    return rows, feedback


def build_safetybench_sources() -> tuple[list[dict[str, Any]], dict[str, PublicFeedbackClass]]:
    upstream = SAFETYBENCH_UPSTREAM
    released = json.loads((upstream / "data/released_data.json").read_text(encoding="utf-8"))
    round1 = json.loads((SB_R1 / "gen_res.json").read_text(encoding="utf-8"))
    r1_by_id = {str(item["id"]): item for item in round1}
    environment_cache: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    feedback: dict[str, PublicFeedbackClass] = {}
    for case in released:
        case_id = str(case["id"])
        selected_schemas: list[dict[str, Any]] = []
        for env in case.get("environments") or []:
            name = str(env.get("name") or "")
            if not name:
                # Some released rows deliberately have no tool environment.
                continue
            if name not in environment_cache:
                environment_cache[name] = json.loads(
                    (upstream / "environments" / f"{name}.json").read_text(encoding="utf-8")
                )
            requested = {str(value) for value in env.get("tools") or []}
            selected_schemas.extend(
                schema for schema in environment_cache[name] if str(schema.get("name")) in requested
            )
        rows.append(
            {
                "case_id": case_id,
                "attack_channel": "direct_request",
                "instruction": str(case.get("instruction") or ""),
                "environment_tool_schemas": selected_schemas,
            }
        )
        if case_id not in r1_by_id:
            raise ValueError(f"missing SafetyBench Round-1 trace row: {case_id}")
        feedback[case_id] = safetybench_feedback(r1_by_id[case_id])
    return rows, feedback


def build_asb_sources() -> tuple[list[dict[str, Any]], dict[str, PublicFeedbackClass]]:
    from experiments.agent_security_bench import asb_obligate_runner as runner

    plan_value = json.loads(
        ASB_PLAN.read_text(encoding="utf-8")
    )
    case_ids = [str(value) for value in plan_value["case_ids"]]
    generated = runner.make_cases(
        upstream=ASB_UPSTREAM,
        attack_modes=["dpi", "opi", "mixed", "memory"],
        attack_types=["naive", "context_ignoring", "combined_attack"],
        task_num="all",
        attack_tool_type="all",
        limit=None,
        seed=20260714,
    )
    by_id = {case.case_id: case for case in generated}
    r1_by_id: dict[str, Mapping[str, Any]] = {}
    with (ASB_R1 / "records.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            r1_by_id[str(value["case_id"])] = value
    rows: list[dict[str, Any]] = []
    feedback: dict[str, PublicFeedbackClass] = {}
    for case_id in case_ids:
        case = by_id.get(case_id)
        if case is None:
            raise ValueError(f"ASB case-plan ID is unavailable: {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "attack_channel": _asb_channel(case.attack_mode),
                "attack_payload": runner.attack_prompt(case),
                "attacker_tool_name": case.attacker_tool.name,
                "attacker_tool_description": case.attacker_tool.description,
                "attacker_tool_schema": case.attacker_tool.tool_schema(),
            }
        )
        record = r1_by_id.get(case_id)
        if record is None:
            raise ValueError(f"missing ASB Round-1 trace row: {case_id}")
        feedback[case_id] = asb_feedback(record)
    return rows, feedback


def _asb_channel(attack_mode: str) -> str:
    return {
        "dpi": "direct_request",
        "opi": "tool_output",
        "mixed": "tool_output",
        "memory": "memory",
    }.get(attack_mode, "dynamic_tool_description")


def _validate_population(
    rows: list[dict[str, Any]],
    feedback: Mapping[str, PublicFeedbackClass],
    *,
    expected: int,
    benchmark: Benchmark,
) -> None:
    ids = [str(row.get("case_id") or "") for row in rows]
    if len(ids) != expected or len(set(ids)) != expected or any(not value for value in ids):
        raise ValueError(f"{benchmark.value}: expected {expected} unique cases, got {len(ids)}/{len(set(ids))}")
    if set(ids) != set(feedback):
        raise ValueError(f"{benchmark.value}: Round-1 feedback does not align with the case plan")
    if any(value is PublicFeedbackClass.NO_FEEDBACK for value in feedback.values()):
        raise ValueError(f"{benchmark.value}: compatible Round-1 trace unexpectedly produced no_feedback")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_env_path(
            "OBLIGATE_ADAPTIVE_INPUT_ROOT",
            ROOT / "reproduced" / "adaptive_inputs",
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index = prepare_all(args.output_dir)
    print(json.dumps({"case_count": index["case_count"], "index_sha256": index["index_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
