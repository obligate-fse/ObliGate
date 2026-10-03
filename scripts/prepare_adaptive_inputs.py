"""Normalize explicitly selected standard runs and freeze adaptive public inputs.

No provider is called. All source runs must use the requested model and ObliGate;
the complete populations and trace availability are checked before output writes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _flag(command: list[str], flag: str) -> list[str]:
    if flag not in command:
        return []
    result = []
    for item in command[command.index(flag) + 1:]:
        if item.startswith("--"):
            break
        result.append(item)
    return result


def _registry_run(directory: Path, benchmark: str, model: str) -> dict[str, Any]:
    value = _read(directory / "manifest.json")
    launch = value.get("launch") or {}
    if (value.get("benchmark_spec") or {}).get("benchmark") != benchmark:
        raise ValueError(f"wrong benchmark run directory: {directory}")
    if launch.get("model") != model:
        raise ValueError(f"run model differs from selected model {model}: {directory}")
    result = _read(directory / "result.json")
    if result.get("status") != "succeeded" or result.get("exit_code") != 0:
        raise ValueError(f"standard run is not completed successfully: {directory}")
    return value


def normalize_runs(*, model: str, agentdojo_dirs: list[Path], safetybench_dir: Path,
                   asb_dir: Path, output: Path, agentdojo_plan: Path, asb_plan: Path,
                   safetybench_count: int = 2000) -> dict[str, str]:
    """Convert registry outputs to prepare_payloads' frozen Round-1 layout."""
    if output.exists():
        raise ValueError(f"refusing to overwrite adaptive inputs: {output}")
    plan = _read(agentdojo_plan)["cases"]
    expected = {(str(row["suite"]), str(row["user_task_id"]), str(row["injection_task_id"])):
                str(row["case_id"]) for row in plan}
    if len(expected) != len(plan):
        raise ValueError("AgentDojo plan contains duplicate case keys")
    selected: dict[str, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    provenance: list[dict[str, Any]] = []
    for directory in agentdojo_dirs:
        manifest = _registry_run(directory, "agentdojo", model)
        native_path = directory / f"{manifest['run_id']}.json"
        native = _read(native_path)
        if (native.get("defense"), native.get("attack"), native.get("benchmark_version")) != (
                "obligate", "important_instructions", "v1.2.2"):
            raise ValueError(f"AgentDojo input must be full attacked ObliGate: {directory}")
        if native.get("model") != model:
            raise ValueError(f"native AgentDojo model mismatch: {directory}")
        for row in native.get("per_run") or []:
            key = (str(row.get("suite")), str(row.get("user_task_id")), str(row.get("injection_task_id")))
            case_id = expected.get(key)
            if case_id is None or case_id in selected:
                raise ValueError(f"unexpected or duplicate AgentDojo case: {key}")
            trace = Path(str(row.get("trace_file") or ""))
            if not trace.is_absolute():
                trace = Path(manifest["launch"]["cwd"]) / trace
            if not trace.is_file() or not row.get("injections"):
                raise ValueError(f"case needs --save-full-trace and injection text: {case_id}")
            selected[case_id] = (native, row, trace)
        provenance.append({"source": str(native_path.resolve()), "sha256": _hash(native_path)})
    if set(selected) != set(expected.values()):
        raise ValueError(f"AgentDojo population incomplete: {len(selected)} of {len(expected)}")
    sb_manifest = _registry_run(safetybench_dir, "agent_safetybench", model)
    if _flag(sb_manifest["launch"]["command"], "--defense") != ["obligate_visible_fair"]:
        raise ValueError("SafetyBench input must use obligate_visible_fair")
    sb_path = safetybench_dir / "gen_res.json"
    sb = _read(sb_path)
    sb_ids = [str(row["id"]) for row in sb]
    if len(sb_ids) != safetybench_count or len(set(sb_ids)) != safetybench_count:
        raise ValueError("SafetyBench population incomplete or contains duplicate IDs")
    asb_manifest = _registry_run(asb_dir, "asb_iclr2025", model)
    if _flag(asb_manifest["launch"]["command"], "--modes") != ["obligate_registry_blind"]:
        raise ValueError("ASB input must use obligate_registry_blind")
    asb_path = asb_dir / "records.jsonl"
    asb_ids = []
    with asb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("mode") != "obligate_registry_blind":
                    raise ValueError("ASB records mix defense modes")
                asb_ids.append(str(row["case_id"]))
    expected_asb = _read(asb_plan)["case_ids"]
    if len(asb_ids) != len(set(asb_ids)) or set(asb_ids) != set(expected_asb):
        raise ValueError("ASB population does not match the frozen case plan")
    # Everything above is read-only; reject model/defense/population drift first.
    for case_id, (native, row, source_trace) in sorted(selected.items()):
        trace = output / "round1" / "agentdojo" / "traces" / f"{case_id}.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_trace, trace)
        normalized_row = deepcopy(row)
        normalized_row["trace_file"] = str(trace.resolve())
        provenance.append({"source": str(source_trace.resolve()), "sha256": _hash(source_trace)})
        raw = {key: native[key] for key in ("model", "defense", "attack", "benchmark_version", "suite") if key in native}
        raw.update(case_id=case_id, run_name=f"{case_id}_obligate_strict", per_run=[normalized_row])
        _write(output / "round1" / "agentdojo" / "raw_runs" / f"{case_id}_obligate_strict.json", raw)
    for label, source in (("agent_safetybench", sb_path), ("agent_security_bench", asb_path)):
        destination = output / "round1" / label / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        provenance.append({"source": str(source.resolve()), "sha256": _hash(source)})
    _write(output / "round1_normalization.json", {
        "schema_version": 1, "model": model, "defense": "obligate", "sources": provenance,
        "agentdojo_cases": len(selected), "safetybench_cases": len(sb_ids), "asb_cases": len(asb_ids),
        "outcomes_modified": False,
    })
    return {
        "OBLIGATE_ADAPTIVE_INPUT_ROOT": str(output.resolve()),
        "OBLIGATE_AGENTDOJO_R1_ROOT": str((output / "round1/agentdojo").resolve()),
        "OBLIGATE_SAFETYBENCH_R1_ROOT": str((output / "round1/agent_safetybench").resolve()),
        "OBLIGATE_ASB_R1_ROOT": str((output / "round1/agent_security_bench").resolve()),
    }


def validate_full_reference(snapshot_dir: Path, raw_dir: Path, model: str,
                            expected_case_ids: set[str] | None = None) -> list[Path]:
    """Validate the selected Full reference before creating an output tree."""
    streams = sorted(snapshot_dir.glob("*.jsonl"))
    raw_paths = sorted(raw_dir.glob("*.json"))
    if not streams or not raw_paths:
        raise ValueError("completed RQ3 Full snapshots and raw_runs are required")
    case_ids = []
    for path in raw_paths:
        raw = _read(path)
        metadata = raw.get("obligate_ablation_v2") or {}
        if metadata.get("variant") != "full" or metadata.get("model") != model:
            raise ValueError(f"Full reference model/variant mismatch: {path}")
        if len(raw.get("per_run") or []) != 1 or not metadata.get("case_id"):
            raise ValueError(f"Full reference must contain one identified case: {path}")
        case_ids.append(str(metadata["case_id"]))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Full reference contains duplicate cases")
    if expected_case_ids is not None and (
            set(case_ids) != expected_case_ids or {path.stem for path in streams} != expected_case_ids):
        raise ValueError("Full reference raw runs/snapshots do not cover the frozen AgentDojo population")
    return streams


def prepare_full_events(snapshot_dir: Path, raw_dir: Path, output: Path, model: str,
                        expected_case_ids: set[str] | None = None) -> dict[str, str]:
    """Keep raw Full events for activation calibration and diagnostic selection."""
    streams = validate_full_reference(snapshot_dir, raw_dir, model, expected_case_ids)
    stream = output / "agentdojo_snapshots.jsonl"
    if stream.exists():
        raise ValueError(f"refusing to overwrite Full stream: {stream}")
    with stream.open("wb") as target:
        for source in streams:
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, target)
                if source.stat().st_size:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        target.write(b"\n")
    _write(output / "full_reference_provenance.json", {
        "schema_version": 1, "model": model, "variant": "full",
        "sources": [{"source": str(path.resolve()), "sha256": _hash(path)} for path in streams],
        "raw_runs_directory": str(raw_dir.resolve()), "stream_sha256": _hash(stream),
    })
    return {
        "OBLIGATE_PREFLIGHT_SNAPSHOT_DIR": str(snapshot_dir.resolve()),
        "OBLIGATE_MECHANISM_PREFLIGHT": str((output / "mechanism_activation_preflight.json").resolve()),
        "OBLIGATE_STRESS_SNAPSHOT_STREAM": str(stream.resolve()),
        "OBLIGATE_STRESS_SNAPSHOT_INDEX": str((output / "agentdojo_snapshot_index.sqlite3").resolve()),
        "OBLIGATE_STRESS_CONFIRMATION_RAW_DIR": str(raw_dir.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--agentdojo-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--safetybench-run-dir", type=Path, required=True)
    parser.add_argument("--asb-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agentdojo-plan", type=Path, default=ROOT / "data/agentdojo_v1.2.2_949_case_plan.json")
    parser.add_argument("--asb-plan", type=Path, default=ROOT / "data/asb_iclr2025_8160_case_plan.json")
    parser.add_argument("--full-snapshot-dir", type=Path, required=True)
    parser.add_argument("--full-raw-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    full_case_ids = {str(row["case_id"]) for row in _read(args.agentdojo_plan)["cases"]}
    validate_full_reference(args.full_snapshot_dir, args.full_raw_dir, args.model, full_case_ids)
    exports = normalize_runs(model=args.model, agentdojo_dirs=args.agentdojo_run_dirs,
                            safetybench_dir=args.safetybench_run_dir, asb_dir=args.asb_run_dir,
                            output=args.output_dir, agentdojo_plan=args.agentdojo_plan, asb_plan=args.asb_plan)
    exports.update(prepare_full_events(args.full_snapshot_dir, args.full_raw_dir, args.output_dir,
                                      args.model, full_case_ids))
    environment = dict(os.environ, **exports)
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", "")])
    for module, arguments in (
        ("experiments.adaptive_ablation.prepare_payloads", ["--output-dir", str(args.output_dir)]),
        ("experiments.adaptive_ablation.mechanism_activation_preflight", ["--agentdojo-snapshot-dir", str(args.full_snapshot_dir), "--output", exports["OBLIGATE_MECHANISM_PREFLIGHT"]]),
    ):
        subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, env=environment, check=True)
    _write(args.output_dir / "adaptive_environment.json", exports)
    (args.output_dir / "adaptive_environment.sh").write_text(
        "# Generated paths for this frozen input population.\n" +
        "".join(f"export {key}={shlex.quote(value)}\n" for key, value in sorted(exports.items())),
        encoding="utf-8",
    )
    print(json.dumps(exports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
