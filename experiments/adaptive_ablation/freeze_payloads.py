"""Freeze one-shot payloads and all required reproducibility hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Iterable, Mapping

from .adapters import adapt_visible_input
from .compiler import TEMPLATE_VERSION, AttackCompiler, template_registry_sha256
from .schema import (
    ATTACKER_VISIBLE_FIELDS,
    ATTACKER_VISIBLE_FIELDS_VERSION,
    EXPERIMENT_MODE,
    SCHEMA_VERSION,
    Benchmark,
    CasePayloadBinding,
    AttackerChannel,
    CompiledPayload,
    CompileStatus,
    FORMAL_EXPERIMENT_NAME,
    PublicFeedbackClass,
    MechanismOperator,
    ToolSchemaClass,
    canonical_json,
)

PACKAGE_DIR = Path(__file__).resolve().parent
COMPILER_CODE_FILES = (
    PACKAGE_DIR / "schema.py",
    PACKAGE_DIR / "adapters.py",
    PACKAGE_DIR / "compiler.py",
)


def bind_case(*, case_id: str, benchmark: Benchmark | str, compiled: CompiledPayload) -> CasePayloadBinding:
    """Associate a case identifier only after payload generation."""

    normalized = str(case_id).strip()
    if not normalized:
        raise ValueError("case_id is required for post-compilation manifest association")
    return CasePayloadBinding(case_id=normalized, benchmark=Benchmark(benchmark), compiled=compiled)


def build_manifest(
    bindings: Iterable[CasePayloadBinding], *, compiler_isolated: bool = False
) -> dict[str, Any]:
    items = list(bindings)
    case_ids = [item.case_id for item in items]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("payload manifest contains duplicate case IDs")
    cases = [
        {
            "case_id": item.case_id,
            "benchmark": item.benchmark.value,
            **item.compiled.as_dict(include_payload=True),
        }
        for item in items
    ]
    base = {
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": EXPERIMENT_MODE,
        "formal_experiment_name": FORMAL_EXPERIMENT_NAME,
        "public_feedback_class_counts": dict(
            sorted(Counter(item.compiled.public_feedback_class.value for item in items).items())
        ),
        "case_count": len(cases),
        "template_version": TEMPLATE_VERSION,
        "template_registry_sha256": template_registry_sha256(),
        "compiler_code_sha256": compiler_code_sha256(),
        "compiler_process_isolated": compiler_isolated,
        "compiler_worker_protocol": (
            "obligate-perturbation-compiler-worker-v1" if compiler_isolated else None
        ),
        "attacker_visible_fields_version": ATTACKER_VISIBLE_FIELDS_VERSION,
        "perturbation_visible_fields": {
            benchmark.value: list(fields) for benchmark, fields in ATTACKER_VISIBLE_FIELDS.items()
        },
        "perturbation_visible_fields_sha256": attacker_visible_fields_sha256(),
        "cases": cases,
    }
    base["manifest_sha256"] = hashlib.sha256(canonical_json(base).encode("utf-8")).hexdigest()
    return base


def freeze_records(
    *,
    benchmark: Benchmark | str,
    records: Iterable[Mapping[str, Any]],
    compiler: AttackCompiler | None = None,
    feedback_by_case: Mapping[str, PublicFeedbackClass | str] | None = None,
) -> dict[str, Any]:
    """Adapt, compile, then bind each record without exposing its ID to the compiler."""

    selected = Benchmark(benchmark)
    active_compiler = compiler
    pending: list[tuple[str, Any, PublicFeedbackClass | str]] = []
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("each payload source record must be a mapping")
        adapted = adapt_visible_input(selected, row)
        case_id = str(row.get("case_id") or "").strip()
        feedback = (
            feedback_by_case.get(case_id, PublicFeedbackClass.NO_FEEDBACK)
            if feedback_by_case is not None
            else PublicFeedbackClass.NO_FEEDBACK
        )
        pending.append((case_id, adapted.canonical, feedback))
    if active_compiler is not None:
        compiled_values = [
            active_compiler.compile(canonical, feedback)
            for _, canonical, feedback in pending
        ]
        isolated = False
    else:
        compiled_values = _compile_isolated(
            [(canonical, feedback) for _, canonical, feedback in pending]
        )
        isolated = True
    bindings = [
        bind_case(case_id=case_id, benchmark=selected, compiled=compiled)
        for (case_id, _, _), compiled in zip(pending, compiled_values)
    ]
    return build_manifest(bindings, compiler_isolated=isolated)


def _compile_isolated(values: list[tuple[Any, PublicFeedbackClass | str]]) -> list[CompiledPayload]:
    requests = "".join(
        json.dumps(
            {
                "canonical": canonical.as_dict(),
                "public_feedback_class": PublicFeedbackClass(feedback).value,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
        for canonical, feedback in values
    )
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.adaptive_ablation.compiler_worker"],
        cwd=PACKAGE_DIR.parents[1],
        input=requests,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"isolated compiler failed with exit code {completed.returncode}")
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != len(values):
        raise RuntimeError("isolated compiler returned an unexpected result count")
    results: list[CompiledPayload] = []
    for row in rows:
        if row.get("error") is not None:
            raise RuntimeError(f"isolated compiler rejected input: {row['error']}")
        results.append(_compiled_from_dict(row["compiled"]))
    return results


def _compiled_from_dict(value: Mapping[str, Any]) -> CompiledPayload:
    return CompiledPayload(
        schema_version=str(value["schema_version"]),
        experiment_mode=str(value["experiment_mode"]),
        status=CompileStatus(str(value["status"])),
        payload=str(value.get("payload") or ""),
        payload_sha256=str(value["payload_sha256"]),
        original_payload_sha256=str(value["original_payload_sha256"]),
        input_sha256=str(value["input_sha256"]),
        attacker_channel=AttackerChannel(str(value["attacker_channel"])),
        tool_schema_class=ToolSchemaClass(str(value["tool_schema_class"])),
        public_feedback_class=PublicFeedbackClass(str(value["public_feedback_class"])),
        template_key=str(value["template_key"]),
        template_sha256=str(value["template_sha256"]),
        requested_operators=tuple(MechanismOperator(item) for item in value.get("requested_operators") or ()),
        operators=tuple(MechanismOperator(item) for item in value.get("operators") or ()),
        generation_invalid=bool(value.get("generation_invalid")),
        fallback_reason=(str(value["fallback_reason"]) if value.get("fallback_reason") is not None else None),
        target_status=str(value.get("target_status") or "target-unavailable"),
    )


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def compiler_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(COMPILER_CODE_FILES, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def attacker_visible_fields_sha256() -> str:
    value = {benchmark.value: list(fields) for benchmark, fields in ATTACKER_VISIBLE_FIELDS.items()}
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze deterministic feedback-conditioned adaptive payloads")
    parser.add_argument("--benchmark", choices=[item.value for item in Benchmark], required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-payload-chars", type=int, default=16_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    compiler = AttackCompiler(max_payload_chars=args.max_payload_chars)
    manifest = freeze_records(
        benchmark=Benchmark(args.benchmark),
        records=read_jsonl(args.input_jsonl),
        compiler=compiler,
    )
    write_manifest(args.output, manifest)
    print(args.output.resolve())
    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
