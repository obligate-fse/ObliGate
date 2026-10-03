"""Run one AgentDojo case for ObliGate closed-loop adaptive evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from obligate.eval.agentdojo.gate import tool_firewall as firewall_module
from obligate.eval.agentdojo.runner import run_tool_firewall_eval as runner

from experiments.adaptive_ablation.mechanism_activation import (
    activation_manifest,
    install_activation_bridge,
    uninstall_activation_bridge,
)
from experiments.adaptive_ablation.provider_ledger import ProviderUsageLedger


class _FrozenAttack:
    def __init__(self, original: Any, *, payload: str, original_sha256: str) -> None:
        self._original = original
        self._payload = payload
        self._original_sha256 = original_sha256

    def attack(self, user_task: Any, injection_task: Any) -> dict[str, str]:
        injections = self._original.attack(user_task, injection_task)
        rendered = "\n\n".join(sorted({str(value) for value in injections.values()}))
        rendered = rendered.replace("\r\n", "\n").replace("\r", "\n").strip()
        if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != self._original_sha256:
            raise ValueError("AgentDojo original public injection drifted after payload freeze")
        return {str(key): _yaml_double_quoted_template_escape(self._payload) for key in injections}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _yaml_double_quoted_template_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _load_payloads(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("payload manifest must contain cases")
    payloads: dict[str, str] = {}
    items: dict[str, dict[str, Any]] = {}
    for item in cases:
        if not isinstance(item, dict):
            continue
        case_id = str(item["case_id"])
        payload = str(item["payload"])
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != item.get("payload_sha256"):
            raise ValueError(f"payload hash mismatch for {case_id}")
        payloads[case_id] = payload
        items[case_id] = item
    if len(payloads) != len(cases):
        raise ValueError("payload manifest has duplicate or invalid case IDs")
    return payloads, items, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    payloads, items, manifest = _load_payloads(args.payload_manifest)
    if args.case_id not in payloads:
        raise ValueError(f"AgentDojo case has no active payload: {args.case_id}")
    manifest_item = items[args.case_id]
    ledger = ProviderUsageLedger(args.usage_ledger, benchmark="agentdojo", run_label=args.run_label)
    original_loader = runner._load_agentdojo_deps

    def patched_loader() -> dict[str, Any]:
        deps = dict(original_loader())
        original_load_attack = deps["load_attack"]
        original_openai = deps["OpenAI"]

        def load_attack(name: str, suite: Any, pipeline: Any) -> _FrozenAttack:
            return _FrozenAttack(
                original_load_attack(name, suite, pipeline),
                payload=payloads[args.case_id],
                original_sha256=str(manifest_item["original_payload_sha256"]),
            )

        deps["load_attack"] = load_attack
        deps["OpenAI"] = ledger.instrument_openai(original_openai)
        return deps

    runner._load_agentdojo_deps = patched_loader  # type: ignore[assignment]
    activation_originals = install_activation_bridge(firewall_module)
    try:
        summary = runner.run_suite(
            args.suite,
            args.model,
            args.defense,
            benchmark_version="v1.2.2",
            attack="important_instructions",
            report_dir=args.output.parent,
            run_name=args.run_name,
            user_task_ids=[args.user_task_id],
            injection_task_ids=[args.injection_task_id],
            confirmation_mode="strict_eval",
            save_full_trace=True,
            trace_dir=args.trace_dir,
            mode="fair",
        )
    finally:
        runner._load_agentdojo_deps = original_loader  # type: ignore[assignment]
        uninstall_activation_bridge(firewall_module, activation_originals)

    summary["obligate_closed_loop"] = {
        "schema_version": "obligate-agentdojo-binding-v2",
        "payload_manifest_sha256": hashlib.sha256(args.payload_manifest.read_bytes()).hexdigest(),
        "payload_manifest_content_sha256": manifest.get("manifest_sha256"),
        "payload_sha256": manifest_item["payload_sha256"],
        "round": manifest.get("round"),
        "victim_model": manifest.get("victim_model"),
        "attacker_seed": manifest.get("attacker_seed"),
        "segment": manifest.get("segment"),
        "primary_strategy": manifest_item.get("primary_strategy"),
        "secondary_strategy": manifest_item.get("secondary_strategy"),
        "simulator_only": True,
        "mechanism_activation": activation_manifest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--user-task-id", required=True)
    parser.add_argument("--injection-task-id", required=True)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--usage-ledger", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--defense", choices=["none", "obligate"], default="obligate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
