"""Agent Security Bench runner for ObliGate closed-loop rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from obligate.eval.agentdojo.gate import tool_firewall as firewall_module
from experiments.agent_security_bench import asb_obligate_runner as runner
from experiments.adaptive_ablation.mechanism_activation import (
    activation_manifest,
    install_activation_bridge,
    uninstall_activation_bridge,
)
from experiments.adaptive_ablation.provider_ledger import ProviderUsageLedger
from experiments.adaptive_ablation.safety import simulator_only_environment

from .common import ROOT, provider_settings, safe_slug, sha256_file, write_json

ASB_UPSTREAM = Path(
    os.environ.get("ASB_UPSTREAM")
    or ROOT / "third_party" / "ASB"
).resolve()


def _load_payloads(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payloads: dict[str, str] = {}
    for item in manifest.get("cases") or []:
        case_id = str(item["case_id"])
        payload = str(item["payload"])
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != item.get("payload_sha256"):
            raise ValueError(f"payload hash mismatch for {case_id}")
        payloads[case_id] = payload
    return payloads, manifest


def run(args: argparse.Namespace) -> int:
    payloads, manifest = _load_payloads(args.payload_manifest)
    key_env, base_url = provider_settings(args.model)
    ledger = ProviderUsageLedger(args.usage_ledger, benchmark="agent_security_bench", run_label=args.run_label)
    if runner.OpenAI is None:
        raise RuntimeError("openai dependency is unavailable")
    original_openai = runner.OpenAI
    original_attack_prompt = runner.attack_prompt
    original_argv = sys.argv[:]
    activation_originals = install_activation_bridge(firewall_module)

    def frozen_attack_prompt(case: Any) -> str:
        try:
            return payloads[str(case.case_id)]
        except KeyError as exc:
            raise ValueError(f"ASB case has no active payload: {case.case_id}") from exc

    run_name = safe_slug(args.run_label)
    cli = [
        "asb_obligate_runner.py",
        "--upstream",
        str(ASB_UPSTREAM),
        "--model",
        args.model,
        "--base-url",
        base_url,
        "--api-key-env",
        key_env,
        "--attack-modes",
        "dpi",
        "opi",
        "mixed",
        "memory",
        "--attack-types",
        "naive",
        "context_ignoring",
        "combined_attack",
        "--attack-tool-type",
        "all",
        "--task-num",
        "all",
        "--case-ids-file",
        str(args.case_ids_file.resolve()),
        "--modes",
        args.mode,
        "--workers",
        str(args.workers),
        "--temperature",
        "0",
        "--max-tokens",
        "512",
        "--workflow-generation-attempts",
        str(args.workflow_generation_attempts),
        "--timeout",
        "60",
        "--max-provider-error-rate",
        "100",
        "--max-workflow-failure-rate",
        "100",
        "--seed",
        "20260716",
        "--out-dir",
        str(args.out_dir.resolve()),
        "--run-name",
        run_name,
    ]
    if args.resume:
        cli.append("--resume")
    runner.OpenAI = ledger.instrument_openai(original_openai)  # type: ignore[assignment]
    runner.attack_prompt = frozen_attack_prompt  # type: ignore[assignment]
    try:
        sys.argv = cli
        with simulator_only_environment():
            runner.main()
        binding = {
            "schema_version": "obligate-asb-batch-v2",
            "benchmark": "agent_security_bench",
            "model": args.model,
            "run_label": args.run_label,
            "run_name": run_name,
            "payload_manifest": str(args.payload_manifest.resolve()),
            "payload_manifest_sha256": sha256_file(args.payload_manifest),
            "payload_manifest_content_sha256": manifest.get("manifest_sha256"),
            "case_ids_file": str(args.case_ids_file.resolve()),
            "case_ids_file_sha256": sha256_file(args.case_ids_file),
            "usage_ledger": str(args.usage_ledger.resolve()),
            "official_protocol_equivalent": False,
            "simulator_only": True,
            "mode": args.mode,
            "mechanism_activation": activation_manifest(),
        }
        write_json(args.out_dir / run_name / "obligate_execution_binding.json", binding)
        return 0
    finally:
        sys.argv = original_argv
        runner.OpenAI = original_openai  # type: ignore[assignment]
        runner.attack_prompt = original_attack_prompt  # type: ignore[assignment]
        uninstall_activation_bridge(firewall_module, activation_originals)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--usage-ledger", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=["obligate_registry_blind", "no_defense"], default="obligate_registry_blind")
    parser.add_argument("--workflow-generation-attempts", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
