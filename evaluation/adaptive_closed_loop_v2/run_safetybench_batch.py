"""Agent-SafetyBench runner for ObliGate closed-loop rounds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from obligate.eval.agentdojo.gate import tool_firewall as firewall_module
from experiments.agent_safetybench import obligate_runner as runner
from experiments.adaptive_ablation.mechanism_activation import (
    activation_manifest,
    install_activation_bridge,
    uninstall_activation_bridge,
)
from experiments.adaptive_ablation.provider_ledger import ProviderUsageLedger
from experiments.adaptive_ablation.safety import simulator_only_environment

from .common import ROOT, provider_settings, sha256_file, write_json

SAFETYBENCH_UPSTREAM = Path(
    os.environ.get("SAFETYBENCH_UPSTREAM")
    or ROOT / "third_party" / "Agent-SafetyBench"
).resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    key_env, base_url = provider_settings(args.model)
    ledger = ProviderUsageLedger(args.usage_ledger, benchmark="agent_safetybench", run_label=args.run_label)
    if runner.OpenAI is None:
        raise RuntimeError("openai dependency is unavailable")
    original_openai = runner.OpenAI
    activation_originals = install_activation_bridge(firewall_module)
    runner.OpenAI = ledger.instrument_openai(original_openai)  # type: ignore[assignment]
    runner_args = argparse.Namespace(
        upstream_dir=str(SAFETYBENCH_UPSTREAM),
        data=str(args.data.resolve()),
        out_dir=str(args.out_dir.resolve()),
        env_file="",
        model=args.model,
        api_key_env=key_env,
        base_url=base_url,
        temperature=0.0,
        max_tokens=2048,
        timeout=180.0,
        max_rounds=10,
        limit=None,
        workers=args.workers,
        max_provider_invalid_rate=100.0,
        case_ids="",
        risk_contains="",
        allow_oracle_diagnostic=False,
        defense=args.defense,
        sanitize_mode="soft",
        enable_thinking=False,
        resume=args.resume,
        fail_fast=False,
        protocol_label="obligate_adaptive_closed_loop_v2_visible_fair",
        formal_clean=False,
    )
    binding = {
        "schema_version": "obligate-safetybench-batch-v2",
        "benchmark": "agent_safetybench",
        "model": args.model,
        "run_label": args.run_label,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "usage_ledger": str(args.usage_ledger.resolve()),
        "simulator_only": True,
        "mechanism_activation": activation_manifest(),
        "defense": args.defense,
    }
    try:
        with simulator_only_environment():
            summary = runner.run(runner_args)
        binding["summary"] = summary
        write_json(args.out_dir / "obligate_execution_binding.json", binding)
        return summary
    finally:
        runner.OpenAI = original_openai  # type: ignore[assignment]
        uninstall_activation_bridge(firewall_module, activation_originals)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--usage-ledger", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--defense",
        choices=[
            "none",
            "obligate_visible_fair",
        ],
        default="obligate_visible_fair",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
