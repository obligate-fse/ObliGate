"""Deterministically enumerate the 8,160 ASB cases used by ObliGate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "third_party" / "ASB"
DEFAULT_OUTPUT = ROOT / "data" / "asb_iclr2025_8160_case_plan.json"
ATTACK_PROTOCOL = (
    ("dpi", "naive"),
    ("opi", "context_ignoring"),
    ("mixed", "combined_attack"),
    ("memory", "combined_attack"),
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(upstream: Path) -> dict[str, Any]:
    task_path = upstream / "data" / "agent_task.jsonl"
    attack_path = upstream / "data" / "all_attack_tools.jsonl"
    tasks = _jsonl(task_path)
    attacks = _jsonl(attack_path)
    attacks_by_agent: dict[str, list[str]] = defaultdict(list)
    for item in attacks:
        attacks_by_agent[str(item["Corresponding Agent"])].append(
            str(item["Attacker Tool"])
        )

    case_ids: list[str] = []
    base_count = 0
    for agent in sorted(tasks, key=lambda item: str(item["agent_name"])):
        agent_name = str(agent["agent_name"])
        tools = sorted(attacks_by_agent[agent_name])
        for task_index, _ in enumerate(agent["tasks"]):
            for tool_name in tools:
                base_count += 1
                case_ids.extend(
                    f"{mode}:{attack_type}:{agent_name}:{task_index}:{tool_name}"
                    for mode, attack_type in ATTACK_PROTOCOL
                )

    if base_count != 2_040 or len(case_ids) != 8_160:
        raise ValueError(
            f"ASB population drift: {base_count} base samples and "
            f"{len(case_ids)} cases; expected 2,040 and 8,160"
        )
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("ASB case plan contains duplicate identifiers")

    return {
        "schema_version": "obligate-asb-case-plan-v1",
        "benchmark": "Agent Security Bench",
        "benchmark_revision": "1f561dccf92d55302368fa67679b4ba9d9c8fdc4",
        "selection": "all 51 tasks x all 40 corresponding attack tools x four matched attack protocols",
        "attack_protocol": [
            {"mode": mode, "attack_type": attack_type}
            for mode, attack_type in ATTACK_PROTOCOL
        ],
        "source_hashes": {
            "agent_task.jsonl": _sha256(task_path),
            "all_attack_tools.jsonl": _sha256(attack_path),
        },
        "base_sample_count": base_count,
        "case_count": len(case_ids),
        "case_ids": case_ids,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args.upstream.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
