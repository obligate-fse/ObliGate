"""Deterministically enumerate the 949 AgentDojo v1.2.2 attack task pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

SUITES = ("banking", "slack", "travel", "workspace")
EXPECTED_COUNTS = {
    "banking": (16, 9, 144),
    "slack": (21, 5, 105),
    "travel": (20, 7, 140),
    "workspace": (40, 14, 560),
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "agentdojo_v1.2.2_949_case_plan.json"


def _case(
    suite: str,
    user_task_id: str,
    injection_task_id: str,
) -> dict[str, str]:
    return {
        "case_id": f"full_{suite}_{user_task_id}_{injection_task_id}",
        "suite": suite,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
    }


def _from_agentdojo(benchmark_version: str) -> list[dict[str, str]]:
    from agentdojo.task_suite.load_suites import get_suite

    cases: list[dict[str, str]] = []
    for suite_name in SUITES:
        suite = get_suite(benchmark_version, suite_name)
        user_ids = sorted(str(value) for value in suite.user_tasks)
        injection_ids = sorted(str(value) for value in suite.injection_tasks)
        cases.extend(
            _case(suite_name, user_id, injection_id)
            for user_id in user_ids
            for injection_id in injection_ids
        )
    return cases


def _from_manifest(path: Path) -> list[dict[str, str]]:
    """Normalize an existing plan; intended only for archive maintenance."""

    value = json.loads(path.read_text(encoding="utf-8"))
    return [
        _case(
            str(item["suite"]),
            str(item["user_task_id"]),
            str(item["injection_task_id"]),
        )
        for item in value["cases"]
    ]


def build_plan(
    cases: Iterable[Mapping[str, str]],
    *,
    benchmark_version: str = "v1.2.2",
) -> dict[str, Any]:
    unique = {
        (
            str(item["suite"]),
            str(item["user_task_id"]),
            str(item["injection_task_id"]),
        )
        for item in cases
    }
    ordered = [
        _case(suite, user_id, injection_id)
        for suite, user_id, injection_id in sorted(unique)
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in ordered:
        grouped[item["suite"]].append(item)

    by_suite: dict[str, dict[str, int]] = {}
    for suite in SUITES:
        rows = grouped[suite]
        counts = (
            len({item["user_task_id"] for item in rows}),
            len({item["injection_task_id"] for item in rows}),
            len(rows),
        )
        if counts != EXPECTED_COUNTS[suite]:
            raise ValueError(
                f"AgentDojo {suite} population drift: {counts}, "
                f"expected {EXPECTED_COUNTS[suite]}"
            )
        by_suite[suite] = {
            "user_task_count": counts[0],
            "injection_task_count": counts[1],
            "case_count": counts[2],
        }
    if len(ordered) != 949:
        raise ValueError(f"AgentDojo population drift: {len(ordered)}, expected 949")

    return {
        "schema_version": "obligate-agentdojo-case-plan-v1",
        "benchmark": "AgentDojo",
        "benchmark_package": "agentdojo==0.1.35",
        "benchmark_version": benchmark_version,
        "attack": "important_instructions",
        "population_definition": "all user-task x injection-task pairs in the four v1.2.2 suites",
        "suites": list(SUITES),
        "case_count": len(ordered),
        "by_suite": by_suite,
        "cases": ordered,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Maintenance-only normalization source; normal reproduction reads AgentDojo.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = (
        _from_manifest(args.source_manifest)
        if args.source_manifest is not None
        else _from_agentdojo(args.benchmark_version)
    )
    plan = build_plan(cases, benchmark_version=args.benchmark_version)
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
