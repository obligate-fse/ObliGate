"""Unified CLI for dependency-isolated ObliGate benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .contracts import BenchmarkRunRequest
from .registry import default_registry, discover_repo_root


def register_benchmark_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``benchmark`` beneath an existing ``obligate eval`` parser."""

    benchmark = subparsers.add_parser(
        "benchmark",
        aliases=["benchmarks"],
        help="Run AgentDojo, Agent-SafetyBench, or ASB in isolated Python environments",
    )
    _register_actions(benchmark)


def _register_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="benchmark_cmd", required=True)

    list_cmd = actions.add_parser("list", help="List pinned benchmark adapters")
    list_cmd.set_defaults(func=cmd_list_benchmarks)

    show = actions.add_parser("show", aliases=["manifest"], help="Show one pinned benchmark specification")
    show.add_argument("benchmark")
    show.add_argument("--repo-root", type=Path, default=None)
    show.set_defaults(func=cmd_show_benchmark)

    doctor = actions.add_parser("doctor", help="Check an isolated interpreter and upstream checkout without running cases")
    doctor.add_argument("benchmark", nargs="?", default="all", help="Canonical ID/alias, or all")
    doctor.add_argument("--python", dest="python_executable", default=sys.executable, help="Python executable in the benchmark environment")
    doctor.add_argument("--repo-root", type=Path, default=None)
    doctor.add_argument("--upstream-dir", type=Path, default=None)
    doctor.add_argument("--strict", action="store_true", help="Return exit code 2 when any required check fails")
    doctor.set_defaults(func=cmd_benchmark_doctor)

    run = actions.add_parser("run", help="Plan or execute a benchmark through its isolated interpreter")
    run.add_argument("benchmark", help="Canonical ID or alias")
    run.add_argument("--python", dest="python_executable", default=sys.executable, help="Python executable in the benchmark environment")
    run.add_argument("--repo-root", type=Path, default=None)
    run.add_argument("--upstream-dir", type=Path, default=None)
    run.add_argument("--output-root", type=Path, default=Path("experiments/benchmark_runs"))
    run.add_argument("--run-id")
    run.add_argument("--model")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--limit", type=int)
    run.add_argument("--suite", help="AgentDojo suite (workspace, slack, travel, or banking)")
    run.add_argument("--attack", help="AgentDojo attack name")
    run.add_argument(
        "--defense",
        choices=["none", "obligate"],
        help="AgentDojo method: No Defense or the ObliGate firewall",
    )
    run.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Adapter option; JSON scalar values are accepted and recorded in the manifest",
    )
    run.add_argument("--dry-run", action="store_true", help="Print the exact manifest without probing, running, or writing files")
    run.add_argument("--skip-doctor", action="store_true", help="Execute even if the isolated environment was not probed")
    run.add_argument(
        "--runner-arg",
        dest="runner_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Forward one argument to the selected runner; repeat this option and use --runner-arg=--flag for flags",
    )
    run.set_defaults(func=cmd_benchmark_run)


def _repo_root(value: Path | None) -> Path:
    return (value or discover_repo_root()).resolve()


def _request(args: argparse.Namespace, *, benchmark: str, repo_root: Path) -> BenchmarkRunRequest:
    upstream = getattr(args, "upstream_dir", None)
    return BenchmarkRunRequest(
        benchmark=benchmark,
        repo_root=repo_root,
        output_root=Path(getattr(args, "output_root", Path("experiments/benchmark_runs"))),
        python_executable=str(args.python_executable),
        run_id=getattr(args, "run_id", None),
        upstream_dir=None if upstream is None else Path(upstream),
        model=getattr(args, "model", None),
        seed=int(getattr(args, "seed", 0)),
        limit=getattr(args, "limit", None),
        runner_args=_runner_args(getattr(args, "runner_args", ())),
        options=_options(args),
    )


def _runner_args(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    items = list(values)
    if items and items[0] == "--":
        items.pop(0)
    return tuple(items)


def _options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in getattr(args, "option", ()):
        if "=" not in item:
            raise ValueError(f"invalid --option {item!r}; expected KEY=VALUE")
        key, raw = item.split("=", 1)
        if not key:
            raise ValueError("--option key may not be empty")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        options[key] = value
    for key in ("suite", "attack", "defense"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    return options


def cmd_list_benchmarks(args: argparse.Namespace) -> int:
    registry = default_registry(discover_repo_root())
    _print_json({"benchmarks": [spec.to_dict() for spec in registry.specs()]})
    return 0


def cmd_show_benchmark(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    adapter = default_registry(root).get(args.benchmark)
    _print_json(adapter.spec.to_dict())
    return 0


def cmd_benchmark_doctor(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    registry = default_registry(root)
    if args.benchmark == "all":
        if args.upstream_dir is not None:
            raise ValueError("--upstream-dir requires a single benchmark")
        adapters = registry.adapters()
    else:
        adapters = (registry.get(args.benchmark),)
    reports = []
    for adapter in adapters:
        request = _request(args, benchmark=adapter.spec.benchmark, repo_root=root)
        reports.append(adapter.doctor(request))
    payload = {"ready": all(report["ready"] for report in reports), "reports": reports}
    _print_json(payload)
    return 2 if args.strict and not payload["ready"] else 0


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    root = _repo_root(args.repo_root)
    adapter = default_registry(root).get(args.benchmark)
    request = _request(args, benchmark=adapter.spec.benchmark, repo_root=root)
    manifest = adapter.plan(request)
    if args.dry_run:
        _print_json(adapter.planned_result(manifest).to_dict())
        return 0

    if not args.skip_doctor:
        doctor = adapter.doctor(request)
        if not doctor["ready"]:
            _print_json(
                {
                    "status": "environment_not_ready",
                    "benchmark": adapter.spec.benchmark,
                    "doctor": doctor,
                    "hint": "Install the benchmark-specific extra/environment, or use --skip-doctor only after manually validating it.",
                }
            )
            return 2

    print(json.dumps({"status": "running", "benchmark": adapter.spec.benchmark, "run_id": manifest.run_id, "output_dir": manifest.output_dir}))
    result = adapter.execute(manifest)
    _print_json(result.to_dict())
    return 0 if result.status == "succeeded" else (result.exit_code or 1)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="obligate-benchmark", description=__doc__)
    _register_actions(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError) as exc:
        print(f"benchmark CLI error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "register_benchmark_commands"]
