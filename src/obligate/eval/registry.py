"""Dependency-isolated registry and subprocess launchers for benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import (
    BENCHMARK_CONTRACT_VERSION,
    BenchmarkEvent,
    BenchmarkManifest,
    BenchmarkResult,
    BenchmarkRunRequest,
    BenchmarkSpec,
    make_run_id,
    utc_now,
)

AGENTDOJO_VERSION = "0.1.35"
AGENTDOJO_BENCHMARK_VERSION = "v1.2.2"
AGENTDOJO_REVISION = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
AGENT_SAFETYBENCH_REVISION = "74feea8de601b3a1449a93fcf70017fe61556f73"
ASB_ICLR2025_REVISION = "1f561dccf92d55302368fa67679b4ba9d9c8fdc4"

_SENSITIVE_FLAGS = frozenset({"--api-key", "--password", "--secret", "--token"})


def _provider_route(model: str) -> tuple[str, str]:
    lowered = model.casefold()
    if "qwen" in lowered:
        return (
            "DASHSCOPE_API_KEY",
            os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    if "deepseek" in lowered:
        return (
            "DEEPSEEK_API_KEY",
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/v1",
        )
    return ("OPENAI_API_KEY", os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")


class BenchmarkAdapter(ABC):
    """Base class for an adapter that launches one benchmark interpreter."""

    spec: BenchmarkSpec
    runner_relative_path: Path
    default_model: str
    required_modules: tuple[str, ...]
    credential_environment_keys: tuple[str, ...] = ()
    reserved_runner_flags: frozenset[str] = frozenset()

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    @property
    def runner_path(self) -> Path:
        return self.repo_root / self.runner_relative_path

    def default_upstream_dir(self) -> Path | None:
        return None

    def resolve_upstream_dir(self, request: BenchmarkRunRequest) -> Path | None:
        if request.upstream_dir is not None:
            return request.upstream_dir.resolve()
        default = self.default_upstream_dir()
        return default.resolve() if default is not None else None

    def plan(self, request: BenchmarkRunRequest) -> BenchmarkManifest:
        if request.benchmark not in {self.spec.benchmark, *self.spec.aliases}:
            raise ValueError(f"request benchmark {request.benchmark!r} does not match adapter {self.spec.benchmark!r}")
        self._validate_runner_args(request.runner_args)
        run_id = request.run_id or make_run_id(self.spec.benchmark)
        run_dir = (request.output_root.resolve() / self.spec.benchmark / run_id).resolve()
        upstream = self.resolve_upstream_dir(request)
        model = request.model or self.default_model
        command = self.build_command(request, run_id=run_id, run_dir=run_dir, upstream_dir=upstream, model=model)
        resolved_python = _resolve_executable(request.python_executable) or request.python_executable
        command[0] = resolved_python
        environment_keys = tuple(sorted({"PYTHONPATH", *self.credential_environment_keys}))
        return BenchmarkManifest(
            spec=self.spec,
            run_id=run_id,
            command=tuple(command),
            cwd=str(request.repo_root.resolve()),
            output_dir=str(run_dir),
            python_executable=resolved_python,
            upstream_dir=None if upstream is None else str(upstream),
            model=model,
            seed=request.seed,
            limit=request.limit,
            runner_args=tuple(request.runner_args),
            options=dict(request.options),
            environment_keys=environment_keys,
        )

    def _validate_runner_args(self, runner_args: Iterable[str]) -> None:
        for raw in runner_args:
            flag = raw.split("=", 1)[0]
            if flag in _SENSITIVE_FLAGS:
                raise ValueError(f"{flag} may expose a credential in the run manifest; use an environment variable")
            if flag in self.reserved_runner_flags:
                raise ValueError(f"{flag} is managed by the unified launcher and may not be forwarded")

    @abstractmethod
    def build_command(
        self,
        request: BenchmarkRunRequest,
        *,
        run_id: str,
        run_dir: Path,
        upstream_dir: Path | None,
        model: str,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def required_upstream_files(self, upstream_dir: Path | None) -> tuple[Path, ...]:
        raise NotImplementedError

    @abstractmethod
    def summary_path(self, manifest: BenchmarkManifest) -> Path:
        raise NotImplementedError

    @abstractmethod
    def split_metrics(self, summary: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return official metrics, ObliGate metrics, and non-metric metadata."""

    def validate_summary(self, summary: Mapping[str, Any]) -> str | None:
        if not summary:
            return "native summary is missing, malformed, or empty"
        return None

    def doctor(self, request: BenchmarkRunRequest) -> dict[str, Any]:
        """Inspect readiness without importing the benchmark into this process."""

        checks: list[dict[str, Any]] = []
        resolved_python = _resolve_executable(request.python_executable)
        if resolved_python is None:
            checks.append(_check("python_executable", "error", f"not found: {request.python_executable}"))
            return _doctor_report(self.spec, checks)
        checks.append(_check("python_executable", "ok", resolved_python))

        probe = _probe_python(resolved_python, self.required_modules, request.repo_root)
        if probe.get("error"):
            checks.append(_check("python_probe", "error", str(probe["error"])))
        else:
            checks.append(_check("python_version", self._python_status(str(probe.get("python", ""))), str(probe.get("python", "unknown"))))
            modules = probe.get("modules") or {}
            for module in self.required_modules:
                info = modules.get(module) or {}
                status = "ok" if info.get("available") else "error"
                detail = str(info.get("version") or "available") if info.get("available") else "not importable"
                checks.append(_check(f"module:{module}", status, detail))
            checks.extend(self._version_checks(modules))

        if self.runner_path.is_file():
            checks.append(_check("runner", "ok", str(self.runner_path)))
        else:
            checks.append(_check("runner", "error", f"missing: {self.runner_path}"))

        upstream = self.resolve_upstream_dir(request)
        required = self.required_upstream_files(upstream)
        if not required:
            checks.append(_check("upstream", "ok", "provided by the pinned Python package"))
        else:
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                checks.append(_check("upstream", "error", "missing required files", missing=missing))
            else:
                checks.append(_check("upstream", "ok", str(upstream)))
                checks.append(self._revision_check(upstream))

        present_keys = [key for key in self.credential_environment_keys if os.getenv(key)]
        if self.credential_environment_keys:
            if present_keys:
                checks.append(_check("credentials", "ok", f"available via {', '.join(present_keys)}"))
            else:
                checks.append(
                    _check(
                        "credentials",
                        "warn",
                        "no known API-key environment variable is set; custom runner arguments may select another provider",
                        expected=list(self.credential_environment_keys),
                    )
                )
        return _doctor_report(self.spec, checks)

    def _python_status(self, version: str) -> str:
        try:
            major, minor = (int(item) for item in version.split(".")[:2])
        except (TypeError, ValueError):
            return "warn"
        return "ok" if major == 3 and minor >= 10 else "error"

    def _version_checks(self, modules: Mapping[str, Any]) -> list[dict[str, Any]]:
        return []

    def _revision_check(self, upstream_dir: Path | None) -> dict[str, Any]:
        if upstream_dir is None:
            return _check("upstream_revision", "warn", f"expected {self.spec.upstream_revision}; no checkout path")
        git_dir = upstream_dir / ".git"
        if not git_dir.exists() or shutil.which("git") is None:
            return _check(
                "upstream_revision",
                "warn",
                f"checkout has no Git metadata; expected revision {self.spec.upstream_revision}",
            )
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=upstream_dir,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except OSError as exc:
            return _check("upstream_revision", "warn", str(exc))
        revision = completed.stdout.strip()
        if completed.returncode != 0:
            return _check("upstream_revision", "warn", completed.stderr.strip() or "unable to read revision")
        status = "ok" if revision == self.spec.upstream_revision else "warn"
        return _check("upstream_revision", status, revision, expected=self.spec.upstream_revision)

    def execute(self, manifest: BenchmarkManifest) -> BenchmarkResult:
        """Execute a prepared manifest and write lifecycle/result artifacts."""

        run_dir = Path(manifest.output_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"run output directory is not empty: {run_dir}; choose a new --run-id")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest.write(run_dir / "manifest.json")
        event_path = run_dir / "events.jsonl"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        if event_path.exists():
            event_path.unlink()

        started_at = utc_now()
        BenchmarkEvent(
            benchmark=self.spec.benchmark,
            run_id=manifest.run_id,
            sequence=0,
            kind="run_started",
            payload={"adapter": self.spec.adapter},
        ).append(event_path)
        BenchmarkEvent(
            benchmark=self.spec.benchmark,
            run_id=manifest.run_id,
            sequence=1,
            kind="process_started",
            payload={"command": list(manifest.command), "cwd": manifest.cwd},
        ).append(event_path)

        exit_code: int | None = None
        error: str | None = None
        try:
            environment = _child_environment(Path(manifest.cwd))
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
                completed = subprocess.run(
                    list(manifest.command),
                    cwd=manifest.cwd,
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                )
            exit_code = completed.returncode
        except OSError as exc:
            error = str(exc)

        BenchmarkEvent(
            benchmark=self.spec.benchmark,
            run_id=manifest.run_id,
            sequence=2,
            kind="process_finished",
            payload={"exit_code": exit_code, "error": error},
        ).append(event_path)

        summary_path = self.summary_path(manifest)
        summary = _read_json_mapping(summary_path)
        official_metrics, obligate_metrics, metric_metadata = self.split_metrics(summary)
        summary_error = self.validate_summary(summary)
        if exit_code == 0 and error is None and summary_error is not None:
            error = summary_error
        status = "succeeded" if exit_code == 0 and error is None else "failed"
        finished_at = utc_now()
        artifacts = {
            "manifest": str(manifest_path),
            "events": str(event_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        if summary_path.is_file():
            artifacts["native_summary"] = str(summary_path)
        result = BenchmarkResult(
            manifest=manifest,
            status=status,
            exit_code=exit_code,
            official_metrics=official_metrics,
            obligate_metrics=obligate_metrics,
            artifacts=artifacts,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
            metadata={
                "native_summary_loaded": bool(summary),
                "native_summary_valid": summary_error is None,
                **metric_metadata,
            },
        )
        result_path = result.write(run_dir / "result.json")
        artifacts_with_result = {**artifacts, "result": str(result_path)}
        result = BenchmarkResult(
            manifest=manifest,
            status=result.status,
            exit_code=result.exit_code,
            official_metrics=result.official_metrics,
            obligate_metrics=result.obligate_metrics,
            artifacts=artifacts_with_result,
            started_at=result.started_at,
            finished_at=result.finished_at,
            error=result.error,
            metadata=result.metadata,
        )
        result.write(result_path)
        BenchmarkEvent(
            benchmark=self.spec.benchmark,
            run_id=manifest.run_id,
            sequence=3,
            kind="run_finished",
            payload={"status": status, "exit_code": exit_code, "result": str(result_path)},
        ).append(event_path)
        return result

    def planned_result(self, manifest: BenchmarkManifest) -> BenchmarkResult:
        return BenchmarkResult(
            manifest=manifest,
            status="planned",
            exit_code=None,
            artifacts={
                "output_dir": manifest.output_dir,
                "manifest": str(Path(manifest.output_dir) / "manifest.json"),
                "events": str(Path(manifest.output_dir) / "events.jsonl"),
                "result": str(Path(manifest.output_dir) / "result.json"),
            },
            metadata={"dry_run": True, "filesystem_mutated": False},
        )


class AgentDojoAdapter(BenchmarkAdapter):
    spec = BenchmarkSpec(
        benchmark="agentdojo",
        display_name="AgentDojo",
        adapter="obligate.eval.agentdojo.gate.tool_firewall",
        upstream_url="https://github.com/ethz-spylab/agentdojo",
        upstream_revision=AGENTDOJO_REVISION,
        upstream_version=AGENTDOJO_VERSION,
        benchmark_version=AGENTDOJO_BENCHMARK_VERSION,
        python_recommendation="Python 3.11 (upstream supports 3.10-3.12)",
        install_extra="eval-agentdojo",
        official_metrics=("benign_utility", "utility_under_attack", "targeted_asr"),
        obligate_metrics=("secure_utility", "blocked_case_rate", "recovery_success_rate", "tool_gate_decision_distribution"),
        aliases=("dojo",),
        notes=("Pin both the package and benchmark-data version; use programmatic pipeline integration, not AgentDojo's fixed CLI defense registry.",),
    )
    runner_relative_path = Path("src/obligate/eval/agentdojo/runner/run_tool_firewall_eval.py")
    default_model = "local"
    required_modules = ("obligate", "agentdojo")
    credential_environment_keys = (
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OBLIGATE_LLM_API_KEY",
    )
    reserved_runner_flags = frozenset(
        {"--report-dir", "--run-name", "--benchmark-version", "--repo-root", "--suite", "--model", "--defense", "--attack", "--limit"}
    )

    def build_command(
        self,
        request: BenchmarkRunRequest,
        *,
        run_id: str,
        run_dir: Path,
        upstream_dir: Path | None,
        model: str,
    ) -> list[str]:
        options = dict(request.options)
        command = [
            request.python_executable,
            "-m",
            "obligate.eval.agentdojo.runner.run_tool_firewall_eval",
            "--suite",
            str(options.get("suite", "travel")),
            "--model",
            model,
            "--defense",
            str(options.get("defense", "obligate")),
            "--attack",
            str(options.get("attack", "none")),
            "--benchmark-version",
            AGENTDOJO_BENCHMARK_VERSION,
            "--report-dir",
            str(run_dir),
            "--run-name",
            run_id,
            "--repo-root",
            str(request.repo_root.resolve()),
        ]
        if request.limit is not None:
            command.extend(["--limit", str(request.limit)])
        command.extend(request.runner_args)
        return command

    def required_upstream_files(self, upstream_dir: Path | None) -> tuple[Path, ...]:
        return ()

    def _version_checks(self, modules: Mapping[str, Any]) -> list[dict[str, Any]]:
        actual = str((modules.get("agentdojo") or {}).get("version") or "unknown")
        status = "ok" if actual == AGENTDOJO_VERSION else "error"
        return [_check("agentdojo_version", status, actual, expected=AGENTDOJO_VERSION)]

    def _python_status(self, version: str) -> str:
        try:
            major, minor = (int(item) for item in version.split(".")[:2])
        except (TypeError, ValueError):
            return "warn"
        return "ok" if major == 3 and 10 <= minor <= 12 else "error"

    def summary_path(self, manifest: BenchmarkManifest) -> Path:
        return Path(manifest.output_dir) / f"{manifest.run_id}.json"

    def split_metrics(self, summary: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        official: dict[str, Any] = _select(summary, ("targeted_asr", "security_rate"))
        attack = str(summary.get("attack") or "none")
        if attack == "none" and "user_utility" in summary:
            official["benign_utility"] = summary["user_utility"]
        elif "utility_under_attack" in summary:
            official["utility_under_attack"] = summary["utility_under_attack"]
        normalized = summary.get("normalized_cases") if isinstance(summary.get("normalized_cases"), list) else []
        blocked = [row for row in normalized if isinstance(row, Mapping) and row.get("blocked_case")]
        recovered = [row for row in blocked if row.get("recovery_success")]
        obligate = _select(summary, ("secure_utility", "obligate_audit_summary", "obligate_audit_summary"))
        obligate["blocked_case_rate"] = 0.0 if not normalized else len(blocked) / len(normalized)
        obligate["recovery_success_rate"] = None if not blocked else len(recovered) / len(blocked)
        return official, obligate, {}

    def validate_summary(self, summary: Mapping[str, Any]) -> str | None:
        error = super().validate_summary(summary)
        if error is not None:
            return error
        if not isinstance(summary.get("normalized_cases"), list):
            return "AgentDojo summary lacks normalized_cases"
        if "user_utility" not in summary:
            return "AgentDojo summary lacks user_utility"
        return None


class AgentSafetyBenchAdapter(BenchmarkAdapter):
    spec = BenchmarkSpec(
        benchmark="agent_safetybench",
        display_name="Agent-SafetyBench",
        adapter="experiments.agent_safetybench.obligate_runner",
        upstream_url="https://github.com/thu-coai/Agent-SafetyBench",
        upstream_revision=AGENT_SAFETYBENCH_REVISION,
        upstream_version=None,
        benchmark_version="2,000-case released_data.json",
        python_recommendation="Python 3.11 in a dedicated generation environment; score with ShieldAgent in a separate Linux/CUDA environment",
        install_extra="eval-agent-safetybench",
        official_metrics=("shieldagent_safety_score", "behavior_safety_score", "content_safety_score", "per_risk_safety_score"),
        obligate_metrics=("dangerous_action_blocking_rate", "unsafe_tool_execution_rate", "safe_action_pass_rate"),
        aliases=("agent-safetybench", "safetybench", "asafetybench"),
        notes=("Official safety metrics require the separate ShieldAgent scorer; runner-side execution metrics are not substitutes.",),
    )
    runner_relative_path = Path("experiments/agent_safetybench/obligate_runner.py")
    default_model = "deepseek-v4-flash"
    required_modules = ("obligate", "openai")
    credential_environment_keys = (
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OBLIGATE_LLM_API_KEY",
    )
    reserved_runner_flags = frozenset({"--upstream-dir", "--data", "--out-dir", "--model", "--limit"})

    def default_upstream_dir(self) -> Path:
        return self.repo_root / "third_party/Agent-SafetyBench"

    def build_command(
        self,
        request: BenchmarkRunRequest,
        *,
        run_id: str,
        run_dir: Path,
        upstream_dir: Path | None,
        model: str,
    ) -> list[str]:
        assert upstream_dir is not None
        command = [
            request.python_executable,
            str(self.runner_path),
            "--upstream-dir",
            str(upstream_dir),
            "--data",
            str(upstream_dir / "data/released_data.json"),
            "--out-dir",
            str(run_dir),
            "--model",
            model,
        ]
        key_env, base_url = _provider_route(model)
        command.extend(["--api-key-env", key_env, "--base-url", base_url])
        if request.limit is not None:
            command.extend(["--limit", str(request.limit)])
        command.extend(request.runner_args)
        return command

    def required_upstream_files(self, upstream_dir: Path | None) -> tuple[Path, ...]:
        if upstream_dir is None:
            return ()
        return (
            upstream_dir / "data/released_data.json",
            upstream_dir / "environments/EnvManager.py",
        )

    def _python_status(self, version: str) -> str:
        try:
            major, minor = (int(item) for item in version.split(".")[:2])
        except (TypeError, ValueError):
            return "warn"
        if (major, minor) == (3, 11):
            return "ok"
        return "warn" if major == 3 and minor >= 10 else "error"

    def _version_checks(self, modules: Mapping[str, Any]) -> list[dict[str, Any]]:
        actual = str((modules.get("openai") or {}).get("version") or "unknown")
        status = "ok" if actual == "1.59.7" else "error"
        return [_check("openai_version", status, actual, expected="1.59.7")]

    def summary_path(self, manifest: BenchmarkManifest) -> Path:
        return Path(manifest.output_dir) / "summary.json"

    def split_metrics(self, summary: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        # ShieldAgent is intentionally a separate, GPU-heavy phase.  These are
        # execution-grounded ObliGate metrics, not the official safety score.
        obligate = _select(
            summary,
            (
                "dangerous_action_blocking_rate",
                "unsafe_tool_execution_rate",
                "safe_action_pass_rate",
                "decision_counts",
                "by_risk_decisions",
            ),
        )
        return {}, obligate, {"official_scoring": "pending ShieldAgent scoring of gen_res.json"}

    def validate_summary(self, summary: Mapping[str, Any]) -> str | None:
        error = super().validate_summary(summary)
        if error is not None:
            return error
        if not isinstance(summary.get("cases"), int):
            return "Agent-SafetyBench summary lacks integer case count"
        required = {
            "dangerous_action_blocking_rate",
            "unsafe_tool_execution_rate",
            "safe_action_pass_rate",
        }
        missing = required - set(summary)
        if missing:
            return f"Agent-SafetyBench summary lacks core metrics: {sorted(missing)!r}"
        return None


class AsbIclr2025Adapter(BenchmarkAdapter):
    spec = BenchmarkSpec(
        benchmark="asb_iclr2025",
        display_name="Agent Security Bench (ASB, ICLR 2025)",
        adapter="experiments.agent_security_bench.asb_obligate_runner",
        upstream_url="https://github.com/agiresearch/ASB",
        upstream_revision=ASB_ICLR2025_REVISION,
        upstream_version=None,
        benchmark_version="ICLR 2025 final",
        python_recommendation="Python 3.11 in a dedicated ASB environment",
        install_extra="eval-asb",
        official_metrics=("ASR_compatible", "PNA_style"),
        obligate_metrics=("executed_target_tool_asr", "dangerous_action_blocking_rate", "unsafe_tool_execution_rate", "safe_action_pass_rate"),
        aliases=("asb", "agent-security-bench", "agent_security_bench"),
        notes=(
            "This ID means agiresearch/ASB, not Open Agent Security Benchmark (OASB), Lakera B3, or SecureAgentBench.",
            "Preserve ASB's goal-string ASR and additionally report structured executed-target-tool ASR.",
            "Judge-based RR/BP/FNR/FPR/NRP require the upstream evaluator and are not emitted by this runner.",
        ),
    )
    runner_relative_path = Path("experiments/agent_security_bench/asb_obligate_runner.py")
    default_model = "qwen-plus"
    required_modules = ("obligate", "openai")
    credential_environment_keys = (
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OBLIGATE_LLM_API_KEY",
    )
    reserved_runner_flags = frozenset({"--upstream", "--out-dir", "--run-name", "--api-key", "--model", "--seed", "--limit"})

    def default_upstream_dir(self) -> Path:
        return self.repo_root / "third_party/ASB"

    def build_command(
        self,
        request: BenchmarkRunRequest,
        *,
        run_id: str,
        run_dir: Path,
        upstream_dir: Path | None,
        model: str,
    ) -> list[str]:
        assert upstream_dir is not None
        command = [
            request.python_executable,
            str(self.runner_path),
            "--upstream",
            str(upstream_dir),
            "--out-dir",
            str(run_dir.parent),
            "--run-name",
            run_id,
            "--model",
            model,
            "--seed",
            str(request.seed),
        ]
        key_env, base_url = _provider_route(model)
        command.extend(["--api-key-env", key_env, "--base-url", base_url])
        if request.limit is not None:
            command.extend(["--limit", str(request.limit)])
        command.extend(request.runner_args)
        return command

    def required_upstream_files(self, upstream_dir: Path | None) -> tuple[Path, ...]:
        if upstream_dir is None:
            return ()
        return (
            upstream_dir / "data/agent_task.jsonl",
            upstream_dir / "data/all_normal_tools.jsonl",
            upstream_dir / "data/all_attack_tools.jsonl",
        )

    def _python_status(self, version: str) -> str:
        try:
            major, minor = (int(item) for item in version.split(".")[:2])
        except (TypeError, ValueError):
            return "warn"
        return "ok" if (major, minor) == (3, 11) else "error"

    def _version_checks(self, modules: Mapping[str, Any]) -> list[dict[str, Any]]:
        actual = str((modules.get("openai") or {}).get("version") or "unknown")
        status = "ok" if actual == "1.20.0" else "error"
        return [_check("openai_version", status, actual, expected="1.20.0")]

    def summary_path(self, manifest: BenchmarkManifest) -> Path:
        return Path(manifest.output_dir) / "summary.json"

    def split_metrics(self, summary: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        by_mode = summary.get("overall") if isinstance(summary.get("overall"), Mapping) else {}
        official: dict[str, Any] = {"by_mode": {}}
        obligate: dict[str, Any] = {"by_mode": {}}
        rr_heuristic: dict[str, Any] = {}
        for mode, raw in by_mode.items():
            if not isinstance(raw, Mapping):
                continue
            official["by_mode"][str(mode)] = {
                "ASR": raw.get("ASB_Official_Compatible_ASR", raw.get("ASR")),
                "PNA": raw.get("Original_Task_Success"),
            }
            obligate["by_mode"][str(mode)] = {
                "executed_target_tool_asr": raw.get("Executed_Target_Tool_ASR"),
                "unsafe_tool_execution_rate": raw.get("Unsafe_Tool_Execution_Rate"),
                "dangerous_action_blocking_rate": raw.get("Dangerous_Action_Blocking_Rate"),
                "safe_action_pass_rate": raw.get("Safe_Action_Pass_Rate"),
                "decision_counts": raw.get("decision_counts"),
                "theory_metrics": raw.get("theory_metrics"),
            }
            if "RR_heuristic" in raw:
                rr_heuristic[str(mode)] = raw["RR_heuristic"]
        return official, obligate, {
            "metric_semantics": {
                "ASR": "ASB-compatible attacker-goal string match",
                "Original_Task_Success": "PNA-style expected-achievement match",
                "RR_heuristic": "not the official judge-based RR",
            },
            "rr_heuristic_by_mode": rr_heuristic,
        }

    def validate_summary(self, summary: Mapping[str, Any]) -> str | None:
        error = super().validate_summary(summary)
        if error is not None:
            return error
        overall = summary.get("overall")
        if not isinstance(overall, Mapping) or not overall:
            return "ASB summary lacks non-empty overall mode metrics"
        for mode, row in overall.items():
            if not isinstance(row, Mapping):
                return f"ASB summary mode {mode!r} is malformed"
            if not isinstance(row.get("cases"), int):
                return f"ASB summary mode {mode!r} lacks integer case count"
            if "ASR" not in row and "ASB_Official_Compatible_ASR" not in row:
                return f"ASB summary mode {mode!r} lacks ASR"
        return None


class BenchmarkAdapterRegistry:
    """Explicit registry with collision-safe aliases."""

    def __init__(self) -> None:
        self._adapters: dict[str, BenchmarkAdapter] = {}
        self._aliases: dict[str, str] = {}

    def register(self, adapter: BenchmarkAdapter) -> None:
        benchmark = adapter.spec.benchmark
        if benchmark in self._adapters or benchmark in self._aliases:
            raise ValueError(f"benchmark is already registered: {benchmark}")
        aliases = (benchmark, *adapter.spec.aliases)
        collisions = [alias for alias in aliases if alias in self._adapters or alias in self._aliases]
        if collisions:
            raise ValueError(f"benchmark aliases are already registered: {', '.join(collisions)}")
        self._adapters[benchmark] = adapter
        for alias in adapter.spec.aliases:
            self._aliases[alias] = benchmark

    def get(self, benchmark: str) -> BenchmarkAdapter:
        canonical = self._aliases.get(benchmark, benchmark)
        try:
            return self._adapters[canonical]
        except KeyError as exc:
            names = ", ".join(self.names())
            raise KeyError(f"unknown benchmark {benchmark!r}; choose one of: {names}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def adapters(self) -> tuple[BenchmarkAdapter, ...]:
        return tuple(self._adapters[name] for name in self.names())

    def specs(self) -> tuple[BenchmarkSpec, ...]:
        return tuple(adapter.spec for adapter in self.adapters())


def default_registry(repo_root: Path | None = None) -> BenchmarkAdapterRegistry:
    root = (repo_root or discover_repo_root()).resolve()
    registry = BenchmarkAdapterRegistry()
    registry.register(AgentDojoAdapter(root))
    registry.register(AgentSafetyBenchAdapter(root))
    registry.register(AsbIclr2025Adapter(root))
    return registry


def discover_repo_root(start: Path | None = None) -> Path:
    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parents[3])
    for candidate in candidates:
        for path in (candidate, *candidate.parents):
            if (path / "pyproject.toml").is_file() and (path / "src/obligate").is_dir():
                return path
    return Path.cwd().resolve()


def _child_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    src = str(repo_root.resolve() / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = src if not existing else os.pathsep.join((src, existing))
    environment.setdefault("PYTHONUTF8", "1")
    return environment


def _resolve_executable(executable: str) -> str | None:
    path = Path(executable)
    if path.is_file():
        return str(path.resolve())
    return shutil.which(executable)


def _probe_python(executable: str, modules: Iterable[str], repo_root: Path) -> dict[str, Any]:
    script = """
import importlib.metadata
import importlib.util
import json
import sys

mods = {}
for name in sys.argv[1:]:
    available = importlib.util.find_spec(name) is not None
    version = None
    if available:
        distribution = "obligate" if name == "obligate" else name
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    mods[name] = {"available": available, "version": version}
print(json.dumps({"python": ".".join(map(str, sys.version_info[:3])), "modules": mods}))
"""
    try:
        completed = subprocess.run(
            [executable, "-c", script, *modules],
            cwd=repo_root,
            env=_child_environment(repo_root),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    if completed.returncode != 0:
        return {"error": completed.stderr.strip() or "Python probe failed"}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": completed.stdout.strip() or "Python probe returned invalid JSON"}


def _check(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


def _doctor_report(spec: BenchmarkSpec, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_CONTRACT_VERSION,
        "benchmark": spec.benchmark,
        "ready": not any(check["status"] == "error" for check in checks),
        "checks": checks,
        "install_extra": spec.install_extra,
        "upstream_revision": spec.upstream_revision,
    }


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _select(data: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data}


__all__ = [
    "AGENTDOJO_BENCHMARK_VERSION",
    "AGENTDOJO_REVISION",
    "AGENTDOJO_VERSION",
    "AGENT_SAFETYBENCH_REVISION",
    "ASB_ICLR2025_REVISION",
    "AgentDojoAdapter",
    "AgentSafetyBenchAdapter",
    "AsbIclr2025Adapter",
    "BenchmarkAdapter",
    "BenchmarkAdapterRegistry",
    "default_registry",
    "discover_repo_root",
]
