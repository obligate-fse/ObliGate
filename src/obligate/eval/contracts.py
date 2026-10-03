"""Versioned contracts shared by all external benchmark adapters.

The contracts deliberately use only the Python standard library.  Benchmark
dependencies live in isolated interpreters and communicate with this layer
through command arguments and JSON artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

BENCHMARK_CONTRACT_VERSION = "obligate.benchmark/v1"
EVENT_KINDS = frozenset(
    {
        "diagnostic",
        "run_started",
        "process_started",
        "process_finished",
        "user",
        "observation",
        "action_proposal",
        "decision",
        "action_result",
        "final",
        "run_finished",
    }
)
RESULT_STATUSES = frozenset({"planned", "succeeded", "failed"})
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with a stable ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_run_id(benchmark: str) -> str:
    """Create a filesystem-safe run identifier."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{benchmark}-{stamp}-{uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> None:
    if not _SAFE_RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("run_id must be 1-128 safe filename characters and may not contain '..'")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Pinned identity and metric namespace for a benchmark adapter."""

    benchmark: str
    display_name: str
    adapter: str
    upstream_url: str
    upstream_revision: str
    upstream_version: str | None
    benchmark_version: str | None
    python_recommendation: str
    install_extra: str
    official_metrics: tuple[str, ...]
    obligate_metrics: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    schema_version: str = BENCHMARK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_CONTRACT_VERSION:
            raise ValueError(f"unsupported benchmark contract version: {self.schema_version}")
        if not self.benchmark or not self.adapter:
            raise ValueError("benchmark and adapter are required")
        if not self.upstream_url.startswith("https://"):
            raise ValueError("upstream_url must be an HTTPS URL")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "display_name": self.display_name,
            "adapter": self.adapter,
            "upstream": {
                "url": self.upstream_url,
                "revision": self.upstream_revision,
                "package_version": self.upstream_version,
                "benchmark_version": self.benchmark_version,
            },
            "python_recommendation": self.python_recommendation,
            "install_extra": self.install_extra,
            "official_metrics": list(self.official_metrics),
            "obligate_metrics": list(self.obligate_metrics),
            "aliases": list(self.aliases),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkSpec":
        upstream = data.get("upstream") or {}
        return cls(
            schema_version=str(data.get("schema_version") or BENCHMARK_CONTRACT_VERSION),
            benchmark=str(data["benchmark"]),
            display_name=str(data["display_name"]),
            adapter=str(data["adapter"]),
            upstream_url=str(upstream["url"]),
            upstream_revision=str(upstream["revision"]),
            upstream_version=None if upstream.get("package_version") is None else str(upstream["package_version"]),
            benchmark_version=None if upstream.get("benchmark_version") is None else str(upstream["benchmark_version"]),
            python_recommendation=str(data["python_recommendation"]),
            install_extra=str(data["install_extra"]),
            official_metrics=tuple(str(item) for item in data.get("official_metrics", ())),
            obligate_metrics=tuple(str(item) for item in data.get("obligate_metrics", ())),
            aliases=tuple(str(item) for item in data.get("aliases", ())),
            notes=tuple(str(item) for item in data.get("notes", ())),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRunRequest:
    """Runtime-neutral request consumed by a registered adapter."""

    benchmark: str
    repo_root: Path
    output_root: Path
    python_executable: str
    run_id: str | None = None
    upstream_dir: Path | None = None
    model: str | None = None
    seed: int = 0
    limit: int | None = None
    runner_args: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if not self.python_executable:
            raise ValueError("python_executable is required")


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Self-contained, replayable launch manifest written before execution."""

    spec: BenchmarkSpec
    run_id: str
    command: tuple[str, ...]
    cwd: str
    output_dir: str
    python_executable: str
    upstream_dir: str | None = None
    model: str | None = None
    seed: int = 0
    limit: int | None = None
    runner_args: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    environment_keys: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    schema_version: str = BENCHMARK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_CONTRACT_VERSION:
            raise ValueError(f"unsupported benchmark contract version: {self.schema_version}")
        validate_run_id(self.run_id)
        if not self.command:
            raise ValueError("manifest command may not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "benchmark_manifest",
            "created_at": self.created_at,
            "run_id": self.run_id,
            "benchmark_spec": self.spec.to_dict(),
            "launch": {
                "command": list(self.command),
                "cwd": self.cwd,
                "output_dir": self.output_dir,
                "python_executable": self.python_executable,
                "upstream_dir": self.upstream_dir,
                "model": self.model,
                "seed": self.seed,
                "limit": self.limit,
                "runner_args": list(self.runner_args),
                "options": _json_value(self.options),
                # Values are intentionally excluded so credentials never enter artifacts.
                "inherited_environment_keys": list(self.environment_keys),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkManifest":
        launch = data.get("launch") or {}
        return cls(
            schema_version=str(data.get("schema_version") or BENCHMARK_CONTRACT_VERSION),
            created_at=str(data.get("created_at") or utc_now()),
            run_id=str(data["run_id"]),
            spec=BenchmarkSpec.from_dict(data["benchmark_spec"]),
            command=tuple(str(item) for item in launch.get("command", ())),
            cwd=str(launch["cwd"]),
            output_dir=str(launch["output_dir"]),
            python_executable=str(launch["python_executable"]),
            upstream_dir=None if launch.get("upstream_dir") is None else str(launch["upstream_dir"]),
            model=None if launch.get("model") is None else str(launch["model"]),
            seed=int(launch.get("seed", 0)),
            limit=None if launch.get("limit") is None else int(launch["limit"]),
            runner_args=tuple(str(item) for item in launch.get("runner_args", ())),
            options=dict(launch.get("options") or {}),
            environment_keys=tuple(str(item) for item in launch.get("inherited_environment_keys", ())),
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True, slots=True)
class BenchmarkEvent:
    """Append-only event envelope shared across benchmark trajectories."""

    benchmark: str
    run_id: str
    sequence: int
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    case_id: str | None = None
    timestamp: str = field(default_factory=utc_now)
    schema_version: str = BENCHMARK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_CONTRACT_VERSION:
            raise ValueError(f"unsupported benchmark contract version: {self.schema_version}")
        validate_run_id(self.run_id)
        if self.sequence < 0:
            raise ValueError("event sequence may not be negative")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unsupported benchmark event kind: {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "sequence": self.sequence,
            "payload": _json_value(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkEvent":
        return cls(
            schema_version=str(data.get("schema_version") or BENCHMARK_CONTRACT_VERSION),
            timestamp=str(data.get("timestamp") or utc_now()),
            benchmark=str(data["benchmark"]),
            run_id=str(data["run_id"]),
            case_id=None if data.get("case_id") is None else str(data["case_id"]),
            sequence=int(data["sequence"]),
            kind=str(data["kind"]),
            payload=dict(data.get("payload") or {}),
        )

    def append(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")
        return path


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Unified result envelope; benchmark-native artifacts remain authoritative."""

    manifest: BenchmarkManifest
    status: str
    exit_code: int | None
    official_metrics: Mapping[str, Any] = field(default_factory=dict)
    obligate_metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = BENCHMARK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_CONTRACT_VERSION:
            raise ValueError(f"unsupported benchmark contract version: {self.schema_version}")
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"unsupported benchmark result status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "benchmark_result",
            "benchmark": self.manifest.spec.benchmark,
            "run_id": self.manifest.run_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "official_metrics": _json_value(self.official_metrics),
            "obligate_metrics": _json_value(self.obligate_metrics),
            "artifacts": _json_value(self.artifacts),
            "error": self.error,
            "metadata": _json_value(self.metadata),
            "manifest": self.manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkResult":
        return cls(
            schema_version=str(data.get("schema_version") or BENCHMARK_CONTRACT_VERSION),
            manifest=BenchmarkManifest.from_dict(data["manifest"]),
            status=str(data["status"]),
            exit_code=None if data.get("exit_code") is None else int(data["exit_code"]),
            official_metrics=dict(data.get("official_metrics") or {}),
            obligate_metrics=dict(data.get("obligate_metrics") or {}),
            artifacts={str(key): str(value) for key, value in (data.get("artifacts") or {}).items()},
            started_at=None if data.get("started_at") is None else str(data["started_at"]),
            finished_at=None if data.get("finished_at") is None else str(data["finished_at"]),
            error=None if data.get("error") is None else str(data["error"]),
            metadata=dict(data.get("metadata") or {}),
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def read_events(path: Path) -> list[BenchmarkEvent]:
    """Read a JSONL event stream, ignoring blank lines but not malformed data."""

    events: list[BenchmarkEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(BenchmarkEvent.from_dict(json.loads(line)))
    return events


def ensure_unique_event_sequences(events: Sequence[BenchmarkEvent]) -> None:
    sequences = [event.sequence for event in events]
    if sequences != list(range(len(sequences))):
        raise ValueError("event sequences must be contiguous and start at zero")
