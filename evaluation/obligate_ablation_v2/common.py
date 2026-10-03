"""Small, dependency-free helpers for the isolated ablation-v2 harness."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Project common Python/Pydantic values into deterministic JSON values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
            "length": len(value),
        }
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", exclude_none=False))
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=canonical_json)
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_jsonable(value), ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    """Append one JSON row while serializing writers across Windows processes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.append.lock")
    token = _acquire_lock_file(lock)
    try:
        payload = canonical_json(value)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        _release_lock_file(lock, token)


def inventory_tree(
    root: Path,
    output_manifest: Path | None = None,
    *,
    part_size: int = 100_000,
) -> dict[str, Any]:
    """Build a read-only, deterministic inventory of every file below ``root``.

    ``output_manifest`` may be a ``.jsonl`` file or a directory.  A directory
    produces bounded ``inventory-00000.jsonl`` parts plus ``summary.json``.
    The destination must be outside the source tree, preventing an audit from
    changing or recursively inventorying the data it is meant to freeze.
    """

    source = Path(root).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    destination = Path(output_manifest).resolve() if output_manifest is not None else None
    if destination is not None and _is_within(destination, source):
        raise ValueError("inventory output must be outside the source tree")
    if part_size < 1:
        raise ValueError("part_size must be >= 1")

    # Sorting paths does not read file contents into memory.  Each file is then
    # hashed in chunks and immediately written to JSONL when a destination was
    # requested, so record payloads do not accumulate for large result trees.
    paths = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    tree_digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    records: list[dict[str, Any]] | None = [] if destination is None else None
    manifest_files: list[str] = []
    writer: Any = None
    temporary: Path | None = None
    current_part = -1

    def open_part(part: int) -> None:
        nonlocal writer, temporary, current_part
        if writer is not None:
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
            assert temporary is not None
            _replace_with_retry(temporary, _part_target(destination, current_part))
        assert destination is not None
        current_part = part
        target = _part_target(destination, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        writer = temporary.open("w", encoding="utf-8", newline="\n")
        manifest_files.append(str(target))

    try:
        for path in paths:
            stat = path.stat()
            record = {
                "path": path.relative_to(source).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
            encoded = canonical_json(record).encode("utf-8") + b"\n"
            tree_digest.update(encoded)
            total_bytes += stat.st_size
            if records is not None:
                records.append(record)
            else:
                target_part = file_count // part_size if destination.is_dir() or destination.suffix == "" else 0
                if writer is None or target_part != current_part:
                    open_part(target_part)
                writer.write(encoded.decode("utf-8"))
            file_count += 1
    finally:
        if writer is not None:
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
            assert temporary is not None and destination is not None
            _replace_with_retry(temporary, _part_target(destination, current_part))

    summary: dict[str, Any] = {
        "schema_version": "obligate-tree-inventory-v1",
        "root": str(source),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "manifest_files": manifest_files,
    }
    if records is not None:
        summary["files"] = records
    elif destination is not None and (destination.is_dir() or destination.suffix == ""):
        atomic_write_json(destination / "summary.json", summary)
    return summary


def _part_target(destination: Path | None, part: int) -> Path:
    assert destination is not None
    if destination.is_dir() or destination.suffix == "":
        return destination / f"inventory-{part:05d}.jsonl"
    return destination


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 80) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    if last_error is not None:
        raise last_error
    os.replace(source, target)


def _acquire_lock_file(
    path: Path,
    *,
    timeout_seconds: float = 60.0,
    stale_seconds: float = 300.0,
) -> str:
    token = f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                if age > stale_seconds:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring append lock: {path}")
            time.sleep(0.01)
            continue
        try:
            os.write(descriptor, token.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return token


def _release_lock_file(path: Path, token: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == token:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        return


__all__ = [
    "append_jsonl",
    "atomic_write_json",
    "canonical_json",
    "inventory_tree",
    "sha256_file",
    "utc_now",
]
