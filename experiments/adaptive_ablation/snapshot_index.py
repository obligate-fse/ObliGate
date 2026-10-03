"""Low-memory disk-offset index for multi-gigabyte snapshot JSONL streams."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class IndexedSnapshot:
    snapshot_id: str
    ordinal: int
    input_offset: int | None
    input_length: int | None
    decision_offset: int | None
    decision_length: int | None
    tool_gate_offset: int | None
    tool_gate_length: int | None
    decision_error_offset: int | None
    decision_error_length: int | None


class SnapshotEventIndex:
    def __init__(self, source: Path, index_path: Path) -> None:
        self.source = source.resolve()
        self.index_path = index_path.resolve()
        self._connection = sqlite3.connect(str(self.index_path))
        self._handle = self.source.open("rb")

    @classmethod
    def build(cls, source: Path, index_path: Path) -> "SnapshotEventIndex":
        source = source.resolve()
        index_path = index_path.resolve()
        if _index_matches(source, index_path):
            return cls(source, index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = index_path.with_suffix(index_path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(str(temporary))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                """
                CREATE TABLE events (
                    snapshot_id TEXT PRIMARY KEY,
                    ordinal INTEGER NOT NULL,
                    input_offset INTEGER,
                    input_length INTEGER,
                    decision_offset INTEGER,
                    decision_length INTEGER,
                    tool_gate_offset INTEGER,
                    tool_gate_length INTEGER,
                    decision_error_offset INTEGER,
                    decision_error_length INTEGER
                )
                """
            )
            connection.execute("CREATE INDEX events_ordinal ON events(ordinal)")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            ordinal = 0
            row_count = 0
            with source.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    snapshot_id = str(event["snapshot_id"])
                    event_type = str(event["event"])
                    column = {
                        "input": "input",
                        "decision": "decision",
                        "tool_gate": "tool_gate",
                        "decision_error": "decision_error",
                    }.get(event_type)
                    if column is None:
                        continue
                    ordinal += 1
                    connection.execute(
                        "INSERT OR IGNORE INTO events(snapshot_id, ordinal) VALUES (?, ?)",
                        (snapshot_id, ordinal),
                    )
                    connection.execute(
                        f"UPDATE events SET {column}_offset = ?, {column}_length = ? WHERE snapshot_id = ?",
                        (offset, len(line), snapshot_id),
                    )
                    row_count += 1
                    if row_count % 10_000 == 0:
                        connection.commit()
            stat = source.stat()
            metadata = {
                "schema_version": "obligate-snapshot-offset-index-v1",
                "source": str(source),
                "source_size": str(stat.st_size),
                "source_mtime_ns": str(stat.st_mtime_ns),
                "indexed_event_row_count": str(row_count),
                "complete": "1",
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, index_path)
        return cls(source, index_path)

    def records(self) -> Iterator[IndexedSnapshot]:
        cursor = self._connection.execute(
            """
            SELECT snapshot_id, ordinal,
                   input_offset, input_length,
                   decision_offset, decision_length,
                   tool_gate_offset, tool_gate_length,
                   decision_error_offset, decision_error_length
            FROM events ORDER BY ordinal
            """
        )
        for values in cursor:
            yield IndexedSnapshot(*values)

    def load(self, offset: int | None, length: int | None) -> dict[str, Any] | None:
        if offset is None or length is None:
            return None
        self._handle.seek(offset)
        payload = self._handle.read(length)
        return json.loads(payload)

    def metadata(self) -> dict[str, str]:
        return dict(self._connection.execute("SELECT key, value FROM metadata"))

    def close(self) -> None:
        self._handle.close()
        self._connection.close()

    def __enter__(self) -> "SnapshotEventIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _index_matches(source: Path, index_path: Path) -> bool:
    if not index_path.exists():
        return False
    try:
        connection = sqlite3.connect(str(index_path))
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        connection.close()
        stat = source.stat()
        return (
            metadata.get("schema_version") == "obligate-snapshot-offset-index-v1"
            and metadata.get("source") == str(source)
            and metadata.get("source_size") == str(stat.st_size)
            and metadata.get("source_mtime_ns") == str(stat.st_mtime_ns)
            and metadata.get("complete") == "1"
        )
    except (OSError, sqlite3.Error, ValueError):
        return False


__all__ = ["IndexedSnapshot", "SnapshotEventIndex"]
