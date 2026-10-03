"""Cross-process, exact-key OpenAI ChatCompletion cache for Ablation v2.

The cache is deliberately isolated from production code.  It stores only a
hash of the normalized request material and the provider's ChatCompletion
JSON.  Prompts, tool state, credentials, and API headers are never written to
the audit log or cache metadata.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from openai.types.chat import ChatCompletion

from .common import append_jsonl, canonical_json, utc_now

CACHE_SCHEMA_VERSION = "obligate-model-cache-v1"
PROMPT_MATERIAL_VERSION = "obligate-prompt-material-v1"

_VISIBLE_STATE_KEYS = frozenset(
    {
        "_obligate_visible_tool_state",
        "_obligate_visible_tool_state_digest",
        "visible_tool_state",
        "visible_tool_state_digest",
        "tool_state",
        "tool_state_digest",
    }
)
_INTERNAL_REQUEST_KEYS = _VISIBLE_STATE_KEYS | frozenset(
    {
        "_obligate_tool_schema_digest",
        "_obligate_cache_metadata",
    }
)
_PRIMARY_REQUEST_KEYS = frozenset({"model", "messages", "tools", "temperature"})
_TRANSPORT_ONLY_KEYS = frozenset({"extra_headers", "extra_query", "timeout"})
_SENSITIVE_KEYS = frozenset(
    {
        "api-key",
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
    }
)


class CacheIntegrityError(RuntimeError):
    """Raised instead of silently reusing a corrupted or ambiguous row."""


class CacheLeaseLost(RuntimeError):
    """Raised when a writer no longer owns the cross-process cache lease."""


@dataclass(frozen=True, slots=True)
class CachedCompletion:
    prompt_hash: str
    request_hash: str
    response_hash: str
    response: ChatCompletion
    usage: dict[str, Any]
    created_at: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    """Normalize representation without collapsing security-relevant text."""

    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump(mode="json", exclude_none=False))
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalize(item) for item in value]
        return sorted(items, key=canonical_json)
    if hasattr(value, "__dict__"):
        return _normalize(vars(value))
    return _normalize_text(str(value))


def _mapping(value: Any) -> dict[str, Any]:
    normalized = _normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"request must normalize to an object, got {type(normalized).__name__}")
    return normalized


def _valid_sha256(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else None


def _visible_history(messages: list[Any]) -> list[Any]:
    visible: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "tool" or message.get("tool_calls") is not None:
            visible.append(message)
    return visible


def _visible_tool_state_digest(
    request: Mapping[str, Any], messages: list[Any]
) -> tuple[str, str]:
    supplied_digests: list[str] = []
    for key in ("_obligate_visible_tool_state_digest", "visible_tool_state_digest", "tool_state_digest"):
        if key in request:
            digest = _valid_sha256(request.get(key))
            if digest is None:
                raise ValueError(f"{key} must be a lowercase or uppercase SHA256 digest")
            supplied_digests.append(digest)
    if len(set(supplied_digests)) > 1:
        raise ValueError("conflicting visible tool state digests were supplied")
    supplied_states: list[str] = []
    for key in ("_obligate_visible_tool_state", "visible_tool_state", "tool_state"):
        if key in request:
            supplied_states.append(_sha256_text(canonical_json(_normalize(request.get(key)))))
    if len(set(supplied_states)) > 1:
        raise ValueError("conflicting visible tool state values were supplied")
    if supplied_digests and supplied_states and supplied_digests[0] != supplied_states[0]:
        raise ValueError("visible tool state does not match its supplied digest")
    if supplied_digests:
        return supplied_digests[0], "explicit_digest"
    if supplied_states:
        source = "explicit_case_reset_state" if "_obligate_visible_tool_state" in request else "explicit_state"
        return supplied_states[0], source
    # OpenAI requests do not otherwise expose the simulator object.  The model-
    # visible tool-call/result subsequence is the conservative fallback state.
    return (
        _sha256_text(canonical_json(_visible_history(messages))),
        "message_tool_subsequence_fallback",
    )


def _system_prompt(messages: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") not in {"system", "developer"}:
            continue
        result.append(
            {
                "role": message.get("role"),
                "name": message.get("name"),
                "content": message.get("content"),
            }
        )
    return result


def compute_prompt_material(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete normalized cache-key material plus ``prompt_hash``.

    The returned object is suitable for preflight inspection but must not be
    written to the cache audit because it contains the visible conversation.
    """

    raw = dict(request)
    messages_value = _normalize(raw.get("messages") or [])
    if not isinstance(messages_value, list):
        raise TypeError("OpenAI messages must normalize to a list")
    tools = _normalize(raw.get("tools") or [])
    temperature = raw.get("temperature", 0.0)
    if temperature is None:
        temperature = 0.0
    temperature = float(temperature)

    model_parameters: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _PRIMARY_REQUEST_KEYS or key in _INTERNAL_REQUEST_KEYS or key in _TRANSPORT_ONLY_KEYS:
            continue
        model_parameters[str(key)] = _scrub_sensitive(_normalize(value))

    tool_schema_digest = _sha256_text(canonical_json(tools))
    supplied_tool_digest = raw.get("_obligate_tool_schema_digest")
    if supplied_tool_digest is not None:
        validated_tool_digest = _valid_sha256(supplied_tool_digest)
        if validated_tool_digest is None:
            raise ValueError("_obligate_tool_schema_digest must be a SHA256 digest")
        if validated_tool_digest != tool_schema_digest:
            raise ValueError("tool schema does not match _obligate_tool_schema_digest")

    state_digest, state_source = _visible_tool_state_digest(raw, messages_value)
    material = {
        "schema_version": PROMPT_MATERIAL_VERSION,
        "model": _normalize_text(str(raw.get("model") or "")),
        "normalized_system_prompt": _system_prompt(messages_value),
        # This is intentionally the complete history, not only user/assistant
        # text.  Tool calls, call IDs, results and ordering are hash-bound.
        "normalized_history": messages_value,
        "tool_schema_digest": tool_schema_digest,
        "visible_tool_state_digest": state_digest,
        "visible_tool_state_source": state_source,
        "temperature": temperature,
        "model_parameters": model_parameters,
    }
    return {**material, "prompt_hash": _sha256_text(canonical_json(material))}


def _scrub_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().casefold()
            if normalized_key in _SENSITIVE_KEYS:
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = _scrub_sensitive(item)
        return result
    if isinstance(value, list):
        return [_scrub_sensitive(item) for item in value]
    return value


def _provider_request(request: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in request.items() if key not in _INTERNAL_REQUEST_KEYS}


def _request_hash(request: Mapping[str, Any]) -> str:
    normalized = _scrub_sensitive(_normalize(dict(request)))
    return _sha256_text(canonical_json(normalized))


def _completion_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, ChatCompletion):
        payload = value.model_dump(mode="json", exclude_none=False)
    elif hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", exclude_none=False)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"expected OpenAI ChatCompletion-compatible response, got {type(value).__name__}")
    # model_validate is intentional: both hits and misses return the same
    # OpenAI 2.45 typed object rather than a handwritten compatibility object.
    return ChatCompletion.model_validate(payload).model_dump(mode="json", exclude_none=False)


def _validated_completion(payload: Mapping[str, Any]) -> ChatCompletion:
    return ChatCompletion.model_validate(dict(payload))


def _usage_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


class SQLiteModelCache:
    """SQLite WAL cache with an expiring cross-process single-flight lease."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = 30_000,
        lease_seconds: float = 600.0,
        poll_seconds: float = 0.025,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.lease_seconds = float(lease_seconds)
        self.poll_seconds = float(poll_seconds)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS completions (
                    prompt_hash TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    response_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    creator_pid INTEGER NOT NULL,
                    schema_version TEXT NOT NULL,
                    openai_version TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    prompt_hash TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    claimed_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS claims_expiry_idx ON claims(expires_at)")

        self._with_retry(operation)

    def _with_retry(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(80):
            connection = self._connect()
            try:
                return operation(connection)
            except sqlite3.OperationalError as exc:
                message = str(exc).casefold()
                if "locked" not in message and "busy" not in message:
                    raise
                last_error = exc
                time.sleep(min(0.01 * (attempt + 1), 0.25))
            finally:
                connection.close()
        assert last_error is not None
        raise last_error

    def lookup(self, prompt_hash: str) -> CachedCompletion | None:
        def operation(connection: sqlite3.Connection) -> sqlite3.Row | None:
            return connection.execute(
                "SELECT * FROM completions WHERE prompt_hash = ?", (prompt_hash,)
            ).fetchone()

        row = self._with_retry(operation)
        if row is None:
            return None
        try:
            payload = json.loads(str(row["response_json"]))
            usage = json.loads(str(row["usage_json"]))
        except json.JSONDecodeError as exc:
            raise CacheIntegrityError(f"invalid cached JSON for {prompt_hash}") from exc
        actual_hash = _sha256_text(canonical_json(payload))
        if actual_hash != str(row["response_hash"]):
            raise CacheIntegrityError(f"cached response hash mismatch for {prompt_hash}")
        return CachedCompletion(
            prompt_hash=prompt_hash,
            request_hash=str(row["request_hash"]),
            response_hash=actual_hash,
            response=_validated_completion(payload),
            usage=dict(usage) if isinstance(usage, dict) else {},
            created_at=str(row["created_at"]),
        )

    def try_claim(self, prompt_hash: str, owner: str) -> bool:
        now = time.time()

        def operation(connection: sqlite3.Connection) -> bool:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if connection.execute(
                    "SELECT 1 FROM completions WHERE prompt_hash = ?", (prompt_hash,)
                ).fetchone():
                    connection.execute("COMMIT")
                    return False
                connection.execute("DELETE FROM claims WHERE expires_at <= ?", (now,))
                row = connection.execute(
                    "SELECT owner FROM claims WHERE prompt_hash = ?", (prompt_hash,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO claims(prompt_hash, owner, claimed_at, expires_at) VALUES(?, ?, ?, ?)",
                        (prompt_hash, owner, now, now + self.lease_seconds),
                    )
                    connection.execute("COMMIT")
                    return True
                owned = str(row["owner"]) == owner
                connection.execute("COMMIT")
                return owned
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        return bool(self._with_retry(operation))

    def store(
        self,
        *,
        prompt_hash: str,
        owner: str,
        request_hash: str,
        response_payload: Mapping[str, Any],
        model: str,
        temperature: float,
    ) -> CachedCompletion:
        payload = _completion_payload(response_payload)
        response_json = canonical_json(payload)
        response_hash = _sha256_text(response_json)
        usage = _usage_from_payload(payload)
        usage_json = canonical_json(usage)
        created_at = utc_now()
        try:
            import openai

            openai_version = str(openai.__version__)
        except Exception:  # pragma: no cover - ChatCompletion import already proves dependency
            openai_version = "unknown"

        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT 1 FROM completions WHERE prompt_hash = ?", (prompt_hash,)
                ).fetchone()
                if existing is not None:
                    connection.execute("DELETE FROM claims WHERE prompt_hash = ? AND owner = ?", (prompt_hash, owner))
                    connection.execute("COMMIT")
                    return
                claim = connection.execute(
                    "SELECT owner FROM claims WHERE prompt_hash = ?", (prompt_hash,)
                ).fetchone()
                if claim is None or str(claim["owner"]) != owner:
                    raise CacheLeaseLost(f"cache lease lost before store for {prompt_hash}")
                connection.execute(
                    """
                    INSERT INTO completions(
                        prompt_hash, request_hash, response_hash, response_json,
                        usage_json, model, temperature, created_at, creator_pid,
                        schema_version, openai_version
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt_hash,
                        request_hash,
                        response_hash,
                        response_json,
                        usage_json,
                        model,
                        float(temperature),
                        created_at,
                        os.getpid(),
                        CACHE_SCHEMA_VERSION,
                        openai_version,
                    ),
                )
                connection.execute("DELETE FROM claims WHERE prompt_hash = ? AND owner = ?", (prompt_hash, owner))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        self._with_retry(operation)
        stored = self.lookup(prompt_hash)
        if stored is None:  # pragma: no cover - transaction invariant
            raise CacheIntegrityError(f"stored completion disappeared for {prompt_hash}")
        return stored

    def release_claim(self, prompt_hash: str, owner: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                "DELETE FROM claims WHERE prompt_hash = ? AND owner = ?", (prompt_hash, owner)
            )

        self._with_retry(operation)

    def journal_mode(self) -> str:
        def operation(connection: sqlite3.Connection) -> str:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

        return str(self._with_retry(operation))


class _CachedCompletionsProxy:
    def __init__(
        self,
        delegate: Any,
        *,
        cache: SQLiteModelCache,
        audit_path: Path,
        model_label: str,
        variant: str,
        case_id: str,
        temperature: float,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._audit_path = Path(audit_path)
        self._model_label = str(model_label)
        self._variant = str(variant)
        self._case_id = str(case_id)
        self._temperature = float(temperature)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        if args:
            raise TypeError("instrumented OpenAI chat.completions.create requires keyword arguments")
        request = dict(kwargs)
        requested_temperature = request.get("temperature", self._temperature)
        # AgentDojo 0.1.35 turns an intended 0.0 into OpenAI's NOT_GIVEN
        # sentinel.  Treat that sentinel as absent and explicitly restore the
        # frozen 0.0 value before hashing and provider dispatch.
        if requested_temperature is None or type(requested_temperature).__name__ in {
            "NotGiven",
            "NotGivenType",
        }:
            requested_temperature = self._temperature
        if float(requested_temperature) != self._temperature:
            raise ValueError(
                f"frozen experiment temperature is {self._temperature}, got {requested_temperature}"
            )
        if bool(request.get("stream", False)):
            raise ValueError("streaming responses are not cacheable in the frozen experiment")
        request["temperature"] = self._temperature
        request["stream"] = False
        request.setdefault("model", self._model_label)
        if not any(key in request for key in _VISIBLE_STATE_KEYS):
            visible_history = _visible_history(_normalize(request.get("messages") or []))
            # AgentDojo resets every case/configuration to the same case-local
            # initial simulator state.  Within that case, deterministic tool
            # transitions are completely represented by the visible ordered
            # tool-call/result subsequence.  Bind both the reset identity and
            # its revision so identical text from different cases cannot share.
            request["_obligate_visible_tool_state"] = {
                "schema_version": "agentdojo-case-reset-visible-state-v1",
                "case_id": self._case_id,
                "reset_protocol": "fresh_simulator_per_case_and_variant",
                "visible_transition_revision": len(visible_history),
                "visible_transition_digest": _sha256_text(canonical_json(visible_history)),
            }
        material = compute_prompt_material(request)
        state_source = str(material["visible_tool_state_source"])
        if state_source != "explicit_case_reset_state":
            raise CacheIntegrityError(
                f"formal cache request used unsupported tool-state source: {state_source}"
            )
        prompt_hash = str(material["prompt_hash"])
        provider_request = _provider_request(request)
        request_hash = _request_hash(provider_request)
        owner = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        started = time.perf_counter()
        waited = False
        provider_called = False

        try:
            while True:
                cached = self._cache.lookup(prompt_hash)
                if cached is not None:
                    self._audit(
                        cache_status="hit_after_wait" if waited else "hit",
                        cache_hit=True,
                        provider_called=False,
                        prompt_hash=prompt_hash,
                        request_hash=request_hash,
                        response_hash=cached.response_hash,
                        usage=cached.usage,
                        started=started,
                        visible_tool_state_source=state_source,
                    )
                    return cached.response
                if self._cache.try_claim(prompt_hash, owner):
                    break
                waited = True
                time.sleep(self._cache.poll_seconds)

            try:
                provider_called = True
                raw_response = self._delegate.create(**provider_request)
                payload = _completion_payload(raw_response)
                cached = self._cache.store(
                    prompt_hash=prompt_hash,
                    owner=owner,
                    request_hash=request_hash,
                    response_payload=payload,
                    model=str(material["model"]),
                    temperature=self._temperature,
                )
            except BaseException:
                self._cache.release_claim(prompt_hash, owner)
                raise
            self._audit(
                cache_status="miss",
                cache_hit=False,
                provider_called=True,
                prompt_hash=prompt_hash,
                request_hash=request_hash,
                response_hash=cached.response_hash,
                usage=cached.usage,
                started=started,
                visible_tool_state_source=state_source,
            )
            return cached.response
        except BaseException as exc:
            self._audit(
                cache_status="error",
                cache_hit=False,
                provider_called=provider_called,
                prompt_hash=prompt_hash,
                request_hash=request_hash,
                response_hash=None,
                usage={},
                started=started,
                visible_tool_state_source=state_source,
                error=exc,
            )
            raise

    def _audit(
        self,
        *,
        cache_status: str,
        cache_hit: bool,
        provider_called: bool,
        prompt_hash: str,
        request_hash: str,
        response_hash: str | None,
        usage: Mapping[str, Any],
        started: float,
        visible_tool_state_source: str,
        error: BaseException | None = None,
    ) -> None:
        row = {
            "schema_version": "obligate-model-cache-audit-v1",
            "timestamp": utc_now(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "model": self._model_label,
            "variant": self._variant,
            "case_id": self._case_id,
            "temperature": self._temperature,
            "cache_status": cache_status,
            "cache_hit": cache_hit,
            "provider_called": provider_called,
            "prompt_hash": prompt_hash,
            "request_hash": request_hash,
            "response_hash": response_hash,
            "token_usage": dict(usage),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "visible_tool_state_source": visible_tool_state_source,
            "visible_tool_state_fallback_used": visible_tool_state_source
            == "message_tool_subsequence_fallback",
        }
        if error is not None:
            row["error_type"] = type(error).__name__
            row["error_message"] = _redact_error(error)
        append_jsonl(self._audit_path, row)


class _ChatProxy:
    def __init__(self, delegate: Any, completions: _CachedCompletionsProxy) -> None:
        self._delegate = delegate
        self.completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _OpenAIClientProxy:
    def __init__(
        self,
        delegate: Any,
        *,
        cache: SQLiteModelCache,
        audit_path: Path,
        model_label: str,
        variant: str,
        case_id: str,
        temperature: float,
    ) -> None:
        self._delegate = delegate
        completions = _CachedCompletionsProxy(
            delegate.chat.completions,
            cache=cache,
            audit_path=audit_path,
            model_label=model_label,
            variant=variant,
            case_id=case_id,
            temperature=temperature,
        )
        self.chat = _ChatProxy(delegate.chat, completions)
        self._proxy_config = {
            "cache": cache,
            "audit_path": audit_path,
            "model_label": model_label,
            "variant": variant,
            "case_id": case_id,
            "temperature": temperature,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def __enter__(self) -> "_OpenAIClientProxy":
        self._delegate.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._delegate.__exit__(exc_type, exc, tb)

    def with_options(self, **kwargs: Any) -> "_OpenAIClientProxy":
        return _OpenAIClientProxy(self._delegate.with_options(**kwargs), **self._proxy_config)


def _redact_error(value: BaseException | str, limit: int = 800) -> str:
    text = str(value)
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted-api-key>", text)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    return text[:limit]


def instrument_openai(
    original: Callable[..., Any],
    *,
    cache_path: Path,
    audit_path: Path,
    model_label: str,
    variant: str,
    case_id: str,
    temperature: float = 0.0,
) -> Callable[..., Any]:
    """Return an OpenAI-constructor factory backed by the shared WAL cache."""

    frozen_temperature = float(temperature)
    if frozen_temperature != 0.0:
        raise ValueError("ObliGate Ablation v2 requires explicit temperature=0.0")
    cache = SQLiteModelCache(Path(cache_path))

    def factory(*args: Any, **kwargs: Any) -> _OpenAIClientProxy:
        client = original(*args, **kwargs)
        return _OpenAIClientProxy(
            client,
            cache=cache,
            audit_path=Path(audit_path),
            model_label=model_label,
            variant=variant,
            case_id=case_id,
            temperature=frozen_temperature,
        )

    factory.__name__ = getattr(original, "__name__", "InstrumentedOpenAI")
    factory.__qualname__ = getattr(original, "__qualname__", factory.__name__)
    factory.__doc__ = getattr(original, "__doc__", None)
    return factory


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "PROMPT_MATERIAL_VERSION",
    "CacheIntegrityError",
    "CacheLeaseLost",
    "CachedCompletion",
    "SQLiteModelCache",
    "compute_prompt_material",
    "instrument_openai",
]
