"""Provider model-call ledger with request/response hashes and token usage."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


class ProviderUsageLedger:
    def __init__(self, path: Path, *, benchmark: str, run_label: str) -> None:
        self.path = path.resolve()
        self.benchmark = benchmark
        self.run_label = run_label
        self._lock = threading.RLock()
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def instrument_openai(self, original: Callable[..., Any]) -> Callable[..., Any]:
        ledger = self

        def factory(*args: Any, **kwargs: Any) -> _ClientProxy:
            # Constructor arguments may contain the API key and are therefore
            # deliberately neither hashed nor logged.
            return _ClientProxy(original(*args, **kwargs), ledger)

        return factory

    def record_success(self, request: Mapping[str, Any], response: Any) -> None:
        response_value = _object_to_json(response)
        usage = _usage(response)
        self._append(
            {
                "status": "success",
                "request_sha256": _sha256_json(_public_request(request)),
                "response_sha256": _sha256_json(response_value),
                "provider_model_identifier": getattr(response, "model", None) or request.get("model"),
                "provider_fingerprint": getattr(response, "system_fingerprint", None),
                **usage,
            }
        )

    def record_error(self, request: Mapping[str, Any], error: BaseException) -> None:
        self._append(
            {
                "status": "error",
                "request_sha256": _sha256_json(_public_request(request)),
                "response_sha256": None,
                "provider_model_identifier": request.get("model"),
                "provider_fingerprint": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "token_usage_available": False,
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            }
        )

    def _append(self, value: Mapping[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            row = {
                "schema_version": "obligate-provider-call-ledger-v1",
                "benchmark": self.benchmark,
                "run_label": self.run_label,
                "model_call_index": self._sequence,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **value,
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


class _ClientProxy:
    def __init__(self, client: Any, ledger: ProviderUsageLedger) -> None:
        self._client = client
        self.chat = _ChatProxy(client.chat, ledger)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, chat: Any, ledger: ProviderUsageLedger) -> None:
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions, ledger)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _CompletionsProxy:
    def __init__(self, completions: Any, ledger: ProviderUsageLedger) -> None:
        self._completions = completions
        self._ledger = ledger

    def create(self, *args: Any, **kwargs: Any) -> Any:
        request = dict(kwargs)
        if args:
            request["_positional_argument_count"] = len(args)
        try:
            response = self._completions.create(*args, **kwargs)
        except BaseException as exc:
            self._ledger.record_error(request, exc)
            raise
        self._ledger.record_success(request, response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


def summarize_ledgers(
    paths: list[Path],
    output: Path,
    *,
    pricing: Mapping[str, float] | None = None,
    victim_episode_count: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    cached_tokens = sum(int(row.get("prompt_cache_hit_tokens") or 0) for row in rows)
    price = None
    if pricing is not None:
        uncached = max(0, input_tokens - cached_tokens)
        price = (
            uncached * float(pricing.get("input_per_million", 0.0))
            + cached_tokens * float(pricing.get("cached_input_per_million", pricing.get("input_per_million", 0.0)))
            + output_tokens * float(pricing.get("output_per_million", 0.0))
        ) / 1_000_000
    by_benchmark: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row.get("benchmark") or "unknown")
        bucket = by_benchmark.setdefault(
            key,
            {"model_call_count": 0, "input_tokens": 0, "output_tokens": 0, "prompt_cache_hit_tokens": 0},
        )
        bucket["model_call_count"] += 1
        bucket["input_tokens"] += int(row.get("input_tokens") or 0)
        bucket["output_tokens"] += int(row.get("output_tokens") or 0)
        bucket["prompt_cache_hit_tokens"] += int(row.get("prompt_cache_hit_tokens") or 0)
    value = {
        "schema_version": "obligate-cost-ledger-v1",
        "victim_episode_count": victim_episode_count,
        "victim_episode_budget": 13_956,
        "victim_episode_budget_compliant": (
            None if victim_episode_count is None else victim_episode_count <= 13_956
        ),
        "model_call_count": len(rows),
        "successful_model_call_count": sum(row.get("status") == "success" for row in rows),
        "failed_model_call_count": sum(row.get("status") != "success" for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_cache_hit_tokens": cached_tokens,
        "token_usage_missing_call_count": sum(not row.get("token_usage_available") for row in rows),
        "estimated_price_usd": price,
        "pricing": dict(pricing) if pricing is not None else None,
        "by_benchmark": by_benchmark,
        "provider_model_identifiers": sorted(
            {str(row.get("provider_model_identifier")) for row in rows if row.get("provider_model_identifier")}
        ),
        "provider_fingerprints": sorted(
            {str(row.get("provider_fingerprint")) for row in rows if row.get("provider_fingerprint")}
        ),
        "sampling_limitation": (
            "The provider does not promise bitwise deterministic responses; end-to-end differences after "
            "the first policy divergence may include sampling and service-side nondeterminism."
        ),
        "source_ledgers": [str(path.resolve()) for path in paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "token_usage_available": False,
        }
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    if not cached:
        # DeepSeek's OpenAI-compatible response may expose cache hits through
        # provider-specific usage fields.
        cached = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": int(getattr(usage, "total_tokens", prompt + completion) or (prompt + completion)),
        "prompt_cache_hit_tokens": cached,
        "token_usage_available": True,
    }


def _public_request(value: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"api_key", "authorization", "headers"}
    return {key: _object_to_json(item) for key, item in value.items() if key.casefold() not in forbidden}


def _object_to_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _object_to_json(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _object_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_object_to_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr_sha256": hashlib.sha256(repr(value).encode("utf-8")).hexdigest()}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["ProviderUsageLedger", "summarize_ledgers"]
