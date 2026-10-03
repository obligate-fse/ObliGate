"""Thread-local, per-case victim API accounting; never serializes credentials."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

_state = threading.local()
_REQUEST_FIELDS = ("model", "messages", "tools", "tool_choice", "temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop", "frequency_penalty", "presence_penalty", "parallel_tool_calls", "response_format", "reasoning_effort", "stream")
_EXTRA_BODY_FIELDS = ("enable_thinking", "thinking", "reasoning_effort", "thinking_budget")
_MESSAGE_FIELDS = ("role", "content", "refusal", "reasoning_content", "reasoning", "tool_calls", "function_call", "name", "tool_call_id")
_SECRET_FIELDS = {"api_key", "apikey", "authorization", "headers", "extra_headers", "access_token", "password", "secret"}


def _safe_value(value, secrets=()):
    """JSON-only diagnostic projection, with credential redaction (not replay)."""
    if isinstance(value, str):
        for secret in secrets:
            if len(secret) >= 8:
                value = value.replace(secret, "[REDACTED_CREDENTIAL]")
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED_CREDENTIAL]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_value(v, secrets) for k, v in value.items() if str(k).lower() not in _SECRET_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v, secrets) for v in value]
    if callable(getattr(value, "model_dump", None)):
        return _safe_value(value.model_dump(mode="json"), secrets)
    # Do not serialize arbitrary __dict__ or repr: either may contain a client/key.
    return f"[NON_JSON_{type(value).__name__}]"


def _diagnostic_request(kwargs, secrets):
    request = {key: kwargs[key] for key in _REQUEST_FIELDS if key in kwargs}
    if isinstance(kwargs.get("extra_body"), dict):
        request["extra_body"] = {key: kwargs["extra_body"][key] for key in _EXTRA_BODY_FIELDS if key in kwargs["extra_body"]}
    return _safe_value(request, secrets)


def _diagnostic_choices(response, secrets):
    choices = []
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        values = {key: getattr(message, key) for key in _MESSAGE_FIELDS if hasattr(message, key)}
        choices.append({"index": getattr(choice, "index", None), "finish_reason": getattr(choice, "finish_reason", None), "message": values})
    return _safe_value(choices, secrets)


def begin_case(case_id: str) -> None:
    _state.case_id = case_id
    _state.events = []
    directory = os.getenv("OBLIGATE_MODEL_JOURNAL_DIR")
    _state.path = Path(directory) / f"victim-{uuid4().hex}.jsonl" if directory else None
    if _state.path:
        _state.path.parent.mkdir(parents=True, exist_ok=True)


def _write(event: dict) -> None:
    path = getattr(_state, "path", None)
    if path:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def snapshot() -> dict:
    events = list(getattr(_state, "events", []))
    return {
        "calls": len(events),
        "input_tokens": sum(e.get("input_tokens", 0) for e in events),
        "output_tokens": sum(e.get("output_tokens", 0) for e in events),
        "latency_seconds": sum(e["latency_seconds"] for e in events),
        "errors": sum(e["status"] == "ERROR" for e in events),
        "usage_unknown_calls": sum(not e["usage_reported"] for e in events),
        "events": events,
        "journal": str(getattr(_state, "path", None) or "") or None,
    }


def instrument_client(client):
    """Wrap each explicit SDK call, with SDK retries disabled by caller.

    Journals persist request-start before the network call, so a killed process
    leaves an unresolved call instead of silently claiming zero consumption.
    """
    original = client.chat.completions.create
    client_secret = str(getattr(client, "api_key", "") or "")

    def create(*args, **kwargs):
        started = time.perf_counter()
        event = {"call_id": uuid4().hex, "case_id": getattr(_state, "case_id", None),
                 "role": "victim", "model": str(kwargs.get("model", ""))}
        secrets = tuple(str(value) for key, value in os.environ.items() if "API_KEY" in key.upper() and value)
        if client_secret:
            secrets += (client_secret,)
        if kwargs.get("api_key"):
            secrets += (str(kwargs["api_key"]),)
        _write({**event, "event": "request", "request": _diagnostic_request(kwargs, secrets)})
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            event.update(status="ERROR", error_type=type(exc).__name__, usage_reported=False,
                         latency_seconds=time.perf_counter() - started)
            getattr(_state, "events", []).append(event)
            _write({**event, "event": "error"})
            raise
        usage = getattr(response, "usage", None)
        event.update(status="COMPLETE", usage_reported=usage is not None,
                     resolved_model=getattr(response, "model", None),
                     input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                     output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                     latency_seconds=time.perf_counter() - started)
        getattr(_state, "events", []).append(event)
        # Keep large raw messages only in the append-only journal, not the
        # cumulative usage snapshot embedded in every case report.
        _write({**event, "event": "response", "choices": _diagnostic_choices(response, secrets)})
        return response

    client.chat.completions.create = create
    return client
