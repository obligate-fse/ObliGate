"""Non-negotiable simulator-only environment controls for formal experiments."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, MutableMapping

SIMULATOR_ENV = {
    "ALLOW_REAL_TOOLS": "0",
    "OBLIGATE_SANDBOX": "1",
    "DEMO_MODE": "1",
    "OBLIGATE_DEMO_MODE": "1",
}


def force_simulator_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    env.update(SIMULATOR_ENV)
    return env


@contextlib.contextmanager
def simulator_only_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in SIMULATOR_ENV}
    force_simulator_env(os.environ)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = ["SIMULATOR_ENV", "force_simulator_env", "simulator_only_environment"]

