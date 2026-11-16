"""Retry transient failures of local HTTP services (e.g. Ollama returning 500 while it swaps models
in and out of GPU memory)."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, httpx.TransportError)  # connection reset, timeout, ...


def with_retries[T](fn: Callable[[], T], attempts: int = 4, base_delay_s: float = 3.0) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            time.sleep(base_delay_s * 2 ** (attempt - 1))
    raise AssertionError("unreachable")
