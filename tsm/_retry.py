"""Retry policy shared by the OpenAI-backed embedder, extractor and summarizer.

Transient failures (rate limits, timeouts, 5xx) are retried with exponential
backoff. Failures that another attempt cannot fix (a rejected API key, a
malformed or over-long request) are raised at once: retrying those only turned
a clear error into several minutes of silence.

The raised message names the error type and HTTP status, never the provider's
message text; the original exception is chained as ``__cause__``.
"""

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# HTTP statuses that mean "this request will never succeed as sent".
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 413, 422})


def _status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def is_permanent(exc: BaseException) -> bool:
    """True when retrying cannot help (authentication, bad request, ...)."""
    return _status(exc) in _PERMANENT_STATUS


def _describe(exc: BaseException) -> str:
    status = _status(exc)
    name = type(exc).__name__
    return f"{name}, HTTP {status}" if status is not None else name


def call_with_retries(
    fn: Callable[[], T],
    what: str,
    max_retries: int,
    logger,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` until it succeeds, a permanent error occurs, or the attempts
    run out. ``what`` names the operation in log lines and error messages."""
    attempts = max(1, int(max_retries))
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            last = exc
            if is_permanent(exc):
                raise RuntimeError(
                    f"{what} failed with an error that retrying cannot fix ({_describe(exc)})"
                ) from exc
            if attempt + 1 == attempts:
                break
            # 429s can be per-minute windows; back off long enough to outlast them.
            wait = min(5.0 * (2 ** attempt), 120.0)
            logger.warning("%s failed (attempt %d/%d, %s); retry in %.0fs",
                           what, attempt + 1, attempts, _describe(exc), wait)
            sleep(wait)
    raise RuntimeError(
        f"{what} failed after {attempts} attempts ({_describe(last)})"
    ) from last
