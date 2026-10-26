"""Request-scoped context helpers.

Provides a lightweight way to share correlation IDs and other security
metadata across services without introducing heavyweight dependencies.
"""

from __future__ import annotations

import re
from contextvars import ContextVar

# Default correlation identifier used when middleware has not populated the value yet.
_DEFAULT_CORRELATION_ID = "anonymous"

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default=_DEFAULT_CORRELATION_ID)


def set_correlation_id(value: str) -> None:
    """Set the correlation ID for the current execution context."""
    _correlation_id.set(value or _DEFAULT_CORRELATION_ID)


def get_correlation_id() -> str:
    """Return the correlation ID for the current execution context."""
    return _correlation_id.get()


def reset_correlation_id() -> None:
    """Reset the correlation ID to the default anonymous value."""
    _correlation_id.set(_DEFAULT_CORRELATION_ID)


#: A caller-supplied correlation id longer than this, or with any other character, is replaced.
MAX_CORRELATION_ID_LENGTH = 64
_SAFE_CORRELATION_ID = re.compile(rf"[A-Za-z0-9._:=+/-]{{1,{MAX_CORRELATION_ID_LENGTH}}}")


def sanitize_correlation_id(value: str | None) -> str | None:
    """The caller's ``X-Correlation-ID`` if it is short and plain, else ``None`` (SEC20-14).

    The id is written to every log line and echoed in a response header, and the caller picks
    it: an unbounded or control-character-laden value was log amplification and log forging.
    A uuid, ULID or hex trace id passes; the caller of this function then generates its own
    id for anything that does not.
    """
    if value is None:
        return None
    text = value.strip()
    return text if _SAFE_CORRELATION_ID.fullmatch(text) else None
