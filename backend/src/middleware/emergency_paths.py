"""The E-stop endpoints: the only state-changing routes that need no credential, and the only
ones no request limiter may ever refuse.

An E-stop that answers ``429 Too many requests`` (the global limiter's shared per-client bucket
ran dry because the UI was polling) or ``401`` is worse than none. These three POSTs are
therefore:

* open (no ``require_operator_auth``; ``tests/integration/test_operator_auth_coverage.py`` pins
  that this set and the open-route allow-list stay identical), and
* skipped by the global rate limiter and the request-validation middleware.

They are matched by EXACT path (a trailing slash is ignored), never by prefix: the authenticated
``/control/emergency_clear`` and ``/tractor/clear-emergency`` must stay limited and gated, and
a prefix such as ``/api/v2/control/emergency`` also matches ``emergency_clear``. The set is a
constant on purpose: it is not read from the environment, so an operator-supplied exemption
list (``GLOBAL_RATE_LIMIT_EXEMPT``) cannot drop it.
"""

from __future__ import annotations

EMERGENCY_STOP_PATHS: frozenset[str] = frozenset(
    {
        "/api/v2/control/emergency",
        "/api/v2/control/emergency-stop",
        "/api/v2/tractor/emergency-stop",
    }
)


def is_emergency_stop_path(path: str) -> bool:
    """``True`` for exactly the E-stop endpoints (the routed path, not a prefix match)."""
    return (path.rstrip("/") or "/") in EMERGENCY_STOP_PATHS
