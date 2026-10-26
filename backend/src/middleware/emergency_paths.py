"""The E-stop endpoints: the only state-changing routes that need no credential, and the only
ones no request limiter may ever refuse.

An E-stop that answers ``429 Too many requests`` (the global limiter's shared per-client bucket
ran dry because the UI was polling) or ``401`` is worse than none. These three POSTs are
therefore:

* open (no ``require_operator_auth``; ``tests/integration/test_operator_auth_coverage.py`` pins
  that this set and the open-route allow-list stay identical), and
* skipped by the global rate limiter, the request-validation middleware, the API-key middleware
  and the request-body bound (``middleware/body_limit.py`` caps what they are handed, it never
  refuses them).

They are matched by EXACT path (a trailing slash is ignored), never by prefix: the authenticated
``/control/emergency_clear`` and ``/tractor/clear-emergency`` must stay limited and gated, and
a prefix such as ``/api/v2/control/emergency`` also matches ``emergency_clear``. The set is a
constant on purpose: it is not read from the environment, so an operator-supplied exemption
list (``GLOBAL_RATE_LIMIT_EXEMPT``) cannot drop it.

Three questions, three functions, so nobody has to guess which one a caller needs:

* :func:`is_emergency_stop_path` - the PATH alone. What the request limiters and the validation
  middleware exempt: a request to one of these paths is never refused by them, whatever its
  method.
* :func:`is_emergency_stop_scope` - the same, for an ASGI scope (``root_path`` ignored: uvicorn
  puts the root path in front of ``scope["path"]``).
* :func:`is_emergency_stop_request` - an actual stop: the PATH **and** the method ``POST``. What
  the connection guard (``core/connection_guard.py``) gives priority over its quota. A ``GET``,
  ``HEAD`` or ``PUT`` to those paths answers 405 and stops nothing, so it earns no priority: the
  guard used to key on the path alone, and a flood of such requests spent the reserve that keeps
  the real stop reachable (SEC3-02).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def is_emergency_stop_scope(scope: Mapping[str, Any]) -> bool:
    """``True`` when the ASGI scope addresses an E-stop path, whatever the method.

    Looks at the routed path and, when the server runs under a ``root_path``, at the path with
    that prefix removed (uvicorn prepends the root path to ``scope["path"]``).
    """
    path = scope.get("path") or ""
    if is_emergency_stop_path(path):
        return True
    root = scope.get("root_path") or ""
    return bool(root) and path.startswith(root) and is_emergency_stop_path(path[len(root) :] or "/")


def is_emergency_stop_request(scope: Mapping[str, Any]) -> bool:
    """``True`` for a real stop: ``POST`` to one of the E-stop paths (see the module docstring)."""
    return scope.get("method") == "POST" and is_emergency_stop_scope(scope)
