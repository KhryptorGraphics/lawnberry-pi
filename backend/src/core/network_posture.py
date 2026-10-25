"""Start-up statement of the network-facing posture, next to ``auth_posture.log_auth_posture``.

The backend's exposure is decided by a handful of switches that live in different modules and in
``<root>/.env`` (which overrides the unit). Logging the RESOLVED values once at start-up, where the
operator reads the journal, is what makes a typo visible instead of silently weakening or
widening something:

* which forwarders are believed to say who the client is (``core/peer_address.py``,
  ``LAWN_BERRY_TRUSTED_PROXIES``), and whether uvicorn's own proxy-header handling, which would
  decide the client address first, is still on;
* whether any WebSocket may be opened without a credential (``core/ws_policy.py``,
  ``LAWN_BERRY_ANONYMOUS_TELEMETRY``).

Imports the standard library and the two modules above only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .peer_address import log_proxy_posture
from .ws_policy import log_ws_posture


def log_network_posture(
    argv: Iterable[str] | None = None, environ: Mapping[str, str] | None = None
) -> None:
    """Log the forwarder trust and WebSocket admission posture (called from the lifespan)."""
    log_proxy_posture(argv, environ)
    log_ws_posture(environ)
