"""WebSocket policy: who may open a socket without a credential, and how much one socket may cost.

Anonymous admission is OFF
--------------------------
A browser on the mower itself used to count as "loopback" and was admitted to the read-only
telemetry streams without a credential. On this host that is not a trust level: every ingress the
product ships (a Cloudflare or ngrok agent pointed at the UI server, nginx) reaches the backend from
a loopback address, so a remote visitor looked exactly like the local browser. No peer, loopback
included, is admitted anonymously any more. A browser trades its bearer token for a single-use
ticket (``POST /api/v2/auth/ws-ticket``); other clients send ``Authorization: Bearer``.

``LAWN_BERRY_ANONYMOUS_TELEMETRY`` is the explicit, documented opt-in for the read-only streams
only (``/ws/telemetry``, ``/api/v2/ws/telemetry``, ``/api/v2/ws/status``; the command-capable
``/ws/control`` is never anonymous while operator authentication is on):

``off`` (default, also unset, empty, ``0``, ``false``, ``no``)
    A credential is always required. Any other unrecognised value (``1``, ``true``, a typo) also
    means off and is warned about at start-up: the opt-in must name what it opens.
``loopback``
    Anonymous clients whose RESOLVED address (``core/peer_address.py``) is a loopback address, e.g.
    a kiosk browser on the mower. Only as safe as the forwarding setup: a same-host forwarder that
    is not declared in ``LAWN_BERRY_TRUSTED_PROXIES`` / ``UI_TRUSTED_PROXIES`` makes every remote
    visitor look like the local one.
``any``
    Anonymous clients from anywhere that can reach the port: anyone on the LAN (and on a tunnel,
    if one is configured) can read position, battery and state.

Anonymous browsers must also pass the Origin check (``api/routers/auth.py``); anonymous sockets
cannot change the shared telemetry cadence and have their own, much smaller, connection budget.

Limits
------
Every socket costs a descriptor, a slot in the telemetry fan-out and, before this module, an
unbounded amount of memory (one anonymous socket sending 330 000 ``subscribe`` frames with distinct
1 kB topic names grew the backend from 241 MiB to 612 MiB, the unit's cap is 512 MiB) and CPU
(``/ws/status`` ran a full sensor read per socket at 5 Hz: 12 sockets pushed the E-stop POST from
4 ms to 560 ms). The limits below are constants on purpose (the unit's memory and descriptor budget
does not change with a typo in ``.env``); :class:`WsRegistry` enforces the connection counts and
:class:`MessageBudget` the per-socket frame rate, the rest is enforced where the frame is handled
(``api/routers/telemetry.py``, ``services/websocket_hub.py``).

Standard library only.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ANONYMOUS_TELEMETRY_ENV = "LAWN_BERRY_ANONYMOUS_TELEMETRY"
ANON_OFF = "off"
ANON_LOOPBACK = "loopback"
ANON_ANY = "any"
_OFF_TOKENS = frozenset({"", "0", "false", "no", ANON_OFF})

# Connections: all sockets, per login session, and the (separate, small) anonymous budget.
MAX_WS_CLIENTS = 32
MAX_WS_PER_SESSION = 8
MAX_ANONYMOUS_WS = 4
MAX_ANONYMOUS_WS_PER_PEER = 2

# Frames: a subscribe frame is ~60 bytes; the UI sends ~12 of them in a burst after a reconnect.
MAX_MESSAGE_BYTES = 4096
MESSAGE_RATE_HZ = 20.0
MESSAGE_BURST = 40
# Subscriptions: topic names are short dotted words (``telemetry.power``, ``telemetry/updates``).
MAX_TOPIC_LENGTH = 64
MAX_TOPICS_PER_CLIENT = 32
_TOPIC_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]*")
# A client that cannot take a frame within this long is dropped instead of stalling the fan-out
# every other client shares.
SEND_TIMEOUT_S = 2.0


def parse_anonymous_telemetry(value: str | None) -> tuple[str, bool]:
    """``(mode, recognised)`` for a ``LAWN_BERRY_ANONYMOUS_TELEMETRY`` value (``None`` = unset).

    Only ``loopback`` and ``any`` (any case, surrounding blanks ignored) open anything. Unset and
    the explicit off tokens are ``off``; every other value is ``off`` too, but not recognised.
    """
    if value is None:
        return ANON_OFF, True
    token = value.strip().lower()
    if token in (ANON_LOOPBACK, ANON_ANY):
        return token, True
    return ANON_OFF, token in _OFF_TOKENS


def anonymous_telemetry_mode(environ: Mapping[str, str] | None = None) -> str:
    """The anonymous read-only WebSocket mode in force: ``off``, ``loopback`` or ``any``."""
    env = os.environ if environ is None else environ
    return parse_anonymous_telemetry(env.get(ANONYMOUS_TELEMETRY_ENV))[0]


def valid_topic(topic: object) -> bool:
    """A topic name a client may subscribe to: short, printable, no whitespace or control codes."""
    return (
        isinstance(topic, str)
        and 0 < len(topic) <= MAX_TOPIC_LENGTH
        and _TOPIC_PATTERN.fullmatch(topic) is not None
    )


def log_ws_posture(environ: Mapping[str, str] | None = None) -> None:
    """Say at start-up whether any WebSocket is open without a credential."""
    env = os.environ if environ is None else environ
    raw = env.get(ANONYMOUS_TELEMETRY_ENV)
    mode, recognised = parse_anonymous_telemetry(raw)
    if not recognised:
        logger.warning(
            "%s=%r is not 'loopback' or 'any': anonymous WebSocket access stays OFF",
            ANONYMOUS_TELEMETRY_ENV,
            raw,
        )
    if mode == ANON_OFF:
        logger.info("WebSockets need a credential (ticket or bearer) from every peer")
    else:
        logger.warning(
            "%s=%s: the read-only telemetry streams are open WITHOUT a credential to %s (position, "
            "battery and state are readable); /ws/control still needs a login",
            ANONYMOUS_TELEMETRY_ENV,
            mode,
            "loopback clients" if mode == ANON_LOOPBACK else "every client that can reach the port",
        )


class WsLimitError(Exception):
    """No more sockets may be opened right now; ``scope`` names the budget that is full."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(f"WebSocket connection limit reached ({scope})")


@dataclass(eq=False)
class WsSlot:
    """One admitted socket's place in the budget; hand it back with :meth:`WsRegistry.release`."""

    session_id: str | None
    peer: str
    anonymous: bool
    released: bool = False


class WsRegistry:
    """Counts the live sockets and refuses the ones over budget.

    Authenticated sockets are counted per login session (a tab storm of one token cannot take the
    whole budget); anonymous sockets (opt-in mode, simulator) per peer and in total against a
    small separate cap. Both count toward ``max_total``. State is a handful of integers, bounded by
    ``max_total`` live slots; release is idempotent (a handler's ``finally`` may run twice).
    """

    def __init__(
        self,
        *,
        max_total: int = MAX_WS_CLIENTS,
        max_per_session: int = MAX_WS_PER_SESSION,
        max_anonymous: int = MAX_ANONYMOUS_WS,
        max_anonymous_per_peer: int = MAX_ANONYMOUS_WS_PER_PEER,
    ) -> None:
        self.max_total = max_total
        self.max_per_session = max_per_session
        self.max_anonymous = max_anonymous
        self.max_anonymous_per_peer = max_anonymous_per_peer
        self._lock = threading.Lock()
        self._total = 0
        self._anonymous = 0
        self._by_session: Counter[str] = Counter()
        self._anonymous_by_peer: Counter[str] = Counter()
        self.refused: Counter[str] = Counter()

    def acquire(self, *, session_id: str | None, peer: str, anonymous: bool) -> WsSlot:
        """Take a slot, or raise :class:`WsLimitError` naming the full budget."""
        with self._lock:
            scope = self._full_scope(session_id, peer, anonymous)
            if scope is not None:
                self.refused[scope] += 1
                raise WsLimitError(scope)
            self._total += 1
            if anonymous:
                self._anonymous += 1
                self._anonymous_by_peer[peer] += 1
            elif session_id is not None:
                self._by_session[session_id] += 1
            return WsSlot(session_id=session_id, peer=peer, anonymous=anonymous)

    def _full_scope(self, session_id: str | None, peer: str, anonymous: bool) -> str | None:
        if self._total >= self.max_total:
            return "total"
        if anonymous:
            if self._anonymous >= self.max_anonymous:
                return "anonymous"
            if self._anonymous_by_peer[peer] >= self.max_anonymous_per_peer:
                return "anonymous-peer"
        elif session_id is not None and self._by_session[session_id] >= self.max_per_session:
            return "session"
        return None

    def release(self, slot: WsSlot) -> None:
        """Give a slot back (idempotent)."""
        with self._lock:
            if slot.released:
                return
            slot.released = True
            self._total -= 1
            if slot.anonymous:
                self._anonymous -= 1
                self._drop(self._anonymous_by_peer, slot.peer)
            elif slot.session_id is not None:
                self._drop(self._by_session, slot.session_id)

    @staticmethod
    def _drop(counter: Counter[str], key: str) -> None:
        counter[key] -= 1
        if counter[key] <= 0:
            del counter[key]

    def snapshot(self) -> dict[str, object]:
        """Gauges and refusal counters for diagnostics."""
        with self._lock:
            return {
                "open": self._total,
                "anonymous": self._anonymous,
                "sessions": len(self._by_session),
                "max_total": self.max_total,
                "refused": dict(self.refused),
            }


class MessageBudget:
    """A token bucket for the frames one socket may send (``rate_hz`` sustained, ``burst`` peak)."""

    def __init__(
        self,
        rate_hz: float = MESSAGE_RATE_HZ,
        burst: int = MESSAGE_BURST,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate = float(rate_hz)
        self._burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._stamp = clock()

    def allow(self) -> bool:
        """``True`` if one more frame may be handled now (and spends the token)."""
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + max(0.0, now - self._stamp) * self._rate)
        self._stamp = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


# The one registry every WebSocket route admits through (``api/routers/auth.py::admit_websocket``).
ws_registry = WsRegistry()
