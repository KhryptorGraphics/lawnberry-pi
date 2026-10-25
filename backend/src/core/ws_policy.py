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

Sockets are bound to their login session
-----------------------------------------
A socket admitted with a login (a ticket or a bearer) lives exactly as long as that login: the
websocket is kept on its :class:`WsSlot` (``api/routers/auth.py::admit_websocket`` hands it over
at admission, so no handler can forget to), and two things end it from outside, with close code
1008 and never later than :data:`CLOSE_TIMEOUT_S` after the decision:

* :meth:`WsRegistry.close_session`, called when the session ends (logout, an expiry found on
  contact, ``cleanup_expired_sessions``, a credential change: ``AuthService.terminate_session``
  tells its listeners and ``api/routers/auth.py`` registers this one). A token that is dead is
  not enough to cut a socket the token already opened; this is what does;
* the expiry timer, :meth:`WsRegistry.watch_expiry`: at the session's expiry it asks the session
  again (``POST /auth/refresh`` extends it: the socket then stays and the timer re-arms) and
  closes the socket when the session is gone or over. It also asks at least every
  :data:`EXPIRY_RECHECK_MAX_S`, so a wall-clock step cannot leave a socket open past its expiry.

Anonymous sockets and the simulator opt-in's throw-away session have no stored session to end and
are not bound. The handlers end cleanly when their socket is closed from outside: the next frame
they send, or the receive they are blocked in, fails, and their ``finally`` hands the slot back.

Handshakes are budgeted too
---------------------------
The connection budget above counts OPEN sockets. A handshake that is refused (a junk bearer, no
credential) never holds one, so nothing bounded how many of them one peer could make: 670 a
second from one LAN address, each with its own WARNING line, on the event loop that carries the
E-stop. :class:`HandshakeLimiter` is a token bucket per key (2 a second, burst 20) that
``admit_websocket`` consults BEFORE it looks at any credential. The key is the login session when
the credential presented belongs to a live one, else the peer (exactly the HTTP limiter's rule,
``middleware/rate_limiting.py::limiter_client_key``), so junk from a shared address can only
empty its own bucket and never the operator's. An empty bucket is a 429 with ``Retry-After``.
:class:`RefusalLog` keeps the refusals out of the journal: at most one line per peer a minute,
saying how many it stands for, and a hard ceiling on lines per minute in all.

Standard library only.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

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

# Ending a socket from outside (its login ended): close code 1008, "policy violation", is what
# "your login is over" means to a client. The close handshake is given this long; a peer that does
# not answer it cannot hold up whoever ended the session (the logout response, a request that
# found an expired token) and is cut off.
SESSION_ENDED_CLOSE_CODE = 1008
SESSION_ENDED_REASON = "session ended"
CLOSE_TIMEOUT_S = 1.0
# The expiry timer asks its session again at least this often (and never busy-waits).
EXPIRY_RECHECK_MAX_S = 30.0
_EXPIRY_RECHECK_MIN_S = 0.01

# Handshakes, whatever their outcome: a token bucket per key (a login session, else a peer),
# sized like the HTTP limiter's default (2 a second, burst 20) because the UI opens at most three
# sockets per (re)connect and reconnects with a back-off.
HANDSHAKE_RATE_HZ = 2.0
HANDSHAKE_BURST = 20
MAX_HANDSHAKE_KEYS = 4096
HANDSHAKE_PRUNE_INTERVAL_S = 30.0
# Log lines about refused handshakes: one per peer per interval, at most this many in all.
REFUSAL_LOG_INTERVAL_S = 60.0
MAX_REFUSAL_LOG_KEYS = 256
MAX_REFUSAL_LOG_LINES = 30


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


class Closable(Protocol):
    """What the registry needs of a socket it may have to end: Starlette's ``WebSocket.close``."""

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


@dataclass(eq=False)
class WsSlot:
    """One admitted socket's place in the budget; hand it back with :meth:`WsRegistry.release`.

    ``websocket`` is set for a socket bound to a stored login session (module docstring): what
    :meth:`WsRegistry.close_session` and the expiry timer end from outside. ``closing`` says a
    close has been started (once is enough), ``watcher`` is the socket's expiry timer.
    """

    session_id: str | None
    peer: str
    anonymous: bool
    released: bool = False
    websocket: Closable | None = None
    closing: bool = False
    watcher: asyncio.Task[None] | None = None


class HandshakeLimiter:
    """A token bucket per key for WebSocket handshakes: ``rate_hz`` sustained, ``burst`` peak.

    The same shape as ``middleware/rate_limiting.py``'s table, and bounded the same way: a bucket
    that has refilled to its burst is indistinguishable from a new one and is dropped, and a
    table that reaches ``max_keys`` loses its least recently used buckets first, so it never
    holds more than ``max_keys`` keys whatever the keys are. ``clock`` is injectable.
    """

    def __init__(
        self,
        *,
        rate_hz: float = HANDSHAKE_RATE_HZ,
        burst: int = HANDSHAKE_BURST,
        max_keys: int = MAX_HANDSHAKE_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate = max(0.001, float(rate_hz))
        self._burst = float(max(1, int(burst)))
        self._max_keys = max(1, int(max_keys))
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, list[float]] = {}  # key -> [tokens, last use]
        self._last_prune = clock()

    def __len__(self) -> int:
        return len(self._buckets)

    def allow(self, key: str) -> int | None:
        """``None`` if a handshake from ``key`` may go on (and spends the token), else the whole
        seconds (at least 1) after which one may."""
        now = self._clock()
        with self._lock:
            if (
                now - self._last_prune >= HANDSHAKE_PRUNE_INTERVAL_S
                or len(self._buckets) >= self._max_keys
            ):
                self._prune(now)
            bucket = self._buckets.pop(key, None)  # re-inserted below: the dict is in recency order
            if bucket is None:
                bucket = [self._burst, now]
            else:
                bucket[0] = min(self._burst, bucket[0] + max(0.0, now - bucket[1]) * self._rate)
                bucket[1] = now
            self._buckets[key] = bucket
            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                return None
            return max(1, math.ceil((1.0 - bucket[0]) / self._rate))

    def _prune(self, now: float) -> None:
        self._last_prune = now
        for key, (tokens, stamp) in list(self._buckets.items()):
            if tokens + max(0.0, now - stamp) * self._rate >= self._burst:
                del self._buckets[key]
        if len(self._buckets) >= self._max_keys:
            keep = self._max_keys - max(1, self._max_keys // 10)
            for key in list(self._buckets)[: len(self._buckets) - keep]:  # least recently used
                del self._buckets[key]


@dataclass
class _RefusalState:
    last_line_at: float = -math.inf
    pending: int = 0


class RefusalLog:
    """Keeps a flood of refused handshakes out of the journal.

    :meth:`note` counts one refusal for a key (a peer) and says whether a log line is due: the
    key's first refusal, then at most one per ``interval_s``, and never more than ``max_lines``
    lines per interval for all keys together (a client that rotates its addresses gets no more
    lines than one that does not). The number it returns is how many refusals that line stands
    for, so nothing is hidden, only summed. The table is bounded: past ``max_keys`` the oldest
    key is forgotten first. ``clock`` is injectable.
    """

    def __init__(
        self,
        *,
        interval_s: float = REFUSAL_LOG_INTERVAL_S,
        max_keys: int = MAX_REFUSAL_LOG_KEYS,
        max_lines: int = MAX_REFUSAL_LOG_LINES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = float(interval_s)
        self._max_keys = max(1, int(max_keys))
        self._max_lines = max(1, int(max_lines))
        self._clock = clock
        self._lock = threading.Lock()
        self._keys: dict[str, _RefusalState] = {}
        self._window_start = clock()
        self._window_lines = 0

    def __len__(self) -> int:
        return len(self._keys)

    def note(self, key: str) -> int | None:
        """Count one refusal; ``None`` = stay quiet, else write ONE line that stands for N."""
        now = self._clock()
        with self._lock:
            if now - self._window_start >= self._interval:
                self._window_start, self._window_lines = now, 0
            state = self._keys.get(key)
            if state is None:
                if len(self._keys) >= self._max_keys:
                    self._keys.pop(next(iter(self._keys)))
                state = self._keys[key] = _RefusalState()
            state.pending += 1
            if now - state.last_line_at < self._interval or self._window_lines >= self._max_lines:
                return None
            state.last_line_at = now
            self._window_lines += 1
            stands_for, state.pending = state.pending, 0
            return stands_for


class WsRegistry:
    """Counts the live sockets, refuses the ones over budget, and ends a login's sockets.

    Authenticated sockets are counted per login session (a tab storm of one token cannot take the
    whole budget); anonymous sockets (opt-in mode, simulator) per peer and in total against a
    small separate cap. Both count toward ``max_total``. State is a handful of integers, bounded by
    ``max_total`` live slots; release is idempotent (a handler's ``finally`` may run twice).

    A socket handed over together with its session (``websocket=``) is bound to it: see
    :meth:`close_session` and :meth:`watch_expiry`. The registry also owns the handshake budget
    (:attr:`handshakes`) and the refusal log (:attr:`refusals`), so a test that swaps in a fresh
    registry starts with fresh buckets.
    """

    def __init__(
        self,
        *,
        max_total: int = MAX_WS_CLIENTS,
        max_per_session: int = MAX_WS_PER_SESSION,
        max_anonymous: int = MAX_ANONYMOUS_WS,
        max_anonymous_per_peer: int = MAX_ANONYMOUS_WS_PER_PEER,
        handshakes: HandshakeLimiter | None = None,
        refusals: RefusalLog | None = None,
    ) -> None:
        self.max_total = max_total
        self.max_per_session = max_per_session
        self.max_anonymous = max_anonymous
        self.max_anonymous_per_peer = max_anonymous_per_peer
        self.handshakes = handshakes if handshakes is not None else HandshakeLimiter()
        self.refusals = refusals if refusals is not None else RefusalLog()
        self._lock = threading.Lock()
        self._total = 0
        self._anonymous = 0
        self._by_session: Counter[str] = Counter()
        self._anonymous_by_peer: Counter[str] = Counter()
        self._bound: dict[str, set[WsSlot]] = {}
        self.refused: Counter[str] = Counter()

    def check_handshake(self, key: str) -> int | None:
        """``None`` if a handshake from ``key`` may go on, else the seconds it must wait."""
        retry_after = self.handshakes.allow(key)
        if retry_after is not None:
            with self._lock:
                self.refused["handshake"] += 1
        return retry_after

    def acquire(
        self,
        *,
        session_id: str | None,
        peer: str,
        anonymous: bool,
        websocket: Closable | None = None,
    ) -> WsSlot:
        """Take a slot, or raise :class:`WsLimitError` naming the full budget.

        ``websocket`` binds the socket to its (stored) login session; leave it out for a socket
        that has no session to end (anonymous, the simulator opt-in).
        """
        with self._lock:
            scope = self._full_scope(session_id, peer, anonymous)
            if scope is not None:
                self.refused[scope] += 1
                raise WsLimitError(scope)
            self._total += 1
            slot = WsSlot(session_id=session_id, peer=peer, anonymous=anonymous)
            if anonymous:
                self._anonymous += 1
                self._anonymous_by_peer[peer] += 1
            elif session_id is not None:
                self._by_session[session_id] += 1
                if websocket is not None:
                    slot.websocket = websocket
                    self._bound.setdefault(session_id, set()).add(slot)
            return slot

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
        """Give a slot back (idempotent) and stop its expiry timer."""
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
                bound = self._bound.get(slot.session_id)
                if bound is not None:
                    bound.discard(slot)
                    if not bound:
                        del self._bound[slot.session_id]
            watcher, slot.watcher = slot.watcher, None
        if watcher is not None:
            try:
                watcher.cancel()
            except RuntimeError:  # its loop is already closed: nothing is left to cancel
                pass

    @staticmethod
    def _drop(counter: Counter[str], key: str) -> None:
        counter[key] -= 1
        if counter[key] <= 0:
            del counter[key]

    def bound_count(self, session_id: str) -> int:
        """How many open sockets are bound to ``session_id`` (what a logout would end)."""
        with self._lock:
            return len(self._bound.get(session_id, ()))

    async def close_session(
        self,
        session_id: str,
        *,
        code: int = SESSION_ENDED_CLOSE_CODE,
        reason: str = SESSION_ENDED_REASON,
        timeout: float = CLOSE_TIMEOUT_S,
    ) -> int:
        """End every socket bound to ``session_id`` (all at once); how many were told to close.

        Each close is bounded by ``timeout`` and never raises, so the caller (the code that ends
        the session) cannot be held up or failed by a peer that does not answer.
        """
        with self._lock:
            slots = list(self._bound.get(session_id, ()))
        if not slots:
            return 0
        results = await asyncio.gather(
            *(self.close_slot(slot, code=code, reason=reason, timeout=timeout) for slot in slots)
        )
        return sum(results)

    async def close_slot(
        self,
        slot: WsSlot,
        *,
        code: int = SESSION_ENDED_CLOSE_CODE,
        reason: str = SESSION_ENDED_REASON,
        timeout: float = CLOSE_TIMEOUT_S,
    ) -> bool:
        """Close ``slot``'s socket from outside; ``True`` if this call started the close.

        Bounded by ``timeout``: a close the peer does not answer is abandoned (and logged), which
        also tears the connection down, so a hostile peer cannot keep the socket open by staying
        silent. A socket that is already closed, or never was bound, is not an error.
        """
        websocket = slot.websocket
        if websocket is None or slot.closing or slot.released:
            return False
        slot.closing = True
        try:
            async with asyncio.timeout(timeout):
                await websocket.close(code=code, reason=reason)
        except TimeoutError:
            logger.warning(
                "WebSocket close not completed within %.1f s (the peer does not answer the "
                "closing handshake): connection dropped",
                timeout,
            )
        except Exception as exc:  # already closed, the peer is gone: nothing left to end
            logger.debug("WebSocket close of a bound socket raised %s", type(exc).__name__)
        return True

    def watch_expiry(
        self, slot: WsSlot, seconds_left: Callable[[], Awaitable[float | None]]
    ) -> None:
        """Arm ``slot``'s expiry timer (module docstring); :meth:`release` cancels it.

        ``seconds_left()`` asks the session: seconds until it ends, or ``None`` (or <= 0) when it
        is gone or over. The timer sleeps that long (at most :data:`EXPIRY_RECHECK_MAX_S`), asks
        again, and closes the socket the first time the answer is "over". A check that raises
        closes the socket too: when in doubt the login is over.
        """
        if slot.websocket is None or slot.released or slot.watcher is not None:
            return
        slot.watcher = asyncio.get_running_loop().create_task(
            self._watch_expiry(slot, seconds_left), name="ws-session-expiry"
        )

    async def _watch_expiry(
        self, slot: WsSlot, seconds_left: Callable[[], Awaitable[float | None]]
    ) -> None:
        while not slot.released:
            try:
                left = await seconds_left()
            except Exception:
                logger.exception("WebSocket session check failed: closing the socket")
                left = None
            if left is None or left <= 0:
                await self.close_slot(slot)
                return
            await asyncio.sleep(min(max(left, _EXPIRY_RECHECK_MIN_S), EXPIRY_RECHECK_MAX_S))

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
