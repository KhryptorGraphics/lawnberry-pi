"""Bounded, E-stop-first admission of HTTP connections (descriptor-exhaustion defence).

Why this exists (SEC20-04)
--------------------------
uvicorn registers a connection the moment the socket is accepted and starts no timer for it:
``timeout_keep_alive`` only runs AFTER a response. A client that connects and sends nothing (or
sends a request one byte at a time, or announces a body and never delivers it) therefore holds
a file descriptor for as long as it likes. Once the process runs out of descriptors the kernel
still completes the TCP handshake, uvicorn cannot ``accept()`` and the next client is reset, and
the next client may be the operator pressing the on-screen E-stop. ``--limit-concurrency`` is NOT
the cure: it answers 503 to every new request once ``len(connections)`` reaches it, the E-stop
included. This module is the cure: it bounds the connection table by descriptor budget and keeps
room in it for the E-stop.

The policy (all of it O(1) per event, nothing here imports uvicorn)
-------------------------------------------------------------------
A live HTTP connection is either PRODUCTIVE or evictable, and the state follows the REQUEST, not
the connection (SEC3-02: it used to be a property of the connection for ever).

* A connection is productive from the moment one COMPLETE request (headers and body) has been
  received until the response to it is complete. A connection that has sent nothing, half a
  header, a header plus a body it never finishes, or that waits idle between two requests on
  a keep-alive connection is evictable: it is the shape of every cheap descriptor-holding
  attack, and nothing is lost by closing it. A request that is still arriving is deliberately
  not protected, E-stop requests included: otherwise ``POST /api/v2/control/emergency`` with a
  ``Content-Length`` and no body, repeated, would pin every slot against the real E-stop. When
  a response is finished the connection falls back to evictable (:meth:`mark_idle`), so a client
  that completes one request and then starts a second it never finishes does not keep the
  protection of the first.
* Every request has a deadline the HTTP layer enforces (``uvicorn_protocol``): its headers
  within ``header_deadline_s`` of its first byte (the first request: of the connection being
  accepted), its body within ``body_deadline_s`` of its headers. A connection that misses one
  is closed whatever it holds.
* ``max_connections`` is the hard cap on live HTTP connections, derived from the descriptor
  limit (half of ``RLIMIT_NOFILE``: the other half is for WebSockets, which leave this table
  when they upgrade and are capped elsewhere, files, the camera, subprocess pipes and the burst
  of connections one ``accept()`` pass takes before the first of them can be judged).
* At the cap a NEW connection evicts the OLDEST evictable one (oldest first: a legitimate
  client sends its request within a round trip, so the junk that has been waiting longest goes
  first and a connection that has just arrived is the last to be touched; a connection that
  became idle after a response counts from then). If every connection is productive the
  newcomer is refused.
* ``reserve`` slots are kept for the E-stop. Requests other than an E-stop are admitted as
  productive only while fewer than ``soft_cap = max_connections - reserve`` connections are
  productive; a request on any other connection is answered ``503`` (see ``uvicorn_protocol``)
  before any middleware runs. So at least ``reserve`` slots are always either free or held by
  evictable junk, which a new connection can take, and the reserve is also the size of the junk
  pool: the number of newcomers that must arrive within one round trip to evict a connection
  that is about to speak. That is why the default reserve is a quarter of the cap, not an eighth.
* An E-stop is a ``POST`` to one of the three E-stop paths (``middleware/emergency_paths.py::
  is_emergency_stop_request``): the method counts, a ``GET`` to those paths stops nothing and is
  an ordinary request. E-stop requests are never refused by the soft cap (they bypass the quota
  check). They do not need to be limited: an E-stop request IS an E-stop, an attacker who sends
  them has stopped the mower, which is the safe direction, and (because productive is revoked at
  the end of the response) they cannot pile up: each one only holds a slot while it is served.

What this does NOT do (the honest residual risk)
------------------------------------------------
* A productive connection is protected, so up to ``soft_cap`` clients that hold a REQUEST open
  on purpose (an unauthenticated MJPEG stream that is never read, a response nobody drains) keep
  those slots for as long as the response lasts. The reserve keeps the E-stop reachable; the UI
  answers 503 until they let go.
* A connection flood that arrives faster than one event-loop iteration can judge it, a SYN
  flood, and a CPU-saturating request flood are outside what a per-connection policy can
  decide: a per-source limit in the packet filter (nftables ``ct count``) is the layer for
  them. Only the hardwired E-stop is independent of host load.
* The guard sees connections after ``connection_made``. A TLS handshake that never finishes
  holds a descriptor before that: terminate TLS in front of the backend (the shipped
  deployment does).

Configuration (strict: a bad value is logged ONCE at ERROR, naming the variable, and the
default is used; the server that carries the E-stop is never refused a start over it)
------------------------------------------------------------------------------------
``LAWN_BERRY_MAX_CONNECTIONS``     integer >= 16 and <= RLIMIT_NOFILE - 64
                                   (default ``clamp(RLIMIT_NOFILE // 2, 32, 1024)``)
``LAWN_BERRY_CONNECTION_RESERVE``  integer >= 4 and < the effective maximum
                                   (default ``clamp(max_connections // 4, 8, 256)``)
``LAWN_BERRY_HEADER_DEADLINE_S``   number in [1, 120] (default 10)
``LAWN_BERRY_BODY_DEADLINE_S``     number in [1, 300] (default 15)

Blank means unset. Digits are plain ASCII (no sign, no ``_``, no exponent, no ``0x``).

The guard keeps its state in two dictionaries and is NOT thread-safe: mutate it from the event
loop only. :meth:`ConnectionGuard.snapshot` only reads sizes and counters and may be called from
anywhere (:func:`process_snapshot` for the guard the HTTP protocol uses).

Standard library only.
"""

from __future__ import annotations

import logging
import math
import os
import re
import resource
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping

logger = logging.getLogger(__name__)

MAX_CONNECTIONS_ENV = "LAWN_BERRY_MAX_CONNECTIONS"
CONNECTION_RESERVE_ENV = "LAWN_BERRY_CONNECTION_RESERVE"
HEADER_DEADLINE_ENV = "LAWN_BERRY_HEADER_DEADLINE_S"
BODY_DEADLINE_ENV = "LAWN_BERRY_BODY_DEADLINE_S"

MIN_MAX_CONNECTIONS = 16
MIN_RESERVE = 4
DEFAULT_HEADER_DEADLINE_S = 10.0
HEADER_DEADLINE_RANGE_S = (1.0, 120.0)
DEFAULT_BODY_DEADLINE_S = 15.0
BODY_DEADLINE_RANGE_S = (1.0, 300.0)

_DEFAULT_MAX_FLOOR = 32
_DEFAULT_MAX_CEILING = 1024
_DEFAULT_RESERVE_FLOOR = 8
_DEFAULT_RESERVE_CEILING = 256
# Descriptors the process needs for everything that is not an HTTP connection in the table
# (log files, the database, the loop's own pipes, the serial ports, the camera, ...). An
# explicit maximum that leaves less than this below RLIMIT_NOFILE cannot work and is refused.
FD_HEADROOM = 64
# ``RLIMIT_NOFILE`` unlimited (or unreadable) is treated as this many descriptors.
_UNLIMITED_NOFILE = 1 << 20
_FALLBACK_NOFILE = 1024

# ASCII digits only: ``int()`` also accepts "1_000", "+5", " 5 " and non-ASCII digits.
_INT_RE = re.compile(r"[0-9]{1,9}")
_NUMBER_RE = re.compile(r"[0-9]{1,6}(?:\.[0-9]{1,6})?")


def soft_nofile_limit() -> int:
    """The process's soft ``RLIMIT_NOFILE`` (unlimited or unreadable becomes a finite number)."""
    try:
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (OSError, ValueError):
        logger.warning("cannot read RLIMIT_NOFILE; assuming %d descriptors", _FALLBACK_NOFILE)
        return _FALLBACK_NOFILE
    if soft == resource.RLIM_INFINITY or soft < 0:
        return _UNLIMITED_NOFILE
    return int(soft)


def default_max_connections(soft_nofile: int) -> int:
    """Half the descriptor budget, clamped to [32, 1024]."""
    return min(max(soft_nofile // 2, _DEFAULT_MAX_FLOOR), _DEFAULT_MAX_CEILING)


def default_reserve(max_connections: int) -> int:
    """A quarter of the cap, clamped to [8, 256] (and always below the cap)."""
    reserve = min(max(max_connections // 4, _DEFAULT_RESERVE_FLOOR), _DEFAULT_RESERVE_CEILING)
    return min(reserve, max_connections - 1)


def descriptor_budget_problem(
    max_connections: int, backlog: int | None, soft_nofile: int
) -> str | None:
    """Why the table can be overrun before the guard sees a connection, or ``None``.

    One ``accept()`` pass takes every connection the kernel has queued (up to the listen
    backlog) before the first of them reaches ``connection_made`` and can be judged. If the
    table at its cap plus a full backlog plus the process's own descriptors does not fit in
    ``RLIMIT_NOFILE``, a burst can use up every descriptor in that one pass (uvicorn on the plain
    asyncio loop then stops accepting for a full second; on uvloop the burst is accepted and
    reset): exactly what the guard exists to prevent, and nothing it can undo afterwards.
    """
    if backlog is None:
        return None
    needed = max_connections + backlog + FD_HEADROOM
    if needed <= soft_nofile:
        return None
    return (
        f"max_connections={max_connections} + listen backlog={backlog} + {FD_HEADROOM} process "
        f"descriptors = {needed} exceeds RLIMIT_NOFILE={soft_nofile}: a burst of connections can "
        "use up every descriptor in one accept pass, before the connection guard can judge any "
        "of them. Lower uvicorn's --backlog (128 is plenty) or raise LimitNOFILE"
    )


class _Entry:
    """One tracked connection: how to close it and when it arrived."""

    __slots__ = ("close", "since")

    def __init__(self, close: Callable[[], object], since: float) -> None:
        self.close = close
        self.since = since


class ConnectionGuard:
    """The connection table: who may stay, who is evicted, who is refused. Pure logic.

    ``conn`` is any hashable handle for one HTTP connection (the protocol instance in
    production, a fake in tests); ``close`` is a callable that aborts that connection. The clock
    only dates the connections for :meth:`snapshot`.
    """

    def __init__(
        self,
        max_connections: int,
        reserve: int,
        header_deadline_s: float = DEFAULT_HEADER_DEADLINE_S,
        *,
        body_deadline_s: float = DEFAULT_BODY_DEADLINE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(max_connections, int) or max_connections < 2:
            raise ValueError(f"max_connections must be an integer >= 2, got {max_connections!r}")
        if not isinstance(reserve, int) or not 1 <= reserve < max_connections:
            raise ValueError(
                f"reserve must be an integer in [1, max_connections), got {reserve!r} "
                f"(max_connections={max_connections})"
            )
        if not math.isfinite(header_deadline_s) or header_deadline_s <= 0:
            raise ValueError(f"header_deadline_s must be finite and > 0, got {header_deadline_s!r}")
        if not math.isfinite(body_deadline_s) or body_deadline_s <= 0:
            raise ValueError(f"body_deadline_s must be finite and > 0, got {body_deadline_s!r}")
        self.max_connections = max_connections
        self.reserve = reserve
        self.soft_cap = max_connections - reserve
        self.header_deadline_s = float(header_deadline_s)
        self.body_deadline_s = float(body_deadline_s)
        self._clock = clock
        # Oldest first. ``_evictable`` holds every connection with no complete request in flight
        # (silent, mid-request, idle between two requests); ``_productive`` holds the ones that
        # have one.
        self._evictable: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._productive: dict[Hashable, _Entry] = {}
        self.peak_connections = 0
        self.accepted = 0
        self.evicted = 0
        self.refused = 0
        self.over_quota = 0
        self.deadline_closed = 0
        self.shed = 0
        self.close_errors = 0

    def __len__(self) -> int:
        return len(self._evictable) + len(self._productive)

    def register(self, conn: Hashable, close: Callable[[], object]) -> bool:
        """Track a new connection. ``False`` means it was refused (the caller must close it).

        At the cap the oldest unproductive connection is evicted (its ``close`` is called) to
        make room; if there is none, every slot belongs to a productive connection and the
        newcomer is refused. Registering a connection twice is a no-op.
        """
        if conn in self._evictable or conn in self._productive:
            return True
        if len(self) >= self.max_connections:
            if not self._evictable:
                self.refused += 1
                return False
            _, victim = self._evictable.popitem(last=False)
            self.evicted += 1
            self._close(victim)
        self._evictable[conn] = _Entry(close, self._clock())
        self.accepted += 1
        live = len(self)
        if live > self.peak_connections:
            self.peak_connections = live
        return True

    def mark_idle(self, conn: Hashable) -> bool:
        """The response of a productive connection is complete: it is evictable again.

        It becomes the NEWEST evictable connection (its wait starts now: a keep-alive client
        that has just been served is the last to be touched). Returns ``True`` if the
        connection was productive; a connection that is not (or not tracked) is left alone.
        """
        entry = self._productive.pop(conn, None)
        if entry is None:
            return False
        entry.since = self._clock()
        self._evictable[conn] = entry
        return True

    def mark_productive(self, conn: Hashable, *, estop: bool = False) -> bool:
        """``conn`` has delivered a complete request: protect it from eviction while it is served.

        Returns ``True`` when the connection is (now) productive. A request that is not an
        E-stop is admitted only while fewer than ``soft_cap`` connections are productive;
        otherwise this returns ``False`` and the connection stays evictable (the HTTP layer
        answers its request ``503``). E-stop requests are always admitted. A connection the
        guard no longer tracks (evicted, refused, upgraded) cannot be protected: ``False``.
        The protection ends with :meth:`mark_idle` (the response is complete) or :meth:`release`.
        """
        if conn in self._productive:
            return True
        entry = self._evictable.get(conn)
        if entry is None:
            return False
        if not estop and len(self._productive) >= self.soft_cap:
            self.over_quota += 1
            return False
        del self._evictable[conn]
        self._productive[conn] = entry
        return True

    def is_productive(self, conn: Hashable) -> bool:
        return conn in self._productive

    def over_soft_cap(self) -> bool:
        """``True`` while the non-E-stop quota (``soft_cap`` productive connections) is full."""
        return len(self._productive) >= self.soft_cap

    def release(self, conn: Hashable) -> None:
        """Forget a connection that is gone (closed, evicted, upgraded). Idempotent."""
        if self._evictable.pop(conn, None) is None:
            self._productive.pop(conn, None)

    def count_deadline_close(self) -> None:
        """A request or header deadline closed a connection (the HTTP layer owns the timers)."""
        self.deadline_closed += 1

    def count_shed(self) -> None:
        """A request was answered 503 because the non-E-stop quota was full."""
        self.shed += 1

    def snapshot(self) -> dict[str, int | float | bool | None]:
        """Counters and gauges for diagnostics and health. Plain JSON-serialisable values."""
        oldest = next(iter(self._evictable.values()), None)
        age = None if oldest is None else max(0.0, self._clock() - oldest.since)
        return {
            "max_connections": self.max_connections,
            "reserve": self.reserve,
            "soft_cap": self.soft_cap,
            "header_deadline_s": self.header_deadline_s,
            "body_deadline_s": self.body_deadline_s,
            "connections": len(self),
            "productive": len(self._productive),
            "unproductive": len(self._evictable),
            "over_soft_cap": self.over_soft_cap(),
            "oldest_unproductive_age_s": age,
            "peak_connections": self.peak_connections,
            "accepted": self.accepted,
            "evicted": self.evicted,
            "refused": self.refused,
            "over_quota": self.over_quota,
            "deadline_closed": self.deadline_closed,
            "shed": self.shed,
            "close_errors": self.close_errors,
        }

    def describe(self) -> str:
        """One line for the start-up log."""
        return (
            f"max_connections={self.max_connections} reserve={self.reserve} "
            f"soft_cap={self.soft_cap} header_deadline_s={self.header_deadline_s:g} "
            f"body_deadline_s={self.body_deadline_s:g}"
        )

    def _close(self, entry: _Entry) -> None:
        try:
            entry.close()
        except Exception:
            # A connection that cannot be closed must not stop its replacement being admitted.
            self.close_errors += 1
            if self.close_errors == 1:
                logger.exception("closing an evicted connection failed (further ones are counted)")


def _read_int(
    env: Mapping[str, str], name: str, *, minimum: int, maximum: int | None = None
) -> int | None:
    """A strictly parsed integer, or ``None`` for unset, blank or invalid (logged once)."""
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    text = raw.strip()
    value = int(text) if _INT_RE.fullmatch(text) else None
    if value is None or value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        logger.error("%s=%r is not an integer %s; using the default", name, raw, bound)
        return None
    return value


def _read_deadline(
    env: Mapping[str, str],
    name: str = HEADER_DEADLINE_ENV,
    bounds: tuple[float, float] | None = None,
) -> float | None:
    """A strictly parsed number of seconds in ``bounds``, or ``None`` (unset, blank, invalid)."""
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    text = raw.strip()
    low, high = HEADER_DEADLINE_RANGE_S if bounds is None else bounds
    value = float(text) if _NUMBER_RE.fullmatch(text) else None
    if value is None or not low <= value <= high:
        logger.error(
            "%s=%r is not a number of seconds between %g and %g; using the default",
            name,
            raw,
            low,
            high,
        )
        return None
    return value


def guard_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    soft_nofile: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ConnectionGuard:
    """Build the guard from ``LAWN_BERRY_*`` and the descriptor limit. Never raises on bad input."""
    env = os.environ if environ is None else environ
    nofile = soft_nofile_limit() if soft_nofile is None else soft_nofile

    max_connections = _read_int(
        env,
        MAX_CONNECTIONS_ENV,
        minimum=MIN_MAX_CONNECTIONS,
        maximum=max(nofile - FD_HEADROOM, MIN_MAX_CONNECTIONS),
    )
    if max_connections is None:
        max_connections = default_max_connections(nofile)
    if max_connections + FD_HEADROOM > nofile:
        logger.warning(
            "RLIMIT_NOFILE is %d: too low for %d connections plus the process's own "
            "descriptors; raise LimitNOFILE (the guard cannot protect a table the kernel "
            "will not let it fill)",
            nofile,
            max_connections,
        )

    reserve = _read_int(
        env, CONNECTION_RESERVE_ENV, minimum=MIN_RESERVE, maximum=max_connections - 1
    )
    if reserve is None:
        reserve = default_reserve(max_connections)

    deadline = _read_deadline(env)
    if deadline is None:
        deadline = DEFAULT_HEADER_DEADLINE_S
    body_deadline = _read_deadline(env, BODY_DEADLINE_ENV, BODY_DEADLINE_RANGE_S)
    if body_deadline is None:
        body_deadline = DEFAULT_BODY_DEADLINE_S
    return ConnectionGuard(
        max_connections, reserve, deadline, body_deadline_s=body_deadline, clock=clock
    )


_process_guard: ConnectionGuard | None = None


def process_guard() -> ConnectionGuard:
    """The one guard of this process (built from the environment on first use)."""
    global _process_guard
    if _process_guard is None:
        _process_guard = guard_from_environment()
    return _process_guard


def reset_process_guard(guard: ConnectionGuard | None = None) -> None:
    """Replace (or, with ``None``, drop) the process guard: tests and re-configuration."""
    global _process_guard
    _process_guard = guard


def process_snapshot() -> dict[str, int | float | bool | None] | None:
    """:meth:`ConnectionGuard.snapshot` of the process guard, ``None`` if there is none yet."""
    guard = _process_guard
    return None if guard is None else guard.snapshot()
