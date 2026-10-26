"""uvicorn HTTP protocol that keeps the E-stop reachable when clients hold descriptors (SEC20-04).

Use it with ``uvicorn --http backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol`` (an import
string for ``--http`` needs uvicorn >= 0.36.0; before that the flag only accepts ``auto``,
``h11`` and ``httptools`` and the service would refuse to start). The policy lives in
:mod:`backend.src.core.connection_guard` (read its docstring first); this module wires it to
uvicorn's connection life cycle. The state follows the REQUEST (SEC3-02): a connection used to
become "productive" for good after its first complete request, and a client that then started a
second request it never finished (or that sent ``GET`` to an E-stop path, which the path-only
classification counted as a stop) kept its slot and the E-stop reserve for ever.

* ``connection_made``: register the connection (evictable). At the cap the oldest evictable
  connection is aborted to make room; if every connection is productive the new one is aborted
  instead. A HEADER deadline (``loop.call_later``) is armed: a connection whose first request has
  not got its headers in within ``header_deadline_s`` of being accepted is aborted, however
  slowly it dribbles (a total deadline, not an inactivity timeout, so one byte per second does
  not keep it alive).
* ``data_received`` (after uvicorn parsed the bytes; ``self.cycle`` is the newest
  ``RequestResponseCycle``, a new object whenever a request's headers complete):

  - a new request whose body has not arrived (``cycle.more_body``): the header deadline ends and
    a BODY deadline (``body_deadline_s``) starts; the connection stays evictable;
  - a request that is complete (``more_body`` false) and whose response is not: the connection
    is marked productive (:meth:`ConnectionGuard.mark_productive`; an E-stop, i.e. a ``POST``
    to an E-stop path, always is, any other request only while the quota has room) and the body
    deadline ends;
  - the first bytes of the NEXT request on a keep-alive connection (no new cycle yet): the
    header deadline starts again, so a second request dribbled for ever is closed like a first.
* ``on_response_complete`` (uvicorn calls it when a response is finished): the connection is
  evictable again (:meth:`ConnectionGuard.mark_idle`) and waits for the next request under
  uvicorn's own ``timeout_keep_alive``; a request whose response was sent before its body was
  complete keeps the body deadline.
* ``handle_websocket_upgrade``: the socket leaves the table (the WebSocket protocol takes over the
  transport and ``connection_lost`` of this object will never run; upgraded sockets are capped by
  the application's WebSocket registry).
* ``connection_lost``: forget the connection and its timer.
* The ASGI app uvicorn serves on each connection (``self.app``, read per request) is wrapped so
  that, while the non-E-stop quota is full, a request that is not an E-stop (and whose connection
  is not productive) is answered ``503`` with ``Connection: close`` and ``Retry-After: 1`` before
  any middleware runs. Lifespan and WebSocket scopes never reach the wrapper (uvicorn serves them
  from ``config.loaded_app``) and E-stop requests always pass through.

What it relies on in uvicorn, all of it guarded: a missing piece degrades that connection to
"no guard" (logged once), never to an exception in the listener.

* the class ``uvicorn.protocols.http.auto.AutoHTTPProtocol`` (httptools if importable, else h11,
  exactly what ``--http auto`` picks); the constructor is passed through ``*args, **kwargs``;
* instance attributes ``app``, ``loop``, ``transport`` (set in ``connection_made``), ``cycle``
  (the latest ``RequestResponseCycle``; ``None`` until a request's headers are complete),
  ``cycle.scope`` (that request's ASGI scope), ``cycle.more_body`` (true until the request body
  is complete: both implementations set it in the parser callback / ``h11.EndOfMessage``) and
  ``cycle.response_complete`` (true once the response is finished; absent means "never": the
  connection then stays productive until it closes, the behaviour before SEC3-02);
* the methods ``connection_made``, ``connection_lost``, ``data_received``,
  ``on_response_complete`` and ``handle_websocket_upgrade`` (called by both implementations just
  before the socket is handed to the WebSocket protocol).

They are identical in every uvicorn release from 0.30.0 to 0.46.0 (the installed one). The pin in
``pyproject.toml`` must therefore be ``uvicorn[standard]>=0.36.0``.

``--limit-concurrency`` must stay unset: uvicorn then answers 503 to EVERY new request once
``len(connections)`` reaches it, the E-stop included.

The guard reads ``LAWN_BERRY_MAX_CONNECTIONS``, ``LAWN_BERRY_CONNECTION_RESERVE``,
``LAWN_BERRY_HEADER_DEADLINE_S`` and ``LAWN_BERRY_BODY_DEADLINE_S`` from the process environment
when this module is imported (uvicorn imports it before the application, so put them in the
unit's ``Environment=``/``EnvironmentFile=``, not only in a ``.env`` the application loads later)
and logs the result at start-up.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from uvicorn.protocols.http.auto import AutoHTTPProtocol

from backend.src.core import connection_guard
from backend.src.middleware.emergency_paths import is_emergency_stop_request

logger = logging.getLogger("uvicorn.error")

GUARDED_PROTOCOL = "backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol"

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

_BUSY_BODY = b'{"detail":"Server busy, retry shortly"}'
_BUSY_HEADERS = [
    (b"content-type", b"application/json"),
    (b"content-length", str(len(_BUSY_BODY)).encode("ascii")),
    (b"connection", b"close"),
    (b"retry-after", b"1"),
]
# uvicorn internals the mixin needs: instance attributes once the base ``__init__`` ran, and
# methods of the base class it extends.
_REQUIRED_ATTRIBUTES = ("app", "loop", "transport", "cycle")
_REQUIRED_BASE_METHODS = (
    "connection_made",
    "connection_lost",
    "data_received",
    "on_response_complete",
    "handle_websocket_upgrade",
)

_HEADERS = "headers"
_BODY = "body"

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: object, exc_info: bool = False) -> None:
    """One log line per problem per process: a hostile client must not be able to flood it."""
    if key in _warned:
        return
    _warned.add(key)
    logger.error(message, *args, exc_info=exc_info)


_budget_checked = False


def _check_listener_budget(guard: connection_guard.ConnectionGuard, config: object) -> None:
    """Warn once per process if the listen backlog can overrun the descriptor budget."""
    global _budget_checked
    if _budget_checked:
        return
    _budget_checked = True
    problem = connection_guard.descriptor_budget_problem(
        guard.max_connections,
        getattr(config, "backlog", None),
        connection_guard.soft_nofile_limit(),
    )
    if problem:
        logger.warning("%s", problem)


async def _send_busy(send: Send) -> None:
    await send({"type": "http.response.start", "status": 503, "headers": _BUSY_HEADERS})
    await send({"type": "http.response.body", "body": _BUSY_BODY})


class _GuardedApp:
    """The ASGI app of one connection, shedding non-E-stop requests while the quota is full."""

    __slots__ = ("_app", "_conn")

    def __init__(self, app: Callable[..., Awaitable[None]], conn: GuardedHttpProtocolMixin) -> None:
        self._app = app
        self._conn = conn

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        guard = self._conn._guard
        if (
            guard is not None
            and scope.get("type") == "http"
            and guard.over_soft_cap()
            and not guard.is_productive(self._conn)
            and not is_emergency_stop_request(scope)
        ):
            guard.count_shed()
            await _send_busy(send)
            return
        await self._app(scope, receive, send)


class GuardedHttpProtocolMixin:
    """Connection guard for uvicorn's HTTP protocol classes. Put it FIRST in the bases."""

    _guard: connection_guard.ConnectionGuard | None = None
    _deadline_handle: asyncio.TimerHandle | None = None
    _deadline_kind: str | None = None
    _seen_cycle: Any = None
    _armed_for: Any = None  # the cycle that was current when the running deadline was armed

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            base = super()
            missing = [name for name in _REQUIRED_ATTRIBUTES if not hasattr(self, name)]
            missing += [name for name in _REQUIRED_BASE_METHODS if not hasattr(base, name)]
            if missing:
                _warn_once(
                    "incompatible",
                    "connection guard disabled: this uvicorn lacks %s (needs >= 0.36.0)",
                    ", ".join(missing),
                )
                return
            guard = connection_guard.process_guard()
            _check_listener_budget(guard, getattr(self, "config", None))
            self.app = _GuardedApp(self.app, self)
            self._guard = guard
        except Exception:
            _warn_once("init", "connection guard disabled: setup failed", exc_info=True)
            self._guard = None

    def connection_made(self, transport: asyncio.Transport) -> None:
        super().connection_made(transport)
        guard = self._guard
        if guard is None:
            return
        try:
            if not guard.register(self, self._abort_transport):
                # Every slot is held by a connection that is serving a request: refuse.
                self._guard = None
                self._abort_transport()
                return
            self._arm_deadline(_HEADERS)
        except Exception:
            _warn_once("connection_made", "connection guard failed", exc_info=True)
            self._detach_guard()

    def data_received(self, data: bytes) -> None:
        super().data_received(data)
        if self._guard is not None:
            self._sync(data_arrived=True)

    def on_response_complete(self) -> None:
        super().on_response_complete()
        if self._guard is not None:
            self._sync(data_arrived=False)

    def handle_websocket_upgrade(self, *args: Any, **kwargs: Any) -> None:
        # The WebSocket protocol takes the transport over: ``connection_lost`` of this object
        # never runs, so the slot has to be given back now.
        self._detach_guard()
        return super().handle_websocket_upgrade(*args, **kwargs)

    def connection_lost(self, exc: Exception | None) -> None:
        try:
            self._detach_guard()
        finally:
            super().connection_lost(exc)

    def _sync(self, *, data_arrived: bool) -> None:
        """Bring the guard's view of this connection in line with uvicorn's (see the module doc).

        ``data_arrived`` is true after ``data_received``, false after a finished response.
        """
        guard = self._guard
        if guard is None:
            return
        try:
            cycle = self.cycle
            new_request = cycle is not self._seen_cycle
            if new_request:
                self._seen_cycle = cycle
                self._cancel_deadline()  # the headers of that request are in
            if cycle is None:
                return  # the first request's headers are still incomplete: its deadline runs
            if getattr(cycle, "more_body", False):
                # The body is still arriving: this connection stays evictable and the request
                # races the body deadline (also when the response was sent before the body ended).
                if self._deadline_kind != _BODY:
                    self._arm_deadline(_BODY)
                return
            if self._deadline_kind == _BODY:
                self._cancel_deadline()
            if getattr(cycle, "response_complete", False):
                guard.mark_idle(self)
                if data_arrived and not new_request and self._deadline_handle is None:
                    # First bytes of the next request on a keep-alive connection: its headers
                    # get a deadline too (uvicorn's keep-alive timer only covers the wait).
                    self._arm_deadline(_HEADERS)
                return
            guard.mark_productive(self, estop=is_emergency_stop_request(cycle.scope or {}))
        except Exception:
            _warn_once("sync", "connection guard failed", exc_info=True)
            self._detach_guard()

    def _arm_deadline(self, kind: str) -> None:
        guard = self._guard
        if guard is None:
            return
        self._cancel_deadline()
        delay = guard.header_deadline_s if kind == _HEADERS else guard.body_deadline_s
        self._deadline_kind = kind
        self._armed_for = self.cycle
        self._deadline_handle = self.loop.call_later(delay, self._on_deadline, kind)

    def _on_deadline(self, kind: str) -> None:
        self._deadline_handle = None
        self._deadline_kind = None
        guard = self._guard
        transport = getattr(self, "transport", None)
        if guard is None or transport is None or transport.is_closing():
            return
        cycle = self.cycle
        if cycle is not self._armed_for:
            return  # a later request has begun since this timer was armed: it is stale
        if kind == _BODY and not getattr(cycle, "more_body", False):
            return  # the body arrived in time
        guard.count_deadline_close()
        self._abort_transport()

    def _cancel_deadline(self) -> None:
        handle, self._deadline_handle = self._deadline_handle, None
        self._deadline_kind = None
        if handle is not None:
            handle.cancel()

    def _detach_guard(self) -> None:
        guard, self._guard = self._guard, None
        try:
            self._cancel_deadline()
            if guard is not None:
                guard.release(self)
        except Exception:
            _warn_once("detach", "connection guard could not release a connection", exc_info=True)

    def _abort_transport(self) -> None:
        transport = getattr(self, "transport", None)
        if transport is None:
            return
        try:
            transport.abort()
        except Exception:
            _warn_once("abort", "connection guard could not abort a connection", exc_info=True)


class LawnBerryHttpProtocol(GuardedHttpProtocolMixin, AutoHTTPProtocol):
    """What ``--http auto`` would pick (httptools, else h11) plus the connection guard."""


def _announce() -> None:
    try:
        guard = connection_guard.process_guard()
        logger.info(
            "connection guard: %s (RLIMIT_NOFILE soft=%d)",
            guard.describe(),
            connection_guard.soft_nofile_limit(),
        )
    except Exception:
        _warn_once("announce", "connection guard unavailable", exc_info=True)


_announce()
