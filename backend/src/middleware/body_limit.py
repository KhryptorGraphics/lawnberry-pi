"""Request-body bound: nobody makes the backend buffer more body than the route could use (SEC3-01).

Why this exists
---------------
``InputValidationMiddleware`` (and FastAPI after it) read a request body with ``await
request.body()``, which collects every chunk before anything is compared with a limit. The check
``len(body) > max`` therefore ran AFTER the whole body was in memory, on a path that needs no
credential: one anonymous ``POST`` announcing 200 MiB took the backend from 164 MiB to a SIGKILL
under the unit's ``MemoryMax=512M`` in 0.4 s (measured, the review's repro), the process that
carries the E-stop route, the deadman and the actuator parking. uvicorn has no body limit of its
own and the connection guard only decides who may keep a socket.

What this middleware does
-------------------------
It is a plain ASGI middleware (not ``BaseHTTPMiddleware``: it must be the one that touches the
raw ``receive``), registered by ``register_input_validation_middleware`` right OUTSIDE the
validation middleware, so nothing that reads a body ever sees one this has not bounded:

* A request that declares no body (no ``Content-Length`` above zero, no ``Transfer-Encoding``)
  is passed straight through; it costs one scan of the headers.
* A declared ``Content-Length`` above the limit is answered ``413`` at once, with ``Connection:
  close``, and not one byte of the body is read.
* Every other body (chunked, or a length that fits) is read here, counted as it arrives, and
  refused with ``413`` (``Connection: close``) the moment it passes the limit; so a chunked
  body, or a client that announces 1 KiB and sends more, cannot get past it either (the HTTP
  layer frames the request: what follows a complete body is a new request, which is judged by
  itself). A body that has not arrived within ``read_timeout_s`` is answered ``408``; the
  connection guard's own per-request deadline (``core/uvicorn_protocol.py``) is shorter and
  also covers a stalled body when no response could be sent. The bytes are then replayed to the
  application as ONE ``http.request`` message.
* **The limit depends on who asks.** A request that carries a bearer token of a live login
  session may send up to ``INPUT_MAX_BODY_BYTES`` (default 1 MB: a mission or a map is the
  largest body any route takes). Anyone else may send ``ANONYMOUS_MAX_BODY_BYTES`` (default
  16 KiB: the login takes a few hundred bytes), because FastAPI reads a body BEFORE it runs the
  route's authentication, so an unauthenticated request to any route gets its body buffered.
  With ``OPERATOR_AUTH_REQUIRED`` off (the documented LAN-bench mode, logged CRITICAL at
  start-up) nobody is asked for a credential and the larger limit applies to everyone. The
  bound on what unauthenticated peers can make the process hold is therefore
  ``connections x 16 KiB`` (16 MiB at the shipped 1024-connection table), not 200 MiB.
* **The E-stop POSTs are never refused** (``middleware/emergency_paths.py``): a stop with a
  huge or odd body still has to stop the machine, and the route does not need the body. They
  are not buffered either (the stop must not wait for a body that trails its headers): the body
  passes through as it arrives, but at most ``estop_body_bytes`` (64 KiB, the cap the route
  itself applies) of it is ever handed on: the next ``receive()`` after that answers "body
  complete" and the rest of what the client sends is discarded.

WebSocket upgrades carry no HTTP body here: uvicorn hands the upgrade to its WebSocket protocol
before any body is parsed, and a frame is bounded by ``--ws-max-size`` (the unit sets 65536) plus
``core/ws_policy.py::MAX_MESSAGE_BYTES`` in the handlers. The ``websocket`` scope passes through
untouched.

Configuration (an invalid value is logged ONCE at ERROR and the default is used: the process that
carries the E-stop is never refused a start over a typo):

``INPUT_MAX_BODY_BYTES``       integer >= 1024 (default 1000000)
``ANONYMOUS_MAX_BODY_BYTES``   integer >= 1024, clamped to the value above (default 16384)
``INPUT_BODY_TIMEOUT_S``       number in [1, 600] (default 30)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable

from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..core.auth_posture import operator_auth_required
from ..services.auth_service import primary_auth_service
from .emergency_paths import is_emergency_stop_scope

logger = logging.getLogger(__name__)

MAX_BODY_ENV = "INPUT_MAX_BODY_BYTES"
ANONYMOUS_MAX_BODY_ENV = "ANONYMOUS_MAX_BODY_BYTES"
BODY_TIMEOUT_ENV = "INPUT_BODY_TIMEOUT_S"

DEFAULT_MAX_BODY_BYTES = 1_000_000
DEFAULT_ANONYMOUS_MAX_BODY_BYTES = 16 * 1024
# What an E-stop request may hand to the application: the cap ``api/rest.py`` applies itself.
ESTOP_BODY_BYTES = 64 * 1024
DEFAULT_READ_TIMEOUT_S = 30.0
MIN_LIMIT_BYTES = 1024
READ_TIMEOUT_RANGE_S = (1.0, 600.0)

_logged: set[str] = set()


def _log_once(name: str, message: str, *args: object) -> None:
    if name in _logged:
        return
    _logged.add(name)
    logger.error(message, *args)


def env_int(name: str, default: int, *, minimum: int = MIN_LIMIT_BYTES) -> int:
    """A strictly parsed integer from the environment; ``default`` (logged once) when invalid."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    if not (text.isascii() and text.isdigit()) or int(text) < minimum:
        _log_once(name, "%s=%r is not an integer >= %d: using %d", name, raw, minimum, default)
        return default
    return int(text)


def env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    low, high = READ_TIMEOUT_RANGE_S
    try:
        value = float(raw.strip())
    except ValueError:
        value = float("nan")
    if not low <= value <= high:  # also false for nan
        _log_once(
            name, "%s=%r is not a number in [%g, %g]: using %g", name, raw, low, high, default
        )
        return default
    return value


class _TooLarge(Exception):
    pass


class _Gone(Exception):
    """The client went away before its body was complete: there is nobody to answer."""


def _framing(headers: Iterable[tuple[bytes, bytes]]) -> tuple[int | None, bool]:
    """``(declared length or None, carries a body)`` from the raw request headers.

    ``Transfer-Encoding`` (any value) or a ``Content-Length`` that is not a plain number means
    a body of unknown size; ``Content-Length: 0`` and no header at all mean no body.
    """
    length: int | None = None
    unknown = False
    for raw_name, raw_value in headers:
        name = raw_name.lower()
        if name == b"transfer-encoding":
            unknown = True
        elif name == b"content-length" and length is None and not unknown:
            text = raw_value.strip()
            if text.isdigit() and len(text) <= 18:
                length = int(text)
            else:
                unknown = True
    if unknown:
        return None, True
    return length, bool(length)


def _bearer(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    for raw_name, raw_value in headers:
        if raw_name.lower() == b"authorization":
            scheme, _, token = raw_value.decode("latin-1").strip().partition(" ")
            return (token.strip() or None) if scheme.lower() == "bearer" else None
    return None


def _capped_receive(receive: Receive, cap: int) -> Receive:
    """``receive`` that hands on at most ``cap`` body bytes and then says "body complete"."""
    remaining = cap
    capped = False

    async def capped_receive() -> Message:
        nonlocal remaining, capped
        if capped:
            # Whatever else the client sends is nobody's business: swallow it until the
            # connection ends or the response is done (what a real ``receive`` waits for).
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return message
        message = await receive()
        if message["type"] != "http.request":
            return message
        body = message.get("body", b"")
        more = bool(message.get("more_body", False))
        if len(body) > remaining or (len(body) == remaining and more):
            capped = True
            return {"type": "http.request", "body": body[:remaining], "more_body": False}
        remaining -= len(body)
        return message

    return capped_receive


async def _read_body(receive: Receive, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":  # http.disconnect
            raise _Gone
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > limit:
            raise _TooLarge
        if chunk:
            chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks)


class BodyLimitMiddleware:
    """See the module docstring."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        anonymous_max_body_bytes: int = DEFAULT_ANONYMOUS_MAX_BODY_BYTES,
        estop_body_bytes: int = ESTOP_BODY_BYTES,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
    ) -> None:
        self.app = app
        self.max_body_bytes = max(1, int(max_body_bytes))
        self.anonymous_max_body_bytes = min(
            self.max_body_bytes, max(1, int(anonymous_max_body_bytes))
        )
        self.estop_body_bytes = max(0, int(estop_body_bytes))
        self.read_timeout_s = float(read_timeout_s)

    def limit_for(self, headers: Iterable[tuple[bytes, bytes]]) -> int:
        """The most body bytes this request may send (see the module docstring)."""
        if not operator_auth_required():
            return self.max_body_bytes
        token = _bearer(headers)
        if token and primary_auth_service.peek_session(token) is not None:
            return self.max_body_bytes
        return self.anonymous_max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        declared, has_body = _framing(headers)
        if not has_body:
            await self.app(scope, receive, send)
            return
        if is_emergency_stop_scope(scope):
            # Never refused, never waited for: only bounded.
            await self.app(scope, _capped_receive(receive, self.estop_body_bytes), send)
            return

        limit = self.limit_for(headers)
        if declared is not None and declared > limit:
            await self._refuse(scope, receive, send, 413, "Payload too large")
            return
        try:
            async with asyncio.timeout(self.read_timeout_s):
                body = await _read_body(receive, limit)
        except _TooLarge:
            await self._refuse(scope, receive, send, 413, "Payload too large")
            return
        except _Gone:
            return
        except TimeoutError:
            await self._refuse(scope, receive, send, 408, "Request body not received in time")
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # after the body: wait for the disconnect, as the server does

        await self.app(scope, replay, send)

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
        # ``Connection: close``: the client may still be sending; this connection is not reused.
        response = JSONResponse(
            status_code=status, content={"detail": detail}, headers={"Connection": "close"}
        )
        await response(scope, receive, send)


def register_body_limit_middleware(app: FastAPI) -> None:
    """Add :class:`BodyLimitMiddleware` (limits from the environment) to ``app``."""
    max_body = env_int(MAX_BODY_ENV, DEFAULT_MAX_BODY_BYTES)
    app.add_middleware(
        BodyLimitMiddleware,
        max_body_bytes=max_body,
        anonymous_max_body_bytes=env_int(ANONYMOUS_MAX_BODY_ENV, DEFAULT_ANONYMOUS_MAX_BODY_BYTES),
        read_timeout_s=env_seconds(BODY_TIMEOUT_ENV, DEFAULT_READ_TIMEOUT_S),
    )
