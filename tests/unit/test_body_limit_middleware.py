"""SEC3-01: one anonymous POST must not make the backend buffer 200 MiB.

``InputValidationMiddleware`` read a request body with ``await request.body()`` and compared its
length with the limit AFTER the whole body was in memory: one anonymous ``POST`` announcing 200 MiB
took the backend from 164 MiB to a SIGKILL under the unit's ``MemoryMax=512M`` (measured, see
``tests/integration/test_body_limit_e2e.py`` for the real-uvicorn version of that run).

These tests drive the REAL middleware stack that ``register_input_validation_middleware`` builds
through raw ASGI with a scripted client that delivers body chunks lazily and counts how many bytes
the application side pulled out of it: that count, not a status code, is what the finding is about.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import FastAPI, Request

from backend.src.middleware import body_limit
from backend.src.middleware.body_limit import BodyLimitMiddleware
from backend.src.middleware.input_validation import register_input_validation_middleware
from backend.src.services.auth_service import AuthService

CHUNK = 16 * 1024
ESTOP = "/api/v2/control/emergency"
ANON_LIMIT = body_limit.DEFAULT_ANONYMOUS_MAX_BODY_BYTES
FULL_LIMIT = body_limit.DEFAULT_MAX_BODY_BYTES
MIB = 1024 * 1024


class Wire:
    """A scripted HTTP client: ``total`` body bytes in ``CHUNK`` pieces, produced on demand."""

    def __init__(self, total: int, *, hang_after: int | None = None) -> None:
        self.total = total
        self.sent = 0
        self.calls = 0
        self.hang_after = hang_after  # stop delivering (but stay connected) after this many bytes
        self._piece = b"0" * CHUNK

    @property
    def pulled(self) -> int:
        return self.sent

    async def receive(self) -> dict:
        self.calls += 1
        if self.hang_after is not None and self.sent >= self.hang_after:
            await asyncio.Event().wait()  # the client went quiet; never returns
        remaining = self.total - self.sent
        if remaining <= 0:
            await asyncio.sleep(0)  # a real server waits for the disconnect here
            return {"type": "http.disconnect"}
        size = min(CHUNK, remaining)
        if self.hang_after is not None:
            size = min(size, self.hang_after - self.sent)
        piece = self._piece[:size]
        self.sent += size
        return {"type": "http.request", "body": piece, "more_body": self.sent < self.total}


class Gone(Wire):
    """A client that sends ``first`` bytes and then disconnects."""

    def __init__(self, first: int) -> None:
        super().__init__(first + CHUNK)
        self.first = first

    async def receive(self) -> dict:
        if self.sent >= self.first:
            return {"type": "http.disconnect"}
        return await super().receive()


def scope(method: str, path: str, headers: dict[str, str] | None = None, **extra) -> dict:
    raw = [(b"host", b"t")]
    for name, value in (headers or {}).items():
        raw.append((name.lower().encode(), value.encode()))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": raw,
        "client": ("192.0.2.7", 40000),
        "server": ("127.0.0.1", 8081),
        "scheme": "http",
        **extra,
    }


async def call(app, sc: dict, wire: Wire, timeout: float = 20.0) -> tuple[int | None, bytes, dict]:
    """Run one request to completion: ``(status, body, response headers)``."""
    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    await asyncio.wait_for(app(sc, wire.receive, send), timeout)
    start = next((m for m in messages if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    headers = {k.decode().lower(): v.decode() for k, v in (start or {}).get("headers", [])}
    return (start["status"] if start else None), body, headers


def json_post(path: str, length: int, *, token: str | None = None, **extra) -> dict:
    headers = {"content-type": "application/json", "content-length": str(length), **extra}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    return scope("POST", path, headers)


def chunked_post(path: str, *, token: str | None = None) -> dict:
    headers = {"content-type": "application/json", "transfer-encoding": "chunked"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    return scope("POST", path, headers)


@pytest.fixture
def seen():
    return {"estop_body": None, "echo_body": None}


@pytest.fixture
def auth(monkeypatch):
    """Operator authentication ON, a real AuthService the middleware asks about bearers."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    credential = secrets.token_urlsafe(24)
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", credential)
    monkeypatch.setenv("LAWN_BERRY_AUTH_SECRET", secrets.token_hex(32))
    service = AuthService()
    monkeypatch.setattr(body_limit, "primary_auth_service", service)
    return service, credential


@pytest.fixture
def app(seen):
    application = FastAPI()
    register_input_validation_middleware(application)

    @application.post("/api/v2/echo")
    async def echo(payload: dict):
        seen["echo_body"] = payload
        return {"keys": len(payload)}

    @application.post(ESTOP)
    async def estop(request: Request):
        seen["estop_body"] = await request.body()
        return {"stopped": True}

    @application.post("/api/v2/control/emergency-stop")
    async def estop_no_body():
        return {"stopped": True}

    return application


def payload_of(size: int) -> bytes:
    """A JSON object of exactly ``size`` bytes."""
    overhead = len(b'{"x":""}')
    assert size >= overhead
    return b'{"x":"' + b"a" * (size - overhead) + b'"}'


class Exact(Wire):
    """A client whose body is a given byte string, delivered in ``chunk`` byte pieces."""

    def __init__(self, data: bytes, chunk: int = CHUNK) -> None:
        super().__init__(len(data))
        self.data = data
        self.chunk = chunk

    async def receive(self) -> dict:
        self.calls += 1
        remaining = self.total - self.sent
        if remaining <= 0:
            await asyncio.sleep(0)
            return {"type": "http.disconnect"}
        size = min(self.chunk, remaining)
        piece = self.data[self.sent : self.sent + size]
        self.sent += size
        return {"type": "http.request", "body": piece, "more_body": self.sent < self.total}


# ---- the finding: how much is read, and when -----------------------------------------------


async def test_a_declared_length_over_the_limit_is_refused_without_reading_a_byte(app, auth):
    wire = Wire(64 * MIB)
    status, body, headers = await call(app, json_post("/api/v2/echo", 64 * MIB), wire)
    assert status == 413
    assert json.loads(body) == {"detail": "Payload too large"}
    assert headers["connection"] == "close"  # the client may still be sending: not reused
    assert wire.calls == 0  # not one byte of the body was asked for


async def test_a_chunked_body_is_cut_off_at_the_limit_not_after_it_was_all_read(app, auth):
    wire = Wire(64 * MIB)
    status, _, headers = await call(app, chunked_post("/api/v2/echo"), wire)
    assert status == 413 and headers["connection"] == "close"
    assert (
        wire.pulled <= ANON_LIMIT + CHUNK
    )  # stopped at the limit (plus the chunk that crossed it)


async def test_a_client_that_sends_more_than_it_announced_is_cut_off_too(app, auth):
    wire = Wire(64 * MIB)
    status, *_ = await call(app, json_post("/api/v2/echo", 1024), wire)  # announces 1 KiB
    assert status == 413
    assert wire.pulled <= ANON_LIMIT + CHUNK


async def test_the_limit_is_inclusive_and_the_body_reaches_the_application_intact(app, auth, seen):
    data = payload_of(ANON_LIMIT)
    status, body, _ = await call(app, json_post("/api/v2/echo", len(data)), Exact(data))
    assert status == 200 and json.loads(body) == {"keys": 1}
    assert len(seen["echo_body"]["x"]) == ANON_LIMIT - len(b'{"x":""}')
    status, *_ = await call(app, json_post("/api/v2/echo", ANON_LIMIT + 1), Wire(ANON_LIMIT + 1))
    assert status == 413


async def test_a_chunked_body_within_the_limit_is_passed_on(app, auth, seen):
    data = payload_of(12 * 1024)  # three 4 KiB chunks, below the anonymous limit
    wire = Exact(data, chunk=4096)
    status, _, _ = await call(app, chunked_post("/api/v2/echo"), wire)
    assert status == 200 and len(seen["echo_body"]["x"]) == 12 * 1024 - len(b'{"x":""}')
    assert wire.calls >= 3


async def test_a_body_that_never_finishes_is_answered_408_not_waited_for_for_ever(app, auth):
    application = BodyLimitMiddleware(app, read_timeout_s=0.1)
    wire = Wire(8 * 1024, hang_after=100)  # 100 bytes, then silence
    status, *_ = await call(application, json_post("/api/v2/echo", 8 * 1024), wire, timeout=5)
    assert status == 408


async def test_a_client_that_disconnects_mid_body_gets_no_response_and_runs_nothing(
    app, auth, seen
):
    status, body, _ = await call(app, json_post("/api/v2/echo", 8 * 1024), Gone(1024))
    assert status is None and body == b"" and seen["echo_body"] is None


@pytest.mark.parametrize("length", [None, "0"])
async def test_a_request_without_a_body_is_not_read_at_all(app, auth, length):
    headers = {} if length is None else {"content-length": length}
    wire = Wire(0)
    status, *_ = await call(app, scope("GET", "/openapi.json", headers), wire)
    assert status == 200 and wire.calls == 0


async def test_the_body_replayed_to_the_application_is_one_message_then_the_servers_disconnect(
    auth,
):
    delivered: list[dict] = []

    async def probe(sc, receive, send):
        delivered.append(await receive())  # the body
        delivered.append(await receive())  # whatever the server says next
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = BodyLimitMiddleware(probe)
    data = payload_of(10 * 1024)  # three pieces of 4 KiB at most
    await call(middleware, json_post("/x", len(data)), Exact(data, chunk=4096))
    assert delivered[0] == {"type": "http.request", "body": data, "more_body": False}
    assert delivered[1]["type"] == "http.disconnect"  # not a second body message


async def test_websocket_and_lifespan_scopes_pass_through_untouched():
    seen_scopes = []

    async def probe(sc, receive, send):
        seen_scopes.append((sc["type"], receive, send))

    middleware = BodyLimitMiddleware(probe)
    sentinel_receive, sentinel_send = object(), object()
    for kind in ("websocket", "lifespan"):
        await middleware(
            {"type": kind, "headers": [(b"content-length", b"999999999")]},
            sentinel_receive,
            sentinel_send,
        )
    assert seen_scopes == [
        ("websocket", sentinel_receive, sentinel_send),
        ("lifespan", sentinel_receive, sentinel_send),
    ]


# ---- who may send how much -------------------------------------------------------------------


async def login(service: AuthService, credential: str) -> str:
    return (await service.authenticate(credential)).token


async def test_a_live_operator_session_may_send_the_full_limit_an_anonymous_peer_may_not(
    app, auth, seen
):
    service, credential = auth
    token = await login(service, credential)
    size = 200 * 1024  # a mission or a map: far above the anonymous limit, far below the full one
    data = payload_of(size)
    anonymous, *_ = await call(app, json_post("/api/v2/echo", size), Exact(data))
    operator, *_ = await call(app, json_post("/api/v2/echo", size, token=token), Exact(data))
    assert (anonymous, operator) == (413, 200)
    over, *_ = await call(app, json_post("/api/v2/echo", FULL_LIMIT + 1, token=token), Wire(1))
    assert over == 413


async def test_a_bearer_that_is_not_a_live_session_gets_the_anonymous_limit(app, auth):
    service, credential = auth
    live = await login(service, credential)
    forged = jwt.encode(
        {
            "sub": "x",
            "sid": next(iter(service.active_sessions)),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        },
        "not-the-signing-key-" + "x" * 32,
        algorithm="HS256",
    )
    unknown_session = service.jwt_manager.create_token(
        session_id="no-such-session",
        user_id="u",
        role=next(iter(service.active_sessions.values())).security_context.role,
    )[0]
    size = 200 * 1024
    data = payload_of(size)
    for token in (forged, unknown_session, "garbage"):
        status, *_ = await call(app, json_post("/api/v2/echo", size, token=token), Exact(data))
        assert status == 413, token
    # the control: the live one passes
    status, *_ = await call(app, json_post("/api/v2/echo", size, token=live), Exact(data))
    assert status == 200


async def test_only_the_bearer_scheme_counts_as_a_credential(app, auth):
    service, credential = auth
    token = await login(service, credential)
    size = 200 * 1024
    data = payload_of(size)
    for authorization, expected in (
        (f"Bearer {token}", 200),
        (f"bearer {token}", 200),
        (f"Basic {token}", 413),
        (f"ApiKey {token}", 413),
        (token, 413),
    ):
        sc = json_post("/api/v2/echo", size)
        sc["headers"].append((b"authorization", authorization.encode()))
        status, *_ = await call(app, sc, Exact(data))
        assert status == expected, authorization


async def test_with_operator_auth_off_nobody_is_asked_for_a_credential(app, monkeypatch, auth):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    size = 200 * 1024
    status, *_ = await call(app, json_post("/api/v2/echo", size), Exact(payload_of(size)))
    assert status == 200  # the documented LAN-bench mode: every route is open anyway


# ---- the E-stop is never refused, never waited for, only bounded --------------------------------


@pytest.mark.parametrize("path", [ESTOP, "/api/v2/control/emergency-stop"])
async def test_an_estop_with_an_enormous_declared_body_is_not_refused(app, auth, path):
    wire = Wire(10 * MIB)
    status, body, _ = await call(app, json_post(path, 10**9), wire)
    assert status == 200 and json.loads(body) == {"stopped": True}


async def test_an_estop_is_not_held_back_by_a_body_that_trails_its_headers(app, auth):
    """The stop must start when the headers are in: this middleware must not buffer its body."""
    wire = Wire(2, hang_after=0)  # the body never arrives at all
    status, body, _ = await call(app, json_post("/api/v2/control/emergency-stop", 2), wire, 5)
    assert status == 200 and json.loads(body) == {"stopped": True}


async def test_an_estop_body_is_capped_before_the_application_sees_it(app, auth, seen):
    wire = Wire(50 * MIB)
    status, *_ = await call(app, json_post(ESTOP, 50 * MIB), wire)
    assert status == 200
    assert len(seen["estop_body"]) == body_limit.ESTOP_BODY_BYTES  # exactly the cap, no more
    assert wire.pulled <= body_limit.ESTOP_BODY_BYTES + CHUNK  # and the rest was never pulled in


async def test_a_small_estop_body_passes_unchanged(app, auth, seen):
    data = b'{"session_id": "manual-x", "command": "stop"}'
    status, *_ = await call(app, json_post(ESTOP, len(data)), Exact(data))
    assert status == 200 and seen["estop_body"] == data


async def test_the_estop_cap_is_not_applied_to_a_clear(app, auth):
    """``/control/emergency_clear`` is NOT an E-stop (exact match): it is bounded like the rest."""
    sc = json_post("/api/v2/control/emergency_clear", 64 * MIB)
    status, *_ = await call(app, sc, Wire(0))
    assert status == 413


# ---- configuration is parsed strictly and never raises -------------------------------------


@pytest.mark.parametrize("bad", ["abc", "-5", "1e9", "10", "", " ", "1_000", "٣٠٠٠"])
def test_a_bad_limit_in_the_environment_falls_back_and_never_raises(bad, monkeypatch, caplog):
    body_limit._logged.clear()
    monkeypatch.setenv("INPUT_MAX_BODY_BYTES", bad)
    monkeypatch.setenv("ANONYMOUS_MAX_BODY_BYTES", bad)
    caplog.set_level(logging.DEBUG, logger=body_limit.__name__)
    application = FastAPI()
    register_input_validation_middleware(application)  # used to raise ValueError on "abc"
    built = [m for m in application.user_middleware if m.cls is BodyLimitMiddleware]
    assert built and built[0].kwargs["max_body_bytes"] == FULL_LIMIT
    assert built[0].kwargs["anonymous_max_body_bytes"] == ANON_LIMIT
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    if bad.strip():
        assert any("INPUT_MAX_BODY_BYTES" in e for e in errors)
        assert any("ANONYMOUS_MAX_BODY_BYTES" in e for e in errors)
    else:
        assert errors == []  # blank means unset


def test_valid_limits_from_the_environment_are_used(monkeypatch):
    monkeypatch.setenv("INPUT_MAX_BODY_BYTES", "2000000")
    monkeypatch.setenv("ANONYMOUS_MAX_BODY_BYTES", "4096")
    monkeypatch.setenv("INPUT_BODY_TIMEOUT_S", "7.5")
    application = FastAPI()
    register_input_validation_middleware(application)
    kwargs = next(m.kwargs for m in application.user_middleware if m.cls is BodyLimitMiddleware)
    assert kwargs == {
        "max_body_bytes": 2_000_000,
        "anonymous_max_body_bytes": 4096,
        "read_timeout_s": 7.5,
    }


def test_the_anonymous_limit_can_never_exceed_the_full_one():
    middleware = BodyLimitMiddleware(None, max_body_bytes=5000, anonymous_max_body_bytes=10**9)
    assert middleware.anonymous_max_body_bytes == 5000


# ---- what the validation behind it still does ----------------------------------------------


async def test_the_validation_behind_the_bound_is_unchanged(app, auth):
    for headers, data, expected in (
        ({"content-type": "text/plain"}, b"x=1", 415),
        ({"content-type": "application/json"}, b"{not json}", 400),
        ({"content-type": "application/json"}, b'{"a": 1}', 200),
    ):
        sc = scope("POST", "/api/v2/echo", {**headers, "content-length": str(len(data))})
        status, *_ = await call(app, sc, Exact(data))
        assert status == expected, data
