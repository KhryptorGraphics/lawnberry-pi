"""An open WebSocket is bound to its login, and handshakes are budgeted (SEC3-07, SEC3-09, SEC3-16).

SEC3-07: a socket was admitted once and then outlived the logout, the 8 h expiry and a credential
change (``/ws/status`` kept sending position and battery after ``POST /auth/logout``). Now every
socket opened with a ticket or bearer is bound to its session: ending the session closes it with
1008, a timer at the session's expiry asks the session again (a refresh extends it, the socket
stays), and the handlers end cleanly when they are closed from outside.

SEC3-09: refused WebSocket handshakes cost the loop that carries the E-stop 670 attempts a second
and one WARNING line each. Now a token bucket (per login session when the credential belongs to a
live one, else per peer) answers 429 before any credential is looked at, and refusals are logged
once per peer per minute.

SEC3-16: ``OPERATOR_AUTH_REQUIRED=0`` alone does not open any socket.

One event loop serves the app and every socket (``start_blocking_portal``), like the real server:
the TestClient makes one loop per call otherwise, and a close issued from another loop is not a
close. Every wait on a socket has a client-side deadline, so a regression FAILS, it never hangs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from anyio.from_thread import start_blocking_portal
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from backend.src.api import status as status_api
from backend.src.api.routers import auth as auth_router
from backend.src.api.routers import telemetry as telemetry_router
from backend.src.core import ws_policy
from backend.src.core.peer_address import TRUSTED_PROXIES_ENV
from backend.src.core.ws_policy import (
    ANONYMOUS_TELEMETRY_ENV,
    HandshakeLimiter,
    RefusalLog,
    WsRegistry,
)
from backend.src.main import app
from backend.src.services import auth_service as auth_module
from backend.src.services.auth_service import AuthService
from backend.src.services.websocket_hub import WebSocketHub, websocket_hub

CRED = "correct-horse-battery-staple"
LOGIN = "/api/v2/auth/login"
TICKET = "/api/v2/auth/ws-ticket"
LOGOUT = "/api/v2/auth/logout"
REFRESH = "/api/v2/auth/refresh"
STATUS = "/api/v2/ws/status"
TELEMETRY = "/api/v2/ws/telemetry"
CONTROL = "/api/v2/ws/control"
LEGACY_TELEMETRY = "/ws/telemetry"
LEGACY_CONTROL = "/ws/control"
ALL_ROUTES = (STATUS, TELEMETRY, CONTROL, LEGACY_TELEMETRY, LEGACY_CONTROL)
PEER = "203.0.113.9"


class Clock:
    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def world(monkeypatch):
    """The real app, operator auth ON, no simulator opt-in, one loop, a fresh registry whose
    handshake bucket and refusal log run on a fake clock."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.delenv("LAWNBERRY_ENV", raising=False)
    monkeypatch.delenv(ANONYMOUS_TELEMETRY_ENV, raising=False)
    monkeypatch.delenv(TRUSTED_PROXIES_ENV, raising=False)
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    service = AuthService()
    clock = Clock()
    registry = WsRegistry(
        handshakes=HandshakeLimiter(clock=clock), refusals=RefusalLog(clock=clock)
    )
    for module in (auth_router, telemetry_router, status_api):
        monkeypatch.setattr(module, "ws_registry", registry)
    monkeypatch.setattr(auth_router, "primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.middleware.rate_limiting.primary_auth_service", service)
    monkeypatch.setattr(auth_router, "_ws_tickets", {})
    monkeypatch.setattr(websocket_hub, "_telemetry_lock", asyncio.Lock())
    monkeypatch.setattr(websocket_hub, "_telemetry_cache", None)
    monkeypatch.setattr(websocket_hub, "_telemetry_task", None)
    monkeypatch.setattr(websocket_hub, "telemetry_cadence_hz", 5.0)
    for table in (websocket_hub.clients, websocket_hub.subscriptions, websocket_hub._client_topics):
        table.clear()
    with start_blocking_portal() as portal:
        client = TestClient(app, client=(PEER, 50000))
        client.portal = portal
        yield SimpleNamespace(
            client=client, service=service, registry=registry, clock=clock, portal=portal
        )
    for table in (websocket_hub.clients, websocket_hub.subscriptions, websocket_hub._client_topics):
        table.clear()


def login(world) -> dict[str, str]:
    response = world.client.post(LOGIN, json={"credential": CRED})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def ticket(world, headers) -> str:
    response = world.client.post(TICKET, headers=headers, json={})
    assert response.status_code == 200, response.text
    return response.json()["ticket"]


def session_id_of(world, headers) -> str:
    return world.service.peek_session(headers["Authorization"].split()[1]).session_id


def refused(client, url: str, **kwargs) -> WebSocketDenialResponse:
    with pytest.raises(WebSocketDenialResponse) as denied:
        with client.websocket_connect(url, **kwargs):
            pass
    return denied.value


def _within(read, timeout: float, what: str):
    """Run one blocking TestClient read with a deadline.

    The TestClient waits for ever for a frame that never comes (a close frame that was lost made
    the whole suite hang once), so every read runs in a helper thread that is given up on after
    ``timeout``: the test fails with ``what`` instead. The abandoned thread ends when the fixture's
    portal is closed.
    """
    outcome: list[tuple[bool, object]] = []

    def run() -> None:
        try:
            outcome.append((True, read()))
        except BaseException as exc:  # WebSocketDisconnect, or the portal going away
            outcome.append((False, exc))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout)
    if not outcome:
        pytest.fail(f"{what}: nothing arrived within {timeout:g} s")
    ok, value = outcome[0]
    if not ok:
        raise value  # type: ignore[misc]
    return value


def recv_json(ws, timeout: float = 10.0):
    return _within(ws.receive_json, timeout, "receive_json")


def recv_text(ws, timeout: float = 10.0) -> str:
    return _within(ws.receive_text, timeout, "receive_text")  # type: ignore[return-value]


def close_code(ws, route: str, *, deadline_s: float = 6.0) -> int:
    """The close code the server sends, polling with a frame the route answers (or, for the status
    stream, just reading its frames). Never blocks past ``deadline_s``: an open socket, or a close
    frame that never arrives, fails."""
    ping = (
        {"type": "ping"}
        if route in (TELEMETRY, CONTROL, LEGACY_TELEMETRY, LEGACY_CONTROL)
        else None
    )
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            if ping is not None:
                ws.send_json(ping)
                time.sleep(0.1)
            _within(
                ws.receive_json,
                max(0.5, end - time.monotonic()),
                f"{route}: no frame and no close frame",
            )
        except WebSocketDisconnect as closed:
            return closed.code
    pytest.fail(f"{route}: the socket is still open {deadline_s} s after its session ended")


def settled(world, *, open_sockets: int = 0, within: float = 3.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        if world.registry.snapshot()["open"] == open_sockets:
            return True
        time.sleep(0.02)
    return world.registry.snapshot()["open"] == open_sockets


# ---- SEC3-07: the end of a login ends its sockets


@pytest.mark.parametrize("route", ALL_ROUTES)
def test_logout_closes_the_socket_with_1008(world, route):
    """All five routes (the review's repro opened three): old code left the socket open and
    answering (``/ws/status`` kept streaming) after the logout returned 200."""
    headers = login(world)
    with world.client.websocket_connect(f"{route}?ticket={ticket(world, headers)}") as ws:
        assert world.registry.bound_count(session_id_of(world, headers)) == 1
        assert world.client.post(LOGOUT, headers=headers, json={}).status_code == 200
        assert close_code(ws, route) == 1008
    assert settled(world)


def test_a_logout_closes_every_socket_of_the_session_and_only_those(world):
    mine, other = login(world), login(world)
    routes = (STATUS, TELEMETRY, CONTROL)
    sockets = []
    try:
        for route in routes:
            ctx = world.client.websocket_connect(f"{route}?ticket={ticket(world, mine)}")
            sockets.append((route, ctx, ctx.__enter__()))
        bystander_ctx = world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, other)}")
        bystander = bystander_ctx.__enter__()
        assert world.registry.bound_count(session_id_of(world, mine)) == 3
        assert world.client.post(LOGOUT, headers=mine, json={}).status_code == 200
        assert [close_code(ws, route) for route, _ctx, ws in sockets] == [1008, 1008, 1008]
        assert "battery_percentage" in recv_json(bystander), "another login's socket stays"
        bystander_ctx.__exit__(None, None, None)
    finally:
        for _route, ctx, _ws in reversed(sockets):
            ctx.__exit__(None, None, None)
    assert settled(world)


def test_a_socket_closes_at_its_sessions_expiry(world):
    """The 8 h expiry: here the session ends 1.5 s after the socket opened (old code: never)."""
    headers = login(world)
    session = world.service.peek_session(headers["Authorization"].split()[1])
    session.expires_at = datetime.now(UTC) + timedelta(seconds=1.5)
    started = time.monotonic()
    with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}") as ws:
        assert close_code(ws, STATUS, deadline_s=8.0) == 1008
        assert 1.0 <= time.monotonic() - started < 6.0
    assert session_id_of_gone(world, session) and settled(world)


def session_id_of_gone(world, session) -> bool:
    return session.session_id not in world.service.active_sessions


def test_a_refresh_extends_the_session_and_the_socket_stays(world):
    headers = login(world)
    token = headers["Authorization"].split()[1]
    session = world.service.peek_session(token)
    session.expires_at = datetime.now(UTC) + timedelta(seconds=1.5)
    with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}") as ws:
        assert world.client.post(REFRESH, headers=headers, json={}).status_code == 200
        end = time.monotonic() + 3.2  # well past the old expiry
        frames = 0
        while time.monotonic() < end:
            assert "battery_percentage" in recv_json(ws)
            frames += 1
        assert frames >= 5, "the socket was cut at the OLD expiry although the session was extended"
        assert world.client.post(LOGOUT, headers=headers, json={}).status_code == 200
        assert close_code(ws, STATUS) == 1008, "and the timer was re-armed, not forgotten"


def test_a_session_removed_behind_the_services_back_is_noticed_by_the_timer(world, monkeypatch):
    monkeypatch.setattr(ws_policy, "EXPIRY_RECHECK_MAX_S", 0.3)
    headers = login(world)
    with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}") as ws:
        del world.service.active_sessions[session_id_of(world, headers)]  # no terminate_session
        assert close_code(ws, STATUS, deadline_s=4.0) == 1008


def test_a_credential_change_closes_the_sockets(world):
    headers = login(world)
    with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}") as ws:
        new_credential = secrets.token_urlsafe(18)
        assert world.portal.call(world.service.update_operator_credential, CRED, new_credential)
        assert close_code(ws, STATUS) == 1008


def test_cleaning_up_expired_sessions_closes_their_sockets(world):
    headers = login(world)
    session = world.service.peek_session(headers["Authorization"].split()[1])
    with world.client.websocket_connect(f"{TELEMETRY}?ticket={ticket(world, headers)}") as ws:
        session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        world.portal.call(world.service.cleanup_expired_sessions)
        assert close_code(ws, TELEMETRY) == 1008


def test_the_simulator_opt_ins_throw_away_session_is_not_bound(world, monkeypatch):
    """No stored session, no login to end: the socket is admitted and left alone."""
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
    headers = login(world)
    with world.client.websocket_connect(STATUS) as ws:  # no credential: the opt-in admits it
        assert "battery_percentage" in recv_json(ws)
        assert world.registry._bound == {} and world.registry.snapshot()["open"] == 1
        assert world.client.post(LOGOUT, headers=headers, json={}).status_code == 200
        assert "battery_percentage" in recv_json(ws), "a logout must not close this socket"


def test_an_anonymous_socket_is_not_bound(world, monkeypatch):
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "any")
    headers = login(world)
    with world.client.websocket_connect(STATUS) as ws:
        assert world.registry._bound == {}
        assert world.client.post(LOGOUT, headers=headers, json={}).status_code == 200
        assert "battery_percentage" in recv_json(ws)


def test_a_socket_closed_before_it_is_accepted_gives_its_slot_back(world, monkeypatch):
    """``/ws/status`` accepts AFTER admission. A logout in that window closes the socket before the
    accept; the accept then fails, and the slot must still go back (it used to be taken outside the
    ``try``)."""
    real_admit = status_api.admit_websocket

    async def admit_then_logout(websocket, **kwargs):
        admission = await real_admit(websocket, **kwargs)
        await world.service.terminate_session(admission.session.session_id, "user_logout")
        return admission

    monkeypatch.setattr(status_api, "admit_websocket", admit_then_logout)
    headers = login(world)
    with pytest.raises(WebSocketDisconnect) as closed:
        with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}"):
            pass
    assert closed.value.code == 1008
    assert settled(world), "the slot taken before the accept leaked"


def test_every_websocket_route_is_covered_by_the_binding_tests():
    """The five routes of the app (``grep websocket(`` / ``add_websocket_route``): a sixth route
    that forgets to go through ``admit_websocket`` would not be bound."""
    routes = {route.path for route in app.routes if route.__class__.__name__ == "APIWebSocketRoute"}
    routes |= {r.path for r in app.routes if r.__class__.__name__ == "WebSocketRoute"}
    assert routes == set(ALL_ROUTES)


# ---- the service side: a listener can neither fail nor hang the end of a login


async def test_a_failing_and_a_hanging_listener_do_not_fail_or_hang_the_logout(monkeypatch, caplog):
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.setattr(auth_module, "TERMINATION_LISTENER_TIMEOUT_S", 0.2)
    service = AuthService()
    seen: list[tuple[str, str]] = []

    async def failing(session_id: str, reason: str) -> None:
        raise RuntimeError("boom")

    async def hanging(session_id: str, reason: str) -> None:
        await asyncio.Event().wait()

    async def recording(session_id: str, reason: str) -> None:
        seen.append((session_id, reason))
        assert session_id not in service.active_sessions, "listeners run AFTER the session is gone"

    for listener in (failing, hanging, recording):
        service.add_termination_listener(listener)
        service.add_termination_listener(listener)  # idempotent
    result = await service.authenticate(CRED, client_ip=PEER)
    sid = result.session.session_id
    started = time.monotonic()
    with caplog.at_level(logging.ERROR, logger=auth_module.logger.name):
        assert await service.terminate_session(sid, "user_logout") is True
    assert time.monotonic() - started < 1.5, "a hanging listener held the logout up"
    assert seen == [(sid, "user_logout")], "one failing listener must not stop the next"
    assert await service.terminate_session(sid) is False and len(seen) == 1
    assert any("failed" in r.getMessage() for r in caplog.records)
    assert any("abandoned" in r.getMessage() for r in caplog.records)


async def test_quiet_verification_is_silent_but_still_ends_an_expired_session(monkeypatch, caplog):
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    service = AuthService()
    result = await service.authenticate(CRED, client_ip=PEER)
    with caplog.at_level(logging.DEBUG, logger=auth_module.logger.name):
        assert await service.verify_token("junk.junk.junk", quiet=True) is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING], "junk logged"
        assert await service.verify_token("junk.junk.junk") is None  # the HTTP default is loud
        assert [r for r in caplog.records if "Invalid JWT token" in r.getMessage()]
        caplog.clear()
        result.session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert await service.verify_token(result.token, quiet=True) is None
    assert result.session.session_id not in service.active_sessions, "terminated on contact"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_the_hub_drops_a_client_whose_socket_was_closed_from_outside():
    """Passes on the old code too (the hub already dropped a client it could not send to): it pins
    that a REAL closed Starlette socket is such a client and gets no frame after the close."""
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "websocket.connect"}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "websocket", "headers": [], "client": (PEER, 1), "query_string": b""}
    socket = WebSocket(scope, receive, send)
    await socket.accept()
    hub = WebSocketHub()
    hub.clients["c"] = socket
    hub.subscriptions["t"] = {"c"}
    await socket.close(code=1008, reason="session ended")
    await hub.broadcast_to_topic("t", {"x": 1})
    assert "c" not in hub.clients and hub.subscriptions == {}
    assert [m["type"] for m in sent] == ["websocket.accept", "websocket.close"]


# ---- SEC3-09: the handshake budget


def junk(n: int) -> dict[str, str]:
    return {"Authorization": f"Bearer junk-{n}.junk.junk"}


def test_junk_handshakes_are_limited_with_a_429_and_retry_after(world):
    codes = [refused(world.client, STATUS, headers=junk(n)).status_code for n in range(25)]
    assert codes == [401] * 20 + [429] * 5, "burst 20, then the bucket is empty"
    denied = refused(world.client, STATUS, headers=junk(99))
    assert denied.status_code == 429 and int(denied.headers["retry-after"]) >= 1
    world.clock.now += 1.0  # two tokens
    assert [refused(world.client, STATUS, headers=junk(n)).status_code for n in range(3)] == [
        401,
        401,
        429,
    ]
    assert world.registry.snapshot()["refused"]["handshake"] == 7


def test_the_limit_holds_without_any_credential_too(world):
    codes = [refused(world.client, STATUS).status_code for _ in range(22)]
    assert codes == [401] * 20 + [429] * 2


def test_junk_from_the_operators_address_does_not_starve_the_operators_handshake(world):
    """The operator and the junk sender share one address (every visitor behind an undeclared
    forwarder does): the operator's bucket is its login session's, so this still connects."""
    headers = login(world)
    for n in range(40):
        refused(world.client, STATUS, headers=junk(n))
    assert refused(world.client, STATUS, headers=junk(41)).status_code == 429
    for _ in range(3):
        with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}") as ws:
            assert "battery_percentage" in recv_json(ws)
    with world.client.websocket_connect(STATUS, headers=headers) as ws:  # a bearer, too
        assert "battery_percentage" in recv_json(ws)


def test_another_peer_is_not_limited_by_this_peers_junk(world):
    for n in range(30):
        refused(world.client, STATUS, headers=junk(n))
    other = TestClient(app, client=("203.0.113.77", 50000))
    other.portal = world.portal
    assert refused(other, STATUS).status_code == 401


def test_a_dead_sessions_token_does_not_buy_a_bucket_of_its_own(world):
    """A client cannot choose its bucket by what it sends: only a LIVE session's credential is
    keyed by session, so tokens of a logged-out session drain the peer's bucket like any junk."""
    headers = login(world)
    assert world.client.post(LOGOUT, headers=headers, json={}).status_code == 200
    codes = [refused(world.client, STATUS, headers=headers).status_code for _ in range(22)]
    assert codes == [401] * 20 + [429] * 2


def test_a_refused_handshake_does_not_burn_the_ticket(world):
    """The budget comes BEFORE the credential is redeemed: a ticket presented while the session's
    bucket is empty is refused with 429 and is still good afterwards."""
    headers = login(world)
    for _ in range(20):
        with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}"):
            pass
    spare = ticket(world, headers)
    assert refused(world.client, f"{STATUS}?ticket={spare}").status_code == 429
    assert spare in auth_router._ws_tickets, "a 429 must not redeem the ticket"
    world.clock.now += 1.0
    with world.client.websocket_connect(f"{STATUS}?ticket={spare}") as ws:
        assert "battery_percentage" in recv_json(ws)


def test_a_flood_of_refusals_is_one_line_per_peer_per_minute_with_the_count(world, caplog):
    with caplog.at_level(logging.DEBUG):
        for n in range(300):
            refused(world.client, STATUS, headers=junk(n))
        lines = [r for r in caplog.records if "handshake refused" in r.getMessage()]
        assert len(lines) == 1 and "401" in lines[0].getMessage(), [r.getMessage() for r in lines]
        assert not [r for r in caplog.records if "JWT" in r.getMessage()], "a WARNING per bearer"
        world.clock.now += 61.0
        refused(world.client, STATUS, headers=junk(1000))
        lines = [r for r in caplog.records if "handshake refused" in r.getMessage()]
    assert len(lines) == 2 and "300 refusal(s)" in lines[1].getMessage()
    assert all(r.levelno == logging.WARNING for r in lines)
    assert "junk-" not in " ".join(r.getMessage() for r in caplog.records), "a token was logged"


def test_a_full_connection_budget_is_logged_once_not_per_attempt(world, caplog):
    world.registry.max_total = 1
    headers = login(world)
    with world.client.websocket_connect(f"{STATUS}?ticket={ticket(world, headers)}"):
        with caplog.at_level(logging.WARNING, logger=auth_router.logger.name):
            codes = [
                refused(world.client, f"{STATUS}?ticket={ticket(world, headers)}").status_code
                for _ in range(5)
            ]
    assert codes == [503] * 5
    assert len([r for r in caplog.records if "handshake refused" in r.getMessage()]) == 1


# ---- SEC3-16: authentication off does not open the sockets


@pytest.mark.parametrize("route", ALL_ROUTES)
def test_operator_auth_off_alone_does_not_open_any_socket(world, monkeypatch, route):
    """Passes on the old code too (the code was right, its docstring was wrong): with
    ``OPERATOR_AUTH_REQUIRED=0`` and neither the simulator opt-in nor the anonymous opt-in, a
    socket with no credential is refused."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    assert refused(world.client, route).status_code == 401
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "any")  # only the explicit opt-in opens one
    if route not in (CONTROL, LEGACY_CONTROL):
        with world.client.websocket_connect(route) as ws:
            assert json.loads(recv_text(ws))
    else:
        with world.client.websocket_connect(route) as ws:  # auth is off: nothing to enforce
            assert json.loads(recv_text(ws))["event"] == "connection.established"
