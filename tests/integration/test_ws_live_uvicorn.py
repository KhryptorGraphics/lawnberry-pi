"""A real uvicorn, real sockets: a login ends its WebSockets, and a handshake flood is limited.

``test_ws_session_binding.py`` proves the behaviour in-process; here the same app is served by the
REAL uvicorn (the unit's protocol and frame limit, ``--no-proxy-headers``) and driven by a real
WebSocket client, because two claims only mean something on the wire:

* SEC3-07: after ``POST /auth/logout`` each socket the login opened receives a real close frame,
  code 1008, within about a second, and nothing after it (the review's repro, ``e1_logout_ws.py``,
  saw all three stay open and keep streaming); a session whose ``expires_at`` is 1.5 s ahead is
  closed at its expiry; a refresh keeps the socket.
* SEC3-09: a handshake denied BEFORE the accept reaches the client as an HTTP 429 with
  ``Retry-After`` (like the 401 and 503 it already sent), a flood of junk handshakes is limited
  and writes a handful of log lines instead of one per attempt, and the operator's own ticket
  handshake succeeds in the middle of it.

The server runs in a thread of this process (so the test can move a session's expiry), clients
connect from the loopback alias ``127.0.0.9``. The operator credential and the signing secret are
generated in memory per run; nothing is written to a file.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from backend.src.api import status as status_api
from backend.src.api.routers import auth as auth_router
from backend.src.api.routers import telemetry as telemetry_router
from backend.src.core.ws_policy import ANONYMOUS_TELEMETRY_ENV, WsRegistry
from backend.src.main import app
from backend.src.services.auth_service import AuthService
from backend.src.services.websocket_hub import websocket_hub

pytestmark = [pytest.mark.integration]

SOURCE = "127.0.0.9"
GUARDED = "backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def live(monkeypatch):
    credential = secrets.token_urlsafe(18)
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("LAWNBERRY_ENV", "production")
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.delenv(ANONYMOUS_TELEMETRY_ENV, raising=False)
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", credential)
    monkeypatch.setenv("LAWN_BERRY_TRUSTED_PROXIES", "127.0.0.2=single,127.0.0.3=last")
    service = AuthService()
    registry = WsRegistry()
    for module in (auth_router, telemetry_router, status_api):
        monkeypatch.setattr(module, "ws_registry", registry)
    monkeypatch.setattr(auth_router, "primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.middleware.rate_limiting.primary_auth_service", service)
    monkeypatch.setattr(auth_router, "_ws_tickets", {})
    monkeypatch.setattr(websocket_hub, "_telemetry_lock", asyncio.Lock())
    monkeypatch.setattr(websocket_hub, "_telemetry_cache", None)
    monkeypatch.setattr(websocket_hub, "_telemetry_task", None)
    websocket_hub.clients.clear()
    websocket_hub.subscriptions.clear()
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        http=GUARDED,
        ws_max_size=65536,
        proxy_headers=False,
        timeout_keep_alive=5,
        backlog=128,
        lifespan="off",
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    http = httpx.Client(
        base_url=f"http://127.0.0.1:{port}",
        timeout=10,
        transport=httpx.HTTPTransport(local_address=SOURCE),
    )
    try:
        yield SimpleNamespace(
            port=port, http=http, service=service, registry=registry, credential=credential
        )
    finally:
        http.close()
        server.should_exit = True
        thread.join(timeout=15)
        websocket_hub.clients.clear()
        websocket_hub.subscriptions.clear()


def login(live) -> dict[str, str]:
    response = live.http.post("/api/v2/auth/login", json={"credential": live.credential})
    assert response.status_code == 200, response.status_code
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def ticket(live, headers) -> str:
    response = live.http.post("/api/v2/auth/ws-ticket", headers=headers, json={})
    assert response.status_code == 200
    return response.json()["ticket"]


def open_socket(live, path: str, *, query: str = "", headers=None):
    return connect(
        f"ws://127.0.0.1:{live.port}{path}{query}",
        additional_headers=headers,
        source_address=(SOURCE, 0),
        open_timeout=10,
        close_timeout=2,
    )


def until_closed(ws, *, within: float) -> tuple[int, int | None, float]:
    """``(frames, close code, seconds)``: read until the server closes. Never past ``within``."""
    started = time.monotonic()
    frames = 0
    while time.monotonic() - started < within:
        try:
            ws.recv(timeout=0.5)
            frames += 1
        except TimeoutError:
            continue
        except ConnectionClosed as closed:
            return frames, closed.rcvd.code if closed.rcvd else None, time.monotonic() - started
    return frames, None, time.monotonic() - started


def test_logout_closes_three_real_sockets_with_1008_and_no_more_frames(live):
    headers = login(live)
    paths = ("/api/v2/ws/telemetry", "/api/v2/ws/status", "/api/v2/ws/control")
    sockets = {p: open_socket(live, p, query=f"?ticket={ticket(live, headers)}") for p in paths}
    try:
        for ws in sockets.values():
            assert ws.recv(timeout=5)  # established / the first status frame
        assert live.registry.snapshot()["open"] == 3
        assert live.http.post("/api/v2/auth/logout", headers=headers, json={}).status_code == 200
        results = {p: until_closed(ws, within=5.0) for p, ws in sockets.items()}
        for path, (frames, code, seconds) in results.items():
            assert code == 1008, f"{path}: closed with {code} after {frames} frames"
            assert seconds < 3.0, f"{path}: took {seconds:.1f} s to close"
            assert frames <= 3, f"{path}: {frames} frames after the logout"
        for ws in sockets.values():
            with pytest.raises(ConnectionClosed):
                ws.recv(timeout=1.0)  # the close frame was the last thing the server sent
        deadline = time.monotonic() + 5
        while live.registry.snapshot()["open"] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert live.registry.snapshot()["open"] == 0, "the slots were not given back"
    finally:
        for ws in sockets.values():
            ws.close()


def test_a_session_expiring_in_two_seconds_closes_a_real_socket_at_expiry(live):
    headers = login(live)
    session = live.service.peek_session(headers["Authorization"].split()[1])
    session.expires_at = datetime.now(UTC) + timedelta(seconds=1.5)
    with open_socket(live, "/api/v2/ws/status", query=f"?ticket={ticket(live, headers)}") as ws:
        frames, code, seconds = until_closed(ws, within=8.0)
    assert code == 1008 and frames >= 3, (frames, code)
    assert 1.0 <= seconds < 5.0, f"closed after {seconds:.1f} s for a 1.5 s session"
    assert session.session_id not in live.service.active_sessions


def test_a_refresh_keeps_a_real_socket_past_the_old_expiry(live):
    headers = login(live)
    session = live.service.peek_session(headers["Authorization"].split()[1])
    session.expires_at = datetime.now(UTC) + timedelta(seconds=1.5)
    with open_socket(live, "/api/v2/ws/status", query=f"?ticket={ticket(live, headers)}") as ws:
        assert live.http.post("/api/v2/auth/refresh", headers=headers, json={}).status_code == 200
        end = time.monotonic() + 3.0
        frames = 0
        while time.monotonic() < end:
            assert ws.recv(timeout=2)
            frames += 1
        assert frames >= 5
        assert live.http.post("/api/v2/auth/logout", headers=headers, json={}).status_code == 200
        assert until_closed(ws, within=5.0)[1] == 1008


def test_the_handlers_end_cleanly_no_traceback_in_the_server_log(live, caplog):
    headers = login(live)
    with caplog.at_level(logging.WARNING):
        ws = open_socket(live, "/api/v2/ws/control", query=f"?ticket={ticket(live, headers)}")
        ws.recv(timeout=5)
        live.http.post("/api/v2/auth/logout", headers=headers, json={})
        assert until_closed(ws, within=5.0)[1] == 1008
    assert not [r for r in caplog.records if r.exc_info or "Exception in ASGI" in r.getMessage()]


def raw_handshake(port: int, path: str, headers: str = "") -> int:
    """One real handshake from ``SOURCE``; the HTTP status of the answer (0: no answer)."""
    try:
        with socket.create_connection(
            ("127.0.0.1", port), timeout=5, source_address=(SOURCE, 0)
        ) as s:
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall(
                (
                    f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                    f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                    f"Sec-WebSocket-Version: 13\r\n{headers}\r\n"
                ).encode()
            )
            head = s.recv(300)
            return int(head.split(b" ", 2)[1]) if head else 0
    except (OSError, ValueError, IndexError):
        return -1


def test_a_429_before_the_accept_reaches_a_real_client_with_retry_after(live):
    junk = {"Authorization": "Bearer junk.junk.junk"}
    statuses = []
    retry_after = None
    for _ in range(25):
        with pytest.raises(InvalidStatus) as denied:
            open_socket(live, "/api/v2/ws/status", headers=junk)
        statuses.append(denied.value.response.status_code)
        if statuses[-1] == 429:
            retry_after = denied.value.response.headers.get("Retry-After")
            break
    assert statuses[:20] == [401] * 20 and statuses[-1] == 429, statuses
    assert retry_after is not None and int(retry_after) >= 1


def test_a_flood_of_junk_handshakes_is_limited_quiet_and_leaves_the_operator_in(live, caplog):
    headers = login(live)
    results: list[int] = []
    lock = threading.Lock()
    stop = time.monotonic() + 2.5

    def flood() -> None:
        while time.monotonic() < stop:
            code = raw_handshake(
                live.port, "/api/v2/ws/telemetry", "Authorization: Bearer j.j.j\r\n"
            )
            with lock:
                results.append(code)

    with caplog.at_level(logging.DEBUG):
        threads = [threading.Thread(target=flood) for _ in range(8)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        with open_socket(live, "/api/v2/ws/status", query=f"?ticket={ticket(live, headers)}") as ws:
            assert ws.recv(timeout=5), "the operator's handshake in the middle of the flood"
        for t in threads:
            t.join(timeout=15)
    attempts = len(results)
    assert attempts >= 100, f"the flood only managed {attempts} handshakes"
    assert set(results) <= {401, 429}, set(results)
    assert results.count(401) <= 20 + 2 * 4, "more than the burst plus the refill got through"
    assert results.count(429) > attempts * 0.8
    lines = [r for r in caplog.records if "handshake refused" in r.getMessage()]
    assert 1 <= len(lines) <= 2, f"{len(lines)} lines for {attempts} handshakes"
    assert not [r for r in caplog.records if "Invalid JWT token" in r.getMessage()]
