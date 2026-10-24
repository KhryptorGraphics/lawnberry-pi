"""WebSocket tickets, the command-capable socket, and the proxy-aware client address.

Before:

* the 8 hour login token went into the WebSocket URL (``?access_token=``), which uvicorn logs;
  ``/ws/control`` admitted any loopback peer (every browser behind the UI server) with no
  credential, and ``/api/v2/ws/status`` checked nothing at all;
* every browser behind the UI server reached the backend from 127.0.0.1, so one wrong-password
  lockout, or one exhausted rate bucket, hit all of them.

Now a browser trades its bearer token for a 30 s single-use ticket (``POST /auth/ws-ticket``),
``/ws/control`` needs a credential whenever operator auth is on, the URL token is no longer read,
and the ``X-Forwarded-For`` of a TRUSTED proxy (last hop only) names the client the limiters
and the lockout count. From any other peer the header is not read at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.testclient import WebSocketDenialResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.src.api import status as status_api
from backend.src.api.routers import auth as auth_router
from backend.src.api.routers.auth import _authorize_websocket
from backend.src.core.peer_address import (
    TRUSTED_PROXIES_ENV,
    client_host,
    resolve_client_host,
    trusted_proxy_networks,
)
from backend.src.main import app
from backend.src.middleware.rate_limiting import register_global_rate_limiter
from backend.src.middleware.security import UNLIMITED_AUTH_PATHS, SecurityMiddleware
from backend.src.services.auth_service import AuthService

CRED = "correct-horse-battery-staple"
LOGIN = "/api/v2/auth/login"
TICKET = "/api/v2/auth/ws-ticket"
CONTROL = "/api/v2/ws/control"

LOOPBACK = trusted_proxy_networks({})  # unset: the default, the loopback only


def _nets(text: str):
    return trusted_proxy_networks({TRUSTED_PROXIES_ENV: text})


def _client_for(target, peer=("127.0.0.1", 123)) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=target, client=peer)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def default_trust(monkeypatch):
    monkeypatch.delenv(TRUSTED_PROXIES_ENV, raising=False)


# ---- who is the client --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("peer", "lines", "expected"),
    [
        # a trusted proxy recorded the real address as the LAST hop; what the client wrote to its
        # left is never read
        ("127.0.0.1", ["203.0.113.9"], "203.0.113.9"),
        ("127.0.0.1", ["6.6.6.6, 203.0.113.9"], "203.0.113.9"),
        ("127.0.0.1", ["6.6.6.6", "203.0.113.9"], "203.0.113.9"),  # two header lines
        ("::1", ["203.0.113.9"], "203.0.113.9"),
        ("::ffff:127.0.0.1", ["203.0.113.9"], "203.0.113.9"),
        ("127.0.0.1", ["203.0.113.9:51234"], "203.0.113.9"),
        ("127.0.0.1", ["[2001:db8::7]:51234"], "2001:db8::7"),
        ("127.0.0.1", ["203.0.113.9, 127.0.0.2"], "203.0.113.9"),  # a trusted hop is skipped
        # nothing usable: the proxy's own address (one shared budget), never a guess
        ("127.0.0.1", [], "127.0.0.1"),
        ("127.0.0.1", ["203.0.113.9, unknown"], "127.0.0.1"),
        ("127.0.0.1", ["garbage"], "127.0.0.1"),
        ("127.0.0.1", ["127.0.0.5"], "127.0.0.1"),  # only trusted hops
        # not a trusted proxy: the header is not read at all
        ("198.51.100.9", ["127.0.0.1"], "198.51.100.9"),
        ("198.51.100.9", ["203.0.113.9"], "198.51.100.9"),
        ("testclient", ["203.0.113.9"], "testclient"),
        (None, ["203.0.113.9"], None),
    ],
)
def test_the_client_is_the_last_forwarded_hop_of_a_trusted_proxy_only(peer, lines, expected):
    assert resolve_client_host(peer, lines, LOOPBACK) == expected


def test_the_trusted_proxy_list_is_configurable():
    ahead = ["203.0.113.9"]
    assert resolve_client_host("10.1.2.3", ahead, _nets("10.0.0.0/8, 192.0.2.1")) == "203.0.113.9"
    assert resolve_client_host("127.0.0.1", ahead, _nets("10.0.0.0/8")) == "127.0.0.1"
    # an explicitly empty value trusts nobody, the loopback included
    assert resolve_client_host("127.0.0.1", ahead, _nets("")) == "127.0.0.1"
    # a bad entry is ignored and the good ones stay
    assert resolve_client_host("10.1.2.3", ahead, _nets("nonsense, 10.0.0.0/8")) == "203.0.113.9"


def test_the_environment_is_read_when_no_networks_are_passed(monkeypatch):
    assert resolve_client_host("10.1.2.3", ["203.0.113.9"]) == "10.1.2.3"
    monkeypatch.setenv(TRUSTED_PROXIES_ENV, "10.0.0.0/8")
    assert resolve_client_host("10.1.2.3", ["203.0.113.9"]) == "203.0.113.9"


def _request(peer: str, *forwarded: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-forwarded-for", line.encode()) for line in forwarded],
        "client": (peer, 1234),
    }
    return Request(scope)


def test_client_host_reads_every_forwarded_line_of_a_real_request():
    assert client_host(_request("127.0.0.1", "6.6.6.6", "203.0.113.9")) == "203.0.113.9"
    assert client_host(_request("198.51.100.9", "203.0.113.9")) == "198.51.100.9"


# ---- the limiters count the real client behind a trusted proxy ----------------------------------


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "3")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "60")
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService()
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setattr(auth_router, "primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.rest_v1._auth_service", service)
    return service


async def _bad_logins(client: httpx.AsyncClient, forwarded: list[str]) -> list[int]:
    codes = []
    for n, value in enumerate(forwarded):
        resp = await client.post(
            LOGIN, json={"credential": f"wrong-{n}"}, headers={"X-Forwarded-For": value}
        )
        codes.append(resp.status_code)
    return codes


async def test_browsers_behind_a_trusted_proxy_are_locked_out_one_by_one(strict):
    async with _client_for(app) as proxy:  # the UI server, on the loopback
        # three wrong passwords from one browser; it also rotates what it writes to the LEFT
        assert await _bad_logins(proxy, [f"6.6.6.{n}, 203.0.113.9" for n in range(3)]) == [401] * 3
        locked = await proxy.post(
            LOGIN, json={"credential": CRED}, headers={"X-Forwarded-For": "6.6.6.99, 203.0.113.9"}
        )
        assert locked.status_code == 429, "the right credential must not get past the lockout"
        # another browser behind the same proxy is untouched
        other = await proxy.post(
            LOGIN, json={"credential": CRED}, headers={"X-Forwarded-For": "203.0.113.10"}
        )
        assert other.status_code == 200, other.text


async def test_an_untrusted_peer_cannot_dodge_the_lockout_with_forwarded_for(strict):
    async with _client_for(app, ("198.51.100.9", 5)) as client:
        assert await _bad_logins(client, [f"10.0.0.{n}" for n in range(3)]) == [401] * 3
        locked = await client.post(
            LOGIN, json={"credential": CRED}, headers={"X-Forwarded-For": "10.0.0.250"}
        )
        assert locked.status_code == 429


async def test_uvicorn_proxy_headers_and_the_backend_agree_on_the_client(strict):
    """In the shipped deployment uvicorn (``--proxy-headers``, trusting 127.0.0.1) has already
    replaced ``request.client`` with the forwarded hop before the backend looks. Both layers take
    the last untrusted hop, so a client-written prefix is ignored and each browser is counted on
    its own. Pins the documented assumption against a uvicorn upgrade."""
    wrapped = ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1")
    async with _client_for(wrapped) as proxy:
        assert await _bad_logins(proxy, [f"6.6.6.{n}, 203.0.113.9" for n in range(3)]) == [401] * 3
        locked = await proxy.post(
            LOGIN, json={"credential": CRED}, headers={"X-Forwarded-For": "6.6.6.99, 203.0.113.9"}
        )
        assert locked.status_code == 429
        other = await proxy.post(
            LOGIN, json={"credential": CRED}, headers={"X-Forwarded-For": "203.0.113.10"}
        )
        assert other.status_code == 200, other.text


def _auth_app() -> FastAPI:
    sub = FastAPI()

    @sub.post(LOGIN)
    async def login() -> JSONResponse:
        return JSONResponse({"detail": "bad credentials"}, status_code=401)

    @sub.post(TICKET)
    async def ticket() -> JSONResponse:
        return JSONResponse({"detail": "expired token"}, status_code=401)

    sub.add_middleware(
        SecurityMiddleware,
        rate_limit_window_seconds=60,
        rate_limit_max_attempts=3,
        lockout_failures=3,
        lockout_seconds=60,
    )
    return sub


async def test_the_auth_prefix_limiter_counts_each_browser_behind_a_trusted_proxy():
    async with _client_for(_auth_app()) as proxy:
        a = [
            (
                await proxy.post(LOGIN, json={}, headers={"X-Forwarded-For": "203.0.113.1"})
            ).status_code
            for _ in range(6)
        ]
        b = await proxy.post(LOGIN, json={}, headers={"X-Forwarded-For": "203.0.113.2"})
    assert a[:3] == [401] * 3 and set(a[3:]) == {429}, a
    assert b.status_code == 401


async def test_ticket_requests_are_not_counted_by_the_auth_prefix_limiter():
    """The UI asks for a ticket before every WebSocket (re)connect, with a token that may have
    expired (401): neither may use up the login's attempt budget or lock the login out."""
    async with _client_for(_auth_app()) as proxy:
        tickets = [(await proxy.post(TICKET, json={})).status_code for _ in range(12)]
        assert tickets == [401] * 12, "a ticket request was rate limited"
        login = await proxy.post(LOGIN, json={})
        assert login.status_code == 401, "the login was locked out by ticket requests"


def test_every_path_exempt_from_the_auth_limiter_is_a_real_post_route():
    routes = {(m, r.path) for r in app.routes if hasattr(r, "methods") for m in r.methods}
    assert UNLIMITED_AUTH_PATHS == {TICKET}
    assert all(("POST", path) in routes for path in UNLIMITED_AUTH_PATHS)


def _limited_app(monkeypatch, burst: str) -> FastAPI:
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_RATE", "0.1")
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_BURST", burst)
    monkeypatch.delenv("GLOBAL_RATE_LIMIT_OVERRIDES", raising=False)
    monkeypatch.delenv("GLOBAL_RATE_LIMIT_EXEMPT", raising=False)
    sub = FastAPI()
    register_global_rate_limiter(sub)

    def ok() -> dict:
        return {"ok": True}

    sub.add_api_route("/poll", ok, methods=["GET"])
    return sub


async def test_the_global_limiter_gives_each_browser_behind_a_trusted_proxy_its_own_bucket(
    monkeypatch,
):
    sub = _limited_app(monkeypatch, burst="3")
    async with _client_for(sub) as proxy:
        noisy = [
            (await proxy.get("/poll", headers={"X-Forwarded-For": "203.0.113.1"})).status_code
            for _ in range(6)
        ]
        quiet = [
            (await proxy.get("/poll", headers={"X-Forwarded-For": "203.0.113.2"})).status_code
            for _ in range(3)
        ]
    assert noisy.count(200) == 3 and noisy.count(429) == 3, noisy
    assert quiet == [200, 200, 200], "one noisy browser must not starve the others"


async def test_an_untrusted_peer_cannot_mint_limiter_buckets_with_forwarded_for(monkeypatch):
    sub = _limited_app(monkeypatch, burst="3")
    async with _client_for(sub, ("198.51.100.9", 1)) as client:
        codes = [
            (await client.get("/poll", headers={"X-Forwarded-For": f"10.0.0.{n}"})).status_code
            for n in range(10)
        ]
    assert codes.count(200) == 3 and codes.count(429) == 7, codes


# ---- WebSocket tickets --------------------------------------------------------------------------


class FakeWebSocket:
    def __init__(self, host="203.0.113.9", headers=None, query=None):
        self.client = (host, 4321)
        self.headers = headers or {}
        self.query_params = query or {}
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, _payload):
        raise AssertionError("streamed to a peer that was never authorized")


@pytest.fixture
def logged_in(strict, monkeypatch):
    """The real app, operator auth ON, one logged-in operator: (client, bearer headers, service)."""
    monkeypatch.setattr(auth_router, "_ws_tickets", {})
    client = TestClient(app)
    login = client.post(LOGIN, json={"credential": CRED})
    assert login.status_code == 200, login.text
    return client, {"Authorization": f"Bearer {login.json()['access_token']}"}, strict


def _ticket(client, headers) -> str:
    resp = client.post(TICKET, headers=headers, json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["expires_in"] == auth_router.WS_TICKET_TTL_SECONDS
    return resp.json()["ticket"]


def _refused(client, url: str, **kwargs) -> int:
    with pytest.raises(WebSocketDenialResponse) as refused:
        with client.websocket_connect(url, **kwargs):
            pass
    return refused.value.status_code


def test_a_ticket_opens_the_control_socket_exactly_once(logged_in):
    client, headers, _ = logged_in
    ticket = _ticket(client, headers)
    with client.websocket_connect(f"{CONTROL}?ticket={ticket}") as ws:
        assert json.loads(ws.receive_text())["event"] == "connection.established"
    assert _refused(client, f"{CONTROL}?ticket={ticket}") == 401, "a ticket is single use"


def test_the_ticket_route_needs_a_login(logged_in):
    client, _, _ = logged_in
    assert client.post(TICKET, json={}).status_code == 401
    assert client.post(TICKET, json={}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_the_control_socket_needs_a_credential_whoever_the_peer_is(logged_in):
    client, headers, _ = logged_in  # the TestClient is a "testclient" peer: admitted elsewhere
    assert _refused(client, CONTROL) == 401
    with client.websocket_connect(CONTROL, headers=headers) as ws:  # non-browser clients
        assert json.loads(ws.receive_text())["event"] == "connection.established"


def test_the_login_token_in_the_url_is_no_longer_a_credential(logged_in):
    client, headers, _ = logged_in
    token = headers["Authorization"].split()[1]
    for name in ("access_token", "token", "auth_token"):
        assert _refused(client, f"{CONTROL}?{name}={token}") == 401, name


def test_unknown_expired_and_orphaned_tickets_are_refused(logged_in):
    client, headers, _ = logged_in
    assert _refused(client, f"{CONTROL}?ticket=never-issued") == 401
    expired = _ticket(client, headers)
    auth_router._ws_tickets[expired]["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    assert _refused(client, f"{CONTROL}?ticket={expired}") == 401
    orphan = _ticket(client, headers)
    assert client.post("/api/v2/auth/logout", headers=headers, json={}).status_code == 200
    assert _refused(client, f"{CONTROL}?ticket={orphan}") == 401, "the login behind it ended"


def test_with_operator_auth_off_the_control_socket_keeps_its_old_behaviour(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    with TestClient(app).websocket_connect(CONTROL) as ws:
        assert json.loads(ws.receive_text())["event"] == "connection.established"


def test_the_ticket_table_is_bounded_and_forgets_expired_tickets(monkeypatch):
    monkeypatch.setattr(auth_router, "_ws_tickets", {})
    monkeypatch.setattr(auth_router, "WS_TICKET_LIMIT", 8)
    stale = auth_router._mint_ws_ticket("old")
    auth_router._ws_tickets[stale]["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    for n in range(50):
        auth_router._mint_ws_ticket(f"s{n}")
    assert len(auth_router._ws_tickets) <= 8
    assert stale not in auth_router._ws_tickets


# ---- the admission helper, directly ------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("SIM_MODE", "LAWN_BERRY_SIM_OPEN_AUTH", "LAWNBERRY_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setattr(auth_router, "_ws_tickets", {})
    return monkeypatch


async def test_a_ticket_admits_a_remote_peer_to_a_read_only_stream_once(clean_env, strict):
    result = await strict.authenticate(CRED, client_ip="203.0.113.9")
    ticket = auth_router._mint_ws_ticket(result.session.session_id)
    remote = FakeWebSocket(host="203.0.113.9", query={"ticket": ticket})
    session = await _authorize_websocket(remote)
    assert session.session_id == result.session.session_id
    with pytest.raises(HTTPException) as again:
        await _authorize_websocket(FakeWebSocket(host="203.0.113.9", query={"ticket": ticket}))
    assert again.value.status_code == 401


async def test_a_bad_ticket_is_refused_even_from_the_loopback(clean_env):
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(FakeWebSocket(host="127.0.0.1", query={"ticket": "nope"}))
    assert refused.value.status_code == 401


async def test_the_control_socket_helper_refuses_the_loopback_only_when_auth_is_on(clean_env):
    loopback = FakeWebSocket(host="127.0.0.1")
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(loopback, command_capable=True)
    assert refused.value.status_code == 401
    assert (await _authorize_websocket(loopback)).session_id, "read-only streams: unchanged"
    clean_env.setenv("OPERATOR_AUTH_REQUIRED", "0")
    assert (await _authorize_websocket(loopback, command_capable=True)).session_id


async def test_the_websocket_helper_never_reads_forwarded_for(clean_env):
    """The peer is ``websocket.client``; the helper itself ignores the header. (In the shipped
    deployment uvicorn has already replaced the peer with the forwarded hop by then, so a LAN
    browser behind the UI server arrives here as its own address: see the auth guide.)"""
    forged = FakeWebSocket(host="203.0.113.9", headers={"X-Forwarded-For": "127.0.0.1"})
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(forged)
    assert refused.value.status_code == 401
    loopback = FakeWebSocket(host="127.0.0.1", headers={"X-Forwarded-For": "203.0.113.9"})
    assert (await _authorize_websocket(loopback)).session_id, "the peer decides, not the header"


async def test_the_status_stream_is_authorized_before_it_is_accepted(clean_env):
    remote = FakeWebSocket(host="203.0.113.9")
    with pytest.raises(HTTPException) as refused:
        await status_api.ws_status(remote)
    assert refused.value.status_code == 401
    assert remote.accepted is False, "accepted (and streamed to) a peer with no credential"
