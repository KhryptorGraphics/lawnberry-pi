"""WebSocket admission without a credential, and what one socket may cost (review SEC20-02/14).

SEC20-02: a loopback peer was admitted to ``/ws/telemetry``, ``/api/v2/ws/telemetry`` and
``/api/v2/ws/status`` with no credential, and every ingress the product ships (a Cloudflare or
ngrok agent pointed at the UI server, nginx) reaches the backend from the loopback, so any remote
visitor was "the browser on the mower". Admitted sockets were unbounded: one anonymous socket
sending 330 000 ``subscribe`` frames with distinct 1 kB topic names grew the backend from 241 to
612 MiB, and ``/api/v2/ws/status`` ran a full sensor read per socket per tick.

SEC20-14: a browser page open on the mower could read the stream cross-origin (CORS does not cover
WebSockets): loopback peer, ``Origin: https://evil.example``, no credential, admitted.

Now no peer is admitted without a credential unless ``LAWN_BERRY_ANONYMOUS_TELEMETRY`` says so (and
then only the read-only streams, only for an allowed ``Origin``), every socket takes a slot in a
bounded budget, frames are size- and rate-limited, subscriptions are capped and pruned, and
the status stream shares one telemetry sample.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from anyio.from_thread import start_blocking_portal
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from backend.src.api import status as status_api
from backend.src.api.routers import auth as auth_router
from backend.src.api.routers import telemetry as telemetry_router
from backend.src.api.routers.auth import _authorize_websocket, _origin_allowed, admit_websocket
from backend.src.core import ws_policy
from backend.src.core.peer_address import TRUSTED_PROXIES_ENV
from backend.src.core.ws_policy import (
    ANONYMOUS_TELEMETRY_ENV,
    MAX_TOPIC_LENGTH,
    MAX_TOPICS_PER_CLIENT,
    MessageBudget,
    WsLimitError,
    WsRegistry,
    anonymous_telemetry_mode,
    parse_anonymous_telemetry,
    valid_topic,
)
from backend.src.main import app
from backend.src.services.auth_service import AuthService
from backend.src.services.websocket_hub import WebSocketHub, websocket_hub

CRED = "correct-horse-battery-staple"
LOGIN = "/api/v2/auth/login"
TICKET = "/api/v2/auth/ws-ticket"
READ_ONLY = ("/ws/telemetry", "/api/v2/ws/telemetry", "/api/v2/ws/status")
CONTROL = "/api/v2/ws/control"
TELEMETRY = "/api/v2/ws/telemetry"
STATUS = "/api/v2/ws/status"


@pytest.fixture
def strict(monkeypatch):
    """The real app, operator auth ON, NO simulator opt-in, a fresh budget and a clean hub."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("SIM_MODE", "1")  # the telemetry source is the simulator, never hardware
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.delenv("LAWNBERRY_ENV", raising=False)
    monkeypatch.delenv(ANONYMOUS_TELEMETRY_ENV, raising=False)
    monkeypatch.delenv(TRUSTED_PROXIES_ENV, raising=False)
    monkeypatch.setenv("LAWN_BERRY_ALLOWED_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    service = AuthService()
    registry = WsRegistry()
    for module in (auth_router, telemetry_router, status_api):
        monkeypatch.setattr(module, "ws_registry", registry)
    monkeypatch.setattr(auth_router, "primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.middleware.rate_limiting.primary_auth_service", service)
    monkeypatch.setattr(auth_router, "_ws_tickets", {})
    # the hub is a module singleton: every test starts it empty, with fresh asyncio primitives
    # (a Lock binds to the first event loop that contends for it; each test has its own)
    monkeypatch.setattr(websocket_hub, "_telemetry_lock", asyncio.Lock())
    monkeypatch.setattr(websocket_hub, "_telemetry_cache", None)
    monkeypatch.setattr(websocket_hub, "_telemetry_task", None)
    monkeypatch.setattr(websocket_hub, "telemetry_cadence_hz", 5.0)
    websocket_hub.clients.clear()
    websocket_hub.subscriptions.clear()
    websocket_hub._client_topics.clear()
    yield registry
    websocket_hub.clients.clear()
    websocket_hub.subscriptions.clear()
    websocket_hub._client_topics.clear()


def _client(peer: str = "127.0.0.1") -> TestClient:
    return TestClient(app, client=(peer, 50000))


def _login(client: TestClient) -> dict[str, str]:
    resp = client.post(LOGIN, json={"credential": CRED})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _ticket(client: TestClient, headers: dict[str, str]) -> str:
    resp = client.post(TICKET, headers=headers, json={})
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _refused(client: TestClient, url: str, **kwargs) -> int:
    with pytest.raises(WebSocketDenialResponse) as refused:
        with client.websocket_connect(url, **kwargs):
            pass
    return refused.value.status_code


# ---- no anonymous admission, for ANY peer (the review's reproductions) ---------------------------


@pytest.mark.parametrize("url", READ_ONLY)
@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "127.0.0.2", "192.168.1.64", "203.0.113.9"])
def test_no_peer_is_admitted_without_a_credential_by_default(strict, url, peer):
    """SEC20-02 repro 1 and 4: the loopback (a tunnel agent, nginx) was ADMITTED with no ticket."""
    assert _refused(_client(peer), url) == 401


@pytest.mark.parametrize("peer", ["127.0.0.1", "203.0.113.9"])
def test_the_command_capable_socket_needs_a_credential_in_every_mode(strict, monkeypatch, peer):
    for mode in ("loopback", "any"):
        monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, mode)
        assert _refused(_client(peer), CONTROL) == 401, mode


def test_a_forwarded_for_header_does_not_make_a_peer_loopback(strict, monkeypatch):
    """A LAN client that writes ``X-Forwarded-For: 127.0.0.1`` is not the local browser."""
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "loopback")
    headers = {"X-Forwarded-For": "127.0.0.1"}
    assert _refused(_client("203.0.113.9"), TELEMETRY, headers=headers) == 401


def test_loopback_mode_is_only_as_safe_as_the_forwarding_setup(strict, monkeypatch):
    """What "loopback" can and cannot know. A same-host forwarder that is NOT declared makes every
    remote visitor behind it look like the local browser (documented hazard of the opt-in); once
    the forwarder is declared, the client it recorded is what is judged."""
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "loopback")
    remote_behind_forwarder = {"X-Forwarded-For": "203.0.113.9"}
    with _client("127.0.0.2").websocket_connect(TELEMETRY, headers=remote_behind_forwarder) as ws:
        assert ws.receive_text()  # undeclared: 127.0.0.2 is "the browser on the mower"
    monkeypatch.setenv(TRUSTED_PROXIES_ENV, "127.0.0.2=single")
    assert _refused(_client("127.0.0.2"), TELEMETRY, headers=remote_behind_forwarder) == 401
    local_browser = {"X-Forwarded-For": "127.0.0.1"}
    with _client("127.0.0.2").websocket_connect(TELEMETRY, headers=local_browser) as ws:
        assert ws.receive_text()


def test_a_valid_ticket_admits_a_remote_peer_once(strict):
    client = _client("203.0.113.9")
    ticket = _ticket(client, _login(client))
    with client.websocket_connect(f"{TELEMETRY}?ticket={ticket}") as ws:
        assert json.loads(ws.receive_text())["event"] == "connection.established"
    assert _refused(client, f"{TELEMETRY}?ticket={ticket}") == 401, "a ticket is single use"


def test_a_valid_bearer_admits_a_non_browser_client(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    for url in (TELEMETRY, "/ws/telemetry", CONTROL):
        with client.websocket_connect(url, headers=headers) as ws:
            assert json.loads(ws.receive_text())["event"] == "connection.established"


@pytest.mark.parametrize("mode", ["off", "loopback", "any"])
@pytest.mark.parametrize("peer", ["127.0.0.1", "203.0.113.9"])
def test_a_presented_bad_credential_is_always_refused(strict, monkeypatch, mode, peer):
    """The survivor of the review (A31): a bad bearer must never fall through to the anonymous
    path, not even from the loopback with the opt-in on."""
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, mode)
    client = _client(peer)
    bad = {"Authorization": "Bearer not-a-token"}
    assert _refused(client, TELEMETRY, headers=bad) == 401
    assert _refused(client, f"{TELEMETRY}?ticket=never-issued") == 401


def test_a_bad_bearer_is_refused_even_with_operator_auth_off(strict, monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    client = _client("127.0.0.1")
    assert _refused(client, TELEMETRY, headers={"Authorization": "Bearer nope"}) == 401
    # nothing presented: switching authentication off does not open the sockets either
    assert _refused(client, TELEMETRY) == 401
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "loopback")  # only the explicit opt-in does
    with client.websocket_connect(TELEMETRY) as ws:
        assert json.loads(ws.receive_text())["event"] == "connection.established"


def test_the_simulator_opt_in_still_opens_the_read_only_streams(strict, monkeypatch):
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
    client = _client("203.0.113.9")
    with client.websocket_connect(TELEMETRY) as ws:
        assert json.loads(ws.receive_text())["event"] == "connection.established"
    assert _refused(client, CONTROL) == 401, "the command-capable socket stays closed"
    monkeypatch.setenv("LAWNBERRY_ENV", "production")  # never honoured in production
    assert _refused(client, TELEMETRY) == 401


# ---- the explicit opt-in -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "mode", "recognised"),
    [
        (None, "off", True),
        ("", "off", True),
        ("off", "off", True),
        ("OFF", "off", True),
        ("0", "off", True),
        ("false", "off", True),
        ("no", "off", True),
        ("loopback", "loopback", True),
        (" Loopback ", "loopback", True),
        ("any", "any", True),
        ("ANY", "any", True),
        # an ambiguous value never opens anything (and is said to be wrong)
        ("1", "off", False),
        ("true", "off", False),
        ("yes", "off", False),
        ("on", "off", False),
        ("*", "off", False),
        ("anyone", "off", False),
        ("loopbak", "off", False),
    ],
)
def test_the_opt_in_is_parsed_fail_closed(value, mode, recognised):
    assert parse_anonymous_telemetry(value) == (mode, recognised)
    assert (
        anonymous_telemetry_mode({} if value is None else {ANONYMOUS_TELEMETRY_ENV: value}) == mode
    )


@pytest.mark.parametrize("url", READ_ONLY)
def test_loopback_mode_admits_only_loopback_clients_to_the_read_only_streams(
    strict, monkeypatch, url
):
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "loopback")
    with _client("127.0.0.1").websocket_connect(url) as ws:
        assert ws.receive_text()
    with _client("::1").websocket_connect(url) as ws:
        assert ws.receive_text()
    assert _refused(_client("203.0.113.9"), url) == 401
    assert _refused(_client("192.168.1.64"), url) == 401


@pytest.mark.parametrize("url", READ_ONLY)
def test_any_mode_admits_every_client_to_the_read_only_streams(strict, monkeypatch, url):
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "any")
    with _client("203.0.113.9").websocket_connect(url) as ws:
        assert ws.receive_text()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "typo"])
def test_an_ambiguous_opt_in_value_opens_nothing(strict, monkeypatch, value):
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, value)
    assert _refused(_client("127.0.0.1"), TELEMETRY) == 401


def test_anonymous_sockets_cannot_change_the_shared_cadence(strict, monkeypatch):
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "any")
    with _client("203.0.113.9").websocket_connect(TELEMETRY) as ws:
        ws.receive_text()
        ws.send_json({"type": "set_cadence", "cadence_hz": 10})
        ws.send_json({"type": "ping"})
        assert json.loads(ws.receive_text())["event"] == "pong"  # no cadence.updated before it
    assert websocket_hub.telemetry_cadence_hz == 5.0


def test_an_authenticated_socket_can_still_set_the_cadence(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        ws.receive_text()
        ws.send_json({"type": "set_cadence", "cadence_hz": 2})
        assert json.loads(ws.receive_text())["event"] == "cadence.updated"
    assert websocket_hub.telemetry_cadence_hz == 2.0


# ---- Origin (SEC20-14) ---------------------------------------------------------------------------


class FakeWS:
    def __init__(self, host="127.0.0.1", headers=None, query=None):
        self.client = (host, 4321)
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.query_params = query or {}


@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({}, True),  # not a browser: the credential rules decide
        ({"Origin": "http://localhost:3000"}, True),  # the allow-list
        ({"Origin": "HTTP://LOCALHOST:3000"}, True),
        ({"Origin": "https://evil.example"}, False),
        ({"Origin": "https://localhost:3000"}, False),  # other scheme: another origin
        ({"Origin": "http://localhost:3001"}, False),
        ({"Origin": "null"}, False),
        ({"Origin": ""}, False),
        ({"Origin": "not a url"}, False),
        ({"Origin": "http://localhost:3000:80"}, False),
        # same origin: the page's host:port is the request's Host
        ({"Origin": "http://192.168.1.50:3000", "Host": "192.168.1.50:3000"}, True),
        ({"Origin": "http://mower.local", "Host": "mower.local"}, True),
        ({"Origin": "http://mower.local:80", "Host": "mower.local"}, True),  # default port
        ({"Origin": "https://mower.local", "Host": "mower.local:443"}, True),
        ({"Origin": "https://evil.example", "Host": "192.168.1.50:3000"}, False),
        ({"Origin": "http://192.168.1.50:3000", "Host": "192.168.1.51:3000"}, False),
        # the UI server rewrites Host to the backend's: X-Forwarded-Host is believed ONLY from a
        # declared forwarder (not declared here)
        (
            {
                "Origin": "http://192.168.1.50:3000",
                "Host": "127.0.0.1:8081",
                "X-Forwarded-Host": "192.168.1.50:3000",
            },
            False,
        ),
    ],
)
def test_origin_rule(strict, headers, allowed):
    assert _origin_allowed(FakeWS(headers=headers)) is allowed


def test_x_forwarded_host_counts_only_from_a_declared_forwarder(strict, monkeypatch):
    headers = {
        "Origin": "http://192.168.1.50:3000",
        "Host": "127.0.0.1:8081",
        "X-Forwarded-Host": "evil.example, 192.168.1.50:3000",
    }
    assert _origin_allowed(FakeWS(host="127.0.0.2", headers=headers)) is False
    monkeypatch.setenv(TRUSTED_PROXIES_ENV, "127.0.0.2=single")
    assert _origin_allowed(FakeWS(host="127.0.0.2", headers=headers)) is True  # last hop
    assert _origin_allowed(FakeWS(host="127.0.0.9", headers=headers)) is False


@pytest.mark.parametrize("mode", ["loopback", "any"])
def test_a_cross_origin_page_cannot_read_the_stream_anonymously(strict, monkeypatch, mode):
    """SEC20-14 repro: loopback peer, ``Origin: https://evil.example``, no credential."""
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, mode)
    client = _client("127.0.0.1")
    for url in READ_ONLY:
        assert _refused(client, url, headers={"Origin": "https://evil.example"}) == 401, url
        assert _refused(client, url, headers={"Origin": "null"}) == 401, url
    with client.websocket_connect(TELEMETRY, headers={"Origin": "http://localhost:3000"}) as ws:
        assert ws.receive_text()  # the UI's own origin still works


# ---- the admission helper's own contract ---------------------------------------------------------


async def test_the_session_of_a_ticket_must_still_be_live(strict):
    """Survivor A40: a ticket minted for a session that has since expired must not redeem."""
    result = await auth_router.primary_auth_service.authenticate(CRED, client_ip="203.0.113.9")
    ticket = auth_router._mint_ws_ticket(result.session.session_id)
    result.session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(FakeWS(host="203.0.113.9", query={"ticket": ticket}))
    assert refused.value.status_code == 401


async def test_the_ticket_lifetime_is_30_seconds(strict):
    """Survivor A11: the TTL is pinned, and an older ticket is dead."""
    assert auth_router.WS_TICKET_TTL_SECONDS == 30
    result = await auth_router.primary_auth_service.authenticate(CRED, client_ip="203.0.113.9")
    ticket = auth_router._mint_ws_ticket(result.session.session_id)
    entry = auth_router._ws_tickets[ticket]
    lifetime = (entry["expires_at"] - datetime.now(UTC)).total_seconds()
    assert 29 <= lifetime <= 30
    entry["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(HTTPException):
        await _authorize_websocket(FakeWS(host="203.0.113.9", query={"ticket": ticket}))


async def test_admit_websocket_takes_a_slot_and_a_full_budget_is_a_503(strict):
    strict.max_total = 1
    result = await auth_router.primary_auth_service.authenticate(CRED, client_ip="203.0.113.9")
    first = await admit_websocket(
        FakeWS(
            host="203.0.113.9",
            query={"ticket": auth_router._mint_ws_ticket(result.session.session_id)},
        )
    )
    assert not first.anonymous and strict.snapshot()["open"] == 1
    second_ticket = auth_router._mint_ws_ticket(result.session.session_id)
    with pytest.raises(HTTPException) as refused:
        await admit_websocket(FakeWS(host="203.0.113.9", query={"ticket": second_ticket}))
    assert refused.value.status_code == 503
    assert refused.value.headers == {"Retry-After": "5"}
    assert strict.snapshot()["refused"] == {"total": 1}
    strict.release(first.slot)
    assert strict.snapshot()["open"] == 0


# ---- the budget ----------------------------------------------------------------------------------


def test_registry_counts_per_session_and_in_total():
    reg = WsRegistry(max_total=4, max_per_session=2, max_anonymous=2, max_anonymous_per_peer=1)
    a1 = reg.acquire(session_id="a", peer="ip:1", anonymous=False)
    a2 = reg.acquire(session_id="a", peer="ip:1", anonymous=False)
    with pytest.raises(WsLimitError) as err:
        reg.acquire(session_id="a", peer="ip:1", anonymous=False)
    assert err.value.scope == "session"
    b1 = reg.acquire(session_id="b", peer="ip:1", anonymous=False)  # another login is unaffected
    b2 = reg.acquire(session_id="b", peer="ip:1", anonymous=False)
    with pytest.raises(WsLimitError) as err:
        reg.acquire(session_id="c", peer="ip:2", anonymous=False)
    assert err.value.scope == "total"
    reg.release(a1)
    reg.release(a1)  # idempotent: a handler's finally may run twice
    assert reg.snapshot()["open"] == 3
    c1 = reg.acquire(session_id="c", peer="ip:2", anonymous=False)
    assert reg.snapshot()["open"] == 4
    for slot in (a2, b1, b2, c1):
        reg.release(slot)
    assert reg.snapshot() == {
        "open": 0,
        "anonymous": 0,
        "sessions": 0,
        "max_total": 4,
        "refused": {"session": 1, "total": 1},
    }


def test_registry_gives_anonymous_sockets_a_small_budget_of_their_own():
    reg = WsRegistry(max_total=10, max_per_session=5, max_anonymous=2, max_anonymous_per_peer=1)
    x = reg.acquire(session_id=None, peer="ip:9", anonymous=True)
    with pytest.raises(WsLimitError) as err:
        reg.acquire(session_id=None, peer="ip:9", anonymous=True)
    assert err.value.scope == "anonymous-peer"
    reg.acquire(session_id=None, peer="ip:8", anonymous=True)
    with pytest.raises(WsLimitError) as err:
        reg.acquire(session_id=None, peer="ip:7", anonymous=True)
    assert err.value.scope == "anonymous"
    for _ in range(5):  # authenticated sockets are not starved by anonymous ones
        reg.acquire(session_id="s", peer="ip:9", anonymous=False)
    reg.release(x)
    reg.acquire(session_id=None, peer="ip:7", anonymous=True)


def test_registry_state_does_not_leak_released_keys():
    reg = WsRegistry(
        max_total=100, max_per_session=10, max_anonymous=100, max_anonymous_per_peer=100
    )
    slots = [
        reg.acquire(session_id=f"s{n}", peer=f"ip:{n}", anonymous=n % 2 == 0) for n in range(20)
    ]
    for slot in slots:
        reg.release(slot)
    assert reg._by_session == {} and reg._anonymous_by_peer == {} and reg._total == 0


def test_the_shipped_limits_are_small_and_ordered():
    assert 0 < ws_policy.MAX_ANONYMOUS_WS_PER_PEER <= ws_policy.MAX_ANONYMOUS_WS
    assert ws_policy.MAX_ANONYMOUS_WS < ws_policy.MAX_WS_PER_SESSION <= ws_policy.MAX_WS_CLIENTS
    assert ws_policy.MAX_WS_CLIENTS <= 64  # the unit's descriptor and memory budget


def test_a_login_cannot_open_more_sockets_than_its_budget(strict):
    strict.max_per_session = 2
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers):
        with client.websocket_connect(TELEMETRY, headers=headers):
            assert strict.snapshot()["open"] == 2
            assert _refused(client, TELEMETRY, headers=headers) == 503
            assert _refused(client, STATUS, headers=headers) == 503
            assert _refused(client, CONTROL, headers=headers) == 503
    assert strict.snapshot()["open"] == 0, "closing must hand the slots back"
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        assert ws.receive_text()


def test_anonymous_sockets_have_their_own_small_budget_end_to_end(strict, monkeypatch):
    monkeypatch.setenv(ANONYMOUS_TELEMETRY_ENV, "any")
    strict.max_anonymous_per_peer = 1
    client = _client("203.0.113.9")
    with client.websocket_connect(TELEMETRY):
        assert _refused(client, TELEMETRY) == 503
        assert _refused(client, STATUS) == 503
        with _client("203.0.113.10").websocket_connect(TELEMETRY):  # another client is fine
            assert strict.snapshot()["anonymous"] == 2
        headers = _login(client)  # and a logged-in operator from the same address is untouched
        with client.websocket_connect(TELEMETRY, headers=headers) as ws:
            assert ws.receive_text()
    assert strict.snapshot()["open"] == 0


def test_a_refused_socket_leaves_no_slot_behind(strict):
    for _ in range(5):
        assert _refused(_client("203.0.113.9"), TELEMETRY) == 401
    assert strict.snapshot()["open"] == 0


# ---- frames: size, rate, topics ------------------------------------------------------------


def _frame(length: int) -> str:
    """A valid ``ping`` frame of exactly ``length`` characters."""
    prefix, suffix = '{"type":"ping","pad":"', '"}'
    return prefix + "x" * (length - len(prefix) - len(suffix)) + suffix


def _answers_until_closed(ws, frames: int) -> tuple[int, int | None]:
    """``(answers seen, close code or None)`` for ``frames`` frames just sent.

    NEVER waits for ever: the socket answers each frame it handles with exactly one frame, so
    after ``frames`` answers with no close there is nothing left to wait for. The loops that used
    to read "until the server closes" HUNG the suite for 900 s when the limit stopped working
    (reviewer's mutants W5, rate 20 -> 20000 Hz, and W9, a budget that never refuses): now such a
    mutant fails in a moment.
    """
    seen = 0
    for _ in range(frames):
        try:
            ws.receive_text()
        except WebSocketDisconnect as closed:
            return seen, closed.code
        seen += 1
    return seen, None


def test_an_oversized_frame_ends_the_connection(strict):
    """The ceiling is 4096 characters, written as a LITERAL: a test that sends
    ``MAX_MESSAGE_BYTES + 1`` follows a mutant that raises the constant (W4)."""
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        ws.receive_text()
        ws.send_text(_frame(4096))
        assert json.loads(ws.receive_text())["event"] == "pong", "4096 is within the ceiling"
        ws.send_text(_frame(4097))
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()  # a limit that does not hold answers "pong" here: no hang
        assert closed.value.code == 1009
    assert strict.snapshot()["open"] == 0


def test_a_frame_flood_ends_the_connection_and_creates_no_unbounded_state(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        ws.receive_text()
        for n in range(200):
            ws.send_json({"type": "subscribe", "topic": f"flood.{n}"})
        seen, code = _answers_until_closed(ws, 200)
        assert code == 1008, f"the frame rate limit never ended the socket ({seen} answers)"
        assert 30 <= seen <= 60, "burst 40 at 20 a second, not 200 frames"
    assert websocket_hub.subscriptions == {}, "a closed socket leaves no topics behind"
    assert strict.snapshot()["open"] == 0


def test_the_control_socket_acks_but_not_a_flood(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(CONTROL, headers=headers) as ws:
        assert json.loads(ws.receive_text())["event"] == "connection.established"
        ws.send_text("{}")
        assert json.loads(ws.receive_text())["event"] == "ack"
        for _ in range(200):
            ws.send_text("{}")
        seen, code = _answers_until_closed(ws, 200)
        assert code == 1008, f"the frame rate limit never ended the socket ({seen} acks)"
        assert 30 <= seen <= 60


def test_distinct_one_kilobyte_topics_create_nothing(strict):
    """SEC20-02 repro 5, scaled down: each frame names a new 1 kB topic. The old hub made one per
    frame (and never forgot it); now each is refused and nothing is stored."""
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        ws.receive_text()
        refused = 0
        for n in range(30):
            ws.send_json({"type": "subscribe", "topic": f"{n}-" + "t" * 1000})
        for _ in range(30):
            event = json.loads(ws.receive_text())
            refused += event["event"] == "subscription.refused"
            assert len(event.get("topic", "")) <= 64, "the refusal does not echo a 1 kB name"
        assert refused == 30
        assert websocket_hub.subscriptions == {}
    assert websocket_hub._client_topics == {}


def test_a_client_holds_at_most_the_topic_limit(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        ws.receive_text()
        for n in range(MAX_TOPICS_PER_CLIENT + 1):
            ws.send_json({"type": "subscribe", "topic": f"t.{n}"})
        events = [json.loads(ws.receive_text()) for _ in range(MAX_TOPICS_PER_CLIENT + 1)]
        assert [e["event"] for e in events[:-1]] == [
            "subscription.confirmed"
        ] * MAX_TOPICS_PER_CLIENT
        assert events[-1]["event"] == "subscription.refused"
        assert events[-1]["reason"] == "too many subscriptions"
        ws.send_json({"type": "subscribe", "topic": "t.0"})  # re-subscribing is not a new topic
        assert json.loads(ws.receive_text())["event"] == "subscription.confirmed"
        assert len(websocket_hub.subscriptions) == MAX_TOPICS_PER_CLIENT
    assert websocket_hub.subscriptions == {}


def test_list_topics_shows_only_live_topics(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(TELEMETRY, headers=headers) as ws:
        ws.receive_text()
        ws.send_json({"type": "subscribe", "topic": "telemetry.power"})
        ws.receive_text()
        ws.send_json({"type": "subscribe", "topic": "telemetry.tof"})
        ws.receive_text()
        ws.send_json({"type": "unsubscribe", "topic": "telemetry.tof"})
        ws.receive_text()
        ws.send_json({"type": "list_topics"})
        assert json.loads(ws.receive_text())["topics"] == ["telemetry.power"]


# ---- the hub itself ------------------------------------------------------------------------------


class _Sink:
    def __init__(self):
        self.sent: list[str] = []

    async def send_text(self, message):
        self.sent.append(message)


class _Stuck:
    """A client that stopped reading: its send never completes."""

    def __init__(self):
        self.closed: list[int] = []

    async def send_text(self, message):
        await asyncio.Event().wait()

    async def close(self, code=1000):
        self.closed.append(code)


@pytest.mark.parametrize(
    ("topic", "ok"),
    [
        ("telemetry.power", True),
        ("telemetry/updates", True),
        ("system.safety", True),
        ("jobs.progress", True),
        ("a", True),
        ("a" * MAX_TOPIC_LENGTH, True),
        ("a" * (MAX_TOPIC_LENGTH + 1), False),
        ("", False),
        (" telemetry", False),
        ("tele metry", False),
        ("tele\nmetry", False),
        (".hidden", False),
        ("-dash", False),
        ("emoji\U0001f600", False),
        ("x\x00y", False),
        (None, False),
        (7, False),
        (b"telemetry.power", False),
    ],
)
def test_valid_topic(topic, ok):
    assert valid_topic(topic) is ok


async def test_a_thousand_distinct_topics_from_one_client_are_capped_and_pruned():
    hub = WebSocketHub()
    sink = _Sink()
    hub.clients["c1"] = sink
    for n in range(1000):
        await hub.subscribe("c1", f"topic.{n}")
    assert len(hub.subscriptions) == MAX_TOPICS_PER_CLIENT
    hub.disconnect("c1")
    assert hub.subscriptions == {} and hub._client_topics == {} and hub.clients == {}


async def test_unsubscribe_and_disconnect_prune_empty_topics():
    hub = WebSocketHub()
    hub.clients.update({"a": _Sink(), "b": _Sink()})
    await hub.subscribe("a", "t.one")
    await hub.subscribe("b", "t.one")
    await hub.subscribe("b", "t.two")
    await hub.unsubscribe("a", "t.one")
    assert set(hub.subscriptions) == {"t.one", "t.two"}
    await hub.unsubscribe("b", "t.one")
    assert set(hub.subscriptions) == {"t.two"}
    hub.disconnect("b")
    assert hub.subscriptions == {}


async def test_a_client_that_stops_reading_is_dropped_and_does_not_stall_the_others(monkeypatch):
    monkeypatch.setattr("backend.src.services.websocket_hub.SEND_TIMEOUT_S", 0.05)
    hub = WebSocketHub()
    good, stuck = _Sink(), _Stuck()
    hub.clients.update({"good": good, "stuck": stuck})
    for client_id in ("good", "stuck"):
        hub.subscriptions.setdefault("t", set()).add(client_id)
    started = time.monotonic()
    await hub.broadcast_to_topic("t", {"x": 1})
    assert time.monotonic() - started < 1.0, "one stuck client stalled the fan-out"
    assert len(good.sent) == 1
    assert "stuck" not in hub.clients, "the stuck client is dropped"
    await asyncio.sleep(0.05)
    assert stuck.closed == [1013], "and its socket is closed (try again later)"
    await hub.broadcast_to_topic("t", {"x": 2})
    assert len(good.sent) == 2


async def test_broadcast_survives_a_client_leaving_during_the_fan_out():
    """The old loop iterated the live subscriber set across awaits: a client that left meanwhile
    raised ``Set changed size during iteration`` into the telemetry loop."""
    hub = WebSocketHub()

    class Leaver(_Sink):
        async def send_text(self, message):
            hub.subscriptions["t"].discard("other")
            await super().send_text(message)

    hub.clients.update({"leaver": Leaver(), "other": _Sink(), "third": _Sink()})
    hub.subscriptions["t"] = {"leaver", "other", "third"}
    await hub.broadcast_to_topic("t", {"x": 1})  # must not raise


async def test_a_failing_client_is_disconnected_by_broadcast():
    hub = WebSocketHub()

    class Broken:
        async def send_text(self, message):
            raise ConnectionError("gone")

    hub.clients.update({"broken": Broken(), "fine": _Sink()})
    await hub.broadcast("hello")
    assert set(hub.clients) == {"fine"}


async def test_set_cadence_ignores_nan_and_clamps():
    hub = WebSocketHub()
    hub.clients["c"] = _Sink()
    await hub.set_cadence("c", float("nan"))
    assert hub.telemetry_cadence_hz == 5.0
    await hub.set_cadence("c", 1000.0)
    assert hub.telemetry_cadence_hz == 10.0
    await hub.set_cadence("c", -3.0)
    assert hub.telemetry_cadence_hz == 1.0


# ---- one telemetry sample for every status socket -----------------------------------------


async def test_concurrent_readers_share_one_sensor_read(monkeypatch):
    hub = WebSocketHub()
    calls = 0

    async def generate():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"n": calls}

    monkeypatch.setattr(hub, "_generate_telemetry", generate)
    samples = await asyncio.gather(*(hub.current_telemetry(max_age_s=0.5) for _ in range(12)))
    assert calls == 1, "twelve status sockets used to mean twelve full sensor reads"
    assert all(s is samples[0] for s in samples)
    await asyncio.sleep(0.01)
    assert (await hub.current_telemetry(max_age_s=0.5)) is samples[0]  # still fresh: cached
    assert calls == 1
    again = await hub.current_telemetry(max_age_s=0.0)  # stale by definition: one more read
    assert calls == 2 and again is not samples[0]


async def test_the_telemetry_loop_feeds_the_same_cache(monkeypatch):
    hub = WebSocketHub()

    async def generate():
        return {"from": "loop"}

    monkeypatch.setattr(hub, "_generate_telemetry", generate)
    monkeypatch.setattr(hub, "_broadcast_telemetry_topics", lambda data: asyncio.sleep(0))
    await hub.start_telemetry_loop()
    await asyncio.sleep(0.05)
    await hub.stop_telemetry_loop()
    assert (await hub.current_telemetry(max_age_s=60.0)) == {"from": "loop"}


def test_status_sockets_do_not_each_read_the_sensors(strict, monkeypatch):
    """SEC20-02 repro 5: each ``/api/v2/ws/status`` socket ran its own sensor read per tick."""
    calls = 0
    original = websocket_hub._generate_telemetry

    async def counting():
        nonlocal calls
        calls += 1
        return await original()

    monkeypatch.setattr(websocket_hub, "_generate_telemetry", counting)
    client = _client("203.0.113.9")
    headers = _login(client)
    sockets = 4
    frames = [0] * sockets
    # ONE event loop for every connection, like the real server (the TestClient makes one per
    # connection otherwise, and the hub's asyncio lock belongs to a single loop)
    with start_blocking_portal() as portal:
        client.portal = portal
        stack = []
        try:
            for _ in range(sockets):
                ctx = client.websocket_connect(STATUS, headers=headers)
                stack.append((ctx, ctx.__enter__()))
            deadline = time.monotonic() + 1.3
            while time.monotonic() < deadline:
                for i, (_ctx, ws) in enumerate(stack):
                    ws.receive_json()
                    frames[i] += 1
        finally:
            for ctx, _ws in reversed(stack):
                ctx.__exit__(None, None, None)
    ticks = max(frames)
    assert min(frames) >= 2, frames
    # one sample per tick, shared; the old handler read the sensors once per socket per tick
    assert calls <= ticks + 3, (calls, frames)
    assert calls < sockets * ticks / 2, (calls, frames)


def test_the_status_stream_hands_its_slot_back(strict):
    client = _client("203.0.113.9")
    headers = _login(client)
    with client.websocket_connect(STATUS, headers=headers) as ws:
        assert "battery_percentage" in ws.receive_json()
        assert strict.snapshot()["open"] == 1
    assert strict.snapshot()["open"] == 0


# ---- the per-socket frame budget -----------------------------------------------------------------


def test_message_budget_allows_a_burst_then_the_sustained_rate():
    now = [0.0]
    budget = MessageBudget(rate_hz=10.0, burst=5, clock=lambda: now[0])
    assert [budget.allow() for _ in range(7)] == [True] * 5 + [False] * 2
    now[0] += 0.25  # 2.5 tokens
    assert [budget.allow() for _ in range(3)] == [True, True, False]
    now[0] += 100.0  # never more than the burst
    assert [budget.allow() for _ in range(7)] == [True] * 5 + [False] * 2


def test_a_reconnect_resubscribe_burst_fits_the_budget():
    """The UI re-subscribes ~12 topics in a burst after every reconnect."""
    budget = MessageBudget(clock=lambda: 0.0)
    assert all(budget.allow() for _ in range(15))
