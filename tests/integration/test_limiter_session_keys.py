"""Logged-in traffic has its own rate-limit bucket (review SEC20-02, reproduction 3).

Every client behind a same-host forwarder that is not told apart is one address, so one anonymous
client shared its bucket with the operator, and the limiter runs BEFORE the route's authentication:
130 anonymous junk ``POST /tractor/left-lever`` answered ``{401: 96, 429: 34}``, and the
operator's authenticated lever POST right after was ``429`` too.

Now a request that carries a bearer the auth service recognises as a LIVE session is counted as
``session:<id>``; everything else (no bearer, junk, expired, revoked) as its client address.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import FastAPI

from backend.src.core.peer_address import TRUSTED_PROXIES_ENV
from backend.src.middleware.rate_limiting import limiter_client_key, register_global_rate_limiter
from backend.src.models import UserRole
from backend.src.services.auth_service import AuthService

CRED = "correct-horse-battery-staple"
LEVER = "/api/v2/tractor/left-lever"
POLL = "/api/v2/poll"
ESTOP = "/api/v2/tractor/emergency-stop"


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.delenv(TRUSTED_PROXIES_ENV, raising=False)
    svc = AuthService()
    monkeypatch.setattr("backend.src.middleware.rate_limiting.primary_auth_service", svc)
    return svc


def _app(monkeypatch, *, rate="0.1", burst="5") -> FastAPI:
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_RATE", rate)
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_BURST", burst)
    monkeypatch.delenv("GLOBAL_RATE_LIMIT_OVERRIDES", raising=False)
    monkeypatch.delenv("GLOBAL_RATE_LIMIT_EXEMPT", raising=False)
    sub = FastAPI()
    register_global_rate_limiter(sub)

    @sub.post(LEVER)
    def lever() -> dict:
        return {"ok": True}

    @sub.get(POLL)
    def poll() -> dict:
        return {"ok": True}

    @sub.post(ESTOP)
    def estop() -> dict:
        return {"stopped": True}

    return sub


def _client(app: FastAPI, peer: str = "127.0.0.1") -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(peer, 40000)), base_url="http://t"
    )


async def _login(service: AuthService) -> tuple[str, str]:
    result = await service.authenticate(CRED, client_ip="127.0.0.1")
    return result.token, result.session.session_id


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_the_operator_is_not_starved_by_anonymous_junk_from_the_same_address(
    service, monkeypatch
):
    """The review's repro: ~100 anonymous POSTs drain the shared bucket; the operator's own
    authenticated POST must still be answered."""
    token, _ = await _login(service)
    async with _client(_app(monkeypatch)) as client:
        # the lever POSTs have a bucket of their own (burst 90, 30/s): drain it
        junk = [(await client.post(LEVER, json={})).status_code for _ in range(200)]
        assert junk.count(429) >= 50, "the anonymous junk must have drained its own bucket"
        operator = await client.post(LEVER, json={}, headers=_bearer(token))
        assert operator.status_code == 200, "the operator was starved by someone else's traffic"


async def test_the_operators_own_bucket_still_limits_the_operator(service, monkeypatch):
    token, _ = await _login(service)
    async with _client(_app(monkeypatch, burst="3")) as client:
        codes = [(await client.get(POLL, headers=_bearer(token))).status_code for _ in range(8)]
    assert codes.count(200) == 3 and codes.count(429) == 5, codes


async def test_two_logins_do_not_share_a_bucket(service, monkeypatch):
    first, _ = await _login(service)
    second, _ = await _login(service)
    async with _client(_app(monkeypatch, burst="2")) as client:
        a = [(await client.get(POLL, headers=_bearer(first))).status_code for _ in range(5)]
        b = [(await client.get(POLL, headers=_bearer(second))).status_code for _ in range(2)]
    assert a.count(429) == 3 and b == [200, 200], (a, b)


@pytest.mark.parametrize("kind", ["junk", "wrong-signature", "expired", "revoked"])
async def test_a_bearer_that_is_not_a_live_session_is_counted_as_the_client_address(
    service, monkeypatch, kind
):
    """A client cannot pick its bucket by what it sends: only a live session earns its own."""
    token, session_id = await _login(service)
    if kind == "junk":
        sent = "not-a-token"
    elif kind == "wrong-signature":
        other = AuthService()  # another signing key
        other.jwt_manager.secret_key = "another-secret-another-secret-another-secret"
        sent = other.jwt_manager.create_token(
            session_id=session_id, user_id="operator", role=UserRole.OPERATOR
        )[0]
    elif kind == "expired":
        service.active_sessions[session_id].expires_at = service.active_sessions[
            session_id
        ].created_at
        sent = token
    else:
        await service.terminate_session(session_id)
        sent = token
    async with _client(_app(monkeypatch, burst="3")) as client:
        # five requests, each with a DIFFERENT dead bearer, from one address: one shared bucket
        codes = []
        for n in range(5):
            dead = sent if kind != "junk" else f"junk-{n}"
            codes.append((await client.get(POLL, headers=_bearer(dead))).status_code)
        anonymous = (await client.get(POLL)).status_code
    assert codes.count(200) == 3 and codes.count(429) == 2, codes
    assert anonymous == 429, "a dead bearer is exactly as anonymous as no bearer"


async def test_junk_bearers_do_not_flood_the_log(service, monkeypatch, caplog):
    """The limiter classifies every request: a flood of junk tokens must not be a flood of lines
    (``verify_token`` warns about each; the limiter uses the silent ``peek_session``)."""
    async with _client(_app(monkeypatch, burst="1000")) as client:
        with caplog.at_level(logging.DEBUG):
            for n in range(25):
                await client.get(POLL, headers=_bearer(f"junk-{n}"))
    noisy = [r for r in caplog.records if "JWT" in r.getMessage() or "jwt" in r.name.lower()]
    assert noisy == []


async def test_the_emergency_stop_is_never_limited_whoever_calls_it(service, monkeypatch):
    async with _client(_app(monkeypatch, burst="1")) as client:
        for _ in range(10):
            assert (await client.get(POLL)).status_code in (200, 429)
        assert [(await client.post(ESTOP, json={})).status_code for _ in range(10)] == [200] * 10


async def test_limiter_client_key_names_sessions_and_addresses(service):
    from starlette.requests import Request

    token, session_id = await _login(service)

    def request(auth: str | None, peer: str = "203.0.113.9") -> Request:
        headers = [(b"authorization", auth.encode())] if auth else []
        return Request(
            {"type": "http", "method": "GET", "path": "/", "headers": headers, "client": (peer, 1)}
        )

    assert limiter_client_key(request(f"Bearer {token}")) == f"session:{session_id}"
    assert limiter_client_key(request(f"bearer {token}")) == f"session:{session_id}"  # any case
    assert limiter_client_key(request(f"Basic {token}")) == "ip:203.0.113.9"
    assert limiter_client_key(request("Bearer ")) == "ip:203.0.113.9"
    assert limiter_client_key(request("Bearer nope")) == "ip:203.0.113.9"
    assert limiter_client_key(request(None)) == "ip:203.0.113.9"
    assert limiter_client_key(request(None, peer="2001:db8:1:2::9")) == "ip6:2001:db8:1:2::/64"
