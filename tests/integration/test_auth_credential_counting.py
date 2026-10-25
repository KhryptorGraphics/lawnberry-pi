"""Only failed credential checks count (SEC20-08); body-less logout works (SEC20-06); SEC20-14.

SEC20-08: ``SecurityMiddleware`` counted EVERY 401 under ``/api/*/auth/*`` as a failed login. In
memory sessions die with the backend, so after a restart every open browser sends ``refresh``,
``logout`` and ``profile`` with a dead token (a 401 each, six requests in all) and the correct
operator login right after was answered 429. A 200 on any of those routes also reset a guesser's
failure counters. Now the limiter counts the login routes only.

SEC20-06: the UI's logout and refresh are body-less POSTs (axios sends no ``Content-Type`` without
data) and the middleware answered 415, so a "logout" never revoked the token or its WebSocket
tickets.

SEC20-14 (browser-origin controls and attribution): the caller picks the correlation id that is
written to every log line and echoed in a header; CORS named two external origins with
credentials by default.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient, WebSocketDenialResponse

from backend.src.api.routers import auth as auth_router
from backend.src.core.context import MAX_CORRELATION_ID_LENGTH, sanitize_correlation_id
from backend.src.main import app
from backend.src.middleware.security import (
    UNLIMITED_AUTH_PATHS,
    SecurityMiddleware,
    _default_allowed_origins,
)
from backend.src.services.auth_service import AuthService
from tests.integration._route_walk import routes_of

LOGIN = "/api/v2/auth/login"
REFRESH = "/api/v2/auth/refresh"
LOGOUT = "/api/v2/auth/logout"
PROFILE = "/api/v2/auth/profile"
TICKET = "/api/v2/auth/ws-ticket"
CREDENTIAL_CHECKS = frozenset({"/api/v2/auth/login", "/api/v1/auth/login"})
DEAD = {"Authorization": "Bearer expired.or.bogus"}
LAN = ("203.0.113.9", 4321)


@pytest.fixture(autouse=True)
def strict(monkeypatch, operator_credential):
    """Operator auth ON and a login that needs the real credential.

    The module-level ``primary_auth_service`` fixed its simulator flag when it was built, so (as
    ``test_ws_tickets_and_proxy_peer.py`` does) a fresh strict service replaces it everywhere.
    """
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "0")
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", operator_credential)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService()
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setattr(auth_router, "primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.rest_v1._auth_service", service)
    return service


def _limited_auth_app() -> FastAPI:
    """The REAL auth routes behind a REAL ``SecurityMiddleware`` with tight limits."""
    sub = FastAPI()
    sub.include_router(auth_router.router, prefix="/api/v2")
    sub.add_middleware(
        SecurityMiddleware,
        rate_limit_window_seconds=60,
        rate_limit_max_attempts=3,
        lockout_failures=3,
        lockout_seconds=60,
    )
    return sub


def _client(target, peer=LAN) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=target, client=peer), base_url="http://test"
    )


async def _login(client, credential) -> dict[str, str]:
    response = await client.post(LOGIN, json={"credential": credential})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def _wrong(client, count=1) -> list[int]:
    return [
        (await client.post(LOGIN, json={"credential": f"wrong-{n}"})).status_code
        for n in range(count)
    ]


# ---- SEC20-08 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", PROFILE), ("POST", REFRESH), ("POST", LOGOUT), ("POST", TICKET)],
    ids=["profile", "refresh", "logout", "ws-ticket"],
)
async def test_a_dead_token_on_a_token_route_never_locks_the_login_out(
    method, path, operator_credential
):
    async with _client(_limited_auth_app()) as client:
        codes = [
            (
                await client.request(
                    method, path, headers=DEAD, json={} if method == "POST" else None
                )
            ).status_code
            for _ in range(8)
        ]
        assert codes == [401] * 8, f"{method} {path}: {codes}"  # never 429 either
        await _login(client, operator_credential)  # the correct login is still open


async def test_the_whole_restart_scenario_leaves_the_correct_login_open(operator_credential):
    """A browser wakes up after a backend restart: refresh -> logout -> profile, all with a dead
    token (the UI's own sequence), twice over, then the operator signs in again."""
    async with _client(_limited_auth_app()) as client:
        for _ in range(2):
            assert (await client.post(REFRESH, headers=DEAD)).status_code == 401
            assert (await client.post(LOGOUT, headers=DEAD)).status_code == 401
            assert (await client.get(PROFILE, headers=DEAD)).status_code == 401
        await _login(client, operator_credential)


async def test_wrong_passwords_still_lock_the_login_out(operator_credential):
    async with _client(_limited_auth_app()) as client:
        assert await _wrong(client, 3) == [401, 401, 401]
        locked = await client.post(LOGIN, json={"credential": operator_credential})
        assert locked.status_code == 429
        assert int(locked.headers["Retry-After"]) > 0


async def test_another_clients_failures_never_block_a_token_holder(operator_credential):
    """The locked-out peer (behind a shared proxy that is every browser) can still use a token."""
    async with _client(_limited_auth_app()) as client:
        auth = await _login(client, operator_credential)
        assert await _wrong(client, 3) == [401, 401, 401]
        assert (
            await client.post(LOGIN, json={"credential": operator_credential})
        ).status_code == 429
        assert (await client.get(PROFILE, headers=auth)).status_code == 200
        assert (await client.post(TICKET, headers=auth, json={})).status_code == 200
        assert (await client.post(REFRESH, headers=auth, json={})).status_code == 200


async def test_a_success_on_a_token_route_does_not_reset_a_guessers_failure_count(
    operator_credential,
):
    async with _client(_limited_auth_app()) as client:
        auth = await _login(client, operator_credential)  # the guesser holds one valid token
        assert await _wrong(client, 2) == [401, 401]
        assert (await client.get(PROFILE, headers=auth)).status_code == 200  # used to reset it
        assert await _wrong(client, 1) == [401]  # the third failure
        locked = await client.post(LOGIN, json={"credential": operator_credential})
        assert locked.status_code == 429


async def test_a_successful_login_does_reset_the_failure_count(operator_credential):
    async with _client(_limited_auth_app()) as client:
        assert await _wrong(client, 2) == [401, 401]
        await _login(client, operator_credential)
        assert await _wrong(client, 2) == [401, 401]
        await _login(client, operator_credential)  # two since the reset: not locked


def test_every_route_under_the_auth_prefixes_is_classified():
    """A new route under ``/api/*/auth`` is limited and counted until it is listed as token-only:
    this forces the decision (credential check or not) instead of letting it default."""
    under = {
        info.path
        for info in routes_of(app)
        if info.kind in {"api", "route"} and info.path.startswith(("/api/v1/auth", "/api/v2/auth"))
    }
    assert under == CREDENTIAL_CHECKS | UNLIMITED_AUTH_PATHS, sorted(
        under ^ (CREDENTIAL_CHECKS | UNLIMITED_AUTH_PATHS)
    )
    assert not CREDENTIAL_CHECKS & UNLIMITED_AUTH_PATHS


# ---- SEC20-06 ----------------------------------------------------------------------------------


async def test_a_body_less_logout_revokes_the_session_and_its_tickets(operator_credential):
    """The UI's logout: POST with no body and no Content-Type (it got 415 and revoked nothing)."""
    client = TestClient(app, client=LAN)
    login = client.post(LOGIN, json={"credential": operator_credential})
    assert login.status_code == 200, login.text
    auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
    ticket = client.post(TICKET, headers=auth, json={}).json()["ticket"]
    assert client.get(PROFILE, headers=auth).status_code == 200

    out = client.post(LOGOUT, headers=auth)  # no data: no Content-Type, Content-Length 0
    assert out.status_code == 200, (out.status_code, out.text)
    assert out.json() == {"ok": True}

    assert client.get(PROFILE, headers=auth).status_code == 401, "the token must be dead"
    assert client.post(TICKET, headers=auth, json={}).status_code == 401
    with pytest.raises(WebSocketDenialResponse) as refused:  # a ticket minted before the logout
        with client.websocket_connect(f"/api/v2/ws/control?ticket={ticket}"):
            pass
    assert refused.value.status_code == 401


async def test_a_body_less_refresh_works(operator_credential):
    client = TestClient(app, client=LAN)
    auth = {
        "Authorization": "Bearer "
        + client.post(LOGIN, json={"credential": operator_credential}).json()["access_token"]
    }
    refreshed = client.post(REFRESH, headers=auth)  # body-less, no Content-Type
    assert refreshed.status_code == 200, (refreshed.status_code, refreshed.text)
    new = {"Authorization": f"Bearer {refreshed.json()['access_token']}"}
    assert client.get(PROFILE, headers=new).status_code == 200


async def test_json_bodies_still_work_on_both_routes(operator_credential):
    client = TestClient(app, client=LAN)
    auth = {
        "Authorization": "Bearer "
        + client.post(LOGIN, json={"credential": operator_credential}).json()["access_token"]
    }
    assert client.post(REFRESH, headers=auth, json={}).status_code == 200
    assert client.post(LOGOUT, headers=auth, json={}).status_code == 200


async def _stream():
    yield b'{"credential":"x"}'


async def test_the_content_type_rule_applies_when_there_is_a_body(operator_credential):
    """No data is fine; a body that is not JSON is still refused (415), on the limited routes."""
    async with _client(_limited_auth_app()) as client:
        empty = await client.post(LOGIN)  # no body at all: the handler's 422, not a 415
        assert empty.status_code == 422, empty.text
        text = await client.post(
            LOGIN, content="credential=x", headers={"Content-Type": "text/plain"}
        )
        assert text.status_code == 415
        chunked = await client.post(LOGIN, content=_stream())
        assert chunked.status_code == 415, "a streamed body with no Content-Type is a body"
        form = await client.post(LOGIN, data={"credential": "x"})
        assert form.status_code == 415
        wrong = await client.post(LOGIN, json={"credential": "x"})
        assert wrong.status_code == 401


async def test_the_whole_app_also_accepts_the_body_less_posts(operator_credential):
    """Through the real middleware stack (input validation included), not just the sub-app."""
    async with _client(app) as client:
        auth = await _login(client, operator_credential)
        out = await client.post(LOGOUT, headers=auth)
        assert out.status_code == 200
        assert (await client.post(LOGOUT, headers=auth)).status_code == 401  # already revoked


# ---- SEC20-14 ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("req-123_ABC.xyz:9", True),
        ("0123456789abcdef0123456789abcdef", True),
        ("A" * MAX_CORRELATION_ID_LENGTH, True),
        ("A" * (MAX_CORRELATION_ID_LENGTH + 1), False),
        ("A" * 5000, False),
        ("has space", False),
        ('quote"inside', False),
        ("<script>alert(1)</script>", False),
        ("new\\nline", False),
        ("", False),
        ("   ", False),
        (None, False),
    ],
)
def test_a_correlation_id_is_kept_only_when_short_and_plain(value, kept):
    assert (sanitize_correlation_id(value) is not None) is kept


@pytest.mark.parametrize("header", ["X-Correlation-ID", "X-Request-ID"])
async def test_the_caller_cannot_choose_a_long_or_hostile_correlation_id(header):
    async with _client(app) as client:
        long = await client.get("/healthz", headers={header: "A" * 5000})
        hostile = await client.get("/healthz", headers={header: '<b>"x"</b>'})
        fine = await client.get("/healthz", headers={header: "req-42"})
    for response in (long, hostile):
        echoed = response.headers["X-Correlation-ID"]
        assert len(echoed) <= MAX_CORRELATION_ID_LENGTH and echoed.isalnum(), echoed
    assert fine.headers["X-Correlation-ID"] == "req-42"


def test_the_default_cors_origins_are_the_uis_own(monkeypatch):
    monkeypatch.delenv("LAWN_BERRY_ALLOWED_ORIGINS", raising=False)
    assert _default_allowed_origins() == ["http://localhost:3000", "http://127.0.0.1:3000"]
    monkeypatch.setenv("LAWN_BERRY_ALLOWED_ORIGINS", "https://a.example, https://b.example ,")
    assert _default_allowed_origins() == ["https://a.example", "https://b.example"]


async def test_the_real_app_grants_cors_to_its_own_origin_only():
    preflight = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization",
    }
    async with _client(app) as client:
        ours = await client.options(LOGIN, headers={"Origin": "http://localhost:3000", **preflight})
        external = await client.options(
            LOGIN, headers={"Origin": "https://app.lawnberry.ai", **preflight}
        )
        evil = await client.options(LOGIN, headers={"Origin": "https://evil.example", **preflight})
    assert ours.headers.get("access-control-allow-origin") == "http://localhost:3000"
    for refused in (external, evil):
        assert refused.status_code == 400  # starlette's "Disallowed CORS origin"
        assert "access-control-allow-origin" not in refused.headers
