"""Login lockout and every rate limit count per TCP peer, never per client-supplied id (SEC-03),
and the E-stop POSTs are never limited (SEC-05 = SF-03).

Before: ``X-Client-Id`` (a header the client writes) keyed the login lockout, the auth-prefix
limiter and the global limiter, so a fresh value per request gave unlimited password guesses and
unbounded limiter tables; the v1 login compared with ``!=`` ahead of any limiter; and after 20
polls the global limiter answered ``429`` to ``POST /tractor/emergency-stop`` and
``/control/emergency-stop``.
"""

from __future__ import annotations

import itertools

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.src.core.peer_address import peer_key
from backend.src.main import app
from backend.src.middleware.emergency_paths import EMERGENCY_STOP_PATHS
from backend.src.middleware.rate_limiting import GlobalRateLimiter, register_global_rate_limiter
from backend.src.middleware.security import SecurityMiddleware
from backend.src.services.auth_service import AuthService, RateLimiter

CRED = "correct-horse-battery-staple"
LOGIN = "/api/v2/auth/login"
ESTOP_PATHS = sorted(EMERGENCY_STOP_PATHS)


def _client_for(target, peer=("127.0.0.1", 123)) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=target, client=peer)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---- the key ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("192.0.2.7", "ip:192.0.2.7"),
        ("::ffff:192.0.2.7", "ip:192.0.2.7"),
        # GLOBAL unicast IPv6 is one client per /64 ...
        ("2606:4700:4700:0:aaaa:bbbb:cccc:dddd", "ip6:2606:4700:4700::/64"),
        ("2606:4700:4700:0:1111:2222:3333:4444", "ip6:2606:4700:4700::/64"),
        # ... everything else is one client per ADDRESS (a LAN's /64 is not one subscriber)
        ("fd00:1234:5678:1::a1", "ip:fd00:1234:5678:1::a1"),
        ("fe80::1%eth0", "ip:fe80::1"),
        ("2001:db8:1:2::9", "ip:2001:db8:1:2::9"),
        ("TestClient", "host:testclient"),
        (None, "anon"),
        ("", "anon"),
        ("   ", "anon"),
    ],
)
def test_peer_key_normalises_the_address(host, expected):
    assert peer_key(host) == expected


def test_peer_key_separates_prefixes_and_bounds_hostile_names():
    assert peer_key("2606:4700:4700:1::1") != peer_key("2606:4700:4700:2::1")
    assert len(peer_key("A" * 10_000)) <= 80


# ---- login lockout ---------------------------------------------------------------------------


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "3")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "60")
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService()
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setattr("backend.src.api.routers.auth.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.rest_v1._auth_service", service)
    return service


async def test_rotating_client_ids_do_not_reset_the_login_lockout(strict):
    async with _client_for(app) as client:
        for n in range(3):
            resp = await client.post(
                LOGIN, json={"credential": f"wrong-{n}"}, headers={"X-Client-Id": f"rot-{n}"}
            )
            assert resp.status_code == 401
        locked = await client.post(
            LOGIN, json={"credential": CRED}, headers={"X-Client-Id": "a-brand-new-id"}
        )
        assert locked.status_code == 429, "the correct credential must not get past the lockout"
        assert int(locked.headers["Retry-After"]) >= 1
    # Another peer is unaffected by this one's lockout.
    async with _client_for(app, ("198.51.100.9", 5)) as other:
        assert (await other.post(LOGIN, json={"credential": CRED})).status_code == 200


async def test_v1_login_counts_against_the_same_limiter_whatever_the_client_id(strict):
    async with _client_for(app) as client:
        for n in range(3):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"credential": f"wrong-{n}"},
                headers={"X-Client-Id": f"v1-{n}"},
            )
            assert resp.status_code == 401
        # The v1 failures were counted (they used to be refused ahead of any limiter), so the
        # v2 login from the same peer is locked out too.
        assert (await client.post(LOGIN, json={"credential": CRED})).status_code == 429


def test_the_login_limiter_state_is_bounded(monkeypatch):
    monkeypatch.setattr(RateLimiter, "MAX_TRACKED_KEYS", 50)
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "5")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "60")
    limiter = RateLimiter(failure_limit=5, lockout_seconds=60)
    for _ in range(5):
        limiter.record_failure("victim")
    assert limiter.lockout_remaining("victim") is not None
    for n in range(500):
        limiter.record_attempt(f"ip:10.0.{n // 250}.{n % 250}")
        limiter.record_failure(f"ip:10.9.{n // 250}.{n % 250}")
    cap = RateLimiter.MAX_TRACKED_KEYS
    assert len(limiter._attempts) <= cap + 1 and len(limiter._failures) <= cap + 1
    assert limiter.lockout_remaining("victim") is not None, "a flood must not unlock anyone"


def test_idle_login_limiter_entries_are_forgotten():
    from datetime import UTC, datetime, timedelta

    limiter = RateLimiter()
    for n in range(20):
        limiter.record_attempt(f"ip:10.0.0.{n}")
        limiter.record_failure(f"ip:10.0.0.{n}")
    long_ago = datetime.now(UTC) - timedelta(hours=3)
    for key in list(limiter._attempts):
        limiter._attempts[key] = [long_ago]
        limiter._failure_at[key] = long_ago
    limiter._last_prune = long_ago
    limiter.record_attempt("ip:192.0.2.1")
    assert set(limiter._attempts) == {"ip:192.0.2.1"}
    assert limiter._failures == {}


# ---- the auth-prefix limiter ------------------------------------------------------------------


def _auth_app() -> FastAPI:
    sub = FastAPI()

    @sub.post(LOGIN)
    async def login() -> JSONResponse:
        return JSONResponse({"detail": "bad credentials"}, status_code=401)

    sub.add_middleware(
        SecurityMiddleware,
        rate_limit_window_seconds=60,
        rate_limit_max_attempts=3,
        lockout_failures=3,
        lockout_seconds=60,
    )
    return sub


async def test_the_auth_prefix_limiter_ignores_client_ids():
    async with _client_for(_auth_app(), ("192.0.2.1", 1)) as client:
        codes = [
            (await client.post(LOGIN, json={}, headers={"X-Client-Id": f"id-{n}"})).status_code
            for n in range(8)
        ]
    assert codes[:3] == [401, 401, 401]
    assert set(codes[3:]) == {429}, codes


async def test_the_auth_prefix_limiter_state_is_bounded(monkeypatch):
    monkeypatch.setattr(SecurityMiddleware, "MAX_TRACKED_KEYS", 50)
    middleware = SecurityMiddleware(FastAPI(), lockout_failures=99, rate_limit_max_attempts=99)
    for n in range(300):
        key = f"ip:10.1.{n // 250}.{n % 250}"
        await middleware._preprocess_rate_limit(key)
        await middleware._postprocess_rate_limit(key, 401)
    cap = SecurityMiddleware.MAX_TRACKED_KEYS
    assert len(middleware._attempts) <= cap + 1 and len(middleware._failures) <= cap + 1


# ---- the global limiter -----------------------------------------------------------------------


def _limited_app(monkeypatch, *, rate="0.1", burst="5", exempt=None) -> FastAPI:
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_RATE", rate)
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_BURST", burst)
    monkeypatch.delenv("GLOBAL_RATE_LIMIT_OVERRIDES", raising=False)
    if exempt is None:
        monkeypatch.delenv("GLOBAL_RATE_LIMIT_EXEMPT", raising=False)
    else:
        monkeypatch.setenv("GLOBAL_RATE_LIMIT_EXEMPT", exempt)
    sub = FastAPI()
    register_global_rate_limiter(sub)

    def ok() -> dict:
        return {"ok": True}

    sub.add_api_route("/poll", ok, methods=["GET"])
    sub.add_api_route("/api/v2/tractor/state", ok, methods=["GET"])
    sub.add_api_route("/api/v2/tractor/left-lever", ok, methods=["POST"])
    sub.add_api_route("/api/v2/control/emergency_clear", ok, methods=["POST"])
    for path in ESTOP_PATHS:
        sub.add_api_route(path, ok, methods=["POST"])
    return sub


async def test_the_global_limiter_ignores_client_ids(monkeypatch):
    sub = _limited_app(monkeypatch, burst="5")
    async with _client_for(sub) as client:
        codes = [
            (await client.get("/poll", headers={"X-Client-Id": f"id-{n}"})).status_code
            for n in range(20)
        ]
    assert codes.count(200) == 5 and codes.count(429) == 15, codes


async def test_the_estop_posts_are_never_rate_limited_even_with_a_custom_exempt_list(monkeypatch):
    sub = _limited_app(monkeypatch, burst="2", exempt="/only-this")
    async with _client_for(sub) as client:
        assert [(await client.get("/poll")).status_code for _ in range(4)][-1] == 429
        paths = itertools.cycle(ESTOP_PATHS)
        codes = [(await client.post(next(paths), json={})).status_code for _ in range(150)]
        assert set(codes) == {200}, {c for c in codes if c != 200}
        # The slash variant is only a redirect to the canonical path, never a 429.
        assert (await client.post(ESTOP_PATHS[0] + "/", json={})).status_code != 429
        # The authenticated clear is NOT exempt: a prefix match would have exempted it too.
        clears = [
            (await client.post("/api/v2/control/emergency_clear", json={})).status_code
            for _ in range(40)
        ]
        assert 429 in clears


async def test_tractor_commands_have_their_own_bucket(monkeypatch):
    """The control page re-sends lever commands at ~18/s; polling must not starve them."""
    sub = _limited_app(monkeypatch, burst="2")
    async with _client_for(sub) as client:
        polls = [(await client.get("/api/v2/tractor/state")).status_code for _ in range(4)]
        assert polls == [200, 200, 429, 429]
        sent = [
            (await client.post("/api/v2/tractor/left-lever", json={"value": 0.5})).status_code
            for _ in range(60)
        ]
        assert set(sent) == {200}, "the command bucket must not share the polling budget"


async def test_the_global_limiter_state_is_bounded(monkeypatch):
    monkeypatch.setattr(GlobalRateLimiter, "MAX_BUCKETS", 50)
    limiter = GlobalRateLimiter(FastAPI(), refill_rate_per_sec=0.1, burst=3)
    for n in range(300):
        await limiter._consume_token(f"ip:10.2.{n // 250}.{n % 250}|", 0.1, 3)
    assert len(limiter._buckets) <= GlobalRateLimiter.MAX_BUCKETS + 1


async def test_the_real_estop_routes_answer_after_the_shipped_limiter_ran_dry(monkeypatch):
    """Shipped defaults (2/s, burst 20, one shared bucket) and the REAL E-stop routes."""
    for name in (
        "GLOBAL_RATE_LIMIT_RATE",
        "GLOBAL_RATE_LIMIT_BURST",
        "GLOBAL_RATE_LIMIT_EXEMPT",
        "GLOBAL_RATE_LIMIT_OVERRIDES",
    ):
        monkeypatch.delenv(name, raising=False)
    from backend.src.api import rest
    from backend.src.api.routers.tractor import router as tractor_router
    from backend.src.core.globals import _blade_state, _safety_state
    from backend.src.services.tractor_service import get_tractor_service

    real = FastAPI()
    register_global_rate_limiter(real)
    real.include_router(rest.router, prefix="/api/v2")
    real.include_router(tractor_router, prefix="/api/v2")
    real.add_api_route("/poll", lambda: {"ok": True}, methods=["GET"])

    try:
        async with _client_for(real) as client:
            polls = [(await client.get("/poll")).status_code for _ in range(30)]
            assert polls.count(429) >= 5, "the limiter must really be active for this test"
            paths = itertools.cycle(ESTOP_PATHS)
            codes = {}
            for _ in range(105):  # more than 100, spread over the three stop endpoints
                path = next(paths)
                resp = await client.post(path)
                codes.setdefault(path, set()).add(resp.status_code)
                assert resp.status_code < 400, (path, resp.status_code, resp.text)
            assert set(codes) == set(ESTOP_PATHS)
    finally:  # leave the process-wide E-stop state as the other tests expect it
        await get_tractor_service().clear_emergency()
        _safety_state["emergency_stop_active"] = False
        _blade_state["active"] = False
        rest._emergency_until = 0.0  # the stop arms a 0.2 s drive-block TTL: do not leak it
        rest._client_emergency.clear()


# ---- SEC3-11: on an IPv6 LAN one device's typos do not lock its neighbours

ULA = "fd00:1234:5678:1::{:x}"  # one SLAAC /64 of a LAN
GLOBAL = "2606:4700:4700::{:x}"  # one ISP /64


@pytest.fixture
def lan(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "5")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "30")
    monkeypatch.setenv("AUTH_RATE_LIMIT_MAX_ATTEMPTS", "1000")
    service = AuthService()
    monkeypatch.setenv("SIM_MODE", "1")
    return service


async def login_as(service: AuthService, ip: str, credential: str = "wrong") -> int:
    from backend.src.services.auth_service import AuthenticationError

    try:
        await service.authenticate(credential, client_ip=ip)
    except AuthenticationError as exc:
        return exc.status_code
    return 200


async def test_one_lan_devices_typos_do_not_lock_its_neighbour(lan):
    """Reviewer's repro: laptop ``::a1`` and phone ``::b2`` were both ``ip6:fd00:...:1::/64``."""
    assert [await login_as(lan, ULA.format(0xA1)) for _ in range(5)] == [401] * 5
    assert await login_as(lan, ULA.format(0xA1), CRED) == 429, "the typist is locked out"
    assert await login_as(lan, ULA.format(0xB2), CRED) == 200, "the neighbour must not be"


async def test_a_host_rotating_addresses_inside_one_lan_prefix_is_still_locked_out(lan):
    """Per-address keys alone would give it a fresh five guesses per address: the /64 group
    counter locks the whole /64 after 25 failures, however the addresses vary."""
    codes = [await login_as(lan, ULA.format(0x100 + n)) for n in range(100)]
    assert codes == [401] * 25 + [429] * 75, codes
    assert await login_as(lan, ULA.format(0x9999), CRED) == 429, "a correct guess does not pass"
    other_prefix = "fd00:1234:5678:2::1"
    assert await login_as(lan, other_prefix, CRED) == 200, "another /64 is another group"


async def test_a_global_prefix_is_one_client_as_before(lan):
    assert [await login_as(lan, GLOBAL.format(n + 1)) for n in range(5)] == [401] * 5
    assert await login_as(lan, GLOBAL.format(0x77), CRED) == 429, "same /64, another address"
    assert await login_as(lan, "2606:4700:4701::1", CRED) == 200, "another /64"


async def test_the_group_lockout_ends_and_a_success_resets_the_count(lan, monkeypatch):
    for n in range(20):
        assert await login_as(lan, ULA.format(0x200 + n)) == 401
    assert await login_as(lan, ULA.format(0x300), CRED) == 200  # a success from the /64: reset
    codes = [await login_as(lan, ULA.format(0x400 + n)) for n in range(30)]
    assert codes == [401] * 25 + [429] * 5, "25 more, not 5: the success cleared the count"


async def test_ipv4_has_no_group_and_is_unchanged(lan):
    assert [await login_as(lan, f"198.51.100.{n}") for n in range(40)] == [401] * 40
    assert [await login_as(lan, "198.51.100.99") for _ in range(5)] == [401] * 5
    assert await login_as(lan, "198.51.100.99", CRED) == 429


async def test_the_login_limiter_counts_one_client_whatever_the_spelling_of_its_address(lan):
    """S11: the limiter key is ``peer_key`` of the peer, not the raw string: a dual-stack socket
    reports ``::ffff:a.b.c.d`` and the same host may also arrive as plain IPv4, and one global /64
    has as many spellings as it has addresses."""
    mapped, plain = "::ffff:198.51.100.7", "198.51.100.7"
    assert [await login_as(lan, ip) for ip in (mapped, plain, mapped, plain, mapped)] == [401] * 5
    assert await login_as(lan, plain, CRED) == 429 and await login_as(lan, mapped, CRED) == 429
    spellings = ["2606:4700:4700::1", "2606:4700:4700:0:0:0:0:2", "2606:4700:4700::3%eth0"]
    for ip in spellings + spellings[:2]:
        assert await login_as(lan, ip) == 401
    assert await login_as(lan, "2606:4700:4700::4", CRED) == 429


async def test_the_unlock_throttle_has_the_same_group_backstop(lan, monkeypatch):
    """``/control/manual-unlock`` is a second place to guess the credential: rotating the source
    address inside one LAN /64 must not get a fresh guess per address there either."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setattr("backend.src.api.routers.auth.primary_auth_service", lan)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", lan)
    token = (await lan.authenticate(CRED, client_ip="198.51.100.1")).token
    headers = {"Authorization": f"Bearer {token}"}
    codes = []
    for n in range(40):
        async with _client_for(app, (ULA.format(0x500 + n), 40000 + n)) as client:
            response = await client.post(
                "/api/v2/control/manual-unlock",
                json={"method": "password", "password": "wrong"},
                headers=headers,
            )
        codes.append(response.status_code)
    assert codes == [401] * 25 + [429] * 15, codes
    async with _client_for(app, (ULA.format(0x5FF), 41000)) as client:
        right = await client.post(
            "/api/v2/control/manual-unlock",
            json={"method": "password", "password": CRED},
            headers=headers,
        )
    assert right.status_code == 429
