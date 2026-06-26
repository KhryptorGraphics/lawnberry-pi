import httpx
import pytest

from backend.src.main import app


@pytest.fixture(autouse=True)
def _strict_auth_limits(monkeypatch):
    """Apply strict login limits and reset limiter state for these tests.

    The global test config keeps auth limits permissive so unrelated tests can
    log in freely; this hardening suite exercises the limits, so it sets them
    low (3 attempts / 3 failures) and clears prior state per test.
    """
    from backend.src.services.auth_service import primary_auth_service

    monkeypatch.setenv("AUTH_RATE_LIMIT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("AUTH_RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "3")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "60")

    limiter = primary_auth_service.rate_limiter
    for store in (limiter._attempts, limiter._failures, limiter._lockout_until):
        store.clear()
    yield
    for store in (limiter._attempts, limiter._failures, limiter._lockout_until):
        store.clear()


@pytest.mark.asyncio
async def test_auth_login_rate_limit():
    transport = httpx.ASGITransport(app=app)
    headers = {"X-Client-Id": "rate-test-1"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Hit rate limit with valid credentials to isolate rate limiting from lockout
        for _ in range(3):
            resp = await client.post(
                "/api/v2/auth/login",
                json={"credential": "ok"},
                headers=headers,
            )
            assert resp.status_code == 200

        # Next request should be rate limited
        resp = await client.post(
            "/api/v2/auth/login",
            json={"credential": "ok"},
            headers=headers,
        )
        assert resp.status_code == 429
        # Retry-After header is recommended
        assert "retry-after" in {k.lower(): v for k, v in resp.headers.items()}


@pytest.mark.asyncio
async def test_auth_login_lockout_on_failed_attempts():
    transport = httpx.ASGITransport(app=app)
    headers = {"X-Client-Id": "lockout-test-1"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Three failed attempts (empty credential) triggers lockout
        for _ in range(3):
            resp = await client.post(
                "/api/v2/auth/login",
                json={"credential": ""},
                headers=headers,
            )
            assert resp.status_code == 401

        # Now even with correct credential, expect lockout (429)
        resp = await client.post(
            "/api/v2/auth/login",
            json={"credential": "ok"},
            headers=headers,
        )
        assert resp.status_code == 429
        assert "retry-after" in {k.lower(): v for k, v in resp.headers.items()}
