import pytest
from fastapi import FastAPI

from backend.src.middleware.security import SecurityMiddleware


class FakeTime:
    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def time(self) -> float:
        return self.value


@pytest.mark.asyncio
async def test_successful_requests_do_not_consume_attempt_quota(monkeypatch):
    app = FastAPI()
    middleware = SecurityMiddleware(
        app,
        rate_limit_window_seconds=60,
        rate_limit_max_attempts=3,
        lockout_failures=3,
        lockout_seconds=60,
    )
    fake_time = FakeTime()
    monkeypatch.setattr("backend.src.middleware.security.time.time", fake_time.time)

    client_key = "client-a"

    # First attempt succeeds
    assert await middleware._preprocess_rate_limit(client_key) is None
    await middleware._postprocess_rate_limit(client_key, 200)

    # Attempts deque should be cleared for successful logins
    assert client_key not in middleware._attempts

    # A subsequent request within the window should still be allowed
    fake_time.advance(10)
    assert await middleware._preprocess_rate_limit(client_key) is None


@pytest.mark.asyncio
async def test_failed_attempts_trigger_rate_limit(monkeypatch):
    app = FastAPI()
    middleware = SecurityMiddleware(
        app,
        rate_limit_window_seconds=30,
        rate_limit_max_attempts=3,
        lockout_failures=99,
        lockout_seconds=60,
    )
    fake_time = FakeTime()
    monkeypatch.setattr("backend.src.middleware.security.time.time", fake_time.time)

    client_key = "client-b"

    for _ in range(3):
        assert await middleware._preprocess_rate_limit(client_key) is None
        await middleware._postprocess_rate_limit(client_key, 401)
        fake_time.advance(1)

    response = await middleware._preprocess_rate_limit(client_key)
    assert response is not None
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0

    fake_time.advance(35)
    assert await middleware._preprocess_rate_limit(client_key) is None


@pytest.mark.asyncio
async def test_successful_auth_resets_failure_state(monkeypatch):
    app = FastAPI()
    middleware = SecurityMiddleware(
        app,
        rate_limit_window_seconds=45,
        rate_limit_max_attempts=3,
        lockout_failures=99,
        lockout_seconds=60,
    )
    fake_time = FakeTime()
    monkeypatch.setattr("backend.src.middleware.security.time.time", fake_time.time)

    client_key = "client-c"

    # Two failed attempts should be counted
    for _ in range(2):
        assert await middleware._preprocess_rate_limit(client_key) is None
        await middleware._postprocess_rate_limit(client_key, 401)
        fake_time.advance(1)

    # Successful attempt clears counters
    assert await middleware._preprocess_rate_limit(client_key) is None
    await middleware._postprocess_rate_limit(client_key, 200)

    assert client_key not in middleware._attempts
    assert client_key not in middleware._failures
    assert client_key not in middleware._lockout_until

    fake_time.advance(5)
    assert await middleware._preprocess_rate_limit(client_key) is None


# ---- SEC3-15 survivors M10 and M16: the lockout itself and the memory bound of the tables ----


@pytest.mark.asyncio
async def test_the_failure_lockout_refuses_a_client_before_its_attempt_budget_is_spent(monkeypatch):
    """M10: with the lockout never set, only the (here huge) attempt budget would stop a guesser."""
    middleware = SecurityMiddleware(
        FastAPI(),
        rate_limit_window_seconds=60,
        rate_limit_max_attempts=99,
        lockout_failures=3,
        lockout_seconds=60,
    )
    fake_time = FakeTime()
    monkeypatch.setattr("backend.src.middleware.security.time.time", fake_time.time)
    for _ in range(3):
        assert await middleware._preprocess_rate_limit("guesser") is None
        await middleware._postprocess_rate_limit("guesser", 401)
        fake_time.advance(1)
    locked = await middleware._preprocess_rate_limit("guesser")
    assert locked is not None and locked.status_code == 429
    assert 1 <= int(locked.headers["Retry-After"]) <= 60
    assert await middleware._preprocess_rate_limit("neighbour") is None  # per client, not global
    fake_time.advance(61)
    assert await middleware._preprocess_rate_limit("guesser") is None  # and it ends


@pytest.mark.asyncio
async def test_no_table_of_the_auth_middleware_grows_past_its_cap(monkeypatch):
    """M16: a client that rotates its address (or a flood of distinct peers) must not make the
    attempt, failure and lockout tables grow without bound."""
    monkeypatch.setattr(SecurityMiddleware, "MAX_TRACKED_KEYS", 40)
    middleware = SecurityMiddleware(
        FastAPI(),
        rate_limit_window_seconds=600,
        rate_limit_max_attempts=5,
        lockout_failures=1,
        lockout_seconds=600,
    )
    fake_time = FakeTime()
    monkeypatch.setattr("backend.src.middleware.security.time.time", fake_time.time)
    for i in range(300):
        key = f"peer-{i}"
        await middleware._preprocess_rate_limit(key)
        await middleware._postprocess_rate_limit(key, 401)
        fake_time.advance(0.01)
    cap = SecurityMiddleware.MAX_TRACKED_KEYS
    assert len(middleware._attempts) <= cap + 1
    assert len(middleware._failures) <= cap + 1
    assert len(middleware._lockout_until) <= cap + 1
    # the newest clients are the ones that are remembered
    assert "peer-299" in middleware._lockout_until and "peer-0" not in middleware._lockout_until
