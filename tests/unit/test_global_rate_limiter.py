from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.src.middleware.rate_limiting import register_global_rate_limiter


def create_app():
    app = FastAPI()
    register_global_rate_limiter(app)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def test_rate_limiter_allows_burst_then_limits(monkeypatch):
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_RATE", "0.1")
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_BURST", "3")
    monkeypatch.setenv("GLOBAL_RATE_LIMIT_EXEMPT", "/health")
    app = create_app()
    client = TestClient(app)

    # 3 allowed
    for _ in range(3):
        r = client.get("/ping")
        assert r.status_code == 200
    # 4th should be limited
    r = client.get("/ping")
    assert r.status_code == 429
    assert r.json()["detail"] == "Too many requests"


# ---- SEC3-15 survivor M15: the bucket table has a memory bound ----


async def test_the_bucket_table_never_grows_past_its_cap_and_keeps_the_newest(monkeypatch):
    """M15: 300 distinct clients (each with a bucket that is not full again) used to be kept."""
    from backend.src.middleware.rate_limiting import GlobalRateLimiter

    monkeypatch.setattr(GlobalRateLimiter, "MAX_BUCKETS", 40)
    limiter = GlobalRateLimiter(FastAPI(), refill_rate_per_sec=0.1, burst=20)
    for i in range(300):
        allowed, _ = await limiter._consume_token(f"peer-{i}|", 0.1, 20)
        assert allowed
    assert len(limiter._buckets) <= GlobalRateLimiter.MAX_BUCKETS + 1
    assert "peer-299|" in limiter._buckets and "peer-0|" not in limiter._buckets
