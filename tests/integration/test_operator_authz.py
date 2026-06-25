"""Operator authorization on config-changing write endpoints.

Writes are gated by ``require_operator_auth`` (enabled via OPERATOR_AUTH_REQUIRED,
on by default in production). The suite disables it globally; here we re-enable it
to verify enforcement, and confirm reads stay open.
"""

import httpx
import pytest

from backend.src.main import app

BASE_URL = "http://test"


@pytest.fixture(autouse=True)
def _enforce_auth(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    yield


async def _login(client: httpx.AsyncClient) -> str:
    resp = await client.post("/api/v2/auth/login", json={"credential": "operator"})
    assert resp.status_code == 200, resp.text
    token = resp.json().get("access_token") or resp.json().get("token")
    assert token
    return token


@pytest.mark.asyncio
async def test_writes_rejected_without_auth():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        # A representative set of the protected write endpoints.
        r1 = await client.put("/api/v2/settings/maps", json={"provider": "osm"})
        assert r1.status_code == 401, r1.text

        r2 = await client.put("/api/v2/map/configuration", json={"zones": []})
        assert r2.status_code == 401, r2.text

        r3 = await client.post(
            "/api/v2/planning/jobs",
            json={"name": "x", "schedule": "08:00", "zones": [], "enabled": True},
        )
        assert r3.status_code == 401, r3.text


@pytest.mark.asyncio
async def test_writes_accepted_with_bearer_token():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        token = await _login(client)
        headers = {"Authorization": f"Bearer {token}"}

        r1 = await client.put("/api/v2/settings/maps", json={"provider": "osm"}, headers=headers)
        assert r1.status_code == 200, r1.text
        assert r1.json()["provider"] == "osm"

        r2 = await client.post(
            "/api/v2/planning/jobs",
            json={"name": "auth-mow", "schedule": "08:00", "zones": [], "enabled": True},
            headers=headers,
        )
        assert r2.status_code == 201, r2.text


@pytest.mark.asyncio
async def test_reads_remain_open():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        # GET endpoints are not gated even with enforcement on.
        assert (await client.get("/api/v2/settings/maps")).status_code == 200
        assert (await client.get("/api/v2/map/configuration")).status_code == 200
        assert (await client.get("/api/v2/planning/jobs")).status_code == 200
