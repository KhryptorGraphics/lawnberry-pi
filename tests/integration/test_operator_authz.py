"""Operator authorization: writes need the bearer token, reads follow a REVIEWED contract.

Writes are gated by ``require_operator_auth`` (enabled via OPERATOR_AUTH_REQUIRED, on by default in
production). The suite disables it globally; here we re-enable it to verify enforcement.

Reads used to be asserted "open" wholesale (``test_reads_remain_open``), which is how an anonymous
``GET /settings/remote-access`` came to return the stored Cloudflare tunnel token (SEC20-01).
The contract now is the explicit list in ``test_operator_auth_coverage.py``: ``OPEN_READS`` (each
with the reason it may be read anonymously) stay open, ``MUST_BE_GATED_READS`` (secret-bearing
settings, the docs hub, the hardware self-test, the RTK diagnostics) need the bearer token.
"""

import httpx
import pytest

from backend.src.main import app
from tests.integration.test_operator_auth_coverage import (
    MUST_BE_GATED_READS,
    OPEN_READS,
    SELF_AUTH_READS,
)

BASE_URL = "http://test"

# Not plain reads: an endless MJPEG stream and the HTTP emulation of a WebSocket handshake (it
# needs upgrade headers and a bearer-shaped header, and answers 400/401 on its own terms).
_NOT_PLAIN_READS = ("/api/v2/camera/stream.mjpeg", "/api/v2/ws/")


@pytest.fixture(autouse=True)
def _enforce_auth(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    yield


async def _login(client: httpx.AsyncClient, credential: str) -> str:
    resp = await client.post("/api/v2/auth/login", json={"credential": credential})
    assert resp.status_code == 200, resp.text
    token = resp.json().get("access_token") or resp.json().get("token")
    assert token
    return token


def _concrete(path: str) -> str:
    import re

    return re.sub(r"\{[^}]+\}", "x", path)


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
async def test_writes_accepted_with_bearer_token(operator_credential):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        token = await _login(client, operator_credential)
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
async def test_the_reviewed_open_reads_stay_open():
    """Every ``OPEN_READS`` path answers an anonymous caller on its own terms (never 401)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        checked = 0
        for path in sorted(OPEN_READS):
            if path.startswith(_NOT_PLAIN_READS):
                continue
            response = await client.get(_concrete(path))
            assert response.status_code != 401, (path, response.text[:200])
            checked += 1
        assert checked > 40
        # the two the old test named, still readable (maps is NOT one of them any more)
        assert (await client.get("/api/v2/map/configuration")).status_code == 200
        assert (await client.get("/api/v2/planning/jobs")).status_code == 200


@pytest.mark.asyncio
async def test_the_gated_reads_need_the_token_and_the_self_authenticating_one_too(
    operator_credential, monkeypatch
):
    from backend.src.api.routers import maintenance

    monkeypatch.setattr(maintenance, "run_selftest", lambda: {"ok": True, "stub": True})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        token = await _login(client, operator_credential)
        auth = {"Authorization": f"Bearer {token}"}
        for path in sorted(MUST_BE_GATED_READS | set(SELF_AUTH_READS)):
            url = _concrete(path)
            anonymous = await client.get(url)
            assert anonymous.status_code == 401, (path, anonymous.status_code)
            with_token = await client.get(url, headers=auth)
            assert with_token.status_code != 401, (path, with_token.status_code, with_token.text)
        # a dead token is no better than none
        dead = await client.get(
            "/api/v2/settings/remote-access", headers={"Authorization": "Bearer x"}
        )
        assert dead.status_code == 401
