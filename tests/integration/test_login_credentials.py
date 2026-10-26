"""Login with the *configured* operator credential, and no hard-wired admin/admin.

The web UI login form posts ``{username, password}``. It used to work only
because the backend mapped the literal pair ``admin/admin`` to the operator
credential, so a deployment that enforced auth (``OPERATOR_AUTH_REQUIRED=1``)
was reachable with a public, hard-wired password. Now the password *is* the
shared operator credential (the username is informational), and ``admin/admin``
works only if the operator credential is literally ``admin``.

Outside SIM_MODE the auth service validates for real (in SIM_MODE any non-empty
credential is accepted), so each test builds a non-SIM ``AuthService`` and
installs it for both the login router and the ``require_operator_auth`` guard.
"""

from __future__ import annotations

import httpx
import pytest

from backend.src.main import app
from backend.src.services.auth_service import AuthService

CRED = "correct-horse-battery-staple"
# The credential the backend used to fall back to. It must not authenticate any more.
REMOVED_PUBLIC_DEFAULT = "operator123"


@pytest.fixture
def strict_auth(monkeypatch):
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService()  # reads SIM_MODE / the credential at construction
    monkeypatch.setenv("SIM_MODE", "1")  # the rest of the app stays in simulation
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setattr("backend.src.api.routers.auth.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.rest_v1._auth_service", service)
    return service


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_frontend_form_login_works_with_the_configured_credential(strict_auth):
    async with _client() as client:
        resp = await client.post(
            "/api/v2/auth/login", json={"username": "operator", "password": CRED}
        )
        assert resp.status_code == 200, resp.text
        token = resp.json()["access_token"]

        # The issued token really authorises an operator-gated write under enforcement...
        clear = "/api/v2/tractor/clear-emergency"
        ok = await client.post(clear, headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200, ok.text
        # ...and the same write without it is refused.
        assert (await client.post(clear)).status_code == 401


async def test_shared_credential_payload_still_works(strict_auth):
    async with _client() as client:
        resp = await client.post("/api/v2/auth/login", json={"credential": CRED})
    assert resp.status_code == 200 and resp.json()["token"]


async def test_the_admin_admin_alias_is_gone(strict_auth):
    async with _client() as client:
        alias = await client.post(
            "/api/v2/auth/login", json={"username": "admin", "password": "admin"}
        )
        assert alias.status_code == 401, "admin/admin must not authenticate any more"
        assert (
            await client.post("/api/v2/auth/login", json={"credential": "admin"})
        ).status_code == 401
        wrong = await client.post(
            "/api/v2/auth/login",
            json={"username": "operator", "password": REMOVED_PUBLIC_DEFAULT},
        )
        assert wrong.status_code == 401, "the public default must not work once a credential is set"


async def test_v1_login_has_no_alias_either(strict_auth):
    # v1 logs in through the same service as v2 (one limiter, strict even under SIM opt-in)
    async with _client() as client:
        ok = await client.post("/api/v1/auth/login", json={"username": "ops", "password": CRED})
        alias = await client.post(
            "/api/v1/auth/login", json={"username": "admin", "password": "admin"}
        )
    assert ok.status_code == 200, ok.text
    assert alias.status_code == 401
