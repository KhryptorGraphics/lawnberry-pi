"""Manual-control unlock must not fail open.

With ``OPERATOR_AUTH_REQUIRED=1`` and no dedicated manual-control password, any non-empty
password used to unlock manual control. The session that returns is accepted by
``/control/drive`` *and* by ``require_operator_auth`` (every operator-gated write), so it
bypassed the operator credential entirely. The password is now checked against the bcrypt
``password_hash``, ``MANUAL_CONTROL_PASSWORD``, or else the shared operator credential, and
nothing matches when none of them is usable.

Outside SIM_MODE the auth service validates for real, so each test builds a non-SIM
``AuthService`` and installs it for the unlock route and the ``require_operator_auth`` guard.
"""

from __future__ import annotations

import httpx
import pytest

from backend.src.core.globals import _manual_control_sessions, _security_settings
from backend.src.main import app
from backend.src.models.auth_security_config import SecurityLevel
from backend.src.services.auth_service import AuthService

CRED = "correct-horse-battery-staple"
UNLOCK = "/api/v2/control/manual-unlock"
OPERATOR_WRITE = "/api/v2/tractor/clear-emergency"  # behind require_operator_auth


def _install_service(monkeypatch, credential: str | None) -> AuthService:
    if credential is None:
        monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
    else:
        monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", credential)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService()  # reads SIM_MODE / the credential at construction
    monkeypatch.setenv("SIM_MODE", "1")  # the rest of the app stays in simulation
    monkeypatch.setattr("backend.src.api.routers.auth.primary_auth_service", service)
    monkeypatch.setattr("backend.src.api.deps.primary_auth_service", service)
    return service


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.delenv("MANUAL_CONTROL_PASSWORD", raising=False)
    monkeypatch.setattr(_security_settings, "password_hash", None)
    monkeypatch.setattr(_security_settings, "security_level", SecurityLevel.PASSWORD)
    _manual_control_sessions.clear()
    service = _install_service(monkeypatch, CRED)
    yield service
    _manual_control_sessions.clear()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _unlock(client: httpx.AsyncClient, password: str | None, method: str = "password"):
    body = {"method": method}
    if password is not None:
        body["password"] = password
    return client.post(UNLOCK, json=body)


async def test_an_arbitrary_password_no_longer_unlocks_manual_control(enforced):
    async with _client() as client:
        resp = await _unlock(client, "anything-at-all")
        assert resp.status_code == 401, resp.text
        assert _manual_control_sessions == {}, "a refused unlock must not create a session"
        invented = await client.post(OPERATOR_WRITE, headers={"X-Session-Id": "made-up"})
        assert invented.status_code == 401


async def test_the_operator_credential_unlocks_and_that_session_is_honoured(enforced):
    async with _client() as client:
        resp = await _unlock(client, CRED)
        assert resp.status_code == 200, resp.text
        session_id = resp.json()["session_id"]
        assert session_id
        ok = await client.post(OPERATOR_WRITE, headers={"X-Session-Id": session_id})
        assert ok.status_code == 200, ok.text
        assert (await client.post(OPERATOR_WRITE)).status_code == 401


async def test_with_no_operator_credential_configured_nothing_unlocks(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.delenv("MANUAL_CONTROL_PASSWORD", raising=False)
    monkeypatch.setattr(_security_settings, "password_hash", None)
    monkeypatch.setattr(_security_settings, "security_level", SecurityLevel.PASSWORD)
    _manual_control_sessions.clear()
    _install_service(monkeypatch, None)
    async with _client() as client:
        for attempt in ("operator123", "admin", "anything-at-all"):
            resp = await _unlock(client, attempt)
            assert resp.status_code == 401, (attempt, resp.text)
    assert _manual_control_sessions == {}


async def test_a_published_default_cannot_be_the_manual_password(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.delenv("MANUAL_CONTROL_PASSWORD", raising=False)
    monkeypatch.setattr(_security_settings, "password_hash", None)
    _manual_control_sessions.clear()
    _install_service(monkeypatch, "operator123")  # configured, but refused by the policy
    async with _client() as client:
        assert (await _unlock(client, "operator123")).status_code == 401
    assert _manual_control_sessions == {}


async def test_a_dedicated_manual_password_replaces_the_operator_credential(enforced, monkeypatch):
    monkeypatch.setenv("MANUAL_CONTROL_PASSWORD", "dedicated-manual-pass-1")
    async with _client() as client:
        assert (await _unlock(client, CRED)).status_code == 401
        assert (await _unlock(client, "dedicated-manual-pass-1")).status_code == 200


@pytest.mark.parametrize(
    "level", [SecurityLevel.PASSWORD, SecurityLevel.TOTP, SecurityLevel.TUNNEL_AUTH]
)
async def test_a_blank_password_is_refused_at_every_security_level(enforced, monkeypatch, level):
    monkeypatch.setattr(_security_settings, "security_level", level)
    async with _client() as client:
        for blank in (None, "", "   "):
            resp = await _unlock(client, blank)
            assert resp.status_code == 400, (level, blank, resp.text)
    assert _manual_control_sessions == {}
