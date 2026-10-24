"""Manual-control unlock is a step-up for a logged-in operator and never fails open.

History: with ``OPERATOR_AUTH_REQUIRED=1`` the unlock minted a session id for any non-empty
password (fixed earlier, see the first tests) AND for any ``CF-Access-Jwt-Assertion`` header value
(a JWT that was decoded but never verified), and ``require_operator_auth`` accepted that id as
``X-Session-Id``, so one unauthenticated request obtained every operator-gated write (SEC-01).

The contract now:

* ``POST /control/manual-unlock`` needs the operator's bearer token AND the credential again;
* the session id it returns is random (not derived from the secret) and is NOT an operator
  credential: ``require_operator_auth`` accepts bearer tokens only;
* Cloudflare Access assertions unlock nothing (501 / ``authorized: false``);
* wrong credentials are throttled per TCP peer, whatever ``X-Client-Id`` says.

Outside SIM_MODE the auth service validates for real, so each test builds a non-SIM
``AuthService`` and installs it for the routes and the ``require_operator_auth`` guard.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import HTTPException

from backend.src.api.routers.auth import (
    MANUAL_SESSION_LIMIT,
    _issue_manual_session,
    _resolve_manual_session,
)
from backend.src.core.globals import _manual_control_sessions, _security_settings
from backend.src.main import app
from backend.src.models.auth_security_config import SecurityLevel, TOTPConfig
from backend.src.services.auth_service import AuthService

CRED = "correct-horse-battery-staple"
UNLOCK = "/api/v2/control/manual-unlock"
STATUS = "/api/v2/control/manual-unlock/status"
LOGIN = "/api/v2/auth/login"
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
    monkeypatch.setattr(_security_settings, "totp_config", None)
    monkeypatch.setattr(_security_settings, "security_level", SecurityLevel.PASSWORD)
    _manual_control_sessions.clear()
    service = _install_service(monkeypatch, CRED)
    yield service
    _manual_control_sessions.clear()


def _client(peer: tuple[str, int] = ("127.0.0.1", 123)) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, client=peer)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _login(client: httpx.AsyncClient) -> dict[str, str]:
    resp = await client.post(LOGIN, json={"credential": CRED})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _unlock(client, password, *, headers=None, method="password"):
    body = {"method": method}
    if password is not None:
        body["password"] = password
    return client.post(UNLOCK, json=body, headers=headers)


def _forged_cloudflare_jwt(email: str = "evil@attacker.example") -> str:
    """A JWT-shaped string nobody signed: exactly what the old code accepted as an identity."""

    def seg(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none'})}.{seg({'email': email, 'exp': 4102444800})}.forged"


# ---- the password path ----------------------------------------------------------------------


async def test_an_arbitrary_password_does_not_unlock_manual_control(enforced):
    async with _client() as client:
        headers = await _login(client)
        resp = await _unlock(client, "anything-at-all", headers=headers)
        assert resp.status_code == 401, resp.text
        assert _manual_control_sessions == {}, "a refused unlock must not create a session"
        invented = await client.post(OPERATOR_WRITE, headers={"X-Session-Id": "made-up"})
        assert invented.status_code == 401


async def test_unlock_needs_a_logged_in_operator_even_with_the_right_password(enforced):
    async with _client() as client:
        resp = await _unlock(client, CRED)  # no bearer token
        assert resp.status_code == 401, resp.text
        assert "authentication required" in resp.json()["detail"].lower()
        assert _manual_control_sessions == {}
        bogus = await _unlock(client, CRED, headers={"Authorization": "Bearer not-a-token"})
        assert bogus.status_code == 401
    assert _manual_control_sessions == {}


async def test_the_operator_credential_unlocks_but_the_session_is_not_an_operator_login(enforced):
    async with _client() as client:
        headers = await _login(client)
        resp = await _unlock(client, CRED, headers=headers)
        assert resp.status_code == 200, resp.text
        session_id = resp.json()["session_id"]
        assert resp.json()["authorized"] is True and session_id
        # The id is a manual-control factor, not a login: alone it opens no operator route...
        alone = await client.post(OPERATOR_WRITE, headers={"X-Session-Id": session_id})
        assert alone.status_code == 401, (
            "a manual session id must not satisfy require_operator_auth"
        )
        # ...the bearer token does, with or without it.
        assert (await client.post(OPERATOR_WRITE, headers=headers)).status_code == 200
        assert (await client.post(OPERATOR_WRITE)).status_code == 401


async def test_session_ids_are_random_and_not_derived_from_the_password(enforced):
    async with _client() as client:
        headers = await _login(client)
        first = (await _unlock(client, CRED, headers=headers)).json()["session_id"]
        second = (await _unlock(client, CRED, headers=headers)).json()["session_id"]
    assert first != second, "each unlock mints its own session"
    assert first.startswith("manual-") and len(first) == len("manual-") + 32  # 24 random bytes
    # The old id was sha256(sha256("password:<peer>:<password>")) truncated: anyone who saw one
    # (the camera stream URL carries it) could test password guesses against it offline.
    material = f"password:password:127.0.0.1:{CRED}"
    seed = "password:" + hashlib.sha256(material.encode()).hexdigest()
    old_style = "manual-" + hashlib.sha256(seed.encode()).hexdigest()[:16]
    assert old_style not in (first, second)


async def test_a_dedicated_manual_password_replaces_the_operator_credential(enforced, monkeypatch):
    monkeypatch.setenv("MANUAL_CONTROL_PASSWORD", "dedicated-manual-pass-1")
    async with _client() as client:
        headers = await _login(client)
        assert (await _unlock(client, CRED, headers=headers)).status_code == 401
        assert (
            await _unlock(client, "dedicated-manual-pass-1", headers=headers)
        ).status_code == 200


async def test_a_bcrypt_password_hash_replaces_both(enforced):
    hashed = enforced.password_manager.hash_password("hash-configured-secret")
    _security_settings.password_hash = hashed
    async with _client() as client:
        headers = await _login(client)
        assert (await _unlock(client, CRED, headers=headers)).status_code == 401
        assert (await _unlock(client, "hash-configured-secret", headers=headers)).status_code == 200


async def test_a_credential_that_vanished_unlocks_nothing(enforced):
    """Fail closed when nothing is configured: no hash, no MANUAL_CONTROL_PASSWORD, no match."""
    async with _client() as client:
        headers = await _login(client)  # a token issued while a credential existed
        enforced.operator_credential_hash = None
        for attempt in ("operator123", "admin", "anything-at-all", CRED):
            resp = await _unlock(client, attempt, headers=headers)
            assert resp.status_code == 401, (attempt, resp.text)
    assert _manual_control_sessions == {}


async def test_a_published_default_cannot_log_in_or_unlock(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.delenv("MANUAL_CONTROL_PASSWORD", raising=False)
    monkeypatch.setattr(_security_settings, "password_hash", None)
    _manual_control_sessions.clear()
    service = _install_service(monkeypatch, "operator123")  # configured, refused by the policy
    assert service.login_available() is False
    async with _client() as client:
        assert (await client.post(LOGIN, json={"credential": "operator123"})).status_code == 401
        assert (await _unlock(client, "operator123")).status_code == 401
    assert _manual_control_sessions == {}


@pytest.mark.parametrize(
    "level", [SecurityLevel.PASSWORD, SecurityLevel.TOTP, SecurityLevel.TUNNEL_AUTH]
)
async def test_a_blank_password_is_refused_at_every_security_level(enforced, monkeypatch, level):
    monkeypatch.setattr(_security_settings, "security_level", level)
    async with _client() as client:
        headers = await _login(client)
        for blank in (None, "", "   "):
            resp = await _unlock(client, blank, headers=headers)
            assert resp.status_code == 400, (level, blank, resp.text)
    assert _manual_control_sessions == {}


# ---- Cloudflare Access: decoded, never verified, so never trusted ----------------------------


async def test_a_forged_cloudflare_assertion_unlocks_nothing(enforced):
    forged = {"CF-Access-Jwt-Assertion": _forged_cloudflare_jwt()}
    async with _client() as client:
        headers = await _login(client)
        # Even a logged-in operator cannot unlock with a header: it is not a credential.
        resp = await _unlock(client, None, headers={**headers, **forged}, method="cloudflare")
        assert resp.status_code == 501, resp.text
        # The status probe of the Control view's Cloudflare gate no longer authorizes either.
        status = await client.get(STATUS, headers=forged)
        assert status.status_code == 200
        assert status.json()["authorized"] is False
        assert status.json().get("session_id") is None
        cookie = await client.get(STATUS, cookies={"CF_Authorization": _forged_cloudflare_jwt()})
        assert cookie.json()["authorized"] is False
        # And without a bearer token the unlock never gets as far as looking at the header.
        anonymous = await _unlock(client, None, headers=forged, method="cloudflare")
        assert anonymous.status_code == 401
    assert _manual_control_sessions == {}


async def test_a_tunnel_security_level_does_not_make_a_header_an_identity(enforced, monkeypatch):
    monkeypatch.setattr(
        "backend.src.api.routers.auth.get_security_level", lambda: SecurityLevel.TUNNEL_AUTH
    )
    async with _client() as client:
        headers = await _login(client)
        resp = await client.post(
            UNLOCK,
            json={},
            headers={**headers, "CF-Access-Jwt-Assertion": _forged_cloudflare_jwt()},
        )
        assert resp.status_code == 501, resp.text  # method defaults to cloudflare at this level
    assert _manual_control_sessions == {}


# ---- throttling: per TCP peer, never per client-supplied id -----------------------------------


async def test_wrong_unlock_passwords_lock_the_peer_out_whatever_client_id_it_sends(
    enforced, monkeypatch
):
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "3")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "60")
    async with _client() as client:
        headers = await _login(client)
        for attempt in range(3):  # a fresh X-Client-Id each time used to reset everything
            resp = await _unlock(
                client, f"wrong-{attempt}", headers={**headers, "X-Client-Id": f"rot-{attempt}"}
            )
            assert resp.status_code == 401
        locked = await _unlock(client, CRED, headers={**headers, "X-Client-Id": "yet-another"})
        assert locked.status_code == 429, locked.text
        assert int(locked.headers["Retry-After"]) >= 1
    assert _manual_control_sessions == {}

    # Another peer is unaffected.
    async with _client(("198.51.100.7", 4242)) as other:
        other_headers = await _login(other)
        assert (await _unlock(other, CRED, headers=other_headers)).status_code == 200


async def test_a_success_clears_the_failure_count(enforced, monkeypatch):
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "3")
    async with _client() as client:
        headers = await _login(client)
        for attempt in range(2):
            assert (await _unlock(client, f"wrong-{attempt}", headers=headers)).status_code == 401
        assert (await _unlock(client, CRED, headers=headers)).status_code == 200  # resets
        for attempt in range(2):  # two more wrong ones are still below the limit of 3
            assert (await _unlock(client, f"again-{attempt}", headers=headers)).status_code == 401
        assert (await _unlock(client, CRED, headers=headers)).status_code == 200


async def test_a_wrong_totp_code_counts_against_the_same_throttle(enforced, monkeypatch):
    monkeypatch.setattr(
        _security_settings,
        "totp_config",
        TOTPConfig(secret="JBSWY3DPEHPK3PXP", backup_codes=[]),
    )
    monkeypatch.setenv("AUTH_LOCKOUT_FAILURES", "3")
    monkeypatch.setenv("AUTH_LOCKOUT_SECONDS", "60")
    async with _client() as client:
        headers = await _login(client)
        for attempt in range(3):
            resp = await client.post(
                UNLOCK,
                json={"method": "totp", "totp_code": f"00000{attempt}", "password": CRED},
                headers=headers,
            )
            assert resp.status_code == 401, resp.text
        resp = await client.post(
            UNLOCK,
            json={"method": "totp", "totp_code": "123456", "password": CRED},
            headers=headers,
        )
        assert resp.status_code == 429
    assert _manual_control_sessions == {}


# ---- the session store itself --------------------------------------------------------------


async def test_the_session_store_is_bounded_and_expired_sessions_are_not_honoured(enforced):
    expiry = datetime.now(UTC) + timedelta(minutes=5)
    for _ in range(MANUAL_SESSION_LIMIT + 10):
        _issue_manual_session(expiry, "test")
    assert len(_manual_control_sessions) == MANUAL_SESSION_LIMIT

    live = _issue_manual_session(expiry, "live")
    assert _resolve_manual_session(live["session_id"])["principal"] == "live"
    stale = _issue_manual_session(datetime.now(UTC) - timedelta(seconds=1), "stale")
    for bad in (stale["session_id"], "manual-never-issued", "", None):
        with pytest.raises(HTTPException) as refused:
            _resolve_manual_session(bad)
        assert refused.value.status_code == 401, bad
