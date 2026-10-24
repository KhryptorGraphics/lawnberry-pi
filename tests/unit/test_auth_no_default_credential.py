"""There is no built-in operator credential, and the signing key is a provisioned secret.

Before: ``AuthService`` and ``/api/v1/auth/login`` fell back to the public ``operator123`` when
``LAWN_BERRY_OPERATOR_CREDENTIAL`` was unset. Now an unset, empty or published-default
credential disables operator login (fail closed; the E-stop routes are unauthenticated by
design and unaffected), and ``LAWN_BERRY_AUTH_SECRET`` is what makes sessions survive a restart.

Outside SIM_MODE the service validates for real, so these tests construct services with
``SIM_MODE=0`` (the rest of the app stays simulated, as in test_login_credentials.py).
"""

from __future__ import annotations

import logging

import httpx
import pytest

from backend.src.main import app
from backend.src.services import auth_service as auth_module
from backend.src.services.auth_service import AuthenticationError, AuthService, JWTManager

STRONG = "correct-horse-battery-staple"
REMOVED_PUBLIC_DEFAULT = "operator123"


@pytest.fixture
def real_auth(monkeypatch):
    """Non-simulated environment with no credential configured."""
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
    monkeypatch.delenv("LAWN_BERRY_AUTH_SECRET", raising=False)
    monkeypatch.setattr(auth_module, "_warned", set())
    return monkeypatch


async def _rejected(service: AuthService, credential: str) -> bool:
    with pytest.raises(AuthenticationError) as excinfo:
        await service.authenticate(credential)
    return excinfo.value.status_code == 401


async def test_no_credential_configured_nobody_logs_in(real_auth, caplog):
    with caplog.at_level(logging.CRITICAL, logger=auth_module.logger.name):
        service = AuthService()
    assert service.operator_credential_hash is None
    assert "operator login is DISABLED" in caplog.text
    for attempt in (REMOVED_PUBLIC_DEFAULT, "admin", "password", "anything", ""):
        assert await _rejected(service, attempt), attempt


async def test_an_empty_credential_is_not_a_credential(real_auth):
    real_auth.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", "")
    service = AuthService()
    assert service.operator_credential_hash is None
    assert await _rejected(service, "")
    assert await _rejected(service, REMOVED_PUBLIC_DEFAULT)


@pytest.mark.parametrize("default", ["operator123", "Operator123!", "admin", "changeme"])
async def test_a_published_default_is_refused_even_when_explicitly_configured(
    real_auth, caplog, default
):
    real_auth.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", default)
    with caplog.at_level(logging.CRITICAL, logger=auth_module.logger.name):
        service = AuthService()
    assert service.operator_credential_hash is None
    assert "operator login is DISABLED" in caplog.text
    assert default not in caplog.text, "the credential must not be logged"
    assert await _rejected(service, default), "configuring a default must not enable it"


async def test_only_the_configured_strong_credential_logs_in(real_auth):
    real_auth.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", STRONG)
    service = AuthService()
    assert service.operator_credential_hash and service.operator_credential_hash != STRONG
    assert await _rejected(service, REMOVED_PUBLIC_DEFAULT)
    assert await _rejected(service, STRONG + "x")
    result = await service.authenticate(STRONG)
    session = await service.verify_token(result.token)
    assert session is not None and session.session_id == result.session.session_id


async def test_an_explicit_argument_beats_the_environment(real_auth):
    real_auth.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", STRONG)
    service = AuthService(operator_credential="another-strong-credential-7")
    assert await _rejected(service, STRONG)
    assert (await service.authenticate("another-strong-credential-7")).token


async def test_sim_mode_alone_does_not_accept_any_credential(monkeypatch):
    """SIM_MODE=1 is not an authentication bypass (SEC-10): it needs the explicit opt-in too."""
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", STRONG)
    service = AuthService()
    assert await _rejected(service, "whatever")
    assert await _rejected(service, REMOVED_PUBLIC_DEFAULT)
    assert (await service.authenticate(STRONG)).token, "the configured credential still works"
    monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL")
    assert await _rejected(AuthService(), "whatever"), "and with none configured nobody logs in"


async def test_the_simulator_opt_in_accepts_any_non_empty_credential(monkeypatch):
    """Simulated runs that opt in (SIM_MODE=1 and LAWN_BERRY_SIM_OPEN_AUTH=1) need no credential."""
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
    monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
    service = AuthService()
    assert service.login_available() is True
    assert (await service.authenticate("whatever")).token
    assert await _rejected(service, "")


@pytest.mark.parametrize(
    "env",
    [
        {"SIM_MODE": "0", "LAWN_BERRY_SIM_OPEN_AUTH": "1"},  # the opt-in needs SIM_MODE
        {"SIM_MODE": "1", "LAWN_BERRY_SIM_OPEN_AUTH": "1", "LAWNBERRY_ENV": "production"},
        {"SIM_MODE": "1", "LAWN_BERRY_SIM_OPEN_AUTH": "maybe"},  # not a true token
        {"SIM_MODE": "true", "LAWN_BERRY_SIM_OPEN_AUTH": "1"},  # SIM_MODE is exactly "1"
    ],
)
async def test_the_simulator_opt_in_has_no_effect_unless_every_condition_holds(monkeypatch, env):
    monkeypatch.delenv("LAWNBERRY_ENV", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
    service = AuthService()
    assert service.login_available() is False
    assert await _rejected(service, "whatever")


async def test_the_credential_cannot_be_set_over_the_api_when_none_is_configured(real_auth):
    service = AuthService()
    assert await service.update_operator_credential(REMOVED_PUBLIC_DEFAULT, STRONG) is False
    assert service.operator_credential_hash is None


async def test_the_credential_can_be_rotated_when_the_current_one_is_known(real_auth):
    real_auth.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", STRONG)
    service = AuthService()
    assert await service.update_operator_credential("wrong-current-credential", "x") is False
    assert await service.update_operator_credential(STRONG, "a-brand-new-credential-9") is True
    assert await _rejected(service, STRONG)
    assert (await service.authenticate("a-brand-new-credential-9")).token


# ---- the HTTP login routes -----------------------------------------------------------------


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _install_v1_service(monkeypatch, credential: str | None) -> AuthService:
    """A strict service for the v1 login (it logs in through the service, like v2)."""
    monkeypatch.setenv("SIM_MODE", "0")
    if credential is None:
        monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
    else:
        monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", credential)
    service = AuthService()
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setattr("backend.src.api.rest_v1._auth_service", service)
    return service


async def test_v1_login_has_no_default_credential(monkeypatch):
    _install_v1_service(monkeypatch, None)
    async with _client() as client:
        for body in (
            {"credential": REMOVED_PUBLIC_DEFAULT},
            {"username": "operator", "password": REMOVED_PUBLIC_DEFAULT},
            {"credential": "anything-at-all-12"},
        ):
            resp = await client.post("/api/v1/auth/login", json=body)
            assert resp.status_code == 401, (body, resp.text)


async def test_v1_login_accepts_exactly_the_configured_credential(monkeypatch):
    _install_v1_service(monkeypatch, STRONG)
    async with _client() as client:
        ok = await client.post("/api/v1/auth/login", json={"credential": STRONG})
        old = await client.post("/api/v1/auth/login", json={"credential": REMOVED_PUBLIC_DEFAULT})
    assert ok.status_code == 200, ok.text
    assert old.status_code == 401


async def test_v1_login_ignores_the_simulator_opt_in(monkeypatch):
    """v2 login honours SIM_MODE + LAWN_BERRY_SIM_OPEN_AUTH; the v1 contract never does."""
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
    monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
    service = AuthService()
    assert service.login_available() is True  # the opt-in is on...
    monkeypatch.setattr("backend.src.api.rest_v1._auth_service", service)
    async with _client() as client:
        resp = await client.post("/api/v1/auth/login", json={"credential": "whatever"})
    assert resp.status_code == 401  # ...and v1 still refuses an unconfigured credential


# ---- the signing key -----------------------------------------------------------------------


def _token_from(manager: JWTManager) -> str:
    from backend.src.models import UserRole

    token, _ = manager.create_token(session_id="s1", user_id="u1", role=UserRole.OPERATOR)
    return token


def test_a_provisioned_signing_secret_survives_a_restart(real_auth):
    real_auth.setenv("LAWN_BERRY_AUTH_SECRET", "a1b2c3d4" * 8)
    before_restart = _token_from(JWTManager())
    after_restart = JWTManager()  # a new process builds a new manager from the same .env
    payload = after_restart.verify_token(before_restart)
    assert payload is not None and payload["sid"] == "s1"


def test_without_a_provisioned_secret_every_restart_ends_all_sessions(real_auth):
    before_restart = _token_from(JWTManager())
    assert JWTManager().verify_token(before_restart) is None


def test_an_ephemeral_signing_key_is_reported_once_per_process(real_auth, caplog):
    with caplog.at_level(logging.WARNING, logger=auth_module.logger.name):
        JWTManager()
        JWTManager()
    hits = [r for r in caplog.records if "LAWN_BERRY_AUTH_SECRET is not set" in r.getMessage()]
    assert len(hits) == 1
    assert "install_services.sh" in hits[0].getMessage()


def test_a_weak_signing_secret_is_reported(real_auth, caplog):
    real_auth.setenv("LAWN_BERRY_AUTH_SECRET", "hunter2")
    with caplog.at_level(logging.WARNING, logger=auth_module.logger.name):
        manager = JWTManager()
    assert manager.verify_token(_token_from(manager)) is not None  # still works, but is flagged
    assert "shorter than 32" in caplog.text
    assert "hunter2" not in caplog.text, "the secret must not be logged"


def test_a_good_signing_secret_and_sim_mode_are_quiet(real_auth, caplog):
    real_auth.setenv("LAWN_BERRY_AUTH_SECRET", "a1b2c3d4" * 8)
    with caplog.at_level(logging.WARNING, logger=auth_module.logger.name):
        JWTManager()
    assert caplog.text == ""
    real_auth.delenv("LAWN_BERRY_AUTH_SECRET")
    real_auth.setenv("SIM_MODE", "1")
    with caplog.at_level(logging.WARNING, logger=auth_module.logger.name):
        JWTManager()
    assert caplog.text == ""
