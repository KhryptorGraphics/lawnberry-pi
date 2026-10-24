"""``OPERATOR_AUTH_REQUIRED`` fails closed, auth-disabled states are loud, and SIM_MODE is not a
credential bypass (SEC-09, SEC-10).

* Only ``0``/``false``/``no``/``off`` disable operator auth. ``true``, ``TRUE``, an empty
  ``OPERATOR_AUTH_REQUIRED=`` line and `` 1`` used to disable it silently (``!= "1"``), and
  ``<root>/.env`` (EnvironmentFile) overrides the unit's value.
* Start-up logs CRITICAL whenever auth is disabled outside SIM_MODE, and for SIM_MODE on a
  production or non-loopback process; ``/health`` reports the state.
* SIM_MODE alone no longer admits any WebSocket client as an operator or any credential at login
  (that needs ``LAWN_BERRY_SIM_OPEN_AUTH=1`` as well, never in production); an unverified
  ``CF-Access-Jwt-Assertion`` header never does.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi import HTTPException

from backend.src.api.routers.auth import _authorize_websocket
from backend.src.core.auth_posture import (
    configured_bind_host,
    is_loopback_host,
    log_auth_posture,
    operator_auth_required,
    read_operator_auth_flag,
)
from backend.src.main import app

LOGGER = "backend.src.core.auth_posture"
UVICORN = ("uvicorn", "backend.src.main:app", "--host", "127.0.0.1")


@pytest.mark.parametrize("value", [None, "1", "true", "TRUE", "yes", "on", " 1 ", "True"])
def test_explicit_true_tokens_and_unset_keep_auth_on(value):
    env = {} if value is None else {"OPERATOR_AUTH_REQUIRED": value}
    assert operator_auth_required(env) is True
    assert read_operator_auth_flag(env).recognised is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " 0 ", "Off"])
def test_only_explicit_false_tokens_disable_auth(value):
    assert operator_auth_required({"OPERATOR_AUTH_REQUIRED": value}) is False


@pytest.mark.parametrize("value", ["", "  ", "2", "disabled", "maybe", "y", "n", "none"])
def test_ambiguous_values_keep_auth_on_and_are_flagged(value):
    flag = read_operator_auth_flag({"OPERATOR_AUTH_REQUIRED": value})
    assert flag.required is True and flag.recognised is False


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize(
    "value", ["true", "TRUE", "yes", "on", "", "  ", " 1", "2", "disabled", "maybe", "1"]
)
async def test_a_write_route_refuses_anonymous_callers_for_every_non_false_value(
    monkeypatch, value
):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", value)
    async with _client() as client:
        resp = await client.post("/api/v2/tractor/clear-emergency")
    assert resp.status_code == 401, (value, resp.text)


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " 0 "])
async def test_explicit_false_still_opens_the_routes_for_the_suite(monkeypatch, value):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", value)
    async with _client() as client:
        resp = await client.post("/api/v2/tractor/clear-emergency")
    assert resp.status_code == 200, (value, resp.text)


async def test_health_reports_the_auth_state(monkeypatch):
    async with _client() as client:
        monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
        off = (await client.get("/health")).json()["operator_auth"]
        monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "maybe")
        ambiguous = (await client.get("/health")).json()["operator_auth"]
        monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
        on = (await client.get("/api/v2/health")).json()["operator_auth"]
    assert off["operator_auth_required"] is False
    assert ambiguous["operator_auth_required"] is True
    assert ambiguous["operator_auth_flag_ambiguous"] is True
    assert on["operator_auth_required"] is True and on["operator_auth_flag_ambiguous"] is False
    assert set(on) == {
        "operator_auth_required",
        "operator_auth_flag_ambiguous",
        "operator_login_available",
        "simulation_open_auth",
    }


# ---- the start-up log ----------------------------------------------------------------------


def _records(caplog, environ, argv=UVICORN, *, pytest_process=False):
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        log_auth_posture(argv=argv, environ=environ, pytest_process=pytest_process)
    return [(r.levelname, r.getMessage()) for r in caplog.records if r.name == LOGGER]


def _critical(records):
    return [message for level, message in records if level == "CRITICAL"]


def test_auth_disabled_outside_sim_mode_is_critical(caplog):
    records = _records(caplog, {"OPERATOR_AUTH_REQUIRED": "0"})
    assert len(_critical(records)) == 1
    assert "DISABLES operator authentication" in _critical(records)[0]
    assert "'0'" in _critical(records)[0]


@pytest.mark.parametrize("value", ["false", "no", "off"])
def test_every_false_token_outside_sim_mode_is_critical(caplog, value):
    assert _critical(_records(caplog, {"OPERATOR_AUTH_REQUIRED": value}))


def test_auth_disabled_in_sim_mode_is_a_warning_not_critical(caplog):
    records = _records(caplog, {"OPERATOR_AUTH_REQUIRED": "0", "SIM_MODE": "1"})
    assert _critical(records) == []
    assert any(level == "WARNING" and "DISABLED" in msg for level, msg in records)


def test_an_ambiguous_value_is_a_warning_and_stays_enforced(caplog):
    records = _records(caplog, {"OPERATOR_AUTH_REQUIRED": "maybe"})
    assert _critical(records) == []
    assert any("not a recognised boolean" in msg and "ENFORCED" in msg for _, msg in records)


@pytest.mark.parametrize("environ", [{}, {"OPERATOR_AUTH_REQUIRED": "1"}, {"SIM_MODE": "0"}])
def test_a_normal_production_configuration_is_silent(caplog, environ):
    assert _records(caplog, environ) == []


@pytest.mark.parametrize(
    ("argv", "environ", "critical"),
    [
        (("uvicorn", "x:app", "--host", "0.0.0.0"), {}, True),
        (("uvicorn", "x:app", "--host=0.0.0.0"), {}, True),
        (("uvicorn", "x:app", "--host", "192.168.1.20"), {}, True),
        (("uvicorn", "x:app"), {"UVICORN_HOST": "0.0.0.0"}, True),
        (("python", "run_sim.py"), {}, True),  # launcher unknown: cannot rule exposure out
        (("uvicorn", "x:app", "--host", "127.0.0.1"), {}, False),
        (("uvicorn", "x:app", "--host=localhost"), {}, False),
        (("uvicorn", "x:app", "--host", "::1"), {}, False),
        (("uvicorn", "x:app"), {}, False),  # uvicorn's default bind is the loopback
    ],
)
def test_sim_mode_on_a_non_loopback_process_is_critical(caplog, argv, environ, critical):
    records = _records(caplog, {"SIM_MODE": "1", **environ}, argv)
    assert bool(_critical(records)) is critical, (argv, environ, records)


def test_sim_mode_under_pytest_is_not_reported(caplog):
    argv = ("uvicorn", "x:app", "--host", "0.0.0.0")
    assert _records(caplog, {"SIM_MODE": "1"}, argv, pytest_process=True) == []


def test_sim_mode_in_a_production_environment_is_critical(caplog):
    records = _records(caplog, {"SIM_MODE": "1", "LAWNBERRY_ENV": "production"})
    assert any("LAWNBERRY_ENV=production" in msg for msg in _critical(records))


@pytest.mark.parametrize(
    ("environ", "argv", "level", "fragment"),
    [
        (
            {"SIM_MODE": "1", "LAWN_BERRY_SIM_OPEN_AUTH": "1"},
            ("uvicorn", "x", "--host", "0.0.0.0"),
            "CRITICAL",
            "ANY non-empty credential",
        ),
        (
            {"SIM_MODE": "1", "LAWN_BERRY_SIM_OPEN_AUTH": "1", "LAWNBERRY_ENV": "production"},
            UVICORN,
            "CRITICAL",
            "IGNORED",
        ),
        ({"SIM_MODE": "0", "LAWN_BERRY_SIM_OPEN_AUTH": "1"}, UVICORN, "WARNING", "no effect"),
        (
            {"SIM_MODE": "1", "LAWN_BERRY_SIM_OPEN_AUTH": "1"},
            UVICORN,
            "WARNING",
            "any non-empty credential logs in",
        ),
    ],
)
def test_the_open_auth_opt_in_is_reported(caplog, environ, argv, level, fragment):
    records = _records(caplog, environ, argv)
    assert any(lv == level and fragment in msg for lv, msg in records), records


def test_bind_host_and_loopback_helpers():
    assert configured_bind_host(["uvicorn", "x", "--host", "10.0.0.5"], {}) == "10.0.0.5"
    assert configured_bind_host(["python", "app.py"], {}) is None
    assert (
        is_loopback_host("127.0.0.1")
        and is_loopback_host("[::1]")
        and is_loopback_host("LocalHost")
    )
    assert (
        not is_loopback_host("0.0.0.0")
        and not is_loopback_host("::")
        and not is_loopback_host(None)
    )


# ---- WebSocket helper: SIM_MODE and Cloudflare headers are not credentials -------------------


class FakeWebSocket:
    def __init__(self, host="203.0.113.9", headers=None, query=None):
        self.client = (host, 4321)
        self.headers = headers or {}
        self.query_params = query or {}


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("SIM_MODE", "LAWN_BERRY_SIM_OPEN_AUTH", "LAWNBERRY_ENV"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


async def test_sim_mode_alone_does_not_admit_a_remote_websocket(clean_env):
    clean_env.setenv("SIM_MODE", "1")
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(FakeWebSocket())
    assert refused.value.status_code == 401


async def test_the_simulator_opt_in_admits_a_remote_websocket(clean_env):
    clean_env.setenv("SIM_MODE", "1")
    clean_env.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
    session = await _authorize_websocket(FakeWebSocket())
    assert session.session_id


async def test_the_opt_in_is_ignored_in_production(clean_env):
    clean_env.setenv("SIM_MODE", "1")
    clean_env.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
    clean_env.setenv("LAWNBERRY_ENV", "production")
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(FakeWebSocket())
    assert refused.value.status_code == 401


@pytest.mark.parametrize("sim", ["0", "1"])
async def test_an_unverified_cloudflare_header_is_not_an_operator(clean_env, sim):
    clean_env.setenv("SIM_MODE", sim)
    forged = FakeWebSocket(headers={"CF-Access-Jwt-Assertion": "eyJhbGciOiJub25lIn0.e30.forged"})
    with pytest.raises(HTTPException) as refused:
        await _authorize_websocket(forged)
    assert refused.value.status_code == 401


async def test_a_loopback_websocket_keeps_working(clean_env):
    assert (await _authorize_websocket(FakeWebSocket(host="127.0.0.1"))).session_id
