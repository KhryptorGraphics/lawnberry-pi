"""The documented ``/control/blade`` body drives the IBT-4, and the answer is what the pads read.

Review 3, HW-02.

``docs/OPERATIONS.md`` documents ``{"active": true|false}``. A body without a ``session_id`` was
caught by a legacy branch (written for integration tests) that answered ``200 DISABLED`` /
``ENABLED`` and never touched the IBT-4: a client that switched the blade OFF was told it was off
while the pad stayed live, and an ON was told it was on while nothing was driven. Now every body
goes through one strict parser and the one hardware path. A blade OFF is the safe direction: always
driven, never refused, no manual session needed. A blade ON needs the manual-control session (the
additional factor of the blade and drive routes) whenever operator auth is on.

The IBT-4 is the real armed adapter on a FAKE ``lgpio``; ``fake.pad`` is what a meter on GPIO24/25
would show; the app and its auth are the real ones.
"""

from __future__ import annotations

import itertools
import sys

import httpx
import pytest
from _tractor_fakes import idle_tractor
from fake_lgpio import FakeLgpio

from backend.src.api import rest as rest_module
from backend.src.core.globals import _blade_state, _client_emergency, _safety_state
from backend.src.services import blade_service
from backend.src.services.blade_service import BladeService

OFF = {24: 0, 25: 0}
ON = {24: 1, 25: 0}
URL = "/api/v2/control/blade"
_clients = itertools.count(1)


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    monkeypatch.setattr(
        "backend.src.services.tractor_service.get_tractor_service", lambda: idle_tractor()
    )
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)
    _safety_state["emergency_stop_active"] = False
    yield
    _safety_state["emergency_stop_active"] = False
    _blade_state["active"] = False
    rest_module._emergency_until = 0.0
    rest_module._legacy_motors_active = False
    _client_emergency.clear()


@pytest.fixture
def rig(monkeypatch):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = BladeService({"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}, "deadman_ms": 10000})
    monkeypatch.setattr(blade_service, "_blade_service", svc)
    return svc, fake


def client(headers: dict | None = None) -> httpx.AsyncClient:
    from backend.src.main import app

    n = next(_clients)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Forwarded-For": f"10.8.{n // 250}.{n % 250 + 1}", **(headers or {})},
    )


# ------------------------------------------------------------- the documented body acts


@pytest.mark.parametrize(
    ("body", "pad", "word"),
    [
        ({"active": True}, ON, None),
        ({"command": "blade_enable"}, ON, "ENABLED"),
        ({"action": "enable"}, ON, None),
    ],
)
async def test_a_blade_on_without_a_session_id_really_switches_the_pad(rig, body, pad, word):
    _svc, fake = rig
    async with client() as c:
        resp = await c.post(URL, json=body)

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["accepted"] is True and answer["result"] == "accepted"
    assert fake.pad == pad, "the answer said ON and the pad was never driven"
    assert answer["blade_state"]["commanded"] == "on"
    assert answer["blade_state"]["pad"] == {"in1": 1, "in2": 0}, "and says what the pads read"
    assert answer.get("blade_status") == word


@pytest.mark.parametrize(
    ("body", "word"),
    [
        ({"active": False}, None),
        ({"command": "blade_disable"}, "DISABLED"),
        ({"action": "disable"}, None),
    ],
)
async def test_a_blade_off_without_a_session_id_really_switches_the_pad_off(rig, body, word):
    _svc, fake = rig
    async with client() as c:
        assert (await c.post(URL, json={"active": True})).status_code == 200
        assert fake.pad == ON

        resp = await c.post(URL, json=body)  # used to answer 200 DISABLED over a live pad

    assert resp.status_code == 200, resp.text
    assert resp.json()["accepted"] is True
    assert fake.pad == OFF, "the blade input stayed live under an answer that said off"
    assert resp.json().get("blade_status") == word


async def test_an_off_that_the_pads_do_not_confirm_is_not_answered_disabled(rig, monkeypatch):
    svc, fake = rig
    async with client() as c:
        assert (await c.post(URL, json={"command": "blade_enable"})).status_code == 200
        monkeypatch.setattr(svc, "_drive_off", lambda *_a, **_k: False)  # the pad will not go off
        svc.last_refusal = None

        resp = await c.post(URL, json={"command": "blade_disable"})

    assert resp.status_code == 409 and resp.json()["accepted"] is False
    assert resp.json()["blade_status"] != "DISABLED", "an unconfirmed OFF is never 'DISABLED'"
    assert fake.pad == ON


@pytest.mark.parametrize(
    "body",
    [
        {"command": "blade_enable", "active": False},
        {"command": "blade_disable", "active": True},
        {"action": "enable", "command": "blade_disable"},
        {"command": "blade_enable", "active": "false"},
    ],
)
async def test_a_body_whose_fields_disagree_is_refused_and_drives_nothing(rig, body):
    _svc, fake = rig
    async with client() as c:
        resp = await c.post(URL, json=body)

    assert resp.status_code == 422, resp.text
    assert fake.calls == [], "nothing was claimed or driven for an ambiguous body"


# ---------------------------------------------------- the legacy interlocks still refuse an ON


async def test_a_sessionless_on_is_still_blocked_in_an_emergency_window_and_drives_nothing(rig):
    _svc, fake = rig
    async with client({"X-Client-Id": "alice"}) as c:
        await c.post("/api/v2/control/emergency", headers={"X-Client-Id": "alice"})
        rest_module._emergency_until = 0.0  # keep only the per-client flag
        calls_after_stop = len(fake.calls)

        on = await c.post(URL, json={"active": True}, headers={"X-Client-Id": "alice"})
        off = await c.post(URL, json={"active": False}, headers={"X-Client-Id": "alice"})

    assert on.status_code == 403 and "emergency_stop_active" in on.json()["detail"]
    assert fake.pad == OFF and len(fake.calls) >= calls_after_stop
    assert off.status_code == 200, "an OFF is never refused"


async def test_a_sessionless_on_is_refused_while_a_sessionless_drive_is_in_force(rig):
    _svc, fake = rig
    async with client() as c:
        drive = await c.post("/api/v2/control/drive", json={"throttle": 0.3, "turn": 0.0})
        assert drive.status_code == 200
        on = await c.post(URL, json={"command": "blade_enable"})

    assert on.status_code == 403 and "motors_active" in on.json()["detail"]
    assert fake.pad.get(24, 0) == 0, "a refused ON never drove the pad"


# ------------------------------------------------------ with operator auth on (production)


async def _login(c, monkeypatch, operator_credential) -> dict[str, str]:
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    login = await c.post("/api/v2/auth/login", json={"credential": operator_credential})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _unlock(c, operator, operator_credential) -> str:
    unlock = await c.post(
        "/api/v2/control/manual-unlock",
        json={"method": "password", "password": operator_credential},
        headers=operator,
    )
    assert unlock.status_code == 200, unlock.text
    return unlock.json()["session_id"]


async def test_with_auth_on_an_on_needs_the_manual_session_and_an_off_never_does(
    rig, monkeypatch, operator_credential
):
    _svc, fake = rig
    async with client() as c:
        operator = await _login(c, monkeypatch, operator_credential)

        bare_on = await c.post(URL, json={"active": True}, headers=operator)
        assert bare_on.status_code == 401, bare_on.text  # fail closed: no second factor, no blade
        assert fake.pad.get(24, 0) == 0 and fake.calls == []

        session = await _unlock(c, operator, operator_credential)
        on = await c.post(URL, json={"active": True, "session_id": session}, headers=operator)
        assert on.status_code == 200 and fake.pad == ON

        # an OFF carrying an expired or unknown id, or none at all, still switches the pad off
        stale = await c.post(
            URL, json={"active": False, "session_id": "expired-or-unknown"}, headers=operator
        )
        assert stale.status_code == 200, stale.text
        assert fake.pad == OFF, "the blade kept running because its OFF carried a dead session"

        assert (
            await c.post(URL, json={"active": True, "session_id": session}, headers=operator)
        ).status_code == 200 and fake.pad == ON
        bare_off = await c.post(URL, json={"command": "blade_disable"}, headers=operator)
        assert bare_off.status_code == 200 and fake.pad == OFF


async def test_the_session_id_form_keeps_working_with_auth_on(
    rig, monkeypatch, operator_credential
):
    _svc, fake = rig
    async with client() as c:
        operator = await _login(c, monkeypatch, operator_credential)
        session = await _unlock(c, operator, operator_credential)

        on = await c.post(URL, json={"session_id": session, "action": "enable"}, headers=operator)
        bad = await c.post(URL, json={"session_id": "nope", "active": True}, headers=operator)

    assert on.status_code == 200 and on.json()["accepted"] is True
    assert bad.status_code == 401, "an ON with a session id that is not the server's is refused"
    assert fake.pad == ON
