"""``/control/blade`` and ``/control/emergency_clear`` tell the truth about the IBT-4 (round 2).

PR #20 review BL-02 (``bool("false")`` switched the blade ON), BL-03 (an audit failure after the
hardware write was answered as "blocked"), BL-04 (the general clear said EMERGENCY_CLEARED over an
unconfirmed blade OFF) and the new read-only ``GET /control/blade/status``. The IBT-4 is the real
armed adapter on a FAKE ``lgpio``; ``fake.pad`` is what a meter on GPIO24/25 would show.
"""

from __future__ import annotations

import itertools
import sys
import time

import httpx
import pytest
from _tractor_fakes import idle_tractor
from fake_lgpio import FakeLgpio

from backend.src.api import rest as rest_module
from backend.src.core.globals import _blade_state, _client_emergency, _safety_state
from backend.src.core.persistence import persistence
from backend.src.services import blade_service
from backend.src.services.blade_service import BladeService

OFF = {24: 0, 25: 0}
ON = {24: 1, 25: 0}
BASE = "/api/v2/control"


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
    _client_emergency.clear()


@pytest.fixture
def rig(monkeypatch):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = BladeService({"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}})
    monkeypatch.setattr(blade_service, "_blade_service", svc)
    return svc, fake


_clients = itertools.count(1)


def client() -> httpx.AsyncClient:
    """A client with its own forwarded address: every test gets a fresh rate-limit bucket."""
    from backend.src.main import app

    n = next(_clients)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Forwarded-For": f"10.9.{n // 250}.{n % 250 + 1}"},
    )


async def post(c, body):
    return await c.post(f"{BASE}/blade", json={"session_id": "s", **body})


# ----------------------------------------------------------------------------- BL-02


@pytest.mark.parametrize(
    "bad", ["false", "0", "off", "no", "False", "true", 0, 1, None, [], {}], ids=repr
)
async def test_a_non_boolean_active_never_switches_the_blade_on(rig, bad):
    svc, fake = rig
    async with client() as c:
        resp = await post(c, {"active": bad})

    assert resp.status_code == 422, resp.text
    assert "boolean" in resp.json()["detail"]
    assert fake.calls == [], "nothing was claimed or driven for an ambiguous command"


async def test_a_disagreeing_action_and_active_is_refused(rig):
    _svc, fake = rig
    async with client() as c:
        resp = await post(c, {"action": "enable", "active": False})
    assert resp.status_code == 422 and "disagree" in resp.json()["detail"]
    assert fake.calls == []


async def test_real_booleans_and_action_words_still_drive_the_blade(rig):
    _svc, fake = rig
    async with client() as c:
        on = await post(c, {"active": True})
        assert on.status_code == 200 and on.json()["accepted"] is True and fake.pad == ON
        off = await post(c, {"action": "disable"})
        assert off.status_code == 200 and fake.pad == OFF
        again = await post(c, {"action": "start"})
        assert again.json()["accepted"] is True and fake.pad == ON
        done = await post(c, {"active": False})
        assert done.status_code == 200 and fake.pad == OFF


# ----------------------------------------------------------------------------- BL-03


async def test_a_failing_audit_store_cannot_turn_a_running_blade_into_a_blocked_answer(
    rig, monkeypatch
):
    svc, fake = rig

    def broken(*args, **kwargs):
        raise RuntimeError("audit db locked")

    monkeypatch.setattr(persistence, "add_audit_log", broken)
    async with client() as c:
        resp = await post(c, {"action": "enable"})

    body = resp.json()
    assert fake.pad == ON, "the blade really is on"
    assert resp.status_code == 200 and body["accepted"] is True and body["result"] == "accepted"
    assert body["audit_logged"] is False, "the missing audit record is reported, not hidden"
    assert body["blade_state"]["observed"] == "on" and body["blade_state"]["active"] is True
    assert (await svc.status())["commanded"] == "on"


async def test_a_refused_command_reports_why_and_the_real_pads(rig):
    svc, fake = rig
    await svc.initialize()
    fake.hold_pad(25, 1)  # something else holds IN2: the ON must not be written
    async with client() as c:
        resp = await post(c, {"active": True})
    body = resp.json()
    assert resp.status_code == 409 and body["accepted"] is False and body["result"] == "rejected"
    assert "could not be confirmed" in body["reason"]
    assert body["blade_state"]["observed"] == "mixed" and body["blade_state"]["active"] is True


async def test_a_blade_off_is_never_refused_but_an_on_is_blocked_in_an_emergency_window(rig):
    svc, fake = rig
    async with client() as c:
        assert (await post(c, {"active": True})).status_code == 200 and fake.pad == ON
        rest_module._emergency_until = time.time() + 30
        blocked = await post(c, {"active": True})
        assert blocked.status_code == 423 and blocked.json()["result"] == "blocked"
        off = await post(c, {"active": False})
        assert off.status_code == 200 and off.json()["accepted"] is True
    assert fake.pad == OFF


async def test_the_status_route_shows_commanded_versus_observed(rig):
    svc, fake = rig
    async with client() as c:
        await post(c, {"active": True})
        fake.hold_pad(24, 0)  # IN1 stops following: commanded on, but the pad no longer says so
        body = (await c.get(f"{BASE}/blade/status")).json()
    assert body["commanded"] == "on" and body["observed"] == "off"
    assert body["pad"] == {"in1": 0, "in2": 0}
    assert body["state_confirmed"] is False, "the pads no longer match what was commanded"
    assert body["deadman_ms"] == 2000 and body["armed"] is True


# ----------------------------------------------------------------------------- BL-04


async def test_the_general_clear_does_not_say_cleared_over_an_unconfirmed_blade_off(rig):
    svc, fake = rig
    await svc.initialize()
    assert await svc.set_active(True) is True
    fake.hold_pad(24, 1)  # the stop cannot bring IN1 low
    async with client() as c:
        stop = await c.post(f"{BASE}/emergency-stop")
        assert stop.status_code == 200
        assert stop.json()["subsystems"]["blade_service"]["status"] == "failed"

        clear = await c.post(f"{BASE}/emergency_clear", json={"confirmation": True})
        body = clear.json()
        assert clear.status_code == 200
        assert body["status"] == "EMERGENCY_PARTIALLY_CLEARED", body
        assert body["still_latched"][0].startswith("blade_service_latched"), body
        assert svc.emergency_latched is True

        fake.release_pad(24)
        again = await c.post(f"{BASE}/emergency_clear", json={"confirmation": True})
        assert again.json()["status"] == "EMERGENCY_CLEARED", again.json()
    assert svc.emergency_latched is False and fake.pad == OFF
