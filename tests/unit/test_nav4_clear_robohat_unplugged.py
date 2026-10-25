"""``POST /control/emergency_clear`` never says cleared over a RoboHAT hold it could not release.

Review 3, NAV-08. The route asked the RoboHAT to clear its stop hold only when the service said
its serial port was connected, and listed ``robohat_hold`` only inside that branch. A RoboHAT that
is not connected (unplugged, not enumerated after a restart) was never asked and never listed: the
answer was ``EMERGENCY_CLEARED`` with ``latch_persisted: true`` while the hold was still set. The
latch file kept the hold, a restart restored a latch the operator had been told was gone, and every
run was refused with no hint in the clear's answer. The fail-safe direction was right; the report
was wrong.

The RoboHAT is the real service over a fake serial line (``FakeFirmware``); the route is the real
app.
"""

from __future__ import annotations

import httpx
from _nav_rig import robohat_with_timeline
from _tractor_fakes import FakeGpiozero, mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.safety import latch_store
from backend.src.services import emergency_stop_service as ess


async def _post_clear() -> httpx.Response:
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.post("/api/v2/control/emergency_clear", json={"confirmation": True})


async def test_a_hold_on_a_robohat_that_is_not_connected_is_listed_not_cleared(monkeypatch):
    svc, _fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    await ess.execute_emergency_stop("api")
    assert svc.emergency_latched and latch_store.load().has(latch_store.ROBOHAT_HOLD)
    svc.status.serial_connected = False  # the RoboHAT went away; nobody can ask it to clear

    response = await _post_clear()

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "EMERGENCY_PARTIALLY_CLEARED", body
    assert "robohat_hold" in body["still_latched"], body
    assert svc.emergency_latched is True, "the hold is still set: the answer must say so"
    assert latch_store.load().has(latch_store.ROBOHAT_HOLD), "and the file still holds it"
    refusal = _start_refusal()
    assert "RoboHAT stop hold" in refusal, "a run is refused for it, so the clear must say it"

    svc.status.serial_connected = True  # plugged back in: the next clear can release it
    again = (await _post_clear()).json()
    assert again["status"] == "EMERGENCY_CLEARED" and "still_latched" not in again, again
    assert svc.emergency_latched is False
    assert not latch_store.latch_path().exists(), "a real clear removes the file"
    await svc.shutdown()


async def test_a_robohat_that_is_not_connected_and_not_held_does_not_hold_up_a_clear(monkeypatch):
    svc, _fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    svc.status.serial_connected = False
    assert svc.emergency_latched is False
    core_globals._safety_state["emergency_stop_active"] = True  # only the process latch is set

    body = (await _post_clear()).json()

    assert body["status"] == "EMERGENCY_CLEARED" and "still_latched" not in body, body
    assert core_globals._safety_state["emergency_stop_active"] is False
    await svc.shutdown()


async def test_the_tractor_is_not_what_decides_the_robohat_listing(monkeypatch):
    """A tractor platform next to a held, disconnected RoboHAT: both latches are listed."""
    svc, _fw, _idle, _tl = robohat_with_timeline(monkeypatch)  # SIM_MODE off: real relays
    FakeGpiozero(monkeypatch)
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    await ess.execute_emergency_stop("api", tractor=tractor)
    svc.status.serial_connected = False

    body = (await _post_clear()).json()

    assert body["status"] == "EMERGENCY_PARTIALLY_CLEARED", body
    assert "robohat_hold" in body["still_latched"], body
    assert "tractor_emergency_stop" in body["still_latched"], body
    await tractor.shutdown()
    await svc.shutdown()


def _start_refusal() -> str:
    from backend.src.services.navigation_service import NavigationService

    return NavigationService()._stop_latched() or ""
