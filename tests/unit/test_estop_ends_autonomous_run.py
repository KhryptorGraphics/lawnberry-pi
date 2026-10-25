"""An E-stop ends the autonomous run it interrupts; a clear never restarts one (audit SAF-01, S1).

Before the fix an E-stop only latched the actuators. The mission loop (or the AI loop) went on
commanding the machine, was refused for as long as the latch held, and the first authenticated
clear let it drive off again although nobody had authorized anything. Three layers stop that;
each is exercised through the real services (only the hardware under them is faked):

(a) lever, throttle and blade commands from every non-operator source are refused while the
    tractor is not authorized (the ``operator`` / ``teleop`` sources keep their documented
    behaviour: a person's own drive needs no authorize());
(b) every safety stop (the E-stop orchestrator, the tractor's own E-stop, the tilt latch) ends
    the missions and the AI run, so there is nothing left that a clear could resume;
(c) a clear releases the latch and nothing else: motion after it needs a fresh authorize().
"""

from __future__ import annotations

import asyncio

import pytest
from _mission_rig import FAR, driving, new_mission, new_nav, robohat_platform, wire_mission
from _tractor_fakes import NEUTRAL, mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode
from backend.src.safety import stop_events
from backend.src.services import emergency_stop_service as ess

AUTONOMOUS_SOURCES = ["navigation", "ai", "thor_plan", "mission", "unspecified"]


# --------------------------------------------------------------------- (a) the authorization gate


@pytest.mark.parametrize("source", AUTONOMOUS_SOURCES)
async def test_an_unauthorized_tractor_refuses_every_autonomous_source(source):
    tractor, pca = mowing_tractor()
    tractor.revoke()
    before = list(pca.log)
    results = [
        await tractor.set_levers(0.5, 0.5, source=source),
        await tractor.set_throttle(0.7, source=source),
        await tractor.engage_blade(True, source=source),
    ]
    assert [r["status"] for r in results] == ["rejected"] * 3, results
    assert all("not authorized" in r["reason"] for r in results), results
    assert pca.log == before, "a refused command must not reach the servos"
    assert tractor.state.left_lever == 0.0 and tractor.state.blade_engaged is False
    await tractor.shutdown()


async def test_the_operator_source_keeps_driving_an_unauthorized_tractor():
    tractor, _pca = mowing_tractor()
    tractor.revoke()
    result = await tractor.set_levers(0.5, 0.5, source="operator")
    assert result["status"] == "ok" and tractor.state.left_lever == 0.5
    await tractor.shutdown()


# ----------------------------------------------------- (b) + (c) the tractor platform, end to end


async def test_an_estop_ends_the_mission_and_a_clear_does_not_restart_it(monkeypatch):
    tractor, pca = mowing_tractor()
    nav = new_nav()
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await asyncio.sleep(0.6)
        assert tractor.state.moving and tractor.state.blade_engaged, "the mission is mowing"

        await tractor.emergency_stop(source="api")
        assert service.mission_statuses[mission.id].status == "aborted"
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE

        await tractor.clear_emergency()  # the operator's authenticated clear
        assert tractor.state.authorized is False, "a clear never authorizes"
        landed = len(pca.log)
        await asyncio.sleep(0.8)  # four control ticks
        assert task.done(), "the mission loop must have ended with the stop"

        tractor.authorize()  # a person authorizes again ...
        await asyncio.sleep(0.8)
        # ... and the mission the stop ended does not come back
        moved = [(ch, us) for _, ch, us in pca.log[landed:] if us != NEUTRAL[ch]]
        assert moved == [], "the machine moved after the clear"
        assert tractor.state.moving is False and tractor.state.blade_engaged is False
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tractor.shutdown()


async def test_a_clear_never_authorizes_and_autonomous_commands_stay_refused(monkeypatch):
    tractor, pca = mowing_tractor()
    await tractor.emergency_stop(source="api")
    await tractor.clear_emergency()
    assert tractor.state.authorized is False
    before = list(pca.log)
    refused = await tractor.set_levers(0.5, 0.5, source="navigation")
    assert refused["status"] == "rejected" and refused["code"] == "not_authorized"
    assert pca.log == before
    await tractor.shutdown()


# ------------------------------------------------------- (b) + (c) the RoboHAT platform, end to end


async def test_an_estop_ends_a_mission_on_the_robohat_platform_and_a_clear_does_not_restart_it(
    monkeypatch,
):
    monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", False)
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await asyncio.sleep(0.8)
        assert driving(fw.sent), "the mission drives the differential platform"

        report = await ess.execute_emergency_stop("api")
        assert svc.emergency_latched and report["emergency_stop_active"] is True
        assert service.mission_statuses[mission.id].status == "aborted"

        assert await svc.clear_emergency() is True  # the authenticated clear
        sent = len(fw.sent)
        await asyncio.sleep(1.0)
        assert task.done(), "the mission loop must have ended with the stop"
        assert driving(fw.sent[sent:]) == [], "the platform moved after the clear"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await svc.shutdown()


# ------------------------------------------------------------------ the AI loop and run starts


async def test_a_safety_stop_ends_the_ai_run():
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    nav._ai_task = asyncio.create_task(asyncio.sleep(30))  # the AI inference loop's task
    try:
        stop_events.record_stop("emergency_stop", "api")
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        assert nav.navigation_state.target_velocity == 0.0
    finally:
        nav._ai_task.cancel()
        await asyncio.gather(nav._ai_task, return_exceptions=True)


async def test_no_autonomous_run_starts_while_a_latch_is_set(monkeypatch):
    tractor, _pca = mowing_tractor()
    nav = new_nav()
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", True)
    assert await nav.start_autonomous_navigation() is False
    assert await nav.start_ai_navigation() is False
    assert await nav.return_home() is False
    with pytest.raises(RuntimeError, match="Cannot start mission"):
        await nav.execute_mission(new_mission([FAR]))
    await tractor.shutdown()


# ------------------------------------------------------------------------ RoboHAT clear (SAF-15)


async def test_a_robohat_clear_that_does_not_get_usb_control_keeps_the_hold(monkeypatch):
    svc, fw, _tractor = robohat_platform(monkeypatch)
    await svc.emergency_stop_detail()
    assert svc.emergency_latched

    async def deaf(line: str) -> bool:  # the firmware never acknowledges the USB handshake
        fw.sent.append(line)
        if line.startswith("pwm,"):
            svc._process_line("[USB] PWM set -> steer=1500 us throttle=1500 us")
        return True

    monkeypatch.setattr(svc, "_send_line", deaf)
    svc._rc_enabled = True  # the firmware fell back to RC mode: only the handshake gets it back
    assert await svc.clear_emergency() is False
    assert svc.emergency_latched is True, "a failed clear must leave the hold set"
    assert svc.status.last_error == "clear_emergency_failed"
    await svc.shutdown()
