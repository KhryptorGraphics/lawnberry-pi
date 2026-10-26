"""Round-3 gaps of the navigation track, each pinned through the real services.

* M10: the tractor's obstacle-hold handler is wired into the REAL lifespan's monitor;
* SAF-15 route half: ``/control/emergency_clear`` lists a failed RoboHAT clear as ``robohat_hold``;
* SAF-16 RoboHAT-hold restore path: a hold the last process left is a real hold again;
* a watchdog trip on a platform whose tractor is disabled still stops the RoboHAT and is recorded;
* SAF-01 end to end: the AI loop and the Thor follower behave like a mission after an E-stop and
  an authenticated clear: nothing drives until an explicit ``authorize()``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from _mission_rig import FAR, driving, new_nav, wire_mission
from _nav_rig import live_sensors, real_lifespan, robohat_with_timeline, wait_until
from _tractor_fakes import LEFT, NEUTRAL, RIGHT, mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode
from backend.src.models.action_prediction import ActionPrediction
from backend.src.nav.location_features import CoverageMap
from backend.src.safety import latch_store, stop_events
from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.watchdog import Watchdog

NEUTRAL_LINE = "pwm,1500,1500"


# ------------------------------------------------------------------ M10: the real lifespan


async def test_the_real_lifespan_wires_the_tractors_hold_handler_into_the_obstacle_monitor(
    monkeypatch,
):
    """No planner tick runs: only the monitor's handler can stop the command in force."""
    tractor, chip = mowing_tractor()
    sensors = live_sensors(2000.0)  # clear ground
    async with real_lifespan(monkeypatch, tractor, sensors) as app:
        guard = app.state.obstacle_guard
        await wait_until(lambda: not guard.assess().hold, timeout=8.0)  # the clear dwell is over
        assert (await tractor.engage_blade(True, source="navigation"))["status"] == "ok"
        moved = await tractor.set_levers(0.3, 0.3, source="navigation", duration_ms=8000)
        assert moved["status"] == "ok" and tractor.state.left_lever > 0.0

        sensors.tof.mm = 100.0  # an obstacle 10 cm ahead: the monitor's next cycle sees a hold

        await wait_until(
            lambda: tractor.state.left_lever == 0.0 and tractor.state.blade_engaged is False,
            timeout=3.0,
        )
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]


# ----------------------------------------------------------------- SAF-15: the clear route


async def _post_clear() -> httpx.Response:
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.post("/api/v2/control/emergency_clear", json={"confirmation": True})


async def test_a_robohat_clear_that_fails_is_listed_as_robohat_hold_not_cleared(monkeypatch):
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    await svc.emergency_stop_detail()
    assert svc.emergency_latched
    core_globals._safety_state["emergency_stop_active"] = True
    real_clear = svc.clear_emergency

    async def deaf() -> bool:  # no USB control: the service keeps the hold and says False
        return False

    monkeypatch.setattr(svc, "clear_emergency", deaf)
    response = await _post_clear()
    body = response.json()
    assert response.status_code == 200 and body["status"] == "EMERGENCY_PARTIALLY_CLEARED", body
    assert "robohat_hold" in body["still_latched"], body
    assert svc.emergency_latched is True, "a failed clear must leave the hold set"

    async def broken() -> bool:  # the clear raises: the latch that is still set is listed anyway
        raise RuntimeError("serial fell over")

    monkeypatch.setattr(svc, "clear_emergency", broken)
    body = (await _post_clear()).json()
    assert (
        body["status"] == "EMERGENCY_PARTIALLY_CLEARED" and "robohat_hold" in body["still_latched"]
    )

    monkeypatch.setattr(svc, "clear_emergency", real_clear)  # now the firmware answers
    core_globals._safety_state["emergency_stop_active"] = True
    body = (await _post_clear()).json()
    assert body["status"] == "EMERGENCY_CLEARED" and "still_latched" not in body, body
    assert svc.emergency_latched is False
    await svc.shutdown()


# ---------------------------------------------------------------- SAF-16: the restore path


async def test_a_robohat_hold_the_last_process_left_is_a_real_hold_after_the_restart(monkeypatch):
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    await svc.emergency_stop_detail()
    assert (await latch_store.persist(robohat=svc)).ok  # the previous process saved its hold
    svc._estop_hold = False  # ... and died: a fresh process starts with nothing latched
    fw.sent.clear()

    found = latch_store.restore(tractor=None, robohat=svc)

    assert found.has(latch_store.ROBOHAT_HOLD) and svc.emergency_latched is True
    assert await svc.send_motor_command(0.4, 0.4) is False, "a restored hold refuses the drive"
    assert await svc.send_blade_command(True) is False, "and the blade"
    assert driving(fw.sent) == []
    svc._last_pwm_at = svc._last_status_at = 0.0
    svc._last_pwm = (1700, 1700)
    await svc._maintain_usb_control()
    assert fw.sent == ["blade=off", NEUTRAL_LINE], (
        "the hold keeps asserting blade off and NEUTRAL, never a stale command"
    )

    assert await svc.clear_emergency() is True  # only the authenticated clear releases it
    assert (await latch_store.persist(robohat=svc)).ok
    assert not latch_store.latch_path().exists(), "nothing latched: the file is removed"
    await svc.shutdown()


async def test_an_unreadable_latch_file_restores_the_robohat_hold_too(monkeypatch):
    svc, _fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    latch_store.latch_path().write_text("{not json")
    found = latch_store.restore(tractor=None, robohat=svc)
    assert found.unreadable is not None and svc.emergency_latched is True, "fail closed"
    await svc.shutdown()


# ------------------------------------------ a watchdog trip with the tractor disabled


async def test_a_watchdog_trip_on_a_platform_with_no_tractor_stops_the_robohat_and_is_recorded(
    monkeypatch,
):
    import backend.src.main as main

    svc, fw, tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 0.05
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await wait_until(lambda: driving(fw.sent))
        epoch = stop_events.epoch()
        clock = SimpleNamespace(t=1000.0)
        watchdog = Watchdog(
            EstopHandler(tractor),
            timeout_ms=200,
            on_timeout=main._watchdog_stop,
            required_sources=("actuation",),
            retrip_interval_s=5.0,
            clock=lambda: clock.t,
        )
        clock.t += 1.0  # the heartbeat went silent

        assert await watchdog.evaluate() is True

        assert stop_events.epoch() > epoch, "the trip must be recorded as a stop"
        assert service.mission_statuses[mission.id].status == "aborted"
        assert svc.emergency_latched is True
        assert fw.sent[-2:] == ["blade=off", NEUTRAL_LINE], fw.sent[-6:]
        await asyncio.wait_for(task, timeout=3.0)
        report = await main._watchdog_stop()
        assert report["subsystems"]["tractor"]["status"] == "not_enabled"
        assert report["subsystems"]["robohat"]["status"] == "stopped", report
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await svc.shutdown()


# ------------------------------------------------- SAF-01: the AI loop after an E-stop


class _AI:
    async def infer(self, frame, coverage_grid=None, datum=None):
        return ActionPrediction(steering=0.0, throttle=0.5, blade=True, confidence=0.9)

    async def disable(self):
        pass


class _Recorder:
    async def capture_frame(self):
        from backend.src.models.mower_data_frame import GPSData, MowerDataFrame, RTKFixType

        frame = MowerDataFrame()
        frame.gps = GPSData(
            latitude=40.0, longitude=-83.0, rtk_fix_type=RTKFixType.FIXED, num_satellites=20
        )
        return frame


def _start_ai_loop(monkeypatch, tractor):
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    nav._ai_datum, nav._ai_epoch = (40.0, -83.0), stop_events.epoch()
    nav._coverage = CoverageMap(nav._ai_datum)
    nav._ai_loop_hz = 50.0
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(
        "backend.src.services.ai_inference_service.get_ai_inference_service", lambda: _AI()
    )
    monkeypatch.setattr(
        "backend.src.services.perimeter_recorder.get_perimeter_recorder", lambda: _Recorder()
    )
    nav._ai_task = asyncio.create_task(nav._ai_inference_loop())
    return nav


async def test_an_estop_ends_the_ai_loop_and_a_clear_plus_authorize_does_not_restart_it(
    monkeypatch,
):
    tractor, chip = mowing_tractor()
    nav = _start_ai_loop(monkeypatch, tractor)
    try:
        await wait_until(lambda: tractor.state.moving)
        await tractor.emergency_stop(source="api")
        await asyncio.wait_for(nav._ai_task, timeout=2.0)  # the loop ended with the stop
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE

        await tractor.clear_emergency()
        assert tractor.state.authorized is False, "a clear never authorizes"
        landed = len(chip.log)
        await asyncio.sleep(0.3)
        assert [1 for _t, ch, us in chip.log[landed:] if us != NEUTRAL[ch]] == []

        tractor.authorize()  # the explicit act
        await asyncio.sleep(0.4)
        assert [1 for _t, ch, us in chip.log[landed:] if us != NEUTRAL[ch]] == []
        assert nav._ai_task.done() and tractor.state.moving is False
    finally:
        nav._ai_task.cancel()
        await asyncio.gather(nav._ai_task, return_exceptions=True)
        await tractor.shutdown()


async def test_the_ai_loops_own_epoch_check_ends_a_run_the_listener_never_told(monkeypatch):
    """The listener is the fast path; the loop's epoch check is the one that cannot be missed."""
    tractor, chip = mowing_tractor()
    nav = _start_ai_loop(monkeypatch, tractor)
    try:
        await wait_until(lambda: tractor.state.moving)
        stop_events.remove_listener(nav._on_stop_event)  # a stop nobody told this service about
        stop_events.record_stop("emergency_stop", "api")
        assert nav.navigation_state.navigation_mode == NavigationMode.AI, "premise: not told"

        await asyncio.wait_for(nav._ai_task, timeout=2.0)

        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        assert tractor.state.moving is False and tractor.state.blade_engaged is False
    finally:
        nav._ai_task.cancel()
        await asyncio.gather(nav._ai_task, return_exceptions=True)
        await tractor.shutdown()


# ---------------------------------------------- SAF-01: the Thor follower after an E-stop


async def test_the_thor_follower_drives_nothing_after_an_estop_until_an_explicit_authorize():
    from test_thor_plan_follower import Rig

    r = Rig()
    await r.step()
    assert len(r.nav.calls) == 1, "premise: an authorized tractor follows the plan"

    await r.tractor.emergency_stop(source="api")
    await r.step()
    assert len(r.nav.calls) == 1 and r.follower.reason == "estop"

    await r.tractor.clear_emergency()  # the authenticated clear releases the latch, nothing else
    await r.step()
    assert len(r.nav.calls) == 1 and r.follower.reason == "not_authorized"

    r.tractor.authorize()  # a person's explicit go-ahead is what lets the plan resume
    await r.step()
    assert len(r.nav.calls) == 2 and r.follower.reason is None
    await r.tractor.shutdown()
