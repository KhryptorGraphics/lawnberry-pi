"""Blade interlock latch after a tilt cutoff.

Navigation re-sends ``blade_engaged=True`` every ~200 ms. Switching the blade
off once on a tilt edge is therefore not a cutoff: the next tick turns it back
on while the machine is still tilted. The latch must hold until an explicit
operator clear, survive the tilt going away, and be re-asserted if a clear
lands while the machine is still tilted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend.src.models import Position
from backend.src.models.mission import MissionWaypoint
from backend.src.models.tractor_control import EngineState, TractorCommand
from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.motor_authorization import MotorAuthorization
from backend.src.safety.safety_triggers import SafetyTriggerManager
from backend.src.safety.tractor_safety_monitor import TractorSafetyMonitor
from backend.src.safety.watchdog import SAFETY_MONITOR_SOURCE, Watchdog
from backend.src.services.navigation_service import NavigationService
from backend.src.services.tractor_service import TractorControlService


def _mowing_tractor() -> TractorControlService:
    svc = TractorControlService(config={"enabled": True})
    svc.authorize()
    svc.state.engine = EngineState.RUNNING  # skip the 800 ms starter pulse
    return svc


def _nav_tick(blade: bool = True) -> TractorCommand:
    return TractorCommand(left_lever=0.3, right_lever=0.3, throttle=0.7, blade_engaged=blade)


class Imu:
    """Mutable IMU reader: flip ``tilted`` mid-test."""

    def __init__(self) -> None:
        self.tilted = False

    async def __call__(self):
        return (45.0, 0.0) if self.tilted else (2.0, 1.0)


def _monitor(tractor: TractorControlService, imu: Imu) -> TractorSafetyMonitor:
    wd = Watchdog(
        EstopHandler(MotorAuthorization()),
        timeout_ms=60_000,
        required_sources=(SAFETY_MONITOR_SOURCE,),
    )
    return TractorSafetyMonitor(
        tractor,
        wd,
        tilt_threshold_degrees=30.0,
        trigger_manager=SafetyTriggerManager(),
        imu_reader=imu,
        poll_interval_s=0.01,
    )


# ------------------------------- the latch itself -------------------------------


async def test_latch_drops_the_blade_and_refuses_every_navigation_reengage():
    svc = _mowing_tractor()
    assert (await svc.apply(_nav_tick()))["results"]["blade"]["status"] == "ok"
    assert svc.state.blade_engaged is True and svc.blade_pto.state is True

    result = await svc.latch_interlock("tilt", source="tilt_monitor")

    assert result["status"] == "latched" and svc.state.blade_engaged is False
    for _ in range(5):  # five navigation ticks, ~1 s of a mission
        res = await svc.apply(_nav_tick())
        blade = res["results"]["blade"]
        assert blade["status"] == "rejected" and "latched" in blade["reason"]
        assert svc.state.blade_engaged is False and svc.blade_pto.state is False
    # The drive itself is not blocked by the *blade* latch (recovery may need to move).
    assert res["results"]["levers"]["status"] == "ok"
    # ...and the latch stays visible to the UI on every accepted command.
    assert "latched (tilt)" in svc.state.interlock_reason
    assert svc.get_state().interlock_latched is True


async def test_latch_holds_even_when_the_blade_was_already_off_at_the_trip():
    """Between mowing legs the blade is off; the next leg's blade_on must still be refused."""
    svc = _mowing_tractor()
    await svc.latch_interlock("tilt")

    assert (await svc.engage_blade(True))["status"] == "rejected"
    assert svc.blade_pto.state is False


async def test_only_an_explicit_clear_releases_the_latch():
    svc = _mowing_tractor()
    await svc.latch_interlock("tilt")

    # Time passing / commands / re-authorizing do not release it.
    svc.authorize()
    await asyncio.sleep(0.05)
    assert (await svc.engage_blade(True))["status"] == "rejected"

    await svc.clear_emergency()  # POST /tractor/clear-emergency (operator-authenticated)
    assert svc.state.interlock_latched is False and svc.state.latch_reason is None
    assert (await svc.engage_blade(True))["status"] == "ok"


async def test_latch_is_idempotent_and_never_raises_when_the_relay_is_dead(monkeypatch):
    svc = _mowing_tractor()
    await svc.engage_blade(True)

    def dead_relay(on: bool) -> None:
        raise OSError("relay GPIO dead")

    monkeypatch.setattr(svc.blade_pto, "set", dead_relay)
    result = await svc.latch_interlock("tilt")  # must not raise into the monitor loop

    assert result["status"] == "fault" and "relay GPIO dead" in result["error"]
    assert svc.state.interlock_latched is True, "the latch is software state and holds regardless"
    assert svc.state.blade_engaged is True, "relay did not switch: do not claim the blade is off"
    assert (await svc.engage_blade(True))["status"] == "rejected"


# ------------------------- tilt monitor -> latch -> navigation -------------------------


async def test_tilt_latches_even_when_the_blade_was_off():
    svc = _mowing_tractor()
    imu = Imu()
    monitor = _monitor(svc, imu)
    await monitor.start()
    imu.tilted = True
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert svc.state.interlock_latched is True and svc.state.latch_reason == "tilt"


async def test_tilt_returning_to_normal_does_not_release_the_latch():
    svc = _mowing_tractor()
    imu = Imu()
    monitor = _monitor(svc, imu)
    await monitor.start()
    imu.tilted = True
    await asyncio.sleep(0.1)
    imu.tilted = False
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert svc.state.interlock_latched is True
    assert (await svc.engage_blade(True))["status"] == "rejected"


async def test_a_clear_while_still_tilted_is_reasserted_by_the_monitor():
    svc = _mowing_tractor()
    imu = Imu()
    imu.tilted = True
    monitor = _monitor(svc, imu)
    await monitor.start()
    await asyncio.sleep(0.1)
    assert svc.state.interlock_latched is True

    await svc.clear_emergency()  # operator clears while the machine is still over the threshold
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert svc.state.interlock_latched is True, "still tilted: the latch must come straight back"
    assert (await svc.engage_blade(True))["status"] == "rejected"


async def test_navigation_loop_does_not_reengage_the_blade_after_a_tilt(monkeypatch):
    """End to end: real mission loop + real monitor + real tractor service (SIM)."""
    svc = _mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: svc)
    monkeypatch.setattr(
        "backend.src.services.mission_service.get_mission_service",
        lambda: SimpleNamespace(missions={}, mission_statuses={}),
    )
    nav = NavigationService()
    nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
    nav.navigation_state.heading = 0.0
    far = MissionWaypoint(lat=40.01, lon=-83.01, blade_on=True)  # never arrives in this test

    imu = Imu()
    monitor = _monitor(svc, imu)
    await monitor.start()
    mission = asyncio.create_task(nav.go_to_waypoint(far))
    try:
        await asyncio.sleep(0.5)
        assert svc.state.blade_engaged is True, "navigation mows with the blade before the tilt"

        imu.tilted = True
        await asyncio.sleep(0.15)  # tilt cutoff window
        assert svc.state.blade_engaged is False

        await asyncio.sleep(1.0)  # five more navigation ticks, each re-sending blade_on=True
        assert svc.state.blade_engaged is False, "navigation re-engaged the blade while tilted"
        assert svc.blade_pto.state is False
        assert svc.state.moving is True, "the mission loop is still commanding the drive"
    finally:
        mission.cancel()
        await asyncio.gather(mission, return_exceptions=True)
        await monitor.stop()
        await svc.shutdown()
