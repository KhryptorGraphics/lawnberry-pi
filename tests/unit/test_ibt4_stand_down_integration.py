"""Every stand-down reaches the IBT-4 and it is CONFIRMED OFF on the pads (audit SAF-03, BL-01).

Cross-track integration the two review-fix tracks could not test alone: the navigation track
calls ``BladeService.request_blade_off`` on every hold, leg end and abort, and the blade track
supervises the E-stop / tilt / lock-out / park state. Here the REAL ``NavigationService`` and
``BladeService`` run together on the fake ``lgpio`` of the blade track (``fake.pad`` is what a
meter on GPIO24/25 shows), on both platforms:

* differential: the real ``RoboHATService`` over a fake serial (SIM_MODE off), tractor disabled;
* tractor: the real ``TractorControlService`` over a fake PCA9685 and fake ``gpiozero``.

Triggers: an obstacle hold and a GNSS hold (once through the running mission loop, once through
the monitor's hold handler with no loop running), the E-stop (orchestrator and HTTP route), a
tilt latch (tractor), and a watchdog trip. After each, the pads read OFF and the service says
the OFF is confirmed. A hold's release never brings the blade back: the operator must re-send
ON (the deadman), and a latch refuses it until cleared.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from _mission_rig import FAR, driving, new_nav, wire_mission
from _nav_rig import (
    OFF,
    ON,
    StubGuard,
    armed_blade,
    robohat_with_timeline,
    wait_until,
)
from _tractor_fakes import FakeGpiozero, mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode
from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.watchdog import Watchdog
from backend.src.services import emergency_stop_service as ess
from backend.src.services import navigation_service as nav_module
from backend.src.services import robohat_service as robohat_module

PLATFORMS = ["differential", "tractor"]


class StubGnss:
    """A GNSS monitor whose verdict the test flips."""

    def __init__(self) -> None:
        self.healthy = True

    def assess(self):
        reasons = () if self.healthy else ("fix is 30.0s old (limit 2s)",)
        return SimpleNamespace(healthy=self.healthy, reasons=reasons)


class Platform:
    """One platform with its blade, guard, GNSS monitor and a mission loop to hold."""

    def __init__(self, monkeypatch, kind: str) -> None:
        self.monkeypatch = monkeypatch
        self.kind = kind
        self.task: asyncio.Task | None = None
        if kind == "differential":
            self.svc, self.fw, self.tractor, _tl = robohat_with_timeline(monkeypatch)
            self.chip = None
        else:
            monkeypatch.setenv("SIM_MODE", "0")  # the real relay / adapter branches
            FakeGpiozero(monkeypatch)
            self.tractor, self.chip = mowing_tractor()
            self.svc, self.fw = None, None
            monkeypatch.setattr(nav_module, "get_robohat_service", lambda: None)
            monkeypatch.setattr(robohat_module, "get_robohat_service", lambda: None)
        monkeypatch.setattr(
            "backend.src.services.tractor_service.get_tractor_service", lambda: self.tractor
        )
        # A 10 s deadman (the maximum): an OFF inside a test can only come from a stand-down,
        # never from the blade's own ON timing out (that would make every assertion pass).
        self.blade, self.pads = armed_blade(monkeypatch, deadman_ms=10000)
        self.nav = new_nav()
        self.nav.control_period_s = 0.05
        self.guard = StubGuard()
        self.gnss = StubGnss()
        self.nav.attach_safety(obstacle_guard=self.guard, gps_monitor=self.gnss)
        self.tractor.motion_limiter = self.guard

    async def arm_blade(self) -> None:
        """The operator switches the IBT-4 blade on (and it stays on while ON is repeated)."""
        assert await self.blade.initialize() is True
        assert await self.blade.set_active(True) is True
        assert self.pads.pad == ON

    async def start_mission(self) -> None:
        mission, _service = wire_mission(self.monkeypatch, self.tractor, self.nav, [FAR])
        self.task = asyncio.create_task(self.nav.execute_mission(mission))
        await wait_until(self.driving)

    def driving(self) -> bool:
        if self.fw is not None:
            return bool(driving(self.fw.sent))
        return self.tractor.state.moving

    def autonomy_owns_the_machine(self) -> None:
        """No loop: only the hold handlers can act (what the 20 Hz monitors rely on)."""
        self.nav.navigation_state.navigation_mode = NavigationMode.AUTO

    async def confirmed_off(self) -> None:
        await wait_until(lambda: self.pads.pad == OFF)
        status = await self.blade.status()
        assert status["observed"] == "off", status
        assert status["state_confirmed"] is True and status["commanded"] == "off", status

    def clear_every_latch(self) -> None:
        core_globals._safety_state["emergency_stop_active"] = False

    async def operator_clears(self) -> None:
        """What the operator's authenticated clears do, service by service."""
        core_globals._safety_state["emergency_stop_active"] = False
        await self.blade.clear_emergency()
        if self.svc is not None:
            assert await self.svc.clear_emergency() is True
        if self.tractor.enabled:
            result = await self.tractor.clear_emergency()
            assert result["status"] == "ok", result

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.svc is not None:
            await self.svc.shutdown()
        await self.tractor.shutdown()
        await self.blade.park()


@pytest.fixture(params=PLATFORMS)
async def platform(request, monkeypatch):
    p = Platform(monkeypatch, request.param)
    yield p
    await p.close()


# ------------------------------------------------------------------ the holds: loop and monitor


async def test_an_obstacle_hold_in_the_mission_loop_switches_the_ibt4_off_and_it_stays_off(
    platform,
):
    await platform.arm_blade()
    await platform.start_mission()

    platform.guard.hold = True
    await platform.confirmed_off()

    platform.guard.hold = False  # the path clears and the mission resumes by itself ...
    before = platform.tractor.state.moving if platform.fw is None else len(platform.fw.sent)
    await wait_until(
        lambda: (
            platform.tractor.state.moving
            if platform.fw is None
            else len(driving(platform.fw.sent[before:])) >= 1
        )
    )
    await asyncio.sleep(0.5)
    # ... but nothing re-sends the IBT-4 ON: the operator must (deadman)
    assert platform.pads.pad == OFF, "a hold's release brought the blade back"
    assert (await platform.blade.status())["commanded"] == "off"
    assert await platform.blade.set_active(True) is True, "a hold is not a latch"
    assert platform.pads.pad == ON


async def test_a_gnss_hold_in_the_mission_loop_switches_the_ibt4_off_and_it_stays_off(platform):
    await platform.arm_blade()
    await platform.start_mission()

    platform.gnss.healthy = False
    await platform.confirmed_off()

    platform.gnss.healthy = True
    await asyncio.sleep(0.6)
    assert platform.pads.pad == OFF, "a hold's release brought the blade back"
    assert await platform.blade.set_active(True) is True
    assert platform.pads.pad == ON


async def test_the_obstacle_monitors_handler_switches_the_ibt4_off_with_no_loop_running(platform):
    """Autonomy owns the machine but no planner tick runs: the 20 Hz handler alone must do it."""
    await platform.arm_blade()
    platform.autonomy_owns_the_machine()
    platform.guard.hold = True

    await platform.nav.enforce_motion_hold()

    await platform.confirmed_off()


async def test_the_gnss_monitors_handler_switches_the_ibt4_off_with_no_loop_running(platform):
    await platform.arm_blade()
    platform.autonomy_owns_the_machine()
    platform.gnss.healthy = False

    await platform.nav.hold_for_gnss(platform.gnss.assess())

    await platform.confirmed_off()


async def test_a_persons_manual_blade_is_not_dropped_by_the_obstacle_monitor(platform):
    """No autonomous run, no scripted drive: the IBT-4 the operator switched on is left alone."""
    await platform.arm_blade()
    platform.guard.hold = True  # e.g. the bench, with the sensors disconnected

    await platform.nav.enforce_motion_hold()
    await asyncio.sleep(0.2)

    assert platform.pads.pad == ON


# ------------------------------------------------------------------------- the latching stops


async def test_the_emergency_stop_switches_the_ibt4_off_and_only_the_clears_let_it_back_on(
    platform,
):
    await platform.arm_blade()
    await platform.start_mission()

    report = await ess.execute_emergency_stop("api")

    await platform.confirmed_off()
    assert report["emergency_stop_active"] is True
    assert report["subsystems"]["blade_service"]["outputs"]["blade"]["status"] == "stopped"
    assert await platform.blade.set_active(True) is False, "the E-stop latch refuses ON"
    assert platform.pads.pad == OFF

    await platform.operator_clears()
    assert await platform.blade.set_active(True) is True, (
        "after the clears the operator re-sends ON"
    )
    assert platform.pads.pad == ON


async def test_the_emergency_stop_route_switches_the_ibt4_off(platform):
    from backend.src.main import app

    await platform.arm_blade()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        response = await c.post("/api/v2/control/emergency-stop")
    assert response.status_code == 200, response.text
    await platform.confirmed_off()
    assert response.json()["emergency_stop_active"] is True


async def test_a_tilt_latch_switches_the_ibt4_off_and_refuses_it_until_the_operator_clears(
    platform,
):
    if platform.kind != "tractor":
        pytest.skip("the tilt cutoff is the tractor platform's (nothing tilt-monitors the RoboHAT)")
    await platform.arm_blade()

    await platform.tractor.latch_interlock("tilt", source="tilt_monitor")

    await platform.confirmed_off()
    assert await platform.blade.set_active(True) is False
    assert "interlock" in (platform.blade.last_refusal or "")
    cleared = await platform.tractor.clear_emergency()
    assert cleared["status"] == "ok"
    assert await platform.blade.set_active(True) is True and platform.pads.pad == ON


async def test_a_watchdog_trip_switches_the_ibt4_off_on_either_platform(platform):
    """The trip's stop is the all-platform stop: it does not depend on an enabled tractor."""
    import backend.src.main as main

    await platform.arm_blade()
    await platform.start_mission()
    clock = SimpleNamespace(t=1000.0)
    watchdog = Watchdog(
        EstopHandler(platform.tractor),
        timeout_ms=200,
        on_timeout=main._watchdog_stop,
        required_sources=("actuation",),
        retrip_interval_s=5.0,
        clock=lambda: clock.t,
    )
    clock.t += 1.0  # the heartbeat went silent

    assert await watchdog.evaluate() is True and watchdog.tripped

    await platform.confirmed_off()
    assert await platform.blade.set_active(True) is False, "a trip latches: the clears come first"
    await platform.operator_clears()
    assert await platform.blade.set_active(True) is True
