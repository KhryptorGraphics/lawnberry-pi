"""A mode field is not a run: nothing acts on a person's own drive and blade for one (NAV-03).

``return_home()`` planned a path and set the mode to ``return_home`` with ``executing``: no loop
drove it, nothing ended it (the stop button only aborts a mission id, the arrival is never
detected), and ``AUTONOMOUS_MODES`` made ``autonomy_active()`` true for it. Round 4 added consumers
that ACT on that answer: the GNSS monitor calls ``hold_for_gnss`` every 0.2 s while the fix is
degraded (the normal case outdoors without RTK), which commanded ``set_levers(0, 0)`` and blade off
over the operator's joystick; the IBT-4 permit read the gate as an autonomous hold and refused the
operator's blade ON; the obstacle handler stood the platform down. The policy the code states
("manual driving included, the blade is the operator's own call") was violated by a state in which
no run exists.

Now ``autonomy_active()`` keys on the loops that exist (a mission loop, a live AI task) and
``return_home()`` sets no autonomous mode. The tractor is the real service over the stallable
PCA9685, the differential platform the real RoboHAT service over a fake serial line, the IBT-4 the
real adapter on a fake ``lgpio``; the GNSS monitor is the real one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from _mission_rig import new_nav
from _nav_rig import OFF, ON, StubGuard, armed_blade, robohat_with_timeline
from _tractor_fakes import mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode, PathStatus, Position
from backend.src.nav.gps_degradation import GPSDegradationConfig, GPSDegradationMonitor
from backend.src.services.navigation_service import (
    AUTONOMOUS_MODES,
    BLADE_PERMIT_NAME,
    NavigationService,
)

HOME = Position(latitude=40.0001, longitude=-83.0001)
PHANTOM_MODES = [
    NavigationMode.AUTO,
    NavigationMode.AI,
    NavigationMode.AI_ASSISTED,
    NavigationMode.RETURN_HOME,
]


class StubGnss:
    """A GNSS monitor whose verdict the test flips (what the permit and the gate read)."""

    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    def assess(self):
        reasons = () if self.healthy else ("accuracy 1.50m exceeds 1m",)
        return SimpleNamespace(healthy=self.healthy, reasons=reasons)


def _home_nav() -> NavigationService:
    nav = new_nav()
    nav.navigation_state.home_position = HOME
    return nav


def _real_gnss_monitor(nav: NavigationService) -> GPSDegradationMonitor:
    """The lifespan's wiring: the real monitor, no fix seen yet (degraded), the nav's handler."""
    cfg = GPSDegradationConfig(
        max_accuracy_m=1.0, max_fix_age_s=2.0, resume_dwell_s=3.0, check_interval_s=0.01
    )
    monitor = GPSDegradationMonitor(
        cfg, autonomy_active=nav.autonomy_active, hold_handlers=(nav.hold_for_gnss,)
    )
    nav.attach_safety(obstacle_guard=StubGuard(), gps_monitor=monitor)
    return monitor


# ------------------------------------------------------------------------- return_home


async def test_return_home_plans_the_path_and_sets_no_autonomous_mode():
    nav = _home_nav()

    assert await nav.return_home() is True

    state = nav.navigation_state
    assert state.navigation_mode not in AUTONOMOUS_MODES, "no loop drives it: no run mode"
    assert nav.autonomy_active() is False
    assert state.path_status == PathStatus.PLANNED, "planned, not 'executing' with nothing driving"
    assert state.current_waypoint_index == 0 and len(state.planned_path) >= 1
    assert (
        state.planned_path[-1].position.latitude,
        state.planned_path[-1].position.longitude,
    ) == (
        HOME.latitude,
        HOME.longitude,
    )


@pytest.mark.parametrize("what", ["no_home", "no_position", "latched", "loop_running"])
async def test_return_home_refuses_and_leaves_the_state_alone(what, monkeypatch):
    nav = _home_nav()
    before = (nav.navigation_state.planned_path, nav.navigation_state.path_status)
    if what == "no_home":
        nav.navigation_state.home_position = None
    elif what == "no_position":
        nav.navigation_state.current_position = None
    elif what == "latched":
        monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", True)
    else:
        nav._mission_loop_active = True  # a mission's loop: its path and index are not ours

    assert await nav.return_home() is False

    assert (nav.navigation_state.planned_path, nav.navigation_state.path_status) == before
    assert nav.navigation_state.navigation_mode not in AUTONOMOUS_MODES


# ---------------------------------------------------- the operator's drive under a degraded fix


async def test_after_return_home_a_degraded_fix_leaves_the_operators_tractor_drive_alone(
    monkeypatch,
):
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    nav = _home_nav()
    monitor = _real_gnss_monitor(nav)
    assert await nav.return_home() is True
    assert (await tractor.engage_blade(True, source="operator"))["status"] == "ok"
    drive = await tractor.set_levers(0.5, 0.5, source="operator", duration_ms=5000)
    assert drive["status"] == "ok"

    await monitor.start()  # no GNSS fix has been seen: the monitor is degraded every cycle
    try:
        await asyncio.sleep(0.3)  # ~30 cycles of the hold handler, had autonomy been "active"
    finally:
        await monitor.stop()

    assert monitor.assess().degraded, "premise: the fix is degraded"
    assert nav._hold_reasons is None, "the hold handler never ran: there is no run to hold"
    assert tractor.state.left_lever == 0.5 and tractor.state.right_lever == 0.5
    assert tractor.state.blade_engaged is True
    await tractor.shutdown()


async def test_after_return_home_a_degraded_fix_does_not_refuse_the_operators_ibt4_blade(
    monkeypatch,
):
    monkeypatch.setenv("SIM_MODE", "0")  # the real lgpio branch, on the fake pads
    blade, pads = armed_blade(monkeypatch, deadman_ms=10000)
    nav = _home_nav()
    gnss = StubGnss()
    nav.attach_safety(obstacle_guard=StubGuard(), gps_monitor=gnss)
    blade.attach_permit(BLADE_PERMIT_NAME, nav.blade_permit)
    assert await blade.initialize() is True
    try:
        assert await nav.return_home() is True
        gnss.healthy = False  # degraded, and nothing moves the machine

        assert nav.blade_permit() is None
        assert await blade.set_active(True) is True, blade.last_refusal
        assert pads.pad == ON
        await asyncio.sleep(0.3)  # the supervisor reads the permit every tick: still on
        assert pads.pad == ON
    finally:
        await blade.park()
        assert pads.pad == OFF


# -------------------------------------------------------------------- the mode field alone


@pytest.mark.parametrize("mode", PHANTOM_MODES)
async def test_a_mode_field_with_no_loop_is_not_autonomy_for_any_consumer(monkeypatch, mode):
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    guard = StubGuard(hold=True)
    nav.attach_safety(obstacle_guard=guard, gps_monitor=StubGnss(healthy=False))
    nav.navigation_state.navigation_mode = mode
    try:
        assert nav.autonomy_active() is False
        assert nav.blade_permit() is None, "a manual blade is the operator's own call"
        await nav.enforce_motion_hold()  # the obstacle monitor's handler, 20 Hz in production
        assert fw.sent == [], "nothing was sent over a person's drive"
    finally:
        await svc.shutdown()


async def test_a_mission_loop_counts_even_when_the_mode_field_moved_off_an_autonomous_mode(
    monkeypatch,
):
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.attach_safety(obstacle_guard=StubGuard(hold=True))
    nav.navigation_state.navigation_mode = NavigationMode.IDLE
    nav._mission_loop_active = True  # execute_mission's loop is running
    try:
        assert nav.autonomy_active() is True
        assert "navigation safety gate holds" in (nav.blade_permit() or "")
        await nav.enforce_motion_hold()
        assert fw.sent == ["blade=off", "pwm,1500,1500"], "the run is stood down, blade first"
    finally:
        nav._mission_loop_active = False
        await svc.shutdown()


async def test_a_live_ai_task_is_autonomy_and_a_finished_one_is_not():
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.IDLE
    nav._ai_task = asyncio.create_task(asyncio.sleep(3600))
    try:
        assert nav.autonomy_active() is True
    finally:
        nav._ai_task.cancel()
        await asyncio.gather(nav._ai_task, return_exceptions=True)
    assert nav._ai_task.done() and nav.autonomy_active() is False
