"""Survivors of the third mutation run on ``navigation_service`` (review 3: NAV-09, NAV-10).

The reviewer mutated the safety code of this round and listed what no test noticed. Every test here
pins one of those behaviours through the real services, with only the hardware faked: the tractor
is the real service over the stallable PCA9685, the differential platform the real RoboHAT service
over a fake serial line (what the firmware would have received is what is asserted), the IBT-4 the
real adapter on a fake ``lgpio``. The one exception is on purpose: to isolate a NAV-10 layer, the
consumer right next to it (the leg under the mission loop, the prediction consumer under the AI
loop) is a stand-in that reports a chosen verdict and commands nothing.

NAV-09 (a behaviour nothing pinned)
    N01 a mission loop that really runs counts as autonomy whatever the mode field says;
    N08/N34/N35 no run starts under any latch (the tilt latch was the one nobody listed);
    N12 a blade command the controller refused is retried after a window, not on every tick;
    N20/N21 a leg holds, blade first, on a heading or a position it cannot steer by;
    N36 the navigation-level E-stop ends the missions; N41 a second mission after a stop sends its
    blade-on again; N43/N44 a mission that loses its status or raises stands the platform down.

NAV-10 (a protective layer that every other layer hides)
    After a stop, the E-stop listener has already ended the run, the loops check the stop epoch,
    and the tractor itself refuses commands, so removing any ONE of those left every test green.
    Each layer below has a test in which the OTHERS cannot fire: the listener is removed
    (``stop_events.remove_listener``), the stop is an epoch bump nobody was told about, the
    refusal is one that records no stop event (a revoke, a restored latch), the leg or the
    prediction consumer is replaced by a stand-in that cannot end the run by itself. What is
    asserted is what a person would see: the run task ended, the mission is ``aborted`` (never
    ``completed``), the mode is ``idle``, and nothing was commanded after the stop.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from types import SimpleNamespace

import pytest
from _mission_rig import FAR, FAR_POSITION, new_nav, robohat_platform, wire_mission
from _nav_rig import OFF, ON, StubGuard, armed_blade, wait_until
from _tractor_fakes import LEFT, NEUTRAL, RIGHT, idle_tractor, mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode, PathStatus, Position, Waypoint
from backend.src.models.action_prediction import ActionPrediction
from backend.src.models.mission import MissionWaypoint
from backend.src.safety import stop_events
from backend.src.services import emergency_stop_service as ess
from backend.src.services import navigation_service as nav_module
from backend.src.services.navigation_service import BLADE_PERMIT_NAME, BLADE_RETRY_S, LegResult

NEUTRAL_LINE = "pwm,1500,1500"
HOME = Position(latitude=40.0001, longitude=-83.0001)
#: About 22 m north of ``HERE``: far enough to drive toward, near enough to "arrive" by a moved fix.
NEAR = MissionWaypoint(lat=40.0002, lon=-83.0, blade_on=True)
NEAR_POSITION = Position(latitude=NEAR.lat, longitude=NEAR.lon)


def _moves(lines: list[str]) -> list[str]:
    """The drive commands on the wire that move the machine: any pwm line but neutral."""
    return [line for line in lines if line.startswith("pwm,") and line != NEUTRAL_LINE]


async def _end(*tasks: asyncio.Future | None) -> None:
    """Cancel ``tasks`` and wait for them, for as long as it takes and never for ever.

    One ``cancel()`` is not enough: CPython 3.11's ``wait_for`` (the tractor's servo write is one)
    returns the finished write's result instead of raising when the cancel lands in the loop
    turn the write completed in, and a mission loop that swallowed its cancel runs on. So the
    cancel is repeated until the task is gone.
    """
    live = [task for task in tasks if task is not None]
    for _ in range(100):
        pending = [task for task in live if not task.done()]
        if not pending:
            break
        for task in pending:
            task.cancel()
        await asyncio.wait(pending, timeout=0.1)
    for task in live:
        if task.done() and not task.cancelled():
            task.exception()  # retrieved: nothing is left to warn about at loop close


async def _result(task: asyncio.Future, timeout: float = 8.0):
    """What ``task`` returned (or raised) once it has ended BY ITSELF.

    The task is never cancelled here (a cancel that is swallowed would make the wait unbounded):
    a task that has not ended after ``timeout`` is a failed test, and ``_end`` clears it away.
    """
    await asyncio.wait({task}, timeout=timeout)
    assert task.done(), "the run did not end by itself"
    return task.result()


def _steady_tractor():
    """The tractor platform whose lever commands stay in force for 10 s: only a stand-down returns
    them to neutral inside a test, never the command deadman."""
    return mowing_tractor(
        config={"command_deadman": {"default_hold_ms": 10_000, "max_hold_ms": 10_000}}
    )


def _good(blade: bool = True) -> ActionPrediction:
    return ActionPrediction(steering=0.0, throttle=0.8, blade=blade, confidence=0.9)


class _FakeAI:
    """The AI inference service as ``start_ai_navigation`` and the inference loop use it."""

    initialized = True
    _model_loaded = True

    def __init__(self) -> None:
        self.enables = 0

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def enable(self) -> bool:
        self.enables += 1
        return True

    async def disable(self) -> bool:
        return True

    async def infer(self, frame, coverage_grid=None, datum=None) -> ActionPrediction:
        return _good(blade=False)


class _Frames:
    """A perimeter recorder that hands the loop a frame every tick (or none at all)."""

    def __init__(self, *, live: bool) -> None:
        self.live = live

    async def capture_frame(self):
        if not self.live:
            return None
        return SimpleNamespace(gps=SimpleNamespace(latitude=40.0, longitude=-83.0))


def _fake_ai(monkeypatch, *, frames: bool = False) -> _FakeAI:
    ai = _FakeAI()
    monkeypatch.setattr(
        "backend.src.services.ai_inference_service.get_ai_inference_service", lambda: ai
    )
    monkeypatch.setattr(
        "backend.src.services.perimeter_recorder.get_perimeter_recorder",
        lambda: _Frames(live=frames),
    )
    return ai


async def _ai_run(monkeypatch, tractor, *, hz: float = 50.0):
    """A real AI run on ``tractor``: ``start_ai_navigation`` and its inference loop (a fake model
    that predicts "straight ahead, blade off" every tick). Returns ``(nav, ai)``."""
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    ai = _fake_ai(monkeypatch, frames=True)
    nav = new_nav()
    nav._ai_loop_hz = hz
    assert await nav.start_ai_navigation() is True
    return nav, ai


# =====================================================================================
# N01: a mission loop that is really running is autonomy, whatever the mode field says
# =====================================================================================


@pytest.mark.parametrize("mode", [NavigationMode.IDLE, NavigationMode.MANUAL])
async def test_a_running_mission_loop_keeps_the_operators_ibt4_blade_off_whatever_the_mode_says(
    monkeypatch, mode
):
    """N01. The permit keys on the loop (NAV-03), and the loop is the REAL ``execute_mission``.

    The existing tests raise ``_mission_loop_active`` by hand: they cannot tell whether the loop
    raises it when it starts and drops it when it ends. Here a stop button, ``stop_ai_navigation``
    or the UI moves the mode field off ``auto`` while the mission goes on holding for an
    obstacle: the blade must stay refused for as long as the loop runs and be the operator's own
    call again the moment it has ended.
    """
    svc, fw, tractor = robohat_platform(monkeypatch)
    blade, pads = armed_blade(monkeypatch, deadman_ms=10000)
    nav = new_nav()
    nav.control_period_s = 0.02
    nav.attach_safety(obstacle_guard=StubGuard(hold=True))
    blade.attach_permit(BLADE_PERMIT_NAME, nav.blade_permit)
    assert await blade.initialize() is True
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    service.mission_tasks[mission.id] = task
    try:
        await wait_until(lambda: "blade=off" in fw.sent, timeout=8.0)  # the loop runs, holding
        nav.navigation_state.navigation_mode = mode  # the field moved; the loop did not

        assert await blade.set_active(True) is False, "the loop is running and the gate holds"
        assert pads.pad == OFF
        assert "navigation safety gate holds" in (blade.last_refusal or "")

        await service.abort_mission(mission.id)  # the run ends ...
        await wait_until(task.done, timeout=8.0)
        assert await blade.set_active(True) is True, blade.last_refusal  # ... the blade is free
        assert pads.pad == ON
    finally:
        await _end(task)
        await blade.park()
        await svc.shutdown()
    assert pads.pad == OFF


async def test_the_obstacle_monitors_handler_acts_on_a_mission_that_really_runs(monkeypatch):
    """N01, second consumer of ``autonomy_active()``: the monitor's hold handler stands the
    platform down only while a run owns the machine. The loop here ticks once and then sleeps
    for a minute, so the handler is on its own, and the mode field has moved off ``auto``."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 60.0
    nav.attach_safety(obstacle_guard=StubGuard(hold=True))
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await wait_until(lambda: NEUTRAL_LINE in fw.sent, timeout=8.0)  # the loop's one hold tick
        nav.navigation_state.navigation_mode = NavigationMode.MANUAL
        sent = len(fw.sent)

        await nav.enforce_motion_hold()  # what the 20 Hz obstacle monitor does on a hold

        assert fw.sent[sent:] == ["blade=off", NEUTRAL_LINE], "the run was not stood down"
    finally:
        await _end(task)
        await svc.shutdown()


# =====================================================================================
# N08 / N34 / N35: no run starts under any latch
# =====================================================================================

#: latch kind -> (platform it lives on, what the refusal tells the operator)
LATCHES = {
    "process_estop": ("tractor", "the emergency stop is latched"),
    "tractor_estop": ("tractor", "the tractor emergency stop is latched"),
    "tilt_latch": ("tractor", "the tractor blade interlock is latched (tilt)"),
    "lock_out": ("tractor", "the tractor is locked out (the tilt monitor did not start)"),
    "robohat_hold": ("differential", "the RoboHAT stop hold is latched"),
    "scripted_drive": ("differential", "a scripted drive (the IMU calibration) is running"),
}
RUNS = ["mission", "ai", "return_home", "autonomous_navigation"]


class Yard:
    """A mower with a mission, a planned path and a home, on either platform.

    ``traffic()`` is the actuator-side record (the servo pulses, or the lines the firmware got):
    comparing it before and after a refused start proves that nothing was driven.
    """

    def __init__(self, monkeypatch, platform: str) -> None:
        self.monkeypatch = monkeypatch
        self.platform = platform
        if platform == "tractor":
            self.tractor, self.chip = mowing_tractor()
            self.svc = self.fw = None
            monkeypatch.setattr(nav_module, "get_robohat_service", lambda: None)
        else:
            self.svc, self.fw, self.tractor = robohat_platform(monkeypatch)
        self.nav = new_nav()
        self.nav.control_period_s = 0.02
        self.nav.navigation_state.home_position = HOME
        self.nav.navigation_state.planned_path = [Waypoint(position=FAR_POSITION)]
        self.nav.navigation_state.path_status = PathStatus.PLANNED
        self.mission, self.service = wire_mission(monkeypatch, self.tractor, self.nav, [FAR])
        self.ai = _fake_ai(monkeypatch)
        self.tasks: list[asyncio.Future] = []

    def traffic(self) -> list:
        return list(self.chip.log if self.fw is None else self.fw.sent)

    async def latch(self, kind: str) -> None:
        if kind == "process_estop":
            self.monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", True)
        elif kind == "tractor_estop":
            await self.tractor.emergency_stop(source="api")
        elif kind == "tilt_latch":
            await self.tractor.latch_interlock("tilt", source="safety")
        elif kind == "lock_out":
            self.tractor.lock_out("the tilt monitor did not start")
        elif kind == "robohat_hold":
            await self.svc.emergency_stop_detail()
        elif kind == "scripted_drive":
            # The IMU calibration manoeuvre, really driving: left and right at 0.3 for a minute.
            self.tasks.append(asyncio.create_task(self.nav.drive_scripted(0.3, 0.3, 60.0)))
            await wait_until(lambda: _moves(self.fw.sent), timeout=8.0)
        else:
            raise AssertionError(kind)

    async def close(self) -> None:
        await _end(*self.tasks, self.nav._ai_task)  # a run a refusal should have prevented
        if self.svc is not None:
            await self.svc.shutdown()
        await self.tractor.shutdown()


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("latch", list(LATCHES))
async def test_no_run_starts_under_a_latch(monkeypatch, latch, run):
    """N08 (the tilt latch), N34 (``return_home``), N35 (``start_autonomous_navigation``): every
    kind of run is refused under every kind of latch, says why (a mission raises it), and drives
    nothing. ``start_autonomous_navigation`` also refuses for other reasons (no path, no fix);
    ``test_every_run_starts_when_nothing_is_latched`` is what proves none of them is in play."""
    platform, phrase = LATCHES[latch]
    yard = Yard(monkeypatch, platform)
    nav, state = yard.nav, yard.nav.navigation_state
    try:
        await yard.latch(latch)
        before = (yard.traffic(), list(state.planned_path), state.path_status)

        if run == "mission":
            run_task = asyncio.create_task(nav.execute_mission(yard.mission))
            yard.tasks.append(run_task)
            with pytest.raises(RuntimeError, match=re.escape(phrase)):
                await _result(run_task)
        elif run == "ai":
            assert await nav.start_ai_navigation() is False
            assert yard.ai.enables == 0, "the AI service was switched on under a latch"
            assert nav._ai_task is None
        elif run == "return_home":
            assert await nav.return_home() is False
        else:
            assert await nav.start_autonomous_navigation() is False

        assert state.navigation_mode == NavigationMode.IDLE, "no autonomous mode was entered"
        assert (yard.traffic(), list(state.planned_path), state.path_status) == before
        assert nav.autonomy_active() is False
    finally:
        await yard.close()


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("platform", ["tractor", "differential"])
async def test_every_run_starts_when_nothing_is_latched(monkeypatch, platform, run):
    """The control for the matrix above: the same set-up, no latch, and the run starts."""
    yard = Yard(monkeypatch, platform)
    nav, state = yard.nav, yard.nav.navigation_state
    try:
        if run == "mission":
            task = asyncio.create_task(nav.execute_mission(yard.mission))
            yard.tasks.append(task)
            await wait_until(lambda: state.navigation_mode == NavigationMode.AUTO, timeout=8.0)
            assert not task.done()
        elif run == "ai":
            assert await nav.start_ai_navigation() is True
            assert yard.ai.enables == 1 and state.navigation_mode == NavigationMode.AI
            await nav.stop_ai_navigation()
        elif run == "return_home":
            assert await nav.return_home() is True
            assert state.planned_path[-1].position.latitude == HOME.latitude
        else:
            assert await nav.start_autonomous_navigation() is True
            assert state.navigation_mode == NavigationMode.AUTO
    finally:
        await yard.close()


# =====================================================================================
# N12: a blade command the controller refused is retried after a window, not on every tick
# =====================================================================================


class _Clock:
    """The navigation module's view of ``time``: ``monotonic`` is the test's, the rest is real.

    No wall-clock threshold anywhere: the retry window is crossed by moving this clock, and the
    control loop (which really runs) is only ever waited on by what it puts on the wire.
    """

    def __init__(self) -> None:
        self.now = 5_000.0

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str):
        return getattr(time, name)


async def test_a_blade_on_the_controller_refused_is_retried_after_the_window_not_every_tick(
    monkeypatch,
):
    """N12. The controller refuses ``blade=on``: the leg drives on blade-less and the command is
    retried once per window, not on every control tick (the window is crossed by moving a test
    clock, so no wall-clock threshold decides anything)."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    firmware_line = svc._send_line

    async def deaf_to_blade_on(line: str) -> bool:
        """The controller takes every line but ``blade=on``, which it refuses (RC handshake)."""
        if line == "blade=on":
            fw.sent.append(line)
            return False
        return await firmware_line(line)

    monkeypatch.setattr(svc, "_send_line", deaf_to_blade_on)
    clock = _Clock()
    monkeypatch.setattr(nav_module, "time", clock)
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))

    async def ticks(n: int) -> None:
        seen = len(_moves(fw.sent))
        await wait_until(lambda: len(_moves(fw.sent)) >= seen + n, timeout=8.0)

    try:
        await ticks(6)  # six control ticks: the leg drives on with the blade off
        assert fw.sent.count("blade=on") == 1, "the refused blade-on was repeated on every tick"

        clock.now += BLADE_RETRY_S - 0.5  # still inside the window
        await ticks(6)
        assert fw.sent.count("blade=on") == 1, "retried inside its window"

        clock.now += 0.6  # the window is over: ONE more try
        await wait_until(lambda: fw.sent.count("blade=on") == 2, timeout=8.0)
        await ticks(6)
        assert fw.sent.count("blade=on") == 2, "and then a new window of its own"
    finally:
        await _end(task)
        await svc.shutdown()


# =====================================================================================
# N20 / N21: a leg holds, blade first, on a heading or a position it cannot steer by
# =====================================================================================


@pytest.mark.parametrize(
    "heading", [math.nan, math.inf, -math.inf, None], ids=["nan", "inf", "-inf", "none"]
)
async def test_a_leg_holds_on_a_heading_that_is_not_a_number(monkeypatch, heading):
    """N20. ``min(1.0, nan)`` is ``1.0``: a NaN heading error was a full-lever pivot, blade on."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 0.02
    nav.navigation_state.heading = heading
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        # six hold ticks (each one puts blade=off on the wire), or the pivot the hold prevents
        await wait_until(lambda: fw.sent.count("blade=off") >= 6 or _moves(fw.sent), timeout=8.0)
        assert _moves(fw.sent) == [], "the leg steered by a heading that is not a number"
        assert "blade=on" not in fw.sent
        assert (await nav.get_navigation_status())["motion_hold"] is True
        assert not task.done(), "a hold waits for a heading, it does not end the mission"

        nav.navigation_state.heading = 0.0  # a measured heading: steered by again
        await wait_until(lambda: _moves(fw.sent) and "blade=on" in fw.sent, timeout=8.0)
        assert (await nav.get_navigation_status())["motion_hold"] is False
    finally:
        await _end(task)
        await svc.shutdown()


async def test_a_leg_that_loses_its_position_stands_the_platform_down_while_it_waits(monkeypatch):
    """N21. Without a fix the last command (and the blade) must not stay in force: blade first."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await wait_until(lambda: "blade=on" in fw.sent and _moves(fw.sent), timeout=8.0)
        lost_at = len(fw.sent)
        nav.navigation_state.current_position = None  # the receiver lost its fix

        await wait_until(lambda: "blade=off" in fw.sent[lost_at:] or task.done(), timeout=8.0)
        assert not task.done(), "the leg waits for a fix, it does not give up"
        off = fw.sent.index("blade=off", lost_at)
        await wait_until(lambda: NEUTRAL_LINE in fw.sent[off:], timeout=8.0)
        assert fw.sent[off : off + 2] == ["blade=off", NEUTRAL_LINE], "blade first, then neutral"
        # level-triggered: the hold is repeated every tick while the fix is missing
        await wait_until(lambda: fw.sent[off:].count("blade=off") >= 6, timeout=8.0)
        assert _moves(fw.sent[off:]) == [] and "blade=on" not in fw.sent[off:]
        assert (await nav.get_navigation_status())["motion_hold"] is True

        nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
        resumed = len(fw.sent)
        await wait_until(
            lambda: _moves(fw.sent[resumed:]) and "blade=on" in fw.sent[resumed:], timeout=8.0
        )
    finally:
        await _end(task)
        await svc.shutdown()


def _non_neutral(chip) -> list[tuple[int, int]]:
    """The servo pulses that moved something: any write that is not the channel's neutral."""
    return [(ch, us) for _t, ch, us in chip.log if us != NEUTRAL[ch]]


@pytest.mark.parametrize("heading", [math.nan, math.inf, None], ids=["nan", "inf", "none"])
async def test_a_tractor_leg_holds_on_a_heading_that_is_not_a_number(monkeypatch, heading):
    """N20 on the tractor platform, where it is a lever command that must not be sent."""
    tractor, chip = mowing_tractor()
    nav = new_nav()
    nav.control_period_s = 0.02
    nav.navigation_state.heading = heading
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        # six hold ticks (each one writes both levers' neutral), or the pivot the hold prevents
        await wait_until(lambda: len(chip.writes(LEFT)) >= 6 or _non_neutral(chip), timeout=8.0)
        assert _non_neutral(chip) == [], "a lever was commanded from a heading that is no number"
        assert tractor.state.blade_engaged is False and tractor.state.moving is False
        assert not task.done()

        nav.navigation_state.heading = 0.0  # a measured heading: steered by again
        await wait_until(lambda: tractor.state.moving and tractor.state.blade_engaged, timeout=8.0)
    finally:
        await _end(task)
        await tractor.shutdown()


async def test_a_tractor_leg_that_loses_its_position_drops_the_pto_and_the_levers(monkeypatch):
    """N21 on the tractor platform: without a fix the PTO relay and the levers must not stay in
    force (a lever command is held for 10 s here: only the hold puts them back to neutral)."""
    tractor, chip = _steady_tractor()
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await wait_until(lambda: tractor.state.moving and tractor.state.blade_engaged, timeout=8.0)
        nav.navigation_state.current_position = None  # the receiver lost its fix

        await wait_until(
            lambda: (
                (tractor.blade_pto.state is False and tractor.state.moving is False) or task.done()
            ),
            timeout=8.0,
        )
        assert not task.done(), "the leg waits for a fix, it does not give up"
        assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]

        nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
        await wait_until(lambda: tractor.state.moving and tractor.state.blade_engaged, timeout=8.0)
    finally:
        await _end(task)
        await tractor.shutdown()


# =====================================================================================
# N36 / N41: the navigation-level E-stop ends the missions; a second mission sends its blade-on
# =====================================================================================


async def test_the_navigation_level_estop_ends_the_running_mission(monkeypatch):
    """N36. On the differential platform nothing else records a stop (the RoboHAT's own hold
    records none), so only ``NavigationService.emergency_stop`` can mark the mission aborted."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    await service.start_mission(mission.id)
    task = service.mission_tasks[mission.id]
    try:
        await wait_until(lambda: _moves(fw.sent), timeout=8.0)
        epoch = stop_events.epoch()

        assert await nav.emergency_stop() is True

        assert stop_events.epoch() == epoch, "premise: no stop event was recorded by this stop"
        assert svc.emergency_latched is True, "premise: the RoboHAT is held"
        assert service.mission_statuses[mission.id].status == "aborted"
        assert nav.navigation_state.navigation_mode == NavigationMode.EMERGENCY_STOP
        await wait_until(task.done, timeout=8.0)
        assert task.exception() is None, "the mission ended with the stop; it did not fail"
        assert service.mission_statuses[mission.id].status == "aborted", "never 'completed'"
    finally:
        await _end(task)
        await svc.shutdown()


async def test_the_navigation_level_estop_ends_an_ai_run_too(monkeypatch):
    """The outcome, not a layer: the mode the stop sets (``emergency_stop``) ends the loop by
    itself as well, so this cannot tell ``abort_autonomy`` from it (a mission can: see above)."""
    svc, fw, _tractor = robohat_platform(monkeypatch)
    _fake_ai(monkeypatch, frames=True)
    nav = new_nav()
    nav._ai_loop_hz = 50.0
    assert await nav.start_ai_navigation() is True
    task = nav._ai_task
    try:
        await wait_until(lambda: _moves(fw.sent), timeout=8.0)
        assert nav.autonomy_active() is True

        assert await nav.emergency_stop() is True

        await wait_until(task.done, timeout=8.0)
        assert nav.navigation_state.navigation_mode == NavigationMode.EMERGENCY_STOP
        assert nav.navigation_state.target_velocity == 0.0
        assert nav.autonomy_active() is False
    finally:
        await _end(task)
        await svc.shutdown()


async def test_a_second_mission_after_a_stop_sends_its_blade_on_again(monkeypatch):
    """N41. The E-stop cuts the RoboHAT blade itself and the mission loop ends without a
    stand-down of its own (the stop owns the outputs): the service still believes it commanded
    the blade ON, so only the reset at the start of a run makes the next mission switch it on."""
    monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", False)
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    await service.start_mission(mission.id)
    first = service.mission_tasks[mission.id]
    second = None
    try:
        await wait_until(lambda: "blade=on" in fw.sent and _moves(fw.sent), timeout=8.0)

        await ess.execute_emergency_stop("api")  # the E-stop route: RoboHAT blade=off, neutral
        # the first run is over, and the mission service has run its done-callback: a callback that
        # ran later would judge the SECOND run's fresh status ("running") and mark it completed
        await wait_until(lambda: mission.id not in service.mission_tasks, timeout=8.0)
        assert first.done()
        assert service.mission_statuses[mission.id].status == "aborted"
        assert await svc.clear_emergency() is True  # the authenticated clears ...
        monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", False)

        before = len(fw.sent)
        await service.start_mission(mission.id)  # ... and a person starts the mission again
        second = service.mission_tasks[mission.id]
        await wait_until(lambda: _moves(fw.sent[before:]) or second.done(), timeout=8.0)
        assert not second.done(), "the second mission is running"
        assert "blade=on" in fw.sent[before:], "the second mission drove with the blade off"
    finally:
        await _end(first, second)
        await svc.shutdown()


# =====================================================================================
# N43 / N44: a mission that loses its status or raises stands the platform down
# =====================================================================================


@pytest.mark.parametrize("how", ["paused", "status_lost"])
async def test_a_run_that_loses_its_status_as_a_leg_arrives_still_stops_the_blade(monkeypatch, how):
    """N43. At an intermediate waypoint the drive stops and the blade is left on for the next leg.
    A pause (``pause_mission`` itself stops nothing) or a status that vanished, landing right
    there, ends the run at the top of the mission loop, which is then the only thing left to
    switch the blade off."""
    tractor, chip = mowing_tractor()
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, service = wire_mission(monkeypatch, tractor, nav, [NEAR, FAR])
    armed = {"hook": False, "fired": False}
    real_set_levers = tractor.set_levers

    async def set_levers(left, right, **kw):
        result = await real_set_levers(left, right, **kw)
        if armed["hook"] and (left, right) == (0.0, 0.0):  # the arrival's drive-only stop
            armed["hook"], armed["fired"] = False, True
            if how == "paused":
                await service.pause_mission(mission.id)
            else:
                del service.mission_statuses[mission.id]
        return result

    monkeypatch.setattr(tractor, "set_levers", set_levers)
    service.mission_tasks[mission.id] = task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await wait_until(lambda: tractor.state.moving and tractor.state.blade_engaged, timeout=8.0)
        armed["hook"] = True
        nav.navigation_state.current_position = NEAR_POSITION  # the leg arrives at NEAR

        await _result(task)

        assert armed["fired"], "premise: the status was lost as the leg arrived"
        if how == "paused":
            assert service.mission_statuses[mission.id].status == "paused"
        else:
            assert mission.id not in service.mission_statuses
        assert nav.navigation_state.current_waypoint_index == 1, "the leg did arrive at NEAR"
        assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False, (
            "the PTO was left on under a mission that is no longer running"
        )
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
    finally:
        await _end(task)
        await tractor.shutdown()


async def test_a_mission_that_raises_ends_with_the_blade_off_and_the_drive_neutral(monkeypatch):
    """N44. The drive controller fails every leg in a row (the blade command still goes through)."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    firmware_line = svc._send_line

    async def refuses_to_move(line: str) -> bool:
        if line.startswith("pwm,") and line != NEUTRAL_LINE:
            fw.sent.append(line)
            return False
        return await firmware_line(line)

    monkeypatch.setattr(svc, "_send_line", refuses_to_move)
    monkeypatch.setattr(nav_module, "MAX_CONSECUTIVE_FAILED_LEGS", 3)
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    await service.start_mission(mission.id)
    task = service.mission_tasks[mission.id]
    try:
        with pytest.raises(RuntimeError, match="failed 3 legs in a row"):
            await _result(task, timeout=10.0)

        assert nav.navigation_state.path_status == PathStatus.FAILED
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        await wait_until(lambda: service.mission_statuses[mission.id].status == "failed")
        assert "blade=on" in fw.sent, "premise: the blade was on while the legs failed"
        moved = [i for i, line in enumerate(fw.sent) if line in _moves(fw.sent)]
        assert moved, "premise: the legs tried to drive"
        after = fw.sent[moved[-1] :]
        assert "blade=off" in after, "the blade was left on under a mission that failed"
        assert NEUTRAL_LINE in after[after.index("blade=off") :], "blade first, then neutral"
    finally:
        await _end(task)
        await svc.shutdown()


# =====================================================================================
# NAV-10: each layer that ends a run after a stop, with the other layers unable to fire
# =====================================================================================


def _stub_leg(monkeypatch, nav, result: LegResult) -> list[str]:
    """Replace ``go_to_waypoint`` by a leg that reports ``result`` at once and commands nothing:
    the mission loop is then the only thing that can end the run, and a leg it is sent on after
    the stop is a command sent after the stop. Returns the ids of the legs it was sent on."""
    legs: list[str] = []

    async def leg(waypoint, *, should_continue=None, run_epoch=None) -> LegResult:
        legs.append(waypoint.id)
        return result

    monkeypatch.setattr(nav, "go_to_waypoint", leg)
    return legs


async def test_the_mission_loops_own_epoch_check_ends_a_run_nothing_else_told(monkeypatch):
    """N17. The listener is removed (nobody marks the mission aborted) and the leg never reports
    a stop (so the loop's STOPPED branch cannot end the run either): after a stop that moved the
    epoch, only the check at the top of the mission loop keeps it from sending the next leg."""
    nav = new_nav()
    mission, service = wire_mission(monkeypatch, idle_tractor(), nav, [FAR])
    legs = _stub_leg(monkeypatch, nav, LegResult.ARRIVED)
    stop_events.remove_listener(nav._on_stop_event)
    await service.start_mission(mission.id)
    task = service.mission_tasks[mission.id]
    try:
        await wait_until(lambda: len(legs) >= 3, timeout=8.0)
        stop_events.record_stop("emergency_stop", "api")  # a stop nobody told this service about
        at_stop = len(legs)
        assert service.mission_statuses[mission.id].status == "running", "premise: not told"

        await wait_until(task.done, timeout=8.0)

        assert len(legs) == at_stop, "the mission sent the machine on another leg after the stop"
        assert service.mission_statuses[mission.id].status == "aborted"
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
    finally:
        await _end(task)


@pytest.mark.parametrize("result", [LegResult.STOPPED, LegResult.REFUSED])
async def test_a_leg_that_reports_stopped_or_refused_ends_the_mission_aborted_not_completed(
    monkeypatch, result
):
    """N37. No stop event, no epoch move, no leg-level abort (the leg is a stand-in): the loop's
    own reaction to the leg's verdict is all that is left. Without its ``abort_autonomy`` the task
    returns normally with the mission still ``running``, which the mission service then reports
    as ``completed`` (100 %), and the navigation mode stays ``auto``."""
    nav = new_nav()
    mission, service = wire_mission(monkeypatch, idle_tractor(), nav, [FAR])
    legs = _stub_leg(monkeypatch, nav, result)
    await service.start_mission(mission.id)
    task = service.mission_tasks[mission.id]
    try:
        await wait_until(task.done, timeout=8.0)
        await wait_until(lambda: service.mission_statuses[mission.id].status != "running")

        assert service.mission_statuses[mission.id].status == "aborted", "reported as completed"
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        assert len(legs) == 1, "the loop went on after the leg said the machine was stopped"
    finally:
        await _end(task)


@pytest.mark.parametrize("refusal", ["revoked", "restored_estop"])
async def test_a_leg_the_tractor_refuses_for_good_ends_the_run_and_stands_the_machine_down(
    monkeypatch, refusal
):
    """N10 + N38. The refusal records no stop event (a revoke; a latch restored from the file), so
    the listener and the epoch checks are silent: only the leg's reading of the tractor's standing
    refusal can end it. Without it the leg retries every tick for ever and drives off the moment
    the refusal is lifted. The levers stay in force for 10 s, so only the leg's own stand-down
    puts them (and the PTO) back to neutral."""
    tractor, chip = _steady_tractor()
    nav = new_nav()
    nav.control_period_s = 0.02
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    nav.navigation_state.navigation_mode = NavigationMode.AUTO  # what execute_mission sets
    leg = asyncio.create_task(nav.go_to_waypoint(FAR))
    try:
        await wait_until(lambda: tractor.state.moving and tractor.state.blade_engaged, timeout=8.0)
        assert chip.regs[LEFT] != NEUTRAL[LEFT] or chip.regs[RIGHT] != NEUTRAL[RIGHT]
        epoch = stop_events.epoch()
        if refusal == "revoked":
            tractor.revoke()
        else:
            tractor.restore_latches(estop=True, interlock=None, why="test")
        assert stop_events.epoch() == epoch, "premise: no stop event, only the refusal"

        assert await _result(leg) is LegResult.REFUSED

        assert service.mission_statuses[mission.id].status == "aborted"
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
        if refusal == "revoked":  # (an E-stop latch refuses even a neutral lever command)
            assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT], (
                "the levers were left where the last command put them"
            )
    finally:
        await _end(leg)
        await tractor.shutdown()


async def test_an_ai_prediction_the_tractor_refuses_for_good_ends_the_run_and_stands_down(
    monkeypatch,
):
    """N10 + the refusal branch of ``apply_ai_prediction``: no loop is running (so the loop's own
    ``finally`` cannot stand the machine down) and no stop event was recorded."""
    tractor, chip = _steady_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    try:
        assert await nav.apply_ai_prediction(_good()) is True, "premise: the run drives"
        assert tractor.state.moving and tractor.state.blade_engaged
        epoch = stop_events.epoch()
        tractor.revoke()
        assert stop_events.epoch() == epoch, "premise: no stop event, only the refusal"

        assert await nav.apply_ai_prediction(_good()) is False

        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE, "the run was not ended"
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT], (
            "the levers were left where the last prediction put them"
        )
        assert tractor.state.blade_engaged is False
    finally:
        await tractor.shutdown()


async def test_an_ai_loop_whose_tractor_refuses_for_good_ends_by_itself(monkeypatch):
    """N10 through the real loop: it ends by itself (the loop's ``finally`` stands down)."""
    tractor, chip = _steady_tractor()
    nav, _ai = await _ai_run(monkeypatch, tractor)
    task = nav._ai_task
    try:
        await wait_until(lambda: tractor.state.moving, timeout=8.0)
        epoch = stop_events.epoch()
        tractor.revoke()
        assert stop_events.epoch() == epoch, "premise: no stop event, only the refusal"

        await _result(task)  # the loop ends by itself ...

        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        assert nav.autonomy_active() is False
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
    finally:
        await _end(task)
        await tractor.shutdown()


async def test_the_ai_loops_own_epoch_check_ends_the_loop_when_nothing_else_can(monkeypatch):
    """N31. The consumer of the predictions is a stand-in (so ``apply_ai_prediction``'s own epoch
    check cannot fire) and the listener is removed: only the check at the top of the loop is left
    to end the run, and every tick it lets through is a prediction applied after the stop."""
    nav, _ai = await _ai_run(monkeypatch, idle_tractor())
    task = nav._ai_task
    applied: list[ActionPrediction] = []

    async def apply(prediction: ActionPrediction) -> bool:
        applied.append(prediction)
        return True

    monkeypatch.setattr(nav, "apply_ai_prediction", apply)
    stop_events.remove_listener(nav._on_stop_event)
    try:
        await wait_until(lambda: len(applied) >= 3, timeout=8.0)
        stop_events.record_stop("emergency_stop", "api")  # a stop nobody told this service about
        at_stop = len(applied)
        assert nav.navigation_state.navigation_mode == NavigationMode.AI, "premise: not told"

        await _result(task)

        assert len(applied) == at_stop, "the loop applied a prediction after the stop"
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
    finally:
        await _end(task)


async def test_a_prediction_of_a_run_a_stop_ended_is_never_applied(monkeypatch):
    """N32. No loop runs, the listener is removed and the tractor itself is willing (a stop that
    was only recorded does not latch it): the consumer's own epoch check is the only thing that
    keeps a prediction made before the stop from reaching the servos."""
    tractor, chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    nav._ai_epoch = stop_events.epoch()  # an AI run that began before the stop
    stop_events.remove_listener(nav._on_stop_event)
    stop_events.record_stop("emergency_stop", "api")
    try:
        assert tractor.state.authorized and not tractor.state.emergency_stop_active
        before = list(chip.log)

        assert await nav.apply_ai_prediction(_good()) is False

        assert chip.log == before, "a prediction of a stopped run reached the servos"
        assert tractor.state.moving is False and tractor.state.blade_engaged is False
        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
    finally:
        await tractor.shutdown()
