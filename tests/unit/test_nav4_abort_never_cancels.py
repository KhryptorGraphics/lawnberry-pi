"""Stopping a run never abandons an actuator write: no cancelled task (review 3, NAV-02).

``MissionService.abort_mission`` (the operator's abort, the autonomy stop route, the jobs timeout),
``NavigationService.stop_ai_navigation`` and the shutdown's ``_stop_autonomy`` cancelled the
running task and then stopped the platform. The PCA9685 transfer runs in a worker thread:
cancelling the task releases the actuation lock while the transfer is still on the bus, the
stop's neutral takes the lock and goes out, and the abandoned transfer lands AFTER it. The state
(and so the UI) said neutral; the chip drove forward; no deadman was armed (a neutral command
clears it). ``abort_autonomy``'s own docstring says why a task must not be cancelled;
``ABORT_UNWIND_TIMEOUT_S`` was defined for the wait that was planned instead and used nowhere.

Now a run is ended cooperatively (its status or mode, which its loop reads every tick), the
platform is stopped at once, the run is awaited (never cancelled) for ``ABORT_UNWIND_TIMEOUT_S``,
and the stop is repeated after it as the last word. ``StallablePCA9685`` is the repo's model of a
transfer that cannot be recalled: a stalled write that the service abandons still lands later.
"""

from __future__ import annotations

import asyncio
import time

from _mission_rig import FAR, new_nav, wire_mission
from _nav_rig import wait_until
from _tractor_fakes import LEFT, NEUTRAL, RIGHT, mowing_tractor

import backend.src.main as main
from backend.src.models import NavigationMode
from backend.src.models.action_prediction import ActionPrediction
from backend.src.nav.location_features import CoverageMap
from backend.src.services import mission_service as mission_module
from backend.src.services.mission_service import MissionService
from backend.src.services.navigation_service import ABORT_UNWIND_TIMEOUT_S


async def _mowing_mission(monkeypatch):
    """A real mission loop on the tractor, 20 Hz, in its steady state."""
    tractor, chip = mowing_tractor()
    nav = new_nav()
    nav.control_period_s = 0.05
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    service.mission_tasks[mission.id] = task
    await wait_until(lambda: tractor.state.moving and len(chip.writes(LEFT)) >= 3, timeout=5.0)
    return tractor, chip, nav, mission, service, task


def _last_write(chip, channel: int) -> int:
    return chip.writes(channel)[-1]


async def test_an_abort_with_a_write_in_flight_cannot_leave_the_lever_off_neutral(monkeypatch):
    tractor, chip, _nav, mission, service, task = await _mowing_mission(monkeypatch)
    try:
        steady = _last_write(chip, LEFT)
        assert steady != NEUTRAL[LEFT], "premise: the mission is driving"
        gate = chip.stall(LEFT, steady)  # the next left-lever write hangs on the bus
        await asyncio.wait_for(chip.stalled.wait(), 5.0)

        abort = asyncio.create_task(service.abort_mission(mission.id))
        await asyncio.sleep(
            0.15
        )  # the abort has done what it can; the transfer still holds the bus
        assert not abort.done(), "the stop's neutral waits behind the transfer, it does not pass it"
        gate.set()  # the transfer completes
        await asyncio.wait_for(abort, 5.0)
        await wait_until(task.done, timeout=5.0)

        assert steady in chip.writes(LEFT), "the hung transfer really landed"
        assert _last_write(chip, LEFT) == NEUTRAL[LEFT], "and the stop's neutral came after it"
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
        assert tractor.state.left_lever == 0.0 and tractor.state.blade_engaged is False
        assert not task.cancelled(), "the run ended by itself; nothing cancelled it mid-write"
        assert service.mission_statuses[mission.id].status == "aborted"
        assert mission.id not in service.mission_tasks
    finally:
        chip.release_all()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tractor.shutdown()


async def test_stop_ai_navigation_with_a_write_in_flight_cannot_leave_the_lever_off_neutral(
    monkeypatch,
):
    tractor, chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)

    class FakeAI:
        async def infer(self, frame, coverage_grid=None, datum=None):
            return ActionPrediction(steering=0.0, throttle=0.9, blade=False, confidence=0.9)

        async def disable(self):
            return True

    class FakeRecorder:
        async def capture_frame(self):
            return object()

    monkeypatch.setattr(
        "backend.src.services.ai_inference_service.get_ai_inference_service", lambda: FakeAI()
    )
    monkeypatch.setattr(
        "backend.src.services.perimeter_recorder.get_perimeter_recorder", lambda: FakeRecorder()
    )
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    nav._ai_datum = (40.0, -83.0)
    nav._coverage = CoverageMap(nav._ai_datum)
    nav._ai_loop_hz = 20.0
    nav._ai_task = task = asyncio.create_task(nav._ai_inference_loop())
    try:
        await wait_until(lambda: tractor.state.moving and len(chip.writes(LEFT)) >= 3, timeout=5.0)
        steady = _last_write(chip, LEFT)
        gate = chip.stall(LEFT, steady)
        await asyncio.wait_for(chip.stalled.wait(), 5.0)

        stop = asyncio.create_task(nav.stop_ai_navigation())
        await asyncio.sleep(0.15)
        gate.set()
        assert await asyncio.wait_for(stop, 5.0) is True

        assert steady in chip.writes(LEFT) and _last_write(chip, LEFT) == NEUTRAL[LEFT]
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
        assert task.done() and not task.cancelled(), "the loop ended by itself"
        assert (
            nav._ai_task is None and nav.navigation_state.navigation_mode == NavigationMode.MANUAL
        )
    finally:
        chip.release_all()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tractor.shutdown()


async def test_the_shutdown_ends_a_mission_without_abandoning_its_write(monkeypatch):
    tractor, chip, nav, mission, service, task = await _mowing_mission(monkeypatch)
    monkeypatch.setattr(mission_module, "_mission_service_instance", service)
    monkeypatch.setattr(main.NavigationService, "_instance", nav)
    try:
        steady = _last_write(chip, LEFT)
        gate = chip.stall(LEFT, steady)
        await asyncio.wait_for(chip.stalled.wait(), 5.0)

        stopping = asyncio.create_task(main._stop_autonomy())
        await asyncio.sleep(0.15)
        gate.set()
        await asyncio.wait_for(stopping, 10.0)

        assert task.done() and not task.cancelled()
        assert steady in chip.writes(LEFT) and _last_write(chip, LEFT) == NEUTRAL[LEFT]
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
    finally:
        chip.release_all()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tractor.shutdown()


async def test_the_shutdown_cancels_only_a_task_that_ignored_the_stop(monkeypatch):
    """The last resort: a task that never reads its status would command the parked machine."""
    tractor, _chip = mowing_tractor()
    nav = new_nav()
    service = MissionService(nav)
    monkeypatch.setattr(mission_module, "_mission_service_instance", service)
    monkeypatch.setattr(main.NavigationService, "_instance", nav)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)

    async def deaf_loop():
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(deaf_loop())
    service.mission_tasks["deaf"] = task
    started = time.monotonic()

    await asyncio.wait_for(main._stop_autonomy(), 10.0)

    assert task.cancelled(), "it ignored the cooperative stop, so it was cancelled"
    assert time.monotonic() - started >= ABORT_UNWIND_TIMEOUT_S * 0.9, "but only after the bound"
    await tractor.shutdown()


# ------------------------------------------------------------- the stop after the run is over


async def test_the_last_stop_comes_after_the_run_has_ended(monkeypatch):
    """A tick that was already past its status check still commands once, after the first stop."""
    tractor, _chip = mowing_tractor()
    nav = new_nav()
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])

    async def last_command_of_a_late_tick():
        while service.mission_statuses[mission.id].status == "running":
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.15)  # the abort's first stop has long been issued
        await tractor.set_levers(0.5, 0.5, source="navigation", duration_ms=5000)

    task = asyncio.create_task(last_command_of_a_late_tick())
    service.mission_tasks[mission.id] = task

    await service.abort_mission(mission.id)

    assert task.done() and not task.cancelled()
    assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0, (
        "the run's late command was left in force: the stop after the run ended is missing"
    )
    await tractor.shutdown()


async def test_await_unwind_gives_up_at_its_bound_and_never_cancels(monkeypatch):
    nav = new_nav()
    ended = asyncio.Event()

    async def slow_to_end():
        await ended.wait()

    task = asyncio.create_task(slow_to_end())
    try:
        assert await nav.await_unwind(task, timeout_s=0.05) is False
        assert not task.done() and not task.cancelled(), "left running, not cancelled"
        ended.set()
        assert await nav.await_unwind(task, timeout_s=2.0) is True
        assert await nav.await_unwind(None, task) is True
    finally:
        ended.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_new_ai_run_is_refused_while_the_previous_one_is_still_ending(monkeypatch):
    nav = new_nav()
    nav._ai_task = asyncio.create_task(asyncio.sleep(30))
    try:
        assert await nav.start_ai_navigation() is False
        assert nav.navigation_state.navigation_mode != NavigationMode.AI
    finally:
        nav._ai_task.cancel()
        await asyncio.gather(nav._ai_task, return_exceptions=True)


# ----------------------------------------------- a pause and a resume never make two loops


async def test_a_resume_right_after_a_pause_never_runs_two_loops_for_one_mission(monkeypatch):
    tractor, _chip = mowing_tractor()
    nav = new_nav()
    nav.control_period_s = 0.05
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    active = peak = 0
    real_execute = nav.execute_mission

    async def counting(m):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            return await real_execute(m)
        finally:
            active -= 1

    monkeypatch.setattr(nav, "execute_mission", counting)
    try:
        await service.start_mission(mission.id)
        await wait_until(lambda: tractor.state.moving, timeout=5.0)

        await service.pause_mission(mission.id)
        await service.resume_mission(mission.id)  # inside the paused loop's last tick
        await wait_until(lambda: tractor.state.moving, timeout=5.0)
        await asyncio.sleep(0.4)

        assert peak == 1, f"{peak} loops drove the machine at once"
        task = service.mission_tasks[mission.id]
        assert not task.done(), (
            "the run that resumed is registered (the old one's callback left it)"
        )
        assert service.mission_statuses[mission.id].status == "running"
    finally:
        await service.abort_mission(mission.id)
        await tractor.shutdown()
