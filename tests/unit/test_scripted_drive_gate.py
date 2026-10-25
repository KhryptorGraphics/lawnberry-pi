"""Scripted motion goes through the same safety path as an autonomous leg (audit SAF-11, S3).

The IMU calibration drive sent one command with a validity window of the whole step and slept:
no obstacle guard, no stop latch, no abort (a refused command just continued the sequence). Now
``IMUCalibrationService`` drives through ``NavigationService.drive_scripted``. Everything below
runs the REAL calibration service, the REAL ``NavigationService`` and the REAL ``RoboHATService``
over a fake serial with SIM_MODE off (``robohat_platform``): the proof is the lines that reach
the firmware, in order.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from _mission_rig import driving
from _nav_rig import (
    SensorStub,
    StubGuard,
    robohat_with_timeline,
    wait_until,
)

from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode
from backend.src.safety import stop_events
from backend.src.safety.obstacle_stop import ObstacleGuard, ObstacleStopMonitor
from backend.src.services import calibration_service as calibration_module
from backend.src.services import emergency_stop_service as ess
from backend.src.services.calibration_service import (
    CalibrationRefusedError,
    DriveControllerUnavailableError,
    IMUCalibrationService,
)
from backend.src.services.navigation_service import (
    NavigationService,
    ScriptedOutcome,
)
from backend.src.services.robohat_service import RoboHATService

NEUTRAL_LINE = "pwm,1500,1500"

#: A short manoeuvre with the same shape as the real one: a spin, then a forward sweep.
SPIN = {
    "name": "spin",
    "description": "spin",
    "duration": 1.5,
    "left_speed": 0.35,
    "right_speed": -0.35,
}
FORWARD = {
    "name": "forward",
    "description": "forward",
    "duration": 1.5,
    "left_speed": 0.4,
    "right_speed": 0.4,
}
REVERSE = {
    "name": "reverse",
    "description": "reverse",
    "duration": 1.0,
    "left_speed": -0.35,
    "right_speed": -0.35,
}


class World:
    """The differential platform with a real calibration service wired onto it."""

    def __init__(self, monkeypatch, sequence=(SPIN, FORWARD)) -> None:
        self.svc, self.fw, self.tractor, self.timeline = robohat_with_timeline(monkeypatch)
        self.nav = NavigationService()
        self.nav.control_period_s = 0.05
        self.guard = StubGuard(timeline=self.timeline)
        self.nav.attach_safety(obstacle_guard=self.guard)
        monkeypatch.setattr(NavigationService, "_instance", self.nav)
        monkeypatch.setattr(
            "backend.src.services.tractor_service.get_tractor_service", lambda: self.tractor
        )
        monkeypatch.setattr(calibration_module, "CALIBRATION_SEQUENCE", tuple(sequence))
        monkeypatch.setattr(calibration_module, "SETTLE_S", 0.0)
        self.sensors = SensorStub()
        self.service = IMUCalibrationService()

    def start(self) -> asyncio.Task:
        return asyncio.create_task(self.service.run(self.sensors))

    async def moving(self, lines: int = 3) -> None:
        await wait_until(lambda: len(driving(self.fw.sent)) >= lines)

    async def close(self) -> None:
        await self.svc.shutdown()


@pytest.fixture
async def world(monkeypatch):
    w = World(monkeypatch)
    yield w
    await w.close()


# ------------------------------------------------------- hold appears mid-drive: neutral at once


async def test_an_obstacle_hold_that_appears_mid_drive_ends_the_calibration_within_one_tick(world):
    task = world.start()
    await world.moving()
    world.guard.hold = True  # the obstacle appears while the mower is spinning
    injected = time.monotonic()

    result = await asyncio.wait_for(task, timeout=5.0)

    tl = world.timeline
    first_hold = tl.index("verdict", "hold")
    assert first_hold > 0, "the loop never saw the hold"
    neutral = tl.index("line", NEUTRAL_LINE, start=first_hold)
    assert neutral > first_hold, "neutral never followed the hold"
    between = tl.events[first_hold:neutral]
    assert not [e for e in between if e[0] == "verdict" and e[1] == "clear"], between
    assert not [e for e in between if e[0] == "line" and e[1] in driving(tl.lines())], between
    assert tl.events[neutral][2] - injected < 1.0, "the neutral was not prompt"
    # blade first, then neutral: the stand-down order every hold uses
    assert tl.index("line", "blade=off", start=first_hold) < neutral
    # and nothing moves the mower afterwards, whatever the guard does next
    after = len(tl.events)
    world.guard.hold = False
    await asyncio.sleep(0.4)
    assert not driving([e[1] for e in tl.events[after:] if e[0] == "line"])

    assert result["status"] == "aborted" and "held" in result["notes"]
    assert [s["name"] for s in result["steps"]] == ["spin"], "the forward sweep must not run"
    assert result["steps"][0]["outcome"] == "held"
    assert "test obstacle" in result["steps"][0]["reasons"]


async def test_an_aborted_calibration_does_not_resume_when_the_hold_clears(world):
    task = world.start()
    await world.moving()
    world.guard.hold = True
    await asyncio.wait_for(task, timeout=5.0)
    world.guard.hold = False
    world.fw.sent.clear()
    await asyncio.sleep(0.5)
    assert world.fw.sent == [], "an aborted manoeuvre resumed on its own"
    assert world.service.is_running() is False


# ------------------------------------------------------------ refused before anything moves


async def test_a_hold_that_is_already_set_refuses_the_calibration_and_moves_nothing(world):
    world.guard.hold = True
    with pytest.raises(CalibrationRefusedError, match="held.*test obstacle"):
        await world.service.run(world.sensors)
    assert world.fw.sent == [], "a refused calibration must not touch the firmware at all"
    assert world.sensors.reads == 0


def _latch_process_estop(w: World) -> None:
    core_globals._safety_state["emergency_stop_active"] = True


def _latch_robohat_hold(w: World) -> None:
    w.svc._estop_hold = True


def _lock_out_tractor(w: World) -> None:
    w.tractor.lock_out("safety wiring failed at start-up")


@pytest.mark.parametrize(
    ("latch", "why"),
    [
        (_latch_process_estop, "emergency stop"),
        (_latch_robohat_hold, "RoboHAT stop hold"),
        (_lock_out_tractor, "locked out"),
    ],
)
async def test_a_stop_latch_refuses_the_calibration_and_moves_nothing(world, latch, why):
    latch(world)
    with pytest.raises(CalibrationRefusedError, match=why):
        await world.service.run(world.sensors)
    assert world.fw.sent == []


async def test_an_autonomous_run_refuses_the_calibration(world):
    world.nav.navigation_state.navigation_mode = NavigationMode.AUTO
    with pytest.raises(CalibrationRefusedError, match="autonomous run"):
        await world.service.run(world.sensors)
    assert world.fw.sent == []


async def test_a_scripted_drive_is_refused_on_the_tractor_platform(world):
    world.tractor.enabled = True
    drive = await world.nav.drive_scripted(0.4, 0.4, 1.0)
    assert drive.outcome is ScriptedOutcome.REFUSED and "tractor platform" in drive.reasons[0]
    assert world.fw.sent == []


async def test_a_second_scripted_drive_is_refused_while_one_runs(world):
    first = asyncio.create_task(world.nav.drive_scripted(0.4, 0.4, 1.0))
    await world.moving(1)
    second = await world.nav.drive_scripted(0.4, 0.4, 1.0)
    assert second.outcome is ScriptedOutcome.REFUSED and "another" in second.reasons[0]
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


async def test_the_calibration_needs_the_drive_controller(monkeypatch):
    w = World(monkeypatch)
    try:
        w.svc.status.serial_connected = False
        with pytest.raises(DriveControllerUnavailableError):
            await w.service.run(w.sensors)
        assert w.fw.sent == []
    finally:
        await w.close()


# ------------------------------------------------------------- a safety stop ends the manoeuvre


async def test_an_estop_mid_calibration_ends_it_and_a_clear_does_not_resume_it(world):
    task = world.start()
    await world.moving()

    await ess.execute_emergency_stop("api")  # the operator hits the E-stop
    stop_index = world.timeline.index("line", "blade=off")  # the stop's own first line is blade off
    assert stop_index >= 0
    result = await asyncio.wait_for(task, timeout=5.0)

    assert result["status"] == "aborted" and "stopped" in result["notes"]
    assert [s["name"] for s in result["steps"]] == ["spin"]
    assert result["steps"][0]["outcome"] == "stopped"
    tl = world.timeline
    stopped_at = tl.index("line", NEUTRAL_LINE, start=stop_index)
    assert stopped_at > 0
    assert not driving([e[1] for e in tl.events[stopped_at:] if e[0] == "line"]), (
        "the mower moved after the stop"
    )

    assert await world.svc.clear_emergency() is True  # the authenticated clear
    world.fw.sent.clear()
    await asyncio.sleep(0.5)
    assert driving(world.fw.sent) == [], "a clear resumed the calibration"


async def test_a_safety_stop_that_never_reaches_the_robohat_still_ends_the_drive(world):
    """A revoke, a lock-out or a tilt latch records a stop but sends the RoboHAT nothing."""
    task = world.start()
    await world.moving()
    world.fw.sent.clear()
    stop_events.record_stop("authorization revoked", "operator")
    result = await asyncio.wait_for(task, timeout=5.0)
    assert result["status"] == "aborted" and result["steps"][0]["outcome"] == "stopped"
    assert world.fw.sent[:2] == ["blade=off", NEUTRAL_LINE], "the drive must stand itself down"
    assert driving(world.fw.sent) == [], "a command went out after the stop"


async def test_a_stop_between_two_steps_ends_the_routine_before_the_next_step_drives(monkeypatch):
    w = World(monkeypatch, sequence=(SPIN, FORWARD))
    try:

        async def read_all_sensors():
            # The stop lands while the routine reads the IMU between the two steps.
            stop_events.record_stop("tilt latch", "test")
            return SimpleNamespace(imu=None)

        w.sensors.read_all_sensors = read_all_sensors
        monkeypatch.setitem(SPIN, "duration", 0.3)
        result = await asyncio.wait_for(w.service.run(w.sensors), timeout=5.0)
        assert result["status"] == "aborted"
        forward = result["steps"][-1]
        assert forward["name"] == "forward" and forward["outcome"] == "stopped"
        assert forward["command_accepted"] is False
        forward_line = "pwm,{},{}".format(*RoboHATService._mix_arcade_to_pwm(0.4, 0.4))
        assert forward_line not in w.fw.sent, "the forward sweep drove after the stop"
    finally:
        await w.close()


async def test_a_clear_of_the_stop_does_not_let_the_next_calibration_inherit_the_old_epoch(world):
    """A routine captures the epoch when it starts: an OLD stop must not abort a NEW routine."""
    stop_events.record_stop("emergency_stop", "api")  # long over, and cleared
    result = await asyncio.wait_for(world.service.run(world.sensors), timeout=8.0)
    assert result["status"] in ("completed", "pending"), result


# --------------------------------------------------------------- speed cap and the blade


async def test_the_calibration_drive_is_capped_to_the_stopping_distance_model(monkeypatch):
    w = World(monkeypatch, sequence=(SPIN, FORWARD, REVERSE))
    try:
        w.guard.cap = 0.2  # an obstacle limits the forward speed to 0.2 m/s
        result = await asyncio.wait_for(w.service.run(w.sensors), timeout=8.0)
        assert result["status"] in ("completed", "pending"), result
        moving = [line for line in w.fw.sent if line.startswith("pwm,") and line != NEUTRAL_LINE]
        mix = RoboHATService._mix_arcade_to_pwm
        full = 0.8  # NavigationService.max_speed
        spin_capped = mix(0.2 / full, -0.2 / full)
        forward_capped = mix(0.2 / full, 0.2 / full)
        reverse_free = mix(-0.35, -0.35)  # reverse is not limited: nothing senses behind
        want = {f"pwm,{s},{t}" for s, t in (spin_capped, forward_capped, reverse_free)}
        assert set(moving) == want, (set(moving), want)
        uncapped = {f"pwm,{s},{t}" for s, t in (mix(0.35, -0.35), mix(0.4, 0.4))}
        assert not (set(moving) & uncapped), "a forward step ran at its uncapped speed"
    finally:
        await w.close()


async def test_the_blade_is_dropped_before_the_first_calibration_command(world):
    result = await asyncio.wait_for(world.service.run(world.sensors), timeout=8.0)
    assert result["status"] in ("completed", "pending"), result
    assert world.fw.sent[0] == "blade=off", "scripted motion must never run over a live blade"
    assert "blade=on" not in world.fw.sent


async def test_the_whole_routine_ends_neutral_with_the_blade_off(world):
    await asyncio.wait_for(world.service.run(world.sensors), timeout=8.0)
    assert world.fw.sent[-2:] == ["blade=off", NEUTRAL_LINE]
    assert world.service.is_running() is False


# --------------------------------------- the gate scope of an open-loop (non-navigating) drive


async def test_a_stale_heading_and_a_bad_gnss_fix_do_not_hold_an_open_loop_calibration(world):
    """The drive neither navigates nor steers by heading: GNSS and heading age are not inputs."""
    world.nav.navigation_state.heading_updated_at = datetime.now(UTC) - timedelta(minutes=5)
    assert world.nav.motion_gate().hold, "premise: a navigating drive WOULD be held"
    world.nav.attach_safety(
        obstacle_guard=world.guard,
        gps_monitor=SimpleNamespace(
            assess=lambda: SimpleNamespace(healthy=False, reasons=("fix is 30.0s old",))
        ),
    )
    assert any(r.startswith("gnss") for r in world.nav.motion_gate().reasons)
    result = await asyncio.wait_for(world.service.run(world.sensors), timeout=8.0)
    assert result["status"] in ("completed", "pending"), result
    assert driving(world.fw.sent), "the calibration did not drive"


async def test_a_guard_that_raises_holds_the_calibration(world):
    def broken():
        raise RuntimeError("sensor bus fell over")

    world.guard.assess = broken
    with pytest.raises(CalibrationRefusedError, match="obstacle guard failed"):
        await world.service.run(world.sensors)
    assert world.fw.sent == []


# ------------------------------------------------------------ the drive itself, one step at a time


async def test_a_command_the_controller_refuses_fails_the_step_and_ends_the_routine(world):
    task = world.start()
    await world.moving()
    world.svc._estop_hold = True  # the controller now refuses every drive command
    result = await asyncio.wait_for(task, timeout=5.0)
    assert result["status"] == "aborted"
    assert result["steps"][0]["outcome"] == "failed"
    assert [s["name"] for s in result["steps"]] == ["spin"]


async def test_a_cancelled_scripted_drive_stands_the_platform_down(world):
    task = asyncio.create_task(world.nav.drive_scripted(0.4, 0.4, 5.0))
    await world.moving(1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert world.fw.sent[-2:] == ["blade=off", NEUTRAL_LINE], world.fw.sent[-4:]
    assert world.nav._scripted_drive_active is False


async def test_a_command_is_re_sent_every_tick_so_a_dead_caller_lets_the_drive_lapse(world):
    drive = await world.nav.drive_scripted(0.4, 0.4, 0.5)
    assert drive.outcome is ScriptedOutcome.DONE
    sent = [line for line in world.fw.sent if line.startswith("pwm,") and line != NEUTRAL_LINE]
    assert len(sent) >= 5, "one command per tick, not one per step"
    assert drive.commands_accepted == len(sent)
    # the command in force when the call returned lapses by itself (default validity 0.5 s)
    world.fw.sent.clear()
    await asyncio.sleep(0.9)
    assert NEUTRAL_LINE in world.fw.sent


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
async def test_a_non_finite_scripted_drive_is_refused(world, bad):
    for args in ((bad, 0.4, 1.0), (0.4, bad, 1.0), (0.4, 0.4, bad)):
        drive = await world.nav.drive_scripted(*args)
        assert drive.outcome is ScriptedOutcome.REFUSED
    assert world.fw.sent == []


# ---------------------------------------------- the monitor's hold handler stops a scripted drive


async def test_the_obstacle_monitor_stops_a_scripted_drive_before_its_own_next_tick(monkeypatch):
    """``enforce_motion_hold`` acts the moment a hold begins, not at the drive's next tick."""
    svc, fw, tractor, timeline = robohat_with_timeline(monkeypatch)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    clock = SimpleNamespace(t=1000.0)
    guard = ObstacleGuard(clock=lambda: clock.t)
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    nav = NavigationService()
    nav.control_period_s = 30.0  # the drive's own loop cannot act inside this test
    nav.attach_safety(obstacle_guard=guard)

    def see(mm: float) -> None:
        guard.observe_range("tof_left", mm / 1000.0, clock.t)
        guard.observe_range("tof_right", mm / 1000.0, clock.t)

    for _ in range(26):  # clear ground for longer than the clear dwell
        clock.t += 0.1
        see(2000.0)
        guard.assess()
    assert not guard.assess().hold

    monitor = ObstacleStopMonitor(
        guard, sensor_provider=lambda: None, hold_handlers=(nav.enforce_motion_hold,)
    )
    drive = asyncio.create_task(nav.drive_scripted(0.4, 0.4, 20.0))
    try:
        await wait_until(lambda: driving(fw.sent))
        fw.sent.clear()
        clock.t += 0.1
        see(100.0)  # an obstacle 10 cm ahead
        await monitor.step()
        assert fw.sent[:2] == ["blade=off", NEUTRAL_LINE], fw.sent
        assert not drive.done(), "premise: the drive's own loop has not seen anything yet"
    finally:
        drive.cancel()
        await asyncio.gather(drive, return_exceptions=True)
        await svc.shutdown()


async def test_a_hold_with_no_run_and_no_scripted_drive_commands_nothing(monkeypatch):
    svc, fw, tractor, _tl = robohat_with_timeline(monkeypatch)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    nav = NavigationService()
    await nav.enforce_motion_hold()  # a person's manual drive is left alone
    assert fw.sent == []
    await svc.shutdown()


# ------------------------------------------------------------------------------- the route


async def _post(monkeypatch, world) -> httpx.Response:
    from backend.src.main import app
    from backend.src.services.websocket_hub import websocket_hub

    async def sensors():
        return world.sensors

    monkeypatch.setattr(websocket_hub, "_ensure_sensor_manager", sensors)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.post("/api/v2/maintenance/imu/calibrate")


async def test_the_route_answers_423_and_moves_nothing_while_the_guard_holds(monkeypatch, world):
    world.guard.hold = True
    response = await _post(monkeypatch, world)
    assert response.status_code == 423, response.text
    assert "Calibration refused" in response.json()["detail"]
    assert "test obstacle" in response.json()["detail"]
    assert world.fw.sent == []


async def test_the_route_answers_an_aborted_run_as_200_with_the_reason(monkeypatch, world):
    async def hold_soon():
        await world.moving()
        world.guard.hold = True

    watcher = asyncio.create_task(hold_soon())
    response = await _post(monkeypatch, world)
    await watcher
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "aborted" and "held" in body["notes"]
    assert body["steps"][0]["outcome"] == "held"


async def test_the_route_answers_503_when_the_drive_controller_is_offline(monkeypatch, world):
    world.svc.status.serial_connected = False
    response = await _post(monkeypatch, world)
    assert response.status_code == 503, response.text
    assert world.fw.sent == []
