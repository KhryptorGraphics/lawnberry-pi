"""The differential platform's drive and blade handling, observed as the lines on the wire.

PR #20 second review, survivors N6/N6b (``set_speed`` under a hold), N15b (the obstacle monitor's
hold handler), N17/N11b (a stand-down drops the RoboHAT blade), N18/N19 (the leg's blade command).
The audit says the differential platform is the deployed one, yet the old tests ran under
``SIM_MODE=1`` where ``_robohat_blade`` and ``set_speed`` never reach a controller. Every test
here runs the REAL ``RoboHATService`` over a fake serial with SIM_MODE off (``robohat_platform``)
and looks at what the firmware would have received.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from _mission_rig import driving, new_nav, wire_mission
from _nav_rig import StubGuard, robohat_with_timeline, wait_until

from backend.src.models import NavigationMode, Position
from backend.src.models.mission import MissionWaypoint
from backend.src.services.robohat_service import RoboHATService

NEUTRAL = "pwm,1500,1500"

#: Two legs about 1.1 km apart, due north of the start (heading 0): the first mows, the second not.
LEG_MOW = MissionWaypoint(lat=40.01, lon=-83.0, blade_on=True)
LEG_TRANSIT = MissionWaypoint(lat=40.02, lon=-83.0, blade_on=False)


def line(left: float, right: float) -> str:
    """The pwm line RoboHATService writes for normalized wheel speeds (full scale = 1.0)."""
    return "pwm,{},{}".format(*RoboHATService._mix_arcade_to_pwm(left, right))


# ------------------------------------------------------------------------- N6 / N6b: set_speed


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (0.4, 0.4),  # forward
        (-0.4, -0.4),  # pure reverse: limit_pair returns it unchanged, only the hold zeroes it
        (0.3, -0.3),  # a spin
        (-0.2, -0.5),  # reverse with a curve
        (0.1, 0.6),
    ],
)
async def test_set_speed_under_a_hold_is_neutral_whatever_the_demand(monkeypatch, left, right):
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.attach_safety(obstacle_guard=StubGuard(hold=True))
    try:
        await nav.set_speed(left, right)
        assert fw.sent == [NEUTRAL], f"a hold must command neutral, not {fw.sent}"
        assert nav.navigation_state.target_velocity == 0.0
    finally:
        await svc.shutdown()


async def test_set_speed_with_a_cap_scales_forward_and_leaves_reverse_alone(monkeypatch):
    """The control for the hold test: a CAP is not a HOLD (reverse is not limited, by design)."""
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.attach_safety(obstacle_guard=StubGuard(cap=0.2))
    try:
        await nav.set_speed(0.8, 0.8)
        await nav.set_speed(-0.4, -0.4)
        assert fw.sent == [line(0.25, 0.25), line(-0.5, -0.5)]
    finally:
        await svc.shutdown()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
async def test_set_speed_refuses_a_non_finite_demand_instead_of_driving_full_speed(
    monkeypatch, bad
):
    """``min``/``max`` clamp a NaN to full scale: a NaN model output used to be full speed."""
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    try:
        for args in ((bad, bad), (bad, 0.2), (0.2, bad)):
            with pytest.raises(ValueError, match="non-finite"):
                await nav.set_speed(*args)
        assert fw.sent == []
    finally:
        await svc.shutdown()


# ------------------------------------------------------------------ N15b: the monitor's handler


async def test_the_obstacle_monitors_handler_stands_a_running_mission_down_by_itself(monkeypatch):
    """No planner tick is running: only ``enforce_motion_hold`` can stop the platform."""
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.attach_safety(obstacle_guard=StubGuard(hold=True))
    nav.navigation_state.navigation_mode = NavigationMode.AUTO  # a mission owns the machine
    nav._robohat_blade_on = True
    try:
        await nav.enforce_motion_hold()
        assert fw.sent == ["blade=off", NEUTRAL], "blade off first, then the drive to neutral"
        assert nav._robohat_blade_on is False

        await nav.enforce_motion_hold()  # again inside 250 ms: not re-sent every 50 ms cycle
        assert fw.sent == ["blade=off", NEUTRAL]
        await asyncio.sleep(0.3)
        await nav.enforce_motion_hold()  # still held: asserted again (retries are level-triggered)
        assert fw.sent == ["blade=off", NEUTRAL, "blade=off", NEUTRAL]
    finally:
        await svc.shutdown()


@pytest.mark.parametrize("mode", [NavigationMode.IDLE, NavigationMode.MANUAL])
async def test_the_obstacle_monitors_handler_leaves_a_persons_manual_drive_alone(monkeypatch, mode):
    svc, fw, _tractor, _tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.attach_safety(obstacle_guard=StubGuard(hold=True))
    nav.navigation_state.navigation_mode = mode
    try:
        await nav.enforce_motion_hold()
        assert fw.sent == []
    finally:
        await svc.shutdown()


# --------------------------------------------- N18 / N19: the leg's blade state and its restore


async def run_legs(monkeypatch, legs, guard=None):
    svc, fw, tractor, tl = robohat_with_timeline(monkeypatch)
    nav = new_nav()
    nav.control_period_s = 0.05
    nav.attach_safety(obstacle_guard=guard or StubGuard())
    mission, service = wire_mission(monkeypatch, tractor, nav, legs)
    task = asyncio.create_task(nav.execute_mission(mission))
    return svc, fw, nav, task, tl


async def finish(svc, task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await svc.shutdown()


async def test_a_leg_without_a_blade_never_switches_the_blade_on(monkeypatch):
    svc, fw, nav, task, _tl = await run_legs(monkeypatch, [LEG_TRANSIT])
    try:
        await wait_until(lambda: len(driving(fw.sent)) >= 4)
        assert "blade=on" not in fw.sent
        assert "blade=off" in fw.sent, (
            "the leg's own blade state (off) is sent once the gate is clear"
        )
        assert fw.sent.count("blade=off") == 1, "and only on a change, not on every tick"
    finally:
        await finish(svc, task)


async def test_a_blade_less_leg_stays_blade_less_through_a_hold_and_its_release(monkeypatch):
    guard = StubGuard()
    svc, fw, nav, task, _tl = await run_legs(monkeypatch, [LEG_TRANSIT], guard)
    try:
        await wait_until(lambda: len(driving(fw.sent)) >= 3)
        guard.hold = True
        await asyncio.sleep(0.4)
        guard.hold = False
        fw.sent.clear()
        await wait_until(lambda: len(driving(fw.sent)) >= 3)  # the release: driving again
        assert "blade=on" not in fw.sent, (
            "a release must not switch on a blade the leg never asked for"
        )
    finally:
        await finish(svc, task)


async def test_the_blade_follows_each_legs_flag_on_then_off(monkeypatch):
    svc, fw, nav, task, _tl = await run_legs(monkeypatch, [LEG_MOW, LEG_TRANSIT])
    try:
        await wait_until(lambda: "blade=on" in fw.sent and len(driving(fw.sent)) >= 3)
        assert fw.sent.count("blade=on") == 1, "blade-on is sent on the change, not every tick"
        before = len(fw.sent)
        nav.navigation_state.current_position = Position(
            latitude=LEG_MOW.lat, longitude=LEG_MOW.lon
        )
        await wait_until(lambda: "blade=off" in fw.sent[before:])
        after_off = fw.sent.index("blade=off", before)
        await asyncio.sleep(0.3)
        assert "blade=on" not in fw.sent[after_off:], "the transit leg must not mow"
        assert driving(fw.sent[after_off:]), "and the machine keeps driving it"
    finally:
        await finish(svc, task)


async def test_a_hold_release_restores_the_blade_for_a_leg_that_mows(monkeypatch):
    guard = StubGuard()
    svc, fw, nav, task, _tl = await run_legs(monkeypatch, [LEG_MOW], guard)
    try:
        await wait_until(lambda: "blade=on" in fw.sent)
        guard.hold = True
        await wait_until(lambda: fw.sent.count("blade=off") >= 1)
        guard.hold = False
        await wait_until(lambda: fw.sent.count("blade=on") >= 2)
        assert fw.sent.index("blade=off") < len(fw.sent) - 1
    finally:
        await finish(svc, task)
