"""Mission end, pause and abort switch the blade off, blade first and then the drive; the final
waypoint completes the mission; a held or unconfirmed gate never lets the differential
platform's blade on (audit SAF-02, SAF-03).

Both platforms run a real ``execute_mission`` over a real ``MissionService``. The differential
platform runs the REAL ``RoboHATService`` over a fake serial with SIM_MODE off, so the blade
handling is observed as the lines that reach the firmware, not as a simulation shortcut.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from _mission_rig import FAR, FAR_POSITION, driving, new_nav, robohat_platform, wire_mission
from _tractor_fakes import LEFT, RIGHT, mowing_tractor

from backend.src.models import NavigationMode
from backend.src.models.navigation_state import PathStatus
from backend.src.safety.obstacle_stop import ObstacleStatus


def record_order(tractor, pca) -> list[str]:
    """The order in which the blade relay and the drive neutral landed (after ``order.clear()``)."""
    order: list[str] = []
    real_set = tractor.blade_pto.set

    def blade_set(value):
        order.append(f"blade:{bool(value)}")
        return real_set(value)

    tractor.blade_pto.set = blade_set
    real_pwm = pca.set_pwm_us

    async def pwm(channel, us):
        if channel in (LEFT, RIGHT) and us == 1500:
            order.append("neutral")
        await real_pwm(channel, us)

    pca.set_pwm_us = pwm
    return order


async def run_mowing(monkeypatch):
    tractor, pca = mowing_tractor()
    order = record_order(tractor, pca)
    nav = new_nav()
    mission, service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    await asyncio.sleep(0.6)
    assert tractor.state.blade_engaged, "the mission is mowing"
    order.clear()
    return tractor, nav, mission, service, task, order


async def finish(tractor, task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await tractor.shutdown()


async def test_the_final_waypoint_completes_the_mission_blade_off_first(monkeypatch):
    tractor, nav, _mission, _service, task, order = await run_mowing(monkeypatch)
    try:
        nav.navigation_state.current_position = FAR_POSITION  # the machine arrives
        await asyncio.wait_for(task, timeout=3.0)  # it used to loop on the last waypoint for ever
        state = nav.navigation_state
        assert state.navigation_mode == NavigationMode.IDLE
        assert state.path_status == PathStatus.COMPLETED
        assert tractor.state.blade_engaged is False and tractor.state.moving is False
        assert order.index("blade:False") < order.index("neutral"), order
        ended = len(order)
        await asyncio.sleep(0.6)
        assert order[ended:] == [], "the finished mission went on commanding the machine"
    finally:
        await finish(tractor, task)


async def test_a_paused_mission_stops_the_blade_and_the_drive(monkeypatch):
    tractor, _nav, mission, service, task, order = await run_mowing(monkeypatch)
    try:
        service.mission_statuses[mission.id].status = "paused"
        await asyncio.wait_for(task, timeout=3.0)
        assert tractor.state.blade_engaged is False
        assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0
        assert order.index("blade:False") < order.index("neutral"), order
    finally:
        await finish(tractor, task)


async def test_an_aborted_mission_stops_the_blade_and_the_drive(monkeypatch):
    tractor, _nav, mission, service, task, order = await run_mowing(monkeypatch)
    try:
        await service.abort_mission(mission.id)
        await asyncio.wait_for(task, timeout=3.0)
        assert tractor.state.blade_engaged is False
        assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0
        assert order.index("blade:False") < order.index("neutral"), order
    finally:
        await finish(tractor, task)


# ---------------------------------------------- the differential platform's blade (SAF-02, SAF-03)


class Guard:
    """An obstacle guard whose verdict the test flips."""

    def __init__(self, *, hold: bool) -> None:
        self.hold = hold

    def assess(self):
        status = ObstacleStatus.STOP if self.hold else ObstacleStatus.CLEAR
        return SimpleNamespace(
            hold=self.hold,
            speed_cap_mps=0.0 if self.hold else None,
            status=status,
            reasons=("test obstacle",) if self.hold else (),
        )


async def test_a_gate_that_holds_from_the_start_never_switches_the_blade_on(monkeypatch):
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    nav.attach_safety(obstacle_guard=Guard(hold=True))
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await asyncio.sleep(0.9)
        assert driving(fw.sent) == [], "the blade or the drive was commanded under a hold"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await svc.shutdown()


async def test_a_hold_drops_the_blade_before_the_drive_and_the_release_restores_it(monkeypatch):
    svc, fw, tractor = robohat_platform(monkeypatch)
    nav = new_nav()
    guard = Guard(hold=False)
    nav.attach_safety(obstacle_guard=guard)
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await asyncio.sleep(0.8)
        assert "blade=on" in fw.sent and driving(fw.sent), "the clear gate lets the leg mow"

        guard.hold = True
        fw.sent.clear()
        await asyncio.sleep(0.8)
        assert fw.sent[:2] == ["blade=off", "pwm,1500,1500"], fw.sent
        assert driving(fw.sent) == [], "the blade or the drive came back under a hold"

        guard.hold = False
        fw.sent.clear()
        await asyncio.sleep(1.2)
        assert "blade=on" in fw.sent, "the release must restore the blade for the current leg"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await svc.shutdown()
