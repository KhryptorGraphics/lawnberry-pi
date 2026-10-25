"""The tractor's PTO obeys the navigation gate's permit for every source (review 3, NAV-05).

Round 4 made ``NavigationService.blade_permit`` a standing reason for the IBT-4 while an autonomous
run owns the machine, so a client that repeats ON cannot undo a hold's stand-down. On the tractor
platform the PTO relay IS the blade, and nothing attached the permit there: ``engage_blade(True)``
consults the obstacle limiter, which exempts a person's own commands (``operator``/``teleop``), so
an operator's ON re-engaged the PTO during an obstacle hold and the mission loop dropped it again on
its next 5 Hz tick: the PTO relay followed whoever wrote last, the fight the IBT-4 decision closed.

The tractor is the real service over the stallable PCA9685; the navigation service and the mission
loop are real; only the guard's verdict is the test's.
"""

from __future__ import annotations

import asyncio

from _mission_rig import FAR, new_nav, wire_mission
from _nav_rig import StubGuard, live_sensors, real_lifespan, wait_until
from _tractor_fakes import mowing_tractor

from backend.src.services.navigation_service import BLADE_PERMIT_NAME


class Rig:
    """A mowing mission on the tractor, the obstacle guard wired as the lifespan wires it."""

    def __init__(self, monkeypatch) -> None:
        self.tractor, self.chip = mowing_tractor()
        self.nav = new_nav()
        self.nav.control_period_s = 0.05
        self.mission, self.service = wire_mission(monkeypatch, self.tractor, self.nav, [FAR])
        self.guard = StubGuard()
        self.nav.attach_safety(obstacle_guard=self.guard)
        self.tractor.motion_limiter = self.guard
        self.tractor.attach_blade_permit(BLADE_PERMIT_NAME, self.nav.blade_permit)
        self.relay_on_calls = 0
        real_set = self.tractor.blade_pto.set

        def counting_set(on: bool) -> None:
            self.relay_on_calls += bool(on)
            real_set(on)

        monkeypatch.setattr(self.tractor.blade_pto, "set", counting_set)
        self.task: asyncio.Task | None = None

    def start_mission(self) -> None:
        self.task = asyncio.create_task(self.nav.execute_mission(self.mission))

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.tractor.shutdown()


async def test_an_operator_on_cannot_re_engage_the_pto_against_an_autonomous_hold(monkeypatch):
    rig = Rig(monkeypatch)
    rig.start_mission()
    try:
        await wait_until(lambda: rig.tractor.state.blade_engaged, timeout=5.0)
        rig.guard.hold = True
        await wait_until(lambda: not rig.tractor.state.blade_engaged, timeout=5.0)
        rig.relay_on_calls = 0

        refused = await rig.tractor.engage_blade(True, source="operator")  # used to be accepted

        assert refused["status"] == "rejected", refused
        assert "navigation safety gate holds" in refused["reason"], refused
        assert "test obstacle" in refused["reason"], "and it says why"
        assert rig.tractor.state.blade_engaged is False
        for _ in range(8):  # an operator that keeps repeating ON, against a loop at 20 Hz
            await rig.tractor.engage_blade(True, source="operator")
            await asyncio.sleep(0.03)
        assert rig.relay_on_calls == 0, "the PTO relay was switched on against the hold"

        rig.guard.hold = False  # a hold is not a latch: the path is clear again
        again = await rig.tractor.engage_blade(True, source="operator")
        assert again["status"] == "ok" and rig.tractor.state.blade_engaged is True
    finally:
        await rig.close()


async def test_the_pto_is_the_operators_call_when_no_run_owns_the_machine(monkeypatch):
    """The policy that stays: manual driving is not gated, so neither is its blade."""
    rig = Rig(monkeypatch)  # no mission started, no autonomous mode
    try:
        rig.guard.hold = True
        assert rig.nav.blade_permit() is None

        result = await rig.tractor.engage_blade(True, source="operator")

        assert result["status"] == "ok" and rig.tractor.state.blade_engaged is True
    finally:
        await rig.close()


async def test_a_refused_on_also_drops_a_pto_that_is_already_turning(monkeypatch):
    """Refuse AND drop, as for an obstacle hold: the repeated ON switches a running PTO off."""
    rig = Rig(monkeypatch)
    try:
        assert (await rig.tractor.engage_blade(True, source="operator"))["status"] == "ok"
        rig.nav._mission_loop_active = True  # a run now owns the machine ...
        rig.guard.hold = True  # ... and the gate holds

        refused = await rig.tractor.engage_blade(True, source="operator")

        assert refused["status"] == "rejected"
        assert rig.tractor.state.blade_engaged is False and rig.tractor.blade_pto.state is False
    finally:
        rig.nav._mission_loop_active = False
        await rig.close()


async def test_a_permit_that_cannot_be_evaluated_forbids_the_pto():
    tractor, _chip = mowing_tractor()

    def broken() -> str | None:
        raise RuntimeError("permit exploded")

    tractor.attach_blade_permit("broken", broken)

    result = await tractor.engage_blade(True, source="operator")

    assert result["status"] == "rejected" and "permit broken failed" in result["reason"]
    assert tractor.state.blade_engaged is False
    await tractor.shutdown()


async def test_a_permit_attached_twice_under_one_name_is_one_permit_and_can_be_detached():
    tractor, _chip = mowing_tractor()
    calls: list[str] = []
    tractor.attach_blade_permit("gate", lambda: calls.append("old") or "old reason")
    tractor.attach_blade_permit("gate", lambda: calls.append("new") or None)

    result = await tractor.engage_blade(True, source="operator")
    assert result["status"] == "ok" and calls == ["new"], "the second attach replaced the first"

    tractor.attach_blade_permit("gate", lambda: "never again")
    tractor.detach_blade_permit("gate")
    assert (await tractor.engage_blade(True, source="operator"))["status"] == "ok"
    await tractor.shutdown()


async def test_the_lifespan_attaches_the_permit_to_the_tractor_and_detaches_it_on_shutdown(
    monkeypatch,
):
    tractor, _chip = mowing_tractor()

    async with real_lifespan(monkeypatch, tractor, live_sensors(2000.0)):
        names = [name for name, _permit in tractor._blade_permits]
        assert names.count(BLADE_PERMIT_NAME) == 1

    assert tractor._blade_permits == []
