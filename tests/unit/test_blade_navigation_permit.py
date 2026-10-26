"""The navigation gate is a STANDING reason for the IBT-4 blade while autonomy runs (audit round 3).

Round 3 left one decision open: the hold handlers (obstacle, GNSS) switch the blade off, but a
client that keeps repeating ``ON`` was accepted the next instant, because a hold was an event for
the blade service, not a reason it re-read. ``NavigationService.blade_permit`` is that reason,
attached to the ``BladeService`` by the lifespan and read before every ON and on every
supervision tick.

Policy kept on purpose: with no autonomous run and no scripted drive the blade is the operator's own
call (the motion limiter exempts manual driving the same way), so a manual ON is still accepted
while the guard holds.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from _mission_rig import new_nav
from _nav_rig import OFF, ON, StubGuard, armed_blade, live_sensors, real_lifespan, wait_until
from _tractor_fakes import mowing_tractor

from backend.src.models import NavigationMode
from backend.src.services import blade_service as blade_module
from backend.src.services.navigation_service import BLADE_PERMIT_NAME


class StubGnss:
    """A GNSS monitor whose verdict the test flips."""

    def __init__(self) -> None:
        self.healthy = True

    def assess(self):
        reasons = () if self.healthy else ("fix is 30.0s old (limit 2s)",)
        return SimpleNamespace(healthy=self.healthy, reasons=reasons)


class Rig:
    """A real ``NavigationService`` and an armed ``BladeService`` joined by the permit."""

    def __init__(self, monkeypatch) -> None:
        monkeypatch.setenv("SIM_MODE", "0")  # the real lgpio branch, on the fake pads
        # A 10 s deadman: an OFF in a test can only come from the permit, never from a timeout.
        self.blade, self.pads = armed_blade(monkeypatch, deadman_ms=10000)
        self.nav = new_nav()
        self.guard = StubGuard()
        self.gnss = StubGnss()
        self.nav.attach_safety(obstacle_guard=self.guard, gps_monitor=self.gnss)
        self.blade.attach_permit(BLADE_PERMIT_NAME, self.nav.blade_permit)
        self.ai_task: asyncio.Task | None = None

    async def open(self) -> None:
        assert await self.blade.initialize() is True

    def autonomy(self, mode: NavigationMode = NavigationMode.AUTO) -> None:
        """A run owns the machine: its mode AND its loop (the permit keys on the loop, NAV-03)."""
        self.nav.navigation_state.navigation_mode = mode
        if mode is NavigationMode.AI:
            self.ai_task = asyncio.get_running_loop().create_task(asyncio.sleep(3600))
            self.nav._ai_task = self.ai_task
        else:
            self.nav._mission_loop_active = True

    async def hammer_on(self, seconds: float = 0.4) -> list[bool]:
        """A client that repeats ON every 50 ms for ``seconds``."""
        results: list[bool] = []
        for _ in range(int(seconds / 0.05)):
            results.append(await self.blade.set_active(True))
            await asyncio.sleep(0.05)
        return results

    async def close(self) -> None:
        self.nav._mission_loop_active = False
        if self.ai_task is not None:
            self.ai_task.cancel()
            await asyncio.gather(self.ai_task, return_exceptions=True)
        await self.blade.park()


async def _rig(monkeypatch) -> Rig:
    rig = Rig(monkeypatch)
    await rig.open()
    return rig


# ----------------------------------------------------------------------------- the permit


async def test_a_repeated_on_cannot_undo_an_obstacle_hold_while_autonomy_runs(monkeypatch):
    rig = await _rig(monkeypatch)
    try:
        assert await rig.blade.set_active(True) is True and rig.pads.pad == ON
        rig.autonomy()
        rig.guard.hold = True
        await rig.nav.enforce_motion_hold()  # what the 20 Hz monitor does
        await wait_until(lambda: rig.pads.pad == OFF)

        answers = await rig.hammer_on()

        assert answers and not any(answers), "every repeated ON was refused"
        assert rig.pads.pad == OFF
        assert "navigation safety gate holds" in (rig.blade.last_refusal or "")
        assert "test obstacle" in rig.blade.last_refusal, "and it says why"

        rig.guard.hold = False  # the path is clear: a hold is not a latch
        assert await rig.blade.set_active(True) is True and rig.pads.pad == ON
    finally:
        await rig.close()


async def test_a_repeated_on_cannot_undo_a_gnss_hold_while_autonomy_runs(monkeypatch):
    rig = await _rig(monkeypatch)
    try:
        rig.autonomy(NavigationMode.AI)
        rig.gnss.healthy = False

        answers = await rig.hammer_on(0.2)

        assert answers and not any(answers) and rig.pads.pad == OFF
        assert "gnss: fix is 30.0s old" in rig.blade.last_refusal

        rig.gnss.healthy = True
        assert await rig.blade.set_active(True) is True
    finally:
        await rig.close()


async def test_the_supervisor_drops_a_blade_that_is_on_when_the_gate_starts_holding(monkeypatch):
    """No hold handler runs here: the permit is read on every tick, as STATE."""
    rig = await _rig(monkeypatch)
    try:
        assert await rig.blade.set_active(True) is True
        rig.autonomy()
        rig.guard.hold = (
            True  # nobody calls enforce_motion_hold(), nobody calls request_blade_off()
        )

        await wait_until(lambda: rig.pads.pad == OFF, timeout=3.0)

        assert "navigation_gate" in (rig.blade.last_drop or "")
    finally:
        await rig.close()


async def test_a_manual_on_is_still_accepted_while_the_guard_holds_and_nothing_is_autonomous(
    monkeypatch,
):
    """The policy that stays: the blade is the operator's call when no run is moving the machine."""
    rig = await _rig(monkeypatch)
    try:
        rig.guard.hold = True
        rig.gnss.healthy = False
        assert rig.nav.blade_permit() is None

        assert await rig.blade.set_active(True) is True and rig.pads.pad == ON
        await asyncio.sleep(0.3)  # the supervisor ticks: still on
        assert rig.pads.pad == ON
    finally:
        await rig.close()


async def test_a_clear_gate_never_refuses_the_blade_while_autonomy_runs(monkeypatch):
    rig = await _rig(monkeypatch)
    try:
        rig.autonomy()
        assert rig.nav.blade_permit() is None

        assert all(await rig.hammer_on(0.2)) and rig.pads.pad == ON
    finally:
        await rig.close()


async def test_a_scripted_drive_is_judged_by_the_inputs_it_is_gated_on(monkeypatch):
    """The IMU calibration drive leaves GNSS and heading age out of its gate; the guard counts."""
    rig = await _rig(monkeypatch)
    try:
        rig.nav._scripted_drive_active = True
        rig.gnss.healthy = False
        assert rig.nav.blade_permit() is None, "a GNSS fault does not gate an open-loop drive"

        rig.guard.hold = True
        assert "navigation safety gate holds" in (rig.nav.blade_permit() or "")
        assert await rig.blade.set_active(True) is False
    finally:
        await rig.close()


async def test_a_gate_that_cannot_be_evaluated_forbids_the_blade(monkeypatch):
    rig = await _rig(monkeypatch)
    try:
        rig.autonomy()

        def broken():
            raise RuntimeError("guard exploded")

        rig.guard.assess = broken  # the nav gate turns a failing check into a hold
        assert await rig.blade.set_active(True) is False
        assert (
            "guard failed" in rig.blade.last_refusal or "guard exploded" in rig.blade.last_refusal
        )
    finally:
        await rig.close()


# ------------------------------------------------------------------------------ the wiring


async def test_the_lifespan_attaches_the_permit_and_detaches_it_on_shutdown(monkeypatch):
    blade, _pads = armed_blade(monkeypatch, deadman_ms=10000)
    tractor, _chip = mowing_tractor()
    sensors = live_sensors(2000.0)

    async with real_lifespan(monkeypatch, tractor, sensors):
        names = [name for name, _permit in blade._permits]
        assert names.count(BLADE_PERMIT_NAME) == 1

    assert BLADE_PERMIT_NAME not in [name for name, _permit in blade._permits]


async def test_a_blade_service_rebuilt_from_a_repaired_config_keeps_its_permits(
    monkeypatch, tmp_path
):
    import sys

    from _nav_rig import FakeLgpio

    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(blade_module, "_blade_service", None)
    monkeypatch.setattr(blade_module, "_config_retry_at", 0.0)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: None)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)
    monkeypatch.setitem(sys.modules, "lgpio", FakeLgpio())
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    (tmp_path / "hardware.yaml").write_text(
        "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 99}\n", encoding="utf-8"
    )
    stub = blade_module.get_blade_service()
    assert stub.config_error
    nav = new_nav()
    stub.attach_permit(BLADE_PERMIT_NAME, nav.blade_permit)

    (tmp_path / "hardware.yaml").write_text(
        "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 25}\n", encoding="utf-8"
    )
    monkeypatch.setattr(blade_module, "_config_retry_at", 0.0)
    fresh = blade_module.get_blade_service()

    assert fresh is not stub and fresh.config_error is None
    assert [name for name, _p in fresh._permits] == [BLADE_PERMIT_NAME]
