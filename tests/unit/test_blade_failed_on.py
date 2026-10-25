"""HW-01 (PR #20 review 3): an ON that fails is followed by an OFF the service itself confirms.

An ON whose readback fails AND whose rollback also fails used to be neither pending nor supervised:
``status`` said "commanded off, observed error, no pending OFF", no supervisor ran and the pad
could stay high with nobody retrying. Everything runs against the FAKE ``lgpio`` of
``fake_lgpio.py``.
"""

from __future__ import annotations

import pytest
from _blade_fakes import OFF, ON, armed, blade_world, eventually
from fake_lgpio import FakeLgpio

from backend.src.core.globals import _safety_state
from backend.src.services.blade_service import BladeService


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    blade_world(monkeypatch)
    yield
    _safety_state["emergency_stop_active"] = False


async def _pending_cleared(svc: BladeService) -> bool:
    return (await svc.status())["off_pending"] is None


class StuckAfterOn(FakeLgpio):
    """The ON write lands, then the pad sticks high and GPIO25's readback dies: neither the ON's
    confirmation nor the adapter's own rollback can be confirmed ("the output state is UNKNOWN")."""

    def __init__(self) -> None:
        super().__init__()
        self.faulty = True
        self._stuck = False

    def gpio_write(self, handle, pin, level):
        rc = super().gpio_write(handle, pin, level)
        if self.faulty and pin == 24 and level == 1:
            self._stuck = True
            self.hold_pad(24, 1)
        return rc

    def gpio_read(self, handle, pin):
        if self._stuck and pin == 25:
            self.calls.append(("gpio_read", handle, pin))
            raise self.error("readback glitch")
        return super().gpio_read(handle, pin)

    def heal(self) -> None:
        self.faulty = False
        self._stuck = False
        self.release_pad(24)


class OnReadbackGlitch(FakeLgpio):
    """GPIO25's readback raises while IN1 is high: the ON's confirmation fails and so does the
    readback of the adapter's rollback."""

    def gpio_read(self, handle, pin):
        if pin == 25 and self.pad.get(24) == 1:
            self.calls.append(("gpio_read", handle, pin))
            raise self.error("readback glitch")
        return super().gpio_read(handle, pin)


async def test_an_on_whose_readback_and_rollback_both_fail_is_a_pending_off_not_a_forgotten_one(
    monkeypatch,
):
    fake = StuckAfterOn()
    svc, _ = armed(monkeypatch, fake=fake)
    assert await svc.initialize() is True

    assert await svc.set_active(True) is False

    assert "UNKNOWN" in svc.last_refusal, "the refusal says what the adapter could not confirm"
    status = await svc.status()
    assert status["off_pending"] == "ON not confirmed", "remembered, not forgotten"
    assert status["supervising"] is True, "and somebody is retrying it"
    assert status["active"] is True and status["state_confirmed"] is False
    assert "unconfirmed" in status["refusal"], "no new ON until that OFF is confirmed"
    assert await svc.set_active(True) is False
    assert "still unconfirmed" in svc.last_refusal

    seen = len(fake.writes())
    await eventually(lambda: len(fake.writes()) >= seen + 4)  # both pins, every tick

    fake.heal()  # the pad follows again
    await eventually(lambda: _pending_cleared(svc))
    status = await svc.status()
    assert (status["observed"], status["state_confirmed"], status["active"]) == ("off", True, False)
    assert await svc.set_active(True) is True and fake.pad == ON


async def test_a_failed_on_is_followed_by_an_off_the_service_confirms_itself(monkeypatch):
    fake = OnReadbackGlitch()
    svc, _ = armed(monkeypatch, fake=fake)
    assert await svc.initialize() is True
    fake.fail_write(24, "ignore", nth=2)  # the adapter's rollback write to IN1 is swallowed

    assert await svc.set_active(True) is False

    assert fake.pad == OFF, "the blade input is not left driven: the service wrote the OFF itself"
    status = await svc.status()
    assert status["observed"] == "off" and status["state_confirmed"] is True
    assert status["off_pending"] is None and "UNKNOWN" in svc.last_refusal


async def test_a_rolled_back_on_leaves_a_confirmed_off_state(monkeypatch):
    svc, fake = armed(monkeypatch)
    assert await svc.initialize() is True
    fake.hold_pad(24, 0)  # IN1 cannot rise: the write returns 0, the pad stays low

    assert await svc.set_active(True) is False

    assert "rolled back" in svc.last_refusal
    status = await svc.status()
    assert (status["observed"], status["active"], status["state_confirmed"]) == ("off", False, True)
    assert status["off_pending"] is None and status["commanded"] == "off"
