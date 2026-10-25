"""IBT-4 driver and lgpio adapter, review round 4: the surviving mutants G03, G04, G08 and L08.

PR #20 review 3 (hardware), HW-07. Everything runs against the FAKE ``lgpio`` of ``fake_lgpio.py``
(the adapter has never run on hardware).
"""

from __future__ import annotations

import asyncio

import pytest
from _blade_fakes import OFF, armed, blade_world, on
from fake_lgpio import FakeLgpio

from backend.src.core.globals import _safety_state
from backend.src.drivers.blade.ibt4_gpio import IBT4BladeDriver
from backend.src.drivers.blade.ibt4_lgpio import IBT4GpioError, LgpioBladeAdapter


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    blade_world(monkeypatch)
    yield
    _safety_state["emergency_stop_active"] = False


def _real_driver(fake: FakeLgpio) -> IBT4BladeDriver:
    return IBT4BladeDriver(
        {"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}},
        gpio_adapter=LgpioBladeAdapter(24, 25, (1, 0), lgpio_module=fake),
    )


# ------------------------------------------------------------------ G08: no clear over no lines


async def test_the_driver_latch_cannot_be_cleared_while_the_lines_are_not_claimed():
    """G08: ``set_estop(False)`` over lines that were never claimed released the latch: the
    clear route then answered EMERGENCY_CLEARED without anything having been confirmed."""
    fake = FakeLgpio()
    fake.fail_chip_open("no chip")
    driver = _real_driver(fake)
    await driver.initialize()
    assert driver.initialized is False and driver.drives_hardware is True
    assert await driver.set_estop(True) is False  # the latch is set, the OFF cannot be confirmed

    assert await driver.set_estop(False) is False

    assert driver.estop_latched is True
    assert "lines are not claimed" in driver.last_error


async def test_a_flag_only_stub_has_nothing_to_confirm_so_its_latch_clears():
    """The refusal above is for real GPIO only: the unarmed stub's latch is a plain flag."""
    stub = IBT4BladeDriver({"pins": {"in1": 24, "in2": 25}})
    assert stub.drives_hardware is False and stub.initialized is False
    await stub.set_estop(True)
    assert stub.estop_latched is True

    assert await stub.set_estop(False) is True

    assert stub.estop_latched is False


async def test_clearing_the_estop_with_unclaimable_lines_keeps_the_latch_and_says_why(
    monkeypatch,
):
    """G08 through the service (the E-stop clear route calls ``clear_emergency``)."""
    fake = FakeLgpio()
    fake.fail_chip_open("no chip")
    svc, _ = armed(monkeypatch, fake=fake)
    report = await svc.emergency_stop()
    assert report["latched"] is True and report["off"] is False

    await svc.clear_emergency()

    assert svc.emergency_latched is True
    assert "lines are not claimed" in svc.clear_refusal
    assert await svc.set_active(True) is False


# ----------------------------------------------------------- G03: on_levels with IN2 high


async def test_a_blade_with_in2_as_its_on_level_reads_back_as_on(monkeypatch):
    """G03: the pad classification compared with (1, 0), not with the configured ON levels: a
    configured (0, 1) blade read "mixed" and was dropped by the supervisor at once."""
    svc, fake = armed(monkeypatch, on_levels={"in1": 0, "in2": 1}, deadman_ms=10_000)
    await on(svc)
    assert fake.pad == {24: 0, 25: 1}

    await asyncio.sleep(0.3)  # several supervision ticks

    assert fake.pad == {24: 0, 25: 1}, "the supervisor did not drop a blade that is simply on"
    status = await svc.status()
    assert status["observed"] == "on" and status["pad"] == {"in1": 0, "in2": 1}
    assert status["commanded"] == "on" and status["state_confirmed"] is True
    assert await svc.set_active(True) is True, "the deadman refresh holds too"
    assert svc.last_drop is None


# ------------------------------------------------------------------ G04: stop() of real GPIO


async def test_stopping_a_real_adapter_leaves_the_state_unconfirmed_but_not_the_stub(monkeypatch):
    """G04: after the lines are released the pad state is UNKNOWN (it may float): a real adapter
    must not go on reporting a confirmed state; the flag-only stub has no pad to be unsure of."""
    fake = FakeLgpio()
    driver = _real_driver(fake)
    await driver.initialize()
    await driver.start()
    assert await driver.set_active(True) is True and driver.state_confirmed is True

    await driver.stop()

    assert driver.initialized is False and driver.running is False
    assert driver.state_confirmed is False
    assert fake.pad == OFF and fake.freed, "driven low and released first"

    stub = IBT4BladeDriver({})
    await stub.initialize()
    await stub.stop()
    assert stub.state_confirmed is True


# ----------------------------------------------------------- L08: a readback that is not a level


@pytest.mark.parametrize("bad", [2, -1, None, 255])
async def test_a_readback_that_is_not_a_level_is_an_error_not_a_pad_state(bad):
    """L08: ``gpio_read`` answering anything but 0 or 1 (a negative error code with lgpio
    exceptions off, garbage) must not be classified as a pad level."""
    fake = FakeLgpio()
    adapter = LgpioBladeAdapter(24, 25, (1, 0), lgpio_module=fake)
    adapter.setup()
    fake.gpio_read = lambda handle, pin: bad

    with pytest.raises(IBT4GpioError, match="expected 0 or 1"):
        adapter.read_pins()

    driver = IBT4BladeDriver({"gpio_enabled": True}, gpio_adapter=adapter)
    await driver.initialize()
    observation = driver.observe()
    assert observation.state == "error" and "expected 0 or 1" in observation.error
    health = await driver.health_check()
    assert health["observed"] == "error" and health["active"] is True
    assert health["state_confirmed"] is False
