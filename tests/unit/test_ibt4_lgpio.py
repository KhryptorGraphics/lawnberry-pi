"""IBT-4 lgpio adapter, the arming gate and the service wiring, against a FAKE ``lgpio``.

Nothing here touches hardware and lgpio is never imported for real. The adapter has never run on
a Pi: these tests prove the logic (all-or-nothing setup, write + readback confirmation, rollback,
fail-closed behaviour), not the electrical behaviour (see the UNVERIFIED list in
``backend/src/drivers/blade/ibt4_lgpio.py``).
"""

from __future__ import annotations

import builtins
import contextlib
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fake_lgpio import FakeLgpio

from backend.src.drivers.blade.ibt4_gpio import IBT4BladeDriver, _GPIOAdapter
from backend.src.drivers.blade.ibt4_lgpio import IBT4GpioError, LgpioBladeAdapter
from backend.src.services import blade_service

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _default_gpiochip_probe(monkeypatch):
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)


@pytest.fixture
def non_sim(monkeypatch):
    """The drivers only call ``setup()`` (and only arm the real adapter) outside SIM_MODE."""
    monkeypatch.setenv("SIM_MODE", "0")


def _adapter(fake: FakeLgpio, on_levels=(1, 0)) -> LgpioBladeAdapter:
    return LgpioBladeAdapter(24, 25, on_levels, lgpio_module=fake)


def _armed(fake: FakeLgpio, on_levels=(1, 0)) -> LgpioBladeAdapter:
    adapter = _adapter(fake, on_levels)
    adapter.setup()
    return adapter


def _driver(fake: FakeLgpio, on_levels=(1, 0)) -> IBT4BladeDriver:
    config = {
        "gpio_enabled": True,
        "pins": {"in1": 24, "in2": 25},
        "on_levels": {"in1": on_levels[0], "in2": on_levels[1]},
    }
    return IBT4BladeDriver(config, gpio_adapter=_adapter(fake, on_levels))


# ------------------------------------- nothing touches hardware at import or construction


def test_importing_and_constructing_never_imports_or_calls_lgpio():
    """A fresh interpreter with lgpio blocked and every ``import lgpio`` recorded: importing both
    driver modules and the service, and building the adapter, the armed driver and the service,
    must succeed without a single attempt. Catches a module-level or ``__init__``-time import."""
    script = textwrap.dedent(
        """
        import builtins, sys
        sys.modules["lgpio"] = None
        attempts = []
        real_import = builtins.__import__
        def spy(name, *args, **kwargs):
            if name == "lgpio" or name.startswith("lgpio."):
                attempts.append(name)
            return real_import(name, *args, **kwargs)
        builtins.__import__ = spy

        import backend.src.drivers.blade.ibt4_gpio as gpio
        import backend.src.drivers.blade.ibt4_lgpio as lg
        from backend.src.services.blade_service import BladeService

        adapter = lg.LgpioBladeAdapter(24, 25)
        driver = gpio.IBT4BladeDriver({"gpio_enabled": True})
        assert type(driver._gpio) is lg.LgpioBladeAdapter, "armed + non-SIM must pick lgpio"
        service = BladeService({"gpio_enabled": True})
        assert not adapter.hardware_backed and not driver.hardware_backed
        print("OK", len(attempts), attempts)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "SIM_MODE": "0"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "OK 0 []"


def test_constructing_the_adapter_calls_no_lgpio_function():
    fake = FakeLgpio()
    adapter = _adapter(fake)
    assert fake.calls == []
    assert adapter.hardware_backed is False and adapter.state_confirmed is False
    assert adapter.is_active() is False and adapter.last_error is None


# ------------------------------------------------------------------------------- setup


def test_setup_opens_the_chip_claims_in1_then_in2_low_and_reads_both_back():
    fake = FakeLgpio()
    adapter = _adapter(fake)

    adapter.setup()

    assert fake.calls == [
        ("gpiochip_open", 4),  # Pi 5 RP1 bank first (drivers.base.open_gpiochip)
        ("gpio_claim_output", 7, 24, 0),  # requested already low: no high glitch
        ("gpio_claim_output", 7, 25, 0),
        ("gpio_read", 7, 24),
        ("gpio_read", 7, 25),
    ]
    assert adapter.hardware_backed is True and adapter.state_confirmed is True
    assert adapter.is_active() is False and adapter.last_error is None
    assert fake.writes() == [], "setup must not write beyond the initial claim level"


def test_setup_uses_the_injected_chip_opener():
    fake = FakeLgpio()
    adapter = LgpioBladeAdapter(24, 25, lgpio_module=fake, open_chip=lambda lg: lg.gpiochip_open(0))
    adapter.setup()
    assert fake.calls[0] == ("gpiochip_open", 0)


def test_setup_is_a_noop_while_already_armed():
    fake = FakeLgpio()
    adapter = _armed(fake)
    before = list(fake.calls)
    adapter.setup()
    assert fake.calls == before, "a second setup() must not re-claim lines it already holds"


def test_setup_imports_lgpio_lazily_when_none_is_injected(monkeypatch):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    adapter = LgpioBladeAdapter(24, 25)
    assert fake.calls == []
    adapter.setup()
    assert adapter.hardware_backed is True
    assert fake.names()[:3] == ["gpiochip_open", "gpio_claim_output", "gpio_claim_output"]


def test_setup_gpio25_busy_releases_gpio24_closes_the_chip_and_names_the_pin():
    """GPIO25 is the HaLow IRQ line on the deployed Pi (audit C-03): the claim must fail the
    WHOLE setup, leave nothing claimed and never drive anything."""
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    adapter = _adapter(fake)

    with pytest.raises(IBT4GpioError, match=r"GPIO25 .*GPIO busy"):
        adapter.setup()

    assert fake.freed == [24], "the line claimed before the failure must be released"
    assert fake.claimed == set() and fake.closed == [7]
    assert fake.writes() == [], "a failed setup never drives a line"
    assert adapter.hardware_backed is False and adapter.state_confirmed is False
    assert "GPIO25" in adapter.last_error and "GPIO busy" in adapter.last_error
    with pytest.raises(IBT4GpioError, match="not set up"):
        adapter.set_active(False)
    assert fake.writes() == []


def test_setup_gpio24_claim_failure_claims_nothing_and_never_tries_gpio25():
    fake = FakeLgpio()
    fake.fail_claim(24, "GPIO busy")
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match="GPIO24"):
        adapter.setup()
    assert ("gpio_claim_output", 7, 25, 0) not in fake.calls
    assert fake.freed == [] and fake.closed == [7]


def test_setup_treats_a_negative_return_code_as_failure_when_lgpio_exceptions_are_off():
    fake = FakeLgpio(exceptions=False)
    fake.fail_claim(25)
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match=r"GPIO25 .*returned -1"):
        adapter.setup()
    assert fake.freed == [24] and fake.closed == [7] and adapter.hardware_backed is False


def test_setup_rejects_a_chip_handle_that_is_an_error_code():
    fake = FakeLgpio(exceptions=False)
    fake.fail_chip_open()
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match="chip could not be opened"):
        adapter.setup()
    assert "gpio_claim_output" not in fake.names(), "nothing is claimed through a bad handle"


def test_setup_fails_when_the_chip_cannot_be_opened():
    fake = FakeLgpio()
    fake.fail_chip_open("can not open gpiochip")
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match="can not open gpiochip"):
        adapter.setup()
    assert fake.names() == ["gpiochip_open", "gpiochip_open"], "chip 4 then chip 0, nothing else"
    assert adapter.hardware_backed is False


@pytest.mark.parametrize("pin", [24, 25])
def test_setup_fails_when_a_pin_does_not_read_back_low(pin):
    """A pin held high by something else must not be accepted as an output we control."""
    fake = FakeLgpio()
    fake.hold_pad(pin, 1)
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match=rf"GPIO{pin} .*reads 1"):
        adapter.setup()
    assert fake.claimed == set() and sorted(fake.freed) == [24, 25] and fake.closed == [7]
    assert adapter.hardware_backed is False


def test_setup_fails_when_the_readback_raises():
    fake = FakeLgpio()
    fake.fail_read(25, "read failed")
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match=r"GPIO25 .*read failed"):
        adapter.setup()
    assert fake.claimed == set() and fake.closed == [7] and adapter.hardware_backed is False


def test_setup_without_the_lgpio_module_is_an_ibt4_gpio_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "lgpio", None)  # import lgpio -> ImportError
    adapter = LgpioBladeAdapter(24, 25)
    with pytest.raises(IBT4GpioError, match="lgpio"):
        adapter.setup()
    assert adapter.hardware_backed is False and "lgpio" in adapter.last_error


def test_setup_survives_a_failing_release_and_still_reports_the_original_error():
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    fake.fail_free(24)
    adapter = _adapter(fake)
    with pytest.raises(IBT4GpioError, match="GPIO busy"):
        adapter.setup()
    assert fake.closed == [7], "the chip is closed even when a free fails"


@pytest.mark.parametrize(
    ("in1", "in2", "levels", "needle"),
    [
        (24, 24, (1, 0), "invalid IBT-4 pins"),
        (-1, 25, (1, 0), "invalid IBT-4 pins"),
        (24, 25, (0, 0), "identical to OFF"),
        (24, 25, (2, 0), "need two of 0/1"),
        (24, 25, (1,), "need two of 0/1"),
        (24, 25, None, "need two of 0/1"),
        (24, 25, "10", "need two of 0/1"),
    ],
)
def test_setup_rejects_an_invalid_configuration_before_touching_lgpio(in1, in2, levels, needle):
    fake = FakeLgpio()
    adapter = LgpioBladeAdapter(in1, in2, levels, lgpio_module=fake)
    with pytest.raises(IBT4GpioError, match=needle):
        adapter.setup()
    assert fake.calls == [] and adapter.hardware_backed is False


# ------------------------------------------------------------------------------- drive


@pytest.mark.parametrize("levels", [(1, 0), (0, 1), (1, 1)])
def test_on_writes_the_configured_levels_in_order_and_confirms_by_readback(levels):
    fake = FakeLgpio()
    adapter = _armed(fake, levels)
    setup_calls = len(fake.calls)

    adapter.set_active(True)

    assert fake.calls[setup_calls:] == [
        ("gpio_write", 7, 24, levels[0]),
        ("gpio_write", 7, 25, levels[1]),
        ("gpio_read", 7, 24),
        ("gpio_read", 7, 25),
    ]
    assert fake.pad == {24: levels[0], 25: levels[1]}
    assert adapter.is_active() is True and adapter.state_confirmed is True
    assert adapter.last_error is None


@pytest.mark.parametrize("levels", [(1, 0), (0, 1), (1, 1)])
def test_off_drives_both_inputs_low_whatever_the_on_levels(levels):
    fake = FakeLgpio()
    adapter = _armed(fake, levels)
    adapter.set_active(True)

    adapter.set_active(False)

    assert fake.pad == {24: 0, 25: 0}
    assert fake.writes()[-2:] == [(24, 0), (25, 0)]
    assert adapter.is_active() is False and adapter.state_confirmed is True


def test_off_with_the_first_write_raising_still_writes_the_second_and_reports_failure():
    """Maximise safing: a failed IN1 write must not stop IN2 from being driven low."""
    fake = FakeLgpio()
    adapter = _armed(fake, (0, 1))  # ON = IN1 low, IN2 high
    adapter.set_active(True)
    fake.fail_write(24, "raise")

    with pytest.raises(IBT4GpioError, match=r"OFF not confirmed.*GPIO24 write\(0\) raised"):
        adapter.set_active(False)

    assert fake.writes()[-2:] == [(24, 0), (25, 0)], "the second write was still attempted"
    assert fake.pad[25] == 0, "and it took effect"
    assert adapter.state_confirmed is False
    assert adapter.is_active() is True, "last CONFIRMED state was ON; it was not confirmed off"
    assert "OFF not confirmed" in adapter.last_error


def test_off_with_a_negative_return_code_is_a_failure_even_if_the_readback_looks_right():
    fake = FakeLgpio()
    adapter = _armed(fake)
    adapter.set_active(True)
    fake.fail_write(25, "rc")  # lgpio error code; the pad happens to already be 0

    with pytest.raises(IBT4GpioError, match=r"GPIO25 write\(0\) returned error code -1"):
        adapter.set_active(False)

    assert fake.pad == {24: 0, 25: 0}
    assert adapter.state_confirmed is False, "readback alone must not hide a failed write"


def test_a_pin_stuck_high_makes_off_unconfirmed():
    fake = FakeLgpio()
    adapter = _armed(fake)
    adapter.set_active(True)
    fake.hold_pad(24, 1)  # shorted/held: the write returns 0 but the pad stays high

    with pytest.raises(IBT4GpioError, match=r"GPIO24 reads back 1, expected 0"):
        adapter.set_active(False)

    assert adapter.is_active() is True
    assert adapter.state_confirmed is False


def test_a_write_silently_ignored_is_caught_by_the_readback():
    fake = FakeLgpio()
    adapter = _armed(fake, (0, 1))
    adapter.set_active(True)  # pad = (0, 1)
    fake.fail_write(25, "ignore")

    with pytest.raises(IBT4GpioError, match=r"GPIO25 reads back 1, expected 0"):
        adapter.set_active(False)

    assert adapter.state_confirmed is False and adapter.is_active() is True


def test_a_failing_readback_makes_the_change_unconfirmed():
    fake = FakeLgpio()
    adapter = _armed(fake)
    fake.fail_read(24, "read failed")

    with pytest.raises(IBT4GpioError, match=r"GPIO24 readback raised .*read failed"):
        adapter.set_active(True)

    assert adapter.is_active() is False, "an unconfirmed ON is never reported active"
    assert fake.pad == {24: 0, 25: 0}, "and it is rolled back"


def test_on_partial_failure_rolls_back_both_inputs_to_off():
    fake = FakeLgpio()
    adapter = _armed(fake)
    setup_writes = len(fake.writes())
    fake.fail_write(25, "raise", nth=1)  # the ON write of IN2 fails once

    with pytest.raises(IBT4GpioError, match=r"ON not confirmed.*rolled back"):
        adapter.set_active(True)

    assert fake.writes()[setup_writes:] == [(24, 1), (25, 0), (24, 0), (25, 0)]
    assert fake.pad == {24: 0, 25: 0}, "IN1 was driven high and must have been driven low again"
    assert adapter.is_active() is False
    assert adapter.state_confirmed is True, "the rollback was confirmed: the state is known (off)"


def test_on_partial_failure_with_a_failed_rollback_reports_an_unknown_state():
    fake = FakeLgpio()
    adapter = _armed(fake)
    fake.fail_write(25, "raise", nth=1)  # ON: IN2 write fails
    fake.fail_write(24, "raise", nth=2)  # rollback: IN1 write fails too

    with pytest.raises(IBT4GpioError, match=r"rollback to OFF also failed.*UNKNOWN"):
        adapter.set_active(True)

    assert fake.pad[24] == 1, "IN1 is really still high: the state is unknown, not off"
    assert adapter.state_confirmed is False
    assert adapter.is_active() is False, "never confirmed ON"


def test_on_with_a_pin_held_by_something_else_is_unconfirmed():
    fake = FakeLgpio()
    adapter = _armed(fake)
    fake.hold_pad(25, 1)  # ON wants IN2 low; something else holds it high

    with pytest.raises(IBT4GpioError, match=r"GPIO25 reads back 1, expected 0"):
        adapter.set_active(True)

    assert fake.writes()[-4:] == [(24, 1), (25, 0), (24, 0), (25, 0)], "rollback was attempted"
    assert adapter.state_confirmed is False, "IN2 still reads high: nothing is confirmed"
    assert adapter.is_active() is False


def test_a_success_clears_the_last_error():
    fake = FakeLgpio()
    adapter = _armed(fake)
    fake.fail_write(24, "raise", nth=1)
    with pytest.raises(IBT4GpioError):
        adapter.set_active(False)
    assert adapter.last_error

    adapter.set_active(False)

    assert adapter.last_error is None and adapter.state_confirmed is True


@pytest.mark.parametrize("state", ["never set up", "after cleanup"])
def test_set_active_without_armed_lines_raises_and_writes_nothing(state):
    fake = FakeLgpio()
    adapter = _adapter(fake)
    if state == "after cleanup":
        adapter.setup()
        adapter.cleanup()
    calls_before = list(fake.calls)

    with pytest.raises(IBT4GpioError, match="not set up"):
        adapter.set_active(False)

    assert fake.calls == calls_before


# ----------------------------------------------------------------------------- cleanup


def test_cleanup_drives_both_low_frees_both_lines_and_closes_the_chip():
    fake = FakeLgpio()
    adapter = _armed(fake)
    adapter.set_active(True)

    adapter.cleanup()

    assert fake.pad == {24: 0, 25: 0}
    assert sorted(fake.freed) == [24, 25] and fake.claimed == set() and fake.closed == [7]
    assert fake.names()[-5:] == [
        "gpio_write",  # both inputs are driven low first ...
        "gpio_write",
        "gpio_free",  # ... then released ...
        "gpio_free",
        "gpiochip_close",  # ... and the chip is closed last
    ]
    assert adapter.hardware_backed is False and adapter.is_active() is False
    assert adapter.state_confirmed is False, "released lines: the pad state is unknown (UNVERIFIED)"


def test_cleanup_never_raises_and_still_closes_the_chip_when_everything_fails():
    fake = FakeLgpio()
    adapter = _armed(fake)
    for pin in (24, 25):
        fake.fail_write(pin, "raise")
        fake.fail_free(pin)

    adapter.cleanup()  # must not raise

    assert fake.closed == [7], "later steps still run after earlier ones fail"
    assert adapter.hardware_backed is False
    assert adapter.last_error.startswith("cleanup:") and "GPIO24" in adapter.last_error


def test_cleanup_is_a_noop_before_setup_and_idempotent_after():
    fake = FakeLgpio()
    adapter = _adapter(fake)
    adapter.cleanup()
    assert fake.calls == []
    adapter.setup()
    adapter.cleanup()
    calls = list(fake.calls)
    adapter.cleanup()
    assert fake.calls == calls, "a second cleanup must not touch lgpio again"


def test_the_adapter_can_be_armed_again_after_cleanup():
    """cleanup() must drop the old handle and claims, or a re-arm would reuse a closed chip."""
    fake = FakeLgpio()
    adapter = _armed(fake)
    adapter.set_active(True)
    adapter.cleanup()

    adapter.setup()
    adapter.set_active(True)

    assert adapter.hardware_backed is True and adapter.is_active() is True
    assert fake.pad == {24: 1, 25: 0}
    assert fake.names().count("gpiochip_open") == 2 and fake.closed == [7]
    assert fake.claimed == {24, 25}


# ------------------------------------------------------------------- driver on the adapter


async def test_driver_initialize_arms_the_adapter(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)

    await drv.initialize()

    health = await drv.health_check()
    assert drv.initialized and drv.hardware_backed
    assert health["active"] is False and health["state_confirmed"] is True
    assert health["gpio_enabled"] is True and health["last_error"] is None


async def test_driver_fails_closed_when_gpio25_is_busy(non_sim):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    drv = _driver(fake)

    await drv.initialize()  # must not raise

    health = await drv.health_check()
    assert drv.initialized is False, "never pretend to be initialised"
    assert drv.hardware_backed is False
    assert "GPIO25" in health["last_error"] and "GPIO busy" in health["last_error"]
    assert await drv.set_active(True) is False
    assert await drv.set_active(False) is False
    assert fake.writes() == [], "a driver that could not claim its lines drives nothing"
    assert await drv.set_estop(True) is False, "no confirmed OFF is possible"
    assert drv.estop_latched is True, "but the latch is set regardless"


async def test_driver_on_off_round_trip_is_confirmed(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    await drv.start()

    assert await drv.set_active(True) is True
    assert fake.pad == {24: 1, 25: 0}
    assert (await drv.health_check())["active"] is True

    assert await drv.set_active(False) is True
    assert fake.pad == {24: 0, 25: 0}
    health = await drv.health_check()
    assert health["active"] is False and health["state_confirmed"] is True


async def test_driver_unconfirmed_off_keeps_active_true_until_an_off_is_confirmed(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    assert await drv.set_active(True) is True
    fake.hold_pad(24, 1)

    assert await drv.set_active(False) is False

    health = await drv.health_check()
    assert health["active"] is True, "an unconfirmed OFF must read as 'may be on'"
    assert health["state_confirmed"] is False
    assert "OFF not confirmed" in health["last_error"]

    fake.release_pad(24)
    assert await drv.set_active(False) is True, "a later confirmed OFF recovers the state"
    health = await drv.health_check()
    assert health["active"] is False and health["state_confirmed"] is True
    assert health["last_error"] is None


async def test_driver_unconfirmed_on_returns_false_and_leaves_the_output_rolled_back(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    fake.fail_write(25, "raise", nth=1)

    assert await drv.set_active(True) is False

    assert fake.pad == {24: 0, 25: 0}, "the half-applied ON was rolled back on the wire"
    health = await drv.health_check()
    assert health["state_confirmed"] is False
    assert health["active"] is True, "conservative: unconfirmed means 'may be on'"


async def test_estop_latch_blocks_a_later_on_without_any_gpio_write(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    await drv.start()

    assert await drv.set_estop(True) is True, "latching returns the confirmed OFF result"
    writes_after_latch = len(fake.writes())
    assert writes_after_latch == 2, "the latch itself drove both inputs low"

    assert await drv.set_active(True) is False
    assert len(fake.writes()) == writes_after_latch, "no GPIO write while latched"
    assert fake.pad == {24: 0, 25: 0}

    assert await drv.set_estop(False) is True
    assert await drv.set_active(True) is True
    assert fake.pad == {24: 1, 25: 0}


async def test_set_estop_reports_an_unconfirmed_off_but_still_latches(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    await drv.set_active(True)
    fake.hold_pad(24, 1)

    assert await drv.set_estop(True) is False

    assert drv.estop_latched is True
    assert (await drv.health_check())["active"] is True


async def test_stop_drives_off_frees_the_lines_and_leaves_the_driver_uninitialised(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    await drv.start()
    await drv.set_active(True)

    await drv.stop()

    assert fake.pad == {24: 0, 25: 0}
    assert sorted(fake.freed) == [24, 25] and fake.closed == [7]
    assert drv.running is False and drv.initialized is False and drv.hardware_backed is False
    assert (await drv.health_check())["active"] is False
    assert await drv.set_active(True) is False


async def test_a_stopped_driver_can_be_initialised_again(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    await drv.stop()

    await drv.initialize()

    assert drv.initialized is True and drv.hardware_backed is True
    assert await drv.set_active(True) is True
    assert fake.pad == {24: 1, 25: 0}


async def test_stop_never_raises_and_reports_a_possibly_on_blade(non_sim):
    fake = FakeLgpio()
    drv = _driver(fake)
    await drv.initialize()
    await drv.set_active(True)
    fake.hold_pad(24, 1)
    fake.fail_free(24)
    fake.fail_free(25)

    await drv.stop()  # must not raise

    assert (await drv.health_check())["active"] is True, "the OFF was never confirmed"
    assert fake.closed == [7]


# ----------------------------------------------------------------------------- the gate


@contextlib.contextmanager
def watch_lgpio_imports(monkeypatch):
    """Block lgpio and record every attempt to import it (lazy imports included)."""
    attempts: list[str] = []
    real_import = builtins.__import__

    def spy(name, *args, **kwargs):
        if name == "lgpio" or name.startswith("lgpio."):
            attempts.append(name)
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, "lgpio", None)
        scoped.setattr(builtins, "__import__", spy)
        yield attempts


@pytest.mark.parametrize(
    ("sim_mode", "config", "armed"),
    [
        ("1", {"gpio_enabled": True}, False),  # SIM_MODE never arms the real adapter
        ("0", {}, False),  # no ibt4 block: stub
        ("0", {"gpio_enabled": False}, False),  # the shipped value
        ("0", {"gpio_enabled": None}, False),
        ("0", {"gpio_enabled": "true"}, False),  # only a real boolean True arms
        ("0", {"gpio_enabled": 1}, False),
        ("0", {"gpio_enabled": True}, True),
    ],
)
def test_the_gate_arms_the_real_adapter_only_for_enabled_true_outside_sim(
    monkeypatch, sim_mode, config, armed
):
    monkeypatch.setenv("SIM_MODE", sim_mode)
    with watch_lgpio_imports(monkeypatch) as attempts:
        drv = IBT4BladeDriver(config)

    if armed:
        assert type(drv._gpio) is LgpioBladeAdapter
    else:
        assert type(drv._gpio) is _GPIOAdapter, "the flag-only stub"
    assert drv.hardware_backed is False, "nothing is armed before initialize()"
    assert attempts == [], "building a driver must never import lgpio"


def test_the_gate_passes_pins_and_on_levels_from_the_config(non_sim):
    drv = IBT4BladeDriver(
        {
            "gpio_enabled": True,
            "pins": {"in1": 20, "in2": 21},
            "on_levels": {"in1": 0, "in2": 1},
        }
    )
    assert (drv.in1, drv.in2) == (20, 21)
    assert isinstance(drv._gpio, LgpioBladeAdapter)
    assert (drv._gpio.in1, drv._gpio.in2, drv._gpio.on_levels) == (20, 21, (0, 1))


def test_the_gate_falls_back_to_the_documented_placeholder_levels(non_sim):
    drv = IBT4BladeDriver({"gpio_enabled": True})
    assert (drv._gpio.in1, drv._gpio.in2, drv._gpio.on_levels) == (24, 25, (1, 0))


def test_an_explicit_adapter_overrides_the_gate(non_sim):
    stub = _GPIOAdapter(24, 25)
    drv = IBT4BladeDriver({"gpio_enabled": True}, gpio_adapter=stub)
    assert drv._gpio is stub


async def test_an_armed_driver_with_an_unusable_on_levels_fails_closed(non_sim, monkeypatch):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    drv = IBT4BladeDriver({"gpio_enabled": True, "on_levels": {"in1": 0, "in2": 0}})

    await drv.initialize()

    assert drv.initialized is False
    assert "identical to OFF" in (await drv.health_check())["last_error"]
    assert fake.calls == [], "a bad config must be rejected before any GPIO is touched"


# ------------------------------------------------------------------ service wiring (YAML)


def _write_config(tmp_path, ibt4_block: str | None) -> None:
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    (tmp_path / "hardware.yaml").write_text(ibt4_block or "motor_controller: robohat-rp2040\n")


ARMED_YAML = """\
ibt4:
  gpio_enabled: true
  pins:
    in1: 24
    in2: 25
  on_levels:
    in1: 0
    in2: 1
"""


def test_the_shipped_config_is_unarmed(monkeypatch):
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(ROOT / "config"))
    cfg = blade_service._load_ibt4_config()
    assert cfg["gpio_enabled"] is False, "shipping config/hardware.yaml must not arm the blade GPIO"
    assert cfg["pins"] == {"in1": 24, "in2": 25}


def test_the_service_reads_the_ibt4_block_from_the_config_dir(monkeypatch, tmp_path):
    _write_config(tmp_path, ARMED_YAML)
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    assert blade_service._load_ibt4_config() == {
        "gpio_enabled": True,
        "pins": {"in1": 24, "in2": 25},
        "on_levels": {"in1": 0, "in2": 1},
    }


@pytest.mark.parametrize(
    ("label", "yaml_text"),
    [
        ("block absent", None),
        ("same pin twice", "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 24}\n"),
        ("pin out of range", "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 99}\n"),
        ("hardware.yaml is not a mapping", "- not\n- a mapping\n"),
    ],
)
def test_a_missing_or_broken_config_never_arms_and_warns(
    monkeypatch, tmp_path, caplog, label, yaml_text
):
    _write_config(tmp_path, yaml_text)
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger=blade_service.logger.name):
        cfg = blade_service._load_ibt4_config()
    assert cfg == {}, label
    assert any("NOT armed" in r.getMessage() for r in caplog.records), label


async def test_get_blade_service_wires_the_yaml_through_to_the_lgpio_adapter(
    non_sim, monkeypatch, tmp_path
):
    """YAML -> ConfigLoader -> BladeService -> driver gate -> adapter -> (lazy) ``import lgpio``."""
    fake = FakeLgpio()
    _write_config(tmp_path, ARMED_YAML)
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    monkeypatch.setattr(blade_service, "_blade_service", None)

    svc = blade_service.get_blade_service()
    assert blade_service.get_blade_service() is svc, "still a process-wide singleton"
    assert fake.calls == [], "constructing the service touches no GPIO"

    assert await svc.initialize() is True
    assert await svc.set_active(True) is True
    assert fake.pad == {24: 0, 25: 1}, "the YAML on_levels (0, 1) reached the wire"

    report = await svc.emergency_stop()

    assert fake.pad == {24: 0, 25: 0}
    assert report == {
        "latched": True,
        "off": True,
        "initialized": True,
        "hardware_backed": True,
        "state_confirmed": True,
        "gpio_error": None,
        "pins": {"in1": 24, "in2": 25},
        "error": None,
    }


async def test_get_blade_service_with_an_unarmed_config_never_touches_lgpio(non_sim, monkeypatch):
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(ROOT / "config"))
    monkeypatch.setattr(blade_service, "_blade_service", None)
    with watch_lgpio_imports(monkeypatch) as attempts:
        svc = blade_service.get_blade_service()
        assert await svc.initialize() is True
        assert await svc.set_active(True) is True
        report = await svc.emergency_stop()

    assert attempts == []
    assert report["hardware_backed"] is False, "still the flag-only stub"
    assert report["off"] is True and report["gpio_error"] is None


async def test_the_service_stays_uninitialised_and_retries_when_the_driver_cannot_claim(
    non_sim, monkeypatch
):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})

    assert await svc.initialize() is False
    assert await svc.set_active(True) is False, "refused, not 'pretend it worked'"
    assert fake.writes() == []

    fake.clear_claim_fault(25)  # the owner moved the pin / the kernel released the line
    assert await svc.initialize() is True, "the next call retries the setup"
    assert await svc.set_active(True) is True
    assert fake.pad == {24: 1, 25: 0}


async def test_emergency_stop_retries_the_off_write_once_and_reports_the_last_one(
    non_sim, monkeypatch
):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})
    await svc.initialize()
    await svc.set_active(True)
    fake.fail_write(24, "raise", nth=2)  # the first OFF write (latch) fails, the retry works

    report = await svc.emergency_stop()

    assert report["off"] is True and report["state_confirmed"] is True
    assert report["gpio_error"] is None, "a confirmed retry clears the failure text"
    assert fake.pad == {24: 0, 25: 0}
    assert fake.writes()[-4:] == [(24, 0), (25, 0), (24, 0), (25, 0)]


async def test_emergency_stop_off_is_false_when_the_last_write_is_not_confirmed(
    non_sim, monkeypatch
):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})
    await svc.initialize()
    await svc.set_active(True)
    fake.hold_pad(24, 1)

    report = await svc.emergency_stop()

    assert report["off"] is False and report["latched"] is True
    assert report["state_confirmed"] is False
    assert "GPIO24 reads back 1" in report["gpio_error"]
    assert svc.emergency_latched is True


async def test_park_reports_the_confirmed_off_without_latching(non_sim, monkeypatch):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})
    await svc.initialize()
    await svc.set_active(True)

    report = await svc.park()

    assert report["off"] is True and report["latched"] is False
    assert fake.pad == {24: 0, 25: 0} and svc.emergency_latched is False


def _stop_report_for(*, latched, off, initialized, hardware_backed, gpio_error=None):
    return {
        "latched": latched,
        "off": off,
        "initialized": initialized,
        "hardware_backed": hardware_backed,
        "state_confirmed": True,
        "gpio_error": gpio_error,
        "pins": {"in1": 24, "in2": 25},
        "error": None,
    }


async def test_emergency_stop_on_an_armed_never_initialised_service_claims_the_lines_first(
    non_sim, monkeypatch
):
    """A restarted backend has claimed nothing: the stop must claim, drive low and read back."""
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})

    report = await svc.emergency_stop()

    assert report == _stop_report_for(
        latched=True, off=True, initialized=True, hardware_backed=True
    )
    assert fake.names()[:5] == [
        "gpiochip_open",
        "gpio_claim_output",
        "gpio_claim_output",
        "gpio_read",
        "gpio_read",
    ]
    assert [c for c in fake.calls if c[0] == "gpio_claim_output"] == [
        ("gpio_claim_output", 7, 24, 0),
        ("gpio_claim_output", 7, 25, 0),
    ], "both lines are claimed already low"
    assert fake.writes() == [(24, 0), (25, 0)] and fake.pad == {24: 0, 25: 0}


async def test_park_on_an_armed_never_initialised_service_claims_the_lines_without_latching(
    non_sim, monkeypatch
):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})

    report = await svc.park()

    assert report == _stop_report_for(
        latched=False, off=True, initialized=True, hardware_backed=True
    )
    assert fake.writes() == [(24, 0), (25, 0)] and svc.emergency_latched is False


async def test_emergency_stop_on_an_armed_service_whose_lines_cannot_be_claimed(
    non_sim, monkeypatch
):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})

    report = await svc.emergency_stop()

    assert report["initialized"] is False and report["off"] is False
    assert report["hardware_backed"] is False and report["latched"] is True
    assert "GPIO25" in report["gpio_error"] and "GPIO busy" in report["gpio_error"]
    assert fake.freed == [24] and fake.closed == [7] and fake.writes() == []
    assert svc.emergency_latched is True


@pytest.mark.parametrize("method", ["emergency_stop", "park"])
async def test_a_stop_on_an_unarmed_never_initialised_service_changes_nothing(
    non_sim, monkeypatch, method
):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": False})

    report = await getattr(svc, method)()

    assert report == _stop_report_for(
        latched=method == "emergency_stop", off=False, initialized=False, hardware_backed=False
    )
    assert fake.calls == [] and svc._driver.initialized is False


async def test_an_armed_service_in_sim_mode_is_still_the_stub_and_never_touches_lgpio(
    monkeypatch,
):
    monkeypatch.setenv("SIM_MODE", "1")
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = blade_service.BladeService({"gpio_enabled": True})

    report = await svc.emergency_stop()

    assert report["hardware_backed"] is False, "SIM_MODE never selects the real adapter"
    assert report["off"] is True and report["initialized"] is True
    assert fake.calls == []
