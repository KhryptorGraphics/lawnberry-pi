"""Unit tests for the PCA9685 I2C PWM driver (ride-on tractor actuation).

Covers the pure microsecond<->tick math and, critically, the failure-mode
behaviors the driver must get right: in SIM_MODE the driver is a simulated
success; hardware absent/failed at ``initialize()`` is tolerated at start-up but
is **never silent** (``is_ready()`` False, ``ok: False`` in ``health_check``,
``set_pwm_us`` raises ``PCA9685UnavailableError``); and a write failure on
hardware that was already confirmed present must propagate (see the driver's
module docstring for why: swallowing it would make the tractor's emergency-stop
escalation path unreachable dead code).
"""

from __future__ import annotations

import sys
import types

import pytest

from backend.src.drivers.actuators.pca9685_driver import (
    PCA9685Driver,
    PCA9685UnavailableError,
)


def _install_fake_smbus2(monkeypatch, *, fail_write: bool = False):
    """Inject a fake smbus2 module (mirrors tests/unit/test_sensor_presence.py).

    The ``initialize()`` setup sequence (MODE1/PRESCALE/MODE2 byte writes)
    always succeeds; the 4-byte block write ``set_pwm_us`` issues fails iff
    ``fail_write`` is True, simulating a mid-mission bus fault.
    """

    class _FakeSMBus:
        last_write: tuple | None = None

        def __init__(self, bus):
            self.bus = bus

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_byte_data(self, addr, reg, val):
            pass

        def write_i2c_block_data(self, addr, reg, data):
            if fail_write:
                raise OSError(121, "Remote I/O error")
            _FakeSMBus.last_write = (addr, reg, data)

    mod = types.ModuleType("smbus2")
    mod.SMBus = _FakeSMBus
    monkeypatch.setitem(sys.modules, "smbus2", mod)
    return _FakeSMBus


def _force_real(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setattr(
        "backend.src.drivers.actuators.pca9685_driver.is_simulation_mode", lambda: False
    )


# ------------------------------ pure µs/tick math ------------------------------


def test_ticks_for_us_maps_pulse_width_at_50hz():
    d = PCA9685Driver({"frequency_hz": 50})
    assert d._ticks_for_us(1000) == (0, 205)
    assert d._ticks_for_us(1500) == (0, 307)
    assert d._ticks_for_us(2000) == (0, 410)


def test_ticks_for_us_clamps_to_12bit_range():
    d = PCA9685Driver({"frequency_hz": 50})
    assert d._ticks_for_us(-500) == (0, 0)
    assert d._ticks_for_us(1_000_000) == (0, 4095)


def test_prescale_matches_datasheet_formula_at_50hz():
    d = PCA9685Driver({"frequency_hz": 50})
    assert d._prescale() == 121


# ------------------------------ channel validation ------------------------------


@pytest.mark.asyncio
async def test_set_pwm_us_rejects_out_of_range_channel():
    d = PCA9685Driver({})
    await d.initialize()  # SIM_MODE=1 under the test runner -> tolerated no-op
    with pytest.raises(ValueError):
        await d.set_pwm_us(16, 1500)
    with pytest.raises(ValueError):
        await d.set_pwm_us(-1, 1500)


# ------------------------- the two-failure-mode distinction -------------------------


@pytest.mark.asyncio
async def test_sim_mode_driver_is_ready_and_a_simulated_noop(monkeypatch):
    """SIM_MODE: there is no chip by design: ready, and set_pwm_us() is a simulated no-op."""
    d = PCA9685Driver({})
    await d.initialize()
    assert d.initialized is True
    assert d._hw_ok is False

    class _ExplodingSMBus:
        def __init__(self, *a, **k):
            raise AssertionError("SMBus must not be touched while hardware is absent")

    mod = types.ModuleType("smbus2")
    mod.SMBus = _ExplodingSMBus
    monkeypatch.setitem(sys.modules, "smbus2", mod)

    await d.set_pwm_us(0, 1500)  # must not raise, must not touch smbus2


@pytest.mark.asyncio
async def test_initialize_without_smbus2_is_a_reported_fault_not_a_noop(monkeypatch):
    """No smbus2 / no I2C bus outside SIM_MODE: initialize() tolerates it so the
    backend can still start, but the driver is NOT ready, says why, and refuses
    to pretend a pulse was emitted."""
    _force_real(monkeypatch)
    monkeypatch.setitem(sys.modules, "smbus2", None)  # import raises ImportError
    d = PCA9685Driver({})
    await d.initialize()  # must not raise
    assert d.initialized is True
    assert d._hw_ok is False
    assert d.is_ready() is False
    health = await d.health_check()
    assert health["ok"] is False and health["hardware_ok"] is False
    assert "not available" in health["error"]
    with pytest.raises(PCA9685UnavailableError):
        await d.set_pwm_us(0, 1500)


@pytest.mark.asyncio
async def test_chip_that_does_not_answer_setup_is_a_reported_fault(monkeypatch):
    """smbus2 present but the chip NAKs (not wired / wrong address / bus down)."""
    _force_real(monkeypatch)

    class _DeadChipSMBus:
        def __init__(self, bus):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_byte_data(self, addr, reg, val):
            raise OSError(121, "Remote I/O error")

    mod = types.ModuleType("smbus2")
    mod.SMBus = _DeadChipSMBus
    monkeypatch.setitem(sys.modules, "smbus2", mod)
    d = PCA9685Driver({"address": 0x41, "bus": 1})
    await d.initialize()
    assert d.is_ready() is False
    assert "0x41" in d.last_error and "Remote I/O error" in d.last_error
    with pytest.raises(PCA9685UnavailableError, match="Remote I/O error"):
        await d.set_pwm_us(2, 1500)


@pytest.mark.asyncio
async def test_never_initialised_driver_outside_sim_refuses(monkeypatch):
    _force_real(monkeypatch)
    d = PCA9685Driver({})
    assert d.is_ready() is False
    with pytest.raises(PCA9685UnavailableError):
        await d.set_pwm_us(0, 1500)


@pytest.mark.asyncio
async def test_write_failure_is_recorded_and_cleared_by_the_next_success(monkeypatch):
    """A transient bus fault shows up in health until a later write succeeds."""
    _force_real(monkeypatch)
    _install_fake_smbus2(monkeypatch, fail_write=True)
    d = PCA9685Driver({})
    await d.initialize()
    with pytest.raises(OSError):
        await d.set_pwm_us(0, 1500)
    health = await d.health_check()
    assert health["ok"] is False and "write failed" in health["error"]

    # Bus recovers: swap in a fake whose writes succeed.
    _install_fake_smbus2(monkeypatch, fail_write=False)
    await d.set_pwm_us(0, 1500)
    assert (await d.health_check())["ok"] is True


@pytest.mark.asyncio
async def test_hardware_present_then_write_fails_propagates(monkeypatch):
    """A mid-mission bus fault on hardware confirmed present must propagate,
    not be swallowed -- this is what lets tractor_service's emergency_stop
    (and the normal command path) see a real transport failure."""
    _force_real(monkeypatch)
    _install_fake_smbus2(monkeypatch, fail_write=True)
    d = PCA9685Driver({})
    await d.initialize()
    assert d._hw_ok is True  # hardware confirmed present during initialize()

    with pytest.raises(OSError):
        await d.set_pwm_us(0, 1500)


@pytest.mark.asyncio
async def test_hardware_present_write_succeeds(monkeypatch):
    """Happy path: confirm the expected channel register + byte count."""
    _force_real(monkeypatch)
    fake_cls = _install_fake_smbus2(monkeypatch, fail_write=False)
    d = PCA9685Driver({"address": 0x41})
    await d.initialize()
    await d.set_pwm_us(2, 1500)

    addr, reg, data = fake_cls.last_write
    assert addr == 0x41
    assert reg == 0x06 + 4 * 2  # LED0_ON_L + 4 * channel
    assert len(data) == 4
