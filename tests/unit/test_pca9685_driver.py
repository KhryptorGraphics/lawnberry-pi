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
    PCA9685ReadbackError,
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


# ------------------------- read-back (the watchdog's actuation evidence) -------------------------


def _install_register_file_smbus2(monkeypatch):
    """Fake smbus2 with a register file behind 4-byte block reads and writes."""
    regs = bytearray(256)
    faults = types.SimpleNamespace(read=False)

    class _Bus:
        def __init__(self, bus):
            self.bus = bus

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_byte_data(self, addr, reg, val):
            regs[reg] = val

        def write_i2c_block_data(self, addr, reg, data):
            for i, byte in enumerate(data):
                regs[reg + i] = byte

        def read_i2c_block_data(self, addr, reg, length):
            if faults.read:
                raise OSError(5, "Input/output error")
            return list(regs[reg : reg + length])

    mod = types.ModuleType("smbus2")
    mod.SMBus = _Bus
    monkeypatch.setitem(sys.modules, "smbus2", mod)
    return regs, faults


async def _ready_driver(monkeypatch):
    _force_real(monkeypatch)
    regs, faults = _install_register_file_smbus2(monkeypatch)
    d = PCA9685Driver({})
    await d.initialize()
    assert d.is_ready()
    return d, regs, faults


@pytest.mark.asyncio
async def test_verify_pwm_us_accepts_exactly_what_was_written(monkeypatch):
    d, _regs, _faults = await _ready_driver(monkeypatch)
    for channel, us in ((0, 1000), (0, 1500), (2, 1950), (1, 2000), (15, 1234)):
        await d.set_pwm_us(channel, us)
        await d.verify_pwm_us(channel, us)  # must not raise


@pytest.mark.asyncio
async def test_verify_pwm_us_detects_a_different_pulse(monkeypatch):
    d, _regs, _faults = await _ready_driver(monkeypatch)
    await d.set_pwm_us(0, 1500)
    with pytest.raises(PCA9685ReadbackError, match="channel 0 holds"):
        await d.verify_pwm_us(0, 1600)


@pytest.mark.asyncio
async def test_verify_pwm_us_detects_a_chip_reset_and_a_stuck_on_output(monkeypatch):
    """After a brown-out the chip's outputs are FULL OFF (LEDn_OFF_H bit 4); a write-only
    keepalive cannot see that, the read-back does. FULL ON is a fault too."""
    d, regs, _faults = await _ready_driver(monkeypatch)
    await d.set_pwm_us(0, 1500)
    base = d._LED0_ON_L

    regs[base : base + 4] = bytes([0x00, 0x00, 0x00, 0x10])  # power-on state: FULL OFF
    with pytest.raises(PCA9685ReadbackError):
        await d.verify_pwm_us(0, 1500)

    regs[base : base + 4] = bytes([0x00, 0x10, 0x00, 0x00])  # FULL ON
    with pytest.raises(PCA9685ReadbackError):
        await d.verify_pwm_us(0, 1500)


@pytest.mark.asyncio
async def test_verify_pwm_us_masks_reserved_bits_only(monkeypatch):
    d, regs, _faults = await _ready_driver(monkeypatch)
    await d.set_pwm_us(3, 1500)
    off_h = d._LED0_ON_L + 4 * 3 + 3
    regs[off_h] |= 0xE0  # reserved bits 7:5 read back as noise
    await d.verify_pwm_us(3, 1500)  # still matches


@pytest.mark.asyncio
async def test_verify_pwm_us_propagates_a_read_failure_and_never_writes(monkeypatch):
    d, regs, faults = await _ready_driver(monkeypatch)
    await d.set_pwm_us(0, 1500)
    before = bytes(regs)
    faults.read = True
    with pytest.raises(OSError, match="Input/output error"):
        await d.verify_pwm_us(0, 1500)
    assert bytes(regs) == before, "verifying must not change an output"
    assert d.last_error is None, "last_error is the write path's record"


@pytest.mark.asyncio
async def test_verify_pwm_us_is_simulated_in_sim_mode_and_refuses_an_absent_chip(monkeypatch):
    d = PCA9685Driver({})
    await d.verify_pwm_us(0, 1500)  # SIM_MODE: no chip by design, simulated success
    _force_real(monkeypatch)
    with pytest.raises(PCA9685UnavailableError):
        await d.verify_pwm_us(0, 1500)
    d2, _regs, _faults = await _ready_driver(monkeypatch)
    with pytest.raises(ValueError):
        await d2.verify_pwm_us(16, 1500)
