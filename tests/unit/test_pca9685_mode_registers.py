"""PCA9685 read-back covers the registers that decide whether ANY pulse is emitted (round 2).

PR #20 review BL-07 (``verify_pwm_us`` read only LEDn, so it passed with MODE1 SLEEP set, PRESCALE
changed or a restart pending), SUSPECTED S-1 (a backend restart against a chip that kept running
latched RESTART and halted every channel) and mutants P01/P02/P07 (the FULL ON / FULL OFF flag bits
were never tested with intact counts).

The chip model follows the PCA9685 datasheet MODE1 semantics (SLEEP set while a channel is active
latches RESTART, reading shows it, writing 1 to it restarts the channels and clears it). That is
datasheet knowledge, NOT a bench measurement: confirm on the breakout (audit M-28).
"""

from __future__ import annotations

import sys
import types

import pytest

from backend.src.drivers.actuators.pca9685_driver import PCA9685Driver, PCA9685ReadbackError


class Chip:
    """A PCA9685 register file with datasheet MODE1 behaviour (power-on: asleep, FULL OFF)."""

    def __init__(self) -> None:
        self.regs = bytearray(256)
        self.regs[0x00] = 0x11
        for ch in range(16):
            self.regs[0x06 + 4 * ch + 3] = 0x10
        self.mode1_writes: list[int] = []

    def channel_active(self) -> bool:
        return any(
            (self.regs[0x06 + 4 * c + 2] or (self.regs[0x06 + 4 * c + 3] & 0x0F))
            and not (self.regs[0x06 + 4 * c + 3] & 0x10)
            for c in range(16)
        )

    def write_mode1(self, value: int) -> None:
        self.mode1_writes.append(value)
        old = self.regs[0]
        latched = bool(old & 0x80)
        if value & 0x80:
            latched = False  # writing 1 to RESTART restarts the channels and clears the bit
        elif (value & 0x10) and not (old & 0x10) and self.channel_active():
            latched = True  # SLEEP set while a channel runs: halted until RESTART is written
        self.regs[0] = (value & 0x7F) | (0x80 if latched else 0)

    def running(self) -> bool:
        """Would pulses come out? (oscillator on and no restart pending; OE is not modelled)"""
        return not (self.regs[0] & 0x10) and not (self.regs[0] & 0x80)


def install(monkeypatch, chip: Chip) -> None:
    class Bus:
        def __init__(self, bus):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_byte_data(self, addr, reg, value):
            if reg == 0x00:
                chip.write_mode1(value)
            else:
                chip.regs[reg] = value

        def write_i2c_block_data(self, addr, reg, data):
            for i, byte in enumerate(data):
                chip.regs[reg + i] = byte

        def read_i2c_block_data(self, addr, reg, length):
            return list(chip.regs[reg : reg + length])

    module = types.ModuleType("smbus2")
    module.SMBus = Bus
    monkeypatch.setitem(sys.modules, "smbus2", module)


@pytest.fixture
def chip(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setattr(
        "backend.src.drivers.actuators.pca9685_driver.is_simulation_mode", lambda: False
    )
    c = Chip()
    install(monkeypatch, c)
    return c


async def ready(chip: Chip) -> PCA9685Driver:
    driver = PCA9685Driver({"address": 0x41})
    await driver.initialize()
    assert driver.is_ready()
    await driver.set_pwm_us(0, 1500)
    await driver.verify_pwm_us(0, 1500)  # the baseline passes
    return driver


# ----------------------------------------------------------------------------- BL-07


async def test_the_readback_catches_a_chip_that_went_to_sleep_behind_our_back(chip):
    driver = await ready(chip)
    chip.regs[0] |= 0x10  # SLEEP: the LED registers still hold 1500 us, no pulse is emitted
    with pytest.raises(PCA9685ReadbackError, match="SLEEP set"):
        await driver.verify_pwm_us(0, 1500)


async def test_the_readback_catches_a_changed_pwm_rate(chip):
    driver = await ready(chip)
    chip.regs[0xFE] = 0x1E  # 200 Hz instead of the configured 50 Hz
    with pytest.raises(PCA9685ReadbackError, match="PRESCALE"):
        await driver.verify_pwm_us(0, 1500)


async def test_the_readback_catches_a_pending_restart_a_lost_autoincrement_and_extclk(chip):
    driver = await ready(chip)
    saved = chip.regs[0]
    for bits, needle in ((saved | 0x80, "RESTART pending"), (saved | 0x40, "EXTCLK")):
        chip.regs[0] = bits
        with pytest.raises(PCA9685ReadbackError, match=needle):
            await driver.verify_pwm_us(0, 1500)
    chip.regs[0] = saved & ~0x20
    with pytest.raises(PCA9685ReadbackError, match="auto-increment lost"):
        await driver.verify_pwm_us(0, 1500)
    chip.regs[0] = saved
    await driver.verify_pwm_us(0, 1500)


# ------------------------------------------------------------------ SUSPECTED S-1 (restart)


async def test_a_backend_restart_against_a_running_chip_restarts_its_channels(chip):
    first = await ready(chip)
    assert chip.running() and first is not None

    chip.mode1_writes.clear()
    second = PCA9685Driver({"address": 0x41})
    await second.initialize()  # the chip kept running: this is a second process bring-up
    await second.set_pwm_us(0, 1500)

    assert chip.running(), "the sleep latched RESTART; without the final RESTART write no pulses"
    assert chip.mode1_writes[-1] & 0x80, "the datasheet restart sequence ends with RESTART = 1"
    await second.verify_pwm_us(0, 1500)


async def test_a_pending_restart_is_what_the_readback_reports_when_the_restart_never_came(chip):
    driver = await ready(chip)
    chip.write_mode1(0x30)  # a stray sleep of a running chip latches RESTART
    chip.write_mode1(0x20)  # and waking it does not restart the channels
    assert not chip.running()
    with pytest.raises(PCA9685ReadbackError, match="RESTART pending"):
        await driver.verify_pwm_us(0, 1500)


# ------------------------------------------------- P01 / P02 / P07: the FULL ON / FULL OFF flags


@pytest.mark.parametrize(
    ("on_h", "off_h"),
    [(0x10, 0x00), (0x00, 0x10), (0x10, 0x10)],
    ids=["full_on", "full_off", "both"],
)
async def test_a_full_flag_with_intact_counts_is_not_the_commanded_pulse(chip, on_h, off_h):
    driver = await ready(chip)
    base = 0x06  # channel 0: ON_L, ON_H, OFF_L, OFF_H
    counts_on_h, counts_off_h = chip.regs[base + 1], chip.regs[base + 3]
    chip.regs[base + 1] = counts_on_h | on_h
    chip.regs[base + 3] = counts_off_h | off_h

    with pytest.raises(PCA9685ReadbackError, match="channel 0 holds"):
        await driver.verify_pwm_us(0, 1500)

    chip.regs[base + 1], chip.regs[base + 3] = counts_on_h, counts_off_h
    await driver.verify_pwm_us(0, 1500)
