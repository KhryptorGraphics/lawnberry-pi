"""PCA9685 16-channel I2C PWM driver for the ride-on tractor's actuators.

Raw ``smbus2`` register access (already a project dependency), matching this
codebase's established precedent (``drivers/sensors/ina3221_driver.py``) of
talking to well-documented I2C parts directly rather than pulling in a vendor
library for a few dozen register writes.

Two distinct failure modes, handled deliberately differently:

- **Hardware absent at ``initialize()`` time** (SIM_MODE, or no PCA9685 wired
  up yet) is tolerated: ``_hw_ok`` stays ``False`` and ``set_pwm_us`` becomes a
  no-op, exactly like ``INA3221Driver`` degrades when its bus isn't present.
- **Hardware was present and a write fails afterward** (a mid-mission I2C bus
  fault) is *not* swallowed: ``set_pwm_us`` lets the exception propagate. The
  PCA9685 holds its last commanded pulse on I2C/host loss rather than failing
  safe on its own, so ``tractor_service.py``'s emergency-stop and
  command-failure paths depend on a real transport fault actually reaching
  them. Swallowing it here (as the previous RoboHAT-serial ``_send_pwm`` did)
  would make that escalation path unreachable dead code.

A physical bus-fault failsafe (e.g. a watchdog driving the board's ``OE`` pin
low) is a hardware-phase concern tracked separately and is *not* implemented
here — this driver only makes the fault observable to its caller.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from ...core.simulation import is_simulation_mode
from ..base import HardwareDriver

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Pca9685Config:
    address: int = 0x40  # datasheet default; override in config (0x40 collides with INA3221)
    bus: int = 1
    frequency_hz: float = 50.0  # standard RC servo/ESC PWM rate


class PCA9685Driver(HardwareDriver):
    """16-channel PCA9685 PWM driver addressed over I2C via smbus2."""

    _MODE1 = 0x00
    _MODE2 = 0x01
    _PRESCALE = 0xFE
    _LED0_ON_L = 0x06  # LED0_ON_L/ON_H/OFF_L/OFF_H, then +4 registers per channel

    _MODE1_AUTOINCREMENT = 0x20
    _MODE1_SLEEP = 0x10
    _MODE2_OUTDRV = 0x04  # totem-pole output (vs. open-drain)

    _OSC_CLOCK_HZ = 25_000_000
    _TICKS_PER_PERIOD = 4096

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        cfg = config or {}
        base_cfg = _Pca9685Config()
        address = cfg.get("address", base_cfg.address)
        bus = cfg.get("bus", base_cfg.bus)
        frequency_hz = cfg.get("frequency_hz", base_cfg.frequency_hz)
        self._cfg = _Pca9685Config(
            address=int(address) if address is not None else base_cfg.address,
            bus=int(bus) if bus is not None else base_cfg.bus,
            frequency_hz=float(frequency_hz) if frequency_hz else base_cfg.frequency_hz,
        )
        # Tolerated-no-op until a real initialize() confirms the chip answers.
        self._hw_ok: bool = False

    def _prescale(self) -> int:
        """PRESCALE register value for the configured PWM frequency (pure)."""
        prescale = round(self._OSC_CLOCK_HZ / (self._TICKS_PER_PERIOD * self._cfg.frequency_hz)) - 1
        return max(3, min(255, int(prescale)))

    def _ticks_for_us(self, us: int) -> tuple[int, int]:
        """Map a pulse width in microseconds to (on_tick, off_tick) (pure).

        Every channel starts its pulse at tick 0 (``on``); ``off`` is the
        fraction of the PWM period the pulse stays high, clamped to the
        12-bit tick range.
        """
        period_us = 1_000_000.0 / self._cfg.frequency_hz
        off = int(round((max(0, us) / period_us) * self._TICKS_PER_PERIOD))
        off = max(0, min(self._TICKS_PER_PERIOD - 1, off))
        return 0, off

    async def initialize(self) -> None:  # noqa: D401
        self.initialized = True
        self._hw_ok = False
        if is_simulation_mode():
            return
        try:
            from smbus2 import SMBus  # type: ignore

            prescale = self._prescale()

            def _setup() -> None:
                with SMBus(self._cfg.bus) as bus:
                    # Sleep + enable register auto-increment so the 4-byte
                    # ON/OFF block writes in set_pwm_us land in one transaction.
                    sleep_and_ai = self._MODE1_SLEEP | self._MODE1_AUTOINCREMENT
                    bus.write_byte_data(self._cfg.address, self._MODE1, sleep_and_ai)
                    bus.write_byte_data(self._cfg.address, self._PRESCALE, prescale)
                    bus.write_byte_data(self._cfg.address, self._MODE2, self._MODE2_OUTDRV)
                    bus.write_byte_data(self._cfg.address, self._MODE1, self._MODE1_AUTOINCREMENT)
                    time.sleep(0.0005)  # datasheet: >=500us after waking before further writes

            await asyncio.to_thread(_setup)
            self._hw_ok = True
        except Exception as exc:
            # Absent board, no I2C bus on this host, SIM off-device, etc.:
            # tolerated — degrade to no-op, matching INA3221Driver's pattern.
            logger.warning(
                "PCA9685 not available at %#x (bus %d): %s; actuation calls will no-op",
                self._cfg.address,
                self._cfg.bus,
                exc,
            )
            self._hw_ok = False

    async def start(self) -> None:  # noqa: D401
        self.running = True

    async def stop(self) -> None:  # noqa: D401
        self.running = False

    async def health_check(self) -> dict[str, Any]:  # noqa: D401
        return {
            "driver": "pca9685",
            "initialized": self.initialized,
            "running": self.running,
            "hardware_ok": self._hw_ok,
            "address": hex(self._cfg.address),
            "bus": self._cfg.bus,
            "frequency_hz": self._cfg.frequency_hz,
            "simulation": is_simulation_mode(),
        }

    async def set_pwm_us(self, channel: int, us: int) -> None:
        """Command one channel's pulse width in microseconds.

        No-ops if the chip was never confirmed present (SIM_MODE or absent at
        ``initialize()``). Once hardware has been confirmed present, a write
        failure is deliberately left to propagate — see module docstring.
        """
        if not 0 <= channel <= 15:
            raise ValueError(f"PCA9685 channel out of range 0-15: {channel}")
        if not self._hw_ok:
            return

        on, off = self._ticks_for_us(us)
        from smbus2 import SMBus  # type: ignore

        def _write() -> None:
            with SMBus(self._cfg.bus) as bus:
                base = self._LED0_ON_L + 4 * channel
                bus.write_i2c_block_data(
                    self._cfg.address,
                    base,
                    [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF],
                )

        # Deliberately unguarded: a fault here must reach the caller.
        await asyncio.to_thread(_write)


__all__ = ["PCA9685Driver"]
