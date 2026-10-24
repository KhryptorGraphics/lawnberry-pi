"""PCA9685 16-channel I2C PWM driver for the ride-on tractor's actuators.

Raw ``smbus2`` register access (already a project dependency), matching this
codebase's established precedent (``drivers/sensors/ina3221_driver.py``) of
talking to well-documented I2C parts directly rather than pulling in a vendor
library for a few dozen register writes.

Three distinct states, handled deliberately differently:

- **SIM_MODE** (``is_simulation_mode()``): there is no chip by design.
  ``initialize()`` succeeds, ``is_ready()`` is ``True`` and ``set_pwm_us`` is a
  simulated no-op so the rest of the stack stays testable.
- **Hardware absent or failed at ``initialize()`` time** (no PCA9685 wired up,
  no I2C bus on this host, chip does not ACK) is *tolerated at start-up* -- the
  backend must still come up, with its watchdog and E-stop wiring -- but it is
  **never silent**: ``_hw_ok`` stays ``False``, ``is_ready()`` is ``False``,
  ``health_check()`` reports ``ok: False`` plus the error, and every
  ``set_pwm_us`` raises :class:`PCA9685UnavailableError` instead of returning
  as if a pulse had been emitted. ``TractorControlService`` turns that into
  refused commands and a reported fault.
- **Hardware was present and a write fails afterward** (a mid-mission I2C bus
  fault) is *not* swallowed either: ``set_pwm_us`` lets the exception
  propagate. The PCA9685 holds its last commanded pulse on I2C/host loss
  rather than failing safe on its own, so ``tractor_service.py``'s
  emergency-stop and command-failure paths depend on a real transport fault
  actually reaching them.

A physical bus-fault failsafe is a hardware-phase concern tracked separately
and is *not* implemented here -- this driver only makes the fault observable
to its caller. Note for that work (PCA9685 datasheet, confirm on the actual
breakout): ``OE`` is active-LOW, so *driving it low enables the outputs*;
driving it HIGH disables them and, with ``MODE2 = 0x04`` (``OUTNE = 00``),
leaves every channel low -- no pulse at all, which is "stop commanding", not
"go to neutral".
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


class PCA9685Error(RuntimeError):
    """Base class for PCA9685 driver faults."""


class PCA9685UnavailableError(PCA9685Error):
    """The chip was never confirmed present (or failed to initialise).

    Raised by :meth:`PCA9685Driver.set_pwm_us` so a command can never report
    success for a pulse that was not emitted.
    """


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
        # False until initialize() confirms the chip answers. Not the same as
        # "simulated": see is_ready().
        self._hw_ok: bool = False
        # Last init/write failure, cleared by the next success. Surfaced via
        # health_check()/last_error so a fault is never only a log line.
        self._last_error: str | None = None

    @property
    def hardware_ok(self) -> bool:
        """True only when a real chip answered ``initialize()``."""
        return self._hw_ok

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        """Can ``set_pwm_us`` be expected to emit a real (or simulated) pulse?

        ``True`` in SIM_MODE (simulated by design) or once a real chip answered
        ``initialize()``. ``False`` when the chip is absent/failed: callers must
        refuse to actuate rather than assume the pulse went out.
        """
        return self._hw_ok or is_simulation_mode()

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
        self._last_error = None
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
            # Absent board, no I2C bus on this host, etc. Start-up tolerates it
            # (the backend must still come up with its safety wiring), but it is
            # recorded and every later set_pwm_us raises -- never a silent no-op.
            self._hw_ok = False
            self._last_error = (
                f"PCA9685 not available at {self._cfg.address:#x} "
                f"(bus {self._cfg.bus}): {type(exc).__name__}: {exc}"
            )
            logger.error("%s; lever/throttle commands will be REFUSED", self._last_error)

    async def start(self) -> None:  # noqa: D401
        self.running = True

    async def stop(self) -> None:  # noqa: D401
        self.running = False

    async def health_check(self) -> dict[str, Any]:  # noqa: D401
        return {
            "driver": "pca9685",
            "initialized": self.initialized,
            "running": self.running,
            "ok": self.is_ready() and self._last_error is None,
            "hardware_ok": self._hw_ok,
            "error": self._last_error,
            "address": hex(self._cfg.address),
            "bus": self._cfg.bus,
            "frequency_hz": self._cfg.frequency_hz,
            "simulation": is_simulation_mode(),
        }

    async def set_pwm_us(self, channel: int, us: int) -> None:
        """Command one channel's pulse width in microseconds.

        In SIM_MODE this is a simulated no-op. Anywhere else it either emits
        the pulse or raises: :class:`PCA9685UnavailableError` when the chip
        was never confirmed present, or the underlying transport error when a
        write to confirmed hardware fails -- see the module docstring.
        """
        if not 0 <= channel <= 15:
            raise ValueError(f"PCA9685 channel out of range 0-15: {channel}")
        if not self._hw_ok:
            if is_simulation_mode():
                return  # simulated by design: there is no chip to talk to
            raise PCA9685UnavailableError(
                self._last_error or "PCA9685 not initialised: no pulse was emitted"
            )

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

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            # Deliberately not swallowed: a fault here must reach the caller.
            self._last_error = f"write failed: {type(exc).__name__}: {exc}"
            raise
        self._last_error = None


__all__ = ["PCA9685Driver", "PCA9685Error", "PCA9685UnavailableError"]
