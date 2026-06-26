"""RC-PWM servo / linear-actuator and GPIO-relay drivers for the tractor.

Positional actuators (steering, throttle, gas pedal, clutch, gear) are RC-PWM
channels: a normalized command maps to a pulse width in microseconds that the
RoboHAT RP2040 emits on the corresponding channel. The microsecond mapping is a
pure function (unit-testable); the actual serial write is performed by
``TractorControlService`` via the RoboHAT bridge.

The starter and blade-PTO are GPIO relays. ``RelayActuator`` is SIM-safe: it
only touches GPIO on real hardware (lazy import) and otherwise tracks state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ...core.simulation import is_simulation_mode

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ServoCalibration:
    """Pulse-width calibration for one RC-PWM actuator channel."""

    channel: int
    us_min: int = 1000
    us_center: int = 1500
    us_max: int = 2000
    # True: command is -1..1 mapped around center (e.g. steering).
    # False: command is 0..1 mapped us_min..us_max (e.g. throttle, pedal, clutch).
    bidirectional: bool = True


class ServoActuator:
    """A single RC-PWM positional actuator (servo / linear actuator)."""

    def __init__(self, name: str, calibration: ServoCalibration):
        self.name = name
        self.cal = calibration
        self._value: float = 0.0
        self._us: int = calibration.us_center if calibration.bidirectional else calibration.us_min

    def microseconds(self, value: float) -> int:
        """Map a normalized command to a pulse width in microseconds (pure)."""
        cal = self.cal
        if cal.bidirectional:
            v = _clamp(value, -1.0, 1.0)
            if v >= 0:
                us = cal.us_center + v * (cal.us_max - cal.us_center)
            else:
                us = cal.us_center + v * (cal.us_center - cal.us_min)
        else:
            v = _clamp(value, 0.0, 1.0)
            us = cal.us_min + v * (cal.us_max - cal.us_min)
        return int(round(us))

    def command(self, value: float) -> int:
        """Record a new command and return its pulse width (µs)."""
        self._us = self.microseconds(value)
        self._value = _clamp(value, -1.0 if self.cal.bidirectional else 0.0, 1.0)
        return self._us

    @property
    def value(self) -> float:
        return self._value

    @property
    def us(self) -> int:
        return self._us


@dataclass
class GearCalibration:
    """Pulse widths for the three forward/neutral/reverse selector positions."""

    channel: int
    us_forward: int = 1900
    us_neutral: int = 1500
    us_reverse: int = 1100


class GearActuator:
    """Discrete forward/neutral/reverse selector driven as an RC-PWM channel."""

    def __init__(self, name: str, calibration: GearCalibration):
        self.name = name
        self.cal = calibration
        self._gear: str = "neutral"

    def microseconds(self, gear: str) -> int:
        mapping = {
            "forward": self.cal.us_forward,
            "neutral": self.cal.us_neutral,
            "reverse": self.cal.us_reverse,
        }
        return int(mapping.get(str(gear), self.cal.us_neutral))

    def command(self, gear: str) -> int:
        self._gear = str(gear)
        return self.microseconds(self._gear)

    @property
    def gear(self) -> str:
        return self._gear


class RelayActuator:
    """A GPIO relay (latching on/off, with a momentary pulse for the starter).

    SIM-safe: on real hardware it drives the GPIO via ``gpiozero``; off-device it
    only tracks state so the rest of the stack stays importable and testable.
    """

    def __init__(self, name: str, gpio: int, active_high: bool = True):
        self.name = name
        self.gpio = gpio
        self.active_high = active_high
        self._on: bool = False
        self._device: Any = None
        self._sim = is_simulation_mode()

    def _ensure_device(self) -> Any:
        if self._sim or self._device is not None:
            return self._device
        try:  # lazy hardware import — never at module load
            from gpiozero import OutputDevice  # type: ignore

            self._device = OutputDevice(
                self.gpio, active_high=self.active_high, initial_value=False
            )
        except Exception as exc:  # pragma: no cover - hardware/import error path
            logger.warning(
                "Relay %s GPIO%s unavailable (%s); running in sim", self.name, self.gpio, exc
            )
            self._sim = True
        return self._device

    def set(self, on: bool) -> None:
        self._on = bool(on)
        dev = self._ensure_device()
        if dev is not None:
            dev.on() if self._on else dev.off()

    async def pulse(self, ms: int) -> None:
        """Momentarily energize the relay (e.g. crank the starter)."""
        self.set(True)
        try:
            await asyncio.sleep(max(0, ms) / 1000.0)
        finally:
            self.set(False)

    @property
    def state(self) -> bool:
        return self._on
