"""PCA9685 positional-servo and GPIO-relay drivers for the zero-turn mower.

Positional actuators (left/right drive lever, throttle) are PCA9685 I2C PWM
channels: a normalized command maps to a pulse width in microseconds that the
PCA9685 emits on the corresponding channel. The microsecond mapping is a pure
function (unit-testable); the actual I2C write is performed by
``TractorControlService`` via ``PCA9685Driver``.

The starter and blade-PTO are GPIO relays. ``RelayActuator`` is SIM-safe: in
SIM_MODE it only tracks state; otherwise it drives the GPIO (lazy import) and
raises ``RelayUnavailableError`` if the pin cannot be driven -- it never
degrades silently to a no-op. A relay built ``armed=False`` (the tractor platform is
disabled in ``config/tractor.yaml``) never claims or drives its pin outside SIM_MODE: its
GPIO may belong to something else on the header (audit H-02b), and a disabled platform
owns no pin.

A non-finite command (NaN/inf) is refused (``ValueError``) instead of being clamped: NaN
clamps to full scale, and an invalid command must never turn into maximum motion.
"""

from __future__ import annotations

import asyncio
import logging
import math
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
        """Map a normalized command to a pulse width in microseconds (pure).

        Raises ``ValueError`` for NaN/inf: ``min``/``max`` clamping turns NaN into full
        scale, so a corrupted command would otherwise drive the lever to its end stop.
        """
        if not math.isfinite(value):
            raise ValueError(f"{self.name}: non-finite command {value!r} refused")
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


class RelayUnavailableError(RuntimeError):
    """A GPIO relay could not be driven (no GPIO library, or the pin is unavailable).

    Raised instead of pretending the relay switched: a blade-PTO relay that
    silently became a no-op would leave ``blade_engaged=False`` reported over a
    clutch that is still energised.
    """


class RelayActuator:
    """A GPIO relay (latching on/off, with a momentary pulse for the starter).

    SIM-safe: in SIM_MODE it only tracks state so the rest of the stack stays
    importable and testable. Outside SIM_MODE it drives the GPIO via
    ``gpiozero`` and **raises** :class:`RelayUnavailableError` when the pin cannot
    be claimed -- it does not fall back to simulation. ``state`` only changes
    after the write succeeded, so it never claims a switch that did not happen.
    """

    def __init__(self, name: str, gpio: int, active_high: bool = True, *, armed: bool = True):
        self.name = name
        self.gpio = gpio
        self.active_high = active_high
        #: False => the pin is never claimed or driven outside SIM_MODE (platform disabled).
        self.armed = armed
        self._on: bool = False
        self._device: Any = None
        self._sim = is_simulation_mode()
        self._fault: str | None = None

    @property
    def simulated(self) -> bool:
        return self._sim

    @property
    def fault(self) -> str | None:
        """Last GPIO fault (claim or write), cleared by the next successful write."""
        return self._fault

    def _ensure_device(self) -> Any:
        if self._sim or self._device is not None:
            return self._device
        if not self.armed:
            self._fault = (
                f"relay {self.name} GPIO{self.gpio} not armed: the tractor platform is not "
                "enabled (config/tractor.yaml), so the pin is not claimed or driven"
            )
            logger.error("%s", self._fault)
            raise RelayUnavailableError(self._fault)
        try:  # lazy hardware import — never at module load
            from gpiozero import OutputDevice  # type: ignore

            self._device = OutputDevice(
                self.gpio, active_high=self.active_high, initial_value=False
            )
        except Exception as exc:  # hardware/import error path
            self._fault = (
                f"relay {self.name} GPIO{self.gpio} unavailable: {type(exc).__name__}: {exc}"
            )
            logger.error("%s", self._fault)
            raise RelayUnavailableError(self._fault) from exc
        return self._device

    def set(self, on: bool) -> None:
        dev = self._ensure_device()
        if dev is not None:
            try:
                dev.on() if on else dev.off()
            except Exception as exc:
                self._fault = (
                    f"relay {self.name} GPIO{self.gpio} write failed: {type(exc).__name__}: {exc}"
                )
                logger.error("%s", self._fault)
                raise RelayUnavailableError(self._fault) from exc
        self._fault = None
        self._on = bool(on)

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
