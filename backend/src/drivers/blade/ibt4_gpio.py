"""
IBT-4 blade controller using Raspberry Pi GPIO 24/25.

Design goals:
- Safe defaults: blade off unless explicitly enabled
- SIM_MODE support: no GPIO access, simulate state
- Pi 4B/5 compatibility: read pin mapping from spec if provided, default to 24/25
- Respect emergency stop by exposing a local latch to be set by safety layer
- Fail closed: an adapter that cannot be set up leaves the driver uninitialised, and a write the
  adapter could not confirm is reported as such (the output is then treated as possibly ON)

Which adapter drives the pins (the arming gate, decided in ``IBT4BladeDriver.__init__``):
- an explicitly injected ``gpio_adapter`` always wins (tests);
- SIM_MODE, or ``gpio_enabled`` not exactly ``True`` in the config: the flag-only
  ``_GPIOAdapter`` below (no GPIO library, no hardware effect). This is what ships, because
  ``ibt4.gpio_enabled`` is ``false`` in ``config/hardware.yaml``;
- otherwise ``LgpioBladeAdapter`` (``ibt4_lgpio.py``), which has never run on hardware.

Importing this module, or constructing the driver, never imports ``lgpio``.
This module stays side-effect free for tests; hardware access is encapsulated behind the tiny
adapter interface that can be stubbed.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.simulation import is_simulation_mode
from ..base import HardwareDriver

logger = logging.getLogger(__name__)


class _GPIOAdapter:
    """Flag-only GPIO adapter: the unarmed default and the base of the real adapters.

    In SIM_MODE or when adapter is not available, methods are no-ops.

    ``hardware_backed`` is ``False`` here: this stub only tracks flags (no GPIO
    library is imported), so toggling it has no effect on the IBT-4. A real
    adapter must set it ``True``; callers that report a stop (the E-stop
    orchestrator) use it to avoid claiming a hardware outcome the stub cannot
    produce.

    Contract used by ``IBT4BladeDriver``: ``setup()`` raises when the lines cannot
    be claimed (the driver then stays uninitialised); ``set_active()`` returns
    normally only when the new output state was applied AND confirmed, and raises
    otherwise (the driver then reports the state as unconfirmed).
    """

    hardware_backed = False

    def __init__(self, in1: int, in2: int):
        self.in1 = in1
        self.in2 = in2
        self._active = False
        self._initialized = False

    def setup(self) -> None:
        # Defer selecting a library until we run on hardware; tests will skip
        self._initialized = True
        self._active = False

    def set_active(self, active: bool) -> None:
        self._active = bool(active)

    def is_active(self) -> bool:
        return self._active

    def cleanup(self) -> None:
        self._initialized = False
        self._active = False


class IBT4BladeDriver(HardwareDriver):
    """IBT-4 blade controller driver.

    Config keys:
    - pins: {"in1": 24, "in2": 25}
    - gpio_enabled: the arming gate; only ``True`` selects the real lgpio adapter (and never in
      SIM_MODE). Absent or false: the flag-only stub.
    - on_levels: {"in1": 1, "in2": 0} level driven on each input when the blade is ON
      (UNVERIFIED placeholder; OFF always drives both low)
    - name: optional identifier
    """

    def __init__(
        self, config: dict[str, Any] | None = None, gpio_adapter: _GPIOAdapter | None = None
    ):
        super().__init__(config)
        pins = self.config.get("pins") or {}
        self.in1 = int(pins.get("in1", 24))
        self.in2 = int(pins.get("in2", 25))
        self.gpio_enabled = self.config.get("gpio_enabled") is True
        # Last failure text, cleared by the next confirmed success (health_check()/E-stop report).
        self.last_error: str | None = None
        self._blade_active = False
        self._estop_latched = False
        # False once a command could not be confirmed: the output state is then UNKNOWN.
        self._state_confirmed = True
        self._gpio = gpio_adapter if gpio_adapter is not None else self._select_adapter()

    def _select_adapter(self) -> _GPIOAdapter:
        """The arming gate: the real adapter only when armed outside SIM_MODE, else the stub."""
        if is_simulation_mode() or not self.gpio_enabled:
            return _GPIOAdapter(self.in1, self.in2)
        from .ibt4_lgpio import LgpioBladeAdapter  # lazy: keeps the stub path import-free

        levels = self.config.get("on_levels")
        if levels is None:
            levels = (1, 0)  # the documented UNVERIFIED placeholder
        elif isinstance(levels, dict):
            levels = (levels.get("in1", 1), levels.get("in2", 0))
        # anything else is passed through: the adapter rejects it in setup() (fail closed)
        return LgpioBladeAdapter(self.in1, self.in2, levels)

    @property
    def hardware_backed(self) -> bool:
        """Does ``set_active`` drive real GPIO (vs. only tracking a flag)?"""
        return bool(getattr(self._gpio, "hardware_backed", False))

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched

    @property
    def state_confirmed(self) -> bool:
        """Was the last blade command applied AND confirmed? False => the blade may be ON."""
        return self._state_confirmed

    async def initialize(self) -> None:
        if self.initialized:
            return
        if is_simulation_mode():
            logger.info("IBT-4 blade driver in SIM_MODE; GPIO disabled")
            self.initialized = True
            return
        try:
            self._gpio.setup()
        except Exception as e:
            # Fail closed: never pretend to be initialised, every command is refused.
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error(
                "IBT-4 GPIO setup failed; the blade driver stays UNINITIALISED and refuses "
                "every command: %s",
                self.last_error,
            )
            return
        self.last_error = None
        self.initialized = True

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        """Drive the blade OFF, release the GPIO lines and stop. Never raises."""
        if self.initialized:
            self._apply(False)
        try:
            self._gpio.cleanup()
        except Exception as e:
            logger.error("IBT-4 GPIO cleanup failed: %s", e)
        self.running = False
        self.initialized = False

    async def health_check(self) -> dict[str, Any]:
        return {
            "driver": "ibt4",
            "initialized": self.initialized,
            "running": self.running,
            # Conservative: after a command that was not confirmed the blade MAY be on.
            "active": self._blade_active or not self._state_confirmed,
            "pins": {"in1": self.in1, "in2": self.in2},
            "estop_latched": self._estop_latched,
            "hardware_backed": self.hardware_backed,
            "gpio_enabled": self.gpio_enabled,
            "state_confirmed": self._state_confirmed,
            "last_error": self.last_error,
        }

    async def set_estop(self, active: bool) -> bool:
        """Set/clear the E-stop latch; setting it also drives the blade OFF.

        Returns the result of the OFF write when latching (True only if it was confirmed),
        and True when releasing the latch. The latch is set whether or not the write works.
        """
        self._estop_latched = bool(active)
        if active:
            return await self.set_active(False)
        return True

    async def set_active(self, active: bool) -> bool:
        """Engage/disengage the blade.

        Returns True only when the adapter confirmed the new state; respects the E-stop latch.
        On any failure the previous state is kept, ``state_confirmed`` turns False and
        ``health_check()["active"]`` reports the blade as possibly on.
        """
        if not self.initialized:
            return False
        if self._estop_latched and active:
            logger.warning("Blade engagement blocked by E-stop latch")
            return False
        return self._apply(bool(active))

    def _apply(self, active: bool) -> bool:
        """Write ``active`` through the adapter and record whether it was confirmed."""
        try:
            self._gpio.set_active(active)
        except Exception as e:
            self._state_confirmed = False
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error(
                "Blade %s NOT confirmed (the output may still be on): %s",
                "ON" if active else "OFF",
                self.last_error,
            )
            return False
        self._blade_active = active
        self._state_confirmed = True
        self.last_error = None
        return True
