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
from dataclasses import dataclass
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
    #: True for an adapter that drives real GPIO once it is set up (the lgpio adapter).
    real_hardware = False

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

    @property
    def on_pins(self) -> tuple[int, int]:
        """The (IN1, IN2) levels of the ON state; meaningful only for a real adapter."""
        return (1, 0)

    def read_pins(self) -> tuple[int, int] | None:
        """The pad levels read right now, or ``None``: a flag-only adapter cannot observe any."""
        return None

    def cleanup(self) -> None:
        self._initialized = False
        self._active = False


@dataclass(frozen=True, slots=True)
class PadObservation:
    """What the IBT-4 inputs read RIGHT NOW (never a cached command).

    ``state`` is ``off`` (both low), ``on`` (the ON levels), ``mixed`` (anything else, e.g. one
    pad forced by something else), ``unobservable`` (flag-only adapter, or the lines are not
    claimed) or ``error`` (a readback failed): only ``off`` is a statement that the blade
    inputs are low.
    """

    state: str
    pins: tuple[int, int] | None
    error: str | None = None


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
        # The last CONFIRMED state, and the state the last command ASKED for (they differ after
        # a failed write): the readback is compared with the latter.
        self._blade_active = False
        self._intended_on = False
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
    def drives_hardware(self) -> bool:
        """Is the adapter a real GPIO one (whether or not it has been set up yet)?

        An armed driver whose setup failed, or that was stopped, is still ``drives_hardware``:
        the blade inputs belong to the machine, so its state is UNKNOWN there, not "off".
        """
        return bool(getattr(self._gpio, "real_hardware", False))

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched

    @property
    def state_confirmed(self) -> bool:
        """Was the last blade command applied AND confirmed? False => the blade may be ON."""
        return self._state_confirmed

    def observe(self) -> PadObservation:
        """Read the blade inputs now (see ``PadObservation``); never raises, never writes."""
        if not (self.initialized and self.hardware_backed):
            return PadObservation("unobservable", None)
        try:
            pins = self._gpio.read_pins()
        except Exception as exc:
            return PadObservation("error", None, f"{type(exc).__name__}: {exc}")
        if pins is None:
            return PadObservation("unobservable", None)
        pins = (int(pins[0]), int(pins[1]))
        if pins == (0, 0):
            return PadObservation("off", pins)
        if pins == tuple(self._gpio.on_pins):
            return PadObservation("on", pins)
        return PadObservation("mixed", pins)

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
        """Drive the blade OFF, release the GPIO lines and stop. Never raises.

        After the lines are released the pad state is UNKNOWN (it may float or revert to a
        default pull, audit M-26), so a real adapter leaves ``state_confirmed`` False.
        """
        if self.initialized:
            self._apply(False)
        try:
            self._gpio.cleanup()
        except Exception as e:
            logger.error("IBT-4 GPIO cleanup failed: %s", e)
        if self.drives_hardware:
            self._state_confirmed = False
        self.running = False
        self.initialized = False

    def snapshot(self) -> dict[str, Any]:
        """The blade output as it reads NOW, not as it was last commanded (synchronous).

        With a real adapter the inputs are read back on every call (``observed``, ``pad``): a
        pad forced high by something else shows ``active`` True and ``state_confirmed`` False
        although no command was issued. ``active`` is conservative: it is False only when the
        inputs were READ low (or, for the flag-only stub, nothing was ever turned on); a failed
        readback, an uninitialised real adapter and an unconfirmed command all read as "may be
        on". ``state_confirmed`` is True only when what was observed equals what was commanded.
        Reads only, never writes, never raises: the sync callers (the safety status) use it
        directly; :meth:`health_check` is the same dict for ``await``.
        """
        obs = self.observe()
        if obs.state in ("off", "on", "mixed"):
            active = obs.state != "off"
            state_confirmed = obs.state == ("on" if self._intended_on else "off")
        elif self.drives_hardware:
            # Lines unclaimed (setup failed, never initialised, released) or unreadable: unknown.
            active, state_confirmed = True, False
        else:
            # Flag-only stub: there is nothing to read, only the last command to report.
            active = self._blade_active or not self._state_confirmed
            state_confirmed = self._state_confirmed
        return {
            "driver": "ibt4",
            "initialized": self.initialized,
            "running": self.running,
            "active": active,
            "pins": {"in1": self.in1, "in2": self.in2},
            "estop_latched": self._estop_latched,
            "hardware_backed": self.hardware_backed,
            "gpio_enabled": self.gpio_enabled,
            "state_confirmed": state_confirmed,
            "observed": obs.state,
            "pad": None if obs.pins is None else {"in1": obs.pins[0], "in2": obs.pins[1]},
            "observe_error": obs.error,
            "last_error": self.last_error,
        }

    async def health_check(self) -> dict[str, Any]:
        """The blade output as it reads NOW: :meth:`snapshot`, for callers that ``await``."""
        return self.snapshot()

    async def set_estop(self, active: bool) -> bool:
        """Set/clear the E-stop latch; setting it also drives the blade OFF.

        Latching returns the result of the OFF write (True only if it was confirmed); the latch
        is set whether or not the write works. Releasing returns True only when the latch was
        released, and that needs a CONFIRMED OFF: the inputs are driven low once more and read
        back, so a clear can never reopen the blade path over an output that was never seen
        off (audit BL-04). An armed driver whose lines are not claimed cannot confirm anything,
        so its latch stays (``last_error`` says why); the service claims the lines first.
        """
        if active:
            self.latch_estop()
            return await self.set_active(False)
        if self.initialized:
            if not self._apply(False):
                return False
        elif self.drives_hardware:
            self.last_error = (
                "IBT-4 lines are not claimed: the blade OFF cannot be confirmed, E-stop latch kept"
            )
            return False
        self._estop_latched = False
        return True

    def latch_estop(self) -> None:
        """Set the E-stop latch (synchronous, no hardware access): ON is refused until cleared."""
        self._estop_latched = True

    async def set_active(self, active: bool) -> bool:
        """Engage/disengage the blade (see :meth:`drive`)."""
        return self.drive(active)

    def drive(self, active: bool) -> bool:
        """Engage/disengage the blade; synchronous, so nothing can interleave with it.

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
        self._intended_on = active
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
