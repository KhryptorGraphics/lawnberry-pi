"""Real ``lgpio`` adapter for the IBT-4 blade driver (IN1 = GPIO24, IN2 = GPIO25 by default).

STATUS: NEVER RUN ON HARDWARE. It is exercised only against a fake ``lgpio``
(``tests/unit/test_ibt4_lgpio.py``). It is armed solely by ``ibt4.gpio_enabled: true`` in
``config/hardware.yaml`` (shipped ``false``) and never in SIM_MODE; the gate lives in
``IBT4BladeDriver.__init__``. Nothing in this module touches hardware or imports ``lgpio`` at
import time or in ``__init__``: ``lgpio`` is imported inside :meth:`LgpioBladeAdapter.setup`
only (or injected through ``lgpio_module`` by tests).

What it does
------------
* ``setup()`` opens the GPIO chip, claims IN1 then IN2 as outputs with initial level 0 (the
  line is requested already low, so it never glitches high) and reads both back; both must read
  0. All or nothing: on any failure everything claimed is freed, the chip is closed and
  ``IBT4GpioError`` names the pin and the cause. Expected on the deployed Pi today: GPIO25 is
  held by the kernel (HaLow HAT SPI IRQ, audit C-03, owner action O-5), so that claim fails.
* ``set_active(False)`` drives BOTH inputs to 0; ``set_active(True)`` drives them to
  ``on_levels``. Both writes are always attempted, so a failed first write cannot stop the
  second from safing the output. Each pin is then read back. The change is CONFIRMED only if
  every write returned success (lgpio returns 0, a negative code is an error, and
  ``lgpio.error`` is raised when lgpio exceptions are on) AND every readback equals the
  requested level; otherwise ``IBT4GpioError`` is raised. A partially applied ON is rolled back
  (both inputs driven OFF) before the error is raised.
* ``is_active()`` is True only for the last CONFIRMED-ON state. ``state_confirmed`` is False
  whenever the output state is unknown (failed write or readback, failed rollback), so a caller
  can tell "unknown" from "off". ``hardware_backed`` is True only after a fully successful
  ``setup()`` and until ``cleanup()``; a failed ``setup()`` is the only fatal fault modelled (a
  failed write must not make the next confirmed write look like the flag-only stub).
  ``last_error`` holds the last failure text and is cleared by the next confirmed success.
* ``cleanup()`` is best effort and never raises: drive both inputs low, free both lines, close
  the chip.

What "confirmed" does NOT mean
------------------------------
A confirmed write means GPIO24/25 were written and read back at the commanded level. It does
NOT observe the IBT-4 power stage, the motor current or whether the blade stopped: there is no
current, speed or PTO feedback anywhere in this software (audit D.2 requires one).

UNVERIFIED / TBD (nothing below was measured; each item names the measurement needed)
-------------------------------------------------------------------------------------
1. IBT-4 truth table. The ``on_levels`` default ``(1, 0)`` is a PLACEHOLDER. Needed: the fitted
   board's markings or datasheet (are IN1/IN2 direction inputs or per-half-bridge inputs, are
   there enable pins), then a bench check with the blade motor DISCONNECTED: a meter on the
   driver's output terminals for each (IN1, IN2) combination.
2. PWM or enable line. Only static levels are driven: no PWM, no soft start, no enable pin.
   Whether the 997 DC blade motor needs PWM or a separate enable line is unknown; an extra pin
   would need the owner's approval (O-5).
3. Pin state after ``gpio_free``, process exit and SIGKILL (the lines may float or revert to an
   input with a default pull). Needed: the audit M-26 measurement ("relay pin state at process
   exit and at SIGKILL": scope, backend stopped and then killed) repeated on GPIO24/25 with the
   IBT-4 unpowered, plus whether the IBT-4 inputs have pull-downs. Until then nothing here keeps
   the output off once the process is gone; ``cleanup()`` only drives low just before releasing.
4. That a readback of an OUTPUT pin on the Pi 5's RP1 reflects the PAD level (so a pin held
   high by something else, or shorted, reads as a mismatch) is ASSUMED, not verified. Two
   other outcomes are possible and both are bad: if lgpio returns the last written value, the
   readback proves nothing about the pad and "confirmed" only means "the write returned 0"; if
   the input path of an output pad is disabled and reads a constant 0, an ON write can never be
   confirmed (the blade never turns on, fail-safe) while an OFF "confirmation" is vacuous.
   Needed: on a spare header pin with the IBT-4 disconnected, claim it as an output and read it
   after writing 1 and after writing 0 (expect 1 then 0), then hold it at the opposite level
   through a resistor and check that ``gpio_read`` follows the pad.
5. Readback settle time. The readback happens immediately after the write; whether the pad
   needs time to settle is unmeasured. A too-early readback shows as a mismatch, i.e. as
   "unconfirmed" (fail-safe), never as stopped.
6. The lgpio conventions used here (0 = success, negative = error, ``lgpio.error`` raised when
   exceptions are on, positional ``gpio_claim_output(h, pin, level)``, ``gpio_write``,
   ``gpio_read``, ``gpio_free`` and ``gpiochip_close``) are taken from the lgpio documentation.
   lgpio is not installed in the development environment, so none of it has run against the
   real library.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..base import open_gpiochip
from .ibt4_gpio import _GPIOAdapter

logger = logging.getLogger(__name__)

_OFF = (0, 0)


class IBT4GpioError(RuntimeError):
    """An IBT-4 GPIO line could not be claimed, driven or confirmed."""


def _is_error(rc: Any) -> bool:
    """lgpio convention: a non-negative integer is success, a negative one is an error code."""
    return not isinstance(rc, int) or rc < 0


class LgpioBladeAdapter(_GPIOAdapter):
    """Drives the IBT-4 inputs through ``lgpio`` and confirms every change by reading it back.

    Contract with ``IBT4BladeDriver``: ``setup()`` raises unless both lines are claimed and read
    low; ``set_active()`` returns only when the new state is CONFIRMED and raises
    ``IBT4GpioError`` otherwise.
    """

    def __init__(
        self,
        in1: int,
        in2: int,
        on_levels: tuple[int, int] = (1, 0),
        *,
        lgpio_module: Any = None,
        open_chip: Callable[[Any], int] = open_gpiochip,
    ) -> None:
        super().__init__(in1, in2)
        # Stored as given and validated in setup(): construction never raises or touches hardware.
        self.on_levels = on_levels
        self._on: tuple[int, int] = _OFF
        self._lgpio = lgpio_module
        self._open_chip = open_chip
        self._handle: int | None = None
        self._claimed: list[int] = []
        self.hardware_backed = False
        self.state_confirmed = False
        self.last_error: str | None = None

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        """Claim IN1 then IN2 as outputs at level 0 and read both back; all or nothing.

        Raises ``IBT4GpioError`` (nothing left claimed, chip closed) on any failure. A no-op
        while already set up.
        """
        if self.hardware_backed:
            return
        try:
            self._claim()
        except Exception as exc:
            error = exc
            if not isinstance(exc, IBT4GpioError):
                error = IBT4GpioError(
                    f"unexpected error during IBT-4 GPIO setup: {type(exc).__name__}: {exc}"
                )
            for problem in self._release(drive_low=False):
                logger.warning("IBT-4 setup abort: %s", problem)
            self.hardware_backed = False
            self.state_confirmed = False
            self._active = False
            self._initialized = False
            self.last_error = str(error)
            logger.error("IBT-4 lgpio setup failed, nothing left claimed: %s", error)
            if error is exc:
                raise
            raise error from exc

    def _lgpio_module(self) -> Any:
        if self._lgpio is None:
            try:
                import lgpio  # lazy: never at module import or in __init__
            except ImportError as exc:
                raise IBT4GpioError(f"the lgpio module cannot be imported: {exc}") from exc
            self._lgpio = lgpio
        return self._lgpio

    def _checked_config(self) -> tuple[int, int, tuple[int, int]]:
        pins = (self.in1, self.in2)
        if (
            any(isinstance(pin, bool) or not isinstance(pin, int) or pin < 0 for pin in pins)
            or self.in1 == self.in2
        ):
            raise IBT4GpioError(
                f"invalid IBT-4 pins in1={self.in1!r}, in2={self.in2!r}: "
                "need two different non-negative GPIO numbers"
            )
        try:
            levels = tuple(self.on_levels)
        except TypeError:
            levels = ()
        if len(levels) != 2 or any(lvl not in (0, 1) for lvl in levels):
            raise IBT4GpioError(f"invalid IBT-4 on_levels {self.on_levels!r}: need two of 0/1")
        on = (int(levels[0]), int(levels[1]))
        if on == _OFF:
            raise IBT4GpioError(
                f"invalid IBT-4 on_levels {self.on_levels!r}: identical to OFF (ON would "
                "never differ)"
            )
        return self.in1, self.in2, on

    def _claim(self) -> None:
        in1, in2, on = self._checked_config()
        lgpio = self._lgpio_module()
        try:
            handle = self._open_chip(lgpio)
        except Exception as exc:
            raise IBT4GpioError(
                f"the GPIO chip could not be opened: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(handle, int) or handle < 0:
            raise IBT4GpioError(f"the GPIO chip could not be opened: lgpio returned {handle!r}")
        self._handle = handle

        for label, pin in (("IN1", in1), ("IN2", in2)):
            try:
                rc = lgpio.gpio_claim_output(handle, pin, 0)
            except Exception as exc:
                raise IBT4GpioError(
                    f"GPIO{pin} (IBT-4 {label}) could not be claimed as an output: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if _is_error(rc):
                raise IBT4GpioError(
                    f"GPIO{pin} (IBT-4 {label}) could not be claimed as an output: "
                    f"lgpio returned {rc!r}"
                )
            self._claimed.append(pin)

        for label, pin in (("IN1", in1), ("IN2", in2)):
            try:
                level = lgpio.gpio_read(handle, pin)
            except Exception as exc:
                raise IBT4GpioError(
                    f"GPIO{pin} (IBT-4 {label}) could not be read back after being claimed low: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if level != 0:
                raise IBT4GpioError(
                    f"GPIO{pin} (IBT-4 {label}) reads {level!r} after being claimed low "
                    "(held by something else?), expected 0"
                )

        self._on = on
        self._active = False
        self._initialized = True
        self.state_confirmed = True
        self.last_error = None
        self.hardware_backed = True
        logger.info(
            "IBT-4 lgpio adapter armed: GPIO%d/GPIO%d claimed as outputs and read back low "
            "(never run on hardware; electrical behaviour UNVERIFIED)",
            in1,
            in2,
        )

    # ------------------------------------------------------------------ drive

    def set_active(self, active: bool) -> None:
        """Drive OFF (both low) or ``on_levels``; return only when the change is CONFIRMED."""
        if not self.hardware_backed:
            raise IBT4GpioError(
                "IBT-4 GPIO adapter is not set up (setup() failed, was never called, or "
                "cleanup() ran): nothing was written"
            )
        want_on = bool(active)
        problems = self._drive(self._on if want_on else _OFF)
        if not problems:
            self._active = want_on
            self.state_confirmed = True
            self.last_error = None
            return

        self.state_confirmed = False
        failure = "; ".join(problems)
        if not want_on:
            message = f"IBT-4 OFF not confirmed ({failure}): the blade inputs may still be driven"
        else:
            # A half-applied ON must never be left standing: force both inputs OFF.
            rollback = self._drive(_OFF)
            if rollback:
                message = (
                    f"IBT-4 ON not confirmed ({failure}) and the rollback to OFF also failed "
                    f"({'; '.join(rollback)}): the output state is UNKNOWN"
                )
            else:
                self._active = False
                self.state_confirmed = True
                message = (
                    f"IBT-4 ON not confirmed ({failure}); rolled back, both inputs driven low "
                    "and read back low"
                )
        self.last_error = message
        logger.error("%s", message)
        raise IBT4GpioError(message)

    def _drive(self, levels: tuple[int, int]) -> list[str]:
        """Write both inputs (always both), read both back; return what did not check out."""
        lgpio, handle = self._lgpio, self._handle
        pins = (self.in1, self.in2)
        problems: list[str] = []
        for pin, level in zip(pins, levels, strict=True):
            try:
                rc = lgpio.gpio_write(handle, pin, level)
            except Exception as exc:
                problems.append(f"GPIO{pin} write({level}) raised {type(exc).__name__}: {exc}")
                continue
            if _is_error(rc):
                problems.append(f"GPIO{pin} write({level}) returned error code {rc!r}")
        for pin, level in zip(pins, levels, strict=True):
            try:
                got = lgpio.gpio_read(handle, pin)
            except Exception as exc:
                problems.append(f"GPIO{pin} readback raised {type(exc).__name__}: {exc}")
                continue
            if got != level:
                problems.append(f"GPIO{pin} reads back {got!r}, expected {level}")
        return problems

    def is_active(self) -> bool:
        """True only for the last CONFIRMED-ON state (see ``state_confirmed`` for 'unknown')."""
        return self._active

    # ---------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        """Best effort, never raises: drive both inputs low, free both lines, close the chip."""
        self.hardware_backed = False
        self.state_confirmed = False
        self._active = False
        self._initialized = False
        problems = self._release(drive_low=True)
        if problems:
            self.last_error = "cleanup: " + "; ".join(problems)
            logger.warning("IBT-4 %s", self.last_error)

    def _release(self, *, drive_low: bool) -> list[str]:
        """Free every claimed line and close the chip, each step best effort; never raises."""
        lgpio, handle, claimed = self._lgpio, self._handle, self._claimed
        self._handle, self._claimed = None, []
        problems: list[str] = []
        if handle is None or lgpio is None:
            return problems
        if drive_low:
            for pin in claimed:
                try:
                    rc = lgpio.gpio_write(handle, pin, 0)
                except Exception as exc:
                    problems.append(f"GPIO{pin} write(0) raised {type(exc).__name__}: {exc}")
                    continue
                if _is_error(rc):
                    problems.append(f"GPIO{pin} write(0) returned error code {rc!r}")
        for pin in reversed(claimed):
            try:
                lgpio.gpio_free(handle, pin)
            except Exception as exc:
                problems.append(f"GPIO{pin} free raised {type(exc).__name__}: {exc}")
        try:
            lgpio.gpiochip_close(handle)
        except Exception as exc:
            problems.append(f"gpiochip_close raised {type(exc).__name__}: {exc}")
        return problems


__all__ = ["IBT4GpioError", "LgpioBladeAdapter"]
