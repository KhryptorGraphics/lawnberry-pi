"""
Blade service for IBT-4 control.

Provides a simple async wrapper around IBT4BladeDriver with a process-wide
singleton accessor. Safe in SIM mode; hardware access is lazy and guarded.

The singleton is configured from the ``ibt4:`` block of ``config/hardware.yaml``. Its
``gpio_enabled`` flag is the arming gate for the real lgpio adapter and ships ``false``; a
missing or unreadable block never arms anything (the driver then uses the flag-only stub).

An ARMED service (``gpio_enabled``) that was never initialised, e.g. right after a backend
restart when the lines are in an unknown state, is initialised by ``emergency_stop()`` and
``park()`` themselves: they claim the lines at level 0, drive both low and read them back, so
the stop is confirmed or reported as failed, never "nothing to stop". An unarmed service is
left untouched.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..drivers.blade.ibt4_gpio import IBT4BladeDriver

logger = logging.getLogger(__name__)


class BladeService:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._driver = IBT4BladeDriver(config or {})
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        async with self._lock:
            if self._initialized:
                return True
            try:
                await self._driver.initialize()
                if not self._driver.initialized:
                    # Fail closed: the driver could not claim its outputs. Stay uninitialised so
                    # every command is refused and the next call retries the setup.
                    logger.error(
                        "BladeService: IBT-4 driver did not initialise (%s); blade stays disabled",
                        self._driver.last_error,
                    )
                    return False
                await self._driver.start()
                self._initialized = True
                return True
            except Exception as e:
                logger.warning("BladeService init failed: %s", e)
                return False

    async def set_active(self, active: bool) -> bool:
        if not self._initialized:
            await self.initialize()
        try:
            return await self._driver.set_active(bool(active))
        except Exception as e:
            logger.error("BladeService set_active failed: %s", e)
            return False

    @property
    def emergency_latched(self) -> bool:
        """Is the blade E-stop latch set (blade enable refused until cleared)?"""
        return self._driver.estop_latched

    def _stop_report(self, *, latched: bool, off: bool, error: str | None) -> dict[str, Any]:
        return {
            "latched": latched,
            # True only when the LAST off write was confirmed (every write succeeded and both
            # inputs read back low); an unconfirmed write is False, never "probably off".
            "off": off,
            "initialized": bool(self._driver.initialized),
            # False => the driver only tracks a flag (IBT-4 GPIO adapter is a stub):
            # "off" then has no hardware effect and must not be reported as a stop.
            "hardware_backed": self._driver.hardware_backed,
            # False => the last command was not confirmed and the blade may still be on.
            "state_confirmed": self._driver.state_confirmed,
            # The driver's last failure text (setup refused, write/readback mismatch, ...).
            "gpio_error": self._driver.last_error,
            "pins": {"in1": self._driver.in1, "in2": self._driver.in2},
            "error": error,
        }

    @property
    def _armed_but_uninitialised(self) -> bool:
        """Armed (``ibt4.gpio_enabled``) yet never initialised, e.g. right after a restart."""
        return self._driver.gpio_enabled and not self._initialized

    async def _claim_lines_for_stop(self) -> str | None:
        """Initialise an armed, never-initialised service so the stop can drive its lines.

        Nothing in a restarted process has claimed the IBT-4 inputs, so their state is unknown
        and "never initialised" must not read as "nothing to stop": claiming them (level 0)
        lets the stop drive both low and read them back. A failure leaves the driver
        uninitialised with ``gpio_error`` set, which the E-stop reports as ``failed``.
        Returns an error text only if initialising itself raised. Never raises.
        """
        try:
            await self.initialize()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("BladeService: claiming the IBT-4 lines for the stop failed: %s", error)
            return f"claiming the IBT-4 lines for the stop failed: {error}"
        return None

    async def emergency_stop(self) -> dict[str, Any]:
        """Latch the blade E-stop and switch the blade off; report what happened.

        An ARMED service (``ibt4.gpio_enabled``) that was never initialised, for example
        right after a backend restart, is initialised first: the stop claims the lines at
        level 0, drives both low and reads them back, so the report carries a confirmed
        ``off`` or a ``gpio_error``, never "nothing to stop". An unarmed service is left
        exactly as it is (never initialised stays never initialised, no GPIO is touched).

        The latch is set even if the lines could not be claimed, so a later initialise +
        enable is refused until ``clear_emergency()``. ``off`` is ``True`` only when the
        last off write was confirmed by the driver (written and read back); the write is
        retried once if the first attempt was not confirmed. Never raises.
        """
        error: str | None = None
        off = False
        if self._armed_but_uninitialised:
            error = await self._claim_lines_for_stop()
        try:
            # Latches first, then drives the blade off: the latch holds even if the write fails.
            off = bool(await self._driver.set_estop(True))
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            logger.error("BladeService E-stop latch failed: %s", exc)
        if not off:
            try:
                off = bool(await self._driver.set_active(False))
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"
                logger.error("BladeService E-stop blade-off failed: %s", exc)
        return self._stop_report(latched=self._driver.estop_latched, off=off, error=error)

    async def clear_emergency(self) -> None:
        """Release the blade E-stop latch (blade stays off until re-enabled)."""
        await self._driver.set_estop(False)

    async def park(self) -> dict[str, Any]:
        """Switch the blade off without latching an E-stop (shutdown). Never raises.

        Like ``emergency_stop()``, an armed service that was never initialised claims its
        lines first so the off write can be driven and confirmed (see there).
        """
        error: str | None = None
        off = False
        if self._armed_but_uninitialised:
            error = await self._claim_lines_for_stop()
        try:
            off = bool(await self._driver.set_active(False))
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            error = error or text
            logger.error("BladeService park failed: %s", text)
        return self._stop_report(latched=False, off=off, error=error)


def _load_ibt4_config() -> dict[str, Any]:
    """The ``ibt4:`` block of hardware.yaml as a plain dict.

    ``{}`` (=> the flag-only stub: fail closed, never armed) on ANY problem or when the block
    is absent, with a warning: a config problem must never arm the real GPIO adapter.
    """
    try:
        from ..core.config_loader import ConfigLoader

        loader = ConfigLoader(config_dir=os.environ.get("LAWNBERRY_CONFIG_DIR") or None)
        ibt4 = loader.get()[0].ibt4_config
        if ibt4 is not None:
            return ibt4.model_dump()
        logger.warning("hardware.yaml has no ibt4 block; IBT-4 GPIO NOT armed (flag-only stub)")
    except Exception as exc:
        logger.warning(
            "ibt4 config unreadable (%s: %s); IBT-4 GPIO NOT armed (flag-only stub)",
            type(exc).__name__,
            exc,
        )
    return {}


_blade_service: BladeService | None = None


def get_blade_service() -> BladeService:
    global _blade_service
    if _blade_service is None:
        _blade_service = BladeService(_load_ibt4_config())
    return _blade_service
