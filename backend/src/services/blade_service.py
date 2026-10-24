"""
Blade service for IBT-4 control.

Provides a simple async wrapper around IBT4BladeDriver with a process-wide
singleton accessor. Safe in SIM mode; hardware access is lazy and guarded.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..drivers.blade.ibt4_gpio import IBT4BladeDriver

logger = logging.getLogger(__name__)


class BladeService:
    def __init__(self) -> None:
        self._driver = IBT4BladeDriver({})
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        async with self._lock:
            if self._initialized:
                return True
            try:
                await self._driver.initialize()
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
            "off": off,
            "initialized": bool(self._driver.initialized),
            # False => the driver only tracks a flag (IBT-4 GPIO adapter is a stub):
            # "off" then has no hardware effect and must not be reported as a stop.
            "hardware_backed": self._driver.hardware_backed,
            "error": error,
        }

    async def emergency_stop(self) -> dict[str, Any]:
        """Latch the blade E-stop and switch the blade off; report what happened.

        The latch is set even if the service was never initialised, so a later
        initialise + enable is refused until ``clear_emergency()``. ``off`` is
        ``True`` only when the driver confirmed it switched the output. Never
        raises.
        """
        error: str | None = None
        latched = False
        off = False
        try:
            await self._driver.set_estop(True)
            latched = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("BladeService E-stop latch failed: %s", error)
        try:
            off = bool(await self._driver.set_active(False))
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            logger.error("BladeService E-stop blade-off failed: %s", exc)
        return self._stop_report(latched=latched, off=off, error=error)

    async def clear_emergency(self) -> None:
        """Release the blade E-stop latch (blade stays off until re-enabled)."""
        await self._driver.set_estop(False)

    async def park(self) -> dict[str, Any]:
        """Switch the blade off without latching an E-stop (shutdown). Never raises."""
        error: str | None = None
        off = False
        try:
            off = bool(await self._driver.set_active(False))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("BladeService park failed: %s", error)
        return self._stop_report(latched=False, off=off, error=error)


_blade_service: BladeService | None = None


def get_blade_service() -> BladeService:
    global _blade_service
    if _blade_service is None:
        _blade_service = BladeService()
    return _blade_service
