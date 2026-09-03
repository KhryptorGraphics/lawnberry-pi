from __future__ import annotations

"""Software watchdog (T031).

Async loop monitors time since last heartbeat; triggers E-stop if exceeded.
"""

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402

from .estop_handler import EstopHandler  # noqa: E402

logger = logging.getLogger(__name__)


class Watchdog:
    def __init__(
        self,
        estop: EstopHandler,
        timeout_ms: int,
        on_timeout: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """
        on_timeout: optional additional async callback invoked on timeout,
        alongside (not instead of) ``estop.trigger_estop``. Lets a specific
        platform (e.g. the tractor) wire its own real emergency_stop() into
        a generic watchdog without that platform owning the estop/auth
        plumbing above -- ``estop`` stays required so every watchdog keeps
        the baseline auth-revoke behavior this class has always provided.
        A callback exception is logged, never allowed to crash the loop.
        """
        self._estop = estop
        self._timeout_ms = max(1, int(timeout_ms))
        self._on_timeout = on_timeout
        self._last_heartbeat = time.perf_counter()
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()

    def heartbeat(self) -> None:
        self._last_heartbeat = time.perf_counter()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_evt.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        try:
            while not self._stop_evt.is_set():
                await asyncio.sleep(0.01)
                elapsed_ms = (time.perf_counter() - self._last_heartbeat) * 1000.0
                if elapsed_ms > self._timeout_ms:
                    self._estop.trigger_estop("watchdog_timeout")
                    if self._on_timeout is not None:
                        try:
                            await self._on_timeout()
                        except Exception:
                            logger.exception("Watchdog on_timeout callback failed")
                    # After triggering, prevent repeated triggers and exit
                    break
        finally:
            pass
