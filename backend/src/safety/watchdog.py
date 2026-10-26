"""Software watchdog (T031; audit N-04, WP-P1-08, D.4).

A watchdog trips when a heartbeat it *requires* goes silent. Each required **source** is a
separate heartbeat, and every one of them must be fresh: a watchdog fed by a single loop that
only proves "the event loop turns" says nothing about whether the machine can still be stopped.
The sources are fed by the code that does the work, after it has done it:

* ``ACTUATION_SOURCE``: ``TractorControlService.supervise_once`` -- after the servo outputs
  were read back from the PCA9685 and matched what was commanded, no write fault is
  outstanding, the command deadman is running, and a latched E-stop holds the safe state.
* ``SAFETY_MONITOR_SOURCE``: ``TractorSafetyMonitor`` -- after a *completed* tilt cycle.

On a trip the watchdog revokes authorization (``estop.trigger_estop``) and awaits
``on_timeout`` (the tractor's real ``emergency_stop``: blade off, levers neutral, throttle
idle). It never exits its loop: while a source stays silent it repeats the stop every
``retrip_interval_s`` (a bounded rate, so a failed neutral write is retried until it is
confirmed instead of being tried once), and it re-arms by itself once every source is fresh
again. The E-stop latch it set stays until an operator clear.

**A hardware watchdog is still required (audit O-7, D.4).** This one runs in the event loop it
supervises: if that loop is blocked, or the process or kernel hangs, it cannot fire, and the
PCA9685 keeps emitting its last pulse. Only a watchdog that is independent of the Pi and
opens the servo power and the PTO supply covers those cases.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .estop_handler import EstopHandler

logger = logging.getLogger(__name__)

#: The source of a watchdog built without ``required_sources`` (``heartbeat()`` with no name).
DEFAULT_SOURCE = "default"
#: Fed by the tractor service's actuation self-check (``TractorControlService.supervise_once``).
ACTUATION_SOURCE = "actuation"
#: Fed by ``TractorSafetyMonitor`` after each completed tilt cycle.
SAFETY_MONITOR_SOURCE = "safety_monitor"

#: A trip that repeats is logged at most this often (the stop itself repeats every
#: ``retrip_interval_s``).
_REPEAT_LOG_INTERVAL_S = 30.0
#: How long ``stop()`` waits for the loop (an in-flight ``on_timeout``) before cancelling it.
_STOP_GRACE_S = 1.0


class Watchdog:
    def __init__(
        self,
        estop: EstopHandler,
        timeout_ms: int,
        on_timeout: Callable[[], Awaitable[Any]] | None = None,
        *,
        required_sources: Iterable[str] = (DEFAULT_SOURCE,),
        retrip_interval_s: float = 1.0,
        poll_interval_s: float = 0.01,
        on_timeout_deadline_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        required_sources: every name must call :meth:`heartbeat` within ``timeout_ms`` or the
        watchdog trips. A name that is not listed here cannot be fed (``KeyError``): a typo
        must not silently leave the real source starving.

        on_timeout: optional async callback awaited on every trip, alongside (not instead of)
        ``estop.trigger_estop`` -- the tractor wires its real ``emergency_stop`` here. It is
        abandoned after ``on_timeout_deadline_s`` (the stop is then reported unconfirmed and
        retried), and an exception is logged; neither may stop the loop.

        retrip_interval_s: while a source stays silent the stop is repeated this often.
        """
        sources = tuple(dict.fromkeys(required_sources))
        if not sources:
            raise ValueError("a watchdog needs at least one required heartbeat source")
        if retrip_interval_s <= 0 or poll_interval_s <= 0 or on_timeout_deadline_s <= 0:
            raise ValueError("retrip, poll and on_timeout intervals must be positive")
        self._estop = estop
        self._timeout_s = max(1, int(timeout_ms)) / 1000.0
        self._on_timeout = on_timeout
        self._retrip_interval_s = float(retrip_interval_s)
        self._poll_interval_s = float(poll_interval_s)
        self._on_timeout_deadline_s = float(on_timeout_deadline_s)
        self._clock = clock
        self._last_beat: dict[str, float] = dict.fromkeys(sources, clock())
        self._tripped = False
        self._trips = 0
        self._next_trip_at = 0.0
        self._last_reason: str | None = None
        self._last_log_at: float | None = None
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()

    # ------------------------------------------------------------------ feeding

    @property
    def required_sources(self) -> tuple[str, ...]:
        return tuple(self._last_beat)

    def heartbeat(self, source: str = DEFAULT_SOURCE) -> None:
        """Record that ``source`` just did its job. Call only after the work succeeded."""
        if source not in self._last_beat:
            raise KeyError(
                f"unknown watchdog heartbeat source {source!r}; required: {list(self._last_beat)}"
            )
        self._last_beat[source] = self._clock()

    # ---------------------------------------------------------------- inspection

    @property
    def tripped(self) -> bool:
        """True from a trip until every source is fresh again."""
        return self._tripped

    @property
    def trips(self) -> int:
        """How many times the stop was driven (repeats of one outage included)."""
        return self._trips

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def stale_sources(self) -> dict[str, float]:
        """``{source: seconds since its last beat}`` for each source past the window."""
        now = self._clock()
        return {
            name: now - beat
            for name, beat in self._last_beat.items()
            if now - beat > self._timeout_s
        }

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable state for status endpoints and logs."""
        now = self._clock()
        return {
            "running": self.running,
            "timeout_ms": round(self._timeout_s * 1000),
            "tripped": self._tripped,
            "trips": self._trips,
            "last_trip_reason": self._last_reason,
            "sources": {
                name: {
                    "age_ms": round((now - beat) * 1000),
                    "stale": now - beat > self._timeout_s,
                }
                for name, beat in self._last_beat.items()
            },
        }

    # ------------------------------------------------------------------ the loop

    async def start(self) -> None:
        if self._task is None or self._task.done():
            now = self._clock()
            for name in self._last_beat:  # a fresh start gives every source one full window
                self._last_beat[name] = now
            self._tripped = False
            self._stop_evt.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_evt.set()
        task, self._task = self._task, None
        if task is None:
            return
        try:
            await asyncio.wait_for(task, timeout=_STOP_GRACE_S)
        except TimeoutError:
            logger.warning("Watchdog loop did not stop within %.1f s; cancelled", _STOP_GRACE_S)

    async def _run(self) -> None:
        # Never exits on its own: a trip does not end supervision, and an unexpected error
        # in one cycle must not leave the machine unsupervised.
        while not self._stop_evt.is_set():
            try:
                await self.evaluate()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Watchdog cycle failed; continuing")
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._poll_interval_s)
            except TimeoutError:
                pass

    async def evaluate(self) -> bool:
        """One watchdog cycle; ``True`` when it drove the stop.

        Public so tests can step it on a fake clock; the loop calls it every poll.
        """
        stale = self.stale_sources()
        if not stale:
            if self._tripped:
                self._tripped = False
                self._last_log_at = None
                logger.warning(
                    "Watchdog re-armed: every heartbeat source is fresh again "
                    "(an E-stop latch it set still needs an operator clear)"
                )
            return False
        if self._tripped and self._clock() < self._next_trip_at:
            return False
        await self._trip(stale)
        return True

    async def _trip(self, stale: dict[str, float]) -> None:
        detail = ", ".join(
            f"{name} silent {age * 1000:.0f} ms" for name, age in sorted(stale.items())
        )
        first = not self._tripped
        self._tripped = True
        self._trips += 1
        self._last_reason = detail
        now = self._clock()
        if first or self._last_log_at is None or now - self._last_log_at >= _REPEAT_LOG_INTERVAL_S:
            self._last_log_at = now
            logger.critical(
                "WATCHDOG TIMEOUT (%s; window %.0f ms): stopping the machine%s",
                detail,
                self._timeout_s * 1000,
                "" if first else f" (repeat; {self._trips} stops so far)",
            )
        try:
            self._estop.trigger_estop("watchdog_timeout")
        except Exception:
            logger.exception("Watchdog: revoking authorization failed")
        if self._on_timeout is not None:
            try:
                result = await asyncio.wait_for(
                    self._on_timeout(), timeout=self._on_timeout_deadline_s
                )
            except TimeoutError:
                logger.critical(
                    "Watchdog on_timeout did not return within %.1f s: the stop is UNCONFIRMED; "
                    "it will be repeated",
                    self._on_timeout_deadline_s,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Watchdog on_timeout callback failed")
            else:
                if isinstance(result, dict) and result.get("complete") is False:
                    logger.critical(
                        "Watchdog stop INCOMPLETE -- unconfirmed outputs: %s; it will be repeated",
                        result.get("faults"),
                    )
        self._next_trip_at = self._clock() + self._retrip_interval_s


__all__ = [
    "ACTUATION_SOURCE",
    "DEFAULT_SOURCE",
    "SAFETY_MONITOR_SOURCE",
    "Watchdog",
]
