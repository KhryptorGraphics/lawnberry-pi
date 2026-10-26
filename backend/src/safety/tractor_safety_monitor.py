"""Tractor safety monitor (Constitution Principle VI follow-up).

Two gaps the tractor platform inherited from the mower platform's safety
system but never had wired to it: IMU tilt-cutoff (<200ms, blade/PTO) and a
software watchdog heartbeat. The tilt-cutoff closes here in one small periodic
task rather than inside ``TractorControlService`` itself, so actuation stays a
pure actuation/interlock concern and sensor fusion stays out of it.

This task feeds the watchdog's ``safety_monitor`` source only after a *completed*
tilt cycle (a hung or failing check is not a heartbeat). It is **not** the
actuation heartbeat: that one comes from the tractor service's own self-check of
its outputs (``TractorControlService.supervise_once``), so a healthy monitor loop
can no longer vouch for a broken command path (audit N-04).

Polls at 50ms (20Hz) -- 4x margin under the 200ms constitutional tilt-cutoff
deadline, since that deadline is threshold-breach-to-blade-stop, not
threshold-breach-to-next-poll.

A tilt trip *latches* the blade interlock on the tractor
(``TractorControlService.latch_interlock``): navigation re-sends
``blade_engaged=True`` every tick, so merely switching the blade off once would
have it re-engaged ~200ms later while still tilted. The latch is released only
by an explicit authenticated ``clear_emergency()``, never by the tilt returning
to normal. While the monitor runs it is also the tractor's *clear gate*
(``interlock_clear_veto``, audit SEC-06): ``clear_emergency()`` keeps the latch while the
latest tilt reading is over the threshold and while the tilt sensor is unavailable (no valid
reading within ``TILT_READING_MAX_AGE_S``), so a clear on a tilted machine cannot reopen the
blade path even for the one poll that remains before this loop would re-latch it. The loop
still re-latches every tick while tilted (a clear that landed before the first tick).

SIM-safe: the default IMU reader lazily reaches into the sensor manager the
same way the rest of the backend does (``websocket_hub._ensure_sensor_manager``)
and simply returns ``None`` on any failure, in which case tilt is not
evaluated that tick -- this task must never raise into its own loop.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from .safety_triggers import SafetyTriggerManager, get_safety_trigger_manager
from .watchdog import SAFETY_MONITOR_SOURCE, Watchdog

if TYPE_CHECKING:
    from ..services.tractor_service import TractorControlService

logger = logging.getLogger(__name__)

ImuReader = Callable[[], Awaitable[tuple[float, float] | None]]

#: How long ``stop()`` waits for the loop (a possibly hung tilt check) before cancelling it.
_STOP_GRACE_S = 1.0
#: A tilt reading older than this no longer shows the machine is level: the clear gate then
#: treats the tilt sensor as unavailable and keeps the latch (the loop polls every 50 ms).
TILT_READING_MAX_AGE_S = 1.0


async def _default_imu_reader() -> tuple[float, float] | None:
    try:
        from ..services import websocket_hub

        sm = await websocket_hub._ensure_sensor_manager()
        reading = await sm.imu.read_imu()
        if reading is None or reading.roll is None or reading.pitch is None:
            return None
        return float(reading.roll), float(reading.pitch)
    except Exception:
        return None


class TractorSafetyMonitor:
    """Periodic tilt-cutoff task for the tractor platform; feeds the watchdog's monitor source."""

    def __init__(
        self,
        tractor: TractorControlService,
        watchdog: Watchdog,
        tilt_threshold_degrees: float,
        trigger_manager: SafetyTriggerManager | None = None,
        imu_reader: ImuReader = _default_imu_reader,
        poll_interval_s: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if SAFETY_MONITOR_SOURCE not in watchdog.required_sources:
            # Fail at start-up, not by raising inside the loop that protects the blade.
            raise ValueError(
                f"the watchdog does not require the {SAFETY_MONITOR_SOURCE!r} heartbeat source"
            )
        self._tractor = tractor
        self._watchdog = watchdog
        self._threshold = float(tilt_threshold_degrees)
        self._triggers = trigger_manager or get_safety_trigger_manager()
        self._imu_reader = imu_reader
        self._poll_interval_s = poll_interval_s
        self._clock = clock
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._tilted = False
        # The latest valid (finite) reading and when it arrived; None until the first one.
        self._last_reading: tuple[float, float] | None = None
        self._last_reading_at: float | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_evt.clear()
            self._tractor.attach_interlock_clear_gate(self.interlock_clear_veto)
            self._task = asyncio.create_task(self._run())

    def interlock_clear_veto(self, latch_reason: str | None) -> str | None:
        """Why the tractor must keep its blade interlock latch now, else ``None``.

        Only the tilt latch is this monitor's to veto. It stays while the latest valid
        reading is over the threshold, and while there is no valid reading younger than
        ``TILT_READING_MAX_AGE_S`` (sensor missing, failing or this loop dead): a machine that
        cannot be shown to be level is treated as tilted.
        """
        if latch_reason != "tilt":
            return None
        if self._tilted and self._last_reading is not None:
            roll, pitch = self._last_reading
            return (
                f"the latest tilt reading (roll {roll:.1f}, pitch {pitch:.1f} deg) is over "
                f"the {self._threshold:.1f} deg limit"
            )
        if self._last_reading_at is None:
            return "the tilt sensor has not produced a valid reading yet"
        age = self._clock() - self._last_reading_at
        if age > TILT_READING_MAX_AGE_S:
            return (
                f"the tilt sensor is unavailable: its last valid reading is {age:.1f} s old "
                f"(limit {TILT_READING_MAX_AGE_S:.1f} s)"
            )
        return None

    async def stop(self) -> None:
        self._stop_evt.set()
        self._tractor.detach_interlock_clear_gate(self.interlock_clear_veto)
        task, self._task = self._task, None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=_STOP_GRACE_S)
            except TimeoutError:
                # A hung IMU read must not hang shutdown; wait_for cancelled the task.
                logger.warning("Tilt monitor did not stop within %.1f s; cancelled", _STOP_GRACE_S)

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                await self._check_tilt()
            except Exception:
                # No heartbeat: a tilt check that keeps failing is a dead blade protection,
                # and the watchdog must see that rather than a loop that merely turns.
                logger.exception("Tractor tilt-cutoff check failed")
            else:
                self._watchdog.heartbeat(SAFETY_MONITOR_SOURCE)
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._poll_interval_s)
            except TimeoutError:
                pass

    async def _check_tilt(self) -> None:
        reading = await self._imu_reader()
        if reading is None:
            return
        roll_deg, pitch_deg = reading
        if not (math.isfinite(roll_deg) and math.isfinite(pitch_deg)):
            # A NaN/inf reading is no reading: it must neither latch nor release anything, and
            # it does not count as proof that the machine is level (see interlock_clear_veto).
            return
        self._last_reading = (roll_deg, pitch_deg)
        self._last_reading_at = self._clock()
        triggered = self._triggers.trigger_tilt(roll_deg, pitch_deg, self._threshold)
        if triggered:
            state = self._tractor.state
            if not self._tilted:
                logger.warning(
                    "Tilt cutoff: roll=%.1f pitch=%.1f >= %.1f deg -- latching blade off",
                    roll_deg,
                    pitch_deg,
                    self._threshold,
                )
            self._tilted = True
            # Level-triggered, not edge-triggered: latch even if the blade is
            # already off (the next navigation leg would turn it on), and
            # re-latch if an operator clear landed while still tilted.
            if not state.interlock_latched or state.blade_engaged:
                await self._tractor.latch_interlock("tilt", source="tilt_monitor")
        elif self._tilted:
            # Back under the threshold: the SafetyTrigger interlock clears, but the
            # tractor's blade latch deliberately stays until an operator clears it.
            self._tilted = False
            self._triggers.clear_tilt()
