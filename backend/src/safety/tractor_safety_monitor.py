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
to normal, and it is re-asserted every tick while the machine is still over the
threshold, so clearing it on a tilted machine cannot reopen the blade path.

SIM-safe: the default IMU reader lazily reaches into the sensor manager the
same way the rest of the backend does (``websocket_hub._ensure_sensor_manager``)
and simply returns ``None`` on any failure, in which case tilt is not
evaluated that tick -- this task must never raise into its own loop.
"""

from __future__ import annotations

import asyncio
import logging
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
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._tilted = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_evt.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_evt.set()
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
