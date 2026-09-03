"""Tractor safety monitor (Constitution Principle VI follow-up).

Two gaps the tractor platform inherited from the mower platform's safety
system but never had wired to it: IMU tilt-cutoff (<200ms, blade/PTO) and a
software watchdog heartbeat. Both close here in one small periodic task
rather than inside ``TractorControlService`` itself, so actuation stays a
pure actuation/interlock concern and sensor fusion stays out of it.

Polls at 50ms (20Hz) -- 4x margin under the 200ms constitutional tilt-cutoff
deadline, since that deadline is threshold-breach-to-blade-stop, not
threshold-breach-to-next-poll.

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
from .watchdog import Watchdog

if TYPE_CHECKING:
    from ..services.tractor_service import TractorControlService

logger = logging.getLogger(__name__)

ImuReader = Callable[[], Awaitable[tuple[float, float] | None]]


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
    """Periodic tilt-cutoff + watchdog-heartbeat task for the tractor platform."""

    def __init__(
        self,
        tractor: TractorControlService,
        watchdog: Watchdog,
        tilt_threshold_degrees: float,
        trigger_manager: SafetyTriggerManager | None = None,
        imu_reader: ImuReader = _default_imu_reader,
        poll_interval_s: float = 0.05,
    ) -> None:
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
        if self._task:
            await self._task

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            # This task's own liveness IS the control-loop heartbeat: it runs
            # continuously whenever the tractor platform is armed, independent
            # of whether any drive/blade command happens to be in flight.
            self._watchdog.heartbeat()
            try:
                await self._check_tilt()
            except Exception:
                logger.exception("Tractor tilt-cutoff check failed")
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
        if triggered and not self._tilted:
            self._tilted = True
            if self._tractor.state.blade_engaged:
                logger.warning(
                    "Tilt cutoff: roll=%.1f pitch=%.1f >= %.1f deg -- disengaging blade",
                    roll_deg,
                    pitch_deg,
                    self._threshold,
                )
                await self._tractor.engage_blade(False)
        elif not triggered and self._tilted:
            self._tilted = False
            self._triggers.clear_tilt()
