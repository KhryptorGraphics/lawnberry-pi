"""GNSS quality monitor and hold (T063; audit N-31, WP-P4-03, D.9).

What it watches. :meth:`GPSDegradationMonitor.observe` is fed the **real** ``GpsReading`` the
sensor manager produced (``NavigationService.update_navigation_state`` on every telemetry
tick), not a debug-route copy in ``RobotState``. Fix age is the age of the *fix itself*
(``reading.timestamp``), so a reading the interface replays after the receiver went quiet
keeps getting older instead of looking fresh on every poll. A fix is unusable when it is
missing, has no position, reports ``NO_FIX``/``DEAD_RECKONING``, is older than
``max_fix_age_s``, or is less accurate than ``max_accuracy_m`` (an unknown accuracy is not a
good accuracy).

What it does about it. :meth:`assess` is the verdict the navigation loops read every tick
(``NavigationService._motion_gate``): while unhealthy, autonomous motion is **held at the
actuators** (levers neutral, blade off), exactly like an obstacle hold, and mission start is
refused. While degraded and autonomy is running, the monitor task also calls its
``hold_handlers`` every cycle so the machine is stopped without waiting for the planner's next
tick. Leaving the degraded state needs a fix that has stayed healthy for ``resume_dwell_s``
(hysteresis); the monitor starts degraded, so a mission never starts on the first lucky fix.

The thresholds are UNVALIDATED defaults (``limits.yaml`` ``gps_degradation``). The accuracy
bound is the owner's and the reviewer's decision (SD-4), to be derived from the geofence
margin and the measured position error (M-50/M-51).

Legacy behaviour kept: while ``RobotState.navigation_mode`` is ``AUTONOMOUS`` (set only by
the debug route ``POST /api/v2/nav/mode``) a poor ``RobotState`` position still drops it to
``MANUAL``. That mode flag never controlled an actuator and is not how real missions run.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.robot_state_manager import get_robot_state_manager
from ..models.robot_state import NavigationMode

logger = logging.getLogger(__name__)

#: ``GpsReading.rtk_status`` values that are not a GNSS position fix at all.
UNUSABLE_FIX_STATUS = frozenset({"NO_FIX", "DEAD_RECKONING"})


@dataclass
class GPSDegradationConfig:
    # Defaults mirror ``models.safety_limits.GpsDegradationLimits`` (a test pins that).
    max_accuracy_m: float = 1.0
    max_fix_age_s: float = 2.0
    check_interval_s: float = 0.2
    resume_dwell_s: float = 3.0

    @classmethod
    def from_limits(cls, limits: Any) -> GPSDegradationConfig:
        """Build from ``SafetyLimits.gps_degradation``."""
        return cls(
            max_accuracy_m=limits.max_accuracy_m,
            max_fix_age_s=limits.max_fix_age_s,
            check_interval_s=limits.check_interval_s,
            resume_dwell_s=limits.resume_dwell_s,
        )


@dataclass(frozen=True, slots=True)
class GnssAssessment:
    """The monitor's verdict at one instant."""

    healthy: bool  # a usable fix that has stayed usable for ``resume_dwell_s``
    reasons: tuple[str, ...]
    fix_age_s: float | None
    accuracy_m: float | None
    rtk_status: str | None
    assessed_at: float

    @property
    def degraded(self) -> bool:
        return not self.healthy


HoldHandler = Callable[[GnssAssessment], Awaitable[Any]]


def _reading_age_s(reading: Any, wall_now: _dt.datetime) -> float:
    """Age of the fix itself (0.0 when the reading carries no usable timestamp)."""
    stamp = getattr(reading, "timestamp", None)
    if isinstance(stamp, str):
        try:
            stamp = _dt.datetime.fromisoformat(stamp)
        except ValueError:
            stamp = None
    if not isinstance(stamp, _dt.datetime):
        return 0.0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=_dt.UTC)
    return max(0.0, (wall_now - stamp).total_seconds())


class GPSDegradationMonitor:
    """GNSS-quality verdict + actuator-level hold (see the module docstring).

    Args:
        config: thresholds (``GPSDegradationConfig.from_limits(limits.gps_degradation)``).
        autonomy_active: ``() -> bool``; true while a mission/AI run is moving the machine.
            Without it the monitor only reports; it never calls its handlers.
        hold_handlers: awaited every cycle while degraded and autonomy is active
            (idempotent, cheap when nothing moves).
        clock / wall_clock: injectable for tests (monotonic seconds / aware ``datetime``).
    """

    HANDLER_TIMEOUT_S = 1.0

    def __init__(
        self,
        config: GPSDegradationConfig | None = None,
        *,
        autonomy_active: Callable[[], bool] | None = None,
        hold_handlers: Sequence[HoldHandler] = (),
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], _dt.datetime] | None = None,
    ):
        self.cfg = config or GPSDegradationConfig()
        self._clock = clock
        self._wall = wall_clock or (lambda: _dt.datetime.now(_dt.UTC))
        self._autonomy_active = autonomy_active
        self._handlers = tuple(hold_handlers)
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._reading: Any | None = None
        self._fix_at: float | None = None  # clock time the newest fix was *produced*
        self._degraded = True  # no fix yet == degraded
        self._good_since: float | None = None
        self._reasons: tuple[str, ...] = ("no GNSS reading yet",)
        self._last_logged: bool | None = None
        self.cycles = 0
        self.errors = 0

    # ----------------------------------------------------------------- feed / verdict

    def observe(self, reading: Any | None, now: float | None = None) -> None:
        """Record the newest ``GpsReading`` (None = nothing new this tick) and judge it."""
        if reading is None:
            return
        t = self._clock() if now is None else now
        self._reading = reading
        # Fix time = arrival minus the fix's own age: a replayed old reading does not get younger.
        self._fix_at = t - _reading_age_s(reading, self._wall())
        self._refresh(t)

    def _unusable_reasons(self, t: float) -> list[str]:
        r = self._reading
        if r is None or self._fix_at is None:
            return ["no GNSS reading yet"]
        reasons: list[str] = []
        lat, lon = getattr(r, "latitude", None), getattr(r, "longitude", None)
        if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
            reasons.append("no position")
        status = getattr(r, "rtk_status", None)
        if status in UNUSABLE_FIX_STATUS:
            reasons.append(f"fix status {status}")
        age = max(0.0, t - self._fix_at)
        if age > self.cfg.max_fix_age_s:
            reasons.append(f"fix is {age:.1f}s old (limit {self.cfg.max_fix_age_s:g}s)")
        acc = getattr(r, "accuracy", None)
        if acc is None or not isinstance(acc, (int, float)) or math.isnan(acc):
            reasons.append("accuracy unknown")
        elif acc > self.cfg.max_accuracy_m:
            reasons.append(f"accuracy {acc:.2f}m exceeds {self.cfg.max_accuracy_m:g}m")
        return reasons

    def _refresh(self, t: float) -> None:
        """Advance the hysteresis state machine to time ``t`` (every observation and poll)."""
        reasons = self._unusable_reasons(t)
        if reasons:
            self._degraded = True
            self._good_since = None
            self._reasons = tuple(reasons)
        elif self._degraded:
            if self._good_since is None:
                self._good_since = t
            remaining = self.cfg.resume_dwell_s - (t - self._good_since)
            if remaining <= 1e-9:  # float tolerance: 15 x 0.2 s is not exactly 3.0 s
                self._degraded = False
                self._good_since = None
                self._reasons = ()
            else:
                self._reasons = (f"recovering: {remaining:.1f}s of healthy fix still required",)
        else:
            self._reasons = ()

    def assess(self, now: float | None = None) -> GnssAssessment:
        """The verdict now. Recomputed from the clock on every call (never cached)."""
        t = self._clock() if now is None else now
        self._refresh(t)
        r = self._reading
        acc = getattr(r, "accuracy", None)
        return GnssAssessment(
            healthy=not self._degraded,
            reasons=self._reasons,
            fix_age_s=None if self._fix_at is None else max(0.0, t - self._fix_at),
            accuracy_m=float(acc) if isinstance(acc, (int, float)) else None,
            rtk_status=getattr(r, "rtk_status", None),
            assessed_at=t,
        )

    # ------------------------------------------------------------------- task

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self.cfg.check_interval_s * 2 + 1.0)
            except Exception:
                # Best-effort shutdown
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                self._tick()
            except Exception:
                # Do not crash on monitoring exceptions
                pass
            try:
                await self._check()
            except Exception:
                self.errors += 1
                logger.exception("GNSS monitor cycle failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.cfg.check_interval_s)
            except TimeoutError:
                continue

    async def _check(self) -> None:
        """One cycle: judge the fix and, if degraded while autonomy runs, stop the machine."""
        a = self.assess()
        self.cycles += 1
        if a.healthy is not self._last_logged:
            if a.healthy:
                logger.info("GNSS fix healthy again (accuracy %s m)", a.accuracy_m)
            else:
                logger.warning("GNSS degraded: %s", "; ".join(a.reasons))
            self._last_logged = a.healthy
        if a.healthy or self._autonomy_active is None or not self._autonomy_active():
            return
        for handler in self._handlers:
            try:
                await asyncio.wait_for(handler(a), timeout=self.HANDLER_TIMEOUT_S)
            except Exception:
                self.errors += 1
                logger.exception("GNSS hold handler failed (the navigation gate still holds)")

    def _tick(self) -> None:
        """Legacy ``RobotState`` fallback (debug-route AUTONOMOUS mode only; see the docstring)."""
        mgr = get_robot_state_manager()
        st = mgr.get_state()

        # Only enforce degradation during autonomous operation
        if st.navigation_mode != NavigationMode.AUTONOMOUS:
            return

        now = _dt.datetime.now(_dt.UTC)
        last = st.last_updated

        # Accuracy threshold
        accuracy = st.position.accuracy_m
        if accuracy is not None and accuracy > self.cfg.max_accuracy_m:
            st.navigation_mode = NavigationMode.MANUAL
            st.touch()
            return

        # Fix age threshold (if last state update older than limit)
        if last and (now - last).total_seconds() > self.cfg.max_fix_age_s:
            st.navigation_mode = NavigationMode.MANUAL
            st.touch()


__all__ = [
    "GPSDegradationConfig",
    "GPSDegradationMonitor",
    "GnssAssessment",
    "UNUSABLE_FIX_STATUS",
]
