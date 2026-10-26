"""Deterministic local obstacle stop path (WP-P4-01; audit N-11, N-25, H-26).

One authority, :class:`ObstacleGuard`, turns range observations into a **speed cap** using the
stopping-distance model (``stopping_distance.py``). It is independent of the navigation
planner, the AI and the Thor: they may only ever *reduce* speed further, never lengthen it.

Fail closed. A required source (ToF left/right, each ultrasonic channel) that has no reading,
a stale reading, an invalid reading or a reading from inside the sensor's blind zone counts as
**blocked** (cap 0), exactly as a close obstacle does. "Nothing heard" from a healthy sensor is
not the same as "clear for ever": it only proves clearance up to ``detection_range_m``, which
therefore bounds the cap. A hold is released only after the path has been continuously clear
for ``clear_dwell_s``.

Where it is enforced (all of them read the same guard):

* ``TractorControlService.set_levers/apply`` (the chokepoint every caller reaches, whoever
  planned the motion, including a Thor plan driven through the service interface) scales
  forward lever demand down to the cap and forces neutral and blade-off during a hold;
* ``NavigationService.go_to_waypoint``/``apply_ai_prediction``/``set_speed`` read the same cap
  and hold (differential platform included);
* :class:`ObstacleStopMonitor` samples the sensors at a fixed rate, publishes the obstacle
  interlocks (``SafetyTriggerManager``, previously with no production caller) and actively
  drives neutral + blade-off the moment a hold begins, without waiting for the next command.

Manual commands (sources ``operator``/``teleop``) are not limited unless
``apply_to_manual`` is set: a human at the machine, or a bench test with the sensors
disconnected, must still be able to move the levers. Anything else, including an unlabeled
caller, is treated as autonomous and limited.

Ages are monotonic. ToF and ultrasonic readings carry a wall-clock stamp; the sampler turns it
into a time on the guard's own monotonic clock ONCE, when the reading is first seen
(``core.freshness.ReceiptAges``), so a backward clock step cannot make a frozen reading fresh
again, and a stamp from the future is unknown, never fresh (audit SAF-05). A ``clear`` verdict
needs a usable horizon; without one it is unknown, not clear to the full detection range
(SAF-10). A ToF source that repeats one identical in-range distance in
``frozen_reading_count`` consecutive fresh readings is treated as frozen, i.e. unknown (SAF-S2).

Limits that remain (see the audit): ultrasonic cones are not an occupancy map and cannot see
soft targets; nothing senses behind the machine, so reverse is not limited; speed is the
commanded lever demand, not a measurement; every number is UNVALIDATED until M-43.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..core.freshness import ReceiptAges
from ..models.safety_limits import ObstacleStopLimits, SafetyLimits
from .safety_triggers import SafetyTriggerManager, get_safety_trigger_manager
from .stopping_distance import StoppingDistanceModel

logger = logging.getLogger(__name__)

#: Command sources that are a human in direct control. Everything else is autonomous.
MANUAL_SOURCES = frozenset({"operator", "teleop"})

TOF_KIND = "tof"
ULTRASONIC_KIND = "ultrasonic"
TOF_SOURCES = {"tof_left": "left_reading", "tof_right": "right_reading"}
ULTRASONIC_PREFIX = "ultrasonic:"

# Ultrasonic channel entries (UltrasonicSensorInterface contract). The meaning of a silent
# echo line is the sensor interface's policy, not ours: we follow its ``verdict`` and only
# ever tighten it (an entry counts as clear only if the interface says so AND it is fresh).
_US_OK = "ok"
_US_BLIND = "blind_zone"  # something may be inside the minimum range: never clear
_US_UNKNOWN = "unknown"
_US_CLEAR = "clear"

_LOG_INTERVAL_S = 5.0


class ObstacleStatus(StrEnum):
    CLEAR = "clear"  # full speed allowed
    SLOW = "slow"  # speed capped below full
    STOP = "stop"  # hold: levers neutral, blade off


class _Kind(StrEnum):
    RANGE = "range"  # a measured distance
    CLEAR = "clear"  # healthy sensor, nothing within its detection range
    BLIND = "blind"  # the object may be inside the blind zone: treated as an obstacle
    UNKNOWN = "unknown"  # no usable data


@dataclass(frozen=True, slots=True)
class _Observation:
    kind: _Kind
    distance_m: float | None
    measured_at: float
    detail: str


@dataclass(frozen=True, slots=True)
class ObstacleAssessment:
    """The guard's verdict at one instant."""

    status: ObstacleStatus
    speed_cap_mps: float
    lever_cap: float
    nearest_m: float | None
    #: ``(source, distance_m or None)`` for sources reporting an obstacle or a blind zone.
    obstacles: tuple[tuple[str, float | None], ...]
    #: Sources with no usable data (missing, stale, unknown): a sensor fault, not an obstacle.
    faults: tuple[str, ...]
    reasons: tuple[str, ...]
    held: bool
    model_validated: bool
    assessed_at: float

    @property
    def hold(self) -> bool:
        return self.status is ObstacleStatus.STOP

    @property
    def obstacle_sources(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.obstacles)


def limit_pair(a: float, b: float, cap: float) -> tuple[float, float]:
    """Scale ``(a, b)`` so the larger forward component does not exceed ``cap``.

    Used for left/right wheel speeds and lever deflections alike. The ratio is kept, so a
    curve stays the same curve at a lower speed. Pure reverse (no forward component) is not
    limited. A ``cap`` of 0 returns neutral for both.
    """
    top = max(a, b)
    if top <= 0.0 or top <= cap:
        return a, b
    if cap <= 0.0:
        return 0.0, 0.0
    f = cap / top
    na, nb = a * f, b * f
    return (min(na, cap) if na > 0 else na), (min(nb, cap) if nb > 0 else nb)


def _finite(value: Any) -> float | None:
    """``value`` as a finite float; ``None`` for anything that is not a real measurement
    (``None``, NaN, infinity, a bool, a string)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


#: A ToF distance at or beyond this is the sensor's "no target" value (8190 mm and the like),
#: which is legitimately constant and is not tested for being frozen.
_TOF_NO_TARGET_M = 4.0
#: Float noise allowed before a measurement time ahead of the clock counts as "from the future".
_FUTURE_TOLERANCE_S = 1e-6


@dataclass(slots=True)
class _Repeat:
    """The run of identical readings a ToF source is on (see ``ObstacleGuard.frozen``)."""

    value: float
    stamp: Any
    count: int


class ObstacleGuard:
    """Fail-closed obstacle assessment shared by every motion path (see module docstring).

    Producers call ``observe_*`` with the time the measurement was *taken* (``measured_at`` on
    ``clock``, never the time they looked). Consumers call :meth:`assess`, which recomputes
    from the clock on every call, so a dead producer cannot leave a stale "clear" behind.
    """

    def __init__(
        self,
        limits: ObstacleStopLimits | None = None,
        *,
        tof_stop_m: float = 0.2,
        ultrasonic_stop_m: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits or ObstacleStopLimits()
        self.model = StoppingDistanceModel(self.limits)
        self._clock = clock
        self._floors = {TOF_KIND: float(tof_stop_m), ULTRASONIC_KIND: float(ultrasonic_stop_m)}
        self._kinds: dict[str, str] = {}
        self._obs: dict[str, _Observation] = {}
        self._ignored = frozenset(self.limits.ignored_sources)
        self._ages = ReceiptAges(clock)
        self._repeats: dict[str, _Repeat] = {}
        self._held = True  # no data yet == hold
        self._clear_since: float | None = None
        self._last = None  # type: ObstacleAssessment | None
        if self._ignored:
            logger.critical(
                "Obstacle guard: sources %s are IGNORED by configuration (obstacle_stop."
                "ignored_sources); they no longer hold the mower",
                sorted(self._ignored),
            )

    # ----------------------------------------------------------------- producers

    def now(self) -> float:
        return self._clock()

    def register_source(self, name: str, kind: str) -> None:
        """Declare a required source. Idempotent; ``kind`` selects the static stop line."""
        if kind not in self._floors:
            raise ValueError(f"unknown source kind {kind!r}")
        self._kinds.setdefault(name, kind)

    @property
    def sources(self) -> tuple[str, ...]:
        """The required sources (registered minus the explicitly ignored)."""
        return tuple(s for s in self._kinds if s not in self._ignored)

    def _store(self, source: str, kind: _Kind, dist: float | None, at: float, detail: str) -> None:
        if source not in self._kinds:
            raise KeyError(f"unregistered obstacle source {source!r}")
        self._obs[source] = _Observation(kind, dist, at, detail)

    def observe_range(self, source: str, distance_m: float, measured_at: float) -> None:
        """A measured distance. Non-finite or non-positive values are unknown, not clear."""
        d = _finite(distance_m)
        if d is None or d <= 0.0:
            self._store(
                source, _Kind.UNKNOWN, None, measured_at, f"invalid distance {distance_m!r}"
            )
        else:
            self._store(source, _Kind.RANGE, d, measured_at, "")

    def observe_clear(self, source: str, measured_at: float, horizon_m: float | None) -> None:
        """A healthy sensor that proved the path free out to ``horizon_m`` (a longer horizon is
        clamped to the guard's detection range). The horizon must be a finite number above 0:
        a sensor that cannot say how far it saw proved nothing, so ``None``, NaN, infinity, 0
        and negatives are UNKNOWN (blocked), never "clear to the full range" (audit SAF-10)."""
        horizon = _finite(horizon_m)
        if horizon is None or horizon <= 0.0:
            self._store(
                source, _Kind.UNKNOWN, None, measured_at, f"invalid clear horizon {horizon_m!r}"
            )
        else:
            self._store(source, _Kind.CLEAR, horizon, measured_at, "")

    def observe_blind(self, source: str, measured_at: float, detail: str = "blind zone") -> None:
        """Something may be inside the sensor's blind zone: treated as an obstacle."""
        self._store(source, _Kind.BLIND, None, measured_at, detail)

    def observe_unknown(self, source: str, measured_at: float, detail: str = "unknown") -> None:
        """No usable data (offline, error, stale, link down): blocked."""
        self._store(source, _Kind.UNKNOWN, None, measured_at, detail)

    # ----------------------------------------------------------------- consumers

    def applies_to(self, source: str | None) -> bool:
        """Is a command with this ``source`` label subject to the limit?"""
        if source in MANUAL_SOURCES:
            return self.limits.apply_to_manual
        return True

    def assess(self, now: float | None = None) -> ObstacleAssessment:
        """Recompute the verdict from the observations and the clock (never cached)."""
        t = self._clock() if now is None else now
        lim = self.limits
        sources = self.sources
        faults: list[str] = []
        obstacles: list[tuple[str, float | None]] = []
        reasons: list[str] = []
        nearest: float | None = None
        cap = lim.full_lever_speed_mps

        if not sources:
            faults.append("no_sources")
            reasons.append("no obstacle sources configured: autonomous motion is not permitted")
            cap = 0.0

        for name in sources:
            obs = self._obs.get(name)
            if obs is None:
                faults.append(name)
                reasons.append(f"{name}: no data")
                cap = 0.0
                continue
            age = t - obs.measured_at
            if age < -_FUTURE_TOLERANCE_S:
                # A measurement from the future is not fresh, it is untrustworthy (the clock it
                # was stamped with disagrees with this one): never clamp it to age 0.
                faults.append(name)
                reasons.append(f"{name}: measurement is {-age:.2f}s in the future")
                cap = 0.0
                continue
            age = max(0.0, age)
            if age > lim.max_reading_age_s:
                faults.append(name)
                reasons.append(f"{name}: stale ({age:.2f}s old)")
                cap = 0.0
                continue
            if obs.kind is _Kind.UNKNOWN:
                faults.append(name)
                reasons.append(f"{name}: {obs.detail}")
                cap = 0.0
                continue
            if obs.kind is _Kind.BLIND:
                obstacles.append((name, None))
                reasons.append(f"{name}: {obs.detail}")
                cap = 0.0
                continue
            free = lim.detection_range_m
            if obs.kind is _Kind.RANGE:
                assert obs.distance_m is not None
                free = min(obs.distance_m, free)
                nearest = obs.distance_m if nearest is None else min(nearest, obs.distance_m)
            elif obs.kind is _Kind.CLEAR and obs.distance_m is not None:
                free = min(free, obs.distance_m)
            floor = self._floors[self._kinds[name]]
            if obs.kind is _Kind.RANGE and obs.distance_m <= floor:
                obstacles.append((name, obs.distance_m))
                reasons.append(
                    f"{name}: obstacle at {obs.distance_m:.2f}m (stop line {floor:.2f}m)"
                )
                cap = 0.0
                continue
            source_cap = self.model.max_speed_for_range(free, extra_latency_s=age)
            if source_cap <= 0.0:
                if obs.kind is _Kind.RANGE:
                    obstacles.append((name, obs.distance_m))
                    reasons.append(
                        f"{name}: obstacle at {obs.distance_m:.2f}m inside the stopping envelope"
                    )
                else:  # cannot happen with a sane model: the horizon alone leaves no room
                    faults.append(name)
                    reasons.append(f"{name}: detection range leaves no room to stop")
                cap = 0.0
            elif obs.kind is _Kind.RANGE and source_cap < lim.full_lever_speed_mps:
                reasons.append(f"{name}: {obs.distance_m:.2f}m limits speed to {source_cap:.2f}m/s")
            cap = min(cap, source_cap)

        # Hysteresis: after any hold, stay held until continuously clear for the dwell.
        if cap <= 0.0:
            self._held = True
            self._clear_since = None
        elif self._held:
            if self._clear_since is None:
                self._clear_since = t
            remaining = lim.clear_dwell_s - (t - self._clear_since)
            if remaining <= 0.0:
                self._held = False
                self._clear_since = None
            else:
                reasons.append(f"clearing: {remaining:.1f}s of clear path still required")
                cap = 0.0

        # CLEAR = nothing in view (the cap may still be bounded by sensor reach); SLOW = a
        # measured obstacle is limiting speed; STOP = hold. The cap is the authority.
        in_view = nearest is not None and nearest < lim.detection_range_m
        status = (
            ObstacleStatus.STOP
            if cap <= 0.0
            else ObstacleStatus.SLOW
            if in_view
            else ObstacleStatus.CLEAR
        )
        result = ObstacleAssessment(
            status=status,
            speed_cap_mps=cap,
            lever_cap=cap / lim.full_lever_speed_mps,
            nearest_m=nearest,
            obstacles=tuple(obstacles),
            faults=tuple(faults),
            reasons=tuple(reasons),
            held=self._held,
            model_validated=self.model.validated,
            assessed_at=t,
        )
        self._last = result
        return result

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable state for telemetry and the Pi->Thor uplink."""
        a = self.assess()
        t = a.assessed_at
        return {
            "status": a.status.value,
            "speed_cap_mps": round(a.speed_cap_mps, 3),
            "nearest_m": None if a.nearest_m is None else round(a.nearest_m, 3),
            "held": a.held,
            "reasons": list(a.reasons),
            "model_validated": a.model_validated,
            "sources": {
                name: {
                    "state": (o.kind.value if o is not None else "missing"),
                    "distance_m": (
                        round(o.distance_m, 3)
                        if o is not None and o.kind is _Kind.RANGE and o.distance_m is not None
                        else None
                    ),
                    "clear_to_m": (
                        round(o.distance_m, 3)
                        if o is not None and o.kind is _Kind.CLEAR and o.distance_m is not None
                        else None
                    ),
                    "age_s": (
                        None
                        if o is None or t - o.measured_at < -_FUTURE_TOLERANCE_S
                        else round(max(0.0, t - o.measured_at), 3)
                    ),
                    "detail": None if o is None or not o.detail else o.detail,
                }
                for name in self.sources
                for o in (self._obs.get(name),)
            },
        }

    def taken_at(self, source: str, stamp: Any, wall_now: datetime) -> float | None:
        """When, on this guard's clock, the measurement stamped ``stamp`` (wall clock) was
        taken; ``None`` if it cannot be aged. See ``core.freshness.ReceiptAges``."""
        return self._ages.taken_at(source, stamp, wall_now)

    def frozen(self, source: str, distance_m: float, stamp: Any) -> bool:
        """Record one reading of ``distance_m`` and say whether the source looks frozen.

        A VL53L0X that stops ranging but keeps answering returns the last value for ever, and
        ``range_status`` stays ``valid`` (audit SAF-S2). A real chip pointed at a real target
        jitters by a millimetre or more, so ``frozen_reading_count`` consecutive *distinct*
        readings (a replay of the same stamp does not count) of exactly one distance means
        frozen, and stays frozen until the value changes. Only distances beyond the hold
        distance (already a hold) and inside the sensor's real range are tested: a constant
        "no target" value is legitimate.
        """
        if not self.model.hold_distance() < distance_m < _TOF_NO_TARGET_M:
            self._repeats.pop(source, None)
            return False
        run = self._repeats.get(source)
        if run is None or run.value != distance_m:
            self._repeats[source] = _Repeat(distance_m, stamp, 1)
            return False
        if run.stamp != stamp:
            run.stamp = stamp
            run.count += 1
        return run.count >= self.limits.frozen_reading_count

    def end_run(self, source: str) -> None:
        """A reading that is not a valid ToF distance breaks the run of identical ones."""
        self._repeats.pop(source, None)

    def reset(self) -> None:
        """Forget every observation and start held (tests; a fresh process starts the same)."""
        self._obs.clear()
        self._ages.forget()
        self._repeats.clear()
        self._held = True
        self._clear_since = None
        self._last = None


# ---------------------------------------------------------------------------- sampling


def _tof_observation(guard: ObstacleGuard, manager: Any, wall_now: datetime) -> None:
    tof = getattr(manager, "tof", None)
    status = getattr(getattr(tof, "status", None), "value", getattr(tof, "status", None))
    now = guard.now()
    for source, attr in TOF_SOURCES.items():
        if source not in guard.sources:
            continue
        if tof is None or status != "online":
            guard.end_run(source)
            guard.observe_unknown(source, now, f"tof sensor {status or 'absent'}")
            continue
        reading = getattr(tof, attr, None)
        if reading is None:
            guard.end_run(source)
            guard.observe_unknown(source, now, "no reading yet")
            continue
        range_status = getattr(reading, "range_status", None)
        distance_mm = _finite(getattr(reading, "distance", None))
        if range_status != "valid" or distance_mm is None:
            guard.end_run(source)
            guard.observe_unknown(source, now, f"range_status={range_status}")
            continue
        stamp = getattr(reading, "timestamp", None)
        taken = guard.taken_at(source, stamp, wall_now)
        if taken is None:
            guard.end_run(source)
            guard.observe_unknown(source, now, "reading timestamp missing or from the future")
        elif guard.frozen(source, distance_mm / 1000.0, stamp):
            guard.observe_unknown(
                source,
                taken,
                f"frozen: {distance_mm:.0f} mm in {guard.limits.frozen_reading_count} "
                "consecutive readings",
            )
        else:
            guard.observe_range(source, distance_mm / 1000.0, taken)


def _ultrasonic_taken_at(
    guard: ObstacleGuard, source: str, entry: dict[str, Any], wall_now: datetime
) -> float | None:
    """When an ultrasonic entry was measured, on the guard's clock; ``None`` if unusable.

    An entry that carries its own monotonic receive time (``measured_mono``, the host's
    ``time.monotonic()`` when the MCU frame arrived) is aged from that. A key that is present
    but not a time at or before now is unusable, never a reason to fall back to the wall stamp.
    """
    if entry.get("measured_mono") is not None:
        mono = _finite(entry["measured_mono"])
        if mono is None or mono > guard.now() + _FUTURE_TOLERANCE_S:
            return None
        return mono
    return guard.taken_at(source, entry.get("timestamp"), wall_now)


def _ultrasonic_observations(guard: ObstacleGuard, manager: Any, wall_now: datetime) -> None:
    if not getattr(manager, "ultrasonic_enabled", False):
        return
    us = getattr(manager, "ultrasonic", None)
    now = guard.now()
    entries = list(getattr(us, "last_readings", None) or [])
    ids = list(getattr(us, "channel_ids", None) or [e.get("sensor_id") for e in entries])
    for channel in ids:
        guard.register_source(f"{ULTRASONIC_PREFIX}{channel}", ULTRASONIC_KIND)
    by_id = {e.get("sensor_id"): e for e in entries if isinstance(e, dict)}
    for channel in ids:
        source = f"{ULTRASONIC_PREFIX}{channel}"
        if source not in guard.sources:
            continue
        entry = by_id.get(channel)
        if entry is None:
            guard.observe_unknown(source, now, "no data")
            continue
        status = entry.get("status")
        verdict = entry.get("verdict")
        taken = _ultrasonic_taken_at(guard, source, entry, wall_now)
        if taken is None:
            guard.observe_unknown(source, now, f"no usable timestamp (status={status})")
        elif status == _US_BLIND:
            guard.observe_blind(source, taken, "ultrasonic blind zone")
        elif verdict == _US_UNKNOWN:
            guard.observe_unknown(source, taken, f"verdict unknown (status={status})")
        elif entry.get("valid") is True and status in (None, _US_OK):
            distance_cm = _finite(entry.get("distance_cm"))
            if distance_cm is None:
                guard.observe_unknown(source, taken, "valid reading without a distance")
            else:
                guard.observe_range(source, distance_cm / 100.0, taken)
        elif verdict == _US_CLEAR:
            horizon_cm = _finite(entry.get("clear_to_cm"))
            if horizon_cm is None or horizon_cm <= 0.0:
                guard.observe_unknown(
                    source,
                    taken,
                    "clear verdict without a usable horizon "
                    f"(clear_to_cm={entry.get('clear_to_cm')!r})",
                )
            else:
                guard.observe_clear(source, taken, horizon_cm / 100.0)
        else:
            guard.observe_unknown(source, taken, f"status={status} verdict={verdict}")


def sample_sensor_manager(guard: ObstacleGuard, manager: Any) -> None:
    """Push the sensor manager's cached readings into ``guard`` (no hardware access).

    A missing manager records nothing, so the observations age out and the guard holds.
    """
    if manager is None:
        return
    wall_now = datetime.now(UTC)
    _tof_observation(guard, manager, wall_now)
    _ultrasonic_observations(guard, manager, wall_now)


# ----------------------------------------------------------------------------- monitor

HoldHandler = Callable[[ObstacleAssessment], Awaitable[Any]]


class ObstacleStopMonitor:
    """Fixed-rate sampler + interlock publisher + active hold (see the module docstring).

    It is the production caller of ``SafetyTriggerManager.trigger_obstacle`` /
    ``trigger_ultrasonic_obstacle``. While the guard says STOP it awaits every
    ``hold_handlers`` entry each cycle (they are idempotent and cheap when nothing moves), so
    a plan that is already in motion is stopped without waiting for its next command.

    If this task dies or stalls the observations stop being refreshed and age out, so the
    guard itself turns to STOP: the failure is fail-closed.
    """

    HANDLER_TIMEOUT_S = 1.0

    def __init__(
        self,
        guard: ObstacleGuard,
        sensor_provider: Callable[[], Any],
        *,
        hold_handlers: Sequence[HoldHandler] = (),
        triggers: SafetyTriggerManager | None = None,
        interval_s: float = 0.05,
    ) -> None:
        self.guard = guard
        self._sensor_provider = sensor_provider
        self._handlers = tuple(hold_handlers)
        self._triggers = triggers or get_safety_trigger_manager()
        self._interval_s = float(interval_s)
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._last_status: ObstacleStatus | None = None
        self._log_at: dict[str, float] = {}
        self.cycles = 0
        self.errors = 0
        self.last_assessment: ObstacleAssessment | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_evt.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                await self.step()
            except Exception:
                self.errors += 1
                self._log_limited(
                    "step", logging.ERROR, "Obstacle stop monitor cycle failed", exc_info=True
                )
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass

    async def step(self) -> ObstacleAssessment:
        """One cycle: sample, assess, publish interlocks, hold if required."""
        sample_sensor_manager(self.guard, self._sensor_provider())
        assessment = self.guard.assess()
        self.last_assessment = assessment
        self.cycles += 1
        self._sync_interlocks(assessment)
        if assessment.status is not self._last_status:
            level = logging.INFO if assessment.status is ObstacleStatus.CLEAR else logging.WARNING
            logger.log(
                level,
                "Obstacle guard %s -> %s (cap %.2f m/s): %s",
                self._last_status.value if self._last_status else "start",
                assessment.status.value,
                assessment.speed_cap_mps,
                "; ".join(assessment.reasons) or "path clear",
            )
            self._last_status = assessment.status
        if assessment.hold:
            await self._hold(assessment)
        return assessment

    def _sync_interlocks(self, a: ObstacleAssessment) -> None:
        """Publish obstacle verdicts as safety interlocks (RobotState, ``system.safety``)."""
        hold_m = self.guard.model.hold_distance()
        tof = [d for s, d in a.obstacles if s.startswith("tof")]
        us = [(s, d) for s, d in a.obstacles if s.startswith(ULTRASONIC_PREFIX)]
        if any(s.startswith("tof") for s in a.obstacle_sources):
            d = min((x for x in tof if x is not None), default=0.0)
            self._triggers.trigger_obstacle(d, max(d, hold_m))
        else:
            self._triggers.clear_obstacle()
        if us:
            known = [x for _, x in us if x is not None]
            d_cm = 100.0 * (min(known) if known else 0.0)
            who = ",".join(s.removeprefix(ULTRASONIC_PREFIX) for s, _ in us)
            self._triggers.trigger_ultrasonic_obstacle(d_cm, max(d_cm, 100.0 * hold_m), who)
        else:
            self._triggers.clear_ultrasonic_obstacle()

    async def _hold(self, a: ObstacleAssessment) -> None:
        for handler in self._handlers:
            try:
                await asyncio.wait_for(handler(a), timeout=self.HANDLER_TIMEOUT_S)
            except Exception:
                self.errors += 1
                self._log_limited(
                    "hold",
                    logging.ERROR,
                    "Obstacle hold handler failed (the speed cap still applies)",
                    exc_info=True,
                )

    def _log_limited(
        self, key: str, level: int, msg: str, *args: object, exc_info: bool = False
    ) -> None:
        now = time.monotonic()
        if now - self._log_at.get(key, -math.inf) >= _LOG_INTERVAL_S:
            self._log_at[key] = now
            logger.log(level, msg, *args, exc_info=exc_info)


# ------------------------------------------------------------------------- construction

_guard: ObstacleGuard | None = None


def build_obstacle_guard(limits: SafetyLimits) -> ObstacleGuard:
    """A guard configured from ``limits.yaml``, ToF sources registered (ultrasonic: by sampling)."""
    guard = ObstacleGuard(
        limits.obstacle_stop,
        tof_stop_m=limits.tof_obstacle_distance_meters,
        ultrasonic_stop_m=limits.ultrasonic_obstacle_distance_cm / 100.0,
    )
    for source in TOF_SOURCES:
        guard.register_source(source, TOF_KIND)
    if limits.obstacle_stop.clear_dwell_s == 0.0:
        logger.warning(
            "Obstacle stop: clear_dwell_s is 0, so a hold is released the moment one reading is "
            "clear (no hysteresis)"
        )
    if not limits.obstacle_stop.validated:
        logger.warning(
            "Obstacle stop: the stopping-distance model is UNVALIDATED (conservative defaults, "
            "audit M-43 not done); the hold distance at minimum speed is %.2f m",
            guard.model.hold_distance(),
        )
    return guard


def configure_obstacle_guard(limits: SafetyLimits) -> ObstacleGuard:
    """Create and install the process-wide guard (called once from the lifespan)."""
    global _guard
    _guard = build_obstacle_guard(limits)
    return _guard


def get_obstacle_guard() -> ObstacleGuard:
    """The process-wide guard; built from ``config/limits.yaml`` on first use."""
    global _guard
    if _guard is None:
        from ..core.config_loader import ConfigLoader

        _guard = build_obstacle_guard(ConfigLoader().get()[1])
    return _guard


__all__ = [
    "MANUAL_SOURCES",
    "ObstacleAssessment",
    "ObstacleGuard",
    "ObstacleStatus",
    "ObstacleStopMonitor",
    "build_obstacle_guard",
    "configure_obstacle_guard",
    "get_obstacle_guard",
    "limit_pair",
    "sample_sensor_manager",
]
