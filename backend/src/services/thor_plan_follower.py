"""Local consumer of the Thor plan (W7): the only road from a Thor plan to the actuators.

``ThorLinkService.current_plan()`` hands over authenticated, fresh waypoint windows. This
module turns them into motion and nothing else may: a plan never reaches an actuator
directly. Each leg is driven by ``NavigationService.go_to_waypoint`` (the same loop a
mission uses), which commands the mower through ``TractorControlService.apply``, so the
service's own interlocks stay in the path of every command: the E-stop latch, the blade
interlock latch (tilt), the command deadman (a command that is not refreshed returns the
levers to neutral in ``default_hold_ms``) and any obstacle limiter fitted to the lever
chokepoint ``set_levers`` or to ``go_to_waypoint``. The follower has no obstacle logic of its
own: an obstacle stop is the navigation loop's and the tractor service's, and it applies to a
plan leg exactly as to a mission leg.
This follower itself only ever issues *stand-down* commands (blade off, levers neutral),
never motion, and never arms, starts the engine or clears a latch: those stay with a human.

**Local gate, re-checked every control tick (5 Hz) before any command is sent.** A leg runs
only while all of these hold; the first failure soft-stops the mower and is counted in
``stand_downs`` (the reason strings are listed in ``docs/tractor-platform.md``):

* the link has a usable plan (not held, not paused, not aged out) and the plan still carries
  the leg being driven, unchanged (``plan_id``, ``start_index`` window, lat/lon/blade);
* the tractor is enabled, authorized, its engine is running, its actuators are healthy and
  neither the E-stop nor the blade interlock latch is set;
* the navigation mode is idle or manual (no mission, AI or return-home run owns the mower);
* position, heading and the GNSS reading are fresh (``max_position_age_s``) and the fix is
  one of ``require_fix`` (default ``rtk_fixed``); stale or blind counts as blocked;
* if a fence is configured, the mower is inside it and the straight leg to the target stays
  inside it and clear of every no-go zone (``ThorLinkService`` already checked the waypoints
  themselves; a line between two inside points can still leave a concave fence).

**Plan indices.** The Pi reports the first waypoint it has not reached as ``waypoint_index``
in every uplink. A new ``plan_id`` (or the first plan of a run) is adopted at its
``start_index``; after that the next window must still contain the Pi's own index, otherwise
the Thor and the Pi disagree and the mower stands still (``out_of_sync``) until they agree.
The speed is local (``speed_percent``), and the blade is requested only if ``allow_blade``
is set and the waypoint asks for it: the Thor cannot enable a blade the Pi forbids.

Not provided here: obstacle detections in the uplink (the detector pipeline has no feed into
this service yet), operator acknowledgement before resuming after a hold (an open decision,
SD-3 in the audit), and any measured stopping behaviour (SIM only).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from ..models.geofence import LatLng
from ..models.mission import MissionWaypoint
from ..models.navigation_state import Position
from ..models.thor_link import Downlink, Uplink, Waypoint
from ..models.tractor_control import LEVER_NEUTRAL_EPS
from ..nav.geofence_validator import GeofenceShape, contains, segment_within

if TYPE_CHECKING:
    from .navigation_service import NavigationService
    from .tractor_service import TractorControlService

logger = logging.getLogger(__name__)

# Label on the stand-down actuator calls, so capture logs do not call them "operator" input.
_SOURCE = "thor_follower"
_LOG_INTERVAL_S = 5.0
# Navigation modes in which nothing else owns the mower.
_FREE_NAV_MODES = frozenset({"idle", "manual"})
# The ``Uplink.fix`` vocabulary, weakest first; ``none`` is never an acceptable fix to drive on.
FIX_LABELS = ("none", "gps", "dgps", "rtk_float", "rtk_fixed")


class PlanSource(Protocol):
    """What the follower needs from ``ThorLinkService``."""

    def current_plan(self) -> Downlink | None: ...


@dataclass(frozen=True)
class GnssSnapshot:
    """The newest GNSS reading in the link's vocabulary, with its age."""

    lat: float | None
    lon: float | None
    fix: str  # one of FIX_LABELS
    age_s: float | None  # None: never read


GnssProvider = Callable[[], GnssSnapshot]


def link_fix_label(rtk_status: str | None, satellites: int | None = None) -> str:
    """Map a driver ``GpsReading.rtk_status`` onto ``Uplink.fix``.

    Anything that is not positively a fix (no fix, dead reckoning, manual or simulator
    receiver states) is ``none``. An empty status with at least 6 satellites counts as a
    plain ``gps`` fix, as the telemetry stream already treats it.
    """
    status = (rtk_status or "").upper()
    if "FIXED" in status:
        return "rtk_fixed"
    if "FLOAT" in status:
        return "rtk_float"
    if "DGPS" in status:
        return "dgps"
    if status == "GPS_FIX" or (status == "" and (satellites or 0) >= 6):
        return "gps"
    return "none"


def gnss_from_sensor_manager(sensor_manager: Any, now: datetime | None = None) -> GnssSnapshot:
    """Snapshot ``sensor_manager.gps.last_reading`` without touching the hardware.

    ``last_reading`` is sample-and-hold: when the receiver stops answering the interface
    keeps the old reading, so its *own* timestamp is what exposes a stalled GNSS feed.
    A missing manager or reading, non-finite or out-of-range coordinates all read as no fix.
    """
    reading = getattr(getattr(sensor_manager, "gps", None), "last_reading", None)
    none = GnssSnapshot(None, None, "none", None)
    if reading is None:
        return none
    lat, lon = getattr(reading, "latitude", None), getattr(reading, "longitude", None)
    if not (_finite(lat) and _finite(lon) and abs(lat) <= 90.0 and abs(lon) <= 180.0):
        return none
    stamp = getattr(reading, "timestamp", None)
    if not isinstance(stamp, datetime):
        return none
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    # abs(): a reading stamped in the future is as unusable as one from the past.
    age = abs(((now or datetime.now(UTC)) - stamp).total_seconds())
    fix = link_fix_label(getattr(reading, "rtk_status", None), getattr(reading, "satellites", None))
    return GnssSnapshot(float(lat), float(lon), fix, age)


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


@dataclass(frozen=True)
class FollowerConfig:
    """Local policy for following a plan. Every value here is the Pi's, never the Thor's."""

    allow_blade: bool = False
    # Share of ``NavigationService.max_speed`` (same scale as ``MissionWaypoint.speed``).
    # UNVALIDATED: the shipped default equals the mission default; size it on the bench.
    speed_percent: int = 50
    require_fix: tuple[str, ...] = ("rtk_fixed",)
    # Age limit for the GNSS reading and for the navigation feed that carries position and
    # heading. UNVALIDATED.
    max_position_age_s: float = 1.0
    poll_s: float = 0.1
    neutral_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be within 1..100")
        if self.max_position_age_s <= 0 or self.poll_s <= 0 or self.neutral_timeout_s <= 0:
            raise ValueError("max_position_age_s, poll_s, neutral_timeout_s must be > 0")
        if not self.require_fix or any(f not in FIX_LABELS[1:] for f in self.require_fix):
            raise ValueError(f"require_fix must be a non-empty subset of {FIX_LABELS[1:]}")


class FollowerState(StrEnum):
    IDLE = "idle"  # not commanding motion (see ``reason``)
    DRIVING = "driving"  # inside a leg of the navigation loop


class ThorPlanFollower:
    """Drives the accepted Thor plan through the local navigation path, behind a local gate.

    Args:
        nav: navigation service whose ``go_to_waypoint`` executes each leg.
        tractor: actuation service; used for gate inputs and stand-down commands only.
        gnss: returns the newest GNSS reading (see ``gnss_from_sensor_manager``).
        config: local policy (blade permission, speed, fix requirement, ages).
        fence: optional ``GeofenceShape`` (boundary, inward margin, no-go zones).

    ``bind`` attaches the link whose ``current_plan()`` is followed; until then the follower
    stays idle. ``step`` runs one iteration (``start``/``stop`` run it in a task).
    """

    def __init__(
        self,
        nav: NavigationService,
        tractor: TractorControlService,
        gnss: GnssProvider,
        config: FollowerConfig | None = None,
        fence: GeofenceShape | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._nav = nav
        self._tractor = tractor
        self._gnss = gnss
        self._cfg = config or FollowerConfig()
        self._fence = fence
        self._clock = clock
        self._link: PlanSource | None = None

        self.plan_id: str | None = None
        self.index: int | None = None  # first waypoint of the plan not yet reached
        self.reached = 0
        self.state = FollowerState.IDLE
        self.reason: str | None = "no_link_bound"
        self.stand_downs: Counter[str] = Counter()
        self.errors = 0
        self._driving = False  # this follower has commanded motion that is not yet confirmed off
        self._stopping = False
        self._task: asyncio.Task | None = None
        self._log_state: dict[str, float] = {}

    # ------------------------------ lifecycle ------------------------------

    def bind(self, link: PlanSource) -> None:
        if self._link is not None:
            raise RuntimeError("follower is already bound to a link")
        self._link = link

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop following: the running leg soft-stops at its next tick, then blade off."""
        self._stopping = True
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await self._stand_down()

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A bug here must not leave the mower driving on a command nobody refreshes:
                # stand down, say so, and back off instead of spinning on the same error.
                self.errors += 1
                self.state = FollowerState.IDLE
                self._note("follower_error")
                self._log_limited("error", logging.ERROR, "Thor plan follower failed", exc=True)
                await self._stand_down()
                await asyncio.sleep(1.0)
                continue
            await asyncio.sleep(self._cfg.poll_s)

    # --------------------------------- step ---------------------------------

    async def step(self) -> None:
        """One iteration: pick the target from the plan, check the gate, drive one leg."""
        if self._link is None:
            await self._idle("no_link_bound")
            return
        plan = self._link.current_plan()
        if plan is None:
            await self._idle("link_hold")
            return
        target = self._select_target(plan)
        if isinstance(target, str):
            await self._idle(target)
            return
        index, waypoint = target
        blocked = self._blocked(waypoint)
        if blocked is not None:
            await self._idle(blocked)
            return
        await self._drive(plan.plan_id, index, waypoint)

    def _select_target(self, plan: Downlink) -> tuple[int, Waypoint] | str:
        """The plan's waypoint for this Pi's own index, or the reason there is none."""
        count = len(plan.waypoints)
        if count == 0:
            return "empty_window"
        if self.plan_id != plan.plan_id or self.index is None:
            # First plan of the run, or the Thor switched plans: take the Thor's start.
            logger.info(
                "Thor plan %r adopted at index %d (%d waypoints in window)",
                plan.plan_id,
                plan.start_index,
                count,
            )
            self.plan_id, self.index = plan.plan_id, plan.start_index
        offset = self.index - plan.start_index
        if not 0 <= offset < count:
            # The Thor is ahead of (or too far behind) what this Pi has actually reached:
            # each side's idea of progress disagrees, so neither may drive on the other's word.
            self._log_limited(
                "sync",
                logging.WARNING,
                "Thor plan %r window %d..%d does not contain this Pi's index %d",
                plan.plan_id,
                plan.start_index,
                plan.start_index + count - 1,
                self.index,
            )
            return "out_of_sync"
        return self.index, plan.waypoints[offset]

    # ------------------------------ local gate ------------------------------

    def _blocked(self, waypoint: Waypoint) -> str | None:
        """Why the mower may not be driven toward ``waypoint`` right now (None: it may)."""
        t = self._tractor.get_state()
        if not t.enabled:
            return "tractor_disabled"
        if t.emergency_stop_active:
            return "estop"
        if t.interlock_latched:
            return "interlock_latched"
        if not t.authorized:
            return "not_authorized"
        if not t.engine_running:
            return "engine_not_running"
        if not t.actuator_ok:
            return "actuator_fault"
        nav = self._nav.navigation_state
        mode = getattr(nav.navigation_mode, "value", nav.navigation_mode)
        if mode not in _FREE_NAV_MODES:
            return f"nav_mode_{mode}"
        if nav.current_position is None:
            return "no_position"
        if nav.heading is None or not _finite(nav.heading):
            return "no_heading"
        if self._age_s(nav.timestamp) > self._cfg.max_position_age_s:
            return "nav_feed_stale"
        snap = self._gnss()
        if snap.age_s is None or snap.age_s > self._cfg.max_position_age_s:
            return "gnss_stale"
        if snap.fix not in self._cfg.require_fix:
            return f"fix_{snap.fix}"
        return self._fence_reason(nav.current_position, waypoint)

    def _fence_reason(self, here: Any, waypoint: Waypoint) -> str | None:
        if self._fence is None:
            return None
        try:
            start = LatLng(latitude=here.latitude, longitude=here.longitude)
            goal = LatLng(latitude=waypoint.lat, longitude=waypoint.lon)
            if not contains(self._fence, start):
                return "outside_fence"
            if not segment_within(self._fence, start, goal):
                return "leg_blocked"
        except Exception:
            # Cannot prove the leg is inside: do not drive it.
            self._log_limited("fence", logging.ERROR, "Thor plan fence check failed", exc=True)
            return "fence_check_failed"
        return None

    @staticmethod
    def _age_s(stamp: datetime) -> float:
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return abs((datetime.now(UTC) - stamp).total_seconds())

    def _make_gate(self, plan_id: str, index: int, waypoint: Waypoint) -> Callable[[], bool]:
        """The per-tick gate handed to ``go_to_waypoint``: False soft-stops the leg."""
        leg = (waypoint.lat, waypoint.lon, bool(waypoint.blade))

        def gate() -> bool:
            try:
                if self._stopping:
                    return False
                link = self._link
                plan = link.current_plan() if link is not None else None
                if plan is None:
                    self._note("link_hold")
                    return False
                offset = index - plan.start_index
                if plan.plan_id != plan_id or not 0 <= offset < len(plan.waypoints):
                    self._note("plan_changed")
                    return False
                current = plan.waypoints[offset]
                if (current.lat, current.lon, bool(current.blade)) != leg:
                    self._note("target_changed")
                    return False
                blocked = self._blocked(current)
                if blocked is not None:
                    self._note(blocked)
                    return False
                return True
            except Exception:
                # A failing gate must stop the mower, not let it carry on unchecked.
                self.errors += 1
                self._log_limited("gate", logging.ERROR, "Thor plan gate failed", exc=True)
                self._note("gate_error")
                return False

        return gate

    # --------------------------------- drive --------------------------------

    async def _drive(self, plan_id: str, index: int, waypoint: Waypoint) -> None:
        blade = bool(waypoint.blade) and self._cfg.allow_blade
        leg = MissionWaypoint(
            lat=waypoint.lat,
            lon=waypoint.lon,
            blade_on=blade,
            speed=self._cfg.speed_percent,
        )
        self.state = FollowerState.DRIVING
        self._driving = True
        self._note(None)
        self._log_limited(
            "leg",
            logging.INFO,
            "Thor plan %r: driving to waypoint %d (%.7f, %.7f) blade=%s",
            plan_id,
            index,
            waypoint.lat,
            waypoint.lon,
            blade,
        )
        try:
            await self._nav.go_to_waypoint(
                leg, should_continue=self._make_gate(plan_id, index, waypoint)
            )
        finally:
            self.state = FollowerState.IDLE
        if self._arrived(waypoint) and self.plan_id == plan_id and self.index == index:
            self.index = index + 1
            self.reached += 1
            logger.info("Thor plan %r: reached waypoint %d", plan_id, index)
        if self.reason is not None:
            # The leg ended on a gate failure, not on arrival: stop the blade as well.
            await self._stand_down()

    def _arrived(self, waypoint: Waypoint) -> bool:
        here = self._nav.navigation_state.current_position
        if here is None:
            return False
        goal = Position(latitude=waypoint.lat, longitude=waypoint.lon)
        return self._nav.path_planner.calculate_distance(here, goal) <= self._nav.waypoint_tolerance

    # ------------------------------- stand down -------------------------------

    async def _idle(self, reason: str) -> None:
        self.state = FollowerState.IDLE
        self._note(reason)
        await self._stand_down()

    async def _stand_down(self) -> None:
        """Blade off, then levers neutral, if this follower is the one that moved the mower.

        Both are attempted and neither raises. Confirmation is read back from the tractor
        state (which only follows a successful write); an unconfirmed stand-down is retried
        on the next step. Nothing is commanded if the follower never drove: another operator
        path (teleop) may legitimately hold the mower.
        """
        if not self._driving:
            return
        t = self._tractor
        steps = (
            ("blade_off", lambda: t.engage_blade(False, source=_SOURCE)),
            ("levers_neutral", lambda: t.set_levers(0.0, 0.0, source=_SOURCE)),
        )
        for name, command in steps:
            try:
                await asyncio.wait_for(command(), timeout=self._cfg.neutral_timeout_s)
            except Exception as exc:
                expired = isinstance(exc, TimeoutError) and not exc.args
                self._log_limited(
                    f"neutral:{name}",
                    logging.ERROR,
                    "Thor plan stand-down %s failed (%s)",
                    name,
                    "timed out" if expired else repr(exc),
                    exc=not expired,
                )
        s = t.state
        if (
            not s.blade_engaged
            and abs(s.left_lever) <= LEVER_NEUTRAL_EPS
            and abs(s.right_lever) <= LEVER_NEUTRAL_EPS
        ):
            self._driving = False
        else:
            self._log_limited(
                "neutral_fault",
                logging.ERROR,
                "Thor plan stand-down NOT confirmed neutral (blade=%s left=%.2f right=%.2f)",
                s.blade_engaged,
                s.left_lever,
                s.right_lever,
            )

    # -------------------------------- uplink --------------------------------

    def build_uplink(self) -> Uplink:
        """The Pi's state for the Thor; ``ThorLinkService`` stamps sid/seq/time and link flags.

        Position and fix go out only while the GNSS reading is fresh, so a stalled receiver
        reads as ``none`` at the Thor, which then pauses instead of planning from old data.
        Heading comes from the navigation state. Detections are not provided (yet).
        """
        base = Uplink(seq=0, t_ms=0, waypoint_index=self.index)
        try:
            snap = self._gnss()
            heading = self._nav.navigation_state.heading
            update: dict[str, Any] = {}
            if (
                snap.age_s is not None
                and snap.age_s <= self._cfg.max_position_age_s
                and snap.lat is not None
                and snap.lon is not None
            ):
                update.update(lat=snap.lat, lon=snap.lon, fix=snap.fix)
            if _finite(heading):
                update["heading_deg"] = float(heading) % 360.0
            return base.model_copy(update=update)
        except Exception:
            # The heartbeat must go out even if a sensor read fails: the Thor pauses on "none".
            self._log_limited("uplink", logging.ERROR, "Thor uplink state failed", exc=True)
            return base

    # -------------------------------- status --------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "plan_id": self.plan_id,
            "index": self.index,
            "reached": self.reached,
            "driving": self._driving,
            "errors": self.errors,
            "stand_downs": dict(self.stand_downs),
        }

    # --------------------------------- logging -------------------------------

    def _note(self, reason: str | None) -> None:
        """Record why the mower is not being driven (None: it is); count each new cause."""
        if reason != self.reason:
            self.reason = reason
            if reason is not None:
                self.stand_downs[reason] += 1
                self._log_limited(
                    f"standdown:{reason}", logging.INFO, "Thor plan follower idle: %s", reason
                )

    def _log_limited(
        self, key: str, level: int, msg: str, *args: object, exc: bool = False
    ) -> None:
        """Log at most once per ``_LOG_INTERVAL_S`` per ``key``."""
        now = self._clock()
        last = self._log_state.get(key)
        if last is not None and now - last < _LOG_INTERVAL_S:
            return
        self._log_state[key] = now
        logger.log(level, msg, *args, exc_info=exc)


__all__ = [
    "FIX_LABELS",
    "FollowerConfig",
    "FollowerState",
    "GnssProvider",
    "GnssSnapshot",
    "PlanSource",
    "ThorPlanFollower",
    "gnss_from_sensor_manager",
    "link_fix_label",
]
