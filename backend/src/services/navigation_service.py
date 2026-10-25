"""
NavigationService for LawnBerry Pi v2
Path planning, navigation, and sensor fusion with safety constraints
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Optional

from ..core.simulation import is_simulation_mode
from ..models import (
    NavigationMode,
    NavigationState,
    Obstacle,
    PathStatus,
    Position,
    SensorData,
    Waypoint,
)
from ..nav.location_features import CoverageMap
from ..nav.path_planner import PathPlanner
from ..safety import stop_events
from ..safety.obstacle_stop import ObstacleStatus, limit_pair
from .robohat_service import get_robohat_service

logger = logging.getLogger(__name__)


## Path planning moved to backend.src.nav.path_planner.PathPlanner


class ObstacleDetector:
    """Obstacle detection and avoidance"""

    def __init__(self, safety_distance: float = 1.0):
        self.safety_distance = safety_distance
        self.detected_obstacles: list[Obstacle] = []

    def update_obstacles_from_sensors(self, sensor_data: SensorData) -> list[Obstacle]:
        """Update obstacle list from sensor data"""
        obstacles = []
        obstacle_id_counter = 0

        # ToF sensor obstacles
        if sensor_data.tof_left and sensor_data.tof_left.distance:
            if sensor_data.tof_left.distance < 1000:  # Less than 1 meter
                obstacles.append(
                    Obstacle(
                        id=f"tof_left_{obstacle_id_counter}",
                        position=Position(latitude=0, longitude=0),  # Relative position
                        confidence=0.8,
                        obstacle_type="static",
                        detection_source="tof",
                    )
                )
                obstacle_id_counter += 1

        if sensor_data.tof_right and sensor_data.tof_right.distance:
            if sensor_data.tof_right.distance < 1000:  # Less than 1 meter
                obstacles.append(
                    Obstacle(
                        id=f"tof_right_{obstacle_id_counter}",
                        position=Position(latitude=0, longitude=0),  # Relative position
                        confidence=0.8,
                        obstacle_type="static",
                        detection_source="tof",
                    )
                )
                obstacle_id_counter += 1

        self.detected_obstacles = obstacles
        return obstacles

    def is_path_clear(self, current_pos: Position, target_pos: Position) -> bool:
        """Check if path to target is clear of obstacles"""
        # Simple obstacle checking - would be more sophisticated in real implementation
        for obstacle in self.detected_obstacles:
            if obstacle.confidence > 0.5:
                # Check if obstacle is near the path
                return False
        return True


class DeadReckoningSystem:
    """Dead reckoning navigation fallback"""

    def __init__(self):
        self.last_gps_position: Position | None = None
        self.last_gps_time: datetime | None = None
        self.estimated_position: Position | None = None
        self.drift_estimate: float = 0.0
        self.active = False

    def update_gps_reference(self, gps_position: Position):
        """Update GPS reference for dead reckoning"""
        self.last_gps_position = gps_position
        self.last_gps_time = datetime.now(UTC)
        self.estimated_position = gps_position
        self.active = False
        self.drift_estimate = 0.0

    def estimate_position(self, heading: float, distance_traveled: float) -> Position | None:
        """Estimate current position using dead reckoning"""
        if not self.last_gps_position:
            # Initialize a local frame at origin if no GPS reference exists yet.
            # This enables dead-reckoning operation even before first GPS fix.
            self.last_gps_position = Position(latitude=0.0, longitude=0.0, accuracy=10.0)
            self.last_gps_time = datetime.now(UTC)
            self.estimated_position = self.last_gps_position

        self.active = True

        # Simple dead reckoning calculation
        lat_offset = distance_traveled * math.cos(math.radians(heading)) / 111000
        lon_offset = distance_traveled * math.sin(math.radians(heading)) / (111000 * 0.7)

        self.estimated_position = Position(
            latitude=self.last_gps_position.latitude + lat_offset,
            longitude=self.last_gps_position.longitude + lon_offset,
            accuracy=max(3.0, distance_traveled * 0.1),  # Increasing uncertainty
        )

        # Update drift estimate
        time_since_gps = (datetime.now(UTC) - self.last_gps_time).total_seconds()
        self.drift_estimate = min(distance_traveled * 0.05, time_since_gps * 0.1)

        return self.estimated_position


NAV_SOURCE = "navigation"  # command-source label for the tractor service (autonomous)
AI_SOURCE = "ai"
AUTONOMOUS_MODES = frozenset(
    {
        NavigationMode.AUTO,
        NavigationMode.AI,
        NavigationMode.AI_ASSISTED,
        NavigationMode.RETURN_HOME,
    }
)

#: A heading older than this (seconds) is not steered by (audit SAF-14): the loops hold.
#: UNVALIDATED placeholder: the IMU yaw arrives at several Hz, so two seconds is many missed
#: readings. Tune on the bench.
DEFAULT_MAX_HEADING_AGE_S = 2.0
#: A mission whose drive commands fail this many legs in a row is ended (blade off, neutral)
#: instead of retrying for ever.
MAX_CONSECUTIVE_FAILED_LEGS = 25
#: How long an aborted mission's cancelled task may take to unwind before the platform is
#: stopped anyway (its last write must not land after the stop's neutral).
ABORT_UNWIND_TIMEOUT_S = 1.0
#: A blade-on that failed is not retried more often than this (the RC to USB handshake behind
#: it can take seconds, and the control loop must keep its period).
BLADE_RETRY_S = 2.0


class LegResult(StrEnum):
    """How ``go_to_waypoint`` ended."""

    ARRIVED = "arrived"  # reached the waypoint; the path goes on
    FINAL = "final"  # reached the last waypoint of the path: the run is complete
    HALTED = "halted"  # the caller's gate said stop (a plan leg that belongs to no mission)
    INTERRUPTED = "interrupted"  # the mission was paused or aborted
    STOPPED = "stopped"  # a safety function stopped the machine while the leg ran
    REFUSED = "refused"  # the actuators refuse autonomous commands (not authorized, ...)
    FAILED = "failed"  # a drive command failed


@dataclass(frozen=True, slots=True)
class MotionGate:
    """The combined safety verdict autonomous motion must obey this tick."""

    hold: bool  # levers neutral, blade off
    speed_cap_mps: float | None  # forward ground-speed cap; None = no guard attached
    reasons: tuple[str, ...]


class NavigationService:
    """Main navigation service with sensor fusion and path planning"""

    def __init__(self, weather=None):
        self.navigation_state = NavigationState()
        self.path_planner = PathPlanner()
        self.obstacle_detector = ObstacleDetector()
        self.dead_reckoning = DeadReckoningSystem()
        # Optional weather service with get_current() and get_planning_advice()
        self.weather = weather

        # Navigation parameters
        self.max_speed = 0.8  # m/s
        self.cruise_speed = 0.5  # m/s
        self.waypoint_tolerance = 0.5  # meters
        self.obstacle_avoidance_distance = 1.0  # meters
        # Tractor: steady engine RPM held during autonomous mowing (not driven
        # by this controller). ponytail: bench/field value, tune on hardware.
        self.tractor_engine_throttle = 0.75

        # State tracking
        self.total_distance = 0.0
        self.last_position: Position | None = None

        # Autonomous AI inference loop state
        self._ai_task: asyncio.Task | None = None
        self._coverage: CoverageMap | None = None
        self._ai_epoch: int | None = None  # stop_events epoch when the AI run started
        self._ai_datum: tuple[float, float] | None = None
        self._ai_loop_hz = 5.0  # inference cadence (Hz)

        # Safety inputs, attached by the lifespan (see attach_safety). None = not gated:
        # unit/bench configuration only; main.py never serves a mission without them.
        self.obstacle_guard: Any | None = None  # safety.obstacle_stop.ObstacleGuard
        self.gps_monitor: Any | None = None  # nav.gps_degradation.GPSDegradationMonitor
        # Control-loop period of go_to_waypoint (5 Hz); a knob so tests can run the loop fast.
        self.control_period_s = 0.2
        self._hold_reasons: tuple[str, ...] | None = None  # set while motion is held
        # Differential platform blade: what this service last commanded on the RoboHAT blade
        # output (None = unknown: the next wanted state is sent), and when a blade command
        # that failed may be tried again.
        self._robohat_blade_on: bool | None = None
        self._robohat_blade_retry_at = 0.0
        # True while execute_mission's loop owns the run (see _update_path_execution).
        self._mission_loop_active = False
        # Heading age (audit SAF-14): the limit, and the IMU reading last taken as a new one.
        self.max_heading_age_s = DEFAULT_MAX_HEADING_AGE_S
        self._last_imu_stamp: datetime | None = None
        self._last_gps_stamp: datetime | None = None
        # Every safety stop ends the autonomous runs of this service (audit SAF-01). Held
        # weakly: a service that is dropped is not kept alive, or called, by the registry.
        stop_events.add_listener(self._on_stop_event)
        self._hold_error_logged = False
        self._last_hold_assert = 0.0

    _instance: Optional["NavigationService"] = None

    @classmethod
    def get_instance(cls, weather=None) -> "NavigationService":
        if cls._instance is None:
            cls._instance = NavigationService(weather=weather)
        return cls._instance

    async def initialize(self) -> bool:
        """Initialize navigation service"""
        logger.info("Initializing navigation service")
        self.navigation_state.navigation_mode = NavigationMode.IDLE
        return True

    # ------------------------------ safety gate ------------------------------

    def attach_safety(
        self, *, obstacle_guard: Any | None = None, gps_monitor: Any | None = None
    ) -> None:
        """Attach the safety inputs every autonomous motion path consults.

        ``obstacle_guard`` is a ``safety.obstacle_stop.ObstacleGuard``; ``gps_monitor`` a
        ``nav.gps_degradation.GPSDegradationMonitor`` (fed from ``update_navigation_state``).
        Both replace what was attached before, so passing neither detaches both. Without them
        nothing is gated: that is the unit/bench configuration only, the lifespan always
        attaches them.
        """
        self.obstacle_guard = obstacle_guard
        self.gps_monitor = gps_monitor

    def autonomy_active(self) -> bool:
        """True while a mission/AI run is the thing moving the machine."""
        return self.navigation_state.navigation_mode in AUTONOMOUS_MODES

    def heading_age_s(self) -> float | None:
        """Seconds since the IMU measured the current heading, ``None`` when it never did.

        The age comes from the reading's own timestamp (audit SAF-14): a reading the sensor
        interface replays after the IMU died keeps its old stamp, so a frozen heading ages
        instead of looking fresh on every call. A stamp in the future counts as that many
        seconds old: a clock that stepped back must not make a frozen heading look fresh.
        """
        stamp = self.navigation_state.heading_updated_at
        if stamp is None:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return abs((datetime.now(UTC) - stamp).total_seconds())

    def _motion_gate(self, *, gnss: bool = True) -> "MotionGate":
        """One verdict from every safety input; a check that cannot run is a hold.

        ``gnss=False`` leaves the GNSS monitor out (the IMU calibration drive does not
        navigate); the obstacle guard and the heading age always count.
        """
        hold = False
        cap: float | None = None
        reasons: list[str] = []
        guard = self.obstacle_guard
        if guard is not None:
            try:
                assessment = guard.assess()
            except Exception as exc:
                logger.exception("Obstacle guard failed: holding autonomous motion")
                hold, cap = True, 0.0
                reasons.append(f"obstacle guard failed: {exc!r}")
            else:
                cap = assessment.speed_cap_mps
                hold = assessment.hold
                if assessment.status is not ObstacleStatus.CLEAR:
                    reasons.extend(assessment.reasons)
        monitor = self.gps_monitor if gnss else None
        if monitor is not None:
            try:
                fix = monitor.assess()
            except Exception as exc:
                logger.exception("GNSS monitor failed: holding autonomous motion")
                hold, cap = True, 0.0
                reasons.append(f"gnss monitor failed: {exc!r}")
            else:
                if not fix.healthy:
                    hold, cap = True, 0.0
                    reasons.extend(f"gnss: {reason}" for reason in fix.reasons)
        age = self.heading_age_s()
        if age is not None and age > self.max_heading_age_s:
            # A frozen heading is steered by for ever by a proportional controller: hold.
            hold, cap = True, 0.0
            reasons.append(f"heading: stale ({age:.1f}s old, limit {self.max_heading_age_s:.1f}s)")
        return MotionGate(hold, cap, tuple(reasons))

    def motion_gate(self, *, gnss: bool = True) -> "MotionGate":
        """The current combined safety verdict (the public face of ``_motion_gate``).

        Read-only status routes use it to report holds as interlocks; calling it has the same
        side effects as the loops' own gate (the guards advance their dwell clocks), which are
        meant to be evaluated often.
        """
        return self._motion_gate(gnss=gnss)

    # ---------------------------- stops end autonomous runs ----------------------------

    def _on_stop_event(self, reason: str, source: str) -> None:
        """``safety.stop_events`` listener: every safety stop ends this service's runs."""
        self.abort_autonomy(f"{reason} ({source})" if source else reason)

    def abort_autonomy(self, reason: str) -> list[str]:
        """End every autonomous run this service owns, now (audit SAF-01); never raises.

        Called by ``safety.stop_events`` in the same event-loop turn as an E-stop, a
        watchdog trip, a deadman escalation, a tilt latch, a lock-out or a revoke, and by the
        loops themselves when the actuators refuse them for good. Synchronous and cooperative:
        it marks the running and paused missions ``aborted``, leaves the navigation mode
        ``idle`` and zeroes the target speed, and the mission and AI loops end at their next
        check (their very next one: the status and mode are read at the top of every tick, and
        the stop epoch before every command). It cancels no task: a task cancelled in the
        middle of an I2C write releases the actuation lock while the write is still on the bus,
        which is exactly what the stop's re-assert-after-the-lock logic exists to prevent, and
        the loop that triggered the stop may be the one being aborted. It commands no actuator:
        the stop that called it owns the outputs.

        The mode becomes ``idle``, not ``emergency_stop``: the latches, not the mode, carry the
        stop, and a Thor plan follower needs ``idle``/``manual`` to resume after a human
        cleared the latch and authorized again.
        """
        ended: list[str] = []
        try:
            from .mission_service import current_mission_service

            service = current_mission_service()
            if service is not None and service.nav_service is self:
                ended += [f"mission:{mission_id}" for mission_id in service.abort_all(reason)]
        except Exception:
            logger.exception("Aborting the missions failed")
        task = self._ai_task
        if task is not None and not task.done():
            ended.append("ai")
        state = self.navigation_state
        mode = state.navigation_mode
        if mode in AUTONOMOUS_MODES or mode == NavigationMode.PAUSED:
            if not ended:
                ended.append(f"mode:{getattr(mode, 'value', mode)}")
            state.navigation_mode = NavigationMode.IDLE
        if ended:
            state.target_velocity = 0.0
            if state.path_status == PathStatus.EXECUTING:
                state.path_status = PathStatus.INTERRUPTED
            logger.warning("Autonomous run ended (%s): %s", reason, ", ".join(ended))
        return ended

    def _stop_latched(self) -> str | None:
        """Why no autonomous run may start now: a stop is latched somewhere, else ``None``.

        A run started under a latch would sit refused until the latch is cleared and then drive
        off by itself: the clear must be followed by a person starting the run.
        """
        from ..core.globals import _safety_state
        from .tractor_service import get_tractor_service

        reasons: list[str] = []
        if _safety_state.get("emergency_stop_active", False):
            reasons.append("the emergency stop is latched")
        tractor = get_tractor_service()
        state = getattr(tractor, "state", None)
        if getattr(tractor, "enabled", False) is True:
            if getattr(state, "emergency_stop_active", False) is True:
                reasons.append("the tractor emergency stop is latched")
            if getattr(state, "interlock_latched", False) is True:
                reasons.append(
                    f"the tractor blade interlock is latched ({getattr(state, 'latch_reason', '')})"
                )
        lockout = getattr(tractor, "lockout_reason", None)
        if isinstance(lockout, str):
            reasons.append(f"the tractor is locked out ({lockout})")
        robohat = get_robohat_service()
        if robohat is not None and getattr(robohat, "emergency_latched", False) is True:
            reasons.append("the RoboHAT stop hold is latched")
        return "; ".join(reasons) or None

    @staticmethod
    def _standing_refusal(result: Any) -> str | None:
        """The standing refusal in a tractor ``apply`` result, else ``None``.

        A standing refusal (E-stop active, not authorized, locked out, parked) lasts until a
        person acts: an autonomous run that gets one has lost the right to drive and ends
        instead of retrying every tick and driving off when the refusal is lifted.
        """
        from .tractor_service import STANDING_REFUSALS

        parts = result.get("results") if isinstance(result, dict) else None
        if not isinstance(parts, dict):
            return None
        for name in ("levers", "throttle", "blade"):
            part = parts.get(name)
            if (
                isinstance(part, dict)
                and part.get("status") == "rejected"
                and part.get("code") in STANDING_REFUSALS
            ):
                return f"{name}: {part.get('reason')}"
        return None

    # ------------------------------ blade handling ------------------------------

    async def hold_for_gnss(self, assessment: Any | None = None) -> None:
        """GNSS monitor hold handler: stop autonomy at the actuators while the fix is degraded.

        Level-triggered like the loops' own gate (blade off first, then levers neutral, both
        always attempted, platform-specific); the loops keep holding on their next tick and
        resume by themselves once the fix has been healthy for the resume dwell.
        """
        reasons = tuple(f"gnss: {r}" for r in getattr(assessment, "reasons", ())) or (
            "gnss: degraded",
        )
        await self._hold_motion(MotionGate(True, 0.0, reasons))

    async def _robohat_blade(self, on: bool) -> bool:
        """Command the differential platform's blade.

        ``False`` when there is no controller (or SIM_MODE) or the command did not go through.
        """
        if is_simulation_mode():
            return False
        robohat = get_robohat_service()
        if robohat is not None and robohat.running and robohat.status.serial_connected:
            return bool(await robohat.send_blade_command(bool(on)))
        return False

    async def _sync_robohat_blade(self, desired: bool) -> None:
        """Differential platform: bring the RoboHAT blade to ``desired`` (never raises).

        Called every control tick but only AFTER the safety gate has said clear (audit SAF-03:
        the leg's blade state used to be sent before the gate was consulted, so every mission
        started with a blade-on that the gate switched off a tick later). It sends only on a
        change, so a leg that goes on mowing does not re-send blade-on every tick, and a blade
        that a hold dropped is restored here for the current leg once the hold is released.
        """
        if self._robohat_blade_on == desired:
            return
        now = time.monotonic()
        if now < self._robohat_blade_retry_at:
            return
        try:
            sent = await self._robohat_blade(desired)
        except Exception:
            logger.warning(
                "Blade command failed or unavailable; continuing movement", exc_info=True
            )
            sent = False
        if sent:
            self._robohat_blade_on = desired
        else:
            self._robohat_blade_retry_at = now + BLADE_RETRY_S

    def _request_blade_off(self, why: str) -> None:
        """Ask the IBT-4 blade service to switch off and stay confirmed off (best effort).

        The IBT-4 is the blade driver the docs name, and no tractor or RoboHAT stop reaches
        it: every blade-off site here asks it too. A no-op for a service that is not armed.
        """
        try:
            from .blade_service import get_blade_service

            get_blade_service().request_blade_off(why)
        except Exception:
            logger.warning("Could not ask the IBT-4 blade service to switch off", exc_info=True)

    async def _drop_blade(self, tractor: Any, why: str) -> list[str]:
        """Blade off on every blade path this process has; returns what failed. Never raises."""
        failures: list[str] = []
        try:
            if tractor.enabled:
                await tractor.engage_blade(False, source=NAV_SOURCE)
            elif await self._robohat_blade(False):
                self._robohat_blade_on = False
        except Exception as exc:
            failures.append(f"blade off: {exc!r}")
        self._request_blade_off(why)
        return failures

    async def drop_blade(self, why: str) -> list[str]:
        """Switch the blade off first (the stop of a mission is blade, then drive)."""
        from .tractor_service import get_tractor_service

        return await self._drop_blade(get_tractor_service(), why)

    async def _neutral_drive(self) -> None:
        """``set_speed(0, 0)`` that leaves ``target_velocity`` (the AI loop's scale) alone."""
        target = self.navigation_state.target_velocity
        try:
            await self.set_speed(0.0, 0.0)
        finally:
            self.navigation_state.target_velocity = target

    async def _stand_down_platform(self, why: str, *, drop_blade: bool = True) -> list[str]:
        """Blade off FIRST, then the drive to neutral, on whichever platform is active.

        Both are always attempted and neither raises; returns what failed. ``drop_blade=False``
        leaves the blade alone (an intermediate waypoint, a plan leg whose follower owns it).
        """
        from .tractor_service import get_tractor_service

        tractor = get_tractor_service()
        failures = await self._drop_blade(tractor, why) if drop_blade else []
        try:
            if tractor.enabled:
                await tractor.set_levers(0.0, 0.0, source=NAV_SOURCE)
            else:
                await self._neutral_drive()
        except Exception as exc:
            failures.append(f"drive neutral: {exc!r}")
        return failures

    async def halt_platform(self, why: str) -> list[str]:
        """Stop the machine without ending a run's bookkeeping: blade off, then neutral."""
        return await self._stand_down_platform(why)

    async def _hold_motion(self, gate: "MotionGate") -> None:
        """Stop autonomous motion and keep it stopped: blade off first, then levers neutral.

        Level-triggered: called every control tick while the gate holds, so a failed stop
        is retried, never assumed. Both stops are always attempted. Errors are logged (once
        per hold) and never raised: the speed cap in the actuation chokepoints and the
        obstacle monitor still apply.
        """
        if self._hold_reasons is None:  # entering a hold (the reason text changes every tick)
            logger.warning("Autonomous motion HELD: %s", "; ".join(gate.reasons) or "hold")
            self._hold_error_logged = False
        self._hold_reasons = gate.reasons
        failures = await self._stand_down_platform("motion hold: " + "; ".join(gate.reasons))
        if failures and not self._hold_error_logged:
            self._hold_error_logged = True
            logger.error("Could not confirm the motion hold (retrying every tick): %s", failures)

    async def _release_hold(self, waypoint: Any | None = None) -> None:
        """End a hold. The differential platform's blade is restored for the current leg by
        ``_sync_robohat_blade`` on the same tick, once the gate is clear."""
        if self._hold_reasons is not None:
            logger.info("Autonomous motion released")
            self._hold_reasons = None

    async def enforce_motion_hold(self, assessment: Any | None = None) -> None:
        """Obstacle monitor hold handler for the differential platform.

        Stops whatever autonomy has moving the moment a hold begins, without waiting for the
        planner's next tick: blade off first, then the drive to neutral. The tractor platform
        has its own handler on the tractor service.
        """
        from .tractor_service import get_tractor_service

        if get_tractor_service().enabled:
            return
        if self.navigation_state.navigation_mode not in AUTONOMOUS_MODES:
            return
        now = time.monotonic()
        if now - self._last_hold_assert < 0.25:
            return
        self._last_hold_assert = now
        failures = await self._stand_down_platform("obstacle hold")
        if failures:
            logger.error("Could not confirm the obstacle hold (retried next cycle): %s", failures)

    async def execute_mission(self, mission: "Mission"):  # noqa: F821
        """Execute a mission by navigating to each waypoint.

        The mission ends (``completed``, mode ``idle``) when its last waypoint is reached, with
        the blade off and the drive neutral (audit SAF-02: the loop used to run on the last
        waypoint for ever, re-commanding blade-on). It ends early, blade off first, when it is
        paused or aborted, and for good when a safety function stops the machine
        (``safety.stop_events``: E-stop, watchdog, deadman escalation, tilt latch, revoke): a
        clear of that stop never re-activates it (audit SAF-01). It will not start while a stop
        is latched.
        """
        from .mission_service import get_mission_service
        from .tractor_service import get_tractor_service

        mission_service = get_mission_service()

        # A run that started under a latch would sit refused and drive off by itself at the
        # clear: after a stop, a person clears the latch AND starts the run again.
        if (latched := self._stop_latched()) is not None:
            raise RuntimeError(f"Cannot start mission: {latched}")

        # GNSS precondition (audit D.9): no mission starts on a missing, stale or inaccurate
        # fix, or on one that has not yet stayed healthy for the resume dwell. Checked before
        # the engine is cranked, so a refused mission has touched nothing.
        gnss = self.gps_monitor
        if gnss is not None:
            fix = gnss.assess()
            if not fix.healthy:
                raise RuntimeError(
                    "Cannot start mission: GNSS fix is not healthy (" + "; ".join(fix.reasons) + ")"
                )

        # Tractor: auto-start the engine before any state mutation, so a failed
        # crank aborts the mission cleanly instead of driving an unpowered
        # vehicle. Guarded on "not already running" so resume_mission()'s
        # re-invocation of execute_mission() doesn't try to re-crank a running
        # engine. Never auto-authorizes — an operator must have called
        # tractor.authorize() already, or this fails loudly with that reason.
        tractor = get_tractor_service()
        if tractor.enabled and not tractor.state.engine_running:
            result = await tractor.start_engine()
            if result.get("status") != "ok":
                reason = result.get("reason", "unknown reason")
                raise RuntimeError(
                    f"Cannot start mission: tractor engine failed to start ({reason})"
                )

        # Every command of this run is checked against the stop epoch captured here.
        run_epoch = stop_events.epoch()
        logger.info(f"Starting mission execution: {mission.id} - {mission.name}")
        self.navigation_state.navigation_mode = NavigationMode.AUTO
        self.navigation_state.path_status = PathStatus.EXECUTING
        self._robohat_blade_on = None  # unknown: the first clear tick sends the leg's state

        # Convert MissionWaypoints to Navigation Waypoints
        self.navigation_state.planned_path = [
            Waypoint(
                position=Position(latitude=wp.lat, longitude=wp.lon),
                target_speed=(wp.speed / 100.0 * self.max_speed),
            )
            for wp in mission.waypoints
        ]

        self.navigation_state.current_waypoint_index = 0
        self.navigation_state.operation_start_time = datetime.now(UTC)

        failed_legs = 0
        self._mission_loop_active = True
        try:
            while self.navigation_state.current_waypoint_index < len(
                self.navigation_state.planned_path
            ):
                if stop_events.epoch() != run_epoch:
                    self.abort_autonomy("a safety stop happened during the mission")
                    return
                status = mission_service.mission_statuses.get(mission.id)
                if not status or status.status != "running":
                    logger.warning(
                        f"Mission {mission.id} interrupted. Status: {status.status if status else 'N/A'}"  # noqa: E501
                    )
                    self.navigation_state.path_status = PathStatus.INTERRUPTED
                    self.navigation_state.navigation_mode = NavigationMode.IDLE
                    # Idempotent: a pause or abort already did this, a task that lost its
                    # status some other way did not (blade off first, then neutral).
                    await self._stop_platform()
                    return

                current_mission_waypoint = mission.waypoints[
                    self.navigation_state.current_waypoint_index
                ]
                leg = await self.go_to_waypoint(current_mission_waypoint, run_epoch=run_epoch)
                if leg is LegResult.FINAL:
                    break
                if leg in (LegResult.STOPPED, LegResult.REFUSED):
                    self.abort_autonomy(f"the mission leg ended: {leg.value}")
                    return
                failed_legs = failed_legs + 1 if leg is LegResult.FAILED else 0
                if failed_legs >= MAX_CONSECUTIVE_FAILED_LEGS:
                    raise RuntimeError(
                        f"Mission {mission.id} failed: the drive controller failed "
                        f"{failed_legs} legs in a row"
                    )

                # The go_to_waypoint method is blocking until the waypoint is reached or
                # interrupted. The loop will then proceed to the next waypoint. We add a small
                # sleep to prevent a tight loop if go_to_waypoint returns immediately.
                await asyncio.sleep(0.1)
        except Exception:
            self.navigation_state.path_status = PathStatus.FAILED
            self.navigation_state.navigation_mode = NavigationMode.IDLE
            await self._stop_platform()
            raise
        finally:
            self._mission_loop_active = False

        if self.navigation_state.navigation_mode == NavigationMode.AUTO:
            self.navigation_state.path_status = PathStatus.COMPLETED
            self.navigation_state.navigation_mode = NavigationMode.IDLE
            logger.info(f"Mission {mission.id} completed.")

    async def go_to_waypoint(
        self,
        waypoint: "MissionWaypoint",  # noqa: F821
        *,
        should_continue: Callable[[], bool] | None = None,
        run_epoch: int | None = None,
    ) -> LegResult:
        """Navigate to a single waypoint and block until arrival; returns how the leg ended.

        ``should_continue`` is an external gate for legs that belong to no mission (the Thor
        plan follower): it is polled at the top of every control tick, and the first time it
        returns False the platform is soft-stopped and the call returns without arriving.
        A gated leg ignores mission status, so a plan can never be steered by (or paused
        through) a mission it is not part of.

        ``run_epoch`` is the ``safety.stop_events`` epoch of the run the leg belongs to (a
        mission passes the one it captured at its start; a plan leg captures its own on
        entry). The first tick that sees it moved ends the leg with ``STOPPED`` without
        commanding anything, so a stop that landed while the leg waited on a slow command is
        never followed by one more motion command, and a clear of the stop cannot revive it.

        A missing position, a missing, non-finite or stale heading, or a held safety gate is a
        hold: blade off first, then levers neutral, every tick. The blade follows the
        waypoint's ``blade_on`` only while the gate is clear (differential platform: sent on a
        change after the first clear verdict, never before; tractor: every tick through
        ``apply``). The leg ends with the blade off when the last waypoint is reached or the
        mission is paused or aborted; an intermediate waypoint keeps the blade for the next
        leg (audit SAF-02, SAF-03).
        """
        from ..models.tractor_control import TractorCommand
        from .mission_service import get_mission_service
        from .tractor_service import get_tractor_service

        mission_service = get_mission_service() if should_continue is None else None
        tractor = get_tractor_service()
        epoch = stop_events.epoch() if run_epoch is None else run_epoch

        target_pos = Position(latitude=waypoint.lat, longitude=waypoint.lon)
        logger.info(f"Navigating to waypoint: {target_pos.latitude}, {target_pos.longitude}")
        desired_blade = bool(waypoint.blade_on)

        while True:
            if stop_events.epoch() != epoch:
                logger.warning("Waypoint navigation ended: a safety function stopped the machine.")
                return LegResult.STOPPED

            if should_continue is not None and not should_continue():
                logger.info("Waypoint navigation halted by the caller's gate.")
                await self._stop_platform(drop_blade=False)  # the plan follower owns the blade
                return LegResult.HALTED

            # Check for interruptions (pause/abort)
            mission_id = next(
                (
                    mid
                    for mid, m in (
                        mission_service.missions.items() if mission_service is not None else ()
                    )
                    if waypoint.id in [w.id for w in m.waypoints]
                ),
                None,
            )
            if mission_id:
                status = mission_service.mission_statuses.get(mission_id)
                if not status or status.status != "running":
                    logger.info("Waypoint navigation interrupted.")
                    await self._stop_platform()  # blade off first, then neutral
                    return LegResult.INTERRUPTED

            position = self.navigation_state.current_position
            if not position:
                # Nothing to navigate by: hold (blade off, neutral), do not leave the last
                # command, and the blade, in force while waiting for a fix.
                await self._hold_motion(MotionGate(True, 0.0, ("no position fix",)))
                await asyncio.sleep(self.control_period_s)
                continue

            distance_to_target = self.path_planner.calculate_distance(position, target_pos)

            if distance_to_target <= self.waypoint_tolerance:
                logger.info(f"Waypoint reached: {waypoint.lat}, {waypoint.lon}")
                if should_continue is not None:
                    # A plan leg belongs to no path: neutral only, the follower owns the blade.
                    await self._stop_platform(drop_blade=False)
                    return LegResult.ARRIVED
                state = self.navigation_state
                last = state.current_waypoint_index >= len(state.planned_path) - 1
                # The last waypoint ends the run: blade off first, then neutral. An
                # intermediate one only stops the drive; the next leg sets its own blade.
                await self._stop_platform(drop_blade=last)
                if last:
                    logger.info("Final waypoint in path reached.")
                    return LegResult.FINAL
                state.advance_waypoint()
                logger.info(
                    f"Advanced to next waypoint index: {state.current_waypoint_index}"  # noqa: E501
                )
                return LegResult.ARRIVED

            # Safety gate (obstacle guard, GNSS, heading age): hold or cap before anything is
            # commanded. A hold keeps the mission alive -- the loop keeps polling the mission
            # status above, so pause/abort still work -- and resumes by itself once the path
            # has stayed clear.
            gate = self._motion_gate()
            current_heading = self.navigation_state.heading
            if current_heading is None or not math.isfinite(current_heading):
                gate = MotionGate(True, 0.0, (*gate.reasons, "no usable heading"))
            if gate.hold:
                await self._hold_motion(gate)
                await asyncio.sleep(self.control_period_s)
                continue
            await self._release_hold(waypoint)

            # Calculate heading and apply motor commands
            heading_to_target = self.path_planner.calculate_bearing(position, target_pos)
            heading_error = (heading_to_target - current_heading + 180) % 360 - 180

            # Proportional controller for turning
            turn_effort = max(
                -1.0, min(1.0, heading_error / 45.0)
            )  # Normalize error to [-1, 1] over a 90 degree arc

            # Set speed (scale by waypoint speed preference when provided)
            base_speed = self.cruise_speed
            try:
                if hasattr(waypoint, "speed") and isinstance(waypoint.speed, int):
                    base_speed = max(
                        0.1, min(self.max_speed, (waypoint.speed / 100.0) * self.max_speed)
                    )
            except Exception:
                base_speed = self.cruise_speed

            forward_speed = base_speed
            if abs(heading_error) > 30:
                forward_speed *= 0.5  # Slow down for sharp turns

            # Same differential math for every platform: a zero-turn mower's
            # twin drive levers and a differential-drive mower's wheel motors
            # both just take independent left/right speed commands -- reuse it
            # rather than inventing per-platform control math.
            # ponytail: proportional heading-error control, not pure pursuit
            # (this codebase has no wheelbase/steering-angle config that would
            # make curvature math meaningful); revisit if field testing on
            # either platform shows oscillation.
            left_speed = forward_speed * (1 - turn_effort)
            right_speed = forward_speed * (1 + turn_effort)

            # Clamp speeds
            left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
            right_speed = max(-self.max_speed, min(self.max_speed, right_speed))

            # Obstacle guard: never faster than the stopping-distance model allows for the
            # range the sensors prove free (the actuation chokepoints enforce it again).
            if gate.speed_cap_mps is not None:
                left_speed, right_speed = limit_pair(left_speed, right_speed, gate.speed_cap_mps)

            if tractor.enabled:
                # TractorCommand's ge=/le= fields reject out-of-range values
                # instead of clamping, so normalize onto the lever range and
                # clamp here before construction.
                left_lever = max(-1.0, min(1.0, left_speed / self.max_speed))
                right_lever = max(-1.0, min(1.0, right_speed / self.max_speed))
                cmd = TractorCommand(
                    left_lever=left_lever,
                    right_lever=right_lever,
                    throttle=self.tractor_engine_throttle,
                    blade_engaged=desired_blade,
                )
                try:
                    result = await tractor.apply(cmd, source=NAV_SOURCE)
                except Exception as e:
                    logger.error(f"Failed to apply tractor command: {e}")
                    try:
                        await tractor.emergency_stop()
                    except Exception:
                        pass
                    return LegResult.FAILED

                refusal = self._standing_refusal(result)
                if refusal is not None:
                    # E-stop, no authorization, lock-out or park: nothing autonomous may move
                    # the machine until a person acts. End the run (a lifted refusal must not
                    # let it drive off), and put the blade and the drive in the safe state.
                    logger.warning("Autonomous run ended: the tractor refuses it (%s)", refusal)
                    self.abort_autonomy(f"the tractor refuses autonomous commands: {refusal}")
                    await self._stop_platform()
                    return LegResult.REFUSED

                # Simulation of movement
                await asyncio.sleep(self.control_period_s)  # Control loop at 5Hz
                continue

            try:
                # Only now, with the gate clear (audit SAF-03), and only on a change.
                await self._sync_robohat_blade(desired_blade)
                await self.set_speed(left_speed, right_speed)
            except Exception as e:
                logger.error(f"Failed to set motor speed: {e}")
                if stop_events.epoch() != epoch:
                    return LegResult.STOPPED
                try:
                    await self.set_speed(0.0, 0.0)
                except Exception:
                    pass
                return LegResult.FAILED

            # Simulation of movement
            await asyncio.sleep(self.control_period_s)  # Control loop at 5Hz

    async def set_speed(self, left_speed: float, right_speed: float) -> None:
        """Drive command helper integrating with the RoboHAT controller.

        - Accepts normalized wheel speeds in m/s-like units scaled to [-1, 1].
        - In SIM_MODE, updates state without touching hardware.
        """
        # Clamp inputs for safety
        ls = max(-self.max_speed, min(self.max_speed, float(left_speed)))
        rs = max(-self.max_speed, min(self.max_speed, float(right_speed)))

        # Differential platform's chokepoint (the tractor has its own, in
        # TractorControlService.set_levers): never faster than the obstacle guard allows,
        # and neutral while it holds. Stops (0, 0) pass through untouched.
        gate = self._motion_gate()
        if gate.hold:
            ls = rs = 0.0
        elif gate.speed_cap_mps is not None:
            ls, rs = limit_pair(ls, rs, gate.speed_cap_mps)

        # Update desired velocity estimate (magnitude)
        try:
            self.navigation_state.target_velocity = float(max(abs(ls), abs(rs)))
        except Exception:
            self.navigation_state.target_velocity = None

        # SIM_MODE: don't touch hardware
        if is_simulation_mode():
            # Roughly reflect commanded speed in reported velocity for UI
            try:
                self.navigation_state.velocity = float((abs(ls) + abs(rs)) / 2.0)
            except Exception:
                pass
            return

        robohat = get_robohat_service()
        if robohat is None or not robohat.running or not robohat.status.serial_connected:
            raise RuntimeError("RoboHAT controller unavailable")

        # Normalize to [-1, 1] before mixing to PWM on RoboHAT layer
        def _normalize(v: float) -> float:
            if self.max_speed <= 0:
                return 0.0
            return max(-1.0, min(1.0, v / self.max_speed))

        accepted = await robohat.send_motor_command(_normalize(ls), _normalize(rs))
        if not accepted:
            raise RuntimeError("Motor command not accepted by controller")

    async def _stop_platform(self, *, drop_blade: bool = True) -> None:
        """Soft stop: blade off first (unless ``drop_blade`` is False), then zero ground speed.

        Use at *expected* stop points (the last waypoint, a mission paused or aborted, a plan
        leg halted). Unlike ``emergency_stop()``, this never revokes authorization and needs no
        operator ``clear_emergency()`` afterward -- reserve the hard stop for an actual command
        failure. Both steps are always attempted and nothing is raised (audit SAF-02): the
        callers are control loops that must go on to end their run. The platform is the active
        one (tractor levers, or the differential drive and its RoboHAT blade); the IBT-4 blade
        service is asked to switch off too.
        """
        failures = await self._stand_down_platform("platform stop", drop_blade=drop_blade)
        if failures:
            logger.error("Platform stop not fully confirmed: %s", failures)

    async def update_navigation_state(self, sensor_data: SensorData) -> NavigationState:
        """Update navigation state with sensor fusion"""

        # Feed the GNSS monitor the real reading (None this tick = nothing new to observe).
        if self.gps_monitor is not None:
            try:
                self.gps_monitor.observe(sensor_data.gps)
            except Exception:
                logger.exception("GNSS monitor could not take the reading (it will go stale)")

        # Update position from GPS or dead reckoning. A fix is NEW only when its own timestamp
        # changed: a reading the sensor interface replays after the receiver went quiet keeps
        # its stamp.
        gps = sensor_data.gps
        gps_stamp = getattr(gps, "timestamp", None)  # a reading without a stamp is always new
        new_fix = bool(
            gps is not None
            and gps.latitude
            and gps.longitude
            and (gps_stamp is None or gps_stamp != self._last_gps_stamp)
        )
        if new_fix:
            self._last_gps_stamp = gps_stamp
        current_position = await self._update_position(sensor_data)
        if current_position:
            self.navigation_state.current_position = current_position

        # Update heading from IMU, with the age of the measurement (audit SAF-14): only an IMU
        # reading that is new (its own timestamp changed) is a measurement. The dead IMU's
        # replayed ``last_reading`` keeps its stamp, so the heading ages and the loops hold
        # (``_motion_gate``) instead of steering by it for ever.
        imu = sensor_data.imu
        imu_stamp = getattr(imu, "timestamp", None)
        new_heading = False
        if (
            imu is not None
            and imu.yaw is not None
            and (imu_stamp is None or imu_stamp != self._last_imu_stamp)
        ):
            self._last_imu_stamp = imu_stamp
            self.navigation_state.heading = imu.yaw
            self.navigation_state.heading_updated_at = (
                imu_stamp if isinstance(imu_stamp, datetime) else datetime.now(UTC)
            )
            new_heading = True

        # Update obstacles
        obstacles = self.obstacle_detector.update_obstacles_from_sensors(sensor_data)
        self.navigation_state.obstacle_map = obstacles

        # Check if obstacle avoidance is needed
        gate = self._motion_gate()
        self.navigation_state.obstacle_avoidance_active = len(obstacles) > 0 or bool(gate.reasons)

        # Update path execution if in auto mode
        if self.navigation_state.navigation_mode == NavigationMode.AUTO:
            await self._update_path_execution()

        # Update distance tracking
        if self.last_position and current_position:
            distance_increment = self.path_planner.calculate_distance(
                self.last_position, current_position
            )
            self.navigation_state.distance_traveled += distance_increment
            self.total_distance += distance_increment

        self.last_position = current_position
        if new_fix or new_heading:
            # The navigation feed's stamp moves only when something new arrived: it used to
            # be refreshed on every call, whatever (or nothing) the call carried (SAF-14).
            self.navigation_state.timestamp = datetime.now(UTC)

        return self.navigation_state

    async def _update_position(self, sensor_data: SensorData) -> Position | None:
        """Update position using sensor fusion"""

        # Primary: GPS position
        if sensor_data.gps and sensor_data.gps.latitude and sensor_data.gps.longitude:
            gps_position = Position(
                latitude=sensor_data.gps.latitude,
                longitude=sensor_data.gps.longitude,
                altitude=sensor_data.gps.altitude,
                accuracy=sensor_data.gps.accuracy,
            )

            # Update dead reckoning reference
            self.dead_reckoning.update_gps_reference(gps_position)
            self.navigation_state.dead_reckoning_active = False
            # The fix's own timestamp, not "now": a reading the sensor interface replayed
            # after the receiver went quiet must not look like a fresh fix.
            fix_stamp = getattr(sensor_data.gps, "timestamp", None)
            self.navigation_state.last_gps_fix = (
                fix_stamp if isinstance(fix_stamp, datetime) else datetime.now(UTC)
            )

            return gps_position

        # Fallback: Dead reckoning
        elif self.navigation_state.heading is not None:
            # Estimate distance traveled since last update (placeholder)
            distance_traveled = 0.1  # meters, would be calculated from wheel encoders

            dead_reckoning_pos = self.dead_reckoning.estimate_position(
                self.navigation_state.heading, distance_traveled
            )

            if dead_reckoning_pos:
                self.navigation_state.dead_reckoning_active = True
                self.navigation_state.dead_reckoning_drift = self.dead_reckoning.drift_estimate
                return dead_reckoning_pos

        return None

    async def _update_path_execution(self):
        """Update path execution for autonomous navigation.

        While a mission loop owns the run (``execute_mission``), ``go_to_waypoint`` is the
        only thing that advances the path and ends it: this telemetry-driven bookkeeping used
        to advance the same index at the same time (a waypoint skipped when both saw the
        arrival) and, at the last waypoint, flip the mode to ``idle`` while the mission loop
        went on commanding the machine (audit SAF-02). It then only keeps the target speed.
        """
        if not self.navigation_state.planned_path:
            return

        current_waypoint = self.navigation_state.get_current_waypoint()
        if not current_waypoint or not self.navigation_state.current_position:
            return

        # Check if we've reached the current waypoint
        distance_to_waypoint = self.path_planner.calculate_distance(
            self.navigation_state.current_position, current_waypoint.position
        )

        if distance_to_waypoint <= self.waypoint_tolerance and not self._mission_loop_active:
            # Advance to next waypoint
            if self.navigation_state.advance_waypoint():
                logger.info(f"Advanced to waypoint {self.navigation_state.current_waypoint_index}")
            else:
                # Reached end of path
                self.navigation_state.path_status = PathStatus.COMPLETED
                self.navigation_state.navigation_mode = NavigationMode.IDLE
                logger.info("Path execution completed")

        # Update target velocity based on conditions
        if self.navigation_state.obstacle_avoidance_active:
            self.navigation_state.target_velocity = 0.0  # Stop for obstacles
        else:
            self.navigation_state.target_velocity = (
                current_waypoint.target_speed or self.cruise_speed
            )

    async def plan_path(
        self, boundaries: list[Position], cutting_pattern: str = "parallel"
    ) -> bool:
        """Plan a mowing path for the given boundaries"""
        logger.info(
            f"Planning {cutting_pattern} path for area with {len(boundaries)} boundary points"
        )

        if cutting_pattern == "parallel":
            waypoints = self.path_planner.generate_parallel_lines_path(boundaries)
        else:
            logger.warning(f"Unsupported cutting pattern: {cutting_pattern}")
            return False

        if waypoints:
            self.navigation_state.planned_path = waypoints
            self.navigation_state.current_waypoint_index = 0
            self.navigation_state.path_status = PathStatus.PLANNED
            self.navigation_state.path_confidence = 0.9

            # Calculate estimated area and time
            self._calculate_path_estimates()

            logger.info(f"Path planned with {len(waypoints)} waypoints")
            return True

        return False

    def _calculate_path_estimates(self):
        """Calculate estimated completion time and area coverage"""
        if not self.navigation_state.planned_path:
            return

        total_distance = 0.0
        for i in range(1, len(self.navigation_state.planned_path)):
            total_distance += self.path_planner.calculate_distance(
                self.navigation_state.planned_path[i - 1].position,
                self.navigation_state.planned_path[i].position,
            )

        # Estimate completion time
        estimated_time_seconds = total_distance / self.cruise_speed
        self.navigation_state.estimated_completion_time = datetime.now(UTC) + timedelta(
            seconds=estimated_time_seconds
        )

    async def start_autonomous_navigation(self) -> bool:
        """Start autonomous navigation"""
        if (latched := self._stop_latched()) is not None:
            logger.error("Cannot start navigation: %s", latched)
            return False
        if (
            self.navigation_state.path_status != PathStatus.PLANNED
            or not self.navigation_state.planned_path
        ):
            logger.error("Cannot start navigation: no planned path")
            return False

        if not self.navigation_state.current_position:
            logger.error("Cannot start navigation: no current position")
            return False

        # Weather gating if service is available
        try:
            if self.weather is not None:
                pos = self.navigation_state.current_position
                latitude = getattr(pos, "latitude", None)
                longitude = getattr(pos, "longitude", None)

                current = None
                if hasattr(self.weather, "get_current_async"):
                    current = await self.weather.get_current_async(
                        latitude=latitude,
                        longitude=longitude,
                    )
                elif hasattr(self.weather, "get_current"):
                    maybe = self.weather.get_current(
                        latitude=latitude,
                        longitude=longitude,
                    )
                    if asyncio.iscoroutine(maybe):
                        current = await maybe
                    else:
                        current = maybe

                advice = self.weather.get_planning_advice(current)
                if advice and advice.get("advice") == "avoid":
                    logger.warning("Navigation start blocked by weather: %s", advice)
                    return False
        except Exception as e:
            # Fail-open to avoid hard-blocking in case of weather service errors
            logger.warning("Weather check failed, proceeding: %s", e)

        self.navigation_state.navigation_mode = NavigationMode.AUTO
        self.navigation_state.path_status = PathStatus.EXECUTING
        self.navigation_state.operation_start_time = datetime.now(UTC)

        logger.info("Started autonomous navigation")
        return True

    async def stop_navigation(self) -> bool:
        """Stop navigation and return to idle: blade off first, then the drive to neutral."""
        self.navigation_state.navigation_mode = NavigationMode.IDLE
        self.navigation_state.target_velocity = 0.0

        if self.navigation_state.path_status == PathStatus.EXECUTING:
            self.navigation_state.path_status = PathStatus.INTERRUPTED

        # Actively command a stop for safety, on the platform that is active: the tractor's
        # levers or the differential drive, and the blade on every blade path (audit SAF-02:
        # this used to send set_speed(0, 0) only, which on the tractor platform commands
        # nothing at all and on neither touched the blade).
        await self._stop_platform()

        logger.info("Navigation stopped")
        return True

    async def start_ai_navigation(self) -> bool:
        """Start AI-controlled autonomous navigation.

        Enables the AI inference service to control steering, throttle,
        and blade based on VLA model predictions.

        Returns:
            True if AI mode started successfully.
        """
        from .ai_inference_service import get_ai_inference_service

        ai_service = get_ai_inference_service()

        if (latched := self._stop_latched()) is not None:
            logger.error("Refusing AI navigation: %s", latched)
            return False
        # A stop that lands while the AI service starts up must end this run before it begins.
        run_epoch = stop_events.epoch()

        # Initialize AI service if needed
        if not ai_service.initialized:
            await ai_service.initialize()
            await ai_service.start()

        # Safety: never drive real motors from placeholder/simulated inference.
        # On hardware, require a real VLA model to be loaded before autonomy.
        sim_mode = is_simulation_mode()
        if not sim_mode and not getattr(ai_service, "_model_loaded", False):
            logger.error(
                "Refusing AI navigation: no VLA model loaded (would drive motors "
                "from placeholder inference). Deploy a model first."
            )
            return False

        # Enable AI control
        success = await ai_service.enable()
        if not success:
            logger.error("Failed to enable AI inference service")
            return False

        self.navigation_state.navigation_mode = NavigationMode.AI
        self.navigation_state.target_velocity = 0.6  # Default AI velocity
        self.navigation_state.operation_start_time = datetime.now(UTC)

        # Establish the local frame (yard datum) and a fresh coverage map for
        # this run, then start the inference loop.
        self._ai_datum = self._select_ai_datum()
        self._coverage = CoverageMap(self._ai_datum)
        self._ai_epoch = run_epoch
        self._ai_task = asyncio.create_task(self._ai_inference_loop())

        logger.info("Started AI-controlled navigation (datum=%s)", self._ai_datum)
        return True

    def _select_ai_datum(self) -> tuple[float, float]:
        """Local-frame origin: home position, else geofence origin, else current
        position. Must match the training datum (config/training.yaml yard_datum)
        for the location features to be in the same frame."""
        hp = self.navigation_state.home_position
        if hp is not None:
            return (hp.latitude, hp.longitude)
        boundaries = self.navigation_state.safety_boundaries
        if boundaries and boundaries[0]:
            return (boundaries[0][0].latitude, boundaries[0][0].longitude)
        cp = self.navigation_state.current_position
        if cp is not None:
            return (cp.latitude, cp.longitude)
        return (0.0, 0.0)

    async def _ai_inference_loop(self) -> None:
        """Drive autonomous mowing from the VLA model.

        Each tick: build a live multi-modal frame, run inference with the current
        location features + causal coverage snapshot, apply the prediction, then
        (causally, after applying) mark the just-mowed cell. Every iteration is
        gated by the existing safety checks in ``apply_ai_prediction`` and stops
        on E-stop / mode change.
        """
        from .ai_inference_service import get_ai_inference_service
        from .perimeter_recorder import get_perimeter_recorder

        ai_service = get_ai_inference_service()
        recorder = get_perimeter_recorder()
        interval = 1.0 / max(self._ai_loop_hz, 0.1)
        logger.info("AI inference loop started at %.1f Hz", self._ai_loop_hz)

        try:
            while self.navigation_state.navigation_mode == NavigationMode.AI:
                if self._ai_epoch is not None and stop_events.epoch() != self._ai_epoch:
                    # A safety stop ended this run; a clear of it must not revive it.
                    self.abort_autonomy("a safety stop happened during the AI run")
                    break
                loop_start = time.perf_counter()
                try:
                    frame = await recorder.capture_frame()
                    if frame is not None:
                        snapshot = self._coverage.snapshot() if self._coverage is not None else None
                        prediction = await ai_service.infer(
                            frame, coverage_grid=snapshot, datum=self._ai_datum
                        )
                        applied = await self.apply_ai_prediction(prediction)
                        # Causally record coverage only after the action is applied,
                        # when the blade is cutting and we have a usable RTK fix.
                        if (
                            applied
                            and self._coverage is not None
                            and prediction.blade
                            and frame.gps is not None
                            and frame.gps.latitude != 0.0
                            and frame.gps.longitude != 0.0
                        ):
                            self._coverage.mark(frame.gps.latitude, frame.gps.longitude)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # never let one bad tick kill the loop
                    logger.warning("AI inference tick failed: %s", exc)

                elapsed = time.perf_counter() - loop_start
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            logger.info("AI inference loop cancelled")
            raise
        finally:
            # Make sure the machine is stopped if the loop exits for any reason: blade off
            # first, then neutral, on the platform that is active.
            try:
                await self._stop_platform()
            except Exception:
                pass

    async def stop_ai_navigation(self) -> bool:
        """Stop AI-controlled navigation and return to manual.

        Returns:
            True if AI mode stopped successfully.
        """
        from .ai_inference_service import get_ai_inference_service

        # Flip mode first so the loop's while-condition exits, then cancel it.
        self.navigation_state.navigation_mode = NavigationMode.MANUAL
        self.navigation_state.target_velocity = 0.0

        if self._ai_task is not None:
            self._ai_task.cancel()
            try:
                await self._ai_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ai_task = None
        self._coverage = None
        self._ai_datum = None
        self._ai_epoch = None

        ai_service = get_ai_inference_service()
        await ai_service.disable()

        # Stop motion: blade off first, then the drive neutral, on the active platform.
        await self._stop_platform()

        logger.info("Stopped AI navigation, returning to manual mode")
        return True

    async def apply_ai_prediction(self, prediction: "ActionPrediction") -> bool:  # noqa: F821
        """Apply an AI action prediction to motor control.

        Called by the AI inference loop to execute predicted actions.
        Respects safety overrides and confidence thresholds.

        Args:
            prediction: ActionPrediction from VLA model.

        Returns:
            True if prediction was applied successfully.
        """
        if self.navigation_state.navigation_mode not in (
            NavigationMode.AI,
            NavigationMode.AI_ASSISTED,
        ):
            return False

        if self._ai_epoch is not None and stop_events.epoch() != self._ai_epoch:
            # A safety stop ended this AI run: nothing it predicts may be applied any more.
            self.abort_autonomy("a safety stop happened during the AI run")
            return False

        # Check confidence threshold
        if prediction.confidence < 0.5:
            logger.warning(f"AI prediction confidence too low: {prediction.confidence:.2f}")
            return False

        from .tractor_service import get_tractor_service

        tractor = get_tractor_service()

        # Check for safety override
        if prediction.safety_override:
            logger.warning("AI prediction has safety override - stopping")
            if tractor.enabled:
                await tractor.emergency_stop()
            else:
                await self.set_speed(0.0, 0.0)
            return True

        # Obstacle guard: a hold stops the machine whatever the model predicts; a cap is
        # enforced again in the actuation chokepoints (tractor.set_levers / set_speed).
        gate = self._motion_gate()
        if gate.hold:
            await self._hold_motion(gate)
            return False
        await self._release_hold()
        if tractor.enabled:
            # Zero-turn mower: twin drive levers + blade PTO. Interlocks (engine
            # running, authorized) in the service gate actual motion.
            cmd = prediction.to_tractor_command()
            scale = self.navigation_state.target_velocity
            # TractorCommand's ge=/le= fields reject out-of-range values instead
            # of clamping, and assignment isn't re-validated, so clamp by hand.
            cmd.left_lever = max(-1.0, min(1.0, cmd.left_lever * scale))
            cmd.right_lever = max(-1.0, min(1.0, cmd.right_lever * scale))
            result = await tractor.apply(cmd, source=AI_SOURCE)
            refusal = self._standing_refusal(result)
            if refusal is not None:
                logger.warning("AI run ended: the tractor refuses it (%s)", refusal)
                self.abort_autonomy(f"the tractor refuses autonomous commands: {refusal}")
                await self._stop_platform()
                return False
            return True

        # Differential-drive path: convert prediction to left/right motor speeds.
        motor_commands = prediction.to_motor_commands()
        scale = self.navigation_state.target_velocity
        left_speed = motor_commands["left_speed"] * scale
        right_speed = motor_commands["right_speed"] * scale

        await self.set_speed(left_speed, right_speed)

        return True

    async def emergency_stop(self) -> bool:
        """Emergency stop navigation"""
        # Every autonomous run ends first, so a clear of the stops below cannot revive one.
        self.abort_autonomy("navigation emergency stop")
        self.navigation_state.navigation_mode = NavigationMode.EMERGENCY_STOP
        self.navigation_state.target_velocity = 0.0
        self.navigation_state.path_status = PathStatus.INTERRUPTED

        # Try to invoke controller-level emergency stop when available
        try:
            if not is_simulation_mode():
                robohat = get_robohat_service()
                if robohat is not None:
                    await robohat.emergency_stop()
        except Exception:
            pass

        # Tractor platform: cut drive/blade and revoke authorization too.
        try:
            from .tractor_service import get_tractor_service

            tractor = get_tractor_service()
            if tractor.enabled:
                await tractor.emergency_stop()
        except Exception:
            pass

        logger.critical("Emergency stop activated")
        return True

    async def return_home(self) -> bool:
        """Navigate back to home position"""
        if (latched := self._stop_latched()) is not None:
            logger.error("Cannot return home: %s", latched)
            return False
        if not self.navigation_state.home_position:
            logger.error("No home position set")
            return False

        # Plan path (with avoidance and boundary constraints if available)
        if self.navigation_state.current_position:
            boundaries = None
            if self.navigation_state.safety_boundaries:
                # Use the outer boundary (first polygon) if provided
                boundaries = self.navigation_state.safety_boundaries[0]
            obstacles = (
                self.navigation_state.no_go_zones if self.navigation_state.no_go_zones else None
            )

            waypoints = self.path_planner.return_to_base(
                current=self.navigation_state.current_position,
                home=self.navigation_state.home_position,
                boundary=boundaries,
                obstacles=obstacles,
            )

            self.navigation_state.planned_path = waypoints
            self.navigation_state.current_waypoint_index = 0
            self.navigation_state.navigation_mode = NavigationMode.RETURN_HOME
            self.navigation_state.path_status = PathStatus.EXECUTING

            logger.info("Returning to home position")
            return True

        return False

    def set_home_position(self, position: Position):
        """Set the home/docking position"""
        self.navigation_state.home_position = position
        logger.info(f"Home position set to {position.latitude}, {position.longitude}")

    def set_safety_boundaries(self, boundaries: list[list[Position]]):
        """Set safety boundaries that must not be crossed"""
        self.navigation_state.safety_boundaries = boundaries
        logger.info(f"Set {len(boundaries)} safety boundaries")

    def add_no_go_zone(self, zone: list[Position]):
        """Add a no-go zone to avoid"""
        self.navigation_state.no_go_zones.append(zone)
        logger.info("Added no-go zone")

    async def get_navigation_status(self) -> dict[str, Any]:
        """Get current navigation status"""
        return {
            "mode": self.navigation_state.navigation_mode,
            "path_status": self.navigation_state.path_status,
            "current_position": self.navigation_state.current_position,
            "heading": self.navigation_state.heading,
            "velocity": self.navigation_state.velocity,
            "target_velocity": self.navigation_state.target_velocity,
            "waypoints_total": len(self.navigation_state.planned_path),
            "waypoints_completed": self.navigation_state.current_waypoint_index,
            "distance_traveled": self.navigation_state.distance_traveled,
            "obstacles_detected": len(self.navigation_state.obstacle_map),
            "dead_reckoning_active": self.navigation_state.dead_reckoning_active,
            "path_confidence": self.navigation_state.path_confidence,
            "motion_hold": self._hold_reasons is not None,
            "motion_gate_reasons": list(self._motion_gate().reasons),
        }
