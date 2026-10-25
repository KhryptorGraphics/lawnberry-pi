"""IMU calibration orchestration helpers.

Provides a high-level routine that gently moves the mower through the
orientations the BNO085 expects during its calibration sequence. The routine
is safe to call in SIM_MODE (returns a simulated success) and guards against
concurrent executions so operators cannot stack conflicting calibration runs.

The manoeuvre is motion like any other, so it goes through the same safety path as an
autonomous leg (audit SAF-11): ``NavigationService.drive_scripted`` re-checks the obstacle
guard and the safety-stop epoch on every control tick, holds (blade off, neutral) and ends the
whole routine the moment either objects, caps forward speed to the stopping-distance model, and
refuses to start while a stop is latched, the guard holds, or an autonomous run owns the drive.
An aborted routine never resumes by itself: the operator starts a new one.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from ..core.simulation import is_simulation_mode
from ..safety import stop_events
from . import robohat_service as robohat_module
from .navigation_service import NavigationService, ScriptedOutcome

logger = logging.getLogger(__name__)


class CalibrationInProgressError(RuntimeError):
    """Raised when a calibration attempt is already running."""


class DriveControllerUnavailableError(RuntimeError):
    """Raised when no drive controller is available for calibration."""


class CalibrationRefusedError(RuntimeError):
    """Raised when a stop latch, an obstacle hold or an autonomous run forbids the drive."""


_CALIBRATION_STATE_TO_SCORE = {
    "fully_calibrated": 3,
    "calibrated": 3,
    "calibrating": 2,
    "partial": 2,
    "unknown": 1,
    None: 0,
}

#: The manoeuvre, in order: normalized wheel speeds (full scale = 1.0) held for ``duration`` s.
CALIBRATION_SEQUENCE: tuple[dict[str, Any], ...] = (
    {
        "name": "spin_clockwise",
        "description": "Rotate clockwise to excite gyro heading",
        "duration": 4.0,
        "left_speed": 0.35,
        "right_speed": -0.35,
    },
    {
        "name": "spin_counter_clockwise",
        "description": "Rotate counter-clockwise",
        "duration": 4.0,
        "left_speed": -0.35,
        "right_speed": 0.35,
    },
    {
        "name": "forward_sweep",
        "description": "Drive forward slowly to settle accelerometer",
        "duration": 2.5,
        "left_speed": 0.4,
        "right_speed": 0.4,
    },
    {
        "name": "reverse_sweep",
        "description": "Drive backwards gently",
        "duration": 2.5,
        "left_speed": -0.35,
        "right_speed": -0.35,
    },
)

#: How long the mower stands still after the manoeuvre before the IMU status is read.
SETTLE_S = 0.75


class IMUCalibrationService:
    """Coordinate IMU calibration manoeuvres and telemetry sampling."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last_result: dict[str, Any] | None = None

    async def run(self, sensor_manager=None) -> dict[str, Any]:
        """Execute the calibration routine and return a status payload.

        Raises ``CalibrationRefusedError`` (nothing moved) when a stop latch, the obstacle
        guard's hold or an autonomous run forbids the drive. A routine that is stopped part way
        (the guard held, a safety stop landed, the controller refused a command) returns
        ``status: "aborted"`` with the reason in ``notes`` and per step; it is not resumed.
        """
        if self._lock.locked():
            raise CalibrationInProgressError("Calibration already in progress")

        async with self._lock:
            result = await self._execute(sensor_manager)
            self._last_result = result
            return result

    async def _execute(self, sensor_manager=None) -> dict[str, Any]:
        # Short-circuit in simulation to avoid hardware requirements.
        if is_simulation_mode():
            await asyncio.sleep(0.5)
            ts = datetime.now(UTC).isoformat()
            return {
                "status": "simulated",
                "calibration_status": "fully_calibrated",
                "calibration_score": 3,
                "steps": [],
                "timestamp": ts,
                "notes": "SIM_MODE enabled; returning canned calibration result.",
            }

        robohat = robohat_module.get_robohat_service()
        if (
            robohat is None
            or not robohat.running
            or not getattr(robohat.status, "serial_connected", False)
        ):
            raise DriveControllerUnavailableError("RoboHAT drive controller is offline")

        # The same rules as an autonomous leg decide whether the machine may move at all.
        nav = NavigationService.get_instance()
        refusal = nav.scripted_drive_refusal()
        if refusal is not None:
            raise CalibrationRefusedError(refusal)

        from .sensor_manager import SensorManager  # local import to avoid circular deps

        manager = sensor_manager
        if manager is None:
            manager = SensorManager()
            await manager.initialize()

        step_results: list[dict[str, Any]] = []
        start_ts = datetime.now(UTC)

        async def _capture_snapshot() -> dict[str, Any] | None:
            try:
                snapshot = await manager.read_all_sensors()  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover - hardware dependent
                logger.debug("Failed to read sensors during calibration: %s", exc)
                return None
            imu = getattr(snapshot, "imu", None)
            if imu is None:
                return None
            return {
                "roll": getattr(imu, "roll", None),
                "pitch": getattr(imu, "pitch", None),
                "yaw": getattr(imu, "yaw", None),
                "calibration_status": getattr(imu, "calibration_status", None),
                "calibration_score": _CALIBRATION_STATE_TO_SCORE.get(
                    getattr(imu, "calibration_status", None), 0
                ),
            }

        # One epoch for the whole routine: a safety stop between two steps ends it as surely as
        # one during a step, and a clear of that stop never resumes it.
        run_epoch = stop_events.epoch()
        aborted: str | None = None
        try:
            for step in CALIBRATION_SEQUENCE:
                drive = await nav.drive_scripted(
                    step["left_speed"],
                    step["right_speed"],
                    step["duration"],
                    run_epoch=run_epoch,
                )
                logger.info(
                    "IMU calibration step %s: %s %s",
                    step["name"],
                    drive.outcome.value,
                    "; ".join(drive.reasons),
                )
                sample = await _capture_snapshot()
                entry: dict[str, Any] = {
                    "name": step["name"],
                    "description": step["description"],
                    "duration_s": step["duration"],
                    "ran_s": round(drive.ran_s, 2),
                    "command_accepted": drive.commands_accepted > 0,
                    "outcome": drive.outcome.value,
                    "sensor_snapshot": sample,
                }
                if drive.reasons:
                    entry["reasons"] = list(drive.reasons)
                step_results.append(entry)
                if drive.outcome is not ScriptedOutcome.DONE:
                    aborted = f"{step['name']}: {drive.outcome.value}" + (
                        f" ({'; '.join(drive.reasons)})" if drive.reasons else ""
                    )
                    logger.warning("IMU calibration aborted at %s", aborted)
                    break
        finally:
            # Always bring the mower to a stop at the end of the routine: blade off, neutral.
            failures = await nav.halt_platform("IMU calibration ended")
            if failures:
                logger.warning("Calibration stand-down not confirmed: %s", failures)

        await asyncio.sleep(SETTLE_S)
        final_snapshot = await _capture_snapshot()
        final_status = None
        final_score = 0
        if final_snapshot:
            final_status = final_snapshot.get("calibration_status")
            final_score = int(final_snapshot.get("calibration_score") or 0)

        if aborted is not None:
            status = "aborted"
            notes: str | None = (
                f"Calibration stopped and not resumed: {aborted}. Start a new run when clear."
            )
        else:
            status = "completed" if final_score >= 2 else "pending"
            notes = (
                None if final_score >= 2 else "Continue moving mower gently until status improves."
            )
        return {
            "status": status,
            "calibration_status": final_status,
            "calibration_score": final_score,
            "steps": step_results,
            "timestamp": datetime.now(UTC).isoformat(),
            "started_at": start_ts.isoformat(),
            "notes": notes,
        }

    def last_result(self) -> dict[str, Any] | None:
        """Return the most recent result, if one exists."""
        return self._last_result

    def is_running(self) -> bool:
        """Return True if a calibration routine is currently executing."""
        return self._lock.locked()


imu_calibration_service = IMUCalibrationService()
