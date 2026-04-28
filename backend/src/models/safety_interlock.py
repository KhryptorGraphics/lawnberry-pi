from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class InterlockType(StrEnum):
    EMERGENCY_STOP = "emergency_stop"
    TILT_DETECTED = "tilt_detected"
    LOW_BATTERY = "low_battery"
    GEOFENCE_VIOLATION = "geofence_violation"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    HIGH_TEMPERATURE = "high_temperature"
    OBSTACLE_DETECTED = "obstacle_detected"  # ToF sensors
    ULTRASONIC_OBSTACLE = "ultrasonic_obstacle"  # HC-SR04 sensors


class InterlockState(StrEnum):
    ACTIVE = "active"
    CLEARED_PENDING_ACK = "cleared_pending_ack"
    ACKNOWLEDGED = "acknowledged"


class SafetyInterlock(BaseModel):
    interlock_id: str
    interlock_type: InterlockType
    triggered_at_us: int
    cleared_at_us: int | None = None
    acknowledged_at_us: int | None = None
    state: InterlockState = Field(default=InterlockState.ACTIVE)
    trigger_value: float | None = None
    description: str = ""
