from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StoppingDistanceLimits(BaseModel):
    """Stopping-distance model parameters (``limits.yaml`` ``obstacle_stop``; WP-P4-02).

    **EVERY DEFAULT BELOW IS UNVALIDATED.** They are conservative engineering guesses chosen
    so the stop path exists and fails safe; none is a measurement. They change only from the
    measured stopping distance at several speeds and slopes (audit M-43) with a reviewer's
    signature, and ``validated`` is set to true by that same change. See
    ``backend/src/safety/stopping_distance.py`` for the formula.
    """

    # A typo in limits.yaml must fail loudly: pydantic would otherwise ignore the key and
    # silently run on the default of a safety limit.
    model_config = ConfigDict(extra="forbid")

    validated: bool = Field(default=False, description="True only with M-43 data and sign-off")
    full_lever_speed_mps: float = Field(
        default=0.8, gt=0, description="Ground speed at full lever (== NavigationService.max_speed)"
    )
    reaction_latency_s: float = Field(
        default=0.3, ge=0, description="Compute + command issue; sensor age is added dynamically"
    )
    actuation_latency_s: float = Field(
        default=0.5, ge=0, description="Command issued until levers are neutral (acceptance s.2)"
    )
    deceleration_mps2: float = Field(
        default=0.5, gt=0, description="Deceleration once neutral; worst case (downhill, wet)"
    )
    sensor_uncertainty_m: float = Field(
        default=0.10, ge=0, description="Range error + mount offset"
    )
    margin_m: float = Field(default=0.20, ge=0, description="Safety allowance on top of the above")
    slow_zone_m: float = Field(
        default=0.30, ge=0, description="Taper band beyond the stop distance"
    )
    min_motion_speed_mps: float = Field(
        default=0.10, ge=0, description="Below this the machine is held, not allowed to creep"
    )
    detection_range_m: float = Field(
        default=1.0, gt=0, description="Reach beyond which a sensor proves nothing about clearance"
    )

    @model_validator(mode="after")
    def _min_speed_within_full_speed(self) -> StoppingDistanceLimits:
        if self.min_motion_speed_mps > self.full_lever_speed_mps:
            raise ValueError("min_motion_speed_mps must not exceed full_lever_speed_mps")
        return self


class ObstacleStopLimits(StoppingDistanceLimits):
    """Stopping-distance model plus the obstacle-guard policy (``limits.yaml`` obstacle_stop)."""

    max_reading_age_s: float = Field(
        default=1.0, gt=0, description="A range reading older than this is unusable (fail closed)"
    )
    clear_dwell_s: float = Field(
        default=2.0, ge=0, description="Clear for this long before a hold is released"
    )
    apply_to_manual: bool = Field(
        default=False,
        description="Also limit commands labelled 'operator'/'teleop' (default: autonomy only)",
    )
    ignored_sources: list[str] = Field(
        default_factory=list,
        description="Sources dropped from the required set; removes a safety input (CRITICAL log)",
    )


class GpsDegradationLimits(BaseModel):
    """GNSS-quality hold policy (``limits.yaml`` ``gps_degradation``; audit D.9, SD-4).

    UNVALIDATED defaults. The accuracy bound is the owner's and the reviewer's decision (SD-4),
    to be derived from the geofence margin and the measured position error (M-50, M-51).
    """

    model_config = ConfigDict(extra="forbid")

    validated: bool = False
    max_accuracy_m: float = Field(default=1.0, gt=0, description="Reported 1-sigma horizontal")
    max_fix_age_s: float = Field(default=2.0, gt=0, description="Age of the fix itself")
    resume_dwell_s: float = Field(
        default=3.0, ge=0, description="Continuously healthy for this long before resuming"
    )
    check_interval_s: float = Field(default=0.2, gt=0)


class SafetyLimits(BaseModel):
    """Defines constitutional safety thresholds and timeouts (FR-004)."""

    estop_latency_ms: int = Field(default=100, ge=1, description="Max E-stop response time")
    tilt_threshold_degrees: float = Field(default=30.0)
    tilt_cutoff_latency_ms: int = Field(default=200, ge=1)
    battery_low_voltage: float = Field(default=10.0)
    battery_critical_voltage: float = Field(default=9.5)
    motor_current_max_amps: float = Field(default=5.0)
    watchdog_timeout_ms: int = Field(default=1000, ge=1)
    geofence_buffer_meters: float = Field(default=0.5)
    high_temperature_celsius: float = Field(default=80.0)
    tof_obstacle_distance_meters: float = Field(default=0.2)
    ultrasonic_obstacle_distance_cm: float = Field(default=30.0, ge=10.0, allow_inf_nan=False)
    ultrasonic_enabled: bool = Field(default=True)
    obstacle_stop: ObstacleStopLimits = Field(default_factory=ObstacleStopLimits)
    gps_degradation: GpsDegradationLimits = Field(default_factory=GpsDegradationLimits)

    @field_validator("estop_latency_ms")
    @classmethod
    def validate_estop_latency(cls, v: int):
        if v > 100:
            raise ValueError("estop_latency_ms must be ≤ 100 (constitutional)")
        return v

    @field_validator("tilt_cutoff_latency_ms")
    @classmethod
    def validate_tilt_cutoff_latency(cls, v: int):
        if v > 200:
            raise ValueError("tilt_cutoff_latency_ms must be ≤ 200 (constitutional)")
        return v

    @field_validator("battery_critical_voltage")
    @classmethod
    def critical_below_low(cls, v: float, info):
        low = info.data.get("battery_low_voltage", 10.0)
        if v >= low:
            raise ValueError("battery_critical_voltage must be < battery_low_voltage")
        return v

    @model_validator(mode="after")
    def _detection_range_exceeds_static_floors(self) -> SafetyLimits:
        floor_m = max(self.tof_obstacle_distance_meters, self.ultrasonic_obstacle_distance_cm / 100)
        if self.obstacle_stop.detection_range_m <= floor_m:
            raise ValueError(
                "obstacle_stop.detection_range_m must exceed the static obstacle distances "
                f"({floor_m:.2f} m): a sensor cannot prove clearance at less than its own stop line"
            )
        return self
