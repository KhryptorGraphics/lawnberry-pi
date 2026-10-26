from __future__ import annotations

import math
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

# ``limits.yaml`` is the file that sets how far the mower may travel before it must stop, how old
# a reading may be, and when a tilt cuts the blade. Every number in it is therefore validated
# the same strict way (audit SAF-08):
#
# * finite: PyYAML turns ``.nan``/``.inf`` into floats, and ``nan``/``inf`` in a threshold either
#   never fire (``x > nan`` is false) or fire for ever, silently, depending on the comparison;
# * inside a physically sensible range: a bound that is far beyond anything the hardware can do
#   (a 24 h watchdog, a 1 km/h^2 deceleration, a 40 000 deg tilt limit) is a typo or an attempt to
#   switch a protection off, so it is an error with the field name and the allowed range;
# * unknown keys are errors, at every level: a misspelt key must not silently run the default;
# * flags are real booleans (``validated: "no"`` and ``validated: yes`` must not coerce).
#
# A value that disables a protection on purpose still has an explicit, loud switch
# (``obstacle_stop.ignored_sources`` logs CRITICAL). The permissive direction of each number
# (a small stopping distance, a long reading age, a high deceleration) has a tight bound; the
# conservative direction (a long latency, a large margin) only needs to be finite and sane.

#: A source name the obstacle guard can know: a ToF side or an ultrasonic channel.
_SOURCE_NAME = re.compile(r"^(tof_left|tof_right|ultrasonic:[^\s:]+)$")


class StoppingDistanceLimits(BaseModel):
    """Stopping-distance model parameters (``limits.yaml`` ``obstacle_stop``; WP-P4-02).

    **EVERY DEFAULT BELOW IS UNVALIDATED.** They are conservative engineering guesses chosen
    so the stop path exists and fails safe; none is a measurement. They change only from the
    measured stopping distance at several speeds and slopes (audit M-43) with a reviewer's
    signature, and ``validated`` is set to true by that same change. See
    ``backend/src/safety/stopping_distance.py`` for the formula.
    """

    # A typo in limits.yaml must fail loudly: pydantic would otherwise ignore the key and
    # silently run on the default of a safety limit. NaN/inf are never a limit.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    validated: StrictBool = Field(
        default=False, description="True only with M-43 data and sign-off"
    )
    full_lever_speed_mps: float = Field(
        default=0.8,
        ge=0.1,
        le=10.0,
        description="Ground speed at full lever (== NavigationService.max_speed)",
    )
    reaction_latency_s: float = Field(
        default=0.3,
        ge=0,
        le=60.0,
        description="Compute + command issue; sensor age is added dynamically",
    )
    actuation_latency_s: float = Field(
        default=0.5,
        ge=0,
        le=120.0,
        description="Command issued until levers are neutral (acceptance s.2)",
    )
    deceleration_mps2: float = Field(
        default=0.5,
        gt=0,
        le=10.0,
        description="Deceleration once neutral; worst case (downhill, wet)",
    )
    sensor_uncertainty_m: float = Field(
        default=0.10, ge=0, le=10.0, description="Range error + mount offset"
    )
    margin_m: float = Field(
        default=0.20, ge=0, le=10.0, description="Safety allowance on top of the above"
    )
    slow_zone_m: float = Field(
        default=0.30, ge=0, le=10.0, description="Taper band beyond the stop distance"
    )
    min_motion_speed_mps: float = Field(
        default=0.10,
        ge=0,
        le=10.0,
        description="Below this the machine is held, not allowed to creep",
    )
    detection_range_m: float = Field(
        default=1.0,
        gt=0,
        le=10.0,
        description="Reach beyond which a sensor proves nothing about clearance",
    )

    @model_validator(mode="after")
    def _min_speed_within_full_speed(self) -> StoppingDistanceLimits:
        if self.min_motion_speed_mps > self.full_lever_speed_mps:
            raise ValueError("min_motion_speed_mps must not exceed full_lever_speed_mps")
        return self


class ObstacleStopLimits(StoppingDistanceLimits):
    """Stopping-distance model plus the obstacle-guard policy (``limits.yaml`` obstacle_stop)."""

    max_reading_age_s: float = Field(
        default=1.0,
        gt=0,
        le=10.0,
        description="A range reading older than this is unusable (fail closed)",
    )
    clear_dwell_s: float = Field(
        default=2.0,
        ge=0,
        le=60.0,
        description="Clear for this long before a hold is released (0 disables the hysteresis)",
    )
    frozen_reading_count: int = Field(
        default=50,
        ge=10,
        le=100_000,
        description=(
            "A ToF source that reports the identical in-range distance in this many consecutive "
            "fresh readings is treated as frozen (unknown), not as a steady scene"
        ),
    )
    apply_to_manual: StrictBool = Field(
        default=False,
        description="Also limit commands labelled 'operator'/'teleop' (default: autonomy only)",
    )
    ignored_sources: list[str] = Field(
        default_factory=list,
        description="Sources dropped from the required set; removes a safety input (CRITICAL log)",
    )

    @field_validator("ignored_sources")
    @classmethod
    def _known_source_names(cls, names: list[str]) -> list[str]:
        for name in names:
            if not isinstance(name, str) or not _SOURCE_NAME.match(name):
                raise ValueError(
                    f"ignored_sources entry {name!r} is not a source: use tof_left, tof_right or "
                    "ultrasonic:<channel> (a typo here would silently ignore nothing)"
                )
        return names


class GpsDegradationLimits(BaseModel):
    """GNSS-quality hold policy (``limits.yaml`` ``gps_degradation``; audit D.9, SD-4).

    UNVALIDATED defaults. The accuracy bound is the owner's and the reviewer's decision (SD-4),
    to be derived from the geofence margin and the measured position error (M-50, M-51).
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    validated: StrictBool = False
    max_accuracy_m: float = Field(
        default=1.0, gt=0, le=50.0, description="Reported 1-sigma horizontal"
    )
    max_fix_age_s: float = Field(default=2.0, gt=0, le=30.0, description="Age of the fix itself")
    resume_dwell_s: float = Field(
        default=3.0,
        ge=0,
        le=300.0,
        description="Continuously healthy for this long before resuming",
    )
    check_interval_s: float = Field(default=0.2, gt=0, le=10.0)


class SafetyLimits(BaseModel):
    """Defines constitutional safety thresholds and timeouts (FR-004)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    estop_latency_ms: int = Field(default=100, ge=1, description="Max E-stop response time")
    tilt_threshold_degrees: float = Field(
        default=30.0,
        gt=0,
        le=60.0,
        description="Roll or pitch at or beyond this cuts the blade (a bound above 60 never fires)",
    )
    tilt_cutoff_latency_ms: int = Field(default=200, ge=1)
    battery_low_voltage: float = Field(default=10.0, gt=0, le=60.0)
    battery_critical_voltage: float = Field(default=9.5, gt=0, le=60.0)
    motor_current_max_amps: float = Field(default=5.0, gt=0, le=100.0)
    watchdog_timeout_ms: int = Field(
        default=1000,
        ge=1,
        le=10_000,
        description="Silence of a required heartbeat before the stop (longer is no watchdog)",
    )
    geofence_buffer_meters: float = Field(default=0.5, ge=0, le=50.0)
    high_temperature_celsius: float = Field(default=80.0, gt=-40.0, le=150.0)
    tof_obstacle_distance_meters: float = Field(
        default=0.2, gt=0, le=2.0, description="ToF static stop line (the sensor reaches ~2 m)"
    )
    ultrasonic_obstacle_distance_cm: float = Field(
        default=30.0,
        ge=10.0,
        le=400.0,
        allow_inf_nan=False,
        description="Ultrasonic static stop line (4 m reach; not below the 10 cm blind zone)",
    )
    ultrasonic_enabled: StrictBool = Field(default=True)
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
        if not math.isfinite(floor_m) or self.obstacle_stop.detection_range_m <= floor_m:
            raise ValueError(
                "obstacle_stop.detection_range_m must exceed the static obstacle distances "
                f"({floor_m:.2f} m): a sensor cannot prove clearance at less than its own stop line"
            )
        return self
