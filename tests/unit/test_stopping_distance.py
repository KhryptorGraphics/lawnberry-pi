"""Stopping-distance model (WP-P4-02): formula, monotonicity, inverse, boundaries, config.

The numbers under test are UNVALIDATED defaults (audit M-43). These tests pin the *shape*
of the model - what must hold whatever the measured parameters turn out to be - plus a few
hand-computed values so a silent change to the formula is caught.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.src.core.config_loader import ConfigLoader
from backend.src.models.safety_limits import (
    GpsDegradationLimits,
    ObstacleStopLimits,
    SafetyLimits,
    StoppingDistanceLimits,
)
from backend.src.safety.stopping_distance import StoppingDistanceModel

SPEEDS = [i / 20 for i in range(0, 17)]  # 0.00 .. 0.80 m/s
RANGES = [0.0, 0.1, 0.3, 0.31, 0.5, 0.6, 0.7, 0.9, 1.0, 1.5, 3.0, 10.0]


def model(**kw) -> StoppingDistanceModel:
    return StoppingDistanceModel(StoppingDistanceLimits(**kw))


def test_stop_distance_at_rest_is_uncertainty_plus_margin():
    m = model(sensor_uncertainty_m=0.15, margin_m=0.25)
    assert m.stop_distance(0.0) == pytest.approx(0.40)


def test_stop_distance_matches_the_hand_computed_formula():
    # v*(t_react + t_act) + v^2/(2a) + unc + margin
    m = model(
        reaction_latency_s=0.4,
        actuation_latency_s=0.6,
        deceleration_mps2=0.8,
        sensor_uncertainty_m=0.1,
        margin_m=0.2,
    )
    v = 0.5
    assert m.stop_distance(v) == pytest.approx(0.5 * 1.0 + 0.25 / 1.6 + 0.3)
    # a reading that is already 0.25 s old costs another v * 0.25
    assert m.stop_distance(v, extra_latency_s=0.25) == pytest.approx(
        m.stop_distance(v) + 0.5 * 0.25
    )


def test_slow_distance_adds_the_slow_zone():
    m = model(slow_zone_m=0.4)
    for v in SPEEDS:
        assert m.slow_distance(v) == pytest.approx(m.stop_distance(v) + 0.4)


def test_stop_distance_strictly_increases_with_speed_and_with_reading_age():
    m = model()
    stops = [m.stop_distance(v) for v in SPEEDS]
    assert all(b > a for a, b in zip(stops, stops[1:], strict=False))
    ages = [0.0, 0.1, 0.5, 1.0]
    at_speed = [m.stop_distance(0.4, extra_latency_s=a) for a in ages]
    assert all(b > a for a, b in zip(at_speed, at_speed[1:], strict=False))


@pytest.mark.parametrize("age", [0.0, 0.2, 0.9])
def test_inverse_round_trips_above_the_zero_speed_floor(age):
    m = model()
    floor = m.stop_distance(0.0)
    for r in [floor + d for d in (0.001, 0.05, 0.3, 1.0, 5.0)]:
        v = m.speed_for_stop_distance(r, extra_latency_s=age)
        assert v > 0
        assert m.stop_distance(v, extra_latency_s=age) == pytest.approx(r, abs=1e-9)


def test_inverse_is_zero_at_and_below_the_floor_and_positive_just_above_it():
    m = model()
    floor = m.stop_distance(0.0)
    assert m.speed_for_stop_distance(floor) == 0.0
    assert m.speed_for_stop_distance(floor - 1e-9) == 0.0
    assert m.speed_for_stop_distance(-5.0) == 0.0
    assert m.speed_for_stop_distance(floor + 1e-6) > 0.0


def test_inverse_is_numerically_sound_for_a_very_slow_actuator():
    # A large latency makes the naive -b + sqrt(b^2 + c) lose every digit of a small answer.
    m = model(actuation_latency_s=60.0)
    r = m.stop_distance(0.0) + 1e-6
    v = m.speed_for_stop_distance(r)
    assert 0.0 < v < 1e-6
    assert m.stop_distance(v) == pytest.approx(r, rel=1e-9)


def test_cap_is_monotone_in_range_never_negative_and_never_above_full_speed():
    m = model()
    caps = [m.max_speed_for_range(r) for r in RANGES]
    assert all(0.0 <= c <= m.full_lever_speed_mps for c in caps)
    assert all(b >= a for a, b in zip(caps, caps[1:], strict=False))
    assert m.max_speed_for_range(1e9) == m.full_lever_speed_mps
    assert m.max_speed_for_range(math.inf) == m.full_lever_speed_mps


def test_cap_is_zero_just_inside_the_hold_distance_and_motion_just_outside():
    m = model()
    hold = m.hold_distance()
    assert m.max_speed_for_range(hold - 1e-6) == 0.0
    just_outside = m.max_speed_for_range(hold + 1e-6)
    assert just_outside == pytest.approx(m.params.min_motion_speed_mps, abs=1e-5)
    assert just_outside >= m.params.min_motion_speed_mps


def test_a_speed_between_zero_and_the_minimum_is_a_hold_not_a_creep():
    m = model(min_motion_speed_mps=0.2)
    hold = m.hold_distance()
    # a hair of room beyond the stop line allows 0 < v < 0.2, which must read as hold
    r = m.params.slow_zone_m + m.stop_distance(0.0) + 1e-3
    assert 0 < m.speed_for_stop_distance(r - m.params.slow_zone_m) < 0.2
    assert m.max_speed_for_range(r) == 0.0
    assert hold > r


def test_obeying_the_cap_keeps_the_machine_out_of_the_stopping_envelope():
    m = model()
    for age in (0.0, 0.3):
        for r in RANGES:
            v = m.max_speed_for_range(r, extra_latency_s=age)
            if v > 0:
                # the slow zone is still in reserve at the speed the cap allows
                assert m.slow_distance(v, extra_latency_s=age) <= r + 1e-9
                assert m.stop_distance(v, extra_latency_s=age) <= r - m.params.slow_zone_m + 1e-9


def test_an_older_reading_never_allows_more_speed():
    m = model()
    for r in RANGES:
        assert m.max_speed_for_range(r, extra_latency_s=0.5) <= m.max_speed_for_range(r)


@pytest.mark.parametrize(
    ("field", "worse"),
    [
        ("reaction_latency_s", 0.9),
        ("actuation_latency_s", 1.2),
        ("sensor_uncertainty_m", 0.4),
        ("margin_m", 0.6),
        ("slow_zone_m", 0.7),
    ],
)
def test_each_pessimistic_parameter_lowers_the_cap(field, worse):
    base, pess = model(), model(**{field: worse})
    assert pess.max_speed_for_range(1.2) < base.max_speed_for_range(1.2)
    assert pess.stop_distance(0.4) >= base.stop_distance(0.4)


def test_better_braking_raises_the_cap():
    assert model(deceleration_mps2=2.0).max_speed_for_range(1.2) > model().max_speed_for_range(1.2)


def test_lever_speed_mapping_is_linear_and_clamped():
    m = model(full_lever_speed_mps=0.8)
    assert m.speed_of_lever(1.0) == pytest.approx(0.8)
    assert m.speed_of_lever(0.5) == pytest.approx(0.4)
    assert m.speed_of_lever(-1.0) == 0.0  # reverse is not forward ground speed
    assert m.lever_fraction(0.4) == pytest.approx(0.5)
    assert m.lever_fraction(5.0) == 1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_inputs_are_refused_rather_than_turned_into_numbers(bad):
    m = model()
    with pytest.raises(ValueError):
        m.stop_distance(bad)
    with pytest.raises(ValueError):
        m.stop_distance(0.3, extra_latency_s=bad)
    with pytest.raises(ValueError):
        m.speed_for_stop_distance(math.nan)


# ----------------------------------------------------------------------------- config


def test_defaults_are_conservative_and_marked_unvalidated():
    assert StoppingDistanceLimits().validated is False
    assert ObstacleStopLimits().validated is False
    assert GpsDegradationLimits().validated is False
    assert StoppingDistanceModel().validated is False
    # conservative: the machine is held well before contact at the slowest useful speed
    assert StoppingDistanceModel().hold_distance() > 0.5


def test_shipped_limits_yaml_loads_and_declares_the_model_unvalidated(tmp_path):
    raw = yaml.safe_load(Path("config/limits.yaml").read_text())
    assert raw["obstacle_stop"]["validated"] is False
    assert raw["gps_degradation"]["validated"] is False
    _hw, limits = ConfigLoader(config_dir="config").load()
    assert limits.obstacle_stop.validated is False
    assert limits.obstacle_stop.full_lever_speed_mps == pytest.approx(0.8)
    assert limits.gps_degradation.max_fix_age_s > 0


def test_limits_yaml_overrides_reach_the_model(tmp_path):
    (tmp_path / "limits.yaml").write_text(
        "obstacle_stop:\n  deceleration_mps2: 1.5\n  clear_dwell_s: 5\n  validated: true\n",
        encoding="utf-8",
    )
    _hw, limits = ConfigLoader(config_dir=str(tmp_path)).load()
    assert limits.obstacle_stop.deceleration_mps2 == 1.5
    assert limits.obstacle_stop.clear_dwell_s == 5
    assert StoppingDistanceModel(limits.obstacle_stop).validated is True


def test_a_typo_in_a_safety_limit_fails_loudly_instead_of_using_the_default(tmp_path):
    (tmp_path / "limits.yaml").write_text(
        "obstacle_stop:\n  decelleration_mps2: 1.5\n", encoding="utf-8"
    )
    with pytest.raises((ValidationError, ValueError)):
        ConfigLoader(config_dir=str(tmp_path)).load()


@pytest.mark.parametrize(
    "bad",
    [
        {"deceleration_mps2": 0},
        {"deceleration_mps2": -1},
        {"full_lever_speed_mps": 0},
        {"detection_range_m": 0},
        {"sensor_uncertainty_m": -0.1},
        {"margin_m": -0.1},
        {"min_motion_speed_mps": 2.0},  # above full lever speed
        {"max_reading_age_s": 0},
    ],
)
def test_nonsensical_model_parameters_are_rejected(bad):
    with pytest.raises(ValidationError):
        ObstacleStopLimits(**bad)


def test_detection_range_must_exceed_the_static_stop_lines():
    with pytest.raises(ValidationError, match="detection_range_m"):
        SafetyLimits(tof_obstacle_distance_meters=1.5)
    with pytest.raises(ValidationError, match="detection_range_m"):
        SafetyLimits(ultrasonic_obstacle_distance_cm=120.0)
    SafetyLimits(tof_obstacle_distance_meters=0.9)  # still inside the 1.0 m default reach
