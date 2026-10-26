"""Obstacle guard, second review round: the holds that must outlast the first 2 s, clock steps,
unusable clear horizons, frozen ToF readings (audit SAF-05, SAF-10, SAF-S2, TOOL-02).

``ObstacleGuard`` starts held and releases only after ``clear_dwell_s``, so a test that asserts a
hold once, at t=0, passes whether or not the "no data" / "no sources" branches hold: the first
group below assesses for far longer than dwell + stale window.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.src.models.safety_limits import ObstacleStopLimits
from backend.src.safety import obstacle_stop
from backend.src.safety.obstacle_stop import ObstacleGuard, sample_sensor_manager


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_guard(clock=None, **limits):
    clock = clock or Clock()
    guard = ObstacleGuard(ObstacleStopLimits(**limits), clock=clock)
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    return guard, clock


def assess_for(guard, clock, seconds, feed=None, step=0.1):
    verdicts = []
    for _ in range(round(seconds / step)):
        clock.t += step
        if feed is not None:
            feed()
        verdicts.append(guard.assess())
    return verdicts


# ------------------------------------------------------------------------------ TOOL-02


def test_no_data_at_all_is_a_hold_for_as_long_as_it_lasts():
    guard, clock = make_guard()
    verdicts = assess_for(guard, clock, 6.0)  # dwell 2 s + stale window 1 s, and far beyond
    assert all(v.hold and v.speed_cap_mps == 0.0 for v in verdicts)
    assert set(verdicts[-1].faults) == {"tof_left", "tof_right"}


def test_no_sources_configured_is_a_hold_for_as_long_as_it_lasts():
    guard = ObstacleGuard(ObstacleStopLimits(), clock=(clock := Clock()))
    verdicts = assess_for(guard, clock, 6.0)
    assert all(v.hold and v.speed_cap_mps == 0.0 for v in verdicts)
    assert "no_sources" in verdicts[-1].faults


def test_one_source_that_never_reports_holds_the_machine_however_clear_the_other_is():
    guard, clock = make_guard()
    verdicts = assess_for(
        guard, clock, 6.0, feed=lambda: guard.observe_range("tof_left", 2.0, clock.t)
    )
    assert all(v.hold and v.speed_cap_mps == 0.0 for v in verdicts)
    assert verdicts[-1].faults == ("tof_right",)


def test_a_reading_exactly_on_the_static_stop_line_is_an_obstacle_not_a_slow_zone():
    guard, clock = make_guard(ignored_sources=["tof_right"])
    assess_for(guard, clock, 3.0, feed=lambda: guard.observe_range("tof_left", 2.0, clock.t))
    guard.observe_range("tof_left", 0.2, clock.t)  # the ToF stop line, to the millimetre
    a = guard.assess()
    assert a.obstacles == (("tof_left", 0.2),) and "stop line" in a.reasons[0]
    guard.observe_range("tof_left", 0.2001, clock.t)  # just outside: the stopping model decides
    a = guard.assess()
    assert a.hold and "stop line" not in a.reasons[0] and "stopping envelope" in a.reasons[0]


def test_the_ultrasonic_stop_line_is_inclusive_too():
    guard, clock = make_guard(ignored_sources=["tof_left", "tof_right"])
    guard.register_source("ultrasonic:a", "ultrasonic")
    guard.observe_range("ultrasonic:a", 0.3, clock.t)
    assert "stop line 0.30m" in guard.assess().reasons[0]


# ------------------------------------------------------------------------------ sampling helpers


def tof_manager(stamp, mm=(2000, 2000), status="valid"):
    def reading(side, value):
        return SimpleNamespace(
            distance=value, range_status=status, timestamp=stamp, sensor_side=side
        )

    tof = SimpleNamespace(
        status="online", left_reading=reading("left", mm[0]), right_reading=reading("right", mm[1])
    )
    return SimpleNamespace(tof=tof, ultrasonic=None, ultrasonic_enabled=False)


def us_manager(*entries):
    us = SimpleNamespace(last_readings=list(entries), channel_ids=[e["sensor_id"] for e in entries])
    return SimpleNamespace(tof=None, ultrasonic=us, ultrasonic_enabled=True)


def us_entry(channel="a", **kw):
    entry = {
        "sensor_id": channel,
        "distance_cm": 0.0,
        "valid": False,
        "status": "no_echo",
        "verdict": "unknown",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    entry.update(kw)
    return entry


def ultrasonic_guard(**limits):
    guard = ObstacleGuard(
        ObstacleStopLimits(clear_dwell_s=0.0, ignored_sources=["tof_left", "tof_right"], **limits)
    )
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    return guard


class SteppedWallClock:
    """``datetime`` as the guard's sampler sees it, with the wall clock steppable."""

    offset = 0.0

    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) + timedelta(seconds=cls.offset)


@pytest.fixture
def wall(monkeypatch):
    SteppedWallClock.offset = 0.0
    monkeypatch.setattr(obstacle_stop, "datetime", SteppedWallClock)
    return SteppedWallClock


# ------------------------------------------------------------------------------ SAF-05


def test_a_backward_wall_clock_step_cannot_make_a_dead_sensors_last_reading_fresh(wall):
    clock = Clock()
    guard, _ = make_guard(clock, clear_dwell_s=0.0)
    replayed = tof_manager(datetime.now(UTC))  # the sensor interface replays this reading
    sample_sensor_manager(guard, replayed)
    assert not guard.assess().hold  # fresh and clear at first sight
    wall.offset = -600.0  # chrony steps the wall clock back ten minutes; the sensor is dead
    for _ in range(15):
        clock.t += 0.1
        sample_sensor_manager(guard, replayed)
    a = guard.assess()
    assert a.hold and any("stale" in r for r in a.reasons), a.reasons
    # a reading the sensor really makes after the step is stamped by the stepped clock: fresh
    fresh = tof_manager(SteppedWallClock.now(UTC))
    sample_sensor_manager(guard, fresh)
    assert not guard.assess().hold


def test_a_reading_stamped_in_the_future_is_unknown_and_stays_unknown(wall):
    clock = Clock()
    guard, _ = make_guard(clock, clear_dwell_s=0.0)
    manager = tof_manager(datetime.now(UTC) + timedelta(minutes=10))
    sample_sensor_manager(guard, manager)
    a = guard.assess()
    assert a.hold and set(a.faults) == {"tof_left", "tof_right"}
    assert "future" in guard.snapshot()["sources"]["tof_left"]["detail"]
    wall.offset = 601.0  # wall time catches up with the stamp: it must not turn fresh
    clock.t += 0.1
    sample_sensor_manager(guard, manager)
    assert guard.assess().hold


def test_an_ultrasonic_entry_is_aged_on_the_monotonic_clock_when_it_carries_that_time(wall):
    guard = ultrasonic_guard()
    now = guard.now()
    ok = dict(valid=True, status="ok", verdict="clear", distance_cm=150.0)
    sample_sensor_manager(guard, us_manager(us_entry(**ok, measured_mono=now - 0.05)))
    assert not guard.assess().hold
    wall.offset = -600.0  # the wall stamp now lies in the future: only measured_mono is trusted
    sample_sensor_manager(guard, us_manager(us_entry(**ok, measured_mono=now - 5.0)))
    a = guard.assess()
    assert a.hold and any("stale" in r for r in a.reasons)
    for bad in (math.nan, now + 60.0, "soon", True):
        sample_sensor_manager(guard, us_manager(us_entry(**ok, measured_mono=bad)))
        assert guard.assess().hold, bad


def test_an_ultrasonic_entry_with_no_monotonic_time_still_cannot_be_revived_by_a_step(wall):
    guard = ultrasonic_guard()
    entry = us_entry(valid=True, status="ok", verdict="clear", distance_cm=150.0)
    manager = us_manager(entry)
    sample_sensor_manager(guard, manager)
    assert not guard.assess().hold
    wall.offset = -600.0
    import time

    time.sleep(1.1)  # the real monotonic clock moves past max_reading_age_s
    sample_sensor_manager(guard, manager)
    assert guard.assess().hold


def test_a_measurement_time_ahead_of_the_clock_is_a_fault_never_clamped_to_fresh():
    guard, clock = make_guard(clear_dwell_s=0.0)
    guard.observe_range("tof_left", 2.0, clock.t + 5.0)
    guard.observe_range("tof_right", 2.0, clock.t)
    a = guard.assess()
    assert a.hold and a.faults == ("tof_left",) and "future" in a.reasons[0]
    assert guard.snapshot()["sources"]["tof_left"]["age_s"] is None


# ------------------------------------------------------------------------------ SAF-10


@pytest.mark.parametrize("horizon", [None, 0.0, -0.05, math.nan, math.inf, True])
def test_a_clear_verdict_without_a_usable_horizon_is_unknown_not_clear(horizon):
    guard = ultrasonic_guard()
    guard.register_source("ultrasonic:a", "ultrasonic")
    guard.observe_clear("ultrasonic:a", guard.now(), horizon)
    a = guard.assess()
    assert a.hold and "ultrasonic:a" in a.faults
    assert guard.snapshot()["sources"]["ultrasonic:a"]["detail"].startswith("invalid clear horizon")


def test_a_clear_verdict_with_a_usable_horizon_is_still_clear_up_to_it():
    guard = ultrasonic_guard()
    guard.register_source("ultrasonic:a", "ultrasonic")
    guard.observe_clear("ultrasonic:a", guard.now(), 0.9)
    a = guard.assess()
    assert not a.hold and a.speed_cap_mps == pytest.approx(
        guard.model.max_speed_for_range(0.9, extra_latency_s=0.0), abs=0.02
    )


@pytest.mark.parametrize("clear_to", [None, 0.0, -5.0, math.nan, "far", True])
def test_the_sampler_does_not_turn_a_missing_horizon_into_a_free_path(clear_to):
    guard = ultrasonic_guard()
    entry = us_entry(verdict="clear", status="no_echo", clear_to_cm=clear_to)
    sample_sensor_manager(guard, us_manager(entry))
    a = guard.assess()
    assert a.hold and "ultrasonic:a" in a.faults
    assert "usable horizon" in guard.snapshot()["sources"]["ultrasonic:a"]["detail"]


def test_a_valid_reading_without_a_distance_is_unknown():
    guard = ultrasonic_guard()
    entry = us_entry(valid=True, status="ok", verdict="clear", distance_cm=None)
    sample_sensor_manager(guard, us_manager(entry))
    source = guard.snapshot()["sources"]["ultrasonic:a"]
    assert guard.assess().hold
    assert source["state"] == "unknown" and source["detail"] == "valid reading without a distance"


# ------------------------------------------------------------------------------ SAF-S2


def feed(guard, mm, n, *, start=0):
    """``n`` distinct fresh readings (one stamp each) of ``mm`` on both sides."""
    base = datetime.now(UTC) - timedelta(milliseconds=200)  # a recent past, never the future
    for i in range(start, start + n):
        sample_sensor_manager(guard, tof_manager(base + timedelta(microseconds=i), (mm, mm)))


def test_a_tof_that_repeats_one_in_range_distance_is_frozen_after_the_configured_count():
    guard = ultrasonic_guard()
    guard = ObstacleGuard(ObstacleStopLimits(clear_dwell_s=0.0, frozen_reading_count=10))
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    feed(guard, 1500, 9)
    assert not guard.assess().hold  # nine identical readings: a steady scene so far
    feed(guard, 1500, 1, start=9)
    a = guard.assess()
    assert a.hold and set(a.faults) == {"tof_left", "tof_right"}
    assert "frozen" in guard.snapshot()["sources"]["tof_left"]["detail"]
    feed(guard, 1501, 1, start=10)  # the value moves: it is alive
    assert not guard.assess().hold


def test_replays_of_one_reading_do_not_count_and_a_changed_value_restarts_the_run():
    guard = ObstacleGuard(ObstacleStopLimits(clear_dwell_s=0.0, frozen_reading_count=10))
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    same = tof_manager(datetime.now(UTC), (1500, 1500))
    for _ in range(50):  # the 20 Hz sampler reads the cached reading again and again
        sample_sensor_manager(guard, same)
    assert not guard.assess().hold
    feed(guard, 1500, 8, start=100)
    feed(guard, 1499, 1, start=200)
    feed(guard, 1500, 8, start=300)
    assert not guard.assess().hold


def test_a_constant_no_target_value_and_a_near_obstacle_are_not_called_frozen():
    guard = ObstacleGuard(ObstacleStopLimits(clear_dwell_s=0.0, frozen_reading_count=10))
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    feed(guard, 8190, 40)  # the sensor's out-of-range value is legitimately constant
    assert not guard.assess().hold
    feed(guard, 400, 40, start=100)  # already inside the hold distance: an obstacle, as before
    a = guard.assess()
    assert (
        a.hold and {name for name, _ in a.obstacles} == {"tof_left", "tof_right"} and not a.faults
    )


def test_the_frozen_reading_count_comes_from_the_limits():
    default = ObstacleStopLimits()
    assert default.frozen_reading_count == 50
    guard = ObstacleGuard(ObstacleStopLimits(clear_dwell_s=0.0))
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    feed(guard, 1500, 49)
    assert not guard.assess().hold
    feed(guard, 1500, 1, start=49)
    assert guard.assess().hold
