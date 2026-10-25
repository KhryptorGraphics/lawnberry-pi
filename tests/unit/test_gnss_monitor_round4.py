"""The GNSS monitor refuses a position that is not a position (audit NAV-06, GPS-07, GPS-10) and
the boundaries and memory its verdicts rest on (the surviving mutants of the third review).

NAV-06: a NaN or infinite latitude/longitude was a healthy fix, and ``max(-1, min(1, nan / 45))``
is ``1.0`` in Python, so the leg controller pivoted at full lever on it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from backend.src.core.freshness import ReceiptAges
from backend.src.drivers.sensors import gps_driver
from backend.src.models.sensor_data import GpsReading
from backend.src.nav.gps_degradation import (
    USABLE_FIX_STATUS,
    GPSDegradationConfig,
    GPSDegradationMonitor,
)

EPOCH = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)


class Clocks:
    def __init__(self) -> None:
        self.t = 500.0

    def mono(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return EPOCH + timedelta(seconds=self.t - 500.0)

    def stamp(self, age: float = 0.0) -> datetime:
        return self.wall() - timedelta(seconds=age)


def fix(c: Clocks, *, lat=40.0, lon=-83.0, acc=0.05, status="RTK_FIXED", age=0.0) -> GpsReading:
    return GpsReading(
        latitude=lat, longitude=lon, accuracy=acc, rtk_status=status, timestamp=c.stamp(age)
    )


def monitor(c: Clocks, *, dwell: float = 0.0) -> GPSDegradationMonitor:
    cfg = GPSDegradationConfig(max_accuracy_m=1.0, max_fix_age_s=2.0, resume_dwell_s=dwell)
    return GPSDegradationMonitor(cfg, clock=c.mono, wall_clock=c.wall)


def settle(m: GPSDegradationMonitor, c: Clocks, seconds: float = 4.0, **kw):
    for _ in range(round(seconds / 0.5) + 1):
        c.t += 0.5
        m.observe(fix(c, **kw))
    return m.assess()


# ------------------------------------------------------------------------------------------ NAV-06


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (math.nan, math.nan),
        (math.nan, -83.0),
        (40.0, math.nan),
        (math.inf, -83.0),
        (40.0, -math.inf),
        (90.0001, -83.0),
        (-90.0001, -83.0),
        (40.0, 180.0001),
        (40.0, -180.0001),
        (999.0, -999.0),
        (1e308, 1e308),
    ],
    ids=repr,
)
def test_a_position_that_is_not_a_place_on_earth_is_never_a_healthy_fix(lat, lon):
    c = Clocks()
    m = monitor(c)
    a = settle(m, c, lat=lat, lon=lon)  # fresh, accurate, RTK fixed: only the position is wrong
    assert a.degraded and any("no usable position" in r for r in a.reasons), a.reasons


@pytest.mark.parametrize("lat, lon", [("40", "-83"), (True, False), (b"40", b"-83"), ([40], [-83])])
def test_a_position_that_is_not_a_number_is_not_a_healthy_fix(lat, lon):
    c = Clocks()
    m = monitor(c)
    reading = fix(c)
    for _ in range(8):
        c.t += 0.5
        reading = fix(c)
        object.__setattr__(reading, "latitude", lat)  # what a model validator would have stopped
        object.__setattr__(reading, "longitude", lon)
        m.observe(reading)
    a = m.assess()
    assert a.degraded and any("no usable position" in r for r in a.reasons), a.reasons


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(40.0, -83.0), (0.0, -83.0), (40.0, 0.0), (90.0, 180.0), (-90.0, -180.0), (51.4778, -0.0015)],
)
def test_a_real_place_is_a_usable_position_including_one_on_the_equator_or_the_meridian(lat, lon):
    """(0, x) and (y, 0) are places (Greenwich is a mowing site); only (0, 0) is a null position."""
    c = Clocks()
    assert settle(monitor(c), c, lat=lat, lon=lon).healthy


def test_the_null_position_is_no_position():
    c = Clocks()
    a = settle(monitor(c), c, lat=0.0, lon=0.0)
    assert a.degraded and a.reasons[0] == "no position"


def test_a_reading_with_no_coordinates_is_no_position():
    c = Clocks()
    a = settle(monitor(c), c, lat=None, lon=None)
    assert a.degraded and a.reasons[0] == "no position"
    a = settle(monitor(c), c, lat=40.0, lon=None)
    assert a.degraded and a.reasons[0] == "no position"


# ------------------------------------------------------------------------------------------ GPS-07


@pytest.mark.parametrize("status", sorted(USABLE_FIX_STATUS))
def test_a_position_fix_status_is_usable(status):
    c = Clocks()
    assert settle(monitor(c), c, status=status).healthy


@pytest.mark.parametrize(
    "status",
    ["NO_FIX", "DEAD_RECKONING", "MANUAL", "SIMULATION", "FLOAT", "rtk_fixed", "", "UNKNOWN", "3"],
)
def test_a_status_that_is_not_a_position_fix_is_unusable_even_if_nobody_listed_it(status):
    """The monitor used to refuse four named statuses and pass the rest: a status nobody had
    thought of (``None`` out of an unknown GGA quality) was a healthy fix."""
    c = Clocks()
    a = settle(monitor(c), c, status=status)
    assert a.degraded and any(f"fix status {status}" in r for r in a.reasons), a.reasons


def test_a_source_that_reports_no_status_is_not_refused_for_that_alone():
    """The simulator and the interface placeholder carry no status: their other facts judge."""
    c = Clocks()
    assert settle(monitor(c), c, status=None).healthy


def test_every_status_the_driver_can_report_is_judged_the_same_way_by_the_monitor():
    """The driver's vocabulary and the monitor's must not drift: a fix is usable exactly where
    the driver says it is a position fix."""
    reported = (
        set(gps_driver._QUALITY_STATUS.values())
        | set(gps_driver._RMC_MODE_STATUS.values())
        | set(gps_driver._GPSD_STATUS.values())
        | {"NO_FIX"}
    )
    assert {s for s in reported if s in USABLE_FIX_STATUS} == set(gps_driver.FIX_STATUSES)
    assert USABLE_FIX_STATUS == gps_driver.FIX_STATUSES


# -------------------------------------------------------------------------------------- boundaries


def test_a_fix_exactly_as_old_as_the_limit_is_usable_and_a_hair_older_is_not():
    c = Clocks()
    m = monitor(c)
    m.observe(fix(c, age=2.0))
    assert m.assess().healthy, m.assess().reasons
    m.observe(fix(c, age=2.001))
    a = m.assess()
    assert a.degraded and any("old" in r for r in a.reasons)


def test_an_accuracy_exactly_the_limit_is_usable_and_a_hair_worse_is_not():
    c = Clocks()
    m = monitor(c)
    m.observe(fix(c, acc=1.0))
    assert m.assess().healthy, m.assess().reasons
    c.t += 0.5
    m.observe(fix(c, acc=1.0000001))
    a = m.assess()
    assert a.degraded and any("exceeds" in r for r in a.reasons)


# --------------------------------------------------------------------------- memory of the verdict


def test_nothing_new_this_tick_does_not_erase_the_last_fix_it_ages_it():
    """``observe(None)`` means the interface had nothing new: the old fix must keep ageing, not
    vanish (a vanished fix would also reset what the monitor knows about its age)."""
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    for _ in range(3):
        c.t += 0.5
        m.observe(None)
    a = m.assess()
    assert a.healthy and a.fix_age_s == pytest.approx(1.5)
    assert a.rtk_status == "RTK_FIXED" and a.accuracy_m == pytest.approx(0.05)
    for _ in range(3):
        c.t += 0.5
        m.observe(None)
    assert m.assess().degraded  # three seconds on it is too old: it was kept, and it aged


def test_a_moment_of_trouble_restarts_the_resume_dwell():
    """Three seconds of GOOD fixes are needed since the last bad one, not since the first good."""
    c = Clocks()
    m = monitor(c, dwell=3.0)
    for _ in range(7):  # 3.0 s of good fixes: healthy
        c.t += 0.5
        m.observe(fix(c))
    assert m.assess().healthy
    c.t += 0.5
    m.observe(fix(c, acc=5.0))  # one bad reading
    assert m.assess().degraded
    for _ in range(5):  # 2.5 s of good ones: not enough
        c.t += 0.5
        m.observe(fix(c))
        assert m.assess().degraded, m.assess().reasons
    for _ in range(2):
        c.t += 0.5
        m.observe(fix(c))
    assert m.assess().healthy


def test_a_moment_of_trouble_while_recovering_starts_the_dwell_again():
    """The dwell counts from the LAST bad moment, also when that moment falls inside a recovery
    that had already begun: a start time left over from the first attempt would let one blip
    resume the machine as soon as three seconds had passed since THEN."""
    c = Clocks()
    m = monitor(c, dwell=3.0)
    c.t += 0.5
    m.observe(fix(c))  # the first good fix: the first dwell starts (t0)
    for _ in range(3):  # 1.5 s of it
        c.t += 0.5
        m.observe(fix(c))
    assert m.assess().degraded
    c.t += 0.5
    m.observe(fix(c, acc=5.0))  # trouble in the middle of the recovery
    assert m.assess().degraded
    for _ in range(6):  # the first good fix after the blip starts the dwell; 2.5 s of it: no
        c.t += 0.5
        m.observe(fix(c))
        assert m.assess().degraded, m.assess().reasons  # (4.5 s and more since t0 by now)
    c.t += 0.5
    m.observe(fix(c))  # 3.0 s of good fixes since the blip
    assert m.assess().healthy


def test_the_machine_does_not_resume_a_moment_before_the_dwell_is_over():
    c = Clocks()
    m = monitor(c, dwell=3.0)
    c.t += 0.5
    m.observe(fix(c))  # the dwell starts
    c.t += 2.95
    m.observe(fix(c))  # 2.95 s of good fixes: 0.05 s short
    a = m.assess()
    assert a.degraded and "recovering" in a.reasons[0]
    c.t += 0.05
    m.observe(fix(c))
    a = m.assess()
    assert a.healthy and a.reasons == ()  # and a healthy verdict carries no leftover reason


def test_a_degraded_monitor_says_how_long_it_still_needs():
    c = Clocks()
    m = monitor(c, dwell=3.0)
    c.t += 0.5
    m.observe(fix(c))
    c.t += 1.0
    m.observe(fix(c))
    a = m.assess()
    assert a.degraded and "recovering: 2.0s" in a.reasons[0]


# ---------------------------------------------------------------------- ReceiptAges (freshness.py)


def test_a_stamp_given_as_an_iso_string_is_aged_like_a_datetime():
    ages = ReceiptAges(lambda: 100.0)
    wall = EPOCH + timedelta(seconds=2.0)
    assert ages.taken_at("k", "2026-10-25T12:00:00+00:00", wall, now=100.0) == pytest.approx(98.0)
    assert ages.taken_at("other", "yesterday-ish", wall, now=100.0) is None  # garbage: unknown


def test_a_naive_stamp_is_read_as_utc():
    """Every model of this code base stamps with UTC; a naive stamp compared with an aware clock
    used to be a TypeError, not an age."""
    ages = ReceiptAges(lambda: 100.0)
    wall = EPOCH + timedelta(seconds=2.0)
    assert ages.taken_at("k", datetime(2026, 10, 25, 12, 0, 0), wall, now=100.0) == pytest.approx(
        98.0
    )


def test_a_stamp_that_comes_back_after_a_stamp_less_reading_keeps_its_first_seen_time():
    """The remembered time is what keeps a replayed reading ageing on the monotonic clock. A bare
    reading in between must not make the next sight of the old stamp 'new': with the wall clock
    stepped back meanwhile that would be a revival."""
    ages = ReceiptAges(lambda: 100.0)
    stamp = EPOCH
    assert ages.taken_at("k", stamp, EPOCH + timedelta(seconds=2.0), now=100.0) == pytest.approx(
        98.0
    )
    assert ages.taken_at("k", None, EPOCH + timedelta(seconds=3.0), now=101.0) is None
    # chrony stepped the wall clock back 7 s: judged afresh the old stamp would be 3 s old
    again = ages.taken_at("k", stamp, EPOCH + timedelta(seconds=3.0), now=108.0)
    assert again == pytest.approx(98.0)


def test_a_stamp_from_the_future_stays_unusable_when_wall_time_reaches_it():
    ages = ReceiptAges(lambda: 100.0)
    future = EPOCH + timedelta(seconds=600.0)
    assert ages.taken_at("k", future, EPOCH, now=100.0) is None
    assert ages.taken_at("k", future, EPOCH + timedelta(seconds=300.0), now=101.0) is None
    # the wall clock gets there: still the reading it was, still unknown
    assert ages.taken_at("k", future, EPOCH + timedelta(seconds=601.0), now=102.0) is None
    # a different stamp is a new reading and is judged afresh
    newer = EPOCH + timedelta(seconds=601.0)
    assert ages.taken_at("k", newer, EPOCH + timedelta(seconds=601.5), now=103.0) == pytest.approx(
        102.5
    )


def test_a_fix_stamped_in_the_future_stays_degraded_after_the_dwell_when_wall_time_catches_up():
    """The monitor-level twin: the old test looked at one tick after the catch-up, which the
    resume dwell alone would have explained."""
    c = Clocks()
    m = monitor(c, dwell=3.0)
    future = fix(c, age=-600.0)
    for _ in range(6):
        c.t += 0.5
        m.observe(future)
    assert m.assess().degraded and m.assess().fix_age_s is None
    c.t += 600.0  # the wall clock reaches the stamp
    for _ in range(20):  # a long time of the very same reading: far more than the dwell
        c.t += 0.5
        m.observe(future)
    a = m.assess()
    assert a.degraded and any("future" in r for r in a.reasons), a.reasons
