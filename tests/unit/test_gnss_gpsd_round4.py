"""The gpsd fallback judges a TPV report as a fix, not as text (audit GPS-11, and its mutants).

gpsd is the fallback when the serial port gives nothing. The old code accepted ANY TPV with a
``lat`` and a ``lon``: a report with no fix, a NaN position (``json`` reads ``NaN``), a position
outside the planet, a ``time`` of 2020, the same report for ever, and it reported the course of a
machine that stands still as its heading.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest
from _gnss_fakes import Clock

from backend.src.drivers.sensors.gps_driver import GPSDriver
from backend.src.models.sensor_data import GpsMode

BASE = {
    "class": "TPV",
    "mode": 3,
    "status": 1,
    "lat": 48.1173,
    "lon": 11.5167,
    "alt": 545.4,
    "eph": 2.5,
    "speed": 1.2,
    "track": 84.4,
    "time": "2026-10-25T12:00:05.000Z",
}


def tpv(**changes) -> dict:
    """A TPV object; a ``None`` value removes the key (gpsd omits what it does not know)."""
    out = {**BASE, **changes}
    return {k: v for k, v in out.items() if v is not None}


def at(seconds: int) -> str:
    return f"2026-10-25T12:00:{seconds:02d}.000Z"


@pytest.fixture
def clk():
    return Clock()


@pytest.fixture
def driver(clk):
    return GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clk.mono, wall_clock=clk.wall)


def test_a_tpv_with_a_fix_is_a_reading(driver, clk):
    r = driver._reading_from_tpv(tpv(), clk.t)
    assert r is not None
    assert (r.latitude, r.longitude, r.altitude) == (48.1173, 11.5167, 545.4)
    assert r.rtk_status == "GPS_FIX" and r.speed == 1.2 and r.heading == 84.4
    assert r.accuracy == pytest.approx(2.5) and r.mode == GpsMode.LC29H_UART
    assert (clk.wall() - r.timestamp).total_seconds() == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("mode", [0, 1, None, "3", 3.0, True, -1])
def test_a_tpv_without_a_2d_or_3d_fix_is_no_reading_even_with_a_position(driver, clk, mode):
    """gpsd normally omits lat/lon below mode 2; a misbehaving one does not."""
    assert driver._reading_from_tpv(tpv(mode=mode), clk.t) is None


@pytest.mark.parametrize("mode", [2, 3])
def test_a_2d_fix_is_enough(driver, clk, mode):
    assert driver._reading_from_tpv(tpv(mode=mode), clk.t) is not None


@pytest.mark.parametrize(
    ("changes", "label"),
    [
        ({"lat": float("nan")}, "nan latitude"),
        ({"lon": float("nan")}, "nan longitude"),
        ({"lat": float("inf")}, "infinite latitude"),
        ({"lon": float("-inf")}, "infinite longitude"),
        ({"lat": 90.0001}, "past the pole"),
        ({"lat": -999.0, "lon": 999.0}, "off the planet"),
        ({"lon": 180.0001}, "past the antimeridian"),
        ({"lat": "48.1"}, "a string latitude"),
        ({"lat": True}, "a boolean latitude"),
        ({"lat": None}, "no latitude"),
        ({"lon": None}, "no longitude"),
    ],
)
def test_a_position_that_is_not_a_position_is_no_reading(driver, clk, changes, label):
    assert driver._reading_from_tpv(tpv(**changes), clk.t) is None, label


def test_the_poles_and_the_antimeridian_themselves_are_positions(driver, clk):
    r = driver._reading_from_tpv(tpv(lat=-90.0, lon=180.0), clk.t)
    assert r is not None and (r.latitude, r.longitude) == (-90.0, 180.0)


def test_a_nan_that_json_itself_reads_never_gets_through(driver, clk):
    """``json.loads`` accepts the literals NaN and Infinity: the path the socket data takes."""
    obj = json.loads('{"class": "TPV", "mode": 3, "lat": NaN, "lon": NaN, "time": "x"}')
    assert driver._reading_from_tpv(obj, clk.t) is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, "GPS_FIX"),  # gpsd before 3.18 has none: mode says there is a fix
        (0, "GPS_FIX"),
        (1, "GPS_FIX"),
        (2, "DGPS"),
        (3, "RTK_FIXED"),
        (4, "RTK_FLOAT"),
    ],
)
def test_the_tpv_status_is_the_fix_status(driver, clk, status, expected):
    r = driver._reading_from_tpv(tpv(status=status), clk.t)
    assert r is not None and r.rtk_status == expected


@pytest.mark.parametrize("status", [5, 6, 7, 8, 9, 99, "x"])
def test_a_tpv_status_that_is_not_a_position_fix_is_no_reading(driver, clk, status):
    """Dead reckoning, a surveyed time-only fix, a simulator, a status nobody defined."""
    assert driver._reading_from_tpv(tpv(status=status), clk.t) is None


@pytest.mark.parametrize(
    ("eph", "accuracy"),
    [
        (2.5, 2.5),
        (0.1, 0.5),  # a plain fix is never more accurate than the floor
        (0, None),
        (-1.0, None),
        (float("nan"), None),
        (float("inf"), None),
        ("2.5", None),
        (True, None),
        (None, None),
    ],
)
def test_only_a_real_eph_is_an_accuracy(driver, clk, eph, accuracy):
    r = driver._reading_from_tpv(tpv(eph=eph), clk.t)
    assert r.accuracy == (pytest.approx(accuracy) if accuracy is not None else None)


def test_an_rtk_status_gets_no_typical_accuracy_from_gpsd(driver, clk):
    """The typical RTK figure needs a GGA that vouches for its corrections; a TPV has none."""
    assert driver._reading_from_tpv(tpv(status=3, eph=None, time=at(5)), clk.t).accuracy is None
    clk.advance(1.0)
    floored = driver._reading_from_tpv(tpv(status=3, eph=0.001, time=at(6)), clk.t)
    assert floored.accuracy == pytest.approx(0.03)  # a real eph, floored at RTK fixed


@pytest.mark.parametrize(
    ("speed", "track", "heading"),
    [
        (1.2, 271.0, 271.0),
        (0.5, 10.0, 10.0),  # exactly the threshold still counts
        (0.49, 271.0, None),  # standing still: the track is noise
        (0.0, 271.0, None),
        (1.2, 361.0, None),
        (1.2, -1.0, None),
        (1.2, float("nan"), None),
        (1.2, None, None),
        (None, 271.0, None),
        (-1.0, 271.0, None),
    ],
)
def test_the_track_is_a_heading_only_when_the_machine_is_moving(driver, clk, speed, track, heading):
    r = driver._reading_from_tpv(tpv(speed=speed, track=track), clk.t)
    assert r.heading == heading
    assert r.speed == (speed if speed is not None and speed >= 0 else None)


def test_a_tpv_without_a_time_is_no_reading_a_fix_that_cannot_be_dated_is_not_fresh(driver, clk):
    assert driver._reading_from_tpv(tpv(time=None), clk.t) is None
    assert driver._reading_from_tpv(tpv(time="not a time"), clk.t) is None
    assert driver._reading_from_tpv(tpv(time=12345), clk.t) is None


def test_the_same_tpv_again_is_not_a_new_fix(driver, clk):
    assert driver._reading_from_tpv(tpv(time=at(5)), clk.t) is not None
    clk.advance(1.0)
    assert driver._reading_from_tpv(tpv(time=at(5)), clk.t) is None  # a stuck gpsd repeats itself
    assert driver.epochs_refused == 1
    clk.advance(1.0)
    assert driver._reading_from_tpv(tpv(time=at(6)), clk.t) is not None


def test_an_older_tpv_after_a_newer_one_is_not_a_new_fix(driver, clk):
    assert driver._reading_from_tpv(tpv(time=at(30)), clk.t) is not None
    clk.advance(1.0)
    assert driver._reading_from_tpv(tpv(time="2020-01-01T00:00:00.000Z"), clk.t) is None
    assert driver._reading_from_tpv(tpv(time=at(10)), clk.t) is None
    clk.advance(1.0)
    assert driver._reading_from_tpv(tpv(time=at(32)), clk.t) is not None


def test_a_tpv_read_late_is_dated_by_its_own_time(driver, clk):
    first = driver._reading_from_tpv(tpv(time=at(5)), clk.t)
    clk.advance(1.0)
    second = driver._reading_from_tpv(tpv(time=at(6)), clk.t)
    clk.advance(1.0)  # the host looks again 1 s later and the next TPV is 1 s old already
    third = driver._reading_from_tpv(tpv(time=at(7)), clk.t - 1.0)
    assert (clk.wall() - first.timestamp).total_seconds() == pytest.approx(2.0, abs=0.01)
    assert (clk.wall() - second.timestamp).total_seconds() == pytest.approx(1.0, abs=0.01)
    assert (clk.wall() - third.timestamp).total_seconds() == pytest.approx(1.0, abs=0.01)
    clk.advance(0.0)
    late = driver._reading_from_tpv(tpv(time=at(8)), clk.t + 1.5)  # read 1.5 s after its time
    assert late is not None


# --------------------------------------------------------------------------------- the socket path


class Gpsd(threading.Thread):
    """A gpsd that answers one connection with ``payload`` (bytes) and then stays quiet."""

    def __init__(self, payload: bytes, hold: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.payload, self.hold = payload, hold
        self.start()

    def run(self) -> None:
        try:
            conn, _ = self.server.accept()
            conn.recv(256)  # ?WATCH=...
            conn.sendall(self.payload)
            time.sleep(self.hold)
            conn.close()
        except OSError:
            pass
        finally:
            self.server.close()


def line(obj: dict) -> bytes:
    return json.dumps(obj).encode() + b"\n"


def ask(monkeypatch, payload: bytes, **driver_kw):
    server = Gpsd(payload)
    monkeypatch.setenv("GPSD_HOST", "127.0.0.1")
    monkeypatch.setenv("GPSD_PORT", str(server.port))
    driver = GPSDriver({"mode": GpsMode.LC29H_UART}, **driver_kw)
    started = time.monotonic()
    reading = driver._read_from_gpsd(timeout_sec=0.6)
    return reading, time.monotonic() - started


def test_a_tpv_that_gpsd_sends_is_read_from_the_socket_past_its_other_reports(monkeypatch):
    payload = (
        line({"class": "VERSION", "release": "3.25"})
        + line({"class": "SKY", "satellites": []})
        + line(tpv(mode=1, lat=None, lon=None))  # no fix yet
        + line(tpv(time=at(9)))
    )
    reading, _ = ask(monkeypatch, payload)
    assert reading is not None and reading.latitude == 48.1173


def test_gpsd_with_only_a_report_without_a_fix_gives_nothing_within_the_bound(monkeypatch):
    reading, took = ask(monkeypatch, line(tpv(mode=1)) + line(tpv(lat=float("nan"))))
    assert reading is None and took < 1.5


def test_a_gpsd_that_sends_for_ever_without_a_line_end_is_bounded(monkeypatch):
    reading, took = ask(monkeypatch, b"x" * 400_000)
    assert reading is None and took < 1.5


def test_a_gpsd_that_says_nothing_is_bounded(monkeypatch):
    reading, took = ask(monkeypatch, b"")
    assert reading is None and took < 1.5


def test_partial_and_garbled_json_lines_are_skipped(monkeypatch):
    payload = (
        b'{"class": "TPV", "mode": 3, "lat"\n' + b"\xff\xfe not json\n" + line(tpv(time=at(9)))
    )
    reading, _ = ask(monkeypatch, payload)
    assert reading is not None


def test_a_negative_speed_and_a_track_outside_the_circle_are_unknown(driver, clk):
    r = driver._reading_from_tpv(tpv(speed=-0.05, track=-0.05), clk.t)
    assert r is not None and r.speed is None and r.heading is None
    clk.advance(1.0)
    r = driver._reading_from_tpv(tpv(time=at(6), speed=1.0, track=-0.05), clk.t)
    assert r is not None and r.speed == 1.0 and r.heading is None  # moving, but no such course
    clk.advance(1.0)
    r = driver._reading_from_tpv(tpv(time=at(7), speed=1.0, track=0.0), clk.t)
    assert r is not None and r.heading == 0.0  # north is a heading


def test_a_tpv_that_arrives_late_is_stamped_in_the_past(driver, clk):
    assert driver._reading_from_tpv(tpv(time=at(5)), clk.t) is not None
    clk.advance(4.0)  # the host looks 4 s later, the receiver's time says 1 s later
    late = driver._reading_from_tpv(tpv(time=at(6)), clk.t)
    assert late is not None
    assert (clk.wall() - late.timestamp).total_seconds() == pytest.approx(3.0, abs=0.05)


def test_a_json_line_that_is_not_an_object_does_not_hide_the_tpv_behind_it(monkeypatch):
    reading, _ = ask(monkeypatch, b'[1, 2, 3]\n"text"\n42\n' + line(tpv(time=at(9))))
    assert reading is not None
