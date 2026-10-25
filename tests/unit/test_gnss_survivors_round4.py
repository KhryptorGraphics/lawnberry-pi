"""Pins for what the mutation pass over the GNSS code found nothing guarding (third review).

Each test says the claim it guards; a number that is only a tuned parameter (a timeout, a retry
period, a simulated coordinate) is not pinned here, and the audit lists them as such.

* the receiver-clock judge (``EpochClock``): a replay, a jump, the tolerance and the delay it
  reports must not depend on the value the host clock happens to read (a host up for a day);
* the sigma an epoch may borrow from the one before it, and for how long;
* the assembler (epochs, repeats, the kinds a receiver is waited for);
* every truncation of a GGA / RMC, the checksum, the line bound, the candidate ports and bauds;
* ``read_position`` hands back what its own read made, the gpsd fallback is throttled;
* the freshness helpers and the timezone lookup's coordinate rules.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from _gnss_fakes import Clock, epoch_of, install_serial, nmea
from test_gnss_driver_round4 import Rig

from backend.src.core.freshness import ReceiptAges, parse_stamp
from backend.src.drivers.sensors import gps_driver as gd
from backend.src.drivers.sensors.gps_driver import (
    EpochClock,
    Gga,
    GPSDriver,
    NmeaLineReader,
    Rmc,
    SerialPortInfo,
    nmea_checksum_ok,
    parse_gga,
    parse_rmc,
    reading_age_s,
)
from backend.src.models.sensor_data import GpsMode
from backend.src.services import timezone_service as tz

NOON = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)


# ------------------------------------------------------------------------------------ EpochClock

HOST_CLOCKS = pytest.mark.parametrize(
    "t0", [0.0, 100_000.0], ids=["host clock near zero", "host up for more than a day"]
)


def started(t0: float) -> EpochClock:
    clk = EpochClock()
    assert clk.accept(1000.0, t0).accepted
    return clk


@HOST_CLOCKS
def test_a_late_read_shows_its_delay_whatever_the_host_clock_reads(t0):
    clk = started(t0)
    assert clk.accept(1001.0, t0 + 1.0).delay_s == 0.0
    late = clk.accept(1002.0, t0 + 5.0)  # read 3 s later than the receiver sent it
    assert late.accepted and late.delay_s == pytest.approx(3.0, abs=0.01)


@HOST_CLOCKS
def test_a_time_that_is_not_later_is_stale(t0):
    clk = started(t0)
    assert clk.accept(1001.0, t0 + 1.0).accepted
    for u in (1001.0, 1000.5, 1000.0):
        verdict = clk.accept(u, t0 + 2.0)
        assert not verdict.accepted and verdict.reason == "stale"


@HOST_CLOCKS
def test_a_jump_is_held_back_and_only_a_time_line_that_continues_it_is_followed(t0):
    clk = started(t0)
    assert clk.accept(1100.0, t0 + 1.0).reason == "jump"
    again = clk.accept(1100.0, t0 + 2.0)  # the same wild time again is a repeat, not a line
    assert not again.accepted and again.reason == "jump"
    followed = clk.accept(1102.0, t0 + 3.0)
    assert followed.accepted and followed.reason == "rebased"


@HOST_CLOCKS
def test_the_tolerance_is_exactly_five_seconds(t0):
    clk = started(t0)
    edge = clk.accept(1007.0, t0 + 2.0)  # 7 s of receiver time in 2 s of host time: 5.0 ahead
    assert edge.accepted and edge.reason == "ok"
    beyond = clk.accept(1014.25, t0 + 4.0)  # 5.25 ahead of the host
    assert not beyond.accepted and beyond.reason == "jump"


@HOST_CLOCKS
def test_a_continuation_within_the_tolerance_rebases_and_one_beyond_it_does_not(t0):
    clk = started(t0)
    assert clk.accept(1100.0, t0 + 1.0).reason == "jump"
    edge = clk.accept(1107.0, t0 + 3.0)  # 7 s of receiver time in 2 s of host: 5.0 apart
    assert edge.accepted and edge.reason == "rebased"
    clk = started(t0)
    clk.accept(1100.0, t0 + 1.0)
    beyond = clk.accept(1107.25, t0 + 3.0)
    assert not beyond.accepted and beyond.reason == "jump"


@HOST_CLOCKS
def test_a_normal_epoch_forgets_a_held_back_jump(t0):
    clk = started(t0)
    assert clk.accept(1100.0, t0 + 1.0).reason == "jump"
    assert clk.accept(1001.0, t0 + 1.5).accepted  # the real time line goes on
    after = clk.accept(1102.0, t0 + 3.0)  # would continue the forgotten jump
    assert not after.accepted and after.reason == "jump"


@HOST_CLOCKS
def test_after_a_rebase_the_delay_estimate_starts_from_the_rebase(t0):
    """The baseline drifts up slowly (the receiver's crystal); the drift counts from the last
    time the baseline was set, not from the start of the session."""
    clk = EpochClock()
    clk.accept(1000.0, t0)
    assert clk.accept(31000.0, t0 + 20_001.0).reason == "jump"  # 10000 s ahead of real time
    assert clk.accept(31002.0, t0 + 20_003.0).reason == "rebased"
    late = clk.accept(31003.0, t0 + 20_007.0)  # 3 s late
    assert late.accepted and late.delay_s == pytest.approx(3.0, abs=0.01)


# --------------------------------------------------------------------------- the borrowed sigma

ORDER = ["GPS_FIX", "DGPS", "RTK_FLOAT", "RTK_FIXED"]
QUALITY = {"GPS_FIX": 1, "DGPS": 2, "RTK_FLOAT": 5, "RTK_FIXED": 4}


def epoch(key: float, status: str, *, gst: float | None = None, seen: bool | None = None, t=0.0):
    gga = Gga(key, QUALITY[status], 48.0, 11.0, 500.0, 10, 0.8, None)
    return gd._Epoch(
        key=key,
        t_pos=t,
        gga=gga,
        gst_seen=(gst is not None) if seen is None else seen,
        gst_sigma=gst,
        gst_t=t,
        kinds={"GGA"},
    )


def borrow(*, prev_status="GPS_FIX", status="GPS_FIX", dkey=1.0, age=0.5, prev_gst=0.5, seen=None):
    clk = Clock()
    drv = GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clk.mono, wall_clock=clk.wall)
    st = gd._Stream(clk.t)
    st.last_delivered = epoch(100.0, prev_status, gst=prev_gst, seen=seen, t=clk.t - age)
    return drv._sigma_for(st, epoch(100.0 + dkey, status), status)


@pytest.mark.parametrize("prev_status", ORDER)
@pytest.mark.parametrize("status", ORDER)
def test_a_borrowed_sigma_is_only_for_a_fix_that_is_not_worse(prev_status, status):
    expected = 0.5 if ORDER.index(status) >= ORDER.index(prev_status) else None
    assert borrow(prev_status=prev_status, status=status) == expected


@pytest.mark.parametrize(
    ("dkey", "age", "borrowed"),
    [
        (1.0, 0.5, True),
        (2.0, 0.5, True),  # exactly as old as the limit still describes the fix
        (2.01, 0.5, False),
        (0.05, 0.5, True),
        (0.0, 0.5, False),  # the same receiver time is not a later epoch
        (-0.05, 0.5, False),
        (1.0, 2.0, True),
        (1.0, 2.01, False),  # read longer ago than the limit on OUR clock
    ],
)
def test_a_borrowed_sigma_must_be_recent_in_the_receivers_time_and_in_ours(dkey, age, borrowed):
    assert borrow(dkey=dkey, age=age) == (0.5 if borrowed else None)


def test_a_previous_gst_without_a_usable_sigma_gives_nothing_to_borrow():
    assert borrow(prev_gst=None, seen=True) is None
    assert borrow(prev_gst=None, seen=False) is None


def test_an_epoch_with_its_own_gst_keeps_its_own_sigma_even_if_that_is_none():
    clk = Clock()
    drv = GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clk.mono, wall_clock=clk.wall)
    st = gd._Stream(clk.t)
    st.last_delivered = epoch(100.0, "GPS_FIX", gst=0.5, t=clk.t - 0.5)
    own = epoch(101.0, "GPS_FIX", gst=None, seen=True)  # a GST that said nothing usable
    assert drv._sigma_for(st, own, "GPS_FIX") is None


def test_a_sigma_survives_from_one_reading_to_the_next_through_the_whole_read_path(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    rig.at(0.5, 0)
    rig.at(1.5, 1)
    rig.at(2.5, 2, gst_sigma=None)  # this epoch's GST never came
    first, second = rig.next(), rig.next()
    assert first is not None and second is not None and epoch_of(second) == 2
    assert first.accuracy is not None and first.accuracy > 4.0
    assert second.accuracy == pytest.approx(first.accuracy)


# -------------------------------------------------------------------------------- the assembler


def test_a_new_session_waits_for_every_kind_and_afterwards_for_those_seen_in_five_seconds():
    asm = gd._Assembler()
    assert asm.discovering and asm.repeats == 0
    assert asm.expected(0.0) == frozenset({"GGA", "RMC", "GST"})
    asm.discovering = False
    asm.learned = {"GGA": 100.0, "GST": 100.0}
    assert asm.expected(105.0) == {"GGA", "GST"}  # seen exactly five seconds ago
    assert asm.expected(105.01) == frozenset()


def test_the_first_closed_epoch_ends_the_discovery_and_a_repeat_counts_once():
    asm = gd._Assembler()
    gga = Gga(10.0, 1, 48.0, 11.0, 500.0, 10, 0.8, None)
    assert asm.add("GGA", 10.0, gga, 5.0) == []
    assert asm.add("GGA", 10.0, gga, 5.2) == []  # a second GGA of the epoch: the first stands
    assert asm.repeats == 1 and asm.cur.t_pos == 5.0 and asm.discovering
    closed = asm.add("GGA", 11.0, gga, 6.0)
    assert [e.key for e in closed] == [10.0] and not asm.discovering


def test_an_epoch_with_no_gga_is_dated_by_its_rmc_not_by_whatever_came_first():
    asm = gd._Assembler()
    asm.add("GST", 12.0, 0.5, 9.0)
    rmc = Rmc(12.0, True, "A", 48.0, 11.0, 0.3, 90.0)
    asm.add("RMC", 12.0, rmc, 9.3)
    assert asm.cur.t_pos == 9.3 and asm.cur.gst_seen and asm.cur.rmc is rmc


# --------------------------------------------------------------------------- the line reader


def test_a_sentence_of_exactly_the_longest_length_is_kept_and_one_more_is_not():
    n = NmeaLineReader.MAX_LINE
    ok, too_long = b"$" + b"A" * (n - 1), b"$" + b"A" * n
    assert NmeaLineReader().feed(ok + b"\n") == [ok.decode()]
    assert NmeaLineReader().feed(too_long + b"\n") == []
    reader = NmeaLineReader()
    reader.feed(b"x" * 300 + ok)  # unfinished, longer than the bound, its tail is a sentence
    assert reader.feed(b"\n") == [ok.decode()]
    reader = NmeaLineReader()
    reader.feed(b"x" * 300 + too_long)  # one character more than the bound after the "$"
    assert reader.feed(b"\n") == []
    reader = NmeaLineReader()
    reader.feed(b"x" * (n + 1))  # an over-long tail with no "$" in it keeps nothing
    assert reader.feed(b"$GNGGA\n") == ["$GNGGA"]


# --------------------------------------------------------------------------------- the parsers

GGA_FIELDS = ["GNGGA", "120000.00", "4807.038", "N", "01131.000", "E", "1", "08", "0.9", "545.4"]
GGA_FIELDS += ["M", "46.9", "M", "1.0", "0000"]
RMC_FIELDS = ["GNRMC", "120000.00", "A", "4807.038", "N", "01131.000", "E", "0.5", "84.4"]
RMC_FIELDS += ["251026", "", "", "A"]


@pytest.mark.parametrize("n", range(1, len(GGA_FIELDS) + 1))
def test_a_gga_cut_short_at_any_field_is_read_as_far_as_it_goes(n):
    got = parse_gga(nmea(",".join(GGA_FIELDS[:n])).decode())
    assert (got is None) == (n < 7)  # the fix quality is the seventh field
    if got is not None:
        assert got.quality == 1 and (got.lat is not None) == (n > 6)
        assert got.satellites == (8 if n > 7 else None)
        assert got.hdop == (0.9 if n > 8 else None)
        assert got.altitude_m == (545.4 if n > 9 else None)
        assert got.correction_age_s == (1.0 if n > 13 else None)


@pytest.mark.parametrize("n", range(1, len(RMC_FIELDS) + 1))
def test_an_rmc_cut_short_at_any_field_is_read_as_far_as_it_goes(n):
    got = parse_rmc(nmea(",".join(RMC_FIELDS[:n])).decode())
    assert (got is None) == (n < 3)
    if got is not None:
        assert (got.lat is not None) == (n > 6) == (got.lon is not None)  # both or none
        assert (got.speed_mps is not None) == (n > 7)
        assert (got.course_deg is not None) == (n > 8)


def test_a_negative_number_is_no_speed_a_course_or_an_hdop():
    assert gd._finite_non_negative("-0.05") is None
    assert gd._finite_non_negative("0.0") == 0.0


def test_the_checksum_rules():
    good = nmea("GNGGA,1").decode()  # ends with the CR LF a port delivers
    assert nmea_checksum_ok(good) is True and nmea_checksum_ok(good.strip()) is True
    assert nmea_checksum_ok("$GNGGA,1") is False  # no checksum at all
    assert nmea_checksum_ok("$GNGGA,1*ZZ") is False
    assert nmea_checksum_ok("$GNGGA,1*00") is False  # the wrong one
    assert nmea_checksum_ok("GNGGA,1*" + good.split("*")[1]) is False  # no dollar


# -------------------------------------------------------------------- bauds and candidate ports


@pytest.mark.parametrize(
    ("mode", "bauds"),
    [
        (GpsMode.F9P_USB, [9600, 115200, 38400]),
        (GpsMode.F9P_UART, [9600, 115200, 38400]),
        (GpsMode.LC29H_UART, [9600, 115200, 460800]),
        (GpsMode.NEO8M_UART, [9600, 38400]),
    ],
)
def test_each_receiver_family_is_probed_at_its_own_baud_rates(mode, bauds):
    assert GPSDriver({"mode": mode})._baud_candidates == bauds
    assert GPSDriver({"mode": mode, "baudrate": 57600})._baud_candidates[0] == 57600


def candidates(monkeypatch, **cfg):
    monkeypatch.setenv("SIM_MODE", "0")
    for var in ("GPS_DEVICE", "GPS_EXCLUDE_DEVICES", "ROBOHAT_SERIAL_PORT", "ROBOHAT_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gd, "_enumerate_serial_ports", lambda: ([], False))
    monkeypatch.setattr(gd.glob, "glob", lambda pattern: [])
    return GPSDriver(cfg)._candidates()[0]


def test_the_configured_device_comes_first_then_the_usual_ones(monkeypatch):
    usual = ["/dev/ttyAMA0", "/dev/serial0", "/dev/ttyS0", "/dev/ttyUSB0", "/dev/ttyUSB1"]
    wanted = candidates(monkeypatch, mode=GpsMode.F9P_USB, usb_device="/dev/ttyACM9")
    assert wanted == ["/dev/ttyACM9", *usual]
    wanted = candidates(monkeypatch, mode=GpsMode.LC29H_UART, uart_device="/dev/ttyAMA3")
    assert wanted == ["/dev/ttyAMA3", *usual]


def test_the_ports_pyserial_enumerates_carry_their_vendor_and_their_descriptor(monkeypatch):
    install_serial(monkeypatch, {}, found=[("/dev/ttyACM0", 0x2E8A, "CircuitPython CDC")])
    ports, complete = gd._enumerate_serial_ports()
    assert complete and [(p.device, p.vid) for p in ports] == [("/dev/ttyACM0", 0x2E8A)]
    assert ports[0].description.strip() == "CircuitPython CDC"
    assert gd.list_serial_ports() == ports


def test_an_enumeration_that_fails_is_reported_as_incomplete_with_what_it_found(monkeypatch):
    install_serial(monkeypatch, {}, found=[("/dev/ttyACM0", 0x1546, "u-blox")])
    import serial.tools.list_ports as lp

    one = lp.comports()[0]

    def comports():
        yield one
        raise OSError("udev went away")

    monkeypatch.setattr(lp, "comports", comports)
    ports, complete = gd._enumerate_serial_ports()
    assert not complete and [p.device for p in ports] == ["/dev/ttyACM0"]

    def nothing():
        raise OSError("no udev")

    monkeypatch.setattr(lp, "comports", nothing)
    assert gd._enumerate_serial_ports() == ([], False)
    assert isinstance(SerialPortInfo("/dev/x").device, str)


# -------------------------------------------------------------------- what a reading is made of


def made(speed_mps, *, late_s=0.0):
    clk = Clock()
    drv = GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clk.mono, wall_clock=clk.wall)
    gga = Gga(10.0, 1, 48.0, 11.0, 500.0, 10, 0.8, None)
    rmc = Rmc(10.0, True, "A", 48.0, 11.0, speed_mps, 90.0)
    ep = gd._Epoch(key=10.0, t_pos=clk.t, gga=gga, rmc=rmc, kinds={"GGA", "RMC"})
    st = gd._Stream(clk.t)
    clk.advance(late_s)
    return drv, st, ep, drv._reading_from(st, ep)


def test_the_course_is_a_heading_from_half_a_metre_per_second_on():
    assert made(0.5)[3].heading == 90.0
    assert made(0.49)[3].heading is None


def test_a_reading_remembers_its_epoch_and_how_late_it_was_made():
    drv, st, ep, reading = made(1.0, late_s=1.0)
    assert st.last_delivered is ep
    assert drv.lag_s == pytest.approx(1.0) == (NOON - NOON).total_seconds() + 1.0
    assert reading.rtk_status == "GPS_FIX"


def test_an_age_is_never_negative_and_never_a_made_up_one():
    now = NOON
    assert reading_age_s(SimpleNamespace(timestamp=now), now) == 0.0
    assert reading_age_s(SimpleNamespace(timestamp=now + timedelta(seconds=0.05)), now) is None
    assert reading_age_s(SimpleNamespace(timestamp=now - timedelta(seconds=2)), now) == 2.0


def test_a_driver_reports_the_sentences_it_saw_and_none_before(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    assert rig.driver.get_last_nmea() == {}
    rig.at(0.5, 0)
    rig.at(1.5, 1)
    assert rig.next() is not None
    assert {"GGA", "GST", "RMC"} <= set(rig.driver.get_last_nmea())


# ------------------------------------------------------------ the worker, gpsd and the lifecycle


@pytest.mark.asyncio
async def test_read_position_returns_as_soon_as_its_read_is_done_not_at_the_deadline(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    for k in range(4):
        rig.at(0.5 + k, k)
    rig.driver.READ_DEADLINE_S = 5.0
    started_at = time.monotonic()
    try:
        reading = await rig.driver.read_position()
        took = time.monotonic() - started_at
    finally:
        await rig.driver.stop()
    assert reading is not None and epoch_of(reading) == 1
    assert took < 2.5  # at the deadline it would have been 5 s


@pytest.mark.asyncio
async def test_stop_ends_the_running_state_and_start_begins_a_new_session(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    rig.driver._clock.accept(5000.0, rig.clk.t)  # the time line of the session that ends
    await rig.driver.stop()
    assert rig.driver.running is False and rig.driver._closing
    await rig.driver.start()
    assert rig.driver.running is True and rig.driver._closing is False
    assert rig.driver._clock.accept(10.0, rig.clk.t).reason == "first"  # not "stale"


def test_gpsd_is_asked_only_when_the_serial_read_gives_nothing_and_not_more_than_every_five_s(
    monkeypatch,
):
    rig = Rig(monkeypatch, mark=True)
    rig.at(0.5, 0)  # the receiver says one epoch, which the probe takes, and then nothing
    asked = []

    def gpsd(timeout_sec=0.5):
        asked.append((round(rig.clk.t - rig.t0, 1), timeout_sec))
        return None

    rig.driver._read_from_gpsd = gpsd
    for _ in range(3):
        assert rig.read() is None
    assert len(asked) == 1 and asked[0][1] == 1.0  # the long window of a session with no fix yet


def test_a_reading_gpsd_gives_is_the_reading_and_ends_the_first_window(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    rig.at(0.5, 0)
    sentinel = object()
    rig.driver._read_from_gpsd = lambda timeout_sec=0.5: sentinel
    assert rig.read() is sentinel
    assert rig.driver._first_read_done is True


# --------------------------------------------------------------------------------- freshness


def test_forgetting_one_key_or_all_of_them():
    def ages():
        a = ReceiptAges(lambda: 100.0)
        for key in ("a", "b"):
            assert a.taken_at(key, NOON - timedelta(seconds=1), NOON, now=100.0) == 99.0
        return a

    stepped_back = NOON - timedelta(seconds=5)  # judged afresh, the stamp is in the future
    remembered = ages()
    assert remembered.taken_at("a", NOON - timedelta(seconds=1), stepped_back, now=105.0) == 99.0
    one = ages()
    one.forget("a")
    assert one.taken_at("a", NOON - timedelta(seconds=1), stepped_back, now=105.0) is None
    assert one.taken_at("b", NOON - timedelta(seconds=1), stepped_back, now=105.0) == 99.0
    everything = ages()
    everything.forget()
    for key in ("a", "b"):
        assert (
            everything.taken_at(key, NOON - timedelta(seconds=1), stepped_back, now=105.0) is None
        )


def test_an_aware_stamp_in_another_zone_is_the_instant_it_names():
    plus_two = timezone(timedelta(hours=2))
    assert parse_stamp(datetime(2026, 10, 25, 14, 0, 0, tzinfo=plus_two)) == NOON


def test_the_injected_clock_is_used_when_no_time_is_given_and_a_future_stamp_is_unknown():
    ages = ReceiptAges(lambda: 100.0)
    assert ages.taken_at("k", NOON - timedelta(seconds=1), NOON) == 99.0
    assert ages.taken_at("f", NOON + timedelta(seconds=0.05), NOON, now=100.0) is None


# -------------------------------------------------------------------------------- the timezone


@pytest.mark.parametrize(
    ("lat", "lon", "usable"),
    [
        (0.0, 0.5, True),
        (0.5, 0.0, True),
        (0.0, 0.05, True),
        (0.05, 0.0, True),
        (0.0, 0.0, False),
        (1e-6, 0.0, False),
        (0.0, -1e-6, False),
        (float("nan"), 1.0, False),
        (None, 1.0, False),
    ],
)
def test_a_position_on_one_axis_alone_is_a_place_and_the_null_island_is_not(lat, lon, usable):
    got = tz._usable_coordinates(SimpleNamespace(latitude=lat, longitude=lon))
    assert (got is not None) == usable


def test_with_no_hub_manager_the_lookup_says_so(monkeypatch):
    from backend.src.services.websocket_hub import websocket_hub

    monkeypatch.setattr(websocket_hub, "_sensor_manager", None)
    assert tz._hub_position() == (False, None)


def test_the_gps_timezone_lookup_can_be_switched_off_and_then_touches_nothing(monkeypatch):
    def forbidden():
        raise AssertionError("the lookup was switched off")

    monkeypatch.setenv("LAWN_BERRY_DISABLE_GPS_TZ", "1")
    monkeypatch.setattr(tz, "_hub_position", forbidden)
    assert tz._default_gps_lookup() is None
