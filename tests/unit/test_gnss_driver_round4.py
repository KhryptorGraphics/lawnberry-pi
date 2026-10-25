"""The GNSS driver's read path, third review round (audit GPS-02, 03, 06 .. 09, 13, 14, 15).

Most tests run the real ``GPSDriver`` on a VIRTUAL clock against a byte-stream fake of the serial
port (``_gnss_fakes``): a read that would wait advances the clock to the next scheduled arrival,
so a test says exactly when each sentence reached the port and when the host looked, and nothing
sleeps. The ``read_position`` ones run the worker thread the way production does. The tests that
need real time and real pyserial (line framing, wedges, locks) are in
``test_gnss_realtime_round4.py``.

A probe of a fresh port consumes the epoch it finds the receiver in, so the first reading of a
session is the epoch AFTER the first one that arrives.
"""

from __future__ import annotations

import pytest
from _gnss_fakes import (
    Clock,
    Port,
    epoch_lines,
    epoch_of,
    gga,
    gst,
    install_serial,
    rmc,
    ubx,
)

from backend.src.drivers.sensors.gps_driver import GPSDriver
from backend.src.models.sensor_data import GpsMode
from backend.src.nav.gps_degradation import GPSDegradationConfig, GPSDegradationMonitor


class Rig:
    """One driver on one fake port, all on one virtual clock."""

    def __init__(self, monkeypatch, *, port_kwargs=None, **epoch) -> None:
        monkeypatch.setenv("SIM_MODE", "0")
        for var in (
            "GPS_DEVICE",
            "GPS_EXCLUDE_DEVICES",
            "ROBOHAT_SERIAL_PORT",
            "ROBOHAT_PORT",
            "LAWN_ROBOHAT_PORT",
        ):
            monkeypatch.delenv(var, raising=False)
        self.clk = Clock()
        self.t0 = self.clk.t
        self.port = Port("stream", clock=self.clk, **(port_kwargs or {}), **epoch)
        self.serial = install_serial(monkeypatch, {"/dev/ttyAMA0": self.port})
        self.driver = GPSDriver(
            {"mode": GpsMode.LC29H_UART}, monotonic=self.clk.mono, wall_clock=self.clk.wall
        )
        self.driver.initialized = True
        self.driver.running = True

    def at(self, seconds: float, k: int, **kw) -> None:
        """Epoch ``k`` reaches the port ``seconds`` after the start."""
        self.port.schedule_epoch(self.t0 + seconds, k, **kw)

    def read(self):
        """One worker-side read (what ``read_position`` runs on the worker thread)."""
        return self.driver._read_blocking()

    def next(self, limit: int = 12):
        """Poll as the telemetry loop does (every 0.2 s) until a read yields a reading."""
        for _ in range(limit):
            reading = self.read()
            if reading is not None:
                return reading
            self.clk.advance(0.2)
        return None

    def all(self, count: int):
        """The readings of ``count`` successive polls that yielded one."""
        return [self.next() for _ in range(count)]

    def aged(self, count: int):
        """``(reading, its age the moment it was returned)`` of ``count`` successive polls."""
        out = []
        for _ in range(count):
            reading = self.next()
            out.append((reading, None if reading is None else self.age(reading)))
        return out

    def age(self, reading) -> float:
        return (self.clk.wall() - reading.timestamp).total_seconds()

    def monitor(self) -> GPSDegradationMonitor:
        cfg = GPSDegradationConfig(max_accuracy_m=1.0, max_fix_age_s=2.0, resume_dwell_s=0.0)
        return GPSDegradationMonitor(cfg, clock=self.clk.mono, wall_clock=self.clk.wall)


def epochs_seen(rig: Rig, polls: int) -> list[int]:
    """The epoch numbers of the readings ``polls`` polls produce (``mark=True`` epochs)."""
    seen = []
    for _ in range(polls):
        reading = rig.next()
        if reading is None:
            break
        seen.append(epoch_of(reading))
    return seen


# ------------------------------------------------------------------------- GPS-02: backlog, dating


def test_the_newest_epoch_in_the_buffer_is_the_reading_not_the_oldest(monkeypatch):
    """The old driver took one epoch per call from the FRONT: after a stall it kept returning
    seven-second-old positions, each stamped 'now'."""
    rig = Rig(monkeypatch, mark=True)
    for k in range(12):
        rig.at(0.5 + k, k)  # a 1 Hz receiver
    assert rig.next() is not None  # opens the port, takes the first epoch
    rig.clk.t = rig.t0 + 9.5  # the host did something else for eight seconds: epochs 2..9 queued
    reading = rig.read()
    assert epoch_of(reading) == 9
    assert rig.age(reading) == pytest.approx(0.0, abs=0.05)  # it arrived this instant


def test_a_reading_read_long_after_it_arrived_says_how_old_it_is(monkeypatch):
    """A stalled host and then a receiver that stopped: the last epoch is old, and says so."""
    rig = Rig(monkeypatch, mark=True)
    for k in range(11):
        rig.at(0.5 + k, k)  # the last epoch (10) arrives at 10.5 s and the receiver stops
    for _ in range(7):
        assert rig.next() is not None  # the host keeps up with the first epochs
    rig.clk.t = rig.t0 + 16.5  # then it stalls until 16.5 s
    reading = rig.read()
    assert epoch_of(reading) == 10
    assert rig.age(reading) == pytest.approx(6.0, abs=0.05)  # not 0: it was sent 6 s ago
    m = rig.monitor()
    m.observe(reading)
    a = m.assess()
    assert a.degraded and any("old" in r for r in a.reasons), a.reasons
    assert a.fix_age_s == pytest.approx(6.0, abs=0.05)


def test_a_reading_that_is_read_as_it_arrives_is_dated_when_it_arrived(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    for k in range(6):
        rig.at(0.5 + k, k)
    for _ in range(5):
        reading = rig.next()
        assert reading is not None and rig.age(reading) == pytest.approx(0.0, abs=0.05)


def test_the_reading_is_dated_by_its_position_sentence_not_by_the_end_of_the_epoch(monkeypatch):
    """The GGA arrives, the GST 0.4 s later (a receiver that spreads its epoch). The position is
    0.4 s old when the epoch is complete; the reading says 0.4, not 0."""
    rig = Rig(monkeypatch, mark=True)
    for k in range(8):
        rig.port.schedule(rig.t0 + 0.5 + k, gga(k, lon=f"01131.{k:03d}"))
        rig.port.schedule(rig.t0 + 0.5 + k + 0.1, rmc(k, lon=f"01131.{k:03d}"))
        rig.port.schedule(rig.t0 + 0.5 + k + 0.4, gst(k))
    ages = []
    for k in range(7):
        rig.clk.t = max(rig.clk.t, rig.t0 + 0.49 + k)  # the host is waiting when the GGA arrives
        reading = rig.read()
        if reading is not None and epoch_of(reading) == k:
            ages.append(rig.age(reading))
    assert len(ages) >= 5
    assert ages[-1] == pytest.approx(0.4, abs=0.02)


def test_the_reading_after_a_stall_is_complete_not_the_half_of_an_epoch(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    for k in range(8):
        rig.at(0.5 + k, k)
    assert rig.next() is not None
    rig.clk.t = rig.t0 + 6.5
    reading = rig.read()
    assert epoch_of(reading) == 6
    assert reading.accuracy is not None and reading.speed is not None  # its GST and its RMC


# ---------------------------------------------------------------------------------- GPS-03: replay


async def test_a_frozen_stream_is_never_a_new_reading_and_the_fix_ages_out(monkeypatch):
    """The receiver repeats its last burst for ever. Every repeat used to build a new reading
    stamped 'now': the monitor saw a fix of age 0 for as long as the receiver stayed stuck."""
    rig = Rig(monkeypatch, mark=True)
    for k in range(40):
        rig.at(0.5 + k, 5)  # the SAME epoch (UTC 12:00:05) every second
    rig.at(0.4, 4)  # the receiver was alive just before it froze
    driver = rig.driver
    m = rig.monitor()
    first = None
    for _ in range(4):
        first = first or await driver.read_position()
    assert first is not None
    readings = []
    for _ in range(25):
        reading = await driver.read_position()
        m.observe(reading)
        readings.append(reading)
        rig.clk.advance(0.2)
    assert all(r is first for r in readings)  # nothing new was built
    assert driver.epochs_refused >= 20  # and every repeat was counted
    a = m.assess()
    assert a.degraded and any("old" in r for r in a.reasons), a.reasons
    assert a.fix_age_s > 10.0
    await driver.stop()


async def test_a_frozen_stream_is_reported_not_just_silent(monkeypatch, caplog):
    rig = Rig(monkeypatch)
    rig.at(0.4, 4)
    for k in range(20):
        rig.at(0.5 + k, 5)
    for _ in range(30):
        await rig.driver.read_position()
        rig.clk.advance(0.2)
    assert any("GPS frozen" in r.getMessage() for r in caplog.records)
    await rig.driver.stop()


def test_an_older_epoch_after_newer_ones_is_refused_until_time_has_passed_it(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    order = [100, 101, 102, 103, 40, 41, 100, 104, 105]
    for i, k in enumerate(order):
        rig.at(0.5 + i, k)
    assert epochs_seen(rig, 5) == [101, 102, 103, 104, 105]  # 40, 41 and the second 100 skipped
    assert rig.driver.epochs_refused == 3


def test_the_midnight_wrap_is_time_going_on(monkeypatch):
    rig = Rig(monkeypatch)
    for i, utc in enumerate(["235958.00", "235959.00", "000000.00", "000001.00", "000002.00"]):
        rig.port.schedule(rig.t0 + 0.5 + i, b"".join(epoch_lines(i, utc=utc)))
    assert all(r is not None for r in rig.all(4))
    assert rig.driver.epochs_refused == 0


def test_one_wild_epoch_cannot_wedge_the_reader(monkeypatch):
    """A single epoch with a time hours ahead (a glitch, or a hostile byte) must not become the
    reference: everything after it would be 'older' and refused for the rest of the day."""
    rig = Rig(monkeypatch, mark=True)
    plan = [100, 101, 102, "wild", 103, 104, 105]
    for i, k in enumerate(plan):
        if k == "wild":
            rig.port.schedule(rig.t0 + 0.5 + i, b"".join(epoch_lines(7, utc="235900.00")))
        else:
            rig.at(0.5 + i, k)
    assert epochs_seen(rig, 5) == [101, 102, 103, 104, 105]
    assert rig.driver.epochs_refused == 1


def test_a_receiver_whose_clock_really_jumps_is_followed_after_one_epoch(monkeypatch):
    rig = Rig(monkeypatch)
    plan = [(100, None), (101, None), (102, None), (7, "180000.00"), (8, "180001.00"), (9, None)]
    for i, (k, utc) in enumerate(plan):
        rig.port.schedule(rig.t0 + 0.5 + i, b"".join(epoch_lines(k, mark=True, utc=utc)))
    seen = epochs_seen(rig, 3)
    assert seen[:2] == [101, 102]
    assert seen[2] == 8  # the first epoch of the new time line was held back, the second taken
    assert rig.driver.epochs_refused == 1


def test_a_reopened_port_keeps_the_time_line_of_the_receiver(monkeypatch):
    """A USB receiver that re-enumerates keeps its clock, and a frozen one must not be taken as
    fresh again by the reopen."""
    rig = Rig(monkeypatch, mark=True)
    for i, k in enumerate([100, 101, 102, 103, 101, 102, 104, 105]):
        rig.at(0.5 + i, k)
    assert epochs_seen(rig, 2) == [101, 102]
    rig.driver._drop_port()  # what a read error does
    seen = epochs_seen(rig, 3)
    assert 101 not in seen and seen[:2] == [103, 104] or seen[:1] == [104]


def test_a_receiver_that_was_silent_for_a_long_time_may_start_a_new_time_line(monkeypatch):
    """After ``SILENCE_REPROBE_S`` without a sentence the port is probed again and the receiver
    may have restarted: its clock is its own."""
    rig = Rig(monkeypatch, mark=True)
    for i, k in enumerate([100, 101, 102]):
        rig.at(0.5 + i, k)
    assert epochs_seen(rig, 2) == [101, 102]
    for i, k in enumerate([10, 11, 12, 13]):  # a restarted receiver, 12:00:10 and on
        rig.port.schedule(rig.clk.t + 8.0 + i, b"".join(epoch_lines(k, mark=True)))
    seen = epochs_seen(rig, 2)
    assert seen == [11, 12] or seen == [12, 13]


# ----------------------------------------------------------------- GPS-06 .. GPS-09: what a fix is


def one(monkeypatch, *, k=5, **epoch):
    """A rig and the third reading of a fresh receiver, taken the moment it arrives."""
    rig = Rig(monkeypatch, **epoch)
    for i in range(4):
        rig.at(0.5 + i, k + i)
    reading = None
    for _ in range(3):
        reading = rig.next()
    return rig, reading


def test_an_rtk_fix_with_no_gst_and_no_age_of_corrections_has_no_accuracy(monkeypatch):
    """One digit of one sentence said 'RTK fixed' and the monitor believed 3 cm. A real RTK fix
    reports the age of its corrections (GGA field 13); without it nothing backs the digit."""
    rig, reading = one(monkeypatch, fix=4, gst_sigma=None, age="")
    assert reading.rtk_status == "RTK_FIXED" and reading.accuracy is None
    m = rig.monitor()
    m.observe(reading)
    assert m.assess().degraded and "accuracy unknown" in m.assess().reasons[0]


@pytest.mark.parametrize("age", ["999.0", "30.5", "nan"])
def test_an_rtk_fix_with_stale_corrections_keeps_no_typical_accuracy(monkeypatch, age):
    _rig, reading = one(monkeypatch, fix=4, gst_sigma=None, age=age)
    assert reading.accuracy is None


@pytest.mark.parametrize(("fix", "expected"), [(4, 0.03), (5, 0.20)])
def test_an_rtk_fix_with_recent_corrections_keeps_its_typical_accuracy(monkeypatch, fix, expected):
    _rig, reading = one(monkeypatch, fix=fix, gst_sigma=None, age="2.0")
    assert reading.accuracy == pytest.approx(expected)


def test_a_gst_still_counts_for_an_rtk_fix_without_an_age_of_corrections(monkeypatch):
    _rig, reading = one(monkeypatch, fix=4, gst_sigma=(0.02, 0.02), age="")
    assert reading.accuracy == pytest.approx(0.03)  # the receiver's 2.8 cm, floored at RTK fixed


def test_a_plain_fix_is_never_more_accurate_than_half_a_metre(monkeypatch):
    _rig, reading = one(monkeypatch, fix=1, gst_sigma=(1e-9, 1e-9))
    assert reading.accuracy == pytest.approx(0.5)
    _rig, reading = one(monkeypatch, fix=2, gst_sigma=(1e-9, 1e-9))
    assert reading.accuracy == pytest.approx(0.3)


@pytest.mark.parametrize("quality", [3, 9, 44, "x", ""])
def test_a_fix_quality_outside_the_table_is_no_fix_and_carries_no_position(monkeypatch, quality):
    """Quality 3, 9, an empty or garbled digit: ``rtk_status`` used to be ``None``, which the
    monitor passed, and the position was published."""
    rig, reading = one(monkeypatch, fix=quality, gst_sigma=(0.02, 0.02))
    assert reading.rtk_status == "NO_FIX"
    assert reading.latitude is None and reading.longitude is None and reading.accuracy is None
    m = rig.monitor()
    m.observe(reading)
    a = m.assess()
    assert a.degraded and any("NO_FIX" in r for r in a.reasons), a.reasons


@pytest.mark.parametrize(
    ("quality", "status"),
    [(0, "NO_FIX"), (6, "DEAD_RECKONING"), (7, "MANUAL"), (8, "SIMULATION")],
)
def test_a_receiver_that_says_it_has_no_fix_is_believed_at_once(monkeypatch, quality, status):
    """Quality 0 with a last-known position (SiRF / MTK style) was published as a normal reading
    and the navigation service steered by it whatever its status."""
    rig, reading = one(monkeypatch, fix=quality, gst_sigma=(0.02, 0.02))
    assert reading.rtk_status == status and reading.latitude is None
    assert reading.satellites == 8  # what the receiver does say is still reported
    m = rig.monitor()
    m.observe(reading)
    assert m.assess().degraded


def test_an_rmc_with_status_void_and_a_last_known_position_is_not_a_fix(monkeypatch):
    rig = Rig(monkeypatch)
    for i in range(4):
        rig.port.schedule(rig.t0 + 0.5 + i, gst(5 + i) + rmc(5 + i, status="V", mode="N"))
    got = rig.next()
    assert got.rtk_status == "NO_FIX" and got.latitude is None and got.longitude is None
    m = rig.monitor()
    m.observe(got)
    assert m.assess().degraded


def test_an_epoch_with_an_rmc_but_no_gga_is_still_a_reading(monkeypatch):
    """A GGA lost to a checksum error: the RMC's position is used, judged by its own status."""
    rig = Rig(monkeypatch)
    for i in range(4):
        rig.port.schedule(
            rig.t0 + 0.5 + i, gst(5 + i, (0.4, 0.3)) + rmc(5 + i, mode="D", speed="4.0")
        )
    got = rig.next()
    assert got.rtk_status == "DGPS" and got.latitude == pytest.approx(48.1173)
    assert got.accuracy == pytest.approx(0.5) and got.satellites is None and got.hdop is None


def test_an_rmc_vetoes_a_gga_that_claims_an_rtk_fix(monkeypatch):
    _rig, reading = one(monkeypatch, fix=4, mode="S", gst_sigma=(0.02, 0.02))
    assert reading.rtk_status == "SIMULATION" and reading.latitude is None


@pytest.mark.parametrize(
    "position",
    [
        {"lat": "9959.999"},
        {"lat": "4860.000"},
        {"lon": "19959.999"},
        {"ew": "N"},
        {"lat": "", "lon": ""},
    ],
)
def test_a_position_that_is_not_a_position_is_never_published(monkeypatch, position):
    rig, reading = one(monkeypatch, fix=1, gst_sigma=(0.6, 0.6), **position)
    assert reading.rtk_status == "NO_FIX" and reading.latitude is None
    m = rig.monitor()
    m.observe(reading)
    assert m.assess().degraded


def test_a_bad_altitude_alone_is_not_a_reason_to_lose_the_fix(monkeypatch):
    _rig, reading = one(monkeypatch, fix=1, gst_sigma=(0.6, 0.6), alt="nan")
    assert reading.rtk_status == "GPS_FIX" and reading.altitude is None
    assert reading.latitude == pytest.approx(48.1173)


def test_a_sigma_from_the_previous_epoch_is_borrowed_only_while_it_still_describes_the_fix(
    monkeypatch,
):
    """A GST that is late or lost used to be replaced silently by the previous epoch's figure
    whatever the fix had become in between."""
    rig = Rig(monkeypatch, mark=True)
    rtk = {"fix": 4, "gst_sigma": (0.01, 0.01)}
    rig.at(0.5, 0, **rtk)
    rig.at(1.5, 1, **rtk)
    rig.at(2.5, 2, **rtk)
    rig.at(3.5, 3, fix=4, gst_sigma=None)  # its GST did not come, the fix is as good as before
    rig.at(4.5, 4, **rtk)
    rig.at(5.5, 5, fix=1, gst_sigma=None)  # its GST did not come and the fix got WORSE
    rig.at(6.5, 6, fix=1, gst_sigma=(1.0, 1.0))
    rig.at(9.5, 9, fix=1, gst_sigma=None)  # and the last GST is three epochs old
    got = {}
    for reading in (rig.next() for _ in range(7)):
        if reading is not None:
            got[epoch_of(reading)] = reading
    assert got[3].accuracy == pytest.approx(0.03)  # max(its borrowed 1.4 cm, the RTK floor)
    assert got[5].accuracy is None  # a plain fix after an RTK one: not the 1.4 cm
    assert got[9].accuracy is None


def test_a_gst_that_comes_first_belongs_to_its_own_epoch_by_its_utc(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    rig.at(0.5, 0, fix=1, gst_sigma=(0.8, 0.8))
    rig.at(1.5, 1, fix=1, gst_sigma=(0.8, 0.8))
    # epoch 2: its GST first, then its GGA and RMC (the order is the receiver's)
    rig.port.schedule(rig.t0 + 2.5, gst(2, (2.0, 2.0)) + gga(2, lon="01131.002") + rmc(2))
    rig.at(3.5, 3, fix=1, gst_sigma=(0.8, 0.8))
    got = {epoch_of(r): r for r in rig.all(3)}
    assert got[2].accuracy == pytest.approx(2.0 * 2**0.5)


# ----------------------------------------------------------------------- GPS-14: speed and heading


def test_speed_and_course_come_from_the_rmc_that_follows_the_gga_and_gst(monkeypatch):
    """The read used to stop at GGA + GST, before the RMC that follows them in the usual order:
    speed was None nearly always and the course was parsed and dropped."""
    rig = Rig(monkeypatch, mark=True, speed="10.0", course="271.5")
    for k in range(5):
        rig.at(0.5 + k, k)
    reading = rig.all(3)[-1]
    assert reading.speed == pytest.approx(10.0 * 1852.0 / 3600.0)
    assert reading.heading == pytest.approx(271.5)


def test_the_course_of_a_machine_that_is_not_moving_is_not_a_heading(monkeypatch):
    rig = Rig(monkeypatch, mark=True, speed="0.1", course="271.5")
    for k in range(4):
        rig.at(0.5 + k, k)
    reading = rig.all(2)[-1]
    assert reading.speed == pytest.approx(0.05144, rel=1e-3) and reading.heading is None


def test_a_receiver_that_sends_no_rmc_has_no_speed(monkeypatch):
    _rig, reading = one(monkeypatch, with_rmc=False)
    assert reading.speed is None and reading.heading is None and reading.latitude is not None


def test_a_late_rmc_is_learned_and_waited_for(monkeypatch):
    """The first read of a session goes without the RMC that follows 0.3 s after the GGA; the
    sentence is learned and every later read waits for it."""
    rig = Rig(monkeypatch, mark=True)
    for k in range(8):
        rig.port.schedule(rig.t0 + 0.5 + k, gga(k, lon=f"01131.{k:03d}") + gst(k))
        rig.port.schedule(rig.t0 + 0.8 + k, rmc(k, lon=f"01131.{k:03d}", speed="4.0"))
    readings = rig.all(5)
    assert readings[-1].speed == pytest.approx(4.0 * 1852.0 / 3600.0)


def test_a_sentence_kind_that_stopped_is_not_waited_for_after_five_seconds(monkeypatch):
    """While a kind is expected an epoch waits for it (up to the window and a grace): a GST that
    stopped makes every reading late for five seconds, and nothing after."""
    rig = Rig(monkeypatch, mark=True)
    for k in range(20):
        rig.at(0.5 + k, k, gst_sigma=(0.4, 0.4) if k < 4 else None)
    ages = [age for _reading, age in rig.aged(15)]
    assert max(ages[:9]) > 0.4  # waited for the missing GST
    assert all(a == pytest.approx(0.0, abs=0.25) for a in ages[-3:])  # no longer


# --------------------------------------------------------------------------- GPS-13: junk in front


def test_a_binary_frame_in_front_of_every_gga_costs_no_fix(monkeypatch):
    """A UBX frame without a line end glued the next sentence into one line that did not start
    with '$G': an F9P with UBX output enabled next to NMEA lost most of its fixes."""
    rig = Rig(monkeypatch, mark=True)
    frame = ubx(0x01, 0x07, bytes(range(92)))
    for k in range(10):
        rig.port.schedule(rig.t0 + 0.5 + k, frame + b"".join(epoch_lines(k, mark=True)))
    seen = epochs_seen(rig, 9)
    assert seen == list(range(1, 10))  # every epoch after the one the probe consumed


# --------------------------------------------------------------------------- GPS-15: a silent port


def test_a_port_that_goes_silent_without_an_error_is_dropped_and_probed_again(monkeypatch):
    """The receiver was reconfigured to another baud rate while the port stayed open: line noise,
    no exception. The driver used to read it for ever while it held the receiver's lock."""
    rig = Rig(monkeypatch, port_kwargs={"baud": 9600}, mark=True)
    for k in range(60):
        rig.at(0.5 + k, k)
    assert rig.all(3)[-1] is not None
    assert rig.serial.opens[-1][1] == 9600
    rig.port.baud = 115200  # the receiver is now at 115200: the open port reads noise
    reading = None
    for _ in range(200):
        reading = rig.next(limit=1)
        if reading is not None:
            break
    assert rig.serial.opens[-1][1] == 115200, "the port was never probed again"
    assert reading is not None and rig.age(reading) < 2.0
    assert rig.port.closed_by_driver >= 1  # the stale open was closed, not leaked


def test_a_port_that_is_merely_quiet_between_epochs_is_not_dropped(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    for k in range(6):
        rig.at(0.5 + 4.0 * k, k)  # one epoch every 4 s
    for _ in range(4):
        assert rig.next(limit=40) is not None
    assert len(rig.serial.opens) == 1


# ------------------------------------------------------------------------------- the health report


async def test_the_health_report_says_how_the_stream_is_doing(monkeypatch):
    rig = Rig(monkeypatch, mark=True)
    for k in range(5):
        rig.at(0.5 + k, k)
    rig.all(3)
    health = await rig.driver.health_check()
    assert health["device"] == "/dev/ttyAMA0" and health["lag_s"] == pytest.approx(0.0, abs=0.05)
    assert health["epochs_refused"] == 0
    assert health["last_sentence_age_s"] == pytest.approx(0.0, abs=0.05)
