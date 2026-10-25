"""GNSS accuracy, staleness and the serial side, second review round (audit SAF-04, SAF-05,
SAF-06, SAF-S1, TOOL-11).

The driver is fed real NMEA through a fake ``serial`` module (no hardware): a plain GPS fix whose
HDOP is 0.8 and whose GST says 4.2 m used to read "0.8 m accurate" and pass the GNSS gate; a
receiver that never answers used to freeze the event loop; probing for the receiver opened the
RoboHAT's and the ultrasonic MCU's ports.
"""

from __future__ import annotations

import asyncio
import math
import time
import types
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from _gnss_fakes import Port, install_serial, nmea

from backend.src.drivers.sensors import gps_driver
from backend.src.drivers.sensors.gps_driver import (
    GPSDriver,
    SerialPortInfo,
    combine_accuracy,
    nmea_checksum_ok,
    parse_gst_sigma,
    preferred_gps_device,
)
from backend.src.models.sensor_data import GpsMode, GpsReading
from backend.src.nav.gps_degradation import GPSDegradationConfig, GPSDegradationMonitor

# ------------------------------------------------------------------------------ monitor

EPOCH = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)


class Clocks:
    def __init__(self) -> None:
        self.t = 500.0
        self.wall_offset = 0.0

    def mono(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return EPOCH + timedelta(seconds=self.t - 500.0 + self.wall_offset)

    def stamp(self, age: float = 0.0) -> datetime:
        return self.wall() - timedelta(seconds=age)


def fix(c, *, acc=0.05, status="RTK_FIXED", age=0.0):
    return GpsReading(
        latitude=40.0, longitude=-83.0, accuracy=acc, rtk_status=status, timestamp=c.stamp(age)
    )


def monitor(c):
    cfg = GPSDegradationConfig(max_accuracy_m=1.0, max_fix_age_s=2.0, resume_dwell_s=3.0)
    return GPSDegradationMonitor(cfg, clock=c.mono, wall_clock=c.wall)


def settle(m, c, seconds=3.4):
    for _ in range(round(seconds / 0.2) + 1):
        c.t += 0.2
        m.observe(fix(c))
    return m.assess()


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf, -1.0, 0.0, True, "0.5", None], ids=repr
)
def test_an_accuracy_that_is_not_a_positive_finite_number_is_unknown_never_healthy(bad):
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    c.t += 0.2
    reading = fix(c)
    object.__setattr__(reading, "accuracy", bad)  # NaN/sentinels a model validator would not stop
    m.observe(reading)
    a = m.assess()
    assert a.degraded and any("accuracy unknown" in r for r in a.reasons), a.reasons
    assert a.accuracy_m is None


@pytest.mark.parametrize("status", ["MANUAL", "SIMULATION"])
def test_a_receiver_in_manual_or_simulator_mode_is_not_a_position_fix(status):
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    c.t += 0.2
    m.observe(fix(c, status=status))
    assert m.assess().degraded


def test_a_backward_wall_clock_step_cannot_revive_a_replayed_fix():
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    replayed = fix(c)
    m.observe(replayed)  # seen once while the clock was right ...
    c.wall_offset = -600.0  # ... then chrony steps it back ten minutes; the receiver goes quiet
    for _ in range(15):
        c.t += 0.2
        m.observe(replayed)
    a = m.assess()
    assert a.degraded and any("old" in r for r in a.reasons), a.reasons
    assert a.fix_age_s == pytest.approx(3.0, abs=0.01)  # aged on the monotonic clock


def test_a_fix_stamped_in_the_future_is_unusable_and_stays_so_when_wall_time_catches_up():
    c = Clocks()
    m = monitor(c)
    future = fix(c, age=-600.0)
    for _ in range(25):
        c.t += 0.2
        m.observe(future)
    a = m.assess()
    assert a.degraded and any("future" in r for r in a.reasons) and a.fix_age_s is None
    c.wall_offset = 601.0  # the clock reaches the stamp: still not a fresh fix
    c.t += 0.2
    m.observe(future)
    assert m.assess().degraded


def test_a_fix_with_no_timestamp_is_unusable_not_fresh():
    c = Clocks()
    m = monitor(c)
    bare = SimpleNamespace(latitude=40.0, longitude=-83.0, accuracy=0.05, rtk_status="RTK_FIXED")
    for _ in range(25):
        c.t += 0.2
        m.observe(bare)
    a = m.assess()
    assert a.degraded and any("timestamp" in r for r in a.reasons)


# ------------------------------------------------------------------------------ NMEA helpers


@pytest.fixture
def hardware(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    for var in (
        "GPS_DEVICE",
        "GPS_EXCLUDE_DEVICES",
        "ROBOHAT_SERIAL_PORT",
        "ROBOHAT_PORT",
        "LAWN_ROBOHAT_PORT",
    ):
        monkeypatch.delenv(var, raising=False)
    drivers: list[GPSDriver] = []

    async def make(mode=GpsMode.LC29H_UART, **cfg):
        driver = GPSDriver({"mode": mode, **cfg})
        driver.PROBE_WINDOW_S = 0.1  # a silent fake port costs real time per baud rate
        drivers.append(driver)
        await driver.initialize()
        await driver.start()
        return driver

    yield make
    for driver in drivers:
        driver._closing = True


@pytest.mark.parametrize(
    ("body", "sigma"),
    [
        ("GNGST,1,2.1,4.4,3.1,45.0,3.0,2.9,5.5", math.hypot(3.0, 2.9)),  # sigma_lat / sigma_lon
        ("GNGST,1,2.1,4.4,3.1,45.0,,,5.5", math.hypot(4.4, 3.1)),  # the error ellipse
        ("GNGST,1,2.1,4.4,,45.0,,,5.5", 4.4 * math.sqrt(2.0)),  # semi-major alone: an upper bound
        ("GNGST,1,2.1,,,45.0,,,5.5", None),
        ("GNGST,1,0,0,0,0,0,0,0", None),  # all-zero is a receiver's "no estimate"
        ("GNGST,1,2.1,4.4,3.1,45.0,nan,2.9,5.5", math.hypot(4.4, 3.1)),
        ("GNGST,1,2.1,inf,3.1,45.0,-3.0,2.9,5.5", None),
        ("GNGST,1,2.1,4.4,3.1,45.0,-3.0,2.9,5.5", math.hypot(4.4, 3.1)),
    ],
)
def test_the_gst_sigma_is_a_finite_non_negative_error_estimate_or_nothing(body, sigma):
    got = parse_gst_sigma("$" + body + "*00")
    assert got == pytest.approx(sigma) if sigma is not None else got is None


def test_the_accuracy_is_the_larger_of_the_receivers_sigma_and_the_rtk_figure():
    assert combine_accuracy(None, "GPS_FIX") is None and combine_accuracy(None, None) is None
    assert combine_accuracy(None, "RTK_FIXED") == 0.03
    assert combine_accuracy(None, "RTK_FLOAT") == 0.20
    assert combine_accuracy(0.5, "RTK_FIXED") == 0.5  # a bad honest sigma is not ignored
    assert combine_accuracy(0.01, "RTK_FIXED") == 0.03  # nor is an optimistic one believed
    assert combine_accuracy(4.2, "GPS_FIX") == 4.2
    assert combine_accuracy(0.0001, None) == 0.005


def test_a_nmea_sentence_needs_its_checksum():
    good = nmea("GNGST,1,2.1,4.4,3.1,45.0,3.0,2.9,5.5").decode().strip()
    assert nmea_checksum_ok(good)
    assert not nmea_checksum_ok(good.replace("3.0,2.9", "0.9,2.9"))  # a corrupted digit
    assert not nmea_checksum_ok(good.split("*")[0]) and not nmea_checksum_ok("GNGST*00")


# ------------------------------------------------------------------------------ SAF-04


async def read_twice(driver):
    await driver.read_position()
    return await driver.read_position()


async def test_a_plain_fix_with_a_good_hdop_is_not_reported_as_accurate(monkeypatch, hardware):
    install_serial(
        monkeypatch, {"/dev/ttyAMA0": Port("nmea", fix=1, hdop="0.8", gst_sigma=(3.0, 2.9))}
    )
    reading = await read_twice(await hardware())
    assert reading.rtk_status == "GPS_FIX" and reading.hdop == 0.8
    assert reading.accuracy == pytest.approx(math.hypot(3.0, 2.9)) and reading.accuracy > 4.0
    m = GPSDegradationMonitor(GPSDegradationConfig(resume_dwell_s=0.0))
    m.observe(reading)
    a = m.assess()
    assert a.degraded and any("exceeds" in r for r in a.reasons), a.reasons


async def test_a_plain_fix_without_gst_has_no_accuracy_and_the_monitor_refuses_it(
    monkeypatch, hardware
):
    install_serial(monkeypatch, {"/dev/ttyAMA0": Port("nmea", fix=1, hdop="0.5", gst_sigma=None)})
    reading = await read_twice(await hardware())
    assert reading.hdop == 0.5 and reading.accuracy is None  # HDOP is not an accuracy
    m = GPSDegradationMonitor(GPSDegradationConfig(resume_dwell_s=0.0))
    m.observe(reading)
    assert m.assess().degraded and "accuracy unknown" in m.assess().reasons[0]


@pytest.mark.parametrize(
    ("fix_quality", "gst", "expected"),
    [
        (4, None, 0.03),  # RTK fixed without a GST keeps its typical figure
        (5, None, 0.20),
        (4, (0.4, 0.3), 0.5),  # a worse honest sigma wins
        (4, (0.005, 0.005), 0.03),  # a better one is not believed past the fix level
    ],
)
async def test_an_rtk_fix_keeps_its_typical_figure_unless_the_receiver_says_worse(
    monkeypatch, hardware, fix_quality, gst, expected
):
    install_serial(
        monkeypatch, {"/dev/ttyAMA0": Port("nmea", fix=fix_quality, hdop="0.5", gst_sigma=gst)}
    )
    assert (await read_twice(await hardware())).accuracy == pytest.approx(expected)


async def test_a_gst_with_a_bad_checksum_is_not_believed(monkeypatch, hardware):
    install_serial(
        monkeypatch,
        {"/dev/ttyAMA0": Port("nmea", fix=1, gst_sigma=(0.2, 0.2), bad_gst_checksum=True)},
    )
    assert (await read_twice(await hardware())).accuracy is None


# ------------------------------------------------------------------------------ SAF-06


async def test_a_receiver_that_never_answers_cannot_stall_the_event_loop(monkeypatch, hardware):
    hung = Port("blocking")
    serial = install_serial(monkeypatch, {"/dev/ttyAMA0": hung})
    driver = await hardware()
    gaps: list[float] = []
    stop = False

    async def ticker():
        last = time.monotonic()
        while not stop:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    started = time.monotonic()
    assert await driver.read_position() is None
    assert time.monotonic() - started < driver.READ_DEADLINE_S + 0.3
    entered, opened = hung.entered, len(serial.opens)
    started = time.monotonic()
    for _ in range(5):  # the worker is still blocked: nothing new is started, nothing waits
        assert await driver.read_position() is None
    assert time.monotonic() - started < 0.2
    assert hung.entered == entered and len(serial.opens) == opened
    stop = True
    await task
    assert max(gaps) < 0.15, f"event loop stalled for {max(gaps):.2f} s"
    hung.release.set()
    await driver.stop()


async def test_the_reading_a_slow_receiver_produces_late_is_adopted_by_the_next_call(
    monkeypatch, hardware
):
    slow = Port("blocking")
    install_serial(monkeypatch, {"/dev/ttyAMA0": slow})
    driver = await hardware()
    assert await driver.read_position() is None  # the worker is stuck in the probe
    slow.mode = "nmea"  # the receiver wakes up
    slow.release.set()
    for _ in range(40):
        await asyncio.sleep(0.05)
        if (reading := await driver.read_position()) is not None:
            break
    assert reading is not None and reading.latitude == pytest.approx(48.1173, rel=1e-4)
    await driver.stop()


async def test_a_failed_probe_is_not_repeated_on_every_tick(monkeypatch, hardware):
    serial = install_serial(monkeypatch, {"/dev/ttyAMA0": Port("silent")})
    driver = await hardware()
    await driver.read_position()
    await asyncio.sleep(0.5)  # the probe pass (a few silent ports) finishes on the worker
    opens = len(serial.opens)
    assert opens > 0
    for _ in range(5):
        assert await driver.read_position() is None
    assert len(serial.opens) == opens  # not probed again within AUTODETECT_RETRY_S
    await driver.stop()


# ------------------------------------------------------------------------------ SAF-S1

ROBOHAT = ("/dev/ttyACM0", 0x2E8A, "Robo HAT MM1 CircuitPython CDC data")
UBLOX = ("/dev/ttyACM1", 0x1546, "u-blox GNSS receiver")
ULTRASONIC_MCU = ("/dev/ttyACM2", 0x2E8A, "RP2040-Zero ultrasonic")


async def test_the_probe_never_opens_a_port_another_driver_owns(monkeypatch, hardware):
    ports = {
        "/dev/ttyACM0": Port("silent"),
        "/dev/ttyACM2": Port("silent"),
        "/dev/ttyACM5": Port("silent"),
        "/dev/ttyAMA0": Port("nmea", fix=1, gst_sigma=(3.0, 2.9)),
    }
    serial = install_serial(monkeypatch, ports, found=[ROBOHAT, UBLOX, ULTRASONIC_MCU])
    monkeypatch.setenv("ROBOHAT_SERIAL_PORT", "/dev/ttyACM5")  # named by configuration
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyACM0")  # what the hub used to guess
    driver = await hardware()
    reading = await read_twice(driver)
    opened = {port for port, _baud, _kw in serial.opens}
    assert reading is not None and "/dev/ttyAMA0" in opened
    assert not opened & {"/dev/ttyACM0", "/dev/ttyACM2", "/dev/ttyACM5"}, opened
    assert ports["/dev/ttyACM0"].entered == ports["/dev/ttyACM2"].entered == 0
    await driver.stop()


async def test_every_probe_is_exclusive_and_a_port_locked_elsewhere_is_skipped_unread(
    monkeypatch, hardware
):
    ports = {
        "/dev/ttyACM3": Port("locked"),
        "/dev/ttyAMA0": Port("nmea", fix=1, gst_sigma=(3.0, 2.9)),
    }
    serial = install_serial(monkeypatch, ports)
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyACM3")
    driver = await hardware()
    assert (await read_twice(driver)) is not None
    assert serial.opens and all(kw.get("exclusive") is True for _p, _b, kw in serial.opens)
    assert ports["/dev/ttyACM3"].entered == 0  # locked by its owner: never read
    await driver.stop()


def test_the_preferred_device_is_found_by_identity_not_by_acm_number(monkeypatch):
    monkeypatch.delenv("GPS_EXCLUDE_DEVICES", raising=False)
    for var in ("ROBOHAT_SERIAL_PORT", "ROBOHAT_PORT", "LAWN_ROBOHAT_PORT"):
        monkeypatch.delenv(var, raising=False)
    info = [SerialPortInfo(d, v, text) for d, v, text in (ROBOHAT, UBLOX, ULTRASONIC_MCU)]
    assert preferred_gps_device(GpsMode.F9P_USB, info) == "/dev/ttyACM1"
    assert preferred_gps_device(GpsMode.F9P_USB, info[:1] + info[2:]) is None
    monkeypatch.setattr(gps_driver.os.path, "exists", lambda p: p == "/dev/ttyAMA0")
    assert preferred_gps_device(GpsMode.LC29H_UART, []) == "/dev/ttyAMA0"
    monkeypatch.setenv("GPS_EXCLUDE_DEVICES", "/dev/ttyAMA0")
    assert preferred_gps_device(GpsMode.LC29H_UART, []) is None


async def test_the_robohat_service_opens_its_port_exclusively(monkeypatch):
    from backend.src.services import robohat_service

    seen: dict = {}

    class SerialException(Exception):
        pass

    def refuse(**kw):
        seen.update(kw)
        raise SerialException("stop here: only the open arguments matter")

    fake = types.SimpleNamespace(Serial=refuse, SerialException=SerialException)
    monkeypatch.setattr(robohat_service, "serial", fake)
    assert await robohat_service.RoboHATService("/dev/ttyACM0").initialize() is False
    assert seen["port"] == "/dev/ttyACM0" and seen["exclusive"] is True
