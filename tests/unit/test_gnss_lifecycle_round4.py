"""The driver's worker, windows, reopen delay, probe and stop (the surviving mutants of the third
review that guard a safety claim: nothing blocks the loop, nothing is closed under a read, a port
is only taken for a receiver when it speaks NMEA).
"""

from __future__ import annotations

import asyncio
import time

import pytest
from _gnss_fakes import Clock, Port, SerialException, epoch_lines, gga, gst, install_serial, nmea

from backend.src.drivers.sensors import gps_driver
from backend.src.drivers.sensors.gps_driver import (
    GPSDriver,
    _IoWorker,
    nmea_checksum_ok,
    parse_gga,
    parse_gst_sigma,
)
from backend.src.models.sensor_data import GpsMode


@pytest.fixture(autouse=True)
def hardware_path(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    for var in ("GPS_DEVICE", "GPS_EXCLUDE_DEVICES", "ROBOHAT_SERIAL_PORT", "ROBOHAT_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gps_driver.glob, "glob", lambda _pattern: [])
    monkeypatch.setattr(gps_driver, "_UART_DEVICES", ("/dev/ttyAMA0",))
    monkeypatch.setattr(gps_driver, "_USB_SERIAL_DEVICES", ())


def virtual(monkeypatch, port: Port, clock: Clock) -> GPSDriver:
    install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clock.mono, wall_clock=clock.wall)
    driver.initialized = True
    driver.running = True
    return driver


# --------------------------------------------------------------------------------------- NMEA text


def test_a_sentence_with_no_dollar_is_never_valid_even_if_its_checksum_would_be():
    body = "GNGGA,120005.00,4807.038,N,01131.000,E,1,08,0.8,545.4,M,46.9,M,,"
    check = 0
    for ch in body:
        check ^= ord(ch)
    assert nmea_checksum_ok(f"${body}*{check:02X}")
    assert not nmea_checksum_ok(f"X{body}*{check:02X}")  # one character in the dollar's place
    assert not nmea_checksum_ok(f"{body}*{check:02X}")
    assert not nmea_checksum_ok("")


def test_a_zero_sigma_component_is_a_value_not_a_missing_one():
    """sigma_lat 0.0 with sigma_lon 2.0 is an estimate of 2.0 m; treating the 0.0 as 'empty' fell
    through to the error ellipse (a different number)."""
    got = parse_gst_sigma("$GNGST,1,2.1,4.4,3.1,45.0,0.0,2.0,5.5*00")
    assert got == pytest.approx(2.0)


def test_the_southern_and_western_hemispheres_are_negative_in_a_whole_gga():
    got = parse_gga(gga(1, ns="S", ew="W").decode().strip())
    assert got.lat == pytest.approx(-48.1173, abs=1e-4)
    assert got.lon == pytest.approx(-11.5167, abs=1e-4)


# -------------------------------------------------------------------------------------- the worker


def test_a_job_that_raises_anything_at_all_is_delivered_and_the_worker_goes_on():
    """SystemExit and KeyboardInterrupt are BaseExceptions: one of them in a driver call used to
    be able to end the only thread that reads the receiver, silently."""
    worker = _IoWorker("test-io")
    try:
        for exc in (RuntimeError("x"), SystemExit(3), KeyboardInterrupt()):

            def boom(exc=exc):
                raise exc

            future = worker.submit(boom)
            assert type(future.exception(timeout=3.0)) is type(exc)
        assert worker.submit(lambda: 42).result(timeout=3.0) == 42  # still serving
        assert worker._thread.is_alive()
    finally:
        worker.close()


def test_a_worker_holds_nothing_between_jobs():
    class Owner:
        def work(self) -> int:
            return 1

    owner = Owner()
    worker = _IoWorker("test-io")
    try:
        assert worker.submit(owner.work).result(timeout=3.0) == 1
        import gc
        import weakref

        ref = weakref.ref(owner)
        del owner
        gc.collect()
        assert ref() is None, "the idle worker kept the object it last worked for alive"
    finally:
        worker.close()


# -------------------------------------------------------------------------------------------- stop


async def test_stop_does_not_close_the_port_under_a_read_that_is_still_on_it(monkeypatch):
    """Closing a tty from the loop thread while the worker is blocked in a read on it races the
    read; the worker closes the port itself when its read ends."""
    port = Port("blocking")
    serial = install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = GPSDriver({"mode": GpsMode.LC29H_UART})
    driver.PROBE_WINDOW_S = 5.0
    await driver.initialize()
    await driver.start()
    await driver.read_position()  # the worker is now inside a read on the port
    assert driver._busy() and serial.serials and port.entered >= 1
    await driver.stop()
    assert port.closed_by_driver == 0, "the port was closed under the blocked read"
    port.release.set()
    for _ in range(100):
        if port.closed_by_driver:
            break
        await asyncio.sleep(0.05)
    assert port.closed_by_driver == 1, "the read that ended after the stop left the port open"


async def test_stop_interrupts_a_read_that_is_waiting_for_the_next_epoch(monkeypatch):
    """A read waits up to its window for data; stop() must not have to wait that window out."""
    port = Port("stream")
    install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = GPSDriver({"mode": GpsMode.LC29H_UART})
    driver.PROBE_WINDOW_S = 0.1
    driver.FIRST_WINDOW_S = driver.WINDOW_S = 4.0  # a long window
    await driver.initialize()
    await driver.start()
    port.feed_epoch(1)
    port.feed_epoch(2)  # enough to open the port and begin a read that then waits for epoch 3
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not (driver._busy() and driver._serial):
        await driver.read_position()
        await asyncio.sleep(0.02)
    assert driver._serial is not None
    worker_thread = driver._worker._thread
    await asyncio.sleep(0.3)  # the worker is in the wait for new data
    started = time.monotonic()
    await driver.stop()
    worker_thread.join(3.0)
    assert not worker_thread.is_alive()
    assert time.monotonic() - started < 1.5, "stop() waited for the window to run out"
    assert (
        driver._serial is None and port.closed_by_driver == 1
    )  # the read closed it on its way out


# --------------------------------------------------------------------------------- windows, delays


def test_the_first_read_of_a_session_gets_a_longer_window_than_the_later_ones(monkeypatch):
    clk = Clock()
    port = Port("stream", clock=clk, mark=True)
    driver = virtual(monkeypatch, port, clk)
    t0 = clk.t
    port.schedule_epoch(t0 + 0.3, 0)  # the probe takes this one
    port.schedule_epoch(t0 + 1.6, 1)  # 1.3 s after the first read began: inside 1.5 s, not 0.75 s
    port.schedule_epoch(t0 + 3.2, 2)  # 1.6 s after the second read began: inside neither window
    first = driver._read_blocking()
    assert first is not None and first.longitude == pytest.approx(11.5167, abs=1e-3)
    started = clk.t
    assert driver._read_blocking() is None  # a later read gives up after its own, shorter window
    assert driver.WINDOW_S <= clk.t - started <= driver.WINDOW_S + 0.3
    assert driver.FIRST_WINDOW_S > driver.WINDOW_S


def test_a_read_that_found_nothing_waits_exactly_its_window(monkeypatch):
    clk = Clock()
    port = Port("stream", clock=clk)
    driver = virtual(monkeypatch, port, clk)
    t0 = clk.t
    port.schedule_epoch(t0 + 0.3, 0)
    port.schedule_epoch(t0 + 0.6, 1)
    assert driver._read_blocking() is not None
    started = clk.t
    assert driver._read_blocking() is None  # nothing else is coming
    assert driver.WINDOW_S <= clk.t - started <= driver.WINDOW_S + 0.3


class FlakySerial:
    """A ``serial.Serial`` whose read fails once, as a vanished USB device does."""

    def __init__(self, inner, fail_after: int) -> None:
        self._inner, self._left = inner, fail_after
        self.port, self.baudrate = inner.port, inner.baudrate

    @property
    def in_waiting(self):
        return self._inner.in_waiting

    def read(self, size=1):
        self._left -= 1
        if self._left < 0:
            raise SerialException(5, "input/output error")
        return self._inner.read(size)

    def close(self):
        self._inner.close()


def test_after_a_read_error_the_port_is_reopened_after_the_delay_not_at_once(monkeypatch):
    clk = Clock()
    port = Port("stream", clock=clk)
    serial = install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clk.mono, wall_clock=clk.wall)
    t0 = clk.t
    for i in range(30):
        port.schedule_epoch(t0 + 0.3 + i, i)
    assert driver._read_blocking() is not None
    opens = len(serial.opens)
    driver._serial = FlakySerial(driver._serial, fail_after=0)  # the next read fails
    driver._read_blocking()  # raises inside, is caught: the port is dropped
    assert driver._serial is None
    failed_at = clk.t
    assert driver._read_blocking() is None  # at once: not reopened yet
    assert len(serial.opens) == opens and clk.t == failed_at
    clk.advance(driver.REOPEN_DELAY_S + 0.01)
    driver._read_blocking()
    assert len(serial.opens) == opens + 1  # and once the delay has passed it is


# --------------------------------------------------------------------------------------- the probe


def probe(monkeypatch, payload: bytes) -> bool:
    clk = Clock()
    port = Port("stream", clock=clk)
    port.schedule(clk.t + 0.1, payload)
    install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = GPSDriver({"mode": GpsMode.LC29H_UART}, monotonic=clk.mono, wall_clock=clk.wall)
    driver.PROBE_WINDOW_S = 1.0
    return driver._open_port()


def test_a_port_that_speaks_nmea_is_a_receiver(monkeypatch):
    assert probe(monkeypatch, b"".join(epoch_lines(1))) is True


def test_a_sentence_with_a_bad_checksum_does_not_make_a_receiver(monkeypatch):
    """Line noise at the wrong baud rate can contain ``$G``; only a sentence that checks out
    counts."""
    bad = b"$GNGGA,120005.00,4807.038,N,01131.000,E,1,08,0.8,545.4,M,46.9,M,,*00\r\n"
    assert probe(monkeypatch, bad * 20) is False


def test_nmea_that_is_not_gnss_does_not_make_a_receiver(monkeypatch):
    """A compass or another instrument on a port the driver may open speaks NMEA too."""
    compass = nmea("HCHDG,98.3,0.0,E,12.6,W")
    assert probe(monkeypatch, compass * 20) is False


def test_a_receiver_is_recognised_whatever_precedes_its_first_sentence(monkeypatch):
    assert probe(monkeypatch, b"\x00\x01garbage\r\n\xb5\x62\x01\x07\x00" + gga(1)) is True


# ------------------------------------------------------------------------- GST without an estimate


def test_a_receiver_that_says_it_has_no_estimate_does_not_inherit_the_previous_one(monkeypatch):
    """GST with an all-zero estimate is the receiver's 'I do not know': the accuracy is unknown, it
    is not the previous epoch's figure."""
    clk = Clock()
    port = Port("stream", clock=clk)
    driver = virtual(monkeypatch, port, clk)
    t0 = clk.t
    port.schedule_epoch(t0 + 0.3, 0, fix=1, gst_sigma=(0.8, 0.8), mark=True)
    port.schedule_epoch(t0 + 1.3, 1, fix=1, gst_sigma=(0.8, 0.8), mark=True)
    port.schedule_epoch(t0 + 2.3, 2, fix=1, gst_sigma=(0.0, 0.0), mark=True)
    readings = []
    for _ in range(8):
        reading = driver._read_blocking()
        if reading is not None:
            readings.append(reading)
        clk.advance(0.2)
    assert readings[0].accuracy == pytest.approx(0.8 * 2**0.5)
    assert readings[-1].accuracy is None


def test_an_epoch_whose_gga_was_lost_does_not_hide_the_next_epoch(monkeypatch):
    """A GST alone says nothing about a position; the epoch after it is the reading."""
    clk = Clock()
    port = Port("stream", clock=clk)
    driver = virtual(monkeypatch, port, clk)
    t0 = clk.t
    port.schedule_epoch(t0 + 0.3, 0, mark=True)
    port.schedule(t0 + 1.3, gst(1))  # epoch 1: only its GST arrived
    port.schedule_epoch(t0 + 2.3, 2, mark=True)
    seen = []
    for _ in range(10):
        reading = driver._read_blocking()
        if reading is not None:
            seen.append(round((reading.longitude - 11.0) * 60 * 1000 - 31 * 1000))
        clk.advance(0.2)
    assert seen[-1] == 2 and 1 not in seen


# ------------------------------------------------------------------------------------------ health


async def test_the_age_of_the_last_read_is_measured_on_the_monotonic_clock(monkeypatch):
    clk = Clock()
    port = Port("stream", clock=clk, mark=True)
    driver = virtual(monkeypatch, port, clk)
    t0 = clk.t
    for i in range(3):
        port.schedule_epoch(t0 + 0.3 + i, i)
    for _ in range(3):
        await driver.read_position()
    assert (await driver.health_check())["last_read_age_s"] == pytest.approx(0.0, abs=0.01)
    clk.advance(30.0)
    assert (await driver.health_check())["last_read_age_s"] == pytest.approx(30.0, abs=0.01)
    await driver.stop()


def test_the_first_health_report_has_no_read_age():
    driver = GPSDriver({"mode": GpsMode.LC29H_UART})
    assert asyncio.run(driver.health_check())["last_read_age_s"] is None
