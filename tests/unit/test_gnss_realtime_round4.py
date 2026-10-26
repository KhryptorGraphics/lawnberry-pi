"""The GNSS driver against real time and real pyserial (audit GPS-01, 02, 03, 04, 15).

The virtual-clock tests (``test_gnss_driver_round4.py``) say exactly what the driver decides; these
are the ones that need a real tty or a real wait: a stream that never ends a line holds a real
thread, a lock held by a dropped driver is a real ``flock`` on a real pseudo-terminal, and a
receiver that is fed at 5 Hz and read at 4 Hz really backs up. They run the unmodified public API
(``read_position``, ``stop``, ``GPSSensorInterface``) so each can be run against the code it
replaced. Everything waits on a condition with a generous bound, never on a fixed sleep.
"""

from __future__ import annotations

import asyncio
import gc
import os
import threading
import time

import pytest
from _gnss_fakes import Port, epoch_lines, epoch_of, install_serial

from backend.src.drivers.sensors import gps_driver
from backend.src.drivers.sensors.gps_driver import GPSDriver
from backend.src.models.sensor_data import GpsMode
from backend.src.nav.gps_degradation import GPSDegradationConfig, GPSDegradationMonitor
from backend.src.services.sensor_manager import GPSSensorInterface, SensorCoordinator

needs_pty = pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs pseudo-terminals")


@pytest.fixture(autouse=True)
def hardware_path(monkeypatch):
    """The real-hardware path, on explicit devices only (no globbing of this machine's ports)."""
    monkeypatch.setenv("SIM_MODE", "0")
    for var in (
        "GPS_DEVICE",
        "GPS_EXCLUDE_DEVICES",
        "ROBOHAT_SERIAL_PORT",
        "ROBOHAT_PORT",
        "LAWN_ROBOHAT_PORT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gps_driver, "_UART_DEVICES", ())
    monkeypatch.setattr(gps_driver, "_USB_SERIAL_DEVICES", ())
    monkeypatch.setattr(gps_driver.glob, "glob", lambda _pattern: [])


def quick(driver: GPSDriver) -> GPSDriver:
    """Short windows: a silent fake port costs real time per baud rate."""
    driver.PROBE_WINDOW_S = 0.3
    driver.FIRST_WINDOW_S = 0.6
    driver.WINDOW_S = 0.4
    return driver


async def until(condition, timeout: float = 20.0, step: float = 0.05):
    """Poll ``condition()`` until it is truthy; its value, or ``None`` at the bound."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = condition()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return value
        await asyncio.sleep(step)
    return None


# -------------------------------------------------------------------------------------- a real pty


class Pty:
    """A pseudo-terminal: the master end is the receiver, the slave's name is the device."""

    def __init__(self) -> None:
        self.master, self._slave = os.openpty()
        os.set_blocking(self.master, False)
        self.name = os.ttyname(self._slave)

    def write(self, data: bytes) -> None:
        try:
            os.write(self.master, data)
        except OSError:  # the queue is full (nobody reads) or the other end is gone
            pass

    def close(self) -> None:
        for fd in (self.master, self._slave):
            try:
                os.close(fd)
            except OSError:
                pass


class Source(threading.Thread):
    """A receiver on a pty: epochs at ``hz``, or a flood of ``junk`` with no line end."""

    def __init__(self, pty: Pty, hz: float = 10.0, **epoch) -> None:
        super().__init__(daemon=True)
        self.pty, self.hz, self.epoch = pty, hz, epoch
        self.junk: bytes | None = None
        self.k = 0
        self._stop_event = threading.Event()
        self.start()

    def run(self) -> None:
        next_at = time.monotonic()
        while not self._stop_event.is_set():
            if self.junk is not None:
                self.pty.write(self.junk)
                time.sleep(0.02)
                continue
            self.pty.write(b"".join(epoch_lines(self.k, mark=True, **self.epoch)))
            self.k += 1
            next_at += 1.0 / self.hz
            time.sleep(max(0.0, next_at - time.monotonic()))

    def stop(self) -> None:
        self._stop_event.set()
        self.join(2.0)


def lc29h(device: str, **kw) -> GPSDriver:
    return quick(GPSDriver({"mode": GpsMode.LC29H_UART, "uart_device": device, **kw}))


def second_exclusive_open_works(device: str) -> bool:
    """Is the port's exclusive lock free? (A real flock: another driver can take the port.)"""
    import serial

    try:
        serial.Serial(device, 9600, timeout=0.1, exclusive=True).close()
    except serial.SerialException:
        return False
    return True


# ------------------------------------------------------------------------------------------ GPS-01


@needs_pty
@pytest.mark.parametrize(
    "junk",
    [b"VALUE=123.4\r", b"A" * 8, bytes(range(128, 256))],
    ids=["cr-only", "no-line-ends", "binary"],
)
async def test_a_stream_with_no_line_feeds_cannot_wedge_the_reader(monkeypatch, junk):
    """pyserial's readline() only returns on a line feed or after one SILENT timeout: a stream
    that keeps sending without a line feed held the worker for ever, ``stop()`` could not end it
    and the receiver stayed locked."""
    pty = Pty()
    source = Source(pty, hz=10.0)
    driver = lc29h(pty.name)
    monkeypatch.setenv("GPS_DEVICE", pty.name)
    try:
        await driver.initialize()
        await driver.start()
        assert await until(driver.read_position), "no first fix from a healthy receiver"
        thread = driver._worker._thread
        source.junk = junk  # the stream loses its line ends and never recovers
        for _ in range(4):  # the loop stays answerable, and keeps getting an answer
            started = time.monotonic()
            await driver.read_position()
            assert time.monotonic() - started < driver.READ_DEADLINE_S + 0.5
            await asyncio.sleep(0.1)
        await driver.stop()
        thread.join(5.0)
        assert not thread.is_alive(), "stop() cannot end a reader that is stuck in readline()"
        assert second_exclusive_open_works(pty.name), "the receiver is still locked"
    finally:
        source.stop()
        pty.write(b"\n")
        pty.close()


@needs_pty
async def test_the_receiver_behind_a_device_that_never_ends_a_line_is_still_found(monkeypatch):
    """Discovery probed a CR-only device first and never reached the real receiver behind it."""
    bad, good = Pty(), Pty()
    flood = Source(bad, hz=1.0)
    flood.junk = b"VALUE=123.4\r"
    receiver = Source(good, hz=10.0)
    monkeypatch.setenv("GPS_DEVICE", bad.name)
    driver = lc29h(good.name)  # the second candidate
    try:
        await driver.initialize()
        await driver.start()
        reading = await until(driver.read_position, timeout=30.0)
        assert reading is not None, "the real receiver was never found"
        assert driver._serial is not None and driver._serial.port == good.name
        await driver.stop()
    finally:
        flood.stop()
        receiver.stop()
        bad.write(b"\n")
        bad.close()
        good.close()


# ------------------------------------------------------------------------------------------ GPS-02


@needs_pty
async def test_a_receiver_that_is_read_slower_than_it_sends_never_falls_behind(monkeypatch):
    """5 Hz in, 4 Hz out: the old driver took one epoch per call from the front, so the backlog
    grew by one epoch a second (up to ~1.8 s) while every stamp said 'now'."""
    pty = Pty()
    source = Source(pty, hz=5.0)
    monkeypatch.setenv("GPS_DEVICE", pty.name)
    driver = lc29h(pty.name)
    lags: list[int] = []
    try:
        await driver.initialize()
        await driver.start()
        assert await until(driver.read_position)
        end = time.monotonic() + 6.0
        while time.monotonic() < end:
            reading = await driver.read_position()
            if reading is not None:
                lags.append(source.k - 1 - epoch_of(reading))
            await asyncio.sleep(0.25)
        assert len(lags) > 10
        assert max(lags[6:]) <= 2, f"the reading fell {max(lags)} epochs behind the receiver"
        await driver.stop()
    finally:
        source.stop()
        pty.close()


async def test_a_backlog_that_built_up_while_nobody_read_is_skipped_to_its_newest_epoch(
    monkeypatch,
):
    """The host stalls for seven seconds of a lean 1 Hz stream (1.5 kB, below the old driver's
    2 kB discard threshold): the reading must be the newest epoch, not the oldest."""
    port = Port("stream", mark=True)
    install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = quick(GPSDriver({"mode": GpsMode.LC29H_UART}))
    await driver.initialize()
    await driver.start()
    feeder_stop = threading.Event()

    def feed_until_first_fix() -> None:
        while not feeder_stop.is_set():
            port.feed_epoch()
            time.sleep(0.05)

    feeder = threading.Thread(target=feed_until_first_fix, daemon=True)
    feeder.start()
    try:
        assert await until(driver.read_position), "no first fix"
    finally:
        feeder_stop.set()
        feeder.join(2.0)
    await asyncio.sleep(0.5)  # let the worker settle on the idle port
    for _ in range(8):  # eight epochs arrive at once
        port.feed_epoch()
    newest = port.k - 1
    reading = None
    for _ in range(6):
        reading = await driver.read_position()
        if reading is not None and epoch_of(reading) == newest:
            break
        await asyncio.sleep(0.05)
    assert epoch_of(reading) == newest, f"read epoch {epoch_of(reading)}, the newest is {newest}"
    await driver.stop()


# ------------------------------------------------------------------------------------------ GPS-03


async def test_a_receiver_stuck_repeating_its_last_epoch_is_not_a_live_fix(monkeypatch):
    """The same epoch (same UTC) over and over used to build a new reading, stamped now, each
    time: the fix looked fresh for as long as the receiver stayed stuck."""
    port = Port("stream", mark=True)
    install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = quick(GPSDriver({"mode": GpsMode.LC29H_UART}))
    await driver.initialize()
    await driver.start()
    m = GPSDegradationMonitor(
        GPSDegradationConfig(max_accuracy_m=10.0, max_fix_age_s=0.6, resume_dwell_s=0.0)
    )
    port.feed_epoch(40)
    port.feed_epoch(41)
    try:
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            port.feed_epoch(42)  # the receiver repeats itself
            m.observe(await driver.read_position())
            await asyncio.sleep(0.1)
        a = m.assess()
        assert a.degraded and any("old" in r for r in a.reasons), a.reasons
        assert a.fix_age_s > 1.5
    finally:
        await driver.stop()


# ------------------------------------------------------------------------------------------ GPS-15


async def test_a_port_that_goes_silent_without_an_error_is_dropped_and_probed_again(monkeypatch):
    port = Port("stream", baud=9600, mark=True)
    serial = install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    driver = quick(GPSDriver({"mode": GpsMode.LC29H_UART}))
    driver.SILENCE_REPROBE_S = 0.8
    driver.REOPEN_DELAY_S = 0.2
    await driver.initialize()
    await driver.start()
    stop = threading.Event()

    def receiver() -> None:
        while not stop.is_set():
            port.feed_epoch()
            time.sleep(0.1)

    threading.Thread(target=receiver, daemon=True).start()
    try:
        assert await until(driver.read_position), "no first fix"
        assert serial.opens[-1][1] == 9600
        port.baud = 115200  # reconfigured: the open port now reads noise, with no exception

        async def reopened_at_the_new_rate():
            reading = await driver.read_position()
            return serial.opens[-1][1] == 115200 and reading is not None and reading

        newest = await until(reopened_at_the_new_rate, timeout=30.0)
        assert newest, "the port was never probed again"
        assert time.time() - newest.timestamp.timestamp() < 2.0  # and the reading is a live one
    finally:
        stop.set()
        await driver.stop()


# ------------------------------------------------------------------------------------------ GPS-04


@needs_pty
async def test_a_stopped_manager_interface_releases_the_receiver(monkeypatch):
    """A throwaway SensorManager (the timezone lookup, the health probe) left a ``gps-io`` thread
    and the receiver's exclusive lock behind: the long-lived driver could never open the port."""
    pty = Pty()
    source = Source(pty, hz=10.0)
    monkeypatch.setenv("GPS_DEVICE", pty.name)
    iface = GPSSensorInterface(GpsMode.LC29H_UART, SensorCoordinator())
    quick(iface._driver)
    try:
        assert await iface.initialize()
        assert await until(iface.read_gps), "no reading"
        thread = iface._driver._worker._thread
        assert not second_exclusive_open_works(pty.name)  # the driver does hold it
        await iface.shutdown()
        thread.join(5.0)
        assert not thread.is_alive()
        assert second_exclusive_open_works(pty.name)
        assert await iface.read_gps() is None  # and it stays stopped
    finally:
        source.stop()
        pty.close()


@needs_pty
async def test_a_driver_that_is_dropped_without_a_stop_releases_its_thread_and_the_lock(
    monkeypatch,
):
    pty = Pty()
    source = Source(pty, hz=10.0)
    monkeypatch.setenv("GPS_DEVICE", pty.name)
    driver = lc29h(pty.name)
    try:
        await driver.initialize()
        await driver.start()
        assert await until(driver.read_position)
        thread = driver._worker._thread
        assert not second_exclusive_open_works(pty.name)
        await asyncio.sleep(0.2)  # the worker is idle between jobs
        del driver
        gc.collect()
        thread.join(5.0)
        assert not thread.is_alive(), "the idle worker keeps its driver alive"
        assert second_exclusive_open_works(pty.name), "the receiver stayed locked"
    finally:
        source.stop()
        pty.close()
