"""The sensor manager's GNSS side (audit GPS-04, GPS-07 and the survivor s01).

GPS-04: ``SensorManager.shutdown()`` only stopped the ultrasonic array, so a throwaway manager
(the timezone lookup, the health probe) left a ``gps-io`` thread and the receiver's exclusive lock
behind and the long-lived driver could never open the receiver again.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import types
from datetime import UTC, datetime

import pytest
from _gnss_fakes import Port, install_serial

from backend.src.drivers.sensors import gps_driver
from backend.src.models import RtkFixType, SensorStatus
from backend.src.models.sensor_data import GpsMode, GpsReading
from backend.src.services import timezone_service
from backend.src.services.sensor_manager import (
    GPSSensorInterface,
    SensorCoordinator,
    SensorManager,
)


class RecordingDriver:
    """Stands in for ``GPSDriver``: what the interface asks of it, and what it answers."""

    def __init__(self, readings=(), stop_error: Exception | None = None) -> None:
        self.readings = list(readings)
        self.stop_error = stop_error
        self.events: list[str] = []

    async def initialize(self) -> None:
        self.events.append("initialize")

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    async def read_position(self):
        self.events.append("read")
        return self.readings.pop(0) if self.readings else None


def interface(driver) -> GPSSensorInterface:
    gps = GPSSensorInterface(GpsMode.LC29H_UART, SensorCoordinator())
    gps._driver = driver
    return gps


def manager(driver) -> SensorManager:
    m = SensorManager(ultrasonic_enabled=False, ultrasonic_config={}, ultrasonic_obstacle_cm=30.0)
    m.gps._driver = driver
    return m


# ------------------------------------------------------------------------------------------ GPS-04


async def test_shutting_the_interface_down_stops_the_driver_and_the_interface_goes_quiet():
    driver = RecordingDriver([GpsReading(latitude=40.0, longitude=-83.0)])
    gps = interface(driver)
    assert await gps.initialize() and gps.status == SensorStatus.ONLINE
    assert await gps.read_gps() is not None
    await gps.shutdown()
    assert driver.events[-1] == "stop" and gps.status == SensorStatus.OFFLINE
    assert await gps.read_gps() is None  # and it stays stopped
    assert driver.events.count("read") == 1


async def test_shutting_the_manager_down_stops_the_gnss_driver():
    driver = RecordingDriver()
    m = manager(driver)
    assert await m.initialize()
    await m.shutdown()
    assert "stop" in driver.events and m.initialized is False


async def test_a_gnss_driver_that_cannot_stop_does_not_keep_the_ultrasonic_array_running(caplog):
    m = manager(RecordingDriver(stop_error=RuntimeError("tty is gone")))
    stopped: list[bool] = []

    async def ultrasonic_shutdown() -> None:
        stopped.append(True)

    m.ultrasonic.shutdown = ultrasonic_shutdown  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        await m.shutdown()
    assert stopped == [True] and m.initialized is False
    assert any("gps" in r.getMessage() and "tty is gone" in r.getMessage() for r in caplog.records)


async def test_an_ultrasonic_array_that_cannot_stop_does_not_keep_the_receiver_locked(caplog):
    driver = RecordingDriver()
    m = manager(driver)

    async def ultrasonic_shutdown() -> None:
        raise RuntimeError("port busy")

    m.ultrasonic.shutdown = ultrasonic_shutdown  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        await m.shutdown()
    assert "stop" in driver.events
    assert any(
        "ultrasonic" in r.getMessage() and "port busy" in r.getMessage() for r in caplog.records
    )


async def test_a_throwaway_manager_on_a_real_driver_leaves_no_thread_and_no_port_behind(
    monkeypatch,
):
    """The whole chain with the real driver on a fake port: read, shut down, nothing left."""
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("GPS_DEVICE", raising=False)
    port = Port("nmea", mark=True)
    serial = install_serial(monkeypatch, {"/dev/ttyAMA0": port})
    monkeypatch.setattr(gps_driver.glob, "glob", lambda _pattern: [])
    gps = GPSSensorInterface(GpsMode.LC29H_UART, SensorCoordinator())
    gps._driver.PROBE_WINDOW_S = 0.1
    assert await gps.initialize()
    reading = None
    for _ in range(40):
        reading = await gps.read_gps()
        if reading is not None and reading.latitude is not None:
            break
        await asyncio.sleep(0.05)
    assert reading is not None and serial.serials
    thread = gps._driver._worker._thread
    await gps.shutdown()
    thread.join(3.0)
    assert not thread.is_alive()
    assert all(not s.is_open for s in serial.serials)  # the port was closed, the lock with it
    assert [t for t in threading.enumerate() if t.name == "gps-io" and t is thread] == []


# ----------------------------------------------------------------------------- the timezone lookup


@pytest.fixture
def lookup(monkeypatch):
    """``_default_gps_lookup`` with its caches cleared and the sensor manager under test control."""
    monkeypatch.delenv("LAWN_BERRY_DISABLE_GPS_TZ", raising=False)
    monkeypatch.setattr(timezone_service, "_coordinate_cache", None)
    return timezone_service._default_gps_lookup


def hub_with(monkeypatch, manager_or_none):
    from backend.src.services import websocket_hub as hub_module

    monkeypatch.setattr(hub_module.websocket_hub, "_sensor_manager", manager_or_none)


def test_the_timezone_lookup_uses_the_hubs_receiver_and_opens_no_second_driver(monkeypatch, lookup):
    """A second driver in this process would take the receiver's exclusive lock from the hub's."""
    gps = types.SimpleNamespace(last_reading=GpsReading(latitude=40.7, longitude=-74.0))
    hub_with(monkeypatch, types.SimpleNamespace(gps=gps))

    def never(*_a, **_k):
        raise AssertionError("the lookup built a second SensorManager")

    monkeypatch.setattr("backend.src.services.sensor_manager.SensorManager", never)
    assert lookup() == (40.7, -74.0)


@pytest.mark.parametrize(
    "reading",
    [
        None,
        GpsReading(latitude=None, longitude=None),
        GpsReading(latitude=0.0, longitude=0.0),
        GpsReading(latitude=float("nan"), longitude=float("nan")),
        GpsReading(latitude=float("inf"), longitude=1.0),
        GpsReading(rtk_status="NO_FIX"),
    ],
    ids=["none", "no-position", "null-island", "nan", "inf", "no-fix"],
)
def test_a_hub_receiver_with_no_position_gives_no_coordinates_and_still_opens_no_second_driver(
    monkeypatch, lookup, reading
):
    hub_with(monkeypatch, types.SimpleNamespace(gps=types.SimpleNamespace(last_reading=reading)))
    monkeypatch.setattr(
        "backend.src.services.sensor_manager.SensorManager",
        lambda *_a, **_k: pytest.fail("a second SensorManager was built"),
    )
    assert lookup() is None


def test_the_lookup_remembers_a_position_it_found(monkeypatch, lookup):
    gps = types.SimpleNamespace(last_reading=GpsReading(latitude=40.7, longitude=-74.0))
    hub_with(monkeypatch, types.SimpleNamespace(gps=gps))
    assert lookup() == (40.7, -74.0)
    gps.last_reading = None  # the receiver is gone, the lookup is cached
    assert lookup() == (40.7, -74.0)


def test_without_a_hub_manager_the_throwaway_is_shut_down_when_the_lookup_is_done(
    monkeypatch, lookup
):
    """Before the hub has built its manager the lookup may read the receiver itself, but what it
    opens is closed again."""
    hub_with(monkeypatch, None)
    events: list[str] = []

    class Throwaway:
        def __init__(self) -> None:
            self.gps = types.SimpleNamespace(read_gps=self.read)

        async def initialize(self) -> bool:
            events.append("initialize")
            return True

        async def read(self):
            events.append("read")
            return GpsReading(latitude=48.0, longitude=11.0)

        async def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr("backend.src.services.sensor_manager.SensorManager", Throwaway)
    assert lookup() == (48.0, 11.0)
    assert events == ["initialize", "read", "shutdown"]


def test_the_throwaway_is_shut_down_even_when_it_found_nothing(monkeypatch, lookup):
    hub_with(monkeypatch, None)
    events: list[str] = []

    class Throwaway:
        def __init__(self) -> None:
            self.gps = types.SimpleNamespace(read_gps=self.read)

        async def initialize(self) -> bool:
            return True

        async def read(self):
            return None

        async def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr("backend.src.services.sensor_manager.SensorManager", Throwaway)
    assert lookup() is None and events == ["shutdown"]


# --------------------------------------------------------------------------------------------- s01


async def test_a_receiver_with_nothing_new_keeps_its_last_reading_and_its_old_stamp():
    """The interface replays ``last_reading`` when the driver returns nothing: the same object,
    with the stamp it had, so every consumer keeps ageing it (nobody sees a fresh fix)."""
    first = GpsReading(latitude=40.0, longitude=-83.0, timestamp=datetime(2026, 10, 25, tzinfo=UTC))
    gps = interface(RecordingDriver([first]))
    await gps.initialize()
    assert await gps.read_gps() is first
    again = await gps.read_gps()  # the driver has nothing
    assert again is first and again.timestamp == datetime(2026, 10, 25, tzinfo=UTC)


async def test_a_receiver_that_never_had_a_reading_gives_none_not_a_made_up_one():
    gps = interface(RecordingDriver())
    await gps.initialize()
    assert await gps.read_gps() is None


async def test_a_driver_that_raises_is_an_error_and_its_last_fix_is_not_served():
    class Broken:
        async def read_position(self):
            raise OSError("GPS serial port vanished")

    gps = interface(Broken())
    gps.status = SensorStatus.ONLINE
    gps.last_reading = GpsReading(latitude=40.0, longitude=-83.0)
    assert await gps.read_gps() is None and gps.status == SensorStatus.ERROR


# ----------------------------------------------------------------------------- GPS-07, the display


@pytest.mark.parametrize(
    ("status", "satellites", "expected"),
    [
        ("RTK_FIXED", 0, RtkFixType.RTK_FIXED),
        ("RTK_FLOAT", 0, RtkFixType.RTK_FLOAT),
        ("rtk_fixed", None, RtkFixType.RTK_FIXED),
        ("DGPS", 3, RtkFixType.DGPS_FIX),
        ("GPS_FIX", 3, RtkFixType.GPS_FIX),  # a plain fix is one with any number of satellites
        ("GPS_FIX", None, RtkFixType.GPS_FIX),
        ("NO_FIX", 14, RtkFixType.NO_FIX),  # in view is not in use
        ("DEAD_RECKONING", 14, RtkFixType.NO_FIX),
        ("MANUAL", 14, RtkFixType.NO_FIX),
        ("SIMULATION", 14, RtkFixType.NO_FIX),
        ("SOMETHING-NEW", 14, RtkFixType.NO_FIX),
        (None, 6, RtkFixType.GPS_FIX),  # no status reported: the satellite count is all there is
        (None, 5, RtkFixType.NO_FIX),
        (None, None, RtkFixType.NO_FIX),
        ("", 8, RtkFixType.GPS_FIX),
    ],
)
def test_the_fix_type_the_operator_sees_is_the_status_the_receiver_reported(
    status, satellites, expected
):
    m = SensorManager(ultrasonic_enabled=False, ultrasonic_config={}, ultrasonic_obstacle_cm=30.0)
    reading = GpsReading(latitude=40.0, longitude=-83.0, rtk_status=status, satellites=satellites)
    assert m._map_rtk_fix_type(reading) == expected
