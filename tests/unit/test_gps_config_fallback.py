"""GPS resilience fixes.

1. The telemetry hub must resolve the RTK GPS mode from config even when
   app.state doesn't carry hardware_config — otherwise it silently falls back
   to the non-RTK NEO8M default (wrong serial baud -> no GPS fix -> blank GPS).
2. The GPS driver must surface hardware failures (permission/device/no-NMEA)
   via a rate-limited log instead of swallowing them.
"""

import logging

from backend.src.drivers.sensors.gps_driver import GPSDriver
from backend.src.models.sensor_data import GpsMode
from backend.src.services.websocket_hub import WebSocketHub


async def test_hub_resolves_rtk_mode_from_config_without_app_state() -> None:
    hub = WebSocketHub()  # never bind_app_state() -> hardware_config absent
    sm = await hub._ensure_sensor_manager()
    # config/hardware.yaml declares LC29H-DA -> RTK LC29H_UART, not NEO8M default.
    assert sm.gps.gps_mode == GpsMode.LC29H_UART


def test_gps_driver_diag_is_rate_limited(caplog) -> None:
    d = GPSDriver({"mode": GpsMode.LC29H_UART})
    with caplog.at_level(logging.WARNING):
        d._diag("serial-permission", "first")
        d._diag("serial-permission", "second (throttled within 30s)")
    emitted = [r.getMessage() for r in caplog.records if "GPS serial-permission" in r.getMessage()]
    assert len(emitted) == 1  # only the first within the window is logged


def test_gps_driver_reports_permission_error_with_fix_hint(monkeypatch, caplog) -> None:
    """A serial PermissionError on the real path logs an actionable hint (not silent)."""
    import asyncio
    import sys
    import types

    fake_serial = types.ModuleType("serial")

    def _deny(*_a, **_k):
        raise PermissionError(13, "Permission denied")

    fake_serial.Serial = _deny  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    # Force the real-hardware path off-Pi.
    monkeypatch.delenv("SIM_MODE", raising=False)
    monkeypatch.setattr(
        "backend.src.drivers.sensors.gps_driver.is_simulation_mode", lambda: False
    )

    d = GPSDriver({"mode": GpsMode.LC29H_UART})
    d.initialized = True
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(d.read_position())

    assert result is None  # no fix available -> last known (None)
    msgs = [r.getMessage() for r in caplog.records if "GPS serial-permission" in r.getMessage()]
    assert msgs and "dialout" in msgs[0]


def test_gps_driver_diag_logs_new_reason_immediately(caplog) -> None:
    d = GPSDriver({"mode": GpsMode.LC29H_UART})
    with caplog.at_level(logging.WARNING):
        d._diag("no-device", "a")
        d._diag("no-nmea", "b")  # different reason -> not throttled
    reasons = [r.getMessage() for r in caplog.records if r.getMessage().startswith("GPS ")]
    assert any("no-device" in m for m in reasons)
    assert any("no-nmea" in m for m in reasons)
