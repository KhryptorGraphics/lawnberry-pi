from __future__ import annotations

import os

import pytest

from backend.src.drivers.sensors.gps_driver import (
    GPSDriver,
    fix_status_from_quality,
    parse_gga,
    parse_gst_sigma,
)
from backend.src.models.sensor_data import GpsMode


@pytest.mark.asyncio
async def test_gps_driver_sim_mode_produces_positions():
    os.environ["SIM_MODE"] = "1"
    drv = GPSDriver({"mode": GpsMode.F9P_USB.value})
    await drv.initialize()
    await drv.start()

    r1 = await drv.read_position()
    r2 = await drv.read_position()
    assert r1 is not None and r2 is not None
    assert r1.latitude is not None and r1.longitude is not None
    # Accuracy should be < 1m for F9P in SIM
    assert r1.accuracy is not None and r1.accuracy < 1.5
    # Subsequent reads should move slightly
    assert r2.latitude != r1.latitude or r2.longitude != r1.longitude


@pytest.mark.asyncio
async def test_gps_driver_health_check_fields():
    os.environ["SIM_MODE"] = "1"
    drv = GPSDriver({"mode": GpsMode.NEO8M_UART.value})
    await drv.initialize()
    await drv.start()
    await drv.read_position()
    hc = await drv.health_check()
    assert hc["driver"] == "gps"
    assert hc["mode"] in {GpsMode.NEO8M_UART.value, GpsMode.F9P_USB.value}
    assert hc["initialized"] is True
    assert hc["running"] is True


def test_gps_driver_parse_gga_and_fix_quality():
    sentence = "$GNGGA,123519,4807.038,N,01131.000,E,4,12,0.7,545.4,M,46.9,M,1.0,0000*5B"
    gga = parse_gga(sentence)
    assert gga is not None
    assert gga.lat == pytest.approx(48.1173, rel=1e-4)
    assert gga.lon == pytest.approx(11.5166667, rel=1e-4)
    assert gga.altitude_m == pytest.approx(545.4)
    assert gga.satellites == 12
    assert gga.hdop == pytest.approx(0.7)
    assert gga.quality == 4 and gga.correction_age_s == pytest.approx(1.0)
    assert gga.utc_s == 12 * 3600 + 35 * 60 + 19
    # Fix quality 4 should map to RTK fixed status
    assert fix_status_from_quality(gga.quality) == "RTK_FIXED"


def test_gps_driver_parse_gst_accuracy():
    # SD values 0.02 m and 0.03 m should combine via sqrt(a^2 + b^2)
    gst = "$GPGST,024603.00,0.010,0.020,0.030,0.000,0.020,0.030,0.050*5C"
    assert parse_gst_sigma(gst) == pytest.approx((0.02**2 + 0.03**2) ** 0.5)


def test_gps_driver_fix_quality_mapping_is_fail_closed():
    assert fix_status_from_quality(None) == "NO_FIX"
    assert fix_status_from_quality(0) == "NO_FIX"
    assert fix_status_from_quality(1) == "GPS_FIX"
    assert fix_status_from_quality(2) == "DGPS"
    assert fix_status_from_quality(5) == "RTK_FLOAT"
    # A quality outside the table is not a fix (3 = PPS, 9 = WAAS on some receivers, or garbage)
    for unknown in (3, 9, 42, -1):
        assert fix_status_from_quality(unknown) == "NO_FIX"
