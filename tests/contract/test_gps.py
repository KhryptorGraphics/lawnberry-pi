"""Contract test (T053): GPS position and accuracy (<5 m) through the API.

FR-028 / FR-033: the position the API serves carries its accuracy, and a fix is only
"good" when that accuracy is under 5 m (the limit ``GPSDegradationConfig.max_accuracy_m``
uses). Two paths are covered:

- the navigation debug path (``POST /api/v2/debug/gps/inject`` -> ``GET /api/v2/nav/status``);
- the sensor path (``SensorManager.gps`` -> ``GET /api/v2/sensors/gps/status`` and
  ``/api/v2/sensors/health``), in SIM_MODE and with an injected driver fault.
"""

from __future__ import annotations

import pytest

from backend.src.models import GpsMode, SensorStatus

ACCURACY_LIMIT_M = 5.0


async def _inject(client, **fix) -> None:
    resp = await client.post("/api/v2/debug/gps/inject", json=fix)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def _reported_position(client) -> dict:
    resp = await client.get("/api/v2/nav/status")
    assert resp.status_code == 200
    return resp.json()["position"]


@pytest.mark.asyncio
async def test_gps_accuracy_under_5m(api_client, nav_state):
    # The position is the placeholder's original fix: 7 decimals is ~1 cm and must survive.
    await _inject(api_client, latitude=37.4219999, longitude=-122.0840575, accuracy_m=4.0)

    position = await _reported_position(api_client)
    assert position == {"latitude": 37.4219999, "longitude": -122.0840575, "accuracy_m": 4.0}
    assert position["accuracy_m"] < ACCURACY_LIMIT_M


@pytest.mark.asyncio
async def test_poor_fix_is_reported_as_poor(api_client, nav_state):
    await _inject(api_client, latitude=37.4220, longitude=-122.0841, accuracy_m=6.5)

    position = await _reported_position(api_client)
    assert position["accuracy_m"] == 6.5
    assert position["accuracy_m"] >= ACCURACY_LIMIT_M


@pytest.mark.asyncio
async def test_fix_without_accuracy_is_never_reported_as_good(api_client, nav_state):
    await _inject(api_client, latitude=37.4220, longitude=-122.0841)

    position = await _reported_position(api_client)
    assert position["accuracy_m"] is not None
    assert position["accuracy_m"] >= ACCURACY_LIMIT_M


@pytest.mark.asyncio
@pytest.mark.parametrize("gps_mode", list(GpsMode), ids=lambda mode: mode.value)
async def test_sensor_gps_status_serves_a_fix_under_5m(
    api_client, install_sensor_manager, gps_mode
):
    await install_sensor_manager(gps_mode=gps_mode)

    resp = await api_client.get("/api/v2/sensors/gps/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == gps_mode.value
    fix = body["last_reading"]
    assert -90.0 <= fix["latitude"] <= 90.0
    assert -180.0 <= fix["longitude"] <= 180.0
    assert fix["satellites"] >= 4
    assert fix["accuracy"] < ACCURACY_LIMIT_M

    health = (await api_client.get("/api/v2/sensors/health")).json()
    assert health["components"]["gps"] == {"status": "online"}


@pytest.mark.asyncio
async def test_gps_driver_failure_reports_error_and_no_position(api_client, install_sensor_manager):
    manager = await install_sensor_manager()

    # Control: the receiver delivers a fix, so there is a cached position to (wrongly) reuse.
    resp = await api_client.get("/api/v2/sensors/gps/status")
    assert resp.json()["last_reading"] is not None
    assert manager.gps.last_reading is not None

    class _BrokenReceiver:
        async def read_position(self):
            raise OSError("GPS serial port vanished")

    manager.gps._driver = _BrokenReceiver()

    resp = await api_client.get("/api/v2/sensors/gps/status")
    assert resp.status_code == 200
    # The cached fix is not served as if it were current.
    assert resp.json()["last_reading"] is None
    assert manager.gps.status == SensorStatus.ERROR

    health = (await api_client.get("/api/v2/sensors/health")).json()
    assert health["components"]["gps"] == {"status": "error"}
