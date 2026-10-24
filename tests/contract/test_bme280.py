"""Contract test (T040): BME280 environmental sensor reports temperature/humidity/pressure.

FR-023: ``GET /api/v2/sensors/environmental/status`` serves the driver's temperature,
humidity and pressure, and ``GET /api/v2/sensors/health`` reports the sensor ``error`` /
``offline`` (never ``online``) when it cannot be read.

What this does NOT prove: the real driver (``bme280_driver.py`` ``read_environment``) swallows
an I2C failure and returns its previous sample, and ``EnvironmentalSensorInterface`` keeps
the sensor ``online`` while it does, so an unreachable BME280 that *was* readable is not
reported degraded on real hardware. Only failures that raise (below) and a sensor that never
answers (``offline``) are surfaced.
"""

from __future__ import annotations

import pytest

from backend.src.models import SensorStatus


async def _health(client) -> dict:
    resp = await client.get("/api/v2/sensors/health")
    assert resp.status_code == 200
    return resp.json()["components"]


@pytest.mark.asyncio
async def test_bme280_reports_environmentals(api_client, install_sensor_manager):
    manager = await install_sensor_manager()

    resp = await api_client.get("/api/v2/sensors/environmental/status")

    assert resp.status_code == 200
    body = resp.json()
    reading = body["last_reading"]
    # Every field is the driver's own sample, under the right name (no swapped channels).
    sample = (await manager.environmental._driver.health_check())["last_env"]
    assert reading["temperature"] == sample["temperature_celsius"]
    assert reading["humidity"] == sample["humidity_percent"]
    assert reading["pressure"] == sample["pressure_hpa"]
    # And each is physically sensible for a BME280 (datasheet operating ranges).
    assert -40.0 <= reading["temperature"] <= 85.0
    assert 0.0 <= reading["humidity"] <= 100.0
    assert 300.0 <= reading["pressure"] <= 1100.0
    assert (await _health(api_client))["environmental"] == {"status": "online"}


class _TimedOutBme280:
    """BME280 stand-in whose I2C read times out."""

    async def read_environment(self):
        raise TimeoutError("BME280 I2C read timed out")


@pytest.mark.asyncio
async def test_bme280_timeout_reports_error_health(api_client, install_sensor_manager):
    manager = await install_sensor_manager()
    ok = (await api_client.get("/api/v2/sensors/environmental/status")).json()
    assert ok["last_reading"] is not None
    assert (await _health(api_client))["environmental"] == {"status": "online"}

    manager.environmental._driver = _TimedOutBme280()
    failed = await api_client.get("/api/v2/sensors/environmental/status")

    assert failed.status_code == 200
    assert failed.json()["last_reading"] is None  # the earlier sample is not replayed
    assert manager.environmental.status is SensorStatus.ERROR
    health = await _health(api_client)
    assert health["environmental"] == {"status": "error"}
    assert health["gps"] == {"status": "online"}


@pytest.mark.asyncio
async def test_bme280_not_answering_on_the_bus_is_reported_offline(
    api_client, i2c_bus_silent, install_sensor_manager
):
    await install_sensor_manager()

    resp = await api_client.get("/api/v2/sensors/environmental/status")

    assert resp.status_code == 200
    assert resp.json()["last_reading"] is None  # no reading is fabricated for an absent sensor
    health = await _health(api_client)
    assert health["environmental"] == {"status": "offline"}
    assert health["gps"] == {"status": "online"}
