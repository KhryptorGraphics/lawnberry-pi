"""Contract test (T041): INA3221 power monitor provides battery/solar metrics.

FR-024: ``GET /api/v2/sensors/power/status`` serves the INA3221 battery and solar voltage
and current, and ``GET /api/v2/sensors/health`` reports the monitor ``offline`` (never
``online``) when it does not answer on the bus.

What this does NOT prove:

- Which physical INA3221 channel is solar and which is battery. The driver says CH1 solar /
  CH3 battery while ``spec/hardware.yaml`` and CI say CH1 battery / CH3 solar (audit N-12,
  an owner decision), so this file checks the driver's *named* outputs end to end and
  nothing about channel numbers.
- Read-time timeouts. ``INA3221Driver.read_power`` returns its last sample when an I2C read
  fails and ``PowerSensorInterface.read_power`` keeps the monitor ``online`` while it does,
  so a monitor that stops answering after a good read is not reported degraded and no
  ``last_error`` exists to assert.
"""

from __future__ import annotations

import math

import pytest

from backend.src.models import SensorStatus

# The shape the hub builds from HardwareConfig.ina3221_config (websocket_hub.py), with the
# shunt values from config/hardware.yaml.
POWER_CONFIG = {
    "ina3221": {"address": 0x40, "bus": 1, "shunt_ohms_ch1": 0.0025, "shunt_ohms_ch3": 0.0015}
}


async def _health(client) -> dict:
    resp = await client.get("/api/v2/sensors/health")
    assert resp.status_code == 200
    return resp.json()["components"]


@pytest.mark.asyncio
async def test_ina3221_reports_battery_and_solar_metrics(api_client, install_sensor_manager):
    manager = await install_sensor_manager(power_config=POWER_CONFIG)

    resp = await api_client.get("/api/v2/sensors/power/status")

    assert resp.status_code == 200
    reading = resp.json()["last_reading"]
    # Each API field is the driver's matching output (no battery/solar or V/A mix-up) ...
    sample = (await manager.power._ina_driver.health_check())["last_power"]
    assert reading == {
        "battery_voltage": sample["battery_voltage"],
        "battery_current": sample["battery_current_amps"],
        "solar_voltage": sample["solar_voltage"],
        "solar_current": sample["solar_current_amps"],
    }
    # ... and none is missing or non-finite.
    assert all(isinstance(v, float) and math.isfinite(v) for v in reading.values())
    # A 12 V LiFePO4 pack: the same 9-16 V window SensorManager treats as plausible.
    assert 9.0 <= reading["battery_voltage"] <= 16.0
    assert (await _health(api_client))["power"] == {"status": "online"}


@pytest.mark.asyncio
async def test_ina3221_not_answering_on_the_bus_is_reported_offline(
    api_client, i2c_bus_silent, install_sensor_manager
):
    manager = await install_sensor_manager(power_config=POWER_CONFIG)

    resp = await api_client.get("/api/v2/sensors/power/status")

    assert resp.status_code == 200
    assert resp.json()["last_reading"] is None  # no voltage or current is fabricated
    assert manager.power.status is SensorStatus.OFFLINE
    health = await _health(api_client)
    assert health["power"] == {"status": "offline"}
    assert health["gps"] == {"status": "online"}


@pytest.mark.asyncio
async def test_power_without_any_driver_configured_is_offline_not_online(
    api_client, install_sensor_manager
):
    # No ina3221/victron block at all (the default SensorManager): nothing to read.
    await install_sensor_manager()

    resp = await api_client.get("/api/v2/sensors/power/status")

    assert resp.json()["last_reading"] is None
    health = await _health(api_client)
    assert health["power"] == {"status": "offline"}
    assert health["gps"] == {"status": "online"}
