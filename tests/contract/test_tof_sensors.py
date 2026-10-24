"""Contract test (T038): ToF obstacle interlock and ToF health.

FR-021 / NFR-008:

- A left/right ToF distance at or inside ``limits.tof_obstacle_distance_meters`` (0.2 m in
  ``config/limits.yaml``) raises the ``obstacle_detected`` interlock in the software path:
  ``POST /api/v2/debug/sensors/inject-tof`` -> ``SafetyTriggerManager.trigger_obstacle`` ->
  ``RobotState.active_interlocks`` -> ``GET /api/v2/status``.
- A ToF that stops answering is reported as ``error`` / ``offline`` by
  ``GET /api/v2/sensors/health``, never as ``online``.

What this does NOT prove: any actuator effect. The injector only records the interlock; nothing
on this path stops a motor (the stop path is the obstacle-stop work of audit N-11 and is tested
with that code). And a real VL53L0X driver keeps its last distance when a read fails
(``vl53l0x_driver.py`` ``read_distance_mm``), so a driver-level timeout is not surfaced; only a
read that raises, or a sensor that never answers on the bus, is.
"""

from __future__ import annotations

import pytest

from backend.src.core import globals as app_globals
from backend.src.core.config_loader import ConfigLoader
from backend.src.models import SensorStatus
from backend.src.models.safety_interlock import InterlockState, InterlockType
from backend.src.safety.safety_triggers import get_safety_trigger_manager


def _threshold_m() -> float:
    threshold = ConfigLoader().get()[1].tof_obstacle_distance_meters
    # Metres, not millimetres: the injector and the limit share this unit.
    assert 0.05 <= threshold <= 1.0
    return threshold


async def _inject_tof(client, position: str, distance_m: float) -> dict:
    resp = await client.post(
        "/api/v2/debug/sensors/inject-tof", json={"position": position, "distance_m": distance_m}
    )
    assert resp.status_code == 200
    return resp.json()


async def _health(client) -> dict:
    resp = await client.get("/api/v2/sensors/health")
    assert resp.status_code == 200
    return resp.json()["components"]


@pytest.mark.asyncio
async def test_tof_reading_inside_threshold_raises_the_obstacle_interlock(api_client, nav_state):
    threshold = _threshold_m()
    inside = threshold * 0.75  # 0.15 m with the shipped 0.2 m limit

    body = await _inject_tof(api_client, "left", inside)
    assert body == {
        "ok": True,
        "override": {"position": "left", "distance_m": inside},
        "safety": {"interlock": "obstacle_detected", "threshold_m": threshold},
    }

    # The interlock is recorded on the canonical robot state...
    assert len(nav_state.active_interlocks) == 1
    interlock = nav_state.active_interlocks[0]
    assert interlock.interlock_type is InterlockType.OBSTACLE_DETECTED
    assert interlock.state is InterlockState.ACTIVE
    assert interlock.trigger_value == inside
    assert app_globals._debug_overrides["tof_left_distance_m"] == inside

    # ...and is what the status endpoint shows, both as the list and as the named flag.
    safety = (await api_client.get("/api/v2/status")).json()["safety_status"]
    assert safety["safety_interlocks"] == ["obstacle_detected"]
    assert safety["obstacle_detected"] is True
    assert safety["tilt_detected"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("position", "offset_m", "triggers"),
    [
        ("left", -0.05, True),
        ("right", -0.05, True),
        ("left", 0.0, True),  # exactly at the limit is inside it
        ("right", 0.001, False),
        ("left", 0.3, False),
        ("right", 1.3, False),
    ],
)
async def test_tof_threshold_boundary_per_side(api_client, nav_state, position, offset_m, triggers):
    threshold = _threshold_m()

    body = await _inject_tof(api_client, position, threshold + offset_m)

    safety = (await api_client.get("/api/v2/status")).json()["safety_status"]
    if triggers:
        assert body["safety"] == {"interlock": "obstacle_detected", "threshold_m": threshold}
        assert [i.interlock_type for i in nav_state.active_interlocks] == [
            InterlockType.OBSTACLE_DETECTED
        ]
        assert safety["safety_interlocks"] == ["obstacle_detected"]
        assert safety["obstacle_detected"] is True
    else:
        assert body["safety"] is None
        assert nav_state.active_interlocks == []
        assert safety["safety_interlocks"] == []
        assert safety["obstacle_detected"] is False
    assert body["override"] == {"position": position, "distance_m": threshold + offset_m}


@pytest.mark.asyncio
async def test_tof_injection_rejects_an_unknown_side(api_client, nav_state):
    resp = await api_client.post(
        "/api/v2/debug/sensors/inject-tof", json={"position": "rear", "distance_m": 0.1}
    )
    assert resp.status_code == 400
    assert nav_state.active_interlocks == []


@pytest.mark.asyncio
async def test_status_flags_an_ultrasonic_obstacle_as_an_obstacle(api_client, nav_state):
    # The status endpoint's single "obstacle_detected" flag covers both obstacle sensors.
    assert get_safety_trigger_manager().trigger_ultrasonic_obstacle(20.0, 30.0, "inner_left")

    safety = (await api_client.get("/api/v2/status")).json()["safety_status"]
    assert safety["safety_interlocks"] == ["ultrasonic_obstacle"]
    assert safety["obstacle_detected"] is True
    assert safety["tilt_detected"] is False


class _Rangefinder:
    """VL53L0X stand-in: answers ``distance_mm``, or raises ``error`` on every read."""

    def __init__(self, distance_mm: int | None = None, error: Exception | None = None) -> None:
        self._distance_mm = distance_mm
        self._error = error

    async def read_distance_mm(self):
        if self._error is not None:
            raise self._error
        return self._distance_mm


@pytest.mark.asyncio
async def test_tof_timeout_reports_degraded_health(api_client, install_sensor_manager):
    manager = await install_sensor_manager()
    manager.tof._left = _Rangefinder(distance_mm=1500)
    manager.tof._right = _Rangefinder(distance_mm=1480)

    # Control: both sensors answer, so the readings flow through and both are online.
    data = await manager.read_all_sensors()
    assert (data.tof_left.distance, data.tof_right.distance) == (1500.0, 1480.0)
    health = await _health(api_client)
    assert health["tof_left"] == {"status": "online"}
    assert health["tof_right"] == {"status": "online"}

    # The right sensor stops answering (I2C timeout) at the next sensor poll.
    manager.tof._right = _Rangefinder(error=TimeoutError("VL53L0X I2C timeout"))
    data = await manager.read_all_sensors()

    assert data.tof_right is None  # no distance is invented
    assert manager.tof.status is SensorStatus.ERROR
    health = await _health(api_client)
    assert health["tof_right"] == {"status": "error"}
    assert health["gps"] == {"status": "online"}  # the fault is not reported for everything


@pytest.mark.asyncio
async def test_tof_not_answering_on_the_bus_is_reported_offline(
    api_client, i2c_bus_silent, install_sensor_manager
):
    manager = await install_sensor_manager()

    health = await _health(api_client)
    assert health["tof_left"] == {"status": "offline"}
    assert health["tof_right"] == {"status": "offline"}
    assert health["gps"] == {"status": "online"}  # serial, not I2C: the bus fault is not global
    assert await manager.tof.read_tof_sensors() == (None, None)
