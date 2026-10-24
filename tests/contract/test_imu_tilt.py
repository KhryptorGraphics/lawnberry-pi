"""Contract test (T039): IMU tilt detection and IMU health.

FR-022 / NFR-008:

- A roll or pitch at or beyond ``limits.tilt_threshold_degrees`` (30 deg in
  ``config/limits.yaml``) raises the ``tilt_detected`` interlock in the software path:
  ``POST /api/v2/debug/sensors/inject-tilt`` -> ``SafetyTriggerManager.trigger_tilt`` ->
  ``RobotState.active_interlocks`` -> ``GET /api/v2/status``.
- An IMU whose bus read fails is reported ``error``, never ``online``, by
  ``GET /api/v2/sensors/health``.

What this does NOT prove: that a blade stops, or how fast. The injector only records the
interlock; nothing on this path commands an actuator. The tilt -> blade-off latch is
``TractorSafetyMonitor`` (``tests/unit/test_tractor_safety_monitor.py``,
``test_tilt_over_threshold_disengages_engaged_blade``), and the 200 ms budget
(``tilt_cutoff_latency_ms``) has no test that measures more than a function call.
"""

from __future__ import annotations

import pytest

from backend.src.core import globals as app_globals
from backend.src.core.config_loader import ConfigLoader
from backend.src.models import SensorStatus
from backend.src.models.safety_interlock import InterlockState, InterlockType
from backend.src.safety.safety_triggers import get_safety_trigger_manager


def _threshold_deg() -> float:
    threshold = ConfigLoader().get()[1].tilt_threshold_degrees
    # Degrees, not radians.
    assert 10.0 <= threshold <= 60.0
    return threshold


async def _inject_tilt(client, **angles: float) -> dict:
    resp = await client.post("/api/v2/debug/sensors/inject-tilt", json=angles)
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_tilt_over_threshold_raises_the_tilt_interlock(api_client, nav_state):
    over = _threshold_deg() + 5.0  # 35 deg with the shipped 30 deg limit
    body = await _inject_tilt(api_client, roll_deg=over, pitch_deg=0.0)

    assert body == {"ok": True, "over_threshold": True}
    assert len(nav_state.active_interlocks) == 1
    interlock = nav_state.active_interlocks[0]
    assert interlock.interlock_type is InterlockType.TILT_DETECTED
    assert interlock.state is InterlockState.ACTIVE
    assert interlock.trigger_value == over
    assert app_globals._debug_overrides["imu_roll_deg"] == over
    assert app_globals._debug_overrides["imu_pitch_deg"] == 0.0

    # The status endpoint shows it both as the list and as the named flag.
    safety = (await api_client.get("/api/v2/status")).json()["safety_status"]
    assert safety["safety_interlocks"] == ["tilt_detected"]
    assert safety["tilt_detected"] is True
    assert safety["obstacle_detected"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("angles", "triggers"),
    [
        pytest.param(lambda limit: {"roll_deg": limit - 0.1}, False, id="roll just under"),
        pytest.param(lambda limit: {"pitch_deg": limit - 0.1}, False, id="pitch just under"),
        pytest.param(lambda limit: {"roll_deg": limit}, True, id="roll exactly at the limit"),
        pytest.param(lambda limit: {"pitch_deg": limit}, True, id="pitch exactly at the limit"),
        pytest.param(lambda limit: {"roll_deg": -(limit + 5.0)}, True, id="negative roll"),
        pytest.param(lambda limit: {"pitch_deg": -(limit + 5.0)}, True, id="negative pitch"),
        pytest.param(
            lambda limit: {"roll_deg": -(limit - 0.1), "pitch_deg": limit - 0.1},
            False,
            id="both just under",
        ),
    ],
)
async def test_tilt_threshold_boundary(api_client, nav_state, angles, triggers):
    body = await _inject_tilt(api_client, **angles(_threshold_deg()))

    assert body == {"ok": True, "over_threshold": triggers}
    kinds = [i.interlock_type for i in nav_state.active_interlocks]
    assert kinds == ([InterlockType.TILT_DETECTED] if triggers else [])


@pytest.mark.parametrize(
    ("roll_deg", "pitch_deg", "triggers"),
    [
        (-35.0, 0.0, True),
        (0.0, -35.0, True),
        (-30.0, 0.0, True),
        (-29.9, 29.9, False),
        (0.0, 0.0, False),
    ],
)
def test_trigger_tilt_uses_the_magnitude_of_signed_imu_angles(
    nav_state, roll_deg, pitch_deg, triggers
):
    # The injector hands the trigger absolute values, but the real monitor
    # (TractorSafetyMonitor) passes the IMU's signed roll/pitch straight through.
    assert get_safety_trigger_manager().trigger_tilt(roll_deg, pitch_deg, 30.0) is triggers
    assert bool(nav_state.active_interlocks) is triggers


@pytest.mark.asyncio
async def test_cleared_tilt_stays_listed_until_acknowledged_but_is_no_longer_detected(
    api_client, nav_state
):
    triggers = get_safety_trigger_manager()
    assert triggers.trigger_tilt(35.0, 0.0, 30.0)
    safety = (await api_client.get("/api/v2/status")).json()["safety_status"]
    assert safety["tilt_detected"] is True

    # The machine is level again: the interlock moves to cleared-pending-ack. It stays in the
    # list (an operator still has to acknowledge it) but the tilt is no longer being detected.
    triggers.clear_tilt()
    safety = (await api_client.get("/api/v2/status")).json()["safety_status"]
    assert safety["safety_interlocks"] == ["tilt_detected"]
    assert safety["tilt_detected"] is False
    assert [i.state for i in nav_state.active_interlocks] == [InterlockState.CLEARED_PENDING_ACK]


class _DeadImu:
    """BNO085 stand-in whose I2C transaction fails."""

    async def read_orientation(self):
        raise OSError("BNO085 I2C read failed (errno 121)")


@pytest.mark.asyncio
async def test_imu_comms_failure_reports_error_health(api_client, install_sensor_manager):
    manager = await install_sensor_manager()

    # Control: the IMU answers and is reported online with an orientation.
    ok = (await api_client.get("/api/v2/sensors/imu/status")).json()
    assert ok["last_reading"]["roll"] is not None
    health = (await api_client.get("/api/v2/sensors/health")).json()["components"]
    assert health["imu"] == {"status": "online"}

    manager.imu._driver = _DeadImu()
    failed = await api_client.get("/api/v2/sensors/imu/status")

    assert failed.status_code == 200
    assert failed.json()["last_reading"] is None  # the earlier orientation is not replayed
    assert manager.imu.status is SensorStatus.ERROR
    health = (await api_client.get("/api/v2/sensors/health")).json()["components"]
    assert health["imu"] == {"status": "error"}
    assert health["gps"] == {"status": "online"}
