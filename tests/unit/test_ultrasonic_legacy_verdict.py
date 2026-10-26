"""S06 (PR #20 review 3, HW-07): the entry-based obstacle verdict leaves unfitted channels out.

``UltrasonicSensorInterface.obstacle_verdict`` has two sources. With the MCU backend running it asks
the driver; otherwise (the legacy HC-SR04 backend, or an interface that is not enabled) it judges
the entries in ``last_readings``. That second path said "Disabled (not fitted) channels are
excluded" and nothing tested it: a channel that is not fitted would have counted as an invalid echo
and held the machine for ever.
"""

from __future__ import annotations

import pytest

from backend.src.services.sensor_manager import SensorCoordinator, UltrasonicSensorInterface


def entry(sensor_id: str, *, cm: float | None = None, disabled: bool = False) -> dict:
    """One ``read_ultrasonic`` entry (only the keys the verdict reads)."""
    valid = cm is not None
    return {
        "sensor_id": sensor_id,
        "distance_cm": cm if valid else 0.0,
        "valid": valid,
        "status": "disabled" if disabled else ("ok" if valid else "no_echo"),
        "verdict": "disabled" if disabled else "unknown",
    }


@pytest.fixture
def iface():
    iface = UltrasonicSensorInterface(
        SensorCoordinator(), config={"backend": "gpio_hcsr04"}, obstacle_threshold_cm=30.0
    )
    iface.enabled = True  # as the manager leaves it after initialize()
    return iface


def test_a_channel_that_is_not_fitted_does_not_hold_the_others_clear_verdict(iface):
    iface.last_readings = [
        entry("front_left", cm=120.0),
        entry("front_center", disabled=True),
        entry("front_right", cm=90.0),
    ]

    verdict = iface.obstacle_verdict()

    assert verdict["state"] == "clear"
    assert verdict["reasons"] == []
    assert verdict["min_cm"] == 90.0


def test_a_fitted_channel_without_an_echo_is_still_not_clear(iface):
    iface.last_readings = [
        entry("front_left", cm=120.0),
        entry("front_center", disabled=True),
        entry("front_right"),
    ]

    verdict = iface.obstacle_verdict()

    assert verdict["state"] == "unknown"
    assert verdict["reasons"] == ["front_right:no_echo"], "only the fitted channel is blamed"


def test_an_obstacle_on_a_fitted_channel_beats_a_missing_one(iface):
    iface.last_readings = [
        entry("front_left", cm=20.0),
        entry("front_center", disabled=True),
        entry("front_right", cm=90.0),
    ]

    assert iface.obstacle_verdict()["state"] == "obstacle"


def test_with_no_channel_fitted_the_answer_is_unknown(iface):
    iface.last_readings = [entry(name, disabled=True) for name in iface.channel_ids]

    verdict = iface.obstacle_verdict()

    assert verdict["state"] == "unknown" and verdict["reasons"] == []
