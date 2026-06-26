"""Unit tests for the EKF sensor-fusion filter."""

import math

from backend.src.fusion.ekf import SimpleEKF


def test_gps_updates_reduce_position_uncertainty():
    ekf = SimpleEKF()
    before = ekf.get_state().position_std_m
    for _ in range(10):
        ekf.update_gps_xy(5.0, -3.0, alpha=0.8)
    after = ekf.get_state()
    # Repeated GPS fixes must shrink the covariance and pull the estimate in.
    assert after.position_std_m < before
    assert abs(after.x - 5.0) < 1.0
    assert abs(after.y + 3.0) < 1.0
    assert after.quality in {"good", "degraded"}
    assert "gps" in after.sources


def test_heading_update_handles_wraparound():
    ekf = SimpleEKF()
    # Drive the heading near the +/-180 boundary, then measure across it.
    for _ in range(20):
        ekf.update_heading(179.0, alpha=0.8)
    for _ in range(20):
        ekf.update_heading(-179.0, alpha=0.8)
    heading = ekf.get_state().heading_deg
    # Should settle near 181 deg (== -179), not swing all the way around.
    diff = abs(((heading - 181.0 + 180.0) % 360.0) - 180.0)
    assert diff < 10.0, heading


def test_predict_grows_uncertainty_without_measurements():
    ekf = SimpleEKF()
    ekf.update_gps_xy(0.0, 0.0, alpha=0.9)
    tight = ekf.get_state().position_std_m
    for _ in range(50):
        ekf.predict(0.1, v_mps=0.5, omega_dps=0.0)
    loose = ekf.get_state().position_std_m
    assert loose > tight


def test_forward_motion_moves_along_heading():
    ekf = SimpleEKF()
    ekf.update_heading(0.0, alpha=0.9)  # face +x
    ekf.update_gps_xy(0.0, 0.0, alpha=0.9)
    for _ in range(10):
        ekf.predict(0.1, v_mps=1.0, omega_dps=0.0)
    state = ekf.get_state()
    # ~1 m/s for ~1 s along heading 0 -> x increases, y stays near 0.
    assert state.x > 0.5
    assert abs(state.y) < 0.5
    assert math.isfinite(state.heading_deg)
