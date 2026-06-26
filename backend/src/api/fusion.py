from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter

from ..core.simulation import is_simulation_mode
from ..fusion.ekf import SimpleEKF
from ..fusion.sensor_health import SensorHealthMonitor

router = APIRouter()


_ekf = SimpleEKF()
_health = SensorHealthMonitor()


@router.get("/api/v2/fusion/state")
async def get_fused_state() -> dict[str, Any]:
    """Return the EKF-fused navigation state with uncertainty.

    The Extended Kalman Filter advances on each request; in SIM_MODE we feed it
    synthetic motion plus periodic GPS/IMU measurements so the covariance (and
    therefore the fusion quality) stays bounded without hardware.
    """
    sim = is_simulation_mode() or os.environ.get("SIM_MODE") == "1"
    if sim:
        # Synthetic slow turn, with periodic measurement corrections.
        state = _ekf.step(v_mps=0.0, omega_dps=5.0)
        if int(time.time()) % 5 == 0:
            _ekf.update_gps_xy(0.0, 0.0, alpha=0.5)
            _ekf.update_heading(90.0, alpha=0.5)
            state = _ekf.get_state()
    else:
        state = _ekf.get_state()

    sensor_quality = _health.get_snapshot()
    # Combine filter uncertainty with sensor health for the reported quality.
    quality_label = state.quality
    if sensor_quality and sum(sensor_quality.values()) / max(len(sensor_quality), 1) <= 0.7:
        quality_label = "degraded"

    return {
        "position": {"x": state.x, "y": state.y},
        "heading": state.heading_deg,
        "timestamp": state.timestamp_s,
        "quality": quality_label,
        "uncertainty": {
            "position_std_m": round(state.position_std_m, 3),
            "heading_std_deg": round(state.heading_std_deg, 3),
        },
        "sources": list(state.sources) or ["gps", "imu", "odometry"],
        "sensor_quality": sensor_quality,
    }
