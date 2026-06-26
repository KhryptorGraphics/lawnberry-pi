"""Extended Kalman Filter for planar navigation (T049, FR-025).

Fuses a constant-velocity / constant-turn-rate motion model with GPS position
and IMU heading measurements. The filter tracks a full 3x3 covariance and uses
standard EKF predict/update with Kalman gain, so the reported fused state comes
with real uncertainty estimates (used to label fusion quality).

State vector: ``[x_m, y_m, heading_rad]`` (heading kept wrapped to [-pi, pi]).
SIM_MODE-safe: pure numpy, no hardware or heavy dependencies.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class FusedState:
    x: float
    y: float
    heading_deg: float
    timestamp_s: float
    quality: str = "unknown"
    sources: tuple[str, ...] = ()
    position_std_m: float = 0.0
    heading_std_deg: float = 0.0


def _wrap_pi(angle: float) -> float:
    """Wrap an angle (radians) to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SimpleEKF:
    """A 2D pose Extended Kalman Filter (x, y, heading)."""

    # Measurement noise (1-sigma).
    GPS_STD_M = 0.5
    HEADING_STD_DEG = 5.0
    # Process noise spectral density (per second).
    PROC_POS_STD = 0.05
    PROC_HEADING_STD_DEG = 1.0

    def __init__(self) -> None:
        self._x = np.zeros(3, dtype=float)  # [x, y, theta]
        # Large initial uncertainty until the first measurements arrive.
        self._P = np.diag([100.0, 100.0, math.radians(180.0) ** 2])
        self._Q = np.diag(
            [
                self.PROC_POS_STD**2,
                self.PROC_POS_STD**2,
                math.radians(self.PROC_HEADING_STD_DEG) ** 2,
            ]
        )
        self._last_ts = time.time()
        self._sources: set[str] = set()

    def reset(self) -> None:
        self.__init__()

    # --- prediction ---------------------------------------------------------

    def predict(self, dt: float, v_mps: float = 0.0, omega_dps: float = 0.0) -> None:
        """Propagate the state with a constant-velocity, constant-turn model."""
        if dt <= 0:
            return
        theta = float(self._x[2])
        omega = math.radians(omega_dps)

        self._x[0] += v_mps * math.cos(theta) * dt
        self._x[1] += v_mps * math.sin(theta) * dt
        self._x[2] = _wrap_pi(theta + omega * dt)

        # Jacobian of the motion model w.r.t. the state.
        f_jac = np.array(
            [
                [1.0, 0.0, -v_mps * math.sin(theta) * dt],
                [0.0, 1.0, v_mps * math.cos(theta) * dt],
                [0.0, 0.0, 1.0],
            ]
        )
        self._P = f_jac @ self._P @ f_jac.T + self._Q * dt

    # --- measurement updates ------------------------------------------------

    def update_gps_xy(self, x: float, y: float, alpha: float = 0.5) -> None:
        """Fuse a GPS position fix. ``alpha`` (0, 1] scales measurement trust."""
        z = np.array([x, y], dtype=float)
        h_jac = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        r = (self.GPS_STD_M**2) / max(alpha, 1e-3)
        self._kalman_update(z, h_jac, np.diag([r, r]))
        self._sources.add("gps")

    def update_heading(self, heading_deg: float, alpha: float = 0.5) -> None:
        """Fuse an IMU heading measurement (handles angular wrap-around)."""
        h_jac = np.array([[0.0, 0.0, 1.0]])
        innovation = np.array([_wrap_pi(math.radians(heading_deg) - float(self._x[2]))])
        r = (math.radians(self.HEADING_STD_DEG) ** 2) / max(alpha, 1e-3)
        s = h_jac @ self._P @ h_jac.T + np.array([[r]])
        gain = self._P @ h_jac.T @ np.linalg.inv(s)
        self._x = self._x + (gain @ innovation)
        self._x[2] = _wrap_pi(float(self._x[2]))
        self._P = (np.eye(3) - gain @ h_jac) @ self._P
        self._sources.add("imu")

    def _kalman_update(self, z: np.ndarray, h_jac: np.ndarray, r_cov: np.ndarray) -> None:
        innovation = z - (h_jac @ self._x)
        s = h_jac @ self._P @ h_jac.T + r_cov
        gain = self._P @ h_jac.T @ np.linalg.inv(s)
        self._x = self._x + gain @ innovation
        self._x[2] = _wrap_pi(float(self._x[2]))
        self._P = (np.eye(3) - gain @ h_jac) @ self._P

    # --- accessors ----------------------------------------------------------

    def step(self, v_mps: float = 0.0, omega_dps: float = 0.0) -> FusedState:
        now = time.time()
        self.predict(now - self._last_ts, v_mps=v_mps, omega_dps=omega_dps)
        self._last_ts = now
        if v_mps or omega_dps:
            self._sources.add("odometry")
        return self.get_state(now)

    def get_state(self, now: float | None = None) -> FusedState:
        ts = now if now is not None else time.time()
        pos_std = math.sqrt(max(float(self._P[0, 0] + self._P[1, 1]), 0.0))
        heading_std = math.degrees(math.sqrt(max(float(self._P[2, 2]), 0.0)))
        if pos_std < 1.0:
            quality = "good"
        elif pos_std < 5.0:
            quality = "degraded"
        else:
            quality = "poor"
        return FusedState(
            x=float(self._x[0]),
            y=float(self._x[1]),
            heading_deg=math.degrees(float(self._x[2])) % 360.0,
            timestamp_s=ts,
            quality=quality,
            sources=tuple(sorted(self._sources)),
            position_std_m=pos_std,
            heading_std_deg=heading_std,
        )


__all__ = ["SimpleEKF", "FusedState"]
