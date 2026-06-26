"""Location & coverage feature engineering for runtime VLA inference.

Runtime counterpart of the training-side feature builder. The deployed model is
trained on RTK-derived *location* features and a causal *coverage map*; to feed
it correctly the Pi must build the **same** features at inference time, in the
**same** local coordinate frame.

> SYNC CONTRACT: this module must stay byte-for-byte equivalent (constants +
> math) with the training module `mower/models/vla/features.py` on the Thor
> training server. ``GPS_FEATURE_DIM``, ``COVERAGE_GRID_SIZE``,
> ``DEFAULT_YARD_HALF_SIZE_M`` and the ENU formula here must equal that file, or
> the model will see inputs from a different distribution than it trained on.
> The local-frame *datum* must also match: set the training
> ``config/training.yaml: data.yard_datum`` to the Pi's home / geofence origin.

Pure numpy / stdlib so it imports without hardware and stays SIM-safe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

# Feature dims — MUST equal mower/models/vla/features.py.
GPS_FEATURE_DIM = 8
COVERAGE_GRID_SIZE = 64  # NxN occupancy grid
DEFAULT_YARD_HALF_SIZE_M = 50.0  # grid / normalisation spans +/- this around datum
DEFAULT_MAX_SPEED_MPS = 2.0

# RTK fix quality -> ordinal confidence in [0, 1] (mirrors RTKFixType).
_FIX_QUALITY = {
    "none": 0.0,
    "single": 0.4,
    "dgps": 0.5,
    "pps": 0.6,
    "float": 0.75,
    "fixed": 1.0,
}

_EARTH_R = 6378137.0  # WGS84 equatorial radius (m)


def latlon_to_enu(
    lat: float, lon: float, lat0: float, lon0: float
) -> tuple[float, float]:
    """Equirectangular lat/lon -> local east/north metres about a datum."""
    lat0r = math.radians(lat0)
    east = math.radians(lon - lon0) * _EARTH_R * math.cos(lat0r)
    north = math.radians(lat - lat0) * _EARTH_R
    return east, north


def _get(gps: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    """Read a field from either a mapping or an object (e.g. GPSData)."""
    if isinstance(gps, Mapping):
        return gps.get(key, default)
    return getattr(gps, key, default)


def gps_feature_vector(
    gps: Mapping[str, Any] | Any,
    datum: tuple[float, float],
    yard_half_size_m: float = DEFAULT_YARD_HALF_SIZE_M,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> np.ndarray:
    """Build the ``GPS_FEATURE_DIM`` location/motion/confidence feature vector.

    Accepts either a dict (training side) or a ``GPSData``-like object (runtime),
    so the math stays identical across train and inference.
    """
    lat = float(_get(gps, "latitude", 0.0))
    lon = float(_get(gps, "longitude", 0.0))
    east, north = latlon_to_enu(lat, lon, datum[0], datum[1])
    scale = max(yard_half_size_m, 1e-6)
    heading = math.radians(float(_get(gps, "heading", 0.0)))
    speed = float(_get(gps, "speed_mps", 0.0))
    fix = str(_get(gps, "rtk_fix_type", "none")).lower()
    hdop = float(_get(gps, "hdop", 99.0))
    sats = float(_get(gps, "num_satellites", 0))
    return np.array(
        [
            float(np.clip(east / scale, -1.0, 1.0)),  # local east (normalised)
            float(np.clip(north / scale, -1.0, 1.0)),  # local north (normalised)
            math.sin(heading),  # heading as a unit vector (no 360->0 wrap)
            math.cos(heading),
            float(np.clip(speed / max(max_speed_mps, 1e-6), 0.0, 1.0)),
            _FIX_QUALITY.get(fix, 0.0),  # RTK fix quality
            1.0 / (1.0 + max(hdop, 0.0)),  # HDOP -> confidence (~1 good, ~0 bad)
            float(np.clip(sats / 16.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


class CoverageMap:
    """Causal NxN occupancy grid of already-mowed yard cells (runtime).

    Centred on ``datum`` and spanning +/- ``yard_half_size_m`` metres. During an
    autonomous mow: :meth:`snapshot` returns coverage so far (the model input),
    then :meth:`mark` records the current position once the action is applied —
    matching the causal ordering used in training.
    """

    def __init__(
        self,
        datum: tuple[float, float],
        grid_size: int = COVERAGE_GRID_SIZE,
        yard_half_size_m: float = DEFAULT_YARD_HALF_SIZE_M,
        stamp_radius: int = 1,
    ):
        self.datum = datum
        self.n = grid_size
        self.half = yard_half_size_m
        self.cell = (2.0 * yard_half_size_m) / grid_size
        self.stamp = stamp_radius
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    def _ij(self, lat: float, lon: float) -> tuple[int, int]:
        east, north = latlon_to_enu(lat, lon, self.datum[0], self.datum[1])
        col = int((east + self.half) / self.cell)
        row = int((self.half - north) / self.cell)  # north points up (row 0 = top)
        return row, col

    def snapshot(self) -> np.ndarray:
        """Coverage so far as a (N, N) float32 grid (copy)."""
        return self.grid.copy()

    def mark(self, lat: float, lon: float) -> None:
        """Stamp the cell(s) at this position as mowed."""
        row, col = self._ij(lat, lon)
        if not (0 <= row < self.n and 0 <= col < self.n):
            return
        r = self.stamp
        self.grid[
            max(0, row - r) : row + r + 1, max(0, col - r) : col + r + 1
        ] = 1.0


__all__ = [
    "GPS_FEATURE_DIM",
    "COVERAGE_GRID_SIZE",
    "DEFAULT_YARD_HALF_SIZE_M",
    "latlon_to_enu",
    "gps_feature_vector",
    "CoverageMap",
]
