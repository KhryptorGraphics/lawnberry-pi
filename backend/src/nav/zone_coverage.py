"""Generate mission coverage waypoints from the persisted map configuration.

Shared by the jobs scheduler and the autonomy service so both derive identical
coverage paths from the operator's zones (boundaries + exclusions).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..models.mission import MissionWaypoint
from .coverage_planner import plan_coverage

# Nominal cutting swath (m) before overlap, and a safety cap on total waypoints.
BASE_SWATH_M = 0.6
MAX_WAYPOINTS = 2000


def _config_path(data_dir: str | None = None) -> Path:
    base = data_dir or os.getenv("LAWNBERRY_DATA_DIR", "./data")
    return Path(base) / "map_configuration_v2.json"


def _ring_latlng(zone: dict) -> list[tuple[float, float]]:
    """Convert a zone's GeoJSON ring ([lng, lat]) to (lat, lng) tuples."""
    coords = (zone.get("geometry") or {}).get("coordinates") or []
    ring = coords[0] if coords and isinstance(coords[0], list) else coords
    return [
        (float(p[1]), float(p[0])) for p in ring if isinstance(p, (list, tuple)) and len(p) >= 2
    ]


def waypoints_from_map_config(
    zone_ids: list[str] | None = None,
    *,
    overlap_factor: float = 0.1,
    speed: int = 50,
    data_dir: str | None = None,
) -> list[MissionWaypoint]:
    """Build serpentine coverage waypoints for the selected boundary zones.

    Matches ``zone_ids`` against the persisted map config (all boundary zones
    when ``zone_ids`` is empty/None), treats exclusion zones as holes, and
    concatenates coverage for each matching boundary. Returns ``[]`` when no
    geometry is available.
    """
    try:
        path = _config_path(data_dir)
        if not path.exists():
            return []
        config = json.loads(path.read_text())
        zones = config.get("zones") or []

        wanted = set(zone_ids or [])
        boundaries = [
            z
            for z in zones
            if z.get("zone_type") == "boundary" and (not wanted or z.get("zone_id") in wanted)
        ]
        exclusions = [
            ring
            for z in zones
            if z.get("zone_type") == "exclusion" and len(ring := _ring_latlng(z)) >= 3
        ]

        overlap = min(max(float(overlap_factor or 0.0), 0.0), 0.5)
        spacing = max(BASE_SWATH_M * (1.0 - overlap), 0.1)

        waypoints: list[MissionWaypoint] = []
        for zone in boundaries:
            ring = _ring_latlng(zone)
            if len(ring) < 3:
                continue
            coverage, _, _ = plan_coverage(
                ring, exclusion_polys=exclusions or None, spacing_m=spacing
            )
            for lat, lng in coverage:
                waypoints.append(MissionWaypoint(lat=lat, lon=lng, blade_on=True, speed=speed))
                if len(waypoints) >= MAX_WAYPOINTS:
                    return waypoints
        return waypoints
    except Exception:
        return []
