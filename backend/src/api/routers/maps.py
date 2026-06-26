"""Map configuration endpoints (v2).

``GET/PUT /api/v2/map/configuration`` manage the operator's map setup: boundary
and exclusion zones, marker placement (home / AM-sun / PM-sun), and the active
map provider with OSM fallback metadata. Overlapping boundary/exclusion polygons
are rejected using shapely geometry checks.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from ..deps import require_operator_auth

router = APIRouter()

_CONFIG_FILE = "map_configuration_v2.json"


def _data_dir() -> Path:
    path = Path(os.getenv("LAWNBERRY_DATA_DIR", "./data"))
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def _default_config() -> dict[str, Any]:
    return {
        "zones": [],
        "provider": "google-maps",
        "updated_by": "system",
        "updated_at": datetime.now(UTC).isoformat(),
        "fallback": {"active": False, "reason": None, "provider": "google-maps"},
    }


def _load_config() -> dict[str, Any]:
    path = _data_dir() / _CONFIG_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return {**_default_config(), **data}
        except Exception:
            pass
    return _default_config()


def _save_config(config: dict[str, Any]) -> None:
    try:
        (_data_dir() / _CONFIG_FILE).write_text(json.dumps(config, indent=2, default=str))
    except Exception:
        pass


# --- Geometry overlap detection ---------------------------------------------


def _ring_from_coords(coords: Any) -> list[tuple[float, float]] | None:
    """Extract an outer polygon ring from GeoJSON or flat coordinate lists."""
    if not isinstance(coords, (list, tuple)) or not coords:
        return None
    first = coords[0]
    # GeoJSON Polygon: [[[x, y], ...]] -> take the outer ring.
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        ring = coords[0]
    else:
        ring = coords
    points: list[tuple[float, float]] = []
    for pt in ring:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                points.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                return None
    return points if len(points) >= 3 else None


def _labeled_polygons(payload: dict[str, Any]) -> list[tuple[str, list[tuple[float, float]]]]:
    labeled: list[tuple[str, list[tuple[float, float]]]] = []
    for zone in payload.get("zones") or []:
        if not isinstance(zone, dict):
            continue
        if zone.get("zone_type") in {"boundary", "exclusion"}:
            ring = _ring_from_coords((zone.get("geometry") or {}).get("coordinates"))
            if ring:
                labeled.append((str(zone.get("zone_id", "zone")), ring))
    for item in payload.get("boundaries") or []:
        if isinstance(item, dict):
            ring = _ring_from_coords(item.get("coordinates"))
            if ring:
                labeled.append((str(item.get("zone_type") or item.get("name") or "boundary"), ring))
    for item in payload.get("exclusion_zones") or []:
        if isinstance(item, dict):
            ring = _ring_from_coords(item.get("coordinates"))
            if ring:
                labeled.append((str(item.get("name") or "exclusion"), ring))
    return labeled


def _find_overlaps(payload: dict[str, Any]) -> list[str]:
    """Return the labels of polygons that overlap another polygon."""
    labeled = _labeled_polygons(payload)
    if len(labeled) < 2:
        return []
    try:
        from shapely.geometry import Polygon
    except Exception:
        return []

    polygons: list[tuple[str, Any]] = []
    for label, ring in labeled:
        try:
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons.append((label, poly))
        except Exception:
            continue

    conflicts: list[str] = []
    for (label_a, poly_a), (label_b, poly_b) in combinations(polygons, 2):
        try:
            if poly_a.intersects(poly_b) and poly_a.intersection(poly_b).area > 0:
                for label in (label_a, label_b):
                    if label not in conflicts:
                        conflicts.append(label)
        except Exception:
            continue
    return conflicts


# --- Endpoints ---------------------------------------------------------------


@router.get("/map/configuration")
async def get_map_configuration(
    simulate_fallback: str | None = Query(default=None),
) -> dict[str, Any]:
    config = _load_config()
    # Always surface a fallback envelope with the canonical keys.
    config.setdefault("fallback", {"active": False, "reason": None, "provider": config["provider"]})

    if simulate_fallback == "google_maps_unavailable":
        config = dict(config)
        config["provider"] = "osm"
        config["fallback"] = {
            "active": True,
            "reason": "GOOGLE_MAPS_UNAVAILABLE",
            "provider": "osm",
        }
    return config


@router.put("/map/configuration", dependencies=[Depends(require_operator_auth)])
async def put_map_configuration(payload: dict, request: Request) -> JSONResponse:
    conflicts = _find_overlaps(payload)
    if conflicts:
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "GEOMETRY_OVERLAP",
                "conflicts": conflicts,
                "detail": f"Geometry overlap detected between zones: {', '.join(conflicts)}",
            },
        )

    updated_at = datetime.now(UTC).isoformat()
    config = _default_config()
    # Preserve any recognised sections from the incoming payload.
    for key in ("zones", "markers", "boundaries", "exclusion_zones", "provider", "updated_by"):
        if key in payload:
            config[key] = payload[key]
    config["updated_at"] = updated_at
    config["fallback"] = {
        "active": False,
        "reason": None,
        "provider": config.get("provider", "google-maps"),
    }
    _save_config(config)
    return JSONResponse(status_code=200, content={"status": "accepted", "updated_at": updated_at})
