"""Geofence validation and enforcement helpers.

- Validates polygons using shapely
- Enforces the boundary pulled *inward* by a safety margin (``inset_m``); the fence is never grown
- Supports exclusion ("no-go") polygons inside the fence
- Provides point-in-geofence checks, straight-segment checks and whole-path checks for plan legs
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from shapely.geometry import Polygon  # type: ignore

from ..models import Geofence, LatLng


@dataclass(frozen=True)
class GeofenceShape:
    """A fence in one local metric frame.

    ``polygon`` is the boundary as given. ``enforced`` is the polygon every point and leg must
    stay inside: the boundary pulled in by ``inset_m`` and never larger than it, so no waypoint
    outside the boundary can pass. ``no_go`` are exclusion polygons in the same frame; nothing may
    be inside or touch them.
    """

    polygon: Polygon
    enforced: Polygon
    origin_lat: float
    origin_lon: float
    no_go: tuple[Polygon, ...] = ()


def _deg_lat_m() -> float:
    return 111_000.0


def _deg_lon_m_at_lat(lat: float) -> float:
    from math import cos, radians

    return 111_000.0 * cos(radians(lat))


def _to_xy(lat: float, lon: float, olat: float, olon: float) -> tuple[float, float]:
    return ((lon - olon) * _deg_lon_m_at_lat(olat), (lat - olat) * _deg_lat_m())


def _to_ll(x: float, y: float, olat: float, olon: float) -> tuple[float, float]:
    return (olat + y / _deg_lat_m(), olon + x / _deg_lon_m_at_lat(olat))


def _polygon_from_latlngs(points: Sequence[LatLng], origin: tuple[float, float] | None = None):
    from shapely.geometry import Polygon  # type: ignore

    if len(points) < 3:
        raise ValueError("Geofence must have >= 3 points")
    olat, olon = origin if origin is not None else (points[0].latitude, points[0].longitude)
    xy = [_to_xy(p.latitude, p.longitude, olat, olon) for p in points]
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area <= 0:
        raise ValueError("Invalid geofence polygon")
    return poly, olat, olon


def build_shape(
    geofence: Geofence,
    *,
    no_go: Sequence[Sequence[LatLng]] | None = None,
    inset_m: float = 0.0,
) -> GeofenceShape:
    """Create the shapely polygons for the geofence, its inward-margin version and its no-go zones.

    ``inset_m`` (finite, >= 0) pulls the enforced boundary inwards as a safety margin; a margin
    that leaves no area is an error, never a silent fall-back to the unshrunk boundary.
    ``no_go`` adds exclusion polygons (each >= 3 points).

    ``geofence.buffer_distance_m`` is deliberately not applied. It grows the boundary outwards, a
    tolerance for deciding that something has *violated* the fence; used here it would let a plan
    leave the fence by that much (a 5 m buffer accepted a waypoint 3 m outside). Only ``inset_m``
    changes what is enforced, and only inwards.
    """
    if not math.isfinite(inset_m) or inset_m < 0:
        raise ValueError("inset_m must be a finite number >= 0")
    base, olat, olon = _polygon_from_latlngs(geofence.boundary)
    enforced = base.buffer(-inset_m) if inset_m else base
    if enforced.is_empty:
        raise ValueError(f"inset_m={inset_m} leaves no area inside the geofence")
    zones = tuple(_polygon_from_latlngs(z, (olat, olon))[0] for z in (no_go or ()))
    return GeofenceShape(
        polygon=base, enforced=enforced, origin_lat=olat, origin_lon=olon, no_go=zones
    )


def contains(shape: GeofenceShape, point: LatLng, use_margin: bool = True) -> bool:
    """True if ``point`` is inside the fence (inset by its margin) and outside every no-go zone.

    ``use_margin=False`` tests the boundary as given instead of the margin-reduced one.
    """
    from shapely.geometry import Point as SPoint  # type: ignore

    x, y = _to_xy(point.latitude, point.longitude, shape.origin_lat, shape.origin_lon)
    poly = shape.enforced if use_margin else shape.polygon
    pt = SPoint(x, y)
    return poly.covers(pt) and not any(z.intersects(pt) for z in shape.no_go)


def segment_within(shape: GeofenceShape, a: LatLng, b: LatLng, use_margin: bool = True) -> bool:
    """True if the straight leg ``a`` -> ``b`` stays inside the fence and clear of every no-go zone.

    A waypoint check alone is not enough: with a concave fence or an exclusion zone, two
    inside points can still be joined by a line that leaves the fence or crosses a zone.
    """
    from shapely.geometry import LineString  # type: ignore

    ax, ay = _to_xy(a.latitude, a.longitude, shape.origin_lat, shape.origin_lon)
    bx, by = _to_xy(b.latitude, b.longitude, shape.origin_lat, shape.origin_lon)
    leg = LineString([(ax, ay), (bx, by)]) if (ax, ay) != (bx, by) else None
    poly = shape.enforced if use_margin else shape.polygon
    if leg is None:
        return contains(shape, a, use_margin=use_margin)
    return poly.covers(leg) and not any(z.intersects(leg) for z in shape.no_go)


def path_within(shape: GeofenceShape, points: Sequence[LatLng], use_margin: bool = True) -> bool:
    """True if every point of ``points`` and every straight leg between consecutive ones is allowed.

    This is the check for a plan window: each waypoint inside the fence and each leg joining two
    of them inside it too and clear of the no-go zones. An empty path is trivially allowed; the
    leg from the mower's own position to the first waypoint is not part of it (the follower checks
    that one against the live position).
    """
    if not all(contains(shape, p, use_margin=use_margin) for p in points):
        return False
    return all(
        segment_within(shape, a, b, use_margin=use_margin)
        for a, b in zip(points, points[1:], strict=False)
    )


__all__ = ["GeofenceShape", "build_shape", "contains", "path_within", "segment_within"]
