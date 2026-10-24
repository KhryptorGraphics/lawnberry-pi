"""Geofence validation and enforcement helpers.

- Validates polygons using shapely
- Supports buffer zones (positive -> expansion, negative -> contraction)
- Supports exclusion ("no-go") polygons inside the fence and an inward ``inset_m`` margin
- Provides point-in-geofence checks, and straight-segment checks for plan legs
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from shapely.geometry import Polygon  # type: ignore

from ..models import Geofence, LatLng


@dataclass(frozen=True)
class GeofenceShape:
    """A fence in one local metric frame.

    ``buffered`` is the polygon enforced by default (``contains``/``segment_within``):
    the boundary grown by ``Geofence.buffer_distance_m`` or shrunk by ``inset_m``.
    ``no_go`` are exclusion polygons in the same frame; nothing may be inside or touch them.
    """

    polygon: Polygon
    buffered: Polygon
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
    """Create a shapely polygon for the geofence and its buffered version.

    ``no_go`` adds exclusion polygons (each >= 3 points). ``inset_m`` (>= 0) pulls the
    enforced boundary inwards as a safety margin; it is netted against the geofence's
    own ``buffer_distance_m``. A margin that leaves no area is an error, never a silent
    fall-back to the unshrunk boundary.
    """
    if inset_m < 0:
        raise ValueError("inset_m must be >= 0")
    base, olat, olon = _polygon_from_latlngs(geofence.boundary)
    buffered = base
    net = (geofence.buffer_distance_m or 0.0) - inset_m
    if net != 0.0:
        buffered = base.buffer(net)
        if buffered.is_empty:
            if inset_m > 0:
                raise ValueError(f"inset_m={inset_m} leaves no area inside the geofence")
            buffered = base
    zones = tuple(_polygon_from_latlngs(z, (olat, olon))[0] for z in (no_go or ()))
    return GeofenceShape(
        polygon=base, buffered=buffered, origin_lat=olat, origin_lon=olon, no_go=zones
    )


def contains(shape: GeofenceShape, point: LatLng, use_buffer: bool = True) -> bool:
    """True if ``point`` is inside the (buffered) fence and outside every no-go zone."""
    from shapely.geometry import Point as SPoint  # type: ignore

    x, y = _to_xy(point.latitude, point.longitude, shape.origin_lat, shape.origin_lon)
    poly = shape.buffered if use_buffer else shape.polygon
    pt = SPoint(x, y)
    return poly.covers(pt) and not any(z.intersects(pt) for z in shape.no_go)


def segment_within(shape: GeofenceShape, a: LatLng, b: LatLng, use_buffer: bool = True) -> bool:
    """True if the straight leg ``a`` -> ``b`` stays inside the fence and clear of every no-go zone.

    A waypoint check alone is not enough: with a concave fence or an exclusion zone, two
    inside points can still be joined by a line that leaves the fence or crosses a zone.
    """
    from shapely.geometry import LineString  # type: ignore

    ax, ay = _to_xy(a.latitude, a.longitude, shape.origin_lat, shape.origin_lon)
    bx, by = _to_xy(b.latitude, b.longitude, shape.origin_lat, shape.origin_lon)
    leg = LineString([(ax, ay), (bx, by)]) if (ax, ay) != (bx, by) else None
    poly = shape.buffered if use_buffer else shape.polygon
    if leg is None:
        return contains(shape, a, use_buffer=use_buffer)
    return poly.covers(leg) and not any(z.intersects(leg) for z in shape.no_go)


__all__ = ["GeofenceShape", "build_shape", "contains", "segment_within"]
