"""Thor-side coverage strategist (W6).

Turns a yard boundary plus no-go zones into an ordered waypoint plan,
tracks coverage progress from Pi uplinks, replans around reported
obstacles and persists state so an interrupted mow resumes where it
stopped. It decides *where* to go only: safety (stop, hold) stays on the
Pi, and a missing or stale strategist just leaves the Pi in safe hold.

Geometry is local metres (equirectangular about the boundary centroid),
adequate for a ~5 acre yard.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..models.thor_link import MAX_WAYPOINTS, Downlink, Uplink, Waypoint

LatLon = tuple[float, float]
_M_PER_DEG = 111_320.0


@dataclass(frozen=True)
class StrategyConfig:
    swath_m: float = 1.27  # 50 in deck
    overlap: float = 0.1
    heading_deg: float = 0.0
    # Keep the mower centre this far inside the boundary / away from no-go.
    edge_margin_m: float = 0.8
    obstacle_radius_m: float = 1.0  # keep-out around a reported obstacle
    min_obstacle_conf: float = 0.5
    require_fix: str = "rtk_fixed"


@dataclass
class _Frame:
    lat0: float
    lon0: float

    def xy(self, p: LatLon) -> tuple[float, float]:
        k = _M_PER_DEG * math.cos(math.radians(self.lat0))
        return ((p[1] - self.lon0) * k, (p[0] - self.lat0) * _M_PER_DEG)

    def ll(self, x: float, y: float) -> LatLon:
        k = _M_PER_DEG * math.cos(math.radians(self.lat0))
        return (self.lat0 + y / _M_PER_DEG, self.lon0 + x / k)


@dataclass
class PlanLeg:
    x: float
    y: float
    blade: bool


@dataclass
class CoverageState:
    plan_id: str
    next_index: int = 0  # first waypoint not yet reached
    obstacles_xy: list[tuple[float, float]] = field(default_factory=list)


class PlanError(RuntimeError):
    """No valid plan exists for the given geometry."""


class ThorStrategist:
    def __init__(
        self,
        boundary: list[LatLon],
        no_go: list[list[LatLon]] | None = None,
        config: StrategyConfig | None = None,
        plan_id: str = "plan",
    ) -> None:
        if len(boundary) < 3:
            raise PlanError("boundary needs >= 3 points")
        self.cfg = config or StrategyConfig()
        self.frame = _Frame(
            sum(p[0] for p in boundary) / len(boundary),
            sum(p[1] for p in boundary) / len(boundary),
        )
        self.boundary = Polygon([self.frame.xy(p) for p in boundary]).buffer(0)
        self.no_go = [Polygon([self.frame.xy(p) for p in z]).buffer(0) for z in (no_go or [])]
        self.state = CoverageState(plan_id=plan_id)
        self.legs: list[PlanLeg] = []
        self._seq = 0
        self.replan()

    # ---- geometry ---------------------------------------------------------
    def keep_out(self):
        """Everything the mower centre must never enter."""
        zones = [z.buffer(self.cfg.edge_margin_m) for z in self.no_go]
        zones += [
            Point(o).buffer(self.cfg.obstacle_radius_m + self.cfg.edge_margin_m)
            for o in self.state.obstacles_xy
        ]
        return unary_union(zones) if zones else None

    def free_area(self):
        area = self.boundary.buffer(-self.cfg.edge_margin_m)
        ko = self.keep_out()
        return area.difference(ko) if ko is not None else area

    def leg_is_legal(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        seg = LineString([a, b]) if a != b else Point(a)
        return self.free_area().buffer(1e-6).contains(seg)

    def _route(self, a, b) -> list[tuple[float, float]] | None:
        """Transit a->b inside the free area: direct, else along free-area vertices."""
        if self.leg_is_legal(a, b):
            return [b]
        free = self.free_area()
        geoms = list(getattr(free, "geoms", [free]))
        nodes = [a, b]
        for g in geoms:
            nodes += list(g.exterior.coords)[:-1]
            for hole in g.interiors:
                nodes += list(hole.coords)[:-1]
        # Dijkstra over the visibility graph of free-area vertices.
        n = len(nodes)
        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        done = [False] * n
        for _ in range(n):
            u = min((i for i in range(n) if not done[i]), key=lambda i: dist[i], default=-1)
            if u < 0 or dist[u] == math.inf:
                break
            done[u] = True
            if u == 1:
                break
            for v in range(n):
                if done[v] or not self.leg_is_legal(nodes[u], nodes[v]):
                    continue
                d = dist[u] + math.dist(nodes[u], nodes[v])
                if d < dist[v]:
                    dist[v], prev[v] = d, u
        if dist[1] == math.inf:
            return None
        path, i = [], 1
        while i != 0:
            path.append(nodes[i])
            i = prev[i]
        return path[::-1]

    # ---- planning ---------------------------------------------------------
    def _stripes(self, area) -> list[list[tuple[float, float]]]:
        from shapely.affinity import rotate

        rot = rotate(area, -self.cfg.heading_deg, origin=(0, 0))
        step = self.cfg.swath_m * (1.0 - self.cfg.overlap)
        minx, miny, maxx, maxy = rot.bounds
        rows, y, flip = [], miny + step / 2, False
        while y < maxy:
            cut = rot.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
            segs = [g for g in getattr(cut, "geoms", [cut]) if g.geom_type == "LineString"]
            segs.sort(key=lambda s: min(s.coords[0][0], s.coords[-1][0]), reverse=flip)
            for s in segs:
                (x0, _), (x1, _) = s.coords[0], s.coords[-1]
                lo, hi = sorted((x0, x1))
                pts = [(hi, y), (lo, y)] if flip else [(lo, y), (hi, y)]
                back = [rotate(Point(p), self.cfg.heading_deg, origin=(0, 0)) for p in pts]
                rows.append([(p.x, p.y) for p in back])
            flip = not flip
            y += step
        return rows

    def replan(self, start: tuple[float, float] | None = None) -> None:
        """(Re)build legs for the uncovered remainder, starting from ``start``."""
        area = self.free_area()
        if area.is_empty:
            raise PlanError("no free area inside boundary")
        done_legs = self.legs[: self.state.next_index]
        covered = self._covered_geom(done_legs)
        stripes = self._stripes(area)
        legs: list[PlanLeg] = list(done_legs)
        pos = start or ((done_legs[-1].x, done_legs[-1].y) if done_legs else None)
        for a, b in stripes:
            stripe = LineString([a, b])
            if covered is not None and covered.buffer(1e-3).contains(stripe):
                continue
            if pos is None:
                legs.append(PlanLeg(*a, blade=False))
            else:
                route = self._route(pos, a)
                if route is None:
                    continue  # unreachable island: reported via coverage_fraction
                legs += [PlanLeg(*p, blade=False) for p in route]
            legs.append(PlanLeg(*b, blade=True))
            pos = b
        self.legs = legs

    # ---- coverage accounting ---------------------------------------------
    def _covered_geom(self, legs: list[PlanLeg]):
        parts = [
            LineString([(p.x, p.y), (q.x, q.y)]).buffer(self.cfg.swath_m / 2, cap_style=2)
            for p, q in zip(legs, legs[1:], strict=False)
            if q.blade and (p.x, p.y) != (q.x, q.y)
        ]
        return unary_union(parts) if parts else None

    def coverage_fraction(self, upto: int | None = None) -> float:
        area = self.free_area()
        cov = self._covered_geom(self.legs[: (len(self.legs) if upto is None else upto)])
        return 0.0 if cov is None or area.area == 0 else cov.intersection(area).area / area.area

    def violations(self) -> list[int]:
        """Indices of legs leaving the free area (must be empty)."""
        return [
            i
            for i in range(1, len(self.legs))
            if not self.leg_is_legal(
                (self.legs[i - 1].x, self.legs[i - 1].y), (self.legs[i].x, self.legs[i].y)
            )
        ]

    # ---- link -------------------------------------------------------------
    def waypoints(self) -> list[Waypoint]:
        out = []
        for leg in self.legs:
            lat, lon = self.frame.ll(leg.x, leg.y)
            out.append(Waypoint(lat=lat, lon=lon, blade=leg.blade))
        return out

    def _down(self, pause: bool = False) -> Downlink:
        self._seq += 1
        i = self.state.next_index
        wps = [] if pause else self.waypoints()[i : i + MAX_WAYPOINTS]
        return Downlink(
            seq=self._seq, plan_id=self.state.plan_id, start_index=i, waypoints=wps, pause=pause
        )

    def on_uplink(self, up: Uplink) -> Downlink:
        """Advance progress, absorb obstacles, and answer with the next window."""
        if up.waypoint_index is not None:
            self.state.next_index = max(
                self.state.next_index, min(up.waypoint_index, len(self.legs))
            )
        if up.lat is None or up.lon is None or up.fix != self.cfg.require_fix:
            return self._down(pause=True)
        here = self.frame.xy((up.lat, up.lon))
        new = []
        for d in up.detections:
            if d.conf < self.cfg.min_obstacle_conf:
                continue
            heading = math.radians((up.heading_deg or 0.0) + d.bearing_deg)
            new.append(
                (here[0] + d.range_m * math.sin(heading), here[1] + d.range_m * math.cos(heading))
            )
        if new:
            self.state.obstacles_xy += new
            self.legs = self.legs[: self.state.next_index]
            self.replan(start=here)
        if up.estop or self.state.next_index >= len(self.legs):
            return self._down(pause=True)
        return self._down()

    # ---- resume -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "plan_id": self.state.plan_id,
                    "next_index": self.state.next_index,
                    "obstacles_xy": self.state.obstacles_xy,
                    "legs": [[leg.x, leg.y, leg.blade] for leg in self.legs],
                }
            )
        )

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.state = CoverageState(
            data["plan_id"], data["next_index"], [tuple(o) for o in data["obstacles_xy"]]
        )
        self.legs = [PlanLeg(x, y, b) for x, y, b in data["legs"]]
