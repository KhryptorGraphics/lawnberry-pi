"""W6 strategist: in-sim coverage, no-go compliance, obstacle replan, resume."""

import math

from backend.src.models.thor_link import MAX_WAYPOINTS, Detection, Uplink
from backend.src.nav.thor_strategist import ThorStrategist

LAT0, LON0 = 40.0, -75.0
K = 111_320.0 * math.cos(math.radians(LAT0))


def ll(x: float, y: float) -> tuple[float, float]:
    return (LAT0 + y / 111_320.0, LON0 + x / K)


YARD = [ll(0, 0), ll(60, 0), ll(60, 40), ll(0, 40)]
BED = [ll(25, 15), ll(35, 15), ll(35, 25), ll(25, 25)]  # flower bed no-go


def _fixed(s: ThorStrategist, idx: int, **kw) -> Uplink:
    leg = s.legs[min(idx, len(s.legs) - 1)]
    lat, lon = s.frame.ll(leg.x, leg.y)
    return Uplink(
        seq=idx,
        t_ms=idx,
        lat=lat,
        lon=lon,
        fix="rtk_fixed",
        heading_deg=0.0,
        waypoint_index=idx,
        **kw,
    )


def test_full_coverage_with_zero_violations():
    s = ThorStrategist(YARD, [BED])
    assert s.violations() == []
    assert s.coverage_fraction() > 0.97
    bed = s.no_go[0]
    for a, b in zip(s.legs, s.legs[1:], strict=False):
        from shapely.geometry import LineString

        assert not LineString([(a.x, a.y), (b.x, b.y)]).intersects(bed)


def test_reported_obstacle_is_avoided_by_replan():
    s = ThorStrategist(YARD, [BED])
    idx = 4
    down = s.on_uplink(
        _fixed(s, idx, detections=[Detection(cls="toy", bearing_deg=0.0, range_m=3.0, conf=0.9)])
    )
    assert not down.pause and down.start_index == idx
    assert s.state.obstacles_xy and s.violations() == []
    obs = s.state.obstacles_xy[0]
    from shapely.geometry import LineString, Point

    keep = Point(obs).buffer(s.cfg.obstacle_radius_m)
    for a, b in zip(s.legs[idx:], s.legs[idx + 1 :], strict=False):
        assert not LineString([(a.x, a.y), (b.x, b.y)]).intersects(keep)


def test_pause_without_rtk_fix_and_window_size():
    s = ThorStrategist(YARD, [BED])
    up = _fixed(s, 0).model_copy(update={"fix": "rtk_float"})
    assert s.on_uplink(up).pause
    down = s.on_uplink(_fixed(s, 2))
    assert len(down.waypoints) == MAX_WAYPOINTS and down.start_index == 2


def test_resume_after_interruption(tmp_path):
    s = ThorStrategist(YARD, [BED], plan_id="mow1")
    mid = len(s.legs) // 2
    s.on_uplink(_fixed(s, mid))
    s.save(tmp_path / "state.json")
    r = ThorStrategist(YARD, [BED], plan_id="other")
    r.load(tmp_path / "state.json")
    down = r.on_uplink(_fixed(r, mid))
    assert down.plan_id == "mow1" and down.start_index == mid
    assert r.waypoints()[mid] == s.waypoints()[mid]
    # progress never regresses on a stale report
    r.on_uplink(_fixed(r, mid - 3))
    assert r.state.next_index == mid
    assert r.coverage_fraction() > 0.97 and r.violations() == []
