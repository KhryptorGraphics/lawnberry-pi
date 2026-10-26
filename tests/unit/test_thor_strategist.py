"""W6 strategist: in-sim coverage, no-go compliance, obstacle replan, resume."""

import math

from backend.src.models.thor_link import MAX_WAYPOINTS, Detection, Uplink
from backend.src.nav.thor_strategist import StrategyConfig, ThorStrategist

LAT0, LON0 = 40.0, -75.0
K = 111_320.0 * math.cos(math.radians(LAT0))


def ll(x: float, y: float) -> tuple[float, float]:
    return (LAT0 + y / 111_320.0, LON0 + x / K)


YARD = [ll(0, 0), ll(60, 0), ll(60, 40), ll(0, 40)]
BED = [ll(25, 15), ll(35, 15), ll(35, 25), ll(25, 25)]  # flower bed no-go


def _fixed(s: ThorStrategist, idx: int, *, seq: int | None = None, sid: int = 4242, **kw) -> Uplink:
    """An uplink from the mower at plan waypoint ``idx``.

    ``seq`` defaults to a counter that never equals ``idx`` (nor ``next_index``): a strategist
    that echoed its own progress instead of the uplink's ``seq`` would otherwise go unnoticed.
    """
    leg = s.legs[min(idx, len(s.legs) - 1)]
    lat, lon = s.frame.ll(leg.x, leg.y)
    return Uplink(
        sid=sid,
        seq=1000 + 7 * idx if seq is None else seq,
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


def test_every_answer_echoes_the_uplink_it_answers():
    """The Pi acts only on a downlink echoing its uplink's sid/seq; that includes every pause."""
    s = ThorStrategist(YARD, [BED])
    cases = {
        "window": (_fixed(s, 3), False),
        "no rtk fix": (_fixed(s, 4).model_copy(update={"fix": "rtk_float"}), True),
        "estop": (_fixed(s, 5, estop=True), True),
        "plan finished": (_fixed(s, len(s.legs)), True),
    }
    for name, (up, expect_pause) in cases.items():
        up = up.model_copy(update={"sid": 777_000 + up.seq})
        down = s.on_uplink(up)
        assert (down.sid, down.ack) == (up.sid, up.seq), name
        assert down.pause is expect_pause, name


def test_the_ack_is_the_uplinks_seq_and_never_the_plans_progress():
    """The mutant that answers ``ack=next_index`` (or ``start_index``, or the reported waypoint
    index) passes any test whose uplink has ``seq == waypoint_index``. Here the three are
    unrelated, so each of those echoes is a different number from the right one."""
    s = ThorStrategist(YARD, [BED])
    for idx, seq, sid in ((3, 9_000_001, 31_337), (9, 5, 48), (0, 123_456_789, 2**47)):
        up = _fixed(s, idx, seq=seq, sid=sid)
        down = s.on_uplink(up)
        assert (down.sid, down.ack) == (sid, seq)
        assert down.ack not in (s.state.next_index, down.start_index, up.waypoint_index, up.t_ms)
        assert down.sid != down.ack
    # ... and the same holds for a pause, which is answered by a different code path
    paused = s.on_uplink(_fixed(s, 4, seq=77_777, sid=99, estop=True))
    assert paused.pause and (paused.sid, paused.ack) == (99, 77_777)


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


def _counting_replans(s: ThorStrategist) -> list:
    """Replace ``replan`` by a counter that keeps the plan: the (slow) planner is not under test."""
    calls: list = []
    planned = list(s.legs)

    def counting(start=None):
        calls.append(start)
        s.legs = list(planned)  # ``on_uplink`` truncated them just before calling replan

    s.replan = counting
    return calls


def test_the_same_obstacle_reported_on_every_uplink_is_one_keep_out_and_one_replan():
    """A visible obstacle is in every 10 Hz uplink: it used to add a keep-out (and replan against a
    growing union of circles) each time, for as long as it stayed in view."""
    s = ThorStrategist(YARD, [BED])
    replans = _counting_replans(s)
    toy = Detection(cls="toy", bearing_deg=0.0, range_m=3.0, conf=0.9)
    for i in range(300):
        down = s.on_uplink(_fixed(s, 4, seq=5000 + i, detections=[toy]))
        assert not down.pause
    assert len(s.state.obstacles_xy) == 1, "300 sightings of one obstacle"
    assert len(replans) == 1, "and one replan, not 300"
    # a different obstacle, more than a keep-out radius away, is a second one
    far = Detection(cls="rock", bearing_deg=90.0, range_m=6.0, conf=0.9)
    s.on_uplink(_fixed(s, 4, seq=9000, detections=[toy, far]))
    assert len(s.state.obstacles_xy) == 2 and len(replans) == 2


def test_the_number_of_remembered_obstacles_is_capped_and_the_oldest_go_first():
    s = ThorStrategist(YARD, [BED], config=StrategyConfig(max_obstacles=5))
    _counting_replans(s)
    ring = [
        Detection(cls="rock", bearing_deg=i * 18.0 - 171.0, range_m=4.0, conf=0.9)
        for i in range(20)  # twenty distinct obstacles, 1.25 m apart on a ring around the mower
    ]
    for i, det in enumerate(ring):
        s.on_uplink(_fixed(s, 4, seq=6000 + i, detections=[det]))
    assert len(s.state.obstacles_xy) == 5
    here = s.frame.xy(s.frame.ll(s.legs[4].x, s.legs[4].y))
    bearings = sorted(
        round(math.degrees(math.atan2(x - here[0], y - here[1]))) for x, y in s.state.obstacles_xy
    )
    assert bearings == sorted(round(d.bearing_deg) for d in ring[-5:]), "the five newest remain"
