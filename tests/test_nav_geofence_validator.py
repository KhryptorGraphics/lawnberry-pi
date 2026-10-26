import math

import pytest

pytest.importorskip("shapely")

from backend.src.models import Geofence, LatLng
from backend.src.nav.geofence_validator import (
    GeofenceShape,
    build_shape,
    contains,
    path_within,
    segment_within,
)


def test_geofence_contains_and_boundary():
    # Square around origin ~10m radius
    square = [
        LatLng(latitude=0.0, longitude=0.0),
        LatLng(latitude=0.0, longitude=0.00009),
        LatLng(latitude=0.00009, longitude=0.00009),
        LatLng(latitude=0.00009, longitude=0.0),
    ]
    gf = Geofence(geofence_id="g1", boundary=square, buffer_distance_m=2.0)
    shape = build_shape(gf)

    inside = LatLng(latitude=0.000045, longitude=0.000045)
    outside = LatLng(latitude=0.0002, longitude=0.0002)

    assert contains(shape, inside, use_margin=False)
    assert not contains(shape, outside, use_margin=False)

    # The geofence's own outward buffer is not a licence to plan outside it: a point just
    # outside the boundary is outside, buffer or not.
    slightly_outside = LatLng(latitude=0.00009 + 1.0 / 111000.0, longitude=0.0)
    assert not contains(shape, slightly_outside)
    assert not contains(shape, slightly_outside, use_margin=False)


# ---- exclusion zones, inward margin and straight legs (Thor plan re-validation) ----

M = 111_000.0  # metres per degree of latitude in this module's flat-earth frame


def _xy(east: float, north: float) -> LatLng:
    return LatLng(latitude=north / M, longitude=east / M)  # at the equator one frame fits both


def _poly(*points: tuple[float, float]) -> list[LatLng]:
    return [_xy(e, n) for e, n in points]


SQUARE_100 = _poly((0, 0), (100, 0), (100, 100), (0, 100))
ZONE = _poly((40, 40), (60, 40), (60, 60), (40, 60))
# A 30 m wide U opening to the north: arms x 0..10 and x 20..30, notch x 10..20 above y = 10.
U_YARD = _poly((0, 0), (30, 0), (30, 30), (20, 30), (20, 10), (10, 10), (10, 30), (0, 30))


def _shape(**kw):
    return build_shape(Geofence(geofence_id="g", boundary=SQUARE_100), **kw)


def test_a_shape_built_the_old_way_is_unchanged_and_has_no_exclusions():
    shape = _shape()
    assert shape.no_go == ()
    assert contains(shape, _xy(50, 50)) and not contains(shape, _xy(150, 50))
    positional = GeofenceShape(shape.polygon, shape.enforced, shape.origin_lat, shape.origin_lon)
    assert positional.no_go == () and contains(positional, _xy(1, 1))


def test_inside_a_no_go_zone_or_on_its_edge_is_outside_the_fence():
    shape = _shape(no_go=[ZONE])
    assert contains(shape, _xy(10, 10))
    assert not contains(shape, _xy(50, 50)), "inside the zone"
    assert not contains(shape, _xy(40, 50)), "on the zone's edge counts as inside it"
    assert not contains(shape, _xy(60, 60)), "on the zone's corner too"
    assert not contains(shape, _xy(50, 50), use_margin=False), "the raw polygon excludes it as well"
    assert contains(shape, _xy(39.9, 50))


def test_the_inward_margin_pulls_the_enforced_boundary_in():
    shape = _shape(inset_m=5.0)
    assert contains(shape, _xy(10, 50)) and contains(shape, _xy(5.5, 50))
    assert not contains(shape, _xy(4, 50)), "inside the raw boundary, inside the margin"
    assert contains(shape, _xy(4, 50), use_margin=False), "the raw boundary is untouched"


def test_a_point_3_m_outside_a_fence_with_a_5_m_buffer_is_refused():
    """The review's repro (THOR-03): a 20 x 20 m yard whose geofence carries a 5 m buffer."""
    yard = Geofence(
        geofence_id="g", boundary=_poly((0, 0), (20, 0), (20, 20), (0, 20)), buffer_distance_m=5.0
    )
    shape = build_shape(yard)
    assert not contains(shape, _xy(-3, 10)), "3 m outside the boundary, inside the 5 m buffer"
    assert not contains(shape, _xy(23, 10)) and not contains(shape, _xy(10, -3))
    assert not contains(shape, _xy(10, 23))
    assert not path_within(shape, [_xy(10, 10), _xy(-3, 10)]), "nor may a plan leave through it"
    assert contains(shape, _xy(10, 10)) and contains(shape, _xy(0.5, 10))


def test_the_geofences_own_outward_buffer_never_changes_what_is_enforced():
    plain = build_shape(Geofence(geofence_id="g", boundary=SQUARE_100), inset_m=2.0)
    buffered = build_shape(
        Geofence(geofence_id="g", boundary=SQUARE_100, buffer_distance_m=2.0), inset_m=2.0
    )
    assert buffered.enforced.equals(plain.enforced)
    # a 2 m margin means 2 m: the buffer does not cancel it
    assert not contains(buffered, _xy(1, 50)) and contains(buffered, _xy(3, 50))
    assert not contains(buffered, _xy(-1, 50))
    assert not contains(build_shape(buffered_geofence(6.0), inset_m=6.0), _xy(3, 50))


def buffered_geofence(buffer_m: float) -> Geofence:
    return Geofence(geofence_id="g", boundary=SQUARE_100, buffer_distance_m=buffer_m)


def test_a_margin_that_leaves_nothing_is_an_error_and_a_negative_one_too():
    with pytest.raises(ValueError, match="no area"):
        _shape(inset_m=60.0)
    with pytest.raises(ValueError, match="inset_m"):
        _shape(inset_m=-1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_margin_that_is_not_a_finite_number_is_an_error(bad):
    with pytest.raises(ValueError, match="inset_m"):
        _shape(inset_m=bad)


def test_an_exclusion_zone_needs_three_vertices():
    with pytest.raises(ValueError):
        _shape(no_go=[_poly((10, 10), (20, 20))])


def test_a_straight_leg_must_stay_inside_the_fence_and_clear_of_every_zone():
    shape = _shape(no_go=[ZONE])
    assert segment_within(shape, _xy(10, 10), _xy(90, 10))
    assert not segment_within(shape, _xy(10, 50), _xy(90, 50)), "straight through the zone"
    assert not segment_within(shape, _xy(10, 40), _xy(90, 40)), "along the zone's edge"
    assert not segment_within(shape, _xy(10, 10), _xy(150, 10)), "leaves the fence"
    assert not segment_within(shape, _xy(150, 10), _xy(90, 10)), "starts outside the fence"


def test_two_points_inside_a_concave_fence_do_not_make_a_legal_leg():
    l_shape = Geofence(
        geofence_id="L",
        boundary=_poly((0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)),
    )
    shape = build_shape(l_shape)
    a, b = _xy(10, 90), _xy(90, 10)
    assert contains(shape, a) and contains(shape, b), "both ends are inside"
    assert not segment_within(shape, a, b), "but the line between them cuts the notch"
    assert segment_within(shape, _xy(10, 90), _xy(10, 10))
    assert segment_within(shape, _xy(10, 10), _xy(90, 10))


def test_a_leg_of_zero_length_is_just_a_point_check():
    shape = _shape(no_go=[ZONE])
    assert segment_within(shape, _xy(10, 10), _xy(10, 10))
    assert not segment_within(shape, _xy(50, 50), _xy(50, 50))
    assert not segment_within(shape, _xy(500, 50), _xy(500, 50))


def test_the_raw_polygon_can_be_asked_for_instead_of_the_enforced_one():
    shape = _shape(inset_m=10.0)
    assert not segment_within(shape, _xy(4, 50), _xy(4, 80))
    assert segment_within(shape, _xy(4, 50), _xy(4, 80), use_margin=False)


# ---- whole paths: every waypoint and every leg between them -------------------------------


def test_a_leg_across_the_notch_of_a_u_shaped_yard_is_refused_though_both_ends_are_inside():
    shape = build_shape(Geofence(geofence_id="U", boundary=U_YARD))
    left_arm, right_arm = _xy(5, 25), _xy(25, 25)
    assert contains(shape, left_arm) and contains(shape, right_arm), "both waypoints are inside"
    assert not segment_within(shape, left_arm, right_arm), "the leg between them crosses the notch"
    assert not path_within(shape, [left_arm, right_arm])
    # the same two points are fine when the plan goes round the bottom of the U
    assert path_within(shape, [left_arm, _xy(5, 5), _xy(25, 5), right_arm])


def test_a_path_is_refused_for_any_one_bad_waypoint_or_leg():
    shape = _shape(no_go=[ZONE], inset_m=1.0)
    good = [_xy(10, 10), _xy(90, 10), _xy(90, 90)]
    assert path_within(shape, good)
    assert not path_within(shape, [*good, _xy(150, 90)]), "last waypoint outside the fence"
    assert not path_within(shape, [_xy(50, 50), *good]), "first waypoint inside the no-go zone"
    assert not path_within(shape, [_xy(10, 50), _xy(90, 50)]), "leg through the no-go zone"
    assert not path_within(shape, [_xy(0.5, 10), _xy(10, 10)]), "waypoint inside the margin band"
    assert not path_within(shape, [_xy(10, 0.5), _xy(10, 10)])
    # a leg that grazes the margin band is refused although both ends are clear of it
    shape_l = build_shape(
        Geofence(
            geofence_id="L",
            boundary=_poly((0, 0), (100, 0), (100, 30), (30, 30), (30, 100), (0, 100)),
        ),
        inset_m=2.0,
    )
    assert contains(shape_l, _xy(28, 40)) and contains(shape_l, _xy(40, 28))
    assert not path_within(shape_l, [_xy(28, 40), _xy(40, 28)]), "cuts the reflex corner"


def test_an_empty_or_single_point_path_is_the_point_check():
    shape = _shape(no_go=[ZONE])
    assert path_within(shape, [])
    assert path_within(shape, [_xy(10, 10)])
    assert not path_within(shape, [_xy(50, 50)])
    assert not path_within(shape, [_xy(500, 50)])
    assert not path_within(shape, [_xy(10, 10), _xy(10, 10), _xy(500, 50)])


def test_the_margin_band_applies_to_legs_and_not_only_to_waypoints():
    shape = _shape(inset_m=5.0)
    assert segment_within(shape, _xy(10, 10), _xy(10, 90))
    assert not segment_within(shape, _xy(3, 10), _xy(3, 90)), "runs inside the 5 m band"
    assert segment_within(shape, _xy(3, 10), _xy(3, 90), use_margin=False)
