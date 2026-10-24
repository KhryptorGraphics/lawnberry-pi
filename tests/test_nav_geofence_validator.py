import pytest

pytest.importorskip("shapely")

from backend.src.models import Geofence, LatLng
from backend.src.nav.geofence_validator import (
    GeofenceShape,
    build_shape,
    contains,
    segment_within,
)


def test_geofence_contains_and_buffer():
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

    assert contains(shape, inside, use_buffer=False)
    assert not contains(shape, outside, use_buffer=False)

    # With buffer, a point just outside original polygon should be contained
    slightly_outside = LatLng(latitude=0.00009 + 1.0 / 111000.0, longitude=0.0)
    assert contains(shape, slightly_outside, use_buffer=True)


# ---- exclusion zones, inward margin and straight legs (Thor plan re-validation) ----

M = 111_000.0  # metres per degree of latitude in this module's flat-earth frame


def _xy(east: float, north: float) -> LatLng:
    return LatLng(latitude=north / M, longitude=east / M)  # at the equator one frame fits both


def _poly(*points: tuple[float, float]) -> list[LatLng]:
    return [_xy(e, n) for e, n in points]


SQUARE_100 = _poly((0, 0), (100, 0), (100, 100), (0, 100))
ZONE = _poly((40, 40), (60, 40), (60, 60), (40, 60))


def _shape(**kw):
    return build_shape(Geofence(geofence_id="g", boundary=SQUARE_100), **kw)


def test_a_shape_built_the_old_way_is_unchanged_and_has_no_exclusions():
    shape = _shape()
    assert shape.no_go == ()
    assert contains(shape, _xy(50, 50)) and not contains(shape, _xy(150, 50))
    positional = GeofenceShape(shape.polygon, shape.buffered, shape.origin_lat, shape.origin_lon)
    assert positional.no_go == () and contains(positional, _xy(1, 1))


def test_inside_a_no_go_zone_or_on_its_edge_is_outside_the_fence():
    shape = _shape(no_go=[ZONE])
    assert contains(shape, _xy(10, 10))
    assert not contains(shape, _xy(50, 50)), "inside the zone"
    assert not contains(shape, _xy(40, 50)), "on the zone's edge counts as inside it"
    assert not contains(shape, _xy(60, 60)), "on the zone's corner too"
    assert not contains(shape, _xy(50, 50), use_buffer=False), "the raw polygon excludes it as well"
    assert contains(shape, _xy(39.9, 50))


def test_the_inward_margin_pulls_the_enforced_boundary_in():
    shape = _shape(inset_m=5.0)
    assert contains(shape, _xy(10, 50)) and contains(shape, _xy(5.5, 50))
    assert not contains(shape, _xy(4, 50)), "inside the raw boundary, inside the margin"
    assert contains(shape, _xy(4, 50), use_buffer=False), "the raw boundary is untouched"


def test_the_margin_is_netted_against_the_geofences_own_buffer():
    gf = Geofence(geofence_id="g", boundary=SQUARE_100, buffer_distance_m=2.0)
    shape = build_shape(gf, inset_m=2.0)
    assert contains(shape, _xy(1, 50)) and not contains(shape, _xy(-1, 50))
    assert not contains(build_shape(gf, inset_m=6.0), _xy(3, 50))


def test_a_margin_that_leaves_nothing_is_an_error_and_a_negative_one_too():
    with pytest.raises(ValueError, match="no area"):
        _shape(inset_m=60.0)
    with pytest.raises(ValueError, match="inset_m"):
        _shape(inset_m=-1.0)


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
    assert segment_within(shape, _xy(4, 50), _xy(4, 80), use_buffer=False)
