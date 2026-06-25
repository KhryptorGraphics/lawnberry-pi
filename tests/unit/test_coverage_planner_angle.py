"""Tests for arbitrary-angle serpentine coverage planning."""

from backend.src.nav.coverage_planner import plan_coverage


def _rectangle(lat0=0.0, lng0=0.0, width_m=20.0, height_m=10.0):
    lat_step = height_m / 111320.0
    lng_step = width_m / 111320.0
    return [
        (lat0, lng0),
        (lat0, lng0 + lng_step),
        (lat0 + lat_step, lng0 + lng_step),
        (lat0 + lat_step, lng0),
    ]


def test_plan_coverage_zero_angle_baseline():
    boundary = _rectangle()
    path, rows, length = plan_coverage(boundary, spacing_m=1.0, angle_deg=0.0)
    assert rows > 0
    assert len(path) > 0
    assert length > 0


def test_plan_coverage_supports_nonzero_angle():
    boundary = _rectangle()
    # A 90-degree rotation should still produce a real coverage path
    # (previously this returned an empty path).
    path, rows, length = plan_coverage(boundary, spacing_m=1.0, angle_deg=90.0)
    assert rows > 0, "Angled coverage produced no rows"
    assert len(path) > 0, "Angled coverage produced no path"
    assert length > 0

    # Rotated passes should remain near the field footprint.
    lats = [p[0] for p in path]
    lngs = [p[1] for p in path]
    b_lats = [p[0] for p in boundary]
    b_lngs = [p[1] for p in boundary]
    span_lat = max(b_lats) - min(b_lats)
    span_lng = max(b_lngs) - min(b_lngs)
    assert min(lats) >= min(b_lats) - span_lat
    assert max(lats) <= max(b_lats) + span_lat
    assert min(lngs) >= min(b_lngs) - span_lng
    assert max(lngs) <= max(b_lngs) + span_lng


def test_plan_coverage_arbitrary_angle_runs():
    boundary = _rectangle()
    path, rows, _ = plan_coverage(boundary, spacing_m=1.0, angle_deg=37.0)
    assert rows > 0
    assert len(path) > 0
