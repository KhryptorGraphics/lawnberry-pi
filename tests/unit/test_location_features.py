"""Unit tests for runtime location/coverage feature engineering.

Mirrors the training-side feature builder (mower/models/vla/features.py); these
tests pin the contract the model was trained against.
"""

from backend.src.nav.location_features import (
    COVERAGE_GRID_SIZE,
    GPS_FEATURE_DIM,
    CoverageMap,
    gps_feature_vector,
    latlon_to_enu,
)


def test_enu_signs():
    east, north = latlon_to_enu(40.001, -83.001, 40.0, -83.0)
    assert north > 0  # north of the datum
    assert east < 0  # more-negative longitude is west of the datum


def test_gps_feature_vector_dim_and_values():
    datum = (40.0, -83.0)
    v = gps_feature_vector(
        {
            "latitude": 40.0,
            "longitude": -83.0,
            "heading": 90.0,
            "speed_mps": 1.0,
            "rtk_fix_type": "fixed",
            "hdop": 0.0,
            "num_satellites": 16,
        },
        datum,
    )
    assert v.shape == (GPS_FEATURE_DIM,)
    assert abs(v[0]) < 1e-6 and abs(v[1]) < 1e-6  # at datum -> zero position
    assert abs(v[2] - 1.0) < 1e-6 and abs(v[3]) < 1e-6  # heading 90deg -> sin=1, cos=0
    assert v[5] == 1.0  # 'fixed' fix quality
    assert v[6] == 1.0  # hdop 0 -> full confidence
    assert v[7] == 1.0  # 16/16 satellites


def test_gps_feature_vector_accepts_object():
    class _G:
        latitude = 40.0
        longitude = -83.0
        heading = 0.0
        speed_mps = 0.0
        rtk_fix_type = "none"
        hdop = 99.0
        num_satellites = 0

    v = gps_feature_vector(_G(), (40.0, -83.0))
    assert v.shape == (GPS_FEATURE_DIM,)
    assert v[5] == 0.0  # no fix


def test_coverage_map_causal_growth():
    cov = CoverageMap((40.0, -83.0))
    assert cov.snapshot().sum() == 0.0  # nothing mowed yet
    cov.mark(40.0, -83.0)
    snap = cov.snapshot()
    assert snap.sum() > 0.0  # cell stamped
    assert snap.shape == (COVERAGE_GRID_SIZE, COVERAGE_GRID_SIZE)
    # snapshot must be a defensive copy
    snap[:] = 9.0
    assert cov.snapshot().sum() < snap.sum()


def test_coverage_out_of_bounds_is_ignored():
    cov = CoverageMap((40.0, -83.0), yard_half_size_m=5.0)
    cov.mark(41.0, -84.0)  # far outside the 5m grid
    assert cov.snapshot().sum() == 0.0
