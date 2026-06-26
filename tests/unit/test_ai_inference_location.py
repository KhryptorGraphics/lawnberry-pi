"""Tests that AI inference preprocessing emits location + coverage inputs."""

import numpy as np

from backend.src.models.mower_data_frame import (
    GPSData,
    IMUData,
    MowerDataFrame,
    RTKFixType,
    UltrasonicData,
)
from backend.src.nav.location_features import (
    COVERAGE_GRID_SIZE,
    GPS_FEATURE_DIM,
    gps_feature_vector,
)
from backend.src.services.ai_inference_service import AIInferenceService

EXPECTED_SENSOR_DIM = GPS_FEATURE_DIM + 9 + 3  # gps + imu + ultrasonic = 20


def _frame() -> MowerDataFrame:
    f = MowerDataFrame()
    f.gps = GPSData(
        latitude=40.0001,
        longitude=-83.0001,
        heading=45.0,
        speed_mps=0.5,
        rtk_fix_type=RTKFixType.FIXED,
        hdop=0.8,
        num_satellites=20,
    )
    f.imu = IMUData(roll=10.0, pitch=-5.0, yaw=90.0)
    f.ultrasonic = UltrasonicData(front_left_cm=100.0, front_center_cm=150.0, front_right_cm=120.0)
    f.pi_camera_rgb = (np.random.rand(72, 128, 3) * 255).astype(np.uint8)
    return f


def test_preprocess_shapes_and_location():
    svc = AIInferenceService()
    datum = (40.0, -83.0)
    frame = _frame()

    inputs = svc._preprocess(frame, coverage_grid=None, datum=datum)

    assert inputs["image"].shape == (1, 3, 224, 224)
    assert inputs["sensors"].shape == (1, EXPECTED_SENSOR_DIM)
    assert inputs["coverage_map"].shape == (1, 1, COVERAGE_GRID_SIZE, COVERAGE_GRID_SIZE)

    # The first 8 sensor dims must be the local-ENU GPS feature vector.
    expected = gps_feature_vector(frame.gps, datum)
    assert np.allclose(inputs["sensors"][0, :GPS_FEATURE_DIM], expected, atol=1e-5)


def test_preprocess_coverage_passthrough():
    svc = AIInferenceService()
    grid = np.ones((COVERAGE_GRID_SIZE, COVERAGE_GRID_SIZE), dtype=np.float32)
    inputs = svc._preprocess(_frame(), coverage_grid=grid, datum=(40.0, -83.0))
    assert inputs["coverage_map"].sum() == COVERAGE_GRID_SIZE * COVERAGE_GRID_SIZE


def test_preprocess_default_datum_self_relative():
    """With no datum, position features are ~zero but motion/fix stay meaningful."""
    svc = AIInferenceService()
    inputs = svc._preprocess(_frame(), coverage_grid=None, datum=None)
    sensors = inputs["sensors"][0]
    assert abs(sensors[0]) < 1e-5 and abs(sensors[1]) < 1e-5  # self-relative position
    assert sensors[5] == 1.0  # fixed RTK still reported
