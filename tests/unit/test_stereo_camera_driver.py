"""Unit tests for stereo camera driver."""
from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_stereo_camera_driver_sim_mode_initializes():
    """Test that driver initializes in SIM_MODE."""
    os.environ["SIM_MODE"] = "1"

    from backend.src.drivers.sensors.stereo_camera_driver import StereoCameraDriver

    driver = StereoCameraDriver()
    await driver.initialize()

    assert driver.initialized is True


@pytest.mark.asyncio
async def test_stereo_camera_driver_sim_mode_captures_frame():
    """Test that driver captures frames in SIM_MODE."""
    os.environ["SIM_MODE"] = "1"

    from backend.src.drivers.sensors.stereo_camera_driver import StereoCameraDriver

    driver = StereoCameraDriver()
    await driver.initialize()
    await driver.start()

    frame = await driver.capture()

    assert frame is not None
    assert frame.combined.shape == (960, 2560, 3)
    assert frame.left.shape == (960, 1280, 3)
    assert frame.right.shape == (960, 1280, 3)
    assert frame.frame_id == 1

    await driver.stop()


@pytest.mark.asyncio
async def test_stereo_camera_driver_health_check():
    """Test health check returns expected fields."""
    os.environ["SIM_MODE"] = "1"

    from backend.src.drivers.sensors.stereo_camera_driver import StereoCameraDriver

    driver = StereoCameraDriver()
    await driver.initialize()
    await driver.start()

    # Capture a frame first
    await driver.capture()

    health = await driver.health_check()

    assert health["sensor"] == "stereo_camera"
    assert health["model"] == "ELP-USB960P2CAM-V90"
    assert health["initialized"] is True
    assert health["running"] is True
    assert health["simulation"] is True
    assert health["frame_count"] == 1
    assert health["last_capture_age_s"] is not None

    await driver.stop()


@pytest.mark.asyncio
async def test_stereo_camera_driver_multiple_captures():
    """Test multiple frame captures increment frame count."""
    os.environ["SIM_MODE"] = "1"

    from backend.src.drivers.sensors.stereo_camera_driver import StereoCameraDriver

    driver = StereoCameraDriver()
    await driver.initialize()
    await driver.start()

    for i in range(5):
        frame = await driver.capture()
        assert frame.frame_id == i + 1

    assert driver.frame_count == 5

    await driver.stop()
