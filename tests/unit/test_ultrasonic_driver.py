"""Unit tests for UltrasonicDriver (BEAD-041)."""
import os
import pytest

# Force simulation mode for tests
os.environ["SIM_MODE"] = "1"

from backend.src.drivers.sensors.ultrasonic_driver import UltrasonicDriver, UltrasonicReading


@pytest.fixture
def driver():
    """Create an ultrasonic driver instance."""
    return UltrasonicDriver()


@pytest.mark.asyncio
async def test_driver_initialization(driver):
    """Test driver initializes correctly in simulation mode."""
    await driver.initialize()
    assert driver.initialized is True


@pytest.mark.asyncio
async def test_driver_start_stop(driver):
    """Test driver start and stop lifecycle."""
    await driver.initialize()
    await driver.start()
    assert driver.running is True

    await driver.stop()
    assert driver.running is False


@pytest.mark.asyncio
async def test_read_single_sensor(driver):
    """Test reading from a single sensor."""
    await driver.initialize()
    await driver.start()

    reading = await driver.read_distance("front_center")

    assert isinstance(reading, UltrasonicReading)
    assert reading.sensor_id == "front_center"
    assert reading.timestamp is not None
    # In simulation, readings should be valid most of the time
    if reading.valid:
        assert 2.0 <= reading.distance_cm <= 400.0


@pytest.mark.asyncio
async def test_read_all_sensors(driver):
    """Test reading from all three sensors."""
    await driver.initialize()
    await driver.start()

    readings = await driver.read_all()

    assert len(readings) == 3
    sensor_ids = {r.sensor_id for r in readings}
    assert sensor_ids == {"front_left", "front_center", "front_right"}


@pytest.mark.asyncio
async def test_invalid_sensor_id(driver):
    """Test reading from an invalid sensor ID."""
    await driver.initialize()
    await driver.start()

    reading = await driver.read_distance("invalid_sensor")

    assert reading.valid is False
    assert reading.distance_cm == 0.0


@pytest.mark.asyncio
async def test_health_check(driver):
    """Test health check returns expected fields."""
    await driver.initialize()
    await driver.start()

    # Get some readings first
    await driver.read_all()

    health = await driver.health_check()

    assert health["sensor"] == "ultrasonic_array"
    assert health["sensor_count"] == 3
    assert health["initialized"] is True
    assert health["running"] is True
    assert health["simulation"] is True
    assert "front_left" in health["sensors"]
    assert "front_center" in health["sensors"]
    assert "front_right" in health["sensors"]


@pytest.mark.asyncio
async def test_get_minimum_distance(driver):
    """Test getting minimum distance from all sensors."""
    await driver.initialize()
    await driver.start()

    # Read all sensors to populate last_readings
    await driver.read_all()

    min_dist = driver.get_minimum_distance()

    # Should have a valid minimum distance in simulation mode
    assert min_dist is not None or min_dist == 0.0  # Could be None if all invalid
    if min_dist is not None:
        assert 2.0 <= min_dist <= 400.0


@pytest.mark.asyncio
async def test_multiple_read_cycles(driver):
    """Test multiple read cycles produce varying results (simulation noise)."""
    await driver.initialize()
    await driver.start()

    readings = []
    for _ in range(10):
        reading = await driver.read_distance("front_center")
        if reading.valid:
            readings.append(reading.distance_cm)

    # Should have some variation due to simulated noise
    if len(readings) >= 2:
        assert max(readings) != min(readings), "Readings should vary due to noise"


@pytest.mark.asyncio
async def test_driver_not_initialized_returns_invalid(driver):
    """Test that reading before initialization returns invalid reading."""
    # Don't initialize
    reading = await driver.read_distance("front_center")

    assert reading.valid is False


@pytest.mark.asyncio
async def test_reading_dataclass_fields():
    """Test UltrasonicReading dataclass has all expected fields."""
    reading = UltrasonicReading(
        sensor_id="test",
        distance_cm=100.0,
        valid=True,
    )

    assert reading.sensor_id == "test"
    assert reading.distance_cm == 100.0
    assert reading.valid is True
    assert reading.timestamp is not None
