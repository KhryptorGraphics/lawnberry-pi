"""Unit tests for AIInferenceService.

Tests the AI inference service for autonomous mowing control.
All tests run in simulation mode without requiring hardware.
"""
import os
import asyncio
import pytest
import numpy as np
from datetime import datetime, timezone

# Force simulation mode for all tests
os.environ["SIM_MODE"] = "1"

from backend.src.services.ai_inference_service import (
    AIInferenceService,
    VLAModelConfig,
    get_ai_inference_service,
)
from backend.src.models.action_prediction import (
    ActionPrediction,
    InferenceMetrics,
    AIControlStatus,
    ActionConfidence,
    ControlMode,
)
from backend.src.models.mower_data_frame import (
    MowerDataFrame,
    GPSData,
    IMUData,
    UltrasonicData,
    ToFData,
    MotorState,
    ControlAction,
    PowerState,
    RTKFixType,
    MowerState,
)


@pytest.fixture
def service():
    """Create a fresh AIInferenceService instance for testing."""
    # Reset singleton
    import backend.src.services.ai_inference_service as module
    module._service_instance = None
    return AIInferenceService()


@pytest.fixture
def sample_frame():
    """Create a sample MowerDataFrame for testing."""
    return MowerDataFrame(
        timestamp=datetime.now(timezone.utc),
        session_id="test-session-001",
        frame_id=42,
        stereo_left=np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8),
        stereo_right=np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8),
        pi_camera_rgb=np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
        gps=GPSData(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=10.0,
            rtk_fix_type=RTKFixType.FIXED,
            hdop=0.8,
            num_satellites=14,
        ),
        imu=IMUData(
            roll=2.0,
            pitch=-1.5,
            yaw=45.0,
            linear_accel_x=0.1,
            linear_accel_y=0.0,
            linear_accel_z=9.8,
        ),
        ultrasonic=UltrasonicData(
            front_left_cm=150.0,
            front_center_cm=200.0,
            front_right_cm=175.0,
            min_distance_cm=150.0,
        ),
        tof=ToFData(left_mm=50.0, right_mm=52.0),
        motor=MotorState(
            wheel_speed_left=100.0,
            wheel_speed_right=100.0,
            blade_speed=3000.0,
            blade_enabled=True,
        ),
        action=ControlAction(steering=0.0, throttle=0.5, blade_command=True),
        power=PowerState(battery_voltage=25.2, battery_soc=85.0),
        mower_state=MowerState.MOWING,
    )


@pytest.fixture
def obstacle_frame():
    """Create a frame with a close obstacle."""
    frame = MowerDataFrame(
        timestamp=datetime.now(timezone.utc),
        session_id="test-obstacle-001",
        frame_id=100,
        stereo_left=np.zeros((960, 1280, 3), dtype=np.uint8),
        stereo_right=np.zeros((960, 1280, 3), dtype=np.uint8),
        ultrasonic=UltrasonicData(
            front_left_cm=25.0,  # Obstacle on left
            front_center_cm=100.0,
            front_right_cm=150.0,
            min_distance_cm=25.0,
        ),
    )
    return frame


class TestAIInferenceServiceInitialization:
    """Tests for service initialization."""

    @pytest.mark.asyncio
    async def test_initialize_in_simulation_mode(self, service):
        """Test that service initializes in simulation mode."""
        await service.initialize()

        assert service.initialized is True
        assert service._hailo is not None
        # In sim mode, Hailo hardware won't be available
        assert service._hailo.hardware_available is False

    @pytest.mark.asyncio
    async def test_start_and_stop(self, service):
        """Test service start and stop lifecycle."""
        await service.initialize()
        await service.start()

        assert service.running is True

        await service.stop()

        assert service.running is False
        assert service._enabled is False

    @pytest.mark.asyncio
    async def test_enable_disable(self, service):
        """Test enabling and disabling AI control."""
        await service.initialize()
        await service.start()

        # Enable
        success = await service.enable()
        assert success is True
        assert service._enabled is True
        assert service._control_mode == ControlMode.AI_AUTONOMOUS

        # Disable
        await service.disable()
        assert service._enabled is False
        assert service._control_mode == ControlMode.MANUAL

    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check returns expected structure."""
        await service.initialize()
        await service.start()

        health = await service.health_check()

        assert "service" in health
        assert health["service"] == "ai_inference"
        assert "initialized" in health
        assert health["initialized"] is True
        assert "running" in health
        assert health["running"] is True
        assert "hailo" in health
        assert "metrics" in health


class TestInference:
    """Tests for inference functionality."""

    @pytest.mark.asyncio
    async def test_inference_returns_valid_prediction(self, service, sample_frame):
        """Test that inference returns a valid ActionPrediction."""
        await service.initialize()
        await service.start()

        prediction = await service.infer(sample_frame)

        assert isinstance(prediction, ActionPrediction)
        assert -1.0 <= prediction.steering <= 1.0
        assert 0.0 <= prediction.throttle <= 1.0
        assert isinstance(prediction.blade, bool)
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.inference_time_ms > 0
        # frame_id is an internal counter that increments with each inference
        assert prediction.frame_id >= 0

    @pytest.mark.asyncio
    async def test_inference_with_obstacle_triggers_safety(self, service, obstacle_frame):
        """Test that close obstacles affect predictions."""
        await service.initialize()
        await service.start()

        prediction = await service.infer(obstacle_frame)

        # With obstacle at 25cm, expect reduced throttle or safety override
        assert prediction.obstacle_detected is True
        # Very close obstacle should trigger safety
        if obstacle_frame.ultrasonic.min_distance_cm < 30:
            assert prediction.safety_override is True or prediction.throttle < 0.3

    @pytest.mark.asyncio
    async def test_multiple_inferences_update_metrics(self, service, sample_frame):
        """Test that multiple inferences update metrics correctly."""
        await service.initialize()
        await service.start()

        # Run several inferences
        for i in range(5):
            sample_frame.frame_id = i
            await service.infer(sample_frame)

        metrics = service.get_metrics()

        assert metrics.total_inferences == 5
        assert metrics.successful_inferences == 5
        assert metrics.avg_inference_time_ms > 0
        assert metrics.avg_confidence > 0

    @pytest.mark.asyncio
    async def test_inference_requires_running_service(self, service, sample_frame):
        """Test that inference fails if service not running."""
        await service.initialize()
        # Don't start the service

        with pytest.raises(RuntimeError, match="not initialized or not running"):
            await service.infer(sample_frame)


class TestActionPrediction:
    """Tests for ActionPrediction data class."""

    def test_action_prediction_bounds(self):
        """Test that prediction values are clamped to valid bounds."""
        prediction = ActionPrediction(
            steering=2.0,  # Should be clamped to 1.0
            throttle=-0.5,  # Should be clamped to 0.0
            blade=True,
            confidence=1.5,  # Should be clamped to 1.0
        )

        assert prediction.steering == 1.0
        assert prediction.throttle == 0.0
        assert prediction.confidence == 1.0

    def test_confidence_levels(self):
        """Test confidence level categorization."""
        high = ActionPrediction(steering=0, throttle=0.5, blade=True, confidence=0.95)
        assert high.confidence_level == ActionConfidence.HIGH

        medium = ActionPrediction(steering=0, throttle=0.5, blade=True, confidence=0.75)
        assert medium.confidence_level == ActionConfidence.MEDIUM

        low = ActionPrediction(steering=0, throttle=0.5, blade=True, confidence=0.55)
        assert low.confidence_level == ActionConfidence.LOW

        uncertain = ActionPrediction(steering=0, throttle=0.5, blade=True, confidence=0.3)
        assert uncertain.confidence_level == ActionConfidence.UNCERTAIN

    def test_to_motor_commands(self):
        """Test conversion to motor commands."""
        # Straight ahead
        prediction = ActionPrediction(steering=0.0, throttle=0.8, blade=True, confidence=0.9)
        commands = prediction.to_motor_commands()

        assert commands["left_speed"] == pytest.approx(0.8, rel=0.01)
        assert commands["right_speed"] == pytest.approx(0.8, rel=0.01)
        assert commands["blade_enabled"] is True

    def test_to_motor_commands_turn_right(self):
        """Test motor commands for right turn."""
        prediction = ActionPrediction(steering=0.5, throttle=0.6, blade=False, confidence=0.85)
        commands = prediction.to_motor_commands()

        # Right turn: left wheel faster than right
        assert commands["left_speed"] > commands["right_speed"]
        assert commands["blade_enabled"] is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        prediction = ActionPrediction(
            steering=0.25,
            throttle=0.6,
            blade=True,
            confidence=0.88,
            inference_time_ms=15.5,
        )
        d = prediction.to_dict()

        assert d["steering"] == 0.25
        assert d["throttle"] == 0.6
        assert d["blade"] is True
        assert d["confidence"] == 0.88
        assert d["inference_time_ms"] == 15.5
        assert "confidence_level" in d


class TestInferenceMetrics:
    """Tests for InferenceMetrics tracking."""

    def test_update_with_prediction(self):
        """Test metrics update with predictions."""
        metrics = InferenceMetrics(model_name="test", target_fps=10.0)

        prediction = ActionPrediction(
            steering=0.0,
            throttle=0.5,
            blade=True,
            confidence=0.9,
            inference_time_ms=12.0,
            preprocessing_time_ms=2.0,
            postprocessing_time_ms=1.0,
        )

        metrics.update_with_prediction(prediction)

        assert metrics.total_inferences == 1
        assert metrics.successful_inferences == 1
        assert metrics.avg_inference_time_ms == 12.0
        assert metrics.high_confidence_count == 1

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = InferenceMetrics()

        # No inferences
        assert metrics.success_rate == 0.0

        # Add successful predictions
        for i in range(8):
            pred = ActionPrediction(steering=0, throttle=0.5, blade=True, confidence=0.8)
            pred.inference_time_ms = 10.0
            metrics.update_with_prediction(pred)

        # Add failed predictions
        for i in range(2):
            metrics.update_with_prediction(
                ActionPrediction(steering=0, throttle=0, blade=False, confidence=0.8),
                success=False
            )

        assert metrics.success_rate == pytest.approx(0.8, rel=0.01)


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_ai_inference_service_returns_same_instance(self):
        """Test that get_ai_inference_service returns the same instance."""
        # Reset singleton
        import backend.src.services.ai_inference_service as module
        module._service_instance = None

        service1 = get_ai_inference_service()
        service2 = get_ai_inference_service()

        assert service1 is service2


class TestPreprocessing:
    """Tests for input preprocessing."""

    @pytest.mark.asyncio
    async def test_preprocessing_handles_missing_images(self, service):
        """Test preprocessing with missing image data."""
        await service.initialize()

        # Frame with no images
        frame = MowerDataFrame(
            timestamp=datetime.now(timezone.utc),
            stereo_left=None,
            stereo_right=None,
            pi_camera_rgb=None,
            ultrasonic=UltrasonicData(min_distance_cm=100),
        )

        inputs = service._preprocess(frame)

        # Should create placeholder tensors
        assert "stereo" in inputs
        assert "rgb" in inputs
        assert "sensors" in inputs

    @pytest.mark.asyncio
    async def test_preprocessing_normalizes_sensor_values(self, service, sample_frame):
        """Test that sensor values are properly normalized."""
        await service.initialize()

        inputs = service._preprocess(sample_frame)

        # Check sensor features are normalized
        sensors = inputs["sensors"]
        assert sensors.shape[-1] == 9  # 9 sensor features
        # All values should be roughly in [-1, 1] range
        assert np.all(sensors >= -2.0)
        assert np.all(sensors <= 2.0)


class TestStatus:
    """Tests for status reporting."""

    @pytest.mark.asyncio
    async def test_get_status(self, service):
        """Test status reporting."""
        await service.initialize()
        await service.start()

        status = service.get_status()

        assert isinstance(status, AIControlStatus)
        assert status.enabled is False
        assert status.mode == ControlMode.MANUAL
        assert status.model_loaded is False  # No VLA model in tests

    @pytest.mark.asyncio
    async def test_status_after_inference(self, service, sample_frame):
        """Test status includes last prediction."""
        await service.initialize()
        await service.start()

        await service.infer(sample_frame)

        status = service.get_status()

        assert status.last_prediction is not None
        assert status.prediction_age_ms >= 0
