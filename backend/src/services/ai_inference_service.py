"""AI Inference Service for autonomous mowing control.

Provides real-time VLA (Vision-Language-Action) model inference using
the Hailo 8L accelerator. Takes multi-modal sensor data and predicts
steering, throttle, and blade actions.

Designed for seamless model deployment from Thor training server.
Supports both hardware inference and simulation mode for testing.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.simulation import is_simulation_mode
from ..drivers.ai.hailo_driver import HailoDriver
from ..drivers.base import HardwareDriver
from ..models.action_prediction import (
    ActionPrediction,
    AIControlStatus,
    ControlMode,
    InferenceMetrics,
)
from ..models.mower_data_frame import MowerDataFrame
from ..nav.location_features import (
    COVERAGE_GRID_SIZE,
    gps_feature_vector,
)


@dataclass
class VLAModelConfig:
    """Configuration for VLA model inference."""

    # Input specifications
    stereo_input_size: tuple = (256, 320)  # (H, W)
    rgb_input_size: tuple = (224, 224)
    normalize_images: bool = True
    image_mean: tuple = (0.485, 0.456, 0.406)  # ImageNet normalization
    image_std: tuple = (0.229, 0.224, 0.225)

    # Output specifications
    steering_bins: int = 21  # -1.0 to 1.0 in 0.1 steps
    throttle_bins: int = 11  # 0.0 to 1.0 in 0.1 steps

    # Safety thresholds
    min_confidence_threshold: float = 0.5
    obstacle_confidence_threshold: float = 0.7

    # Performance
    target_fps: float = 10.0
    max_inference_time_ms: float = 100.0


# Singleton instance holder
_service_instance: AIInferenceService | None = None


def get_ai_inference_service() -> AIInferenceService:
    """Get or create the singleton AIInferenceService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AIInferenceService()
    return _service_instance


class AIInferenceService(HardwareDriver):
    """AI inference service for autonomous mowing control.

    Uses the Hailo 8L accelerator to run VLA model inference on
    multi-modal sensor data and produce control actions.

    Features:
    - Async lifecycle management (HardwareDriver pattern)
    - Hardware acceleration via Hailo 8L
    - Graceful fallback to simulation mode
    - Input preprocessing pipeline
    - Output postprocessing with confidence scoring
    - Performance metrics tracking
    - Safety override integration

    Example usage:
        service = AIInferenceService()
        await service.initialize()
        await service.start()

        # Inference loop
        prediction = await service.infer(sensor_frame)
        if prediction.confidence > 0.7:
            motor_commands = prediction.to_motor_commands()

        await service.stop()
    """

    DEFAULT_MODEL_PATH = Path("/home/kp/repos/lawnberry_pi/models/lawnmower_vla.hef")

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)

        # Model configuration
        self._model_config = VLAModelConfig()
        if config:
            for key, value in config.items():
                if hasattr(self._model_config, key):
                    setattr(self._model_config, key, value)

        # Hailo driver
        self._hailo: HailoDriver | None = None
        self._model_loaded = False
        self._model_path: Path | None = None
        self._model_name = "lawnmower_vla"
        self._model_version = "1.0.0"

        # State
        self._enabled = False
        self._control_mode = ControlMode.MANUAL
        self._frame_id = 0
        self._last_prediction: ActionPrediction | None = None
        self._last_prediction_time: float = 0.0

        # Metrics
        self._metrics = InferenceMetrics(
            model_name=self._model_name,
            target_fps=self._model_config.target_fps,
        )

        # Simulation state
        self._sim_steering = 0.0
        self._sim_throttle = 0.5
        self._sim_obstacle_counter = 0

    async def initialize(self) -> None:
        """Initialize the AI inference service."""
        is_simulation_mode() or os.environ.get("SIM_MODE") == "1"

        # Initialize Hailo driver
        hailo_config = {
            "model": "yolov8m",  # Placeholder until VLA model available
        }

        # Check if VLA model exists
        model_path = self.config.get("model_path", self.DEFAULT_MODEL_PATH)
        if isinstance(model_path, str):
            model_path = Path(model_path)

        if model_path.exists():
            hailo_config["hef_path"] = str(model_path)
            self._model_path = model_path
            self._model_loaded = True
        else:
            # Use default model for testing
            self._model_path = None
            self._model_loaded = False

        self._hailo = HailoDriver(hailo_config)
        await self._hailo.initialize()

        self._metrics.hardware_accelerated = self._hailo.hardware_available
        self._metrics.model_name = self._model_name
        self._metrics.model_version = self._model_version

        self.initialized = True

    async def start(self) -> None:
        """Start the inference service."""
        if self._hailo:
            await self._hailo.start()
        self.running = True

    async def stop(self) -> None:
        """Stop the inference service."""
        self.running = False
        self._enabled = False
        if self._hailo:
            await self._hailo.stop()

    async def health_check(self) -> dict[str, Any]:
        """Return health status."""
        hailo_health = {}
        if self._hailo:
            hailo_health = await self._hailo.health_check()

        return {
            "service": "ai_inference",
            "initialized": self.initialized,
            "running": self.running,
            "enabled": self._enabled,
            "control_mode": self._control_mode.value,
            "model_loaded": self._model_loaded,
            "model_name": self._model_name,
            "model_version": self._model_version,
            "model_path": str(self._model_path) if self._model_path else None,
            "hailo": hailo_health,
            "metrics": {
                "total_inferences": self._metrics.total_inferences,
                "success_rate": round(self._metrics.success_rate, 3),
                "avg_inference_ms": round(self._metrics.avg_inference_time_ms, 2),
                "current_fps": round(self._metrics.inferences_per_second, 1),
            },
        }

    async def load_model(self, model_path: Path | str) -> bool:
        """Load a new VLA model.

        Args:
            model_path: Path to the .hef model file.

        Returns:
            True if model loaded successfully.
        """
        if isinstance(model_path, str):
            model_path = Path(model_path)

        if not model_path.exists():
            return False

        # Stop current inference
        was_running = self.running
        if was_running:
            await self.stop()

        # Reinitialize with new model
        self._hailo = HailoDriver({"hef_path": str(model_path)})
        await self._hailo.initialize()

        self._model_path = model_path
        self._model_loaded = True

        # Extract model info from filename
        stem = model_path.stem
        if "_v" in stem:
            parts = stem.rsplit("_v", 1)
            self._model_name = parts[0]
            self._model_version = parts[1] if len(parts) > 1 else "1.0.0"
        else:
            self._model_name = stem
            self._model_version = "1.0.0"

        if was_running:
            await self.start()

        return True

    async def enable(self) -> bool:
        """Enable AI control mode."""
        if not self.initialized or not self.running:
            return False
        self._enabled = True
        self._control_mode = ControlMode.AI_AUTONOMOUS
        return True

    async def disable(self) -> None:
        """Disable AI control mode."""
        self._enabled = False
        self._control_mode = ControlMode.MANUAL

    async def infer(
        self,
        frame: MowerDataFrame,
        coverage_grid: np.ndarray | None = None,
        datum: tuple[float, float] | None = None,
    ) -> ActionPrediction:
        """Run inference on sensor data.

        Args:
            frame: MowerDataFrame with current sensor readings.
            coverage_grid: Optional (G, G) already-mowed occupancy grid from the
                autonomous loop; zeros are used when omitted.
            datum: Optional (lat, lon) local-frame origin (the yard datum). When
                omitted, the frame's own position is used (self-relative), which
                keeps heading/speed/fix meaningful but zeroes the position.

        Returns:
            ActionPrediction with recommended control actions.
        """
        if not self.initialized or not self.running:
            raise RuntimeError("Service not initialized or not running")

        time.perf_counter()

        # Preprocess inputs
        preprocess_start = time.perf_counter()
        model_inputs = self._preprocess(frame, coverage_grid, datum)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        # Run inference
        inference_start = time.perf_counter()
        if self._hailo and self._hailo.hardware_available:
            result = await self._hailo.infer(model_inputs)
            raw_outputs = result.outputs
            inference_time = result.inference_time_ms
        else:
            # Simulation mode inference
            raw_outputs = await self._simulate_vla_inference(frame)
            inference_time = (time.perf_counter() - inference_start) * 1000

        # Postprocess outputs
        postprocess_start = time.perf_counter()
        prediction = self._postprocess(raw_outputs, frame)
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000

        # Update timing
        prediction.preprocessing_time_ms = preprocess_time
        prediction.inference_time_ms = inference_time
        prediction.postprocessing_time_ms = postprocess_time
        prediction.frame_id = self._frame_id
        prediction.model_name = self._model_name
        prediction.model_version = self._model_version

        # Update state
        self._frame_id += 1
        self._last_prediction = prediction
        self._last_prediction_time = time.time()

        # Update metrics
        self._metrics.update_with_prediction(prediction)

        return prediction

    def _preprocess(
        self,
        frame: MowerDataFrame,
        coverage_grid: np.ndarray | None = None,
        datum: tuple[float, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Preprocess sensor data into the deployed (distilled) model's inputs.

        Emits the three inputs the edge student / HEF consumes:
        ``image`` (1, 3, H, W), ``sensors`` (1, 20) and ``coverage_map``
        (1, 1, G, G). The 20-d sensor vector is local-ENU GPS features
        (see nav.location_features.gps_feature_vector) + IMU(9) + ultrasonic(3),
        normalised identically to the training dataset so the model sees the same
        distribution it learned on.
        """
        inputs: dict[str, np.ndarray] = {}

        # Single RGB image (pi_camera), CHW to match the student ONNX inputs.
        if frame.pi_camera_rgb is not None:
            img = self._preprocess_image(frame.pi_camera_rgb, self._model_config.rgb_input_size)
        else:
            h, w = self._model_config.rgb_input_size
            img = np.zeros((h, w, 3), dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        inputs["image"] = img[np.newaxis, ...].astype(np.float32)  # (1, 3, H, W)

        # Sensor vector: local-ENU GPS features (8) + IMU (9) + ultrasonic (3).
        if datum is None:
            datum = (frame.gps.latitude, frame.gps.longitude)  # self-relative fallback
        gps_feat = gps_feature_vector(frame.gps, datum)
        imu = frame.imu
        imu_feat = np.array(
            [
                imu.roll / 180.0,
                imu.pitch / 180.0,
                imu.yaw / 180.0,
                imu.linear_accel_x / 10.0,
                imu.linear_accel_y / 10.0,
                imu.linear_accel_z / 10.0,
                imu.angular_vel_x / 5.0,
                imu.angular_vel_y / 5.0,
                imu.angular_vel_z / 5.0,
            ],
            dtype=np.float32,
        )
        ultra = frame.ultrasonic
        ultra_feat = np.array(
            [
                ultra.front_left_cm / 100.0,
                ultra.front_center_cm / 100.0,
                ultra.front_right_cm / 100.0,
            ],
            dtype=np.float32,
        )
        sensors = np.concatenate([gps_feat, imu_feat, ultra_feat]).astype(np.float32)
        inputs["sensors"] = sensors[np.newaxis, ...]  # (1, 20)

        # Coverage map (already-mowed grid); zeros when the loop hasn't supplied one.
        if coverage_grid is None:
            coverage_grid = np.zeros((COVERAGE_GRID_SIZE, COVERAGE_GRID_SIZE), dtype=np.float32)
        inputs["coverage_map"] = coverage_grid[np.newaxis, np.newaxis, ...].astype(np.float32)

        return inputs

    def _preprocess_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize and normalize a single image.

        Uses OpenCV when available; falls back to PIL for simulation environments.
        """
        h, w = target_size

        try:
            import cv2

            resized = cv2.resize(image, (w, h))
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image

            # PIL expects HWC uint8 — image is already in that format
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((w, h), Image.BILINEAR)
            rgb = np.array(pil_img, dtype=np.float32) / 255.0

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        if self._model_config.normalize_images:
            # Apply ImageNet normalization
            mean = np.array(self._model_config.image_mean, dtype=np.float32)
            std = np.array(self._model_config.image_std, dtype=np.float32)
            normalized = (normalized - mean) / std

        return normalized

    def _postprocess(
        self, outputs: dict[str, np.ndarray], frame: MowerDataFrame
    ) -> ActionPrediction:
        """Convert model outputs to ActionPrediction.

        Interprets raw model output tensors as discretized action
        probabilities and converts to continuous control values.
        """
        # For simulation or when model outputs don't match expected format,
        # use simulated values from _simulate_vla_inference
        if "steering" in outputs and "throttle" in outputs:
            # Real model output with named tensors
            steering_probs = outputs["steering"]
            throttle_probs = outputs["throttle"]
            blade_prob = outputs.get("blade", np.array([0.5]))

            # Decode steering from probability distribution
            steering_bins = np.linspace(-1.0, 1.0, self._model_config.steering_bins)
            steering_idx = np.argmax(steering_probs)
            steering = float(steering_bins[steering_idx])
            steering_conf = float(steering_probs.max())

            # Decode throttle
            throttle_bins = np.linspace(0.0, 1.0, self._model_config.throttle_bins)
            throttle_idx = np.argmax(throttle_probs)
            throttle = float(throttle_bins[throttle_idx])
            throttle_conf = float(throttle_probs.max())

            # Decode blade
            blade = bool(blade_prob[0] > 0.5)
            blade_conf = float(abs(blade_prob[0] - 0.5) * 2)

            confidence = (steering_conf + throttle_conf + blade_conf) / 3.0
        else:
            # Simulation or generic model output
            # Extract from pre-computed simulation values
            steering = outputs.get("_sim_steering", 0.0)
            throttle = outputs.get("_sim_throttle", 0.5)
            blade = outputs.get("_sim_blade", True)
            confidence = outputs.get("_sim_confidence", 0.85)
            steering_conf = confidence
            throttle_conf = confidence
            blade_conf = confidence

        # Safety checks
        obstacle_detected = frame.ultrasonic.min_distance_cm < 50
        boundary_warning = False  # Would check geofence here

        # Apply safety overrides
        safety_override = False
        if obstacle_detected and frame.ultrasonic.min_distance_cm < 30:
            throttle = 0.0  # Stop if very close obstacle
            safety_override = True

        return ActionPrediction(
            steering=steering,
            throttle=throttle,
            blade=blade,
            confidence=confidence,
            steering_confidence=steering_conf,
            throttle_confidence=throttle_conf,
            blade_confidence=blade_conf,
            safety_override=safety_override,
            obstacle_detected=obstacle_detected,
            boundary_warning=boundary_warning,
        )

    async def _simulate_vla_inference(self, frame: MowerDataFrame) -> dict[str, Any]:
        """Simulate VLA model inference for testing.

        Generates realistic-looking control predictions based on
        sensor data without a real model.
        """
        # Simulate latency
        await asyncio.sleep(0.015)  # 15ms simulated inference

        # Simple reactive behavior simulation
        ultrasonic_min = frame.ultrasonic.min_distance_cm

        # Obstacle avoidance behavior
        if ultrasonic_min < 50:
            self._sim_obstacle_counter += 1
            # Turn away from nearest obstacle
            if frame.ultrasonic.front_left_cm < frame.ultrasonic.front_right_cm:
                steering = 0.5  # Turn right
            else:
                steering = -0.5  # Turn left
            throttle = max(0.2, 0.5 * (ultrasonic_min / 50))
        else:
            self._sim_obstacle_counter = 0
            # Nominal mowing: slight random variation
            steering = self._sim_steering + np.random.uniform(-0.1, 0.1)
            steering = max(-1.0, min(1.0, steering))
            self._sim_steering = steering * 0.9  # Decay toward center

            throttle = 0.6 + np.random.uniform(-0.1, 0.1)

        # Blade control: on unless obstacle very close
        blade = ultrasonic_min > 20

        # Confidence decreases with uncertainty
        confidence = 0.85
        if ultrasonic_min < 30:
            confidence = 0.7
        if self._sim_obstacle_counter > 5:
            confidence = 0.6

        return {
            "_sim_steering": steering,
            "_sim_throttle": throttle,
            "_sim_blade": blade,
            "_sim_confidence": confidence,
        }

    def get_status(self) -> AIControlStatus:
        """Get current AI control status."""
        prediction_age_ms = 0.0
        if self._last_prediction_time > 0:
            prediction_age_ms = (time.time() - self._last_prediction_time) * 1000

        return AIControlStatus(
            enabled=self._enabled,
            mode=self._control_mode,
            model_loaded=self._model_loaded,
            last_prediction=self._last_prediction.to_dict() if self._last_prediction else None,
            prediction_age_ms=prediction_age_ms,
            hailo_available=self._hailo.hardware_available if self._hailo else False,
            hailo_temperature=None,  # Would query from Hailo driver
            using_hardware=self._hailo.hardware_available if self._hailo else False,
            safety_engaged=self._last_prediction.safety_override
            if self._last_prediction
            else False,
            safety_reason="obstacle_detected"
            if (self._last_prediction and self._last_prediction.obstacle_detected)
            else None,
            success_rate=self._metrics.success_rate,
            avg_latency_ms=self._metrics.avg_inference_time_ms,
            current_fps=self._metrics.inferences_per_second,
        )

    def get_metrics(self) -> InferenceMetrics:
        """Get inference performance metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = InferenceMetrics(
            model_name=self._model_name,
            model_version=self._model_version,
            target_fps=self._model_config.target_fps,
            hardware_accelerated=self._hailo.hardware_available if self._hailo else False,
        )


__all__ = [
    "AIInferenceService",
    "VLAModelConfig",
    "get_ai_inference_service",
]
