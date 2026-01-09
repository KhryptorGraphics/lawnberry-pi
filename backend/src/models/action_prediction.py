"""
Action prediction models for LawnBerry Pi VLA inference.

Defines the output structure for Vision-Language-Action model predictions
used in autonomous mowing control.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dataclasses import dataclass, field as dataclass_field
import time


class ActionConfidence(str, Enum):
    """Confidence levels for action predictions."""
    HIGH = "high"       # >0.9 confidence
    MEDIUM = "medium"   # 0.7-0.9 confidence
    LOW = "low"         # 0.5-0.7 confidence
    UNCERTAIN = "uncertain"  # <0.5 confidence


class ControlMode(str, Enum):
    """Active control mode."""
    MANUAL = "manual"
    AI_AUTONOMOUS = "ai_autonomous"
    AI_ASSISTED = "ai_assisted"
    WAYPOINT = "waypoint"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ActionPrediction:
    """Single action prediction from VLA model.

    Represents the model's recommended control actions based on
    current sensor inputs and navigation state.
    """
    # Control outputs
    steering: float  # -1.0 (full left) to 1.0 (full right)
    throttle: float  # 0.0 (stop) to 1.0 (full speed)
    blade: bool      # True = blade on, False = blade off

    # Confidence metrics
    confidence: float  # Overall prediction confidence 0.0-1.0
    steering_confidence: float = 0.0
    throttle_confidence: float = 0.0
    blade_confidence: float = 0.0

    # Performance metrics
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0

    # Metadata
    model_name: str = "lawnmower_vla"
    model_version: str = "1.0.0"
    frame_id: int = 0
    timestamp: float = dataclass_field(default_factory=time.time)

    # Safety annotations
    safety_override: bool = False  # True if action was modified by safety system
    obstacle_detected: bool = False
    boundary_warning: bool = False

    def __post_init__(self):
        """Validate action bounds."""
        self.steering = max(-1.0, min(1.0, self.steering))
        self.throttle = max(0.0, min(1.0, self.throttle))
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def confidence_level(self) -> ActionConfidence:
        """Get categorical confidence level."""
        if self.confidence >= 0.9:
            return ActionConfidence.HIGH
        elif self.confidence >= 0.7:
            return ActionConfidence.MEDIUM
        elif self.confidence >= 0.5:
            return ActionConfidence.LOW
        return ActionConfidence.UNCERTAIN

    @property
    def total_time_ms(self) -> float:
        """Total processing time."""
        return self.preprocessing_time_ms + self.inference_time_ms + self.postprocessing_time_ms

    def to_motor_commands(self) -> Dict[str, float]:
        """Convert to motor controller format.

        Returns dict compatible with MotorService.set_speeds()

        Steering convention:
        - Positive steering (+1.0) = turn right (right wheel slows, left wheel faster)
        - Negative steering (-1.0) = turn left (left wheel slows, right wheel faster)
        """
        # Differential drive: steering affects left/right balance
        base_speed = self.throttle

        # Apply steering as differential
        # For right turn (positive steering): slow down right wheel
        # For left turn (negative steering): slow down left wheel
        left_speed = base_speed * (1.0 + min(0, self.steering))  # Reduced for left turns
        right_speed = base_speed * (1.0 - max(0, self.steering))  # Reduced for right turns

        return {
            "left_speed": left_speed,
            "right_speed": right_speed,
            "blade_enabled": self.blade,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "steering": self.steering,
            "throttle": self.throttle,
            "blade": self.blade,
            "confidence": self.confidence,
            "steering_confidence": self.steering_confidence,
            "throttle_confidence": self.throttle_confidence,
            "blade_confidence": self.blade_confidence,
            "confidence_level": self.confidence_level.value,
            "inference_time_ms": self.inference_time_ms,
            "total_time_ms": self.total_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "safety_override": self.safety_override,
            "obstacle_detected": self.obstacle_detected,
            "boundary_warning": self.boundary_warning,
        }


class InferenceMetrics(BaseModel):
    """Aggregated inference performance metrics."""

    # Counters
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    safety_overrides: int = 0

    # Timing
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = float('inf')
    max_inference_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0

    # Throughput
    inferences_per_second: float = 0.0
    target_fps: float = 10.0

    # Confidence distribution
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    uncertain_count: int = 0
    avg_confidence: float = 0.0

    # Model info
    model_name: str = ""
    model_version: str = ""
    hardware_accelerated: bool = False

    # Timestamps
    first_inference_time: Optional[datetime] = None
    last_inference_time: Optional[datetime] = None

    model_config = ConfigDict(use_enum_values=True)

    @property
    def success_rate(self) -> float:
        """Calculate inference success rate."""
        if self.total_inferences == 0:
            return 0.0
        return self.successful_inferences / self.total_inferences

    @property
    def meets_target_fps(self) -> bool:
        """Check if inference rate meets target."""
        return self.inferences_per_second >= self.target_fps * 0.9  # 90% of target

    def update_with_prediction(self, prediction: ActionPrediction, success: bool = True):
        """Update metrics with a new prediction."""
        self.total_inferences += 1

        if success:
            self.successful_inferences += 1
        else:
            self.failed_inferences += 1
            return

        # Update timing
        inference_ms = prediction.inference_time_ms
        total_ms = prediction.total_time_ms

        self.min_inference_time_ms = min(self.min_inference_time_ms, inference_ms)
        self.max_inference_time_ms = max(self.max_inference_time_ms, inference_ms)

        # Running average
        n = self.successful_inferences
        self.avg_inference_time_ms = (
            (self.avg_inference_time_ms * (n - 1) + inference_ms) / n
        )
        self.avg_total_time_ms = (
            (self.avg_total_time_ms * (n - 1) + total_ms) / n
        )

        # Update confidence distribution
        level = prediction.confidence_level
        if level == ActionConfidence.HIGH:
            self.high_confidence_count += 1
        elif level == ActionConfidence.MEDIUM:
            self.medium_confidence_count += 1
        elif level == ActionConfidence.LOW:
            self.low_confidence_count += 1
        else:
            self.uncertain_count += 1

        self.avg_confidence = (
            (self.avg_confidence * (n - 1) + prediction.confidence) / n
        )

        # Update safety stats
        if prediction.safety_override:
            self.safety_overrides += 1

        # Update timestamps
        now = datetime.now(timezone.utc)
        if self.first_inference_time is None:
            self.first_inference_time = now
        self.last_inference_time = now

        # Calculate throughput
        if self.first_inference_time and self.last_inference_time:
            duration = (self.last_inference_time - self.first_inference_time).total_seconds()
            if duration > 0:
                self.inferences_per_second = self.successful_inferences / duration


class AIControlStatus(BaseModel):
    """Current AI control system status."""

    # State
    enabled: bool = False
    mode: ControlMode = ControlMode.MANUAL
    model_loaded: bool = False

    # Current prediction
    last_prediction: Optional[Dict[str, Any]] = None
    prediction_age_ms: float = 0.0

    # Hardware
    hailo_available: bool = False
    hailo_temperature: Optional[float] = None
    using_hardware: bool = False

    # Safety
    safety_engaged: bool = False
    safety_reason: Optional[str] = None

    # Metrics summary
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    current_fps: float = 0.0

    model_config = ConfigDict(use_enum_values=True)


__all__ = [
    "ActionPrediction",
    "ActionConfidence",
    "ControlMode",
    "InferenceMetrics",
    "AIControlStatus",
]
