"""Multi-modal data frame for AI training (BEAD-102).

Defines the unified sensor data structure used for:
- Recording training data during manual/autonomous operation
- Uploading to Thor server for AI model training
- Replay and analysis of recorded sessions

Data Format:
- Timestamps in UTC ISO8601 format
- Images as numpy arrays (BGR color order)
- Coordinates in decimal degrees (WGS84)
- Distances in metric units (meters, centimeters, millimeters)
- Angles in degrees
- Serializes to MessagePack for efficient storage/transmission
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import msgpack
    import msgpack_numpy as m
    m.patch()  # Enable numpy array serialization
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


class RTKFixType(str, Enum):
    """RTK GPS fix quality types."""
    NONE = "none"
    SINGLE = "single"         # Standard GPS fix
    FLOAT = "float"           # RTK float solution
    FIXED = "fixed"           # RTK fixed solution (cm accuracy)
    DGPS = "dgps"             # Differential GPS
    PPS = "pps"               # PPS fix


class MowerState(str, Enum):
    """Operational states of the mower."""
    IDLE = "idle"
    MOWING = "mowing"
    TURNING = "turning"
    RETURNING = "returning"   # Returning to dock
    DOCKED = "docked"
    PERIMETER_RECORDING = "perimeter_recording"
    MANUAL = "manual"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"


class ActionSource(str, Enum):
    """Source of control actions."""
    MANUAL = "manual"         # Human teleop
    AUTONOMOUS = "autonomous" # AI policy
    PLAYBACK = "playback"     # Replaying recorded path
    SAFETY = "safety"         # Safety override


@dataclass
class GPSData:
    """GPS sensor data."""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    rtk_fix_type: RTKFixType = RTKFixType.NONE
    hdop: float = 99.0        # Horizontal dilution of precision
    vdop: float = 99.0        # Vertical dilution of precision
    num_satellites: int = 0
    speed_mps: float = 0.0    # Ground speed in m/s
    heading: float = 0.0      # GPS-derived heading in degrees


@dataclass
class IMUData:
    """IMU sensor data (BNO085)."""
    roll: float = 0.0         # degrees
    pitch: float = 0.0        # degrees
    yaw: float = 0.0          # degrees (heading)
    linear_accel_x: float = 0.0  # m/s^2
    linear_accel_y: float = 0.0
    linear_accel_z: float = 0.0
    angular_vel_x: float = 0.0   # rad/s
    angular_vel_y: float = 0.0
    angular_vel_z: float = 0.0
    calibration_status: str = "unknown"


@dataclass
class UltrasonicData:
    """Ultrasonic sensor array data (HC-SR04 x3)."""
    front_left_cm: float = 0.0
    front_center_cm: float = 0.0
    front_right_cm: float = 0.0
    min_distance_cm: float = 0.0  # Minimum of all readings


@dataclass
class ToFData:
    """Time-of-Flight sensor data (VL53L0X x2)."""
    left_mm: float = 0.0      # Blade height left
    right_mm: float = 0.0     # Blade height right


@dataclass
class MotorState:
    """Motor controller state."""
    wheel_speed_left: float = 0.0   # RPM
    wheel_speed_right: float = 0.0  # RPM
    blade_speed: float = 0.0        # RPM
    blade_enabled: bool = False
    wheel_current_left: float = 0.0  # Amps
    wheel_current_right: float = 0.0
    blade_current: float = 0.0


@dataclass
class ControlAction:
    """Control action being executed."""
    steering: float = 0.0     # -1.0 (full left) to 1.0 (full right)
    throttle: float = 0.0     # 0.0 to 1.0
    blade_command: bool = False
    source: ActionSource = ActionSource.MANUAL


@dataclass
class PowerState:
    """Power system state."""
    battery_voltage: float = 0.0
    battery_current: float = 0.0
    battery_soc: float = 0.0  # State of charge 0-100%
    charging: bool = False


@dataclass
class MowerDataFrame:
    """Single frame of multi-modal mower sensor data for AI training.

    This is the primary data structure used for:
    - Recording training data during operation
    - Transmission to Thor training server
    - Replay for debugging and analysis

    All images are stored as numpy arrays in BGR format.
    """
    # Identifiers
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    frame_id: int = 0

    # Camera data (stored as numpy arrays, dtype=uint8, shape=[H,W,3] BGR)
    stereo_left: Optional[np.ndarray] = None      # 1280x960 or 960x540
    stereo_right: Optional[np.ndarray] = None     # 1280x960 or 960x540
    stereo_depth: Optional[np.ndarray] = None     # Computed disparity map
    pi_camera_rgb: Optional[np.ndarray] = None    # 1920x1080 or 1280x720

    # Sensor data
    gps: GPSData = field(default_factory=GPSData)
    imu: IMUData = field(default_factory=IMUData)
    ultrasonic: UltrasonicData = field(default_factory=UltrasonicData)
    tof: ToFData = field(default_factory=ToFData)

    # Motor and control state
    motor: MotorState = field(default_factory=MotorState)
    action: ControlAction = field(default_factory=ControlAction)
    power: PowerState = field(default_factory=PowerState)

    # Operational state
    mower_state: MowerState = MowerState.IDLE

    # AI inference results (if running)
    ai_inference_time_ms: float = 0.0
    ai_detections: List[Dict[str, Any]] = field(default_factory=list)
    ai_depth_available: bool = False

    # Recording metadata
    recording_name: str = ""
    lap_number: int = 0
    waypoint_index: int = 0

    def to_dict(self, include_images: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Args:
            include_images: If False, excludes image data for lightweight transmission.

        Returns:
            Dictionary representation of the data frame.
        """
        result = {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "gps": asdict(self.gps),
            "imu": asdict(self.imu),
            "ultrasonic": asdict(self.ultrasonic),
            "tof": asdict(self.tof),
            "motor": asdict(self.motor),
            "action": asdict(self.action),
            "power": asdict(self.power),
            "mower_state": self.mower_state.value,
            "ai_inference_time_ms": self.ai_inference_time_ms,
            "ai_detections": self.ai_detections,
            "ai_depth_available": self.ai_depth_available,
            "recording_name": self.recording_name,
            "lap_number": self.lap_number,
            "waypoint_index": self.waypoint_index,
        }

        # Handle enum conversion in nested dicts
        result["gps"]["rtk_fix_type"] = self.gps.rtk_fix_type.value
        result["action"]["source"] = self.action.source.value

        if include_images:
            # Include image data as numpy arrays (for msgpack serialization)
            result["stereo_left"] = self.stereo_left
            result["stereo_right"] = self.stereo_right
            result["stereo_depth"] = self.stereo_depth
            result["pi_camera_rgb"] = self.pi_camera_rgb
        else:
            # Include image metadata only
            result["stereo_left_shape"] = list(self.stereo_left.shape) if self.stereo_left is not None else None
            result["stereo_right_shape"] = list(self.stereo_right.shape) if self.stereo_right is not None else None
            result["stereo_depth_shape"] = list(self.stereo_depth.shape) if self.stereo_depth is not None else None
            result["pi_camera_shape"] = list(self.pi_camera_rgb.shape) if self.pi_camera_rgb is not None else None

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MowerDataFrame":
        """Create MowerDataFrame from dictionary."""
        # Parse timestamp
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)

        # Parse nested data classes
        gps_data = data.get("gps", {})
        if "rtk_fix_type" in gps_data:
            gps_data["rtk_fix_type"] = RTKFixType(gps_data["rtk_fix_type"])
        gps = GPSData(**gps_data) if gps_data else GPSData()

        imu = IMUData(**data.get("imu", {})) if data.get("imu") else IMUData()
        ultrasonic = UltrasonicData(**data.get("ultrasonic", {})) if data.get("ultrasonic") else UltrasonicData()
        tof = ToFData(**data.get("tof", {})) if data.get("tof") else ToFData()
        motor = MotorState(**data.get("motor", {})) if data.get("motor") else MotorState()

        action_data = data.get("action", {})
        if "source" in action_data:
            action_data["source"] = ActionSource(action_data["source"])
        action = ControlAction(**action_data) if action_data else ControlAction()

        power = PowerState(**data.get("power", {})) if data.get("power") else PowerState()

        mower_state = data.get("mower_state", "idle")
        if isinstance(mower_state, str):
            mower_state = MowerState(mower_state)

        return cls(
            timestamp=ts,
            session_id=data.get("session_id", str(uuid.uuid4())),
            frame_id=data.get("frame_id", 0),
            stereo_left=data.get("stereo_left"),
            stereo_right=data.get("stereo_right"),
            stereo_depth=data.get("stereo_depth"),
            pi_camera_rgb=data.get("pi_camera_rgb"),
            gps=gps,
            imu=imu,
            ultrasonic=ultrasonic,
            tof=tof,
            motor=motor,
            action=action,
            power=power,
            mower_state=mower_state,
            ai_inference_time_ms=data.get("ai_inference_time_ms", 0.0),
            ai_detections=data.get("ai_detections", []),
            ai_depth_available=data.get("ai_depth_available", False),
            recording_name=data.get("recording_name", ""),
            lap_number=data.get("lap_number", 0),
            waypoint_index=data.get("waypoint_index", 0),
        )

    def serialize(self) -> bytes:
        """Serialize to MessagePack bytes for efficient storage/transmission.

        Returns:
            MessagePack-encoded bytes.

        Raises:
            ImportError: If msgpack is not installed.
        """
        if not HAS_MSGPACK:
            raise ImportError("msgpack and msgpack_numpy required for serialization")

        data = self.to_dict(include_images=True)
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> "MowerDataFrame":
        """Deserialize from MessagePack bytes.

        Args:
            data: MessagePack-encoded bytes.

        Returns:
            MowerDataFrame instance.
        """
        if not HAS_MSGPACK:
            raise ImportError("msgpack and msgpack_numpy required for deserialization")

        unpacked = msgpack.unpackb(data, raw=False)
        return cls.from_dict(unpacked)

    def estimate_size_bytes(self) -> int:
        """Estimate serialized size in bytes (without compression)."""
        size = 1024  # Base overhead for metadata

        for img in [self.stereo_left, self.stereo_right, self.stereo_depth, self.pi_camera_rgb]:
            if img is not None:
                size += img.nbytes

        return size

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get lightweight telemetry summary for real-time streaming."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
            "gps": {
                "lat": round(self.gps.latitude, 7),
                "lon": round(self.gps.longitude, 7),
                "fix": self.gps.rtk_fix_type.value,
                "sats": self.gps.num_satellites,
            },
            "imu": {
                "roll": round(self.imu.roll, 1),
                "pitch": round(self.imu.pitch, 1),
                "yaw": round(self.imu.yaw, 1),
            },
            "ultrasonic_min_cm": round(self.ultrasonic.min_distance_cm, 1),
            "blade_enabled": self.motor.blade_enabled,
            "battery_soc": round(self.power.battery_soc, 1),
            "state": self.mower_state.value,
            "action": {
                "steering": round(self.action.steering, 2),
                "throttle": round(self.action.throttle, 2),
                "source": self.action.source.value,
            },
        }


@dataclass
class RecordingSession:
    """Metadata for a recording session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    frame_count: int = 0
    total_distance_m: float = 0.0
    mowing_area_m2: float = 0.0
    session_type: str = "training"  # training, perimeter, calibration
    notes: str = ""

    # File paths
    data_file: Optional[Path] = None
    video_file: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "frame_count": self.frame_count,
            "total_distance_m": self.total_distance_m,
            "mowing_area_m2": self.mowing_area_m2,
            "session_type": self.session_type,
            "notes": self.notes,
            "data_file": str(self.data_file) if self.data_file else None,
            "video_file": str(self.video_file) if self.video_file else None,
        }


__all__ = [
    "RTKFixType",
    "MowerState",
    "ActionSource",
    "GPSData",
    "IMUData",
    "UltrasonicData",
    "ToFData",
    "MotorState",
    "ControlAction",
    "PowerState",
    "MowerDataFrame",
    "RecordingSession",
]
