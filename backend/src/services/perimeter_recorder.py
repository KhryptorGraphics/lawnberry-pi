"""Perimeter Recording Service for AI Training Data Collection (BEAD-103).

Records multi-modal sensor data during manual operation to create training
datasets for autonomous mowing AI. Captures synchronized data from:
- Stereo camera (left/right images, depth)
- Pi Camera RGB
- GPS/RTK position
- IMU orientation
- Ultrasonic sensors
- ToF blade height sensors
- Motor/control state
- Power system state

Data is stored as MowerDataFrames serialized to MessagePack for efficient
storage and later upload to Thor training server.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

import numpy as np

from ..models.mower_data_frame import (
    MowerDataFrame, RecordingSession, MowerState, ActionSource,
    GPSData, IMUData, UltrasonicData, ToFData, MotorState, ControlAction, PowerState,
    RTKFixType, HAS_MSGPACK
)
from ..models import SensorStatus, GpsMode

logger = logging.getLogger(__name__)


class RecordingState:
    """Recording session states."""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class RecordingConfig:
    """Configuration for recording sessions."""
    # Capture rate
    target_fps: float = 10.0  # Target frames per second

    # Image settings
    capture_stereo: bool = True
    capture_pi_camera: bool = True
    stereo_resolution: tuple = (960, 540)  # Downscaled for training
    pi_camera_resolution: tuple = (1280, 720)

    # Storage settings
    storage_dir: Path = field(default_factory=lambda: Path("data/recordings"))
    max_session_frames: int = 36000  # 1 hour at 10fps
    compress_images: bool = False  # Use msgpack default encoding

    # Filtering
    min_gps_satellites: int = 4
    require_rtk_fix: bool = False


class PerimeterRecordingService:
    """Service for recording training data during manual mowing operations.

    Captures synchronized multi-modal sensor data for AI training.
    Sessions are stored as MessagePack files for efficient upload to Thor.

    Example usage:
        recorder = PerimeterRecordingService(sensor_manager, robohat_service)
        await recorder.initialize()

        session = await recorder.start_recording("lawn_perimeter_001")
        # ... manual driving ...
        await recorder.stop_recording()

        # Access recorded data
        sessions = recorder.list_sessions()
    """

    def __init__(
        self,
        sensor_manager: Any = None,
        robohat_service: Any = None,
        camera_service: Any = None,
        config: Optional[RecordingConfig] = None,
    ):
        """Initialize the recording service.

        Args:
            sensor_manager: SensorManager instance for sensor data.
            robohat_service: RoboHAT service for motor/control state.
            camera_service: Camera stream service for Pi Camera.
            config: Recording configuration.
        """
        self._sensor_manager = sensor_manager
        self._robohat_service = robohat_service
        self._camera_service = camera_service
        self.config = config or RecordingConfig()

        # Session state
        self._state = RecordingState.IDLE
        self._current_session: Optional[RecordingSession] = None
        self._session_frames: List[MowerDataFrame] = []
        self._frame_counter = 0

        # Recording task
        self._recording_task: Optional[asyncio.Task] = None
        self._capture_interval = 1.0 / self.config.target_fps

        # Statistics
        self._total_frames_recorded = 0
        self._total_sessions = 0
        self._last_frame_time = 0.0
        self._avg_capture_time_ms = 0.0

        # Callbacks for real-time streaming
        self._frame_callbacks: List[Callable[[MowerDataFrame], None]] = []

        # Initialize storage
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)

        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the recording service."""
        try:
            # Verify MessagePack is available
            if not HAS_MSGPACK:
                logger.warning("msgpack not available - recordings will be limited to JSON")

            # Verify storage directory is writable
            test_file = self.config.storage_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                logger.error(f"Storage directory not writable: {e}")
                return False

            self.initialized = True
            logger.info(f"PerimeterRecordingService initialized, storage: {self.config.storage_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize recording service: {e}")
            return False

    async def start_recording(
        self,
        name: str,
        session_type: str = "perimeter",
        notes: str = "",
    ) -> RecordingSession:
        """Start a new recording session.

        Args:
            name: Human-readable name for the session.
            session_type: Type of recording (perimeter, training, calibration).
            notes: Optional notes about the session.

        Returns:
            RecordingSession metadata.

        Raises:
            RuntimeError: If already recording or not initialized.
        """
        if not self.initialized:
            raise RuntimeError("Recording service not initialized")

        if self._state == RecordingState.RECORDING:
            raise RuntimeError("Already recording - stop current session first")

        # Create session
        session_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        self._current_session = RecordingSession(
            session_id=session_id,
            name=name,
            start_time=timestamp,
            session_type=session_type,
            notes=notes,
            data_file=self.config.storage_dir / f"{session_id}.msgpack",
        )

        # Reset frame buffer
        self._session_frames = []
        self._frame_counter = 0

        # Start recording loop
        self._state = RecordingState.RECORDING
        self._recording_task = asyncio.create_task(self._recording_loop())

        self._total_sessions += 1
        logger.info(f"Started recording session: {name} ({session_id})")

        return self._current_session

    async def stop_recording(self) -> RecordingSession:
        """Stop the current recording session and save to disk.

        Returns:
            Final RecordingSession with frame count and file path.

        Raises:
            RuntimeError: If not currently recording.
        """
        if self._state not in (RecordingState.RECORDING, RecordingState.PAUSED):
            raise RuntimeError("Not currently recording")

        self._state = RecordingState.STOPPING

        # Cancel recording task
        if self._recording_task:
            self._recording_task.cancel()
            try:
                await self._recording_task
            except asyncio.CancelledError:
                pass
            self._recording_task = None

        # Finalize session
        if self._current_session:
            self._current_session.end_time = datetime.now(timezone.utc)
            self._current_session.frame_count = len(self._session_frames)

            # Calculate total distance from GPS points
            self._current_session.total_distance_m = self._calculate_total_distance()

            # Save to disk
            await self._save_session()

            logger.info(
                f"Stopped recording: {self._current_session.name}, "
                f"{self._current_session.frame_count} frames"
            )

        session = self._current_session
        self._state = RecordingState.IDLE
        self._current_session = None
        self._session_frames = []

        return session

    async def pause_recording(self) -> None:
        """Pause the current recording session."""
        if self._state != RecordingState.RECORDING:
            raise RuntimeError("Not currently recording")

        self._state = RecordingState.PAUSED
        logger.info(f"Paused recording: {self._current_session.name if self._current_session else 'unknown'}")

    async def resume_recording(self) -> None:
        """Resume a paused recording session."""
        if self._state != RecordingState.PAUSED:
            raise RuntimeError("Recording is not paused")

        self._state = RecordingState.RECORDING
        logger.info(f"Resumed recording: {self._current_session.name if self._current_session else 'unknown'}")

    async def _recording_loop(self) -> None:
        """Main recording loop - captures frames at target FPS."""
        logger.info(f"Recording loop started at {self.config.target_fps} fps")

        while self._state in (RecordingState.RECORDING, RecordingState.PAUSED):
            loop_start = time.perf_counter()

            if self._state == RecordingState.RECORDING:
                try:
                    frame = await self._capture_frame()
                    if frame:
                        self._session_frames.append(frame)
                        self._frame_counter += 1
                        self._total_frames_recorded += 1

                        # Notify callbacks
                        for callback in self._frame_callbacks:
                            try:
                                callback(frame)
                            except Exception as e:
                                logger.warning(f"Frame callback error: {e}")

                        # Check frame limit
                        if len(self._session_frames) >= self.config.max_session_frames:
                            logger.warning("Max session frames reached, auto-stopping")
                            await self.stop_recording()
                            return

                except Exception as e:
                    logger.error(f"Frame capture error: {e}")

            # Maintain target frame rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self._capture_interval - elapsed)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Update timing stats
            capture_time = time.perf_counter() - loop_start
            self._avg_capture_time_ms = (
                self._avg_capture_time_ms * 0.9 + capture_time * 1000 * 0.1
            )
            self._last_frame_time = time.time()

    async def _capture_frame(self) -> Optional[MowerDataFrame]:
        """Capture a single multi-modal data frame."""
        frame_start = time.perf_counter()

        # Initialize frame with identifiers
        frame = MowerDataFrame(
            session_id=self._current_session.session_id if self._current_session else "",
            frame_id=self._frame_counter,
            mower_state=MowerState.PERIMETER_RECORDING,
            recording_name=self._current_session.name if self._current_session else "",
        )

        # Capture sensor data concurrently
        tasks = []

        if self._sensor_manager and hasattr(self._sensor_manager, 'read_all_sensors'):
            tasks.append(self._capture_sensor_data(frame))

        if self._robohat_service:
            tasks.append(self._capture_motor_state(frame))

        if self.config.capture_stereo and self._sensor_manager:
            tasks.append(self._capture_stereo_images(frame))

        if self.config.capture_pi_camera and self._camera_service:
            tasks.append(self._capture_pi_camera(frame))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Update capture timing
        capture_ms = (time.perf_counter() - frame_start) * 1000

        return frame

    async def _capture_sensor_data(self, frame: MowerDataFrame) -> None:
        """Capture GPS, IMU, ultrasonic, ToF, and power data."""
        try:
            sensor_data = await self._sensor_manager.read_all_sensors()

            # GPS
            if sensor_data.gps:
                gps = sensor_data.gps
                frame.gps = GPSData(
                    latitude=gps.latitude or 0.0,
                    longitude=gps.longitude or 0.0,
                    altitude=gps.altitude or 0.0,
                    rtk_fix_type=self._map_rtk_fix(gps),
                    hdop=gps.hdop or 99.0,
                    num_satellites=gps.satellites or 0,
                    speed_mps=getattr(gps, 'speed', 0.0) or 0.0,
                    heading=getattr(gps, 'heading', 0.0) or 0.0,
                )

            # IMU
            if sensor_data.imu:
                imu = sensor_data.imu
                frame.imu = IMUData(
                    roll=imu.roll or 0.0,
                    pitch=imu.pitch or 0.0,
                    yaw=imu.yaw or 0.0,
                    linear_accel_x=imu.accel_x or 0.0,
                    linear_accel_y=imu.accel_y or 0.0,
                    linear_accel_z=imu.accel_z or 0.0,
                    angular_vel_x=imu.gyro_x or 0.0,
                    angular_vel_y=imu.gyro_y or 0.0,
                    angular_vel_z=imu.gyro_z or 0.0,
                    calibration_status=imu.calibration_status or "unknown",
                )

            # Ultrasonic - get from sensor manager's ultrasonic interface
            if hasattr(self._sensor_manager, 'ultrasonic'):
                ultrasonic_readings = await self._sensor_manager.ultrasonic.read_ultrasonic()
                if ultrasonic_readings:
                    distances = [r.get('distance_cm', 0.0) for r in ultrasonic_readings if r.get('valid', False)]
                    frame.ultrasonic = UltrasonicData(
                        front_left_cm=ultrasonic_readings[0].get('distance_cm', 0.0) if len(ultrasonic_readings) > 0 else 0.0,
                        front_center_cm=ultrasonic_readings[1].get('distance_cm', 0.0) if len(ultrasonic_readings) > 1 else 0.0,
                        front_right_cm=ultrasonic_readings[2].get('distance_cm', 0.0) if len(ultrasonic_readings) > 2 else 0.0,
                        min_distance_cm=min(distances) if distances else 0.0,
                    )

            # ToF
            if sensor_data.tof_left or sensor_data.tof_right:
                frame.tof = ToFData(
                    left_mm=sensor_data.tof_left.distance if sensor_data.tof_left else 0.0,
                    right_mm=sensor_data.tof_right.distance if sensor_data.tof_right else 0.0,
                )

            # Power
            if sensor_data.power:
                power = sensor_data.power
                frame.power = PowerState(
                    battery_voltage=power.battery_voltage or 0.0,
                    battery_current=power.battery_current or 0.0,
                    battery_soc=self._estimate_soc(power.battery_voltage) if power.battery_voltage else 0.0,
                    charging=False,  # TODO: Get from power service
                )

        except Exception as e:
            logger.warning(f"Sensor data capture error: {e}")

    async def _capture_motor_state(self, frame: MowerDataFrame) -> None:
        """Capture motor and control state from RoboHAT."""
        try:
            if hasattr(self._robohat_service, 'get_motor_state'):
                motor_state = await self._robohat_service.get_motor_state()
                if motor_state:
                    frame.motor = MotorState(
                        wheel_speed_left=motor_state.get('left_speed', 0.0),
                        wheel_speed_right=motor_state.get('right_speed', 0.0),
                        blade_speed=motor_state.get('blade_speed', 0.0),
                        blade_enabled=motor_state.get('blade_enabled', False),
                    )

            if hasattr(self._robohat_service, 'get_control_state'):
                control_state = await self._robohat_service.get_control_state()
                if control_state:
                    frame.action = ControlAction(
                        steering=control_state.get('steering', 0.0),
                        throttle=control_state.get('throttle', 0.0),
                        blade_command=control_state.get('blade_command', False),
                        source=ActionSource.MANUAL,  # Perimeter recording is manual
                    )

        except Exception as e:
            logger.warning(f"Motor state capture error: {e}")

    async def _capture_stereo_images(self, frame: MowerDataFrame) -> None:
        """Capture stereo camera images."""
        try:
            if hasattr(self._sensor_manager, 'stereo_camera') and self._sensor_manager.stereo_camera._driver:
                stereo_frame = await self._sensor_manager.stereo_camera._driver.capture()
                if stereo_frame:
                    # Store images (optionally resized)
                    if stereo_frame.left is not None:
                        frame.stereo_left = self._resize_image(
                            stereo_frame.left, self.config.stereo_resolution
                        )
                    if stereo_frame.right is not None:
                        frame.stereo_right = self._resize_image(
                            stereo_frame.right, self.config.stereo_resolution
                        )
                    # Depth/disparity map if available
                    if hasattr(stereo_frame, 'depth') and stereo_frame.depth is not None:
                        frame.stereo_depth = stereo_frame.depth

        except Exception as e:
            logger.warning(f"Stereo capture error: {e}")

    async def _capture_pi_camera(self, frame: MowerDataFrame) -> None:
        """Capture Pi Camera RGB image."""
        try:
            if hasattr(self._camera_service, 'capture_frame'):
                pi_frame = await self._camera_service.capture_frame()
                if pi_frame and hasattr(pi_frame, 'data') and pi_frame.data is not None:
                    frame.pi_camera_rgb = self._resize_image(
                        pi_frame.data, self.config.pi_camera_resolution
                    )
        except Exception as e:
            logger.warning(f"Pi camera capture error: {e}")

    def _resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image if needed."""
        try:
            import cv2
            if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
                return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        except ImportError:
            pass
        return image

    def _map_rtk_fix(self, gps_reading: Any) -> RTKFixType:
        """Map GPS reading to RTKFixType enum."""
        if hasattr(gps_reading, 'rtk_status') and gps_reading.rtk_status:
            status = str(gps_reading.rtk_status).upper()
            if "FIXED" in status:
                return RTKFixType.FIXED
            elif "FLOAT" in status:
                return RTKFixType.FLOAT
            elif "DGPS" in status:
                return RTKFixType.DGPS

        # Fallback based on satellites
        if hasattr(gps_reading, 'satellites') and gps_reading.satellites:
            if gps_reading.satellites >= 6:
                return RTKFixType.SINGLE

        return RTKFixType.NONE

    def _estimate_soc(self, voltage: Optional[float]) -> float:
        """Estimate battery state of charge from voltage."""
        if voltage is None:
            return 0.0

        min_v, max_v = 11.5, 13.0
        if voltage <= min_v:
            return 0.0
        if voltage >= max_v:
            return 100.0

        return round((voltage - min_v) / (max_v - min_v) * 100.0, 1)

    def _calculate_total_distance(self) -> float:
        """Calculate total distance traveled from GPS points."""
        if len(self._session_frames) < 2:
            return 0.0

        try:
            from math import radians, sin, cos, sqrt, atan2

            total_distance = 0.0
            prev_lat, prev_lon = None, None

            for frame in self._session_frames:
                if frame.gps and frame.gps.latitude != 0.0 and frame.gps.longitude != 0.0:
                    if prev_lat is not None and prev_lon is not None:
                        # Haversine formula
                        R = 6371000  # Earth radius in meters
                        lat1, lat2 = radians(prev_lat), radians(frame.gps.latitude)
                        dlat = lat2 - lat1
                        dlon = radians(frame.gps.longitude - prev_lon)

                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1-a))
                        total_distance += R * c

                    prev_lat = frame.gps.latitude
                    prev_lon = frame.gps.longitude

            return round(total_distance, 2)

        except Exception as e:
            logger.warning(f"Distance calculation error: {e}")
            return 0.0

    async def _save_session(self) -> None:
        """Save session data to disk."""
        if not self._current_session or not self._session_frames:
            return

        try:
            if HAS_MSGPACK:
                # Save as MessagePack
                import msgpack
                data = {
                    "session": self._current_session.to_dict(),
                    "frames": [f.to_dict(include_images=True) for f in self._session_frames],
                }

                file_path = self._current_session.data_file
                with open(file_path, 'wb') as f:
                    msgpack.pack(data, f, use_bin_type=True)

                logger.info(f"Saved session to {file_path} ({file_path.stat().st_size / 1e6:.1f} MB)")
            else:
                # Fallback to JSON (without images)
                import json
                data = {
                    "session": self._current_session.to_dict(),
                    "frames": [f.to_dict(include_images=False) for f in self._session_frames],
                }

                file_path = self.config.storage_dir / f"{self._current_session.session_id}.json"
                with open(file_path, 'w') as f:
                    json.dump(data, f)

                logger.info(f"Saved session to {file_path} (JSON fallback, no images)")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all recorded sessions."""
        sessions = []

        for file_path in self.config.storage_dir.glob("*.msgpack"):
            try:
                if HAS_MSGPACK:
                    import msgpack
                    with open(file_path, 'rb') as f:
                        data = msgpack.unpack(f, raw=False)
                        sessions.append(data.get("session", {}))
            except Exception as e:
                logger.warning(f"Failed to read session {file_path}: {e}")

        for file_path in self.config.storage_dir.glob("*.json"):
            try:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    sessions.append(data.get("session", {}))
            except Exception as e:
                logger.warning(f"Failed to read session {file_path}: {e}")

        return sorted(sessions, key=lambda s: s.get("start_time", ""), reverse=True)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session by ID."""
        # Check MessagePack first
        msgpack_path = self.config.storage_dir / f"{session_id}.msgpack"
        if msgpack_path.exists() and HAS_MSGPACK:
            import msgpack
            with open(msgpack_path, 'rb') as f:
                return msgpack.unpack(f, raw=False)

        # Check JSON fallback
        json_path = self.config.storage_dir / f"{session_id}.json"
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                return json.load(f)

        return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        deleted = False

        for ext in [".msgpack", ".json"]:
            file_path = self.config.storage_dir / f"{session_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True
                logger.info(f"Deleted session: {session_id}")

        return deleted

    def add_frame_callback(self, callback: Callable[[MowerDataFrame], None]) -> None:
        """Add callback for real-time frame notifications."""
        self._frame_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable[[MowerDataFrame], None]) -> None:
        """Remove a frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    @property
    def state(self) -> str:
        """Current recording state."""
        return self._state

    @property
    def current_session(self) -> Optional[RecordingSession]:
        """Current recording session if active."""
        return self._current_session

    @property
    def frame_count(self) -> int:
        """Number of frames in current session."""
        return len(self._session_frames)

    async def health_check(self) -> Dict[str, Any]:
        """Return health status for /health endpoint."""
        return {
            "service": "perimeter_recorder",
            "initialized": self.initialized,
            "state": self._state,
            "current_session": self._current_session.session_id if self._current_session else None,
            "current_session_name": self._current_session.name if self._current_session else None,
            "frame_count": len(self._session_frames),
            "total_frames_recorded": self._total_frames_recorded,
            "total_sessions": self._total_sessions,
            "target_fps": self.config.target_fps,
            "avg_capture_time_ms": round(self._avg_capture_time_ms, 2),
            "storage_dir": str(self.config.storage_dir),
            "msgpack_available": HAS_MSGPACK,
        }


# Singleton instance management
_recorder_instance: Optional[PerimeterRecordingService] = None


def get_perimeter_recorder() -> PerimeterRecordingService:
    """Get or create the PerimeterRecordingService singleton."""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = PerimeterRecordingService()
    return _recorder_instance


def set_perimeter_recorder(recorder: PerimeterRecordingService) -> None:
    """Set the PerimeterRecordingService singleton (for dependency injection)."""
    global _recorder_instance
    _recorder_instance = recorder


__all__ = [
    "PerimeterRecordingService",
    "RecordingConfig",
    "RecordingState",
    "get_perimeter_recorder",
    "set_perimeter_recorder",
]
