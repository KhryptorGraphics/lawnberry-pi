"""Stereo Camera Driver for ELP-USB960P2CAM-V90.

Provides async lifecycle management and frame capture for the USB stereo camera.
SIM_MODE yields synthetic stereo frames for testing without hardware.

The stereo camera outputs a combined 2560x960 frame (1280x960 per eye).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.simulation import is_simulation_mode
from ..base import HardwareDriver

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class StereoFrame:
    """Container for stereo camera frame data."""

    left: np.ndarray  # 1280x960 BGR
    right: np.ndarray  # 1280x960 BGR
    combined: np.ndarray  # 2560x960 BGR
    timestamp: float
    frame_id: int


class StereoCameraDriver(HardwareDriver):
    """Driver for ELP-USB960P2CAM-V90 stereo camera.

    Captures synchronized stereo frames at 2560x960 resolution.
    Supports SIM_MODE for testing without hardware.
    """

    # Default camera settings
    DEFAULT_WIDTH = 2560
    DEFAULT_HEIGHT = 960
    DEFAULT_FPS = 30

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        self._device_index: int = config.get("device_index", 0) if config else 0
        self._width: int = config.get("width", self.DEFAULT_WIDTH) if config else self.DEFAULT_WIDTH
        self._height: int = config.get("height", self.DEFAULT_HEIGHT) if config else self.DEFAULT_HEIGHT
        self._fps: int = config.get("fps", self.DEFAULT_FPS) if config else self.DEFAULT_FPS

        self._cap: Any = None  # cv2.VideoCapture
        self._frame_count: int = 0
        self._last_frame: StereoFrame | None = None
        self._last_capture_time: float | None = None
        self._sim_cycle: int = 0

    async def initialize(self) -> None:
        """Initialize stereo camera hardware."""
        if is_simulation_mode():
            self.initialized = True
            return

        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available for stereo camera")

        # Find stereo camera
        device_idx = self._find_stereo_camera()
        if device_idx is None:
            raise RuntimeError("Stereo camera not found")

        self._device_index = device_idx
        self._cap = cv2.VideoCapture(device_idx)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stereo camera at /dev/video{device_idx}")

        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Verify resolution
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w != self._width or actual_h != self._height:
            raise RuntimeError(
                f"Stereo camera resolution mismatch: expected {self._width}x{self._height}, "
                f"got {actual_w}x{actual_h}"
            )

        self.initialized = True

    def _find_stereo_camera(self, max_index: int = 10) -> int | None:
        """Find the stereo camera device index."""
        if not OPENCV_AVAILABLE:
            return None

        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None and frame.shape[1] >= self._width:
                return idx

        return None

    async def start(self) -> None:
        """Start camera capture."""
        if not self.initialized:
            raise RuntimeError("Camera not initialized")
        self.running = True

    async def stop(self) -> None:
        """Stop camera and release resources."""
        self.running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    async def health_check(self) -> dict[str, Any]:
        """Return camera health status."""
        return {
            "sensor": "stereo_camera",
            "model": "ELP-USB960P2CAM-V90",
            "initialized": self.initialized,
            "running": self.running,
            "device_index": self._device_index,
            "resolution": f"{self._width}x{self._height}",
            "frame_count": self._frame_count,
            "last_capture_age_s": (
                (time.time() - self._last_capture_time)
                if self._last_capture_time
                else None
            ),
            "simulation": is_simulation_mode(),
        }

    async def capture(self) -> StereoFrame | None:
        """Capture a stereo frame.

        Returns:
            StereoFrame with left, right, and combined images, or None on failure.
        """
        if not self.initialized:
            return None

        if is_simulation_mode():
            return self._generate_sim_frame()

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # Split into left/right
        mid = frame.shape[1] // 2
        left = frame[:, :mid].copy()
        right = frame[:, mid:].copy()

        self._frame_count += 1
        self._last_capture_time = time.time()

        stereo_frame = StereoFrame(
            left=left,
            right=right,
            combined=frame,
            timestamp=self._last_capture_time,
            frame_id=self._frame_count,
        )
        self._last_frame = stereo_frame

        return stereo_frame

    def _generate_sim_frame(self) -> StereoFrame:
        """Generate a simulated stereo frame for testing."""
        # Create synthetic gradient images
        h, w = self._height, self._width // 2

        # Left image: horizontal gradient
        left = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(w):
            left[:, i] = [int(255 * i / w), 128, int(255 * (1 - i / w))]

        # Right image: shifted version (simulated parallax)
        right = np.zeros((h, w, 3), dtype=np.uint8)
        shift = 20 + int(10 * np.sin(self._sim_cycle / 30))
        for i in range(w):
            shifted_i = (i + shift) % w
            right[:, i] = [int(255 * shifted_i / w), 128, int(255 * (1 - shifted_i / w))]

        # Add some variation
        noise = np.random.randint(0, 10, (h, w, 3), dtype=np.uint8)
        left = np.clip(left.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        right = np.clip(right.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        combined = np.hstack([left, right])

        self._sim_cycle += 1
        self._frame_count += 1
        self._last_capture_time = time.time()

        stereo_frame = StereoFrame(
            left=left,
            right=right,
            combined=combined,
            timestamp=self._last_capture_time,
            frame_id=self._frame_count,
        )
        self._last_frame = stereo_frame

        return stereo_frame

    @property
    def last_frame(self) -> StereoFrame | None:
        """Get the most recently captured frame."""
        return self._last_frame

    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count


__all__ = ["StereoCameraDriver", "StereoFrame"]
