#!/usr/bin/env python3
"""Test ELP-USB960P2CAM-V90 stereo camera capture.

This script verifies the stereo camera is working correctly by:
1. Finding the stereo camera device
2. Capturing a frame at full resolution (2560x960)
3. Splitting into left/right frames
4. Saving test images
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2


def find_stereo_camera(max_index: int = 10) -> int | None:
    """Find the stereo camera device index.

    Args:
        max_index: Maximum device index to check

    Returns:
        Device index if found, None otherwise
    """
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue

        # Try to set stereo resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        ret, frame = cap.read()
        cap.release()

        if ret and frame.shape[1] >= 2560:
            return idx

    return None


def test_stereo_camera() -> bool:
    """Test stereo camera capture.

    Returns:
        True if test passed, False otherwise
    """
    print("Searching for stereo camera...")
    device_idx = find_stereo_camera()

    if device_idx is None:
        print("ERROR: Stereo camera not found")
        print("Make sure the ELP-USB960P2CAM-V90 is connected")
        return False

    print(f"Stereo camera found at /dev/video{device_idx}")

    # Open camera
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open /dev/video{device_idx}")
        return False

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to capture frame")
        cap.release()
        return False

    print(f"Frame shape: {frame.shape}")

    # Verify resolution
    if frame.shape[1] != 2560 or frame.shape[0] != 960:
        print(f"WARNING: Expected 2560x960, got {frame.shape[1]}x{frame.shape[0]}")

    # Split into left/right
    mid = frame.shape[1] // 2
    left = frame[:, :mid]
    right = frame[:, mid:]

    print(f"Left eye: {left.shape[1]}x{left.shape[0]}")
    print(f"Right eye: {right.shape[1]}x{right.shape[0]}")

    # Save test images
    output_dir = Path("/tmp")
    cv2.imwrite(str(output_dir / "stereo_left.jpg"), left)
    cv2.imwrite(str(output_dir / "stereo_right.jpg"), right)
    cv2.imwrite(str(output_dir / "stereo_combined.jpg"), frame)

    print(f"\nSaved test images to {output_dir}/")
    print("  - stereo_left.jpg")
    print("  - stereo_right.jpg")
    print("  - stereo_combined.jpg")

    # Test frame rate
    print("\nTesting frame rate (5 seconds)...")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 5.0:
        ret, _ = cap.read()
        if ret:
            frame_count += 1

    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    print(f"Captured {frame_count} frames in {elapsed:.1f}s = {fps:.1f} FPS")

    cap.release()

    # Summary
    print("\n=== TEST SUMMARY ===")
    print(f"Device: /dev/video{device_idx}")
    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Left/Right: {left.shape[1]}x{left.shape[0]} each")
    print(f"Frame rate: {fps:.1f} FPS")
    print("Status: PASS")

    return True


if __name__ == "__main__":
    success = test_stereo_camera()
    sys.exit(0 if success else 1)
