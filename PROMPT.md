# Task: THE ULTIMATE AUTONOMOUS MOWING AI SYSTEM

---

## RALPH ORCHESTRATOR META-INSTRUCTIONS

**YOU HAVE FULL AUTONOMY TO COMPLETE THIS PROJECT.**

### Dynamic Task Management
- **CREATE** new BEAD tasks when you discover additional work needed
- **EDIT** existing BEAD tasks to refine scope, update status, or clarify requirements
- **SPLIT** large tasks into smaller sub-tasks (e.g., BEAD-031a, BEAD-031b)
- **MERGE** tasks that are better handled together
- **REORDER** task priorities based on dependencies discovered during implementation
- **MARK COMPLETE** tasks as you finish them by changing `**Status**: pending` to `**Status**: complete`

### Adaptive Iteration Strategy
1. **On Success**: Mark task complete, update this PROMPT.md, proceed to next task
2. **On Partial Success**: Document what worked, create follow-up tasks for remaining work
3. **On Failure**: Analyze the problem, create diagnostic/fix tasks, retry with new approach
4. **On Blocker**: Document the blocker, skip to non-dependent tasks, return when unblocked

### Completion Criteria
**ITERATE UNTIL ALL OF THE FOLLOWING ARE TRUE:**
- [ ] All hardware sensors integrated and streaming telemetry
- [ ] All drivers implemented following project patterns (HardwareDriver base class)
- [ ] Safety interlocks working for all sensors
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] Code follows existing project conventions (check similar files first)
- [ ] Integration with LawnBerry Pi backend complete

### Self-Modification Rules
1. Always update this PROMPT.md file after completing tasks
2. Add learnings and gotchas to relevant BEAD descriptions
3. Create new BEADs for unexpected dependencies or issues
4. Remove or mark as `**Status**: skipped` tasks that become irrelevant
5. Document any architectural decisions in the Notes section

### Memory & Progress Tracking
**USE MCP TOOLS TO MAINTAIN PERSISTENT MEMORY:**
- **Serena MCP** (`mcp__serena__write_memory`): Write progress summaries, architectural decisions, and learnings to Serena memories periodically (every 3-5 completed tasks)
- **Cipher MCP** (`mcp__cipher__cipher_bash`): Use for complex bash operations when needed
- **Update memories when**:
  - Completing a major phase (e.g., finishing Phase 0)
  - Discovering important patterns or gotchas
  - Making architectural decisions
  - Encountering and resolving blockers
- **Memory naming convention**: `lawnberry_progress_YYYYMMDD`, `lawnberry_architecture`, `lawnberry_gotchas`

### Execution Philosophy
- **Morphological Adaptation**: Tasks evolve based on what you learn
- **Continuous Integration**: Test after each significant change
- **Fail Fast, Learn Fast**: Try approaches, document failures, pivot quickly
- **Code Quality**: Follow existing patterns in the codebase (check similar drivers/services first)

---

## Vision
Build a revolutionary autonomous mowing system that leverages cutting-edge AI/ML infrastructure to achieve human-expert-level lawn care through simulation-trained vision-language-action models, world models, and reinforcement learning.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING INFRASTRUCTURE (NVIDIA Thor)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Isaac Sim    │  │    Cosmos     │  │  LawnMower    │  │    World     │ │
│  │  Environment  │──│   Augment     │──│     VLA       │──│    Model     │ │
│  │  (Simulation) │  │  (Synthetic)  │  │   (Policy)    │  │  (Dynamics)  │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│           │                 │                  │                  │         │
│           └─────────────────┴──────────────────┴──────────────────┘         │
│                                      │                                       │
│                           Model Distillation                                 │
│                                      ▼                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              HaLow WiFi Link
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EDGE INFERENCE (Raspberry Pi 5)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │   Hailo 8L    │  │    Stereo     │  │     GPS       │  │  Ultrasonic  │ │
│  │   (13 TOPS)   │──│    Camera     │──│   LC29H-DA    │──│    Array     │ │
│  │   INT8 Model  │  │  ELP-960P     │  │   RTK/GNSS    │  │   HC-SR04    │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│                              │                                               │
│                              ▼                                               │
│                    LawnBerry Pi Backend (FastAPI)                            │
│                              │                                               │
│                              ▼                                               │
│                    Motor Control (RoboHAT MM1)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

### Training Server: NVIDIA Jetson AGX Thor
- **GPU**: Blackwell architecture, 2070 TFLOPS (FP4)
- **Memory**: 128GB unified RAM
- **Storage**: 8TB NVMe array
- **Network**: 10GbE + HaLow bridge
- **Purpose**: Isaac Sim, Cosmos, VLA training, World Model, RL

### Edge Device: Raspberry Pi 5 + Hailo 8L
- **CPU**: Broadcom BCM2712, Cortex-A76 @ 2.4GHz
- **AI Accelerator**: Hailo 8L (13 TOPS INT8)
- **Memory**: 8GB LPDDR4X
- **Storage**: 128GB microSD + USB SSD
- **Purpose**: Real-time inference, sensor fusion, motor control

### Sensors
| Sensor | Model | Interface | Purpose |
|--------|-------|-----------|---------|
| Stereo Camera | ELP-USB960P2CAM-V90 | USB 2.0 UVC | Depth perception, obstacle detection |
| Pi Camera | Raspberry Pi Camera 2 | CSI | High-res lawn analysis |
| GPS/RTK | LC29H(DA) HAT | UART 115200 | Centimeter positioning |
| Ultrasonic | HC-SR04 x3 | GPIO | Close-range obstacle detection |
| IMU | BNO085 | UART4 | Orientation, tilt safety |
| ToF | VL53L0X x2 | I2C | Blade height sensing |

### Communication
- **HaLow WiFi**: Long-range (1km+) 900MHz link between mower and Thor
- **Standard WiFi**: Backup 2.4/5GHz connection
- **NTRIP**: RTK corrections via cellular/WiFi

---

## Environment Requirements
- Raspberry Pi 5 (8GB+) with Pi OS Bookworm 64-bit
- Python 3.11 with pip/venv (NOT conda)
- Deactivate any conda environments before starting
- NVIDIA Jetson AGX Thor with JetPack 7.0+

---

## Beads Task List

### PHASE 0: ENVIRONMENT & HARDWARE FOUNDATION

---

### BEAD-001: Environment Verification
**Status**: complete
**Priority**: critical
**Description**: Verify Python environment and deactivate conda
**Completed**: 2026-01-08 - Recreated venv with Python 3.11.2, installed all base dependencies successfully
```bash
# Deactivate any conda environments
conda deactivate 2>/dev/null || true
for i in $(seq 1 10); do conda deactivate 2>/dev/null || break; done

# Verify Python version
python3 --version  # Must be 3.11.x

# Activate project venv
cd /home/kp/repos/lawnberry_pi
source .venv/bin/activate || python3 -m venv .venv && source .venv/bin/activate

# Install base dependencies
pip install -e ".[hardware]"
```
**Acceptance**: Python 3.11.x active, no conda, venv activated

---

### BEAD-002: Install Hardware Dependencies
**Status**: complete
**Priority**: critical
**Description**: Install all required packages for new hardware
**Completed**: 2026-01-08 - All system packages were already installed. Python packages installed successfully. Enabled system-site-packages in venv for libcamera access. Fixed numpy version to 1.26.4 (project requirement <2.0). Installed opencv-python-headless 4.10.0 for numpy compatibility.
```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    gpsd gpsd-clients \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    i2c-tools

# Python packages
pip install \
    picamera2 \
    gpiozero \
    lgpio \
    RPi.GPIO \
    pynmea2 \
    opencv-python-headless
```
**Acceptance**: All packages installed without errors

---

### BEAD-003: Enable Raspberry Pi Interfaces
**Status**: complete
**Priority**: critical
**Description**: Configure Pi interfaces for all hardware
**Completed**: 2026-01-08 - All interfaces verified working:
- Serial: Removed console from cmdline.txt, created udev rule for dialout group access
- I2C: i2c-1 and i2c-11 available
- SPI: spidev0.0, spidev0.1, spidev10.0 available
- Camera: Pi Camera 2 (imx219) detected via rpicam-hello
- **NOTE**: On Pi 5, use `/dev/ttyAMA0` (not `/dev/ttyS0`) for GPS serial
```bash
# Enable required interfaces via raspi-config
sudo raspi-config nonint do_serial_hw 0      # Enable serial port hardware
sudo raspi-config nonint do_serial_cons 1    # Disable serial console
sudo raspi-config nonint do_i2c 0            # Enable I2C
sudo raspi-config nonint do_spi 0            # Enable SPI
sudo raspi-config nonint do_camera 0         # Enable camera

# Reboot required after changes
```
**Acceptance**: All interfaces enabled, system rebooted

---

### BEAD-010: USB Stereo Camera - Detection
**Status**: complete
**Priority**: high
**Description**: Verify USB stereo camera is detected
**Completed**: 2026-01-08 - Camera detected and verified:
- USB: Bus 002 Device 002: ID 32e4:9750 3D USB Camera
- Video device: /dev/video0
- Resolution: 2560x960 combined (1280x960 per eye)
- Test images saved to /tmp/stereo_*.jpg
```bash
# Check USB devices
lsusb | grep -i "3D USB Camera"
# Expected: Bus XXX Device YYY: ID 32e4:9750 3D USB Camera

# Check video devices
ls -la /dev/video*
v4l2-ctl --list-devices
```
**Acceptance**: Camera detected as /dev/videoN

---

### BEAD-011: USB Stereo Camera - Test Capture
**Status**: complete
**Priority**: high
**Description**: Test stereo camera frame capture
**Completed**: 2026-01-08 - Created scripts/test_stereo_camera.py with:
- Auto-detection of stereo camera device
- Resolution setting to 2560x960
- Left/right frame splitting
- Test image saving
- Frame rate measurement (5.5 FPS at full res via USB 2.0)
```python
#!/usr/bin/env python3
"""Test ELP-USB960P2CAM-V90 stereo camera capture."""
import cv2
import sys

def test_stereo_camera():
    # Find stereo camera device
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame.shape[1] == 2560:  # Combined stereo width
                print(f"Stereo camera found at /dev/video{idx}")
                print(f"Frame shape: {frame.shape}")

                # Split into left/right
                left = frame[:, :1280]
                right = frame[:, 1280:]
                print(f"Left: {left.shape}, Right: {right.shape}")

                # Save test images
                cv2.imwrite("/tmp/stereo_left.jpg", left)
                cv2.imwrite("/tmp/stereo_right.jpg", right)
                print("Saved test images to /tmp/")

                cap.release()
                return True
        cap.release()

    print("ERROR: Stereo camera not found")
    return False

if __name__ == "__main__":
    sys.exit(0 if test_stereo_camera() else 1)
```
**Acceptance**: Test images saved, left/right frames 1280x960

---

### BEAD-012: USB Stereo Camera - Driver Implementation
**Status**: complete
**Priority**: high
**Description**: Create stereo camera driver following project patterns
**Completed**: 2026-01-08 - Created backend/src/drivers/sensors/stereo_camera_driver.py:
- HardwareDriver base class inheritance with async lifecycle
- SIM_MODE support with synthetic gradient frames
- Left/right frame splitting (1280x960 each)
- StereoFrame dataclass with timestamp and frame_id
- Auto-detection of stereo camera device
- Health check with frame count and capture age
- Unit tests passing (tests/unit/test_stereo_camera_driver.py)
- Hardware tests verified working
**Acceptance**: Driver passes unit tests, integrates with telemetry

---

### BEAD-020: Pi Camera 2 - Detection
**Status**: complete
**Priority**: high
**Description**: Verify Raspberry Pi Camera 2 is detected
**Completed**: 2026-01-08 - Camera detected via rpicam-hello:
- Model: imx219 (Sony IMX219 8MP sensor)
- Max resolution: 3280x2464 at 21 fps
- 1920x1080 at 47 fps, 640x480 at 103 fps
- Test image captured to /tmp/picam_test.jpg (943KB)
```bash
# Check camera detection
rpicam-hello --list-cameras  # Note: On Bookworm, use rpicam-* not libcamera-*

# Test capture
rpicam-still -o /tmp/picam_test.jpg
```
**Acceptance**: Camera detected, test image captured

---

### BEAD-021: Pi Camera 2 - Integration
**Status**: complete
**Priority**: high
**Description**: Integrate Pi Camera 2 with existing camera service
**Completed**: 2026-01-08 - Integration verified:
- camera_stream_service.py already supports PiCamera2 + OpenCV backends
- Added camera configuration to config/hardware.yaml:
  - primary: Pi Camera 2 (picamera2) at 1920x1080@30fps
  - stereo: USB ELP stereo camera at 2560x960@30fps
- Verified picamera2 capture works (1920x1080 BGR)
- Both cameras detected by libcamera
**Acceptance**: Both cameras selectable via config

---

### BEAD-030: LC29H GPS - Hardware Setup
**Status**: complete
**Priority**: critical
**Description**: Configure LC29H(DA) GPS/RTK HAT
**Completed**: 2026-01-08 - GPS verified working:
- Device: /dev/ttyAMA0 at 115200 baud
- Fix Quality: DGPS/RTK (quality 2)
- Satellites: 19+ visible
- HDOP: 0.73 (excellent)
- Masked serial-getty@ttyAMA0 service to free port
- NMEA sentences: GNGGA, GNRMC, GNGLL, GNVTG, GNGSA, GPGSV, etc.
```bash
# Verify HAT is connected
ls -la /dev/ttyAMA0  # Note: Pi 5 uses ttyAMA0, not ttyS0

# Test serial communication
sudo apt-get install -y minicom
minicom -D /dev/ttyAMA0 -b 115200

# Should see NMEA sentences: $GNRMC, $GNGGA, etc.
```
**Acceptance**: NMEA data streaming from GPS module

---

### BEAD-031: LC29H GPS - Driver Implementation
**Status**: pending
**Priority**: critical
**Description**: Create LC29H GPS driver (replacing ZED-F9P)
Create `backend/src/drivers/sensors/lc29h_driver.py`:
- Inherit from HardwareDriver base class
- NMEA parsing (GGA, RMC, GST sentences)
- PQTM/PAIR command support for configuration
- RTK status detection (Float, Fixed)
- NTRIP corrections forwarding
- SIM_MODE with deterministic coordinates
- Auto-baud detection (9600, 115200, 460800)
**Acceptance**: GPS readings in telemetry, RTK status reported

---

### BEAD-032: LC29H GPS - NTRIP Configuration
**Status**: pending
**Priority**: high
**Description**: Configure RTK corrections via NTRIP
Update `.env` with NTRIP credentials:
```
NTRIP_HOST=<your_caster>
NTRIP_PORT=2101
NTRIP_MOUNTPOINT=<mountpoint>
NTRIP_USERNAME=<user>
NTRIP_PASSWORD=<pass>
NTRIP_SERIAL_DEVICE=/dev/ttyS0
NTRIP_SERIAL_BAUD=115200
```
Modify `ntrip_client.py` for LC29H compatibility.
**Acceptance**: RTK Fix achieved with corrections

---

### BEAD-033: Update Hardware Config for LC29H
**Status**: pending
**Priority**: high
**Description**: Update hardware configuration
Modify `config/hardware.yaml`:
```yaml
gps:
  type: LC29H-DA
  uart_device: /dev/ttyS0
  baudrate: 115200
  ntrip_enabled: true
```
Modify `backend/src/models/hardware_config.py`:
- Add GPSType.LC29H_DA enum value
- Remove/deprecate ZED-F9P references
**Acceptance**: Config loads without errors

---

### BEAD-040: Ultrasonic Sensors - Wiring Guide
**Status**: pending
**Priority**: high
**Description**: Document ultrasonic sensor wiring

**IMPORTANT**: Wire the sensors AFTER the software is configured and tested in simulation mode.

Create `docs/ultrasonic-wiring-guide.md` with:

## Components Required
- 3x HC-SR04 ultrasonic sensors
- 6x 1k ohm resistors (voltage dividers for ECHO pins)
- 3x 2k ohm resistors (voltage dividers for ECHO pins)
- Jumper wires
- Breadboard or custom PCB

## GPIO Pin Assignments
| Sensor | Position | VCC | TRIG | ECHO | GND |
|--------|----------|-----|------|------|-----|
| 1 | Front-Left | 5V (Pin 2) | GPIO4 (Pin 7) | GPIO17 (Pin 11)* | GND (Pin 6) |
| 2 | Front-Center | 5V (Pin 4) | GPIO27 (Pin 13) | GPIO10 (Pin 19)* | GND (Pin 14) |
| 3 | Front-Right | 5V (Pin 17) | GPIO11 (Pin 23) | GPIO9 (Pin 21)* | GND (Pin 20) |

*ECHO pins require voltage divider (5V to 3.3V)

## Voltage Divider Circuit (per ECHO pin)
```
HC-SR04 ECHO ---+--- 1k ohm ---+--- Pi GPIO (ECHO)
                |              |
                +--- 2k ohm ---+--- GND
```
Output: 5V x (2k / 3k) = 3.3V (safe for Pi GPIO)

## Installation Steps
1. Mount sensors on front bumper (spacing: 15-20cm apart)
2. Wire VCC to 5V rail
3. Wire GND to ground rail
4. Connect TRIG pins directly to GPIO
5. Connect ECHO pins through voltage dividers
6. Secure wiring with cable ties

**Acceptance**: Documentation complete with diagrams

---

### BEAD-041: Ultrasonic Sensors - Driver Implementation
**Status**: pending
**Priority**: high
**Description**: Create HC-SR04 driver
Create `backend/src/drivers/sensors/ultrasonic_driver.py`:
```python
"""HC-SR04 Ultrasonic Sensor Driver for LawnBerry Pi."""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import Any
from drivers.base import HardwareDriver
from core.simulation import is_simulation_mode

@dataclass
class UltrasonicReading:
    sensor_id: str  # "front_left", "front_center", "front_right"
    distance_cm: float
    timestamp: float
    valid: bool = True

class UltrasonicDriver(HardwareDriver):
    """Driver for 3x HC-SR04 ultrasonic sensors."""

    SENSORS = {
        "front_left": {"trig": 4, "echo": 17},
        "front_center": {"trig": 27, "echo": 10},
        "front_right": {"trig": 11, "echo": 9},
    }
    SPEED_OF_SOUND = 34300  # cm/s at 20C

    async def initialize(self) -> None:
        if is_simulation_mode():
            self.initialized = True
            return
        # Initialize GPIO pins using lgpio or RPi.GPIO
        # ...

    async def read_distance(self, sensor_id: str) -> UltrasonicReading:
        """Read distance from specified sensor."""
        # Implementation with timeout protection
        # ...

    async def read_all(self) -> list[UltrasonicReading]:
        """Read all three sensors."""
        # ...

    async def health_check(self) -> dict[str, Any]:
        """Return driver health status."""
        # ...
```
**Acceptance**: All three sensors return valid readings

---

### BEAD-042: Ultrasonic - Test Script
**Status**: pending
**Priority**: medium
**Description**: Create ultrasonic sensor test script
Create `scripts/test_ultrasonic.py`:
- Test each sensor individually
- Verify voltage dividers working (no GPIO damage)
- Display distance readings in real-time
- Warn if readings seem invalid
**Acceptance**: All sensors report distances 2-400cm

---

### BEAD-043: Ultrasonic - Safety Integration
**Status**: pending
**Priority**: critical
**Description**: Integrate ultrasonic sensors with safety system
Modify `backend/src/safety/safety_triggers.py`:
- Add ultrasonic obstacle threshold (default: 30cm)
- Trigger e-stop if any sensor < threshold
- Log which sensor triggered
Modify `config/limits.yaml`:
```yaml
ultrasonic_obstacle_distance_cm: 30
ultrasonic_enabled: true
```
**Acceptance**: Blade stops when obstacle < 30cm from any sensor

---

### BEAD-050: Sensor Manager Integration
**Status**: pending
**Priority**: high
**Description**: Register all new sensors with SensorManager
Modify `backend/src/services/sensor_manager.py`:
- Add UltrasonicSensorInterface
- Add StereoCameraSensorInterface
- Update LC29H GPS registration
- Ensure no I2C/UART conflicts
**Acceptance**: All sensors appear in /api/v2/sensors/status

---

### BEAD-051: Telemetry Integration
**Status**: pending
**Priority**: high
**Description**: Add new sensors to WebSocket telemetry
Modify `backend/src/services/websocket_hub.py`:
- Add ultrasonic distance streaming
- Add stereo camera frame metadata
- Update GPS telemetry for LC29H fields
**Acceptance**: All sensor data visible in frontend dashboard

---

### BEAD-060: Hardware Self-Test Update
**Status**: pending
**Priority**: medium
**Description**: Update self-test for new hardware
Modify `backend/src/services/hw_selftest.py`:
- Add LC29H GPS connectivity test
- Add ultrasonic sensor response test
- Add stereo camera detection test
- Add Pi Camera 2 detection test
**Acceptance**: `ralph status` shows all hardware healthy

---

### PHASE 1: DATA COLLECTION INFRASTRUCTURE

---

### BEAD-100: Hailo 8L Installation
**Status**: pending
**Priority**: critical
**Description**: Install and configure Hailo 8L AI accelerator
```bash
# Install Hailo runtime
wget https://hailo.ai/downloads/hailo-8l-pcie-driver.deb
sudo dpkg -i hailo-8l-pcie-driver.deb

# Install HailoRT
pip install hailort

# Verify installation
hailortcli fw-control identify
# Expected: Hailo-8L, 13 TOPS
```
Create `backend/src/drivers/ai/hailo_driver.py`:
- HailoRT integration
- Model loading/inference
- Performance metrics
**Acceptance**: Hailo device detected, sample inference works

---

### BEAD-101: HaLow WiFi Bridge Setup
**Status**: pending
**Priority**: high
**Description**: Configure long-range HaLow WiFi link
```bash
# HaLow operates at 900MHz with 1km+ range
# Install HaLow driver and utilities
# Configure bridge mode between mower and Thor server
```
Create network configuration for:
- Mower side: WiFi client to Thor's HaLow AP
- Thor side: HaLow AP + bridge to training network
- Fallback: Standard 2.4GHz WiFi
**Acceptance**: Reliable link at 500m+ range, <100ms latency

---

### BEAD-102: Multi-Modal Data Frame Definition
**Status**: pending
**Priority**: critical
**Description**: Define unified sensor data structure for training
Create `backend/src/models/training_data.py`:
```python
"""Multi-modal data frame for AI training."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from typing import Optional

@dataclass
class MowerDataFrame:
    """Single frame of multi-modal mower sensor data."""
    timestamp: datetime
    session_id: str
    frame_id: int

    # Stereo camera (ELP-USB960P2CAM-V90)
    stereo_left: np.ndarray      # 1280x960 BGR
    stereo_right: np.ndarray     # 1280x960 BGR
    stereo_depth: Optional[np.ndarray] = None  # Computed disparity

    # Pi Camera 2 (high-res lawn analysis)
    pi_camera_rgb: np.ndarray    # 1920x1080 BGR

    # GPS (LC29H-DA RTK)
    latitude: float
    longitude: float
    altitude: float
    rtk_fix_type: str            # "None", "Float", "Fixed"
    hdop: float
    num_satellites: int

    # IMU (BNO085)
    roll: float                  # degrees
    pitch: float                 # degrees
    yaw: float                   # degrees (heading)
    linear_accel: tuple[float, float, float]

    # Ultrasonic (HC-SR04 x3)
    ultrasonic_left: float       # cm
    ultrasonic_center: float     # cm
    ultrasonic_right: float      # cm

    # ToF (VL53L0X x2)
    tof_left: float              # mm (blade height)
    tof_right: float             # mm (blade height)

    # Motor state
    wheel_speed_left: float      # RPM
    wheel_speed_right: float     # RPM
    blade_speed: float           # RPM
    blade_enabled: bool

    # Actions taken
    action_steering: float       # -1.0 to 1.0
    action_throttle: float       # 0.0 to 1.0
    action_blade: bool

    # Metadata
    battery_voltage: float
    battery_soc: float           # State of charge %
    mower_state: str             # "mowing", "turning", "returning", etc.
```
**Acceptance**: Data structure validated, serialization works

---

### BEAD-103: Perimeter Recording Service
**Status**: pending
**Priority**: critical
**Description**: Record training data during manual mowing sessions
Create `backend/src/services/perimeter_recorder.py`:
```python
"""Record multi-modal sensor data for training."""
from __future__ import annotations
import asyncio
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py
from models.training_data import MowerDataFrame

class PerimeterRecorder:
    """Records sensor data during perimeter teaching sessions."""

    def __init__(self, output_dir: Path = Path("/data/recordings")):
        self.output_dir = output_dir
        self.session_id: str = ""
        self.recording: bool = False
        self.frame_count: int = 0
        self.h5_file: Optional[h5py.File] = None

    async def start_session(self, session_name: str) -> str:
        """Start a new recording session."""
        self.session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{session_name}"
        session_path = self.output_dir / f"{self.session_id}.h5"
        self.h5_file = h5py.File(session_path, "w")

        # Create datasets for each modality
        self._create_datasets()
        self.recording = True
        self.frame_count = 0
        return self.session_id

    async def record_frame(self, frame: MowerDataFrame) -> None:
        """Record a single multi-modal frame."""
        if not self.recording:
            return

        # Write to HDF5 datasets
        # ...
        self.frame_count += 1

    async def stop_session(self) -> dict:
        """Stop recording and finalize session."""
        self.recording = False
        if self.h5_file:
            self.h5_file.close()
        return {
            "session_id": self.session_id,
            "frame_count": self.frame_count,
            "duration_seconds": self._get_duration()
        }
```
**Acceptance**: Records 10+ FPS, files playable, no data loss

---

### BEAD-104: Recording API Endpoints
**Status**: pending
**Priority**: high
**Description**: Add REST API for recording control
Create `backend/src/api/routers/recording.py`:
```python
"""Recording API for training data collection."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2/recording", tags=["recording"])

class RecordingSession(BaseModel):
    session_name: str

class RecordingStatus(BaseModel):
    recording: bool
    session_id: str
    frame_count: int
    duration_seconds: float
    disk_usage_mb: float

@router.post("/start")
async def start_recording(session: RecordingSession) -> dict:
    """Start a new recording session."""
    # ...

@router.post("/stop")
async def stop_recording() -> dict:
    """Stop current recording session."""
    # ...

@router.get("/status")
async def get_recording_status() -> RecordingStatus:
    """Get current recording status."""
    # ...

@router.get("/sessions")
async def list_sessions() -> list[dict]:
    """List all recorded sessions."""
    # ...
```
**Acceptance**: API functional, recordings manageable via frontend

---

### BEAD-105: Data Upload to Thor
**Status**: pending
**Priority**: high
**Description**: Stream recorded data to Thor training server
Create `backend/src/services/thor_uploader.py`:
- Async upload over HaLow WiFi
- Resume interrupted transfers
- Compression (LZ4) for bandwidth
- Integrity verification (SHA256)
- Upload queue management
**Acceptance**: Reliable upload, handles disconnections

---

### PHASE 2: NVIDIA THOR TRAINING INFRASTRUCTURE

---

### BEAD-200: Thor Server Setup
**Status**: pending
**Priority**: critical
**Description**: Configure NVIDIA Jetson AGX Thor for training
```bash
# On Thor server
# Install JetPack 7.0+
sudo apt-get update && sudo apt-get upgrade

# Install NVIDIA Isaac Sim 6.0
# Install NVIDIA Cosmos 2.5
# Install PyTorch with CUDA support

# Create training environment
python3 -m venv /opt/lawnberry/venv
source /opt/lawnberry/venv/bin/activate
pip install torch torchvision nvidia-isaac-sim nvidia-cosmos
```
Create `thor/` directory structure:
```
thor/
├── config/
│   ├── training.yaml
│   └── simulation.yaml
├── models/
│   ├── vla/
│   ├── world_model/
│   └── distilled/
├── data/
│   ├── real/
│   ├── synthetic/
│   └── augmented/
└── scripts/
    ├── train_vla.py
    ├── train_world_model.py
    └── distill_hailo.py
```
**Acceptance**: Thor operational, all frameworks installed

---

### BEAD-201: Isaac Sim Lawn Environment
**Status**: pending
**Priority**: critical
**Description**: Create photorealistic lawn simulation in Isaac Sim
Create `thor/simulation/lawn_env.py`:
```python
"""Isaac Sim lawn mowing environment."""
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
import numpy as np

class LawnSimEnvironment:
    """Photorealistic lawn simulation for training."""

    def __init__(self, config_path: str):
        self.world = World(stage_units_in_meters=1.0)
        self.config = self._load_config(config_path)

        # Environment parameters
        self.lawn_types = ["kentucky_bluegrass", "bermuda", "fescue", "zoysia"]
        self.obstacle_types = ["tree", "rock", "flower_bed", "fence", "pet"]
        self.weather_conditions = ["sunny", "cloudy", "morning_dew", "afternoon_heat"]

    def create_lawn_scene(self, lawn_type: str, size_m: tuple[float, float],
                          obstacles: list[dict], weather: str) -> None:
        """Generate a randomized lawn environment."""
        # Create terrain with realistic grass rendering
        # Add obstacles with physics
        # Configure lighting and weather
        # ...

    def spawn_mower(self, position: tuple[float, float, float]) -> None:
        """Spawn the LawnBerry Pi mower model."""
        # Load URDF/USD model
        # Attach simulated sensors
        # ...

    def get_sensor_data(self) -> dict:
        """Get simulated sensor readings matching real hardware."""
        return {
            "stereo_left": self._render_stereo_left(),
            "stereo_right": self._render_stereo_right(),
            "pi_camera": self._render_pi_camera(),
            "gps": self._get_simulated_gps(),
            "imu": self._get_simulated_imu(),
            "ultrasonic": self._get_simulated_ultrasonic(),
        }

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Execute action and return (obs, reward, done, info)."""
        # Apply motor commands
        # Step physics
        # Calculate reward
        # Check termination
        # ...
```
**Acceptance**: Simulation renders at 30+ FPS, sensors match real

---

### BEAD-202: Cosmos Synthetic Data Generation
**Status**: pending
**Priority**: high
**Description**: Generate synthetic training data with NVIDIA Cosmos
Create `thor/synthetic/cosmos_augmentation.py`:
```python
"""NVIDIA Cosmos synthetic data generation for sim-to-real transfer."""
import cosmos
from cosmos.world_generation import WorldGenerator
from cosmos.physics import PhysicsEngine

class CosmosLawnDataAugmentation:
    """Generate synthetic lawn data with domain randomization."""

    def __init__(self):
        self.generator = WorldGenerator()
        self.physics = PhysicsEngine()

    def generate_lawn_variations(self, base_scene: str,
                                  num_variations: int = 1000) -> list[dict]:
        """Generate diverse lawn environments."""
        variations = []
        for i in range(num_variations):
            scene = self.generator.create_scene(
                base=base_scene,
                randomize={
                    "grass_height": (2, 15),      # cm
                    "grass_density": (0.3, 1.0),
                    "grass_color": "seasonal",
                    "lighting": "time_of_day",
                    "weather": ["clear", "cloudy", "overcast"],
                    "shadows": True,
                    "obstacles": self._random_obstacles(),
                    "terrain_slope": (-15, 15),   # degrees
                }
            )
            variations.append(scene)
        return variations

    def augment_real_data(self, real_frame: dict) -> list[dict]:
        """Apply Cosmos augmentation to real recorded data."""
        augmented = []

        # Weather variations
        for weather in ["sunny", "cloudy", "overcast"]:
            aug = self.generator.transfer_style(
                real_frame,
                target_weather=weather,
                preserve_geometry=True
            )
            augmented.append(aug)

        # Lighting variations
        for hour in [8, 12, 16, 19]:  # Morning to evening
            aug = self.generator.relighting(
                real_frame,
                time_of_day=hour,
                latitude=self.config.latitude
            )
            augmented.append(aug)

        return augmented
```
**Acceptance**: Generates 10K+ variations per real session

---

### BEAD-203: Custom VLA Model Architecture
**Status**: pending
**Priority**: critical
**Description**: Design LawnMower Vision-Language-Action model
Create `thor/models/lawn_vla.py`:
```python
"""LawnMower VLA: Vision-Language-Action model for autonomous mowing."""
import torch
import torch.nn as nn
from transformers import AutoModel

class LawnMowerVLA(nn.Module):
    """
    Custom VLA fine-tuned for lawn mowing.
    Based on GR00T N1.5 architecture with lawn-specific adaptations.
    """

    def __init__(self, config: dict):
        super().__init__()

        # Vision encoder (stereo + RGB)
        self.stereo_encoder = StereoDepthEncoder(
            backbone="efficientnet_v2_s",
            output_dim=512
        )
        self.rgb_encoder = RGBEncoder(
            backbone="dinov2_vits14",
            output_dim=768
        )

        # Sensor fusion
        self.sensor_fusion = MultiModalFusion(
            vision_dim=512 + 768,
            gps_dim=6,      # lat, lon, alt, hdop, fix_type, sats
            imu_dim=9,      # roll, pitch, yaw, accel_xyz
            ultrasonic_dim=3,
            hidden_dim=1024
        )

        # Language conditioning (for task instructions)
        self.language_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Transformer backbone
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=8),
            num_layers=6
        )

        # Action head
        self.action_head = ActionHead(
            input_dim=1024,
            steering_bins=21,    # -1.0 to 1.0 in 0.1 steps
            throttle_bins=11,    # 0.0 to 1.0 in 0.1 steps
            blade_classes=2      # on/off
        )

    def forward(self,
                stereo_left: torch.Tensor,
                stereo_right: torch.Tensor,
                rgb: torch.Tensor,
                gps: torch.Tensor,
                imu: torch.Tensor,
                ultrasonic: torch.Tensor,
                instruction: str = "mow the lawn efficiently") -> dict:
        """
        Forward pass: sensors + instruction -> action distribution.
        """
        # Encode vision
        stereo_features = self.stereo_encoder(stereo_left, stereo_right)
        rgb_features = self.rgb_encoder(rgb)

        # Encode language instruction
        lang_features = self.language_encoder(instruction)

        # Fuse all modalities
        fused = self.sensor_fusion(
            stereo_features, rgb_features,
            gps, imu, ultrasonic, lang_features
        )

        # Transformer processing
        context = self.transformer(fused)

        # Predict actions
        actions = self.action_head(context)

        return actions
```
**Acceptance**: Model trains, inference <100ms on Thor

---

### BEAD-204: World Model Implementation
**Status**: pending
**Priority**: high
**Description**: DreamerV3-style world model for lawn dynamics
Create `thor/models/world_model.py`:
```python
"""World model for predicting lawn mowing dynamics."""
import torch
import torch.nn as nn

class LawnWorldModel(nn.Module):
    """
    DreamerV3-style world model for lawn mowing.
    Predicts future states and rewards from current state + action.
    """

    def __init__(self, config: dict):
        super().__init__()

        # State encoder (compresses observations)
        self.encoder = StateEncoder(
            obs_dim=config.obs_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim
        )

        # Recurrent state model (RSSM)
        self.rssm = RecurrentStateSpaceModel(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        # Prediction heads
        self.image_decoder = ImageDecoder(config.latent_dim)
        self.reward_predictor = RewardPredictor(config.latent_dim)
        self.continuation_predictor = ContinuationPredictor(config.latent_dim)

        # Lawn-specific predictors
        self.grass_height_predictor = GrassHeightPredictor(config.latent_dim)
        self.coverage_predictor = CoveragePredictor(config.latent_dim)

    def imagine(self,
                start_state: torch.Tensor,
                actions: torch.Tensor,
                horizon: int = 50) -> dict:
        """
        Imagine future trajectories given actions.
        Used for planning and RL training.
        """
        imagined_states = []
        imagined_rewards = []
        state = start_state

        for t in range(horizon):
            # Predict next state
            next_state = self.rssm.imagine_step(state, actions[t])

            # Predict reward
            reward = self.reward_predictor(next_state)

            imagined_states.append(next_state)
            imagined_rewards.append(reward)
            state = next_state

        return {
            "states": torch.stack(imagined_states),
            "rewards": torch.stack(imagined_rewards),
            "coverage": self.coverage_predictor(imagined_states[-1])
        }
```
**Acceptance**: Predicts 50-step futures, coverage accuracy >90%

---

### BEAD-205: Reinforcement Learning Training
**Status**: pending
**Priority**: high
**Description**: RL for coverage optimization
Create `thor/training/rl_trainer.py`:
```python
"""Reinforcement learning for lawn coverage optimization."""
import torch
from torch.distributions import Categorical

class LawnMowingRL:
    """
    RL trainer for optimizing mowing coverage and efficiency.
    Uses world model for imagination-based training.
    """

    def __init__(self, vla: LawnMowerVLA, world_model: LawnWorldModel, config: dict):
        self.vla = vla
        self.world_model = world_model
        self.config = config

        # Reward components
        self.coverage_weight = 1.0
        self.efficiency_weight = 0.3
        self.safety_weight = 2.0
        self.smoothness_weight = 0.1

    def compute_reward(self, state: dict, action: dict, next_state: dict) -> float:
        """
        Multi-objective reward function.
        """
        reward = 0.0

        # Coverage reward (new area mowed)
        new_coverage = self._compute_new_coverage(state, next_state)
        reward += self.coverage_weight * new_coverage

        # Efficiency reward (minimize overlap)
        overlap_penalty = self._compute_overlap(state, next_state)
        reward -= self.efficiency_weight * overlap_penalty

        # Safety reward (avoid obstacles, stay in bounds)
        safety_penalty = self._compute_safety_penalty(state, action)
        reward -= self.safety_weight * safety_penalty

        # Smoothness reward (minimize jerky movements)
        smoothness_penalty = self._compute_smoothness_penalty(action)
        reward -= self.smoothness_weight * smoothness_penalty

        return reward

    def train_in_imagination(self, batch_size: int = 64, horizon: int = 50):
        """
        Train VLA using imagined trajectories from world model.
        """
        # Sample starting states from replay buffer
        start_states = self.replay_buffer.sample(batch_size)

        # Generate imagined trajectories
        with torch.no_grad():
            actions = self.vla.sample_actions(start_states, horizon)
            imagined = self.world_model.imagine(start_states, actions, horizon)

        # Compute returns
        returns = self._compute_returns(imagined["rewards"])

        # Policy gradient update
        loss = self._policy_gradient_loss(actions, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```
**Acceptance**: Coverage improves 20%+ over baseline

---

### BEAD-206: Training Pipeline Orchestration
**Status**: pending
**Priority**: high
**Description**: End-to-end training pipeline
Create `thor/training/pipeline.py`:
```python
"""Complete training pipeline for LawnBerry AI."""
import asyncio
from pathlib import Path

class LawnBerryTrainingPipeline:
    """
    Orchestrates the complete training workflow:
    1. Real data preprocessing
    2. Synthetic data generation
    3. VLA pre-training
    4. World model training
    5. RL fine-tuning
    6. Model distillation
    7. Deployment
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.isaac_sim = LawnSimEnvironment(self.config.sim)
        self.cosmos = CosmosLawnDataAugmentation()
        self.vla = LawnMowerVLA(self.config.vla)
        self.world_model = LawnWorldModel(self.config.world_model)
        self.rl_trainer = LawnMowingRL(self.vla, self.world_model, self.config.rl)

    async def run_full_pipeline(self):
        """Execute complete training pipeline."""

        # Stage 1: Data Preparation
        print("Stage 1: Preparing training data...")
        real_data = await self._load_real_recordings()
        synthetic_data = await self._generate_synthetic_data()
        augmented_data = await self._augment_real_data(real_data)

        # Stage 2: VLA Pre-training
        print("Stage 2: Pre-training VLA on behavior cloning...")
        await self._pretrain_vla(real_data + augmented_data)

        # Stage 3: World Model Training
        print("Stage 3: Training world model...")
        await self._train_world_model(real_data + synthetic_data)

        # Stage 4: RL Fine-tuning
        print("Stage 4: RL fine-tuning in imagination...")
        await self._finetune_with_rl()

        # Stage 5: Evaluation
        print("Stage 5: Evaluating in simulation...")
        metrics = await self._evaluate_in_sim()

        # Stage 6: Distillation
        if metrics["coverage"] > 0.95:
            print("Stage 6: Distilling to Hailo format...")
            await self._distill_to_hailo()

        return metrics
```
**Acceptance**: Pipeline runs end-to-end, produces deployable model

---

### PHASE 3: MODEL DISTILLATION & EDGE DEPLOYMENT

---

### BEAD-300: Hailo Model Compiler Setup
**Status**: pending
**Priority**: critical
**Description**: Configure Hailo Dataflow Compiler for model conversion
```bash
# Install Hailo Dataflow Compiler
pip install hailo-dataflow-compiler

# Download optimization profiles
hailo_model_zoo download --target hailo8l
```
Create `thor/distillation/hailo_compiler.py`:
- ONNX export from PyTorch
- Quantization (FP32 -> INT8)
- Hailo HEF compilation
- Performance profiling
**Acceptance**: Compiler produces valid HEF files

---

### BEAD-301: Knowledge Distillation
**Status**: pending
**Priority**: critical
**Description**: Distill full VLA to edge-deployable model
Create `thor/distillation/distiller.py`:
```python
"""Knowledge distillation for Hailo 8L deployment."""
import torch
import torch.nn as nn

class HailoDistillation:
    """
    Distill full LawnMowerVLA to lightweight edge model.
    Target: 13 TOPS INT8, <50ms inference.
    """

    def __init__(self, teacher: LawnMowerVLA, config: dict):
        self.teacher = teacher
        self.teacher.eval()

        # Student model (smaller, quantization-friendly)
        self.student = LawnMowerVLALight(
            backbone="mobilenet_v3_small",
            hidden_dim=256,
            num_layers=2
        )

    def distill(self, train_loader, epochs: int = 50):
        """
        Train student to match teacher outputs.
        """
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-4)

        for epoch in range(epochs):
            for batch in train_loader:
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_out = self.teacher(**batch)

                # Student forward
                student_out = self.student(**batch)

                # Distillation loss (KL divergence + MSE)
                loss = self._distillation_loss(teacher_out, student_out)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def export_to_hailo(self, output_path: str):
        """Export quantized model to Hailo HEF format."""
        # Export to ONNX
        onnx_path = output_path.replace(".hef", ".onnx")
        torch.onnx.export(self.student, dummy_input, onnx_path)

        # Quantize and compile for Hailo
        from hailo_sdk_client import ClientRunner
        runner = ClientRunner(hw_arch="hailo8l")
        runner.translate_onnx_model(onnx_path)
        runner.optimize_model()
        runner.compile()
        runner.save_har(output_path)
```
**Acceptance**: Student achieves 95%+ of teacher performance

---

### BEAD-302: Edge Inference Runtime
**Status**: pending
**Priority**: critical
**Description**: Deploy distilled model to Hailo 8L
Create `backend/src/services/ai_inference_service.py`:
```python
"""AI inference service using Hailo 8L accelerator."""
import asyncio
from hailo_platform import HailoRT, Device

class AIInferenceService:
    """
    Real-time inference on Hailo 8L.
    Runs distilled LawnMowerVLA for autonomous control.
    """

    def __init__(self, model_path: str):
        self.device = Device()
        self.network = self.device.configure(model_path)
        self.input_buffer = None
        self.output_buffer = None

    async def initialize(self):
        """Initialize Hailo device and load model."""
        self.input_buffer = self.network.create_input_buffer()
        self.output_buffer = self.network.create_output_buffer()

    async def infer(self, sensor_data: dict) -> dict:
        """
        Run inference on sensor data.
        Returns action predictions.
        """
        # Preprocess inputs
        inputs = self._preprocess(sensor_data)

        # Copy to input buffer
        self.input_buffer.set_buffer(inputs)

        # Run inference
        await asyncio.to_thread(self.network.run,
                                self.input_buffer,
                                self.output_buffer)

        # Postprocess outputs
        actions = self._postprocess(self.output_buffer.get_buffer())

        return {
            "steering": actions["steering"],
            "throttle": actions["throttle"],
            "blade": actions["blade"],
            "confidence": actions["confidence"],
            "inference_ms": self._last_inference_time
        }
```
**Acceptance**: Inference <50ms, actions valid

---

### BEAD-303: Autonomous Control Integration
**Status**: pending
**Priority**: critical
**Description**: Integrate AI inference with motor control
Modify `backend/src/services/navigation_service.py`:
- Add AI mode alongside manual/waypoint modes
- Use AI predictions for steering/throttle
- Maintain safety overrides
- Log AI decisions for analysis
**Acceptance**: Mower follows AI commands smoothly

---

### PHASE 4: CONTINUOUS IMPROVEMENT

---

### BEAD-400: Online Learning Pipeline
**Status**: pending
**Priority**: medium
**Description**: Continuous model improvement from field data
Create `backend/src/services/online_learning.py`:
- Collect edge cases during operation
- Upload anomalies to Thor
- Retrain on new data
- OTA model updates
**Acceptance**: Model improves over deployment lifetime

---

### BEAD-401: A/B Testing Framework
**Status**: pending
**Priority**: medium
**Description**: Compare model versions in production
Create `backend/src/services/ab_testing.py`:
- Deploy multiple models simultaneously
- Random assignment per session
- Collect performance metrics
- Statistical significance testing
**Acceptance**: Can compare model performance rigorously

---

### BEAD-402: Monitoring Dashboard
**Status**: pending
**Priority**: medium
**Description**: Training and inference monitoring
Create monitoring for:
- Training loss curves (Thor)
- Inference latency (Edge)
- Coverage efficiency
- Safety interventions
- Model version performance
**Acceptance**: Full visibility into AI system health

---

### PHASE 5: TESTING & VALIDATION

---

### BEAD-500: Simulation Test Suite
**Status**: pending
**Priority**: high
**Description**: Automated testing in Isaac Sim
Create `thor/tests/simulation_tests.py`:
- Randomized lawn scenarios
- Coverage benchmarks
- Obstacle avoidance tests
- Edge case handling
- Regression testing
**Acceptance**: 95%+ pass rate on test scenarios

---

### BEAD-501: Hardware-in-Loop Testing
**Status**: pending
**Priority**: high
**Description**: Test AI control on real hardware
Create `tests/hil/ai_control_tests.py`:
- Motor response validation
- Sensor data quality
- Inference latency
- Safety system integration
**Acceptance**: All HIL tests pass

---

### BEAD-502: Field Validation Protocol
**Status**: pending
**Priority**: critical
**Description**: Real-world testing procedure
Create `docs/field-validation-protocol.md`:
- Test lawn configurations
- Safety observer requirements
- Success criteria
- Failure handling
- Data collection during tests
**Acceptance**: Documentation complete, protocol executable

---

### BEAD-503: Integration Testing
**Status**: pending
**Priority**: critical
**Description**: End-to-end system integration
```bash
# Run full test suite
cd /home/kp/repos/lawnberry_pi
pytest tests/ -v

# Hardware-in-loop tests
pytest tests/hil/ -v

# AI model tests
pytest thor/tests/ -v
```
**Acceptance**: All tests pass, no regressions

---

### BEAD-504: Documentation Update
**Status**: pending
**Priority**: medium
**Description**: Update project documentation
- Update `docs/hardware-overview.md` with new sensors
- Create `docs/ai-architecture.md` for AI system
- Create `docs/training-guide.md` for model training
- Create `docs/deployment-guide.md` for edge deployment
- Update `README.md` with AI capabilities
**Acceptance**: Documentation matches implementation

---

## Completion Criteria
- [ ] All BEAD tasks marked complete
- [ ] All sensors operational and streaming telemetry
- [ ] Isaac Sim environment functional
- [ ] Cosmos generating synthetic data
- [ ] VLA model trained and evaluated
- [ ] World model predicting accurately
- [ ] RL improving coverage efficiency
- [ ] Model distilled to Hailo 8L
- [ ] Edge inference <50ms
- [ ] Autonomous mowing demo successful
- [ ] Safety interlocks verified
- [ ] Documentation complete

---

## Hardware Reference

### ELP Stereo Camera (ELP-USB960P2CAM-V90)
- USB 2.0 UVC device (Vendor: 32e4, Product: 9750)
- Resolution: 2560x960 combined (1280x960 per eye)
- Framerate: Up to 60fps
- Synchronized shutters (< 1ms difference)

### LC29H(DA) GPS/RTK Module
- Interface: UART at 115200 baud
- Protocol: NMEA + PQTM/PAIR proprietary
- RTK accuracy: < 1cm with corrections
- Commands: $PQTMSAVEPAR, $PQTMRESTOREPAR, $PAIR050

### HC-SR04 Ultrasonic Sensors
- Range: 2cm - 400cm
- Voltage: 5V power, 5V logic (ECHO needs divider!)
- GPIO pins selected to avoid conflicts with existing hardware

### Hailo 8L AI Accelerator
- Compute: 13 TOPS INT8
- Interface: PCIe/M.2
- Power: <5W
- Latency: <50ms for VLA inference

### NVIDIA Jetson AGX Thor
- GPU: Blackwell, 2070 TFLOPS FP4
- Memory: 128GB unified
- Storage: 8TB NVMe
- Network: 10GbE + HaLow

---

## Notes
- Run `conda deactivate` multiple times if nested conda envs
- LC29H replaces ZED-F9P (not additive)
- Ultrasonic wiring must be done AFTER software is ready
- Test each component individually before integration
- Training requires significant compute (expect 1-2 weeks for full pipeline)
- Start with simulation validation before field testing
- Always have safety observer during autonomous tests
