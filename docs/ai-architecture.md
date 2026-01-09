# AI Architecture - LawnBerry Pi Autonomous Mowing System

## Overview

The LawnBerry Pi AI system enables autonomous mowing through a Vision-Language-Action (VLA) model architecture. The system combines multi-modal sensor fusion with edge inference on the Hailo 8L accelerator.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SENSOR INPUTS                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Stereo Camera│  │  Pi Camera 2 │  │  LC29H GPS   │              │
│  │  1280x960x2  │  │  1920x1080   │  │   RTK/GNSS   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐              │
│  │  BNO085 IMU  │  │ Ultrasonic   │  │   VL53L0X    │              │
│  │  9-axis      │  │   HC-SR04x3  │  │   ToF x2     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┴──────────────────┴──────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AI INFERENCE SERVICE                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   PREPROCESSING PIPELINE                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │   │
│  │  │ Image      │  │ Sensor     │  │ Feature Normalization  │  │   │
│  │  │ Resize/Norm│  │ Encoding   │  │ GPS/IMU/Ultrasonic     │  │   │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   HAILO 8L ACCELERATOR                        │   │
│  │                   13 TOPS INT8 Inference                      │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │              VLA Model (lawnmower_vla.hef)              │  │   │
│  │  │  Stereo Encoder → RGB Encoder → Sensor Fusion → Actions │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   POSTPROCESSING                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │   │
│  │  │ Action     │  │ Confidence │  │ Safety Override Check  │  │   │
│  │  │ Decoding   │  │ Scoring    │  │ Obstacle/Boundary      │  │   │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTROL OUTPUTS                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Steering   │  │   Throttle   │  │    Blade     │              │
│  │   -1.0..1.0  │  │   0.0..1.0   │  │   On/Off     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. AI Inference Service

**Location**: `backend/src/services/ai_inference_service.py`

The central service that orchestrates AI-based autonomous control:

```python
from backend.src.services.ai_inference_service import get_ai_inference_service

# Get singleton instance
service = get_ai_inference_service()

# Initialize and start
await service.initialize()
await service.start()

# Run inference on sensor data
prediction = await service.infer(sensor_frame)

# Check confidence before applying
if prediction.confidence > 0.7:
    motor_commands = prediction.to_motor_commands()
```

**Features**:
- Async lifecycle management (HardwareDriver pattern)
- Hardware acceleration via Hailo 8L
- Graceful fallback to simulation mode
- Input preprocessing pipeline
- Output postprocessing with confidence scoring
- Performance metrics tracking
- Safety override integration

### 2. Hailo Driver

**Location**: `backend/src/drivers/ai/hailo_driver.py`

Manages the Hailo 8L AI accelerator:

```python
from backend.src.drivers.ai.hailo_driver import HailoDriver

driver = HailoDriver({"model": "yolov8m"})
await driver.initialize()

# Check hardware availability
if driver.hardware_available:
    result = await driver.infer(input_tensors)
```

**Capabilities**:
- Model loading from HEF format
- Async inference execution
- Temperature monitoring
- Hardware health checks
- Simulation fallback

### 3. Action Prediction Models

**Location**: `backend/src/models/action_prediction.py`

Data models for AI predictions:

```python
@dataclass
class ActionPrediction:
    steering: float          # -1.0 to 1.0
    throttle: float          # 0.0 to 1.0
    blade: bool              # On/Off
    confidence: float        # 0.0 to 1.0
    safety_override: bool    # Safety system engaged
    obstacle_detected: bool  # Obstacle proximity warning

    def to_motor_commands(self) -> Dict[str, float]:
        """Convert to differential drive commands."""
        ...
```

### 4. Navigation Integration

**Location**: `backend/src/services/navigation_service.py`

AI control modes in navigation:

```python
# Enable AI autonomous mode
await navigation_service.start_ai_navigation()

# Apply AI predictions to motors
await navigation_service.apply_ai_prediction(prediction)

# Return to manual control
await navigation_service.stop_ai_navigation()
```

**Navigation Modes**:
- `MANUAL` - Direct user control
- `AUTO` - Waypoint-based navigation
- `AI` - Full AI autonomous control
- `AI_ASSISTED` - AI suggestions with human override

## REST API

**Base URL**: `/api/v2/ai`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/enable` | Enable AI control mode |
| POST | `/disable` | Disable AI control mode |
| GET | `/status` | Get AI system status |
| POST | `/model` | Load a new AI model |
| GET | `/metrics` | Get inference metrics |
| POST | `/metrics/reset` | Reset performance metrics |
| GET | `/health` | Health check endpoint |

### Example Usage

```bash
# Enable AI control
curl -X POST http://localhost:8000/api/v2/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"mode": "ai_autonomous"}'

# Check status
curl http://localhost:8000/api/v2/ai/status

# Get metrics
curl http://localhost:8000/api/v2/ai/metrics
```

## VLA Model Architecture

The Vision-Language-Action model is designed for lawn mowing:

### Input Specifications

| Input | Shape | Description |
|-------|-------|-------------|
| Stereo Left | 256x320x3 | Resized left camera |
| Stereo Right | 256x320x3 | Resized right camera |
| RGB | 224x224x3 | Pi Camera 2 frame |
| Sensors | 9 | GPS, IMU, Ultrasonic features |

### Output Specifications

| Output | Shape | Description |
|--------|-------|-------------|
| Steering | 21 | Probability distribution (-1.0 to 1.0) |
| Throttle | 11 | Probability distribution (0.0 to 1.0) |
| Blade | 2 | Binary classification (on/off) |

### Model Files

Models are stored in HEF format for Hailo:
- Default path: `/home/kp/repos/lawnberry_pi/models/lawnmower_vla.hef`
- Placeholder models: `yolov8m.hef`, `scdepthv3.hef`

## Safety Integration

### Safety Overrides

The AI system respects safety constraints:

1. **Obstacle Proximity**: Throttle reduced/stopped when ultrasonic < 30cm
2. **Confidence Threshold**: Actions only applied if confidence > 0.5
3. **Boundary Warning**: GPS geofence integration
4. **Tilt Detection**: IMU-based slope safety

### Safety Priority

```
1. Emergency Stop (hardware) - Highest
2. Safety Interlocks (software)
3. AI Safety Overrides
4. AI Predictions - Lowest
```

## Performance Metrics

The AI service tracks comprehensive metrics:

```python
@dataclass
class InferenceMetrics:
    total_inferences: int
    successful_inferences: int
    failed_inferences: int
    safety_overrides: int
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    inferences_per_second: float
    target_fps: float  # Default: 10.0
    avg_confidence: float
    hardware_accelerated: bool
```

### Target Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Inference Time | < 50ms | Including pre/post processing |
| FPS | 10+ | Control loop frequency |
| Confidence | > 0.7 | For reliable operation |
| Latency | < 100ms | End-to-end response |

## Simulation Mode

When hardware is unavailable, the system falls back to simulation:

```python
# Set environment variable
os.environ["SIM_MODE"] = "1"

# Or check programmatically
from backend.src.core.simulation import is_simulation_mode
if is_simulation_mode():
    # Using simulated inference
```

Simulation features:
- Realistic obstacle avoidance behavior
- Simulated inference latency (~15ms)
- Confidence variation based on conditions
- Safety override simulation

## Future Enhancements

### Thor Training Server Integration

The system is designed for future integration with NVIDIA Thor:

1. **Model Training**: Isaac Sim + Cosmos for synthetic data
2. **Distillation**: Full VLA → Edge-optimized model
3. **OTA Updates**: Remote model deployment
4. **Online Learning**: Continuous improvement from field data

### Hailo Version Resolution

Current status: Version mismatch (driver 4.21.0 vs library 4.20.0)

Resolution options:
1. Wait for Pi repo update to hailo-all 4.21.0
2. Manual HailoRT 4.21.0 installation from Hailo Developer Zone
3. Continue using simulation mode (current fallback)

## Testing

Run AI inference tests:

```bash
SIM_MODE=1 python -m pytest tests/unit/test_ai_inference_service.py -v
```

All 20 tests should pass in simulation mode.
