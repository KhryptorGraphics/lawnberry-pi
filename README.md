# LawnBerry Pi v3

**Professional Autonomous Mowing System with AI-Powered Navigation**

An advanced autonomous lawn mower powered by Raspberry Pi 5 with Hailo 8L AI accelerator, featuring Vision-Language-Action (VLA) model inference, RTK GPS positioning, multi-sensor fusion, and a professional web dashboard.

---

## Features

### AI-Powered Autonomous Navigation
- **Vision-Language-Action (VLA) Model**: 37M parameter neural network with ResNet50 visual encoder and 4-layer Transformer
- **Hailo 8L Acceleration**: 13 TOPS INT8 inference at 10+ FPS for real-time control
- **Multi-Modal Sensor Fusion**: Stereo depth, RGB vision, GPS, IMU, ultrasonic inputs
- **Model-Based Reinforcement Learning**: DreamerV3-style imagination training with world model

### Precision Positioning
- **RTK GPS**: Centimeter-level accuracy via LC29H-DA dual-band GNSS module
- **NTRIP Client**: Real-time RTK corrections from CORS networks
- **Dead Reckoning**: IMU-based position estimation during GPS outages
- **Geofencing**: Virtual boundary enforcement with configurable zones

### Safety Systems
- **Multi-Layer Obstacle Detection**: Ultrasonic (HC-SR04 array), ToF (VL53L0X), stereo vision
- **Emergency Stop**: Hardware and software E-STOP with watchdog timer
- **Tilt Protection**: BNO085 IMU monitors orientation for rollover prevention
- **Safety Interlocks**: Motor authorization, blade guards, perimeter enforcement

### Professional Dashboard
- **Real-Time Telemetry**: 5Hz WebSocket streaming of all sensor data
- **Interactive Mission Planner**: Draw boundaries, set waypoints, configure mowing patterns
- **Live Camera Feeds**: Stereo and Pi Camera streams with overlay data
- **1980s Cyberpunk UI**: Professional dark theme with Orbitron fonts and neon accents

### Training Infrastructure (Thor Server)
- **NVIDIA Jetson AGX Thor**: Blackwell GPU with 2070 TFLOPS for AI training
- **Isaac Sim Environment**: Physics-accurate lawn simulation for data generation
- **NVIDIA Cosmos**: Synthetic data augmentation and domain randomization
- **Knowledge Distillation**: Teacher-student pipeline for Hailo 8L deployment

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING INFRASTRUCTURE (NVIDIA Thor)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Isaac Sim    │  │    Cosmos     │  │  LawnMower    │  │    World     │ │
│  │  Environment  │──│   Augment     │──│     VLA       │──│    Model     │ │
│  │  (Simulation) │  │  (Synthetic)  │  │   (Policy)    │  │  (Dynamics)  │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│                                      │                                       │
│                           Model Distillation                                 │
│                                      ▼                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              Wireless Link
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
│                    LawnBerry Pi Backend (FastAPI)                            │
│                              │                                               │
│                    Motor Control (RoboHAT MM1)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Edge Device: Raspberry Pi 5 + Hailo 8L
| Component | Specification |
|-----------|---------------|
| CPU | Broadcom BCM2712, Cortex-A76 @ 2.4GHz |
| AI Accelerator | Hailo 8L M.2 (13 TOPS INT8) |
| Memory | 8-16GB LPDDR4X |
| Storage | 128GB microSD + USB SSD |
| OS | Raspberry Pi OS Lite Bookworm (64-bit) |

### Sensors
| Sensor | Model | Interface | Purpose |
|--------|-------|-----------|---------|
| Stereo Camera | ELP-USB960P2CAM-V90 | USB 2.0 | Depth perception |
| Pi Camera | Raspberry Pi Camera 2 | CSI | High-res analysis |
| GPS/RTK | LC29H(DA) HAT | UART | Centimeter positioning |
| Ultrasonic | HC-SR04 x3 | GPIO | Close-range detection |
| IMU | BNO085 | UART | Orientation sensing |
| ToF | VL53L0X x2 | I2C | Blade height sensing |

### Motor Control
| Component | Model | Interface |
|-----------|-------|-----------|
| Motor Controller | RoboHAT MM1 | I2C/UART |
| Drive Motors | 24V DC brushless | PWM |
| Blade Motor | 24V DC brushless | PWM |
| Battery | Victron 25.6V LiFePO4 | BLE monitoring |

### Training Server (Optional)
| Component | Specification |
|-----------|---------------|
| Platform | NVIDIA Jetson AGX Thor |
| GPU | Blackwell architecture (2070 TFLOPS FP4) |
| Memory | 128GB unified RAM |
| Storage | 8TB NVMe array |
| Software | JetPack 7.x, Isaac Sim 6.0, Cosmos 2.5 |

---

## Quick Start

### Prerequisites
- Raspberry Pi 5 (8GB+ recommended)
- Raspberry Pi OS Lite Bookworm 64-bit
- Python 3.11.x
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/KhryptorGraphics/lawnberry-pi.git
cd lawnberry-pi

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
cd backend
pip install -e ".[hardware]"

# Install frontend dependencies
cd ../frontend
npm install
```

### Running the System

```bash
# Terminal 1: Backend API
cd backend
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend Dashboard
cd frontend
npm run dev -- --host 0.0.0.0 --port 3001
```

### Access Points
| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3001 |
| API Docs | http://localhost:8000/docs |
| WebSocket | ws://localhost:8000/api/v2/ws/telemetry |

---

## API Reference

### AI Control

```bash
# Enable AI autonomous mode
curl -X POST http://localhost:8000/api/v2/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"mode": "ai_autonomous"}'

# Check AI status
curl http://localhost:8000/api/v2/ai/status

# View inference metrics
curl http://localhost:8000/api/v2/ai/metrics

# Disable AI mode (return to manual)
curl -X POST http://localhost:8000/api/v2/ai/disable
```

### Telemetry

```bash
# Get current sensor readings
curl http://localhost:8000/api/v2/telemetry/current

# Get GPS position
curl http://localhost:8000/api/v2/telemetry/gps

# Get battery status
curl http://localhost:8000/api/v2/telemetry/battery
```

### Mission Control

```bash
# Create a new mission
curl -X POST http://localhost:8000/api/v2/missions \
  -H "Content-Type: application/json" \
  -d '{"name": "Front Yard", "waypoints": [...]}'

# Start mission
curl -X POST http://localhost:8000/api/v2/missions/{id}/start

# Pause/Resume mission
curl -X POST http://localhost:8000/api/v2/missions/{id}/pause
curl -X POST http://localhost:8000/api/v2/missions/{id}/resume

# Emergency stop
curl -X POST http://localhost:8000/api/v2/control/estop
```

---

## Project Structure

```
lawnberry-pi/
├── backend/
│   └── src/
│       ├── api/              # FastAPI routes
│       │   └── routers/      # API endpoint modules
│       ├── core/             # Configuration and startup
│       ├── drivers/          # Hardware abstraction
│       │   ├── ai/           # Hailo inference driver
│       │   ├── blade/        # Blade motor control
│       │   ├── motor/        # Drive motor control
│       │   └── sensors/      # Sensor drivers
│       ├── fusion/           # Sensor fusion algorithms
│       ├── models/           # Pydantic data models
│       ├── nav/              # Navigation algorithms
│       ├── safety/           # Safety systems
│       │   ├── estop_handler.py
│       │   ├── safety_monitor.py
│       │   └── watchdog.py
│       └── services/         # Business logic
│           ├── ai_inference_service.py
│           ├── navigation_service.py
│           ├── sensor_manager.py
│           ├── ntrip_client.py
│           └── websocket_hub.py
├── frontend/
│   └── src/
│       ├── components/       # Vue.js components
│       ├── views/            # Page views
│       ├── stores/           # Pinia state management
│       └── services/         # API clients
├── config/                   # Configuration files
│   ├── hardware.yaml
│   └── safety.yaml
├── docs/                     # Documentation
├── tests/                    # Test suites
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── hil/                  # Hardware-in-loop tests
├── scripts/                  # Utility scripts
└── systemd/                  # Service definitions
```

---

## Services Overview

### Core Services
| Service | Description |
|---------|-------------|
| `ai_inference_service` | Hailo 8L VLA model inference |
| `navigation_service` | Path planning and waypoint following |
| `sensor_manager` | Multi-sensor data aggregation |
| `motor_service` | Drive and blade motor control |
| `safety_monitor` | Real-time safety enforcement |

### Communication Services
| Service | Description |
|---------|-------------|
| `websocket_hub` | Real-time telemetry streaming |
| `ntrip_client` | RTK correction data client |
| `telemetry_hub` | Data recording and playback |
| `thor_uploader` | Training data sync to Thor |

### Support Services
| Service | Description |
|---------|-------------|
| `auth_service` | JWT authentication |
| `maps_service` | Map tile serving |
| `weather_service` | Weather data integration |
| `calibration_service` | Sensor calibration |

---

## AI Model Architecture

### VLA Model Specifications
| Parameter | Value |
|-----------|-------|
| Total Parameters | 37.4M |
| Visual Encoder | ResNet50 (pretrained) |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Hidden Dimension | 512 |
| Action Heads | 3 (steering, throttle, blade) |

### Input Modalities
| Input | Dimensions | Encoding |
|-------|------------|----------|
| Stereo Camera | 540x960x6 | ResNet50 features |
| Pi Camera | 720x1280x3 | ResNet50 features |
| GPS | 4 (lat, lon, alt, accuracy) | Linear projection |
| IMU | 9 (accel, gyro, mag) | Linear projection |
| Ultrasonic | 3 (front, left, right) | Linear projection |

### Output Actions
| Output | Range | Description |
|--------|-------|-------------|
| Steering | [-1.0, 1.0] | Left/right turn |
| Throttle | [0.0, 1.0] | Forward speed |
| Blade | [0.0, 1.0] | Blade engagement probability |

---

## Testing

### Run All Tests
```bash
# Activate virtual environment
source .venv/bin/activate

# Run full test suite
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
pytest tests/contract/ -v       # API contract tests
pytest tests/hil/ -v            # Hardware-in-loop tests
```

### Test Coverage
| Category | Tests | Status |
|----------|-------|--------|
| Unit | 50+ | Passing |
| Integration | 40+ | Passing |
| Contract | 30+ | Passing |
| HIL | 15+ | Passing |

---

## Configuration

### Hardware Configuration (`config/hardware.yaml`)
```yaml
sensors:
  gps:
    port: /dev/ttyAMA3
    baud: 115200
  imu:
    port: /dev/ttyAMA4
    baud: 115200
  stereo_camera:
    device: /dev/video0
    resolution: [960, 540]

ai:
  model_path: /opt/hailo/models/lawnmower_vla.hef
  inference_fps: 10
  confidence_threshold: 0.7

ntrip:
  enabled: true
  host: rtk2go.com
  port: 2101
  mountpoint: NEAREST
```

### Safety Configuration (`config/safety.yaml`)
```yaml
estop:
  gpio_pin: 17
  active_low: true

watchdog:
  timeout_ms: 500

interlocks:
  tilt_max_degrees: 25
  obstacle_min_distance_cm: 30
  boundary_margin_m: 0.5
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/installation-setup-guide.md) | Complete setup instructions |
| [AI Architecture](docs/ai-architecture.md) | VLA model and inference details |
| [Deployment Guide](docs/deployment-guide.md) | Production deployment |
| [GPS RTK Setup](docs/gps-ntrip-setup.md) | RTK corrections configuration |
| [Hardware Overview](docs/hardware-overview.md) | Sensor specifications |
| [Operations Guide](docs/OPERATIONS.md) | Day-to-day operations |
| [Field Validation](docs/field-validation-protocol.md) | Testing protocols |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and conventions
- Pull request process
- Testing requirements
- Documentation standards

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **NVIDIA** for Isaac Sim and Cosmos platforms
- **Hailo** for the 8L AI accelerator
- **Raspberry Pi Foundation** for the Pi 5 platform
- **Quectel** for the LC29H GNSS module
- Original fork from [acredsfan/lawnberry_pi](https://github.com/acredsfan/lawnberry_pi)
