# LawnBerry Pi v3

**Professional Autonomous Mowing System with AI-Powered Navigation**

An advanced autonomous lawn mower powered by Raspberry Pi 5 with Hailo 8L AI accelerator, featuring Vision-Language-Action (VLA) model inference, RTK GPS positioning, multi-sensor fusion, and a professional web dashboard.

> **Platform variants (H-11).** This repository supports two **different** machines, selected by
> `config/tractor.yaml` `enabled`: the **differential-drive LawnBerry mower** (RoboHAT RP2040 → Cytron MDDRC10,
> IBT-4 blade driver, 12 V LiFePO4 + solar) and the **Toro TimeCutter 50-inch zero-turn conversion** (PCA9685 lever
> servos, starter and PTO relays; BOM in [`hardware.md`](hardware.md), design in
> [`docs/tractor-platform.md`](docs/tractor-platform.md)). They are separate wiring harnesses and a deployment
> runs exactly one. The hardware tables below are tagged per platform.
>
> **Status: read first.** Capability statements in this README are targets unless marked otherwise. What is
> implemented, simulated, stubbed or missing, checked against the source tree and the deployed Pi on 2026-10-24,
> is in [`docs/2026-10-24-integration-audit.md`](docs/2026-10-24-integration-audit.md). The Toro variant is
> bench-only and **not cleared for unattended motion**
> ([`docs/tractor-acceptance-criteria.md`](docs/tractor-acceptance-criteria.md)).

---

## Features

### AI-Powered Autonomous Navigation
- **Vision-Language-Action (VLA) Model**: 37M parameter neural network with ResNet50 visual encoder and 4-layer Transformer
- **Hailo 8L Acceleration** (target): 13 TOPS INT8 is the vendor rating and "10+ FPS" for the VLA policy is a target;
  the only measured Hailo figure is the YOLO26 detector (yolo26s 41.7 FPS, `docs/autonomy-toolchain-matrix.md`).
  `lspci` on the deployed Pi says "Hailo-8" (shared PCI ID) while `hailortcli` said "HAILO8L" on 2026-10-06: re-verify
- **Multi-Modal Sensor Fusion** (target): stereo depth, 2D LiDAR and ultrasonic fusion are not implemented
- **Model-Based Reinforcement Learning**: DreamerV3-style imagination training with world model

### Precision Positioning
- **RTK GPS**: centimeter-level accuracy is a target. The receiver is unreconciled across documents (Quectel LC29H-DA
  on UART0 per `config/hardware.yaml`, or SparkFun ZED-F9P over USB per `spec/hardware.yaml`; H-09), and an
  LC29H-BS base station exists only as an owner requirement
- **NTRIP Client**: Real-time RTK corrections from CORS networks
- **Dead Reckoning**: IMU-based position estimation during GPS outages
- **Geofencing**: Virtual boundary enforcement with configurable zones

### Safety Systems
- **Obstacle stop** (software; every number UNVALIDATED, audit M-43): the two VL53L0X ToF sensors (and the
  ultrasonic array when enabled) feed one fail-closed `ObstacleGuard`. A stale, blind, unknown or missing required
  sensor, or an obstacle inside the speed-dependent stopping envelope, holds autonomous motion (levers neutral, blade
  off) for missions, the AI loop and Thor plans; manual operator drive is not limited by default. Four JSN-SR04T
  ultrasonics (owner requirement), stereo depth and optional 2D LiDAR are not implemented; the legacy 3x HC-SR04
  driver is disabled because its pins collide with the HaLow radio
- **GNSS hold** (software; thresholds UNVALIDATED): autonomous motion holds and a mission will not start while the fix
  is missing, stale, inaccurate or newly regained (resume dwell)
- **Emergency Stop**: software only. An in-process watchdog needs two heartbeats (the tractor's own actuation
  self-check with PCA9685 read-back, and the tilt monitor) and repeats the stop until the outputs verify; it shares the
  event loop it supervises. An independent hardwired E-stop and a **hardware watchdog (O-7)** are **not implemented**
  (H-08). Which stop routes actually actuate hardware is listed under "Emergency stop and recovery" below
- **Tilt Protection**: tractor path only, and the BNO085 driver is a simulation stub, so the cutoff cannot trip on real
  hardware at the audit baseline
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
- **Planned live role (H-23, not an implemented primary navigator):** the owner intends the Thor to lead
  high-level navigation over Wi-Fi HaLow while the Pi keeps local safety authority. Implemented vs target:
  [audit Part C](docs/2026-10-24-integration-audit.md)

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

*Diagram scope: differential-drive mower, target architecture. The Toro variant replaces the RoboHAT stage with the
PCA9685 and relays; the owner's ultrasonic requirement is four JSN-SR04T (the HC-SR04 array shown is the legacy design,
H-19); the "Wireless Link" is Wi-Fi HaLow. Implemented vs target:
[`docs/2026-10-24-integration-audit.md`](docs/2026-10-24-integration-audit.md) Part C.*

---

## Hardware Requirements

### Edge Device: Raspberry Pi 5 + Hailo 8L
| Component | Specification |
|-----------|---------------|
| CPU | Broadcom BCM2712, Cortex-A76 @ 2.4GHz |
| AI Accelerator | Hailo-8L M.2 (13 TOPS INT8, vendor rating); `lspci` on the deployed Pi shows "Hailo-8", verify |
| Memory | 8-16GB LPDDR4X |
| Storage | 128GB microSD + USB SSD |
| OS | Raspberry Pi OS Lite Bookworm (64-bit) per `spec/hardware.yaml`; the deployed Pi runs kernel 6.12.21-morse+ and `docs/autonomy-toolchain-matrix.md` records Ubuntu 24.04.4: reconcile |

### Sensors
| Sensor | Model | Interface | Purpose |
|--------|-------|-----------|---------|
| Stereo Camera | ELP-USB960P2CAM-V90 | USB 2.0 | Depth perception |
| Pi Camera | Raspberry Pi Camera 2 | CSI | High-res analysis |
| GPS/RTK | **Unreconciled (H-09):** Quectel LC29H-DA rover (`config/hardware.yaml`) or SparkFun ZED-F9P (`spec/hardware.yaml`); LC29H-BS base = owner requirement, not implemented | UART / USB | Centimeter positioning (target) |
| Ultrasonic | **4x JSN-SR04T** (owner requirement, variant unverified; H-19); the legacy 3x HC-SR04 driver is disabled and uses HaLow SPI0 pins ([`docs/ultrasonic-wiring-guide.md`](docs/ultrasonic-wiring-guide.md), H-02) | GPIO (TBD) | Close-range detection (not implemented for JSN-SR04T) |
| IMU | BNO085 (driver is a simulation stub; UART4 vs I2C unverified, H-03) | UART4 (spec) | Orientation sensing |
| ToF | VL53L0X x2 | I2C | Close-range obstacle detection (front-left / front-right per `docs/hardware-overview.md`), not blade height |

### Motor Control — differential-drive mower platform
| Component | Model | Interface |
|-----------|-------|-----------|
| Motor Controller | RoboHAT MM1 (RP2040-Zero) → Cytron MDDRC10 | USB CDC `/dev/ttyACM0` at 115200 (`docs/servo-hat-pins.md`); the MDDRC10 takes RC PWM |
| Drive Motors | 2x 12 V worm-gear DC with Hall encoders | via the MDDRC10 |
| Blade Motor | 997 DC motor via IBT-4 | GPIO24/25 (driver is a stub; GPIO25 collides with the HaLow IRQ) |
| Battery | 12.8 V 30 Ah LiFePO4 (`spec/hardware.yaml`) + Victron SmartSolar MPPT | BLE monitoring + INA3221 |

*The earlier rows here ("24V DC brushless" motors, "Victron 25.6V LiFePO4", RoboHAT "I2C/UART") contradicted
`.github/copilot-instructions.md`, `spec/hardware.yaml` and `config/hardware.yaml`.*

### Actuation — Toro TimeCutter zero-turn variant
| Component | Model | Interface |
|-----------|-------|-----------|
| Lever servos | 2x 12 V-class, 150 kg·cm-class (owner-reported 10–12.6 V; `hardware.md` lists DSSERVO/ANNIMOS RDS51150SG); label unverified (H-05, H-20) | PWM from PCA9685 ch 0 / ch 2 |
| Throttle actuator | model not yet identified | PCA9685 ch 1 |
| PWM driver | PCA9685 at I2C 0x41 (A0 bridged) | I2C bus 1 is **absent** on the deployed Pi (H-17) |
| Starter / PTO | GPIO relays (BCM5 / BCM26); the OEM PTO switch is reported momentary (H-21) | GPIO5 collides with the HaLow RESET line (H-02b) |

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
# Terminal 1: Backend API (from the repo root; the package is backend.src)
SIM_MODE=1 uvicorn backend.src.main:app --host 0.0.0.0 --port 8081 --reload

# Terminal 2: Frontend Dashboard
cd frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

The Vite dev proxy still targets `127.0.0.1:8000` (`frontend/vite.config.ts`): either start the backend on port 8000
for `npm run dev`, or point the proxy at 8081.

### Access Points
Deployed units (`systemd/lawnberry-backend.service`, `systemd/lawnberry-frontend.service`) use 8081 and 3000.

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API Docs | http://localhost:8081/docs |
| WebSocket | ws://localhost:8081/api/v2/ws/telemetry |

---

## API Reference

Examples assume the deployed port. **Operator authentication is enforced** (`OPERATOR_AUTH_REQUIRED=1` in
the repo unit; the deployed Pi keeps the old `0` until `systemd/install_services.sh` is run there, which also
provisions the credential). Log in with `POST /api/v2/auth/login` and send `Authorization: Bearer <token>` on
operator writes; the E-stop routes stay open by design. Where a route below has no auth dependency (N-16), it is still open.

```bash
export API=http://localhost:8081
```

### AI Control

```bash
# Enable AI autonomous mode (an operator write: TOKEN is the access token of POST /api/v2/auth/login)
curl -X POST $API/api/v2/ai/enable \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode": "ai_autonomous"}'

# Check AI status
curl $API/api/v2/ai/status

# View inference metrics
curl $API/api/v2/ai/metrics

# Disable AI mode (return to manual)
curl -X POST $API/api/v2/ai/disable -H "Authorization: Bearer $TOKEN"
```

### Telemetry

```bash
# Real-time dashboard telemetry snapshot (the earlier /telemetry/current, /gps and /battery routes do not exist)
curl $API/api/v2/dashboard/telemetry

# Contract-shaped telemetry stream (polling)
curl "$API/api/v2/telemetry/stream?limit=5"

# Sensor health (GPS, IMU, ToF, power)
curl $API/api/v2/sensors/health
```

### Mission Control

```bash
# Create a new mission
curl -X POST $API/api/v2/missions/create \
  -H "Content-Type: application/json" \
  -d '{"name": "Front Yard", "waypoints": [...]}'

# Start / pause / resume / abort / status / list
curl -X POST $API/api/v2/missions/{id}/start
curl -X POST $API/api/v2/missions/{id}/pause
curl -X POST $API/api/v2/missions/{id}/resume
curl -X POST $API/api/v2/missions/{id}/abort
curl $API/api/v2/missions/{id}/status
curl $API/api/v2/missions/list
```

### Emergency stop and recovery

The route the earlier README gave (`POST /api/v2/control/estop`) does not exist. Routes that exist:

| Route | Platform | What it does |
|---|---|---|
| `POST /api/v2/tractor/emergency-stop` | Toro | Opens the PTO and starter relays, commands both levers to neutral and the throttle to idle, revokes authorization and latches the tractor E-stop; no auth by design (`TractorControlService.emergency_stop`). Cleared by `POST /api/v2/tractor/clear-emergency` |
| `POST /api/v2/control/emergency` | any | Latches the software E-stop and fans out through `execute_emergency_stop` (`backend/src/services/emergency_stop_service.py`) to every stop path in the process: the tractor, the RoboHAT drive/blade controller and the IBT-4 blade service, concurrently. The response carries the per-subsystem report (`emergency_stop`, `complete`); never gated on a session |
| `POST /api/v2/control/emergency-stop` | any (alias) | The same fan-out, always HTTP 200: `emergency_stop_active` is the software latch, `motors_stopped` / `blade_disabled` / `complete` are `true` only for confirmed stops, `subsystems` and `faults` say what each output did |
| `POST /api/v2/control/emergency_clear` | any | Operator-authenticated (`require_operator_auth`) and needs `confirmation: true`; clears the general latch, the IBT-4 latch and the RoboHAT hold. It does **not** clear a latched tractor E-stop or interlock: those need `POST /api/v2/tractor/clear-emergency` and are listed in `still_latched` |

A subsystem reports one of `stopped`, `simulated` (SIM_MODE), `unconfirmed`, `failed`, `not_present` or
`not_enabled`; absent or unconfirmed hardware never reads as "stopped" outside SIM_MODE. Outside SIM_MODE the IBT-4
blade output stays `unconfirmed` while `ibt4.gpio_enabled` is `false`. None of this replaces the hardwired E-stop
(audit G0).

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
│   ├── hardware.yaml         # differential-drive mower hardware
│   ├── limits.yaml           # safety limits
│   └── tractor.yaml          # Toro variant actuation (enabled: false by default)
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
Baseline on 2026-10-24 (commit 765c737): `SIM_MODE=1 pytest tests --ignore=tests/soak` gave **530 passed, 69 skipped,
0 failed** (that was the baseline before this branch's work; the integrated branch collects 4248 tests: 4221 passed, 27 skipped,
0 failed, see `docs/2026-10-24-integration-audit.md`, "Integration round 4"). The skips include placeholder tests that never run by default, gated perf tests and TDD stubs for
WebSocket and manual-control authentication. `tests/hil/` is a probe only and no CI job runs it or `tests/soak`.
Every passing safety test runs against fakes or simulation; none proves a physical outcome. Details:
[`docs/2026-10-24-integration-audit.md`](docs/2026-10-24-integration-audit.md) Part A (tests).

---

## Configuration

Configuration files live in `config/`. The earlier README showed a non-existent `config/safety.yaml` and an
`estop: gpio_pin: 17` that no code reads; the safety limits are in `config/limits.yaml`.

### Hardware Configuration (`config/hardware.yaml`, differential-drive mower; excerpt)
```yaml
gps:
  type: LC29H-DA       # Quectel LC29H(DA) HAT via UART at 115200
  uart_device: /dev/ttyAMA0
  baudrate: 115200
imu:
  type: BNO085
sensors:
  tof:
    - left
    - right
motor_controller: robohat-rp2040
blade_controller: ibt-4
```

NTRIP corrections are configured through `.env` (`NTRIP_HOST`, `NTRIP_PORT`, `NTRIP_MOUNTPOINT`, `NTRIP_USERNAME`,
`NTRIP_PASSWORD`, `NTRIP_SERIAL_DEVICE`) and `gps_ntrip_enabled` in `config/hardware.yaml`; see
[`docs/gps-ntrip-setup.md`](docs/gps-ntrip-setup.md).

### Safety Limits (`config/limits.yaml`; excerpt)
```yaml
estop_latency_ms: 100
tilt_threshold_degrees: 30.0
tilt_cutoff_latency_ms: 200
watchdog_timeout_ms: 1000
tof_obstacle_distance_meters: 0.2
ultrasonic_obstacle_distance_cm: 30.0
ultrasonic_enabled: false  # Heltec HaLow HAT owns GPIO4/9/10/11 on this Pi
```

These are configured thresholds, not measured performance. `estop_latency_ms` is validated only as
`watchdog_timeout_ms >= estop_latency_ms` (`backend/src/safety/safety_validator.py:37-38`). The two obstacle
thresholds above raise the obstacle interlocks that `ObstacleStopMonitor` publishes from the real sensor readings
(a hold, not an E-stop); the speed-dependent slow/stop distances come from the separate `obstacle_stop` block and the
GNSS bound from `gps_degradation`, both `validated: false` in `config/limits.yaml`.

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
| [Integration Audit (2026-10-24)](docs/2026-10-24-integration-audit.md) | Implemented vs target, hardware conflicts, safety review, development plan |
| [Tractor Platform](docs/tractor-platform.md) | Toro zero-turn variant design and physical decisions |
| [Tractor Acceptance Criteria](docs/tractor-acceptance-criteria.md) | Safety sign-off checklist for the Toro variant |

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
