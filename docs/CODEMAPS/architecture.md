<!-- Generated: 2026-10-06 | Files scanned: 474 | Token estimate: ~750 -->
# Architecture

LawnBerry Pi v3 is an autonomous mower. Three hosts are involved, and only the Pi is on the runtime safety path.

```
 operator browser ──HTTP/WS──► nginx / frontend/server.mjs (Express static + proxy)
                                   │  /api, /api/v2/ws/telemetry
                                   ▼
 ┌──────────────── Raspberry Pi 5 + Hailo-8L (systemd, /apps/lawnberry-pi) ────────────────┐
 │ FastAPI backend.src.main:app                                                              │
 │  middleware ─► api/ + api/routers/ ─► services/ ─► drivers/ ─► GPIO / I2C / UART / USB   │
 │                                     │            └► safety/ (E-stop, watchdog, interlocks)│
 │                                     ├► nav/ (pure algorithms)                             │
 │                                     ├► core/persistence ─► SQLite data/lawnberry.db       │
 │                                     └► websocket_hub ─► WS telemetry topics               │
 │ lawnberry-camera.service owns the camera ─unix sock─► camera_ipc_client (backend)         │
 └───────────────────────────────┬───────────────────────────────────────────────────────────┘
                                 │ WiFi HaLow, compact JSON UDP (≤1,200 B), thor_link_service
                                 ▼
   NVIDIA Thor: coverage strategist (nav/thor_strategist.py); decides WHERE, never WHETHER safe
   Workshop x86 (offline, $LB): Isaac Sim SDG, auto-labelling, YOLO training, Hailo compile
     (workshop/*.py, scripts/isaacsim-*.sh, scripts/hailo_compile.sh) → artifacts only
```

## Platforms
- **mower** (default): RoboHAT RP2040 drive + IBT-4 blade.
- **tractor** (dormant, `config/tractor.yaml enabled:false`): Toro 77502 zero-turn. Two lever servos plus throttle on a PCA9685, with starter/PTO relays.

## Data flow (telemetry)
```
drivers/sensors/* → sensor_manager → telemetry_hub / websocket_hub
  → topics telemetry.{navigation,power,sensors,tof,tractor,...}, system.*, safety_state
  → frontend services/websocket.ts → Pinia stores → views
```

## Control flow (drive command)
```
ControlView / teleop.py → POST /api/v2/control/drive | /tractor/* → middleware chain
  → motor_service | tractor_service → safety interlock / authorization check
  → robohat_rp2040 (serial) | pca9685_driver (I2C) ; watchdog + safety_monitor can E-stop
```

## Boot (lifespan, backend/src/main.py)
1. Load the config into `app.state`: hardware.yaml and limits.yaml via core/config_loader.
2. Start the GPS degradation monitor.
3. Run the safety-limits validation.
4. Start the RoboHAT service.
5. If enabled, start the tractor service, the tractor watchdog and the tractor safety monitor.

## Constraints (see docs/constitution.md)
- **Camera ownership:** camera-stream.service alone owns the camera; everything else consumes it over IPC.
- **Dependencies:** Coral/edgetpu packages are banned from the main env.
- **Simulation:** `SIM_MODE=1` makes every hardware path simulated (CI and off-Pi).
- **Driver imports:** hardware libraries are imported lazily inside functions.
