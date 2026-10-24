# Edge Deployment Guide - LawnBerry Pi

This guide covers deploying and running the LawnBerry Pi autonomous mowing system on the Raspberry Pi 5 with Hailo 8L accelerator.

> **Names and versions.** "LawnBerry Pi" is the product. The HTTP API is versioned separately
> (`/api/v2`, the backend reports `2.0.0`), and the README title carries the project generation
> ("v3"). systemd unit descriptions and this guide carry no version, so there is nothing to drift.
> The constitution has its own version (4.0.0). Capability statements follow the 2026-10-24
> audit, [`2026-10-24-integration-audit.md`](2026-10-24-integration-audit.md).

## Prerequisites

### Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| Raspberry Pi 5 | 8GB+ RAM | Main compute platform |
| Hailo 8L | M.2 AI accelerator | Edge inference |
| Stereo Camera | ELP-USB960P2CAM-V90 | Depth perception |
| Pi Camera 2 | IMX219 | RGB vision |
| GPS Module | LC29H(DA) | RTK positioning |
| Ultrasonic | HC-SR04 x3 | Obstacle detection |
| IMU | BNO085 | Orientation sensing |

### Software Requirements

- Raspberry Pi OS Bookworm (64-bit)
- Python 3.11+
- HailoRT 4.20.0+ (matching kernel driver version)

## Installation

### 1. System Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    gpsd gpsd-clients \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    i2c-tools

# Enable required interfaces
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_camera 0

# Reboot to apply changes
sudo reboot
```

### 2. Python Environment

```bash
# Navigate to project directory
cd /apps/lawnberry-pi

# Create virtual environment (use system site-packages for libcamera)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install project dependencies
pip install -e ".[hardware]"
```

### 3. Hailo Installation

```bash
# Install Hailo runtime from Raspberry Pi repository
sudo apt-get install -y hailo-all

# Verify installation
hailortcli fw-control identify
# Expected: Hailo-8L device detected

# Check version compatibility
hailortcli --version
# Should match kernel module version
```

**Version Compatibility Note**: The kernel driver and userspace library must match. If you see version mismatch errors:

1. Check current versions:
   ```bash
   dmesg | grep hailo
   dpkg -l | grep hailo
   ```

2. Resolution options:
   - Wait for Pi repo update
   - Download matching version from Hailo Developer Zone
   - Use simulation mode for testing

### 4. Model Deployment

Place trained models in the models directory:

```bash
# Create models directory if needed
mkdir -p /apps/lawnberry-pi/data/models

# Copy model files (from Thor training server or pre-trained)
# Models must be in Hailo HEF format
cp /path/to/lawnmower_vla.hef /apps/lawnberry-pi/data/models/
```

## Configuration

### Hardware Configuration

Edit `config/hardware.yaml`:

```yaml
# GPS Configuration
gps:
  type: LC29H-DA
  uart_device: /dev/ttyAMA0
  baudrate: 115200

# Camera Configuration
camera:
  primary:
    backend: picamera2
    width: 1920
    height: 1080
    fps: 30
  stereo:
    backend: opencv
    device: /dev/video0
    width: 2560
    height: 960
    fps: 30

# AI Configuration
ai:
  # Optional -- defaults to $LAWNBERRY_DATA_DIR/models/lawnmower_vla.hef
  model_path: /apps/lawnberry-pi/data/models/lawnmower_vla.hef
  target_fps: 10
  min_confidence: 0.5
  safety_threshold: 0.7
```

### Runtime environment (`.env`)

The backend unit loads `<root>/.env` (`EnvironmentFile=`). Start from the template:

```bash
cp .env.example .env
```

Fill in the NTRIP settings if you use RTK corrections:

```env
NTRIP_HOST=your_caster.com
NTRIP_PORT=2101
NTRIP_MOUNTPOINT=your_mountpoint
NTRIP_USERNAME=your_username
NTRIP_PASSWORD=your_password
NTRIP_SERIAL_DEVICE=/dev/ttyAMA0
NTRIP_SERIAL_BAUD=115200
```

Do not set the two operator-authentication keys by hand unless you want to: the installer
(below) generates `LAWN_BERRY_AUTH_SECRET` and validates `LAWN_BERRY_OPERATOR_CREDENTIAL`.

## Running the System

Ports: the backend listens on **8081** and the web UI (`frontend/server.mjs`, proxying `/api`
to the backend) on **3000**.

### Development Mode

```bash
# Activate virtual environment
source .venv/bin/activate

# Run in simulation mode (for testing)
SIM_MODE=1 python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8081

# Run with hardware
python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8081
```

### Production Deployment

The units in `systemd/` are templates (`@LB_USER@`, `@LB_GROUP@`, `@LB_ROOT@`): do not copy
them into `/etc/systemd/system` by hand. `systemd/install_services.sh` renders them for the
real account and checkout, provisions `<root>/.env`, installs and enables them:

```bash
# See exactly what would be installed (units rendered for the account and root); writes nothing.
bash systemd/install_services.sh --dry-run

# Install. The installer REFUSES (exit 3, nothing changed) without an operator credential of
# at least 12 characters that is not a known default; give it one without typing it on the
# command line:
f=$(mktemp) && ${EDITOR:-nano} "$f"        # one line: the credential
sudo bash systemd/install_services.sh --operator-credential-file "$f"; shred -u "$f"

# Re-running is safe: LAWN_BERRY_AUTH_SECRET (the token signing key) is generated once and
# kept, so operator sessions survive an update. --rotate-auth-secret replaces it.
```

It defaults to the account in `$SUDO_USER` and to the checkout it lives in
(`--user`, `--group`, `--root` override). It enables the units and starts nothing:

```bash
sudo systemctl start lawnberry-database lawnberry-backend lawnberry-health \
     lawnberry-frontend lawnberry-camera lawnberry-remote-access
sudo systemctl status lawnberry-backend
```

**On the shed Pi do not stop or restart `lawnberry-backend` without physical access**: it also
stops `lawnberry-camera` (`PartOf=`), and on 2026-10-06 the HaLow radio failed about 10 s after
such a stop and stayed dark until a power cycle (`autonomy-toolchain-matrix.md`).

HaLow units that exist only on the deployed Pi (`halow-bringup`, `halow-failover`,
`halow-lan-routes.timer`, `wifi-halow-*`, `wificommand`, the `wpa_supplicant@halow0` drop-in and
`lawnberry-backend.service.d/thor-upload.conf`) are not in the repository yet: the Pi was
unreachable when this was written (audit, "Finish track: deployment"). Read them read-only
(`systemctl cat <unit>`; check for credentials before committing) and add them under `systemd/`;
`<unit>.d/*.conf` drop-ins next to a unit are rendered and installed by the installer.

### Operator authentication

The backend unit sets `OPERATOR_AUTH_REQUIRED=1`. Log in with the credential you installed
(the web UI form takes any username and that password), then send the bearer token:

```bash
TOKEN=$(curl -s -X POST http://localhost:8081/api/v2/auth/login \
  -H "Content-Type: application/json" \
  -d '{"credential": "<your operator credential>"}' | jq -r .access_token)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8081/api/v2/auth/profile
```

There is no built-in credential: with none configured the backend still starts, but nobody can
log in (the E-stop routes stay open by design). A 401 on every login means the credential in
`<root>/.env` is missing, a published default, or not what you typed.

### Enabling AI Control

Via REST API:

```bash
# Enable AI autonomous mode
curl -X POST http://localhost:8081/api/v2/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"mode": "ai_autonomous"}'

# Check status
curl http://localhost:8081/api/v2/ai/status
```

Via Web Interface:
1. Navigate to http://your-pi-ip:3000
2. Go to Control Panel
3. Select "AI Autonomous" mode
4. Click "Start"

## Monitoring

### Real-time Telemetry

Access WebSocket telemetry at:
- `ws://localhost:8081/api/v2/ws/telemetry`

### Performance Metrics

```bash
# Get AI inference metrics
curl http://localhost:8081/api/v2/ai/metrics
```

Key metrics to monitor:
- `avg_inference_time_ms` - Should be < 50ms
- `inferences_per_second` - Target: 10+ FPS
- `success_rate` - Should be > 0.95
- `safety_overrides` - Track safety interventions

### Health Checks

```bash
# Overall system health
curl http://localhost:8081/api/v2/health

# AI-specific health
curl http://localhost:8081/api/v2/ai/health

# Sensor status
curl http://localhost:8081/api/v2/sensors/health
```

## Troubleshooting

### Hailo Not Detected

```bash
# Check PCIe connection
lspci | grep Hailo

# Check kernel module
lsmod | grep hailo

# Check device node
ls -la /dev/hailo*

# View kernel messages
dmesg | grep hailo
```

### Camera Issues

```bash
# List video devices
v4l2-ctl --list-devices

# Test stereo camera
python scripts/test_stereo_camera.py

# Test Pi camera
rpicam-hello --list-cameras
```

### GPS Not Getting Fix

```bash
# Check serial port
ls -la /dev/ttyAMA0

# Test NMEA output
sudo minicom -D /dev/ttyAMA0 -b 115200

# Check if serial-getty is disabled
sudo systemctl status serial-getty@ttyAMA0
```

### AI Inference Slow

1. Verify Hailo hardware is being used:
   ```bash
   curl http://localhost:8081/api/v2/ai/status | jq .using_hardware
   ```

2. Check model is loaded:
   ```bash
   curl http://localhost:8081/api/v2/ai/status | jq .model_loaded
   ```

3. Review metrics:
   ```bash
   curl http://localhost:8081/api/v2/ai/metrics | jq .avg_inference_time_ms
   ```

## Safety Considerations

The safety state of this machine is what the 2026-10-24 audit records (A.10 and Part D of
[`2026-10-24-integration-audit.md`](2026-10-24-integration-audit.md)), not what the lists below
once promised. Nothing here clears the machine for unattended motion.

### Pre-flight Checklist

Before autonomous operation:

1. [ ] Emergency stop reachable (see below: there is no hardwired E-stop yet)
2. [ ] Boundary geofence configured
3. [ ] Obstacle detection verified
4. [ ] GPS fix obtained (RTK preferred)
5. [ ] Battery charge > 50%
6. [ ] Clear weather conditions
7. [ ] Safety observer present

### Safety Interlocks

Design intent, not a guarantee: automatic stops on an obstacle under 30 cm, excessive tilt,
GPS loss, low AI confidence and a critical battery. As of the audit the ultrasonic array is
disabled in the deployed build (HaLow pin conflict), obstacle triggers are advisory (N-11), the
tilt cutoff exists on the tractor path only and the BNO085 driver is a stub (N-06), and the GPS
degradation monitor does not observe real missions (N-31). Verify each one on the machine before
relying on it.

### Emergency Stop

Methods to stop the mower:
1. A hardwired E-stop: **not built** (H-08, audit D.1). Do not assume one exists.
2. Web interface "Stop" button
3. API (unauthenticated by design): `POST /api/v2/control/emergency-stop` (alias of
   `/control/emergency`). It latches the software E-stop and drives every stop path that exists;
   the response says per subsystem what was actually confirmed
   (`stopped`/`simulated`/`unconfirmed`/`failed`/`not_present`), so read it. Toro variant:
   `POST /api/v2/tractor/emergency-stop`.
4. Loss of the operator link: the tractor command deadman and the RoboHAT command expiry return
   the drive to neutral/stop, but there is no general operator-link failsafe (audit D.5).

Clearing needs operator authentication: `POST /api/v2/control/emergency_clear`
(`{"confirmation": true}`) and, for the tractor, `POST /api/v2/tractor/clear-emergency`.

## Updating the System

### Software Updates

```bash
# Pull latest code
cd /apps/lawnberry-pi
git pull

# Update dependencies
pip install -e ".[hardware]"

# Re-render and install the units (keeps .env and the signing secret)
sudo bash systemd/install_services.sh

# Restart: on the shed Pi only with physical access (see above)
sudo systemctl restart lawnberry-backend
```

### Model Updates

```bash
# Stop service (physical access on the shed Pi, see above)
sudo systemctl stop lawnberry-backend

# Replace model file
cp /path/to/new_model.hef /apps/lawnberry-pi/data/models/lawnmower_vla.hef

# Start service
sudo systemctl start lawnberry-backend

# Verify new model loaded
curl http://localhost:8081/api/v2/ai/health
```

## Performance Tuning

### Optimizing Inference

1. **Reduce input resolution**: Smaller images = faster inference
2. **Batch processing**: Group sensor updates for efficiency
3. **Async operations**: Use non-blocking I/O throughout

### Memory Management

```bash
# Monitor memory usage
free -h

# Check Python memory
ps aux | grep python

# View GPU memory (if applicable)
vcgencmd get_mem gpu
```

### Power Management

For battery-powered operation:
- Reduce camera resolution when not needed
- Lower inference FPS during idle
- Disable unused sensors
- Use governor: `cpufreq-set -g powersave`

## Logs and Debugging

### Log Locations

- Application logs: `journalctl -u lawnberry-backend`
- Hailo logs: `/var/log/hailort.log`
- System logs: `/var/log/syslog`

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8081
```

### Common Log Patterns

```bash
# View AI inference logs
journalctl -u lawnberry-backend | grep -i "ai\|inference\|hailo"

# View safety events
journalctl -u lawnberry-backend | grep -i "safety\|emergency\|interlock"

# View sensor errors
journalctl -u lawnberry-backend | grep -i "sensor\|gps\|camera"
```
