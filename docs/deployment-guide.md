# Edge Deployment Guide - LawnBerry Pi

This guide covers deploying and running the LawnBerry Pi autonomous mowing system on the Raspberry Pi 5 with Hailo 8L accelerator.

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
cd /home/kp/repos/lawnberry_pi

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
mkdir -p /home/kp/repos/lawnberry_pi/models

# Copy model files (from Thor training server or pre-trained)
# Models must be in Hailo HEF format
cp /path/to/lawnmower_vla.hef /home/kp/repos/lawnberry_pi/models/
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
  model_path: /home/kp/repos/lawnberry_pi/models/lawnmower_vla.hef
  target_fps: 10
  min_confidence: 0.5
  safety_threshold: 0.7
```

### NTRIP Configuration (for RTK GPS)

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your NTRIP credentials:

```env
NTRIP_HOST=your_caster.com
NTRIP_PORT=2101
NTRIP_MOUNTPOINT=your_mountpoint
NTRIP_USERNAME=your_username
NTRIP_PASSWORD=your_password
NTRIP_SERIAL_DEVICE=/dev/ttyAMA0
NTRIP_SERIAL_BAUD=115200
```

## Running the System

### Development Mode

```bash
# Activate virtual environment
source .venv/bin/activate

# Run in simulation mode (for testing)
SIM_MODE=1 python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8000

# Run with hardware
python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8000
```

### Production Deployment

Use systemd for automatic startup:

```bash
# Copy service file
sudo cp systemd/lawnberry.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable lawnberry
sudo systemctl start lawnberry

# Check status
sudo systemctl status lawnberry
```

### Enabling AI Control

Via REST API:

```bash
# Enable AI autonomous mode
curl -X POST http://localhost:8000/api/v2/ai/enable \
  -H "Content-Type: application/json" \
  -d '{"mode": "ai_autonomous"}'

# Check status
curl http://localhost:8000/api/v2/ai/status
```

Via Web Interface:
1. Navigate to http://your-pi-ip:8000
2. Go to Control Panel
3. Select "AI Autonomous" mode
4. Click "Start"

## Monitoring

### Real-time Telemetry

Access WebSocket telemetry at:
- `ws://localhost:8000/ws/telemetry`

### Performance Metrics

```bash
# Get AI inference metrics
curl http://localhost:8000/api/v2/ai/metrics
```

Key metrics to monitor:
- `avg_inference_time_ms` - Should be < 50ms
- `inferences_per_second` - Target: 10+ FPS
- `success_rate` - Should be > 0.95
- `safety_overrides` - Track safety interventions

### Health Checks

```bash
# Overall system health
curl http://localhost:8000/api/v2/health

# AI-specific health
curl http://localhost:8000/api/v2/ai/health

# Sensor status
curl http://localhost:8000/api/v2/sensors/status
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
   curl http://localhost:8000/api/v2/ai/status | jq .using_hardware
   ```

2. Check model is loaded:
   ```bash
   curl http://localhost:8000/api/v2/ai/status | jq .model_loaded
   ```

3. Review metrics:
   ```bash
   curl http://localhost:8000/api/v2/ai/metrics | jq .avg_inference_time_ms
   ```

## Safety Considerations

### Pre-flight Checklist

Before autonomous operation:

1. [ ] Emergency stop button accessible
2. [ ] Boundary geofence configured
3. [ ] Obstacle detection verified
4. [ ] GPS fix obtained (RTK preferred)
5. [ ] Battery charge > 50%
6. [ ] Clear weather conditions
7. [ ] Safety observer present

### Safety Interlocks

The system automatically stops if:
- Ultrasonic detects obstacle < 30cm
- IMU detects excessive tilt (> 25°)
- GPS fix lost for > 30 seconds
- AI confidence drops below threshold
- Battery voltage critical

### Emergency Stop

Methods to stop the mower:
1. Physical e-stop button
2. Web interface "Stop" button
3. API call: `POST /api/v2/control/emergency_stop`
4. Loss of network connection (failsafe)

## Updating the System

### Software Updates

```bash
# Pull latest code
cd /home/kp/repos/lawnberry_pi
git pull

# Update dependencies
pip install -e ".[hardware]"

# Restart service
sudo systemctl restart lawnberry
```

### Model Updates

```bash
# Stop service
sudo systemctl stop lawnberry

# Replace model file
cp /path/to/new_model.hef /home/kp/repos/lawnberry_pi/models/lawnmower_vla.hef

# Start service
sudo systemctl start lawnberry

# Verify new model loaded
curl http://localhost:8000/api/v2/ai/health
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

- Application logs: `journalctl -u lawnberry`
- Hailo logs: `/var/log/hailort.log`
- System logs: `/var/log/syslog`

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python -m uvicorn backend.src.main:app --host 0.0.0.0 --port 8000
```

### Common Log Patterns

```bash
# View AI inference logs
journalctl -u lawnberry | grep -i "ai\|inference\|hailo"

# View safety events
journalctl -u lawnberry | grep -i "safety\|emergency\|interlock"

# View sensor errors
journalctl -u lawnberry | grep -i "sensor\|gps\|camera"
```
