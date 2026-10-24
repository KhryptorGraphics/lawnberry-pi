# LawnBerry Pi v2 - Hardware Integration Guide

> Platform: differential-drive mower only (RoboHAT/Cytron/IBT-4); not the Toro tractor variant.

This guide documents wiring, pin assignments, and integration steps for Raspberry Pi 5 and Raspberry Pi 4B. Always disconnect power before wiring. Refer to `spec/hardware.yaml` for canonical configuration.

## Platform Notes
- Supported: Raspberry Pi OS Bookworm (64-bit) only
- Python 3.11.x required
- GPIO differences: Some UART pins differ between Pi 4B and Pi 5; use alternatives noted below.

## Pin Assignments (from spec/hardware.yaml)

### Blade Driver (IBT-4 H-Bridge)
- GPIO 24 → IN1 (Blade IN1)
- GPIO 25 → IN2 (Blade IN2)
- Power: 12V from battery through appropriate fuse
- Ground: Common ground with Raspberry Pi

Safety:
- Ensure E-stop physically cuts blade power path.
- Software interlocks: tilt, obstacle, watchdog enforced.

Software status (IBT-4 driver, 2026-10-24):
- A real `lgpio` adapter exists (`backend/src/drivers/blade/ibt4_lgpio.py`) behind `ibt4.gpio_enabled` in `config/hardware.yaml`. It defaults to `false` and is never armed in SIM_MODE; with it off the driver is a flag-only stub, GPIO24/25 are never driven, and outside SIM_MODE the general E-stop reports the blade output `unconfirmed`.
- It has never run on hardware; it is tested only against a fake `lgpio`. When armed, the E-stop reports the blade output `stopped` only after GPIO24/25 were written low and read back low. That does not observe the IBT-4 power stage or the blade motor.
- When armed, an E-stop or shutdown park on a service that was never initialised (for example right after a backend restart, when the lines are in an unknown state) first claims GPIO24/25 at level 0, drives both low and reads them back itself, so the blade output is reported `stopped` or `failed`, never `not_present`. With the gate off, a never-initialised service stays `not_present` and no GPIO is touched.
- GPIO25 (IN2) is the HaLow HAT's SPI IRQ line on the deployed Pi (audit C-03, owner action O-5; live `gpioinfo` showed it held by `mm610x_spi_irq_gpio`). The claim is therefore expected to fail and arming then fails closed (the E-stop reports the blade output `failed`) until the pin is moved; this has not been tried on the Pi.
- UNVERIFIED, measure before arming: the IBT-4 truth table (`ibt4.on_levels` is a placeholder), whether the blade motor needs PWM or an enable line (only static levels are driven), whether `gpio_read` of an output pin on the Pi 5 follows the pad (the "read back low" evidence depends on it), and the pin state after `gpio_free`, process exit and SIGKILL (the audit M-26 scope measurement, repeated on GPIO24/25). Do the first bench check with the blade motor disconnected.

### Time-of-Flight Sensors (VL53L0X x2)
- I2C Bus 1: SDA (GPIO 2), SCL (GPIO 3)
- Left Sensor Address: 0x29
- Right Sensor Address: 0x30
- Optional shutdown pins:
  - ToF Left Shutdown: GPIO 22
  - ToF Right Shutdown: GPIO 23
- Optional interrupts:
  - ToF Left IRQ: GPIO 6
  - ToF Right IRQ: GPIO 12

Driver behavior (backend/src/drivers/sensors/vl53l0x_driver.py):
- Uses Python VL53L0X bindings if present (modules: `VL53L0X` or `vl53l0x`).
- Falls back to previous reading if hardware access errors occur.
- Honors environment overrides:
  - TOF_BUS (default 1)
  - TOF_LEFT_ADDR (default 0x29)
  - TOF_RIGHT_ADDR (default 0x30)
  - TOF_RANGING_MODE (short|better_accuracy|best_accuracy|long_range|high_speed; default better_accuracy)
- You can also set these in `config/hardware.yaml` under `sensors.tof_config`.

Verification steps:
1. Ensure SIM_MODE=0 in the systemd unit or shell.
2. `sudo i2cdetect -y 1` should show 0x29 and 0x30.
3. Start the backend and open the dashboard; the ToF tiles should update with live distances.
4. If no readings appear, check journal logs for `vl53l0x` messages and verify the Python binding is installed.

### Environmental Sensor (BME280)
- I2C Bus 1: Address 0x76

### Power Monitor (INA3221)
- I2C Bus 1: Address 0x40
- Channel mapping per spec:
  - CH1: Battery
  - CH2: Unused
  - CH3: Solar Input
- **Contradiction (unresolved):** `ina3221_driver.py` and `config/hardware.yaml` use CH1 = Solar (0.0025 ohm), CH3 = Battery (0.0015 ohm). No shunt has been measured; do not change either side without a bench measurement of which input carries battery current.

### Victron SmartSolar BLE telemetry
- Optional replacement or supplement for INA3221 readings.
- Follow `docs/victron-ble-integration.md` to install the `victron-ble` CLI, patch the upstream scanner guard, and register the Instant Readout key.
- Configure the `victron` block in `config/hardware.yaml`; set `prefer_battery`, `prefer_solar`, and `prefer_load` to prioritize SmartSolar data when both sources are available. Keep secrets in the untracked `config/hardware.local.yaml` override so the repository never stores live keys.
- Backend automatically falls back to INA3221 values if Victron telemetry is temporarily unavailable.

### IMU (BNO085)
- Preferred: UART4 at 3,000,000 baud
- Pi 5: /dev/ttyAMA4 → GPIO 12 (TXD4), GPIO 13 (RXD4)
- Pi 4B: Alternative pins may be required; map to rpi4_alt_pin as specified (e.g., TXD4 on GPIO 24, RXD4 on GPIO 21). Verify with `gpioinfo`. (On the Pi 4B the overlay is `uart4`; the `rpi4_alt_pin` values in `spec/hardware.yaml` are **physical header pin numbers** (21/24), not BCM GPIOs.)

### GPS
- Preferred: USB (ZED-F9P RTK) → /dev/ttyACM*
- Alternative: UART0 for NEO-8M → /dev/serial0 at 115200 baud

#### RTK Positioning with NTRIP Corrections

For centimeter-level accuracy, configure RTK corrections via NTRIP. **See the comprehensive guide:**
- **[GPS RTK with NTRIP Configuration Guide](gps-ntrip-setup.md)** - Complete step-by-step instructions

**Quick Setup Summary:**

Two methods are supported:

1. **Method 1: Direct GPS Configuration** (Recommended)
   - Configure NTRIP in the ZED-F9P using u-blox u-center (Windows)
   - GPS connects directly to NTRIP caster
   - More reliable, no Pi dependency
   - Set `gps_ntrip_enabled: true` in `config/hardware.yaml`

2. **Method 2: Pi-Forwarded NTRIP**
   - Pi connects to NTRIP caster and forwards corrections to GPS
   - Set `gps_ntrip_enabled: true` in `config/hardware.yaml`
   - Configure `.env` with: `NTRIP_HOST`, `NTRIP_PORT`, `NTRIP_MOUNTPOINT`, `NTRIP_USERNAME`, `NTRIP_PASSWORD`, `NTRIP_SERIAL_DEVICE`
   - Restart backend: `sudo systemctl restart lawnberry-backend`

**Verification:**
```bash
# Check GPS status
curl http://localhost:8081/api/v2/sensors/health | jq '.gps'

# Look for fix_type: "RTK_FIXED" (best) or "RTK_FLOAT" (processing)
# Target accuracy: < 0.05 meters (5cm)
```

### Drive Controller: RoboHAT RP2040 → Cytron MDDRC10 (preferred)
- Serial link: /dev/serial0 or /dev/ttyACM0
- PWM and direction handled on RoboHAT; Pi communicates via serial API
- Encoders wired to RoboHAT for real-time counting

Alternative drive controller (fallback): L298N dual H-Bridge (basic PWM/dir via Pi GPIO, no encoder feedback).

## UART Configuration (Pi OS)
1. Enable UARTs in `raspi-config` (disable login shell on serial)
2. For Pi 5 IMU on UART4:
   - Add to `/boot/firmware/config.txt`:
     ```
     dtoverlay=uart4-pi5
     ```
     (`uart4` is BCM2711/Pi 4 only, GPIOs 8-11; Pi 5 needs `uart4-pi5`, GPIO12/13. Neither overlay is loaded on the deployed Pi.)
3. Reboot and verify devices:
   ```bash
   ls -l /dev/ttyAMA4 /dev/serial0 || true
   ```

## I2C Verification
```bash
sudo apt install -y i2c-tools
sudo i2cdetect -y 1
# Expect addresses: 0x29 (VL53L0X Left), 0x30 (VL53L0X Right), 0x3C (SSD1306 OLED), 0x76 (BME280), 0x40 (INA3221)
# NOTE: the deployed Pi sets dtparam=i2c_arm=off (GPIO3 is the HaLow HAT WAKE line), so /dev/i2c-1 does not exist there.
```

## Emergency Stop Wiring
- Hardware E-stop must cut power to blade driver and optionally signal RoboHAT input.
- Software API complements hardware E-stop for remote control.

## Testing Checklist
- Sensors health: GET /api/v2/sensors/health
- Telemetry: GET /api/v2/dashboard/telemetry (SIM_MODE=1 allowed)
- Blade control: POST /api/v2/control/blade (expect safety lockout by default)
- Drive control: POST /api/v2/control/drive (simulation or with RoboHAT)

## Troubleshooting
- No I2C devices: Check 3.3V and pull-ups; confirm I2C enabled
- UART not present: Ensure dtoverlay is set; verify no console on serial
- GPS not detected: Try different USB cable/port; check `dmesg`
- Blade doesn’t stop: Verify E-stop wiring and IBT-4 enable path; check software interlocks

## References
- spec/hardware.yaml (pin map and device list)
- docs/hardware-overview.md (system diagram)
- docs/OPERATIONS.md (operational procedures)

## Pin-conflict guard
`python3 scripts/check_hardware_pin_conflicts.py` merges the claims in `spec/hardware.yaml` (GPIO rows incl. UART4 TXD/RXD and HaLow HAT lines), `config/tractor.yaml` relays/PCA9685, and the legacy HC-SR04 driver pins (only when `config/limits.yaml` has `ultrasonic_enabled: true` **and** `config/hardware.yaml` `ultrasonic.backend` is `gpio_hcsr04`; the default `mcu_usb` backend, four JSN-SR04T behind a USB MCU, claims no Pi pin). Every conflict must be listed in `spec/known_conflicts.yaml` (pin, claimants, gap ID, reason, owner action); a new or stale entry fails the check, and an allow-listed conflict that involves a tractor claimant fails once `config/tractor.yaml` has `enabled: true`. The `deployment:` block in the spec records what the deployed Pi actually exposes (e.g. no header I2C bus 1) and drives the warnings. `claims with status: unverified` only warn.
