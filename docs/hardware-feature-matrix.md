# LawnBerryPi Hardware Feature Matrix

**Version:** 1.0  
**Purpose:** Comprehensive hardware configuration guide and capability matrix  
**Last Updated:** December 2024

**Audit annotations (2026-10-24):** platform tags and corrected rows added by the integration audit
([`2026-10-24-integration-audit.md`](2026-10-24-integration-audit.md)). In this document "✅ Implemented" had come to
mean "a driver file exists", not "verified on hardware"; rows the audit found wrong now say so explicitly.

> **Platform scope (H-11).** The tables describe the **differential-drive LawnBerry mower** (RoboHAT → Cytron
> MDDRC10, IBT-4 blade driver, LiFePO4 + solar) except the "Ride-On Tractor Platform (Alternate Configuration)"
> tables, which are the **Toro TimeCutter variant**. They are different wiring harnesses. The GPIO and I2C tables
> near the end are the mower registry only and omit the Toro PCA9685, the BNO085 UART, the HaLow radio and the
> legacy ultrasonic pins.

## Overview

This document provides a complete matrix of hardware components, their capabilities, configuration options, and compatibility information. Use this guide to understand what features are available with different hardware configurations and plan system deployments.

---

## Core Processing Platform

### Raspberry Pi 5 Configuration

| Component | Specification | Status | Notes |
|-----------|---------------|--------|-------|
| **CPU** | ARM Cortex-A76 Quad-core 2.4GHz | ✅ Implemented | Sufficient for all processing needs |
| **RAM** | 8GB LPDDR4X | ✅ Implemented | Supports advanced computer vision |
| **Storage** | MicroSD 64GB+ Class 10 | ✅ Implemented | High-speed card recommended |
| **GPIO** | 40-pin header | ⚠️ Conflicts found (audit 2026-10-24) | The earlier "100% compliant" claim covered only `spec/hardware.yaml` + `config/tractor.yaml`. The HaLow HAT's GPIO3/5/7/8-11/25 are not in that registry and header I2C1 is disabled on the deployed Pi (see "Pin Usage" below; H-02, H-17) |
| **Camera Interface** | CSI connector | ✅ Implemented | 1920x1080@30fps capability |
| **USB Ports** | 4x USB 3.0 | ✅ Utilized | GPS, TPU, external devices |
| **Network** | Gigabit Ethernet + WiFi | ✅ Implemented | Dual connectivity options |

**Performance Capabilities:**
- Concurrent processing of 11 microservices
- Real-time computer vision at 30fps
- WebSocket streaming to multiple clients
- Complex mowing pattern calculations
- Sensor fusion processing at 100Hz

---

## Navigation and Positioning Systems

### GPS/GNSS Configuration

| Component | Model | Interface | Status | Accuracy | Features |
|-----------|-------|-----------|--------|----------|----------|
| **Primary GPS** | **Unreconciled (H-09):** SparkFun GPS-RTK-SMA (u-blox ZED-F9P) per `spec/hardware.yaml:9`, `hardware.md`; Quectel LC29H-DA per `config/hardware.yaml:9-11`, `docs/autonomy-toolchain-matrix.md:14` | USB `/dev/ttyACM0` (ZED-F9P) or UART0 `/dev/ttyAMA0` (LC29H-DA) | ⚠️ `gps_driver.py` supports both; the fitted receiver is UNVERIFIED (live Pi 2026-10-24: no `/dev/ttyACM*`, `/dev/ttyAMA0` present, one CP2102N on `/dev/ttyUSB0`) | <10cm RTK (target, unmeasured) | Real-time corrections |
| **RTK Corrections** | NTRIP caster (Ohio DOT per the earlier text) | Internet | ⚠️ Client implemented (`backend/src/services/ntrip_client.py`); live status not evidenced (no `.env` on the deployed Pi); LC29H-BS local base MISSING (H-09, H-22) | <2cm (target, unmeasured) | Caster-to-serial RTCM forwarding |
| **Backup GPS** | Pi Hat GPS (optional) | GPIO UART | ⚠️ Optional | 3-5m | Fallback capability |

**GPS Feature Matrix:**
- ✅ Real-time positioning
- ✅ RTK correction integration  
- ✅ Multi-constellation support (GPS, GLONASS, Galileo)
- ✅ Autonomous fix acquisition
- ✅ Position accuracy monitoring
- ⚠️ Backup GPS configuration (optional)

### Inertial Measurement Unit (IMU)

| Component | Model | Interface | Status | Capabilities |
|-----------|-------|-----------|--------|--------------|
| **IMU** | BNO085 9-DOF | UART4 `/dev/ttyAMA4` per spec — **not enabled on the deployed Pi** (no `uart4-pi5` overlay, device absent); `sensor_manager.py:188` probes I2C 0x4A/0x4B instead | ⚠️ **Driver STUBBED** (`backend/src/drivers/sensors/bno085_driver.py:69-78` returns constants on real hardware); transport UNVERIFIED (H-03) | Full orientation (chip feature; not read by the backend) |
| **Accelerometer** | Built-in 3-axis | Hardware | ⚠️ Chip capability only (driver stub) | ±16g range |
| **Gyroscope** | Built-in 3-axis | Hardware | ⚠️ Chip capability only (driver stub) | ±2000°/s |
| **Magnetometer** | Built-in 3-axis | Hardware | ⚠️ Chip capability only (driver stub) | Compass heading |

**IMU Capabilities (BNO085 chip features; none is delivered by the backend until the driver reads real hardware):**
- ⚠️ Tilt and slope detection (on real hardware the IMU interface is ONLINE only after an I2C ACK at 0x4A/0x4B
  and the stub then reports roll = pitch = 0, so the 30° tilt cutoff cannot trip; `sensor_manager.py:186-190`,
  `bno085_driver.py:69-78`)
- ⚠️ Collision impact detection
- ⚠️ Heading determination
- ⚠️ Motion detection
- ⚠️ Stability monitoring
- ⚠️ Drop detection

---

## Sensor Systems

### Obstacle Detection Sensors

| Sensor Type | Model | Interface | Range | Status | Use Case |
|-------------|-------|-----------|-------|--------|----------|
| **ToF Left** | VL53L0X | I2C (0x29) | 30-2000mm | ⚠️ Driver implemented; header I2C1 is disabled on the deployed Pi so the sensor is unreachable there (H-17); INT pin GPIO6 is not read by any code | Left side obstacles |
| **ToF Right** | VL53L0X | I2C (0x30) | 30-2000mm | ⚠️ As above; INT pin GPIO12 also collides with BNO085 TXD4 in `spec/hardware.yaml` (H-03) | Right side obstacles |
| **Mono Vision** | Raspberry Pi Camera 2 | CSI | Visual range | ✅ Implemented | RGB scene capture |
| **Stereo Vision** | ELP-USB960P2CAM-V90 | USB | Visual range with depth cues | ⚠️ Capture only (2560x960 split, `stereo_camera_driver.py`); **stereo depth MISSING** (no disparity or calibration code, H-26); a UVC camera `32e4:9750` enumerates on the live Pi | Stereo capture and recording |
| **Ultrasonic** | **4x JSN-SR04T** (owner requirement, H-19); variant, mode and voltage UNVERIFIED | Trigger/echo vs serial TBD; no GPIO allocation is free of HaLow conflicts (H-02) | Near-field deck clearance; blind zone to be measured | ⚠️ **Software only, UNVERIFIED on hardware** (T2 item 6): firmware for a dedicated RP2040 over USB serial (`firmware/ultrasonic-mcu/`, no Pi header pin), host driver `jsn_sr04t_driver.py`, 4-channel `UltrasonicSensorInterface`; shipped OFF (`ultrasonic_enabled: false`) and fail-closed until M-40. MCU pins, JSN-SR04T mode and levels, blind zone and thresholds are TBD. The legacy 3x HC-SR04 driver (`ultrasonic_driver.py`) stays selectable (`ultrasonic.backend: gpio_hcsr04`) and uses HaLow SPI0 pins | Deck-width obstacle detection (target) |

**Sensor Fusion Matrix (corrected 2026-10-24, H-26):** there is no fused obstacle model. `ObstacleDetector` (class in
`backend/src/services/navigation_service.py`) still turns the two ToF sensors into advisory flags only (distance
< 1000, unit assumed mm) with a `Position(0, 0)` placeholder. What can hold the machine is the separate
fail-closed `ObstacleGuard` (`backend/src/safety/obstacle_stop.py`, README "Obstacle stop"), which the mission loop
`go_to_waypoint` consults through `_motion_gate()` every control period; every threshold in it is UNVALIDATED
(audit M-43). `tests/contract/test_sensor_fusion.py` covers the pose-fusion route (`/api/v2/fusion/state`), not
obstacle fusion (audit WP-P2-06 does not exist yet).
- ⚠️ ToF-only obstacle flag (advisory)
- ❌ Cross-validation between sensors (no second obstacle source is connected)
- ⚠️ Confidence is a constant (0.8 per ToF hit)
- ⚠️ Real-time processing: not measured
- ❌ Environmental adaptation

### Environmental Monitoring

| Parameter | Sensor | Interface | Range | Status | Purpose |
|-----------|--------|-----------|-------|--------|---------|
| **Temperature** | BME280 | I2C (0x76) | -40°C to +85°C | ✅ Implemented | Operating limits |
| **Humidity** | BME280 | I2C (0x76) | 0-100% RH | ✅ Implemented | Moisture detection |
| **Pressure** | BME280 | I2C (0x76) | 300-1100 hPa | ✅ Implemented | Weather prediction |
| **Light Level** | Optional sensor | I2C/ADC | Variable | ⚠️ Optional | Dawn/dusk detection |

---

## Vision and AI Systems

### Camera Configuration

| Component | Specification | Status | Capabilities |
|-----------|---------------|--------|--------------|
| **Mono Camera** | Raspberry Pi Camera 2 8MP | ✅ Implemented | RGB capture and operator video |
| **Stereo Camera** | ELP-USB960P2CAM-V90 | ✅ Implemented | Left/right synchronized capture |
| **Lens** | Fixed focus | ✅ Standard | Baseline field of view |
| **Night Vision / Pan-Tilt** | Not in baseline | ❌ Not in Baseline | Future work only if spec changes |

**Vision Processing Capabilities:**
- ✅ Real-time object detection
- ✅ Obstacle identification
- ✅ Edge detection for boundaries
- ✅ Color-based grass detection
- ⚠️ Advanced AI with TPU (optional)

### AI Acceleration (Optional)

| Component | Model | Interface | Status | Performance Gain |
|-----------|-------|-----------|--------|------------------|
| **Hailo-8L** | M.2 AI accelerator | PCIe/M.2 | ⚠️ Optional Baseline | Edge inference target |
| **Google Coral TPU** | USB Accelerator | USB 3.0 | ⚠️ Optional Baseline | Alternate edge acceleration |
| **Neural Compute Stick** | Intel Movidius | USB 3.0 | ❌ Not in Baseline | Not supported |
| **Jetson Nano** | NVIDIA | GPIO/USB | ❌ Not in Baseline | Not supported |

> **Hailo identification (2026-10-24).** `lspci` on the deployed Pi lists `0001:01:00.0 Hailo-8 AI Processor`;
> the PCI ID is shared by Hailo-8 and Hailo-8L, so `lspci` cannot tell them apart. `spec/hardware.yaml:37` says
> "Hailo-8 AI Hat"; `docs/autonomy-toolchain-matrix.md:21-26` records `hailortcli fw-control identify` →
> `HAILO8L`, part `HM21LB1C2LAE` (2026-10-06), and measured detector throughput on HailoRT 4.24.0
> (yolo26s 41.7 FPS; `:87`). Re-run `hailortcli fw-control identify` on the current unit and photograph the
> module label; the HEF target must match the device (`HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE` was observed for
> hailo8 builds on the 8L).

**AI Enhancement Matrix:**
- ❌ TPU-accelerated object detection
- ❌ Advanced computer vision models  
- ❌ Real-time semantic segmentation
- ❌ Custom lawn-specific AI models
- ✅ CPU-based OpenCV processing (current)

---

## Power Management System

### Battery Configuration

| Component | Specification | Status | Capabilities |
|-----------|---------------|--------|--------------|
| **Primary Battery** | 30Ah LiFePO4 12V | ✅ Implemented | 8-12 hours operation |
| **Chemistry** | Lithium Iron Phosphate | ✅ Confirmed | Safe, long-life |
| **BMS** | Built-in protection | ✅ Active | Overcurrent, thermal |
| **Backup Battery** | Optional 12V SLA | ⚠️ Optional | Emergency power |

**Battery Management Features:**
- ✅ Real-time voltage/current monitoring
- ✅ State of charge calculation
- ✅ Health monitoring
- ✅ Temperature protection
- ✅ Low voltage cutoff
- ⚠️ Automatic shutdown (requires RP2040 enhancement)

### Solar Charging System

| Component | Specification | Status | Performance |
|-----------|---------------|--------|-------------|
| **Solar Panel** | 30W Monocrystalline | ✅ Implemented | Peak 30W output |
| **Charge Controller** | 20A MPPT | ✅ Implemented | 95%+ efficiency |
| **Mounting** | Adjustable tilt | ✅ Physical | Seasonal optimization |
| **Tracking** | Optional sun tracking | ❌ Not Implemented | 20-30% gain potential |

**Solar Capabilities:**
- ✅ Maximum Power Point Tracking
- ✅ Battery protection during charging
- ✅ Efficiency monitoring
- ✅ Weather-based optimization
- ❌ Automatic sun tracking (future enhancement)

### Power Monitoring

| Channel | Monitor Point | Sensor | Status | Purpose |
|---------|---------------|--------|--------|---------|
| **Channel 1** | Solar Input | INA3221 | ✅ Active | Generation monitoring |
| **Channel 2** | Reserved | INA3221 | ⚠️ Available | Future expansion |
| **Channel 3** | Battery Bank | INA3221 | ✅ Active | Consumption monitoring |

> **Channel map disputed (audit 2026-10-24).** This table follows the driver
> (`backend/src/drivers/sensors/ina3221_driver.py:3-6,101`: CH1 solar, CH3 battery; `config/hardware.yaml`
> shunt comments). `spec/hardware.yaml:61-70`, `docs/hardware-integration.md` ("Power Monitor (INA3221)") and
> the "Verify INA3221 documented channel mapping" step of `.github/workflows/hardware-guard.yml` say CH1
> battery / CH3 solar. Which shunt carries which current is unmeasured.

---

## Motor and Drive Systems

### Drive Motors

| Component | Specification | Status | Control Method |
|-----------|---------------|--------|----------------|
| **Left Motor** | 12V Worm Gear DC | ✅ Implemented | PWM via RoboHAT → MDDRC10 |
| **Right Motor** | 12V Worm Gear DC | ✅ Implemented | PWM via RoboHAT → MDDRC10 |
| **Motor Driver (Drive)** | Cytron MDDRC10 | ✅ Implemented | Dual H-bridge (via RoboHAT) |
| **Encoders** | Hall Effect | ✅ Implemented | Speed feedback |

**Drive Capabilities:**
- ✅ Variable speed control (0-100%)
- ✅ Precise directional control
- ✅ Speed feedback monitoring
- ✅ Stall detection
- ✅ Emergency stop capability

### Cutting System

| Component | Specification | Status | Safety Features |
|-----------|---------------|--------|-----------------|
| **Blade Motor** | 997 DC Motor | ✅ Implemented | High-speed cutting |
| **Blade Driver** | IBT-4 H-Bridge | ❌ **STUBBED as shipped**: `ibt4.gpio_enabled` is `false` in `config/hardware.yaml`, so `IBT4BladeDriver` (`backend/src/drivers/blade/ibt4_gpio.py`) uses the flag-only `_GPIOAdapter` (no GPIO library is imported, GPIO24/25 are never driven). A real `lgpio` adapter (`ibt4_lgpio.py`) exists behind that flag and has **never run on hardware**; GPIO25 is also the HaLow IRQ line on the deployed Pi (H-02c), so arming it is expected to fail closed | Bidirectional control (GPIO24/25): not functional |
| **Safety Switch** | Tilt cutoff | ⚠️ Tractor path only, and it cannot trip on real hardware at baseline (see IMU rows); the mission loop re-asserts `blade_engaged` every 0.2 s tick (`go_to_waypoint` in `navigation_service.py` calls `tractor.apply(...)` with `blade_engaged=bool(waypoint.blade_on)` every `control_period_s`) | Auto-disable on tilt: not demonstrated |
| **Blade Guard** | Physical protection | ✅ Required | User safety |

### Ride-On Tractor Platform (Alternate Configuration)

A second, constitutionally-recognized platform (Constitution Principle V):
a converted 50" Toro TimeCutter zero-turn mower (gas engine, twin-lever
hydrostatic drive) instead of the differential-drive chassis above. No
steering, gas pedal, clutch, or gear selector — each drive lever's position
sets that side's speed/direction directly. Enabled via `config/tractor.yaml`
`enabled: true`; see `docs/tractor-platform.md` and `docs/tractor-acceptance-criteria.md`.

| Component | Specification | Status | Notes |
|-----------|---------------|--------|-------|
| **left_lever** | Positional, PCA9685 I2C PWM (ch 0) | ⚠️ Driver and service implemented and unit-tested in SIM (`pca9685_driver.py`, `tractor_service.py`); the header I2C bus the config names does not exist on the deployed Pi; no `OE` failsafe (H-04, H-08) | −1 (full reverse)..0 (neutral)..+1 (full forward) |
| **throttle** | Positional, PCA9685 I2C PWM (ch 1) | ⚠️ As above; throttle actuator model not yet identified | 0 (idle)..1 (full RPM) |
| **right_lever** | Positional, PCA9685 I2C PWM (ch 2) | ⚠️ As for left_lever | −1 (full reverse)..0 (neutral)..+1 (full forward) |
| **Starter** | Momentary GPIO relay (BCM5, `relays.starter` in `config/tractor.yaml`) | ⚠️ Software path exists (`RelayActuator` in `tractor_actuators.py`), **not commissioned**: GPIO5 is the HaLow RESET line on the deployed Pi (H-02b); relay-board polarity and boot-time state unmeasured; autonomous crank is blocked by the OEM PARK interlock (H-06) | Crank pulse |
| **Blade PTO** | Latching GPIO relay (BCM26); the owner reports a **momentary** OEM PTO switch (H-21) | ⚠️ Software path exists, **not commissioned**; no feedback that the PTO is actually off | Reassigned off GPIO 6 (ToF Left Interrupt conflict). The pin checker finds no registry claim on GPIO26 (the registry includes the HaLow HAT rows), but the Heltec reference overlay, not loaded on the deployed Pi, would configure it as a pulled-up input: confirm the pin state with the HAT fitted |

**Known gaps (corrected 2026-10-24):** the earlier text here said `tractor_service.py`'s actuator logic was
"mid-rework" from the superseded 7-actuator set; that rework landed (`docs/RELEASE_NOTES.md`, 2026-09-01 entry) and
`config/tractor.yaml` still ships `enabled: false`. What remains is physical and safety work: no independent
hardwired E-stop, no `OE`/servo-power failsafe, no I2C route on the deployed Pi, an unverified PTO circuit and no
verified lever neutral. Do not treat lever actuation as field-ready until
`docs/tractor-acceptance-criteria.md` is satisfied on real hardware.

---

## Communication Systems

### Primary Communication

| Interface | Protocol | Status | Usage |
|-----------|----------|--------|-------|
| **WiFi** | 802.11n/ac | ✅ Implemented | Primary connectivity |
| **Ethernet** | Gigabit | ✅ Available | Backup/setup |
| **Bluetooth** | 5.0 | ⚠️ Available | Optional RC/maintenance |
| **Wi-Fi HaLow (802.11ah)** | Morse Micro MM6108 on SPI0 (HAT identity inconsistent across docs: Heltec HT-HC01P V2 vs Seeed WM6108) | ⚠️ Measured working at one location (2026-10-06, 2026-10-13: ~4.8 Mbit/s TCP, 0% loss; `docs/autonomy-toolchain-matrix.md:30-70`); `halow0` was DOWN on the live Pi during the 2026-10-24 audit (traffic via `eth0`/`tun0`) | Pi to AP/Thor link; occupies GPIO3/5/7/8-11/25 |

### Internal Communication

| Interface | Protocol | Devices | Status |
|-----------|----------|---------|--------|
| **I2C Bus** | 400kHz | Sensors, Display | ⚠️ Header I2C1 is **disabled** on the deployed Pi (`dtparam=i2c_arm=off`; GPIO3 is HaLow WAKE); `/dev/i2c-1` is absent (H-17) |
| **UART** | Various baud | GPS, IMU | ⚠️ UART0 `/dev/ttyAMA0` present (GPS per config); UART4 for the BNO085 is not enabled (the Pi 5 needs `uart4-pi5`; overlay `uart4` is BCM2711-only) |
| **USB** | USB 3.0 | GPS, TPU | ⚠️ Live Pi 2026-10-24: ELP `32e4:9750` and a CP2102N `10c4:ea60` enumerate; no ZED-F9P, RoboHAT or Coral seen (sensors are detached at this site, so absence is not proof of design) |
| **GPIO** | Digital I/O | Motors, switches | ⚠️ Conflicts with the HaLow HAT; see "Pin Usage" |

---

## Display and User Interface

### On-Board Display

| Component | Specification | Status | Capabilities |
|-----------|---------------|--------|--------------|
| **OLED Display** | SSD1306 128x64 | ⚠️ UNVERIFIED: no OLED driver in `backend/src/` (only a baseline entry, the "OLED Display" component in `hardware_baseline.py`) | Status information |
| **Interface** | I2C (0x3c) | ✅ Active | Real-time updates |
| **Backlight** | None (OLED) | ✅ Self-lit | Day/night readable |

**Display Information:**
- ✅ System status
- ✅ Battery level
- ✅ GPS status
- ✅ Current operation mode
- ✅ Error messages
- ✅ Network connectivity

### Remote Interface

| Interface | Type | Status | Capabilities |
|-----------|------|--------|--------------|
| **Web UI** | Responsive web app | ✅ Implemented | Full system control |
| **Mobile App** | Progressive web app | ✅ Available | Touch-optimized |
| **API Access** | REST + WebSocket | ✅ Implemented | Developer integration |

---

## Optional and Future Hardware

### Remote Control (Optional)

| Component | Interface | Status | Capabilities |
|-----------|-----------|--------|--------------|
| **RC Receiver** | PPM/PWM | ❌ Not in Baseline | Not implemented |
| **Emergency Remote** | 2.4GHz | ⚠️ Optional | Safety backup |
| **Range** | 100-500m | ⚠️ Dependent | Line of sight |

### Advanced Sensors (Optional)

| Sensor Type | Purpose | Interface | Status |
|-------------|---------|-----------|--------|
| **LiDAR** | 360° obstacle detection | USB/Serial | ❌ Not in Baseline |
| **Radar** | Weather-resistant detection | GPIO/SPI | ⚠️ Compatible |
| **Ground sensors** | Soil moisture, pH | I2C/ADC | ⚠️ Expandable |
| **Weather station** | Local micro-climate | I2C/1-Wire | ⚠️ Compatible |

---

## Hardware Configuration Matrix

### Standard Configuration (Current Implementation)

| System | Components | Status | Capability Level |
|--------|------------|--------|------------------|
| **Processing** | Pi 5 8GB | ✅ Complete | Professional |
| **Navigation** | RTK GPS + IMU | ⚠️ GPS driver and EKF implemented; IMU driver stubbed; RTK accuracy unmeasured here | Not demonstrated |
| **Sensors** | ToF + Camera + Environmental | ✅ Complete | Comprehensive |
| **Power** | 30Ah LiFePO4 + 30W Solar | ✅ Complete | Full-day operation |
| **Communication** | HaLow + WiFi + I2C + UART | ⚠️ HaLow measured once; header I2C disabled; UART4 not enabled | Partially connected |
| **Safety** | Software interlocks | ❌ No independent hardwired E-stop, no fused obstacle model (H-08, H-26) | Not field-ready |

### Enhanced Configuration (With Optional Components)

| System | Additional Components | Status | Capability Gain |
|--------|----------------------|--------|-----------------|
| **AI Processing** | Google Coral TPU | ❌ Available | 10x vision performance |
| **Remote Control** | RC receiver system | ❌ Not in Baseline | Not implemented |
| **Advanced Sensors** | Additional ToF only | ⚠️ Compatible | Per spec |
| **Power Expansion** | Backup battery | ⚠️ Compatible | Extended operation |
| **Weather Station** | Local sensors | ⚠️ Compatible | Micro-climate data |

### Minimal Configuration (Cost-Optimized)

| System | Reduced Components | Impact | Cost Savings |
|--------|-------------------|--------|--------------|
| **Processing** | Pi 5 8GB | Moderate | 20% reduction |
| **GPS** | Standard GPS (no RTK) | Reduced accuracy | 40% GPS cost |
| **Sensors** | Single ToF sensor | Reduced safety | 30% sensor cost |
| **Power** | Smaller battery/panel | Shorter operation | 25% power cost |

---

## Pin Usage and Expansion Capabilities

### GPIO Pin Allocation

| Physical Pin | GPIO | Function | Device | Status |
|--------------|------|----------|--------|--------|
| 15 | 22 | Output | Left ToF Shutdown | ✅ Used (held LOW by the backend via lgpio on the live Pi) |
| 16 | 23 | Output | Right ToF Shutdown | ✅ Used (held HIGH on the live Pi) |
| 18 | 24 | Output | Blade Controller IN1 | ⚠️ Assigned; the IBT-4 driver is a stub and never drives it |
| 22 | 25 | Output | Blade Controller IN2 | ❌ **Conflict:** HaLow IRQ (`mm610x_spi_irq_gpio`) on the deployed Pi (H-02c) |
| 31 | 6 | Input | Left ToF Interrupt | ⚠️ Reserved; no code reads it |
| 32 | 12 | Input | Right ToF Interrupt | ❌ **Conflict:** also UART4 TXD4 for the BNO085 (rows "ToF Right Interrupt" and "BNO085 TX" in `spec/hardware.yaml`, H-03); no code reads it |
| 19 | 10 | SPI0 MOSI | HaLow radio | ❌ **Not free** (earlier listed "Available") |
| 23 | 11 | SPI0 SCLK | HaLow radio | ❌ **Not free** (earlier listed "Available") |
| 21 | 9 | SPI0 MISO | HaLow radio | ❌ Not free (missing from the earlier table) |
| 24 | 8 | SPI0 CE0 | HaLow radio chip select | ❌ Not free (missing from the earlier table) |
| 26 | 7 | HaLow BUSY (live `config.txt` comment) | HaLow HAT | ❌ **Not free** (earlier listed "Available") |
| 29 | 5 | HaLow RESET | HaLow radio; also the `starter` relay (`relays.starter` in `config/tractor.yaml`) | ❌ **Conflict** (H-02b); earlier listed "Available" |
| 5 | 3 | HaLow WAKE (live `config.txt` comment) | HaLow HAT; also header I2C1 SCL | ❌ Not free; I2C1 is disabled because of it |

### I2C Device Addresses

| Address | Device | Status | Notes |
|---------|--------|--------|-------|
| 0x29 | Left ToF Sensor | ✅ Used | Default address |
| 0x30 | Right ToF Sensor | ✅ Used | Reprogrammed address |
| 0x3c | OLED Display | ✅ Used | Standard SSD1306 |
| 0x40 | Power Monitor | ✅ Used | INA3221 |
| 0x76 | Environmental Sensor | ✅ Used | BME280 |
| 0x48 | **Available** | ⚠️ Free | ADC expansion |
| 0x50-0x57 | **Available** | ⚠️ Free | EEPROM range |

> **Header I2C1 is disabled on the deployed Pi** (`dtparam=i2c_arm=off`), so none of the addresses above can be
> reached there until a bus route is decided (H-17, H-25). Not in this registry: PCA9685 `0x41` (Toro,
> `pca9685.address` in `config/tractor.yaml`), the PCA9685 all-call address `0x70`, and the BNO085 `0x4A/0x4B`
> probed in `sensor_manager.py:188`. The "Available" rows above are registry gaps, not verified-free addresses.

### Expansion Capabilities

**Available Interfaces (corrected 2026-10-24; earlier text listed "5x GPIO pins" and "SPI interface" as available):**
- Header GPIO is **not** freely available: see the allocation table above and
  [`2026-10-24-integration-audit.md`](2026-10-24-integration-audit.md) Part B.1 for the unclaimed-pin candidates,
  none of which may be used before the HAT stack is inspected.
- I2C: no header bus on the deployed Pi.
- SPI0 is consumed by the HaLow radio (CS0 only, `dtoverlay=spi0-1cs`).
- USB: the live Pi shows the ELP camera and one CP2102N; the spare-port count is not verified.
- UART: UART4 needs `uart4-pi5` (not enabled).

**Potential Expansions:**
- Additional ToF sensors (front/rear)
- Soil moisture sensors
- Weather station integration
- Advanced lighting systems
- Security cameras
- Emergency beacons

---

## Installation and Configuration Guide

### Hardware Setup Sequence

1. **Core Assembly:**
   - Mount Raspberry Pi and RoboHAT
   - Install sensors per pin mapping
   - Connect power distribution

2. **Sensor Installation:**
   - Mount ToF sensors at front corners
   - Install IMU with proper orientation
   - Mount camera with clear field of view
   - Install environmental sensor

3. **Power System:**
   - Install battery in weatherproof enclosure
   - Mount solar panel with optimal exposure
   - Connect charge controller and monitoring

4. **Verification:**
   - Test all sensor connections
   - Verify power system operation
   - Confirm GPIO pin assignments
   - Validate communication interfaces

### Configuration Requirements

**Software Configuration:**
- Hardware configuration in `config/hardware.yaml`
- Pin mapping verification required
- Sensor calibration recommended
- Power thresholds adjustment

**Hardware Configuration:**
- GPS antenna positioning for sky view
- Camera angle and focus adjustment
- Solar panel optimal positioning
- Sensor mounting and protection

---

## Troubleshooting Matrix

### Common Hardware Issues

| Symptom | Possible Cause | Hardware Check | Solution |
|---------|----------------|----------------|----------|
| **No GPS fix** | Antenna obstruction | Sky view clearance | Relocate antenna |
| **ToF sensor errors** | Dirty sensor face | Visual inspection | Clean sensor lens |
| **Power issues** | Connection problems | Voltage measurements | Check all connections |
| **Camera not working** | Cable/connection | CSI cable check | Reseat camera cable |
| **I2C device missing** | Address conflict | I2C scan | Check address assignment |

### Diagnostic Tools

**Hardware Tests:**
- GPIO test scripts
- I2C device scanning
- Sensor individual testing
- Power system monitoring
- Communication verification

**Built-in Diagnostics:**
- System health dashboard
- Sensor status monitoring
- Power consumption tracking
- Error logging and analysis

---

## Maintenance Schedule

### Hardware Maintenance Matrix

| Component | Frequency | Maintenance Task | Tools Required |
|-----------|-----------|------------------|----------------|
| **Camera** | Weekly | Clean lens | Soft cloth |
| **ToF Sensors** | Weekly | Clean sensor face | Cotton swabs |
| **Solar Panel** | Monthly | Clean surface | Water, soft brush |
| **Connections** | Monthly | Inspect and tighten | Screwdrivers |
| **Battery** | Quarterly | Performance test | Multimeter |
| **Mechanical** | Seasonally | Lubricate, inspect | Lubricants |

### Replacement Schedule

| Component | Expected Life | Replacement Indicator |
|-----------|---------------|----------------------|
| **MicroSD Card** | 2-3 years | Performance degradation |
| **Battery** | 5-8 years | Capacity < 80% |
| **Camera** | 3-5 years | Image quality decline |
| **Sensors** | 5-10 years | Accuracy drift |
| **Motors** | 3-5 years | Performance issues |

---

## Specifications Summary

### System Capabilities Matrix

| Capability | Current Implementation | Optional Enhancement | Future Potential |
|------------|----------------------|---------------------|------------------|
| **Positioning Accuracy** | <10cm RTK | <2cm with base station | mm-level precision |
| **Obstacle Detection** | 0.03-2m ToF + vision | — | — |
| **Operation Time** | 8-12 hours | 16+ hours with backup | 24/7 with tracking |
| **AI Processing** | CPU OpenCV | TPU acceleration | Edge AI integration |
| **Connectivity** | WiFi primary | — | — |
| **Weather Resistance** | IP54 rating | IP67 upgrade | Extreme weather |

**Performance Benchmarks:**
- Sensor processing: 100Hz update rate
- Vision processing: 30fps real-time
- Navigation accuracy: Sub-centimeter positioning
- Power efficiency: 45-60W typical consumption
- Communication latency: <50ms local network
- Safety response time: <100ms emergency stop

*These benchmark figures are targets, not measurements. In particular "<100ms emergency stop" is the speed of
flipping a Python flag (`test_estop_latency_under_limit` in `tests/integration/test_safety_systems.py`), not a
hardware E-stop (H-08).*

---

## Software Integration Features (V2 API)

### Backend Services

| Service | Purpose | Status | Key Features |
|---------|---------|--------|--------------|
| **TelemetryHub** | Real-time data aggregation | ✅ Implemented | WebSocket streaming, persistence, export |
| **SensorManager** | Hardware sensor coordination | ✅ Implemented | Multi-sensor fusion, calibration |
| **RoboHATService** | Serial bridge to RP2040 | ✅ Implemented | Command translation, status monitoring |
| **MotorService** | Motor control coordination | ✅ Implemented | Safety interlocks, speed control |
| **MapsService** | Map provider management | ✅ Implemented | GeoJSON validation, fallback support |
| **SettingsService** | Configuration management | ✅ Implemented | Profile versioning, validation |

### Frontend Components

| Component | Purpose | Status | Capabilities |
|-----------|---------|--------|--------------|
| **SystemStore** | Telemetry state management | ✅ Implemented | Real-time updates, error handling |
| **ControlStore** | Control command management | ✅ Implemented | Lockout handling, command queuing |
| **MapStore** | Map configuration state | ✅ Implemented | Boundary editing, marker management |
| **DashboardView** | System overview UI | ✅ Implemented | Telemetry display, status cards |
| **ControlView** | Motor control interface | ✅ Implemented | Command buttons, safety indicators |
| **MapView** | Interactive map UI | ✅ Implemented | Boundary editor, zone management |
| **SettingsView** | Configuration UI | ✅ Implemented | Profile management, validation |
| **DocsHubView** | Documentation browser | ✅ Implemented | Offline bundle support |

### API Endpoints (V2)

| Endpoint Category | Count | Status | Features |
|------------------|-------|--------|----------|
| **Telemetry** | 3 | ✅ Complete | Stream, export, subscribe |
| **Control** | 4 | ✅ Complete | Drive, blade, stop, status |
| **Maps** | 4 | ✅ Complete | Configuration CRUD, fallback |
| **Settings** | 2 | ✅ Complete | Profile GET/PUT, validation |
| **Documentation** | 2 | ✅ Complete | Bundle generation, artifacts |
| **Planning** | 3 | ✅ Complete | Job management, scheduling |
| **AI** | 2 | ✅ Complete | Dataset management, export |

### WebSocket Topics

| Topic | Purpose | Update Rate | Status |
|-------|---------|-------------|--------|
| **telemetry** | Sensor data streaming | 1-10Hz | ✅ Active |
| **control** | Command echoes, lockouts | On event | ✅ Active |
| **maps** | Map updates | On change | ✅ Active |
| **ai** | Processing results | On completion | ✅ Active |

### Data Models

| Model Category | Models | Status | Validation |
|----------------|--------|--------|------------|
| **Telemetry** | 8 models | ✅ Complete | Pydantic v2 |
| **Control** | 4 models | ✅ Complete | Pydantic v2 |
| **Maps** | 5 models | ✅ Complete | GeoJSON + Shapely |
| **Settings** | 9 models | ✅ Complete | Branding checksums |
| **Planning** | 3 models | ✅ Complete | Pydantic v2 |

### Constitutional Compliance

| Requirement | Implementation | Status | Validation Method |
|-------------|----------------|--------|-------------------|
| **Latency Targets** | ≤250ms | ✅ Enforced | Performance tests |
| **Audit Logging** | All state changes logged | ✅ Implemented | Audit trail verification |
| **Remediation Links** | All errors include docs links | ✅ Implemented | Error response inspection |
| **Branding Validation** | Checksum verification | ✅ Implemented | Settings validation |
| **SIM Mode Coverage** | All hardware operations | ✅ Implemented | Test suite verification |
| **Offline Documentation** | Bundle generation | ✅ Implemented | Archive creation script |

---

**Document Version:** 2.0  
**Last Updated:** October 2025  
**Next Review:** January 2026

This hardware feature matrix provides comprehensive information for planning, deploying, and maintaining LawnBerryPi systems across different hardware configurations and use cases. For V2 API details and operational procedures, see [OPERATIONS.md](OPERATIONS.md).
