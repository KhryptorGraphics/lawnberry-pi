# LawnBerry-pi — physical integration and wiring reconciliation
Date: 2026-10-24. Status: **engineering design / NOT an as-built wiring diagram**. Target: Toro TimeCutter MAX MyRIDE 50-inch model 77502 (verify actual serial plate).

## Evidence and precedence
- Tractor-specific authoritative *software intentions*: `config/tractor.yaml`, `docs/tractor-platform.md`, `docs/2026-10-30-primary-backup-software-control.md`.
- Older differential-drive documentation: `docs/hardware-integration.md`, `docs/servo-hat-pins.md` sections 1–6, parts of `spec/hardware.yaml`. **Do not apply their RoboHAT/Cytron/IBT-4 motor wiring to the Toro.**
- Hardware owned/installed versus planned must be verified with photos, part labels and measurements. No connection below is represented as physically tested.

## Toro control path (proposed; existing config, not commissioned)
```
Pi 5 I2C1 GPIO2 SDA (header pin 3) --+--> PCA9685 SDA, configured 0x41
Pi 5 I2C1 GPIO3 SCL (header pin 5) --+--> PCA9685 SCL
Pi ground (header pin 6 or another GND) ---> PCA9685 logic ground
PCA9685 CH0 PWM ---> left hydrostatic-lever actuator signal
PCA9685 CH1 PWM ---> throttle actuator signal
PCA9685 CH2 PWM ---> right hydrostatic-lever actuator signal
Pi BCM GPIO5 (physical pin 29) ---> starter-relay CONTROL input
Pi BCM GPIO26 (physical pin 37) ---> PTO-relay CONTROL input
```
**These are control-signal assignments, not safe high-current wiring instructions.** Use suitable electrically isolated/rated relay drivers and protective circuitry; Pi GPIO must never directly power a starter solenoid or PTO clutch. Verify active-high/low polarity and safe state at boot, power loss and GPIO floating. Starter operation remains human-supervised because OEM engine-start requires both levers swung outboard to PARK; the proposed fore/aft servo linkage cannot do that. Keep OEM seat/presence and blade interlocks functional pending formal engineering review. Software attestation is not a physical substitute for a safety switch.

### Pi 5 header reference, only for signals currently evidenced
| Function | BCM GPIO | Physical header pin | Confidence |
|---|---:|---:|---|
| I2C1 SDA | 2 | 3 | Pi standard pinout; configured bus must be checked |
| I2C1 SCL | 3 | 5 | Pi standard pinout; configured bus must be checked |
| Starter relay input | 5 | 29 | Configured in `config/tractor.yaml`; hardware unverified |
| Blade PTO relay input | 26 | 37 | Configured in `config/tractor.yaml`; hardware unverified |
| I2C ground reference | — | 6 | Header GND; check common-reference/isolation design |
| 3.3 V rail | — | 17 | **NOT 5 V**; erroneous ultrasonic guide says 5 V |
| 5 V rail | — | 2 or 4 | Logic rail only within Pi supply design, not servo power |
| GPIO12 / ALT UART4 TX candidate | 12 | 32 | **Conflict** with ToF interrupt in `spec/hardware.yaml` |
| GPIO13 / ALT UART4 RX candidate | 13 | 33 | Verify ALT mapping, overlay and BNO085 interface |
| GPIO4 / ultrasonic left trigger in old guide | 4 | 7 | Old three-sensor design, **not approved** |
| GPIO17 / ultrasonic left echo in old guide | 17 | 11 | Old three-sensor design, **not approved** |
| GPIO27 / ultrasonic center trigger in old guide | 27 | 13 | Old three-sensor design, **not approved** |
| GPIO10 / ultrasonic center echo in old guide | 10 | 19 | **Potential HaLow SPI collision; do not connect** |
| GPIO11 / ultrasonic right trigger in old guide | 11 | 23 | **Potential HaLow SPI collision; do not connect** |
| GPIO9 / ultrasonic right echo in old guide | 9 | 21 | **Potential HaLow SPI collision; do not connect** |

The last six assignments are transcribed from `docs/ultrasonic-wiring-guide.md` for conflict analysis **only**. The owner wants four units, not three. Before allocating new GPIO, identify the installed HaLow interface (SPI/UART/USB), Hailo PCIe/stacked HAT footprint, I2C/UART overlays and actual GPIO reservations; run `scripts/check_hardware_pin_conflicts.py` and confirm with `gpioinfo`/scope. Prefer a dedicated microcontroller for time-sensitive ultrasonic triggering/echo capture if pins or timing are constrained. HC-SR04 ECHO can be 5 V and **must not** connect directly to Pi GPIO; choose a level shifter/divider with adequate tolerance margin. `docs/ultrasonic-wiring-guide.md` has a contradictory divider diagram.

## I2C, power and actuator design
- `config/tractor.yaml` requests PCA9685 bus 1, address **0x41**, 50 Hz. Address 0x40 is reserved in older spec for INA3221; verify PCA9685 A0 strap and real `i2cdetect` results before energizing actuators.
- `config/tractor.yaml` contains **illustrative** 1000/1500/2000 µs lever endpoints and 1100/1900 µs throttle endpoints. They are NOT calibrated values. Validate actual linkage geometry and actuator supply voltage.
- `docs/tractor-platform.md` selects 12 V RDS51150SG servos and explicitly says older 24 V design is superseded; a superseded (deleted) acceptance checklist assumed 24 V in a latency section. **Resolve the exact purchased servo label and datasheet before connecting power.** Do not connect a 24 V supply to a 12 V servo or to Pi/PCA9685 logic rails. The servo supply and control electronics need separate appropriately rated, fused distribution and a deliberate ground/isolation strategy.
- PCA9685 retains last PWM on host/I2C failure. The PCA9685 **OE input is active-low**: driving OE **high** disables PWM outputs; driving it low enables them. The existing acceptance checklist incorrectly suggests driving OE low to disable outputs. Disabling PWM does **not** itself guarantee lever return to neutral; verify spring return or design independent mechanically safe disengagement. Test physical lever movement with engine OFF, machine immobilized, blades disabled.
- Hardwired E-stop, independent watchdog and physically verified blade power removal must remain effective when Pi, Thor, HaLow, I2C or servo software fails. E-stop safety must not depend on Wi-Fi or an API call. Do not bypass OEM operator-presence protections.

## Sensors and compute: connections to verify, not guess
| Device | Proposed role | Physical connection evidence still needed |
|---|---|---|
| Quectel LC29H-DA rover | RTK position onboard | Exact carrier board, voltage, USB/UART pinout, antenna bias, corrections format, fix/status handling |
| Quectel LC29H-BS base | Stationary RTK corrections | House-side host/power/antenna, base survey, RTCM transport over HaLow/network |
| BNO085 | Heading/IMU | I2C vs UART selection, address/overlay, interrupt/reset and bus collision |
| Two VL53L0X | Near-range obstacles | I2C addresses, XSHUT pins, voltage, mounting and sunlight limits |
| Four HC-SR04 | Forward/deck-width near obstacles | New pin plan or microcontroller, level shifting, time-division triggering, angular coverage |
| Pi Camera Module 2 | Fixed forward scene | Pi 5 camera cable/adapter and CSI port; verify libcamera |
| ELP-USB960P2CAM-V90 | Elevated stereo / possible VR | USB bus/power/bandwidth, baseline, mount, optional pan-tilt mechanics |
| Optional 2D 360-degree LiDAR | Planar obstacle scan | Exact model, USB/UART power and driver, scan height, blind spots |
| Hailo 8L | Onboard inference | Actual HAT/M.2/PCIe adapter and GPIO header clearance |
| Heltec HT-HC01 V2 HaLow | Pi↔house/Thor communications | Actual transport pins/adapter, supply and recovery after power faults |

## Mandatory physical sign-off
1. Confirm Toro model/serial, servo voltage and part numbers, and photograph actual Pi/HAT stack.
2. Produce a measured **as-built** connector-to-connector netlist with pin numbers, wire gauge, fuses, ground points, level shifting and cable strain relief.
3. Resolve every shared pin and I2C address; no HaLow collision.
4. Bench-test relays with loads disconnected; confirm OFF on reboot, loss of Pi power, loss of signal and E-stop.
5. Verify physical lever neutral/parking-brake behavior and independent stop mechanism before connecting the engine or PTO.
6. Verify sensor fault and wireless-loss response in simulation before supervised field testing.


## Corrections recovered from the October 24 design conversation (requirements superseding generic examples)
**Ultrasonic part correction:** The owner's four purchased/intended units are **JSN-SR04T waterproof ultrasonic modules**, NOT the three HC-SR04 boards used in the old guide. The earlier HC-SR04 GPIO map is only a warning/example; it is not an approved JSN-SR04T wiring design. Verify each JSN module revision, operating voltage, trigger/echo versus serial mode, timing, minimum detectable range and connector pinout from its actual label/datasheet. Approximately 10-inch close-range blind zones have been discussed but must be measured; do not rely on ultrasonics for close-contact safety.

**Four-sensor front mounting geometry:** two inner sensors point forward, with two outer sensors angled outward to cover the full **50-inch deck** edges and approach to trees. Mounts should be mechanically adjustable for yaw/pitch, weather-sealed, vibration-resistant and clear of deck travel, chute and steering envelope. Measure cone overlap and blind zones, not merely the nominal mounting angle. Slow/stop thresholds must incorporate vehicle speed, braking distance, detection uncertainty and obstacle class; distinguish a detected cone range from a 2D/3D map.

**Actual steering actuators:** Owner reports **two matching 150 kg-cm-class servos**, one for each Toro lap bar; described supply **10–12.6 V** with possible **8 A or higher stall current**. These are owner-reported specifications, not bench-verified ratings. Identify exact make/revision, stall current and mechanical mount; size independent fused supply and wiring for worst-case demand. Throttle actuator remains a separate third channel and needs its own verified model. Owner also reports a power relay visible in an earlier photo; identity, contact ratings, coil drive and purpose are **not established** by that report.

**PTO control correction:** The owner described a **momentary toggle/button PTO control** on the mower. The existing `config/tractor.yaml` describes a **latching** `blade_pto` relay. Do not assume the mower's actual switch is a latching electrical input or directly wire a relay across it. Determine the exact model's OEM PTO circuit behavior from schematic and meter tests, including whether a momentary actuation toggles a latched controller state, how OFF is guaranteed on faults, and whether the existing software abstraction is correct. No PTO energization before this is resolved.

**House/base station topology previously discussed:** A possible **Pi Zero-based RTK base with Orange Pi network intermediary**, connecting the Quectel LC29H-BS to HaLow and NVIDIA Thor, was discussed. It was **not included in the earlier house-side wiring map** and must be treated as a candidate design until the actual boards and connections are confirmed. Create an explicit base-station netlist: antenna → LC29H-BS → Pi Zero (serial/USB and power TBD) → Orange Pi (network transport TBD) → HaLow → Thor/Pi rover. Avoid assuming the base is directly attached to Thor.

**Header-stack / cabling requirement:** Produce a photographic or CAD-like, labeled visual of the actual Pi 5/Hailo/HaLow stack, with a **40-pin breakout or GPIO riser** if HaLow HAT physically blocks pins. Show physical pin numbers, connector orientation/keying, voltage, ground, cable destinations, USB-C power, USB devices, CSI camera ribbon adapter, PCIe/Hailo link, and how all wires route to external sensor boards. Do not confuse a visual rendering with measured as-built connectivity.

**Mechanical visualization requirement:** Provide labeled component photos, 3D-style exploded placement views, and a front view of the Toro TimeCutter MAX MyRIDE with **four adjustable ultrasonic mounts overlaid**. Photos or model-specific renderings require real images/measurements; do not fabricate physical mounting holes or pretend a schematic is a photograph.

**Localization and spatial perception:** RTK + BNO085 heading + ToF + ultrasonic cone detections + camera depth/stereo + optional 2D LiDAR should be fused with explicit timestamp/frame transforms and confidence; 2D LiDAR or ultrasonics alone do not create a reliable 3D obstacle model. Investigate ROS 2/Nav2 compatibility rather than asserting it is installed.
