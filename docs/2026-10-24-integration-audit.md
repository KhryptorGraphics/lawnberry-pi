# LawnBerry-pi integration audit — 2026-10-24

**Status:** audit and engineering plan. It is **not** an as-built record and it authorises no powered actuation.
Everything below is traceable to a file in the repository, to a read-only inspection of the deployed Pi, or to the
owner's requirement documents. Anything that is none of those is marked `TBD` (needs a measurement) or `[DS]`
(general datasheet knowledge, not evidenced in this repository; bench-confirm before relying on it).

## Contents

- [0. Scope, baseline and conventions](#0-scope-baseline-and-conventions)
- [Part A. Subsystem status matrix](#part-a-subsystem-status-matrix) (A.1 to A.15)
- [Part B. Phase 3 hardware deliverables](#part-b-phase-3-hardware-deliverables) (B.1 to B.7)
- [Part C. Phase 4 distributed AI and navigation architecture](#part-c-phase-4-distributed-ai-and-navigation-architecture)
- [Part D. Phase 5 safety review](#part-d-phase-5-safety-review)
- [Part E. Prioritised development plan (P0 to P6)](#part-e-prioritised-development-plan-p0-to-p6)
- [Part F. Remaining hardware conflicts and unresolved safety requirements](#part-f-remaining-hardware-conflicts-and-unresolved-safety-requirements)
- [Appendix. Gap register index (H-01 to H-28)](#appendix-gap-register-index-h-01-to-h-28)
- [Implemented in this PR](#implemented-in-this-pr)

## 0. Scope, baseline and conventions

### 0.1 Baseline

| Item | Value |
|---|---|
| Repository baseline | `origin/main` at `765c737` (branch `audit/docs` adds documentation only) |
| Deployed Pi | host `pi5`, checkout `/apps/lawnberry-pi` at `23a6e4d`, kernel `6.12.21-morse+`; **read-only inspection on 2026-10-24**, nothing written, no actuator driven, no hardware config changed |
| Requirement documents | draft PR #18, branch `docs/2026-10-24-hardware-architecture-audit`: `docs/OMP-THOR-HANDOFF-2026-10-24.md`, `docs/2026-10-24-physical-integration-plan.md`, `docs/2026-10-24-development-roadmap.md`, `docs/2026-10-24-compatibility-gap-register.md`. They are **not on this branch**; they are cited by file name and treated as proposed requirements, not as evidence of implementation |
| Gap IDs | **H-01 … H-28** are the register's IDs; **H-02b** and **H-02c** are sub-findings of H-02 introduced here (C-02, C-03). Findings the register does not contain carry **N-01 … N-31** (section A.15); they are proposed additions to the register. Conflicts are **C-xx** (B.1), measurements **M-xx** (B.7), owner decisions **D-n** and safety decisions **SD-n** (B.1.4, F.3), requirements **R-Pnn** (B.3), work packages **WP-Pn-nn** and sign-off gates **G0 … G6** (Part E). The gap register index is in the [Appendix](#appendix-gap-register-index-h-01-to-h-28) |
| Test baseline | `SIM_MODE=1 pytest tests --ignore=tests/soak -p no:cacheprovider`: **530 passed, 69 skipped, 0 failed** (collection: 600 items including 1 soak test). Parallel implementation branches (`audit/checker`, `audit/safety`, `audit/thorlink`) are **not** reflected in the statuses below; the integrator records what landed under [Implemented in this PR](#implemented-in-this-pr) |

### 0.2 Status vocabulary

| Status | Meaning in this document |
|---|---|
| **IMPLEMENTED** | Code exists, is wired to its consumer and has at least one test that exercises it (SIM or hardware). It does **not** mean verified on the machine unless the row says "live" or cites a measurement |
| **PARTIALLY IMPLEMENTED** | Exists, but part of the behaviour, wiring, configuration or test coverage is missing or contradicted |
| **SIMULATED** | Behaviour exists only under `SIM_MODE` or against fakes; no hardware effect |
| **STUBBED** | Placeholder that returns constants or sets flags with no hardware path |
| **MISSING** | No code, configuration or document for the capability |
| **UNVERIFIED** | Claimed in documents or configuration; no evidence either way |

### 0.3 Evidence tags

- **LIVE** = read-only inspection of the deployed Pi on 2026-10-24 (`/boot/firmware/config.txt`, decoded
  `halow-dw.dtbo`, `/sys/kernel/debug/pinctrl`, `gpioinfo`, `lspci`, `lsusb`, `ls /dev`, `ip`, `systemctl`).
  The `pinctrl` CLI was not installed (pin states came from debugfs and `gpioinfo`) and `i2cdetect` was
  deliberately not run. Absence of a device at this site is **not** proof of a design error: most sensors are
  detached there.
- **[DS]** = datasheet or general engineering knowledge, not evidenced in this repository.
- **[INF]** = inference from reading code; not executed.
- **TBD** = unknown; a measurement in B.7 closes it.
- Test counts in parentheses are the passing tests the audit counted for that file. Every passing test named here
  runs on fakes or `SIM_MODE`; none demonstrates a physical outcome.
- **Citations.** `file:line` references are against baseline `765c737`. Documents this PR itself edits (`README.md`,
  `hardware.md`, `docs/hardware-feature-matrix.md`, `docs/hardware-overview.md`, `docs/servo-hat-pins.md`,
  `docs/tractor-platform.md`, `docs/tractor-acceptance-criteria.md`, `docs/field-validation-protocol.md`,
  `docs/RELEASE_NOTES.md`, `docs/ultrasonic-wiring-guide.md`) are cited by section because their line numbers moved.
  Files other branches may change (`spec/hardware.yaml`, `scripts/check_hardware_pin_conflicts.py`,
  `docs/hardware-integration.md`, `backend/src/**`) keep their baseline line numbers.

### 0.4 Release gates carried over from the gap register

1. **No live motion** while H-01 to H-08 are unresolved.
2. **No autonomous mowing** until sensor and RTK health, a physical fail-safe, operator protections and field
   acceptance are proven.
3. **No remote VR driving** until authentication, a dead-man input, measured latency and a local safety override
   are proven.

State of the gates at this baseline: **all three are closed** (see Part F).

## Part A. Subsystem status matrix

### A.1 Backend services and drivers

| Subsystem | Status | Evidence (file:line) | Tests | Gap IDs |
|---|---|---|---|---|
| Application lifespan wiring | PARTIALLY IMPLEMENTED | `backend/src/main.py:67-201`. Tractor service, watchdog and tilt monitor start only if `tractor.enabled` (`:144-170`, shipped `false`); an exception anywhere in that block is logged and swallowed (`:169-170`) while `/api/v2/tractor/*` stays routed (N-05). `ThorLinkService` is never instantiated (no import in `main.py`). Teardown (`:184-201`) stops the monitors and the RoboHAT service but never parks the tractor (N-09) | `tests/unit/test_tractor_safety_monitor.py` | H-08, H-12, H-13 |
| Sensor manager | PARTIALLY IMPLEMENTED | `backend/src/services/sensor_manager.py`: IMU interface `:166-225` (ONLINE only on an I2C ACK at 0x4A/0x4B, `:186-190`), ToF `:228-`, stereo `:795-850`, ultrasonic gating `:858-886` | `tests/unit/test_sensor_presence.py`, `test_sensor_manager_ultrasonic.py` | H-02, H-03 |
| GPS driver (ZED-F9P, LC29H-DA, Neo-8M) | IMPLEMENTED (driver); the fitted receiver is UNVERIFIED | `backend/src/drivers/sensors/gps_driver.py`; modes `F9P_USB`, `F9P_UART`, `NEO8M_UART`, `LC29H_UART` (`backend/src/models/sensor_data.py:37-43`). NMEA only: no LC29H `PAIR`/`PQTM` configuration exists anywhere in `backend/`, `scripts/` or `config/` | `tests/unit/test_gps_driver.py` (5), `test_gps_config_fallback.py` | H-09 |
| BNO085 IMU driver | **STUBBED** | `backend/src/drivers/sensors/bno085_driver.py:1-9` ("simulation stub"), `:69-78` returns zeros and constants on real hardware; presence is decided by an I2C ACK, not by the driver (`sensor_manager.py:186-190`) | none exercises hardware | H-03, N-06 |
| VL53L0X ToF driver (x2) | PARTIALLY IMPLEMENTED | `backend/src/drivers/sensors/vl53l0x_driver.py:262-300` (XSHUT pair addressing), boot hook `backend/src/main.py:78-107`. The interrupt pins (GPIO6, GPIO12) are read by no code. Header I2C is absent on the deployed Pi | `tests/unit/test_sensor_presence.py`; `tests/contract/test_tof_sensors.py` is a skipped placeholder | H-03, H-17 |
| Ultrasonic driver (3x HC-SR04) | PARTIALLY IMPLEMENTED; disabled by configuration | `backend/src/drivers/sensors/ultrasonic_driver.py:80-84` (GPIO4/17, 27/10, 11/9); `config/limits.yaml:17`; fail-closed gating `sensor_manager.py:871-886` | `tests/unit/test_ultrasonic_driver.py` (10), `test_sensor_manager_ultrasonic.py` (SIM, asserts three sensors) | H-02, H-10, H-19 |
| **JSN-SR04T x4** | **MISSING** | No driver, configuration, model or test. `UltrasonicData` is three HC-SR04 values (`backend/src/models/mower_data_frame.py:104-110`) | none | H-10, H-19 |
| BME280, INA3221, Victron | PARTIALLY IMPLEMENTED | `backend/src/drivers/sensors/bme280_driver.py`, `ina3221_driver.py` (CH1 solar, CH3 battery: `:3-6,101`); Victron via the `victron-ble` CLI (`docs/victron-ble-integration.md`, `config/hardware.yaml:34-42`). Header I2C absent on the deployed Pi | `tests/contract/test_ina3221.py`, `tests/unit/test_power_sensor_interface.py` | H-17, N-12 |
| Stereo camera driver | IMPLEMENTED (capture only) | `backend/src/drivers/sensors/stereo_camera_driver.py:40-79` (2560x960 split into 1280x960 per eye; SIM synthetic frames). LIVE: a UVC camera `32e4:9750` enumerates | `tests/unit/test_stereo_camera_driver.py` (4) | H-16, H-26 |
| Camera stream service | IMPLEMENTED (live) | `backend/src/services/camera_stream_service.py`, `systemd/lawnberry-camera.service`, `docs/pi5-csi-camera-libcamera-status.md`. LIVE: `lawnberry-camera` active, imx219 on the RP1 CFE | `tests/unit/test_camera_stream_service.py`, `test_camera_ai.py` | H-17 |
| Hailo driver and inference | IMPLEMENTED (SIM and hardware path) | `backend/src/drivers/ai/hailo_driver.py:55-80` (`detector` = YOLO26 and `scdepthv3` = monocular depth models), `yolo26.py`, `backend/src/services/camera_stream_service.py:899-1034`, `backend/src/services/ai_inference_service.py`. `docs/autonomy-toolchain-matrix.md:87`: `hailortcli benchmark` on the Pi, HailoRT 4.24.0, yolo26s 41.7 FPS. LIVE: `/dev/hailo0` | `tests/unit/test_hailo_driver_lifecycle.py` (4), `test_ai_inference_service.py`, `test_camera_ai.py`, `test_ai_navigation_loop.py` (3); `test_yolo26_decode.py` (1 skipped, no cv2) | H-17 |
| Tractor control service | PARTIALLY IMPLEMENTED | `backend/src/services/tractor_service.py`: levers `:176-203`, blade `:218-231`, engine `:233-256` (no engine-run sensor, `:248`), e-stop `:258-288`. State fields are commanded values, not feedback | `tests/unit/test_tractor.py` (16), `test_tractor_navigation.py` | H-06, H-08, H-21, N-26 |
| Tractor actuators | PARTIALLY IMPLEMENTED | `backend/src/drivers/actuators/tractor_actuators.py`: `ServoActuator` `:42-77` (updates `_us`/`_value` before the write, `:65-69`), `RelayActuator` `:80-125` (gpiozero `OutputDevice(initial_value=False)` `:102`, silent fallback to simulation `:104-108`, unguarded `set` `:111-115`, starter `pulse` with `finally` `:117-124`) | `tests/unit/test_tractor.py` | H-02b, H-21, N-27 |
| PCA9685 driver | PARTIALLY IMPLEMENTED | `backend/src/drivers/actuators/pca9685_driver.py:47-171`: raw smbus2. An absent chip degrades to a silent no-op (`:117-126,155-156`, N-07); a write fault propagates once the chip was seen (`:170-171`). `MODE2 = 0x04` (`:57,111`). `MODE1` is written `0x30` then `0x20` (`:108-112`), which clears the ALLCALL bit [DS]. No `OE` handling, no read-back, no timeout on the thread write (N-28) | `tests/unit/test_pca9685_driver.py` (8) | H-04, H-08 |
| Tractor safety monitor and watchdog | PARTIALLY IMPLEMENTED | `backend/src/safety/tractor_safety_monitor.py:82-115` (20 Hz; feeds the heartbeat unconditionally at `:87`; the tilt edge calls `engage_blade(False)` once, `:103-112`; IMU loss returns silently, `:37-47,98-100`); `backend/src/safety/watchdog.py:54-69` (one-shot: `break` after the first timeout) | `tests/unit/test_tractor_safety_monitor.py` (9), `tests/contract/test_watchdog.py` | H-08, N-04, N-06 |
| RoboHAT service | PARTIALLY IMPLEMENTED (differential-drive platform only) | `backend/src/services/robohat_service.py`: `_maintain_usb_control` re-sends the last PWM every about 0.9 s (`:246-269`, N-14); `emergency_stop` `:441-459`; `shutdown` `:501-` | `tests/unit/test_robohat_service_usb_control.py` | H-08, H-11 |
| RoboHAT JSON driver | STUBBED (not used by the service; protocol mismatch) | `backend/src/drivers/motor/robohat_rp2040.py:84-134` sends JSON `MOTOR_COMMAND`/`EMERGENCY_STOP`; the firmware parser accepts a text protocol (`robohat-rp2040-code/code.py:186-225`) | `tests/unit/test_robohat_rp2040_driver.py` (fake) | H-11 |
| Blade service and IBT-4 driver | **STUBBED** | `backend/src/drivers/blade/ibt4_gpio.py:25-50` (`setup`/`set_active` only set flags; no GPIO library imported). `BladeService.emergency_stop` (`backend/src/services/blade_service.py:46-55`) and `IBT4BladeDriver.set_estop` (`ibt4_gpio.py:103-106`) have no callers (N-03) | `tests/unit/test_ibt4_blade_driver.py` (flags only) | H-02c, H-08 |
| Motor service | PARTIALLY IMPLEMENTED | `backend/src/services/motor_service.py:409-419` command-timeout handler; `control_drive_v2` does not use it (`backend/src/api/rest.py:310-417` calls `robohat.send_motor_command` directly) | none for the REST path | H-08, N-14 |
| Navigation service | PARTIALLY IMPLEMENTED | `backend/src/services/navigation_service.py`: `ObstacleDetector` `:33-82`, `execute_mission` `:175-238` (auto-crank `:188-195`), `go_to_waypoint` `:240-372`, `apply_ai_prediction` `:818-860` | `tests/unit/test_tractor_navigation.py`, `test_return_to_home.py`, `tests/test_nav_*.py` | H-06, H-23, H-26, N-08, N-11 |
| Mission API and service | PARTIALLY IMPLEMENTED | `backend/src/api/mission.py:9-90` (create, start, pause, resume, abort, status, list; no auth dependency), `backend/src/services/mission_service.py` | `tests/test_mission_api.py` | H-14, N-16 |
| Teleop capture and Thor uploader | IMPLEMENTED (software); the Pi capture session was blocked | `backend/src/services/capture_service.py`, `backend/src/services/thor_uploader.py`, `scripts/teleop.py`; `docs/autonomy-toolchain-matrix.md:76` | `tests/unit/test_capture_service.py`, `test_thor_uploader_config.py` | H-16 |
| Sensor-manager systemd unit | **STUBBED** | `systemd/lawnberry-sensors.service:16-17`: `ExecStart` is `echo "...placeholder..." && sleep infinity`. Sensors actually run inside the backend process | none | N-20 |

### A.2 Firmware

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| RoboHAT RP2040 firmware (CircuitPython) | IMPLEMENTED for the differential-drive platform; UNVERIFIED on hardware (no RP2040 enumerates at this site) | `robohat-rp2040-code/code.py`: text protocol `pwm,<steer>,<throttle>`, `rc=enable`/`rc=disable`, `blade=...`; hardware watchdog 8 s with RESET (`:98-101`); USB silence of 2 s falls back to RC mode (`:105-107,305-309`); RC signal loss for 1 s centres the outputs (`:111,315-322`); RC channel-5 stop (`:326-336`) is evaluated **only inside `if rc_enabled`** (`:312`), i.e. not while the Pi holds USB control | host side only (`tests/unit/test_robohat_*`); no firmware tests | H-08, H-11 |
| MDDRC10 input state during a firmware WDT reset | UNVERIFIED | TBD: scope the steer/throttle pins during an induced reset (M-23) | none | H-08 |
| Toro platform firmware | MISSING | `code.py` drives GP10/GP11 servo PWM, GP12 blade PWM, GP13 mode, GP8/9 encoder, GP2-7 RC inputs (`:4-7,58-59,72-77,90-91,95`). There is no PCA9685, relay or tractor path in firmware | none | H-11 |
| Safety / sensor coprocessor firmware | MISSING (design option, B.1.4 option B) | no code | none | H-08, H-19 |

### A.3 Hardware configuration (repository and deployed Pi)

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| `spec/hardware.yaml` | PARTIALLY IMPLEMENTED; internally contradictory; no platform tag | Mixed registry: mower sections plus a `tractor:` block (`:94-133`). Pin 32 is booked twice (ToF-R interrupt `:142`, BNO085 TX `:144`); BNO085 RX is `rxd: 4` on pin 33 (`:143`). INA3221 `channel_1` battery / `channel_3` solar (`:61-70`) is the reverse of the driver (N-12). GPS ZED-F9P preferred (`:9-12`). Hailo recorded as "Hailo-8 AI Hat" (`:37`). UART1 RoboHAT 38400 (`:155`) vs USB CDC 115200 (`docs/servo-hat-pins.md` section 2). `i2c_addresses` (`:146-151`) omits 0x41, 0x4A/0x4B and 0x70 | none loads it | H-03, H-09, H-11 |
| `config/hardware.yaml` | PARTIALLY IMPLEMENTED (mower) | LC29H-DA on `/dev/ttyAMA0` (`:8-11`) contradicts the spec; INA3221 shunt comments (`:29-33`); a Victron MAC (`:37`, device identification, no key); ToF bus 1 (`:21`) | `tests/unit/test_config_loader.py` | H-09, H-17 |
| `config/tractor.yaml` | PARTIALLY IMPLEMENTED; shipped disabled | `enabled: false` (`:11`; LIVE identical); PCA9685 0x41, bus 1 (`:17-18`); relays BCM5 and BCM26 (`:31,36`); microsecond endpoints are illustrative (`:22-27`); "no clutch/brake pedal" (`:38-39`) contradicts the parking brake at PARK (`docs/tractor-platform.md`, "Physical integration decisions" item 2); PTO "latching" (`:32`) vs the owner-reported momentary switch | none | H-12, H-21 |
| `config/limits.yaml` | PARTIALLY IMPLEMENTED | `ultrasonic_enabled: false` (`:17`; LIVE identical), `estop_latency_ms: 100` (`:4`), `watchdog_timeout_ms: 1000` (`:10`); the comment at `:16` "Trigger e-stop if any sensor reads below this" is false (N-19) | `tests/unit/test_config_loader.py`; `scripts/validate_safety_config.py` | H-02, H-08 |
| Deployed boot configuration (unversioned) | UNVERIFIED in the repository | LIVE `/boot/firmware/config.txt`: `dtparam=i2c_arm=off` (`:23`), `dtoverlay=halow-dw`, `dtoverlay=spi0-1cs`, `enable_uart=1`, `dtparam=pciex1_gen=3`, a `dwc2` overlay, `device_tree=bcm2712-rpi-5-b-morse.dtb`, `kernel=kernel_2712-morse.img`; HaLow pin comments at `:20-31`; **no `uart4-pi5`**. Nothing in the repository versions this file | none | H-02, H-17, H-25 |
| Deployed application configuration | IMPLEMENTED (matches repo) | LIVE `/apps/lawnberry-pi/config/tractor.yaml` and `limits.yaml` identical to the repository; no `config/hardware.local.yaml`, no `.env` | none | H-12 |

### A.4 GPIO and buses

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| Pin and address conflict checker | PARTIALLY IMPLEMENTED | `scripts/check_hardware_pin_conflicts.py:53-87` compares only `pins.gpio_used[].gpio`, `config/tractor.yaml relays.*.gpio`, `pins.i2c_addresses` and the tractor PCA9685 address. Rows without a `gpio` key are skipped (`:57-59`); the self-test asserts that rxd/txd rows must not collide (`:129-141`). It never reads HaLow, overlays, `config/hardware.yaml`, `limits.yaml` or physical pin numbers. What-if probes by the audit: moving the starter to GPIO17 or GPIO3 reports nothing | inline `--self-test` only; no file in `tests/` references it | H-01, H-02, H-03 |
| hardware-guard workflow | PARTIALLY IMPLEMENTED; enforces a documented error | `.github/workflows/hardware-guard.yml:14-59`; `:54-59` requires CH1 = Battery and CH3 = Solar, the reverse of the driver | none | N-12, N-22 |
| Header I2C1 on the deployed Pi | **MISSING (disabled)** | LIVE `dtparam=i2c_arm=off`; `/dev/i2c-1` absent (only `i2c-11` = RP1 CSI bus with the imx219 at 0x10, `i2c-13`, `i2c-14` = HDMI DDC) | none | H-02, H-17, H-25 |
| UART0 (GNSS) | IMPLEMENTED (device present) | LIVE `/dev/ttyAMA0`, GPIO14/15 alt4 `uart0`; receiver behind it unverified | `tests/unit/test_gps_driver.py` (SIM) | H-09 |
| UART4 (BNO085) | **MISSING** | LIVE no `uart4-pi5` overlay and no `/dev/ttyAMA4`. `docs/hardware-integration.md` (baseline line 116) says `dtoverlay=uart4`; the Pi's overlay README documents `uart4` as BCM2711-only (GPIO8-11) and `uart4-pi5` as GPIO12-13 (N-13) | none | H-03, N-13 |
| SPI0 and the HaLow radio | IMPLEMENTED (live) | LIVE `halow-dw` + `spi0-1cs`; GPIO8-11 SPI0, GPIO25 IRQ, GPIO5 RESET, GPIO3 WAKE, GPIO7 BUSY | none | H-02, H-17 |
| ToF XSHUT (GPIO22/23) | PARTIALLY IMPLEMENTED | LIVE `lgpio` holds GPIO22 LOW and GPIO23 HIGH; the addressing sequence cannot reach the sensors without an I2C bus | none | H-17 |
| HaLow claims in the pin registry | MISSING | HaLow is absent from `spec/hardware.yaml` | none | H-02, H-17 |
| Hardware self-test | PARTIALLY IMPLEMENTED; unsafe on this Pi | `backend/src/services/hw_selftest.py:36-41,192-255`, route `backend/src/api/routers/maintenance.py:62-69` (N-18) | none | H-02 |

### A.5 Electrical and power

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| Servo supply rail | UNVERIFIED; documents contradict | 12 V (`hardware.md` Actuation; `docs/tractor-platform.md` introduction) vs 24 V (`docs/hardware-overview.md`, `docs/tractor-acceptance-criteria.md`, `docs/RELEASE_NOTES.md`, `docs/field-validation-protocol.md`, `docs/servo-hat-pins.md`; **superseded notes added in this PR**). Owner-reported: two matching 150 kg·cm-class servos at nominal 10-12.6 V, possibly 8 A or more stall (not bench-verified) | none | H-05, H-20 |
| Fuses, wire gauges, stall-current budget | **MISSING** | `hardware.md` lists only "inline fuse holders + assorted fuses, 12 AWG, 4-pack" and 14 AWG primary wire; no rating, no per-branch gauge, no budget anywhere in the repository | none | H-20 |
| Independent hardwired E-stop | **MISSING** | No hardware input in code or configuration; `backend/src/safety/estop_handler.py:5-7` ("Hardware GPIO integration will be added later"); `spec/hardware.yaml:87` is a flag with no pin. `hardware.md` (Safety) lists a mushroom button but no circuit | `tests/integration/test_safety_systems.py:14-29` measures a Python flag flip | H-07, H-08 |
| PCA9685 `OE` / servo-power cut | **MISSING** | No `OE` GPIO anywhere (`config/tractor.yaml`, `spec/hardware.yaml`, code); `pca9685_driver.py:21-23`, `docs/tractor-platform.md` (PCA9685 paragraph). The 74HC123 watchdog (`hardware.md` Safety) is a BOM line with "any free GPIO" unassigned | none | H-04, H-08 |
| Relay modules (starter, PTO) | UNVERIFIED | `hardware.md` (Engine and implement): GPIO to opto module to Bosch-style relay; trigger polarity, boot-time state, pull configuration, flyback and contact ratings unknown; software assumes `active_high: true` (`config/tractor.yaml:31,36`) | `tests/unit/test_tractor.py` (fakes) | H-21, H-02b |
| Power monitor (INA3221) | PARTIALLY IMPLEMENTED | `backend/src/drivers/sensors/ina3221_driver.py:3-6,101`; which shunt carries which current is unmeasured (N-12) | `tests/contract/test_ina3221.py` | H-17 |
| Pi 5 V supply | UNVERIFIED | DROK 9-36 V to 5.2 V module (`hardware.md` Power) vs Pololu D24V50F5 (`docs/hardware-overview.md`); neither load-tested. LIVE single idle sample: `vcgencmd get_throttled = 0x0`, `EXT5V_V = 5.13 V` | none | H-17 |
| Grounding and isolation | **MISSING** | No document defines the ground topology (Pi, PCA9685, servo supply, relay supply, sensors) | none | H-20 |
| Battery and charging | UNVERIFIED | mower: 12.8 V 30 Ah LiFePO4 + 30 W solar (`spec/hardware.yaml:47-60`); Toro: the 12 V starting battery and an unmeasured charging stator (`docs/hardware-overview.md` Power) | none | H-05 |

### A.6 AI and navigation

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| VLA policy (37M parameters) on the Hailo | UNVERIFIED | Claimed in the README; the repository holds no throughput or accuracy measurement of the policy on the Pi. The only measured Hailo figure is detector throughput (`docs/autonomy-toolchain-matrix.md:87`). Performance tests are gated by `RUN_PERF_TESTS` | 15 + 1 perf tests skipped | H-23 |
| AI inference loop and action mapping | PARTIALLY IMPLEMENTED | `backend/src/services/ai_inference_service.py:494-504` sets `safety_override` when `min_distance_cm < 30`; `backend/src/services/navigation_service.py:735-760` (AI loop) and `:835-860` (`apply_ai_prediction`: override calls `tractor.emergency_stop()` at `:845-849`). `to_tractor_command()` reuses the differential mix; the twin-lever mixing is an open item (`docs/tractor-platform.md`, Configuration section) | `tests/unit/test_ai_inference_service.py`, `test_ai_navigation_loop.py` (3) | H-23, N-25 |
| Local coverage and path planning | IMPLEMENTED (unit) | `backend/src/nav/coverage_planner.py`, `coverage_patterns.py`, `path_planner.py`, `obstacle_avoidance.py:83` (`plan_path_astar`), `geofence_validator.py`, `zone_coverage.py` | `tests/test_nav_*.py`, `tests/unit/test_coverage_planner_angle.py` | H-26 |
| EKF and odometry | IMPLEMENTED (unit) | `backend/src/nav/odometry.py` | `tests/unit/test_ekf.py` (4), `test_odometry.py` | H-26 |
| Mission execution | PARTIALLY IMPLEMENTED | `navigation_service.py:175-238`: RTK fix quality is not a start precondition; the waypoint loop polls mission status only (`:265-284`); the tractor path re-asserts `blade_engaged` every 0.2 s tick (`:354-371`) | `tests/unit/test_tractor_navigation.py` | H-23, N-06, N-08 |
| Obstacle model and fusion | **STUBBED** | `navigation_service.py:33-82`: ToF only (distance < 1000, unit assumed mm and unchecked against the metres in `limits.yaml`), `Position(0, 0)` placeholder (`:51,64`), constant confidence 0.8, `is_path_clear` returns False on any hit and is not called by the mission loop; `go_to_waypoint` (`:240-372`) never reads obstacle state | `tests/contract/test_sensor_fusion.py` skipped placeholder | H-26, N-11 |
| Stereo depth / disparity | **MISSING** | No `StereoSGBM`, `reprojectImageTo3D` or calibration code. `stereo_depth` is never filled (`backend/src/models/mower_data_frame.py:174`); `backend/src/services/perimeter_recorder.py:510-511` reads a `.depth` attribute the driver never sets. Monocular `scdepthv3` HEF support exists (`hailo_driver.py:63,77`) and is unevaluated | none | H-16, H-26 |
| 2D LiDAR | **MISSING** | No driver, configuration, model or document beyond the requirement; no model selected | none | H-15, H-26 |
| ROS 2 / Nav2 | **MISSING** | No `rclpy`, `nav2` or `ros2` code, dependency or configuration anywhere in `pyproject.toml`, `backend/`, `scripts/`, `config/` | none | H-26 |
| Thor coverage strategist | IMPLEMENTED (SIM) | `backend/src/nav/thor_strategist.py:73-262` (boustrophedon plan, no-go zones, replans from uplink detections, pauses unless the fix is `rtk_fixed`, resume) | `tests/unit/test_thor_strategist.py` (4; more than 97 % coverage on a synthetic 60 x 40 m yard) | H-23 |
| Detector output to the Thor uplink | **MISSING** | `ThorLinkService(build_uplink=None)` sends an empty `Uplink` (`backend/src/services/thor_link_service.py:57,150`); nothing builds an uplink from GPS or detections | none | H-23 |
| Manual teleop (keyboard) | IMPLEMENTED | `scripts/teleop.py:36-37` (20 Hz, client-side dead-man 0.25 s, `--confirm-supervised`, REST `/api/v2/tractor/command`); the server has no dead-man (N-14) | none | H-14, H-16 |
| VR stereo streaming and head tracking | **MISSING** | No head tracking, pan/tilt, WebXR or stereo stream endpoint; `camera_stream_service.py` has no stereo path | none | H-16 |
| Agent bridge (PR #16, PR #17) | status only | PR #16: draft, one documentation file. PR #17 `feat/agent-bridge-omp`: draft, 62 files under `tools/agent_bridge/**`, not merged, nothing under `backend/` | not run | H-27 |

### A.7 RTK and GNSS

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| Rover: u-blox ZED-F9P | IMPLEMENTED (driver); UNVERIFIED on hardware | `gps_driver.py` (`F9P_USB`, `F9P_UART`); `spec/hardware.yaml:9-12` "preferred"; LIVE: no `/dev/ttyACM*` | `tests/unit/test_gps_driver.py` (5, SIM) | H-09 |
| Rover: Quectel LC29H-DA | PARTIALLY IMPLEMENTED | NMEA path only: `gps_driver.py:77-81`, `backend/src/models/hardware_config.py:74,120-122`, `backend/src/services/websocket_hub.py:98-99`, `config/hardware.yaml:8-11`. No `PAIR`/`PQTM` configuration; no LC29H-specific test | `tests/unit/test_gps_config_fallback.py` | H-09 |
| Base: Quectel LC29H-BS | **MISSING** | Zero hits for `LC29H-BS`, base survey-in or RTCM base-mode handling in `backend/`, `scripts/`, `config/` | none | H-09, H-22 |
| Pi Zero RTK base host | **MISSING** | No code or document in the repository; owner's candidate topology only | none | H-22 |
| Orange Pi 4 Pro HaLow AP / gateway | UNVERIFIED in the repository | Documented as the AP (192.168.1.140, HT-HC01P on SPI3) in `docs/autonomy-toolchain-matrix.md:47-69`; its units live on that host. `halow-link-watch` is the only HaLow unit versioned here | none | H-22 |
| NTRIP forwarder | IMPLEMENTED (client to serial) | `backend/src/services/ntrip_client.py:35-300` (caster to serial RTCM, GGA keepalive, backoff); configured through `.env`, which is absent on the deployed Pi. Serves a public caster only; there is no local base source | no dedicated test file | H-22 |
| GPS degradation handling | PARTIALLY IMPLEMENTED | `backend/src/nav/gps_degradation.py:22-35`: AUTONOMOUS to MANUAL in the in-memory `RobotState` when accuracy is above 5 m or the fix is older than 10 s; no actuator-level stop | `tests/unit/test_gps_degradation.py` | H-09 |
| Correction age and fix-transition handling | **MISSING** | No correction-age metric consumed by navigation; the Thor strategist pauses on a non-fixed position but is not wired | none | H-09 |
| RTK diagnostics | PARTIALLY IMPLEMENTED | `scripts/diagnose_gps_rtk.py`, `scripts/rtk_diagnostics_watch.py` (ZED-oriented; not run by the audit) | none | H-09 |

### A.8 Communications

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| HaLow link operations | IMPLEMENTED (operations); measured at two locations | `scripts/halow_link_watch.sh`, `systemd/halow-link-watch.{service,timer}`, `systemd/install_services.sh:34-39,66`. `docs/autonomy-toolchain-matrix.md:30-70`: Pi to Thor 2026-10-06 TCP 4.79 / 4.61 Mbit/s, 50 pings at 5 Hz 0 % loss, RTT 7.3 / 18.0 / 73.2 ms (min / avg / max); Pi to AP 2026-10-13 TCP 4.84 / 5.17 Mbit/s, 30-minute soak 0 % loss. LIVE `halow0` was DOWN at audit time (traffic via `eth0` and `tun0`) | none | H-16, H-17, H-22, H-23 |
| HaLow recovery (power cycle, driver reload, Pi reboot) | UNVERIFIED | Same document: "A real HaLow drop and a Pi reboot were not exercised"; a stopped backend once took the radio down (`:76`); the AP radio vanished after a TX-cap change (`:49`) | none | H-23 |
| Pi-Thor link protocol | IMPLEMENTED (protocol) | `backend/src/models/thor_link.py:22-113`: v1, JSON over UDP, 1200 B cap, 10 Hz uplink, waypoint windows of up to 16 | `tests/unit/test_thor_link.py` (8) | H-13, H-23 |
| Pi-side Thor link service | PARTIALLY IMPLEMENTED; **not wired** | `backend/src/services/thor_link_service.py:49-172`; instantiated only in tests; `current_plan()` (`:115-117`) has no consumer; LIVE no listener on UDP 47101 | `tests/unit/test_thor_link.py` (includes a loopback UDP round trip) | H-13, H-23, N-15 |
| Link authentication and integrity | **MISSING** | `_on_datagram` ignores the source address (`:98-110`); the socket binds `0.0.0.0:47101` (`:54`); no key, HMAC, nonce or source check | none | H-13, H-14, N-15 |
| Replay and stale-plan protection | PARTIALLY IMPLEMENTED | `seq <= _last_rx_seq` drop (`:106-107`); no epoch, plan age or `start_index` check; one datagram with a very large `seq` makes every later legitimate datagram look reordered until the Pi restarts | `tests/unit/test_thor_link.py` (duplicate case) | H-13, N-15 |
| Link-loss to safe hold | IMPLEMENTED (software, fake clock) | `_evaluate_hold` (`:132-145`): `timeout_s = 0.5`, blade off then levers neutral, non-revoking; `tick` swallows exceptions (`:121-130`), so `safe_hold` can read true while actuation failed | `tests/unit/test_thor_link.py` | H-08, H-23 |
| Retransmission and acknowledgement | MISSING | Uplink is fire-and-forget; the Thor re-sends waypoint windows on each uplink (implicit) | none | H-13 |
| REST and WebSocket API | IMPLEMENTED | `backend/src/main.py:227-247` (prefix `/api/v2`); `/ws/telemetry` and `/ws/control` under `/api/v2/` and as legacy root routes (`backend/src/api/rest.py:34-36`); `/ws/control` drains and acknowledges only (`backend/src/api/routers/telemetry.py:175-206`) | `tests/contract/*`, `tests/integration/*` | H-14 |

### A.9 Security

| Subsystem | Status | Evidence | Tests | Gap IDs |
|---|---|---|---|---|
| Operator authentication | PARTIALLY IMPLEMENTED; **disabled in the shipped and deployed unit** | `backend/src/api/deps.py:33-57` (code default is on); `systemd/lawnberry-backend.service:24` sets `OPERATOR_AUTH_REQUIRED=0` (LIVE identical); the backend binds `0.0.0.0:8081` (`:33`) and is reachable over the VPN. `PRODUCT.md:11` and `docs/authentication-config.md` record it as a decision | `tests/integration/test_operator_authz.py`, `test_auth_hardening.py`; TDD skips in `tests/unit/test_websocket_auth.py` and `tests/integration/test_manual_control_auth.py` | H-14 |
| Routes with no auth dependency | PARTIALLY IMPLEMENTED | Missions (`backend/src/api/mission.py:9-90`), AI enable (`backend/src/api/routers/ai_control.py:143-176`), `POST /control/emergency_clear` (`backend/src/api/safety.py:19`), `POST /control/emergency` and `/control/emergency-stop`, the debug injectors for GPS, geofence, ToF and tilt (`backend/src/api/navigation.py:32-100`, `backend/src/api/routers/sensors.py:450-516`), `GET /system/selftest` | none | H-14, N-16 |
| Tractor routes | PARTIALLY IMPLEMENTED | `backend/src/api/routers/tractor.py:36-93`: writes use `require_operator_auth` (off in production); `/tractor/emergency-stop` is unauthenticated by design; `/tractor/clear-emergency` can be called remotely with no local acknowledgement. With auth off, any host that can reach the port can `authorize`, start, command and engage | `tests/integration/test_operator_authz.py` (tractor routes not among the confirmed cases) | H-14, N-10 |
| WebSocket authorisation | PARTIALLY IMPLEMENTED | `backend/src/api/routers/auth.py:318-386` admits localhost, `testserver` or any client under `SIM_MODE=1` as an operator; the HTTP handshake helper `backend/src/api/routers/telemetry.py:96-111` accepts any non-empty bearer string | skipped | H-14, N-16 |
| Default credentials | **present** | `operator123` and `admin`/`admin`: `backend/src/api/rest_v1.py:108`, `backend/src/api/routers/auth.py:396-397`, `backend/src/services/auth_service.py:233` | tests use them | H-14, N-17 |
| JWT secret persistence | PARTIALLY IMPLEMENTED | `auth_service.py:83-85` generates a random key per process start when `LAWN_BERRY_AUTH_SECRET` is unset (no `.env` live); a second mechanism exists (`backend/src/core/secrets_manager.py:81-86`) | none | N-21 |
| CORS | PARTIALLY IMPLEMENTED | `backend/src/middleware/security.py:191-228`: origin allow-list with credentials; `http://<pi-ip>:3000` is not in the defaults unless `LAWN_BERRY_ALLOWED_ORIGINS` is set (not set) | `tests/unit/test_security_middleware.py` | N-21 |
| Pi-Thor link authentication | **MISSING** | see A.8 | none | H-13, N-15 |
| Secret hygiene | PARTIALLY IMPLEMENTED | A local pre-commit hook only (`scripts/pre-commit-secret-scan.sh`); no secret-scanning workflow; no secrets found in tracked files by the audit's search | none | N-22 |

### A.10 Safety

Summary of the safety audit; the full traces are in Part D.

| # | Subsystem | Status | Evidence | H / N IDs |
|---|---|---|---|---|
| 1 | Independent hardwired E-stop | **MISSING** | A.5. The only quasi-independent stop is the RC channel-5 stop on the RP2040 (`code.py:326-336`), inactive while the Pi holds USB control, on the same MCU as the drive output, and irrelevant to the PCA9685 Toro platform | H-07, H-08 |
| 2 | Software E-stop (tractor) | PARTIALLY IMPLEMENTED | `POST /api/v2/tractor/emergency-stop` (`backend/src/api/routers/tractor.py:85-88`) to `tractor_service.py:258-288`: flag, revoke, PTO relay off first, then each channel in its own try/except. A failed channel keeps its last pulse and the reply is still `status: emergency_stop`; `RelayActuator.set` is unguarded (`tractor_actuators.py:111-115`), so a GPIO error raises before the servos are safed | H-08, N-27 |
| 3 | General E-stop routes | **STUBBED / false assurance** | `POST /control/emergency-stop` (`backend/src/api/rest.py:676-692`) only sets flags and replies `motors_stopped: true, blade_disabled: true`; `POST /control/emergency` (`:600-673`) stops the RoboHAT only if its serial link is up and blocks follow-up commands for 0.2-0.3 s; neither touches the tractor or the IBT-4 | N-01, N-02, N-03 |
| 4 | Motor watchdog | PARTIALLY IMPLEMENTED | Event-loop liveness only, tractor-only, one-shot (`watchdog.py:54-69`); `main.py:155-158` gives it a private `MotorAuthorization` nothing else reads; `safety_validator.py:37-38` validates `watchdog_timeout_ms >= estop_latency_ms` (1000 ms vs 100 ms) | H-08, N-04 |
| 5 | Loss of comms | **MISSING** for the operator; Thor link implemented but unwired | A.8; `/tractor/command` has no TTL or lease; `/control/drive` validates `duration_ms` but never schedules a stop (`rest.py:378-388`); the RoboHAT host loop re-sends the last PWM about every 0.9 s (`robohat_service.py:246-269`) | H-13, H-14, N-14, N-15 |
| 6 | Servo neutral and parking brake | PARTIALLY IMPLEMENTED | Neutral is a command (`ServoCalibration` defaults `tractor_actuators.py:30-39`; `config/tractor.yaml:22-27` illustrative values); lever self-centring unverified; a real parking brake exists only at the outboard PARK position, reachable by hand only (`docs/tractor-platform.md`, Physical integration decisions item 2) | H-06, H-08 |
| 7 | PCA9685 `OE` and bus faults | `OE` **MISSING**; bus-fault propagation IMPLEMENTED; silent no-op when absent at boot | A.1, A.5 | H-04, H-08, N-07, N-28 |
| 8 | Safe startup | PARTIALLY IMPLEMENTED | Safe defaults (`tractor_service.py:109-110`), parking in `initialize()` (`:127-136`), start interlocks (`:233-244`). Not latched across restart (N-10); no engine-run sensor (`:248`); mission auto-crank (N-08); parking skipped entirely while `tractor.enabled` is false (`main.py:146`) | H-06, H-12, N-08, N-10 |
| 9 | Safe shutdown | **MISSING** | Teardown never parks the tractor (`main.py:184-201`); `systemd/lawnberry-backend.service:37-44` has no `ExecStopPost`; the relay GPIO state at process exit and SIGKILL is unknown; `finally: set(False)` in `pulse` does not run on SIGKILL (`tractor_actuators.py:117-124`); `MemoryMax=512M` makes OOM kills possible | H-08, N-09 |
| 10 | PTO latching vs momentary | **MISSING / CONTRADICTED** | Software models a latching relay (`config/tractor.yaml:32-36`, `tractor_service.py:228-229`); the owner reports a momentary OEM switch; no PTO feedback; no pulse-to-toggle logic | H-21, N-26 |
| 11 | Operator authentication | PARTIALLY IMPLEMENTED; deployed OFF | A.9 | H-14 |
| 12 | OEM interlock preservation | MISSING in software (documentation only) | Nothing senses the seat, PARK or OEM ROS; `docs/tractor-platform.md` (items 3-4) and `docs/tractor-acceptance-criteria.md` sections 8-9 and 11 record the physical facts. The mission auto-crank contradicts "a human starts the engine with the levers in PARK" (N-08) | H-06, H-07 |
| 13 | Tilt cutoff | PARTIALLY IMPLEMENTED; defeated in autonomous mode | N-06; 30° threshold (`config/limits.yaml:5`); the blade-stop latency is measured only as a function call (`tests/integration/test_safety_systems.py:32-45`) | H-08, N-06, N-24 |
| 14 | Obstacle detection and stop | **MISSING** for actuation | `SafetyTriggerManager.trigger_obstacle`, `trigger_ultrasonic_obstacle`, `trigger_low_battery`, `trigger_high_temp` (`safety_triggers.py:94-141`) have no production callers; thresholds are consumed only by the debug injector (`sensors.py:474`) | H-02, H-10, H-19, H-26, N-11 |
| 15 | Stopping distance | **MISSING** | No speed-dependent stopping distance, reaction time or braking model; fixed constants only (ToF 0.2 m, ultrasonic 30 cm, AI 30/50 cm, navigation 1000 of unclear unit); `ramp_rate` is an unused field (`backend/src/models/motor_control.py:53`) | H-26 |

### A.11 Tests

| Item | Status | Evidence |
|---|---|---|
| Full suite | PASS in SIM | `SIM_MODE=1 pytest tests --ignore=tests/soak -p no:cacheprovider`: **530 passed, 69 skipped, 0 failed**, 68.8 s, 2 warnings (httpx deprecation). HIL and soak were not run |
| Skipped tests | UNVERIFIED paths | 29 + 1 "Placeholder integration test" and 10 "Placeholder contract test" (`RUN_PLACEHOLDER_*`); 15 + 1 perf tests (`RUN_PERF_TESTS`); `_require_bearer_auth` TODO(v3) skips and a legacy `/ws/telemetry` skip (`tests/unit/test_websocket_auth.py:20,27,35,58`; the stated reason is stale because `/ws/telemetry` is registered at `backend/src/api/rest.py:35`); "Auth enforcement not yet implemented - TDD test" (`tests/integration/test_manual_control_auth.py:23,69,119`); hardware self-test (`RUN_HW_TESTS`), 8-hour soak (`RUN_8H_SOAK`), quick soak (`RUN_SOAK_CI`); one test needing `cv2` (`tests/unit/test_yolo26_decode.py:185`) |
| HIL | **MISSING** | `tests/hil/test_hil_probe.py` runs `scripts/hil_probe.py` under `SIM_MODE` and parses a CSV; it is not a hardware test. `pyproject.toml:60` sets `testpaths = ["tests"]`, so the CI `pytest -q` collects `tests/hil` (the SIM probe) and `tests/soak` (skipped without its `RUN_*` flag); no job touches hardware |
| Pin checker | UNTESTED | No file in `tests/` references `scripts/check_hardware_pin_conflicts.py`; none loads `spec/hardware.yaml` |
| Areas with **no test at all** | MISSING | Stop-distance logic, comms-loss behaviour, shutdown parking, PCA9685-absent reporting in production, tilt re-engagement by navigation, IBT-4 e-stop, PTO momentary behaviour, Thor link authentication |

What the passing safety tests prove and do not prove:

| Area | Tests (all pass, SIM) | What they do not prove |
|---|---|---|
| Servo mapping, relays, interlocks, e-stop sequence | `tests/unit/test_tractor.py` (16) | Any physical outcome |
| PCA9685 arithmetic and fault modes | `tests/unit/test_pca9685_driver.py` (8) | `OE`, power-up state, a real I2C bus |
| Tilt and watchdog wiring | `tests/unit/test_tractor_safety_monitor.py` (9) | A real IMU, re-engagement by navigation, latency |
| Thor link logic | `tests/unit/test_thor_link.py` (8) | Wiring, authentication, consumers, hardware |
| Bare watchdog and bool e-stop | `tests/contract/test_watchdog.py`, `tests/integration/test_safety_systems.py` | Any hardware; the "latency under 100 ms" test times a flag flip (`:14-29`) and the tilt test times a function call (`:32-45`) |
| RoboHAT driver and service | `tests/unit/test_robohat_rp2040_driver.py`, `test_robohat_service_usb_control.py` | The firmware protocol |
| Operator auth | `tests/integration/test_operator_authz.py`, `test_auth_hardening.py` | Tractor, mission, AI and `emergency_clear` routes; WebSocket paths |

### A.12 Documentation

| Contradiction or gap | Where | State after this PR |
|---|---|---|
| GPS receiver (ZED-F9P vs LC29H-DA) | `README.md`, `spec/hardware.yaml:9`, `hardware.md`, `config/hardware.yaml:8-11`, `docs/autonomy-toolchain-matrix.md:14,27`, `.github/copilot-instructions.md` | Annotated as unreconciled in README, matrix and `hardware.md` (H-09). `spec/` and `config/` unchanged: an owner decision with a physical check |
| Servo rail 12 V vs 24 V | `hardware-overview.md`, `tractor-acceptance-criteria.md`, `RELEASE_NOTES.md`, `field-validation-protocol.md`, `servo-hat-pins.md`, `tractor-platform.md` | Superseded notes added; historical text kept (H-05, H-20) |
| `OE` polarity | `tractor-acceptance-criteria.md` section 1 | Corrected, bench measurement added (H-04) |
| Pin 17 voltage, divider schematic and margin, resistor BOM | `ultrasonic-wiring-guide.md` | Corrected; whole guide marked legacy (H-01, H-18) |
| Ultrasonic part and count | guide, matrix, README | Marked legacy / MISSING for JSN-SR04T (H-10, H-19). `spec/hardware.yaml`, `PRODUCT.md:32` unchanged |
| Matrix rows (IBT-4, BNO085, ultrasonic, stereo, GPIO/I2C/UART/USB tables, starter/PTO) | `hardware-feature-matrix.md` | Corrected |
| Ports 8000/3001 vs 8081/3000, `/control/estop`, `/missions`, telemetry routes, fictional config files | `README.md` | Corrected. **Still open:** `docs/deployment-guide.md:172-244` uses 8000 and `ws://localhost:8000/ws/telemetry` |
| `/control/emergency-stop` listed as an e-stop | `docs/OPERATIONS.md:59` | **Open** (N-01): the route only sets flags |
| Damaged paragraph and unqualified "verified conflict-free" | `tractor-platform.md` | Repaired and qualified (H-02b) |
| Platform tagging (differential-drive vs Toro) | README, matrix, hardware-overview, servo-hat-pins, hardware.md | Tags added (H-11). `docs/hardware-integration.md` and `spec/hardware.yaml` are changed on the checker branch, not here |
| Hailo-8 vs 8L | `spec/hardware.yaml:37`, README, `hardware.md`, matrix | Annotated; verify with `hailortcli fw-control identify` |
| HaLow HAT identity | live `config.txt` (Heltec HT-HC01P V2) vs `halow-bringup.service` and `docs/autonomy-toolchain-matrix.md:17,47` (Seeed WM6108 / HT-HC01P) | **Open**, TBD by inspection |
| OS and kernel | `spec/hardware.yaml:3`, `docs/hardware-integration.md` (Bookworm); `docs/autonomy-toolchain-matrix.md:14` (Ubuntu 24.04.4, 6.8.0-raspi); LIVE 6.12.21-morse+ | Annotated in README; **open** |
| Version naming v2 / v3 | README title vs `systemd/*.service` descriptions | **Open** |
| `PRODUCT.md:25` ride-on Craftsman tractor with Ackermann steering | superseded on 2026-09-01 (`docs/RELEASE_NOTES.md`) | **Open** (owner file) |
| `docs/authentication-config.md:650` "emergency123" | not found in code | **Open** |
| Documentation-sync rule for `docs/code_structure_overview.md` | `.github/copilot-instructions.md`; `scripts/check_docs_drift.sh` only checks that some docs file changed | **Open** (N-22) |
| `AGENTS.md` is generic code-index boilerplate | repository root | **Open** |
| `hardware.md` Safety: 74HC123 fed from "any free GPIO" | `hardware.md` | **Open**: no GPIO is assigned (B.1) |

### A.13 Workflows (CI)

| Workflow | Trigger | What it runs | Gap |
|---|---|---|---|
| `sim-tests.yml` | push, PR | `uv sync --frozen`, `SIM_MODE=1 LBY_ACCEL=cpu uv run pytest -q`, an API smoke on port 8000 (`:16-26`) | Includes the 69 skips (auth TDD tests never run); collects the SIM-only `tests/hil` probe and the skipped soak test (`pyproject.toml:60`); no hardware job and no dedicated safety job; port 8000 vs the deployed 8081 |
| `lint.yml` | push, PR | `ruff@0.15.19 check .` and `format --check .` (`:17-19`) | Not run by the audit |
| `hardware-guard.yml` | PR | the pin checker, then `grep` presence checks on `spec/hardware.yaml` (`:14-59`) | Checks the manifest, not reality; `:54-59` enforces CH1 battery / CH3 solar (N-12); `:39-44` blocks the strings "soil", "weather station", "jetson", "movidius" |
| `docs-drift.yml` | PR, push to main | `scripts/check_docs_drift.sh` | Path-regex only (`:40`); passes if any `docs/**` file changed; locally `git fetch --depth=0` errors and is swallowed (`:18`), so a local run on an unchanged branch is vacuous; does not look at `config/*.yaml`, `hardware/` or `robohat-rp2040-code/` |
| `config-lint.yml` | PR | yamllint, dotenv-linter, shellcheck via `ludeeus/action-shellcheck@master` (unpinned, `:14`), `systemd-analyze verify` with ExecStart-missing errors filtered | Does not compare units with the deployed Pi |
| `dep-audit.yml` | PR | `pip-audit` and `npm audit` | Both commands end in `\|\| true` (`:18,28`), so they never fail the job |
| `deps-lock.yml` | PR | `uv.lock` present and locked; forbids `pycoral`/`edgetpu` in `pyproject.toml` (`:22`) | none |
| `webui-build.yml` | PR | `npm run build` and Playwright e2e against a mocked backend (`:16-28`) | vitest unit tests are not run in CI |
| `codeql.yml`, `pr-hygiene.yml`, `todo-policy-check.yml` | various | not individually audited | UNVERIFIED |
| **Absent** | n/a | A hardware-in-the-loop or scheduled soak job that touches hardware, secret scanning, a check that `systemd/` matches the deployed units, any deploy workflow | N-22 |

### A.14 Deployment

| Item | Status | Evidence |
|---|---|---|
| Backend unit vs deployed | PARTIALLY IMPLEMENTED | LIVE unit equals `systemd/lawnberry-backend.service` except `User=kp` (repo `pi`) and comments; both `--port 8081`, `SIM_MODE=0`, `OPERATOR_AUTH_REQUIRED=0`. A live drop-in `/etc/systemd/system/lawnberry-backend.service.d/thor-upload.conf` sets `THOR_UPLOAD_ENABLED=1` (repo: `0`, `:27`) and is not in the repository |
| Installer | PARTIALLY IMPLEMENTED | `systemd/install_services.sh:20-66` copies units verbatim; units hard-code `User=pi` (`lawnberry-backend.service:10-11`; the database unit `chown -R pi:pi`); the deployed user is `kp` and the installer does not substitute it |
| Units installed live | PARTIALLY IMPLEMENTED | Active: backend, camera, database, frontend, sensors (placeholder). `lawnberry-health.service` present but disabled. Not installed: remote-access, cert-renewal and timer, backup and timer, `halow-link-watch` timer, acme-renew, although the installer enables most of them (`:55-66`) |
| HaLow units | UNVERIFIED in the repository | LIVE `halow-bringup`, `halow-failover`, `halow-lan-routes.timer`, `wifi-halow-*`, `wificommand` and a `wpa_supplicant@halow0` drop-in exist on the Pi and are not versioned here |
| Listeners | UNVERIFIED | LIVE: 8081 and 3000 on all interfaces; also 1337, 1377 (wificommand), 5015, 2947 (gpsd, loopback), 11434 (ollama, loopback). Owners of 1337 and 5015 unknown (TBD) |
| Secrets | PARTIALLY IMPLEMENTED | No `.env` live (the unit uses `EnvironmentFile=-`); `config/secrets.json` exists, mode 600, git-ignored |
| Host hardware watchdog | UNVERIFIED live | `systemd/10-watchdog.conf` (`RuntimeWatchdogSec=20s`) reboots a hung host and does not neutralise actuators; whether it is installed on the Pi was not checked |
| Process limits | PARTIALLY IMPLEMENTED | `MemoryMax=512M`, `CPUQuota=80%` (`lawnberry-backend.service:64-65`): an OOM kill is a SIGKILL with no cleanup (N-09) |
| Containers and deploy path | MISSING | No Dockerfile, compose file or deploy workflow |

### A.15 Findings outside the PR #18 register (N-01 to N-31)

These are proposed additions to the gap register. Each cites the file and line (baseline `765c737`).

| ID | Finding | Evidence | Related H |
|---|---|---|---|
| N-01 | General E-stop routes do not actuate everything yet reply that motors and blade are stopped | `backend/src/api/rest.py:600-692` | H-08 |
| N-02 | The e-stop block on follow-up drive and blade commands lasts 0.2-0.3 s and is not latched | `rest.py:175-179,204-222,324,359,503,550,621-627` | H-08 |
| N-03 | The IBT-4 blade is never e-stopped; `BladeService.emergency_stop` and `IBT4BladeDriver.set_estop` have no callers | `blade_service.py:46-55`, `ibt4_gpio.py:103-106`, `rest.py:550-555` | H-08 |
| N-04 | The watchdog heartbeat is the monitor's own loop liveness; the watchdog is one-shot; its `EstopHandler` holds a private `MotorAuthorization` | `watchdog.py:54-69`, `tractor_safety_monitor.py:84-87`, `main.py:155-158` | H-08 |
| N-05 | If tractor initialisation raises, the actuation API stays live with no watchdog and no tilt monitor, and nothing alerts | `main.py:144-170` | H-08, H-12 |
| N-06 | Tilt cutoff: one-shot blade-off, re-engaged by the mission loop within about 200 ms [INF]; IMU loss is fail-open; on real hardware the IMU is a stub that reports zero roll and pitch | `navigation_service.py:354-371`, `tractor_service.py:218-231,304`, `tractor_safety_monitor.py:37-47,97-115`, `sensor_manager.py:186-200`, `bno085_driver.py:69-78` | H-08 |
| N-07 | A PCA9685 absent or unreachable at boot makes every lever command and e-stop a silent success | `pca9685_driver.py:117-126,155-156`, `tractor_service.py:157-172,199-203` | H-04, H-08 |
| N-08 | `execute_mission` cranks the engine whenever `engine_running` is false, and the state is OFF after any backend restart, so a mission start can pulse the starter on a running engine; it contradicts "a human starts the engine with the levers in PARK" | `navigation_service.py:188-195`, `tractor_service.py:109-110,246-250` | H-06 |
| N-09 | No safe shutdown: teardown does not park actuators, no `ExecStopPost`, relay pin state at exit is unknown | `main.py:184-201`, `systemd/lawnberry-backend.service:37-44`, `tractor_actuators.py:95-124` | H-08 |
| N-10 | E-stop is not latched across a restart; `clear-emergency` needs only the (off) operator auth, with no local acknowledgement | `tractor_service.py:109-110,290-293`, `routers/tractor.py:91-93` | H-08, H-14 |
| N-11 | Obstacle triggers are advisory: no production caller, thresholds used only by the debug injector, `go_to_waypoint` ignores obstacle state, the detector uses a `Position(0, 0)` placeholder | `safety_triggers.py:94-141`, `routers/sensors.py:474`, `navigation_service.py:33-82,240-372` | H-10, H-19, H-26 |
| N-12 | INA3221 channel map: spec and CI say CH1 battery / CH3 solar; driver and `config/hardware.yaml` say CH1 solar / CH3 battery | `spec/hardware.yaml:61-70`, `hardware-guard.yml:54-59`, `ina3221_driver.py:3-6,101`, `config/hardware.yaml:29-33` | none |
| N-13 | `dtoverlay=uart4` is the wrong overlay for the Pi 5 (needs `uart4-pi5`); `rpi4_alt_pin: 21/24` are physical pins read as BCM | `docs/hardware-integration.md` (baseline lines 68, 116), `spec/hardware.yaml:143-144`, LIVE overlay README | H-03 |
| N-14 | The RoboHAT host loop re-sends the last PWM about every 0.9 s, defeating the firmware's USB-silence failsafe; `duration_ms` is validated and ignored; `/tractor/command` has no TTL, lease or server dead-man | `robohat_service.py:246-269`, `rest.py:378-388`, `routers/tractor.py:67-70` | H-08, H-14 |
| N-15 | Thor link: any source address accepted, no HMAC or epoch, sequence-poisoning denial of service (a Thor restart that resets `seq` is dropped the same way), hold is non-revoking, `tick` swallows actuation errors, `current_plan()` has no consumer | `thor_link_service.py:45-62,98-145` | H-13, H-23 |
| N-16 | Missions, AI enable, `emergency_clear`, debug injectors and `/system/selftest` have no auth dependency; WebSocket helpers accept any bearer or any `SIM_MODE` client | see A.9 | H-14 |
| N-17 | Hard-coded default credentials | A.9 | H-14 |
| N-18 | `hw_selftest.ultrasonic_probe()` claims and drives TRIG pins GPIO4/27/11 regardless of `ultrasonic_enabled`, with `gpiochip_open(4)` hard-coded | `hw_selftest.py:36-41,192-255`, `routers/maintenance.py:62-69` | H-02 |
| N-19 | `limits.yaml` says an ultrasonic reading "triggers e-stop" and no code does that; `SafetyLimits.ultrasonic_enabled` defaults to True | `config/limits.yaml:16-17`, `models/safety_limits.py:20` | H-02 |
| N-20 | The sensor-manager systemd unit is a placeholder (`sleep infinity`) and reports active | `systemd/lawnberry-sensors.service:16-17` | none |
| N-21 | Deployment drift: undocumented drop-in, unversioned HaLow units, `User=pi` vs `kp`, no `.env` (random JWT secret per restart), CORS defaults exclude the LAN origin | A.9, A.14 | H-11 |
| N-22 | CI blind spots: manifest-only hardware guard, path-regex docs drift, audits that cannot fail, no hardware-in-the-loop, scheduled soak, dedicated safety or secret-scan job | A.13 | H-11 |
| N-23 | Documentation defects beyond those corrected here (A.12 "Open" rows) | A.12 | H-11 |
| N-24 | Unmeasured safety claims: "e-stop latency under 100 ms" times a flag flip; "tilt cutoff under 200 ms" times a function call | `tests/integration/test_safety_systems.py:14-45` | H-08 |
| N-25 | A frame with no ultrasonic data reads as an obstacle at 0 cm, which stops the AI path by accident rather than by design [INF] | `mower_data_frame.py:104-110`, `ai_inference_service.py:494-502`, `navigation_service.py:845-849` | H-26 |
| N-26 | Engine and PTO state are commanded, not sensed; `stop_engine` only sets a flag, there is no ignition-kill actuator | `tractor_service.py:246-256` | H-06, H-21 |
| N-27 | `set_levers` applies left then right with no rollback; `ServoActuator.command` records the state before the write; a relay failure can pre-empt servo safing | `tractor_service.py:196-197`, `tractor_actuators.py:65-69,95-115` | H-08 |
| N-28 | The PCA9685 write runs in a thread with no timeout, so a hung bus can stall the e-stop sequence for as long as the kernel I2C timeout allows [INF, DS] | `pca9685_driver.py:170-171`, `tractor_service.py:272-283` | H-04, H-08 |
| N-29 | The watchdog circuit in `hardware.md` (Safety, "Circuit") does not follow from its own formula: `t = 0.45 x R x C` with 50 kOhm and 1 uF gives 22.5 ms, not "near a 200 ms window", and is shorter than the 50 ms heartbeat period it must ride through. Its GPIO is unassigned. No value from that paragraph may be used; the window is derived from the measured stopping distance (M-43) with the safety reviewer [DS formula as quoted by the repository] | `hardware.md` (Safety, "Circuit"; section cited because this PR edits the file) | H-08 |
| N-30 | Entering safe hold sets `safe_hold` before the actuation awaits. If `engage_blade(False)` raises, the lever call is skipped, and the retry branch re-asserts only the levers (and only while `state.moving`); the blade is never retried, while the uplink reports `safe_hold: true` [INF, not executed] | `thor_link_service.py:132-145,150-160` | H-08, H-13 |
| N-31 | The GPS-degradation monitor does not observe real missions: it acts only when the `RobotState` mode is `AUTONOMOUS`, whose only writer in `backend/src` is the debug route `POST /api/v2/nav/mode`, while missions and the AI loop use a different enum and state (`AUTO`, `AI`); its "fix age" is the age of the whole state's `last_updated`, not of the GNSS fix; when it does fire it changes an in-memory flag and no actuator | `nav/gps_degradation.py:69-89`, `api/navigation.py:93-100`, `navigation_service.py:198`, `models/robot_state.py:18-22`, `models/navigation_state.py:12-22`, `main.py:108-110` | H-09, H-26 |

## Part B. Phase 3 hardware deliverables

Everything in Part B is a **requirement or a proposal**, not an as-built record. No connection below has been
made, measured or energised. Every unverified electrical detail is `TBD`, and ratings are deliberately absent
where the repository has no evidence for them.

### B.1 GPIO allocation and conflict report

#### B.1.1 Evidence base

- Repository: `spec/hardware.yaml` (`pins`), `config/tractor.yaml`, `config/hardware.yaml`, `config/limits.yaml`,
  the drivers, `docs/ultrasonic-wiring-guide.md`, `docs/hardware-integration.md`.
- LIVE, read-only, 2026-10-24: `/boot/firmware/config.txt` (HaLow pin comments: WAKE=GPIO3 at `:20-22`,
  Reset=GPIO5 and IRQ=GPIO25 at `:28`, BUSY=GPIO7 at `:30-31`; `dtparam=i2c_arm=off` at `:23`), the decoded
  `halow-dw.dtbo` (`reset-gpios = <&gpio 5>`, `spi-irq-gpios = <&gpio 25>`), debugfs pinctrl and `gpioinfo`.
- Reported by the checker branch (`audit/checker`, **UNVERIFIED by this document**): the vendor schematic in `~/halow`
  on the Pi suggests the HaLow HAT reference design can also occupy GPIO2, GPIO6 and GPIO16 (RGB LED), GPIO14 and
  GPIO15 (UART header), GPIO21 (button) and GPIO4 (a TRST reference overlay). These are warnings to measure
  (M-02), not facts.
- The physical-pin to BCM mapping is the standard Raspberry Pi header [DS]; every repository table that lists a
  pin agrees with it.

#### B.1.2 The 40-pin header on the deployed Pi

| Pin | Signal [DS] | Claimed by the repository | LIVE 2026-10-24 | Verdict |
|---|---|---|---|---|
| 1 | 3V3 | rail | | rail |
| 2 | 5V | legacy ultrasonic S1 VCC | | rail |
| 3 | GPIO2 / SDA1 | I2C1 SDA: ToF, BME280, INA3221, OLED, PCA9685 (`config/tractor.yaml:18`; PR #18 plan) | `i2c_arm=off`: no header I2C; function none. HAT note: RGB LED? | **BLOCKED** (C-05) |
| 4 | 5V | legacy ultrasonic S2 VCC | | rail |
| 5 | GPIO3 / SCL1 | I2C1 SCL (same list) | **HaLow WAKE** (`config.txt` comment); I2C1 disabled for this reason | **CONFLICT** (C-05) |
| 6 | GND | legacy S1 GND; PCA9685 logic GND (PR #18 plan) | | ground |
| 7 | GPIO4 | legacy S1 TRIG (`ultrasonic_driver.py:81`); `limits.yaml:17` says the HaLow HAT owns it | unclaimed, function none. HAT note: TRST reference overlay? | **UNVERIFIED** (C-01) |
| 8 | GPIO14 / TXD0 | UART0 TX to the GNSS receiver (`config/hardware.yaml:9-11`) | alt4 `uart0` (claimed by `1f00030000.serial`). HAT note: UART header? | in use |
| 9 | GND | | | ground |
| 10 | GPIO15 / RXD0 | UART0 RX from the GNSS receiver | alt4 `uart0`. HAT note: UART header? | in use |
| 11 | GPIO17 | legacy S1 ECHO | unclaimed | free live; legacy claim |
| 12 | GPIO18 | none | unclaimed | **candidate** |
| 13 | GPIO27 | legacy S2 TRIG | unclaimed | free live; legacy claim |
| 14 | GND | legacy S2 GND | | ground |
| 15 | GPIO22 | ToF-L XSHUT (`spec/hardware.yaml:137`, `config/hardware.yaml:25`) | held by `lgpio` (the backend), output LOW | in use |
| 16 | GPIO23 | ToF-R XSHUT (`spec/hardware.yaml:138`) | held by `lgpio`, output HIGH | in use |
| 17 | 3V3 | rail (the legacy guide called it 5 V) | | rail (C-10) |
| 18 | GPIO24 | Blade IN1 (`spec/hardware.yaml:139`) | unclaimed | assigned (mower); driver stub |
| 19 | GPIO10 / SPI0 MOSI | legacy S2 ECHO | alt0 `spi0` (`1f00050000.spi`) | **CONFLICT** (C-01) |
| 20 | GND | legacy S3 GND | | ground |
| 21 | GPIO9 / SPI0 MISO | legacy S3 ECHO | alt0 `spi0` | **CONFLICT** (C-01) |
| 22 | GPIO25 | Blade IN2 (`spec/hardware.yaml:140`) | claimed `mm610x_spi_irq_gpio`, input, IRQ 170 | **CONFLICT** (C-03) |
| 23 | GPIO11 / SPI0 SCLK | legacy S3 TRIG | alt0 `spi0` | **CONFLICT** (C-01) |
| 24 | GPIO8 / SPI0 CE0 | HaLow chip select | claimed `spi0 CS0` | in use (HaLow) |
| 25 | GND | | | ground |
| 26 | GPIO7 / SPI0 CE1 | HaLow BUSY (`config.txt` comment); CE1 disabled by `spi0-1cs` | function none | in use (HaLow); continuity UNVERIFIED |
| 27, 28 | ID_SD, ID_SC | HAT EEPROM | | reserved |
| 29 | GPIO5 | starter relay (`config/tractor.yaml:31`) | function gpio, level high, `gpioinfo` "unused"; the `halow-dw` overlay names it `reset-gpios` | **CONFLICT** (C-02) |
| 30 | GND | | | ground |
| 31 | GPIO6 | ToF-L IRQ (`spec/hardware.yaml:141`) | unclaimed. HAT note: RGB LED? | reserved; no reader (C-08) |
| 32 | GPIO12 | ToF-R IRQ (`spec/hardware.yaml:142`) **and** BNO085 TXD4 (`:144`) | unclaimed, function none | **CONFLICT** (C-04) |
| 33 | GPIO13 | BNO085 RXD4 (`spec/hardware.yaml:143`, `docs/hardware-integration.md:67`) | unclaimed | assigned; overlay absent (C-06) |
| 34 | GND | | | ground |
| 35 | GPIO19 | none | unclaimed | **candidate** |
| 36 | GPIO16 | none | unclaimed. HAT note: RGB LED? | free live; HAT warning |
| 37 | GPIO26 | blade PTO relay (`config/tractor.yaml:36`) | unclaimed | assigned (Toro) |
| 38 | GPIO20 | none | unclaimed | **candidate** |
| 39 | GND | | | ground |
| 40 | GPIO21 | none | unclaimed. HAT note: button? | free live; HAT warning |

#### B.1.3 Conflict register

| ID | Conflict | Evidence | Caught by the checker at baseline? | Gap | Resolution path |
|---|---|---|---|---|---|
| C-01 | Legacy ultrasonic GPIO9/10/11 are the HaLow SPI0 MISO/MOSI/SCLK; the `limits.yaml` claim that the HAT also owns GPIO4 is not supported by the live overlay | `ultrasonic_driver.py:80-84`, `config/limits.yaml:17`, LIVE pinctrl | No: the ultrasonic pins are not in the registry and HaLow is absent from it | H-02 | Retire the legacy pin map. Place the four JSN-SR04T on a coprocessor or on pins that M-02 proves free. Confirm GPIO4 with the HAT fitted |
| C-02 | Starter relay GPIO5 is the HaLow RESET line | `config/tractor.yaml:31`; LIVE `halow-dw` overlay | No | H-02b | Reassign the starter input. Before any relay is attached, scope GPIO5 during a HaLow reset (M-03) |
| C-03 | Blade IN2 GPIO25 is the HaLow IRQ line | `spec/hardware.yaml:140`; LIVE `gpioinfo` | No | H-02c | Mower platform only and the IBT-4 driver is a stub; reassign before that driver is ever completed |
| C-04 | GPIO12 is both the ToF-R interrupt and BNO085 TXD4 | `spec/hardware.yaml:142,144` | No: rxd/txd rows are skipped by design (`check_hardware_pin_conflicts.py:57-59`) | H-03 | Choose one: BNO085 on UART4 via `uart4-pi5`, or on I2C. Drop the ToF interrupt (no code reads it) |
| C-05 | Header I2C1 (GPIO2/3) is disabled and GPIO3 is HaLow WAKE, yet the PCA9685, INA3221, ToF, BME280 and OLED are all configured on bus 1; the PR #18 plan routes the PCA9685 to pins 3 and 5 | `config/tractor.yaml:18`, LIVE `config.txt:23` | No | H-17, H-25 | Decide the bus route (B.1.4); candidate pins TBD |
| C-06 | The Pi 5 needs `uart4-pi5` (GPIO12/13); the docs say `uart4` (BCM2711-only, GPIO8-11) | N-13 | n/a | H-03 | Correct the docs; enable the overlay only after C-04 is decided |
| C-07 | `rpi4_alt_pin: 21 / 24` are physical pins but are read as BCM GPIO 24 / 21 | `spec/hardware.yaml:143-144`, `docs/hardware-integration.md:68` | No | H-03 | Documentation fix |
| C-08 | ToF interrupts GPIO6/12 are reserved but no code reads them | `vl53l0x_driver.py` | n/a | H-03 | Release the pins in the registry unless an interrupt path is built |
| C-09 | `hw_selftest.ultrasonic_probe()` claims and drives GPIO4/27/11 regardless of `ultrasonic_enabled`, with chip 4 hard-coded | N-18 | No | H-02 | Gate on the flag, discover the chip, never drive pins in a probe |
| C-10 | Legacy guide: pin 17 is 3V3, not 5 V | `docs/ultrasonic-wiring-guide.md` | No (prose) | H-01 | **Corrected in this PR** |
| C-11 | The HaLow HAT may also occupy GPIO2, 4, 6, 14, 15, 16, 21 (reported, UNVERIFIED) | `~/halow` on the Pi (via the checker branch) | n/a | H-17, H-25 | Measure with the HAT fitted (M-02) |
| C-12 | PCA9685 at 0x41 sits in the INA3221 strap family 0x40-0x43 [DS]; a re-strap of the INA3221 would collide. The PCA9685 all-call 0x70 is undocumented | `config/tractor.yaml:17`, `spec/hardware.yaml:146-151` | Partly (0x40 only) | H-17 | Confirm both straps (M-20, M-16); register 0x41 and 0x70 |
| C-13 | The header GPIO chip is `gpiochip0` on the deployed kernel (chips 10-14 also exist); `hw_selftest.py` assumes `gpiochip4`; the drivers probe `[4, 0]` | LIVE `/dev/gpiochip*`, `hw_selftest.py:202,214`, `drivers/base.py:14-35` | n/a | H-17 | Probe, do not hard-code |
| C-14 | No GPIO is allocated for PCA9685 `OE`, for the watchdog heartbeat, or for an e-stop state input | A.5 | n/a | H-04, H-08 | Allocate in the Toro profile (B.1.4) |

What the checker saw at baseline (what-if probes by the audit, nothing written): moving the starter to GPIO17 or
GPIO3 reported no conflict; moving it to GPIO12 reported a conflict only because the ToF interrupt is a `gpio:`
row; changing the PCA9685 to 0x29 or 0x40 reported conflicts. The checker also does not validate duplicate physical
pins, `pin` against `gpio` consistency, or any address from `config/hardware.yaml`. A parallel branch
(`audit/checker`) extends it (reviewed conflicts allow-listed in `spec/known_conflicts.yaml`, HAT-occupancy
warnings); it is **not** part of this baseline.

#### B.1.4 Candidate allocation for the Toro build (not approved)

Demand for the Toro build with four JSN-SR04T and the safety I/O the roadmap requires:

| Signal | Count | Allocated today | Note |
|---|---|---|---|
| I2C bus (PCA9685, INA3221, ToF, BME280; BNO085 if I2C) | 2 wires | GPIO2/3, **blocked** | needs a route (decision D-1) |
| Starter relay control | 1 out | GPIO5, **conflict** | reassign |
| PTO relay control | 1 out | GPIO26 | candidate; polarity and boot state unmeasured |
| PCA9685 `OE` control | 1 out | none | a "stop commanding" path only (H-04) |
| Heartbeat to the hardware watchdog | 1 out | none ("any free GPIO" in `hardware.md`) | |
| E-stop state input (monitoring only; the e-stop itself must not depend on it) | 1 in | none | |
| JSN-SR04T x4, trigger/echo mode | 8 (4 TRIG + 4 ECHO) | none | mode UNVERIFIED (M-40); UART mode needs 4 serial ports or a multiplexer |
| ToF XSHUT | 2 out | GPIO22/23 | in use |
| ToF interrupt | 0 to 2 in | GPIO6/12 | no reader; drop |
| BNO085 | UART TX/RX (2) or I2C | GPIO12/13 | transport undecided (D-4) |
| GNSS UART | 2 | GPIO14/15 | in use |

Supply: header GPIOs that are unclaimed on the live Pi **and** not flagged by the vendor-schematic warnings are
**GPIO12, 13, 17, 18, 19, 20, 24, 26, 27** (nine). GPIO26 is the PTO relay; GPIO12/13 carry the BNO085 and
ToF-interrupt claims; GPIO17/27 carry the legacy ultrasonic claims; GPIO24 is the mower's blade IN1. With the BNO085
on I2C and the ToF interrupts dropped, **eight** pins are free of claims, against a new demand of 12 (starter, `OE`,
heartbeat, e-stop input, eight ultrasonic lines) or 14 with an I2C route. Every one of them still needs the
continuity check in M-02. **Conclusion: putting four trigger/echo ultrasonics and the safety I/O on the 40-pin
header does not fit.**

| Option | Description | For | Against | State |
|---|---|---|---|---|
| A | Everything on the Pi header | no extra hardware | does not fit (above); ultrasonic timing in a non-real-time process; the HaLow HAT may block more pins (C-11); I2C needs a re-route that touches the HaLow WAKE strap | not viable as stated |
| B | A USB safety and sensor **coprocessor** (for example an RP2040 board, reusing the CircuitPython and USB-CDC pattern the repository already has for the RoboHAT) owns the ultrasonic timing, the PCA9685 I2C bus, `OE`, the servo-power relay driver, the watchdog heartbeat and the e-stop state input, and generates the servo PWM itself so it can substitute a neutral pulse when the Pi's heartbeat stops | frees the header; hard real time for echoes; I2C isolated from the HaLow HAT; neutral-on-fault becomes possible (D.3) | new firmware, protocol and tests; its own failure modes (the RoboHAT firmware's hardware watchdog is 8 s, `code.py:100`; reset-state pulls must hold relays and `OE` safe); it must **not** be the only stop path | design option, nothing exists (A.2) |
| C | Hybrid: ultrasonics and `OE` on a coprocessor, relays and the GNSS UART on the header | smaller firmware | I2C route and relay pins still unresolved | not evaluated |

Decisions the owner must make before any wiring (none is made here): **D-1** I2C route for the PCA9685; **D-2**
ultrasonic architecture (A, B or C); **D-3** which device owns the relays, `OE` and the watchdog heartbeat; **D-4**
BNO085 transport; **D-5** HaLow HAT occupancy (inspect first, M-01 and M-02).

### B.2 Connector-to-connector wiring proposal (signal level)

> **Proposal only. Not as-built.** It names signals and connectors; it does not give wire gauges, fuse ratings,
> connector part numbers or pin numbers that the evidence does not support. Connector labels `Jn` are local to this
> document. Anything marked `TBD` must be resolved by the measurements in B.7 and by an independent electrical
> review before a wire is cut. The PR #18 plan's rule applies: Pi GPIO must never directly power a starter
> solenoid or a PTO clutch, and no design here bypasses an OEM interlock.

```mermaid
flowchart LR
  subgraph POWER["Power (all ratings TBD)"]
    BAT["12 V battery"] --> FUSE["Fuse block"]
    FUSE --> BUCK["5 V buck (part TBD)"]
    FUSE --> SVRELAY["Servo-power relay"]
    FUSE --> RLYPWR["Relay-module supply"]
  end
  subgraph PI["Raspberry Pi 5"]
    J5["J5 USB-C or 5 V pins"]
    J1["J1 40-pin header with HaLow HAT stacked"]
    J2["J2 USB ports"]
    J3["J3 CSI connector"]
    J4["J4 PCIe FFC"]
  end
  BUCK -->|"5 V"| J5
  J1 --- HALOW["HaLow HAT: SPI0, GPIO3, 5, 7, 25"]
  J1 -->|"UART0 GPIO14, 15"| GNSS["LC29H-DA rover"]
  J1 -->|"I2C route TBD (C-05)"| PCA["PCA9685 breakout"]
  J1 -->|"I2C and XSHUT GPIO22, 23"| TOF["2x VL53L0X"]
  J1 -->|"UART4 or I2C (D-4)"| IMU["BNO085"]
  J1 -->|"GPIO per B.1.4"| RLY["Opto relay module"]
  J1 -->|"GPIO or coprocessor (D-2)"| JSN["4x JSN-SR04T"]
  J2 --- ELP["ELP stereo camera"]
  J2 -.->|"optional"| LIDAR["2D LiDAR"]
  J2 -.->|"option B"| MCU["Safety and sensor coprocessor"]
  J3 --- CAM["Pi Camera Module 2"]
  J4 --- HAILO["Hailo module on M.2 adapter"]
  PCA -->|"PWM ch0, ch2"| SERVOS["2x lever servos"]
  PCA -->|"PWM ch1"| THR["Throttle actuator"]
  SVRELAY -->|"servo supply, direct"| SERVOS
  SVRELAY --> THR
  RLY --> STARTER["Starter circuit (OEM)"]
  RLY --> PTO["PTO clutch circuit (OEM)"]
  RLYPWR --> RLY
  ESTOP["Hardwired E-stop"] -.->|"removes coil supply"| SVRELAY
  ESTOP -.-> RLY
  WDOG["Hardware watchdog (design TBD)"] -.-> SVRELAY
```

| Net | From | To | Signal and level | Status |
|---|---|---|---|---|
| PWR-12V | battery + via fuse block | each branch (Pi, servo, relay) | 12 V nominal | ratings, gauges, fuse curves **TBD** (B.3) |
| PWR-PI | 5 V buck output | Pi USB-C (J5) or the header 5 V pins | 5 V, current TBD | supply part TBD (DROK vs Pololu); whether the Pi negotiates its USB current limit from a non-PD supply [DS] TBD |
| PCA-I2C | Pi SDA, SCL (route TBD) | PCA9685 SDA, SCL | 3.3 V open-drain, pull-ups TBD | **blocked** on the deployed Pi (C-05) |
| PCA-GND | Pi GND | PCA9685 GND | logic reference | star-point design TBD (R-P07) |
| PCA-VCC | 3V3 or 5 V (pin TBD) | PCA9685 logic supply | breakout-dependent | TBD |
| PCA-OE | GPIO, coprocessor or hardware watchdog | PCA9685 `OE` | active-low: HIGH disables outputs and gives no pulse [DS] | unallocated (C-14); pull state at power-up TBD (M-21) |
| SERVO-SIG | PCA9685 ch0 (left), ch1 (throttle), ch2 (right) | servo signal inputs | 50 Hz, endpoints illustrative (`config/tractor.yaml:23-27`); high level equals the PCA9685 logic supply [DS] | endpoints uncalibrated; servo input threshold TBD |
| SERVO-PWR | fuse, then servo-power relay contacts, then each servo V+ **directly** (not via PCA9685 V+ unless that path is proven for the stall current) | servo supply pins | 12 V class | owner-reported 10-12.6 V, 8 A or more stall; rail, relay rating, gauge **TBD** |
| RLY-IN-START | a Pi or coprocessor output (not GPIO5) | opto-module IN (starter channel) | module polarity unknown; software assumes active-high | TBD (M-17) |
| RLY-IN-PTO | GPIO26 (candidate) or coprocessor | opto-module IN (PTO channel) | as above | TBD (M-17) |
| RLY-COIL | opto-module output | Bosch-style 12 V relay coil | 12 V, flyback TBD | TBD |
| STARTER | relay contacts | OEM starter circuit | OEM; the PARK interlock stays | circuit not traced; supervised manual start remains (H-06) |
| PTO | relay contacts | OEM PTO clutch circuit | momentary vs latching **unknown** | **do not wire** until traced (H-21, M-30) |
| ESTOP-CHAIN | hardwired E-stop NC contact | in series with the servo-power relay and PTO relay coil supplies (proposal: loss of coil supply opens both) | hardware only | **needs design review**; whether it should also act on the engine or park brake is a safety-engineering decision (D.1) |
| WDOG-HB | Pi or coprocessor heartbeat output | watchdog trigger input (74HC123 per `hardware.md`) | pulse train; window TBD | circuit not designed in the repository |
| TOF | I2C (SDA, SCL), XSHUT GPIO22/23, interrupt not used | two VL53L0X boards | 3.3 V logic | bus blocked; XSHUT pulls TBD (M-41) |
| IMU | UART4 GPIO12/13 (needs `uart4-pi5`) or I2C 0x4A/0x4B | BNO085 | 3.3 V; mode straps TBD | transport undecided; driver is a stub |
| GNSS | UART0 GPIO14/15 | LC29H-DA TX/RX | 3.3 V UART | carrier-board pinout, supply, antenna bias TBD (M-50) |
| JSN x4 | per D-2 | four JSN-SR04T | supply, mode and ECHO level TBD (M-40) | **no wiring may be drawn** before M-40 |
| HALOW | J1 stack | HaLow HAT | SPI0 CS0, IRQ, RESET, WAKE, BUSY | HAT revision unverified (M-53) |
| CSI | J3 | Pi Camera Module 2 | 22-pin FFC; the 15-pin camera cable needs an adapter [DS] | connector in use and cable length TBD |
| STEREO | J2 USB | ELP-USB960P2CAM-V90 | UVC; 2560x960 combined | bus speed and power TBD |
| HAILO | J4 PCIe FFC | Hailo module on its M.2 adapter | PCIe Gen3 x1 | adapter identity and clearance beside the HaLow HAT TBD (H-17) |

House side (candidate topology from the owner's 2026-10-24 requirements; nothing here is in the repository):

```mermaid
flowchart LR
  ANT["GNSS antenna"] --> BS["LC29H-BS base"]
  BS -->|"UART or USB (TBD)"| PZ["Pi Zero base host (TBD)"]
  PZ -->|"network (TBD)"| OPI["Orange Pi 4 Pro: HaLow AP and gateway"]
  THOR["NVIDIA Thor"] -->|"Ethernet"| OPI
  OPI -.->|"Wi-Fi HaLow 802.11ah"| RADIO["Mower HaLow HAT"]
  RADIO --> PI["Mower Pi 5"]
  PI -->|"RTCM to the rover UART"| ROVER["LC29H-DA rover"]
```

Only the Orange Pi 4 Pro AP (192.168.1.140, HT-HC01P on SPI3) and the Thor's route to it
(`192.168.50.0/24 via 192.168.1.140`) are documented (`docs/autonomy-toolchain-matrix.md:47-69`). The Pi Zero, the
LC29H-BS and the correction path are owner requirements only. P3a requires a failure analysis of each intermediate
node (base, Pi Zero, Orange Pi, HaLow radio, Thor) and measured correction age and recovery time.

### B.3 Power distribution and fuse requirements

The repository specifies **no** fuse rating, wire gauge or stall-current budget for any branch (A.5). This section
lists what the design must satisfy and what must be measured. It gives no ratings: they follow from M-10 to M-17.

```mermaid
flowchart TB
  BAT["12 V starting battery (OEM)"] --> MAIN["Main disconnect and fuse (rating TBD)"]
  MAIN --> F1["Fuse: Pi branch (TBD)"] --> BUCK["5 V buck (part TBD)"] --> PI["Pi 5 + Hailo + HaLow + USB devices"]
  MAIN --> F2["Fuse: servo branch (TBD, from measured stall)"] --> SVREL["Servo-power relay (E-stop and watchdog controlled)"] --> SV["2x lever servos + throttle servo"]
  MAIN --> F3["Fuse: relay and control branch (TBD)"] --> RLYMOD["Opto relay module and 12 V relays"]
  RLYMOD --> STARTER["Starter solenoid (OEM circuit)"]
  RLYMOD --> PTO["PTO clutch (OEM circuit)"]
  ESTOP["Hardwired E-stop"] -.-> SVREL
  ESTOP -.-> RLYMOD
```

| ID | Requirement | Why | Measurement |
|---|---|---|---|
| R-P01 | Every branch off the battery has its own overcurrent protection, rated to protect the wire of that branch (fuse rating at or below the wire's ampacity under its installed conditions) | `hardware.md` lists "assorted fuses" only; `docs/hardware-overview.md` calls for "three separately-fused taps" without values | M-13 |
| R-P02 | The servo branch is sized from the **measured** stall current of the purchased servos at the real supply, with both lever servos stalled at once (the e-stop worst case), not from the owner-reported "8 A or more" | H-20 | M-10, M-11 |
| R-P03 | The fuse type and curve ride through normal servo inrush and brief stalls, yet open before the wire is damaged | H-20 | M-13 |
| R-P04 | Voltage drop at the servo terminals during stall is measured and the gauge chosen so servo V+ stays above the servo's minimum and the Pi and PCA9685 logic are unaffected | H-05, H-20 | M-11, M-15 |
| R-P05 | The Pi's 5 V rail holds under worst-case load **and** battery sag during an engine crank (the Pi and the starter share the battery; `starter.pulse_ms = 800`, `config/tractor.yaml:31`). Undervoltage is detected (`vcgencmd get_throttled`) and treated as a fault. A Pi brown-out during mowing is a safety event (`hardware.md` Power warning) | H-17 | M-12, M-14 |
| R-P06 | Sensors that need 5 V (JSN-SR04T supply TBD) do not share a rail that sags with the Pi, and their current is counted in the Pi supply budget | H-19 | M-14, M-40 |
| R-P07 | A ground topology is defined: battery negative to a star point; Pi, PCA9685, relay module and sensors referenced to it; servo return current never flows through logic ground wires; chassis bonding and shield/drain handling decided. No document defines this today | H-20 | M-15 |
| R-P08 | Inductive loads (relay coils, starter solenoid, PTO clutch) are suppressed at the load, and the kick is measured to confirm it does not disturb logic rails or I2C | H-21 | M-17 |
| R-P09 | The hardwired E-stop de-energises the servo-power relay and the PTO relay with the Pi powered off and with the Pi hung (Part D.1) | H-07, H-08 | M-24 |
| R-P10 | At power-up, during Pi boot and with GPIOs floating, every relay output defaults OFF; if the modules are active-low this needs a pull-up or an inverting buffer | H-21 | M-17 |
| R-P11 | Input protection against reverse polarity and charging-system transients on the 12 V input is decided and documented | none | M-12 |
| R-P12 | The Toro's battery and charging output is compared with the sustained servo, Pi and relay load | `docs/hardware-overview.md` Power caution | M-12 |
| R-P13 | The mower variant's 12.8 V LiFePO4, solar and INA3221 monitoring (`spec/hardware.yaml:47-70`) stay separate from the Toro power design | H-11 | n/a |

### B.4 I2C, UART, USB, CSI, PCIe inventory (configuration vs the deployed Pi)

**I2C.** The header bus does not exist on the deployed Pi; `i2cdetect` was not run. Addresses below are what the
configuration expects.

| Address | Device | Configured in | LIVE |
|---|---|---|---|
| 0x10 | imx219 (Pi Camera Module 2) | n/a | present on `i2c-11` (RP1 CSI bus), **not the header** |
| 0x29 | VL53L0X left | `spec/hardware.yaml:22,148`, `config/hardware.yaml:22` | header bus absent |
| 0x30 | VL53L0X right (reprogrammed at runtime, `vl53l0x_driver.py:297`) | `spec/hardware.yaml:23,149` | header bus absent |
| 0x3C | SSD1306 OLED (no driver, A.1) | `spec/hardware.yaml:27,150` | header bus absent |
| 0x40 | INA3221 | `spec/hardware.yaml:60,151`, `ina3221_driver.py:26` | header bus absent |
| 0x41 | PCA9685 (A0 bridged) | `config/tractor.yaml:17` | header bus absent; not in the registry |
| 0x70 | PCA9685 all-call (power-on default [DS]; the driver clears ALLCALL in `MODE1`) | not documented | TBD |
| 0x4A, 0x4B | BNO085 if on I2C | `sensor_manager.py:186-188` | header bus absent |
| 0x76 | BME280 | `spec/hardware.yaml:25,152` | header bus absent |

**UART.**

| UART | Configuration says | LIVE |
|---|---|---|
| UART0, GPIO14/15, `/dev/ttyAMA0` | GNSS at 115200 (`spec/hardware.yaml:154`, `config/hardware.yaml:9-11`) | device present, pins alt4 `uart0`; the receiver behind it is unverified |
| "UART1" RoboHAT 38400 | `spec/hardware.yaml:155` | no `ttyAMA1`; the code uses USB CDC `/dev/ttyACM0` at 115200 (`robohat_service.py:86`, `docs/servo-hat-pins.md` section 2): the spec entry is wrong |
| UART4, GPIO12/13, BNO085 at 3 Mbaud | `spec/hardware.yaml:156` | **not enabled**: no `uart4-pi5`, no `/dev/ttyAMA4` |
| `/dev/ttyAMA10` | not documented | present (the Pi 5 debug UART [DS]) |
| `/dev/ttyUSB0` CP2102N `10c4:ea60` | not documented | present; identity of what it connects to unknown (TBD, M-50) |

**USB.**

| Device | Configuration says | LIVE |
|---|---|---|
| GNSS receiver | u-blox ZED-F9P via SparkFun GPS-RTK-SMA (`spec/hardware.yaml:158-159`) | not seen (`/dev/ttyACM*` absent) |
| RoboHAT RP2040 | USB CDC | not seen |
| Google Coral USB accelerator | `spec/hardware.yaml:160` | not seen |
| High-gain Wi-Fi adapter | `spec/hardware.yaml:161` | not seen |
| ELP stereo camera | README, matrix | `32e4:9750 "3D USB Camera"` on Bus 001 (UVC) |
| CP2102N UART bridge | none | `10c4:ea60` on Bus 003 |
| USB-C OTG controller | not documented | `dwc2` overlay loaded; a root hub on Bus 005 |

Absence of the sensors at this site is not proof of a design error: they are detached.

**CSI.** The imx219 is bound through `rp1_cfe` (`/dev/video0` onward); the `cam0_reg` and `cam1_reg` regulators are
claimed, so at least one CSI port is initialised. Which of the two connectors carries the camera is TBD (M-01). The
camera is owned exclusively by `lawnberry-camera.service` (`docs/pi5-csi-camera-libcamera-status.md`).

**PCIe.** `0001:01:00.0 Hailo-8 AI Processor` with `/dev/hailo0` and `dtparam=pciex1_gen=3`; `0002:01:00.0` is the
RP1 south bridge. The adapter between the Pi's PCIe connector and the Hailo module, and its mechanical clearance
against the HaLow HAT on the 40-pin header, are unverified (H-17, H-25; M-01).

**SPI.** `spi0` carries the HaLow radio on CS0 only (`dtoverlay=spi0-1cs`). `spidev10.0` is the BCM2712 boot SPI,
not a header bus.

**Network and OS (LIVE).** `eth0` 192.168.1.171 and `tun0` 10.8.0.2 up; `halow0` and `wlan0` down at audit time;
kernel `6.12.21-morse+` with `device_tree=bcm2712-rpi-5-b-morse.dtb` and `kernel=kernel_2712-morse.img`. The Pi was
documented as Ubuntu 24.04.4 with kernel `6.8.0-raspi` on 2026-10-06 (`docs/autonomy-toolchain-matrix.md:14`).

**Power telemetry (LIVE, one idle sample).** `vcgencmd get_throttled = 0x0`, `EXT5V_V = 5.13 V`.

### B.5 Sensor mounting and coverage requirements

**Known geometry.** Deck cutting width 127 cm (50 in), machine mass 292 kg, maximum rated slope 15°
(`docs/tractor-platform.md`, "Machine data from the manual"). The target machine is the Toro 50 in TimeCutter MAX
MyRIDE, model 77502; the serial plate is to be verified (M-33). The owner's sensor set:

| Sensor | Count | Placement requirement | Source |
|---|---|---|---|
| JSN-SR04T | 4 | Two inner sensors point forward; two outer sensors angle outward on **adjustable** mounts (yaw and pitch) so the union of cones spans the 50-inch deck edges and the approach to trees; weather-sealed, vibration-resistant, clear of deck travel, chute and steering envelope | PR #18 plan, "Four-sensor front mounting geometry" |
| VL53L0X | 2 | Front-left and front-right, 25° cone each (`docs/hardware-overview.md`); sunlight and dust limits unmeasured | repository |
| Stereo camera | 1 | Elevated on the camera tower (`hardware.md` Enclosure: a USB camera is required for the tower because the CSI camera cannot make the run); baseline and calibration unknown | repository, PR #18 |
| Pi Camera Module 2 | 1 | Fixed forward scene | repository |
| 2D LiDAR | 0 or 1 | Optional; model unselected; scan-plane height and blind spots TBD | PR #18 plan |

**Existing printable mounts (`hardware/mounts/`).** The repository holds OpenSCAD sources and STLs for the camera
tower, the stereo and Pi-camera housings, an E-stop bracket, the enclosure, the lap-bar clamps and push-rods, and a
generic drill-to-fit `sensor_carrier` with a tilt foot (ToF at the front corners aimed forward; the IMU rigid near the
middle of the machine and away from servos and high-current wiring: `hardware/mounts/INSTALL.md:378`). **There is no
JSN-SR04T mount:** the module's waterproof probe has no geometry in the repository and no yaw- and pitch-adjustable
fixture exists for the outer pair. The E-stop bracket is a fit fixture whose rail dimensions are unverified
(`hardware/mounts/README.md:44`).

**Coverage condition.** Approximate each ultrasonic beam as a cone about its axis. For sensor `i` at lateral offset
`x_i` (m, positive right), yaw `psi_i` (positive outward right) and half-beam angle `a_i`, the lateral interval it
covers at forward distance `d` is `[x_i + d*tan(psi_i - a_i), x_i + d*tan(psi_i + a_i)]`. The requirement is that for
every `d` between the largest blind zone and the required detection distance, the union of those intervals (from
sensors whose range at that point is above their blind zone and within their reliable range) contains
`[-(W/2 + m), +(W/2 + m)]`, where `W = 1.27 m` and `m` is a lateral margin set by the safety review. This must be
verified on the real mounts by measuring the beam pattern, not by trusting nominal angles.

**Detection distance requirement.** With ground speed `v`, the shortest range at which an obstacle is reliably
detected must satisfy
`d_detect_min >= v*(t_sense + t_compute + t_actuate) + d_coast(v, slope) + d_margin`;
where it does not, the autonomous speed is capped until it does. Ultrasonic cones are not an occupancy map (H-26),
and the stop path must not depend on a single unverified sensor.

| Parameter | Known value | Status |
|---|---|---|
| Deck width `W` | 1.27 m | known |
| Lateral margin `m` | none | TBD (safety review) |
| Sensor positions `x_i`, yaw, pitch | none; no mount geometry is defined and none may be invented | TBD (mount design, M-01) |
| Beam half-angle `a_i` | none | TBD (M-40) |
| Blind zone | about 10 inches (about 25 cm) was discussed by the owner | **unmeasured** (M-40); do not rely on ultrasonics for close-contact safety (PR #18 plan) |
| Reliable range on grass, people, children, pets, low objects | none | TBD (M-40) |
| Sensor update period and filter delay `t_sense` | the legacy HC-SR04 driver assumes a 30 ms echo timeout and about 60 ms per sensor cycle (`ultrasonic_driver.py:76,91`); not a JSN-SR04T figure | TBD (M-40) |
| Compute latency `t_compute` | none | TBD (end-to-end measurement) |
| Actuation latency `t_actuate` (command to neutral, PTO drop) | the constitution budgets 100 ms to issue and 500 ms to settle (`docs/tractor-acceptance-criteria.md` section 2) | **unmeasured** (M-24) |
| Coast distance `d_coast(v, slope)` | none; the field protocol warns the machine creeps after neutral (hydrostatic self-braking only) | TBD (M-43) |
| Maximum autonomous speed `v` | not in the repository | TBD (owner and Toro manual) |

**Hazard classes the coverage tests must include** (PR #18 roadmap P2): children and pets, low objects, negative
obstacles (drop-offs), deck overhang past the sensor line, tall grass (false positives), rain and spray, direct
sunlight (ToF), and obstacles above or below a 2D LiDAR plane.

**Visual package (H-24): not produced here.** Annotated photographs of the actual Pi / HaLow / Hailo stack, a labelled
Toro front view with the four adjustable mounts overlaid, a GPIO riser map and a connector-by-connector as-built
schedule need real photographs and measurements; fabricating mounting holes or presenting a rendering as a photograph
is prohibited by the requirement. The photographs and dimensions needed are M-01, M-33 and the mount survey in P0a
(Part E).

### B.6 Hardware compatibility matrix

One row per component the owner listed. "Documented" means a repository or PR #18 statement; `TBD` means nothing in
the evidence settles it. No electrical rating is asserted unless a source is cited.

| Component | Role and interface | Software support at baseline | Conflicts and risks | Verify before use |
|---|---|---|---|---|
| **Raspberry Pi 5** | Onboard compute and (target) local safety supervisor. 40-pin header (3.3 V GPIO, not 5 V tolerant [DS]), PCIe FFC, CSI/DSI, USB-A, USB-C power. RAM is documented as 8 GB (`docs/hardware-feature-matrix.md`, `docs/autonomy-toolchain-matrix.md:14`) and 16 GB (`docs/hardware-overview.md`, `.github/copilot-instructions.md`) | Backend runs live (`lawnberry-backend` active) | Header carries the HaLow HAT; I2C1 disabled (B.1); OS and kernel drift (B.4) | RAM size, image, 5 V under load (M-14), thermals in the printed enclosure (M-55). H-17 |
| **Hailo AI module** | Onboard detector and monocular depth inference. PCIe Gen3 x1 (LIVE `0001:01:00.0`, `/dev/hailo0`); the M.2 adapter is unknown | IMPLEMENTED (A.1); HailoRT 4.24.0; HEFs compiled for `hailo8l` (`scripts/hailo_compile.sh`) | **Identity:** `lspci` says "Hailo-8" (shared PCI ID), `hailortcli fw-control identify` said `HAILO8L`, part `HM21LB1C2LAE` on 2026-10-06; `spec/hardware.yaml:37` says "Hailo-8 AI Hat". A wrong HEF target is rejected (`HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE`). Clearance beside the HaLow HAT (H-25); power draw unmeasured | `hailortcli fw-control identify` and a label photo (M-54); adapter identity (M-01). H-17 |
| **NVIDIA Thor (AGX Thor)** | House compute: coverage strategy, optional heavy inference, training; **target** primary navigation and decision computer (H-23). Ethernet to the Orange Pi; not on the mower. Aarch64 Ubuntu 24.04.5, 131.86 GB, driver 595.78 (`docs/autonomy-toolchain-matrix.md:16`) | Strategist IMPLEMENTED in SIM; link service unwired; detections-to-uplink MISSING; TensorRT FP16 verified; Isaac Sim launches but does not render (`:78`) | Link measured at one location only (4.79 Mbit/s); no message authentication; a Thor outage must leave the mower locally safe (D.6) | Far-yard throughput, recovery, plan-age policy (M-52). H-23, H-28 |
| **Heltec HaLow HAT (MM6108)** | Pi to AP/Thor link. SPI0 CS0, IRQ GPIO25, RESET GPIO5, WAKE GPIO3, BUSY GPIO7 (LIVE comments); kernel modules `morse`, `dot11ah`, interface `halow0` | Link-watch units versioned (A.8); bring-up and failover units are not | Occupies GPIO3/5/7/8-11/25; the HAT's identity is inconsistent (Heltec HT-HC01P V2 in `config.txt`, Seeed WM6108 in `halow-bringup.service` and the matrix); a stopped backend once took the radio down and the AP radio vanished after a TX-cap change (`docs/autonomy-toolchain-matrix.md:49,76`); `halow0` down at audit time | HAT label (M-53), pin occupancy (M-02), power-cycle recovery (M-52). H-16, H-17, H-23 |
| **Quectel LC29H-DA (rover)** | RTK rover. UART0 at 115200 per `config/hardware.yaml:9-11`; carrier board, supply, antenna bias, PPS: TBD | NMEA driver; NTRIP forwarder writes RTCM to a serial device; no `PAIR`/`PQTM` configuration; GGA quality is the fix metric (`docs/autonomy-toolchain-matrix.md:27`) | `spec/hardware.yaml`, `hardware.md` and `.github/copilot-instructions.md` still say ZED-F9P (H-09); no receiver verified live | Photograph and label the fitted module (M-50); where RTCM is injected; antenna and sky view. H-09 |
| **Quectel LC29H-BS (base)** | House RTK base; connection to the Pi Zero (UART or USB) TBD | **MISSING** | Survey position error becomes rover absolute error; antenna placement; transport and correction age unknown | Survey-in procedure, RTCM message set, transport, correction age (M-51). H-09, H-22 |
| **Pi Zero base host and Orange Pi 4 Pro gateway** | Base host and HaLow AP/gateway. Orange Pi: HT-HC01P on SPI3, 192.168.1.140, LAN to the Thor (documented, `docs/autonomy-toolchain-matrix.md:47-69`). Pi Zero: nothing documented | AP-side units are not in the repository; no base-host code | Each node is a single point of failure for corrections; the AP radio vanished on 2026-10-08 after its TX cap was removed (documented) | Actual boards, power, service ownership and behaviour on each node failing (M-52). H-22 |
| **BNO085 IMU** | Heading and tilt. UART4 at 3 Mbaud on GPIO12/13 (`spec/hardware.yaml:29,156`) **or** I2C 0x4A/0x4B (the only presence probe, `sensor_manager.py:188`); straps TBD | **STUBBED** | GPIO12 double booking (C-04); `uart4-pi5` absent; the tilt cutoff cannot trip on real hardware (N-06) | Transport, mode straps, reset and interrupt wiring, mounting frame and axes, tilt-to-blade-off latency (M-42). H-03 |
| **2x VL53L0X ToF** | Near obstacles, front-left and front-right, 25° cone, 30-2000 mm (`docs/hardware-overview.md`). I2C 0x29/0x30, XSHUT GPIO22/23; supply and level-shifting TBD | Driver PARTIALLY IMPLEMENTED; interrupts unused; header bus absent | Units (mm vs the metres in `limits.yaml`) unchecked; sunlight limit unmeasured | M-41. H-03, H-17 |
| **4x JSN-SR04T** | Forward and deck-edge obstacles. Supply, trigger/echo vs serial mode, ECHO level, current, blind zone: **all unverified** | **MISSING** | Legacy pins collide with HaLow (C-01); legacy divider unsafe (H-18); blind zone about 10 in discussed; cone range is not a map | M-40, M-02, mount survey. H-10, H-19 |
| **2D LiDAR (optional)** | Planar obstacle scan. Model unselected | **MISSING** | Cannot see above or below its plane; weather and sunlight unknown; needs power and a driver | Model, interface, scan plane, minimum range (M-45). H-15, H-26 |
| **Raspberry Pi Camera Module 2 (IMX219)** | Fixed forward scene. CSI; the 15-pin camera cable needs an adapter for the Pi 5's connector [DS] | IMPLEMENTED (live); libcamera built into a private prefix (`docs/pi5-csi-camera-libcamera-status.md`) | Owned exclusively by `lawnberry-camera.service`; the CSI cable cannot reach an elevated tower (`hardware.md` Enclosure) | Connector in use and cable length (M-01). H-17 |
| **ELP-USB960P2CAM-V90 stereo** | Elevated stereo and possible VR. UVC `32e4:9750`; 2560x960 combined at 30 fps (`stereo_camera_driver.py:48-50`). Documented as USB 2.0 (README): 2560x960 at 30 fps is about 74 MB/s at one byte per pixel, above USB 2.0's 60 MB/s signalling rate [DS], so the camera must deliver a compressed format or fewer frames | Capture only (A.1); no depth, no streaming | Baseline, lens and calibration unknown; HaLow cannot carry the stream (C.5) | Pixel format, fps, bus speed, baseline and calibration (M-44). H-16, H-26 |
| **PCA9685 breakout** | 16-channel 50 Hz PWM, I2C 0x41 (A0 bridged), `OE` active-low [DS], `V+` is a pass-through to all servo headers. Documented part: generic board (`hardware.md`, listing `B07RMTN4NZ`) | PARTIALLY IMPLEMENTED (A.1) | No header I2C on the Pi (C-05); `OE` unallocated (C-14); power-up output state TBD; the `V+` path's current rating is unknown | Straps, `OE` pull and levels, no-pulse servo behaviour (M-20, M-21). H-04, H-08 |
| **Two lap-bar servos** | Left and right drive levers over PCA9685 ch0 and ch2. `hardware.md`: DSSERVO/ANNIMOS RDS51150SG, 12 V, 150 kg, 165 kg·cm at 12 V, 0.21 s per 60° (listing `B0C69W2QP7`). Owner: two matching 150 kg·cm-class servos, 10-12.6 V, possibly 8 A or more stall | Actuator calibration illustrative (`config/tractor.yaml:23-27`) | **24 V instructions in older documents are superseded.** Whether the purchased units are the listed ones is unverified; stall current, neutral pulse and no-pulse behaviour unknown; lever spring return unverified | Label and rating (M-10), stall current and voltage drop (M-11), no-pulse and neutral behaviour (M-21, M-22, M-25). H-05, H-08, H-20 |
| **Throttle actuator** | Third PCA9685 channel (ch1); `hardware.md` makes the third servo optional because the throttle is set once; software default `tractor_engine_throttle` 0.75 | Model **not identified** | Failure state of the throttle on loss of pulse unknown; shares the supply question | Model, rating, travel, failure state (M-10, M-25). H-20 |
| **Toro TimeCutter MAX MyRIDE 50 in, model 77502, OEM controls** | Motion levers move fore and aft and swing outboard to PARK (parking brake); seat/operator-presence switch shuts the engine off within 1 s when the levers are out of PARK and the operator rises; OEM Reverse Operation System; manual choke; PTO switch (owner: momentary); 12 V starting battery; 292 kg, 15° maximum slope (`docs/tractor-platform.md`) | Software interlocks only; nothing senses the seat, PARK or ROS | Autonomous start is physically impossible (H-06); unoccupied travel needs a professionally reviewed safety architecture (H-07); PTO behaviour unknown (H-21); lever self-centring unknown (H-08) | Serial plate and manual revision (M-33), PTO trace (M-30), 1 s cutoff (M-31), ROS (M-32), spring return (M-22), charging output (M-12). H-06, H-07, H-21 |

### B.7 Physical measurements still required

None of these exists in the repository. They are listed so the owner can plan bench time; acceptance values are
**not** set here: they are agreed with the independent safety reviewer before measuring. All actuator tests are
engine-off, PTO/blade disconnected, machine immobilised and wheels clear, with a safety observer, unless a later
commissioning step explicitly says otherwise.

| ID | Measurement | Method | Closes |
|---|---|---|---|
| M-01 | Photograph the full Pi 5 stack (HaLow HAT, Hailo adapter, ribbons, CSI connector in use) from every side; record board revisions and silkscreen | photographs with a scale reference | H-17, H-24, H-25 |
| M-02 | With the HaLow HAT fitted, measure level and continuity of every header GPIO against the HAT schematic; record which pins are tied to the HAT | meter, scope | C-01, C-11 |
| M-03 | Scope GPIO5 and GPIO25 during a HaLow reset and driver reload, with **no** relay attached | scope | C-02, C-03 |
| M-04 | List the pin groups the Pi 5 I2C and UART overlays support on this kernel (read-only overlay help) and which are free of the HAT | `dtoverlay -h <name>` | C-05, D-1 |
| M-10 | Servo label, part number, rated voltage; no-load and stall current at the real supply on the real linkage, both servos stalled together | clamp meter or shunt, supply with current limit | H-05, H-20 |
| M-11 | Voltage at the servo V+ pins during stall; voltage drop along the harness | meter, scope | R-P04 |
| M-12 | Battery open-circuit and under-load voltage; battery sag at the Pi supply input during a starter crank; the Toro's charging output against a representative load | scope on the supply input, meter | R-P05, R-P11, R-P12 |
| M-13 | Wire gauge and fuse rating per branch; fuse curve against the measured stall | inspection, datasheets | R-P01, R-P03 |
| M-14 | Pi 5 V rail under full load (Pi, Hailo, HaLow, camera, USB, sensors): dropout and buck temperature; `vcgencmd get_throttled`, `EXT5V_V` | meter, thermal probe | R-P05, R-P06 |
| M-15 | Ground topology: continuity between the Pi GND, PCA9685 GND, servo return, relay supply return and chassis; logic-ground noise during a servo stall | meter, scope | R-P07 |
| M-16 | INA3221 strap address and which shunt carries battery and solar current | meter, one-channel load test | N-12, C-12 |
| M-17 | Relay modules: trigger polarity, GPIO input current, state with the GPIO floating at boot, reset and Pi power-off, coil kick, contact ratings; starter solenoid current; PTO clutch coil current and inductive kick | meter, scope, datasheets | R-P08, R-P10, H-21 |
| M-20 | PCA9685 address straps A0-A5; confirm 0x41 and the all-call 0x70 once a bus exists | `i2cdetect` after the bus is decided | C-12 |
| M-21 | `OE` level at power-up, during Pi boot and with I2C lost; whether the board has an `OE` pull-down; the output with `OE` high; lever behaviour with servo power on and no pulse (hold, drift or limp); whether the last pulse persists after SDA/SCL loss | scope, observation | H-04, H-08 |
| M-22 | Lever return with servos unpowered, with and without the linkage fitted; time to neutral at the real supply | by hand, then video | H-08 (acceptance section 10) |
| M-23 | RoboHAT and MDDRC10 input states during an induced firmware watchdog reset (differential-drive platform only) | scope | H-08 |
| M-24 | Time from the e-stop command to physical lever neutral (each lever separately) and to PTO drop (relay and clutch), at the deployed supply | high-speed video, position sensor | acceptance sections 2 and 4; R-P09 |
| M-25 | Servo neutral and limit pulse widths per channel, travel stops, linkage geometry | bench calibration | H-20 |
| M-26 | Relay pin state at process exit and at SIGKILL | scope with the backend stopped and killed | N-09 |
| M-27 | Stuck-output behaviour: PWM held with the lever blocked; servo behaviour with no pulse | bench | Part D.8 |
| M-28 | Fault-to-safe-state time per fault class: backend `SIGSTOP`, `SIGKILL`, host hang, heartbeat wire cut, USB or serial loss, SDA or SCL held low during a write (kernel timeout, PCA9685 output, recovery); servo-power cut and lever neutral each timed | scope, high-speed video, position sensor | N-28, Parts D.4 and D.7 |
| M-29 | Hardwired E-stop chain: acts with the Pi unpowered, hung and unplugged; an open wire reads as a stop; release does not re-energise; relay coil-supply and contact states at each step | meter, scope | R-P09, Part D.1 |
| M-30 | Trace the OEM PTO circuit: is a momentary press latched by an OEM controller; how is OFF guaranteed on fault | meter, schematic, engine off, blade disconnected | H-21 |
| M-31 | Observe the seat-switch 1 s engine cutoff on the actual machine (measure it, do not defeat it) | stopwatch, supervised | acceptance section 8 |
| M-32 | Does the OEM Reverse Operation System cut the blade in reverse independently | supervised bench | acceptance section 9 |
| M-33 | Toro serial plate (model and serial) and manual revision | photograph | H-06 |
| M-40 | JSN-SR04T: revision and mode, supply range, ECHO high level at the real supply, current draw, timing, blind zone, beam pattern, range on grass, people, children, pets and low objects, rain | scope, meter, target rig | H-10, H-19, B.5 |
| M-41 | VL53L0X: units against `limits.yaml`, XSHUT pull states, 3.3 V current, sunlight limit | meter, outdoor test | H-03 |
| M-42 | BNO085: transport, straps, reset and interrupt wiring, mounting axes; the machine's rollover angle and the end-to-end tilt-to-blade-off latency | bench, tilt rig | H-03, D.2 |
| M-43 | Stopping distance against speed and slope (hydrostatic coast after neutral) | supervised field test, measured | H-26, B.5 |
| M-44 | Stereo: baseline, calibration, pixel format and frame rate over the actual USB bus, end-to-end latency | calibration target, host tools | H-16, H-26 |
| M-45 | LiDAR (if selected): model, scan-plane height, field of view, minimum range, sunlight and rain behaviour | datasheet, bench | H-15 |
| M-50 | Which GNSS module is physically fitted (photo and label); what the CP2102N on `/dev/ttyUSB0` connects to; LC29H-DA carrier pinout, supply and antenna bias | inspection | H-09 |
| M-51 | RTK: time to fix, fix transitions, correction age, with and without the base, poor sky view, antenna failure | logged runs | H-09, H-22 |
| M-52 | HaLow at far and obstructed yard points; drop and reconnect; Pi reboot; radio power-cycle recovery; link-loss-to-lever-neutral latency end to end | iperf, ping, fault injection, video | H-16, H-23 |
| M-53 | Which board is the HaLow HAT (label and photo) | inspection | H-17 |
| M-54 | Hailo identity (`hailortcli fw-control identify`, label), power draw, temperature in the enclosure | CLI, meter, thermal probe | H-17 |
| M-55 | Pi 5 plus Hailo thermals in the not-IP-rated printed enclosure at hot ambient | thermal probe | `hardware.md` Weatherproofing |

## Part C. Phase 4 distributed AI and navigation architecture

"Target" is the owner's requirement from the PR #18 documents; "implemented" is the baseline in Part A. Nothing in
this part is a claim that a target exists. The authority rule is the owner's and is kept unchanged: **safety
authority stays local.** The Thor may suggest waypoints or model outputs; it may never energise an actuator, clear
an e-stop or override a Pi interlock, and the mower must follow a defined locally safe behaviour when the Thor or
the HaLow link disappears.

### C.1 Authority split

```mermaid
flowchart TB
  subgraph HOUSE["House"]
    THOR["Thor: coverage strategy, optional heavy inference (strategist in SIM only)"]
    RTKB["RTK base chain: LC29H-BS, Pi Zero, Orange Pi (MISSING)"]
  end
  subgraph MOWER["Mower"]
    LINK["ThorLinkService (library, not wired, no sender auth)"]
    NAV["NavigationService (partial)"]
    FUSE["Obstacle fusion (stub)"]
    SAFE["Local safety supervisor as one unit (MISSING)"]
    TRACTOR["TractorControlService, PCA9685, relays (partial)"]
  end
  HW["Independent hardware: E-stop, watchdog, OEM interlocks (MISSING)"]
  THOR -->|"plan windows over HaLow"| LINK
  LINK --> NAV
  FUSE --> SAFE
  NAV --> SAFE
  SAFE --> TRACTOR
  HW --> TRACTOR
  classDef missing fill:#fdd,stroke:#c00
  classDef partial fill:#ffe9b3,stroke:#b80
  classDef impl fill:#d8f5d0,stroke:#287
  class RTKB,SAFE,HW missing
  class LINK,NAV,FUSE,TRACTOR partial
  class THOR impl
```

*Legend: green = implemented (in simulation), amber = partially implemented, red = missing.*

| Decision | Target owner | Implemented at baseline | Status | Gap |
|---|---|---|---|---|
| Coverage plan and waypoint windows | Thor, high level | strategist in SIM (`nav/thor_strategist.py`); link service unwired; no consumer of `current_plan()` | PARTIAL | H-23 |
| Re-validation of every plan (geofence, no-go zones, speed, blade permission) | Pi | `nav/geofence_validator.py` exists; not applied to Thor plans | MISSING | H-23 |
| Local obstacle stop | Pi, deterministic, independent of the AI and the Thor | none for actuation | MISSING | H-26, N-11 |
| Speed cap from detection range and stopping distance | Pi | none | MISSING | H-26 |
| E-stop | independent hardware | software only | MISSING | H-08 |
| Clearing an e-stop | local human acknowledgement | remote API call, auth off | MISSING | N-10 |
| Engine start | a human, levers in PARK | auto-crank in the mission start | CONTRADICTED | H-06, N-08 |
| PTO energise | Pi, gated by interlocks and a hardware fail-safe OFF | software relay, no feedback | PARTIAL | H-21 |
| Perception used for safety | Pi, local (Hailo, ToF, ultrasonics, stereo, optional LiDAR) | detector in the camera service; nothing feeds a stop | PARTIAL | H-26 |
| Perception used for planning | Thor, optional | detections-to-uplink MISSING | MISSING | H-23 |

### C.2 Link-loss and degraded-mode behaviour

| Event | Required local behaviour (target) | Implemented at baseline | Gap |
|---|---|---|---|
| Thor silent for more than 0.5 s | Hold: PTO off, levers neutral, authorisation kept; resume only on a fresh, authenticated plan | `ThorLinkService._evaluate_hold` (library; `thor_link_service.py:132-145`) | H-23, N-15 |
| Thor alive but its planner is broken | Plan sanity checks, plan age, local re-validation; hold if the plan disagrees with the Pi's state | none: freshness is "any valid downlink in 0.5 s" | N-15 |
| HaLow link down (Pi isolated) | As for Thor silent, plus an explicit, tested decision on local continuation | not wired | H-23 |
| HaLow radio reset or power cycle | Defined re-association; no unsafe command on rejoin | unverified (`docs/autonomy-toolchain-matrix.md:68`: "not exercised") | H-23 |
| Operator client lost (manual or VR) | Dead-man: the command lease expires, levers go neutral, PTO off | the client stops sending (`scripts/teleop.py:37`); the server keeps the last command indefinitely | N-14 |
| Backend process dies or hangs | An independent hardware path reaches a defined safe state | none (A.5, N-09) | H-08 |
| RTK fix degrades (fixed to float or none) | Speed cap, hold or geofence margin by fix quality | in-memory mode flip only (`nav/gps_degradation.py:22-35`) | H-09 |
| Correction age above its limit | Treated as a degraded fix | no metric consumed | H-09 |
| IMU stale or lost | Fault, then hold (fail closed) | fail-open (N-06) | N-06 |
| Obstacle sensors stale or disagreeing | Fail closed: speed cap or stop | none | H-26 |
| Camera or inference stalled | The deterministic stop path does not depend on it | the AI path stops by accident when ultrasonic data is absent (N-25) | N-25 |
| Battery low, over-temperature | Stop or return | `trigger_low_battery`, `trigger_high_temp` have no callers (`safety_triggers.py:112-141`) | N-11 |
| Pi brown-out | Detect and reach a safe state | `get_throttled` is not read by any service | R-P05 |

### C.3 Secure HaLow and an authenticated Thor link

| ID | Requirement (target) | Status at baseline |
|---|---|---|
| RS-1 | Link-layer authentication and encryption on the HaLow association (AP and station); network identity documented. The AP-side configuration lives on the Orange Pi, not in this repository | UNVERIFIED |
| RS-2 | The Thor-link socket binds the HaLow interface address only and accepts datagrams only from the configured Thor address | MISSING: binds `0.0.0.0:47101` and ignores the source (`thor_link_service.py:54,98-110`) |
| RS-3 | Every datagram is authenticated (for example an HMAC with a pre-shared key held outside git) and carries an epoch or boot id with a sequence number, so a Thor restart and a replay are both handled correctly | MISSING |
| RS-4 | Plan freshness: `plan_id`, an age bound, and `start_index` validated against the Pi's `waypoint_index` | PARTIAL: `seq` only (`:106-107`) |
| RS-5 | Rate limits and bounded sizes | PARTIAL: 1200 B cap (`models/thor_link.py:24,92-100`); rejected counter (`thor_link_service.py:74`) |
| RS-6 | Every accepted plan is re-validated locally (geofence, no-go zones, speed, blade permission); the Thor cannot enable the blade where the Pi forbids it | MISSING |
| RS-7 | The downlink carries no e-stop clear, arm or velocity message | IMPLEMENTED by omission (`models/thor_link.py:70-82`: waypoints and `pause` only) |
| RS-8 | Key storage, rotation and behaviour across reboots | MISSING |
| RS-9 | Counters for rejected, stale and replayed datagrams exposed to observability | PARTIAL |

Without RS-2 to RS-4, a spoofed downlink could inject waypoints with `blade: true` legs or force or release a hold,
and one datagram with a very large `seq` blocks every later legitimate one until the Pi restarts (N-15). The service
is unwired today, so nothing consumes a plan: the exposure is latent and must be closed **before** it is wired.

### C.4 RTK correction distribution (candidate topology)

```mermaid
flowchart LR
  BASE["LC29H-BS base: antenna, survey, RTCM out"] --> PZ["Pi Zero: serial or USB host"]
  PZ --> OPI["Orange Pi: HaLow AP and gateway"]
  OPI -.->|"HaLow"| MPI["Mower Pi: forwarder"]
  MPI -->|"RTCM"| ROVER["LC29H-DA rover"]
  CASTER["Public NTRIP caster (existing client)"] -.->|"fallback over Internet"| MPI
```

| ID | Requirement | Status at baseline |
|---|---|---|
| RT-1 | The base antenna position is surveyed or known, and the antenna placement is recorded | MISSING |
| RT-2 | The RTCM message set the LC29H-DA needs is identified from Quectel documentation and the base is configured to emit it | MISSING (no `PAIR`/`PQTM` handling; the documents are not in the repository) |
| RT-3 | Transport from the base to the mower has bounded latency and the rover measures correction age | MISSING (no age metric consumed) |
| RT-4 | The RTCM injection path to the rover is defined (UART0 RX or a second UART) | PARTIAL: `ntrip_client.py` writes to a serial device (`NTRIP_SERIAL_DEVICE`) |
| RT-5 | Failure of each node (base, Pi Zero, Orange Pi, radio, Thor) has a defined result | MISSING |
| RT-6 | Fallback to the public caster when the local base is down | PARTIAL: the caster client exists; the switchover logic does not |
| RT-7 | Alarms for fix transitions, correction age and antenna failure | MISSING |

### C.5 Stereo depth and the bandwidth it implies

Capture is implemented; depth is not. At 2560x960 and 30 fps (`stereo_camera_driver.py:48-50`):

| Representation | Rate | Against the measured 0.60 MB/s HaLow TCP (`docs/autonomy-toolchain-matrix.md:34`) |
|---|---|---|
| 1 byte per pixel | 73.7 MB/s | about 120 times too large |
| 2 bytes per pixel (YUYV) | 147.5 MB/s | about 245 times |
| 3 bytes per pixel (BGR after decode) | 221.2 MB/s | about 370 times |
| Thor link protocol at its worst case | 12 kB/s (`models/thor_link.py:111-113`) | 2 % |

Conclusion: **raw or lightly compressed stereo cannot cross HaLow.** Depth and obstacle extraction must run on the
mower, and only distilled results (detections, a coarse range image, or compressed video at a few Mbit/s) may cross
the link; the throughput at the far yard is unmeasured (M-52). Candidate depth paths, all unevaluated: classical
stereo matching on the Pi at reduced resolution; the monocular `scdepthv3` HEF on the Hailo (driver support exists,
`hailo_driver.py:63,77`); a neural stereo model on the Thor from compressed video (latency-bound). Needed for any
of them: stored intrinsics and extrinsics, rectification, a confidence output, a timestamp, a transform into the
vehicle frame, and validation against measured ranges on real grass (low texture, glare, dust).

### C.6 2D LiDAR (optional) and C.7 ultrasonic

- **LiDAR:** MISSING. Selection criteria to fix first: sunlight and rain rating, minimum range, scan rate, interface
  and power, driver licence, mounting height, and the blind spots above and below its scan plane. A planar scan
  alone cannot cover negative obstacles or objects off the plane (PR #18 plan).
- **Ultrasonic (4x JSN-SR04T):** MISSING. Needed: a driver for the actual module mode, time-division firing so the
  four sensors do not hear each other, per-sensor blind-zone handling (a reading inside the blind zone means
  **unknown**, never "clear"), speed-of-sound compensation if justified by M-40, staleness and outlier handling,
  health reporting, and use as one input among several, never alone (roadmap P2).

### C.8 Fusion

Target: timestamped measurements from every sensor, static extrinsics per sensor into the vehicle frame, one
obstacle representation (range-bearing hits, keep-out polygons or a local occupancy grid), per-source confidence and
staleness, and deterministic rules for the stop path: any valid detection inside the stopping distance `d_stop(v)`
stops the machine; stale or blind readings count as blocked for the speed cap; the AI and the Thor may only reduce
speed further. Implemented: **none** (A.6, N-11, H-26). `tests/contract/test_sensor_fusion.py` is a skipped
placeholder that currently stands in for it.

### C.9 ROS 2 / Nav2 feasibility

Status: **no ROS code, dependency or configuration exists**. Assessment (unevaluated; nothing was built or run):

| Question | Finding |
|---|---|
| Runtime | The repository pins Python 3.11.x (`docs/hardware-integration.md`), while the Pi image documented on 2026-10-06 was Ubuntu 24.04.4 with system Python 3.12.3 (`docs/autonomy-toolchain-matrix.md:14`) and the live kernel suggests a different image. ROS 2 binary distributions are tied to the OS release [DS]; Debian Bookworm (`spec/hardware.yaml:3`) is not a tier-1 target [DS], so a source build or a container may be needed. `rclpy` binds the system Python, not the repository's venv [DS], which implies a separate process. Unknown: which OS image is intended |
| Resource budget | Shares the Pi with the backend (`MemoryMax=512M`, `CPUQuota=80%` in `systemd/lawnberry-backend.service:64-65`), the camera service and the Hailo runtime; unmeasured |
| Kinematics | Nav2 controllers assume differential, omni or Ackermann bases [DS]. At command level the zero-turn machine is a differential drive (`navigation_service.py:333-346` reuses the differential mix), but the lever-to-speed map, hydrostatic lag and coast are unmeasured (M-43), so a controller would need system identification |
| Localisation | RTK-fixed GNSS plus a real IMU feed (BNO085 driver is a stub) and the ROS frame conventions [DS] |
| Obstacle input | Nav2 costmaps accept range, laser-scan and point-cloud inputs [DS]; four ultrasonic ranges, a stereo point cloud and an optional LiDAR are plausible inputs, and **each driver is MISSING** |
| Safety | Nav2 is not a safety system. The deterministic stop path and the interlocked tractor methods must stay outside it (`docs/tractor-acceptance-criteria.md` section 5) |
| Coverage | The repository already has a coverage planner and a Thor strategist; Nav2 coverage add-ons would duplicate them [DS] |
| Verdict | **Feasible in principle, unevaluated, and premature** before the P2 sensor drivers and the P4 stop path. Run a bounded spike on the Thor against recorded data first; decide at gate G4 (Part E) |

### C.10 Coverage planning and C.11 VR teleoperation

- **Coverage planning.** Implemented: a local planner (`nav/coverage_planner.py`, `coverage_patterns.py`,
  `path_planner.py`, `zone_coverage.py`) and a Thor-side strategist that plans a boustrophedon inside the boundary
  minus no-go zones, replans around detections, pauses unless the fix is `rtk_fixed`, and resumes. Target: the Thor
  plans, the Pi validates and executes. Missing: the consumer, the uplink builder, plan-age and `start_index`
  validation, and any run on the machine (H-23).
- **VR and manual driving (P5, optional).** Target: a pan/tilt stereo camera, head tracking, a codec and a measured
  latency and jitter budget, authenticated operator control with a heartbeat and a **server-enforced** dead-man,
  local obstacle interlocks, and no use of the video path as the sole safety feedback. Implemented: keyboard
  teleoperation with a client-side dead-man (`scripts/teleop.py`) and capture logging; no pan/tilt, head tracking,
  stereo stream or server lease (H-16, H-14, N-14). The bandwidth arithmetic is in C.5.

### C.12 Multi-agent automation (H-27) and future compute portability (H-28)

PR #16 is a one-file draft design document. PR #17 (`feat/agent-bridge-omp`) is a draft of 62 files under
`tools/agent_bridge/**`; it is not merged and adds nothing under `backend/`. Do not claim Codex, Claude Code or Oh My
Pi orchestration until it is tested. Agents work in separate Git worktrees, exchange results through draft PRs and
human review, and no agent drives mower hardware because it read a handoff. The HPE server migration (H-28) is a
future optional deployment: it needs containerised services and hardware-agnostic interfaces defined first, and it
must not be confused with the Thor's role.

## Part D. Phase 5 safety review

The review covers the ten topics the owner's handoff names for the safety case (`OMP-THOR-HANDOFF-2026-10-24.md`,
deliverable 3): independent hardwired E-stop, blade/PTO disable, neutral-on-fault, watchdog, loss-of-link,
loss-of-Thor, I2C bus hang, servo stuck-output, GPS/RTK loss and physical bench acceptance. It reads the source at
baseline and the repository's own acceptance criteria; nothing was run against hardware. It does not recommend
bypassing any OEM interlock (seat or presence switch, PARK, Reverse Operation System, PTO switch).

### D.0 Ground rules, target safe-state model and summary

1. **Authority order.** Independent hardware, then a hardware watchdog or coprocessor, then the Pi's safety code, then
   navigation, then the Thor. A higher layer can always stop the machine. A lower layer can never prevent that stop
   or clear it.
2. **Fail-safe direction.** Loss of power, signal, heartbeat or a wire leaves the machine in the safe state: servo
   power off, PTO supply off, every relay output defaulting OFF (R-P10).
3. **Software states are not physical states.** `state.blade_engaged`, the lever values and `engine` are commanded
   values (N-26). A safety property is shown by a physical observation (`docs/tractor-acceptance-criteria.md`,
   preamble); a passing unit test is necessary and never sufficient.
4. **Hold is not E-stop.** Hold: PTO off, levers neutral, authorisation kept, left on a fresh good input. E-stop:
   energy removed, latched across restarts, cleared only by a local act.
5. **Numbers come from measurement.** The only numeric limits used are those the repository already states: relay
   within 100 ms, lever safe-value command within 100 ms, lever physical settle within 500 ms (acceptance criteria
   sections 2 and 4), tilt cutoff within 200 ms (section 7) and the shipped `config/limits.yaml`. Nothing electrical
   or temporal is invented here.

Target safe-state model (**not implemented**: the baseline has only the flags `authorized` and
`emergency_stop_active` in `tractor_service.py:109-110,266-268`, and `safe_hold` in the unwired `ThorLinkService`):

```mermaid
stateDiagram-v2
  [*] --> SAFE_OFF
  SAFE_OFF --> ARMED : local arm, attestation, self-checks pass
  ARMED --> RUN : valid command lease
  RUN --> HOLD : lease or link expired, Thor lost, fix degraded, sensor stale, obstacle
  HOLD --> RUN : fresh authenticated input and conditions healthy
  RUN --> ESTOP : E-stop, watchdog, tilt, actuator fault
  HOLD --> ESTOP : E-stop, watchdog, tilt, actuator fault
  ARMED --> ESTOP : E-stop, watchdog, actuator fault
  ESTOP --> SAFE_OFF : local acknowledgement only
```

`SAFE_OFF`: actuator power off, outputs default OFF. `HOLD`: PTO off, levers neutral. `ESTOP`: servo power and PTO
supply removed by hardware, latched across restarts.

| # | Topic | Status at baseline | Principal reason | Work packages | Physical evidence |
|---|---|---|---|---|---|
| D.1 | Independent hardwired E-stop | **MISSING** | no input, circuit or GPIO; the software routes do not all actuate (N-01 to N-03) | WP-P0-05, WP-P1-01, WP-P1-05, WP-P0-08 | M-24, M-29; acceptance 2, 4 |
| D.2 | Blade/PTO disable | PARTIALLY IMPLEMENTED; contradicted by the momentary-PTO report | software relay, no feedback, unguarded `set`, tilt defeated by the mission loop | WP-P0-06, WP-P1-07, WP-P1-04, WP-P1-10 | M-17, M-24, M-30, M-32, M-42; acceptance 2, 7, 9 |
| D.3 | Neutral-on-fault | PARTIALLY IMPLEMENTED (command only) | neutral is a number; self-centring unknown; `OE` gives no pulse | WP-P0-05, WP-P1-01, WP-P1-02, WP-P1-09 | M-21, M-22, M-24, M-25, M-27; acceptance 1, 4, 10 |
| D.4 | Watchdog | PARTIALLY IMPLEMENTED (software liveness) | in-process, one-shot, fed unconditionally; circuit values inconsistent (N-29) | WP-P0-05, WP-P1-08, WP-P1-02 | M-23, M-26, M-28; acceptance 1, 7 |
| D.5 | Loss of link (operator, VR, network) | **MISSING** for the operator | commands persist; dead-man only in the client; the host refresh defeats the firmware failsafe | WP-P1-09, WP-P1-11, WP-P5-01, WP-P3-06 | M-52, M-24 |
| D.6 | Loss of Thor | PARTIALLY IMPLEMENTED (library, unwired) | not wired; no authentication; hold retry gap (N-30) | WP-P3-01 to WP-P3-03, WP-P3-06 | M-52, M-51 |
| D.7 | I2C bus hang | PARTIALLY IMPLEMENTED | error propagation only; absent chip is silent; no timeout; no header bus on the deployed Pi | WP-P0-01, WP-P1-03, WP-P1-04, WP-P1-08 | M-20, M-21, M-28; acceptance 3 |
| D.8 | Servo stuck-output | **MISSING** | no feedback, stall detection or per-servo power cut; linkage unqualified for powered use | WP-P0-04, WP-P0-07, WP-P1-01, WP-P1-12 | M-10, M-11, M-22, M-25, M-27 |
| D.9 | GPS/RTK loss | PARTIALLY IMPLEMENTED | the monitor does not see real missions (N-31); no start precondition; no correction age; no local base | WP-P2-04, WP-P2-05, WP-P4-03, WP-P3-06 | M-50, M-51 |
| D.10 | Physical bench acceptance | **MISSING** | procedure written, nothing measured; no harness; no hardware test in CI | WP-P0-07, WP-P1-13, WP-P6-03 | M-10 to M-32; the sign-off table |

### D.1 Independent hardwired E-stop (MISSING)

**Requirement.** A mushroom E-stop whose action does not pass through the Pi, the backend, the I2C bus, a network or
any firmware. It removes the energy that moves the actuators (servo power, the PTO relay coil supply) and is
demonstrated with the Pi powered off, hung and unplugged. It is a normally-closed loop, so a broken wire reads as a
stop [DS: common practice, for the reviewer to confirm]. Releasing it re-energises nothing by itself. Whether it also
stops the engine or applies a brake is a safety-engineering decision (SD-1), not a software one.

**Baseline.** No input, pin or circuit exists: `backend/src/safety/estop_handler.py:5-7` says GPIO integration "will
be added later"; `spec/hardware.yaml:87` is a flag with no pin; `hardware.md` (Safety) lists a mushroom button and
`hardware/mounts/README.md:44-45,100-101` prints a bracket for it (rail dimensions unverified) but no circuit is
defined. The RC channel-5 stop on the RP2040 is evaluated only while `rc_enabled`
(`robohat-rp2040-code/code.py:312,328-336`), shares the MCU with the drive output and does not exist on the PCA9685
Toro platform. `TractorControlService.emergency_stop` (`tractor_service.py:258-288`) is a function in the process it
is meant to supervise; the general routes do not actuate the tractor or the IBT-4 and reply that they did (N-01 to
N-03); the "E-stop latency under 100 ms" test times a flag flip (N-24).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Button pressed | no input | servo-power relay and PTO supply open whatever the Pi is doing; times per actuator measured (M-24) |
| Pi off, hung, backend killed | the PCA9685 holds its last pulse [DS]; relay pin state unknown (M-26); the software e-stop is unreachable | the hardwired path acts as above; the hardware watchdog (D.4) opens the same relays |
| Wire to the button breaks | not applicable | reads as a stop |
| Button released | not applicable | nothing re-energises; re-arm is a local act (WP-P1-05); the software latch survives a restart (N-10) |
| Relay contact welds or its driver fails | not considered | in the hazard analysis (WP-P0-07); monitoring or redundancy is the reviewer's decision |
| Engine keeps running after the stop | by design (`tractor_service.py:259`); no brake reachable by the fore/aft linkage (`docs/tractor-platform.md`, Physical integration decisions 2); coast unmeasured (M-43) | decided by SD-1 and the measured stopping distance |

**Required work:** WP-P0-05 (design, SD-1), WP-P1-01 (build and bench), WP-P1-05 (latch), WP-P0-08 (truthful
routes). **Physical evidence:** M-24, M-29; acceptance sections 2 and 4.

### D.2 Blade/PTO disable (PARTIALLY IMPLEMENTED; contradicted)

**Requirement.** Every stop condition (E-stop, tilt, reverse, loss of link or Thor, watchdog, backend death, power
loss) ends with the PTO off by a path that does not depend on the software state, and the real PTO state is observed.
Engagement needs the engine sensed running, an operator attestation, not-reverse and not-tilted (acceptance
sections 6 and 9). OEM seat, PARK and PTO-switch interlocks stay in force.

**Baseline.** `blade_pto` is modelled as a latching relay on GPIO26 (`config/tractor.yaml:32-36`,
`tractor_service.py:218-231`) and is switched off first and unconditionally in the e-stop (`:269`). But
`RelayActuator.set` is unguarded (`tractor_actuators.py:111-115`), so a GPIO error raises before the servos are safed
(N-27); a failed `OutputDevice` construction falls back silently to a simulated relay while the state still reads off
(`:104-108`); the owner reports a momentary OEM PTO toggle (H-21), and if an OEM controller latches it, dropping the
relay does not disengage the PTO (N-26: no pulse-to-toggle logic, no feedback); the tilt cutoff fires once on the
edge and the mission loop re-engages the blade within about 200 ms (N-06 [INF]) on top of an IMU stub; relay pin
state at process exit and `SIGKILL` is unknown and the `finally` in `pulse` does not run on `SIGKILL` (N-09); the
mower variant's IBT-4 blade driver is a stub that is never e-stopped (N-03); the software reverse interlock exists,
but whether the OEM Reverse Operation System also cuts the blade is unverified (acceptance section 9).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| E-stop and the GPIO raises | exception before servo safing (N-27); state may disagree with the pin | each safing step independent and reported; the hardwired chain also removes the PTO supply |
| Relay driver missing or failing in production | silent simulation fallback | fault; arming refused |
| Tilt over the threshold | one-shot blade-off, re-engaged by the mission loop | latched until a local clear; IMU loss fails closed (WP-P1-10) |
| The OEM controller latches the PTO | state says off, the PTO may still be on | trace first (M-30); a fail-safe OFF element in the clutch supply, for example a contactor energised only while commanded (design decision, not made here); observed PTO state |
| Backend killed or Pi power lost | relay pin state unknown | default OFF at power-up, boot and pin release (R-P10); hardwired chain |
| Reverse entered | software disengage | OEM mechanism characterised (M-32); both documented, neither assumed |

**Required work:** WP-P0-06, WP-P1-07, WP-P1-04, WP-P1-10, WP-P1-05 (shutdown), WP-P1-01. **Physical evidence:**
M-17, M-24 (PTO drop, relay and clutch), M-26, M-30, M-32, M-42; acceptance sections 2, 7 and 9.

### D.3 Neutral-on-fault (PARTIALLY IMPLEMENTED, command only)

**Requirement.** On any fault or loss of control authority both drive levers reach neutral and the throttle reaches
idle within the physical settle time the acceptance criteria require (500 ms), by a mechanism that works without
software. "Neutral" is a lever position, not 1500 µs, and the stopping distance after neutral is known (M-43).

**Baseline.** Neutral is a commanded value (`ServoCalibration`, `tractor_actuators.py:30-39`; endpoints are
illustrative, `config/tractor.yaml:22-27`). Whether the OEM levers self-centre is unknown (acceptance section 10), and
the repository itself warns that "a geared servo power cut alone does not prove failsafe return"
(`hardware/mounts/README.md:415-418`). The `OE` mechanism in the acceptance document was wrong (corrected in this PR,
H-04): `OE` is active-low [DS]; high disables the outputs and with `MODE2 = 0x04` gives no pulse at all
(`pca9685_driver.py:57,111`), so it is "stop commanding", and what a servo then does (hold, drift, limp) is
unmeasured (M-21, M-27). No `OE` GPIO is allocated (C-14). `hardware.md` (Safety) intends a 74HC123 watchdog to cut
servo power through a spare relay-module channel and calls it an "unproven failsafe"; the circuit is not designed
(N-29). A parking brake exists only at the outboard PARK position, by hand (`docs/tractor-platform.md`, Physical
integration decisions 2); neutral gives hydrostatic braking only.

| Option | Safe state comes from | Depends on | Open measurement |
|---|---|---|---|
| 1. Servo power cut | servo unpowered; lever return springs or OEM spring return | levers self-centre with the linkage fitted; the servo is back-drivable | M-22; acceptance section 10 |
| 2. Active neutral from an independent controller | a coprocessor outputs the neutral pulse when the Pi heartbeat stops | coprocessor firmware and its reset state; servo power stays on | M-21, M-27; its reset behaviour |
| 3. `OE` high | no pulse | servo behaviour with no pulse | M-21, M-27 |
| 4. Software neutral command | a commanded neutral | the bus, the process and the PCA9685 all alive | none: this is today's behaviour and is not a failsafe |

Options 1 and 2 may be combined. Option 3 alone and option 4 alone are not failsafes. No option is chosen here
(SD-2). Other faults to cover: a Pi crash or I2C break (the PCA9685 holds its last pulse [DS]); servo power cut with
levers that do not self-centre (a runaway, which invalidates option 1); an e-stop with a jammed lever (D.8).

**Required work:** WP-P0-05, WP-P1-01, WP-P1-02 (if option 2), WP-P1-03, WP-P1-09. **Physical evidence:** M-21, M-22,
M-24, M-25, M-27, M-43; acceptance sections 1, 2, 4 and 10.

### D.4 Watchdog (PARTIALLY IMPLEMENTED, software liveness only)

**Requirement.** The watchdog is (a) independent of the supervised process (hardware, or a separate MCU with its own
reset behaviour); (b) fed only by evidence that the control path works: a valid command lease, a completed
safety-supervisor cycle, a good PCA9685 read-back, fresh sensors; (c) wired so that its timeout produces the same
stop as the E-stop; (d) re-armed after a clear; (e) sized from the stopping distance (M-43), not from
`watchdog >= estop_latency`; (f) started and fail-closed even when tractor initialisation fails; (g) backed by a
hardware path for a hung host.

**Baseline.** `Watchdog._run` (`watchdog.py:54-69`) runs in the event loop it supervises and is one-shot (`break`
after the first timeout). `TractorSafetyMonitor` feeds the heartbeat unconditionally every 50 ms
(`tractor_safety_monitor.py:84-87`), so it proves only that the loop turns. Its `EstopHandler` holds a private
`MotorAuthorization` that nothing else reads (`main.py:155-158`), so only `on_timeout=tractor.emergency_stop` has an
effect. `config/limits.yaml:10` sets 1000 ms against `estop_latency_ms: 100`, and the validator checks only
`watchdog >= estop` (`safety_validator.py:37-38`). Any exception in the tractor block of `main.py:144-170` leaves the
actuation API live with no watchdog and no tilt monitor (N-05). `systemd/10-watchdog.conf`
(`RuntimeWatchdogSec=20s`) reboots a hung host and does not neutralise actuators; whether it is installed on the Pi
was not checked. The RoboHAT firmware watchdog (8 s, `code.py:98-101`) protects the MCU only, and the MDDRC10 input
state during its reset is unknown (M-23). The 74HC123 design in `hardware.md` uses an unassigned GPIO and values that
do not agree with their own formula (N-29).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Event loop blocked or process hung | the in-process watchdog is in the same loop and cannot fire | the hardware watchdog window expires; servo power and PTO supply open |
| Process killed | no supervisor | same |
| Tractor initialisation fails | no watchdog, no tilt monitor, API live (N-05) | fail closed: actuation routes refuse; the un-fed hardware watchdog opens the supply |
| Watchdog fires once | one-shot; a later hang is unsupervised | re-armed after a local clear |
| Heartbeat healthy while the command path is broken (hung bus, stuck servo, lost operator) | still fed | not fed |
| Host kernel hang | actuators hold their last state until a host reboot at 20 s [DS] | the hardware watchdog acts within its window |
| MCU watchdog reset (RoboHAT) | output states unknown | measured (M-23) |

**Required work:** WP-P0-05 (window, circuit), WP-P1-08, WP-P1-02, WP-P1-01. **Physical evidence:** M-23, M-26, M-28
(stop the backend with `SIGSTOP`, kill it, hang the host, cut the heartbeat wire; the method is for the reviewer to
confirm); acceptance sections 1 and 7.

### D.5 Loss of link: operator, teleoperation, VR, network (MISSING for the operator)

**Requirement.** Every source that can move the machine (REST, WebSocket, teleoperation, VR) holds a lease that the
**server** enforces: a command carries a validity window, only the client can refresh it, expiry puts the machine in
HOLD, reconnection never resumes motion by itself, and one authenticated controller holds the lease. The window comes
from the measured stopping distance (M-43, SD-6).

**Baseline.** `POST /api/v2/tractor/command` applies a command that persists until another arrives
(`backend/src/api/routers/tractor.py:67-70`, `tractor_service.py:176-203`). The only dead-man is in the client:
`scripts/teleop.py:36-37` stops sending after 0.25 s without a key and does not send neutral, and its exit-time
neutral does not run if the client crashes. `POST /control/drive` validates `duration_ms` and never uses it
(`rest.py:378-388`); `MotorService` has a command-timeout handler (`motor_service.py:409-419`) that the REST path
does not use (`rest.py:310-417`). On the differential-drive platform the host loop re-sends the last PWM about every
0.9 s (`robohat_service.py:246-269`), which defeats the firmware's fall-back to RC mode after 2 s of USB silence and
its centring after 1 s of RC loss (`code.py:305-309,111,315-322`). `/ws/control` drains and acknowledges only
(`routers/telemetry.py:175-206`), so no VR control channel exists. HaLow drop and rejoin were not exercised
(`docs/autonomy-toolchain-matrix.md:68`).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Client dies after a non-neutral command | the lever value persists | the lease expires; HOLD |
| Network partition, operator to Pi | same | same |
| HaLow radio reset, power cycle or AP loss | unexercised (A.8) | same as above; recovery tested (M-52) |
| Reconnect after a loss | no state machine | no automatic resume; a new lease, and a local re-arm if the stop was an E-stop |
| Late or replayed command | commands carry no sequence or time | sequence and validity window; stale commands dropped |
| Unauthenticated client | authentication disabled in the shipped unit | only the authenticated lease holder (WP-P1-11) |

**Required work:** WP-P1-09, WP-P1-11, WP-P5-01, WP-P3-06. **Physical evidence:** M-52 (link loss to lever neutral,
end to end), M-24; the roadmap's P5 acceptance ("no motion on dead-man release").

### D.6 Loss of Thor (PARTIALLY IMPLEMENTED, library only)

**Requirement.** The Thor is never safety authority. Its loss or failure (process, host, AP, radio, an intermediate
node, a wrong plan) leaves the mower locally safe: HOLD by default, and a pre-validated local continuation only if
the owner and the reviewer approve one (SD-3). Every plan is re-validated locally (geofence and no-go zones, speed,
blade permission, plan age, `start_index` against the Pi's waypoint index); datagrams are authenticated; the downlink
can never arm, clear an E-stop or override an interlock (RS-7).

**Baseline.** The protocol (`models/thor_link.py:22-113`) and the service (`thor_link_service.py:49-172`) exist with
eight tests on a fake clock plus a loopback round trip (`tests/unit/test_thor_link.py`). Nothing instantiates the
service outside the tests, nothing consumes `current_plan()`, and `/tractor/*` and `/control/drive` do not consult
`safe_hold`. Defects: any source address is accepted and the socket binds `0.0.0.0:47101` (`:54,98-110`); no MAC or
epoch; `seq <= _last_rx_seq` drops the datagram (`:106-107`), so a spoofed large `seq` or a Thor restart that resets
its counter is dropped until the Pi restarts (N-15); a first `engage_blade(False)` failure skips the lever call and
is never retried for the blade (N-30); `tick` swallows actuation errors (`:121-130`); freshness means "any valid
downlink within 0.5 s" (`:112-113`), so a Thor with a broken planner keeps the link fresh. The service starts in hold
(`:71`), which is the right default. The strategist pauses unless the fix is `rtk_fixed`
(`backend/src/nav/thor_strategist.py:38`) but runs in simulation only. House-side node failures (base, Pi Zero,
Orange Pi) were never analysed (H-22).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Thor silent for more than 0.5 s | `_evaluate_hold` (library) | hold, wired into the backend, with the effects defined in D.0 |
| Thor alive, planner wrong | none | local re-validation; hold on disagreement |
| Spoofed or replayed downlink | accepted by a source-agnostic socket | rejected (MAC, source address, epoch); counters exposed |
| Thor restart | later datagrams dropped until the Pi restarts [INF] | epoch or boot id: accepted once, replays still rejected |
| AP, radio or an intermediate node fails | unexercised | same as silent; recovery tested by power cycle and driver reload (M-52) |
| Link returns | a fresh downlink leaves hold (`:143-145`) | HOLD ends only on a fresh authenticated plan that passes local re-validation (SD-3 decides whether an operator acknowledgement is also needed) |
| Hold entered while actuation fails | `safe_hold` true, blade not retried | each step reported and retried until the hardware stop acts |

**Required work:** WP-P3-01, WP-P3-02, WP-P3-03, WP-P3-06, WP-P4-01. **Physical evidence:** M-52, M-51; the roadmap's
P3 acceptance ("unsafe or stale commands never reach the tractor service").

### D.7 I2C bus hang (PARTIALLY IMPLEMENTED)

**Requirement.** A hung, shorted or absent actuator bus is detected in bounded time, never stalls the stop path, is
reported, and drives the machine to the safe state by a mechanism that does not use that bus. The bus the actuator
driver uses is not shared with sensors whose faults it cannot tolerate (D-1). Bus recovery happens only in a safe
state.

**Baseline.** A write fault propagates once the chip was seen (`pca9685_driver.py:170-171`;
`tests/unit/test_pca9685_driver.py`). But a chip absent at boot degrades to a silent no-op, so every lever command
and the e-stop reply success (`:117-126,155-156`; N-07); the write runs in `asyncio.to_thread` with no timeout, so a
hung transaction can stall the e-stop sequence for the kernel's timeout (N-28 [INF, DS]); there is no read-back of
`MODE1`, `MODE2` or the pulse registers, no retry or re-initialisation, and `hardware_ok` is not surfaced in the
tractor state; the header I2C bus does not exist on the deployed Pi (`dtparam=i2c_arm=off`, GPIO3 is the HaLow WAKE
line: C-05); and the configuration puts the PCA9685, INA3221, ToF, BME280 and OLED on one bus
(`config/tractor.yaml:18`, `spec/hardware.yaml:146-151`), so a sensor fault can wedge the actuator driver.

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Chip absent at boot (wiring, power, wrong bus) | silent no-op; the API and the e-stop report success | fault state; arming refused; the e-stop reply names the failure |
| SDA or SCL stuck mid-mission | an exception or an unbounded wait | bounded detection; the hardware path cuts servo power; operator alert |
| A single failed write | exception to the caller | counted; consecutive failures fault |
| Another device wedges a shared bus | not considered | separate bus, multiplexer or coprocessor ownership (D-1, D-3) |
| PCA9685 resets (brown-out) | outputs off until the next write [DS]; unnoticed | a read-back mismatch faults |
| Pi reboot | outputs undefined during boot | a hardware-defined default (R-P10, M-21) |

**Required work:** WP-P0-01, WP-P1-03, WP-P1-04, WP-P1-08, WP-P1-01. **Physical evidence:** M-20, M-21, M-28;
acceptance section 3 (physically interrupt the bus during an e-stop).

### D.8 Servo stuck-output (MISSING)

**Requirement.** A servo or channel that does not follow its command is detected by position or current feedback, not
by a log timestamp (acceptance section 4), and its power can be removed regardless of what the software believes. A
blocked lever cannot overheat the servo, the harness or the supply (R-P01 to R-P04). The mechanical load path is
qualified before the servos are ever powered.

**Baseline.** `ServoActuator.command` records the command before the write (`tractor_actuators.py:65-69`) and the
lever fields of `TractorState` are commanded values. There is no position or current sensing, no stall detection, no
per-servo power switch and no read-back. What a servo does with no pulse is unmeasured (M-21, M-27). Supply, stall
current, fuse and gauge are unknown (A.5). Mechanical gate: the mount documents say that no part of the servo
mounting or adapter is qualified for powered steering, that a single ball-hole bolt does not give positive torque
restraint, and that powered steering is not approved until an anti-rotation connection and the force path are
verified (`hardware/mounts/README.md:406-414`, `hardware/mounts/INSTALL.md:240-244`).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Servo jammed or stalled | nothing detects it | detected; servo power cut; fault latched; stall current inside the fuse budget |
| PCA9685 channel stuck at a pulse | no read-back | a mismatch faults and cuts servo power |
| Servo electronics fail full-travel | not considered | the hardwired chain cuts servo power; travel stops in the linkage |
| Lever blocked at an e-stop | neutral is commanded anyway | the power cut removes drive force; measured with the lever blocked (M-27) |
| Linkage or hitch fails | unqualified | engineering qualification before any powered test (WP-P0-07) |
| PTO or starter relay welds | not considered | PTO state feedback (D.2); a series contactor |

**Required work:** WP-P0-04, WP-P0-07, WP-P1-01, WP-P1-12, WP-P1-03. **Physical evidence:** M-10, M-11, M-22, M-25,
M-27; acceptance section 4.

### D.9 GPS/RTK loss (PARTIALLY IMPLEMENTED)

**Requirement.** Fix quality, accuracy estimate, correction age and antenna health are evaluated continuously.
Autonomy needs the fix quality the owner and the reviewer require (SD-4); losing it produces an actuator-level HOLD
within a bounded time, not a mode flag. Geofence margins are sized from the worst-case measured position error.
Recovery needs hysteresis and a fresh fix. Base, gateway or link outages have a defined result and a documented
fallback.

**Baseline.** NMEA parsing exists for four receiver modes including `LC29H_UART` (`gps_driver.py`,
`models/sensor_data.py:37-43`), the NTRIP client forwards a public caster's RTCM to a serial device
(`ntrip_client.py`) and `GPSDegradationMonitor` exists. Gaps: the monitor acts only on
`RobotState.navigation_mode == AUTONOMOUS`, whose only writer is the debug route, while missions and the AI loop use
another enum and state, and its "fix age" is the age of the whole state (N-31); when it fires it flips an in-memory
flag and no actuator; mission start has no fix precondition and the waypoint loop does not read GNSS quality
(`navigation_service.py:175-238`); `MissionService.create_mission` holds the geofence validation as a comment
(`mission_service.py:19-22`); `config/limits.yaml:11` sets a 0.5 m geofence buffer with no link to measured error; no
LC29H `PAIR`/`PQTM` configuration, correction-age metric, antenna-fault handling or local base exists (A.7, H-09,
H-22).

| Event | Baseline behaviour | Required behaviour |
|---|---|---|
| Fixed to float | no effect (the monitor needs the debug mode and an accuracy above 5 m) | policy by accuracy against the geofence margin: speed cap or hold |
| No fix, antenna fault or jamming | no actuator effect | HOLD within a bounded time; alarm |
| Correction age grows (base, gateway, link) | not measured | a limit from M-51; treated as a degraded fix |
| Base outage | no local base exists | a documented fallback to the caster and its fix policy (WP-P3-06) |
| Implausible jump (multipath, survey error) | no check | cross-check against IMU and odometry; speed limit |
| Mission start without a fix | allowed | refused until the required fix has been held for a period set from field data |

**Required work:** WP-P2-04, WP-P2-05, WP-P4-03, WP-P3-06. **Physical evidence:** M-50, M-51 (time to fix,
transitions, correction age, base off, poor sky view, antenna failure).

### D.10 Physical bench acceptance (MISSING)

**Requirement.** Before any powered actuation beyond the bench, every item of `docs/tractor-acceptance-criteria.md`
(sections 1 to 11) is demonstrated on the machine with measured values logged with date, operator and result, and an
independent reviewer signs; unit tests never stand in. Test conditions: engine off, PTO and blade disconnected,
machine immobilised with wheels clear, a safety observer present (B.7). The roadmap's P1 says engine-off tests while
acceptance sections 1 and 2 say engine running; this document reconciles them by ordering: the engine-running steps
come last and only after the reviewer approves the engine-off results.

**Baseline.** The procedure is written and its 11-row sign-off table is empty. No instrumentation or harness exists;
the only HIL-named test runs a SIM probe (`tests/hil/test_hil_probe.py`); every passing test runs on fakes (A.11); the
mount documents' stop-work gates are open (D.8).

| Step | What | Acceptance section | Measurements | Stop criterion |
|---|---|---|---|---|
| BA-0 | Inspection and photographs; nothing powered | none | M-01, M-33, M-53 | any unknown label or part |
| BA-1 | Rail and passive measurements, actuators disconnected | 4 (supply) | M-10 to M-17 (no load), M-20 | a rail outside the design |
| BA-2 | Servos on a current-limited bench supply, linkage off | 4 | M-10, M-11, M-25 | stall above the fuse budget |
| BA-3 | Linkage fitted, engine off: self-centring and settle times | 10, 4 | M-22, M-24 | levers do not return; settle time over the requirement |
| BA-4 | Fault injection with the hardwired chain: Pi off, hung, killed; bus broken; heartbeat cut | 1, 3, 7 | M-21, M-23, M-26, M-28, M-29 | any fault leaves a lever or the PTO driven |
| BA-5 | PTO relay and clutch circuit, blade disconnected | 2 | M-17, M-24, M-30 | PTO state not observable |
| BA-6 | Interlocks and sequences: reverse, PARK, seat cutoff observation, manual start | 8, 9, 11 | M-31, M-32 | any OEM protection weakened |
| BA-7 | Engine running, immobilised: the e-stop's three effects and per-actuator timing | 1, 2, 4 | M-24 | the reviewer has approved BA-0 to BA-6 |

**Required work:** WP-P0-07, WP-P1-13, WP-P6-03. The sign-off table in the acceptance document stays the single
record.

### D.11 Safety items outside the ten topics

| Item | Status | Where | Required work |
|---|---|---|---|
| Operator authentication and authorisation | PARTIALLY IMPLEMENTED; off in the deployed unit | A.9; N-10, N-16, N-17 | WP-P0-03, WP-P1-11 |
| OEM interlock preservation (seat, PARK, Reverse Operation System, PTO switch) | documentation only | A.10 row 12; H-06, H-07 | SD-5, WP-P0-07 |
| Safe startup and shutdown | PARTIALLY IMPLEMENTED; shutdown MISSING | A.10 rows 8 and 9; N-08, N-09, N-10 | WP-P1-05, WP-P1-06 |
| Tilt cutoff | PARTIALLY IMPLEMENTED; defeated by the mission loop | A.10 row 13; N-06 | WP-P1-10, WP-P2-01 |
| Obstacle stop and stopping distance | MISSING for actuation | A.10 rows 14 and 15; N-11, N-25 | WP-P4-01, WP-P4-02 |
| False assurance (stop routes, latency tests, comments) | present | A.10 row 3; N-01 to N-03, N-19, N-24 | WP-P0-08 |

## Part E. Prioritised development plan (P0 to P6)

Priorities follow the owner's roadmap (P0 to P6; the sub-workstreams P0a, P2a, P3a, P4a and P5a are folded into the
work packages that name them). Each work package (WP) names the files it changes, the tests that must exist, the
observable acceptance and the gaps it closes. The plan is written against baseline `765c737`. Three parallel
branches (`audit/checker`, `audit/safety`, `audit/thorlink`) implement parts of P0 to P3; this document does not
claim what they delivered: the integrator records it under [Implemented in this PR](#implemented-in-this-pr) and
re-marks the affected rows. Paths are existing files unless marked "new". A test addition extends the named existing
test file where one exists. Nothing in this plan authorises powered actuation.

### E.1 Sign-off gates

```mermaid
flowchart LR
  P0["P0 freeze assumptions"] --> G0["G0 design freeze"]
  G0 --> P1["P1 bench interface"] --> G1["G1 bench sign-off"]
  G0 --> P2["P2 sensing"] --> G2["G2 sensing"]
  G0 --> P3["P3 comms"] --> G3["G3 comms"]
  G1 --> P4["P4 navigation"]
  G2 --> P4
  G3 --> P4
  P4 --> G4["G4 autonomy"]
  G1 --> P5["P5 optional VR"]
  G3 --> P5
  G4 --> P5
  P5 --> G5["G5 remote driving"]
  G1 --> P6["P6 field validation"]
  G4 --> P6
  P6 --> G6["G6 operations"]
```

Every gate needs the owner and an **independent** safety reviewer (a qualified person who did not author the change).
A gate is a human decision on recorded evidence; passing tests never close one.

| Gate | Closes | Evidence required | Signed by | Unlocks |
|---|---|---|---|---|
| G0 | P0 | signed netlist and pin matrix; measured supply rails; risk register; independent stop-test procedure; every P0 gap closed or explicitly blocked (roadmap P0 acceptance); a mechanical-qualification statement for the linkage and hitch | owner, independent reviewer | building the P1 stop chain and powering actuators on the bench under the B.7 conditions (engine off, PTO and blade disconnected, machine immobilised, observer). It does not permit machine motion |
| G1 | P1 | acceptance sections 1 to 4, 7 and 10 demonstrated with measured values in the sign-off table; no unsafe motion on any injected fault; OEM protections functional (roadmap P1 acceptance) | owner, independent reviewer, a second software reviewer for WP-P1-03 to WP-P1-11 | release gate 1 (no live motion) for supervised, restricted tests only, and only for H-01 to H-08 each closed or blocked with evidence |
| G2 | P2 | recorded datasets, time sync, health alarms, measured detection zones and stop margins; no autonomy on a single unverified sensor (roadmap P2 acceptance) | owner, independent reviewer | contributes to release gate 2 |
| G3 | P3 | repeated disconnect and reconnect fault tests with logs; unsafe or stale commands never reach the tractor service (roadmap P3 acceptance) | owner, independent reviewer | contributes to release gates 2 and 3 |
| G4 | P4 | replay and simulation, adversarial sensor loss, manual override, supervised blades-disabled tests (roadmap P4 acceptance); the ROS 2 / Nav2 decision recorded | owner, independent reviewer | contributes to release gate 2 |
| G5 | P5 (optional) | measured end-to-end latency and jitter, loss-of-video and loss-of-control tests, no motion on dead-man release (roadmap P5 acceptance) | owner, independent reviewer | release gate 3 (remote VR driving) |
| G6 | P6 | reproducible test records, rollback procedure, maintenance schedule, as-built wiring, operator handbook (roadmap P6 acceptance) | owner, independent reviewer | the only gate that can close release gate 2, and only for the scope the sign-off names |

### E.2 P0 - freeze dangerous assumptions (documents, decisions, guards, measurements)

Goal: nothing is powered until the unknowns that can injure someone or damage the machine are closed or explicitly
blocked. Rollback: documents and guards only; revert the commits.

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P0-01 | Record D-1 to D-5 and SD-1 to SD-8; one platform-tagged pin and bus registry for the Toro build; retire the legacy ultrasonic map from the registry | `spec/hardware.yaml`, `config/hardware.yaml`, `config/tractor.yaml`, `docs/hardware-integration.md` | the checker run (WP-P0-02) | decisions signed; every header pin has one owner; `gpioinfo` with the HAT fitted matches (M-02) | H-02, H-03, H-11, C-01 to C-08, C-14 |
| WP-P0-02 | Extend the checker: HaLow and overlay awareness, physical pins, boot configuration, `limits.yaml` claims, platform scope; reviewed conflicts in an allow-list. In flight on `audit/checker` | `scripts/check_hardware_pin_conflicts.py`, `spec/known_conflicts.yaml` (new), `.github/workflows/hardware-guard.yml` | new `tests/unit/test_check_hardware_pin_conflicts.py`: the starter on GPIO5, ultrasonics on GPIO9/10/11, a duplicate physical pin and a duplicate I2C address each fail | the B.1.3 what-if probes all report a conflict; CI fails on them | H-01 to H-03 |
| WP-P0-03 | Gate the hardware self-test (no pin driving by a probe, chip discovery, `ultrasonic_enabled` honoured); disable the injection and debug routes and the default credentials unless `SIM_MODE=1` | `backend/src/services/hw_selftest.py`, `backend/src/api/routers/maintenance.py`, `backend/src/api/navigation.py`, `backend/src/api/routers/sensors.py`, `backend/src/api/routers/auth.py`, `backend/src/api/rest_v1.py`, `backend/src/services/auth_service.py` | new unit tests: the probe claims no GPIO when the flag is false; the injection routes return 404 with `SIM_MODE=0`; no default credential authenticates | `GET /system/selftest` on the Pi leaves SPI0 untouched (scope SCLK) | N-16 (part), N-17, N-18, C-09, C-13 |
| WP-P0-04 | Power measurements and the signed power tree and fuse schedule | new wiring schedule under `docs/` | none (measurement) | M-10 to M-17 recorded; each branch has a measured stall budget and a fuse and gauge chosen from it; R-P01 to R-P13 each answered | H-05, H-17, H-20 |
| WP-P0-05 | Stop-mechanism design: E-stop chain, watchdog circuit and window, neutral-on-fault option (SD-1, SD-2) from the measurements | `docs/tractor-acceptance-criteria.md` (independent stop-test procedure), `hardware.md` | none | M-21, M-22, M-27, M-28, M-29 recorded; the reviewer signs the design and the test procedure | H-04, H-07, H-08, N-29 |
| WP-P0-06 | Trace the OEM PTO circuit and design a fail-safe OFF; decide the `blade_pto` model | `docs/tractor-platform.md`, `config/tractor.yaml` (schema, after the trace) | none (measurement) | M-30 recorded; a drawing the reviewer signs | H-21, N-26 |
| WP-P0-07 | Hazard analysis and risk register by a qualified person; OEM interlock decision (SD-5); mechanical qualification of the servo linkage and hitch anti-rotation | new risk register under `docs/`; the stop-work gate in `hardware/mounts/INSTALL.md:240-244` is recorded, not edited | none | the register covers every Part D topic; the stop-work gate is closed by an engineering statement, not by a hand fit | H-06, H-07, H-08 |
| WP-P0-08 | Make the stop API and its tests truthful | `backend/src/api/rest.py`, `backend/src/api/safety.py`, `backend/src/services/blade_service.py`, `config/limits.yaml` (comment), `tests/integration/test_safety_systems.py`, `docs/OPERATIONS.md` | new tests: `/control/emergency-stop` stops the RoboHAT and the IBT-4 and reports what failed; the block is latched, not 0.2 s; the "latency" tests are renamed to what they measure | no route claims an outcome it did not cause | N-01 to N-03, N-19, N-24 |
| WP-P0-09 | As-built visual package (P0a) and documentation closure | photographs, labelled placement, mount overlay; the A.12 open rows in `docs/deployment-guide.md`, `PRODUCT.md`, `spec/hardware.yaml` | `scripts/check_docs_drift.sh` made meaningful (WP-P6-03) | every visual traceable to a measured part (roadmap P0a gate) | H-24, N-23 |

### E.3 P1 - bench-only tractor interface

Goal: a stop chain that works without the Pi, and a software stack that fails loudly, proven on the bench. Rollback:
keep `tractor.enabled: false` (shipped default, `config/tractor.yaml:11`); every software change is inert behind
that flag; hardware: disconnect the actuators and the servo supply; revert the commits.

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P1-01 | Build and bench the E-stop chain and the servo-power cut; the PTO coil supply runs through the same chain | the wiring schedule (WP-P0-04); `hardware/mounts/estop_bracket.scad` for fit only | bench: M-24, M-29 | with the Pi off, hung and unplugged, servo power and PTO supply open; times per actuator recorded | H-07, H-08 |
| WP-P1-02 | Safety coprocessor, only if D-2, D-3 or SD-2 choose it: owns `OE`, the servo-power relay driver, the heartbeat, the e-stop state input, optionally the ultrasonic timing | new firmware directory beside `robohat-rp2040-code/`, new `backend/src/services/safety_coprocessor_service.py`, `backend/src/drivers/actuators/pca9685_driver.py` | host side with a fake serial port: heartbeat loss, protocol errors, reset; bench: reset state of every output | outputs default safe in reset and brown-out; its own watchdog behaviour measured | H-04, H-08, H-19 |
| WP-P1-03 | PCA9685 driver: `OE` control, `MODE1`/`MODE2` and pulse read-back, bounded I/O, an explicit fault when the chip is absent in production, re-initialisation | `backend/src/drivers/actuators/pca9685_driver.py`, `config/tractor.yaml` | extend `tests/unit/test_pca9685_driver.py`: an absent chip with `SIM_MODE=0` faults; a blocking fake bus times out; a read-back mismatch faults | M-20, M-21; acceptance section 3 on the bench | N-07, N-28, H-04 |
| WP-P1-04 | Tractor service: a fail-loud stop path (per-step results and a degraded status, guarded relay, state recorded after the write succeeds, lever pair applied with rollback, no silent simulation fallback in production) | `backend/src/services/tractor_service.py`, `backend/src/drivers/actuators/tractor_actuators.py` | extend `tests/unit/test_tractor.py`: a failing relay, a failing channel and both together each produce a failure report and never a false neutral state | acceptance sections 2 and 3 on the bench | N-27, N-07, N-03 |
| WP-P1-05 | E-stop latched across restarts and cleared only locally; safe shutdown (park, PTO off) and safe boot defaults | `backend/src/services/tractor_service.py`, `backend/src/api/routers/tractor.py`, `backend/src/main.py`, `systemd/lawnberry-backend.service` (`ExecStopPost`) | a restart with the latch set; teardown parks; the `SIGTERM` path; bench M-26 for `SIGKILL` | the machine comes up stopped; relay pins default OFF with the Pi unpowered, booting and killed | N-09, N-10, R-P10 |
| WP-P1-06 | Remove the mission auto-crank; an operator-confirmed "engine running" input (a human starts the engine with the levers in PARK) | `backend/src/services/navigation_service.py`, `backend/src/services/tractor_service.py` | extend `tests/unit/test_tractor_navigation.py`: a mission start never pulses the starter | acceptance section 11 | N-08, N-26, H-06 |
| WP-P1-07 | PTO per the traced circuit, with state feedback; momentary or latching selected by configuration | `backend/src/services/tractor_service.py`, `config/tractor.yaml` | fakes for each OEM behaviour: off must be observed, not assumed | acceptance section 2 (PTO drop measured) | H-21, N-26 |
| WP-P1-08 | Watchdog: fed by end-to-end evidence, re-armed, fail-closed startup, window from M-43; correct the validator | `backend/src/safety/watchdog.py`, `backend/src/safety/tractor_safety_monitor.py`, `backend/src/safety/safety_validator.py`, `backend/src/main.py` | new tests: re-arm after a trigger; a failed tractor initialisation refuses actuation routes; a broken command path stops the heartbeat | acceptance section 7; M-28 | N-04, N-05, N-29 |
| WP-P1-09 | Server-enforced command lease and expiry for every manual command; the RoboHAT host loop stops refreshing non-neutral commands | `backend/src/api/routers/tractor.py`, `backend/src/services/tractor_service.py`, `backend/src/services/robohat_service.py`, `backend/src/api/rest.py`, `backend/src/services/motor_service.py` | a fake clock: a command expires and the server cannot refresh it | M-52, M-24 | N-14 |
| WP-P1-10 | Tilt: latched against re-engagement by the mission loop; IMU loss fails closed | `backend/src/safety/tractor_safety_monitor.py`, `backend/src/services/tractor_service.py`, `backend/src/safety/safety_triggers.py` | extend `tests/unit/test_tractor_safety_monitor.py`: the mission loop cannot re-engage while tilted; no IMU reading holds the blade off | M-42 | N-06, N-24 |
| WP-P1-11 | Operator authentication and a local arming act on every actuating and safety route; WebSocket helpers verify tokens | `backend/src/api/mission.py`, `backend/src/api/routers/ai_control.py`, `backend/src/api/safety.py`, `backend/src/api/deps.py`, `backend/src/api/routers/auth.py`, `backend/src/api/routers/telemetry.py`, `systemd/lawnberry-backend.service` | extend `tests/integration/test_operator_authz.py` to the tractor, mission, AI and clear routes; remove the TDD skips | an unauthenticated client reaches no actuating route | H-14, N-10, N-16 |
| WP-P1-12 | Actuator feedback and stall detection: lever position and servo-branch current; fault latch and servo-power cut | a new driver under `backend/src/drivers/sensors/`, `backend/src/services/tractor_service.py` | fakes: a stalled or non-following servo faults | M-10, M-25, M-27 | D.8 |
| WP-P1-13 | Bench harness and HIL tests; execute acceptance sections 1 to 4, 7 and 10 and sign the table | new tests under `tests/hil/` behind `RUN_HIL=1`, the sign-off table in `docs/tractor-acceptance-criteria.md` | the HIL tests skip without the rig and log their results | G1 | H-01 to H-08 |

### E.4 P2 - localisation and obstacle sensing

Goal: every sensor the safety and navigation path will trust exists as a driver with health, timestamps and measured
coverage. Rollback: new drivers are off by default (the pattern of `ultrasonic_enabled: false`); revert per driver.

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P2-01 | BNO085 driver on the chosen transport (D-4); calibration status; mounting frame and axes | `backend/src/drivers/sensors/bno085_driver.py`, `backend/src/services/sensor_manager.py`, boot configuration (`uart4-pi5` if UART) | unit tests on recorded byte streams; the stub's zeros can no longer read as a level machine | M-42 recorded; tilt-to-blade-off latency measured | H-03, N-06, N-13 |
| WP-P2-02 | JSN-SR04T driver (mode from M-40), four-sensor model, time-division firing, a blind-zone reading means unknown, staleness and outlier handling, health | new `backend/src/drivers/sensors/jsn_sr04t_driver.py`, `backend/src/models/mower_data_frame.py`, `backend/src/services/sensor_manager.py`, `config/hardware.yaml` | new `tests/unit/test_jsn_sr04t_driver.py`: blind zone, crosstalk schedule, stale data fail closed | measured detection zones (WP-P2-08) | H-10, H-19, H-02 |
| WP-P2-03 | VL53L0X: units fixed against `limits.yaml` (mm versus m), interrupt decision, health | `backend/src/drivers/sensors/vl53l0x_driver.py`, `backend/src/services/navigation_service.py` | a unit test of the unit conversion and the threshold | M-41 | H-03 |
| WP-P2-04 | LC29H-DA rover and LC29H-BS base: receiver configuration, RTCM source and injection path, correction-age and fix-transition metrics, survey procedure | `backend/src/drivers/sensors/gps_driver.py`, `backend/src/services/ntrip_client.py`, a new base-side script under `scripts/`, `docs/gps-ntrip-setup.md` | recorded NMEA and RTCM replays: transitions, age, antenna fault | M-50, M-51 | H-09, H-22 |
| WP-P2-05 | Sensor health model and a common time base: timestamp, age, validity and fault code per sensor in telemetry; alarms | `backend/src/services/sensor_manager.py`, `backend/src/services/websocket_hub.py`, `backend/src/models/sensor_data.py` | stale and lost sensor tests | health alarms demonstrated | roadmap P2 |
| WP-P2-06 | Obstacle data model and fusion contract: extrinsics, confidence, staleness; replaces the placeholder detector | new `backend/src/services/obstacle_fusion.py`, `backend/src/services/navigation_service.py` | replay datasets; `tests/contract/test_sensor_fusion.py` made real | no autonomy on a single unverified sensor | H-26, N-11 |
| WP-P2-07 | Stereo depth evaluation (P2a), LiDAR selection (H-15) and a ROS 2 / Nav2 spike on recorded data | new scripts under `scripts/`; no safety-path code | an evaluation report with accuracy and latency against measured ranges | documented accuracy and latency; input to G4 | H-15, H-16, H-26 |
| WP-P2-08 | Hazard-class coverage tests and recorded datasets: children and pets, low objects, negative obstacles, deck overhang, tall grass, rain, sunlight, off-plane objects (B.5) | field data; replay under `tests/` | measured detection zones and stop margins | M-40, M-43, M-44, M-45 | H-10, H-26 |

### E.5 P3 - reliable Pi-to-Thor communications

Goal: an authenticated, bounded, loss-tolerant link whose failure leaves the mower in HOLD. Rollback: the link
service starts only when configured; revert the commit; the mower stays in safe hold without it.

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P3-01 | Authenticated link: an HMAC with a key held outside git, an epoch or boot id, a source check, binding to the HaLow address, bounded rates; an attacker cannot block legitimate datagrams | `backend/src/models/thor_link.py`, `backend/src/services/thor_link_service.py`, a `config/` template | extend `tests/unit/test_thor_link.py`: spoofed source, bad MAC, replay, huge `seq`, Thor restart | RS-2 to RS-5 and RS-8 satisfied | H-13, N-15 |
| WP-P3-02 | Wire the service into `main.py`; `/tractor/*` and the control routes consult hold; hold reports each step and retries the blade | `backend/src/main.py`, `backend/src/services/thor_link_service.py`, `backend/src/services/tractor_service.py` | new tests for the lifespan wiring and for a failing `engage_blade` during hold | M-52 | H-13, N-30 |
| WP-P3-03 | Plan validation: age, `plan_id`, `start_index` against the waypoint index, geofence and no-go re-validation, speed and blade permission; only then a consumer of `current_plan()` | `backend/src/nav/geofence_validator.py`, `backend/src/services/navigation_service.py`, `backend/src/services/thor_link_service.py` | adversarial plans never reach `TractorControlService` (roadmap P3 acceptance) | G3 | H-23, RS-6 |
| WP-P3-04 | Uplink builder (GNSS, detections, battery, e-stop, link counters) inside the bandwidth budget | `backend/src/services/thor_link_service.py`, `backend/src/models/thor_link.py` | a byte-budget test against `worst_case_link_bytes_per_second` | measured at the far yard (M-52) | H-23, H-16 |
| WP-P3-05 | HaLow identity and encryption on both ends, with versioned units (bring-up, failover, AP side) | `systemd/`, `docs/autonomy-toolchain-matrix.md` | none (configuration review) | identity and encryption verified from both ends | H-22, RS-1 |
| WP-P3-06 | Loss and rejoin fault tests (radio power cycle, dead link, latency, stale plan) and the failure analysis of base, Pi Zero, Orange Pi, radio and Thor with measured correction age (P3a) | `scripts/`, `docs/` | a fault-injection harness with logs | repeated tests pass; each node failure has a defined result | M-51, M-52, RT-5 |

### E.6 P4 - navigation, autonomy and AI

Goal: a deterministic local stop path that the AI and the Thor cannot lengthen. Rollback: the stop path is additive
(it only reduces speed or stops); revert the commit.

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P4-01 | Deterministic obstacle stop path and a speed cap from detection range and stopping distance; stale or blind readings count as blocked; independent of the AI and the Thor; applied inside `TractorControlService.apply` so the navigation loop cannot bypass it | new `backend/src/safety/obstacle_stop.py`, `backend/src/services/tractor_service.py`, `backend/src/services/navigation_service.py`, `backend/src/safety/safety_triggers.py` | replay per hazard class; adversarial sensor loss; blind-zone and stale cases | M-43; stop margins measured | H-26, N-11, N-25 |
| WP-P4-02 | Stopping-distance model and limits from measurement | `config/limits.yaml`, `backend/src/models/safety_limits.py`, `backend/src/models/motor_control.py` | monotonicity and boundary tests | M-43 at several speeds and slopes | H-26 |
| WP-P4-03 | Mission preconditions (fix quality held, geofence validated, sensor health, operator attestation) and a GNSS-quality HOLD at the actuator level; replace the inert monitor | `backend/src/services/navigation_service.py`, `backend/src/services/mission_service.py`, `backend/src/nav/gps_degradation.py`, `backend/src/main.py` | extend `tests/unit/test_gps_degradation.py` and `tests/unit/test_tractor_navigation.py`: a fix loss during a mission holds the levers | M-51 | N-31, H-09 |
| WP-P4-04 | AI output to twin-lever mapping; Hailo identity; on-device policy throughput and accuracy | `backend/src/services/ai_inference_service.py`, `backend/src/services/navigation_service.py`, `backend/src/models/tractor_control.py` | mapping tests; a hardware measurement | M-54; `hailortcli fw-control identify` | H-17, H-23 |
| WP-P4-05 | Supervised blades-disabled series, adversarial replay, manual override; the ROS 2 / Nav2 decision (G4); optional containerised portability (P4a) | `docs/field-validation-protocol.md`, `scripts/` | replay and simulation reports | roadmap P4 acceptance | H-26, H-28 |

### E.7 P5 - optional stereo VR and manual driving

Goal: remote driving only behind authentication, a server-enforced dead-man and a local override. Rollback: the
control channel is disabled by default.

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P5-01 | Authenticated control channel with a heartbeat and a server-enforced dead-man; one controller at a time | `backend/src/api/routers/telemetry.py` (`/ws/control`), `backend/src/api/deps.py`, `backend/src/services/tractor_service.py` | dead-man release stops motion; a second controller is refused | roadmap P5 acceptance | H-14, H-16 |
| WP-P5-02 | Stereo streaming, pan/tilt and head tracking; codec and latency budget from the bandwidth arithmetic (C.5) | `backend/src/services/camera_stream_service.py`, `backend/src/drivers/sensors/stereo_camera_driver.py`, new pan/tilt hardware | measured end-to-end latency and jitter | M-44, M-52 | H-16 |
| WP-P5-03 | Local obstacle interlocks and loss-of-video and loss-of-control tests in remote mode; video is never the only safety feedback | `backend/src/safety/obstacle_stop.py` | fault tests | no motion on dead-man release | H-14, H-16 |
| WP-P5-04 | Multi-agent rules (P5a): separate worktrees, draft PRs and review; PR #16 and PR #17 inspected before any use | `tools/agent_bridge/**` exists only on PR #17 (not merged here) | none | no agent drives mower hardware | H-27 |

### E.8 P6 - field validation and operations

| WP | Work | Files | Tests | Acceptance | Closes |
|---|---|---|---|---|---|
| WP-P6-01 | Supervised commissioning and the field series (tractor addendum tests 6 and 7, plus new ones) | `docs/field-validation-protocol.md` | recorded runs | reproducible test records | roadmap P6 |
| WP-P6-02 | Observability: e-stop state, GNSS quality, sensor health, power, network, Thor plan age, actuation faults, authentication events | `backend/src/services/websocket_hub.py`, `backend/src/api/routers/telemetry.py`, `frontend/src/` | alarm tests | alarms demonstrated | roadmap P6 |
| WP-P6-03 | CI: hardware-in-the-loop (manual, with the rig), a scheduled soak, a safety test job, secret scanning, systemd-versus-deployed drift, a docs-drift check that sees `config/`, `hardware/` and firmware | `.github/workflows/`, `scripts/check_docs_drift.sh` | the jobs themselves | CI fails on the cases in A.13 | N-22 |
| WP-P6-04 | Deployment hygiene: installer user substitution, versioned drop-ins and HaLow units, `.env`, the CORS origin, a listener audit | `systemd/install_services.sh`, `systemd/`, `backend/src/middleware/security.py` | `systemd-analyze verify` plus a comparison with the deployed units | the repository equals the deployment | N-21 |
| WP-P6-05 | Rollback procedure, maintenance schedule, as-built wiring update, operator handbook | `docs/` | none | G6 | roadmap P6 |

## Part F. Remaining hardware conflicts and unresolved safety requirements

Everything below is open at this baseline. Documentation corrections made by this PR are noted; none of them changes
a configured pin, an actuator calibration or `tractor.enabled`.

### F.1 Hardware conflicts that remain (C-01 to C-14)

| ID | Conflict | State after this PR | Closes with |
|---|---|---|---|
| C-01 | Legacy ultrasonic GPIO9/10/11 are the HaLow SPI0 lines | open; the guide is marked legacy and do-not-wire; the self-test still drives the pins (N-18) | WP-P0-01, WP-P0-03, M-02 |
| C-02 | Starter relay GPIO5 is the HaLow RESET line | open; `config/tractor.yaml` unchanged; documents warn; no relay may be attached (M-03) | WP-P0-01 |
| C-03 | Blade IN2 GPIO25 is the HaLow IRQ line | open (mower variant, driver stub) | WP-P0-01 |
| C-04 | GPIO12 is both the ToF-R interrupt and BNO085 TXD4 | open | D-4, WP-P0-01 |
| C-05 | Header I2C is disabled and GPIO3 is the HaLow WAKE line; the actuator bus has no route | open; this blocks the whole actuator bus | D-1, M-04 |
| C-06 | `uart4` versus `uart4-pi5` | open here; the documentation fix is on `audit/checker` | D-4, WP-P2-01 |
| C-07 | `rpi4_alt_pin` physical pins read as BCM | open here; documentation fix on `audit/checker` | WP-P0-01 |
| C-08 | ToF interrupts reserved, no reader | open | WP-P0-01 |
| C-09 | The self-test drives GPIO4/27/11 regardless of the flag | open code defect | WP-P0-03 |
| C-10 | Legacy guide: pin 17 is 3V3, not 5 V | documentation corrected in this PR; the physical rail still to be metered | M-02, M-14 |
| C-11 | The HaLow HAT may occupy more pins (reported, unverified) | open | M-02 |
| C-12 | PCA9685 0x41 in the INA3221 strap family | open | M-16, M-20 |
| C-13 | The header GPIO chip is `gpiochip0` here; code assumes 4 | open code defect | WP-P0-03 |
| C-14 | No GPIO for `OE`, the watchdog heartbeat or the e-stop state input | open | D-3, WP-P0-05 |

### F.2 Unresolved safety requirements

| ID | Requirement | Why it is unmet at the baseline | Closed by | Gate |
|---|---|---|---|---|
| S-01 | An independent hardwired E-stop | no circuit, input or GPIO | WP-P0-05, WP-P1-01 | G1 |
| S-02 | Neutral-on-fault proven physically | self-centring unknown; neutral is a command | WP-P0-05, WP-P1-01, WP-P1-02 | G1 |
| S-03 | PTO fail-safe OFF with the circuit traced and the state observed | momentary versus latching unresolved; no feedback | WP-P0-06, WP-P1-07 | G1 |
| S-04 | A watchdog independent of the process and fed by end-to-end evidence | software liveness only | WP-P1-08 | G1 |
| S-05 | Bounded I2C failure handling | silent no-op; unbounded writes | WP-P1-03 | G1 |
| S-06 | Stuck-output detection and a servo-power cut | none | WP-P1-12, WP-P1-01 | G1 |
| S-07 | A server-enforced command lease | commands persist | WP-P1-09 | G1 |
| S-08 | E-stop cleared and arming done locally and authenticated | remote clear; authentication off | WP-P1-05, WP-P1-11 | G1 |
| S-09 | A latched e-stop; safe boot and shutdown | none | WP-P1-05 | G1 |
| S-10 | No auto-crank; a human starts the engine | missions crank | WP-P1-06 | G1 |
| S-11 | Tilt latched; a real IMU that fails closed | stub; defeated by the mission loop | WP-P1-10, WP-P2-01 | G1, G2 |
| S-12 | A measured and fused power tree | no ratings anywhere | WP-P0-04 | G0 |
| S-13 | Relay outputs default OFF at power-up and boot | unknown | WP-P0-04, WP-P1-05 | G0, G1 |
| S-14 | OEM interlocks preserved; the seat and PARK decision made by a qualified reviewer | SD-5 undecided | WP-P0-07 | G0 |
| S-15 | Mechanical qualification of the servo linkage and hitch | the stop-work gate is open | WP-P0-07 | G0 |
| S-16 | An obstacle stop path independent of the AI and the Thor, with a speed cap | advisory only | WP-P4-01, WP-P4-02 | G4 |
| S-17 | Four-sensor ultrasonic coverage measured | no driver | WP-P2-02, WP-P2-08 | G2 |
| S-18 | An authenticated Thor link and local plan re-validation | none | WP-P3-01 to WP-P3-03 | G3 |
| S-19 | A GNSS-quality HOLD at the actuator level | the monitor is inert (N-31) | WP-P4-03, WP-P2-04 | G2, G4 |
| S-20 | A truthful stop API and honest tests | false assurance (N-01 to N-03, N-24) | WP-P0-08 | G0 |
| S-21 | Bench acceptance signed | the sign-off table is empty | WP-P1-13 | G1 |
| S-22 | Remote VR driving behind a dead-man and authentication | no control channel | WP-P5-01 | G5 |

### F.3 Open decisions

| ID | Decision | Decided by | Needed before | Inputs |
|---|---|---|---|---|
| D-1 | I2C route for the PCA9685 | owner | any actuator wiring | M-04, B.1.4 |
| D-2 | Ultrasonic architecture (A, B or C) | owner | the JSN driver | M-40, M-02 |
| D-3 | Which device owns the relays, `OE` and the watchdog heartbeat | owner and reviewer | WP-P0-05 | M-21 to M-29 |
| D-4 | BNO085 transport | owner | WP-P2-01 | M-42 |
| D-5 | HaLow HAT occupancy | owner | any GPIO assignment | M-01, M-02, M-53 |
| SD-1 | Whether the E-stop also stops the engine; whether a positive parking brake is needed | reviewer | WP-P0-05 | M-43, the hazard analysis |
| SD-2 | The neutral-on-fault mechanism | reviewer | WP-P0-05 | M-21, M-22, M-27 |
| SD-3 | Behaviour on loss of Thor or link: hold or a bounded local continuation; the resume rule | owner and reviewer | WP-P3-02 | M-43, M-52 |
| SD-4 | Required fix quality per phase; geofence margins | owner and reviewer | WP-P4-03 | M-51 |
| SD-5 | How the seat and presence switch is handled for unoccupied operation | a qualified reviewer | G0 | M-31, the OEM manual |
| SD-6 | The command-lease window and speed caps | reviewer | WP-P1-09, WP-P4-02 | M-43 |
| SD-7 | Who may clear an E-stop and how | owner and reviewer | WP-P1-05 | none |
| SD-8 | The authentication posture per platform (differential-drive LAN-only versus the Toro build) | owner | WP-P1-11 | a threat model |

### F.4 Release gates at this baseline

| Release gate | State | Blocking items |
|---|---|---|
| 1. No live motion while H-01 to H-08 are unresolved | **CLOSED** | H-01 documentation corrected, physical rail check open; H-02 open (live SPI0 conflict); H-03 open; H-04 documentation corrected, no `OE` or E-stop design; H-05 24 V text superseded, label and stall unmeasured; H-06 documented, contradicted by the mission auto-crank; H-07 unresolved (SD-5); H-08 unresolved (S-01 to S-06) |
| 2. No autonomous mowing until sensor and RTK health, a physical fail-safe, operator protections and field acceptance are proven | **CLOSED** | S-01 to S-21; no obstacle stop path; no JSN driver; the BNO085 is a stub; no local base |
| 3. No remote VR driving until authentication, a dead-man, measured latency and a local override are proven | **CLOSED** | S-07, S-08, S-22; no control channel; no latency measurement (M-52) |

### F.5 Limits of this audit

- The deployed Pi was inspected read-only. `i2cdetect` and the `pinctrl` CLI were not run and most sensors are
  detached at that site, so absence of a device is not evidence of a design error.
- No bench instrument was used and no photograph was taken. Owner-reported values (the two 150 kg·cm servos, 10 to
  12.6 V, 8 A or more stall) are unverified.
- Only simulation tests were run. Parallel branches are not reflected; line numbers are against baseline `765c737`.
- Statements tagged [DS] or [INF] are datasheet knowledge or inference from reading code; none was executed.
- `codeql.yml`, `pr-hygiene.yml`, `todo-policy-check.yml`, the frontend beyond its route usage and the training and
  workshop code were not audited.

## Appendix. Gap register index (H-01 to H-28)

The register is in draft PR #18 (`https://github.com/khryptorgraphics/lawnberry-pi/pull/18`, branch
`docs/2026-10-24-hardware-architecture-audit`, file `docs/2026-10-24-compatibility-gap-register.md`). H-02b and
H-02c are sub-findings of H-02 introduced by this audit.

| ID | P | Register finding | State at this baseline | Where |
|---|---|---|---|---|
| H-01 | P0 | Ultrasonic guide says pin 17 is 5 V | documentation corrected in this PR; the rails are still to be metered | A.12, B.1 (C-10), M-02, M-14 |
| H-02 | P0 | Ultrasonic GPIO10/11/9 collide with the HaLow radio | confirmed live (SPI0 alt0); guide marked legacy; the self-test still drives the pins | A.4, B.1 (C-01, C-09), WP-P0-01 to WP-P0-03 |
| H-02b | P0 | Starter relay GPIO5 is the HaLow RESET line (sub-finding) | open; no relay may be attached | B.1 (C-02), WP-P0-01 |
| H-02c | P0 | Blade IN2 GPIO25 is the HaLow IRQ line (sub-finding) | open (mower variant, driver stub) | B.1 (C-03), WP-P0-01 |
| H-03 | P0 | GPIO12 double booking; UART4 mapping | open; `uart4` versus `uart4-pi5` found (N-13) | A.3, A.4, B.1 (C-04 to C-08), D-4, WP-P2-01 |
| H-04 | P0 | PCA9685 `OE` polarity | acceptance text corrected in this PR; no `OE` allocated; mechanism undecided | D.3, D.7, WP-P0-05, WP-P1-03 |
| H-05 | P0 | 12 V versus 24 V servo rail | 24 V text superseded in this PR; label, supply and stall unmeasured | A.5, B.3, M-10, WP-P0-04 |
| H-06 | P0 | Lever linkage cannot reach PARK; no autonomous start | documented constraint; the mission auto-crank contradicts it (N-08) | A.10, D.11, WP-P1-06 |
| H-07 | P0 | Seat and presence switch versus unoccupied travel | unresolved; a qualified review is required | D.1, D.11, SD-5, WP-P0-07 |
| H-08 | P0 | PCA9685 holds the last PWM; no independent failsafe | unresolved; Part D is the safety case | Part D, F.2, WP-P0-05, WP-P1-01 to WP-P1-13 |
| H-09 | P1 | GPS receiver mismatch (ZED-F9P versus LC29H) | annotated as unreconciled in this PR; owner decision and physical check open | A.7, B.6, D.9, WP-P2-04 |
| H-10 | P1 | Three sensors in the guide, four required | guide marked legacy; the four-sensor model is missing | A.1, B.5, WP-P2-02 |
| H-11 | P1 | Differential-drive and Toro variants mixed | platform tags added in README, matrix, overview, servo-hat-pins and hardware.md; `spec/` and `docs/hardware-integration.md` are on the checker branch | A.3, A.12, WP-P0-01 |
| H-12 | P1 | `enabled: false` default | confirmed in the repository and live | A.3 |
| H-13 | P1 | Thor link documented | exists as a library, unwired and unauthenticated (the register's direction is corrected) | A.8, C.3, D.6, WP-P3-01 to WP-P3-03 |
| H-14 | P1 | Authentication off by decision | confirmed; more unauthenticated routes found | A.9, D.5, WP-P1-11, SD-8 |
| H-15 | P2 | LiDAR unspecified | no model selected | B.6, C.6, WP-P2-07 |
| H-16 | P2 | Stereo and VR workflow | capture only; bandwidth arithmetic in C.5 | A.6, C.5, C.11, WP-P5-01, WP-P5-02 |
| H-17 | P1 | The stack competes for header, PCIe, USB and power | inventoried; photographs and measurements open | B.4, B.6, M-01, M-14, M-54 |
| H-18 | P1 | Divider schematic inconsistent | legacy guide corrected; not a validated divider for the JSN hardware | A.12, M-40 |
| H-19 | P0 | Four JSN-SR04T required | driver missing; electrical interface unverified | A.1, B.5, WP-P2-02, M-40 |
| H-20 | P0 | Two 150 kg·cm servos, high stall current, fuse and gauge | unmeasured; no ratings anywhere | A.5, B.3, WP-P0-04 |
| H-21 | P0 | Momentary PTO versus a latching relay | contradicted in code; trace first; do not wire | D.2, WP-P0-06, WP-P1-07 |
| H-22 | P1 | Pi Zero base and Orange Pi gateway | unverified in the repository; failure analysis open | A.7, B.2, C.4, WP-P3-05, WP-P3-06 |
| H-23 | P1 | Thor as primary navigation compute | target; strategist in simulation; link unwired | A.6, C.1, WP-P3-03, WP-P4-04 |
| H-24 | P1 | Visual package | not produced; it needs photographs and measurements | B.5, WP-P0-09 |
| H-25 | P1 | HAT stacking blocks signals | unverified | B.1 (C-05, C-11), M-01, M-02 |
| H-26 | P1 | No measured fused obstacle model | missing | A.6, B.5, C.8, WP-P2-06, WP-P4-01 |
| H-27 | P2 | Multi-agent automation, PR #16 | PR #16 and PR #17 are drafts, not merged; nothing under `backend/` | A.6, C.12, WP-P5-04 |
| H-28 | P2 | HPE server migration | future optional deployment | C.12, WP-P4-05 |

## Implemented in this PR

Part A and Part E describe baseline `765c737`. This section records what this PR changes on top of it; where a row
below disagrees with Part A, this section is the current state. Nothing here was exercised on mower hardware: every
item is software behaviour proven in `SIM_MODE` against fakes. Physical stopping, neutral and power removal remain
unverified (gates G0-G1).

| Branch | Commits | What changed | Gaps advanced or closed |
|---|---|---|---|
| `audit/docs` | `f4a7c52` `9371078` `79a545a` `81fe90b` `302a867` `45c2e1b` `3883f36` `17d4378` | Ultrasonic guide marked legacy / DO NOT WIRE, pin 17 = 3.3 V, divider fixed; PCA9685 `OE` active-low corrected; 24 V servo rail superseded (12 V-class pending label); feature matrix and README corrected; platform tags; this document | H-01, H-04, H-05, H-18 (documentation); H-10/H-19 (documented, not designed); H-11 |
| `audit/checker` | `65a189c` `bb2edda` `4c6de02` | `spec/hardware.yaml` gains HaLow HAT, UART0/UART4, ultrasonic and I2C1 claims, each with evidence and status, plus a `deployment` block from the live Pi; checker rewritten (txd/rxd, tractor relays, ultrasonic only when enabled, I2C availability); reviewed allow-list `spec/known_conflicts.yaml` that fails on new or stale entries and on tractor conflicts once `enabled: true`; `check_docs_drift.sh` no longer vacuous; guard workflow runs checker and tests | H-02, H-03, H-17, H-25 now **detected and gated** (not resolved) |
| `audit/safety` | `3a9dfdf` `fe3ea6d` `f70d315` `615bb90` `2095702` | Truthful per-subsystem E-stop on `/control/emergency*` (`services/emergency_stop_service.py`); authenticated `emergency_clear`; PCA9685 absent/failed raises and tractor motion commands are rejected; tractor command deadman (`command_deadman` in `config/tractor.yaml`, 500 ms default, 2 s ceiling) returns levers to neutral and throttle to idle; RoboHAT commands expire, no blind PWM resend; tilt/E-stop latch blocks blade re-engagement until an authenticated clear; shutdown parks actuators; repo unit sets `OPERATOR_AUTH_REQUIRED=1`, `admin/admin` alias removed | H-08 (software side), H-14, loss-of-comms, false-assurance E-stop |
| `audit/thorlink` | `6cc2076` `2deae0f` `dc3941f` | HMAC-SHA256 datagrams (`THOR_LINK_KEY`, refuses to start without a key outside SIM); Pi session id + uplink echo replaces `seq`, so replays and seq poisoning are rejected and Thor restarts are accepted; 1.0 s plan age; bounded size and waypoint count; optional geofence; failed safe-hold neutral surfaced as `hold_fault`. Still **not wired** into `main.py` | H-13 (protocol hardened), H-23 (unchanged: not live) |

### Status changes against Part A

| Part A row | Baseline | After this PR |
|---|---|---|
| General E-stop routes | STUBBED / false assurance | PARTIALLY IMPLEMENTED: invokes tractor, RoboHAT and blade service; reports `stopped/simulated/unconfirmed/failed/not_present`; IBT-4 GPIO is still a stub so blade reports `unconfirmed` outside SIM |
| Independent hardwired E-stop | MISSING | MISSING (hardware work, G0) |
| PCA9685 `OE` / servo-power cut | MISSING | MISSING in hardware; absent-PCA9685 is now reported instead of silent success |
| Loss of comms (operator) | MISSING | SIMULATED: tractor deadman and RoboHAT command expiry, tested with fakes |
| Tractor safety monitor | PARTIALLY IMPLEMENTED | PARTIALLY IMPLEMENTED: blade re-engagement after tilt now blocked by a latch; watchdog still event-loop liveness only |
| Safe shutdown | MISSING | SIMULATED: lifespan teardown parks actuators (tested through the real `lifespan`) |
| Operator authentication | disabled in shipped unit | Enabled in the repo unit; **the deployed Pi still runs `OPERATOR_AUTH_REQUIRED=0` until the operator installs the unit and sets credentials** |
| Pi-Thor link authentication | MISSING | IMPLEMENTED (protocol, tests); not wired |
| Replay and stale-plan protection | PARTIALLY IMPLEMENTED | IMPLEMENTED (protocol, tests) |
| Pin and address conflict checker | PARTIALLY IMPLEMENTED | IMPLEMENTED: reports GPIO5, GPIO25, GPIO12, GPIO3 conflicts and the disabled header I2C bus; tested |

### Verification (integrated branch)

| Check | Command | Result |
|---|---|---|
| Full suite | `SIM_MODE=1 pytest tests -o addopts="" -p no:cacheprovider --ignore=tests/soak -W ignore` | 762 collected: **693 passed, 69 skipped, 0 failed, 0 errors** (baseline 530 passed, 69 skipped) |
| Pin checker | `python scripts/check_hardware_pin_conflicts.py` / `--self-test` | exit 0; 4 allow-listed conflicts (GPIO 25, 12, 5, 3), 2 deployment warnings, 4 unverified-overlap warnings; `self-test OK` |
| Lint | `ruff check backend tests scripts workshop`; `ruff format --check` on changed files | All checks passed; 36 files already formatted |
| Docs drift | `bash scripts/check_docs_drift.sh` | `code_changed=true, docs_changed=true` (non-vacuous) |

The suite rewrites the tracked `verification_artifacts/001-integrate-hardware-and/telemetry_integration_evidence.json`
on every run; it was restored and is not part of this PR.

### Finish track deploy

Branch `finish/deploy` (goal T3: deployment and security hygiene, repo side). Software behaviour proven by tests that run
the real scripts against temp directories (`--dry-run`, `--destdir`); nothing was installed on the mower Pi, which was
unreachable (last row). Test counts are in the final summary of the PR.

| Item | What changed | Evidence | Rows advanced |
|---|---|---|---|
| Units are templates | `User=pi`, `Group=pi`, `chown -R pi:pi` and `/apps/lawnberry-pi` replaced by `@LB_USER@`, `@LB_GROUP@`, `@LB_ROOT@` in every unit (camera keeps `Group=video`); unit descriptions no longer carry a version | `tests/unit/test_install_services.py::test_repo_units_carry_no_hard_coded_account_or_checkout` | A.14 "Installer", N-21, A.12 "Version naming" |
| Installer renders and installs | `systemd/install_services.sh --user/--group/--root/--destdir/--dry-run`; account defaults to `$SUDO_USER`, then the checkout owner, never root; values validated before they reach `sed`; explicit unit manifest plus a drift test; `<unit>.d/*.conf` drop-ins are rendered and installed; the removed `lawnberry-sensors.service` is disabled and deleted on upgrade | 43 tests in `test_install_services.py`: rendering, modes, manifest drift, drop-ins, unknown-token abort, `systemd-analyze verify` of every rendered unit | A.14, N-21 |
| `--dry-run` | prints every rendered unit and action, writes nothing, needs no root; its exit status is what a real run would return (0 ok, 3 refused) | `test_dry_run_*` | T3 acceptance |
| Persistent signing secret | `scripts/provision_env.py`: `LAWN_BERRY_AUTH_SECRET` (64 hex) generated once into `<root>/.env` (0600), kept across installs, replaced only by `--rotate-auth-secret`; a short existing secret is refused; atomic write, every other line untouched, never printed | `test_provision_env.py` (41), `test_the_signing_secret_is_generated_once_and_persists_across_installs` | A.9 "JWT secret persistence", N-21 |
| Credential refusal | the installer exits 3, with nothing installed or written, for a missing, empty, published-default or weak operator credential (under 12 characters, under 5 distinct characters, or a default word plus digits or punctuation), whether it comes from the environment, `--operator-credential-file` or an existing `.env` | `test_default_weak_or_empty_credentials_refuse_the_install` (8 cases), `test_no_credential_at_all_refuses_and_says_how_to_fix_it`; mutation checks (refusal ignored, default list emptied, minimum length 3) fail the suite | A.9 "Default credentials", N-17 |
| No built-in credential in code | `AuthService` and `/api/v1/auth/login` no longer fall back to `operator123`; unset or a published default disables operator login (fail closed, E-stop routes unaffected, CRITICAL log without the value); `JWTManager` warns once about an ephemeral or weak signing key; `backend/src/core/credential_policy.py` is the single policy; the suite configures an explicit credential | `test_auth_no_default_credential.py` (18), `test_credential_policy.py` (27) | A.9, N-17 |
| Manual-control unlock was fail-open | `POST /api/v2/control/manual-unlock` with `method: password` accepted any non-empty password when neither a bcrypt `password_hash` nor `MANUAL_CONTROL_PASSWORD` was configured (the default), and the session it returned passes `/control/drive` and `require_operator_auth`, i.e. it bypassed the operator credential for every operator-gated write (found by `FinishJsn.IntegrationPlaceholders`, confirmed by probe). `_validate_manual_password` now checks the password against `password_hash`, else `MANUAL_CONTROL_PASSWORD`, else the operator credential (`AuthService.verify_operator_credential`), and against nothing when none is usable; a blank password is a 400 at every security level. `OPERATOR_AUTH_REQUIRED=0` and `SIM_MODE` behave as before | `tests/integration/test_manual_unlock_fails_closed.py` (8; two mutations, fail-open and blank accepted, fail 3 tests each) | A.9 "Routes with no auth dependency" / "WebSocket authorisation" family, N-16; new finding |
| Placeholder unit | `lawnberry-sensors.service` (`ExecStart` = echo + `sleep infinity`) removed; backup/restore scripts and docs drop it; guard: no unit's `ExecStart` may be a placeholder | `test_no_unit_runs_a_placeholder_command` | A.1 "Sensor-manager systemd unit", N-20 |
| Unit defects found by verifying timers | `lawnberry-cert-renewal.timer` had `Documentation=README.md` (invalid URI); `lawnberry-acme-renew.timer` fired at 00:00 and 00:12, not 12:00 | `test_installed_units_have_valid_systemd_syntax`; `systemd-analyze calendar` | A.13 |
| CI | `config-lint` verifies the units the installer renders, timers included, and shellchecks the installer | `.github/workflows/config-lint.yml` | A.13 "config-lint" |
| Documentation (A.12) | `deployment-guide.md`: ports 8081/3000, unit names, installer, E-stop and interlock claims as audited, "Names and versions"; `OPERATIONS.md`: `/control/emergency-stop` described as the software E-stop it is; `PRODUCT.md`: auth enforced, Toro variant, safety and fusion statements; `authentication-config.md`: "emergency123" and the default credential removed; README API intro, installation guide | A.12 rows below | A.12 |
| Live-only HaLow units | **Not done, blocked on owner action O-8.** `halow-bringup`, `halow-failover`, `halow-lan-routes.{service,timer}`, `wifi-halow-*`, `wificommand`, the `wpa_supplicant@halow0` drop-in and `lawnberry-backend.service.d/thor-upload.conf` were not read: on 2026-10-24 `ssh kp@10.8.0.2` gave `ssh: connect to host 10.8.0.2 port 22: Connection timed out`, `ssh kp@192.168.50.20` gave `ssh: connect to host 192.168.50.20 port 22: No route to host`, and the AP forward `ssh -p 2222 kp@192.168.1.140` gave `No route to host` while the AP itself answered ping (2/2). Reading them from the workshop box is not possible and nothing is invented. The installer already carries `<unit>.d/*.conf` drop-ins; once the Pi is reachable run (read-only) `ssh kp@<pi> 'systemctl cat lawnberry-backend.service wpa_supplicant@halow0.service halow-bringup.service halow-failover.service halow-lan-routes.service halow-lan-routes.timer wificommand.service; systemctl list-units --all "wifi-halow-*" --no-legend'`, strip credentials (PSK/SAE), add the files under `systemd/` and to the installer manifest (the drift test fails until it is). | `ssh` output above | A.14 "HaLow units" and "Backend unit vs deployed" stay UNVERIFIED |

| A.12 row | After this track |
|---|---|
| Ports 8000/3001 vs 8081/3000 in `docs/deployment-guide.md` | Closed: 8081 (API) and 3000 (UI), `ws://.../api/v2/ws/telemetry` |
| `/control/emergency-stop` listed as an e-stop (`docs/OPERATIONS.md`) | Closed: described as the software E-stop with its per-subsystem reply, not as a hardware stop |
| Version naming v2 / v3 | Closed for the units: descriptions carry no version; the README "v3" (project generation), `/api/v2` (API) and constitution 4.0.0 are documented as separate axes in `deployment-guide.md`. Whether the product should be renamed is the owner's call |
| `PRODUCT.md:25` Craftsman tractor | Closed: Toro TimeCutter, bench-only, `enabled: false` |
| `docs/authentication-config.md` "emergency123" | Closed: no such feature; the real recovery is the installer |

Deployed Pi after merge (T4, not done): `OPERATOR_AUTH_REQUIRED` stays `0` there and no `.env` exists until the owner runs
`sudo bash systemd/install_services.sh --operator-credential-file ...` with the credential from O-2.

Verification on `finish/deploy`: `env -u PYTEST_ADDOPTS SIM_MODE=1 pytest tests -o addopts="" -p no:cacheprovider --ignore=tests/soak -W ignore`
collected 899: **830 passed, 69 skipped, 0 failed, 0 errors** (baseline 762 / 693 / 69 / 0; the 137 new tests are 43 installer, 41
`provision_env`, 27 policy, 18 auth and 8 manual-unlock), `ruff check .` clean, `shellcheck -x systemd/install_services.sh` clean,
`systemd-analyze verify` clean on every rendered unit. `ruff format --check .` still reports `hardware/mounts/build_printables.py`
and `workshop/isaac_sdg.py`, which this track did not touch.

Observed, not changed (outside this track; reported by `FinishJsn.IntegrationPlaceholders`, verified in `backend/src/api/rest.py`
`control_drive_v2`): `POST /api/v2/control/drive` without a `session_id` takes the legacy branch, which needs no operator
authentication even under `OPERATOR_AUTH_REQUIRED=1`. It does not call the RoboHAT, so nothing moves, but it returns computed motor
speeds, writes an audit row and sets the in-memory `_legacy_motors_active` flag that the blade interlock tests read. The legacy
integration tests depend on that contract, so closing it (reject, or require the operator gate) is a decision for the safety/API owner.
