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
- [Implemented in this PR](#implemented-in-this-pr) (placeholder: the integrator fills it)

## 0. Scope, baseline and conventions

### 0.1 Baseline

| Item | Value |
|---|---|
| Repository baseline | `origin/main` at `765c737` (branch `audit/docs` adds documentation only) |
| Deployed Pi | host `pi5`, checkout `/apps/lawnberry-pi` at `23a6e4d`, kernel `6.12.21-morse+`; **read-only inspection on 2026-10-24**, nothing written, no actuator driven, no hardware config changed |
| Requirement documents | draft PR #18, branch `docs/2026-10-24-hardware-architecture-audit`: `docs/OMP-THOR-HANDOFF-2026-10-24.md`, `docs/2026-10-24-physical-integration-plan.md`, `docs/2026-10-24-development-roadmap.md`, `docs/2026-10-24-compatibility-gap-register.md`. They are **not on this branch**; they are cited by file name and treated as proposed requirements, not as evidence of implementation |
| Gap IDs | **H-01 … H-28** are the register's IDs. Findings the register does not contain carry **N-01 … N-28** (section A.15); they are proposed additions to the register. Conflicts are **C-xx** (B.1), measurements **M-xx** (B.7) |
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
| Tractor safety monitor and watchdog | PARTIALLY IMPLEMENTED | `backend/src/safety/tractor_safety_monitor.py:144-209` (20 Hz; feeds the heartbeat unconditionally at `:181`; tilt edge calls `engage_blade(False)` once, `:197-206`; IMU loss returns silently, `:131-141,193-194`); `backend/src/safety/watchdog.py:54-69` (one-shot: `break` after the first timeout) | `tests/unit/test_tractor_safety_monitor.py` (9), `tests/contract/test_watchdog.py` | H-08, N-04, N-06 |
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
| CORS | PARTIALLY IMPLEMENTED | `backend/src/middleware/security.py:191-230`: origin allow-list with credentials; `http://<pi-ip>:3000` is not in the defaults unless `LAWN_BERRY_ALLOWED_ORIGINS` is set (not set) | `tests/unit/test_security_middleware.py` | N-21 |
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
| HIL | **MISSING** | `tests/hil/test_hil_probe.py` runs `scripts/hil_probe.py` under `SIM_MODE` and parses a CSV; it is not a hardware test, and no CI job runs `tests/hil` or `tests/soak` |
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
| `sim-tests.yml` | push, PR | `uv sync --frozen`, `SIM_MODE=1 LBY_ACCEL=cpu uv run pytest -q`, an API smoke on port 8000 (`:16-26`) | Includes the 69 skips (auth TDD tests never run); no HIL, soak or dedicated safety job; port 8000 vs the deployed 8081 |
| `lint.yml` | push, PR | `ruff@0.15.19 check .` and `format --check .` (`:17-19`) | Not run by the audit |
| `hardware-guard.yml` | PR | the pin checker, then `grep` presence checks on `spec/hardware.yaml` (`:14-59`) | Checks the manifest, not reality; `:54-59` enforces CH1 battery / CH3 solar (N-12); `:39-44` blocks the strings "soil", "weather station", "jetson", "movidius" |
| `docs-drift.yml` | PR, push to main | `scripts/check_docs_drift.sh` | Path-regex only (`:40`); passes if any `docs/**` file changed; locally `git fetch --depth=0` errors and is swallowed (`:18`), so a local run on an unchanged branch is vacuous; does not look at `config/*.yaml`, `hardware/` or `robohat-rp2040-code/` |
| `config-lint.yml` | PR | yamllint, dotenv-linter, shellcheck via `ludeeus/action-shellcheck@master` (unpinned, `:14`), `systemd-analyze verify` with ExecStart-missing errors filtered | Does not compare units with the deployed Pi |
| `dep-audit.yml` | PR | `pip-audit` and `npm audit` | Both commands end in `\|\| true` (`:18,28`), so they never fail the job |
| `deps-lock.yml` | PR | `uv.lock` present and locked; forbids `pycoral`/`edgetpu` in `pyproject.toml` (`:22`) | none |
| `webui-build.yml` | PR | `npm run build` and Playwright e2e against a mocked backend (`:16-28`) | vitest unit tests are not run in CI |
| `codeql.yml`, `pr-hygiene.yml`, `todo-policy-check.yml` | various | not individually audited | UNVERIFIED |
| **Absent** | n/a | HIL, soak, secret scanning, a check that `systemd/` matches the deployed units, any deploy workflow | N-22 |

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

### A.15 Findings outside the PR #18 register (N-01 to N-28)

These are proposed additions to the gap register. Each cites the file and line (baseline `765c737`).

| ID | Finding | Evidence | Related H |
|---|---|---|---|
| N-01 | General E-stop routes do not actuate everything yet reply that motors and blade are stopped | `backend/src/api/rest.py:600-692` | H-08 |
| N-02 | The e-stop block on follow-up drive and blade commands lasts 0.2-0.3 s and is not latched | `rest.py:175-179,204-222,324,359,503,550,621-627` | H-08 |
| N-03 | The IBT-4 blade is never e-stopped; `BladeService.emergency_stop` and `IBT4BladeDriver.set_estop` have no callers | `blade_service.py:46-55`, `ibt4_gpio.py:103-106`, `rest.py:550-555` | H-08 |
| N-04 | The watchdog heartbeat is the monitor's own loop liveness; the watchdog is one-shot; its `EstopHandler` holds a private `MotorAuthorization` | `watchdog.py:54-69`, `tractor_safety_monitor.py:176-181`, `main.py:155-158` | H-08 |
| N-05 | If tractor initialisation raises, the actuation API stays live with no watchdog and no tilt monitor, and nothing alerts | `main.py:144-170` | H-08, H-12 |
| N-06 | Tilt cutoff: one-shot blade-off, re-engaged by the mission loop within about 200 ms [INF]; IMU loss is fail-open; on real hardware the IMU is a stub that reports zero roll and pitch | `navigation_service.py:354-371`, `tractor_service.py:218-231,304`, `tractor_safety_monitor.py:131-141,191-209`, `sensor_manager.py:186-200`, `bno085_driver.py:69-78` | H-08 |
| N-07 | A PCA9685 absent or unreachable at boot makes every lever command and e-stop a silent success | `pca9685_driver.py:117-126,155-156`, `tractor_service.py:157-172,199-203` | H-04, H-08 |
| N-08 | `execute_mission` cranks the engine whenever `engine_running` is false, and the state is OFF after any backend restart, so a mission start can pulse the starter on a running engine; it contradicts "a human starts the engine with the levers in PARK" | `navigation_service.py:188-195`, `tractor_service.py:109-110,246-250` | H-06 |
| N-09 | No safe shutdown: teardown does not park actuators, no `ExecStopPost`, relay pin state at exit is unknown | `main.py:184-201`, `systemd/lawnberry-backend.service:37-44`, `tractor_actuators.py:95-124` | H-08 |
| N-10 | E-stop is not latched across a restart; `clear-emergency` needs only the (off) operator auth, with no local acknowledgement | `tractor_service.py:109-110,290-293`, `routers/tractor.py:91-93` | H-08, H-14 |
| N-11 | Obstacle triggers are advisory: no production caller, thresholds used only by the debug injector, `go_to_waypoint` ignores obstacle state, the detector uses a `Position(0, 0)` placeholder | `safety_triggers.py:94-141`, `routers/sensors.py:474`, `navigation_service.py:33-82,240-372` | H-10, H-19, H-26 |
| N-12 | INA3221 channel map: spec and CI say CH1 battery / CH3 solar; driver and `config/hardware.yaml` say CH1 solar / CH3 battery | `spec/hardware.yaml:61-70`, `hardware-guard.yml:54-59`, `ina3221_driver.py:3-6,101`, `config/hardware.yaml:29-33` | none |
| N-13 | `dtoverlay=uart4` is the wrong overlay for the Pi 5 (needs `uart4-pi5`); `rpi4_alt_pin: 21/24` are physical pins read as BCM | `docs/hardware-integration.md` (baseline lines 68, 116), `spec/hardware.yaml:143-144`, LIVE overlay README | H-03 |
| N-14 | The RoboHAT host loop re-sends the last PWM about every 0.9 s, defeating the firmware's USB-silence failsafe; `duration_ms` is validated and ignored; `/tractor/command` has no TTL, lease or server dead-man | `robohat_service.py:246-269`, `rest.py:378-388`, `routers/tractor.py:67-70` | H-08, H-14 |
| N-15 | Thor link: any source address accepted, no HMAC or epoch, sequence-poisoning denial of service, hold is non-revoking, `tick` swallows actuation errors, `current_plan()` has no consumer | `thor_link_service.py:45-62,98-145` | H-13, H-23 |
| N-16 | Missions, AI enable, `emergency_clear`, debug injectors and `/system/selftest` have no auth dependency; WebSocket helpers accept any bearer or any `SIM_MODE` client | see A.9 | H-14 |
| N-17 | Hard-coded default credentials | A.9 | H-14 |
| N-18 | `hw_selftest.ultrasonic_probe()` claims and drives TRIG pins GPIO4/27/11 regardless of `ultrasonic_enabled`, with `gpiochip_open(4)` hard-coded | `hw_selftest.py:36-41,192-255`, `routers/maintenance.py:62-69` | H-02 |
| N-19 | `limits.yaml` says an ultrasonic reading "triggers e-stop" and no code does that; `SafetyLimits.ultrasonic_enabled` defaults to True | `config/limits.yaml:16-17`, `models/safety_limits.py:20` | H-02 |
| N-20 | The sensor-manager systemd unit is a placeholder (`sleep infinity`) and reports active | `systemd/lawnberry-sensors.service:16-17` | none |
| N-21 | Deployment drift: undocumented drop-in, unversioned HaLow units, `User=pi` vs `kp`, no `.env` (random JWT secret per restart), CORS defaults exclude the LAN origin | A.9, A.14 | H-11 |
| N-22 | CI blind spots: manifest-only hardware guard, path-regex docs drift, audits that cannot fail, no HIL, soak, safety or secret-scan job | A.13 | H-11 |
| N-23 | Documentation defects beyond those corrected here (A.12 "Open" rows) | A.12 | H-11 |
| N-24 | Unmeasured safety claims: "e-stop latency under 100 ms" times a flag flip; "tilt cutoff under 200 ms" times a function call | `tests/integration/test_safety_systems.py:14-45` | H-08 |
| N-25 | A frame with no ultrasonic data reads as an obstacle at 0 cm, which stops the AI path by accident rather than by design [INF] | `mower_data_frame.py:104-110`, `ai_inference_service.py:494-502`, `navigation_service.py:845-849` | H-26 |
| N-26 | Engine and PTO state are commanded, not sensed; `stop_engine` only sets a flag, there is no ignition-kill actuator | `tractor_service.py:246-256` | H-06, H-21 |
| N-27 | `set_levers` applies left then right with no rollback; `ServoActuator.command` records the state before the write; a relay failure can pre-empt servo safing | `tractor_service.py:196-197`, `tractor_actuators.py:65-69,95-115` | H-08 |
| N-28 | The PCA9685 write runs in a thread with no timeout, so a hung bus can stall the e-stop sequence for as long as the kernel I2C timeout allows [INF, DS] | `pca9685_driver.py:170-171`, `tractor_service.py:272-283` | H-04, H-08 |
