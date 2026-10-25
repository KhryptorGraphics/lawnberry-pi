# Interchangeable onboard SBC architecture — LawnBerry-Pi
Status: implementation specification, hardware validation pending. Updated 2026-10-25.

## Objective
Support Raspberry Pi 5 (8 GB), Orange Pi 4 Pro (4 GB), and Libre Computer Alta (4 GB) as selectable primary onboard computers without rewriting the autonomy application. These boards are NOT electrical or software drop-in replacements. Keep an identical *functional contract* while changing board support packages, device mappings, acceleration, and performance profiles.

## Hardware profiles
| Profile | RAM | Role | Hardware-dependent constraints |
|---|---:|---|---|
| raspberry_pi_5 | 8 GB | Preferred baseline; optional Raspberry Pi AI HAT+ (Hailo-8L) | HAT+ requires Pi 5 PCIe FFC link, appropriate driver/runtime, and power budget. |
| orange_pi_4_pro | 4 GB | Temporary or alternative main onboard host | Validate exact board SKU, OS image, pinmux, kernel, camera and available PCIe interfaces before selecting peripherals. No assumption that Pi HATs or Pi 5 pin numbers apply. |
| libre_alta | 4 GB | Alternative onboard host and possible edge inference | Record exact manufacturer SKU, SoC and NPU identity; confirm supported compiler/runtime and actual usable PCIe interfaces. Do not assume the integrated NPU is supported until a sample model runs. |

**Important:** Product names are insufficient to confirm connectors or acceleration. Require board photographs, PCB revision, manufacturer schematics and a working OS inventory. Do not wire motors or sensors from Raspberry Pi pinout tables on another board.

## Architecture: same interfaces, board-specific adapters
Application modules: `perception`, `localization`, `navigation`, `safety`, `motor_gateway`, `telemetry`, `network`, `health`. Publish versioned messages and timestamps; avoid hardcoded GPIO, camera names, UART paths, I2C bus IDs, and accelerators.

Provide `BoardSupport` interface with `discover_devices()`, `get_camera_stream()`, `get_gnss_port()`, `get_gpio_line()`, `get_accelerators()`, `read_health()`, `emergency_stop()`. Implement one adapter per board. Use libgpiod GPIO character-device interfaces where supported; map lines by chip/offset with tested electrical levels. Put PWM and time-critical motor control on a dedicated microcontroller with watchdog, rather than relying on Linux scheduling.

Configure via `BOARD_PROFILE=raspberry_pi_5|orange_pi_4_pro|libre_alta` and a per-board YAML file. Config fields: OS/kernel, architecture, device aliases, serial/I2C/SPI assignments, pin electrical limits, motor MCU link, camera driver/pipeline, GNSS receiver, RTK correction transport, network radios, accelerator providers, memory budgets, watchdog timeouts and safety interlocks. All profiles must pass schema validation at boot; unknown wiring disables actuation.

## Three-tier perception (conditional)
1. Lightweight always-on onboard inference on a *verified* NPU (Libre Alta only if its runtime is confirmed), otherwise CPU fallback.
2. Hailo-8L on Raspberry Pi 5 AI HAT+ when Pi 5 and supported runtime are available. The HAT+ is not assumed attachable to Orange Pi 4 Pro or Alta. For other hosts, use a separately supported accelerator interface only after compatibility tests.
3. NVIDIA Thor receives selected video/metadata, performs higher-cost perception/fusion/planning. Thor network loss must not defeat local stop/obstacle safety. If Thor unavailable, transition to reduced-speed local operation only after validation; otherwise stop safely.

All model results must include source, model version, confidence, capture timestamp, inference timestamp, frame/calibration ID, coordinate frame and health status. Fuse detections with stale-data rejection, disagreement handling and explicit priority rules. **Only one motion authority** issues commands; independent hardware E-stop and safety watchdog can inhibit actuators. Never interpret an NPU model's output as a direct motor command.

## RAM/performance rules
4 GB systems must run a lean OS, minimal containers/services, bounded camera buffers, no large local vision-language model and no local training. Profile RSS, peak memory, swap pressure, thermal throttling, frame drops, end-to-end latency and power during simultaneous camera + GNSS + networking + inference workloads. Establish a measured pass/fail envelope; do not claim equivalent throughput to Pi 5 until benchmarked. Reduce stream resolution/FPS and offload heavy models to Thor when necessary.

## RTK base station and Wi-Fi HaLow
Raspberry Pi Zero 2 W can act as a base-station host if USB/power/driver/throughput testing succeeds. Specify a survey-capable, dual-band, multi-constellation GNSS RTK receiver with documented base mode, raw observations, RTCM3 output, supported antenna, and stable USB serial interface; **exact model and antenna must be chosen and verified before purchase**. Fixed antenna needs sky visibility, weatherproof mounting, surge/grounding and surveyed or accurately established coordinates. Feed RTCM3 to rover over monitored IP transport (e.g. NTRIP); do not confuse GNSS corrections with navigation fixes. Log correction age, fix type, satellites and accuracy; stop precision operations if fix/correction thresholds are exceeded.

Wi-Fi HaLow is an IP transport, not a substitute for RTK GNSS or E-stop. Record radio model, regulatory region, frequency/channel, antenna, interface (USB/SPI/SDIO), Linux driver/kernel compatibility and throughput/latency under field conditions. Do not assume Pi Zero 2 W can host an arbitrary HaLow radio. Validate both sides, roaming/reconnection and recovery from outages.

## Hardware bring-up sequence
1. Inventory exact board SKU, revision, OS, kernel, power supply, peripherals and connectors; photograph both sides and save `hardware/inventory.yaml`.
2. Boot each candidate headless; verify network, storage, thermals, memory and system clock.
3. Verify camera capture and timestamp accuracy; GPS/RTK input and correction age; then each sensor bus individually.
4. Create verified per-board wiring table: signal, physical connector, voltage, chip/line or bus/address, direction, pull-up, isolation, current, test method. Never reuse Raspberry Pi BCM numbers across boards.
5. Test MCU motor gateway with wheels off ground; watchdog, E-stop, loss of heartbeat, stale commands and reboot behavior.
6. Benchmark local model providers and Thor link, including disconnect, jitter, thermal throttling and overload.
7. Test full autonomy in simulation, then controlled low-speed field trials with an accessible E-stop. Preserve test logs and signoff per board.

## Coding-agent work packages and acceptance criteria
A. Add typed board profiles, schema and boot-time validation; tests for invalid mappings.
B. Implement Linux device abstraction and per-board adapters; fake/sim adapters for CI.
C. Implement camera/GNSS/RTCM/HaLow health monitoring and reconnect policies.
D. Add inference-provider abstraction (`cpu`, `hailo8l`, `alta_npu_if_verified`, `thor_remote`) with capability detection and deterministic fallback.
E. Add single-authority command arbitration, MCU watchdog and hard E-stop checks.
F. CI matrix for all three profiles with mocked hardware; hardware-in-loop smoke tests for each actual board.
G. Document reproducible OS install, pinned dependencies, power budgets, connector diagrams, benchmarks and operator recovery.

**Done** means each board can boot the same app interfaces, discover only verified devices, publish consistent messages, tolerate remote inference loss, enforce stop behavior and pass its own hardware-in-loop checklist. Mark unsupported peripherals explicitly rather than silently claiming compatibility.

## Open decisions requiring physical validation
- Exact Libre Alta product identifier, NPU SDK and supported Linux image.
- Orange Pi 4 Pro kernel/driver and camera/PCIe capabilities.
- Actual sensor and motor-controller part numbers and pin wiring.
- Exact RTK base receiver/antenna, survey method and rover receiver.
- HaLow USB radio SKU, host driver support and regulatory configuration.
- Quantified CPU/RAM/FPS/latency/power budgets and safe degradation limits.
