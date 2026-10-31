# LawnBerry-Pi: proposed three-tier perception and controller migration plan

Status: DESIGN PROPOSAL, not tested on physical mower. This document records user-approved investigation direction, not verified integration or authorization to operate actuators.

## Hardware roles
- **Libre Computer Alta AML-A311D-CC (4 GB RAM, Amlogic A311D)**: candidate temporary mower computer while Pi 5 is unavailable; later a possible independent lightweight perception node. Integrated NPU advertised at ~5 TOPS; actual Linux driver, inference runtime, model compatibility, camera ingestion and thermal behavior MUST be validated. Four GB is a hypothesis, not a verified complete-stack RAM budget.
- **Raspberry Pi 5 (8 GB)**: intended return as primary mower Linux host and carrier for the user's **Raspberry Pi AI HAT+ Hailo-8L (~13 TOPS)**. HAT+ requires Pi 5 PCIe ribbon connector and appropriate firmware/driver stack; **not GPIO-only**.
- **NVIDIA Jetson AGX Thor**: high-level, networked perception and planning, not a single point of failure for stopping.
- **Raspberry Pi Zero 2 W at property base station**: proposed Wi-Fi HaLow network endpoint and host for USB-connected **u-blox ZED-F9P RTK base**. Validate Zero 2 W USB OTG hub/power budget, HaLow driver support, host CPU load and stable serial service before committing.
- **Orange Pi 4 Pro (4 GB)**: alternate temporary Linux controller target; exact board revision, OS, UART/I2C/GPIO pinout and drivers must be established before wiring.

## Critical physical compatibility
The Libre Alta manufacturer specifications do not identify the Raspberry Pi 5-style PCIe ribbon connector. **Do not connect AI HAT+ to Alta's 40-pin header or assume MIPI is PCIe.** The Hailo HAT+ stays on Pi 5 unless a supported alternate carrier and electrical/driver interface are independently proven. Inspect exact HAT+ revision and pinout.

## Functional architecture
1. Sensors/cameras publish timestamped observations, calibration metadata and health to local middleware.
2. Alta NPU runs a small supported detector (candidate only); Pi 5 Hailo runs a different validated model (e.g. semantic segmentation / secondary detection); Thor runs high-compute scene analysis. Models must be selected by supported compilation/runtime, not TOPS rating.
3. Local fusion process aligns frames/time, transforms coordinates, merges detections and uncertainties, records provenance and disagreement; a **single navigation arbiter** issues bounded commands to a separate actuator controller.
4. Independent safety controller, hardware E-stop, blade/PTO interlock, watchdog, obstacle proximity and communications-loss response override all AI/navigation decisions. No model or Thor message can bypass these safeguards. Loss of localization, sensor validity, model availability or fusion confidence causes a defined safe state; validate independently before any physical testing.
5. Thor inference is advisory and must never be required to stop. Use authenticated, encrypted links and explicit bandwidth/latency budgets. Wi-Fi HaLow may transport telemetry/corrections but does not replace local safety.

## Coding agent deliverables, in order
1. Audit current repository (including draft PRs #18, #19, #20), hardware inventory, pin maps and safety issues. Resolve contradictions; do not merge unreviewed PRs.
2. Create board capability matrix: Pi 5, Alta, Orange Pi, Zero 2 W; enumerate CPU, RAM, OS/kernel, camera, GPIO, I2C, UART, SPI, USB, PCIe, NPU, Ethernet, HaLow and RTK dependencies. Mark each as verified / pending / unsupported with evidence.
3. Introduce board-specific hardware abstraction/configuration, stable device identification, and runtime feature detection. No implicit Pi GPIO numbering on Alta/Orange Pi.
4. Create resource budget for Alta: baseline OS, ROS 2 if applicable, camera buffers, video encode/decode, NPU runtime, detector model, telemetry, RTK and safety processes. Record peak RSS, CPU, thermals and latency with concurrent workloads; establish headroom and OOM recovery.
5. Prove Alta NPU toolchain with a reproducible minimal model, compatible Linux image, model compilation, inference test, and documented accuracy/FPS/latency. If blocked, retain CPU-only or alternative host fallback.
6. Prove Pi 5 Hailo-8L AI HAT+ on a supported OS/driver/runtime with independent model; document hardware cabling, power and cooling.
7. Implement versioned message schemas for observations, detections, timestamps, transforms, confidence, health and navigation proposals; record and replay sensor data for deterministic simulation.
8. Integrate Thor advisory inference, including timeout, packet loss, network outage and conflicting-detection simulation; never let remote output directly control actuators.
9. Implement RTK base station from `docs/rtk-base-station-hardware-selection.md`: ZED-F9P fixed base -> USB -> Zero 2 W -> RTCM3/NTRIP over authenticated link -> rover receiver on mower. Confirm exact antenna, correction messages, datum, base coordinates, rover model and network topology before purchase/wiring.
10. Test safety cases in simulation: human/animal detection, sensor disagreement, stale frames, dropped corrections, loss of Thor/HaLow, runaway inference, watchdog expiration, reboot, low voltage and E-stop. Hardware testing requires separately reviewed wiring, independent stop circuitry and human sign-off.
11. Publish deployment/runbooks: image versions, installation commands, service units/containers, dependency lockfiles, diagnostics, logs, telemetry, rollback, benchmark results and acceptance checklist.

## Open decisions / blockers
- Confirm exact Alta revision, supported Linux image and working NPU driver/compiler.
- Confirm camera model/count, physical connections, and shared vs separate camera streams.
- Confirm AI HAT+ exact revision and Pi 5 PCIe power/cooling.
- Confirm whether Alta is temporary replacement or permanent secondary perception node.
- Confirm Zero 2 W USB HaLow device, host-mode compatibility, antenna/power budget, RTK antenna and rover model.
- Confirm safety controller, motor/PTO hardware and approved physical pin assignments.
- No assumption that 4 GB suffices without profiling; no guarantee of three simultaneous inference pipelines before validation.

Acceptance gate: demonstrable simulated perception/fusion, board-specific reproducible builds, bounded failure handling, and verified hardware compatibility; **no real blade or motor operation** based on this proposal alone.
