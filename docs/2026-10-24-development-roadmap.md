# LawnBerry-pi — integrated development roadmap and acceptance gates
Date: 2026-10-24. **Planning document**; no claim that physical validation has occurred.

## Target architecture and authority
```
HOUSE: LC29H-BS RTK base -> corrections source -> secure HaLow link
HOUSE: NVIDIA Thor 128GB -> map/coverage strategy, optional heavy inference, training
                         | bounded authenticated plans; optional compressed sensor data
                         v
MOWER: HaLow adapter -> Raspberry Pi 5 + Hailo 8L
                       | fixed CSI camera, elevated USB stereo, GNSS rover, IMU,
                       | ToF, four ultrasonics, optional 2D LiDAR
                       v
                 local safety supervisor / sensor fusion
                       v
                 tractor actuation service -> PCA9685/relays -> Toro levers/PTO
INDEPENDENT: physical E-stop, safety watchdog, OEM protections, safe actuator mechanism
```
**Safety authority remains local**. Thor may suggest waypoints or model outputs but may never directly energize actuators, clear E-stop, or override Pi interlocks. If Thor or HaLow disappears, the mower follows a defined locally safe behavior; remote control requires an independently proven link-loss stop. Existing `docs/tractor-platform.md` already describes `backend/src/models/thor_link.py` and `backend/src/services/thor_link_service.py` as a compact UDP protocol, not an absent link. Verify current implementation and security rather than replacing it blindly.

## P0 — freeze dangerous assumptions (before any powered actuation)
- Inventory exact purchased hardware and identify installed vs planned modules.
- Resolve 12 V vs 24 V servo supply conflict, ultrasonic guide's 3.3 V/5 V header mistake, GPIO12 multiplex conflict, HaLow SPI vs ultrasonic GPIO10/11/9, PCA9685 OE polarity, and all pin reservations.
- Preserve OEM interlocks; define independent E-stop, positive parking/immobilization plan, physical neutral behavior and fail-safe on stuck PCA9685 outputs.
- **Acceptance:** signed netlist, measured supply rails, unambiguous pin matrix, risk register, independent stop test procedure, all P0 issues closed or explicitly blocked.

## P1 — bench-only tractor interface
- Review `config/tractor.yaml`, `docs/tractor-platform.md`, `docs/2026-10-30-primary-backup-software-control.md` and driver code. Note `enabled: false` is current default.
- Verify address 0x41, ch0 left lever, ch1 throttle, ch2 right lever, BCM5 starter control, BCM26 PTO control; confirm relay input polarity and physical connector labels.
- Instrument physical lever and throttle position; use engine-OFF, blades-disconnected, immobilized tests. Test power cycle, I2C hang, GPIO fault, watchdog timeout, E-stop, and no-PWM/servo-power loss.
- **Acceptance:** measured safe positions and stop timing per actuator; no unsafe motion on fault; all OEM safety protections functional; independent sign-off. Unit tests alone do not count.

## P2 — localization and obstacle sensing
- Integrate actual LC29H-DA rover and LC29H-BS base, record GNSS protocol and RTCM corrections; test correction age, fix transitions, antenna failure, base outage and poor sky view.
- Integrate BNO085, two VL53L0X, four angled ultrasonics and optional 2D LiDAR with timestamps, calibration and sensor health.
- Check 50-inch deck overhang, children/pets, low objects, negative obstacles and above/below-scan hazards; 2D LiDAR cannot independently cover them.
- **Acceptance:** recorded datasets, time sync, health alarms, measured detection zones and stop margins; no autonomy based on single unverified sensor.

## P3 — reliable Pi↔Thor communications
- Audit existing `thor_link.py` and `thor_link_service.py` before modifying protocol.
- Establish HaLow network identity, encryption/authentication, replay protection, rate limits and telemetry, loss/rejoin state machine, backpressure and bounded packet sizes.
- Explicitly test radio power-cycle recovery, dead link, high latency and stale Thor plan; onboard Pi maintains local stop authority.
- **Acceptance:** repeated disconnect/reconnect fault tests and logs; unsafe/stale commands never reach tractor service.

## P4 — navigation, autonomy and AI
- Define mower-local planner/control loop, RTK geofences, deck footprint and speed limits, model confidence and deterministic obstacle overrides.
- Use Thor for high-level coverage plans and optionally heavier inference only when transport latency and bandwidth support it; Hailo 8L remains onboard for essential perception.
- Reconcile differential-style AI action output with twin hydrostatic lever mixing, as identified in `docs/tractor-platform.md`.
- **Acceptance:** replay/simulation, adversarial sensor loss, manual override, and gated supervised tests with blades disabled before mowing.

## P5 — optional stereo VR/manual driving
- Specify pan/tilt mechanics, stereo USB camera formats, calibrated intrinsics/extrinsics, video codec and HaLow bandwidth budget, head-tracking, motion sickness/latency bounds and local fail-safe.
- Keep manual VR command authority behind authenticated operator controls, heartbeat, dead-man input and local obstacle interlocks; **never use the video path as the sole safety feedback**.
- **Acceptance:** measured end-to-end latency/jitter, loss-of-video and loss-of-control tests; no motion on dead-man release.

## P6 — field validation and operations
- Supervised, engine-off and PTO-off commissioning; then controlled operating tests only after safety sign-off.
- Add observability for E-stop, GPS quality, sensor health, power, network, Thor plan age, actuation faults and operator authentication.
- **Acceptance:** reproducible test records, rollback procedure, maintenance schedule, updated as-built wiring and operator handbook.

## Oh My Pi / Codex / Claude Code handoff
All three CLIs are installed on Thor. They can work on **separate Git branches** in the same repository, but do not let them simultaneously edit the same working tree. Prefer separate `git worktree` directories per agent. Assign OMP code inventory/tests, Codex safety and interface review, Claude Code documentation or isolated modules; exchange results via draft PRs and human review. No agent gets permission to drive live mower hardware merely by reading a handoff.

## Deliverable tracking
| Item | Current evidence | Status |
|---|---|---|
| Initial repo conflicts and OMP prompt | `docs/OMP-THOR-HANDOFF-2026-10-24.md` | Drafted |
| Wiring reconciliation | `docs/2026-10-24-physical-integration-plan.md` | Proposed, **not as-built** |
| Development roadmap | This document | Proposed |
| Actual pin measurements and purchased-parts audit | Operator + Thor bench inspection | **Open** |
| Independent physical safety sign-off | Acceptance protocol | **Open** |
| Implemented and tested LiDAR/VR/live Thor perception | Source and bench verification | **Open** |


## Recovered scope and architecture decisions
- **Thor is intended as the primary navigation/decision computer**, not just an offline training node. Pi 5 should offload high-level perception/route decisions to Thor when connected, while retaining immediate local sensor processing, motion safety supervision, actuator interlocks and emergency-stop authority. Clearly distinguish *owner target* from *implemented behavior*.
- The house communications design previously included a **Pi Zero RTK base and Orange Pi HaLow intermediary**. Confirm their actual installation, network links, power and service ownership before treating Thor as directly connected to the base station.
- Two **150 kg-cm-class 10–12.6 V servos** are reported for the left/right Toro lap bars, and a separate throttle actuator remains to be identified. Current documents describing a 24 V servo rail may be dangerous if used uncritically.
- Four **JSN-SR04T waterproof** front sensors are required; two outer sensors should angle outward on adjustable mounts to cover 50-inch deck edges. Do not use legacy HC-SR04 pin diagrams as the installed wiring.
- Owner explicitly requested **visual hardware documentation**: annotated real-part photographs, 3D/exploded views, a Toro front mounting overlay, physical GPIO/header riser map, and a complete end-to-end wiring trace to the house-side computers.
- Investigate **ROS 2/Nav2** as a possible mapping/navigation integration and assess stereo-based 3D obstacle mapping, without claiming either already exists.
- Account for the **momentary PTO control** reported by the owner; resolve its OEM electrical behavior versus the configured latching relay before actuation.

## Additional engineering workstreams and gates
**P0a — as-built visual package:** collect owner-supplied Pi/HAT/servo/relay/ultrasonic/RTK photographs and Toro front photograph. Produce a labeled placement plan, GPIO riser/cable map, per-connector wiring schedule, power/fuse chart, and a real-photo-based ultrasonic mount overlay. **Gate:** every visual is traceable to measured hardware; unknowns are explicitly labeled.

**P2a — spatial mapping feasibility:** bench-evaluate four JSN-SR04T cones, stereo depth, optional 2D LiDAR and RTK/IMU fusion; characterize weather, sunlight, grass, slopes, scan occlusion and blind spots. Decide whether ROS 2/Nav2 offers useful integration without sacrificing deterministic local safety. **Gate:** documented accuracy and latency; no claim of complete 3D awareness without measured coverage.

**P3a — house compute distribution:** document Pi Zero/Orange Pi/HaLow/Thor roles, GNSS correction flow, camera/depth telemetry bandwidth and what happens on each intermediate-node failure. **Gate:** measured throughput, latency, correction age and link recovery.

**P4a — future compute portability:** prior conversation mentioned an **HPE server transfer** and simulation/benchmarking. Treat this as a *future optional compute deployment* rather than an installed LawnBerry-pi component; define containerized services and hardware-agnostic interfaces first.

**P5a — multi-agent execution:** Codex CLI, Claude Code and Oh My Pi are all installed on Thor. Use separate Git worktrees and task ownership, simulations, draft PRs and review. A previously discussed GitHub task-queue/approval bridge (referenced as PR #16 in prior conversation) must be inspected for current status before assuming agent automation exists. Never auto-merge or deploy to the mower.
