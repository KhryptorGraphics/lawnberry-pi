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
- Review `config/tractor.yaml`, `docs/tractor-platform.md`, `docs/tractor-acceptance-criteria.md` and driver code. Note `enabled: false` is current default.
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
