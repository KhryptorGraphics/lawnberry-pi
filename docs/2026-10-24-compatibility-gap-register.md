# LawnBerry-pi compatibility and implementation gap register — 2026-10-24

This register separates **observed documentation/configuration contradictions** from **implementation not yet verified**. Findings are based on inspected repository files; they are not a claim of exhaustive source-code testing or live-hardware verification.

| ID | Priority | Finding and evidence | Resolution / verification | Status |
|---|---|---|---|---|
| H-01 | P0 | `docs/ultrasonic-wiring-guide.md` says physical pin 17 supplies 5 V; Pi header pin 17 supplies 3.3 V. | Correct guide, inspect supply rails with meter, review wiring before connection. | Confirmed doc error; open |
| H-02 | P0 | Ultrasonic guide uses GPIO10/11/9 while `PRODUCT.md` says ultrasound disabled due to HaLow GPIO conflict #15. | Inspect actual HaLow transport and occupied pins, redesign four-channel trigger/echo mapping or offload to microcontroller. | Confirmed conflict history; actual wiring open |
| H-03 | P0 | `spec/hardware.yaml` allocates physical pin 32/GPIO12 to both ToF interrupt and UART4 TX. | Reconcile GPIO/overlay assignments with `gpioinfo`, schematics and physical inspection. | Confirmed doc collision; open |
| H-04 | P0 | `docs/tractor-acceptance-criteria.md` proposes PCA9685 OE LOW for disabling outputs; OE is active-low and LOW enables outputs. | Correct docs and prove watchdog/OE behavior on bench. PWM disable does not guarantee neutral. | Confirmed electrical error; open |
| H-05 | P0 | `docs/tractor-platform.md` references 12 V RDS51150SG while acceptance checklist still describes 24 V servo rail. | Identify purchased servo and verify label/datasheet; correct every power design. | Confirmed inconsistency; open |
| H-06 | P0 | Tractor lever linkage only fore/aft; OEM PARK requires outboard lever movement; autonomous starter relay cannot satisfy OEM PARK. | Retain supervised manual start, document sequence; never bypass OEM start interlock. | Documented physical constraint |
| H-07 | P0 | OEM seat/operator-presence switch incompatible with unoccupied travel unless a professionally reviewed safety architecture is implemented. | Preserve OEM interlocks; require safety engineering review, not software attestation alone. | Unresolved safety gate |
| H-08 | P0 | PCA9685 holds last PWM on I2C loss; lever spring return unverified; current docs lack independently validated physical failsafe. | Independent hardwired stop, actuator-safe mechanism and physical fault-injection testing. | Unresolved safety gate |
| H-09 | P1 | `spec/hardware.yaml` GPS preferred ZED-F9P; owner specifies Quectel LC29H-DA rover + LC29H-BS base. | Check physical modules, NMEA/RTCM support, correction path and integration tests. | Config/documentation mismatch |
| H-10 | P1 | `docs/ultrasonic-wiring-guide.md` covers three sensors; owner specifies four across 50-inch deck. | Revise mounting, trigger schedule, echo isolation, obstacle coverage and tests. | Requirements mismatch |
| H-11 | P1 | Differential-drive RoboHAT/Cytron/IBT-4 docs coexist with Toro PCA9685 tractor variant. | Tag docs per platform; establish one deployment-specific authoritative inventory and pin map. | Documentation drift |
| H-12 | P1 | `config/tractor.yaml` defaults `enabled: false`; bench calibration and activation remain separate. | Do not enable until P0 gates passed and actuator tests signed. | Config confirmed |
| H-13 | P1 | `docs/tractor-platform.md` already documents Thor protocol `backend/src/models/thor_link.py` and service `backend/src/services/thor_link_service.py`. | Inspect implementation, tests, auth, retransmission, timeouts and link-loss behavior; don't call it missing without source audit. | Existence documented; functionality unverified |
| H-14 | P1 | `PRODUCT.md` mentions `OPERATOR_AUTH_REQUIRED=0` as selected default. | Threat model remote control; enforce authenticated sessions, authorization and dead-man controls before any live VR/manual operation. | Security review needed |
| H-15 | P2 | Optional low-cost 360-degree 2D LiDAR model unspecified. | Select actual model and confirm ROS/SDK driver, power, range, angle, scan plane and weather tolerance. | Hardware selection open |
| H-16 | P2 | Elevated ELP stereo camera and pan/tilt VR workflow proposed. | Confirm exact camera interfaces, encoder, HaLow throughput/latency and operator dead-man. | Implementation unverified |
| H-17 | P1 | Hailo 8L, Pi 5, CSI camera, HaLow adapter and sensor stack may compete for physical header/PCIe/USB/power. | Photograph installed stack; prepare mechanical and bus occupancy diagram and power budget. | Hardware audit open |
| H-18 | P1 | First ultrasonic divider ASCII schematic is inconsistent with later breadboard wiring; nominal 1k/2k ratio yields 3.33 V from 5 V. | Correct schematic, choose margin-rated level conversion and verify tolerance. | Confirmed documentation risk |

## Immediate source-audit tasks for Thor agents
1. Run `scripts/check_hardware_pin_conflicts.py` and `scripts/check_docs_drift.sh`; record exact outputs, not just pass/fail.
2. Inspect `config/hardware.yaml`, `config/limits.yaml`, `config/tractor.yaml`, `spec/hardware.yaml`, `backend/src/models/thor_link.py`, `backend/src/services/thor_link_service.py`, tractor driver/service, navigation path, and associated tests.
3. For each capability (RTK base/rover, BNO085, ToF x2, ultrasonic x4, LiDAR, fixed camera, stereo camera, Hailo, HaLow, Thor planning, manual VR, physical E-stop), mark **implemented/tested/simulated/stub/missing/unknown**, cite exact source file and test command.
4. Confirm physical inventory from photographs, labels and measured rails before proposing a final wiring netlist.
5. Return findings as a draft PR; do not modify live deployment or energize motors.

## Release gates
- **No live motion:** while H-01 through H-08 remain unresolved.
- **No autonomous mowing:** until sensor/RTK health, physical fail-safe, operator protections, and field acceptance are proven.
- **No remote VR driving:** until authentication, dead-man, measured latency and local safety override are proven.

See `docs/2026-10-24-physical-integration-plan.md` and `docs/2026-10-24-development-roadmap.md` for proposed integration and phases.
