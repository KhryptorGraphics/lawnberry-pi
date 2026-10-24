# Oh My Pi / Thor development handoff — 2026-10-24

**Status:** proposed tasks, NOT evidence that Oh My Pi has received or executed them. **Branch:** `docs/2026-10-24-hardware-architecture-audit`. **Do not merge or actuate hardware automatically.**

## Operator workflow on Thor
1. In the existing LawnBerry-pi checkout, `git fetch origin`.
2. Inspect this handoff using `git show origin/docs/2026-10-24-hardware-architecture-audit:docs/OMP-THOR-HANDOFF-2026-10-24.md`. Do not switch branches if the working tree has uncommitted changes.
3. Ask Oh My Pi to audit the referenced files, prepare a conflict report, and implement documentation improvements on its **own new branch**. Do not alter active deployment, motor commands, safety interlocks, or hardware config without explicit operator approval.
4. Run repository checks in simulation, document commands/results, push the branch and open a draft PR. Human review and hardware sign-off are mandatory.

## Audit findings verified from repository files
- `spec/hardware.yaml` lists ZED-F9P as preferred GPS; owner's actual intended receiver is Quectel LC29H-DA, with LC29H-BS base. This requires a protocol, serial interface, corrections and power audit; do not assume interchangeable drivers.
- `docs/ultrasonic-wiring-guide.md` describes **three** HC-SR04 units, while current owner plan is **four** front-facing units, with outward-angled edge coverage for a 50-inch deck. Existing guide also incorrectly says physical pin **17** is 5V; pin 17 is **3.3V**. **Do not wire from this guide as written.**
- The same ultrasonic guide's first ASCII divider schematic is ambiguous/incorrect; use its later series-1k/shunt-2k topology only after electrical review, noting nominal 3.33V is at/above a nominal 3.3V rail and actual supply/tolerance must be considered. Prefer a correctly rated level shifter or divider with margin.
- `spec/hardware.yaml` assigns physical header pin 32 to both GPIO12 ToF interrupt and UART4 TXD4; it also lists UART4 RX on physical pin 33 while `docs/hardware-integration.md` states GPIO13 RXD4. Pin multiplexing and physical mapping must be verified against Pi 5 overlays and actual configuration.
- `PRODUCT.md` reports ultrasonic sensing disabled because of HaLow GPIO conflict, issue #15. Validate current code/config and close the conflict before using ultrasound as a safety input.
- `spec/hardware.yaml` describes a small differential-drive platform with RoboHAT/IBT-4 and also a Toro TimeCutter tractor with PCA9685 servos/relays. These are **different platform variants**, not one wiring harness. Owner's 50-inch zero-turn mower must have its own authoritative bill of materials and wiring diagram.
- `docs/tractor-acceptance-criteria.md` explicitly leaves PCA9685 bus-fault failsafe, E-stop physical behavior, tilt cutoff, motor watchdog, lever return-to-neutral, and operator-presence integration unverified. No unattended mowing until independently validated.
- `PRODUCT.md` currently describes NVIDIA Thor primarily as an offline training server; owner is also considering Thor-in-house for live high-level inference/planning over Wi-Fi HaLow. Treat this as a **proposed architectural change**, not an implemented capability.

## Deliverables for Oh My Pi
1. **Hardware inventory and traceability matrix:** component/model, bought vs proposed, mower vs house, interface, supply voltage, driver, config, owner confirmation, and evidence link.
2. **Physical integration design:** complete Pi 5 40-pin physical-pin/BCM map, I2C addresses, UART devices, USB/CSI/PCIe occupancy, PCA9685 channels, level shifting, fused power tree, cable routing, common ground and isolation strategy; mark every unverified electrical detail TBD rather than guessing.
3. **Safety case and test plan:** independent hardwired E-stop, blade/PTO disable, neutral-on-fault, watchdog, loss-of-link, loss-of-Thor, I2C bus hang, servo stuck-output, GPS/RTK loss, and physical bench acceptance. Preserve OEM interlocks; do not recommend bypassing seat/presence switches.
4. **Software gap matrix:** locate current implementations and tests for LC29H-DA/BS RTK, four ultrasonic sensors, optional 2D LiDAR, HaLow reconnection, Thor/Pi communication, local obstacle stop, manual VR stereo streaming/head tracking, autonomous fixed camera, and fail-safe behavior.
5. **Phased development roadmap:** milestones, dependencies, specific files/modules, tests, acceptance criteria, risk/rollback and explicit human sign-off gates.
6. **Documentation consistency:** reconcile README, `PRODUCT.md`, `spec/hardware.yaml`, `config/tractor.yaml`, hardware guides and deployed config. Do not silently rewrite canonical hardware specs without measured evidence.

## Proposed split of responsibilities
- **ChatGPT through GitHub:** requirements reconciliation, hardware and architecture documentation, PR review, audit findings.
- **Oh My Pi on Thor:** local source inspection, code changes, simulated tests, PR submission and measured diagnostics from connected equipment when explicitly authorized.
- **Operator:** confirms actual purchased hardware and electrical measurements, authorizes physical changes, validates safety.

## Initial OMP prompt
> In the LawnBerry-pi repository, review docs/OMP-THOR-HANDOFF-2026-10-24.md on origin/docs/2026-10-24-hardware-architecture-audit. First produce a source-linked hardware conflict matrix and software gap report. Do not drive actuators or modify active hardware configuration. Work on a separate branch, run SIM_MODE=1 checks, and open a draft PR with exact tests and unresolved questions.

**Integration limit:** A GitHub commit is a shared handoff, not a live RPC connection. ChatGPT cannot currently issue commands to the Thor's running Oh My Pi session or verify it consumed this file.


## Additional owner requirements recovered in conversation review
- Replace the legacy three-HC-SR04 assumption with **four JSN-SR04T waterproof ultrasonic units** (inner pair forward, outer pair angled outward on adjustable mounts, spanning 50-inch Toro deck). Confirm module revision, voltage and near-range blind zone.
- Two owner-reported matching **150 kg-cm steering servos**, nominal **10–12.6 V**, may draw high stall current. Confirm labels, measured supply and fuse sizing; do not use stale 24 V servo instructions.
- Inspect reported **momentary PTO toggle** against the software's latching relay abstraction before touching wiring.
- Add **Pi Zero RTK base / Orange Pi HaLow gateway / Thor** as a candidate house topology to confirm, with full network and power diagrams.
- Thor is intended to lead navigation/decision making; Pi remains local safety authority. Evaluate real-time camera/depth telemetry and fallback behavior.
- Deliver **photo-annotated and 3D-style physical assembly documentation**: actual Toro front view with four adjustable ultrasonic mounts, GPIO/HAT riser access, USB-C, PCIe/Hailo, CSI camera, full sensor and servo wiring, house-side base and gateway.
- Audit ROS 2/Nav2 feasibility, stereo/2D-LiDAR obstacle fusion, and any existing agent automation bridge (earlier PR #16) without assuming installed/functional capability.
- Codex CLI, Claude Code and OMP all run on Thor. Coordinate via **separate Git worktrees**, never simultaneous edits to one working tree, and no unattended hardware actions.

See the appended corrections in `docs/2026-10-24-physical-integration-plan.md`, `docs/2026-10-24-development-roadmap.md` and `docs/2026-10-24-compatibility-gap-register.md`. Treat this as a **superseding requirements addendum**, not evidence of completed implementation.
