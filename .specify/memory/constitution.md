<!--
Sync Impact Report:
Version: 2.0.0 → 3.0.0 (Major: Constitutionally recognized the second physical platform —
  a Craftsman-class ride-on Ackermann tractor conversion, alongside the original
  differential-drive mower — introduced by commit d2cb1ae on 2026-06-26 but never ratified
  into this document; closed two pre-existing silent violations of this constitution's own
  text; replaced the blanket motor E-stop latency rule with honest actuator-class tiers)
Modified principles:
  - Principle V expanded: formal two-platform recognition; `config/tractor.yaml`'s `enabled`
    flag made constitutionally gated on a complete `spec/hardware.yaml` tractor section
  - Principle VI expanded: human-proximity/ROPS risk tier, operator-attestation gate for
    blade/PTO engagement and autonomous-motion start, tiered E-stop latency by actuator
    class (relay / positional-command-issuance / positional-physical-settle), explicit
    bus-fault-failsafe requirement, named closure of two silent-violation gaps (IMU
    tilt-cutoff and motor watchdog not wired to `tractor_service.py`)
Added sections: None (extended existing Principles V and VI; no new principles)
Removed sections: None
Templates requiring updates:
  - Bottom-of-document "Acceptance Criteria (Core Safety Requirements)" bullets (this file)
    ✅ reconciled with tiered Principle VI language in this amendment
  - docs/tractor-acceptance-criteria.md (NEW) ✅ added as the operational checklist this
    amendment's Principle VI requirements point to
  - docs/RELEASE_NOTES.md, docs/hardware-feature-matrix.md, docs/hardware-overview.md,
    docs/field-validation-protocol.md ✅ updated for tractor platform in the same pass
Follow-up TODOs:
  - Open implementation gap: IMU tilt-cutoff (200ms, Principle VI) is not wired to
    `backend/src/services/tractor_service.py`. Tracked here; not closed by this amendment.
  - Open implementation gap: motor-watchdog heartbeat (Principle VI) is not wired to
    `backend/src/services/tractor_service.py`. Tracked here; not closed by this amendment.
  - Open implementation gap: bus-fault failsafe for the PCA9685 hold-last-value hazard
    (Principle VI) has no hardware/software design yet. Tracked in
    docs/tractor-acceptance-criteria.md as the top acceptance item.
-->

# LawnBerry Pi Constitution

## Core Principles

### I. Platform Exclusivity
The LawnBerry Pi system MUST operate exclusively on Raspberry Pi OS Bookworm (64-bit) with Python 3.11.x runtime on Raspberry Pi 5 (primary) or Pi 4B (compatible). No cross-platform dependencies, alternate interpreters, or non-ARM64 Linux distributions are permitted. All development, testing, and deployment MUST assume this target platform exclusively.

### II. Package Isolation (NON-NEGOTIABLE)
AI acceleration dependencies MUST maintain strict isolation: `pycoral`/`edgetpu` are BANNED from the main environment and MUST use dedicated venv-coral. Coral USB acceleration operates in complete isolation from the main system. Hardware acceleration follows constitutional hierarchy: Coral USB (isolated venv-coral) → Hailo HAT (optional) → CPU fallback (TFLite/OpenCV). Package management via uv with committed lock files ensures reproducible builds.

### III. Test-First Development (NON-NEGOTIABLE)
TDD methodology is mandatory: Tests written → User approved → Tests fail → Implementation begins. Red-Green-Refactor cycle strictly enforced. Every feature starts with failing tests that define expected behavior. No implementation code without corresponding test coverage. Hardware simulation (SIM_MODE=1) MUST provide complete test coverage for CI execution without physical hardware dependencies. Mock drivers MUST replicate hardware behavior including latency, failure modes, and state transitions for comprehensive testing without physical hardware.

### IV. Hardware Resource Coordination
Hardware interfaces are single-owner resources requiring explicit coordination. Camera access is brokered exclusively through `camera-stream.service`; other services MUST subscribe to feeds via IPC and NEVER open the device directly. Sensors, motor controllers, and communication buses require coordination mechanisms (locks, IPC queues, or dedicated daemons) to prevent concurrent access conflicts. Resource ownership must be clearly defined and enforced. Motor control commands MUST pass through safety interlock validation before hardware execution. Emergency stop (E-stop) signals override all other commands with <100ms latency requirement (see Principle VI for per-actuator-class latency tiers).

### V. Constitutional Hardware Compliance
Hardware configuration MUST align with `spec/hardware.yaml` requirements. INA3221 power monitoring uses fixed channel assignments: Channel 1 (Battery), Channel 2 (Unused), Channel 3 (Solar Input). GPS supports either ZED-F9P USB with NTRIP corrections OR Neo-8M UART (mutually exclusive). Motor control via RoboHAT RP2040→Cytron MDDRC10 (preferred) or L298N fallback. HAT stacking conflicts (RoboHAT + Hailo HAT) are prohibited without constitutional amendment.

**Recognized platforms**: The system MAY be deployed as either (a) the original differential-drive autonomous push-mower, or (b) a Craftsman-class ride-on tractor conversion (Ackermann steering, gas engine, seven discrete actuators — see `docs/tractor-platform.md`). Both are constitutionally valid platforms; a given deployment operates exactly one, never both concurrently. `config/tractor.yaml`'s `enabled: true` flag is constitutionally meaningful, not a cosmetic toggle: it is valid only when `spec/hardware.yaml` carries a complete `tractor:` section for the deployed hardware AND `scripts/check_hardware_pin_conflicts.py` passes with zero conflicts across the merged mower+tractor pin/address registry. A deployment running `enabled: true` while that check fails is a constitutional violation, not merely a failing CI job.

### VI. Safety-First Engineering (NON-NEGOTIABLE)
Safety is the paramount concern in all system operations. Motion MUST only occur when hard and soft safety failsafes are operational and verified. System MUST default to OFF state on startup; motion requires explicit operator authorization. Safety interlocks MUST prevent blade operation when drive motors are active. All safety violations MUST be logged with timestamps and require operator acknowledgement for recovery.

**Human-proximity / ROPS risk tier**: The ride-on tractor platform (Principle V) introduces a risk class the original small autonomous push-mower never contemplated — a person may be seated on, or standing near, a powered machine with a running gas engine. Software cannot physically verify a human is clear of the machine or its blade. The system MUST therefore require an explicit **operator-attestation gate** — a deliberate, logged operator action affirming the seat/area is clear — before (a) blade/PTO engagement and (b) the start of any autonomous motion, in addition to (never in place of) the existing authorization and interlock requirements. This is a policy control, not a substitute for a sensor that does not exist.

**Tiered emergency-stop latency (replaces the prior blanket "100ms for all motors" rule)**: one latency number is dishonest across actuator classes with materially different physics. E-stop latency is defined per actuator class:
- **GPIO relay actuators** (e.g., blade PTO, starter): MUST de-energize within 100ms of the E-stop signal. This is the original mower-era requirement and is unchanged for this actuator class.
- **Positional actuators — command issuance** (e.g., steering, throttle, gas pedal, clutch, gear, over PCA9685 or equivalent PWM transport): the commanded safe value (idle throttle, neutral gear, pressed clutch/brake, centered steering) MUST be *issued* within 100ms of the E-stop signal.
- **Positional actuators — physical settle**: the actuator MUST *physically reach* its commanded safe position within 500ms of the E-stop signal, accounting for real mechanical travel time. This is an observed, measured outcome, not an inferred one — a passing `state.field == value` unit test does not, by itself, satisfy this requirement (see `docs/tractor-acceptance-criteria.md`).

IMU tilt detection MUST trigger blade cutoff within 200ms. Watchdog timer enforcement is mandatory for all control loops; software watchdog heartbeat MUST be enforced for all motor control operations with automatic emergency stop on timeout.

**Bus-fault failsafe (NON-NEGOTIABLE)**: any actuation transport that can silently hold its last commanded value on communication loss (e.g., a PCA9685 PWM driver, which does not fail safe on its own on I2C/host loss) MUST be paired with an independent failsafe that drives affected actuators to a safe state on bus/host fault — for example, a hardware watchdog holding the driver's output-enable line low, or spring-return actuator hardware — and that failsafe MUST be verified by fault injection, not assumed (see `docs/tractor-acceptance-criteria.md`). Replacing a "does nothing on fault" transport with a "freezes at last value with the engine running" transport, without an accompanying failsafe, is a constitutional violation regardless of how the rest of the interlock chain behaves.

**Closed gaps (named explicitly, not left implicit)**: as of this amendment, the IMU tilt-cutoff (200ms) and motor-watchdog requirements above already existed in this Principle but are **not yet wired to `backend/src/services/tractor_service.py`** — the ride-on tractor's actuator control currently ships without either. This is a tracked, open implementation gap (see the Sync Impact Report's Follow-up TODOs), not a silent one going forward: no tractor deployment may be represented as constitutionally compliant while this gap remains open.

### VII. Modular Architecture
System architecture follows strict modular boundaries aligned with Engineering Plan phases. Core modules include: `drivers/` (hardware shims for motors, blade, IMU, ToF, environmental sensors, power, GPS), `safety/` (interlocks, triggers, watchdog, E-stop coordination), `fusion/` (sensor fusion and state estimation), `nav/` (geofencing, path planner, controller), `api/` (REST + WebSocket), `ui/` (retro-neon frontend), `scheduler/` (calendar, weather integration, charge management), and `tools/` (CLIs, analyzers, calibration utilities). Each module exposes defined contracts and MUST NOT bypass interfaces to access implementation internals. Drivers MUST be hardware-agnostic with clean adapter interfaces to enable simulation, testing, and future hardware substitution.

### VIII. Navigation & Geofencing (MANDATORY)
Autonomous navigation MUST respect geofence boundaries with zero tolerance for incursions. GPS localization with optional RTK corrections provides primary positioning; odometry provides secondary dead-reckoning between GPS updates. Geofence violations MUST trigger immediate motor stop and operator notification. Waypoint navigation follows parallel-line coverage patterns with configurable overlap. Navigation mode manager coordinates state transitions between MANUAL, AUTONOMOUS, EMERGENCY_STOP, CALIBRATION, and IDLE modes. Missing or degraded GPS MUST NOT compromise safety; system reverts to MANUAL mode with restricted operation. All navigation commands are subject to safety interlock validation before motor execution.

### IX. Scheduling & Autonomy
Autonomous mowing operations execute via calendar-based scheduling with weather-aware postponement logic. Jobs MUST NOT start during rain, high wind, or low battery conditions. Solar charge management integrates with scheduling to optimize energy availability. Mowing schedules respect user-defined operating windows and geofence boundaries. Job execution state machine tracks IDLE → SCHEDULED → RUNNING → PAUSED → COMPLETED → FAILED transitions with audit logging. Autonomous operations MUST verify all safety systems operational before commencing; any safety fault aborts the job and requires operator intervention. Return-to-home and return-to-solar-waypoint behaviors are mandatory for charge management and safe parking.

### X. Observability & Debuggability
System MUST maintain comprehensive structured logging (JSON format) with configurable log levels and rotating file management. All safety events, motor commands, navigation decisions, and operator interactions are logged with microsecond-precision timestamps. Real-time telemetry streaming via WebSocket provides live system state visibility at 5Hz minimum. Diagnostic CLI tools enable live sensor testing, motor calibration, and fault analysis without UI dependency. Fault injection capabilities support reliability testing and operator training. Log bundles aggregate system state, sensor data, and audit trails for incident analysis. Metrics exposure via `/metrics` endpoint (Prometheus format) is recommended for production deployments. Dashboard visualizations present key performance indicators including battery state, coverage progress, safety system status, and environmental conditions.

## Technology Stack Requirements

All system components must use approved technologies and interfaces: Picamera2 + GStreamer for camera handling, python-periphery + lgpio for GPIO control, pyserial for UART communication, and systemd for service management. Backend API uses FastAPI with asyncio for concurrent operations. Frontend implements Vue.js 3 with retro 1980s cyberpunk aesthetic (Orbitron fonts, neon color palette: #00ffff, #ff00ff, #ffff00). Real-time communication via WebSocket telemetry streaming at 5Hz minimum. No frameworks or libraries outside the approved stack without constitutional amendment and compelling technical justification. All dependencies MUST be ARM64-compatible; x86-only dependencies are BANNED.

## Development Workflow

Every code change MUST update `/docs` and `/spec` documentation with CI validation preventing drift. No TODOs are permitted unless formatted as `TODO(v3):` with linked GitHub issue for future version planning. All services operate as managed systemd units with automatic startup, monitoring, and graceful shutdown. Service coordination respects camera ownership and hardware resource management protocols. Development follows phased approach: Phase 0 (Foundation & Tooling) → Phase 1 (Core Abstractions) → Phase 2 (Safety & Motor Control) → Phase 3 (Sensors & Extended Safety) → Phase 4 (Navigation Core) → Phase 5 (Web UI & Remote Access) → Phase 6 (Scheduling & Autonomy) → Phase 7 (Reliability & Polish). Feature branches MUST NOT skip phases; dependencies must be satisfied before advancing.

AI agents MUST maintain an `AGENT_JOURNAL.md` file in the `.specify/memory/` folder, documenting progress, changes made, decisions taken, and any information necessary for seamless handoff to other agents or developers. The journal MUST include timestamps, rationale for major decisions, and current project state to ensure continuity across development sessions. Safety-critical changes (E-stop, motor control, blade interlocks, geofencing) require explicit constitutional compliance verification in commit messages.

After completing assigned tasks for a session, the responsible agent MUST execute the repository workflows defined in `.github/workflows/`. Upon successful completion of these workflows, the agent MUST commit any resulting changes (generated artifacts, updated docs/specs, version bumps) to the current working branch with a clear, conventional commit message referencing the workflows run. If workflows fail, the agent MUST document failures and remediation steps in `AGENT_JOURNAL.md` and refrain from committing broken artifacts.

Agent execution rules (MANDATORY):
- Apply code changes directly in the repository.
- If tests or linting are needed, run them in the terminal on Linux/ARM64 (Raspberry Pi OS Bookworm). Do NOT add Windows/macOS-only dependencies or instructions.
- If any dependency is not available on ARM64, STOP and propose a Pi-compatible alternative in `AGENT_JOURNAL.md` (and PR description, if applicable) before proceeding.
- When done, summarize the changes and update `.specify/memory/AGENT_JOURNAL.md` with outcomes, rationale, and next steps.
  
Workspace editing and local context (STRICT):
- Always edit files directly in the local workspace. Do NOT download or fetch files via remote content APIs/tools to modify them. Locate and open files from the checked-out repository tree.
- Prefer local codebase search (editor search, ripgrep) to discover files and symbols. Use repository-aware tools before any remote content fetchers. If a remote fetch is ever necessary, justify it in the journal.

Git operations (OPTIONS):
- You MAY use MCP GitHub tools for commits and PRs if available and part of the workflow.
- You MAY alternatively use the GitHub CLI (`gh`) for git operations when it is installed and authenticated in the workspace. Choose one method per session and document which was used in `AGENT_JOURNAL.md`.
- All PRs MUST use `.github/pull_request_template.md` and include a brief note confirming constitutional compliance.

- For teams using MCP GitHub tools, use `#mcp_github_add_comment_to_pending_review` to perform the push/commit step in accordance with the established review flow after successful workflow completion. If using GitHub CLI, use `gh pr create` with the repository template.

## Governance

This constitution supersedes all other development practices and requirements. Constitutional amendments require formal documentation, approval process, and migration plan for affected systems. All pull requests and code reviews MUST verify constitutional compliance before approval. Complexity deviations require explicit justification with simpler alternatives documented. Development teams MUST use `spec/agent_rules.md` for runtime development guidance and implementation constraints.

**Acceptance Criteria (Core Safety Requirements)**:
- Emergency stop latency — relay actuators: <100ms from signal to de-energize
- Emergency stop latency — positional actuators, command issuance: <100ms from signal to safe-value command
- Emergency stop latency — positional actuators, physical settle: <500ms from signal to physically-observed safe position
- Bus-fault failsafe: any hold-last-value actuation transport (e.g., PCA9685) MUST reach a safe state on I2C/host loss, independent of software E-stop, verified by fault injection
- Operator-attestation gate: required before blade/PTO engagement and before autonomous-motion start on human-proximity/ROPS-tier platforms
- IMU tilt cutoff latency: <200ms from threshold breach to blade stop
- UI telemetry update rate: ≤1s (1Hz minimum, 5Hz target)
- Navigation geofence incursions: 0 tolerance (immediate stop)
- Graceful degradation: Missing GPS → Manual mode remains safe
- Watchdog timeout enforcement: Mandatory for all motor operations (tractor platform: open implementation gap, see Constitutional Change Log 3.0.0)

**Version**: 3.0.0 | **Ratified**: 2025-09-25 | **Last Amended**: 2026-07-08

---

## Constitutional Change Log

### 3.0.0 (2026-07-08) - Major: Ride-On Tractor Platform Constitutional Recognition
**Modified Principles**:
- Principle V: Formally recognized two constitutionally-valid platforms (differential-drive mower, Craftsman-class Ackermann ride-on tractor); made `config/tractor.yaml`'s `enabled` flag constitutionally gated on a complete `spec/hardware.yaml` `tractor:` section passing `scripts/check_hardware_pin_conflicts.py` with zero conflicts.
- Principle VI: Added a human-proximity/ROPS risk tier and an operator-attestation gate for blade/PTO engagement and autonomous-motion start; replaced the blanket "100ms for all motors" E-stop rule with three actuator-class latency tiers (GPIO relay / positional-command-issuance / positional-physical-settle); added an explicit, fault-injection-verified bus-fault-failsafe requirement; named and tracked two pre-existing gaps where this constitution's own text was silently unmet (IMU tilt-cutoff and motor watchdog not wired to `tractor_service.py`).
- Reconciled the document's closing "Acceptance Criteria (Core Safety Requirements)" bullets with the new tiered Principle VI language so the two sections no longer contradict each other.

**Rationale**: The ride-on tractor platform (`d2cb1ae`, 2026-06-26) shipped a materially different risk profile — a human-proximity, ROPS-relevant, gas-engine ride-on vehicle — into a constitution written entirely for a small autonomous push-mower, more than eight months after the prior amendment and without ever being ratified into this document. Two requirements this constitution already imposed on "all motor control operations" (tilt-cutoff, watchdog) were silently unmet by the new platform's code. The planned actuation-transport swap to PCA9685 introduces a new hold-last-value bus-fault hazard that the prior blanket E-stop language did not address. This amendment closes the documentation/governance gap and converts both the pre-existing gaps and the new hazard into explicit, tracked, enforceable requirements instead of implicit assumptions.

### 2.0.0 (2025-10-02) - Major: Engineering Plan Alignment
**Added Principles**:
- Principle VI: Safety-First Engineering - Codifies E-stop latency, watchdog enforcement, safety interlock requirements
- Principle VII: Modular Architecture - Defines system decomposition aligned with Engineering Plan module map
- Principle VIII: Navigation & Geofencing - Establishes zero-tolerance geofence policy and GPS degradation behavior
- Principle IX: Scheduling & Autonomy - Mandates weather-aware job execution and charge management integration
- Principle X: Observability & Debuggability - Requires structured logging, telemetry streaming, and diagnostic tooling

**Enhanced Principles**:
- Principle III: Added hardware simulation mock driver requirements for CI testing
- Principle IV: Expanded with motor control safety interlock mandates and E-stop override priority

**Rationale**: Project analysis revealed gaps between Engineering Plan safety requirements (Phases 2-7) and constitutional governance. Safety-first mandate was implicit but not constitutionally enforced. Navigation, scheduling, and observability principles were absent despite significant implementation. This amendment brings constitutional authority in line with engineering requirements and current system capabilities.

### 1.4.0 (2025-09-26) - Minor: Agent Execution Rules
**Changes**: Clarified mandatory local workspace editing, added GitHub CLI option for git operations
**Rationale**: Standardize agent workspace interaction patterns

### 1.3.0 (2025-09-25) - Minor: Initial Ratification
**Changes**: Established foundational principles I-V
**Rationale**: Bootstrap constitutional governance for LawnBerry Pi project