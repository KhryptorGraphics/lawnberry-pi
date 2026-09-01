# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary user: the builder/owner, operating the mower on their own property. Single-operator today — the dashboard has no auth gate (`OPERATOR_AUTH_REQUIRED=0`) by deliberate decision, not oversight. Not designed yet for multi-tenant or third-party operators.

## Product Purpose

LawnBerry Pi is an autonomous lawn mower: a Raspberry Pi 5 + Hailo 8L edge system that drives real hardware (RTK GPS, stereo/Pi cameras, ultrasonic/ToF sensors, IMU, RoboHAT motor controller, IBT-4 blade driver) to mow a defined area on its own, with a Vue 3 operator dashboard for setup, monitoring, and manual override. Success is a mower that plans and executes a mowing job inside a user-drawn boundary safely and correctly, with the operator able to watch, intervene, and trust the safety chain at any time.

## Positioning

Combines a trained Vision-Language-Action model running on dedicated edge AI silicon (Hailo 8L, 13 TOPS INT8) with RTK-GPS centimeter-level positioning and full operator transparency — live telemetry, camera feeds, and inference metrics are all inspectable, not hidden behind a sealed commercial appliance. Distinguishes it from consumer robotic mowers (e.g. Husqvarna Automower): this is an open, self-built and self-owned system where the owner can see and control exactly what the AI is doing and why.

## Operating Context

- Outdoor deployment on a physical mower chassis; the Pi runs the stack via systemd (`SIM_MODE=0`), off-device dev/CI runs simulated (`SIM_MODE=1`).
- Core operator workflows: draw/edit mow boundaries and exclusion zones (Maps), plan jobs and schedules (Planning), monitor live telemetry and RTK/IMU/power diagnostics, drive manually with joystick + emergency stop (Manual Control), enable/deploy the AI model and watch inference metrics (AI & Model Control).
- A second platform variant — a ride-on Craftsman tractor with Ackermann steering — is fully implemented (manual actuation, telemetry, safety governance) but currently dormant pending physical bring-up and acceptance testing.
- Training (VLA model authoring) happens separately on an NVIDIA Thor server; the Pi only ever runs INT8 inference, never trains.

## Capabilities and Constraints

- Hardware safety is layered and independent: E-stop (hardware + software watchdog), tilt/rollover protection, motor authorization interlocks, geofence enforcement. Treat safety-chain changes as high-risk.
- Camera ownership is exclusive to `camera-stream.service`; other services consume frames via IPC, per the project constitution.
- Ultrasonic obstacle sensing is currently disabled in the deployed build due to a HaLow GPIO pin conflict (tracked as issue #15) — not a design choice, a known gap.
- Multi-sensor fusion (stereo depth, RGB, GPS, IMU, ultrasonic/ToF) feeds both the safety chain and the VLA policy.
- Never assume hardware is present in code paths — hardware libraries are imported lazily so the stack stays testable under `SIM_MODE=1`.

## Brand Commitments

Existing dashboard identity, already implemented in `frontend/src/` (Orbitron font, dark theme, neon accents) and documented in the README as a "1980s Cyberpunk UI" — a professional dark theme, not a playful or consumer-friendly one. Preserve this identity in refinement work; a request to replace it would be a deliberate redesign decision, not a default.

## Evidence on Hand

Real hardware specs and architecture are documented and current: `README.md` (hardware tables, system architecture diagram), `docs/hardware-overview.md`, `docs/hardware-feature-matrix.md`, `docs/hardware-integration.md`, `docs/operator-dashboard.md` (route-by-route dashboard reference), `docs/tractor-platform.md` / `docs/tractor-acceptance-criteria.md`. No fabricated customer testimonials, pricing, or third-party benchmarks exist or should be invented — this is a single-owner deployed system, not a marketed product.

## Product Principles

1. Transparency over polish where they conflict — the operator must always be able to see real sensor/AI state, never a smoothed-over abstraction of it.
2. Safety systems are non-negotiable and independent of the AI/navigation stack; UI must never make an unsafe state look benign.
3. The dashboard serves one operator doing real-time monitoring and control, not a broad audience — optimize for fast, correct situational awareness over onboarding ease.
4. Simulated (`SIM_MODE=1`) and real hardware states must never be visually confusable in the UI.
5. Preserve the existing dark/cyberpunk professional identity; it is a confirmed brand commitment, not incidental styling.
