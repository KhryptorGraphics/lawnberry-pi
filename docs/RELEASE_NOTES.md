# LawnBerry Pi v2.0.0 - Release Notes (2025-09-27)

This release replaces version 1 with a unified v2 backend and frontend, contract-tested APIs, and on-device readiness for Raspberry Pi (ARM64, Bookworm).

## Maintenance (2025-10-22)

- Hardened backend authentication by switching to a direct bcrypt-based password manager that safely handles operator credentials longer than 72 bytes.
- Secrets manager now self-generates a persistent `JWT_SECRET` when missing, avoiding startup failures on fresh installs.

## Enhancements (2025-10-25)

- Manual control camera feed now streams via `/api/v2/camera/stream.mjpeg` with automatic snapshot fallback, reducing perceived latency on the WebUI.
- Drive commands carry the operator speed slider as `max_speed_limit`, ensuring forward throttle reaches the RoboHAT controller while preserving safety clamps.
- Motor safety logging correctly reports safety violations instead of raising attribute errors, improving diagnostics when commands are rejected.
- Camera API endpoints now include dedicated rate-limit allowances and the WebUI identifies itself via `X-Client-Id`, eliminating `429 Too Many Requests` errors during streaming fallback.
- WebUI detects unavailable MJPEG streams, switches to snapshot mode automatically, and suppresses repetitive console errors for smoother diagnostics.
- MJPEG snapshot fallback now auto-retries the primary stream after short cooldowns, so camera feeds recover without manual refreshes.
- Joystick drive commands are coalesced and dispatched at a higher cadence, cutting manual control input lag while staying within rate limits.

## Fixes (2025-10-26)

- RoboHAT USB watchdog now detects firmware timeouts and reasserts `rc=disable`, ensuring manual joystick drive commands continue to reach the motor controller after idle periods.
- Added backend unit coverage for RoboHAT USB control keep-alive logic to prevent regressions in manual movement.
- RoboHAT PWM handshake now waits for explicit firmware acknowledgement before issuing keep-alives or drive commands, eliminating `[USB] Invalid command: pwm` spam and restoring manual joystick responsiveness.

## Fixes (2025-10-27)

- Manual drive endpoints retry RoboHAT USB takeover with adaptive timeouts and surface firmware errors in responses, eliminating silent command drops when RC passthrough lingers.
- Control WebUI now highlights motor-controller health, blocks dispatch when the USB link is offline, and raises toasts when the controller disconnects or recovers.
- Camera JPEG encoder respects explicit colour-space hints from PiCamera2 frames, preventing the magenta tint seen in MJPEG streams after the migration to RGB888 capture.
- Added regression tests for RoboHAT USB acquisition retries, manual unlock TOTP window tolerance, and camera colour encoding to prevent future regressions.

## Feature (2026-06-26)

- Added ride-on lawn tractor actuation platform (`d2cb1ae`) — this entry is retroactive; the commit predated this changelog entry and should have been logged at merge time per this document's own convention. The target mower is a converted Craftsman-class ride-on tractor (Ackermann steering, gas engine), not a differential-drive robot. Adds a second, constitutionally-recognized actuation subsystem (steering, throttle, ground-speed pedal, clutch, gear, starter, blade PTO) alongside the original differential-drive platform:
  - `models/tractor_control.py` (Transmission F/N/R, `TractorCommand`, `TractorState`)
  - `drivers/actuators/` (`ServoActuator`, `GearActuator`, `RelayActuator` — SIM-safe GPIO + momentary starter pulse)
  - `services/tractor_service.py` — steering/throttle/ground-speed/clutch/gear via RoboHAT RC-PWM plus starter/blade-PTO GPIO relays, with standard lawn-tractor interlocks (start requires authorized+neutral+clutch+blade-off; blade engages only with the engine running and not in reverse; reverse auto-disengages the blade; emergency stop disengages drive+blade and brakes while the engine keeps running)
  - `config/tractor.yaml`; 14 new REST endpoints under `/api/v2/tractor`
  - `ActionPrediction.to_tractor_command()`; the autonomous AI loop drives the tractor when `tractor.enabled`, the differential path otherwise
  - Frontend `TractorControlView` (`/tractor`) + API client + nav entry
  - `docs/tractor-platform.md`; 13 new unit tests

**Real-hardware readiness note (added 2026-07-08 with the v3.0.0 constitutional amendment)**: this commit implements the actuation *logic* only. The RC-PWM transport underneath the five positional actuators is not functional on real hardware as shipped (only the two GPIO relays work today); the service is never started at boot; and the navigation e-stop path does not yet reach the tractor. See `docs/tractor-acceptance-criteria.md` for what must be verified on real hardware before unsupervised use.

## Highlights

- Hardware telemetry behind SIM_MODE (T102, T110)
  - Default simulation mode (SIM_MODE=1) keeps CI/dev safe
  - Set `SIM_MODE=0` on the Pi to lazily initialize sensors via SensorManager and publish hardware-backed telemetry over WebSocket
  - Graceful fallback to simulated data on hardware errors

- Docs Hub polish (T099)
  - Backend serves markdown/text with `ETag`, `Last-Modified`, and `Cache-Control`
  - Frontend renders markdown safely using markdown-it + DOMPurify (sanitization tests included)

- Stability & health
  - Liveness/readiness endpoints verified
  - WebSocket telemetry with cadence control and reconnection
  - Privacy guard + log rotation retained

## Validation

- Backend: pytest – PASS (contract + integration tests green)
- Frontend: vitest – PASS (UI/composables + markdown sanitization green)
- Frontend build: Type-check + Vite production build – PASS
- Lint: Warnings only (types tightening optionally tracked)

## Deployment Notes

- Systemd services under `systemd/` can be used to run backend and related services
- To enable hardware telemetry on the device:
  - Ensure the `pi` user is in required groups (e.g., `i2c`, `dialout`)
  - Export `SIM_MODE=0` in the backend service environment
- Docs Hub available via `/docs` route in the WebUI; content served from `docs/` directory

## Breaking Changes

- API endpoints are under `/api/v2/*`; WebSocket at `/api/v2/ws/telemetry`
- Configuration and persistence aligned with v2 schema (migration tests included)

## Next Steps

- Add calibrated battery percentage mapping per chemistry
- Extend SensorManager coverage and readiness checks when `SIM_MODE=0`
- Optional TS type tightening to reduce `any` usage in frontend composables
