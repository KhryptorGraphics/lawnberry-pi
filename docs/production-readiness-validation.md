# Production Readiness & On-Device Validation Checklist

This checklist covers validating the production-readiness work (security fixes,
new `/api/v2` endpoints, real ACME/EKF/camera-AI/jobs) on the actual Raspberry Pi
5 + Hailo 8L hardware. Everything in CI runs under `SIM_MODE=1`; the items below
are the things that can only be confirmed on-device.

> Convention: `[ ]` = to verify on-device. Run from `/apps/lawnberry-pi`.

## 1. Dependencies (aarch64)

- [ ] `uv sync --frozen --all-extras` resolves and installs on the Pi (Linux aarch64).
- [ ] New deps land cleanly: `google-auth`, `cryptography`, `pyasn1*` (all pure-python or
      have aarch64 wheels). Confirm: `uv run python -c "import google.oauth2.id_token, cryptography; print('ok')"`.
- [ ] `certbot` is installed (`which certbot`) — required by the ACME service.

## 2. Service bring-up

- [ ] systemd units start: `systemctl status lawnberry-backend lawnberry-frontend lawnberry-camera lawnberry-sensors`.
- [ ] Backend runs with `SIM_MODE=0` (set in `lawnberry-backend.service`).
- [ ] `curl -fsS http://127.0.0.1:8081/health` returns `status: healthy` with
      `message_bus / drivers / persistence / safety` all healthy on real hardware.

## 3. Security / auth (new)

- [ ] Set the operator credential (`LAWN_BERRY_OPERATOR_CREDENTIAL` / secrets) and confirm
      `POST /api/v2/auth/login` issues a bearer token.
- [ ] Confirm the new write-endpoint authorization (added in this PR): `PUT /api/v2/settings`,
      `PUT /api/v2/map/configuration`, `POST /api/v2/planning/jobs`, AI export, and
      `verification-artifacts` reject unauthenticated writes (401) and accept an authenticated one.
- [ ] Google OAuth (if enabled): a tampered/invalid `id_token` is **rejected** (fail-closed),
      and login does NOT fall back to a default identity.
- [ ] WebSocket topics (`/api/v2/ws/telemetry|control|settings|notifications`) reject a
      handshake without a bearer token (401) and accept one with it (101).
- [ ] Login rate-limit + lockout: repeated logins / failed attempts return `429` + `Retry-After`.

## 4. Hardware control paths (genuinely unvalidated off-device)

- [ ] **RoboHAT serial**: a manual drive command (`POST /api/v2/control/drive` with a valid
      session) actually moves the wheels; `MotorService.send_drive_command` reaches the
      `RoboHATService` serial bridge (check logs for the `pwm,...` line, not the
      "no serial bridge" debug message).
- [ ] **Blade**: blade engage/disengage works and respects the safety lockout (423 path).
- [ ] **E-stop**: `POST /api/v2/control/emergency-stop` halts motors + blade immediately.

## 5. Sensors → fusion (EKF)

- [ ] GPS/RTK + IMU feed the SensorManager; `GET /api/v2/fusion/state` shows the EKF
      converging: `uncertainty.position_std_m` shrinks with RTK fix and `quality` reaches
      `good`; heading tracks the IMU.
- [ ] Drive a known straight line and confirm fused `position.x/y` advances along the heading.

## 6. Camera AI (Hailo)

- [ ] `lawnberry-camera.service` owns the camera; backend consumes via IPC (per constitution).
- [ ] With a real `yolov8m` HEF loaded, frames produce **real** detections
      (`ai_annotations[*].objects` non-empty, `source: "hardware"`), using the new YOLOv8
      post-processor. Verify against a known object in view.
- [ ] Inference latency (`processing_time_ms`) is within budget (Hailo 8L ~5–20 ms).

## 7. ACME / TLS (needs a real domain + DNS)

- [ ] **Staging first** (`ACME_STAGING=1`): `python -m backend.src.cli.acme_renew --domain <d> --dry-run`
      validates the HTTP-01 flow; then a real staging issuance writes
      `<ACME_LE_DIR>/live/<domain>/fullchain.pem`.
- [ ] Promote to production (`ACME_STAGING=0`), issue, and confirm nginx serves HTTPS.
- [ ] `lawnberry-acme-renew.timer` runs and `acme_renew` renews certs within the 30-day window.

## 8. Autonomous control & jobs

Direct autonomous control API (auth-gated writes; status is open):

- [ ] With a boundary drawn, `POST /api/v2/navigation/start` (optionally `{"zones":[...]}`)
      returns `{status:"started", mode:"autonomous", mission_id, waypoint_count>0}` and the
      mower begins driving the coverage path.
- [ ] `GET /api/v2/navigation/status` reflects `mode`/`active`/`completion_percentage` as the
      run progresses.
- [ ] `POST /api/v2/navigation/pause` halts motion (mode `paused`); `/resume` continues;
      `/stop` aborts and returns to `idle`.
- [ ] `POST /api/v2/control/mode {"mode":"autonomous"|"idle"|"manual"}` starts/stops a run.
- [ ] **E-stop during an autonomous run** aborts the mission cleanly and the status returns to idle.

Planning jobs (a saved job = an autonomous run over its zones):

- [ ] Create a boundary + exclusion in the map UI, then a scheduled mow job over those zones.
- [ ] `POST /api/v2/planning/jobs/{id}/start` dispatches a real mission (coverage waypoints from
      the zones, avoiding exclusions); `/pause`, `/resume`, `/cancel` drive the run; job
      `status` updates accordingly.

## 9. Frontend integration

- [ ] `npm run build` artifact served by `lawnberry-frontend`; dashboard loads.
- [ ] **Login issues a usable bearer token** (the sanitization exemption is in place) that the
      API client attaches; protected writes succeed authenticated and 401 without.
- [ ] **Telemetry WebSocket authenticates remotely**: the client appends `?access_token=` and
      `/api/v2/ws/telemetry` upgrades (101) from a non-loopback origin.
- [ ] Control view drives manually (auto manual-unlock session); Planning "Quick Mow" / job
      Start triggers an autonomous run and progress is visible.
- [ ] Map planner reads/writes `/api/v2/map/configuration`; settings, weather, planning, and
      telemetry views populate from the live backend (no console 404s/shape errors).

## 10. Endurance & rollback

- [ ] HIL suite on-device: `RUN_PLACEHOLDER_INTEGRATION=1 SIM_MODE=0 pytest tests/hil`.
- [ ] Soak: `pytest tests/soak` (long-running) — telemetry stable, no leaks.
- [ ] Telemetry performance budgets on the Pi: `RUN_PERF_TESTS=1 pytest tests/integration/test_telemetry_perf.py`.
- [ ] Rollback path verified: `systemctl` stop/disable, restore previous release, restart.

## Known follow-ups (tracked, not blockers)

- Camera-AI: class map / thresholds may need tuning for the deployed model.
- ACME: production issuance is an ops cutover (requires the real domain/DNS to be live).
