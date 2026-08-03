# Agent Journal

Chronological record of substantive automated changes (required by the
`pr-hygiene` guard when code under `backend/`, `frontend/`, `systemd/`,
`scripts/`, `spec/`, or `docs/` changes).

## 2026-06-25 — Production-readiness pass

Brought the project from a red test suite to a production-ready control surface
(branch `feat/production-readiness`, PR #1):

- **Security:** fail-closed Google OAuth verification; bearer auth on WebSocket
  topics; operator auth (`OPERATOR_AUTH_REQUIRED`) on config-changing write
  endpoints; fixed sanitization redacting login tokens.
- **API:** implemented the missing `/api/v2` surface (map configuration,
  settings, weather, AI dataset export, planning jobs, docs hub, verification
  artifacts, health rollup) + auth rate-limit/lockout.
- **Autonomy:** real autonomous mowing orchestration (zone coverage → mission →
  motors), navigation control API (`/navigation/{start,stop,pause,resume,status}`,
  `/control/mode`), and planning-job lifecycle.
- **Hardware paths:** real EKF sensor fusion; YOLOv8 detection decoder for the
  camera; RoboHAT serial delegation; multi-zone coverage with exclusions.
- **ACME/TLS:** real certbot-backed issuance/renewal/revocation.
- **Frontend:** aligned API integration to the v2 contracts; WS access-token in
  the handshake URL; autonomous control store + wiring.
- **Ops:** CI ruff lint workflow; `docs/production-readiness-validation.md`
  on-device checklist.

Backend suite: 368 passed / 0 failed (was 51 failed). Frontend: build + 83
vitest tests green.

## 2026-06-26 — Pi→Thor data path + location-aware autonomy

Branch `feat/pi-location-aware-ai-loop` (consolidates the earlier
`feat/thor-ingest-and-pi-setup` work):

- **Pi→Thor upload:** recordings auto-queue for upload to the Thor training
  receiver on stop (`THOR_UPLOAD_ENABLED`/`THOR_BASE_URL`); env-driven uploader
  config; systemd env wiring.
- **Location-aware inference:** new `backend/src/nav/location_features.py`
  (runtime port of the training feature builder) turns RTK GPS into local-ENU
  location features + a causal 64×64 coverage map. `ai_inference_service`
  `_preprocess` now emits `image` + `sensors(20)` + `coverage_map` for the
  distilled student/HEF.
- **Autonomous loop:** implemented the previously-missing AI inference loop in
  `navigation_service` (build frame via `perimeter_recorder.capture_frame` →
  infer with live coverage snapshot → `apply_ai_prediction` → causally mark
  mowed cells). Datum from home/geofence. Safety: refuses autonomy on hardware
  when no real model is loaded.
- **Docs:** `docs/ai-training-pipeline.md` (record → upload → train → distill →
  deploy → autonomous, incl. the yard-datum contract).

New unit tests for location features, preprocess, and the loop; full unit suite
+ ruff green.

## 2026-06-26 — Frontend completeness pass

Audited the operator dashboard (type-check/build/83 vitest all green) and fixed
the real gaps (see docs/operator-dashboard.md):

- **ControlView**: return-to-base / pause / resume were no-op placeholder
  toasts; wired to the navigation API. Added `POST /api/v2/navigation/return`
  (`autonomy_service.return_to_base` → `NavigationService.return_home`).
- **TelemetryView**: was a "coming soon" stub; rebuilt as a live RTK/IMU/power/
  hardware-stream view with diagnostic export, from the system telemetry store.
- **PlanningView** zones: replaced mock zones with the real mowing zones from the
  saved map configuration (areas computed from polygons); Add/Edit now open the
  Maps polygon editor instead of dead-ending at "coming soon".
- **AIView**: replaced a 1764-line mock image-labeling/training studio (which
  called non-existent `/api/v2/training/*` endpoints) with a real ~370-line
  AI & Model Control panel wired to `/api/v2/ai/*` (status, enable/disable,
  model deploy, metrics + reset, health, datasets + export).

Verified: vue-tsc, vitest 83/83, vite build, backend ruff + autonomy tests green.

## 2026-08-02 — Green the lint job on main (ruff/black split)

`main` had a red `lint` job. Two independent faults gated sequentially behind
`bash -e`, so fixing either alone left it red:

1. `ruff check .` — 2× E501 in `scripts/check_hardware_pin_conflicts.py`
   (new in fa438a8), so the job died before ever reaching the format step.
2. `ruff format --check .` — `backend/src/api/routers/camera.py` and
   `tests/unit/test_gps_config_fallback.py` were unformatted.

**Root cause, not just symptom:** `CONTRIBUTING.md` told contributors to format
with `black`, but CI enforces `ruff format --check` and black appears in no
workflow. They genuinely disagree — verified on camera.py, where black leaves
`(...).encode() + jpeg_bytes + b"\r\n"` inline and ruff explodes it. The
committed file was black-formatted, i.e. someone followed the docs and CI
rejected it. `docs/OPERATIONS.md` was worse: it ran `ruff format .` *then*
`black .`, actively undoing the CI-correct result.

Retired black repo-wide: CONTRIBUTING.md, docs/OPERATIONS.md (both blocks),
.github/pull_request_template.md, and the now-dead `[tool.black]` section in
pyproject.toml. `ruff format` is the single source of truth.

Verified: `ruff check .` + `ruff format --check .` both green (344 files),
`check_hardware_pin_conflicts.py --self-test` OK.

**Out of scope, filed for follow-up — and it looks like a PRODUCTION bug, not
just a test-env one:** `ai_inference_service.DEFAULT_MODEL_PATH` and the two
`hailo_driver` model paths are hardcoded to `/home/kp/repos/lawnberry_pi/...`,
which is not this repo's location and not the deploy location either (the Pi
runs from `/apps/lawnberry-pi`). `model_path` is overridable via service config
but is set in no `config/*.yaml` or `*.json`, so the hardcoded default is what
actually resolves. On the Pi that path won't exist, `_model_loaded` stays False,
and autonomous inference silently reports no model loaded — no error raised,
just a quiet no-op.

Locally it's noisier: the path is a dead mount here, so `.exists()` raises
`OSError: [Errno 19]` instead of returning False, failing 12 tests in
`tests/unit/test_ai_inference_service.py`. CI never sees either symptom (no such
path, and `.exists()` returns a clean False).

Pre-existing on clean main; untouched here. Filed as issue #13 — verify against
the deployed unit before assuming the mower's autonomy has been dark.
