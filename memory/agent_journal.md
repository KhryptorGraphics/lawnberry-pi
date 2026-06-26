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
