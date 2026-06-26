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
