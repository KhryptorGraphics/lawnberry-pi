# Operator Dashboard (Frontend) — View Reference

The Vue 3 operator dashboard (`frontend/`) talks to the backend over `/api/v2`
REST + WebSocket. Routes are auth-gated (`requiresAuth`) except `/login`.

| Route | View | What it does |
|-------|------|--------------|
| `/` | Dashboard | Live telemetry tiles + map, WebSocket-driven |
| `/control` | Manual Control | Joystick drive, blade, **emergency stop**, and system controls (return-to-base / pause / resume) wired to the navigation API |
| `/maps` | Maps | The polygon editor — draw/edit boundary, exclusion, and mowing zones; set home/sun markers |
| `/planning` | Planning | Jobs, schedules, and a **Zones** tab that lists the real mowing zones from the saved map configuration (Add/Edit opens the Maps editor) |
| `/ai` | AI & Model Control | Enable/disable autonomous AI, deploy a `.hef` model, live inference metrics, dataset export — all via `/api/v2/ai/*` |
| `/telemetry` | Telemetry | Live RTK/IMU/power/hardware-stream metrics + diagnostic export, from the telemetry stream |
| `/rtk` | RTK Diagnostics | NTRIP throughput + GPS fix quality (`RtkDiagnosticsPanel`) |
| `/settings` | Settings | Hardware/network/telemetry profile, branding |
| `/docs` | Docs Hub | In-app documentation browser |
| `/mission-planner` | Mission Planner | Map-based waypoint mission planning |

## Control surface ↔ backend

- Manual control: `POST /api/v2/control/{drive,blade,emergency}` (session-gated).
- System controls: `POST /api/v2/navigation/{pause,resume,return}` and
  `/control/mode` (operator-auth).
- AI/model: `GET /api/v2/ai/status|metrics|health|datasets`,
  `POST /api/v2/ai/{enable,disable,model,metrics/reset}`,
  `POST /api/v2/ai/datasets/{id}/export`.
- Zones: `GET/PUT /api/v2/map/configuration` (the Maps view is the editor of
  record; Planning reads the same zones).

## Build / test

```bash
cd frontend
npm run type-check   # vue-tsc --noEmit
npm run test         # vitest
npm run build        # vue-tsc && vite build
```
