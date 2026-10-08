<!-- Generated: 2026-10-06 | Files scanned: 67 (frontend/src) | Token estimate: ~550 -->
# Frontend (Vue 3 + TS, Pinia, vue-router, Vite; `frontend/`)

## Page tree (src/router)
```
/                 DashboardView
/control          ControlView (1446 L; manual drive, camera feed, E-stop)
/tractor          TractorControlView (zero-turn levers/throttle/PTO; platform-gated)
/maps             MapsView (1201 L; Leaflet/Google, zones, perimeter)
/mission-planner  MissionPlannerView
/planning         PlanningView (zones, schedules, jobs cards)
/telemetry        TelemetryView
/rtk              RtkDiagnosticsView
/ai               AIView
/settings         SettingsView
/docs             DocsHubView
/login            LoginView
```

## Components (src/components)
control/, dashboard/ (EngineCard, CalibrationCard, …), map/, mission/ (MissionWaypointList), planning/ (ZonesCard, SchedulesCard, JobsCard), ui/, plus top-level MetricWidget and TopProgress.

## State and transport
```
services/api.ts (axios, baseURL defaultBase) ──REST──► /api/v2/*
services/websocket.ts ──WS──► /api/v2/ws/telemetry ─► topic handlers
services/auth.ts
        ▼
stores/: control, tractor, autonomy, mission, map, system, auth, userSettings, preferences, toast, confirm
composables/: useCameraFeed (MJPEG /api/v2/camera/stream.mjpeg), useOfflineMaps, useFocusTrap
```

## Serving
`server.mjs` (Express + http-proxy-middleware + compression) serves `dist/` and proxies `/api` and the WS to the backend on :8081. In dev, the Vite dev server does the same.

## Tests
vitest unit tests (`frontend/tests`) and Playwright e2e (`npm run test:e2e`; on aarch64 use LB_CHROME_PATH).
