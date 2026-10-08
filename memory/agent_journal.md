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

## 2026-08-02 — Green the e2e suite (stale specs, not real regressions)

`build-ui` has been failing on every PR since the June AIView rebuild. It runs
`on: [pull_request]` with no path filter and never on `main`, so nothing on the
default branch ever reported it and the rot accumulated invisibly.

Measured a real baseline on clean `main` (3 failed / 2 passed), which showed
neither failure had anything to do with the PR that surfaced them:

- **`ai-training.spec.ts` (2 tests)** — asserted a heading `"AI Training"` and a
  `#ai-start-training` button. The 2026-06-26 frontend pass replaced the mock
  training studio (which called non-existent `/api/v2/training/*` endpoints)
  with the real AI & Model Control panel. `grep -c training src/views/AIView.vue`
  is now 0 — the UI under test no longer exists, so the spec was deleted rather
  than repaired. **AIView currently has no e2e coverage.**
- **`manual-control.spec.ts:27` (fail-closed)** — `getByText(/unlock is
  unavailable/i)` hit a Playwright strict-mode violation by matching 3 elements
  (toast, the gate card's inline error, the view-level status alert). The
  behaviour was always correct; only the selector was over-broad. Scoped it to
  `.security-gate .alert-danger`.

Verified: full `npx playwright test` → 12 passed, 0 failed.

Local note: on aarch64 the pinned chromium-1140 download wedges mid-extract, and
the newer chromium-1169 build rejects Playwright 1.48's `--headless=old`. The
already-present `chromium_headless_shell-1217` works. CI is unaffected (it uses
the official Playwright container).

## 2026-08-02 — Auth removed project-wide (PR #8, rebased)

Landed the long-open auth-removal branch, rebased from its July base (8547cb1)
onto current main. Deliberate product decision for a LAN-only, single-operator
unit with no internet exposure: login, operator-auth-gated writes, and the
manual-control unlock step were friction with no real security benefit here.

Mechanism is one existing toggle, not a teardown. `OPERATOR_AUTH_REQUIRED`
already gated `require_operator_auth`; this extends it to `_resolve_manual_session`,
`manual_unlock`, and `manual_unlock_status`, and flips the systemd unit to 0.
It defaults to "1" when unset, so auth stays ON anywhere the env var is absent
— tests included. The login view and JWT service are untouched; re-enabling is
one env var.

**The fail-closed contract survives.** Frontend auto-unlock is a silent attempt
on mount whose failure path falls through to the normal gate. Verified rather
than assumed: the e2e fail-closed test (forces 404) still passes unmodified,
and ControlView.unlock.spec.ts (7 tests: 404/501/generic/three Cloudflare
paths) is green.

Two e2e tests asserted the gate and had to be rewritten to the no-auth
contract. Only one was named in CI's failure list — the second
(`raises safety lockout when drive command is blocked`) fails on
`getByLabel('Confirm Password')` once the gate stops rendering, and was caught
by running the spec against the branch instead of trusting the CI list. Worth
remembering: CI's reported failures aren't necessarily the complete set when
earlier assertions short-circuit a spec.

docs/authentication-config.md described the auth system as active throughout;
it now opens with the disabled-state notice and re-enable instructions.

Verified: playwright 12/12, vitest 132/132, vue-tsc, backend auth-related
tests 52 passed, full backend suite unchanged vs. baseline.

## 2026-08-02 — Live 500 on /ai/status + model paths off a dev home dir

Probed the deployed Pi while confirming the model-path follow-up (issue #13) and
found `GET /api/v2/ai/status` returning **500** on the running unit. Every other
endpoint probed was 200 (`ai/health`, `ai/metrics`, `ai/datasets`,
`dashboard/telemetry`, `camera/status`) — so the AI dashboard panel's primary
call was the one thing broken, and no test covered it.

**Enum coercion, not a typo.** `AIControlStatus` sets
`ConfigDict(use_enum_values=True)`, so pydantic coerces `mode` to a plain `str`
— but *only during validation*, not for field defaults. `AIControlStatus()`
keeps a real `ControlMode`; `AIControlStatus(mode=...)` yields a `str`.
`get_status()` passes `mode=` explicitly, so the router's `status.mode.value`
raised `AttributeError` → 500. Replaced with `str(...)`, which is correct for
both shapes (these are `StrEnum`s).

Bounded the sweep by model rather than by directory: enumerated every model with
`use_enum_values=True`, collected their enum-typed fields, and grepped all of
`backend/src/` for `.value` on those names. Five real sites — `ai_control.py`
(the live 500) plus four latent `navigation_mode` accesses in `api/status.py`
and `api/navigation.py` that fire whenever navigation mode is explicitly set.
`auth.py`'s `role.value` was checked and is safe: the `use_enum_values` in
`user_session.py` belongs to `UserSession`, not `SecurityContext`.

**Model paths (issue #13).** `ai_inference_service` and `hailo_driver` hardcoded
`/home/kp/repos/lawnberry_pi/...` — a developer home dir that is neither this
repo nor the deploy root. Now resolved from `LAWNBERRY_DATA_DIR` (default
`./data`), matching maps/planning/settings. Resolved at **call time**, not in the
class body: class attributes evaluate at import and would capture whatever env
existed then, breaking monkeypatched tests. Added the WARNING log #13 asked for,
plus an `OSError` guard — `.exists()` raises rather than returning False when a
path component is an unreachable mount.

That guard is also what cleared the 12 long-standing failures in
`tests/unit/test_ai_inference_service.py`: same root cause, louder symptom.
**Full backend suite is now 0 failures.**

Verified on the live unit: no `.hef` exists anywhere on the Pi, so the bad
default was never masking a working model — the bug was latent. Noted on #13
that `POST /ai/model` doesn't persist the path either, so a deployed model
wouldn't survive a restart; not fixed here.

New contract test `test_get_ai_status_serializes_control_mode` — confirmed it
returns 500 without the fix and 200 with it, rather than assuming.

## 2026-10-06 — Toro zero-turn conversion, autonomy pipeline, W1 capture

Branch `feat/toro-zero-turn-platform`. The tractor platform moved from the
never-built Craftsman Ackermann design to a Toro TimeCutter MAX 50" MyRIDE
(model 77502) zero-turn: twin-lever servos, the reverse rule "both levers
negative", and IMU tilt-cutoff plus a watchdog wired to the tractor
(constitution v4.0.0). The printable mounts were redesigned around measured
splayed servo arms and estimated 940 mm pushrods. They are fit/load-test
prototypes only. Autonomy work added the Pi↔Thor link and strategist (W6/W7),
the workshop SDG/training pipeline (W2/W4/W5), and the W1 capture logger with
its integrity gate.

Verified facts that change earlier assumptions:

- The Pi's accelerator is a **Hailo-8L**, not a Hailo-8. Compiles target
  `hailo8l`. HailoRT 4.24 benchmarks: yolo26s 41.7 FPS, yolo26m 20.6 FPS.
- **Stopping `lawnberry-backend` has taken the Pi off the network.** It
  also stops the camera (PartOf=). About 10 s later the HaLow SPI driver
  failed (`ret:-71`) and only a power-cycle recovered it. Restart the
  backend only with physical access.
- **The IMU is not usable.** `config.txt` has `dtparam=i2c_arm=off`, and
  `BNO085Driver` returns a constant 0/0/0 on hardware. Buses 13/14 ACK every
  address, so probing them would fake an ONLINE level IMU and silently
  disable tilt cutoff.
- **The latest SDG v1 run passed the rejection gate with blown-out RGB.**
  37 of 40 sampled frames were white. The gate checks labels, not image
  sanity. The run is kept as `v1-blown` and needs a re-render.

Not done: the 60 s Pi capture session (indoors, no RTK; backend-restart
hazard above), real IMU support, DFC compile (wheel needs a Hailo login).

## 2026-10-08 — Servo envelope, mount reachability and a size-independent clamp

The printable mounts workstream continued: the RDS51150 servo interface, the pushrod
datum it feeds, the arm-head holder fastening, and the lap-bar clamp. It corrects
several numbers the 2026-10-06 entry recorded.

Verified facts that change earlier assumptions:

- **The 61.4 mm on the vendor drawing is the servo's overall axial envelope, not a
  disc diameter.** The case is 65 × 30 × 48 with the output shaft along the 48 mm
  dimension; the kit's disc adds 13.4 mm. The model had the shaft along the 30 mm
  dimension and a Ø61.4 × 3 mm "disc", putting the crank's inner face at 33 mm instead
  of the drawing's 61.4. Correcting it moved the crank pin 28.4 mm outboard, so the
  rods re-derived: **936.9 mm** pin-to-pin (was 940), lap-bar pin 74.2 mm outboard of
  the crank (was 103), built skew 4.54° (was 6.3°). Earlier commits and docs still
  quote 940 mm.
- **The holder's 4-M2.5 pattern is concentric with the output**, so the arm head's
  four bores line up with the holder's pattern *and* the rotation axis at once. The
  ~19 mm offset the dimensioned drawing appears to show is a composite of the two case
  ends: that offset would bury a hole inside the Ø26.5 mm output opening carried by the
  same face. Established from photos of the real boss, calibrated twice (case 65 mm =
  498 px; tape 1/8 in ticks), not from the drawing.
- **The holder could not be bolted on.** Its four M2.5 screws must come from the arm's
  inboard side — the holder covers the outboard face — but the rib bracing the servo
  plate was solid behind the bores. Each head now has a Ø4.8 driver tunnel per bore and
  loses 1148 mm³; nothing else in the set moved.
- **`arm_servo_pattern` was passing vacuously** — the trap its own comment claimed was
  closed. It framed only the head with `arm_vertical()` and placed its rays in bare
  absolute coordinates ~190 mm away, so it intersected nothing; enlarging the ray to
  Ø14 still returned empty, which is how it was caught. Both children are framed now and
  the new `arm_servo_access` check proves the two-stage screw path. A spot-check of
  `arm_brace_bolt_bores` with a Ø20 ray finds material, so that pose-frame class is not
  affected.
- **The clamp is size-independent AND measurement-free**: a fixed Ø36.4 bore with a set of
  split sleeves spanning 20 / 22.2 / 25.4 / 28.6 / 30 mm bar OD, so one body, bolt set,
  yoke and pin serve every bar and the size is chosen by *trying the sleeves on the bar* —
  each sleeve's bore is its own slip fit, so the sleeve is its own coupon and the separate
  gauge parts are retired. Proved by rendering: the anchor measures 110949.3 mm³ at both a
  Ø25.4 and a Ø22.2 bar, and the five sleeves are single solids whose volumes fall
  19658 → 8504 mm³ as the bore grows.
- **The kit crank's pin radius and the disc's OD no longer need measuring for clearance.**
  `rod_servo_sweep_band` sweeps 40–70 mm pin radius × Ø26–34 disc × the full ±35° travel on
  both sides and passes. What a different radius still implies is *length* — up to 30 mm of
  pin travel — which the rod's trim does not separately cover, so the rod's length stays
  estimate-driven until its adjustment range is widened (multi-position spigot splices).
- **The rod's length is now set by the parts** — landed, after one failed attempt. Four outer pieces
  with two multi-position spigot joints give **210 mm of adjustment in 15 mm steps** (against ±15 mm
  before), covering the fore-aft estimate's ±4 in without a tape. The joint feeding the inner bar stays
  short and single-position because its spigot shares the bore the inner slides in — hence a fourth
  piece rather than a longer overlap, and three assertions forced exactly that shape. The first
  attempt failed on `rod_lock_bores`, and the cause was the CHECK, not the design: it hard-coded one
  bolt list for every joint, so probing the row offsets at the short joint read material. The offsets
  are now per joint (`rod_splice_offsets_at`), splice probing moved to `rod_splice_bores` (extremes of
  each row), and the servo-end stub shortened 90 → 70 mm to suit the shorter first piece — which
  `rod_servo_sweep` and `rod_servo_sweep_band` still pass.
- **A dual-lens machine-vision camera now has its own station on the column**, 304.8 mm (12 in) below
  the cap's Pi Camera Module v2.1, so both axes stay parallel. `stereo_camera_bracket` is a collar
  around the column with a web and a front plate facing the same way as the cap's carrier: one 34 × 22
  lens window (baseline-tolerant) and two 12 mm body slots at 24 mm pitch (PCB-pattern-tolerant), since
  the SVPRO camera's own dimensions are still unverified and the design refuses to invent them. Its
  bolt pair crosses the column, so the bracket's holes are the drill guide at whatever height you set;
  the column's brace bores are the only pre-drilled alternative. Errors caught while building it: the
  tower's bore constant lives in the tower file and not the shared one, and an inner cut larger than the
  collar removed the web's inner half, leaving two bodies - the mesh check refuses that.
  The first version of its fit check also PASSED VACUOUSLY: it called a placement helper that used
  `tower_cap_top_z()`, which lives in the tower file, so the bracket was translated to an undefined z
  and the probe intersected empty space. Caught by the build (the part itself still rendered, since it
  never calls that helper) and then by a deliberately fat probe, which is the only way to tell an
  empty check from a vacuous one: the fat probe now finds real material at the collar wall.
- **The camera column is drilled through at the sway-bar stations.** The bars' collars clamp around
  the column, but their M4 clearance holes only pierced the collar's own wall, so the screws gripped
  nothing: the column itself had no holes. `tower_brace_bores()` now cuts through the COMPLETE square
  tube, both walls, at each station — one bolt per station makes collar pair plus column a single
  bolted joint. Station 0 (z=110) lands on the base, which reaches z=160; the rest fall inside the
  first segment at local z=30 and 90, and the segment part is shared, so every segment carries the
  same bores. An assert refuses a station outside the first segment, and `tower_brace_bores` probes
  the screw line at every station.
- **OpenSCAD's STL export is not reliably byte-stable**, so `validation.json`'s per-part
  bytes hash cannot fingerprint a shape. Parts now also carry `mesh_sha256`, an
  order-independent digest that is stable across runs and moves only on a real change.
- Slicer projects are audited, never rewritten; the three stale ones — one still holding
  the pre-fix rod — were removed at the user's instruction, and the audit now reports none.

Not done: the clamp point along the lever, which is a constrained choice rather than a reading; any
check on the sleeve set against a real bar; and the tower stations' collar/screw hardware, which the
column now accommodates but no screw is specified. No part is load-, creep- or powered-steering
qualified; the printed linkage remains first-article fit prototypes.
