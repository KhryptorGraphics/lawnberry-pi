# Goal: finish the LawnBerry-pi 2026-10-24 workstreams

Paste the **Objective** block into goal mode (`omp` goal), or hand this file to any agent on Thor.
Written 2026-10-24 against `feat/2026-10-24-integration-audit` @ `29ca9e2` (draft PR #19).

## Objective

Finish every software-completable item from the 2026-10-24 integration audit
(`docs/2026-10-24-integration-audit.md`, PR #19) and the detector pipeline, deliver each as tested,
committed work in draft PRs, and leave a precise, current list of what only the owner can do.
"Done" means every track below meets its acceptance criteria **or** is recorded as blocked on a named
owner action (from the Human-gated list) with the exact evidence of the block. Do not shrink scope
by declaring items out of scope.

## Hard rules (apply to every track)

1. **No physical actuation.** Never drive servos, relays, starter, PTO/blade, motors or GPIO outputs on
   the mower Pi. Never run `i2cset`, `gpioset`, `pinctrl set` or `i2cdetect` there. Read-only inspection
   only, and only when a track says so.
2. **No pin, calibration or `tractor.enabled` changes** without the owner's written approval in the
   conversation. Propose them in docs and in `spec/known_conflicts.yaml` owner actions instead.
3. **Never merge to `main`, never auto-deploy.** Draft PRs only; the owner merges.
4. **One worktree per agent/branch** under `/home/kp/thordrive/`; never edit
   `/home/kp/thordrive/lawnberry-pi` (main checkout, owner's untracked files) directly.
5. **Evidence, not inference.** Every claim of "works" cites a command and its output. Unknown electrical
   facts stay TBD with the measurement needed (see audit B.7). Do not invent specs or pin numbers.
6. **Tests:** `SIM_MODE=1 pytest tests -o addopts="" -p no:cacheprovider --ignore=tests/soak -W ignore`
   must stay at 0 failures (current: 762 collected, 693 passed, 69 skipped). The suite rewrites
   `verification_artifacts/001-integrate-hardware-and/telemetry_integration_evidence.json`; restore it,
   never commit it. `ruff check` clean; update `docs/code_structure_overview.md` for interface changes.
7. **Shared machines:** workshop box `kp@192.168.1.53` (sudo `terramax123`), GPUs PCI order
   0 = P100, 1 = 3080 Ti, 2 = V100 (PG500). Never `pkill -f` a pattern that can match your own ssh
   command; kill by session id. Long jobs detached (`nohup setsid ... &`) with a log.

## Tracks

### T1 — Land the open PRs cleanly (do first)

- PRs: #19 (audit + safety fixes), #17 (agent bridge, OhMyPi default), #16 (bridge spec), #18 (requirements docs).
- Make CI green on #19 and #17 (`gh pr checks`); fix real failures, do not skip tests.
- Answer or fix every review comment (Sourcery etc.) with a commit or a reasoned reply.
- Rebase #17 and #19 on current `main` if they conflict; keep #16/#18 as reference (comment that #17/#19 implement them).
- **Acceptance:** all four PRs have green CI or a documented, non-code reason; a comment on each states what is left for the owner to decide.

### T2 — Remaining software safety work (from audit Part E, software-only work packages)

Each item: its own commit with fault-injection tests on fakes; update the audit's "Implemented" table.

1. **Obstacle stop wired for real:** production callers for ToF/ultrasonic obstacle triggers; the
   `go_to_waypoint` loop must read obstacle state; deterministic local override of any plan.
2. **Stopping-distance model:** speed-, latency- and sensor-uncertainty-dependent slow/stop thresholds
   (config-driven, conservative defaults clearly marked unvalidated), used by the nav loop.
3. **N-30:** Thor safe-hold retries blade-off until confirmed. **N-31:** GPS-degradation monitor fed by
   real GPS state, not only the debug route.
4. **Watchdog:** replace one-shot event-loop liveness with a real heartbeat from the actuation path;
   tractor watchdog trips to neutral; document that a hardware watchdog is still required.
5. **Thor link wiring:** behind a config flag (default off), instantiate `ThorLinkService` in
   `main.py`, feed its plan through the local safety supervisor only; add a minimal Thor-side client
   in `workshop/` or `scripts/` with key loading and `sid/ack` echo; end-to-end SIM test of link loss
   → hold → recovery.
6. **Four JSN-SR04T via a microcontroller (software side):** firmware for a dedicated MCU (RP2040,
   CircuitPython or C, sequential triggering, timeouts, per-sensor health) speaking a framed serial
   protocol; backend driver + `UltrasonicSensorInterface` support for 4 channels; SIM tests. No pins
   are assigned on the Pi header; the MCU link is USB serial. Mark wiring/thresholds TBD.
7. **IBT-4 blade driver:** real `lgpio` implementation behind the existing interface (still not run
   on hardware); the E-stop then reports `stopped` only from a confirmed write.
8. **Placeholder services and tests:** replace `lawnberry-sensors.service` placeholder with the real
   entrypoint or remove it; implement or delete the 40 placeholder tests the audit lists (A.11) — no
   tautologies.
- **Acceptance:** each item merged into one draft PR (or stacked on #19) with test names and counts.

### T3 — Deployment and security hygiene (repo side)

- `systemd/*.service`: no hard-coded `User=pi`; `scripts/install_services.sh` substitutes the real
  user; repo copies of the live-only HaLow units (`halow-bringup`, `halow-failover`, `wifi-halow-*`,
  `wificommand`, `thor-upload.conf` drop-in) after reading them read-only from the Pi.
- Persistent `LAWN_BERRY_AUTH_SECRET` generation at install time; installer refuses to finish with
  default credentials; remove remaining `operator123` defaults from code paths.
- Remaining doc fixes from audit A.12 (`deployment-guide.md` ports, `OPERATIONS.md`, `PRODUCT.md`,
  version naming).
- **Acceptance:** `install_services.sh --dry-run` (add if missing) shows the substituted units; tests
  for the installer's credential refusal.

### T4 — Deploy to the mower Pi (only when reachable and only after the owner merges)

- Pi: `kp@10.8.0.2` (via the girlfriend's network VPN) or `kp@192.168.50.20` (home HaLow). Sudo
  `terramax123`. Check reachability first; if unreachable, record it and move on.
- After the owner merges #19: `git pull` in `/apps/lawnberry-pi`, install the updated units, set the
  operator credential the owner provides, restart backend/camera/frontend, rebuild frontend, run
  `scripts/pi_detector_ctl.py status`, `/health`, login through the UI flow, unit suite on the Pi.
- **Acceptance:** command outputs recorded in the PR; auth verified on (401 without token, 200 with).

### T5 — Detector pipeline to a gate-passing HEF

1. **Box health first:** read `nvidia-smi` power/PCIe for the 3080 Ti. If it still shows ~400 W idle or a
   degraded link, do not use it; record readings and continue on V100/P100.
2. **Compile with DALI** on the V100: `DFC_ENV=/home/kp/repossata1/env-dfc-dali scripts/hailo_compile.sh
   --tag <tag> --gpu 2` (verify `dali_train_dataset` resolves to `dali_dataset`, not the mock).
3. **SDG v2 full run** (≥ 2000 frames, all 24 classes present; `workshop/dataset_stats.py` audit, no
   zero-instance class). Use the 3080 Ti only if healthy; otherwise the V100 with the non-RTX renderer
   fallback documented in `scripts/isaacsim-headless.sh`, and record image quality.
4. **Train** `yolo26s` on SDG v1 + v2 (+ any real frames that exist; none do yet) in `$LB/env` on the
   V100; export end2end ONNX; ONNX baseline eval.
5. **Compile** the trained model (opt level 4 + adaround, DALI), deploy only to a staging path, run
   `scripts/eval_detector.py --backend hef --gate --baseline <onnx report>` on the Pi (T4 rules) or
   record why not.
- **Acceptance:** per-class recall table ONNX vs HEF, gate pass/fail, HEF + sidecar sha256. Never
  replace the live `detector.hef` (zoo COCO) unless the gate passes **and** the owner approves.

### T6 — Cosmos and GR00T (Thor)

- Re-run the official GR00T N1.7 path (`~/thordrive/nvidia-sim/scripts/groot-n17-official.sh`) and the
  Cosmos guardrail path once the owner has accepted the HF licences; until then record the 403.
- Optional, only after T5.3: Cosmos Transfer augmentation of SDG frames (weather/lighting) into a
  separate dataset, audited with `dataset_stats.py`; do not mix into training without an ablation.
- **Acceptance:** smoke outputs with paths, or the exact 403 per model.

### T7 — Reporting

- Keep `docs/2026-10-24-integration-audit.md` "Implemented in this PR" and the release-gate table
  current. Final summary lists: PRs and their state, tests, what changed on each machine, and the
  remaining Human-gated items with the exact action needed.

## Human-gated (the agent cannot do these; it must ask, then continue other tracks)

| # | Owner action | Unblocks |
|---|---|---|
| O-1 | Merge decisions on #16–#19 | T4 |
| O-2 | Provide the operator credential for the Pi | T4 |
| O-3 | Accept HF licences for user `giggahost`: `nvidia/Cosmos-Reason2-2B`, `-8B`, `nvidia/Cosmos-1.0-Guardrail` | T6 |
| O-4 | Inspect the RTX 3080 Ti (power cables, seating/riser; reads ~415 W idle, PCIe x4); confirm whether `magnitude-inference` processes on all GPUs are yours | T5.3 on the 3080 Ti |
| O-5 | Approve pin moves: starter off GPIO5, blade IN2 off GPIO25, I2C devices off header bus 1 or resolve GPIO3 WAKE | enabling the tractor; I2C devices |
| O-6 | Audit B.7 measurements: Pi/Hailo/HaLow stack photos, servo labels + stall current, throttle actuator model, PTO switch behaviour, Toro serial plate, JSN-SR04T revision, LC29H carrier boards, Hailo-8 vs 8L | wiring netlist, power/fuse design, hardware sign-off |
| O-7 | Hardwired E-stop + PCA9685 OE + servo-power cut built and bench-tested (engine off, blades disconnected) | any powered actuation (gates G0/G1) |
| O-8 | Power-cycle the Pi / restore its network if unreachable | T4 |
