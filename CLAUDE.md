# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LawnBerry Pi v3 — an autonomous lawn mower. A **FastAPI backend** (Python 3.11) runs on a Raspberry Pi 5 + Hailo 8L, driving real hardware (RTK GPS, stereo/Pi cameras, ultrasonic/ToF sensors, IMU, RoboHAT RP2040 motor controller, IBT-4 blade driver). A **Vue 3 + TypeScript frontend** serves the operator dashboard. AI/VLA model *training* lives separately on an NVIDIA Thor server; the Pi only runs INT8 inference. Deployed on the Pi under `/apps/lawnberry-pi` via systemd.

## Commands

All backend commands run from the repo root (the package is `backend.src`, `pythonpath = ["."]`).

```bash
# Backend deps (editable; [hardware] extra pulls Pi-only GPIO/I2C libs)
python -m pip install -e .[hardware]

# Run the API locally (no hardware — see SIM_MODE below)
SIM_MODE=1 uvicorn backend.src.main:app --host 0.0.0.0 --port 8081 --reload

# Tests — ALWAYS set SIM_MODE=1 off-device, or hardware probes will fail/hang
SIM_MODE=1 pytest                       # full suite (testpaths=tests)
SIM_MODE=1 pytest tests/unit            # one directory
SIM_MODE=1 pytest tests/test_mission_api.py::test_name   # one test
SIM_MODE=1 pytest tests/contract        # contract tests (one per API/sensor)

# Lint / format (line-length 100; ruff selects E,F,I,UP,B)
# CI runs both and fails on either. Never use black — it disagrees with
# ruff format on some constructs, so black-formatted code fails CI.
ruff check .          # add --fix to autofix
ruff format .         # CI: ruff format --check .

# Frontend (from frontend/)
npm install
npm run dev           # vite dev server
npm run build         # vue-tsc type-check + vite build
npm run test          # vitest
npm run test:e2e      # playwright (builds first)
npm run lint          # eslint --fix
npm run type-check    # vue-tsc --noEmit
```

`SIM_MODE=1` is the single most important env var for development: it makes services use simulated sensors/actuators and skips hardware I/O (GPIO XSHUT addressing, I2C/serial probes). CI and any off-Pi work must set it. On the Pi, systemd sets `SIM_MODE=0`.

## Architecture

**Backend (`backend/src/`)** is a layered FastAPI app assembled in `main.py`:
- `api/` + `api/routers/` — HTTP/WebSocket route modules, each mounted in `main.py`. `rest.py` is a legacy monolith (carved-out ruff ignores in `pyproject.toml`); newer endpoints live in `rest_v1.py` and `api/routers/`.
- `services/` — the domain layer; one long-lived service object per concern (motors, blade, sensors, navigation, missions, telemetry/websocket hubs, camera, NTRIP, ACME/TLS, settings, weather). Most are reached through `get_<thing>_service()` accessor functions, not direct construction.
- `drivers/` — thin hardware abstractions (`drivers/motor/robohat_rp2040.py`, `drivers/blade/ibt4_gpio.py`, `drivers/sensors/*`, `drivers/ai/hailo_driver.py`). Hardware libs are **imported lazily inside functions** so the package stays importable (and testable) without Pi hardware.
- `nav/` — pure-ish navigation algorithms (coverage patterns/planner, geofence validator, path planner, obstacle avoidance, odometry, GPS degradation). These are the most unit-testable modules.
- `safety/` — independent safety chain: E-stop handler, watchdog, interlock/motor-authorization validators, `safety_monitor`. `validate_on_start()` runs at boot and safety triggers gate motor commands. Treat changes here as high-risk.
- `core/` — config loading (`config_loader.py` reads `config/hardware.yaml` + `config/limits.yaml` into `app.state` at lifespan startup), message bus/IPC, persistence, secrets, logging/observability, env validation.
- `middleware/` — security, API-key auth, rate limiting, input validation, sanitization, correlation IDs — all registered in `main.py`.
- `cli/` — operator/maintenance entrypoints (safety, sensors, control, secrets, ACME renew, remote-access daemon).

**Config layering**: `config/*.yaml` + `config/*.json` (hardware, limits, logging, default, nginx) loaded at startup; secrets via `config/secrets.json` and `.env` (loaded early in `main.py` for `NTRIP_*` etc.). `LAWNBERRY_CONFIG_DIR`/`_LOG_DIR`/`_DATA_DIR` override locations on-device.

**Frontend (`frontend/`)** — Vue 3 Composition API + Pinia + vue-router, Vite build, socket.io + axios to the backend, Leaflet/Google maps for the mission planner. `server.mjs` is an Express static/proxy server used in production.

**Deployment** — `systemd/` units run the stack on the Pi (`lawnberry-backend`, `-frontend`, `-camera`, `-sensors`, `-database`, plus ACME/cert/backup timers). Per the constitution (`docs/constitution.md`, `.specify/memory/constitution.md`): camera-stream.service exclusively owns the camera (others consume via IPC); Coral/edgetpu deps are banned from the main env and isolated in a separate venv.

## Conventions

- **Never assume hardware is present.** Guard hardware code paths and import hardware libs lazily; keep modules importable under `SIM_MODE=1`.
- **TODOs must be `TODO(vX): description - Issue #NNN`** — a pre-commit hook (`scripts/pre-commit-todo-check.sh`) and CI enforce this; `FIXME`/`XXX`/`HACK` are rejected. See `CONTRIBUTING.md`.
- Commit messages follow Conventional Commits (`feat(scope):`, `fix(scope):`, …).
- This is a spec-kit project (`.specify/`, `spec/`); structural decisions are governed by the constitution — check it before changing service ownership or dependency isolation.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **lawnberry-pi** (12639 symbols, 20728 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/lawnberry-pi/context` | Codebase overview, check index freshness |
| `gitnexus://repo/lawnberry-pi/clusters` | All functional areas |
| `gitnexus://repo/lawnberry-pi/processes` | All execution flows |
| `gitnexus://repo/lawnberry-pi/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

## PLUR Memory

You have persistent memory via PLUR. Corrections, preferences, and conventions persist across sessions as engrams.

### Architecture

PLUR is installed **globally** — one MCP server, one engram store (`~/.plur/`), available in every project. You do NOT need per-project installation. The `plur` MCP server provides tools named `plur_session_start`, `plur_learn`, `plur_recall_hybrid`, `plur_feedback`, `plur_session_end`, etc. If you cannot find these tools, run `plur doctor` to diagnose. Do **not** substitute tools from other MCP servers (e.g. `datacore_*`) — those belong to a different system.

A PreToolUse guard enforces that `plur_session_start` is called at the beginning of every session. All other tools are blocked until this is done. The flow is: ToolSearch to load `plur_session_start` → call it with a task description → proceed.

### Session Workflow

1. **Start**: Call `plur_session_start` with task description — enforced by guard hook
2. **Learn**: When corrected or discovering something new, call `plur_learn` immediately
3. **Recall**: Before answering factual questions, call `plur_recall_hybrid` — check memory first
4. **Feedback**: Rate injected engrams with `plur_feedback` (positive/negative) — trains relevance
5. **End**: Call `plur_session_end` with summary + engram_suggestions

Do not ask permission to use these tools — they are your memory system.

### Multi-project scoping

PLUR uses `domain` and `scope` fields on engrams to separate knowledge by project. When calling `plur_learn`, set `scope` (e.g. `project:my-app`) to namespace the engram. Scoped recall automatically includes global engrams.

### When to check memory

Before reaching for web search, file reads, or guessing — apply this priority:
1. Is the answer already in engrams? → `plur_recall_hybrid`
2. Is the answer in the local filesystem? → Read/Grep/Glob
3. Is the answer derivable from context already loaded? → Just answer
4. Only if 1-3 fail → Use external tools

### When corrected

When the user corrects you ("no, use X not Y", "that's wrong"):
1. Call `plur_learn` immediately — before continuing the task
2. Call `plur_feedback` with negative signal on the wrong engram if one was injected
3. Then continue with the corrected approach
