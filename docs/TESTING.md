# Testing Guide (ARM64/Raspberry Pi OS Bookworm)

This project prioritizes TDD and ARM64 compatibility. Below are the common workflows to run tests locally on a Raspberry Pi.

## 1) Python backend tests

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Run the full test suite:

```bash
pytest -q
```

- Contract tests validate the FastAPI REST + WebSocket API.
- Integration tests (`tests/integration`) include backups, SQLite schema migrations, the ACME/TLS
  renewal flow (against a fake `certbot` executable and `openssl`-generated certificates, never
  Let's Encrypt), map provider/cost control, and manual control during GPS loss.
- There is no placeholder gate for `tests/integration`: every test there runs by default. The only
  opt-in gates are the Pi-target ones: `RUN_PERF_TESTS=1` (latency/throughput budgets) and
  `RUN_HW_TESTS=1` (hardware self-test, see "On-device hardware testing" below).

### Order independence

CI runs the default order, which says nothing about a test that only passes after a particular
predecessor. `scripts/run_tests_shuffled.py` runs the same tests in a seeded random order:

```bash
env -u PYTEST_ADDOPTS SIM_MODE=1 python scripts/run_tests_shuffled.py --seed 11 -- \
  tests -o addopts="" -p no:cacheprovider --ignore=tests/soak --ignore=tests/agent_bridge -W ignore
```

`--mode module` (default) shuffles the test files and the tests in each, `item` shuffles every
test, `reverse` runs the default order backwards. The first line printed names the seed and a
digest of the order, so a failure is replayed by repeating the command. State a test would leave
behind for the next one (the `auth_service` failed-login counter, the E-stop drive-block window)
is reset before every test by `tests/conftest.py`; a new module-level singleton that keeps state
needs the same reset there, and `tests/unit/test_conftest_isolation.py` fails when the auth
facade grows state the reset does not know.

### A hung test fails

Every test is bounded by `timeout = 300` (pytest-timeout, signal method, `pyproject.toml`): a test
that stalls fails with the stacks of every thread dumped and the run goes on, instead of holding the
job until GitHub's six-hour limit with no verdict (audit TOOL3-01: the suite once sat 11 minutes in
an async fixture teardown). The opt-in 8 h soak is exempt (`@pytest.mark.timeout(0)`), the
`sim-tests` job has `timeout-minutes: 45`, and the `platform` fixture of
`tests/unit/test_ibt4_stand_down_integration.py` reports the teardown step it is stuck in. The plugin
is a project dependency: install with `uv sync --frozen --all-extras --dev`.

### Strict operator auth

`tests/conftest.py` defaults `LAWN_BERRY_SIM_OPEN_AUTH` to `1` (the simulation admits an anonymous
WebSocket client). `LAWN_BERRY_SIM_OPEN_AUTH=0 pytest ...` runs the strict path, where a peer needs a
login and a ticket. The eight tests in `NEEDS_SIM_OPEN_AUTH` (`tests/conftest.py`) open an anonymous
WebSocket, which only the opt-in admits, and are skipped with that reason when it is off; a test in
that list that no longer exists fails `test_conftest_isolation.py`. Everything else is the same run.

## 2) Frontend tests and lint (optional)

From `frontend/`:

```bash
npm ci
npm run lint
npm run test
npm run build
```

All frontend dependencies are compatible with ARM64.

## 3) Frontend end-to-end tests (Playwright)

From `frontend/`. The specs mock the backend (`tests/e2e/utils/mock-backend.ts`)
and Playwright serves the built frontend itself, so no API server is needed:

```bash
npx playwright install chromium   # first run only
npm run test:e2e                  # builds, then runs
```

`playwright.config.ts` serves `dist/` via `npm run preview`, so a build must
exist first — `npm run test:e2e` does both. Bare `npx playwright test` only
works if `frontend/dist/` is already current.

The `webui-build` workflow runs these on **pull requests only** — it never runs
on `main`. A spec that goes stale after a UI change therefore stays green on the
default branch and only turns red on the next PR, whoever opens it. If
`build-ui` fails on a PR that doesn't touch the frontend, check whether the
failure predates the branch (`git switch --detach origin/main` and re-run)
before assuming the PR caused it.

On aarch64 workstations the pinned Chromium may fail to install. If
`npx playwright test` reports a missing executable or `Old Headless mode has
been removed`, point it at a working headless shell for the run:

```bash
# Resolve the glob first -- a bare wildcard on an assignment RHS is NOT
# pathname-expanded by bash, so LB_CHROME_PATH would keep the literal '*'.
export LB_CHROME_PATH=$(ls -d ~/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell | head -1)
npm run build          # not `npm run test:e2e` -- that re-runs the install
npx playwright test    # step that hangs on this platform
```

(CI is unaffected — it uses the official Playwright container.)

## 4) Docs drift guard

CI will fail if code changes without corresponding documentation or journal updates. You can run the check locally:

```bash
bash scripts/check_docs_drift.sh
```

Update one of the following to satisfy the guard:
- `docs/**`
- `spec/**`
- `README.md`
- `.specify/memory/AGENT_JOURNAL.md`

### Secret scan

`scripts/check_no_secrets.py` fails when a credential-looking literal is committed (a sudo password
was committed twice; the rules also cover the shapes this product creates: `.env` assignments such as
`NTRIP_PASSWORD=...`, JSON/YAML `"password"` keys, tunnel and maps keys, `curl -u`, URL userinfo).

```bash
python3 scripts/check_no_secrets.py --tree          # every tracked file (every file when there is no .git)
python3 scripts/check_no_secrets.py --base origin/main   # every line the branch adds: commits, merges, net diff
```

CI runs `--base` on a pull request with the BASE branch's copy of the scanner and of
`spec/secret_scan_allow.txt`, so a pull request cannot loosen its own check (the pull request that
introduces the scanner is judged by its own copy and says so). Limit: the workflow file is the pull
request's own, so `.github/CODEOWNERS` names an owner for it, the scanner and the allow-list, and that
binds only while branch protection requires code-owner review (a repository setting, not in the
tree). A reviewed false positive is silenced with `<path>:<digest>  # reason` in
`spec/secret_scan_allow.txt`; the scan prints the digest, never the value. Not covered: a password that
is a plain lower-case word, a secret split across lines, and the skipped paths (`tests/`, fixtures,
generated artefacts).

### Code structure overview

`docs/code_structure_overview.md` has to follow the callable interfaces of the PYTHON files a change
touches (`backend/src`, `scripts`, `workshop`, `firmware`, `.specify/scripts`). CI runs the check on
pull requests; locally:

```bash
python3 scripts/sync_code_structure.py --check            # exit 1 on drift, 2 if it cannot compare
python3 scripts/sync_code_structure.py --fix              # regenerate the rows in place
python3 scripts/sync_code_structure.py --check --base origin/main --all   # a REPORT: every file in scope
```

`--fix` rewrites the interface cell of a stale row from the source (purpose and subsystem are kept),
adds a row for a new file (its purpose is the first two sentences of the module docstring, so a new
file needs one), merges duplicate rows and drops the row of a deleted file.

`--all` is not a gate and is not clean today: it lists every in-scope file that has public callables
and no row (about a hundred, mostly under `backend/src`) and every hand-written interface summary that
is not the generated cell. `--fix --all` would write them, but exits 2 and writes nothing while any of
those files has no module docstring, and then names all of them. What is NOT checked: TypeScript,
shell and OpenSCAD rows are hand-maintained; for them the check only sees that the file still exists
(a function added to `frontend/src/services/api.ts` is invisible to it).

## 4) On-device hardware testing (RPi)

Run the backend and call the hardware self-test endpoint:

```bash
uvicorn backend.src.main:app --host 0.0.0.0 --port 8001
# Then from another shell on the Pi:
curl http://localhost:8001/api/v2/system/selftest | jq
```

Enable the hardware self-test integration test:

```bash
RUN_HW_TESTS=1 pytest -q tests/integration/test_hardware_selftest.py
```

Notes:
- The self-test is safe on systems without hardware: it catches missing devices and returns a report.
- For I2C/serial access, ensure the user is in groups `i2c` and `dialout`.

## 5) Troubleshooting

- Ensure you’re on Raspberry Pi OS (64-bit) Bookworm.
- Avoid non-ARM64 dependencies. If needed, propose a Pi-compatible alternative first.
- If placeholder tests fail intentionally, run them only when implementing the corresponding features.
