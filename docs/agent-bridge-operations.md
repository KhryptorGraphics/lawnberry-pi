# Agent bridge: operations guide

How to install, configure, run, monitor and recover the approval-gated agent bridge
(`tools/agent_bridge/`). Design and safety model: [agent-bridge.md](agent-bridge.md). How to write an issue the
bridge accepts: [agent-bridge-task-template.md](agent-bridge-task-template.md). Every setting is in
[`config/agent-bridge.example.yaml`](../config/agent-bridge.example.yaml).

> **The bridge belongs on a developer workstation only.** Never install it on the mower Pi, the Orange Pi base
> station or any host that holds mower credentials or SSH keys. It is developer tooling, not a mower service:
> its systemd unit (below) is a *user* unit that lives in `~/.config/systemd/user/`, **not** in the repository's
> `systemd/` directory, which holds the units deployed to the mower.

Contents: [Prerequisites](#prerequisites) - [Install](#install) - [Configuration](#configuration-walkthrough) -
[CLI reference](#cli-reference) - [Dry run to real run](#from-dry-run-to-a-real-run) -
[Start and stop](#start-and-stop) - [State, logs, rotation](#state-logs-and-rotation) -
[Monitoring](#monitoring) - [Failures and recovery](#failures-and-recovery) - [Retry](#retry-procedure) -
[Permissions](#permissions) - [Sandboxing](#sandboxing) - [Cost](#cost) - [Troubleshooting](#troubleshooting)

## Prerequisites

| Need | Detail |
|------|--------|
| OS | Linux (developed on aarch64, kernel 6.8). Process supervision uses `/proc` and `prctl` |
| Python | 3.11 (`requires-python = ">=3.11,<3.13"`); the repository virtual environment already provides `httpx`, `pydantic` and `PyYAML`, so nothing extra is installed |
| git | with `git worktree` (tested with 2.43.0); the local clone must have a remote (`origin`) that you can push to |
| Agent CLI | `omp` (OhMyPi, the default; tested with 18.12.6, a Bun script, so `bun` must be on `PATH`) and/or `claude` (Claude Code, tested with 2.1.293). At least one, logged in as the user that runs the bridge |
| GitHub | a fine-grained personal access token for this one repository (scopes below), and an account that can apply labels |
| Network | outbound HTTPS to `api.github.com`, `github.com` and the model provider. No inbound port is opened |

## Install

Run everything from the repository root (`tools` is a developer package that `pip install -e .` does not
install).

```bash
cd ~/lawnberry-pi
python3.11 -m venv .venv && .venv/bin/python -m pip install -e .   # skip if the venv exists

# 1. Which agent CLIs are installed?
.venv/bin/python -m tools.agent_bridge --config config/agent-bridge.example.yaml backends
```

Example output (this workstation):

```json
[
  {"backend": "omp", "binary": "/home/kp/.nvm/versions/node/v24.16.0/bin/omp", "default": true,
   "version": "omp/18.12.6"},
  {"backend": "claude", "binary": "/home/kp/.local/bin/claude", "default": false,
   "version": "2.1.293 (Claude Code)"}
]
```

(The real output is pretty-printed JSON with the same fields.)

```bash
# 2. Log in to the agent CLI(s) as the bridge user (interactive, once)
omp login                # OhMyPi: opens the provider login
claude auth login        # Claude Code (optional alternative)

# 3. Create the GitHub token (see Permissions) and store it OUTSIDE the repository
install -d -m 700 ~/.config/lawnberry-agent-bridge
install -m 600 /dev/null ~/.config/lawnberry-agent-bridge/token
$EDITOR ~/.config/lawnberry-agent-bridge/token          # paste the token, nothing else

# 4. Configuration (kept outside the repository; see Configuration walkthrough)
cp config/agent-bridge.example.yaml ~/.config/lawnberry-agent-bridge/config.yaml
$EDITOR ~/.config/lawnberry-agent-bridge/config.yaml
export LAWNBERRY_AGENT_BRIDGE_CONFIG=~/.config/lawnberry-agent-bridge/config.yaml
export LAWNBERRY_BRIDGE_GITHUB_TOKEN="$(cat ~/.config/lawnberry-agent-bridge/token)"

# 5. Check, create the labels, smoke-test the real agent
.venv/bin/python -m tools.agent_bridge validate-config
.venv/bin/python -m tools.agent_bridge setup-labels --yes
.venv/bin/python -m tools.agent_bridge smoke --backend omp --probe-guard
```

`setup-labels --yes` creates `agent-task`, `agent-approved`, `agent-running`, `agent-blocked` and
`agent-done` with colours and descriptions; labels that already exist are left alone. It is the one
GitHub-writing command that works while `dry_run: true`, which is why it needs `--yes`.

`smoke` runs the real agent once in a throwaway repository (no GitHub, no token, no configuration file) and
checks the result with the same policy code a real run uses. Expect a few cents. Trimmed result of
2026-10-24 (omp 18.12.6, `--model anthropic/claude-sonnet-5-5`):

```json
{"backend": "omp", "version": "omp/18.12.6", "ok": true, "note_content": "agent bridge smoke ok\n",
 "result": {"status": "succeeded", "duration_s": 5.3, "cost_usd": 0.008283, "tool_calls": 1},
 "guard_probe": {"blocked_tool_calls": 2, "outside_file_created": false, "inside_file_created": true,
                 "blocked_samples": ["write: path escapes the task worktree: /tmp/agent-bridge-outside-.../outside.txt",
                                     "read: path escapes the task worktree: /etc/hostname"]}}
```

`ok: false` with `result.failure_kind` tells you what is wrong (for `claude`: `auth` or `credit`, see
Troubleshooting).

## Configuration walkthrough

The file is YAML, strictly validated (unknown keys are errors). Lookup order: `--config FILE`, then
`$LAWNBERRY_AGENT_BRIDGE_CONFIG`, then `config/agent-bridge.yaml` under the current directory. Relative
`repo_path`, `state_dir` and `github.token_file` are resolved against the directory that holds the file.
`config/agent-bridge.yaml` is git-ignored and `config/agent-bridge*.yaml` is a protected path, so an agent run
can never edit it; keeping the live file outside the clone, as above, is simpler still. **No credential
belongs in it.**

Minimal useful file:

```yaml
repository: KhryptorGraphics/lawnberry-pi
repo_path: /home/you/lawnberry-pi            # clone the worktrees are made from (default: current dir)
state_dir: ~/.local/state/lawnberry-agent-bridge
dry_run: true                                # keep true until the dry run below has been reviewed
approvers: [KhryptorGraphics]                # who may apply agent-approved (required, never empty)
task_authors: [KhryptorGraphics]             # who may author tasks (default: the approvers)
agent:
  default: omp
  allowed: [omp, claude]
  omp: {binary: omp, model: anthropic/claude-sonnet-5-5}
limits: {budget_usd: 1.0, max_budget_usd: 3.0, timeout_minutes: 10, max_timeout_minutes: 30}
scopes: {allowed: [docs-only]}
checks: {}
```

| Section | Keys (defaults) and notes |
|---------|---------------------------|
| top level | `repository` (`owner/name`), `repo_path`, `state_dir` (`~/.local/state/lawnberry-agent-bridge`), `dry_run` (`true`), `approvers` (required), `task_authors` (= approvers), `dedupe_by_content` (`true`), `extra_secret_env` (env var names whose *values* are scrubbed from logs and published text) |
| `labels` | `task agent-task`, `approved agent-approved`, `running agent-running`, `blocked agent-blocked`, `done agent-done`; must be distinct |
| `github` | `api_url` (`https://api.github.com`, https only), `token_env` (`LAWNBERRY_BRIDGE_GITHUB_TOKEN`), `token_file` (optional, must be mode 0600), `request_timeout_seconds` 30, `max_retries` 3, `retry_backoff_seconds` 1, `rate_limit_max_wait_seconds` 60 |
| `git` | `remote origin`, `base_branch main`, `branch_prefix agent/`, `author_name`/`author_email` for commits, `push_auth` (`ambient` = whatever credentials git already has; `token` = push over https with the bridge token via `GIT_ASKPASS`, never on a command line), `fetch_before_run true`, `keep_worktrees` (`never`, `on-failure` default, `always`), `git_timeout_seconds` 120 |
| `pull_request` | `closing_keyword` (`Refs` default, or `Closes`/`Fixes`), `title_prefix "[agent] "`, `labels []` (added to the PR). Draft is fixed and cannot be turned off |
| `poll` | `interval_seconds` 60, `max_jobs_per_poll` 1, `error_backoff_seconds` 120 |
| `agent` | `default omp`, `allowed [omp, claude]`, `omp`/`claude` each with `binary`, `model`, `thinking` (`off minimal low medium high xhigh max auto`), `extra_args`, `pass_env` (claude defaults to `ANTHROPIC_API_KEY`); `command_prefix` (sandbox hook, below), `kill_grace_seconds` 10, `cancel_poll_seconds` 15 |
| `limits` | `budget_usd` 2.0 / `max_budget_usd` 5.0, `timeout_minutes` 20 / `max_timeout_minutes` 60, `max_attempts_per_issue` 3, `auto_retry_interrupted` 1, `max_start_retries` 3, `max_publish_attempts` 5, `publish_retry_delays [2, 10, 30]`, `lease_ttl_seconds` 180, `max_log_bytes` 4,000,000, `max_final_text_chars` 6000, `max_changed_files` 40, `max_diff_bytes` 400,000, `max_file_bytes` 1,000,000 |
| `scopes` | `allowed [docs-only]` (add `code` deliberately), `docs_paths [docs/**, "*.md"]`, `code_paths [docs/**, "*.md", tools/**, tests/**]` |
| `protected_paths` | leave it out. The built-in list (43 globs) keeps agents out of the safety chain, drivers, hardware and limit configuration, secrets, `systemd/`, `scripts/`, `.github/`, agent policy files, `tools/agent_bridge/` and dependency manifests. **Setting it replaces the list**: copy the defaults first if you extend it |
| `checks`, `default_checks` | named verification commands the bridge itself runs in the worktree: `argv` (list), `timeout_seconds` (600), `env` (literal values). A task can request a check by name and can never supply a command. Use absolute paths for tools outside the worktree: a worktree has no `.venv` of its own |
| `prompt` | `context_files [CLAUDE.md, .github/copilot-instructions.md]` (read from the base branch into every prompt), `max_context_chars` 20,000 |

Example checks (run with `SIM_MODE=1`, as the repository requires):

```yaml
checks:
  ruff:
    argv: [uvx, ruff@0.15.19, check, .]
    timeout_seconds: 300
  unit-tests:
    argv: [/home/you/lawnberry-pi/.venv/bin/python, -m, pytest, tests/unit, -q]
    env: {SIM_MODE: "1"}
    timeout_seconds: 1500
default_checks: []
```

A check must leave the worktree exactly as it found it; one that creates files (caches, reports) ends the run
`checks_modified_tree`.

## CLI reference

```text
python -m tools.agent_bridge [--config FILE] [-v] [--version] <command> [options]
```

Global options: `--config FILE` (see lookup order above), `-v/--verbose` (debug logging), `--version`
(`agent-bridge 1.0.0`). Exit codes: **0** success, **1** failure (blocked run, invalid task, nothing to
cancel, API errors), **2** usage or not authorised (missing `--yes`, no terminal), **3** configuration error
or `--execute` while `dry_run: true`, **130** interrupted.

| Command | Needs token | Writes to GitHub / ledger |
|---------|-------------|---------------------------|
| `backends`, `validate-config`, `validate-task --file`, `status`, `cancel`, `smoke` | no | none / `cancel` flags ledger rows |
| `validate-task --issue`, `preview` | yes (read) | none |
| `setup-labels --yes` | yes | creates labels |
| `run-once`, `serve`, `recover` | yes | dry run: none. `--execute`: yes |

**`backends`** - prints the installed agent CLIs (JSON: `backend`, `default`, `binary`, `version`).

**`validate-config`** - loads and validates the configuration:

```text
$ python -m tools.agent_bridge --config config/agent-bridge.example.yaml validate-config
configuration for KhryptorGraphics/lawnberry-pi is valid
  dry_run: True
  default agent: omp (allowed: omp, claude)
  approvers: KhryptorGraphics; task authors: KhryptorGraphics
  scopes: docs-only; state: /home/kp/.local/state/lawnberry-agent-bridge
```

**`validate-task (--file FILE | --issue N) [--title TITLE]`** - parses an issue body against the task schema and
then against this configuration. `--file` is offline (the title defaults to the file name without extension);
`--issue` reads the issue from GitHub. Exit 0 when valid, 1 otherwise. Run it before asking for approval:

```text
$ python -m tools.agent_bridge validate-task --file task.md --title "Document directory overrides"
task is valid
  revision 370f428565c2bca6  scope docs-only  agent omp
```

Invalid and not-permitted outputs are shown in the task template.

**`setup-labels [--yes]`** - creates the five labels. Without `--yes` it only explains and exits 2.

**`preview [--issue N] [--show-prompt] [--json]`** - the dry run. Lists open issues carrying `agent-task` and
`agent-approved`, applies exactly the checks a real run applies and prints, for each eligible issue, what would
happen: approver and approval event, backend, scope, budget, timeout, branch, the paths it may and may never
write, checks, the full agent command line and the prompt size. `--show-prompt` prints the exact prompt that
would go to the agent's stdin; `--json` prints a machine-readable report. Refused issues are listed with their
code. It reads GitHub and writes nothing anywhere.

**`run-once [--execute] [--yes] [--issue N] [--show-prompt]`** - processes approved issues once (at most
`poll.max_jobs_per_poll` runs). Without `--execute` it behaves like `preview`. With `--execute` it needs
`dry_run: false`, and either `--yes` or an interactive answer `y` to
`<what> - authorise real execution? [y/N]`; with no terminal and no `--yes` it refuses (exit 2). Exit 1 when a
run ended `blocked`.

**`serve [--execute] [--yes]`** - polls forever every `poll.interval_seconds`. `--execute` always requires
`--yes` (it is unattended) and `dry_run: false`. SIGINT/SIGTERM stop it cleanly (see Start and stop). Without
`--execute` it only polls and logs problems; use `preview` to see what would run.

**`status [--limit N] [--json]`** - the last N runs (default 20) from the ledger:

```text
339432bd41ad  #44    blocked    finished      omp    PR -     cost $0.0731   policy_violation
68110d422b87  #43    blocked    finished      claude PR -     cost $0.0000   agent_auth
9c55fb654886  #42    done       finished      omp    PR 57    cost $0.0412
```

Columns: run id (12 characters), issue, status, phase, backend, PR number, cost, failure kind.
`--json` gives `run_id`, `issue`, `status`, `phase`, `backend`, `pr`, `failure`, `started`, `finished`.

**`recover [--execute] [--yes]`** - resolves runs a stopped worker left behind (leases past
`lease_ttl_seconds`): an interrupted agent run is requeued, an interrupted publication is resumed, finished
runs whose GitHub update failed are re-reported. Without `--execute` it prints `would recover ...`. The
worker also does this at the start of every poll, so you normally never run it by hand.

**`cancel (--run-id ID | --issue N)`** - flags pending, running or publishing runs for cancellation (exit 1 if
none matched). `--run-id` takes a full run id or any unique prefix, such as the 12 characters that `status` and
the bot comments show. A worker stops the agent or check within `cancel_poll_seconds` and ends the run
`cancelled`. It is also honoured right after the lease (a pending run never starts), at the start of
verification and before each publish attempt, so nothing is pushed after it. The other ways to cancel are
closing the issue or removing `agent-approved`.

**`smoke [--backend omp|claude] [--model M] [--binary PATH] [--timeout S] [--budget USD] [--probe-guard]
[--keep] [--pass-env NAME]`** - real-agent smoke test (defaults: `omp`, 240 s, $1.00). `--probe-guard` also
asks the agent to write outside the worktree and read `/etc/hostname`, and reports whether the guard blocked
them (omp only: the claude backend has no bridge guard). `--keep` keeps the temporary workspace.
`--pass-env` (repeatable) forwards an environment variable; claude defaults to `ANTHROPIC_API_KEY`. Exit 0 only
when the agent succeeded, wrote the exact file and the diff passed the policy.

## From dry run to a real run

1. **Prerequisites:** `validate-config`, `backends`, `setup-labels --yes` and a passing
   `smoke --backend omp --probe-guard` (see Install).
2. **Write a harmless task** from the template: documentation only, one file, `budget_usd: 1.0`,
   `timeout_minutes: 10`. Check it offline: `validate-task --file task.md` (add `--title`), or after publishing
   `validate-task --issue N`.
3. **Approve it as the owner** (an account listed in `approvers`): read the whole issue, then apply
   `agent-approved`. An edit after this moment invalidates the approval.
4. **Dry run:** `python -m tools.agent_bridge preview --issue N --show-prompt`. Confirm the approver, backend,
   scope, write allow-sets, deny list, command line and prompt. Nothing was written anywhere.
5. **Flip the switch deliberately:** set `dry_run: false` in the configuration.
6. **Execute one issue:** `python -m tools.agent_bridge run-once --execute --issue N` (answer `y`) or add
   `--yes`. Watch the issue: a "started" comment appears, then "Draft pull request ready for review" with a
   draft PR, or a "blocked" comment with the reason.
7. **Review the draft PR by hand** and merge or close it yourself. The bridge never merges.
8. **Unattended:** `serve --execute --yes`, first in a terminal, then as a user service. To go back to dry run
   set `dry_run: true`: `--execute` is then refused with exit 3.

## Start and stop

Foreground: `python -m tools.agent_bridge serve --execute --yes`. Stop with Ctrl-C or `kill -TERM <pid>`. The
signal is checked when the abort check next runs (at most `cancel_poll_seconds`, 15 s, later); a running
agent is then terminated and its run is **requeued** (`pending`, not cancelled, not failed) and resumed by the
next worker that starts, up to `auto_retry_interrupted` times (`max_start_retries` before the agent started).
A requeue is not shown to the issue except that `agent-running` disappears. `kill -9` is recovered too: the
stale lease expires after `lease_ttl_seconds` (180 s) and the next poll requeues or resumes the run.

### Optional: user-level systemd unit (developer workstation only)

This is an example, not a shipped file. It must **not** be added to the repository's `systemd/` directory, and
it must never run on the mower or the base station. Create `~/.config/systemd/user/lawnberry-agent-bridge.service`:

```ini
[Unit]
Description=LawnBerry agent bridge (developer workstation only)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/lawnberry-pi
Environment=LAWNBERRY_AGENT_BRIDGE_CONFIG=%h/.config/lawnberry-agent-bridge/config.yaml
# omp is a Bun script: put bun and the agent CLIs on PATH explicitly (adjust the nvm version)
Environment=PATH=%h/.bun/bin:%h/.nvm/versions/node/v24.16.0/bin:%h/.local/bin:/usr/local/bin:/usr/bin:/bin
# one line, LAWNBERRY_BRIDGE_GITHUB_TOKEN=github_pat_...  (chmod 600)
EnvironmentFile=%h/.config/lawnberry-agent-bridge/token.env
ExecStart=%h/lawnberry-pi/.venv/bin/python -m tools.agent_bridge serve --execute --yes
Restart=on-failure
RestartSec=30
# SIGTERM only the bridge; it stops and requeues its own agent. The default (control-group) would also
# SIGTERM the agent, which the run would then report as an agent failure instead of an interruption.
KillMode=mixed
TimeoutStopSec=90
NoNewPrivileges=yes

[Install]
WantedBy=default.target
```

```bash
loginctl enable-linger "$USER"                       # keep it running without a login session
systemctl --user daemon-reload
systemctl --user enable --now lawnberry-agent-bridge
journalctl --user -u lawnberry-agent-bridge -f
systemctl --user stop lawnberry-agent-bridge         # clean stop, run is requeued
```

Sandboxing directives such as `PrivateTmp` or `ProtectHome` need user namespaces and fail on hosts that deny
them (this workstation does); test any you add with `systemctl --user status`.

## State, logs and rotation

Everything lives under `state_dir` (mode 0700; files 0600):

```text
ledger.sqlite (+ -wal, -shm)                        the ledger: runs, events, notices
logs/bridge.log[.1-.5]                              bridge log, secrets redacted
runs/<run_id>/prompt.txt                            exact prompt sent to the agent
runs/<run_id>/worktree.pointer                      the worktree's .git pointer (tamper check)
runs/<run_id>/agent/<backend>.stdout.log / .stderr.log
runs/<run_id>/agent/omp-overlay.yml, omp-guard-policy.json, omp-guard-audit.jsonl   (omp)
runs/<run_id>/checks/check-<name>.stdout.log / .stderr.log
worktrees/<run_id>/                                 the task worktree (kept per git.keep_worktrees)
git-askpass.sh                                      only with git.push_auth: token
```

- **`bridge.log`** rotates by itself: `RotatingFileHandler`, 5 MB per file, 5 backups (about 30 MB in total).
  The same lines go to stderr, so a user service also has them in the journal.
- **Per-run logs** are not rotated. Each stream is capped (agent: `limits.max_log_bytes`, 4 MB, at least
  64 MB for claude whose JSON result can be large; checks: 2 MB); beyond the cap a marker line
  `[agent-bridge: log truncated, capture limit reached]` is written and logging stops. Delete old run
  directories of finished runs now and then, for example runs older than 30 days:
  `find ~/.local/state/lawnberry-agent-bridge/runs -mindepth 1 -maxdepth 1 -mtime +30 -exec rm -rf {} +`
  (`status` first: never remove the directory of a `running` or `publishing` run).
- **Worktrees** of failed or blocked runs are kept (`keep_worktrees: on-failure`) for inspection; a `done` run's
  worktree is removed. Remove a kept one with
  `git -C <repo_path> worktree remove --force <state_dir>/worktrees/<run_id>`, then `git worktree prune` and
  `git branch -D agent/issue-N-xxxxxxxx`.
- The ledger is tiny and never pruned. Back it up with the rest of `state_dir` if you care about history; losing
  it is safe but makes `agent-running` labels without a ledger row show up as `running_label` refusals.

## Monitoring

- `status` (table) or `status --json` for the latest runs; `journalctl --user -u lawnberry-agent-bridge`.
- On GitHub: the "started", "ready for review" or "blocked" comment (agent, model, run id, cost, files, checks,
  tool calls and policy-blocked calls), the `agent-*` labels, and the PR body table.
- The ledger, read-only (safe while the worker runs):

```bash
S=~/.local/state/lawnberry-agent-bridge/ledger.sqlite
sqlite3 -readonly "$S" "select issue_number, status, phase, backend, failure_kind, pr_number, finished_at
                        from runs order by id desc limit 10;"
sqlite3 -readonly "$S" "select at, kind, detail from events where issue_number = 42 order by id;"
sqlite3 -readonly "$S" "select run_id, status, phase, lease_expires_at from runs
                        where status in ('running','publishing');"          # who is working right now
```

Event kinds: `lease.granted`, `lease.resumed`, `lease.denied`, `phase.<phase>`, `finish.<status>`,
`run.requeued`, `refused.<code>` (once per distinct refusal), `notify.failed`.

## Failures and recovery

Everything below is reported to the issue (blocked and cancelled runs get a comment with the reason and hints)
and stored in the ledger (`runs.failure_kind`, `failure_detail`, `result_json`). A blocked run never pushed a
branch or opened a PR, **except** `publish_failed`, `push_failed` and `github_error`, which happen during
publication (the commit stays in the kept worktree).

### Run endings (`failure_kind`)

| `failure_kind` | Status | Cause | Operator action |
|----------------|--------|-------|-----------------|
| `agent_auth` | blocked | the agent CLI is not logged in (omp: not authenticated; claude: OAuth expired or invalid key) | `omp login` / `claude auth login` as the bridge user, then re-approve |
| `agent_credit` | blocked | the model account is out of credit ("Credit balance is too low") | top up, re-approve |
| `agent_rate_limit` | blocked | provider rate limit or overload (429) | wait, re-approve |
| `agent_api_error` | blocked | any other provider or CLI error | read `runs/<id>/agent/*.log`, re-approve |
| `agent_timeout` | blocked | the agent exceeded `timeout_minutes` (or omp hit its `--max-time`) | narrow the task or raise `timeout_minutes` (capped by `max_timeout_minutes`) |
| `agent_budget` | blocked | the spend cap was reached | narrow the task or raise `budget_usd` (capped by `max_budget_usd`) |
| `agent_max_turns` | blocked | the agent ran out of turns (claude) | split the task |
| `agent_nonzero_exit` | blocked | the agent CLI exited abnormally | read the agent logs |
| `agent_no_result` | blocked | no terminal result (crash, killed, truncated output) | read the agent logs |
| `agent_hook` | blocked | the bridge's own stream parser failed and ended the run | read `bridge.log`; report it |
| `agent_unavailable` | blocked | the agent binary is missing, not executable or could not be started | install it, fix `agent.<name>.binary`, or pick the other backend with `agent:` |
| `agent_blocked` | blocked | the agent reported `blocked`: it judged it could not do the task safely and changed nothing | read its summary in the comment; rewrite the task or do it by hand |
| `no_changes` | blocked | the agent finished without changing any file | rephrase the objective |
| `policy_violation` | blocked | the staged diff broke the change policy; the comment lists each violation: `protected_path`, `outside_scope`, `too_many_files`, `diff_too_big`, `too_large`, `path_invalid`, `symlink`, `submodule`, `executable`, `type_change`, `binary`, `secret` | a human makes protected or out-of-scope changes; narrow the task; for `secret`, rotate the credential if it was real |
| `checks_failed` | blocked | a named check failed (the first failure stops the sequence) | read `runs/<id>/checks/check-<name>.*.log` and the comment tail; fix the task or the check |
| `checks_modified_tree` | blocked | the checks created or changed files in the worktree | make the check side-effect free |
| `worktree_tampered` | blocked | the worktree's `.git` pointer changed or `HEAD` moved during the agent run | treat the agent as faulty: inspect `worktrees/<run_id>` before re-approving anything |
| `commit_failed` | blocked | `git commit` failed | read `bridge.log` (disk space, git setup) |
| `task_changed` | blocked | the issue title or body changed after approval (found at start, during the agent or check poll, or before the push) | review the edit, then remove and re-apply `agent-approved` |
| `github_permission` | blocked | the token is rejected: at start it cannot update labels/comments, at publish it needs Contents, Pull requests and Issues read/write | fix the token scopes, re-approve |
| `git_failed` | blocked | fetch or worktree creation failed for a non-network reason | check `repo_path`, the remote and `git.base_branch` |
| `push_failed` | blocked | `git push` was refused for a non-network reason | check the remote, branch protection and `git.push_auth` credentials |
| `github_error` | blocked | GitHub rejected an API call during publication | read `failure_detail` |
| `publish_failed` | blocked | publication did not complete in `max_publish_attempts` (5) | restore connectivity, re-approve (the commit is kept locally) |
| `interrupted` | blocked | interrupted more often than the automatic retries allow | find out why the worker keeps stopping, re-approve |
| `internal_error` | blocked | unexpected exception in the worker | read the traceback in `bridge.log`; report it |
| `cancelled` | cancelled | `agent-bridge cancel`, or the issue was closed or un-approved while the agent or a check ran | nothing, or re-approve |
| `issue_closed`, `approval_withdrawn`, `issue_missing` | cancelled | issue closed / `agent-approved` removed / issue deleted, seen at a gate | nothing, or reopen and re-approve |
| `approval_changed` | cancelled | before publication the approval event was withdrawn or replaced | nothing; the newer approval starts its own run |

Network problems do not end a run by themselves: GitHub or the git remote unreachable at the start requeues
the run (`max_start_retries`); an unreachable GitHub during the abort poll is ignored; publication retries after
2, 10 and 30 seconds and is then deferred, the run staying `publishing` until a later pass resumes it
(`max_publish_attempts`). Duplicates are prevented by the ledger, never by comments.

### Refusals (no run is created)

Each distinct refusal is logged once as an event `refused.<code>`. Those marked * also post one explanatory
comment (never repeated) because the owner needs to act.

| Code | Meaning | Action |
|------|---------|--------|
| `not_open` | closed issue or a pull request | none |
| `missing_labels` | lacks `agent-task` or `agent-approved` | apply the label |
| `already_handled` | carries `agent-blocked` or `agent-done` | remove it, then re-approve (Retry) |
| `author_not_trusted` | issue author not in `task_authors` | silent by design; recreate the issue as an allowed author |
| `no_approval_event` | cannot find who applied the label | remove and re-apply `agent-approved` |
| `approver_not_trusted` | the label was applied by someone outside `approvers` | silent by design; the owner re-applies it |
| `already_processed` | this approval event already has a terminal or active run | re-approve with a new label event |
| `running_label` | `agent-running` without a ledger row for this approval | remove `agent-running` by hand |
| `approval_stale`* | edited at or after approval | re-read, remove and re-apply `agent-approved` |
| `invalid_task`* | no valid `agent-task` block (messages listed) | fix the issue, re-approve |
| `task_not_permitted`* | valid but not allowed here (scope, agent, check, budget, timeout) | change the task or the configuration, re-approve |
| `attempt_limit`* | the issue used `max_attempts_per_issue` (3) runs | open a new issue |
| `duplicate_task`* | identical task content already ran or is running in another issue | use that issue |

Lease denials seen only in the log and `lease.denied` events: `busy` (another run is active; it resolves
itself), `duplicate-active`, `recovery-pending` (a stale lease the next poll will recover),
`already-processed`, `attempt-limit`, `duplicate-content`.

## Retry procedure

A blocked or cancelled run is never retried automatically; the owner decides.

1. Read the bot comment (reason, hints) and, if needed, `runs/<run_id>/agent/*.log` and `checks/`.
2. Fix the cause: the task text, the configuration, the login, the token, the check.
3. Remove the **`agent-blocked`** label (and **`agent-done`** after a finished run you want repeated; close the
   old draft PR first).
4. **Remove and re-apply `agent-approved`.** The new label event is a new approval, which allows a new run
   (up to `max_attempts_per_issue`). Removing only `agent-blocked` does nothing: the old approval already has
   its run (`already_processed`).
5. If you edited the issue, re-applying the label is mandatory anyway: an edit at or after the approval makes it
   stale. Wait a few seconds between the edit and the approval; equal timestamps count as stale.
6. `preview --issue N` shows whether it is now eligible.

## Permissions

**GitHub token** (fine-grained, restricted to this one repository, with a short expiry you renew on purpose):

| Repository permission | Access | Used for |
|-----------------------|--------|----------|
| Metadata | read | mandatory |
| Issues | read and write | list issues, issue events, labels, comments |
| Pull requests | read and write | find and open the draft PR |
| Contents | read and write | push the `agent/...` branch |

Nothing else: no Actions, Workflows, Administration, Secrets or Deployments. Preferably create the token under a
dedicated machine account that is **not** in `approvers`, so a bot comment can never be mistaken for an owner
action and the bridge cannot approve its own work. Protect `main` with required pull-request reviews: the bridge
only ever pushes its own `agent/issue-N-xxxxxxxx` branch, never forced, but the branch rule is the backstop.

**Operating system:** run as an unprivileged user with no access to mower credentials, SSH keys for the Pi or
Orange Pi, or other projects. `state_dir` is created 0700, the ledger and logs 0600, and a `github.token_file`
must be 0600 (group or other access is an error). With `git.push_auth: ambient` the push uses whatever
credential helper or ssh-agent the user has, so that credential is as powerful as the user; `token` limits it
to the fine-grained token. The agent never sees `SSH_AUTH_SOCK`, `GIT_ASKPASS`, `GITHUB_TOKEN` or anything
named `LAWNBERRY_BRIDGE_*`.

## Sandboxing

The bridge confines agents with its own layers (tool allow-list, omp guard, post-run diff policy) but does not
sandbox the agent process. `agent.command_prefix` is the hook: its words are placed in front of the agent
command **and of every check command** (not of `git`). A wrapper must keep the working directory (the per-run
worktree), stdin (the prompt), the environment and the exit status, and pass signals on.

What was verified on the development workstation (2026-10-24):

- **bubblewrap is unusable here.** `/usr/bin/bwrap` exists, but `bwrap --ro-bind / / --dev /dev true` fails with
  `setting up uid map: Permission denied`: unprivileged user namespaces are denied
  (`kernel.apparmor_restrict_unprivileged_userns = 1`, `unshare --user` fails). A bwrap prefix is untested. On a
  host that allows it, the wrapper must bind the worktree directory read-write, the repository's
  `tools/agent_bridge` and the run directory read-only, the CLI's runtime and login directory, and allow the
  network.
- **`systemd-run --user --scope` works as a resource limiter** (stdin, cwd, timeout kill all verified):

  ```yaml
  agent:
    command_prefix: [systemd-run, --user, --scope, --quiet, -p, MemoryMax=4G, -p, TasksMax=512, --]
    omp: {binary: omp, pass_env: [XDG_RUNTIME_DIR]}
  checks:
    unit-tests:
      argv: [/home/you/lawnberry-pi/.venv/bin/python, -m, pytest, tests/unit, -q]
      env: {SIM_MODE: "1", XDG_RUNTIME_DIR: /run/user/1000}
  ```

  The bridge scrubs the environment, and without `XDG_RUNTIME_DIR` the command fails with `Failed to connect to
  bus: No medium found`; hence `pass_env` for each backend you use and `env` for each check. This limits memory
  and process counts only; it does **not** confine files or the network.
- The practical baseline is a dedicated unprivileged OS user for the bridge, as described under Permissions.

## Cost

- Defaults: `limits.budget_usd` $2.00 per run, hard cap `max_budget_usd` $5.00; a task may lower or raise its own
  `budget_usd` up to the cap. There is no daily or monthly cap in the bridge: set one in the provider console.
- omp has no spend-cap flag. The bridge adds up the cost of every streamed assistant message and ends the run
  before the next model call once the budget is reached, so a single expensive call can overshoot. Claude Code
  enforces `--max-budget-usd` itself.
- Observed: a trivial real `omp` task cost $0.024 on a cold prompt cache (the first call pays to cache the
  system prompt and tool definitions) and about $0.008 afterwards (Sonnet via omp). Real tasks cost more in
  proportion to files read and turns; the "ready for review" comment and the PR body show the cost, `status`
  shows it per run.
- `thinking: high`/`max` and a larger model raise the cost per turn. `omp models` lists the models omp can use.

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| `configuration error: ...` (exit 3) | the message names the key; unknown keys, a missing `approvers`, a `default` agent not in `allowed`, or `budget_usd` above `max_budget_usd` all fail loudly |
| `no GitHub token: set $LAWNBERRY_BRIDGE_GITHUB_TOKEN ...` | export the token or set `github.token_file` (0600). `validate-task --file`, `status`, `cancel` and `smoke` do not need it |
| `refusing to execute: the configuration has dry_run: true` | intended; set `dry_run: false` after the dry run |
| `refusing to execute without a human present` / `serve --execute ... pass --yes` | add `--yes` |
| issue is ignored | `preview --issue N` prints the refusal code; walk the Refusals table. Typical: issue edited after approval, approved by an account outside `approvers`, an HTML comment (`<!-- -->`) from an issue template left in the body (`invalid_task`), or `agent-blocked` still on the issue |
| `claude` smoke returns `failure_kind: auth` | "Failed to authenticate: OAuth session expired and could not be refreshed": run `claude auth login` as the bridge user (check `claude auth status`). This was the state of the development workstation on 2026-10-24 |
| `claude` returns `failure_kind: credit` | "Credit balance is too low": an `ANTHROPIC_API_KEY` is being passed through (the default `pass_env`) and that account has no credit; top it up or remove the variable to use the OAuth login |
| `omp` smoke returns `auth` | `omp login` as the bridge user; `omp usage` shows the provider usage limits of every authenticated account |
| `omp: command not found`, or `bun` missing, under systemd | the service `PATH` lacks the Bun and omp directories; set `Environment=PATH=...` as in the unit above |
| `agent binary 'omp' was not found on PATH` in `preview` | the dry run uses your shell `PATH`; fix it or set `agent.omp.binary` to an absolute path |
| model not found | set `agent.omp.model` to a name `omp models` lists, for example `anthropic/claude-sonnet-5-5` |
| run stuck `running` after a crash | wait `lease_ttl_seconds` (180 s); the next poll recovers it, or run `recover --execute --yes` |
| a worktree and branch are left behind | by design after a failure; remove them as described under State |
| `GitHub rate limit` problems | the client waits up to `rate_limit_max_wait_seconds`, then reports unavailable and the poll backs off `error_backoff_seconds` |
| ledger cannot be opened by another tool | use `sqlite3 -readonly`; the worker holds the only writer |
