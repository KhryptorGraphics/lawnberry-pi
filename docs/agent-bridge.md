# Agent bridge: GitHub-issue-driven local coding agent (OhMyPi primary, Claude Code alternative)

**Status:** Implemented in `tools/agent_bridge/` (v1.0.0), dry run by default. Written as a specification
and coding-agent assignment on 2026-10-24; this revision records what was built and what was not.
**Date:** 2026-10-24. **Priority:** build this workflow before expanding hardware implementation.

> **Supersedes the Claude-only wording of PR #16** ("Add Claude Code development bridge implementation
> spec", commit `8223f7d`, `docs/claude-code-bridge.md`). OhMyPi (`omp`) is the primary agent everywhere;
> Claude Code (`claude`) is a selectable alternative behind the same backend interface. The original
> wording stays in git history.

Related documents: [operations](agent-bridge-operations.md) (install, run, recover),
[task template](agent-bridge-task-template.md) (how to write an issue the bridge accepts),
[`config/agent-bridge.example.yaml`](../config/agent-bridge.example.yaml) (every setting),
[`docs/code_structure_overview.md`](code_structure_overview.md) (module and function index).

Contents: [Goal](#goal) - [How it works](#how-it-works) - [MVP architecture](#mvp-architecture) -
[Pluggable agent backends](#pluggable-agent-backends) - [Repository layout](#repository-layout) -
[Build order](#build-order-for-coding-agent) - [Acceptance criteria](#acceptance-criteria) -
[Implementation status](#implementation-status) - [Agent backends](#agent-backends) -
[Safety model](#safety-model) - [Known limitations](#known-limitations) -
[Deviations from the original spec](#deviations-from-the-original-spec) -
[Initial agent instruction](#initial-agent-instruction)

## Goal

Allow the owner to discuss LawnBerry Pi with ChatGPT, have ChatGPT publish structured tasks to GitHub,
and let a locally running coding-agent worker (OhMyPi `omp` by default, Claude Code `claude` as an
alternative) pick up **approved** tasks, execute the development work, run checks, create a branch and
pull request, and publish a summary for ChatGPT to inspect on the next conversation turn. This is a
GitHub-mediated workflow, **not** a direct live connection to the ChatGPT voice session. The user may
need to explicitly request task publication/review; ChatGPT does not automatically observe speech or
run continuously.

## How it works

```text
 owner / ChatGPT                GitHub                          developer workstation
 ---------------                ------                          ---------------------
 1. issue with a fenced   --->  issue + label agent-task
    "yaml agent-task" block
 2. read the text, apply  --->  label agent-approved            (the label EVENT records who approved)
    agent-approved                       ^
                                         | poll every 60 s, outbound HTTPS only, no inbound ports
                                         +------------------  agent-bridge worker (SQLite ledger,
                                                              one run at a time, dry run by default)
 3. review the draft PR   <---  branch agent/issue-N-xxxxxxxx   lease -> git worktree -> agent (omp|claude)
                                + DRAFT pull request                -> verify -> commit -> push -> PR
                                + bot comments + labels
 4. merge (a human, never the bridge)
```

Life cycle of one approved issue. Every arrow is a durable ledger checkpoint (`models.RunPhase`):

```text
lease -> worktree -> agent -> verify -> commit -> push -> pull request -> notify -> done
             \          \         \
              +----------+---------+--> blocked / cancelled   (nothing is pushed)
```

Label states on the issue: `agent-task` + `agent-approved` (owner) -> `agent-running` (bridge) ->
`agent-done` (draft PR opened) or `agent-blocked` (needs a human). A cancelled run leaves no terminal label.

## MVP architecture

1. **Task transport:** GitHub issues with a machine-readable `agent-task` label and a strictly parsed
   fenced YAML block (` ```yaml agent-task `) in the issue body; a fine-grained token with minimal
   repository access (a GitHub App was not built). The ChatGPT GitHub connector can publish tasks subject
   to authorization; do not assume automatic task creation.
2. **Local worker:** Python 3.11 service/CLI (`python -m tools.agent_bridge`) on the developer
   workstation, isolated from the mower and base station. It polls GitHub issues; webhooks were not
   built, so no public ingress exists.
3. **Task state:** labels `agent-task`, `agent-approved`, `agent-running`, `agent-blocked`, `agent-done`
   and a durable SQLite ledger for issue number, revision, deduplication key, status, worktree, run ID,
   outputs, PR number and timestamps. Only explicitly approved issues execute. Issue comments and PR text
   are never tool authority: the only GitHub text that reaches an agent is the issue title and the one
   validated `agent-task` block.
4. **Execution:** check out a task-specific branch/worktree, constrain the workspace to it, and launch the
   installed agent CLI non-interactively (`omp -p` by default, `claude -p` as an alternative) with a
   constrained prompt synthesized from trusted configuration plus quoted task content. Structured JSON
   output, wall-clock timeout, per-run budget, concurrency limit 1 and bounded retries. Flags were taken
   from the locally installed CLIs (omp 18.12.6, Claude Code 2.1.293), not assumed.
5. **Agent policy:** `CLAUDE.md` and `AGENTS.md` carry an "Unattended agent runs (agent bridge)" section
   that points at this document; small incremental changes, tests and citations to repository files are
   required. `CLAUDE.md` and `.github/copilot-instructions.md` are also placed in every prompt
   (`prompt.context_files`). `docs/field-integration-observations.md` is not on this branch; add it to
   `prompt.context_files` once it is merged.
6. **Results:** exit status, changed files, check output, commit SHA and a concise summary (secrets
   redacted) are captured; the branch is pushed and a *draft* PR linked to the issue is opened. Status is
   published as an issue comment plus a label transition. ChatGPT can review by fetching the PR and
   commits through its GitHub connection.
7. **Safety:** no auto-merge, no auto-deploy to the Raspberry Pi or Orange Pi, no change to mower
   firmware, control thresholds, starter/PTO/drive or physical safety interlocks. Safety-critical paths
   are manual-review-only (see [Safety model](#safety-model)). Dry run is the default.
8. **Secrets:** tokens never live in the repository or in the configuration file; they come from an
   environment variable or a mode-0600 file. Least privilege, restricted tools, injection defences, no new
   runtime dependencies.
9. **Failure and recovery:** idempotent leasing per issue and approval; stale leases are recovered,
   interrupted work is requeued a bounded number of times, a changed task halts the run, failures are
   reported with actionable hints. There is no unapproved retry and no self-loop.
10. **Testability:** GitHub is replaced by an in-memory fake server and the agent CLIs by fake `omp` and
    `claude` executables; integration tests run in temporary local git repositories (see
    [Implementation status](#implementation-status)). No test touches hardware.

## Pluggable agent backends

One interface (`AgentBackend`) and one result contract (`AgentResult`) serve every agent CLI, so the worker
never needs to know which one ran. Two backends are registered in `agents/__init__.py`: `omp` (OhMyPi, the
default) and `claude` (Claude Code).

Backend selection, first match wins:

1. the task's `agent:` field (`omp` or `claude`),
2. the configuration's `agent.default`,
3. `omp`, the built-in default.

`agent.allowed` (default `[omp, claude]`) lists the backends the operator enables. A task that names a
backend outside that list is refused (`task_not_permitted`: `agent 'x' is not enabled (allowed: ...)`),
and a configuration whose `agent.default` is not in `agent.allowed`, or that names an unknown backend,
fails to load. The choice is stored in the ledger (`runs.backend`) and shown in the "started" and "done"
comments and in the pull request body. Adding a third backend means a new `AgentBackend` subclass plus
entries in `BACKENDS`, `BACKEND_NAMES` and `AgentConfig`.

## Repository layout

As built (the spec asked for `tools/agent_bridge/`, `tests/agent_bridge/`, the example configuration and
two documents):

```text
tools/agent_bridge/           runner, queue API adapter, schema validation, subprocess wrapper, publisher
  __main__.py  cli.py         entry point; sub-commands, dry-run/--execute gating, logging
  config.py    schema.py      strict YAML configuration; task block parser and revision hash
  worker.py    ledger.py      poll/approve/lease/run/verify/publish/recover; SQLite ledger
  models.py    errors.py      shared data types; exception hierarchy
  github.py    gitops.py      REST+GraphQL client and approval helpers; worktrees, diff inspection, push
  policy.py    globs.py       change policy; path normalisation and the glob dialect shared with the guard
  checks.py    prompt.py      named verification commands; prompt synthesis
  redact.py    publisher.py   secret redaction; labels, bot comments, PR body, read-only dry-run wrapper
  process.py   smoke.py       bounded subprocess supervision; real-agent smoke test
  agents/{__init__,base,omp,claude}.py   pluggable backends
  omp_guard.mjs               omp tool-call guard extension
tests/agent_bridge/           unit, security and integration tests; bridgefakes/ holds the fake GitHub
                              server, fake omp, fake claude and the test rig
config/agent-bridge.example.yaml   polling interval, repository, budget, timeout, labels, branch prefix,
                                   dry run; no credentials
docs/agent-bridge.md          this document
docs/agent-bridge-operations.md      installation, start/stop, permissions, systemd (optional), logs,
                                     monitoring, failures
docs/agent-bridge-task-template.md   canonical issue template and acceptance criteria
```

## Build order for coding agent

**Phase 1:** Inspect actual repository structure; produce a short dependency/permissions/design analysis.
Preserve current mower configuration and code behavior.

**Phase 2:** Implement task schema, approval gate, GitHub poller, SQLite ledger, dry-run preview, tests.
No agent execution yet.

**Phase 3:** Add supervised agent invocation (OhMyPi first, Claude Code as an alternative) with explicit
user authorization for each real execution and constrained tool permissions, worktree management, timeout,
budget and structured result capture. Use an SDK only if it materially simplifies reliability and can be
justified.

**Phase 4:** Publish branch + draft PR and bot issue comments. Add failure recovery, tests, documentation,
example configuration.

**Phase 5:** Demonstrate an end-to-end **dry run** with a harmless documentation-change issue. After the
owner opts in, demonstrate one approved real documentation-only task. Do not enable unattended physical
deployment.

## Acceptance criteria

- One approved test issue becomes at most one locally executed agent job, with restart-safe idempotence.
- An unapproved or malformed task cannot execute.
- Agent CLI exit/failure, timeout, network loss, duplicate and cancellation are recoverable and clearly
  reported.
- Tests run without GitHub credentials or any agent CLI installed, via mocks.
- A successful real **documentation-only** job creates a draft PR; no branch is merged automatically.
- The resulting PR and issue state contain enough evidence for ChatGPT to review on request.
- Existing RTK, HaLow, mowing, and safety behavior is unchanged by the bridge.

## Implementation status

Paths are relative to the repository root. "Tests" lists files under `tests/agent_bridge/` (prefix
`test_agent_bridge_`, suffix `.py`); the fakes they use live in `tests/agent_bridge/bridgefakes/`.

### Spec items

| # | Spec item | Delivered by | Tests |
|---|-----------|--------------|-------|
| 1 | Task transport: issues, `agent-task` block, minimal token | `schema.py` (`parse_task`, `TaskSpec`, `hygiene_errors`), `github.py` (`HttpGitHubClient`), `config.py` (`GitHubConfig`, `resolve_github_token`) | `schema`, `github_client`, `fake_github`, `config` |
| 2 | Local worker, polling, no ingress | `worker.py` (`Worker.poll_once`, `serve`), `cli.py` (`run-once`, `serve`), `config.py` (`PollConfig`) | `worker_lifecycle`, `cli`, `worker_flow` |
| 3 | Labels, SQLite ledger, approval, comments are not authority | `ledger.py`, `models.py`, `publisher.py` (`Publisher`), `github.py` (`find_approval`, `edited_after`), `worker.py` (`_evaluate`), `prompt.py` | `ledger`, `publisher`, `worker_security`, `prompt` |
| 4 | Worktree, constrained prompt, JSON output, timeout, budget, one run, bounded retries | `gitops.py`, `prompt.py`, `agents/` (`omp.py`, `claude.py`, `base.py`), `process.py`, `worker.py` (`_agent_phase`, `_limits`), `ledger.py` (`lease`) | `backends`, `process`, `gitops`, `worker_failures`, `fake_clis`, `real_agents` (opt-in) |
| 5 | Agent policy files, small changes, citations | `CLAUDE.md` and `AGENTS.md` sections, `prompt.py` (`build_prompt`, `load_context`), `config.py` (`PromptConfig`) | `prompt`, `config` |
| 6 | Results: files, checks, SHA, summary, draft PR, comment, labels | `worker.py` (`_verify_and_commit`, `_publish`), `checks.py`, `gitops.py` (`commit`, `push`), `publisher.py` (`render_done`, `render_pull_request_body`), `github.py` (`create_pull_request`), `redact.py` | `publisher`, `worker_flow`, `worker_security`, `gitops` |
| 7 | Safety: no merge/deploy, protected paths, draft only, dry run | `config.py` (`DEFAULT_PROTECTED_PATHS`, `dry_run`, `PullRequestConfig.draft`), `policy.py`, `omp_guard.mjs`, `cli.py` (`_authorize_execution`), `publisher.py` (`ReadOnlyGitHub`) | `policy`, `guard`, `config`, `cli`, `worker_security` |
| 8 | Secrets and least privilege | `redact.py`, `config.py` (`resolve_github_token`), `agents/base.py` (environment allow-list), `gitops.py` (askpass, hooks off), `cli.py` (redacting log filter) | `redact`, `backends`, `gitops`, `worker_security` |
| 9 | Idempotent leasing, stale-lock clean-up, recovery, halt on edits | `ledger.py` (`lease`, `stale_runs`, `requeue_interrupted`), `worker.py` (`recover`, `_requeue`, `_resume_publish`, `_gate`), `process.py`, `cli.py` (`recover`, `cancel`) | `worker_lifecycle`, `ledger`, `process`, `worker_failures` |
| 10 | Testability with mocks, temp repos, no hardware | `tests/agent_bridge/bridgefakes/` (`github_server.py`, `fake_omp.py`, `fake_claude.py`, `rig.py`); `tools/agent_bridge` imports nothing from `backend/` | every module above; `fake_github`, `fake_clis`, `worker_flow` test the fakes |

### Phases

| Phase | Status | Evidence |
|-------|--------|----------|
| 1. Inspect, analyse, preserve mower behaviour | Done | design recorded in this document; the change set only adds `tools/agent_bridge/`, `tests/agent_bridge/`, `config/agent-bridge.example.yaml`, documents and one `.gitignore` line; nothing under `backend/`, `frontend/`, `systemd/`, `scripts/` or `spec/` changed |
| 2. Schema, approval gate, poller, ledger, dry-run preview | Done | `schema.py`, `worker._evaluate`, `github.py`, `ledger.py`, `preview` command (`cli.py`, `worker.build_preview`) |
| 3. Supervised agent, tool constraints, worktrees, timeout, budget | Done | `agents/`, `omp_guard.mjs`, `gitops.py`, `process.py`; real execution needs `dry_run: false`, `--execute` and `--yes` (or an interactive "y"). No SDK: both CLIs are driven as subprocesses |
| 4. Branch, draft PR, comments, recovery, docs, example config | Done | `publisher.py`, `worker._publish`, `worker.recover`, these documents, `config/agent-bridge.example.yaml` |
| 5. Live demonstration | **Not done** | the hermetic end-to-end flow runs in `worker_flow` (fake GitHub, fake agents), and `smoke --backend omp --probe-guard` ran against the real `omp` in a throwaway repository on 2026-10-24. A dry run and a real documentation-only task against the real GitHub repository need the owner's token, labels and approval; the procedure is in the operations guide |

### Acceptance criteria

| Criterion | Delivered by | Tests |
|-----------|--------------|-------|
| At most one job per approved issue, restart-safe | `Ledger.lease` (unique `(repo, issue, approval event)` row, `dedup_key`), `Worker._evaluate` (`already_processed`) | `worker_lifecycle`, `ledger` |
| Unapproved or malformed task cannot execute | approval gate in `Worker._evaluate`, `schema.parse_task`, `schema.validate_against_config` | `worker_security`, `schema` |
| Failure, timeout, network loss, duplicate, cancellation recoverable and reported | `_Stop` kinds, `HINTS` and `publisher.render_*` in `worker.py`, `Worker.recover`, `agent-bridge recover/cancel`; failure table in the operations guide | `worker_failures`, `worker_lifecycle`, `github_client` |
| Tests need no credentials and no agent CLI | `bridgefakes/` (fake GitHub through `httpx.MockTransport`, executable fake `omp` and `claude`); real-agent tests are opt-in (`LAWNBERRY_BRIDGE_REAL_AGENTS=1`) | whole suite |
| Real docs-only job creates a draft PR, nothing merged | `PullRequestConfig.draft: Literal[True]`, `create_pull_request(draft=True)`, no merge call anywhere. **The run against live GitHub is pending the owner's opt-in** (Phase 5) | `publisher`, `worker_flow` |
| PR and issue carry enough evidence | `render_done`, `render_pull_request_body` (agent, model, run id, base, approver, cost, files, checks, blocked tool calls, summary), commit trailers (`Agent-Bridge-Run`, `Agent-Backend`, `Approved-By`), ledger `result_json` | `publisher`, `worker_flow` |
| Existing RTK, HaLow, mowing, safety behaviour unchanged | no tracked mower file changed; no import from `backend/`; protected paths keep agents out of hardware code | review of the change set |

## Agent backends

### Shared contract

Every backend receives an `AgentRequest` (prompt, working directory, run directory, scope, timeout,
budget, write allow-sets and deny list, optional exact command allow-list, abort callback) and returns an
`AgentResult`: `status` (`succeeded`, `failed`, `timed_out`, `budget_exceeded`, `cancelled`,
`unavailable`), `failure_kind` (`auth`, `credit`, `rate_limit`, `api_error`, `timeout`, `budget`,
`max_turns`, `nonzero_exit`, `no_result`, `unavailable`, `cancelled`, `hook`), exit code, duration, final
text, the parsed `report`, `cost_usd`, token `usage`, model, session id, turn and tool-call counts,
blocked tool calls and samples, log paths and the command that ran. The agent ends its final message with
one fenced block that is parsed as `AgentReport`:

````text
```json agent-result
{"status": "completed", "summary": "<what changed and why, citing files>",
 "files_changed": ["docs/example.md"], "notes": "<optional>"}
```
````

`status` is `completed`, `blocked` (the agent refused to change anything; the run ends `agent_blocked`) or
`needs_human` (a PR is still opened and flagged). The report is advisory: what is published is decided by
the staged diff, never by the report.

The process is started in its own session and process group with the prompt on **stdin**, a scrubbed
environment and a deadline. Environment: `PATH`, `HOME`, `USER`, `LOGNAME`, locale, temp dir, proxy and CA
variables, plus `TERM=dumb`, `NO_COLOR=1`, `LAWNBERRY_AGENT_RUN_ID` and the names in the backend's
`pass_env`. GitHub tokens, `SSH_AUTH_SOCK`, `GIT_ASKPASS` and anything starting with `LAWNBERRY_BRIDGE_`
can never be passed (listing one in `pass_env` makes the backend refuse to start). When
`agent.command_prefix` is set, the prefix goes in front of the agent command and of every check command.

**Why the prompt goes on stdin:** (1) real `omp` treats positional `@path` arguments as files to include,
and Claude Code's variadic `--tools`/`--allowedTools` options swallow trailing positional arguments, so a
prompt in argv is either misread or eaten; (2) text from a GitHub issue never reaches an argument list, so
there is no argument injection; (3) argv is visible in the process table and limited in length, a pipe is
neither.

### OhMyPi (`omp`) - primary

Exact command, as built by `OmpBackend.build_argv` (docs-only scope; the prompt arrives on stdin):

```text
omp -p --mode json --no-session --no-title --no-skills --no-rules --no-lsp --no-pty \
    --tools read,grep,glob,edit,write --approval-mode write \
    --config <state_dir>/runs/<run_id>/agent/omp-overlay.yml \
    --trusted-extension <repo>/tools/agent_bridge/omp_guard.mjs \
    --max-time <timeout seconds - 5, at least 10> --cwd <state_dir>/worktrees/<run_id> \
    [--model <model>] [--thinking <level>] [<agent.omp.extra_args> ...]
```

For `code` scope with named checks, `bash` is appended to `--tools` and the guard allows exactly the
command lines of those checks. Environment additions: `LAWNBERRY_BRIDGE_GUARD_POLICY`
(`runs/<run_id>/agent/omp-guard-policy.json`), `PI_NO_TITLE=1`, `OMP_MCP_TIMEOUT_MS=2000`.

| Flag | Purpose |
|------|---------|
| `-p`, `--mode json` | non-interactive; one JSON event per stdout line (cost, tool calls, final text) |
| `--no-session`, `--no-title` | nothing written to omp's session store; no title-generation model call |
| `--no-skills`, `--no-rules`, `--no-lsp`, `--no-pty` | no skill or rule discovery, no LSP tools, no PTY shell |
| `--tools ...` | explicit tool allow-list; no web, browser or MCP tools |
| `--approval-mode write` | omp auto-approves read and workspace-write tools; exec tools such as `bash` are not auto-approved (see Known limitations) |
| `--config <overlay>` | hermetic settings overlay (below) |
| `--trusted-extension <abs path>` | loads `omp_guard.mjs`; a flag the installed 18.12.6 accepts although `omp --help` does not list it (absolute path required, cannot be combined with `-e/--extension/--hook`) |
| `--max-time` | omp's own wall-clock stop, five seconds earlier than the bridge's deadline |
| `--cwd` | start in the task worktree |

**Hermetic overlay.** `omp-overlay.yml` is generated per run: all 19 ambient discovery providers are
disabled (`native`, `omp-plugins`, `claude`, `agent-plugins`, `codex`, `agents`, `claude-plugins`,
`gemini`, `opencode`, `cursor`, `windsurf`, `cline`, `github`, `vscode`, `agents-md`, `claude-md`,
`custom-context`, `mcp-json`, `ssh-json`), so no user MCP server, plugin, skill, hook, rule or context file
leaks in; `mcp.enableProjectConfig: false`; `tools.approvalMode: write`; `tools.xdev: false`;
`retry.modelFallback: false` (no silent switch to another model); memory, autolearn, auto-QA, telemetry
export and skills are off. The bridge supplies the repository context itself so both backends see the same
instructions.

**Why the guard exists.** omp's file tools honour absolute paths, `~` and `..`, so by themselves they are
not confined to the task worktree, and its `write` tool can address `xd://` tool devices (the overlay
turns the devices off, the guard also rejects them). `omp_guard.mjs` registers a `tool_call` hook and
checks every call before it executes, against a JSON policy that lives outside the worktree
(`LAWNBERRY_BRIDGE_GUARD_POLICY`):

- only tools named in the policy run (plus the hidden `yield` and `think`);
- every path argument must stay inside the worktree after symlink resolution, must not be a URL or `xd://`
  device, must not start with `~` and must not match the read deny list (`.git`, `.env*`, `*.pem`,
  `*.key`, `id_rsa*`, `config/secrets*.json`, `config/ssl/**`); deny list and write rules are applied
  to the path as written *and* to the symlink-resolved path, so a directory symlink inside `docs/` cannot
  lead to a protected or out-of-scope file;
- `edit` and `write` must also match every write allow-set (scope, then the task's `allowed_paths`) and no
  protected-path glob, may not write through a symlink and may not exceed 1,000,000 bytes;
- `bash`, when enabled at all, runs only command lines listed verbatim, and its path arguments (`cwd`)
  must stay inside the worktree like any other;
- a missing or invalid policy blocks everything (fail closed); each decision is appended to
  `omp-guard-audit.jsonl`, and blocked calls reach the model as errors starting with
  `[agent-bridge guard]`.

The glob dialect is shared with `globs.py`; globs from the configuration are canonicalised on load
(`scripts/` means `scripts/**`, `a//b` becomes `a/b`, a bare `.` is rejected) so a protected glob can never
silently match nothing. `tests/agent_bridge/fixtures/glob_vectors.json` is checked
against both implementations. The guard is defence in depth: the authoritative decision is the staged-diff
policy applied after the run.

**Verified on 2026-10-24** (omp 18.12.6, model `anthropic/claude-sonnet-5-5`):
`python -m tools.agent_bridge smoke --backend omp --model anthropic/claude-sonnet-5-5 --probe-guard`
created the requested file (cost $0.024 for the cold-cache first run, about $0.008 for later ones), and
the guard blocked an attempted write outside the worktree and a read of `/etc/hostname` (2 blocked calls);
the outside file was not created.

### Claude Code (`claude`) - alternative

Exact command, as built by `ClaudeBackend.build_argv` (prompt on stdin):

```text
claude -p --output-format json --no-session-persistence --safe-mode --strict-mcp-config \
       --disable-slash-commands --permission-mode dontAsk --permission-prompts none \
       --max-budget-usd <budget, two decimals, at least 0.01> [--model <model>] [--effort <level>] \
       [<agent.claude.extra_args> ...] \
       --tools Read Edit Write Glob Grep [Bash] \
       --allowedTools Read Edit Write Glob Grep [Bash(<exact command line>) ...]
```

Variadic options come last so nothing can swallow an argument. `--safe-mode` disables CLAUDE.md, hooks,
plugins, MCP servers, skills and custom commands; `dontAsk` with `--permission-prompts none` denies
anything not pre-approved, so no prompt ever waits for an answer that cannot come. `thinking` maps to
`--effort` (`minimal`/`low` -> `low`, `medium`, `high`, `xhigh`, `max`; `off` and `auto` add nothing).
`ANTHROPIC_API_KEY` is passed through by default; with no key Claude Code uses the OAuth login of the
workstation user. `--output-format json` prints one `result` object, or a message array when verbose mode
is on in the user's Claude configuration; both are parsed.

**Not verified end to end.** The real `claude` 2.1.293 accepts every flag above, but its account cannot
authenticate on the workstation used for development: the smoke run returns a structured failure with
`failure_kind: auth` ("Failed to authenticate: OAuth session expired and could not be refreshed"), and with
`ANTHROPIC_API_KEY` passed through `failure_kind: credit` ("Credit balance is too low"). The parsing of
both is covered by captured real outputs (`tests/agent_bridge/fixtures/claude_real_*.json`); a successful
real Claude run has not been observed.

### Budget, timeout and cancellation

| | omp | claude |
|---|-----|--------|
| Budget | omp has no spend cap. `OmpStreamHandler` sums `usage.cost.total` of every streamed assistant message; when the total reaches the budget and the model wants to continue (`stopReason` `toolUse` or `length`) the bridge ends the run before the next model call (`agent_budget`). The check sits between model calls, so one call can overshoot; a final answer that crosses the line is kept | Claude Code's own `--max-budget-usd` stops the run (`error_max_budget_usd`); the bridge reports `agent_budget` |
| Timeout | bridge deadline `timeout_minutes` (SIGTERM to the process group, SIGKILL after `kill_grace_seconds`) plus omp's own `--max-time` (deadline minus 5 s) | bridge deadline only |
| Cancellation | the abort check below, identical for both: process group SIGTERM, grace, SIGKILL | same |

Effective budget = `min(task budget_usd or limits.budget_usd, limits.max_budget_usd)` (default $2, cap $5);
effective timeout = `min(task timeout_minutes or limits.timeout_minutes, limits.max_timeout_minutes)`
(default 20 min, cap 60). The agent's cost is reported in the comment, the PR and `status`; a run that
produces no cost event is shown as `n/a`.

The **abort check** runs every `agent.cancel_poll_seconds` (15 s) while the agent or a check is running. It
refreshes the lease and ends the run when the bridge is shutting down (the run is requeued), when
`agent-bridge cancel` was issued, when the issue was closed, when `agent-approved` was removed (all
cancelled) or when the issue text no longer matches the approved revision (blocked `task_changed`). A GitHub
outage during the check does not cancel anything. `agent-bridge cancel` is also honoured right after the
lease (a pending run never starts), at the start of verification and before every publish attempt, so
nothing is pushed after it.

## Safety model

| Control | What it does | Where |
|---------|--------------|-------|
| **Approval gate** | an issue runs only if it carries `agent-task` and `agent-approved`, its author is in `task_authors` (default: the approvers), and the actor of the *label event* that applied `agent-approved` (read from GitHub's issue events, not from anything the issue says) is in `approvers`. Anyone else's approval is ignored silently. An empty `approvers` list is a configuration error (fail closed) | `worker._evaluate`, `github.find_approval`, `config.BridgeConfig` |
| **Edit-after-approval invalidation** | an edit or rename at or after the approval time (GraphQL `lastEditedAt`, `renamed` events) makes the approval stale (`approval_stale`; fail closed on equal timestamps). The owner must re-read the text and re-apply the label | `github.edited_after` |
| **Revision binding** | the run is bound to `sha256(title + body)` and the approval event id. The hash is compared at lease time, every abort poll and before each publish attempt; a difference ends the run `task_changed` and nothing is pushed. The ledger key `(repo, issue, approval event)` is unique, so one approval is one run | `schema.compute_revision`, `ledger.lease` |
| **Strict task parsing** | exactly one fenced `agent-task` block; YAML without anchors, aliases, tags or duplicate keys; unknown fields rejected; hidden HTML comments, invisible and bidirectional characters, control characters and oversize input rejected so the approver sees everything an agent receives | `schema.py` |
| **Protected paths** | 43 globs (safety chain, drivers, motor/blade/RoboHAT/tractor/Thor-link/power services, core, hardware and limit configuration, secrets, SSL, `systemd/`, `scripts/`, `spec/`, `hardware/`, `.github/`, agent policy files, `tools/agent_bridge/`, dependency manifests) can never be changed by a run. Setting `protected_paths` *replaces* the list | `config.DEFAULT_PROTECTED_PATHS`, `policy.py` |
| **Scope and size limits** | `docs-only` (`docs/**`, `*.md`) or `code` (adds `tools/**`, `tests/**`), further narrowed by the task's `allowed_paths`. At most 40 files, 400,000 diff bytes, 1,000,000 bytes per file; no symlinks, submodules, executables, binary files or type changes; paths must be canonical | `policy.evaluate_changes` |
| **Draft only, no merge, no deploy** | pull requests are always drafts (`draft: Literal[True]`); the code has no merge or deploy call; the only network peers are GitHub and the agent's model provider; the only subprocesses are `git`, the agent CLI and configured checks. Pushes are never forced | `config.PullRequestConfig`, `github.create_pull_request`, `gitops.push` |
| **Dry run** | `dry_run: true` by default; a real run needs `dry_run: false`, `--execute` and `--yes` (or an interactive "y"); a dry run wraps the GitHub client so any write raises | `cli._authorize_execution`, `publisher.ReadOnlyGitHub` |
| **Scrubbed environment** | agents get an allow-list environment only; checks get `PATH HOME USER LOGNAME LANG TMPDIR` plus the values written in the configuration; git gets a small allow-list | `agents/base.py`, `checks.py`, `gitops.py` |
| **Secret redaction** | the GitHub token and `extra_secret_env` values are scrubbed literally, and GitHub/Anthropic/OpenAI/AWS/Slack/Google keys, private keys, bearer tokens, JWTs, URL credentials and `NAME=value` secrets by pattern, from logs, captured output, comments, PR text and commit messages. Added diff lines are scanned and a hit blocks the run (`policy_violation`, code `secret`; the value is never stored) | `redact.py` |
| **Prompt-injection defences** | only the title and the validated block reach the agent; trusted rules come first and the task is quoted between random-nonce markers with every line prefixed, labelled as data; commands come only from configuration by name (an issue can never supply a command line); agent text is untrusted, redacted and rendered inside a code fence | `prompt.py`, `schema.py`, `publisher.py` |
| **Git hardening** | hooks disabled (`core.hooksPath=/dev/null`, `--no-verify`), no fsmonitor, no external protocols, no prompts, no signing; the worktree's `.git` pointer and `HEAD` are checked after the agent (`worktree_tampered`); agents are told not to use git and the file-only tool set cannot | `gitops.py`, `worker._verify_and_commit` |
| **One run at a time** | a ledger lease allows a single active run (`busy` otherwise); `poll.max_jobs_per_poll` defaults to 1; at most 3 runs per issue; identical task content in another issue is refused (`duplicate_task`) | `ledger.lease`, `config.LimitsConfig` |
| **Not on the mower** | the worker runs on the developer workstation only; it imports nothing from `backend/`, holds no mower credentials and has no SSH or hardware path | design; install guidance in the operations guide |

What the bridge does **not** protect against: a model that makes a wrong but in-scope change (a human reviews
every draft PR); an agent CLI that is itself compromised or holds credentials in its own configuration
directory (the omp process runs as the workstation user, the guard confines its *tool calls*); network egress
(the bridge does not firewall the agent). `agent.command_prefix` is the hook for an operating-system sandbox
(see the operations guide).

## Known limitations

- **Phase 5 is pending**: no run against the real GitHub repository has happened yet.
- **Claude Code is unproven end to end** (authentication fails on the development workstation) and has no
  bridge-owned per-path guard: its confinement is Claude Code's own permission system (`--safe-mode`,
  `dontAsk`, explicit tool list). The post-run diff policy still decides what is published. `claude
  --restricted` (file tools confined to the working directories) exists in 2.1.293 but is not enabled.
  Prefer `omp` for unattended runs.
- **`code` scope has only been exercised against the fake CLIs.** Real omp in approval mode `write`
  auto-approves read and workspace-write tools; whether it lets the agent run an allow-listed `bash`
  command unattended has not been verified. The bridge runs the named checks itself after the agent
  regardless, so the gate does not depend on it.
- **No operating-system sandbox by default.** Bubblewrap is installed on the development workstation but
  unusable there (unprivileged user namespaces are denied). `agent.command_prefix` can add one elsewhere.
- A requeued (`pending`) run whose issue was closed or un-approved while it waited stays `pending`; it blocks
  nothing. If the bridge dies abruptly (`kill -9`, power loss) the kernel's `PR_SET_PDEATHSIG` reaps only
  the agent's direct child; processes it spawned can outlive it.
- Agent and omp/claude CLI versions are not pinned or enforced; `agent-bridge backends` prints what is
  installed and `smoke` is the compatibility check after an upgrade.

## Deviations from the original spec

| Spec | As built | Why |
|------|----------|-----|
| Python 3.12+ | Python **3.11** | the repository pins `requires-python = ">=3.11,<3.13"` and `.python-version` is 3.11; CI lints with 3.11; nothing in the bridge needs 3.12 |
| `tools/agent_bridge/` | `tools/agent_bridge/`, run from the repository root as `python -m tools.agent_bridge` | `pip install -e .` packages only `backend*`, so `tools` is not installed; it is a developer tool and must never ship to the mower |
| Claude Code worker | OhMyPi primary, Claude Code selectable | one backend interface; see [Pluggable agent backends](#pluggable-agent-backends) |
| Labels `agent-task`, `agent-approved`, `agent-running`, `agent-blocked`, `agent-done` | exactly those names (configurable under `labels:`, created by `setup-labels --yes`), plus rules: the bridge never applies or removes `agent-approved`; `agent-done` or `agent-blocked` makes an issue ineligible until a human removes it; `agent-running` is removed on every ending; a cancelled run gets no terminal label | the approval label is the owner's signal, so only a human may touch it |
| Cancellation recoverable and reported | cancellation = issue closed, `agent-approved` removed, or `agent-bridge cancel`; effective within `cancel_poll_seconds` (15 s) while the agent or a check runs, and at the phase boundaries described above. A cancelled run is terminal: a new run needs a new approval event. A bridge shutdown is *not* a cancel: that run is requeued (`auto_retry_interrupted` 1, `max_start_retries` 3) | separates "owner changed their mind" from "the worker stopped" |
| Webhooks or polling | polling only | no public ingress |
| GitHub App or fine-grained token | fine-grained token | simpler; comments appear as the token's account, so a dedicated machine account (not in `approvers`) is recommended |
| "Explicit user authorization for each real execution" | the approval label authorises each task; `--execute --yes` authorises the worker session | per-task approval is recorded as a GitHub label event, which is auditable |
| "Bounded retries" | interrupted runs requeue automatically (above); a failed run is never retried automatically; at most `max_attempts_per_issue` (3) runs per issue | no unapproved retry |
| "Pin external dependency versions" | no new runtime dependency (`httpx`, `pydantic`, `PyYAML` already in `pyproject.toml`); CI pins ruff 0.15.19; agent CLI versions are reported, not enforced | see Known limitations |

## Initial agent instruction

> In KhryptorGraphics/lawnberry-pi, implement `docs/agent-bridge.md` as the top-priority development
> project. Build a GitHub-issue-driven local coding-agent worker (OhMyPi `omp` primary, Claude Code
> `claude` as a selectable alternative behind one backend interface) with approval-gated execution,
> durable state, worktree isolation, dry run default, tests and draft PR reporting. Begin by reading the
> entire specification and inspecting the repository. Keep all mower hardware controls untouched; do not
> deploy to the mower or auto-merge. Document setup and demonstrate the workflow with a harmless
> documentation task.
