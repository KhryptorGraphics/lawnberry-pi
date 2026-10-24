# Agent bridge: task template

How to write a GitHub issue that the agent bridge will accept, and how to approve it. Background:
[agent-bridge.md](agent-bridge.md). Running the worker: [agent-bridge-operations.md](agent-bridge-operations.md).

An issue becomes a task when it carries the labels `agent-task` and `agent-approved`, its author is in the
configured `task_authors`, **and** one fenced YAML block tagged `agent-task` in the body validates. Only the
issue title and that block ever reach the agent.

## Canonical template

Title: a short imperative sentence, at most 256 characters (it becomes the pull request title, prefixed
`[agent] `). Body, copied as is (every optional field is commented out; uncomment what you need):

````markdown
One or two sentences of context for the human reviewer. The agent never sees this text.

```yaml agent-task
version: 1                    # required, always 1
scope: docs-only              # docs-only (default) | code
# agent: omp                  # omp | claude; default comes from the bridge configuration
objective: |
  What to change and why, 10 to 8000 characters. Name the files. One outcome per task.
# allowed_paths:              # narrow the writable paths further (globs, at most 30)
#   - docs/OPERATIONS.md
# context_files:              # files the agent reads first (plain paths, at most 20)
#   - CLAUDE.md
acceptance_criteria:          # checkable statements, at most 30, each 1 to 500 characters
  - Only the named file changes.
# checks:                     # names defined in the bridge configuration, at most 10
#   - ruff
# budget_usd: 1.0             # above 0; capped by limits.max_budget_usd
# timeout_minutes: 10         # above 0; capped by limits.max_timeout_minutes
# notes: Free text, at most 4000 characters.
```
````

Do not leave HTML comments (`<!-- ... -->`) in the body, for instance hints from a GitHub issue template:
GitHub hides them from reviewers, so the bridge rejects the issue. YAML `#` comments inside the block are fine.

## Fields

Unknown keys are errors (`extra_field: Extra inputs are not permitted`); surrounding whitespace of strings is
stripped. Limits marked *config* come from the bridge configuration and are checked in a second stage.

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `version` | integer | yes | exactly `1` |
| `objective` | string | yes | 10 to 8000 characters. What the agent should achieve; shown to the agent as quoted data |
| `scope` | `docs-only` or `code` | no (`docs-only`) | `docs-only` may change `docs/**` and `*.md`; `code` may also change `tools/**` and `tests/**` (the `scopes.docs_paths` / `scopes.code_paths` lists). The scope must be enabled in `scopes.allowed` (*config*; the stock configuration enables only `docs-only`). Protected paths are never writable, in any scope |
| `agent` | string | no | `omp` or `claude` (pattern `^[a-z][a-z0-9_-]{0,31}$`); must be in `agent.allowed` (*config*). Default: `agent.default` (`omp`) |
| `allowed_paths` | list of globs | no | at most 30. ASCII letters, digits and `. _ - / * ?` only; repository-relative (no leading `/` or `~`, no `..` segment). Further **narrows** the scope: a changed file must match the scope **and** one of these. `*` and `?` stay inside one directory, `**` crosses directories |
| `context_files` | list of paths | no | at most 20 plain ASCII repository paths (no `*` or `?`). The agent is told to read them first. Reading is not restricted by the write rules |
| `acceptance_criteria` | list of strings | no | at most 30 items, each 1 to 500 characters |
| `checks` | list of names | no | at most 10, unique, pattern `^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$`. Each name must be defined under `checks:` in the bridge configuration (*config*); the bridge runs them itself after the agent, together with `default_checks`. An issue can never supply a command line |
| `budget_usd` | number | no | above 0 and at most 1000; at most `limits.max_budget_usd` (*config*, 5.0). Default `limits.budget_usd` (2.0) |
| `timeout_minutes` | number | no | above 0 and at most 1440; at most `limits.max_timeout_minutes` (*config*, 60). Default `limits.timeout_minutes` (20) |
| `notes` | string | no | at most 4000 characters; shown to the agent as quoted data |

Block rules: exactly one fenced block whose info string is `agent-task` or `yaml agent-task` (or `yml`), made
of at least three backticks or tildes; the block is at most 20,000 characters, the whole body at most 60,000.
It is a single YAML document with string keys and **no anchors, aliases, custom tags or duplicate keys**. The
title and body may not contain hidden HTML comments, control characters or invisible/bidirectional Unicode
(zero-width and direction-override characters are rejected, so the approver sees everything the agent gets).

What the agent receives, as quoted untrusted data: the issue number, the title, the scope, the objective, the
acceptance criteria, the files to read first and the notes. What the bridge enforces itself regardless of what
the agent does: scope, `allowed_paths`, protected paths, limits on files and bytes, checks, budget, timeout
and agent choice.

## What is ignored

- All prose before, between and after the block, and every other fenced block (for example ` ```json `).
- Issue comments, the discussion thread, assignees, milestones, projects and every label except the five
  `agent-*` labels.
- YAML `#` comments inside the block.

Ignored is not the same as safe to edit: **any change to the title or body after approval** (prose included)
makes the approval stale, and a change during a run ends it (`task_changed`). Content that GitHub would hide
from a reviewer is not ignored but rejected.

## Writing good acceptance criteria

The agent reads them as goals; the bridge does not execute them. Only `checks` and the change policy are
enforced mechanically, so pair a criterion with a check where one exists.

- Make each one verifiable by looking at the diff or at a check result: "Only `docs/OPERATIONS.md` changes",
  "`ruff check` and `ruff format --check` are clean", not "the docs are better".
- Say what must **not** change, and name files: the diff is limited to 40 files, 400,000 bytes, 1,000,000
  bytes per file, so ask for small steps.
- One outcome per task. If it needs hardware control, safety logic, firmware, thresholds, deployment, secrets
  or CI, it is not a bridge task: the agent is told to stop (`blocked`) and those paths are protected anyway.
- Ask for citations ("cite the source file for every default you document") so the reviewer, or ChatGPT reading
  the PR later, can check the claims.
- Put exact commands in the bridge configuration, never in the issue; request them with `checks`.

## Worked examples

All three were run through `python -m tools.agent_bridge validate-task --file <body> --title <title>`;
the outputs below are real.

### 1. Documentation only

Title: `Document the LAWNBERRY_*_DIR overrides in the operations guide`

````markdown
The `LAWNBERRY_*_DIR` overrides are mentioned in CLAUDE.md but not in the operations guide.

```yaml agent-task
version: 1
scope: docs-only
objective: |
  Add a short "Directory overrides" section to docs/OPERATIONS.md that lists
  LAWNBERRY_CONFIG_DIR, LAWNBERRY_LOG_DIR and LAWNBERRY_DATA_DIR: what each one
  moves, its default location on the Pi, and one example of setting it in a
  systemd drop-in. Read backend/src/core/config_loader.py to confirm the defaults.
allowed_paths:
  - docs/OPERATIONS.md
context_files:
  - CLAUDE.md
  - backend/src/core/config_loader.py
acceptance_criteria:
  - Only docs/OPERATIONS.md changes.
  - Every default path in the new section is confirmed in the source code, with the file cited.
  - The new section does not suggest restarting lawnberry-backend remotely.
budget_usd: 1.0
timeout_minutes: 10
```
````

```text
task is valid
  revision 370f428565c2bca6  scope docs-only  agent omp
```

The last criterion reflects a hard-won rule in this repository: stopping the backend can take the Pi off the
network, so documentation must not casually advise it.

### 2. Code scope with checks and an agent override

Title: `Add unit tests for backend/src/nav/geoutils.py`

````markdown
`backend/src/nav/geoutils.py` has no direct tests. Add some without touching the module.

```yaml agent-task
version: 1
scope: code
agent: claude
objective: |
  Create tests/unit/test_geoutils.py with pytest tests for haversine_m and
  point_in_polygon from backend/src/nav/geoutils.py: identical points, a known
  distance (one degree of latitude is about 111.2 km), antimeridian-free
  polygons, a point on an edge and a point on a vertex. Do not modify
  backend/src/nav/geoutils.py; if a test reveals a bug, keep the failing case
  out of the file and describe the bug in the final summary.
allowed_paths:
  - tests/unit/test_geoutils.py
context_files:
  - backend/src/nav/geoutils.py
  - tests/unit/conftest.py
acceptance_criteria:
  - Only tests/unit/test_geoutils.py is added.
  - The new tests pass under SIM_MODE=1 and ruff check/format are clean.
  - No test needs network access or hardware.
checks:
  - ruff
  - unit-tests
budget_usd: 3.0
timeout_minutes: 30
notes: Pure functions only; do not import hardware drivers.
```
````

Checked against a configuration that enables `code` and defines `ruff` and `unit-tests`:

```text
task is valid
  revision ca7bd6738de02a24  scope code  agent claude
```

Checked against the stock `config/agent-bridge.example.yaml` it is refused, and the issue gets one comment with
these reasons (exit 1):

```text
task is well formed but not permitted by this configuration:
  - scope 'code' is not enabled (allowed: docs-only)
  - check 'ruff' is not defined in the bridge configuration
  - check 'unit-tests' is not defined in the bridge configuration
```

Enable the scope (`scopes.allowed: [docs-only, code]`) and define the checks (examples in the operations
guide) deliberately before approving such a task: checks execute agent-written code on the workstation.
`agent: claude` only works when the `claude` CLI is logged in for the bridge user; drop the line to use the
default `omp`. `allowed_paths` shrinks the writable set to one new file, and `tools/agent_bridge/**`,
`pyproject.toml` and the rest of the protected list stay out of reach anyway.

### 3. A deliberately invalid task

Title: `Fix everything`

````markdown
Please fix everything about the mower.

```yaml agent-task
version: 2
scope: everything
objective: fix it
agent: gemini
allowed_paths:
  - /etc/passwd
  - ../secrets/**
  - docs/**
checks:
  - make-all
  - ruff
  - ruff
budget_usd: 50
extra_field: true
```
````

Stage 1, the schema (the issue gets an `invalid_task` comment listing these):

```text
INVALID task:
  - version: Input should be 1
  - objective: String should have at least 10 characters
  - scope: Input should be 'docs-only' or 'code'
  - allowed_paths: Value error, glob must be repository-relative: '/etc/passwd'
  - checks: Value error, checks must not repeat
  - extra_field: Extra inputs are not permitted
```

A field that fails reports its first problem only (`../secrets/**` surfaces after `/etc/passwd` is fixed).
`agent: gemini` and `budget_usd: 50` are well-formed values, so they are judged in stage 2, once the block is
otherwise valid. A repaired variant (`version: 1`, a real objective, `agent: gemini`, `checks: [make-all]`,
`budget_usd: 50`, `timeout_minutes: 500`) gets the stage 2 verdict against the stock configuration
(`task_not_permitted`):

```text
task is well formed but not permitted by this configuration:
  - agent 'gemini' is not enabled (allowed: omp, claude)
  - check 'make-all' is not defined in the bridge configuration
  - budget_usd 50.0 exceeds the configured maximum 5.0
  - timeout_minutes 500.0 exceeds the configured maximum 60.0
```

Other rejections seen in practice:

| Mistake | Message |
|---------|---------|
| an HTML comment such as `<!-- note -->` outside a code fence | `body contains an HTML comment outside a code fence; GitHub hides it from reviewers, so it is not allowed` |
| two `agent-task` blocks | ``expected exactly one fenced ```yaml agent-task block, found 2`` |
| YAML anchors or aliases (`&goal`, `*goal`) | `YAML anchors and aliases are not allowed` |
| a zero-width space pasted into the block | `body contains invisible or control characters (U+200B@100); remove them so a reviewer can see the whole task` |
| no block, or a block tagged only `yaml` | ``expected exactly one fenced ```yaml agent-task block, found 0`` |

## Approval procedure

The owner (a login in the bridge's `approvers`) approves, in the GitHub web interface, in this order:

1. **Create the issue** with the template. ChatGPT or the GitHub connector may draft and publish it when its
   account is in `task_authors`; publication is never approval.
2. **Apply the label `agent-task`** (it marks the issue as a candidate; it does not authorise anything).
3. **Validate it:** `python -m tools.agent_bridge validate-task --issue N` (or `--file` before publishing).
   A refusal here is the exact message the issue would receive.
4. **Read the whole issue as GitHub renders it.** What you see is what the agent gets: hidden content is
   rejected. Check the objective, the scope, `allowed_paths`, `checks`, the budget and timeout.
5. **Apply `agent-approved` yourself.** The bridge trusts the *actor of that label event*, so do not delegate
   this step to a tool that acts under your account: it would be indistinguishable from you. An approval by
   anyone outside `approvers` is ignored without a trace.
6. **Do not edit the issue afterwards.** An edit at or after the approval, even to prose or the title, makes
   it stale: the bridge posts one comment (`approval_stale`) and waits. To fix a typo, edit, wait a few
   seconds, then remove and re-apply `agent-approved`.
7. Within `poll.interval_seconds` (60 s) a running worker takes it: the label `agent-running` and a
   "started" comment appear. Then either a draft pull request and the label `agent-done`, or a comment and the
   label `agent-blocked`.
8. **Review the draft PR** like any other (it is never merged for you). To run the task again, follow the
   retry procedure in the operations guide.

To withdraw an approval before or during a run, remove `agent-approved` or close the issue: a running agent is
stopped at the next abort poll (`agent.cancel_poll_seconds`, 15 s, plus the kill grace), and nothing is pushed
afterwards.
