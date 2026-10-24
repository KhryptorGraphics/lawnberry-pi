# Claude Code ↔ LawnBerry Pi development bridge

**Status:** Specification / coding-agent implementation assignment. **Date:** 2026-10-24. **Priority:** Build this workflow before expanding hardware implementation.

## Goal
Allow the owner to discuss LawnBerry Pi with ChatGPT, have ChatGPT publish structured tasks to GitHub, and let a locally running Claude Code worker pick up **approved** tasks, execute the development work, run checks, create a branch and pull request, and publish a summary for ChatGPT to inspect on the next conversation turn. This is a GitHub-mediated workflow, **not** a direct live connection to the ChatGPT voice session. The user may need to explicitly request task publication/review; ChatGPT does not automatically observe speech or run continuously.

## MVP architecture
1. **Task transport:** GitHub issues with a machine-readable `agent-task` label and structured issue body (or a strictly parsed fenced YAML block); use a GitHub App or fine-grained token with minimal repository access. The ChatGPT GitHub connector can publish tasks subject to authorization; do not assume automatic task creation.
2. **Local worker:** Python 3.12+ service/CLI on the developer workstation, isolated from the mower and base station, polls GitHub issues or receives validated webhook notifications. Polling first avoids exposing a public ingress.
3. **Task state:** Use labels `agent-task`, `agent-approved`, `agent-running`, `agent-blocked`, `agent-done` and a durable SQLite ledger for issue number, revision, deduplication key, status, worktree, run ID, outputs, PR number, timestamps. Only explicitly approved issues execute. Never treat issue comments or PR text as untrusted tool authority.
4. **Execution:** Check out task-specific branch/worktree, constrain workspace to this repository, launch installed `claude -p` noninteractively with a constrained prompt synthesized from trusted configuration plus quoted task content. Prefer structured JSON output, timeout, per-run budget, concurrency limit 1, and bounded retries. Check locally installed CLI options and documented flags instead of assuming exact syntax.
5. **Agent policy:** Add `AGENTS.md` or `CLAUDE.md` referencing project conventions, hardware inventory, `docs/field-integration-observations.md` if present, and this bridge document; use smaller incremental changes, tests and citations to repository files.
6. **Results:** Capture exit status, changed files, test output, commit SHA, and concise summary (redact secrets); push to agent branch; create *draft* PR linked to issue. Publish status as a bot issue comment and label transition. ChatGPT can then review by fetching PR/commits through its GitHub connection.
7. **Safety:** Do not auto-merge, auto-deploy to Raspberry Pi or Orange Pi, alter mower firmware/control thresholds, actuate starter/PTO/drive, or weaken physical safety interlocks. Any safety-critical changes must be manual-review-only. Dry-run mode default for initial verification.
8. **Secrets:** Never commit tokens, logs containing credentials or local secrets. Configure via OS keyring/environment variables or secret store. Least privilege. Restrict executable tools/commands, protect against prompt injection, and pin external dependency versions.
9. **Failure & recovery:** Idempotent leasing per issue/revision; clean stale locks, recover interrupted work, halt on changed task content, report failures with actionable logs. No infinite self-loop or unapproved retries.
10. **Testability:** Mock GitHub API and Claude subprocess; integration test in temporary local Git repo; exercise duplicate issue, cancelled issue, malformed input, command failure, lost network, task edited mid-run, unauthorized task, dry run and permission denial. Test no remote hardware access.

## Suggested repo layout
- `tools/agent_bridge/`: runner, queue API adapter, schema validation, subprocess wrapper, result publisher
- `tests/agent_bridge/`: unit, security, and integration tests with mocks
- `config/agent-bridge.example.yaml`: polling interval, repository, budget, timeout, approved labels, branch prefix, dry run; no credentials
- `docs/agent-bridge-operations.md`: local installation, start/stop, permissions, systemd **optional**, log rotation, monitoring, failures
- `docs/agent-bridge-task-template.md`: canonical issue template and acceptance criteria

## Build order for coding agent
**Phase 1:** Inspect actual repository structure; produce short dependency/permissions/design analysis. Preserve current mower configuration and code behavior.

**Phase 2:** Implement task schema, approval gate, GitHub poller, SQLite ledger, dry-run preview, tests. No Claude execution yet.

**Phase 3:** Add supervised Claude Code invocation with explicit user authorization for each real execution and constrained tool permissions, worktree management, timeout, budget, structured result capture. Use SDK only if it materially simplifies reliability and can be justified.

**Phase 4:** Publish branch + draft PR and bot issue comments. Add failure recovery, tests, documentation, example configuration.

**Phase 5:** Demonstrate end-to-end **dry run** with a harmless documentation-change issue. After owner opts in, demonstrate one approved real documentation-only task. Do not enable unattended physical deployment.

## Acceptance criteria
- One approved test issue becomes at most one locally executed Claude job, with restart-safe idempotence.
- An unapproved or malformed task cannot execute.
- CLI exit/failure, timeout, network loss, duplicate and cancellation are recoverable and clearly reported.
- Tests run without GitHub credentials or Claude installed via mocks.
- A successful real **documentation-only** job creates a draft PR; no branch is merged automatically.
- The resulting PR and issue state contain enough evidence for ChatGPT to review on request.
- Existing RTK, HaLow, mowing, and safety behavior is unchanged by the bridge.

## Initial agent instruction
> In KhryptorGraphics/lawnberry-pi, implement `docs/claude-code-bridge.md` as the top-priority development project. Build a GitHub-issue-driven local Claude Code worker with approval-gated execution, durable state, worktree isolation, dry run default, tests and draft PR reporting. Begin by reading the entire specification and inspecting the repository. Keep all mower hardware controls untouched; do not deploy to the mower or auto-merge. Document setup and demonstrate the workflow with a harmless documentation task.
