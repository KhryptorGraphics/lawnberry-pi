"""The bridge worker: poll, decide, lease, run, verify, publish - and recover after a crash.

Life cycle of one approved issue (every arrow is a durable ledger checkpoint)::

    lease -> worktree -> agent -> verify -> commit -> push -> pull request -> notify -> done
                 \\          \\         \\
                  +----------+---------+--> blocked / cancelled (never pushed)

Nothing here can reach the mower: the only subprocesses are ``git``, the configured agent CLI and
the configured checks, and the only network peers are GitHub and the model provider the agent
CLI itself talks to.
"""

from __future__ import annotations

import logging
import os
import shlex
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agents.base import AgentBackend, AgentRequest, AgentResult, AgentStatus
from .checks import CheckResult, run_checks
from .config import BridgeConfig
from .errors import (
    CheckAborted,
    GitError,
    GitHubError,
    GitHubNotFound,
    GitHubPermissionError,
    GitHubUnavailable,
    RunStateError,
    TaskValidationError,
)
from .github import GitHubClient, edited_after, find_approval
from .gitops import GitOps, worktree_pointer
from .ledger import LeaseRequest, Ledger, RunRecord
from .models import Approval, Issue, RunPhase, RunStatus
from .policy import Violation, build_write_policy, evaluate_changes
from .process import SubprocessRunner, is_same_process, kill_process_group
from .prompt import build_prompt, load_context
from .publisher import (
    Publisher,
    render_blocked,
    render_cancelled,
    render_done,
    render_pull_request_body,
    render_refusal,
    render_started,
)
from .redact import Redactor, strip_control_chars, truncate
from .schema import ParsedTask, compute_revision, parse_task, validate_against_config

__all__ = ["Eligible", "PollReport", "Preview", "Refused", "Worker"]

LOG = logging.getLogger("agent_bridge.worker")

_NETWORK_GIT_HINTS = (
    "could not resolve host",
    "connection timed out",
    "connection refused",
    "failed to connect",
    "network is unreachable",
    "unable to access",
    "early eof",
    "rpc failed",
    "remote end hung up",
    "operation timed out",
    "temporary failure in name resolution",
    "timed out after",
)

HINTS: dict[str, list[str]] = {
    "auth": [
        "The agent CLI is not authenticated on the bridge workstation. Log in again "
        "(for omp: `omp login`; for claude: `claude auth login`) and re-approve.",
    ],
    "credit": ["The model account is out of credit. Top it up, then re-approve."],
    "rate_limit": ["The model provider rate-limited the run. Wait a while, then re-approve."],
    "api_error": ["The model API returned an error; see the run logs on the workstation."],
    "timeout": [
        "The run hit its time limit. Narrow the task, or raise `timeout_minutes` "
        "(capped by the bridge configuration), then re-approve."
    ],
    "budget": [
        "The run hit its spend cap. Narrow the task, or raise `budget_usd` "
        "(capped by the bridge configuration), then re-approve."
    ],
    "max_turns": ["The agent ran out of turns; split the task into smaller ones."],
    "nonzero_exit": ["The agent CLI exited abnormally; see the run logs on the workstation."],
    "no_result": ["The agent CLI produced no result; see the run logs on the workstation."],
    "unavailable": ["Install the agent CLI on the workstation or choose the other backend."],
}


# ----------------------------------------------------------------------------- data types
@dataclass(frozen=True)
class Eligible:
    issue: Issue
    task: ParsedTask
    approval: Approval
    backend: str


@dataclass(frozen=True)
class Refused:
    issue_number: int
    code: str
    messages: list[str]
    notify: bool = False
    approval: Approval | None = None

    def render(self) -> str:
        return f"#{self.issue_number}: {self.code}: {'; '.join(self.messages)}"


@dataclass
class Preview:
    issue_number: int
    title: str
    backend: str
    scope: str
    approver: str
    approval_event_id: int
    revision: str
    budget_usd: float
    timeout_minutes: float
    branch: str
    base: str
    write_allow_sets: list[list[str]]
    write_deny: list[str]
    checks: list[str]
    command: list[str]
    prompt: str
    prompt_chars: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "prompt"}
        if include_prompt:
            data["prompt"] = self.prompt
        return data


@dataclass
class PollReport:
    considered: int = 0
    previews: list[Preview] = field(default_factory=list)
    refused: list[Refused] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class _Stop(Exception):
    """Controlled end of a run in a terminal state."""

    def __init__(
        self,
        status: RunStatus,
        kind: str,
        detail: str,
        *,
        hints: list[str] | None = None,
        violations: list[str] | None = None,
    ) -> None:
        super().__init__(f"{status.value}: {kind}: {detail}")
        self.status = status
        self.kind = kind
        self.detail = detail
        self.hints = hints or []
        self.violations = violations or []


class _Requeue(Exception):
    """The run never got going (or was interrupted): return it to ``pending``."""


class _Deferred(Exception):
    """Publishing is stuck on the network; resume on a later pass."""


@dataclass
class _Ctx:
    run: RunRecord
    el: Eligible
    run_dir: Path
    worktree: Path | None = None
    pointer: str | None = None
    agent: AgentResult | None = None
    files: list[str] = field(default_factory=list)
    added: int = 0
    deleted: int = 0
    checks: list[CheckResult] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        agent = self.agent.to_dict() if self.agent else None
        if agent:
            agent["final_text"] = truncate(agent.get("final_text") or "", 6000)
        return {
            "title": self.el.task.title,
            "agent": agent,
            "files": self.files,
            "added": self.added,
            "deleted": self.deleted,
            "checks": [c.to_dict() for c in self.checks],
            "violations": self.violations,
        }


def _network_git(exc: GitError) -> bool:
    text = f"{exc} {exc.stderr}".lower()
    return any(hint in text for hint in _NETWORK_GIT_HINTS)


class Worker:
    def __init__(
        self,
        cfg: BridgeConfig,
        *,
        github: GitHubClient,
        ledger: Ledger,
        git: GitOps,
        backends: Mapping[str, AgentBackend],
        redactor: Redactor | None = None,
        runner: SubprocessRunner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        worker_id: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.github = github
        self.ledger = ledger
        self.git = git
        self.backends = dict(backends)
        self.redactor = redactor or Redactor()
        self.runner = runner or SubprocessRunner(redact=self.redactor.redact)
        self.publisher = Publisher(github, cfg, self.redactor)
        self._sleep = sleep
        self._environ = dict(os.environ if environ is None else environ)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:6]}"
        self.state_dir = cfg.resolved_state_dir()
        self._stop = threading.Event()

    # ================================================================= decisions
    def _evaluate(self, issue: Issue) -> Eligible | Refused:
        """Decide whether ``issue`` may run.  Pure with respect to GitHub and the ledger."""
        cfg, labels = self.cfg, self.cfg.labels
        n = issue.number
        if issue.is_pull_request or not issue.is_open:
            return Refused(n, "not_open", ["issue is closed or is a pull request"])
        if not {labels.task, labels.approved} <= issue.labels:
            return Refused(n, "missing_labels", [f"needs {labels.task} and {labels.approved}"])
        if issue.labels & {labels.blocked, labels.done}:
            return Refused(n, "already_handled", ["carries agent-blocked or agent-done"])
        if issue.author not in cfg.trusted_authors:
            return Refused(
                n, "author_not_trusted", [f"{issue.author!r} may not author agent tasks"]
            )
        events = self.github.issue_events(n)
        approval = find_approval(events, labels.approved, issue.labels)
        if approval is None:
            return Refused(n, "no_approval_event", ["cannot find who approved this issue"])
        if approval.actor not in cfg.approvers:
            return Refused(
                n,
                "approver_not_trusted",
                [f"{approval.actor!r} may not approve agent tasks"],
                approval=approval,
            )
        existing = self.ledger.find_by_approval(cfg.repository, n, approval.event_id)
        if existing is not None:
            if existing.status.terminal or existing.status.active:
                return Refused(
                    n,
                    "already_processed",
                    [f"run {existing.run_id[:12]} is {existing.status.value}"],
                    approval=approval,
                )
        elif labels.running in issue.labels:
            return Refused(
                n,
                "running_label",
                ["carries agent-running without a ledger entry"],
                approval=approval,
            )
        last_edit = self.github.issue_last_edited_at(n)
        if edited_after(approval, events, last_edit):
            return Refused(
                n,
                "approval_stale",
                [
                    "the issue was edited at or after the approval; approval applies to the text "
                    "that was approved"
                ],
                notify=True,
                approval=approval,
            )
        try:
            task = parse_task(issue.title, issue.body)
        except TaskValidationError as exc:
            return Refused(n, "invalid_task", exc.errors, notify=True, approval=approval)
        problems = validate_against_config(task.spec, cfg)
        if problems:
            return Refused(n, "task_not_permitted", problems, notify=True, approval=approval)
        return Eligible(issue, task, approval, task.spec.agent or cfg.agent.default)

    # ================================================================= polling
    def poll_once(self, *, execute: bool, only_issue: int | None = None) -> PollReport:
        report = PollReport()
        report.recovered = self.recover(execute=execute)
        try:
            if only_issue is not None:
                issues = [self.github.get_issue(only_issue)]
            else:
                issues = self.github.list_candidate_issues(
                    [self.cfg.labels.task, self.cfg.labels.approved]
                )
        except GitHubError as exc:
            report.errors.append(f"cannot list issues: {exc}")
            return report
        for issue in issues:
            if self._stop.is_set():
                break
            if execute and len(report.runs) >= self.cfg.poll.max_jobs_per_poll:
                break
            report.considered += 1
            try:
                decision = self._evaluate(issue)
            except GitHubError as exc:
                report.errors.append(f"#{issue.number}: {exc}")
                continue
            if isinstance(decision, Refused):
                report.refused.append(decision)
                if execute:
                    self._record_refusal(decision)
                    if decision.notify:
                        self._notify_refusal(decision)
                continue
            if not execute:
                report.previews.append(self.build_preview(decision))
                continue
            run = self._execute(decision)
            if run is not None:
                report.runs.append(run)
        return report

    def serve(self, *, execute: bool, stop: threading.Event | None = None) -> None:
        """Poll forever (until ``stop`` is set); never lets one bad cycle kill the loop."""
        if stop is not None:
            self._stop = stop
        while not self._stop.is_set():
            delay = self.cfg.poll.interval_seconds
            try:
                report = self.poll_once(execute=execute)
                if report.errors:
                    LOG.warning("poll problems: %s", "; ".join(report.errors[:3]))
                    delay = self.cfg.poll.error_backoff_seconds
            except Exception:
                LOG.exception("poll cycle failed")
                delay = self.cfg.poll.error_backoff_seconds
            self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()

    # ================================================================= dry run
    def build_preview(self, el: Eligible) -> Preview:
        """Everything a real run would do, computed without side effects on GitHub or git."""
        import tempfile

        spec = el.task.spec
        policy = build_write_policy(spec, self.cfg)
        budget, timeout_min = self._limits(spec.budget_usd, spec.timeout_minutes)
        check_names = self._check_names(spec)
        bash_allow = self._bash_allowlist(spec, check_names)
        notes: list[str] = []
        try:
            base = self.git.resolve_base()
        except GitError:
            base = f"(origin/{self.cfg.git.base_branch} not fetched yet)"
            notes.append("base branch is not available locally; a real run fetches it first")
        context = load_context(
            self.git.repo_path,
            self.cfg.prompt.context_files,
            self.cfg.prompt.max_context_chars,
            self.redactor,
        )
        prompt = build_prompt(
            issue_number=el.issue.number,
            task=el.task,
            policy=policy,
            context=context,
            bash_allowlist=bash_allow,
        )
        backend = self.backends[el.backend]
        with tempfile.TemporaryDirectory(prefix="agent-bridge-preview-") as tmp:
            request = AgentRequest(
                prompt=prompt,
                cwd=Path("<worktree>"),
                run_dir=Path(tmp),
                scope=spec.scope,
                timeout_s=timeout_min * 60,
                budget_usd=budget,
                write_allow_sets=policy.allow_sets,
                write_deny=policy.deny,
                bash_allowlist=tuple(bash_allow),
                run_id="preview",
            )
            binary = backend.resolve_binary()
            if binary is None:
                notes.append(f"agent binary {backend.cfg.binary!r} was not found on PATH")
            command = backend.build_argv(request, binary or backend.cfg.binary)
        return Preview(
            issue_number=el.issue.number,
            title=el.task.title,
            backend=el.backend,
            scope=spec.scope,
            approver=el.approval.actor,
            approval_event_id=el.approval.event_id,
            revision=el.task.revision,
            budget_usd=budget,
            timeout_minutes=timeout_min,
            branch=f"{self.cfg.git.branch_prefix}issue-{el.issue.number}-<run id>",
            base=base,
            write_allow_sets=[list(s) for s in policy.allow_sets],
            write_deny=list(policy.deny),
            checks=check_names,
            command=command,
            prompt=prompt,
            prompt_chars=len(prompt),
            notes=notes,
        )

    # ================================================================= helpers
    def _limits(self, budget: float | None, timeout_minutes: float | None) -> tuple[float, float]:
        limits = self.cfg.limits
        return (
            min(budget or limits.budget_usd, limits.max_budget_usd),
            min(timeout_minutes or limits.timeout_minutes, limits.max_timeout_minutes),
        )

    def _check_names(self, spec: Any) -> list[str]:
        return list(dict.fromkeys([*self.cfg.default_checks, *spec.checks]))

    def _bash_allowlist(self, spec: Any, check_names: list[str]) -> list[str]:
        if spec.scope != "code":
            return []
        return [shlex.join(self.cfg.checks[name].argv) for name in check_names]

    # ================================================================= refusal notices
    def _record_refusal(self, refusal: Refused) -> None:
        """Log each distinct refusal once (not on every poll)."""
        approval_id = refusal.approval.event_id if refusal.approval else 0
        key = f"seen-{refusal.issue_number}-{approval_id}-{refusal.code}"
        if self.ledger.notice_once(key):
            self.ledger.record_event(
                f"refused.{refusal.code}",
                repo=self.cfg.repository,
                issue_number=refusal.issue_number,
                detail={"messages": refusal.messages[:5]},
            )
            LOG.info("refused %s", refusal.render())

    def _notify_refusal(self, refusal: Refused) -> None:
        approval_id = refusal.approval.event_id if refusal.approval else 0
        key = f"refusal-{refusal.issue_number}-{approval_id}-{refusal.code}"
        if not self.ledger.notice_once(key):
            return
        body = render_refusal(refusal.code, refusal.messages, key=key)
        try:
            self.publisher.comment_once(refusal.issue_number, key, body)
        except GitHubError as exc:
            self.ledger.forget_notice(key)
            LOG.warning("could not post refusal for #%s: %s", refusal.issue_number, exc)

    # ================================================================= execution
    def _execute(self, el: Eligible) -> RunRecord | None:
        cfg = self.cfg
        lease = self.ledger.lease(
            LeaseRequest(
                repo=cfg.repository,
                issue_number=el.issue.number,
                issue_title=el.task.title,
                revision=el.task.revision,
                spec_hash=el.task.spec_hash,
                approval_event_id=el.approval.event_id,
                approved_by=el.approval.actor,
                approved_at=el.approval.created_at,
                backend=el.backend,
                worker_id=self.worker_id,
                worker_pid=os.getpid(),
                ttl_seconds=cfg.limits.lease_ttl_seconds,
                max_attempts_per_issue=cfg.limits.max_attempts_per_issue,
                dedupe_by_content=cfg.dedupe_by_content,
            )
        )
        if not lease.granted or lease.run is None:
            self._handle_denied(el, lease.reason or "denied", lease.other)
            return None
        run = lease.run
        ctx = _Ctx(run=run, el=el, run_dir=self.state_dir / "runs" / run.run_id)
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(
            "run %s started for issue #%s with %s", run.run_id[:12], run.issue_number, run.backend
        )
        try:
            self._check_cancel(run)
            self._begin(ctx)
            self._agent_phase(ctx)
            self._verify_and_commit(ctx)
            self._publish(self.ledger.get(run.run_id) or run, ctx.payload())
        except _Stop as stop:
            self._conclude(ctx.run.run_id, stop, ctx.payload())
        except _Requeue as why:
            self._requeue(run.run_id, str(why), self._retry_limit(run.run_id), ctx)
        except _Deferred as why:
            LOG.warning("publishing deferred for run %s: %s", run.run_id[:12], why)
        except Exception as exc:
            LOG.exception("run %s failed unexpectedly", run.run_id[:12])
            self._conclude(
                run.run_id,
                _Stop(RunStatus.BLOCKED, "internal_error", f"{type(exc).__name__}: {exc}"),
                ctx.payload(),
            )
        return self.ledger.get(run.run_id)

    def _handle_denied(self, el: Eligible, reason: str, other: RunRecord | None) -> None:
        detail = {"reason": reason, "other": other.run_id if other else None}
        self.ledger.record_event(
            "lease.denied",
            repo=self.cfg.repository,
            issue_number=el.issue.number,
            detail=detail,
        )
        LOG.info("lease denied for #%s: %s", el.issue.number, reason)
        if reason in {"attempt-limit", "duplicate-content"}:
            refusal = Refused(
                el.issue.number,
                "attempt_limit" if reason == "attempt-limit" else "duplicate_task",
                [
                    f"this issue has used its {self.cfg.limits.max_attempts_per_issue} runs; "
                    "open a new issue"
                    if reason == "attempt-limit"
                    else "identical task content already ran or is running in "
                    f"#{other.issue_number if other else '?'}"
                ],
                notify=True,
                approval=el.approval,
            )
            self._notify_refusal(refusal)

    def _check_cancel(self, run: RunRecord) -> None:
        """Honour a local ``agent-bridge cancel`` at a phase boundary, not only mid-agent."""
        if self.ledger.is_cancel_requested(run.run_id):
            raise _Stop(
                RunStatus.CANCELLED,
                "cancelled",
                "cancel: cancelled locally with `agent-bridge cancel`",
            )

    # ----------------------------------------------------------------- begin
    def _gate(self, run: RunRecord, issue: Issue) -> None:
        """Re-check, against fresh GitHub state, that the run is still authorised."""
        if not issue.is_open:
            raise _Stop(RunStatus.CANCELLED, "issue_closed", "the issue was closed")
        if self.cfg.labels.approved not in issue.labels:
            raise _Stop(RunStatus.CANCELLED, "approval_withdrawn", "the approval label was removed")
        if compute_revision(issue.title, issue.body) != run.revision:
            raise _Stop(
                RunStatus.BLOCKED,
                "task_changed",
                "the issue text changed after approval",
                hints=["Review the edit, then remove and re-apply `agent-approved`."],
            )

    def _begin(self, ctx: _Ctx) -> None:
        run = ctx.run
        try:
            issue = self.github.get_issue(run.issue_number)
        except GitHubUnavailable as exc:
            raise _Requeue(f"GitHub unavailable at start: {exc}") from exc
        except GitHubNotFound as exc:
            raise _Stop(
                RunStatus.CANCELLED, "issue_missing", f"the issue no longer exists: {exc}"
            ) from exc
        self._gate(run, issue)
        spec = ctx.el.task.spec
        budget, timeout_min = self._limits(spec.budget_usd, spec.timeout_minutes)
        try:
            self.publisher.mark_running(run.issue_number)
            self.publisher.comment_once(
                run.issue_number,
                f"run-{run.run_id}-started",
                render_started(
                    run, budget_usd=budget, timeout_minutes=timeout_min, scope=spec.scope
                ),
            )
        except GitHubPermissionError as exc:
            raise _Stop(
                RunStatus.BLOCKED,
                "github_permission",
                f"cannot update labels/comments: {exc}",
                hints=["Grant the bridge token Issues: read/write on this repository."],
            ) from exc
        except GitHubUnavailable as exc:
            raise _Requeue(f"GitHub unavailable at start: {exc}") from exc
        self._worktree(ctx)

    def _worktree(self, ctx: _Ctx) -> None:
        run = ctx.run
        try:
            if self.cfg.git.fetch_before_run:
                self.git.fetch()
            base_sha = self.git.resolve_base()
            branch = f"{self.cfg.git.branch_prefix}issue-{run.issue_number}-{run.run_id[:8]}"
            path = self.state_dir / "worktrees" / run.run_id
            self.git.remove_worktree(path, branch)
            ctx.pointer = self.git.create_worktree(path, branch, base_sha)
        except GitError as exc:
            if _network_git(exc):
                raise _Requeue(f"git could not reach the remote: {exc}") from exc
            raise _Stop(
                RunStatus.BLOCKED,
                "git_failed",
                str(exc),
                hints=["Check the local clone, the remote and the base branch name."],
            ) from exc
        ctx.worktree = path
        self.ledger.set_phase(
            run.run_id, RunPhase.WORKTREE, branch=branch, worktree_path=str(path), base_sha=base_sha
        )
        (ctx.run_dir / "worktree.pointer").write_text(ctx.pointer or "", encoding="utf-8")
        ctx.run = self.ledger.get(run.run_id) or run

    # ----------------------------------------------------------------- agent
    def _abort_check(self, ctx: _Ctx) -> Callable[[], str | None]:
        run = ctx.run
        ttl = self.cfg.limits.lease_ttl_seconds

        def check() -> str | None:
            self.ledger.heartbeat(run.run_id, ttl)
            if self._stop.is_set():
                return "interrupted: the bridge is shutting down"
            if self.ledger.is_cancel_requested(run.run_id):
                return "cancel: cancelled locally with `agent-bridge cancel`"
            try:
                issue = self.github.get_issue(run.issue_number)
            except GitHubError as exc:
                LOG.warning("cancel check could not reach GitHub: %s", exc)
                return None
            if not issue.is_open:
                return "cancel: the issue was closed"
            if self.cfg.labels.approved not in issue.labels:
                return "cancel: the approval label was removed"
            if compute_revision(issue.title, issue.body) != run.revision:
                return "changed: the issue text was edited after approval"
            return None

        return check

    def _agent_phase(self, ctx: _Ctx) -> None:
        run, el = ctx.run, ctx.el
        assert ctx.worktree is not None
        self.ledger.set_phase(run.run_id, RunPhase.AGENT)
        spec = el.task.spec
        policy = build_write_policy(spec, self.cfg)
        budget, timeout_min = self._limits(spec.budget_usd, spec.timeout_minutes)
        check_names = self._check_names(spec)
        bash_allow = self._bash_allowlist(spec, check_names)
        context = load_context(
            ctx.worktree,
            self.cfg.prompt.context_files,
            self.cfg.prompt.max_context_chars,
            self.redactor,
        )
        prompt = build_prompt(
            issue_number=run.issue_number,
            task=el.task,
            policy=policy,
            context=context,
            bash_allowlist=bash_allow,
        )
        (ctx.run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        backend = self.backends[el.backend]
        request = AgentRequest(
            prompt=prompt,
            cwd=ctx.worktree,
            run_dir=ctx.run_dir / "agent",
            scope=spec.scope,
            timeout_s=timeout_min * 60,
            budget_usd=budget,
            write_allow_sets=policy.allow_sets,
            write_deny=policy.deny,
            bash_allowlist=tuple(bash_allow),
            run_id=run.run_id,
            abort_check=self._abort_check(ctx),
            abort_poll_s=self.cfg.agent.cancel_poll_seconds,
            on_start=lambda pid, ticks: self.ledger.update(
                run.run_id, agent_pid=pid, agent_start_ticks=ticks
            ),
        )
        result = backend.run(request)
        ctx.agent = result
        self.ledger.update(run.run_id, result_json=ctx.payload())
        self._interpret(result)

    def _interpret(self, result: AgentResult) -> None:
        if result.status is AgentStatus.SUCCEEDED:
            return
        detail = result.error or result.status.value
        kind = result.failure_kind.value if result.failure_kind else result.status.value
        if result.status is AgentStatus.CANCELLED:
            raise self._abort_outcome(detail)
        if result.status is AgentStatus.UNAVAILABLE:
            raise _Stop(RunStatus.BLOCKED, "agent_unavailable", detail, hints=HINTS["unavailable"])
        raise _Stop(
            RunStatus.BLOCKED,
            f"agent_{kind}",
            detail,
            hints=HINTS.get(kind, []),
        )

    @staticmethod
    def _abort_outcome(reason: str) -> _Requeue | _Stop:
        """Map the reason an abort check gave to the way the run ends."""
        if reason.startswith("interrupted:"):
            return _Requeue(reason)
        if reason.startswith("changed:"):
            return _Stop(
                RunStatus.BLOCKED,
                "task_changed",
                reason,
                hints=["Review the edit, then remove and re-apply `agent-approved`."],
            )
        return _Stop(RunStatus.CANCELLED, "cancelled", reason)

    # ----------------------------------------------------------------- verify & commit
    def _verify_and_commit(self, ctx: _Ctx) -> None:
        run, el = ctx.run, ctx.el
        assert ctx.worktree is not None and ctx.agent is not None
        path = ctx.worktree
        run = self.ledger.get(run.run_id) or run
        ctx.run = run
        self.ledger.set_phase(run.run_id, RunPhase.VERIFY)
        self._check_cancel(run)
        base_sha = run.base_sha or ""
        if worktree_pointer(path) != ctx.pointer:
            raise _Stop(
                RunStatus.BLOCKED, "worktree_tampered", "the worktree's .git pointer changed"
            )
        if self.git.worktree_head(path) != base_sha:
            raise _Stop(RunStatus.BLOCKED, "worktree_tampered", "the agent moved the worktree HEAD")
        report = ctx.agent.report
        if report is not None and report.status == "blocked":
            raise _Stop(
                RunStatus.BLOCKED,
                "agent_blocked",
                report.summary,
                hints=["The agent decided it could not do this safely; read its summary."],
            )

        policy = build_write_policy(el.task.spec, self.cfg)
        self.git.stage_all(path)
        staged = self.git.staged_changes(
            path, base_sha, max_diff_bytes=self.cfg.limits.max_diff_bytes
        )
        if not staged.changes:
            raise _Stop(
                RunStatus.BLOCKED,
                "no_changes",
                "the agent finished without changing any file",
                hints=["Rephrase the objective or check the agent summary below."],
            )
        violations = evaluate_changes(
            staged.changes, policy, self.cfg.limits, diff_bytes=staged.diff_bytes
        )
        if staged.truncated and not any(v.code == "diff_too_big" for v in violations):
            violations.append(Violation("diff_too_big", "diff exceeds the inspection window"))
        for finding in self.redactor.scan_added_lines(staged.diff_text):
            violations.append(
                Violation("secret", f"possible {finding.kind} on line {finding.line}", finding.path)
            )
        ctx.files = sorted(change.path for change in staged.changes)
        ctx.added = sum(c.added for c in staged.changes)
        ctx.deleted = sum(c.deleted for c in staged.changes)
        if violations:
            ctx.violations = [v.render() for v in violations]
            raise _Stop(
                RunStatus.BLOCKED,
                "policy_violation",
                f"{len(violations)} change(s) are not allowed",
                hints=["Protected or out-of-scope changes need a human; nothing was pushed."],
                violations=ctx.violations,
            )

        names = self._check_names(el.task.spec)
        if names:
            tree_before = self.git.write_tree(path)
            try:
                ctx.checks = run_checks(
                    names,
                    self.cfg.checks,
                    path,
                    runner=self.runner,
                    redactor=self.redactor,
                    log_dir=ctx.run_dir / "checks",
                    environ=self._environ,
                    command_prefix=self.cfg.agent.command_prefix,
                    abort_check=self._abort_check(ctx),
                    abort_poll_s=self.cfg.agent.cancel_poll_seconds,
                )
            except CheckAborted as exc:
                raise self._abort_outcome(exc.reason) from exc
            failed = [c for c in ctx.checks if not c.passed]
            if failed:
                raise _Stop(
                    RunStatus.BLOCKED,
                    "checks_failed",
                    f"check {failed[0].name!r} failed",
                    hints=["Fix the failing check (see the run logs) and re-approve."],
                )
            self.git.stage_all(path)
            if self.git.write_tree(path) != tree_before:
                raise _Stop(
                    RunStatus.BLOCKED,
                    "checks_modified_tree",
                    "running the checks changed tracked or untracked files",
                )

        self.ledger.set_phase(run.run_id, RunPhase.COMMIT)
        message = self._commit_message(ctx)
        try:
            sha = self.git.commit(path, message)
        except GitError as exc:
            raise _Stop(RunStatus.BLOCKED, "commit_failed", str(exc)) from exc
        payload = ctx.payload()
        self.ledger.set_phase(
            run.run_id,
            RunPhase.PUSH,
            status=RunStatus.PUBLISHING,
            commit_sha=sha,
            result_json=payload,
        )
        ctx.run = self.ledger.get(run.run_id) or run

    def _commit_message(self, ctx: _Ctx) -> str:
        run = ctx.run
        # No agent text goes into the commit.  This message is what lands on the default branch
        # once a human merges the PR, and a closing keyword, an @mention or a Co-authored-by line
        # in it would take effect there.  The agent's summary stays in the PR body and the issue
        # comment, where it is redacted and fenced.
        title = " ".join(strip_control_chars(ctx.el.task.title).split())
        subject = f"agent(#{run.issue_number}): {truncate(title, 60, '...')}"
        trailers = "\n".join(
            [
                f"{self.cfg.pull_request.closing_keyword} #{run.issue_number}",
                f"Agent-Bridge-Run: {run.run_id}",
                f"Agent-Backend: {run.backend}",
                f"Approved-By: {run.approved_by}",
            ]
        )
        return f"{subject}\n\n{trailers}\n"

    # ----------------------------------------------------------------- publish
    def _final_gate(self, run: RunRecord) -> None:
        issue = self.github.get_issue(run.issue_number)
        self._gate(run, issue)
        self._check_cancel(run)
        approval = find_approval(
            self.github.issue_events(run.issue_number), self.cfg.labels.approved, issue.labels
        )
        if approval is None or approval.event_id != run.approval_event_id:
            raise _Stop(
                RunStatus.CANCELLED,
                "approval_changed",
                "the approval was withdrawn or replaced before publication",
            )

    def _publish(self, run: RunRecord, payload: dict[str, Any]) -> None:
        """Push, open the draft PR, tell the issue.  Safe to call again after any failure."""
        delays = list(self.cfg.limits.publish_retry_delays)
        attempt = 0
        while True:
            try:
                self._publish_once(self.ledger.get(run.run_id) or run, payload)
                return
            except GitHubPermissionError as exc:
                raise _Stop(
                    RunStatus.BLOCKED,
                    "github_permission",
                    str(exc),
                    hints=[
                        "The bridge token needs Contents and Pull requests (read/write) and "
                        "Issues (read/write) on this repository."
                    ],
                ) from exc
            except GitHubUnavailable as exc:
                problem = str(exc)
            except GitHubError as exc:
                raise _Stop(RunStatus.BLOCKED, "github_error", str(exc)) from exc
            except GitError as exc:
                if not _network_git(exc):
                    raise _Stop(
                        RunStatus.BLOCKED,
                        "push_failed",
                        str(exc),
                        hints=["Check the remote and the bridge's push credentials."],
                    ) from exc
                problem = str(exc)
            if attempt >= len(delays):
                raise _Deferred(problem)
            self.ledger.heartbeat(run.run_id, self.cfg.limits.lease_ttl_seconds)
            LOG.warning("publish attempt %d failed (%s); retrying", attempt + 1, problem[:120])
            self._sleep(delays[attempt])
            attempt += 1

    def _publish_once(self, run: RunRecord, payload: dict[str, Any]) -> None:
        self._final_gate(run)
        if run.phase is RunPhase.PUSH:
            if not run.branch:
                raise _Stop(RunStatus.BLOCKED, "internal_error", "run has no branch")
            self.git.push(run.branch)
            self.ledger.set_phase(run.run_id, RunPhase.PULL_REQUEST)
            run = self.ledger.get(run.run_id) or run
        if run.phase is RunPhase.PULL_REQUEST:
            assert run.branch is not None
            pr = self.github.find_pull_request(run.branch)
            if pr is None:
                body = render_pull_request_body(
                    run,
                    payload,
                    issue_number=run.issue_number,
                    keyword=self.cfg.pull_request.closing_keyword,
                    redactor=self.redactor,
                )
                title = truncate(
                    f"{self.cfg.pull_request.title_prefix}"
                    f"{' '.join(strip_control_chars(run.issue_title).split())}",
                    120,
                    "...",
                )
                pr = self.github.create_pull_request(
                    title=title, head=run.branch, base=self.cfg.git.base_branch, body=body
                )
            if pr.number and self.cfg.pull_request.labels:
                self.github.add_labels(pr.number, self.cfg.pull_request.labels)
            self.ledger.set_phase(
                run.run_id, RunPhase.NOTIFY, pr_number=pr.number, pr_url=pr.html_url
            )
            run = self.ledger.get(run.run_id) or run
        if run.phase is RunPhase.NOTIFY:
            self.publisher.comment_once(
                run.issue_number, f"run-{run.run_id}-done", render_done(run, payload, self.redactor)
            )
            self.publisher.mark_terminal(run.issue_number, RunStatus.DONE)
            payload = {**payload, "notified": True}
            if self.ledger.finish(run.run_id, RunStatus.DONE, result=payload):
                self._cleanup(run, RunStatus.DONE)
            else:
                LOG.warning(
                    "run %s was concluded elsewhere before it could be marked done; "
                    "leaving its ledger row and worktree alone",
                    run.run_id[:12],
                )

    # ----------------------------------------------------------------- conclusion
    def _conclude(self, run_id: str, stop: _Stop, payload: dict[str, Any]) -> bool:
        """End a run in ``stop.status`` and report it; False when another process already did.

        Terminal states are final, so a refused :meth:`Ledger.finish` means the run is not ours
        to report or to clean up after.
        """
        payload = {**payload, "violations": stop.violations or payload.get("violations", [])}
        if not self.ledger.finish(
            run_id,
            stop.status,
            failure_kind=stop.kind,
            failure_detail=stop.detail,
            result=payload,
        ):
            LOG.warning(
                "run %s was already concluded elsewhere; not reporting %s/%s",
                run_id[:12],
                stop.status.value,
                stop.kind,
            )
            return False
        run = self.ledger.get(run_id)
        if run is not None:
            self._notify_terminal(run, stop, payload)
            self._cleanup(run, stop.status)
        return True

    def _notify_terminal(self, run: RunRecord, stop: _Stop, payload: dict[str, Any]) -> None:
        try:
            if stop.status is RunStatus.CANCELLED:
                body = render_cancelled(run, stop.detail)
                self.publisher.comment_once(run.issue_number, f"run-{run.run_id}-cancelled", body)
            else:
                body = render_blocked(
                    run,
                    kind=stop.kind,
                    detail=stop.detail,
                    hints=stop.hints,
                    payload=payload,
                    redactor=self.redactor,
                )
                self.publisher.comment_once(run.issue_number, f"run-{run.run_id}-blocked", body)
            self.publisher.mark_terminal(run.issue_number, stop.status)
            self.ledger.update(run.run_id, result_json={**payload, "notified": True})
        except GitHubError as exc:
            self.ledger.record_event(
                "notify.failed",
                repo=run.repo,
                issue_number=run.issue_number,
                run_id=run.run_id,
                detail={"error": str(exc)[:200]},
            )
            LOG.warning("could not report run %s on GitHub: %s", run.run_id[:12], exc)

    def _retry_limit(self, run_id: str) -> int:
        """Automatic retries allowed at the phase the run was in when it stopped."""
        run = self.ledger.get(run_id)
        limits = self.cfg.limits
        if run is not None and run.phase in (RunPhase.LEASED, RunPhase.WORKTREE):
            return limits.max_start_retries
        return limits.auto_retry_interrupted

    def _requeue(self, run_id: str, reason: str, limit: int, ctx: _Ctx | None = None) -> None:
        run = self.ledger.get(run_id)
        if run is None or run.status is not RunStatus.RUNNING:
            return  # finished, cancelled or already requeued elsewhere: nothing to interrupt
        self._kill_agent(run)
        self._cleanup(run, RunStatus.BLOCKED, force=True)
        if self.ledger.requeue_interrupted(run_id, limit):
            try:
                self.publisher.clear_running(run.issue_number)
            except GitHubError:
                pass
            LOG.warning("run %s requeued: %s", run_id[:12], reason)
            return
        stop = _Stop(
            RunStatus.BLOCKED,
            "interrupted",
            f"{reason} (no automatic retries left)",
            hints=["Re-approve to try again."],
        )
        self._conclude(run_id, stop, ctx.payload() if ctx else (run.result or {}))

    # ----------------------------------------------------------------- cleanup
    def _kill_agent(self, run: RunRecord) -> None:
        if run.agent_pid and is_same_process(run.agent_pid, run.agent_start_ticks):
            LOG.warning("terminating orphaned agent process group %s", run.agent_pid)
            kill_process_group(run.agent_pid, grace_seconds=self.cfg.agent.kill_grace_seconds)

    def _cleanup(self, run: RunRecord, status: RunStatus, *, force: bool = False) -> None:
        keep = self.cfg.git.keep_worktrees
        if not force:
            if keep == "always" or (keep == "on-failure" and status is not RunStatus.DONE):
                return
        if not run.worktree_path:
            return
        try:
            self.git.remove_worktree(Path(run.worktree_path), run.branch)
        except GitError as exc:
            LOG.warning("worktree cleanup failed for run %s: %s", run.run_id[:12], exc)

    # ================================================================= recovery
    def recover(self, *, execute: bool) -> list[str]:
        """Resolve runs whose worker disappeared; returns human-readable actions."""
        actions: list[str] = []
        for run in self.ledger.stale_runs():
            where = (
                f"run {run.run_id[:12]} (issue #{run.issue_number}, "
                f"{run.status.value}/{run.phase.value})"
            )
            if not execute:
                actions.append(f"would recover {where}")
                continue
            self._kill_agent(run)
            if run.status is RunStatus.RUNNING:
                actions.append(f"interrupted {where}")
                self._requeue(
                    run.run_id,
                    "the worker stopped while the agent was running",
                    self._retry_limit(run.run_id),
                )
            elif run.status is RunStatus.PUBLISHING:
                actions.append(self._resume_publish(run, where))
        if execute:
            actions += self._reconcile_unreported()
        return actions

    def _resume_publish(self, run: RunRecord, where: str) -> str:
        if run.publish_attempts >= self.cfg.limits.max_publish_attempts:
            gave_up = self._conclude(
                run.run_id,
                _Stop(
                    RunStatus.BLOCKED,
                    "publish_failed",
                    f"publication did not complete after {run.publish_attempts} attempts",
                    hints=["The commit is kept locally; re-approve after fixing connectivity."],
                ),
                run.result or {},
            )
            if not gave_up:
                return f"abandoned {where}: finished elsewhere"
            return f"gave up publishing {where}"
        if not self.ledger.claim_publishing(
            run.run_id, self.worker_id, os.getpid(), self.cfg.limits.lease_ttl_seconds
        ):
            return f"could not claim {where}"
        payload = run.result or {}
        try:
            self._publish(run, payload)
            return f"resumed publishing {where}"
        except _Stop as stop:
            if self._conclude(run.run_id, stop, payload):
                return f"publishing {where} ended {stop.status.value}"
            return f"abandoned {where}: finished elsewhere"
        except _Deferred as why:
            return f"publishing {where} still deferred: {str(why)[:80]}"
        except RunStateError as exc:
            LOG.warning("publishing run %s stopped: %s", run.run_id[:12], exc)
            return f"abandoned {where}: {exc}"

    def _reconcile_unreported(self) -> list[str]:
        """Re-send labels/comments for finished runs whose GitHub update failed earlier."""
        actions: list[str] = []
        for run in self.ledger.list_runs(limit=20):
            if not run.status.terminal or (run.result or {}).get("notified"):
                continue
            payload = run.result or {}
            stop = _Stop(
                run.status,
                run.failure_kind or run.status.value,
                run.failure_detail or "",
            )
            if run.status is RunStatus.DONE:
                try:
                    self.publisher.comment_once(
                        run.issue_number,
                        f"run-{run.run_id}-done",
                        render_done(run, payload, self.redactor),
                    )
                    self.publisher.mark_terminal(run.issue_number, RunStatus.DONE)
                    self.ledger.update(run.run_id, result_json={**payload, "notified": True})
                    actions.append(f"reported run {run.run_id[:12]}")
                except GitHubError:
                    continue
            else:
                self._notify_terminal(run, stop, payload)
                actions.append(f"reported run {run.run_id[:12]}")
        return actions
