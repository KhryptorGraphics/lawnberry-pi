"""Worker life cycle against the fake GitHub, a real git origin/clone and the fake agent CLIs.

These tests cover what happens *around* a run rather than inside it: idempotence across polls and
restarts, leasing and the concurrency limit, duplicate and re-approved issues, cancellation,
edits made while the agent works, dry run, crash recovery, worktree hygiene, base-branch
movement and the serve loop.  Every test runs once per backend (``omp`` and ``claude``).

Policy violations, injection, secrets, network loss, permission denial, agent failures and
checks belong to the security and failure suites and are deliberately not repeated here.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.agent_fixtures import docs_edit_steps, pid_is_alive
from bridgefakes.rig import REPO, Rig, publisher_for, run_git, task_body

from tools.agent_bridge.agents import build_backends
from tools.agent_bridge.errors import BridgeError
from tools.agent_bridge.github import find_approval
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.ledger import LeaseRequest, Ledger, RunRecord
from tools.agent_bridge.models import RunPhase, RunStatus
from tools.agent_bridge.process import proc_start_ticks
from tools.agent_bridge.publisher import ReadOnlyGitHub, render_started
from tools.agent_bridge.schema import parse_task
from tools.agent_bridge.worker import Preview, Worker

# An agent scenario that is only ever left by being killed.
HANG = [{"op": "sleep", "seconds": 60}]
# An agent that finishes without touching a file: the run ends BLOCKED ``no_changes``.
NO_CHANGES = [{"op": "final", "text": "Nothing needed changing."}]

STARTED_HEADING = "Agent run started"
DONE_HEADING = "Draft pull request ready for review"
BLOCKED_HEADING = "Agent run blocked"
CANCELLED_HEADING = "Agent run cancelled"
REFUSAL_HEADING = "Agent bridge cannot run this task"

# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def wait_until(
    predicate: Callable[[], Any], *, timeout: float = 20.0, what: str = "the condition"
) -> Any:
    """Poll ``predicate`` until it returns something truthy (returned); fail after ``timeout``."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:g}s waiting for {what}")
        time.sleep(0.02)


def describe(run: RunRecord) -> str:
    return f"{run.status.value}/{run.phase.value} {run.failure_kind}: {run.failure_detail}"


def agent_starts(rig: Rig) -> int:
    """How many times the fake agent CLI was started (tolerates a half-written log line)."""
    try:
        return len(rig.invocations())
    except ValueError:
        return 0


def runs_of(rig: Rig, number: int) -> list[RunRecord]:
    return rig.ledger.runs_for_issue(REPO, number)


def assert_reported(
    actions: list[str], verb: str, run: RunRecord, *, names_issue: bool = True
) -> None:
    """``actions`` is one line that starts with ``verb`` and names the run (and its issue)."""
    assert len(actions) == 1, actions
    assert actions[0].startswith(verb), actions
    assert run.run_id[:12] in actions[0], actions
    if names_issue:
        assert f"#{run.issue_number}" in actions[0], actions


def current(rig: Rig, run_id: str) -> RunRecord:
    """The run's row as the ledger holds it now."""
    run = rig.ledger.get(run_id)
    assert run is not None, f"no run {run_id}"
    return run


def only_run(rig: Rig, number: int) -> RunRecord:
    runs = runs_of(rig, number)
    assert len(runs) == 1, [describe(r) for r in runs]
    return runs[0]


def comments_with(rig: Rig, number: int, text: str) -> list[str]:
    return [body for body in rig.comments(number) if text in body]


def add_approved_issue(rig: Rig, topic: str, *, number: int | None = None) -> int:
    """An approved docs task whose content is unique per ``topic`` (so it is never a duplicate)."""
    body = task_body(objective=f"Document the {topic} topic in docs/guide.md.")
    issue = rig.fake.add_issue(
        title=f"Document {topic}",
        body=body,
        author="owner",
        labels=["agent-task"],
        number=number,
    )
    rig.fake.label_issue(issue, "agent-approved", actor="owner")
    return issue


def local_agent_branches(rig: Rig) -> list[str]:
    out = rig.git_env.git("for-each-ref", "--format=%(refname:short)", "refs/heads/agent")
    return out.split()


def registered_worktrees(rig: Rig) -> list[Path]:
    root = rig.cfg.resolved_state_dir() / "worktrees"
    return [path for path in rig.git.list_worktree_paths() if root in path.parents]


def expire_leases(rig: Rig) -> None:
    rig.clock.advance(rig.cfg.limits.lease_ttl_seconds + 1)


def configure_git(rig: Rig, **fields: Any) -> None:
    """Change ``git.*`` settings of a built rig.

    ``make_rig(git=...)`` is the repository fixture, so these cannot go through the config
    overrides; the worker and its GitOps share ``rig.cfg.git``, so assigning here reaches both.
    """
    for name, value in fields.items():
        setattr(rig.cfg.git, name, value)


def table_counts(rig: Rig) -> dict[str, int]:
    """Row counts of the ledger tables, read through a separate connection."""
    cx = sqlite3.connect(rig.ledger.path)
    try:
        return {
            table: cx.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("runs", "events", "notices")
        }
    finally:
        cx.close()


def world_state(rig: Rig) -> dict[str, Any]:
    """Everything a dry run must leave alone: git refs, worktrees, state files, ledger tables."""
    state_dir = rig.cfg.resolved_state_dir()
    return {
        "clone_refs": rig.git_env.git("for-each-ref"),
        "origin_refs": rig.git_env.origin_git("for-each-ref"),
        "worktrees": rig.git_env.git("worktree", "list", "--porcelain"),
        "files": sorted(
            str(path.relative_to(state_dir))
            for path in state_dir.rglob("*")
            if path.is_file() and not path.name.startswith("ledger.sqlite")
        ),
        "tables": table_counts(rig),
    }


class Background:
    """Run a callable on a daemon thread; :meth:`result` re-raises whatever it raised."""

    def __init__(self, target: Callable[[], Any]) -> None:
        self.value: Any = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._main, args=(target,), daemon=True)
        self._thread.start()

    def _main(self, target: Callable[[], Any]) -> None:
        try:
            self.value = target()
        except BaseException as exc:  # reported to the test by result()
            self.error = exc

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def result(self, timeout: float = 30.0) -> Any:
        self.join(timeout)
        assert not self.alive, f"background task still running after {timeout:g}s"
        if self.error is not None:
            raise self.error
        return self.value


@pytest.fixture
def background() -> Iterator[Callable[..., Background]]:
    """Start work on a thread; at teardown stop loops, cancel active runs and join."""
    started: list[tuple[Rig, Background, threading.Event | None]] = []

    def start(
        rig: Rig, target: Callable[[], Any], *, stop: threading.Event | None = None
    ) -> Background:
        task = Background(target)
        started.append((rig, task, stop))
        return task

    yield start
    for rig, task, stop in started:
        if stop is not None:
            stop.set()
        with contextlib.suppress(Exception):
            for run in rig.ledger.active_runs():
                rig.ledger.request_cancel(run_id=run.run_id)
        task.join(15)


def wait_for_agent(rig: Rig, *, starts: int = 1) -> RunRecord:
    """Block until the fake agent has started ``starts`` times and its pid is in the ledger."""
    wait_until(lambda: agent_starts(rig) >= starts, what="the agent CLI to start")
    return wait_until(
        lambda: next((r for r in rig.ledger.active_runs() if r.agent_pid), None),
        what="the agent pid to be recorded",
    )


def build_worker(rig: Rig, *, github: Any = None, worker_id: str = "second-worker") -> Worker:
    """A new ledger connection, git wrapper, backends and worker over the rig's state."""
    state_dir = rig.cfg.resolved_state_dir()
    home = str(rig.git_env.home)
    ledger = Ledger(state_dir / "ledger.sqlite", clock=rig.clock)
    git = GitOps(
        rig.git_env.clone,
        rig.cfg.git,
        state_dir=state_dir,
        redactor=rig.redactor,
        environ={"PATH": os.environ.get("PATH", ""), "HOME": home, "LC_ALL": "C"},
    )
    return Worker(
        rig.cfg,
        github=github if github is not None else rig.client,
        ledger=ledger,
        git=git,
        backends=build_backends(rig.cfg.agent, redactor=rig.redactor, environ=rig.agent_environ),
        redactor=rig.redactor,
        sleep=lambda _s: None,
        worker_id=worker_id,
        environ={"PATH": os.environ.get("PATH", ""), "HOME": home},
    )


def restart(rig: Rig, *, worker_id: str = "restarted-worker") -> Worker:
    """Simulate a bridge restart: the old process's ledger connection is gone, a new one starts."""
    rig.ledger.close()
    worker = build_worker(rig, worker_id=worker_id)
    rig.worker, rig.ledger, rig.git = worker, worker.ledger, worker.git
    return worker


@pytest.fixture
def spawn_worker() -> Iterator[Callable[..., Worker]]:
    """Additional workers (a second bridge on the same state); their ledgers close at teardown."""
    built: list[Worker] = []

    def spawn(rig: Rig, **kwargs: Any) -> Worker:
        worker = build_worker(rig, **kwargs)
        built.append(worker)
        return worker

    yield spawn
    for worker in built:
        worker.ledger.close()


def lease_request(rig: Rig, number: int, **overrides: Any) -> LeaseRequest:
    """The LeaseRequest the worker would build for ``number`` right now."""
    issue = rig.client.get_issue(number)
    task = parse_task(issue.title, issue.body)
    approval = find_approval(rig.client.issue_events(number), rig.cfg.labels.approved, issue.labels)
    assert approval is not None, f"issue #{number} is not approved"
    fields: dict[str, Any] = {
        "repo": REPO,
        "issue_number": number,
        "issue_title": task.title,
        "revision": task.revision,
        "spec_hash": task.spec_hash,
        "approval_event_id": approval.event_id,
        "approved_by": approval.actor,
        "approved_at": approval.created_at,
        "backend": rig.backend,
        "worker_id": "other-worker",
        "worker_pid": os.getpid(),
        "ttl_seconds": rig.cfg.limits.lease_ttl_seconds,
        "max_attempts_per_issue": rig.cfg.limits.max_attempts_per_issue,
        "dedupe_by_content": rig.cfg.dedupe_by_content,
    }
    fields.update(overrides)
    return LeaseRequest(**fields)


@pytest.fixture
def orphan() -> Iterator[Callable[[], subprocess.Popen[bytes]]]:
    """Factory for stand-ins of an agent left running by a dead bridge (own session and group)."""
    spawned: list[subprocess.Popen[bytes]] = []

    def spawn() -> subprocess.Popen[bytes]:
        sleeper = shutil.which("sleep")
        command = (
            [sleeper, "300"] if sleeper else [sys.executable, "-c", "import time; time.sleep(300)"]
        )
        proc = subprocess.Popen(
            command,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Reap it the moment it dies so a killed process never lingers as a zombie.
        threading.Thread(target=proc.wait, daemon=True).start()
        spawned.append(proc)
        return proc

    yield spawn
    for proc in spawned:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)


_REAL_TICKS: Any = object()


def crash_mid_agent(
    rig: Rig,
    number: int,
    *,
    orphan: subprocess.Popen[bytes] | None = None,
    start_ticks: Any = _REAL_TICKS,
    phase: RunPhase = RunPhase.AGENT,
) -> RunRecord:
    """Leave ledger, GitHub and git as a worker that died while its agent was running would.

    The run is leased and ``running`` at ``phase`` (``leased``, ``worktree`` or ``agent``); the
    issue carries ``agent-running`` and the "started" comment; the worktree and branch exist; and,
    given an ``orphan``, its pid and start time are recorded as the agent process.
    """
    leased = rig.ledger.lease(lease_request(rig, number, worker_id="crashed-worker"))
    assert leased.granted and leased.run is not None, leased.reason
    run = leased.run
    publisher = publisher_for(rig)
    publisher.mark_running(number)
    publisher.comment_once(
        number,
        f"run-{run.run_id}-started",
        render_started(run, budget_usd=1.0, timeout_minutes=1.0, scope="docs-only"),
    )
    if phase in (RunPhase.WORKTREE, RunPhase.AGENT):
        branch = f"{rig.cfg.git.branch_prefix}issue-{number}-{run.run_id[:8]}"
        path = rig.cfg.resolved_state_dir() / "worktrees" / run.run_id
        base = rig.git.resolve_base()
        rig.git.create_worktree(path, branch, base)
        rig.ledger.set_phase(
            run.run_id, RunPhase.WORKTREE, branch=branch, worktree_path=str(path), base_sha=base
        )
    if phase is RunPhase.AGENT:
        rig.ledger.set_phase(run.run_id, RunPhase.AGENT)
        if orphan is not None:
            ticks = proc_start_ticks(orphan.pid) if start_ticks is _REAL_TICKS else start_ticks
            rig.ledger.update(run.run_id, agent_pid=orphan.pid, agent_start_ticks=ticks)
    crashed = rig.ledger.get(run.run_id)
    assert crashed is not None
    return crashed


class SimulatedCrash(BaseException):
    """The bridge process dying: nothing in the worker catches ``BaseException``."""


def advance_origin(rig: Rig, relpath: str = "docs/upstream.md", text: str = "upstream\n") -> str:
    """Commit to origin/main from a second clone, so the rig's own clone has not seen it yet."""
    other = rig.tmp_path / "other-clone"
    env = rig.git_env.env
    if other.exists():
        run_git(other, "pull", "-q", "--ff-only", "origin", "main", env=env)
    else:
        run_git(rig.tmp_path, "clone", "-q", str(rig.git_env.origin), str(other), env=env)
    target = other / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    run_git(other, "add", "-A", env=env)
    run_git(other, "commit", "-q", "-m", f"upstream change to {relpath}", env=env)
    run_git(other, "push", "-q", "origin", "main", env=env)
    return run_git(other, "rev-parse", "HEAD", env=env)


# ----------------------------------------------------------------------------------------------
# one approval, one run: idempotence, leasing, the concurrency limit
# ----------------------------------------------------------------------------------------------


def test_one_approved_issue_makes_one_run_and_polling_again_does_nothing(rig: Rig) -> None:
    number = rig.new_task_issue()

    report = rig.poll()

    assert [r.issue_number for r in report.runs] == [number]
    run = only_run(rig, number)
    assert run.status is RunStatus.DONE, describe(run)
    assert len(rig.invocations()) == 1

    rig.fake.reset_requests()
    for _ in range(3):
        again = rig.poll()
        assert again.runs == [] and again.previews == []
        assert [r.code for r in again.refused] == ["already_handled"]
    assert rig.fake.writes() == []
    assert len(rig.invocations()) == 1
    assert [r.run_id for r in runs_of(rig, number)] == [run.run_id]
    assert len(rig.fake.pulls) == 1


def test_a_restarted_worker_over_the_same_state_does_nothing(rig: Rig) -> None:
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    assert run.status is RunStatus.DONE, describe(run)

    worker = restart(rig)
    assert worker.ledger is rig.ledger

    rig.fake.reset_requests()
    report = worker.poll_once(execute=True)
    assert report.runs == []
    assert [r.code for r in report.refused] == ["already_handled"]
    assert rig.fake.writes() == []
    assert len(rig.invocations()) == 1
    assert [r.run_id for r in runs_of(rig, number)] == [run.run_id]


def test_the_ledger_alone_blocks_a_rerun_when_the_done_label_disappears(rig: Rig) -> None:
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    assert run.status is RunStatus.DONE, describe(run)
    # A human (or a failed GitHub update) drops the label; the approval has already been spent.
    rig.fake.unlabel_issue(number, "agent-done")

    worker = restart(rig)
    rig.fake.reset_requests()
    report = worker.poll_once(execute=True)

    assert report.runs == []
    assert [r.code for r in report.refused] == ["already_processed"]
    assert rig.fake.writes() == []
    assert len(rig.invocations()) == 1
    assert len(runs_of(rig, number)) == 1


def test_default_poll_starts_one_run_per_poll_oldest_issue_first(rig: Rig) -> None:
    # Issue numbers are deliberately not in creation order: "oldest first" is by creation time.
    numbers = [
        add_approved_issue(rig, "alpha", number=30),
        add_approved_issue(rig, "beta", number=12),
        add_approved_issue(rig, "gamma", number=21),
    ]

    for index, expected in enumerate(numbers):
        report = rig.poll()
        assert [r.issue_number for r in report.runs] == [expected]
        assert report.runs[0].status is RunStatus.DONE, describe(report.runs[0])
        assert len(rig.invocations()) == index + 1
        for waiting in numbers[index + 1 :]:
            assert runs_of(rig, waiting) == []
            assert rig.comments(waiting) == []
            assert rig.fake.issue_labels(waiting) == ["agent-approved", "agent-task"]

    final = rig.poll()
    assert final.runs == []
    assert len(rig.invocations()) == 3


def test_max_jobs_per_poll_bounds_the_runs_one_poll_may_start(make_rig_for: Any) -> None:
    rig: Rig = make_rig_for(poll={"max_jobs_per_poll": 2})
    # A malformed approved issue is refused, and must not use up the poll's quota.
    broken = rig.new_task_issue(title="Not a task", body="Please just do something.")
    numbers = [
        add_approved_issue(rig, "alpha", number=30),
        add_approved_issue(rig, "beta", number=12),
        add_approved_issue(rig, "gamma", number=21),
    ]

    first = rig.poll()
    assert [r.issue_number for r in first.runs] == numbers[:2]
    assert [r.code for r in first.refused] == ["invalid_task"]
    assert runs_of(rig, broken) == []
    assert len(rig.invocations()) == 2
    assert runs_of(rig, numbers[2]) == []

    second = rig.poll()
    assert [r.issue_number for r in second.runs] == [numbers[2]]
    assert len(rig.invocations()) == 3
    assert all(only_run(rig, n).status is RunStatus.DONE for n in numbers)
    assert rig.poll().runs == []


def test_a_second_lease_is_denied_busy_while_a_run_is_active(rig: Rig) -> None:
    first_issue = add_approved_issue(rig, "first")
    second_issue = add_approved_issue(rig, "second")

    first = rig.ledger.lease(lease_request(rig, first_issue))
    assert first.granted and first.run is not None
    assert first.run.status is RunStatus.RUNNING and first.run.phase is RunPhase.LEASED

    denied = rig.ledger.lease(lease_request(rig, second_issue))
    assert not denied.granted and denied.run is None
    assert denied.reason == "busy"
    assert denied.other is not None and denied.other.run_id == first.run.run_id
    assert runs_of(rig, second_issue) == []

    # A run that is publishing still holds the single slot.
    rig.ledger.set_phase(first.run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING)
    assert rig.ledger.lease(lease_request(rig, second_issue)).reason == "busy"

    # ... and an expired lease does too, until recovery has dealt with the run.
    expire_leases(rig)
    assert rig.ledger.lease(lease_request(rig, second_issue)).reason == "busy"

    rig.ledger.finish(first.run.run_id, RunStatus.CANCELLED, failure_kind="cancelled")
    granted = rig.ledger.lease(lease_request(rig, second_issue))
    assert granted.granted and granted.run is not None
    assert granted.run.issue_number == second_issue


def test_one_approval_can_never_be_leased_twice(rig: Rig) -> None:
    number = add_approved_issue(rig, "once")
    request = lease_request(rig, number)

    first = rig.ledger.lease(request)
    assert first.granted and first.run is not None

    again = rig.ledger.lease(request)
    assert (again.granted, again.reason) == (False, "duplicate-active")

    expire_leases(rig)
    assert rig.ledger.lease(request).reason == "recovery-pending"

    rig.ledger.finish(first.run.run_id, RunStatus.DONE, result={"notified": True})
    spent = rig.ledger.lease(request)
    assert (spent.granted, spent.reason) == (False, "already-processed")
    assert [r.run_id for r in runs_of(rig, number)] == [first.run.run_id]


def test_a_second_bridge_on_the_same_state_cannot_run_concurrently(
    rig: Rig, background: Any, spawn_worker: Any
) -> None:
    first_issue = add_approved_issue(rig, "first")
    second_issue = add_approved_issue(rig, "second")
    rig.set_scenario(HANG)
    first = background(rig, lambda: rig.worker.poll_once(execute=True))
    active = wait_for_agent(rig)
    assert active.issue_number == first_issue

    other = spawn_worker(rig, worker_id="second-bridge")
    report = other.poll_once(execute=True)

    assert report.runs == []
    assert [r.code for r in report.refused] == ["already_processed"]
    assert agent_starts(rig) == 1
    assert runs_of(rig, second_issue) == []
    assert rig.comments(second_issue) == []
    assert "agent-running" not in rig.fake.issue_labels(second_issue)
    denied = [
        e for e in other.ledger.events(issue_number=second_issue) if e["kind"] == "lease.denied"
    ]
    assert [e["detail"]["reason"] for e in denied] == ["busy"]

    # Once the first run ends, the waiting issue gets its turn.
    rig.fake.close_issue(first_issue)
    cancelled = first.result().runs[0]
    assert cancelled.status is RunStatus.CANCELLED, describe(cancelled)
    rig.docs_scenario()
    follow_up = other.poll_once(execute=True)
    assert [r.issue_number for r in follow_up.runs] == [second_issue]
    assert follow_up.runs[0].status is RunStatus.DONE, describe(follow_up.runs[0])
    assert agent_starts(rig) == 2


# ----------------------------------------------------------------------------------------------
# duplicates and re-approval
# ----------------------------------------------------------------------------------------------


def test_identical_task_content_in_another_issue_is_refused_as_a_duplicate(rig: Rig) -> None:
    body = task_body(objective="Document the shared topic in docs/guide.md.")
    original = rig.new_task_issue(title="Original", body=body)
    twin = rig.new_task_issue(title="The same task, filed again", body=body)

    first = rig.poll()
    assert [r.issue_number for r in first.runs] == [original]
    assert first.runs[0].status is RunStatus.DONE, describe(first.runs[0])

    second = rig.poll()
    assert second.runs == []
    refusals = comments_with(rig, twin, REFUSAL_HEADING)
    assert len(refusals) == 1
    assert "duplicate_task" in refusals[0] and f"#{original}" in refusals[0]
    assert runs_of(rig, twin) == []
    assert len(rig.invocations()) == 1
    assert rig.fake.issue_labels(twin) == ["agent-approved", "agent-task"]
    denied = [e for e in rig.ledger.events(issue_number=twin) if e["kind"] == "lease.denied"]
    assert denied[0]["detail"]["reason"] == "duplicate-content"
    assert denied[0]["detail"]["other"] == only_run(rig, original).run_id

    # The notice is posted once, however often the issue is polled.
    rig.poll()
    rig.poll()
    assert len(rig.comments(twin)) == 1
    assert len(rig.invocations()) == 1


def test_identical_task_content_runs_when_dedupe_by_content_is_off(make_rig_for: Any) -> None:
    rig: Rig = make_rig_for(dedupe_by_content=False, poll={"max_jobs_per_poll": 2})
    body = task_body(objective="Document the shared topic in docs/guide.md.")
    first = rig.new_task_issue(title="Original", body=body)
    second = rig.new_task_issue(title="The same task, filed again", body=body)

    report = rig.poll()

    assert [r.issue_number for r in report.runs] == [first, second]
    assert all(r.status is RunStatus.DONE for r in report.runs), [describe(r) for r in report.runs]
    assert len(rig.invocations()) == 2
    assert len(rig.fake.pulls) == 2
    assert comments_with(rig, second, REFUSAL_HEADING) == []


def test_a_blocked_twin_does_not_make_a_new_issue_a_duplicate(rig: Rig) -> None:
    body = task_body(objective="Document the shared topic in docs/guide.md.")
    first = rig.new_task_issue(title="First attempt", body=body)
    rig.set_scenario(NO_CHANGES)
    blocked = rig.run_issue(first)
    assert blocked.status is RunStatus.BLOCKED, describe(blocked)

    # "Open a new issue" is the advice after a failed attempt, so it must be allowed to run.
    rig.docs_scenario()
    second = rig.new_task_issue(title="Second attempt", body=body)
    run = rig.run_issue(second)

    assert run.status is RunStatus.DONE, describe(run)
    assert comments_with(rig, second, REFUSAL_HEADING) == []


def test_approving_the_same_issue_twice_without_removal_runs_it_once(rig: Rig) -> None:
    number = rig.new_task_issue()
    approval_event = rig.fake.events[number][-1]["id"]
    assert rig.fake.label_issue(number, "agent-approved", actor="owner") == approval_event
    assert rig.fake.label_issue(number, "agent-approved", actor="owner") == approval_event

    report = rig.poll()
    assert [r.issue_number for r in report.runs] == [number]
    assert report.runs[0].approval_event_id == approval_event

    assert rig.fake.label_issue(number, "agent-approved", actor="owner") == approval_event
    assert rig.poll().runs == []
    assert len(runs_of(rig, number)) == 1
    assert len(rig.invocations()) == 1


def test_removing_and_reapplying_approval_after_done_does_nothing(rig: Rig) -> None:
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    assert run.status is RunStatus.DONE, describe(run)

    rig.fake.unlabel_issue(number, "agent-approved")
    rig.fake.label_issue(number, "agent-approved")
    rig.fake.reset_requests()
    report = rig.poll()

    assert report.runs == []
    assert [r.code for r in report.refused] == ["already_handled"]
    assert "agent-done" in rig.fake.issue_labels(number)
    assert rig.fake.writes() == []
    assert len(runs_of(rig, number)) == 1
    assert len(rig.invocations()) == 1


def test_reapproval_after_a_blocked_run_starts_a_new_run_for_the_same_issue(rig: Rig) -> None:
    number = rig.new_task_issue()
    rig.set_scenario(NO_CHANGES)
    first = rig.run_issue(number)
    assert first.status is RunStatus.BLOCKED and first.failure_kind == "no_changes"
    assert {"agent-blocked", "agent-approved"} <= set(rig.fake.issue_labels(number))
    assert "agent-running" not in rig.fake.issue_labels(number)

    # The issue stays parked while it carries the blocked label ...
    assert [r.code for r in rig.poll().refused] == ["already_handled"]

    # ... and removing only that label does not restart it: the approval has been spent.
    rig.docs_scenario()
    rig.fake.unlabel_issue(number, "agent-blocked")
    parked = rig.poll()
    assert parked.runs == []
    assert [r.code for r in parked.refused] == ["already_processed"]
    assert agent_starts(rig) == 1

    # A fresh approval starts the next run.
    rig.fake.unlabel_issue(number, "agent-approved")
    rig.fake.label_issue(number, "agent-approved")
    second = rig.run_issue(number)

    assert second.status is RunStatus.DONE, describe(second)
    assert second.run_id != first.run_id
    assert second.approval_event_id > first.approval_event_id
    runs = runs_of(rig, number)
    assert [r.run_id for r in runs] == [first.run_id, second.run_id]
    assert [r.status for r in runs] == [RunStatus.BLOCKED, RunStatus.DONE]
    assert len(comments_with(rig, number, BLOCKED_HEADING)) == 1
    assert len(comments_with(rig, number, DONE_HEADING)) == 1
    assert agent_starts(rig) == 2


def test_max_attempts_per_issue_caps_reapproval(make_rig_for: Any) -> None:
    rig: Rig = make_rig_for(limits={"max_attempts_per_issue": 2})
    body = task_body(objective="Document the stubborn topic in docs/guide.md.")
    number = rig.new_task_issue(body=body)
    rig.set_scenario(NO_CHANGES)

    for attempt in (1, 2):
        run = rig.run_issue(number)
        assert run.status is RunStatus.BLOCKED, describe(run)
        assert len(runs_of(rig, number)) == attempt
        rig.fake.unlabel_issue(number, "agent-blocked")
        rig.fake.unlabel_issue(number, "agent-approved")
        rig.fake.label_issue(number, "agent-approved")

    report = rig.poll()

    assert report.runs == []
    assert len(runs_of(rig, number)) == 2
    assert agent_starts(rig) == 2
    capped = comments_with(rig, number, REFUSAL_HEADING)
    assert len(capped) == 1 and "attempt_limit" in capped[0]
    assert "agent-running" not in rig.fake.issue_labels(number)
    denied = [e for e in rig.ledger.events(issue_number=number) if e["kind"] == "lease.denied"]
    assert denied[-1]["detail"]["reason"] == "attempt-limit"
    # Polling again does not repeat the notice.
    rig.poll()
    assert len(comments_with(rig, number, REFUSAL_HEADING)) == 1

    # The advice in the notice works: a new issue with the same task content is not a duplicate,
    # because the capped issue's runs were all blocked.
    rig.docs_scenario()
    fresh = rig.new_task_issue(title="A fresh issue", body=body)
    run = rig.run_issue(fresh)
    assert run.status is RunStatus.DONE, describe(run)


# ----------------------------------------------------------------------------------------------
# cancellation
# ----------------------------------------------------------------------------------------------


def assert_cancelled_without_publishing(rig: Rig, number: int, *, detail: str) -> RunRecord:
    """The run ended CANCELLED, said so once, released the issue and published nothing."""
    run = only_run(rig, number)
    assert run.status is RunStatus.CANCELLED, describe(run)
    assert run.phase is RunPhase.FINISHED
    assert detail in (run.failure_detail or ""), describe(run)
    assert "agent-running" not in rig.fake.issue_labels(number)
    assert not {"agent-done", "agent-blocked"} & set(rig.fake.issue_labels(number))
    assert len(comments_with(rig, number, STARTED_HEADING)) == 1
    cancelled = comments_with(rig, number, CANCELLED_HEADING)
    assert len(cancelled) == 1, rig.comments(number)
    assert detail in cancelled[0]
    assert comments_with(rig, number, DONE_HEADING) == []
    assert rig.fake.pulls == {}
    assert rig.git_env.origin_branches() == ["main"]
    assert run.pr_number is None
    return run


def test_issue_closed_before_the_poll_never_executes(rig: Rig) -> None:
    number = rig.new_task_issue()
    rig.fake.close_issue(number)
    rig.fake.reset_requests()

    report = rig.poll()

    assert report.considered == 0 and report.runs == []
    assert rig.invocations() == []
    assert rig.ledger.list_runs() == []
    assert rig.fake.writes() == []

    # Asked for explicitly, the closed issue is refused rather than silently skipped.
    explicit = rig.poll(only_issue=number)
    assert [r.code for r in explicit.refused] == ["not_open"]
    assert explicit.runs == []
    assert rig.invocations() == []
    assert rig.ledger.list_runs() == []
    assert rig.fake.writes() == []
    assert rig.worktrees() == []


def test_issue_closed_while_the_agent_runs_cancels_the_run(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue()
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    active = wait_for_agent(rig)
    assert "agent-running" in rig.fake.issue_labels(number)
    agent_pid = active.agent_pid
    assert agent_pid is not None and pid_is_alive(agent_pid)

    rig.fake.close_issue(number)
    report = poll.result()

    assert [r.run_id for r in report.runs] == [active.run_id]
    run = assert_cancelled_without_publishing(rig, number, detail="the issue was closed")
    assert run.failure_kind == "cancelled"
    assert not pid_is_alive(agent_pid)
    assert agent_starts(rig) == 1


def test_approval_removed_while_the_agent_runs_cancels_the_run(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue()
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    active = wait_for_agent(rig)
    agent_pid = active.agent_pid
    assert agent_pid is not None

    rig.fake.unlabel_issue(number, "agent-approved")
    poll.result()

    assert_cancelled_without_publishing(rig, number, detail="the approval label was removed")
    assert not pid_is_alive(agent_pid)
    # The owner's own labels are left as they are: the approval label stays removed.
    assert "agent-approved" not in rig.fake.issue_labels(number)
    assert rig.poll().runs == []
    assert agent_starts(rig) == 1


@pytest.mark.parametrize("selector", ["issue", "run"])
def test_local_cancel_while_the_agent_runs_cancels_the_run(
    selector: str, make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue()
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    active = wait_for_agent(rig)
    agent_pid = active.agent_pid
    assert agent_pid is not None

    if selector == "issue":
        flagged = rig.ledger.request_cancel(issue_number=number)
    else:
        flagged = rig.ledger.request_cancel(run_id=active.run_id)
    assert flagged == 1
    poll.result()

    run = assert_cancelled_without_publishing(rig, number, detail="cancelled locally")
    assert run.cancel_requested is True
    assert not pid_is_alive(agent_pid)
    # A finished run cannot be cancelled again.
    assert rig.ledger.request_cancel(issue_number=number) == 0
    assert rig.poll().runs == []
    assert agent_starts(rig) == 1


def test_approval_withdrawn_between_commit_and_publish_is_not_pushed(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    number = rig.new_task_issue()
    original = rig.worker._publish_once

    def withdraw_then_publish(run: RunRecord, payload: dict[str, Any]) -> None:
        assert run.status is RunStatus.PUBLISHING and run.commit_sha  # the commit is made
        rig.fake.unlabel_issue(number, "agent-approved")
        original(run, payload)

    monkeypatch.setattr(rig.worker, "_publish_once", withdraw_then_publish)
    report = rig.poll()

    assert [r.issue_number for r in report.runs] == [number]
    run = assert_cancelled_without_publishing(rig, number, detail="the approval label was removed")
    assert run.commit_sha is not None
    assert agent_starts(rig) == 1
    # The commit exists only in the kept local worktree; it never reached the remote.
    assert run.branch in local_agent_branches(rig)
    assert rig.git.worktree_head(Path(run.worktree_path or "")) == run.commit_sha


def test_a_replaced_approval_cancels_the_old_run_and_the_new_one_runs_afresh(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    number = rig.new_task_issue()
    original = rig.worker._publish_once

    def replace_approval_then_publish(run: RunRecord, payload: dict[str, Any]) -> None:
        rig.fake.unlabel_issue(number, "agent-approved")
        rig.fake.label_issue(number, "agent-approved")
        original(run, payload)

    monkeypatch.setattr(rig.worker, "_publish_once", replace_approval_then_publish)
    report = rig.poll()
    first = assert_cancelled_without_publishing(
        rig, number, detail="the approval was withdrawn or replaced"
    )
    assert [r.run_id for r in report.runs] == [first.run_id]

    monkeypatch.undo()
    second = rig.run_issue(number)

    assert second.status is RunStatus.DONE, describe(second)
    assert second.run_id != first.run_id
    assert second.approval_event_id > first.approval_event_id
    assert len(rig.fake.pulls) == 1
    assert rig.git_env.origin_branches() == sorted(["main", second.branch or ""])
    assert agent_starts(rig) == 2


def test_local_cancel_requested_after_the_agent_finished_stops_the_publication(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    number = rig.new_task_issue()
    original = rig.worker._publish_once

    def cancel_then_publish(run: RunRecord, payload: dict[str, Any]) -> None:
        assert rig.ledger.request_cancel(issue_number=number) == 1
        original(run, payload)

    monkeypatch.setattr(rig.worker, "_publish_once", cancel_then_publish)
    rig.poll()

    assert_cancelled_without_publishing(rig, number, detail="cancelled locally")


def test_a_run_another_process_concluded_is_neither_resurrected_nor_pushed(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal states are final: this worker's next ledger write is refused and it stops (F10)."""
    number = rig.new_task_issue()
    other = Ledger(rig.ledger.path, clock=rig.clock)  # a second process on the same state
    real_commit = rig.git.commit

    def commit_then_lose_the_run(path: Path, message: str) -> str:
        sha = real_commit(path, message)
        (active,) = rig.ledger.active_runs()
        assert other.finish(
            active.run_id,
            RunStatus.CANCELLED,
            failure_kind="cancelled",
            failure_detail="cancel: concluded by another process",
        )
        return sha

    monkeypatch.setattr(rig.git, "commit", commit_then_lose_the_run)
    try:
        report = rig.poll()
    finally:
        other.close()

    run = only_run(rig, number)
    assert [r.run_id for r in report.runs] == [run.run_id]
    # The other process's verdict stands: not moved back to publishing, not turned into BLOCKED.
    assert run.status is RunStatus.CANCELLED and run.phase is RunPhase.FINISHED, describe(run)
    assert run.failure_kind == "cancelled"
    assert run.failure_detail == "cancel: concluded by another process"
    assert run.pr_number is None
    assert rig.git_env.origin_branches() == ["main"] and rig.fake.pulls == {}
    # This worker reported nothing: that is up to the process that concluded the run.
    assert comments_with(rig, number, DONE_HEADING) == []
    assert comments_with(rig, number, BLOCKED_HEADING) == []
    assert not {"agent-done", "agent-blocked"} & set(rig.fake.issue_labels(number))
    assert agent_starts(rig) == 1


def test_a_done_that_comes_after_another_process_concluded_the_run_is_dropped(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    number = rig.new_task_issue()
    other = Ledger(rig.ledger.path, clock=rig.clock)
    real_mark = rig.worker.publisher.mark_terminal

    def mark_then_lose_the_run(issue_number: int, status: RunStatus) -> None:
        real_mark(issue_number, status)
        (active,) = rig.ledger.active_runs()
        assert other.finish(active.run_id, RunStatus.CANCELLED, failure_kind="cancelled")

    monkeypatch.setattr(rig.worker.publisher, "mark_terminal", mark_then_lose_the_run)
    try:
        rig.poll()
    finally:
        other.close()

    run = only_run(rig, number)
    assert run.status is RunStatus.CANCELLED, describe(run)  # not overwritten with DONE
    assert run.failure_kind == "cancelled" and not (run.result or {}).get("notified")
    # The DONE clean-up did not run for a run that is not done.
    assert Path(run.worktree_path or "").exists() and run.branch in local_agent_branches(rig)


def test_recovery_that_loses_the_run_while_resuming_says_so_instead_of_crashing(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    number = rig.new_task_issue()
    monkeypatch.setattr(
        rig.worker, "_publish", lambda *a, **k: (_ for _ in ()).throw(SimulatedCrash("publish"))
    )
    with pytest.raises(SimulatedCrash):
        rig.poll()
    monkeypatch.undo()
    crashed = only_run(rig, number)
    assert crashed.status is RunStatus.PUBLISHING and crashed.phase is RunPhase.PUSH
    worker = restart(rig)
    expire_leases(rig)
    other = Ledger(rig.ledger.path, clock=rig.clock)
    real_push = worker.git.push

    def push_then_lose_the_run(branch: str) -> None:
        real_push(branch)
        assert other.finish(
            crashed.run_id,
            RunStatus.CANCELLED,
            failure_kind="cancelled",
            result={"notified": True},  # the other process reported it itself
        )

    monkeypatch.setattr(worker.git, "push", push_then_lose_the_run)
    try:
        actions = worker.recover(execute=True)
    finally:
        other.close()

    assert_reported(actions, "abandoned", crashed)
    assert f"run {crashed.run_id} is not active (status cancelled)" in actions[0]
    assert current(rig, crashed.run_id).status is RunStatus.CANCELLED
    assert rig.fake.pulls == {}  # it never got as far as the pull request


def test_a_pending_run_cancelled_locally_never_starts_its_agent(rig: Rig) -> None:
    number = rig.new_task_issue()
    run = crash_mid_agent(rig, number)
    assert rig.ledger.requeue_interrupted(run.run_id, 1)
    assert current(rig, run.run_id).status is RunStatus.PENDING
    assert rig.ledger.request_cancel(issue_number=number) == 1

    rig.poll()

    assert agent_starts(rig) == 0
    final = current(rig, run.run_id)
    assert final.status is RunStatus.CANCELLED, describe(final)
    assert rig.fake.pulls == {}
    assert "agent-running" not in rig.fake.issue_labels(number)


# ----------------------------------------------------------------------------------------------
# the task is edited while the agent works
# ----------------------------------------------------------------------------------------------


def assert_blocked_task_changed(rig: Rig, number: int) -> RunRecord:
    run = only_run(rig, number)
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "task_changed", describe(run)
    labels = rig.fake.issue_labels(number)
    assert "agent-blocked" in labels and "agent-running" not in labels
    assert "agent-done" not in labels
    blocked = comments_with(rig, number, BLOCKED_HEADING)
    assert len(blocked) == 1 and "`task_changed`" in blocked[0]
    assert "edited after approval" in blocked[0] and "re-apply" in blocked[0]
    assert rig.fake.pulls == {}
    assert rig.git_env.origin_branches() == ["main"]
    assert run.pr_number is None
    return run


def test_editing_the_body_while_the_agent_runs_blocks_the_run(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue()
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    active = wait_for_agent(rig)
    agent_pid = active.agent_pid
    assert agent_pid is not None

    rig.fake.edit_issue(
        number, body=task_body(objective="Rewrite the whole guide and delete the old text.")
    )
    poll.result()

    assert_blocked_task_changed(rig, number)
    assert not pid_is_alive(agent_pid)
    assert agent_starts(rig) == 1
    assert rig.poll().runs == []


def test_editing_only_the_title_while_the_agent_runs_is_detected(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue(title="Document the bridge")
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    wait_for_agent(rig)

    rig.fake.edit_issue(number, title="Document the bridge and also the mower")
    poll.result()

    assert_blocked_task_changed(rig, number)


def test_a_fresh_approval_after_the_edit_runs_the_edited_task(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue(objective="Document the original topic in docs/guide.md.")
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    wait_for_agent(rig)
    edited = task_body(objective="Document the revised topic in docs/guide.md.")
    rig.fake.edit_issue(number, body=edited)
    poll.result()
    first = assert_blocked_task_changed(rig, number)

    # The comment's advice: review the edit, then remove and re-apply the approval.
    rig.docs_scenario()
    rig.fake.unlabel_issue(number, "agent-blocked")
    rig.fake.unlabel_issue(number, "agent-approved")
    rig.fake.label_issue(number, "agent-approved")
    second = rig.run_issue(number)

    assert second.status is RunStatus.DONE, describe(second)
    assert second.run_id != first.run_id
    assert second.revision != first.revision
    prompt = rig.invocations()[-1]["prompt"]
    assert "revised topic" in prompt and "original topic" not in prompt


def test_an_edit_that_only_changes_line_endings_is_not_a_task_change(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    body = task_body(objective="Document the bridge in docs/guide.md.")
    number = rig.new_task_issue(body=body)
    # The agent works for a moment, so the edit lands while it runs.
    rig.set_scenario([{"op": "sleep", "seconds": 1.0}, *docs_edit_steps()])
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    wait_for_agent(rig)

    # GitHub's web editor stores bodies with CRLF line endings.
    rig.fake.edit_issue(number, body=body.replace("\n", "\r\n"))
    report = poll.result()

    run = report.runs[0]
    assert run.status is RunStatus.DONE, describe(run)
    assert len(rig.fake.pulls) == 1


# ----------------------------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------------------------

NONCE = re.compile(r"TASK-DATA-[0-9a-f]{16}")


def stable_argv(argv: list[str]) -> list[str]:
    """argv with the two per-run paths (omp overlay, worktree) replaced by placeholders."""
    out = list(argv)
    for flag in ("--config", "--cwd"):
        if flag in out:
            out[out.index(flag) + 1] = f"<{flag.lstrip('-')}>"
    return out


def stable_preview(preview: Preview) -> dict[str, Any]:
    """The deterministic part of a preview (the prompt nonce and temp overlay path vary)."""
    data = preview.to_dict(include_prompt=True)
    data["command"] = stable_argv(data["command"])
    data["prompt"] = NONCE.sub("TASK-DATA-<nonce>", data["prompt"])
    return data


def new_budgeted_issue(rig: Rig) -> int:
    return rig.new_task_issue(
        objective="Document the dry run in docs/guide.md.",
        checks=["pass"],
        budget_usd=0.5,
        timeout_minutes=2,
    )


def test_dry_run_builds_previews_and_changes_nothing(rig: Rig) -> None:
    number = new_budgeted_issue(rig)
    rig.fake.reset_requests()
    before = world_state(rig)

    report = rig.poll(execute=False)

    assert report.considered == 1 and report.runs == [] and report.refused == []
    [preview] = report.previews
    assert preview.issue_number == number
    assert preview.backend == rig.backend and preview.approver == "owner"
    assert preview.scope == "docs-only"
    assert preview.budget_usd == 0.5 and preview.timeout_minutes == 2
    assert preview.checks == ["pass"]
    assert preview.write_allow_sets == [list(rig.cfg.scopes.docs_paths)]
    assert preview.write_deny == list(rig.cfg.protected_paths)
    assert "backend/src/safety/**" in preview.write_deny
    assert preview.base == rig.git_env.git("rev-parse", "origin/main")
    assert preview.notes == []
    assert preview.prompt_chars == len(preview.prompt)
    assert "Document the dry run in docs/guide.md." in preview.prompt
    assert preview.branch.startswith(f"agent/issue-{number}-")

    # Nothing happened anywhere.
    assert rig.fake.writes() == []
    assert rig.invocations() == []
    assert rig.worktrees() == []
    assert registered_worktrees(rig) == []
    assert local_agent_branches(rig) == [] and rig.git_env.origin_branches() == ["main"]
    assert rig.ledger.list_runs() == [] and rig.ledger.events() == []
    assert table_counts(rig) == {"runs": 0, "events": 0, "notices": 0}
    assert world_state(rig) == before

    # Dry running is repeatable and just as quiet.
    again = rig.poll(execute=False)
    assert [p.issue_number for p in again.previews] == [number]
    assert world_state(rig) == before


def test_dry_run_preview_is_what_the_real_run_executes(rig: Rig) -> None:
    number = new_budgeted_issue(rig)
    [preview] = rig.poll(execute=False).previews
    binary = rig.cfg.agent.backend(rig.backend).binary
    assert preview.command[0] == binary

    run = rig.run_issue(number)
    assert run.status is RunStatus.DONE, describe(run)
    [invocation] = rig.invocations()

    assert stable_argv(preview.command[1:]) == stable_argv(invocation["argv"])
    assert NONCE.sub("N", preview.prompt) == NONCE.sub("N", invocation["prompt"])
    # The preview is for the right CLI and shows the task's own limits.
    if rig.backend == "omp":
        assert preview.command[1:4] == ["-p", "--mode", "json"]
        assert preview.command[preview.command.index("--cwd") + 1] == "<worktree>"
        assert Path(invocation["parsed"]["flags"]["--cwd"]).name == run.run_id
        assert invocation["cwd"] == invocation["parsed"]["flags"]["--cwd"]
    else:
        assert preview.command[1:4] == ["-p", "--output-format", "json"]
        assert preview.command[preview.command.index("--max-budget-usd") + 1] == "0.50"
        assert Path(invocation["cwd"]).name == run.run_id


def test_dry_run_flags_a_missing_agent_binary(
    make_rig_for: Any, backend_name: str, tmp_path: Path
) -> None:
    missing = str(tmp_path / "nowhere" / backend_name)
    rig: Rig = make_rig_for(agent={backend_name: {"binary": missing}})
    rig.new_task_issue()

    [preview] = rig.poll(execute=False).previews

    assert preview.command[0] == missing
    assert any(missing in note and "not found" in note for note in preview.notes), preview.notes
    assert rig.invocations() == [] and rig.fake.writes() == []


def test_dry_run_flags_a_base_branch_that_was_never_fetched(rig: Rig) -> None:
    rig.new_task_issue()
    rig.git_env.git("update-ref", "-d", "refs/remotes/origin/main")
    before = rig.git_env.git("for-each-ref")

    [preview] = rig.poll(execute=False).previews

    assert preview.base == "(origin/main not fetched yet)"
    assert any("base branch" in note for note in preview.notes), preview.notes
    assert rig.git_env.git("for-each-ref") == before  # a dry run does not even fetch


def test_dry_run_through_the_read_only_client_behaves_identically(rig: Rig) -> None:
    new_budgeted_issue(rig)
    rig.new_task_issue(title="Not a task", body="Please just do something.")
    readonly = ReadOnlyGitHub(rig.client)
    worker = build_worker(rig, github=readonly, worker_id="dry-run-worker")
    try:
        rig.fake.reset_requests()
        before = world_state(rig)

        plain = rig.poll(execute=False)
        guarded = worker.poll_once(execute=False)

        assert [stable_preview(p) for p in guarded.previews] == [
            stable_preview(p) for p in plain.previews
        ]
        assert [r.render() for r in guarded.refused] == [r.render() for r in plain.refused]
        assert [r.code for r in guarded.refused] == ["invalid_task"]
        assert guarded.considered == plain.considered == 2
        assert rig.fake.writes() == []
        assert world_state(rig) == before

        # The wrapper itself refuses writes and lets reads through.
        for call in (
            lambda: readonly.create_comment(1, "hello"),
            lambda: readonly.add_labels(1, ["agent-running"]),
            lambda: readonly.remove_label(1, "agent-approved"),
            lambda: readonly.ensure_labels(["agent-task"]),
            lambda: readonly.create_pull_request(title="t", head="h", base="main", body="b"),
        ):
            with pytest.raises(BridgeError, match="dry-run"):
                call()
        assert [i.number for i in readonly.list_candidate_issues(["agent-task"])] == [1, 2]
        assert rig.fake.writes() == []
    finally:
        worker.ledger.close()


def test_dry_run_reports_refusals_without_recording_or_commenting(rig: Rig) -> None:
    rig.new_task_issue(title="Not a task", body="Please just do something.")
    rig.fake.reset_requests()

    report = rig.poll(execute=False)

    assert [r.code for r in report.refused] == ["invalid_task"]
    assert report.refused[0].notify is True  # a real run would comment ...
    assert report.previews == [] and report.runs == []
    assert rig.fake.writes() == []  # ... a dry run does not
    assert table_counts(rig) == {"runs": 0, "events": 0, "notices": 0}

    executed = rig.poll()
    assert [r.code for r in executed.refused] == ["invalid_task"]
    assert len(rig.fake.writes()) == 1  # the one refusal comment
    assert table_counts(rig)["notices"] == 2


def test_recover_without_execute_only_reports(rig: Rig, orphan: Any) -> None:
    number = rig.new_task_issue()
    agent = orphan()
    crashed = crash_mid_agent(rig, number, orphan=agent)
    expire_leases(rig)
    rig.fake.reset_requests()
    before = world_state(rig)

    actions = rig.worker.recover(execute=False)

    assert_reported(actions, "would recover", crashed)
    assert rig.ledger.get(crashed.run_id) == crashed
    assert pid_is_alive(agent.pid)
    assert rig.fake.writes() == []
    assert world_state(rig) == before
    assert "agent-running" in rig.fake.issue_labels(number)

    # A dry-run poll shows the same report and still changes nothing.
    report = rig.poll(execute=False)
    assert report.recovered == actions
    assert report.previews == [] and report.runs == []
    assert rig.ledger.get(crashed.run_id) == crashed
    assert pid_is_alive(agent.pid)
    assert rig.fake.writes() == []


# ----------------------------------------------------------------------------------------------
# crash recovery
# ----------------------------------------------------------------------------------------------


def test_a_lease_is_stale_only_after_it_expires(rig: Rig, orphan: Any) -> None:
    stuck = rig.new_task_issue(objective="Document the stuck topic in docs/guide.md.")
    waiting = add_approved_issue(rig, "waiting")
    agent = orphan()
    crashed = crash_mid_agent(rig, stuck, orphan=agent)

    # One second before the lease runs out the worker may merely be slow: hands off.
    rig.clock.advance(rig.cfg.limits.lease_ttl_seconds - 1)
    rig.fake.reset_requests()
    assert rig.worker.recover(execute=True) == []
    report = rig.poll()
    assert report.recovered == [] and report.runs == []
    assert [r.code for r in report.refused] == ["already_processed"]
    assert rig.ledger.get(crashed.run_id) == crashed
    assert pid_is_alive(agent.pid)
    assert agent_starts(rig) == 0
    assert runs_of(rig, waiting) == []  # the single slot is still held
    assert rig.fake.writes() == []

    # Once it has expired, the next poll recovers the run before anything else.
    rig.clock.advance(2)
    report = rig.poll()

    assert_reported(report.recovered, "interrupted", crashed)
    assert [r.run_id for r in report.runs] == [crashed.run_id]
    assert not pid_is_alive(agent.pid)
    assert current(rig, crashed.run_id).status is RunStatus.DONE
    assert runs_of(rig, waiting) == []  # one run per poll: the other issue is next
    assert [r.issue_number for r in rig.poll().runs] == [waiting]


def test_a_run_whose_worker_keeps_heartbeating_is_never_recovered(
    rig: Rig, background: Any, spawn_worker: Any
) -> None:
    number = rig.new_task_issue()
    rig.set_scenario(HANG)
    poll = background(rig, lambda: rig.worker.poll_once(execute=True))
    active = wait_for_agent(rig)
    agent_pid = active.agent_pid
    assert agent_pid is not None and active.lease_expires_at is not None
    first_expiry = active.lease_expires_at

    # Far more than a lease passes; the live worker's next abort check renews the lease.
    rig.clock.advance(rig.cfg.limits.lease_ttl_seconds * 2)
    wait_until(
        lambda: (current(rig, active.run_id).lease_expires_at or "") > first_expiry,
        what="the worker to renew its lease",
    )

    other = spawn_worker(rig, worker_id="second-bridge")
    assert other.recover(execute=True) == []
    assert pid_is_alive(agent_pid)
    assert current(rig, active.run_id).status is RunStatus.RUNNING

    rig.fake.close_issue(number)
    poll.result()
    assert current(rig, active.run_id).status is RunStatus.CANCELLED


def test_a_crashed_agent_run_is_recovered_and_the_same_run_resumes_to_done(
    rig: Rig, orphan: Any
) -> None:
    number = rig.new_task_issue()
    agent = orphan()
    crashed = crash_mid_agent(rig, number, orphan=agent)
    assert pid_is_alive(agent.pid)
    assert "agent-running" in rig.fake.issue_labels(number)
    worktree = Path(crashed.worktree_path or "")
    assert worktree.is_dir() and crashed.branch in local_agent_branches(rig)
    expire_leases(rig)

    worker = restart(rig)  # a fresh bridge process finds what the dead one left behind
    actions = worker.recover(execute=True)

    assert_reported(actions, "interrupted", crashed)
    wait_until(lambda: not pid_is_alive(agent.pid), timeout=5, what="the orphan to be killed")
    requeued = rig.ledger.get(crashed.run_id)
    assert requeued is not None
    assert requeued.status is RunStatus.PENDING and requeued.phase is RunPhase.LEASED
    assert requeued.attempt == crashed.attempt + 1 == 2
    assert requeued.interrupted_count == 1
    assert requeued.agent_pid is None and requeued.lease_expires_at is None
    assert "agent-running" not in rig.fake.issue_labels(number)
    assert not worktree.exists() and local_agent_branches(rig) == []
    assert len(comments_with(rig, number, STARTED_HEADING)) == 1
    assert rig.fake.pulls == {} and agent_starts(rig) == 0

    report = worker.poll_once(execute=True)

    [run] = report.runs
    assert run.run_id == crashed.run_id
    assert run.status is RunStatus.DONE, describe(run)
    assert run.attempt == 2 and run.interrupted_count == 1
    assert run.worker_id == "restarted-worker"
    assert [r.run_id for r in runs_of(rig, number)] == [crashed.run_id]
    assert agent_starts(rig) == 1 and len(rig.fake.pulls) == 1
    assert len(comments_with(rig, number, STARTED_HEADING)) == 1  # not posted a second time
    assert len(comments_with(rig, number, DONE_HEADING)) == 1
    assert {"agent-done"} <= set(rig.fake.issue_labels(number))
    assert "agent-running" not in rig.fake.issue_labels(number)


def test_an_interrupted_run_without_retry_budget_is_blocked_as_interrupted(
    make_rig_for: Any, orphan: Any
) -> None:
    rig: Rig = make_rig_for(limits={"auto_retry_interrupted": 0})
    number = rig.new_task_issue()
    agent = orphan()
    crashed = crash_mid_agent(rig, number, orphan=agent)
    expire_leases(rig)

    actions = rig.worker.recover(execute=True)

    assert_reported(actions, "interrupted", crashed)
    wait_until(lambda: not pid_is_alive(agent.pid), timeout=5, what="the orphan to be killed")
    run = rig.ledger.get(crashed.run_id)
    assert run is not None
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "interrupted", describe(run)
    assert "no automatic retries left" in (run.failure_detail or "")
    labels = rig.fake.issue_labels(number)
    assert "agent-blocked" in labels and "agent-running" not in labels
    blocked = comments_with(rig, number, BLOCKED_HEADING)
    assert len(blocked) == 1 and "`interrupted`" in blocked[0] and "Re-approve" in blocked[0]
    assert rig.fake.pulls == {} and agent_starts(rig) == 0
    # Blocked is final until a human re-approves.
    assert rig.poll().runs == [] and agent_starts(rig) == 0


def test_an_interrupted_run_is_retried_only_within_the_retry_budget(rig: Rig, orphan: Any) -> None:
    number = rig.new_task_issue()
    first_agent = orphan()
    crashed = crash_mid_agent(rig, number, orphan=first_agent)
    expire_leases(rig)
    rig.worker.recover(execute=True)
    assert current(rig, crashed.run_id).status is RunStatus.PENDING

    # The resumed attempt dies as well.
    second_agent = orphan()
    again = crash_mid_agent(rig, number, orphan=second_agent)
    assert again.run_id == crashed.run_id and again.interrupted_count == 1
    expire_leases(rig)
    actions = rig.worker.recover(execute=True)

    assert_reported(actions, "interrupted", again)
    wait_until(lambda: not pid_is_alive(second_agent.pid), timeout=5, what="the orphan to die")
    run = rig.ledger.get(crashed.run_id)
    assert run is not None
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "interrupted", describe(run)
    assert run.attempt == 2 and run.interrupted_count == 1
    assert "agent-blocked" in rig.fake.issue_labels(number)
    assert len(comments_with(rig, number, BLOCKED_HEADING)) == 1
    assert [r.run_id for r in runs_of(rig, number)] == [crashed.run_id]
    assert agent_starts(rig) == 0 and rig.fake.pulls == {}


def test_a_crash_before_the_agent_started_uses_the_start_retry_budget(
    make_rig_for: Any,
) -> None:
    rig: Rig = make_rig_for(limits={"auto_retry_interrupted": 0, "max_start_retries": 1})
    number = rig.new_task_issue()
    crashed = crash_mid_agent(rig, number, phase=RunPhase.LEASED)
    expire_leases(rig)

    assert_reported(rig.worker.recover(execute=True), "interrupted", crashed)
    requeued = rig.ledger.get(crashed.run_id)
    assert requeued is not None and requeued.status is RunStatus.PENDING, describe(requeued)
    assert requeued.interrupted_count == 1

    # The start budget (1) is now spent: the next crash, at the worktree step, blocks the run.
    crash_mid_agent(rig, number, phase=RunPhase.WORKTREE)
    expire_leases(rig)
    rig.worker.recover(execute=True)
    run = rig.ledger.get(crashed.run_id)
    assert run is not None
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "interrupted", describe(run)
    assert agent_starts(rig) == 0


@pytest.mark.parametrize("recorded", ["start-time-off", "start-time-unknown"])
def test_a_recorded_pid_that_is_not_the_agent_is_never_killed(
    recorded: str, rig: Rig, orphan: Any
) -> None:
    number = rig.new_task_issue()
    bystander = orphan()  # the pid was reused by an unrelated process
    ticks = proc_start_ticks(bystander.pid)
    assert ticks is not None
    crashed = crash_mid_agent(
        rig,
        number,
        orphan=bystander,
        start_ticks=ticks + 1 if recorded == "start-time-off" else None,
    )
    assert crashed.agent_pid == bystander.pid
    expire_leases(rig)

    actions = rig.worker.recover(execute=True)

    assert_reported(actions, "interrupted", crashed)
    assert pid_is_alive(bystander.pid)
    assert current(rig, crashed.run_id).status is RunStatus.PENDING
    assert "agent-running" not in rig.fake.issue_labels(number)


PUBLISH_CRASH_POINTS = ["before_push", "after_push", "after_pull_request", "after_done_comment"]


@pytest.mark.parametrize("crash_point", PUBLISH_CRASH_POINTS)
def test_a_crash_while_publishing_is_finished_by_recovery_without_rerunning_the_agent(
    crash_point: str, rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    number = rig.new_task_issue()

    def crash_after(owner: Any, name: str, *, when: Callable[..., bool] = lambda *a, **k: True):
        real = getattr(owner, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = real(*args, **kwargs)
            if when(*args, **kwargs):
                raise SimulatedCrash(name)
            return result

        monkeypatch.setattr(owner, name, wrapper)

    if crash_point == "before_push":
        monkeypatch.setattr(
            rig.worker, "_publish", lambda *a, **k: (_ for _ in ()).throw(SimulatedCrash("publish"))
        )
    elif crash_point == "after_push":
        crash_after(rig.git, "push")
    elif crash_point == "after_pull_request":
        crash_after(rig.client, "create_pull_request")
    else:
        crash_after(
            rig.client,
            "create_comment",
            when=lambda number, body: "Draft pull request ready" in body,
        )
    with pytest.raises(SimulatedCrash):
        rig.poll()
    monkeypatch.undo()  # the replacement process has none of these hooks

    crashed = only_run(rig, number)
    assert crashed.status is RunStatus.PUBLISHING, describe(crashed)
    assert crashed.commit_sha and crashed.branch
    worktree = Path(crashed.worktree_path or "")
    assert rig.git.worktree_head(worktree) == crashed.commit_sha
    assert "agent-running" in rig.fake.issue_labels(number)
    assert agent_starts(rig) == 1
    pulls_at_crash = len(rig.fake.pulls)
    assert pulls_at_crash == (
        1 if crash_point in ("after_pull_request", "after_done_comment") else 0
    )

    worker = restart(rig)
    rig.fake.reset_requests()
    assert worker.recover(execute=True) == []  # the lease has not expired yet
    assert rig.fake.writes() == []
    expire_leases(rig)
    actions = worker.recover(execute=True)

    assert_reported(actions, "resumed publishing", crashed)
    run = rig.ledger.get(crashed.run_id)
    assert run is not None and run.status is RunStatus.DONE, describe(run)
    assert run.publish_attempts == 1 and run.pr_number is not None
    [(number_, pull)] = rig.fake.pulls.items()
    assert number_ == run.pr_number and pull.draft is True and pull.head_ref == crashed.branch
    assert rig.git_env.origin_branches() == sorted(["main", crashed.branch or ""])
    assert rig.git_env.origin_git("rev-parse", crashed.branch or "") == crashed.commit_sha
    labels = rig.fake.issue_labels(number)
    assert "agent-done" in labels and "agent-running" not in labels
    assert len(comments_with(rig, number, STARTED_HEADING)) == 1
    assert len(comments_with(rig, number, DONE_HEADING)) == 1
    assert agent_starts(rig) == 1  # the agent was not run again
    assert not worktree.exists() and local_agent_branches(rig) == []

    rig.fake.reset_requests()
    assert worker.poll_once(execute=True).runs == []
    assert rig.fake.writes() == []


def test_recover_reports_a_finished_run_whose_github_update_failed(rig: Rig) -> None:
    rig.fake.deny("POST", "/comments")
    number = rig.new_task_issue()
    run = rig.run_issue(number)

    # The bridge could not even say it had started, so the run ended blocked - silently.
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "github_permission"
    assert not (run.result or {}).get("notified")
    assert rig.comments(number) == []
    assert "agent-running" in rig.fake.issue_labels(number)  # the label change never happened
    assert "agent-blocked" not in rig.fake.issue_labels(number)
    assert agent_starts(rig) == 0

    # A dry run only reports; it neither retries nor pretends to.
    assert rig.worker.recover(execute=False) == []
    assert rig.comments(number) == []

    rig.fake.allow("POST", "/comments")
    report = rig.poll()

    assert_reported(report.recovered, "reported run", run, names_issue=False)
    blocked = comments_with(rig, number, BLOCKED_HEADING)
    assert len(blocked) == 1 and "`github_permission`" in blocked[0]
    labels = rig.fake.issue_labels(number)
    assert "agent-blocked" in labels and "agent-running" not in labels
    assert (current(rig, run.run_id).result or {}).get("notified") is True
    # Reported once: later recoveries find nothing to do.
    assert rig.worker.recover(execute=True) == []
    assert len(comments_with(rig, number, BLOCKED_HEADING)) == 1
    assert agent_starts(rig) == 0


def test_recover_reports_a_done_run_whose_notification_was_lost(rig: Rig) -> None:
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    assert run.status is RunStatus.DONE, describe(run)
    # Lose the GitHub side of the completion and the ledger's record that it was delivered.
    rig.fake.comments[number] = [
        c for c in rig.fake.comments[number] if f"run-{run.run_id}-done" not in c.body
    ]
    rig.fake.unlabel_issue(number, "agent-done")
    lost = {k: v for k, v in (run.result or {}).items() if k != "notified"}
    rig.ledger.update(run.run_id, result_json=lost)
    assert comments_with(rig, number, DONE_HEADING) == []

    actions = rig.worker.recover(execute=True)

    assert_reported(actions, "reported run", run, names_issue=False)
    assert len(comments_with(rig, number, DONE_HEADING)) == 1
    assert "agent-done" in rig.fake.issue_labels(number)
    assert (current(rig, run.run_id).result or {}).get("notified") is True
    assert rig.worker.recover(execute=True) == []
    assert len(comments_with(rig, number, DONE_HEADING)) == 1
    assert len(rig.fake.pulls) == 1 and agent_starts(rig) == 1


# ----------------------------------------------------------------------------------------------
# worktree hygiene
# ----------------------------------------------------------------------------------------------


def withdraw_approval_before_publishing(
    rig: Rig, number: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make the run end CANCELLED right after its commit (the approval is taken back)."""
    original = rig.worker._publish_once

    def withdraw_then_publish(run: RunRecord, payload: dict[str, Any]) -> None:
        rig.fake.unlabel_issue(number, "agent-approved")
        original(run, payload)

    monkeypatch.setattr(rig.worker, "_publish_once", withdraw_then_publish)


def test_a_done_run_removes_its_worktree_and_local_branch_but_keeps_its_logs(rig: Rig) -> None:
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, describe(run)
    assert not Path(run.worktree_path or "").exists()
    assert rig.worktrees() == [] and registered_worktrees(rig) == []
    assert local_agent_branches(rig) == []
    assert run.branch in rig.git_env.origin_branches()  # the pushed branch is what remains
    run_dir = rig.cfg.resolved_state_dir() / "runs" / run.run_id
    assert (run_dir / "prompt.txt").is_file()
    assert (run_dir / "agent" / f"{rig.backend}.stdout.log").is_file()


def test_a_blocked_run_keeps_its_worktree_and_branch_for_inspection(rig: Rig) -> None:
    number = rig.new_task_issue()
    rig.set_scenario(NO_CHANGES)

    run = rig.run_issue(number)

    assert run.status is RunStatus.BLOCKED, describe(run)
    worktree = Path(run.worktree_path or "")
    assert worktree.is_dir() and (worktree / ".git").is_file()
    assert registered_worktrees(rig) == [worktree]
    assert local_agent_branches(rig) == [run.branch]
    assert run.branch not in rig.git_env.origin_branches()


@pytest.mark.parametrize(
    ("policy", "keep_done", "keep_blocked", "keep_cancelled"),
    [
        ("never", False, False, False),
        ("on-failure", False, True, True),
        ("always", True, True, True),
    ],
)
def test_keep_worktrees_policy_is_honoured(
    policy: str,
    keep_done: bool,
    keep_blocked: bool,
    keep_cancelled: bool,
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_git(rig, keep_worktrees=policy)

    done = rig.run_issue(add_approved_issue(rig, "succeeding"))
    assert done.status is RunStatus.DONE, describe(done)

    rig.set_scenario(NO_CHANGES)
    blocked = rig.run_issue(add_approved_issue(rig, "failing"))
    assert blocked.status is RunStatus.BLOCKED, describe(blocked)

    rig.docs_scenario()
    withdrawn = add_approved_issue(rig, "withdrawn")
    withdraw_approval_before_publishing(rig, withdrawn, monkeypatch)
    cancelled = rig.run_issue(withdrawn)
    assert cancelled.status is RunStatus.CANCELLED, describe(cancelled)

    expected = [(done, keep_done), (blocked, keep_blocked), (cancelled, keep_cancelled)]
    for run, kept in expected:
        assert Path(run.worktree_path or "").exists() is kept, (run.status, policy)
        assert (run.branch in local_agent_branches(rig)) is kept, (run.status, policy)
    assert len(registered_worktrees(rig)) == sum(kept for _, kept in expected)


def test_leftovers_of_an_interrupted_attempt_are_replaced_when_the_run_resumes(rig: Rig) -> None:
    number = rig.new_task_issue()
    crashed = crash_mid_agent(rig, number)
    worktree = Path(crashed.worktree_path or "")
    # The dead attempt left a commit of its own on the run's branch and an untracked file.
    (worktree / "docs").mkdir(exist_ok=True)
    (worktree / "docs" / "leftover.md").write_text("committed by the dead attempt\n")
    rig.git_env.git("add", "-A", cwd=worktree)
    rig.git_env.git("commit", "-q", "-m", "stale work", cwd=worktree)
    (worktree / "docs" / "untracked.md").write_text("never committed\n")
    assert rig.git.worktree_head(worktree) != crashed.base_sha
    # The run is already waiting to be resumed, so recovery does not clean up first.
    assert rig.ledger.requeue_interrupted(crashed.run_id, 1)

    report = rig.poll()

    [run] = report.runs
    assert run.run_id == crashed.run_id and run.status is RunStatus.DONE, describe(run)
    assert run.branch == crashed.branch and run.base_sha == crashed.base_sha
    changed = rig.git_env.origin_git("show", "--name-only", "--format=", run.branch or "")
    assert changed.split() == ["docs/agent-note.md"]
    assert rig.git_env.origin_git("rev-parse", f"{run.branch}^") == crashed.base_sha
    assert not worktree.exists() and local_agent_branches(rig) == []


# ----------------------------------------------------------------------------------------------
# the base branch moves between runs
# ----------------------------------------------------------------------------------------------


def test_a_later_run_starts_from_the_advanced_base_branch(rig: Rig) -> None:
    old_base = rig.git_env.git("rev-parse", "origin/main")
    first = rig.run_issue(add_approved_issue(rig, "before"))
    assert first.status is RunStatus.DONE, describe(first)
    assert first.base_sha == old_base

    new_base = advance_origin(rig)
    assert new_base != old_base
    assert rig.git_env.git("rev-parse", "origin/main") == old_base  # not fetched yet

    second = rig.run_issue(add_approved_issue(rig, "after"))

    assert second.status is RunStatus.DONE, describe(second)
    assert second.base_sha == new_base
    assert rig.git_env.git("rev-parse", "origin/main") == new_base
    assert rig.git_env.origin_git("rev-parse", f"{second.branch}^") == new_base
    assert rig.git_env.origin_git("rev-parse", f"{first.branch}^") == old_base
    # The new base is what the agent saw: the second run's tree contains the upstream change.
    upstream = rig.git_env.origin_git("show", f"{second.branch}:docs/upstream.md")
    assert upstream == "upstream"


def test_without_fetch_before_run_the_stale_local_base_is_used(rig: Rig) -> None:
    configure_git(rig, fetch_before_run=False)
    old_base = rig.git_env.git("rev-parse", "origin/main")
    new_base = advance_origin(rig)
    assert new_base != old_base

    run = rig.run_issue(add_approved_issue(rig, "stale"))

    assert run.status is RunStatus.DONE, describe(run)
    assert run.base_sha == old_base
    assert rig.git_env.git("rev-parse", "origin/main") == old_base
    assert rig.git_env.origin_git("rev-parse", f"{run.branch}^") == old_base


# ----------------------------------------------------------------------------------------------
# the serve loop
# ----------------------------------------------------------------------------------------------


def test_serve_runs_an_approved_issue_then_stops_promptly_when_asked(
    make_rig_for: Any, background: Any
) -> None:
    # An hour between polls: only the stop event can end the wait, so it must be honoured at once.
    rig: Rig = make_rig_for(poll={"interval_seconds": 3600})
    number = rig.new_task_issue()
    stop = threading.Event()
    loop = background(rig, lambda: rig.worker.serve(execute=True, stop=stop), stop=stop)

    wait_until(lambda: runs_of(rig, number) and runs_of(rig, number)[0].status.terminal)
    assert loop.alive

    asked = time.monotonic()
    stop.set()
    loop.result(timeout=10)

    assert time.monotonic() - asked < 5
    run = only_run(rig, number)
    assert run.status is RunStatus.DONE, describe(run)
    assert agent_starts(rig) == 1 and len(rig.fake.pulls) == 1


def test_serve_keeps_polling_and_picks_up_issues_approved_later(rig: Rig, background: Any) -> None:
    stop = threading.Event()
    loop = background(rig, lambda: rig.worker.serve(execute=True, stop=stop), stop=stop)
    wait_until(
        lambda: any(r.path.endswith("/issues") for r in list(rig.fake.requests)),
        what="the first poll",
    )
    assert rig.ledger.list_runs() == [] and loop.alive

    number = rig.new_task_issue()
    wait_until(
        lambda: runs_of(rig, number) and runs_of(rig, number)[0].status.terminal,
        what="the later issue to be processed",
    )
    stop.set()
    loop.result(timeout=10)

    assert only_run(rig, number).status is RunStatus.DONE
    assert agent_starts(rig) == 1


class LogCollector(logging.Handler):
    """Collects the records of one logger regardless of how other tests configured logging."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_an_exception_inside_a_poll_cycle_does_not_kill_the_serve_loop(
    make_rig_for: Any, background: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig: Rig = make_rig_for(poll={"interval_seconds": 1, "error_backoff_seconds": 1})
    number = rig.new_task_issue()
    real_poll = rig.worker.poll_once
    calls: list[dict[str, Any]] = []

    def flaky_poll(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("boom in the first poll cycle")
        return real_poll(**kwargs)

    monkeypatch.setattr(rig.worker, "poll_once", flaky_poll)
    collector = LogCollector()
    logger = logging.getLogger("agent_bridge.worker")
    logger.addHandler(collector)
    stop = threading.Event()
    try:
        loop = background(rig, lambda: rig.worker.serve(execute=True, stop=stop), stop=stop)
        wait_until(
            lambda: runs_of(rig, number) and runs_of(rig, number)[0].status.terminal,
            what="the issue to be processed after the failed cycle",
        )
        assert loop.alive  # the failed cycle did not end the loop
        stop.set()
        loop.result(timeout=10)
    finally:
        logger.removeHandler(collector)

    assert len(calls) >= 2 and all(call.get("execute") is True for call in calls)
    assert only_run(rig, number).status is RunStatus.DONE
    failed = [r for r in collector.records if "poll cycle failed" in r.getMessage()]
    assert len(failed) == 1 and failed[0].exc_info is not None
    assert "boom in the first poll cycle" in str(failed[0].exc_info[1])


def test_serve_in_dry_run_mode_never_executes_or_writes(rig: Rig, background: Any) -> None:
    number = rig.new_task_issue()
    stop = threading.Event()
    loop = background(rig, lambda: rig.worker.serve(execute=False, stop=stop), stop=stop)
    wait_until(
        lambda: any(r.path.endswith(f"/issues/{number}/events") for r in list(rig.fake.requests)),
        what="the dry-run poll to look at the issue",
    )
    stop.set()
    loop.result(timeout=10)

    assert rig.ledger.list_runs() == []
    assert rig.fake.writes() == []
    assert rig.invocations() == []
    assert rig.worktrees() == []


def test_stopping_the_worker_mid_run_requeues_the_run_and_the_next_start_resumes_it(
    make_rig_for: Any, background: Any
) -> None:
    rig: Rig = make_rig_for(agent={"cancel_poll_seconds": 0.05})
    number = rig.new_task_issue()
    rig.set_scenario(HANG)
    stop = threading.Event()
    loop = background(rig, lambda: rig.worker.serve(execute=True, stop=stop), stop=stop)
    active = wait_for_agent(rig)
    agent_pid = active.agent_pid
    assert agent_pid is not None

    stop.set()
    loop.result(timeout=20)

    assert not pid_is_alive(agent_pid)
    parked = rig.ledger.get(active.run_id)
    assert parked is not None
    assert parked.status is RunStatus.PENDING, describe(parked)
    assert parked.interrupted_count == 1 and parked.attempt == 2
    assert "agent-running" not in rig.fake.issue_labels(number)
    assert comments_with(rig, number, BLOCKED_HEADING) == []
    assert comments_with(rig, number, CANCELLED_HEADING) == []
    assert rig.fake.pulls == {}
    assert not Path(active.worktree_path or "").exists()

    # The next start of the bridge resumes the very same run.
    rig.docs_scenario()
    worker = restart(rig)
    report = worker.poll_once(execute=True)

    [run] = report.runs
    assert run.run_id == active.run_id and run.status is RunStatus.DONE, describe(run)
    assert [r.run_id for r in runs_of(rig, number)] == [active.run_id]
    assert agent_starts(rig) == 2  # the interrupted attempt and the resumed one
    assert len(comments_with(rig, number, STARTED_HEADING)) == 1
    assert len(rig.fake.pulls) == 1
