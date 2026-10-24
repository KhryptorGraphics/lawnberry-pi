"""What a run may reach and how long it may hold on: remote, report and lease.

Three review findings of the approval/transport/state half of the bridge meet in the worker and
are pinned here, each through the real worker flow (real git origin and clone, in-memory GitHub,
the fake agent CLIs):

* F7: the clone's git remote must be the repository the bridge is configured for, checked before
  every fetch and push, because approvals, labels and pull requests go to the configured
  repository while code is fetched from and pushed to whatever the remote says.
* F8: reporting a finished run to GitHub is bounded and never claims success it did not have.
* F9: a healthy run's lease must outlive a missed abort poll (the abort check is the heartbeat).

Every test runs once per backend (``omp`` and ``claude``).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pydantic
import pytest
from bridgefakes.rig import REPO, Rig

from tools.agent_bridge.config import LimitsConfig
from tools.agent_bridge.errors import (
    GitHubConflict,
    GitHubError,
    GitHubNotFound,
    GitHubPermissionError,
    GitHubUnavailable,
)
from tools.agent_bridge.ledger import RunRecord
from tools.agent_bridge.models import RunStatus
from tools.agent_bridge.publisher import Publisher
from tools.agent_bridge.worker import _Stop


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def describe(run: RunRecord | None) -> str:
    assert run is not None
    return f"{run.status.value}/{run.phase.value} {run.failure_kind}: {run.failure_detail}"


def blocked_comments(rig: Rig, number: int) -> list[str]:
    return [body for body in rig.comments(number) if "Agent run blocked" in body]


class SimulatedCrash(BaseException):
    """The bridge process dying: nothing in the worker catches ``BaseException``."""


# --------------------------------------------------------------------------------------------
# F7: the git remote must be the configured repository
# --------------------------------------------------------------------------------------------
def point_origin_at(rig: Rig, url: str) -> None:
    """Give the clone's ``origin`` another URL; ``git remote get-url`` reads it back."""
    rig.git_env.git("remote", "set-url", "origin", url)


@pytest.fixture
def git_subcommands(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The subcommand (``fetch``, ``push``, ...) of every ``git`` process started from now on."""
    seen: list[str] = []
    original = subprocess.Popen.__init__

    def recording(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
        argv = args[0] if args else kwargs.get("args")
        if isinstance(argv, list | tuple) and argv and Path(str(argv[0])).name == "git":
            rest = [str(arg) for arg in argv[1:]]
            while rest and rest[0] in {"-c", "-C"}:
                del rest[:2]
            seen.append(rest[0] if rest else "")
        original(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", recording)
    return seen


class ReportedRemote:
    """What the clone's remote reads back as, while fetch and push keep using the real origin.

    The check compares the URLs git reports for the remote (``fetch``, and ``push`` for every
    push URL) with the configured repository; the transport underneath is untouched, so a run
    that passes the check really fetches from and pushes to the rig's local origin.
    """

    def __init__(
        self,
        rig: Rig,
        monkeypatch: pytest.MonkeyPatch,
        fetch: str,
        push: list[str] | None = None,
    ) -> None:
        self.fetch = fetch
        self.push = push
        self.reads: list[str] = []
        monkeypatch.setattr(rig.git, "remote_url", self._remote_url)
        monkeypatch.setattr(rig.git, "push_urls", self._push_urls)

    def _remote_url(self) -> str:
        self.reads.append("fetch")
        return self.fetch

    def _push_urls(self) -> list[str]:
        self.reads.append("push")
        return list(self.push) if self.push is not None else [self.fetch]


# The test API is https://api.github.test, so the repository lives on github.test.
FOREIGN_REMOTES = {
    "a-fork": "https://github.test/fork-owner/repo.git",
    "another-repository-of-the-owner": "https://github.test/owner/another-repo.git",
    "the-right-name-on-another-host": "https://gitlab.test/owner/repo.git",
    "a-fork-over-scp-ssh": "git@github.test:fork-owner/repo.git",
    "another-host-over-ssh": "ssh://git@gitlab.test/owner/repo.git",
}


@pytest.mark.parametrize("url", FOREIGN_REMOTES.values(), ids=FOREIGN_REMOTES.keys())
def test_a_clone_whose_remote_is_not_the_configured_repository_is_refused_before_any_git_traffic(
    rig: Rig, git_subcommands: list[str], url: str
) -> None:
    number = rig.new_task_issue()
    point_origin_at(rig, url)
    git_subcommands.clear()

    run = rig.run_issue(number)

    assert not {"fetch", "push"} & set(git_subcommands), git_subcommands
    assert run.status is RunStatus.BLOCKED, describe(run)
    assert run.failure_kind == "remote_mismatch"
    detail = run.failure_detail or ""
    assert "owner/repo" in detail and "github.test" in detail
    assert rig.invocations() == [] and rig.worktrees() == []
    assert rig.git_env.origin_branches() == ["main"] and rig.fake.pulls == {}
    labels = set(rig.fake.issue_labels(number))
    assert "agent-blocked" in labels and "agent-running" not in labels
    [comment] = blocked_comments(rig, number)
    assert "`remote_mismatch`" in comment
    assert run.result is not None and run.result["notified"] is True


def test_credentials_embedded_in_the_remote_never_reach_the_issue_the_ledger_or_the_log(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    secret, user = "S3cr3tPassw0rd_9z", "deploy-alice"
    number = rig.new_task_issue()
    point_origin_at(rig, f"https://{user}:{secret}@gitlab.test/owner/repo.git")

    with caplog.at_level(logging.DEBUG, logger="agent_bridge"):
        run = rig.run_issue(number)

    assert run.failure_kind == "remote_mismatch", describe(run)
    assert "gitlab.test/owner/repo" in (run.failure_detail or "")
    everything = "\n".join(
        [
            run.failure_detail or "",
            json.dumps(run.result),
            json.dumps(rig.ledger.events(run_id=run.run_id)),
            *rig.comments(number),
            caplog.text,
        ]
    )
    assert secret not in everything and user not in everything


OWN_REMOTES = {
    "https": "https://github.test/owner/repo",
    "https-git-suffix": "https://github.test/owner/repo.git",
    "https-other-case": "https://GITHUB.TEST/Owner/REPO.git",
    "ssh-url": "ssh://git@github.test/owner/repo.git",
    "scp-like": "git@github.test:owner/repo.git",
    "credentials": "https://x-access-token:tok3n_9z@github.test/owner/repo.git",
}


@pytest.mark.parametrize("url", OWN_REMOTES.values(), ids=OWN_REMOTES.keys())
def test_every_spelling_of_the_configured_repository_runs_to_completion(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, url: str
) -> None:
    remote = ReportedRemote(rig, monkeypatch, url)
    number = rig.new_task_issue()

    with caplog.at_level(logging.DEBUG, logger="agent_bridge"):
        run = rig.run_issue(number)

    assert {"fetch", "push"} <= set(remote.reads)  # both sides were looked at
    assert run.status is RunStatus.DONE, describe(run)
    assert len(rig.fake.pulls) == 1
    assert "tok3n_9z" not in caplog.text
    assert not [
        r for r in caplog.records if r.levelno >= logging.WARNING and "remote" in r.getMessage()
    ]


def test_a_local_path_remote_is_allowed_with_a_single_warning(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    # The rigs (and anyone mirroring locally) have a filesystem path as origin: it cannot be
    # compared with a GitHub repository, so it is allowed, loudly.
    number = rig.new_task_issue()

    with caplog.at_level(logging.WARNING, logger="agent_bridge"):
        run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, describe(run)
    warnings = [
        record
        for record in caplog.records
        if record.name == "agent_bridge.worker"
        and record.levelno == logging.WARNING
        and "remote" in record.getMessage()
        and "local" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert str(rig.git_env.origin) in warnings[0].getMessage()


def test_the_remote_is_read_again_for_every_run(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = ReportedRemote(rig, monkeypatch, "https://github.test/owner/repo.git")
    first = rig.run_issue(
        rig.new_task_issue(objective="Document the first topic in docs/guide.md.")
    )
    assert first.status is RunStatus.DONE, describe(first)

    remote.fetch = "https://github.test/fork-owner/repo.git"  # someone repoints the clone
    second = rig.run_issue(
        rig.new_task_issue(objective="Document the second topic in docs/guide.md.")
    )

    assert second.status is RunStatus.BLOCKED and second.failure_kind == "remote_mismatch"
    assert len(rig.fake.pulls) == 1  # only the first run published anything


def test_the_remote_is_checked_even_when_fetching_is_switched_off(
    rig: Rig, git_subcommands: list[str]
) -> None:
    # No fetch is not "no remote": the branch is still pushed there.
    rig.cfg.git.fetch_before_run = False
    number = rig.new_task_issue()
    point_origin_at(rig, "https://github.test/fork-owner/repo.git")
    git_subcommands.clear()

    run = rig.run_issue(number)

    assert "push" not in git_subcommands, git_subcommands
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "remote_mismatch", describe(run)
    assert rig.invocations() == []


def test_a_publication_resumed_after_a_crash_checks_the_remote_before_pushing(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, git_subcommands: list[str]
) -> None:
    number = rig.new_task_issue()
    monkeypatch.setattr(
        rig.worker, "_publish", lambda *a, **k: (_ for _ in ()).throw(SimulatedCrash("publish"))
    )
    with pytest.raises(SimulatedCrash):
        rig.poll()
    monkeypatch.undo()  # the replacement process has no such hook
    [crashed] = rig.ledger.runs_for_issue(REPO, number)
    assert crashed.status is RunStatus.PUBLISHING and crashed.commit_sha

    # The operator repoints the clone while the bridge is down.
    point_origin_at(rig, "https://github.test/fork-owner/repo.git")
    rig.clock.advance(rig.cfg.limits.lease_ttl_seconds + 1)
    git_subcommands.clear()
    actions = rig.worker.recover(execute=True)

    assert "push" not in git_subcommands, git_subcommands
    run = rig.ledger.get(crashed.run_id)
    assert run is not None
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "remote_mismatch", describe(run)
    assert len(actions) == 1 and "ended blocked" in actions[0], actions
    assert rig.git_env.origin_branches() == ["main"] and rig.fake.pulls == {}
    assert "agent-blocked" in rig.fake.issue_labels(number)


FORK = "https://github.test/fork-owner/repo.git"


def redirect_pushes(rig: Rig, how: str) -> None:
    """Leave fetches on the real origin and send pushes to a fork, in each way git allows."""
    origin = str(rig.git_env.origin)
    if how == "pushurl":
        rig.git_env.git("config", "remote.origin.pushurl", FORK)
    elif how == "push-instead-of":
        rig.git_env.git("config", f"url.{FORK}.pushInsteadOf", origin)
    elif how == "second-pushurl":  # git pushes to every pushurl: the real origin and the fork
        rig.git_env.git("config", "remote.origin.pushurl", origin)
        rig.git_env.git("config", "--add", "remote.origin.pushurl", FORK)
    else:  # a second url: fetches use the first one, pushes go to both
        rig.git_env.git("remote", "set-url", "--add", "origin", FORK)


@pytest.mark.parametrize("how", ["pushurl", "push-instead-of", "second-pushurl", "second-url"])
def test_a_remote_that_pushes_somewhere_else_is_refused_before_any_git_traffic(
    rig: Rig, git_subcommands: list[str], how: str
) -> None:
    # The fetch URL is the (allowed) local origin; only the push side names another repository.
    number = rig.new_task_issue()
    redirect_pushes(rig, how)
    git_subcommands.clear()

    run = rig.run_issue(number)

    assert not {"fetch", "push"} & set(git_subcommands), git_subcommands
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "remote_mismatch", describe(run)
    assert "pushes to github.test/fork-owner/repo" in (run.failure_detail or "")
    assert rig.invocations() == [] and rig.worktrees() == []
    assert rig.git_env.origin_branches() == ["main"] and rig.fake.pulls == {}


def test_a_push_url_in_another_spelling_of_the_configured_repository_is_accepted(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    ReportedRemote(
        rig, monkeypatch, "https://github.test/owner/repo.git", ["git@github.test:owner/repo.git"]
    )

    run = rig.run_issue(rig.new_task_issue())

    assert run.status is RunStatus.DONE, describe(run)


def test_the_push_urls_are_checked_again_right_before_the_push(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, git_subcommands: list[str]
) -> None:
    number = rig.new_task_issue()
    real_verify_and_commit = rig.worker._verify_and_commit

    def commit_then_repoint(ctx: Any) -> None:
        real_verify_and_commit(ctx)
        # The start-of-run check is long past: something edits the shared git config now.
        rig.git_env.git("config", "remote.origin.pushurl", FORK)

    monkeypatch.setattr(rig.worker, "_verify_and_commit", commit_then_repoint)
    git_subcommands.clear()

    run = rig.run_issue(number)

    assert "push" not in git_subcommands, git_subcommands
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "remote_mismatch", describe(run)
    assert "pushes to github.test/fork-owner/repo" in (run.failure_detail or "")
    assert rig.git_env.origin_branches() == ["main"] and rig.fake.pulls == {}
    assert run.commit_sha  # the change is kept locally


# --------------------------------------------------------------------------------------------
# F8: reporting a finished run is bounded and tells the truth
# --------------------------------------------------------------------------------------------
NO_CHANGES = [{"op": "final", "text": "Nothing needed changing."}]  # BLOCKED ``no_changes``
ATTEMPT_LIMIT = 5  # the default of ``limits.max_notify_attempts``


def current(rig: Rig, run: RunRecord) -> RunRecord:
    found = rig.ledger.get(run.run_id)
    assert found is not None
    return found


def events_of(rig: Rig, run: RunRecord, kind: str) -> list[dict[str, Any]]:
    return [e for e in rig.ledger.events(run_id=run.run_id) if e["kind"] == kind]


class FailingReports:
    """The real publisher, except that every report of a finished run raises ``error``."""

    def __init__(self, real: Publisher, error: GitHubError) -> None:
        self._real = real
        self.error = error
        self.reports = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def comment_once(self, issue_number: int, key: str, body: str) -> bool:
        if key.endswith(("-done", "-blocked", "-cancelled")):
            self.reports += 1
            raise self.error
        return self._real.comment_once(issue_number, key, body)


GONE = "issue not found (404)"
SPENT = "attempt limit reached"
DELIVERY_ERRORS = {
    # The issue is gone: there is nothing left to report to, so the first 404 ends it.
    "not-found": (GitHubNotFound("GitHub object not found: Not Found", status=404), 1, GONE),
    # Everything else may heal (permissions get fixed, outages end), so it gets the full limit.
    "permission": (
        GitHubPermissionError("GitHub denied the request", status=403),
        ATTEMPT_LIMIT,
        SPENT,
    ),
    "unavailable": (GitHubUnavailable("connection reset"), ATTEMPT_LIMIT, SPENT),
    "rejected": (GitHubConflict("GitHub rejected the request", status=422), ATTEMPT_LIMIT, SPENT),
    "other": (GitHubError("GitHub error HTTP 500: boom", status=500), ATTEMPT_LIMIT, SPENT),
}


@pytest.mark.parametrize(
    ("error", "calls", "reason"), DELIVERY_ERRORS.values(), ids=DELIVERY_ERRORS.keys()
)
def test_a_report_that_cannot_be_delivered_is_attempted_a_bounded_number_of_times(
    rig: Rig, error: GitHubError, calls: int, reason: str
) -> None:
    stub = FailingReports(rig.worker.publisher, error)
    rig.worker.publisher = stub  # type: ignore[assignment]
    rig.set_scenario(NO_CHANGES)
    run = rig.run_issue(rig.new_task_issue())
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "no_changes", describe(run)

    for _ in range(ATTEMPT_LIMIT + 3):  # many polls: the loop must not go on forever
        rig.worker.recover(execute=True)

    assert stub.reports == calls
    stored = current(rig, run)
    assert stored.result is not None
    assert stored.result["notified"] == "gave_up"
    assert stored.result["notify_gave_up"] == reason
    assert stored.result["notify_attempts"] == calls
    assert len(events_of(rig, run, "notify.failed")) == calls
    [gave_up] = events_of(rig, run, "notify.gave_up")
    assert gave_up["detail"]["reason"] == reason


def test_recover_only_claims_a_report_that_was_delivered(rig: Rig) -> None:
    rig.fake.deny("POST", "/comments")
    number = rig.new_task_issue()
    run = rig.run_issue(number)  # even "started" could not be posted: blocked, and not told
    short = run.run_id[:12]
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "github_permission"
    assert run.result is not None and run.result["notify_attempts"] == 1
    assert not run.result.get("notified")

    assert rig.worker.recover(execute=True) == [f"could not report run {short} (attempt 2 of 5)"]
    assert rig.worker.recover(execute=True) == [f"could not report run {short} (attempt 3 of 5)"]
    assert "agent-blocked" not in rig.fake.issue_labels(number)

    rig.fake.allow("POST", "/comments")  # a transient fault: no reason to have given up
    assert rig.worker.recover(execute=True) == [f"reported run {short}"]

    healed = current(rig, run)
    assert healed.result is not None and healed.result["notified"] is True
    labels = set(rig.fake.issue_labels(number))
    assert "agent-blocked" in labels and "agent-running" not in labels
    assert len(blocked_comments(rig, number)) == 1
    assert len(events_of(rig, run, "notify.failed")) == 3
    assert events_of(rig, run, "notify.gave_up") == []
    assert rig.worker.recover(execute=True) == []


def test_a_report_that_keeps_failing_is_given_up_after_the_attempt_limit(rig: Rig) -> None:
    rig.fake.deny("POST", "/comments")
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    short = run.run_id[:12]

    for attempt in range(2, ATTEMPT_LIMIT):
        expected = [f"could not report run {short} (attempt {attempt} of {ATTEMPT_LIMIT})"]
        assert rig.worker.recover(execute=True) == expected
    [last] = rig.worker.recover(execute=True)
    assert last == f"could not report run {short} (attempt 5 of 5); giving up: {SPENT}"

    rig.fake.allow("POST", "/comments")  # a healthy GitHub is not asked again either
    rig.fake.reset_requests()
    for _ in range(3):
        assert rig.worker.recover(execute=True) == []
    assert rig.fake.requests == []
    stored = current(rig, run)
    assert stored.result is not None
    assert stored.result["notified"] == "gave_up" and stored.result["notify_attempts"] == 5
    assert len(events_of(rig, run, "notify.failed")) == 5
    [gave_up] = events_of(rig, run, "notify.gave_up")
    assert gave_up["detail"]["attempts"] == 5
    assert "agent-blocked" not in rig.fake.issue_labels(number)  # the ledger says what happened


def test_a_report_to_an_issue_that_is_gone_is_given_up_at_once_and_says_so(rig: Rig) -> None:
    rig.set_scenario(NO_CHANGES)
    number = rig.new_task_issue()
    run = rig.run_issue(number)  # blocked and reported
    assert run.result is not None and run.result["notified"] is True
    lost = {k: v for k, v in run.result.items() if k != "notified"}
    rig.ledger.update(run.run_id, result_json=lost)  # the delivery was never recorded
    gone = FailingReports(rig.worker.publisher, GitHubNotFound("not found", status=404))
    rig.worker.publisher = gone  # type: ignore[assignment]

    [action] = rig.worker.recover(execute=True)  # the issue was deleted meanwhile

    assert action == f"could not report run {run.run_id[:12]} (attempt 1 of 5); giving up: {GONE}"
    for _ in range(3):
        assert rig.worker.recover(execute=True) == []
    assert gone.reports == 1


def test_a_done_run_whose_report_keeps_failing_is_reported_truthfully_and_given_up(
    rig: Rig,
) -> None:
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    assert run.status is RunStatus.DONE, describe(run)
    short = run.run_id[:12]
    assert run.result is not None
    lost = {k: v for k, v in run.result.items() if k != "notified"}
    rig.ledger.update(run.run_id, result_json=lost)  # the completion was never recorded as sent
    rig.fake.comments[number] = [  # ...and the done comment is gone from the issue
        c for c in rig.fake.comments[number] if f"run-{run.run_id}-done" not in c.body
    ]
    rig.fake.deny("POST", "/comments")

    for attempt in range(1, ATTEMPT_LIMIT):
        expected = [f"could not report run {short} (attempt {attempt} of {ATTEMPT_LIMIT})"]
        assert rig.worker.recover(execute=True) == expected
    [last] = rig.worker.recover(execute=True)
    assert last == f"could not report run {short} (attempt 5 of 5); giving up: {SPENT}"

    rig.fake.allow("POST", "/comments")
    rig.fake.reset_requests()
    assert rig.worker.recover(execute=True) == [] and rig.fake.requests == []
    stored = current(rig, run)
    assert stored.result is not None and stored.result["notified"] == "gave_up"
    assert len(events_of(rig, run, "notify.failed")) == ATTEMPT_LIMIT
    assert len(events_of(rig, run, "notify.gave_up")) == 1


def test_the_attempt_limit_is_configurable(make_rig_for: Any) -> None:
    rig: Rig = make_rig_for(limits={"max_notify_attempts": 2})
    rig.fake.deny("POST", "/comments")
    run = rig.run_issue(rig.new_task_issue())

    [action] = rig.worker.recover(execute=True)

    assert action == f"could not report run {run.run_id[:12]} (attempt 2 of 2); giving up: {SPENT}"
    assert rig.worker.recover(execute=True) == []


def test_notify_terminal_returns_true_only_when_comment_labels_and_flag_all_landed(
    rig: Rig,
) -> None:
    rig.fake.deny("POST", "/labels")  # the comment goes through, the label does not
    number = rig.new_task_issue()
    run = rig.run_issue(number)
    assert run.status is RunStatus.BLOCKED and run.failure_kind == "github_permission"
    stop = _Stop(run.status, run.failure_kind or "", run.failure_detail or "")
    payload = current(rig, run).result or {}

    assert rig.worker._notify_terminal(current(rig, run), stop, payload) is False
    assert not (current(rig, run).result or {}).get("notified")

    rig.fake.allow("POST", "/labels")
    assert rig.worker._notify_terminal(current(rig, run), stop, payload) is True
    assert (current(rig, run).result or {})["notified"] is True
    assert "agent-blocked" in rig.fake.issue_labels(number)


def test_the_notify_attempt_limit_defaults_to_five_and_is_bounded() -> None:
    assert LimitsConfig().max_notify_attempts == ATTEMPT_LIMIT
    for allowed in (1, 50):
        assert LimitsConfig(max_notify_attempts=allowed).max_notify_attempts == allowed
    for refused in (0, -1, 51):
        with pytest.raises(pydantic.ValidationError):
            LimitsConfig(max_notify_attempts=refused)
