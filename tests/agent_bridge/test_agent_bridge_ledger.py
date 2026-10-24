"""Tests for the durable SQLite run ledger.

Covers the database file (schema, WAL, permissions, reopening), the idempotent ``lease``
protocol and every denial reason, the recovery primitives (heartbeat, stale runs, requeue,
publishing claims), cancellation flags, the event log and one-shot notices, atomicity of the
multi-statement transitions, and real concurrency: threads sharing or not sharing a connection
and separate OS processes racing on one database file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tools.agent_bridge import ledger as ledger_module
from tools.agent_bridge.ledger import (
    LeaseRequest,
    Ledger,
    RunRecord,
    dedup_key,
    iso,
)
from tools.agent_bridge.models import RunPhase, RunStatus

REPO = "owner/repo"
REPO_ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
TTL = 60
TERMINAL = [RunStatus.DONE, RunStatus.BLOCKED, RunStatus.CANCELLED]


# ------------------------------------------------------------------------------------ helpers
class FakeClock:
    """Controllable ledger clock."""

    def __init__(self, start: datetime = T0) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def make_request(**overrides: Any) -> LeaseRequest:
    """A valid lease request for issue 7 / approval 1001; any field can be overridden."""
    fields: dict[str, Any] = {
        "repo": REPO,
        "issue_number": 7,
        "issue_title": "Fix the docs",
        "revision": "rev-a",
        "spec_hash": "spec-a",
        "approval_event_id": 1001,
        "approved_by": "owner",
        "approved_at": "2026-03-01T11:59:00Z",
        "backend": "omp",
        "worker_id": "worker-1",
        "worker_pid": 4242,
        "ttl_seconds": TTL,
        "max_attempts_per_issue": 3,
        "dedupe_by_content": True,
    }
    fields.update(overrides)
    return LeaseRequest(**fields)


def other_issue(number: int, **overrides: Any) -> LeaseRequest:
    """A request for another issue, with its own approval event and spec hash."""
    fields: dict[str, Any] = {
        "issue_number": number,
        "approval_event_id": 1000 + number,
        "spec_hash": f"spec-{number}",
        "revision": f"rev-{number}",
    }
    fields.update(overrides)
    return make_request(**fields)


def grant(ledger: Ledger, request: LeaseRequest | None = None) -> RunRecord:
    """Lease and return the granted run (fails the test when the lease is denied)."""
    result = ledger.lease(request or make_request())
    assert result.granted, (result.reason, result.other)
    assert result.run is not None
    return result.run


def kinds(ledger: Ledger, **filters: Any) -> list[str]:
    return [event["kind"] for event in ledger.events(**filters)]


def state_of(ledger: Ledger) -> tuple[list[RunRecord], list[dict[str, Any]]]:
    """Every run and every event: equal snapshots mean the ledger was not touched."""
    return ledger.list_runs(limit=10_000), ledger.events(limit=100_000)


def sql_rows(path: Path, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Read the database through a separate connection, independent of ``Ledger``."""
    cx = sqlite3.connect(path, timeout=30)
    try:
        return cx.execute(query, params).fetchall()
    finally:
        cx.close()


def sql_exec(path: Path, statement: str) -> None:
    cx = sqlite3.connect(path, isolation_level=None, timeout=30)
    try:
        cx.execute(statement)
    finally:
        cx.close()


def raw_insert_run(path: Path, **overrides: Any) -> None:
    """Insert a run row behind the ledger's back (to probe the schema constraints)."""
    values: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "repo": REPO,
        "issue_number": 99,
        "issue_title": "raw",
        "revision": "r",
        "spec_hash": "s",
        "approval_event_id": 9999,
        "approved_by": "owner",
        "approved_at": "2026-03-01T00:00:00Z",
        "dedup_key": uuid.uuid4().hex,
        "backend": "omp",
        "status": "running",
        "phase": "leased",
        "created_at": "2026-03-01T00:00:00Z",
        "updated_at": "2026-03-01T00:00:00Z",
    }
    values.update(overrides)
    marks = ", ".join("?" * len(values))
    cx = sqlite3.connect(path, timeout=30)
    try:
        cx.execute(
            f"INSERT INTO runs ({', '.join(values)}) VALUES ({marks})", tuple(values.values())
        )
        cx.commit()
    finally:
        cx.close()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "ledger.db"


@pytest.fixture
def ledger(db_path: Path, clock: FakeClock) -> Iterator[Ledger]:
    instance = Ledger(db_path, clock=clock)
    try:
        yield instance
    finally:
        instance.close()


# ------------------------------------------------------------------------------ database file
def test_new_database_has_schema_wal_mode_and_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / "state" / "nested" / "ledger.db"
    previous = os.umask(0)  # a permissive umask must not leak into the ledger's file mode
    try:
        Ledger(path, clock=FakeClock()).close()
    finally:
        os.umask(previous)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ledger_module.SCHEMA_VERSION >= 1
    assert sql_rows(path, "PRAGMA user_version") == [(ledger_module.SCHEMA_VERSION,)]
    assert sql_rows(path, "PRAGMA journal_mode") == [("wal",)]
    tables = {n for (n,) in sql_rows(path, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "events", "notices"} <= tables


def test_reopening_keeps_runs_events_and_notices(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "ledger.db"
    first = Ledger(path, clock=clock)
    run = grant(first)
    first.set_phase(run.run_id, RunPhase.AGENT, branch="agent/issue-7")
    first.finish(run.run_id, RunStatus.DONE, result={"pr": 12})
    assert first.notice_once("comment-1") is True
    expected = first.get(run.run_id)
    expected_kinds = kinds(first, run_id=run.run_id)
    first.close()

    second = Ledger(path, clock=clock)
    try:
        assert second.get(run.run_id) == expected
        assert kinds(second, run_id=run.run_id) == expected_kinds
        assert second.notice_once("comment-1") is False
        assert second.runs_for_issue(REPO, 7) == [expected]
        assert second.lease(make_request()).reason == "already-processed"
        assert sql_rows(path, "PRAGMA user_version") == [(ledger_module.SCHEMA_VERSION,)]
    finally:
        second.close()


def test_reopening_with_an_unset_schema_version_reapplies_it_without_losing_rows(
    tmp_path: Path, clock: FakeClock
) -> None:
    """A crash between creating the tables and stamping ``user_version`` must be survivable."""
    path = tmp_path / "ledger.db"
    first = Ledger(path, clock=clock)
    run = grant(first)
    first.close()
    sql_exec(path, "PRAGMA user_version=0")

    second = Ledger(path, clock=clock)
    try:
        assert second.get(run.run_id) is not None
        assert second.lease(make_request()).reason == "duplicate-active"
    finally:
        second.close()
    assert sql_rows(path, "PRAGMA user_version") == [(ledger_module.SCHEMA_VERSION,)]


def test_refuses_a_database_written_by_a_newer_schema(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "ledger.db"
    first = Ledger(path, clock=clock)
    grant(first)
    first.close()
    newer = ledger_module.SCHEMA_VERSION + 1
    sql_exec(path, f"PRAGMA user_version={newer}")

    with pytest.raises(RuntimeError, match=rf"schema v{newer}"):
        Ledger(path, clock=clock)

    assert sql_rows(path, "PRAGMA user_version") == [(newer,)]  # left exactly as found
    assert sql_rows(path, "SELECT COUNT(*) FROM runs") == [(1,)]


@pytest.mark.parametrize(
    "clash",
    [
        pytest.param(lambda run: {"run_id": run.run_id}, id="run-id"),
        pytest.param(lambda run: {"dedup_key": run.dedup_key}, id="dedup-key"),
        pytest.param(
            lambda run: {
                "repo": run.repo,
                "issue_number": run.issue_number,
                "approval_event_id": run.approval_event_id,
            },
            id="repo-issue-approval",
        ),
    ],
)
def test_database_rejects_a_second_row_for_the_same_identity(
    ledger: Ledger, db_path: Path, clash: Callable[[RunRecord], dict[str, Any]]
) -> None:
    """Idempotence does not only rely on ``lease``: the schema itself forbids duplicates."""
    run = grant(ledger)
    raw_insert_run(db_path)  # sanity: an unrelated row is accepted
    assert len(ledger.list_runs()) == 2

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        raw_insert_run(db_path, **clash(run))

    assert len(ledger.list_runs()) == 2


def test_default_clock_is_timezone_aware_utc_wall_time(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        before = datetime.now(UTC).replace(microsecond=0)
        run = grant(ledger)
        after = datetime.now(UTC)
        stamp = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        assert before <= stamp <= after
        assert ledger.now().tzinfo is not None
    finally:
        ledger.close()


def test_iso_renders_whole_second_utc_timestamps() -> None:
    assert iso(datetime(2026, 1, 2, 3, 4, 5, 999999, tzinfo=UTC)) == "2026-01-02T03:04:05Z"
    plus_two = timezone(timedelta(hours=2))
    assert iso(datetime(2026, 1, 2, 3, 4, 5, tzinfo=plus_two)) == "2026-01-02T01:04:05Z"


def test_dedup_key_is_the_sha256_of_repo_issue_approval_and_revision() -> None:
    key = dedup_key("owner/repo", 5, 77, "rev")
    assert key == hashlib.sha256(b"owner/repo|5|77|rev").hexdigest()
    assert len(key) == 64
    variants = [
        dedup_key("other/repo", 5, 77, "rev"),
        dedup_key("owner/repo", 6, 77, "rev"),
        dedup_key("owner/repo", 5, 78, "rev"),
        dedup_key("owner/repo", 5, 77, "rev2"),
    ]
    assert len({key, *variants}) == 5


# ------------------------------------------------------------------------------------- lease
def test_first_lease_creates_a_fully_populated_running_run(ledger: Ledger) -> None:
    result = ledger.lease(make_request(issue_title="x" * 500))

    assert result.granted is True and bool(result) is True
    assert result.reason is None and result.other is None
    run = result.run
    assert run is not None
    assert run.status is RunStatus.RUNNING and run.phase is RunPhase.LEASED
    assert (run.repo, run.issue_number) == (REPO, 7)
    assert run.issue_title == "x" * 300  # titles are bounded
    assert (run.revision, run.spec_hash) == ("rev-a", "spec-a")
    assert (run.approval_event_id, run.approved_by) == (1001, "owner")
    assert run.approved_at == "2026-03-01T11:59:00Z"
    assert (run.backend, run.worker_id, run.worker_pid) == ("omp", "worker-1", 4242)
    assert (run.attempt, run.interrupted_count, run.publish_attempts) == (1, 0, 0)
    assert run.cancel_requested is False
    stamp = iso(T0)
    assert run.created_at == run.updated_at == run.started_at == run.heartbeat_at == stamp
    assert run.lease_expires_at == iso(T0 + timedelta(seconds=TTL))
    assert run.finished_at is None and run.result is None
    assert (run.failure_kind, run.failure_detail) == (None, None)
    assert (run.branch, run.worktree_path, run.base_sha, run.commit_sha) == (None,) * 4
    assert (run.pr_number, run.pr_url, run.agent_pid, run.agent_start_ticks) == (None,) * 4
    assert run.dedup_key == dedup_key(REPO, 7, 1001, "rev-a")
    assert len(run.run_id) == 32 and int(run.run_id, 16) >= 0

    assert ledger.get(run.run_id) == run
    assert ledger.find_by_approval(REPO, 7, 1001) == run
    assert ledger.runs_for_issue(REPO, 7) == [run]
    assert ledger.active_runs() == [run]
    assert ledger.pending_runs() == []
    assert kinds(ledger) == ["lease.granted"]
    assert ledger.events()[0]["detail"] == {"approval_event_id": 1001, "backend": "omp"}


@pytest.mark.parametrize(
    ("elapsed", "reason"),
    [
        pytest.param(0, "duplicate-active", id="immediately"),
        pytest.param(TTL - 1, "duplicate-active", id="one-second-before-expiry"),
        pytest.param(TTL, "recovery-pending", id="exactly-at-expiry"),
        pytest.param(TTL + 1, "recovery-pending", id="after-expiry"),
        pytest.param(10 * TTL, "recovery-pending", id="long-after-expiry"),
    ],
)
def test_second_lease_for_one_approval_is_denied_by_lease_liveness(
    ledger: Ledger, clock: FakeClock, elapsed: int, reason: str
) -> None:
    run = grant(ledger)
    clock.advance(elapsed)
    before = state_of(ledger)

    denied = ledger.lease(make_request(worker_id="worker-2", worker_pid=9999))

    assert denied.granted is False and not denied
    assert denied.reason == reason
    assert denied.run is None
    assert denied.other is not None and denied.other.run_id == run.run_id
    assert state_of(ledger) == before  # a denial never touches the ledger
    held = ledger.get(run.run_id)
    assert held is not None and (held.worker_id, held.worker_pid) == ("worker-1", 4242)


@pytest.mark.parametrize(
    ("elapsed", "reason"), [(TTL - 1, "duplicate-active"), (TTL, "recovery-pending")]
)
def test_publishing_runs_are_guarded_like_running_ones(
    ledger: Ledger, clock: FakeClock, elapsed: int, reason: str
) -> None:
    run = grant(ledger)
    ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING, commit_sha="c" * 40)
    clock.advance(elapsed)

    denied = ledger.lease(make_request())

    assert denied.reason == reason
    assert denied.other is not None and denied.other.status is RunStatus.PUBLISHING


@pytest.mark.parametrize("status", TERMINAL, ids=lambda s: s.value)
def test_second_lease_for_a_processed_approval_is_denied_forever(
    ledger: Ledger, clock: FakeClock, status: RunStatus
) -> None:
    run = grant(ledger)
    ledger.finish(run.run_id, status, failure_kind="x" if status is not RunStatus.DONE else None)
    before = state_of(ledger)

    for waited in (0, TTL * 100):
        clock.advance(waited)
        denied = ledger.lease(make_request(worker_id="worker-2"))
        assert not denied
        assert denied.reason == "already-processed"
        assert denied.other is not None and denied.other.run_id == run.run_id
        assert denied.other.status is status

    assert len(ledger.list_runs()) == 1
    assert len(ledger.events()) == len(before[1])


def test_an_already_processed_approval_is_reported_even_while_another_run_is_active(
    ledger: Ledger,
) -> None:
    done = grant(ledger)
    ledger.finish(done.run_id, RunStatus.DONE)
    busy = grant(ledger, other_issue(8))

    denied = ledger.lease(make_request())

    assert denied.reason == "already-processed"
    assert denied.other is not None and denied.other.run_id == done.run_id
    assert ledger.active_runs() == [ledger.get(busy.run_id)]


@pytest.mark.parametrize("status", TERMINAL, ids=lambda s: s.value)
def test_a_new_approval_event_on_the_same_issue_is_a_new_run(
    ledger: Ledger, status: RunStatus
) -> None:
    first = grant(ledger)
    ledger.finish(first.run_id, status)

    second = grant(ledger, make_request(approval_event_id=1002, revision="rev-b"))

    assert second.run_id != first.run_id
    assert second.approval_event_id == 1002
    assert second.attempt == 1 and second.interrupted_count == 0
    assert second.dedup_key != first.dedup_key
    assert second.dedup_key == dedup_key(REPO, 7, 1002, "rev-b")
    assert [r.run_id for r in ledger.runs_for_issue(REPO, 7)] == [first.run_id, second.run_id]
    assert ledger.get(first.run_id).status is status  # the old run is untouched


def test_identical_issue_and_approval_numbers_in_another_repository_are_independent(
    ledger: Ledger,
) -> None:
    first = grant(ledger)
    ledger.finish(first.run_id, RunStatus.DONE)

    second = grant(ledger, make_request(repo="other/repo"))

    assert second.run_id != first.run_id and second.dedup_key != first.dedup_key
    assert ledger.find_by_approval("other/repo", 7, 1001) == second
    assert ledger.find_by_approval(REPO, 7, 1001) == ledger.get(first.run_id)
    assert ledger.find_by_approval("third/repo", 7, 1001) is None


@pytest.mark.parametrize("active_status", [RunStatus.RUNNING, RunStatus.PUBLISHING])
def test_only_one_run_may_be_active_across_issues_and_repositories(
    ledger: Ledger, active_status: RunStatus
) -> None:
    first = grant(ledger)
    if active_status is RunStatus.PUBLISHING:
        ledger.set_phase(first.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING)
    requests = [
        other_issue(8),
        make_request(approval_event_id=2002),  # same issue, newer approval
        make_request(repo="other/repo", spec_hash="elsewhere"),
    ]
    before = state_of(ledger)

    for request in requests:
        denied = ledger.lease(request)
        assert not denied and denied.reason == "busy"
        assert denied.other is not None and denied.other.run_id == first.run_id

    assert state_of(ledger) == before
    ledger.finish(first.run_id, RunStatus.DONE)
    assert ledger.lease(other_issue(8)).granted


def test_a_pending_run_waits_for_the_slot_without_holding_it(ledger: Ledger) -> None:
    pending = grant(ledger)
    assert ledger.requeue_interrupted(pending.run_id, 2)

    running = grant(ledger, other_issue(8))  # the slot is free while #7 is pending
    waiting = ledger.lease(make_request())  # ...but #7 cannot resume until #8 is done
    assert waiting.reason == "busy"
    assert waiting.other is not None and waiting.other.run_id == running.run_id
    assert ledger.get(pending.run_id).status is RunStatus.PENDING

    ledger.finish(running.run_id, RunStatus.DONE)
    resumed = ledger.lease(make_request())
    assert resumed.granted
    assert resumed.run is not None and resumed.run.run_id == pending.run_id


def test_pending_run_is_resumed_by_lease_keeping_run_id_attempt_and_start_time(
    ledger: Ledger, clock: FakeClock
) -> None:
    first = grant(ledger)
    ledger.update(first.run_id, agent_pid=999, agent_start_ticks=123_456)
    clock.advance(100)
    assert ledger.requeue_interrupted(first.run_id, 2) is True
    clock.advance(30)

    result = ledger.lease(make_request(worker_id="worker-2", worker_pid=2222, ttl_seconds=90))

    assert result.granted
    run = result.run
    assert run is not None
    assert run.run_id == first.run_id  # one approval, one run, however often it is resumed
    assert run.status is RunStatus.RUNNING and run.phase is RunPhase.LEASED
    assert (run.attempt, run.interrupted_count) == (2, 1)
    assert (run.worker_id, run.worker_pid) == ("worker-2", 2222)
    assert run.lease_expires_at == iso(clock() + timedelta(seconds=90))
    assert run.heartbeat_at == iso(clock()) == run.updated_at
    assert run.started_at == first.started_at == iso(T0)  # the first start is preserved
    assert (run.agent_pid, run.agent_start_ticks) == (None, None)
    assert ledger.runs_for_issue(REPO, 7) == [run]
    assert kinds(ledger) == ["lease.granted", "run.requeued", "lease.resumed"]


def test_resuming_a_pending_run_ignores_the_per_issue_attempt_limit(ledger: Ledger) -> None:
    run = grant(ledger, make_request(max_attempts_per_issue=1))
    assert ledger.requeue_interrupted(run.run_id, 3)

    resumed = ledger.lease(make_request(max_attempts_per_issue=1))

    assert resumed.granted and resumed.run is not None and resumed.run.run_id == run.run_id


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_attempt_limit_counts_every_run_of_the_issue(ledger: Ledger, limit: int) -> None:
    outcomes = [RunStatus.BLOCKED, RunStatus.CANCELLED, RunStatus.DONE]
    for n in range(limit):
        run = grant(ledger, make_request(approval_event_id=2000 + n, max_attempts_per_issue=limit))
        assert run.attempt == 1
        ledger.finish(run.run_id, outcomes[n])
    before = state_of(ledger)

    denied = ledger.lease(make_request(approval_event_id=3000, max_attempts_per_issue=limit))

    assert not denied and denied.reason == "attempt-limit"
    assert denied.other is None
    assert state_of(ledger) == before
    assert len(ledger.runs_for_issue(REPO, 7)) == limit


def test_attempt_limit_is_scoped_to_one_issue_in_one_repository(ledger: Ledger) -> None:
    first = grant(ledger, make_request(max_attempts_per_issue=1))
    ledger.finish(first.run_id, RunStatus.BLOCKED)
    again = ledger.lease(make_request(approval_event_id=1002, max_attempts_per_issue=1))
    assert again.reason == "attempt-limit"

    elsewhere = grant(ledger, other_issue(8, max_attempts_per_issue=1))
    ledger.finish(elsewhere.run_id, RunStatus.BLOCKED)
    in_other_repo = ledger.lease(
        make_request(repo="other/repo", spec_hash="elsewhere", max_attempts_per_issue=1)
    )
    assert in_other_repo.granted


@pytest.mark.parametrize("setup", ["done", "pending"])
def test_same_spec_on_another_issue_is_a_duplicate_content(ledger: Ledger, setup: str) -> None:
    twin = grant(ledger, make_request(spec_hash="shared"))
    if setup == "done":
        ledger.finish(twin.run_id, RunStatus.DONE)
    else:
        assert ledger.requeue_interrupted(twin.run_id, 1)
    before = state_of(ledger)

    denied = ledger.lease(other_issue(8, spec_hash="shared"))

    assert not denied and denied.reason == "duplicate-content"
    assert denied.other is not None and denied.other.run_id == twin.run_id
    assert state_of(ledger) == before
    assert ledger.runs_for_issue(REPO, 8) == []


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.PUBLISHING])
def test_an_active_twin_is_reported_as_busy_rather_than_duplicate_content(
    ledger: Ledger, status: RunStatus
) -> None:
    twin = grant(ledger, make_request(spec_hash="shared"))
    if status is RunStatus.PUBLISHING:
        ledger.set_phase(twin.run_id, RunPhase.PUSH, status=status)

    assert ledger.lease(other_issue(8, spec_hash="shared")).reason == "busy"


@pytest.mark.parametrize("status", [RunStatus.BLOCKED, RunStatus.CANCELLED], ids=lambda s: s.value)
def test_blocked_and_cancelled_runs_do_not_count_as_duplicate_content(
    ledger: Ledger, status: RunStatus
) -> None:
    earlier = grant(ledger, make_request(spec_hash="shared"))
    ledger.finish(earlier.run_id, status)

    assert ledger.lease(other_issue(8, spec_hash="shared")).granted


def test_content_dedup_can_be_switched_off_per_request(ledger: Ledger) -> None:
    twin = grant(ledger, make_request(spec_hash="shared"))
    ledger.finish(twin.run_id, RunStatus.DONE)

    request = other_issue(8, spec_hash="shared", dedupe_by_content=False)

    assert ledger.lease(request).granted


def test_content_dedup_ignores_the_same_issue_and_other_repositories(ledger: Ledger) -> None:
    first = grant(ledger, make_request(spec_hash="shared"))
    ledger.finish(first.run_id, RunStatus.DONE)

    same_issue_new_approval = grant(
        ledger, make_request(approval_event_id=1002, spec_hash="shared")
    )
    ledger.finish(same_issue_new_approval.run_id, RunStatus.DONE)
    other_repo = ledger.lease(make_request(repo="other/repo", spec_hash="shared"))

    assert other_repo.granted


@pytest.mark.parametrize(
    ("request_factory", "reason"),
    [
        pytest.param(make_request, "already-processed", id="processed-approval"),
        pytest.param(lambda: other_issue(9), "busy", id="another-issue"),
        pytest.param(lambda: make_request(approval_event_id=77), "busy", id="new-approval"),
        pytest.param(lambda: other_issue(10, spec_hash="spec-8"), "busy", id="content-twin"),
        pytest.param(
            lambda: make_request(approval_event_id=78, max_attempts_per_issue=1),
            "busy",
            id="over-the-attempt-limit",
        ),
    ],
)
def test_denials_write_nothing_and_a_live_run_outranks_every_other_new_run_check(
    ledger: Ledger, request_factory: Callable[[], LeaseRequest], reason: str
) -> None:
    done = grant(ledger)
    ledger.finish(done.run_id, RunStatus.DONE)
    live = grant(ledger, other_issue(8))
    before = state_of(ledger)

    denied = ledger.lease(request_factory())

    assert not denied and denied.reason == reason
    assert state_of(ledger) == before
    assert ledger.active_runs() == [live]


def test_an_expired_run_keeps_blocking_other_approvals_until_it_is_recovered(
    ledger: Ledger, clock: FakeClock
) -> None:
    stale = grant(ledger)
    clock.advance(TTL + 5)
    assert [r.run_id for r in ledger.stale_runs()] == [stale.run_id]

    assert ledger.lease(other_issue(8)).reason == "busy"  # its agent may still be alive
    assert ledger.lease(make_request()).reason == "recovery-pending"

    assert ledger.requeue_interrupted(stale.run_id, 1)  # recovery frees the slot...
    assert ledger.lease(other_issue(8)).granted  # ...for other work


# ------------------------------------------------------------------------------- heartbeats
def test_heartbeat_extends_the_lease(ledger: Ledger, clock: FakeClock) -> None:
    run = grant(ledger)  # expires at T0 + 60
    clock.advance(50)

    ledger.heartbeat(run.run_id, 60)

    beat = ledger.get(run.run_id)
    assert beat is not None
    assert beat.heartbeat_at == beat.updated_at == iso(clock())
    assert beat.lease_expires_at == iso(clock() + timedelta(seconds=60))
    clock.advance(30)  # T0 + 80: the original lease would have lapsed at T0 + 60
    assert ledger.stale_runs() == []
    assert ledger.lease(make_request()).reason == "duplicate-active"
    clock.advance(30)  # T0 + 110 == the extended expiry
    assert [r.run_id for r in ledger.stale_runs()] == [run.run_id]


def test_heartbeat_ttl_replaces_the_previous_expiry(ledger: Ledger, clock: FakeClock) -> None:
    run = grant(ledger, make_request(ttl_seconds=600))

    ledger.heartbeat(run.run_id, 5)

    clock.advance(5)
    assert [r.run_id for r in ledger.stale_runs()] == [run.run_id]


def test_heartbeat_keeps_publishing_runs_alive(ledger: Ledger, clock: FakeClock) -> None:
    run = grant(ledger)
    ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING)
    clock.advance(TTL - 1)

    ledger.heartbeat(run.run_id, TTL)

    assert ledger.get(run.run_id).lease_expires_at == iso(clock() + timedelta(seconds=TTL))


@pytest.mark.parametrize("setup", ["finished", "pending", "unknown"])
def test_heartbeat_ignores_runs_that_are_not_active(
    ledger: Ledger, clock: FakeClock, setup: str
) -> None:
    run = grant(ledger)
    if setup == "finished":
        ledger.finish(run.run_id, RunStatus.DONE)
    elif setup == "pending":
        assert ledger.requeue_interrupted(run.run_id, 1)
    clock.advance(500)
    before = state_of(ledger)

    ledger.heartbeat(run.run_id if setup != "unknown" else "no-such-run", TTL)

    assert state_of(ledger) == before
    if setup != "unknown":
        assert ledger.get(run.run_id).lease_expires_at is None  # no zombie lease appears


# ------------------------------------------------------------------------------- stale runs
def test_stale_runs_lists_only_active_runs_whose_lease_has_expired(
    ledger: Ledger, clock: FakeClock
) -> None:
    old = grant(ledger)  # expires at T0 + 60
    assert ledger.stale_runs() == []
    clock.advance(TTL - 1)
    assert ledger.stale_runs() == []
    clock.advance(1)
    assert [r.run_id for r in ledger.stale_runs()] == [old.run_id]  # the boundary is stale

    ledger.set_phase(old.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING)
    assert [r.run_id for r in ledger.stale_runs()] == [old.run_id]  # publishing runs too
    ledger.finish(old.run_id, RunStatus.BLOCKED)  # terminal: not stale
    assert ledger.stale_runs() == []

    waiting = grant(ledger, other_issue(8))
    assert ledger.requeue_interrupted(waiting.run_id, 1)  # pending: waiting, not stale
    assert ledger.stale_runs() == []
    finished = grant(ledger, other_issue(9))
    ledger.finish(finished.run_id, RunStatus.DONE)
    assert ledger.stale_runs() == []
    live = grant(ledger, other_issue(10))  # fresh lease: not stale
    assert ledger.stale_runs() == []

    clock.advance(TTL)
    assert [r.run_id for r in ledger.stale_runs()] == [live.run_id]


def test_an_active_run_without_a_lease_is_stale(ledger: Ledger) -> None:
    run = grant(ledger)
    assert ledger.requeue_interrupted(run.run_id, 1)  # clears the lease
    ledger.set_phase(run.run_id, RunPhase.AGENT, status=RunStatus.RUNNING)

    assert [r.run_id for r in ledger.stale_runs()] == [run.run_id]


# ---------------------------------------------------------------------------- claim_publishing
def test_claim_publishing_takes_over_a_publishing_run_and_counts_attempts(
    ledger: Ledger, clock: FakeClock
) -> None:
    run = grant(ledger)
    ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING, commit_sha="d" * 40)
    clock.advance(500)  # the publishing worker died long ago
    assert [r.run_id for r in ledger.stale_runs()] == [run.run_id]

    assert ledger.claim_publishing(run.run_id, "worker-2", 777, 120) is True

    claimed = ledger.get(run.run_id)
    assert claimed is not None
    assert claimed.status is RunStatus.PUBLISHING and claimed.phase is RunPhase.PUSH
    assert claimed.publish_attempts == 1
    assert (claimed.worker_id, claimed.worker_pid) == ("worker-2", 777)
    assert claimed.lease_expires_at == iso(clock() + timedelta(seconds=120))
    assert claimed.heartbeat_at == claimed.updated_at == iso(clock())
    assert claimed.commit_sha == "d" * 40
    assert ledger.stale_runs() == []

    assert ledger.claim_publishing(run.run_id, "worker-2", 777, 120) is True  # refreshing is fine
    assert ledger.get(run.run_id).publish_attempts == 2


@pytest.mark.parametrize("setup", ["running", "pending", *[s.value for s in TERMINAL], "unknown"])
def test_claim_publishing_only_applies_to_publishing_runs(
    ledger: Ledger, clock: FakeClock, setup: str
) -> None:
    run = grant(ledger)
    if setup == "pending":
        assert ledger.requeue_interrupted(run.run_id, 1)
    elif setup in {s.value for s in TERMINAL}:
        ledger.finish(run.run_id, RunStatus(setup))
    clock.advance(500)
    before = state_of(ledger)

    claimed = ledger.claim_publishing(
        "no-such-run" if setup == "unknown" else run.run_id, "worker-2", 777, 120
    )

    assert claimed is False
    assert state_of(ledger) == before


# -------------------------------------------------------------------------- requeue_interrupted
def test_requeue_returns_the_run_to_pending_and_counts_the_interruption(
    ledger: Ledger, clock: FakeClock
) -> None:
    run = grant(ledger)
    ledger.update(run.run_id, agent_pid=555, agent_start_ticks=123)
    ledger.set_phase(run.run_id, RunPhase.AGENT, branch="agent/issue-7")
    clock.advance(10)

    assert ledger.requeue_interrupted(run.run_id, 2) is True

    pending = ledger.get(run.run_id)
    assert pending is not None
    assert pending.status is RunStatus.PENDING and pending.phase is RunPhase.LEASED
    assert (pending.attempt, pending.interrupted_count) == (2, 1)
    assert pending.lease_expires_at is None
    assert (pending.agent_pid, pending.agent_start_ticks) == (None, None)
    assert pending.branch == "agent/issue-7"  # recovered work is kept
    assert pending.updated_at == iso(clock())
    assert ledger.pending_runs() == [pending] and ledger.active_runs() == []
    last = ledger.events()[-1]
    assert (last["kind"], last["detail"], last["run_id"]) == (
        "run.requeued",
        {"after": "interrupted"},
        run.run_id,
    )


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_requeue_honours_the_automatic_retry_limit(ledger: Ledger, limit: int) -> None:
    run = grant(ledger)
    for expected in range(1, limit + 1):
        assert ledger.requeue_interrupted(run.run_id, limit) is True
        assert ledger.get(run.run_id).interrupted_count == expected
        resumed = ledger.lease(make_request())
        assert resumed.granted and resumed.run is not None
        assert (resumed.run.run_id, resumed.run.attempt) == (run.run_id, expected + 1)
    before = state_of(ledger)

    assert ledger.requeue_interrupted(run.run_id, limit) is False  # budget exhausted (0: never)

    assert state_of(ledger) == before
    assert ledger.get(run.run_id).status is RunStatus.RUNNING  # the caller must finish() it


def test_requeue_of_an_unknown_run_is_refused(ledger: Ledger) -> None:
    assert ledger.requeue_interrupted("no-such-run", 3) is False
    assert ledger.events() == []


@pytest.mark.parametrize("setup", ["pending", "publishing", *[s.value for s in TERMINAL]])
def test_only_a_running_run_can_be_requeued(ledger: Ledger, setup: str) -> None:
    """A late recovery pass must not resurrect (and so re-execute) a processed approval."""
    run = grant(ledger)
    if setup == "pending":
        assert ledger.requeue_interrupted(run.run_id, 5)
    elif setup == "publishing":
        ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING)
    else:
        ledger.finish(run.run_id, RunStatus(setup))
    before = state_of(ledger)

    assert ledger.requeue_interrupted(run.run_id, 3) is False

    assert state_of(ledger) == before
    if setup in {s.value for s in TERMINAL}:
        assert ledger.lease(make_request()).reason == "already-processed"


# ------------------------------------------------------------------------------------ finish
@pytest.mark.parametrize("status", [RunStatus.PENDING, RunStatus.RUNNING, RunStatus.PUBLISHING])
def test_finish_rejects_non_terminal_statuses(ledger: Ledger, status: RunStatus) -> None:
    run = grant(ledger)
    before = state_of(ledger)

    with pytest.raises(ValueError, match="terminal"):
        ledger.finish(run.run_id, status)

    assert state_of(ledger) == before


@pytest.mark.parametrize("status", TERMINAL, ids=lambda s: s.value)
def test_finish_records_the_outcome_and_clears_lease_and_agent_state(
    ledger: Ledger, clock: FakeClock, status: RunStatus
) -> None:
    run = grant(ledger)
    ledger.update(run.run_id, agent_pid=555, agent_start_ticks=123)
    clock.advance(42)

    ledger.finish(
        run.run_id,
        status,
        failure_kind="policy_violation",
        failure_detail="2 changes are not allowed",
        result={"files": ["docs/a.md"], "note": "héllo ✓", "nested": {"n": [1, 2]}},
    )

    done = ledger.get(run.run_id)
    assert done is not None
    assert done.status is status and done.phase is RunPhase.FINISHED
    assert done.finished_at == done.updated_at == iso(clock())
    assert (done.failure_kind, done.failure_detail) == (
        "policy_violation",
        "2 changes are not allowed",
    )
    assert done.result == {"files": ["docs/a.md"], "note": "héllo ✓", "nested": {"n": [1, 2]}}
    assert done.lease_expires_at is None
    assert (done.agent_pid, done.agent_start_ticks) == (None, None)
    assert ledger.active_runs() == [] and ledger.stale_runs() == []
    last = ledger.events()[-1]
    assert last["kind"] == f"finish.{status.value}"
    assert last["detail"] == {"failure_kind": "policy_violation"}


def test_finish_bounds_failure_detail_and_normalises_empty_values(ledger: Ledger) -> None:
    long_run = grant(ledger)
    ledger.finish(long_run.run_id, RunStatus.BLOCKED, failure_kind="k", failure_detail="d" * 5000)
    assert ledger.get(long_run.run_id).failure_detail == "d" * 4000

    empty_run = grant(ledger, other_issue(8))
    ledger.finish(empty_run.run_id, RunStatus.DONE, failure_detail="")
    record = ledger.get(empty_run.run_id)
    assert (record.failure_kind, record.failure_detail) == (None, None)


def test_finish_keeps_a_previously_stored_result_unless_given_a_new_one(ledger: Ledger) -> None:
    kept = grant(ledger)
    ledger.update(kept.run_id, result_json={"stage": "agent", "files": 2})
    ledger.finish(kept.run_id, RunStatus.DONE)
    assert ledger.get(kept.run_id).result == {"stage": "agent", "files": 2}

    replaced = grant(ledger, other_issue(8))
    ledger.update(replaced.run_id, result_json={"stage": "agent"})
    ledger.finish(replaced.run_id, RunStatus.DONE, result={"stage": "final"})
    assert ledger.get(replaced.run_id).result == {"stage": "final"}


def test_finish_on_an_unknown_run_is_a_silent_no_op(ledger: Ledger) -> None:
    ledger.finish("no-such-run", RunStatus.DONE, result={"x": 1})

    assert ledger.list_runs() == [] and ledger.events() == []


# ----------------------------------------------------------------------- update / set_phase
UPDATABLE_SAMPLES: dict[str, Any] = {
    "agent_pid": 4321,
    "agent_start_ticks": 987_654,
    "backend": "claude",
    "base_sha": "a" * 40,
    "branch": "agent/issue-7-abc",
    "commit_sha": "b" * 40,
    "failure_detail": "boom",
    "failure_kind": "agent_failed",
    "pr_number": 12,
    "pr_url": "https://github.com/owner/repo/pull/12",
    "result_json": {"files": ["docs/a.md"], "ok": True},
    "worktree_path": "/tmp/worktrees/issue-7",
}
REJECTED_COLUMNS = [
    "id",
    "repo",
    "issue_number",
    "attempt",
    "interrupted_count",
    "worker_id",
    "lease_expires_at",
    "cancel_requested",
    "dedup_key",
    "approval_event_id",
    "created_at",
    "finished_at",
    "no_such_column",
    "branch=?, status='done' --",
    "result_json, status",
]


def attribute(name: str) -> str:
    return "result" if name == "result_json" else name


@pytest.mark.parametrize("name", sorted(UPDATABLE_SAMPLES))
def test_update_stores_each_allowed_column(ledger: Ledger, clock: FakeClock, name: str) -> None:
    run = grant(ledger)
    clock.advance(5)

    ledger.update(run.run_id, **{name: UPDATABLE_SAMPLES[name]})

    record = ledger.get(run.run_id)
    assert record is not None
    assert getattr(record, attribute(name)) == UPDATABLE_SAMPLES[name]
    assert record.updated_at == iso(clock())
    assert record.status is RunStatus.RUNNING and record.phase is RunPhase.LEASED


@pytest.mark.parametrize("name", sorted(UPDATABLE_SAMPLES))
def test_set_phase_stores_each_allowed_column(ledger: Ledger, name: str) -> None:
    run = grant(ledger)

    ledger.set_phase(run.run_id, RunPhase.WORKTREE, **{name: UPDATABLE_SAMPLES[name]})

    record = ledger.get(run.run_id)
    assert record is not None
    assert getattr(record, attribute(name)) == UPDATABLE_SAMPLES[name]
    assert record.phase is RunPhase.WORKTREE


def test_update_stores_every_allowed_column_in_one_statement(ledger: Ledger) -> None:
    run = grant(ledger)

    ledger.update(run.run_id, **UPDATABLE_SAMPLES)

    record = ledger.get(run.run_id)
    assert record is not None
    for name, value in UPDATABLE_SAMPLES.items():
        assert getattr(record, attribute(name)) == value


@pytest.mark.parametrize("column", [*REJECTED_COLUMNS, "status", "phase"])
def test_update_rejects_columns_outside_the_allow_list(ledger: Ledger, column: str) -> None:
    run = grant(ledger)
    before = state_of(ledger)

    with pytest.raises(ValueError, match="cannot update"):
        ledger.update(run.run_id, branch="agent/x", **{column: "x"})

    assert state_of(ledger) == before  # not even the valid field was applied


@pytest.mark.parametrize("column", REJECTED_COLUMNS)
def test_set_phase_rejects_columns_outside_the_allow_list(ledger: Ledger, column: str) -> None:
    run = grant(ledger)
    before = state_of(ledger)

    with pytest.raises(ValueError, match="cannot update"):
        ledger.set_phase(run.run_id, RunPhase.AGENT, branch="agent/x", **{column: "x"})

    assert state_of(ledger) == before  # neither the phase, the field nor an event was written


def test_update_without_fields_or_for_an_unknown_run_changes_nothing(
    ledger: Ledger, clock: FakeClock
) -> None:
    run = grant(ledger)
    clock.advance(5)
    before = state_of(ledger)

    ledger.update(run.run_id)
    ledger.update("no-such-run", branch="agent/x")
    ledger.set_phase("no-such-run", RunPhase.AGENT, branch="agent/x")

    assert state_of(ledger) == before


@pytest.mark.parametrize("writer", ["update", "set_phase"])
def test_result_json_accepts_dicts_json_text_and_none(
    ledger: Ledger, db_path: Path, writer: str
) -> None:
    run = grant(ledger)

    def write(value: Any) -> None:
        if writer == "update":
            ledger.update(run.run_id, result_json=value)
        else:
            ledger.set_phase(run.run_id, RunPhase.AGENT, result_json=value)

    payload = {"summary": "héllo ✓", "files": ["a", "b"], "n": {"deep": [1, 2, 3]}}
    write(payload)
    assert ledger.get(run.run_id).result == payload
    (stored,) = sql_rows(db_path, "SELECT result_json FROM runs")[0]
    assert json.loads(stored) == payload  # a real JSON document, not a Python repr

    write('{"raw": "text"}')
    assert ledger.get(run.run_id).result == {"raw": "text"}

    write(None)
    assert ledger.get(run.run_id).result is None
    assert sql_rows(db_path, "SELECT result_json FROM runs") == [(None,)]


def test_set_phase_moves_phase_status_and_records_an_event(
    ledger: Ledger, clock: FakeClock
) -> None:
    run = grant(ledger)
    clock.advance(3)

    ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING, commit_sha="e" * 40)

    record = ledger.get(run.run_id)
    assert record is not None
    assert (record.status, record.phase) == (RunStatus.PUBLISHING, RunPhase.PUSH)
    assert record.commit_sha == "e" * 40 and record.updated_at == iso(clock())
    assert ledger.active_runs() == [record]
    assert kinds(ledger) == ["lease.granted", "phase.push"]

    ledger.set_phase(run.run_id, RunPhase.PULL_REQUEST)  # no status: unchanged
    assert ledger.get(run.run_id).status is RunStatus.PUBLISHING
    assert kinds(ledger)[-1] == "phase.pull_request"


# ------------------------------------------------------------------------------- cancellation
@pytest.mark.parametrize("by", ["run_id", "issue_number"])
@pytest.mark.parametrize("setup", ["running", "publishing", "pending"])
def test_request_cancel_flags_non_terminal_runs(ledger: Ledger, by: str, setup: str) -> None:
    run = grant(ledger)
    if setup == "publishing":
        ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING)
    elif setup == "pending":
        assert ledger.requeue_interrupted(run.run_id, 1)
    assert ledger.is_cancel_requested(run.run_id) is False

    flagged = ledger.request_cancel(**{by: run.run_id if by == "run_id" else run.issue_number})

    assert flagged == 1
    assert ledger.is_cancel_requested(run.run_id) is True
    assert ledger.get(run.run_id).cancel_requested is True


@pytest.mark.parametrize("by", ["run_id", "issue_number"])
@pytest.mark.parametrize("status", TERMINAL, ids=lambda s: s.value)
def test_request_cancel_ignores_finished_runs(ledger: Ledger, by: str, status: RunStatus) -> None:
    run = grant(ledger)
    ledger.finish(run.run_id, status)
    before = state_of(ledger)

    flagged = ledger.request_cancel(**{by: run.run_id if by == "run_id" else run.issue_number})

    assert flagged == 0
    assert ledger.is_cancel_requested(run.run_id) is False
    assert state_of(ledger) == before


def test_cancelling_an_issue_flags_only_its_own_unfinished_runs(ledger: Ledger) -> None:
    old = grant(ledger)
    ledger.finish(old.run_id, RunStatus.BLOCKED)
    neighbour = grant(ledger, other_issue(8))
    assert ledger.requeue_interrupted(neighbour.run_id, 1)  # unfinished, but another issue
    current = grant(ledger, make_request(approval_event_id=1002))

    assert ledger.request_cancel(issue_number=7) == 1

    assert ledger.is_cancel_requested(current.run_id) is True
    assert ledger.is_cancel_requested(old.run_id) is False
    assert ledger.is_cancel_requested(neighbour.run_id) is False
    assert ledger.request_cancel(issue_number=99) == 0


def test_cancel_request_requires_a_target_and_unknown_runs_report_false(ledger: Ledger) -> None:
    with pytest.raises(ValueError, match="required"):
        ledger.request_cancel()
    assert ledger.request_cancel(run_id="no-such-run") == 0
    assert ledger.is_cancel_requested("no-such-run") is False


def test_cancel_flag_survives_requeue_and_resume(ledger: Ledger) -> None:
    run = grant(ledger)
    assert ledger.request_cancel(run_id=run.run_id) == 1
    assert ledger.requeue_interrupted(run.run_id, 1)

    resumed = ledger.lease(make_request())

    assert resumed.granted
    assert ledger.is_cancel_requested(run.run_id) is True  # the worker will stop it again


# ------------------------------------------------------------------------------------ events
def test_record_event_stores_run_issue_detail_and_timestamp(
    ledger: Ledger, clock: FakeClock
) -> None:
    run = grant(ledger)
    clock.advance(7)

    ledger.record_event(
        "lease.denied",
        repo=REPO,
        issue_number=8,
        run_id=run.run_id,
        detail={"reason": "busy", "other": run.run_id, "note": "héllo ✓", "n": [1, {"a": 2}]},
    )
    ledger.record_event("bare", repo=REPO)

    first, second = ledger.events()[-2:]
    assert first["kind"] == "lease.denied"
    assert (first["run_id"], first["issue_number"]) == (run.run_id, 8)
    assert first["at"] == iso(clock())
    assert first["detail"] == {
        "reason": "busy",
        "other": run.run_id,
        "note": "héllo ✓",
        "n": [1, {"a": 2}],
    }
    assert (second["kind"], second["run_id"], second["issue_number"]) == ("bare", None, None)
    assert second["detail"] == {}
    assert second["id"] > first["id"]


def test_events_are_returned_oldest_first_and_limit_keeps_the_newest(ledger: Ledger) -> None:
    for n in range(5):
        ledger.record_event(f"e{n}", repo=REPO)

    assert kinds(ledger) == ["e0", "e1", "e2", "e3", "e4"]
    assert kinds(ledger, limit=3) == ["e2", "e3", "e4"]
    assert kinds(ledger, limit=1) == ["e4"]
    assert kinds(ledger, limit=0) == []
    ids = [e["id"] for e in ledger.events()]
    assert ids == sorted(ids) and len(set(ids)) == 5


def test_events_filter_by_run_and_issue(ledger: Ledger) -> None:
    first = grant(ledger)
    ledger.finish(first.run_id, RunStatus.DONE)
    second = grant(ledger, other_issue(8))
    ledger.record_event("note.run", repo=REPO, run_id=second.run_id, issue_number=8)
    ledger.record_event("note.issue-only", repo=REPO, issue_number=7)
    ledger.record_event("note.global", repo=REPO)

    assert kinds(ledger, run_id=first.run_id) == ["lease.granted", "finish.done"]
    assert kinds(ledger, run_id=second.run_id) == ["lease.granted", "note.run"]
    assert kinds(ledger, issue_number=7) == ["lease.granted", "finish.done", "note.issue-only"]
    assert kinds(ledger, issue_number=8) == ["lease.granted", "note.run"]
    assert kinds(ledger, run_id=second.run_id, issue_number=7) == []
    assert kinds(ledger, run_id=second.run_id, issue_number=8, limit=1) == ["note.run"]
    assert kinds(ledger, run_id="no-such-run") == []
    assert kinds(ledger)[-1] == "note.global"


def test_a_run_lifecycle_leaves_an_ordered_audit_trail(ledger: Ledger, clock: FakeClock) -> None:
    run = grant(ledger)
    ledger.set_phase(run.run_id, RunPhase.WORKTREE, branch="agent/7", worktree_path="/w")
    ledger.set_phase(run.run_id, RunPhase.AGENT)
    assert ledger.requeue_interrupted(run.run_id, 1)
    assert ledger.lease(make_request()).granted
    ledger.set_phase(run.run_id, RunPhase.PUSH, status=RunStatus.PUBLISHING, commit_sha="f" * 40)
    clock.advance(1)
    ledger.finish(run.run_id, RunStatus.DONE, result={"pr": 3})

    events = ledger.events(run_id=run.run_id)

    assert [e["kind"] for e in events] == [
        "lease.granted",
        "phase.worktree",
        "phase.agent",
        "run.requeued",
        "lease.resumed",
        "phase.push",
        "finish.done",
    ]
    assert {e["issue_number"] for e in events} == {7}
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)
    assert [e["at"] for e in events] == sorted(e["at"] for e in events)
    assert events[-1]["detail"] == {"failure_kind": None}


# ------------------------------------------------------------------------------------ notices
def test_notice_once_is_true_only_the_first_time_per_key(ledger: Ledger) -> None:
    assert ledger.notice_once("refusal-7-1001-unapproved") is True
    assert ledger.notice_once("refusal-7-1001-unapproved") is False
    assert ledger.notice_once("refusal-7-1002-unapproved") is True
    assert ledger.notice_once("refusal-7-1001-unapproved") is False


def test_forget_notice_allows_the_notice_to_be_sent_again(ledger: Ledger) -> None:
    assert ledger.notice_once("comment-1") is True

    ledger.forget_notice("comment-1")

    assert ledger.notice_once("comment-1") is True
    assert ledger.notice_once("comment-1") is False
    ledger.forget_notice("never-recorded")  # forgetting an unknown key is harmless
    assert ledger.notice_once("never-recorded") is True


# ---------------------------------------------------------------------------- listing helpers
def test_listing_helpers_order_filter_and_limit(ledger: Ledger) -> None:
    done = grant(ledger)
    ledger.finish(done.run_id, RunStatus.DONE)
    blocked = grant(ledger, other_issue(8))
    ledger.finish(blocked.run_id, RunStatus.BLOCKED)
    waiting = grant(ledger, other_issue(9))
    assert ledger.requeue_interrupted(waiting.run_id, 1)
    running = grant(ledger, other_issue(10))

    assert [r.run_id for r in ledger.list_runs()] == [
        running.run_id,
        waiting.run_id,
        blocked.run_id,
        done.run_id,
    ]
    assert [r.run_id for r in ledger.list_runs(limit=2)] == [running.run_id, waiting.run_id]
    assert [r.run_id for r in ledger.list_runs(status=RunStatus.DONE)] == [done.run_id]
    assert ledger.list_runs(status=RunStatus.CANCELLED) == []
    assert [r.run_id for r in ledger.active_runs()] == [running.run_id]
    assert [r.run_id for r in ledger.pending_runs()] == [waiting.run_id]
    assert ledger.get("no-such-run") is None
    assert ledger.find_by_approval(REPO, 7, 31337) is None
    assert ledger.runs_for_issue(REPO, 99) == []


# --------------------------------------------------------------------------------- atomicity
class _Abort(BaseException):
    """Not an ``Exception``: transactions must roll back for every kind of unwinding."""


@pytest.mark.parametrize("failure", [RuntimeError("boom"), _Abort()], ids=["exception", "base"])
def test_an_exception_inside_a_transaction_rolls_everything_back(
    ledger: Ledger, failure: BaseException
) -> None:
    with pytest.raises(type(failure)):
        with ledger._tx() as cx:
            cx.execute("INSERT INTO notices (key, at) VALUES ('partial', 'now')")
            raise failure

    assert ledger.notice_once("partial") is True  # the insert never happened
    with ledger._tx() as cx:  # and the ledger is not stuck inside a transaction
        cx.execute("INSERT INTO notices (key, at) VALUES ('kept', 'now')")
    assert ledger.notice_once("kept") is False  # a clean exit commits


def _block_events(path: Path) -> None:
    """Make every event insert fail, as a full disk or a broken index would."""
    sql_exec(
        path,
        "CREATE TRIGGER block_events BEFORE INSERT ON events "
        "BEGIN SELECT RAISE(ABORT, 'event log unavailable'); END",
    )


def _unblock_events(path: Path) -> None:
    sql_exec(path, "DROP TRIGGER block_events")


ON_RUNNING_RUN: dict[str, Callable[[Ledger, RunRecord], Any]] = {
    "set_phase": lambda led, run: led.set_phase(run.run_id, RunPhase.AGENT, branch="agent/x"),
    "finish": lambda led, run: led.finish(run.run_id, RunStatus.DONE, result={"ok": True}),
    "requeue": lambda led, run: led.requeue_interrupted(run.run_id, 3),
}


@pytest.mark.parametrize("operation", sorted(ON_RUNNING_RUN))
def test_a_failing_event_write_rolls_back_the_whole_state_change(
    ledger: Ledger, db_path: Path, operation: str
) -> None:
    run = grant(ledger)
    before = state_of(ledger)
    _block_events(db_path)

    with pytest.raises(sqlite3.DatabaseError, match="event log unavailable"):
        ON_RUNNING_RUN[operation](ledger, run)

    assert state_of(ledger) == before  # no half-finished run, no half-moved phase
    assert ledger.get(run.run_id).status is RunStatus.RUNNING
    _unblock_events(db_path)
    ON_RUNNING_RUN[operation](ledger, run)  # the ledger did not wedge inside a transaction
    assert state_of(ledger) != before


def test_a_failing_event_write_leaves_no_partial_run_behind_a_new_lease(
    ledger: Ledger, db_path: Path
) -> None:
    _block_events(db_path)

    with pytest.raises(sqlite3.DatabaseError, match="event log unavailable"):
        ledger.lease(make_request())

    assert ledger.list_runs() == [] and ledger.active_runs() == []
    _unblock_events(db_path)
    assert ledger.lease(make_request()).granted  # not blocked by a ghost row


def test_a_failing_event_write_leaves_a_pending_run_pending(ledger: Ledger, db_path: Path) -> None:
    run = grant(ledger)
    assert ledger.requeue_interrupted(run.run_id, 1)
    before = state_of(ledger)
    _block_events(db_path)

    with pytest.raises(sqlite3.DatabaseError, match="event log unavailable"):
        ledger.lease(make_request())

    assert state_of(ledger) == before
    assert ledger.get(run.run_id).status is RunStatus.PENDING
    _unblock_events(db_path)
    assert ledger.lease(make_request()).granted


def test_record_event_failure_stores_nothing(ledger: Ledger, db_path: Path) -> None:
    _block_events(db_path)

    with pytest.raises(sqlite3.DatabaseError, match="event log unavailable"):
        ledger.record_event("x", repo=REPO)

    _unblock_events(db_path)
    assert ledger.events() == []


# --------------------------------------------------------------------------- thread concurrency
def race(count: int, work: Callable[[int], Any]) -> list[Any]:
    """Run ``work(i)`` in ``count`` threads released together and return the results."""
    barrier = threading.Barrier(count, timeout=30)
    results: list[Any] = [None] * count
    errors: list[BaseException] = []

    def runner(index: int) -> None:
        try:
            barrier.wait()
            results[index] = work(index)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(i,), daemon=True) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a racing thread never finished"
    if errors:
        raise errors[0]
    return results


def test_sixteen_threads_sharing_one_ledger_race_for_one_approval(ledger: Ledger) -> None:
    results = race(
        16, lambda i: ledger.lease(make_request(worker_id=f"worker-{i}", worker_pid=1000 + i))
    )

    winners = [r for r in results if r.granted]
    assert len(winners) == 1
    assert {r.reason for r in results if not r.granted} == {"duplicate-active"}
    assert len(ledger.list_runs()) == 1
    assert kinds(ledger) == ["lease.granted"]
    stored = ledger.get(winners[0].run.run_id)
    assert stored is not None and stored.worker_id == winners[0].run.worker_id


def test_sixteen_independent_connections_race_for_one_approval(
    db_path: Path, clock: FakeClock
) -> None:
    Ledger(db_path, clock=clock).close()  # create the schema once
    ledgers = [Ledger(db_path, clock=clock) for _ in range(16)]
    try:
        results = race(
            16,
            lambda i: ledgers[i].lease(make_request(worker_id=f"worker-{i}", worker_pid=1000 + i)),
        )
    finally:
        for instance in ledgers:
            instance.close()

    winners = [r for r in results if r.granted]
    assert len(winners) == 1
    assert {r.reason for r in results if not r.granted} == {"duplicate-active"}
    assert sql_rows(db_path, "SELECT COUNT(*) FROM runs") == [(1,)]
    assert sql_rows(db_path, "SELECT COUNT(*) FROM events WHERE kind='lease.granted'") == [(1,)]


def test_sixteen_threads_racing_for_different_issues_get_one_slot(
    db_path: Path, clock: FakeClock
) -> None:
    Ledger(db_path, clock=clock).close()
    ledgers = [Ledger(db_path, clock=clock) for _ in range(16)]
    try:
        results = race(16, lambda i: ledgers[i].lease(other_issue(100 + i)))
    finally:
        for instance in ledgers:
            instance.close()

    assert sum(1 for r in results if r.granted) == 1
    assert {r.reason for r in results if not r.granted} == {"busy"}
    assert sql_rows(db_path, "SELECT COUNT(*) FROM runs") == [(1,)]


def test_sixteen_threads_racing_one_notice_send_it_once(ledger: Ledger) -> None:
    results = race(16, lambda _i: ledger.notice_once("refusal-7-1001-unapproved"))

    assert results.count(True) == 1 and results.count(False) == 15


def test_a_shared_ledger_is_safe_under_mixed_concurrent_use(ledger: Ledger) -> None:
    run = grant(ledger)
    workers, rounds = 8, 25

    def work(index: int) -> None:
        for step in range(rounds):
            ledger.record_event(f"w{index}", repo=REPO, run_id=run.run_id, issue_number=7)
            ledger.heartbeat(run.run_id, TTL)
            assert ledger.get(run.run_id) is not None
            assert ledger.events(run_id=run.run_id, limit=5)
            assert ledger.is_cancel_requested(run.run_id) is False
            assert ledger.notice_once(f"n-{index}-{step}") is True

    race(workers, work)

    events = ledger.events(run_id=run.run_id, limit=100_000)
    assert len(events) == 1 + workers * rounds
    assert len({e["id"] for e in events}) == len(events)


# ----------------------------------------------------------------------- process concurrency
LEDGER_CHILD = textwrap.dedent(
    """
    import json, os, sys, time
    from datetime import UTC, datetime, timedelta

    root, db, go, ident, mode, offset = sys.argv[1:7]
    sys.path.insert(0, root)
    from tools.agent_bridge.ledger import Ledger, LeaseRequest

    open(f"{go}.ready.{ident}", "w").close()
    deadline = time.monotonic() + 60
    while not os.path.exists(go) and time.monotonic() < deadline:
        time.sleep(0.0005)
    shift = timedelta(seconds=float(offset))
    try:
        ledger = Ledger(db, clock=lambda: datetime.now(UTC) + shift)
        same = mode.startswith("same")
        result = ledger.lease(
            LeaseRequest(
                repo="owner/repo",
                issue_number=7 if same else 100 + int(ident),
                issue_title="race",
                revision="rev",
                spec_hash="spec" if same else f"spec-{ident}",
                approval_event_id=1001 if same else 5000 + int(ident),
                approved_by="owner",
                approved_at="2026-03-01T00:00:00Z",
                backend="omp",
                worker_id=f"proc-{ident}",
                worker_pid=os.getpid(),
                ttl_seconds=60,
                max_attempts_per_issue=3,
            )
        )
        reply = {
            "granted": result.granted,
            "reason": result.reason,
            "run_id": result.run.run_id if result.run else None,
        }
    except Exception as exc:
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(reply), flush=True)
    if mode.endswith("crash"):
        os._exit(0)  # die without closing the database or checkpointing the WAL
    """
)
CHILD_ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONCOERCECLOCALE": "0"}


def run_ledger_processes(
    tmp_path: Path,
    count: int,
    mode: str,
    *,
    db: Path | None = None,
    offset: float = 0.0,
    precreate: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    """Start ``count`` OS processes that lease at the same instant; return their replies."""
    token = uuid.uuid4().hex[:8]
    db = db or tmp_path / f"race-{token}" / "ledger.db"
    go = tmp_path / f"go-{token}"
    if precreate:
        Ledger(db).close()
    procs: list[subprocess.Popen[str]] = []
    try:
        for ident in range(count):
            argv = [sys.executable, "-S", "-c", LEDGER_CHILD, str(REPO_ROOT), str(db), str(go)]
            argv += [str(ident), mode, str(offset)]
            procs.append(
                subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=CHILD_ENV,
                )
            )
        ready = [Path(f"{go}.ready.{i}") for i in range(count)]
        for _ in range(3000):  # up to 30 s for every child to import and park
            if all(p.exists() for p in ready):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("child processes never became ready")
        go.write_text("go")
        replies: list[dict[str, Any]] = []
        for proc in procs:
            out, err = proc.communicate(timeout=90)
            assert proc.returncode == 0, err
            replies.append(json.loads(out.strip().splitlines()[-1]))
        return db, replies
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.parametrize("processes", [2, 4])
def test_separate_processes_racing_one_approval_get_exactly_one_lease(
    tmp_path: Path, processes: int
) -> None:
    for _ in range(3):  # a fresh database per round makes every round a new race
        db, replies = run_ledger_processes(tmp_path, processes, "same")

        assert [r for r in replies if "error" in r] == []
        granted = [r for r in replies if r["granted"]]
        assert len(granted) == 1
        assert {r["reason"] for r in replies if not r["granted"]} == {"duplicate-active"}
        ledger = Ledger(db)
        try:
            (run,) = ledger.list_runs()
            assert run.run_id == granted[0]["run_id"]
            assert run.worker_id.startswith("proc-")
            assert kinds(ledger) == ["lease.granted"]
        finally:
            ledger.close()


def test_separate_processes_racing_for_different_issues_get_one_slot(tmp_path: Path) -> None:
    for _ in range(3):
        db, replies = run_ledger_processes(tmp_path, 4, "distinct")

        assert [r for r in replies if "error" in r] == []
        assert sum(1 for r in replies if r["granted"]) == 1
        assert {r["reason"] for r in replies if not r["granted"]} == {"busy"}
        assert sql_rows(db, "SELECT COUNT(*) FROM runs") == [(1,)]


def test_processes_opening_a_brand_new_ledger_at_once_all_succeed(tmp_path: Path) -> None:
    """First start of two bridge commands in a fresh state directory must not fail."""
    for _ in range(4):
        db, replies = run_ledger_processes(tmp_path, 8, "same", precreate=False)

        assert [r for r in replies if "error" in r] == []
        assert sum(1 for r in replies if r["granted"]) == 1
        assert sql_rows(db, "SELECT COUNT(*) FROM runs") == [(1,)]


def test_a_crashed_process_leaves_a_lease_that_survives_restarts(tmp_path: Path) -> None:
    db, (reply,) = run_ledger_processes(tmp_path, 1, "same-crash")
    assert reply["granted"] is True  # the child died right after leasing, without closing

    restarted = Ledger(db)  # "restart": a new interpreter-level view of the same file
    try:
        run = restarted.get(reply["run_id"])
        assert run is not None and run.status is RunStatus.RUNNING
        assert run.worker_id == "proc-0"
        assert restarted.lease(make_request(worker_id="restart")).reason == "duplicate-active"
    finally:
        restarted.close()

    later = Ledger(db, clock=lambda: datetime.now(UTC) + timedelta(seconds=10 * TTL))
    try:
        assert later.lease(make_request()).reason == "recovery-pending"
        assert [r.run_id for r in later.stale_runs()] == [reply["run_id"]]
        assert later.requeue_interrupted(reply["run_id"], 1) is True
        resumed = later.lease(make_request(worker_id="restart"))
        assert resumed.granted and resumed.run is not None
        assert (resumed.run.run_id, resumed.run.attempt) == (reply["run_id"], 2)
        assert len(later.list_runs()) == 1
    finally:
        later.close()
