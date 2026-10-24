"""Durable SQLite ledger: one row per approved task run, plus an append-only event log.

The ledger is the source of truth for idempotence.  A run is keyed by
``(repository, issue, approval event id)``: one approval can start at most one run, a new run
needs a new approval event, and a crashed worker is recovered from the last durable phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import RunPhase, RunStatus

__all__ = [
    "LeaseRequest",
    "LeaseResult",
    "Ledger",
    "RunRecord",
    "dedup_key",
    "iso",
    "utcnow",
]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    repo TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    issue_title TEXT NOT NULL,
    revision TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    approval_event_id INTEGER NOT NULL,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    dedup_key TEXT NOT NULL UNIQUE,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    interrupted_count INTEGER NOT NULL DEFAULT 0,
    publish_attempts INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    worker_pid INTEGER,
    lease_expires_at TEXT,
    heartbeat_at TEXT,
    agent_pid INTEGER,
    agent_start_ticks INTEGER,
    branch TEXT,
    worktree_path TEXT,
    base_sha TEXT,
    commit_sha TEXT,
    pr_number INTEGER,
    pr_url TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    failure_kind TEXT,
    failure_detail TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE (repo, issue_number, approval_event_id)
);
CREATE INDEX IF NOT EXISTS runs_issue ON runs (repo, issue_number);
CREATE INDEX IF NOT EXISTS runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS runs_spec ON runs (repo, spec_hash);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    repo TEXT NOT NULL,
    issue_number INTEGER,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events (run_id);
CREATE TABLE IF NOT EXISTS notices (
    key TEXT PRIMARY KEY,
    at TEXT NOT NULL
);
"""

_UPDATABLE = frozenset(
    {
        "agent_pid",
        "agent_start_ticks",
        "backend",
        "base_sha",
        "branch",
        "commit_sha",
        "failure_detail",
        "failure_kind",
        "pr_number",
        "pr_url",
        "result_json",
        "worktree_path",
    }
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def dedup_key(repo: str, issue_number: int, approval_event_id: int, revision: str) -> str:
    return hashlib.sha256(
        f"{repo}|{issue_number}|{approval_event_id}|{revision}".encode()
    ).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    repo: str
    issue_number: int
    issue_title: str
    revision: str
    spec_hash: str
    approval_event_id: int
    approved_by: str
    approved_at: str
    dedup_key: str
    backend: str
    status: RunStatus
    phase: RunPhase
    attempt: int
    interrupted_count: int
    publish_attempts: int
    worker_id: str | None
    worker_pid: int | None
    lease_expires_at: str | None
    heartbeat_at: str | None
    agent_pid: int | None
    agent_start_ticks: int | None
    branch: str | None
    worktree_path: str | None
    base_sha: str | None
    commit_sha: str | None
    pr_number: int | None
    pr_url: str | None
    cancel_requested: bool
    failure_kind: str | None
    failure_detail: str | None
    result: dict[str, Any] | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RunRecord:
        raw = row["result_json"]
        return cls(
            run_id=row["run_id"],
            repo=row["repo"],
            issue_number=row["issue_number"],
            issue_title=row["issue_title"],
            revision=row["revision"],
            spec_hash=row["spec_hash"],
            approval_event_id=row["approval_event_id"],
            approved_by=row["approved_by"],
            approved_at=row["approved_at"],
            dedup_key=row["dedup_key"],
            backend=row["backend"],
            status=RunStatus(row["status"]),
            phase=RunPhase(row["phase"]),
            attempt=row["attempt"],
            interrupted_count=row["interrupted_count"],
            publish_attempts=row["publish_attempts"],
            worker_id=row["worker_id"],
            worker_pid=row["worker_pid"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            agent_pid=row["agent_pid"],
            agent_start_ticks=row["agent_start_ticks"],
            branch=row["branch"],
            worktree_path=row["worktree_path"],
            base_sha=row["base_sha"],
            commit_sha=row["commit_sha"],
            pr_number=row["pr_number"],
            pr_url=row["pr_url"],
            cancel_requested=bool(row["cancel_requested"]),
            failure_kind=row["failure_kind"],
            failure_detail=row["failure_detail"],
            result=json.loads(raw) if raw else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


@dataclass(frozen=True)
class LeaseRequest:
    repo: str
    issue_number: int
    issue_title: str
    revision: str
    spec_hash: str
    approval_event_id: int
    approved_by: str
    approved_at: str
    backend: str
    worker_id: str
    worker_pid: int
    ttl_seconds: int
    max_attempts_per_issue: int
    dedupe_by_content: bool = True


@dataclass(frozen=True)
class LeaseResult:
    granted: bool
    run: RunRecord | None = None
    reason: str | None = None
    other: RunRecord | None = field(default=None)

    def __bool__(self) -> bool:
        return self.granted


class Ledger:
    """SQLite-backed run ledger (single process, thread-safe)."""

    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] = utcnow) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        # Create the file private from the first byte: SQLite gives the -wal/-shm companions
        # the same mode as the database file, so they are never briefly world readable.
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        self._cx = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False, timeout=30
        )
        self._cx.row_factory = sqlite3.Row
        deadline = time.monotonic() + 20.0
        while True:
            # Processes opening a brand-new ledger at the same moment race on the switch to WAL
            # mode; SQLite reports that as "locked" without waiting, so retry the whole setup.
            try:
                self._cx.execute("PRAGMA journal_mode=WAL")
                self._cx.execute("PRAGMA synchronous=FULL")
                self._cx.execute("PRAGMA busy_timeout=30000")
                self._migrate()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    # ------------------------------------------------------------------ plumbing
    def close(self) -> None:
        with self._lock:
            self._cx.close()

    def now(self) -> datetime:
        return self._clock()

    def _stamp(self) -> str:
        return iso(self._clock())

    def _migrate(self) -> None:
        with self._lock:
            version = self._cx.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"ledger {self.path} has schema v{version}; "
                    f"this bridge supports v{SCHEMA_VERSION}"
                )
            if version < SCHEMA_VERSION:
                self._cx.executescript(_SCHEMA)
                self._cx.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                try:
                    self.path.chmod(0o600)
                except OSError:
                    pass

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._cx.execute("BEGIN IMMEDIATE")
            try:
                yield self._cx
            except BaseException:
                self._cx.execute("ROLLBACK")
                raise
            else:
                self._cx.execute("COMMIT")

    def _fetch(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._cx.execute(sql, params))

    # ------------------------------------------------------------------ leasing
    def lease(self, req: LeaseRequest) -> LeaseResult:
        """Atomically claim the right to execute one approved task.

        Denials never raise: ``reason`` is one of ``busy``, ``duplicate-active``,
        ``already-processed``, ``recovery-pending``, ``attempt-limit`` or ``duplicate-content``.
        """
        now = self._clock()
        stamp = iso(now)
        expires = iso(now + timedelta(seconds=req.ttl_seconds))
        with self._tx() as cx:
            existing = cx.execute(
                "SELECT * FROM runs WHERE repo=? AND issue_number=? AND approval_event_id=?",
                (req.repo, req.issue_number, req.approval_event_id),
            ).fetchone()
            active = cx.execute(
                "SELECT * FROM runs WHERE status IN ('running','publishing') ORDER BY id"
            ).fetchall()
            if existing is not None:
                record = RunRecord.from_row(existing)
                if record.status.terminal:
                    return LeaseResult(False, reason="already-processed", other=record)
                if record.status.active:
                    live = record.lease_expires_at is not None and record.lease_expires_at > stamp
                    return LeaseResult(
                        False,
                        reason="duplicate-active" if live else "recovery-pending",
                        other=record,
                    )
                # status == pending: an interrupted run waiting for its automatic retry
                if active:
                    return LeaseResult(False, reason="busy", other=RunRecord.from_row(active[0]))
                cx.execute(
                    "UPDATE runs SET status='running', phase='leased', worker_id=?, worker_pid=?, "
                    "lease_expires_at=?, heartbeat_at=?, started_at=COALESCE(started_at, ?), "
                    "updated_at=?, agent_pid=NULL, agent_start_ticks=NULL WHERE run_id=?",
                    (
                        req.worker_id,
                        req.worker_pid,
                        expires,
                        stamp,
                        stamp,
                        stamp,
                        record.run_id,
                    ),
                )
                self._event(cx, record.run_id, req.repo, req.issue_number, "lease.resumed", {})
                row = cx.execute("SELECT * FROM runs WHERE run_id=?", (record.run_id,)).fetchone()
                return LeaseResult(True, run=RunRecord.from_row(row))
            if active:
                return LeaseResult(False, reason="busy", other=RunRecord.from_row(active[0]))
            attempts = cx.execute(
                "SELECT COUNT(*) FROM runs WHERE repo=? AND issue_number=?",
                (req.repo, req.issue_number),
            ).fetchone()[0]
            if attempts >= req.max_attempts_per_issue:
                return LeaseResult(False, reason="attempt-limit")
            if req.dedupe_by_content:
                twin = cx.execute(
                    "SELECT * FROM runs WHERE repo=? AND spec_hash=? AND issue_number<>? "
                    "AND status IN ('pending','running','publishing','done') ORDER BY id LIMIT 1",
                    (req.repo, req.spec_hash, req.issue_number),
                ).fetchone()
                if twin is not None:
                    return LeaseResult(
                        False, reason="duplicate-content", other=RunRecord.from_row(twin)
                    )
            run_id = uuid4().hex
            cx.execute(
                "INSERT INTO runs (run_id, repo, issue_number, issue_title, revision, spec_hash, "
                "approval_event_id, approved_by, approved_at, dedup_key, backend, status, phase, "
                "worker_id, worker_pid, lease_expires_at, heartbeat_at, created_at, updated_at, "
                "started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,'running','leased',?,?,?,?,?,?,?)",
                (
                    run_id,
                    req.repo,
                    req.issue_number,
                    req.issue_title[:300],
                    req.revision,
                    req.spec_hash,
                    req.approval_event_id,
                    req.approved_by,
                    req.approved_at,
                    dedup_key(req.repo, req.issue_number, req.approval_event_id, req.revision),
                    req.backend,
                    req.worker_id,
                    req.worker_pid,
                    expires,
                    stamp,
                    stamp,
                    stamp,
                    stamp,
                ),
            )
            self._event(
                cx,
                run_id,
                req.repo,
                req.issue_number,
                "lease.granted",
                {"approval_event_id": req.approval_event_id, "backend": req.backend},
            )
            row = cx.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return LeaseResult(True, run=RunRecord.from_row(row))

    # ------------------------------------------------------------------ reads
    def get(self, run_id: str) -> RunRecord | None:
        rows = self._fetch("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return RunRecord.from_row(rows[0]) if rows else None

    def find_by_approval(
        self, repo: str, issue_number: int, approval_event_id: int
    ) -> RunRecord | None:
        rows = self._fetch(
            "SELECT * FROM runs WHERE repo=? AND issue_number=? AND approval_event_id=?",
            (repo, issue_number, approval_event_id),
        )
        return RunRecord.from_row(rows[0]) if rows else None

    def runs_for_issue(self, repo: str, issue_number: int) -> list[RunRecord]:
        rows = self._fetch(
            "SELECT * FROM runs WHERE repo=? AND issue_number=? ORDER BY id", (repo, issue_number)
        )
        return [RunRecord.from_row(r) for r in rows]

    def list_runs(self, *, limit: int = 50, status: RunStatus | None = None) -> list[RunRecord]:
        if status is None:
            rows = self._fetch("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self._fetch(
                "SELECT * FROM runs WHERE status=? ORDER BY id DESC LIMIT ?", (status.value, limit)
            )
        return [RunRecord.from_row(r) for r in rows]

    def active_runs(self) -> list[RunRecord]:
        rows = self._fetch(
            "SELECT * FROM runs WHERE status IN ('running','publishing') ORDER BY id"
        )
        return [RunRecord.from_row(r) for r in rows]

    def pending_runs(self) -> list[RunRecord]:
        rows = self._fetch("SELECT * FROM runs WHERE status='pending' ORDER BY id")
        return [RunRecord.from_row(r) for r in rows]

    def stale_runs(self) -> list[RunRecord]:
        """Active runs whose lease has expired (the worker holding them is gone)."""
        rows = self._fetch(
            "SELECT * FROM runs WHERE status IN ('running','publishing') "
            "AND (lease_expires_at IS NULL OR lease_expires_at <= ?) ORDER BY id",
            (self._stamp(),),
        )
        return [RunRecord.from_row(r) for r in rows]

    # ------------------------------------------------------------------ writes
    def heartbeat(self, run_id: str, ttl_seconds: int) -> None:
        now = self._clock()
        with self._tx() as cx:
            cx.execute(
                "UPDATE runs SET heartbeat_at=?, lease_expires_at=?, updated_at=? "
                "WHERE run_id=? AND status IN ('running','publishing')",
                (iso(now), iso(now + timedelta(seconds=ttl_seconds)), iso(now), run_id),
            )

    def claim_publishing(
        self, run_id: str, worker_id: str, worker_pid: int, ttl_seconds: int
    ) -> bool:
        """Take over a stale ``publishing`` run (or refresh our own) for the next attempt."""
        now = self._clock()
        with self._tx() as cx:
            cur = cx.execute(
                "UPDATE runs SET worker_id=?, worker_pid=?, heartbeat_at=?, lease_expires_at=?, "
                "updated_at=?, publish_attempts=publish_attempts+1 "
                "WHERE run_id=? AND status='publishing'",
                (
                    worker_id,
                    worker_pid,
                    iso(now),
                    iso(now + timedelta(seconds=ttl_seconds)),
                    iso(now),
                    run_id,
                ),
            )
            return cur.rowcount == 1

    def update(self, run_id: str, **fields: Any) -> None:
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"cannot update ledger columns: {sorted(bad)}")
        if not fields:
            return
        if "result_json" in fields and not isinstance(fields["result_json"], str | type(None)):
            fields["result_json"] = json.dumps(fields["result_json"], sort_keys=True)
        assignments = ", ".join(f"{name}=?" for name in fields)
        with self._tx() as cx:
            cx.execute(
                f"UPDATE runs SET {assignments}, updated_at=? WHERE run_id=?",
                (*fields.values(), self._stamp(), run_id),
            )

    def set_phase(
        self,
        run_id: str,
        phase: RunPhase,
        *,
        status: RunStatus | None = None,
        **fields: Any,
    ) -> None:
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"cannot update ledger columns: {sorted(bad)}")
        if "result_json" in fields and not isinstance(fields["result_json"], str | type(None)):
            fields["result_json"] = json.dumps(fields["result_json"], sort_keys=True)
        columns = ["phase=?", "updated_at=?"]
        params: list[Any] = [phase.value, self._stamp()]
        if status is not None:
            columns.append("status=?")
            params.append(status.value)
        for name, value in fields.items():
            columns.append(f"{name}=?")
            params.append(value)
        params.append(run_id)
        with self._tx() as cx:
            cx.execute(f"UPDATE runs SET {', '.join(columns)} WHERE run_id=?", params)
            row = cx.execute(
                "SELECT repo, issue_number FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is not None:
                self._event(
                    cx, run_id, row["repo"], row["issue_number"], f"phase.{phase.value}", {}
                )

    def finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        failure_kind: str | None = None,
        failure_detail: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        if not status.terminal:
            raise ValueError("finish() needs a terminal status")
        stamp = self._stamp()
        with self._tx() as cx:
            cx.execute(
                "UPDATE runs SET status=?, phase='finished', failure_kind=?, failure_detail=?, "
                "result_json=COALESCE(?, result_json), finished_at=?, updated_at=?, "
                "lease_expires_at=NULL, agent_pid=NULL, agent_start_ticks=NULL WHERE run_id=?",
                (
                    status.value,
                    failure_kind,
                    (failure_detail or "")[:4000] or None,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    stamp,
                    stamp,
                    run_id,
                ),
            )
            row = cx.execute(
                "SELECT repo, issue_number FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is not None:
                self._event(
                    cx,
                    run_id,
                    row["repo"],
                    row["issue_number"],
                    f"finish.{status.value}",
                    {"failure_kind": failure_kind},
                )

    def requeue_interrupted(self, run_id: str, max_auto_retries: int) -> bool:
        """Requeue a ``running`` run if its retry budget allows; any other status returns False."""
        stamp = self._stamp()
        with self._tx() as cx:
            row = cx.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if (
                row is None
                or row["status"] != RunStatus.RUNNING.value
                or row["interrupted_count"] >= max_auto_retries
            ):
                return False
            cur = cx.execute(
                "UPDATE runs SET status='pending', phase='leased', attempt=attempt+1, "
                "interrupted_count=interrupted_count+1, lease_expires_at=NULL, agent_pid=NULL, "
                "agent_start_ticks=NULL, updated_at=? WHERE run_id=? AND status='running'",
                (stamp, run_id),
            )
            if cur.rowcount != 1:
                return False
            self._event(
                cx,
                run_id,
                row["repo"],
                row["issue_number"],
                "run.requeued",
                {"after": "interrupted"},
            )
            return True

    def request_cancel(self, *, run_id: str | None = None, issue_number: int | None = None) -> int:
        """Ask a running/pending run to stop; returns the number of rows flagged."""
        if run_id is None and issue_number is None:
            raise ValueError("run_id or issue_number is required")
        clause, param = ("run_id=?", run_id) if run_id else ("issue_number=?", issue_number)
        with self._tx() as cx:
            cur = cx.execute(
                f"UPDATE runs SET cancel_requested=1, updated_at=? WHERE {clause} "
                "AND status IN ('pending','running','publishing')",
                (self._stamp(), param),
            )
            return cur.rowcount

    def is_cancel_requested(self, run_id: str) -> bool:
        rows = self._fetch("SELECT cancel_requested FROM runs WHERE run_id=?", (run_id,))
        return bool(rows and rows[0][0])

    # ------------------------------------------------------------------ events & notices
    def _event(
        self,
        cx: sqlite3.Connection,
        run_id: str | None,
        repo: str,
        issue_number: int | None,
        kind: str,
        detail: dict[str, Any],
    ) -> None:
        cx.execute(
            "INSERT INTO events (run_id, repo, issue_number, at, kind, detail) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, repo, issue_number, self._stamp(), kind, json.dumps(detail, sort_keys=True)),
        )

    def record_event(
        self,
        kind: str,
        *,
        repo: str,
        issue_number: int | None = None,
        run_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._tx() as cx:
            self._event(cx, run_id, repo, issue_number, kind, detail or {})

    def events(
        self, *, run_id: str | None = None, issue_number: int | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        if issue_number is not None:
            clauses.append("issue_number=?")
            params.append(issue_number)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetch(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", (*params, limit)
        )
        return [
            {
                "id": r["id"],
                "run_id": r["run_id"],
                "issue_number": r["issue_number"],
                "at": r["at"],
                "kind": r["kind"],
                "detail": json.loads(r["detail"]),
            }
            for r in reversed(rows)
        ]

    def notice_once(self, key: str) -> bool:
        """True the first time ``key`` is seen (used to post each bot comment only once)."""
        with self._tx() as cx:
            cur = cx.execute(
                "INSERT OR IGNORE INTO notices (key, at) VALUES (?, ?)", (key, self._stamp())
            )
            return cur.rowcount == 1

    def forget_notice(self, key: str) -> None:
        """Undo :meth:`notice_once` when the comment it guarded could not be delivered."""
        with self._tx() as cx:
            cx.execute("DELETE FROM notices WHERE key=?", (key,))
