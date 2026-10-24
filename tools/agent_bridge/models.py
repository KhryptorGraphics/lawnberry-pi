"""Plain data types shared by the GitHub client, ledger and worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Approval",
    "Comment",
    "Issue",
    "IssueEvent",
    "PullRequest",
    "RunPhase",
    "RunStatus",
]


class RunStatus(StrEnum):
    """Coarse life-cycle state stored in the ledger."""

    PENDING = "pending"
    RUNNING = "running"
    PUBLISHING = "publishing"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.DONE, RunStatus.BLOCKED, RunStatus.CANCELLED)

    @property
    def active(self) -> bool:
        return self in (RunStatus.RUNNING, RunStatus.PUBLISHING)


class RunPhase(StrEnum):
    """Durable checkpoints; recovery resumes from the last one reached."""

    LEASED = "leased"
    WORKTREE = "worktree"
    AGENT = "agent"
    VERIFY = "verify"
    COMMIT = "commit"
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    NOTIFY = "notify"
    FINISHED = "finished"


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    state: str
    author: str
    labels: frozenset[str]
    state_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""
    html_url: str = ""
    is_pull_request: bool = False

    @property
    def is_open(self) -> bool:
        return self.state == "open"


@dataclass(frozen=True)
class IssueEvent:
    id: int
    event: str
    actor: str | None
    created_at: str
    label: str | None = None
    rename_from: str | None = None
    rename_to: str | None = None


@dataclass(frozen=True)
class Approval:
    """The GitHub event through which an approver applied the approval label."""

    event_id: int
    actor: str
    created_at: str


@dataclass(frozen=True)
class Comment:
    id: int
    body: str
    author: str
    created_at: str = ""


@dataclass(frozen=True)
class PullRequest:
    number: int
    html_url: str
    state: str
    draft: bool
    head_ref: str
    base_ref: str
    title: str = ""
    extra: dict[str, object] = field(default_factory=dict)
