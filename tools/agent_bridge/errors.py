"""Exception hierarchy for the agent bridge."""

from __future__ import annotations


class BridgeError(Exception):
    """Base class for every error the bridge raises on purpose."""


class ConfigError(BridgeError):
    """The bridge configuration is missing, malformed or unsafe."""


class TaskValidationError(BridgeError):
    """A GitHub issue does not contain a valid, strictly parsed task."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) if self.errors else "invalid task")


class GitHubError(BridgeError):
    """A GitHub API call failed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GitHubUnavailable(GitHubError):
    """The API could not be reached (network loss, 5xx, rate limit after retries)."""


class GitHubPermissionError(GitHubError):
    """The token is rejected or lacks permission for the call (401/403)."""


class GitHubNotFound(GitHubError):
    """The requested GitHub object does not exist (404)."""


class GitHubConflict(GitHubError):
    """GitHub refused the request because the object already exists or is invalid (422)."""


class GitError(BridgeError):
    """A git command failed."""

    def __init__(self, message: str, *, argv: list[str] | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.argv = list(argv or [])
        self.stderr = stderr


class PolicyViolation(BridgeError):
    """The agent produced a change set the bridge refuses to publish."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        super().__init__("; ".join(violations))


class RunStateError(BridgeError):
    """A ledger transition was refused: the run is not in a state that allows it.

    Terminal states are final, so a worker that lost its run (another process cancelled,
    blocked or finished it, or requeued it) is refused when it tries to move it on.
    """

    def __init__(self, run_id: str, status: str) -> None:
        super().__init__(f"run {run_id} is not active (status {status})")
        self.run_id = run_id
        self.status = status


class CheckAborted(BridgeError):
    """A verification check was stopped because the run was cancelled, edited or interrupted."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
