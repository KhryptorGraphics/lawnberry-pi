"""The agent backend interface and its structured result contract.

Every backend (OhMyPi ``omp``, Claude Code ``claude``) takes the same :class:`AgentRequest`
and returns the same :class:`AgentResult`, so the worker never needs to know which CLI ran.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import BackendConfig
from ..process import ProcessOutcome, ProcessSpec, SubprocessRunner
from ..redact import Redactor, truncate

__all__ = [
    "AgentBackend",
    "AgentReport",
    "AgentRequest",
    "AgentResult",
    "AgentStatus",
    "BASE_ENV_ALLOWLIST",
    "FailureKind",
    "StreamHandler",
    "classify_error_text",
    "extract_report",
]

# Environment variables every agent subprocess may inherit.  Everything else (GitHub tokens,
# cloud credentials, SSH agents, ...) is dropped unless a backend lists it in ``pass_env``.
BASE_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "all_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
)

# Names that must never reach an agent, even if an operator lists them in ``pass_env``.
FORBIDDEN_ENV_PREFIXES: tuple[str, ...] = ("LAWNBERRY_BRIDGE_",)
FORBIDDEN_ENV_NAMES: frozenset[str] = frozenset(
    {
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_PAT",
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
    }
)


class AgentStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"


class FailureKind(StrEnum):
    AUTH = "auth"
    CREDIT = "credit"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    BUDGET = "budget"
    MAX_TURNS = "max_turns"
    NONZERO_EXIT = "nonzero_exit"
    NO_RESULT = "no_result"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    HOOK = "hook"
    GUARD_INACTIVE = "guard_inactive"


class AgentReport(BaseModel):
    """The ``agent-result`` JSON block an agent ends its final message with (untrusted)."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["completed", "blocked", "needs_human"]
    summary: str = Field(max_length=4000)
    files_changed: list[str] = Field(default_factory=list, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)


# What may follow the backticks of an opening fence: ``agent-result`` or ``json agent-result``.
_REPORT_INFO = re.compile(r"\s*(?:json\s+)?agent-result\s*\Z")


def _report_bodies(text: str) -> list[str]:
    """Bodies of the ``agent-result`` fences in ``text``, in order, found in linear time.

    An opening fence is a line of up to three spaces, at least three backticks and the info
    string (see ``_REPORT_INFO``).  It closes at the first later line of up to three spaces, the
    *same number* of backticks and only whitespace.  An opener without a closer is skipped, and a
    block never starts inside the previous one.  (A regular expression with a lazy ``.*?`` tried
    from every opener is quadratic on a message made of unterminated openers.)
    """
    lines = text.split("\n")
    openers: list[tuple[int, int]] = []  # (line index, backticks)
    closers: dict[int, list[int]] = {}  # backticks -> indexes of lines that could close
    for index, line in enumerate(lines):
        body = line.lstrip(" ")
        if len(line) - len(body) > 3:
            continue
        rest = body.lstrip("`")
        ticks = len(body) - len(rest)
        if ticks < 3:
            continue
        if not rest or rest.isspace():
            closers.setdefault(ticks, []).append(index)
        elif _REPORT_INFO.match(rest):
            openers.append((index, ticks))
    bodies: list[str] = []
    resume = 0
    for index, ticks in openers:
        if index < resume:
            continue
        later = closers.get(ticks, [])
        found = bisect.bisect_right(later, index)
        if found == len(later):
            continue
        bodies.append("\n".join(lines[index + 1 : later[found]]))
        resume = later[found] + 1
    return bodies


def extract_report(text: str) -> AgentReport | None:
    """Parse the last well-formed ``agent-result`` block in ``text`` (or ``None``)."""
    for body in reversed(_report_bodies(text)):
        try:
            return AgentReport.model_validate(json.loads(body))
        except (ValueError, ValidationError, RecursionError):
            continue
    return None


def classify_error_text(text: str) -> FailureKind:
    """Map a provider/CLI error message to a :class:`FailureKind`."""
    low = text.lower()
    if any(
        s in low for s in ("credit balance", "insufficient credit", "out of credits", "billing")
    ):
        return FailureKind.CREDIT
    if any(
        s in low
        for s in (
            "failed to authenticate",
            "oauth",
            "invalid api key",
            "invalid x-api-key",
            "unauthorized",
            "authentication",
            "401",
            "403 ",
            "not logged in",
            "please run /login",
        )
    ):
        return FailureKind.AUTH
    if any(
        s in low for s in ("rate limit", "rate_limit", "429", "overloaded", "too many requests")
    ):
        return FailureKind.RATE_LIMIT
    return FailureKind.API_ERROR


@dataclass(frozen=True)
class AgentRequest:
    prompt: str
    cwd: Path
    run_dir: Path
    scope: Literal["docs-only", "code"]
    timeout_s: float
    budget_usd: float
    write_allow_sets: tuple[tuple[str, ...], ...] = ()
    write_deny: tuple[str, ...] = ()
    bash_allowlist: tuple[str, ...] = ()
    run_id: str = ""
    model: str | None = None
    abort_check: Callable[[], str | None] | None = None
    abort_poll_s: float = 5.0
    on_start: Callable[[int, int | None], None] | None = None


@dataclass
class AgentResult:
    """What a backend reports, identical for every CLI."""

    backend: str
    status: AgentStatus
    exit_code: int | None = None
    duration_s: float = 0.0
    final_text: str = ""
    report: AgentReport | None = None
    cost_usd: float | None = None
    budget_usd: float | None = None
    usage: dict[str, int | float] = field(default_factory=dict)
    model: str | None = None
    session_id: str | None = None
    stop_reason: str | None = None
    num_turns: int | None = None
    error: str | None = None
    failure_kind: FailureKind | None = None
    tool_calls: int | None = None
    blocked_tool_calls: int = 0
    blocked_samples: list[str] = field(default_factory=list)
    stdout_log: str | None = None
    stderr_log: str | None = None
    command: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status is AgentStatus.SUCCEEDED

    @property
    def over_budget(self) -> bool:
        return (
            self.cost_usd is not None
            and self.budget_usd is not None
            and self.cost_usd > self.budget_usd + 1e-9
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["failure_kind"] = self.failure_kind.value if self.failure_kind else None
        data["report"] = self.report.model_dump() if self.report else None
        data["ok"] = self.ok
        data["over_budget"] = self.over_budget
        return data


class StreamHandler:
    """Optional per-line hook (budget enforcement) for streaming CLIs."""

    def feed(self, line: str) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError


class AgentBackend(ABC):
    """Template for one agent CLI.  Subclasses supply argv construction and result parsing."""

    name: ClassVar[str]

    def __init__(
        self,
        cfg: BackendConfig,
        *,
        redactor: Redactor | None = None,
        runner: SubprocessRunner | None = None,
        command_prefix: Sequence[str] = (),
        kill_grace_s: float = 10.0,
        max_log_bytes: int = 4_000_000,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.redactor = redactor or Redactor()
        self.runner = runner or SubprocessRunner(redact=self.redactor.redact)
        self.command_prefix = list(command_prefix)
        self.kill_grace_s = kill_grace_s
        self.max_log_bytes = max_log_bytes
        self._environ = dict(os.environ if environ is None else environ)
        for name in cfg.pass_env:
            if name in FORBIDDEN_ENV_NAMES or name.startswith(FORBIDDEN_ENV_PREFIXES):
                raise ValueError(f"environment variable {name} must never be passed to an agent")

    # ---------------------------------------------------------------- introspection
    def resolve_binary(self) -> str | None:
        """Absolute path of the CLI, or ``None`` when it is not installed/executable."""
        candidate = self.cfg.binary
        if os.sep in candidate:
            path = Path(candidate).expanduser()
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        return shutil.which(candidate, path=self._environ.get("PATH"))

    def version(self, timeout_s: float = 20.0) -> str | None:
        """First line of ``<binary> --version`` (``None`` if unavailable)."""
        binary = self.resolve_binary()
        if binary is None:
            return None
        try:
            done = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=self._base_env(),
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        out = (done.stdout or done.stderr or "").strip().splitlines()
        return out[0][:120] if out else None

    # ---------------------------------------------------------------- environment
    def _base_env(self) -> dict[str, str]:
        env = {k: v for k, v in self._environ.items() if k in BASE_ENV_ALLOWLIST}
        for name in self.cfg.pass_env:
            if name in self._environ:
                env[name] = self._environ[name]
        env.setdefault("TERM", "dumb")
        env["NO_COLOR"] = "1"
        return env

    def build_env(self, req: AgentRequest) -> dict[str, str]:
        env = self._base_env()
        env["LAWNBERRY_AGENT_RUN_ID"] = req.run_id
        return env

    # ---------------------------------------------------------------- subclass hooks
    @abstractmethod
    def build_argv(self, req: AgentRequest, binary: str) -> list[str]:
        """Complete argv (starting with ``binary``); the prompt is sent on stdin."""

    def capture_limit(self) -> int:
        """Bytes of stdout/stderr kept in memory (logs on disk follow the same cap)."""
        return self.max_log_bytes

    def make_stream_handler(self, req: AgentRequest) -> StreamHandler | None:
        return None

    @abstractmethod
    def parse(
        self, outcome: ProcessOutcome, req: AgentRequest, handler: StreamHandler | None
    ) -> AgentResult:
        """Turn the raw process outcome into the shared :class:`AgentResult`."""

    # ---------------------------------------------------------------- template method
    def run(self, req: AgentRequest) -> AgentResult:
        binary = self.resolve_binary()
        if binary is None:
            return AgentResult(
                backend=self.name,
                status=AgentStatus.UNAVAILABLE,
                failure_kind=FailureKind.UNAVAILABLE,
                error=f"{self.cfg.binary!r} is not installed or not executable",
                budget_usd=req.budget_usd,
            )
        req.run_dir.mkdir(parents=True, exist_ok=True)
        argv = self.build_argv(req, binary)
        spec = ProcessSpec(
            argv=[*self.command_prefix, *argv],
            cwd=req.cwd,
            env=self.build_env(req),
            stdin_text=req.prompt,
            timeout_s=req.timeout_s,
            kill_grace_s=self.kill_grace_s,
            max_capture_bytes=self.capture_limit(),
            stdout_log=req.run_dir / f"{self.name}.stdout.log",
            stderr_log=req.run_dir / f"{self.name}.stderr.log",
        )
        handler = self.make_stream_handler(req)
        outcome = self.runner.run(
            spec,
            on_line=handler.feed if handler is not None else None,
            abort_check=req.abort_check,
            abort_poll_s=req.abort_poll_s,
            on_start=req.on_start,
        )
        if outcome.spawn_error is not None:
            return AgentResult(
                backend=self.name,
                status=AgentStatus.UNAVAILABLE,
                failure_kind=FailureKind.UNAVAILABLE,
                error=self.redactor.redact(outcome.spawn_error),
                command=argv,
                budget_usd=req.budget_usd,
            )
        result = self.parse(outcome, req, handler)
        result.command = argv
        result.exit_code = outcome.returncode
        result.duration_s = round(outcome.duration_s, 3)
        result.budget_usd = req.budget_usd
        result.stdout_log = str(spec.stdout_log)
        result.stderr_log = str(spec.stderr_log)
        result.truncated = outcome.stdout_truncated or outcome.stderr_truncated
        return result

    # ---------------------------------------------------------------- shared helpers
    def _finalize_text(self, text: str, limit: int) -> str:
        return truncate(self.redactor.redact(text).strip(), limit)

    def _interrupted(
        self, outcome: ProcessOutcome, req: AgentRequest, **partial: Any
    ) -> AgentResult | None:
        """Result for runs the bridge cut short (timeout, budget, cancellation, hook failure)."""
        if outcome.timed_out:
            return AgentResult(
                backend=self.name,
                status=AgentStatus.TIMED_OUT,
                failure_kind=FailureKind.TIMEOUT,
                error=f"agent exceeded the {req.timeout_s:.0f}s timeout and was terminated",
                **partial,
            )
        reason = outcome.kill_reason
        if not reason:
            return None
        if reason.startswith("budget"):
            return AgentResult(
                backend=self.name,
                status=AgentStatus.BUDGET_EXCEEDED,
                failure_kind=FailureKind.BUDGET,
                error=reason,
                **partial,
            )
        if reason.startswith("hook-error"):
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=FailureKind.HOOK,
                error=f"stream handler failed ({reason}); run terminated",
                **partial,
            )
        return AgentResult(
            backend=self.name,
            status=AgentStatus.CANCELLED,
            failure_kind=FailureKind.CANCELLED,
            error=reason,
            **partial,
        )
