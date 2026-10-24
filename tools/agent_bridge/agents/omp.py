"""OhMyPi (``omp``) backend - the default agent.

Invocation (flags verified against ``omp --help`` of v18.12.6, except ``--trusted-extension``:
a hidden flag present in the 18.12.6 bundle that takes an absolute path and cannot be combined
with ``-e/--extension/--hook``)::

    omp -p --mode json --no-session --no-title --no-lsp --no-pty --no-skills --no-rules \\
        --tools read,grep,glob,edit,write --approval-mode write \\
        --config <hermetic overlay> --trusted-extension <omp_guard.mjs> \\
        --max-time <s> --cwd <worktree>            # prompt arrives on stdin

``--mode json`` streams one JSON event per line.  The bridge parses the stream as it arrives to
track cost (omp has no native spend cap, so the budget is enforced here between model calls),
count tool calls, and read the final assistant message.  The guard extension confines every tool
call to the task worktree and the task's allowed paths.  A run that used tools only counts as
succeeded when the guard's own audit log holds a row for every call (``guard_inactive`` if not).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..process import ProcessOutcome
from ..redact import strip_control_chars
from .base import (
    AgentBackend,
    AgentRequest,
    AgentResult,
    AgentStatus,
    FailureKind,
    StreamHandler,
    classify_error_text,
    extract_report,
)

__all__ = [
    "EDIT_MODE",
    "GUARD_AUDIT_MAX_ROWS",
    "GUARD_AUDIT_NAME",
    "GUARD_PATH",
    "GUARD_PREFIX",
    "GuardAudit",
    "OmpBackend",
    "OmpStreamHandler",
    "hermetic_overlay",
]

GUARD_PATH = Path(__file__).resolve().parent.parent / "omp_guard.mjs"
GUARD_PREFIX = "[agent-bridge guard]"
FINAL_TEXT_STORE_CHARS = 20_000

# The guard appends one JSON row per tool-call decision to this file of the run directory and
# stops after GUARD_AUDIT_MAX_ROWS rows (omp_guard.mjs: MAX_AUDIT_LINES).  The reader looks at
# no more than that many lines of at most GUARD_AUDIT_MAX_LINE characters each: the file lives
# next to code the agent may have run, so its size is not trusted.
GUARD_AUDIT_NAME = "omp-guard-audit.jsonl"
GUARD_AUDIT_MAX_ROWS = 5000
GUARD_AUDIT_MAX_LINE = 65_536

# omp's `edit` tool has five syntaxes and picks one per model (default: hashline).  Only
# `replace` takes its target in a `path` argument; hashline, apply_patch and sloppy put file
# headers and MV/REM/rename targets inside a text `input` that the guard refuses to trust.  The
# mode is pinned both in the settings overlay and with the variant environment variable, which
# wins over every per-model default.
EDIT_MODE = "replace"

# Every omp discovery provider except the built-in default rules.  Disabling them removes
# ambient MCP servers, plugins, skills, hooks, rules and context files; the bridge supplies the
# repository context itself so both backends see the same instructions.
DISABLED_PROVIDERS: tuple[str, ...] = (
    "native",
    "omp-plugins",
    "claude",
    "agent-plugins",
    "codex",
    "agents",
    "claude-plugins",
    "gemini",
    "opencode",
    "cursor",
    "windsurf",
    "cline",
    "github",
    "vscode",
    "agents-md",
    "claude-md",
    "custom-context",
    "mcp-json",
    "ssh-json",
)

READ_DENY: tuple[str, ...] = (
    ".git",
    ".git/**",
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
    "config/secrets*.json",
    "config/ssl/**",
)

MAX_WRITE_BYTES = 1_000_000


def hermetic_overlay() -> dict[str, Any]:
    """Per-run omp settings overlay: nothing ambient, no fallback models, no telemetry."""
    return {
        "disabledProviders": list(DISABLED_PROVIDERS),
        "mcp": {"enableProjectConfig": False},
        "tools": {"approvalMode": "write", "xdev": False},
        "edit": {"mode": EDIT_MODE},
        "retry": {"modelFallback": False},
        "memory": {"backend": "off"},
        "autolearn": {"enabled": False},
        "dev": {"autoqa": False},
        "telemetry": {"otlpExportEnabled": False},
        "skills": {"enabled": False},
    }


def _result_text(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    return "".join(
        part.get("text", "")
        for part in result.get("content", [])
        if isinstance(part, dict) and part.get("type") == "text"
    )


@dataclass(frozen=True)
class GuardAudit:
    """What the guard's own audit log says about one run."""

    rows: int | None  # decisions recorded; None when the log is missing or unreadable
    blocked: int  # tool calls the guard refused
    samples: list[str]  # cleaned "<tool>: <reason>" lines of the first refused calls


def _audit_row(line: str) -> dict[str, Any] | None:
    """One audit line as a row, or ``None`` when it is not a well-formed guard decision."""
    try:
        row = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if (
        isinstance(row, dict)
        and isinstance(row.get("tool"), str)
        and isinstance(row.get("allow"), bool)
    ):
        return row
    return None


class OmpStreamHandler(StreamHandler):
    """Incremental parser for ``omp --mode json`` output with budget enforcement."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.cost_usd = 0.0
        self.usage: dict[str, float] = {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
        }
        self.session_id: str | None = None
        self.model: str | None = None
        self.provider: str | None = None
        self.stop_reason: str | None = None
        self.error_message: str | None = None
        self.final_text = ""
        self.agent_end = False
        self.terminal = False
        self.assistant_messages = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.blocked: list[str] = []
        self.blocked_count = 0
        self.warnings: list[str] = []
        self._seen: set[str] = set()

    def feed(self, line: str) -> str | None:
        line = line.strip()
        if not line.startswith("{"):
            return None
        try:
            event = json.loads(line)
        except ValueError:
            return None
        kind = event.get("type")
        if kind == "session":
            self.session_id = event.get("id") or self.session_id
        elif kind == "notice" and event.get("level") in {"warn", "warning", "error"}:
            if len(self.warnings) < 10:
                self.warnings.append(str(event.get("message", ""))[:300])
        elif kind == "tool_execution_start":
            self.tool_calls += 1
        elif kind == "tool_execution_end":
            if event.get("isError"):
                self.tool_errors += 1
                text = _result_text(event.get("result"))
                if text.startswith(GUARD_PREFIX):
                    self.blocked_count += 1
                    if len(self.blocked) < 8:
                        self.blocked.append(text[:200])
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                return self._assistant(message)
        elif kind == "agent_end":
            self.agent_end = True
            self.terminal = bool(event.get("isTerminal"))
        return None

    def _assistant(self, message: dict[str, Any]) -> str | None:
        key = str(message.get("responseId") or message.get("timestamp") or self.assistant_messages)
        if key in self._seen:
            return None
        self._seen.add(key)
        self.assistant_messages += 1
        usage = message.get("usage") or {}
        for name in self.usage:
            value = usage.get(name)
            if isinstance(value, int | float):
                self.usage[name] += value
        cost = (usage.get("cost") or {}).get("total")
        if isinstance(cost, int | float):
            self.cost_usd += float(cost)
        self.model = message.get("model") or self.model
        self.provider = message.get("provider") or self.provider
        self.stop_reason = message.get("stopReason")
        self.error_message = message.get("errorMessage") or self.error_message
        text = "".join(
            part.get("text", "")
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if text.strip():
            self.final_text = text
        # A final answer that happens to cross the line is kept; only a run that wants to keep
        # going (tool use) is cut off.
        if self.cost_usd >= self.budget_usd and self.stop_reason in {"toolUse", "length"}:
            return (
                f"budget: spent ${self.cost_usd:.4f} of the ${self.budget_usd:.2f} budget; "
                "run terminated before the next model call"
            )
        return None


class OmpBackend(AgentBackend):
    name = "omp"

    # ------------------------------------------------------------------ request shaping
    def tool_names(self, req: AgentRequest) -> list[str]:
        tools = ["read", "grep", "glob", "edit", "write"]
        if req.bash_allowlist:
            tools.append("bash")
        return tools

    def _write_run_files(self, req: AgentRequest, tools: list[str]) -> Path:
        overlay = req.run_dir / "omp-overlay.yml"
        overlay.write_text(yaml.safe_dump(hermetic_overlay(), sort_keys=True), encoding="utf-8")
        audit = req.run_dir / GUARD_AUDIT_NAME
        # A retried run reuses its directory and the guard only appends: rows an earlier attempt
        # left behind must not vouch for this one.
        with contextlib.suppress(OSError):
            audit.unlink(missing_ok=True)
        policy = req.run_dir / "omp-guard-policy.json"
        policy.write_text(
            json.dumps(
                {
                    "version": 1,
                    "root": str(req.cwd.resolve()),
                    "tools": tools,
                    "write_allow_sets": [list(s) for s in req.write_allow_sets],
                    "write_deny": list(req.write_deny),
                    "read_deny": list(READ_DENY),
                    "bash_exact": list(req.bash_allowlist),
                    "audit_log": str(audit),
                    "max_write_bytes": MAX_WRITE_BYTES,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return overlay

    def build_argv(self, req: AgentRequest, binary: str) -> list[str]:
        tools = self.tool_names(req)
        overlay = self._write_run_files(req, tools)
        argv = [
            binary,
            "-p",
            "--mode",
            "json",
            "--no-session",
            "--no-title",
            "--no-skills",
            "--no-rules",
            "--no-lsp",
            "--no-pty",
            "--tools",
            ",".join(tools),
            "--approval-mode",
            "write",
            "--config",
            str(overlay),
            "--trusted-extension",
            str(GUARD_PATH),
            "--max-time",
            str(max(10, int(req.timeout_s) - 5)),
            "--cwd",
            str(req.cwd),
        ]
        model = req.model or self.cfg.model
        if model:
            argv += ["--model", model]
        if self.cfg.thinking:
            argv += ["--thinking", self.cfg.thinking]
        argv += list(self.cfg.extra_args)
        return argv

    def build_env(self, req: AgentRequest) -> dict[str, str]:
        env = super().build_env(req)
        env["LAWNBERRY_BRIDGE_GUARD_POLICY"] = str(req.run_dir / "omp-guard-policy.json")
        env["PI_NO_TITLE"] = "1"
        env["PI_EDIT_VARIANT"] = EDIT_MODE
        env["OMP_MCP_TIMEOUT_MS"] = "2000"
        return env

    def make_stream_handler(self, req: AgentRequest) -> StreamHandler:
        return OmpStreamHandler(req.budget_usd)

    # ------------------------------------------------------------------ parsing
    def parse(
        self, outcome: ProcessOutcome, req: AgentRequest, handler: StreamHandler | None
    ) -> AgentResult:
        assert isinstance(handler, OmpStreamHandler)
        audit = self._guard_audit(req, handler)
        full_text = self._finalize_text(handler.final_text, 400_000)
        common: dict[str, Any] = {
            "cost_usd": round(handler.cost_usd, 6) if handler.assistant_messages else None,
            "usage": {k: v for k, v in handler.usage.items() if v},
            "model": handler.model,
            "session_id": handler.session_id,
            "stop_reason": handler.stop_reason,
            "num_turns": handler.assistant_messages or None,
            "tool_calls": handler.tool_calls,
            "blocked_tool_calls": audit.blocked,
            "blocked_samples": audit.samples,
            "report": extract_report(full_text),
            "final_text": full_text[-FINAL_TEXT_STORE_CHARS:],
        }
        cut = self._interrupted(outcome, req, **common)
        if cut is not None:
            return cut
        stderr_tail = self._finalize_text(outcome.stderr[-2000:], 600)
        if handler.stop_reason in {"error", "aborted"}:
            message = self._finalize_text(
                handler.error_message or stderr_tail or "agent error", 600
            )
            if handler.stop_reason == "aborted" and "deadline" in message.lower():
                return AgentResult(
                    backend=self.name,
                    status=AgentStatus.TIMED_OUT,
                    failure_kind=FailureKind.TIMEOUT,
                    error="omp reached its --max-time deadline and stopped",
                    **common,
                )
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=classify_error_text(message),
                error=message,
                **common,
            )
        if outcome.returncode not in (0, None):
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=FailureKind.NONZERO_EXIT,
                error=stderr_tail or f"omp exited with status {outcome.returncode}",
                **common,
            )
        if not handler.agent_end or handler.stop_reason != "stop":
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=FailureKind.NO_RESULT,
                error=stderr_tail or "omp ended without a terminal result event",
                **common,
            )
        gap = self._guard_gap(audit, handler)
        if gap is not None:
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=FailureKind.GUARD_INACTIVE,
                error=gap,
                **common,
            )
        return AgentResult(backend=self.name, status=AgentStatus.SUCCEEDED, **common)

    @staticmethod
    def _guard_gap(audit: GuardAudit, handler: OmpStreamHandler) -> str | None:
        """Why the guard cannot be shown to have seen every tool call (``None``: it can).

        The guard is the only control for reads, URL fetches and writes outside the worktree.  An
        omp that loads it without ever firing ``tool_call`` would leave every call unguarded, so a
        run that used tools must leave one audit row per call (the guard stops at the cap).
        """
        calls = handler.tool_calls
        if calls <= 0:
            return None
        required = min(calls, GUARD_AUDIT_MAX_ROWS)
        if audit.rows is None:
            return (
                f"omp ran {calls} tool call(s) but the guard audit log ({GUARD_AUDIT_NAME}) is "
                "missing or unreadable (0 row(s) recorded); the guard extension did not confine "
                "them"
            )
        if audit.rows < required:
            return (
                f"omp ran {calls} tool call(s) but the guard audit log ({GUARD_AUDIT_NAME}) has "
                f"only {audit.rows} row(s), at least {required} expected; the guard extension "
                "did not see every call"
            )
        return None

    def _guard_audit(self, req: AgentRequest, handler: OmpStreamHandler) -> GuardAudit:
        """What the guard's own audit log says: rows recorded, blocked calls and samples.

        Guard blocks that the stream reported count too (the larger number wins) and supply the
        samples when the log has none.  Everything copied out of either source is agent-controlled
        text (the guard echoes the raw path argument), so it is cleaned before it is stored.
        """
        rows = 0
        blocked = 0
        samples: list[str] = []
        try:
            with (req.run_dir / GUARD_AUDIT_NAME).open(encoding="utf-8", errors="replace") as log:
                for _ in range(GUARD_AUDIT_MAX_ROWS):
                    line = log.readline(GUARD_AUDIT_MAX_LINE)
                    if not line:
                        break
                    row = _audit_row(line)
                    if row is None:
                        continue
                    rows += 1
                    if row["allow"] is False:
                        blocked += 1
                        if len(samples) < 8:
                            samples.append(self._clean(f"{row['tool']}: {row.get('reason')}", 160))
        except OSError:
            return GuardAudit(None, handler.blocked_count, self._clean_all(handler.blocked))
        return GuardAudit(
            rows, max(blocked, handler.blocked_count), samples or self._clean_all(handler.blocked)
        )

    def _clean(self, text: str, limit: int) -> str:
        """``text`` without control characters or secrets, cut to ``limit`` characters.

        Control characters go first so that none can split a secret and dodge the redactor;
        the cut comes last so that it cannot leave half of a token behind.
        """
        return self.redactor.redact(strip_control_chars(text[:2000]))[:limit]

    def _clean_all(self, samples: Sequence[str]) -> list[str]:
        return [self._clean(sample, 200) for sample in samples]
