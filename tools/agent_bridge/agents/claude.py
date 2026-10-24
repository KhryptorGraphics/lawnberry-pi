"""Claude Code (``claude``) backend - the alternative agent.

Invocation (flags taken from ``claude --help`` of v2.1.293, accepted by the real binary)::

    claude -p --output-format json --no-session-persistence --safe-mode \\
        --strict-mcp-config --disable-slash-commands \\
        --permission-mode dontAsk --permission-prompts none --max-budget-usd <n> \\
        --tools Read Edit Write Glob Grep --allowedTools Read Edit Write Glob Grep
                                                   # prompt arrives on stdin

``--safe-mode`` turns off every user/project customisation (CLAUDE.md, hooks, plugins, MCP,
skills); ``dontAsk`` + ``--permission-prompts none`` denies anything not pre-approved, so there
is never a prompt nobody can answer.  The spend cap is Claude Code's own ``--max-budget-usd``.

``--output-format json`` prints one ``result`` object, or - when verbose mode is enabled in the
user's Claude config - a JSON array ``[system/init, assistant..., result]``.  Both are parsed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..process import ProcessOutcome
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

__all__ = ["ClaudeBackend", "load_messages"]

FINAL_TEXT_STORE_CHARS = 20_000
# The array form carries every message; keep enough of it that the closing ``result`` survives.
MIN_CAPTURE_BYTES = 64_000_000
EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


_RESULT_MARK = re.compile(r'"type"\s*:\s*"result"')
_SALVAGE_ATTEMPTS = 64


def _salvage_result(text: str) -> dict[str, Any] | None:
    """Find the ``result`` object in a truncated array, wherever ``type`` sits in its keys."""
    decoder = json.JSONDecoder()
    for mark in reversed(list(_RESULT_MARK.finditer(text))):
        start = text.rfind("{", 0, mark.start())
        for _attempt in range(_SALVAGE_ATTEMPTS):
            if start < 0:
                break
            try:
                item, _end = decoder.raw_decode(text, start)
            except ValueError:
                item = None
            if isinstance(item, dict) and item.get("type") == "result":
                return item
            start = text.rfind("{", 0, start)
    return None


def load_messages(stdout: str) -> list[dict[str, Any]]:
    """Parse ``--output-format json`` output into a list of message objects.

    Accepts a single object, a JSON array, line-delimited JSON (``stream-json``), and - as a
    last resort for a truncated array - a ``result`` object found by scanning from the end.
    """
    text = stdout.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        messages: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict):
                    messages.append(item)
        if messages:
            return messages
        salvaged = _salvage_result(text)
        return [salvaged] if salvaged is not None else []
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    return [data] if isinstance(data, dict) else []


class ClaudeBackend(AgentBackend):
    name = "claude"

    # ------------------------------------------------------------------ request shaping
    def tool_names(self, req: AgentRequest) -> list[str]:
        tools = ["Read", "Edit", "Write", "Glob", "Grep"]
        if req.bash_allowlist:
            tools.append("Bash")
        return tools

    def allowed_rules(self, req: AgentRequest) -> list[str]:
        rules = ["Read", "Edit", "Write", "Glob", "Grep"]
        rules += [f"Bash({command})" for command in req.bash_allowlist]
        return rules

    def capture_limit(self) -> int:
        return max(self.max_log_bytes, MIN_CAPTURE_BYTES)

    def build_argv(self, req: AgentRequest, binary: str) -> list[str]:
        argv = [
            binary,
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--safe-mode",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--max-budget-usd",
            f"{req.budget_usd:.2f}",
        ]
        model = req.model or self.cfg.model
        if model:
            argv += ["--model", model]
        effort = EFFORT.get(self.cfg.thinking or "")
        if effort:
            argv += ["--effort", effort]
        argv += list(self.cfg.extra_args)
        # Variadic options last: the prompt travels on stdin, so nothing can be swallowed.
        argv += ["--tools", *self.tool_names(req)]
        argv += ["--allowedTools", *self.allowed_rules(req)]
        return argv

    def make_stream_handler(self, req: AgentRequest) -> StreamHandler | None:
        return None

    # ------------------------------------------------------------------ parsing
    def parse(
        self, outcome: ProcessOutcome, req: AgentRequest, handler: StreamHandler | None
    ) -> AgentResult:
        messages = load_messages(outcome.stdout)
        init = next((m for m in messages if m.get("type") == "system"), {})
        result = next((m for m in reversed(messages) if m.get("type") == "result"), None)
        stderr_tail = self._finalize_text(outcome.stderr[-2000:], 600)

        tool_calls: int | None = None
        if any(m.get("type") == "assistant" for m in messages):
            tool_calls = 0
            for message in messages:
                if message.get("type") != "assistant":
                    continue
                content = (message.get("message") or {}).get("content") or []
                tool_calls += sum(
                    1
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "tool_use"
                )

        common: dict[str, Any] = {
            "session_id": (result or init).get("session_id"),
            "model": self._model_name(result, init),
            "tool_calls": tool_calls,
        }
        if result is None:
            cut = self._interrupted(outcome, req, **common)
            if cut is not None:
                return cut
            truncated = outcome.stdout_truncated
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=(
                    FailureKind.NONZERO_EXIT
                    if outcome.returncode not in (0, None)
                    else FailureKind.NO_RESULT
                ),
                error=stderr_tail
                or (
                    "claude output was truncated before its result message"
                    if truncated
                    else "claude produced no result message"
                ),
                **common,
            )

        text = self._finalize_text(str(result.get("result") or ""), 400_000)
        denials = result.get("permission_denials") or []
        usage = result.get("usage") or {}
        cost = result.get("total_cost_usd")
        common.update(
            {
                "final_text": text[-FINAL_TEXT_STORE_CHARS:],
                "report": extract_report(text),
                "cost_usd": float(cost) if isinstance(cost, int | float) else None,
                "usage": {
                    key: usage[key]
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                    if isinstance(usage.get(key), int | float)
                },
                "stop_reason": result.get("terminal_reason") or result.get("stop_reason"),
                "num_turns": result.get("num_turns"),
                "blocked_tool_calls": len(denials),
                "blocked_samples": [
                    f"{d.get('tool_name')}: denied by permission policy"
                    for d in denials[:8]
                    if isinstance(d, dict)
                ],
            }
        )
        cut = self._interrupted(outcome, req, **common)
        if cut is not None:
            return cut

        subtype = str(result.get("subtype") or "")
        terminal = str(result.get("terminal_reason") or "")
        if subtype == "error_max_budget_usd" or terminal == "budget_exhausted":
            return AgentResult(
                backend=self.name,
                status=AgentStatus.BUDGET_EXCEEDED,
                failure_kind=FailureKind.BUDGET,
                error=(
                    f"claude stopped at its ${req.budget_usd:.2f} budget "
                    f"(spent ${common['cost_usd'] or 0:.4f})"
                ),
                **common,
            )
        if subtype == "error_max_turns" or terminal == "max_turns":
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=FailureKind.MAX_TURNS,
                error="claude stopped after reaching its turn limit",
                **common,
            )
        if result.get("is_error") or subtype.startswith("error"):
            message = text or stderr_tail or subtype or "claude reported an error"
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=classify_error_text(message),
                error=self._finalize_text(message, 600),
                **common,
            )
        if outcome.returncode not in (0, None):
            return AgentResult(
                backend=self.name,
                status=AgentStatus.FAILED,
                failure_kind=FailureKind.NONZERO_EXIT,
                error=stderr_tail or f"claude exited with status {outcome.returncode}",
                **common,
            )
        return AgentResult(backend=self.name, status=AgentStatus.SUCCEEDED, **common)

    @staticmethod
    def _model_name(result: dict[str, Any] | None, init: dict[str, Any]) -> str | None:
        usage = (result or {}).get("modelUsage")
        if isinstance(usage, dict) and usage:
            return str(next(iter(usage)))
        model = init.get("model")
        return str(model) if model else None
