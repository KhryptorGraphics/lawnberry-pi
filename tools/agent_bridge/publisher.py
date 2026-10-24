"""Everything the bridge writes back to GitHub: labels, bot comments and pull request text.

Comments carry a hidden marker so a retried publication never posts twice.  Text that originates
from an agent (including error text that may embed it) is redacted and rendered inside a code
fence or an inline code span it cannot close, so Markdown, HTML, images and ``@mentions`` in it
stay inert.  So is text chosen by whoever wrote the issue (the task title, the YAML keys that
validation messages quote): posted as live Markdown it would come from the bridge's account.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .config import BridgeConfig
from .errors import BridgeError
from .github import GitHubClient
from .ledger import RunRecord
from .models import Comment, RunPhase, RunStatus
from .redact import Redactor, fenced, strip_control_chars, truncate

__all__ = [
    "Publisher",
    "ReadOnlyGitHub",
    "marker",
    "render_blocked",
    "render_cancelled",
    "render_done",
    "render_pull_request_body",
    "render_refusal",
    "render_started",
]

SUMMARY_CHARS = 3000
FOOTER = (
    "_Automated by the LawnBerry agent bridge. Nothing is merged or deployed automatically; "
    "a human reviews every pull request._"
)


def marker(key: str) -> str:
    return f"<!-- agent-bridge:{key} -->"


def _one_line(text: str, limit: int = 160) -> str:
    return truncate(" ".join(strip_control_chars(text).split()), limit, "...")


def _code(text: str) -> str:
    return "`" + text.replace("`", "'") + "`"


def _verbatim(text: str) -> str:
    """Inline code span that ``text`` cannot close: its delimiter outgrows every backtick run."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def _untrusted_line(text: str, limit: int, redactor: Redactor) -> str:
    """Agent-derived text as one redacted, truncated line inside an inline code span."""
    line = _one_line(redactor.redact(strip_control_chars(text)), limit)
    return _verbatim(line) if line else "_none_"


def _issue_line(text: str, limit: int, *, table_cell: bool = False) -> str:
    """Issue-derived text as one line inside an inline code span it cannot close.

    Task titles and validation messages (which quote the author's YAML keys) are chosen by the
    issue's author, so ``@mentions``, links, HTML and ``#refs`` in them must stay inert.  In a
    table cell every pipe is backslash-escaped first: GFM ends a cell at any unescaped pipe,
    code spans included.
    """
    line = _one_line(text, limit)
    if table_cell:
        line = line.replace("|", "\\|")
    return _verbatim(line) if line else "_none_"


def _publication_note(run: RunRecord) -> str:
    """What reached the remote, as far as the run record proves (it never claims more).

    ``Ledger.finish`` overwrites the phase, so a finished run that had already committed cannot
    show whether the push happened; the note then says so instead of claiming nothing was pushed.
    """
    branch = _code(run.branch or "")
    if run.pr_number or run.phase is RunPhase.NOTIFY:
        number = f" #{run.pr_number}" if run.pr_number else ""
        return f"The branch {branch} was pushed and draft pull request{number} was opened."
    if run.phase is RunPhase.PULL_REQUEST:
        return f"The branch {branch} was pushed; no pull request was opened."
    if run.commit_sha:
        return (
            f"The change was committed on branch {branch}, which may already have been pushed; "
            "no pull request was opened."
        )
    return "No branch was pushed and no pull request was opened."


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _files_line(files: Sequence[str], limit: int = 25) -> str:
    shown = ", ".join(_code(f) for f in files[:limit])
    more = f" and {len(files) - limit} more" if len(files) > limit else ""
    return shown + more if files else "none"


def _agent(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload.get("agent") or {}


def _agent_block(payload: Mapping[str, Any], redactor: Redactor) -> str:
    agent = _agent(payload)
    report = agent.get("report") or {}
    text = report.get("summary") or agent.get("final_text") or ""
    if not text:
        return "_The agent returned no summary._"
    clean = truncate(redactor.redact(strip_control_chars(str(text))).strip(), SUMMARY_CHARS)
    return "Agent summary (untrusted text, shown as code):\n\n" + fenced(clean)


def _checks_lines(payload: Mapping[str, Any]) -> list[str]:
    checks = payload.get("checks") or []
    if not checks:
        return ["- Checks: none requested"]
    return [
        f"- Check {_code(str(c.get('name')))}: {'passed' if c.get('passed') else 'FAILED'} "
        f"in {c.get('duration_s', 0)}s"
        for c in checks
    ]


def _run_line(run: RunRecord, payload: Mapping[str, Any]) -> str:
    agent = _agent(payload)
    model = f" ({_code(str(agent['model']))})" if agent.get("model") else ""
    budget = f" of {_money(agent['budget_usd'])} budget" if agent.get("budget_usd") else ""
    return (
        f"- Run {_code(run.run_id[:12])} - backend {_code(run.backend)}{model} - "
        f"{agent.get('duration_s', 0):.0f}s - cost {_money(agent.get('cost_usd'))}{budget}"
    )


def render_started(run: RunRecord, *, budget_usd: float, timeout_minutes: float, scope: str) -> str:
    return "\n".join(
        [
            "### Agent run started",
            "",
            f"- Run {_code(run.run_id[:12])} - backend {_code(run.backend)} - scope {_code(scope)}",
            f"- Approved by {_code(run.approved_by)} (label event {run.approval_event_id})",
            f"- Limits: ${budget_usd:.2f} budget, {timeout_minutes:g} minute timeout, "
            "one run at a time",
            "- The agent works in an isolated git worktree; it cannot reach the mower, "
            "deploy, or merge.",
            "",
            FOOTER,
            marker(f"run-{run.run_id}-started"),
        ]
    )


def render_done(run: RunRecord, payload: Mapping[str, Any], redactor: Redactor) -> str:
    files = payload.get("files") or []
    agent = _agent(payload)
    lines = [
        "### Draft pull request ready for review",
        "",
        f"- Pull request: #{run.pr_number} (draft) {run.pr_url or ''}".rstrip(),
        f"- Branch {_code(run.branch or '')} at {_code((run.commit_sha or '')[:10])}",
        _run_line(run, payload),
        f"- Files changed ({len(files)}): {_files_line(files)}",
        *_checks_lines(payload),
        f"- Tool calls: {agent.get('tool_calls')}, "
        f"blocked by policy: {agent.get('blocked_tool_calls', 0)}",
        "",
        _agent_block(payload, redactor),
        "",
        "Nothing was merged or deployed. To rerun, apply a fresh approval after removing "
        "`agent-blocked`/`agent-done` as appropriate.",
        "",
        FOOTER,
        marker(f"run-{run.run_id}-done"),
    ]
    return "\n".join(lines)


def render_blocked(
    run: RunRecord,
    *,
    kind: str,
    detail: str,
    hints: Sequence[str],
    payload: Mapping[str, Any] | None,
    redactor: Redactor,
) -> str:
    payload = payload or {}
    agent = _agent(payload)
    lines = [
        "### Agent run blocked",
        "",
        f"- Reason: {_code(kind)} - {_untrusted_line(detail, 400, redactor)}",
        _run_line(run, payload),
    ]
    if agent.get("blocked_samples"):
        samples = "; ".join(_untrusted_line(s, 100, redactor) for s in agent["blocked_samples"][:4])
        lines.append(f"- Policy-blocked tool calls: {samples}")
    if payload.get("checks"):
        lines += _checks_lines(payload)
    violations = payload.get("violations") or []
    if violations:
        lines.append("- Change policy violations:")
        lines += [f"  - {_untrusted_line(v, 200, redactor)}" for v in violations[:15]]
    lines += ["", *(f"- {hint}" for hint in hints)]
    if agent:
        lines += ["", _agent_block(payload, redactor)]
    lines += [
        "",
        f"{_publication_note(run)} To retry after fixing the cause: remove the `agent-blocked` "
        "label, then remove and re-apply `agent-approved`.",
        "",
        FOOTER,
        marker(f"run-{run.run_id}-blocked"),
    ]
    return "\n".join(lines)


def render_cancelled(run: RunRecord, reason: str, redactor: Redactor | None = None) -> str:
    return "\n".join(
        [
            "### Agent run cancelled",
            "",
            f"- Reason: {_untrusted_line(reason, 300, redactor or Redactor())}",
            f"- Run {_code(run.run_id[:12])} - backend {_code(run.backend)}",
            f"- {_publication_note(run)}",
            "",
            FOOTER,
            marker(f"run-{run.run_id}-cancelled"),
        ]
    )


def render_refusal(code: str, messages: Sequence[str], *, key: str) -> str:
    bullets = [f"- {_issue_line(m, 300)}" for m in messages[:20]]
    return "\n".join(
        [
            "### Agent bridge cannot run this task",
            "",
            f"- Reason: {_code(code)}",
            *bullets,
            "",
            "Fix the issue, then remove and re-apply `agent-approved` (an edit after approval "
            "always needs a fresh approval).",
            "",
            FOOTER,
            marker(key),
        ]
    )


def render_pull_request_body(
    run: RunRecord,
    payload: Mapping[str, Any],
    *,
    issue_number: int,
    keyword: str,
    redactor: Redactor,
) -> str:
    files = payload.get("files") or []
    agent = _agent(payload)
    report = agent.get("report") or {}
    model = f" ({_code(str(agent['model']))})" if agent.get("model") else ""
    base = _code(run.base_sha[:10]) if run.base_sha else "n/a"
    lines = [
        f"Draft pull request opened by the LawnBerry agent bridge for #{issue_number}. "
        "It has not been merged or deployed, and a human must review it before it can be.",
        "",
    ]
    if report.get("status") == "needs_human":
        lines += ["**The agent asked for human attention; read its summary below.**", ""]
    lines += [
        "| | |",
        "|---|---|",
        f"| Task | #{issue_number} {_issue_line(run.issue_title, 120, table_cell=True)} |",
        f"| Agent | {_code(run.backend)}{model} |",
        f"| Run | {_code(run.run_id)} |",
        f"| Base | {base} |",
        f"| Approved by | {_code(run.approved_by)} (label event {run.approval_event_id}) |",
        f"| Cost | {_money(agent.get('cost_usd'))} of {_money(agent.get('budget_usd'))} budget |",
        f"| Changes | {len(files)} file(s), +{payload.get('added', 0)} / "
        f"-{payload.get('deleted', 0)} lines |",
        "",
        f"Files: {_files_line(files, 40)}",
        "",
        *_checks_lines(payload),
        f"- Policy: no protected paths touched; {agent.get('blocked_tool_calls') or 0} tool "
        "call(s) were blocked by the agent guard.",
        "",
        _agent_block(payload, redactor),
        "",
        f"{keyword} #{issue_number}",
    ]
    return "\n".join(lines)


class ReadOnlyGitHub:
    """Wraps a client and refuses every write; used for dry runs."""

    _WRITES = frozenset(
        {"add_labels", "remove_label", "ensure_labels", "create_comment", "create_pull_request"}
    )

    def __init__(self, inner: GitHubClient) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in self._WRITES:

            def refuse(*_args: Any, **_kwargs: Any) -> Any:
                raise BridgeError(f"refusing GitHub write {name!r} in dry-run mode")

            return refuse
        return getattr(self._inner, name)


class Publisher:
    """Idempotent label transitions and bot comments."""

    def __init__(self, github: GitHubClient, cfg: BridgeConfig, redactor: Redactor) -> None:
        self.github = github
        self.cfg = cfg
        self.redactor = redactor

    # ---------------------------------------------------------------- labels
    def mark_running(self, issue_number: int) -> None:
        self.github.add_labels(issue_number, [self.cfg.labels.running])

    def mark_terminal(self, issue_number: int, status: RunStatus) -> None:
        labels = self.cfg.labels
        self.github.remove_label(issue_number, labels.running)
        if status is RunStatus.DONE:
            self.github.add_labels(issue_number, [labels.done])
        elif status is RunStatus.BLOCKED:
            self.github.add_labels(issue_number, [labels.blocked])
        # cancelled: leave only the owner's own labels

    def clear_running(self, issue_number: int) -> None:
        self.github.remove_label(issue_number, self.cfg.labels.running)

    # ---------------------------------------------------------------- comments
    def comment_once(self, issue_number: int, key: str, body: str) -> bool:
        """Post ``body`` unless the bridge's own account already posted a comment with its marker.

        A marker is only trusted on comments written by the account the token belongs to
        (``GET /user``): the markers are public and guessable from the run id in the "started"
        comment, so a comment from anyone else must never suppress the bridge's report.
        """
        tag = marker(key)
        text = body if tag in body else f"{body}\n{tag}"
        me = self.github.viewer_login()
        existing: list[Comment] = self.github.list_comments(issue_number)
        if any(tag in c.body and c.author.lower() == me for c in existing):
            return False
        self.github.create_comment(issue_number, self.redactor.redact(text))
        return True
