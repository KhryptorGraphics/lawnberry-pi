"""Guard evidence of the omp backend: proof that the guard ran (C-03), clean audit text (C-13).

The omp guard extension (``tools/agent_bridge/omp_guard.mjs``) is the only control that confines
omp's reads, URL fetches and writes outside the worktree.  If a future omp loaded the module but
never fired ``tool_call``, every call would be unguarded and the run would still end
``succeeded``.  The backend therefore compares the guard's audit log with the tool calls omp
reported and fails the run (``guard_inactive``) when the log shows fewer decisions than calls.

The audit log is also agent-controlled text (the guard echoes up to 80 characters of a path
argument into the reason), so everything the backend copies out of it passes the redactor and
loses its control characters before it reaches ``blocked_samples`` and the ledger.

The fake omp (``bridgefakes/fake_omp.py``) emulates the guard: one audit row per ``tool_call``
step, none for a scenario with ``"guard": "off"`` (a guard that never fires).  The tests drive
the real backend class, the real subprocess runner and, for the worker flow, the real worker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.agent_fixtures import docs_edit_steps, install_fake_cli
from bridgefakes.rig import Rig, make_rig

from tools.agent_bridge.agents import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    OmpBackend,
    build_backend,
)
from tools.agent_bridge.agents.omp import GUARD_PATH, GUARD_PREFIX, OmpStreamHandler
from tools.agent_bridge.config import AgentConfig, BackendConfig
from tools.agent_bridge.models import RunStatus
from tools.agent_bridge.process import ProcessOutcome

NODE = shutil.which("node")
AUDIT_NAME = "omp-guard-audit.jsonl"
SESSION = "00000000-0000-4000-8000-000000000000"
# Assembled from fragments so that this file never contains a secret-shaped string.
TOKEN = "ghp_" + "Zy9x8W7v6U5t4S3r2Q1p0O9n8M7l6K5j4I3h"
FINAL = {"op": "final", "text": "done"}
HINT = "guard extension did not record every tool call"


# ----------------------------------------------------------------------------------------------
# scenario helpers
# ----------------------------------------------------------------------------------------------
def call(tool: str, **args: Any) -> dict[str, Any]:
    """A tool call the guard allows."""
    return {"op": "tool_call", "tool": tool, "args": args}


def guard_block(tool: str, reason: str, **args: Any) -> dict[str, Any]:
    """A tool call the guard refuses (omp reports the guard's reason as the tool error)."""
    return {
        "op": "tool_call",
        "tool": tool,
        "args": args,
        "blocked": True,
        "result": f"{GUARD_PREFIX} {reason}",
    }


def guard_setting(steps: Sequence[dict[str, Any]], guard: str | None) -> dict[str, Any]:
    """The scenario document: ``guard`` is the top-level option only the fake omp reads."""
    document: dict[str, Any] = {"steps": list(steps), "log_env": [], "ignore_sigterm": False}
    if guard is not None:
        document["guard"] = guard
    return document


def set_guard(scenario: Path, guard: str) -> None:
    """Switch the guard of an already written scenario file on or off."""
    document = json.loads(scenario.read_text(encoding="utf-8"))
    document["guard"] = guard
    scenario.write_text(json.dumps(document), encoding="utf-8")


class Harness:
    """A real backend bound to one of the fake CLIs, with one scenario and one run directory."""

    def __init__(self, root: Path, kind: str = "omp") -> None:
        self.root = root
        self.kind = kind
        root.mkdir(parents=True, exist_ok=True)
        (root / "home").mkdir(exist_ok=True)
        self.scenario = root / "scenario.json"
        self.cwd = root / "worktree"
        self.cwd.mkdir(exist_ok=True)
        self.run_dir = root / "run"
        binary = install_fake_cli(root / "bin", "omp" if kind == "omp" else "claude")
        environ = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root / "home"),
            "FAKE_AGENT_SCENARIO": str(self.scenario),
        }
        config = AgentConfig.model_validate(
            {kind: {"binary": str(binary), "pass_env": ["FAKE_AGENT_SCENARIO"]}}
        )
        self.backend = build_backend(kind, config, environ=environ)

    def run(
        self, steps: Sequence[dict[str, Any]], *, guard: str | None = None, **overrides: Any
    ) -> AgentResult:
        self.scenario.write_text(json.dumps(guard_setting(steps, guard)), encoding="utf-8")
        fields: dict[str, Any] = {
            "prompt": "Document the bridge.\n",
            "cwd": self.cwd,
            "run_dir": self.run_dir,
            "scope": "docs-only",
            "timeout_s": 30.0,
            "budget_usd": 1.0,
            "run_id": "run-1",
            "abort_poll_s": 0.05,
            "write_allow_sets": (("docs/**", "*.md"),),
            "write_deny": (".github/**",),
        }
        fields.update(overrides)
        return self.backend.run(AgentRequest(**fields))

    @property
    def audit_log(self) -> Path:
        return self.run_dir / AUDIT_NAME

    def audit_rows(self) -> list[dict[str, Any]]:
        if not self.audit_log.exists():
            return []
        lines = self.audit_log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path / "omp")


# ----------------------------------------------------------------------------------------------
# direct parsing of a recorded stream next to a hand-made audit log
# ----------------------------------------------------------------------------------------------
def stream(*, calls: int, blocked_text: str | None = None) -> str:
    """A clean omp run with ``calls`` tool calls; the first one ends in ``blocked_text``."""
    events: list[dict[str, Any]] = [
        {"type": "session", "version": 3, "id": SESSION, "timestamp": "t", "cwd": "/w"}
    ]
    for index in range(calls):
        text = blocked_text if (index == 0 and blocked_text is not None) else "ok"
        events += [
            {
                "type": "tool_execution_start",
                "toolCallId": f"toolu_{index}",
                "toolName": "read",
                "args": {},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": f"toolu_{index}",
                "toolName": "read",
                "result": {"content": [{"type": "text", "text": text}], "details": {}},
                "isError": blocked_text is not None and index == 0,
            },
        ]
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
        "usage": {"input": 1, "output": 1, "cost": {"total": 0.01}},
        "stopReason": "stop",
        "timestamp": 1792837028520,
        "responseId": "msg_1",
    }
    events += [
        {"type": "message_end", "message": message},
        {"type": "agent_end", "messages": [], "isTerminal": True, "yielded": True},
    ]
    return "".join(json.dumps(event) + "\n" for event in events)


def audit_row(tool: str = "read", *, allow: bool = True, reason: str | None = None) -> str:
    row = {"ts": "t", "tool": tool, "allow": allow, "reason": reason, "paths": []}
    return json.dumps(row)


def parse_stream(text: str, run_dir: Path, audit_lines: Sequence[str] | None) -> AgentResult:
    """``OmpBackend.parse`` over ``text`` with the audit log ``audit_lines`` (None: no log)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if audit_lines is not None:
        (run_dir / AUDIT_NAME).write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    backend = OmpBackend(BackendConfig(binary="omp"), environ={})
    request = AgentRequest(
        prompt="p",
        cwd=Path("/nonexistent/worktree"),
        run_dir=run_dir,
        scope="docs-only",
        timeout_s=60,
        budget_usd=1.0,
    )
    handler = OmpStreamHandler(1.0)
    for line in text.splitlines(keepends=True):
        handler.feed(line)
    outcome = ProcessOutcome(returncode=0, stdout=text, stderr="", duration_s=1.0)
    return backend.parse(outcome, request, handler)


def failure_kind(result: AgentResult) -> str | None:
    return result.failure_kind.value if result.failure_kind else None


# ----------------------------------------------------------------------------------------------
# the fake omp emulates the guard (the tests below rely on it)
# ----------------------------------------------------------------------------------------------
def test_the_fake_omp_audits_every_tool_call_like_the_real_guard(harness: Harness) -> None:
    steps = [
        call("read", path="docs/guide.md"),
        guard_block(
            "write", "path escapes the task worktree: /etc/passwd", path="/etc/passwd", content="x"
        ),
        # omp's own error: the guard allowed this call, the tool failed afterwards
        {"op": "tool_call", "tool": "write", "args": {}, "blocked": True, "result": "ENOENT"},
        call("think"),
        call("yield"),
        FINAL,
    ]

    result = harness.run(steps)

    assert result.ok and result.tool_calls == 5
    rows = harness.audit_rows()
    assert [(r["tool"], r["allow"]) for r in rows] == [
        ("read", True),
        ("write", False),
        ("write", True),
        ("think", True),
        ("yield", True),
    ]
    assert all(set(row) == {"ts", "tool", "allow", "reason", "paths"} for row in rows)
    assert rows[0]["reason"] is None and rows[0]["paths"] == ["docs/guide.md"]
    assert rows[1]["reason"] == "path escapes the task worktree: /etc/passwd"


def test_the_fake_omp_writes_no_rows_when_the_guard_is_off(harness: Harness) -> None:
    result = harness.run([call("read", path="a.md"), call("read", path="b.md"), FINAL], guard="off")

    assert result.tool_calls == 2
    assert not harness.audit_log.exists()


def test_a_single_call_can_escape_the_guard(harness: Harness) -> None:
    steps = [call("read", path="a.md"), {**call("read", path="b.md"), "guard": "off"}, FINAL]

    harness.run(steps)

    assert [row["paths"] for row in harness.audit_rows()] == [["a.md"]]


@pytest.mark.parametrize(
    ("steps", "guard", "complaint"),
    [
        ([FINAL], "maybe", 'scenario "guard"'),
        ([{**call("read", path="a.md"), "guard": "maybe"}, FINAL], None, 'tool_call "guard"'),
    ],
)
def test_the_fake_omp_rejects_a_guard_option_it_does_not_know(
    harness: Harness, steps: list[dict[str, Any]], guard: str | None, complaint: str
) -> None:
    result = harness.run(steps, guard=guard)

    assert result.status is AgentStatus.FAILED and failure_kind(result) == "nonzero_exit"
    assert complaint in (result.error or "")


# ----------------------------------------------------------------------------------------------
# C-03: a run whose guard cannot be shown to have run is not a success
# ----------------------------------------------------------------------------------------------
def test_a_guard_that_never_fires_fails_the_run_as_guard_inactive(harness: Harness) -> None:
    steps = [
        call("read", path="docs/guide.md"),
        call("write", path="docs/a.md", content="x"),
        FINAL,
    ]

    result = harness.run(steps, guard="off")

    assert result.status is AgentStatus.FAILED and not result.ok
    assert failure_kind(result) == "guard_inactive"
    assert "2 tool call(s)" in (result.error or "") and "0 row(s)" in (result.error or "")
    assert result.tool_calls == 2  # the partial result is kept for the operator
    assert not harness.audit_log.exists()


def test_fewer_audit_rows_than_tool_calls_fail_the_run(harness: Harness) -> None:
    steps = [
        call("read", path="docs/a.md"),
        {**call("write", path="docs/a.md", content="x"), "guard": "off"},
        call("read", path="docs/b.md"),
        FINAL,
    ]

    result = harness.run(steps)

    assert result.status is AgentStatus.FAILED
    assert failure_kind(result) == "guard_inactive"
    assert "3 tool call(s)" in (result.error or "") and "2 row(s)" in (result.error or "")
    assert len(harness.audit_rows()) == 2


def test_a_guard_that_audited_every_call_lets_the_run_succeed(harness: Harness) -> None:
    steps = [
        call("read", path="docs/a.md"),
        guard_block("write", "path is protected: .github/ci.yml", path=".github/ci.yml"),
        call("think"),
        call("yield"),
        FINAL,
    ]

    result = harness.run(steps)

    assert result.ok and result.failure_kind is None
    assert result.tool_calls == 4 and result.blocked_tool_calls == 1
    assert len(harness.audit_rows()) == 4


def test_a_run_without_tool_calls_cannot_have_misbehaved(harness: Harness) -> None:
    for guard in ("off", "on"):
        result = harness.run([FINAL], guard=guard, run_dir=harness.root / f"run-{guard}")

        assert result.ok and result.tool_calls == 0
        assert not (harness.root / f"run-{guard}" / AUDIT_NAME).exists()


def test_an_unreadable_audit_log_fails_the_run(harness: Harness) -> None:
    harness.audit_log.mkdir(parents=True)  # a directory where the log belongs: not readable

    result = harness.run([call("read", path="docs/a.md"), FINAL])

    assert result.status is AgentStatus.FAILED
    assert failure_kind(result) == "guard_inactive"
    assert "missing or unreadable" in (result.error or "")


def test_rows_of_an_earlier_attempt_do_not_vouch_for_a_retry(harness: Harness) -> None:
    steps = [call("read", path="docs/a.md"), call("read", path="docs/b.md"), FINAL]
    first = harness.run(steps)
    assert first.ok and len(harness.audit_rows()) == 2

    retry = harness.run(steps, guard="off")  # the same run directory, a guard that is dead now

    assert retry.status is AgentStatus.FAILED
    assert failure_kind(retry) == "guard_inactive"
    assert harness.audit_rows() == []


def test_claude_runs_need_no_guard_audit(tmp_path: Path) -> None:
    claude = Harness(tmp_path / "claude", kind="claude")
    steps = [
        {"op": "tool_call", "tool": "Read", "args": {"path": "docs/a.md"}},
        {"op": "tool_call", "tool": "Write", "args": {"path": "docs/b.md"}},
        FINAL,
    ]

    result = claude.run(steps, guard="off")  # the Claude emulator has no guard to switch off

    assert result.ok and result.failure_kind is None
    assert not claude.audit_log.exists()


# -- the guard stops auditing after 5000 decisions ------------------------------------------------
def test_five_thousand_rows_vouch_for_any_number_of_calls(tmp_path: Path) -> None:
    rows = [audit_row()] * 5000

    result = parse_stream(stream(calls=5100), tmp_path / "enough", rows)

    assert result.ok, result.error


def test_one_row_short_of_the_cap_still_fails(tmp_path: Path) -> None:
    rows = [audit_row()] * 4999

    result = parse_stream(stream(calls=5100), tmp_path / "short", rows)

    assert result.status is AgentStatus.FAILED and failure_kind(result) == "guard_inactive"
    assert "5100 tool call(s)" in (result.error or "") and "4999 row(s)" in (result.error or "")


def test_more_rows_than_calls_are_fine(tmp_path: Path) -> None:
    result = parse_stream(stream(calls=2), tmp_path / "extra", [audit_row()] * 3)

    assert result.ok, result.error


def test_only_well_formed_rows_count(tmp_path: Path) -> None:
    lines = [
        audit_row(),
        "this line is not json",
        json.dumps({"ts": "t", "allow": True}),  # no tool
        json.dumps({"ts": "t", "tool": "read", "allow": "yes"}),  # allow is not a boolean
        json.dumps(["read", True]),
    ]

    result = parse_stream(stream(calls=3), tmp_path / "garbage", lines)

    assert result.status is AgentStatus.FAILED and failure_kind(result) == "guard_inactive"
    assert "1 row(s)" in (result.error or "")


def test_a_missing_log_with_calls_fails_and_without_calls_passes(tmp_path: Path) -> None:
    missing = parse_stream(stream(calls=1), tmp_path / "missing", None)
    idle = parse_stream(stream(calls=0), tmp_path / "idle", None)

    assert failure_kind(missing) == "guard_inactive" and "missing or unreadable" in missing.error
    assert idle.ok


# -- through the worker ---------------------------------------------------------------------------
@pytest.fixture
def omp_rig(tmp_path: Path) -> Iterator[Rig]:
    rig = make_rig(tmp_path, "omp")
    try:
        yield rig
    finally:
        rig.close()


def published(rig: Rig) -> tuple[list[str], str, dict[int, Any]]:
    """What the world looks like after a run: origin branches, object counts, pull requests."""
    origin = rig.git_env
    return origin.origin_branches(), origin.origin_git("count-objects", "-v"), dict(rig.fake.pulls)


def test_a_guard_that_never_fired_blocks_the_run_and_publishes_nothing(omp_rig: Rig) -> None:
    rig = omp_rig
    number = rig.new_task_issue()
    rig.docs_scenario()  # a perfectly good documentation edit with one reported tool call
    set_guard(rig.scenario, "off")
    before = published(rig)

    run = rig.run_issue(number)

    assert run.status is RunStatus.BLOCKED, (run.failure_kind, run.failure_detail)
    assert run.failure_kind == "agent_guard_inactive"
    assert "1 tool call(s)" in (run.failure_detail or "")
    assert published(rig) == before
    assert rig.fake.pulls == {}
    assert run.pr_number is None and run.commit_sha is None
    assert HINT in rig.comments(number)[-1]
    assert run.result is not None and run.result["agent"]["failure_kind"] == "guard_inactive"


def test_a_working_guard_does_not_get_in_the_way_of_a_normal_run(omp_rig: Rig) -> None:
    rig = omp_rig
    rig.set_scenario([call("read", path="docs/guide.md"), *docs_edit_steps()])
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.result is not None and run.result["agent"]["tool_calls"] == 2


# ----------------------------------------------------------------------------------------------
# C-13: agent-controlled text in the audit log is cleaned before it is stored
# ----------------------------------------------------------------------------------------------
def test_a_token_in_a_blocked_path_is_redacted_in_the_samples(harness: Harness) -> None:
    block = guard_block("read", f"path escapes the task worktree: /{TOKEN}", path=f"/{TOKEN}")

    result = harness.run([block, FINAL])

    assert result.ok and result.blocked_tool_calls == 1
    (sample,) = result.blocked_samples
    assert sample.startswith("read: path escapes the task worktree: /[REDACTED")
    assert TOKEN not in json.dumps(result.to_dict())


def test_a_token_in_the_tool_name_is_redacted_too(tmp_path: Path) -> None:
    rows = [audit_row(f"{TOKEN}", allow=False, reason="tool is not enabled")]

    result = parse_stream(stream(calls=1), tmp_path / "tool", rows)

    (sample,) = result.blocked_samples
    assert TOKEN not in sample and "[REDACTED" in sample


def test_a_token_that_straddles_the_sample_limit_is_not_cut_in_half(tmp_path: Path) -> None:
    reason = "x" * 140 + TOKEN  # redacting after the cut would leave a recognisable prefix
    rows = [audit_row("read", allow=False, reason=reason)]

    result = parse_stream(stream(calls=1), tmp_path / "straddle", rows)

    (sample,) = result.blocked_samples
    assert "ghp_" not in sample and len(sample) <= 160


def test_control_characters_never_reach_the_samples(harness: Harness) -> None:
    reason = "path escapes the task worktree: \x1b[2J\x07/etc/passwd\x00"

    result = harness.run([guard_block("read", reason, path="x"), FINAL])

    (sample,) = result.blocked_samples
    assert sample == "read: path escapes the task worktree: [2J/etc/passwd"


def test_samples_taken_from_the_stream_are_cleaned_as_well(tmp_path: Path) -> None:
    # the audit log saw only allowed calls, so the samples come from the stream
    text = f"{GUARD_PREFIX} path escapes: /{TOKEN}\x1b[31m"

    result = parse_stream(stream(calls=1, blocked_text=text), tmp_path / "stream", [audit_row()])

    assert result.blocked_tool_calls == 1
    (sample,) = result.blocked_samples
    assert TOKEN not in sample and "\x1b" not in sample and "[REDACTED" in sample


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_row_written_by_the_real_guard_is_redacted(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    audit = tmp_path / "run" / AUDIT_NAME
    audit.parent.mkdir()
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "root": str(root),
                "tools": ["read", "write"],
                "write_allow_sets": [["docs/**"]],
                "write_deny": [],
                "read_deny": [],
                "bash_exact": [],
                "audit_log": str(audit),
                "max_write_bytes": 1000,
            }
        ),
        encoding="utf-8",
    )
    driver = tmp_path / "driver.mjs"
    driver.write_text(
        'import { pathToFileURL } from "node:url";\n'
        "const [guardFile, policyFile, target] = process.argv.slice(2);\n"
        "process.env.LAWNBERRY_BRIDGE_GUARD_POLICY = policyFile;\n"
        "const guard = await import(pathToFileURL(guardFile).href);\n"
        "const handlers = {};\n"
        "guard.default({ on(event, fn) { handlers[event] = fn; } });\n"
        'await handlers.tool_call({ toolName: "read", input: { path: target } }, {});\n',
        encoding="utf-8",
    )
    subprocess.run(
        [str(NODE), str(driver), str(GUARD_PATH), str(policy), f"/{TOKEN}"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    written = audit.read_text(encoding="utf-8")
    assert TOKEN in written, "the real guard echoes the path argument into its audit row"

    result = parse_stream(stream(calls=1), tmp_path / "run", None)

    assert result.blocked_tool_calls == 1
    (sample,) = result.blocked_samples
    assert sample.startswith("read: path escapes the task worktree: /[REDACTED")
    assert TOKEN not in json.dumps(result.to_dict())


def test_the_ledger_never_holds_the_token_a_blocked_path_carried(omp_rig: Rig) -> None:
    rig = omp_rig
    block = guard_block("read", f"path escapes the task worktree: /{TOKEN}", path=f"/{TOKEN}")
    rig.set_scenario([block, *docs_edit_steps()])
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.result is not None
    samples = run.result["agent"]["blocked_samples"]
    assert len(samples) == 1 and "[REDACTED" in samples[0]
    assert TOKEN not in json.dumps(run.result)
    ledger_row = rig.ledger.get(run.run_id)
    assert ledger_row is not None and TOKEN not in json.dumps(ledger_row.result)
