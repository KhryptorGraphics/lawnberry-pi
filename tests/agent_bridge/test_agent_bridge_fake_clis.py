"""Tests for the fake ``omp`` and ``claude`` CLIs used by the agent bridge test-suite.

Both emulators run as real subprocesses through ``install_fake_cli`` wrappers and are checked
against the captured real outputs in ``fixtures/`` and against the real ``--help`` flag lists.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import pty
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.agent_fixtures import (
    FAKES_DIR,
    docs_edit_steps,
    fake_agent_env,
    install_fake_cli,
    pid_is_alive,
    read_invocations,
    wait_for_file,
    write_scenario,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KINDS = ("omp", "claude")
BAD_SCENARIO_TEXT = "cannot load scenario"

# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


class Fakes:
    """A scratch area with both fake CLIs installed plus a scenario file and invocation log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.work = root / "work"
        self.work.mkdir()
        self.scenario = root / "scenario.json"
        self.log = root / "invocations.jsonl"
        self.binaries = {kind: install_fake_cli(root / "bin", kind) for kind in KINDS}

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "LC_ALL": "C.UTF-8",
        }
        env.update(fake_agent_env(self.scenario if self.scenario.exists() else None, self.log))
        env.update(extra or {})
        return env

    def argv(self, kind: str, args: list[str]) -> list[str]:
        return [str(self.binaries[kind]), *args]

    def set_steps(self, steps: list[dict[str, Any]], **options: Any) -> Path:
        return write_scenario(self.scenario, steps, **options)

    def run(
        self,
        kind: str,
        args: list[str],
        *,
        steps: list[dict[str, Any]] | None = None,
        stdin: str | None = "the prompt\n",
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 30,
        **scenario_options: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Run a fake to completion; ``stdin=None`` connects stdin to /dev/null."""
        if steps is not None:
            self.set_steps(steps, **scenario_options)
        feed: dict[str, Any] = {"stdin": subprocess.DEVNULL} if stdin is None else {"input": stdin}
        return subprocess.run(
            self.argv(kind, args),
            capture_output=True,
            text=True,
            env=self.env(env),
            cwd=cwd or self.work,
            timeout=timeout,
            check=False,
            **feed,
        )

    def last_invocation(self) -> dict[str, Any]:
        return read_invocations(self.log)[-1]


@pytest.fixture
def fakes(tmp_path: Path) -> Fakes:
    return Fakes(tmp_path)


def _kill_group(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


@contextlib.contextmanager
def running(
    fakes: Fakes,
    kind: str,
    args: list[str],
    *,
    steps: list[dict[str, Any]],
    stdin: str = "go\n",
    watchdog: float = 60.0,
    **scenario_options: Any,
) -> Iterator[subprocess.Popen[str]]:
    """Start a fake in its own process group; the group is always killed on exit."""
    fakes.set_steps(steps, **scenario_options)
    proc = subprocess.Popen(
        fakes.argv(kind, args),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=fakes.env(),
        cwd=fakes.work,
        start_new_session=True,
    )
    timer = threading.Timer(watchdog, _kill_group, args=(proc,))
    timer.daemon = True
    timer.start()
    try:
        assert proc.stdin is not None
        proc.stdin.write(stdin)
        proc.stdin.close()
        yield proc
    finally:
        timer.cancel()
        _kill_group(proc)
        proc.wait(timeout=10)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def parse_ndjson(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_ndjson(name: str) -> list[dict[str, Any]]:
    return parse_ndjson((FIXTURES / name).read_text(encoding="utf-8"))


def signature(events: list[dict[str, Any]]) -> list[str]:
    """Event type sequence; message_update events carry their assistantMessageEvent type."""
    return [
        event["type"]
        + (
            ":" + event["assistantMessageEvent"]["type"]
            if event["type"] == "message_update"
            else ""
        )
        for event in events
    ]


def key_paths(value: Any, prefix: str = "") -> set[str]:
    """Flattened key paths of a JSON value (lists are summarised by their first element)."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= key_paths(child, path)
    elif isinstance(value, list) and value:
        paths |= key_paths(value[0], prefix + "[]")
    return paths


def pick(
    events: list[dict[str, Any]],
    etype: str,
    *,
    role: str | None = None,
    content: str | None = None,
    update: str | None = None,
    last: bool = False,
) -> dict[str, Any]:
    """First (or last) event of a type, optionally narrowed by message role/content/update."""
    matches = []
    for event in events:
        if event["type"] != etype:
            continue
        message = event.get("message") or {}
        if role is not None and message.get("role") != role:
            continue
        if content is not None:
            blocks = message.get("content") or []
            if not blocks or blocks[0].get("type") != content:
                continue
        if update is not None and event["assistantMessageEvent"]["type"] != update:
            continue
        matches.append(event)
    return matches[-1] if last else matches[0]


def assistant_ends(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event["message"]
        for event in events
        if event["type"] == "message_end" and event["message"]["role"] == "assistant"
    ]


def final_text(events: list[dict[str, Any]]) -> str:
    return assistant_ends(events)[-1]["content"][0]["text"]


# ----------------------------------------------------------------------------------------------
# real flag lists (copied from the --help output of omp 18.12.6 and claude 2.1.293)
# ----------------------------------------------------------------------------------------------

OMP_HELP_BOOL_FLAGS = [
    "--prewalk",
    "--no-prewalk",
    "--plan-yolo",
    "--allow-home",
    "--print",
    "--continue",
    "--from-claude",
    "--from-codex",
    "--no-session",
    "--no-tools",
    "--no-lsp",
    "--no-pty",
    "--hide-thinking",
    "--advisor",
    "--external-thinking",
    "--no-extensions",
    "--no-skills",
    "--no-rules",
    "--no-title",
    "--no-ui",
    "--print-thoughts",
    "--auto-approve",
]
OMP_HELP_OPTIONAL_FLAGS = ["--resume"]  # "by ID prefix, path, or picker if omitted"
OMP_HELP_VALUE_FLAGS = {
    "--model": "claude-opus-5-5",
    "--smol": "haiku",
    "--slow": "opus",
    "--plan": "opus",
    "--goal": "Investigate the importer",
    "--prewalk-into": "haiku",
    "--plan-yolo-into": "haiku",
    "--provider": "anthropic",
    "--api-key": "sk-test",
    "--system-prompt": "be brief",
    "--system-prompt-template": "template.hbs",
    "--append-system-prompt": "extra rules",
    "--profile": "work",
    "--alias": "omp-work",
    "--cwd": "<dir>",
    "--mode": "json",
    "--config": "overlay.yml",
    "--add-dir": "extra",
    "--restore": "session.jsonl",
    "--session-dir": "sessions",
    "--models": "claude-sonnet,claude-haiku",
    "--tools": "read,edit",
    "--thinking": "low",
    "--service-tier": "none",
    "--hook": "hook.ts",
    "--extension": "ext.ts",
    "--skills": "git-*,docker",
    "--export": "session.jsonl",
    "--max-time": "10m",
    "--approval-mode": "yolo",
}
OMP_EXTRA_VALUE_FLAGS = {"--trusted-extension": "<file>", "--plugin-dir": "plugins"}
OMP_REPEATABLE = {"--config", "--add-dir", "--hook", "--extension", "--plugin-dir"}
OMP_ALL_VALUE_FLAGS = {**OMP_HELP_VALUE_FLAGS, **OMP_EXTRA_VALUE_FLAGS}

CLAUDE_BOOL = [
    "--allow-dangerously-skip-permissions",
    "--ax-screen-reader",
    "--bg",
    "--background",
    "--bare",
    "--brief",
    "--chrome",
    "-c",
    "--continue",
    "--dangerously-skip-permissions",
    "--desktop",
    "--disable-slash-commands",
    "--exclude-dynamic-system-prompt-sections",
    "--fork-session",
    "--forward-subagent-text",
    "--ide",
    "--include-hook-events",
    "--include-partial-messages",
    "--no-chrome",
    "--no-session-persistence",
    "-p",
    "--print",
    "--replay-user-messages",
    "--restricted",
    "--safe-mode",
    "--strict-mcp-config",
    "--tmux",
    "--verbose",
]
CLAUDE_OPTIONAL = {
    "--cloud": "my cloud session",
    "-d": "api,hooks",
    "--debug": "api",
    "--from-pr": "42",
    "--prompt-suggestions": "true",
    "--remote-control": "my-remote",
    "-r": "abc123",
    "--resume": "abc123",
    "--teleport": "session-1",
    "-w": "feature-tree",
    "--worktree": "feature-tree",
}
CLAUDE_REQUIRED = {
    "--agent": "reviewer",
    "--agents": "{}",
    "--append-system-prompt": "be brief",
    "--autocompact": "auto",
    "--debug-file": "debug.log",
    "--effort": "high",
    "--environment": "ccpool_test",
    "--fallback-model": "sonnet",
    "--input-format": "text",
    "--json-schema": "{}",
    "--max-budget-usd": "1.5",
    "--model": "opus",
    "-n": "my-session",
    "--name": "my-session",
    "--output-format": "json",
    "--permission-mode": "plan",
    "--permission-prompts": "none",
    "--plugin-dir": "plugins",
    "--plugin-url": "https://example.test/plugin.zip",
    "--remote-control-session-name-prefix": "prefix",
    "--session-id": "<uuid>",
    "--setting-sources": "user,project",
    "--settings": "{}",
    "--system-prompt": "be brief",
    "--system-prompt-snapshot": "off",
}
CLAUDE_VARIADIC = [
    "--add-dir",
    "--allowedTools",
    "--allowed-tools",
    "--betas",
    "--disallowedTools",
    "--disallowed-tools",
    "--file",
    "--mcp-config",
    "--tools",
]
CLAUDE_ALIASES = {
    "--background": "--bg",
    "-c": "--continue",
    "--allowed-tools": "--allowedTools",
    "--disallowed-tools": "--disallowedTools",
    "-d": "--debug",
    "-n": "--name",
    "-p": "--print",
    "-r": "--resume",
    "-w": "--worktree",
    "-h": "--help",
    "-v": "--version",
}
CLAUDE_REPEATABLE = {"--plugin-dir", "--plugin-url"}


def canonical(flag: str) -> str:
    return CLAUDE_ALIASES.get(flag, flag)


def omp_value(flag: str, fakes: Fakes) -> str:
    if flag == "--cwd":
        target = fakes.root / "other-cwd"
        target.mkdir(exist_ok=True)
        return str(target)
    if flag == "--trusted-extension":
        extension = fakes.root / "guard.ts"
        extension.write_text("export default function () {}\n", encoding="utf-8")
        return str(extension)
    return OMP_ALL_VALUE_FLAGS[flag]


# ----------------------------------------------------------------------------------------------
# fixture helpers and module structure
# ----------------------------------------------------------------------------------------------


def test_install_fake_cli_writes_an_executable_posix_wrapper(tmp_path: Path) -> None:
    omp = install_fake_cli(tmp_path / "bin", "omp")
    named = install_fake_cli(tmp_path / "bin", "claude", name="claude-fake")
    assert omp.name == "omp"
    assert named.name == "claude-fake"
    assert os.access(omp, os.X_OK)
    text = omp.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    script = shlex.quote(str(FAKES_DIR / "fake_omp.py"))
    assert f"exec {shlex.quote(sys.executable)} {script}" in text
    assert text.rstrip().endswith('"$@"')
    with pytest.raises(ValueError, match="unknown fake CLI kind"):
        install_fake_cli(tmp_path / "bin", "codex")  # type: ignore[arg-type]


def test_write_scenario_and_read_invocations_roundtrip(tmp_path: Path) -> None:
    steps = [{"op": "final", "text": "hi"}]
    path = write_scenario(tmp_path / "nested" / "s.json", steps, log_env=["A"], ignore_sigterm=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "steps": steps,
        "log_env": ["A"],
        "ignore_sigterm": True,
    }
    log = tmp_path / "log.jsonl"
    assert read_invocations(log) == []
    log.write_text('{"argv": ["-p"]}\n\n{"argv": []}\n', encoding="utf-8")
    assert read_invocations(log) == [{"argv": ["-p"]}, {"argv": []}]
    assert fake_agent_env(path, log) == {
        "FAKE_AGENT_SCENARIO": str(path),
        "FAKE_AGENT_LOG": str(log),
    }
    assert fake_agent_env() == {}


def test_docs_edit_steps_defaults_and_overrides() -> None:
    steps = docs_edit_steps()
    assert [step["op"] for step in steps] == ["write_file", "tool_call", "cost", "final"]
    assert steps[0] == {
        "op": "write_file",
        "path": "docs/agent-note.md",
        "content": "Hello from the agent\n",
    }
    assert steps[1]["tool"] == "write"
    assert steps[2]["usd"] == 0.02
    assert steps[3]["report"] == {
        "status": "completed",
        "summary": "Created docs/agent-note.md",
        "files_changed": ["docs/agent-note.md"],
    }
    custom = docs_edit_steps("docs/x.md", "body", 0.5, {"status": "blocked"})
    assert custom[0]["path"] == "docs/x.md" and custom[2]["usd"] == 0.5
    assert custom[3]["report"] == {"status": "blocked"}


def test_fakes_import_only_the_standard_library() -> None:
    for name in ("fake_omp.py", "fake_claude.py"):
        tree = ast.parse((FAKES_DIR / name).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        assert imported <= set(sys.stdlib_module_names), (name, imported)


def test_shared_plumbing_is_identical_in_both_fakes() -> None:
    blocks = []
    for name in ("fake_omp.py", "fake_claude.py"):
        source = (FAKES_DIR / name).read_text(encoding="utf-8")
        match = re.search(
            r"# >>> shared plumbing[^\n]*\n(.*?)\n# <<< end shared plumbing", source, re.S
        )
        assert match, name
        blocks.append(match.group(1).strip())
    assert blocks[0] == blocks[1]


# ----------------------------------------------------------------------------------------------
# command line surface common to both CLIs
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "flag", "expected"),
    [
        ("omp", "--version", "omp/18.12.6"),
        ("omp", "-v", "omp/18.12.6"),
        ("claude", "--version", "2.1.293 (Claude Code)"),
        ("claude", "-v", "2.1.293 (Claude Code)"),
    ],
)
def test_version_strings(fakes: Fakes, kind: str, flag: str, expected: str) -> None:
    proc = fakes.run(kind, [flag], stdin=None)
    assert proc.returncode == 0
    assert proc.stdout == expected + "\n"
    assert proc.stderr == ""


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_is_a_one_line_stub(fakes: Fakes, kind: str, flag: str) -> None:
    proc = fakes.run(kind, [flag], stdin=None)
    assert proc.returncode == 0
    assert len(proc.stdout.splitlines()) == 1
    assert kind in proc.stdout.lower()


@pytest.mark.parametrize(
    ("kind", "flag", "message"),
    [
        ("omp", "--bogus-flag", "Error: unknown flag: --bogus-flag"),
        ("omp", "--bogus-flag=1", "Error: unknown flag: --bogus-flag"),
        ("omp", "-x", "Error: unknown flag: -x"),
        ("claude", "--bogus-flag", "error: unknown option '--bogus-flag'"),
        ("claude", "--bogus-flag=1", "error: unknown option '--bogus-flag=1'"),
        ("claude", "-x", "error: unknown option '-x'"),
    ],
)
def test_unknown_flag_exits_2_like_the_real_cli(
    fakes: Fakes, kind: str, flag: str, message: str
) -> None:
    proc = fakes.run(kind, ["-p", flag, "prompt"])
    assert proc.returncode == 2
    assert proc.stderr.startswith(message)
    assert proc.stdout == ""


@pytest.mark.parametrize(
    ("kind", "flag"),
    [
        ("omp", "--prewalks"),
        ("omp", "--no-model"),
        ("omp", "--Print"),
        ("omp", "--max_time"),
        ("claude", "--allowedtools"),
        ("claude", "--output_format"),
        ("claude", "--no-verbose"),
        ("claude", "--permission-modes"),
    ],
)
def test_near_miss_flags_are_not_accepted(fakes: Fakes, kind: str, flag: str) -> None:
    assert fakes.run(kind, ["-p", flag, "x"]).returncode == 2


@pytest.mark.parametrize(
    ("kind", "args"),
    [
        ("omp", ["-p", "--model"]),
        ("omp", ["-p", "--max-time"]),
        ("claude", ["-p", "--model"]),
        ("claude", ["-p", "--tools"]),
        ("claude", ["-p", "--allowedTools"]),
    ],
)
def test_missing_flag_value_exits_2(fakes: Fakes, kind: str, args: list[str]) -> None:
    proc = fakes.run(kind, args)
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert args[-1] in proc.stderr


@pytest.mark.parametrize("kind", KINDS)
def test_prompt_via_stdin_and_positional(fakes: Fakes, kind: str) -> None:
    proc = fakes.run(kind, ["-p"], stdin="prompt from stdin\n")
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["prompt"] == "prompt from stdin\n"
    proc = fakes.run(kind, ["-p", "positional prompt"], stdin="ignored stdin")
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["prompt"] == "positional prompt"
    proc = fakes.run(kind, ["-p", "joined", "prompt", "words"], stdin=None)
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["prompt"] == "joined prompt words"


@pytest.mark.parametrize("kind", KINDS)
def test_large_multiline_stdin_prompt_is_received_intact(fakes: Fakes, kind: str) -> None:
    prompt = ("line of text with unicode \u00e9\u4e2d\n" * 20000) + "no trailing newline"
    proc = fakes.run(kind, ["-p"], stdin=prompt)
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["prompt"] == prompt


@pytest.mark.parametrize("kind", KINDS)
def test_default_scenario_is_a_final_ok(fakes: Fakes, kind: str) -> None:
    assert not fakes.scenario.exists()
    proc = fakes.run(kind, ["-p"])
    assert proc.returncode == 0
    assert proc.stdout == "ok\n"


@pytest.mark.parametrize("kind", KINDS)
def test_a_run_without_final_finishes_with_ok(fakes: Fakes, kind: str) -> None:
    marker = fakes.work / "ran.txt"
    proc = fakes.run(
        kind,
        ["-p"],
        steps=[{"op": "write_file", "path": "ran.txt", "content": "1"}],
    )
    assert proc.returncode == 0
    assert proc.stdout == "ok\n"
    assert marker.read_text(encoding="utf-8") == "1"


# ----------------------------------------------------------------------------------------------
# flag acceptance: every flag of the real help
# ----------------------------------------------------------------------------------------------


def test_omp_flag_lists_match_the_size_of_the_real_help() -> None:
    assert len(OMP_HELP_BOOL_FLAGS) + len(OMP_HELP_OPTIONAL_FLAGS) + len(OMP_HELP_VALUE_FLAGS) == 53


@pytest.mark.parametrize("flag", OMP_HELP_BOOL_FLAGS)
def test_omp_accepts_every_boolean_flag(fakes: Fakes, flag: str) -> None:
    proc = fakes.run("omp", ["-p", flag])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "ok\n"
    assert fakes.last_invocation()["parsed"]["flags"][flag] is True


@pytest.mark.parametrize("flag", sorted(OMP_ALL_VALUE_FLAGS))
def test_omp_accepts_every_value_flag(fakes: Fakes, flag: str) -> None:
    value = omp_value(flag, fakes)
    proc = fakes.run("omp", ["-p", flag, value, "the prompt"])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() != ""
    recorded = fakes.last_invocation()["parsed"]
    assert recorded["flags"][flag] == ([value] if flag in OMP_REPEATABLE else value)
    assert recorded["positional"] == ["the prompt"]


@pytest.mark.parametrize("flag", sorted(OMP_ALL_VALUE_FLAGS))
def test_omp_accepts_the_equals_form_of_value_flags(fakes: Fakes, flag: str) -> None:
    value = omp_value(flag, fakes)
    proc = fakes.run("omp", ["-p", f"{flag}={value}"])
    assert proc.returncode == 0, proc.stderr
    recorded = fakes.last_invocation()["parsed"]["flags"][flag]
    assert recorded == ([value] if flag in OMP_REPEATABLE else value)


@pytest.mark.parametrize(
    ("args", "flag", "expected"),
    [
        (["-p"], "--print", True),
        (["-p", "-c"], "--continue", True),
        (["-pc"], "--continue", True),
        (["-p", "-r"], "--resume", True),
        (["-p", "-r", "abc123"], "--resume", "abc123"),
        (["-p", "--resume=abc123"], "--resume", "abc123"),
        (["-p", "-e", "ext.ts"], "--extension", ["ext.ts"]),
        (["-p", "-eext.ts"], "--extension", ["ext.ts"]),
        (["-p", "-e", "a.ts", "--extension", "b.ts"], "--extension", ["a.ts", "b.ts"]),
    ],
)
def test_omp_short_flags_and_optional_values(
    fakes: Fakes, args: list[str], flag: str, expected: Any
) -> None:
    proc = fakes.run("omp", args)
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["parsed"]["flags"][flag] == expected


def test_omp_double_dash_ends_options(fakes: Fakes) -> None:
    proc = fakes.run("omp", ["-p", "--", "--not-a-flag", "text"])
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["prompt"] == "--not-a-flag text"


def test_claude_flag_lists_match_the_real_help() -> None:
    names = [*CLAUDE_BOOL, *CLAUDE_OPTIONAL, *CLAUDE_REQUIRED, *CLAUDE_VARIADIC]
    names += ["--help", "-h", "--version", "-v"]
    distinct = {canonical(name) for name in names}
    assert len(distinct) == 66  # one entry per option of the real claude --help
    assert {"-h", "-v"}.isdisjoint(distinct)


@pytest.mark.parametrize("flag", CLAUDE_BOOL)
def test_claude_accepts_every_boolean_flag(fakes: Fakes, flag: str) -> None:
    proc = fakes.run("claude", ["-p", flag])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "ok\n"
    assert fakes.last_invocation()["parsed"]["flags"][canonical(flag)] is True


@pytest.mark.parametrize("flag", sorted(CLAUDE_OPTIONAL))
def test_claude_accepts_optional_value_flags_bare_and_with_a_value(fakes: Fakes, flag: str) -> None:
    proc = fakes.run("claude", ["-p", flag])
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["parsed"]["flags"][canonical(flag)] is True
    proc = fakes.run("claude", ["-p", flag, CLAUDE_OPTIONAL[flag]])
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["parsed"]["flags"][canonical(flag)] == CLAUDE_OPTIONAL[flag]


@pytest.mark.parametrize("flag", sorted(CLAUDE_REQUIRED))
def test_claude_accepts_every_required_value_flag(fakes: Fakes, flag: str) -> None:
    value = str(uuid.uuid4()) if CLAUDE_REQUIRED[flag] == "<uuid>" else CLAUDE_REQUIRED[flag]
    proc = fakes.run("claude", ["-p", flag, value])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() != ""
    recorded = fakes.last_invocation()["parsed"]["flags"][canonical(flag)]
    assert recorded == ([value] if canonical(flag) in CLAUDE_REPEATABLE else value)


@pytest.mark.parametrize("flag", CLAUDE_VARIADIC)
def test_claude_accepts_every_variadic_flag(fakes: Fakes, flag: str) -> None:
    proc = fakes.run("claude", ["-p", flag, "A", "B"])
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["parsed"]["flags"][canonical(flag)] == ["A", "B"]
    proc = fakes.run("claude", ["-p", flag, "A,B"])
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["parsed"]["flags"][canonical(flag)] == ["A,B"]


# ----------------------------------------------------------------------------------------------
# claude: commander parsing gotchas
# ----------------------------------------------------------------------------------------------


def test_claude_variadic_option_swallows_a_following_positional(fakes: Fakes) -> None:
    proc = fakes.run(
        "claude", ["-p", "--allowedTools", "Edit", "Read", "do the work"], stdin="prompt on stdin"
    )
    assert proc.returncode == 0, proc.stderr
    recorded = fakes.last_invocation()
    assert recorded["parsed"]["flags"]["--allowedTools"] == ["Edit", "Read", "do the work"]
    assert recorded["parsed"]["positional"] == []
    assert recorded["prompt"] == "prompt on stdin"  # the positional prompt was swallowed


def test_claude_double_dash_delivers_the_prompt_after_a_variadic_option(fakes: Fakes) -> None:
    proc = fakes.run(
        "claude", ["-p", "--allowedTools", "Edit", "--", "do the work"], stdin="ignored"
    )
    assert proc.returncode == 0, proc.stderr
    recorded = fakes.last_invocation()
    assert recorded["parsed"]["flags"]["--allowedTools"] == ["Edit"]
    assert recorded["prompt"] == "do the work"


def test_claude_variadic_option_stops_at_the_next_flag(fakes: Fakes) -> None:
    args = ["-p", "--tools", "Read", "Edit", "--model", "sonnet", "the prompt"]
    proc = fakes.run("claude", args, stdin=None)
    assert proc.returncode == 0, proc.stderr
    recorded = fakes.last_invocation()["parsed"]
    assert recorded["flags"]["--tools"] == ["Read", "Edit"]
    assert recorded["flags"]["--model"] == "sonnet"
    assert recorded["positional"] == ["the prompt"]


def test_claude_inline_value_does_not_make_a_variadic_option_greedy(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p", "--allowedTools=Edit", "the prompt"], stdin=None)
    assert proc.returncode == 0, proc.stderr
    recorded = fakes.last_invocation()["parsed"]
    assert recorded["flags"]["--allowedTools"] == ["Edit"]
    assert recorded["positional"] == ["the prompt"]


def test_claude_optional_value_option_swallows_a_following_prompt(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p", "--debug", "the prompt"], stdin="prompt on stdin")
    assert proc.returncode == 0, proc.stderr
    recorded = fakes.last_invocation()
    assert recorded["parsed"]["flags"]["--debug"] == "the prompt"
    assert recorded["prompt"] == "prompt on stdin"


def test_claude_required_argument_is_taken_even_when_it_looks_like_a_flag(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p", "--model", "--verbose", "x"], stdin=None)
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["parsed"]["flags"]["--model"] == "--verbose"


def test_claude_clustered_short_flags(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-pc", "the prompt"], stdin=None)
    assert proc.returncode == 0, proc.stderr
    flags = fakes.last_invocation()["parsed"]["flags"]
    assert flags["--print"] is True and flags["--continue"] is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--output-format", "xml"],
            "error: option '--output-format <format>' argument 'xml' is invalid. "
            "Allowed choices are text, json, stream-json.",
        ),
        (
            ["--input-format", "bogus"],
            "error: option '--input-format <format>' argument 'bogus' is invalid. "
            "Allowed choices are text, stream-json.",
        ),
        (
            ["--permission-mode", "default"],
            "error: option '--permission-mode <mode>' argument 'default' is invalid. "
            "Allowed choices are acceptEdits, auto, bypassPermissions, manual, dontAsk, plan.",
        ),
        (
            ["--permission-prompts", "x"],
            "error: option '--permission-prompts <target>' argument 'x' is invalid. "
            "Allowed choices are host, none.",
        ),
        (
            ["--system-prompt-snapshot", "maybe"],
            "error: option '--system-prompt-snapshot <on|off>' argument 'maybe' is invalid. "
            "Allowed choices are on, off.",
        ),
        (
            ["--max-budget-usd", "0"],
            "error: option '--max-budget-usd <amount>' argument '0' is invalid. "
            "--max-budget-usd must be a positive number greater than 0",
        ),
        (
            ["--max-budget-usd", "abc"],
            "error: option '--max-budget-usd <amount>' argument 'abc' is invalid. "
            "--max-budget-usd must be a positive number greater than 0",
        ),
        (
            ["--allowedTools"],
            "error: option '--allowedTools, --allowed-tools <tools...>' argument missing",
        ),
    ],
)
def test_claude_option_argument_errors_use_the_commander_messages(
    fakes: Fakes, args: list[str], message: str
) -> None:
    proc = fakes.run("claude", ["-p", *args])
    assert proc.returncode == 2
    assert proc.stderr == message + "\n"
    assert proc.stdout == ""


def test_claude_version_wins_over_later_unknown_flags(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["--version", "--bogus"], stdin=None)
    assert proc.returncode == 0
    assert proc.stdout == "2.1.293 (Claude Code)\n"


def test_claude_requires_print_mode(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["--output-format", "json"])
    assert proc.returncode == 2
    assert "interactive mode is not emulated" in proc.stderr


def test_claude_empty_prompt_is_the_real_error(fakes: Fakes) -> None:
    proc = fakes.run(
        "claude",
        ["-p"],
        stdin="",
        steps=[{"op": "write_file", "path": "ran.txt", "content": "1"}],
    )
    assert proc.returncode == 1
    assert proc.stderr == (
        "Error: Input must be provided either through stdin or as a prompt argument "
        "when using --print\n"
    )
    assert not (fakes.work / "ran.txt").exists()


def test_claude_session_id_must_be_a_uuid(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p", "--session-id", "not-a-uuid"])
    assert proc.returncode == 1
    assert proc.stderr == "Error: Invalid session ID. Must be a valid UUID.\n"


def test_claude_unknown_effort_only_warns(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p", "--effort", "bogus"])
    assert proc.returncode == 0
    assert "Warning: Unknown --effort value 'bogus'" in proc.stderr


# ----------------------------------------------------------------------------------------------
# claude: output formats
# ----------------------------------------------------------------------------------------------

REAL_CLAUDE_ERROR = json.loads((FIXTURES / "claude_real_credit_error.json").read_text("utf-8"))


def test_claude_verbose_json_matches_the_real_fixture_key_sets(fakes: Fakes) -> None:
    args = ["-p", "--verbose", "--output-format", "json", "--permission-mode", "dontAsk"]
    proc = fakes.run("claude", [*args, "--tools", ""], steps=[{"op": "final", "text": "pong"}])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    init, assistant, result = data
    assert [message["type"] for message in data] == ["system", "assistant", "result"]
    real_init, real_assistant, real_result = REAL_CLAUDE_ERROR
    assert set(init) == set(real_init)
    assert init["tools"] == [] and init["permissionMode"] == "dontAsk"
    assert init["claude_code_version"] == "2.1.293" and init["mcp_servers"] == []
    assert set(assistant) == set(real_assistant) - {"error", "request_id", "is_api_error_message"}
    assert set(assistant["message"]) == set(real_assistant["message"])
    assert set(assistant["message"]["usage"]) == set(real_assistant["message"]["usage"])
    assert assistant["message"]["content"] == [{"type": "text", "text": "pong"}]
    assert set(result) == set(real_result)
    assert result["is_error"] is False
    assert result["subtype"] == "success" and result["terminal_reason"] == "completed"
    assert result["stop_reason"] == "end_turn" and result["result"] == "pong"
    assert result["api_error_status"] is None and result["permission_denials"] == []
    nested = {path for path in key_paths(result) if not path.startswith("modelUsage.")}
    assert nested == key_paths(real_result)
    assert set(result["modelUsage"]) == {"claude-opus-5-5"}
    assert "costUSD" in result["modelUsage"]["claude-opus-5-5"]


def test_claude_json_without_verbose_is_a_single_result_object(fakes: Fakes) -> None:
    """The real CLI prints only the result message unless --verbose is also given."""
    proc = fakes.run(
        "claude", ["-p", "--output-format", "json"], steps=[{"op": "final", "text": "x"}]
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert isinstance(result, dict)
    assert result["type"] == "result" and result["result"] == "x"
    assert set(result) == set(REAL_CLAUDE_ERROR[2])


def test_claude_stream_json_requires_verbose(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p", "--output-format", "stream-json"])
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr == (
        "Error: When using --print, --output-format=stream-json requires --verbose\n"
    )


def test_claude_stream_json_emits_the_three_messages_as_lines(fakes: Fakes) -> None:
    args = ["-p", "--verbose", "--output-format", "stream-json"]
    proc = fakes.run("claude", args, steps=[{"op": "final", "text": "streamed"}])
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert len(lines) == 3
    messages = [json.loads(line) for line in lines]
    assert [message["type"] for message in messages] == ["system", "assistant", "result"]
    assert messages[2]["result"] == "streamed"


def test_claude_init_reflects_the_command_line(fakes: Fakes) -> None:
    session = str(uuid.uuid4())
    args = ["-p", "--verbose", "--output-format", "json", "--model", "sonnet"]
    args += ["--permission-mode", "acceptEdits", "--session-id", session, "--tools", "Read", "Edit"]
    proc = fakes.run("claude", args)
    assert proc.returncode == 0, proc.stderr
    init, assistant, result = json.loads(proc.stdout)
    assert init["model"] == "sonnet" and assistant["message"]["model"] == "sonnet"
    assert init["permissionMode"] == "acceptEdits"
    assert init["tools"] == ["Read", "Edit"]
    assert init["session_id"] == session == result["session_id"]
    assert os.path.realpath(init["cwd"]) == os.path.realpath(fakes.work)


def test_claude_tools_flag_variants_in_init(fakes: Fakes) -> None:
    for args, expected in [
        (["--tools", ""], []),
        (["--tools", "Bash,Edit"], ["Bash", "Edit"]),
        (["--tools", "Bash", "Edit"], ["Bash", "Edit"]),
    ]:
        proc = fakes.run(
            "claude", ["-p", "--verbose", "--output-format", "json", *args, "--", "go"]
        )
        assert json.loads(proc.stdout)[0]["tools"] == expected
    proc = fakes.run("claude", ["-p", "--verbose", "--output-format", "json", "--", "go"])
    assert "Bash" in json.loads(proc.stdout)[0]["tools"]


def test_claude_text_output_is_the_final_text(fakes: Fakes) -> None:
    proc = fakes.run("claude", ["-p"], steps=[{"op": "final", "text": "plain answer"}])
    assert proc.returncode == 0
    assert proc.stdout == "plain answer\n"


def test_claude_blocked_tool_calls_become_permission_denials(fakes: Fakes) -> None:
    steps = [
        {"op": "tool_call", "tool": "Read", "args": {"file_path": "a.md"}},
        {"op": "tool_call", "tool": "Write", "args": {"file_path": "/etc/x"}, "blocked": True},
        {"op": "tool_call", "tool": "Bash", "args": {"command": "rm -rf /"}, "blocked": True},
        {"op": "final", "text": "done"},
    ]
    proc = fakes.run("claude", ["-p", "--output-format", "json"], steps=steps)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    denials = result["permission_denials"]
    assert [entry["tool_name"] for entry in denials] == ["Write", "Bash"]
    assert denials[0]["tool_input"] == {"file_path": "/etc/x"}
    assert all(set(entry) == {"tool_name", "tool_use_id", "tool_input"} for entry in denials)
    assert len({entry["tool_use_id"] for entry in denials}) == 2
    assert result["num_turns"] == 4


def test_claude_budget_overrun_stops_the_run_with_error_max_budget_usd(fakes: Fakes) -> None:
    steps = [
        {"op": "cost", "usd": 0.03},
        {"op": "cost", "usd": 0.03},
        {"op": "write_file", "path": "after.txt", "content": "must not run"},
        {"op": "final", "text": "unreachable"},
    ]
    args = ["-p", "--output-format", "json", "--max-budget-usd", "0.05"]
    proc = fakes.run("claude", args, steps=steps)
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["subtype"] == "error_max_budget_usd"
    assert result["is_error"] is True
    assert result["terminal_reason"] == "budget_exhausted"
    assert result["total_cost_usd"] == pytest.approx(0.06)
    assert result["total_cost_usd"] >= 0.05
    assert set(result) == set(REAL_CLAUDE_ERROR[2])
    assert not (fakes.work / "after.txt").exists()


def test_claude_spend_below_the_budget_succeeds_and_reports_the_cost(fakes: Fakes) -> None:
    steps = [
        {"op": "cost", "usd": 0.01},
        {"op": "cost", "usd": 0.02},
        {"op": "final", "text": "ok"},
    ]
    args = ["-p", "--verbose", "--output-format", "json", "--max-budget-usd", "1"]
    proc = fakes.run("claude", args, steps=steps)
    assert proc.returncode == 0
    result = json.loads(proc.stdout)[2]
    assert result["is_error"] is False and result["subtype"] == "success"
    assert result["total_cost_usd"] == pytest.approx(0.03)
    assert result["modelUsage"]["claude-opus-5-5"]["costUSD"] == pytest.approx(0.03)


CLAUDE_FAILS = [
    ("credit", 400, "Credit balance is too low", "billing_error"),
    (
        "auth",
        None,
        "Failed to authenticate: OAuth session expired and could not be refreshed",
        "authentication_failed",
    ),
    ("rate_limit", 429, "rate limit exceeded", "rate_limit"),
    ("api_error", 500, "API Error: 500", "server_error"),
]


@pytest.mark.parametrize(("kind", "status", "text", "error"), CLAUDE_FAILS)
def test_claude_fail_kinds_print_the_real_error_result(
    fakes: Fakes, kind: str, status: int | None, text: str, error: str
) -> None:
    args = ["-p", "--verbose", "--output-format", "json"]
    proc = fakes.run("claude", args, steps=[{"op": "fail", "kind": kind}])
    assert proc.returncode == 1
    _, assistant, result = json.loads(proc.stdout)
    assert result["is_error"] is True
    assert result["terminal_reason"] == "api_error"
    assert result["api_error_status"] == status
    assert result["result"].startswith(text)
    assert result["total_cost_usd"] == 0 and result["modelUsage"] == {}
    assert assistant["error"] == error and assistant["is_api_error_message"] is True
    assert assistant["message"]["model"] == "<synthetic>"
    assert assistant["message"]["content"][0]["text"] == result["result"]


@pytest.mark.parametrize(
    ("kind", "fixture"),
    [("credit", "claude_real_credit_error.json"), ("auth", "claude_real_auth_expired.json")],
)
def test_claude_fail_output_equals_the_real_capture(fakes: Fakes, kind: str, fixture: str) -> None:
    """Apart from ids, timestamps, durations and cwd the output is the captured one."""
    real = json.loads((FIXTURES / fixture).read_text("utf-8"))
    args = ["-p", "--verbose", "--output-format", "json", "--permission-mode", "dontAsk"]
    proc = fakes.run("claude", [*args, "--tools", ""], steps=[{"op": "fail", "kind": kind}])
    mine = json.loads(proc.stdout)
    volatile = {"cwd", "session_id", "uuid", "timestamp", "request_id", "duration_ms", "id"}

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in volatile}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    assert scrub(mine) == scrub(real)
    assert ("request_id" in mine[1]) == ("request_id" in real[1])


def test_claude_fail_message_override_and_text_mode(fakes: Fakes) -> None:
    steps = [{"op": "fail", "kind": "credit", "message": "custom billing text"}]
    proc = fakes.run("claude", ["-p"], steps=steps)
    assert proc.returncode == 1
    assert proc.stdout == "custom billing text\n"
    proc = fakes.run("claude", ["-p", "--output-format", "json"], steps=steps)
    assert json.loads(proc.stdout)["result"] == "custom billing text"


# ----------------------------------------------------------------------------------------------
# omp: --mode json event stream
# ----------------------------------------------------------------------------------------------

OMP_JSON = ["--mode", "json", "-p"]
PONG = "Reply with exactly the single word: pong"
TOOL_SHAPE_PROBES: list[dict[str, Any]] = [
    {"etype": "session"},
    {"etype": "thinking_level_changed"},
    {"etype": "agent_start"},
    {"etype": "turn_start"},
    {"etype": "tool_execution_start"},
    {"etype": "tool_execution_update"},
    {"etype": "tool_execution_end"},
    {"etype": "message_start", "role": "user"},
    {"etype": "message_start", "role": "toolResult"},
    {"etype": "message_end", "role": "toolResult"},
    {"etype": "message_start", "role": "assistant", "content": "toolCall"},
    {"etype": "message_end", "role": "assistant", "content": "toolCall"},
    {"etype": "turn_end", "role": "assistant", "content": "toolCall"},
    {"etype": "message_start", "role": "assistant", "content": "text", "last": True},
    {"etype": "message_end", "role": "assistant", "content": "text", "last": True},
    {"etype": "turn_end", "role": "assistant", "content": "text", "last": True},
    {"etype": "agent_end"},
    *[
        {"etype": "message_update", "update": name}
        for name in (
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
            "text_start",
            "text_delta",
            "text_end",
        )
    ],
]


def test_omp_event_sequence_matches_the_real_minimal_fixture(fakes: Fakes) -> None:
    args = [*OMP_JSON, "--tools", "read,grep,glob,edit,write"]
    proc = fakes.run("omp", args, stdin=PONG, steps=[{"op": "final", "text": "pong"}])
    assert proc.returncode == 0, proc.stderr
    events = parse_ndjson(proc.stdout)
    real = load_ndjson("omp_real_minimal.ndjson")
    assert signature(events) == signature(real)
    for mine, captured in zip(events, real, strict=True):
        assert key_paths(mine) == key_paths(captured), mine["type"]
    assert events[0]["type"] == "session" and events[0]["version"] == 3
    assert events[4]["message"]["content"][0]["text"] == PONG
    assert events[-1]["isTerminal"] is True and events[-1]["yielded"] is True
    assert final_text(events) == "pong"


def test_omp_assistant_usage_keys_match_the_real_fixture(fakes: Fakes) -> None:
    proc = fakes.run("omp", OMP_JSON, stdin=PONG, steps=[{"op": "final", "text": "pong"}])
    mine = assistant_ends(parse_ndjson(proc.stdout))[-1]
    captured = assistant_ends(load_ndjson("omp_real_minimal.ndjson"))[-1]
    assert set(mine["usage"]) == set(captured["usage"])
    assert set(mine["usage"]["cost"]) == set(captured["usage"]["cost"])
    assert set(mine) == set(captured)
    assert mine["stopReason"] == "stop" and mine["usage"]["cost"]["total"] == 0
    assert (
        mine["requestControls"]["tools"]["active"] == mine["requestControls"]["tools"]["declared"]
    )


def test_omp_tool_calls_have_the_real_fixture_shapes(fakes: Fakes) -> None:
    steps = [
        {
            "op": "tool_call",
            "tool": "write",
            "args": {"path": "hello.md", "content": "hi"},
            "intent": "Creating hello file",
            "partial": "Writing 2 bytes to hello.md...",
            "result": "Successfully wrote 2 bytes to hello.md",
        },
        {"op": "final", "text": "DONE"},
    ]
    proc = fakes.run("omp", [*OMP_JSON, "--tools", "read,grep,glob,edit,write"], steps=steps)
    assert proc.returncode == 0, proc.stderr
    events = parse_ndjson(proc.stdout)
    real = load_ndjson("omp_real_tool_calls.ndjson")
    problems = []
    for probe in TOOL_SHAPE_PROBES:
        mine_paths, real_paths = key_paths(pick(events, **probe)), key_paths(pick(real, **probe))
        if mine_paths != real_paths:
            problems.append((probe, sorted(mine_paths ^ real_paths)))
    assert not problems


def test_omp_tool_execution_events_and_blocked_calls_set_is_error(fakes: Fakes) -> None:
    steps = [
        {"op": "tool_call", "tool": "write", "args": {"path": "a.md", "content": "x"}},
        {
            "op": "tool_call",
            "tool": "write",
            "args": {"path": "/etc/hosts", "content": "x"},
            "blocked": True,
            "result": "proto: writes outside the worktree are blocked",
        },
        {"op": "final", "text": "done"},
    ]
    proc = fakes.run("omp", OMP_JSON, steps=steps)
    assert proc.returncode == 0, proc.stderr
    events = parse_ndjson(proc.stdout)
    starts = [e for e in events if e["type"] == "tool_execution_start"]
    ends = [e for e in events if e["type"] == "tool_execution_end"]
    assert [e["isError"] for e in ends] == [False, True]
    assert [e["toolCallId"] for e in starts] == [e["toolCallId"] for e in ends]
    assert len({e["toolCallId"] for e in starts}) == 2
    assert starts[0]["args"] == {"path": "a.md", "content": "x"} and "i" not in starts[0]["args"]
    assert (
        ends[1]["result"]["content"][0]["text"] == "proto: writes outside the worktree are blocked"
    )
    results = [e["message"] for e in events if e["type"] == "message_end"]
    results = [m for m in results if m["role"] == "toolResult"]
    assert [m["isError"] for m in results] == [False, True]
    turn_ends = [e for e in events if e["type"] == "turn_end"]
    assert [len(e["toolResults"]) for e in turn_ends] == [1, 1, 0]
    assert [m["stopReason"] for m in assistant_ends(events)] == ["toolUse", "toolUse", "stop"]
    call = pick(events, "message_update", update="toolcall_end")["assistantMessageEvent"][
        "toolCall"
    ]
    assert call["arguments"]["i"] == "Calling write" and call["name"] == "write"
    opening = pick(events, "message_start", role="assistant", content="toolCall")["message"]
    assert opening["content"][0]["arguments"] == {}  # real streams start with empty arguments
    assert opening["stopReason"] == "stop"  # placeholder until the message completes
    closing = pick(events, "message_end", role="assistant", content="toolCall")["message"]
    assert closing["content"][0]["intent"] == "Calling write"
    assert closing["stopReason"] == "toolUse"
    pieces = [
        e["assistantMessageEvent"]["delta"]
        for e in events
        if e["type"] == "message_update" and e["assistantMessageEvent"]["type"] == "toolcall_delta"
    ]
    assert json.loads("".join(pieces[:0]) or "{}") == {}


def test_omp_events_are_flushed_one_by_one(fakes: Fakes) -> None:
    steps = [
        {"op": "cost", "usd": 0.01},
        {"op": "sleep", "seconds": 1.0},
        {"op": "cost", "usd": 0.02},
        {"op": "sleep", "seconds": 1.0},
        {"op": "final", "text": "done"},
    ]
    arrivals: list[tuple[float, float]] = []
    alive_at_first_cost = None
    with running(fakes, "omp", OMP_JSON, steps=steps) as proc:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            event = json.loads(line)
            message = event.get("message") or {}
            if (
                event["type"] == "message_end"
                and message.get("role") == "assistant"
                and message["usage"]["cost"]["total"]
            ):
                arrivals.append((time.monotonic(), message["usage"]["cost"]["total"]))
                if len(arrivals) == 1:
                    alive_at_first_cost = proc.poll() is None
        assert proc.wait(timeout=10) == 0
    assert [cost for _, cost in arrivals] == [0.01, 0.02]
    assert alive_at_first_cost is True, "the first cost event only arrived after exit"
    assert arrivals[1][0] - arrivals[0][0] >= 0.8, "events were buffered, not flushed one by one"


def test_omp_cost_steps_are_summed_over_assistant_message_ends(fakes: Fakes) -> None:
    steps = [{"op": "cost", "usd": 0.01}, {"op": "cost", "usd": 0.02}, {"op": "cost", "usd": 0.005}]
    proc = fakes.run("omp", OMP_JSON, steps=[*steps, {"op": "final", "text": "ok"}])
    events = parse_ndjson(proc.stdout)
    totals = [m["usage"]["cost"]["total"] for m in assistant_ends(events)]
    assert totals == [0.01, 0.02, 0.005, 0]
    assert sum(totals) == pytest.approx(0.035)
    for message in assistant_ends(events)[:3]:
        parts = message["usage"]["cost"]
        assert parts["input"] + parts["output"] == pytest.approx(parts["total"])
        assert message["stopReason"] == "toolUse"
    assert len(events[-1]["messages"]) == 5  # user + three spend messages + final


def test_omp_request_controls_follow_the_tool_flags(fakes: Fakes) -> None:
    def tools_of(args: list[str]) -> dict[str, Any]:
        proc = fakes.run("omp", [*OMP_JSON, *args])
        assert proc.returncode == 0, proc.stderr
        return assistant_ends(parse_ndjson(proc.stdout))[-1]["requestControls"]["tools"]

    assert tools_of(["--tools", "read,grep"]) == {
        "declared": ["read", "grep"],
        "deferred": [],
        "active": ["read", "grep"],
    }
    assert tools_of(["--no-tools"])["declared"] == []
    assert tools_of(["--tools", ""])["declared"] == []
    assert "bash" in tools_of([])["declared"]


def test_omp_model_provider_and_thinking_flags(fakes: Fakes) -> None:
    proc = fakes.run("omp", [*OMP_JSON, "--model", "openai/gpt-5.2", "--thinking", "high"])
    events = parse_ndjson(proc.stdout)
    message = assistant_ends(events)[-1]
    assert (message["provider"], message["model"]) == ("openai", "gpt-5.2")
    assert message["requestControls"]["effort"] == {"topLevel": "high", "tail": "high"}
    level = pick(events, "thinking_level_changed")
    assert (level["thinkingLevel"], level["configured"], level["resolved"]) == ("high",) * 3
    proc = fakes.run("omp", OMP_JSON)
    events = parse_ndjson(proc.stdout)
    message = assistant_ends(events)[-1]
    assert (message["provider"], message["model"]) == ("anthropic", "claude-opus-5-5")
    level = pick(events, "thinking_level_changed")
    assert (level["thinkingLevel"], level["configured"]) == ("low", "auto")
    proc = fakes.run("omp", [*OMP_JSON, "--model", "opus", "--provider", "anthropic"])
    assert assistant_ends(parse_ndjson(proc.stdout))[-1]["model"] == "opus"


def test_omp_text_mode_prints_only_the_final_text(fakes: Fakes) -> None:
    steps = [
        {"op": "tool_call", "tool": "read", "args": {"path": "a"}},
        {"op": "cost", "usd": 0.5},
        {"op": "final", "text": "the answer"},
    ]
    proc = fakes.run("omp", ["-p"], steps=steps)
    assert proc.returncode == 0
    assert proc.stdout == "the answer\n"
    assert proc.stderr == "Working...\n"


OMP_FAILS = [
    ("auth", 401, "Failed to authenticate"),
    ("credit", 400, "Credit balance is too low"),
    ("rate_limit", 429, "rate limit exceeded"),
    ("api_error", 500, "Internal server error"),
]


@pytest.mark.parametrize(("kind", "status", "needle"), OMP_FAILS)
def test_omp_fail_kinds_end_with_an_error_message_and_agent_end(
    fakes: Fakes, kind: str, status: int, needle: str
) -> None:
    proc = fakes.run("omp", OMP_JSON, steps=[{"op": "fail", "kind": kind}])
    assert proc.returncode == 1
    events = parse_ndjson(proc.stdout)
    assert events[-1]["type"] == "agent_end" and events[-1]["isTerminal"] is True
    failed = events[-1]["messages"][-1]
    # key set of a real stopReason "error" message captured from omp 18.12.6
    assert set(failed) == {
        "role",
        "content",
        "api",
        "provider",
        "model",
        "usage",
        "stopReason",
        "timestamp",
        "errorStatus",
        "errorId",
        "errorMessage",
    }
    assert failed["stopReason"] == "error" and failed["content"] == []
    assert failed["errorStatus"] == status and needle in failed["errorMessage"]
    assert assistant_ends(events)[-1]["stopReason"] == "error"
    assert signature(events)[-4:] == ["message_start", "message_end", "turn_end", "agent_end"]


def test_omp_fail_message_override_and_text_mode(fakes: Fakes) -> None:
    steps = [{"op": "fail", "kind": "credit", "message": "400 custom (type=x)"}]
    proc = fakes.run("omp", OMP_JSON, steps=steps)
    assert assistant_ends(parse_ndjson(proc.stdout))[-1]["errorMessage"] == "400 custom (type=x)"
    proc = fakes.run("omp", ["-p"], steps=steps)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr == "Working...\n400 custom (type=x)\n"


def test_omp_final_report_is_a_fenced_agent_result_block(fakes: Fakes) -> None:
    report = {"status": "completed", "summary": "did it", "files_changed": ["docs/a.md"]}
    expected = "All done.\n\n```json agent-result\n" + json.dumps(report) + "\n```"
    steps = [{"op": "final", "text": "All done.", "report": report}]
    proc = fakes.run("omp", OMP_JSON, steps=steps)
    events = parse_ndjson(proc.stdout)
    assert final_text(events) == expected
    deltas = [
        e["assistantMessageEvent"]["delta"]
        for e in events
        if e["type"] == "message_update" and e["assistantMessageEvent"]["type"] == "text_delta"
    ]
    assert "".join(deltas) == expected
    match = re.search(r"```json agent-result\n(.*?)\n```", final_text(events), re.S)
    assert match and json.loads(match.group(1)) == report
    assert fakes.run("omp", ["-p"], steps=steps).stdout == expected + "\n"


def test_omp_notice_events(fakes: Fakes) -> None:
    steps = [
        {"op": "notice", "message": "xd://: mounted tools"},
        {"op": "cost", "usd": 0.01},
        {"op": "notice", "message": "late notice", "level": "warn", "source": "hooks"},
        {"op": "final", "text": "ok"},
    ]
    proc = fakes.run("omp", OMP_JSON, steps=steps)
    events = parse_ndjson(proc.stdout)
    kinds = signature(events)
    assert kinds[:3] == ["session", "notice", "thinking_level_changed"]
    first, late = [e for e in events if e["type"] == "notice"]
    assert first == {
        "type": "notice",
        "level": "info",
        "message": "xd://: mounted tools",
        "source": "xdev",
    }
    assert late["level"] == "warn" and late["source"] == "hooks"
    assert kinds.index("notice", 3) > kinds.index("agent_start")


def test_omp_stdout_stays_free_of_events_until_a_step_produces_one(fakes: Fakes) -> None:
    proc = fakes.run("omp", OMP_JSON, steps=[{"op": "exit", "code": 0}])
    assert (proc.returncode, proc.stdout) == (0, "")
    steps = [{"op": "garbage_stdout", "text": "not json at all"}, {"op": "final", "text": "ok"}]
    proc = fakes.run("omp", OMP_JSON, steps=steps)
    lines = proc.stdout.splitlines()
    assert lines[0] == "not json at all"
    assert [json.loads(line)["type"] for line in lines[1:]][0] == "session"
    proc = fakes.run("omp", OMP_JSON, steps=[{"op": "fail", "kind": "crash"}])
    assert (proc.returncode, proc.stdout) == (3, "")


# ----------------------------------------------------------------------------------------------
# omp: --max-time, --cwd, @files, empty prompts, TTY handling
# ----------------------------------------------------------------------------------------------


def test_omp_hang_with_max_time_aborts_like_the_real_deadline(fakes: Fakes) -> None:
    started = time.monotonic()
    proc = fakes.run("omp", [*OMP_JSON, "--max-time", "1"], steps=[{"op": "hang"}], timeout=20)
    elapsed = time.monotonic() - started
    assert proc.returncode == 1
    assert 0.9 <= elapsed < 10
    events = parse_ndjson(proc.stdout)
    assert events[-1]["type"] == "agent_end"
    aborted = events[-1]["messages"][-1]
    # key set of the aborted message captured from the real omp --max-time deadline
    assert set(aborted) == {
        "role",
        "content",
        "api",
        "provider",
        "model",
        "usage",
        "stopReason",
        "errorMessage",
        "timestamp",
    }
    assert aborted["stopReason"] == "aborted" and aborted["errorMessage"] == "Deadline exceeded"
    assert signature(events)[-4:] == ["message_start", "message_end", "turn_end", "agent_end"]


def test_omp_text_mode_deadline_prints_the_error_on_stderr(fakes: Fakes) -> None:
    proc = fakes.run("omp", ["-p", "--max-time", "1s"], steps=[{"op": "hang"}], timeout=20)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr == "Working...\nDeadline exceeded\n"


def test_omp_deadline_cuts_a_long_sleep_short(fakes: Fakes) -> None:
    started = time.monotonic()
    steps = [{"op": "sleep", "seconds": 30}, {"op": "final", "text": "never"}]
    proc = fakes.run("omp", [*OMP_JSON, "--max-time", "1"], steps=steps, timeout=20)
    assert proc.returncode == 1 and time.monotonic() - started < 10
    assert assistant_ends(parse_ndjson(proc.stdout))[-1]["stopReason"] == "aborted"


def test_omp_a_run_that_finishes_before_the_deadline_is_unaffected(fakes: Fakes) -> None:
    proc = fakes.run("omp", ["-p", "--max-time", "1h"], steps=[{"op": "final", "text": "fine"}])
    assert (proc.returncode, proc.stdout) == (0, "fine\n")


@pytest.mark.parametrize("value", ["600", "5s", "10m", "1h", "1.5s", "1m30s", "250ms"])
def test_omp_max_time_accepts_seconds_and_durations(fakes: Fakes, value: str) -> None:
    assert fakes.run("omp", ["-p", "--max-time", value]).returncode == 0


@pytest.mark.parametrize("value", ["abc", "0", "-5", "10x", "", "1h-", "m"])
def test_omp_max_time_rejects_everything_else(fakes: Fakes, value: str) -> None:
    proc = fakes.run("omp", ["-p", "--max-time", value])
    assert proc.returncode == 2
    assert proc.stderr == (
        f'Error: Invalid --max-time value: "{value}". Expected a positive number of seconds '
        'or duration like "5s", "10m", "1h".\nRun `omp --help` for available flags.\n'
    )


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--mode", "bogus", "text, json, rpc, rpc-ui, acp"),
        ("--thinking", "bogus", "off, minimal, low, medium, high, xhigh, max, auto"),
        ("--approval-mode", "bogus", "always-ask, always-ask-cached, write, yolo"),
    ],
)
def test_omp_enumerated_values_are_validated(
    fakes: Fakes, flag: str, value: str, expected: str
) -> None:
    proc = fakes.run("omp", ["-p", flag, value])
    assert proc.returncode == 2
    assert proc.stderr == f'Error: Invalid {flag} value: "{value}". Expected one of: {expected}.\n'


@pytest.mark.parametrize("mode", ["rpc", "rpc-ui", "acp"])
def test_omp_non_print_modes_are_not_emulated(fakes: Fakes, mode: str) -> None:
    proc = fakes.run("omp", ["-p", "--mode", mode])
    assert proc.returncode == 2 and "not emulated" in proc.stderr


def test_omp_cwd_flag_changes_the_working_directory(fakes: Fakes) -> None:
    target = fakes.root / "elsewhere"
    target.mkdir()
    steps = [{"op": "write_file", "path": "marker.txt", "content": "here"}]
    proc = fakes.run("omp", [*OMP_JSON, "--cwd", str(target)], steps=steps)
    assert proc.returncode == 0, proc.stderr
    assert (target / "marker.txt").read_text(encoding="utf-8") == "here"
    assert not (fakes.work / "marker.txt").exists()
    assert os.path.realpath(parse_ndjson(proc.stdout)[0]["cwd"]) == os.path.realpath(target)
    assert os.path.realpath(fakes.last_invocation()["cwd"]) == os.path.realpath(target)


def test_omp_cwd_flag_failure_matches_the_real_message(fakes: Fakes) -> None:
    proc = fakes.run("omp", ["-p", "--cwd", "/nonexistent-dir-for-test"])
    assert proc.returncode == 1
    assert proc.stderr.startswith(
        "Error: Cannot change working directory to /nonexistent-dir-for-test: "
        "ENOENT: no such file or directory, chdir '"
    )
    assert fakes.last_invocation()["cwd"] == "/nonexistent-dir-for-test"


def test_omp_trusted_extension_is_validated_like_the_real_cli(fakes: Fakes) -> None:
    hint = "Run `omp --help` for available flags.\n"
    proc = fakes.run("omp", ["-p", "--trusted-extension", "relative.ts"])
    assert proc.returncode == 2
    assert (
        proc.stderr == f"Error: --trusted-extension requires an absolute path: relative.ts\n{hint}"
    )
    proc = fakes.run("omp", ["-p", "--trusted-extension", "--print"])
    assert proc.returncode == 2
    assert proc.stderr == f"Error: --trusted-extension requires a non-empty, non-flag value\n{hint}"
    proc = fakes.run("omp", ["-p", "x", "--trusted-extension"])
    assert proc.returncode == 2 and "non-empty, non-flag value" in proc.stderr
    missing = fakes.root / "missing-guard.ts"
    proc = fakes.run("omp", ["-p", "--trusted-extension", str(missing)])
    assert proc.returncode == 1
    assert proc.stderr == f"error: Trusted extension must be an existing module file: {missing}\n"
    present = fakes.root / "guard.ts"
    present.write_text("export default function () {}\n", encoding="utf-8")
    assert fakes.run("omp", ["-p", "--trusted-extension", str(present)]).returncode == 0


def test_omp_at_prefixed_positionals_are_read_as_files(fakes: Fakes) -> None:
    """A positional prompt that starts with '@' is a file reference in the real CLI."""
    proc = fakes.run("omp", ["-p", "@/nonexistent-prompt-file.md"])
    assert proc.returncode == 1
    assert proc.stderr == "Error: File not found: /nonexistent-prompt-file.md\n"
    (fakes.work / "task.md").write_text("task from a file", encoding="utf-8")
    proc = fakes.run("omp", ["-p", "@task.md", "and a question"])
    assert proc.returncode == 0, proc.stderr
    assert fakes.last_invocation()["prompt"] == "task from a file and a question"


def test_omp_without_any_prompt_does_nothing_like_the_real_cli(fakes: Fakes) -> None:
    marker = {"op": "write_file", "path": "ran.txt", "content": "1"}
    proc = fakes.run("omp", OMP_JSON, stdin="", steps=[marker])
    assert proc.returncode == 0
    events = parse_ndjson(proc.stdout)
    assert [e["type"] for e in events] == ["session"]
    assert "no prompt received" in proc.stderr
    assert not (fakes.work / "ran.txt").exists()
    proc = fakes.run("omp", ["-p"], stdin="")
    assert (proc.returncode, proc.stdout) == (0, "")
    proc = fakes.run("omp", ["--mode", "json"], stdin="")
    assert proc.returncode == 2
    assert proc.stderr == (
        "Error: interactive mode requires a terminal, but stdin is not a TTY.\n"
        'Pass a prompt (`omp -p "…"`), pipe one on stdin, or use `--mode rpc`.\n'
    )


def test_omp_piped_stdin_implies_print_mode(fakes: Fakes) -> None:
    proc = fakes.run("omp", ["--mode", "json"], stdin="a prompt on stdin")
    assert proc.returncode == 0, proc.stderr
    assert final_text(parse_ndjson(proc.stdout)) == "ok"
    assert fakes.last_invocation()["parsed"]["flags"] == {"--mode": "json"}


@pytest.mark.parametrize("kind", KINDS)
def test_tty_stdin_is_reported_and_never_read(fakes: Fakes, kind: str) -> None:
    master, slave = pty.openpty()
    try:
        proc = subprocess.run(
            fakes.argv(kind, ["--verbose"] if kind == "claude" else []),
            stdin=slave,
            capture_output=True,
            text=True,
            env=fakes.env(),
            cwd=fakes.work,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 2
        assert "interactive mode is not emulated" in proc.stderr
        assert fakes.last_invocation()["stdin_was_tty"] is True
        proc = subprocess.run(
            fakes.argv(kind, ["-p", "typed prompt"]),
            stdin=slave,
            capture_output=True,
            text=True,
            env=fakes.env(),
            cwd=fakes.work,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert fakes.last_invocation()["stdin_was_tty"] is True
        assert fakes.last_invocation()["prompt"] == "typed prompt"
    finally:
        os.close(master)
        os.close(slave)


# ----------------------------------------------------------------------------------------------
# invocation log
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_invocation_log_records_argv_cwd_env_prompt_and_tty(fakes: Fakes, kind: str) -> None:
    proc = fakes.run(
        kind,
        ["-p", "--model", "some-model"],
        steps=[{"op": "final", "text": "ok"}],
        stdin="hello there",
        env={"VISIBLE_VAR": "visible-value", "SECRET_TOKEN": "s3cr3t-value"},
        log_env=["VISIBLE_VAR", "ABSENT_VAR"],
    )
    assert proc.returncode == 0, proc.stderr
    entries = read_invocations(fakes.log)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["argv"] == ["-p", "--model", "some-model"]
    assert os.path.realpath(entry["cwd"]) == os.path.realpath(fakes.work)
    assert entry["prompt"] == "hello there"
    assert entry["stdin_was_tty"] is False
    assert entry["env"] == {"VISIBLE_VAR": "visible-value"}
    assert entry["env_keys"] == sorted(entry["env_keys"])
    assert {"PATH", "SECRET_TOKEN", "VISIBLE_VAR", "FAKE_AGENT_LOG"} <= set(entry["env_keys"])
    assert "s3cr3t-value" not in fakes.log.read_text(encoding="utf-8")
    assert entry["parsed"] == {
        "flags": {"--print": True, "--model": "some-model"},
        "positional": [],
    }


@pytest.mark.parametrize("kind", KINDS)
def test_invocation_log_appends_one_line_per_run(fakes: Fakes, kind: str) -> None:
    for index in range(3):
        fakes.run(kind, ["-p", f"prompt {index}"])
    entries = read_invocations(fakes.log)
    assert [entry["prompt"] for entry in entries] == ["prompt 0", "prompt 1", "prompt 2"]
    assert len(fakes.log.read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("args", "steps", "code"),
    [
        (["-p", "--bogus"], None, 2),
        (["-p"], [{"op": "fail", "kind": "crash"}], 3),
        (["-p"], [{"op": "no-such-op"}], 99),
        (["--version"], None, 0),
        (["-p"], [{"op": "exit", "code": 5}], 5),
    ],
)
def test_the_invocation_is_logged_even_when_the_run_fails(
    fakes: Fakes, kind: str, args: list[str], steps: list[dict[str, Any]] | None, code: int
) -> None:
    proc = fakes.run(kind, args, steps=steps, stdin="logged prompt")
    assert proc.returncode == code
    entry = fakes.last_invocation()
    assert entry["argv"] == args
    if args == ["-p"]:
        assert entry["prompt"] == "logged prompt"


@pytest.mark.parametrize("kind", KINDS)
def test_without_a_log_variable_nothing_is_written(fakes: Fakes, kind: str) -> None:
    env = fakes.env()
    env.pop("FAKE_AGENT_LOG")
    proc = subprocess.run(
        fakes.argv(kind, ["-p"]),
        input="x",
        capture_output=True,
        text=True,
        env=env,
        cwd=fakes.work,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert not fakes.log.exists()


# ----------------------------------------------------------------------------------------------
# scenario ops common to both CLIs
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_echo_prompt_to_writes_the_received_prompt(fakes: Fakes, kind: str) -> None:
    target = fakes.root / "echo" / "prompt.txt"
    steps = [{"op": "echo_prompt_to", "path": str(target)}]
    proc = fakes.run(kind, ["-p"], steps=steps, stdin="line one\nline two")
    assert proc.returncode == 0, proc.stderr
    assert target.read_text(encoding="utf-8") == "line one\nline two"
    proc = fakes.run(
        kind, ["-p"], steps=[{"op": "echo_prompt_to", "path": "rel/echo.txt"}], stdin="again"
    )
    assert (fakes.work / "rel" / "echo.txt").read_text(encoding="utf-8") == "again"


@pytest.mark.parametrize("kind", KINDS)
def test_filesystem_ops(fakes: Fakes, kind: str) -> None:
    outside = fakes.root / "outside" / "escaped.txt"
    steps = [
        {"op": "write_file", "path": "docs/new/a.md", "content": "one\n"},
        {"op": "append_file", "path": "docs/new/a.md", "content": "two\n"},
        {"op": "append_file", "path": "fresh.txt", "content": "created by append"},
        {"op": "write_file", "path": "doomed.txt", "content": "x"},
        {"op": "write_file", "path": "replaced.txt", "content": "old content that is longer"},
        {"op": "write_file", "path": "replaced.txt", "content": "new"},
        {"op": "delete_file", "path": "doomed.txt"},
        {"op": "delete_file", "path": "never-existed.txt"},
        {"op": "write_file", "path": "tool.sh", "content": "#!/bin/sh\n"},
        {"op": "chmod_exec", "path": "tool.sh"},
        {"op": "symlink", "path": "links/pointer", "target": "../docs/new/a.md"},
        {"op": "symlink", "path": "links/pointer", "target": "../fresh.txt"},
        {"op": "write_file", "path": str(outside), "content": "escaped", "abs": True},
        {"op": "final", "text": "ok"},
    ]
    proc = fakes.run(kind, ["-p"], steps=steps)
    assert proc.returncode == 0, proc.stderr
    work = fakes.work
    assert (work / "docs/new/a.md").read_text(encoding="utf-8") == "one\ntwo\n"
    assert (work / "fresh.txt").read_text(encoding="utf-8") == "created by append"
    assert not (work / "doomed.txt").exists()
    assert os.access(work / "tool.sh", os.X_OK)
    assert os.readlink(work / "links/pointer") == "../fresh.txt"
    assert (work / "links/pointer").read_text(encoding="utf-8") == "created by append"
    assert outside.read_text(encoding="utf-8") == "escaped"
    assert (work / "replaced.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.parametrize("kind", KINDS)
def test_a_symlinked_directory_lets_a_misbehaving_agent_write_outside(
    fakes: Fakes, kind: str
) -> None:
    outside = fakes.root / "outside-dir"
    outside.mkdir()
    steps = [
        {"op": "symlink", "path": "docs/link", "target": str(outside)},
        {"op": "write_file", "path": "docs/link/escape.md", "content": "gotcha"},
    ]
    proc = fakes.run(kind, ["-p"], steps=steps)
    assert proc.returncode == 0, proc.stderr
    assert (outside / "escape.md").read_text(encoding="utf-8") == "gotcha"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("op", ["write_file", "append_file", "delete_file", "chmod_exec"])
@pytest.mark.parametrize("escape", ["absolute", "dotdot"])
def test_paths_outside_the_cwd_need_abs_true(fakes: Fakes, kind: str, op: str, escape: str) -> None:
    victim = fakes.root / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    path = str(victim) if escape == "absolute" else "../victim.txt"
    steps = [{"op": op, "path": path, "content": "overwritten"}]
    proc = fakes.run(kind, ["-p"], steps=steps)
    assert proc.returncode == 99
    assert '"abs": true' in proc.stderr
    assert victim.read_text(encoding="utf-8") == "original"
    assert not victim.stat().st_mode & 0o111
    steps[0]["abs"] = True
    if op in ("write_file", "append_file"):
        assert fakes.run(kind, ["-p"], steps=steps).returncode == 0
        assert "overwritten" in victim.read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", KINDS)
def test_exit_op_prints_raw_texts_and_returns_the_code(fakes: Fakes, kind: str) -> None:
    steps = [{"op": "exit", "code": 7, "stdout": "raw out", "stderr": "boom\n"}]
    proc = fakes.run(kind, ["-p"], steps=steps)
    assert proc.returncode == 7
    assert proc.stdout == "raw out"
    assert "boom\n" in proc.stderr
    proc = fakes.run(kind, ["-p"], steps=[{"op": "exit"}])
    assert (proc.returncode, proc.stdout) == (1, "")


@pytest.mark.parametrize("kind", KINDS)
def test_garbage_stdout_op_prints_non_json_lines(fakes: Fakes, kind: str) -> None:
    steps = [{"op": "garbage_stdout", "text": "oops, not json"}, {"op": "final", "text": "ok"}]
    args = ["-p", "--output-format", "json"] if kind == "claude" else OMP_JSON
    proc = fakes.run(kind, args, steps=steps)
    assert proc.stdout.splitlines()[0] == "oops, not json"
    with pytest.raises(ValueError, match="Extra data|Expecting value"):
        json.loads(proc.stdout)


@pytest.mark.parametrize("kind", KINDS)
def test_crash_prints_a_python_traceback_and_no_json(fakes: Fakes, kind: str) -> None:
    args = ["-p", "--output-format", "json"] if kind == "claude" else OMP_JSON
    steps = [{"op": "fail", "kind": "crash", "message": "kaboom"}]
    proc = fakes.run(kind, args, steps=steps)
    assert proc.returncode == 3
    assert proc.stdout == ""
    assert "Traceback (most recent call last):" in proc.stderr
    assert proc.stderr.rstrip().endswith("RuntimeError: kaboom")


@pytest.mark.parametrize("kind", KINDS)
def test_sleep_op_delays_the_run(fakes: Fakes, kind: str) -> None:
    started = time.monotonic()
    proc = fakes.run(kind, ["-p"], steps=[{"op": "sleep", "seconds": 0.6}])
    assert proc.returncode == 0
    assert time.monotonic() - started >= 0.6


@pytest.mark.parametrize("kind", KINDS)
def test_final_report_block_format(fakes: Fakes, kind: str) -> None:
    report = {"status": "completed", "summary": "s", "files_changed": ["a.md"]}
    block = "```json agent-result\n" + json.dumps(report) + "\n```"
    steps = [{"op": "final", "text": "Intro line.\n", "report": report}]
    proc = fakes.run(kind, ["-p"], steps=steps)
    assert proc.stdout == f"Intro line.\n\n{block}\n"
    steps = [{"op": "final", "text": "", "report": report}]
    assert fakes.run(kind, ["-p"], steps=steps).stdout == block + "\n"


@pytest.mark.parametrize("kind", KINDS)
def test_docs_edit_steps_run_end_to_end(fakes: Fakes, kind: str) -> None:
    args = ["-p", "--output-format", "json"] if kind == "claude" else OMP_JSON
    proc = fakes.run(kind, args, steps=docs_edit_steps())
    assert proc.returncode == 0, proc.stderr
    assert (fakes.work / "docs/agent-note.md").read_text(
        encoding="utf-8"
    ) == "Hello from the agent\n"
    if kind == "claude":
        result = json.loads(proc.stdout)
        assert result["total_cost_usd"] == pytest.approx(0.02)
        text = result["result"]
    else:
        events = parse_ndjson(proc.stdout)
        assert sum(m["usage"]["cost"]["total"] for m in assistant_ends(events)) == pytest.approx(
            0.02
        )
        text = final_text(events)
    report = json.loads(re.search(r"```json agent-result\n(.*?)\n```", text, re.S).group(1))
    assert report == {
        "status": "completed",
        "summary": "Created docs/agent-note.md",
        "files_changed": ["docs/agent-note.md"],
    }


# ----------------------------------------------------------------------------------------------
# harness errors
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "scenario",
    [
        {"steps": [{"op": "write_file", "path": "ran.txt", "content": "1"}, {"op": "bogus"}]},
        {"steps": [{"op": "write_file", "path": "ran.txt", "content": "1"}, {"op": "symlink"}]},
        {"steps": [{"op": "write_file", "path": "ran.txt", "content": "1"}, {"op": "cost"}]},
        {"steps": [{"op": "write_file", "path": "ran.txt", "content": "1"}, "not-an-object"]},
        {"steps": [{"op": "fail", "kind": "mystery"}]},
        {"steps": "nope"},
        ["not", "an", "object"],
    ],
)
def test_bad_scenarios_are_harness_errors_that_execute_nothing(
    fakes: Fakes, kind: str, scenario: Any
) -> None:
    fakes.scenario.write_text(json.dumps(scenario), encoding="utf-8")
    proc = fakes.run(kind, ["-p"])
    assert proc.returncode == 99
    assert BAD_SCENARIO_TEXT in proc.stderr
    assert not (fakes.work / "ran.txt").exists(), "a harness error must not execute any step"
    assert fakes.last_invocation()["prompt"] == "the prompt\n"


@pytest.mark.parametrize("kind", KINDS)
def test_unreadable_scenarios_are_harness_errors(fakes: Fakes, kind: str) -> None:
    fakes.scenario.write_text("{not json", encoding="utf-8")
    assert fakes.run(kind, ["-p"]).returncode == 99
    missing = fakes.root / "missing.json"
    proc = fakes.run(kind, ["-p"], env={"FAKE_AGENT_SCENARIO": str(missing)})
    assert proc.returncode == 99
    assert str(missing) in proc.stderr


# ----------------------------------------------------------------------------------------------
# processes and signals
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_spawn_child_writes_the_pid_of_a_process_group_member(
    fakes: Fakes, kind: str, tmp_path: Path
) -> None:
    pid_file = tmp_path / "pids" / "child.pid"
    steps = [{"op": "spawn_child", "pid_file": str(pid_file)}, {"op": "hang"}]
    child = None
    try:
        with running(fakes, kind, ["-p"], steps=steps) as proc:
            wait_for_file(pid_file)
            child = int(pid_file.read_text(encoding="utf-8"))
            assert child != proc.pid and pid_is_alive(child)
            assert os.getpgid(child) == os.getpgid(proc.pid) == proc.pid
            assert Path(f"/proc/{child}/cmdline").exists()
            os.killpg(proc.pid, signal.SIGKILL)
            assert proc.wait(timeout=10) == -signal.SIGKILL
            deadline = time.monotonic() + 10
            while pid_is_alive(child) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not pid_is_alive(child), "killing the process group must take the child too"
    finally:
        if child is not None and pid_is_alive(child):
            os.kill(child, signal.SIGKILL)


@pytest.mark.parametrize("kind", KINDS)
def test_spawn_child_does_not_hold_the_stdout_pipe_open(
    fakes: Fakes, kind: str, tmp_path: Path
) -> None:
    pid_file = tmp_path / "child.pid"
    steps = [{"op": "spawn_child", "pid_file": str(pid_file)}, {"op": "final", "text": "ok"}]
    started = time.monotonic()
    try:
        proc = fakes.run(kind, ["-p"], steps=steps, timeout=20)
        assert proc.returncode == 0
        assert time.monotonic() - started < 15, "a leaked child kept the stdout pipe open"
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)


@pytest.mark.parametrize("kind", KINDS)
def test_sigterm_terminates_by_default(fakes: Fakes, kind: str) -> None:
    steps = [{"op": "write_file", "path": "ready.txt", "content": "1"}, {"op": "hang"}]
    with running(fakes, kind, ["-p"], steps=steps) as proc:
        wait_for_file(fakes.work / "ready.txt")
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == -signal.SIGTERM


@pytest.mark.parametrize("kind", KINDS)
def test_ignore_sigterm_forces_a_sigkill_escalation(fakes: Fakes, kind: str) -> None:
    steps = [{"op": "write_file", "path": "ready.txt", "content": "1"}, {"op": "hang"}]
    with running(fakes, kind, ["-p"], steps=steps, ignore_sigterm=True) as proc:
        wait_for_file(fakes.work / "ready.txt")
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.6)
        assert proc.poll() is None, "SIGTERM must be ignored"
        proc.send_signal(signal.SIGKILL)
        assert proc.wait(timeout=10) == -signal.SIGKILL


def test_hang_without_a_deadline_stays_alive_until_killed(fakes: Fakes) -> None:
    with running(fakes, "claude", ["-p"], steps=[{"op": "hang"}]) as proc:
        time.sleep(0.8)
        assert proc.poll() is None
        proc.kill()
        assert proc.wait(timeout=10) == -signal.SIGKILL
