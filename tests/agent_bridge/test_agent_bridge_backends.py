"""Backend contract tests: ``omp`` and ``claude`` behind the one ``AgentBackend`` interface.

Shared behaviour is parametrised over both backends (the ``backend_name`` fixture of the package
conftest) and exercised end to end through the real backend classes, the real subprocess runner
and the executable emulators of the two CLIs (``bridgefakes``).  Backend specific behaviour
(argv, overlay and guard policy for ``omp``; ``load_messages`` and result subtypes for
``claude``) has its own sections, followed by ``extract_report``, ``classify_error_text`` and
the ``build_backend(s)`` / ``resolve_binary`` / ``version`` helpers.

Streams that the fakes cannot produce (a final answer that crosses the budget, real captures)
are replayed by a tiny canned-output CLI written into the test's temporary directory.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from bridgefakes.agent_fixtures import (
    docs_edit_steps,
    install_fake_cli,
    pid_is_alive,
    read_invocations,
    write_scenario,
)

import tools.agent_bridge as bridge_package
from tools.agent_bridge.agents import (
    BACKENDS,
    AgentRequest,
    AgentResult,
    AgentStatus,
    ClaudeBackend,
    FailureKind,
    OmpBackend,
    build_backend,
    build_backends,
    extract_report,
)
from tools.agent_bridge.agents import claude as claude_module
from tools.agent_bridge.agents.base import classify_error_text
from tools.agent_bridge.agents.claude import load_messages
from tools.agent_bridge.agents.omp import (
    GUARD_PATH,
    GUARD_PREFIX,
    OmpStreamHandler,
    hermetic_overlay,
)
from tools.agent_bridge.config import AgentConfig, BackendConfig
from tools.agent_bridge.globs import match_any
from tools.agent_bridge.process import ProcessOutcome
from tools.agent_bridge.redact import Redactor

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
LITERAL_SECRET = "hunter2-literal-secret-value"
MARKER = "PROMPT-MARKER-5c1e"
PROMPT = f"Document the bridge.\nUnicode: Mähdrescher 割草机 ✓\n{MARKER}\n"
SESSION = "00000000-0000-4000-8000-000000000000"
FAKE_ENV = ("FAKE_AGENT_SCENARIO", "FAKE_AGENT_LOG")
FINAL = [{"op": "final", "text": "ok"}]

# Everything a parent process might hold that an agent must never see.
SECRET_ENV: dict[str, str] = {
    "GITHUB_TOKEN": TOKEN,
    "GH_TOKEN": TOKEN,
    "GH_ENTERPRISE_TOKEN": TOKEN,
    "GITHUB_PAT": TOKEN,
    "LAWNBERRY_BRIDGE_GITHUB_TOKEN": TOKEN,
    "LAWNBERRY_BRIDGE_ANYTHING": "internal",
    "SSH_AUTH_SOCK": "/tmp/fake-ssh-agent.sock",
    "GIT_ASKPASS": "/usr/bin/fake-askpass",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-value-1234567890",
    "AWS_ACCESS_KEY_ID": "AKIAFAKEFAKEFAKE",
    "OPENAI_API_KEY": "openai-parent-key",
    "ANTHROPIC_API_KEY": "anthropic-parent-key",
    "CLAUDE_ADDITIONAL_ARGS": "--dangerously-skip-permissions",
    "MCP_SERVERS": "evil=http://127.0.0.1:1",
    "OMP_MCP_CONFIG": "/tmp/ambient-mcp.json",
    "PI_CODING_AGENT_DIR": "/tmp/ambient-agent-dir",
}
BENIGN_ENV = {
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
}
SHELL_ARTIFACTS = {"PWD", "OLDPWD", "SHLVL", "_"}
OMP_ENV = {"LAWNBERRY_BRIDGE_GUARD_POLICY", "PI_NO_TITLE", "OMP_MCP_TIMEOUT_MS"}

# Every discovery provider of omp 18.12.6 that could import ambient configuration.
OMP_PROVIDERS = {
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
}

# ----------------------------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------------------------


def make_script(path: Path, *lines: str) -> Path:
    """Write an executable POSIX shell script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + "\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def install_replay_cli(directory: Path, stdout: str, stderr: str, exit_code: int) -> Path:
    """A CLI that drains stdin, prints canned output and exits with ``exit_code``."""
    directory.mkdir(parents=True, exist_ok=True)
    out, err = directory / "replay.stdout", directory / "replay.stderr"
    out.write_text(stdout, encoding="utf-8")
    err.write_text(stderr, encoding="utf-8")
    return make_script(
        directory / "replay-cli",
        "cat > /dev/null",
        f"cat {shlex.quote(str(out))}",
        f"cat {shlex.quote(str(err))} >&2",
        f"exit {int(exit_code)}",
    )


def tool_name(kind: str, tool: str) -> str:
    """The tool name a backend's CLI uses (omp: lower case, Claude Code: capitalised)."""
    return tool if kind == "omp" else tool.capitalize()


def tool_step(
    kind: str,
    tool: str,
    args: Mapping[str, Any] | None = None,
    *,
    blocked: bool = False,
    result: str | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "op": "tool_call",
        "tool": tool_name(kind, tool),
        "args": dict(args or {}),
        "blocked": blocked,
    }
    if result is not None:
        step["result"] = result
    return step


def wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while pid_is_alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def assert_child_gone(pid_file: Path) -> None:
    """The ``spawn_child`` grandchild recorded in ``pid_file`` no longer runs."""
    assert pid_file.exists(), "the fake agent never reached spawn_child (machine too slow?)"
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    assert wait_dead(pid), "a descendant of the agent survived the run"


class Harness:
    """One backend bound to a fake CLI, a scenario file and an invocation log."""

    def __init__(
        self,
        root: Path,
        kind: str,
        *,
        backend: Mapping[str, Any] | None = None,
        agent: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
        redactor: Redactor | None = None,
        max_log_bytes: int = 4_000_000,
        replay: tuple[str, str, int] | None = None,
    ) -> None:
        self.root = root
        self.kind = kind
        root.mkdir(parents=True, exist_ok=True)
        (root / "home").mkdir(exist_ok=True)
        if replay is None:
            self.binary = install_fake_cli(root / "bin", kind)
        else:
            self.binary = install_replay_cli(root / "bin", *replay)
        self.scenario = root / "scenario.json"
        self.log = root / "invocations.jsonl"
        self.cwd = root / "worktree"
        self.cwd.mkdir(exist_ok=True)
        self.run_dir = root / "run"
        self.pid_files: list[Path] = []
        self.environ: dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root / "home"),
            "LANG": "C.UTF-8",
            **SECRET_ENV,
            "FAKE_AGENT_SCENARIO": str(self.scenario),
            "FAKE_AGENT_LOG": str(self.log),
            **(environ or {}),
        }
        settings: dict[str, Any] = {"binary": str(self.binary), **(backend or {})}
        settings["pass_env"] = [*FAKE_ENV, *settings.get("pass_env", [])]
        config = AgentConfig.model_validate({kind: settings, **(agent or {})})
        self.backend = build_backend(
            kind,
            config,
            redactor=redactor,
            environ=self.environ,
            max_log_bytes=max_log_bytes,
        )

    # -- requests and runs ------------------------------------------------------------------

    def request(self, **overrides: Any) -> AgentRequest:
        fields: dict[str, Any] = {
            "prompt": PROMPT,
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
        return AgentRequest(**fields)

    def run(
        self,
        steps: Sequence[Mapping[str, Any]] | None = None,
        *,
        scenario_options: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> AgentResult:
        if steps is not None:
            write_scenario(self.scenario, [dict(s) for s in steps], **(scenario_options or {}))
        return self.backend.run(self.request(**overrides))

    def argv(self, **overrides: Any) -> list[str]:
        """The argv the backend would use (also writes omp's per-run files)."""
        req = self.request(**overrides)
        req.run_dir.mkdir(parents=True, exist_ok=True)
        return self.backend.build_argv(req, str(self.binary))

    # -- observations -------------------------------------------------------------------------

    def invocations(self) -> list[dict[str, Any]]:
        return read_invocations(self.log)

    def invocation(self) -> dict[str, Any]:
        calls = self.invocations()
        assert len(calls) == 1, calls
        return calls[0]

    def pid_file(self, name: str = "child") -> Path:
        path = self.root / f"{name}.pid"
        self.pid_files.append(path)
        return path

    def reap(self) -> None:
        """Safety net: kill any ``spawn_child`` grandchild a failed test left behind."""
        for path in self.pid_files:
            try:
                os.kill(int(path.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


@pytest.fixture
def new_harness(tmp_path: Path) -> Iterator[Callable[..., Harness]]:
    created: list[Harness] = []

    def build(kind: str, **kwargs: Any) -> Harness:
        harness = Harness(tmp_path / f"{kind}-{len(created)}", kind, **kwargs)
        created.append(harness)
        return harness

    yield build
    for harness in created:
        harness.reap()


@pytest.fixture
def h(new_harness: Callable[..., Harness], backend_name: str) -> Harness:
    """A harness for each backend in turn."""
    return new_harness(backend_name)


def budget_steps(kind: str, pid_file: Path) -> list[dict[str, Any]]:
    """Three 0.40 USD model calls with tool use between them, then a fourth that must not run."""
    read = tool_step(kind, "read", {"path": "docs/a.md"})
    return [
        {"op": "spawn_child", "pid_file": str(pid_file)},
        {"op": "cost", "usd": 0.4},
        read,
        {"op": "cost", "usd": 0.4},
        read,
        {"op": "cost", "usd": 0.4},
        {"op": "sleep", "seconds": 30},
        {"op": "cost", "usd": 0.4},
        {"op": "final", "text": "never reached"},
    ]


# ----------------------------------------------------------------------------------------------
# synthetic streams
# ----------------------------------------------------------------------------------------------


def omp_line(event: Mapping[str, Any]) -> str:
    return json.dumps(event, separators=(",", ":")) + "\n"


def omp_session(session_id: str = SESSION) -> str:
    return omp_line(
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": "2026-10-24T10:00:00.000Z",
            "cwd": "/tmp/worktree",
        }
    )


def omp_assistant(
    *,
    text: str = "",
    stop: str = "stop",
    cost: float | None = 0.0,
    response_id: str | None = "msg_1",
    error: str | None = None,
    model: str = "claude-opus-5-5",
    timestamp: int = 1792837028520,
    kind: str = "message_end",
) -> str:
    """One assistant message event (``message_end`` by default) with the real key set."""
    usage: dict[str, Any] = {
        "input": 4,
        "output": 10,
        "cacheRead": 2,
        "cacheWrite": 100,
        "totalTokens": 116,
    }
    if cost is not None:
        usage["cost"] = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": cost}
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}] if text else [],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": model,
        "usage": usage,
        "stopReason": stop,
        "timestamp": timestamp,
    }
    if response_id is not None:
        message["responseId"] = response_id
    if error is not None:
        message["errorMessage"] = error
    return omp_line({"type": kind, "message": message})


def omp_tool_end(text: str, *, error: bool) -> str:
    return omp_line(
        {
            "type": "tool_execution_end",
            "toolCallId": "toolu_1",
            "toolName": "write",
            "result": {"content": [{"type": "text", "text": text}], "details": {}},
            "isError": error,
        }
    )


def omp_agent_end() -> str:
    return omp_line({"type": "agent_end", "messages": [], "isTerminal": True, "yielded": True})


def feed_all(handler: OmpStreamHandler, text: str) -> str | None:
    """Feed every line; the first kill reason (if any)."""
    reason = None
    for line in text.splitlines(keepends=True):
        reason = reason or handler.feed(line)
    return reason


def claude_result(text: str = "ok", **fields: Any) -> dict[str, Any]:
    """A ``result`` message shaped like the real CLI's (``type`` late in the object)."""
    result: dict[str, Any] = {
        "duration_api_ms": 12,
        "stop_reason": "end_turn",
        "session_id": SESSION,
        "total_cost_usd": 0.05,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 7,
            "cache_creation_input_tokens": 1,
            "cache_read_input_tokens": 2,
        },
        "modelUsage": {"claude-opus-5-5": {"costUSD": 0.05}},
        "permission_denials": [],
        "terminal_reason": "completed",
        "is_error": False,
        "num_turns": 2,
        "subtype": "success",
        "result": text,
        "type": "result",
    }
    result.update(fields)
    return result


def success_stdout(kind: str, text: str, *, cost: float = 0.05) -> str:
    """Canned stdout of a clean run that ends with ``text``."""
    if kind == "omp":
        return omp_session() + omp_assistant(text=text, cost=cost) + omp_agent_end()
    return json.dumps(claude_result(text, total_cost_usd=cost)) + "\n"


def parse_omp(
    text: str,
    *,
    returncode: int | None = 0,
    stderr: str = "",
    budget: float = 1.0,
    run_dir: Path | None = None,
    **outcome: Any,
) -> AgentResult:
    """Run recorded omp output through the stream handler and ``OmpBackend.parse``."""
    backend = OmpBackend(BackendConfig(binary="omp"), environ={})
    req = AgentRequest(
        prompt="p",
        cwd=Path("/nonexistent/worktree"),
        run_dir=run_dir or Path("/nonexistent/run"),
        scope="docs-only",
        timeout_s=60,
        budget_usd=budget,
    )
    handler = OmpStreamHandler(budget)
    reason = feed_all(handler, text)
    done = ProcessOutcome(
        returncode=returncode,
        stdout=text,
        stderr=stderr,
        kill_reason=outcome.pop("kill_reason", reason),
        duration_s=1.0,
        **outcome,
    )
    return backend.parse(done, req, handler)


def parse_claude(stdout: str, *, returncode: int | None = 0, stderr: str = "") -> AgentResult:
    backend = ClaudeBackend(BackendConfig(binary="claude"), environ={})
    req = AgentRequest(
        prompt="p",
        cwd=Path("/nonexistent/worktree"),
        run_dir=Path("/nonexistent/run"),
        scope="docs-only",
        timeout_s=60,
        budget_usd=1.0,
    )
    done = ProcessOutcome(returncode=returncode, stdout=stdout, stderr=stderr, duration_s=1.0)
    return backend.parse(done, req, None)


# ==============================================================================================
# shared contract (omp and claude)
# ==============================================================================================


def test_success_with_a_report_fills_the_whole_result(h: Harness) -> None:
    result = h.run(docs_edit_steps(cost=0.02))
    assert result.status is AgentStatus.SUCCEEDED and result.ok
    assert result.failure_kind is None and result.error is None
    assert result.exit_code == 0
    assert result.report is not None
    assert result.report.status == "completed"
    assert result.report.summary == "Created docs/agent-note.md"
    assert result.report.files_changed == ["docs/agent-note.md"]
    assert result.cost_usd == pytest.approx(0.02)
    assert result.budget_usd == 1.0 and not result.over_budget
    assert result.model == "claude-opus-5-5"
    assert result.session_id is not None
    assert str(uuid.UUID(result.session_id)) == result.session_id
    assert 0 < result.duration_s < 30
    assert result.final_text.startswith("Created docs/agent-note.md.")
    assert "agent-result" in result.final_text
    assert result.stop_reason == {"omp": "stop", "claude": "completed"}[h.kind]
    assert (h.cwd / "docs" / "agent-note.md").read_text(
        encoding="utf-8"
    ) == "Hello from the agent\n"


def test_success_without_a_report_block_keeps_the_final_text(h: Harness) -> None:
    result = h.run([{"op": "final", "text": "Everything is done, no block."}])
    assert result.ok and result.report is None
    assert result.final_text == "Everything is done, no block."
    assert result.to_dict()["report"] is None


def test_prompt_travels_on_stdin_only_and_the_command_omits_it(h: Harness) -> None:
    result = h.run(FINAL)
    call = h.invocation()
    assert call["prompt"] == PROMPT
    assert call["stdin_was_tty"] is False
    assert call["parsed"]["positional"] == []
    assert not any(arg.startswith("@") for arg in call["argv"])
    assert not any(MARKER in arg for arg in call["argv"])
    assert result.command[0] == str(h.binary)
    assert result.command[1:] == call["argv"]
    assert MARKER not in json.dumps(result.to_dict()["command"])


def test_a_prompt_larger_than_the_pipe_buffer_arrives_intact(h: Harness) -> None:
    prompt = "lorem ipsum dolor sit amet\n" * 30_000 + MARKER
    result = h.run(FINAL, prompt=prompt)
    assert result.ok
    assert h.invocation()["prompt"] == prompt


def test_logs_are_private_files_holding_the_raw_output(h: Harness) -> None:
    run_dir = h.root / "nested" / "run"
    result = h.run(FINAL, run_dir=run_dir)
    assert result.stdout_log == str(run_dir / f"{h.kind}.stdout.log")
    assert result.stderr_log == str(run_dir / f"{h.kind}.stderr.log")
    for log in (result.stdout_log, result.stderr_log):
        assert stat.S_IMODE(os.stat(log).st_mode) == 0o600
    raw = Path(result.stdout_log).read_text(encoding="utf-8")
    assert ('"type":"agent_end"' if h.kind == "omp" else '"type":"result"') in raw


def test_to_dict_is_json_serialisable_for_every_outcome(
    h: Harness, new_harness: Callable[..., Harness]
) -> None:
    pid_file = h.pid_file()
    missing = new_harness(h.kind, backend={"binary": str(h.root / "no-such-cli")})
    results = [
        h.run(docs_edit_steps()),
        h.run([{"op": "fail", "kind": "auth"}], run_dir=h.root / "run-auth"),
        h.run(budget_steps(h.kind, pid_file), run_dir=h.root / "run-budget", timeout_s=60),
        missing.run(FINAL),
    ]
    assert [r.status for r in results] == [
        AgentStatus.SUCCEEDED,
        AgentStatus.FAILED,
        AgentStatus.BUDGET_EXCEEDED,
        AgentStatus.UNAVAILABLE,
    ]
    for result in results:
        data = result.to_dict()
        assert json.loads(json.dumps(data)) == data
        assert data["status"] == result.status.value
        assert data["ok"] is (result.status is AgentStatus.SUCCEEDED)
        assert data["over_budget"] is result.over_budget
        assert data["failure_kind"] == (result.failure_kind.value if result.failure_kind else None)
    assert results[2].over_budget is True


@pytest.mark.parametrize(
    ("fail", "kind", "phrase"),
    [
        ("auth", FailureKind.AUTH, "authenticate"),
        ("credit", FailureKind.CREDIT, "credit balance"),
        ("rate_limit", FailureKind.RATE_LIMIT, "rate limit"),
        ("api_error", FailureKind.API_ERROR, "500"),
    ],
)
def test_provider_errors_map_to_failure_kinds(
    h: Harness, fail: str, kind: FailureKind, phrase: str
) -> None:
    result = h.run([{"op": "fail", "kind": fail}])
    assert result.status is AgentStatus.FAILED and not result.ok
    assert result.failure_kind is kind
    assert phrase in (result.error or "").lower()
    assert result.exit_code == 1
    assert result.report is None


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Your credit balance is too low", FailureKind.CREDIT),
        ("OAuth session expired, please run /login", FailureKind.AUTH),
        ("429 Too Many Requests", FailureKind.RATE_LIMIT),
        ("The API is overloaded, try again", FailureKind.RATE_LIMIT),
        ("upstream connect error", FailureKind.API_ERROR),
    ],
)
def test_the_failure_kind_follows_the_error_text(
    h: Harness, message: str, kind: FailureKind
) -> None:
    result = h.run([{"op": "fail", "kind": "api_error", "message": message}])
    assert result.status is AgentStatus.FAILED
    assert result.failure_kind is kind
    assert result.error == message


@pytest.mark.parametrize(
    ("steps", "kind", "code", "snippet"),
    [
        pytest.param([{"op": "exit", "code": 0}], FailureKind.NO_RESULT, 0, "", id="silent-exit-0"),
        pytest.param(
            [{"op": "garbage_stdout", "text": "this is not json"}, {"op": "exit", "code": 0}],
            FailureKind.NO_RESULT,
            0,
            "",
            id="garbage-exit-0",
        ),
        pytest.param(
            [{"op": "garbage_stdout", "text": "{broken json"}, {"op": "exit", "code": 0}],
            FailureKind.NO_RESULT,
            0,
            "",
            id="broken-json-exit-0",
        ),
        pytest.param(
            [{"op": "exit", "code": 7, "stderr": "boom: disk on fire\n"}],
            FailureKind.NONZERO_EXIT,
            7,
            "boom: disk on fire",
            id="exit-7",
        ),
        pytest.param(
            [
                {"op": "garbage_stdout", "text": "oops"},
                {"op": "exit", "code": 2, "stderr": "bad\n"},
            ],
            FailureKind.NONZERO_EXIT,
            2,
            "bad",
            id="garbage-exit-2",
        ),
        pytest.param(
            [{"op": "fail", "kind": "crash"}],
            FailureKind.NONZERO_EXIT,
            3,
            "simulated crash",
            id="crash",
        ),
    ],
)
def test_runs_that_end_without_a_result_fail_cleanly(
    h: Harness, steps: list[dict[str, Any]], kind: FailureKind, code: int, snippet: str
) -> None:
    result = h.run(steps)
    assert result.status is AgentStatus.FAILED and not result.ok
    assert result.failure_kind is kind
    assert result.exit_code == code
    assert snippet in (result.error or "")
    assert result.report is None and result.final_text == ""


def test_noise_around_the_result_is_tolerated(h: Harness) -> None:
    steps = [{"op": "garbage_stdout", "text": "WARNING: MCP server x failed to connect"}, *FINAL]
    result = h.run(steps)
    assert result.ok and result.final_text == "ok"


def test_timeout_terminates_the_whole_process_group(h: Harness) -> None:
    pid_file = h.pid_file()
    result = h.run(
        [{"op": "spawn_child", "pid_file": str(pid_file)}, {"op": "hang"}], timeout_s=2.0
    )
    assert result.status is AgentStatus.TIMED_OUT and not result.ok
    assert result.failure_kind is FailureKind.TIMEOUT
    assert "timeout" in (result.error or "")
    assert result.exit_code == -signal.SIGTERM
    assert 1.5 <= result.duration_s < 10
    assert_child_gone(pid_file)


def test_a_sigterm_proof_agent_is_killed_after_the_grace_period(
    new_harness: Callable[..., Harness], backend_name: str
) -> None:
    h = new_harness(backend_name, agent={"kill_grace_seconds": 0.3})
    pid_file = h.pid_file()
    result = h.run(
        [{"op": "spawn_child", "pid_file": str(pid_file)}, {"op": "hang"}],
        scenario_options={"ignore_sigterm": True},
        timeout_s=2.0,
    )
    assert result.status is AgentStatus.TIMED_OUT
    assert result.exit_code == -signal.SIGKILL
    assert result.duration_s < 10
    assert_child_gone(pid_file)


def test_a_cooperative_agent_gets_the_configured_grace_period_to_exit(
    new_harness: Callable[..., Harness], backend_name: str, tmp_path: Path
) -> None:
    slow_exit = make_script(
        tmp_path / "slow-exit-cli",
        "trap 'sleep 0.6; exit 42' TERM",
        "sleep 30 &",
        "wait $!",
    )
    h = new_harness(
        backend_name, backend={"binary": str(slow_exit)}, agent={"kill_grace_seconds": 3}
    )
    result = h.run(timeout_s=1.0)
    assert result.status is AgentStatus.TIMED_OUT
    assert result.exit_code == 42  # it was allowed to finish its clean-up, not killed outright
    assert 1.4 <= result.duration_s < 3


@pytest.mark.parametrize(
    "reason",
    [
        "cancel: the issue was closed",
        "changed: the issue text was edited after approval",
        "interrupted: the bridge is shutting down",
    ],
)
def test_abort_reasons_cancel_the_run_and_kill_the_group(h: Harness, reason: str) -> None:
    pid_file = h.pid_file()
    polls: list[int] = []

    def abort_check() -> str | None:
        polls.append(1)
        return reason if pid_file.exists() else None

    result = h.run(
        [{"op": "spawn_child", "pid_file": str(pid_file)}, {"op": "hang"}], abort_check=abort_check
    )
    assert result.status is AgentStatus.CANCELLED and not result.ok
    assert result.failure_kind is FailureKind.CANCELLED
    assert result.error == reason
    assert result.exit_code == -signal.SIGTERM
    assert result.duration_s < 10 and polls
    assert_child_gone(pid_file)


def test_a_failing_abort_check_does_not_stop_the_run(h: Harness) -> None:
    calls: list[int] = []

    def broken() -> str | None:
        calls.append(1)
        raise RuntimeError("github is down")

    result = h.run([{"op": "sleep", "seconds": 0.4}, *FINAL], abort_check=broken)
    assert result.ok
    assert calls


def test_on_start_reports_the_agent_pid_and_survives_callback_errors(h: Harness) -> None:
    seen: list[tuple[int, int | None]] = []
    result = h.run(FINAL, on_start=lambda pid, ticks: seen.append((pid, ticks)))
    assert result.ok and len(seen) == 1
    pid, ticks = seen[0]
    assert pid > 0
    if sys.platform.startswith("linux"):
        assert isinstance(ticks, int) and ticks > 0

    def explode(pid: int, ticks: int | None) -> None:
        raise RuntimeError("ledger is locked")

    assert h.run(FINAL, run_dir=h.root / "run-2", on_start=explode).ok


@pytest.mark.parametrize("flavour", ["missing", "not-executable", "directory"])
def test_unusable_binaries_are_unavailable_and_never_raise(
    new_harness: Callable[..., Harness], backend_name: str, tmp_path: Path, flavour: str
) -> None:
    target = tmp_path / f"cli-{flavour}"
    if flavour == "not-executable":
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o644)
    elif flavour == "directory":
        target.mkdir()
    h = new_harness(backend_name, backend={"binary": str(target)})
    assert h.backend.resolve_binary() is None
    result = h.run(FINAL)
    assert result.status is AgentStatus.UNAVAILABLE and not result.ok
    assert result.failure_kind is FailureKind.UNAVAILABLE
    assert str(target) in (result.error or "")
    assert result.exit_code is None and result.command == []
    assert result.budget_usd == 1.0
    assert not h.run_dir.exists()
    assert h.invocations() == []


def test_a_binary_that_cannot_be_started_is_unavailable(
    new_harness: Callable[..., Harness], backend_name: str, tmp_path: Path
) -> None:
    broken = tmp_path / "broken-cli"
    broken.write_text("#!/nonexistent/interpreter\n", encoding="utf-8")
    broken.chmod(0o755)
    h = new_harness(backend_name, backend={"binary": str(broken)})
    result = h.run(FINAL)
    assert result.status is AgentStatus.UNAVAILABLE
    assert result.failure_kind is FailureKind.UNAVAILABLE
    assert result.error
    assert result.command[0] == str(broken)


def test_a_missing_sandbox_wrapper_is_unavailable(
    new_harness: Callable[..., Harness], backend_name: str
) -> None:
    h = new_harness(backend_name, agent={"command_prefix": ["/nonexistent/sandbox-wrapper"]})
    result = h.run(FINAL)
    assert result.status is AgentStatus.UNAVAILABLE
    assert "/nonexistent/sandbox-wrapper" in (result.error or "")
    assert h.invocations() == []


def test_child_environment_is_scrubbed(
    new_harness: Callable[..., Harness], backend_name: str
) -> None:
    h = new_harness(
        backend_name,
        backend={"pass_env": ["MY_PASSED_VAR", "MY_ABSENT_VAR"]},
        environ={
            "MY_PASSED_VAR": "passed-value",
            "MY_UNLISTED_VAR": "must-not-pass",
            "HTTPS_PROXY": "http://proxy.invalid:3128",
            "SSL_CERT_FILE": "/etc/ssl/cert.pem",
            "LC_ALL": "C.UTF-8",
        },
    )
    h.run(
        FINAL,
        scenario_options={
            "log_env": [
                "MY_PASSED_VAR",
                "HOME",
                "TERM",
                "NO_COLOR",
                "LAWNBERRY_AGENT_RUN_ID",
                "HTTPS_PROXY",
                *SECRET_ENV,
            ]
        },
    )
    call = h.invocation()
    keys = set(call["env_keys"])
    assert not keys & set(SECRET_ENV)
    assert {"MY_UNLISTED_VAR", "MY_ABSENT_VAR"}.isdisjoint(keys)
    required = {"PATH", "HOME", "LANG", "LC_ALL", "HTTPS_PROXY", "SSL_CERT_FILE", "MY_PASSED_VAR"}
    assert required | {"TERM", "NO_COLOR", "LAWNBERRY_AGENT_RUN_ID"} <= keys
    allowed = BENIGN_ENV | set(FAKE_ENV) | SHELL_ARTIFACTS | {"MY_PASSED_VAR"}
    allowed |= {"TERM", "NO_COLOR", "LAWNBERRY_AGENT_RUN_ID"}
    allowed |= OMP_ENV if h.kind == "omp" else set()
    assert keys <= allowed, sorted(keys - allowed)
    values = call["env"]
    assert values["MY_PASSED_VAR"] == "passed-value"
    assert values["HOME"] == str(h.root / "home")
    assert values["HTTPS_PROXY"] == "http://proxy.invalid:3128"
    assert values["TERM"] == "dumb" and values["NO_COLOR"] == "1"
    assert values["LAWNBERRY_AGENT_RUN_ID"] == "run-1"
    assert set(values) <= {
        "MY_PASSED_VAR",
        "HOME",
        "TERM",
        "NO_COLOR",
        "LAWNBERRY_AGENT_RUN_ID",
        "HTTPS_PROXY",
    }


@pytest.mark.parametrize(
    "name",
    [
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_PAT",
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
        "LAWNBERRY_BRIDGE_GITHUB_TOKEN",
        "LAWNBERRY_BRIDGE_ANYTHING",
    ],
)
def test_forbidden_variables_cannot_be_passed_to_an_agent(
    new_harness: Callable[..., Harness], backend_name: str, name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        new_harness(backend_name, backend={"pass_env": ["MY_PASSED_VAR", name]})


def test_version_probe_uses_the_scrubbed_environment(h: Harness) -> None:
    assert h.backend.version() is not None
    call = h.invocation()
    assert call["argv"] == ["--version"]
    assert not set(call["env_keys"]) & set(SECRET_ENV)


def test_command_prefix_wraps_the_agent_command(
    new_harness: Callable[..., Harness], backend_name: str, tmp_path: Path
) -> None:
    record = tmp_path / "wrapper-args.txt"
    wrapper = make_script(
        tmp_path / "sandbox-wrapper",
        'record="$1"',
        "shift",
        'printf \'%s\\n\' "$@" > "$record"',
        'exec "$@"',
    )
    h = new_harness(backend_name, agent={"command_prefix": [str(wrapper), str(record)]})
    result = h.run(FINAL)
    assert result.ok
    assert record.read_text(encoding="utf-8").splitlines() == result.command
    call = h.invocation()
    assert call["argv"] == result.command[1:]
    assert call["prompt"] == PROMPT
    assert not set(call["env_keys"]) & set(SECRET_ENV)


@pytest.mark.parametrize("secret", [TOKEN, LITERAL_SECRET], ids=["pattern", "literal"])
def test_secrets_in_agent_output_never_reach_results_or_logs(
    new_harness: Callable[..., Harness], backend_name: str, secret: str
) -> None:
    h = new_harness(backend_name, redactor=Redactor([LITERAL_SECRET]))
    steps = [
        tool_step(
            h.kind,
            "write",
            {"path": "docs/a.md", "content": secret},
            blocked=True,
            result=f"{GUARD_PREFIX} refused, saw {secret}",
        ),
        {
            "op": "final",
            "text": f"Done. key={secret} end",
            "report": {"status": "completed", "summary": f"leaked {secret}?", "files_changed": []},
        },
    ]
    result = h.run(steps)
    assert result.ok
    assert secret not in json.dumps(result.to_dict())
    for log in (result.stdout_log, result.stderr_log):
        assert secret not in Path(log).read_text(encoding="utf-8")
    assert "[REDACTED:" in result.final_text
    assert result.report is not None and secret not in result.report.summary


@pytest.mark.parametrize("secret", [TOKEN, LITERAL_SECRET], ids=["pattern", "literal"])
def test_secrets_in_error_output_are_redacted(
    new_harness: Callable[..., Harness], backend_name: str, secret: str
) -> None:
    h = new_harness(backend_name, redactor=Redactor([LITERAL_SECRET]))
    crashed = h.run([{"op": "exit", "code": 4, "stderr": f"fatal: auth with {secret} failed\n"}])
    assert crashed.failure_kind is FailureKind.NONZERO_EXIT
    assert "[REDACTED:" in (crashed.error or "")
    failed = h.run(
        [{"op": "fail", "kind": "auth", "message": f"401 Failed to authenticate with {secret}"}],
        run_dir=h.root / "run-2",
    )
    assert failed.failure_kind is FailureKind.AUTH
    for result in (crashed, failed):
        assert secret not in json.dumps(result.to_dict())
        for log in (result.stdout_log, result.stderr_log):
            assert secret not in Path(log).read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["omp", "claude"])
def test_parse_redacts_secrets_even_when_the_runner_did_not(kind: str) -> None:
    """Defense in depth: unredacted recorded output is still cleaned by ``parse``."""
    text = f"Done: token {TOKEN} end"
    if kind == "omp":
        ok = parse_omp(omp_session() + omp_assistant(text=text) + omp_agent_end())
        error_stream = omp_session() + omp_assistant(stop="error", error=f"401 bad {TOKEN}")
        failed = parse_omp(error_stream + omp_agent_end(), returncode=1)
        crashed = parse_omp("", returncode=4, stderr=f"fatal: auth with {TOKEN} failed\n")
    else:
        ok = parse_claude(json.dumps(claude_result(text)))
        failed = parse_claude(
            json.dumps(claude_result(f"401 bad {TOKEN}", is_error=True)), returncode=1
        )
        crashed = parse_claude("", returncode=4, stderr=f"fatal: auth with {TOKEN} failed\n")
    for result in (ok, failed, crashed):
        assert TOKEN not in json.dumps(result.to_dict())
    assert "[REDACTED:github-token]" in ok.final_text
    assert failed.failure_kind is FailureKind.AUTH
    assert "[REDACTED:github-token]" in (failed.error or "")
    assert crashed.failure_kind is FailureKind.NONZERO_EXIT
    assert "[REDACTED:github-token]" in (crashed.error or "")


def test_blocked_tool_calls_are_counted(h: Harness) -> None:
    steps = [
        tool_step(h.kind, "read", {"path": "docs/a.md"}),
        tool_step(
            h.kind,
            "write",
            {"path": "/etc/passwd", "content": "x"},
            blocked=True,
            result=f"{GUARD_PREFIX} path escapes the task worktree: /etc/passwd",
        ),
        tool_step(
            h.kind,
            "write",
            {"path": ".github/ci.yml", "content": "x"},
            blocked=True,
            result=f"{GUARD_PREFIX} path is protected and needs a human change: .github/ci.yml",
        ),
        {"op": "final", "text": "done"},
    ]
    result = h.run(steps)
    assert result.ok
    assert result.blocked_tool_calls == 2 and len(result.blocked_samples) == 2
    if h.kind == "omp":
        assert result.tool_calls == 3
        assert all(s.startswith(GUARD_PREFIX) for s in result.blocked_samples)
        assert "/etc/passwd" in result.blocked_samples[0]
    else:
        assert result.tool_calls is None
        assert result.blocked_samples == ["Write: denied by permission policy"] * 2


def test_exceeding_the_budget_stops_the_agent_and_reports_the_partial_cost(h: Harness) -> None:
    pid_file = h.pid_file()
    result = h.run(budget_steps(h.kind, pid_file), budget_usd=1.0, timeout_s=60)
    assert result.status is AgentStatus.BUDGET_EXCEEDED and not result.ok
    assert result.failure_kind is FailureKind.BUDGET
    assert result.cost_usd == pytest.approx(1.2)  # the fourth model call never happened
    assert result.budget_usd == 1.0 and result.over_budget
    assert result.duration_s < 20
    assert "1.2" in (result.error or "") or "budget" in (result.error or "").lower()
    assert_child_gone(pid_file)
    if h.kind == "claude":
        assert result.command[result.command.index("--max-budget-usd") + 1] == "1.00"


def test_spending_exactly_the_budget_stops_the_run_without_overrun(h: Harness) -> None:
    steps = [
        {"op": "cost", "usd": 0.5},
        {"op": "cost", "usd": 0.5},
        {"op": "sleep", "seconds": 30},
        {"op": "final", "text": "never reached"},
    ]
    result = h.run(steps, budget_usd=1.0, timeout_s=60)
    assert result.status is AgentStatus.BUDGET_EXCEEDED
    assert result.cost_usd == pytest.approx(1.0)
    assert not result.over_budget
    assert result.duration_s < 20


def test_staying_under_the_budget_succeeds(h: Harness) -> None:
    read = tool_step(h.kind, "read", {"path": "docs/a.md"})
    steps = [{"op": "cost", "usd": 0.4}, read, {"op": "cost", "usd": 0.4}, read, *FINAL]
    result = h.run(steps, budget_usd=1.0)
    assert result.ok
    assert result.cost_usd == pytest.approx(0.8)
    assert not result.over_budget


def test_a_final_answer_that_crosses_the_budget_is_kept(
    new_harness: Callable[..., Harness], backend_name: str
) -> None:
    text = "All done.\n\n```json agent-result\n" + json.dumps(
        {"status": "completed", "summary": "Finished", "files_changed": []}
    )
    stdout = success_stdout(backend_name, text + "\n```", cost=1.5)
    h = new_harness(backend_name, replay=(stdout, "", 0))
    result = h.run(budget_usd=1.0)
    assert result.status is AgentStatus.SUCCEEDED
    assert result.cost_usd == pytest.approx(1.5)
    assert result.over_budget and result.report is not None


def test_long_final_text_keeps_its_tail_and_the_report_is_read_from_the_full_text(
    new_harness: Callable[..., Harness], backend_name: str
) -> None:
    block = '```json agent-result\n{"status": "blocked", "summary": "needs a human"}\n```'
    stdout = success_stdout(backend_name, "x" * 50_000 + "\n\n" + block)
    h = new_harness(backend_name, replay=(stdout, "", 0))
    result = h.run()
    assert result.ok
    assert len(result.final_text) <= 20_000
    assert result.final_text.endswith(block)
    assert result.report is not None and result.report.status == "blocked"


def test_request_model_overrides_the_configured_model(
    new_harness: Callable[..., Harness], backend_name: str
) -> None:
    h = new_harness(backend_name, backend={"model": "cfg-model"})
    first = h.run(FINAL, model="req-model")
    argv = first.command
    assert argv.count("--model") == 1 and argv[argv.index("--model") + 1] == "req-model"
    assert first.model == "req-model"
    second = h.run(FINAL, run_dir=h.root / "run-2")
    assert second.command[second.command.index("--model") + 1] == "cfg-model"
    assert second.model == "cfg-model"
    bare = new_harness(backend_name).run(FINAL)
    assert "--model" not in bare.command


def test_forbidden_flags_never_appear_in_argv(h: Harness) -> None:
    forbidden = {
        "omp": {"--yolo", "--auto-approve", "-e", "--extension", "--hook", "--plan-yolo"},
        "claude": {
            "--dangerously-skip-permissions",
            "--allow-dangerously-skip-permissions",
            "--verbose",
            "--bare",
        },
    }[h.kind]
    for overrides in (
        {},
        {"bash_allowlist": ("pytest -q tests/x",), "scope": "code", "model": "some-model"},
    ):
        argv = h.argv(**overrides)
        assert not forbidden & set(argv), argv
        assert not any(MARKER in arg for arg in argv)


# ==============================================================================================
# omp
# ==============================================================================================


def test_omp_argv_is_exactly_the_verified_invocation(new_harness: Callable[..., Harness]) -> None:
    h = new_harness("omp")
    result = h.run(FINAL, timeout_s=600)
    guard = Path(bridge_package.__file__).resolve().parent / "omp_guard.mjs"
    assert guard.is_file() and Path(GUARD_PATH) == guard
    assert result.ok  # the fake validates every flag and the extension path like the real CLI
    assert result.command == [
        str(h.binary),
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
        "read,grep,glob,edit,write",
        "--approval-mode",
        "write",
        "--config",
        str(h.run_dir / "omp-overlay.yml"),
        "--trusted-extension",
        str(guard),
        "--max-time",
        "595",
        "--cwd",
        str(h.cwd),
    ]
    flags = h.invocation()["parsed"]["flags"]
    assert flags["--approval-mode"] == "write" and flags["--config"] == [
        str(h.run_dir / "omp-overlay.yml")
    ]
    assert h.invocation()["cwd"] == str(h.cwd)


@pytest.mark.parametrize(
    ("timeout_s", "max_time"),
    [(1, "10"), (14.9, "10"), (15, "10"), (16, "11"), (20, "15"), (600, "595"), (1200.7, "1195")],
)
def test_omp_max_time_trails_the_bridge_timeout_by_five_seconds(
    new_harness: Callable[..., Harness], timeout_s: float, max_time: str
) -> None:
    argv = new_harness("omp").argv(timeout_s=timeout_s)
    assert argv[argv.index("--max-time") + 1] == max_time


def test_omp_enables_bash_only_with_an_allowlist(new_harness: Callable[..., Harness]) -> None:
    h = new_harness("omp")
    docs = h.argv()
    code = h.argv(scope="code", bash_allowlist=("pytest -q tests/unit",))
    assert docs[docs.index("--tools") + 1] == "read,grep,glob,edit,write"
    assert code[code.index("--tools") + 1] == "read,grep,glob,edit,write,bash"


@pytest.mark.parametrize(
    "thinking", ["off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"]
)
def test_omp_thinking_level_is_passed_through(
    new_harness: Callable[..., Harness], thinking: str
) -> None:
    h = new_harness("omp", backend={"thinking": thinking})
    result = h.run(FINAL)
    assert result.ok  # the fake rejects levels the real CLI would reject
    assert result.command[result.command.index("--thinking") + 1] == thinking


def test_omp_omits_model_and_thinking_unless_configured(
    new_harness: Callable[..., Harness],
) -> None:
    argv = new_harness("omp").argv()
    assert "--model" not in argv and "--thinking" not in argv


def test_omp_extra_args_come_last(new_harness: Callable[..., Harness]) -> None:
    h = new_harness(
        "omp",
        backend={
            "model": "m",
            "thinking": "low",
            "extra_args": ["--hide-thinking", "--print-thoughts"],
        },
    )
    result = h.run(FINAL)
    assert result.ok
    assert result.command[-6:] == [
        "--model",
        "m",
        "--thinking",
        "low",
        "--hide-thinking",
        "--print-thoughts",
    ]


def test_omp_overlay_is_hermetic(new_harness: Callable[..., Harness]) -> None:
    h = new_harness("omp")
    h.run(FINAL)
    overlay = yaml.safe_load((h.run_dir / "omp-overlay.yml").read_text(encoding="utf-8"))
    assert overlay == hermetic_overlay()
    providers = overlay["disabledProviders"]
    assert OMP_PROVIDERS <= set(providers) and len(providers) == len(set(providers))
    assert overlay["mcp"]["enableProjectConfig"] is False
    assert overlay["tools"]["approvalMode"] == "write"
    assert overlay["retry"]["modelFallback"] is False
    assert overlay["memory"]["backend"] == "off"
    assert overlay["telemetry"]["otlpExportEnabled"] is False
    assert overlay["dev"]["autoqa"] is False
    assert overlay["autolearn"]["enabled"] is False
    assert overlay["skills"]["enabled"] is False


def test_omp_guard_policy_carries_the_request_rules(new_harness: Callable[..., Harness]) -> None:
    h = new_harness("omp")
    h.run(
        FINAL,
        write_allow_sets=(("docs/**", "*.md"), ("docs/**",)),
        write_deny=(".github/**", "docs/private/**"),
        bash_allowlist=("pytest -q tests/x",),
    )
    policy = json.loads((h.run_dir / "omp-guard-policy.json").read_text(encoding="utf-8"))
    assert policy["version"] == 1
    assert Path(policy["root"]).resolve() == h.cwd.resolve()
    assert policy["tools"] == ["read", "grep", "glob", "edit", "write", "bash"]
    assert policy["write_allow_sets"] == [["docs/**", "*.md"], ["docs/**"]]
    assert policy["write_deny"] == [".github/**", "docs/private/**"]
    assert policy["bash_exact"] == ["pytest -q tests/x"]
    assert policy["audit_log"] == str(h.run_dir / "omp-guard-audit.jsonl")
    assert policy["max_write_bytes"] > 0
    for secret_path in (
        ".env",
        "sub/.env.local",
        "keys/server.pem",
        "id_rsa",
        "config/secrets.json",
        "config/ssl/cert.crt",
        ".git/config",
        ".git",
    ):
        assert match_any(secret_path, policy["read_deny"]), secret_path
    assert not match_any("docs/readme.md", policy["read_deny"])
    assert not (h.cwd / "omp-guard-policy.json").exists()  # outside the agent's reach


def test_omp_guard_policy_without_bash_has_no_commands(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp")
    h.run(FINAL)
    policy = json.loads((h.run_dir / "omp-guard-policy.json").read_text(encoding="utf-8"))
    assert policy["tools"] == ["read", "grep", "glob", "edit", "write"]
    assert policy["bash_exact"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_omp_policy_written_by_the_backend_is_enforced_by_the_real_guard(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp")
    h.run(
        FINAL,
        write_allow_sets=(("docs/**", "*.md"),),
        write_deny=(".github/**",),
        bash_allowlist=("pytest -q tests/x",),
    )
    calls = [
        {"toolName": "write", "input": {"path": "docs/new.md", "content": "x"}},
        {"toolName": "write", "input": {"path": ".github/ci.yml", "content": "x"}},
        {"toolName": "write", "input": {"path": "/etc/passwd", "content": "x"}},
        {"toolName": "write", "input": {"path": "src/app.py", "content": "x"}},
        {"toolName": "read", "input": {"path": ".env"}},
        {"toolName": "read", "input": {"path": "docs/readme.md"}},
        {"toolName": "bash", "input": {"command": "pytest -q tests/x"}},
        {"toolName": "bash", "input": {"command": "rm -rf /"}},
        {"toolName": "web_search", "input": {}},
    ]
    script = (
        f"import {{ decide }} from {json.dumps(GUARD_PATH.as_uri())};"
        "import fs from 'node:fs';"
        "const policy = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));"
        "const calls = JSON.parse(process.argv[2]);"
        "console.log(JSON.stringify(calls.map((c) => decide(policy, c).allow)));"
    )
    done = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            script,
            str(h.run_dir / "omp-guard-policy.json"),
            json.dumps(calls),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [True, False, False, False, False, True, True, False, False]


def test_omp_environment_has_the_guard_policy_and_no_ambient_mcp_configuration(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp")
    h.run(FINAL, scenario_options={"log_env": sorted(OMP_ENV)})
    call = h.invocation()
    assert call["env"]["LAWNBERRY_BRIDGE_GUARD_POLICY"] == str(h.run_dir / "omp-guard-policy.json")
    assert call["env"]["PI_NO_TITLE"] == "1"
    ambient = {"MCP_SERVERS", "OMP_MCP_CONFIG", "PI_CODING_AGENT_DIR", "ANTHROPIC_API_KEY"}
    assert ambient.isdisjoint(call["env_keys"])


def test_omp_stops_a_run_whose_stream_handler_breaks(new_harness: Callable[..., Harness]) -> None:
    h = new_harness("omp")
    broken_event = '{"type":"message_end","message":"not an object"}'
    result = h.run([{"op": "garbage_stdout", "text": broken_event}, {"op": "hang"}], timeout_s=60)
    assert result.status is AgentStatus.FAILED
    assert result.failure_kind is FailureKind.HOOK
    assert "stream handler failed" in (result.error or "")
    assert result.duration_s < 20


def test_omp_guard_audit_log_is_preferred_over_the_stream(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp")
    h.run_dir.mkdir(parents=True)
    rows = [
        {"ts": "t", "tool": "write", "allow": False, "reason": "path escapes: /etc/passwd"},
        "this line is not json",
        {"ts": "t", "tool": "edit", "allow": False, "reason": "path is protected: .github/ci.yml"},
        {"ts": "t", "tool": "read", "allow": True, "reason": None},
    ]
    (h.run_dir / "omp-guard-audit.jsonl").write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    blocked = tool_step(
        "omp", "write", {"path": "/etc/passwd"}, blocked=True, result=f"{GUARD_PREFIX} nope"
    )
    result = h.run([blocked, {"op": "final", "text": "done"}])
    assert result.ok
    assert result.blocked_tool_calls == 2
    assert result.blocked_samples == [
        "write: path escapes: /etc/passwd",
        "edit: path is protected: .github/ci.yml",
    ]


def test_omp_stream_blocks_win_when_the_audit_log_saw_fewer(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp")
    h.run_dir.mkdir(parents=True)
    audit = {"ts": "t", "tool": "write", "allow": False, "reason": "outside"}
    (h.run_dir / "omp-guard-audit.jsonl").write_text(json.dumps(audit) + "\n", encoding="utf-8")
    blocked = tool_step("omp", "write", {"path": "x"}, blocked=True, result=f"{GUARD_PREFIX} a")
    result = h.run([blocked, blocked, {"op": "final", "text": "done"}])
    assert result.blocked_tool_calls == 2


def test_omp_tool_errors_that_are_not_guard_blocks_are_not_counted(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp")
    failing = tool_step("omp", "write", {"path": "x"}, blocked=True, result="ENOENT: no such file")
    result = h.run([failing, {"op": "final", "text": "done"}])
    assert result.ok and result.blocked_tool_calls == 0 and result.blocked_samples == []


def test_omp_capture_overflow_is_flagged_and_the_log_stops_at_the_cap(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("omp", max_log_bytes=3000)
    result = h.run([{"op": "final", "text": "y" * 20_000}])
    assert result.truncated is True
    assert result.ok and len(result.final_text) == 20_000  # the stream handler saw every line
    log = Path(result.stdout_log).read_bytes()
    assert len(log) < 3000 + 300
    assert b"capture limit reached" in log


# -- recorded streams ----------------------------------------------------------------------------


def with_unique_response_ids(text: str) -> str:
    """The captured tool-call stream carries ``msg_FIXTURE`` on every message; real ids differ."""
    ids: dict[int, str] = {}
    lines = []
    for line in text.splitlines():
        event = json.loads(line)
        messages = [event["message"]] if "message" in event else []
        messages += event.get("messages", [])
        for message in messages:
            if message.get("role") == "assistant" and "responseId" in message:
                message["responseId"] = ids.setdefault(
                    message["timestamp"], f"msg_fixture_{len(ids) + 1}"
                )
        lines.append(json.dumps(event, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def test_omp_handler_reads_the_real_minimal_capture() -> None:
    handler = OmpStreamHandler(1.0)
    stream = (FIXTURES / "omp_real_minimal.ndjson").read_text(encoding="utf-8")
    assert feed_all(handler, stream) is None
    assert handler.cost_usd == pytest.approx(0.033796)
    assert handler.usage == {
        "input": 4,
        "output": 4,
        "cacheRead": 0,
        "cacheWrite": 6740,
        "totalTokens": 6748,
    }
    assert handler.model == "claude-opus-5-5" and handler.provider == "anthropic"
    assert handler.session_id == SESSION
    assert handler.stop_reason == "stop" and handler.final_text == "pong"
    assert handler.agent_end and handler.terminal
    assert handler.assistant_messages == 1
    assert handler.tool_calls == 0 and handler.blocked_count == 0


def test_omp_handler_reads_the_real_tool_call_capture() -> None:
    handler = OmpStreamHandler(1.0)
    stream = with_unique_response_ids(
        (FIXTURES / "omp_real_tool_calls.ndjson").read_text(encoding="utf-8")
    )
    assert feed_all(handler, stream) is None
    assert handler.assistant_messages == 2
    assert handler.cost_usd == pytest.approx(0.039656 + 0.005482)
    assert handler.usage == {
        "input": 6,
        "output": 378,
        "cacheRead": 6820,
        "cacheWrite": 7238,
        "totalTokens": 14442,
    }
    assert handler.tool_calls == 3 and handler.tool_errors == 1
    assert handler.blocked_count == 0  # omp's own refusal is not the bridge guard
    assert handler.stop_reason == "stop" and handler.final_text.startswith("DONE")
    assert handler.agent_end and handler.terminal


@pytest.mark.parametrize(
    ("fixture", "unique", "cost", "calls", "text"),
    [
        ("omp_real_minimal.ndjson", False, 0.033796, 0, "pong"),
        ("omp_real_tool_calls.ndjson", True, 0.045138, 3, "DONE"),
    ],
)
def test_omp_real_captures_run_end_to_end(
    new_harness: Callable[..., Harness],
    fixture: str,
    unique: bool,
    cost: float,
    calls: int,
    text: str,
) -> None:
    stream = (FIXTURES / fixture).read_text(encoding="utf-8")
    h = new_harness("omp", replay=(with_unique_response_ids(stream) if unique else stream, "", 0))
    result = h.run()
    assert result.status is AgentStatus.SUCCEEDED
    assert result.cost_usd == pytest.approx(cost)
    assert result.tool_calls == calls and result.blocked_tool_calls == 0
    assert result.final_text.startswith(text)
    assert result.model == "claude-opus-5-5" and result.session_id == SESSION
    assert result.stop_reason == "stop" and result.report is None


def test_omp_recorded_deadline_event_is_a_timeout() -> None:
    stream = (
        omp_session()
        + omp_assistant(stop="aborted", cost=0.025051, error="Deadline exceeded")
        + omp_agent_end()
    )
    result = parse_omp(stream, returncode=1)
    assert result.status is AgentStatus.TIMED_OUT
    assert result.failure_kind is FailureKind.TIMEOUT
    assert "--max-time" in (result.error or "")
    assert result.cost_usd == pytest.approx(0.025051)
    assert result.stop_reason == "aborted"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("401 Failed to authenticate: invalid x-api-key", FailureKind.AUTH),
        ("400 Credit balance is too low to access the Anthropic API", FailureKind.CREDIT),
        ("429 rate limit exceeded", FailureKind.RATE_LIMIT),
        ("500 Internal server error", FailureKind.API_ERROR),
    ],
)
def test_omp_error_and_aborted_messages_are_classified(message: str, kind: FailureKind) -> None:
    for stop in ("error", "aborted"):
        stream = omp_session() + omp_assistant(stop=stop, error=message) + omp_agent_end()
        result = parse_omp(stream, returncode=1)
        assert result.status is AgentStatus.FAILED
        assert result.failure_kind is kind
        assert result.error == message


def test_omp_end_states_without_a_clean_stop() -> None:
    clean = omp_session() + omp_assistant(text="done") + omp_agent_end()
    assert parse_omp(clean).status is AgentStatus.SUCCEEDED
    nonzero = parse_omp(clean, returncode=2, stderr="late crash\n")
    assert nonzero.failure_kind is FailureKind.NONZERO_EXIT and nonzero.error == "late crash"
    no_end = parse_omp(omp_session() + omp_assistant(text="done"))
    assert no_end.failure_kind is FailureKind.NO_RESULT
    mid_turn = parse_omp(omp_session() + omp_assistant(stop="toolUse") + omp_agent_end())
    assert mid_turn.failure_kind is FailureKind.NO_RESULT
    assert parse_omp("").failure_kind is FailureKind.NO_RESULT


# -- handler edge cases --------------------------------------------------------------------------


def test_omp_handler_counts_a_duplicated_message_end_once() -> None:
    handler = OmpStreamHandler(10.0)
    line = omp_assistant(text="hi", cost=0.3, response_id="msg_dup")
    for _ in range(3):
        assert handler.feed(line) is None
    assert handler.assistant_messages == 1
    assert handler.cost_usd == pytest.approx(0.3)
    assert handler.usage["output"] == 10
    handler.feed(omp_assistant(text="again", cost=0.3, response_id="msg_other"))
    assert handler.cost_usd == pytest.approx(0.6) and handler.assistant_messages == 2


def test_omp_handler_dedupes_on_timestamp_when_a_provider_sends_no_response_id() -> None:
    handler = OmpStreamHandler(10.0)
    first = omp_assistant(cost=0.1, response_id=None, timestamp=1)
    handler.feed(first)
    handler.feed(first)
    handler.feed(omp_assistant(cost=0.1, response_id=None, timestamp=2))
    assert handler.assistant_messages == 2 and handler.cost_usd == pytest.approx(0.2)


@pytest.mark.parametrize(
    "noise",
    [
        "",
        "   \n",
        "Warning: MCP server graphiti failed to connect\n",
        "[1, 2, 3]\n",
        '"just a string"\n',
        "{not json at all\n",
        '{"type": "message_end", "message": {"role": "assistant"\n',
        "\x1b[31mcolour\x1b[0m\n",
    ],
)
def test_omp_handler_ignores_lines_that_are_not_events(noise: str) -> None:
    handler = OmpStreamHandler(1.0)
    assert handler.feed(noise) is None
    assert handler.feed(omp_assistant(text="real", cost=0.2)) is None
    assert handler.assistant_messages == 1 and handler.cost_usd == pytest.approx(0.2)
    assert handler.final_text == "real"


@pytest.mark.parametrize(
    ("stop", "cuts"),
    [("toolUse", True), ("length", True), ("stop", False), ("error", False), ("aborted", False)],
)
def test_omp_budget_only_cuts_runs_that_want_to_continue(stop: str, cuts: bool) -> None:
    handler = OmpStreamHandler(1.0)
    reason = handler.feed(omp_assistant(stop=stop, cost=1.5))
    assert (reason is not None) is cuts
    if cuts:
        assert reason is not None and reason.startswith("budget")
        assert "1.5000" in reason and "$1.00" in reason


def test_omp_budget_is_cumulative_and_inclusive() -> None:
    handler = OmpStreamHandler(1.0)
    assert handler.feed(omp_assistant(stop="toolUse", cost=0.4, response_id="a")) is None
    assert handler.feed(omp_assistant(stop="toolUse", cost=0.5999, response_id="b")) is None
    assert handler.feed(omp_assistant(stop="toolUse", cost=0.0001, response_id="c")) is not None


def test_omp_handler_keeps_the_last_non_empty_text_and_ignores_bad_costs() -> None:
    handler = OmpStreamHandler(10.0)
    handler.feed(omp_assistant(text="first answer", cost=None, response_id="a"))
    handler.feed(omp_assistant(text="", stop="toolUse", cost=0.2, response_id="b"))
    handler.feed(omp_assistant(text="   ", stop="toolUse", cost=0.0, response_id="c"))
    assert handler.final_text == "first answer"
    assert handler.cost_usd == pytest.approx(0.2)
    handler.feed(
        omp_assistant(text="last", cost=0.0, response_id="d").replace('"total":0.0', '"total":"x"')
    )
    assert handler.final_text == "last" and handler.cost_usd == pytest.approx(0.2)


def test_omp_handler_counts_only_guard_prefixed_errors_as_blocked() -> None:
    handler = OmpStreamHandler(1.0)
    handler.feed(omp_tool_end(f"{GUARD_PREFIX} path escapes the task worktree", error=True))
    handler.feed(omp_tool_end("ENOENT: no such file", error=True))
    handler.feed(omp_tool_end(f"{GUARD_PREFIX} reads are not an error", error=False))
    assert handler.tool_errors == 2 and handler.blocked_count == 1
    assert handler.blocked == [f"{GUARD_PREFIX} path escapes the task worktree"]


def test_omp_handler_bounds_what_it_remembers() -> None:
    handler = OmpStreamHandler(1.0)
    for index in range(30):
        handler.feed(omp_tool_end(f"{GUARD_PREFIX} blocked {index}" + "z" * 400, error=True))
        notice = {"type": "notice", "level": "warn", "message": f"warn {index}" + "w" * 500}
        handler.feed(omp_line(notice))
    assert handler.blocked_count == 30 and len(handler.blocked) == 8
    assert all(len(sample) <= 200 for sample in handler.blocked)
    assert len(handler.warnings) == 10 and all(len(w) <= 300 for w in handler.warnings)


# ==============================================================================================
# claude
# ==============================================================================================


def test_claude_argv_is_exactly_the_verified_invocation(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("claude")
    result = h.run(FINAL)
    assert result.ok  # the fake enforces the real option table, choices and variadics
    assert result.command == [
        str(h.binary),
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
        "1.00",
        "--tools",
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
        "--allowedTools",
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
    ]
    parsed = h.invocation()["parsed"]
    tools = ["Read", "Edit", "Write", "Glob", "Grep"]
    assert parsed["flags"]["--tools"] == tools and parsed["flags"]["--allowedTools"] == tools
    assert parsed["positional"] == []


def test_claude_bash_rules_name_the_exact_commands(new_harness: Callable[..., Harness]) -> None:
    h = new_harness("claude")
    commands = ("pytest -q tests/unit", "ruff check .")
    result = h.run(FINAL, scope="code", bash_allowlist=commands)
    assert result.ok
    flags = h.invocation()["parsed"]["flags"]
    assert flags["--tools"] == ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
    assert flags["--allowedTools"] == [
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
        "Bash(pytest -q tests/unit)",
        "Bash(ruff check .)",
    ]
    assert h.invocation()["parsed"]["positional"] == []
    docs = new_harness("claude").argv()
    assert "Bash" not in docs and not any(arg.startswith("Bash(") for arg in docs)


@pytest.mark.parametrize(
    ("budget", "text"), [(0.5, "0.50"), (2, "2.00"), (1.234, "1.23"), (12.5, "12.50")]
)
def test_claude_budget_flag_is_formatted_to_cents(
    new_harness: Callable[..., Harness], budget: float, text: str
) -> None:
    argv = new_harness("claude").argv(budget_usd=budget)
    assert argv[argv.index("--max-budget-usd") + 1] == text


@pytest.mark.parametrize(
    ("thinking", "effort"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "max"),
        ("off", None),
        ("auto", None),
        (None, None),
    ],
)
def test_claude_thinking_maps_to_effort(
    new_harness: Callable[..., Harness], thinking: str | None, effort: str | None
) -> None:
    backend = {} if thinking is None else {"thinking": thinking}
    h = new_harness("claude", backend=backend)
    result = h.run(FINAL)
    assert result.ok
    if effort is None:
        assert "--effort" not in result.command
    else:
        assert result.command[result.command.index("--effort") + 1] == effort
    assert h.invocation()["parsed"]["flags"].get("--effort") == effort


def test_claude_extra_args_precede_the_variadic_options(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("claude", backend={"model": "m", "extra_args": ["--verbose", "--brief"]})
    result = h.run(FINAL)
    assert result.ok
    argv = result.command
    assert argv.index("--verbose") < argv.index("--brief") < argv.index("--tools")
    assert argv.index("--model") < argv.index("--verbose")
    assert argv[argv.index("--tools") :].count("--allowedTools") == 1
    assert h.invocation()["parsed"]["positional"] == []


def test_claude_ambient_additional_args_do_not_leak_into_the_child(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("claude")
    h.run(FINAL)
    call = h.invocation()
    assert "CLAUDE_ADDITIONAL_ARGS" not in call["env_keys"]
    assert "--dangerously-skip-permissions" not in call["argv"]
    assert call["parsed"]["flags"]["--permission-mode"] == "dontAsk"


def test_claude_array_form_reports_an_integer_tool_count(
    new_harness: Callable[..., Harness],
) -> None:
    single = new_harness("claude").run(FINAL)
    assert single.ok and single.tool_calls is None
    verbose = new_harness("claude", backend={"extra_args": ["--verbose"]}).run(FINAL)
    assert verbose.ok and verbose.tool_calls == 0
    assert verbose.session_id is not None


def test_claude_array_form_counts_tool_use_blocks() -> None:
    def assistant(*parts: dict[str, Any]) -> dict[str, Any]:
        return {"type": "assistant", "message": {"content": list(parts)}}

    use = {"type": "tool_use", "name": "Read", "input": {}}
    text = {"type": "text", "text": "hello"}
    messages = [
        {"type": "system", "subtype": "init", "session_id": SESSION, "model": "m1"},
        assistant(use, text),
        assistant(use, use),
        assistant(text),
        claude_result("done"),
    ]
    result = parse_claude(json.dumps(messages))
    assert result.ok and result.tool_calls == 3
    assert result.model == "claude-opus-5-5"  # modelUsage of the result wins over init
    only_init = [m for m in messages if m["type"] != "result"][:1] + [claude_result(modelUsage={})]
    assert parse_claude(json.dumps(only_init)).model == "m1"


def test_claude_budget_exhaustion_is_reported_by_the_cli_itself(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("claude")
    steps = [{"op": "cost", "usd": 0.7}, {"op": "cost", "usd": 0.7}, *FINAL]
    result = h.run(steps, budget_usd=1.0)
    assert result.status is AgentStatus.BUDGET_EXCEEDED
    assert result.failure_kind is FailureKind.BUDGET
    assert result.cost_usd == pytest.approx(1.4) and result.over_budget
    assert result.exit_code == 1
    assert "$1.00" in (result.error or "")
    assert result.stop_reason == "budget_exhausted"


@pytest.mark.parametrize(
    ("fields", "status", "kind"),
    [
        pytest.param(
            {"subtype": "error_max_turns", "is_error": True, "result": ""},
            AgentStatus.FAILED,
            FailureKind.MAX_TURNS,
            id="error_max_turns",
        ),
        pytest.param(
            {"terminal_reason": "max_turns", "is_error": True, "result": "turn limit"},
            AgentStatus.FAILED,
            FailureKind.MAX_TURNS,
            id="terminal-max_turns",
        ),
        pytest.param(
            {"subtype": "error_max_budget_usd", "is_error": True, "result": ""},
            AgentStatus.BUDGET_EXCEEDED,
            FailureKind.BUDGET,
            id="error_max_budget_usd",
        ),
        pytest.param(
            {"terminal_reason": "budget_exhausted", "is_error": True, "result": "budget"},
            AgentStatus.BUDGET_EXCEEDED,
            FailureKind.BUDGET,
            id="terminal-budget_exhausted",
        ),
        pytest.param(
            {"subtype": "error_during_execution", "is_error": True, "result": ""},
            AgentStatus.FAILED,
            FailureKind.API_ERROR,
            id="error_during_execution",
        ),
        pytest.param(
            {"is_error": True, "result": "Invalid API key"},
            AgentStatus.FAILED,
            FailureKind.AUTH,
            id="is_error-auth-text",
        ),
    ],
)
def test_claude_result_subtypes_map_to_status_and_kind(
    fields: dict[str, Any], status: AgentStatus, kind: FailureKind
) -> None:
    result = parse_claude(json.dumps(claude_result(**fields)), returncode=1)
    assert result.status is status
    assert result.failure_kind is kind
    assert result.error


def test_claude_max_turns_error_text_is_stable() -> None:
    result = parse_claude(json.dumps(claude_result(subtype="error_max_turns", is_error=True)))
    assert result.error == "claude stopped after reaching its turn limit"


def test_claude_nonzero_exit_with_a_clean_result_still_fails() -> None:
    result = parse_claude(json.dumps(claude_result("fine")), returncode=2, stderr="segfault\n")
    assert result.status is AgentStatus.FAILED
    assert result.failure_kind is FailureKind.NONZERO_EXIT
    assert result.error == "segfault"
    silent = parse_claude(json.dumps(claude_result("fine")), returncode=2)
    assert silent.error == "claude exited with status 2"


def test_claude_denials_are_counted_with_bounded_samples() -> None:
    denials = [{"tool_name": "Write", "tool_use_id": f"t{i}", "tool_input": {}} for i in range(10)]
    result = parse_claude(json.dumps(claude_result(permission_denials=denials)))
    assert result.ok and result.blocked_tool_calls == 10
    assert result.blocked_samples == ["Write: denied by permission policy"] * 8


def test_claude_denials_from_the_fake_cli_are_reported(
    new_harness: Callable[..., Harness],
) -> None:
    h = new_harness("claude")
    steps = [
        tool_step("claude", "write", {"path": "/etc/passwd"}, blocked=True),
        tool_step("claude", "bash", {"command": "rm -rf /"}, blocked=True),
        *FINAL,
    ]
    result = h.run(steps)
    assert result.ok and result.blocked_tool_calls == 2
    assert result.blocked_samples == [
        "Write: denied by permission policy",
        "Bash: denied by permission policy",
    ]


def test_claude_capture_overflow_is_reported_not_mistaken_for_success(
    new_harness: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(claude_module, "MIN_CAPTURE_BYTES", 1000)
    h = new_harness("claude", max_log_bytes=1000)
    result = h.run(FINAL)
    assert result.truncated is True
    assert result.status is AgentStatus.FAILED
    assert result.failure_kind is FailureKind.NO_RESULT
    assert result.error == "claude output was truncated before its result message"
    assert b"capture limit reached" in Path(result.stdout_log).read_bytes()


# -- load_messages -------------------------------------------------------------------------------


def real_claude_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_load_messages_single_result_object() -> None:
    text = json.dumps(claude_result("one"))
    assert [m["result"] for m in load_messages(text)] == ["one"]
    assert load_messages("\n  " + text + "\n\n")[0]["type"] == "result"


@pytest.mark.parametrize("name", ["claude_real_credit_error.json", "claude_real_auth_expired.json"])
def test_load_messages_real_three_message_array(name: str) -> None:
    messages = load_messages(real_claude_text(name))
    assert [m["type"] for m in messages] == ["system", "assistant", "result"]
    compact = json.dumps(json.loads(real_claude_text(name)), separators=(",", ":"))
    assert [m["type"] for m in load_messages(compact)] == ["system", "assistant", "result"]


def test_load_messages_ndjson_stream() -> None:
    items = json.loads(real_claude_text("claude_real_credit_error.json"))
    stream = "\n\n".join(json.dumps(item) for item in items) + "\n"
    assert [m["type"] for m in load_messages(stream)] == ["system", "assistant", "result"]
    noisy = "warning: something\n" + stream + "trailing noise\n[1, 2]\n"
    assert [m["type"] for m in load_messages(noisy)] == ["system", "assistant", "result"]


@pytest.mark.parametrize(
    "text",
    ["", "   \n\t\n", "Error: nothing useful happened", "{not json", "[1, 2, 3]", "42", '"str"'],
)
def test_load_messages_without_objects_is_empty(text: str) -> None:
    assert load_messages(text) == []


def test_load_messages_array_drops_non_objects() -> None:
    assert load_messages(json.dumps([1, {"type": "result"}, "x", None])) == [{"type": "result"}]


def test_load_messages_salvages_the_result_of_a_truncated_array() -> None:
    result = {"type": "result", "subtype": "success", "result": "kept", "is_error": False}
    system = {"type": "system", "subtype": "init"}
    text = json.dumps([system, {"type": "assistant", "message": {"content": []}}, result])
    cut = text[:-1]  # the closing bracket never arrived
    assert load_messages(cut) == [result]
    head_only = text[: text.index('{"type": "result"') - 3]
    assert load_messages(head_only) == []


def test_load_messages_salvages_a_truncated_array_with_the_real_key_order() -> None:
    """The real CLI prints ``type`` late in the result object (see the captured fixtures)."""
    raw = json.dumps(
        json.loads(real_claude_text("claude_real_credit_error.json")), separators=(",", ":")
    )
    messages = load_messages(raw[:-1])
    assert [m["type"] for m in messages] == ["result"]
    assert messages[0]["result"] == "Credit balance is too low"


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("claude_real_credit_error.json", FailureKind.CREDIT, "Credit balance is too low"),
        (
            "claude_real_auth_expired.json",
            FailureKind.AUTH,
            "Failed to authenticate: OAuth session expired and could not be refreshed",
        ),
    ],
)
def test_claude_real_error_captures_are_classified(
    new_harness: Callable[..., Harness], name: str, kind: FailureKind, message: str
) -> None:
    raw = real_claude_text(name)
    parsed = parse_claude(raw, returncode=1)
    assert parsed.status is AgentStatus.FAILED and parsed.failure_kind is kind
    assert parsed.error == message
    assert parsed.session_id == SESSION and parsed.model == "claude-opus-5-5"
    assert parsed.cost_usd == 0 and parsed.tool_calls == 0 and parsed.num_turns == 1
    assert parsed.stop_reason == "api_error" and parsed.blocked_tool_calls == 0
    h = new_harness("claude", replay=(json.dumps(json.loads(raw)) + "\n", "", 1))
    replayed = h.run()
    assert replayed.failure_kind is kind and replayed.error == message
    assert replayed.exit_code == 1


# ==============================================================================================
# extract_report
# ==============================================================================================

GOOD_REPORT = {"status": "completed", "summary": "Edited docs", "files_changed": ["docs/a.md"]}


def fence(body: str, ticks: str = "```", info: str = "json agent-result") -> str:
    return f"{ticks}{info}\n{body}\n{ticks}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(fence(json.dumps(GOOD_REPORT)), id="json-agent-result"),
        pytest.param(fence(json.dumps(GOOD_REPORT), info="agent-result"), id="agent-result"),
        pytest.param(fence(json.dumps(GOOD_REPORT), ticks="````"), id="four-backticks"),
        pytest.param(fence(json.dumps(GOOD_REPORT), ticks="`````"), id="five-backticks"),
        pytest.param(fence(json.dumps(GOOD_REPORT, indent=2)), id="pretty-printed"),
        pytest.param(fence(json.dumps(GOOD_REPORT), info="json   agent-result  "), id="spaces"),
        pytest.param("   " + fence(json.dumps(GOOD_REPORT)), id="indented-three"),
        pytest.param(fence(json.dumps(GOOD_REPORT)).replace("\n", "\r\n"), id="crlf"),
        pytest.param(
            "I changed the docs.\n\n" + fence(json.dumps(GOOD_REPORT)) + "\n\nThanks!",
            id="inside-prose",
        ),
    ],
)
def test_extract_report_accepts_fence_variants(text: str) -> None:
    report = extract_report(text)
    assert report is not None
    assert report.status == "completed" and report.summary == "Edited docs"
    assert report.files_changed == ["docs/a.md"] and report.notes is None


@pytest.mark.parametrize("status", ["completed", "blocked", "needs_human"])
def test_extract_report_accepts_every_status(status: str) -> None:
    body = json.dumps({**GOOD_REPORT, "status": status, "notes": "n", "extra": "ignored"})
    report = extract_report(fence(body))
    assert report is not None and report.status == status and report.notes == "n"


def test_extract_report_accepts_the_documented_limits() -> None:
    body = {"status": "completed", "summary": "s" * 4000, "files_changed": ["f"] * 200}
    report = extract_report(fence(json.dumps({**body, "notes": "n" * 2000})))
    assert report is not None and len(report.summary) == 4000 and len(report.files_changed) == 200


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("no block here", id="no-block"),
        pytest.param(fence("{not json"), id="invalid-json"),
        pytest.param(fence("[1, 2]"), id="list"),
        pytest.param(fence('"completed"'), id="string"),
        pytest.param(fence("null"), id="null"),
        pytest.param(fence(json.dumps({**GOOD_REPORT, "status": "done"})), id="bad-status"),
        pytest.param(fence(json.dumps({"status": "completed"})), id="missing-summary"),
        pytest.param(fence(json.dumps({**GOOD_REPORT, "summary": "x" * 4001})), id="big-summary"),
        pytest.param(fence(json.dumps({**GOOD_REPORT, "files_changed": "a"})), id="files-string"),
        pytest.param(
            fence(json.dumps({**GOOD_REPORT, "files_changed": ["f"] * 201})), id="too-many-files"
        ),
        pytest.param(fence(json.dumps({**GOOD_REPORT, "notes": "n" * 2001})), id="big-notes"),
        pytest.param(fence(json.dumps(GOOD_REPORT), info="python agent-result"), id="other-lang"),
        pytest.param(fence(json.dumps(GOOD_REPORT), info="json"), id="no-agent-result"),
        pytest.param(fence(json.dumps(GOOD_REPORT), info="agent-results"), id="similar-info"),
        pytest.param("    " + fence(json.dumps(GOOD_REPORT)), id="indented-four"),
        pytest.param("```json agent-result\n" + json.dumps(GOOD_REPORT), id="unterminated"),
    ],
)
def test_extract_report_ignores_everything_else(text: str) -> None:
    assert extract_report(text) is None


def test_extract_report_last_valid_block_wins() -> None:
    first = fence(json.dumps({**GOOD_REPORT, "summary": "first"}))
    second = fence(json.dumps({**GOOD_REPORT, "summary": "second", "status": "blocked"}))
    broken = fence("{nope")
    bad_status = fence(json.dumps({**GOOD_REPORT, "status": "bogus"}))
    assert (extract_report(f"{first}\n\n{second}") or pytest.fail("no report")).summary == "second"
    assert (extract_report(f"{first}\n\n{broken}") or pytest.fail("no report")).summary == "first"
    assert (extract_report(f"{broken}\n\n{second}") or pytest.fail("no report")).summary == "second"
    assert (extract_report(f"{first}\n{bad_status}") or pytest.fail("no report")).summary == "first"


def test_extract_report_survives_hostile_deeply_nested_json() -> None:
    assert extract_report(fence("[" * 20_000)) is None


# ==============================================================================================
# classify_error_text
# ==============================================================================================


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Credit balance is too low", FailureKind.CREDIT),
        (
            "400 Credit balance is too low to access the Anthropic API. Please go to Plans & "
            "Billing to upgrade or purchase credits. (type=invalid_request_error)",
            FailureKind.CREDIT,
        ),
        ("insufficient credits remaining", FailureKind.CREDIT),
        ("You are out of credits", FailureKind.CREDIT),
        ("Billing issue detected on the account", FailureKind.CREDIT),
        ("CREDIT BALANCE TOO LOW", FailureKind.CREDIT),
        ("Failed to authenticate: OAuth session expired", FailureKind.AUTH),
        (
            "401 Failed to authenticate: invalid x-api-key (type=authentication_error)",
            FailureKind.AUTH,
        ),
        ("Invalid API key provided", FailureKind.AUTH),
        ("401 Unauthorized", FailureKind.AUTH),
        ("Not logged in · Please run /login", FailureKind.AUTH),
        ("403 Forbidden", FailureKind.AUTH),
        ("429 rate limit exceeded (type=rate_limit_error)", FailureKind.RATE_LIMIT),
        ("Too many requests, slow down", FailureKind.RATE_LIMIT),
        ("The API is overloaded", FailureKind.RATE_LIMIT),
        ("rate_limit_error", FailureKind.RATE_LIMIT),
        ("500 Internal server error (type=api_error)", FailureKind.API_ERROR),
        ("connection reset by peer", FailureKind.API_ERROR),
        ("something odd happened", FailureKind.API_ERROR),
        ("", FailureKind.API_ERROR),
        ("invalid x-api-key and also credit balance is too low", FailureKind.CREDIT),
    ],
)
def test_classify_error_text(text: str, kind: FailureKind) -> None:
    assert classify_error_text(text) is kind


# ==============================================================================================
# build_backend(s), resolve_binary, version
# ==============================================================================================


def test_registry_names_both_backends() -> None:
    assert BACKENDS == {"omp": OmpBackend, "claude": ClaudeBackend}
    assert OmpBackend.name == "omp" and ClaudeBackend.name == "claude"


def test_an_unknown_backend_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown agent backend 'gpt'"):
        build_backend("gpt", AgentConfig())


def test_build_backends_builds_only_the_allowed_ones() -> None:
    both = build_backends(AgentConfig(), environ={})
    assert sorted(both) == ["claude", "omp"]
    assert isinstance(both["omp"], OmpBackend) and isinstance(both["claude"], ClaudeBackend)
    only_claude = build_backends(AgentConfig(default="claude", allowed=["claude"]), environ={})
    assert list(only_claude) == ["claude"]
    only_omp = build_backends(AgentConfig(default="omp", allowed=["omp"]), environ={})
    assert list(only_omp) == ["omp"]


def test_build_backends_refuses_a_forbidden_pass_env() -> None:
    cfg = AgentConfig(claude=BackendConfig(binary="claude", pass_env=["GH_TOKEN"]))
    with pytest.raises(ValueError, match="GH_TOKEN"):
        build_backends(cfg, environ={})


def test_build_backend_applies_the_agent_settings() -> None:
    redactor = Redactor()
    cfg = AgentConfig(
        omp=BackendConfig(binary="omp-custom"),
        command_prefix=["sandbox", "--strict"],
        kill_grace_seconds=3.5,
    )
    backend = build_backend("omp", cfg, redactor=redactor, max_log_bytes=12_345, environ={})
    assert backend.cfg.binary == "omp-custom"
    assert backend.command_prefix == ["sandbox", "--strict"]
    assert backend.kill_grace_s == 3.5 and backend.max_log_bytes == 12_345
    assert backend.redactor is redactor


def test_resolve_binary_by_absolute_path(tmp_path: Path) -> None:
    good = make_script(tmp_path / "good", "exit 0")
    plain = tmp_path / "plain"
    plain.write_text("x", encoding="utf-8")
    plain.chmod(0o644)

    def resolved(binary: Path) -> str | None:
        return OmpBackend(BackendConfig(binary=str(binary)), environ={}).resolve_binary()

    assert resolved(good) == str(good)
    assert resolved(plain) is None
    assert resolved(tmp_path / "absent") is None
    assert resolved(tmp_path) is None


def test_resolve_binary_searches_the_backend_environment_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    installed = install_fake_cli(bin_dir, "omp", name="omp-under-test")
    cfg = BackendConfig(binary="omp-under-test")
    assert OmpBackend(cfg, environ={"PATH": str(bin_dir)}).resolve_binary() == str(installed)
    monkeypatch.setenv("PATH", str(bin_dir))  # the process PATH must not leak into the lookup
    assert OmpBackend(cfg, environ={"PATH": str(tmp_path / "elsewhere")}).resolve_binary() is None
    assert (
        OmpBackend(
            BackendConfig(binary="nope-cli"), environ={"PATH": str(bin_dir)}
        ).resolve_binary()
        is None
    )


def test_resolve_binary_expands_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = install_fake_cli(tmp_path / "bin", "omp")
    monkeypatch.setenv("HOME", str(tmp_path))
    backend = OmpBackend(BackendConfig(binary="~/bin/omp"), environ={})
    assert backend.resolve_binary() == str(installed)


@pytest.mark.parametrize(
    ("kind", "expected"), [("omp", "omp/18.12.6"), ("claude", "2.1.293 (Claude Code)")]
)
def test_version_is_the_first_line_of_dash_dash_version(
    new_harness: Callable[..., Harness], kind: str, expected: str
) -> None:
    assert new_harness(kind).backend.version() == expected


def test_version_edge_cases(tmp_path: Path) -> None:
    def version_of(*lines: str, timeout_s: float = 20.0) -> str | None:
        script = make_script(tmp_path / "cli", *lines)
        return OmpBackend(BackendConfig(binary=str(script)), environ={}).version(timeout_s)

    assert version_of("echo " + "v" * 300, "echo second line") == "v" * 120
    assert version_of("echo 'only on stderr 1.2.3' >&2") == "only on stderr 1.2.3"
    assert version_of("exit 0") is None
    assert version_of("exec sleep 30", timeout_s=0.3) is None
    missing = OmpBackend(BackendConfig(binary=str(tmp_path / "absent")), environ={})
    assert missing.version() is None
    started = time.monotonic()
    assert version_of("exec sleep 30", timeout_s=0.3) is None
    assert time.monotonic() - started < 10
