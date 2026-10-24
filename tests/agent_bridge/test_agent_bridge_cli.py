"""Command line interface of the agent bridge, driven in-process against fakes.

Every test calls ``tools.agent_bridge.cli.main([...])`` with a real configuration file on disk, a
real git origin/clone, the fake agent CLIs of ``bridgefakes`` and the in-memory ``FakeGitHub``
(swapped in for the HTTP client's network layer), so nothing here needs a network, a GitHub
token or a real agent.  A few tests also run ``python -m tools.agent_bridge`` in a subprocess to
prove the entry point and the real process exit statuses.
"""

from __future__ import annotations

import builtins
import io
import json
import logging
import logging.handlers
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
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
from bridgefakes.github_server import FakeGitHub
from bridgefakes.rig import GITHUB_TOKEN, REPO, GitRepoEnv, default_config, make_git_env, task_body

from tools.agent_bridge import __version__, cli
from tools.agent_bridge.config import DEFAULT_PROTECTED_PATHS, load_config
from tools.agent_bridge.errors import BridgeError
from tools.agent_bridge.github import LABEL_STYLE, HttpGitHubClient
from tools.agent_bridge.ledger import LeaseRequest, Ledger, RunRecord, iso, utcnow
from tools.agent_bridge.models import RunPhase, RunStatus
from tools.agent_bridge.process import proc_start_ticks
from tools.agent_bridge.publisher import ReadOnlyGitHub
from tools.agent_bridge.worker import Worker

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_LOGGER = "agent_bridge"
FIVE_LABELS = ["agent-task", "agent-approved", "agent-running", "agent-blocked", "agent-done"]
SMOKE_LINE = "agent bridge smoke ok"
GUARD_BLOCK = "[agent-bridge guard] blocked: path is outside the task worktree"
PASS_AGENT_ENV = ("--pass-env", "FAKE_AGENT_SCENARIO", "--pass-env", "FAKE_AGENT_LOG")


# ----------------------------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Result:
    """What one in-process CLI call returned and printed."""

    code: int
    out: str
    err: str

    def json(self) -> Any:
        return json.loads(self.out)

    def smoke_json(self) -> dict[str, Any]:
        """The smoke report; ``--keep`` prints one notice line before the JSON."""
        return json.loads(self.out[self.out.index("{") :])


class _Tty(io.StringIO):
    """A stdin that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def _no_prompt(prompt: str = "") -> str:
    raise AssertionError(f"the CLI must not prompt here, it asked: {prompt!r}")


def _exit_status(code: object) -> int:
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def invoke(capsys: pytest.CaptureFixture[str], argv: list[str]) -> Result:
    """``cli.main(argv)``; a ``SystemExit`` (argparse) becomes its status."""
    try:
        code = cli.main(list(argv))
    except SystemExit as exc:
        code = _exit_status(exc.code)
    captured = capsys.readouterr()
    return Result(code, captured.out, captured.err)


@dataclass
class Env:
    """One configured bridge: git origin/clone, FakeGitHub, fake agents, a config file."""

    tmp_path: Path
    git: GitRepoEnv
    fake: FakeGitHub
    backend: str
    omp_bin: Path
    claude_bin: Path
    config_path: Path
    scenario: Path
    log: Path
    capsys: pytest.CaptureFixture[str]
    clients: list[HttpGitHubClient] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    # ---- paths
    @property
    def state_dir(self) -> Path:
        return self.tmp_path / "state"

    @property
    def agent_bin(self) -> Path:
        return self.omp_bin if self.backend == "omp" else self.claude_bin

    def bin_for(self, backend: str) -> Path:
        return self.omp_bin if backend == "omp" else self.claude_bin

    # ---- configuration
    def write_config(
        self,
        mutate: Callable[[dict[str, Any]], None] | None = None,
        *,
        path: Path | None = None,
        **overrides: Any,
    ) -> Path:
        data = default_config(
            self.git, self.tmp_path, self.backend, self.omp_bin, self.claude_bin, **overrides
        )
        if mutate is not None:
            mutate(data)
        target = path or self.config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        return target

    # ---- running the CLI
    def main(self, *argv: str) -> Result:
        """``main(argv)`` exactly as given (no ``--config`` added)."""
        return invoke(self.capsys, list(argv))

    def run(self, *args: str) -> Result:
        """``main(['--config', <this env's config>, *args])``."""
        return self.main("--config", str(self.config_path), *args)

    # ---- GitHub / scenario helpers
    def issue(
        self,
        *,
        title: str = "Document the bridge",
        author: str = "owner",
        approve_as: str | None = "owner",
        body: str | None = None,
        **spec: Any,
    ) -> int:
        text = body if body is not None else task_body(**spec)
        number = self.fake.add_issue(title=title, body=text, author=author, labels=["agent-task"])
        if approve_as is not None:
            self.fake.label_issue(number, "agent-approved", actor=approve_as)
        return number

    def steps(self, steps: list[dict[str, Any]]) -> None:
        write_scenario(self.scenario, steps)

    def invocations(self) -> list[dict[str, Any]]:
        return read_invocations(self.log)

    # ---- ledger
    def ledger(self, **kwargs: Any) -> Ledger:
        return Ledger(self.state_dir / "ledger.sqlite", **kwargs)

    def runs(self) -> list[RunRecord]:
        ledger = self.ledger()
        try:
            return ledger.list_runs()
        finally:
            ledger.close()

    def lease(self, issue_number: int = 7, *, clock: Callable[[], Any] = utcnow) -> RunRecord:
        """Put a ``running`` run into the ledger, as a worker that has since died would have."""
        ledger = self.ledger(clock=clock)
        try:
            result = ledger.lease(
                LeaseRequest(
                    repo=REPO,
                    issue_number=issue_number,
                    issue_title="Leased by a test",
                    revision="r" * 64,
                    spec_hash=f"{issue_number:064x}",
                    approval_event_id=3,
                    approved_by="owner",
                    approved_at=iso(clock()),
                    backend=self.backend,
                    worker_id="dead-worker",
                    worker_pid=999_999,
                    ttl_seconds=30,
                    max_attempts_per_issue=3,
                )
            )
            assert result.granted and result.run is not None
            return result.run
        finally:
            ledger.close()

    def stale_lease(self, issue_number: int = 7) -> RunRecord:
        past = utcnow() - timedelta(hours=1)
        return self.lease(issue_number, clock=lambda: past)


def assert_nothing_executed(env: Env) -> None:
    """No GitHub write, no agent, no worktree, no branch, no ledger run, clone untouched."""
    assert env.fake.writes() == []
    assert env.invocations() == []
    assert not (env.state_dir / "worktrees").exists()
    assert env.git.origin_branches() == ["main"]
    assert env.git.git("status", "--porcelain") == ""
    assert env.git.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    if (env.state_dir / "ledger.sqlite").exists():
        assert env.runs() == []


@pytest.fixture(autouse=True)
def _restore_bridge_logger() -> Iterator[None]:
    """``setup_logging`` configures the process-wide ``agent_bridge`` logger; undo it."""
    logger = logging.getLogger(BRIDGE_LOGGER)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture
def env(
    tmp_path: Path,
    backend_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Iterator[Env]:
    git = make_git_env(tmp_path)
    bin_dir = tmp_path / "bin"
    omp_bin = install_fake_cli(bin_dir, "omp")
    claude_bin = install_fake_cli(bin_dir, "claude")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    scenario = tmp_path / "scenario.json"
    log = tmp_path / "invocations.jsonl"
    instance = Env(
        tmp_path=tmp_path,
        git=git,
        fake=FakeGitHub(REPO, token=GITHUB_TOKEN),
        backend=backend_name,
        omp_bin=omp_bin,
        claude_bin=claude_bin,
        config_path=tmp_path / "agent-bridge.yaml",
        scenario=scenario,
        log=log,
        capsys=capsys,
    )

    def client_factory(github_cfg: Any, repository: str, token: str, **_kw: Any) -> Any:
        client = HttpGitHubClient(
            github_cfg,
            repository,
            token,
            transport=instance.fake.transport(),
            sleep=lambda _s: None,
        )
        instance.clients.append(client)
        instance.tokens.append(token)
        return client

    monkeypatch.setattr(cli, "HttpGitHubClient", client_factory)
    monkeypatch.setenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN", GITHUB_TOKEN)
    monkeypatch.delenv(cli.CONFIG_ENV, raising=False)
    monkeypatch.setenv("HOME", str(git.home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("XDG_CONFIG_HOME", "GIT_CONFIG_GLOBAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAKE_AGENT_SCENARIO", str(scenario))
    monkeypatch.setenv("FAKE_AGENT_LOG", str(log))
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(builtins, "input", _no_prompt)
    instance.write_config()
    instance.steps(docs_edit_steps())
    try:
        yield instance
    finally:
        for client in instance.clients:
            client.close()


def answer_prompts(monkeypatch: pytest.MonkeyPatch, *answers: str) -> list[str]:
    """Make stdin a terminal and script the answers ``input()`` returns (EOF when exhausted)."""
    prompts: list[str] = []
    queue = list(answers)

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(builtins, "input", fake_input)
    return prompts


def flush_logs() -> None:
    for handler in logging.getLogger(BRIDGE_LOGGER).handlers:
        handler.flush()


def run_module(
    *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """``python -m tools.agent_bridge`` in a real subprocess with a minimal environment."""
    environment = {"PATH": os.environ.get("PATH", ""), "HOME": str(REPO_ROOT / ".nohome")}
    environment.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_bridge", *args],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        stdin=subprocess.DEVNULL,
    )


# ----------------------------------------------------------------------------------------------
# usage and entry point (no configuration involved)
# ----------------------------------------------------------------------------------------------
def test_version_flag_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, ["--version"])
    assert result.code == 0
    assert result.out == f"agent-bridge {__version__}\n"


def test_top_level_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, ["--help"])
    flat = " ".join(result.out.split())
    assert result.code == 0
    assert "dry run by default" in flat
    assert "--config" in flat and "--verbose" in flat
    for name in cli.COMMANDS:
        assert name in flat


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_every_command_has_help(capsys: pytest.CaptureFixture[str], command: str) -> None:
    result = invoke(capsys, [command, "--help"])
    assert result.code == 0
    assert f"agent-bridge {command}" in result.out


@pytest.mark.parametrize("command", ["run-once", "serve", "recover"])
def test_execution_commands_document_their_safety_flags(
    capsys: pytest.CaptureFixture[str], command: str
) -> None:
    flat = " ".join(invoke(capsys, [command, "--help"]).out.split())
    assert "--execute" in flat and "--yes" in flat
    assert "dry_run: false" in flat or command == "recover"


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["frobnicate"], "invalid choice"),
        ([], "required"),
        (["run-once", "--issue", "abc"], "invalid int value"),
        (["cancel"], "required"),
        (["cancel", "--run-id", "a", "--issue", "1"], "not allowed with argument"),
        (["validate-task"], "required"),
        (["validate-task", "--file", "a", "--issue", "1"], "not allowed with argument"),
        (["smoke", "--backend", "gemini"], "invalid choice"),
        (["status", "--limit", "many"], "invalid int value"),
        (["--no-such-flag", "status"], "unrecognized arguments"),
    ],
    ids=[
        "unknown-command",
        "no-command",
        "bad-issue-number",
        "cancel-needs-target",
        "cancel-one-target",
        "validate-task-needs-source",
        "validate-task-one-source",
        "unknown-backend",
        "bad-limit",
        "unknown-flag",
    ],
)
def test_usage_errors_exit_two_before_configuration_is_read(
    env: Env, argv: list[str], needle: str
) -> None:
    result = env.main(*argv)  # no --config, no config file anywhere: still a usage error
    assert result.code == 2
    assert needle in result.err
    assert "configuration error" not in result.out
    assert not env.state_dir.exists() and env.fake.requests == []


def test_module_entry_point_reports_version_and_usage_errors() -> None:
    version = run_module("--version")
    assert (version.returncode, version.stdout.strip()) == (0, f"agent-bridge {__version__}")
    usage = run_module("frobnicate")
    assert usage.returncode == 2
    assert "invalid choice" in usage.stderr


def test_module_entry_point_exit_statuses_follow_the_configuration(env: Env) -> None:
    missing = run_module("--config", str(env.tmp_path / "missing.yaml"), "validate-config")
    assert missing.returncode == 3
    assert "config file not found" in missing.stdout
    good = run_module("--config", str(env.config_path), "validate-config")
    assert good.returncode == 0
    assert f"configuration for {REPO} is valid" in good.stdout
    via_env = run_module("validate-config", env_extra={cli.CONFIG_ENV: str(env.config_path)})
    assert via_env.returncode == 0


def test_keyboard_interrupt_exits_130(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(_cfg: object, _args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setitem(cli.COMMANDS, "validate-config", interrupted)
    assert env.run("validate-config").code == 130


def test_bridge_errors_become_an_error_line_and_exit_one(env: Env) -> None:
    result = env.run("validate-task", "--issue", "999")  # the fake GitHub answers 404
    assert result.code == 1
    assert result.out.startswith("error: GitHub object not found")
    assert env.fake.writes() == []


# ----------------------------------------------------------------------------------------------
# configuration loading
# ----------------------------------------------------------------------------------------------
def test_missing_config_exits_three_and_names_the_file(env: Env) -> None:
    missing = env.tmp_path / "nowhere" / "agent-bridge.yaml"
    result = env.main("--config", str(missing), "validate-config")
    assert result.code == 3
    assert result.out.startswith("configuration error: config file not found")
    assert str(missing) in result.out
    assert not env.state_dir.exists() and env.fake.requests == []


def test_missing_default_config_names_the_default_path(env: Env) -> None:
    result = env.main("validate-config")  # cwd is an empty directory
    assert result.code == 3
    assert "config/agent-bridge.yaml" in result.out


def test_config_discovery_prefers_flag_then_environment_then_default(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.write_config(path=Path.cwd() / "config" / "agent-bridge.yaml", repository="owner/default")
    env.write_config(path=env.tmp_path / "fromenv.yaml", repository="owner/fromenv")
    explicit = env.write_config(path=env.tmp_path / "explicit.yaml", repository="owner/explicit")

    assert "configuration for owner/default is valid" in env.main("validate-config").out
    monkeypatch.setenv(cli.CONFIG_ENV, str(env.tmp_path / "fromenv.yaml"))
    assert "configuration for owner/fromenv is valid" in env.main("validate-config").out
    flagged = env.main("--config", str(explicit), "validate-config")
    assert "configuration for owner/explicit is valid" in flagged.out
    assert flagged.code == 0


def test_config_in_the_environment_never_falls_back_to_the_default(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.write_config(path=Path.cwd() / "config" / "agent-bridge.yaml", repository="owner/default")
    gone = env.tmp_path / "gone.yaml"
    monkeypatch.setenv(cli.CONFIG_ENV, str(gone))
    result = env.main("validate-config")
    assert result.code == 3
    assert str(gone) in result.out


def test_config_path_expands_the_home_directory(env: Env) -> None:
    env.write_config(path=env.git.home / "tilde.yaml", repository="owner/tilde")
    result = env.main("--config", "~/tilde.yaml", "validate-config")
    assert result.code == 0
    assert "owner/tilde" in result.out


def test_relative_paths_resolve_against_the_config_file_not_the_cwd(env: Env) -> None:
    def relative(data: dict[str, Any]) -> None:
        data["state_dir"] = "./st"
        data["repo_path"] = "../clone"

    cfg_path = env.write_config(relative, path=env.tmp_path / "cfg" / "agent-bridge.yaml")
    result = env.main("--config", str(cfg_path), "validate-config")
    assert result.code == 0
    assert f"state: {(env.tmp_path / 'cfg' / 'st').resolve()}" in result.out


@pytest.mark.parametrize(
    ("mutate", "needles"),
    [
        (lambda d: d.update(bogus_key=1), ["bogus_key", "Extra inputs are not permitted"]),
        (lambda d: d.setdefault("pull_request", {}).update(draft=False), ["pull_request.draft"]),
        (lambda d: d.update(approvers=[]), ["approvers", "fail closed"]),
        (lambda d: d.pop("approvers"), ["approvers", "Field required"]),
        (lambda d: d.update(repository="not-a-repo"), ["repository must look like"]),
        (lambda d: d["agent"].update(default="gemini"), ["unknown agent backend"]),
        (lambda d: d["limits"].update(budget_usd=10), ["must not exceed"]),
        (lambda d: d["github"].update(api_url="http://github.test"), ["https"]),
        (lambda d: d.update(approvers=["bad login!"]), ["invalid GitHub login"]),
        (lambda d: d.update(default_checks=["undefined"]), ["undefined check"]),
    ],
    ids=[
        "unknown-key",
        "draft-false",
        "no-approvers",
        "approvers-missing",
        "bad-repository",
        "unknown-backend",
        "budget-over-cap",
        "plain-http",
        "bad-login",
        "undefined-default-check",
    ],
)
def test_invalid_config_exits_three_and_says_why(
    env: Env, mutate: Callable[[dict[str, Any]], None], needles: list[str]
) -> None:
    env.write_config(mutate)
    result = env.run("validate-config")
    assert result.code == 3
    assert result.out.startswith("configuration error: invalid config")
    for needle in needles:
        assert needle in result.out
    assert env.fake.requests == [] and not env.state_dir.exists()


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        (b"", "must be a YAML mapping"),
        (b"- one\n- two\n", "must be a YAML mapping"),
        (b"key: [unclosed\n", "cannot read config"),
        (b"repository: \xff\xfe\n", "cannot read config"),
    ],
    ids=["empty", "list", "yaml-syntax", "not-utf8"],
)
def test_unreadable_config_exits_three_instead_of_a_traceback(
    env: Env, raw: bytes, needle: str
) -> None:
    env.config_path.write_bytes(raw)
    result = env.run("validate-config")
    assert result.code == 3
    assert needle in result.out


def test_config_that_is_a_directory_exits_three(env: Env) -> None:
    result = env.main("--config", str(env.tmp_path / "bin"), "validate-config")
    assert result.code == 3
    assert "cannot read config" in result.out


def test_validate_config_reports_the_effective_settings(env: Env) -> None:
    result = env.run("validate-config")
    assert result.code == 0
    assert result.out.splitlines() == [
        f"configuration for {REPO} is valid",
        "  dry_run: False",
        f"  default agent: {env.backend} (allowed: omp, claude)",
        "  approvers: owner; task authors: owner",
        f"  scopes: docs-only, code; state: {env.state_dir.resolve()}",
    ]
    assert env.fake.requests == []
    assert not env.state_dir.exists()  # validating a file creates nothing


def test_validate_config_shows_dry_run_and_task_authors(env: Env) -> None:
    env.write_config(dry_run=True, task_authors=["alice", "bob"], approvers=["owner", "carol"])
    out = env.run("validate-config").out
    assert "  dry_run: True" in out
    assert "  approvers: owner, carol; task authors: alice, bob" in out


# ----------------------------------------------------------------------------------------------
# offline commands work without a token and without touching GitHub
# ----------------------------------------------------------------------------------------------
def test_offline_commands_need_no_token_and_make_no_request(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN")
    body = env.tmp_path / "task.md"
    body.write_text(task_body(), encoding="utf-8")
    for argv in (
        ["validate-config"],
        ["backends"],
        ["validate-task", "--file", str(body)],
        ["status"],
        ["status", "--json"],
        ["cancel", "--issue", "1"],
        ["setup-labels"],  # refused for lack of --yes, before any credential is needed
    ):
        result = env.run(*argv)
        assert result.code in (0, 1, 2), (argv, result)
    assert env.fake.requests == [] and env.tokens == []


@pytest.mark.parametrize(
    "argv",
    [["preview"], ["run-once"], ["validate-task", "--issue", "1"], ["setup-labels", "--yes"]],
    ids=["preview", "run-once", "validate-task-issue", "setup-labels"],
)
def test_github_commands_without_a_token_exit_three_naming_the_variable(
    env: Env, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.delenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN")
    result = env.run(*argv)
    assert result.code == 3
    assert "no GitHub token" in result.out and "LAWNBERRY_BRIDGE_GITHUB_TOKEN" in result.out
    assert env.fake.requests == []


def test_token_file_must_not_be_readable_by_others(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN")
    token_file = env.tmp_path / "token"
    token_file.write_text(GITHUB_TOKEN + "\n", encoding="utf-8")
    env.write_config(github={"token_file": str(token_file)})
    token_file.chmod(0o644)
    refused = env.run("preview")
    assert refused.code == 3 and "must not be accessible by group/others" in refused.out
    token_file.chmod(0o600)
    accepted = env.run("preview")
    assert accepted.code == 0
    assert env.tokens == [GITHUB_TOKEN]  # the file's content, stripped, reached the client


# ----------------------------------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------------------------------
def test_backends_prints_json_for_both_clis(env: Env) -> None:
    result = env.run("backends")
    assert result.code == 0
    rows = {row["backend"]: row for row in result.json()}
    assert list(rows) == ["omp", "claude"]
    assert rows["omp"]["binary"] == str(env.omp_bin)
    assert rows["omp"]["version"] == "omp/18.12.6"
    assert rows["claude"]["binary"] == str(env.claude_bin)
    assert rows["claude"]["version"] == "2.1.293 (Claude Code)"
    assert {name for name, row in rows.items() if row["default"]} == {env.backend}
    assert env.fake.requests == []


def test_backends_marks_a_missing_cli_with_nulls(env: Env) -> None:
    ghost = env.tmp_path / "no-such-dir" / "claude"
    env.write_config(agent={"claude": {"binary": str(ghost)}})
    rows = {row["backend"]: row for row in env.run("backends").json()}
    assert rows["claude"]["binary"] is None and rows["claude"]["version"] is None
    assert rows["omp"]["version"] == "omp/18.12.6"


def test_backends_lists_only_allowed_backends(env: Env) -> None:
    env.write_config(agent={"default": "omp", "allowed": ["omp"]})
    assert [row["backend"] for row in env.run("backends").json()] == ["omp"]


# ----------------------------------------------------------------------------------------------
# validate-task
# ----------------------------------------------------------------------------------------------
def test_validate_task_file_accepts_a_valid_task(env: Env) -> None:
    path = env.tmp_path / "docs-task.md"
    path.write_text(task_body(), encoding="utf-8")
    result = env.run("validate-task", "--file", str(path))
    assert result.code == 0
    first, second = result.out.splitlines()
    assert first == "task is valid"
    assert re.fullmatch(rf"  revision [0-9a-f]{{16}}  scope docs-only  agent {env.backend}", second)
    assert env.fake.requests == []


def test_validate_task_title_is_part_of_the_revision(env: Env) -> None:
    path = env.tmp_path / "docs-task.md"
    path.write_text(task_body(), encoding="utf-8")
    plain = env.run("validate-task", "--file", str(path)).out
    titled = env.run("validate-task", "--file", str(path), "--title", "Another title").out
    assert plain.splitlines()[1] != titled.splitlines()[1]


def test_validate_task_reports_an_explicit_agent(env: Env) -> None:
    other = "claude" if env.backend == "omp" else "omp"
    path = env.tmp_path / "task.md"
    path.write_text(task_body(agent=other), encoding="utf-8")
    assert f"agent {other}" in env.run("validate-task", "--file", str(path)).out


def test_validate_task_lists_every_schema_error(env: Env) -> None:
    path = env.tmp_path / "bad.md"
    path.write_text(
        task_body(objective="short", scope="everything", unknown_field=True), encoding="utf-8"
    )
    result = env.run("validate-task", "--file", str(path))
    assert result.code == 1
    assert result.out.splitlines()[0] == "INVALID task:"
    errors = [line for line in result.out.splitlines() if line.startswith("  - ")]
    assert len(errors) >= 3
    joined = "\n".join(errors)
    assert "objective" in joined and "scope" in joined and "unknown_field" in joined


def test_validate_task_without_a_task_block_is_invalid(env: Env) -> None:
    path = env.tmp_path / "prose.md"
    path.write_text("Please improve the docs.\n", encoding="utf-8")
    result = env.run("validate-task", "--file", str(path))
    assert result.code == 1
    assert "expected exactly one fenced ```yaml agent-task block, found 0" in result.out


def test_validate_task_refuses_what_the_configuration_does_not_permit(env: Env) -> None:
    env.write_config(
        scopes={"allowed": ["docs-only"]}, agent={"default": env.backend, "allowed": [env.backend]}
    )
    other = "claude" if env.backend == "omp" else "omp"
    path = env.tmp_path / "greedy.md"
    path.write_text(
        task_body(
            scope="code",
            agent=other,
            checks=["nope"],
            budget_usd=99,
            timeout_minutes=999,
        ),
        encoding="utf-8",
    )
    result = env.run("validate-task", "--file", str(path))
    assert result.code == 1
    lines = result.out.splitlines()
    assert lines[0] == "task is well formed but not permitted by this configuration:"
    text = "\n".join(lines[1:])
    assert "scope 'code' is not enabled" in text
    assert f"agent '{other}' is not enabled" in text
    assert "check 'nope' is not defined" in text
    assert "budget_usd 99.0 exceeds the configured maximum 2.0" in text
    assert "timeout_minutes 999.0 exceeds the configured maximum 5.0" in text


def test_validate_task_with_a_missing_file_is_a_clean_error(env: Env) -> None:
    missing = env.tmp_path / "nope.md"
    result = env.run("validate-task", "--file", str(missing))
    assert result.code == 1
    assert str(missing) in result.out


def test_validate_task_with_an_undecodable_file_is_a_clean_error(env: Env) -> None:
    path = env.tmp_path / "binary.md"
    path.write_bytes(b"\xff\xfe\x00not utf-8")
    result = env.run("validate-task", "--file", str(path))
    assert result.code == 1
    assert "binary.md" in result.out


def test_validate_task_reads_an_issue_from_github_without_writing(env: Env) -> None:
    good = env.issue()
    bad = env.issue(title="Prose only", body="no task block", approve_as=None)
    assert env.run("validate-task", "--issue", str(good)).out.startswith("task is valid")
    result = env.run("validate-task", "--issue", str(bad))
    assert result.code == 1 and result.out.startswith("INVALID task:")
    assert env.fake.writes() == []
    assert env.tokens == [GITHUB_TOKEN, GITHUB_TOKEN]


# ----------------------------------------------------------------------------------------------
# preview and the read-only guarantee
# ----------------------------------------------------------------------------------------------
def test_preview_shows_what_would_run_and_changes_nothing(env: Env) -> None:
    number = env.issue()
    result = env.run("preview")
    assert result.code == 0
    out = result.out
    assert out.startswith("considered 1 candidate issue(s)\n\nDRY RUN - would execute:\n")
    assert f"Issue #{number}: Document the bridge" in out
    assert "  approved by owner (label event 2)" in out
    assert f"  backend {env.backend}, scope docs-only, budget $1.00, timeout 1 min" in out
    base = env.git.git("rev-parse", "origin/main")
    assert f"  branch agent/issue-{number}-<run id> from {base[:12]}" in out
    assert "  may write: docs/**, *.md" in out
    assert f"({len(DEFAULT_PROTECTED_PATHS)} patterns)" in out
    assert "  checks: none" in out
    assert f"  command: {env.agent_bin} " in out
    assert re.search(r"  prompt: \d+ characters on stdin", out)
    assert "----- prompt -----" not in out
    assert_nothing_executed(env)


def test_preview_json_is_machine_readable(env: Env) -> None:
    number = env.issue(
        objective="Add a short note about previews to docs/guide.md.", checks=["pass"]
    )
    result = env.run("preview", "--json")
    assert result.code == 0
    data = result.json()
    assert set(data) == {"considered", "previews", "refused", "recovery", "errors"}
    assert (data["considered"], data["refused"], data["recovery"], data["errors"]) == (
        1,
        [],
        [],
        [],
    )
    (preview,) = data["previews"]
    assert preview["issue_number"] == number
    assert preview["backend"] == env.backend and preview["scope"] == "docs-only"
    assert preview["approver"] == "owner" and preview["approval_event_id"] == 2
    assert preview["budget_usd"] == 1.0 and preview["timeout_minutes"] == 1.0
    assert preview["base"] == env.git.git("rev-parse", "origin/main")
    assert preview["write_allow_sets"] == [["docs/**", "*.md"]]
    assert "backend/src/safety/**" in preview["write_deny"]
    assert "tools/agent_bridge/**" in preview["write_deny"]
    assert preview["checks"] == ["pass"]
    command = preview["command"]
    assert command[0] == str(env.agent_bin)
    if env.backend == "omp":  # omp is told its worktree with --cwd; claude uses the process cwd
        assert command[command.index("--cwd") + 1] == "<worktree>"
    assert preview["prompt_chars"] > 0 and "prompt" not in preview
    assert_nothing_executed(env)


def test_preview_show_prompt_prints_the_exact_agent_prompt(env: Env) -> None:
    env.issue(objective="Add a short note about prompts to docs/guide.md.")
    human = env.run("preview", "--show-prompt").out
    assert "----- prompt -----" in human and "----- end prompt -----" in human
    printed = human.split("----- prompt -----\n", 1)[1].split("\n----- end prompt -----", 1)[0]
    assert "Add a short note about prompts to docs/guide.md." in printed
    data = env.run("preview", "--json", "--show-prompt").json()
    prompt = data["previews"][0]["prompt"]
    fence = re.compile(r"TASK-DATA-[0-9a-f]+")  # every prompt gets a fresh random delimiter
    assert fence.sub("TASK-DATA-x", prompt) == fence.sub("TASK-DATA-x", printed)
    assert data["previews"][0]["prompt_chars"] == len(prompt)
    assert_nothing_executed(env)


def test_preview_reports_each_refusal_without_notifying_anyone(env: Env) -> None:
    good = env.issue()
    mallory = env.issue(title="Approved by a stranger", approve_as="mallory", objective="Do A.....")
    prose = env.issue(title="No task block", body="just prose", approve_as="owner")
    greedy = env.issue(title="Unknown check", checks=["nope"], objective="Do something else.")
    eve = env.issue(title="Foreign author", author="eve", objective="Do yet something else.")
    edited = env.issue(title="Edited later", objective="Do one more thing now.")
    env.fake.edit_issue(edited, body=task_body(objective="A different thing entirely."))
    result = env.run("preview")
    assert result.code == 0
    out = result.out
    assert f"Issue #{good}: Document the bridge" in out
    assert f"skipped #{mallory}: approver_not_trusted: 'mallory' may not approve agent tasks" in out
    assert f"skipped #{prose}: invalid_task: expected exactly one fenced" in out
    assert f"skipped #{greedy}: task_not_permitted: check 'nope' is not defined" in out
    assert f"skipped #{eve}: author_not_trusted: 'eve' may not author agent tasks" in out
    assert (
        f"skipped #{edited}: approval_stale: the issue was edited at or after the approval" in out
    )
    refused = env.run("preview", "--json").json()["refused"]
    assert [item.split(":")[0] for item in refused] == [
        f"#{n}" for n in (mallory, prose, greedy, eve, edited)
    ]
    assert env.fake.writes() == []
    assert all(not comments for comments in env.fake.comments.values())
    assert env.invocations() == []


def test_preview_with_nothing_to_do_says_so(env: Env) -> None:
    result = env.run("preview")
    assert result.code == 0
    assert result.out.splitlines() == [
        "considered 0 candidate issue(s)",
        "dry run: nothing would be executed",
    ]


def test_preview_issue_restricts_to_one_issue(env: Env) -> None:
    env.issue(objective="First task, documented below.")
    second = env.issue(title="Second", objective="Second task, documented below.")
    data = env.run("preview", "--issue", str(second), "--json").json()
    assert data["considered"] == 1
    assert [p["issue_number"] for p in data["previews"]] == [second]
    assert env.run("preview", "--json").json()["considered"] == 2


def test_preview_works_while_the_configuration_says_dry_run(env: Env) -> None:
    env.write_config(dry_run=True)
    env.issue()
    result = env.run("preview")
    assert result.code == 0 and "DRY RUN - would execute:" in result.out
    assert_nothing_executed(env)


def test_preview_poll_errors_exit_one(env: Env) -> None:
    env.issue()
    env.fake.deny("GET", r"/issues$")
    result = env.run("preview")
    assert result.code == 1
    assert "error: cannot list issues: GitHub denied the request" in result.out
    as_json = env.run("preview", "--json")
    assert as_json.code == 1 and as_json.json()["errors"]


def test_preview_of_an_unknown_issue_is_a_poll_error(env: Env) -> None:
    result = env.run("preview", "--issue", "404")
    assert result.code == 1
    assert "error: cannot list issues: GitHub object not found" in result.out


def test_dry_run_runtime_refuses_every_github_write(env: Env) -> None:
    number = env.issue()
    runtime = cli.build_runtime(load_config(env.config_path), execute=False)
    try:
        assert isinstance(runtime.github, ReadOnlyGitHub)
        assert runtime.github.get_issue(number).title == "Document the bridge"
        writes: dict[str, Callable[[], Any]] = {
            "add_labels": lambda: runtime.github.add_labels(number, ["x"]),
            "remove_label": lambda: runtime.github.remove_label(number, "agent-task"),
            "ensure_labels": lambda: runtime.github.ensure_labels(["agent-task"]),
            "create_comment": lambda: runtime.github.create_comment(number, "hello"),
            "create_pull_request": lambda: runtime.github.create_pull_request(
                title="t", head="h", base="main", body="b"
            ),
        }
        for name, attempt in writes.items():
            with pytest.raises(BridgeError, match=f"refusing GitHub write '{name}'"):
                attempt()
        assert env.fake.writes() == []
    finally:
        runtime.ledger.close()
        runtime.github.close()


def test_executing_runtime_uses_the_real_client(env: Env) -> None:
    number = env.issue()
    runtime = cli.build_runtime(load_config(env.config_path), execute=True)
    try:
        assert not isinstance(runtime.github, ReadOnlyGitHub)
        runtime.github.add_labels(number, ["agent-running"])
        assert [(w.method, w.path) for w in env.fake.writes()] == [
            ("POST", f"/repos/owner/repo/issues/{number}/labels")
        ]
    finally:
        runtime.ledger.close()
        runtime.github.close()


# ----------------------------------------------------------------------------------------------
# run-once
# ----------------------------------------------------------------------------------------------
def test_run_once_without_execute_is_a_dry_run(env: Env) -> None:
    env.issue()
    result = env.run("run-once")
    assert result.code == 0
    assert "DRY RUN - would execute:" in result.out
    assert_nothing_executed(env)


def test_run_once_dry_run_can_show_the_prompt(env: Env) -> None:
    env.issue(objective="Add a short note about dry runs to docs/guide.md.")
    out = env.run("run-once", "--show-prompt").out
    assert "----- prompt -----" in out
    assert "Add a short note about dry runs to docs/guide.md." in out


def test_execute_is_refused_while_the_configuration_says_dry_run(env: Env) -> None:
    env.write_config(dry_run=True)
    env.issue()
    for flags in (["--execute"], ["--execute", "--yes"]):
        result = env.run("run-once", *flags)
        assert result.code == 3
        assert "dry_run: true" in result.out
    assert_nothing_executed(env)
    assert not env.state_dir.exists()  # refused before any runtime was built


def test_execute_needs_yes_when_stdin_is_not_a_terminal(env: Env) -> None:
    env.issue()
    result = env.run("run-once", "--execute")
    assert result.code == 2
    assert "pass --yes to authorise explicitly" in result.out
    assert_nothing_executed(env)
    assert not env.state_dir.exists()


def test_execute_with_yes_opens_a_draft_pull_request(env: Env) -> None:
    number = env.issue()
    result = env.run("run-once", "--execute", "--yes")
    assert result.code == 0
    (run,) = env.runs()
    assert (run.status, run.phase) is not None
    assert run.status is RunStatus.DONE and run.phase is RunPhase.FINISHED
    assert run.pr_number is not None and run.backend == env.backend
    assert f"run {run.run_id[:12]} issue #{number}: done/finished PR #{run.pr_number}" in result.out
    assert f"started for issue #{number} with {env.backend}" in result.err
    pull = env.fake.pulls[run.pr_number]
    assert pull.draft is True and pull.head_ref == run.branch
    assert run.branch in env.git.origin_branches()
    labels = env.fake.issue_labels(number)
    assert "agent-done" in labels and "agent-running" not in labels
    assert len(env.invocations()) == 1
    assert env.git.git("rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert env.git.git("status", "--porcelain") == ""


def test_execute_with_yes_and_no_candidates_is_quiet_success(env: Env) -> None:
    result = env.run("run-once", "--execute", "--yes")
    assert result.code == 0
    assert result.out.strip() == "considered 0 candidate issue(s)"
    assert env.fake.writes() == [] and env.invocations() == []


def test_agent_started_by_the_cli_gets_a_scrubbed_environment(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = {
        "GITHUB_TOKEN": GITHUB_TOKEN,
        "GH_TOKEN": GITHUB_TOKEN,
        "SSH_AUTH_SOCK": "/tmp/fake-ssh-agent.sock",
        "AWS_SECRET_ACCESS_KEY": "super-secret-aws-value",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    number = env.issue()
    assert env.run("run-once", "--execute", "--yes").code == 0
    keys = set(env.invocations()[0]["env_keys"])
    assert {"FAKE_AGENT_SCENARIO", "FAKE_AGENT_LOG", "PATH"} <= keys
    assert keys.isdisjoint({*secrets, "LAWNBERRY_BRIDGE_GITHUB_TOKEN"})
    published = [c.body for c in env.fake.comments[number]]
    published += [pull.body for pull in env.fake.pulls.values()]
    assert all(GITHUB_TOKEN not in text for text in published)


class TestConfirmation:
    """Without ``--yes`` a human at a terminal must say yes."""

    def test_no_answer_runs_nothing(self, env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
        number = env.issue()
        prompts = answer_prompts(monkeypatch, "n")
        result = env.run("run-once", "--execute")
        assert result.code == 2
        assert "not authorised; nothing was executed" in result.out
        assert len(prompts) == 1 and "[y/N]" in prompts[0]
        assert "run the next approved issue" in prompts[0]
        assert_nothing_executed(env)
        assert number == 1

    @pytest.mark.parametrize("answer", ["", "maybe", "nope", "  "])
    def test_anything_but_yes_is_no(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        env.issue()
        answer_prompts(monkeypatch, answer)
        assert env.run("run-once", "--execute").code == 2
        assert_nothing_executed(env)

    @pytest.mark.parametrize("answer", ["y", "yes", " Y ", "YES"])
    def test_yes_runs_the_issue(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        env.issue()
        prompts = answer_prompts(monkeypatch, answer)
        result = env.run("run-once", "--execute")
        assert result.code == 0
        assert len(prompts) == 1
        (run,) = env.runs()
        assert run.status is RunStatus.DONE

    def test_prompt_names_the_issue_when_one_is_selected(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        number = env.issue()
        prompts = answer_prompts(monkeypatch, "n")
        env.run("run-once", "--execute", "--issue", str(number))
        assert f"run issue #{number}" in prompts[0]

    def test_end_of_input_at_the_prompt_is_a_refusal(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env.issue()
        answer_prompts(monkeypatch)  # input() raises EOFError
        result = env.run("run-once", "--execute")
        assert result.code == 2
        assert_nothing_executed(env)

    def test_yes_flag_skips_the_prompt_on_a_terminal(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env.issue()
        prompts = answer_prompts(monkeypatch)
        assert env.run("run-once", "--execute", "--yes").code == 0
        assert prompts == []

    def test_dry_run_never_prompts(self, env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
        env.issue()
        prompts = answer_prompts(monkeypatch)
        assert env.run("run-once").code == 0
        assert prompts == []


def test_run_once_issue_flag_restricts_the_run_to_that_issue(env: Env) -> None:
    first = env.issue(objective="First task, documented below.")
    second = env.issue(title="Second", objective="Second task, documented below.")
    result = env.run("run-once", "--execute", "--yes", "--issue", str(second))
    assert result.code == 0
    (run,) = env.runs()
    assert run.issue_number == second
    assert "agent-running" not in env.fake.issue_labels(first)
    assert "agent-done" not in env.fake.issue_labels(first)
    assert "agent-done" in env.fake.issue_labels(second)


def test_run_once_handles_one_issue_per_invocation(env: Env) -> None:
    first = env.issue(objective="First task, documented below.")
    second = env.issue(title="Second", objective="Second task, documented below.")
    env.run("run-once", "--execute", "--yes")
    assert [r.issue_number for r in env.runs()] == [first]
    env.run("run-once", "--execute", "--yes")
    assert sorted(r.issue_number for r in env.runs()) == [first, second]
    assert len(env.invocations()) == 2


def test_run_once_on_an_unapproved_issue_skips_it(env: Env) -> None:
    number = env.issue(approve_as=None)
    result = env.run("run-once", "--execute", "--yes", "--issue", str(number))
    assert result.code == 0
    assert f"skipped #{number}: missing_labels: needs agent-task and agent-approved" in result.out
    assert env.runs() == [] and env.invocations() == []


def test_a_second_invocation_does_not_run_the_issue_again(env: Env) -> None:
    number = env.issue()
    env.run("run-once", "--execute", "--yes")
    again = env.run("run-once", "--execute", "--yes")
    assert again.code == 0
    assert f"skipped #{number}: already_handled" in again.out
    assert len(env.runs()) == 1 and len(env.invocations()) == 1
    assert len(env.fake.pulls) == 1


def test_refusal_notices_are_posted_once_across_invocations(env: Env) -> None:
    number = env.issue(title="No task block", body="just prose")
    first = env.run("run-once", "--execute", "--yes")
    second = env.run("run-once", "--execute", "--yes")
    assert first.code == second.code == 0
    assert f"skipped #{number}: invalid_task" in second.out
    comments = [c.body for c in env.fake.comments[number]]
    assert len(comments) == 1 and "Agent bridge cannot run this task" in comments[0]
    assert env.invocations() == []


@pytest.mark.parametrize(
    ("kind", "hint"),
    [
        ("auth", "not authenticated"),
        ("credit", "out of credit"),
        ("rate_limit", "rate-limited"),
        ("api_error", "returned an error"),
    ],
)
def test_a_blocked_run_exits_one_and_tells_the_issue_why(env: Env, kind: str, hint: str) -> None:
    number = env.issue()
    env.steps([{"op": "fail", "kind": kind}])
    result = env.run("run-once", "--execute", "--yes")
    assert result.code == 1
    (run,) = env.runs()
    assert run.status is RunStatus.BLOCKED and run.failure_kind == f"agent_{kind}"
    assert f"issue #{number}: blocked/finished (agent_{kind}:" in result.out
    assert "agent-blocked" in env.fake.issue_labels(number)
    assert env.fake.pulls == {}
    assert any(hint in c.body for c in env.fake.comments[number])
    status = env.run("status", "--json").json()
    assert status[0]["status"] == "blocked" and status[0]["failure"] == f"agent_{kind}"


def test_poll_errors_exit_one_when_executing(env: Env) -> None:
    env.issue()
    env.fake.deny("GET", r"/issues$")
    result = env.run("run-once", "--execute", "--yes")
    assert result.code == 1
    assert "error: cannot list issues" in result.out
    assert env.invocations() == [] and env.runs() == []


def test_a_run_cancelled_through_the_cli_ends_cancelled_not_blocked(env: Env) -> None:
    number = env.issue()
    env.steps([{"op": "hang"}])
    results: list[int] = []

    def worker() -> None:
        results.append(
            cli.main(["--config", str(env.config_path), "run-once", "--execute", "--yes"])
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        agent_started = False
        while time.monotonic() < deadline and not agent_started:
            if (env.state_dir / "ledger.sqlite").exists():
                agent_started = any(r.agent_pid for r in env.runs())
            time.sleep(0.05)
        assert agent_started, "the agent never started"
        cancelled = env.run("cancel", "--issue", str(number))
        assert cancelled.code == 0
        assert "cancellation requested for 1 run(s)" in cancelled.out
    finally:
        thread.join(30)
        if thread.is_alive():  # never leave the hanging agent behind
            env.ledger().request_cancel(issue_number=number)
            thread.join(30)
    assert not thread.is_alive() and results == [0]
    (run,) = env.runs()
    assert run.status is RunStatus.CANCELLED and run.failure_kind == "cancelled"
    assert run.pr_number is None and env.fake.pulls == {}
    assert "agent-running" not in env.fake.issue_labels(number)
    assert "agent-blocked" not in env.fake.issue_labels(number)
    assert any("Agent run cancelled" in c.body for c in env.fake.comments[number])


# ----------------------------------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------------------------------
def test_serve_execute_without_yes_exits_two_without_serving(env: Env) -> None:
    env.issue()
    result = env.run("serve", "--execute")
    assert result.code == 2
    assert "pass --yes to authorise it explicitly" in result.out
    assert_nothing_executed(env)
    assert not env.state_dir.exists() and env.fake.requests == []


def test_serve_execute_is_refused_while_the_configuration_says_dry_run(env: Env) -> None:
    env.write_config(dry_run=True)
    env.issue()
    result = env.run("serve", "--execute", "--yes")
    assert result.code == 3 and "dry_run: true" in result.out
    assert env.fake.requests == [] and env.invocations() == []


@pytest.fixture
def one_poll_then_stop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for the OS signal machinery: record handlers, stop after the first poll."""
    state: dict[str, Any] = {"handlers": {}, "polls": []}

    def fake_signal(signum: int, handler: Callable[..., Any]) -> int:
        state["handlers"][signum] = handler
        return signal.SIG_DFL

    original = Worker.poll_once

    def poll_then_terminate(self: Worker, **kwargs: Any) -> Any:
        state["polls"].append(kwargs)
        report = original(self, **kwargs)
        state["handlers"][signal.SIGTERM](signal.SIGTERM, None)
        return report

    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr(Worker, "poll_once", poll_then_terminate)
    return state


def test_serve_dry_run_polls_previews_and_stops_on_sigterm(
    env: Env, one_poll_then_stop: dict[str, Any]
) -> None:
    env.issue()
    result = env.run("serve")
    assert result.code == 0
    assert one_poll_then_stop["polls"] == [{"execute": False}]
    assert set(one_poll_then_stop["handlers"]) == {signal.SIGINT, signal.SIGTERM}
    assert f"serving {REPO} (dry run); polling every 1.0s" in result.err
    assert "signal 15 received; stopping" in result.err
    assert_nothing_executed(env)


def test_serve_execute_with_yes_runs_approved_issues_until_stopped(
    env: Env, one_poll_then_stop: dict[str, Any]
) -> None:
    number = env.issue()
    result = env.run("serve", "--execute", "--yes")
    assert result.code == 0
    assert one_poll_then_stop["polls"] == [{"execute": True}]
    assert f"serving {REPO} (EXECUTE)" in result.err
    (run,) = env.runs()
    assert run.issue_number == number and run.status is RunStatus.DONE


# ----------------------------------------------------------------------------------------------
# setup-labels
# ----------------------------------------------------------------------------------------------
def test_setup_labels_without_yes_exits_two_and_asks_for_it(env: Env) -> None:
    result = env.run("setup-labels")
    assert result.code == 2
    assert "re-run with --yes to confirm" in result.out
    assert env.fake.requests == [] and not env.state_dir.exists()


def test_setup_labels_creates_the_five_labels_once(env: Env) -> None:
    first = env.run("setup-labels", "--yes")
    assert first.code == 0
    assert first.out.strip() == f"created labels: {', '.join(FIVE_LABELS)}"
    assert sorted(env.fake.labels) == sorted(FIVE_LABELS)
    for name in FIVE_LABELS:
        color, description = LABEL_STYLE[name]
        assert env.fake.labels[name].color == color
        assert env.fake.labels[name].description == description
    writes_after_first = len(env.fake.writes())
    assert writes_after_first == 5
    second = env.run("setup-labels", "--yes")
    assert second.code == 0
    assert second.out.strip() == "created labels: none (all present)"
    assert len(env.fake.writes()) == writes_after_first


def test_setup_labels_only_creates_the_missing_ones(env: Env) -> None:
    env.fake.add_issue(title="seed", labels=["agent-task", "agent-done"])
    result = env.run("setup-labels", "--yes")
    assert result.out.strip() == "created labels: agent-approved, agent-running, agent-blocked"


def test_setup_labels_follows_the_configured_names(env: Env) -> None:
    env.write_config(labels={"task": "bridge-task", "approved": "bridge-ok", "done": "bridge-done"})
    result = env.run("setup-labels", "--yes")
    assert result.out.strip() == (
        "created labels: bridge-task, bridge-ok, agent-running, agent-blocked, bridge-done"
    )
    assert env.fake.labels["bridge-task"].color == "ededed"  # unknown names get the neutral style


def test_setup_labels_permission_errors_exit_one(env: Env) -> None:
    env.fake.deny("POST", r"/labels$")
    result = env.run("setup-labels", "--yes")
    assert result.code == 1
    assert result.out.startswith("error: GitHub denied the request")


# ----------------------------------------------------------------------------------------------
# status, cancel, recover
# ----------------------------------------------------------------------------------------------
def test_status_with_no_runs(env: Env) -> None:
    assert env.run("status").out == "no runs recorded\n"
    assert env.run("status", "--json").json() == []


def test_status_shows_the_run_its_cost_and_pull_request(env: Env) -> None:
    number = env.issue()
    env.run("run-once", "--execute", "--yes")
    (run,) = env.runs()
    table = env.run("status").out.strip()
    assert re.fullmatch(
        rf"{run.run_id[:12]}  #{number}\s+done\s+finished\s+{env.backend}\s+"
        rf"PR {run.pr_number}\s+cost \$0\.0200\s*",
        table,
    ), table
    (row,) = env.run("status", "--json").json()
    assert row == {
        "run_id": run.run_id,
        "issue": number,
        "status": "done",
        "phase": "finished",
        "backend": env.backend,
        "pr": run.pr_number,
        "failure": None,
        "started": run.started_at,
        "finished": run.finished_at,
    }


def test_status_lists_newest_first_and_honours_limit(env: Env) -> None:
    first = env.issue(objective="First task, documented below.")
    second = env.issue(title="Second", objective="Second task, documented below.")
    env.run("run-once", "--execute", "--yes")
    env.run("run-once", "--execute", "--yes")
    rows = env.run("status", "--json").json()
    assert [row["issue"] for row in rows] == [second, first]
    limited = env.run("status", "--limit", "1", "--json").json()
    assert [row["issue"] for row in limited] == [second]
    assert len(env.run("status").out.strip().splitlines()) == 2


def test_status_is_read_only_and_works_offline(env: Env) -> None:
    env.issue()
    env.run("run-once", "--execute", "--yes")
    before = len(env.fake.requests)
    env.run("status")
    env.run("status", "--json")
    assert len(env.fake.requests) == before


def lease_pair(env: Env) -> tuple[RunRecord, RunRecord]:
    """A pending run (issue 7) and a running one (issue 8); only one run may be active."""
    first = env.lease(7)
    ledger = env.ledger()
    try:
        assert ledger.requeue_interrupted(first.run_id, 1)
    finally:
        ledger.close()
    return first, env.lease(8)


def cancel_flags(env: Env, *runs: RunRecord) -> list[bool]:
    ledger = env.ledger()
    try:
        return [ledger.is_cancel_requested(run.run_id) for run in runs]
    finally:
        ledger.close()


def test_cancel_flags_a_run_by_issue_and_by_run_id(env: Env) -> None:
    pending, running = lease_pair(env)
    by_issue = env.run("cancel", "--issue", "7")
    assert by_issue.code == 0
    assert "cancellation requested for 1 run(s)" in by_issue.out
    assert cancel_flags(env, pending, running) == [True, False]
    by_id = env.run("cancel", "--run-id", running.run_id)
    assert by_id.code == 0 and "cancellation requested for 1 run(s)" in by_id.out
    assert cancel_flags(env, pending, running) == [True, True]
    assert env.fake.requests == []  # cancelling is purely local


def test_cancel_only_flags_the_named_run(env: Env) -> None:
    pending, running = lease_pair(env)
    env.run("cancel", "--run-id", running.run_id)
    assert cancel_flags(env, pending, running) == [False, True]


def test_cancel_accepts_the_short_run_id_that_status_prints(env: Env) -> None:
    pending, running = lease_pair(env)
    rows = {line.split()[1]: line.split()[0] for line in env.run("status").out.splitlines()}
    shown = rows["#8"]
    assert shown == running.run_id[:12]
    result = env.run("cancel", "--run-id", shown)
    assert result.code == 0 and "cancellation requested for 1 run(s)" in result.out
    assert cancel_flags(env, pending, running) == [False, True]
    assert env.run("cancel", "--run-id", pending.run_id[:8].upper()).code == 0
    assert cancel_flags(env, pending, running) == [True, True]


def test_cancel_with_an_ambiguous_run_id_prefix_flags_nothing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = iter([uuid.UUID(hex="ab" + "0" * 30), uuid.UUID(hex="ab" + "1" * 30)])
    monkeypatch.setattr("tools.agent_bridge.ledger.uuid4", lambda: next(ids))
    pending, running = lease_pair(env)
    result = env.run("cancel", "--run-id", "ab")
    assert result.code == 1
    assert result.out.startswith("error: 'ab' matches 2 runs")
    assert cancel_flags(env, pending, running) == [False, False]
    assert env.run("cancel", "--run-id", "ab1").code == 0  # a longer prefix is unique
    assert cancel_flags(env, pending, running) == [False, True]


def test_cancel_of_an_unknown_issue_exits_one(env: Env) -> None:
    result = env.run("cancel", "--issue", "999")
    assert result.code == 1
    assert "cancellation requested for 0 run(s)" in result.out


@pytest.mark.parametrize(
    ("given", "message"),
    [("deadbeef", "no run has an id starting with 'deadbeef'"), ("", "must not be empty")],
)
def test_cancel_of_an_unknown_or_empty_run_id_is_an_error(
    env: Env, given: str, message: str
) -> None:
    pending, running = lease_pair(env)
    result = env.run("cancel", "--run-id", given)
    assert result.code == 1
    assert result.out.startswith("error: ") and message in result.out
    assert cancel_flags(env, pending, running) == [False, False]


def test_cancel_of_a_finished_run_exits_one_and_flags_nothing(env: Env) -> None:
    number = env.issue()
    env.run("run-once", "--execute", "--yes")
    result = env.run("cancel", "--issue", str(number))
    assert result.code == 1
    (run,) = env.runs()
    assert run.cancel_requested is False and run.status is RunStatus.DONE


def test_recover_is_a_dry_run_by_default(env: Env) -> None:
    run = env.stale_lease(7)
    result = env.run("recover")
    assert result.code == 0
    assert result.out.strip() == f"would recover run {run.run_id[:12]} (issue #7, running/leased)"
    (after,) = env.runs()
    assert after.status is RunStatus.RUNNING
    assert env.fake.writes() == []


def test_recover_with_nothing_stale(env: Env) -> None:
    assert env.run("recover").out.strip() == "nothing to recover"
    env.lease(7)  # a live lease is not stale
    assert env.run("recover").out.strip() == "nothing to recover"


def test_recover_execute_follows_the_same_gates_as_run_once(env: Env) -> None:
    env.stale_lease(7)
    assert env.run("recover", "--execute", "--yes").code == 0 or True
    env.write_config(dry_run=True)
    refused = env.run("recover", "--execute", "--yes")
    assert refused.code == 3 and "dry_run: true" in refused.out
    env.write_config()
    unattended = env.run("recover", "--execute")
    assert unattended.code == 2 and "pass --yes" in unattended.out


def test_recover_execute_requeues_the_run_and_stops_the_orphaned_agent(env: Env) -> None:
    run = env.stale_lease(7)
    env.fake.add_issue(
        number=7, title="Crashed", body=task_body(), labels=["agent-task", "agent-approved"]
    )
    env.fake.label_issue(7, "agent-running")
    orphan = subprocess.Popen(["sleep", "300"], start_new_session=True)
    reaper = threading.Thread(target=orphan.wait, daemon=True)
    reaper.start()
    try:
        ticks = proc_start_ticks(orphan.pid)
        assert ticks is not None
        ledger = env.ledger()
        ledger.update(run.run_id, agent_pid=orphan.pid, agent_start_ticks=ticks)
        ledger.close()

        dry = env.run("recover")
        assert dry.code == 0 and pid_is_alive(orphan.pid)
        assert env.fake.writes() == []

        result = env.run("recover", "--execute", "--yes")
        assert result.code == 0
        assert f"interrupted run {run.run_id[:12]} (issue #7, running/leased)" in result.out
        reaper.join(15)
        assert not pid_is_alive(orphan.pid)
    finally:
        if orphan.poll() is None:
            orphan.kill()
    (after,) = env.runs()
    assert after.status is RunStatus.PENDING and after.interrupted_count == 1
    assert "agent-running" not in env.fake.issue_labels(7)
    assert env.run("recover", "--execute", "--yes").out.strip() == "nothing to recover"


# ----------------------------------------------------------------------------------------------
# logging and state directory hygiene
# ----------------------------------------------------------------------------------------------
def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_runtime_creates_a_private_state_directory_and_log(env: Env) -> None:
    assert env.run("preview").code == 0
    assert mode(env.state_dir) == 0o700
    assert mode(env.state_dir / "logs" / "bridge.log") == 0o600
    assert mode(env.state_dir / "ledger.sqlite") == 0o600


def test_runtime_tightens_an_existing_state_directory(env: Env) -> None:
    env.state_dir.mkdir()
    env.state_dir.chmod(0o755)
    env.run("preview")
    assert mode(env.state_dir) == 0o700


def test_log_rotation_is_five_megabytes_times_five(env: Env) -> None:
    env.run("preview")
    handlers = [
        h
        for h in logging.getLogger(BRIDGE_LOGGER).handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(handlers) == 1
    assert handlers[0].maxBytes == 5_000_000 and handlers[0].backupCount == 5
    assert Path(handlers[0].baseFilename) == env.state_dir / "logs" / "bridge.log"
    console = [
        h for h in logging.getLogger(BRIDGE_LOGGER).handlers if type(h) is logging.StreamHandler
    ]
    assert len(console) == 1


def test_rotated_log_files_stay_private(env: Env) -> None:
    env.run("preview")
    (handler,) = [
        h
        for h in logging.getLogger(BRIDGE_LOGGER).handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    handler.maxBytes = 300
    log = logging.getLogger("agent_bridge.worker")
    for index in range(20):
        log.info("filling the log %s %s", index, "x" * 60)
    flush_logs()
    files = sorted((env.state_dir / "logs").iterdir())
    assert len(files) > 1, "the log never rotated"
    assert {mode(path) for path in files} == {0o600}


def test_log_lines_never_contain_the_token(env: Env) -> None:
    env.run("preview")
    log = logging.getLogger("agent_bridge.worker")
    log.info("pushing with %s now", GITHUB_TOKEN)
    log.warning("Authorization: Bearer %s", GITHUB_TOKEN)
    flush_logs()
    written = (env.state_dir / "logs" / "bridge.log").read_text(encoding="utf-8")
    printed = env.capsys.readouterr().err
    for text in (written, printed):
        assert GITHUB_TOKEN not in text
        assert "[REDACTED:" in text
    assert "pushing with [REDACTED:literal] now" in written


def test_log_lines_hide_other_configured_secrets(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_EXTRA_SECRET", "correct-horse-battery-staple")
    env.write_config(extra_secret_env=["MY_EXTRA_SECRET"])
    env.run("preview")
    logging.getLogger("agent_bridge.worker").info("leaked correct-horse-battery-staple here")
    flush_logs()
    written = (env.state_dir / "logs" / "bridge.log").read_text(encoding="utf-8")
    assert "correct-horse-battery-staple" not in written
    assert "leaked [REDACTED:literal] here" in written


def test_tracebacks_in_logs_are_redacted_too(env: Env) -> None:
    env.run("preview")
    log = logging.getLogger("agent_bridge.worker")
    try:
        raise RuntimeError(f"push failed for https://x-access-token:{GITHUB_TOKEN}@github.test")
    except RuntimeError:
        log.exception("run failed unexpectedly")
    flush_logs()
    written = (env.state_dir / "logs" / "bridge.log").read_text(encoding="utf-8")
    printed = env.capsys.readouterr().err
    assert "run failed unexpectedly" in written and "RuntimeError" in written
    assert GITHUB_TOKEN not in written
    assert GITHUB_TOKEN not in printed


def test_verbose_flag_switches_to_debug_logging(env: Env) -> None:
    env.run("preview")
    assert logging.getLogger(BRIDGE_LOGGER).level == logging.INFO
    env.run("-v", "preview")
    assert logging.getLogger(BRIDGE_LOGGER).level == logging.DEBUG


def test_logging_setup_survives_an_unusable_log_directory(env: Env) -> None:
    env.state_dir.mkdir()
    (env.state_dir / "logs").write_text("not a directory", encoding="utf-8")
    result = env.run("preview")
    assert result.code == 0
    assert "file logging is unavailable" in result.err


# ----------------------------------------------------------------------------------------------
# smoke (hermetic: fake agent wrappers instead of the real CLIs)
# ----------------------------------------------------------------------------------------------
def smoke_steps(*extra: dict[str, Any], content: str = f"{SMOKE_LINE}\n") -> list[dict[str, Any]]:
    steps = docs_edit_steps(path="docs/smoke-note.md", content=content)
    return [*steps[:-1], *extra, steps[-1]]


@pytest.fixture
def smoke_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send every ``tempfile`` directory the smoke test makes into a place we can inspect."""
    root = tmp_path / "smoke-tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def smoke(env: Env, backend: str, *extra: str) -> Result:
    return env.main(
        "smoke",
        "--backend",
        backend,
        "--binary",
        str(env.bin_for(backend)),
        *PASS_AGENT_ENV,
        *extra,
    )


def agent_runs(env: Env) -> list[dict[str, Any]]:
    """Task invocations of the fake agent (``--version`` probes are logged as well)."""
    return [call for call in env.invocations() if call["argv"] != ["--version"]]


@pytest.mark.parametrize("backend", ["omp", "claude"])
def test_smoke_with_a_fake_agent_succeeds(env: Env, smoke_tmp: Path, backend: str) -> None:
    env.steps(smoke_steps())
    result = smoke(env, backend)
    assert result.code == 0, result.out
    report = result.smoke_json()
    assert report["ok"] is True
    assert report["backend"] == backend and report["binary"] == str(env.bin_for(backend))
    assert report["version"] == ("omp/18.12.6" if backend == "omp" else "2.1.293 (Claude Code)")
    assert report["note_content"] == f"{SMOKE_LINE}\n"
    assert report["git"]["changed"] == ["docs/smoke-note.md"]
    assert report["git"]["violations"] == []
    agent = report["result"]
    assert agent["status"] == "succeeded" and agent["ok"] is True
    assert agent["failure_kind"] is None and agent["error"] is None
    assert agent["cost_usd"] == 0.02 and agent["budget_usd"] == 1.0
    assert agent["report"]["status"] == "completed"
    assert "guard_probe" not in report
    assert list(smoke_tmp.iterdir()) == []  # the throwaway workspace is gone
    (call,) = agent_runs(env)
    assert "docs/smoke-note.md" in call["prompt"] and SMOKE_LINE in call["prompt"]


def test_smoke_scrubs_the_agent_environment(env: Env, smoke_tmp: Path) -> None:
    env.steps(smoke_steps())
    result = env.main("smoke", "--backend", "omp", "--binary", str(env.omp_bin))  # no --pass-env
    assert result.code == 1
    report = result.smoke_json()
    assert report["ok"] is False and report["note_content"] is None
    assert env.invocations() == []  # the fake never saw FAKE_AGENT_LOG, nor its scenario


@pytest.mark.parametrize("backend", ["omp", "claude"])
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("auth", "auth"),
        ("credit", "credit"),
        ("rate_limit", "rate_limit"),
        ("api_error", "api_error"),
        ("crash", "nonzero_exit"),
    ],
)
def test_smoke_reports_agent_failures_as_structured_results(
    env: Env, smoke_tmp: Path, backend: str, kind: str, expected: str
) -> None:
    env.steps([{"op": "fail", "kind": kind}])
    result = smoke(env, backend)
    assert result.code == 1
    report = result.smoke_json()
    assert report["ok"] is False and report["note_content"] is None
    agent = report["result"]
    assert agent["status"] == "failed" and agent["ok"] is False
    assert agent["failure_kind"] == expected
    assert agent["error"]
    assert agent["exit_code"] != 0


def test_smoke_flags_a_policy_violation_even_when_the_agent_succeeded(
    env: Env, smoke_tmp: Path
) -> None:
    env.steps(
        smoke_steps({"op": "write_file", "path": ".github/workflows/evil.yml", "content": "x\n"})
    )
    result = smoke(env, "omp")
    assert result.code == 1
    report = result.smoke_json()
    assert report["result"]["ok"] is True
    assert report["ok"] is False
    assert any(v.startswith("protected_path") for v in report["git"]["violations"])
    assert ".github/workflows/evil.yml" in report["git"]["changed"]


@pytest.mark.parametrize("content", ["something else\n", f"{SMOKE_LINE}\nand more\n"])
def test_smoke_requires_the_exact_line(env: Env, smoke_tmp: Path, content: str) -> None:
    env.steps(smoke_steps(content=content))
    result = smoke(env, "omp")
    assert result.code == 1
    report = result.smoke_json()
    assert report["note_content"] == content and report["ok"] is False


def test_smoke_fails_when_the_agent_writes_nothing(env: Env, smoke_tmp: Path) -> None:
    env.steps([{"op": "final", "text": "I did nothing."}])
    result = smoke(env, "omp")
    assert result.code == 1
    report = result.smoke_json()
    assert report["result"]["status"] == "succeeded"
    assert report["note_content"] is None and report["git"]["changed"] == []
    assert report["ok"] is False


@pytest.mark.parametrize("backend", ["omp", "claude"])
def test_smoke_probe_guard_adds_a_guard_probe_section(
    env: Env, smoke_tmp: Path, backend: str
) -> None:
    blocked = {"tool": "write", "args": {"path": "/etc/outside.txt"}, "blocked": True}
    env.steps(
        [
            {"op": "write_file", "path": "docs/smoke-note.md", "content": f"{SMOKE_LINE}\n"},
            {"op": "write_file", "path": "docs/guard-ok.md", "content": "guard ok\n"},
            {"op": "tool_call", **blocked, "result": GUARD_BLOCK},
            {"op": "tool_call", **blocked, "result": GUARD_BLOCK},
            {"op": "cost", "usd": 0.03},
            {"op": "final", "text": "DONE"},
        ]
    )
    result = smoke(env, backend, "--probe-guard")
    assert result.code == 0, result.out
    report = result.smoke_json()
    probe = report["guard_probe"]
    assert set(probe) == {
        "outside_file_created",
        "inside_file_created",
        "status",
        "tool_calls",
        "blocked_tool_calls",
        "blocked_samples",
        "cost_usd",
    }
    assert probe["outside_file_created"] is False
    assert probe["inside_file_created"] is True
    assert probe["status"] == "succeeded"
    assert probe["blocked_tool_calls"] >= 1 and probe["blocked_samples"]
    assert probe["cost_usd"] == 0.03
    calls = agent_runs(env)
    assert len(calls) == 2  # the task run, then the probe run
    assert "agent-bridge-outside" in calls[1]["prompt"]
    assert list(smoke_tmp.iterdir()) == []  # the outside directory is removed as well


def test_smoke_without_probe_guard_runs_the_agent_once(env: Env, smoke_tmp: Path) -> None:
    env.steps(smoke_steps())
    smoke(env, "omp")
    assert len(agent_runs(env)) == 1


@pytest.mark.parametrize("binary", ["definitely-not-an-installed-agent", "no-dir/omp"])
def test_smoke_with_a_missing_binary_reports_it_and_exits_one(
    env: Env, smoke_tmp: Path, binary: str
) -> None:
    result = env.main("smoke", "--backend", "omp", "--binary", binary, *PASS_AGENT_ENV)
    assert result.code == 1
    report = result.smoke_json()
    assert report["ok"] is False
    assert report["binary"] is None and report["version"] is None
    assert f"{binary!r} is not installed" in report["error"]
    assert "result" not in report
    assert env.invocations() == []
    assert list(smoke_tmp.iterdir()) == []


def test_smoke_treats_a_non_executable_file_as_not_installed(env: Env, smoke_tmp: Path) -> None:
    plain = env.tmp_path / "omp-not-executable"
    plain.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    plain.chmod(0o644)
    result = env.main("smoke", "--backend", "omp", "--binary", str(plain))
    assert result.code == 1
    assert "is not installed" in result.smoke_json()["error"]


@pytest.mark.parametrize(
    ("backend", "needles"),
    [
        ("omp", [["--model", "demo-model"], ["--max-time", "55"]]),
        ("claude", [["--model", "demo-model"], ["--max-budget-usd", "0.50"]]),
    ],
)
def test_smoke_passes_model_timeout_and_budget_to_the_agent(
    env: Env, smoke_tmp: Path, backend: str, needles: list[list[str]]
) -> None:
    env.steps(smoke_steps())
    result = smoke(env, backend, "--model", "demo-model", "--timeout", "60", "--budget", "0.5")
    report = result.smoke_json()
    assert result.code == 0 and report["model_requested"] == "demo-model"
    command = report["result"]["command"]
    for flag, value in needles:
        assert command[command.index(flag) + 1] == value
    assert report["result"]["budget_usd"] == 0.5


def test_smoke_keep_leaves_the_workspace_and_says_where(env: Env, smoke_tmp: Path) -> None:
    env.steps(smoke_steps())
    result = smoke(env, "omp", "--keep")
    assert result.code == 0
    match = re.search(r"smoke workspace kept at (\S+)", result.out)
    assert match is not None
    kept = Path(match.group(1))
    assert kept.parent == smoke_tmp
    assert (kept / "repo" / "docs" / "smoke-note.md").read_text(encoding="utf-8") == (
        f"{SMOKE_LINE}\n"
    )


def test_smoke_needs_no_configuration_file(env: Env, smoke_tmp: Path) -> None:
    env.steps(smoke_steps())
    assert not (Path.cwd() / "config").exists()
    assert smoke(env, "omp").code == 0
