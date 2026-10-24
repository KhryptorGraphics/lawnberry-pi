"""Test rig: a real git origin/clone, a fake GitHub, fake agent CLIs and a wired-up Worker."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tools.agent_bridge.agents import build_backends
from tools.agent_bridge.config import BridgeConfig
from tools.agent_bridge.github import HttpGitHubClient
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.ledger import Ledger, RunRecord
from tools.agent_bridge.publisher import Publisher
from tools.agent_bridge.redact import Redactor
from tools.agent_bridge.worker import PollReport, Worker

from .agent_fixtures import (
    docs_edit_steps,
    install_fake_cli,
    read_invocations,
    write_scenario,
)
from .github_server import FakeGitHub

GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
REPO = "owner/repo"


class FakeClock:
    """Controllable ledger clock."""

    def __init__(self) -> None:
        self.current = datetime.now(UTC).replace(microsecond=0)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def git_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "LC_ALL": "C",
    }


def run_git(cwd: Path, *args: str, env: Mapping[str, str]) -> str:
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=dict(env), capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


BASE_FILES: dict[str, str] = {
    "README.md": "# Test repo\n",
    "CLAUDE.md": "# Guide\nUse small commits.\n",
    ".github/copilot-instructions.md": "# Copilot\nFollow the style guide.\n",
    "docs/index.md": "# Docs\n",
    "docs/guide.md": "# Guide\n\nOriginal text.\n",
    "backend/src/safety/estop.py": "ESTOP = True\n",
    "backend/src/services/thing.py": "VALUE = 1\n",
    "config/hardware.yaml": "motor: robohat\n",
    "tools/agent_bridge/placeholder.py": "x = 1\n",
    "tests/test_placeholder.py": "def test_ok():\n    assert True\n",
}


@dataclass
class GitRepoEnv:
    origin: Path
    clone: Path
    env: dict[str, str]
    home: Path

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return run_git(cwd or self.clone, *args, env=self.env)

    def origin_git(self, *args: str) -> str:
        return run_git(self.origin, *args, env=self.env)

    def origin_branches(self) -> list[str]:
        out = self.origin_git("for-each-ref", "--format=%(refname:short)", "refs/heads")
        return sorted(out.split())

    def push_main_commit(self, relpath: str, content: str, message: str = "update") -> str:
        """Add a commit to origin/main from the clone (used to move the base forward)."""
        target = self.clone / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        self.git("push", "-q", "origin", "main")
        return self.git("rev-parse", "HEAD")


def make_git_env(tmp_path: Path, files: Mapping[str, str] | None = None) -> GitRepoEnv:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = git_env(home)
    origin = tmp_path / "origin.git"
    clone = tmp_path / "clone"
    run_git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin), env=env)
    run_git(tmp_path, "clone", "-q", str(origin), str(clone), env=env)
    for rel, content in (files or BASE_FILES).items():
        target = clone / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    run_git(clone, "add", "-A", env=env)
    run_git(clone, "commit", "-q", "-m", "initial", env=env)
    run_git(clone, "push", "-q", "-u", "origin", "main", env=env)
    return GitRepoEnv(origin=origin, clone=clone, env=env, home=home)


def task_yaml(
    objective: str = "Add a short note about the agent bridge to docs/guide.md.",
    *,
    scope: str = "docs-only",
    **extra: Any,
) -> str:
    """The fenced ``agent-task`` block for an issue body."""
    import yaml

    data: dict[str, Any] = {"version": 1, "objective": objective, "scope": scope, **extra}
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()
    return f"```yaml agent-task\n{dumped}\n```"


def task_body(*args: Any, prose: str = "Please do this.", **kwargs: Any) -> str:
    return f"{prose}\n\n{task_yaml(*args, **kwargs)}\n"


def default_config(
    git: GitRepoEnv,
    tmp_path: Path,
    backend: str,
    omp_bin: Path,
    claude_bin: Path,
    **overrides: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "repository": REPO,
        "repo_path": str(git.clone),
        "state_dir": str(tmp_path / "state"),
        "dry_run": False,
        "approvers": ["owner"],
        "github": {
            "api_url": "https://api.github.test",
            "max_retries": 2,
            "retry_backoff_seconds": 0.0,
            "rate_limit_max_wait_seconds": 5,
        },
        "git": {"git_timeout_seconds": 60, "keep_worktrees": "on-failure"},
        "poll": {"interval_seconds": 1},
        "agent": {
            "default": backend,
            "kill_grace_seconds": 1,
            "cancel_poll_seconds": 0.1,
            "omp": {
                "binary": str(omp_bin),
                "pass_env": ["FAKE_AGENT_SCENARIO", "FAKE_AGENT_LOG"],
            },
            "claude": {
                "binary": str(claude_bin),
                "pass_env": ["FAKE_AGENT_SCENARIO", "FAKE_AGENT_LOG"],
            },
        },
        "limits": {
            "budget_usd": 1.0,
            "max_budget_usd": 2.0,
            "timeout_minutes": 1,
            "max_timeout_minutes": 5,
            "lease_ttl_seconds": 30,
            "publish_retry_delays": [0.0, 0.0],
            "max_publish_attempts": 3,
        },
        "scopes": {"allowed": ["docs-only", "code"]},
        "checks": {
            "pass": {"argv": [sys.executable, "-c", "print('check ok')"]},
            "fail": {"argv": [sys.executable, "-c", "import sys; print('nope'); sys.exit(3)"]},
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = _merge(data[key], value)
        else:
            data[key] = value
    return data


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class Rig:
    cfg: BridgeConfig
    git_env: GitRepoEnv
    fake: FakeGitHub
    client: HttpGitHubClient
    ledger: Ledger
    git: GitOps
    worker: Worker
    clock: FakeClock
    scenario: Path
    log: Path
    backend: str
    tmp_path: Path
    redactor: Redactor
    agent_environ: dict[str, str]
    spawned: list[list[str]] = field(default_factory=list)

    # ------------------------------------------------------------ scenario & agent
    def set_scenario(
        self, steps: Sequence[dict[str, Any]], *, ignore_sigterm: bool = False
    ) -> None:
        write_scenario(
            self.scenario,
            list(steps),
            log_env=[],
            ignore_sigterm=ignore_sigterm,
        )

    def invocations(self) -> list[dict[str, Any]]:
        return read_invocations(self.log)

    def docs_scenario(self, **kwargs: Any) -> None:
        self.set_scenario(docs_edit_steps(**kwargs))

    # ------------------------------------------------------------ GitHub
    def new_task_issue(
        self,
        *,
        title: str = "Document the bridge",
        body: str | None = None,
        author: str = "owner",
        approve_as: str | None = "owner",
        **spec: Any,
    ) -> int:
        text = body if body is not None else task_body(**spec)
        number = self.fake.add_issue(title=title, body=text, author=author, labels=["agent-task"])
        if approve_as is not None:
            self.fake.label_issue(number, "agent-approved", actor=approve_as)
        return number

    # ------------------------------------------------------------ running
    def poll(self, *, execute: bool = True, only_issue: int | None = None) -> PollReport:
        return self.worker.poll_once(execute=execute, only_issue=only_issue)

    def run_issue(self, number: int) -> RunRecord:
        report = self.poll(only_issue=number)
        assert report.runs, f"no run started: {[r.render() for r in report.refused]} {report.errors}"
        return report.runs[0]

    def comments(self, number: int) -> list[str]:
        return [c.body for c in self.fake.comments.get(number, [])]

    def worktrees(self) -> list[Path]:
        root = self.cfg.resolved_state_dir() / "worktrees"
        return sorted(root.iterdir()) if root.exists() else []

    def close(self) -> None:
        self.client.close()
        self.ledger.close()


def make_rig(
    tmp_path: Path,
    backend: str,
    *,
    git: GitRepoEnv | None = None,
    sleep: Callable[[float], None] | None = None,
    **config_overrides: Any,
) -> Rig:
    git = git or make_git_env(tmp_path)
    bin_dir = tmp_path / "bin"
    omp_bin = install_fake_cli(bin_dir, "omp")
    claude_bin = install_fake_cli(bin_dir, "claude")
    data = default_config(git, tmp_path, backend, omp_bin, claude_bin, **config_overrides)
    cfg = BridgeConfig.model_validate(data)
    fake = FakeGitHub(REPO, token=GITHUB_TOKEN)
    client = HttpGitHubClient(
        cfg.github,
        cfg.repository,
        GITHUB_TOKEN,
        transport=fake.transport(),
        sleep=lambda _s: None,
    )
    clock = FakeClock()
    state_dir = cfg.resolved_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(state_dir / "ledger.sqlite", clock=clock)
    redactor = Redactor([GITHUB_TOKEN])
    scenario = tmp_path / "scenario.json"
    log = tmp_path / "invocations.jsonl"
    agent_environ = {
        **os.environ,
        "FAKE_AGENT_SCENARIO": str(scenario),
        "FAKE_AGENT_LOG": str(log),
        "GITHUB_TOKEN": GITHUB_TOKEN,
        "GH_TOKEN": GITHUB_TOKEN,
        "LAWNBERRY_BRIDGE_GITHUB_TOKEN": GITHUB_TOKEN,
        "SSH_AUTH_SOCK": "/tmp/fake-ssh-agent.sock",
        "AWS_SECRET_ACCESS_KEY": "super-secret-aws-value",
    }
    gitops = GitOps(
        git.clone,
        cfg.git,
        state_dir=state_dir,
        redactor=redactor,
        environ={"PATH": os.environ.get("PATH", ""), "HOME": str(git.home), "LC_ALL": "C"},
    )
    backends = build_backends(cfg.agent, redactor=redactor, environ=agent_environ)
    worker = Worker(
        cfg,
        github=client,
        ledger=ledger,
        git=gitops,
        backends=backends,
        redactor=redactor,
        sleep=sleep or (lambda _s: None),
        worker_id="test-worker",
        environ={"PATH": os.environ.get("PATH", ""), "HOME": str(git.home)},
    )
    rig = Rig(
        cfg=cfg,
        git_env=git,
        fake=fake,
        client=client,
        ledger=ledger,
        git=gitops,
        worker=worker,
        clock=clock,
        scenario=scenario,
        log=log,
        backend=backend,
        tmp_path=tmp_path,
        redactor=redactor,
        agent_environ=agent_environ,
    )
    rig.docs_scenario()
    return rig


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def publisher_for(rig: Rig) -> Publisher:
    return Publisher(rig.client, rig.cfg, rig.redactor)
