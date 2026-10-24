"""Check isolation (C-02): what a check may inherit and what it must leave alone.

The bridge runs the configured checks (``pytest tests/unit``...) in the task worktree.  In
``code`` scope those tests are agent-written, so a check is agent-written code running as the
workstation user, and the same holds for the allow-listed ``bash`` command of an omp run.  The
bridge cannot sandbox that without an operating-system sandbox (``agent.command_prefix`` is the
hook for one).  What it does:

* a check starts with a scrubbed environment: an empty private ``HOME`` that disappears with the
  check, a pinned ``PATH``, no inherited credentials of any kind;
* the shared git state that decides what is published (the repository's config, the worktree's
  ``.git`` pointer and HEAD, the agent branches) is fingerprinted when the worktree is created
  and compared after the agent phase and after the checks; any difference blocks the run
  (``worktree_tampered``) before anything is committed or pushed;
* the new commit must sit directly on the base commit.

Real git repositories and the real worker flow throughout; the "attacker" is a check script, or
a fake agent that writes outside its worktree.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.rig import GitRepoEnv, Rig, make_git_env, run_git

from tools.agent_bridge.checks import run_checks
from tools.agent_bridge.config import CheckConfig, GitConfig
from tools.agent_bridge.errors import CheckAborted, GitError
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.ledger import RunRecord
from tools.agent_bridge.models import RunStatus
from tools.agent_bridge.process import SubprocessRunner
from tools.agent_bridge.redact import Redactor

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or sys.platform == "win32",
    reason="needs git and POSIX file modes",
)

PINNED_PATH = ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
CANARY = "PRIVATE-DEMO-KEY-CONTENT"
DOCS_NOTE = "docs/agent-note.md"

# Secret-shaped strings are assembled from fragments so that this file never contains one.
POISON = {
    "GITHUB_TOKEN": "ghp_" + "P0i5on1111111111111111111111111111111111",
    "GH_TOKEN": "ghp_" + "P0i5on2222222222222222222222222222222222",
    "LAWNBERRY_BRIDGE_GITHUB_TOKEN": "ghp_" + "P0i5on3333333333333333333333333333333333",
    "ANTHROPIC_API_KEY": "sk-ant-" + "poison-anthropic-key-0123456789",
    "OPENAI_API_KEY": "poison-openai-key-0123456789",
    "AWS_SECRET_ACCESS_KEY": "poison-aws-secret-0123456789",
    "SSH_AUTH_SOCK": "/tmp/poison-ssh-agent.sock",
    "GIT_ASKPASS": "/usr/bin/poison-askpass",
    "GIT_SSH_COMMAND": "ssh -i /home/poison/.ssh/id_poison",
    "NPM_TOKEN": "poison-npm-token-0123456789",
    "PYTHONPATH": "/home/poison/pythonpath",
    "LD_PRELOAD": "/home/poison/evil.so",
    "BASH_ENV": "/home/poison/bash_env",
    "XDG_CONFIG_HOME": "/home/poison/.config",
    "XDG_CACHE_HOME": "/home/poison/.cache",
    "XDG_DATA_HOME": "/home/poison/.local/share",
    "TMPDIR": "/home/poison/tmp",
    "USER": "poison-user",
    "LOGNAME": "poison-user",
}
SCRUBBED_KEYS = {
    "PATH",
    "HOME",
    "TMPDIR",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_TERMINAL_PROMPT",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
}


# ----------------------------------------------------------------------------------------------
# the environment of a check (run_checks on its own)
# ----------------------------------------------------------------------------------------------
ENV_REPORT = """
import json, os, stat, tempfile
home = os.environ["HOME"]
report = {
    "env": dict(os.environ),
    "cwd": os.getcwd(),
    "home_is_dir": os.path.isdir(home),
    "home_listing": sorted(os.listdir(home)),
    "home_mode": stat.S_IMODE(os.stat(home).st_mode),
    "tmpdir_is_dir": os.path.isdir(os.environ["TMPDIR"]),
    "gettempdir": tempfile.gettempdir(),
    "canary": os.path.exists(os.path.expanduser("~/.ssh/id_demo")),
}
open("report.json", "w").write(json.dumps(report))
"""


def check(script: str, **fields: Any) -> CheckConfig:
    return CheckConfig(argv=[sys.executable, "-S", "-c", textwrap.dedent(script)], **fields)


def go(
    names: list[str],
    checks: dict[str, CheckConfig],
    worktree: Path,
    environ: dict[str, str] | None,
    **kwargs: Any,
):
    return run_checks(
        names,
        checks,
        worktree,
        runner=SubprocessRunner(),
        redactor=Redactor(),
        log_dir=worktree.parent / f"{worktree.name}-logs",
        environ=environ,
        **kwargs,
    )


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    path = tmp_path / "worktree"
    path.mkdir()
    return path


def report_of(worktree: Path) -> dict[str, Any]:
    return json.loads((worktree / "report.json").read_text(encoding="utf-8"))


def test_a_check_sees_nothing_but_the_scrubbed_environment(worktree: Path) -> None:
    environ = {**POISON, "PATH": "/home/poison/bin:/usr/bin", "HOME": "/home/real", "LANG": "x.y"}
    cfg_env = {"CHECK_EXTRA": "from config", "PYTHONCOERCECLOCALE": "0"}

    (result,) = go(["env"], {"env": check(ENV_REPORT, env=cfg_env)}, worktree, environ)

    assert result.passed, result.output_tail
    seen = report_of(worktree)["env"]
    assert set(seen) == SCRUBBED_KEYS | set(cfg_env)
    home = Path(seen["HOME"])
    assert str(home) != "/home/real"
    assert not [v for v in POISON.values() if v in json.dumps(seen)]
    assert seen["USER"] == seen["LOGNAME"] == "agent-bridge"
    assert seen["LANG"] == "x.y"  # the locale is the one thing that is kept
    assert seen["CHECK_EXTRA"] == "from config"
    assert seen["GIT_CONFIG_GLOBAL"] == seen["GIT_CONFIG_SYSTEM"] == "/dev/null"
    assert seen["GIT_CONFIG_NOSYSTEM"] == "1" and seen["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["PYTHONDONTWRITEBYTECODE"] == "1" and seen["PYTHONNOUSERSITE"] == "1"
    for name in ("TMPDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        assert Path(seen[name]).is_relative_to(home), name


def test_the_path_is_pinned_to_the_system_directories_and_the_bridge_interpreter(
    worktree: Path,
) -> None:
    environ = {"PATH": "/home/poison/bin:/usr/bin:.:relative/bin"}

    (result,) = go(["env"], {"env": check(ENV_REPORT)}, worktree, environ)

    assert result.passed, result.output_tail
    entries = report_of(worktree)["env"]["PATH"].split(":")
    assert [entry for entry in entries if entry in PINNED_PATH] == PINNED_PATH
    assert set(entries) - set(PINNED_PATH) <= {os.path.dirname(sys.executable)}
    assert os.path.dirname(sys.executable) in entries


def test_a_tool_next_to_the_bridge_interpreter_resolves_whatever_the_inherited_path(
    worktree: Path,
) -> None:
    checks = {"tool": CheckConfig(argv=[Path(sys.executable).name, "-S", "-c", "print('ok')"])}

    (result,) = go(["tool"], checks, worktree, {"PATH": "/nonexistent"})

    assert result.passed, result.output_tail


def test_the_home_is_private_empty_and_not_the_real_one(worktree: Path, tmp_path: Path) -> None:
    real_home = tmp_path / "real-home"
    (real_home / ".ssh").mkdir(parents=True)
    (real_home / ".ssh" / "id_demo").write_text(CANARY, encoding="utf-8")

    (result,) = go(["env"], {"env": check(ENV_REPORT)}, worktree, {"HOME": str(real_home)})

    assert result.passed, result.output_tail
    report = report_of(worktree)
    assert report["canary"] is False
    assert report["home_is_dir"] and report["tmpdir_is_dir"]
    assert report["home_listing"] == [".cache", ".config", ".local", "tmp"]
    assert report["home_mode"] == 0o700
    assert Path(report["gettempdir"]).is_relative_to(report["env"]["HOME"])
    assert not Path(report["env"]["HOME"]).is_relative_to(real_home)
    assert report["cwd"] == str(worktree)  # the checks need the files: the cwd stays the worktree


def test_configured_values_still_apply_and_win_over_the_scrubbed_ones(worktree: Path) -> None:
    cfg_env = {"HOME": "/configured/home", "MY_API_TOKEN": "configured-explicitly"}
    dump = "import json, os; open('env.json', 'w').write(json.dumps(dict(os.environ)))"

    (result,) = go(["env"], {"env": check(dump, env=cfg_env)}, worktree, {})

    assert result.passed, result.output_tail
    seen = json.loads((worktree / "env.json").read_text(encoding="utf-8"))
    assert seen["HOME"] == "/configured/home" and seen["MY_API_TOKEN"] == "configured-explicitly"


def test_lang_is_not_invented_and_the_process_environment_is_the_default(
    worktree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (without,) = go(["env"], {"env": check(ENV_REPORT)}, worktree, {"PATH": "/usr/bin"})
    assert without.passed, without.output_tail
    assert "LANG" not in report_of(worktree)["env"]

    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("BRIDGE_TEST_CANARY", "leak")
    (inherited,) = go(["env"], {"env": check(ENV_REPORT)}, worktree, None)
    assert inherited.passed, inherited.output_tail
    seen = report_of(worktree)["env"]
    assert seen["LANG"] == "C.UTF-8" and "BRIDGE_TEST_CANARY" not in seen


HOME_LITTER = """
import os, pathlib, stat
home = pathlib.Path(os.environ['HOME'])
pathlib.Path('home.txt').write_text(str(home))
(home / '.cache' / 'pip').mkdir(parents=True)
(home / '.cache' / 'pip' / 'wheel').write_text('cached')
(home / 'readonly').mkdir()
(home / 'readonly' / 'file').write_text('x')
os.chmod(home / 'readonly', stat.S_IRUSR | stat.S_IXUSR)
"""
OUTCOMES = {
    "passing": "pass",
    "failing": "import sys; sys.exit(3)",
    "timing-out": "import time; time.sleep(60)",
}


@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_the_home_is_removed_whatever_the_check_did(worktree: Path, outcome: str) -> None:
    script = HOME_LITTER + OUTCOMES[outcome] + "\n"

    (result,) = go(["home"], {"home": check(script, timeout_seconds=1.5)}, worktree, {})

    assert result.passed is (outcome == "passing")
    home = Path((worktree / "home.txt").read_text(encoding="utf-8"))
    assert not home.exists(), "the check's private HOME outlived the check"


def test_the_home_is_removed_when_a_check_is_aborted(worktree: Path) -> None:
    script = HOME_LITTER + "import time; time.sleep(60)\n"
    started = worktree / "home.txt"

    with pytest.raises(CheckAborted):
        go(
            ["home"],
            {"home": check(script)},
            worktree,
            {},
            abort_check=lambda: "cancel: cancelled locally" if started.exists() else None,
            abort_poll_s=0.05,
        )

    assert not Path(started.read_text(encoding="utf-8")).exists()


def test_every_check_gets_a_home_of_its_own(worktree: Path) -> None:
    script = (
        "import os, pathlib; "
        "pathlib.Path(os.environ['MARK'] + '.home').write_text(os.environ['HOME'])"
    )
    checks = {"a": check(script, env={"MARK": "a"}), "b": check(script, env={"MARK": "b"})}

    results = go(["a", "b"], checks, worktree, {})

    assert [r.passed for r in results] == [True, True]
    assert (worktree / "a.home").read_text() != (worktree / "b.home").read_text()


# ----------------------------------------------------------------------------------------------
# the shared git state: GitOps.state_snapshot / GitSnapshot.differences / commit_parents
# ----------------------------------------------------------------------------------------------
@dataclass
class World:
    git: GitRepoEnv
    ops: GitOps
    wt: Path
    base: str
    branch: str

    def in_wt(self, *args: str) -> str:
        return self.git.git(*args, cwd=self.wt)

    def side_commit(self, message: str = "elsewhere") -> str:
        """A commit nothing points at (the attacker's history)."""
        return self.in_wt("commit-tree", "HEAD^{tree}", "-m", message)


@pytest.fixture
def world(tmp_path: Path) -> World:
    git = make_git_env(tmp_path)
    ops = GitOps(git.clone, GitConfig(), state_dir=tmp_path / "state", environ=git.env)
    base = ops.resolve_base()
    wt = tmp_path / "state" / "worktrees" / "r1"
    ops.create_worktree(wt, "agent/issue-1-r1", base)
    git.git("branch", "agent/issue-9-other", base)  # the branch of another run, already pushed
    return World(git=git, ops=ops, wt=wt, base=base, branch="agent/issue-1-r1")


def rewrite_pointer(w: World) -> None:
    pointer = (w.wt / ".git").read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
    (w.wt / ".git").write_text(f"gitdir: {pointer}/../{Path(pointer).name}\n", encoding="utf-8")


def append_config(w: World) -> None:
    common = Path(w.in_wt("rev-parse", "--path-format=absolute", "--git-common-dir"))
    with (common / "config").open("a", encoding="utf-8") as handle:
        handle.write("[agent]\n\tprobe = 1\n")


def attach_to_another_branch(w: World) -> None:
    w.in_wt("branch", "other", w.base)
    w.in_wt("symbolic-ref", "HEAD", "refs/heads/other")


CHANGES: dict[str, tuple[Callable[[World], None], str]] = {
    "config-edited-with-git": (
        lambda w: w.in_wt("config", "remote.origin.pushurl", "/tmp/elsewhere.git"),
        "git config changed",
    ),
    "config-appended-as-a-file": (append_config, "git config changed"),
    "pointer-rewritten": (rewrite_pointer, ".git pointer changed"),
    "head-moved": (lambda w: w.in_wt("reset", "--soft", w.side_commit()), "HEAD moved from"),
    "head-detached": (lambda w: w.in_wt("checkout", "--detach"), "HEAD is now detached"),
    "head-attached-to-another-branch": (
        attach_to_another_branch,
        "HEAD is now attached to refs/heads/other",
    ),
    "agent-branch-created": (
        lambda w: w.in_wt("update-ref", "refs/heads/agent/planted", "HEAD"),
        "agent branches created: refs/heads/agent/planted",
    ),
    "nested-agent-branch-created": (
        lambda w: w.in_wt("update-ref", "refs/heads/agent/a/b/planted", "HEAD"),
        "agent branches created: refs/heads/agent/a/b/planted",
    ),
    "agent-branch-moved": (
        lambda w: w.in_wt("update-ref", "refs/heads/agent/issue-9-other", w.side_commit()),
        "agent branches moved: refs/heads/agent/issue-9-other",
    ),
    "agent-branch-deleted": (
        lambda w: w.in_wt("update-ref", "-d", "refs/heads/agent/issue-9-other"),
        "agent branches deleted: refs/heads/agent/issue-9-other",
    ),
    "agent-branch-turned-into-a-symref": (
        lambda w: w.in_wt("symbolic-ref", "refs/heads/agent/issue-9-other", "refs/heads/main"),
        "agent branches moved: refs/heads/agent/issue-9-other",
    ),
}


@pytest.mark.parametrize("case", sorted(CHANGES))
def test_the_snapshot_notices_what_a_check_could_use_to_publish_something_else(
    world: World, case: str
) -> None:
    mutate, expected = CHANGES[case]
    before = world.ops.state_snapshot(world.wt)
    assert before == world.ops.state_snapshot(world.wt)  # taking it changes nothing

    mutate(world)

    after = world.ops.state_snapshot(world.wt)
    differences = before.differences(after)
    assert any(expected in line for line in differences), differences
    assert before != after


def pack_refs(w: World) -> None:
    w.in_wt("pack-refs", "--all")


def another_worktree(w: World) -> None:
    w.git.git("worktree", "add", "--detach", str(w.wt.parent / "elsewhere"), w.base)


def stage_a_file(w: World) -> None:
    (w.wt / "docs" / "staged.md").write_text("x\n")
    w.in_wt("add", "docs/staged.md")


IGNORED: dict[str, Callable[[World], object]] = {
    "the-owner-commits-on-main": lambda w: w.git.push_main_commit("docs/owner.md", "owner\n"),
    "the-owner-fetches": lambda w: w.git.git("update-ref", "refs/remotes/origin/feature", w.base),
    "a-feature-branch": lambda w: w.git.git("branch", "feature/login", w.base),
    "a-tag": lambda w: w.git.git("tag", "v1.0", w.base),
    "refs-are-packed-by-gc": pack_refs,
    "another-worktree-appears": another_worktree,
    "files-change-in-the-worktree": lambda w: (w.wt / "docs" / "new.md").write_text("x\n"),
    "the-index-changes": stage_a_file,
    "a-branch-that-merely-starts-like-the-prefix": lambda w: w.git.git(
        "branch", "agentless/x", w.base
    ),
}


@pytest.mark.parametrize("case", sorted(IGNORED))
def test_the_snapshot_ignores_what_the_owner_does_in_the_same_clone(
    world: World, case: str
) -> None:
    before = world.ops.state_snapshot(world.wt)

    IGNORED[case](world)

    after = world.ops.state_snapshot(world.wt)
    assert before.differences(after) == []
    assert before == after


def test_a_clone_that_git_cannot_read_any_more_raises_instead_of_looking_unchanged(
    world: World,
) -> None:
    common = Path(world.in_wt("rev-parse", "--path-format=absolute", "--git-common-dir"))
    (common / "config").write_text("[core\n", encoding="utf-8")  # not a config file

    with pytest.raises(GitError):
        world.ops.state_snapshot(world.wt)


def test_commit_parents_are_read_from_the_commit_itself(world: World) -> None:
    (world.wt / "docs" / "x.md").write_text("x\n")
    world.ops.stage_all(world.wt)
    sha = world.ops.commit(world.wt, "x\n")
    other = world.side_commit("other")
    merge = world.in_wt("commit-tree", "HEAD^{tree}", "-p", sha, "-p", other, "-m", "merge")

    assert world.ops.commit_parents(sha) == [world.base]
    assert world.ops.commit_parents(merge) == [sha, other]
    assert world.ops.commit_parents(world.base) == []  # the repository's first commit
    with pytest.raises(GitError):
        world.ops.commit_parents(world.in_wt("rev-parse", "HEAD^{tree}"))  # not a commit


def test_commit_parents_ignore_replacement_refs(world: World) -> None:
    (world.wt / "docs" / "x.md").write_text("x\n")
    world.ops.stage_all(world.wt)
    sha = world.ops.commit(world.wt, "x\n")
    world.in_wt("replace", sha, world.side_commit("decoy"))  # git now shows the decoy for sha

    assert world.in_wt("rev-list", "--parents", "-n", "1", sha).split()[1:] != [world.base]
    assert world.ops.commit_parents(sha) == [world.base]


# ----------------------------------------------------------------------------------------------
# the worker: scrubbed environment, tampering during the checks and during the agent phase
# ----------------------------------------------------------------------------------------------
GIT = """
import os, subprocess, sys, tempfile

IDENT = {
    "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@example.invalid",
    "GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": "x@example.invalid",
}

def git(*args, env=None, input=None):
    done = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, env=env, input=input
    )
    return done.stdout.strip()

def other_commit(message):
    return git("commit-tree", "HEAD^{tree}", "-m", message, env={**os.environ, **IDENT})
"""

PROBE = """
import json, os, sys
home = os.path.expanduser("~")
report = {
    "env": dict(os.environ),
    "expanduser": home,
    "canary_via_tilde": os.path.exists(os.path.join(home, ".ssh", "id_demo")),
    "canary_via_home": os.path.exists(os.path.join(os.environ["HOME"], ".ssh", "id_demo")),
    "home_is_dir": os.path.isdir(os.environ["HOME"]),
    "cwd": os.getcwd(),
}
open(sys.argv[1], "w").write(json.dumps(report))
"""

SOFT_RESET_TO_EVIL_HISTORY = (
    GIT
    + """
base = git("rev-parse", "HEAD")
blob = git("hash-object", "-w", "--stdin", input="echo pwned\\n")
env = {**os.environ, **IDENT, "GIT_INDEX_FILE": os.path.join(tempfile.mkdtemp(), "index")}
git("read-tree", base, env=env)
git("update-index", "--add", "--cacheinfo", f"100644,{blob},scripts/deploy.sh", env=env)
evil = git("commit-tree", git("write-tree", env=env), "-p", base, "-m", "evil", env=env)
git("reset", "--soft", evil)
"""
)

CONFIG_TAMPERING = (
    GIT
    + """
mode = sys.argv[1]
common = git("rev-parse", "--git-common-dir")
if mode == "append":
    open(os.path.join(common, "config"), "a").write("[agent]\\n\\tprobe = 1\\n")
elif mode == "remote-url":
    git("config", "remote.origin.url", "/tmp/elsewhere.git")
elif mode == "insteadof":
    git("config", "url./tmp/elsewhere/.insteadOf", git("remote", "get-url", "origin"))
elif mode == "corrupt":
    open(os.path.join(common, "config"), "w").write("[core\\n")
"""
)

REF_TAMPERING = (
    GIT
    + """
mode = sys.argv[1]
if mode == "create":
    git("update-ref", "refs/heads/agent/planted", "HEAD")
elif mode == "create-nested":
    git("update-ref", "refs/heads/agent/a/b/planted", "HEAD")
elif mode == "move":
    git("update-ref", "refs/heads/agent/issue-9-other", other_commit("elsewhere"))
elif mode == "delete":
    git("update-ref", "-d", "refs/heads/agent/issue-9-other")
"""
)

POINTER_TAMPERING = """
import os
pointer = open(".git").read().strip().removeprefix("gitdir: ")
open(".git", "w").write("gitdir: " + os.path.join(pointer, "..", os.path.basename(pointer)) + "\\n")
"""

DETACH = GIT + "git('checkout', '--detach')\n"

BENIGN_NEIGHBOURS = (
    GIT
    + """
other = other_commit("owner work")
git("update-ref", "refs/heads/feature/owner", other)
git("update-ref", "refs/remotes/origin/feature", other)
git("tag", "v-from-the-owner", other)
git("pack-refs", "--all")
git("worktree", "add", "--detach", os.path.join(tempfile.mkdtemp(), "elsewhere"), "HEAD")
"""
)


def write_script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / "check-scripts" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def rig_with_check(
    make_rig_for: Callable[..., Rig], tmp_path: Path, body: str, *args: str, **check_fields: Any
) -> Rig:
    """A rig whose check ``probe`` runs ``body`` (a script outside the worktree) with ``args``."""
    script = write_script(tmp_path, "probe", body)
    probe = {"argv": [sys.executable, str(script), *args], **check_fields}
    return make_rig_for(checks={"probe": probe})


def assert_tampered(rig: Rig, run: RunRecord, *, detail: str | None = None) -> None:
    """The run ended BLOCKED ``worktree_tampered`` with nothing committed, pushed or opened."""
    assert run.status is RunStatus.BLOCKED, (run.status, run.failure_kind, run.failure_detail)
    assert run.failure_kind == "worktree_tampered", (run.failure_kind, run.failure_detail)
    if detail is not None:
        assert detail in (run.failure_detail or ""), run.failure_detail
    assert run.commit_sha is None and run.pr_number is None
    assert rig.git_env.origin_branches() == ["main"]
    assert rig.fake.pulls == {}


def test_a_clean_check_passes_and_the_run_is_published(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, "print('check ok')")
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.commit_sha is not None and run.pr_number is not None
    assert run.result is not None and run.result["checks"][0]["passed"] is True
    assert rig.git_env.origin_branches() == sorted(["main", run.branch or ""])


def test_checks_run_with_a_scrubbed_environment_and_an_empty_home(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    out = tmp_path / "probe.json"
    rig = rig_with_check(make_rig_for, tmp_path, PROBE, str(out), env={"CONFIGURED": "yes"})
    real_home = rig.git_env.home
    (real_home / ".ssh").mkdir()
    (real_home / ".ssh" / "id_demo").write_text(CANARY, encoding="utf-8")
    rig.worker._environ.update({**POISON, "LANG": "C.UTF-8"})
    rig.worker._environ["PATH"] = "/home/poison/bin:" + rig.worker._environ["PATH"]
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    report = json.loads(out.read_text(encoding="utf-8"))
    env = report["env"]
    assert report["canary_via_tilde"] is False and report["canary_via_home"] is False
    assert env["HOME"] != str(real_home) and report["expanduser"] == env["HOME"]
    assert report["home_is_dir"] is True and not Path(env["HOME"]).exists()  # gone after the check
    assert Path(report["cwd"]).resolve() == Path(run.worktree_path or "").resolve()
    assert str(real_home) not in json.dumps(env) and "poison" not in json.dumps(env)
    names = {name.upper() for name in env}
    assert not [n for n in names if any(w in n for w in ("TOKEN", "SECRET", "GITHUB", "ANTHROPIC"))]
    assert "SSH_AUTH_SOCK" not in env
    assert [d for d in env["PATH"].split(":") if d in PINNED_PATH] == PINNED_PATH
    assert env["USER"] == env["LOGNAME"] == "agent-bridge" and env["LANG"] == "C.UTF-8"
    assert env["CONFIGURED"] == "yes"


def test_a_check_that_moves_the_branch_onto_another_history_blocks_the_run(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, SOFT_RESET_TO_EVIL_HISTORY)
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail="moved the worktree HEAD")
    assert "scripts/deploy.sh" not in rig.git_env.origin_git("log", "--all", "--name-only")
    assert run.result is not None and run.result["checks"][0]["passed"] is True


@pytest.mark.parametrize("mode", ["append", "remote-url", "insteadof", "corrupt"])
def test_a_check_that_edits_the_shared_git_config_blocks_the_run(
    make_rig_for: Callable[..., Rig], tmp_path: Path, mode: str
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, CONFIG_TAMPERING, mode)
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert_tampered(rig, run)
    assert "the checks" in (run.failure_detail or "")


@pytest.mark.parametrize("mode", ["create", "create-nested", "move", "delete"])
def test_a_check_that_touches_an_agent_branch_blocks_the_run(
    make_rig_for: Callable[..., Rig], tmp_path: Path, mode: str
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, REF_TAMPERING, mode)
    rig.git_env.git("branch", "agent/issue-9-other", "main")  # another run's branch
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail="agent branches")
    assert "refs/heads/agent/" in (run.failure_detail or "")


def test_a_check_that_rewrites_the_worktree_pointer_blocks_the_run(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, POINTER_TAMPERING)
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail=".git pointer")


def test_a_check_that_detaches_head_blocks_the_run(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, DETACH)
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail="HEAD is now detached")


def test_what_the_owner_does_next_to_a_run_is_not_tampering(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = rig_with_check(make_rig_for, tmp_path, BENIGN_NEIGHBOURS)
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert rig.git_env.origin_branches() == sorted(["main", run.branch or ""])


def agent_edit(*ops: dict[str, Any]) -> list[dict[str, Any]]:
    """A fake agent that does ``ops`` (outside its worktree if it likes), edits docs, finishes."""
    report = {"status": "completed", "summary": "Edited the documentation.", "files_changed": []}
    return [
        *ops,
        {"op": "write_file", "path": DOCS_NOTE, "content": "Hello\n"},
        {"op": "final", "text": "Done.", "report": report},
    ]


def agent_appends_to_the_shared_config(rig: Rig) -> list[dict[str, Any]]:
    config = rig.git_env.clone / ".git" / "config"
    text = f'[url "/tmp/elsewhere/"]\n\tinsteadOf = {rig.git_env.origin}\n'
    return agent_edit({"op": "append_file", "path": str(config), "content": text, "abs": True})


def agent_plants_an_agent_branch(rig: Rig) -> list[dict[str, Any]]:
    planted = rig.git_env.clone / ".git" / "refs" / "heads" / "agent" / "planted"
    sha = rig.git_env.git("rev-parse", "main")
    op = {"op": "write_file", "path": str(planted), "content": sha + "\n", "abs": True}
    return agent_edit(op)


AGENT_PHASE_TAMPERING = {
    "config": (agent_appends_to_the_shared_config, "git config changed"),
    "agent-branch": (agent_plants_an_agent_branch, "agent branches created"),
}


@pytest.mark.parametrize("case", sorted(AGENT_PHASE_TAMPERING))
@pytest.mark.parametrize("with_checks", [False, True], ids=["no-checks", "checks"])
def test_shared_git_state_changed_during_the_agent_phase_blocks_the_run(
    make_rig_for: Callable[..., Rig], case: str, with_checks: bool
) -> None:
    # An omp run in code scope executes agent-written code through its allow-listed bash before a
    # single check runs; the fake agent stands in for that code by writing outside the worktree.
    rig: Rig = make_rig_for(default_checks=["pass"] if with_checks else [])
    build, expected = AGENT_PHASE_TAMPERING[case]
    rig.set_scenario(build(rig))
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail="the agent changed shared git state")
    assert expected in (run.failure_detail or "")
    assert run.result is not None and run.result["checks"] == []  # stopped before any check


def test_the_agent_phase_is_verified_before_agent_written_checks_run(
    make_rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    marker = tmp_path / "check-ran.marker"
    rig = rig_with_check(make_rig_for, tmp_path, f"open({str(marker)!r}, 'w').write('x')")
    rig.set_scenario(agent_appends_to_the_shared_config(rig))
    number = rig.new_task_issue(checks=["probe"])

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail="the agent changed shared git state")
    assert not marker.exists(), "agent-written checks ran in a repository that was already tampered"


def test_a_commit_that_does_not_sit_on_the_base_is_never_published(
    make_rig_for: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig: Rig = make_rig_for(default_checks=["pass"])
    real_commit = rig.git.commit

    def commit_on_a_branch_that_was_moved(path: Path, message: str) -> str:
        # Something moves the branch after the last verification; git commit builds on its tip.
        env = rig.git_env.env
        elsewhere = run_git(path, "commit-tree", "HEAD^{tree}", "-m", "elsewhere", env=env)
        run_git(path, "update-ref", "HEAD", elsewhere, env=env)
        return real_commit(path, message)

    monkeypatch.setattr(rig.git, "commit", commit_on_a_branch_that_was_moved)
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert_tampered(rig, run, detail="instead of the base")
