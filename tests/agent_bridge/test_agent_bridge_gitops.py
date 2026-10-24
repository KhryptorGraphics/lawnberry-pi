"""``GitOps`` against real git: a bare origin, a clone, linked worktrees, staging analysis,
commit and push, plus the safety rules (no hooks, no ambient config, no forced push, token
handling, scrubbed environment) that keep an agent-written worktree from running code.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.rig import GitRepoEnv, make_git_env, run_git

from tools.agent_bridge.config import GitConfig
from tools.agent_bridge.errors import GitError
from tools.agent_bridge.gitops import GitOps, worktree_pointer
from tools.agent_bridge.policy import Change
from tools.agent_bridge.redact import Redactor

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or sys.platform == "win32",
    reason="needs git and POSIX file modes",
)

AUTHOR_NAME = "Bridge Bot"
AUTHOR_EMAIL = "bridge-bot@example.invalid"
# Matches no secret pattern, so only literal redaction (the token passed to GitOps) can hide it.
TOKEN = "tok-AbCdEf0123456789xyz"
HOOKS = (
    "pre-commit",
    "prepare-commit-msg",
    "commit-msg",
    "post-commit",
    "post-checkout",
    "pre-push",
    "reference-transaction",
)
HARDENING = (
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "core.autocrlf=false",
    "protocol.ext.allow=never",
)


# ------------------------------------------------------------------------------ plumbing
@dataclass
class Spawn:
    """One subprocess the GitOps under test started."""

    argv: list[str]
    kwargs: dict[str, Any]

    @property
    def env(self) -> dict[str, str]:
        return self.kwargs["env"]

    @property
    def config(self) -> list[str]:
        """The ``-c key=value`` settings placed before the git subcommand."""
        args, found, index = self.argv[1:], [], 0
        while index < len(args) and args[index] == "-c":
            found.append(args[index + 1])
            index += 2
        return found

    @property
    def args(self) -> list[str]:
        """Everything after the git executable and its ``-c`` settings."""
        return self.argv[1 + 2 * len(self.config) :]

    @property
    def subcommand(self) -> str:
        return self.args[0]


@dataclass
class SpawnLog:
    """Wraps ``subprocess.run`` and remembers every call (the ``run=`` injection point)."""

    calls: list[Spawn] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(Spawn(list(argv), dict(kwargs)))
        return subprocess.run(argv, **kwargs)

    def of(self, subcommand: str) -> list[Spawn]:
        return [c for c in self.calls if c.subcommand == subcommand]


@dataclass
class Harness:
    git: GitRepoEnv
    ops: GitOps
    spawns: SpawnLog
    cfg: GitConfig
    environ: dict[str, str]
    state_dir: Path
    tmp: Path

    @property
    def clone(self) -> Path:
        return self.git.clone

    def worktree(self, name: str = "r1", branch: str | None = None) -> tuple[Path, str]:
        """A fresh worktree on its own branch at the current base; returns (path, base sha)."""
        base = self.ops.resolve_base()
        path = self.state_dir / "worktrees" / name
        self.ops.create_worktree(path, branch or f"agent/issue-1-{name}", base)
        return path, base

    def commit_file(self, name: str = "r1", content: str = "hello\n") -> tuple[Path, str, str]:
        """A worktree with one committed file; returns (path, base sha, commit sha)."""
        path, base = self.worktree(name)
        put(path / "docs" / f"{name}.md", content)
        self.ops.stage_all(path)
        return path, base, self.ops.commit(path, f"agent: {name}\n")


def put(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


def base_environ(git: GitRepoEnv) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(git.home),
        "GIT_CONFIG_NOSYSTEM": "1",
    }


HarnessFactory = Callable[..., Harness]


@pytest.fixture
def make_harness(tmp_path: Path) -> Iterator[HarnessFactory]:
    """Build a repository environment plus a GitOps bound to it (``root`` isolates extras)."""

    def factory(
        *,
        root: Path | None = None,
        token: str | None = None,
        environ_extra: Mapping[str, str] | None = None,
        redactor: Redactor | None = None,
        **cfg: Any,
    ) -> Harness:
        root = root or tmp_path
        git = make_git_env(root)
        settings: dict[str, Any] = {
            "author_name": AUTHOR_NAME,
            "author_email": AUTHOR_EMAIL,
            "git_timeout_seconds": 60,
            **cfg,
        }
        config = GitConfig(**settings)
        environ = {**base_environ(git), **(environ_extra or {})}
        spawns = SpawnLog()
        ops = GitOps(
            git.clone,
            config,
            state_dir=root / "state",
            token=token,
            redactor=redactor,
            environ=environ,
            run=spawns,
        )
        return Harness(git, ops, spawns, config, environ, root / "state", root)

    yield factory


@pytest.fixture
def h(make_harness: HarnessFactory) -> Harness:
    return make_harness()


def other_clone(h: Harness) -> Path:
    """A second clone of origin, used to move origin without touching the first clone."""
    other = h.tmp / "other"
    if not other.exists():
        run_git(h.tmp, "clone", "-q", str(h.git.origin), str(other), env=h.git.env)
    return other


def advance_origin(h: Harness, relpath: str, content: str, *, branch: str = "main") -> str:
    """Commit ``relpath`` on ``branch`` from the second clone and push it; returns the sha."""
    other = other_clone(h)

    def git(*args: str) -> str:
        return run_git(other, *args, env=h.git.env)

    if branch in h.git.origin_branches():
        git("checkout", "-q", branch)
        git("pull", "-q", "--ff-only", "origin", branch)
    else:
        git("checkout", "-q", "-b", branch)
    put(other / relpath, content)
    git("add", "-A")
    git("commit", "-q", "-m", f"update {relpath}")
    git("push", "-q", "origin", f"{branch}:{branch}")
    return git("rev-parse", "HEAD")


def tree_of(h: Harness, rev: str) -> str:
    return h.git.git("rev-parse", f"{rev}^{{tree}}")


# ------------------------------------------------------------------- repository facts
def test_toplevel_is_the_repository_root_even_from_a_subdirectory(h: Harness) -> None:
    inside = GitOps(h.clone / "docs", h.cfg, state_dir=h.state_dir, environ=h.environ)

    assert h.ops.toplevel() == h.clone.resolve()
    assert inside.toplevel() == h.clone.resolve()


def test_remote_url_reports_the_configured_remote(make_harness: HarnessFactory) -> None:
    h = make_harness()
    assert h.ops.remote_url() == str(h.git.origin)

    missing = GitOps(h.clone, GitConfig(remote="nope"), state_dir=h.state_dir, environ=h.environ)
    with pytest.raises(GitError, match="No such remote"):
        missing.remote_url()


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("agent/issue-1-abc12345", True),
        ("feature/ok_1.2", True),
        ("bad..name", False),
        ("-leading-dash", False),
        ("with space", False),
        ("ends.lock", False),
        ("ends/", False),
        ("", False),
        ("a~b", False),
        ("a^b", False),
        ("a:b", False),
        ("a?b", False),
        ("a*b", False),
        ("a[b", False),
        ("a//b", False),
        ("/leading-slash", False),
        (".hidden", False),
        ("trailing.", False),
        ("back\\slash", False),
        ("HEAD", False),
    ],
)
def test_check_branch_name_follows_git_ref_rules(h: Harness, name: str, valid: bool) -> None:
    assert h.ops.check_branch_name(name) is valid


# ------------------------------------------------------------------- fetch / resolve_base
def test_fetch_moves_the_base_to_what_origin_has_and_nothing_else(h: Harness) -> None:
    old = h.ops.resolve_base()
    assert h.git.git("rev-parse", "main") == old
    new = advance_origin(h, "docs/upstream.md", "from upstream\n")
    advance_origin(h, "docs/feature.md", "feature\n", branch="feature-x")
    run_git(other_clone(h), "tag", "v1", env=h.git.env)
    run_git(other_clone(h), "push", "-q", "origin", "v1", env=h.git.env)
    assert h.ops.resolve_base() == old  # nothing has been fetched yet

    h.ops.fetch()

    assert h.ops.resolve_base() == new
    assert h.git.git("rev-parse", "main") == old  # the operator's own branch is untouched
    assert h.git.git("tag", "--list") == ""  # no tags
    remote_refs = h.git.git("for-each-ref", "--format=%(refname)", "refs/remotes")
    assert remote_refs.split() == ["refs/remotes/origin/main"]  # only the base branch


def test_fetch_and_resolve_base_fail_for_a_base_branch_that_does_not_exist(
    h: Harness,
) -> None:
    ghost = GitOps(h.clone, GitConfig(base_branch="nope"), state_dir=h.state_dir, environ=h.environ)
    with pytest.raises(GitError, match="couldn't find remote ref"):
        ghost.fetch()
    with pytest.raises(GitError):
        ghost.resolve_base()


def test_resolve_base_returns_the_commit_even_for_an_annotated_tip(h: Harness) -> None:
    sha = h.ops.resolve_base()
    assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")
    assert h.git.git("rev-parse", "refs/remotes/origin/main") == sha


# ------------------------------------------------------------------------- worktrees
def test_create_worktree_checks_out_a_new_branch_at_the_base_commit(h: Harness) -> None:
    base = h.ops.resolve_base()
    path = h.state_dir / "worktrees" / "deeper" / "r1"  # parents do not exist yet

    pointer = h.ops.create_worktree(path, "agent/issue-1-abc", base)

    assert pointer == worktree_pointer(path)
    assert pointer.startswith("gitdir: ")
    assert Path(pointer[len("gitdir: ") :]).resolve() == (h.clone / ".git/worktrees/r1").resolve()
    assert h.ops.worktree_head(path) == base
    assert h.ops.worktree_branch(path) == "agent/issue-1-abc"
    assert h.git.git("rev-parse", "refs/heads/agent/issue-1-abc") == base
    assert h.git.git("symbolic-ref", "HEAD", cwd=path) == "refs/heads/agent/issue-1-abc"
    assert (path / "docs/guide.md").read_text() == "# Guide\n\nOriginal text.\n"
    assert h.git.git("status", "--porcelain", cwd=path) == ""
    listed = {p.resolve() for p in h.ops.list_worktree_paths()}
    assert listed == {h.clone.resolve(), path.resolve()}


def test_create_worktree_uses_the_given_commit_not_the_clones_head(h: Harness) -> None:
    base = h.ops.resolve_base()
    put(h.clone / "docs/unpushed.md", "local only\n")
    h.git.git("add", "-A")
    h.git.git("commit", "-q", "-m", "local work")

    path = h.state_dir / "worktrees" / "r1"
    h.ops.create_worktree(path, "agent/issue-1-abc", base)

    assert h.ops.worktree_head(path) == base
    assert not (path / "docs/unpushed.md").exists()


@pytest.mark.parametrize("branch", ["bad..name", "with space", ""])
def test_create_worktree_rejects_invalid_branch_names_before_touching_disk(
    h: Harness, branch: str
) -> None:
    path = h.state_dir / "worktrees" / "r1"
    with pytest.raises(GitError, match="invalid branch name"):
        h.ops.create_worktree(path, branch, h.ops.resolve_base())
    assert not path.exists()
    assert (
        h.ops.list_worktree_paths() == [h.clone.resolve()] or len(h.ops.list_worktree_paths()) == 1
    )


def test_create_worktree_fails_for_unknown_commits_existing_branches_and_busy_paths(
    h: Harness,
) -> None:
    base = h.ops.resolve_base()
    root = h.state_dir / "worktrees"
    with pytest.raises(GitError, match="invalid reference"):
        h.ops.create_worktree(root / "nobase", "agent/nobase", "0" * 40)
    assert not (root / "nobase").exists()

    h.ops.create_worktree(root / "first", "agent/taken", base)
    with pytest.raises(GitError, match="already exists"):
        h.ops.create_worktree(root / "second", "agent/taken", base)
    assert not (root / "second").exists()

    busy = put(root / "busy" / "keep.txt", "precious\n").parent
    with pytest.raises(GitError, match="already exists"):
        h.ops.create_worktree(busy, "agent/busy", base)
    assert (busy / "keep.txt").read_text() == "precious\n"


def test_remove_worktree_deletes_directory_and_branch_and_is_idempotent(h: Harness) -> None:
    path, _base = h.worktree("r1", "agent/issue-1-abc")
    put(path / "docs/dirty.md", "uncommitted\n")  # force-removal must cope with a dirty tree

    h.ops.remove_worktree(path, "agent/issue-1-abc")

    assert not path.exists()
    assert (
        h.ops.list_worktree_paths() == [h.clone.resolve()] or len(h.ops.list_worktree_paths()) == 1
    )
    assert h.git.git("branch", "--list", "agent/*") == ""
    h.ops.remove_worktree(path, "agent/issue-1-abc")  # nothing left: still fine
    h.ops.remove_worktree(h.state_dir / "worktrees" / "never-existed", "agent/never")


def test_remove_worktree_tolerates_a_directory_deleted_behind_its_back(h: Harness) -> None:
    path, _base = h.worktree("r1", "agent/issue-1-abc")
    shutil.rmtree(path)

    h.ops.remove_worktree(path, "agent/issue-1-abc")

    assert len(h.ops.list_worktree_paths()) == 1  # the stale registration was pruned
    assert h.git.git("branch", "--list", "agent/*") == ""


def test_remove_worktree_without_a_branch_keeps_the_branch(h: Harness) -> None:
    path, _base = h.worktree("r1", "agent/issue-1-abc")

    h.ops.remove_worktree(path)

    assert not path.exists()
    assert h.git.git("branch", "--list", "agent/*").strip() == "agent/issue-1-abc"


def test_worktree_pointer_is_none_without_a_git_file(tmp_path: Path) -> None:
    assert worktree_pointer(tmp_path) is None
    assert worktree_pointer(tmp_path / "missing") is None


# ----------------------------------------------------------- stage_all / staged_changes
Setup = Callable[[Path, GitRepoEnv], None]


def add(rel: str, content: str | bytes) -> Setup:
    def run(wt: Path, _git: GitRepoEnv) -> None:
        put(wt / rel, content)

    return run


def remove(rel: str) -> Setup:
    return lambda wt, _git: (wt / rel).unlink()


def symlink(rel: str, target: str) -> Setup:
    def run(wt: Path, _git: GitRepoEnv) -> None:
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, wt / rel)

    return run


def several(*setups: Setup) -> Setup:
    def run(wt: Path, git: GitRepoEnv) -> None:
        for setup in setups:
            setup(wt, git)

    return run


def make_executable(rel: str) -> Setup:
    return lambda wt, _git: (wt / rel).chmod(0o755)


def replace_with_symlink(rel: str, target: str) -> Setup:
    def run(wt: Path, _git: GitRepoEnv) -> None:
        (wt / rel).unlink()
        os.symlink(target, wt / rel)

    return run


def move(src: str, dst: str) -> Setup:
    return lambda wt, _git: (wt / src).rename(wt / dst)


def nested_repository(rel: str) -> Setup:
    def run(wt: Path, git: GitRepoEnv) -> None:
        sub = wt / rel
        sub.mkdir(parents=True)
        run_git(sub, "init", "-q", env=git.env)
        put(sub / "inner.txt", "inner\n")
        run_git(sub, "add", "-A", env=git.env)
        run_git(sub, "commit", "-q", "-m", "inner", env=git.env)

    return run


def non_utf8_name(wt: Path, _git: GitRepoEnv) -> None:
    raw = os.fsencode(str(wt / "docs")) + b"/\xff.md"
    try:
        with open(raw, "wb") as handle:
            handle.write(b"z\n")
    except OSError:
        pytest.skip("this file system rejects non-UTF-8 file names")


NON_UTF8 = b"docs/\xff.md".decode("utf-8", errors="surrogateescape")


def new_file(
    path: str, size: int | None, added: int, mode: str = "100644", binary: bool = False
) -> Change:
    return Change(path, "A", "000000", mode, size, binary, added, 0)


STAGED_CASES = [
    pytest.param(add("docs/new.md", "a\nb\nc\n"), [new_file("docs/new.md", 6, 3)], id="added-file"),
    pytest.param(add("docs/empty.md", ""), [new_file("docs/empty.md", 0, 0)], id="empty-file"),
    pytest.param(
        add("docs/guide.md", "# Guide\n\nChanged text.\nmore\n"),
        [Change("docs/guide.md", "M", "100644", "100644", 28, False, 2, 1)],
        id="modified-file",
    ),
    pytest.param(
        remove("docs/index.md"),
        [Change("docs/index.md", "D", "100644", "000000", None, False, 0, 1)],
        id="deleted-file",
    ),
    pytest.param(
        make_executable("README.md"),
        [Change("README.md", "M", "100644", "100755", 12, False, 0, 0)],
        id="mode-change-to-executable",
    ),
    pytest.param(
        several(add("docs/run.sh", "#!/bin/sh\n"), make_executable("docs/run.sh")),
        [new_file("docs/run.sh", 10, 1, mode="100755")],
        id="new-executable-file",
    ),
    pytest.param(
        symlink("docs/link.md", "guide.md"),
        [new_file("docs/link.md", 8, 1, mode="120000")],
        id="symlink",
    ),
    pytest.param(
        replace_with_symlink("docs/guide.md", "index.md"),
        [Change("docs/guide.md", "T", "100644", "120000", 8, False, 1, 3)],
        id="file-turned-into-symlink",
    ),
    pytest.param(
        nested_repository("vendor/sub"),
        [new_file("vendor/sub", None, 1, mode="160000")],
        id="nested-repository-gitlink",
    ),
    pytest.param(
        add("docs/blob.dat", b"\x00\x01\x02binary"),
        [new_file("docs/blob.dat", 9, 0, binary=True)],
        id="binary-file",
    ),
    pytest.param(
        add("big.bin", b"a" * 3_000_000), [new_file("big.bin", 3_000_000, 1)], id="big-file-size"
    ),
    pytest.param(
        add("docs/héllo wörld.md", "x\n"),
        [new_file("docs/héllo wörld.md", 2, 1)],
        id="unicode-and-space",
    ),
    pytest.param(
        several(
            add('docs/"quoted".md', "x\n"),
            add("docs/:colon.md", "x\n"),
            add("docs/-dash.md", "x\n"),
            add("docs/a b.md", "x\n"),
        ),
        [
            new_file('docs/"quoted".md', 2, 1),
            new_file("docs/-dash.md", 2, 1),
            new_file("docs/:colon.md", 2, 1),
            new_file("docs/a b.md", 2, 1),
        ],
        id="quotes-colon-dash-and-space",
    ),
    pytest.param(
        add("docs/new\nline.md", "x\n"),
        [new_file("docs/new\nline.md", 2, 1)],
        id="newline-in-name",
    ),
    pytest.param(
        add("docs/tab\there.md", "x\n"),
        [new_file("docs/tab\there.md", 2, 1)],
        id="tab-in-name",
    ),
    pytest.param(non_utf8_name, [new_file(NON_UTF8, 2, 1)], id="non-utf8-name"),
    pytest.param(
        move("docs/guide.md", "docs/moved.md"),
        [
            Change("docs/guide.md", "D", "100644", "000000", None, False, 0, 3),
            new_file("docs/moved.md", 24, 3),
        ],
        id="rename-is-reported-as-delete-plus-add",
    ),
    pytest.param(
        several(add("docs/same1.md", "same\n"), add("docs/same2.md", "same\n")),
        [new_file("docs/same1.md", 5, 1), new_file("docs/same2.md", 5, 1)],
        id="identical-blobs-both-get-a-size",
    ),
]


@pytest.mark.parametrize(("setup", "expected"), STAGED_CASES)
def test_staged_changes_describes_each_kind_of_change(
    h: Harness, setup: Setup, expected: list[Change]
) -> None:
    wt, base = h.worktree()
    setup(wt, h.git)

    h.ops.stage_all(wt)
    staged = h.ops.staged_changes(wt, base, max_diff_bytes=10_000_000)

    def order(changes: list[Change]) -> list[Change]:
        return sorted(changes, key=lambda c: c.path.encode("utf-8", errors="surrogateescape"))

    assert order(staged.changes) == order(expected)
    assert staged.diff_bytes > 0 and not staged.truncated


def test_staged_changes_of_an_untouched_worktree_is_empty(h: Harness) -> None:
    wt, base = h.worktree()
    h.ops.stage_all(wt)

    staged = h.ops.staged_changes(wt, base, max_diff_bytes=1000)

    assert staged.changes == []
    assert (staged.diff_text, staged.diff_bytes, staged.truncated) == ("", 0, False)


def test_diff_text_is_bounded_while_diff_bytes_reports_the_full_size(h: Harness) -> None:
    wt, base = h.worktree()
    put(wt / "docs/long.md", "".join(f"line {i:04d} {'x' * 40}\n" for i in range(400)))
    h.ops.stage_all(wt)

    whole = h.ops.staged_changes(wt, base, max_diff_bytes=10_000_000)
    total = whole.diff_bytes
    assert not whole.truncated and len(whole.diff_text.encode()) == total > 20_000
    assert "+++ b/docs/long.md" in whole.diff_text and "@@ -0,0 +1,400 @@" in whole.diff_text

    small = h.ops.staged_changes(wt, base, max_diff_bytes=1_000)
    assert small.truncated and small.diff_bytes == total  # the policy still sees the real size
    assert 0 < len(small.diff_text) <= 2 * 1_000
    assert whole.diff_text.startswith(small.diff_text)

    # the retained window is twice max_diff_bytes: just enough room is not truncation
    roomy = h.ops.staged_changes(wt, base, max_diff_bytes=(total + 1) // 2)
    tight = h.ops.staged_changes(wt, base, max_diff_bytes=(total + 1) // 2 - 1)
    assert not roomy.truncated and tight.truncated


def test_diff_text_uses_no_context_lines(h: Harness) -> None:
    wt, base = h.worktree()
    put(wt / "docs/guide.md", "# Guide\n\nChanged text.\n")
    h.ops.stage_all(wt)

    diff = h.ops.staged_changes(wt, base, max_diff_bytes=100_000).diff_text

    assert "@@ -3 +3 @@" in diff
    assert "-Original text." in diff and "+Changed text." in diff
    assert "# Guide" not in diff  # -U0: unchanged neighbours are not shown


def test_write_tree_is_stable_and_tracks_the_index(h: Harness) -> None:
    wt, _base = h.worktree()
    clean = h.ops.write_tree(wt)
    assert clean == tree_of(h, "HEAD") == h.ops.write_tree(wt)

    put(wt / "docs/guide.md", "# Guide\n\nEdited.\n")
    assert h.ops.write_tree(wt) == clean  # nothing staged yet
    h.ops.stage_all(wt)
    edited = h.ops.write_tree(wt)
    assert edited != clean
    h.ops.stage_all(wt)
    assert h.ops.write_tree(wt) == edited  # staging twice changes nothing

    put(wt / "docs/guide.md", "# Guide\n\nOriginal text.\n")
    h.ops.stage_all(wt)
    assert h.ops.write_tree(wt) == clean  # content addressed

    put(wt / "docs/guide.md", "# Guide\n\nEdited.\n")
    h.ops.stage_all(wt)
    commit = h.ops.commit(wt, "edit\n")
    assert tree_of(h, commit) == edited  # the commit holds exactly the tree that was inspected


# --------------------------------------------------------------------- hooks never run
def install_hooks(hooks_dir: Path, markers: Path) -> None:
    hooks_dir.mkdir(parents=True, exist_ok=True)
    markers.mkdir(parents=True, exist_ok=True)
    for name in HOOKS:
        script = hooks_dir / name
        script.write_text(f"#!/bin/sh\necho ran >> '{markers}/{name}'\nexit 0\n")
        script.chmod(0o755)


def arm_hooks(h: Harness, mode: str) -> Path:
    markers = h.tmp / "markers"
    if mode == "core.hooksPath":
        install_hooks(h.tmp / "custom-hooks", markers)
        h.git.git("config", "core.hooksPath", str(h.tmp / "custom-hooks"))
    else:
        install_hooks(h.clone / ".git" / "hooks", markers)
    return markers


def fired(markers: Path) -> set[str]:
    return {p.name for p in markers.iterdir()}


def run_full_lifecycle(h: Harness, branch: str = "agent/issue-1-life") -> str:
    h.ops.fetch()
    base = h.ops.resolve_base()
    path = h.state_dir / "worktrees" / "life"
    h.ops.create_worktree(path, branch, base)
    put(path / "docs/life.md", "alive\n")
    h.ops.stage_all(path)
    h.ops.staged_changes(path, base, max_diff_bytes=10_000)
    h.ops.write_tree(path)
    sha = h.ops.commit(path, "life\n")
    h.ops.push(branch)
    assert h.ops.remote_branch_sha(branch) == sha
    h.ops.remove_worktree(path, branch)
    return sha


@pytest.mark.parametrize("mode", ["core.hooksPath", "git-dir-hooks"])
def test_repository_hooks_never_run_during_the_whole_lifecycle(
    make_harness: HarnessFactory, tmp_path: Path, mode: str
) -> None:
    control = make_harness(root=tmp_path / "control")
    subject = make_harness(root=tmp_path / "subject")
    control_markers, subject_markers = arm_hooks(control, mode), arm_hooks(subject, mode)

    # control: the very same repository setup does fire hooks for plain git
    base = control.git.git("rev-parse", "origin/main")
    wt = control.tmp / "plain-worktree"
    control.git.git("worktree", "add", "-q", "-b", "agent/plain", str(wt), base)
    put(wt / "docs/plain.md", "x\n")
    control.git.git("add", "-A", cwd=wt)
    control.git.git("commit", "-q", "-m", "plain", cwd=wt)
    control.git.git("push", "-q", "origin", "agent/plain")
    assert fired(control_markers) == set(HOOKS)

    run_full_lifecycle(subject)

    assert fired(subject_markers) == set()


def hostile_config_commit_control(h: Harness) -> subprocess.CompletedProcess[str]:
    """Plain ``git commit`` in a fresh worktree of ``h`` (the control for a hostile config)."""
    base = h.git.git("rev-parse", "origin/main")
    wt = h.tmp / "plain-worktree"
    h.git.git("worktree", "add", "-q", "-b", "agent/plain", str(wt), base)
    put(wt / "docs/plain.md", "x\n")
    run_git(wt, "add", "-A", env=h.git.env)
    return subprocess.run(
        ["git", "commit", "-q", "-m", "plain"],
        cwd=wt,
        env=h.git.env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_configured_fsmonitor_hook_is_never_executed(
    make_harness: HarnessFactory, tmp_path: Path
) -> None:
    markers = tmp_path / "markers"
    markers.mkdir()
    script = tmp_path / "fsmonitor.sh"
    script.write_text(f"#!/bin/sh\necho ran >> '{markers}/fsmonitor'\nprintf '\\0'\n")
    script.chmod(0o755)
    control = make_harness(root=tmp_path / "control")
    subject = make_harness(root=tmp_path / "subject")
    for h in (control, subject):
        h.git.git("config", "core.fsmonitor", str(script))

    control_wt, _ = control.worktree()
    put(control_wt / "docs/x.md", "x\n")
    run_git(control_wt, "add", "-A", env=control.git.env)
    assert fired(markers) == {"fsmonitor"}  # plain git runs it
    (markers / "fsmonitor").unlink()

    wt, base = subject.worktree()
    put(wt / "docs/x.md", "x\n")
    subject.ops.stage_all(wt)
    subject.ops.staged_changes(wt, base, max_diff_bytes=1000)
    subject.ops.commit(wt, "x\n")

    assert fired(markers) == set()


def test_the_ext_transport_stays_disabled_even_if_the_repository_enables_it(
    make_harness: HarnessFactory, tmp_path: Path
) -> None:
    markers = tmp_path / "markers"
    markers.mkdir()
    script = tmp_path / "ext.sh"
    script.write_text(f"#!/bin/sh\necho ran >> '{markers}/ext'\nexit 1\n")
    script.chmod(0o755)
    control = make_harness(root=tmp_path / "control")
    subject = make_harness(root=tmp_path / "subject")
    for h in (control, subject):
        h.git.git("config", "protocol.ext.allow", "always")
        h.git.git("remote", "set-url", "origin", f"ext::{script} %S")

    run_git_failure = subprocess.run(
        ["git", "fetch", "origin"], cwd=control.clone, env=control.git.env, capture_output=True
    )
    assert run_git_failure.returncode != 0 and fired(markers) == {"ext"}  # plain git runs it
    (markers / "ext").unlink()

    with pytest.raises(GitError, match="transport 'ext' not allowed"):
        subject.ops.fetch()

    assert fired(markers) == set()


def test_ambient_autocrlf_cannot_rewrite_what_the_agent_wrote(h: Harness) -> None:
    h.git.git("config", "core.autocrlf", "true")
    wt, base = h.worktree()
    put(wt / "docs/crlf.md", b"a\r\nb\r\n")
    plain = h.git.git("hash-object", "--path=docs/crlf.md", "docs/crlf.md", cwd=wt)
    raw = h.git.git("hash-object", "--no-filters", "docs/crlf.md", cwd=wt)
    assert plain != raw  # control: with this config plain git would normalise the line endings

    h.ops.stage_all(wt)
    staged = h.ops.staged_changes(wt, base, max_diff_bytes=10_000)

    assert h.git.git("rev-parse", ":docs/crlf.md", cwd=wt) == raw
    assert [c.size for c in staged.changes] == [6]


# --------------------------------------------------------------------------- commit
def test_commit_uses_the_configured_identity_whatever_the_environment_says(
    make_harness: HarnessFactory,
) -> None:
    h = make_harness(
        environ_extra={
            "GIT_AUTHOR_NAME": "Ambient Author",
            "GIT_AUTHOR_EMAIL": "ambient@example.invalid",
            "GIT_COMMITTER_NAME": "Ambient Committer",
            "GIT_COMMITTER_EMAIL": "ambient-c@example.invalid",
        }
    )
    h.git.git("config", "user.name", "Repo Person")
    h.git.git("config", "user.email", "repo@example.invalid")
    wt, base = h.worktree()
    put(wt / "docs/id.md", "x\n")
    h.ops.stage_all(wt)

    sha = h.ops.commit(wt, "who am i\n")

    identity = h.git.git("log", "-1", "--format=%an|%ae|%cn|%ce", sha)
    assert identity == f"{AUTHOR_NAME}|{AUTHOR_EMAIL}|{AUTHOR_NAME}|{AUTHOR_EMAIL}"
    assert h.git.git("rev-parse", f"{sha}^") == base
    assert h.ops.worktree_head(wt) == sha


MESSAGE = (
    "agent(#7): Document the bridge \u00fcn\u00ef \u2713\n\n"
    "Body with `ticks`, \"quotes\", 'single', $(touch pwned), $HOME and # not-a-comment.\n\n"
    "Refs #7\n"
    "Agent-Bridge-Run: 0123456789abcdef\n"
    "Agent-Backend: omp\n"
    "Approved-By: owner\n"
)


def test_commit_message_travels_verbatim_through_stdin(h: Harness) -> None:
    wt, _base = h.worktree()
    put(wt / "docs/msg.md", "x\n")
    h.ops.stage_all(wt)

    sha = h.ops.commit(wt, MESSAGE)

    spawn = h.spawns.of("commit")[0]
    assert spawn.kwargs["input"] == MESSAGE.encode("utf-8")
    assert spawn.args[-2:] == ["-F", "-"]
    assert not any("Document the bridge" in arg for arg in spawn.argv)  # never on the command line
    raw = h.git.git("cat-file", "commit", sha)
    assert raw.split("\n\n", 1)[1] + "\n" == MESSAGE
    assert not (wt / "pwned").exists() and not (h.clone / "pwned").exists()


def test_commit_never_signs_even_if_the_repository_asks_for_it(
    make_harness: HarnessFactory, tmp_path: Path
) -> None:
    markers = tmp_path / "markers"
    markers.mkdir()
    gpg = tmp_path / "fake-gpg.sh"
    gpg.write_text(f"#!/bin/sh\necho ran >> '{markers}/gpg'\nexit 1\n")
    gpg.chmod(0o755)
    control = make_harness(root=tmp_path / "control")
    subject = make_harness(root=tmp_path / "subject")
    for h in (control, subject):
        h.git.git("config", "commit.gpgsign", "true")
        h.git.git("config", "gpg.program", str(gpg))

    failed = hostile_config_commit_control(control)
    assert failed.returncode != 0 and fired(markers) == {"gpg"}  # plain git tries to sign
    (markers / "gpg").unlink()

    wt, _base = subject.worktree()
    put(wt / "docs/signed.md", "x\n")
    subject.ops.stage_all(wt)
    sha = subject.ops.commit(wt, "unsigned\n")

    assert "gpgsig" not in subject.git.git("cat-file", "commit", sha)
    assert fired(markers) == set()


def test_commit_with_nothing_staged_is_an_error(h: Harness) -> None:
    wt, _base = h.worktree()
    with pytest.raises(GitError, match="git commit"):
        h.ops.commit(wt, "empty\n")


# ----------------------------------------------------------------------------- push
def test_push_publishes_exactly_one_branch_and_never_forces(h: Harness) -> None:
    wt, _base, sha = h.commit_file("r1")
    h.commit_file("r2")  # a second local branch that must stay local
    main_before = h.git.origin_git("rev-parse", "refs/heads/main")

    h.ops.push("agent/issue-1-r1")

    assert h.git.origin_branches() == ["agent/issue-1-r1", "main"]
    assert h.git.origin_git("rev-parse", "refs/heads/agent/issue-1-r1") == sha
    assert h.git.origin_git("rev-parse", "refs/heads/main") == main_before
    spawn = h.spawns.of("push")[0]
    assert spawn.args == [
        "push",
        "--quiet",
        "origin",
        "refs/heads/agent/issue-1-r1:refs/heads/agent/issue-1-r1",
    ]
    assert not any(a in ("-f", "--force") or a.startswith(("--force", "+")) for a in spawn.args)


def test_push_allows_fast_forwards_and_repeats_of_the_same_commit(h: Harness) -> None:
    wt, _base, first = h.commit_file("r1")
    h.ops.push("agent/issue-1-r1")
    h.ops.push("agent/issue-1-r1")  # nothing new: still fine

    put(wt / "docs/more.md", "more\n")
    h.ops.stage_all(wt)
    second = h.ops.commit(wt, "more\n")
    h.ops.push("agent/issue-1-r1")

    assert h.ops.remote_branch_sha("agent/issue-1-r1") == second != first


def test_push_to_a_diverged_remote_branch_fails_and_leaves_the_remote_alone(h: Harness) -> None:
    theirs = advance_origin(h, "docs/theirs.md", "theirs\n", branch="agent/issue-1-r1")
    h.ops.fetch()
    wt, _base = h.worktree("r1", "agent/issue-1-r1")  # branched from main: it has diverged
    put(wt / "docs/ours.md", "ours\n")
    h.ops.stage_all(wt)
    h.ops.commit(wt, "ours\n")

    with pytest.raises(GitError, match="rejected|failed to push") as raised:
        h.ops.push("agent/issue-1-r1")

    assert "rejected" in raised.value.stderr
    assert h.ops.remote_branch_sha("agent/issue-1-r1") == theirs
    assert h.git.origin_git("rev-parse", "refs/heads/agent/issue-1-r1") == theirs


def test_remote_branch_sha_is_exact_and_none_for_unknown_branches(h: Harness) -> None:
    _wt, _base, sha = h.commit_file("r1")
    h.ops.push("agent/issue-1-r1")

    assert h.ops.remote_branch_sha("agent/issue-1-r1") == sha
    assert h.ops.remote_branch_sha("agent/issue-1-r") is None  # a prefix is not a match
    assert h.ops.remote_branch_sha("agent/issue-1-r10") is None
    assert h.ops.remote_branch_sha("agent/issue-9-none") is None
    assert h.ops.remote_branch_sha("main") == h.git.origin_git("rev-parse", "refs/heads/main")


# ------------------------------------------------------------------------ token mode
def network_spawns(h: Harness) -> list[Spawn]:
    return [c for c in h.spawns.calls if c.subcommand in {"fetch", "push", "ls-remote"}]


def test_token_mode_gives_git_an_askpass_helper_without_exposing_the_token(
    make_harness: HarnessFactory,
) -> None:
    h = make_harness(token=TOKEN, push_auth="token")
    previous_environ = dict(os.environ)

    run_full_lifecycle(h)

    script = h.state_dir / "git-askpass.sh"
    assert TOKEN not in script.read_text()
    assert stat.S_IMODE(script.stat().st_mode) == 0o700 and os.access(script, os.X_OK)
    network = network_spawns(h)
    assert {c.subcommand for c in network} == {"fetch", "push", "ls-remote"}
    for call in network:
        assert call.env["GIT_ASKPASS"] == str(script)
        assert call.env["LAWNBERRY_BRIDGE_PUSH_TOKEN"] == TOKEN
        assert "credential.helper=" in call.config  # a stored helper must not answer first
    local = [c for c in h.spawns.calls if c not in network]
    assert local
    for call in local:  # the secret reaches only the commands that talk to the remote
        assert "GIT_ASKPASS" not in call.env and "LAWNBERRY_BRIDGE_PUSH_TOKEN" not in call.env
        assert "credential.helper=" not in call.config
    assert all(TOKEN not in " ".join(c.argv) for c in h.spawns.calls)  # never in argv
    assert dict(os.environ) == previous_environ  # child environment only


def test_the_askpass_helper_answers_git_credential_prompts(make_harness: HarnessFactory) -> None:
    h = make_harness(token=TOKEN, push_auth="token")
    h.ops.fetch()
    env = h.spawns.of("fetch")[0].env
    script = h.state_dir / "git-askpass.sh"

    answers = {
        prompt: subprocess.run(
            [str(script), prompt],
            env={"LAWNBERRY_BRIDGE_PUSH_TOKEN": TOKEN},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for prompt in ("Username for 'https://github.test': ", "Password for 'https://x@h': ")
    }
    assert answers == {
        "Username for 'https://github.test': ": "x-access-token\n",
        "Password for 'https://x@h': ": f"{TOKEN}\n",
    }

    # and git itself, driven with the environment the bridge builds, gets both answers
    filled = subprocess.run(
        ["git", "-c", "credential.helper=", "credential", "fill"],
        input=b"protocol=https\nhost=github.test\n\n",
        env=env,
        capture_output=True,
        cwd=h.clone,
        check=True,
    ).stdout.decode()
    assert "username=x-access-token\n" in filled and f"password={TOKEN}\n" in filled


def test_the_askpass_helper_is_restored_if_it_was_tampered_with(
    make_harness: HarnessFactory,
) -> None:
    h = make_harness(token=TOKEN, push_auth="token")
    h.ops.fetch()
    script = h.state_dir / "git-askpass.sh"
    original = script.read_text()
    script.write_text("#!/bin/sh\necho stolen\n")
    script.chmod(0o755)

    h.ops.fetch()

    assert script.read_text() == original
    assert stat.S_IMODE(script.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("push_auth", "token"),
    [("ambient", TOKEN), ("ambient", None), ("token", None)],
    ids=["ambient-with-token", "ambient-no-token", "token-mode-without-a-token"],
)
def test_git_gets_no_askpass_wiring_unless_token_mode_has_a_token(
    make_harness: HarnessFactory, push_auth: str, token: str | None
) -> None:
    h = make_harness(token=token, push_auth=push_auth)

    run_full_lifecycle(h)

    for call in h.spawns.calls:
        assert "GIT_ASKPASS" not in call.env
        assert "LAWNBERRY_BRIDGE_PUSH_TOKEN" not in call.env
        assert token is None or token not in " ".join(call.argv)
        if push_auth == "ambient":  # ambient mode leaves the operator's credential helpers alone
            assert "credential.helper=" not in call.config
    assert not (h.state_dir / "git-askpass.sh").exists()


def test_the_askpass_helper_directory_is_created_on_demand(
    make_harness: HarnessFactory, tmp_path: Path
) -> None:
    h = make_harness(token=TOKEN, push_auth="token")
    nested = tmp_path / "does" / "not" / "exist"
    ops = GitOps(h.clone, h.cfg, state_dir=nested, token=TOKEN, environ=h.environ)

    ops.fetch()

    assert (nested / "git-askpass.sh").exists()


def test_git_errors_never_carry_the_token(make_harness: HarnessFactory, tmp_path: Path) -> None:
    leaked = (
        f"remote: Invalid credentials {TOKEN}\n"
        f"fatal: Authentication failed for 'https://x-access-token:{TOKEN}@github.test/o/r.git'\n"
        f"also a shaped one ghp_{'A1b2C3d4E5' * 4}"
    )

    def failing(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 128, stdout=b"", stderr=leaked.encode())

    ops = GitOps(
        tmp_path / "repo",
        GitConfig(push_auth="token"),
        state_dir=tmp_path / "state",
        token=TOKEN,
        environ={"PATH": "/usr/bin"},
        run=failing,
    )

    with pytest.raises(GitError) as raised:
        ops.push("agent/issue-1-abc")

    error = raised.value
    assert "Authentication failed" in str(error)
    for text in (str(error), error.stderr, repr(error), " ".join(error.argv)):
        assert TOKEN not in text and "ghp_A1b2C3d4E5" not in text
    assert "[REDACTED" in error.stderr


# --------------------------------------------------- scrubbed environment, hardened argv
def test_every_git_process_is_hardened_and_non_interactive(h: Harness) -> None:
    run_full_lifecycle(h)

    assert len(h.spawns.calls) > 15
    for call in h.spawns.calls:
        assert call.argv[0] == "git"
        for setting in HARDENING:
            assert setting in call.config, f"{setting} missing from {call.argv}"
        assert call.env["GIT_TERMINAL_PROMPT"] == "0"
        assert call.env["GIT_OPTIONAL_LOCKS"] == "0"
        assert call.env["LC_ALL"] == "C"
        assert call.kwargs["capture_output"] is True and call.kwargs["check"] is False
        assert call.kwargs["timeout"] == 60
        if call.subcommand != "commit":
            assert call.kwargs["stdin"] == subprocess.DEVNULL or call.kwargs["input"]


ALLOWED = {
    "PATH": "/usr/bin:/bin",
    "HOME": "{home}",
    "USER": "tester",
    "LOGNAME": "tester",
    "LANG": "C.UTF-8",
    "TMPDIR": "/tmp",
    "SSH_AUTH_SOCK": "/tmp/agent.sock",
    "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
    "XDG_CONFIG_HOME": "{home}/.config",
    "XDG_RUNTIME_DIR": "/run/user/1000",
    "HTTPS_PROXY": "http://proxy.invalid:3128",
    "HTTP_PROXY": "http://proxy.invalid:3128",
    "NO_PROXY": "localhost",
    "https_proxy": "http://proxy.invalid:3128",
    "http_proxy": "http://proxy.invalid:3128",
    "no_proxy": "localhost",
    "SSL_CERT_FILE": "/etc/ssl/cert.pem",
    "SSL_CERT_DIR": "/etc/ssl/certs",
}
FORBIDDEN = {
    "GITHUB_TOKEN": "ghp-leak",
    "GH_TOKEN": "gh-leak",
    "LAWNBERRY_BRIDGE_GITHUB_TOKEN": "bridge-leak",
    "AWS_SECRET_ACCESS_KEY": "aws-leak",
    "ANTHROPIC_API_KEY": "anthropic-leak",
    "GIT_ASKPASS": "/bin/evil",
    "SSH_ASKPASS": "/bin/evil",
    "GIT_SSH": "/bin/evil",
    "GIT_PROXY_COMMAND": "evil",
    "GIT_DIR": "/elsewhere",
    "GIT_WORK_TREE": "/elsewhere",
    "GIT_INDEX_FILE": "/elsewhere/index",
    "GIT_CONFIG_GLOBAL": "/evil/gitconfig",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "/evil",
    "GIT_CONFIG_PARAMETERS": "'core.pager=evil'",
    "GIT_EXEC_PATH": "/evil",
    "GIT_EXTERNAL_DIFF": "/evil",
    "GIT_EDITOR": "evil",
    "GIT_ALLOW_PROTOCOL": "ext",
    "EDITOR": "evil",
    "PAGER": "evil",
    "LD_PRELOAD": "/evil.so",
    "PYTHONPATH": "/evil",
}
ALWAYS_SET = {"LC_ALL", "GIT_TERMINAL_PROMPT", "GIT_OPTIONAL_LOCKS", "GIT_CONFIG_NOSYSTEM"}
COMMIT_IDENTITY = {
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
}


def test_only_an_allowlist_of_the_ambient_environment_reaches_git(
    make_harness: HarnessFactory, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    allowed = {k: v.format(home=home) for k, v in ALLOWED.items() if k not in {"PATH", "HOME"}}
    h = make_harness(environ_extra={**allowed, **FORBIDDEN})

    run_full_lifecycle(h)

    expected = set(ALLOWED) | set(allowed) | ALWAYS_SET
    for call in h.spawns.calls:
        assert set(call.env) <= expected | COMMIT_IDENTITY, set(call.env) - expected
        assert FORBIDDEN.keys().isdisjoint(call.env)
        for name, value in allowed.items():
            assert call.env[name] == value
        assert call.env["HOME"] == str(h.git.home)
        assert call.env["GIT_CONFIG_NOSYSTEM"] == "1"
        if call.subcommand != "commit":
            assert COMMIT_IDENTITY.isdisjoint(call.env)


def test_the_real_process_environment_is_filtered_when_none_is_given(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-leak")
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    spawns = SpawnLog()
    ops = GitOps(h.clone, h.cfg, state_dir=h.state_dir, run=spawns)  # environ=None: os.environ

    ops.toplevel()

    env = spawns.calls[0].env
    assert "GITHUB_TOKEN" not in env and "GIT_DIR" not in env
    assert env["PATH"] == os.environ["PATH"] and env["GIT_CONFIG_NOSYSTEM"] == "1"


# --------------------------------------------------------------------- failures
def stub_ops(tmp_path: Path, run: Callable[..., Any], **cfg: Any) -> GitOps:
    return GitOps(
        tmp_path / "repo",
        GitConfig(**cfg),
        state_dir=tmp_path / "state",
        environ={"PATH": "/usr/bin"},
        run=run,
    )


def test_a_git_timeout_is_a_git_error_and_uses_the_configured_limit(tmp_path: Path) -> None:
    seen: list[float] = []

    def too_slow(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    ops = stub_ops(tmp_path, too_slow, git_timeout_seconds=7.5)

    with pytest.raises(GitError, match=r"fetch timed out after 7\.5s") as raised:
        ops.fetch()

    assert seen == [7.5]
    assert raised.value.argv[0] == "fetch"
    with pytest.raises(GitError, match="timed out"):
        ops.toplevel()


def test_a_missing_git_binary_is_a_git_error(tmp_path: Path) -> None:
    def no_git(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(2, "No such file or directory", "git")

    with pytest.raises(GitError, match="cannot run git"):
        stub_ops(tmp_path, no_git).toplevel()


def test_failing_commands_raise_git_errors_with_the_command_and_stderr(h: Harness) -> None:
    not_a_repo = h.tmp / "not-a-repo"
    not_a_repo.mkdir()

    with pytest.raises(GitError, match=r"failed \(128\)") as raised:
        h.ops.worktree_head(not_a_repo)
    assert raised.value.argv == ["rev-parse", "HEAD"]
    assert "not a git repository" in raised.value.stderr
    with pytest.raises(GitError, match="cannot run git"):
        h.ops.worktree_head(h.tmp / "does-not-exist")  # cwd missing: the spawn itself fails


def test_git_stderr_is_redacted_with_the_shared_redactor(
    make_harness: HarnessFactory, tmp_path: Path
) -> None:
    secret = "sup3r-s3cret-v4lue"
    shared = Redactor([secret])

    def noisy(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 1, b"", f"fatal: bad {secret}\n".encode())

    ops = GitOps(
        tmp_path / "repo",
        GitConfig(),
        state_dir=tmp_path / "s",
        redactor=shared,
        environ={},
        run=noisy,
    )

    with pytest.raises(GitError) as raised:
        ops.resolve_base()

    assert secret not in str(raised.value) and secret not in raised.value.stderr
    assert "[REDACTED:literal]" in raised.value.stderr
