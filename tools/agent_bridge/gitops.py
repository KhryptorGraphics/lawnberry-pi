"""Git operations for task worktrees: create, inspect, commit, push, clean up.

All git commands run through one choke point (:meth:`GitOps._run`) with hooks disabled, a
scrubbed environment and no interactive prompts.  Repository hooks are never executed: they live
outside the worktree and the agent must not be able to turn a commit into code execution.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import GitConfig
from .errors import GitError
from .policy import MODE_GITLINK, MODE_SYMLINK, Change
from .redact import Redactor

__all__ = ["GitOps", "GitSnapshot", "StagedDiff", "worktree_pointer"]

GIT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "TMPDIR",
    "SSH_AUTH_SOCK",
    "GIT_SSH_COMMAND",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
SAFE_CONFIG: tuple[str, ...] = (
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "core.autocrlf=false",
    "core.quotepath=false",
    "gc.auto=0",
    "advice.detachedHead=false",
    "protocol.ext.allow=never",
)
ZERO_SHA = "0" * 40
# Default per-file limit of the content scan in GitOps.staged_changes: the default of
# ``limits.max_file_bytes``.  Blobs above it are refused as too large and are never read.
DEFAULT_MAX_FILE_BYTES = 1_000_000
# The scan reads blobs from ``git cat-file --batch`` in groups of at most this many bytes.
SCAN_BATCH_BYTES = 16 * 1024 * 1024
PUSH_TOKEN_ENV = "LAWNBERRY_BRIDGE_PUSH_TOKEN"

_ASKPASS_SCRIPT = """#!/bin/sh
case "$1" in
  Username*) printf '%s\\n' "x-access-token" ;;
  *) printf '%s\\n' "$LAWNBERRY_BRIDGE_PUSH_TOKEN" ;;
esac
"""


@dataclass
class StagedDiff:
    changes: list[Change] = field(default_factory=list)
    diff_text: str = ""
    diff_bytes: int = 0
    truncated: bool = False


def worktree_pointer(path: Path) -> str | None:
    """Content of the ``.git`` file of a linked worktree (its pointer to the shared repo)."""
    try:
        return (path / ".git").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _is_text(content: bytes) -> bool:
    """True for UTF-8 text without NUL bytes (the empty content counts as text)."""
    if b"\0" in content:
        return False
    if content.isascii():
        return True
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


@dataclass(frozen=True)
class GitSnapshot:
    """The shared git state that decides what a task branch publishes.

    Taken when a task worktree is created and compared after the agent phase and after the checks
    (:meth:`GitOps.state_snapshot`).  Agent-written code runs unsandboxed in both phases, so the
    gate covers what such code could use to publish something other than the reviewed tree: the
    repository's config (remote URLs, ``url.*.insteadOf``, credential and ssh settings), the
    worktree's ``.git`` pointer and HEAD, and the agent branches (``refs/heads/<branch_prefix>*``)
    that the bridge pushes.  Every other ref and worktree is ignored: the owner may work in the
    same clone while a run is in progress.
    """

    config: str  # sha256 of the shared repository's ``config`` file ("" when unreadable)
    pointer: str | None  # text of the worktree's ``.git`` file
    head: str  # commit HEAD names
    head_ref: str | None  # branch HEAD is attached to (None: detached)
    agent_refs: tuple[tuple[str, str], ...]  # (refname, commit [-> symref target]), sorted

    def differences(self, after: GitSnapshot) -> list[str]:
        """What changed between this snapshot and ``after``: one short sentence per difference."""
        found: list[str] = []
        if self.config != after.config:
            found.append("the repository's git config changed")
        if self.pointer != after.pointer:
            found.append("the worktree's .git pointer changed")
        if self.head != after.head:
            found.append(f"HEAD moved from {self.head[:12]} to {after.head[:12]}")
        if self.head_ref != after.head_ref:
            if after.head_ref is None:
                found.append(f"HEAD is now detached (was attached to {self.head_ref})")
            elif self.head_ref is None:
                found.append(f"HEAD is now attached to {after.head_ref} (was detached)")
            else:
                found.append(f"HEAD is now attached to {after.head_ref} (was {self.head_ref})")
        before, now = dict(self.agent_refs), dict(after.agent_refs)
        changed = sorted(name for name in before.keys() & now.keys() if before[name] != now[name])
        for label, names in (
            ("created", sorted(now.keys() - before.keys())),
            ("deleted", sorted(before.keys() - now.keys())),
            ("moved", changed),
        ):
            if names:
                more = f" and {len(names) - 5} more" if len(names) > 5 else ""
                found.append(f"agent branches {label}: {', '.join(names[:5])}{more}")
        return found


class GitOps:
    def __init__(
        self,
        repo_path: Path,
        cfg: GitConfig,
        *,
        state_dir: Path,
        token: str | None = None,
        redactor: Redactor | None = None,
        environ: Mapping[str, str] | None = None,
        run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.cfg = cfg
        self.state_dir = Path(state_dir)
        self._token = token
        self._redactor = redactor or Redactor()
        if token:
            self._redactor.add_literal(token)
        self._environ = dict(os.environ if environ is None else environ)
        self._subprocess_run = run

    # ---------------------------------------------------------------- plumbing
    def _env(self, *, auth: bool = False) -> dict[str, str]:
        env = {k: v for k, v in self._environ.items() if k in GIT_ENV_ALLOWLIST}
        env["LC_ALL"] = "C"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_CONFIG_NOSYSTEM"] = self._environ.get("GIT_CONFIG_NOSYSTEM", "0")
        if auth and self.cfg.push_auth == "token" and self._token:
            env["GIT_ASKPASS"] = str(self._askpass_path())
            env[PUSH_TOKEN_ENV] = self._token
        return env

    def _askpass_path(self) -> Path:
        script = self.state_dir / "git-askpass.sh"
        if not script.exists() or script.read_text(encoding="utf-8") != _ASKPASS_SCRIPT:
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text(_ASKPASS_SCRIPT, encoding="utf-8")
        script.chmod(stat.S_IRWXU)
        return script

    def _config_args(self, *, auth: bool = False, identity: bool = False) -> list[str]:
        args: list[str] = []
        for item in SAFE_CONFIG:
            args += ["-c", item]
        if auth and self.cfg.push_auth == "token":
            args += ["-c", "credential.helper="]
        if identity:
            args += [
                "-c",
                f"user.name={self.cfg.author_name}",
                "-c",
                f"user.email={self.cfg.author_email}",
                "-c",
                "commit.gpgsign=false",
            ]
        return args

    def _run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        input_bytes: bytes | None = None,
        check: bool = True,
        timeout: float | None = None,
        auth: bool = False,
        identity: bool = False,
        env_extra: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        argv = ["git", *self._config_args(auth=auth, identity=identity), *args]
        env = self._env(auth=auth)
        if env_extra:
            env.update(env_extra)
        try:
            done = self._subprocess_run(
                argv,
                cwd=str(cwd or self.repo_path),
                env=env,
                input=input_bytes,
                capture_output=True,
                timeout=timeout or self.cfg.git_timeout_seconds,
                check=False,
                stdin=None if input_bytes is not None else subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(
                f"git {args[0]} timed out after {exc.timeout}s", argv=list(args), stderr=""
            ) from exc
        except OSError as exc:
            raise GitError(f"cannot run git: {exc}", argv=list(args)) from exc
        if check and done.returncode != 0:
            stderr = self._redactor.redact(done.stderr.decode("utf-8", errors="replace")).strip()
            raise GitError(
                f"git {' '.join(args[:2])} failed ({done.returncode}): {stderr[:500]}",
                argv=list(args),
                stderr=stderr,
            )
        return done

    def _text(self, args: Sequence[str], **kwargs: Any) -> str:
        return self._run(args, **kwargs).stdout.decode("utf-8", errors="replace").strip()

    # ---------------------------------------------------------------- repository facts
    def toplevel(self) -> Path:
        return Path(self._text(["rev-parse", "--show-toplevel"])).resolve()

    def remote_url(self) -> str:
        return self._text(["remote", "get-url", self.cfg.remote])

    def push_urls(self) -> list[str]:
        """Every URL ``git push <remote>`` sends to: ``pushurl`` and the rewrite rules applied.

        A push does not necessarily go where a fetch does (``remote.<name>.pushurl``,
        ``url.<base>.pushInsteadOf``), and with several URLs configured it goes to all of them.
        """
        return self._text(["remote", "get-url", "--push", "--all", self.cfg.remote]).splitlines()

    def check_branch_name(self, branch: str) -> bool:
        done = self._run(["check-ref-format", "--branch", branch], check=False)
        return done.returncode == 0

    # ---------------------------------------------------------------- base commit
    def fetch(self) -> None:
        remote, base = self.cfg.remote, self.cfg.base_branch
        self._run(
            [
                "fetch",
                "--no-tags",
                "--prune",
                remote,
                f"+refs/heads/{base}:refs/remotes/{remote}/{base}",
            ],
            auth=True,
        )

    def resolve_base(self) -> str:
        return self._text(
            [
                "rev-parse",
                "--verify",
                f"refs/remotes/{self.cfg.remote}/{self.cfg.base_branch}^{{commit}}",
            ]
        )

    # ---------------------------------------------------------------- worktrees
    def create_worktree(self, path: Path, branch: str, base_sha: str) -> str:
        """Check out ``base_sha`` on a new branch in ``path``; return the ``.git`` pointer."""
        if not self.check_branch_name(branch):
            raise GitError(f"invalid branch name {branch!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._run(["worktree", "add", "--quiet", "-b", branch, str(path), base_sha])
        pointer = worktree_pointer(path)
        if pointer is None:
            raise GitError(f"worktree {path} has no .git pointer")
        return pointer

    def remove_worktree(self, path: Path, branch: str | None = None) -> None:
        if path.exists():
            self._run(["worktree", "remove", "--force", str(path)], check=False)
        self._run(["worktree", "prune"], check=False)
        if branch:
            self._run(["branch", "-D", branch], check=False)

    def worktree_head(self, path: Path) -> str:
        return self._text(["rev-parse", "HEAD"], cwd=path)

    def worktree_branch(self, path: Path) -> str:
        return self._text(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path)

    def list_worktree_paths(self) -> list[Path]:
        out = self._text(["worktree", "list", "--porcelain"])
        return [
            Path(line[len("worktree ") :])
            for line in out.splitlines()
            if line.startswith("worktree ")
        ]

    # ---------------------------------------------------------------- inspection
    def stage_all(self, path: Path) -> None:
        self._run(["add", "-A", "--", "."], cwd=path)

    def write_tree(self, path: Path) -> str:
        """Tree id of the current index (used to prove the checks left the tree untouched)."""
        return self._text(["write-tree"], cwd=path)

    def staged_changes(
        self,
        path: Path,
        base_sha: str,
        *,
        max_diff_bytes: int,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> StagedDiff:
        """Describe everything staged against ``base_sha`` (modes, sizes, line counts, diff).

        ``Change.binary`` is git's own verdict or, for every added or changed regular file of at
        most ``max_file_bytes``, a NUL byte or invalid UTF-8 anywhere in the content.
        """
        raw = self._run(
            [
                "diff",
                "--cached",
                "--raw",
                "-z",
                "--no-abbrev",
                "--no-renames",
                "--no-ext-diff",
                base_sha,
            ],
            cwd=path,
        ).stdout
        numstat = self._run(
            ["diff", "--cached", "--numstat", "-z", "--no-renames", "--no-ext-diff", base_sha],
            cwd=path,
        ).stdout

        entries: list[tuple[str, str, str, str, str, str]] = []
        tokens = raw.split(b"\0")
        index = 0
        while index < len(tokens):
            head = tokens[index]
            if not head.startswith(b":"):
                index += 1
                continue
            fields = head[1:].decode("ascii", errors="replace").split(" ")
            if len(fields) < 5 or index + 1 >= len(tokens):
                break
            old_mode, new_mode, old_sha, new_sha, status = fields[:5]
            name = tokens[index + 1].decode("utf-8", errors="surrogateescape")
            entries.append((old_mode, new_mode, old_sha, new_sha, status[:1], name))
            index += 2

        stats: dict[str, tuple[int, int, bool]] = {}
        for token in numstat.split(b"\0"):
            if not token:
                continue
            parts = token.split(b"\t", 2)
            if len(parts) != 3:
                continue
            added_raw, deleted_raw, name_raw = parts
            name = name_raw.decode("utf-8", errors="surrogateescape")
            if added_raw == b"-" or deleted_raw == b"-":
                stats[name] = (0, 0, True)
            else:
                stats[name] = (int(added_raw), int(deleted_raw), False)

        sizes = self._blob_sizes(
            [e[3] for e in entries if e[4] != "D" and e[3] != ZERO_SHA and e[1] != "160000"]
        )
        # numstat only sees a NUL in the first 8000 bytes; read what the file limit lets through
        # (a bigger blob is refused as too large anyway) to catch a payload that turns binary later
        non_text = self._non_text_blobs(
            [
                sha
                for _old_mode, new_mode, _old_sha, sha, status, _name in entries
                if status != "D"
                and sha != ZERO_SHA
                and new_mode not in (MODE_GITLINK, MODE_SYMLINK)
                and 0 < sizes.get(sha, 0) <= max_file_bytes
            ],
            sizes,
        )
        changes: list[Change] = []
        for old_mode, new_mode, _old_sha, new_sha, status, name in entries:
            added, deleted, binary = stats.get(name, (0, 0, False))
            changes.append(
                Change(
                    path=name,
                    status=status,
                    old_mode=old_mode,
                    new_mode=new_mode,
                    size=sizes.get(new_sha),
                    binary=binary or new_sha in non_text,
                    added=added,
                    deleted=deleted,
                )
            )

        diff = self._run(
            [
                "diff",
                "--cached",
                "-U0",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                base_sha,
            ],
            cwd=path,
        ).stdout
        limit = max(1, max_diff_bytes) * 2
        truncated = len(diff) > limit
        return StagedDiff(
            changes=changes,
            diff_text=diff[:limit].decode("utf-8", errors="replace"),
            diff_bytes=len(diff),
            truncated=truncated,
        )

    def _blob_sizes(self, shas: Sequence[str]) -> dict[str, int]:
        unique = sorted(set(shas))
        if not unique:
            return {}
        out = self._run(
            ["cat-file", "--batch-check"], input_bytes=("\n".join(unique) + "\n").encode("ascii")
        ).stdout.decode("ascii", errors="replace")
        sizes: dict[str, int] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2].isdigit():
                sizes[parts[0]] = int(parts[2])
        return sizes

    def _non_text_blobs(self, shas: Sequence[str], sizes: Mapping[str, int]) -> set[str]:
        """Ids among ``shas`` whose content is not UTF-8 text: a NUL byte or invalid UTF-8.

        ``git diff --numstat`` only looks for a NUL in the first 8000 bytes, so a file may start
        as text and turn binary later.  The blobs are read through ``cat-file --batch`` in groups
        of at most ``SCAN_BATCH_BYTES``, so the bridge never holds much more than that at once.
        """
        pending = sorted(set(shas))
        flagged: set[str] = set()
        position = 0
        while position < len(pending):
            batch: list[str] = []
            total = 0
            while position < len(pending) and (
                not batch or total + sizes[pending[position]] <= SCAN_BATCH_BYTES
            ):
                total += sizes[pending[position]]
                batch.append(pending[position])
                position += 1
            out = self._run(
                ["cat-file", "--batch"], input_bytes=("\n".join(batch) + "\n").encode("ascii")
            ).stdout
            offset = 0
            for sha in batch:
                end = out.find(b"\n", offset)
                if end < 0:
                    break
                header = out[offset:end].split()
                offset = end + 1
                if len(header) != 3 or not header[2].isdigit():
                    continue  # "<sha> missing"
                size = int(header[2])
                if not _is_text(out[offset : offset + size]):
                    flagged.add(sha)
                offset += size + 1
        return flagged

    # ---------------------------------------------------------------- commit & push
    def commit(self, path: Path, message: str) -> str:
        self._run(
            ["commit", "--quiet", "--no-verify", "--no-gpg-sign", "-F", "-"],
            cwd=path,
            input_bytes=message.encode("utf-8"),
            identity=True,
            env_extra={
                "GIT_AUTHOR_NAME": self.cfg.author_name,
                "GIT_AUTHOR_EMAIL": self.cfg.author_email,
                "GIT_COMMITTER_NAME": self.cfg.author_name,
                "GIT_COMMITTER_EMAIL": self.cfg.author_email,
            },
        )
        return self.worktree_head(path)

    def push(self, branch: str) -> None:
        """Push ``branch`` (never forced) to the configured remote."""
        self._run(
            ["push", "--quiet", self.cfg.remote, f"refs/heads/{branch}:refs/heads/{branch}"],
            auth=True,
        )

    def remote_branch_sha(self, branch: str) -> str | None:
        out = self._text(
            ["ls-remote", "--heads", self.cfg.remote, f"refs/heads/{branch}"], auth=True
        )
        return out.split()[0] if out else None

    # ---------------------------------------------------------------- shared-state integrity
    def state_snapshot(self, path: Path) -> GitSnapshot:
        """Fingerprint the shared git state a run must not change (see :class:`GitSnapshot`).

        Raises :class:`GitError` when git cannot read the repository any more: an unreadable
        repository must never look unchanged.
        """
        common = Path(self._text(["rev-parse", "--git-common-dir"], cwd=path))
        if not common.is_absolute():
            common = path / common
        try:
            config = hashlib.sha256((common / "config").read_bytes()).hexdigest()
        except OSError:
            config = ""
        symbolic = self._run(["symbolic-ref", "-q", "HEAD"], cwd=path, check=False)
        if symbolic.returncode not in (0, 1):  # 1: HEAD is detached
            stderr = self._redactor.redact(symbolic.stderr.decode("utf-8", errors="replace"))
            raise GitError(
                f"git symbolic-ref failed ({symbolic.returncode}): {stderr.strip()[:500]}",
                argv=["symbolic-ref", "-q", "HEAD"],
                stderr=stderr.strip(),
            )
        listing = self._text(
            [
                "for-each-ref",
                "--format=%(objectname) %(refname) %(symref)",
                f"refs/heads/{self.cfg.branch_prefix}",
            ],
            cwd=path,
        )
        refs: list[tuple[str, str]] = []
        for line in listing.splitlines():
            commit, _, rest = line.partition(" ")
            name, _, symref = rest.partition(" ")
            refs.append((name, f"{commit} -> {symref.strip()}" if symref.strip() else commit))
        return GitSnapshot(
            config=config,
            pointer=worktree_pointer(path),
            head=self.worktree_head(path),
            head_ref=symbolic.stdout.decode("utf-8", errors="replace").strip() or None,
            agent_refs=tuple(sorted(refs)),
        )

    def commit_parents(self, sha: str) -> list[str]:
        """The parents recorded inside commit ``sha`` itself.

        Read from the raw object, so replacement refs and grafts cannot make a commit look like
        it sits on the base commit.
        """
        raw = self._run(["--no-replace-objects", "cat-file", "commit", sha]).stdout
        parents: list[str] = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line:  # the header ends at the first empty line
                break
            if line.startswith("parent "):
                parents.append(line[len("parent ") :].strip())
        return parents
