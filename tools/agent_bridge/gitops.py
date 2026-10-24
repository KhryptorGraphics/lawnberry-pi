"""Git operations for task worktrees: create, inspect, commit, push, clean up.

All git commands run through one choke point (:meth:`GitOps._run`) with hooks disabled, a
scrubbed environment and no interactive prompts.  Repository hooks are never executed: they live
outside the worktree and the agent must not be able to turn a commit into code execution.
"""

from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import GitConfig
from .errors import GitError
from .policy import Change
from .redact import Redactor

__all__ = ["GitOps", "StagedDiff", "worktree_pointer"]

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

    def staged_changes(self, path: Path, base_sha: str, *, max_diff_bytes: int) -> StagedDiff:
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
                    binary=binary,
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
