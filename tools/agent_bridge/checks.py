"""Named verification checks (lint, tests) run by the bridge itself inside a task worktree.

Commands come only from the bridge configuration; an issue can request a check by name but can
never supply a command line.

In ``code`` scope a check runs agent-written code (``pytest`` imports the agent's tests and
``conftest.py``) as the workstation user, so it starts with a scrubbed environment: a private,
empty ``HOME`` that is removed when the check ends, a pinned ``PATH`` and no inherited
credentials.  A check that needs a credential must have it configured explicitly (``env`` of the
check).  Without an operating-system sandbox this does not stop a check from reading a file it
names by absolute path; ``agent.command_prefix`` is where a sandbox goes.  What protects the
publication is the integrity gate around the checks (``worker._verify_and_commit``).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CheckConfig
from .errors import CheckAborted
from .process import ProcessSpec, SubprocessRunner
from .redact import Redactor

__all__ = ["CheckResult", "run_checks"]

# The PATH of a check: these system directories, preceded by the directory of the bridge's own
# interpreter so that the python, pytest and ruff of its environment win over system copies.
CHECK_PATH_DIRS: tuple[str, ...] = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
CHECK_IDENTITY = "agent-bridge"


@dataclass(frozen=True)
class CheckResult:
    name: str
    argv: list[str]
    passed: bool
    returncode: int | None
    duration_s: float
    timed_out: bool
    output_tail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tail(text: str, limit: int) -> str:
    """Last ``limit`` characters of ``text``: a failing check reports its cause at the end."""
    if len(text) <= limit:
        return text
    return "... [earlier output omitted]\n" + text[-limit:]


def _check_env(home: Path, source: Mapping[str, str]) -> dict[str, str]:
    """The environment of one check: a private ``HOME``, nothing inherited but the locale."""
    interpreter = os.path.dirname(sys.executable)
    path_dirs = list(CHECK_PATH_DIRS)
    if interpreter and interpreter not in path_dirs:
        path_dirs.insert(0, interpreter)
    env = {
        "PATH": os.pathsep.join(path_dirs),
        "HOME": str(home),
        "TMPDIR": str(home / "tmp"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "USER": CHECK_IDENTITY,
        "LOGNAME": CHECK_IDENTITY,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if "LANG" in source:
        env["LANG"] = source["LANG"]
    return env


def _remove_tree(path: Path) -> None:
    """Delete ``path`` even when the check left read-only or unreadable directories in it.

    Best effort: a check that nests directories deeper than ``shutil`` can handle leaves its
    litter behind (never an error that would end the run).  Symbolic links are removed, not
    followed.
    """
    pending = [str(path)]
    while pending:
        directory = pending.pop()
        try:
            os.chmod(directory, 0o700)
            with os.scandir(directory) as entries:
                pending.extend(e.path for e in entries if e.is_dir(follow_symlinks=False))
        except OSError:
            continue
    with contextlib.suppress(RecursionError):
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def _private_home() -> Iterator[Path]:
    """A fresh empty ``HOME`` (with the directories tools expect below it), removed afterwards."""
    home = Path(tempfile.mkdtemp(prefix="agent-bridge-check-"))
    try:
        for sub in ("tmp", ".config", ".cache", ".local/share"):
            (home / sub).mkdir(parents=True)
        yield home
    finally:
        _remove_tree(home)


def run_checks(
    names: Sequence[str],
    checks: Mapping[str, CheckConfig],
    worktree: Path,
    *,
    runner: SubprocessRunner,
    redactor: Redactor,
    log_dir: Path,
    environ: Mapping[str, str] | None = None,
    command_prefix: Sequence[str] = (),
    abort_check: Callable[[], str | None] | None = None,
    abort_poll_s: float = 5.0,
) -> list[CheckResult]:
    """Run each named check in order, stopping at the first failure.

    Every check gets its own scrubbed environment (see the module docstring); ``environ`` only
    supplies the locale (``LANG``) and the check's configured ``env`` is applied on top.  The
    working directory is the worktree.

    ``abort_check`` is polled while a check runs (the worker also uses it to keep the run's lease
    alive); when it returns a reason the check is killed and :class:`CheckAborted` is raised.
    """
    source = os.environ if environ is None else environ
    results: list[CheckResult] = []
    for name in names:
        cfg = checks[name]
        with _private_home() as home:
            env = _check_env(home, source)
            env.update(cfg.env)
            spec = ProcessSpec(
                argv=[*command_prefix, *cfg.argv],
                cwd=worktree,
                env=env,
                timeout_s=cfg.timeout_seconds,
                kill_grace_s=5.0,
                max_capture_bytes=2_000_000,
                stdout_log=log_dir / f"check-{name}.stdout.log",
                stderr_log=log_dir / f"check-{name}.stderr.log",
            )
            if abort_check is None:
                outcome = runner.run(spec)
            else:
                outcome = runner.run(spec, abort_check=abort_check, abort_poll_s=abort_poll_s)
        if outcome.kill_reason:
            raise CheckAborted(outcome.kill_reason)
        tail = (outcome.stdout + ("\n" + outcome.stderr if outcome.stderr else "")).strip()
        if outcome.spawn_error:
            tail = outcome.spawn_error
        passed = outcome.returncode == 0 and not outcome.timed_out and outcome.spawn_error is None
        results.append(
            CheckResult(
                name=name,
                argv=list(cfg.argv),
                passed=passed,
                returncode=outcome.returncode,
                duration_s=round(outcome.duration_s, 2),
                timed_out=outcome.timed_out,
                output_tail=_tail(redactor.redact(tail[-8000:]), 1500),
            )
        )
        if not passed:
            break
    return results
