"""Named verification checks (lint, tests) run by the bridge itself inside a task worktree.

Commands come only from the bridge configuration; an issue can request a check by name but can
never supply a command line.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CheckConfig
from .errors import CheckAborted
from .process import ProcessSpec, SubprocessRunner
from .redact import Redactor

__all__ = ["CheckResult", "run_checks"]

CHECK_ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "TMPDIR")


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

    ``abort_check`` is polled while a check runs (the worker also uses it to keep the run's lease
    alive); when it returns a reason the check is killed and :class:`CheckAborted` is raised.
    """
    source = os.environ if environ is None else environ
    results: list[CheckResult] = []
    for name in names:
        cfg = checks[name]
        env = {k: v for k, v in source.items() if k in CHECK_ENV_ALLOWLIST}
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
