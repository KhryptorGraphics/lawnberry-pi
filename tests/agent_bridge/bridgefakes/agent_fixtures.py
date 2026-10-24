"""Helpers for driving the fake agent CLIs (``fake_omp.py`` / ``fake_claude.py``) from tests.

Plain functions, not a pytest plugin: a conftest imports and wraps them.  The scenario and
invocation-log paths reach the fakes through the subprocess environment (see
:func:`fake_agent_env`); the wrappers written by :func:`install_fake_cli` set no environment.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

SCENARIO_ENV = "FAKE_AGENT_SCENARIO"
LOG_ENV = "FAKE_AGENT_LOG"
FAKES_DIR = Path(__file__).resolve().parent

__all__ = [
    "FAKES_DIR",
    "LOG_ENV",
    "SCENARIO_ENV",
    "docs_edit_steps",
    "fake_agent_env",
    "install_fake_cli",
    "pid_is_alive",
    "read_invocations",
    "wait_for_file",
    "write_scenario",
]


def install_fake_cli(
    directory: Path, kind: Literal["omp", "claude"], *, name: str | None = None
) -> Path:
    """Write ``directory/<name or kind>``, a POSIX shell wrapper that execs the fake CLI.

    The wrapper runs ``<sys.executable> <fake_<kind>.py> "$@"``; the returned path can be used
    as ``agent.<kind>.binary`` in a bridge configuration.
    """
    if kind not in ("omp", "claude"):
        raise ValueError(f"unknown fake CLI kind: {kind!r}")
    script = FAKES_DIR / f"fake_{kind}.py"
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / (name or kind)
    wrapper.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def write_scenario(
    path: Path,
    steps: list[dict[str, Any]],
    *,
    log_env: Sequence[str] = (),
    ignore_sigterm: bool = False,
) -> Path:
    """Write a scenario file for the fakes and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"steps": steps, "log_env": list(log_env), "ignore_sigterm": ignore_sigterm}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def read_invocations(log_path: Path) -> list[dict[str, Any]]:
    """Parse the invocation log (one JSON object per line); empty if the fake never ran."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def fake_agent_env(scenario: Path | None = None, log: Path | None = None) -> dict[str, str]:
    """The environment variables that point a fake CLI at its scenario and invocation log."""
    env: dict[str, str] = {}
    if scenario is not None:
        env[SCENARIO_ENV] = str(scenario)
    if log is not None:
        env[LOG_ENV] = str(log)
    return env


def pid_is_alive(pid: int) -> bool:
    """True while ``pid`` is a live process; a zombie awaiting its reaper counts as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return not Path("/proc/self").exists()  # with /proc mounted the process just vanished
    except OSError:
        return True
    # The process name may contain spaces or parentheses: the state follows the last ")".
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def wait_for_file(path: Path, timeout: float = 10.0) -> Path:
    """Poll until ``path`` exists and is not empty; raise TimeoutError otherwise."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if path.stat().st_size > 0:
                return path
        except FileNotFoundError:
            pass
        time.sleep(0.02)
    raise TimeoutError(f"{path} did not appear within {timeout} seconds")


def docs_edit_steps(
    path: str = "docs/agent-note.md",
    content: str = "Hello from the agent\n",
    cost: float = 0.02,
    report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """A harmless documentation edit: write the file, report the tool call, spend, then finish.

    The final text carries the fenced ``agent-result`` block with ``report`` (default: status
    ``completed``, a summary and ``files_changed`` listing ``path``).
    """
    if report is None:
        report = {
            "status": "completed",
            "summary": f"Created {path}",
            "files_changed": [path],
        }
    size = len(content.encode("utf-8"))
    return [
        {"op": "write_file", "path": path, "content": content},
        {
            "op": "tool_call",
            "tool": "write",
            "args": {"path": path, "content": content},
            "result": f"Successfully wrote {size} bytes to {path}",
        },
        {"op": "cost", "usd": cost},
        {"op": "final", "text": f"Created {path}.", "report": report},
    ]
