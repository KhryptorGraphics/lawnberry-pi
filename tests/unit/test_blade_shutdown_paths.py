"""The paths that end a stop or the process must not skip the safing sequence (round 2).

BL-08: ``execute_emergency_stop`` cancelled in its first loop turn ran no stop path although the
latch was set. E07/E14: an overrunning path must keep counting as a stop in flight and must never be
cancelled. BL-05: ``systemctl reload`` sent SIGHUP, which makes uvicorn exit WITHOUT the lifespan
shutdown (so ``park_actuators`` never ran); the unit has no reload verb now.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from backend.src.core.globals import _safety_state
from backend.src.services import emergency_stop_service as ess

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "systemd" / "lawnberry-backend.service"


@pytest.fixture(autouse=True)
def _reset_latch():
    yield
    _safety_state["emergency_stop_active"] = False


def _recording_paths(monkeypatch, ran):
    def make(name):
        async def run(kind, source):
            ran.append(name)
            return ess._absent(ess.NOT_PRESENT, f"{name} (test)")

        return run

    for name, attr in (
        ("tractor", "_tractor"),
        ("robohat", "_robohat"),
        ("blade_service", "_blade_service"),
    ):
        monkeypatch.setattr(ess, attr, make(name))


# ----------------------------------------------------------------------------- BL-08


async def test_cancelling_the_stop_in_its_first_loop_turn_still_runs_every_path(monkeypatch):
    ran: list[str] = []
    _recording_paths(monkeypatch, ran)

    task = asyncio.ensure_future(ess.execute_emergency_stop("cancel-test"))
    await asyncio.sleep(0)  # the stop runs up to its first await, then the caller is cancelled
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)

    assert sorted(ran) == ["blade_service", "robohat", "tractor"], "the safing paths still ran"
    assert _safety_state["emergency_stop_active"] is True
    assert ess.stop_in_flight() is False, "and they are done: a clear is possible again"


async def test_a_stop_cancelled_while_its_paths_run_leaves_them_running_to_the_end(monkeypatch):
    release = asyncio.Event()
    finished: list[str] = []

    async def slow(kind, source):
        await release.wait()
        finished.append("slow")
        return ess._absent(ess.NOT_PRESENT, "slow")

    _recording_paths(monkeypatch, [])
    monkeypatch.setattr(ess, "_robohat", slow)

    task = asyncio.ensure_future(ess.execute_emergency_stop("cancel-test"))
    await asyncio.sleep(0.02)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert ess.stop_in_flight() is True, "the half-run path is still a stop in flight"
    release.set()
    await asyncio.sleep(0.05)
    assert finished == ["slow"] and ess.stop_in_flight() is False


# ------------------------------------------------------------------------- E07 and E14


async def test_an_overrunning_path_counts_as_a_stop_in_flight_and_is_never_cancelled(monkeypatch):
    release = asyncio.Event()
    outcome: list[str] = []

    async def slow(kind, source):
        try:
            await release.wait()
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append("finished")
        return ess._absent(ess.NOT_PRESENT, "slow")

    _recording_paths(monkeypatch, [])
    monkeypatch.setattr(ess, "_robohat", slow)

    report = await ess.execute_emergency_stop("overrun-test", path_timeout_s=0.05)

    robohat = report["subsystems"]["robohat"]
    assert robohat["status"] == "failed" and "still running" in robohat["error"]
    assert ess.stop_in_flight() is True, "an overrun path keeps blocking a clear (E07)"
    release.set()
    await asyncio.sleep(0.05)
    assert outcome == ["finished"], "the path was not cancelled at its timeout (E14)"
    assert ess.stop_in_flight() is False


# ------------------------------------------------------------------------------ BL-05


def _directives() -> list[str]:
    lines = [line.strip() for line in UNIT.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def test_the_backend_unit_has_no_reload_verb():
    directives = _directives()
    assert not [d for d in directives if d.startswith("ExecReload")], directives
    comments = "\n".join(
        line.strip() for line in UNIT.read_text().splitlines() if line.strip().startswith("#")
    )
    assert "NO ExecReload" in comments and "systemctl restart" in comments
    assert any(d.startswith("ExecStart=") and "--workers 1" in d for d in directives)
    assert "KillSignal=SIGTERM" in directives


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize(("sig", "shutdown_runs"), [(signal.SIGTERM, True), (signal.SIGHUP, False)])
def test_uvicorn_runs_the_lifespan_shutdown_on_sigterm_but_not_on_sighup(
    tmp_path, sig, shutdown_runs
):
    """Why the unit must not offer ``reload``: its SIGHUP skips the lifespan shutdown (the park)."""
    pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    marker = tmp_path / "marker"
    (tmp_path / "hup_app.py").write_text(
        textwrap.dedent(
            """
            import os
            from contextlib import asynccontextmanager

            from fastapi import FastAPI


            @asynccontextmanager
            async def lifespan(app):
                yield
                open(os.environ["MARKER"], "w").write("shutdown ran")


            app = FastAPI(lifespan=lifespan)
            """
        )
    )
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "hup_app:app", "--port", str(port), "--workers", "1"],
        cwd=tmp_path,
        env={**os.environ, "MARKER": str(marker)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.2):
                break
            assert proc.poll() is None, "uvicorn exited before it listened"
            time.sleep(0.1)
        else:
            pytest.fail("uvicorn never listened")
        proc.send_signal(sig)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert marker.exists() is shutdown_runs, (sig, proc.returncode)
