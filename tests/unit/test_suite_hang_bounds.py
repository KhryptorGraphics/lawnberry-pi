"""A hung test fails and says where; it must not hold the run (audit review 3, TOOL3-01).

``scripts/run_tests_shuffled.py --seed 202 --mode item`` once sat for 11 minutes at test 3470 of
4248, in the teardown of an async generator fixture (``Platform.close()`` of
``test_ibt4_stand_down_integration.py``), and nothing in the repository bounded it: no per-test
timeout, no job timeout. CI would have waited for GitHub's six-hour limit and given no verdict.
Three bounds now exist; each is checked here against the hang it is meant to cut:

* the suite's own ``timeout`` (pytest-timeout, ``pyproject.toml``) fails the stalled test with the
  stacks of every thread, and the run goes on;
* a test that is *meant* to run for hours (the opt-in 8 h soak) is exempt from it;
* the teardown that stalled reports the step it is stuck in instead of waiting for it.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_ibt4_stand_down_integration as stand_down

REPO = Path(__file__).resolve().parents[2]


def _pytest(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """A real pytest process under the repository's own ``pyproject.toml`` configuration."""
    environment = {**os.environ, "SIM_MODE": "1", **env}
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_TIMEOUT", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(REPO / "pyproject.toml"), "-o", "addopts="]
        + ["--rootdir", str(tmp_path), "-p", "no:cacheprovider", "-W", "ignore", "-q"]
        + ["--tb=short", *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


HANGING_TEARDOWN = textwrap.dedent(
    """
    import asyncio
    import pytest_asyncio

    @pytest_asyncio.fixture
    async def platform():
        yield "ready"
        await asyncio.sleep(3600)  # a teardown that never returns, like the one that stalled

    async def test_a_uses_the_fixture(platform):
        assert platform == "ready"

    async def test_b_runs_after_the_hang():
        assert True
    """
)


def test_a_hang_in_an_async_fixture_teardown_fails_that_test_and_the_run_goes_on(tmp_path):
    (tmp_path / "test_probe.py").write_text(HANGING_TEARDOWN)

    done = _pytest(tmp_path, "--timeout=3", str(tmp_path / "test_probe.py"))

    out = done.stdout + done.stderr
    assert "unrecognized arguments: --timeout" not in out, (
        "pytest-timeout is not installed in this environment: install the project's "
        "dependencies (`uv sync --frozen --all-extras --dev`)"
    )
    assert done.returncode == 1, out
    assert "Timeout (>3.0s) from pytest-timeout" in out
    assert "1 error" in out and "2 passed" in out, "the test after the hang must still run"


def test_the_shipped_bound_is_a_real_one(tmp_path):
    """The ``timeout`` of ``pyproject.toml``, as pytest itself resolves it, is finite and short."""
    (tmp_path / "test_probe.py").write_text(
        "def test_probe(request):\n    print('BOUND=' + str(request.config.getini('timeout')))\n"
    )

    done = _pytest(tmp_path, "-s", str(tmp_path / "test_probe.py"))

    assert done.returncode == 0, done.stdout + done.stderr
    raw = re.search(r"BOUND=(\S*)", done.stdout).group(1)
    bound = float(raw) if raw else 0.0  # nothing configured: pytest-timeout's own default is off
    # 0 disables it; the slowest test is 14 s, so anything past 15 minutes is no bound at all
    assert 0 < bound <= 900, f"timeout = {raw!r}: the suite has no per-test bound"


def test_the_opt_in_8_hour_soak_is_not_cut_off_by_the_suite_bound(tmp_path):
    probe = tmp_path / "probe_markers.py"
    probe.write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        marker = item.get_closest_marker('timeout')\n"
        "        print('TIMEOUT_MARKER', item.name, marker.args if marker else None)\n"
    )

    done = _pytest(
        tmp_path,
        "--collect-only",
        "-s",
        "-p",
        "probe_markers",
        str(REPO / "tests" / "soak" / "test_8hour_operation.py"),
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(REPO)]),
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "TIMEOUT_MARKER test_long_soak_8h (0,)" in done.stdout


# ---- the teardown that stalled ---------------------------------------------------------------


def _stalled_platform(*, mission, calls):
    """A ``Platform`` with only what ``close()`` touches: a mission task and three services."""

    async def record(name):
        calls.append(name)

    platform = object.__new__(stand_down.Platform)
    platform.task = mission
    platform.svc = SimpleNamespace(shutdown=lambda: record("robohat shutdown"))
    platform.tractor = SimpleNamespace(shutdown=lambda: record("tractor shutdown"))
    platform.blade = SimpleNamespace(park=lambda: record("blade park"))
    platform.nav = SimpleNamespace(_mission_loop_active=False)
    return platform


async def _mission_that_ignores_cancellation(seconds: float) -> None:
    """``execute_mission`` that swallows the cancel and carries on, for a while."""
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while loop.time() < end:
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass


async def test_a_mission_that_ignores_cancel_is_reported_and_the_rest_of_the_teardown_still_runs(
    monkeypatch,
):
    monkeypatch.setattr(stand_down, "STEP_LIMIT_S", 0.3)
    calls: list[str] = []
    mission = asyncio.create_task(_mission_that_ignores_cancellation(1.5))
    await asyncio.sleep(0)  # let it start, so the cancel reaches the loop it is in
    platform = _stalled_platform(mission=mission, calls=calls)

    closing = asyncio.ensure_future(platform.close())
    finished, _ = await asyncio.wait({closing}, timeout=1.2)  # well inside the mission's 1.5 s

    assert finished, "close() waited for the mission instead of reporting it: the old hang"
    with pytest.raises(AssertionError, match=r"the mission task did not finish within 0\.3 s"):
        closing.result()
    # the blade must still be parked and the services shut down: one stuck step is not a reason
    # to leave the IBT-4 pads energised for the next test
    assert calls == ["robohat shutdown", "tractor shutdown", "blade park"]
    await asyncio.wait({mission}, timeout=5)  # the stubborn task ends by itself: nothing leaks


async def test_a_step_that_never_returns_names_itself_and_where_it_is_stuck(monkeypatch):
    monkeypatch.setattr(stand_down, "STEP_LIMIT_S", 0.2)
    never = asyncio.Event()

    async def waits_for_a_latch_nobody_clears():
        await never.wait()

    platform = _stalled_platform(mission=None, calls=[])
    platform.tractor = SimpleNamespace(shutdown=waits_for_a_latch_nobody_clears)

    closing = asyncio.ensure_future(platform.close())
    finished, _ = await asyncio.wait({closing}, timeout=2)

    assert finished, "close() hung on a step that never returns"
    with pytest.raises(AssertionError) as caught:
        closing.result()
    message = str(caught.value)
    assert "the tractor shutdown did not finish within 0.2 s" in message
    assert "waits_for_a_latch_nobody_clears" in message, "the stack says where it is stuck"


async def test_a_clean_teardown_raises_nothing_and_runs_every_step_in_order():
    calls: list[str] = []

    async def finishes_quickly():
        await asyncio.sleep(0)

    mission = asyncio.create_task(finishes_quickly())
    platform = _stalled_platform(mission=mission, calls=calls)

    await platform.close()

    assert calls == ["robohat shutdown", "tractor shutdown", "blade park"]
