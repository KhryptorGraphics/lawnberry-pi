"""``tests/conftest.py`` hands every test a clean auth facade and E-stop window.

Two process-wide singletons used to make tests depend on their predecessors (audit review of PR
#20): the failed-login counter of ``auth_service`` (five failures lock ``admin`` out, so
``test_password_authentication_*`` failed after the tests that fail logins) and the 0.2 s / 0.3 s
drive-block window an E-stop arms in ``rest`` (a legacy ``/control/drive`` test answered 403 after
an E-stop test). ``scripts/run_tests_shuffled.py`` found both; these tests keep the fix honest.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The facade's attributes, as the reset in tests/conftest.py knows them. A new one that is not in
# this tuple is state the reset does not clear: add it there and here.
FACADE_ATTRIBUTES = (
    "_core",
    "config",
    "active_sessions",
    "_initialized",
    "_failed_attempts",
    "_invalidated_session_ids",
)


def _dirty_the_shared_state():
    from backend.src.api import rest
    from backend.src.services.auth_service import auth_service

    auth_service._failed_attempts["admin"] = 99
    auth_service.active_sessions["admin"] = [object()]
    auth_service._invalidated_session_ids.add("session-1")
    auth_service._initialized = True
    auth_service.config.session_timeout_minutes = 1
    rest._emergency_until = time.time() + 600
    rest._client_emergency["client-1"] = time.time() + 600
    return rest, auth_service


def test_the_reset_clears_the_failed_login_counter_and_the_rest_of_the_facade(reset_shared_state):
    from backend.src.models.auth_security_config import AuthSecurityConfig

    _, auth_service = _dirty_the_shared_state()

    reset_shared_state()

    assert auth_service._failed_attempts == {}
    assert auth_service.active_sessions == {}
    assert auth_service._invalidated_session_ids == set()
    assert auth_service._initialized is False
    assert auth_service.config == AuthSecurityConfig()


def test_the_reset_ends_the_estop_drive_block_window(reset_shared_state):
    rest, _ = _dirty_the_shared_state()
    assert rest._emergency_active()

    reset_shared_state()

    assert rest._emergency_until == 0.0
    assert rest._client_emergency == {}
    assert not rest._emergency_active()


async def test_a_locked_out_user_is_let_in_again_after_the_reset(reset_shared_state):
    # The symptom: five failures locked "admin" out for every later test of the process.
    from backend.src.services.auth_service import AuthenticationError, auth_service

    auth_service._failed_attempts["admin"] = 5
    with pytest.raises(AuthenticationError, match="Too many attempts"):
        await auth_service.authenticate_password("admin", "x")

    reset_shared_state()

    with pytest.raises(AuthenticationError, match="Invalid credentials"):  # no longer locked out
        await auth_service.authenticate_password("admin", "x")


def test_the_reset_clears_the_dict_the_rest_module_and_globals_share(reset_shared_state):
    # rest imported `_client_emergency` by name: clearing a copy would leave rest's own entry.
    from backend.src.api import rest
    from backend.src.core import globals as shared

    rest._client_emergency["c"] = time.time() + 600
    assert shared._client_emergency is rest._client_emergency

    reset_shared_state()

    assert shared._client_emergency == {}


def test_the_facade_has_no_state_the_reset_does_not_know():
    from backend.src.services.auth_service import auth_service

    assert set(vars(auth_service)) == set(FACADE_ATTRIBUTES), (
        "auth_service has state tests/conftest.py::reset_shared_backend_state does not reset: "
        "reset it there and list it in FACADE_ATTRIBUTES"
    )


# A throw-away test module run by a real pytest process with tests/conftest.py loaded as a
# plugin, so what is checked is the autouse fixture and not just the function it calls.
PROBE = textwrap.dedent(
    """
    import sys, time

    def test_0_a_test_that_never_touches_the_backend_does_not_get_it_imported():
        assert "backend.src.api.rest" not in sys.modules
        assert "backend.src.services.auth_service" not in sys.modules

    def test_1_leaves_the_shared_state_dirty():
        from backend.src.api import rest
        from backend.src.services.auth_service import auth_service

        auth_service._failed_attempts["admin"] = 99
        auth_service.active_sessions["admin"] = [object()]
        rest._emergency_until = time.time() + 600
        rest._client_emergency["client-1"] = time.time() + 600
        assert rest._emergency_active()

    def test_2_starts_clean_whatever_the_test_before_it_did():
        from backend.src.api import rest
        from backend.src.services.auth_service import auth_service

        assert auth_service._failed_attempts == {}
        assert auth_service.active_sessions == {}
        assert rest._emergency_until == 0.0
        assert rest._client_emergency == {}
        assert not rest._emergency_active()
    """
)


def test_every_test_starts_clean_because_the_autouse_fixture_runs(tmp_path):
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "tests"), str(REPO)]),
        "SIM_MODE": "1",
    }
    env.pop("PYTEST_ADDOPTS", None)

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider"]
        + ["-p", "no:randomly", "-o", "addopts=", "-W", "ignore", "-q", "--tb=short", str(probe)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    assert "3 passed" in done.stdout


@pytest.mark.asyncio
async def test_the_reset_empties_the_limiter_buckets_a_route_s_own_tests_drained(
    reset_shared_state,
):
    """``/control/blade`` is limited to 8 req/s, burst 20, and every test client is one peer.

    Hammered anonymously it answers 429; after the reset the next request is counted from a full
    bucket again. Without the reset in ``tests/conftest.py`` a route's own tests drain the bucket
    for whichever test runs next (the blade status test failed with a 429 in the full suite).
    """
    import httpx

    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        codes = [(await client.get("/api/v2/control/blade/status")).status_code for _ in range(60)]
        assert 429 in codes, "the premise: the strict /control/blade bucket drains"

        reset_shared_state()

        assert (await client.get("/api/v2/control/blade/status")).status_code != 429


# ---- the other autouse fixtures: each one has a test that fails without it (audit TOOL3-12) ------

FIXTURE_PROBE = textwrap.dedent(
    """
    import os
    from datetime import UTC, datetime

    def test_0_the_latch_file_is_this_tests_own(tmp_path):
        assert os.environ["LAWNBERRY_LATCH_FILE"] == str(tmp_path / "emergency_latch.json")

    def test_1_the_credential_is_the_suites_never_a_developers():
        assert os.environ["LAWN_BERRY_OPERATOR_CREDENTIAL"] == "test-operator-credential"

    def test_2_leaves_process_wide_state_set():
        from backend.src.core import globals as shared
        from backend.src.services import blade_service
        from backend.src.services.navigation_service import NavigationService

        nav = NavigationService.get_instance()
        nav.navigation_state.heading_updated_at = datetime.now(UTC)
        nav._last_imu_stamp = datetime.now(UTC)
        shared._safety_state["emergency_stop_active"] = True
        shared._client_emergency["client-1"] = 1e12
        blade_service._blade_service = object()

    def test_3_starts_with_all_of_it_reset():
        from backend.src.core import globals as shared
        from backend.src.services import blade_service
        from backend.src.services.navigation_service import NavigationService

        nav = NavigationService.get_instance()
        assert nav.navigation_state.heading_updated_at is None
        assert nav._last_imu_stamp is None
        assert shared._safety_state["emergency_stop_active"] is False
        assert shared._client_emergency == {}
        assert blade_service._blade_service is None
    """
)


def test_the_latch_file_the_heading_stamp_the_safety_latches_and_the_blade_service_are_reset(
    tmp_path,
):
    probe = tmp_path / "test_fixture_probe.py"
    probe.write_text(FIXTURE_PROBE)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "tests"), str(REPO)]),
        "SIM_MODE": "1",
        "LAWN_BERRY_OPERATOR_CREDENTIAL": "a-developers-real-credential",  # must not leak in
    }
    env.pop("PYTEST_ADDOPTS", None)

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider", "-W", "ignore"]
        + ["-p", "no:randomly", "-o", "addopts=", "-q", "--tb=short", str(probe)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-1000:]
    assert "4 passed" in done.stdout


def _root_conftest():
    spec = importlib.util.spec_from_file_location(
        "lawnberry_root_conftest", REPO / "tests" / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid, self.markers = nodeid, []

    def add_marker(self, marker) -> None:
        self.markers.append(marker)


def test_the_tests_that_need_the_anonymous_websocket_are_skipped_only_without_the_opt_in(
    monkeypatch,
):
    conftest = _root_conftest()
    listed = sorted(conftest.NEEDS_SIM_OPEN_AUTH)
    bystander = "tests/unit/test_something_else.py::test_x"

    leaky = sorted(conftest.ORDER_DEPENDENT_IN_STRICT)
    for value, marked in (("0", True), ("false", True), ("", True), ("1", False), ("true", False)):
        monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", value)
        items = [_Item(n) for n in [*listed, *leaky, bystander]]
        conftest.pytest_collection_modifyitems(None, items)
        assert [bool(i.markers) for i in items[:-1]] == [marked] * (len(listed) + len(leaky))
        assert items[-1].markers == [], "only the listed tests are marked"
        if marked:
            assert {m.name for i in items[: len(listed)] for m in i.markers} == {"skip"}
            assert {m.name for i in items[len(listed) : -1] for m in i.markers} == {"xfail"}
    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH")
    items = [_Item(listed[0])]
    conftest.pytest_collection_modifyitems(None, items)
    assert items[0].markers == [], "unset means the suite's default: on"


def test_every_test_the_strict_run_skips_still_exists():
    conftest = _root_conftest()
    stale = []
    for nodeid in sorted(conftest.NEEDS_SIM_OPEN_AUTH | conftest.ORDER_DEPENDENT_IN_STRICT):
        path, _, name = nodeid.partition("::")
        tree = ast.parse((REPO / path).read_text())
        if name not in {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }:
            stale.append(nodeid)
    assert stale == [], "remove them from the strict-mode lists in tests/conftest.py"
