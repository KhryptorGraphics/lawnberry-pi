"""``tests/conftest.py`` hands every test a clean auth facade and E-stop window.

Two process-wide singletons used to make tests depend on their predecessors (audit review of PR
#20): the failed-login counter of ``auth_service`` (five failures lock ``admin`` out, so
``test_password_authentication_*`` failed after the tests that fail logins) and the 0.2 s / 0.3 s
drive-block window an E-stop arms in ``rest`` (a legacy ``/control/drive`` test answered 403 after
an E-stop test). ``scripts/run_tests_shuffled.py`` found both; these tests keep the fix honest.
"""

from __future__ import annotations

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
