"""Global test configuration for LawnBerry Pi v2."""

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

collect_ignore = [
    "unit/test_health_endpoints.py",
    # Avoid module name collision with integration/test_auth_security_levels.py
    "unit/test_auth_security_levels.py",
]


def pytest_ignore_collect(collection_path, config):  # pragma: no cover - hook signature
    try:
        str(collection_path)
        return collection_path.match(
            "tests/unit/test_health_endpoints.py"
        ) or collection_path.match("tests/unit/test_auth_security_levels.py")
    except AttributeError:
        return False


# Ensure repository root on sys.path for 'backend' imports and compat stubs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make optional dependency stubs importable in all tests (bcrypt, pyotp, google, psutil)
compat_stubs = ROOT / "backend" / "src" / "compat_stubs"


def _ensure_optional_dependency(module_name: str) -> None:
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ImportError:
        if compat_stubs.exists():
            stub_path = str(compat_stubs)
            if stub_path not in sys.path:
                sys.path.insert(0, stub_path)
            importlib.import_module(module_name)


for optional_module in ("bcrypt", "pyotp", "google", "jwt", "psutil", "timezonefinder"):
    try:
        _ensure_optional_dependency(optional_module)
    except ImportError:
        # Allow missing optional dependencies when no stub is present.
        pass

# Ensure SIM_MODE=1 for all tests unless explicitly overridden
os.environ.setdefault("SIM_MODE", "1")
# Operator auth on write endpoints is secure-by-default in production; disable it
# for the suite so existing contract tests can call writes without a token.
os.environ.setdefault("OPERATOR_AUTH_REQUIRED", "0")
# SIM_MODE alone no longer makes the login accept any non-empty credential or admits any
# WebSocket client as an operator (SEC-10, backend/src/core/auth_posture.py): that needs this
# explicit opt-in as well. Tests that log in use the `operator_credential` fixture, never a
# made-up value, so the strict path is exercised for real: the whole suite also passes with the
# opt-in turned OFF (`LAWN_BERRY_SIM_OPEN_AUTH=0 pytest ...`; it did not until TOOL-13 fixed the
# four tests that logged in with "secret", "ok" and "operator"). setdefault, so a run can
# override it.
os.environ.setdefault("LAWN_BERRY_SIM_OPEN_AUTH", "1")
# There is no built-in operator credential (backend/src/core/credential_policy.py). The suite
# configures an explicit one before the backend is imported; tests that log in take it from
# the `operator_credential` fixture instead of hard-coding a value. Forced, not setdefault:
# a developer's real credential must never leak into a test run.
TEST_OPERATOR_CREDENTIAL = "test-operator-credential"
os.environ["LAWN_BERRY_OPERATOR_CREDENTIAL"] = TEST_OPERATOR_CREDENTIAL
os.environ.setdefault("GLOBAL_RATE_LIMIT_RATE", "1000")
os.environ.setdefault("GLOBAL_RATE_LIMIT_BURST", "10000")
os.environ.setdefault("AUTH_RATE_LIMIT_WINDOW", "60")
os.environ.setdefault("AUTH_RATE_LIMIT_MAX_ATTEMPTS", "1000")
os.environ.setdefault("AUTH_LOCKOUT_FAILURES", "1000")
os.environ.setdefault("AUTH_LOCKOUT_SECONDS", "0")


# ---- process-wide state that one test must not leave for the next ----------------------------
# Two singletons keep state that decides a later test's outcome, so the suite passed only in
# its default order (audit review of PR #20, "order-dependent tests"; found with
# scripts/run_tests_shuffled.py, which is how to check this block still holds):
#
# * `auth_service` (backend/src/services/auth_service.py, the facade the auth-level unit tests
#   use) counts failed logins per user in `_failed_attempts`; five failures lock the user out
#   ("Too many attempts"), so test_password_authentication_* failed whenever the tests that fail
#   logins ran before them.
# * `/control/emergency` arms a 0.2 s drive-block window (`rest._emergency_until`) and a 0.3 s
#   per-client one (`_client_emergency`); a legacy `/control/drive` test that started inside it
#   got "403 Emergency stop active" after any E-stop test.
#
# Only modules the test run has already imported are touched: a reset must not import the
# backend for a test that never uses it. tests/unit/test_conftest_isolation.py proves the reset
# and fails when the facade grows state this does not reset.
def reset_shared_backend_state() -> None:
    """Put the auth facade's counters and the E-stop drive-block window back to their start."""
    auth = sys.modules.get("backend.src.services.auth_service")
    if auth is not None:
        facade = auth.auth_service
        facade._failed_attempts.clear()
        facade.active_sessions.clear()
        facade._invalidated_session_ids.clear()
        facade._initialized = False
        facade.config = auth.AuthSecurityConfig()
    rest = sys.modules.get("backend.src.api.rest")
    if rest is not None:
        rest._emergency_until = 0.0
        rest._client_emergency.clear()
    _clear_rate_limit_buckets()


def _clear_rate_limit_buckets() -> None:
    """Empty the global limiter's per-client buckets.

    The per-prefix overrides are deliberately strict (``/control/blade`` is 8 req/s, burst 20) and
    every test client is the same anonymous peer, so without this a route's own tests drain the
    bucket for whichever test runs next and get a 429 that has nothing to do with what it tests.
    """
    main = sys.modules.get("backend.src.main")
    layer = getattr(getattr(main, "app", None), "middleware_stack", None)
    for _ in range(64):  # the middleware stack is a linked chain: walk it, bounded
        if layer is None:
            return
        buckets = getattr(layer, "_buckets", None)
        if type(layer).__name__ == "GlobalRateLimiter" and isinstance(buckets, dict):
            buckets.clear()
        layer = getattr(layer, "app", None)


@pytest.fixture(autouse=True)
def _isolate_shared_backend_state():
    reset_shared_backend_state()


@pytest.fixture
def reset_shared_state():
    """The reset the autouse fixture above runs, for a test that wants to call it itself."""
    return reset_shared_backend_state


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables and configurations."""
    # Ensure simulation mode for tests
    os.environ["SIM_MODE"] = "1"
    # Set test database path if needed
    os.environ.setdefault("DB_PATH", ":memory:")
    # Reduce log noise in tests
    os.environ.setdefault("LOG_LEVEL", "WARNING")

    # Make rate limiters permissive for testing
    os.environ["GLOBAL_RATE_LIMIT_RATE"] = "1000"
    os.environ["GLOBAL_RATE_LIMIT_BURST"] = "10000"
    os.environ["AUTH_RATE_LIMIT_WINDOW"] = "60"
    os.environ["AUTH_RATE_LIMIT_MAX_ATTEMPTS"] = "1000"
    os.environ["AUTH_LOCKOUT_FAILURES"] = "1000"
    os.environ["AUTH_LOCKOUT_SECONDS"] = "0"

    print("CONFTEST: Set rate limiting environment variables")

    yield

    # Cleanup if necessary
    pass


@pytest_asyncio.fixture
async def test_client():
    """Create a test client for the FastAPI application."""
    import httpx

    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def operator_credential() -> str:
    """The explicit operator credential the suite configures (there is no built-in one)."""
    return TEST_OPERATOR_CREDENTIAL


@pytest.fixture(autouse=True)
def _isolated_latch_file(tmp_path, monkeypatch):
    """Every test gets its own durable-latch file (audit SAF-16).

    An E-stop persists its latch to disk and the next start-up restores it: without this a
    stop in one test would be restored by the next test's lifespan (or the next session's).
    """
    monkeypatch.setenv("LAWNBERRY_LATCH_FILE", str(tmp_path / "emergency_latch.json"))


@pytest.fixture(autouse=True)
def _stale_heading_does_not_leak_between_tests():
    """The process-wide NavigationService must not carry one test's IMU stamp into the next.

    The heading gate (audit SAF-14) holds autonomous motion once the heading is older than a
    couple of seconds, which is the point of it. A test that fed one IMU reading must not leave
    the singleton to age into a hold for every test that follows.
    """
    # Lazy on purpose: a test that never touches the backend must not get it imported by a
    # fixture (see tests/unit/test_conftest_isolation.py), so reset only what is already loaded.
    module = sys.modules.get("backend.src.services.navigation_service")
    nav = getattr(getattr(module, "NavigationService", None), "_instance", None)
    if nav is not None:
        nav.navigation_state.heading_updated_at = None
        nav._last_imu_stamp = None
    yield


@pytest.fixture(autouse=True)
def _process_latches_do_not_leak_between_tests():
    """A latch one test set in the process-wide state must not be in force for the next.

    The status routes report the real safety state (audit SAF-13), so an E-stop or an IBT-4
    blade latch an earlier test left set (the E-stop routes) would now show up in every later
    status read; the safety state and the blade service are rebuilt for each test instead.
    """
    # Lazy for the same reason as above: only modules a test has already imported can hold state.
    core_globals = sys.modules.get("backend.src.core.globals")
    if core_globals is not None:
        core_globals._safety_state["emergency_stop_active"] = False
        core_globals._client_emergency.clear()
    blade = sys.modules.get("backend.src.services.blade_service")
    if blade is not None:
        blade._blade_service = None
    yield
