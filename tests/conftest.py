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
    from backend.src.services.navigation_service import NavigationService

    nav = NavigationService._instance
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
    from backend.src.core import globals as core_globals
    from backend.src.services import blade_service

    core_globals._safety_state["emergency_stop_active"] = False
    core_globals._client_emergency.clear()
    blade_service._blade_service = None
    yield
