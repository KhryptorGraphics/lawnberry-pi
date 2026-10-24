"""Fixtures for the contract tests that drive process-wide navigation / sensor state.

The navigation debug API, the safety-trigger manager, the robot-state manager and the
WebSocket hub's sensor manager are module-level singletons. Every fixture here swaps the
singleton for a fresh one (or snapshots and restores it) so a test cannot see another
test's position, geofence, interlocks or sensors, and cannot leave its own behind.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio


@pytest.fixture
def sim_env(monkeypatch):
    """SIM_MODE=1 for the test, with no fault injection leaking in from the environment."""
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.delenv("FAULT_INJECT", raising=False)


@pytest.fixture
def nav_state(monkeypatch, sim_env):
    """Fresh RobotState, safety-trigger manager and nav debug globals; restored afterwards.

    Yields the fresh ``RobotState`` so a test can inspect it without going through HTTP.
    """
    from backend.src.api import navigation as nav_api
    from backend.src.core import globals as app_globals
    from backend.src.core import robot_state_manager as state_mgr
    from backend.src.safety import safety_triggers as triggers

    manager = state_mgr.RobotStateManager()
    monkeypatch.setattr(state_mgr, "_manager", manager)
    monkeypatch.setattr(triggers, "_manager", triggers.SafetyTriggerManager())
    # A running app lifespan would route interlock events to the websocket hub.
    monkeypatch.setattr(triggers, "_event_handler", None)
    monkeypatch.setattr(nav_api, "_debug_position", None)
    monkeypatch.setattr(nav_api, "_active_geofence", None)
    monkeypatch.setattr(nav_api, "_waypoints", [])
    monkeypatch.setattr(nav_api, "_waypoint_reached", None)
    monkeypatch.setattr(nav_api, "_current_waypoint_id", None)

    # The routers hold a reference to this dict, so it is cleared and restored in place.
    saved_overrides = dict(app_globals._debug_overrides)
    app_globals._debug_overrides.clear()
    try:
        yield manager.get_state()
    finally:
        app_globals._debug_overrides.clear()
        app_globals._debug_overrides.update(saved_overrides)


@pytest_asyncio.fixture
async def api_client(sim_env):
    """httpx client bound straight to the ASGI app (no network, no lifespan)."""
    from backend.src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def install_sensor_manager(monkeypatch, sim_env):
    """Build a SIM ``SensorManager`` and make ``/api/v2/sensors/*`` read it.

    ``await install_sensor_manager(**kwargs)`` forwards ``kwargs`` to ``SensorManager``
    (the ultrasonic array stays disabled: it is not under test and would otherwise
    depend on the safety-limits file). The hub's previous manager is restored afterwards.
    """
    from backend.src.services.sensor_manager import SensorManager
    from backend.src.services.websocket_hub import websocket_hub

    async def _install(**kwargs):
        kwargs.setdefault("ultrasonic_enabled", False)
        manager = SensorManager(**kwargs)
        await manager.initialize()
        monkeypatch.setattr(websocket_hub, "_sensor_manager", manager)
        return manager

    return _install


@pytest.fixture
def i2c_bus_silent(monkeypatch):
    """No I2C device ACKs (unpowered or miswired bus); serial devices like the GPS still work.

    Patches the presence probe ``SensorManager`` uses at initialisation, so the drivers stay in
    SIM_MODE and nothing touches a real bus.
    """
    from backend.src.services import sensor_manager

    monkeypatch.setattr(sensor_manager, "_i2c_ack", lambda bus, addresses: False)
