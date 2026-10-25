"""Status surfaces report the real safety state (audit SAF-13).

``/api/v2/status``, ``/api/v2/ws/status``, ``/api/v2/dashboard/status``,
``/api/v2/dashboard/metrics`` and ``/api/v1/status`` used to answer with literals
(``emergency_stop_active: false``, ``blade_safety_ok: true``, no interlocks) while the machine
was latched. They now share ``core.safety_status.collect_safety_status``; these tests latch the
real services/latches and read the answers through the real app.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.src.core import globals as core_globals
from backend.src.core.robot_state_manager import get_robot_state_manager
from backend.src.main import app
from backend.src.models.safety_limits import ObstacleStopLimits
from backend.src.nav.gps_degradation import GPSDegradationConfig, GPSDegradationMonitor
from backend.src.safety.obstacle_stop import ObstacleGuard
from backend.src.safety.safety_triggers import get_safety_trigger_manager
from backend.src.services import blade_service as blade_mod
from backend.src.services import robohat_service as robohat_mod
from backend.src.services import tractor_service as tractor_mod
from backend.src.services.blade_service import BladeService
from backend.src.services.navigation_service import NavigationService
from backend.src.services.robohat_service import RoboHATService
from tests.unit._tractor_fakes import idle_tractor, mowing_tractor

STATUS_ROUTES = ("/api/v2/status", "/api/v2/dashboard/status", "/api/v1/status")


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    """Every source at rest, restored afterwards; each test latches what it needs."""
    monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", False)
    monkeypatch.setattr(tractor_mod, "_tractor_service", idle_tractor())
    monkeypatch.setattr(robohat_mod, "robohat_service", None)
    monkeypatch.setattr(blade_mod, "_blade_service", BladeService({}))
    nav = NavigationService.get_instance()
    nav.attach_safety()
    # Interlocks other tests left in the process-wide trigger manager / robot state (a tilt, an
    # obstacle) are not this machine's: the "nothing in force" claims need a quiet start.
    monkeypatch.setattr(get_safety_trigger_manager(), "_active", {})
    monkeypatch.setattr(get_robot_state_manager()._state, "active_interlocks", [])
    yield
    nav.attach_safety()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def safety(client, route: str) -> dict:
    resp = await client.get(route)
    assert resp.status_code == 200, resp.text
    return resp.json()["safety_status"]


async def all_safety(client) -> dict[str, dict]:
    return {route: await safety(client, route) for route in STATUS_ROUTES}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def guard_on_clock() -> tuple[ObstacleGuard, Clock]:
    clock = Clock()
    guard = ObstacleGuard(ObstacleStopLimits(), clock=clock)
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    return guard, clock


def make_guard_clear(guard: ObstacleGuard, clock: Clock) -> None:
    for _ in range(40):
        clock.t += 0.1
        guard.observe_range("tof_left", 2.0, clock.t)
        guard.observe_range("tof_right", 2.0, clock.t)
        guard.assess()


# --------------------------------------------------------------------------- (a) process latch


async def test_a_quiet_machine_reports_nothing_in_force(client):
    for route, block in (await all_safety(client)).items():
        assert block["emergency_stop_active"] is False, route
        assert block["blade_safety_ok"] is True, route
        assert block["safety_interlocks"] == [], route


async def test_the_estop_route_is_visible_on_every_status_surface_until_cleared(client):
    resp = await client.post("/api/v2/control/emergency-stop")
    assert resp.status_code == 200
    try:
        for route, block in (await all_safety(client)).items():
            assert block["emergency_stop_active"] is True, route
            assert block["blade_safety_ok"] is False, route
            assert "emergency_stop" in block["safety_interlocks"], route
    finally:
        core_globals._safety_state["emergency_stop_active"] = False
    # the E-stop also latched the blade service: reported truthfully until that is cleared too
    assert (await safety(client, "/api/v2/status"))["emergency_stop_active"] is True
    await blade_mod.get_blade_service().clear_emergency()
    for route, block in (await all_safety(client)).items():
        assert block["emergency_stop_active"] is False, route
        assert block["blade_safety_ok"] is True, route
        assert block["safety_interlocks"] == [], route


# --------------------------------------------------------------------------- (b) services


async def test_a_tractor_estop_latch_alone_is_reported(client, monkeypatch):
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr(tractor_mod, "_tractor_service", tractor)
    await tractor.emergency_stop(source="test")
    assert core_globals._safety_state["emergency_stop_active"] is False
    for route, block in (await all_safety(client)).items():
        assert block["emergency_stop_active"] is True, route
        assert block["blade_safety_ok"] is False, route
        assert "tractor_emergency_stop" in block["safety_interlocks"], route


async def test_a_tilt_latch_on_the_tractor_is_a_tilt_and_inhibits_the_blade(client, monkeypatch):
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr(tractor_mod, "_tractor_service", tractor)
    await tractor.latch_interlock("tilt")
    for route, block in (await all_safety(client)).items():
        assert block["tilt_detected"] is True, route
        assert block["blade_safety_ok"] is False, route
        assert "blade_interlock:tilt" in block["safety_interlocks"], route


async def test_a_disabled_tractor_latch_flags_are_not_meaningful(client, monkeypatch):
    tractor = idle_tractor()
    tractor.state.interlock_latched = True
    tractor.state.latch_reason = "tilt"
    monkeypatch.setattr(tractor_mod, "_tractor_service", tractor)
    block = await safety(client, "/api/v2/status")
    assert block["tilt_detected"] is False
    assert block["safety_interlocks"] == []


async def test_a_locked_out_tractor_is_reported(client, monkeypatch):
    tractor, _chip = mowing_tractor()
    tractor.lock_out("pca9685 write failed")
    monkeypatch.setattr(tractor_mod, "_tractor_service", tractor)
    for route, block in (await all_safety(client)).items():
        assert block["emergency_stop_active"] is True, route
        assert block["blade_safety_ok"] is False, route
        assert "tractor_lockout" in block["safety_interlocks"], route


async def test_a_parked_tractor_inhibits_the_blade_without_claiming_an_estop(client, monkeypatch):
    tractor, _chip = mowing_tractor()
    await tractor.park()
    monkeypatch.setattr(tractor_mod, "_tractor_service", tractor)
    block = await safety(client, "/api/v2/status")
    assert "tractor_parked" in block["safety_interlocks"]
    assert block["blade_safety_ok"] is False


async def test_a_robohat_estop_hold_is_reported(client, monkeypatch):
    hat = RoboHATService()
    hat._estop_hold = True
    monkeypatch.setattr(robohat_mod, "robohat_service", hat)
    for route, block in (await all_safety(client)).items():
        assert block["emergency_stop_active"] is True, route
        assert block["blade_safety_ok"] is False, route
        assert "robohat_emergency_hold" in block["safety_interlocks"], route


async def test_a_blade_service_latch_is_reported(client):
    await blade_mod.get_blade_service()._driver.set_estop(True)
    for route, block in (await all_safety(client)).items():
        assert block["emergency_stop_active"] is True, route
        assert block["blade_safety_ok"] is False, route
        assert "blade_service_latched" in block["safety_interlocks"], route


# --------------------------------------------------------------------------- (c) motion gate


async def test_an_obstacle_guard_hold_is_published_and_a_clear_guard_is_not(client):
    nav = NavigationService.get_instance()
    guard, clock = guard_on_clock()
    nav.attach_safety(obstacle_guard=guard)  # no readings yet: the guard holds
    for route, block in (await all_safety(client)).items():
        assert "obstacle_hold" in block["safety_interlocks"], route
        assert block["blade_safety_ok"] is False, route
    make_guard_clear(guard, clock)
    assert nav.motion_gate().hold is False
    for route, block in (await all_safety(client)).items():
        assert block["safety_interlocks"] == [], route
        assert block["blade_safety_ok"] is True, route
        assert block["obstacle_detected"] is False, route


async def test_a_sighted_obstacle_sets_obstacle_detected(client):
    guard, clock = guard_on_clock()
    make_guard_clear(guard, clock)
    clock.t += 0.1
    guard.observe_range("tof_left", 0.1, clock.t)
    guard.observe_range("tof_right", 2.0, clock.t)
    NavigationService.get_instance().attach_safety(obstacle_guard=guard)
    block = await safety(client, "/api/v2/status")
    assert block["obstacle_detected"] is True
    assert "obstacle_hold" in block["safety_interlocks"]
    assert block["blade_safety_ok"] is False


async def test_a_degraded_gnss_fix_is_published_as_a_hold(client):
    clock = Clock()
    monitor = GPSDegradationMonitor(
        GPSDegradationConfig(max_accuracy_m=1.0, max_fix_age_s=2.0, resume_dwell_s=3.0),
        clock=clock,
    )  # a monitor that has seen no fix starts degraded
    NavigationService.get_instance().attach_safety(gps_monitor=monitor)
    for route, block in (await all_safety(client)).items():
        assert "gnss_hold" in block["safety_interlocks"], route
        assert "obstacle_hold" not in block["safety_interlocks"], route
        assert block["blade_safety_ok"] is False, route
    NavigationService.get_instance().attach_safety(
        gps_monitor=SimpleNamespace(assess=lambda: SimpleNamespace(healthy=True, reasons=()))
    )
    block = await safety(client, "/api/v2/status")
    assert block["safety_interlocks"] == []
    assert block["blade_safety_ok"] is True


# --------------------------------------------------------------------------- (d) fail closed


class _Unreadable:
    @property
    def emergency_latched(self):
        raise RuntimeError("bus gone")


async def test_an_unreadable_source_is_an_interlock_and_the_route_still_answers(
    client, monkeypatch
):
    monkeypatch.setattr(blade_mod, "_blade_service", _Unreadable())
    for route, block in (await all_safety(client)).items():
        assert "state_unreadable:blade_service" in block["safety_interlocks"], route
        assert block["blade_safety_ok"] is False, route


async def test_one_unreadable_source_does_not_hide_a_latch_in_another(client, monkeypatch):
    monkeypatch.setattr(blade_mod, "_blade_service", _Unreadable())
    core_globals._safety_state["emergency_stop_active"] = True
    block = await safety(client, "/api/v2/status")
    assert block["emergency_stop_active"] is True
    assert {"emergency_stop", "state_unreadable:blade_service"} <= set(block["safety_interlocks"])


# --------------------------------------------------------------------------- (e) websocket


def test_the_status_websocket_frame_carries_the_real_safety_block(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    with TestClient(app).websocket_connect("/api/v2/ws/status") as ws:
        quiet = ws.receive_json()["safety_status"]
        assert quiet["emergency_stop_active"] is False
        assert quiet["blade_safety_ok"] is True
        core_globals._safety_state["emergency_stop_active"] = True
        frame = ws.receive_json()
        # a frame already in flight may predate the latch; the next one must carry it
        if not frame["safety_status"]["emergency_stop_active"]:
            frame = ws.receive_json()
    latched = frame["safety_status"]
    assert latched["emergency_stop_active"] is True
    assert latched["blade_safety_ok"] is False
    assert "emergency_stop" in latched["safety_interlocks"]
    assert {"timestamp", "battery_percentage", "navigation_state", "position"} <= set(frame)


# --------------------------------------------------------------------------- (f) metrics


async def test_dashboard_metrics_count_the_listed_interlocks(client, monkeypatch):
    core_globals._safety_state["emergency_stop_active"] = True
    hat = RoboHATService()
    hat._estop_hold = True
    monkeypatch.setattr(robohat_mod, "robohat_service", hat)
    listed = (await safety(client, "/api/v2/status"))["safety_interlocks"]
    assert len(listed) >= 2
    resp = await client.get("/api/v2/dashboard/metrics")
    assert resp.status_code == 200
    assert resp.json()["safety"]["interlocks_active"] == len(listed)


async def test_dashboard_blade_active_follows_the_tractor(client, monkeypatch):
    tractor, _chip = mowing_tractor()
    tractor.state.blade_engaged = True
    monkeypatch.setattr(tractor_mod, "_tractor_service", tractor)
    resp = await client.get("/api/v2/dashboard/status")
    assert resp.json()["blade_active"] is True
    tractor.state.blade_engaged = False
    resp = await client.get("/api/v2/dashboard/status")
    assert resp.json()["blade_active"] is False
