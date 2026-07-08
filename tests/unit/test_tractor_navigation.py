"""Tests for Ackermann tractor autonomous waypoint navigation in NavigationService.

Modeled on tests/unit/test_ai_navigation_loop.py's style: direct NavigationService
instantiation with monkeypatch on the lazily-imported get_tractor_service()/
get_mission_service() factories (no FastAPI app, no real hardware).
"""

from types import SimpleNamespace

import pytest

from backend.src.models import NavigationMode, Position
from backend.src.models.mission import MissionWaypoint
from backend.src.models.tractor_control import Transmission
from backend.src.services.navigation_service import NavigationService


class _FakeTractorState:
    def __init__(self, engine_running: bool):
        self.engine_running = engine_running


class _FakeTractor:
    """Minimal stand-in for TractorControlService, fully controlled by tests.

    Deliberately has no authorize()/revoke()/clear_emergency() — if production
    code ever calls one of those, the test fails loudly with AttributeError
    instead of silently passing.
    """

    def __init__(self, enabled: bool, engine_running: bool = False):
        self.enabled = enabled
        self.state = _FakeTractorState(engine_running)
        self.applied: list = []
        self.ground_speed_calls: list[float] = []
        self.emergency_stop_calls = 0
        self.start_engine_calls = 0
        self.start_engine_result = {"status": "ok", "engine": "running"}

    async def apply(self, cmd):
        self.applied.append(cmd)
        return {"status": "applied"}

    async def set_ground_speed(self, value: float):
        self.ground_speed_calls.append(value)
        return {"status": "ok", "ground_speed": value}

    async def start_engine(self):
        self.start_engine_calls += 1
        if self.start_engine_result.get("status") == "ok":
            self.state.engine_running = True
        return self.start_engine_result

    async def emergency_stop(self):
        self.emergency_stop_calls += 1
        return {"status": "emergency_stop"}


def _patch_tractor(monkeypatch, tractor: _FakeTractor) -> None:
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)


def _patch_empty_mission_service(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.src.services.mission_service.get_mission_service",
        lambda: SimpleNamespace(missions={}, mission_statuses={}),
    )


# --------------------- go_to_waypoint: tractor command math ---------------------


async def test_go_to_waypoint_tractor_command_sign_and_values(monkeypatch):
    """A known +20deg heading error must produce a positive (right) steering
    command and a ground_speed normalized from the same forward_speed math the
    differential path uses, with the waypoint's blade flag carried through."""
    nav = NavigationService()
    nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)

    waypoint = MissionWaypoint(lat=40.0005, lon=-83.0005, blade_on=True, speed=50)
    target_pos = Position(latitude=waypoint.lat, longitude=waypoint.lon)
    bearing = nav.path_planner.calculate_bearing(nav.navigation_state.current_position, target_pos)
    # Current heading trails the target bearing by 20deg -> heading_error = +20
    # (a right turn is needed), safely away from the 30deg sharp-turn threshold.
    nav.navigation_state.heading = (bearing - 20.0) % 360

    _patch_empty_mission_service(monkeypatch)
    tractor = _FakeTractor(enabled=True)

    async def _apply_then_arrive(cmd):
        tractor.applied.append(cmd)
        nav.navigation_state.current_position = target_pos  # arrive next tick
        return {"status": "applied"}

    tractor.apply = _apply_then_arrive
    _patch_tractor(monkeypatch, tractor)

    await nav.go_to_waypoint(waypoint)

    assert len(tractor.applied) == 1
    cmd = tractor.applied[0]
    assert cmd.steering == pytest.approx(20.0 / 45.0, abs=1e-6)
    assert cmd.ground_speed == pytest.approx(0.5, abs=1e-6)  # 0.4 base / 0.8 max_speed
    assert cmd.gear == Transmission.FORWARD
    assert cmd.clutch == 0.0
    assert cmd.blade_engaged is True
    assert cmd.throttle == pytest.approx(nav.tractor_engine_throttle)

    # Arrival is a soft stop: ground speed zeroed, no authorization revoked.
    assert tractor.ground_speed_calls == [0.0]
    assert tractor.emergency_stop_calls == 0


# ------------------------- soft-stop vs. hard-stop -------------------------


async def test_go_to_waypoint_interrupted_uses_soft_stop(monkeypatch):
    """Mission interruption (pause/abort) must zero ground speed without
    revoking tractor authorization -- never emergency_stop()."""
    nav = NavigationService()
    nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
    nav.navigation_state.heading = 0.0

    waypoint = MissionWaypoint(lat=40.01, lon=-83.01)
    monkeypatch.setattr(
        "backend.src.services.mission_service.get_mission_service",
        lambda: SimpleNamespace(
            missions={"m1": SimpleNamespace(waypoints=[waypoint])},
            mission_statuses={"m1": SimpleNamespace(status="paused")},
        ),
    )
    tractor = _FakeTractor(enabled=True)
    _patch_tractor(monkeypatch, tractor)

    await nav.go_to_waypoint(waypoint)

    assert tractor.ground_speed_calls == [0.0]
    assert tractor.emergency_stop_calls == 0
    assert tractor.applied == []


async def test_go_to_waypoint_apply_failure_triggers_hard_stop(monkeypatch):
    """An actual tractor.apply() fault is the one case that escalates to the
    full emergency_stop() (authorization revoked), not the soft stop."""
    nav = NavigationService()
    nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
    nav.navigation_state.heading = 0.0

    waypoint = MissionWaypoint(lat=40.001, lon=-83.001)
    _patch_empty_mission_service(monkeypatch)
    tractor = _FakeTractor(enabled=True)

    async def _boom(cmd):
        raise RuntimeError("i2c bus fault")

    tractor.apply = _boom
    _patch_tractor(monkeypatch, tractor)

    await nav.go_to_waypoint(waypoint)

    assert tractor.emergency_stop_calls == 1
    assert tractor.ground_speed_calls == []


# --------------------------- engine auto-start gating ---------------------------


async def test_execute_mission_auto_starts_engine_when_not_running(monkeypatch):
    nav = NavigationService()
    _patch_empty_mission_service(monkeypatch)
    tractor = _FakeTractor(enabled=True, engine_running=False)
    _patch_tractor(monkeypatch, tractor)
    mission = SimpleNamespace(id="m1", name="Mission 1", waypoints=[])

    await nav.execute_mission(mission)

    assert tractor.start_engine_calls == 1
    assert tractor.state.engine_running is True
    # Mission proceeded past the engine check (no waypoints -> completes idle).
    assert nav.navigation_state.navigation_mode == NavigationMode.IDLE


async def test_execute_mission_skips_auto_start_when_engine_already_running(monkeypatch):
    nav = NavigationService()
    _patch_empty_mission_service(monkeypatch)
    tractor = _FakeTractor(enabled=True, engine_running=True)
    _patch_tractor(monkeypatch, tractor)
    mission = SimpleNamespace(id="m2", name="Mission 2", waypoints=[])

    await nav.execute_mission(mission)

    # Guarded on "not already running" so resume_mission()'s re-invocation
    # never re-cranks a running engine.
    assert tractor.start_engine_calls == 0


async def test_execute_mission_raises_clearly_when_engine_start_rejected(monkeypatch):
    nav = NavigationService()
    _patch_empty_mission_service(monkeypatch)
    tractor = _FakeTractor(enabled=True, engine_running=False)
    tractor.start_engine_result = {"status": "rejected", "reason": "motors not authorized"}
    _patch_tractor(monkeypatch, tractor)
    mission = SimpleNamespace(id="m3", name="Mission 3", waypoints=[])

    with pytest.raises(RuntimeError, match="motors not authorized"):
        await nav.execute_mission(mission)

    # Failed before any state mutation, and never auto-authorized on the
    # operator's behalf (the fake has no authorize() -- would AttributeError).
    assert nav.navigation_state.navigation_mode == NavigationMode.IDLE


# ------------------------------ emergency_stop() ------------------------------


async def test_emergency_stop_calls_tractor_when_enabled(monkeypatch):
    nav = NavigationService()
    tractor = _FakeTractor(enabled=True)
    _patch_tractor(monkeypatch, tractor)

    assert await nav.emergency_stop() is True
    assert tractor.emergency_stop_calls == 1


async def test_emergency_stop_skips_tractor_when_disabled(monkeypatch):
    nav = NavigationService()
    tractor = _FakeTractor(enabled=False)
    _patch_tractor(monkeypatch, tractor)

    assert await nav.emergency_stop() is True
    assert tractor.emergency_stop_calls == 0


# --------------------- regression guard: differential path ---------------------


async def test_go_to_waypoint_differential_path_unchanged_when_tractor_disabled(monkeypatch):
    """Pins today's exact differential-drive output so the tractor branch added
    alongside it can never change what the existing mower fleet receives."""
    scenarios = (
        # (heading_offset_deg, expected_left, expected_right)
        (20.0, 0.4 * (1 - 20.0 / 45.0), 0.4 * (1 + 20.0 / 45.0)),
        (40.0, 0.2 * (1 - 40.0 / 45.0), 0.2 * (1 + 40.0 / 45.0)),  # >30 -> halved
    )
    for heading_offset, expected_left, expected_right in scenarios:
        nav = NavigationService()
        nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
        waypoint = MissionWaypoint(lat=40.0005, lon=-83.0005, speed=50)
        target_pos = Position(latitude=waypoint.lat, longitude=waypoint.lon)
        bearing = nav.path_planner.calculate_bearing(
            nav.navigation_state.current_position, target_pos
        )
        nav.navigation_state.heading = (bearing - heading_offset) % 360

        _patch_empty_mission_service(monkeypatch)
        tractor = _FakeTractor(enabled=False)
        _patch_tractor(monkeypatch, tractor)

        speed_calls: list[tuple[float, float]] = []

        async def _capture(left, right, _calls=speed_calls, _nav=nav, _arrive_at=target_pos):
            _calls.append((left, right))
            _nav.navigation_state.current_position = _arrive_at  # arrive next tick

        monkeypatch.setattr(nav, "set_speed", _capture)

        await nav.go_to_waypoint(waypoint)

        assert len(speed_calls) == 2, heading_offset
        left, right = speed_calls[0]
        assert left == pytest.approx(expected_left, abs=1e-6), heading_offset
        assert right == pytest.approx(expected_right, abs=1e-6), heading_offset
        assert speed_calls[1] == (0.0, 0.0)  # soft stop at arrival
        assert tractor.applied == []
