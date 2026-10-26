"""``NavigationService.go_to_waypoint(should_continue=...)``: the gate the Thor plan follower uses.

A leg driven for the Thor plan belongs to no mission. The gate is polled at the top of every
control tick (before anything is commanded, even while waiting for a position); the first
False soft-stops the platform and returns without arriving; and such a leg must never touch
the mission-service singleton (calling it unprimed would create it around a bogus
navigation service).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.src.models import Position
from backend.src.models.mission import MissionWaypoint
from backend.src.services.navigation_service import NavigationService

HERE = Position(latitude=40.0, longitude=-83.0)
TARGET = Position(latitude=40.0005, longitude=-83.0005)


class _State:
    engine_running = True


class _Tractor:
    enabled = True

    def __init__(self) -> None:
        self.state = _State()
        self.applied: list = []
        self.levers_calls: list[tuple[float, float]] = []
        self.emergency_stops = 0
        self.on_apply = None

    async def apply(self, cmd, **_kw):
        self.applied.append(cmd)
        if self.on_apply is not None:
            self.on_apply()
        return {"status": "applied"}

    async def set_levers(self, left, right, **_kw):
        self.levers_calls.append((left, right))
        return {"status": "ok"}

    async def emergency_stop(self, **_kw):
        self.emergency_stops += 1


def _wire(monkeypatch, tractor, *, mission_service=None):
    def no_mission_service():
        raise AssertionError("a gated leg must not call get_mission_service()")

    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(
        "backend.src.services.mission_service.get_mission_service",
        (lambda: mission_service) if mission_service is not None else no_mission_service,
    )


def _nav(position=HERE) -> NavigationService:
    nav = NavigationService()
    nav.navigation_state.current_position = position
    nav.navigation_state.heading = 0.0
    return nav


def _leg() -> MissionWaypoint:
    return MissionWaypoint(lat=TARGET.latitude, lon=TARGET.longitude, speed=50)


async def test_a_gate_that_is_false_at_once_stops_the_platform_without_commanding_motion(
    monkeypatch,
):
    tractor = _Tractor()
    _wire(monkeypatch, tractor)
    await _nav().go_to_waypoint(_leg(), should_continue=lambda: False)
    assert tractor.applied == [] and tractor.levers_calls == [(0.0, 0.0)]
    assert tractor.emergency_stops == 0, "a halted leg is a soft stop, not an e-stop"


async def test_the_gate_is_polled_every_tick_and_the_first_false_ends_the_leg(monkeypatch):
    tractor = _Tractor()
    _wire(monkeypatch, tractor)
    polls: list[int] = []

    def gate() -> bool:
        polls.append(len(tractor.applied))
        return len(tractor.applied) < 2

    await _nav().go_to_waypoint(_leg(), should_continue=gate)
    assert len(tractor.applied) == 2, "two ticks were allowed, the third was not"
    assert polls == [0, 1, 2], "polled before each command, never after one"
    assert tractor.levers_calls == [(0.0, 0.0)]


async def test_the_gate_is_polled_even_while_the_leg_waits_for_a_position(monkeypatch):
    tractor = _Tractor()
    _wire(monkeypatch, tractor)
    nav = _nav(position=None)  # no fix yet: the loop sleeps between polls
    gate_open = {"v": True}
    task = asyncio.create_task(nav.go_to_waypoint(_leg(), should_continue=lambda: gate_open["v"]))
    await asyncio.sleep(0.1)
    assert not task.done()
    gate_open["v"] = False
    await asyncio.wait_for(task, timeout=2.0)
    assert tractor.applied == []
    # Held while it waits for a fix: neutral is commanded, a motion command never is.
    assert tractor.levers_calls and set(tractor.levers_calls) == {(0.0, 0.0)}


async def test_a_gated_leg_that_arrives_ends_the_way_every_leg_does(monkeypatch):
    tractor = _Tractor()
    nav = _nav()
    tractor.on_apply = lambda: setattr(nav.navigation_state, "current_position", TARGET)
    _wire(monkeypatch, tractor)
    await nav.go_to_waypoint(_leg(), should_continue=lambda: True)
    assert len(tractor.applied) == 1 and tractor.levers_calls == [(0.0, 0.0)]


async def test_a_gated_leg_is_not_paused_by_a_mission_that_happens_to_hold_its_waypoint(
    monkeypatch,
):
    """Missions pause by status; a plan leg is governed by its gate alone."""
    leg = _leg()
    mission = SimpleNamespace(waypoints=[leg])
    service = SimpleNamespace(
        missions={"m1": mission}, mission_statuses={"m1": SimpleNamespace(status="paused")}
    )
    tractor = _Tractor()
    nav = _nav()
    tractor.on_apply = lambda: setattr(nav.navigation_state, "current_position", TARGET)
    _wire(monkeypatch, tractor, mission_service=service)
    await nav.go_to_waypoint(leg, should_continue=lambda: True)
    assert len(tractor.applied) == 1, "it drove, although the mission is paused"


async def test_an_ungated_leg_is_still_paused_by_its_mission(monkeypatch):
    leg = _leg()
    service = SimpleNamespace(
        missions={"m1": SimpleNamespace(waypoints=[leg])},
        mission_statuses={"m1": SimpleNamespace(status="paused")},
    )
    tractor = _Tractor()
    _wire(monkeypatch, tractor, mission_service=service)
    await _nav().go_to_waypoint(leg)
    assert tractor.applied == [] and tractor.levers_calls == [(0.0, 0.0)]


def test_the_gate_is_keyword_only():
    with pytest.raises(TypeError):
        asyncio.run(_nav().go_to_waypoint(_leg(), lambda: True))  # type: ignore[misc]
