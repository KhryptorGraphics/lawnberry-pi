"""Shared rig for the autonomous-run safety tests (not a test module).

A real ``NavigationService`` and a real ``MissionService`` run a real mission over either
platform. Only the hardware is faked: ``_tractor_fakes`` for the tractor's PCA9685 / relays, and
``FakeFirmware`` for the RoboHAT, which stands under the REAL ``RoboHATService`` as a fake serial
line sink. ``robohat_platform`` turns SIM_MODE off, so the real (non-simulated) write paths run.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _tractor_fakes import idle_tractor

from backend.src.models import Position
from backend.src.models.mission import Mission, MissionStatus, MissionWaypoint
from backend.src.services import blade_service as blade_module
from backend.src.services import mission_service as mission_module
from backend.src.services import navigation_service as nav_module
from backend.src.services import robohat_service as robohat_module
from backend.src.services.blade_service import BladeService
from backend.src.services.mission_service import MissionService
from backend.src.services.navigation_service import NavigationService
from backend.src.services.robohat_service import RoboHATService

HERE = Position(latitude=40.0, longitude=-83.0)
#: A waypoint about 1.4 km away: a leg that never arrives inside a test.
FAR = MissionWaypoint(lat=40.01, lon=-83.01, blade_on=True)
FAR_POSITION = Position(latitude=FAR.lat, longitude=FAR.lon)


def new_nav() -> NavigationService:
    nav = NavigationService()
    nav.navigation_state.current_position = HERE
    nav.navigation_state.heading = 0.0
    return nav


def new_mission(waypoints) -> Mission:
    return Mission(name="m", waypoints=waypoints, created_at=datetime.now(UTC).isoformat())


def wire_mission(monkeypatch, tractor, nav, waypoints):
    """A real MissionService for ``nav`` holding one running mission over ``waypoints``."""
    mission = new_mission(waypoints)
    service = MissionService(nav)
    service.missions[mission.id] = mission
    service.mission_statuses[mission.id] = MissionStatus(mission_id=mission.id, status="running")
    monkeypatch.setattr(mission_module, "_mission_service_instance", service)
    monkeypatch.setattr(mission_module, "get_mission_service", lambda *a, **k: service)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    return mission, service


class _Serial:
    is_open = True
    in_waiting = 0

    def close(self) -> None:
        self.is_open = False


class FakeFirmware:
    """The RoboHAT firmware as the service sees it on the wire: every line is recorded and acked."""

    def __init__(self, svc: RoboHATService) -> None:
        self.svc = svc
        self.sent: list[str] = []

    async def send_line(self, line: str) -> bool:
        self.sent.append(line)
        if line == "rc=disable":
            self.svc._process_line("[USB] RC disabled: USB control")
        elif line.startswith("pwm,"):
            self.svc._process_line("[USB] PWM set -> steer=1500 us throttle=1500 us")
        elif line.startswith("blade="):
            self.svc._process_line(f"[USB] Blade {'enabled' if line == 'blade=on' else 'disabled'}")
        return True


def driving(lines: list[str]) -> list[str]:
    """The lines that move or spin something: a non-neutral pwm, or the blade switched on."""
    return [
        line
        for line in lines
        if (line.startswith("pwm,") and line != "pwm,1500,1500") or line == "blade=on"
    ]


def robohat_platform(monkeypatch):
    """A disabled tractor and the real RoboHAT service over a fake serial, SIM_MODE off."""
    monkeypatch.setenv("SIM_MODE", "0")  # the real write path, not the simulation shortcut
    svc = RoboHATService()
    svc.serial_conn = _Serial()
    svc.running = True
    svc.status.serial_connected = True
    svc._rc_enabled = False  # already under USB control
    fw = FakeFirmware(svc)
    monkeypatch.setattr(svc, "_send_line", fw.send_line)
    monkeypatch.setattr(nav_module, "get_robohat_service", lambda: svc)
    monkeypatch.setattr(robohat_module, "get_robohat_service", lambda: svc)
    monkeypatch.setattr(blade_module, "_blade_service", BladeService({}))
    return svc, fw, idle_tractor()
