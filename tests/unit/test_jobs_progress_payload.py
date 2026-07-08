"""The ``jobs.progress`` WebSocket payload is derived from real mission state.

Covers ``mission_service.build_jobs_progress_payload`` which replaced the former
random stub in the telemetry hub (websocket_hub._broadcast_additional_topics).
"""

from backend.src.models.mission import MissionStatus
from backend.src.services.mission_service import (
    MissionService,
    build_jobs_progress_payload,
    current_mission_service,
)


class _NavState:
    current_waypoint_index = 2
    planned_path = (0, 1, 2, 3)


class _FakeNav:
    def __init__(self) -> None:
        self.navigation_state = _NavState()


class _Mission:
    def __init__(self, mission_id: str, name: str) -> None:
        self.id = mission_id
        self.name = name


def _service_with(status: str) -> MissionService:
    svc = MissionService(_FakeNav())
    svc.missions["m1"] = _Mission("m1", "Front Lawn")
    svc.mission_statuses["m1"] = MissionStatus(mission_id="m1", status=status)
    return svc


async def test_idle_payload_when_no_service() -> None:
    payload = await build_jobs_progress_payload(None)
    assert payload["status"] == "idle"
    assert payload["progress_percent"] == 0.0
    assert payload["source"] == "idle"


async def test_idle_payload_when_no_active_mission() -> None:
    payload = await build_jobs_progress_payload(_service_with("idle"))
    assert payload["status"] == "idle"
    assert payload["current_job"] == ""


async def test_running_mission_reports_real_progress() -> None:
    payload = await build_jobs_progress_payload(_service_with("running"))
    # current_waypoint_index 2 of 4 planned points => 50%
    assert payload["current_job"] == "Front Lawn"
    assert payload["status"] == "running"
    assert payload["progress_percent"] == 50.0
    assert payload["source"] == "mission"


async def test_paused_mission_reports_last_known_progress() -> None:
    svc = _service_with("paused")
    # Paused missions keep their last known completion percentage (get_mission_status
    # only recomputes for running missions).
    svc.mission_statuses["m1"].completion_percentage = 33.0
    payload = await build_jobs_progress_payload(svc)
    assert payload["status"] == "paused"
    assert payload["progress_percent"] == 33.0


async def test_current_mission_service_accessor_is_safe() -> None:
    # Returns the module singleton (None until the API creates it, or an
    # instance if another test already did) — never raises.
    result = current_mission_service()
    assert result is None or isinstance(result, MissionService)
