import asyncio
import datetime

from fastapi import Depends

from ..models import NavigationMode
from ..models.mission import Mission, MissionStatus, MissionWaypoint
from ..services.navigation_service import NavigationService


class MissionService:
    def __init__(self, navigation_service: NavigationService):
        self.nav_service = navigation_service
        self.missions: dict[str, Mission] = {}
        self.mission_statuses: dict[str, MissionStatus] = {}
        self.mission_tasks: dict[str, asyncio.Task] = {}

    async def create_mission(self, name: str, waypoints: list[MissionWaypoint]) -> Mission:
        # Your geofence validation logic here
        # For example:
        # if not self.nav_service.are_waypoints_in_geofence(waypoints):
        #     raise ValueError("One or more waypoints are outside the geofence.")

        mission = Mission(
            name=name,
            waypoints=waypoints,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        self.missions[mission.id] = mission
        self.mission_statuses[mission.id] = MissionStatus(mission_id=mission.id, status="idle")
        return mission

    async def start_mission(self, mission_id: str):
        if mission_id not in self.missions:
            raise ValueError("Mission not found.")

        mission = self.missions[mission_id]
        self.mission_statuses[mission_id] = MissionStatus(
            mission_id=mission.id, status="running", current_waypoint_index=0
        )

        task = asyncio.create_task(self.nav_service.execute_mission(mission))
        self.mission_tasks[mission_id] = task

        # Monitor task completion
        task.add_done_callback(self._mission_completed_callback(mission_id))

    def _mission_completed_callback(self, mission_id: str):
        def callback(task: asyncio.Task, mid=mission_id):
            try:
                task.result()
                if self.mission_statuses[mid].status == "running":
                    self.mission_statuses[mid].status = "completed"
                    self.mission_statuses[mid].completion_percentage = 100
            except asyncio.CancelledError:
                self.mission_statuses[mid].status = "aborted"
            except Exception as e:
                self.mission_statuses[mid].status = "failed"
                print(f"Mission {mid} failed: {e}")
            finally:
                del self.mission_tasks[mid]

        return callback

    async def pause_mission(self, mission_id: str):
        if (
            mission_id not in self.mission_statuses
            or self.mission_statuses[mission_id].status != "running"
        ):
            raise ValueError("Mission is not running or does not exist.")
        self.mission_statuses[mission_id].status = "paused"
        if mission_id in self.mission_tasks:
            # Pausing is handled by the navigation service by changing the mode
            self.nav_service.navigation_state.navigation_mode = NavigationMode.PAUSED

    async def resume_mission(self, mission_id: str):
        if (
            mission_id not in self.mission_statuses
            or self.mission_statuses[mission_id].status != "paused"
        ):
            raise ValueError("Mission is not paused or does not exist.")
        self.mission_statuses[mission_id].status = "running"

        # The navigation service's execute_mission loop will continue
        self.nav_service.navigation_state.navigation_mode = NavigationMode.AUTO

        # Re-create a task to continue the mission from where it left off.
        # The state is maintained in navigation_service.
        mission = self.missions[mission_id]
        task = asyncio.create_task(self.nav_service.execute_mission(mission))
        self.mission_tasks[mission_id] = task
        task.add_done_callback(self._mission_completed_callback(mission_id))

    async def abort_mission(self, mission_id: str):
        if mission_id not in self.mission_statuses:
            raise ValueError("Mission not found.")
        self.mission_statuses[mission_id].status = "aborted"
        if mission_id in self.mission_tasks:
            self.mission_tasks[mission_id].cancel()
            del self.mission_tasks[mission_id]
        await self.nav_service.stop_navigation()

    async def get_mission_status(self, mission_id: str) -> MissionStatus:
        if mission_id not in self.mission_statuses:
            raise ValueError("Mission not found.")

        status = self.mission_statuses[mission_id]

        if status.status == "running":
            nav_state = self.nav_service.navigation_state
            status.current_waypoint_index = nav_state.current_waypoint_index
            if len(nav_state.planned_path) > 0:
                status.completion_percentage = (
                    nav_state.current_waypoint_index / len(nav_state.planned_path)
                ) * 100
            else:
                status.completion_percentage = 0

        return status

    async def list_missions(self) -> list[Mission]:
        return list(self.missions.values())


# Dependency injection
_mission_service_instance = None


def get_mission_service(
    nav_service: NavigationService = Depends(NavigationService.get_instance),  # noqa: B008
) -> "MissionService":
    global _mission_service_instance
    if _mission_service_instance is None:
        _mission_service_instance = MissionService(nav_service)
    return _mission_service_instance


def current_mission_service() -> "MissionService | None":
    """Return the active MissionService singleton if one has been created (via
    the API's dependency injection), else None.

    Lets background consumers such as the telemetry WebSocket hub read live
    mission progress without forcing creation or needing a NavigationService
    for FastAPI dependency injection.
    """
    return _mission_service_instance


async def build_jobs_progress_payload(
    service: "MissionService | None",
) -> dict[str, object]:
    """Build the ``jobs.progress`` WebSocket payload from live mission state.

    Replaces the former random stub in the telemetry hub with real progress:
    the first running (else paused) mission's completion percentage and status,
    or an idle payload when nothing is active.

    ``current_job`` is the mission's name. Reconciling it with the separate
    Planning-job store (the frontend matches ``jobs.progress`` to a Planning
    job by name) is best-effort until missions and planning jobs share an
    identity; ``remaining_time_min`` is not tracked yet and is reported as 0
    (the dashboard hides it when zero).
    """
    idle: dict[str, object] = {
        "current_job": "",
        "progress_percent": 0.0,
        "remaining_time_min": 0,
        "status": "idle",
        "source": "idle",
    }
    if service is None:
        return idle

    # Prefer a running mission; fall back to a paused one.
    active_id: str | None = None
    for mission_id, mission_status in service.mission_statuses.items():
        if mission_status.status == "running":
            active_id = mission_id
            break
        if mission_status.status == "paused" and active_id is None:
            active_id = mission_id
    if active_id is None:
        return idle

    status = await service.get_mission_status(active_id)
    mission = service.missions.get(active_id)
    ui_status = (
        "running"
        if status.status == "running"
        else "paused"
        if status.status == "paused"
        else "idle"
    )
    return {
        "current_job": mission.name if mission is not None else active_id,
        "progress_percent": round(float(status.completion_percentage or 0.0), 1),
        "remaining_time_min": 0,
        "status": ui_status,
        "source": "mission",
    }
