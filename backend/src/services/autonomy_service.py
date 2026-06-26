"""Autonomous mowing orchestration.

Turns the operator's map zones into a real mowing run by building a coverage
path and driving it through the existing mission pipeline
(``MissionService`` -> ``NavigationService.execute_mission`` -> motors). Holds
the currently-active autonomous mission so the operator can pause/resume/stop and
poll progress.
"""

from __future__ import annotations

from typing import Any

from ..nav.zone_coverage import waypoints_from_map_config
from .mission_service import get_mission_service
from .navigation_service import NavigationService


class AutonomyService:
    def __init__(self) -> None:
        self._mission_id: str | None = None

    def _mission_service(self):
        # Prime the shared mission-service singleton with a real nav service so the
        # status that execute_mission updates is the one we poll here.
        return get_mission_service(NavigationService.get_instance())

    async def start(self, zone_ids: list[str] | None = None) -> dict[str, Any]:
        """Begin an autonomous mowing run over the selected (or all) zones."""
        waypoints = waypoints_from_map_config(zone_ids)
        mission_service = self._mission_service()
        mission = await mission_service.create_mission(name="autonomous-mow", waypoints=waypoints)
        await mission_service.start_mission(mission.id)
        self._mission_id = mission.id
        return {
            "status": "started",
            "mode": "autonomous",
            "mission_id": mission.id,
            "waypoint_count": len(waypoints),
        }

    async def stop(self) -> dict[str, Any]:
        """Abort the active run and return to idle."""
        mission_service = self._mission_service()
        mid = self._mission_id
        if mid and mid in mission_service.mission_statuses:
            try:
                await mission_service.abort_mission(mid)
            except Exception:
                pass
        self._mission_id = None
        return {"status": "stopped", "mode": "idle", "mission_id": mid}

    async def pause(self) -> dict[str, Any]:
        if not self._mission_id:
            return {"status": "idle", "mode": "idle", "mission_id": None}
        try:
            await self._mission_service().pause_mission(self._mission_id)
        except Exception:
            pass
        return {"status": "paused", "mode": "paused", "mission_id": self._mission_id}

    async def resume(self) -> dict[str, Any]:
        if not self._mission_id:
            return {"status": "idle", "mode": "idle", "mission_id": None}
        try:
            await self._mission_service().resume_mission(self._mission_id)
        except Exception:
            pass
        return {"status": "running", "mode": "autonomous", "mission_id": self._mission_id}

    async def return_to_base(self) -> dict[str, Any]:
        """Command the mower to navigate back to its home/dock position."""
        nav = NavigationService.get_instance()
        ok = await nav.return_home()
        return {
            "status": "returning" if ok else "error",
            "mode": "return_home" if ok else "idle",
            "mission_id": self._mission_id,
            "detail": None if ok else "No home position set or no current position fix",
        }

    async def status(self) -> dict[str, Any]:
        mission_service = self._mission_service()
        mission_status: str | None = None
        completion = 0.0
        if self._mission_id and self._mission_id in mission_service.mission_statuses:
            st = await mission_service.get_mission_status(self._mission_id)
            mission_status = st.status
            completion = float(st.completion_percentage or 0.0)
        active = mission_status in {"running", "paused"}
        if mission_status == "running":
            mode = "autonomous"
        elif mission_status == "paused":
            mode = "paused"
        else:
            mode = "idle"
        return {
            "mode": mode,
            "active": active,
            "mission_id": self._mission_id if active else None,
            "mission_status": mission_status,
            "completion_percentage": completion,
        }


autonomy_service = AutonomyService()


def get_autonomy_service() -> AutonomyService:
    return autonomy_service
