"""Autonomous navigation control endpoints (v2).

Start/stop/pause/resume an autonomous mowing run and poll its status, plus a
simple control-mode switch. Writes require operator auth; status is open so the
dashboard can poll it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...services.autonomy_service import get_autonomy_service
from ..deps import require_operator_auth

router = APIRouter()


@router.post("/navigation/start", dependencies=[Depends(require_operator_auth)])
async def navigation_start(payload: dict | None = None) -> dict[str, Any]:
    """Begin autonomous mowing over the given ``zones`` (or all boundary zones)."""
    zones = payload.get("zones") if isinstance(payload, dict) else None
    return await get_autonomy_service().start(zones)


@router.post("/navigation/stop", dependencies=[Depends(require_operator_auth)])
async def navigation_stop() -> dict[str, Any]:
    return await get_autonomy_service().stop()


@router.post("/navigation/pause", dependencies=[Depends(require_operator_auth)])
async def navigation_pause() -> dict[str, Any]:
    return await get_autonomy_service().pause()


@router.post("/navigation/resume", dependencies=[Depends(require_operator_auth)])
async def navigation_resume() -> dict[str, Any]:
    return await get_autonomy_service().resume()


@router.get("/navigation/status")
async def navigation_status() -> dict[str, Any]:
    return await get_autonomy_service().status()


@router.post("/control/mode", dependencies=[Depends(require_operator_auth)])
async def control_mode(payload: dict) -> dict[str, Any]:
    """Switch control mode: ``autonomous`` starts a run, ``idle``/``manual`` stops it."""
    mode = (payload or {}).get("mode", "idle")
    service = get_autonomy_service()
    if mode == "autonomous":
        return await service.start(payload.get("zones"))
    if mode in {"idle", "manual"}:
        result = await service.stop()
        result["mode"] = mode
        return result
    return {"mode": mode}
