"""W1 capture API: start/stop/status for teleop capture sessions.

Sessions are recorded on the Pi's own disk (``LAWNBERRY_DATA_DIR/captures``)
in the ``capture_integrity`` format; ``stop`` runs the integrity gate
synchronously and returns its report, so a failing session is visible in the
response (and left on disk renamed ``<name>.rejected``). Writes require
operator auth like the tractor endpoints; status is always available.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...services.capture_service import CaptureError, get_capture_service
from ..deps import require_operator_auth

router = APIRouter()


class StartBody(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    )


@router.post("/capture/start", dependencies=[Depends(require_operator_auth)])
async def capture_start(body: StartBody) -> dict[str, Any]:
    try:
        session_dir = await get_capture_service().start_session(body.name)
    except CaptureError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "recording", "session": body.name, "dir": str(session_dir)}


@router.post("/capture/stop", dependencies=[Depends(require_operator_auth)])
async def capture_stop() -> dict[str, Any]:
    try:
        report = await get_capture_service().stop_session()
    except CaptureError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "ok" if report.ok else "rejected",
        "ok": report.ok,
        "problems": report.problems,
        "stats": report.stats,
    }


@router.get("/capture/status")
async def capture_status() -> dict[str, Any]:
    return get_capture_service().status()
