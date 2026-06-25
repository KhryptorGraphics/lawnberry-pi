"""Mission planning job endpoints (v2).

CRUD for operator-defined scheduled mowing jobs (name, schedule, zones,
priority, enabled). Jobs are persisted to the data directory so the schedule
survives a restart.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response

from ..deps import require_operator_auth

router = APIRouter()

_JOBS_FILE = "planning_jobs_v2.json"


def _data_dir() -> Path:
    path = Path(os.getenv("LAWNBERRY_DATA_DIR", "./data"))
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def _load_jobs() -> dict[str, dict[str, Any]]:
    path = _data_dir() / _JOBS_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    try:
        (_data_dir() / _JOBS_FILE).write_text(json.dumps(jobs, indent=2, default=str))
    except Exception:
        pass


@router.get("/planning/jobs")
async def list_planning_jobs() -> list[dict[str, Any]]:
    return list(_load_jobs().values())


@router.post("/planning/jobs", dependencies=[Depends(require_operator_auth)])
async def create_planning_job(payload: dict) -> JSONResponse:
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "name": payload.get("name", ""),
        "schedule": payload.get("schedule"),
        "zones": payload.get("zones", []),
        "priority": payload.get("priority", 1),
        "enabled": payload.get("enabled", True),
        "created_at": datetime.now(UTC).isoformat(),
    }
    jobs = _load_jobs()
    jobs[job_id] = job
    _save_jobs(jobs)
    return JSONResponse(status_code=201, content=job)


@router.delete("/planning/jobs/{job_id}", dependencies=[Depends(require_operator_auth)])
async def delete_planning_job(job_id: str) -> Response:
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Planning job not found")
    del jobs[job_id]
    _save_jobs(jobs)
    return Response(status_code=204)


# Job lifecycle: a planning job is an autonomous mowing run over its zones.
_JOB_ACTIONS = {"start", "pause", "resume", "cancel"}


@router.post("/planning/jobs/{job_id}/{action}", dependencies=[Depends(require_operator_auth)])
async def planning_job_action(job_id: str, action: str) -> JSONResponse:
    if action not in _JOB_ACTIONS:
        raise HTTPException(status_code=422, detail=f"Unknown action: {action}")
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Planning job not found")

    from ...services.autonomy_service import get_autonomy_service

    autonomy = get_autonomy_service()
    job = jobs[job_id]

    if action == "start":
        result = await autonomy.start(job.get("zones") or None)
        job["status"] = "running"
        job["mission_id"] = result.get("mission_id")
    elif action == "pause":
        result = await autonomy.pause()
        job["status"] = "paused"
    elif action == "resume":
        result = await autonomy.resume()
        job["status"] = "running"
    else:  # cancel
        result = await autonomy.stop()
        job["status"] = "cancelled"

    jobs[job_id] = job
    _save_jobs(jobs)
    return JSONResponse(status_code=200, content={"job": job, "autonomy": result})
