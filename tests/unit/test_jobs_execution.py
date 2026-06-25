"""Unit tests for job execution dispatching to the mission pipeline."""

import asyncio

import pytest

from backend.src.models.job import JobStatus, JobType
from backend.src.services.jobs_service import JobsService


async def _wait_for_terminal(job, timeout_s: float = 5.0):
    deadline = asyncio.get_event_loop().time() + timeout_s
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    while job.status not in terminal and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    return job.status


@pytest.mark.asyncio
async def test_scheduled_mow_job_runs_via_mission_pipeline(tmp_path, monkeypatch):
    # Empty data dir -> no map configuration -> empty mission -> completes fast.
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))

    svc = JobsService()
    job = svc.create_job(name="test mow", job_type=JobType.SCHEDULED_MOW, zones=[])

    assert svc.start_job(job.id) is True
    assert job.status == JobStatus.RUNNING

    status = await _wait_for_terminal(job)
    assert status == JobStatus.COMPLETED
    assert job.progress is not None
    assert job.progress.percentage_complete == 100.0
    assert any("Started mission" in log for log in job.execution_logs)


@pytest.mark.asyncio
async def test_maintenance_job_completes_without_navigation():
    svc = JobsService()
    job = svc.create_job(name="maint", job_type=JobType.MAINTENANCE)

    assert svc.start_job(job.id) is True
    status = await _wait_for_terminal(job)

    assert status == JobStatus.COMPLETED
    assert job.progress.percentage_complete == 100.0
