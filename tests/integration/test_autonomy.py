"""Integration tests for autonomous navigation control + job lifecycle."""

import json

import httpx
import pytest

from backend.src.main import app

BASE_URL = "http://test"


def _write_boundary_config(tmp_path):
    (tmp_path / "map_configuration_v2.json").write_text(
        json.dumps(
            {
                "zones": [
                    {
                        "zone_id": "yard",
                        "zone_type": "boundary",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0.0, 0.0],
                                    [0.0003, 0.0],
                                    [0.0003, 0.0003],
                                    [0.0, 0.0003],
                                    [0.0, 0.0],
                                ]
                            ],
                        },
                    }
                ]
            }
        )
    )


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    _write_boundary_config(tmp_path)
    yield


async def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL)


@pytest.mark.asyncio
async def test_navigation_start_status_stop():
    async with await _client() as client:
        start = await client.post("/api/v2/navigation/start", json={"zones": ["yard"]})
        assert start.status_code == 200, start.text
        body = start.json()
        assert body["status"] == "started"
        assert body["mode"] == "autonomous"
        assert body["mission_id"]
        assert body["waypoint_count"] > 0

        status = await client.get("/api/v2/navigation/status")
        assert status.status_code == 200
        assert "mode" in status.json() and "completion_percentage" in status.json()

        stop = await client.post("/api/v2/navigation/stop")
        assert stop.status_code == 200
        assert stop.json()["mode"] == "idle"

        idle = await client.get("/api/v2/navigation/status")
        assert idle.json()["active"] is False


@pytest.mark.asyncio
async def test_control_mode_switch():
    async with await _client() as client:
        on = await client.post("/api/v2/control/mode", json={"mode": "autonomous"})
        assert on.status_code == 200
        assert on.json()["mode"] == "autonomous"

        off = await client.post("/api/v2/control/mode", json={"mode": "idle"})
        assert off.status_code == 200
        assert off.json()["mode"] == "idle"


@pytest.mark.asyncio
async def test_planning_job_lifecycle():
    async with await _client() as client:
        created = await client.post(
            "/api/v2/planning/jobs",
            json={"name": "mow", "schedule": "08:00", "zones": ["yard"], "enabled": True},
        )
        assert created.status_code == 201
        job_id = created.json()["id"]

        started = await client.post(f"/api/v2/planning/jobs/{job_id}/start")
        assert started.status_code == 200, started.text
        assert started.json()["job"]["status"] == "running"

        cancelled = await client.post(f"/api/v2/planning/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["job"]["status"] == "cancelled"

        bad_action = await client.post(f"/api/v2/planning/jobs/{job_id}/explode")
        assert bad_action.status_code == 422

        missing = await client.post("/api/v2/planning/jobs/nope/start")
        assert missing.status_code == 404
