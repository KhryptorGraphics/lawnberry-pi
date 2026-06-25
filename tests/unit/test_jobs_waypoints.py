"""Tests for resolving a job's zones into coverage waypoints."""

import json

from backend.src.models.job import JobType
from backend.src.services.jobs_service import JobsService


def _square(lat0: float, lng0: float, size_deg: float) -> list[list[float]]:
    # GeoJSON ring [lng, lat], closed.
    return [
        [lng0, lat0],
        [lng0 + size_deg, lat0],
        [lng0 + size_deg, lat0 + size_deg],
        [lng0, lat0 + size_deg],
        [lng0, lat0],
    ]


def _write_config(path, zones):
    path.write_text(json.dumps({"zones": zones}))


def test_no_config_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    svc = JobsService()
    job = svc.create_job(name="m", job_type=JobType.SCHEDULED_MOW, zones=["front"])
    assert svc._waypoints_for_job(job) == []


def test_resolves_named_boundary_zone(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    cfg = tmp_path / "map_configuration_v2.json"
    _write_config(
        cfg,
        [
            {
                "zone_id": "front",
                "zone_type": "boundary",
                "geometry": {"type": "Polygon", "coordinates": [_square(0.0, 0.0, 0.0003)]},
            },
            {
                "zone_id": "back",
                "zone_type": "boundary",
                "geometry": {"type": "Polygon", "coordinates": [_square(1.0, 1.0, 0.0003)]},
            },
        ],
    )
    svc = JobsService()
    job = svc.create_job(name="m", job_type=JobType.SCHEDULED_MOW, zones=["front"])
    wps = svc._waypoints_for_job(job)

    assert wps, "expected waypoints for the matching boundary"
    # Only the 'front' zone (near 0,0) should be covered, not 'back' (near 1,1).
    assert all(w.lat < 0.5 and w.lon < 0.5 for w in wps)
    assert all(w.blade_on for w in wps)


def test_exclusion_zone_is_avoided(tmp_path, monkeypatch):
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    cfg = tmp_path / "map_configuration_v2.json"
    # Boundary ~33m square; exclusion covering the upper half.
    _write_config(
        cfg,
        [
            {
                "zone_id": "yard",
                "zone_type": "boundary",
                "geometry": {"type": "Polygon", "coordinates": [_square(0.0, 0.0, 0.0003)]},
            },
            {
                "zone_id": "bed",
                "zone_type": "exclusion",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.00015],
                            [0.0003, 0.00015],
                            [0.0003, 0.0003],
                            [0.0, 0.0003],
                            [0.0, 0.00015],
                        ]
                    ],
                },
            },
        ],
    )
    svc = JobsService()
    job = svc.create_job(
        name="m", job_type=JobType.SCHEDULED_MOW, zones=[]
    )  # empty -> all boundaries
    wps = svc._waypoints_for_job(job)

    assert wps, "expected waypoints"
    # The exclusion covers latitudes >= ~0.00015; coverage should stay below it.
    upper = [w for w in wps if w.lat > 0.000155]
    assert len(upper) < len(wps) * 0.3, "too many waypoints inside the exclusion zone"
