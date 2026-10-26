"""A database written by an older schema version keeps its data and works with the current code.

This replaces ``test_migration_v1_to_v2.py``, a ``pytest.fail("not implemented")`` placeholder for
the LawnBerry *v1 -> v2 system import* (``lawnberry-pi migrate from-v1 / zones / jobs`` and
``verify migration``, ``docs/installation-setup-guide.md``). That importer does not exist: there is
no such command, no ``[project.scripts]`` entry and no reader of v1 files. The only migration code
in the repository is the versioned SQLite schema chain in ``PersistenceLayer`` (v1 -> v4), which is
what ``docs/RELEASE_NOTES.md`` calls "migration tests" and which had none.

``V1_SCHEMA`` is a frozen copy of the schema-version-1 DDL, so these tests stay meaningful even if
someone edits ``PersistenceLayer.MIGRATIONS[0]`` (the classic way to break existing installs).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.src.core.persistence import Migration, PersistenceLayer

V1_SCHEMA = """
CREATE TABLE system_config (
    id INTEGER PRIMARY KEY,
    config_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE planning_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,
    zones_json TEXT NOT NULL,
    priority INTEGER DEFAULT 1,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run TIMESTAMP,
    status TEXT DEFAULT 'pending'
);
CREATE TABLE telemetry_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_json TEXT NOT NULL
);
CREATE TABLE hardware_telemetry_streams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    component_id TEXT NOT NULL,
    value TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    stream_json TEXT NOT NULL,
    verification_artifact_id TEXT,
    UNIQUE(timestamp, component_id)
);
CREATE INDEX idx_telemetry_timestamp ON hardware_telemetry_streams(timestamp DESC);
CREATE INDEX idx_telemetry_component ON hardware_telemetry_streams(component_id);
CREATE TABLE map_zones (
    id TEXT PRIMARY KEY,
    name TEXT,
    polygon_json TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    exclusion_zone BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO schema_version (version) VALUES (1);
"""

SYSTEM_CONFIG = {"telemetry": {"cadence_hz": 5}, "device_name": "mower-1"}
JOB_ZONES = ["front", "back"]
POLYGON = [{"latitude": 40.0, "longitude": -75.0}, {"latitude": 40.0, "longitude": -74.9}]


def make_v1_database(path: Path) -> None:
    """Create a database exactly as schema version 1 left it, holding user data."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(V1_SCHEMA)
        conn.execute(
            "INSERT INTO system_config (id, config_json) VALUES (1, ?)",
            (json.dumps(SYSTEM_CONFIG),),
        )
        conn.execute(
            "INSERT INTO planning_jobs "
            "(id, name, schedule, zones_json, priority, enabled, created_at, status) "
            "VALUES ('job-v1', 'Front lawn', '08:00', ?, 2, 1, '2025-09-01 08:00:00', 'pending')",
            (json.dumps(JOB_ZONES),),
        )
        conn.execute(
            "INSERT INTO map_zones (id, name, polygon_json, priority, exclusion_zone) "
            "VALUES ('front', 'Front', ?, 3, 0), ('bed', 'Flower bed', ?, 9, 1)",
            (json.dumps(POLYGON), json.dumps(POLYGON)),
        )
        conn.commit()
    finally:
        conn.close()


def tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {name for (name,) in rows}


def schema_versions(path: Path) -> list[tuple[int, str]]:
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT version, applied_at FROM schema_version ORDER BY version"
        ).fetchall()


@pytest.mark.asyncio
async def test_schema_v1_database_is_upgraded_in_place(tmp_path):
    db = tmp_path / "mower.db"
    make_v1_database(db)
    assert [v for v, _ in schema_versions(db)] == [1]
    assert not {"audit_logs", "map_config"} & tables(db)

    layer = PersistenceLayer(str(db))

    versions = [v for v, _ in schema_versions(db)]
    assert versions[-1] == PersistenceLayer.SCHEMA_VERSION == 4
    assert {"audit_logs", "map_config"} <= tables(db)

    # Everything the operator saved under v1 is still there and readable through today's API.
    assert layer.load_system_config() == SYSTEM_CONFIG
    assert layer.load_planning_jobs() == [
        {
            "id": "job-v1",
            "name": "Front lawn",
            "schedule": "08:00",
            "priority": 2,
            "enabled": 1,
            "created_at": "2025-09-01 08:00:00",
            "last_run": None,
            "status": "pending",
            "zones": JOB_ZONES,
        }
    ]
    zones = {z["id"]: z for z in layer.load_map_zones()}
    assert set(zones) == {"front", "bed"}
    assert zones["bed"]["exclusion_zone"] == 1 and zones["bed"]["priority"] == 9
    assert zones["front"]["polygon"] == POLYGON and zones["front"]["name"] == "Front"

    # The upgraded database accepts writes through the features added after v1.
    layer.add_audit_log("upgrade.check", details={"from": 1})
    assert layer.load_audit_logs(limit=1)[0]["action"] == "upgrade.check"
    await layer.save_map_configuration("default", json.dumps({"zones": []}))
    assert json.loads(await layer.load_map_configuration("default")) == {"zones": []}
    layer.save_planning_job({"id": "job-new", "name": "Back", "schedule": "09:00", "zones": ["x"]})
    assert {j["id"] for j in layer.load_planning_jobs()} == {"job-v1", "job-new"}


def test_reopening_an_upgraded_database_reapplies_nothing(tmp_path):
    db = tmp_path / "mower.db"
    make_v1_database(db)
    PersistenceLayer(str(db))
    # Back-date the bookkeeping so a re-applied migration (INSERT OR REPLACE ... applied_at =
    # now) is visible even though both opens happen within the same second.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE schema_version SET applied_at = '2001-01-01 00:00:00'")

    reopened = PersistenceLayer(str(db))

    assert schema_versions(db) == [(v, "2001-01-01 00:00:00") for v in (1, 2, 3, 4)], (
        "migrations must run once, not on every start"
    )
    assert [j["id"] for j in reopened.load_planning_jobs()] == ["job-v1"]
    assert reopened.load_system_config() == SYSTEM_CONFIG


def test_failed_migration_is_not_swallowed_and_is_retried(tmp_path, monkeypatch):
    """A broken migration stops startup, claims no new version, and the retry succeeds."""
    db = tmp_path / "mower.db"
    make_v1_database(db)
    PersistenceLayer(str(db))
    good = list(PersistenceLayer.MIGRATIONS)

    broken = Migration(
        version=5,
        description="broken on purpose",
        sql="CREATE TABLE IF NOT EXISTS later (id INTEGER); THIS IS NOT SQL;"
        "INSERT OR REPLACE INTO schema_version (version) VALUES (5);",
    )
    monkeypatch.setattr(PersistenceLayer, "MIGRATIONS", [*good, broken])
    with pytest.raises(sqlite3.OperationalError):
        PersistenceLayer(str(db))
    assert max(v for v, _ in schema_versions(db)) == 4, "a failed upgrade must not claim success"

    fixed = Migration(
        version=5,
        description="fixed",
        sql="CREATE TABLE IF NOT EXISTS later (id INTEGER);"
        "INSERT OR REPLACE INTO schema_version (version) VALUES (5);",
    )
    monkeypatch.setattr(PersistenceLayer, "MIGRATIONS", [*good, fixed])
    layer = PersistenceLayer(str(db))

    assert max(v for v, _ in schema_versions(db)) == 5
    assert [j["id"] for j in layer.load_planning_jobs()] == ["job-v1"], "user data survives both"
