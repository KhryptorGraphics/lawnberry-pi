"""SQLite persistence layer and database migrations.

This module provides a lightweight persistence layer for the LawnBerry Pi v2 system,
handling database schema creation, migrations, and basic CRUD operations for
persistent data like job schedules, configuration, and telemetry history.
"""
# ruff: noqa: E501

import json
import logging
import sqlite3
import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .simulation import is_simulation_mode

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Database migration definition."""

    version: int
    description: str
    sql: str


class AuditCoalescer:
    """Coalesce floods of identical audit events into one row plus a count.

    The first event for an action after a quiet period is written at once (an operator can
    always see that it happened, and from where: its ``client_id``). Further events for the same
    action inside ``window_s`` only bump a counter and note their ``client_id``; when the window
    closes one ``<action>.coalesced`` summary row records how many were suppressed and WHO they
    came from (``peers``, at most ``MAX_PEERS`` distinct client ids, ``peers_truncated`` when
    more were seen), so nothing is silently lost: the unauthenticated E-stop routes write only
    through here, and this is the only record of who stopped the mower (SEC3-13). Writes run on
    a worker thread (never on the caller / event-loop thread) and failures are logged, never
    raised. The number of tracked keys is capped; excess keys share one overflow bucket.
    """

    MAX_KEYS = 64
    MAX_PEERS = 16
    OVERFLOW_KEY = "*overflow*"

    def __init__(self, layer: "PersistenceLayer") -> None:
        self._layer = layer
        self._lock = threading.Lock()
        self._windows: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit-coalesce")

    def _write(self, action: str, client_id, resource, details) -> None:
        try:
            self._layer.add_audit_log(
                action, client_id=client_id, resource=resource, details=details
            )
        except Exception as exc:  # audit must never break the caller
            logger.error("Coalesced audit write failed for %s: %s", action, exc)

    def _submit(self, *args: Any) -> None:
        try:
            self._executor.submit(self._write, *args)
        except Exception as exc:  # executor shut down
            logger.error("Coalesced audit submit failed: %s", exc)

    def record(
        self,
        action: str,
        client_id: str | None,
        resource: str | None,
        details: dict[str, Any] | None,
        window_s: float,
    ) -> bool:
        """Record an event; return True when it was written (first in its window)."""
        try:
            with self._lock:
                key = action
                if key not in self._windows and len(self._windows) >= self.MAX_KEYS - 1:
                    key = self.OVERFLOW_KEY
                win = self._windows.get(key)
                if win is not None:
                    win["suppressed"] += 1
                    self._note_peer(win, client_id)
                    return False
                win = {
                    "suppressed": 0,
                    "action": key,
                    "client_id": client_id,
                    "resource": resource,
                    "window_s": window_s,
                    "peers": {client_id} if client_id else set(),
                    "peers_truncated": False,
                }
                timer = threading.Timer(max(0.0, window_s), self._close, args=(key,))
                timer.daemon = True
                win["timer"] = timer
                self._windows[key] = win
                timer.start()
            first = dict(details or {})
            first["coalesce_window_s"] = window_s
            self._submit(action, client_id, resource, first)
            return True
        except Exception as exc:
            logger.error("Audit coalescing failed for %s: %s", action, exc)
            return False

    @classmethod
    def _note_peer(cls, win: dict[str, Any], client_id: str | None) -> None:
        """Remember a distinct client of a window, up to ``MAX_PEERS`` (the caller holds the lock)."""
        if not client_id or client_id in win["peers"]:
            return
        if len(win["peers"]) < cls.MAX_PEERS:
            win["peers"].add(client_id)
        else:
            win["peers_truncated"] = True

    def _close(self, key: str) -> None:
        with self._lock:
            win = self._windows.pop(key, None)
        if win is None:
            return
        win["timer"].cancel()
        if win["suppressed"]:
            self._submit(
                win["action"] + ".coalesced",
                win["client_id"],
                win["resource"],
                {
                    "suppressed": win["suppressed"],
                    "window_s": win["window_s"],
                    "of_action": win["action"],
                    "peers": sorted(win["peers"]),
                    "peers_truncated": win["peers_truncated"],
                },
            )

    def flush(self, timeout: float = 5.0) -> None:
        """Close every open window now and wait for queued writes (shutdown / tests)."""
        with self._lock:
            keys = list(self._windows)
        for key in keys:
            self._close(key)
        try:
            self._executor.submit(lambda: None).result(timeout=timeout)
        except Exception as exc:
            logger.error("Audit coalescer flush failed: %s", exc)


class PersistenceLayer:
    """SQLite-based persistence layer for LawnBerry Pi v2."""

    SCHEMA_VERSION = 4

    MIGRATIONS = [
        Migration(
            version=1,
            description="Initial schema",
            sql="""
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY,
                config_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS planning_jobs (
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
            
            CREATE TABLE IF NOT EXISTS telemetry_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_json TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS hardware_telemetry_streams (
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
            
            CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON hardware_telemetry_streams(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_telemetry_component ON hardware_telemetry_streams(component_id);
            
            CREATE TABLE IF NOT EXISTS map_zones (
                id TEXT PRIMARY KEY,
                name TEXT,
                polygon_json TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                exclusion_zone BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Insert initial schema version
            INSERT OR REPLACE INTO schema_version (version) VALUES (1);
            """,
        ),
        Migration(
            version=2,
            description="Add audit_logs table",
            sql="""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                details_json TEXT
            );

            INSERT OR REPLACE INTO schema_version (version) VALUES (2);
            """,
        ),
        Migration(
            version=3,
            description="Ensure telemetry streams table exists",
            sql="""
            CREATE TABLE IF NOT EXISTS hardware_telemetry_streams (
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
            CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON hardware_telemetry_streams(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_telemetry_component ON hardware_telemetry_streams(component_id);
            INSERT OR REPLACE INTO schema_version (version) VALUES (3);
            """,
        ),
        Migration(
            version=4,
            description="Add map_config table for map configuration storage",
            sql="""
            CREATE TABLE IF NOT EXISTS map_config (
                id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            INSERT OR REPLACE INTO schema_version (version) VALUES (4);
            """,
        ),
    ]

    def __init__(self, db_path: str = "data/lawnberry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database and run migrations."""
        with self.get_connection() as conn:
            # Get current schema version
            current_version = self._get_schema_version(conn)

            # Apply migrations
            for migration in self.MIGRATIONS:
                if migration.version > current_version:
                    logger.info(f"Applying migration {migration.version}: {migration.description}")
                    conn.executescript(migration.sql)
                    conn.commit()

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get current database schema version."""
        try:
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else 0
        except sqlite3.OperationalError:
            return 0

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # System Configuration
    def save_system_config(self, config: dict[str, Any]) -> None:
        """Save system configuration to database."""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_config (id, config_json, updated_at) VALUES (1, ?, ?)",
                (json.dumps(config), datetime.now(UTC)),
            )
            conn.commit()

    def load_system_config(self) -> dict[str, Any] | None:
        """Load system configuration from database."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT config_json FROM system_config WHERE id = 1")
            result = cursor.fetchone()
            if result:
                return json.loads(result["config_json"])
            return None

    # Planning Jobs
    def save_planning_job(self, job_data: dict[str, Any]) -> None:
        """Save planning job to database."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO planning_jobs 
                (id, name, schedule, zones_json, priority, enabled, created_at, last_run, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    job_data["id"],
                    job_data["name"],
                    job_data["schedule"],
                    json.dumps(job_data["zones"]),
                    job_data.get("priority", 1),
                    job_data.get("enabled", True),
                    job_data.get("created_at"),
                    job_data.get("last_run"),
                    job_data.get("status", "pending"),
                ),
            )
            conn.commit()

    def load_planning_jobs(self) -> list[dict[str, Any]]:
        """Load all planning jobs from database."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM planning_jobs ORDER BY created_at")
            jobs = []
            for row in cursor.fetchall():
                job = dict(row)
                job["zones"] = json.loads(job["zones_json"])
                del job["zones_json"]
                jobs.append(job)
            return jobs

    def delete_planning_job(self, job_id: str) -> bool:
        """Delete planning job from database."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM planning_jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    # Map Zones
    def save_map_zones(self, zones: list[dict[str, Any]]) -> None:
        """Save map zones to database."""
        with self.get_connection() as conn:
            # Clear existing zones
            conn.execute("DELETE FROM map_zones")

            # Insert new zones
            for zone in zones:
                conn.execute(
                    """
                    INSERT INTO map_zones (id, name, polygon_json, priority, exclusion_zone)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        zone["id"],
                        zone.get("name"),
                        json.dumps(zone["polygon"]),
                        zone.get("priority", 0),
                        zone.get("exclusion_zone", False),
                    ),
                )
            conn.commit()

    def load_map_zones(self) -> list[dict[str, Any]]:
        """Load map zones from database."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM map_zones ORDER BY priority DESC")
            zones = []
            for row in cursor.fetchall():
                zone = dict(row)
                zone["polygon"] = json.loads(zone["polygon_json"])
                del zone["polygon_json"]
                zones.append(zone)
            return zones

    # Telemetry History
    def save_telemetry_snapshot(self, data: dict[str, Any]) -> None:
        """Save telemetry snapshot for historical analysis."""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO telemetry_snapshots (timestamp, data_json) VALUES (?, ?)",
                (datetime.now(UTC), json.dumps(data)),
            )
            conn.commit()

    def load_telemetry_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Load recent telemetry history."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            snapshots = []
            for row in cursor.fetchall():
                snapshot = {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "data": json.loads(row["data_json"]),
                }
                snapshots.append(snapshot)
            return snapshots

    def cleanup_old_telemetry(self, days_to_keep: int = 7) -> int:
        """Clean up old telemetry data to manage disk space."""
        cutoff = datetime.now(UTC).timestamp() - (days_to_keep * 24 * 3600)
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM telemetry_snapshots WHERE timestamp < datetime(?, 'unixepoch')",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    # Map Configuration
    async def save_map_configuration(self, config_id: str, config_json: str) -> None:
        """Save map configuration to database."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO map_config (id, config_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (config_id, config_json, datetime.now(UTC)),
            )
            conn.commit()
            logger.info(f"Saved map configuration {config_id} to persistence")

    async def load_map_configuration(self, config_id: str) -> str | None:
        """Load map configuration from database."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT config_json FROM map_config WHERE id = ?",
                (config_id,),
            )
            result = cursor.fetchone()
            if result:
                logger.info(f"Loaded map configuration {config_id} from persistence")
                return result["config_json"]
            return None

    # Audit Logs
    def add_audit_log(
        self,
        action: str,
        client_id: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_logs (client_id, action, resource, details_json) VALUES (?, ?, ?, ?)",
                (client_id, action, resource, json.dumps(details or {})),
            )
            conn.commit()

    def add_audit_log_coalesced(
        self,
        action: str,
        client_id: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
        window_s: float = 5.0,
    ) -> bool:
        """Fire-and-forget audit write for flood-prone, unauthenticated paths.

        Never blocks the caller on SQLite and never raises. Returns True when this event was
        the first of its window (and so is being written); see :class:`AuditCoalescer`.
        """
        co = getattr(self, "_audit_coalescer", None)
        if co is None:
            co = AuditCoalescer(self)
            self._audit_coalescer = co
        return co.record(action, client_id, resource, details, window_s)

    def flush_audit_coalesced(self, timeout: float = 5.0) -> None:
        """Close open coalescing windows and wait for queued audit writes."""
        co = getattr(self, "_audit_coalescer", None)
        if co is not None:
            co.flush(timeout)

    def load_audit_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, client_id, action, resource, details_json FROM audit_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = []
            for r in cursor.fetchall():
                rows.append(
                    {
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "client_id": r["client_id"],
                        "action": r["action"],
                        "resource": r["resource"],
                        "details": json.loads(r["details_json"]) if r["details_json"] else {},
                    }
                )
            return rows

    # Hardware Telemetry Streams
    def save_telemetry_streams(self, streams: list[dict[str, Any]]) -> None:
        """Save hardware telemetry streams to database."""
        with self.get_connection() as conn:
            for stream in streams:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO hardware_telemetry_streams
                        (timestamp, component_id, value, status, latency_ms, stream_json, verification_artifact_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            stream.get("timestamp"),
                            stream.get("component_id"),
                            str(stream.get("value", "")),
                            stream.get("status"),
                            stream.get("latency_ms", 0.0),
                            json.dumps(stream),
                            stream.get("verification_artifact_id"),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Failed to save telemetry stream: {e}")
            conn.commit()

    def load_telemetry_streams(
        self,
        limit: int = 100,
        component_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load hardware telemetry streams from database with optional filters."""
        with self.get_connection() as conn:
            query = "SELECT * FROM hardware_telemetry_streams WHERE 1=1"
            params = []

            if component_id:
                query += " AND component_id = ?"
                params.append(component_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            streams = []
            for row in cursor.fetchall():
                stream = json.loads(row["stream_json"])
                stream["db_id"] = row["id"]
                streams.append(stream)
            return streams

    def compute_telemetry_latency_stats(
        self,
        component_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Compute latency statistics for telemetry streams."""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    component_id
                FROM hardware_telemetry_streams
                WHERE 1=1
            """
            params = []

            if component_id:
                query += " AND component_id = ?"
                params.append(component_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if not component_id:
                query += " GROUP BY component_id"

            cursor = conn.execute(query, params)

            if component_id:
                row = cursor.fetchone()
                if row:
                    return {
                        "component_id": component_id,
                        "count": row["count"],
                        "avg_latency_ms": row["avg_latency"],
                        "min_latency_ms": row["min_latency"],
                        "max_latency_ms": row["max_latency"],
                    }
                return {}
            else:
                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "component_id": row["component_id"],
                            "count": row["count"],
                            "avg_latency_ms": row["avg_latency"],
                            "min_latency_ms": row["min_latency"],
                            "max_latency_ms": row["max_latency"],
                        }
                    )
                return {"by_component": results}

    def cleanup_old_telemetry_streams(self, days_to_keep: int = 7) -> int:
        """Clean up old telemetry stream data to manage disk space."""
        cutoff = datetime.now(UTC).timestamp() - (days_to_keep * 24 * 3600)
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM hardware_telemetry_streams WHERE timestamp < datetime(?, 'unixepoch')",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    def export_telemetry_diagnostic(
        self,
        component_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Export telemetry diagnostic data including power metrics and status."""
        streams = self.load_telemetry_streams(
            limit=1000, component_id=component_id, start_time=start_time, end_time=end_time
        )

        stats = self.compute_telemetry_latency_stats(
            component_id=component_id, start_time=start_time, end_time=end_time
        )

        return {
            "export_timestamp": datetime.now(UTC).isoformat(),
            "filters": {
                "component_id": component_id,
                "start_time": start_time,
                "end_time": end_time,
            },
            "statistics": stats,
            "stream_count": len(streams),
            "streams": streams,
        }

    # Test helper: seed minimal simulated streams when SIM_MODE enabled
    def seed_simulated_streams(self, count: int = 10) -> None:
        if not is_simulation_mode():
            return
        now = datetime.now(UTC)
        rows = []
        for _i in range(count):
            ts = (now).isoformat()
            rows.append(
                {
                    "timestamp": ts,
                    "component_id": "power",
                    "value": {"voltage": 12.5, "percentage": 80},
                    "status": "healthy",
                    "latency_ms": 0.0,
                }
            )
        try:
            self.save_telemetry_streams(rows)
        except Exception:
            pass


# Global persistence instance
persistence = PersistenceLayer()
