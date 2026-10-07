"""W1 capture logger: one session dir, four JSONL streams, integrity gate.

On-disk contract (``capture_integrity.check_session``): ``manifest.json`` plus
``gps.jsonl`` / ``imu.jsonl`` / ``commands.jsonl`` / ``camera.jsonl``; every
record carries ``t_ns`` from ONE monotonic clock; per-stream max gaps
(gps 0.5 s, imu 0.05 s, commands 0.5 s, camera 0.2 s); cross-stream start/end
skew <= 1 s; ``rtk_fixed`` >= 95% of GPS rows.

Design:
- One asyncio task per polled stream. GPS/IMU rows are sample-and-hold on the
  sensor interfaces' cached ``last_reading``: the telemetry loop already drives
  the serial reads, and a second poller would interleave NMEA on the shared
  port. ``t_ns`` is stamped when the record is *formed*, so a held row is
  honest — "at this instant the logger observed this state".
- ``commands.jsonl`` rows come from the tractor ``on_change`` hook (change
  rows) plus a heartbeat of the held state (the gap budget applies while the
  vehicle is stationary too).
- ``camera.jsonl`` indexes JPEGs pulled from lawnberry-camera.service's IPC
  frame stream (the constitution reserves ``/dev/video0`` to that unit).
- Buffered writers flush every ``flush_every`` rows and on stop: bounded loss
  (<= flush_every rows) on a Pi crash, which the gap gate then catches — that
  is the point of the gate.
- ``stop_session`` runs ``check_session`` immediately; a failing session is
  renamed to ``<name>.rejected`` (never deleted).

Sessions live on the Pi's own disk (``LAWNBERRY_DATA_DIR/captures``); finished
sessions are rsync-ed to the workshop data root afterwards — an offload step,
never a live dependency.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import os
import platform
import re
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .capture_integrity import IntegrityReport, check_session

logger = logging.getLogger(__name__)

DEFAULT_GPS_INTERVAL_S = 0.1
DEFAULT_IMU_INTERVAL_S = 0.02
DEFAULT_COMMAND_INTERVAL_S = 0.2
DEFAULT_FLUSH_EVERY = 100

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CaptureError(RuntimeError):
    """Session could not be started (bad name, already active, dir exists)."""


def _fix_of(reading: Any) -> str:
    """Map a GpsReading to the integrity contract's ``fix`` vocabulary."""
    status = (getattr(reading, "rtk_status", None) or "").upper()
    if "FIXED" in status:
        return "rtk_fixed"
    if "FLOAT" in status:
        return "rtk_float"
    if "DGPS" in status:
        return "dgps_fix"
    if (getattr(reading, "satellites", None) or 0) >= 6:
        return "gps_fix"
    return "no_fix"


class _JsonlWriter:
    """Append-only buffered JSONL writer with a per-stream monotonic clamp.

    ``check_session`` requires strictly increasing ``t_ns``; two records formed
    inside one clock tick are clamped to last+1 rather than rejected.
    """

    def __init__(self, path: Path, flush_every: int):
        self._fh = path.open("w", encoding="utf-8")
        self._flush_every = flush_every
        self._pending = 0
        self._last_t = -1
        self.records = 0

    def write(self, rec: dict[str, Any], t_ns: int) -> None:
        if t_ns <= self._last_t:
            t_ns = self._last_t + 1
        self._last_t = t_ns
        rec["t_ns"] = t_ns
        self._fh.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        self.records += 1
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        self._pending = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()


class CaptureService:
    """Records one teleop session at a time into the capture-integrity format.

    Providers (``sensor_provider``/``tractor_provider``) default to the live
    backend singletons and are resolved at ``start_session`` time, so
    constructing the service never touches hardware (SIM-safe). Tests pass
    fakes plus an injected ``clock``/``frame_source``.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        sensor_provider: Callable[[], Any] | None = None,
        tractor_provider: Callable[[], Any] | None = None,
        frame_source: Callable[[], AsyncIterator[dict[str, Any]]] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        gps_interval: float = DEFAULT_GPS_INTERVAL_S,
        imu_interval: float = DEFAULT_IMU_INTERVAL_S,
        command_interval: float = DEFAULT_COMMAND_INTERVAL_S,
        flush_every: int = DEFAULT_FLUSH_EVERY,
    ):
        base = data_dir or os.getenv("LAWNBERRY_DATA_DIR") or "/home/kp/lawnberry-data"
        self.data_dir = Path(base) / "captures"
        self._sensor_provider = sensor_provider
        self._tractor_provider = tractor_provider
        self._frame_source = frame_source
        self._clock = clock
        self._gps_interval = gps_interval
        self._imu_interval = imu_interval
        self._command_interval = command_interval
        self._flush_every = flush_every

        self._dir: Path | None = None
        self._name: str | None = None
        self._writers: dict[str, _JsonlWriter] = {}
        self._tasks: list[asyncio.Task] = []
        self._prev_on_change: Callable[..., None] | None = None
        self._tractor: Any = None
        self._cmd_state: dict[str, Any] = {}
        self._frame_seq = 0
        self._camera_error: str | None = None
        self._started_mono: int | None = None
        self._started_utc: datetime | None = None
        self._last_report: IntegrityReport | None = None

    # ----------------------------- lifecycle -----------------------------

    @staticmethod
    async def _resolve(provider: Callable[[], Any] | None) -> Any:
        if provider is None:
            return None
        result = provider()
        return await result if inspect.isawaitable(result) else result

    async def start_session(self, name: str) -> Path:
        if self._dir is not None:
            raise CaptureError(f"session already active: {self._name}")
        if not _NAME_RE.match(name):
            raise CaptureError(f"invalid session name {name!r} (want {_NAME_RE.pattern})")
        session_dir = self.data_dir / name
        if session_dir.exists():
            raise CaptureError(f"session dir already exists: {session_dir}")

        sensor = await self._resolve(self._sensor_provider)
        tractor = await self._resolve(self._tractor_provider)
        frames = self._frame_source or self._default_frame_source

        self._dir = session_dir
        self._name = name
        self._tractor = tractor
        self._camera_error = None
        self._frame_seq = 0
        self._cmd_state = {
            "left": 0.0,
            "right": 0.0,
            "throttle": 0.0,
            "blade": False,
            "source": "operator",
        }
        (session_dir / "frames").mkdir(parents=True, exist_ok=False)
        self._writers = {
            s: _JsonlWriter(session_dir / f"{s}.jsonl", self._flush_every)
            for s in ("gps", "imu", "commands", "camera")
        }
        self._started_mono = self._clock()
        self._started_utc = datetime.now(UTC)
        self._write_manifest(running=True, sensor=sensor, tractor=tractor)

        if tractor is not None:
            self._prev_on_change = getattr(tractor, "on_change", None)
            tractor.on_change = self._on_tractor_change

        self._tasks = [
            asyncio.create_task(
                self._poll_loop("gps", sensor, self._gps_interval, self._gps_record)
            ),
            asyncio.create_task(
                self._poll_loop("imu", sensor, self._imu_interval, self._imu_record)
            ),
            asyncio.create_task(self._command_loop()),
            asyncio.create_task(self._camera_loop(frames)),
        ]
        logger.info("capture session started: %s", session_dir)
        return session_dir

    async def stop_session(self) -> IntegrityReport:
        if self._dir is None:
            raise CaptureError("no active session")
        session_dir, name = self._dir, self._name
        assert name is not None

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._tractor is not None:
            self._tractor.on_change = self._prev_on_change
            self._prev_on_change = None
        counts = {s: w.records for s, w in self._writers.items()}
        for w in self._writers.values():
            w.close()
        self._writers = {}

        report = check_session(session_dir)
        self._write_manifest(running=False, report=report, counts=counts)
        self._last_report = report
        self._dir = None
        self._name = None
        self._tractor = None
        self._started_mono = None
        self._started_utc = None

        if not report.ok:
            rejected = session_dir.with_name(f"{name}.rejected")
            if rejected.exists():
                rejected = session_dir.with_name(
                    f"{name}.rejected.{datetime.now(UTC):%Y%m%dT%H%M%S}"
                )
            session_dir.rename(rejected)
            logger.warning(
                "capture session %s REJECTED (%s) -> %s",
                name,
                "; ".join(report.problems),
                rejected.name,
            )
        else:
            logger.info("capture session %s ok: %s", name, counts)
        return report

    def status(self) -> dict[str, Any]:
        return {
            "active": self._dir is not None,
            "session": self._name,
            "dir": str(self._dir) if self._dir else None,
            "records": {s: w.records for s, w in self._writers.items()},
            "camera_error": self._camera_error,
            "last_report": (
                {
                    "ok": self._last_report.ok,
                    "problems": self._last_report.problems,
                    "stats": self._last_report.stats,
                }
                if self._last_report
                else None
            ),
        }

    # ------------------------------ streams ------------------------------

    def _on_tractor_change(self, kind: str, values: dict[str, Any], source: str) -> None:
        """TractorControlService.on_change hook: change row + held-state update."""
        for key in ("left", "right", "throttle", "blade"):
            if key in values:
                self._cmd_state[key] = values[key]
        self._cmd_state["source"] = source
        if self._dir is not None and "commands" in self._writers:
            self._writers["commands"].write(
                {**self._cmd_state, "kind": "change", "change": kind}, self._clock()
            )

    async def _poll_loop(
        self, stream: str, sensor: Any, interval: float, form: Callable[[Any], dict[str, Any]]
    ) -> None:
        while True:
            t = self._clock()
            try:
                self._writers[stream].write(form(sensor), t)
            except KeyError:  # stopped between iterations
                return
            await asyncio.sleep(interval)

    @staticmethod
    def _gps_record(sensor: Any) -> dict[str, Any]:
        r = getattr(getattr(sensor, "gps", None), "last_reading", None)
        if r is None:
            return {
                "lat": None,
                "lon": None,
                "alt": None,
                "sats": None,
                "hdop": None,
                "speed": None,
                "fix": "no_fix",
            }
        return {
            "lat": r.latitude,
            "lon": r.longitude,
            "alt": r.altitude,
            "acc": r.accuracy,
            "sats": r.satellites,
            "hdop": r.hdop,
            "speed": getattr(r, "speed", None),
            "fix": _fix_of(r),
        }

    @staticmethod
    def _imu_record(sensor: Any) -> dict[str, Any]:
        r = getattr(getattr(sensor, "imu", None), "last_reading", None)
        if r is None:
            return {"roll": None, "pitch": None, "yaw": None, "cal": "unknown"}
        return {
            "roll": r.roll,
            "pitch": r.pitch,
            "yaw": r.yaw,
            "ax": r.accel_x,
            "ay": r.accel_y,
            "az": r.accel_z,
            "gx": r.gyro_x,
            "gy": r.gyro_y,
            "gz": r.gyro_z,
            "cal": r.calibration_status or "unknown",
        }

    async def _command_loop(self) -> None:
        """Heartbeat of the held command state (change rows come from the hook)."""
        while True:
            t = self._clock()
            try:
                self._writers["commands"].write({**self._cmd_state, "kind": "hold"}, t)
            except KeyError:
                return
            await asyncio.sleep(self._command_interval)

    def _default_frame_source(self) -> AsyncIterator[dict[str, Any]]:
        from .camera_ipc_client import frame_stream

        return frame_stream()

    async def _camera_loop(self, frames: Callable[[], AsyncIterator[dict[str, Any]]]) -> None:
        assert self._dir is not None
        frames_dir = self._dir / "frames"
        try:
            async for frame in frames():
                t = self._clock()
                b64 = frame.get("data")
                if not b64:
                    continue
                try:
                    raw = base64.b64decode(b64, validate=True)
                except (binascii.Error, ValueError):
                    continue
                seq = self._frame_seq
                self._frame_seq += 1
                (frames_dir / f"{seq:06d}.jpg").write_bytes(raw)
                meta = frame.get("metadata") or {}
                self._writers["camera"].write(
                    {
                        "path": f"frames/{seq:06d}.jpg",
                        "seq": meta.get("sequence_number"),
                        "w": meta.get("width"),
                        "h": meta.get("height"),
                        "bytes": len(raw),
                    },
                    t,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._camera_error = f"{type(exc).__name__}: {exc}"
            logger.error("capture camera stream died: %s", self._camera_error)

    # ------------------------------ manifest ------------------------------

    def _write_manifest(
        self,
        *,
        running: bool,
        report: IntegrityReport | None = None,
        sensor: Any = None,
        tractor: Any = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        assert self._dir is not None
        manifest: dict[str, Any] = {
            "schema": "capture-v1",
            "session": self._name,
            "started_utc": self._started_utc.isoformat() if self._started_utc else None,
            "ended_utc": None if running else datetime.now(UTC).isoformat(),
            "running": running,
            "host": platform.node(),
            "clock": "monotonic_ns",
            "clock_epoch_offset_ns": time.time_ns() - self._clock(),
            "intervals_s": {
                "gps": self._gps_interval,
                "imu": self._imu_interval,
                "commands": self._command_interval,
                "camera": None,  # every IPC frame, no polling interval
            },
            "sampling": "gps/imu rows are sample-and-hold of the telemetry loop's "
            "last_reading, stamped at record formation",
            "sources": {
                "gps": type(getattr(sensor, "gps", None)).__name__ if sensor else None,
                "camera": "injected" if self._frame_source else "lawnberry-camera IPC",
                "commands": ("TractorControlService.on_change + heartbeat" if tractor else None),
            },
            "records": counts
            if counts is not None
            else {s: w.records for s, w in self._writers.items()},
            "camera_error": self._camera_error,
        }
        if self._started_mono is not None:
            manifest["duration_s"] = round((self._clock() - self._started_mono) / 1e9, 3)
        if report is not None:
            manifest["integrity"] = {
                "ok": report.ok,
                "problems": report.problems,
                "stats": report.stats,
            }
        tmp = self._dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._dir / "manifest.json")


_capture_service: CaptureService | None = None


def get_capture_service() -> CaptureService:
    """Live singleton wired to the backend's sensor/tractor/camera seams."""
    global _capture_service
    if _capture_service is None:
        from .tractor_service import get_tractor_service
        from .websocket_hub import websocket_hub

        async def _sensors() -> Any:
            # The telemetry loop creates the manager on first poll; a capture
            # started before that creates it here instead of failing.
            if websocket_hub._sensor_manager is None:
                await websocket_hub._ensure_sensor_manager()
            return websocket_hub._sensor_manager

        _capture_service = CaptureService(
            sensor_provider=_sensors,
            tractor_provider=get_tractor_service,
        )
    return _capture_service


__all__ = ["CaptureError", "CaptureService", "get_capture_service"]
