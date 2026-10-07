"""Unit tests for the W1 capture service (session format + integrity gate).

Fake clock + fake sensor/tractor/frame sources: no hardware, no wall time.
The gate under test is ``capture_integrity.check_session`` on a real session
dir written by the real service.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.src.models.tractor_control import TractorCommand
from backend.src.services.capture_integrity import check_session
from backend.src.services.capture_service import CaptureError, CaptureService
from backend.src.services.tractor_service import TractorControlService

# 1 ms per clock call; every stream reads the clock once per record.
_STEP = 1_000_000


class FakeClock:
    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        self.t += _STEP
        return self.t


def _gps_reading() -> SimpleNamespace:
    return SimpleNamespace(
        latitude=40.0,
        longitude=-74.0,
        altitude=10.0,
        accuracy=0.03,
        satellites=18,
        hdop=0.5,
        speed=0.0,
        rtk_status="RTK_FIXED",
    )


def _imu_reading() -> SimpleNamespace:
    return SimpleNamespace(
        roll=0.1,
        pitch=0.2,
        yaw=3.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=9.8,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        calibration_status="stable",
    )


class FakeSensor:
    def __init__(self) -> None:
        self.gps = SimpleNamespace(last_reading=_gps_reading())
        self.imu = SimpleNamespace(last_reading=_imu_reading())


class FakeTractor:
    """Stands in for TractorControlService: only the on_change seam matters here."""

    def __init__(self) -> None:
        self.on_change = None


def _jpeg_frame(seq: int) -> dict:
    return {
        "data": base64.b64encode(b"\xff\xd8\xff\xd9").decode(),
        "metadata": {"sequence_number": seq, "width": 1280, "height": 720},
    }


async def _frame_source(n: int | None = None):
    i = 0
    while n is None or i < n:
        yield _jpeg_frame(i)
        i += 1
        await asyncio.sleep(0.002)


def _service(
    tmp_path: Path, *, frames=None, flush_every: int = 100, command_interval: float = 0.005
):
    clock = FakeClock()
    sensor = FakeSensor()
    tractor = FakeTractor()
    svc = CaptureService(
        data_dir=tmp_path,
        sensor_provider=lambda: sensor,
        tractor_provider=lambda: tractor,
        frame_source=(frames or (lambda: _frame_source())),
        clock=clock,
        gps_interval=0.01,
        imu_interval=0.005,
        command_interval=command_interval,
        flush_every=flush_every,
    )
    return svc, clock, sensor, tractor


@pytest.mark.asyncio
async def test_clean_session_passes_gate(tmp_path: Path) -> None:
    svc, _, _, _ = _service(tmp_path)
    await svc.start_session("s1")
    await asyncio.sleep(0.15)  # 150 ms of fake time
    report = await svc.stop_session()
    assert report.ok, report.problems
    session = tmp_path / "captures" / "s1"
    assert check_session(session).ok
    manifest = json.loads((session / "manifest.json").read_text())
    assert manifest["schema"] == "capture-v1"
    assert manifest["running"] is False
    assert manifest["integrity"]["ok"] is True
    for stream in ("gps", "imu", "commands", "camera"):
        assert manifest["records"][stream] >= 2, stream
    # camera rows index files that exist
    row = json.loads((session / "camera.jsonl").read_text().splitlines()[0])
    assert (session / row["path"]).exists()
    assert not (tmp_path / "captures" / "s1.rejected").exists()


@pytest.mark.asyncio
async def test_midstream_hole_lands_rejected(tmp_path: Path) -> None:
    """A stream that stalls mid-session leaves a gap the gate must catch."""
    svc, _, _, _ = _service(tmp_path)
    await svc.start_session("s2")
    await asyncio.sleep(0.02)
    # Stall the IMU task while the other streams keep consuming the clock:
    # its next row lands >0.05 s (fake time) after its last one.
    svc._tasks[1].cancel()
    await asyncio.sleep(0.3)
    svc._tasks[1] = asyncio.create_task(
        svc._poll_loop("imu", svc._sensor_provider(), svc._imu_interval, svc._imu_record)
    )
    await asyncio.sleep(0.02)
    report = await svc.stop_session()
    assert not report.ok
    assert any("gap" in p for p in report.problems), report.problems
    rejected = tmp_path / "captures" / "s2.rejected"
    assert rejected.is_dir()
    assert not (tmp_path / "captures" / "s2").exists()
    assert json.loads((rejected / "manifest.json").read_text())["integrity"]["ok"] is False


@pytest.mark.asyncio
async def test_tail_loss_is_bounded_and_undetected(tmp_path: Path) -> None:
    """Crash tail loss (<= flush_every rows) leaves NO trace: the gate only
    sees mid-stream holes. Assert the documented bound so the contract stays
    explicit — losing the last buffered rows is invisible by design."""
    svc, _, _, _ = _service(tmp_path, flush_every=10_000)
    await svc.start_session("s6")
    await asyncio.sleep(0.05)
    for t in svc._tasks:
        t.cancel()
    paths = {s: Path(w._fh.name) for s, w in svc._writers.items()}
    buffered = {s: w.records for s, w in svc._writers.items()}
    for w in svc._writers.values():
        w._fh.close()
    svc._writers = {}
    # Crash = buffered rows never reach disk; on restart the files are as-is.
    on_disk = {s: len(p.read_text().splitlines()) for s, p in paths.items()}
    assert all(0 <= buffered[s] - on_disk[s] <= 10_000 for s in on_disk)
    report = await svc.stop_session()
    assert report.ok, report.problems  # hole-free to the last flushed row


@pytest.mark.asyncio
async def test_teleop_change_row_has_values_and_source(tmp_path: Path) -> None:
    svc, _, _, tractor = _service(tmp_path)
    await svc.start_session("s3")
    await asyncio.sleep(0.02)
    tractor.on_change("levers", {"left": 0.4, "right": -0.1}, "teleop")
    await asyncio.sleep(0.02)
    report = await svc.stop_session()
    assert report.ok, report.problems
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "captures" / "s3" / "commands.jsonl").read_text().splitlines()
    ]
    change = [r for r in rows if r["kind"] == "change"]
    assert change, rows
    r = change[-1]
    assert r["left"] == 0.4 and r["right"] == -0.1
    assert r["source"] == "teleop"
    assert r["change"] == "levers"
    assert all(rows[i]["t_ns"] < rows[i + 1]["t_ns"] for i in range(len(rows) - 1))


@pytest.mark.asyncio
async def test_real_tractor_hook_feeds_capture(tmp_path: Path) -> None:
    """End-to-end with the real TractorControlService on_change seam (SIM servos)."""
    sensor = FakeSensor()
    tractor = TractorControlService(config={})
    svc = CaptureService(
        data_dir=tmp_path,
        sensor_provider=lambda: sensor,
        tractor_provider=lambda: tractor,
        frame_source=lambda: _frame_source(),
        clock=FakeClock(),
        gps_interval=0.01,
        imu_interval=0.005,
        command_interval=0.005,
    )
    await svc.start_session("s4")
    await tractor.apply(
        TractorCommand(left_lever=0.3, right_lever=0.5, throttle=0.2), source="teleop"
    )
    await asyncio.sleep(0.05)
    report = await svc.stop_session()
    assert report.ok, report.problems
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "captures" / "s4" / "commands.jsonl").read_text().splitlines()
    ]
    teleop = [r for r in rows if r.get("source") == "teleop"]
    assert teleop, rows
    kinds = {r["change"] for r in teleop if r["kind"] == "change"}
    assert {"throttle", "levers", "blade"} <= kinds
    # hook is detached again after stop
    assert tractor.on_change is None


@pytest.mark.asyncio
async def test_start_rejects_bad_names_and_collisions(tmp_path: Path) -> None:
    svc, _, _, _ = _service(tmp_path)
    with pytest.raises(CaptureError):
        await svc.start_session("../escape")
    with pytest.raises(CaptureError):
        await svc.start_session("")
    await svc.start_session("ok1")
    with pytest.raises(CaptureError):
        await svc.start_session("ok2")  # one session at a time
    await svc.stop_session()
    # The session above was rejected (instant stop -> <2 records) and renamed;
    # the freed name may be reused, but an existing dir under it may not.
    (tmp_path / "captures" / "ok1").mkdir(parents=True, exist_ok=True)
    with pytest.raises(CaptureError):
        await svc.start_session("ok1")


@pytest.mark.asyncio
async def test_status_reports_activity(tmp_path: Path) -> None:
    svc, _, _, _ = _service(tmp_path)
    st = svc.status()
    assert st["active"] is False and st["session"] is None
    await svc.start_session("s5")
    await asyncio.sleep(0.03)
    st = svc.status()
    assert st["active"] is True and st["session"] == "s5"
    assert st["records"]["imu"] >= 2
    await svc.stop_session()
    st = svc.status()
    assert st["active"] is False
    assert st["last_report"]["ok"] is True
