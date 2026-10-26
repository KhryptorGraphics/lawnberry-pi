"""E-stop and interlock latches survive a restart (audit SAF-16).

The latches were in-memory flags and the unit runs with ``Restart=always`` (10 s): a crash, an
out-of-memory kill or ``systemctl restart`` released every one of them silently with nobody at the
machine. They are now written durably (atomic, fsynced, 0600) and restored before anything can
actuate; only an authenticated explicit clear releases them, a corrupt file fails closed, and
authorization is never restored.
"""

from __future__ import annotations

import json
import os
import stat

import pytest
from _tractor_fakes import mowing_tractor
from test_tractor_actuation_watchdog import _wire_lifespan

import backend.src.main as main
from backend.src.core import globals as core_globals
from backend.src.safety import latch_store
from backend.src.services import emergency_stop_service as ess
from backend.src.services import robohat_service as robohat_module
from backend.src.services.tractor_service import TractorControlService


@pytest.fixture(autouse=True)
def quiet_process(monkeypatch):
    monkeypatch.setitem(core_globals._safety_state, "emergency_stop_active", False)
    monkeypatch.setattr(robohat_module, "robohat_service", None)


def _restarted() -> TractorControlService:
    """The tractor service of the process that came up after the crash: nothing latched."""
    fresh = TractorControlService(config={"enabled": True})
    assert fresh.state.emergency_stop_active is False
    return fresh


async def test_an_estop_is_written_durably_and_privately(monkeypatch):
    synced: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    tractor, _pca = mowing_tractor()
    await tractor.emergency_stop(source="api")

    path = latch_store.latch_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert latch_store.PROCESS_ESTOP not in json.loads(path.read_text())["latches"]
    assert latch_store.TRACTOR_ESTOP in json.loads(path.read_text())["latches"]
    assert len(synced) >= 2, "the file and its directory must both be fsynced"
    assert [p.name for p in path.parent.iterdir()] == [path.name], "no temp file left behind"
    await tractor.shutdown()


async def test_a_restart_restores_the_latch_and_only_an_explicit_clear_releases_it():
    old, _pca = mowing_tractor()
    await old.emergency_stop(source="watchdog")  # the stop the previous process was holding
    fresh = _restarted()

    found = latch_store.restore(tractor=fresh)
    assert found.has(latch_store.TRACTOR_ESTOP)
    assert fresh.state.emergency_stop_active is True
    assert fresh.state.authorized is False, "authorization is never restored"
    refused = await fresh.set_levers(0.5, 0.5, source="operator")
    assert refused["status"] == "rejected", "even the operator's drive waits for the clear"

    cleared = await fresh.clear_emergency()
    assert cleared["status"] == "ok" and cleared["latch_persisted"] is True
    assert not latch_store.latch_path().exists(), "a clear removes the file"
    assert not latch_store.restore(tractor=_restarted()), "and the next restart finds nothing"
    await old.shutdown()
    await fresh.shutdown()


async def test_a_tilt_latch_survives_a_restart_with_its_reason():
    old, _pca = mowing_tractor()
    await old.latch_interlock("tilt", source="tilt_monitor")
    fresh = _restarted()
    latch_store.restore(tractor=fresh)
    assert fresh.state.interlock_latched is True and fresh.state.latch_reason == "tilt"
    assert (await fresh.engage_blade(True, source="operator"))["status"] == "rejected"
    await old.shutdown()
    await fresh.shutdown()


async def test_a_corrupt_latch_file_fails_closed(monkeypatch):
    latch_store.latch_path().write_text("{ this is not json")
    found = latch_store.load()
    assert found.unreadable
    fresh = _restarted()
    latch_store.restore(tractor=fresh)
    assert fresh.state.emergency_stop_active is True
    assert core_globals._safety_state["emergency_stop_active"] is True
    await fresh.shutdown()


async def test_the_orchestrators_process_latch_is_persisted_and_restored():
    report = await ess.execute_emergency_stop("api")
    assert report["latch_persisted"] is True
    assert latch_store.PROCESS_ESTOP in json.loads(latch_store.latch_path().read_text())["latches"]

    core_globals._safety_state["emergency_stop_active"] = False  # the process restarted
    latch_store.restore(tractor=_restarted())
    assert core_globals._safety_state["emergency_stop_active"] is True


async def test_a_failed_write_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setenv(latch_store.ENV_FILE, "/proc/nonexistent/emergency_latch.json")
    tractor, _pca = mowing_tractor()
    result = await tractor.emergency_stop(source="api")
    assert result["latch_persisted"] is False and result["latch_error"]
    assert tractor.state.emergency_stop_active is True, "the in-memory latch holds regardless"
    await tractor.shutdown()


async def test_the_real_lifespan_restores_what_the_last_process_left_latched(monkeypatch):
    old, _pca = mowing_tractor()
    await old.emergency_stop(source="watchdog")
    fresh = _restarted()
    _wire_lifespan(monkeypatch, fresh)
    async with main.lifespan(main.app):
        assert fresh.state.emergency_stop_active is True
        assert fresh.state.authorized is False
    await old.shutdown()
