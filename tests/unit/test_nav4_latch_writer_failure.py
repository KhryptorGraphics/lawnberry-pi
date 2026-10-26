"""A latch writer that cannot take the job never stops an E-stop from stopping (review 3, NAV-01).

Every stop path queues its latch for the disk between the latch and the outputs it drives, and
``request_persist`` documents "never raises". It guarded the snapshot but not the thread-pool
``submit``: that raises ``RuntimeError`` when the worker thread cannot be started (the first stop of
a process is the first ``submit``; the unit runs with ``MemoryMax=512M``) or when the executor is
shut down. The exception left ``TractorControlService.emergency_stop`` after the latch, the
revoke and the cancelled deadman and BEFORE the PTO relay, the starter relay, the levers and the
throttle; ``execute_emergency_stop`` had no ``try`` around it at all, so no stop path (tractor,
RoboHAT, IBT-4) was even scheduled. The machine kept running under a stop that said it was latched.

Faults are injected at the writer (an executor whose ``submit`` raises, and a real executor whose
thread refuses to start); the tractor is the real service over the stallable PCA9685.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from _nav_rig import live_sensors, real_lifespan
from _tractor_fakes import NEUTRAL, mowing_tractor

from backend.src.core import globals as core_globals
from backend.src.safety import latch_store, stop_events
from backend.src.services import emergency_stop_service as ess


class DeadWriter:
    """An executor that cannot take a job (no thread can be started, or it was shut down)."""

    def submit(self, *_args, **_kwargs):
        raise RuntimeError("can't start new thread")


@pytest.fixture
def dead_writer(monkeypatch) -> DeadWriter:
    writer = DeadWriter()
    monkeypatch.setattr(latch_store, "_writer", writer)
    return writer


@pytest.fixture
def writer_threads(monkeypatch):
    """A fresh, cold writer whose thread can be made to fail to start: ``(executor, switch)``.

    ``switch["exhausted"]`` makes ``Thread.start`` raise for the writer's own threads only (named
    ``latch-store_N``), exactly what a process out of threads sees; everything else keeps working.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="latch-store")
    monkeypatch.setattr(latch_store, "_writer", executor)
    switch = {"exhausted": False}
    real_start = threading.Thread.start

    def start(self, *args, **kwargs):
        if switch["exhausted"] and self.name.startswith("latch-store"):
            raise RuntimeError("can't start new thread")
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", start)
    yield executor, switch
    switch["exhausted"] = False
    executor.shutdown(wait=True)


async def _mowing(tractor) -> None:
    assert (await tractor.engage_blade(True, source="operator"))["status"] == "ok"
    moved = await tractor.set_levers(0.6, 0.6, source="operator", duration_ms=8000)
    assert moved["status"] == "ok" and tractor.state.blade_engaged and tractor.state.moving


# --------------------------------------------------------------------------- request_persist


async def test_request_persist_reports_a_writer_that_cannot_take_the_job_and_does_not_raise(
    dead_writer,
):
    pending = latch_store.request_persist()  # used to raise RuntimeError out of submit()

    result = pending.result(timeout=1)
    assert result.ok is False and "can't start new thread" in result.error
    assert "can't start new thread" in (latch_store.last_error() or ""), "status shows it"
    awaited = await latch_store.wait(pending)
    assert awaited.ok is False and awaited.error == result.error


@pytest.mark.parametrize("failing", ["latch_path", "collect_live"])
async def test_request_persist_reports_a_snapshot_or_path_that_fails_and_does_not_raise(
    monkeypatch, failing
):
    def boom(*_a, **_k):
        raise OSError(f"{failing} exploded")

    monkeypatch.setattr(latch_store, failing, boom)

    pending = latch_store.request_persist()  # latch_path() used to sit outside every guard

    result = pending.result(timeout=1)
    assert result.ok is False and f"{failing} exploded" in result.error


async def test_warm_up_reports_a_writer_that_cannot_start_and_does_not_raise(dead_writer):
    assert latch_store.warm_up() is False
    assert "can't start new thread" in (latch_store.last_error() or "")


# ------------------------------------------------------------------------------ the stops


async def test_a_tractor_estop_stops_every_output_when_the_latch_writer_cannot_start(dead_writer):
    tractor, chip = mowing_tractor()
    await _mowing(tractor)

    result = await tractor.emergency_stop(source="api")  # used to raise before any output

    assert result["status"] == "emergency_stop" and result["complete"] is True
    assert result["latch_persisted"] is False and "can't start new thread" in result["latch_error"]
    assert tractor.state.emergency_stop_active is True and tractor.state.authorized is False
    assert tractor.blade_pto.state is False and tractor.state.blade_engaged is False
    assert chip.regs == NEUTRAL, "levers neutral and throttle idle on the wire"
    assert tractor._lever_deadline is None
    await tractor.shutdown()


async def test_the_orchestrator_drives_every_stop_path_when_the_latch_writer_cannot_start(
    monkeypatch, dead_writer
):
    tractor, chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    await _mowing(tractor)
    epoch = stop_events.epoch()

    report = await ess.execute_emergency_stop("api")  # used to raise before scheduling a path

    assert report["emergency_stop_active"] is True
    assert report["latch_persisted"] is False and report["latch_error"]
    assert report["subsystems"]["tractor"]["status"] in {"stopped", "simulated"}, report
    assert core_globals._safety_state["emergency_stop_active"] is True
    assert stop_events.epoch() > epoch, "every autonomous run still ends with the stop"
    assert tractor.state.emergency_stop_active and tractor.blade_pto.state is False
    assert chip.regs == NEUTRAL
    await tractor.shutdown()


async def test_a_tilt_latch_and_a_clear_work_without_a_latch_writer(dead_writer):
    tractor, _chip = mowing_tractor()
    await _mowing(tractor)

    latched = await tractor.latch_interlock("tilt", source="tilt_monitor")
    assert latched["status"] == "latched" and latched["latch_persisted"] is False
    assert tractor.blade_pto.state is False and tractor.state.interlock_latched

    cleared = await tractor.clear_emergency()
    assert cleared["status"] == "ok" and cleared["latch_persisted"] is False
    assert tractor.state.interlock_latched is False
    await tractor.shutdown()


async def test_the_relays_open_before_the_stop_event_and_the_latch_write(monkeypatch):
    """Nothing that can fail sits between the latch and the first output: the order is pinned.

    ``request_persist`` is made to blow up (it cannot, by contract; this is the regression guard
    for whoever edits the sequence between the latch and the outputs): the relays are already
    off when it does.
    """
    tractor, _chip = mowing_tractor()
    await _mowing(tractor)

    def boom(*_a, **_k):
        raise RuntimeError("the latch queue exploded")

    monkeypatch.setattr(latch_store, "request_persist", boom)
    try:
        await tractor.emergency_stop(source="api")
    except RuntimeError:
        pass

    assert tractor.state.emergency_stop_active is True
    assert tractor.blade_pto.state is False and tractor.state.blade_engaged is False
    assert tractor.starter.state is False
    await tractor.shutdown()


# ----------------------------------------------------------------- a thread that cannot start


async def test_a_cold_writer_that_cannot_start_a_thread_is_reported_and_the_stop_still_stops(
    writer_threads,
):
    _executor, switch = writer_threads
    switch["exhausted"] = True  # no warm-up ran: the first stop is the first submit
    tractor, chip = mowing_tractor()
    await _mowing(tractor)

    result = await tractor.emergency_stop(source="api")

    assert result["complete"] is True and result["latch_persisted"] is False
    assert tractor.blade_pto.state is False and chip.regs == NEUTRAL
    await tractor.shutdown()


async def test_the_lifespan_starts_the_latch_writer_before_anything_can_be_stopped(
    monkeypatch, writer_threads
):
    """After start-up the thread budget is gone: the stop still saves its latch (NAV-01).

    The writer's thread is made at start-up (``latch_store.warm_up``, called by the lifespan), so
    an E-stop never asks the exhausted process for a new one. A fresh process whose lifespan did
    not warm the writer would report ``latch_persisted: false`` here.
    """
    _executor, switch = writer_threads
    tractor, chip = mowing_tractor()
    async with real_lifespan(monkeypatch, tractor, live_sensors(2000.0)):
        switch["exhausted"] = True  # start-up is over; no thread can be made from now on
        await _mowing(tractor)

        report = await ess.execute_emergency_stop("api", tractor=tractor)

        assert report["latch_persisted"] is True, report["latch_error"]
        assert latch_store.load().has(latch_store.PROCESS_ESTOP)
        assert latch_store.load().has(latch_store.TRACTOR_ESTOP)
        assert tractor.blade_pto.state is False and chip.regs == NEUTRAL
