"""The E-stop clear endpoints and the E-stop response tell the truth (audit SEC-07, 11, 12, 13).

SEC-07: the idempotent branch of ``/control/emergency_clear`` skipped the tractor-latch check and
answered ``EMERGENCY_CLEARED`` while the tractor stayed E-stopped, and the helper swallowed every
exception. SEC-11: clears left no audit record and discarded ``reason``. SEC-12: a clear that
arrived while a stop was still driving the outputs undid the latch the stop had just set.
SEC-13: ``/control/emergency`` said ``accepted: true, result: "accepted"`` for an incomplete stop.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from _tractor_fakes import idle_tractor, mowing_tractor

from backend.src.api import rest as rest_module
from backend.src.core.globals import _client_emergency, _safety_state
from backend.src.core.persistence import persistence
from backend.src.services import emergency_stop_service as ess

CLEAR = {"confirmation": True}


class FakeRoboHAT:
    def __init__(self, *, drive="acked", delay_s=0.0) -> None:
        self.running = True
        self.status = SimpleNamespace(serial_connected=True)
        self.drive, self.delay_s = drive, delay_s
        self.latched = False
        self.clear_calls = 0

    async def emergency_stop_detail(self):
        self.latched = True
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return {"drive": self.drive, "blade": "acked", "usb_control": True, "error": None}

    async def park(self):
        return {"drive": "acked", "blade": "acked", "error": None}

    @property
    def emergency_latched(self) -> bool:
        return self.latched

    async def clear_emergency(self) -> bool:
        self.clear_calls += 1
        self.latched = False
        return True


class FakeBlade:
    def __init__(self) -> None:
        self.latched = False

    async def emergency_stop(self):
        self.latched = True
        return {"latched": True, "off": True, "initialized": True, "hardware_backed": True}

    async def park(self):
        return {"latched": False, "off": True, "initialized": True, "hardware_backed": True}

    @property
    def emergency_latched(self) -> bool:
        return self.latched

    async def clear_emergency(self) -> None:
        self.latched = False


@pytest.fixture(autouse=True)
def _isolate_global_estop_state():
    _safety_state["emergency_stop_active"] = False
    yield
    _safety_state["emergency_stop_active"] = False
    rest_module._emergency_until = 0.0
    _client_emergency.clear()


def wire(monkeypatch, tractor=None, robohat=None, blade=None) -> None:
    base = "backend.src.services"
    monkeypatch.setattr(f"{base}.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(f"{base}.robohat_service.get_robohat_service", lambda: robohat)
    monkeypatch.setattr(f"{base}.blade_service.get_blade_service", lambda: blade)


def _client():
    from backend.src.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _audit(action: str, needle: str) -> dict | None:
    for row in persistence.load_audit_logs(limit=200):
        if row["action"] == action and needle in str(row["details"]):
            return row
    return None


# ----------------------------------- SEC-07 -----------------------------------


async def test_the_idempotent_clear_does_not_say_cleared_while_the_tractor_is_latched(monkeypatch):
    tractor, _pca = mowing_tractor()
    wire(monkeypatch, tractor, FakeRoboHAT(), FakeBlade())
    await (
        tractor.emergency_stop()
    )  # what POST /tractor/emergency-stop does: _safety_state untouched
    assert _safety_state["emergency_stop_active"] is False

    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency_clear", json=CLEAR)

    body = resp.json()
    assert resp.status_code == 200 and body["idempotent"] is True
    assert body["status"] == "EMERGENCY_STILL_LATCHED", body
    assert body["still_latched"] == ["tractor_emergency_stop"]
    assert tractor.state.emergency_stop_active is True
    await tractor.shutdown()


async def test_the_idempotent_clear_with_nothing_latched_is_still_cleared(monkeypatch):
    tractor, _pca = mowing_tractor()
    wire(monkeypatch, tractor, FakeRoboHAT(), FakeBlade())
    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency_clear", json=CLEAR)
    assert resp.json() == {"status": "EMERGENCY_CLEARED", "idempotent": True}
    await tractor.shutdown()


async def test_an_unreadable_tractor_state_is_reported_as_latched_never_as_clear(monkeypatch):
    def broken():
        raise RuntimeError("tractor state unreadable")

    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", broken)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)
    monkeypatch.setattr("backend.src.services.blade_service.get_blade_service", lambda: FakeBlade())

    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency_clear", json=CLEAR)

    body = resp.json()
    assert body["status"] == "EMERGENCY_STILL_LATCHED"
    assert body["still_latched"][0].startswith("tractor_state_unreadable")
    assert "RuntimeError" in body["still_latched"][0]


# ----------------------------------- SEC-11 -----------------------------------


async def test_a_general_clear_is_audited_with_its_reason(monkeypatch):
    wire(monkeypatch, idle_tractor(), FakeRoboHAT(), FakeBlade())
    _safety_state["emergency_stop_active"] = True

    async with _client() as client:
        resp = await client.post(
            "/api/v2/control/emergency_clear",
            json={"confirmation": True, "reason": "checked the yard, mower is upright SEC11a"},
        )

    assert resp.json()["status"] == "EMERGENCY_CLEARED"
    row = _audit("control.emergency_clear", "SEC11a")
    assert row is not None, "an E-stop clear must leave an audit record"
    assert row["details"]["outcome"] == "cleared"
    assert row["details"]["reason"] == "checked the yard, mower is upright SEC11a"


async def test_a_tractor_clear_is_audited_with_its_reason(monkeypatch):
    from backend.src.api.routers import tractor as tractor_router

    tractor, _pca = mowing_tractor()
    monkeypatch.setattr(tractor_router, "get_tractor_service", lambda: tractor)
    await tractor.emergency_stop()

    async with _client() as client:
        resp = await client.post(
            "/api/v2/tractor/clear-emergency", json={"reason": "blade guard refitted SEC11b"}
        )
        bare = await client.post("/api/v2/tractor/clear-emergency")  # no body: still allowed

    assert resp.json()["status"] == "ok" and bare.status_code == 200
    row = _audit("tractor.clear_emergency", "SEC11b")
    assert row is not None and row["details"]["reason"] == "blade guard refitted SEC11b"
    assert row["details"]["result"]["status"] == "ok"
    await tractor.shutdown()


# ----------------------------------- SEC-12 -----------------------------------


async def test_a_clear_arriving_mid_stop_is_refused_and_does_not_undo_the_latch(monkeypatch):
    robohat = FakeRoboHAT(delay_s=0.5)  # the stop is mid-flight for half a second
    wire(monkeypatch, idle_tractor(), robohat, FakeBlade())

    async with _client() as client:
        stop = asyncio.create_task(client.post("/api/v2/control/emergency-stop"))
        await asyncio.sleep(0.15)
        assert ess.stop_in_flight()
        clear = await client.post("/api/v2/control/emergency_clear", json=CLEAR)
        report = (await stop).json()

    assert clear.status_code == 409 and clear.json()["status"] == "EMERGENCY_STOP_IN_PROGRESS"
    assert _safety_state["emergency_stop_active"] is True, "the stop's latch must survive"
    assert robohat.clear_calls == 0 and robohat.latched is True, "its hold must not be released"
    assert report["emergency_stop_active"] is True
    assert _audit("control.emergency_clear", "refused") is not None


async def test_a_stop_that_starts_while_the_clear_runs_wins(monkeypatch):
    robohat, blade = FakeRoboHAT(), FakeBlade()

    async def clear_that_races_a_stop():
        await ess.execute_emergency_stop(source="race")  # a stop lands mid-clear

    blade.clear_emergency = clear_that_races_a_stop  # type: ignore[method-assign]
    wire(monkeypatch, idle_tractor(), robohat, blade)
    _safety_state["emergency_stop_active"] = True

    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency_clear", json=CLEAR)

    assert resp.status_code == 409
    assert _safety_state["emergency_stop_active"] is True
    assert robohat.clear_calls == 0, "the remaining releases must be abandoned"
    assert robohat.latched is True


async def test_the_tractor_refuses_a_clear_while_its_own_stop_is_running():
    svc, pca = mowing_tractor()
    release = pca.stall(0, 1950)
    cmd = asyncio.create_task(svc.set_levers(0.9, 0.9, source="operator"))
    await asyncio.wait_for(pca.stalled.wait(), 2)
    stop = asyncio.create_task(svc.emergency_stop())
    await asyncio.sleep(0.15)  # waiting for the lock to re-assert neutral

    refused = await svc.clear_emergency()

    assert refused["status"] == "rejected" and "in progress" in refused["reason"]
    assert svc.state.emergency_stop_active is True and svc.stop_in_flight is True
    release.set()
    await asyncio.wait_for(stop, 3)
    await cmd
    assert (await svc.clear_emergency())["status"] == "ok"
    await svc.shutdown()


# ----------------------------------- SEC-13 -----------------------------------


async def test_the_v2_estop_response_does_not_say_accepted_for_an_incomplete_stop(monkeypatch):
    wire(monkeypatch, idle_tractor(), FakeRoboHAT(drive="unacked"), FakeBlade())
    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency", json={"session_id": "x"})

    body = resp.json()
    assert resp.status_code == 202, "the latch is set: the HTTP answer stays 202"
    assert body["accepted"] is False and body["result"] == "incomplete"
    assert body["status_reason"] == "EMERGENCY_STOP_INCOMPLETE"
    assert body["emergency_stop"]["complete"] is False


async def test_the_v2_estop_response_still_says_accepted_for_a_confirmed_stop(monkeypatch):
    wire(monkeypatch, idle_tractor(), FakeRoboHAT(), FakeBlade())
    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency", json={"session_id": "x"})

    body = resp.json()
    assert body["accepted"] is True and body["result"] == "accepted"
    assert body["status_reason"] == "EMERGENCY_STOP_TRIGGERED"
