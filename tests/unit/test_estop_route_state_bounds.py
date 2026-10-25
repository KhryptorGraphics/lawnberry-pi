"""The limiter-exempt E-stop route keeps bounded state and a bounded audit trail (SEC20-03)."""

from __future__ import annotations

import asyncio
import threading
import time as real_time

import httpx
import pytest

from backend.src.api import rest as rest_module
from backend.src.core.globals import _blade_state, _client_emergency, _safety_state
from backend.src.core.persistence import PersistenceLayer

CAP = 256
KEY_BOUND = 64
REPORT = {
    "emergency_stop_active": True,
    "motors_stopped": True,
    "blade_disabled": True,
    "complete": True,
    "subsystems": {},
    "faults": [],
}


class _Clock:
    def __init__(self) -> None:
        self.now = real_time.time()

    def __getattr__(self, name):  # everything but time() is the real module
        return getattr(real_time, name)

    def time(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    layer = PersistenceLayer(db_path=str(tmp_path / "audit.db"))
    monkeypatch.setattr(rest_module, "persistence", layer)
    _safety_state["emergency_stop_active"] = False
    rest_module._emergency_until = 0.0
    _client_emergency.clear()
    yield layer
    layer.flush_audit_coalesced()
    _safety_state["emergency_stop_active"] = False
    _blade_state["active"] = False
    rest_module._emergency_until = 0.0
    _client_emergency.clear()


@pytest.fixture
def stop_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def fake_execute(source: str = "api"):
        calls.append(source)
        return dict(REPORT)

    monkeypatch.setattr(
        "backend.src.services.emergency_stop_service.execute_emergency_stop", fake_execute
    )
    return calls


def _client() -> httpx.AsyncClient:
    from backend.src.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _rows(layer: PersistenceLayer) -> list[dict]:
    layer.flush_audit_coalesced()
    return layer.load_audit_logs(limit=10_000)


# ------------------------------- client table bounds --------------------------------


async def test_unique_oversized_authorization_headers_leave_a_bounded_small_keyed_table(
    stop_calls,
):
    async with _client() as client:
        for i in range(1500):
            resp = await client.post(
                "/api/v2/control/emergency",
                headers={"Authorization": f"Bearer {i}-" + "A" * 8000},
            )
            assert resp.status_code == 202
    assert len(stop_calls) == 1500
    assert 0 < len(_client_emergency) <= CAP
    assert max(len(k) for k in _client_emergency) <= KEY_BOUND


async def test_headerless_requests_add_no_client_entries(stop_calls):
    async with _client() as client:
        for _ in range(100):
            assert (await client.post("/api/v2/control/emergency")).status_code == 202
    assert len(_client_emergency) == 0
    assert len(stop_calls) == 100


async def test_expired_entries_are_pruned_by_an_insert_without_any_lookup(monkeypatch, stop_calls):
    clock = _Clock()
    monkeypatch.setattr(rest_module, "time", clock)
    async with _client() as client:
        for i in range(20):
            await client.post("/api/v2/control/emergency", headers={"X-Client-Id": f"old-{i}"})
        assert len(_client_emergency) == 20
        clock.now += 5.0  # all expired
        await client.post("/api/v2/control/emergency", headers={"X-Client-Id": "fresh"})
    assert len(_client_emergency) == 1


async def test_the_oldest_entry_is_evicted_at_the_cap(monkeypatch, stop_calls):
    clock = _Clock()
    monkeypatch.setattr(rest_module, "time", clock)
    async with _client() as client:
        for i in range(CAP + 10):
            clock.now += 0.0001  # all still live
            await client.post("/api/v2/control/emergency", headers={"X-Client-Id": f"c-{i}"})
        assert len(_client_emergency) == CAP
        # the newest client is still blocked, the evicted oldest one is not
        blocked_new = await client.post(
            "/api/v2/control/drive",
            json={"mode": "arcade", "throttle": 0.1, "turn": 0.0},
            headers={"X-Client-Id": f"c-{CAP + 9}"},
        )
        rest_module._emergency_until = 0.0
        evicted = await client.post(
            "/api/v2/control/drive",
            json={"mode": "arcade", "throttle": 0.1, "turn": 0.0},
            headers={"X-Client-Id": "c-0"},
        )
    assert blocked_new.status_code == 403
    assert evicted.status_code == 200


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v2/control/drive", {"mode": "arcade", "throttle": 0.1, "turn": 0.0}),
        ("/api/v2/control/blade", {"active": True}),
    ],
)
async def test_followup_command_is_blocked_for_the_same_client_only(
    monkeypatch, stop_calls, path, payload
):
    clock = _Clock()
    monkeypatch.setattr(rest_module, "time", clock)
    async with _client() as client:
        await client.post("/api/v2/control/emergency", headers={"X-Client-Id": "alice"})
        rest_module._emergency_until = 0.0  # isolate the per-client flag from the global TTL
        mine = await client.post(path, json=payload, headers={"X-Client-Id": "alice"})
        other = await client.post(path, json=payload, headers={"X-Client-Id": "bob"})
        clock.now += 1.0
        later = await client.post(path, json=payload, headers={"X-Client-Id": "alice"})
    assert mine.status_code == 403
    assert other.status_code == 200
    assert later.status_code == 200


async def test_authorization_header_identifies_the_client_too(monkeypatch, stop_calls):
    async with _client() as client:
        await client.post(
            "/api/v2/control/emergency", headers={"Authorization": "Bearer token-a" + "z" * 5000}
        )
        rest_module._emergency_until = 0.0
        same = await client.post(
            "/api/v2/control/drive",
            json={"mode": "arcade", "throttle": 0.1, "turn": 0.0},
            headers={"Authorization": "Bearer token-a" + "z" * 5000},
        )
    assert same.status_code == 403


# ------------------------------- audit coalescing --------------------------------


async def test_500_anonymous_stops_run_500_times_but_write_a_handful_of_rows(_isolate, stop_calls):
    async with _client() as client:
        for _ in range(500):
            assert (await client.post("/api/v2/control/emergency")).status_code == 202
    assert len(stop_calls) == 500

    rows = [r for r in _rows(_isolate) if r["action"].startswith("control.emergency")]
    assert 1 <= len(rows) <= 4
    first = [r for r in rows if r["action"] == "control.emergency.triggered"]
    summary = [r for r in rows if r["action"] == "control.emergency.triggered.coalesced"]
    assert len(first) == 1
    suppressed = sum(r["details"]["suppressed"] for r in summary)
    assert len(first) + suppressed == 500


async def test_500_alias_stops_are_coalesced_and_accounted(_isolate, stop_calls):
    async with _client() as client:
        for _ in range(500):
            assert (await client.post("/api/v2/control/emergency-stop")).status_code == 200
    assert len(stop_calls) == 500
    rows = [r for r in _rows(_isolate) if r["action"].startswith("control.emergency_stop")]
    assert len(rows) <= 4
    total = sum(
        1 if r["action"] == "control.emergency_stop" else r["details"]["suppressed"] for r in rows
    )
    assert total == 500


async def test_first_stop_after_a_quiet_period_is_written(monkeypatch, _isolate, stop_calls):
    monkeypatch.setattr(rest_module, "ESTOP_AUDIT_WINDOW_S", 0.05)
    async with _client() as client:
        await client.post("/api/v2/control/emergency-stop")
        _rows(_isolate)
        await asyncio.sleep(0.15)  # quiet period: the window closes
        await client.post("/api/v2/control/emergency-stop")
    singles = [r for r in _rows(_isolate) if r["action"] == "control.emergency_stop"]
    assert len(singles) == 2


async def test_audit_write_is_off_the_event_loop_thread(monkeypatch, _isolate, stop_calls):
    writers: list[int] = []
    real_add = _isolate.add_audit_log

    def recording(*a, **k):
        writers.append(threading.get_ident())
        return real_add(*a, **k)

    monkeypatch.setattr(_isolate, "add_audit_log", recording)
    async with _client() as client:
        await client.post("/api/v2/control/emergency")
    _isolate.flush_audit_coalesced()
    assert writers, "the first stop must be audited"
    assert threading.get_ident() not in writers


async def test_failing_audit_store_does_not_change_the_response(monkeypatch, _isolate, stop_calls):
    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(_isolate, "add_audit_log", broken)
    async with _client() as client:
        a = await client.post("/api/v2/control/emergency")
        b = await client.post("/api/v2/control/emergency-stop")
    _isolate.flush_audit_coalesced()
    assert a.status_code == 202 and a.json()["accepted"] is True
    assert b.status_code == 200 and b.json()["complete"] is True
    assert _safety_state["emergency_stop_active"] is True


def test_coalescer_keys_are_capped(_isolate):
    from backend.src.core.persistence import AuditCoalescer

    for i in range(AuditCoalescer.MAX_KEYS * 3):
        _isolate.add_audit_log_coalesced(f"action-{i}", window_s=60.0)
    assert len(_isolate._audit_coalescer._windows) <= AuditCoalescer.MAX_KEYS
    _isolate.flush_audit_coalesced()
    rows = _isolate.load_audit_logs(limit=10_000)
    # nothing silently lost: every event is a row or counted in a summary
    total = sum(
        1 if not r["action"].endswith(".coalesced") else r["details"]["suppressed"] for r in rows
    )
    assert total == AuditCoalescer.MAX_KEYS * 3
