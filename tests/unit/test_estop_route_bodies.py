"""The E-stop POSTs latch for EVERY request body shape (audit SEC20-05, mutation survivor A15).

FastAPI validated ``body: Optional[dict]`` before the handler ran, so a form, text, JSON list /
string or invalid JSON answered 422 and never latched ``/control/emergency``. The stop must not
depend on the body: it latches first, then the body is parsed best-effort.
"""

from __future__ import annotations

import json

import httpx
import pytest

from backend.src.api import rest as rest_module
from backend.src.core.globals import _blade_state, _client_emergency, _safety_state

JSON = {"Content-Type": "application/json"}
BIG_OBJECT = json.dumps({"reason": "x" * 2_000_000}).encode()

SHAPES = {
    "form": dict(
        content=b"reason=stop", headers={"Content-Type": "application/x-www-form-urlencoded"}
    ),
    "text_plain": dict(content=b"STOP", headers={"Content-Type": "text/plain"}),
    "json_list": dict(content=b'["a", 1]', headers=JSON),
    "json_string": dict(content=b'"stop"', headers=JSON),
    "json_number": dict(content=b"42", headers=JSON),
    "invalid_json_ct_json": dict(content=b"{not json", headers=JSON),
    "invalid_json_no_ct": dict(content=b"{not json", headers={}),
    "empty_body": dict(content=b"", headers={}),
    "empty_object": dict(content=b"{}", headers=JSON),
    "json_null": dict(content=b"null", headers=JSON),
    "big_object_2mb": dict(content=BIG_OBJECT, headers=JSON),
    "gzip_claim_garbage": dict(
        content=b"\x00\x01garbage",
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
    ),
    "binary_garbage": dict(
        content=bytes(range(256)) * 8, headers={"Content-Type": "application/octet-stream"}
    ),
    "wrong_ct_valid_json": dict(content=b'{"reason": "x"}', headers={"Content-Type": "text/plain"}),
}

ROUTES = {
    "/api/v2/control/emergency": 202,
    "/api/v2/control/emergency-stop": 200,
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from backend.src.core.persistence import PersistenceLayer  # noqa: F401

    monkeypatch.setattr(rest_module.persistence, "add_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(
        rest_module.persistence, "add_audit_log_coalesced", lambda *a, **k: True, raising=False
    )
    _safety_state["emergency_stop_active"] = False
    _blade_state["active"] = True
    rest_module._emergency_until = 0.0
    _client_emergency.clear()
    yield
    _safety_state["emergency_stop_active"] = False
    rest_module._emergency_until = 0.0
    _client_emergency.clear()


@pytest.fixture
def stop_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def fake_execute(source: str = "api"):
        # the latch must already be set when the orchestrator runs
        assert _safety_state["emergency_stop_active"] is True
        calls.append(source)
        return {
            "emergency_stop_active": True,
            "motors_stopped": True,
            "blade_disabled": True,
            "complete": True,
            "subsystems": {},
            "faults": [],
        }

    monkeypatch.setattr(
        "backend.src.services.emergency_stop_service.execute_emergency_stop", fake_execute
    )
    return calls


def _client() -> httpx.AsyncClient:
    from backend.src.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("route", list(ROUTES))
async def test_every_body_shape_latches_the_stop_once(route, shape, stop_calls):
    async with _client() as client:
        resp = await client.post(route, **SHAPES[shape])

    assert resp.status_code == ROUTES[route], (shape, resp.status_code, resp.text[:200])
    assert _safety_state["emergency_stop_active"] is True
    assert _blade_state["active"] is False
    assert len(stop_calls) == 1


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_tractor_estop_route_accepts_every_body_shape(shape, monkeypatch):
    ran: list[int] = []

    class FakeTractor:
        async def emergency_stop(self):
            ran.append(1)
            return {"emergency_stop_active": True, "complete": True}

    monkeypatch.setattr(
        "backend.src.api.routers.tractor.get_tractor_service", lambda: FakeTractor()
    )
    async with _client() as client:
        resp = await client.post("/api/v2/tractor/emergency-stop", **SHAPES[shape])
    assert resp.status_code == 200, resp.text[:200]
    assert ran == [1]


async def test_legacy_command_payload_and_session_still_work(stop_calls):
    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency", json={"command": "emergency_stop"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "EMERGENCY_STOP_ACTIVE"
    assert resp.json()["emergency_stop_active"] is True
    assert len(stop_calls) == 1


async def test_valid_json_object_returns_the_v2_response(stop_calls):
    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency", json={"reason": "operator"})
    body = resp.json()
    assert resp.status_code == 202
    assert body["accepted"] is True and body["emergency_stop"]["complete"] is True


async def test_stop_still_runs_when_audit_and_client_key_code_raise(monkeypatch, stop_calls):
    def boom(*_a, **_k):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(rest_module.persistence, "add_audit_log_coalesced", boom, raising=False)
    monkeypatch.setattr(rest_module, "_client_key", boom)
    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency", content=b"junk")
    assert resp.status_code == 202
    assert _safety_state["emergency_stop_active"] is True
    assert len(stop_calls) == 1


async def test_real_orchestrator_runs_for_a_non_object_body():
    """No fakes: the real orchestrator answers a text/plain body and reports truthfully."""
    async with _client() as client:
        resp = await client.post(
            "/api/v2/control/emergency", content=b"STOP", headers={"Content-Type": "text/plain"}
        )
    body = resp.json()
    assert resp.status_code == 202, body
    assert _safety_state["emergency_stop_active"] is True
    assert body["accepted"] is body["emergency_stop"]["complete"]


# -------- A15: the real middleware stack must not refuse an E-stop body ------------------


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(content=b"x" * 1_200_000, headers={"Content-Type": "application/json"}),
        dict(content=b'{"a": 1}', headers={"Content-Type": "text/plain"}),
        dict(content=b"y" * 1_200_000, headers={"Content-Type": "text/plain"}),
    ],
    ids=["oversized_json", "odd_content_type", "oversized_odd_content_type"],
)
async def test_input_validation_middleware_never_refuses_an_estop_body(route, kwargs, stop_calls):
    async with _client() as client:
        resp = await client.post(route, **kwargs)
    assert resp.status_code in (200, 202), (resp.status_code, resp.text[:200])
    assert _safety_state["emergency_stop_active"] is True
    assert len(stop_calls) == 1


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(content=b"x" * 1_200_000, headers={"Content-Type": "application/json"}), 413),
        (dict(content=b'{"a": 1}', headers={"Content-Type": "text/plain"}), 415),
    ],
    ids=["oversized_json", "odd_content_type"],
)
async def test_same_bodies_are_refused_on_a_non_exempt_route(kwargs, expected):
    async with _client() as client:
        resp = await client.post("/api/v2/control/drive", **kwargs)
    assert resp.status_code == expected
