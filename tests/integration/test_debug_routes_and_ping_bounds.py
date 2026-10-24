"""The ``/debug/*`` injectors exist only in a simulator or when explicitly enabled, and
``/telemetry/ping`` is bounded.

* The GPS, geofence, ToF and tilt injectors were reachable on a real mower (operator auth was
  their only gate). They now answer 404, before any authentication check, unless ``SIM_MODE=1`` or
  ``LAWNBERRY_ENABLE_DEBUG_ROUTES`` is a true token; operator auth is required on top, always.
* ``POST /telemetry/ping`` took an unbounded ``sample_count`` (a loop with a 1 ms sleep per sample,
  one request could pin a worker coroutine and its memory for hours): 1 to 100 now, else 422.

The bodies below are deliberately invalid, so a request that gets past the gates stops at 422 and
no injector ever changes any state.
"""

from __future__ import annotations

import httpx
import pytest

from backend.src.main import app

DEBUG_POSTS = [
    ("/api/v2/debug/gps/inject", {"latitude": "not-a-number"}),
    ("/api/v2/debug/geofence", {"boundary": "not-a-list"}),
    ("/api/v2/debug/sensors/inject-tilt", {"roll_deg": "not-a-number"}),
    ("/api/v2/debug/sensors/inject-tof", {"position": 7}),
]
IDS = [path.rsplit("/", 1)[-1] for path, _ in DEBUG_POSTS]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def real_mower(monkeypatch):
    """What the unit ships: SIM_MODE=0 and no debug-routes flag."""
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("LAWNBERRY_ENABLE_DEBUG_ROUTES", raising=False)
    return monkeypatch


@pytest.mark.parametrize(("path", "body"), DEBUG_POSTS, ids=IDS)
async def test_the_injectors_do_not_exist_on_a_real_mower(real_mower, path, body):
    real_mower.setenv("OPERATOR_AUTH_REQUIRED", "0")
    async with _client() as client:
        resp = await client.post(path, json=body)
    assert resp.status_code == 404, (path, resp.status_code, resp.text)


@pytest.mark.parametrize(("path", "body"), DEBUG_POSTS, ids=IDS)
async def test_a_hidden_injector_is_not_revealed_by_authentication(
    real_mower, operator_credential, path, body
):
    """With the injectors off, logged in or not, the answer is the same 404."""
    real_mower.setenv("OPERATOR_AUTH_REQUIRED", "1")
    async with _client() as client:
        anonymous = await client.post(path, json=body)
        real_mower.setenv("SIM_MODE", "1")  # the suite's login accepts any credential
        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        real_mower.setenv("SIM_MODE", "0")
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
        logged_in = await client.post(path, json=body, headers=auth)
    assert anonymous.status_code == 404 and logged_in.status_code == 404


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
async def test_the_explicit_flag_enables_them_for_bench_use(real_mower, flag):
    real_mower.setenv("OPERATOR_AUTH_REQUIRED", "0")
    real_mower.setenv("LAWNBERRY_ENABLE_DEBUG_ROUTES", flag)
    async with _client() as client:
        for path, body in DEBUG_POSTS:
            assert (await client.post(path, json=body)).status_code == 422, path


@pytest.mark.parametrize("flag", ["0", "false", "", "maybe", "2"])
async def test_anything_but_a_true_token_keeps_them_off(real_mower, flag):
    real_mower.setenv("OPERATOR_AUTH_REQUIRED", "0")
    real_mower.setenv("LAWNBERRY_ENABLE_DEBUG_ROUTES", flag)
    async with _client() as client:
        for path, body in DEBUG_POSTS:
            assert (await client.post(path, json=body)).status_code == 404, (flag, path)


@pytest.mark.parametrize(("path", "body"), DEBUG_POSTS, ids=IDS)
async def test_enabled_injectors_still_need_operator_auth(
    real_mower, operator_credential, path, body
):
    real_mower.setenv("OPERATOR_AUTH_REQUIRED", "1")
    real_mower.setenv("LAWNBERRY_ENABLE_DEBUG_ROUTES", "1")
    async with _client() as client:
        assert (await client.post(path, json=body)).status_code == 401
        real_mower.setenv("SIM_MODE", "1")
        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        real_mower.setenv("SIM_MODE", "0")
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
        assert (await client.post(path, json=body, headers=auth)).status_code == 422


@pytest.mark.parametrize(("path", "body"), DEBUG_POSTS, ids=IDS)
async def test_a_simulator_has_them_without_any_flag(monkeypatch, path, body):
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.delenv("LAWNBERRY_ENABLE_DEBUG_ROUTES", raising=False)
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "0")
    async with _client() as client:
        assert (await client.post(path, json=body)).status_code == 422


# ---- /telemetry/ping -------------------------------------------------------------------------


@pytest.mark.parametrize("count", [101, 1000, 0, -1])
async def test_ping_refuses_an_out_of_range_sample_count(count):
    async with _client() as client:
        resp = await client.post(
            "/api/v2/telemetry/ping", json={"component_id": "power", "sample_count": count}
        )
    assert resp.status_code == 422, (count, resp.text)


@pytest.mark.parametrize("count", [1, 5, 100])
async def test_ping_accepts_the_documented_range(count):
    async with _client() as client:
        resp = await client.post(
            "/api/v2/telemetry/ping", json={"component_id": "power", "sample_count": count}
        )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["samples"]) == count


async def test_ping_default_is_unchanged():
    async with _client() as client:
        resp = await client.post("/api/v2/telemetry/ping", json={"component_id": "power"})
    assert resp.status_code == 200 and len(resp.json()["samples"]) == 10
