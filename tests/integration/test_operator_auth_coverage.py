"""Every state-changing route needs operator auth, except a short reviewed list (SEC-02).

``POST /missions/{id}/start`` (drives the mower, sets ``blade_on``), ``/maintenance/imu/calibrate``
(drives the RoboHAT for about 13 s), ``/ai/enable``, ``/nav/*``, the ``/debug/*`` injectors and
``/control/drive|blade`` had no ``require_operator_auth`` dependency although the unit and the docs
said missions were gated. This module:

* enumerates EVERY POST/PUT/PATCH/DELETE route of ``backend.src.main.app`` and fails when one
  without the dependency is not in ``OPEN_BY_DESIGN`` (a new unauthenticated mutating route needs
  a reviewed entry with a reason) or when an entry there no longer matches a route;
* proves each gated route really answers 401 without a bearer token (the dependency runs before
  body validation, so an empty body is enough and no handler ever executes).
"""

from __future__ import annotations

import re

import httpx
import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from backend.src.api.deps import require_operator_auth
from backend.src.main import app
from backend.src.middleware.emergency_paths import EMERGENCY_STOP_PATHS

MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

_ESTOP = "E-stop: must work with no credential, a dead login service or a hostile network"
_SELF_AUTH = "checks the bearer token itself (_require_session): 401 without one at any setting"
_LOGIN = (
    "the login: exchanges the operator credential for a token; per-peer limiter, off-loop bcrypt"
)

# (method, path) -> why no require_operator_auth. Adding a line here is a security decision.
OPEN_BY_DESIGN: dict[tuple[str, str], str] = {
    ("POST", "/api/v2/auth/login"): _LOGIN,
    ("POST", "/api/v1/auth/login"): _LOGIN,
    ("POST", "/api/v2/auth/logout"): _SELF_AUTH,
    ("POST", "/api/v2/auth/refresh"): _SELF_AUTH,
    ("POST", "/api/v2/control/emergency"): _ESTOP,
    ("POST", "/api/v2/control/emergency-stop"): _ESTOP,
    ("POST", "/api/v2/tractor/emergency-stop"): _ESTOP,
}


def _depends_on(dependant: Dependant, call) -> bool:
    return any(sub.call is call or _depends_on(sub, call) for sub in dependant.dependencies)


def _api_routes(routes) -> list[APIRoute]:
    found: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            found.append(route)
        elif hasattr(route, "routes"):  # a Mount / sub-router
            found.extend(_api_routes(route.routes))
    return found


def _mutating() -> list[tuple[str, str, APIRoute]]:
    rows = []
    for route in _api_routes(app.routes):
        for method in sorted(route.methods & MUTATING):
            rows.append((method, route.path, route))
    return rows


def test_every_mutating_route_is_gated_or_reviewed():
    ungated = {
        (method, path)
        for method, path, route in _mutating()
        if not _depends_on(route.dependant, require_operator_auth)
    }
    unreviewed = sorted(ungated - set(OPEN_BY_DESIGN))
    stale = sorted(set(OPEN_BY_DESIGN) - ungated)
    assert not unreviewed, (
        "state-changing routes with no require_operator_auth dependency and no reviewed "
        f"OPEN_BY_DESIGN entry: {unreviewed}"
    )
    assert not stale, f"OPEN_BY_DESIGN entries that are gated or no longer exist: {stale}"


def test_no_sub_application_is_mounted_unseen():
    """A Mount hides its routes from the enumeration above unless it exposes ``routes``."""
    mounts = [r for r in app.routes if type(r).__name__ == "Mount"]
    assert mounts == [], f"review these mounts and extend the enumeration: {mounts}"


def test_the_open_estop_routes_are_exactly_the_rate_limit_exempt_paths():
    open_estop = {
        path for (method, path), why in OPEN_BY_DESIGN.items() if why == _ESTOP and method == "POST"
    }
    assert open_estop == set(EMERGENCY_STOP_PATHS)
    routed = {path for _, path, _ in _mutating()}
    assert set(EMERGENCY_STOP_PATHS) <= routed, "an exempt E-stop path has no route"


def _gated() -> list[tuple[str, str]]:
    return sorted(
        (method, path)
        for method, path, route in _mutating()
        if _depends_on(route.dependant, require_operator_auth)
    )


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize(("method", "path"), _gated(), ids=lambda v: str(v))
async def test_a_gated_route_refuses_a_request_without_a_token(enforced, method, path):
    async with _client() as client:
        anonymous = await client.request(method, _concrete(path), json={})
        forged = await client.request(
            method,
            _concrete(path),
            json={},
            headers={"Authorization": "Bearer not-a-real-token", "X-Session-Id": "manual-forged"},
        )
    for response in (anonymous, forged):
        assert response.status_code == 401, (method, path, response.status_code, response.text)
        assert response.json()["detail"] == "Operator authentication required"


async def test_the_routes_the_review_listed_are_in_the_gated_set():
    gated = set(_gated())
    expected = {
        ("POST", "/api/v2/missions/create"),
        ("POST", "/api/v2/missions/{mission_id}/start"),
        ("POST", "/api/v2/missions/{mission_id}/pause"),
        ("POST", "/api/v2/missions/{mission_id}/resume"),
        ("POST", "/api/v2/missions/{mission_id}/abort"),
        ("POST", "/api/v2/maintenance/imu/calibrate"),
        ("POST", "/api/v2/ai/enable"),
        ("POST", "/api/v2/ai/disable"),
        ("POST", "/api/v2/ai/model"),
        ("POST", "/api/v2/nav/mode"),
        ("POST", "/api/v2/nav/waypoints"),
        ("POST", "/api/v2/debug/gps/inject"),
        ("POST", "/api/v2/debug/geofence"),
        ("POST", "/api/v2/debug/sensors/inject-tilt"),
        ("POST", "/api/v2/debug/sensors/inject-tof"),
        ("POST", "/api/v2/control/drive"),
        ("POST", "/api/v2/control/blade"),
        ("POST", "/api/v2/control/manual-unlock"),
    }
    assert expected <= gated, sorted(expected - gated)


async def test_a_valid_bearer_token_passes_the_gate(enforced, operator_credential):
    """The dependency admits a real login; the handler then answers on its own terms."""
    async with _client() as client:
        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        assert login.status_code == 200, login.text
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
        # No handler side effect: an unknown id and an invalid body, after authentication.
        unknown = await client.post("/api/v2/missions/does-not-exist/abort", headers=auth)
        assert unknown.status_code == 404, unknown.text
        invalid = await client.post("/api/v2/nav/mode", json={"mode": "bogus"}, headers=auth)
        assert invalid.status_code == 422, invalid.text
