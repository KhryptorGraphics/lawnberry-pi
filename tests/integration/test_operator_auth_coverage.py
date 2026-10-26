"""Every route of the app is gated, or reviewed with a reason, whatever kind of route it is.

SEC-02 gated every FastAPI POST/PUT/PATCH/DELETE route and pinned the open ones. The first
version only looked at ``APIRoute``: a plain Starlette ``Route``, a route of a mounted
sub-application and a ``WebSocketRoute`` were invisible, so the PR #20 review could add a
``Route("/api/v2/backdoor", POST)`` that set the levers and a ``@app.websocket`` that actuated them
with no authorization, and 112 tests stayed green (mutants M-ROUTE and M-WS). The reads were never
pinned at all, which let ``GET /settings/remote-access`` hand out the tunnel token (SEC20-01).
This module:

* walks EVERY route (``tests/integration/_route_walk.py``: API routes, plain routes, websockets,
  and the routes inside ``Mount``/``Host`` sub-applications; an opaque mount fails) and fails on
  any state-changing route with no ``require_operator_auth`` dependency that is not in
  ``OPEN_BY_DESIGN``, any read route that is neither gated nor in ``OPEN_READS``/``SELF_AUTH_READS``
  with a reason, and any stale entry;
* requires every WebSocket route to be in ``WEBSOCKET_CONTRACT`` and to start with
  ``await admit_websocket(...)`` (or the inner ``_authorize_websocket``), with
  ``command_capable=True`` exactly for the command socket, and proves the contract at the
  handshake: no anonymous peer (loopback included, unless explicitly opted in), no bad
  credential for anyone, and no credential-less command socket while operator auth is on;
* proves every gated route answers 401 without a real bearer token;
* proves the walk itself with the two backdoors of the review (and a mounted sub-application, an
  opaque mount, an unreviewed read) added to a COPY of the app.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient, WebSocketDenialResponse
from starlette.websockets import WebSocket

from backend.src.api.deps import require_operator_auth
from backend.src.api.routers.auth import _authorize_websocket
from backend.src.main import app
from backend.src.middleware.emergency_paths import EMERGENCY_STOP_PATHS
from tests.integration._route_walk import gated, routes_of

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

# ---- reads (SEC20-01) --------------------------------------------------------------------------
# path -> why an anonymous caller may read it (GET, and the HEAD/OPTIONS the route declares).
# A read is OPEN only if it returns nothing an anonymous caller must not have and does nothing
# but read. Everything else carries `Depends(require_operator_auth)`: the secret-bearing settings
# stores, the whole documentation hub (the folder holds runbooks and hand-off notes that quote LAN
# addresses and credentials), the hardware self-test (a GET that opens serial ports, probes I2C and
# drives GPIO) and the RTK diagnostics (a GET that can start the NTRIP forwarder).

_API_DESCRIPTION = "the API's own description (route and schema names); no values"
_HEALTH = (
    "liveness/readiness/health rollup for monitors and the UI: subsystem status, configuration "
    "state and server paths; secret-named keys are redacted from the report"
)
_STATE = (
    "live robot state the unauthenticated dashboard shows (battery, mode, interlocks, "
    "readings); read-only, no credentials"
)
_SENSORS = (
    "live sensor reading for the dashboard; read-only. The first call may create the sensor "
    "manager that the telemetry loop creates at start-up anyway"
)
_GEOMETRY = (
    "lawn zones / home position drawn on the dashboard map; no credentials. That an open LAN UI "
    "shows the property is an owner decision (see the audit)"
)
_SCHEDULE = "mowing schedule / mission status the dashboard lists; no credentials"
_CAMERA = (
    "live camera: an <img src> cannot send an Authorization header, so it stays open; owner "
    "decision (privacy of the feed; a ticket scheme would be needed to gate it)"
)
_SETTINGS = (
    "settings the dashboard needs before sign-in; every secret-named key is dropped and not even "
    "its presence is reported (api/routers/settings.py, core/secret_redaction.py)"
)
_HANDSHAKE = (
    "HTTP emulation of a WebSocket handshake: returns only the computed Sec-WebSocket-Accept "
    "headers, no data and no state; its bearer-presence check is not authentication"
)
_RECORDING = (
    "recording sessions of the perimeter recorder; read-only. Includes server-side file paths "
    "(not credentials)"
)

OPEN_READS: dict[str, str] = {
    "/": "service banner: name, version and the documented endpoints",
    "/openapi.json": _API_DESCRIPTION,
    "/docs": _API_DESCRIPTION,
    "/docs/oauth2-redirect": _API_DESCRIPTION,
    "/redoc": _API_DESCRIPTION,
    "/health": _HEALTH,
    "/healthz": "liveness: a constant",
    "/api/v2/health": _HEALTH,
    "/api/v2/health/liveness": _HEALTH,
    "/api/v2/health/readiness": _HEALTH,
    "/metrics": "Prometheus counters and gauges (CPU, memory, request counts, timings, TLS expiry)",
    "/api/v2/status": _STATE,
    "/api/v1/status": _STATE,
    "/api/v2/dashboard/status": _STATE,
    "/api/v2/dashboard/telemetry": _STATE,
    "/api/v2/dashboard/metrics": _STATE,
    "/api/v2/fusion/state": _STATE,
    "/api/v2/nav/status": _STATE,
    "/api/v2/navigation/status": _STATE,
    "/api/v2/tractor/state": _STATE,
    "/api/v2/control/blade/status": _STATE,
    "/api/v2/capture/status": _STATE,
    "/api/v2/hardware/robohat": _STATE,
    "/api/v2/ai/status": _STATE,
    "/api/v2/ai/metrics": _STATE,
    "/api/v2/ai/health": _STATE,
    "/api/v2/ai/datasets": "names of the datasets that can be exported",
    "/api/v2/sensors/health": _SENSORS,
    "/api/v2/sensors/tof/status": _SENSORS,
    "/api/v2/sensors/gps/status": _SENSORS,
    "/api/v2/sensors/imu/status": _SENSORS,
    "/api/v2/sensors/environmental/status": _SENSORS,
    "/api/v2/sensors/power/status": _SENSORS,
    "/api/v2/telemetry/stream": "telemetry history; (in SIM_MODE only) seeds simulated rows",
    "/api/v2/telemetry/export": "telemetry diagnostic history (power, latency) as a download",
    "/api/v2/system/timezone": "the detected time zone",
    "/api/v2/maintenance/imu/calibrate": "status of the last IMU calibration (the POST is gated)",
    "/api/v2/weather/current": "the BME280 reading",
    "/api/v2/weather/planning-advice": "mowing advice derived from the reading",
    "/api/v2/missions/{mission_id}/status": _SCHEDULE,
    "/api/v2/missions/list": _SCHEDULE,
    "/api/v2/planning/jobs": _SCHEDULE,
    "/api/v1/mow/jobs": _SCHEDULE,
    "/api/v2/map/zones": _GEOMETRY,
    "/api/v2/map/locations": _GEOMETRY,
    "/api/v2/map/configuration": _GEOMETRY,
    "/api/v1/maps/zones": _GEOMETRY,
    "/api/v2/recording/status": _RECORDING,
    "/api/v2/recording/sessions": _RECORDING,
    "/api/v2/recording/sessions/{session_id}": _RECORDING,
    "/api/v2/recording/health": _RECORDING,
    "/api/v2/camera/status": _CAMERA,
    "/api/v2/camera/frame": _CAMERA,
    "/api/v2/camera/stream.mjpeg": _CAMERA,
    "/api/v2/ws/telemetry": _HANDSHAKE,
    "/api/v2/ws/control": _HANDSHAKE,
    "/api/v2/ws/settings": _HANDSHAKE,
    "/api/v2/ws/notifications": _HANDSHAKE,
    "/api/v2/settings": _SETTINGS,
    "/api/v2/settings/system": _SETTINGS,
    "/api/v2/settings/security": _SETTINGS,
    "/api/v2/control/manual-unlock/status": (
        "the Control view's Cloudflare-gate probe: authorized=false while auth is enforced; "
        "no session is minted"
    ),
}

# Reads that verify the bearer themselves: not a dependency, so listed with the proof below.
SELF_AUTH_READS: dict[str, str] = {
    "/api/v2/auth/profile": _SELF_AUTH,
}

# Reads that MUST carry the dependency (a regression that drops it must fail, not just move the
# route into OPEN_READS).
MUST_BE_GATED_READS = frozenset(
    {
        "/api/v2/settings/maps",
        "/api/v2/settings/remote-access",
        "/api/v2/docs/list",
        "/api/v2/docs/checksums",
        "/api/v2/docs/freshness",
        "/api/v2/docs/bundle",
        "/api/v2/docs/{doc_path:path}",
        "/api/v2/system/selftest",
        "/api/v2/sensors/gps/rtk/diagnostics",
    }
)

# ---- websockets --------------------------------------------------------------------------------
READ_ONLY = "read-only stream (what the open GET routes carry); no handler changes mower state"
COMMAND = (
    "command-capable socket: refused without a credential whenever operator auth is on, "
    "loopback peer and SIM_MODE included"
)
WEBSOCKET_CONTRACT: dict[str, str] = {
    "/api/v2/ws/telemetry": READ_ONLY,
    "/ws/telemetry": READ_ONLY,
    "/api/v2/ws/status": READ_ONLY,
    "/api/v2/ws/control": COMMAND,
    "/ws/control": COMMAND,
}


# ---- the checks, as functions of an app so the mutants below can run them on a copy ------------


def _ungated_mutating(target) -> set[tuple[str, str]]:
    return {
        (method, info.path)
        for info in routes_of(target)
        if info.kind in {"api", "route"} and info.mutating
        for method in info.methods & {"POST", "PUT", "PATCH", "DELETE"}
        if not gated(info, require_operator_auth)
    }


def _read_problems(target) -> list[str]:
    """Read routes that are neither gated nor reviewed (by path)."""
    problems = []
    for info in routes_of(target):
        if info.kind not in {"api", "route"} or not info.reading:
            continue
        if gated(info, require_operator_auth):
            continue
        if info.path in OPEN_READS or info.path in SELF_AUTH_READS:
            continue
        problems.append(info.path)
    return sorted(set(problems))


def _opaque_mounts(target) -> list[str]:
    return sorted(info.path for info in routes_of(target) if info.kind == "mount" and info.opaque)


def _other_routes(target) -> list[str]:
    return sorted(info.path for info in routes_of(target) if info.kind == "other")


def _unreviewed_websockets(target) -> list[str]:
    return sorted(
        info.path
        for info in routes_of(target)
        if info.kind == "websocket" and info.path not in WEBSOCKET_CONTRACT
    )


# ``admit_websocket`` is the handlers' admission call: it wraps ``_authorize_websocket`` and adds
# the per-session connection slot. A handler may also call the inner authorizer directly.
WEBSOCKET_AUTHORIZERS = frozenset({"admit_websocket", "_authorize_websocket"})


def _first_statement_authorizes(endpoint: Callable) -> tuple[bool, bool]:
    """(the first statement awaits an admission call, with command_capable=True)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoint)))
    function = tree.body[0]
    body = [
        stmt
        for stmt in function.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))  # docstring
    ]
    first = body[0] if body else None
    call = None
    if isinstance(first, (ast.Assign, ast.Expr)) and isinstance(first.value, ast.Await):
        call = first.value.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in WEBSOCKET_AUTHORIZERS
    ):
        return False, False
    command = any(
        kw.arg == "command_capable"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in call.keywords
    )
    return True, command


def _websocket_handler_problems(target) -> list[str]:
    problems = []
    for info in routes_of(target):
        if info.kind != "websocket":
            continue
        authorizes, command = _first_statement_authorizes(info.endpoint)
        wants_command = WEBSOCKET_CONTRACT.get(info.path) == COMMAND
        if not authorizes:
            problems.append(f"{info.path}: does not start with `await admit_websocket(...)`")
        elif command != wants_command:
            problems.append(
                f"{info.path}: command_capable={command}, the contract says {wants_command}"
            )
    return sorted(problems)


# ---- the real app ------------------------------------------------------------------------------


def test_every_mutating_route_is_gated_or_reviewed():
    ungated = _ungated_mutating(app)
    unreviewed = sorted(ungated - set(OPEN_BY_DESIGN))
    stale = sorted(set(OPEN_BY_DESIGN) - ungated)
    assert not unreviewed, (
        "state-changing routes with no require_operator_auth dependency and no reviewed "
        f"OPEN_BY_DESIGN entry: {unreviewed}"
    )
    assert not stale, f"OPEN_BY_DESIGN entries that are gated or no longer exist: {stale}"


def test_every_read_route_is_gated_or_reviewed():
    assert _read_problems(app) == [], (
        "read routes with no require_operator_auth dependency and no OPEN_READS/SELF_AUTH_READS "
        "entry (a GET can leak: SEC20-01)"
    )


def test_the_read_allow_lists_have_no_stale_or_contradictory_entries():
    reads = {
        info.path: info for info in routes_of(app) if info.kind in {"api", "route"} and info.reading
    }
    stale = sorted((set(OPEN_READS) | set(SELF_AUTH_READS)) - set(reads))
    assert not stale, f"allow-listed reads that no longer exist: {stale}"

    now_gated = sorted(
        path
        for path in set(OPEN_READS) | set(SELF_AUTH_READS)
        if all(
            gated(i, require_operator_auth) for i in routes_of(app) if i.path == path and i.reading
        )
    )
    assert not now_gated, f"allow-listed reads that now carry the dependency: {now_gated}"
    assert not set(OPEN_READS) & set(SELF_AUTH_READS)
    assert all(reason.strip() for reason in {**OPEN_READS, **SELF_AUTH_READS}.values())


def test_the_reads_that_must_stay_gated_are_gated():
    for path in sorted(MUST_BE_GATED_READS):
        matching = [i for i in routes_of(app) if i.path == path and i.reading]
        assert matching, f"{path}: no such read route"
        assert all(gated(i, require_operator_auth) for i in matching), path
        assert path not in OPEN_READS, path


def test_no_route_hides_in_a_mount_or_is_of_an_unknown_kind():
    assert _opaque_mounts(app) == [], "a mount that exposes no routes cannot be reviewed"
    assert _other_routes(app) == [], "a route of a kind the walk does not understand"


def test_the_open_estop_routes_are_exactly_the_rate_limit_exempt_paths():
    open_estop = {
        path for (method, path), why in OPEN_BY_DESIGN.items() if why == _ESTOP and method == "POST"
    }
    assert open_estop == set(EMERGENCY_STOP_PATHS)
    routed = {info.path for info in routes_of(app) if info.mutating}
    assert set(EMERGENCY_STOP_PATHS) <= routed, "an exempt E-stop path has no route"


def test_every_websocket_route_is_reviewed_and_starts_with_the_authorizer():
    assert _unreviewed_websockets(app) == []
    present = {info.path for info in routes_of(app) if info.kind == "websocket"}
    assert present == set(WEBSOCKET_CONTRACT), "a reviewed websocket route no longer exists"
    assert _websocket_handler_problems(app) == []


# ---- the walk itself: the review's backdoors, on a COPY of the app -----------------------------


def _variant(*extra) -> FastAPI:
    copy = FastAPI()
    copy.router.routes[:] = [*app.router.routes, *extra]
    return copy


async def _would_set_the_levers(request):  # never called: the checks only read the routes
    from backend.src.services.tractor_service import get_tractor_service

    await get_tractor_service().set_levers(0.9, 0.9)
    return JSONResponse({"ok": True})


async def _ws_backdoor(websocket: WebSocket):  # the M-WS mutant: actuates with no authorizer
    from backend.src.services.tractor_service import get_tractor_service

    await websocket.accept()
    await get_tractor_service().set_levers(0.9, 0.9)


async def _ws_authorized_but_unlisted(websocket: WebSocket):
    # (the module-level import keeps the authorizer call the handler's first statement)

    await _authorize_websocket(websocket)


def test_the_baseline_copy_is_clean():
    copy = _variant()
    assert _ungated_mutating(copy) - set(OPEN_BY_DESIGN) == set()
    assert _read_problems(copy) == [] and _opaque_mounts(copy) == []
    assert _unreviewed_websockets(copy) == [] and _websocket_handler_problems(copy) == []


def test_a_plain_starlette_post_route_is_caught():
    """Mutant M-ROUTE: `Route("/api/v2/backdoor", POST)` that sets the levers, unauthenticated."""
    copy = _variant(Route("/api/v2/backdoor", _would_set_the_levers, methods=["POST"]))
    assert ("POST", "/api/v2/backdoor") in _ungated_mutating(copy) - set(OPEN_BY_DESIGN)


def test_a_route_of_a_mounted_sub_application_is_caught():
    sub = FastAPI()
    sub.add_api_route("/lever", _would_set_the_levers, methods=["PUT"])
    copy = _variant(Mount("/hidden", app=sub))
    assert ("PUT", "/hidden/lever") in _ungated_mutating(copy) - set(OPEN_BY_DESIGN)


def test_a_mounted_asgi_callable_is_caught_as_opaque():
    async def asgi(scope, receive, send):  # an app the walk cannot look into
        raise AssertionError("never called")

    copy = _variant(Mount("/opaque", app=asgi))
    assert _opaque_mounts(copy) == ["/opaque"]


def test_an_unreviewed_read_is_caught():
    async def leak(request):
        return JSONResponse({})

    copy = _variant(Route("/api/v2/debug/env", leak, methods=["GET"]))
    assert _read_problems(copy) == ["/api/v2/debug/env"]
    api = _variant()
    api.add_api_route("/api/v2/debug/config", leak, methods=["GET"])
    assert _read_problems(api) == ["/api/v2/debug/config"]


def test_an_unauthenticated_actuating_websocket_is_caught():
    """Mutant M-WS: `@app.websocket("/api/v2/ws/backdoor")` that actuates, no authorizer."""
    copy = _variant()
    copy.add_api_websocket_route("/api/v2/ws/backdoor", _ws_backdoor)
    assert _unreviewed_websockets(copy) == ["/api/v2/ws/backdoor"]
    problems = _websocket_handler_problems(copy)
    assert any("/api/v2/ws/backdoor" in p and "admit_websocket" in p for p in problems), problems


def test_a_websocket_that_authorizes_but_is_not_reviewed_is_still_caught():
    copy = _variant()
    copy.add_api_websocket_route("/api/v2/ws/extra", _ws_authorized_but_unlisted)
    assert _unreviewed_websockets(copy) == ["/api/v2/ws/extra"]
    assert _websocket_handler_problems(copy) == []  # it authorizes; only the review is missing


def test_a_command_socket_that_forgets_command_capable_is_caught():
    async def control(websocket: WebSocket):
        # (module-level import: the authorizer call is the first statement)

        await _authorize_websocket(websocket)  # forgot command_capable=True
        await websocket.accept()

    copy = FastAPI()
    copy.router.routes[:] = [
        r for r in app.router.routes if getattr(r, "path", "") != "/ws/control"
    ]
    copy.add_api_websocket_route("/ws/control", control)
    problems = _websocket_handler_problems(copy)
    assert any("/ws/control" in p and "command_capable=False" in p for p in problems), problems


# ---- the gates work ----------------------------------------------------------------------------


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


@pytest.fixture
def enforced(monkeypatch):
    """Operator auth ON and the simulator open-auth opt-in OFF (the suite has them reversed)."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "0")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _gated() -> list[tuple[str, str]]:
    return sorted(
        {
            (method, info.path)
            for info in routes_of(app)
            if info.kind == "api" and gated(info, require_operator_auth)
            for method in info.methods
        }
    )


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


@pytest.mark.parametrize("path", sorted(SELF_AUTH_READS))
async def test_a_self_authenticating_read_refuses_a_request_without_a_token(enforced, path):
    async with _client() as client:
        for headers in ({}, {"Authorization": "Bearer not-a-real-token"}):
            response = await client.get(path, headers=headers)
            assert response.status_code == 401, (path, response.status_code, response.text)


async def test_the_routes_the_review_listed_are_in_the_gated_set():
    gated_routes = set(_gated())
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
        ("GET", "/api/v2/settings/maps"),
        ("GET", "/api/v2/settings/remote-access"),
    }
    assert expected <= gated_routes, sorted(expected - gated_routes)


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
        # a gated read behind the same token
        assert (await client.get("/api/v2/docs/list", headers=auth)).status_code == 200
        profile = await client.get("/api/v2/auth/profile", headers=auth)
        assert profile.status_code == 200, profile.text


# ---- the websocket contract at the handshake ---------------------------------------------------

LAN_PEER = ("203.0.113.50", 50000)  # not the loopback: a LAN or tunnel client
LOOPBACK_PEER = ("127.0.0.1", 50000)


def _refused(peer, path: str, **kwargs) -> int | None:
    """The HTTP status a WebSocket handshake was refused with, ``None`` if it was accepted."""
    client = TestClient(app, client=peer)
    try:
        with client.websocket_connect(path, **kwargs):
            return None
    except WebSocketDenialResponse as denied:
        return denied.status_code


@pytest.mark.parametrize("path", sorted(WEBSOCKET_CONTRACT))
def test_no_anonymous_lan_peer_is_admitted_on_any_websocket(enforced, path):
    assert _refused(LAN_PEER, path) == 401


@pytest.mark.parametrize("path", sorted(WEBSOCKET_CONTRACT))
def test_a_bad_credential_is_refused_on_every_websocket_even_from_the_loopback(enforced, path):
    assert _refused(LOOPBACK_PEER, f"{path}?ticket=not-a-ticket") == 401
    bad_bearer = {"Authorization": "Bearer not-a-real-token"}
    assert _refused(LOOPBACK_PEER, path, headers=bad_bearer) == 401


@pytest.mark.parametrize(
    "path", sorted(p for p, contract in WEBSOCKET_CONTRACT.items() if contract == COMMAND)
)
def test_the_command_websocket_needs_a_credential_even_from_the_loopback(enforced, path):
    assert _refused(LOOPBACK_PEER, path) == 401
