"""No read route leaks a secret: seed canaries everywhere, read everything, find none (SEC20-01).

The PR #20 review found that anonymous ``GET /settings/remote-access`` and ``GET /settings/maps``
returned the stored Cloudflare tunnel token, the ngrok authtoken and the Google Maps key, and
that nothing in the suite looked at reads at all. This test seeds a unique canary into every
place the app keeps a secret:

* the environment variables the backend reads (signing secret, manual-unlock password, NTRIP
  login, API key secret, tunnel tokens, maps key, Thor link key) and a secrets file;
* every settings store, through the real PUT routes (maps, remote access, security, profile);
* the hardware config and the remote-access status the health report reads (a key such as
  ``ntrip_password`` or the Victron ``encryption_key`` under a block the report echoes);
* the live sessions: the login token, its session id, a WebSocket ticket and a manual-unlock
  session id;

then calls EVERY read route of the app (GET, and the HEAD and OPTIONS a route answers, plain and
as a CORS preflight) anonymously and with the operator's bearer, and fails if any canary appears
in any body or header. WebSocket routes are connected and their first frame is checked too.
It also proves the scan has teeth (a route that dumps the environment, one that echoes the
credential header and one that reads a store raw are all caught) and that the documentation hub,
which holds a real sudo password in ``docs/GOAL-2026-10-24-finish.md``, is closed to anonymous
callers.
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.testclient import TestClient

from backend.src.api import health as health_api
from backend.src.core.secret_redaction import redact_secrets
from backend.src.main import app
from tests.integration._route_walk import READING, routes_of

ROOT = Path(__file__).resolve().parents[2]
RUN = uuid.uuid4().hex[:10]

ENV_SEEDS = {
    "LAWN_BERRY_AUTH_SECRET": "signing-secret",
    "MANUAL_CONTROL_PASSWORD": "manual-password",
    "API_KEY_SECRET": "api-key-secret",
    "JWT_SECRET": "jwt-secret",
    "NTRIP_USERNAME": "ntrip-user",
    "NTRIP_PASSWORD": "ntrip-password",
    "CLOUDFLARE_TUNNEL_TOKEN": "env-cloudflare",
    "NGROK_AUTHTOKEN": "env-ngrok",
    "GOOGLE_MAPS_API_KEY": "env-maps-key",
    "LAWN_BERRY_THOR_LINK_KEY": "thor-link-key",
}
STORE_SEEDS = {
    "cloudflare_token": "store-cloudflare-token",
    "ngrok_token": "store-ngrok-token",
    "google_api_key": "store-maps-key",
    "legacy_api_key": "store-legacy-api-key",
    "totp_secret": "store-totp-secret",
    "password_hash": "store-password-hash",
    "backup_code": "store-backup-code",
    "webhook_url": "store-webhook-url",
    "nested_token": "store-nested-token",
}
CONFIG_SEEDS = {
    "gps_ntrip_password": "config-ntrip-password",
    "victron_key": "config-victron-key",
    "imu_token": "config-imu-token",
    "status_token": "status-file-token",
}
SECRETS_FILE_SEED = "secrets-file-jwt"


def _canary(label: str) -> str:
    return f"CANARY-{label}-{RUN}"


SEEDS: dict[str, str] = {
    **{f"env:{name}": _canary(f"E{i}") for i, name in enumerate(ENV_SEEDS)},
    **{f"store:{name}": _canary(f"S{i}") for i, name in enumerate(STORE_SEEDS)},
    **{f"config:{name}": _canary(f"C{i}") for i, name in enumerate(CONFIG_SEEDS)},
    "secrets-file": _canary("F0"),
}


def seed(group: str, name: str) -> str:
    return SEEDS[f"{group}:{name}"]


@pytest.fixture
def strict(monkeypatch):
    """Operator auth ON, the simulator open-auth opt-in OFF (the suite sets both the other way)."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "0")


@pytest.fixture
def seeded_files(tmp_path, monkeypatch):
    """Data dir, hardware config, health inputs and a secrets file, all carrying canaries."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(data))

    hardware = yaml.safe_load((ROOT / "config" / "hardware.yaml").read_text())
    hardware["gps"]["ntrip_password"] = seed("config", "gps_ntrip_password")
    hardware["victron"]["encryption_key"] = seed("config", "victron_key")
    hardware["imu"] = {"type": "BNO085", "api_token": seed("config", "imu_token")}
    config = tmp_path / "config"
    config.mkdir()
    (config / "hardware.yaml").write_text(yaml.safe_dump(hardware))
    (config / "default.json").write_text((ROOT / "config" / "default.json").read_text())
    (data / "remote_access_status.json").write_text(
        json.dumps(
            {"health": "healthy", "provider": "ngrok", "token": seed("config", "status_token")}
        )
    )
    secrets_file = config / "secrets.json"
    secrets_file.write_text(
        json.dumps({"JWT_SECRET": {"value": SEEDS["secrets-file"], "updated_at": "", "version": 1}})
    )
    secrets_file.chmod(0o600)
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(secrets_file))
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(config))

    service = health_api.health_service
    monkeypatch.setattr(service, "hardware_config_path", config / "hardware.yaml")
    monkeypatch.setattr(service, "system_config_path", config / "default.json")
    monkeypatch.setattr(service, "remote_access_status_path", data / "remote_access_status.json")
    for name in ENV_SEEDS:
        monkeypatch.setenv(name, seed("env", name))
    return tmp_path


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    """The scan reads every route as the operator: keep self-test and camera off real devices."""
    from backend.src.api.routers import maintenance
    from backend.src.services import camera_ipc_client

    monkeypatch.setattr(maintenance, "run_selftest", lambda: {"ok": True, "stub": True})

    async def one_frame() -> AsyncIterator[dict]:
        yield {"data": "AAAA"}

    monkeypatch.setattr(camera_ipc_client, "frame_stream", one_frame)


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def read_targets(target) -> list[tuple[str, frozenset[str]]]:
    """(concrete path, declared read methods) of every read route of ``target``."""
    found: dict[str, set[str]] = {}
    for info in routes_of(target):
        if info.kind in {"api", "route"} and info.reading:
            found.setdefault(_concrete(info.path), set()).update(info.methods & READING)
    return sorted((path, frozenset(methods)) for path, methods in found.items())


def leaks_in(response: httpx.Response, seeds: dict[str, str]) -> list[str]:
    text = response.content.decode("utf-8", "ignore")
    headers = "\n".join(f"{name}: {value}" for name, value in response.headers.items())
    return [label for label, value in seeds.items() if value in text or value in headers]


PREFLIGHTS = (
    {"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
)


async def scan(
    target, seeds: dict[str, str], operator: dict[str, str] | None
) -> tuple[list[str], int]:
    """Every read route, anonymous and as the operator: (the leaks found, requests made)."""
    found: list[str] = []
    requests = 0
    identities: list[tuple[str, dict[str, str]]] = [("anonymous", {})]
    if operator:
        identities.append(("operator", operator))
    transport = httpx.ASGITransport(app=target)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for who, headers in identities:
            for path, methods in read_targets(target):
                calls = [("GET", headers)] if "GET" in methods else []
                calls += [("HEAD", headers)] if "HEAD" in methods else []
                calls += [("OPTIONS", headers)]
                calls += [("OPTIONS", {**headers, **pre}) for pre in PREFLIGHTS]
                for method, call_headers in calls:
                    response = await client.request(method, path, headers=call_headers)
                    requests += 1
                    found += [
                        f"{who} {method} {path} -> {label}" for label in leaks_in(response, seeds)
                    ]
    return found, requests


# ---- the real app ------------------------------------------------------------------------------


async def _seed_the_stores(client: httpx.AsyncClient, operator: dict[str, str]) -> None:
    puts = [
        (
            "/api/v2/settings/remote-access",
            {
                "method": "ngrok",
                "cloudflare_token": seed("store", "cloudflare_token"),
                "ngrok_token": seed("store", "ngrok_token"),
            },
        ),
        (
            "/api/v2/settings/maps",
            {
                "provider": "google",
                "google_api_key": seed("store", "google_api_key"),
                "api_key": seed("store", "legacy_api_key"),
            },
        ),
        (
            "/api/v2/settings/security",
            {
                "level": "password_totp",
                "totp_secret": seed("store", "totp_secret"),
                "password_hash": seed("store", "password_hash"),
                "backup_codes": [seed("store", "backup_code")],
            },
        ),
        (
            "/api/v2/settings/system",
            {
                "webhook_url": seed("store", "webhook_url"),
                "integrations": {"slack": {"token": seed("store", "nested_token")}},
            },
        ),
    ]
    for path, body in puts:
        response = await client.put(path, json=body, headers=operator)
        assert response.status_code == 200, (path, response.text)


@pytest_asyncio.fixture
async def session_seeds(strict, seeded_files, operator_credential):
    """Log in, mint a ticket and a manual-unlock session; their ids become canaries too."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        assert login.status_code == 200, login.text
        token = login.json()["access_token"]
        operator = {"Authorization": f"Bearer {token}"}
        await _seed_the_stores(client, operator)

        ticket = await client.post("/api/v2/auth/ws-ticket", json={}, headers=operator)
        unlock = await client.post(
            "/api/v2/control/manual-unlock",
            json={"method": "password", "password": seed("env", "MANUAL_CONTROL_PASSWORD")},
            headers=operator,
        )
        assert ticket.status_code == 200 and unlock.status_code == 200, (ticket.text, unlock.text)
        claims = token.split(".")[1]
        sid = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4))).get("sid")
        seeds = {
            **SEEDS,
            "operator-token": token,
            "operator-credential": operator_credential,
            "ws-ticket": ticket.json()["ticket"],
            "manual-session": unlock.json()["session_id"],
        }
        if sid:
            seeds["login-session-id"] = sid
        yield seeds, operator


async def test_no_read_route_returns_a_seeded_secret(session_seeds):
    seeds, operator = session_seeds
    found, requests = await scan(app, seeds, operator)
    assert requests > 400, "the scan did not reach the routes it should"
    assert found == [], "secrets leaked by a read route:\n" + "\n".join(found)


async def test_the_scan_reaches_every_read_route_of_the_app(session_seeds):
    seeds, operator = session_seeds
    paths = {path for path, _ in read_targets(app)}
    # the secret-bearing ones and the open ones both
    for expected in (
        "/api/v2/settings/maps",
        "/api/v2/settings/remote-access",
        "/api/v2/settings",
        "/api/v2/docs/list",
        "/health",
        "/openapi.json",
        "/api/v2/system/selftest",
        "/api/v2/camera/stream.mjpeg",
        "/api/v2/auth/profile",
    ):
        assert expected in paths, expected
    assert len(paths) > 60


async def test_the_operator_can_still_read_what_the_stores_hold_but_not_the_values(session_seeds):
    seeds, operator = session_seeds
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        remote = (await client.get("/api/v2/settings/remote-access", headers=operator)).json()
        maps = (await client.get("/api/v2/settings/maps", headers=operator)).json()
    assert remote["cloudflare_token_set"] is True and remote["ngrok_token_set"] is True
    assert maps["google_api_key_set"] is True and maps["api_key_set"] is True
    assert seed("store", "cloudflare_token") not in json.dumps([remote, maps])


async def test_the_health_report_redacts_secret_named_config_keys(session_seeds):
    """``/health`` echoes whole blocks of the hardware config; an ``ntrip_password`` or
    ``api_token`` an operator added there must not come out (it did, verbatim)."""
    seeds, _ = session_seeds
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in ("/health", "/api/v2/health", "/api/v2/health/readiness"):
            body = (await client.get(path)).text
            assert seed("config", "gps_ntrip_password") not in body, path
            assert seed("config", "imu_token") not in body, path
        report = (await client.get("/health")).json()
    gps = next(c for c in report["hardware"]["checks"] if c["component"] == "gps")
    assert gps["detail"]["type"] == "LC29H-DA"  # the non-secret part of the block is still there


def test_the_redaction_drops_nothing_from_the_real_health_report(monkeypatch):
    """No legitimate key of today's report (shipped config) reads as a secret name."""
    service = health_api.health_service
    monkeypatch.setattr(service, "hardware_config_path", ROOT / "config" / "hardware.yaml")
    monkeypatch.setattr(service, "system_config_path", ROOT / "config" / "default.json")
    report = service.evaluate()
    assert redact_secrets(report) == report


# ---- the docs hub (a real credential sits in docs/) --------------------------------------------


async def test_the_docs_hub_is_closed_to_anonymous_callers(strict, seeded_files):
    docs = sorted(p.relative_to(ROOT / "docs").as_posix() for p in (ROOT / "docs").rglob("*.md"))
    assert "GOAL-2026-10-24-finish.md" in docs
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in ("list", "checksums", "freshness", "bundle", *docs):
            for headers in ({}, {"Authorization": "Bearer not-a-real-token"}):
                response = await client.get(f"/api/v2/docs/{path}", headers=headers)
                assert response.status_code == 401, (path, response.status_code)
                assert "terramax" not in response.text


async def test_the_docs_route_serves_only_markdown_inside_the_docs_folder(
    strict, seeded_files, operator_credential, monkeypatch
):
    docs = seeded_files / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# guide\n")
    (docs / "config.json").write_text('{"token": "x"}')
    private = seeded_files / "docs-private"  # shares the folder name as a prefix
    private.mkdir()
    (private / "secret.md").write_text("# private\n")
    (docs / "link").symlink_to(private)
    monkeypatch.setenv("LAWNBERRY_DOCS_DIR", str(docs))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        operator = {"Authorization": f"Bearer {login.json()['access_token']}"}
        assert (await client.get("/api/v2/docs/guide.md", headers=operator)).status_code == 200
        assert (await client.get("/api/v2/docs/config.json", headers=operator)).status_code == 404
        escaped = await client.get("/api/v2/docs/link/secret.md", headers=operator)
        assert escaped.status_code == 403, "a startswith() prefix check let a sibling folder in"
        assert "# private" not in escaped.text


# ---- the scan has teeth ------------------------------------------------------------------------


def _leaky_app() -> FastAPI:
    leaky = FastAPI()

    @leaky.get("/debug/env")
    def env() -> dict:
        return dict(os.environ)

    @leaky.get("/debug/echo")
    def echo(request: Request) -> dict:
        return {"you_sent": request.headers.get("authorization")}

    @leaky.get("/debug/store", response_class=PlainTextResponse)
    def store():
        return Path(os.environ["LAWNBERRY_DATA_DIR"], "settings.json").read_text()

    @leaky.get("/debug/header")
    def header():
        return JSONResponse({}, headers={"X-Debug-Secret": os.environ["NTRIP_PASSWORD"]})

    @leaky.get("/fine")
    def fine() -> dict:
        return {"ok": True}

    return leaky


async def test_the_scan_catches_a_route_that_dumps_the_environment_or_a_store(
    session_seeds, seeded_files
):
    seeds, operator = session_seeds
    (seeded_files / "data" / "settings.json").write_text(json.dumps({"k": seeds["secrets-file"]}))
    found, _ = await scan(_leaky_app(), seeds, operator)
    text = "\n".join(found)
    assert "operator GET /debug/env -> env:NTRIP_PASSWORD" in text
    assert "anonymous GET /debug/env -> env:LAWN_BERRY_AUTH_SECRET" in text
    assert "operator GET /debug/echo -> operator-token" in text  # the credential header echoed
    assert "GET /debug/store -> secrets-file" in text
    assert "GET /debug/header -> env:NTRIP_PASSWORD" in text  # a leak in a header
    assert "/fine" not in text


# ---- WebSockets --------------------------------------------------------------------------------


def test_the_first_frames_of_the_websockets_carry_no_secret(session_seeds):
    """Anonymous loopback (read-only sockets) and ticketed connections; first frame only."""
    seeds, operator = session_seeds
    client = TestClient(app, client=("127.0.0.1", 50000))
    frames = []
    with client.websocket_connect("/api/v2/ws/status") as ws:
        frames.append(ws.receive_text())
    with client.websocket_connect("/api/v2/ws/telemetry") as ws:
        frames.append(ws.receive_text())
    ticket = client.post("/api/v2/auth/ws-ticket", json={}, headers=operator).json()["ticket"]
    with client.websocket_connect(f"/api/v2/ws/control?ticket={ticket}") as ws:
        frames.append(ws.receive_text())
    assert len(frames) == 3
    for frame in frames:
        for label, value in seeds.items():
            if label == "ws-ticket":
                continue  # the ticket was just redeemed; it cannot be in a frame it opened
            assert value not in frame, (label, frame[:200])
