import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.src.middleware.api_key_auth import register_api_key_auth_middleware


def create_app():
    app = FastAPI()
    register_api_key_auth_middleware(app)

    @app.get("/api/v2/internal/status")
    def status():
        return {"ok": True}

    @app.get("/public")
    def public():
        return {"ok": True}

    return app


def test_api_key_required_on_protected_prefix(monkeypatch):
    monkeypatch.setenv("API_KEY_REQUIRED", "1")
    monkeypatch.setenv("API_KEY_PATH_PREFIXES", "/api/v2/internal")
    monkeypatch.setenv("API_KEY_SECRET", "abc123")

    app = create_app()
    client = TestClient(app)

    # Missing key
    r = client.get("/api/v2/internal/status")
    assert r.status_code == 401

    # Wrong key
    r = client.get("/api/v2/internal/status", headers={"X-API-Key": "nope"})
    assert r.status_code == 401

    # Correct key via header
    r = client.get("/api/v2/internal/status", headers={"X-API-Key": "abc123"})
    assert r.status_code == 200

    # Public path unaffected
    r = client.get("/public")
    assert r.status_code == 200


# ---- SEC3-14: the E-stop POSTs are open by design; a key requirement must not close them ----

ESTOP_PATHS = [
    "/api/v2/control/emergency",
    "/api/v2/control/emergency-stop",
    "/api/v2/tractor/emergency-stop",
]


def create_app_with_estops():
    app = FastAPI()
    register_api_key_auth_middleware(app)
    for path in ESTOP_PATHS + [
        "/api/v2/control/emergency_clear",
        "/api/v2/tractor/clear-emergency",
    ]:

        @app.post(path)
        def stop():
            return {"reached": True}

    @app.get("/api/v2/status")
    def status():
        return {"ok": True}

    return app


@pytest.mark.parametrize("prefixes", ["/api/v2", "/api/v2/control,/api/v2/tractor", "/"])
def test_the_estop_posts_need_no_api_key_whatever_prefixes_are_protected(monkeypatch, prefixes):
    """Old: ``API_KEY_REQUIRED=1`` and ``API_KEY_PATH_PREFIXES=/api/v2`` made all three 401."""
    monkeypatch.setenv("API_KEY_REQUIRED", "1")
    monkeypatch.setenv("API_KEY_PATH_PREFIXES", prefixes)
    monkeypatch.setenv("API_KEY_SECRET", "abc123")
    client = TestClient(create_app_with_estops())

    for path in ESTOP_PATHS:
        assert client.post(path).json() == {"reached": True}, path
        assert client.post(path + "/").status_code in (200, 307), path  # trailing slash
    # the authenticated neighbours and every other route under the prefix still need the key
    assert client.post("/api/v2/control/emergency_clear").status_code == 401
    assert client.post("/api/v2/tractor/clear-emergency").status_code == 401
    if prefixes in ("/api/v2", "/"):  # the whole API is protected: an ordinary route needs a key
        assert client.get("/api/v2/status").status_code == 401
        assert client.get("/api/v2/status", headers={"X-API-Key": "abc123"}).status_code == 200


def test_the_estop_exemption_holds_under_a_root_path(monkeypatch):
    monkeypatch.setenv("API_KEY_REQUIRED", "1")
    monkeypatch.setenv("API_KEY_PATH_PREFIXES", "/lawnberry")
    monkeypatch.setenv("API_KEY_SECRET", "abc123")
    client = TestClient(create_app_with_estops(), root_path="/lawnberry")
    # uvicorn puts the root path in front of scope["path"]
    assert client.post("/lawnberry/api/v2/tractor/emergency-stop").status_code == 200
    assert client.post("/lawnberry/api/v2/control/emergency_clear").status_code == 401


def test_a_wrong_key_on_an_estop_is_ignored_not_refused(monkeypatch):
    monkeypatch.setenv("API_KEY_REQUIRED", "1")
    monkeypatch.setenv("API_KEY_PATH_PREFIXES", "/api/v2")
    monkeypatch.setenv("API_KEY_SECRET", "abc123")
    client = TestClient(create_app_with_estops())
    reply = client.post(ESTOP_PATHS[0], headers={"X-API-Key": "wrong"})
    assert reply.status_code == 200
