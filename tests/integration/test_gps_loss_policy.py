"""GPS-loss behaviour that exists today: loss reaches navigation, manual control stays available.

FR-002 (``.specify/out/spec_005.md``) asks for dead reckoning for at most two minutes at reduced
speed with stricter obstacle thresholds, then stop and alert. None of that is implemented: there
is no grace-period timer, no GPS-aware speed cap (``control_drive_v2`` and
``NavigationService.max_speed`` ignore GPS state), no GPS-dependent obstacle threshold and no GPS
alert, so those scenarios are not tested here. What does exist is covered elsewhere and is not
repeated:

* dead reckoning switching on without a fix, off with one, bounded drift:
  ``tests/integration/test_dead_reckoning.py``;
* AUTONOMOUS -> MANUAL on poor accuracy or a stale fix (``GPSDegradationMonitor``):
  ``tests/unit/test_gps_degradation.py``.

The test below drives the production wiring between them: the telemetry hub reads the sensors, feeds
``NavigationService``, and the operator keeps a working, audited, authenticated manual control path
while the GPS is gone.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
import pytest_asyncio

from backend.src.core.persistence import persistence
from backend.src.main import app
from backend.src.models import GpsReading, ImuReading, SensorData
from backend.src.services.navigation_service import NavigationService
from backend.src.services.websocket_hub import websocket_hub

# A test-local value, set through the same variable operators use; not a shipped default.
MANUAL_PASSWORD = "gps-loss-test-" + uuid.uuid4().hex

FIX = {"latitude": 40.7128, "longitude": -74.0060}


class StubSensorManager:
    """The hub's sensor source. ``gps = None`` is a receiver that has lost its fix."""

    initialized = True
    ultrasonic = None
    stereo_camera = None

    def __init__(self) -> None:
        self.gps: GpsReading | None = self.with_fix()
        self.imu = ImuReading(yaw=90.0)

    @staticmethod
    def with_fix() -> GpsReading:
        return GpsReading(**FIX, altitude=10.0, accuracy=0.6, satellites=18, rtk_status="RTK_FIXED")

    async def read_all_sensors(self) -> SensorData:
        return SensorData(gps=self.gps, imu=self.imu)


@pytest.fixture
def sensors(monkeypatch) -> tuple[StubSensorManager, NavigationService]:
    """Route telemetry through a stub sensor manager and a private navigation service."""
    manager = StubSensorManager()
    nav = NavigationService()
    monkeypatch.setattr(websocket_hub, "_sensor_manager", manager)
    monkeypatch.setattr(websocket_hub, "_gps_warm_done", True)
    monkeypatch.setattr(NavigationService, "_instance", nav)
    return manager, nav


@pytest.fixture
def production_auth(tmp_path, monkeypatch):
    """Operator auth on, as deployed; the manual-unlock password is a value private to this test."""
    monkeypatch.setenv("LAWNBERRY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.setenv("MANUAL_CONTROL_PASSWORD", MANUAL_PASSWORD)


@pytest_asyncio.fixture
async def client(production_auth):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.mark.asyncio
async def test_manual_override_during_gps_loss(client, sensors, operator_credential):
    """With the fix lost: nav falls back to dead reckoning and the operator can still drive."""
    manager, nav = sensors

    async def position() -> dict:
        resp = await client.get("/api/v2/dashboard/telemetry")
        assert resp.status_code == 200, resp.text
        return resp.json()["position"]

    # Healthy: the fix is what the dashboard shows and navigation uses.
    shown = await position()
    assert (shown["latitude"], shown["longitude"]) == (FIX["latitude"], FIX["longitude"])
    assert nav.navigation_state.dead_reckoning_active is False

    # The receiver loses its fix: no stale coordinates are shown; navigation reports dead reckoning.
    manager.gps = None
    lost = await position()
    assert lost["latitude"] is None and lost["longitude"] is None
    assert nav.navigation_state.dead_reckoning_active is True
    assert nav.navigation_state.current_position is not None, "dead reckoning keeps a position"

    # The operator logs in, unlocks manual control (wrong password refused) and drives. The
    # unlock is a step-up: it needs the login's bearer token as well as the password.
    login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
    assert login.status_code == 200, login.text
    auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
    wrong = await client.post(
        "/api/v2/control/manual-unlock",
        json={"method": "password", "password": "not-it"},
        headers=auth,
    )
    assert wrong.status_code == 401
    unlock = await client.post(
        "/api/v2/control/manual-unlock",
        json={"method": "password", "password": MANUAL_PASSWORD},
        headers=auth,
    )
    assert unlock.status_code == 200 and unlock.json()["authorized"] is True
    session_id = unlock.json()["session_id"]

    reason = f"gps-loss-override-{uuid.uuid4().hex}"
    command = {
        "session_id": session_id,
        "vector": {"linear": 0.5, "angular": 0.1},
        "duration_ms": 500,
        "reason": reason,
    }
    drive = await client.post("/api/v2/control/drive", json=command, headers=auth)

    assert drive.status_code == 202, drive.text
    body = drive.json()
    assert body["accepted"] is True and body["result"] in {"queued", "accepted"}
    assert body["active_interlocks"] == []
    # a GPS-loss policy may lower the cap, but never to zero and never above the default
    assert 0.0 < body["telemetry_snapshot"]["speed_limit"] <= 0.8

    # ... and the override is on the audit trail with the unlocking principal, session masked.
    trail = [
        entry
        for entry in persistence.load_audit_logs(limit=500)
        if entry["action"] == "control.drive.v2"
        and (entry["details"].get("command") or {}).get("reason") == reason
    ]
    assert len(trail) == 1
    audited = trail[0]["details"]
    assert audited["command"]["session_id"] == "***"
    assert audited["command"]["principal"] == unlock.json()["principal"]
    assert audited["response"]["accepted"] is True

    # Losing GPS does not open the control path: no/invalid session is still refused.
    refused = await client.post(
        "/api/v2/control/drive", json={**command, "session_id": "forged"}, headers=auth
    )
    assert refused.status_code == 401
    # ... and a valid manual session without the operator's login is refused as well.
    assert (await client.post("/api/v2/control/drive", json=command)).status_code == 401

    # The fix returns: dead reckoning stops and the dashboard shows the fix again.
    manager.gps = StubSensorManager.with_fix()
    back = await position()
    assert (back["latitude"], back["longitude"]) == (FIX["latitude"], FIX["longitude"])
    assert nav.navigation_state.dead_reckoning_active is False
