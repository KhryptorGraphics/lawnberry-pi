"""A disabled tractor platform is never touched outside SIM_MODE (audit SF-05).

With the shipped ``enabled: false`` the *unauthenticated* ``POST /tractor/emergency-stop`` used to
claim and drive GPIO26 and, since the starter-off was added to the safing path, GPIO5 (the HaLow
HAT's RESET_N, audit H-02b). A disabled platform owns no pin: E-stop and park answer
``not_enabled`` and the relays are built unarmed, so not even the authenticated blade-off /
stop-engine routes can claim one. This is the conservative default pending owner decision O-5
(the pin moves themselves); in SIM_MODE no GPIO exists and a disabled service stays a simulated
tractor so the stack remains testable.
"""

from __future__ import annotations

import httpx
import pytest
from _tractor_fakes import FakeGpiozero, StallablePCA9685

from backend.src.api.routers import tractor as tractor_router
from backend.src.drivers.actuators import RelayActuator, RelayUnavailableError
from backend.src.services.tractor_service import TractorControlService


@pytest.fixture
def real_mode(monkeypatch):
    """Leave SIM_MODE and record every GPIO claim and write."""
    monkeypatch.setenv("SIM_MODE", "0")
    return FakeGpiozero(monkeypatch)


async def test_a_disabled_tractor_estop_claims_no_gpio_outside_sim(real_mode):
    svc = TractorControlService(config={"enabled": False})

    result = await svc.emergency_stop()

    assert result["status"] == "not_enabled" and result["complete"] is False
    assert "not enabled" in result["faults"][0]
    assert real_mode.devices == [], "no pin may be claimed or driven for a disabled platform"
    assert svc.state.emergency_stop_active is False


async def test_a_disabled_tractor_park_claims_no_gpio(real_mode):
    svc = TractorControlService(config={"enabled": False})

    result = await svc.park()

    assert result["status"] == "not_enabled"
    assert real_mode.devices == []
    assert svc.parked is False, "nothing was parked, so nothing is latched"


async def test_the_http_estop_on_a_disabled_tractor_claims_no_gpio(real_mode, monkeypatch):
    from backend.src.main import app

    svc = TractorControlService(config={"enabled": False})
    monkeypatch.setattr(tractor_router, "get_tractor_service", lambda: svc)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        stop = await c.post("/api/v2/tractor/emergency-stop")
        blade_off = await c.post("/api/v2/tractor/blade", json={"engaged": False})
        stop_engine = await c.post("/api/v2/tractor/stop-engine")
        lever = await c.post("/api/v2/tractor/left-lever", json={"value": 0.0})

    assert stop.status_code == 200 and stop.json()["status"] == "not_enabled"
    for refused in (blade_off, stop_engine, lever):
        body = refused.json()
        assert refused.status_code == 200 and body["status"] == "rejected"
        assert "not enabled" in body["reason"]
    assert real_mode.devices == [], "even the safe-direction writes must not claim a pin"


def test_unarmed_relays_never_claim_a_pin(real_mode):
    relay = RelayActuator("starter", 5, armed=False)

    with pytest.raises(RelayUnavailableError, match="not armed"):
        relay.set(False)

    assert real_mode.devices == [] and relay.state is False


async def test_an_enabled_tractor_still_drives_both_relays(real_mode):
    """The guard must not be over-broad: an enabled platform claims and drives its pins."""
    svc = TractorControlService(config={"enabled": True})
    svc._pca9685 = StallablePCA9685()

    result = await svc.emergency_stop()

    assert result["status"] == "emergency_stop" and result["complete"] is True
    assert real_mode.claimed_pins == [5, 26]
    assert real_mode.last(26) == "off" and real_mode.last(5) == "off"


async def test_sim_mode_keeps_a_disabled_service_a_simulated_tractor():
    svc = TractorControlService(config={"enabled": False})  # SIM_MODE=1 under the test runner

    result = await svc.emergency_stop()

    assert result["status"] == "emergency_stop" and result["complete"] is True
    assert all(out["simulated"] for out in result["subsystems"].values())
