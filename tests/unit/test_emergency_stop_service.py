"""General emergency stop and shutdown parking: every stop path is driven, and
the report never claims a stop that did not happen.

Before this, ``POST /control/emergency-stop`` set two Python flags and returned
``motors_stopped: true, blade_disabled: true``; ``/control/emergency`` only
reached the RoboHAT. Faults are injected at each path (PCA9685 write, relay,
RoboHAT firmware/serial, IBT-4 blade driver) and the paths are hung/absent.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from backend.src.api import rest as rest_module
from backend.src.core.globals import _client_emergency, _safety_state
from backend.src.models.tractor_control import EngineState
from backend.src.services import emergency_stop_service as ess
from backend.src.services.blade_service import BladeService
from backend.src.services.tractor_service import TractorControlService

GOOD_BLADE = {
    "latched": True,
    "off": True,
    "initialized": True,
    "hardware_backed": True,
    "error": None,
}


class Pca:
    last_error = None

    def __init__(self) -> None:
        self.writes: list[tuple[int, int]] = []
        self.fail: set[int] = set()

    def is_ready(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass

    async def set_pwm_us(self, channel: int, us: int) -> None:
        if channel in self.fail:
            raise OSError("bus fault")
        self.writes.append((channel, us))

    def last_us(self, channel: int):
        return next((us for ch, us in reversed(self.writes) if ch == channel), None)


class FakeRoboHAT:
    def __init__(self, drive="acked", blade="acked", raises=None, hang=False) -> None:
        self.running = True
        self.status = SimpleNamespace(serial_connected=True)
        self.drive, self.blade, self.raises, self.hang = drive, blade, raises, hang
        self.calls: list[str] = []
        self.latched = False

    async def emergency_stop_detail(self):
        self.calls.append("stop")
        if self.hang:
            await asyncio.sleep(30)
        if self.raises:
            raise self.raises
        return {"drive": self.drive, "blade": self.blade, "usb_control": True, "error": None}

    async def park(self):
        self.calls.append("park")
        return {"drive": self.drive, "blade": self.blade, "error": None}

    @property
    def emergency_latched(self) -> bool:
        return self.latched

    async def clear_emergency(self) -> bool:
        self.latched = False
        return True


class FakeBlade:
    def __init__(self, report=None) -> None:
        self.report = dict(report or GOOD_BLADE)
        self.calls: list[str] = []
        self.latched = False

    async def emergency_stop(self):
        self.calls.append("stop")
        return dict(self.report)

    async def park(self):
        self.calls.append("park")
        return {**self.report, "latched": False}

    @property
    def emergency_latched(self) -> bool:
        return self.latched

    async def clear_emergency(self) -> None:
        self.latched = False


def _tractor(pca: Pca | None = None, enabled: bool = True) -> TractorControlService:
    svc = TractorControlService(config={"enabled": enabled})
    svc._pca9685 = pca or Pca()
    return svc


@pytest.fixture(autouse=True)
def _isolate_global_estop_state():
    yield
    _safety_state["emergency_stop_active"] = False
    rest_module._emergency_until = 0.0
    _client_emergency.clear()


def wire(monkeypatch, tractor=None, robohat=None, blade=None) -> None:
    base = "backend.src.services"
    monkeypatch.setattr(f"{base}.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(f"{base}.robohat_service.get_robohat_service", lambda: robohat)
    monkeypatch.setattr(f"{base}.blade_service.get_blade_service", lambda: blade)


# ------------------------- every stop path is driven -------------------------


async def test_every_stop_path_is_invoked_and_confirmed(monkeypatch):
    pca, robohat, blade = Pca(), FakeRoboHAT(), FakeBlade()
    tractor = _tractor(pca)
    tractor.authorize()
    tractor.state.engine = EngineState.RUNNING
    await tractor.engage_blade(True)
    await tractor.set_levers(0.9, 0.9, duration_ms=2000)
    wire(monkeypatch, tractor, robohat, blade)

    report = await ess.execute_emergency_stop(source="test")

    assert report["emergency_stop_active"] is True
    assert report["motors_stopped"] and report["blade_disabled"] and report["complete"]
    assert report["faults"] == []
    # tractor: servos neutral on the wire, PTO relay off, latch set
    assert pca.last_us(0) == 1500 and pca.last_us(2) == 1500
    assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
    assert tractor.state.emergency_stop_active is True
    # RoboHAT and IBT-4 blade service were both actually told to stop
    assert robohat.calls == ["stop"] and blade.calls == ["stop"]
    assert report["subsystems"]["robohat"]["outputs"]["drive"]["status"] == "stopped"
    assert report["subsystems"]["tractor"]["status"] == "simulated"  # SIM_MODE servos/relays
    await tractor.shutdown()


@pytest.mark.parametrize(
    ("label", "build", "motors", "blade_ok", "fault_prefix"),
    [
        (
            "robohat serial/firmware raises",
            lambda: dict(robohat=FakeRoboHAT(raises=OSError("tty gone"))),
            False,
            False,
            "robohat",
        ),
        (
            "robohat firmware never echoes the stop",
            lambda: dict(robohat=FakeRoboHAT(drive="unacked", blade="unacked")),
            False,
            False,
            "robohat.drive",
        ),
        (
            "robohat serial write fails",
            lambda: dict(robohat=FakeRoboHAT(drive="send_failed")),
            False,
            True,
            "robohat.drive",
        ),
        (
            "tractor PCA9685 channel fails",
            lambda: dict(pca_fail={0}),
            False,
            True,
            "tractor.left_lever",
        ),
        (
            "tractor PTO relay will not release",
            lambda: dict(relay_fail=True),
            True,
            False,
            "tractor.blade_pto",
        ),
        (
            "IBT-4 driver refuses the off command",
            lambda: dict(blade=FakeBlade({**GOOD_BLADE, "off": False})),
            True,
            False,
            "blade_service",
        ),
        (
            "IBT-4 driver raised during the stop",
            lambda: dict(blade=FakeBlade({**GOOD_BLADE, "error": "OSError: gpio"})),
            True,
            False,
            "blade_service",
        ),
    ],
)
async def test_a_failed_path_is_reported_and_never_claimed_stopped(
    monkeypatch, label, build, motors, blade_ok, fault_prefix
):
    cfg = build()
    pca = Pca()
    pca.fail = cfg.get("pca_fail", set())
    tractor = _tractor(pca)
    if cfg.get("relay_fail"):
        monkeypatch.setattr(
            tractor.blade_pto, "set", lambda on: (_ for _ in ()).throw(OSError("relay stuck"))
        )
    robohat = cfg.get("robohat", FakeRoboHAT())
    blade = cfg.get("blade", FakeBlade())
    wire(monkeypatch, tractor, robohat, blade)

    report = await ess.execute_emergency_stop(source="test")

    assert report["emergency_stop_active"] is True, label  # the latch is always set
    assert report["motors_stopped"] is motors, label
    assert report["blade_disabled"] is blade_ok, label
    assert report["complete"] is False, label
    assert any(f.startswith(fault_prefix) for f in report["faults"]), (label, report["faults"])
    # Independence: one bad path never prevents the others from being driven.
    assert tractor.state.emergency_stop_active is True
    assert blade.calls == ["stop"] and robohat.calls == ["stop"]
    await tractor.shutdown()


async def test_a_hung_path_is_bounded_and_does_not_delay_or_skip_the_others(monkeypatch):
    pca, robohat, blade = Pca(), FakeRoboHAT(hang=True), FakeBlade()
    tractor = _tractor(pca)
    wire(monkeypatch, tractor, robohat, blade)

    started = time.monotonic()
    report = await ess.execute_emergency_stop(source="test", path_timeout_s=0.15)

    assert time.monotonic() - started < 1.0
    assert report["subsystems"]["robohat"]["status"] == "failed"
    assert "did not finish" in report["subsystems"]["robohat"]["error"]
    assert report["motors_stopped"] is False and report["complete"] is False
    assert tractor.state.emergency_stop_active is True and blade.calls == ["stop"]
    assert pca.last_us(0) == 1500
    await tractor.shutdown()


# ------------------- absent hardware is never "stopped" (outside SIM) -------------------


async def test_absent_hardware_outside_sim_is_not_reported_stopped(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    tractor = _tractor(enabled=False)  # the deployed default: tractor platform disabled
    wire(monkeypatch, tractor, None, BladeService())  # no RoboHAT, IBT-4 never initialised

    report = await ess.execute_emergency_stop(source="test")

    assert report["emergency_stop_active"] is True
    assert report["motors_stopped"] is False, "nothing drove or confirmed any motor stop"
    assert report["blade_disabled"] is False
    assert report["complete"] is False
    statuses = {k: v["status"] for k, v in report["subsystems"].items()}
    assert statuses == {
        "tractor": "not_enabled",
        "robohat": "not_present",
        "blade_service": "not_present",
    }
    assert tractor.state.emergency_stop_active is False, "a disabled tractor is not touched"


async def test_ibt4_flag_only_stub_is_not_a_confirmed_blade_stop_outside_sim(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    stub = BladeService()
    await stub.initialize()
    assert stub._driver.hardware_backed is False  # _GPIOAdapter only tracks a flag
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), stub)

    report = await ess.execute_emergency_stop(source="test")

    blade_out = report["subsystems"]["blade_service"]["outputs"]["blade"]
    assert blade_out["status"] == "unconfirmed" and "stub" in blade_out["detail"]
    assert report["blade_disabled"] is False, "RoboHAT blade=off alone must not hide the stub"
    assert report["motors_stopped"] is True
    assert stub.emergency_latched is True, "the blade latch is still set"


async def test_sim_mode_with_nothing_attached_is_simulated_not_failed(monkeypatch):
    wire(monkeypatch, _tractor(enabled=False), None, BladeService())
    report = await ess.execute_emergency_stop(source="test")
    assert report["motors_stopped"] and report["blade_disabled"] and report["complete"]
    assert {v["status"] for v in report["subsystems"].values()} == {"simulated"}


# ------------------------------- REST endpoints -------------------------------


def _client():
    from backend.src.main import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_rest_emergency_stop_alias_reports_the_truth(monkeypatch):
    pca = Pca()
    tractor = _tractor(pca)
    wire(monkeypatch, tractor, FakeRoboHAT(raises=OSError("tty gone")), FakeBlade())

    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency-stop")

    assert resp.status_code == 200
    body = resp.json()
    assert body["emergency_stop_active"] is True
    assert body["motors_stopped"] is False, "RoboHAT stop failed: must not say motors stopped"
    assert body["blade_disabled"] is False
    assert any("robohat" in f for f in body["faults"])
    assert "NOT every output" in body["remediation"]["message"]
    assert tractor.state.emergency_stop_active is True and pca.last_us(0) == 1500
    await tractor.shutdown()


async def test_rest_emergency_v2_works_without_any_session_even_with_auth_required(monkeypatch):
    """The E-stop must never be refused for a missing/expired manual session."""
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    robohat, blade = FakeRoboHAT(), FakeBlade()
    wire(monkeypatch, _tractor(), robohat, blade)

    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency", json={})

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status_reason"] == "EMERGENCY_STOP_TRIGGERED"
    assert body["emergency_stop"]["motors_stopped"] is True
    assert robohat.calls == ["stop"] and blade.calls == ["stop"]


async def test_rest_emergency_v2_flags_an_incomplete_stop(monkeypatch):
    wire(monkeypatch, _tractor(), FakeRoboHAT(drive="unacked"), FakeBlade())
    async with _client() as client:
        legacy = await client.post("/api/v2/control/emergency", json={"command": "emergency_stop"})
        v2 = await client.post("/api/v2/control/emergency", json={"session_id": "x"})

    assert legacy.status_code == 200 and legacy.json()["motors_stopped"] is False
    body = v2.json()
    assert v2.status_code == 202 and body["status_reason"] == "EMERGENCY_STOP_INCOMPLETE"
    assert "emergency_stop_unconfirmed" in body["active_interlocks"]
    assert body["telemetry_snapshot"]["status"] == "fault"


async def test_emergency_clear_requires_operator_auth_but_estop_does_not(monkeypatch):
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    wire(monkeypatch, _tractor(), FakeRoboHAT(), FakeBlade())
    clear_body = {"confirmation": True}

    async with _client() as client:
        assert (await client.post("/api/v2/control/emergency-stop")).status_code == 200
        assert _safety_state["emergency_stop_active"] is True

        denied = await client.post("/api/v2/control/emergency_clear", json=clear_body)
        assert denied.status_code == 401
        assert _safety_state["emergency_stop_active"] is True, (
            "an unauthenticated clear must not clear"
        )

        login = await client.post("/api/v2/auth/login", json={"credential": "operator"})
        token = login.json()["access_token"]
        ok = await client.post(
            "/api/v2/control/emergency_clear",
            json=clear_body,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert ok.status_code == 200 and ok.json()["status"] == "EMERGENCY_CLEARED"
    assert _safety_state["emergency_stop_active"] is False


async def test_general_clear_reports_a_tractor_latch_it_does_not_release(monkeypatch):
    tractor = _tractor()
    wire(monkeypatch, tractor, FakeRoboHAT(), BladeService())
    async with _client() as client:
        await client.post("/api/v2/control/emergency-stop")
        resp = await client.post("/api/v2/control/emergency_clear", json={"confirmation": True})

    assert resp.json()["still_latched"] == ["tractor_emergency_stop"]
    assert tractor.state.emergency_stop_active is True
    await tractor.shutdown()


# --------------------------------- shutdown park ---------------------------------


async def test_park_actuators_does_not_latch_an_estop(monkeypatch):
    pca, robohat, blade = Pca(), FakeRoboHAT(), FakeBlade()
    tractor = _tractor(pca)
    tractor.authorize()
    tractor.state.engine = EngineState.RUNNING
    await tractor.engage_blade(True)
    await tractor.set_levers(0.9, 0.9, duration_ms=2000)
    wire(monkeypatch, tractor, robohat, blade)

    report = await ess.park_actuators(source="test")

    assert report["complete"] is True
    assert pca.last_us(0) == 1500 and pca.last_us(2) == 1500 and pca.last_us(1) == 1000
    assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
    assert tractor.state.emergency_stop_active is False and tractor.state.authorized is False
    assert robohat.calls == ["park"] and blade.calls == ["park"]
    await tractor.shutdown()


async def test_lifespan_shutdown_parks_the_actuators_before_tearing_services_down(monkeypatch):
    """Real ``main.lifespan``: on shutdown the machine is commanded safe FIRST."""
    import backend.src.main as main

    order: list[str] = []
    pca, robohat, blade = Pca(), FakeRoboHAT(), FakeBlade()
    tractor = _tractor(pca)
    wire(monkeypatch, tractor, robohat, blade)
    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)

    async def noop(*_a, **_k):
        return None

    async def robohat_shutdown(*_a, **_k):
        order.append("robohat_shutdown")

    class _Stub:
        def __init__(self, *_a, **_k) -> None:
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

        async def initialize(self):
            pass

        async def start_streaming(self):
            pass

        async def shutdown(self):
            pass

    real_park = robohat.park

    async def recording_park():
        order.append("robohat_park")
        return await real_park()

    robohat.park = recording_park
    monkeypatch.setattr(main, "initialize_robohat_service", noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", robohat_shutdown)
    monkeypatch.setattr(main, "GPSDegradationMonitor", _Stub)
    monkeypatch.setattr(main, "Watchdog", _Stub)
    monkeypatch.setattr(main, "TractorSafetyMonitor", _Stub)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", noop)

    async with main.lifespan(main.app):
        tractor.authorize()
        tractor.state.engine = EngineState.RUNNING
        await tractor.engage_blade(True)
        await tractor.set_levers(0.9, 0.9, duration_ms=2000)  # mowing at shutdown time
        assert tractor.state.blade_engaged is True
        assert pca.last_us(0) == 1950

    assert pca.last_us(0) == 1500 and pca.last_us(2) == 1500, "levers must be parked neutral"
    assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
    assert tractor.state.authorized is False
    assert robohat.calls[-1] == "park" and blade.calls == ["park"]
    assert order == ["robohat_park", "robohat_shutdown"], "park must precede service teardown"
    await tractor.shutdown()
