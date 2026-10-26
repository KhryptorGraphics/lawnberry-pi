"""General emergency stop and shutdown parking: every stop path is driven, and
the report never claims a stop that did not happen.

Before this, ``POST /control/emergency-stop`` set two Python flags and returned
``motors_stopped: true, blade_disabled: true``; ``/control/emergency`` only
reached the RoboHAT. Faults are injected at each path (PCA9685 write, relay,
RoboHAT firmware/serial, IBT-4 blade driver) and the paths are hung/absent.
"""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace

import httpx
import pytest
from fake_lgpio import FakeLgpio

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

    async def verify_pwm_us(self, channel: int, us: int) -> None:
        if self.last_us(channel) != us:
            raise OSError("read-back mismatch")

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


# -------------- IBT-4 through the real lgpio adapter (a FAKE lgpio, never hardware) --------------


def _armed_blade(monkeypatch, fake: FakeLgpio) -> BladeService:
    """A real BladeService armed through its config; ``import lgpio`` in setup() yields the fake."""
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    return BladeService({"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}})


async def _armed_and_on(monkeypatch, fake: FakeLgpio) -> BladeService:
    blade = _armed_blade(monkeypatch, fake)
    assert await blade.initialize() is True
    assert await blade.set_active(True) is True
    return blade


async def test_armed_ibt4_confirmed_off_is_stopped_and_says_what_was_confirmed(monkeypatch):
    fake = FakeLgpio()
    blade = await _armed_and_on(monkeypatch, fake)
    assert fake.pad == {24: 1, 25: 0}
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.execute_emergency_stop(source="test")

    assert report["subsystems"]["blade_service"]["outputs"]["blade"] == {
        "status": "stopped",
        "detail": "GPIO24/25 driven low and read back low; "
        "the IBT-4 power stage and the blade motor are not observed",
    }
    assert fake.pad == {24: 0, 25: 0}, "the wire really went low"
    assert report["blade_disabled"] is True and report["complete"] is True
    assert report["faults"] == []
    assert blade.emergency_latched is True


async def test_armed_ibt4_stuck_pin_is_failed_and_blade_disabled_is_false(monkeypatch):
    fake = FakeLgpio()
    blade = await _armed_and_on(monkeypatch, fake)
    fake.hold_pad(24, 1)  # the write returns 0 but the pad stays high
    robohat = FakeRoboHAT()
    wire(monkeypatch, _tractor(enabled=False), robohat, blade)

    report = await ess.execute_emergency_stop(source="test")

    path = report["subsystems"]["blade_service"]
    out = path["outputs"]["blade"]
    assert path["status"] == "failed" and out["status"] == "failed"
    assert "GPIO24 reads back 1, expected 0" in out["error"]
    assert report["blade_disabled"] is False and report["complete"] is False
    assert any(f.startswith("blade_service.blade: failed") for f in report["faults"])
    assert blade.emergency_latched is True, "the latch holds although the output is unconfirmed"
    assert robohat.calls == ["stop"] and report["motors_stopped"] is True, "other paths still run"


async def test_armed_ibt4_write_that_raises_is_failed_and_both_inputs_are_still_driven(monkeypatch):
    fake = FakeLgpio()
    blade = await _armed_and_on(monkeypatch, fake)
    fake.fail_write(24, "raise")
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.execute_emergency_stop(source="test")

    out = report["subsystems"]["blade_service"]["outputs"]["blade"]
    assert out["status"] == "failed" and "GPIO24 write(0) raised" in out["error"]
    assert report["blade_disabled"] is False
    assert fake.writes()[-4:] == [(24, 0), (25, 0), (24, 0), (25, 0)], (
        "IN2 is driven even though IN1 raised, and the stop retried once"
    )


async def test_armed_ibt4_setup_failure_is_failed_not_absent(monkeypatch):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")  # the HaLow HAT's IRQ line on the deployed Pi (audit C-03)
    blade = _armed_blade(monkeypatch, fake)
    assert await blade.initialize() is False
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.execute_emergency_stop(source="test")

    path = report["subsystems"]["blade_service"]
    assert path["status"] == "failed", "attempted and failed is not 'nothing to stop'"
    assert not path.get("absent")
    out = path["outputs"]["blade"]
    assert "GPIO25" in out["error"] and "GPIO busy" in out["error"]
    assert report["blade_disabled"] is False and report["complete"] is False
    assert fake.writes() == [], "nothing was ever driven"
    assert blade.emergency_latched is True


# An ARMED service that was never initialised (a restarted backend: nothing in this process has
# claimed the lines, so their state is unknown) is claimed, driven low and read back BY the stop.

_STOPPED_DETAIL = (
    "GPIO24/25 driven low and read back low; "
    "the IBT-4 power stage and the blade motor are not observed"
)
_CLAIM_THEN_OFF = [
    ("gpiochip_open", 4),
    ("gpio_claim_output", 7, 24, 0),  # claimed already low ...
    ("gpio_claim_output", 7, 25, 0),
    ("gpio_read", 7, 24),  # ... and read back low
    ("gpio_read", 7, 25),
    ("gpio_write", 7, 24, 0),  # then the stop itself: both inputs written low ...
    ("gpio_write", 7, 25, 0),
    ("gpio_read", 7, 24),  # ... and read back
    ("gpio_read", 7, 25),
]


async def _stop_or_park(kind: str) -> dict:
    if kind == "stop":
        return await ess.execute_emergency_stop(source="test")
    return await ess.park_actuators(source="test")


async def test_armed_ibt4_never_initialised_is_claimed_driven_low_and_confirmed_by_the_stop(
    monkeypatch,
):
    fake = FakeLgpio()
    blade = _armed_blade(monkeypatch, fake)  # never initialize()d in this process
    assert fake.calls == []
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.execute_emergency_stop(source="test")

    assert report["subsystems"]["blade_service"]["outputs"]["blade"] == {
        "status": "stopped",
        "detail": _STOPPED_DETAIL,
    }
    assert fake.calls == _CLAIM_THEN_OFF
    assert fake.pad == {24: 0, 25: 0}
    assert report["blade_disabled"] is True and report["complete"] is True
    assert report["faults"] == []
    assert blade.emergency_latched is True

    # The lines are held from now on: a second stop only drives and reads, it never re-claims.
    again = await ess.execute_emergency_stop(source="test")
    assert again["subsystems"]["blade_service"]["outputs"]["blade"]["status"] == "stopped"
    assert fake.names().count("gpio_claim_output") == 2
    assert fake.names().count("gpiochip_open") == 1


async def test_armed_ibt4_never_initialised_with_gpio25_busy_is_failed_and_gpio24_is_released(
    monkeypatch,
):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")  # the HaLow HAT's IRQ line on the deployed Pi (audit C-03)
    blade = _armed_blade(monkeypatch, fake)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.execute_emergency_stop(source="test")

    path = report["subsystems"]["blade_service"]
    assert path["status"] == "failed" and not path.get("absent")
    out = path["outputs"]["blade"]
    assert out["status"] == "failed"
    assert "GPIO25" in out["error"] and "GPIO busy" in out["error"]
    assert fake.freed == [24] and fake.closed == [7] and fake.claimed == set()
    assert fake.writes() == [], "nothing was driven: the lines could not be claimed"
    assert report["blade_disabled"] is False and report["complete"] is False
    assert blade.emergency_latched is True, "the latch is set although the lines were not claimed"


async def test_every_stop_retries_a_failed_claim_until_the_lines_are_ours(monkeypatch):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    blade = _armed_blade(monkeypatch, fake)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)
    first = await ess.execute_emergency_stop(source="test")
    assert first["subsystems"]["blade_service"]["status"] == "failed"

    fake.clear_claim_fault(25)  # the owner moved the pin
    second = await ess.execute_emergency_stop(source="test")

    assert second["subsystems"]["blade_service"]["outputs"]["blade"]["status"] == "stopped"
    assert second["blade_disabled"] is True
    assert fake.pad == {24: 0, 25: 0}


async def test_park_with_an_armed_never_initialised_ibt4_claims_drives_low_and_latches_nothing(
    monkeypatch,
):
    fake = FakeLgpio()
    blade = _armed_blade(monkeypatch, fake)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.park_actuators(source="test")

    assert report["subsystems"]["blade_service"]["outputs"]["blade"] == {
        "status": "stopped",
        "detail": _STOPPED_DETAIL,
    }
    assert fake.calls == _CLAIM_THEN_OFF
    assert report["complete"] is True
    assert blade.emergency_latched is False, "a park is not an E-stop"


async def test_park_with_an_armed_never_initialised_ibt4_and_gpio25_busy_is_failed(monkeypatch):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    blade = _armed_blade(monkeypatch, fake)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.park_actuators(source="test")

    path = report["subsystems"]["blade_service"]
    assert path["status"] == "failed" and "GPIO busy" in path["outputs"]["blade"]["error"]
    assert fake.freed == [24] and fake.closed == [7] and fake.writes() == []
    assert report["complete"] is False
    assert blade.emergency_latched is False


@pytest.mark.parametrize("kind", ["stop", "park"])
@pytest.mark.parametrize("config", [None, {"gpio_enabled": False, "pins": {"in1": 24, "in2": 25}}])
async def test_unarmed_ibt4_never_initialised_is_unchanged_and_touches_no_lgpio(
    monkeypatch, kind, config
):
    monkeypatch.setenv("SIM_MODE", "0")
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)  # importable, so any touch would be recorded
    blade = BladeService(config)  # the shipped default: the gate is off
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await _stop_or_park(kind)

    assert report["subsystems"]["blade_service"] == {
        "status": "not_present",
        "outputs": {},
        "detail": "IBT-4 blade service never initialised (nothing to switch off)",
        "absent": True,
    }
    assert fake.calls == [], "an unarmed service never reaches lgpio"
    assert blade._initialized is False and blade._driver.initialized is False
    assert blade.emergency_latched is (kind == "stop")


async def test_two_concurrent_stops_claim_the_lines_once(monkeypatch):
    fake = FakeLgpio()
    blade = _armed_blade(monkeypatch, fake)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)
    entered = 0
    real_initialize = blade._driver.initialize

    async def initialize_that_yields():
        nonlocal entered
        entered += 1
        await asyncio.sleep(0)  # the other stop gets to run while this one is mid-initialise
        await real_initialize()

    monkeypatch.setattr(blade._driver, "initialize", initialize_that_yields)

    first, second = await asyncio.gather(
        ess.execute_emergency_stop(source="a"), ess.execute_emergency_stop(source="b")
    )

    assert entered == 1, "the service lock must keep the second stop out of initialize()"
    assert fake.names().count("gpiochip_open") == 1
    assert fake.names().count("gpio_claim_output") == 2, "one claim per line, not two"
    for report in (first, second):
        assert report["subsystems"]["blade_service"]["outputs"]["blade"]["status"] == "stopped"
        assert report["blade_disabled"] is True
    assert fake.pad == {24: 0, 25: 0}


@pytest.mark.parametrize("kind", ["stop", "park"])
async def test_a_claim_attempt_that_raises_never_escapes_the_stop_and_is_reported_failed(
    monkeypatch, kind
):
    fake = FakeLgpio()
    blade = _armed_blade(monkeypatch, fake)

    async def exploding_initialize():
        raise RuntimeError("arming blew up")

    monkeypatch.setattr(blade, "initialize", exploding_initialize)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await _stop_or_park(kind)

    path = report["subsystems"]["blade_service"]
    assert path["status"] == "failed" and "arming blew up" in path["outputs"]["blade"]["error"]
    assert fake.calls == []
    assert blade.emergency_latched is (kind == "stop")


async def test_ibt4_initialised_stub_in_sim_mode_is_simulated(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "1")
    stub = BladeService()
    await stub.initialize()
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), stub)

    report = await ess.execute_emergency_stop(source="test")

    assert report["subsystems"]["blade_service"]["outputs"]["blade"] == {"status": "simulated"}
    assert report["blade_disabled"] is True


async def test_park_with_an_armed_ibt4_reports_the_confirmed_off_and_latches_nothing(monkeypatch):
    fake = FakeLgpio()
    blade = await _armed_and_on(monkeypatch, fake)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    report = await ess.park_actuators(source="test")

    out = report["subsystems"]["blade_service"]["outputs"]["blade"]
    assert out["status"] == "stopped" and out["detail"].startswith("GPIO24/25 driven low")
    assert fake.pad == {24: 0, 25: 0}
    assert blade.emergency_latched is False


async def test_rest_emergency_stop_does_not_claim_the_blade_disabled_for_a_stuck_ibt4(monkeypatch):
    fake = FakeLgpio()
    blade = await _armed_and_on(monkeypatch, fake)
    fake.hold_pad(24, 1)
    wire(monkeypatch, _tractor(enabled=False), FakeRoboHAT(), blade)

    async with _client() as client:
        resp = await client.post("/api/v2/control/emergency-stop")

    body = resp.json()
    assert resp.status_code == 200 and body["emergency_stop_active"] is True
    assert body["blade_disabled"] is False, "the IBT-4 output was NOT confirmed off"
    assert any(f.startswith("blade_service.blade: failed") for f in body["faults"])
    assert "NOT every output" in body["remediation"]["message"]


_BLADE_REPORT = {
    "latched": True,
    "off": True,
    "initialized": True,
    "hardware_backed": True,
    "state_confirmed": True,
    "gpio_error": None,
    "pins": {"in1": 24, "in2": 25},
    "error": None,
}


@pytest.mark.parametrize(
    ("label", "sim", "overrides", "status", "needle"),
    [
        ("confirmed hardware off", "0", {}, "stopped", "GPIO24/25 driven low and read back low"),
        (
            "confirmed hardware off, pins not reported",
            "0",
            {"pins": None},
            "stopped",
            "the IBT-4 inputs driven low and read back low",
        ),
        (
            "initialised, off not confirmed",
            "0",
            {"off": False, "state_confirmed": False, "gpio_error": "IBT4GpioError: GPIO24 stuck"},
            "failed",
            "GPIO24 stuck",
        ),
        ("off not confirmed, no text", "0", {"off": False}, "failed", "refused the off command"),
        (
            "a service error wins over the driver text",
            "0",
            {"off": False, "error": "RuntimeError: boom", "gpio_error": "older"},
            "failed",
            "boom",
        ),
        (
            "setup failed",
            "0",
            {
                "initialized": False,
                "off": False,
                "hardware_backed": False,
                "gpio_error": "IBT4GpioError: GPIO25 busy",
            },
            "failed",
            "GPIO25 busy",
        ),
        (
            "never initialised",
            "0",
            {"initialized": False, "off": False, "hardware_backed": False},
            "not_present",
            "never initialised",
        ),
        ("flag-only stub outside SIM", "0", {"hardware_backed": False}, "unconfirmed", "stub"),
        ("flag-only stub in SIM", "1", {"hardware_backed": False}, "simulated", None),
        (
            "never initialised in SIM",
            "1",
            {"initialized": False, "off": False, "hardware_backed": False},
            "simulated",
            None,
        ),
    ],
)
def test_blade_service_path_maps_a_report_to_exactly_one_status(
    monkeypatch, label, sim, overrides, status, needle
):
    monkeypatch.setenv("SIM_MODE", sim)

    path = ess._blade_service_path({**_BLADE_REPORT, **overrides})

    assert path["status"] == status, (label, path)
    if needle is not None:
        assert needle in repr(path), (label, path)


def test_a_blade_report_without_the_newer_keys_still_maps(monkeypatch):
    """``FakeBlade`` reports (and any older caller) carry no gpio_error/pins/state_confirmed."""
    monkeypatch.setenv("SIM_MODE", "0")
    path = ess._blade_service_path(dict(GOOD_BLADE))
    assert path["status"] == "stopped"
    assert "read back low" in path["outputs"]["blade"]["detail"]


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


async def test_emergency_clear_requires_operator_auth_but_estop_does_not(
    monkeypatch, operator_credential
):
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

        login = await client.post("/api/v2/auth/login", json={"credential": operator_credential})
        token = login.json()["access_token"]
        ok = await client.post(
            "/api/v2/control/emergency_clear",
            json=clear_body,
            headers={"Authorization": f"Bearer {token}"},
        )

    # The tractor E-stop the stop just latched is NOT released by the general clear, so the
    # answer must say "partially cleared" and name it (audit SEC-07).
    assert ok.status_code == 200 and ok.json()["status"] == "EMERGENCY_PARTIALLY_CLEARED"
    assert ok.json()["still_latched"] == ["tractor_emergency_stop"]
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
        # A human at the machine (source "operator"): not limited by the obstacle guard that
        # the real lifespan attaches (no sensors in this test, so autonomy would be held).
        await tractor.engage_blade(True, source="operator")
        await tractor.set_levers(0.9, 0.9, source="operator", duration_ms=2000)
        assert tractor.state.blade_engaged is True
        assert pca.last_us(0) == 1950

    assert pca.last_us(0) == 1500 and pca.last_us(2) == 1500, "levers must be parked neutral"
    assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
    assert tractor.state.authorized is False
    assert robohat.calls[-1] == "park" and blade.calls == ["park"]
    assert order == ["robohat_park", "robohat_shutdown"], "park must precede service teardown"
    await tractor.shutdown()
