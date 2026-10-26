"""Actuator-hardware health for the zero-turn mower (PCA9685 + relay GPIO).

Absent or failed hardware must surface as a *refused command and a reported
fault* -- never as a silent ``ok`` that leaves ``state`` claiming a lever moved
or a relay switched. SIM_MODE is the one exception: the simulated driver
succeeds so the rest of the stack stays testable.

These tests leave SIM_MODE (``real_mode``) and inject faults at the bottom of
the stack: a missing ``smbus2``/chip, a NAKing chip, a failing GPIO write, a
write that hangs.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

from backend.src.drivers.actuators import RelayActuator, RelayUnavailableError
from backend.src.models.tractor_control import EngineState
from backend.src.services.tractor_service import TractorControlService

# ------------------------------- fixtures ---------------------------------


@pytest.fixture
def real_mode(monkeypatch):
    """Leave SIM_MODE so drivers/relays behave as on the machine."""
    monkeypatch.setenv("SIM_MODE", "0")


def _no_smbus(monkeypatch) -> None:
    """No I2C library/bus: ``from smbus2 import SMBus`` raises ImportError."""
    monkeypatch.setitem(sys.modules, "smbus2", None)


class FakeBus:
    """Shared switchboard for the fake smbus2: flip ``fail_write`` mid-test."""

    def __init__(self) -> None:
        self.fail_write = False
        self.writes: list[tuple[int, list[int]]] = []


def _fake_smbus(monkeypatch) -> FakeBus:
    bus_state = FakeBus()

    class _SMBus:
        def __init__(self, bus):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_byte_data(self, addr, reg, val):
            pass

        def write_i2c_block_data(self, addr, reg, data):
            if bus_state.fail_write:
                raise OSError(121, "Remote I/O error")
            bus_state.writes.append((reg, list(data)))

    mod = types.ModuleType("smbus2")
    mod.SMBus = _SMBus
    monkeypatch.setitem(sys.modules, "smbus2", mod)
    return bus_state


class FakeGpio:
    """Shared switchboard for the fake gpiozero: devices + fault flags."""

    def __init__(self) -> None:
        self.devices: list = []
        self.fail_off = False


def _fake_gpiozero(monkeypatch) -> FakeGpio:
    gpio = FakeGpio()

    class _OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            self.pin = pin
            self.value = bool(initial_value)
            gpio.devices.append(self)

        def on(self):
            self.value = True

        def off(self):
            if gpio.fail_off:
                raise OSError("GPIO write failed")
            self.value = False

    mod = types.ModuleType("gpiozero")
    mod.OutputDevice = _OutputDevice
    monkeypatch.setitem(sys.modules, "gpiozero", mod)
    return gpio


def _no_gpiozero(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gpiozero", None)


def _enabled_service(**extra) -> TractorControlService:
    return TractorControlService(config={"enabled": True, **extra})


# --------------------- absent PCA9685: refuse and report ---------------------


@pytest.mark.asyncio
async def test_absent_pca9685_refuses_every_actuation_and_reports_the_fault(real_mode, monkeypatch):
    _no_smbus(monkeypatch)
    gpio = _fake_gpiozero(monkeypatch)
    svc = _enabled_service()
    await svc.initialize()
    svc.authorize()
    svc.state.engine = EngineState.RUNNING  # so only the hardware fault can reject

    for result in (
        await svc.set_levers(0.5, 0.5),
        await svc.set_left_lever(0.5),
        await svc.set_throttle(0.5),
        await svc.engage_blade(True),
        await svc.start_engine(),
    ):
        assert result["status"] == "rejected", result
        assert "actuator hardware not ready" in result["reason"]
        assert "not available" in result["reason"]

    # Nothing was claimed: state is untouched and no relay was energised.
    assert svc.state.left_lever == 0.0 and svc.state.right_lever == 0.0
    assert svc.state.throttle == 0.0 and svc.state.blade_engaged is False
    assert all(d.value is False for d in gpio.devices)

    state = svc.get_state()
    assert state.actuator_ok is False
    assert "PCA9685 not available" in state.actuator_fault
    assert state.simulated is False


@pytest.mark.asyncio
async def test_apply_does_not_report_applied_when_the_motion_parts_were_refused(
    real_mode, monkeypatch
):
    from backend.src.models.tractor_control import TractorCommand

    _no_smbus(monkeypatch)
    _fake_gpiozero(monkeypatch)
    svc = _enabled_service()
    await svc.initialize()
    svc.authorize()

    res = await svc.apply(TractorCommand(left_lever=0.5, right_lever=0.5, throttle=0.5))

    assert res["results"]["throttle"]["status"] == "rejected"
    assert res["results"]["levers"]["status"] == "rejected"
    # blade_engaged=False is the safe direction and always succeeds, so this is
    # "partial" -- never "applied" while the drive was refused.
    assert res["results"]["blade"]["status"] == "ok"
    assert res["status"] == "partial"
    assert res["moving"] is False


@pytest.mark.asyncio
async def test_initialize_survives_absent_hardware_so_safety_wiring_can_still_start(
    real_mode, monkeypatch
):
    """A dead PCA9685 *and* an unclaimable relay must not make initialize() raise:
    main.py would otherwise skip starting the watchdog and tilt monitor."""
    _no_smbus(monkeypatch)
    _no_gpiozero(monkeypatch)
    svc = _enabled_service()

    await svc.initialize()  # must not raise

    assert svc.initialized is True
    state = svc.get_state()
    assert state.actuator_ok is False and state.actuator_fault


# --------------------- failed hardware mid-operation ---------------------


@pytest.mark.asyncio
async def test_bus_fault_mid_operation_reaches_the_caller_and_the_state(real_mode, monkeypatch):
    bus = _fake_smbus(monkeypatch)
    _fake_gpiozero(monkeypatch)
    svc = _enabled_service()
    await svc.initialize()
    svc.authorize()
    assert svc.get_state().actuator_ok is True

    bus.fail_write = True
    with pytest.raises(OSError):
        await svc.set_levers(0.5, 0.5)
    # The write failed, so the lever must not be recorded as moved.
    assert svc.state.left_lever == 0.0
    state = svc.get_state()
    assert state.actuator_ok is False
    assert "Remote I/O error" in state.actuator_fault

    # The bus recovers: the next successful write clears the reported fault.
    bus.fail_write = False
    assert (await svc.set_levers(0.0, 0.0))["status"] == "ok"
    assert svc.get_state().actuator_ok is True
    assert svc.get_state().actuator_fault is None
    await svc.shutdown()


@pytest.mark.asyncio
async def test_a_hung_write_times_out_instead_of_blocking_forever(real_mode, monkeypatch):
    _fake_gpiozero(monkeypatch)
    svc = _enabled_service()

    class _HungDriver:
        last_error = None

        def is_ready(self) -> bool:
            return True

        async def initialize(self) -> None:
            pass

        async def set_pwm_us(self, channel: int, us: int) -> None:
            await asyncio.sleep(30)

    svc._pca9685 = _HungDriver()
    svc.ACTUATION_TIMEOUT_S = 0.05
    svc.authorize()

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await svc.set_levers(0.5, 0.5)
    assert time.monotonic() - started < 1.0
    assert "timed out" in svc.get_state().actuator_fault

    # E-stop with every servo hung: bounded, and honest about it.
    started = time.monotonic()
    result = await svc.emergency_stop()
    assert time.monotonic() - started < 2.0
    assert result["complete"] is False
    assert {n for n, s in result["subsystems"].items() if not s["ok"]} == {
        "left_lever",
        "right_lever",
        "throttle",
    }
    assert result["subsystems"]["blade_pto"]["ok"] is True
    assert svc.state.emergency_stop_active is True
    await svc.shutdown()


# --------------------- SIM_MODE keeps the stack testable ---------------------


@pytest.mark.asyncio
async def test_sim_mode_succeeds_and_says_it_is_simulated():
    svc = TractorControlService(config={})  # SIM_MODE=1 under the test runner
    state = svc.get_state()
    assert state.actuator_ok is True and state.simulated is True
    svc.authorize()  # a lever command from an unlabeled caller needs authorization (SAF-01)
    assert (await svc.set_levers(0.5, 0.5))["status"] == "ok"
    assert svc.state.left_lever == pytest.approx(0.5)
    await svc.shutdown()


# ----------------------------- relay GPIO -----------------------------


def test_relay_without_gpio_library_raises_instead_of_pretending(real_mode, monkeypatch):
    _no_gpiozero(monkeypatch)
    relay = RelayActuator("blade_pto", gpio=26)

    with pytest.raises(RelayUnavailableError):
        relay.set(True)

    assert relay.state is False  # never claims a switch that did not happen
    assert relay.simulated is False
    assert "GPIO26" in relay.fault


def test_relay_write_failure_keeps_the_old_state_and_reports_the_fault(real_mode, monkeypatch):
    gpio = _fake_gpiozero(monkeypatch)
    relay = RelayActuator("blade_pto", gpio=26)
    relay.set(True)
    assert relay.state is True and gpio.devices[0].value is True

    gpio.fail_off = True
    with pytest.raises(RelayUnavailableError):
        relay.set(False)

    assert relay.state is True  # still energised, and the state says so
    assert "write failed" in relay.fault
    gpio.fail_off = False
    relay.set(False)
    assert relay.state is False and relay.fault is None


@pytest.mark.asyncio
async def test_relay_in_sim_mode_only_tracks_state():
    relay = RelayActuator("starter", gpio=5)
    assert relay.simulated is True
    relay.set(True)
    assert relay.state is True
    await relay.pulse(1)
    assert relay.state is False


@pytest.mark.asyncio
async def test_blade_relay_that_cannot_switch_off_is_not_reported_off(real_mode, monkeypatch):
    """The dangerous case: the PTO clutch relay will not release."""
    _fake_smbus(monkeypatch)
    gpio = _fake_gpiozero(monkeypatch)
    svc = _enabled_service()
    await svc.initialize()
    svc.authorize()
    svc.state.engine = EngineState.RUNNING
    assert (await svc.engage_blade(True))["status"] == "ok"
    assert svc.state.blade_engaged is True

    gpio.fail_off = True
    with pytest.raises(RelayUnavailableError):
        await svc.engage_blade(False)
    assert svc.state.blade_engaged is True, "must not claim the blade is off"

    result = await svc.emergency_stop()
    assert result["subsystems"]["blade_pto"]["ok"] is False
    assert result["complete"] is False
    assert any(f.startswith("blade_pto:") for f in result["faults"])
    assert svc.state.emergency_stop_active is True
    assert svc.state.blade_engaged is True, "the stop failed: state must still say so"
    assert "GPIO26" in svc.get_state().actuator_fault
    await svc.shutdown()


@pytest.mark.asyncio
async def test_starter_relay_fault_aborts_the_start_and_is_reported(real_mode, monkeypatch):
    _fake_smbus(monkeypatch)
    _no_gpiozero(monkeypatch)
    svc = _enabled_service()
    await svc.initialize()
    svc.authorize()

    result = await svc.start_engine()

    assert result["status"] == "rejected"
    assert "starter relay fault" in result["reason"]
    assert svc.state.engine == EngineState.OFF
    await svc.shutdown()
