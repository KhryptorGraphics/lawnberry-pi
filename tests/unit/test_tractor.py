"""Unit tests for the ride-on lawn tractor actuation + safety interlocks."""

import pytest

from backend.src.drivers.actuators import (
    GearActuator,
    GearCalibration,
    RelayActuator,
    ServoActuator,
    ServoCalibration,
)
from backend.src.models.action_prediction import ActionPrediction
from backend.src.models.tractor_control import (
    EngineState,
    TractorCommand,
    Transmission,
)
from backend.src.services.tractor_service import TractorControlService

# --------------------------- actuator mapping ---------------------------


def test_servo_bidirectional_mapping():
    s = ServoActuator(
        "steer", ServoCalibration(channel=1, us_min=1000, us_center=1500, us_max=2000)
    )
    assert s.microseconds(0.0) == 1500
    assert s.microseconds(1.0) == 2000
    assert s.microseconds(-1.0) == 1000
    assert s.microseconds(0.5) == 1750
    assert s.microseconds(5.0) == 2000  # clamped


def test_servo_unidirectional_mapping():
    t = ServoActuator(
        "thr", ServoCalibration(channel=2, us_min=1100, us_max=1900, bidirectional=False)
    )
    assert t.microseconds(0.0) == 1100
    assert t.microseconds(1.0) == 1900
    assert t.microseconds(0.5) == 1500


def test_gear_mapping():
    g = GearActuator(
        "gear", GearCalibration(channel=5, us_forward=1900, us_neutral=1500, us_reverse=1100)
    )
    assert g.microseconds("forward") == 1900
    assert g.microseconds("neutral") == 1500
    assert g.microseconds("reverse") == 1100


@pytest.mark.asyncio
async def test_relay_pulse_returns_off():
    r = RelayActuator("starter", gpio=5)
    assert r.state is False
    await r.pulse(1)
    assert r.state is False  # momentary: off after the pulse


# ----------------------------- interlocks -------------------------------


def _svc() -> TractorControlService:
    return TractorControlService(config={})  # defaults; safe initial state


@pytest.mark.asyncio
async def test_start_requires_authorization():
    s = _svc()
    r = await s.start_engine()
    assert r["status"] == "rejected" and "authorized" in r["reason"]
    s.authorize()
    r = await s.start_engine()
    assert r["status"] == "ok"
    assert s.state.engine == EngineState.RUNNING


@pytest.mark.asyncio
async def test_start_rejected_when_not_neutral():
    s = _svc()
    s.authorize()
    await s.set_gear(Transmission.FORWARD)
    r = await s.start_engine()
    assert r["status"] == "rejected" and "neutral" in r["reason"]


@pytest.mark.asyncio
async def test_start_rejected_when_clutch_released():
    s = _svc()
    s.authorize()
    await s.set_clutch(0.0)  # released
    r = await s.start_engine()
    assert r["status"] == "rejected" and "clutch" in r["reason"]


@pytest.mark.asyncio
async def test_blade_requires_engine_running():
    s = _svc()
    s.authorize()
    assert (await s.engage_blade(True))["status"] == "rejected"
    await s.start_engine()
    r = await s.engage_blade(True)
    assert r["status"] == "ok" and s.state.blade_engaged is True


@pytest.mark.asyncio
async def test_reverse_disengages_blade_ros():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.engage_blade(True)
    assert s.state.blade_engaged is True
    await s.set_gear(Transmission.REVERSE)
    assert s.state.gear == Transmission.REVERSE
    assert s.state.blade_engaged is False  # Reverse Operation System


@pytest.mark.asyncio
async def test_blade_rejected_in_reverse():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.set_gear(Transmission.REVERSE)
    r = await s.engage_blade(True)
    assert r["status"] == "rejected" and "reverse" in r["reason"]


@pytest.mark.asyncio
async def test_emergency_stop_disengages_but_engine_runs():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.set_clutch(0.0)
    await s.set_gear(Transmission.FORWARD)
    await s.set_ground_speed(0.8)
    await s.engage_blade(True)

    r = await s.emergency_stop()
    assert r["status"] == "emergency_stop"
    assert s.state.emergency_stop_active is True
    assert s.state.blade_engaged is False
    assert s.state.gear == Transmission.NEUTRAL
    assert s.state.clutch == pytest.approx(1.0)  # clutch/brake pressed
    assert s.state.ground_speed == 0.0
    assert s.state.throttle == 0.0
    assert s.state.engine == EngineState.RUNNING  # engine keeps running
    assert s.state.authorized is False

    # commands are rejected until the emergency is cleared
    assert (await s.set_steering(0.5))["status"] == "rejected"
    await s.clear_emergency()
    assert (await s.set_steering(0.5))["status"] == "ok"


@pytest.mark.asyncio
async def test_apply_full_command_moving():
    s = _svc()
    s.authorize()
    await s.start_engine()
    cmd = TractorCommand(
        steering=0.3,
        throttle=0.7,
        ground_speed=0.5,
        gear=Transmission.FORWARD,
        clutch=0.0,
        blade_engaged=True,
    )
    res = await s.apply(cmd)
    assert res["status"] == "applied"
    st = s.get_state()
    assert st.steering == pytest.approx(0.3)
    assert st.gear == Transmission.FORWARD
    assert st.blade_engaged is True
    assert st.moving is True


def test_action_prediction_to_tractor_command():
    p = ActionPrediction(steering=-0.4, throttle=0.6, blade=True, confidence=0.9)
    cmd = p.to_tractor_command()
    assert cmd.steering == pytest.approx(-0.4)
    assert cmd.ground_speed == pytest.approx(0.6)
    assert cmd.gear == Transmission.FORWARD
    assert cmd.blade_engaged is True
    assert cmd.clutch == 0.0
