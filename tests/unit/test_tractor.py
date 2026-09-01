"""Unit tests for the zero-turn mower actuation + safety interlocks."""

import pytest

from backend.src.drivers.actuators import RelayActuator, ServoActuator, ServoCalibration
from backend.src.models.tractor_control import EngineState, TractorCommand
from backend.src.services.tractor_service import TractorControlService

# --------------------------- actuator mapping ---------------------------


def test_servo_bidirectional_mapping():
    s = ServoActuator(
        "left_lever", ServoCalibration(channel=0, us_min=1000, us_center=1500, us_max=2000)
    )
    assert s.microseconds(0.0) == 1500
    assert s.microseconds(1.0) == 2000
    assert s.microseconds(-1.0) == 1000
    assert s.microseconds(0.5) == 1750
    assert s.microseconds(5.0) == 2000  # clamped


def test_servo_unidirectional_mapping():
    t = ServoActuator(
        "throttle", ServoCalibration(channel=1, us_min=1100, us_max=1900, bidirectional=False)
    )
    assert t.microseconds(0.0) == 1100
    assert t.microseconds(1.0) == 1900
    assert t.microseconds(0.5) == 1500


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
async def test_start_rejected_when_levers_not_neutral():
    s = _svc()
    s.authorize()
    await s.set_left_lever(0.5)
    r = await s.start_engine()
    assert r["status"] == "rejected" and "neutral" in r["reason"]


@pytest.mark.asyncio
async def test_start_rejected_when_blade_on():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.engage_blade(True)
    r = await s.start_engine()  # re-crank attempt: blade is now on
    assert r["status"] == "rejected" and "blade" in r["reason"]


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
    await s.set_levers(-0.5, -0.5)  # both levers back = reverse
    assert s.state.reversing is True
    assert s.state.blade_engaged is False  # Reverse Operation System


@pytest.mark.asyncio
async def test_one_lever_pivot_does_not_disengage_blade():
    """A zero-radius pivot turn (one lever back, one forward) is not reverse.

    This is the regression test for the reverse-definition decision: treating
    a single negative lever as "reverse" would drop the blade on every pivot
    while mowing, which is wrong.
    """
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.engage_blade(True)
    await s.set_levers(-0.5, 0.5)  # pivot turn, not reverse
    assert s.state.reversing is False
    assert s.state.blade_engaged is True  # blade must stay engaged through a pivot


@pytest.mark.asyncio
async def test_blade_rejected_in_reverse():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.set_levers(-0.5, -0.5)
    r = await s.engage_blade(True)
    assert r["status"] == "rejected" and "reverse" in r["reason"]


@pytest.mark.asyncio
async def test_emergency_stop_disengages_but_engine_runs():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.set_levers(0.8, 0.6)
    await s.set_throttle(0.9)
    await s.engage_blade(True)

    r = await s.emergency_stop()
    assert r["status"] == "emergency_stop"
    assert s.state.emergency_stop_active is True
    assert s.state.blade_engaged is False
    assert s.state.left_lever == 0.0
    assert s.state.right_lever == 0.0
    assert s.state.throttle == 0.0
    assert s.state.engine == EngineState.RUNNING  # engine keeps running
    assert s.state.authorized is False

    # commands are rejected until the emergency is cleared
    assert (await s.set_left_lever(0.5))["status"] == "rejected"
    await s.clear_emergency()
    assert (await s.set_left_lever(0.5))["status"] == "ok"


@pytest.mark.asyncio
async def test_apply_full_command_moving():
    s = _svc()
    s.authorize()
    await s.start_engine()
    cmd = TractorCommand(left_lever=0.5, right_lever=0.5, throttle=0.7, blade_engaged=True)
    res = await s.apply(cmd)
    assert res["status"] == "applied"
    st = s.get_state()
    assert st.left_lever == pytest.approx(0.5)
    assert st.right_lever == pytest.approx(0.5)
    assert st.blade_engaged is True
    assert st.moving is True


# ------------------------- PCA9685 transport -------------------------
#
# The driver's own two-failure-mode behavior (tolerated no-op vs. propagate)
# is covered in tests/unit/test_pca9685_driver.py. These tests cover
# tractor_service's orchestration on top of it: channel assignment, that a
# driver failure actually reaches callers, and that emergency_stop's safing
# sequence survives a single bad channel.


class _FaultyPCA9685:
    """Test double: set_pwm_us raises OSError for configured channels only."""

    def __init__(self, fail_channels: set[int]):
        self.fail_channels = fail_channels
        self.calls: list[tuple[int, int]] = []

    async def set_pwm_us(self, channel: int, us: int) -> None:
        self.calls.append((channel, us))
        if channel in self.fail_channels:
            raise OSError("simulated I2C bus fault")

    async def initialize(self) -> None:
        pass


def test_default_channels_are_pca9685_zero_indexed():
    s = _svc()
    assert s.left_lever.cal.channel == 0
    assert s.throttle.cal.channel == 1
    assert s.right_lever.cal.channel == 2


@pytest.mark.asyncio
async def test_initialize_brings_up_pca9685_driver_first():
    s = _svc()
    calls: list[str] = []

    class _Spy:
        async def initialize(self) -> None:
            calls.append("init")

        async def set_pwm_us(self, channel: int, us: int) -> None:
            assert calls == ["init"], "driver must be initialized before it is used"

    s._pca9685 = _Spy()
    await s.initialize()
    assert calls == ["init"]
    assert s.initialized is True


@pytest.mark.asyncio
async def test_send_pwm_propagates_driver_write_failure():
    """A real transport fault must reach the caller, not be swallowed."""
    s = _svc()
    s._pca9685 = _FaultyPCA9685(fail_channels={s.left_lever.cal.channel})
    s.authorize()
    with pytest.raises(OSError):
        await s.set_left_lever(0.5)


@pytest.mark.asyncio
async def test_emergency_stop_survives_one_bad_channel():
    s = _svc()
    s.authorize()
    await s.start_engine()
    await s.set_levers(0.8, 0.6)
    await s.set_throttle(0.9)
    await s.engage_blade(True)

    # Fail only the left-lever channel; the other 2 PWM calls and the
    # blade-PTO relay cutoff must still complete instead of the whole
    # sequence aborting.
    s._pca9685 = _FaultyPCA9685(fail_channels={s.left_lever.cal.channel})

    r = await s.emergency_stop()
    assert r["status"] == "emergency_stop"
    assert s.state.emergency_stop_active is True
    assert s.state.authorized is False
    assert s.state.blade_engaged is False  # GPIO relay, unaffected by I2C fault
    assert s.state.right_lever == 0.0  # its channel succeeded
    assert s.state.throttle == 0.0  # its channel succeeded
    # left lever's own channel failed: its state is left stale, but that
    # alone didn't stop the rest of the safing sequence from being attempted.
    assert s.state.left_lever == pytest.approx(0.8)
