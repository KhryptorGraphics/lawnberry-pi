"""Command deadman for the zero-turn mower.

A lever/throttle command is only in force for a bounded hold. If the client
disappears (browser tab closed, HaLow link dropped, nav loop wedged) nothing
refreshes it, and the service itself must return the levers to neutral and the
throttle to idle -- the machine must not keep driving on a command nobody is
still making. Faults are injected at the PCA9685 write.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.src.models.tractor_control import EngineState, TractorCommand
from backend.src.services.tractor_service import TractorControlService

LEFT, THROTTLE, RIGHT = 0, 1, 2  # default PCA9685 channels


class RecordingPCA9685:
    """PCA9685 test double that records every pulse and can fail or stall on cue."""

    last_error = None

    def __init__(self) -> None:
        self.writes: list[tuple[int, int]] = []
        self.fail_channels: set[int] = set()
        self.delay_s: dict[int, float] = {}

    def is_ready(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass

    async def set_pwm_us(self, channel: int, us: int) -> None:
        if channel in self.fail_channels:
            raise OSError("simulated I2C bus fault")
        delay = self.delay_s.get(channel, 0.0)
        if delay and us != 1500:
            await asyncio.sleep(delay)
        self.writes.append((channel, us))

    def last_us(self, channel: int) -> int | None:
        for ch, us in reversed(self.writes):
            if ch == channel:
                return us
        return None

    def count(self) -> int:
        return len(self.writes)


async def _running(**deadman) -> tuple[TractorControlService, RecordingPCA9685]:
    svc = TractorControlService(config={"command_deadman": deadman} if deadman else {})
    rec = RecordingPCA9685()
    svc._pca9685 = rec
    svc.authorize()
    # The 800 ms starter pulse is not what these tests are about: mark the engine running.
    svc.state.engine = EngineState.RUNNING
    return svc, rec


@pytest.fixture
async def svc_factory():
    made: list[TractorControlService] = []

    async def make(**deadman):
        svc, rec = await _running(**deadman)
        made.append(svc)
        return svc, rec

    yield make
    for svc in made:
        await svc.shutdown()


# ------------------------------- expiry ---------------------------------


async def test_lever_command_returns_to_neutral_when_nobody_refreshes_it(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=80, max_hold_ms=300)

    result = await svc.set_levers(0.8, 0.6)

    assert result["status"] == "ok" and result["hold_ms"] == 80
    assert svc.state.left_lever == pytest.approx(0.8)
    assert svc.state.moving is True
    assert rec.last_us(LEFT) == 1900 and rec.last_us(RIGHT) == 1800

    await asyncio.sleep(0.3)

    assert svc.state.left_lever == 0.0 and svc.state.right_lever == 0.0
    assert svc.state.moving is False
    # The *outputs* went neutral, not just the bookkeeping.
    assert rec.last_us(LEFT) == 1500 and rec.last_us(RIGHT) == 1500


async def test_a_refreshed_command_stays_in_force_then_lapses_when_the_client_stops(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=120, max_hold_ms=300)

    # A client that keeps re-sending inside the hold (like nav at 5 Hz / teleop at 20 Hz)...
    for _ in range(12):
        await svc.set_levers(0.5, 0.5)
        await asyncio.sleep(0.04)
        assert svc.state.left_lever == pytest.approx(0.5), "refreshed command must not lapse"

    # ...then vanishes: no more commands, the machine must stop on its own.
    await asyncio.sleep(0.35)
    assert svc.state.left_lever == 0.0 and svc.state.right_lever == 0.0
    assert rec.last_us(LEFT) == 1500


async def test_duration_ms_is_honoured_and_clamped_to_the_configured_maximum(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=400, max_hold_ms=500)

    short = await svc.set_levers(0.5, 0.5, duration_ms=40)
    assert short["hold_ms"] == 40
    await asyncio.sleep(0.2)  # well past 40 ms but well inside the 400 ms default
    assert svc.state.left_lever == 0.0, "a client-requested short hold must be honoured"

    huge = await svc.set_levers(0.5, 0.5, duration_ms=60_000)
    assert huge["hold_ms"] == 500, "a client cannot ask for more than the configured maximum"
    default = await svc.set_levers(0.5, 0.5)
    assert default["hold_ms"] == 400


async def test_throttle_returns_to_idle_independently_of_the_levers(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=400, max_hold_ms=800)

    await svc.set_levers(0.5, 0.5)  # 400 ms hold
    thr = await svc.set_throttle(0.9, duration_ms=50)
    assert thr["hold_ms"] == 50 and svc.state.throttle == pytest.approx(0.9)

    await asyncio.sleep(0.2)
    assert svc.state.throttle == 0.0, "throttle hold expired -> idle"
    assert rec.last_us(THROTTLE) == 1000  # default calibration us_min = idle
    assert svc.state.left_lever == pytest.approx(0.5), "levers have their own, longer hold"

    await asyncio.sleep(0.4)
    assert svc.state.left_lever == 0.0


async def test_apply_honours_the_commands_duration_ms(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=400, max_hold_ms=800)

    res = await svc.apply(
        TractorCommand(left_lever=0.5, right_lever=0.5, throttle=0.5, duration_ms=40)
    )

    assert res["status"] == "applied"
    assert res["results"]["levers"]["hold_ms"] == 40
    assert res["results"]["throttle"]["hold_ms"] == 40
    await asyncio.sleep(0.2)
    assert svc.state.left_lever == 0.0 and svc.state.throttle == 0.0


async def test_expiry_is_reported_to_the_capture_hook_as_a_deadman_change(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=50, max_hold_ms=200)
    events: list[tuple[str, dict, str]] = []
    svc.on_change = lambda kind, values, source: events.append((kind, values, source))

    await svc.set_levers(0.5, 0.5, source="teleop")
    await asyncio.sleep(0.2)

    assert ("levers", {"left": 0.0, "right": 0.0}, "deadman") in events


# ------------------------- what the deadman must NOT do -------------------------


async def test_a_neutral_command_cancels_the_pending_expiry(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=80, max_hold_ms=200)
    await svc.set_levers(0.5, 0.5)
    await svc.set_levers(0.0, 0.0)
    writes_after_neutral = rec.count()

    await asyncio.sleep(0.25)

    assert rec.count() == writes_after_neutral, "nothing left to expire: no extra writes"


async def test_the_deadman_does_not_touch_the_blade(svc_factory):
    """Blade handling stays with the interlocks/latches; the deadman is levers+throttle."""
    svc, rec = await svc_factory(default_hold_ms=50, max_hold_ms=200)
    assert (await svc.engage_blade(True))["status"] == "ok"
    await svc.set_levers(0.5, 0.5)

    await asyncio.sleep(0.25)

    assert svc.state.left_lever == 0.0
    assert svc.state.blade_engaged is True
    assert svc.blade_pto.state is True


async def test_estop_cancels_the_deadman_and_commands_stay_refused(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=80, max_hold_ms=200)
    await svc.set_levers(0.5, 0.5)

    result = await svc.emergency_stop()
    assert result["complete"] is True
    writes_after_estop = rec.count()

    await asyncio.sleep(0.25)
    assert rec.count() == writes_after_estop, "no deadman write after the E-stop owns the outputs"
    assert (await svc.set_levers(0.5, 0.5))["status"] == "rejected"
    assert rec.last_us(LEFT) == 1500


# ------------------------------ fault injection ------------------------------


async def test_a_deadman_that_cannot_neutralise_escalates_to_a_latched_estop(svc_factory):
    svc, rec = await svc_factory(default_hold_ms=60, max_hold_ms=200)
    assert (await svc.engage_blade(True))["status"] == "ok"
    await svc.set_levers(0.8, 0.8)
    events: list[tuple[str, dict, str]] = []
    svc.on_change = lambda kind, values, source: events.append((kind, values, source))

    rec.fail_channels = {LEFT}  # the bus dies while the machine is moving
    await asyncio.sleep(0.3)

    assert svc.state.emergency_stop_active is True, "could not neutralise -> latch the E-stop"
    assert svc.state.authorized is False
    assert svc.state.blade_engaged is False, "the E-stop still drops the blade"
    assert svc.state.left_lever == pytest.approx(0.8), "left never reached neutral; say so"
    assert svc.state.right_lever == 0.0, "the healthy channel did reach neutral"
    assert any(kind == "emergency_stop" and src == "deadman" for kind, _, src in events)
    fault = svc.get_state().actuator_fault
    assert fault is not None and "channel 0" in fault and "I2C" in fault


async def test_a_failed_second_lever_write_still_brings_the_first_lever_back(svc_factory):
    """set_levers writes left then right with no rollback. If the right write
    fails the left lever has already moved: the deadman must still be armed so
    it does not stay pulled (an unintended pivot) forever."""
    svc, rec = await svc_factory(default_hold_ms=80, max_hold_ms=200)
    rec.fail_channels = {RIGHT}

    with pytest.raises(OSError):
        await svc.set_levers(0.9, 0.9)
    assert rec.last_us(LEFT) == 1950  # the left lever did move
    assert svc.state.left_lever == pytest.approx(0.9)

    rec.fail_channels = set()  # bus recovers
    await asyncio.sleep(0.3)

    assert rec.last_us(LEFT) == 1500
    assert svc.state.left_lever == 0.0


async def test_an_inflight_write_cannot_land_after_the_estop(svc_factory):
    """A slow servo write already in flight must finish BEFORE the E-stop's neutral
    write, not after it (which would leave the lever pulled after 'stopped')."""
    svc, rec = await svc_factory(default_hold_ms=500, max_hold_ms=1000)
    rec.delay_s = {LEFT: 0.12}  # the non-neutral left write takes 120 ms

    command = asyncio.create_task(svc.set_levers(0.9, 0.9))
    await asyncio.sleep(0.02)  # the command is now mid-write, holding the actuation lock
    result = await svc.emergency_stop()
    await command  # may be rejected or complete; either way the E-stop must win

    assert result["complete"] is True
    assert rec.last_us(LEFT) == 1500, "E-stop's neutral must be the LAST write on the channel"
    assert rec.last_us(RIGHT) == 1500
    assert svc.state.left_lever == 0.0 and svc.state.emergency_stop_active is True
