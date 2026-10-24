"""The E-stop wins, durably, over a lever/throttle write that is already on the bus.

Audit SF-01 (S1): the stop waited only ``ESTOP_LOCK_WAIT_S`` for the actuation lock. A write
stalled longer than that (a slow I2C bus) landed AFTER the stop's neutral, its command went on
to issue its second lever write, the stop had cancelled the deadman that could have brought the
lever back, and the result still said ``complete: true``. SF-04: the stop opened the PTO relay
only after that wait (252 ms measured against the 100 ms acceptance limit). SF-09: a stop that
cut the starter mid-crank still left the engine "running". Also here: the lever-pair race a
concurrent Vue page produced, and the test-quality items the safety review's mutation run found
(deadman clock, in-lock stop re-check, starter relay).

Faults are injected at the PCA9685 write (``StallablePCA9685``: last-write-wins registers, a write
that hangs on cue and, like the threaded driver, still lands after the service stopped waiting)
and at the GPIO (``FakeGpiozero``). Nothing here touches hardware.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from _tractor_fakes import (
    LEFT,
    NEUTRAL,
    RIGHT,
    THROTTLE,
    FakeGpiozero,
    mowing_tractor,
)

from backend.src.api.routers import tractor as tractor_router
from backend.src.models.tractor_control import EngineState
from backend.src.services.tractor_service import TractorControlService

# ------------------------------- SF-01: durability -------------------------------


@pytest.mark.parametrize(
    ("what", "channel", "pulse", "command"),
    [
        ("levers", LEFT, 1950, lambda svc: svc.set_levers(0.9, 0.9, source="operator")),
        ("throttle", THROTTLE, 1900, lambda svc: svc.set_throttle(0.9, source="operator")),
    ],
)
async def test_a_write_in_flight_when_the_stop_lands_cannot_leave_the_output_off_neutral(
    what, channel, pulse, command
):
    svc, pca = mowing_tractor()
    release = pca.stall(channel, pulse)
    cmd = asyncio.create_task(command(svc))
    await asyncio.wait_for(pca.stalled.wait(), 2)  # mid-write, holding the actuation lock
    estop = asyncio.create_task(svc.emergency_stop())
    await asyncio.sleep(0.4)  # well past any lock wait: the stop has written without the lock
    assert not estop.done(), (
        "the stop must wait for the in-flight write before it calls itself done"
    )
    release.set()  # NOW the stalled pulse reaches the chip, after the stop's first neutral

    result = await asyncio.wait_for(estop, 3)
    refused = await asyncio.wait_for(cmd, 3)

    assert pulse in pca.writes(channel), "the late write really landed after the stop"
    assert pca.writes(channel)[-1] == NEUTRAL[channel], "the stop's neutral must be the last word"
    assert pca.regs == NEUTRAL
    if what == "levers":
        assert 1950 not in pca.writes(RIGHT), (
            "the command's second lever write must never be issued"
        )
    assert refused["status"] == "rejected"
    assert result["complete"] is True and result["inflight_write"] is True
    assert svc.state.emergency_stop_active is True
    assert svc.state.left_lever == 0.0 and svc.state.throttle == 0.0
    assert svc._lever_deadline is None and svc._throttle_deadline is None
    await svc.shutdown()


async def test_a_stop_that_cannot_reassert_neutral_after_a_stuck_holder_is_not_complete():
    svc, pca = mowing_tractor()
    svc.ACTUATION_TIMEOUT_S = 0.1
    svc.ESTOP_REASSERT_MARGIN_S = 0.1  # the stop waits 0.2 s for the lock, then gives up
    hold = asyncio.Event()

    async def stuck_holder():  # an earlier command whose write never returns
        async with svc._act_lock:
            await hold.wait()

    holder = asyncio.create_task(stuck_holder())
    await asyncio.sleep(0.01)

    result = await asyncio.wait_for(svc.emergency_stop(), 2)
    hold.set()
    await holder

    assert result["complete"] is False and result["inflight_write"] is True
    for name in ("left_lever", "right_lever", "throttle"):
        out = result["subsystems"][name]
        assert out["ok"] is False, "neutral was not confirmed after the in-flight write"
        assert "in-flight actuation still held the actuation lock" in out["error"]
    # The relays need no lock: they were driven, and the first neutral pass still went out.
    assert result["subsystems"]["blade_pto"]["ok"] and result["subsystems"]["starter"]["ok"]
    assert pca.regs == NEUTRAL
    assert svc.state.emergency_stop_active is True
    await svc.shutdown()


async def test_a_command_queued_before_the_stop_is_refused_even_after_the_latch_was_cleared():
    """A stop must outlast a clear for every command that began before it (stop generation)."""
    svc, pca = mowing_tractor()
    svc.ACTUATION_TIMEOUT_S = 0.1
    svc.ESTOP_REASSERT_MARGIN_S = 0.05  # the stop gives up on the stuck lock quickly
    release = asyncio.Event()

    async def stuck_holder():
        async with svc._act_lock:
            await release.wait()

    holder = asyncio.create_task(stuck_holder())
    await asyncio.sleep(0.01)
    queued = asyncio.create_task(svc.set_levers(0.9, 0.9, source="operator"))
    await asyncio.sleep(0.01)  # queued behind the stuck holder, past every pre-lock check

    stop = await asyncio.wait_for(svc.emergency_stop(), 2)
    assert stop["complete"] is False  # the lock never freed
    assert (await svc.clear_emergency())["status"] == "ok"  # the operator clears; flag is False
    release.set()
    await holder
    result = await asyncio.wait_for(queued, 2)

    assert result["status"] == "rejected", "a command from before the stop must not land after it"
    assert 1950 not in pca.writes(LEFT)
    await svc.shutdown()


@pytest.mark.parametrize(
    ("what", "channel", "pulse", "stalled_cmd", "queued_cmd", "deadline"),
    [
        (
            "levers",
            LEFT,
            1750,
            lambda svc: svc.set_levers(0.5, 0.5, source="operator"),
            lambda svc: svc.set_levers(0.9, 0.9, source="operator"),
            "_lever_deadline",
        ),
        (
            "throttle",
            THROTTLE,
            1500,
            lambda svc: svc.set_throttle(0.5, source="operator"),
            lambda svc: svc.set_throttle(0.9, source="operator"),
            "_throttle_deadline",
        ),
    ],
)
async def test_a_refused_stale_command_leaves_no_trace(
    what, channel, pulse, stalled_cmd, queued_cmd, deadline
):
    """Refused means refused: no hold armed for it, no deadman started, no write ever follows."""
    svc, pca = mowing_tractor(
        config={"command_deadman": {"default_hold_ms": 80, "max_hold_ms": 200}}
    )
    release = pca.stall(channel, pulse)
    first = asyncio.create_task(stalled_cmd(svc))
    await asyncio.wait_for(pca.stalled.wait(), 2)
    queued = asyncio.create_task(queued_cmd(svc))  # waits behind the stalled write
    await asyncio.sleep(0.01)
    estop = asyncio.create_task(svc.emergency_stop())
    await asyncio.sleep(0.1)
    release.set()
    await asyncio.wait_for(estop, 3)
    assert (await asyncio.wait_for(first, 3))["status"] == "rejected"
    assert (await asyncio.wait_for(queued, 3))["status"] == "rejected"

    assert getattr(svc, deadline) is None, "a refused command must not arm a hold"
    landed = len(pca.log)
    await asyncio.sleep(0.3)  # longer than the 80 ms hold it would have armed
    assert len(pca.log) == landed, "nothing may write after the stop: no deadman, no stale command"
    assert pca.regs == NEUTRAL
    await svc.shutdown()


async def test_the_tractor_estop_route_runs_to_completion_when_the_request_is_cancelled(
    monkeypatch,
):
    svc, pca = mowing_tractor()
    monkeypatch.setattr(tractor_router, "get_tractor_service", lambda: svc)
    release = pca.stall(LEFT, 1950)
    cmd = asyncio.create_task(svc.set_levers(0.9, 0.9, source="operator"))
    await asyncio.wait_for(pca.stalled.wait(), 2)

    request = asyncio.create_task(tractor_router.tractor_emergency_stop())
    await asyncio.sleep(0.02)
    request.cancel()  # the client went away in the middle of the stop
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.sleep(0.2)
    release.set()
    await asyncio.sleep(0.3)  # the shielded stop finishes: first pass, then the re-assert
    await cmd

    assert svc.state.emergency_stop_active is True
    assert pca.regs == NEUTRAL, "a cancelled request must not abort the stop half way"
    await svc.shutdown()


# --------------------- SF-04: relays first, independent of the lock ---------------------


async def test_the_estop_opens_the_pto_relay_before_it_waits_for_a_stalled_write(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    gpio = FakeGpiozero(monkeypatch)
    svc, pca = mowing_tractor()
    assert (await svc.engage_blade(True, source="operator"))["status"] == "ok"
    release = pca.stall(LEFT, 1950)
    cmd = asyncio.create_task(svc.set_levers(0.9, 0.9, source="operator"))
    await asyncio.wait_for(pca.stalled.wait(), 2)

    t0 = time.monotonic()
    estop = asyncio.create_task(svc.emergency_stop())
    await asyncio.sleep(0.02)

    pto_off = gpio.first(26, "off")
    starter_off = gpio.first(5, "off")
    assert pto_off is not None and pto_off - t0 < 0.1, "PTO relay must open within 100 ms"
    assert starter_off is not None and starter_off - t0 < 0.1
    assert svc._act_lock.locked(), "the relays opened while the stalled write still held the lock"
    assert not estop.done()
    assert svc.state.blade_engaged is False

    release.set()
    await asyncio.wait_for(estop, 3)
    await cmd
    first_neutral = min(t for t, ch, us in pca.log if t >= t0 and us == NEUTRAL[ch])
    assert pto_off < first_neutral, "the relay is cut before any servo is touched"
    await svc.shutdown()


async def test_the_estop_opens_the_starter_relay_too():
    svc, _pca = mowing_tractor()
    svc.starter.set(True)  # a crank in progress
    assert svc.starter.state is True

    result = await svc.emergency_stop()

    assert svc.starter.state is False
    assert result["subsystems"]["starter"]["ok"] is True
    await svc.shutdown()


# ------------------------------ SF-09: interrupted crank ------------------------------


async def test_a_stop_during_the_crank_leaves_the_engine_not_running():
    svc, _pca = mowing_tractor(config={"relays": {"starter": {"gpio": 5, "pulse_ms": 400}}})
    svc.state.engine = EngineState.OFF
    crank = asyncio.create_task(svc.start_engine())
    await asyncio.sleep(0.05)
    assert svc.state.engine == EngineState.STARTING and svc.starter.state is True

    await svc.emergency_stop()
    assert svc.starter.state is False, "the stop cuts the crank at once"
    result = await asyncio.wait_for(crank, 2)

    assert result["status"] == "rejected" and "cut short" in result["reason"]
    assert svc.state.engine == EngineState.OFF, "nobody observed this engine run"
    await svc.clear_emergency()
    svc.authorize()
    blade = await svc.engage_blade(True, source="operator")
    assert blade["status"] == "rejected" and "engine" in blade["reason"]
    await svc.shutdown()


# ----------------------- the lever-pair race (concurrent single-lever requests) -----------------


async def test_concurrent_single_lever_requests_cannot_overwrite_each_other():
    """'Both levers to neutral' as two concurrent requests read the OTHER lever before the lock
    and wrote (0, R_old) then (L_old, 0): one lever stayed forward."""
    svc, pca = mowing_tractor()
    await svc.set_levers(0.8, 0.8, source="operator")

    await asyncio.gather(
        svc.set_left_lever(0.0, source="operator"), svc.set_right_lever(0.0, source="operator")
    )

    assert svc.state.left_lever == 0.0 and svc.state.right_lever == 0.0
    assert pca.regs[LEFT] == 1500 and pca.regs[RIGHT] == 1500
    await svc.shutdown()


async def test_a_single_lever_command_keeps_the_other_levers_current_value():
    svc, pca = mowing_tractor()
    await svc.set_levers(0.4, -0.6, source="operator")

    result = await svc.set_left_lever(0.8, source="operator")

    assert result["status"] == "ok"
    assert svc.state.left_lever == pytest.approx(0.8)
    assert svc.state.right_lever == pytest.approx(-0.6), "the other lever is left where it was"
    assert pca.regs[RIGHT] == 1200
    await svc.shutdown()


# --------------------------- deadman clock (survived mutant M3) ---------------------------


async def test_the_deadman_is_not_defeated_by_a_wall_clock_step_backwards(monkeypatch):
    """The hold runs on the monotonic clock. Against the wall clock, an NTP/RTC step backwards
    leaves the lever in force for as long as the step is large."""
    svc, pca = mowing_tractor(
        config={"command_deadman": {"default_hold_ms": 150, "max_hold_ms": 500}}
    )
    await svc.set_levers(0.5, 0.5, source="operator")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 3600.0)

    await asyncio.sleep(0.4)

    assert svc.state.left_lever == 0.0 and pca.regs[LEFT] == 1500, "the hold must still expire"
    await svc.shutdown()


async def test_the_service_never_builds_its_own_gpio_in_sim_mode_for_a_stop():
    """SIM_MODE keeps relays state-only: the stop must not need gpiozero (guards the new order)."""
    svc, _pca = mowing_tractor()
    assert isinstance(svc, TractorControlService) and svc.blade_pto.simulated is True
    result = await svc.emergency_stop()
    assert result["complete"] is True and result["inflight_write"] is False
    assert all(v["simulated"] for v in result["subsystems"].values())
    await svc.shutdown()


async def test_the_stalled_chip_fake_really_lands_a_write_the_service_gave_up_on():
    """Guards the premise of the tests above: an abandoned write still reaches the chip."""
    svc, pca = mowing_tractor()
    svc.ACTUATION_TIMEOUT_S = 0.05
    release = pca.stall(LEFT, 1950)

    with pytest.raises(TimeoutError):
        await svc.set_levers(0.9, 0.9, source="operator")
    assert pca.regs.get(LEFT) is None  # not landed yet: the service timed out waiting
    release.set()
    await asyncio.sleep(0.05)

    assert pca.regs[LEFT] == 1950, "the threaded I2C transfer completes after the timeout"
    await svc.shutdown()
