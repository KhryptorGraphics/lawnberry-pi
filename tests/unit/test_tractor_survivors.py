"""Tractor survivors of the PR #20 second review: T9, T15, T16, T30, T38, T43.

Each test pins one branch that no earlier test reached (the reviewer applied a one-line mutant
for each and the whole safety suite still passed). They run on the real ``TractorControlService``
over ``StallablePCA9685`` so the proof is what reached the chip, in order.
"""

from __future__ import annotations

import asyncio
import math

import httpx
import pytest
from _nav_rig import wait_until
from _tractor_fakes import LEFT, NEUTRAL, RIGHT, mowing_tractor

from backend.src.api.routers import tractor as tractor_router
from backend.src.safety import stop_events

# ------------------------------------------------------------------------------ T16


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
async def test_a_non_finite_lever_command_is_a_clean_rejection_and_writes_nothing(bad):
    """Without the pre-check the ValueError of the second guard escaped to the caller (and the
    command hold had been armed first)."""
    tractor, chip = mowing_tractor()
    try:
        for args in ((bad, 0.0), (0.0, bad), (bad, bad)):
            result = await tractor.set_levers(*args, source="navigation")
            assert result["status"] == "rejected" and "non-finite" in result["reason"], result
        assert chip.log == [], "a refused command must not reach the servos"
        assert tractor._lever_deadline is None, "and must not arm the deadman"
        throttle = await tractor.set_throttle(bad, source="navigation")
        assert throttle["status"] == "rejected" and "non-finite" in throttle["reason"]
        assert chip.log == []
    finally:
        await tractor.shutdown()


# ------------------------------------------------------------------------------ T30


async def test_a_crashed_deadman_fails_safe_into_an_emergency_stop():
    """A dead deadman would leave the last command in force for ever: it must stop the machine."""
    tractor, chip = mowing_tractor()
    events: list[tuple[str, str]] = []
    tractor.on_change = lambda kind, values, source: events.append((kind, source))
    await tractor.engage_blade(True, source="navigation")
    assert tractor.state.blade_engaged

    async def crash() -> None:
        raise RuntimeError("deadman bug")

    tractor._expire_due = crash  # the expiry path is the one that blows up
    try:
        await tractor.set_levers(0.5, 0.5, source="navigation", duration_ms=50)
        await wait_until(lambda: ("emergency_stop", "deadman") in events)
        assert tractor.state.emergency_stop_active, events
        assert tractor.state.authorized is False and tractor.state.blade_engaged is False
        assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
    finally:
        await tractor.shutdown()


# ------------------------------------------------------------------------------ T43


async def test_a_clear_gate_that_raises_keeps_the_blade_latch():
    """Fail closed: a veto that cannot be evaluated is a veto."""
    tractor, _chip = mowing_tractor()
    await tractor.latch_interlock("tilt", source="tilt_monitor")

    def broken(reason):
        raise RuntimeError("tilt monitor bug")

    tractor.attach_interlock_clear_gate(broken)
    try:
        result = await tractor.clear_emergency()
        assert result["status"] == "partial", result
        assert result["still_latched"] == ["blade_interlock:tilt"]
        assert "clear gate failed" in result["reason"] and "tilt monitor bug" in result["reason"]
        assert tractor.state.interlock_latched is True
        blocked = await tractor.engage_blade(True, source="navigation")
        assert blocked["status"] == "rejected"
        assert tractor.state.blade_engaged is False
    finally:
        await tractor.shutdown()


# ------------------------------------------------------------------------- T38 and T9


async def test_a_stop_between_the_left_and_the_right_lever_write_lands_no_second_pulse():
    """A command that began before the stop must not put its SECOND write on the chip.

    ``set_levers(0.0, 0.5)``: the left neutral write is in flight when the E-stop lands (a
    neutral write is not 'guarded', so nothing supersedes it); when it releases the right-lever
    write must be refused before it starts, not written and then re-asserted away.
    """
    tractor, chip = mowing_tractor()
    gate = chip.stall(LEFT, NEUTRAL[LEFT])
    command = asyncio.create_task(tractor.set_levers(0.0, 0.5, source="navigation"))
    await asyncio.wait_for(chip.stalled.wait(), timeout=2.0)
    stop = asyncio.create_task(tractor.emergency_stop(source="api"))
    # The stop cannot take the lock (the left write holds it); it writes neutral without it.
    await wait_until(lambda: chip.writes(LEFT).count(NEUTRAL[LEFT]) >= 1 and chip.regs.get(RIGHT))
    gate.set()  # the left write finally lands
    result = await asyncio.wait_for(command, timeout=3.0)
    await asyncio.wait_for(stop, timeout=3.0)

    assert result["status"] == "rejected", result
    right_pulse = tractor.right_lever.microseconds(0.5)
    assert right_pulse not in chip.writes(RIGHT), (
        f"the second lever write landed after the stop: {chip.writes(RIGHT)}"
    )
    assert chip.regs[RIGHT] == NEUTRAL[RIGHT] and chip.regs[LEFT] == NEUTRAL[LEFT]
    await tractor.shutdown()


async def test_a_lock_out_lands_no_second_pulse_from_an_in_flight_command():
    """``lock_out`` is not an E-stop (no neutral writes of its own), but the command in flight
    must still not write its second lever after it: the generation is bumped."""
    tractor, chip = mowing_tractor()
    left_pulse = tractor.left_lever.microseconds(0.5)
    right_pulse = tractor.right_lever.microseconds(0.5)
    gate = chip.stall(LEFT, left_pulse)
    command = asyncio.create_task(tractor.set_levers(0.5, 0.5, source="navigation"))
    await asyncio.wait_for(chip.stalled.wait(), timeout=2.0)

    tractor.lock_out("safety wiring failed at start-up")
    gate.set()
    result = await asyncio.wait_for(command, timeout=3.0)

    assert result["status"] == "rejected", result
    assert right_pulse not in chip.writes(RIGHT), "the right lever moved after the lock-out"
    await tractor.shutdown()


# ------------------------------------------------------------------------------ T15


@pytest.mark.parametrize("source", ["operator", "teleop", "navigation", "ai", "unspecified"])
async def test_the_blade_needs_the_authorization_whoever_asks(source):
    """Levers of a person need no authorize() (a hold-to-run joystick); the blade ALWAYS does."""
    tractor, _chip = mowing_tractor()
    tractor.revoke()
    try:
        result = await tractor.engage_blade(True, source=source)
        assert result["status"] == "rejected" and result["code"] == "not_authorized", result
        assert tractor.state.blade_engaged is False
    finally:
        await tractor.shutdown()


async def test_a_blade_that_turns_when_the_authorization_goes_is_switched_off_by_the_next_request():
    tractor, _chip = mowing_tractor()
    try:
        assert (await tractor.engage_blade(True, source="operator"))["status"] == "ok"
        assert tractor.state.blade_engaged is True
        tractor.revoke()  # the flag alone: nothing has stood the machine down yet
        result = await tractor.engage_blade(True, source="operator")
        assert result["status"] == "rejected" and result["code"] == "not_authorized"
        assert tractor.state.blade_engaged is False, "refuse AND drop"
    finally:
        await tractor.shutdown()


# ------------------------------------------------------------- /tractor/revoke and /blade routes


async def _client(monkeypatch, tractor) -> httpx.AsyncClient:
    from backend.src.main import app

    monkeypatch.setattr(tractor_router, "get_tractor_service", lambda: tractor)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_the_blade_route_refuses_an_unauthorized_operator_and_revoke_stops_a_running_blade(
    monkeypatch,
):
    tractor, chip = mowing_tractor()
    tractor.revoke()
    heard: list[tuple[str, str]] = []

    def listener(reason: str, source: str) -> None:
        heard.append((reason, source))

    stop_events.add_listener(listener)
    async with await _client(monkeypatch, tractor) as client:
        refused = await client.post("/api/v2/tractor/blade", json={"engaged": True})
        assert refused.json()["status"] == "rejected" and tractor.state.blade_engaged is False

        assert (await client.post("/api/v2/tractor/authorize")).json() == {"status": "authorized"}
        mowing = await client.post("/api/v2/tractor/blade", json={"engaged": True})
        assert mowing.json()["status"] == "ok" and tractor.state.blade_engaged is True
        # an autonomous source has the levers in force when the operator revokes
        await tractor.set_levers(0.5, 0.5, source="navigation", duration_ms=5000)
        assert tractor.state.moving

        revoked = await client.post("/api/v2/tractor/revoke")

    body = revoked.json()
    assert body["status"] == "revoked" and body["errors"] == [], body
    assert tractor.state.authorized is False
    assert tractor.state.blade_engaged is False, "a revoke must stop the blade, not only a flag"
    assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0
    assert chip.regs[LEFT] == NEUTRAL[LEFT] and chip.regs[RIGHT] == NEUTRAL[RIGHT]
    assert ("authorization revoked", "operator") in heard, "every run ends with the revoke"
    stop_events.remove_listener(listener)
    await tractor.shutdown()


async def test_a_revoke_leaves_a_persons_own_manual_drive_alone(monkeypatch):
    tractor, _chip = mowing_tractor()
    await tractor.set_levers(0.4, 0.4, source="operator", duration_ms=5000)
    async with await _client(monkeypatch, tractor) as client:
        body = (await client.post("/api/v2/tractor/revoke")).json()
    assert body["status"] == "revoked"
    assert tractor.state.left_lever == 0.4, (
        "the joystick is the person's, a revoke does not take it"
    )
    await tractor.shutdown()
