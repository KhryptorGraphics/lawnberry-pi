"""The obstacle guard fed by the REAL ultrasonic stack, end to end.

Every test here drives: the real firmware core -> ``FakeMcuSerial`` -> ``JsnSr04tMcuDriver`` ->
``UltrasonicSensorInterface`` -> the real ``sample_sensor_manager`` -> ``ObstacleGuard`` (tests of
the guard alone use hand-made entries, so they cannot see a mismatch between what the interface
says and what the guard understands). The guard runs on the fake clock; its wall clock follows it.

Where a test needs an entry the real interface never produces (a corrupted flag, a missing
distance) it wraps the REAL interface and changes exactly that one key, so the rest of the entry
is still what the interface really says.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.src.models.safety_limits import ObstacleStopLimits
from backend.src.safety.obstacle_stop import ObstacleGuard, ObstacleStatus, sample_sensor_manager
from tests.unit._usonic_fakes import (
    FakeClock,
    FakeMcuFactory,
    SensorModel,
    approach_then_silence,
    patch_guard_wall_clock,
)
from tests.unit.test_ultrasonic_interface import make

OUTER_LEFT = "ultrasonic:outer_left"
NAMES = ["outer_left", "inner_left", "inner_right", "outer_right"]


def far(cm: float = 200.0) -> SensorModel:
    return SensorModel(distance_cm=cm)


def blind_latched(assessment) -> bool:
    """The guard holds because outer_left reads as a blind zone (the dead zone latch)."""
    return any(OUTER_LEFT in r and "blind zone" in r for r in assessment.reasons)


class Stack:
    """MCU + driver + interface + manager + guard on one fake clock."""

    def __init__(self, monkeypatch, sensors, cfg=None, limits=None, wrap=None):
        self.clock = FakeClock()
        patch_guard_wall_clock(monkeypatch, self.clock)
        self.factory = FakeMcuFactory(self.clock, sensors=sensors)
        self.iface = make(self.clock, self.factory, **(cfg or {}))
        ultrasonic = wrap(self.iface) if wrap else self.iface
        self.manager = SimpleNamespace(tof=None, ultrasonic=ultrasonic, ultrasonic_enabled=True)
        self.guard = ObstacleGuard(ObstacleStopLimits(**(limits or {})), clock=self.clock.now)

    async def start(self) -> Stack:
        assert await self.iface.initialize() is True
        return self

    async def tick(self, dt: float = 0.02, *, poll: bool = True, read: bool = True):
        """One guard cycle: service loop, (optionally) the telemetry read, sample, assess."""
        self.clock.advance(dt)
        if poll:
            self.iface._driver.poll()
        if read:
            await self.iface.read_ultrasonic()
        sample_sensor_manager(self.guard, self.manager)
        return self.guard.assess()

    async def run(self, seconds: float, dt: float = 0.02, **kw):
        return [await self.tick(dt, **kw) for _ in range(round(seconds / dt))]


async def stack(monkeypatch, sensors, **kw) -> Stack:
    return await Stack(monkeypatch, sensors, **kw).start()


# ---- US-01: the guard must not release toward an object that vanished at 28 cm ----------------


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"no_echo_is_clear": True},
        {"no_echo_is_clear": True, "no_echo_requires_alive": False},
    ],
    ids=["shipped", "silent_is_clear_if_alive", "silent_is_clear_always"],
)
async def test_the_guard_never_releases_toward_an_object_that_vanished_at_28_cm(
    monkeypatch, policy
):
    sensors = [approach_then_silence(), far(), far(), far()]
    s = await stack(monkeypatch, sensors, cfg=policy)  # default hold: 30 s
    ticks = await s.run(25.0)
    vanished = next((i for i, a in enumerate(ticks) if blind_latched(a)), None)
    assert vanished is not None and vanished * 0.02 < 4.0, "the silence was never latched"
    after = ticks[vanished:]
    assert len(after) > 20 / 0.02
    # not a single cycle with a speed cap, however long the silence lasts inside the hold
    assert {a.status for a in after} == {ObstacleStatus.STOP}
    assert max(a.speed_cap_mps for a in after) == 0.0
    assert all(OUTER_LEFT in a.obstacle_sources for a in after)


async def test_after_the_hold_the_guard_may_release_for_an_owner_who_trusts_silence(monkeypatch):
    # The shipped policy never does; this is the opt-in owner's explicit risk. no_echo_is_clear
    # is not made useless, though: once the (here 5 s) hold has passed, silence reads clear again.
    cfg = {"no_echo_is_clear": True, "no_echo_requires_alive": False, "near_hold_s": 5.0}
    s = await stack(monkeypatch, [approach_then_silence(), far(), far(), far()], cfg=cfg)
    ticks = await s.run(12.0)
    vanished = next((i for i, a in enumerate(ticks) if blind_latched(a)), None)
    assert vanished is not None
    released = next(i for i, a in enumerate(ticks) if i > vanished and a.speed_cap_mps > 0)
    assert (released - vanished) * 0.02 >= 5.0, "released inside the hold"
    assert ticks[-1].status is ObstacleStatus.CLEAR and not ticks[-1].obstacles


# ---- US-03: a channel that is not fitted must not hold the guard for ever ---------------------


async def test_a_three_sensor_build_reaches_clear_with_the_unfitted_channel_disabled(monkeypatch):
    sensors = [far(150), far(), far(), far(180)]
    s = await stack(monkeypatch, sensors, cfg={"disabled_channels": ["inner_right"]})
    ticks = await s.run(6.0)  # the 2 s clear dwell, then release
    last = ticks[-1]
    assert last.status is not ObstacleStatus.STOP and last.speed_cap_mps > 0 and not last.faults
    assert "ultrasonic:inner_right" not in s.guard.sources
    assert set(s.guard.sources) == {f"ultrasonic:{n}" for n in NAMES if n != "inner_right"}


async def test_with_every_channel_disabled_the_guard_still_holds_and_the_aggregate_is_unknown(
    monkeypatch,
):
    s = await stack(monkeypatch, [far()] * 4, cfg={"disabled_channels": NAMES})
    ticks = await s.run(6.0)
    assert {a.status for a in ticks} == {ObstacleStatus.STOP}
    assert s.iface.obstacle_verdict()["state"] == "unknown"


# ---- US-04: the guard's input must not depend on the telemetry cadence ------------------------


@pytest.mark.parametrize("telemetry_hz", [10, 2, 1, 0.5])
async def test_the_guard_is_not_held_by_a_slow_telemetry_loop(monkeypatch, telemetry_hz):
    s = await stack(monkeypatch, [far(150), far(), far(), far(180)])
    period, next_read, ticks = 1.0 / telemetry_hz, s.clock.now(), []
    for _ in range(round(30.0 / 0.05)):  # the guard samples at 20 Hz, 30 s of clean air
        due = s.clock.now() + 0.05 >= next_read
        ticks.append(await s.tick(0.05, read=due))  # the UI-tunable telemetry loop reads rarely
        if due:
            next_read += period
    settled = ticks[round(3.0 / 0.05) :]  # after the initial 2 s dwell
    assert [a.reasons for a in settled if a.hold] == []
    assert settled[-1].speed_cap_mps > 0


async def test_a_dead_service_loop_still_ages_out_into_a_hold(monkeypatch):
    s = await stack(monkeypatch, [far(150), far(), far(), far(180)])
    warm = await s.run(4.0)
    assert not warm[-1].hold
    ticks = await s.run(2.0, dt=0.05, poll=False, read=False)  # nothing polls the driver any more
    assert ticks[-1].hold and set(ticks[-1].faults) >= {"ultrasonic:outer_left"}
    assert not any(a.hold for a in ticks[:2]), "the first 100 ms of data are still fresh"
