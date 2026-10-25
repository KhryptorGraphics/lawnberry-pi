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
from backend.src.safety.obstacle_stop import (
    TOF_KIND,
    TOF_SOURCES,
    ObstacleGuard,
    ObstacleStatus,
    sample_sensor_manager,
)
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

    def __init__(self, monkeypatch, sensors, cfg=None, limits=None, wrap=None, tof_clear=False):
        self.clock = FakeClock()
        patch_guard_wall_clock(monkeypatch, self.clock)
        self.factory = FakeMcuFactory(self.clock, sensors=sensors)
        self.iface = make(self.clock, self.factory, **(cfg or {}))
        ultrasonic = wrap(self.iface) if wrap else self.iface
        self.tof = self._clear_tof() if tof_clear else None
        self.manager = SimpleNamespace(tof=self.tof, ultrasonic=ultrasonic, ultrasonic_enabled=True)
        self.guard = ObstacleGuard(ObstacleStopLimits(**(limits or {})), clock=self.clock.now)
        if tof_clear:  # what build_obstacle_guard does: the ToF sources are required sources
            for source in TOF_SOURCES:
                self.guard.register_source(source, TOF_KIND)
        self._ticks = 0

    def _clear_tof(self):
        """Two ToF sensors that always see 2 m: the only thing left to hold the guard is us."""

        def reading():
            return SimpleNamespace(distance=2000, range_status="valid", timestamp=self.clock.wall())

        return SimpleNamespace(
            status=SimpleNamespace(value="online"), left_reading=reading(), right_reading=reading()
        )

    def _refresh_tof(self) -> None:
        if self.tof is None:
            return
        self._ticks += 1
        for side in (self.tof.left_reading, self.tof.right_reading):
            side.timestamp = self.clock.wall()
            side.distance = 2000 + self._ticks % 7  # a frozen reading would be unknown

    async def start(self) -> Stack:
        assert await self.iface.initialize() is True
        return self

    async def tick(self, dt: float = 0.02, *, poll: bool = True, read: bool = True):
        """One guard cycle: service loop, (optionally) the telemetry read, sample, assess."""
        self.clock.advance(dt)
        self._refresh_tof()
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


async def test_the_rig_with_clear_tof_sensors_reaches_clear_when_the_ultrasonics_are_fitted(
    monkeypatch,
):
    # the control of the test below: if this rig could not reach CLEAR, "still holds" proves nothing
    s = await stack(monkeypatch, [far(150), far(), far(), far(180)], tof_clear=True)
    last = (await s.run(6.0))[-1]
    assert last.status is ObstacleStatus.CLEAR and not last.hold and not last.faults
    assert {"tof_left", "tof_right"} <= set(s.guard.sources)


async def test_with_every_channel_disabled_the_guard_still_holds_and_the_aggregate_is_unknown(
    monkeypatch,
):
    # US3-04: with no ToF sources registered an ObstacleGuard already answers STOP ("no obstacle
    # sources configured"), so this could not fail. With two clear ToF sensors registered, only
    # the ultrasonic side of the intake can hold the guard: every channel disabled is no free pass
    s = await stack(monkeypatch, [far()] * 4, cfg={"disabled_channels": NAMES}, tof_clear=True)
    ticks = await s.run(6.0)
    assert {a.status for a in ticks[-100:]} == {ObstacleStatus.STOP}
    last = ticks[-1]
    assert all("tof" not in fault for fault in last.faults), "the ToF side is healthy"
    assert set(last.faults) == {f"ultrasonic:{n}" for n in NAMES}
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


# ---- guard intake (G04 / G06 / G07 / G10): what the guard does with the entries it is given ----


class Tampered:
    """The REAL interface with one key of one channel's entry changed (a corrupted entry)."""

    def __init__(self, iface, channel: str, **changes):
        self._iface, self._channel, self._changes = iface, channel, changes

    @property
    def channel_ids(self):
        return self._iface.channel_ids

    @property
    def last_readings(self):
        return [
            {**e, **self._changes} if e["sensor_id"] == self._channel else e
            for e in self._iface.last_readings
        ]


def tamper(channel: str, **changes):
    return lambda iface: Tampered(iface, channel, **changes)


async def test_g04_an_entry_whose_valid_flag_is_not_true_is_never_taken_as_a_measured_range(
    monkeypatch,
):
    sensors = [far(150), far(), far(), far()]
    s = await stack(monkeypatch, sensors, wrap=tamper("outer_left", valid=False))
    last = (await s.run(4.0))[-1]
    # outer_left (150 cm, verdict clear) is a clear-horizon observation, not a range: the nearest
    # measured range is the 2.0 m of the other channels, not 1.5 m
    assert last.nearest_m == pytest.approx(2.0)
    assert not last.hold


async def test_g10_a_valid_entry_without_a_distance_is_unknown_never_clear(monkeypatch):
    sensors = [far(150), far(), far(), far()]
    s = await stack(monkeypatch, sensors, wrap=tamper("outer_left", distance_cm=None))
    last = (await s.run(4.0))[-1]
    assert last.hold and OUTER_LEFT in last.faults
    assert f"{OUTER_LEFT}: valid reading without a distance" in last.reasons


async def test_g11_an_entry_dated_in_the_future_is_unusable_not_fresh_for_ever(monkeypatch):
    # O01: a measured_mono beyond "now" would age as negative: fresh until the clock caught up
    sensors = [far(150), far(), far(), far()]
    s = await stack(monkeypatch, sensors, wrap=tamper("outer_left", measured_mono=1e9))
    last = (await s.run(4.0))[-1]
    assert last.hold and OUTER_LEFT in last.faults
    assert any("no usable timestamp" in r and OUTER_LEFT in r for r in last.reasons)


async def test_g07_the_ultrasonic_enabled_gate_keeps_a_switched_off_array_out_of_the_guard(
    monkeypatch,
):
    on = await stack(monkeypatch, [far()] * 4)
    await on.run(1.0)
    assert OUTER_LEFT in on.guard.sources  # enabled: every fitted channel is a required source
    off = await stack(monkeypatch, [far()] * 4)
    off.manager.ultrasonic_enabled = False  # the master switch in limits.yaml is off
    await off.run(1.0)
    assert not [name for name in off.guard.sources if name.startswith("ultrasonic:")]


async def test_g06_the_guard_ages_a_valid_range_entry_by_its_own_timestamp(monkeypatch):
    # stale_after_ms 5 s: the INTERFACE keeps calling the frozen data valid, so only the guard's
    # own max_reading_age_s (1 s) can stop it. Re-stamping it as "just measured" would never.
    s = await stack(monkeypatch, [far(150), far(), far(), far(180)], cfg={"stale_after_ms": 5000})
    assert not (await s.run(4.0))[-1].hold
    t_quiet = s.clock.now()  # from here on nothing reaches the driver
    for _ in range(40):
        a = await s.tick(0.05, poll=False, read=False)
        if a.hold:
            break
    assert a.hold and 0.6 < s.clock.now() - t_quiet < 1.3
    assert any("stale" in reason for reason in a.reasons)
    assert all(e["valid"] and e["status"] == "ok" for e in s.iface.last_readings)


async def test_g12_an_entry_whose_status_is_not_ok_is_never_taken_as_a_measured_range(
    monkeypatch,
):
    # O04: `valid` True next to a status that is not "ok" (a tampered or foreign entry) must not
    # become a measured 20 cm obstacle: it is what its status and verdict say, a clear horizon
    sensors = [far(150), far(), far(), far()]
    tampered = tamper(
        "outer_left",
        status="out_of_range",
        distance_cm=20.0,
        verdict="clear",
        clear_to_cm=400.0,
    )
    s = await stack(monkeypatch, sensors, wrap=tampered)
    last = (await s.run(4.0))[-1]
    assert last.nearest_m == pytest.approx(2.0)
    assert not last.hold
