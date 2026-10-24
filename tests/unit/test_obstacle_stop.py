"""Obstacle stop (WP-P4-01, audit N-11): guard, sampling, chokepoints, nav loop, monitor, wiring.

Fault injection on fakes: stale/blind/unknown sensors, an obstacle appearing mid-leg, a dead
monitor, a failing limiter, a Thor plan driven through the tractor service interface. The
guard runs on a fake clock wherever time matters.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.src.models import Position
from backend.src.models.mission import MissionWaypoint
from backend.src.models.safety_interlock import InterlockState, InterlockType
from backend.src.models.safety_limits import ObstacleStopLimits
from backend.src.models.tractor_control import EngineState, TractorCommand
from backend.src.safety.obstacle_stop import (
    ObstacleGuard,
    ObstacleStatus,
    ObstacleStopMonitor,
    limit_pair,
    sample_sensor_manager,
)
from backend.src.safety.safety_triggers import SafetyTriggerManager
from backend.src.services.navigation_service import NavigationService
from backend.src.services.tractor_service import TractorControlService


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_guard(clock=None, **limits) -> tuple[ObstacleGuard, Clock]:
    clock = clock or Clock()
    guard = ObstacleGuard(ObstacleStopLimits(**limits), clock=clock)
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    return guard, clock


def see(guard, clock, left=2.0, right=2.0, age=0.0) -> None:
    guard.observe_range("tof_left", left, clock.t - age)
    guard.observe_range("tof_right", right, clock.t - age)


def settle(guard, clock, **kw):
    """Feed fresh readings for 2.5 s (longer than the dwell) and return the last verdict."""
    for _ in range(25):
        clock.t += 0.1
        see(guard, clock, **kw)
        guard.assess()
    return guard.assess()


def armed_tractor(guard=None) -> TractorControlService:
    tractor = TractorControlService(config={})
    tractor.authorize()
    tractor.state.engine = EngineState.RUNNING
    tractor.motion_limiter = guard
    return tractor


# ------------------------------------------------------------------------------ guard


def test_no_data_at_all_holds_the_machine():
    guard, _ = make_guard()
    a = guard.assess()
    assert a.status is ObstacleStatus.STOP and a.speed_cap_mps == 0.0
    assert set(a.faults) == {"tof_left", "tof_right"}


def test_no_sources_configured_is_a_hold_not_a_free_pass():
    guard = ObstacleGuard(ObstacleStopLimits(), clock=Clock())
    a = guard.assess()
    assert a.hold and "no_sources" in a.faults


def test_clear_path_is_released_only_after_the_dwell():
    guard, clock = make_guard()
    see(guard, clock)
    assert guard.assess().hold  # starts held: nothing proves the path has been clear for long
    clock.t += 1.0
    see(guard, clock)
    assert guard.assess().hold  # 1 s clear < 2 s dwell
    clock.t += 1.0
    see(guard, clock)
    a = guard.assess()
    assert not a.hold and a.speed_cap_mps > 0


def test_a_stale_reading_holds_even_though_it_said_clear():
    guard, clock = make_guard()
    assert not settle(guard, clock).hold
    clock.t += 1.2  # nothing refreshes the readings: a dead producer
    a = guard.assess()
    assert a.hold and any("stale" in r for r in a.reasons)


def test_one_unknown_or_blind_source_holds_even_if_the_other_sees_clear_ground():
    guard, clock = make_guard()
    settle(guard, clock)
    guard.observe_unknown("tof_right", clock.t, "range_status=signal_fail")
    a = guard.assess()
    assert a.hold and a.faults == ("tof_right",) and not a.obstacles
    see(guard, clock)
    settle(guard, clock)
    guard.observe_blind("tof_left", clock.t)
    a = guard.assess()
    assert a.hold and a.obstacle_sources == ("tof_left",)


def test_an_obstacle_inside_the_stop_line_holds_and_names_the_source():
    guard, clock = make_guard()
    settle(guard, clock)
    see(guard, clock, left=0.15)  # at or below the 0.2 m static stop line
    a = guard.assess()
    assert a.hold and a.obstacles == (("tof_left", 0.15),)


def test_a_measured_obstacle_caps_speed_to_the_model_and_older_readings_cap_it_lower():
    guard, clock = make_guard()
    settle(guard, clock)
    see(guard, clock, right=0.9)
    a = guard.assess()
    expected = guard.model.max_speed_for_range(0.9)
    assert a.status is ObstacleStatus.SLOW and a.speed_cap_mps == pytest.approx(expected)
    assert 0 < a.speed_cap_mps < guard.limits.full_lever_speed_mps
    guard.observe_range("tof_right", 0.9, clock.t - 0.5)  # same range, half a second old
    older = guard.assess()
    assert older.speed_cap_mps < a.speed_cap_mps


def test_nothing_in_view_is_still_bounded_by_the_sensor_reach():
    guard, clock = make_guard()
    a = settle(guard, clock, left=5.0, right=5.0)
    assert a.status is ObstacleStatus.CLEAR
    assert a.speed_cap_mps == pytest.approx(guard.model.max_speed_for_range(1.0, extra_latency_s=0))
    assert a.speed_cap_mps < guard.limits.full_lever_speed_mps


def test_hold_is_released_only_after_the_path_stays_clear_for_the_whole_dwell():
    guard, clock = make_guard()
    settle(guard, clock)
    see(guard, clock, left=0.15)
    assert guard.assess().hold
    clock.t += 0.5
    see(guard, clock)  # obstacle gone ...
    assert guard.assess().hold
    clock.t += 1.5
    see(guard, clock, left=0.15)  # ... and back: the dwell restarts
    assert guard.assess().hold
    clock.t += 1.5
    see(guard, clock)
    assert guard.assess().hold  # only 0 s of clear since the restart
    clock.t += 1.9
    see(guard, clock)
    assert guard.assess().hold  # 1.9 s < 2.0 s
    clock.t += 0.2
    see(guard, clock)
    assert not guard.assess().hold


def test_ignored_sources_are_dropped_from_the_required_set():
    guard, clock = make_guard(ignored_sources=["tof_right"])
    guard.observe_range("tof_left", 2.0, clock.t)
    for _ in range(25):
        clock.t += 0.1
        guard.observe_range("tof_left", 2.0, clock.t)
        guard.assess()
    assert not guard.assess().hold  # tof_right never reports, and is not required


def test_manual_sources_are_not_limited_unless_configured():
    guard, _ = make_guard()
    assert guard.applies_to("navigation") and guard.applies_to("thor_plan")
    assert guard.applies_to("unspecified") and guard.applies_to(None)
    assert not guard.applies_to("operator") and not guard.applies_to("teleop")
    strict, _ = make_guard(apply_to_manual=True)
    assert strict.applies_to("operator")


@pytest.mark.parametrize(
    ("pair", "cap", "expected"),
    [
        ((0.8, 0.4), 0.2, (0.2, 0.1)),  # ratio (the curve) is kept
        ((0.8, 0.4), 0.9, (0.8, 0.4)),  # already inside the cap
        ((0.5, -0.5), 0.0, (0.0, 0.0)),  # a pivot is held too
        ((-0.6, -0.6), 0.0, (-0.6, -0.6)),  # reverse is not forward motion
        ((0.0, 0.0), 0.0, (0.0, 0.0)),
    ],
)
def test_limit_pair(pair, cap, expected):
    assert limit_pair(*pair, cap) == pytest.approx(expected)


# ------------------------------------------------------------------------------ sensors


def fake_manager(
    *, tof_mm=(2000, 2000), tof_status="online", range_status="valid", age_s=0.0, us=None
):
    stamp = datetime.now(UTC) - timedelta(seconds=age_s)

    def reading(side, mm):
        return SimpleNamespace(
            distance=mm, range_status=range_status, timestamp=stamp, sensor_side=side
        )

    tof = SimpleNamespace(
        status=tof_status,
        left_reading=reading("left", tof_mm[0]),
        right_reading=reading("right", tof_mm[1]),
    )
    ultrasonic = None
    if us is not None:
        ultrasonic = SimpleNamespace(last_readings=us, channel_ids=[e["sensor_id"] for e in us])
    return SimpleNamespace(tof=tof, ultrasonic=ultrasonic, ultrasonic_enabled=us is not None)


def real_guard(**limits) -> ObstacleGuard:
    guard = ObstacleGuard(ObstacleStopLimits(clear_dwell_s=0.0, **limits))
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    return guard


def test_sampler_turns_fresh_tof_readings_into_observations():
    guard = real_guard()
    sample_sensor_manager(guard, fake_manager(tof_mm=(1500, 900)))
    a = guard.assess()
    assert a.nearest_m == pytest.approx(0.9) and not a.hold
    assert a.status is ObstacleStatus.SLOW


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tof_status": "offline"},
        {"tof_status": "error"},
        {"range_status": "stale"},  # a replayed value: the driver's last read failed
        {"range_status": "unknown"},
        {"age_s": 5.0},  # the telemetry loop stopped refreshing the cache
        {"tof_mm": (None, 2000)},
    ],
)
def test_sampler_never_reads_a_bad_tof_as_clear(kwargs):
    guard = real_guard()
    sample_sensor_manager(guard, fake_manager(**kwargs))
    assert guard.assess().hold


def test_sampler_with_no_sensor_manager_ages_out_into_a_hold():
    clock = Clock()
    guard, _ = make_guard(clock)
    settle(guard, clock)
    sample_sensor_manager(guard, None)  # records nothing
    clock.t += 1.5
    assert guard.assess().hold


def us_entry(channel, **kw):
    entry = {
        "sensor_id": channel,
        "distance_cm": 0.0,
        "valid": False,
        "status": "no_data",
        "verdict": "unknown",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    entry.update(kw)
    return entry


def test_ultrasonic_channels_follow_the_interface_verdict_and_only_ever_tighten_it():
    guard = real_guard(ignored_sources=["tof_left", "tof_right"])
    ok = us_entry("a", distance_cm=80.0, valid=True, status="ok", verdict="clear")
    sample_sensor_manager(guard, fake_manager(us=[ok]))
    assert guard.assess().nearest_m == pytest.approx(0.8)  # a real echo is used as a distance

    clear = us_entry("a", verdict="clear", status="no_echo", clear_to_cm=90.0)
    sample_sensor_manager(guard, fake_manager(us=[clear]))
    a = guard.assess()
    assert not a.hold and a.speed_cap_mps == pytest.approx(
        guard.model.max_speed_for_range(0.9, extra_latency_s=0.0), abs=0.02
    )

    # a sensor whose proven reach is shorter than the slowest useful stopping distance
    # cannot permit any motion, however healthy it is
    short = us_entry("a", verdict="clear", status="no_echo", clear_to_cm=60.0)
    sample_sensor_manager(guard, fake_manager(us=[short]))
    assert guard.assess().hold

    for entry in (
        us_entry("a", verdict="unknown", status="no_echo"),  # silent line the policy won't trust
        us_entry("a", verdict="unknown", status="ok", valid=True, distance_cm=300.0),
        us_entry("a", verdict="clear", status="stale", timestamp=None),  # no timestamp
        us_entry("a", verdict="clear", status="no_echo", timestamp="2000-01-01T00:00:00+00:00"),
        us_entry("a", status="no_echo", verdict=None),  # no policy verdict: never loosen
    ):
        sample_sensor_manager(guard, fake_manager(us=[entry]))
        assert guard.assess().hold, entry

    blind = us_entry("a", status="blind_zone", verdict="unknown")
    sample_sensor_manager(guard, fake_manager(us=[blind]))
    a = guard.assess()
    assert a.hold and a.obstacle_sources == ("ultrasonic:a",)  # an object may be right there


def test_an_ultrasonic_channel_that_never_reports_still_blocks():
    guard = real_guard(ignored_sources=["tof_left", "tof_right"])
    manager = fake_manager(us=[us_entry("a", verdict="clear", status="no_echo"), us_entry("b")])
    manager.ultrasonic.last_readings = manager.ultrasonic.last_readings[:1]  # b never reported
    sample_sensor_manager(guard, manager)
    a = guard.assess()
    assert a.hold and "ultrasonic:b" in a.faults


# ----------------------------------------------------------------- tractor chokepoint


async def test_autonomous_levers_are_scaled_to_the_cap_and_manual_ones_are_not():
    guard, clock = make_guard()
    settle(guard, clock)
    see(guard, clock, left=0.9, right=0.9)
    cap = guard.assess().lever_cap
    tractor = armed_tractor(guard)

    res = await tractor.set_levers(0.9, 0.9, source="navigation")
    assert res["limited"] is True and tractor.state.left_lever == pytest.approx(cap)
    assert tractor.state.right_lever == pytest.approx(cap) and cap < 0.9

    res = await tractor.set_levers(0.9, 0.9, source="operator")
    assert res["limited"] is False and tractor.state.left_lever == pytest.approx(0.9)
    await tractor.shutdown()


async def test_a_thor_plan_driven_through_the_service_interface_cannot_outrun_a_hold():
    guard, clock = make_guard()  # no readings: held
    tractor = armed_tractor(guard)
    full = TractorCommand(left_lever=1.0, right_lever=1.0, throttle=0.7, blade_engaged=True)
    for source in ("thor_plan", "unspecified", "navigation", "ai"):
        res = await tractor.apply(full, source=source)
        assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0, source
        assert tractor.state.blade_engaged is False, source
        assert res["results"]["blade"]["status"] == "rejected"
        assert res["results"]["levers"]["limited"] is True
    # the default label is autonomous too: forgetting to say who is asking is not a bypass
    await tractor.set_levers(1.0, 1.0)
    assert tractor.state.left_lever == 0.0
    await tractor.shutdown()


async def test_a_hold_refuses_the_blade_and_drops_one_that_is_already_turning():
    guard, clock = make_guard()
    settle(guard, clock)
    tractor = armed_tractor(guard)
    assert (await tractor.engage_blade(True, source="navigation"))["status"] == "ok"
    see(guard, clock, left=0.1)  # an obstacle appears while the blade turns
    res = await tractor.engage_blade(True, source="navigation")  # navigation re-sends blade-on
    assert res["status"] == "rejected" and "obstacle hold" in res["reason"]
    assert tractor.state.blade_engaged is False and tractor.blade_pto.state is False
    await tractor.shutdown()


async def test_reverse_neutral_and_the_estop_are_never_limited():
    guard, _ = make_guard()  # held
    tractor = armed_tractor(guard)
    await tractor.set_levers(-0.5, -0.5, source="navigation")
    assert tractor.state.left_lever == -0.5  # nothing senses behind: reverse is not limited
    assert (await tractor.set_levers(0.0, 0.0, source="navigation"))["status"] == "ok"
    res = await tractor.emergency_stop()
    assert res["complete"] is True
    await tractor.shutdown()


async def test_a_limiter_that_raises_is_a_hold_never_a_free_pass():
    class Broken:
        def applies_to(self, source):
            return True

        def assess(self):
            raise RuntimeError("guard bug")

    tractor = armed_tractor()
    tractor.motion_limiter = Broken()
    res = await tractor.set_levers(0.6, 0.6, source="navigation")
    assert tractor.state.left_lever == 0.0 and res["limited"] is True
    assert "guard bug" in res["limit_reason"]
    assert tractor.get_state().motion_limited is True  # and it is visible at /tractor/state
    await tractor.shutdown()


async def test_enforce_motion_hold_stops_autonomous_motion_but_leaves_a_manual_drive_alone():
    guard, clock = make_guard()
    settle(guard, clock)
    tractor = armed_tractor(guard)
    await tractor.set_levers(0.3, 0.3, source="navigation")
    await tractor.engage_blade(True, source="navigation")
    see(guard, clock, left=0.1)
    await tractor.enforce_motion_hold()
    assert tractor.state.left_lever == 0.0 and tractor.state.blade_engaged is False

    settle(guard, clock)
    await tractor.set_levers(0.3, 0.3, source="operator")
    await tractor.engage_blade(True, source="operator")
    see(guard, clock, left=0.1)
    await tractor.enforce_motion_hold()
    assert tractor.state.left_lever == pytest.approx(0.3) and tractor.state.blade_engaged is True
    await tractor.shutdown()


async def test_enforce_motion_hold_attempts_both_stops_and_reports_the_failure():
    guard, clock = make_guard()
    settle(guard, clock)
    tractor = armed_tractor(guard)
    await tractor.set_levers(0.3, 0.3, source="navigation")
    await tractor.engage_blade(True, source="navigation")
    see(guard, clock, left=0.1)

    def stuck_relay(on):
        raise OSError("gpio busy")

    tractor.blade_pto.set = stuck_relay
    with pytest.raises(RuntimeError, match="blade off"):
        await tractor.enforce_motion_hold()
    assert tractor.state.left_lever == 0.0  # the levers were still neutralised
    assert tractor.state.blade_engaged is True  # and the blade failure is not hidden
    await tractor.shutdown()


# ---------------------------------------------------------------------- navigation loop


class Rec:
    """Tractor double for the nav loop: records every call, runs a scripted hook."""

    enabled = True

    def __init__(self, hook=None):
        self.state = SimpleNamespace(engine_running=True)
        self.calls: list[tuple] = []
        self.hook = hook

    async def apply(self, cmd, source=None):
        self.calls.append(("apply", cmd.left_lever, cmd.right_lever, cmd.blade_engaged, source))
        if self.hook:
            self.hook(self, "apply")
        return {"status": "applied"}

    async def set_levers(self, left, right, source=None):
        self.calls.append(("levers", left, right, source))
        if self.hook:
            self.hook(self, "levers")
        return {"status": "ok"}

    async def engage_blade(self, on, source=None):
        self.calls.append(("blade", on, source))
        return {"status": "ok"}

    async def emergency_stop(self, source=None):
        self.calls.append(("estop",))


def nav_rig(monkeypatch, guard, tractor, status=None):
    nav = NavigationService()
    nav.control_period_s = 0.001
    nav.attach_safety(obstacle_guard=guard)
    nav.navigation_state.current_position = Position(latitude=40.0, longitude=-83.0)
    nav.navigation_state.heading = 0.0
    waypoint = MissionWaypoint(lat=40.001, lon=-83.001, blade_on=True)
    statuses = {} if status is None else {"m1": status}
    missions = {} if status is None else {"m1": SimpleNamespace(waypoints=[waypoint])}
    monkeypatch.setattr(
        "backend.src.services.mission_service.get_mission_service",
        lambda: SimpleNamespace(missions=missions, mission_statuses=statuses),
    )
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    return nav, waypoint


async def test_mission_leg_holds_for_an_obstacle_that_appears_then_resumes_when_clear(monkeypatch):
    guard, clock = make_guard()
    settle(guard, clock)
    applies = 0

    def script(rec, kind):
        nonlocal applies
        if kind == "apply":
            applies += 1
            cap = guard.assess().lever_cap
            assert max(rec.calls[-1][1:3]) <= cap + 1e-9, "commanded more than the cap allows"
            if applies == 2:
                see(guard, clock, left=0.15)  # an obstacle steps in front
            elif applies == 3:
                nav.navigation_state.current_position = Position(latitude=40.001, longitude=-83.001)
        elif sum(1 for c in rec.calls if c[0] == "levers") == 5:
            # the obstacle leaves: fresh clear readings, and the clear dwell elapses
            see(guard, clock)
            guard.assess()
            clock.t += 2.5
            see(guard, clock)

    tractor = Rec(script)
    nav, waypoint = nav_rig(monkeypatch, guard, tractor)
    await asyncio.wait_for(nav.go_to_waypoint(waypoint), timeout=5)

    kinds = [c[0] for c in tractor.calls]
    second_apply = [i for i, k in enumerate(kinds) if k == "apply"][1]
    third_apply = [i for i, k in enumerate(kinds) if k == "apply"][2]
    held = tractor.calls[second_apply + 1 : third_apply]
    assert len([c for c in held if c[0] == "levers" and c[1:3] == (0.0, 0.0)]) >= 5
    assert ("blade", False, "navigation") in held, "the blade is dropped during the hold"
    assert all(c[0] != "apply" for c in held), "no motion command during the hold"
    assert tractor.calls[second_apply][4] == "navigation"  # autonomous label on every command
    assert ("estop",) not in tractor.calls  # a hold is not an E-stop


async def test_mission_leg_never_commands_more_than_the_cap_with_an_obstacle_in_view(monkeypatch):
    guard, clock = make_guard()
    settle(guard, clock)
    see(guard, clock, left=0.9, right=0.9)
    cap = guard.assess().lever_cap
    applies = 0

    def script(rec, kind):
        nonlocal applies
        see(guard, clock, left=0.9, right=0.9)
        applies += 1
        if applies == 4:
            nav.navigation_state.current_position = Position(latitude=40.001, longitude=-83.001)

    tractor = Rec(script)
    nav, waypoint = nav_rig(monkeypatch, guard, tractor)
    await asyncio.wait_for(nav.go_to_waypoint(waypoint), timeout=5)

    cmds = [c for c in tractor.calls if c[0] == "apply"]
    assert len(cmds) >= 4
    assert all(max(c[1], c[2]) <= cap + 1e-9 for c in cmds)
    assert max(max(c[1], c[2]) for c in cmds) == pytest.approx(cap)  # it does move, at the cap


async def test_a_held_mission_can_still_be_paused(monkeypatch):
    guard, _ = make_guard()  # held: no readings
    status = SimpleNamespace(status="running")
    holds = 0

    def script(rec, kind):
        nonlocal holds
        holds += 1
        if holds == 3:
            status.status = "paused"

    tractor = Rec(script)
    nav, waypoint = nav_rig(monkeypatch, guard, tractor, status)
    await asyncio.wait_for(nav.go_to_waypoint(waypoint), timeout=5)  # returns, not hangs
    assert not any(c[0] == "apply" for c in tractor.calls)
    assert holds >= 3


async def test_a_nav_loop_without_a_working_guard_holds_rather_than_driving(monkeypatch):
    class Broken:
        def assess(self):
            raise RuntimeError("guard bug")

    status = SimpleNamespace(status="running")
    calls = 0

    def script(rec, kind):
        nonlocal calls
        calls += 1
        if calls == 3:
            status.status = "paused"

    tractor = Rec(script)
    nav, waypoint = nav_rig(monkeypatch, Broken(), tractor, status)
    await asyncio.wait_for(nav.go_to_waypoint(waypoint), timeout=5)
    assert not any(c[0] == "apply" for c in tractor.calls)


async def test_differential_platform_set_speed_obeys_the_same_guard():
    guard, clock = make_guard()
    nav = NavigationService()
    nav.attach_safety(obstacle_guard=guard)
    await nav.set_speed(0.6, 0.6)  # held: no readings
    assert nav.navigation_state.target_velocity == 0.0
    settle(guard, clock)
    see(guard, clock, left=0.9, right=0.9)
    cap = guard.assess().speed_cap_mps
    await nav.set_speed(0.8, 0.4)
    assert nav.navigation_state.target_velocity == pytest.approx(cap)  # and the curve is kept


# ---------------------------------------------------------------------------- monitor


async def test_monitor_publishes_the_interlocks_and_stops_autonomous_motion_at_once():
    guard = real_guard()
    tractor = armed_tractor(guard)
    manager = fake_manager()
    triggers = SafetyTriggerManager()
    monitor = ObstacleStopMonitor(
        guard, lambda: manager, hold_handlers=[tractor.enforce_motion_hold], triggers=triggers
    )

    a = await monitor.step()
    assert not a.hold and not [
        i for i in triggers.list_active() if i.state is InterlockState.ACTIVE
    ]
    await tractor.set_levers(0.3, 0.3, source="navigation")
    await tractor.engage_blade(True, source="navigation")

    manager.tof.left_reading.distance = 150  # an obstacle 15 cm ahead of the left sensor
    manager.tof.left_reading.timestamp = datetime.now(UTC)
    a = await monitor.step()
    assert a.hold and a.obstacle_sources == ("tof_left",)
    assert tractor.state.left_lever == 0.0 and tractor.state.blade_engaged is False
    active = {i.interlock_type for i in triggers.list_active() if i.state is InterlockState.ACTIVE}
    assert active == {InterlockType.OBSTACLE_DETECTED}

    manager.tof.left_reading.distance = 2000
    manager.tof.left_reading.timestamp = datetime.now(UTC)
    a = await monitor.step()
    assert not a.hold
    assert not [i for i in triggers.list_active() if i.state is InterlockState.ACTIVE]
    await tractor.shutdown()


async def test_a_sensor_fault_holds_without_claiming_an_obstacle():
    guard = real_guard()
    triggers = SafetyTriggerManager()
    monitor = ObstacleStopMonitor(
        guard, lambda: fake_manager(tof_status="error"), triggers=triggers
    )
    a = await monitor.step()
    assert a.hold and not a.obstacles and set(a.faults) == {"tof_left", "tof_right"}
    assert not [i for i in triggers.list_active() if i.state is InterlockState.ACTIVE]


async def test_monitor_survives_a_failing_handler_and_a_failing_sensor_provider():
    guard = real_guard()

    async def broken(_a):
        raise RuntimeError("actuator bus down")

    monitor = ObstacleStopMonitor(
        guard,
        lambda: fake_manager(tof_status="error"),
        hold_handlers=[broken],
        triggers=SafetyTriggerManager(),
        interval_s=0.005,
    )
    await monitor.start()
    await asyncio.sleep(0.1)
    assert monitor.cycles >= 3 and monitor.errors >= 3  # kept cycling, failures counted
    await monitor.stop()

    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        raise RuntimeError("no sensor manager")

    monitor = ObstacleStopMonitor(
        guard, provider, triggers=SafetyTriggerManager(), interval_s=0.005
    )
    await monitor.start()
    await asyncio.sleep(0.05)
    await monitor.stop()
    assert calls >= 3 and monitor.errors >= 3
    assert guard.assess().hold  # nothing was sampled: fail closed


# ----------------------------------------------------------------------------- wiring


async def test_lifespan_attaches_the_guard_to_nav_and_tractor_and_detaches_it_on_shutdown(
    monkeypatch,
):
    import backend.src.main as main

    tractor = TractorControlService(config={})
    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)

    async def noop(*_a, **_k):
        return None

    class Stub:
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

    monkeypatch.setattr(main, "initialize_robohat_service", noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", noop)

    async def fake_park(**_k):
        return {"faults": []}

    monkeypatch.setattr(main, "park_actuators", fake_park)
    monkeypatch.setattr(main, "GPSDegradationMonitor", Stub)
    monkeypatch.setattr(main, "camera_service", Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", noop)

    nav = NavigationService.get_instance()
    async with main.lifespan(main.app):
        guard = main.app.state.obstacle_guard
        assert nav.obstacle_guard is guard and tractor.motion_limiter is guard
        assert nav.gps_monitor is main.app.state.gps_deg_monitor  # the GNSS verdict is attached too
        assert guard.limits.validated is False  # the shipped model is declared UNVALIDATED
        assert sorted(guard.sources) == ["tof_left", "tof_right"]
        await asyncio.sleep(0.12)  # no sensors in the test: the running monitor keeps it held
        assert main.app.state.obstacle_monitor.cycles >= 1 and guard.assess().hold
        res = await tractor.set_levers(0.5, 0.5, source="navigation")
        assert res["limited"] is True and tractor.state.left_lever == 0.0
    assert nav.obstacle_guard is None and nav.gps_monitor is None
    assert tractor.motion_limiter is None


async def test_tof_interface_marks_a_value_the_driver_replayed_as_stale_not_valid():
    from backend.src.models.sensor_data import SensorStatus
    from backend.src.services.sensor_manager import SensorCoordinator, ToFSensorInterface

    class Driver:
        def __init__(self, mm, ok):
            self.mm, self.last_read_ok = mm, ok

        async def read_distance_mm(self):
            return self.mm

    iface = ToFSensorInterface(SensorCoordinator())
    iface._left, iface._right = Driver(1200, True), Driver(1200, False)
    iface.status = SensorStatus.ONLINE
    left, right = await iface.read_tof_sensors()
    assert left.range_status == "valid"
    assert right.range_status == "stale" and right.distance == 1200.0  # visible, but not trusted

    guard = real_guard()
    sample_sensor_manager(
        guard, SimpleNamespace(tof=iface, ultrasonic=None, ultrasonic_enabled=False)
    )
    assert guard.assess().hold and "tof_right" in guard.assess().faults
