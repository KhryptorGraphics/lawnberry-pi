"""GNSS-quality monitor (T063; audit N-31, WP-P4-03).

The first two tests are the legacy ``RobotState`` fallback (debug-route ``AUTONOMOUS`` mode),
kept as they were. The rest drive the monitor the way production does: with real
``GpsReading`` objects, through ``NavigationService.update_navigation_state``, on a fake
clock, and watch what the navigation loop and the tractor actually do.
"""

import asyncio
import datetime as dt
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.src.core.robot_state_manager import get_robot_state_manager
from backend.src.models import Position
from backend.src.models.mission import Mission, MissionWaypoint
from backend.src.models.robot_state import NavigationMode
from backend.src.models.safety_limits import GpsDegradationLimits
from backend.src.models.sensor_data import GpsReading, SensorData
from backend.src.models.tractor_control import EngineState
from backend.src.nav.gps_degradation import (
    GPSDegradationConfig,
    GPSDegradationMonitor,
)
from backend.src.safety.obstacle_stop import ObstacleGuard
from backend.src.services.navigation_service import NavigationService
from backend.src.services.tractor_service import TractorControlService

# ---------------------------------------------------------------- legacy RobotState path


@pytest.mark.asyncio
async def test_gps_degradation_switches_to_manual_on_poor_accuracy():
    os.environ["SIM_MODE"] = "1"

    # Ensure AUTONOMOUS mode with poor accuracy
    mgr = get_robot_state_manager()
    st = mgr.get_state()
    st.navigation_mode = NavigationMode.AUTONOMOUS
    st.position.latitude = 37.0
    st.position.longitude = -122.0
    st.position.accuracy_m = 6.5  # > 5m threshold
    st.touch()

    monitor = GPSDegradationMonitor(
        GPSDegradationConfig(max_accuracy_m=5.0, max_fix_age_s=10.0, check_interval_s=0.05)
    )
    await monitor.start()
    await asyncio.sleep(0.2)
    await monitor.stop()

    assert st.navigation_mode == NavigationMode.MANUAL


@pytest.mark.asyncio
async def test_gps_degradation_switches_to_manual_on_fix_timeout():
    os.environ["SIM_MODE"] = "1"

    mgr = get_robot_state_manager()
    st = mgr.get_state()
    st.navigation_mode = NavigationMode.AUTONOMOUS
    st.position.latitude = 37.0
    st.position.longitude = -122.0
    st.position.accuracy_m = 1.0  # good accuracy, but we'll age it out
    # Manually set last_updated sufficiently in the past
    st.last_updated = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1.0)

    monitor = GPSDegradationMonitor(
        GPSDegradationConfig(max_accuracy_m=5.0, max_fix_age_s=0.2, check_interval_s=0.05)
    )
    await monitor.start()
    await asyncio.sleep(0.3)
    await monitor.stop()

    assert st.navigation_mode == NavigationMode.MANUAL


# ------------------------------------------------------------------------- helpers

EPOCH = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)


class Clocks:
    """One fake time for the monitor's monotonic clock and its wall clock."""

    def __init__(self) -> None:
        self.t = 500.0

    def mono(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return EPOCH + timedelta(seconds=self.t - 500.0)

    def stamp(self, age: float = 0.0) -> datetime:
        return EPOCH + timedelta(seconds=self.t - 500.0 - age)


def fix(c: Clocks, *, acc=0.05, status="RTK_FIXED", age=0.0, lat=40.0, lon=-83.0) -> GpsReading:
    return GpsReading(
        latitude=lat, longitude=lon, accuracy=acc, rtk_status=status, timestamp=c.stamp(age)
    )


def monitor(c: Clocks, **kw) -> GPSDegradationMonitor:
    cfg = GPSDegradationConfig(
        max_accuracy_m=1.0,
        max_fix_age_s=2.0,
        resume_dwell_s=3.0,
        check_interval_s=kw.pop("check_interval_s", 0.2),
    )
    return GPSDegradationMonitor(cfg, clock=c.mono, wall_clock=c.wall, **kw)


def settle(m: GPSDegradationMonitor, c: Clocks, seconds: float = 3.4, **kw):
    """Fresh good fixes every 0.2 s for ``seconds``; returns the verdict."""
    for _ in range(round(seconds / 0.2) + 1):
        c.t += 0.2
        m.observe(fix(c, **kw))
    return m.assess()


# ------------------------------------------------------------------- the monitor itself


def test_monitor_starts_degraded_so_a_mission_never_starts_on_no_fix():
    c = Clocks()
    a = monitor(c).assess()
    assert a.degraded and not a.healthy
    assert "no GNSS reading yet" in a.reasons and a.fix_age_s is None


def test_a_good_fix_must_hold_for_the_resume_dwell_before_the_machine_may_move():
    c = Clocks()
    m = monitor(c)
    m.observe(fix(c))
    a = m.assess()
    assert a.degraded and a.reasons[0].startswith("recovering")  # a usable fix is not enough yet
    for k in range(1, 15):  # 2.8 s of healthy fixes
        c.t = 500.0 + 0.2 * k
        m.observe(fix(c))
    assert m.assess().degraded
    c.t = 503.0
    m.observe(fix(c))  # exactly the 3.0 s dwell
    a = m.assess()
    assert a.healthy and a.reasons == () and a.accuracy_m == 0.05
    assert a.rtk_status == "RTK_FIXED"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"acc": 3.2}, "accuracy 3.20m exceeds 1m"),
        ({"acc": None}, "accuracy unknown"),  # an unknown accuracy is not a good accuracy
        ({"status": "NO_FIX"}, "fix status NO_FIX"),
        ({"status": "DEAD_RECKONING"}, "fix status DEAD_RECKONING"),
        ({"lat": None}, "no position"),
        ({"lat": 0.0, "lon": 0.0}, "no position"),  # the (0, 0) of a receiver with no fix
        ({"age": 2.5}, "old"),
    ],
)
def test_each_way_a_fix_can_be_unusable_degrades_a_healthy_machine(kwargs, reason):
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    c.t += 0.2
    m.observe(fix(c, **kwargs))
    a = m.assess()
    assert a.degraded and any(reason in r for r in a.reasons), a.reasons


def test_a_single_bad_observation_between_polls_restarts_the_dwell():
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    c.t += 0.5
    m.observe(fix(c, acc=2.5))  # nobody polled assess() during this blip
    c.t += 0.1
    m.observe(fix(c))
    a = m.assess()
    assert a.degraded and a.reasons[0].startswith("recovering")


def test_fix_age_is_the_age_of_the_fix_and_a_replayed_reading_keeps_ageing():
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    replayed = fix(c)  # the last real fix; then the receiver goes quiet
    for _ in range(12):  # the sensor interface replays it on every poll
        c.t += 0.2
        m.observe(replayed)
    a = m.assess()
    assert a.degraded and any("old" in r for r in a.reasons)
    assert a.fix_age_s == pytest.approx(2.4, abs=0.01)  # not reset by 12 'observations'


def test_a_quiet_receiver_with_no_new_readings_at_all_also_goes_stale():
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    c.t += 2.5  # nothing is observed any more (telemetry loop dead): polls alone age it
    a = m.assess()
    assert a.degraded and a.fix_age_s == pytest.approx(2.5, abs=0.01)


def test_defaults_match_the_shipped_limits_and_come_from_config():
    assert GPSDegradationConfig() == GPSDegradationConfig.from_limits(GpsDegradationLimits())
    limits = GpsDegradationLimits(max_accuracy_m=0.3, max_fix_age_s=1.5, resume_dwell_s=9.0)
    cfg = GPSDegradationConfig.from_limits(limits)
    assert (cfg.max_accuracy_m, cfg.max_fix_age_s, cfg.resume_dwell_s) == (0.3, 1.5, 9.0)


async def test_monitor_task_stops_the_machine_only_while_degraded_and_autonomy_runs():
    c = Clocks()
    calls = []
    running = {"on": False}

    async def handler(assessment):
        calls.append(assessment)

    m = monitor(
        c, check_interval_s=0.005, autonomy_active=lambda: running["on"], hold_handlers=[handler]
    )
    await m.start()
    await asyncio.sleep(0.05)
    assert calls == []  # degraded, but nothing autonomous is moving: report only
    running["on"] = True
    await asyncio.sleep(0.05)
    assert len(calls) >= 3 and all(a.degraded for a in calls)
    assert settle(m, c).healthy  # a healthy fix: the handler goes quiet
    await asyncio.sleep(0.03)  # let a call that was already in flight land
    seen = len(calls)
    await asyncio.sleep(0.05)
    assert len(calls) == seen
    await m.stop()


async def test_a_failing_hold_handler_is_counted_and_never_stops_the_monitor():
    c = Clocks()

    async def broken(_a):
        raise RuntimeError("actuator bus down")

    m = monitor(c, check_interval_s=0.005, autonomy_active=lambda: True, hold_handlers=[broken])
    await m.start()
    await asyncio.sleep(0.08)
    assert m.cycles >= 4 and m.errors >= 4
    await m.stop()


# ------------------------------------------------- navigation: the real production path


class Rec:
    """Tractor double for the nav loop: records every call, runs a scripted hook."""

    enabled = True

    def __init__(self, hook=None):
        self.state = SimpleNamespace(engine_running=False)
        self.calls: list[tuple] = []
        self.hook = hook
        self.start_engine_calls = 0

    async def start_engine(self):
        self.start_engine_calls += 1
        self.state.engine_running = True
        return {"status": "ok"}

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


def nav_rig(monkeypatch, tractor, gps=None, guard=None, status=None):
    nav = NavigationService()
    nav.control_period_s = 0.001
    nav.attach_safety(obstacle_guard=guard, gps_monitor=gps)
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


async def test_update_navigation_state_feeds_the_monitor_the_real_reading():
    c = Clocks()
    m = monitor(c)
    nav = NavigationService()
    nav.attach_safety(gps_monitor=m)
    reading = fix(c, acc=0.7, status="RTK_FLOAT")
    await nav.update_navigation_state(SensorData(gps=reading))
    a = m.assess()
    assert a.accuracy_m == 0.7 and a.rtk_status == "RTK_FLOAT"
    # the navigation state records the fix's own time, not the moment it was looked at
    assert nav.navigation_state.last_gps_fix == reading.timestamp

    c.t += 5.0
    old = fix(c, age=4.0)  # a replayed reading from 4 s ago
    await nav.update_navigation_state(SensorData(gps=old))
    assert nav.navigation_state.last_gps_fix == old.timestamp
    assert any("old" in r for r in m.assess().reasons)


async def test_gnss_loss_during_a_mission_leg_holds_levers_and_blade_then_resumes(monkeypatch):
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    applies = 0

    def script(rec, kind):
        nonlocal applies
        if kind == "apply":
            applies += 1
            if applies == 2:
                c.t += 0.2
                m.observe(fix(c, acc=4.0))  # the fix degrades mid-leg (float -> single)
            elif applies == 3:
                nav.navigation_state.current_position = Position(latitude=40.001, longitude=-83.001)
        elif sum(1 for x in rec.calls if x[0] == "levers") == 5:
            settle(m, c)  # the fix is good again and stays good for the resume dwell

    tractor = Rec(script)
    nav, waypoint = nav_rig(monkeypatch, tractor, gps=m)
    await asyncio.wait_for(nav.go_to_waypoint(waypoint), timeout=5)

    applies_at = [i for i, x in enumerate(tractor.calls) if x[0] == "apply"]
    assert len(applies_at) == 3
    held = tractor.calls[applies_at[1] + 1 : applies_at[2]]
    assert len([x for x in held if x[0] == "levers" and x[1:3] == (0.0, 0.0)]) >= 5
    assert ("blade", False, "navigation") in held  # blade dropped, levers neutral: at the actuators
    assert all(x[0] != "apply" for x in held)  # no motion command while the fix is bad
    assert ("estop",) not in tractor.calls  # a hold, not an E-stop


async def test_a_fix_that_goes_stale_with_no_new_reading_holds_the_leg(monkeypatch):
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    status = SimpleNamespace(status="running")
    ticks = 0

    def script(rec, kind):
        nonlocal ticks
        ticks += 1
        if kind == "apply":
            c.t += 0.5  # time passes, the receiver (or the telemetry loop) is silent
        elif ticks > 12:
            status.status = "paused"

    tractor = Rec(script)
    nav, waypoint = nav_rig(monkeypatch, tractor, gps=m, status=status)
    await asyncio.wait_for(nav.go_to_waypoint(waypoint), timeout=5)
    applies = [x for x in tractor.calls if x[0] == "apply"]
    assert 1 <= len(applies) <= 5  # a few commands while the last fix was still fresh ...
    assert tractor.calls[-1][0] == "levers"  # ... then holds (and the pause is honoured)
    assert any(x[0] == "blade" and x[1] is False for x in tractor.calls)


async def test_a_mission_will_not_start_without_a_healthy_fix(monkeypatch):
    c = Clocks()
    m = monitor(c)
    tractor = Rec()
    nav, _ = nav_rig(monkeypatch, tractor, gps=m)
    mission = Mission(name="m", waypoints=[], created_at="2026-10-24T12:00:00+00:00")

    with pytest.raises(RuntimeError, match="GNSS fix is not healthy.*no GNSS reading yet"):
        await nav.execute_mission(mission)
    assert tractor.start_engine_calls == 0  # refused before the engine was cranked

    m.observe(fix(c))  # a usable fix that has not yet held for the dwell is still refused
    with pytest.raises(RuntimeError, match="recovering"):
        await nav.execute_mission(mission)

    assert settle(m, c).healthy
    await nav.execute_mission(mission)  # healthy: the (empty) mission runs to completion
    assert tractor.start_engine_calls == 1


async def test_hold_for_gnss_stops_a_running_autonomous_leg_between_planner_ticks(monkeypatch):
    tractor = TractorControlService(config={"enabled": True})
    tractor.authorize()
    tractor.state.engine = EngineState.RUNNING
    await tractor.set_levers(0.4, 0.4, source="navigation")
    await tractor.engage_blade(True, source="navigation")
    nav = NavigationService()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    await nav.hold_for_gnss(SimpleNamespace(reasons=("fix is 3.0s old",)))
    assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0
    assert tractor.state.blade_engaged is False
    assert nav._hold_reasons == ("gnss: fix is 3.0s old",)
    await tractor.shutdown()


async def test_a_gnss_monitor_that_raises_is_a_hold_never_a_free_pass():
    class Broken:
        def assess(self):
            raise RuntimeError("monitor bug")

        def observe(self, reading):
            raise RuntimeError("monitor bug")

    nav = NavigationService()
    nav.attach_safety(gps_monitor=Broken())
    gate = nav._motion_gate()
    assert gate.hold and gate.speed_cap_mps == 0.0 and "monitor bug" in gate.reasons[0]
    await nav.update_navigation_state(SensorData())  # a failing feed is logged, not raised


async def test_differential_set_speed_and_the_ai_path_obey_the_gnss_hold(monkeypatch):
    c = Clocks()
    m = monitor(c)  # no fix yet: held
    nav = NavigationService()
    nav.attach_safety(gps_monitor=m)
    await nav.set_speed(0.6, 0.6)
    assert nav.navigation_state.target_velocity == 0.0

    nav.navigation_state.navigation_mode = "ai"
    tractor = Rec()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    applied = await nav.apply_ai_prediction(SimpleNamespace(confidence=0.9, safety_override=False))
    assert applied is False
    assert ("levers", 0.0, 0.0, "navigation") in tractor.calls
    assert not any(x[0] == "apply" for x in tractor.calls)


async def test_gnss_and_obstacle_holds_combine_and_either_one_holds():
    c = Clocks()
    m = monitor(c)
    assert settle(m, c).healthy
    clock = SimpleNamespace(t=1000.0)
    guard = ObstacleGuard(clock=lambda: clock.t)
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    nav = NavigationService()
    nav.attach_safety(obstacle_guard=guard, gps_monitor=m)

    gate = nav._motion_gate()  # fix healthy, no obstacle data yet: the obstacle guard holds
    assert gate.hold and not any(r.startswith("gnss") for r in gate.reasons)

    for _ in range(26):  # clear ground on both sensors for longer than the clear dwell
        clock.t += 0.1
        guard.observe_range("tof_left", 2.0, clock.t)
        guard.observe_range("tof_right", 2.0, clock.t)
        guard.assess()
    gate = nav._motion_gate()
    assert not gate.hold and gate.speed_cap_mps and gate.speed_cap_mps > 0

    c.t += 3.0  # the GNSS fix goes stale while the path is clear
    gate = nav._motion_gate()
    assert gate.hold and gate.speed_cap_mps == 0.0
    assert any(r.startswith("gnss: fix is") for r in gate.reasons)


async def test_lifespan_wires_the_real_gnss_monitor_into_navigation_and_its_hold(monkeypatch):
    import backend.src.main as main

    tractor = TractorControlService(config={})
    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)

    async def noop(*_a, **_k):
        return None

    async def fake_park(**_k):
        return {"faults": []}

    class Stub:
        async def initialize(self):
            pass

        async def start_streaming(self):
            pass

        async def shutdown(self):
            pass

    monkeypatch.setattr(main, "initialize_robohat_service", noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", noop)
    monkeypatch.setattr(main, "park_actuators", fake_park)
    monkeypatch.setattr(main, "camera_service", Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", noop)

    nav = NavigationService.get_instance()
    nav.navigation_state.navigation_mode = "idle"
    try:
        async with main.lifespan(main.app):
            monitor_ = main.app.state.gps_deg_monitor
            assert isinstance(monitor_, GPSDegradationMonitor) and nav.gps_monitor is monitor_
            limits = main.app.state.safety_limits.gps_degradation
            assert monitor_.cfg == GPSDegradationConfig.from_limits(limits)
            gate = nav._motion_gate()  # the lifespan starts degraded: no fix has been seen
            assert gate.hold and any(r.startswith("gnss:") for r in gate.reasons)

            await asyncio.sleep(0.5)  # autonomy idle: the monitor reports, it does not act
            assert nav._hold_reasons is None

            nav.navigation_state.navigation_mode = "auto"  # a mission runs on a bad fix
            nav._mission_loop_active = True  # ... its loop: autonomy_active() keys on it (NAV-03)
            await asyncio.sleep(0.6)  # two monitor cycles: the hold handler stopped the machine
            assert nav._hold_reasons is not None and nav._hold_reasons[0].startswith("gnss:")
        assert nav.gps_monitor is None and nav.obstacle_guard is None  # detached again
    finally:
        nav.navigation_state.navigation_mode = "idle"
        nav._mission_loop_active = False
        nav._hold_reasons = None
