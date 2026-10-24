"""ThorPlanFollower: the local gate, plan-index rules, stand-down and the uplink it builds.

The follower runs against a real ``TractorControlService`` (SIM_MODE) and real
``NavigationState``; only the navigation loop is replaced by a recorder, so each test can
look at exactly which leg was requested, with what policy, behind which gate. The real loop
is exercised end to end in ``tests/integration/test_thor_link_e2e.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from backend.src.models import Geofence, LatLng
from backend.src.models.navigation_state import NavigationState, Position
from backend.src.models.sensor_data import GpsReading
from backend.src.models.thor_link import (
    MAX_DATAGRAM_BYTES,
    Downlink,
    Waypoint,
    decode_uplink,
    encode,
)
from backend.src.models.tractor_control import EngineState
from backend.src.nav.geofence_validator import build_shape
from backend.src.nav.path_planner import PathPlanner
from backend.src.services.thor_plan_follower import (
    FollowerConfig,
    FollowerState,
    GnssSnapshot,
    ThorPlanFollower,
    gnss_from_sensor_manager,
    link_fix_label,
)
from backend.src.services.tractor_service import TractorControlService

LAT0, LON0 = 45.0, -93.0
M = 111_195.0
KEY = bytes.fromhex("42" * 32)


def ll(east: float, north: float) -> tuple[float, float]:
    return LAT0 + north / M, LON0 + east / (M * math.cos(math.radians(LAT0)))


def wp(east: float, north: float, blade: bool = False) -> Waypoint:
    lat, lon = ll(east, north)
    return Waypoint(lat=lat, lon=lon, blade=blade)


def down(waypoints, *, start=0, plan_id="p", pause=False) -> Downlink:
    return Downlink(sid=1, ack=1, plan_id=plan_id, start_index=start, waypoints=waypoints)


class FakeLink:
    def __init__(self, plan: Downlink | None = None) -> None:
        self.plan = plan

    def current_plan(self) -> Downlink | None:
        return self.plan


class FakeNav:
    """Stands in for ``NavigationService``: records each requested leg and its gate."""

    def __init__(self) -> None:
        self.navigation_state = NavigationState()
        self.path_planner = PathPlanner()
        self.waypoint_tolerance = 0.5
        self.calls: list[tuple] = []
        self.on_leg = None  # optional async hook(leg, gate)
        self.fresh()

    def fresh(self, east: float = 5.0, north: float = 5.0) -> None:
        lat, lon = ll(east, north)
        s = self.navigation_state
        s.current_position = Position(latitude=lat, longitude=lon)
        s.heading = 0.0
        s.timestamp = datetime.now(UTC)

    async def go_to_waypoint(self, waypoint, *, should_continue=None):
        self.calls.append((waypoint, should_continue))
        if self.on_leg is not None:
            await self.on_leg(waypoint, should_continue)


class Rig:
    def __init__(self, *, fenced: bool = True, **cfg) -> None:
        cfg.setdefault("max_position_age_s", 30.0)
        self.tractor = TractorControlService(config={"enabled": True})
        self.tractor.authorize()
        self.tractor.state.engine = EngineState.RUNNING
        self.nav = FakeNav()
        self.snap = GnssSnapshot(*ll(5, 5), "rtk_fixed", 0.05)
        fence = None
        if fenced:
            fence = build_shape(
                Geofence(
                    geofence_id="f",
                    boundary=[
                        LatLng(latitude=a, longitude=b)
                        for a, b in (ll(-5, -5), ll(105, -5), ll(105, 105), ll(-5, 105))
                    ],
                ),
                no_go=[
                    [
                        LatLng(latitude=a, longitude=b)
                        for a, b in (ll(40, 40), ll(60, 40), ll(60, 60), ll(40, 60))
                    ]
                ],
                inset_m=0.5,
            )
        self.link = FakeLink(down([wp(5, 35, blade=True), wp(35, 35), wp(35, 80)]))
        self.follower = ThorPlanFollower(
            self.nav, self.tractor, lambda: self.snap, FollowerConfig(**cfg), fence
        )
        self.follower.bind(self.link)
        self.events: list = []
        self.tractor.on_change = lambda kind, values, source: self.events.append(
            (kind, values, source)
        )

    async def step(self) -> None:
        await self.follower.step()


# ------------------------------ pure helpers ------------------------------


@pytest.mark.parametrize(
    ("status", "sats", "expected"),
    [
        ("RTK_FIXED", 12, "rtk_fixed"),
        ("rtk_float", 12, "rtk_float"),
        ("DGPS", 12, "dgps"),
        ("GPS_FIX", 3, "gps"),
        (None, 8, "gps"),
        (None, 5, "none"),
        (None, None, "none"),
        ("NO_FIX", 20, "none"),
        ("DEAD_RECKONING", 20, "none"),
        ("MANUAL", 20, "none"),
        ("SIMULATION", 20, "none"),
    ],
)
def test_driver_fix_states_map_onto_the_link_vocabulary(status, sats, expected):
    assert link_fix_label(status, sats) == expected


def _manager(reading) -> SimpleNamespace:
    return SimpleNamespace(gps=SimpleNamespace(last_reading=reading))


def test_gnss_snapshot_reports_age_and_fix_without_touching_hardware():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    reading = GpsReading(
        latitude=LAT0, longitude=LON0, rtk_status="RTK_FIXED", timestamp=now - timedelta(seconds=2)
    )
    snap = gnss_from_sensor_manager(_manager(reading), now=now)
    assert (snap.lat, snap.lon, snap.fix) == (LAT0, LON0, "rtk_fixed")
    assert snap.age_s == pytest.approx(2.0)


def test_a_stalled_receiver_shows_as_an_old_reading_even_though_last_reading_is_kept():
    old = GpsReading(
        latitude=LAT0,
        longitude=LON0,
        rtk_status="RTK_FIXED",
        timestamp=datetime.now(UTC) - timedelta(minutes=5),
    )
    assert gnss_from_sensor_manager(_manager(old)).age_s > 299


@pytest.mark.parametrize(
    "reading",
    [
        None,
        SimpleNamespace(latitude=None, longitude=LON0, timestamp=datetime.now(UTC)),
        SimpleNamespace(latitude=math.nan, longitude=LON0, timestamp=datetime.now(UTC)),
        SimpleNamespace(latitude=LAT0, longitude=math.inf, timestamp=datetime.now(UTC)),
        SimpleNamespace(latitude=91.0, longitude=LON0, timestamp=datetime.now(UTC)),
        SimpleNamespace(latitude=LAT0, longitude=-181.0, timestamp=datetime.now(UTC)),
        SimpleNamespace(latitude=LAT0, longitude=LON0, timestamp="yesterday"),
    ],
)
def test_unusable_readings_read_as_no_fix(reading):
    snap = gnss_from_sensor_manager(_manager(reading))
    assert (snap.lat, snap.lon, snap.fix, snap.age_s) == (None, None, "none", None)
    assert gnss_from_sensor_manager(None).fix == "none"


def test_a_reading_stamped_in_the_future_is_as_unusable_as_an_old_one():
    now = datetime.now(UTC)
    future = GpsReading(
        latitude=LAT0, longitude=LON0, rtk_status="RTK_FIXED", timestamp=now + timedelta(hours=1)
    )
    assert gnss_from_sensor_manager(_manager(future), now=now).age_s == pytest.approx(3600.0)
    naive = GpsReading(
        latitude=LAT0, longitude=LON0, timestamp=datetime.now(UTC).replace(tzinfo=None)
    )
    assert gnss_from_sensor_manager(_manager(naive)).age_s is not None


@pytest.mark.parametrize(
    "bad",
    [
        {"speed_percent": 0},
        {"speed_percent": 101},
        {"max_position_age_s": 0},
        {"poll_s": 0},
        {"neutral_timeout_s": -1},
        {"require_fix": ()},
        {"require_fix": ("none",)},
        {"require_fix": ("rtk_fixed", "bogus")},
    ],
)
def test_follower_policy_refuses_values_that_would_loosen_the_gate(bad):
    with pytest.raises(ValueError):
        FollowerConfig(**bad)


# ------------------------- target selection and plan indices -------------------------


async def test_first_plan_is_adopted_at_its_start_index_and_the_leg_carries_local_policy():
    r = Rig(speed_percent=35)
    r.link.plan = down([wp(5, 35, blade=True), wp(35, 35)], start=7)
    await r.step()
    ((leg, gate),) = r.nav.calls
    expected = wp(5, 35)
    assert (leg.lat, leg.lon) == (expected.lat, expected.lon)
    assert leg.speed == 35, "speed is the Pi's, never the Thor's"
    assert leg.blade_on is False, "the plan asked for the blade; allow_blade is off"
    assert r.follower.plan_id == "p" and r.follower.index == 7
    assert callable(gate) and gate() is True
    await r.tractor.shutdown()


async def test_blade_is_requested_only_when_both_the_plan_and_the_pi_want_it():
    r = Rig(allow_blade=True)
    await r.step()
    assert r.nav.calls[0][0].blade_on is True
    r.nav.calls.clear()
    r.follower.index = 1  # the transit leg asks for no blade
    await r.step()
    assert r.nav.calls[0][0].blade_on is False
    await r.tractor.shutdown()


async def test_arrival_advances_the_index_and_the_next_step_takes_the_next_window_entry():
    r = Rig()

    async def arrive(leg, gate):
        r.nav.navigation_state.current_position = Position(latitude=leg.lat, longitude=leg.lon)

    r.nav.on_leg = arrive
    await r.step()
    assert (r.follower.index, r.follower.reached) == (1, 1)
    await r.step()
    leg = r.nav.calls[1][0]
    assert (leg.lat, leg.lon) == (wp(35, 35).lat, wp(35, 35).lon)
    assert (r.follower.index, r.follower.reached) == (2, 2)
    await r.tractor.shutdown()


async def test_a_leg_that_ends_without_arriving_does_not_count_as_progress():
    r = Rig()
    await r.step()  # the recorder returns at once: the mower is still 30 m short
    assert (r.follower.index, r.follower.reached) == (0, 0)
    await r.tractor.shutdown()


async def test_a_window_that_has_moved_on_without_the_pi_never_drives():
    r = Rig()
    await r.step()
    r.nav.calls.clear()
    # The Thor believes the mower is at waypoint 20; this Pi has reached only 0.
    r.link.plan = down([wp(35, 35), wp(35, 80)], start=20)
    await r.step()
    assert r.nav.calls == [] and r.follower.reason == "out_of_sync" and r.follower.index == 0
    # ... and neither does a window that stops short of the Pi's index.
    r.follower.index = 9
    r.link.plan = down([wp(35, 35), wp(35, 80)], start=2)
    await r.step()
    assert r.nav.calls == [] and r.follower.reason == "out_of_sync"
    await r.tractor.shutdown()


async def test_a_window_that_starts_behind_the_pi_but_still_contains_its_index_is_fine():
    r = Rig()
    r.follower.plan_id, r.follower.index = "p", 3
    r.link.plan = down([wp(1, 1), wp(2, 2), wp(3, 3), wp(5, 35), wp(35, 35)], start=0)
    await r.step()
    assert (r.nav.calls[0][0].lat, r.nav.calls[0][0].lon) == (wp(5, 35).lat, wp(5, 35).lon)
    assert r.follower.index == 3
    await r.tractor.shutdown()


async def test_a_new_plan_id_is_adopted_at_its_own_start_index():
    r = Rig()
    await r.step()
    assert r.follower.index == 0
    r.link.plan = down([wp(5, 35), wp(35, 35)], start=4, plan_id="other")
    await r.step()
    assert (r.follower.plan_id, r.follower.index) == ("other", 4)
    await r.tractor.shutdown()


async def test_an_empty_window_or_no_plan_or_no_link_is_idle():
    r = Rig()
    r.link.plan = down([], start=3)
    await r.step()
    assert r.follower.reason == "empty_window" and r.nav.calls == []
    r.link.plan = None
    await r.step()
    assert r.follower.reason == "link_hold"
    unbound = ThorPlanFollower(r.nav, r.tractor, lambda: r.snap)
    await unbound.step()
    assert unbound.reason == "no_link_bound" and r.nav.calls == []
    with pytest.raises(RuntimeError, match="already bound"):
        r.follower.bind(r.link)
    await r.tractor.shutdown()


# --------------------------------- the gate ---------------------------------


def _estop(r):
    r.tractor.state.emergency_stop_active = True


BLOCKS = {
    "estop": _estop,
    "interlock_latched": lambda r: setattr(r.tractor.state, "interlock_latched", True),
    "not_authorized": lambda r: r.tractor.revoke(),
    "engine_not_running": lambda r: setattr(r.tractor.state, "engine", EngineState.OFF),
    "actuator_fault": lambda r: r.tractor._write_faults.update({0: "PCA9685 write failed"}),
    "nav_mode_auto": lambda r: setattr(r.nav.navigation_state, "navigation_mode", "auto"),
    "nav_mode_ai": lambda r: setattr(r.nav.navigation_state, "navigation_mode", "ai"),
    "nav_mode_paused": lambda r: setattr(r.nav.navigation_state, "navigation_mode", "paused"),
    "nav_mode_emergency_stop": lambda r: setattr(
        r.nav.navigation_state, "navigation_mode", "emergency_stop"
    ),
    "nav_mode_return_home": lambda r: setattr(
        r.nav.navigation_state, "navigation_mode", "return_home"
    ),
    "no_position": lambda r: setattr(r.nav.navigation_state, "current_position", None),
    "no_heading": lambda r: setattr(r.nav.navigation_state, "heading", None),
    "nav_feed_stale": lambda r: setattr(
        r.nav.navigation_state, "timestamp", datetime.now(UTC) - timedelta(seconds=60)
    ),
    "gnss_stale": lambda r: setattr(r, "snap", GnssSnapshot(*ll(5, 5), "rtk_fixed", 60.0)),
    "fix_none": lambda r: setattr(r, "snap", GnssSnapshot(*ll(5, 5), "none", 0.05)),
    "fix_gps": lambda r: setattr(r, "snap", GnssSnapshot(*ll(5, 5), "gps", 0.05)),
    "fix_dgps": lambda r: setattr(r, "snap", GnssSnapshot(*ll(5, 5), "dgps", 0.05)),
    "fix_rtk_float": lambda r: setattr(r, "snap", GnssSnapshot(*ll(5, 5), "rtk_float", 0.05)),
    "outside_fence": lambda r: r.nav.fresh(500, 500),
    "leg_blocked": lambda r: setattr(
        r.link,
        "plan",
        down([wp(90, 50)], start=0),  # from (5, 5) the line to it crosses the zone
    ),
}


@pytest.mark.parametrize("reason", sorted(BLOCKS))
async def test_each_local_condition_blocks_the_leg(reason):
    r = Rig()
    if reason == "leg_blocked":
        r.nav.fresh(5, 50)
    BLOCKS[reason](r)
    await r.step()
    assert r.nav.calls == [], f"{reason} must stop a leg from being requested"
    assert r.follower.reason == reason
    assert r.follower.stand_downs[reason] == 1
    assert r.events == [], "a follower that never drove commands nothing"
    await r.tractor.shutdown()


async def test_a_gnss_receiver_that_was_never_read_counts_as_stale():
    r = Rig()
    r.snap = GnssSnapshot(None, None, "none", None)
    await r.step()
    assert r.follower.reason == "gnss_stale" and r.nav.calls == []
    await r.tractor.shutdown()


async def test_a_nan_heading_counts_as_no_heading():
    r = Rig()
    r.nav.navigation_state.heading = math.nan
    await r.step()
    assert r.follower.reason == "no_heading" and r.nav.calls == []
    await r.tractor.shutdown()


async def test_a_disabled_tractor_is_never_driven_through_the_differential_path():
    r = Rig()
    r.tractor.state.enabled = False
    await r.step()
    assert r.follower.reason == "tractor_disabled" and r.nav.calls == []
    await r.tractor.shutdown()


async def test_the_fix_requirement_is_the_owners_to_relax_and_only_in_the_listed_steps():
    r = Rig(require_fix=("rtk_fixed", "rtk_float"))
    r.snap = GnssSnapshot(*ll(5, 5), "rtk_float", 0.05)
    await r.step()
    assert len(r.nav.calls) == 1
    r.nav.calls.clear()
    r.snap = GnssSnapshot(*ll(5, 5), "dgps", 0.05)
    await r.step()
    assert r.nav.calls == [] and r.follower.reason == "fix_dgps"
    await r.tractor.shutdown()


async def test_without_a_configured_fence_only_the_other_gates_apply():
    r = Rig(fenced=False)
    r.nav.fresh(500, 500)
    await r.step()
    assert len(r.nav.calls) == 1
    await r.tractor.shutdown()


async def test_a_fence_that_cannot_be_evaluated_blocks_instead_of_waving_the_leg_through(caplog):
    r = Rig()
    r.follower._fence = SimpleNamespace()  # not a GeofenceShape
    with caplog.at_level(logging.ERROR):
        await r.step()
    assert r.follower.reason == "fence_check_failed" and r.nav.calls == []
    await r.tractor.shutdown()


async def test_the_gate_handed_to_the_nav_loop_tracks_the_plan_tick_by_tick():
    r = Rig(allow_blade=True)
    await r.step()
    gate = r.nav.calls[0][1]
    original = r.link.plan
    assert gate() is True

    r.link.plan = None  # the link went into hold
    assert gate() is False and r.follower.reason == "link_hold"
    r.link.plan = original
    assert gate() is True

    # the plan's waypoint for this index moved: aim at the new one, do not finish the old leg
    r.link.plan = down([wp(6, 35, blade=True), wp(35, 35), wp(35, 80)])
    assert gate() is False and r.follower.reason == "target_changed"
    # ... or only its blade request changed
    r.link.plan = down([wp(5, 35, blade=False), wp(35, 35), wp(35, 80)])
    assert gate() is False and r.follower.reason == "target_changed"
    # ... or the window no longer carries this index, or the Thor switched plans
    r.link.plan = down([wp(5, 35, blade=True)], start=5)
    assert gate() is False and r.follower.reason == "plan_changed"
    r.link.plan = down([wp(5, 35, blade=True)], plan_id="x")
    assert gate() is False and r.follower.reason == "plan_changed"
    r.link.plan = original
    assert gate() is True

    # a local block flips it too, and clearing the cause flips it back
    r.tractor.state.interlock_latched = True
    assert gate() is False and r.follower.reason == "interlock_latched"
    r.tractor.state.interlock_latched = False
    assert gate() is True
    await r.tractor.shutdown()


async def test_a_failing_gate_stops_the_leg_rather_than_letting_it_run_unchecked(caplog):
    r = Rig()
    await r.step()
    gate = r.nav.calls[0][1]

    def boom():
        raise RuntimeError("tractor state unreadable")

    r.tractor.get_state = boom
    with caplog.at_level(logging.ERROR):
        assert gate() is False
    assert r.follower.reason == "gate_error" and r.follower.errors == 1
    await r.tractor.shutdown()


async def test_the_gate_stops_the_leg_once_the_follower_is_asked_to_stop():
    r = Rig()
    await r.step()
    gate = r.nav.calls[0][1]
    assert gate() is True
    r.follower._stopping = True
    assert gate() is False
    await r.tractor.shutdown()


# --------------------------------- stand-down ---------------------------------


async def test_after_driving_a_blocked_follower_drops_the_blade_and_neutralises_the_levers():
    r = Rig(allow_blade=True)
    await r.tractor.engage_blade(True)
    await r.tractor.set_levers(0.5, 0.5, duration_ms=2000)
    r.events.clear()

    async def leg(leg, gate):
        r.tractor.state.interlock_latched = True
        assert gate() is False

    r.nav.on_leg = leg
    await r.step()
    s = r.tractor.state
    assert not s.blade_engaged and abs(s.left_lever) < 0.05 and abs(s.right_lever) < 0.05
    kinds = [(k, src) for k, _, src in r.events]
    assert ("blade", "thor_follower") in kinds and ("levers", "thor_follower") in kinds
    assert r.follower._driving is False
    await r.tractor.shutdown()


async def test_a_follower_that_never_drove_leaves_the_mower_to_whoever_is_driving_it():
    r = Rig()
    await r.tractor.engage_blade(True)
    await r.tractor.set_levers(0.5, 0.5, duration_ms=2000)  # an operator on teleop
    r.events.clear()
    r.tractor.state.interlock_latched = False
    r.tractor.revoke()  # blocks the follower ("not_authorized"), not the operator's levers
    await r.step()
    await r.step()
    assert r.events == [] and r.tractor.state.left_lever == 0.5 and r.tractor.state.blade_engaged
    await r.tractor.shutdown()


async def test_an_unconfirmed_stand_down_is_retried_and_not_forgotten(caplog):
    r = Rig(allow_blade=True)
    await r.tractor.engage_blade(True)
    await r.tractor.set_levers(0.5, 0.5, duration_ms=2000)
    real_engage = r.tractor.engage_blade
    attempts = {"n": 0}

    async def failing_engage(on, source="operator"):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise OSError("relay driver gone")
        return await real_engage(on, source=source)

    r.tractor.engage_blade = failing_engage

    async def leg(leg, gate):
        r.tractor.state.interlock_latched = True
        assert gate() is False

    r.nav.on_leg = leg
    with caplog.at_level(logging.ERROR):
        await r.step()
        assert r.tractor.state.blade_engaged, "the blade relay failed, so the blade is still on"
        assert r.follower._driving is True, "an unconfirmed stand-down must stay pending"
        assert abs(r.tractor.state.left_lever) < 0.05, "the lever call is still attempted"
        await r.step()  # blocked again (latched): retried
        assert r.follower._driving is True
        await r.step()
    assert not r.tractor.state.blade_engaged and r.follower._driving is False
    assert any("stand-down" in rec.getMessage() for rec in caplog.records)
    await r.tractor.shutdown()


async def test_a_hung_actuator_cannot_stall_the_follower_forever():
    r = Rig(neutral_timeout_s=0.05)
    await r.tractor.set_levers(0.5, 0.5, duration_ms=2000)
    r.follower._driving = True

    async def hang(*_a, **_k):
        await asyncio.sleep(30)

    r.tractor.engage_blade = hang
    r.tractor.set_levers = hang
    r.link.plan = None
    await asyncio.wait_for(r.step(), timeout=2.0)
    assert r.follower._driving is True
    r.tractor.state.left_lever = 0.0
    await r.tractor.shutdown()


async def test_the_follower_never_actuates_on_its_own_account_only_the_nav_loop_does():
    """Plan -> motion happens only inside ``go_to_waypoint``: with it stubbed out, nothing moves."""
    r = Rig(allow_blade=True)
    for _ in range(5):
        await r.step()
    assert len(r.nav.calls) == 5
    s = r.tractor.state
    assert s.left_lever == 0.0 and s.right_lever == 0.0 and s.throttle == 0.0
    assert not s.blade_engaged
    assert r.events == [], "no actuator was commanded by the follower itself"
    await r.tractor.shutdown()


# ------------------------------ task lifecycle ------------------------------


async def test_a_crashing_nav_loop_stands_the_mower_down_and_the_follower_keeps_running():
    r = Rig(poll_s=0.01)
    await r.tractor.engage_blade(True)
    await r.tractor.set_levers(0.5, 0.5, duration_ms=2000)
    calls = {"n": 0}

    async def crash_once(leg, gate):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("nav bug")

    r.nav.on_leg = crash_once
    await r.follower.start()
    await r.follower.start()  # a second start is a no-op
    for _ in range(200):
        await asyncio.sleep(0.02)
        if calls["n"] >= 2:
            break
    await r.follower.stop()
    assert r.follower.errors == 1 and calls["n"] >= 2, "it recovered after backing off"
    s = r.tractor.state
    assert not s.blade_engaged and abs(s.left_lever) < 0.05
    await r.follower.stop()  # idempotent
    await r.tractor.shutdown()


async def test_stop_ends_a_leg_in_progress_through_the_gate():
    r = Rig(poll_s=0.01)
    seen: list[bool] = []

    async def long_leg(leg, gate):
        while gate():
            await asyncio.sleep(0.01)
        seen.append(True)

    r.nav.on_leg = long_leg
    await r.follower.start()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if r.follower.state is FollowerState.DRIVING:
            break
    assert r.follower.state is FollowerState.DRIVING
    await asyncio.wait_for(r.follower.stop(), timeout=3.0)
    assert seen == [True] and r.follower._task.done()
    await r.tractor.shutdown()


# -------------------------------------- uplink --------------------------------------


def _uplink_rig(**kw) -> Rig:
    r = Rig(max_position_age_s=1.0, **kw)
    r.nav.navigation_state.heading = 90.0
    return r


async def test_uplink_carries_fresh_position_fix_heading_and_the_pis_progress():
    r = _uplink_rig()
    r.follower.index = 12
    up = r.follower.build_uplink()
    lat, lon = ll(5, 5)
    assert (up.lat, up.lon, up.fix, up.heading_deg, up.waypoint_index) == (
        lat,
        lon,
        "rtk_fixed",
        90.0,
        12,
    )
    decode_uplink(encode(up.model_copy(update={"sid": 1}), KEY), KEY)  # fits the datagram
    assert len(encode(up, KEY)) < MAX_DATAGRAM_BYTES
    await r.tractor.shutdown()


async def test_a_stale_gnss_reading_is_not_reported_as_a_position():
    r = _uplink_rig()
    r.snap = GnssSnapshot(*ll(5, 5), "rtk_fixed", 5.0)
    up = r.follower.build_uplink()
    assert (up.lat, up.lon, up.fix) == (None, None, "none"), "so the Thor pauses"
    r.snap = GnssSnapshot(None, None, "none", None)
    assert r.follower.build_uplink().fix == "none"
    await r.tractor.shutdown()


@pytest.mark.parametrize(("heading", "sent"), [(-90.0, 270.0), (360.0, 0.0), (725.0, 5.0)])
async def test_heading_is_normalised_and_non_finite_headings_are_dropped(heading, sent):
    r = _uplink_rig()
    r.nav.navigation_state.heading = heading
    assert r.follower.build_uplink().heading_deg == pytest.approx(sent)
    for bad in (math.nan, math.inf, None):
        r.nav.navigation_state.heading = bad
        assert r.follower.build_uplink().heading_deg is None
    await r.tractor.shutdown()


async def test_a_failing_sensor_read_still_produces_the_heartbeat(caplog):
    r = _uplink_rig()

    def boom():
        raise OSError("gps gone")

    r.follower._gnss = boom
    with caplog.at_level(logging.ERROR):
        up = r.follower.build_uplink()
    assert (up.lat, up.fix, up.waypoint_index) == (None, "none", None)
    await r.tractor.shutdown()


async def test_status_reports_what_the_follower_is_doing_and_why_it_is_not():
    r = Rig()
    BLOCKS["nav_mode_auto"](r)
    await r.step()
    st = r.follower.status()
    assert st["state"] == "idle" and st["reason"] == "nav_mode_auto" and st["plan_id"] == "p"
    assert st["stand_downs"] == {"nav_mode_auto": 1} and st["driving"] is False
    await r.tractor.shutdown()
