"""Mutants the round-4 tests left alive in the stop, latch, lifespan and route code (NAV-09/10).

The third review mutated the new safety code and listed the changes after which every test still
passed. Each of them is a claim the audit makes that nothing checked:

* the stop events: a lock-out, a park and the tractor's own E-stop each end the autonomous runs that
  are going, in the event-loop turn of the stop; a parked tractor's refusal ends a run by itself
  (T08, T11, T12, T13);
* the latch file: a file of an unknown version fails closed, and the process latch alone re-latches
  the RoboHAT and the tractor (L04, L06, L08, T06);
* the lifespan: the watchdog's trip is the all-platform stop, a start-up that fails closed
  switches the IBT-4 off at once, the shutdown stops the Thor link before the autonomy loops
  (M03, M06, M09);
* the clear route: it lists an IBT-4 whose OFF is unconfirmed and the latch file follows it
  (A02, A03);
* the orchestrator saves the latch while a stop path hangs and stops the tractor it was given
  (E02, E04); the ultrasonic intake refuses a time from the future (O05); the RoboHAT opens its
  port exclusively (R05).

Several of these layers protect each other (the stop event, the standing refusal of the tractor and
the epoch check of the loops each end a run), so a test that merely waits for the run to end would
pass without the layer it is named for. These tests make the other layers unable to answer in time
(a control tick of seconds, a loop that cannot see the refusal, a stop event that is not recorded)
and assert on the turn of the stop itself. The tractor is the real service over the stallable
PCA9685, the navigation service and the mission loop are real, the RoboHAT is the real service over
a fake serial line, the IBT-4 the real adapter on a fake ``lgpio``, the lifespan is the real one.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import pty
from types import SimpleNamespace

import httpx
import pytest
import serial
from _mission_rig import FAR, driving, new_nav, robohat_platform, wire_mission
from _nav_rig import (
    OFF,
    ON,
    armed_blade,
    live_sensors,
    real_lifespan,
    robohat_with_timeline,
    wait_until,
)
from _tractor_fakes import NEUTRAL, FakeGpiozero, idle_tractor, mowing_tractor

import backend.src.main as main
from backend.src.core import globals as core_globals
from backend.src.models import NavigationMode
from backend.src.models.action_prediction import ActionPrediction
from backend.src.models.mower_data_frame import MowerDataFrame
from backend.src.models.safety_limits import ObstacleStopLimits
from backend.src.safety import latch_store, stop_events
from backend.src.safety.obstacle_stop import ObstacleGuard, sample_sensor_manager
from backend.src.safety.watchdog import Watchdog
from backend.src.services import emergency_stop_service as ess
from backend.src.services import robohat_service as robohat_module
from backend.src.services.navigation_service import NavigationService
from backend.src.services.robohat_service import RoboHATService
from backend.src.services.tractor_service import TractorControlService

STOPS = ["lockout", "park", "estop"]


@pytest.fixture(autouse=True)
def _no_robohat_from_an_earlier_test(monkeypatch):
    """The process-wide RoboHAT service of an earlier test must not be this test's stop hold."""
    monkeypatch.setattr(robohat_module, "robohat_service", None)


# ================================================================ the stop events end the runs


async def _end(task: asyncio.Future | None) -> None:
    """Cancel a run's task and wait for it, again if the cancel was absorbed.

    ``TractorControlService._send_pwm`` wraps the PCA9685 write in ``asyncio.wait_for``; on Python
    3.11 a cancel that lands as that write completes is swallowed and a ticking loop runs on for
    ever (and the suite waits for it). A bounded number of retries ends it.
    """
    for _ in range(10):
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.wait({task}, timeout=1.0)


class Mowing:
    """A real mission mowing on the real tractor: the loop, the navigation service, the mission."""

    def __init__(self, monkeypatch, *, period_s: float) -> None:
        self.tractor, self.chip = mowing_tractor()
        self.nav = new_nav()
        self.nav.control_period_s = period_s
        self.mission, self.service = wire_mission(monkeypatch, self.tractor, self.nav, [FAR])
        self.task: asyncio.Task | None = None

    async def start(self) -> Mowing:
        self.task = asyncio.create_task(self.nav.execute_mission(self.mission))
        try:
            await wait_until(
                lambda: self.tractor.state.moving and self.tractor.state.blade_engaged, timeout=8.0
            )
        except BaseException:
            await self.close()
            raise
        return self

    @property
    def mission_status(self) -> str:
        return self.service.mission_statuses[self.mission.id].status

    @property
    def mode(self) -> NavigationMode:
        return self.nav.navigation_state.navigation_mode

    async def close(self) -> None:
        await _end(self.task)
        await self.tractor.shutdown()


class Model:
    """The AI inference service and the frame recorder, as ``start_ai_navigation`` finds them."""

    initialized = True

    async def enable(self) -> bool:
        return True

    async def disable(self) -> None:
        pass

    async def capture_frame(self) -> MowerDataFrame:
        return MowerDataFrame()

    async def infer(self, frame, coverage_grid=None, datum=None) -> ActionPrediction:
        return _good()


def _good() -> ActionPrediction:
    return ActionPrediction(steering=0.0, throttle=0.8, blade=True, confidence=0.9)


class Ai:
    """A real AI run on the real tractor: ``start_ai_navigation`` and its inference loop."""

    def __init__(self, monkeypatch) -> None:
        self.tractor, self.chip = mowing_tractor()
        model = Model()
        monkeypatch.setattr(
            "backend.src.services.ai_inference_service.get_ai_inference_service", lambda: model
        )
        monkeypatch.setattr(
            "backend.src.services.perimeter_recorder.get_perimeter_recorder", lambda: model
        )
        monkeypatch.setattr(
            "backend.src.services.tractor_service.get_tractor_service", lambda: self.tractor
        )
        self.nav = new_nav()
        # one tick, then ten seconds of sleep: only a stop EVENT can end the run in the turn of the
        # stop (the loop's own checks, and the tractor's refusals, need a tick)
        self.nav._ai_loop_hz = 0.1

    async def start(self) -> Ai:
        try:
            assert await self.nav.start_ai_navigation() is True
            await wait_until(
                lambda: self.tractor.state.moving and self.tractor.state.blade_engaged, timeout=8.0
            )
            assert self.nav.navigation_state.navigation_mode == NavigationMode.AI
        except BaseException:
            await self.close()
            raise
        return self

    @property
    def mode(self) -> NavigationMode:
        return self.nav.navigation_state.navigation_mode

    async def close(self) -> None:
        await _end(self.nav._ai_task)
        await self.tractor.shutdown()


async def _stop(tractor: TractorControlService, kind: str):
    """Start the stop and let it run to its first real wait: it must have announced itself by then.

    The lock-out is synchronous (nothing to wait for); the park and the E-stop are coroutines, run
    as tasks and given exactly one event-loop turn.
    """
    if kind == "lockout":
        tractor.lock_out("the safety wiring failed at start-up")
        return None
    stop = asyncio.ensure_future(
        tractor.park("shutdown") if kind == "park" else tractor.emergency_stop(source="api")
    )
    await asyncio.sleep(0)
    return stop


@pytest.mark.parametrize("kind", STOPS)
async def test_a_tractor_stop_ends_the_running_mission_in_the_turn_of_the_stop(monkeypatch, kind):
    """T11 (lock-out), T12 (park), T08 (the tractor's own E-stop): ``record_stop`` is what does it.

    The loop sleeps five seconds between ticks, so neither its own epoch check nor the standing
    refusal the stopped tractor gives it (both need a tick) can have ended the run when the
    assertion reads the mission: only the stop event, announced before the stop's first wait.
    """
    run = await Mowing(monkeypatch, period_s=5.0).start()
    try:
        epoch = stop_events.epoch()

        stop = await _stop(run.tractor, kind)

        assert run.mission_status == "aborted", "the mission was still running after the stop"
        assert run.mode == NavigationMode.IDLE
        assert run.nav.navigation_state.target_velocity == 0.0
        assert stop_events.epoch() > epoch, "no loop can tell that a stop happened"
        if stop is not None:
            await stop
    finally:
        await run.close()


@pytest.mark.parametrize("kind", STOPS)
async def test_a_tractor_stop_ends_the_running_ai_run_in_the_turn_of_the_stop(monkeypatch, kind):
    """The same three stops, for the AI run: the mode is off ``ai`` before the stop's first wait."""
    run = await Ai(monkeypatch).start()
    try:
        stop = await _stop(run.tractor, kind)

        assert run.mode == NavigationMode.IDLE, "the AI run was still going after the stop"
        assert run.nav.navigation_state.target_velocity == 0.0
        if stop is not None:
            await stop
    finally:
        await run.close()


@pytest.mark.parametrize("kind", STOPS)
async def test_a_tractor_stop_alone_ends_a_loop_that_cannot_see_any_refusal(monkeypatch, kind):
    """T11, T12, T08 again, isolated the other way round: the standing-refusal layer is blind.

    Every command the loop sends to the stopped tractor is refused for good and, blind to it, the
    loop goes on sending them. Without the stop's own event nothing tells it to end; with it the
    mission is aborted and the loop leaves at its next tick, whatever the tractor answers.
    """
    monkeypatch.setattr(NavigationService, "_standing_refusal", staticmethod(lambda result: None))
    run = await Mowing(monkeypatch, period_s=0.05).start()
    try:
        stop = await _stop(run.tractor, kind)
        if stop is not None:
            await stop

        await wait_until(lambda: run.task.done(), timeout=8.0)

        assert run.mission_status == "aborted"
        assert run.mode == NavigationMode.IDLE
    finally:
        await run.close()


async def test_a_parked_tractors_refusal_ends_the_mission_with_no_stop_event(monkeypatch):
    """T13: the real tractor is parked, nothing announced it: its ``parked`` answer ends the run."""
    run = await Mowing(monkeypatch, period_s=0.05).start()
    try:
        epoch = stop_events.epoch()
        monkeypatch.setattr(stop_events, "record_stop", lambda *_a, **_k: stop_events.epoch())

        await run.tractor.park("shutdown")

        assert stop_events.epoch() == epoch, "premise: this park announced nothing"
        refused = await run.tractor.set_levers(0.5, 0.5, source="navigation")
        assert refused["status"] == "rejected" and refused["code"] == "parked", refused
        await wait_until(lambda: run.task.done(), timeout=8.0)
        assert run.mission_status == "aborted"
        assert run.mode == NavigationMode.IDLE
    finally:
        await run.close()


async def test_a_parked_tractors_refusal_ends_the_ai_run_with_no_stop_event(monkeypatch):
    """T13 for the AI run, driven tick by tick: the first refused prediction ends the run."""
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    nav = new_nav()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    try:
        assert await nav.apply_ai_prediction(_good()) is True, "premise: the run is mowing"
        epoch = stop_events.epoch()
        monkeypatch.setattr(stop_events, "record_stop", lambda *_a, **_k: stop_events.epoch())
        await tractor.park("shutdown")
        assert stop_events.epoch() == epoch, "premise: this park announced nothing"

        assert await nav.apply_ai_prediction(_good()) is False

        assert nav.navigation_state.navigation_mode == NavigationMode.IDLE
        assert tractor.state.blade_engaged is False
    finally:
        await tractor.shutdown()


# ============================================================================ the latch file


def _latch_file(latches: dict | None = None, **record) -> None:
    """What the previous process left on disk: a version 1 record naming ``latches``."""
    body = {
        "version": 1,
        "saved_at": "2026-10-25T00:00:00+00:00",
        "latches": {} if latches is None else latches,
        **record,
    }
    latch_store.latch_path().write_text(json.dumps(body), encoding="utf-8")


UNKNOWN_VERSIONS = [
    pytest.param({"version": 2, "latches": {}}, id="a newer writer: nothing latched"),
    pytest.param({"version": 0, "latches": {"tractor_interlock": "tilt"}}, id="version 0"),
    pytest.param({"version": "1", "latches": {}}, id="the version as text"),
    pytest.param({"latches": {}}, id="no version"),
]


@pytest.mark.parametrize("record", UNKNOWN_VERSIONS)
async def test_a_latch_file_of_an_unknown_version_fails_closed(monkeypatch, record):
    """L04: a file of a format this code does not know is unreadable: every latch is assumed set."""
    latch_store.latch_path().write_text(json.dumps(record), encoding="utf-8")

    found = latch_store.load()

    assert found.unreadable, "an unknown version was read as if it were a record"
    assert all(found.has(name) for name in latch_store.LATCH_NAMES)
    # and what the restart does with it: everything latched, the RoboHAT hold included
    fresh = TractorControlService(config={"enabled": True})
    robohat = RoboHATService()
    latch_store.restore(tractor=fresh, robohat=robohat)
    assert core_globals._safety_state["emergency_stop_active"] is True
    assert fresh.state.emergency_stop_active is True
    assert robohat.emergency_latched is True
    refused = await fresh.set_levers(0.5, 0.5, source="operator")
    assert refused["status"] == "rejected", "even the operator's drive waits for the clear"
    await fresh.shutdown()


async def test_a_latch_file_of_the_current_version_latches_exactly_what_it_names():
    """The control of the test above: a version 1 record is read, not failed closed."""
    _latch_file({"tractor_interlock": "tilt"})

    found = latch_store.load()

    assert not found.unreadable and found.latches == {"tractor_interlock": "tilt"}
    assert found.has(latch_store.TRACTOR_INTERLOCK) and not found.has(latch_store.TRACTOR_ESTOP)
    fresh = TractorControlService(config={"enabled": True})
    latch_store.restore(tractor=fresh)
    assert fresh.state.interlock_latched is True and fresh.state.emergency_stop_active is False
    assert core_globals._safety_state["emergency_stop_active"] is False
    await fresh.shutdown()


async def test_the_process_latch_alone_re_latches_the_robohat_hold(monkeypatch):
    """L06: the file names ``process_estop`` only (the RoboHAT was unplugged when the stop ran)."""
    svc, fw, tractor = robohat_platform(monkeypatch)  # the differential platform, tractor disabled
    _latch_file({"process_estop": None})

    found = latch_store.restore(tractor=tractor, robohat=svc)

    assert found.has(latch_store.PROCESS_ESTOP) and not found.has(latch_store.ROBOHAT_HOLD)
    assert svc.emergency_latched is True, "the restarted RoboHAT was left free to drive"
    assert await svc.send_motor_command(0.5, 0.5) is False
    assert driving(fw.sent) == [], "a drive command reached the firmware under a restored stop"
    await svc.shutdown()


async def test_the_process_latch_alone_re_latches_the_tractors_estop():
    """L08: the file names ``process_estop`` only; the restarted tractor is E-stopped anyway."""
    _latch_file({"process_estop": None})
    fresh = TractorControlService(config={"enabled": True})
    assert fresh.state.emergency_stop_active is False

    found = latch_store.restore(tractor=fresh)

    assert found.has(latch_store.PROCESS_ESTOP) and not found.has(latch_store.TRACTOR_ESTOP)
    assert fresh.state.emergency_stop_active is True, "the restarted tractor was left free to drive"
    refused = await fresh.set_levers(0.5, 0.5, source="operator")
    assert refused["status"] == "rejected" and refused["code"] == "estop", refused
    await fresh.shutdown()


async def test_a_restore_takes_away_an_authorization_the_tractor_already_had():
    """T06: ``restore_latches`` never authorizes and revokes what it finds; a clear leaves it."""
    tractor, _chip = mowing_tractor()  # authorized, engine running
    assert tractor.state.authorized is True
    _latch_file({"tractor_estop": None})

    latch_store.restore(tractor=tractor)

    assert tractor.state.emergency_stop_active is True
    assert tractor.state.authorized is False
    cleared = await tractor.clear_emergency()
    assert cleared["status"] == "ok", cleared
    refused = await tractor.set_levers(0.5, 0.5, source="navigation")
    assert refused["status"] == "rejected" and refused["code"] == "not_authorized", refused
    await tractor.shutdown()


# =============================================================================== the lifespan


@pytest.fixture
async def world(monkeypatch):
    """Every platform at once, as the lifespan finds them.

    The RoboHAT is the real service over a fake serial line (``tl`` is what reached the firmware),
    the IBT-4 the real adapter on a fake ``lgpio`` (``pads`` is what a meter on GPIO24/25 shows),
    the tractor the real service over a stallable PCA9685 and fake relays. SIM_MODE is off.
    """
    svc, _fw, _idle, tl = robohat_with_timeline(monkeypatch)
    FakeGpiozero(monkeypatch)
    blade, pads = armed_blade(monkeypatch, deadman_ms=10000)
    tractor, chip = mowing_tractor(config={"actuation_supervision": {"interval_ms": 20}})
    yield SimpleNamespace(svc=svc, tl=tl, blade=blade, pads=pads, tractor=tractor, chip=chip)
    await blade.park()
    await tractor.shutdown()


async def test_a_watchdog_trip_in_the_lifespan_stops_the_robohat_and_the_ibt4_too(
    monkeypatch, world
):
    """M03: the lifespan's ``on_timeout`` is the all-platform stop, not the tractor's alone.

    A PCA9685 brown-out silences the actuation heartbeat and the real watchdog (its window
    shortened by the test, everything else as ``main.lifespan`` builds it) trips. The tractor-only
    stop would leave the RoboHAT driving and the IBT-4 without its latch (its supervisor merely
    notices the tractor's latch a moment later, and drops the blade without latching it).
    """
    real_watchdog = main.Watchdog
    monkeypatch.setattr(
        main,
        "Watchdog",
        lambda *a, **k: real_watchdog(*a, **{**k, "timeout_ms": 1500, "retrip_interval_s": 0.25}),
    )
    async with real_lifespan(monkeypatch, world.tractor, live_sensors(2000.0)) as app:
        watchdog = app.state.tractor_watchdog
        assert watchdog.running and watchdog.trips == 0
        assert await world.blade.set_active(True) is True and world.pads.pad == ON
        sent = len(world.tl.lines())

        world.chip.regs.clear()  # a brown-out: the self-check cannot read the outputs back

        await wait_until(lambda: world.svc.emergency_latched, timeout=10.0)
        assert watchdog.trips >= 1
        lines = world.tl.lines()[sent:]
        assert "blade=off" in lines and "pwm,1500,1500" in lines, lines
        assert lines.index("blade=off") < lines.index("pwm,1500,1500"), "blade first, then neutral"
        await wait_until(lambda: world.pads.pad == OFF, timeout=10.0)
        assert world.blade.emergency_latched is True, "the IBT-4 was never told of the stop"
        assert core_globals._safety_state["emergency_stop_active"] is True
        assert world.tractor.state.emergency_stop_active is True
        await wait_until(lambda: latch_store.load().has(latch_store.ROBOHAT_HOLD), timeout=10.0)
        assert latch_store.load().has(latch_store.PROCESS_ESTOP)


async def test_a_start_up_that_fails_closed_switches_a_running_ibt4_off_before_it_waits(
    monkeypatch, world
):
    """M06: the IBT-4 is a separate output; the fail-closed puts it OFF before it awaits anything.

    The watchdog cannot be started while the IBT-4 blade is turning. The first thing the
    fail-closed awaits is the watchdog's stop: the pads are read there, with no event-loop turn
    between the failure and that point, so the IBT-4's own supervisor (it acts on a tractor
    lock-out one 50 ms tick later) cannot have been the one to switch it off.
    """
    turning: list[tuple[bool, dict[int, int]]] = []
    at_first_wait: list[dict[int, int]] = []
    real_stop = Watchdog.stop

    async def watchdog_cannot_start(self) -> None:
        # recorded, not asserted: the lifespan swallows whatever a failing step raises
        accepted = await world.blade.set_active(True)
        turning.append((accepted, dict(world.pads.pad)))
        raise RuntimeError("the watchdog task could not be started")

    async def spying_stop(self) -> None:
        at_first_wait.append(dict(world.pads.pad))
        await real_stop(self)

    monkeypatch.setattr(Watchdog, "start", watchdog_cannot_start)
    monkeypatch.setattr(Watchdog, "stop", spying_stop)
    async with real_lifespan(monkeypatch, world.tractor, live_sensors(2000.0)):
        assert turning == [(True, ON)], "premise: the IBT-4 blade was turning at the failure"
        assert world.tractor.lockout_reason, "premise: the failed wiring locked the tractor out"
        assert at_first_wait, "premise: the fail-closed stopped the watchdog"
        assert at_first_wait[0] == OFF, "the IBT-4 was still turning when the fail-closed waited"
        assert world.pads.pad == OFF


async def test_the_shutdown_stops_the_thor_link_before_the_autonomy_loops_and_both_before_the_park(
    monkeypatch,
):
    """M09: a Thor leg still being followed could undo the stop of the loops (and the park)."""
    tractor, _chip = mowing_tractor()
    order: list[str] = []

    def stopped(name: str):
        async def stop() -> None:
            order.append(name)

        return stop

    thor = SimpleNamespace(
        follower=SimpleNamespace(stop=stopped("thor follower")),
        link=SimpleNamespace(stop=stopped("thor link")),
    )
    real_autonomy = main._stop_autonomy

    async def autonomy() -> None:
        order.append("autonomy loops")
        await real_autonomy()

    async def park(**_kwargs) -> dict:
        order.append("park")
        return {"faults": []}

    async with real_lifespan(monkeypatch, tractor, live_sensors(2000.0)) as app:
        app.state.thor_link = thor
        monkeypatch.setattr(main, "_stop_autonomy", autonomy)
        monkeypatch.setattr(main, "park_actuators", park)
        assert order == []

    assert order == ["thor follower", "thor link", "autonomy loops", "park"]


# ============================================================================ the clear route

CLEAR = "/api/v2/control/emergency_clear"
_clients = itertools.count(1)


def _client() -> httpx.AsyncClient:
    """An ASGI client with its own forwarded address: a fresh rate-limit bucket per test."""
    n = next(_clients)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app),
        base_url="http://test",
        headers={"X-Forwarded-For": f"10.8.{n // 250}.{n % 250 + 1}"},
    )


@pytest.fixture
def idle_process(monkeypatch):
    """No RoboHAT and a disabled tractor: the general E-stop and its clear are all there is."""
    monkeypatch.setattr(
        "backend.src.services.tractor_service.get_tractor_service", lambda: idle_tractor()
    )
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)


async def test_the_clear_lists_an_ibt4_whose_off_cannot_be_confirmed(monkeypatch, idle_process):
    """A02: the IBT-4 stays latched, so the answer is "partially cleared" and says which latch."""
    monkeypatch.setenv("SIM_MODE", "0")  # the real lgpio branch, on the fake pads
    blade, pads = armed_blade(monkeypatch)
    assert await blade.initialize() is True and await blade.set_active(True) is True
    pads.hold_pad(24, 1)  # the pad cannot be brought low: the stop cannot confirm the OFF
    report = await ess.execute_emergency_stop("api")
    assert report["subsystems"]["blade_service"]["status"] == "failed", report

    async with _client() as client:
        refused = await client.post(CLEAR, json={"confirmation": True})

        body = refused.json()
        assert refused.status_code == 200
        assert body["status"] == "EMERGENCY_PARTIALLY_CLEARED", body
        assert [item for item in body["still_latched"] if item.startswith("blade_service_latched")]
        assert blade.emergency_latched is True

        pads.release_pad(24)  # the pad follows again: the next clear gets the OFF confirmed
        cleared = await client.post(CLEAR, json={"confirmation": True})
        assert cleared.json()["status"] == "EMERGENCY_CLEARED", cleared.json()
    assert blade.emergency_latched is False and pads.pad == OFF
    await blade.park()


async def test_a_clear_removes_the_latch_file_when_nothing_is_latched_any_more(idle_process):
    """A03: the file mirrors the latches after the clear; a restart must not restore a stop."""
    await ess.execute_emergency_stop("api")
    path = latch_store.latch_path()
    assert latch_store.PROCESS_ESTOP in json.loads(path.read_text())["latches"]

    async with _client() as client:
        resp = await client.post(CLEAR, json={"confirmation": True})

    body = resp.json()
    assert body["status"] == "EMERGENCY_CLEARED" and body["latch_persisted"] is True, body
    assert not path.exists(), "the latch file still names a stop the operator cleared"
    assert not latch_store.restore(tractor=idle_tractor()), "and the next restart finds nothing"


async def test_a_clear_leaves_the_latch_file_holding_what_is_still_latched(
    monkeypatch, idle_process
):
    """A03: the tractor's own latch survives the general clear, in the answer and in the file."""
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    await ess.execute_emergency_stop("api", tractor=tractor)
    path = latch_store.latch_path()
    assert set(json.loads(path.read_text())["latches"]) == {"process_estop", "tractor_estop"}

    async with _client() as client:
        resp = await client.post(CLEAR, json={"confirmation": True})

    body = resp.json()
    assert body["status"] == "EMERGENCY_PARTIALLY_CLEARED", body
    assert body["still_latched"] == ["tractor_emergency_stop"] and body["latch_persisted"] is True
    assert json.loads(path.read_text())["latches"] == {"tractor_estop": None}
    await tractor.shutdown()


# ========================================================================== the orchestrator


class HungRoboHAT:
    """A RoboHAT whose stop does not answer until the test lets it."""

    def __init__(self) -> None:
        self.running = True
        self.status = SimpleNamespace(serial_connected=True)
        self.release = asyncio.Event()
        self.latched = False

    async def emergency_stop_detail(self) -> dict:
        self.latched = True
        await self.release.wait()
        return {"drive": "acked", "blade": "acked", "usb_control": True, "error": None}

    async def park(self) -> dict:
        return {"drive": "acked", "blade": "acked", "error": None}

    @property
    def emergency_latched(self) -> bool:
        return self.latched


class QuietBlade:
    """An IBT-4 service that confirms every stop at once."""

    emergency_latched = False

    async def emergency_stop(self) -> dict:
        return {"latched": True, "off": True, "initialized": True, "hardware_backed": True}

    async def park(self) -> dict:
        return {"latched": False, "off": True, "initialized": True, "hardware_backed": True}


def _wire(monkeypatch, tractor, robohat, blade) -> None:
    base = "backend.src.services"
    monkeypatch.setattr(f"{base}.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(f"{base}.robohat_service.get_robohat_service", lambda: robohat)
    monkeypatch.setattr(f"{base}.blade_service.get_blade_service", lambda: blade)


async def test_the_latch_is_on_disk_while_a_stop_path_is_still_hung(monkeypatch):
    """E02: a restart in the middle of a slow stop must not release it: the latch is saved first."""
    robohat = HungRoboHAT()
    _wire(monkeypatch, idle_tractor(), robohat, QuietBlade())
    stop = asyncio.ensure_future(ess.execute_emergency_stop("api", path_timeout_s=60.0))
    try:
        await wait_until(lambda: latch_store.latch_path().exists(), timeout=8.0)

        assert not stop.done() and ess.stop_in_flight() is True, "premise: the RoboHAT path hangs"
        saved = json.loads(latch_store.latch_path().read_text())["latches"]
        assert latch_store.PROCESS_ESTOP in saved
    finally:
        robohat.release.set()
        report = await stop
    assert report["emergency_stop_active"] is True and report["latch_persisted"] is True
    assert ess.stop_in_flight() is False


async def test_the_orchestrator_stops_the_tractor_it_was_given_not_the_singleton(monkeypatch):
    """E04: the orchestrator stops the tractor the watchdog hands it, not a stand-in singleton."""
    singleton = idle_tractor()  # what get_tractor_service() answers: nothing enabled to stop
    other, chip = mowing_tractor()
    _wire(monkeypatch, singleton, None, QuietBlade())
    assert (await other.set_levers(0.6, 0.6, source="operator", duration_ms=8000))["status"] == "ok"

    report = await ess.execute_emergency_stop("watchdog", tractor=other)

    assert other.state.emergency_stop_active is True, "the tractor handed over kept driving"
    assert other.state.left_lever == 0.0 and chip.regs == NEUTRAL
    assert report["subsystems"]["tractor"]["status"] in {"stopped", "simulated"}, report
    assert singleton.state.emergency_stop_active is False
    await other.shutdown()


# ======================================================================= the ultrasonic intake


class Clock:
    """The guard's monotonic clock, in the test's hands."""

    def __init__(self) -> None:
        self.t = 5000.0

    def __call__(self) -> float:
        return self.t


def _ultrasonic_guard() -> tuple[ObstacleGuard, Clock]:
    clock = Clock()
    limits = ObstacleStopLimits(clear_dwell_s=0.0, ignored_sources=["tof_left", "tof_right"])
    guard = ObstacleGuard(limits, clock=clock)
    guard.register_source("tof_left", "tof")
    guard.register_source("tof_right", "tof")
    return guard, clock


def _ultrasonic_manager(**entry) -> SimpleNamespace:
    """What the sensor manager caches for one ultrasonic channel: a valid reading, 1.5 m away."""
    reading = {
        "sensor_id": "a",
        "valid": True,
        "status": "ok",
        "verdict": "clear",
        "distance_cm": 150.0,
        **entry,
    }
    us = SimpleNamespace(last_readings=[reading], channel_ids=["a"])
    return SimpleNamespace(tof=None, ultrasonic=us, ultrasonic_enabled=True)


def test_an_ultrasonic_reading_a_good_moment_ago_is_a_measurement():
    """The control of the test below: the same entry, stamped in the past, clears the path."""
    guard, clock = _ultrasonic_guard()

    sample_sensor_manager(guard, _ultrasonic_manager(measured_mono=clock.t - 0.05))

    verdict = guard.assess()
    assert not verdict.hold and verdict.speed_cap_mps > 0.0, verdict.reasons
    assert guard.snapshot()["sources"]["ultrasonic:a"]["state"] == "range"


@pytest.mark.parametrize(
    ("ahead_s", "later_s"),
    [
        pytest.param(0.5, 0.6, id="half a second ahead, the clock then passes the stamp"),
        pytest.param(60.0, 0.6, id="a minute ahead"),
    ],
)
def test_an_ultrasonic_reading_stamped_in_the_future_is_unusable_and_never_turns_fresh(
    ahead_s, later_s
):
    """O05: the entry's own monotonic stamp is ahead of the guard's clock: no measurement at all.

    It is unknown at once (the telemetry and the Thor uplink show no distance for it) and it stays
    unknown when the guard's clock later passes the stamp without a new sample: a time that was
    from the future when the reading was first seen never ages into a fresh measurement.
    """
    guard, clock = _ultrasonic_guard()

    sample_sensor_manager(guard, _ultrasonic_manager(measured_mono=clock.t + ahead_s))

    verdict = guard.assess()
    assert verdict.hold and verdict.faults == ("ultrasonic:a",), verdict.reasons
    source = guard.snapshot()["sources"]["ultrasonic:a"]
    assert source["state"] == "unknown" and source["distance_m"] is None, source

    clock.t += later_s
    later = guard.assess()
    assert later.hold and later.faults == ("ultrasonic:a",), later.reasons
    assert guard.snapshot()["sources"]["ultrasonic:a"]["state"] == "unknown"


# ====================================================================== the RoboHAT's own port


class _QuickAsyncio:
    """``robohat_service``'s view of ``asyncio`` with its 2 s port-settle sleep made short."""

    def __getattr__(self, name: str):
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(delay: float, *args, **kwargs):
        return await asyncio.sleep(0.01 if delay >= 2.0 else delay, *args, **kwargs)


async def test_the_robohat_service_holds_its_tty_against_a_second_opener(monkeypatch):
    """R05: the service opens its port exclusively; a second exclusive opener is refused.

    The port is a real pty and the opener the real pyserial, so the refusal is the kernel's, not a
    fake's. The GPS probes open every candidate port the same way (SAF-S1): a service that did
    not lock its port would be shared with them and lose the firmware's acknowledgements.
    """
    try:
        master, slave = pty.openpty()
    except OSError as exc:  # a sandbox without /dev/ptmx: there is no tty to hold
        pytest.skip(f"no pseudo-terminal here: {exc}")
    monkeypatch.setattr(robohat_module, "asyncio", _QuickAsyncio())
    path = os.ttyname(slave)
    svc = RoboHATService(path)

    async def end_service() -> None:
        svc.running = False
        tasks = [t for t in (svc.watchdog_task, svc.read_task) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if svc.serial_conn is not None:
            svc.serial_conn.close()

    try:
        assert await svc.initialize() is True, svc.status.last_error

        with pytest.raises(serial.SerialException):
            serial.Serial(path, 115200, exclusive=True)

        await end_service()
        serial.Serial(path, 115200, exclusive=True).close()  # released with it: the next owner
    finally:
        await end_service()
        os.close(master)
        os.close(slave)
