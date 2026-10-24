"""End to end over real UDP on loopback: Thor client <-> ThorLinkService <-> follower <-> mower.

Everything between the two sockets is the production code: ``start_thor_link`` builds the
authenticated link and the plan follower, the follower drives each leg through
``NavigationService.go_to_waypoint``, and that commands a ``TractorControlService`` in
SIM_MODE (simulated PCA9685 and relays). The Thor is ``scripts/thor_link_client.py``. Only
the mower's body is stood in for: ``Plant`` feeds position, heading and the navigation
timestamp the way the telemetry loop does, and a test moves the mower by setting that
position (the steering law is not under test here).

What is proven: a plan from the Thor is accepted and driven; losing the Thor holds the mower
(blade off, levers neutral) and it stays held; a returning Thor recovers it; datagrams from
anyone without the key, replays, and plans outside the fence never move it; and every local
interlock (E-stop, tilt latch, stale position, degraded fix, fence, a mission, a limiter at the
lever chokepoint) wins over a plan that stays valid on the wire the whole time (a "blind" Thor
that never pauses).

Timing is real (link timeout 0.5 s, uplinks 40 Hz, navigation tick 0.2 s) and nothing here
measures latency on hardware: waits are polls with generous ceilings.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import math
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.src.models.navigation_state import Position
from backend.src.models.thor_link import MAX_WAYPOINTS, Downlink, Waypoint, encode
from backend.src.models.tractor_control import LEVER_NEUTRAL_EPS, EngineState
from backend.src.safety.safety_triggers import SafetyTriggerManager
from backend.src.safety.tractor_safety_monitor import TractorSafetyMonitor
from backend.src.services.navigation_service import NavigationService
from backend.src.services.thor_link_service import HoldReason
from backend.src.services.thor_link_wiring import (
    FenceSettings,
    ThorLinkRuntime,
    ThorLinkSettings,
    start_thor_link,
    stop_thor_link,
)
from backend.src.services.thor_plan_follower import GnssSnapshot
from backend.src.services.tractor_service import TractorControlService

_spec = importlib.util.spec_from_file_location(
    "scripts.thor_link_client", Path(__file__).resolve().parents[2] / "scripts/thor_link_client.py"
)
thor_client = importlib.util.module_from_spec(_spec)
sys.modules["scripts.thor_link_client"] = thor_client
_spec.loader.exec_module(thor_client)

KEY = bytes.fromhex(
    "6b1ecbe04e1f6197ffd65f59ff3baa94bfdd6cca75eac11be83c9bf3da0582008991ed4fe6452786"
)
OTHER_KEY = bytes.fromhex(
    "705eefb5c4b96bfa3cfcaee64245af7ade9f4a0470722e8e590e2bf112a8e186a9482dd2f5d9e179"
)
LAT0, LON0 = 45.0, -93.0
M_PER_DEG = 111_195.0
NEUTRAL = LEVER_NEUTRAL_EPS


def ll(east: float, north: float) -> tuple[float, float]:
    """Metres east/north of the yard origin as (lat, lon)."""
    return (
        LAT0 + north / M_PER_DEG,
        LON0 + east / (M_PER_DEG * math.cos(math.radians(LAT0))),
    )


def wp(east: float, north: float, blade: bool = False) -> Waypoint:
    lat, lon = ll(east, north)
    return Waypoint(lat=lat, lon=lon, blade=blade)


def fence() -> FenceSettings:
    """A 110 m square with a 20 m no-go square in the middle, 0.5 m margin."""
    return FenceSettings(
        boundary=[ll(-5, -5), ll(105, -5), ll(105, 105), ll(-5, 105)],
        no_go=[[ll(40, 40), ll(60, 40), ll(60, 60), ll(40, 60)]],
        margin_m=0.5,
    )


# Start -> A -> B -> C stay clear of the no-go square (x 40..60, y 40..60).
PLAN = [wp(5, 35, blade=True), wp(35, 35, blade=True), wp(35, 80)]


def blind_thor(waypoints: list[Waypoint]):
    """A Thor planner that never pauses: it ignores fix, e-stop and hold flags (a broken one)."""

    def planner(up):
        i = min(max(up.waypoint_index or 0, 0), len(waypoints))
        return Downlink(
            sid=up.sid,
            ack=up.seq,
            plan_id="blind",
            start_index=i,
            waypoints=waypoints[i : i + MAX_WAYPOINTS],
        )

    return planner


async def eventually(predicate, timeout: float = 6.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


async def holds_for(predicate, seconds: float, what: str) -> None:
    """``predicate`` is true at every poll for ``seconds``."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        assert predicate(), f"{what} stopped holding"
        await asyncio.sleep(0.02)


class Plant:
    """The mower's body and sensors as seen by the backend: feeds the navigation state."""

    def __init__(self, nav: NavigationService, east: float, north: float) -> None:
        self.nav = nav
        self.east, self.north = east, north
        self.heading = 0.0
        self.fix = "rtk_fixed"
        self.gnss_age_s = 0.02  # how old the GNSS reading is
        self.nav_feed_alive = True  # False: the telemetry loop has died
        self._task: asyncio.Task | None = None

    def teleport(self, east: float, north: float) -> None:
        self.east, self.north = east, north
        self._feed()

    def snapshot(self) -> GnssSnapshot:
        lat, lon = ll(self.east, self.north)
        return GnssSnapshot(lat, lon, self.fix, self.gnss_age_s)

    def _feed(self) -> None:
        if not self.nav_feed_alive:
            return
        lat, lon = ll(self.east, self.north)
        state = self.nav.navigation_state
        state.current_position = Position(latitude=lat, longitude=lon)
        state.heading = self.heading
        state.timestamp = datetime.now(UTC)

    async def _run(self) -> None:
        while True:
            self._feed()
            await asyncio.sleep(0.03)

    async def start(self) -> None:
        self._feed()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


@dataclass
class Stack:
    tractor: TractorControlService
    nav: NavigationService
    plant: Plant
    runtime: ThorLinkRuntime
    client: object
    events: list = field(default_factory=list)

    @property
    def link(self):
        return self.runtime.link

    @property
    def follower(self):
        return self.runtime.follower

    @property
    def pi_addr(self) -> tuple[str, int]:
        return ("127.0.0.1", self.link._transport.get_extra_info("sockname")[1])

    def moving(self) -> bool:
        s = self.tractor.state
        return abs(s.left_lever) > NEUTRAL or abs(s.right_lever) > NEUTRAL

    def neutral(self) -> bool:
        return not self.moving() and not self.tractor.state.blade_engaged

    def drive_events_after(self, t: float) -> list:
        """Lever commands with real motion in them, issued after monotonic time ``t``."""
        return [
            e
            for e in self.events
            if e[0] >= t
            and e[1] == "levers"
            and (abs(e[2]["left"]) > NEUTRAL or abs(e[2]["right"]) > NEUTRAL)
        ]


@contextlib.asynccontextmanager
async def pi_stack(
    monkeypatch,
    *,
    plan: list[Waypoint] | None = None,
    planner=None,
    allow_blade: bool = False,
    start: tuple[float, float] = (5.0, 5.0),
    authorize: bool = True,
    client_key: bytes = KEY,
):
    """A Pi with the link on, a Thor client answering, and an authorized, running mower."""
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("THOR_LINK_KEY", KEY.hex())
    tractor = TractorControlService(config={"enabled": True})
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    if authorize:
        tractor.authorize()
    tractor.state.engine = EngineState.RUNNING
    nav = NavigationService()
    plant = Plant(nav, *start)
    await plant.start()

    if planner is None:
        planner = thor_client.StaticPlanner(list(plan or PLAN), plan_id="e2e").on_uplink
    client = await thor_client.ThorLinkClient.serve(client_key, planner, "127.0.0.1", 0)
    settings = ThorLinkSettings(
        enabled=True,
        thor_host="127.0.0.1",
        thor_port=client.port,
        bind_host="127.0.0.1",
        bind_port=0,
        timeout_s=0.5,
        uplink_hz=40,
        allow_blade=allow_blade,
        geofence=fence(),
    )
    runtime = await start_thor_link(
        tractor, safety_monitor_running=True, gnss=plant.snapshot, nav=nav, settings=settings
    )
    assert runtime is not None
    stack = Stack(tractor, nav, plant, runtime, client)
    tractor.on_change = lambda kind, values, source: stack.events.append(
        (time.monotonic(), kind, values, source)
    )
    try:
        yield stack
    finally:
        await stop_thor_link(runtime)
        stack.client.close()
        await plant.stop()
        await tractor.shutdown()


async def drive_to_first_leg(s: Stack) -> None:
    await eventually(lambda: not s.link.safe_hold, what="the link to leave hold on a plan")
    await eventually(s.moving, what="levers to move toward the first waypoint")


# ------------------------------- accepted plan -------------------------------


async def test_accepted_plan_is_driven_and_progress_goes_back_to_the_thor(monkeypatch):
    async with pi_stack(monkeypatch) as s:
        await drive_to_first_leg(s)
        assert s.follower.state.value == "driving" and s.follower.index == 0
        # The uplinks the Thor hears carry the Pi's real state and progress.
        up = s.client.last_uplink
        assert up.fix == "rtk_fixed" and up.sid == s.link.session_id
        lat, lon = ll(5, 5)
        assert up.lat == pytest.approx(lat) and up.lon == pytest.approx(lon)
        # The plan asked for the blade on this leg; the Pi has not permitted it.
        assert PLAN[0].blade is True and s.tractor.state.blade_engaged is False
        assert s.tractor.state.left_lever > 0.2 and s.tractor.state.right_lever > 0.2

        s.plant.teleport(5, 35)  # waypoint 0 reached
        await eventually(lambda: s.follower.reached >= 1, what="waypoint 0 to be reached")
        await eventually(
            lambda: s.client.last_uplink.waypoint_index == 1 and s.link.plan.start_index == 1,
            what="the Thor to move its window on to waypoint 1",
        )
        await eventually(lambda: s.follower.index == 1 and s.moving(), what="leg 1 to start")


async def test_blade_follows_the_plan_only_where_the_pi_permits_it(monkeypatch):
    async with pi_stack(monkeypatch, allow_blade=True) as s:
        await drive_to_first_leg(s)
        await eventually(lambda: s.tractor.state.blade_engaged, what="blade on, as permitted")
        s.plant.teleport(5, 35)
        await eventually(lambda: s.follower.index == 1, what="leg 1 (a mowing leg too)")
        s.plant.teleport(35, 35)
        await eventually(lambda: s.follower.index == 2, what="the transit leg (blade off)")
        await eventually(lambda: not s.tractor.state.blade_engaged, what="blade off in transit")


async def test_real_strategist_plan_is_followed_over_the_wire(monkeypatch):
    """The Thor's actual planner, fed only by the Pi's real uplinks, over real UDP."""
    from backend.src.nav.thor_strategist import ThorStrategist

    strategist = ThorStrategist([ll(0, 0), ll(30, 0), ll(30, 30), ll(0, 30)], plan_id="yard")
    planner = thor_client.StrategistPlanner(strategist)
    async with pi_stack(monkeypatch, planner=planner.on_uplink, start=(2.0, 2.0)) as s:
        await drive_to_first_leg(s)
        assert s.link.plan.plan_id == "yard" and 1 <= len(s.link.plan.waypoints) <= MAX_WAYPOINTS
        first = s.link.plan.waypoints[0]
        s.plant.teleport(*_en(first.lat, first.lon))
        await eventually(lambda: s.follower.reached >= 1, what="the first strategist leg")
        await eventually(
            lambda: s.client.last_uplink.waypoint_index >= 1 and strategist.state.next_index >= 1,
            what="the strategist to record the Pi's progress",
        )


def _en(lat: float, lon: float) -> tuple[float, float]:
    return (
        (lon - LON0) * M_PER_DEG * math.cos(math.radians(LAT0)),
        (lat - LAT0) * M_PER_DEG,
    )


# --------------------------- link loss, hold, recovery ---------------------------


async def test_link_loss_holds_the_mower_and_a_returning_thor_recovers_it(monkeypatch):
    async with pi_stack(monkeypatch, allow_blade=True) as s:
        await drive_to_first_leg(s)
        await eventually(lambda: s.tractor.state.blade_engaged, what="mowing")
        s.plant.teleport(5, 35)
        await eventually(lambda: s.follower.index == 1, what="progress to waypoint 1")
        port = s.client.port

        s.client.close()  # the Thor dies
        await eventually(
            lambda: s.link.safe_hold and s.link.hold_reason is HoldReason.LINK_TIMEOUT,
            what="safe hold after the Thor went silent",
        )
        await eventually(s.neutral, what="blade off and levers neutral")
        await eventually(lambda: s.follower.state.value == "idle", what="the leg to end")
        # A command already in flight can land once more: bounded by one navigation tick and
        # one link tick (the link re-asserts neutral every tick). After that, nothing moves it.
        await asyncio.sleep(0.5)
        settled = time.monotonic()
        await holds_for(s.neutral, 1.0, "the held mower being neutral")
        assert s.drive_events_after(settled) == []
        assert s.tractor.state.authorized is True  # a hold is a soft stop, not a revoke

        # A new Thor process answers on the same address: the Pi recovers on its own.
        planner = thor_client.StaticPlanner(list(PLAN), plan_id="e2e").on_uplink
        s.client = await thor_client.ThorLinkClient.serve(KEY, planner, "127.0.0.1", port)
        await eventually(lambda: not s.link.safe_hold, what="hold to end on a fresh plan")
        await eventually(s.moving, what="driving again")
        assert s.follower.index == 1, "progress survived the outage"
        assert s.link.session_id == s.client.last_uplink.sid


async def test_a_thor_pause_holds_and_resuming_continues(monkeypatch):
    async with pi_stack(monkeypatch) as s:
        await drive_to_first_leg(s)
        s.client.paused = True
        await eventually(
            lambda: s.link.safe_hold and s.link.hold_reason is HoldReason.PAUSED, what="pause"
        )
        await eventually(s.neutral, what="neutral under the pause")
        s.client.paused = False
        await eventually(lambda: not s.link.safe_hold and s.moving(), what="resume")


async def test_a_cooperative_thor_pauses_on_a_degraded_fix_and_on_an_estop(monkeypatch):
    async with pi_stack(monkeypatch) as s:
        await drive_to_first_leg(s)
        s.plant.fix = "rtk_float"  # reported to the Thor in the uplink
        await eventually(
            lambda: s.link.safe_hold and s.link.hold_reason is HoldReason.PAUSED,
            what="the Thor to pause on a degraded fix",
        )
        await eventually(s.neutral, what="neutral")
        s.plant.fix = "rtk_fixed"
        await eventually(lambda: not s.link.safe_hold and s.moving(), what="recovery on a fix")
        await s.tractor.emergency_stop()
        await eventually(
            lambda: s.link.safe_hold and s.link.hold_reason is HoldReason.PAUSED,
            what="the Thor to pause on the reported e-stop",
        )
        assert s.neutral()


# ------------------------------ spoofed and bad senders ------------------------------


class Attacker:
    """A raw UDP socket on the same host: the shape of a stranger on the HaLow link."""

    def __init__(self, target: tuple[str, int]) -> None:
        self.target = target
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data: bytes) -> None:
        self.sock.sendto(data, self.target)

    def close(self) -> None:
        self.sock.close()


async def test_spoofed_senders_cannot_pause_steer_or_poison_the_link(monkeypatch):
    async with pi_stack(monkeypatch) as s:
        await drive_to_first_leg(s)
        captured: list[bytes] = []
        real_handle = s.client.handle

        def sniffing_handle(data, addr):
            reply = real_handle(data, addr)
            if reply is not None:
                captured.append(reply)
            return reply

        s.client.handle = sniffing_handle
        await eventually(lambda: len(captured) >= 3, what="some genuine traffic to sniff")
        attacker = Attacker(s.pi_addr)
        try:
            # sid and seq travel in the clear, so the attacker knows them; it has no key.
            sid, ack = s.link.session_id, s.client.last_uplink.seq + 1
            far = wp(500, 500)  # nowhere near the yard
            forged = [
                encode(Downlink(sid=sid, ack=ack, plan_id="x", pause=True), OTHER_KEY),
                encode(Downlink(sid=sid, ack=ack, plan_id="x", waypoints=[far]), OTHER_KEY),
                encode(Downlink(sid=sid, ack=ack + 9, plan_id="x", pause=True), OTHER_KEY),
                b'{"v":2,"sid":%d,"ack":%d,"plan_id":"x","pause":true}' % (sid, ack),
                b"\x00" * 64,
                b"x" * 5000,
            ]
            for datagram in forged * 5:
                attacker.send(datagram)
            # Genuine replies the Pi already accepted, replayed by someone who recorded them.
            for old in captured[:3]:
                attacker.send(old)
            await eventually(lambda: s.link.rejected >= 30, what="the forgeries to be counted")
            assert s.link.rejects["auth"] >= 20 and s.link.rejects["oversize"] >= 5
            assert s.link.rejects["replay"] + s.link.rejects["stale"] >= 3
            # None of it paused, held or steered anything.
            await holds_for(lambda: not s.link.safe_hold, 0.6, "the live link")
            assert s.moving() and s.follower.index == 0
            assert far not in (s.link.plan.waypoints if s.link.plan else [])
        finally:
            attacker.close()


class Impostor(asyncio.DatagramProtocol):
    """A fake Thor: reads the Pi's uplinks (they are authenticated, not secret) and forges
    perfectly addressed answers, but with a key that is not the Pi's."""

    def __init__(self, key: bytes, waypoints: list[Waypoint]) -> None:
        self.key, self.waypoints = key, waypoints
        self.sent = 0

    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            body = json.loads(data[32:])
            forged = Downlink(
                sid=body["sid"], ack=body["seq"], plan_id="impostor", waypoints=self.waypoints
            )
        except Exception:
            return
        self.transport.sendto(encode(forged, self.key), addr)
        self.sent += 1


async def test_a_fake_thor_that_reads_every_uplink_but_lacks_the_key_is_never_obeyed(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("THOR_LINK_KEY", KEY.hex())
    tractor = TractorControlService(config={"enabled": True})
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    tractor.authorize()
    tractor.state.engine = EngineState.RUNNING
    nav = NavigationService()
    plant = Plant(nav, 5.0, 5.0)
    await plant.start()
    loop = asyncio.get_running_loop()
    impostor = Impostor(OTHER_KEY, PLAN)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: impostor, local_addr=("127.0.0.1", 0)
    )
    runtime = await start_thor_link(
        tractor,
        safety_monitor_running=True,
        gnss=plant.snapshot,
        nav=nav,
        settings=ThorLinkSettings(
            enabled=True,
            thor_host="127.0.0.1",
            thor_port=transport.get_extra_info("sockname")[1],
            bind_port=0,
            timeout_s=0.5,
            uplink_hz=40,
            geofence=fence(),
        ),
    )
    commands: list = []
    tractor.on_change = lambda kind, values, source: commands.append((kind, values, source))
    try:
        await eventually(lambda: runtime.link.rejects["auth"] >= 20, what="forged answers counted")
        assert impostor.sent >= 20
        await asyncio.sleep(0.5)
        link, follower = runtime.link, runtime.follower
        assert link.safe_hold and link.hold_reason is HoldReason.NO_LINK
        assert link.plan is None and follower.plan_id is None and follower.index is None
        assert follower.state.value == "idle" and follower.reason == "link_hold"
        assert commands == [] or all(
            abs(v.get("left", 0)) <= NEUTRAL and abs(v.get("right", 0)) <= NEUTRAL
            for k, v, _ in commands
            if k == "levers"
        ), "nothing was ever driven"
        assert tractor.state.left_lever == 0 and not tractor.state.blade_engaged
    finally:
        await stop_thor_link(runtime)
        transport.close()
        await plant.stop()
        await tractor.shutdown()


async def test_a_datagram_from_another_host_is_shed_even_with_the_right_key(monkeypatch):
    """Source allowlist (default: the Thor's address alone), ahead of any decoding."""
    async with pi_stack(monkeypatch) as s:
        await drive_to_first_leg(s)
        stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            stranger.bind(("127.0.0.2", 0))
        except OSError:
            stranger.close()
            pytest.skip("127.0.0.2 is not available as a second loopback address")
        try:
            sid, ack = s.link.session_id, s.client.last_uplink.seq
            pause = encode(Downlink(sid=sid, ack=ack, plan_id="x", pause=True), KEY)
            for _ in range(20):
                stranger.sendto(pause, s.pi_addr)
            await eventually(lambda: s.link.rejects["source"] >= 20, what="the stranger shed")
            await holds_for(lambda: not s.link.safe_hold and s.moving(), 0.6, "a live, driving Pi")
        finally:
            stranger.close()


async def test_a_plan_outside_the_fence_is_refused_even_from_the_real_thor(monkeypatch):
    """A broken (or key-holding) Thor planner still cannot send the mower out of the yard."""
    async with pi_stack(monkeypatch, plan=[wp(5, 35), wp(300, 300)]) as s:
        await eventually(lambda: s.link.rejects["geofence"] >= 3, what="geofence rejections")
        assert s.link.safe_hold, "no plan was ever accepted, so the mower never left hold"
        assert s.neutral() and s.follower.state.value == "idle"
        assert s.drive_events_after(0.0) == []


async def test_a_plan_whose_leg_crosses_the_no_go_zone_is_refused_whole_by_the_link(monkeypatch):
    across = [wp(5, 50), wp(90, 50)]  # both ends legal; the line crosses the 40..60 square
    async with pi_stack(monkeypatch, plan=across, start=(5.0, 50.0)) as s:
        await eventually(lambda: s.link.rejects["geofence"] >= 3, what="the crossing refused")
        assert s.link.safe_hold, "no plan was accepted, so the mower never left hold"
        await holds_for(s.neutral, 0.5, "the mower standing still")
        assert s.follower.state.value == "idle" and s.drive_events_after(0.0) == []


async def test_a_leg_from_the_mower_across_the_no_go_zone_is_never_driven(monkeypatch):
    # The one waypoint is legal and a window of one has no leg of its own, so the link accepts
    # it; the leg from where the mower stands (5, 50) to it crosses the zone: the follower's.
    async with pi_stack(monkeypatch, plan=[wp(90, 50)], start=(5.0, 50.0)) as s:
        await eventually(lambda: not s.link.safe_hold, what="the link to accept the plan")
        await eventually(lambda: s.follower.reason == "leg_blocked", what="the crossing refused")
        await holds_for(s.neutral, 0.5, "the mower standing still")
        assert s.drive_events_after(0.0) == []


# ---------------------------- local interlocks win ----------------------------


async def test_estop_wins_over_a_plan_that_stays_valid(monkeypatch):
    async with pi_stack(monkeypatch, planner=blind_thor(PLAN)) as s:
        await drive_to_first_leg(s)
        await s.tractor.emergency_stop()
        stopped_at = time.monotonic()
        answered = s.client.counters["answered"]
        await eventually(lambda: s.follower.reason == "estop", what="the follower to stand down")
        await holds_for(s.neutral, 0.8, "neutral after the E-stop")
        assert s.client.counters["answered"] > answered, "the Thor kept sending a valid plan"
        assert s.link.current_plan() is not None and not s.link.safe_hold
        assert s.drive_events_after(stopped_at + 0.4) == []
        # Only a human restarts it: clear, then re-authorize.
        await s.tractor.clear_emergency()
        await asyncio.sleep(0.4)
        assert s.neutral(), "clearing the latch must not restart the mower by itself"
        s.tractor.authorize()
        await eventually(s.moving, what="driving after the operator re-authorized")


async def test_tilt_latch_wins_and_the_blade_stays_off_until_an_operator_clears_it(monkeypatch):
    async with pi_stack(monkeypatch, planner=blind_thor(PLAN), allow_blade=True) as s:
        tilt = {"deg": 0.0}

        async def imu():
            return (tilt["deg"], 0.0)

        class _Heartbeat:
            required_sources = ("actuation", "safety_monitor")

            def heartbeat(self, source: str = "default") -> None:
                pass

        monitor = TractorSafetyMonitor(
            s.tractor,
            _Heartbeat(),
            tilt_threshold_degrees=30.0,
            trigger_manager=SafetyTriggerManager(),
            imu_reader=imu,
            poll_interval_s=0.02,
        )
        await monitor.start()
        try:
            await drive_to_first_leg(s)
            await eventually(lambda: s.tractor.state.blade_engaged, what="mowing")
            tilt["deg"] = 35.0
            tilted_at = time.monotonic()
            await eventually(lambda: s.tractor.state.interlock_latched, what="the tilt latch")
            await eventually(s.neutral, what="blade off and the drive stopped")
            await holds_for(s.neutral, 0.8, "neutral while tilted and the plan is valid")
            assert s.follower.reason == "interlock_latched" and not s.link.safe_hold
            assert s.drive_events_after(tilted_at + 0.4) == []
            tilt["deg"] = 0.0  # level again: the latch deliberately stays
            await holds_for(s.neutral, 0.5, "neutral after the tilt cleared")
            await s.tractor.clear_emergency()  # the operator's authenticated clear
            await eventually(lambda: s.moving() and s.tractor.state.blade_engaged, what="restart")
        finally:
            await monitor.stop()


@pytest.mark.parametrize(
    ("cause", "reason"),
    [
        ("stale_gnss", "gnss_stale"),
        ("dead_nav_feed", "nav_feed_stale"),
        ("degraded_fix", "fix_rtk_float"),
        ("outside_fence", "outside_fence"),
        ("mission_running", "nav_mode_auto"),
    ],
)
async def test_local_conditions_stop_the_mower_and_it_resumes_when_they_clear(
    monkeypatch, cause, reason
):
    """A Thor that never pauses cannot drive through any of these."""
    async with pi_stack(monkeypatch, planner=blind_thor(PLAN)) as s:
        await drive_to_first_leg(s)
        plant, nav = s.plant, s.nav
        apply = {
            "stale_gnss": lambda: setattr(plant, "gnss_age_s", 30.0),
            "dead_nav_feed": lambda: setattr(plant, "nav_feed_alive", False),
            "degraded_fix": lambda: setattr(plant, "fix", "rtk_float"),
            "outside_fence": lambda: plant.teleport(200, 200),
            "mission_running": lambda: setattr(nav.navigation_state, "navigation_mode", "auto"),
        }[cause]
        clear = {
            "stale_gnss": lambda: setattr(plant, "gnss_age_s", 0.02),
            "dead_nav_feed": lambda: setattr(plant, "nav_feed_alive", True),
            "degraded_fix": lambda: setattr(plant, "fix", "rtk_fixed"),
            "outside_fence": lambda: plant.teleport(5, 5),
            "mission_running": lambda: setattr(nav.navigation_state, "navigation_mode", "idle"),
        }[cause]

        apply()
        await eventually(lambda: s.follower.reason == reason, what=f"stand-down ({reason})")
        await eventually(s.neutral, what="levers neutral")
        await asyncio.sleep(0.5)  # an in-flight command may land once more
        settled = time.monotonic()
        await holds_for(s.neutral, 0.6, f"neutral while {cause}")
        assert s.drive_events_after(settled) == []
        assert not s.link.safe_hold, "the plan itself stayed valid the whole time"
        clear()
        await eventually(s.moving, what=f"driving again once {cause} cleared")


async def test_a_limiter_at_the_tractors_lever_chokepoint_holds_back_a_plan_leg(monkeypatch):
    """Plans go through ``apply()`` and so through ``set_levers``: whatever limiter sits there
    (the obstacle stop of audit WP-P4-01) governs a Thor leg exactly as it does a mission leg.
    The follower has no way round it and does not need to know it is there."""
    async with pi_stack(monkeypatch, planner=blind_thor(PLAN)) as s:
        engaged = {"on": False}
        real_set_levers = s.tractor.set_levers

        async def limiter(left, right, source="operator", duration_ms=None):
            if engaged["on"]:
                left, right = min(left, 0.0), min(right, 0.0)  # forward motion refused
            return await real_set_levers(left, right, source=source, duration_ms=duration_ms)

        s.tractor.set_levers = limiter
        await drive_to_first_leg(s)
        engaged["on"] = True
        await eventually(s.neutral, what="the limiter to bring the levers to neutral")
        await asyncio.sleep(0.5)
        settled = time.monotonic()
        await holds_for(s.neutral, 0.6, "neutral while the limiter is on")
        assert s.drive_events_after(settled) == []
        assert s.follower.state.value == "driving", "the leg was never abandoned, only held back"
        engaged["on"] = False
        await eventually(s.moving, what="driving again once the limiter lets go")


async def test_nothing_drives_unless_an_operator_authorized_and_started_the_engine(monkeypatch):
    async with pi_stack(monkeypatch, authorize=False) as s:
        await eventually(lambda: not s.link.safe_hold, what="the link live")
        await eventually(lambda: s.follower.reason == "not_authorized", what="authorization gate")
        await holds_for(lambda: s.neutral() and not s.moving(), 0.6, "an unauthorized mower idle")
        s.tractor.state.engine = EngineState.OFF
        s.tractor.authorize()
        await eventually(lambda: s.follower.reason == "engine_not_running", what="engine gate")
        assert s.neutral()


# ----------------------------------- shutdown -----------------------------------


async def test_stopping_the_link_stops_the_follower_drops_the_blade_and_closes_the_socket(
    monkeypatch,
):
    async with pi_stack(monkeypatch, allow_blade=True) as s:
        await drive_to_first_leg(s)
        await eventually(lambda: s.tractor.state.blade_engaged, what="mowing")
        runtime, transport = s.runtime, s.link._transport
        await stop_thor_link(runtime)
        assert transport.is_closing() and runtime.follower._task.done()
        assert s.neutral(), "a stopped follower leaves the blade off and the levers neutral"
        await stop_thor_link(runtime)  # idempotent: shutdown must never raise
