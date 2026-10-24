"""W7 Pi<->Thor link: datagram budget, HMAC authentication, replay/staleness defences, safe hold.

The service runs here on a fake clock and a fake socket. ``Rig.answer`` plays
the Thor: it answers an uplink the way ``ThorStrategist`` + the Thor's sender do
(echo ``sid``/``seq`` as ``sid``/``ack``, sign with the shared key) and keeps no
state, which is exactly what makes a Thor restart a non-event.
"""

import asyncio
import json
import logging
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.src.core.secrets_manager import SecretsManager
from backend.src.models import thor_link as wire
from backend.src.models.geofence import Geofence, LatLng
from backend.src.models.thor_link import (
    KEY_MIN_BYTES,
    MAX_COUNTER,
    MAX_DATAGRAM_BYTES,
    MAX_DETECTIONS,
    MAX_PLAN_INDEX,
    MAX_SESSION_ID,
    MAX_WAYPOINTS,
    PROTOCOL_VERSION,
    SIM_LINK_KEY,
    TAG_BYTES,
    Detection,
    Downlink,
    LinkAuthError,
    LinkBudgetError,
    LinkKeyError,
    Uplink,
    Waypoint,
    decode_downlink,
    decode_uplink,
    encode,
    parse_key,
)
from backend.src.models.tractor_control import EngineState
from backend.src.nav.geofence_validator import build_shape
from backend.src.nav.thor_strategist import ThorStrategist
from backend.src.services.thor_link_service import (
    HoldReason,
    ThorLinkService,
    resolve_link_key,
)
from backend.src.services.tractor_service import TractorControlService

KEY = b"thorlink-test-key/0123456789abcdef!"
OTHER_KEY = b"someone-elses-key/0123456789abcdef!!"
THOR = ("10.0.0.2", 47101)
WP = Waypoint(lat=45.0, lon=-93.0, blade=True)
LOGGER = "backend.src.services.thor_link_service"


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeWire:
    """Stands in for the UDP socket: records every datagram the Pi sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple]] = []

    def sendto(self, data: bytes, addr: tuple) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:
        pass


class Rig:
    """A ThorLinkService on a fake clock and socket, plus the Thor's side of the wire."""

    def __init__(self, *, clock: FakeClock | None = None, key: bytes = KEY, **kw) -> None:
        kw.setdefault("timeout_s", 0.5)
        self.clock = clock or FakeClock()
        self.tractor = TractorControlService(config={})
        self.link = ThorLinkService(self.tractor, THOR, clock=self.clock, key=key, **kw)
        self.wire = FakeWire()
        self.link._transport = self.wire

    @property
    def uplinks(self) -> list[Uplink]:
        return [decode_uplink(data, KEY) for data, _ in self.wire.sent]

    async def tick(self) -> None:
        await self.link.tick()

    def answer(
        self,
        up: Uplink | None = None,
        *,
        waypoints: list[Waypoint] | None = None,
        pause: bool = False,
        plan_id: str = "p1",
        key: bytes = KEY,
        **override,
    ) -> bytes:
        """What a healthy Thor sends for ``up`` (default: the newest uplink)."""
        up = up or self.uplinks[-1]
        down = Downlink(
            sid=up.sid,
            ack=up.seq,
            plan_id=plan_id,
            waypoints=[WP] if waypoints is None else waypoints,
            pause=pause,
        ).model_copy(update=override)  # model_copy skips validation: crafts out-of-range values
        return encode(down, key)

    def deliver(self, data: bytes, addr: tuple = THOR) -> None:
        self.link._on_datagram(data, addr)

    async def connect(self) -> bytes:
        """Uplink #0, the Thor's answer to it, then a tick that leaves hold. Returns the answer."""
        await self.tick()
        answer = self.answer()
        self.deliver(answer)
        await self.tick()
        assert not self.link.safe_hold
        return answer

    async def run_to_neutral_failure(self) -> None:
        """Engine on, moving, blade on, then the Thor goes silent past the timeout."""
        self.tractor.state.engine = EngineState.RUNNING
        self.tractor.authorize()
        await self.connect()
        await self.tractor.set_levers(0.6, 0.6)
        await self.tractor.engage_blade(True)


# ------------------------------- the datagram -------------------------------


def test_worst_case_uplink_fits_one_datagram():
    full = Uplink(
        sid=MAX_SESSION_ID,
        seq=MAX_COUNTER,
        t_ms=MAX_COUNTER,
        lat=-89.12345678901234,  # full-precision doubles, not tidy 9-decimal test values
        lon=-179.12345678901234,
        fix="rtk_fixed",
        heading_deg=359.99999999999994,
        waypoint_index=MAX_PLAN_INDEX,
        detections=[
            Detection(cls="unknown_obstacle", bearing_deg=-179.99, range_m=199.99, conf=0.999)
            for _ in range(MAX_DETECTIONS)
        ],
    )
    data = encode(full, KEY)
    assert len(data) <= MAX_DATAGRAM_BYTES
    back = decode_uplink(data, KEY)
    assert back.seq == full.seq and back.sid == full.sid and len(back.detections) == MAX_DETECTIONS
    assert back.detections[0].cls == "unknown_obstacle"
    assert back.detections[0].bearing_deg == pytest.approx(-180.0, abs=0.05)


def test_worst_case_downlink_window_fits_one_datagram():
    window = Downlink(
        sid=MAX_SESSION_ID,
        ack=MAX_COUNTER,
        plan_id="x" * 32,
        start_index=MAX_PLAN_INDEX,
        waypoints=[
            Waypoint(lat=-89.12345678901234, lon=-179.12345678901234, blade=True)
            for _ in range(MAX_WAYPOINTS)
        ],
    )
    data = encode(window, KEY)
    assert len(data) <= MAX_DATAGRAM_BYTES
    # Quantised to 1e-7 deg (~1.1 cm) on the wire; that is what keeps the window inside the budget.
    back = decode_downlink(data, KEY)
    assert back.waypoints[0].lat == pytest.approx(-89.1234568, abs=1e-9)
    assert back.waypoints[0].lon == pytest.approx(-179.1234568, abs=1e-9)


def _yard():
    lat0, lon0 = 40.0, -75.0
    k = 111_320.0 * math.cos(math.radians(lat0))

    def ll(x: float, y: float) -> tuple[float, float]:
        return (lat0 + y / 111_320.0, lon0 + x / k)

    return ll, [ll(0, 0), ll(60, 0), ll(60, 40), ll(0, 40)]


def test_every_window_the_real_strategist_emits_fits_one_datagram():
    ll, yard = _yard()
    s = ThorStrategist(yard, [[ll(25, 15), ll(35, 15), ll(35, 25), ll(25, 25)]])
    lat, lon = ll(1, 1)
    for i in range(0, len(s.legs), 3):
        s.state.next_index = 0
        up = Uplink(sid=5, seq=i, t_ms=1, lat=lat, lon=lon, fix="rtk_fixed", waypoint_index=i)
        down = s.on_uplink(up)
        assert len(encode(down, KEY)) <= MAX_DATAGRAM_BYTES
        assert (down.sid, down.ack) == (5, i)


def test_oversized_datagram_rejected_for_both_directions():
    for decode in (decode_downlink, decode_uplink):
        with pytest.raises(LinkBudgetError):
            decode(b" " * (MAX_DATAGRAM_BYTES + 1), KEY)


def test_size_cap_holds_even_for_an_authentic_sender():
    """A correctly tagged body over the cap is refused before its tag is even checked."""
    body = json.dumps({"v": PROTOCOL_VERSION, "sid": 1, "ack": 0, "plan_id": "p" * 1200}).encode()
    data = wire._tag(KEY, wire._DOWNLINK_DOMAIN, body) + body
    assert len(data) > MAX_DATAGRAM_BYTES
    with pytest.raises(LinkBudgetError):
        decode_downlink(data, KEY)


@pytest.mark.parametrize("where", ["first", "last_tag", "first_body", "middle", "last"])
def test_any_flipped_byte_fails_authentication(where):
    data = bytearray(encode(Downlink(sid=1, ack=2, plan_id="p", waypoints=[WP]), KEY))
    index = {
        "first": 0,
        "last_tag": TAG_BYTES - 1,
        "first_body": TAG_BYTES,
        "middle": len(data) // 2,
        "last": len(data) - 1,
    }[where]
    data[index] ^= 0x01
    with pytest.raises(LinkAuthError):
        decode_downlink(bytes(data), KEY)


def test_tag_is_bound_to_the_key_and_the_direction():
    down = encode(Downlink(sid=1, ack=2, plan_id="p"), KEY)
    up = encode(Uplink(sid=1, seq=2, t_ms=3), KEY)
    with pytest.raises(LinkAuthError):
        decode_downlink(down, OTHER_KEY)
    # The Pi's own uplink reflected back at it, or a downlink fed to the Thor, must not verify.
    with pytest.raises(LinkAuthError):
        decode_downlink(up, KEY)
    with pytest.raises(LinkAuthError):
        decode_uplink(down, KEY)
    assert decode_downlink(down, KEY).ack == 2 and decode_uplink(up, KEY).seq == 2


@pytest.mark.parametrize("data", [b"", b"x" * TAG_BYTES, b'{"v":2,"sid":1,"ack":0,"plan_id":"p"}'])
def test_unsigned_or_truncated_datagram_is_not_authentic(data):
    with pytest.raises(LinkAuthError):
        decode_downlink(data, KEY)


def test_authentic_but_invalid_content_is_rejected_by_the_schema():
    """HMAC proves who sent it, not that it is sane: count and range limits still apply."""
    too_many = Downlink.model_construct(
        v=PROTOCOL_VERSION,
        sid=1,
        ack=0,
        plan_id="p",
        start_index=0,
        waypoints=[WP] * (MAX_WAYPOINTS + 1),
        pause=False,
    )
    nan_waypoint = Downlink.model_construct(
        v=PROTOCOL_VERSION,
        sid=1,
        ack=0,
        plan_id="p",
        start_index=0,
        waypoints=[Waypoint.model_construct(lat=math.nan, lon=0.0, blade=False)],
        pause=False,
    )
    for bad in (too_many, nan_waypoint):
        data = encode(bad, KEY)  # genuinely signed, and still within the byte budget
        assert len(data) <= MAX_DATAGRAM_BYTES
        with pytest.raises(ValidationError):
            decode_downlink(data, KEY)


def test_weak_or_malformed_keys_are_refused():
    msg = Downlink(sid=1, ack=2, plan_id="p")
    for bad in (b"", b"short", b"x" * (KEY_MIN_BYTES - 1), None, "a-string-not-bytes" * 4):
        with pytest.raises(LinkKeyError):
            encode(msg, bad)
        with pytest.raises(LinkKeyError):
            decode_downlink(b"y" * 64, bad)
    assert parse_key("ab" * KEY_MIN_BYTES) == bytes.fromhex("ab" * KEY_MIN_BYTES)
    with pytest.raises(LinkKeyError, match="hex") as bad_hex:
        parse_key("zz" * 40)
    assert "zz" not in str(bad_hex.value)  # never echo the secret
    with pytest.raises(LinkKeyError):
        parse_key("ab" * (KEY_MIN_BYTES - 1))


# ----------------------------- hold and heartbeat -----------------------------


async def test_starts_in_safe_hold_and_leaves_on_fresh_plan():
    rig = Rig()
    await rig.tick()
    assert rig.link.safe_hold and rig.link.current_plan() is None
    assert rig.link.hold_reason is HoldReason.NO_LINK
    rig.deliver(rig.answer())
    await rig.tick()
    assert not rig.link.safe_hold and rig.link.current_plan().plan_id == "p1"
    assert rig.link.hold_reason is None


async def test_uplink_carries_session_counter_and_hold_state():
    rig = Rig()
    times = []
    for _ in range(3):
        times.append(rig.clock.t)
        await rig.tick()
        rig.clock.t += 0.1
    ups = rig.uplinks
    assert [u.seq for u in ups] == [0, 1, 2]
    assert {u.sid for u in ups} == {rig.link.session_id} and rig.link.session_id > 0
    assert ups[0].safe_hold
    assert [u.t_ms for u in ups] == [int(t * 1000) for t in times]


async def test_uplink_that_failed_to_send_is_not_consumed():
    rig = Rig()

    def boom(data, addr):
        raise OSError("network unreachable")

    rig.wire.sendto = boom
    await rig.tick()  # never raises
    assert rig.link.uplink_errors == 1
    del rig.wire.sendto
    await rig.tick()
    assert [u.seq for u in rig.uplinks] == [
        0
    ]  # same seq reused; nothing was ever sent as #0 before


async def test_link_timeout_neutralises_levers_and_blade_without_revoking():
    rig = Rig()
    sources: list[tuple[str, str]] = []
    rig.tractor.on_change = lambda kind, values, source: sources.append((kind, source))
    await rig.run_to_neutral_failure()
    rig.clock.t += 0.6  # Thor silent past the timeout
    await rig.tick()
    link, state = rig.link, rig.tractor.state
    assert link.safe_hold and link.current_plan() is None
    assert link.hold_reason is HoldReason.LINK_TIMEOUT
    assert state.left_lever == 0.0 and state.right_lever == 0.0
    assert state.blade_engaged is False
    assert state.authorized is True  # soft stop, not e-stop
    assert state.emergency_stop_active is False
    # Capture logs must not call a link-driven hold "operator" input.
    assert ("blade", "thor_link") in sources and ("levers", "thor_link") in sources


async def test_pause_holds_and_reordered_answers_are_ignored():
    rig = Rig()
    await rig.connect()  # uplinks #0 and #1 sent, #0 answered
    rig.clock.t += 0.1
    await rig.tick()  # uplink #2
    older, newer = rig.answer(rig.uplinks[1]), rig.answer(rig.uplinks[2], pause=True)
    rig.deliver(newer)
    await rig.tick()
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.PAUSED
    rig.deliver(older)  # reordered: answers an uplink older than one already answered
    await rig.tick()
    assert rig.link.safe_hold and rig.link.rejects["replay"] == 1


async def test_held_mower_moved_elsewhere_is_reneutralised():
    rig = Rig()
    rig.tractor.state.engine = EngineState.RUNNING
    await rig.tick()
    await rig.tractor.set_levers(0.4, 0.4)
    await rig.tick()
    assert rig.tractor.state.left_lever == 0.0 and rig.tractor.state.right_lever == 0.0


# ------------------------- spoofing, replay, restart --------------------------


async def test_unsigned_downlink_is_ignored():
    rig = Rig()
    await rig.tick()
    up = rig.uplinks[0]
    raw = json.dumps(
        {"v": PROTOCOL_VERSION, "sid": up.sid, "ack": up.seq, "plan_id": "evil", "waypoints": []}
    ).encode()  # perfectly well-formed protocol JSON, just not signed
    rig.deliver(raw)
    await rig.tick()
    assert rig.link.rejects["auth"] == 1
    assert rig.link.plan is None and rig.link.safe_hold and not rig.link.link_fresh()
    assert rig.link.hold_reason is HoldReason.NO_LINK


async def test_spoofed_downlinks_are_ignored():
    rig = Rig()
    await rig.tick()
    good = rig.answer(waypoints=[Waypoint(lat=45.0, lon=-93.0, blade=False)])
    tampered = bytearray(good)
    tampered[TAG_BYTES + good[TAG_BYTES:].index(b"45.0")] = ord("1")  # move the waypoint
    forged = [
        rig.answer(key=OTHER_KEY),  # signed by someone without the key
        bytes(tampered),  # genuine tag, altered body
        rig.wire.sent[0][0],  # the Pi's own uplink reflected back at it
    ]
    for datagram in forged:
        rig.deliver(datagram)
    await rig.tick()
    assert rig.link.rejects["auth"] == 3 and rig.link.rejected == 3
    assert rig.link.plan is None and rig.link.safe_hold
    rig.deliver(good)  # the real one still works afterwards
    await rig.tick()
    assert not rig.link.safe_hold


async def test_forged_replayed_and_unsigned_packets_never_refresh_the_heartbeat():
    rig = Rig()
    accepted = await rig.connect()
    for _ in range(8):  # 0.8 s of an attacker hammering the port while the real Thor is gone
        rig.clock.t += 0.1
        rig.deliver(rig.answer(key=OTHER_KEY))
        rig.deliver(accepted)
        rig.deliver(b'{"v":2,"sid":1,"ack":0,"plan_id":"p"}')
        await rig.tick()
    assert not rig.link.link_fresh()
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.LINK_TIMEOUT
    assert rig.link.current_plan() is None
    assert rig.link.rejects["auth"] == 16 and rig.link.rejects["replay"] == 8


async def test_replayed_downlink_is_ignored():
    rig = Rig()
    first = await rig.connect()
    rig.clock.t += 0.1
    await rig.tick()
    rig.deliver(first)
    assert rig.link.rejects["replay"] == 1
    rig.deliver(rig.answer(plan_id="newer"))  # the genuine next answer is accepted ...
    rig.deliver(first)  # ... and the old one stays dead
    assert rig.link.rejects["replay"] == 2 and rig.link.plan.plan_id == "newer"


async def test_thor_restart_is_accepted_and_old_thor_traffic_is_not():
    rig = Rig()
    await rig.connect()
    old_answers = {}
    for _ in range(4):
        rig.clock.t += 0.1
        await rig.tick()
        old_answers[rig.uplinks[-1].seq] = rig.answer(plan_id="thor-run-1")
        rig.deliver(old_answers[rig.uplinks[-1].seq])
    assert rig.link.plan.plan_id == "thor-run-1"
    # The Thor reboots. A fresh process has no counters, no epoch and no handshake: it just
    # answers the next uplink it hears with a different plan id.
    rig.clock.t += 0.1
    await rig.tick()
    rig.deliver(rig.answer(plan_id="thor-run-2"))
    await rig.tick()
    assert rig.link.plan.plan_id == "thor-run-2" and not rig.link.safe_hold
    # Run 1's packets, captured earlier, are dead.
    for answer in old_answers.values():
        rig.deliver(answer)
    assert rig.link.rejects["replay"] == len(old_answers)
    assert rig.link.plan.plan_id == "thor-run-2"


async def test_traffic_captured_in_a_previous_pi_run_is_rejected():
    run1 = Rig()
    captured = await run1.connect()  # sid of run 1, ack 0
    run2 = Rig(clock=run1.clock)
    assert run2.link.session_id != run1.link.session_id
    await run2.tick()  # run 2's own uplink #0: ack 0 is otherwise perfectly plausible
    run2.deliver(captured)
    assert run2.link.rejects["session"] == 1 and run2.link.plan is None
    run2.deliver(run2.answer())
    await run2.tick()
    assert not run2.link.safe_hold


async def test_huge_counter_cannot_poison_the_link():
    rig = Rig()
    await rig.connect()  # uplinks #0, #1; #0 answered
    up1 = rig.uplinks[1]
    unsigned = json.dumps(
        {"v": 2, "sid": up1.sid, "ack": 2**62, "plan_id": "p", "waypoints": []}
    ).encode()
    rig.deliver(unsigned)  # unsigned: cannot be spoofed in the first place
    rig.deliver(rig.answer(up1, ack=MAX_COUNTER))  # authentic, in-schema, but from the future
    rig.deliver(rig.answer(up1, ack=2**62))  # authentic but outside the schema's bound
    assert dict(rig.link.rejects) == {"auth": 1, "future_ack": 1, "malformed": 1}
    # Had any of them moved the replay baseline, this legitimate answer would now be "stale".
    rig.deliver(rig.answer(up1, plan_id="after"))
    assert rig.link.rejected == 3 and rig.link.plan.plan_id == "after"
    rig.clock.t += 0.1
    await rig.tick()
    assert not rig.link.safe_hold and rig.link.link_fresh()


# ------------------------------ plan age limit --------------------------------


async def test_authentic_but_delayed_downlink_is_stale():
    rig = Rig()
    await rig.tick()
    rig.clock.t += 1.5  # older than plan_max_age_s (1.0)
    rig.deliver(rig.answer(rig.uplinks[0]))
    assert rig.link.rejects["stale"] == 1 and not rig.link.link_fresh()
    await rig.tick()
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.NO_LINK


async def test_plan_age_limit_expires_the_plan_while_the_heartbeat_is_fresh():
    rig = Rig()  # timeout_s 0.5, plan_max_age_s 1.0
    await rig.tick()  # uplink #0 at t=100.0
    rig.clock.t = 100.9  # the answer takes 0.9 s: slow, but inside the limit
    rig.deliver(rig.answer(rig.uplinks[0]))
    await rig.tick()
    assert not rig.link.safe_hold and rig.link.current_plan() is not None
    rig.clock.t = 101.15  # heartbeat only 0.25 s old, but the plan's information is 1.15 s old
    assert rig.link.link_fresh()
    assert rig.link.current_plan() is None  # the consumer is cut off with no tick needed
    await rig.tick()
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.PLAN_STALE


async def test_the_mower_resumes_when_the_thor_follows_a_pause_with_a_plan():
    rig = Rig()
    await rig.connect()
    rig.clock.t += 0.1
    await rig.tick()
    rig.deliver(rig.answer(pause=True))
    await rig.tick()
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.PAUSED
    rig.clock.t += 0.1
    await rig.tick()
    rig.deliver(rig.answer(plan_id="resumed"))
    await rig.tick()
    assert not rig.link.safe_hold and rig.link.current_plan().plan_id == "resumed"


async def test_steady_answers_keep_the_plan_alive_well_past_the_age_limits():
    rig = Rig()  # heartbeat timeout 0.5 s, plan age cap 1.0 s
    for _ in range(30):  # 3 s of a healthy 10 Hz exchange
        await rig.tick()
        rig.deliver(rig.answer())
        rig.clock.t += 0.1
    await rig.tick()
    assert not rig.link.safe_hold and rig.link.current_plan() is not None
    assert rig.link.rejected == 0


async def test_current_plan_expires_without_a_tick():
    rig = Rig()
    await rig.connect()
    assert rig.link.current_plan() is not None
    rig.clock.t += 0.6  # no tick() ran: safe_hold has not been updated yet
    assert rig.link.safe_hold is False
    assert rig.link.current_plan() is None


async def test_answers_to_uplinks_the_pi_no_longer_remembers_are_stale():
    rig = Rig(plan_max_age_s=2.0)  # memory of ~28 uplinks; age alone would still allow #0
    for _ in range(40):
        await rig.tick()
        rig.clock.t += 0.001
    rig.deliver(rig.answer(rig.uplinks[0]))
    assert rig.link.rejects["stale"] == 1
    rig.deliver(rig.answer(rig.uplinks[39]))
    assert rig.link.rejected == 1 and rig.link.link_fresh()


# ------------------------------ geofence --------------------------------------


def _fence():
    corners = [(45.0, -93.0), (45.0, -92.999), (45.001, -92.999), (45.001, -93.0)]
    fence = Geofence(
        geofence_id="yard", boundary=[LatLng(latitude=a, longitude=b) for a, b in corners]
    )
    return build_shape(fence)


INSIDE = Waypoint(lat=45.0005, lon=-92.9995, blade=True)
OUTSIDE = Waypoint(lat=45.01, lon=-92.9995, blade=True)


async def test_plan_with_a_waypoint_outside_the_geofence_is_rejected_whole():
    rig = Rig(geofence=_fence())
    await rig.tick()
    rig.deliver(rig.answer(waypoints=[INSIDE, INSIDE, OUTSIDE]))
    assert rig.link.rejects["geofence"] == 1 and rig.link.plan is None
    assert not rig.link.link_fresh()  # a rejected plan is not a heartbeat either
    await rig.tick()
    assert rig.link.safe_hold
    # Answer uplink #0 again: the geofence rejection must not have burned its replay slot.
    rig.deliver(rig.answer(rig.uplinks[0], waypoints=[INSIDE, INSIDE]))
    await rig.tick()
    assert not rig.link.safe_hold and len(rig.link.current_plan().waypoints) == 2


async def test_bad_plans_do_not_keep_a_running_mower_going():
    rig = Rig(geofence=_fence())
    await rig.tick()
    rig.deliver(rig.answer(waypoints=[INSIDE]))
    await rig.tick()
    assert not rig.link.safe_hold
    for _ in range(6):  # the Thor now keeps sending out-of-fence plans
        rig.clock.t += 0.1
        rig.deliver(rig.answer(waypoints=[OUTSIDE]))
        await rig.tick()
    assert rig.link.rejects["geofence"] == 6
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.LINK_TIMEOUT


async def test_a_failing_geofence_check_rejects_the_plan_but_never_a_pause(caplog):
    rig = Rig(geofence=SimpleNamespace())  # not a GeofenceShape: contains() raises on it
    await rig.tick()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        rig.deliver(rig.answer(waypoints=[INSIDE]))
    assert rig.link.rejects["geofence"] == 1 and rig.link.plan is None  # cannot prove: reject
    assert any("geofence check failed" in r.getMessage() for r in caplog.records)
    rig.deliver(rig.answer(rig.uplinks[0], pause=True, waypoints=[INSIDE]))
    assert rig.link.link_fresh() and rig.link.rejects["geofence"] == 1
    await rig.tick()
    assert rig.link.hold_reason is HoldReason.PAUSED


async def test_pause_is_never_refused_for_its_content():
    rig = Rig(geofence=_fence())
    await rig.tick()
    rig.deliver(rig.answer(waypoints=[INSIDE]))
    await rig.tick()
    assert not rig.link.safe_hold
    rig.clock.t += 0.1
    await rig.tick()
    rig.deliver(rig.answer(pause=True, waypoints=[OUTSIDE]))  # junk riding on a stop request
    await rig.tick()
    assert rig.link.rejects["geofence"] == 0
    assert rig.link.safe_hold and rig.link.hold_reason is HoldReason.PAUSED


async def test_real_strategist_plans_are_accepted_end_to_end():
    ll, yard = _yard()
    s = ThorStrategist(yard, [[ll(25, 15), ll(35, 15), ll(35, 25), ll(25, 25)]])
    fence = build_shape(
        Geofence(
            geofence_id="yard",
            boundary=[LatLng(latitude=a, longitude=b) for a, b in yard],
            buffer_distance_m=1.0,
        )
    )
    lat, lon = ll(1, 1)
    progress = {"i": 0}
    rig = Rig(
        geofence=fence,
        build_uplink=lambda: Uplink(
            seq=0, t_ms=0, lat=lat, lon=lon, fix="rtk_fixed", waypoint_index=progress["i"]
        ),
    )
    for step in range(8):
        await rig.tick()
        up = decode_uplink(rig.wire.sent[-1][0], KEY)  # what the Thor receives and verifies
        assert up.lat == lat and up.waypoint_index == progress["i"]
        rig.deliver(encode(s.on_uplink(up), KEY))
        progress["i"] = 3 * step
        rig.clock.t += 0.1
    await rig.tick()
    plan = rig.link.current_plan()
    assert plan is not None and rig.link.rejected == 0
    assert plan.start_index == s.state.next_index and len(plan.waypoints) == MAX_WAYPOINTS
    expected = s.waypoints()[plan.start_index]
    assert plan.waypoints[0].lat == pytest.approx(expected.lat, abs=1e-7)
    assert plan.waypoints[0].lon == pytest.approx(expected.lon, abs=1e-7)


# ------------------------- rejection accounting, logging ----------------------


async def test_oversized_and_malformed_datagrams_are_counted_and_do_not_move_state():
    rig = Rig()
    await rig.tick()
    rig.deliver(b"\x00" * (MAX_DATAGRAM_BYTES + 1))
    too_many = Downlink.model_construct(
        v=PROTOCOL_VERSION,
        sid=rig.link.session_id,
        ack=0,
        plan_id="p",
        start_index=0,
        waypoints=[WP] * (MAX_WAYPOINTS + 1),
        pause=False,
    )
    rig.deliver(encode(too_many, KEY))
    assert dict(rig.link.rejects) == {"oversize": 1, "malformed": 1}
    assert rig.link.plan is None and not rig.link.link_fresh()
    rig.deliver(rig.answer())  # state untouched: the real answer to #0 is still acceptable
    assert rig.link.link_fresh()


async def test_rejection_logging_is_rate_limited(caplog):
    rig = Rig()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(1000):
            rig.deliver(b"junk" * 20, ("203.0.113.9", 4000))
        assert rig.link.rejects["auth"] == 1000
        assert len([r for r in caplog.records if "(auth)" in r.getMessage()]) == 1
        rig.clock.t += 6.0
        rig.deliver(b"junk" * 20, ("203.0.113.9", 4000))
    lines = [r.getMessage() for r in caplog.records if "(auth)" in r.getMessage()]
    assert len(lines) == 2 and "+999 similar suppressed" in lines[1]


# ------------------------------ source allowlist ------------------------------


async def test_source_allowlist_sheds_other_sources_even_with_a_valid_signature():
    rig = Rig(allowed_sources=["10.0.0.0/24", "192.168.1.5"])
    await rig.tick()
    good = rig.answer()
    rig.deliver(good, ("203.0.113.9", 47101))
    rig.deliver(good, ("10.0.1.1", 47101))
    rig.deliver(good, ("garbage", 47101))
    rig.deliver(good, ())
    assert rig.link.rejects["source"] == 4 and not rig.link.link_fresh()
    # A source rejection happens before any decoding, so it burned no replay state.
    rig.deliver(good, ("::ffff:10.0.0.7", 47101, 0, 0))  # IPv4-mapped form of an allowed address
    assert rig.link.link_fresh() and rig.link.rejected == 4


async def test_source_allowlist_does_not_replace_the_signature():
    rig = Rig(allowed_sources=["10.0.0.2"])
    await rig.tick()
    rig.deliver(rig.answer(key=OTHER_KEY), ("10.0.0.2", 47101))
    assert rig.link.rejects["auth"] == 1 and not rig.link.link_fresh()


def test_bad_allowlists_and_timings_are_refused():
    tractor = TractorControlService(config={})
    with pytest.raises(ValueError, match="allowed_sources"):
        ThorLinkService(tractor, THOR, key=KEY, allowed_sources=[])
    with pytest.raises(ValueError, match="CIDR"):
        ThorLinkService(tractor, THOR, key=KEY, allowed_sources=["not-an-ip"])
    for kw in (
        {"timeout_s": 0},
        {"uplink_hz": 0},
        {"plan_max_age_s": -1},
    ):
        with pytest.raises(ValueError, match="must be > 0"):
            ThorLinkService(tractor, THOR, key=KEY, **kw)


# ------------------------------ key policy ------------------------------------


def _isolated_secrets(monkeypatch, tmp_path, *, sim: bool):
    monkeypatch.setenv("SIM_MODE", "1" if sim else "0")
    monkeypatch.delenv("THOR_LINK_KEY", raising=False)
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(tmp_path / "no-secrets.json"))
    return SecretsManager(store_path=str(tmp_path / "no-secrets.json"))


def test_service_refuses_to_start_without_a_key_outside_sim(monkeypatch, tmp_path):
    sm = _isolated_secrets(monkeypatch, tmp_path, sim=False)
    with pytest.raises(LinkKeyError, match="THOR_LINK_KEY"):
        resolve_link_key(sm)
    with pytest.raises(LinkKeyError, match="THOR_LINK_KEY"):
        ThorLinkService(TractorControlService(config={}), THOR)  # key=None: resolved, not assumed


async def test_sim_mode_without_a_key_falls_back_to_the_public_sim_key(
    monkeypatch, tmp_path, caplog
):
    _isolated_secrets(monkeypatch, tmp_path, sim=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        rig = Rig(key=None)
    assert any("SIM_LINK_KEY" in r.getMessage() for r in caplog.records)
    await rig.tick()
    assert decode_uplink(rig.wire.sent[0][0], SIM_LINK_KEY).sid == rig.link.session_id


def test_public_sim_key_is_refused_outside_sim_mode(monkeypatch, tmp_path):
    _isolated_secrets(monkeypatch, tmp_path, sim=False)
    with pytest.raises(LinkKeyError, match="SIM_MODE"):
        ThorLinkService(TractorControlService(config={}), THOR, key=SIM_LINK_KEY)
    _isolated_secrets(monkeypatch, tmp_path, sim=True)
    ThorLinkService(TractorControlService(config={}), THOR, key=SIM_LINK_KEY)  # explicit: fine


async def test_configured_key_comes_from_the_environment_as_hex(monkeypatch, tmp_path, caplog):
    _isolated_secrets(monkeypatch, tmp_path, sim=False)
    secret_hex = "9f" * 40
    monkeypatch.setenv("THOR_LINK_KEY", secret_hex)
    with caplog.at_level(logging.DEBUG):
        rig = Rig(key=None)
    await rig.tick()
    assert decode_uplink(rig.wire.sent[0][0], bytes.fromhex(secret_hex)).seq == 0
    with pytest.raises(LinkAuthError):  # and the public sim key does not open it
        decode_uplink(rig.wire.sent[0][0], SIM_LINK_KEY)
    assert secret_hex not in caplog.text  # secrets are never logged

    for bad in ("abcd", "not hex at all " * 4, "  "):
        monkeypatch.setenv("THOR_LINK_KEY", bad)
        with pytest.raises(LinkKeyError):
            ThorLinkService(TractorControlService(config={}), THOR)


def test_key_can_come_from_the_secrets_store(monkeypatch, tmp_path):
    sm = _isolated_secrets(monkeypatch, tmp_path, sim=False)
    sm.set("THOR_LINK_KEY", "5a" * 32)
    assert resolve_link_key(sm) == bytes.fromhex("5a" * 32)


# ------------------------------ real sockets ----------------------------------


async def test_udp_round_trip_on_loopback():
    received: list[Uplink] = []

    class Thor(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            up = decode_uplink(data, KEY)  # the Thor verifies the Pi, too
            received.append(up)
            down = Downlink(sid=up.sid, ack=up.seq, plan_id="p1", waypoints=[WP])
            self.transport.sendto(encode(down, KEY), addr)

    loop = asyncio.get_running_loop()
    thor_t, _ = await loop.create_datagram_endpoint(Thor, local_addr=("127.0.0.1", 0))
    port = thor_t.get_extra_info("sockname")[1]
    tractor = TractorControlService(config={})
    link = ThorLinkService(
        tractor,
        ("127.0.0.1", port),
        bind_addr=("127.0.0.1", 0),
        timeout_s=0.5,
        uplink_hz=50,
        key=KEY,
    )
    await link.start()
    spoofer = None
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            if not link.safe_hold:
                break
        assert received and not link.safe_hold
        assert received[0].v == PROTOCOL_VERSION and received[0].sid == link.session_id
        # A stranger on the same host fires an unsigned "pause" at the Pi's port.
        pi_port = link._transport.get_extra_info("sockname")[1]
        spoofer, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol, remote_addr=("127.0.0.1", pi_port)
        )
        spoofer.sendto(
            json.dumps(
                {"v": 2, "sid": link.session_id, "ack": 5, "plan_id": "x", "pause": True}
            ).encode()
        )
        await asyncio.sleep(0.1)
        assert link.rejects["auth"] == 1 and not link.safe_hold
        thor_t.close()  # Thor goes silent
        await asyncio.sleep(0.8)
        assert link.safe_hold and link.hold_reason is HoldReason.LINK_TIMEOUT
    finally:
        if spoofer is not None:
            spoofer.close()
        await link.stop()
        thor_t.close()
