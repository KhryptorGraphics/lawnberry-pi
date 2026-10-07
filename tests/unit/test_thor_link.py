"""W7 Pi<->Thor link: budget, decoding and the safe-hold transition."""

import asyncio

import pytest

from backend.src.models.thor_link import (
    MAX_DATAGRAM_BYTES,
    MAX_DETECTIONS,
    MAX_WAYPOINTS,
    Detection,
    Downlink,
    LinkBudgetError,
    Uplink,
    Waypoint,
    decode_downlink,
    decode_uplink,
    encode,
)
from backend.src.models.tractor_control import EngineState
from backend.src.services.thor_link_service import ThorLinkService
from backend.src.services.tractor_service import TractorControlService


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _plan(seq: int, pause: bool = False) -> Downlink:
    return Downlink(
        seq=seq, plan_id="p1", waypoints=[Waypoint(lat=45.0, lon=-93.0, blade=True)], pause=pause
    )


def test_worst_case_uplink_fits_one_datagram():
    full = Uplink(
        seq=2**31,
        t_ms=2**40,
        lat=-89.123456789,
        lon=-179.123456789,
        fix="rtk_fixed",
        heading_deg=359.999,
        waypoint_index=9999,
        detections=[
            Detection(cls="unknown_obstacle", bearing_deg=-179.99, range_m=199.99, conf=0.999)
            for _ in range(MAX_DETECTIONS)
        ],
    )
    data = encode(full)
    assert len(data) <= MAX_DATAGRAM_BYTES
    back = decode_uplink(data)
    assert back.seq == full.seq and len(back.detections) == MAX_DETECTIONS
    assert back.detections[0].cls == "unknown_obstacle"
    assert back.detections[0].bearing_deg == pytest.approx(-180.0, abs=0.05)


def test_worst_case_downlink_window_fits_one_datagram():
    window = Downlink(
        seq=2**31,
        plan_id="x" * 32,
        start_index=99999,
        waypoints=[
            Waypoint(lat=-89.123456789, lon=-179.123456789, blade=True)
            for _ in range(MAX_WAYPOINTS)
        ],
    )
    assert len(encode(window)) <= MAX_DATAGRAM_BYTES


def test_oversized_datagram_rejected():
    with pytest.raises(LinkBudgetError):
        decode_downlink(b" " * (MAX_DATAGRAM_BYTES + 1))


def _svc(clock: FakeClock) -> tuple[ThorLinkService, TractorControlService]:
    tractor = TractorControlService(config={})
    link = ThorLinkService(tractor, ("127.0.0.1", 9), timeout_s=0.5, clock=clock)
    return link, tractor


async def test_starts_in_safe_hold_and_leaves_on_fresh_plan():
    clock = FakeClock()
    link, _ = _svc(clock)
    await link.tick()
    assert link.safe_hold and link.current_plan() is None
    link._on_datagram(encode(_plan(1)), ("thor", 1))
    await link.tick()
    assert not link.safe_hold and link.current_plan().plan_id == "p1"


async def test_link_timeout_neutralises_levers_and_blade_without_revoking():
    clock = FakeClock()
    link, tractor = _svc(clock)
    tractor.state.engine = EngineState.RUNNING
    tractor.authorize()
    link._on_datagram(encode(_plan(1)), ("thor", 1))
    await link.tick()
    await tractor.set_levers(0.6, 0.6)
    await tractor.engage_blade(True)
    clock.t += 0.6  # Thor silent past the timeout
    await link.tick()
    assert link.safe_hold and link.current_plan() is None
    assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0
    assert tractor.state.blade_engaged is False
    assert tractor.state.authorized is True  # soft stop, not e-stop
    assert tractor.state.emergency_stop_active is False


async def test_pause_holds_and_stale_or_duplicate_seq_ignored():
    clock = FakeClock()
    link, _ = _svc(clock)
    link._on_datagram(encode(_plan(5)), ("thor", 1))
    await link.tick()
    assert not link.safe_hold
    link._on_datagram(encode(_plan(6, pause=True)), ("thor", 1))
    await link.tick()
    assert link.safe_hold
    link._on_datagram(encode(_plan(4)), ("thor", 1))  # reordered old packet
    await link.tick()
    assert link.safe_hold


async def test_held_mower_moved_elsewhere_is_reneutralised():
    clock = FakeClock()
    link, tractor = _svc(clock)
    tractor.state.engine = EngineState.RUNNING
    await link.tick()
    await tractor.set_levers(0.4, 0.4)
    await link.tick()
    assert tractor.state.left_lever == 0.0 and tractor.state.right_lever == 0.0


async def test_udp_round_trip_on_loopback():
    received: list[bytes] = []

    class Thor(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            received.append(data)
            self.transport.sendto(encode(_plan(len(received))), addr)

    loop = asyncio.get_running_loop()
    thor_t, _ = await loop.create_datagram_endpoint(Thor, local_addr=("127.0.0.1", 0))
    port = thor_t.get_extra_info("sockname")[1]
    tractor = TractorControlService(config={})
    link = ThorLinkService(
        tractor, ("127.0.0.1", port), bind_addr=("127.0.0.1", 0), timeout_s=0.5, uplink_hz=50
    )
    await link.start()
    try:
        for _ in range(100):
            await asyncio.sleep(0.02)
            if not link.safe_hold:
                break
        assert received and not link.safe_hold
        assert decode_uplink(received[0]).v == 1
        thor_t.close()  # Thor goes silent
        await asyncio.sleep(0.8)
        assert link.safe_hold
    finally:
        await link.stop()
        thor_t.close()
