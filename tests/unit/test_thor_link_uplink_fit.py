"""THOR-01: an uplink is the Pi's heartbeat, so it is thinned to fit one datagram, never refused.

The review's repro: 16 detections with long class names and full-precision GPS came to 1,240 B
(1,288 B with 24-character names), ``encode`` raised, no uplink was sent, the Thor could not
answer, the link timed out and the mower held. These tests pin the cure: coordinates are
quantised on the wire, class names are clipped, the least confident detections are dropped last,
the number dropped is reported, and whatever the builder hands over, a heartbeat goes out.
"""

import json
import logging
import random

import pytest

from backend.src.models import thor_link as wire
from backend.src.models.thor_link import (
    MAX_COUNTER,
    MAX_DATAGRAM_BYTES,
    MAX_DETECTIONS,
    MAX_PLAN_INDEX,
    MAX_SESSION_ID,
    MIN_CLASS_CHARS,
    TAG_BYTES,
    Detection,
    LinkBudgetError,
    Uplink,
    decode_uplink,
    encode,
    fit_uplink,
)
from tests.unit._thor_link_rig import KEY, LOGGER, Rig

# Class names of the shipped detector (workshop/classes.py): the longest is 16 characters.
REAL_LONGEST_CLASS = "unknown_obstacle"


def _worst(cls: str, *, confs: list[float] | None = None, **kw) -> Uplink:
    """Every field at its widest: max counters, full-precision doubles at the extremes."""
    confs = confs or [0.999] * MAX_DETECTIONS
    return Uplink(
        sid=MAX_SESSION_ID,
        seq=MAX_COUNTER,
        t_ms=MAX_COUNTER,
        lat=-89.12345678901234,
        lon=-179.12345678901235,
        fix="rtk_float",
        heading_deg=359.99999999999994,
        waypoint_index=MAX_PLAN_INDEX,
        detections=[Detection(cls=cls, bearing_deg=-179.99, range_m=199.99, conf=c) for c in confs],
        **kw,
    )


def _size(message: Uplink) -> int:
    return wire._wire_size(message)


def _with_names(message: Uplink, cls: str) -> Uplink:
    dets = [d.model_copy(update={"cls": cls}) for d in message.detections]
    return message.model_copy(update={"detections": dets})


# ------------------------------------- the model -------------------------------------


@pytest.mark.parametrize("length", range(1, 25))
def test_the_review_repro_fits_for_every_class_name_length_up_to_the_schema_limit(length):
    """16 detections, full-precision GPS, class names of 1..24 characters: always one datagram."""
    data = encode(fit_uplink(_worst("x" * length)), KEY)  # the old code raised from length 21 up
    assert len(data) <= MAX_DATAGRAM_BYTES
    back = decode_uplink(data, KEY)
    assert len(back.detections) == MAX_DETECTIONS and back.detections_dropped is None


def test_the_shipped_classes_at_their_worst_need_no_thinning_at_all():
    message = _worst(REAL_LONGEST_CLASS)
    assert _size(message) <= MAX_DATAGRAM_BYTES
    assert fit_uplink(message) is message, "an uplink that fits is passed through untouched"


def test_the_unthinned_review_repro_really_overflows():
    """Guards the guard: if the budget or the schema moves, the thinning tests must still bite."""
    assert _size(_worst("x" * 24)) > MAX_DATAGRAM_BYTES
    with pytest.raises(LinkBudgetError):
        encode(_worst("x" * 24), KEY)


def test_position_and_heading_are_quantised_on_the_wire_not_in_memory():
    message = _worst("person", confs=[])
    body = json.loads(encode(message, KEY)[TAG_BYTES:])
    assert body["lat"] == -89.1234568 and body["lon"] == -179.1234568
    assert body["heading_deg"] == 360.0
    assert message.lat == -89.12345678901234, "the message itself keeps its full precision"
    back = decode_uplink(encode(message, KEY), KEY)
    assert back.lat == pytest.approx(-89.12345678901234, abs=1e-7)
    assert back.lon == pytest.approx(-179.12345678901235, abs=1e-7)
    absent = json.loads(encode(Uplink(seq=0, t_ms=0), KEY)[TAG_BYTES:])
    assert "lat" not in absent and "lon" not in absent and "heading_deg" not in absent


def test_class_names_are_clipped_before_any_detection_is_dropped():
    message = _worst("a" * 24)
    fitted = fit_uplink(message)
    assert len(fitted.detections) == MAX_DETECTIONS and fitted.detections_dropped is None
    cap = len(fitted.detections[0].cls)
    assert MIN_CLASS_CHARS <= cap < 24
    assert all(d.cls == "a" * cap for d in fitted.detections)
    assert _size(fitted) <= MAX_DATAGRAM_BYTES
    # ... and by as little as possible: one character more would not have fitted
    assert _size(_with_names(message, "a" * (cap + 1))) > MAX_DATAGRAM_BYTES


def test_a_clipped_name_keeps_its_own_prefix_and_the_other_fields():
    names = ["sprinkler_head_pop_up", "garden_hose_coil_long", "person", "traffic_cone_orange"]
    dets = [
        Detection(cls=names[i % 4], bearing_deg=i * 10.0 - 80, range_m=1.5 + i, conf=0.5 + i / 40)
        for i in range(MAX_DETECTIONS)
    ]
    message = _worst("x").model_copy(update={"detections": dets, "lat": 45.0, "lon": -93.0})
    assert _size(message) > 1100  # these (realistic) names need clipping to fit a tighter budget
    fitted = fit_uplink(message, budget=1100)
    assert _size(fitted) <= 1100 and fitted.detections_dropped is None
    for before, after in zip(dets, fitted.detections, strict=True):
        assert before.cls.startswith(after.cls)
        assert len(after.cls) >= min(MIN_CLASS_CHARS, len(before.cls))
        assert (after.bearing_deg, after.range_m, after.conf) == (
            before.bearing_deg,
            before.range_m,
            before.conf,
        )
    assert fitted.detections[2].cls == "person", "a name that already fits is not touched"


def test_the_least_confident_detections_are_dropped_last_and_the_count_is_reported():
    """Names that cannot be clipped small enough (6 B per non-ASCII character on the wire)."""
    confs = [0.9, 0.1, 0.5, 0.95, 0.2, 0.7, 0.3, 0.99, 0.4, 0.6, 0.15, 0.85, 0.25, 0.8, 0.35, 0.05]
    fitted = fit_uplink(_worst("é" * 20, confs=confs))
    assert _size(fitted) <= MAX_DATAGRAM_BYTES
    dropped = fitted.detections_dropped
    assert dropped is not None and dropped >= 1
    assert len(fitted.detections) + dropped == MAX_DETECTIONS
    kept = [d.conf for d in fitted.detections]
    gone = sorted(set(confs) - set(kept))
    assert max(gone) <= min(kept), "everything dropped was less confident than everything kept"
    assert kept == [c for c in confs if c in set(kept)], "the survivors keep their order"
    assert decode_uplink(encode(fitted, KEY), KEY).detections_dropped == dropped


def test_among_equally_confident_detections_the_farthest_goes_first():
    ranges = (5, 150, 20, 190, 2, 90, 60, 120, 30, 10, 70, 180, 40, 100, 80, 160)
    dets = [Detection(cls="é" * 20, bearing_deg=0.0, range_m=float(r), conf=0.8) for r in ranges]
    fitted = fit_uplink(_worst("x").model_copy(update={"detections": dets}))
    kept = sorted(d.range_m for d in fitted.detections)
    gone = sorted({d.range_m for d in dets} - set(kept))
    assert fitted.detections_dropped == len(gone) >= 1
    assert min(gone) >= max(kept), "the nearest obstacles are the last to be dropped"


def test_a_tighter_budget_drops_detections_the_same_way():
    confs = [0.1 * (1 + i % 9) for i in range(MAX_DETECTIONS)]
    fitted = fit_uplink(_worst("person", confs=confs), budget=600)
    assert _size(fitted) <= 600 and fitted.detections_dropped
    assert len(fitted.detections) + fitted.detections_dropped == MAX_DETECTIONS
    gone = sorted(confs)
    for kept in (d.conf for d in fitted.detections):
        gone.remove(kept)
    assert max(gone) <= min(d.conf for d in fitted.detections)


def test_drops_the_builder_already_reported_are_added_to_not_replaced():
    fitted = fit_uplink(_worst("é" * 20, detections_dropped=3))
    assert fitted.detections_dropped == 3 + (MAX_DETECTIONS - len(fitted.detections))
    assert fitted.detections_dropped > 3


def test_an_uplink_too_big_for_a_reason_other_than_its_detections_is_an_error():
    huge = Uplink.model_construct(
        v=2, sid=1, seq=1, t_ms=1, fix="x" * 3000, detections=[], estop=False,
        safe_hold=False, hold_fault=False, blade=False,
    )  # fmt: skip
    with pytest.raises(LinkBudgetError, match="even with no detections"):
        fit_uplink(huge)
    with pytest.raises(LinkBudgetError, match="even with no detections"):
        fit_uplink(_worst("x"), budget=50)


def test_no_uplink_the_schema_allows_ever_fails_to_fit():
    """Seeded random uplinks over the whole schema (non-ASCII names, escapes, extremes)."""
    rng = random.Random(20261024)
    alphabet = 'abcXYZ_-.é€漢字 "\\/'  # escapes and multi-byte characters cost more on the wire
    for _ in range(400):
        count = rng.randint(0, MAX_DETECTIONS)
        dets = [
            Detection(
                cls="".join(rng.choice(alphabet) for _ in range(rng.randint(1, 24))),
                bearing_deg=rng.uniform(-180, 180),
                range_m=rng.uniform(0, 200),
                conf=rng.random(),
            )
            for _ in range(count)
        ]
        message = Uplink(
            sid=rng.randint(0, MAX_SESSION_ID),
            seq=rng.randint(0, MAX_COUNTER),
            t_ms=rng.randint(0, MAX_COUNTER),
            lat=rng.uniform(-90, 90) if rng.random() < 0.9 else None,
            lon=rng.uniform(-180, 180) if rng.random() < 0.9 else None,
            fix=rng.choice(["none", "gps", "dgps", "rtk_float", "rtk_fixed"]),
            heading_deg=rng.uniform(0, 360) if rng.random() < 0.9 else None,
            waypoint_index=rng.choice([None, rng.randint(0, MAX_PLAN_INDEX)]),
            detections=dets,
        )
        fitted = fit_uplink(message)
        data = encode(fitted, KEY)
        assert len(data) <= MAX_DATAGRAM_BYTES
        assert len(fitted.detections) + (fitted.detections_dropped or 0) == count
        assert decode_uplink(data, KEY).seq == message.seq


# ------------------------------- the service's heartbeat -------------------------------


async def test_a_scene_full_of_long_class_names_does_not_stop_the_heartbeat():
    """The review's scenario, end to end: the Pi keeps sending, the Thor answers, no hold."""
    rig = Rig(build_uplink=lambda: _worst("sprinkler_head_pop_up"))
    await rig.tick()
    assert rig.link.uplink_errors == 0, "the uplink was refused for its size (the old behaviour)"
    assert len(rig.wire.sent) == 1 and len(rig.wire.sent[0][0]) <= MAX_DATAGRAM_BYTES
    first = rig.uplinks[0]
    assert first.seq == 0 and first.sid == rig.link.session_id
    assert len(first.detections) == MAX_DETECTIONS and first.lat == pytest.approx(-89.1234568)
    assert rig.link.uplinks_degraded == 1 and rig.link.detections_dropped == 0
    # the Thor hears it and answers it: the link comes up instead of timing out
    rig.deliver(rig.answer())
    await rig.tick()
    assert not rig.link.safe_hold and rig.link.current_plan() is not None
    assert [u.seq for u in rig.uplinks] == [0, 1]


async def test_dropped_detections_are_reported_to_the_thor_and_counted_by_the_pi(caplog):
    confs = [0.2 + 0.05 * i for i in range(16)]
    rig = Rig(build_uplink=lambda: _worst("é" * 24, confs=confs))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await rig.tick()
    sent = rig.uplinks[0]
    assert sent.detections_dropped and rig.link.detections_dropped == sent.detections_dropped
    assert len(sent.detections) + sent.detections_dropped == MAX_DETECTIONS
    assert rig.link.uplinks_degraded == 1 and rig.link.uplink_errors == 0
    assert any("thinned" in r.getMessage() for r in caplog.records)
    await rig.tick()
    assert rig.link.uplinks_degraded == 2
    assert rig.link.detections_dropped == 2 * sent.detections_dropped


async def test_an_uplink_that_fits_is_not_counted_as_degraded():
    rig = Rig(build_uplink=lambda: _worst(REAL_LONGEST_CLASS))
    for _ in range(3):
        await rig.tick()
    assert rig.link.uplinks_degraded == 0 and rig.link.detections_dropped == 0
    assert all(u.detections_dropped is None for u in rig.uplinks)


async def test_degraded_uplinks_are_logged_at_a_bounded_rate(caplog):
    rig = Rig(build_uplink=lambda: _worst("x" * 24))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(200):
            await rig.tick()
            rig.clock.t += 0.02
    assert rig.link.uplinks_degraded == 200 and rig.link.uplink_errors == 0
    assert len([r for r in caplog.records if "thinned" in r.getMessage()]) <= 2


async def test_a_failing_state_builder_still_sends_a_heartbeat(caplog):
    def broken():
        raise RuntimeError("camera pipeline fell over")

    rig = Rig(build_uplink=broken)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        await rig.tick()
        await rig.tick()
    assert rig.link.uplink_errors == 0 and rig.link.builder_errors == 2
    assert [u.seq for u in rig.uplinks] == [0, 1], "the heartbeat kept its counter"
    bare = rig.uplinks[0]
    assert bare.sid == rig.link.session_id and bare.safe_hold and bare.lat is None
    assert bare.fix == "none", "no position: the Thor pauses rather than plans from nothing"
    assert len([r for r in caplog.records if "bare heartbeat" in r.getMessage()]) == 1
    assert any(r.exc_info for r in caplog.records), "the cause is logged with its traceback"


async def test_a_builder_payload_that_cannot_be_made_to_fit_becomes_a_bare_heartbeat():
    def oversized():
        return Uplink.model_construct(
            v=2, sid=0, seq=0, t_ms=0, fix="x" * 3000, detections=[], estop=False,
            safe_hold=False, hold_fault=False, blade=False,
        )  # fmt: skip

    rig = Rig(build_uplink=oversized)
    await rig.tick()
    assert rig.link.builder_errors == 1 and rig.link.uplink_errors == 0
    assert len(rig.wire.sent) == 1 and rig.uplinks[0].fix == "none"


async def test_the_service_flags_win_over_whatever_the_builder_claims():
    """sid, seq, clock and the safety flags are the service's own, builder or not."""
    claims = Uplink(seq=99, sid=7, t_ms=5, estop=True, safe_hold=False, hold_fault=True, blade=True)
    rig = Rig(build_uplink=lambda: claims)
    await rig.tick()
    up = rig.uplinks[0]
    assert (up.seq, up.sid) == (0, rig.link.session_id)
    assert up.estop is False and up.hold_fault is False and up.blade is False
    assert up.safe_hold is True and up.t_ms == int(rig.clock.t * 1000)
