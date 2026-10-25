"""SEC20-09: a flood that wears the Thor's source address must not starve the Thor.

The Pi's datagram budget was a token bucket per *source address*, spent before the HMAC. A UDP
source address is whatever its sender writes, so 400 junk datagrams a second carrying the Thor's
address left the real Thor 1 answer in 10 (the review's repro, fake clock): the link ended in
``link_timeout`` and safe hold. Now free filters (size, copies of verified datagrams, the cleartext
``sid`` and ``ack``) refuse what cannot be an answer, a ceiling bounds the HMAC checks of the whole
port, and the authentic rate of the peer is a bucket junk never touches.

The scenario tests use only what the old code had (``Rig``, ``rejects``, ``decode_downlink``), so
they can run against it and fail for behavioural reasons.
"""

import random

import pytest

from backend.src.models.thor_link import Downlink, decode_uplink, encode
from backend.src.services import thor_link_service as svc
from tests.unit._thor_link_rig import KEY, OTHER_KEY, THOR, Rig

STRANGER = ("10.0.0.99", 5000)
RNG = random.Random(20)
BLIND = [RNG.randbytes(100) for _ in range(32)]  # junk that knows nothing: no sid, no tag


def count_hmac_checks(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = svc.decode_downlink

    def counting(data, key):
        calls.append(1)
        return real(data, key)

    monkeypatch.setattr(svc, "decode_downlink", counting)
    return calls


def blind(rig, up, captured):
    return BLIND


def sniffing(rig, up, captured):
    """Forgeries by someone who reads the uplinks: this run's sid, the ack the Pi waits for."""
    return [encode(Downlink(sid=up.sid, ack=up.seq, plan_id=f"j{i}"), OTHER_KEY) for i in range(16)]


def replaying(rig, up, captured):
    return captured[-200:] or BLIND


async def flood(rig, *, seconds, hz, pool, source=THOR, rtt=0.010, uplink_hz=10):
    """The Thor answers each uplink ``rtt`` after it goes out; junk arrives at ``hz``/s from
    ``source`` all the while (1 ms steps, fake clock). Returns (accepted, answers, junk sent)."""
    accepted = answers = junk_sent = 0
    owed = 0.0
    captured: list[bytes] = []
    steps = round(1000 / uplink_hz)
    for _ in range(round(seconds * uplink_hz)):
        await rig.tick()
        up = decode_uplink(rig.wire.sent[-1][0], KEY)
        answer, junk = rig.answer(up), pool(rig, up, captured)
        for step in range(steps):
            rig.clock.t += 0.001
            owed += hz / 1000.0
            while owed >= 1.0:
                owed -= 1.0
                junk_sent += 1
                rig.deliver(RNG.choice(junk), source)
            if step == round(rtt / 0.001) - 1:
                rig.deliver(answer, THOR)
                answers += 1
                accepted += rig.link._last_ack == up.seq
                captured.append(answer)
    return accepted, answers, junk_sent


async def connected_rig(**kw) -> Rig:
    rig = Rig(allowed_sources=[THOR[0]], **kw)  # the default wiring: the Thor's address only
    await rig.connect()
    rig.link.rejects.clear()
    return rig


async def test_the_review_repro_400_junk_a_second_with_the_thors_address_leaves_every_answer(
    monkeypatch,
):
    checks = count_hmac_checks(monkeypatch)
    rig = await connected_rig()
    before = len(checks)
    accepted, answers, junk = await flood(rig, seconds=1, hz=400, pool=blind)
    assert (accepted, answers, junk) == (10, 10, 400), "the old code accepted 1 of 10"
    assert rig.link.rejects["rate"] == 0 and rig.link.link_fresh() and not rig.link.safe_hold
    assert len(checks) - before == 10, "not one HMAC went to the junk"


@pytest.mark.parametrize(
    ("kind", "pool", "hz"),
    [
        ("blind", blind, 4000),
        ("sniffing", sniffing, 400),
        ("sniffing", sniffing, 4000),
        ("replaying", replaying, 4000),
    ],
)
async def test_floods_below_the_ceiling_shed_not_one_authentic_answer(monkeypatch, kind, pool, hz):
    checks = count_hmac_checks(monkeypatch)
    rig = await connected_rig()
    before = len(checks)
    accepted, answers, junk = await flood(rig, seconds=5, hz=hz, pool=pool)
    assert accepted == answers == 50, f"{kind} at {hz}/s: {accepted} of {answers} accepted"
    assert rig.link.rejects["rate"] == 0 and rig.link.link_fresh() and not rig.link.safe_hold
    spent = len(checks) - before
    if kind == "sniffing":  # only what arrives before the answer, within one round trip, is checked
        assert spent <= answers * (hz * 0.010 + 2) < junk / 4
    else:
        assert spent == answers, "blind junk and copies cost no HMAC at all"


async def test_a_flood_from_an_address_off_the_allowlist_never_reaches_the_budget(monkeypatch):
    checks = count_hmac_checks(monkeypatch)
    rig = await connected_rig()
    before = len(checks)
    for pool in (blind, sniffing):
        accepted, answers, junk = await flood(rig, seconds=2, hz=4000, pool=pool, source=STRANGER)
        assert accepted == answers == 20
    assert rig.link.rejects["source"] == 2 * 8000 and len(checks) - before == 40
    assert rig.link.verifications == 41, "the Thor's 40 answers and the one of connect()"


async def test_beyond_the_ceiling_the_work_stays_bounded_and_the_loss_is_the_documented_one():
    """40,000 forgeries/s that wear the Thor's address, this run's sid and a live ack: nothing
    application-level can tell them from the Thor's answers once the ceiling is spent, so some
    answers are lost (volumetric spoofing needs ingress filtering). What the Pi guarantees: the
    HMAC work stays at the ceiling, no forgery is ever acted on, and the link fails safe."""
    rig = await connected_rig()
    before = rig.link.verifications
    accepted, answers, junk = await flood(rig, seconds=2, hz=40_000, pool=sniffing)
    spent = rig.link.verifications - before
    assert spent <= svc.VERIFY_BURST + svc.VERIFY_CEILING_HZ * 2 + 1 < junk / 30
    assert accepted < answers and rig.link.rejects["rate"] > 0, "the limit is real: say so"
    assert rig.link.plan is None or rig.link.plan.plan_id == "p1", "no forgery was acted on"
    assert rig.link.safe_hold and not rig.link.link_fresh(), "and the mower holds"


async def test_the_thors_first_answer_gets_through_a_sniffing_burst_before_it_is_trusted():
    rig = Rig(allowed_sources=[THOR[0]])  # nothing accepted yet: the Thor is only configured
    await rig.tick()
    for forged in sniffing(rig, decode_uplink(rig.wire.sent[-1][0], KEY), []) * 7:
        rig.deliver(forged, THOR)  # 112 forgeries in a burst, the old per-source allowance was 18
    rig.deliver(rig.answer(), THOR)
    assert rig.link._last_ack == 0, "the old code shed it: the link never came up"


@pytest.mark.parametrize("uplink_hz", [10, 50])
async def test_authentic_answers_at_the_uplink_rate_are_never_shed_amid_every_kind_of_junk(
    uplink_hz,
):
    rig = await connected_rig(uplink_hz=uplink_hz)
    total = 0
    for pool, source in ((blind, THOR), (sniffing, THOR), (replaying, THOR), (sniffing, STRANGER)):
        accepted, answers, _ = await flood(
            rig, seconds=15, hz=300, pool=pool, source=source, uplink_hz=uplink_hz
        )
        assert accepted == answers
        total += answers
    assert total == 60 * uplink_hz
    assert rig.link.rejects["rate"] == 0 and rig.link.rejects["peer_rate"] == 0


async def test_every_table_stays_bounded_whatever_the_flood_rotates():
    rig = await connected_rig()
    up = decode_uplink(rig.wire.sent[-1][0], KEY)
    forged = sniffing(rig, up, [])
    for i in range(60_000):
        rig.deliver(forged[i % 16], (f"198.51.{(i >> 8) & 255}.{i & 255}", 1000 + i % 5000))
        rig.deliver(RNG.choice(BLIND), (f"203.0.{(i >> 8) & 255}.{i & 255}", 1))
        if i % 500 == 0:
            rig.clock.t += 0.5
            await rig.tick()
            forged = sniffing(rig, decode_uplink(rig.wire.sent[-1][0], KEY), [])
            rig.deliver(rig.answer(), THOR)
    link = rig.link
    assert len(link._budget) <= 64 and len(link._budget.peers) <= 8
    assert len(link._verified) <= svc.VERIFIED_CACHE_SIZE and len(link._log_state) <= 16


async def test_a_copy_of_a_verified_datagram_costs_a_lookup_not_an_hmac(monkeypatch):
    """Authentic but unusable (so never accepted, and its ack stays live): only the cache of
    verified datagrams stands between a replay of it and an HMAC each time."""
    checks = count_hmac_checks(monkeypatch)
    rig = Rig()
    await rig.tick()
    unusable = rig.answer(start_index=-1)  # the right key, a schema violation
    for _ in range(50):
        rig.deliver(unusable)
    assert len(checks) == 1 and dict(rig.link.rejects) == {"malformed": 1, "replay": 49}


async def test_an_address_that_delivered_an_accepted_answer_becomes_a_peer(monkeypatch):
    """No allowlist, and the Thor answers from an address the Pi was not configured with: its
    first accepted answer makes that address a peer, so forgeries wearing it cannot cost it its
    share."""
    rig = Rig()
    thor_elsewhere = ("10.0.0.77", 47101)
    await rig.tick()
    rig.deliver(rig.answer(), thor_elsewhere)
    assert rig.link._last_ack == 0
    await rig.tick()
    for forged in sniffing(rig, decode_uplink(rig.wire.sent[-1][0], KEY), []) * 7:
        rig.deliver(forged, thor_elsewhere)  # 112 forgeries: a stranger's allowance is 18
    rig.deliver(rig.answer(), thor_elsewhere)
    assert rig.link._last_ack == 1 and rig.link.rejects["rate"] == 0


async def test_each_kind_of_datagram_is_counted_as_what_it_claims_to_be():
    rig = Rig()
    await rig.tick()  # uplink #0
    up = decode_uplink(rig.wire.sent[-1][0], KEY)
    forged = lambda ack: encode(Downlink(sid=up.sid, ack=ack, plan_id="f"), OTHER_KEY)  # noqa: E731
    for data in (
        RNG.randbytes(100),  # blind junk: no sid
        b"x" * 10,  # no room for a body behind a tag
        b"x" * 2000,  # oversize
        forged(0),  # this run's sid, an ack the Pi waits for, the wrong key: the only HMAC spent
        forged(5),  # an ack from the future
        rig.wire.sent[0][0],  # the Pi's own uplink: this run's sid, but no ack
    ):
        rig.deliver(data)
    answer = rig.answer()
    rig.deliver(answer)  # accepted
    rig.deliver(answer)  # a copy of a verified datagram
    rig.deliver(forged(0))  # claims the ack just answered
    assert dict(rig.link.rejects) == {
        "session": 1,
        "auth": 2,  # the short datagram (never verified) and the forgery (verified, failed)
        "oversize": 1,
        "future_ack": 1,
        "malformed": 1,
        "replay": 2,
    }
    assert rig.link.verifications == 2, "the forgery and the Thor's answer, and nothing else"
    assert rig.link.link_fresh()


# ---------- the cleartext prefilters must accept everything the encoder can produce ----------


def test_the_prefilters_accept_every_datagram_the_encoder_makes():
    """If the encoder and the prefilters drifted apart, the Pi would refuse every answer."""
    from backend.src.models.thor_link import (
        MAX_COUNTER,
        MAX_SESSION_ID,
        Waypoint,
        claimed_ack,
        session_filter,
    )

    rng = random.Random(5)
    alphabet = ['"', "\\", "é", "𝄞", '"sid":5', '"ack":9', " ", ":", "x", "{", "}", "0"]
    for _ in range(400):
        sid = rng.choice([0, 1, 12, 99, MAX_SESSION_ID, rng.randrange(MAX_SESSION_ID)])
        plan_id = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 12)))[:32]
        down = Downlink(
            sid=sid,
            ack=rng.choice([0, 7, 10**6, MAX_COUNTER]),
            plan_id=plan_id,
            start_index=rng.randrange(1000),
            waypoints=[Waypoint(lat=rng.uniform(-90, 90), lon=rng.uniform(-180, 180))]
            * rng.randrange(17),
            pause=bool(rng.getrandbits(1)),
        )
        data = encode(down, KEY)
        other = sid + 1 if sid < MAX_SESSION_ID else sid - 1
        assert session_filter(sid)(data) and claimed_ack(data) == down.ack, plan_id
        assert not session_filter(other)(data), plan_id


def test_the_session_prefilter_reads_the_body_robustly_and_only_the_body():
    from backend.src.models.thor_link import MAX_COUNTER, claimed_ack, session_filter

    tag = b"\x00" * 32
    has = session_filter(123)
    assert has(tag + b'{"v":2,"sid":123,"ack":4}') and has(
        tag + b'{"ack": 4 , "sid" :\n123 ,"v":2}'
    )
    assert has(tag + b'{"sid":123}') and has(tag + b'{"sid":123.0,"ack":1}'), "ends where sid does"
    for body in (b'{"sid":1234}', b'{"sid":12}', b'{"sid":0123}', b'{"sid":-123}', b'{"ack":123}'):
        assert not has(tag + body), body
    assert not has(tag + b'{"plan_id":"\\"sid\\":123"}'), "a quoted sid inside a string is no sid"
    assert not has(b'"sid":123'.ljust(32) + b"{}"), "the tag's own bytes are not the body"
    assert session_filter(1)(tag + b'{"sid":1}') and not session_filter(1)(tag + b'{"sid":12}')
    assert claimed_ack(tag + b'{"sid":1}') is None and claimed_ack(tag + b'{"ack":-1}') is None
    assert claimed_ack(tag + b'{"ack": 7,"ack":8}') == 7, "the first member only"
    assert claimed_ack(tag + b'{"ack":%d}' % MAX_COUNTER) == MAX_COUNTER
    assert claimed_ack(tag + b'{"ack":%s}' % (b"9" * 400)) == MAX_COUNTER + 1
    for bad in (-1, 2**48):
        with pytest.raises(ValueError):
            session_filter(bad)
