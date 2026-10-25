"""S-3 / S-4: a flood must not cost the link per packet, and an allowlist must say what it means.

S-3 (review, not demonstrated at the time): the default bind was ``0.0.0.0`` with no source
allowlist and no rate limit, so every junk datagram cost an HMAC, a counter bump and a log check.
Measured here on a loopback flood against the real transport: ~107,000 datagrams/s took 89 % of a
core, all of it HMAC and bookkeeping. Now free filters and a verification budget sit in front of
the HMAC, the tables are bounded, and strangers cannot starve the peer. (SEC20-09 changed what the
budget is keyed on: see ``test_thor_link_spoof_budget.py`` for why a bucket per source address is
not enough, and the ``VerificationBudget`` tests below for its parts.)

S-4: ``ip_network(strict=False)`` turned ``192.168.1.64/24`` into the whole ``/24`` without a word
(and would take ``0.0.0.0/0`` or a slipped ``/8``). Entries are parsed strictly and a wide one has
to be acknowledged.
"""

import asyncio
import inspect
import logging
import math
import socket
import sys
import time

import pytest

from backend.src.models.thor_link import (
    MAX_UPLINK_HZ,
    DatagramBudget,
    VerificationBudget,
)
from backend.src.services import thor_link_service as svc
from backend.src.services.thor_link_service import (
    MAX_SENT_UPLINKS,
    VERIFY_BURST,
    VERIFY_CEILING_HZ,
    ThorLinkService,
    parse_allowed_sources,
)
from backend.src.services.tractor_service import TractorControlService
from tests.unit._thor_link_rig import KEY, LOGGER, OTHER_KEY, THOR, Rig

STRANGER = ("203.0.113.9", 4000)  # an address that is neither the Thor's nor on any allowlist


def _count_hmac_checks(monkeypatch) -> list[int]:
    """Every call into ``decode_downlink`` (the HMAC) the service makes."""
    calls: list[int] = []
    real = svc.decode_downlink

    def counting(data, key):
        calls.append(1)
        return real(data, key)

    monkeypatch.setattr(svc, "decode_downlink", counting)
    return calls


def _stranger(i: int) -> tuple[str, int]:
    return (f"198.51.{(i >> 8) & 255}.{i & 255}", 4000 + i % 1000)


# ------------------------------------ the budget ------------------------------------


def _budget(**kw) -> DatagramBudget:
    p = {"rate_hz": 10.0, "burst": 5, "total_rate_hz": 40.0, "total_burst": 20, **kw}
    return DatagramBudget(p.pop("rate_hz"), p.pop("burst"), **p)


def test_a_source_gets_its_burst_then_its_rate():
    b = _budget()
    assert [b.admit("a", 0.0) for _ in range(7)] == [True] * 5 + [False] * 2
    assert not b.admit("a", 0.05), "half a token is not a token"
    assert b.admit("a", 0.1) and not b.admit("a", 0.1)
    assert [b.admit("a", 10.0) for _ in range(7)] == [True] * 5 + [False] * 2, "refill is capped"


def test_sources_have_separate_buckets_but_share_the_total():
    b = _budget(total_burst=8)
    assert sum(b.admit("a", 0.0) for _ in range(5)) == 5
    assert sum(b.admit("b", 0.0) for _ in range(5)) == 3, "the port's bucket ran dry first"
    assert not b.admit("c", 0.0), "a fresh source cannot get past an empty port bucket"
    assert b.admit("c", 0.25), "the port bucket refills too (40/s)"


def test_a_refusal_by_the_port_bucket_costs_the_source_nothing():
    b = _budget(total_burst=1)
    assert b.admit("a", 0.0)
    assert not b.admit("b", 0.0)  # port empty: refused, and "b" keeps its own tokens ...
    assert sum(b.admit("b", 0.0, exempt_total=True) for _ in range(10)) == 5, "... all five"


def test_the_trusted_source_skips_the_port_bucket_but_not_its_own():
    b = _budget(total_burst=1)
    assert b.admit("stranger", 0.0) and not b.admit("stranger2", 0.0), "port bucket empty"
    admitted = sum(b.admit("peer", 0.0, exempt_total=True) for _ in range(20))
    assert admitted == 5, "bounded by its own bucket, though the port's is empty"


def test_the_table_of_sources_is_bounded():
    b = _budget(max_sources=16, total_burst=10**6, total_rate_hz=10**6)
    for i in range(10_000):
        b.admit(f"10.{i >> 8}.{i & 255}.1", 0.0)
    assert len(b) <= 16
    b.admit("10.0.0.1", 0.0)  # an evicted source is simply new again
    assert len(b) <= 16


def test_the_least_recently_used_source_is_the_one_evicted():
    b = _budget(max_sources=3, burst=2)
    for source in ("a", "b", "c"):
        assert b.admit(source, 0.0) and b.admit(source, 0.0) and not b.admit(source, 0.0)
    b.admit("a", 0.0)  # touch "a": "b" is now the oldest
    b.admit("d", 0.0)  # evicts "b"
    assert not b.admit("c", 0.0), "c is still remembered, and still empty"
    assert not b.admit("a", 0.0), "so is a"
    assert b.admit("b", 0.0), "b was evicted, so it comes back with a full bucket"


def test_a_clock_that_steps_backwards_does_not_mint_tokens():
    b = _budget()
    assert sum(b.admit("a", 100.0) for _ in range(10)) == 5
    assert sum(b.admit("a", 50.0) for _ in range(10)) == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"rate_hz": 0.0},
        {"rate_hz": math.inf},
        {"rate_hz": math.nan},
        {"total_rate_hz": -1.0},
        {"total_rate_hz": math.nan},
        {"burst": 0},
        {"total_burst": 0},
        {"max_sources": 0},
    ],
)
def test_a_budget_that_is_not_a_finite_positive_rate_is_refused(kw):
    with pytest.raises(ValueError):
        _budget(**kw)


# ------------------------------- the verification budget -------------------------------


def _vbudget(**kw) -> VerificationBudget:
    p = {
        "verify_hz": 100.0,
        "verify_burst": 10,
        "peer_hz": 10.0,
        "peer_burst": 5,
        "stranger_hz": 10.0,
        "stranger_burst": 4,
        "strangers_hz": 20.0,
        "strangers_burst": 8,
        **kw,
    }
    return VerificationBudget(**p)


def test_the_ceiling_bounds_the_hmac_checks_of_everyone_together():
    b = _vbudget(peers=["thor"])
    assert sum(b.admit_verification("thor", 0.0) for _ in range(50)) == 10, "the burst"
    assert sum(b.admit_verification("thor", 0.05) for _ in range(50)) == 5, "100/s for 50 ms"
    assert not b.admit_verification("someone", 0.05), "nobody gets past an empty ceiling"
    assert b.verify_tokens(1000.0) == 10, "refill is capped at the burst"


def test_a_stranger_needs_its_own_allowance_and_the_shared_one_as_well():
    b = _vbudget()
    assert sum(b.admit_verification("a", 0.0) for _ in range(50)) == 4, "its own burst"
    assert sum(b.admit_verification("b", 0.0) for _ in range(50)) == 4, "the shared bucket's rest"
    assert not b.admit_verification("c", 0.0), "a fresh stranger meets an empty shared bucket"
    assert b.verify_tokens(0.0) == 2, "and the ceiling was charged for the 8 that got through only"


def test_a_peer_is_not_limited_by_what_strangers_may_have():
    b = _vbudget(verify_burst=1000, peers=["thor"])
    for i in range(5000):
        b.admit_verification(f"198.51.100.{i % 250}", 0.0)  # strangers flooding, rotating
    assert b.verify_tokens(0.0) == 1000 - 8, "they took no more than the shared allowance"
    assert sum(b.admit_verification("thor", 0.0) for _ in range(2000)) == 1000 - 8


def test_a_refusal_by_the_ceiling_costs_a_stranger_nothing():
    b = _vbudget(verify_burst=100, verify_hz=1000.0, peers=["thor"])
    assert b.admit_verification("a", 0.0)  # the stranger spends one of its four
    for _ in range(99):
        assert b.admit_verification("thor", 0.0)  # and now the ceiling is empty
    assert not any(b.admit_verification("a", 0.0) for _ in range(50)), "refused by the ceiling"
    # 50 ms on, the ceiling has 50 tokens again and the stranger still has its other three
    # (plus the half a token it earned): the 50 refusals cost it nothing.
    assert sum(b.admit_verification("a", 0.05) for _ in range(50)) == 3


def test_a_refusal_by_a_strangers_allowance_costs_the_ceiling_nothing():
    b = _vbudget()
    assert sum(b.admit_verification("a", 0.0) for _ in range(100)) == 4
    assert b.verify_tokens(0.0) == 10 - 4


def test_junk_cannot_drain_the_authentic_bucket_of_a_peer():
    b = _vbudget(verify_burst=10_000, peers=["thor"])
    for _ in range(5000):
        assert b.admit_verification("thor", 0.0)  # however much junk carries the peer's address
    assert sum(b.admit_authentic("thor", 0.0) for _ in range(50)) == 5, "the whole burst is left"
    assert sum(b.admit_authentic("thor", 0.2) for _ in range(50)) == 2, "10/s for 200 ms"


def test_a_stranger_has_no_authentic_bucket_until_it_is_trusted():
    b = _vbudget()
    assert all(b.admit_authentic("new", 0.0) for _ in range(100)), "paid for at its own allowance"
    b.trust("new", 0.0)
    assert b.peers == ("new",)
    assert sum(b.admit_authentic("new", 0.0) for _ in range(50)) == 4, "5 minus the accepted one"


def test_peers_are_bounded_and_the_least_recently_trusted_goes_first():
    b = _vbudget(max_peers=3)
    for name in "abcd":
        b.trust(name, 0.0)
    assert b.peers == ("b", "c", "d")
    b.trust("b", 0.0)
    assert b.peers == ("c", "d", "b")
    b.trust("e", 0.0)
    assert b.peers == ("d", "b", "e")
    assert len(_vbudget(max_peers=2, peers=["a", "b", "c"]).peers) == 2


def test_a_configured_peer_starts_with_a_full_authentic_bucket():
    b = _vbudget(peers=["thor"])
    assert sum(b.admit_authentic("thor", 0.0) for _ in range(50)) == 5


def test_the_table_of_strangers_is_bounded_whatever_rotates():
    b = _vbudget(max_strangers=16, verify_burst=10**6, strangers_burst=10**6)
    for i in range(10_000):
        b.admit_verification(f"10.{i >> 8}.{i & 255}.1", 0.0)
    assert len(b) <= 16 and len(b.peers) == 0


def test_a_clock_that_steps_backwards_neither_mints_tokens_nor_rewinds_the_stamp():
    b = _vbudget(verify_burst=10_000, peers=["thor"])
    assert sum(b.admit_verification("thor", 100.0) for _ in range(10_050)) == 10_000
    assert not any(b.admit_verification("thor", 50.0) for _ in range(5)), "no tokens from the past"
    # Back at 100.0625 only 62.5 ms have passed since the last charge (6.25 tokens at 100/s), not
    # 50 s: stepping back and forward again must not have rewound the stamp and minted the lot.
    assert sum(b.admit_verification("thor", 100.0625) for _ in range(100)) == 6


@pytest.mark.parametrize(
    "kw",
    [
        {"verify_hz": 0.0},
        {"verify_hz": math.inf},
        {"peer_hz": -1.0},
        {"stranger_hz": math.nan},
        {"strangers_hz": 0.0},
        {"verify_burst": 0},
        {"peer_burst": 0},
        {"stranger_burst": 0},
        {"strangers_burst": 0},
        {"max_strangers": 0},
        {"max_peers": 0},
    ],
)
def test_a_verification_budget_that_is_not_a_finite_positive_rate_is_refused(kw):
    with pytest.raises(ValueError):
        _vbudget(**kw)


# --------------------------------- the service under flood ---------------------------------


async def test_a_flood_from_one_stranger_reaches_the_hmac_only_up_to_its_allowance(monkeypatch):
    checks = _count_hmac_checks(monkeypatch)
    rig = Rig()
    await rig.tick()
    junk = rig.answer(key=OTHER_KEY)  # the sid, an ack the Pi waits for, the wrong key
    for _ in range(5000):
        rig.deliver(junk, STRANGER)
    assert 0 < len(checks) <= rig.link._sent_cap, "5000 packets, at most a burst of HMACs"
    assert rig.link.rejected == 5000
    assert rig.link.rejects["rate"] == 5000 - len(checks)
    assert rig.link.rejects["auth"] == len(checks) == rig.link.verifications
    # a second of quiet earns the source its rate again (20/s), no more
    rig.clock.t += 1.0
    before = len(checks)
    for _ in range(5000):
        rig.deliver(junk, STRANGER)
    assert 0 < len(checks) - before <= 21


async def test_a_flood_that_rotates_its_source_is_stopped_by_the_strangers_shared_bucket(
    monkeypatch,
):
    checks = _count_hmac_checks(monkeypatch)
    rig = Rig()
    await rig.tick()
    junk = rig.answer(key=OTHER_KEY)
    for i in range(5000):
        rig.deliver(junk, _stranger(i))
    shared_burst = 4 * rig.link._sent_cap
    assert 0 < len(checks) <= shared_burst
    assert rig.link.rejects["rate"] >= 5000 - shared_burst
    assert len(rig.link._budget) <= 64, "and the table of sources did not grow with the sources"


async def test_the_thor_is_not_starved_by_a_flood_of_strangers():
    rig = Rig()
    await rig.connect()
    await rig.tick()
    junk = rig.answer(key=OTHER_KEY)
    for i in range(5000):
        rig.deliver(junk, _stranger(i))
    # no time passes: the strangers' shared bucket is empty when the Thor's next answer arrives
    rejected_before = rig.link.rejected
    rig.deliver(rig.answer(plan_id="through-the-flood"))  # the Thor's own answer
    assert rig.link.rejected == rejected_before and rig.link.plan.plan_id == "through-the-flood"
    # ... whereas the very same kind of answer from a stranger is shed before it is looked at
    await rig.tick()
    rig.deliver(rig.answer(plan_id="from-a-stranger"), ("192.0.2.77", 47101))
    assert rig.link.rejects["rate"] > 0 and rig.link.plan.plan_id == "through-the-flood"


async def test_a_flood_that_wears_the_thors_address_is_bounded_by_the_ceiling_alone(monkeypatch):
    rig = Rig()
    await rig.connect()
    rig.clock.t += 1.0  # the ceiling is full again (connect() spent a token on the Thor's answer)
    await rig.tick()
    checks = _count_hmac_checks(monkeypatch)
    junk = rig.answer(key=OTHER_KEY)
    for _ in range(5000):
        rig.deliver(junk, THOR)
    assert len(checks) == VERIFY_BURST, "the ceiling's whole burst and not one HMAC more"
    assert rig.link.rejects["rate"] == 5000 - VERIFY_BURST
    rig.clock.t += 0.125  # 125 ms of ceiling
    before = len(checks)
    for _ in range(5000):
        rig.deliver(junk, THOR)
    assert abs((len(checks) - before) - VERIFY_CEILING_HZ * 0.125) <= 1


async def test_a_peer_that_holds_the_key_is_bounded_in_the_work_it_causes():
    rig = Rig()
    await rig.tick()
    burst = rig.link._sent_cap
    # Authentic (the right key) but not usable: a negative start_index breaks the schema. Distinct
    # bytes each, so none is a copy of another and every one costs its HMAC.
    for i in range(100):
        rig.deliver(rig.answer(plan_id=f"flood-{i}", start_index=-1))
    assert rig.link.verifications == 100
    assert rig.link.rejects["malformed"] == burst, "the authentic bucket's burst"
    assert rig.link.rejects["peer_rate"] == 100 - burst
    rig.clock.t += 0.125  # at 20/s: two and a half more tokens
    for i in range(100, 200):
        rig.deliver(rig.answer(plan_id=f"flood-{i}", start_index=-1))
    assert rig.link.rejects["malformed"] == burst + 2


async def test_honest_traffic_never_touches_the_budget():
    rig = Rig()
    for _ in range(600):  # a minute at 10 Hz
        await rig.tick()
        rig.deliver(rig.answer())
        rig.clock.t += 0.1
    assert rig.link.rejected == 0 and rig.link.rejects["rate"] == 0
    assert rig.link.verifications == 600, "one HMAC per answer and not one more"
    assert rig.link.link_fresh() and not rig.link.safe_hold


async def test_answers_queued_by_a_radio_stall_arrive_together_and_all_count():
    rig = Rig()
    for _ in range(9):  # nine uplinks go out while the radio is stalled ...
        await rig.tick()
        rig.clock.t += 0.1
    for up in rig.uplinks:  # ... and the Thor's nine answers arrive in one burst
        rig.deliver(rig.answer(up))
    assert rig.link.rejected == 0 and rig.link._last_ack == 8


async def test_a_failure_inside_the_datagram_handler_is_counted_not_raised(caplog):
    rig = Rig()
    await rig.tick()

    def boom(msg, now):
        raise RuntimeError("bug in admit")

    rig.link._admit = boom
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        for _ in range(15):
            rig.deliver(rig.answer())  # must not raise: asyncio would log a traceback each time
    assert rig.link.rejects["internal"] == 15 and rig.link.rejected == 15
    assert rig.link.plan is None and rig.link._last_ack == -1 and not rig.link.link_fresh()
    failed = [r for r in caplog.records if "handler failed" in r.getMessage()]
    assert len(failed) == 1 and failed[0].exc_info, "one logged traceback, however many datagrams"


@pytest.mark.parametrize(
    ("source", "who"), [("127.0.0.2", "a stranger"), ("127.0.0.1", "the Thor's own address")]
)
async def test_a_real_loopback_flood_leaves_the_loop_responsive_and_the_hmac_bounded(
    monkeypatch, source, who
):
    """The review's 'confirm with a loopback flood against the real transport', at small scale.

    The flood claims the sid and an ack the Pi still waits for, so every datagram of it reaches the
    budget. From a stranger the bound is its own allowance (a burst, then 20/s); from the Thor's
    address it is the verification ceiling, and nothing else.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((source, 0))
    except OSError:
        pytest.skip(f"{source} is not available as a loopback address")
    finally:
        probe.close()
    checks = _count_hmac_checks(monkeypatch)
    tractor = TractorControlService(config={})
    link = ThorLinkService(
        tractor,
        ("127.0.0.1", 9),
        bind_addr=("127.0.0.1", 0),
        key=KEY,
        uplink_hz=10,
        plan_max_age_s=30.0,  # ack #0, which the flood claims, stays one the Pi waits for
    )
    await link.start()
    port = link._transport.get_extra_info("sockname")[1]
    forged = b"\x00" * 32 + b'{"v":2,"sid":%d,"ack":0,"plan_id":"p"}' % link.session_id
    flooder = (
        "import socket, sys, time\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "s.bind((sys.argv[3], 0))\n"
        "data = bytes.fromhex(sys.argv[4])\n"
        "end = time.monotonic() + float(sys.argv[2])\n"
        "while time.monotonic() < end:\n"
        "    for _ in range(200):\n"
        "        try:\n"
        "            s.sendto(data, ('127.0.0.1', int(sys.argv[1])))\n"
        "        except OSError:\n"
        "            pass\n"
    )
    lags: list[float] = []
    running = True

    async def ticker():
        last = time.monotonic()
        while running:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            lags.append(now - last - 0.01)
            last = now

    started = time.monotonic()
    watcher = asyncio.create_task(ticker())
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", flooder, str(port), "0.8", source, forged.hex()
        )
        await proc.wait()
        await asyncio.sleep(0.2)  # let the socket buffer drain
        elapsed = time.monotonic() - started
    finally:
        running = False
        await watcher
        await link.stop()
    if source == "127.0.0.1":
        allowed = VERIFY_BURST + VERIFY_CEILING_HZ * elapsed + 2  # the ceiling and nothing else
    else:
        allowed = link._sent_cap + 20 * elapsed + 2  # one stranger: its burst plus 20/s
    assert link.rejected > allowed, f"the flood from {who} did not arrive: {link.rejected}"
    assert link.rejects["rate"] > 0
    assert len(checks) <= allowed, f"{len(checks)} HMAC checks in {elapsed:.2f} s from {who}"
    assert link.rejects["auth"] == len(checks) == link.verifications
    assert max(lags) < 0.75, f"the event loop stalled for {max(lags):.3f} s"
    assert link.uplink_errors == 0 and link.safe_hold, "a flood moved nothing"


# ------------------------------- S-4: the allowlist, strictly -------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("10.0.0.2", "10.0.0.2/32"),
        ("10.0.0.0/24", "10.0.0.0/24"),
        ("192.168.0.0/16", "192.168.0.0/16"),
        ("192.168.1.5/32", "192.168.1.5/32"),
        ("fd00:1::/64", "fd00:1::/64"),
        ("::1", "::1/128"),
    ],
)
def test_exact_addresses_and_ordinary_networks_are_accepted(entry, expected):
    assert [str(n) for n in parse_allowed_sources([entry])] == [expected]


def test_a_host_written_as_a_wider_network_is_refused_not_widened():
    with pytest.raises(ValueError, match="host bits") as err:
        parse_allowed_sources(["192.168.1.64/24"])
    text = str(err.value)
    assert "192.168.1.0/24" in text and "192.168.1.64" in text, "it says what to write instead"
    for typo in ("10.1.2.3/8", "172.16.5.9/12", "192.168.1.1/16", "fd00::1/64"):
        with pytest.raises(ValueError, match="host bits"):
            parse_allowed_sources([typo])


@pytest.mark.parametrize("entry", ["0.0.0.0/0", "::/0", "0.0.0.0/0.0.0.0"])
def test_an_entry_that_matches_everyone_is_refused_even_when_wide_is_allowed(entry):
    for wide in (False, True):
        with pytest.raises(ValueError, match="matches every address"):
            parse_allowed_sources([entry], allow_wide=wide)


@pytest.mark.parametrize("entry", ["10.0.0.0/8", "172.16.0.0/12", "192.0.0.0/15", "fe80::/10"])
def test_a_network_wider_than_one_lan_needs_to_be_acknowledged(entry):
    with pytest.raises(ValueError, match="allow_wide_sources"):
        parse_allowed_sources([entry])
    assert parse_allowed_sources([entry], allow_wide=True)[0].with_prefixlen == entry


@pytest.mark.parametrize(
    "entry",
    ["not-an-ip", "10.0.0.256", "10.0.0.0/33", "10.0.0.0/", "", " ", "10.0.0.1 ", None, 3232235777],
)
def test_garbage_is_refused_whatever_it_looks_like(entry):
    with pytest.raises(ValueError, match="allowed_sources"):
        parse_allowed_sources([entry])


def test_an_empty_allowlist_is_an_error_and_none_means_no_allowlist():
    with pytest.raises(ValueError, match="empty"):
        parse_allowed_sources([])
    assert parse_allowed_sources(None) == ()


def test_one_bad_entry_spoils_the_whole_list():
    with pytest.raises(ValueError, match="host bits"):
        parse_allowed_sources(["10.0.0.2", "192.168.1.64/24", "10.9.9.9"])


async def test_the_review_repro_a_typo_no_longer_opens_the_whole_subnet():
    """``192.168.1.64/24`` used to allow 192.168.1.200 without a word."""
    with pytest.raises(ValueError, match="host bits"):
        Rig(allowed_sources=["192.168.1.64/24"])


async def test_a_wide_list_works_once_it_is_acknowledged_and_still_filters():
    rig = Rig(allowed_sources=["10.0.0.0/8"], allow_wide_sources=True)
    await rig.tick()
    good = rig.answer()
    rig.deliver(good, ("11.0.0.1", 1))
    assert rig.link.rejects["source"] == 1
    rig.deliver(good, ("10.200.1.1", 1))
    assert rig.link.link_fresh()
    with pytest.raises(ValueError, match="allow_wide_sources"):
        Rig(allowed_sources=["10.0.0.0/8"])


# ------------------------------------ bounded work ------------------------------------


@pytest.mark.parametrize("hz", [1e9, 1e6, MAX_UPLINK_HZ + 0.1, math.inf])
def test_an_uplink_rate_that_would_make_the_slow_loop_a_busy_loop_is_refused(hz):
    with pytest.raises(ValueError, match="uplink_hz"):
        ThorLinkService(TractorControlService(config={}), THOR, key=KEY, uplink_hz=hz)


def test_the_uplink_ceiling_itself_is_allowed():
    ThorLinkService(TractorControlService(config={}), THOR, key=KEY, uplink_hz=MAX_UPLINK_HZ)


@pytest.mark.parametrize("name", ["timeout_s", "uplink_hz", "plan_max_age_s", "neutral_timeout_s"])
@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_a_timing_that_is_not_finite_would_mean_never_time_out_and_is_refused(name, bad):
    with pytest.raises(ValueError, match="finite"):
        ThorLinkService(TractorControlService(config={}), THOR, key=KEY, **{name: bad})


async def test_a_huge_plan_age_cannot_grow_the_table_of_sent_uplinks_without_limit():
    rig = Rig(plan_max_age_s=1e9, uplink_hz=50)
    assert rig.link._sent_cap <= MAX_SENT_UPLINKS
    for _ in range(MAX_SENT_UPLINKS + 500):
        await rig.tick()
        rig.clock.t += 0.02
    assert len(rig.link._sent) <= MAX_SENT_UPLINKS


def test_the_service_listens_on_loopback_unless_told_otherwise():
    """A forgotten bind address must hear nobody, as in the wiring's own default."""
    default = inspect.signature(ThorLinkService.__init__).parameters["bind_addr"].default
    assert default[0] == "127.0.0.1"
