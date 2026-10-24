"""S-3 / S-4: a flood must not cost the link per packet, and an allowlist must say what it means.

S-3 (review, not demonstrated at the time): the default bind was ``0.0.0.0`` with no source
allowlist and no rate limit, so every junk datagram cost an HMAC, a counter bump and a log check.
Measured here on a loopback flood against the real transport: ~107,000 datagrams/s took 89 % of a
core, all of it HMAC and bookkeeping. Now a token bucket per source and one for the port sit in
front of the HMAC, the table of sources is bounded, and the peer that is already authenticated is
not starved by strangers.

S-4: ``ip_network(strict=False)`` turned ``192.168.1.64/24`` into the whole ``/24`` without a word
(and would take ``0.0.0.0/0`` or a slipped ``/8``). Entries are parsed strictly and a wide one has
to be acknowledged.
"""

import asyncio
import inspect
import logging
import math
import sys
import time

import pytest

from backend.src.models.thor_link import (
    MAX_UPLINK_HZ,
    DatagramBudget,
)
from backend.src.services import thor_link_service as svc
from backend.src.services.thor_link_service import (
    MAX_SENT_UPLINKS,
    ThorLinkService,
    parse_allowed_sources,
)
from backend.src.services.tractor_service import TractorControlService
from tests.unit._thor_link_rig import KEY, LOGGER, THOR, Rig

JUNK = b"\x5a" * 1200  # a full-size datagram that is not authentic


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


# --------------------------------- the service under flood ---------------------------------


async def test_a_flood_from_one_source_reaches_the_hmac_only_up_to_the_budget(monkeypatch):
    checks = _count_hmac_checks(monkeypatch)
    rig = Rig()
    for _ in range(5000):
        rig.deliver(JUNK, ("203.0.113.9", 4000))
    assert 0 < len(checks) <= rig.link._sent_cap, "5000 packets, one HMAC each (the old code)"
    assert rig.link.rejected == 5000
    assert rig.link.rejects["rate"] == 5000 - len(checks)
    assert rig.link.rejects["auth"] == len(checks)
    # a second of quiet earns the source its rate again (20/s), no more
    rig.clock.t += 1.0
    before = len(checks)
    for _ in range(5000):
        rig.deliver(JUNK, ("203.0.113.9", 4000))
    assert 0 < len(checks) - before <= 21


async def test_a_flood_that_rotates_its_source_is_stopped_by_the_port_bucket(monkeypatch):
    checks = _count_hmac_checks(monkeypatch)
    rig = Rig()
    for i in range(5000):
        rig.deliver(JUNK, _stranger(i))
    total_burst = 4 * rig.link._sent_cap
    assert 0 < len(checks) <= total_burst
    assert rig.link.rejects["rate"] >= 5000 - total_burst
    assert len(rig.link._budget) <= 64, "and the table of sources did not grow with the sources"


async def test_the_peer_already_accepted_is_not_starved_by_a_flood_of_strangers():
    rig = Rig()
    await rig.connect()  # the Thor (10.0.0.2) has delivered an accepted downlink: it is trusted
    for i in range(5000):
        rig.deliver(JUNK, _stranger(i))
    # no time passes: the port's bucket is still empty when the Thor's next answer arrives
    await rig.tick()
    rejected_before = rig.link.rejected
    rig.deliver(rig.answer(plan_id="through-the-flood"))  # the Thor's own answer
    assert rig.link.rejected == rejected_before and rig.link.plan.plan_id == "through-the-flood"
    # ... whereas the very same kind of answer from anyone else is shed before it is looked at
    await rig.tick()
    rig.deliver(rig.answer(plan_id="from-a-stranger"), ("192.0.2.77", 47101))
    assert rig.link.rejects["rate"] > 0 and rig.link.plan.plan_id == "through-the-flood"


async def test_the_trusted_peer_is_still_bounded_by_its_own_bucket(monkeypatch):
    rig = Rig()
    await rig.connect()
    checks = _count_hmac_checks(monkeypatch)
    for _ in range(5000):
        rig.deliver(JUNK, THOR)  # a flood that spoofs the Thor's own address
    assert len(checks) <= rig.link._sent_cap
    assert rig.link.rejects["rate"] >= 5000 - rig.link._sent_cap


async def test_honest_traffic_never_touches_the_budget():
    rig = Rig()
    for _ in range(600):  # a minute at 10 Hz
        await rig.tick()
        rig.deliver(rig.answer())
        rig.clock.t += 0.1
    assert rig.link.rejected == 0 and rig.link.rejects["rate"] == 0
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


async def test_a_real_loopback_flood_leaves_the_loop_responsive_and_the_hmac_idle(monkeypatch):
    """The review's 'confirm with a loopback flood against the real transport', at small scale."""
    checks = _count_hmac_checks(monkeypatch)
    tractor = TractorControlService(config={})
    link = ThorLinkService(
        tractor, ("127.0.0.1", 9), bind_addr=("127.0.0.1", 0), key=KEY, uplink_hz=10
    )
    await link.start()
    port = link._transport.get_extra_info("sockname")[1]
    flooder = (
        "import socket, sys, time\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "end = time.monotonic() + float(sys.argv[2])\n"
        "while time.monotonic() < end:\n"
        "    for _ in range(200):\n"
        "        try:\n"
        "            s.sendto(b'Z' * 1200, ('127.0.0.1', int(sys.argv[1])))\n"
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
        proc = await asyncio.create_subprocess_exec(sys.executable, "-c", flooder, str(port), "0.8")
        await proc.wait()
        await asyncio.sleep(0.2)  # let the socket buffer drain
        elapsed = time.monotonic() - started
    finally:
        running = False
        await watcher
        await link.stop()
    assert link.bytes_received > 50 * 1200, "the flood did arrive"
    assert link.rejects["rate"] > 0
    allowed = link._sent_cap + 20 * elapsed + 2  # one source: burst plus 20/s
    assert len(checks) <= allowed, f"{len(checks)} HMAC checks in {elapsed:.2f} s"
    assert link.rejects["auth"] == len(checks)
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
