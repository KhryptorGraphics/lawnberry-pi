"""Which forwarders may name the client address (review SEC20-07 and SEC20-12).

SEC20-07: ``X-Forwarded-For`` was believed from every address of the loopback by default, so any
forwarder on the host that passes a client-written header through (``ssh -R``, ``socat``, a custom
tunnel command, an nginx WebSocket location without the header) let its client pick the identity
the limiters and the login lockout count: twelve wrong logins with rotating
``X-Forwarded-For: 198.51.100.<i>`` from 127.0.0.1 were all 401, no 429.

SEC20-12: ``LAWN_BERRY_TRUSTED_PROXIES`` was parsed with ``strict=False``: ``192.168.1.64/24``
silently became the whole ``192.168.1.0/24`` and ``0.0.0.0/0`` was accepted.

Now nobody is trusted unless ``LAWN_BERRY_TRUSTED_PROXIES`` names the forwarder, its address
(each forwarder gets its own) and what it does to the header (``single`` or ``last``); the list is
parsed strictly and an invalid one trusts nobody instead of widening.
"""

from __future__ import annotations

import logging

import pytest

from backend.src.core import peer_address
from backend.src.core.peer_address import (
    TRUSTED_PROXIES_ENV,
    ProxyConfigError,
    TrustedProxy,
    describe_trusted_proxies,
    is_loopback_address,
    is_trusted_proxy_peer,
    log_proxy_posture,
    parse_trusted_proxies,
    peer_key,
    resolve_client_host,
    trusted_proxies,
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.delenv(TRUSTED_PROXIES_ENV, raising=False)
    peer_address._read_trusted_proxies.cache_clear()
    peer_address._last_warned.clear()
    yield
    peer_address._read_trusted_proxies.cache_clear()
    peer_address._last_warned.clear()


UI = "127.0.0.2=single"  # the shipped UI server: replaces the header with ONE hop
NGINX = "127.0.0.3=last"  # a repo nginx: appends the address it saw


# ---- strict parsing (SEC20-12) -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("", []),
        ("   ", []),
        (",,", []),
        ("127.0.0.2", ["127.0.0.2/32=single"]),  # the default shape is the strict one
        ("127.0.0.2=single", ["127.0.0.2/32=single"]),
        ("127.0.0.3=last", ["127.0.0.3/32=last"]),
        (" 127.0.0.3 = LAST ,", ["127.0.0.3/32=last"]),  # blanks, case and a trailing comma
        ("127.0.0.2=single,127.0.0.3=last", ["127.0.0.2/32=single", "127.0.0.3/32=last"]),
        ("10.1.2.0/24=last", ["10.1.2.0/24=last"]),
        ("::1", ["::1/128=single"]),
        ("2001:db8:1:2::/64=last", ["2001:db8:1:2::/64=last"]),
    ],
)
def test_valid_lists_parse_exactly(raw, expected):
    assert [str(p) for p in parse_trusted_proxies(raw)] == expected


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        # a host written as a wider network is refused, not quietly widened (the review's repro)
        ("192.168.1.64/24", "host bits"),
        ("10.1.2.3/8", "host bits"),
        ("fd00::1/64", "host bits"),
        # everyone
        ("0.0.0.0/0", "matches every address"),
        ("::/0", "matches every address"),
        # more than one LAN: a forwarder is a host
        ("10.0.0.0/8", "more than one LAN"),
        ("127.0.0.0/8", "more than one LAN"),  # the old default
        ("192.168.0.0/16", "more than one LAN"),
        ("172.16.0.0/12", "more than one LAN"),
        ("fe80::/10", "more than one LAN"),
        # not addresses, unknown shapes, zone ids
        ("nonsense", "not an IP address"),
        ("localhost", "not an IP address"),
        ("127.0.0.2:8081", "not an IP address"),
        ("127.0.0.2=weird", "unknown shape"),
        ("127.0.0.2=", "unknown shape"),
        ("fe80::1%eth0", "zone id"),
    ],
)
def test_defective_entries_are_refused_with_an_explanation(raw, needle):
    with pytest.raises(ProxyConfigError) as err:
        parse_trusted_proxies(raw)
    assert needle in str(err.value)
    assert TRUSTED_PROXIES_ENV in str(err.value)


def test_a_host_with_host_bits_is_told_what_to_write_instead():
    with pytest.raises(ProxyConfigError) as err:
        parse_trusted_proxies("192.168.1.64/24")
    text = str(err.value)
    assert "192.168.1.0/24" in text and "192.168.1.64" in text


def test_one_bad_entry_spoils_the_whole_list_and_every_problem_is_listed():
    with pytest.raises(ProxyConfigError) as err:
        parse_trusted_proxies("127.0.0.2, 0.0.0.0/0, 192.168.1.64/24, 127.0.0.3=nope")
    assert len(err.value.problems) == 3  # the good entry is not one of them
    joined = " ".join(err.value.problems)
    assert "matches every address" in joined and "host bits" in joined and "unknown shape" in joined


def test_the_widest_accepted_networks_are_one_lan():
    assert parse_trusted_proxies("10.1.2.0/24") and parse_trusted_proxies("fd00:1:2:3::/64")
    for refused in ("10.1.0.0/23", "fd00:1:2::/63"):
        with pytest.raises(ProxyConfigError):
            parse_trusted_proxies(refused)


# ---- the runtime reader fails closed, loudly -----------------------------------------------------


def test_nobody_is_trusted_by_default():
    assert trusted_proxies({}) == ()
    assert trusted_proxies({TRUSTED_PROXIES_ENV: ""}) == ()
    assert describe_trusted_proxies(()) == "none"
    # the old default: the whole loopback believed on every address
    for peer in ("127.0.0.1", "127.0.0.2", "::1"):
        assert resolve_client_host(peer, ["203.0.113.9"], trusted_proxies({})) == peer


@pytest.mark.parametrize(
    "raw", ["0.0.0.0/0", "192.168.1.64/24", "10.0.0.0/8", "127.0.0.2, bogus", "127.0.0.2=nope"]
)
def test_an_invalid_value_trusts_nobody_and_says_why_once(raw, caplog):
    environ = {TRUSTED_PROXIES_ENV: raw}
    with caplog.at_level(logging.ERROR, logger=peer_address.logger.name):
        assert trusted_proxies(environ) == ()
        assert trusted_proxies(environ) == ()  # cached: still nobody, no second line
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert TRUSTED_PROXIES_ENV in errors[0].getMessage()
    assert "NO forwarder is trusted" in errors[0].getMessage()
    # the review's repros: the network is NOT believed
    assert resolve_client_host("10.1.2.3", ["203.0.113.9"], trusted_proxies(environ)) == "10.1.2.3"
    assert (
        resolve_client_host("192.168.1.9", ["6.6.6.6"], trusted_proxies(environ)) == "192.168.1.9"
    )


def test_a_valid_value_is_read_from_the_process_environment(monkeypatch):
    monkeypatch.setenv(TRUSTED_PROXIES_ENV, f"{UI},{NGINX}")
    assert [str(p) for p in trusted_proxies()] == ["127.0.0.2/32=single", "127.0.0.3/32=last"]
    assert describe_trusted_proxies() == "127.0.0.2/32=single, 127.0.0.3/32=last"


# ---- the header is read according to the declared shape (SEC20-07) -------------------------------


def _resolve(peer, lines, raw=f"{UI},{NGINX}"):
    return resolve_client_host(peer, lines, parse_trusted_proxies(raw))


@pytest.mark.parametrize(
    ("peer", "lines", "expected"),
    [
        # single: exactly one valid hop is believed
        ("127.0.0.2", ["203.0.113.9"], "203.0.113.9"),
        ("127.0.0.2", [" 203.0.113.9 "], "203.0.113.9"),
        ("127.0.0.2", ["203.0.113.9:51234"], "203.0.113.9"),
        ("127.0.0.2", ["[2001:db8::7]:51234"], "2001:db8::7"),
        ("127.0.0.2", ["::ffff:203.0.113.9"], "203.0.113.9"),
        ("127.0.0.2", ["127.0.0.1"], "127.0.0.1"),  # the local browser: its own bucket
        ("::ffff:127.0.0.2", ["203.0.113.9"], "203.0.113.9"),
        # single: anything else does not fit the declared shape and is NOT believed
        ("127.0.0.2", ["6.6.6.6, 203.0.113.9"], "127.0.0.2"),  # a pass-through proxy appended
        ("127.0.0.2", ["6.6.6.6", "203.0.113.9"], "127.0.0.2"),  # two header lines
        ("127.0.0.2", ["203.0.113.9,"], "203.0.113.9"),  # one hop and a stray comma is still one
        ("127.0.0.2", ["unknown"], "127.0.0.2"),
        ("127.0.0.2", ["garbage"], "127.0.0.2"),
        ("127.0.0.2", [""], "127.0.0.2"),
        ("127.0.0.2", [], "127.0.0.2"),
        # last: the hop the forwarder appended, nothing to its left
        ("127.0.0.3", ["203.0.113.9"], "203.0.113.9"),
        ("127.0.0.3", ["6.6.6.6, 203.0.113.9"], "203.0.113.9"),
        ("127.0.0.3", ["6.6.6.6", "203.0.113.9"], "203.0.113.9"),
        ("127.0.0.3", ["6.6.6.6, 203.0.113.9, 127.0.0.3"], "203.0.113.9"),  # a chain: skip hops
        ("127.0.0.3", ["203.0.113.9, unknown"], "127.0.0.3"),  # the appended hop is junk
        ("127.0.0.3", ["127.0.0.2"], "127.0.0.3"),  # only forwarder hops: nobody to name
        ("127.0.0.3", [], "127.0.0.3"),
        # NOT declared: the header is never read, whatever it says
        ("127.0.0.1", ["203.0.113.9"], "127.0.0.1"),
        ("127.0.0.4", ["203.0.113.9"], "127.0.0.4"),
        ("::1", ["203.0.113.9"], "::1"),
        ("198.51.100.9", ["203.0.113.9"], "198.51.100.9"),
        ("198.51.100.9", ["127.0.0.1"], "198.51.100.9"),
        ("testclient", ["203.0.113.9"], "testclient"),
        ("", ["203.0.113.9"], ""),
        (None, ["203.0.113.9"], None),
    ],
)
def test_the_header_is_believed_only_in_the_declared_shape(peer, lines, expected):
    assert _resolve(peer, lines) == expected


def test_a_header_transparent_loopback_forwarder_cannot_choose_its_identity():
    """The review's repro: from 127.0.0.1 (an ssh -R / socat / frp on the host that does not
    rewrite the header) twelve different ``X-Forwarded-For`` values must be ONE client."""
    keys = {
        peer_key(resolve_client_host("127.0.0.1", [f"198.51.100.{n}"], trusted_proxies({})))
        for n in range(12)
    }
    assert keys == {"ip:127.0.0.1"}
    # ... and declaring a DIFFERENT forwarder does not make the loopback believable either
    keys = {peer_key(_resolve("127.0.0.1", [f"198.51.100.{n}"], raw=UI)) for n in range(12)}
    assert keys == {"ip:127.0.0.1"}


def test_a_proxy_that_passes_a_client_written_header_through_cannot_choose_either():
    """The UI server is declared ``single``: an old server.mjs that APPENDED to the client's header
    sends two hops, so the client-written left part is never believed and cannot pick a bucket."""
    keys = {peer_key(_resolve("127.0.0.2", [f"198.51.100.{n}, 203.0.113.9"])) for n in range(12)}
    assert keys == {"ip:127.0.0.2"}  # one shared budget: safe, never attacker-chosen


def test_a_declared_forwarder_still_gets_one_budget_per_client():
    keys = {peer_key(_resolve("127.0.0.2", [f"198.51.100.{n}"])) for n in range(12)}
    assert len(keys) == 12


def test_the_second_hop_of_a_single_forwarder_is_warned_about_once_a_minute(monkeypatch, caplog):
    clock = {"t": 1000.0}
    monkeypatch.setattr(peer_address.time, "monotonic", lambda: clock["t"])
    with caplog.at_level(logging.WARNING, logger=peer_address.logger.name):
        for _ in range(50):
            _resolve("127.0.0.2", ["6.6.6.6, 203.0.113.9"])
        assert len([r for r in caplog.records if "single" in r.getMessage()]) == 1
        clock["t"] += peer_address._WARN_INTERVAL_S + 1
        _resolve("127.0.0.2", ["6.6.6.6, 203.0.113.9"])
        assert len([r for r in caplog.records if "single" in r.getMessage()]) == 2


def test_the_warning_table_is_bounded(monkeypatch):
    for n in range(peer_address._MAX_WARN_KEYS * 3):
        peer_address._warn_limited(f"k{n}", "x")
    assert len(peer_address._last_warned) <= peer_address._MAX_WARN_KEYS


# ---- helpers used by the WebSocket admission -----------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.5.5.5", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("[::1]:80", True),
        ("203.0.113.9", False),
        ("192.168.1.9", False),
        ("localhost", False),  # a name is not an address: never guessed
        ("", False),
        (None, False),
    ],
)
def test_is_loopback_address(host, expected):
    assert is_loopback_address(host) is expected


def test_is_trusted_proxy_peer_follows_the_declaration(monkeypatch):
    class Conn:
        def __init__(self, host):
            self.client = (host, 1234)

    assert not is_trusted_proxy_peer(Conn("127.0.0.2"))
    monkeypatch.setenv(TRUSTED_PROXIES_ENV, UI)
    assert is_trusted_proxy_peer(Conn("127.0.0.2"))
    assert not is_trusted_proxy_peer(Conn("127.0.0.1"))
    assert not is_trusted_proxy_peer(Conn("203.0.113.9"))

    class NoClient:
        client = None

    assert not is_trusted_proxy_peer(NoClient())


def test_trusted_proxy_objects_are_value_types():
    a = parse_trusted_proxies(UI)[0]
    assert a == TrustedProxy(a.network, "single") and a != TrustedProxy(a.network, "last")


# ---- start-up posture ----------------------------------------------------------------------------


def test_the_posture_log_names_the_resolved_forwarders(caplog):
    with caplog.at_level(logging.INFO, logger=peer_address.logger.name):
        log_proxy_posture(argv=["pytest"], environ={TRUSTED_PROXIES_ENV: f"{UI},{NGINX}"})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "127.0.0.2/32=single" in text and "127.0.0.3/32=last" in text


def test_the_posture_log_says_none_when_nobody_is_trusted(caplog):
    with caplog.at_level(logging.INFO, logger=peer_address.logger.name):
        log_proxy_posture(argv=["pytest"], environ={})
    assert any("none" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("argv", "warns"),
    [
        (["/venv/bin/uvicorn", "backend.src.main:app", "--port", "8081"], True),
        (["/venv/bin/uvicorn", "backend.src.main:app", "--proxy-headers"], True),
        (["/venv/bin/uvicorn", "backend.src.main:app", "--no-proxy-headers"], False),
        (["python", "-m", "uvicorn", "x:app"], True),
        (["pytest", "tests"], False),
        ([], False),
    ],
)
def test_the_posture_log_warns_when_uvicorn_still_decides_the_client(argv, warns, caplog):
    with caplog.at_level(logging.INFO, logger=peer_address.logger.name):
        log_proxy_posture(argv=argv, environ={"FORWARDED_ALLOW_IPS": "*"})
    flagged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert bool(flagged) is warns
    if warns:
        assert "--no-proxy-headers" in flagged[0].getMessage()
        assert "FORWARDED_ALLOW_IPS=*" in flagged[0].getMessage()


def test_an_invalid_value_shows_up_in_the_posture_log_as_an_error(caplog):
    with caplog.at_level(logging.INFO, logger=peer_address.logger.name):
        log_proxy_posture(argv=["pytest"], environ={TRUSTED_PROXIES_ENV: "0.0.0.0/0"})
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert any("none" in r.getMessage() for r in caplog.records)
