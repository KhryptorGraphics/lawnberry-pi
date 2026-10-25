"""The key every rate limit and login lockout is counted against: the client's address.

Login lockout and the request limiters used to be keyed on the client-supplied ``X-Client-Id``
header: sending a fresh value with each request gave an unlimited number of password guesses and
an unbounded number of limiter entries. The only inputs that are not the caller's to choose are
what the kernel reports for the connection (``request.client.host``) and, behind a forwarder that
was explicitly declared trustworthy, the address that forwarder recorded.

Forwarder trust is EXPLICIT and defaults to NONE
------------------------------------------------
``LAWN_BERRY_TRUSTED_PROXIES`` lists the forwarders (reverse proxy, UI server, tunnel agent) whose
``X-Forwarded-For`` is believed, as comma-separated ``ADDRESS[/PREFIX][=SHAPE]`` entries. Unset or
empty trusts nobody: every client behind a forwarder then shares the forwarder's own address (one
rate-limit bucket, one login lockout) which is safe but coarse. It used to default to the whole
loopback (``127.0.0.0/8,::1``), which made every forwarder on the host believable, including the
ones that pass a client-written ``X-Forwarded-For`` through untouched (``ssh -R``, ``socat``,
``frp``, a custom tunnel command, an nginx WebSocket location without the header): through any of
them the client chose its own identity and so its own bucket and lockout.

``SHAPE`` says what the forwarder does to the header, and the hop count is validated against it:

``single`` (default)
    The forwarder REPLACES the header with exactly one hop, the address of the socket it
    accepted (the shipped UI server ``frontend/server.mjs``; nginx with
    ``proxy_set_header X-Forwarded-For $remote_addr``). The header must hold exactly one valid
    address. Anything else (two hops: a forwarder that appended to a client-written value; a
    non-address) means the request did not come through the declared forwarder, so the header is
    NOT believed and the client is counted as the forwarder's own address.
``last``
    The forwarder APPENDS the address it saw to whatever was there (nginx
    ``$proxy_add_x_forwarded_for``, a CDN or tunnel edge such as Cloudflare). Only the LAST hop,
    the one the forwarder wrote, is believed; hops to its left are whatever the client wrote and
    are never read. Trailing hops that are themselves listed forwarders (a chain) are skipped.

An entry names a forwarder HOST: a network wider than one LAN (IPv4 shorter than /24, IPv6 shorter
than /64) is refused, so is ``/0``, an entry with host bits set (``192.168.1.64/24`` is an error,
not a quietly widened ``192.168.1.0/24``), an unknown shape and anything that is not an address.
One bad entry spoils the whole list: :func:`parse_trusted_proxies` raises ``ProxyConfigError``
listing every problem, and the runtime reader (:func:`trusted_proxies`) then logs the problems at
ERROR and trusts NOBODY (never a widened or half-read list) instead of crashing the backend that
carries the E-stop. Every other peer's header is ignored entirely.

Give each trusted forwarder its own source address. All of ``127.0.0.0/8`` is the loopback and any
process on the host can use any address in it, so ``127.0.0.1`` cannot tell the UI server from a
tunnel agent or an ``ssh -R``: the UI server connects to the backend from ``127.0.0.2``
(``BACKEND_SOURCE_ADDRESS``) and nginx uses ``proxy_bind 127.0.0.3``, and only those are listed
(the shipped unit sets ``LAWN_BERRY_TRUSTED_PROXIES=127.0.0.2=single,127.0.0.3=last``).

uvicorn's own proxy-header handling must be OFF (``--no-proxy-headers``, set by the unit): it runs
before this code, believes ``X-Forwarded-For`` from ``FORWARDED_ALLOW_IPS`` (default 127.0.0.1) and
takes any string it finds as the client, so with it on this module would only ever see the
address uvicorn already chose. :func:`log_proxy_posture` says so at start-up when it is on.

Other consequences, all deliberate:

* Which IPv6 addresses share a key. A GLOBAL UNICAST address (inside ``2000::/3`` and
  ``ipaddress``'s ``is_global``, so not the documentation, 6to4 or Teredo blocks) is counted per
  /64 (``ip6:<prefix>/64``): that is what an ISP delegates to one subscriber, so rotating the low
  64 bits does not mint fresh budgets. EVERY OTHER IPv6 address (unique-local ``fc00::/7``,
  link-local, loopback, unspecified, site-local, NAT64, the special-purpose blocks) is counted
  per ADDRESS (``ip:<address>``, zone id dropped): on a LAN every device of a SLAAC network is in
  one /64, and a per-/64 key made one device's five wrong logins lock the login for all of them.
  The price is that a host can alias as many addresses of its own /64 as it likes, each a fresh
  budget, so :func:`ipv6_group_key` names the /64 of EVERY IPv6 client for the one counter that
  must not be rotated around, the login failure lockout (``services/auth_service.py``,
  ``GroupFailureLimiter``: 25 failures lock the whole /64 for 30 s). Residuals, design limits: a
  LAN whose devices all hold GLOBAL addresses from one ISP /64 still shares that /64's budget,
  and a LAN host rotating its own non-global addresses gets a fresh rate-limit bucket per
  address (the tables are bounded, there is no group backstop for them).
* An IPv4-mapped IPv6 address (``::ffff:a.b.c.d``, what a dual-stack socket reports) counts as
  that IPv4 address.

Standard library only.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

TRUSTED_PROXIES_ENV = "LAWN_BERRY_TRUSTED_PROXIES"
# Nobody: an unset variable must never make a forwarder believable.
DEFAULT_TRUSTED_PROXIES = ""
SHAPE_SINGLE = "single"
SHAPE_LAST = "last"
SHAPES = (SHAPE_SINGLE, SHAPE_LAST)
ANONYMOUS_PEER = "anon"
_MAX_NAME_LENGTH = 64
# A forwarder is a host, not a population: IPv4 shorter than /24 and IPv6 shorter than /64 name
# more than one LAN and are refused (every host in the network could write the header).
_WIDEST_PREFIX = {4: 24, 6: 64}
# One log line per misbehaving forwarder per this many seconds (a flood cannot flood the journal).
_WARN_INTERVAL_S = 60.0
_MAX_WARN_KEYS = 64

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class ProxyConfigError(ValueError):
    """``LAWN_BERRY_TRUSTED_PROXIES`` does not say what it means; ``problems`` lists them all."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{TRUSTED_PROXIES_ENV}: " + "; ".join(self.problems))


@dataclass(frozen=True)
class TrustedProxy:
    """One forwarder (a host or a small network) and what it does to ``X-Forwarded-For``."""

    network: IPNetwork
    shape: str = SHAPE_SINGLE

    def __str__(self) -> str:
        return f"{self.network}={self.shape}"


_GLOBAL_UNICAST = ipaddress.IPv6Network("2000::/3")


def _slash64(address: ipaddress.IPv6Address) -> ipaddress.IPv6Address:
    return ipaddress.IPv6Network(f"{address}/64", strict=False).network_address


def _is_global_unicast(address: ipaddress.IPv6Address) -> bool:
    """``True`` for an address an ISP delegates a prefix of to ONE subscriber.

    Inside ``2000::/3`` (the allocated global unicast space) and ``is_global`` (not one of the
    IANA special-purpose blocks that are not globally reachable: documentation ``2001:db8::/32``
    and ``3fff::/20``, Teredo, 6to4, benchmarking, ORCHID...). ``is_global`` alone is not
    enough: it is also true for multicast and for the deprecated site-local ``fec0::/10``.
    """
    return address in _GLOBAL_UNICAST and address.is_global


def _parse_ip(host: str | None) -> IPAddress | None:
    """The IP address of a peer host string (zone id dropped), else ``None``."""
    text = (host or "").strip()
    if not text:
        return None
    try:
        return ipaddress.ip_address(text.split("%", 1)[0])  # drop an IPv6 zone id
    except ValueError:
        return None


def peer_key(host: str | None) -> str:
    """Stable, bounded key for a client host.

    ``None`` or an empty host (no connection information, a unit test calling a service
    directly) maps to :data:`ANONYMOUS_PEER`. Anything that is not an IP address (the ASGI
    test client reports ``"testclient"``, a unix socket reports its path) is lower-cased and
    truncated, never used verbatim as an unbounded dictionary key.

    The address forms (module docstring has the why):

    * IPv4, and IPv4-mapped IPv6 (``::ffff:a.b.c.d``): ``ip:a.b.c.d``;
    * a GLOBAL UNICAST IPv6 address (:func:`_is_global_unicast`): its /64, ``ip6:<prefix>/64``;
    * any other IPv6 address (unique-local, link-local, loopback, ...): itself, ``ip:<address>``.
    """
    text = (host or "").strip()
    if not text:
        return ANONYMOUS_PEER
    address = _parse_ip(text)
    if address is None:
        return f"host:{text.lower()[:_MAX_NAME_LENGTH]}"
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return f"ip:{address.ipv4_mapped}"
        if _is_global_unicast(address):
            return f"ip6:{_slash64(address)}/64"
    return f"ip:{address}"


def ipv6_group_key(host: str | None) -> str | None:
    """The /64 of ANY IPv6 client (``ip6:<prefix>/64``), ``None`` for IPv4, mapped IPv4 and
    anything that is not an address.

    Where :func:`peer_key` splits a LAN's /64 into its devices, this joins them again for the
    counters that must not be dodged by rotating source addresses (the login failure lockout).
    For a global address it is the same /64 :func:`peer_key` already uses.
    """
    address = _parse_ip(host)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is None:
        return f"ip6:{_slash64(address)}/64"
    return None


def _parse_address(text: str) -> IPAddress | None:
    """An IP address from a peer host or a forwarded hop (``a.b.c.d:port``, ``[v6]:port`` ok)."""
    text = text.strip()
    if text.startswith("["):
        end = text.find("]")
        if end == -1:
            return None
        text = text[1:end]
    elif text.count(":") == 1:  # "a.b.c.d:port"
        text = text.split(":", 1)[0]
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def is_loopback_address(host: str | None) -> bool:
    """``True`` for an IP address that is on the loopback (``127.0.0.0/8``, ``::1``, mapped)."""
    address = _parse_address(host or "")
    return address is not None and address.is_loopback


def _parse_entry(entry: str) -> tuple[TrustedProxy | None, str | None]:
    """One ``ADDRESS[/PREFIX][=SHAPE]`` entry as ``(proxy, None)`` or ``(None, problem)``."""
    text = entry.strip()
    network_text, separator, shape = text.rpartition("=")
    if not separator:
        network_text, shape = text, SHAPE_SINGLE
    network_text, shape = network_text.strip(), shape.strip().lower()
    if shape not in SHAPES:
        return None, f"{entry!r} has the unknown shape {shape!r} (use one of: {', '.join(SHAPES)})"
    if "%" in network_text:
        return None, f"{entry!r} has an IPv6 zone id: list the address without it"
    try:
        net = ipaddress.ip_network(network_text, strict=True)
    except ValueError:
        try:
            widened = ipaddress.ip_network(network_text, strict=False)
            host = ipaddress.ip_interface(network_text).ip
        except ValueError:
            return None, f"{entry!r} is not an IP address or CIDR network"
        return None, (
            f"{entry!r} has host bits set, so it would silently become {widened}: "
            f"write {widened} for the whole network or {host} for the one host"
        )
    if net.prefixlen == 0:
        return None, f"{entry!r} matches every address: name the forwarder's own address"
    if net.prefixlen < _WIDEST_PREFIX[net.version]:
        return None, (
            f"{entry!r} names {net.num_addresses} addresses, more than one LAN: a forwarder is "
            "a host, list its own address (the shipped UI server connects from 127.0.0.2, a "
            "repo nginx from 127.0.0.3)"
        )
    return TrustedProxy(net, shape), None


def parse_trusted_proxies(raw: str | None) -> tuple[TrustedProxy, ...]:
    """Parse ``LAWN_BERRY_TRUSTED_PROXIES`` strictly; raise :class:`ProxyConfigError` on any defect.

    ``None``, an empty string and blanks mean "trust nobody". Empty list items (a trailing comma)
    are ignored; every other defect is an error and one bad entry spoils the whole list.
    """
    proxies: list[TrustedProxy] = []
    problems: list[str] = []
    for item in (raw or "").split(","):
        if not item.strip():
            continue
        proxy, problem = _parse_entry(item)
        if problem is not None:
            problems.append(problem)
        elif proxy is not None:
            proxies.append(proxy)
    if problems:
        raise ProxyConfigError(problems)
    return tuple(proxies)


@lru_cache(maxsize=16)
def _read_trusted_proxies(raw: str) -> tuple[TrustedProxy, ...]:
    try:
        return parse_trusted_proxies(raw)
    except ProxyConfigError as exc:  # logged once per distinct value (the result is cached)
        logger.error(
            "%s: refusing the whole list, NO forwarder is trusted until it is fixed "
            "(every client behind a forwarder shares the forwarder's own rate-limit bucket): %s",
            TRUSTED_PROXIES_ENV,
            "; ".join(exc.problems),
        )
        return ()


def trusted_proxies(environ: Mapping[str, str] | None = None) -> tuple[TrustedProxy, ...]:
    """The configured forwarders; nobody when unset, empty or invalid (fail closed, logged)."""
    raw = (os.environ if environ is None else environ).get(TRUSTED_PROXIES_ENV)
    return _read_trusted_proxies(DEFAULT_TRUSTED_PROXIES if raw is None else raw)


def describe_trusted_proxies(proxies: Iterable[TrustedProxy] | None = None) -> str:
    """The resolved list for a log line or a diagnostic (``none`` when nobody is trusted)."""
    resolved = trusted_proxies() if proxies is None else tuple(proxies)
    return ", ".join(str(proxy) for proxy in resolved) or "none"


def _match(address: IPAddress, proxies: Iterable[TrustedProxy]) -> TrustedProxy | None:
    for proxy in proxies:
        if address.version == proxy.network.version and address in proxy.network:
            return proxy
    return None


_last_warned: dict[str, float] = {}


def _warn_limited(key: str, message: str, *args: object) -> None:
    now = time.monotonic()
    if now - _last_warned.get(key, -_WARN_INTERVAL_S) < _WARN_INTERVAL_S:
        return
    if len(_last_warned) >= _MAX_WARN_KEYS:
        _last_warned.pop(next(iter(_last_warned)))
    _last_warned[key] = now
    logger.warning(message, *args)


def _hops(forwarded_for: Iterable[str]) -> list[str]:
    return [hop.strip() for line in forwarded_for for hop in line.split(",") if hop.strip()]


def resolve_client_host(
    peer_host: str | None,
    forwarded_for: Iterable[str],
    proxies: Iterable[TrustedProxy] | None = None,
) -> str | None:
    """The address of the real client, or ``peer_host`` itself when nothing better is known.

    ``forwarded_for`` is every ``X-Forwarded-For`` header line of the request. It is consulted
    only when ``peer_host`` is a configured forwarder, and then according to that forwarder's
    shape (module docstring): ``single`` believes a header that holds exactly one address,
    ``last`` believes the last hop (skipping trailing hops that are themselves forwarders). A
    header that does not fit the declared shape, or a hop that is not an address, is never
    guessed at: the answer is ``peer_host``, the forwarder's own (shared) address.
    """
    nets = trusted_proxies() if proxies is None else tuple(proxies)
    peer = _parse_address(peer_host or "")
    if peer is None:
        return peer_host
    proxy = _match(peer, nets)
    if proxy is None:
        return peer_host
    hops = _hops(forwarded_for)
    if not hops:
        return peer_host  # not through the forwarder (or it sent nothing): its own address
    if proxy.shape == SHAPE_SINGLE:
        address = _parse_address(hops[0]) if len(hops) == 1 else None
        if address is None:
            _warn_limited(
                f"single:{peer}",
                "forwarder %s is declared 'single' (one X-Forwarded-For hop) but sent %d hop(s) "
                "that do not fit: ignoring the header (is a proxy passing a client-written "
                "X-Forwarded-For through?)",
                peer,
                len(hops),
            )
            return peer_host
        return str(address)
    for hop in reversed(hops):
        address = _parse_address(hop)
        if address is None:
            _warn_limited(
                f"last:{peer}",
                "forwarder %s is declared 'last' but its last X-Forwarded-For hop is not an "
                "address: ignoring the header",
                peer,
            )
            return peer_host
        if _match(address, nets) is None:
            return str(address)
    return peer_host


def _peer_host(connection) -> str | None:
    client = getattr(connection, "client", None)
    if client is None:
        return None
    host = client[0] if isinstance(client, (list, tuple)) else getattr(client, "host", None)
    return str(host) if host else None


def _forwarded_for_lines(connection) -> list[str]:
    headers = getattr(connection, "headers", None)
    if headers is None:
        return []
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        return list(getlist("x-forwarded-for"))
    value = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
    return [value] if value else []


def client_host(connection) -> str | None:
    """The client address for a Starlette ``Request`` or ``WebSocket``, proxy-aware (see above)."""
    return resolve_client_host(_peer_host(connection), _forwarded_for_lines(connection))


def request_peer_key(request) -> str:
    """:func:`peer_key` of :func:`client_host` (a Starlette ``Request`` or ``WebSocket``)."""
    return peer_key(client_host(request))


def is_trusted_proxy_peer(connection) -> bool:
    """``True`` when the TCP peer of ``connection`` is a configured forwarder."""
    peer = _parse_address(_peer_host(connection) or "")
    return peer is not None and _match(peer, trusted_proxies()) is not None


def _uvicorn_proxy_headers_may_be_on(argv: Iterable[str]) -> bool:
    args = list(argv)
    return any("uvicorn" in arg for arg in args[:3]) and "--no-proxy-headers" not in args


def log_proxy_posture(
    argv: Iterable[str] | None = None, environ: Mapping[str, str] | None = None
) -> None:
    """Say at start-up who is believed to forward client addresses (and what is not).

    Logs the resolved forwarder list (INFO; ERROR for an invalid value, which trusts nobody) and
    warns when this process looks like a uvicorn launched without ``--no-proxy-headers``: its own
    handling would then decide the client address before this module is consulted.
    """
    env = os.environ if environ is None else environ
    resolved = trusted_proxies(env)
    logger.info(
        "trusted forwarders (%s): %s", TRUSTED_PROXIES_ENV, describe_trusted_proxies(resolved)
    )
    if _uvicorn_proxy_headers_may_be_on(sys.argv if argv is None else argv):
        allow = env.get("FORWARDED_ALLOW_IPS", "127.0.0.1")
        logger.warning(
            "uvicorn's own proxy-header handling looks ON (no --no-proxy-headers; it trusts "
            "FORWARDED_ALLOW_IPS=%s and believes any X-Forwarded-For string those peers send): "
            "it decides the client address before %s is consulted. Start uvicorn with "
            "--no-proxy-headers (the shipped unit does).",
            allow,
            TRUSTED_PROXIES_ENV,
        )
