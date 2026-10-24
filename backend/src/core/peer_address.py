"""The key every rate limit and login lockout is counted against: the client's address.

Login lockout and the request limiters used to be keyed on the client-supplied ``X-Client-Id``
header: sending a fresh value with each request gave an unlimited number of password guesses and
an unbounded number of limiter entries. The only inputs that are not the caller's to choose are
what the kernel reports for the connection (``request.client.host``) and, behind a trusted
reverse proxy, the address that proxy recorded.

Trusted proxies (``LAWN_BERRY_TRUSTED_PROXIES``, comma-separated addresses or networks, default
``127.0.0.0/8,::1``; an explicitly empty value trusts none): when the TCP peer is one of them,
the client is the LAST ``X-Forwarded-For`` hop that is not itself a trusted proxy, i.e. the
address the proxy appended from its own socket. Everything to the left of that hop is whatever the
client wrote and is never read. A last hop that is not an address falls back to the proxy's own
address (the shared bucket) instead of being guessed. From any peer that is NOT a trusted proxy
the header is ignored entirely, so it cannot be used to pick a bucket or to claim a loopback
identity. The shipped web UI server (``frontend/server.mjs``) is such a proxy: it must delete
inbound ``X-Forwarded-For`` / ``Forwarded`` / ``X-Real-IP`` and set ``xfwd: true`` so the last
hop is the browser's real socket address. Without that header every browser behind it shares the
proxy's one budget, and a lockout caused by one applies to all of them.

uvicorn applies the same rule first: its ProxyHeadersMiddleware (``--proxy-headers``, on by
default, trusting only ``FORWARDED_ALLOW_IPS``, default 127.0.0.1) replaces ``request.client``
with the last untrusted ``X-Forwarded-For`` hop before any of this code runs, so in the shipped
deployment the "TCP peer" seen here is already the browser's address. Never run with
``FORWARDED_ALLOW_IPS=*``: uvicorn then takes the FIRST hop, which the client wrote.

Other consequences, all deliberate:

* An IPv6 client is counted per /64, the smallest prefix a single subscriber normally owns, so
  rotating the low 64 bits does not mint fresh budgets.
* An IPv4-mapped IPv6 address (``::ffff:a.b.c.d``, what a dual-stack socket reports) counts as
  that IPv4 address.

Standard library only.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Iterable, Mapping
from functools import lru_cache

logger = logging.getLogger(__name__)

TRUSTED_PROXIES_ENV = "LAWN_BERRY_TRUSTED_PROXIES"
DEFAULT_TRUSTED_PROXIES = "127.0.0.0/8,::1"
ANONYMOUS_PEER = "anon"
_MAX_NAME_LENGTH = 64

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def peer_key(host: str | None) -> str:
    """Stable, bounded key for a client host.

    ``None`` or an empty host (no connection information, a unit test calling a service
    directly) maps to :data:`ANONYMOUS_PEER`. Anything that is not an IP address (the ASGI
    test client reports ``"testclient"``, a unix socket reports its path) is lower-cased and
    truncated, never used verbatim as an unbounded dictionary key.
    """
    text = (host or "").strip()
    if not text:
        return ANONYMOUS_PEER
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])  # drop an IPv6 zone id
    except ValueError:
        return f"host:{text.lower()[:_MAX_NAME_LENGTH]}"
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return f"ip:{address.ipv4_mapped}"
        network = ipaddress.ip_network(f"{address}/64", strict=False)
        return f"ip6:{network.network_address}/64"
    return f"ip:{address}"


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


@lru_cache(maxsize=16)
def _parse_networks(raw: str) -> tuple[IPNetwork, ...]:
    networks: list[IPNetwork] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            logger.warning("%s: ignoring %r (not an address or network)", TRUSTED_PROXIES_ENV, item)
    return tuple(networks)


def trusted_proxy_networks(environ: Mapping[str, str] | None = None) -> tuple[IPNetwork, ...]:
    """The networks whose ``X-Forwarded-For`` is believed (default: the loopback)."""
    raw = (os.environ if environ is None else environ).get(TRUSTED_PROXIES_ENV)
    return _parse_networks(DEFAULT_TRUSTED_PROXIES if raw is None else raw)


def _trusted(address: IPAddress, networks: Iterable[IPNetwork]) -> bool:
    return any(address in network for network in networks)


def resolve_client_host(
    peer_host: str | None,
    forwarded_for: Iterable[str],
    networks: Iterable[IPNetwork] | None = None,
) -> str | None:
    """The address of the real client, or ``peer_host`` itself when nothing better is known.

    ``forwarded_for`` is every ``X-Forwarded-For`` header line of the request. It is consulted
    only when ``peer_host`` is a trusted proxy, and then only from the right: hops that are
    themselves trusted proxies are skipped and the first other hop is the client. A hop that is
    not an address ends the search at the proxy's own address.
    """
    nets = tuple(trusted_proxy_networks() if networks is None else networks)
    peer = _parse_address(peer_host or "")
    if peer is None or not _trusted(peer, nets):
        return peer_host
    hops = [hop.strip() for line in forwarded_for for hop in line.split(",") if hop.strip()]
    for hop in reversed(hops):
        address = _parse_address(hop)
        if address is None:
            return peer_host
        if not _trusted(address, nets):
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
