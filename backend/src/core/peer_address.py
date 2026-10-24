"""The key every rate limit and login lockout is counted against: the TCP peer address.

Login lockout and the request limiters used to be keyed on the client-supplied ``X-Client-Id``
header (then ``X-Request-ID``, ``X-Correlation-ID``): sending a fresh value with each request
gave an unlimited number of password guesses and an unbounded number of limiter entries. Only
what the kernel reports for the connection (``request.client.host``) is not chosen by the
caller, so that is the one input here. No ``Forwarded`` / ``X-Forwarded-For`` header is
consulted, on purpose: a header the caller can write is as forgeable as ``X-Client-Id``.

Consequences, all deliberate:

* Behind a reverse proxy on the same host (the shipped web UI server proxies to the backend on
  127.0.0.1, so does a local ``cloudflared``) every client of that proxy shares ONE key. The
  shared budget is the price of not trusting a header; tune ``GLOBAL_RATE_LIMIT_*`` if the
  default is too tight for several browsers at once.
* An IPv6 client is counted per /64, the smallest prefix a single subscriber normally owns,
  so rotating the low 64 bits does not mint fresh budgets.
* An IPv4-mapped IPv6 peer (``::ffff:a.b.c.d``, what a dual-stack socket reports) counts as
  that IPv4 address.

Standard library only.
"""

from __future__ import annotations

import ipaddress

ANONYMOUS_PEER = "anon"
_MAX_NAME_LENGTH = 64


def peer_key(host: str | None) -> str:
    """Stable, bounded key for a peer host as reported by the transport.

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


def request_peer_key(request) -> str:
    """:func:`peer_key` of ``request.client`` (a Starlette ``Request`` or ``WebSocket``)."""
    client = getattr(request, "client", None)
    host = None
    if client is not None:
        host = client[0] if isinstance(client, (list, tuple)) else getattr(client, "host", None)
    return peer_key(str(host) if host else None)
