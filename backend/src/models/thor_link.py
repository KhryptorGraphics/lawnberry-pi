"""Pi <-> Thor link protocol (W7), version 2 (authenticated).

Wire format: one UDP datagram per message: a 32-byte HMAC-SHA256 tag, then
compact JSON with short keys. The tag covers the exact JSON bytes plus a
direction label, so an uplink can never be replayed as a downlink and no
re-serialisation is needed to verify it. Raw imagery never crosses the link;
only distilled state goes up and waypoints/corrections come down. Every
encoded message (tag included) is checked against ``MAX_DATAGRAM_BYTES`` so
the stream stays inside the Wi-Fi HaLow budget, measured at one site so far
(see ``docs/tractor-platform.md`` "Pi <-> Thor link" and
``docs/autonomy-toolchain-matrix.md``).

Uplink (Pi -> Thor): position, distilled detections, health. Sent at
``UPLINK_HZ``; it doubles as the Pi's heartbeat and carries ``sid`` (a random
id the Pi picks per service run) and ``seq`` (its own counter). Because it is
the heartbeat, an uplink is never refused for its size: coordinates are
quantised on the wire and ``fit_uplink`` thins the detections (clip class
names, then drop the least confident ones and say how many) until it fits.
Downlink (Thor -> Pi): waypoints and corrections. Each answers exactly one
uplink by echoing its ``sid`` and ``seq`` as ``ack``; the Pi accepts a
downlink only if it answers a recent uplink of this run that no earlier
accepted downlink answered. That makes replay, reordering, staleness and a
Thor restart all decidable from the Pi's own counters and clock, with no
Thor state and no clock sync. Silence longer than the link timeout puts the
Pi in safe hold.

Anything that fails the tag check is dropped before it is parsed, so
unauthenticated bytes never reach the JSON/pydantic layer. There is no
encryption: payloads (position, detections, waypoints) are authenticated,
not secret. Protocol v1 (unauthenticated) was never wired into ``main.py``, so v2
replaces it outright with no compatibility path.

The key is the only real control, so it must look like ``openssl rand -hex 32``:
``validate_key`` refuses short keys and keys that are a single repeated byte, a short
repeating pattern or built from a handful of byte values, and ``refuse_published_key``
refuses the public simulation key anywhere but in simulation. ``DatagramBudget`` bounds
how many received datagrams per second get any real work done, so a flood of junk costs a
dictionary lookup per datagram instead of an HMAC.
"""

from __future__ import annotations

import functools
import hmac
import json
import math
from collections import OrderedDict
from hashlib import sha256
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

PROTOCOL_VERSION = 2
# One datagram (tag + JSON) must fit an unfragmented Ethernet/HaLow MTU payload.
MAX_DATAGRAM_BYTES = 1200
UPLINK_HZ = 10.0
# Ceiling on the uplink rate: 50 Hz of worst-case 1,200 B uplinks is 60 kB/s, about a tenth of
# the one HaLow throughput measurement so far. It guards against a typo (or a bad caller)
# turning the slow loop into a busy loop that floods the link; it is not a recommendation.
MAX_UPLINK_HZ = 50.0
MAX_DETECTIONS = 16
# ``fit_uplink`` never clips a class name below this many characters: past that a name no
# longer says what was seen, so whole detections are dropped instead.
MIN_CLASS_CHARS = 8
# Plans travel in windows of this many waypoints (``Downlink.start_index``) so a
# worst-case window still fits one datagram.
MAX_WAYPOINTS = 16
# Plan indices are bounded: no yard plan has 100k legs, and a bound keeps the
# worst-case datagram size computable.
MAX_PLAN_INDEX = 99_999
# Counters stay inside a JSON-safe integer (2**53 - 1) and session ids inside 48
# bits, so no peer can make the Pi compare or log arbitrarily large integers.
MAX_COUNTER = 2**53 - 1
MAX_SESSION_ID = 2**48 - 1

TAG_BYTES = sha256().digest_size  # 32
KEY_MIN_BYTES = 32
# A key made of a unit shorter than this repeated (``passwordpassword...``) holds only that many
# bytes of secret, whatever its length. 16 bytes is 128 bits, the least a link key may carry.
KEY_MIN_PERIOD_BYTES = 16
# ``openssl rand -hex 32`` gives ~30 distinct byte values in 32 bytes; fewer than this has odds
# below 1 in 10**13 for a random key and is a typed pattern, not a key.
KEY_MIN_DISTINCT_BYTES = 16
# Public, deliberately insecure key for simulation and tests ONLY. The service
# refuses it unless SIM_MODE is on; never provision it on a real mower.
SIM_LINK_KEY = b"lawnberry-thor-link/SIM-ONLY/public-key/never-on-hardware"
# Every key that is published in this repository and so authenticates nothing.
PUBLISHED_KEYS = (SIM_LINK_KEY,)

_UPLINK_DOMAIN = b"lawnberry.thor-link.v2.uplink\x00"
_DOWNLINK_DOMAIN = b"lawnberry.thor-link.v2.downlink\x00"

_M = TypeVar("_M", bound=BaseModel)


class Detection(BaseModel):
    """One distilled obstacle detection relative to the mower.

    Serialised with one-letter keys and rounded floats so that
    ``MAX_DETECTIONS`` worst-case detections still fit one datagram.
    """

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    cls: str = Field(min_length=1, max_length=24, alias="c")
    bearing_deg: float = Field(ge=-180.0, le=180.0, alias="b")
    range_m: float = Field(ge=0.0, le=200.0, alias="r")
    conf: float = Field(ge=0.0, le=1.0, alias="p")

    @field_serializer("bearing_deg", "range_m")
    def _deci(self, v: float) -> float:
        return round(v, 1)

    @field_serializer("conf")
    def _centi(self, v: float) -> float:
        return round(v, 2)


class Uplink(BaseModel):
    """The Pi's state for the Thor. On the wire ``lat``/``lon`` are quantised to 1e-7 deg
    (~1.1 cm, as ``Waypoint``) and ``heading_deg`` to 0.1 deg: full-precision doubles are ~18
    characters each, enough to push 16 detections past one datagram.
    """

    v: Literal[2] = PROTOCOL_VERSION
    # Stamped by ThorLinkService (0 = not yet stamped): the session id and
    # counter the Thor must echo back as ``Downlink.sid`` / ``Downlink.ack``.
    sid: int = Field(default=0, ge=0, le=MAX_SESSION_ID)
    seq: int = Field(ge=0, le=MAX_COUNTER)
    t_ms: int = Field(ge=0, le=MAX_COUNTER)  # Pi monotonic clock, ms
    lat: float | None = None
    lon: float | None = None
    fix: str = Field(default="none", max_length=16)  # none|gps|dgps|rtk_float|rtk_fixed
    heading_deg: float | None = None
    detections: list[Detection] = Field(default_factory=list, max_length=MAX_DETECTIONS)
    # How many detections the Pi had to leave out so this uplink fits one datagram
    # (``fit_uplink``). Absent when the list is complete, so the Thor can tell "no obstacle
    # reported" from "obstacles reported that did not fit".
    detections_dropped: int | None = Field(default=None, ge=1, le=MAX_COUNTER)
    estop: bool = False
    safe_hold: bool = False
    # True while the Pi is held and could not confirm both levers neutral and
    # the blade off (see ThorLinkService.hold_fault).
    hold_fault: bool = False
    blade: bool = False
    waypoint_index: int | None = Field(default=None, ge=0, le=MAX_COUNTER)

    @field_serializer("lat", "lon")
    def _quantise_position(self, v: float | None) -> float | None:
        return None if v is None else round(v, 7)

    @field_serializer("heading_deg")
    def _deci_heading(self, v: float | None) -> float | None:
        return None if v is None else round(v, 1)


class Waypoint(BaseModel):
    """One plan point. On the wire lat/lon are quantised to 1e-7 deg (~1.1 cm).

    Full-precision doubles are ~18 characters each; 16 of them plus the tag
    overflow one datagram at high latitudes/longitudes, so quantising makes the
    largest legal window fit by construction (RTK-fixed is ~1-2 cm anyway).
    """

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    blade: bool = False

    @field_serializer("lat", "lon")
    def _quantise(self, v: float) -> float:
        return round(v, 7)


class Downlink(BaseModel):
    v: Literal[2] = PROTOCOL_VERSION
    # Echo of the answered uplink's ``sid`` / ``seq``. The Thor keeps no
    # sequence state of its own: after a restart it just answers the next
    # uplink it hears.
    sid: int = Field(ge=0, le=MAX_SESSION_ID)
    ack: int = Field(ge=0, le=MAX_COUNTER)
    plan_id: str = Field(min_length=1, max_length=32)
    start_index: int = Field(default=0, ge=0, le=MAX_PLAN_INDEX)
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=MAX_WAYPOINTS)
    pause: bool = False


class LinkBudgetError(ValueError):
    """A message or datagram exceeds the per-datagram byte budget."""


class LinkAuthError(ValueError):
    """A datagram is too short to be framed or its authentication tag is wrong."""


class LinkKeyError(ValueError):
    """The shared link key is missing, malformed or too weak to use."""


def _smallest_period(data: bytes) -> int:
    """Length of the shortest unit ``data`` is a repetition of (``len(data)`` if there is none)."""
    n = len(data)
    fail = [0] * n  # KMP failure function: longest proper prefix that is also a suffix
    k = 0
    for i in range(1, n):
        while k and data[i] != data[k]:
            k = fail[k - 1]
        if data[i] == data[k]:
            k += 1
        fail[i] = k
    return n - fail[-1]


@functools.lru_cache(maxsize=16)
def _weakness(key: bytes) -> str | None:
    """Why ``key`` cannot be a secret, or None. Cached: ``encode`` and ``decode_*`` revalidate the
    key on every datagram, so a flood must not pay for the analysis each time. Never contains key
    material."""
    if not any(key):
        return "link key is all zero bytes"
    if len(set(key)) == 1:
        return "link key is one byte repeated"
    period = _smallest_period(key)
    if period < KEY_MIN_PERIOD_BYTES:
        return f"link key is a {period}-byte pattern repeated, holding {period} byte(s) of secret"
    distinct = len(set(key))
    if distinct < KEY_MIN_DISTINCT_BYTES:
        return f"link key uses only {distinct} distinct byte values: it is not random"
    return None


def validate_key(key: bytes) -> bytes:
    """Return ``key`` as bytes, or raise ``LinkKeyError`` if it is too short or not random-looking.

    Refused with an explicit reason (never echoing the key): fewer than ``KEY_MIN_BYTES`` bytes,
    all zero, one byte repeated, a repeating unit shorter than ``KEY_MIN_PERIOD_BYTES`` bytes, or
    fewer than ``KEY_MIN_DISTINCT_BYTES`` distinct byte values. A key from ``openssl rand -hex 32``
    passes; a typed or placeholder key does not.
    """
    if not isinstance(key, bytes | bytearray) or len(key) < KEY_MIN_BYTES:
        raise LinkKeyError(f"link key must be at least {KEY_MIN_BYTES} bytes")
    key = bytes(key)
    weakness = _weakness(key)
    if weakness is not None:
        raise LinkKeyError(weakness)
    return key


def refuse_published_key(key: bytes, *, simulation: bool) -> bytes:
    """Return ``key``, or raise ``LinkKeyError`` if it is a published key and not ``simulation``.

    A published key is in the repository, so using it would let anyone steer the mower. Callers
    pass whether SIM_MODE is on; ``validate_key`` cannot know, and ``encode``/``decode_*`` must
    keep working with the sim key under simulation.
    """
    if not simulation and key in PUBLISHED_KEYS:
        raise LinkKeyError("SIM_LINK_KEY is public: it is only accepted with SIM_MODE=1")
    return key


def parse_key(text: str, *, simulation: bool = False) -> bytes:
    """Decode a hex-encoded key as stored in the environment or secrets file.

    The value is never echoed in the error. Generate one with ``openssl rand -hex 32``. The
    published sim key is refused unless ``simulation`` (SIM_MODE) is set by the caller.
    """
    try:
        raw = bytes.fromhex(text.strip())
    except ValueError:
        raise LinkKeyError("link key must be hex-encoded") from None
    return refuse_published_key(validate_key(raw), simulation=simulation)


def _tag(key: bytes, domain: bytes, body: bytes) -> bytes:
    return hmac.new(key, domain + body, sha256).digest()


def _body(message: Uplink | Downlink) -> bytes:
    payload = message.model_dump(exclude_none=True, by_alias=True)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _wire_size(message: Uplink | Downlink) -> int:
    return TAG_BYTES + len(_body(message))


def encode(message: Uplink | Downlink, key: bytes) -> bytes:
    """Serialise ``message`` as ``HMAC-SHA256 tag || compact JSON`` (<= MAX_DATAGRAM_BYTES)."""
    key = validate_key(key)
    domain = _UPLINK_DOMAIN if isinstance(message, Uplink) else _DOWNLINK_DOMAIN
    body = _body(message)
    data = _tag(key, domain, body) + body
    if len(data) > MAX_DATAGRAM_BYTES:
        raise LinkBudgetError(f"{type(message).__name__} is {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    return data


def fit_uplink(message: Uplink, *, budget: int = MAX_DATAGRAM_BYTES) -> Uplink:
    """Return ``message`` if it fits ``budget`` bytes on the wire, else the least-degraded copy.

    An uplink is the Pi's heartbeat: one that is refused for its size means no heartbeat, the
    Thor cannot answer and the mower holds. So it is thinned instead, in this order and each step
    only if the one before was not enough:

    1. coordinates and heading are already quantised by the serialiser;
    2. class names are clipped to the longest length that fits, never below ``MIN_CLASS_CHARS``
       (an obstacle with a clipped name is still an obstacle);
    3. the least confident detections are dropped (the farthest first among equals) and the number
       dropped is added to ``detections_dropped``.

    Raises ``LinkBudgetError`` only when it does not fit with no detections at all: then the
    detections are not the problem and no thinning of them can help.
    """

    def fits(candidate: Uplink) -> bool:
        return _wire_size(candidate) <= budget

    if fits(message):
        return message
    detections = message.detections

    def clipped(cap: int) -> list[Detection]:
        return [
            d if len(d.cls) <= cap else d.model_copy(update={"cls": d.cls[:cap]})
            for d in detections
        ]

    # Wire size never shrinks as the cap grows, so bisect for the largest cap that fits.
    lo, hi = MIN_CLASS_CHARS, max((len(d.cls) for d in detections), default=0) - 1
    best: Uplink | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = message.model_copy(update={"detections": clipped(mid)})
        if fits(trial):
            best, lo = trial, mid + 1
        else:
            hi = mid - 1
    if best is not None:
        return best

    shortened = clipped(MIN_CLASS_CHARS)
    least_useful_first = sorted(
        range(len(shortened)), key=lambda i: (shortened[i].conf, -shortened[i].range_m)
    )
    already = message.detections_dropped or 0
    for count in range(1, len(shortened) + 1):
        gone = set(least_useful_first[:count])
        trial = message.model_copy(
            update={
                "detections": [d for i, d in enumerate(shortened) if i not in gone],
                "detections_dropped": already + count,
            }
        )
        if fits(trial):
            return trial
    bare = message.model_copy(update={"detections": []})
    raise LinkBudgetError(
        f"Uplink is {_wire_size(bare)} B > {budget} B even with no detections: "
        "something other than the detections is too large"
    )


def _decode(data: bytes, model: type[_M], domain: bytes, key: bytes) -> _M:
    key = validate_key(key)
    # Size first, tag second, parse last: oversized or forged bytes cost one
    # length check or one HMAC and never reach the JSON parser.
    if len(data) > MAX_DATAGRAM_BYTES:
        raise LinkBudgetError(f"datagram {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    if len(data) <= TAG_BYTES:
        raise LinkAuthError("datagram too short to hold an authentication tag and a body")
    tag, body = data[:TAG_BYTES], data[TAG_BYTES:]
    if not hmac.compare_digest(tag, _tag(key, domain, body)):
        raise LinkAuthError("authentication tag mismatch")
    return model.model_validate_json(body)


def decode_uplink(data: bytes, key: bytes) -> Uplink:
    """Verify and parse an uplink (what the Thor does with Pi datagrams)."""
    return _decode(data, Uplink, _UPLINK_DOMAIN, key)


def decode_downlink(data: bytes, key: bytes) -> Downlink:
    """Verify and parse a downlink (what the Pi does with Thor datagrams)."""
    return _decode(data, Downlink, _DOWNLINK_DOMAIN, key)


def worst_case_link_bytes_per_second() -> float:
    """Upper bound on uplink bytes/s at ``UPLINK_HZ``; compare with measured HaLow throughput."""
    return MAX_DATAGRAM_BYTES * UPLINK_HZ


class DatagramBudget:
    """Token buckets that bound how many received datagrams per second get any real work done.

    A datagram is admitted only if its source has a token in its own bucket and the whole port
    has one in the total bucket; either empty means it is dropped before the HMAC, so a flood
    (or a source that rotates addresses) costs a few dictionary operations per datagram, not a
    hash and a log line. ``exempt_total`` lets one known-good source skip the total bucket
    (still bounded by its own), so a flood from strangers cannot starve the peer already
    authenticated. Buckets refill from the ``now`` the caller passes, so a fake clock makes it
    deterministic. Memory is bounded: at most ``max_sources`` per-source buckets are kept, the least
    recently used one evicted first.

    ``rate_hz`` / ``burst`` size each source's bucket, ``total_rate_hz`` / ``total_burst`` the
    shared one. The legitimate peer sends one datagram per uplink, so a rate of twice the
    uplink rate leaves room for a burst of queued answers after a radio stall.
    """

    def __init__(
        self,
        rate_hz: float,
        burst: int,
        *,
        total_rate_hz: float,
        total_burst: int,
        max_sources: int = 64,
    ) -> None:
        for name, value in (("rate_hz", rate_hz), ("total_rate_hz", total_rate_hz)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0")
        if burst < 1 or total_burst < 1 or max_sources < 1:
            raise ValueError("burst, total_burst and max_sources must be >= 1")
        self._rate = float(rate_hz)
        self._burst = float(burst)
        self._total_rate = float(total_rate_hz)
        self._total_burst = float(total_burst)
        self._max_sources = max_sources
        self._sources: OrderedDict[str, list[float]] = OrderedDict()  # source -> [tokens, stamp]
        self._total: list[float] | None = None

    def __len__(self) -> int:
        """Number of per-source buckets currently kept (never more than ``max_sources``)."""
        return len(self._sources)

    def admit(self, source: str, now: float, *, exempt_total: bool = False) -> bool:
        """True if one more datagram from ``source`` may be processed at time ``now``."""
        bucket = self._sources.get(source)
        if bucket is None:
            bucket = [self._burst, now]
            self._sources[source] = bucket
            if len(self._sources) > self._max_sources:
                self._sources.popitem(last=False)
        else:
            self._sources.move_to_end(source)
            bucket[0] = min(self._burst, bucket[0] + max(0.0, now - bucket[1]) * self._rate)
            bucket[1] = now
        if bucket[0] < 1.0:
            return False
        if not exempt_total:
            total = self._total
            if total is None:
                total = self._total = [self._total_burst, now]
            else:
                refill = max(0.0, now - total[1]) * self._total_rate
                total[0] = min(self._total_burst, total[0] + refill)
                total[1] = now
            if total[0] < 1.0:
                return False
            total[0] -= 1.0
        bucket[0] -= 1.0
        return True
