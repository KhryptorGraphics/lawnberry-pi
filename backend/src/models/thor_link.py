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
refuses the public simulation key anywhere but in simulation.

The tag is also the only thing a UDP source address cannot fake, and checking it costs an HMAC, so
what a flood may make the Pi verify is bounded twice. ``session_filter`` and ``claimed_ack`` read
the cleartext ``sid`` and ``ack`` of a datagram by byte search (never by parsing it), which lets
the Pi discard anything that cannot answer one of its own recent uplinks before it spends an
HMAC. ``VerificationBudget`` then caps the HMAC checks per second for the whole port and keeps
what junk does to the budget of a *source address*, which a spoofer chooses, apart from what the
authenticated peer is allowed to send. ``DatagramBudget`` is the plain per-source and per-port
token bucket the Thor-side client puts in front of its own HMAC.
"""

from __future__ import annotations

import functools
import hmac
import json
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable
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

    The value is never echoed in the error, nor is the reason ``bytes.fromhex`` gives (it names a
    position inside the key), and the error is raised outside the ``except`` so that nothing chains
    to it. Generate one with ``openssl rand -hex 32``. The published sim key is refused unless
    ``simulation`` (SIM_MODE) is set by the caller.
    """
    try:
        raw: bytes | None = bytes.fromhex(text.strip())
    except ValueError:
        raw = None
    if raw is None:
        raise LinkKeyError("link key must be hex-encoded")
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


# JSON allows whitespace around the colon. ``encode`` emits none, but a prefilter that insisted on
# the encoder's exact bytes would turn a Thor that formats its JSON differently into a link that
# never comes up; the HMAC, not the format, decides what is authentic.
_WS = rb"[ \t\r\n]*"
_ACK_FIELD = re.compile(rb'"ack"' + _WS + rb":" + _WS + rb"([0-9]+)")
# An ack with more digits than ``MAX_COUNTER`` has is out of range whatever its digits are, and
# is not converted: ``int`` on a datagram-sized digit string is not free.
_MAX_ACK_DIGITS = len(str(MAX_COUNTER))


def session_filter(session_id: int) -> Callable[[bytes], bool]:
    """A predicate: does a datagram's body carry ``"sid":<session_id>``?

    This is the cheapest test of whether a datagram can be a downlink for this run, and it costs
    no HMAC. Every genuine downlink echoes the sid of the uplink it answers, in cleartext, as a
    JSON member of its body (the bytes after the tag). The member is found by byte search, not by
    parsing, so unauthenticated bytes still never reach the JSON layer: it does not depend on the
    order of the members or on whitespace around the colon, and the number must end where the
    sid does (``"sid":12`` is not sid 1). A match proves nothing (anyone who can read an uplink
    knows the sid); a miss proves the datagram is not an authentic downlink for this run, so
    the Pi may drop it without verifying anything.

    A Thor that writes its JSON with the sid as a string or a float (``"123"``, ``1.2e2``)
    would be treated as another run's: ``Downlink`` would accept them, ``encode`` never emits them.
    """
    if not 0 <= session_id <= MAX_SESSION_ID:
        raise ValueError(f"session_id must be in 0..{MAX_SESSION_ID}")
    search = re.compile(
        rb'"sid"' + _WS + rb":" + _WS + str(session_id).encode("ascii") + rb"(?![0-9])"
    ).search

    def carries_session(data: bytes) -> bool:
        return search(data, TAG_BYTES) is not None

    return carries_session


def claimed_ack(data: bytes) -> int | None:
    """The ``ack`` a datagram's body claims, read as ``session_filter`` reads the sid.

    ``None`` if the body has no readable ``"ack":<digits>`` member (a downlink always has one), and
    ``MAX_COUNTER + 1`` for a number with too many digits to be a counter. Only the first such
    member counts, so a datagram stuffed with ``"ack"`` members costs one match, not a hundred.
    The result says what the datagram *claims*, which an attacker chooses: it can only be used to
    refuse work (an ack that no outstanding uplink can have), never to accept anything.
    """
    match = _ACK_FIELD.search(data, TAG_BYTES)
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) > _MAX_ACK_DIGITS:
        return MAX_COUNTER + 1
    return int(digits)


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


def _refill(bucket: list[float], rate: float, burst: float, now: float) -> float:
    """Top ``bucket`` (``[tokens, stamp]``) up to ``now`` and return its tokens.

    A clock that steps backwards neither mints tokens nor moves the stamp back, so stepping forward
    again later cannot mint them either.
    """
    if now > bucket[1]:
        bucket[0] = min(burst, bucket[0] + (now - bucket[1]) * rate)
        bucket[1] = now
    return bucket[0]


class VerificationBudget:
    """What the Pi may spend on HMAC checks, so that junk cannot starve the peer it keeps out.

    The source address of a UDP datagram is whatever its sender writes into it. A budget keyed only
    on that address lets anyone who knows the Thor's address spend the Thor's share: 400 forged
    datagrams a second took nine of the Thor's ten answers a second from the old per-source bucket,
    though the forger never held the key. Nothing in the datagram tells forged from genuine before
    the HMAC, so the limit on what a flood can cost has to be global, and the share of the peer has
    to be one that junk cannot reach. Three layers, each taking ``now`` so a fake clock makes them
    deterministic:

    * **the verification ceiling** (``verify_hz``, ``verify_burst``): at most this many HMAC checks
      per second for the whole port, whoever sends. This is the bound on CPU; size it from the
      measured cost of one check (``admit_verification``).
    * **peers and strangers**: a *peer* is a source that has delivered an accepted datagram, or is
      named when the budget is built (the Thor the Pi is configured for). Any other source is a
      *stranger*. A stranger also draws on a ``DatagramBudget`` of its own (a bucket per source
      and one shared by all strangers), so however many addresses a flood rotates through, strangers
      can never take more than ``strangers_hz`` of the ceiling. A peer draws on the ceiling alone: a
      flood from strangers cannot starve it, and a flood that spoofs its address cannot either, for
      as long as the whole flood stays inside the ceiling.
    * **the authentic bucket of each peer** (``peer_hz``, ``peer_burst``, ``admit_authentic``):
      taken only by datagrams that passed the HMAC. A flood of junk never gets that far, so it
      cannot drain it, and a peer that holds the key but sends faster than it should (a
      compromised or broken Thor) is still bounded in the work its datagrams cause afterwards.

    Beyond the ceiling nothing here can tell the peer's genuine datagrams from a forger's, so they
    compete for the same tokens and some genuine ones are shed: a forger that writes the peer's
    address, the run's sid and an ack the Pi still waits for can do that at any rate above
    ``verify_hz``. Only keeping such datagrams off the link (ingress filtering, link-layer
    authentication) removes that; this class bounds what they cost and keeps everything else out.

    Memory is bounded: ``max_strangers`` stranger buckets (the least recently used evicted first)
    and ``max_peers`` peers, however many sources send.
    """

    def __init__(
        self,
        *,
        verify_hz: float,
        verify_burst: int,
        peer_hz: float,
        peer_burst: int,
        stranger_hz: float,
        stranger_burst: int,
        strangers_hz: float,
        strangers_burst: int,
        max_strangers: int = 64,
        max_peers: int = 8,
        peers: Iterable[str] = (),
        now: float = 0.0,
    ) -> None:
        rates = (
            ("verify_hz", verify_hz),
            ("peer_hz", peer_hz),
            ("stranger_hz", stranger_hz),
            ("strangers_hz", strangers_hz),
        )
        for name, value in rates:
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0")
        bursts = (verify_burst, peer_burst, stranger_burst, strangers_burst, max_strangers)
        if min(bursts) < 1 or max_peers < 1:
            raise ValueError("every burst, max_strangers and max_peers must be >= 1")
        self._verify_hz = float(verify_hz)
        self._verify_burst = float(verify_burst)
        self._peer_hz = float(peer_hz)
        self._peer_burst = float(peer_burst)
        self._max_peers = max_peers
        self._verify = [self._verify_burst, now]
        self._strangers = DatagramBudget(
            stranger_hz,
            stranger_burst,
            total_rate_hz=strangers_hz,
            total_burst=strangers_burst,
            max_sources=max_strangers,
        )
        # peer -> [authentic tokens, stamp]; the configured ones start with a full bucket
        self._peers: OrderedDict[str, list[float]] = OrderedDict()
        for peer in peers:
            self._peers[peer] = [self._peer_burst, now]
            if len(self._peers) > max_peers:
                self._peers.popitem(last=False)

    def __len__(self) -> int:
        """Number of stranger buckets currently kept (never more than ``max_strangers``)."""
        return len(self._strangers)

    @property
    def peers(self) -> tuple[str, ...]:
        """The peers, least recently trusted first (never more than ``max_peers``)."""
        return tuple(self._peers)

    def verify_tokens(self, now: float) -> float:
        """HMAC checks that could start at ``now`` (the ceiling's bucket, refilled, not charged)."""
        return _refill(self._verify, self._verify_hz, self._verify_burst, now)

    def admit_verification(self, source: str, now: float) -> bool:
        """May one datagram from ``source`` be HMAC-checked at ``now``? True spends the token.

        It needs a token from the ceiling and, from a stranger, one from its own bucket and one
        from the strangers' shared bucket as well. Nothing is spent unless all of them are there,
        so a refusal costs the source nothing.
        """
        if _refill(self._verify, self._verify_hz, self._verify_burst, now) < 1.0:
            return False
        if source not in self._peers and not self._strangers.admit(source, now):
            return False
        self._verify[0] -= 1.0
        return True

    def admit_authentic(self, source: str, now: float) -> bool:
        """A datagram from ``source`` passed the HMAC: may the Pi act on it? True spends the token.

        Only a peer has an authentic bucket. A stranger's genuine datagram has already been paid
        for at its own bucket, and it becomes a peer (``trust``) once one is accepted.
        """
        bucket = self._peers.get(source)
        if bucket is None:
            return True
        if _refill(bucket, self._peer_hz, self._peer_burst, now) < 1.0:
            return False
        bucket[0] -= 1.0
        return True

    def trust(self, source: str, now: float) -> None:
        """``source`` delivered a datagram the Pi accepted: it is a peer from now on."""
        if source in self._peers:
            self._peers.move_to_end(source)
            return
        # Its accepted datagram counts against the authentic rate, like any later one.
        self._peers[source] = [self._peer_burst - 1.0, now]
        while len(self._peers) > self._max_peers:
            self._peers.popitem(last=False)
