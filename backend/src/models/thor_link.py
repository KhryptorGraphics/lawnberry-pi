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
id the Pi picks per service run) and ``seq`` (its own counter).
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
"""

from __future__ import annotations

import hmac
import json
from hashlib import sha256
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

PROTOCOL_VERSION = 2
# One datagram (tag + JSON) must fit an unfragmented Ethernet/HaLow MTU payload.
MAX_DATAGRAM_BYTES = 1200
UPLINK_HZ = 10.0
MAX_DETECTIONS = 16
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
# Public, deliberately insecure key for simulation and tests ONLY. The service
# refuses it unless SIM_MODE is on; never provision it on a real mower.
SIM_LINK_KEY = b"lawnberry-thor-link/SIM-ONLY/public-key/never-on-hardware"

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
    v: Literal[2] = PROTOCOL_VERSION
    # Stamped by ThorLinkService (0 = not yet stamped): the session id and
    # counter the Thor must echo back as ``Downlink.sid`` / ``Downlink.ack``.
    sid: int = Field(default=0, ge=0, le=MAX_SESSION_ID)
    seq: int = Field(ge=0, le=MAX_COUNTER)
    t_ms: int = Field(ge=0)  # Pi monotonic clock, ms
    lat: float | None = None
    lon: float | None = None
    fix: str = "none"  # RTK fix quality: none|gps|dgps|rtk_float|rtk_fixed
    heading_deg: float | None = None
    detections: list[Detection] = Field(default_factory=list, max_length=MAX_DETECTIONS)
    estop: bool = False
    safe_hold: bool = False
    blade: bool = False
    waypoint_index: int | None = None


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


def validate_key(key: bytes) -> bytes:
    """Return ``key`` as bytes, or raise ``LinkKeyError`` if it is too short to be safe."""
    if not isinstance(key, bytes | bytearray) or len(key) < KEY_MIN_BYTES:
        raise LinkKeyError(f"link key must be at least {KEY_MIN_BYTES} bytes")
    return bytes(key)


def parse_key(text: str) -> bytes:
    """Decode a hex-encoded key as stored in the environment or secrets file.

    The value is never echoed in the error. Generate one with
    ``openssl rand -hex 32``.
    """
    try:
        raw = bytes.fromhex(text.strip())
    except ValueError:
        raise LinkKeyError("link key must be hex-encoded") from None
    return validate_key(raw)


def _tag(key: bytes, domain: bytes, body: bytes) -> bytes:
    return hmac.new(key, domain + body, sha256).digest()


def encode(message: Uplink | Downlink, key: bytes) -> bytes:
    """Serialise ``message`` as ``HMAC-SHA256 tag || compact JSON`` (<= MAX_DATAGRAM_BYTES)."""
    key = validate_key(key)
    domain = _UPLINK_DOMAIN if isinstance(message, Uplink) else _DOWNLINK_DOMAIN
    payload = message.model_dump(exclude_none=True, by_alias=True)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    data = _tag(key, domain, body) + body
    if len(data) > MAX_DATAGRAM_BYTES:
        raise LinkBudgetError(f"{type(message).__name__} is {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    return data


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
