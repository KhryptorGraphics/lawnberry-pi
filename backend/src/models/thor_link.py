"""Pi <-> Thor link protocol (W7), version 1.

Wire format: one UDP datagram per message, compact JSON with short keys.
Raw imagery never crosses the link; only distilled state goes up and
waypoints/corrections come down. Every encoded message is checked against
``MAX_DATAGRAM_BYTES`` so the stream stays inside the measured Wi-Fi HaLow
budget (see ``docs/tractor-platform.md`` "Pi <-> Thor link").

Uplink (Pi -> Thor): position, distilled detections, health. Sent at
``UPLINK_HZ``; it doubles as the Pi's heartbeat.
Downlink (Thor -> Pi): waypoints and corrections. Each carries the Thor's
heartbeat; silence longer than the link timeout puts the Pi in safe hold.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

PROTOCOL_VERSION = 1
# One datagram must fit an unfragmented Ethernet/HaLow MTU payload.
MAX_DATAGRAM_BYTES = 1200
UPLINK_HZ = 10.0
MAX_DETECTIONS = 16
# Plans travel in windows of this many waypoints (``Downlink.start_index``) so a
# worst-case window still fits one datagram.
MAX_WAYPOINTS = 16


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
    v: Literal[1] = PROTOCOL_VERSION
    seq: int = Field(ge=0)
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
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    blade: bool = False


class Downlink(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    seq: int = Field(ge=0)
    plan_id: str = Field(min_length=1, max_length=32)
    start_index: int = Field(default=0, ge=0)
    waypoints: list[Waypoint] = Field(default_factory=list, max_length=MAX_WAYPOINTS)
    pause: bool = False


class LinkBudgetError(ValueError):
    """An encoded message would exceed the per-datagram byte budget."""


def encode(message: Uplink | Downlink) -> bytes:
    payload = message.model_dump(exclude_none=True, by_alias=True)
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_DATAGRAM_BYTES:
        raise LinkBudgetError(f"{type(message).__name__} is {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    return data


def _decode(data: bytes, model: type[Any]) -> Any:
    if len(data) > MAX_DATAGRAM_BYTES:
        raise LinkBudgetError(f"datagram {len(data)} B > {MAX_DATAGRAM_BYTES} B")
    return model.model_validate_json(data)


def decode_uplink(data: bytes) -> Uplink:
    return _decode(data, Uplink)


def decode_downlink(data: bytes) -> Downlink:
    return _decode(data, Downlink)


def worst_case_link_bytes_per_second() -> float:
    """Upper bound on uplink bytes/s at ``UPLINK_HZ``; compare with measured HaLow throughput."""
    return MAX_DATAGRAM_BYTES * UPLINK_HZ
