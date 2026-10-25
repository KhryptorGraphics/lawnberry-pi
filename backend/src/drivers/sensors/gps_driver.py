"""GPS Driver (T058)

Supports u-blox ZED-F9P via USB or UART, Neo-8M via UART and Quectel LC29H(DA) via UART. This
driver follows the HardwareDriver lifecycle and returns GPS positions when polled.

SIM_MODE notes:
- When SIM_MODE=1 (default in CI), returns deterministic positions without
  touching hardware. Accuracy reflects module class (F9P <1m, Neo-8M ~3m).

Hardware notes (audit SAF-04, SAF-06, SAF-S1, GPS-01 .. GPS-15):

* **Accuracy is an error estimate, never HDOP.** ``GpsReading.accuracy`` is the receiver's own
  horizontal 1-sigma in metres, from the GST sentence (``sqrt(sigma_lat^2 + sigma_lon^2)``, else
  the error ellipse), and never better than a floor for the fix level (plain fix 0.5 m, DGPS
  0.3 m, RTK float 0.2 m, RTK fixed 3 cm); when both exist the **larger** wins. HDOP is a
  dimensionless geometry factor: it was once copied into ``accuracy`` as metres, so a plain 3-4 m
  fix with a good satellite geometry read "0.8 m accurate" and passed the GNSS health gate. It is
  still reported, as ``hdop``. Without a GST the accuracy is ``None``: unknown, which the GNSS
  monitor does not accept, except for an RTK fix whose GGA carries a plausible age of corrections
  (at most ``RTK_MAX_CORRECTION_AGE_S``): that keeps the typical figure of its level (fixed 3 cm,
  float 20 cm). A fix level claimed by one GGA digit, with no GST and no age of corrections, is not
  believed. A receiver that is expected to drive autonomously therefore has to output GST
  (``$GNGST``) or report the age of its RTK corrections.
* **A reading is dated by the receiver's epoch and is never a replay.** Sentences are grouped
  into epochs by their UTC field. The newest epoch in the tty buffer is the one reported (the
  buffer is drained, never read oldest-first), and an epoch is stamped with the host time of its
  position sentence minus the delay estimated from the receiver's own UTC (:class:`EpochClock`:
  how much later than ever before in this session, per the receiver's clock, the epoch was read).
  An epoch whose UTC is not later than the last accepted one (a frozen, repeated or backwards
  stream) is not a new fix and builds nothing, so the previous reading keeps its old stamp and
  ages out; a UTC that jumps ahead of real time is held back until a second epoch continues from
  it (one wild epoch cannot move the reference). Design limit: with no independent time source a
  replay that advances its own UTC at the real rate is indistinguishable from a live receiver, and
  a receiver whose clock is reset to an earlier time without a pause of ``SILENCE_REPROBE_S`` is
  refused until its UTC passes the last accepted one (reopening the port starts a new session).
* **Only a real fix is a position.** A GGA fix quality of 1, 2, 4 or 5 (an RMC with status ``A``
  and a fix mode) with a valid, finite, in-range coordinate pair is a position. Quality 0, 6, 7,
  8, 3, 9, an unknown or garbled digit, a coordinate that is out of range or has minutes >= 60,
  a hemisphere letter on the wrong axis, or an RMC with status ``V`` yields a reading with *no*
  position and the fix status, so the monitor and the navigation loops see "no fix" at once.
* **Nothing blocks the event loop and no line can wedge the reader.** Every serial open, read and
  the gpsd socket run on one daemon worker thread; ``read_position`` waits at most
  ``READ_DEADLINE_S`` for it and then returns the last reading (which ages out and degrades the
  fix) while the worker finishes. Lines are assembled here from ``read()`` chunks
  (:class:`NmeaLineReader`): CR or LF ends a line, a line is bounded, a sentence is taken from its
  last ``$`` (binary UBX frames and junk in front of it cost nothing), and the deadline and the
  closing flag are checked per chunk, so a stream with no line ends cannot hold the worker (and
  discovery of the real receiver behind it) for longer than one port timeout. A port that
  delivers no valid NMEA for ``SILENCE_REPROBE_S`` is dropped and probed again (a receiver
  reconfigured to another baud). Autodetection is never on the hot path: after a probe pass finds
  nothing it is not repeated for ``AUTODETECT_RETRY_S``.
* **The driver can be stopped and forgotten.** ``stop()`` closes the port and ends the worker; a
  driver that is dropped without it releases its worker and, with the port object, the exclusive
  lock (the idle worker holds no reference to its owner).
* **No other driver's port.** The RoboHAT MM1 and the ultrasonic MCU enumerate as USB CDC ACM
  ports next to a USB GNSS receiver, and a second reader on a tty steals the bytes the owner is
  waiting for (a RoboHAT ``pwm`` acknowledgement, an ultrasonic frame). The candidate list drops
  every device that is configured for another driver (``ROBOHAT_*`` environment, the RoboHAT port
  of the settings profile, ``ultrasonic.port`` in ``hardware.yaml``, ``GPS_EXCLUDE_DEVICES``) or
  identifies as one (USB vendor 0x2E8A, a RoboHAT/CircuitPython descriptor); a ``/dev/ttyACM*``
  device is probed only if it was enumerated (so its identity could be checked) and the
  configuration that names the other drivers' ports could be read; and every port is opened with
  ``exclusive=True`` so one that another LawnBerry driver has locked is skipped, not read.
  Design limits: ``exclusive`` is an advisory ``flock`` that only helps against openers that take
  it too (the operator self-test in ``services/hw_selftest.py`` does not), and probing an
  unidentified port asserts DTR and holds the lock for up to ``PROBE_WINDOW_S`` per baud rate.

Platform notes:
- Real hardware access is guarded by lazy imports and not exercised in tests.
- UART/USB device paths are configurable via config or environment variables.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import glob
import json
import logging
import math
import os
import queue
import re
import socket
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ...core.freshness import parse_stamp
from ...core.simulation import is_simulation_mode
from ...models.sensor_data import GpsMode, GpsReading
from ..base import HardwareDriver

logger = logging.getLogger(__name__)

#: USB vendor id of u-blox (ZED-F9P over USB shows up as a CDC ACM port with it).
UBLOX_VID = 0x1546
#: USB vendor id of Raspberry Pi RP2040 boards: the RoboHAT MM1 and the ultrasonic MCU. Never a GNSS
#: receiver.
RP2040_VID = 0x2E8A
#: Descriptor words (spaces, ``_`` and ``-`` removed) that mark another LawnBerry board.
_OTHER_DRIVER_WORDS = ("robohat", "circuitpython", "rp2040", "pico", "cytron", "lawnberry")
#: Environment variables that name the RoboHAT's serial port (``services/robohat_service.py``).
_OTHER_DRIVER_PORT_ENV = ("ROBOHAT_SERIAL_PORT", "ROBOHAT_PORT", "LAWN_ROBOHAT_PORT")
#: Comma-separated extra devices this driver must never open.
EXCLUDE_ENV = "GPS_EXCLUDE_DEVICES"

#: Hardware UARTs a UART-wired receiver (LC29H HAT, Neo-8M) is on, preferred first.
_UART_DEVICES = ("/dev/ttyAMA0", "/dev/serial0", "/dev/ttyS0")
#: USB-serial adapters a receiver may sit behind.
_USB_SERIAL_DEVICES = ("/dev/ttyUSB0", "/dev/ttyUSB1")

#: Knots to metres per second.
_KNOTS_TO_MS = 1852.0 / 3600.0

#: The statuses that are a GNSS position fix (GGA quality 1, 2, 4, 5). Everything else the driver
#: reports (``NO_FIX``, ``DEAD_RECKONING``, ``MANUAL``, ``SIMULATION``) carries no position.
FIX_STATUSES = frozenset({"GPS_FIX", "DGPS", "RTK_FLOAT", "RTK_FIXED"})
_RTK_STATUSES = frozenset({"RTK_FLOAT", "RTK_FIXED"})
#: GGA fix quality -> status. Anything not listed (3, 9, a garbled digit, ...) is ``NO_FIX``.
_QUALITY_STATUS = {
    0: "NO_FIX",
    1: "GPS_FIX",
    2: "DGPS",
    4: "RTK_FIXED",
    5: "RTK_FLOAT",
    6: "DEAD_RECKONING",
    7: "MANUAL",
    8: "SIMULATION",
}
#: RMC mode indicator (NMEA 2.3+) -> status. ``A`` and ``P`` are a plain fix; an unlisted letter is
#: ``NO_FIX``. An RMC without a mode field says nothing beyond its ``A``/``V`` status.
_RMC_MODE_STATUS = {
    "A": "GPS_FIX",
    "P": "GPS_FIX",
    "D": "DGPS",
    "F": "RTK_FLOAT",
    "R": "RTK_FIXED",
    "E": "DEAD_RECKONING",
    "M": "MANUAL",
    "S": "SIMULATION",
    "N": "NO_FIX",
}
#: gpsd TPV ``status`` -> status (gpsd >= 3.18). 0 / absent with ``mode`` >= 2 is a plain fix.
_GPSD_STATUS = {
    0: "GPS_FIX",
    1: "GPS_FIX",
    2: "DGPS",
    3: "RTK_FIXED",
    4: "RTK_FLOAT",
    5: "DEAD_RECKONING",
    6: "DEAD_RECKONING",
    8: "SIMULATION",
}
#: Quality order used to decide whether a borrowed GST still describes the fix: a worse status
#: than the one it was measured under may not inherit its sigma.
_STATUS_RANK = {
    "NO_FIX": 0,
    "DEAD_RECKONING": 0,
    "MANUAL": 0,
    "SIMULATION": 0,
    "GPS_FIX": 1,
    "DGPS": 2,
    "RTK_FLOAT": 3,
    "RTK_FIXED": 4,
}


# ------------------------------------------------------------------------------ NMEA helpers


def nmea_checksum_ok(sentence: str) -> bool:
    """True iff ``sentence`` is ``$<body>*HH`` and HH is the XOR of every byte of ``<body>``.

    A corrupted digit in a GST sigma could turn a bad accuracy into a good one, so no sentence
    is parsed unless its checksum matches.
    """
    sentence = sentence.strip()
    if not sentence.startswith("$"):
        return False
    body, star, tail = sentence[1:].partition("*")
    if not star or len(tail) != 2:
        return False
    try:
        given = int(tail, 16)
    except ValueError:
        return False
    check = 0
    for ch in body:
        check ^= ord(ch)
    return check == given


def _finite(text: str) -> float | None:
    """``text`` as a finite number, else ``None`` (empty, ``nan``, ``inf``, garbage)."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _finite_non_negative(text: str) -> float | None:
    """``text`` as a finite number >= 0, else ``None`` (empty, ``nan``, ``inf``, negative)."""
    value = _finite(text)
    return value if value is not None and value >= 0.0 else None


def _digits(text: str) -> int | None:
    """``text`` as a non-negative integer if it is nothing but ASCII digits, else ``None``."""
    return int(text) if text.isascii() and text.isdigit() else None


_UTC_FIELD = re.compile(r"(\d{2})(\d{2})(\d{2})(\.\d{1,9})?")
_LAT_FIELD = re.compile(r"(\d{2})(\d{2}\.\d{1,10})")
_LON_FIELD = re.compile(r"(\d{3})(\d{2}\.\d{1,10})")


def parse_nmea_utc(text: str) -> float | None:
    """Seconds since 00:00 UTC from an ``hhmmss[.ss]`` field, or ``None`` if it is not one.

    Hours 0-23, minutes 0-59, seconds 0-60 (a leap second is ``..60``).
    """
    match = _UTC_FIELD.fullmatch(text)
    if match is None:
        return None
    hours, minutes, seconds = int(match[1]), int(match[2]), int(match[3])
    if hours > 23 or minutes > 59 or seconds > 60:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds + (float(match[4]) if match[4] else 0.0)


def parse_nmea_coordinate(value: str, hemisphere: str, axis: str) -> float | None:
    """Decimal degrees from NMEA ``ddmm.mmmm`` (``axis`` ``"lat"``) / ``dddmm.mmmm`` (``"lon"``).

    ``None`` unless the field has exactly that shape (digits only, 4 or 5 before the dot), the
    minutes are below 60, the hemisphere letter belongs to the axis (``N``/``S`` for latitude,
    ``E``/``W`` for longitude, upper case) and the result is within 90 / 180 degrees.
    """
    if axis == "lat":
        match, signs, limit = _LAT_FIELD.fullmatch(value), {"N": 1.0, "S": -1.0}, 90.0
    else:
        match, signs, limit = _LON_FIELD.fullmatch(value), {"E": 1.0, "W": -1.0}, 180.0
    sign = signs.get(hemisphere)
    if match is None or sign is None:
        return None
    minutes = float(match[2])
    if minutes >= 60.0:
        return None
    degrees = int(match[1]) + minutes / 60.0
    return sign * degrees if degrees <= limit else None


@dataclass(frozen=True, slots=True)
class Gga:
    """The fields of a GGA sentence this driver uses; ``None`` where a field is empty or bad."""

    utc_s: float | None
    quality: int | None  # the raw fix-quality digit
    lat: float | None
    lon: float | None
    altitude_m: float | None
    satellites: int | None
    hdop: float | None
    correction_age_s: float | None  # age of the differential / RTK corrections


@dataclass(frozen=True, slots=True)
class Rmc:
    """The fields of an RMC sentence this driver uses."""

    utc_s: float | None
    valid: bool  # status field ``A``
    mode: str | None  # NMEA 2.3+ mode indicator, ``None`` if the sentence has none
    lat: float | None
    lon: float | None
    speed_mps: float | None
    course_deg: float | None


def parse_gga(sentence: str) -> Gga | None:
    """``$--GGA,time,lat,N,lon,E,quality,sats,hdop,alt,M,sep,M,corr_age,station`` or ``None``.

    ``None`` only when the sentence is too short to carry a fix quality. A latitude without its
    longitude (or the other way round) is no position at all.
    """
    parts = sentence.split("*")[0].split(",")
    if len(parts) < 7:
        return None
    lat = parse_nmea_coordinate(parts[2], parts[3], "lat")
    lon = parse_nmea_coordinate(parts[4], parts[5], "lon")
    if lat is None or lon is None:
        lat = lon = None
    return Gga(
        utc_s=parse_nmea_utc(parts[1]),
        quality=_digits(parts[6]),
        lat=lat,
        lon=lon,
        altitude_m=_finite(parts[9]) if len(parts) > 9 else None,
        satellites=_digits(parts[7]) if len(parts) > 7 else None,
        hdop=_finite_non_negative(parts[8]) if len(parts) > 8 else None,
        correction_age_s=_finite_non_negative(parts[13]) if len(parts) > 13 else None,
    )


def parse_rmc(sentence: str) -> Rmc | None:
    """``$--RMC,time,status,lat,N,lon,E,speed_kn,course,date,magvar,E,mode`` or ``None``."""
    parts = sentence.split("*")[0].split(",")
    if len(parts) < 3:
        return None
    lat = parse_nmea_coordinate(parts[3], parts[4], "lat") if len(parts) > 4 else None
    lon = parse_nmea_coordinate(parts[5], parts[6], "lon") if len(parts) > 6 else None
    if lat is None or lon is None:
        lat = lon = None
    knots = _finite_non_negative(parts[7]) if len(parts) > 7 else None
    course = _finite_non_negative(parts[8]) if len(parts) > 8 else None
    mode = parts[12] if len(parts) > 12 and parts[12] else None
    return Rmc(
        utc_s=parse_nmea_utc(parts[1]),
        valid=parts[2] == "A",
        mode=mode,
        lat=lat,
        lon=lon,
        speed_mps=None if knots is None else knots * _KNOTS_TO_MS,
        course_deg=course if course is not None and course <= 360.0 else None,
    )


def parse_gst_sigma(sentence: str) -> float | None:
    """Horizontal 1-sigma error in metres from a GST sentence, or ``None`` if it has none.

    ``$--GST,time,rms,semi_major,semi_minor,orientation,sigma_lat,sigma_lon,sigma_alt``. Taken
    from ``sigma_lat``/``sigma_lon`` (``sqrt(a^2 + b^2)``); if those are empty, from the error
    ellipse (the same quantity); if only the semi-major axis is there, ``sqrt(2)`` times it (the
    largest the horizontal error can be given that one axis). Empty, non-numeric, non-finite and
    negative fields, and an all-zero estimate (a receiver's "no estimate"), are no estimate.
    """
    parts = sentence.split("*")[0].split(",")

    def field(i: int) -> float | None:
        return _finite_non_negative(parts[i]) if len(parts) > i else None

    lat, lon = field(6), field(7)
    major, minor = field(3), field(4)
    if lat is not None and lon is not None:
        sigma = math.hypot(lat, lon)
    elif major is not None and minor is not None:
        sigma = math.hypot(major, minor)
    elif major is not None:
        sigma = major * math.sqrt(2.0)
    else:
        return None
    return sigma if sigma > 0.0 else None


def fix_status_from_quality(quality: int | None) -> str:
    """The status of a GGA fix quality digit; anything outside the table is ``NO_FIX``."""
    return _QUALITY_STATUS.get(quality, "NO_FIX") if quality is not None else "NO_FIX"


def fix_status_from_rmc(rmc: Rmc) -> str:
    """The status an RMC claims: ``V`` is ``NO_FIX``, else its mode indicator, else a plain fix."""
    if not rmc.valid:
        return "NO_FIX"
    if rmc.mode is None:
        return "GPS_FIX"
    return _RMC_MODE_STATUS.get(rmc.mode, "NO_FIX")


#: Typical 1-sigma horizontal accuracy of an RTK solution, by receiver fix status. Typical, not
#: measured here: used as a floor under (never instead of) the receiver's own estimate, and as the
#: estimate itself only when the GGA vouches for the corrections (see ``classify``).
_RTK_TYPICAL_M = {"RTK_FIXED": 0.03, "RTK_FLOAT": 0.20}
#: No fix level is ever more accurate than this, whatever a GST claims (a sigma of 1e-9 on a plain
#: fix is a corrupt or lying figure). Plain fix and DGPS are floored well above RTK.
_ACCURACY_FLOOR_M = {"GPS_FIX": 0.5, "DGPS": 0.3, "RTK_FLOAT": 0.20, "RTK_FIXED": 0.03}
_ABSOLUTE_FLOOR_M = 0.005


def combine_accuracy(
    gst_sigma_m: float | None, rtk_status: str | None, assume_typical: bool = True
) -> float | None:
    """The accuracy to report: the larger of the receiver's GST sigma, the RTK figure and a floor.

    ``None`` when there is neither a sigma nor (``assume_typical``) an RTK figure (HDOP does not
    count, audit SAF-04). The floor depends on the fix level: a plain fix is never reported better
    than 0.5 m, DGPS 0.3 m, RTK float 0.2 m, RTK fixed 3 cm (anything else 5 mm).
    """
    typical = _RTK_TYPICAL_M.get(rtk_status or "") if assume_typical else None
    known = [v for v in (gst_sigma_m, typical) if v is not None]
    if not known:
        return None
    return max(_ACCURACY_FLOOR_M.get(rtk_status or "", _ABSOLUTE_FLOOR_M), *known)


def reading_age_s(reading: Any, now: datetime | None = None) -> float | None:
    """Seconds since ``reading``'s own timestamp, or ``None`` if it has none that can be aged.

    A missing or unreadable stamp, and one from the future, are unknown rather than fresh. Meant
    for displays: the GNSS monitor ages a fix on the monotonic clock.
    """
    stamp = parse_stamp(getattr(reading, "timestamp", None))
    if stamp is None:
        return None
    age = ((now or datetime.now(UTC)) - stamp).total_seconds()
    return age if age >= 0.0 else None


# ------------------------------------------------------------------------------ line framing


class NmeaLineReader:
    """Bounded assembly of NMEA sentences from the chunks a serial port returns.

    ``pyserial``'s ``readline`` has no overall timeout: it returns on a line feed or after one
    silent port timeout, so a stream that never sends a line feed (a CR-only device, a stuck
    line, a binary protocol) holds the caller for ever and its buffer grows without bound. This
    reader is fed whatever ``read`` returned and never waits. CR and LF both end a line; the text
    after the last ``$`` of a line is the candidate sentence (a UBX frame or other junk in front
    of it is dropped, and nothing in a sentence can contain ``$``); a line that is still open
    after ``MAX_LINE`` bytes is not a sentence and is dropped, keeping only a tail that starts at
    a ``$``.
    """

    MAX_LINE = 256

    def __init__(self) -> None:
        self._open = b""

    def feed(self, data: bytes) -> list[str]:
        """The complete candidate sentences ``data`` finished (not yet checksum-checked)."""
        parts = (self._open + data).replace(b"\r", b"\n").split(b"\n")
        tail = parts.pop()  # after the last line end: an unfinished line
        sentences: list[str] = []
        for part in parts:
            start = part.rfind(b"$")
            if start >= 0:
                text = part[start:].decode("ascii", errors="ignore")
                if len(text) <= self.MAX_LINE:
                    sentences.append(text)
        if len(tail) > self.MAX_LINE:
            start = tail.rfind(b"$")
            tail = tail[start:] if len(tail) - start <= self.MAX_LINE else b""  # start -1: no "$"
        self._open = tail
        return sentences


# ------------------------------------------------------------------------------ epoch clock


@dataclass(frozen=True, slots=True)
class EpochVerdict:
    """What :meth:`EpochClock.accept` decided about one epoch."""

    accepted: bool
    delay_s: float  # how much older than the freshest epoch of the session it was when read
    reason: str  # ``first``, ``ok``, ``rebased``, ``stale`` or ``jump``


class EpochClock:
    """Judges a source's epoch times (UTC) against the host's monotonic clock.

    ``accept(u, t)`` takes the receiver's time ``u`` of an epoch (seconds; time of day when
    ``period_s`` is 86400, absolute otherwise) and the host monotonic time ``t`` at which it was
    read. It returns whether the epoch is a NEW fix and, if so, how late it was read.

    * An epoch whose time is not later than the last accepted one is **stale**: repeated by a
      frozen receiver, replayed, or running backwards. It is never accepted, and never moves the
      reference.
    * An epoch that is later than real time says it should be (the receiver's time advanced more
      than the host's did, by more than ``TOLERANCE_S``) is a **jump**: held back, and accepted
      (with the reference moved) only when the NEXT epoch continues from it at the host's rate.
      So one wild epoch cannot poison the reference (the next genuine epoch is judged against the
      old one and the wild one is forgotten), and a receiver whose clock really moves forward
      loses one epoch.
    * Otherwise the epoch is new. Its **delay** is its host-minus-receiver time offset above the
      lowest offset of the session (the freshest the stream has been), which creeps up at
      ``CREEP`` seconds per second so crystal drift between the two clocks cannot read as lag. A
      queue that grows, a stream running slower than real time, a backlog read late: all show as
      delay. A constant delay from the very first epoch cannot be seen from the stream.
    """

    TOLERANCE_S = 5.0
    CREEP = 2e-4

    def __init__(self, period_s: float | None = 86400.0) -> None:
        self._period = period_s
        self._ref: tuple[float, float] | None = None  # last accepted (u, t)
        self._candidate: tuple[float, float] | None = None  # a held-back jump (u, t)
        self._base_off = 0.0
        self._base_t = 0.0

    def _near(self, u: float, reference: float) -> float:
        """``u`` shifted by whole periods to the representation nearest ``reference``."""
        if self._period is None:
            return u
        return u + self._period * round((reference - u) / self._period)

    def _rebase(self, u: float, t: float, reason: str) -> EpochVerdict:
        self._ref = (u, t)
        self._candidate = None
        self._base_off = t - u
        self._base_t = t
        return EpochVerdict(True, 0.0, reason)

    def accept(self, u: float, t: float) -> EpochVerdict:
        """Judge the epoch with receiver time ``u`` that was read at host time ``t``."""
        if self._ref is None:
            return self._rebase(u, t, "first")
        ref_u, ref_t = self._ref
        u = self._near(u, ref_u)
        if u <= ref_u:
            return EpochVerdict(False, 0.0, "stale")
        if (u - ref_u) - (t - ref_t) > self.TOLERANCE_S:
            held = self._candidate
            if held is not None:
                u_held = self._near(u, held[0])
                if u_held > held[0] and abs((u_held - held[0]) - (t - held[1])) <= self.TOLERANCE_S:
                    return self._rebase(u_held, t, "rebased")
            self._candidate = (u, t)
            return EpochVerdict(False, 0.0, "jump")
        self._candidate = None
        self._ref = (u, t)
        offset = t - u
        self._base_off = min(offset, self._base_off + self.CREEP * (t - self._base_t))
        self._base_t = t
        return EpochVerdict(True, offset - self._base_off, "ok")


# ------------------------------------------------------------------------------ epochs


@dataclass(slots=True)
class _Epoch:
    """What one burst of NMEA sentences (one UTC field) has said so far."""

    key: float  # receiver time of the burst, seconds of the UTC day
    t_pos: float  # host monotonic time its position sentence (GGA, else RMC) was read
    gga: Gga | None = None
    rmc: Rmc | None = None
    gst_seen: bool = False
    gst_sigma: float | None = None
    gst_t: float = 0.0  # host monotonic time its GST was read
    kinds: set[str] = field(default_factory=set)
    judged: bool = False
    delay_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _Classified:
    """What an epoch amounts to: a status, a position only for a real fix."""

    status: str
    lat: float | None
    lon: float | None
    altitude_m: float | None
    typical_ok: bool  # may the typical RTK figure stand in for a missing GST sigma?

    @property
    def is_fix(self) -> bool:
        return self.status in FIX_STATUSES


def classify(epoch: _Epoch, max_correction_age_s: float) -> _Classified:
    """The status and position of ``epoch``.

    The GGA is authoritative; an RMC alone (a GGA lost or never sent) is judged by its own status
    and mode indicator; an RMC that says the fix is invalid, estimated, manual or simulated vetoes
    a GGA that claims a fix. A claimed fix without a valid coordinate pair is ``NO_FIX``. The
    typical RTK figure is allowed only for a GGA RTK fix that reports a correction age within
    ``max_correction_age_s``.
    """
    gga, rmc = epoch.gga, epoch.rmc
    if gga is not None:
        status = fix_status_from_quality(gga.quality)
        lat, lon, alt = gga.lat, gga.lon, gga.altitude_m
        age = gga.correction_age_s
        typical = status in _RTK_STATUSES and age is not None and age <= max_correction_age_s
    elif rmc is not None:
        status = fix_status_from_rmc(rmc)
        lat, lon, alt, typical = rmc.lat, rmc.lon, None, False
    else:
        return _Classified("NO_FIX", None, None, None, False)
    if status in FIX_STATUSES and (lat is None or lon is None):
        status = "NO_FIX"
    if status in FIX_STATUSES and gga is not None and rmc is not None:
        veto = fix_status_from_rmc(rmc)
        if veto not in FIX_STATUSES:
            status = veto
    if status not in FIX_STATUSES:
        return _Classified(status, None, None, None, False)
    return _Classified(status, lat, lon, alt, typical)


class _Assembler:
    """Groups sentences into epochs and learns which kinds a receiver sends.

    ``add`` returns the epochs a sentence CLOSED (the burst before it); the epoch being built is
    ``cur``. A sentence of the epoch just closed (a GST that follows the next epoch's first
    sentence, an RMC the reader did not wait for) still counts: it is merged into that epoch and
    the kind is learned, so the next read waits for it. Until the first epoch has closed every
    kind is awaited (a new session finds out what the receiver sends); afterwards the kinds
    awaited are those seen within ``EXPECT_WITHIN_S``.
    """

    KINDS = ("GGA", "RMC", "GST")
    EXPECT_WITHIN_S = 5.0

    def __init__(self) -> None:
        self.cur: _Epoch | None = None
        self.prev: _Epoch | None = None
        self.learned: dict[str, float] = {}
        self.discovering = True
        #: Sentences that an epoch already had (a repeated burst, a replay), ignored.
        self.repeats = 0

    def expected(self, now: float) -> frozenset[str]:
        if self.discovering:
            return frozenset(self.KINDS)
        return frozenset(
            kind for kind, seen in self.learned.items() if now - seen <= self.EXPECT_WITHIN_S
        )

    def complete(self, epoch: _Epoch, now: float) -> bool:
        """Has ``epoch`` every kind this receiver is known to send (and a position sentence)?"""
        return bool({"GGA", "RMC"} & epoch.kinds) and self.expected(now) <= epoch.kinds

    def add(self, kind: str, utc_s: float | None, payload: Any, t: float) -> list[_Epoch]:
        self.learned[kind] = t
        if utc_s is None:
            return []  # an untimed sentence belongs to no epoch
        closed: list[_Epoch] = []
        cur = self.cur
        if cur is None or utc_s != cur.key:
            if self.prev is not None and utc_s == self.prev.key:
                # late: it belongs to the epoch just closed
                self._merge(self.prev, kind, payload, t)
                return closed
            if cur is not None:
                self.prev = cur
                self.discovering = False
                closed.append(cur)
            cur = self.cur = _Epoch(key=utc_s, t_pos=t)
        self._merge(cur, kind, payload, t)
        return closed

    def _merge(self, epoch: _Epoch, kind: str, payload: Any, t: float) -> None:
        if kind in epoch.kinds:
            self.repeats += 1  # a second GGA (RMC, GST) of one epoch is a repeat: the first stands
            return
        epoch.kinds.add(kind)
        if kind == "GGA":
            epoch.gga, epoch.t_pos = payload, t
        elif kind == "RMC":
            epoch.rmc = payload
            if epoch.gga is None:
                epoch.t_pos = t
        else:
            epoch.gst_seen, epoch.gst_sigma, epoch.gst_t = True, payload, t


class _Stream:
    """Everything that belongs to one open port; dropped with it."""

    def __init__(self, opened_at: float) -> None:
        self.reader = NmeaLineReader()
        self.asm = _Assembler()
        self.last_valid_t = opened_at  # when a checksum-valid ``$G`` sentence last arrived
        self.last_accept_t = opened_at  # when an epoch was last taken as a new reading
        self.last_delivered: _Epoch | None = None
        self.last_nofix_key: float | None = None


class _PortSilent(Exception):
    """The port delivered no valid NMEA for ``SILENCE_REPROBE_S``."""


# ------------------------------------------------------------------------------ serial ports


@dataclass(frozen=True)
class SerialPortInfo:
    """What ``serial.tools.list_ports`` knows about one port (the fields this driver uses)."""

    device: str
    vid: int | None = None
    description: str = ""


def _enumerate_serial_ports() -> tuple[list[SerialPortInfo], bool]:
    """``(ports pyserial can see, whether the enumeration worked)``."""
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception:
        return [], False
    found: list[SerialPortInfo] = []
    try:
        for info in list_ports.comports():
            text = " ".join(
                str(getattr(info, name, "") or "")
                for name in ("description", "manufacturer", "product")
            )
            vid = getattr(info, "vid", None)
            found.append(
                SerialPortInfo(
                    device=str(info.device),
                    vid=vid if isinstance(vid, int) else None,
                    description=text,
                )
            )
    except Exception:
        return found, False
    return found, True


def list_serial_ports() -> list[SerialPortInfo]:
    """The ports pyserial can see, with their USB identity; ``[]`` if it cannot enumerate."""
    return _enumerate_serial_ports()[0]


def _normalised(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch not in " _-")


def _real(device: str) -> str:
    try:
        return os.path.realpath(device)
    except OSError:
        return device


def _profile_robohat_port() -> str:
    """The RoboHAT port of the persisted settings profile (``hardware.robohat_port``), or ``""``.

    Read from the same files, in the same order, as ``services.robohat_service`` does (the
    service honours it, so it is that driver's port); not imported from there, because importing
    the services package from a hardware driver's worker thread drags in the whole backend.
    """
    bases: list[str] = []
    env_dir = os.getenv("LAWN_SETTINGS_DIR") or os.getenv("LAWN_CONFIG_DIR")
    if env_dir:
        bases.append(env_dir)
    bases.extend(["./config", os.path.join(os.getcwd(), "config")])
    for base in bases:
        for name in ("default.json", "settings.json"):
            try:
                with open(os.path.join(base, name), encoding="utf-8") as handle:
                    data = json.load(handle)
                hardware = data.get("hardware") if isinstance(data, dict) else None
                port = hardware.get("robohat_port") if isinstance(hardware, dict) else None
            except Exception:
                continue
            if isinstance(port, str) and port.strip():
                return port.strip()
    return ""


_warned_ownership_config = False


def _other_driver_ports(ports: list[SerialPortInfo] | None) -> tuple[set[str], bool]:
    """``(real paths another LawnBerry driver owns, whether every source could be read)``."""
    global _warned_ownership_config
    named: list[str] = []
    complete = True
    for var in _OTHER_DRIVER_PORT_ENV:
        named.append(os.environ.get(var, ""))
    named.extend(os.environ.get(EXCLUDE_ENV, "").split(","))
    try:
        from ...core.config_loader import ConfigLoader

        ultrasonic = ConfigLoader().get()[0].ultrasonic_config
        named.append(str(getattr(ultrasonic, "port", "") or ""))
    except Exception as exc:
        complete = False
        if not _warned_ownership_config:
            _warned_ownership_config = True
            logger.warning(
                "GPS: the hardware configuration could not be read (%r); the ultrasonic port is "
                "unknown, so no unidentified /dev/ttyACM* device will be probed",
                exc,
            )
    named.append(_profile_robohat_port())
    owned = {_real(p.strip()) for p in named if p and p.strip()}
    for info in ports if ports is not None else list_serial_ports():
        text = _normalised(f"{info.device} {info.description}")
        if info.vid == RP2040_VID or any(word in text for word in _OTHER_DRIVER_WORDS):
            owned.add(_real(info.device))
    return owned, complete


def ports_owned_by_other_drivers(ports: list[SerialPortInfo] | None = None) -> set[str]:
    """Real paths of the serial devices that belong to another LawnBerry driver.

    From configuration (``ROBOHAT_*`` variables, the RoboHAT port of the settings profile,
    ``GPS_EXCLUDE_DEVICES``, ``ultrasonic.port`` in ``hardware.yaml``) and from identity (USB
    vendor 0x2E8A, a RoboHAT or CircuitPython descriptor). Never raises.
    """
    return _other_driver_ports(ports)[0]


def preferred_gps_device(mode: GpsMode, ports: list[SerialPortInfo] | None = None) -> str | None:
    """The device a receiver wired as ``mode`` is expected on, or ``None`` if there is no sign.

    A UART-wired receiver (Neo-8M, LC29H HAT) is on the hardware UART. A ZED-F9P on USB is found
    by its USB vendor id, not by guessing an ``ACM`` number: the RoboHAT and the ultrasonic MCU
    are ``ACM`` ports too (audit SAF-S1). Never a port another driver owns.
    """
    owned = ports_owned_by_other_drivers(ports)
    if mode == GpsMode.F9P_USB:
        for info in ports if ports is not None else list_serial_ports():
            if info.vid == UBLOX_VID and _real(info.device) not in owned:
                return info.device
        return None
    for device in _UART_DEVICES:
        if os.path.exists(device) and _real(device) not in owned:
            return device
    return None


# ------------------------------------------------------------------------------ the worker


class _IoWorker:
    """One daemon thread that runs blocking calls one at a time.

    A daemon thread, not a ``ThreadPoolExecutor``: a tty that never answers must not be able to
    hold the interpreter open at exit. Between jobs it holds no reference to the job it ran, so an
    idle worker never keeps the driver that owns it (and the driver's port) alive.
    """

    def __init__(self, name: str) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], Any]) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((future, fn))
        return future

    def close(self) -> None:
        """Let the thread end after the job it is on (if any)."""
        self._jobs.put(None)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, fn = job
            job = None
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except BaseException as exc:  # delivered to the future, never lost
                    future.set_exception(exc)
            future = fn = None  # nothing is held while the thread waits for the next job


# ------------------------------------------------------------------------------ the driver


@dataclass
class GPSDriverConfig:
    mode: GpsMode = GpsMode.NEO8M_UART
    # Serial/USB configuration (used when not in SIM_MODE). ``usb_device`` None: find the
    # receiver by USB identity (a fixed ``ACM`` number may be the RoboHAT's port).
    usb_device: str | None = None
    uart_device: str = "/dev/ttyAMA0"  # common UART on Pi
    baudrate: int = 9600


class GPSDriver(HardwareDriver):
    """GPS hardware driver.

    Methods:
    - read_position() -> GpsReading | None
    """

    #: The longest ``read_position`` waits for the hardware before returning the last reading.
    READ_DEADLINE_S = 0.5
    #: NMEA window of one read (the first read of a session gets longer to acquire).
    FIRST_WINDOW_S = 1.5
    WINDOW_S = 0.75
    #: After a probe pass found no receiver, do not probe again for this long.
    AUTODETECT_RETRY_S = 15.0
    #: After a read error on a port that worked, reopen it after this long.
    REOPEN_DELAY_S = 1.0
    #: How long a freshly opened port is read to decide it streams NMEA.
    PROBE_WINDOW_S = 1.5
    #: A port that delivers no valid NMEA for this long is dropped and probed again.
    SILENCE_REPROBE_S = 5.0
    #: A GST (of the previous epoch) this old no longer describes the fix.
    GST_MAX_AGE_S = 2.0
    #: An epoch that is still missing a kind the receiver normally sends is reported as it is
    #: only once its position sentence is this old; before that the read ends without it and the
    #: next read goes on waiting.
    EPOCH_GRACE_S = 0.5
    #: An RTK fix keeps its typical accuracy only if its corrections are at most this old.
    RTK_MAX_CORRECTION_AGE_S = 30.0
    #: Below this speed over ground the course is noise and is not reported as a heading.
    COURSE_MIN_SPEED_MS = 0.5
    #: The most one ``read`` takes from the port.
    READ_CHUNK = 4096
    #: No gpsd retry sooner than this after one that found nothing.
    GPSD_RETRY_S = 5.0

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        super().__init__(config=config)
        cfg = config or {}
        self._mono = monotonic
        self._wall = wall_clock
        self.cfg = GPSDriverConfig(
            mode=GpsMode(cfg.get("mode", GpsMode.NEO8M_UART)),
            usb_device=cfg.get("usb_device"),
            uart_device=cfg.get("uart_device", "/dev/ttyAMA0"),
            baudrate=int(cfg.get("baudrate", 9600)),
        )
        self._last_read: GpsReading | None = None
        self._last_read_mono: float | None = None
        self._sim_counter: int = 0
        # Real hardware handles (lazy). Only the worker thread touches them while a read is on.
        self._serial = None
        self._stream: _Stream | None = None
        self._first_read_done = False
        self._worker: _IoWorker | None = None
        self._worker_finalizer: weakref.finalize | None = None
        self._inflight: concurrent.futures.Future | None = None
        self._closing = False
        self._next_probe_at = 0.0
        self._next_gpsd_at = 0.0
        # The receiver's time line. It outlives a port that was reopened after an error (a USB
        # receiver that re-enumerates keeps its clock, and a frozen one must not be taken as fresh
        # again by the reopen); it starts afresh after silence or ``start()``.
        self._clock = EpochClock()
        self._gpsd_clock = EpochClock(period_s=None)
        #: Epochs refused as repeated, backwards or too far ahead (a frozen / replayed stream).
        self.epochs_refused = 0
        #: Estimated delay of the newest reading behind the receiver's own clock, seconds.
        self.lag_s = 0.0
        # Last observed NMEA sentences (for diagnostics)
        self._last_nmea: dict[str, str] = {}
        # Rate-limit state for hardware failure diagnostics (see _diag)
        self._diag_last_ts: float | None = None
        self._diag_last_reason: str | None = None
        # Cached baudrates to try for different modules
        self._baud_candidates = [self.cfg.baudrate]
        if self.cfg.mode in (GpsMode.F9P_USB, GpsMode.F9P_UART):
            # F9P often uses higher baud
            for b in (115200, 38400, 9600):
                if b not in self._baud_candidates:
                    self._baud_candidates.append(b)
        elif self.cfg.mode == GpsMode.LC29H_UART:
            # LC29H defaults to 115200 baud
            for b in (115200, 460800, 9600):
                if b not in self._baud_candidates:
                    self._baud_candidates.append(b)
        else:
            for b in (9600, 38400):
                if b not in self._baud_candidates:
                    self._baud_candidates.append(b)

    async def initialize(self) -> None:  # noqa: D401
        # In SIM_MODE, we don't touch hardware
        if is_simulation_mode():
            self.initialized = True
            return
        # Lazy import serial only when needed
        try:
            import serial  # type: ignore  # noqa: F401
        except Exception:
            # If pyserial isn't available, keep initialized False to avoid runtime errors
            self.initialized = True  # allow running even without serial in CI
            return
        # Defer opening until the first read; some platforms require permissions
        self.initialized = True

    async def start(self) -> None:  # noqa: D401
        if not self.initialized:
            await self.initialize()
        # In real mode the port is opened by the first read, on the worker thread
        self._closing = False
        self._clock = EpochClock()
        self.running = True

    async def stop(self) -> None:  # noqa: D401
        self.running = False
        self._closing = True
        self._worker = None
        finalizer, self._worker_finalizer = self._worker_finalizer, None
        if finalizer is not None:
            finalizer()  # lets the worker thread end after the job it is on
        # A read that is still on the worker closes the port itself when it ends (it checks
        # ``_closing``); closing it from this thread under a blocked read would race.
        if not self._busy():
            self._drop_port()

    async def health_check(self) -> dict[str, Any]:  # noqa: D401
        stream = self._stream
        return {
            "driver": "gps",
            "mode": self.cfg.mode.value,
            "initialized": self.initialized,
            "running": self.running,
            "last_read_age_s": (
                (self._mono() - self._last_read_mono) if self._last_read_mono is not None else None
            ),
            "simulation": is_simulation_mode(),
            "device": getattr(self._serial, "port", None),
            "read_in_flight": self._busy(),
            "last_sentence_age_s": (
                self._mono() - stream.last_valid_t if stream is not None else None
            ),
            "epochs_refused": self.epochs_refused,
            "lag_s": self.lag_s,
        }

    def _diag(self, reason: str, message: str) -> None:
        """Rate-limited WARNING for hardware GPS failures (~once/30s per reason).

        read_position() runs at ~1 Hz, so unconditional logging would flood the
        journal. This surfaces *why* GPS is blank (permission / missing device /
        no NMEA) — otherwise these failures are swallowed and the dashboard just
        shows "SEARCHING" with no clue in the logs.
        """
        now = self._mono()
        if (
            reason == self._diag_last_reason
            and self._diag_last_ts is not None
            and (now - self._diag_last_ts) < 30.0
        ):
            return
        self._diag_last_reason = reason
        self._diag_last_ts = now
        logger.warning("GPS %s: %s", reason, message)

    # ------------------------------------------------------------------ the event-loop side

    async def read_position(self) -> GpsReading | None:
        """Read current GPS position.

        Returns a GpsReading or None if not available. In SIM_MODE this
        generates a deterministic position around a fixed coordinate suitable
        for tests and local development. On hardware it never blocks the event loop and
        never waits more than ``READ_DEADLINE_S``: it returns the newest reading the worker
        has produced (an old one is old, its timestamp says so).
        """
        if not self.initialized:
            return None

        if is_simulation_mode():
            # Deterministic walking pattern near Googleplex-like coords
            base_lat = 37.4220
            base_lon = -122.0841
            # Small oscillation to emulate movement
            d = (self._sim_counter % 20) * 0.000001
            lat = base_lat + d
            lon = base_lon - d
            self._sim_counter += 1
            is_rtk = self.cfg.mode in (GpsMode.F9P_USB, GpsMode.F9P_UART, GpsMode.LC29H_UART)
            acc = 0.6 if is_rtk else 3.0
            sats = 18 if is_rtk else 8
            rtk = "RTK_FIXED" if is_rtk else None
            hdop = 0.5 if is_rtk else 2.5
            reading = GpsReading(
                latitude=lat,
                longitude=lon,
                altitude=10.0,
                accuracy=acc,
                satellites=sats,
                mode=self.cfg.mode,
                rtk_status=rtk,
                hdop=hdop,
            )
            self._last_read = reading
            self._last_read_mono = self._mono()
            return reading

        if self._closing or self._busy():
            return self._last_read
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        future = self._submit(self._read_blocking)

        def _signal(_future: concurrent.futures.Future) -> None:
            try:
                loop.call_soon_threadsafe(done.set)
            except RuntimeError:  # the loop is gone (shutdown): nobody is waiting
                pass

        future.add_done_callback(_signal)
        try:
            async with asyncio.timeout(self.READ_DEADLINE_S):
                await done.wait()
        except TimeoutError:
            pass  # the worker keeps going; the next call sees it busy and returns at once
        return self._last_read

    def _busy(self) -> bool:
        return self._inflight is not None and not self._inflight.done()

    def _submit(self, fn: Callable[[], Any]) -> concurrent.futures.Future:
        if self._worker is None:
            self._worker = _IoWorker("gps-io")
            # A driver that is dropped without ``stop()`` must not leave its thread behind.
            self._worker_finalizer = weakref.finalize(self, self._worker.close)
        future = self._worker.submit(fn)
        # A weak method: the worker's future must not keep the driver (and its port) alive.
        adopt = weakref.WeakMethod(self._adopt)

        def _adopt_if_alive(done: concurrent.futures.Future) -> None:
            method = adopt()
            if method is not None:
                method(done)

        future.add_done_callback(_adopt_if_alive)
        self._inflight = future
        return future

    def _adopt(self, future: concurrent.futures.Future) -> None:
        """Take the worker's result (runs on the worker thread, or at once if it is done)."""
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:  # _read_blocking reports its own failures; this is a safety net
            self._diag("read-error", f"GPS read failed: {error!r}")
            return
        reading = future.result()
        if reading is not None:
            self._last_read = reading
            self._last_read_mono = self._mono()

    # ------------------------------------------------------------------ the worker side

    def _read_blocking(self) -> GpsReading | None:
        """One hardware read. Runs on the worker thread only; may block, must not raise."""
        try:
            if self._serial is None:
                if self._mono() < self._next_probe_at or not self._open_port():
                    return None
            reading = self._read_nmea()
            if reading is not None:
                self._first_read_done = True
                return reading
            # Optional gpsd fallback (if service is present on localhost:2947)
            if self._mono() >= self._next_gpsd_at:
                reading = self._read_from_gpsd(timeout_sec=0.5 if self._first_read_done else 1.0)
                if reading is not None:
                    self._first_read_done = True
                    return reading
                self._next_gpsd_at = self._mono() + self.GPSD_RETRY_S
            return None
        except _PortSilent as exc:
            # The port is open but says nothing (a receiver reconfigured to another baud, a tty
            # another opener changed): drop it and probe again, as for a read error. A receiver
            # that was silent that long may have restarted, so its time line starts afresh.
            self._diag("port-silent", str(exc))
            self._drop_port()
            self._clock = EpochClock()
            self._next_probe_at = self._mono() + self.REOPEN_DELAY_S
            return None
        except Exception as exc:
            # On errors, keep the last reading; surface the cause (rate-limited) and reopen the
            # port next time (a USB receiver that was unplugged is a new device when it returns).
            self._diag("read-error", f"GPS read failed: {exc!r}")
            self._drop_port()
            self._next_probe_at = self._mono() + self.REOPEN_DELAY_S
            return None
        finally:
            if self._closing:
                self._drop_port()

    def _drop_port(self) -> None:
        ser, self._serial = self._serial, None
        self._stream = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _candidates(self) -> tuple[list[str], list[str]]:
        """``(devices to probe in order, devices skipped because another driver owns them)``."""
        ports, enumerated = _enumerate_serial_ports()
        owned, complete = _other_driver_ports(ports)
        identified = {info.device for info in ports}
        wanted: list[str] = []

        def add(device: str | None) -> None:
            if device and device not in wanted:
                wanted.append(device)

        add(os.environ.get("GPS_DEVICE"))
        add(self.cfg.usb_device if self.cfg.mode == GpsMode.F9P_USB else self.cfg.uart_device)
        for info in ports:
            if info.vid == UBLOX_VID:
                add(info.device)
        for device in (*_UART_DEVICES, *_USB_SERIAL_DEVICES):
            add(device)
        for device in sorted(glob.glob("/dev/ttyACM*")):
            # An unidentified ACM device may be another board (the RoboHAT, the ultrasonic MCU):
            # probe it only if its identity was checked and the configuration was readable.
            if enumerated and complete and device in identified:
                add(device)
        for device in sorted(glob.glob("/dev/ttyUSB*")):
            add(device)
        usable = [d for d in wanted if _real(d) not in owned]
        return usable, [d for d in wanted if _real(d) in owned]

    def _open_port(self) -> bool:
        """Find the receiver: open candidates (exclusively) until one streams NMEA."""
        import serial  # type: ignore

        candidates, skipped = self._candidates()
        if skipped:
            self._diag(
                "port-owned",
                f"not probing {skipped}: configured for or identified as another driver",
            )
        last_open_error: Exception | None = None
        opened_but_no_nmea = False
        for dev in candidates:
            for baud in self._baud_candidates:
                if self._closing:
                    return False  # stopped: no further port is opened (or locked) for nothing
                try:
                    ser = serial.Serial(dev, baud, timeout=0.25, exclusive=True)  # type: ignore
                except Exception as exc:  # hardware dependent: missing, busy, no permission
                    last_open_error = exc
                    break  # an open failure does not depend on the baud rate
                try:
                    streams = self._streams_nmea(ser)
                except Exception as exc:
                    last_open_error = exc
                    streams = False
                if streams:
                    self._serial = ser
                    self._stream = _Stream(self._mono())
                    self.cfg.baudrate = baud
                    self._next_probe_at = 0.0
                    return True
                opened_but_no_nmea = True
                try:
                    ser.close()
                except Exception:
                    pass
        # Not found: say why (rate-limited), and do not probe again soon.
        self._next_probe_at = self._mono() + self.AUTODETECT_RETRY_S
        if isinstance(last_open_error, PermissionError):
            self._diag(
                "serial-permission",
                f"cannot open GPS serial ({last_open_error!r}); add the service "
                "user to the 'dialout' group and restart the service",
            )
        elif opened_but_no_nmea:
            self._diag(
                "no-nmea",
                f"opened a serial port ({self.cfg.mode.value}) but saw no NMEA "
                "($GxGGA/$GxRMC) — check module power/wiring and baud "
                f"(tried {self._baud_candidates})",
            )
        else:
            self._diag(
                "no-device",
                f"no GPS serial device streaming NMEA (tried {candidates} at "
                f"baud {self._baud_candidates}; last error: {last_open_error!r})",
            )
        return False

    def _take(self, ser: Any) -> bytes:
        """What the port has now (at most ``READ_CHUNK``); waits for one byte if it has none."""
        waiting = ser.in_waiting
        return ser.read(min(waiting, self.READ_CHUNK) if waiting else 1)

    def _streams_nmea(self, ser: Any) -> bool:
        """Does ``ser`` carry NMEA? A complete sentence with a good checksum within the window."""
        reader = NmeaLineReader()
        deadline = self._mono() + self.PROBE_WINDOW_S
        while not self._closing and self._mono() < deadline:
            for text in reader.feed(self._take(ser)):
                if text.startswith("$G") and nmea_checksum_ok(text):
                    return True
        return False

    def _judge(self, st: _Stream, epoch: _Epoch) -> bool:
        """Does ``epoch`` just become a new reading? An epoch is judged once, when it is first
        complete or closed (whichever comes first). A fix must be a new time; a no-fix just not a
        repeat."""
        if epoch.judged:
            return False
        epoch.judged = True
        if epoch.gga is None and epoch.rmc is None:
            return False  # a GST alone says nothing about a position
        if classify(epoch, self.RTK_MAX_CORRECTION_AGE_S).is_fix:
            verdict = self._clock.accept(epoch.key, epoch.t_pos)
            if not verdict.accepted:
                self.epochs_refused += 1
                self._diag(
                    "stale-epoch",
                    f"NMEA epoch {epoch.key:.2f} s of the UTC day is not a new fix "
                    f"({verdict.reason}): a repeated, replayed or too-far-ahead time",
                )
                return False
            epoch.delay_s = verdict.delay_s
        elif epoch.key == st.last_nofix_key:
            return False
        else:
            st.last_nofix_key = epoch.key
        st.last_accept_t = epoch.t_pos
        return True

    def _ingest(self, st: _Stream, text: str, now: float) -> list[_Epoch]:
        """Feed one candidate sentence read at host time ``now``; the epochs it closed."""
        if not (text.startswith("$G") and nmea_checksum_ok(text)):
            return []
        st.last_valid_t = now
        kind = text[3:6]
        payload: Gga | Rmc | float | None
        if kind == "GGA":
            payload = parse_gga(text)
            utc = payload.utc_s if payload is not None else None
        elif kind == "RMC":
            payload = parse_rmc(text)
            utc = payload.utc_s if payload is not None else None
        elif kind == "GST":
            payload = parse_gst_sigma(text)
            fields = text.split(",", 2)
            utc = parse_nmea_utc(fields[1]) if len(fields) > 2 else None
        else:
            return []
        self._last_nmea[kind] = text
        if payload is None and kind != "GST":
            return []
        repeats = st.asm.repeats
        closed = st.asm.add(kind, utc, payload, now)
        self.epochs_refused += st.asm.repeats - repeats
        return closed

    def _read_nmea(self) -> GpsReading | None:
        """Read NMEA for one window and build a reading from the newest epoch it has.

        An epoch is the burst of sentences that share one UTC field. What is waiting in the port
        is read first and every epoch in it judged in order, so the reading is the NEWEST complete
        one, never the oldest of a backlog; the read waits for more only while it has no epoch
        yet. An epoch is complete when it has every kind this receiver is known to send (GGA,
        RMC for speed and course, GST for accuracy), so the accuracy belongs to the position it is
        reported with; otherwise it ends when the next epoch starts or the window closes.
        """
        ser, st = self._serial, self._stream
        if ser is None or st is None:
            return None
        deadline = self._mono() + (self.WINDOW_S if self._first_read_done else self.FIRST_WINDOW_S)
        best: _Epoch | None = None
        while not self._closing:
            cur = st.asm.cur
            if cur is not None and not cur.judged and st.asm.complete(cur, self._mono()):
                if self._judge(st, cur):
                    best = cur
            if not ser.in_waiting and best is not None:
                break  # nothing more is queued: this is the newest there is
            if self._mono() >= deadline:
                break
            data = self._take(ser)
            now = self._mono()
            for text in st.reader.feed(data):
                for epoch in self._ingest(st, text, now):
                    if self._judge(st, epoch):
                        best = epoch
        cur = st.asm.cur
        if (
            best is None
            and cur is not None
            and not cur.judged
            and not self._closing
            and self._mono() - cur.t_pos >= self.EPOCH_GRACE_S
        ):
            # The window closed on an epoch that is still missing a sentence and has waited long
            # enough for it: report what it has.
            if self._judge(st, cur):
                best = cur
        if best is None:
            now = self._mono()
            if now - st.last_valid_t >= self.SILENCE_REPROBE_S:
                raise _PortSilent(
                    f"no valid NMEA for {now - st.last_valid_t:.1f} s on "
                    f"{getattr(ser, 'port', '?')} at {self.cfg.baudrate} baud"
                )
            if now - st.last_accept_t >= self.SILENCE_REPROBE_S:
                self._diag(
                    "frozen",
                    "valid NMEA keeps arriving but nothing new for "
                    f"{now - st.last_accept_t:.1f} s: the receiver's UTC is not advancing "
                    "(a frozen or replayed stream)",
                )
            return None
        return self._reading_from(st, best)

    def _sigma_for(self, st: _Stream, epoch: _Epoch, status: str) -> float | None:
        """The GST sigma that describes ``epoch``: its own, else the previous epoch's if recent."""
        if epoch.gst_seen:
            return epoch.gst_sigma
        prev = st.last_delivered
        if (
            prev is None
            or not prev.gst_seen
            or prev.gst_sigma is None
            or not 0.0 < epoch.key - prev.key <= self.GST_MAX_AGE_S
            or self._mono() - prev.gst_t > self.GST_MAX_AGE_S
        ):
            return None
        prev_status = classify(prev, self.RTK_MAX_CORRECTION_AGE_S).status
        if _STATUS_RANK[status] < _STATUS_RANK[prev_status]:
            return None  # the fix got worse: the old sigma described a better one
        return prev.gst_sigma

    def _reading_from(self, st: _Stream, epoch: _Epoch) -> GpsReading:
        """The ``GpsReading`` of an accepted epoch, dated by the receiver's own epoch."""
        got = classify(epoch, self.RTK_MAX_CORRECTION_AGE_S)
        sigma = self._sigma_for(st, epoch, got.status) if got.is_fix else None
        accuracy = (
            combine_accuracy(sigma, got.status, assume_typical=got.typical_ok)
            if got.is_fix
            else None
        )
        speed = epoch.rmc.speed_mps if got.is_fix and epoch.rmc is not None else None
        course = epoch.rmc.course_deg if got.is_fix and epoch.rmc is not None else None
        heading = course if speed is not None and speed >= self.COURSE_MIN_SPEED_MS else None
        gga = epoch.gga
        age = max(0.0, self._mono() - epoch.t_pos) + epoch.delay_s
        st.last_delivered = epoch
        self.lag_s = age
        return GpsReading(
            latitude=got.lat,
            longitude=got.lon,
            altitude=got.altitude_m,
            accuracy=accuracy,
            heading=heading,
            speed=speed,
            satellites=gga.satellites if gga is not None else None,
            hdop=gga.hdop if gga is not None else None,
            mode=self.cfg.mode,
            rtk_status=got.status,
            timestamp=self._wall() - timedelta(seconds=age),
        )

    def get_last_nmea(self) -> dict[str, str]:
        """Return a shallow copy of last seen NMEA sentences for diagnostics."""
        try:
            return dict(self._last_nmea)
        except Exception:
            return {}

    def _read_from_gpsd(self, timeout_sec: float = 0.5) -> GpsReading | None:
        """Try reading a TPV report from gpsd if available.

        Uses a raw TCP socket to 127.0.0.1:2947 to avoid external deps. Blocking: only ever
        called on the worker thread. Returns a GpsReading for the first TPV that is a fix
        (``mode`` >= 2, a finite in-range position, a ``time`` that is later than the last one
        accepted), else None. ``accuracy`` is gpsd's ``eph`` (an error estimate in metres);
        gpsd's TPV carries no HDOP.
        """
        host = os.environ.get("GPSD_HOST", "127.0.0.1")
        port = int(os.environ.get("GPSD_PORT", "2947"))
        try:
            with socket.create_connection((host, port), timeout=0.3) as s:
                s.settimeout(timeout_sec)
                # Issue WATCH command
                s.sendall(b'?WATCH={"enable":true,"json":true}\n')
                start = self._mono()
                buf = b""
                while self._mono() - start < timeout_sec and len(buf) < 65536:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    # gpsd streams JSON objects delimited by newlines
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8", errors="ignore"))
                        except Exception:
                            continue
                        if isinstance(obj, dict) and obj.get("class") == "TPV":
                            reading = self._reading_from_tpv(obj, self._mono())
                            if reading is not None:
                                return reading
                return None
        except Exception:
            return None

    def _reading_from_tpv(self, obj: dict[str, Any], t: float) -> GpsReading | None:
        """A fix ``GpsReading`` from a gpsd TPV object read at host time ``t``, else ``None``."""

        def number(key: str) -> float | None:
            value = obj.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value) if math.isfinite(value) else None

        mode = obj.get("mode")
        status = _GPSD_STATUS.get(obj.get("status", 0), "NO_FIX")
        lat, lon = number("lat"), number("lon")
        if (
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 2
            or status not in FIX_STATUSES
            or lat is None
            or lon is None
            or abs(lat) > 90.0
            or abs(lon) > 180.0
        ):
            return None
        stamp = parse_stamp(obj.get("time"))
        if stamp is None:
            return None  # a fix that cannot be dated cannot be called fresh
        verdict = self._gpsd_clock.accept(stamp.timestamp(), t)
        if not verdict.accepted:
            self.epochs_refused += 1
            self._diag(
                "stale-epoch",
                f"gpsd TPV time {obj.get('time')!r} is not a new fix ({verdict.reason})",
            )
            return None
        eph = number("eph")
        sigma = eph if eph is not None and eph > 0 else None
        speed, track = number("speed"), number("track")
        speed = speed if speed is not None and speed >= 0.0 else None
        track = track if track is not None and 0.0 <= track <= 360.0 else None
        age = max(0.0, self._mono() - t) + verdict.delay_s
        return GpsReading(
            latitude=lat,
            longitude=lon,
            altitude=number("alt"),
            accuracy=combine_accuracy(sigma, status, assume_typical=False),
            speed=speed,
            heading=track if speed is not None and speed >= self.COURSE_MIN_SPEED_MS else None,
            mode=self.cfg.mode,
            rtk_status=status,
            timestamp=self._wall() - timedelta(seconds=age),
        )


__all__ = [
    "FIX_STATUSES",
    "EpochClock",
    "EpochVerdict",
    "GPSDriver",
    "GPSDriverConfig",
    "Gga",
    "NmeaLineReader",
    "Rmc",
    "SerialPortInfo",
    "classify",
    "combine_accuracy",
    "fix_status_from_quality",
    "fix_status_from_rmc",
    "list_serial_ports",
    "nmea_checksum_ok",
    "parse_gga",
    "parse_gst_sigma",
    "parse_nmea_coordinate",
    "parse_nmea_utc",
    "parse_rmc",
    "ports_owned_by_other_drivers",
    "preferred_gps_device",
    "reading_age_s",
]
