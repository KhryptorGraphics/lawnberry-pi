"""Host side of the framed serial protocol to the ultrasonic MCU (protocol version 1).

Wire format, message set and rationale: ``firmware/ultrasonic-mcu/README.md``. This is an
independent implementation (``binascii.crc_hqx`` for the CRC, a ``bytearray`` decoder) of what
``firmware/ultrasonic-mcu/usonic_proto.py`` implements for the MCU;
``tests/unit/test_ultrasonic_mcu_protocol.py`` proves the two agree byte for byte.

Frame (little-endian)::

    0xA5 0x5A | VER | TYPE | SEQ | LEN | PAYLOAD (LEN) | CRC16-CCITT-FALSE (VER..PAYLOAD)

Pure data: no I/O, no clock. Meaning (blind zone, staleness, verdicts) is decided in
``jsn_sr04t_driver.py`` from the owner's configuration, never here.
"""

from __future__ import annotations

import binascii
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

SYNC0 = 0xA5
SYNC1 = 0x5A
_SYNC = bytes((SYNC0, SYNC1))
PROTO_VERSION = 1
MAX_PAYLOAD = 48
HEADER_LEN = 6
CRC_LEN = 2
MAX_FRAME = HEADER_LEN + MAX_PAYLOAD + CRC_LEN


class FrameType(IntEnum):
    # host -> MCU
    HELLO = 0x01
    CONFIG = 0x02
    PING = 0x03
    # MCU -> host
    INFO = 0x81
    CONFIG_ACK = 0x82
    READING = 0x83
    PONG = 0x84
    NAK = 0x8F


class EchoStatus(IntEnum):
    """What the echo line did (not what it means)."""

    OK = 0  # complete echo pulse; echo_us is its width
    NO_EDGE = 1  # no pulse in the window: open air, dead sensor and wiring are indistinguishable
    NO_FALL = 2  # rising edge, no falling edge within the window (pulse longer than the window)
    STUCK = 3  # line high before the trigger, or a HAL fault
    DISABLED = 4  # channel masked off


class McuHealth(IntEnum):
    ACTIVE = 0
    QUIET = 1  # many consecutive silent measurements (still fired every cycle)
    FAULT = 2
    DISABLED = 3


class NakReason(IntEnum):
    BAD_VERSION = 1
    BAD_LENGTH = 2
    UNKNOWN_TYPE = 3
    BAD_VALUE = 4


class ProtocolError(ValueError):
    """A frame was well formed (CRC ok) but its payload is not valid for its type."""


_F_HELLO = "<BB"
_F_CONFIG = "<BHHB"
_F_PING = "<I"
_F_PONG = "<II"
_F_INFO = "<BBBBIIBB"
_F_READING = "<BBBBIHI"
_F_NAK = "<BB"

PAYLOAD_SIZES: dict[int, int] = {
    FrameType.HELLO: struct.calcsize(_F_HELLO),
    FrameType.CONFIG: struct.calcsize(_F_CONFIG),
    FrameType.PING: struct.calcsize(_F_PING),
    FrameType.INFO: struct.calcsize(_F_INFO),
    FrameType.CONFIG_ACK: struct.calcsize(_F_CONFIG),
    FrameType.READING: struct.calcsize(_F_READING),
    FrameType.PONG: struct.calcsize(_F_PONG),
    FrameType.NAK: struct.calcsize(_F_NAK),
}


def crc16(data: bytes | bytearray | memoryview, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). ``crc16(b"123456789") == 0x29B1``."""
    return binascii.crc_hqx(bytes(data), crc)


def encode_frame(ftype: int, seq: int, payload: bytes = b"", version: int = PROTO_VERSION) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload too long")
    body = bytes((version & 0xFF, int(ftype) & 0xFF, seq & 0xFF, len(payload))) + bytes(payload)
    crc = crc16(body)
    return _SYNC + body + bytes((crc & 0xFF, crc >> 8))


@dataclass(frozen=True, slots=True)
class Frame:
    version: int
    ftype: int
    seq: int
    payload: bytes


class FrameDecoder:
    """Streaming decoder. ``feed(data, now)`` returns every complete, CRC-valid frame.

    Resynchronisation: on a bad length or CRC exactly one byte is dropped and the scan resumes at
    the next sync word, so a corrupted frame costs only itself. If the line has been quiet for
    longer than ``partial_timeout`` (same unit as ``now``) while an incomplete frame is pending,
    nothing legitimate can still be arriving (a real frame is contiguous): the rest is scanned for
    complete frames hidden behind the junk and then discarded. Without this a corrupted LEN byte
    could hold newer, good frames hostage until enough further bytes arrive. Call
    ``feed(b"", now)`` regularly so that timeout can fire. Input is scanned in frame-sized pieces,
    so the buffer never holds more than one incomplete frame (``<= MAX_FRAME`` bytes) however much
    is fed at once.
    """

    def __init__(self, partial_timeout: float | None = None) -> None:
        self._buf = bytearray()
        self._partial_timeout = partial_timeout
        self._last_rx: float | None = None
        self.crc_errors = 0
        self.bad_length = 0
        self.garbage = 0
        self.partial_timeouts = 0

    def reset(self) -> None:
        self._buf.clear()
        self._last_rx = None

    @property
    def pending(self) -> int:
        return len(self._buf)

    def feed(self, data: bytes | bytearray | memoryview, now: float | None = None) -> list[Frame]:
        out: list[Frame] = []
        if (
            now is not None
            and self._partial_timeout is not None
            and self._buf
            and self._last_rx is not None
            and now - self._last_rx > self._partial_timeout
        ):
            self._flush(out)
        if data:
            view = memoryview(data)
            for i in range(0, len(view), MAX_FRAME):
                self._buf.extend(view[i : i + MAX_FRAME])
                self._scan(out)
            self._last_rx = now
        return out

    def _flush(self, out: list[Frame]) -> None:
        self.partial_timeouts += 1
        while self._buf:
            self._scan(out)
            if not self._buf:
                break
            self.garbage += 1
            del self._buf[0]

    def _scan(self, out: list[Frame]) -> None:
        buf = self._buf
        while True:
            i = buf.find(_SYNC)
            if i < 0:
                keep = 1 if buf and buf[-1] == SYNC0 else 0
                self.garbage += len(buf) - keep
                del buf[: len(buf) - keep]
                return
            if i:
                self.garbage += i
                del buf[:i]
            if len(buf) < HEADER_LEN:
                return
            length = buf[5]
            if length > MAX_PAYLOAD:
                self.bad_length += 1
                del buf[0]
                continue
            total = HEADER_LEN + length + CRC_LEN
            if len(buf) < total:
                return
            if crc16(buf[2 : total - 2]) != (buf[total - 2] | (buf[total - 1] << 8)):
                self.crc_errors += 1
                del buf[0]
                continue
            out.append(Frame(buf[2], buf[3], buf[4], bytes(buf[HEADER_LEN : HEADER_LEN + length])))
            del buf[:total]


# ---- payloads -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Info:
    fw_major: int
    fw_minor: int
    proto_version: int
    sensor_count: int
    boot_id: int
    uptime_ms: int
    reset_reason: int
    enable_mask: int


@dataclass(frozen=True, slots=True)
class McuConfig:
    """Scheduling parameters the host writes and the MCU echoes back (confirmed write)."""

    enable_mask: int
    echo_window_ms: int
    gap_ms: int
    quiet_after: int


@dataclass(frozen=True, slots=True)
class Reading:
    sensor: int
    status: int
    health: int
    fails: int
    echo_us: int
    cycle: int
    t_ms: int


@dataclass(frozen=True, slots=True)
class Pong:
    nonce: int
    uptime_ms: int


@dataclass(frozen=True, slots=True)
class Nak:
    ref_type: int
    reason: int


def build_hello(flags: int = 0) -> bytes:
    return struct.pack(_F_HELLO, PROTO_VERSION, flags)


def build_config(cfg: McuConfig) -> bytes:
    return struct.pack(
        _F_CONFIG,
        cfg.enable_mask,
        cfg.echo_window_ms,
        cfg.gap_ms,
        cfg.quiet_after,
    )


def build_ping(nonce: int) -> bytes:
    return struct.pack(_F_PING, nonce & 0xFFFFFFFF)


def parse_payload(frame: Frame) -> Info | McuConfig | Reading | Pong | Nak:
    """Decode an MCU-to-host payload. Raises ``ProtocolError`` for a wrong length or type."""
    expected = PAYLOAD_SIZES.get(frame.ftype)
    if expected is None or frame.ftype < 0x80:
        raise ProtocolError(f"unexpected frame type 0x{frame.ftype:02x} from the MCU")
    if len(frame.payload) != expected:
        raise ProtocolError(
            f"frame 0x{frame.ftype:02x} payload is {len(frame.payload)} bytes, expected {expected}"
        )
    p = frame.payload
    if frame.ftype == FrameType.INFO:
        return Info(*struct.unpack(_F_INFO, p))
    if frame.ftype == FrameType.CONFIG_ACK:
        return McuConfig(*struct.unpack(_F_CONFIG, p))
    if frame.ftype == FrameType.READING:
        return Reading(*struct.unpack(_F_READING, p))
    if frame.ftype == FrameType.PONG:
        return Pong(*struct.unpack(_F_PONG, p))
    return Nak(*struct.unpack(_F_NAK, p))


def decode_stream(data: Iterable[int] | bytes) -> list[Frame]:
    """Convenience for tests and scripts: decode a complete capture in one go."""
    return FrameDecoder().feed(bytes(data))


__all__ = [
    "PROTO_VERSION",
    "MAX_PAYLOAD",
    "MAX_FRAME",
    "FrameType",
    "EchoStatus",
    "McuHealth",
    "NakReason",
    "ProtocolError",
    "Frame",
    "FrameDecoder",
    "Info",
    "McuConfig",
    "Reading",
    "Pong",
    "Nak",
    "crc16",
    "encode_frame",
    "build_hello",
    "build_config",
    "build_ping",
    "parse_payload",
    "decode_stream",
]
