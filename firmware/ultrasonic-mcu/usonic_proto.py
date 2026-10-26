"""Wire protocol between the LawnBerry host and the ultrasonic MCU (version 1).

Pure Python with no hardware imports, written to run unchanged on CPython (host tests) and on
CircuitPython (RP2040): only ``struct.pack/unpack/calcsize``, ``bytes`` and plain loops are used
(no ``struct.Struct``, no ``bytearray`` deletion, no ``dataclasses``/``typing``). The host has its
own independent implementation in ``backend/src/drivers/sensors/ultrasonic_mcu_protocol.py``;
``tests/unit/test_ultrasonic_mcu_protocol.py`` checks that both produce identical bytes (golden
vectors and randomised differential tests).

Frame (all multi-byte fields little-endian)::

    0xA5 0x5A | VER | TYPE | SEQ | LEN | PAYLOAD (LEN bytes) | CRC16 (2 bytes)

* ``VER``  protocol major version (``PROTO_VERSION``). Covered by the CRC, so a frame from another
  protocol version is still recognisable (and refused) instead of being mistaken for corruption.
* ``SEQ``  per-sender counter, +1 per frame sent, wraps at 256. Lets the receiver count lost frames.
* ``LEN``  payload length, ``0..MAX_PAYLOAD``.
* ``CRC16`` CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no final xor) over
  ``VER .. PAYLOAD``. ``crc16(b"123456789") == 0x29B1``.

A receiver resynchronises on a bad CRC or length by dropping ONE byte and scanning for the next
sync word, so one corrupted frame never costs more than itself (see ``Decoder``).

This module only describes bytes. What a status means for safety is decided on the host, never
here: the MCU reports raw echo widths and electrical outcomes.
"""

import struct

SYNC0 = 0xA5
SYNC1 = 0x5A
PROTO_VERSION = 1
MAX_PAYLOAD = 48
HEADER_LEN = 6  # SYNC0 SYNC1 VER TYPE SEQ LEN
CRC_LEN = 2
MAX_FRAME = HEADER_LEN + MAX_PAYLOAD + CRC_LEN

# Host -> MCU
T_HELLO = 0x01
T_CONFIG = 0x02
T_PING = 0x03
# MCU -> host
T_INFO = 0x81
T_CONFIG_ACK = 0x82
T_READING = 0x83
T_PONG = 0x84
T_NAK = 0x8F

# READING.status: what the line did, not what it means.
ST_OK = 0  # a complete echo pulse was measured; echo_us is its width
ST_NO_EDGE = 1  # no echo pulse within the window (open air, dead sensor and wiring look the same)
ST_NO_FALL = 2  # a rising edge but no falling edge within the window (pulse longer than the window)
ST_STUCK = 3  # the echo line was already high before the trigger (fault); HAL-dependent
ST_DISABLED = 4  # channel masked off by the enable mask

# READING.health (MCU view of the channel)
HEALTH_ACTIVE = 0
HEALTH_QUIET = 1  # many consecutive silent measurements (still fired every cycle)
HEALTH_FAULT = 2  # stuck-high line
HEALTH_DISABLED = 3

# NAK.reason
NAK_BAD_VERSION = 1
NAK_BAD_LENGTH = 2
NAK_UNKNOWN_TYPE = 3
NAK_BAD_VALUE = 4

# INFO.reset_reason
RESET_UNKNOWN = 0
RESET_POWER_ON = 1
RESET_WATCHDOG = 2
RESET_SOFTWARE = 3
RESET_PIN = 4

ECHO_NONE = 0  # echo_us when no pulse was measured

_F_HELLO = "<BB"
_F_CONFIG = "<BHHB"
_F_PING = "<I"
_F_PONG = "<II"
_F_INFO = "<BBBBIIBB"
_F_READING = "<BBBBIHI"
_F_NAK = "<BB"

PAYLOAD_SIZES = {
    T_HELLO: struct.calcsize(_F_HELLO),
    T_CONFIG: struct.calcsize(_F_CONFIG),
    T_PING: struct.calcsize(_F_PING),
    T_INFO: struct.calcsize(_F_INFO),
    T_CONFIG_ACK: struct.calcsize(_F_CONFIG),
    T_READING: struct.calcsize(_F_READING),
    T_PONG: struct.calcsize(_F_PONG),
    T_NAK: struct.calcsize(_F_NAK),
}

_crc_table = None


def _table():
    global _crc_table
    if _crc_table is None:
        table = []
        for i in range(256):
            crc = i << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
            table.append(crc)
        _crc_table = table
    return _crc_table


def crc16(data, crc=0xFFFF):
    """CRC-16/CCITT-FALSE of ``data`` (bytes-like), continuing from ``crc``."""
    table = _table()
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ table[((crc >> 8) ^ b) & 0xFF]
    return crc


def encode(ftype, seq, payload=b"", version=PROTO_VERSION):
    """Build one frame. A payload longer than ``MAX_PAYLOAD`` is a programming error."""
    n = len(payload)
    if n > MAX_PAYLOAD:
        raise ValueError("payload too long")
    body = bytes((version & 0xFF, ftype & 0xFF, seq & 0xFF, n)) + bytes(payload)
    crc = crc16(body)
    return bytes((SYNC0, SYNC1)) + body + bytes((crc & 0xFF, crc >> 8))


def _find_sync(buf):
    for i in range(len(buf) - 1):
        if buf[i] == SYNC0 and buf[i + 1] == SYNC1:
            return i
    return -1


class Decoder:
    """Streaming frame decoder: ``feed(bytes, now)`` returns the complete valid frames.

    Each frame is ``(version, ftype, seq, payload)``. Counters: ``crc_errors``, ``bad_length``,
    ``garbage`` (bytes discarded), ``partial_timeouts``. Input is scanned in frame-sized pieces, so
    the buffer never holds more than one incomplete frame (``<= MAX_FRAME`` bytes) however much is
    fed at once.

    ``now`` is any monotonically increasing number (ms on the MCU, seconds on the host) and
    ``partial_timeout`` is in the same unit. If the line has been quiet for longer than that while
    an incomplete frame is pending, nothing legitimate can still be arriving (a real frame is
    contiguous): whatever is left is scanned for complete frames hidden behind the junk and then
    discarded. Without this, one corrupted LEN byte could hold newer, good frames hostage until
    enough further bytes arrive. Call ``feed(b"", now)`` regularly so that timeout can fire.
    """

    def __init__(self, partial_timeout=None):
        self.buf = b""
        self.partial_timeout = partial_timeout
        self._last_rx = None
        self.crc_errors = 0
        self.bad_length = 0
        self.garbage = 0
        self.partial_timeouts = 0

    def reset(self):
        self.buf = b""
        self._last_rx = None

    def feed(self, data, now=None):
        out = []
        if (
            now is not None
            and self.partial_timeout is not None
            and self.buf
            and self._last_rx is not None
            and now - self._last_rx > self.partial_timeout
        ):
            self._flush(out)
        if data:
            data = bytes(data)
            for i in range(0, len(data), MAX_FRAME):
                self.buf = self.buf + data[i : i + MAX_FRAME]
                self._scan(out)
            self._last_rx = now
        return out

    def _flush(self, out):
        self.partial_timeouts += 1
        while self.buf:
            self._scan(out)
            if not self.buf:
                break
            self.garbage += 1
            self.buf = self.buf[1:]

    def _scan(self, out):
        while True:
            i = _find_sync(self.buf)
            if i < 0:
                # Keep a trailing SYNC0: the SYNC1 may arrive in the next read.
                keep = 1 if self.buf and self.buf[-1] == SYNC0 else 0
                self.garbage += len(self.buf) - keep
                self.buf = self.buf[len(self.buf) - keep :]
                return
            if i > 0:
                self.garbage += i
                self.buf = self.buf[i:]
            buf = self.buf
            if len(buf) < HEADER_LEN:
                return
            length = buf[5]
            if length > MAX_PAYLOAD:
                self.bad_length += 1
                self.buf = buf[1:]
                continue
            total = HEADER_LEN + length + CRC_LEN
            if len(buf) < total:
                return
            crc_rx = buf[total - 2] | (buf[total - 1] << 8)
            if crc16(buf[2 : total - 2]) != crc_rx:
                self.crc_errors += 1
                self.buf = buf[1:]
                continue
            out.append((buf[2], buf[3], buf[4], buf[HEADER_LEN : HEADER_LEN + length]))
            self.buf = buf[total:]


# ---- payloads -------------------------------------------------------------------------------


def pack_hello(host_version=PROTO_VERSION, flags=0):
    return struct.pack(_F_HELLO, host_version, flags)


def unpack_hello(payload):
    host_version, flags = struct.unpack(_F_HELLO, payload)
    return {"host_version": host_version, "flags": flags}


def pack_config(enable_mask, echo_window_ms, gap_ms, quiet_after):
    return struct.pack(_F_CONFIG, enable_mask, echo_window_ms, gap_ms, quiet_after)


def unpack_config(payload):
    enable_mask, echo_window_ms, gap_ms, quiet_after = struct.unpack(_F_CONFIG, payload)
    return {
        "enable_mask": enable_mask,
        "echo_window_ms": echo_window_ms,
        "gap_ms": gap_ms,
        "quiet_after": quiet_after,
    }


def pack_ping(nonce):
    return struct.pack(_F_PING, nonce & 0xFFFFFFFF)


def unpack_ping(payload):
    return {"nonce": struct.unpack(_F_PING, payload)[0]}


def pack_pong(nonce, uptime_ms):
    return struct.pack(_F_PONG, nonce & 0xFFFFFFFF, uptime_ms & 0xFFFFFFFF)


def unpack_pong(payload):
    nonce, uptime_ms = struct.unpack(_F_PONG, payload)
    return {"nonce": nonce, "uptime_ms": uptime_ms}


def pack_info(fw_major, fw_minor, sensor_count, boot_id, uptime_ms, reset_reason, enable_mask):
    return struct.pack(
        _F_INFO,
        fw_major,
        fw_minor,
        PROTO_VERSION,
        sensor_count,
        boot_id & 0xFFFFFFFF,
        uptime_ms & 0xFFFFFFFF,
        reset_reason,
        enable_mask,
    )


def unpack_info(payload):
    fw_major, fw_minor, proto, sensor_count, boot_id, uptime_ms, reset_reason, mask = struct.unpack(
        _F_INFO, payload
    )
    return {
        "fw_major": fw_major,
        "fw_minor": fw_minor,
        "proto_version": proto,
        "sensor_count": sensor_count,
        "boot_id": boot_id,
        "uptime_ms": uptime_ms,
        "reset_reason": reset_reason,
        "enable_mask": mask,
    }


def pack_reading(sensor, status, health, fails, echo_us, cycle, t_ms):
    return struct.pack(
        _F_READING,
        sensor,
        status,
        health,
        min(fails, 255),
        echo_us & 0xFFFFFFFF,
        cycle & 0xFFFF,
        t_ms & 0xFFFFFFFF,
    )


def unpack_reading(payload):
    sensor, status, health, fails, echo_us, cycle, t_ms = struct.unpack(_F_READING, payload)
    return {
        "sensor": sensor,
        "status": status,
        "health": health,
        "fails": fails,
        "echo_us": echo_us,
        "cycle": cycle,
        "t_ms": t_ms,
    }


def pack_nak(ref_type, reason):
    return struct.pack(_F_NAK, ref_type, reason)


def unpack_nak(payload):
    ref_type, reason = struct.unpack(_F_NAK, payload)
    return {"ref_type": ref_type, "reason": reason}
