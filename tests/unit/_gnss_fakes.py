"""Fakes for the GNSS driver tests (not a test module).

``Port`` is what sits behind one device node and ``Serial`` is the ``serial.Serial`` the driver
opens on it. The port is a BYTE STREAM with the semantics real pyserial has on a tty, because the
driver's line framing is exactly what some of these tests are about:

* ``read(size)`` returns what is there (up to ``size``) at once, and waits up to the port timeout
  for one byte only when nothing is; ``in_waiting`` is the bytes queued;
* ``readline()`` is ``read(1)`` until a line feed, and returns a partial line only after one
  silent timeout: it NEVER returns while bytes keep arriving without a line feed (the property
  that wedged the old driver);
* reading a closed port raises, ``exclusive=True`` is refused while another driver holds the port.

A port runs in real time (``feed`` from a test or a thread), or on a virtual clock (``clock``): then
nothing sleeps, a read that would wait advances the clock to the next scheduled arrival (or by the
port timeout), and a test can say exactly when each byte reached the receiver's side.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

# ------------------------------------------------------------------------------ NMEA builders


def nmea(body: str) -> bytes:
    """``$body*HH\\r\\n`` with the right checksum."""
    check = 0
    for ch in body:
        check ^= ord(ch)
    return f"${body}*{check:02X}\r\n".encode()


def utc_of(k: int, frac: str = ".00") -> str:
    """The UTC field of epoch ``k``: 12:00:00 + ``k`` seconds."""
    return f"12{(k // 60) % 60:02d}{k % 60:02d}{frac}"


def gga(
    k: int,
    *,
    fix: int | str = 1,
    lat: str = "4807.038",
    ns: str = "N",
    lon: str = "01131.000",
    ew: str = "E",
    sats: str = "08",
    hdop: str = "0.8",
    alt: str = "545.4",
    age: str = "",
    station: str = "",
    utc: str | None = None,
) -> bytes:
    t = utc_of(k) if utc is None else utc
    return nmea(
        f"GNGGA,{t},{lat},{ns},{lon},{ew},{fix},{sats},{hdop},{alt},M,46.9,M,{age},{station}"
    )


def gst(k: int, sigma: tuple[float, float] = (3.0, 2.9), *, utc: str | None = None) -> bytes:
    t = utc_of(k) if utc is None else utc
    return nmea(f"GNGST,{t},2.1,4.4,3.1,45.0,{sigma[0]},{sigma[1]},5.5")


def rmc(
    k: int,
    *,
    status: str = "A",
    lat: str = "4807.038",
    ns: str = "N",
    lon: str = "01131.000",
    ew: str = "E",
    speed: str = "0.5",
    course: str = "84.4",
    mode: str | None = "A",
    utc: str | None = None,
) -> bytes:
    t = utc_of(k) if utc is None else utc
    tail = f",{mode}" if mode is not None else ""
    return nmea(f"GNRMC,{t},{status},{lat},{ns},{lon},{ew},{speed},{course},230394,,{tail}")


def epoch_lines(
    k: int,
    *,
    fix: int | str = 1,
    hdop: str = "0.8",
    gst_sigma: tuple[float, float] | None = (3.0, 2.9),
    bad_gst_checksum: bool = False,
    age: str | None = None,
    with_rmc: bool = True,
    mark: bool = False,
    utc: str | None = None,
    **position,
) -> list[bytes]:
    """One epoch as a receiver sends it: GGA, GST (if any), RMC (if any).

    ``age`` is the GGA age-of-corrections field; by default an RTK fix (4, 5) reports ``1.0`` and
    anything else leaves it empty, as a receiver does. ``mark`` writes the epoch number into the
    longitude (``epoch_of`` reads it back), so a test can tell WHICH epoch a reading came from.
    """
    if age is None:
        age = "1.0" if fix in (4, 5, "4", "5") else ""
    if mark:
        position.setdefault("lon", f"01131.{k % 1000:03d}")
    gga_part = {key: v for key, v in position.items() if key in _GGA_KEYS}
    rmc_part = {key: v for key, v in position.items() if key in _RMC_KEYS}
    unknown = set(position) - _GGA_KEYS - _RMC_KEYS
    assert not unknown, f"epoch_lines: unknown fields {unknown}"
    lines = [
        gga(k, fix=fix, hdop=hdop, age=age, station="0000" if age else "", utc=utc, **gga_part)
    ]
    if gst_sigma is not None:
        line = gst(k, gst_sigma, utc=utc)
        lines.append(line.replace(b"*", b"*0") if bad_gst_checksum else line)
    if with_rmc:
        lines.append(rmc(k, utc=utc, **rmc_part))
    return lines


_GGA_KEYS = {"lat", "ns", "lon", "ew", "sats", "alt"}
_RMC_KEYS = {"lat", "ns", "lon", "ew", "speed", "course", "mode", "status"}


def ubx(cls: int, mid: int, payload: bytes) -> bytes:
    """A UBX frame (binary: its payload may contain ``\\n``, ``\\r`` and ``$``)."""
    body = bytes([cls, mid]) + len(payload).to_bytes(2, "little") + payload
    a = b = 0
    for x in body:
        a = (a + x) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def epoch_of(reading) -> int:
    """The epoch number a ``mark=True`` epoch carries in its longitude."""
    return round((reading.longitude - 11.0) * 60.0 * 1000.0 - 31.0 * 1000.0)


# ------------------------------------------------------------------------------ clocks


class Clock:
    """One fake time: monotonic seconds, and a wall clock that follows it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self._t0 = start
        self.wall_epoch = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)

    def mono(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return self.wall_epoch + timedelta(seconds=self.t - self._t0)

    def advance(self, dt: float) -> None:
        self.t += dt


# ------------------------------------------------------------------------------ the device


class Port:
    """What is behind one device node: a receiver, silence, a hang, or a lock held elsewhere.

    ``mode``: ``"nmea"`` a receiver that has its next epoch ready whenever the queue is empty
    (epoch ``k`` has UTC 12:00:k, so every epoch is a later one); ``"stream"`` only what ``feed``
    queues (a test or a thread plays the receiver); ``"silent"`` never any byte; ``"blocking"`` a
    read hangs until ``release`` is set; ``"locked"`` another driver holds the exclusive lock.
    """

    def __init__(
        self, kind: str = "nmea", clock: Clock | None = None, baud: int | None = None, **epoch
    ) -> None:
        self.mode = kind
        self.epoch = epoch
        self.clock = clock
        #: The only baud rate the receiver is configured for (``None``: any). A port opened at
        #: another one reads line noise: bytes with no ``$``, no CR and no LF.
        self.baud = baud
        self.release = threading.Event()
        self.entered = 0  # reads that reached this port
        self.k = 0
        self._buf = bytearray()
        self._cv = threading.Condition()
        self._schedule: list[tuple[float, bytes]] = []  # virtual-clock arrivals, in time order
        self.closed_by_driver = 0

    # -- the receiver's side

    def feed(self, data: bytes) -> None:
        with self._cv:
            self._buf += data
            self._cv.notify_all()

    def feed_epoch(self, k: int | None = None, **kw) -> None:
        if k is None:
            k = self.k
            self.k += 1
        self.feed(b"".join(epoch_lines(k, **{**self.epoch, **kw})))

    def schedule(self, at: float, data: bytes) -> None:
        """Virtual clock only: ``data`` reaches the port's queue at fake time ``at``."""
        self._schedule.append((at, data))
        self._schedule.sort(key=lambda item: item[0])

    def schedule_epoch(self, at: float, k: int, **kw) -> None:
        self.schedule(at, b"".join(epoch_lines(k, **{**self.epoch, **kw})))

    def _due(self) -> None:
        assert self.clock is not None
        while self._schedule and self._schedule[0][0] <= self.clock.t:
            self._buf += self._schedule.pop(0)[1]

    # -- the driver's side (through Serial)

    @property
    def queued(self) -> int:
        if self.clock is not None:
            self._due()
        with self._cv:
            return len(self._buf)

    def take(self, size: int, timeout: float, baud: int | None = None) -> bytes:
        self.entered += 1
        if self.baud is not None and baud != self.baud:
            if self.clock is not None:
                self.clock.advance(min(timeout, 0.05))
            else:
                time.sleep(0.005)
            return (b"\xe0\x80\x9f\xfe\x81" * 64)[: max(1, min(size, 40))]
        if self.mode == "blocking":
            self.release.wait(3.0)
            return b""
        if self.mode in ("silent", "locked"):
            if self.clock is not None:
                self.clock.advance(timeout)
            else:
                time.sleep(0.005)
            return b""
        if self.clock is not None:
            return self._take_virtual(size, timeout)
        with self._cv:
            if not self._buf and self.mode == "nmea":
                self._buf += b"".join(epoch_lines(self.k, **self.epoch))
                self.k += 1
            if not self._buf:
                self._cv.wait(timeout)
            out = bytes(self._buf[:size])
            del self._buf[:size]
            return out

    def _take_virtual(self, size: int, timeout: float) -> bytes:
        assert self.clock is not None
        self._due()
        if not self._buf:
            if self.mode == "nmea":
                self._buf += b"".join(epoch_lines(self.k, **self.epoch))
                self.k += 1
            else:
                horizon = self.clock.t + timeout
                nxt = self._schedule[0][0] if self._schedule else horizon
                self.clock.t = min(nxt, horizon)
                self._due()
        out = bytes(self._buf[:size])
        del self._buf[:size]
        return out


class SerialException(Exception):
    pass


class Serial:
    """The ``serial.Serial`` the driver gets for one open of a ``Port``."""

    def __init__(self, port: Port, name: str, baudrate: int, timeout: float | None) -> None:
        self._port = port
        self.port = name
        self.baudrate = baudrate
        self.timeout = 0.25 if timeout is None else timeout
        self.is_open = True

    def _check(self) -> None:
        if not self.is_open:
            raise SerialException("port is closed")

    @property
    def in_waiting(self) -> int:
        self._check()
        return self._port.queued

    def read(self, size: int = 1) -> bytes:
        self._check()
        return self._port.take(size, self.timeout, self.baudrate)

    def readline(self) -> bytes:
        """pyserial's ``read_until(LF)``: only a line feed or one silent timeout ends it."""
        line = bytearray()
        while True:
            byte = self.read(1)
            if not byte:
                break
            line += byte
            if byte == b"\n":
                break
        return bytes(line)

    def reset_input_buffer(self) -> None:
        self._check()
        with self._port._cv:
            self._port._buf.clear()

    def close(self) -> None:
        if self.is_open:
            self.is_open = False
            self._port.closed_by_driver += 1


def install_serial(monkeypatch, ports: dict[str, Port], found=()) -> types.ModuleType:
    """A fake ``serial`` module. ``found`` is what ``list_ports.comports()`` enumerates:
    ``(device, vid, description)`` tuples. ``module.opens`` records ``(device, baud, kwargs)``;
    ``module.serials`` the ``Serial`` objects opened."""
    module = types.ModuleType("serial")
    module.opens = []
    module.serials = []

    def open_serial(port=None, baudrate=9600, timeout=None, **kw):
        module.opens.append((port, baudrate, dict(kw)))
        script = ports.get(port)
        if script is None:
            raise SerialException(2, f"could not open port {port}: no such file")
        if script.mode == "locked" and kw.get("exclusive"):
            raise SerialException(11, f"Could not exclusively lock port {port}")
        ser = Serial(script, port, baudrate, timeout)
        module.serials.append(ser)
        return ser

    module.SerialException, module.Serial = SerialException, open_serial
    listing = types.ModuleType("serial.tools.list_ports")
    listing.comports = lambda: [
        SimpleNamespace(device=d, vid=v, description=text, manufacturer="", product="")
        for d, v, text in found
    ]
    tools = types.ModuleType("serial.tools")
    tools.list_ports = listing
    module.tools = tools
    for name, mod in (
        ("serial", module),
        ("serial.tools", tools),
        ("serial.tools.list_ports", listing),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return module
