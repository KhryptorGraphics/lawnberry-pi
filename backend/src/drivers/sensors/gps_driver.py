"""GPS Driver (T058)

Supports u-blox ZED-F9P via USB or UART, Neo-8M via UART and Quectel LC29H(DA) via UART. This
driver follows the HardwareDriver lifecycle and returns GPS positions when polled.

SIM_MODE notes:
- When SIM_MODE=1 (default in CI), returns deterministic positions without
  touching hardware. Accuracy reflects module class (F9P <1m, Neo-8M ~3m).

Hardware notes (audit SAF-04, SAF-06, SAF-S1):

* **Accuracy is an error estimate, never HDOP.** ``GpsReading.accuracy`` is the receiver's own
  horizontal 1-sigma in metres, from the GST sentence (``sqrt(sigma_lat^2 + sigma_lon^2)``, else
  the error ellipse), and for an RTK fix never better than a typical figure for that fix level
  (fixed 3 cm, float 20 cm); when both exist the **larger** wins. HDOP is a dimensionless
  geometry factor: it was once copied into ``accuracy`` as metres, so a plain 3-4 m fix with a
  good satellite geometry read "0.8 m accurate" and passed the GNSS health gate. It is still
  reported, as ``hdop``. Without a GST (and no RTK fix) the accuracy is ``None``: unknown, which
  the GNSS monitor does not accept. A receiver that is expected to drive autonomously therefore
  has to output GST (``$GNGST``) or hold an RTK fix.
* **Nothing blocks the event loop.** Every serial open, ``readline`` and the gpsd socket run on
  one daemon worker thread; ``read_position`` waits at most ``READ_DEADLINE_S`` for it and then
  returns the last reading (which ages out and degrades the fix) while the worker finishes. A
  second call while the worker is still busy starts nothing: a wedged tty can occupy one thread,
  never the loop, never more threads. Autodetection is never on the hot path: after a probe pass
  finds nothing it is not repeated for ``AUTODETECT_RETRY_S``.
* **No other driver's port.** The RoboHAT MM1 and the ultrasonic MCU enumerate as USB CDC ACM
  ports next to a USB GNSS receiver, and a second reader on a tty steals the bytes the owner is
  waiting for (a RoboHAT ``pwm`` acknowledgement, an ultrasonic frame). The candidate list drops
  every device that is configured for another driver (``ROBOHAT_*`` environment, ``ultrasonic.port``
  in ``hardware.yaml``, ``GPS_EXCLUDE_DEVICES``) or identifies as one (USB vendor 0x2E8A, a
  RoboHAT/CircuitPython descriptor), and every port is opened with ``exclusive=True`` so one that
  another LawnBerry driver has locked is skipped, not read.

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
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def _finite_non_negative(text: str) -> float | None:
    """``text`` as a finite number >= 0, else ``None`` (empty, ``nan``, ``inf``, negative)."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


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


#: Typical 1-sigma horizontal accuracy of an RTK solution, by receiver fix status. Typical, not
#: measured here: used as a floor under (never instead of) the receiver's own estimate.
_RTK_TYPICAL_M = {"RTK_FIXED": 0.03, "RTK_FLOAT": 0.20}


def combine_accuracy(gst_sigma_m: float | None, rtk_status: str | None) -> float | None:
    """The accuracy to report: the larger of the receiver's GST sigma and the RTK figure.

    ``None`` when there is neither (HDOP does not count, audit SAF-04).
    """
    typical = _RTK_TYPICAL_M.get(rtk_status or "")
    known = [v for v in (gst_sigma_m, typical) if v is not None]
    if not known:
        return None
    return max(0.005, *known)


# ------------------------------------------------------------------------------ serial ports


@dataclass(frozen=True)
class SerialPortInfo:
    """What ``serial.tools.list_ports`` knows about one port (the fields this driver uses)."""

    device: str
    vid: int | None = None
    description: str = ""


def list_serial_ports() -> list[SerialPortInfo]:
    """The ports pyserial can see, with their USB identity; ``[]`` if it cannot enumerate."""
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception:
        return []
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
        return found
    return found


def _normalised(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch not in " _-")


def _real(device: str) -> str:
    try:
        return os.path.realpath(device)
    except OSError:
        return device


def ports_owned_by_other_drivers(ports: list[SerialPortInfo] | None = None) -> set[str]:
    """Real paths of the serial devices that belong to another LawnBerry driver.

    From configuration (``ROBOHAT_*`` variables, ``GPS_EXCLUDE_DEVICES``, ``ultrasonic.port`` in
    ``hardware.yaml``) and from identity (USB vendor 0x2E8A, a RoboHAT or CircuitPython
    descriptor). Never raises.
    """
    named: list[str] = []
    for var in _OTHER_DRIVER_PORT_ENV:
        named.append(os.environ.get(var, ""))
    named.extend(os.environ.get(EXCLUDE_ENV, "").split(","))
    try:
        from ...core.config_loader import ConfigLoader

        ultrasonic = ConfigLoader().get()[0].ultrasonic_config
        named.append(str(getattr(ultrasonic, "port", "") or ""))
    except Exception:
        pass
    owned = {_real(p.strip()) for p in named if p and p.strip()}
    for info in ports if ports is not None else list_serial_ports():
        text = _normalised(f"{info.device} {info.description}")
        if info.vid == RP2040_VID or any(word in text for word in _OTHER_DRIVER_WORDS):
            owned.add(_real(info.device))
    return owned


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


#: While the receiver has not been seen to send GST, one read in this many seconds waits for one.
_GST_PROBE_EVERY_S = 30.0


class _IoWorker:
    """One daemon thread that runs blocking calls one at a time.

    A daemon thread, not a ``ThreadPoolExecutor``: a tty that never answers must not be able to
    hold the interpreter open at exit.
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
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as exc:  # delivered to the future, never lost
                future.set_exception(exc)


# ------------------------------------------------------------------------------ the driver


@dataclass
class _Epoch:
    """What one burst of NMEA sentences (one UTC field) has said so far."""

    utc: str | None = None
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    spd: float | None = None
    hdop: float | None = None
    sats: int | None = None
    status: str | None = None
    gga: bool = False
    gst: float | None = None


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
    #: Lines read from a freshly opened port to decide it streams NMEA.
    PROBE_LINES = 5
    #: A GST this old no longer describes the fix.
    GST_MAX_AGE_S = 2.0
    #: GST is awaited (to the end of the epoch) only while the receiver has sent one this recently.
    GST_EXPECTED_WITHIN_S = 5.0
    #: More input than this waiting means the reader fell behind: the backlog is old, drop it.
    BACKLOG_BYTES = 2048
    #: No gpsd retry sooner than this after one that found nothing.
    GPSD_RETRY_S = 5.0

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        cfg = config or {}
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
        self._first_read_done = False
        self._worker: _IoWorker | None = None
        self._inflight: concurrent.futures.Future | None = None
        self._closing = False
        self._next_probe_at = 0.0
        # The first sentence of the epoch after the one a read ended on (kept for the next read)
        self._pushback: str | None = None
        self._next_gpsd_at = 0.0
        # Newest GST: (horizontal sigma m, receive time on the monotonic clock, UTC field)
        self._gst: tuple[float, float, str] | None = None
        self._gst_seen_at: float | None = None
        self._next_gst_probe_at = 0.0
        self.backlog_discards = 0
        # Last observed NMEA sentences (for diagnostics)
        self._last_nmea: dict[str, str] = {}
        # Rate-limit state for hardware failure diagnostics (see _diag)
        self._diag_last_ts: float = 0.0
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
        self.running = True

    async def stop(self) -> None:  # noqa: D401
        self.running = False
        self._closing = True
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()
        # A read that is still on the worker closes the port itself when it ends (it checks
        # ``_closing``); closing it from this thread under a blocked readline would race.
        if not self._busy():
            self._drop_port()

    async def health_check(self) -> dict[str, Any]:  # noqa: D401
        return {
            "driver": "gps",
            "mode": self.cfg.mode.value,
            "initialized": self.initialized,
            "running": self.running,
            "last_read_age_s": (
                (time.monotonic() - self._last_read_mono)
                if self._last_read_mono is not None
                else None
            ),
            "simulation": is_simulation_mode(),
            "device": getattr(self._serial, "port", None),
            "read_in_flight": self._busy(),
            "backlog_discards": self.backlog_discards,
        }

    def _diag(self, reason: str, message: str) -> None:
        """Rate-limited WARNING for hardware GPS failures (~once/30s per reason).

        read_position() runs at ~1 Hz, so unconditional logging would flood the
        journal. This surfaces *why* GPS is blank (permission / missing device /
        no NMEA) — otherwise these failures are swallowed and the dashboard just
        shows "SEARCHING" with no clue in the logs.
        """
        now = time.time()
        if reason == self._diag_last_reason and (now - self._diag_last_ts) < 30.0:
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
            self._last_read_mono = time.monotonic()
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
        future = self._worker.submit(fn)
        future.add_done_callback(self._adopt)
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
            self._last_read_mono = time.monotonic()

    # ------------------------------------------------------------------ the worker side

    def _read_blocking(self) -> GpsReading | None:
        """One hardware read. Runs on the worker thread only; may block, must not raise."""
        try:
            if self._serial is None:
                if time.monotonic() < self._next_probe_at or not self._open_port():
                    return None
            reading = self._read_nmea()
            if reading is not None:
                self._first_read_done = True
                return reading
            # Optional gpsd fallback (if service is present on localhost:2947)
            if time.monotonic() >= self._next_gpsd_at:
                reading = self._read_from_gpsd(timeout_sec=0.5 if self._first_read_done else 1.0)
                if reading is not None:
                    self._first_read_done = True
                    return reading
                self._next_gpsd_at = time.monotonic() + self.GPSD_RETRY_S
            return None
        except Exception as exc:
            # On errors, keep the last reading; surface the cause (rate-limited) and reopen the
            # port next time (a USB receiver that was unplugged is a new device when it returns).
            self._diag("read-error", f"GPS read failed: {exc!r}")
            self._drop_port()
            self._next_probe_at = time.monotonic() + self.REOPEN_DELAY_S
            return None
        finally:
            if self._closing:
                self._drop_port()

    def _drop_port(self) -> None:
        ser, self._serial = self._serial, None
        self._pushback = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _candidates(self) -> tuple[list[str], list[str]]:
        """``(devices to probe in order, devices skipped because another driver owns them)``."""
        ports = list_serial_ports()
        owned = ports_owned_by_other_drivers(ports)
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
        for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
            for device in sorted(glob.glob(pattern)):
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
            if self._closing:
                return False
            for baud in self._baud_candidates:
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
                    self.cfg.baudrate = baud
                    self._next_probe_at = 0.0
                    return True
                opened_but_no_nmea = True
                try:
                    ser.close()
                except Exception:
                    pass
        # Not found: say why (rate-limited), and do not probe again soon.
        self._next_probe_at = time.monotonic() + self.AUTODETECT_RETRY_S
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

    def _streams_nmea(self, ser: Any) -> bool:
        """Does ``ser`` carry NMEA? A few lines, each a complete sentence with a good checksum."""
        for _ in range(self.PROBE_LINES):
            line = ser.readline()
            if not line:
                continue
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("$G") and nmea_checksum_ok(text):
                return True
        return False

    def _read_nmea(self) -> GpsReading | None:
        """Read NMEA for one window and build a reading from one complete epoch.

        An epoch is the burst of sentences that share one UTC field. A reading needs the epoch's
        GGA (position, fix status, satellites, HDOP); the read goes on to the end of the epoch to
        pick up its GST too, but only if this receiver sends GST at all, so the accuracy belongs
        to the position it is reported with. The first sentence of the next epoch is kept for the
        next read, not lost. A window that ends with only an RMC position still yields a reading
        (no fix status, so no RTK figure).
        """
        ser = self._serial
        started = time.monotonic()
        deadline = started + (self.WINDOW_S if self._first_read_done else self.FIRST_WINDOW_S)
        waiting = getattr(ser, "in_waiting", 0)
        if isinstance(waiting, int) and waiting > self.BACKLOG_BYTES:
            ser.reset_input_buffer()
            self._pushback = None
            self.backlog_discards += 1
        # GST is awaited, to the end of the epoch, while this receiver is known to send it, and
        # on one read in ``_GST_PROBE_EVERY_S`` otherwise (to find out when it starts to).
        expect_gst = (
            self._gst_seen_at is not None
            and started - self._gst_seen_at <= self.GST_EXPECTED_WITHIN_S
        )
        if not expect_gst and started >= self._next_gst_probe_at:
            expect_gst = True
            self._next_gst_probe_at = started + _GST_PROBE_EVERY_S

        epoch = _Epoch()
        while time.monotonic() < deadline and not self._closing:
            if self._pushback is not None:
                raw, self._pushback = self._pushback, None
            else:
                raw = ser.readline().decode("ascii", errors="ignore").strip()
            if not raw.startswith("$G") or not nmea_checksum_ok(raw):
                continue
            kind = raw[3:6]
            if kind not in ("GGA", "RMC", "GST"):
                continue
            utc = raw.split(",", 2)[1] if raw.count(",") >= 2 else ""
            if epoch.utc is not None and utc != epoch.utc:
                if epoch.gga:
                    self._pushback = raw
                    break  # the next epoch has started: this one is complete
                epoch = _Epoch()  # a new epoch began before this one had a GGA: start over
            epoch.utc = utc
            self._last_nmea[kind] = raw
            if kind == "GGA":
                gga = self._parse_gga(raw)
                if gga and gga[0] is not None and gga[1] is not None:
                    epoch.lat, epoch.lon, epoch.alt, epoch.sats, hdop, fix_quality = gga
                    epoch.hdop = hdop if hdop is not None and math.isfinite(hdop) else None
                    epoch.status = self._map_fix_quality(fix_quality)
                    epoch.gga = True
            elif kind == "RMC":
                rmc = self._parse_rmc(raw)
                if rmc:
                    lat, lon, spd_knots, _course = rmc
                    if not epoch.gga and lat is not None and lon is not None:
                        epoch.lat, epoch.lon = lat, lon
                    if spd_knots is not None:
                        epoch.spd = spd_knots * 0.514444  # knots -> m/s
            else:  # GST
                sigma = self._parse_gst(raw)
                self._gst_seen_at = time.monotonic()
                if sigma is not None:
                    self._gst = (sigma, time.monotonic(), utc)
            # This epoch's GST, wherever in the burst it came (it may precede the position).
            if self._gst is not None and self._gst[2] == epoch.utc:
                epoch.gst = self._gst[0]
            if epoch.gga and (epoch.gst is not None or not expect_gst):
                break

        if epoch.lat is None or epoch.lon is None:
            return None
        gst_sigma = epoch.gst
        if gst_sigma is None and self._gst is not None:
            sigma, received, _utc = self._gst
            if time.monotonic() - received <= self.GST_MAX_AGE_S:
                gst_sigma = sigma
        return GpsReading(
            latitude=epoch.lat,
            longitude=epoch.lon,
            altitude=epoch.alt,
            accuracy=combine_accuracy(gst_sigma, epoch.status),
            speed=epoch.spd,
            satellites=epoch.sats,
            mode=self.cfg.mode,
            rtk_status=epoch.status,
            hdop=epoch.hdop,
        )

    def get_last_nmea(self) -> dict[str, str]:
        """Return a shallow copy of last seen NMEA sentences for diagnostics."""
        try:
            return dict(self._last_nmea)
        except Exception:
            return {}

    @staticmethod
    def _parse_nmea_coord(val: str, hemi: str) -> float | None:
        """Convert NMEA ddmm.mmmm (lat) / dddmm.mmmm (lon) to decimal degrees.

        hemi is 'N'/'S' or 'E'/'W'. Returns None if invalid.
        """
        try:
            if not val or not hemi:
                return None
            # Split degrees and minutes
            if "." not in val:
                return None
            dot = val.find(".")
            deg_len = dot - 2  # two digits of minutes before dot
            deg = int(val[:deg_len])
            mins = float(val[deg_len:])
            dec = deg + mins / 60.0
            if hemi in ("S", "W"):
                dec = -dec
            return dec if math.isfinite(dec) else None
        except Exception:
            return None

    def _parse_gga(
        self, line: str
    ) -> (
        tuple[float | None, float | None, float | None, int | None, float | None, int | None] | None
    ):
        """Parse GGA: returns (lat, lon, altitude_m, satellites, hdop, fix_quality)."""
        try:
            parts = line.split(",")
            # GGA fields: 2=lat,3=N/S,4=lon,5=E/W,6=fix,7=sats,8=HDOP,9=altitude
            lat = self._parse_nmea_coord(parts[2], parts[3]) if len(parts) > 4 else None
            lon = self._parse_nmea_coord(parts[4], parts[5]) if len(parts) > 6 else None
            fix_quality = parts[6] if len(parts) > 6 else "0"
            sats = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None
            hdop = float(parts[8]) if len(parts) > 8 and parts[8] not in ("", None) else None
            alt = float(parts[9]) if len(parts) > 9 and parts[9] not in ("", None) else None
            if fix_quality in {"0", ""}:  # no fix
                # Keep lat/lon if parser produced values but no fix -> treat as unreliable
                pass
            try:
                fix_int = int(fix_quality)
            except (TypeError, ValueError):
                fix_int = None
            return (lat, lon, alt, sats, hdop, fix_int)
        except Exception:
            return None

    def _parse_rmc(
        self, line: str
    ) -> tuple[float | None, float | None, float | None, float | None] | None:
        """Parse RMC: returns (lat, lon, speed_knots, course_deg)."""
        try:
            parts = line.split(",")
            # RMC fields: 3=lat,4=N/S,5=lon,6=E/W,7=speed(knots),8=track/course
            lat = self._parse_nmea_coord(parts[3], parts[4]) if len(parts) > 4 else None
            lon = self._parse_nmea_coord(parts[5], parts[6]) if len(parts) > 6 else None
            spd = float(parts[7]) if len(parts) > 7 and parts[7] not in ("", None) else None
            course = float(parts[8]) if len(parts) > 8 and parts[8] not in ("", None) else None
            return (lat, lon, spd, course)
        except Exception:
            return None

    def _parse_gst(self, line: str) -> float | None:
        """Horizontal 1-sigma accuracy in metres from a GST sentence (see ``parse_gst_sigma``)."""
        return parse_gst_sigma(line)

    @staticmethod
    def _map_fix_quality(fix_quality: int | None) -> str | None:
        if fix_quality is None:
            return None
        mapping = {
            0: "NO_FIX",
            1: "GPS_FIX",
            2: "DGPS",
            4: "RTK_FIXED",
            5: "RTK_FLOAT",
            6: "DEAD_RECKONING",
            7: "MANUAL",
            8: "SIMULATION",
        }
        return mapping.get(fix_quality)

    def _read_from_gpsd(self, timeout_sec: float = 0.5) -> GpsReading | None:
        """Try reading a TPV report from gpsd if available.

        Uses a raw TCP socket to 127.0.0.1:2947 to avoid external deps. Blocking: only ever
        called on the worker thread. Returns a GpsReading if lat/lon are present, else None.
        ``accuracy`` is gpsd's ``eph`` (an error estimate in metres); gpsd's TPV carries no HDOP.
        """
        host = os.environ.get("GPSD_HOST", "127.0.0.1")
        port = int(os.environ.get("GPSD_PORT", "2947"))
        try:
            with socket.create_connection((host, port), timeout=0.3) as s:
                s.settimeout(timeout_sec)
                # Issue WATCH command
                s.sendall(b'?WATCH={"enable":true,"json":true}\n')
                start = time.monotonic()
                buf = b""
                while time.monotonic() - start < timeout_sec:
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
                        if obj.get("class") == "TPV":
                            lat = obj.get("lat")
                            lon = obj.get("lon")
                            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                                alt = (
                                    obj.get("alt")
                                    if isinstance(obj.get("alt"), (int, float))
                                    else None
                                )
                                spd = (
                                    obj.get("speed")
                                    if isinstance(obj.get("speed"), (int, float))
                                    else None
                                )
                                crs = (
                                    obj.get("track")
                                    if isinstance(obj.get("track"), (int, float))
                                    else None
                                )
                                eph = (
                                    obj.get("eph")
                                    if isinstance(obj.get("eph"), (int, float))
                                    else None
                                )
                                accuracy = (
                                    float(eph)
                                    if eph is not None and math.isfinite(eph) and eph > 0
                                    else None
                                )
                                return GpsReading(
                                    latitude=float(lat),
                                    longitude=float(lon),
                                    altitude=float(alt) if alt is not None else None,
                                    accuracy=accuracy,
                                    speed=float(spd) if spd is not None else None,
                                    heading=float(crs) if crs is not None else None,
                                    mode=self.cfg.mode,
                                )
                return None
        except Exception:
            return None


__all__ = [
    "GPSDriver",
    "GPSDriverConfig",
    "SerialPortInfo",
    "combine_accuracy",
    "list_serial_ports",
    "nmea_checksum_ok",
    "parse_gst_sigma",
    "ports_owned_by_other_drivers",
    "preferred_gps_device",
]
