"""Fakes for the ultrasonic MCU tests (not a test module).

``FakeMcuSerial`` is a serial port wired to the REAL firmware ``Core`` (``firmware/ultrasonic-mcu``)
running against a simulated HAL, so the host driver is tested against the actual firmware logic and
wire bytes, not against a hand-written imitation. Everything runs on one fake clock; the MCU is
evaluated lazily whenever the host touches the port, frames become readable at the instant they
were emitted, and fault injection works at frame granularity (corrupt, truncate, drop, garbage).
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

FW_DIR = Path(__file__).resolve().parents[2] / "firmware" / "ultrasonic-mcu"
SPEED_OF_SOUND = 343.2


def load_module(name: str, *, fresh: bool = False):
    """Load ``firmware/ultrasonic-mcu/<name>.py`` under its flat CircuitPython module name.

    The firmware directory must NEVER be put on ``sys.path``: it contains ``code.py`` (required by
    CircuitPython), which would shadow the standard library's ``code`` module that ``pdb`` imports
    and then run the firmware's main loop at import time. Sibling imports (``import usonic_proto``)
    resolve through ``sys.modules`` because the modules are loaded in dependency order.
    """
    if not fresh and name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, FW_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_firmware():
    """The hardware-independent firmware modules: ``(usonic_proto, usonic_core)``."""
    proto = load_module("usonic_proto")
    core = load_module("usonic_core")
    return proto, core


P, CORE = load_firmware()


class FakeClock:
    """Monotonic seconds plus a matching wall clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self._wall0 = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC)
        self._t0 = start

    def now(self) -> float:
        return self.t

    def wall(self) -> datetime:
        return self._wall0 + timedelta(seconds=self.t - self._t0)

    def advance(self, dt: float) -> None:
        self.t += dt

    async def sleep(self, dt: float) -> None:
        """Advance fake time and let the event loop run.

        It MUST yield: the driver's background service loop sleeps with this, and a sleep that
        never suspends would spin forever and starve the loop (and make the task uncancellable).
        """
        self.t += dt
        await asyncio.sleep(0)


@dataclass
class SensorModel:
    """What one physical sensor does when triggered.

    mode: ``echo`` (a target at ``distance_cm``), ``silent`` (no pulse: open air OR a dead
    sensor), ``stuck`` (line stuck high), ``no_fall`` (pulse never ends), ``error`` (the HAL
    raises), ``script`` (``(status, echo_us)`` pairs consumed in order, the last repeats).
    """

    mode: str = "echo"
    distance_cm: float = 150.0
    script: list[tuple[int, int]] = field(default_factory=list)
    _i: int = 0

    def measure(self, window_us: int) -> tuple[int, int, int]:
        """Return ``(status, echo_us, duration_us)``."""
        if self.mode == "silent":
            return P.ST_NO_EDGE, 0, window_us
        if self.mode == "stuck":
            return P.ST_STUCK, 0, 200
        if self.mode == "no_fall":
            return P.ST_NO_FALL, window_us, window_us
        if self.mode == "error":
            raise RuntimeError("simulated HAL fault")
        if self.mode == "script":
            i = min(self._i, len(self.script) - 1)
            status, echo = self.script[i]
            self._i += 1
            return status, echo, (echo if status == P.ST_OK else window_us)
        echo_us = round(self.distance_cm * 20000.0 / SPEED_OF_SOUND)
        if echo_us > window_us:
            return P.ST_NO_FALL, window_us, window_us
        return P.ST_OK, echo_us, echo_us


class SimHal:
    """Simulated RP2040 HAL: per-sensor models, a byte queue each way, a trigger log."""

    def __init__(self, n: int, sensors: list[SensorModel] | None, start_ms: int, boot_id: int):
        self.n = n
        self.sensors = sensors or [SensorModel() for _ in range(n)]
        self.t_ms = start_ms
        self._boot_id = boot_id
        self.reset = P.RESET_POWER_ON
        self.rx: list[tuple[int, bytes]] = []  # (time available, bytes) host -> MCU
        self.tx: list[tuple[int, bytes]] = []  # (time emitted, frame) MCU -> host
        self.triggers: list[tuple[int, int, int]] = []  # (sensor, start_ms, end_ms)
        self.windows: list[int] = []  # echo window (us) each measurement was asked for
        self.frozen = False

    def now_ms(self) -> int:
        return self.t_ms

    def measure(self, index: int, window_us: int) -> tuple[int, int]:
        start = self.t_ms
        self.windows.append(window_us)
        status, echo_us, dur_us = self.sensors[index].measure(window_us)
        self.t_ms += math.ceil(dur_us / 1000)
        self.triggers.append((index, start, self.t_ms))
        return status, echo_us

    def read(self) -> bytes:
        ready = bytearray()
        keep = []
        for t, data in self.rx:
            if t <= self.t_ms:
                ready.extend(data)
            else:
                keep.append((t, data))
        self.rx = keep
        return bytes(ready)

    def write(self, data: bytes) -> None:
        self.tx.append((self.t_ms, data))

    def boot_id(self) -> int:
        return self._boot_id

    def reset_reason(self) -> int:
        return self.reset


class FakeMcuSerial:
    """A pyserial-like port connected to a real firmware ``Core``."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        sensors: list[SensorModel] | None = None,
        n: int = 4,
        boot_id: int | None = None,
        max_read: int | None = None,
        **core_kwargs,
    ) -> None:
        self.clock = clock
        self.n = n
        self.max_read = max_read
        self._core_kwargs = core_kwargs
        self._boot_counter = boot_id if boot_id is not None else random.Random(1).getrandbits(32)
        self.hal = SimHal(n, sensors, self._host_ms(), self._boot_counter)
        self.core = CORE.Core(self.hal, n, **core_kwargs)
        self.core.start()
        self.closed = False
        self.broken: Exception | None = None
        self._out = bytearray()
        self._delivered_upto = 0
        self.mutators: list[Callable[[bytes], bytes | None]] = []
        self.written: list[bytes] = []  # raw host -> MCU writes, for assertions

    # ---- time ------------------------------------------------------------------------------

    def _host_ms(self) -> int:
        return int(round(self.clock.t * 1000))

    def sync(self) -> None:
        """Run the MCU up to the host's current time."""
        if self.hal.frozen:
            return
        target = self._host_ms()
        while self.hal.t_ms < target:
            self.core.poll()
            self.hal.t_ms += 1

    # ---- injection -------------------------------------------------------------------------

    def corrupt_next(self, count: int, fn: Callable[[bytes], bytes | None]) -> None:
        """Apply ``fn`` to the next ``count`` MCU->host frames (return None to drop one)."""
        remaining = [count]

        def mutate(frame: bytes) -> bytes | None:
            if remaining[0] <= 0:
                return frame
            remaining[0] -= 1
            return fn(frame)

        self.mutators.append(mutate)

    def inject(self, data: bytes) -> None:
        """Garbage appears on the line right now."""
        self.hal.tx.append((self._host_ms(), data))

    def unplug(self, exc: Exception | None = None) -> None:
        self.broken = exc or OSError("device reports readiness to read but returned no data")

    def freeze(self) -> None:
        """The MCU hangs: it neither measures nor talks."""
        self.sync()
        self.hal.frozen = True

    def reboot(self, boot_id: int | None = None) -> None:
        """Watchdog reset in place: new Core, new boot_id, bytes in flight lost."""
        self._boot_counter = boot_id if boot_id is not None else self._boot_counter + 1
        sensors = self.hal.sensors
        self.hal = SimHal(self.n, sensors, self._host_ms(), self._boot_counter)
        self.hal.reset = P.RESET_WATCHDOG
        self.core = CORE.Core(self.hal, self.n, **self._core_kwargs)
        self.core.start()

    def set_sensor(self, index: int, model: SensorModel) -> None:
        self.sync()
        self.hal.sensors[index] = model

    # ---- pyserial-like API -----------------------------------------------------------------

    def _check(self) -> None:
        if self.broken is not None:
            raise self.broken
        if self.closed:
            raise OSError("port closed")

    def _deliverable(self) -> bytes:
        self.sync()
        now = self._host_ms()
        keep = []
        for t, frame in self.hal.tx:
            if t > now:
                keep.append((t, frame))
                continue
            data: bytes | None = frame
            for mutate in self.mutators:
                if data is None:
                    break
                data = mutate(data)
            if data:
                self._out.extend(data)
        self.hal.tx = keep
        return bytes(self._out)

    @property
    def in_waiting(self) -> int:
        self._check()
        return len(self._deliverable())

    def read(self, size: int = 1) -> bytes:
        self._check()
        self._deliverable()
        n = size if self.max_read is None else min(size, self.max_read)
        data = bytes(self._out[:n])
        del self._out[:n]
        return data

    def write(self, data: bytes) -> int:
        self._check()
        self.written.append(bytes(data))
        self.sync()
        self.hal.rx.append((self.hal.t_ms, bytes(data)))
        return len(data)

    def close(self) -> None:
        self.closed = True


class FakeMcuFactory:
    """``serial_factory`` that hands out FakeMcuSerial ports and can fail to open."""

    def __init__(self, clock: FakeClock, **port_kwargs) -> None:
        self.clock = clock
        self.port_kwargs = port_kwargs
        self.ports: list[FakeMcuSerial] = []
        self.opens: list[tuple[str, int]] = []
        self.fail_opens = 0
        self._next_boot = 0xC0FFEE00

    def __call__(self, port: str, baudrate: int) -> FakeMcuSerial:
        self.opens.append((port, baudrate))
        if self.fail_opens > 0:
            self.fail_opens -= 1
            raise OSError(f"could not open port {port}: No such file or directory")
        self._next_boot += 1
        kwargs = dict(self.port_kwargs)
        kwargs.setdefault("boot_id", self._next_boot)
        mcu = FakeMcuSerial(self.clock, **kwargs)
        self.ports.append(mcu)
        return mcu

    @property
    def current(self) -> FakeMcuSerial:
        return self.ports[-1]


# ---- frame mutators ---------------------------------------------------------------------------


def flip_payload_bit(frame: bytes) -> bytes:
    b = bytearray(frame)
    b[P.HEADER_LEN] ^= 0x01
    return bytes(b)


def flip_crc_bit(frame: bytes) -> bytes:
    b = bytearray(frame)
    b[-1] ^= 0x80
    return bytes(b)


def truncate(frame: bytes) -> bytes:
    return frame[: len(frame) // 2]


def corrupt_length(frame: bytes) -> bytes:
    b = bytearray(frame)
    b[5] = P.MAX_PAYLOAD  # claims a long payload: the receiver must not wait for it forever
    return bytes(b)


def drop(frame: bytes) -> None:
    return None


def with_garbage(frame: bytes) -> bytes:
    return b"\x00\xa5\x13\x7f\xa5" + frame
