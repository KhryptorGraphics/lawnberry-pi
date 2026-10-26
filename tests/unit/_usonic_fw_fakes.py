"""HAL test double for the firmware ``Core`` and ``usonic_main.run`` (not a test module).

``_usonic_fakes.py`` wires a real ``Core`` to a simulated sensor array so the HOST driver can be
tested against it. The firmware tests need the opposite: a HAL whose clock, receive queue, reset
reason, boot id and failures the test controls to the millisecond, and which records everything the
core and the main loop do to it (frames written, watchdog feeds, resets).
"""

from __future__ import annotations

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as host
from tests.unit._usonic_fakes import P as fw


class Rebooted(BaseException):
    """What ``reboot()`` does on the real MCU: execution ends there. Not an ``Exception``."""


class RecordingHal:
    """Duck-typed firmware HAL. One ``read()`` call is one main-loop pass and costs ``tick_ms``.

    ``reboot_mode`` models ``microcontroller.reset()``: ``never_returns`` (the real thing: it
    raises ``Rebooted``, which unwinds like a reset), ``returns`` (impossible on hardware, and so
    the worst case for the caller), ``raises`` (a reset that fails) or ``absent`` (no ``reboot``
    attribute at all).
    """

    def __init__(
        self,
        n: int = 0,
        *,
        start_ms: int = 1000,
        boot_id: int = 0x1234ABCD,
        reset_reason: int = fw.RESET_POWER_ON,
        reboot_mode: str = "never_returns",
        tick_ms: int = 1,
    ) -> None:
        self.sensor_count = n
        self.t_ms = start_ms
        self.tick_ms = tick_ms
        self._boot_id = boot_id
        self._reset_reason = reset_reason
        self.rx: list[tuple[int, bytes]] = []  # (time available, bytes) host -> MCU
        self.tx: list[tuple[int, bytes]] = []  # (time written, one frame) MCU -> host
        self.calls = {"read": 0, "write": 0, "feed_watchdog": 0}
        self.fed = 0  # feed_watchdog() calls that returned normally
        self.reboots = 0
        self._armed: dict[str, tuple[int, BaseException]] = {}
        self._reboot_mode = reboot_mode
        if reboot_mode != "absent":
            self.reboot = self._reboot

    # ---- test controls ---------------------------------------------------------------------

    def arm(self, method: str, call_no: int, exc: BaseException) -> None:
        """Make the ``call_no``-th call (1-based) of ``method`` raise ``exc``."""
        self._armed[method] = (call_no, exc)

    def queue_rx(self, data: bytes, at_ms: int | None = None) -> None:
        self.rx.append((self.t_ms if at_ms is None else at_ms, bytes(data)))

    def messages(self, ftype: int | None = None) -> list[tuple[int, object]]:
        """``(time_written, parsed message)`` for every frame the MCU wrote, in order."""
        out = []
        for t, data in self.tx:
            for frame in host.decode_stream(data):
                if ftype is None or frame.ftype == ftype:
                    out.append((t, host.parse_payload(frame)))
        return out

    # ---- the HAL contract ------------------------------------------------------------------

    def _count(self, method: str) -> None:
        self.calls[method] += 1
        armed = self._armed.get(method)
        if armed is not None and self.calls[method] == armed[0]:
            raise armed[1]

    def now_ms(self) -> int:
        return self.t_ms

    def measure(self, index: int, window_us: int) -> tuple[int, int]:
        """A silent sensor: the whole window elapses and nothing comes back."""
        self.t_ms += window_us // 1000
        return fw.ST_NO_EDGE, 0

    def read(self) -> bytes:
        self._count("read")
        ready = bytearray()
        keep = []
        for t, data in self.rx:
            if t <= self.t_ms:
                ready.extend(data)
            else:
                keep.append((t, data))
        self.rx = keep
        self.t_ms += self.tick_ms
        return bytes(ready)

    def write(self, data: bytes) -> None:
        self._count("write")
        self.tx.append((self.t_ms, bytes(data)))

    def boot_id(self) -> int:
        return self._boot_id

    def reset_reason(self) -> int:
        return self._reset_reason

    def feed_watchdog(self) -> None:
        self._count("feed_watchdog")
        self.fed += 1

    def _reboot(self) -> None:
        self.reboots += 1
        if self._reboot_mode == "never_returns":
            raise Rebooted
        if self._reboot_mode == "raises":
            raise RuntimeError("reset is not available")
