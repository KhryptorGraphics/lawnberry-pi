"""In-memory stand-in for the ``lgpio`` module with fault injection (no hardware, no lgpio).

It follows the conventions the IBT-4 adapter relies on: a call returns 0 (``gpio_read``: the
level, ``gpiochip_open``: the handle) on success and a negative code on failure, and raises
``error`` instead when ``exceptions`` is True (the real module's default). ``pad`` is the level a
meter on the pin would show; ``calls`` records every call in order so tests can assert on what
was (and was not) touched. Used by ``test_ibt4_lgpio.py`` and ``test_emergency_stop_service.py``.
"""

from __future__ import annotations

from typing import Any, Literal

WriteFault = Literal["raise", "rc", "ignore"]


class FakeLgpioError(Exception):
    """Stand-in for ``lgpio.error``."""


class FakeLgpio:
    error = FakeLgpioError

    def __init__(self, *, exceptions: bool = True, handle: int = 7) -> None:
        self.exceptions = exceptions
        self.handle = handle
        self.calls: list[tuple[Any, ...]] = []
        self.pad: dict[int, int] = {}
        self.claimed: set[int] = set()
        self.freed: list[int] = []
        self.closed: list[int] = []
        self._open = False
        self._writes_seen: dict[int, int] = {}
        self._chip_fault: str | None = None
        self._claim_faults: dict[int, str] = {}
        self._write_faults: list[tuple[int, WriteFault, int | None]] = []
        self._read_faults: dict[int, str] = {}
        self._free_faults: set[int] = set()
        self._held: dict[int, int] = {}

    # ------------------------------------------------------------ fault injection

    def fail_chip_open(self, message: str = "can not open gpiochip") -> None:
        self._chip_fault = message

    def fail_claim(self, pin: int, message: str = "GPIO busy") -> None:
        """The claim of ``pin`` fails (e.g. the kernel already holds the line)."""
        self._claim_faults[pin] = message

    def clear_claim_fault(self, pin: int) -> None:
        """The line is free again (the kernel released it / the pin was moved)."""
        self._claim_faults.pop(pin, None)

    def clear_read_fault(self, pin: int) -> None:
        """The readback of ``pin`` works again."""
        self._read_faults.pop(pin, None)

    def clear_write_faults(self) -> None:
        """Every write works again (the ``fail_write`` faults of every pin are removed)."""
        self._write_faults.clear()

    def fail_write(self, pin: int, how: WriteFault = "raise", *, nth: int | None = None) -> None:
        """Writes to ``pin`` fail: ``raise`` raises, ``rc`` returns -1, ``ignore`` returns 0 but
        changes nothing. ``nth`` limits the fault to the nth write to that pin (1-based)."""
        self._write_faults.append((pin, how, nth))

    def fail_read(self, pin: int, message: str = "read failed") -> None:
        self._read_faults[pin] = message

    def fail_free(self, pin: int) -> None:
        self._free_faults.add(pin)

    def hold_pad(self, pin: int, level: int) -> None:
        """Something else holds the pad at ``level``: writes still succeed, the pad ignores them."""
        self._held[pin] = level

    def release_pad(self, pin: int) -> None:
        self._held.pop(pin, None)

    # ------------------------------------------------------------------ inspection

    def writes(self) -> list[tuple[int, int]]:
        """``(pin, level)`` of every ``gpio_write`` call, in order (failed ones included)."""
        return [(c[2], c[3]) for c in self.calls if c[0] == "gpio_write"]

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    # ------------------------------------------------------------------- lgpio API

    def _fail(self, message: str, rc: int = -1) -> int:
        if self.exceptions:
            raise FakeLgpioError(message)
        return rc

    def _bad_handle(self, handle: int) -> bool:
        return not self._open or handle != self.handle

    def gpiochip_open(self, chip: int) -> int:
        self.calls.append(("gpiochip_open", chip))
        if self._chip_fault is not None:
            return self._fail(self._chip_fault)
        self._open = True
        return self.handle

    def gpio_claim_output(self, handle: int, pin: int, level: int = 0, lFlags: int = 0) -> int:
        self.calls.append(("gpio_claim_output", handle, pin, level))
        if self._bad_handle(handle):
            return self._fail("bad handle")
        if pin in self._claim_faults:
            return self._fail(self._claim_faults[pin])
        self.claimed.add(pin)
        self.pad[pin] = level
        return 0

    def gpio_write(self, handle: int, pin: int, level: int) -> int:
        self.calls.append(("gpio_write", handle, pin, level))
        if self._bad_handle(handle):
            return self._fail("bad handle")
        seen = self._writes_seen[pin] = self._writes_seen.get(pin, 0) + 1
        if pin not in self.claimed:
            return self._fail("GPIO not allocated")
        for fault_pin, how, nth in self._write_faults:
            if fault_pin != pin or (nth is not None and nth != seen):
                continue
            if how == "raise":
                raise FakeLgpioError(f"GPIO{pin} write failed")
            if how == "rc":
                return -1
            return 0  # "ignore": success is reported, the output does not change
        self.pad[pin] = level
        return 0

    def gpio_read(self, handle: int, pin: int) -> int:
        self.calls.append(("gpio_read", handle, pin))
        if self._bad_handle(handle):
            return self._fail("bad handle")
        if pin in self._read_faults:
            return self._fail(self._read_faults[pin])
        if pin not in self.claimed:
            return self._fail("GPIO not allocated")
        return self._held.get(pin, self.pad.get(pin, 0))

    def gpio_free(self, handle: int, pin: int) -> int:
        self.calls.append(("gpio_free", handle, pin))
        if self._bad_handle(handle):
            return self._fail("bad handle")
        if pin in self._free_faults:
            return self._fail(f"GPIO{pin} free failed")
        self.claimed.discard(pin)
        self.freed.append(pin)
        return 0

    def gpiochip_close(self, handle: int) -> int:
        self.calls.append(("gpiochip_close", handle))
        if self._bad_handle(handle):
            return self._fail("bad handle")
        self._open = False
        self.closed.append(handle)
        return 0
