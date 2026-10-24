"""Fakes for the actuation-safety tests (not a test module).

``StallablePCA9685`` models the PCA9685 the way the stop sequence has to be reasoned about: the
registers are "last write wins", every landed pulse is logged with a timestamp, and a chosen write
can be stalled until the test releases it -- and, like the real driver (the I2C transfer runs in a
worker thread), a stalled write that the service abandons still lands later. ``FakeGpiozero``
records every GPIO claim and write with a timestamp so a test can prove what was (not) touched.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from typing import Any

from backend.src.models.tractor_control import EngineState
from backend.src.services.tractor_service import TractorControlService

LEFT, THROTTLE, RIGHT = 0, 1, 2  # default PCA9685 channels
#: What the registers hold once every servo is at neutral / idle (default calibration).
NEUTRAL = {LEFT: 1500, RIGHT: 1500, THROTTLE: 1000}


class StallablePCA9685:
    """A PCA9685 stand-in with last-write-wins registers and stallable writes."""

    last_error = None

    def __init__(self, *, uncancellable: bool = True) -> None:
        self.regs: dict[int, int] = {}
        #: Every pulse that reached the chip, in landing order: ``(monotonic time, channel, us)``.
        self.log: list[tuple[float, int, int]] = []
        #: Set when a stalled write has started (it is "on the bus" but has not landed).
        self.stalled = asyncio.Event()
        self.uncancellable = uncancellable
        self.fail_channels: set[int] = set()
        self._gates: dict[tuple[int, int], asyncio.Event] = {}
        self._tasks: list[asyncio.Future] = []

    def stall(self, channel: int, us: int) -> asyncio.Event:
        """Hang the next write of exactly this pulse until the returned event is set."""
        gate = asyncio.Event()
        self._gates[(channel, us)] = gate
        return gate

    def is_ready(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass

    async def set_pwm_us(self, channel: int, us: int) -> None:
        if channel in self.fail_channels:
            raise OSError("simulated I2C bus fault")
        gate = self._gates.pop((channel, us), None)
        if gate is None:
            self._land(channel, us)
            return
        self.stalled.set()
        landing = asyncio.ensure_future(self._land_after(gate, channel, us))
        self._tasks.append(landing)
        if self.uncancellable:
            await asyncio.shield(landing)  # abandoning the await does not stop the transfer
        else:
            await landing

    async def _land_after(self, gate: asyncio.Event, channel: int, us: int) -> None:
        await gate.wait()
        self._land(channel, us)

    def _land(self, channel: int, us: int) -> None:
        self.regs[channel] = us
        self.log.append((time.monotonic(), channel, us))

    async def verify_pwm_us(self, channel: int, us: int) -> None:
        if self.regs.get(channel) != us:
            raise OSError(f"channel {channel} holds {self.regs.get(channel)}, not {us}")

    def writes(self, channel: int) -> list[int]:
        return [us for _, ch, us in self.log if ch == channel]

    def release_all(self) -> None:
        for gate in self._gates.values():
            gate.set()
        self._gates.clear()


class FakeGpiozero:
    """A fake ``gpiozero`` that records every claim and write (``events``) with a timestamp."""

    def __init__(self, monkeypatch) -> None:
        self.devices: list[Any] = []
        self.events: list[tuple[float, int, str]] = []
        gz = self

        class _OutputDevice:
            def __init__(self, pin, active_high=True, initial_value=False):
                self.pin = pin
                self.value = bool(initial_value)
                gz.devices.append(self)
                gz.events.append((time.monotonic(), pin, "claim"))

            def on(self):
                self.value = True
                gz.events.append((time.monotonic(), self.pin, "on"))

            def off(self):
                self.value = False
                gz.events.append((time.monotonic(), self.pin, "off"))

        module = types.ModuleType("gpiozero")
        module.OutputDevice = _OutputDevice
        monkeypatch.setitem(sys.modules, "gpiozero", module)

    @property
    def claimed_pins(self) -> list[int]:
        return sorted({d.pin for d in self.devices})

    def first(self, pin: int, action: str) -> float | None:
        """Timestamp of the first ``action`` ("claim"/"on"/"off") on ``pin``, else ``None``."""
        return next((t for t, p, a in self.events if p == pin and a == action), None)

    def last(self, pin: int) -> str | None:
        return next((a for _, p, a in reversed(self.events) if p == pin), None)


def mowing_tractor(
    *, pca: StallablePCA9685 | None = None, config: dict | None = None
) -> tuple[TractorControlService, StallablePCA9685]:
    """An enabled, authorized tractor with its engine 'running' on a stallable PCA9685."""
    svc = TractorControlService(config={"enabled": True, **(config or {})})
    chip = pca or StallablePCA9685()
    svc._pca9685 = chip
    svc.authorize()
    svc.state.engine = EngineState.RUNNING  # skip the 800 ms starter pulse
    return svc, chip


def idle_tractor() -> TractorControlService:
    """A disabled tractor (the shipped default): nothing latched, nothing to stop."""
    return TractorControlService(config={"enabled": False})
