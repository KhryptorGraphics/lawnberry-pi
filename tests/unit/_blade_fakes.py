"""Shared helpers for the IBT-4 blade tests of review round 4 (a manual clock, an armed service).

Everything runs against ``fake_lgpio.FakeLgpio`` (the adapter has never run on hardware):
``fake.pad`` is the last level written to each pin, ``fake.hold_pad`` makes the pad ignore writes
and read a fixed level, like a pin held by something else. SIM_MODE must be off for the real
adapter to be armed, so every module using these calls :func:`blade_world` from an autouse
fixture.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from fake_lgpio import FakeLgpio

from backend.src.core.globals import _safety_state
from backend.src.services.blade_service import BladeService

OFF = {24: 0, 25: 0}
ON = {24: 1, 25: 0}


class Clock:
    """A manual monotonic clock: the deadman is proved without sleeping through it."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def blade_world(monkeypatch) -> None:
    """SIM_MODE off, no E-stop latch, no tractor, no RoboHAT: the blade service alone."""
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    _safety_state["emergency_stop_active"] = False
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: None)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)


def armed(
    monkeypatch, fake: FakeLgpio | None = None, clock: Callable[[], float] | None = None, **cfg: Any
) -> tuple[BladeService, FakeLgpio]:
    """A service armed over ``fake`` (a fresh one by default); ``cfg`` overrides the config."""
    fake = fake or FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    config = {"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}, **cfg}
    return BladeService(config, clock=clock or Clock()), fake


async def eventually(cond: Callable[[], Any], timeout: float = 3.0) -> None:
    """Poll ``cond`` (sync or async) until true; the supervisor runs every 50 ms."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        result = cond()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(0.02)


async def on(svc: BladeService) -> None:
    """Initialise the service and switch the blade ON (both must succeed)."""
    assert await svc.initialize() is True
    assert await svc.set_active(True) is True


def fake_robohat(*, latched: bool = False, blade_on: bool = False) -> SimpleNamespace:
    """The two attributes of ``RoboHATService`` the safety code reads."""
    return SimpleNamespace(emergency_latched=latched, blade_commanded_on=blade_on)
