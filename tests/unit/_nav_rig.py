"""Shared pieces for the round-3 navigation / actuation tests (not a test module).

Builds on ``_mission_rig`` (a real ``NavigationService`` over the REAL ``RoboHATService`` on a fake
serial with SIM_MODE off) and ``fake_lgpio`` (the IBT-4 pads). Adds an ordered ``Timeline`` of
"what the safety verdict said" and "what reached the firmware" so a test can prove an ORDER
("neutral before the next verdict") without depending on how fast the machine happens to be.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from _mission_rig import FakeFirmware, robohat_platform
from fake_lgpio import FakeLgpio

from backend.src.safety.obstacle_stop import ObstacleStatus
from backend.src.services import blade_service as blade_module
from backend.src.services.blade_service import BladeService

#: What a meter on the IBT-4 inputs (GPIO24 / GPIO25) shows.
OFF = {24: 0, 25: 0}
ON = {24: 1, 25: 0}


class Timeline:
    """One ordered record of safety verdicts and of the lines that reached the firmware."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, float]] = []

    def add(self, kind: str, what: Any) -> None:
        self.events.append((kind, what, time.monotonic()))

    def lines(self) -> list[str]:
        return [what for kind, what, _ in self.events if kind == "line"]

    def index(self, kind: str, what: Any, start: int = 0) -> int:
        """Position of the first ``(kind, what)`` event at or after ``start``; -1 if none."""
        for i in range(start, len(self.events)):
            if self.events[i][0] == kind and self.events[i][1] == what:
                return i
        return -1


class TimelineFirmware(FakeFirmware):
    """``FakeFirmware`` that also writes every line it receives to a ``Timeline``."""

    def __init__(self, svc, timeline: Timeline) -> None:
        super().__init__(svc)
        self.timeline = timeline

    async def send_line(self, line: str) -> bool:
        self.timeline.add("line", line)
        return await super().send_line(line)


def robohat_with_timeline(monkeypatch):
    """``robohat_platform`` whose firmware also feeds a ``Timeline``: ``(svc, fw, tractor, tl)``."""
    svc, fw, tractor = robohat_platform(monkeypatch)
    timeline = Timeline()
    fw = TimelineFirmware(svc, timeline)
    monkeypatch.setattr(svc, "_send_line", fw.send_line)
    return svc, fw, tractor, timeline


class StubGuard:
    """An obstacle guard whose verdict the test flips; every ``assess()`` goes on the timeline.

    ``cap`` is the forward speed cap (m/s) while it does not hold; ``None`` = no cap.
    """

    def __init__(
        self, *, hold: bool = False, cap: float | None = None, timeline: Timeline | None = None
    ) -> None:
        self.hold = hold
        self.cap = cap
        self.timeline = timeline
        self.calls = 0

    def assess(self, now: float | None = None):
        self.calls += 1
        if self.timeline is not None:
            self.timeline.add("verdict", "hold" if self.hold else "clear")
        if self.hold:
            status, cap, reasons = ObstacleStatus.STOP, 0.0, ("test obstacle",)
        elif self.cap is not None:
            status, cap, reasons = ObstacleStatus.SLOW, self.cap, ("test obstacle ahead",)
        else:
            status, cap, reasons = ObstacleStatus.CLEAR, self.cap, ()
        lever_cap = 0.0 if self.hold else (1.0 if cap is None else min(1.0, cap / 0.8))
        return SimpleNamespace(
            hold=self.hold, speed_cap_mps=cap, lever_cap=lever_cap, status=status, reasons=reasons
        )

    def applies_to(self, source: str | None) -> bool:
        return source not in ("operator", "teleop")


async def wait_until(cond, timeout: float = 4.0, step: float = 0.01) -> None:
    """Poll ``cond`` (sync or async) until it is true, or fail the test after ``timeout`` s."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        result = cond()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(step)


def armed_blade(monkeypatch, **cfg) -> tuple[BladeService, FakeLgpio]:
    """An ARMED ``BladeService`` on the fake ``lgpio`` (GPIO24/25), installed as the singleton."""
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    service = BladeService({"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}, **cfg})
    monkeypatch.setattr(blade_module, "_blade_service", service)
    return service, fake


class SensorStub:
    """The sensor manager the calibration routine reads its IMU status from."""

    def __init__(self, status: str = "calibrating") -> None:
        self.status = status
        self.reads = 0

    async def read_all_sensors(self):
        self.reads += 1
        return SimpleNamespace(
            imu=SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0, calibration_status=self.status)
        )


class LiveTof:
    """A ToF pair that always has a fresh, valid reading (``mm`` is the distance both report).

    Every read stamps ``now`` and jitters the distance by one millimetre, so the guard's frozen-
    reading detector (SAF-S2) never sees a constant value, exactly like a real chip.
    """

    status = "online"

    def __init__(self, mm: float = 2000.0) -> None:
        self.mm = mm
        self._n = 0

    def _reading(self):
        self._n += 1
        return SimpleNamespace(
            range_status="valid", distance=self.mm + (self._n % 2), timestamp=datetime.now(UTC)
        )

    @property
    def left_reading(self):
        return self._reading()

    @property
    def right_reading(self):
        return self._reading()


def live_sensors(mm: float = 2000.0) -> SimpleNamespace:
    """What ``app.state.sensor_manager`` is to the real obstacle monitor: a live ToF pair."""
    return SimpleNamespace(tof=LiveTof(mm), ultrasonic=None, ultrasonic_enabled=False)


class _Stub:
    async def initialize(self) -> None:
        pass

    async def start_streaming(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


async def _noop(*_a, **_k):
    return None


async def _flat_imu():
    return (0.0, 0.0)


@contextlib.asynccontextmanager
async def real_lifespan(monkeypatch, tractor, sensors=None):
    """The REAL ``main.lifespan`` (real monitors, real wiring) with only the hardware stubbed.

    Stubbed: the RoboHAT init/shutdown (the test injects its own service), the shutdown park,
    the camera, the telemetry loop, the tilt monitor's IMU reader, and ``main``'s own
    ``is_simulation_mode`` (so the VL53L0X XSHUT hook that drives GPIO is skipped while the
    services under test still run their real, non-simulated branches). ``sensors`` becomes
    ``app.state.sensor_manager``, which the real obstacle monitor samples at 20 Hz.
    """
    import backend.src.main as main

    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(main, "initialize_robohat_service", _noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", _noop)

    async def fake_park(**_k):
        return {"faults": []}

    monkeypatch.setattr(main, "park_actuators", fake_park)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", _noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", _noop)
    monkeypatch.setattr(main, "is_simulation_mode", lambda: True)
    real_monitor = main.TractorSafetyMonitor
    monkeypatch.setattr(
        main,
        "TractorSafetyMonitor",
        lambda *a, **k: real_monitor(*a, imu_reader=_flat_imu, **k),
    )
    main.app.state.sensor_manager = sensors
    try:
        async with main.lifespan(main.app):
            yield main.app
    finally:
        main.app.state.sensor_manager = None
