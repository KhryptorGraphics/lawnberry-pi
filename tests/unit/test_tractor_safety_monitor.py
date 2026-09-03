"""Unit tests (Constitution Principle VI follow-up): IMU tilt-cutoff + watchdog
heartbeat wiring for the tractor platform.

Covers the gap the constitution's v3.0.0 Sync Impact Report named explicitly:
"IMU tilt-cutoff (200ms) and motor-watchdog requirements... are not yet
wired to backend/src/services/tractor_service.py." Both close via
``TractorSafetyMonitor`` (safety/tractor_safety_monitor.py) and a small
optional ``on_timeout`` extension to the existing ``Watchdog`` (T031).
"""

from __future__ import annotations

import asyncio
import os

import pytest

os.environ.setdefault("SIM_MODE", "1")

from backend.src.safety.estop_handler import EstopHandler  # noqa: E402
from backend.src.safety.motor_authorization import MotorAuthorization  # noqa: E402
from backend.src.safety.safety_triggers import SafetyTriggerManager  # noqa: E402
from backend.src.safety.tractor_safety_monitor import TractorSafetyMonitor  # noqa: E402
from backend.src.safety.watchdog import Watchdog  # noqa: E402
from backend.src.services.tractor_service import TractorControlService  # noqa: E402


def _tractor() -> TractorControlService:
    # enabled=False keeps PCA9685/GPIO access in its SIM-safe no-op path.
    return TractorControlService(config={"enabled": False})


def _watchdog(on_timeout=None, timeout_ms: int = 1000) -> Watchdog:
    return Watchdog(
        EstopHandler(MotorAuthorization()), timeout_ms=timeout_ms, on_timeout=on_timeout
    )


class TestWatchdogOnTimeoutCallback:
    """The additive extension to the existing (already contract-tested) Watchdog."""

    @pytest.mark.asyncio
    async def test_on_timeout_callback_invoked_alongside_estop(self):
        called = asyncio.Event()

        async def on_timeout():
            called.set()

        auth = MotorAuthorization()
        auth.authorize()
        estop = EstopHandler(auth)
        wd = Watchdog(estop, timeout_ms=50, on_timeout=on_timeout)

        await wd.start()
        await asyncio.sleep(0.15)
        await wd.stop()

        assert called.is_set(), "on_timeout callback must fire alongside estop.trigger_estop"
        assert not auth.is_enabled(), "existing estop.trigger_estop behavior must be unchanged"

    @pytest.mark.asyncio
    async def test_on_timeout_exception_does_not_crash_the_loop(self):
        async def on_timeout():
            raise RuntimeError("boom")

        auth = MotorAuthorization()
        auth.authorize()
        estop = EstopHandler(auth)
        wd = Watchdog(estop, timeout_ms=50, on_timeout=on_timeout)

        await wd.start()
        await asyncio.sleep(0.15)
        await wd.stop()  # must not raise / hang

        assert not auth.is_enabled()

    @pytest.mark.asyncio
    async def test_heartbeat_prevents_timeout(self):
        called = asyncio.Event()

        async def on_timeout():
            called.set()

        wd = _watchdog(on_timeout=on_timeout, timeout_ms=50)
        await wd.start()
        for _ in range(5):
            await asyncio.sleep(0.02)
            wd.heartbeat()
        await wd.stop()

        assert not called.is_set(), "steady heartbeats must suppress the timeout entirely"


class TestTractorSafetyMonitorTilt:
    @pytest.mark.asyncio
    async def test_tilt_over_threshold_disengages_engaged_blade(self):
        tractor = _tractor()
        tractor.state.blade_engaged = True
        wd = _watchdog()
        triggers = SafetyTriggerManager()

        async def imu_reader():
            return (35.0, 0.0)  # exceeds a 30 deg threshold

        monitor = TractorSafetyMonitor(
            tractor,
            wd,
            tilt_threshold_degrees=30.0,
            trigger_manager=triggers,
            imu_reader=imu_reader,
            poll_interval_s=0.01,
        )

        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

        assert tractor.state.blade_engaged is False

    @pytest.mark.asyncio
    async def test_tilt_under_threshold_leaves_blade_alone(self):
        tractor = _tractor()
        tractor.state.blade_engaged = True
        wd = _watchdog()
        triggers = SafetyTriggerManager()

        async def imu_reader():
            return (5.0, 5.0)  # well under a 30 deg threshold

        monitor = TractorSafetyMonitor(
            tractor,
            wd,
            tilt_threshold_degrees=30.0,
            trigger_manager=triggers,
            imu_reader=imu_reader,
            poll_interval_s=0.01,
        )

        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

        assert tractor.state.blade_engaged is True

    @pytest.mark.asyncio
    async def test_tilt_does_not_reengage_a_blade_the_operator_turned_off(self):
        """Tilt-cutoff only ever turns the blade OFF; recovery must stay manual."""
        tractor = _tractor()
        tractor.state.blade_engaged = False
        wd = _watchdog()
        triggers = SafetyTriggerManager()

        async def imu_reader():
            return (35.0, 0.0)

        monitor = TractorSafetyMonitor(
            tractor,
            wd,
            tilt_threshold_degrees=30.0,
            trigger_manager=triggers,
            imu_reader=imu_reader,
            poll_interval_s=0.01,
        )

        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

        assert tractor.state.blade_engaged is False

    @pytest.mark.asyncio
    async def test_missing_imu_reading_does_not_raise_or_trip(self):
        tractor = _tractor()
        tractor.state.blade_engaged = True
        wd = _watchdog()
        triggers = SafetyTriggerManager()

        async def imu_reader():
            return None  # sensor unavailable this tick

        monitor = TractorSafetyMonitor(
            tractor,
            wd,
            tilt_threshold_degrees=30.0,
            trigger_manager=triggers,
            imu_reader=imu_reader,
            poll_interval_s=0.01,
        )

        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()  # must not raise / hang

        assert tractor.state.blade_engaged is True


class TestTractorSafetyMonitorHeartbeat:
    @pytest.mark.asyncio
    async def test_monitor_running_suppresses_watchdog_timeout(self):
        """The monitor's own tick loop is the heartbeat source: while it runs,
        a watchdog timeout must never fire on its own."""
        timed_out = asyncio.Event()

        async def on_timeout():
            timed_out.set()

        wd = _watchdog(on_timeout=on_timeout, timeout_ms=30)
        tractor = _tractor()
        triggers = SafetyTriggerManager()

        async def imu_reader():
            return (0.0, 0.0)

        monitor = TractorSafetyMonitor(
            tractor,
            wd,
            tilt_threshold_degrees=30.0,
            trigger_manager=triggers,
            imu_reader=imu_reader,
            poll_interval_s=0.01,  # well under the 30ms watchdog timeout
        )

        await wd.start()
        await monitor.start()
        await asyncio.sleep(0.15)
        await monitor.stop()
        await wd.stop()

        assert not timed_out.is_set(), "monitor ticking at 10ms must keep a 30ms watchdog fed"

    @pytest.mark.asyncio
    async def test_monitor_stopped_lets_watchdog_time_out(self):
        """If the monitor task itself dies, nothing feeds the watchdog and it
        must still fire -- this is the actual failure mode being guarded against."""
        timed_out = asyncio.Event()

        async def on_timeout():
            timed_out.set()

        wd = _watchdog(on_timeout=on_timeout, timeout_ms=30)
        await wd.start()
        # No monitor started at all: nothing ever calls wd.heartbeat().
        await asyncio.sleep(0.1)
        await wd.stop()

        assert timed_out.is_set()
