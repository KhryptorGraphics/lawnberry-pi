"""A clear never reopens the blade path on a tilted machine (audit SEC-06), and the tilt monitor
keeps retrying a blade-off that failed (the review's surviving mutant M12).

``clear_emergency`` used to release the E-stop and interlock latches unconditionally; the monitor
re-latched only on its next 50 ms tick, so ``engage_blade(True)`` could succeed in between on a
machine that was still 45 degrees over. While the monitor runs it is now the tractor's clear
gate: the blade latch stays while the latest valid reading is over the limit and while the tilt
sensor is unavailable (no valid reading younger than ``TILT_READING_MAX_AGE_S``); the E-stop
latch is still released (``status: "partial"``) because the blade path stays closed.
"""

from __future__ import annotations

import asyncio

from _tractor_fakes import mowing_tractor

from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.motor_authorization import MotorAuthorization
from backend.src.safety.safety_triggers import SafetyTriggerManager
from backend.src.safety.tractor_safety_monitor import TractorSafetyMonitor
from backend.src.safety.watchdog import SAFETY_MONITOR_SOURCE, Watchdog


class Imu:
    """Mutable IMU: tilt it, or take the sensor away (returns None)."""

    def __init__(self) -> None:
        self.tilted = False
        self.available = True
        self.value: tuple[float, float] | None = None

    async def __call__(self):
        if self.value is not None:
            return self.value
        if not self.available:
            return None
        return (45.0, 0.0) if self.tilted else (2.0, 1.0)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _monitor(svc, imu: Imu, clock: Clock | None = None) -> TractorSafetyMonitor:
    wd = Watchdog(
        EstopHandler(MotorAuthorization()),
        timeout_ms=60_000,
        required_sources=(SAFETY_MONITOR_SOURCE,),
    )
    kwargs = {} if clock is None else {"clock": clock}
    return TractorSafetyMonitor(
        svc,
        wd,
        tilt_threshold_degrees=30.0,
        trigger_manager=SafetyTriggerManager(),
        imu_reader=imu,
        poll_interval_s=0.01,
        **kwargs,
    )


async def test_a_clear_on_a_tilted_machine_cannot_reopen_the_blade_path():
    svc, _pca = mowing_tractor()
    imu = Imu()
    imu.tilted = True
    monitor = _monitor(svc, imu)
    await monitor.start()
    await asyncio.sleep(0.1)
    assert svc.state.interlock_latched is True

    # Neither call yields to the event loop: the monitor cannot tick in between, which is the
    # window the old clear left open.
    clear = await svc.clear_emergency()
    blade = await svc.engage_blade(True, source="operator")

    assert blade["status"] == "rejected", "the blade path must stay closed on a tilted machine"
    assert clear["status"] == "partial"
    assert clear["still_latched"] == ["blade_interlock:tilt"]
    assert "tilt reading" in clear["reason"] and "over" in clear["reason"]
    assert svc.state.interlock_latched is True and svc.state.blade_engaged is False
    await monitor.stop()
    await svc.shutdown()


async def test_the_estop_is_released_but_the_tilt_latch_stays_while_tilted():
    svc, _pca = mowing_tractor()
    imu = Imu()
    imu.tilted = True
    monitor = _monitor(svc, imu)
    await monitor.start()
    await asyncio.sleep(0.1)
    await svc.emergency_stop()
    assert svc.state.emergency_stop_active is True

    clear = await svc.clear_emergency()

    assert clear["status"] == "partial" and clear["cleared"] == ["emergency_stop"]
    assert svc.state.emergency_stop_active is False, "recovery may need to move: the drive is free"
    assert svc.state.interlock_latched is True
    assert (await svc.set_levers(0.3, 0.3, source="operator"))["status"] == "ok"
    assert "latched (tilt)" in svc.get_state().interlock_reason
    await monitor.stop()
    await svc.shutdown()


async def test_a_clear_keeps_the_blade_latch_while_the_tilt_sensor_is_unavailable():
    svc, _pca = mowing_tractor()
    imu, clock = Imu(), Clock()
    imu.tilted = True
    monitor = _monitor(svc, imu, clock)
    await monitor.start()
    await asyncio.sleep(0.1)
    imu.tilted = False  # level again ...
    await asyncio.sleep(0.1)
    imu.available = False  # ... then the sensor goes away
    clock.t += 5.0
    await asyncio.sleep(0.05)

    clear = await svc.clear_emergency()

    assert clear["status"] == "partial" and "unavailable" in clear["reason"]
    assert (await svc.engage_blade(True, source="operator"))["status"] == "rejected"

    imu.available = True  # the sensor returns, level: now the clear may release it
    await asyncio.sleep(0.1)
    released = await svc.clear_emergency()
    assert released["status"] == "ok" and svc.state.interlock_latched is False
    assert (await svc.engage_blade(True, source="operator"))["status"] == "ok"
    await monitor.stop()
    await svc.shutdown()


async def test_a_clear_on_a_level_machine_with_a_fresh_reading_releases_the_latch():
    svc, _pca = mowing_tractor()
    imu = Imu()
    monitor = _monitor(svc, imu)
    await monitor.start()
    await svc.latch_interlock("tilt", source="tilt_monitor")
    await asyncio.sleep(0.1)  # level readings keep arriving

    clear = await svc.clear_emergency()

    assert clear["status"] == "ok" and svc.state.interlock_latched is False
    await monitor.stop()
    await svc.shutdown()


async def test_a_clear_before_any_valid_reading_keeps_the_blade_latch():
    svc, _pca = mowing_tractor()
    monitor = _monitor(svc, Imu())
    svc.attach_interlock_clear_gate(monitor.interlock_clear_veto)  # attached, never ticked
    await svc.latch_interlock("tilt")

    clear = await svc.clear_emergency()

    assert clear["status"] == "partial"
    assert "has not produced a valid reading" in clear["reason"]
    assert svc.state.interlock_latched is True
    await svc.shutdown()


async def test_a_nan_reading_is_no_reading_and_proves_nothing():
    svc, _pca = mowing_tractor()
    imu = Imu()
    imu.value = (float("nan"), 0.0)
    monitor = _monitor(svc, imu)
    await monitor.start()
    await asyncio.sleep(0.1)

    assert svc.state.interlock_latched is False, "NaN must not latch"
    assert "has not produced a valid reading" in monitor.interlock_clear_veto("tilt")
    await monitor.stop()
    await svc.shutdown()


async def test_a_stopped_monitor_no_longer_vetoes_a_clear():
    svc, _pca = mowing_tractor()
    imu = Imu()
    imu.tilted = True
    monitor = _monitor(svc, imu)
    await monitor.start()
    await asyncio.sleep(0.1)
    await monitor.stop()  # orderly shutdown detaches the gate

    assert (await svc.clear_emergency())["status"] == "ok"
    await svc.shutdown()


async def test_the_monitor_keeps_retrying_a_blade_off_the_relay_refused():
    """Surviving mutant M12: while latched AND still engaged the monitor must retry every tick."""
    svc, _pca = mowing_tractor()
    assert (await svc.engage_blade(True, source="operator"))["status"] == "ok"
    real_set = svc.blade_pto.set
    relay_ok = {"value": False}

    def flaky(on: bool) -> None:
        if not relay_ok["value"]:
            raise OSError("relay GPIO write failed")
        real_set(on)

    svc.blade_pto.set = flaky  # type: ignore[method-assign]
    imu = Imu()
    imu.tilted = True
    monitor = _monitor(svc, imu)
    await monitor.start()
    await asyncio.sleep(0.1)
    assert svc.state.interlock_latched is True and svc.state.blade_engaged is True, (
        "the relay would not switch: do not claim the blade is off"
    )

    relay_ok["value"] = True  # the relay recovers
    await asyncio.sleep(0.1)

    assert svc.state.blade_engaged is False, "the monitor must have retried the blade-off"
    await monitor.stop()
    await svc.shutdown()
