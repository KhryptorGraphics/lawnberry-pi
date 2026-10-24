"""Actuation-path watchdog (audit N-04, N-05, WP-P1-08, D.4; acceptance criteria section 7).

The heartbeat the watchdog needs from the actuation path is not "a loop is turning": it is the
tractor service's own self-check (``supervise_once``) coming back clean -- every servo channel
read back from the PCA9685 holds what was commanded, no write fault is outstanding, an armed
command hold has a running deadman, and a latched stop really is at the safe state. Faults are
injected where they happen: a chip that resets, a bus that fails, a loop that dies, a deadman that
hangs. ``FakeChip`` keeps real register semantics (a pulse persists until written or reset), so a
write-only keepalive could not have seen the reset.

SIM_MODE: nothing here touches hardware. A *hardware* watchdog (audit O-7) is still required.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace

import pytest

import backend.src.main as main
from backend.src.models.tractor_control import EngineState
from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.safety_triggers import SafetyTriggerManager
from backend.src.safety.tractor_safety_monitor import TractorSafetyMonitor
from backend.src.safety.watchdog import ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE, Watchdog
from backend.src.services.tractor_service import TractorControlService

LEFT, THROTTLE, RIGHT = 0, 1, 2  # default PCA9685 channels


class FakeChip:
    """A PCA9685 stand-in: a pulse persists until it is written again or the chip resets."""

    last_error = None

    def __init__(self) -> None:
        self.regs: dict[int, int] = {}
        self.fail_writes = False
        self.fail_reads = False
        self.hang_reads = False
        self.verify_calls = 0
        self.ready = True

    def is_ready(self) -> bool:
        return self.ready

    async def initialize(self) -> None:
        pass

    async def set_pwm_us(self, channel: int, us: int) -> None:
        if self.fail_writes:
            raise OSError("simulated I2C bus fault")
        self.regs[channel] = us

    async def verify_pwm_us(self, channel: int, us: int) -> None:
        self.verify_calls += 1
        if self.hang_reads:
            await asyncio.sleep(3600)
        if self.fail_reads:
            raise OSError("simulated I2C read failure")
        if self.regs.get(channel) != us:
            raise RuntimeError(f"channel {channel} holds {self.regs.get(channel)}, not {us}")

    def reset(self) -> None:
        """A brown-out: every output is back to its power-on state."""
        self.regs.clear()


async def _tractor(*, interval_ms: int = 20, mowing: bool = False):
    svc = TractorControlService(
        config={"enabled": True, "actuation_supervision": {"interval_ms": interval_ms}}
    )
    chip = FakeChip()
    svc._pca9685 = chip
    svc.ACTUATION_TIMEOUT_S = 0.1
    await svc.initialize()  # parks levers/throttle/blade: the chip now holds the neutral pulses
    svc.authorize()
    svc.state.engine = EngineState.RUNNING  # skip the 800 ms starter pulse
    if mowing:
        assert (await svc.engage_blade(True))["status"] == "ok"
        assert (await svc.set_levers(0.6, 0.6, duration_ms=2000))["status"] == "ok"
    return svc, chip


async def _flat_imu():
    return (0.0, 0.0)


@contextlib.asynccontextmanager
async def running(*, timeout_ms: int = 200, retrip_s: float = 0.05):
    """The production wiring of main.py on small windows: watchdog + self-check + tilt monitor."""
    svc, chip = await _tractor(mowing=True)
    events: list[tuple[str, str]] = []
    svc.on_change = lambda kind, values, source: events.append((kind, source))

    async def watchdog_stop():
        return await svc.emergency_stop(source="watchdog")

    wd = Watchdog(
        EstopHandler(svc),
        timeout_ms=timeout_ms,
        on_timeout=watchdog_stop,
        required_sources=(ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE),
        retrip_interval_s=retrip_s,
        poll_interval_s=0.005,
    )
    svc.attach_heartbeat(lambda: wd.heartbeat(ACTUATION_SOURCE))
    monitor = TractorSafetyMonitor(
        svc,
        wd,
        tilt_threshold_degrees=30.0,
        trigger_manager=SafetyTriggerManager(),
        imu_reader=_flat_imu,
        poll_interval_s=0.01,
    )
    await wd.start()
    await svc.start_supervision()
    await monitor.start()
    try:
        yield SimpleNamespace(svc=svc, chip=chip, wd=wd, monitor=monitor, events=events)
    finally:
        await wd.stop()
        await monitor.stop()
        await svc.shutdown()


# ------------------------------------------------------------ a healthy machine


async def test_a_healthy_machine_is_never_tripped_and_the_self_check_really_ran():
    async with running() as r:
        await asyncio.sleep(0.6)  # three windows
        assert r.wd.trips == 0 and not r.wd.tripped
        assert r.chip.verify_calls >= 15, "the outputs were read back, not assumed"
        state = r.svc.get_state()
        assert state.blade_engaged and state.left_lever == 0.6
        assert state.actuator_ok and state.actuator_fault is None
        assert not r.svc.state.emergency_stop_active


# ----------------------------------------- the faults a loop-liveness beat missed


async def test_a_chip_reset_trips_the_dog_the_stop_rewrites_neutral_and_it_rearms():
    """A brown-out resets the PCA9685 while the loops keep turning: only a read-back sees it."""
    async with running() as r:
        await asyncio.sleep(0.15)
        assert r.wd.trips == 0
        r.chip.reset()
        await asyncio.sleep(0.8)

        assert r.wd.trips >= 1
        s = r.svc.state
        assert s.emergency_stop_active and not s.blade_engaged
        assert s.left_lever == 0.0 and s.right_lever == 0.0
        assert r.chip.regs[LEFT] == 1500 and r.chip.regs[RIGHT] == 1500, "neutral was rewritten"
        assert ("emergency_stop", "watchdog") in r.events
        assert not r.wd.tripped, "re-armed: the outputs read back correctly again"
        trips = r.wd.trips
        await asyncio.sleep(0.3)
        assert r.wd.trips == trips, "no further stops once verified"
        assert s.emergency_stop_active, "the latch stays until an operator clears it"


async def test_a_dead_bus_is_stopped_again_and_again_until_the_neutral_lands():
    """The one-shot baseline tried once; a stop whose neutral write fails must be retried."""
    async with running(timeout_ms=150) as r:
        await asyncio.sleep(0.1)
        r.chip.fail_writes = True
        r.chip.fail_reads = True
        await asyncio.sleep(0.8)

        assert r.wd.trips >= 3, "bounded repeats, not a single attempt"
        s = r.svc.state
        assert s.emergency_stop_active and r.wd.tripped
        assert s.left_lever == 0.6, "state follows the outputs: the neutral never landed"
        assert not s.blade_engaged, "the blade relay is GPIO: it does not need the I2C bus"

        r.chip.fail_writes = False
        r.chip.fail_reads = False  # the bus comes back
        await asyncio.sleep(0.5)
        assert r.chip.regs[LEFT] == 1500 and r.chip.regs[RIGHT] == 1500
        assert r.svc.state.left_lever == 0.0
        assert not r.wd.tripped


async def test_an_operator_stop_that_failed_is_finished_by_the_watchdog():
    """Reads work but writes do not: the stop is latched with the levers still forward."""
    async with running(timeout_ms=150) as r:
        r.chip.fail_writes = True
        result = await r.svc.emergency_stop(source="operator")
        assert result["complete"] is False
        assert r.svc.state.left_lever == 0.6
        await asyncio.sleep(0.6)

        assert r.wd.trips >= 2, "an unconfirmed latched stop is not a heartbeat"
        assert r.svc.get_state().actuator_ok is False
        r.chip.fail_writes = False
        await asyncio.sleep(0.5)
        assert r.svc.state.left_lever == 0.0 and r.chip.regs[LEFT] == 1500
        assert not r.wd.tripped


async def test_a_latched_stop_that_has_not_reached_the_safe_state_is_not_a_heartbeat():
    svc, _chip = await _tractor(mowing=True)
    svc.state.emergency_stop_active = True  # what emergency_stop sets first, before any output
    reason = await svc.supervise_once()
    assert reason is not None and "E-stop is latched but" in reason
    assert "left_lever" in reason and "blade" in reason

    await svc.emergency_stop(source="operator")  # the outputs reach the safe state
    assert await svc.supervise_once() is None, "verified neutral under a latched stop is healthy"
    await svc.shutdown()


async def test_a_dead_actuation_loop_trips_even_while_the_tilt_monitor_keeps_beating():
    async with running(timeout_ms=150) as r:
        await asyncio.sleep(0.1)
        assert r.wd.trips == 0
        await r.svc.stop_supervision()  # the actuator control loop dies
        await asyncio.sleep(0.5)

        assert r.wd.trips >= 1
        reason = r.wd.snapshot()["last_trip_reason"]
        assert ACTUATION_SOURCE in reason and SAFETY_MONITOR_SOURCE not in reason
        assert r.svc.state.emergency_stop_active and not r.svc.state.blade_engaged


# ----------------------------------------------- what withholds the heartbeat


async def test_a_hold_nobody_will_expire_withholds_the_heartbeat():
    svc, _chip = await _tractor()
    await svc.set_levers(0.5, 0.5, duration_ms=2000)
    assert await svc.supervise_once() is None

    svc._deadman_task.cancel()  # the deadman dies and the hold stays armed
    await asyncio.sleep(0.01)
    reason = await svc.supervise_once()
    assert reason is not None and "deadman task is not running" in reason
    await svc.shutdown()


async def test_a_deadman_that_hangs_past_its_grace_withholds_the_heartbeat():
    svc, _chip = await _tractor()
    svc.ACTUATION_TIMEOUT_S = 0.05
    svc.supervise_interval_s = 0.02  # grace 0.07 s

    async def stuck():
        await asyncio.sleep(3600)

    svc._expire_due = stuck  # when the hold runs out the deadman hangs
    await svc.set_levers(0.5, 0.5, duration_ms=30)
    await asyncio.sleep(0.2)
    reason = await svc.supervise_once()
    assert reason is not None and "overdue" in reason
    await svc.shutdown()


async def test_a_blade_that_will_not_switch_off_under_a_latched_interlock_is_not_a_heartbeat(
    monkeypatch,
):
    svc, _chip = await _tractor()
    await svc.engage_blade(True)

    def stuck_relay(on):
        raise RuntimeError("relay driver fault")

    monkeypatch.setattr(svc.blade_pto, "set", stuck_relay)
    await svc.latch_interlock("tilt", source="tilt_monitor")
    assert svc.state.interlock_latched and svc.state.blade_engaged, "state does not lie"
    reason = await svc.supervise_once()
    assert reason is not None and "latched but the blade is engaged" in reason
    await svc.shutdown()


async def test_an_outstanding_write_fault_withholds_the_heartbeat_until_a_write_succeeds():
    svc, chip = await _tractor()
    chip.fail_writes = True
    with pytest.raises(OSError):
        await svc.set_levers(0.3, 0.3)
    reason = await svc.supervise_once()
    assert reason is not None and "actuator fault outstanding" in reason

    chip.fail_writes = False
    assert (await svc.set_levers(0.0, 0.0))["status"] == "ok"
    assert await svc.supervise_once() is None
    await svc.shutdown()


async def test_an_absent_pca9685_is_not_a_heartbeat():
    svc, chip = await _tractor()
    chip.ready = False
    chip.last_error = "PCA9685 not available at 0x40"
    try:
        reason = await svc.supervise_once()
        assert reason is not None and "not available" in reason
        assert svc.get_state().actuator_ok is False
    finally:
        chip.last_error = None
        await svc.shutdown()


async def test_a_hung_readback_times_out_releases_the_lock_and_does_not_block_commands():
    svc, chip = await _tractor()
    svc.ACTUATION_TIMEOUT_S = 0.05
    chip.hang_reads = True
    started = time.monotonic()
    reason = await svc.supervise_once()
    assert reason is not None and "timed out" in reason
    assert time.monotonic() - started < 1.0
    assert not svc._act_lock.locked(), "the lock must not be held across a hung read"

    chip.hang_reads = False
    assert (await svc.set_levers(0.1, 0.1))["status"] == "ok"
    assert await svc.supervise_once() is None
    await svc.shutdown()


async def test_a_crashing_self_check_has_not_passed_and_the_loop_survives():
    svc, _chip = await _tractor()
    calls = 0

    def exploding_beat():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyError("misconfigured watchdog source")

    svc.attach_heartbeat(exploding_beat)
    reason = await svc.supervise_once()
    assert reason is not None and "crashed" in reason and "KeyError" in reason
    assert await svc.supervise_once() is None, "the next cycle recovers"

    await svc.start_supervision()
    await asyncio.sleep(0.1)
    assert svc._supervise_task is not None and not svc._supervise_task.done()
    await svc.shutdown()
    assert svc._supervise_task is None


async def test_the_self_check_never_races_a_command():
    """A command lands in the chip a moment before the service records it. Without the lock the
    read-back would see the new pulse against the old expectation and report a false fault."""
    svc, chip = await _tractor()
    write = chip.set_pwm_us

    async def slow_write(channel: int, us: int) -> None:
        await asyncio.sleep(0.002)
        await write(channel, us)

    chip.set_pwm_us = slow_write

    async def commander():
        for i in range(40):
            level = 0.2 if i % 2 else 0.4
            await svc.set_levers(level, level, duration_ms=2000)

    async def checker():
        out = []
        for _ in range(80):
            out.append(await svc.supervise_once())
            await asyncio.sleep(0.001)
        return out

    _, verdicts = await asyncio.gather(commander(), checker())
    assert all(v is None for v in verdicts), [v for v in verdicts if v]
    await svc.shutdown()


# ------------------------------------------------------------------- lock-out


async def test_lock_out_refuses_every_actuation_and_cannot_be_cleared_by_the_api():
    svc, chip = await _tractor(mowing=True)
    svc.lock_out("watchdog failed to start")

    assert not svc.state.authorized
    for rejected in (
        await svc.set_levers(0.3, 0.3),
        await svc.set_throttle(0.5),
        await svc.engage_blade(True),
        await svc.start_engine(),
    ):
        assert rejected["status"] == "rejected"
    cleared = await svc.clear_emergency()
    assert cleared["status"] == "rejected" and "locked out" in cleared["reason"]

    stop = await svc.emergency_stop(source="startup_failure")  # safing is never refused
    assert stop["complete"] is True and chip.regs[LEFT] == 1500 and not svc.state.blade_engaged
    again = await svc.clear_emergency()
    assert again["status"] == "rejected" and svc.state.emergency_stop_active
    await svc.shutdown()


# ------------------------------------------------------- main.py lifespan wiring


class _Stub:
    async def initialize(self):
        pass

    async def start_streaming(self):
        pass

    async def shutdown(self):
        pass


def _wire_lifespan(monkeypatch, svc):
    async def noop(*_a, **_k):
        return None

    async def fake_park(**_k):
        return {"faults": []}

    monkeypatch.setattr(main, "get_tractor_service", lambda: svc)
    monkeypatch.setattr(main, "initialize_robohat_service", noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", noop)
    monkeypatch.setattr(main, "park_actuators", fake_park)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", noop)

    real_monitor = main.TractorSafetyMonitor
    monkeypatch.setattr(
        main,
        "TractorSafetyMonitor",
        lambda *a, **k: real_monitor(*a, imu_reader=_flat_imu, **k),
    )


async def test_lifespan_runs_a_two_source_watchdog_fed_by_the_actuation_self_check(monkeypatch):
    svc = TractorControlService(
        config={"enabled": True, "actuation_supervision": {"interval_ms": 20}}
    )
    chip = FakeChip()
    svc._pca9685 = chip
    events: list[tuple[str, str]] = []
    svc.on_change = lambda kind, values, source: events.append((kind, source))
    _wire_lifespan(monkeypatch, svc)
    real_watchdog = main.Watchdog
    monkeypatch.setattr(
        main,
        "Watchdog",
        lambda *a, **k: real_watchdog(*a, **{**k, "timeout_ms": 150, "retrip_interval_s": 0.05}),
    )

    async with main.lifespan(main.app):
        wd = main.app.state.tractor_watchdog
        assert wd.required_sources == (ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE)
        await asyncio.sleep(0.4)
        assert wd.running and wd.trips == 0, wd.snapshot()
        assert svc._supervise_task is not None and not svc._supervise_task.done()

        chip.reset()  # PCA9685 brown-out
        await asyncio.sleep(0.8)
        assert wd.trips >= 1 and svc.state.emergency_stop_active
        assert ("emergency_stop", "watchdog") in events, "the stop is labelled as the watchdog's"
        assert chip.regs[LEFT] == 1500

    assert not wd.running and svc._supervise_task is None, "teardown stops watchdog and loop"


async def test_lifespan_fails_closed_when_the_safety_wiring_cannot_start(monkeypatch):
    svc = TractorControlService(config={"enabled": True})
    chip = FakeChip()
    svc._pca9685 = chip
    _wire_lifespan(monkeypatch, svc)

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("tilt monitor wiring broke")

    monkeypatch.setattr(main, "TractorSafetyMonitor", Broken)

    async with main.lifespan(main.app):
        # N-05: the actuation API used to stay live with no watchdog and no tilt monitor.
        assert svc.state.emergency_stop_active and not svc.state.authorized
        cleared = await svc.clear_emergency()
        assert cleared["status"] == "rejected" and "locked out" in cleared["reason"]
        assert (await svc.set_levers(0.3, 0.3))["status"] == "rejected"
        assert not main.app.state.tractor_watchdog.running
        assert svc._supervise_task is None
        assert chip.regs[LEFT] == 1500 and not svc.state.blade_engaged


async def test_lifespan_refuses_a_watchdog_window_shorter_than_the_self_check_cadence(
    monkeypatch,
):
    """A window under three self-check periods would trip on a healthy machine: fail closed
    (locked out, E-stopped, no watchdog running) instead of e-stopping it on every cycle."""
    svc = TractorControlService(config={"enabled": True})
    svc._pca9685 = FakeChip()
    svc.supervise_interval_s = 1.0  # 3000 ms minimum window against the shipped 1000 ms
    _wire_lifespan(monkeypatch, svc)

    async with main.lifespan(main.app):
        assert main.app.state.tractor_watchdog is None
        assert svc.state.emergency_stop_active
        cleared = await svc.clear_emergency()
        assert cleared["status"] == "rejected" and "locked out" in cleared["reason"]
        assert svc._supervise_task is None
