"""Shutdown park latches; the lifespan stops the loops before it parks (audit SF-06, SEC-08).

``park()`` used to drive the outputs neutral and latch nothing, while the lifespan never stopped a
running mission: the 5 Hz navigation loop's next command landed after the park, and the PCA9685
keeps its last pulse after the process exits (the new deadman cannot help once the process is
gone). Now ``park()`` latches (only neutral / idle / blade-off pass until ``initialize()``) and
the lifespan cancels mission, AI and navigation loops BEFORE the park.
"""

from __future__ import annotations

import asyncio
import time

from _tractor_fakes import NEUTRAL, mowing_tractor

import backend.src.main as main
from backend.src.models.tractor_control import EngineState, TractorCommand


async def test_park_latches_so_a_surviving_loop_cannot_recommand_the_machine():
    svc, pca = mowing_tractor()
    await svc.set_levers(0.4, 0.4, source="operator")
    await svc.set_throttle(0.6, source="operator")

    park = await svc.park()

    assert park["status"] == "parked" and park["complete"] is True
    assert svc.parked is True
    assert svc.state.emergency_stop_active is False, "a park is not an E-stop"
    for refused in (
        await svc.set_levers(0.6, 0.6, source="navigation"),
        await svc.set_left_lever(0.6),
        await svc.set_right_lever(0.6),
        await svc.set_throttle(0.7),
        await svc.engage_blade(True),
        await svc.start_engine(),
    ):
        assert refused["status"] == "rejected" and "parked" in refused["reason"], refused
    assert pca.regs == NEUTRAL
    # Neutral, idle and blade-off still pass: the shutdown path and the hold handlers use them.
    assert (await svc.set_levers(0.0, 0.0))["status"] == "ok"
    assert (await svc.set_throttle(0.0))["status"] == "ok"
    assert (await svc.engage_blade(False))["status"] == "ok"
    assert "parked" in svc.get_state().interlock_reason, "the latch stays visible"

    await svc.initialize()  # a fresh bring-up re-arms the service
    assert svc.parked is False
    assert (await svc.set_levers(0.5, 0.5, source="operator"))["status"] == "ok"
    await svc.shutdown()


async def test_a_navigation_loop_that_outlives_the_park_writes_nothing_after_it():
    svc, pca = mowing_tractor()
    command = TractorCommand(left_lever=0.3, right_lever=0.3, throttle=0.7, blade_engaged=False)

    async def nav_loop():  # the 5 Hz loop of go_to_waypoint
        while True:
            await svc.apply(command, source="operator")
            await asyncio.sleep(0.05)

    loop = asyncio.create_task(nav_loop())
    await asyncio.sleep(0.3)
    assert svc.state.left_lever == 0.3 and svc.state.throttle == 0.7

    await svc.park()
    after_park = len(pca.log)
    await asyncio.sleep(0.4)
    loop.cancel()
    await asyncio.gather(loop, return_exceptions=True)

    assert all(us == NEUTRAL[ch] for _, ch, us in pca.log[after_park:])
    assert pca.regs == NEUTRAL, "the PCA9685 keeps this pulse after the process exits"
    assert svc.state.left_lever == 0.0 and svc.state.throttle == 0.0
    await svc.shutdown()


async def test_a_command_in_flight_when_the_park_lands_cannot_leave_the_output_off_neutral():
    svc, pca = mowing_tractor()
    release = pca.stall(0, 1950)
    cmd = asyncio.create_task(svc.set_levers(0.9, 0.9, source="operator"))
    await asyncio.wait_for(pca.stalled.wait(), 2)
    park = asyncio.create_task(svc.park())
    await asyncio.sleep(0.3)
    release.set()

    result = await asyncio.wait_for(park, 3)
    await asyncio.wait_for(cmd, 3)

    assert result["complete"] is True and result["inflight_write"] is True
    assert pca.regs == NEUTRAL
    await svc.shutdown()


# ------------------------------- the real lifespan -------------------------------


class _Stub:
    def __init__(self, *_a, **_k) -> None:
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def initialize(self):
        pass

    async def start_streaming(self):
        pass

    async def shutdown(self):
        pass

    def heartbeat(self, *_a):
        pass


async def test_the_lifespan_stops_a_running_mission_before_it_parks(monkeypatch):
    """Real ``main.lifespan`` with a mission-like task commanding the levers at 50 Hz."""
    from backend.src.services import emergency_stop_service as ess
    from backend.src.services import mission_service as mission_module
    from backend.src.services.navigation_service import NavigationService

    tractor, pca = mowing_tractor()
    tractor.state.engine = EngineState.RUNNING
    base = "backend.src.services"
    monkeypatch.setattr(f"{base}.tractor_service.get_tractor_service", lambda: tractor)
    monkeypatch.setattr(f"{base}.robohat_service.get_robohat_service", lambda: None)
    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(main, "initialize_robohat_service", noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", noop)
    monkeypatch.setattr(main, "GPSDegradationMonitor", _Stub)
    monkeypatch.setattr(main, "Watchdog", _Stub)
    monkeypatch.setattr(main, "TractorSafetyMonitor", _Stub)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", noop)

    missions = mission_module.MissionService(NavigationService.get_instance())
    monkeypatch.setattr(mission_module, "_mission_service_instance", missions)
    seen: dict[str, object] = {}
    real_park = ess.park_actuators

    async def recording_park(*args, **kwargs):
        seen["mission_done"] = seen["task"].done()
        seen["writes_before_park"] = len(pca.log)
        seen["parked_at"] = time.monotonic()
        return await real_park(*args, **kwargs)

    monkeypatch.setattr(main, "park_actuators", recording_park)

    async def mission():
        while True:
            await tractor.set_levers(0.5, 0.5, source="operator")
            await asyncio.sleep(0.02)

    async with main.lifespan(main.app):
        tractor.authorize()
        task = asyncio.create_task(mission())
        seen["task"] = task
        missions.mission_tasks["m1"] = task
        await asyncio.sleep(0.2)
        assert tractor.state.left_lever == 0.5

    assert seen["mission_done"] is True, "the mission loop must be gone BEFORE the park"
    assert pca.regs == NEUTRAL, "nothing may re-command the levers after the park"
    assert all(us == NEUTRAL[ch] for _, ch, us in pca.log[seen["writes_before_park"] :])
    assert tractor.parked is True
    await tractor.shutdown()
