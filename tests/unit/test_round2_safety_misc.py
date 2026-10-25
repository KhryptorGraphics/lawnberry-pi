"""Two small second-round pins: the watchdog window edge (W6) and a truthful
``safety_monitor_running`` for the Thor link (audit SAF-12).
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.watchdog import Watchdog
from backend.src.services.tractor_service import TractorControlService

# ------------------------------------------------------------------------------ W6


async def test_a_heartbeat_exactly_one_window_old_is_not_yet_stale_one_tick_later_it_is():
    clock = SimpleNamespace(t=1000.0)
    watchdog = Watchdog(
        EstopHandler(SimpleNamespace(revoke=lambda: None)), timeout_ms=1000, clock=lambda: clock.t
    )
    clock.t += 1.0  # exactly timeout_ms since the last beat
    assert watchdog.stale_sources() == {}
    assert watchdog.snapshot()["sources"]["default"]["stale"] is False
    assert await watchdog.evaluate() is False and not watchdog.tripped
    clock.t += 0.001
    assert set(watchdog.stale_sources()) == {"default"}
    assert watchdog.snapshot()["sources"]["default"]["stale"] is True
    assert await watchdog.evaluate() is True and watchdog.tripped


# ------------------------------------------------------------------------------ SAF-12


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


class _MonitorThatCannotStart(_Stub):
    async def start(self):
        raise RuntimeError("the tilt monitor failed to start")


async def _noop(*_a, **_k):
    return None


async def run_lifespan(monkeypatch, tmp_path, monitor_cls):
    """The real lifespan with the rest of the backend stubbed; returns what the Thor link got."""
    import backend.src.main as main

    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    tractor = TractorControlService(config={"enabled": True})
    for name in ("tractor_safety_monitor", "tractor_watchdog", "thor_link", "thor_link_error"):
        main.app.state._state.pop(name, None)  # main.app is one object for the whole session

    async def park(source="shutdown"):
        return {"faults": [], "complete": True}

    seen: dict = {}

    async def thor_link(_tractor, *, safety_monitor_running, gnss):
        seen["safety_monitor_running"] = safety_monitor_running
        return None

    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)
    monkeypatch.setattr(main, "initialize_robohat_service", _noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", _noop)
    monkeypatch.setattr(main, "park_actuators", park)
    monkeypatch.setattr(main, "GPSDegradationMonitor", _Stub)
    monkeypatch.setattr(main, "Watchdog", _Stub)
    monkeypatch.setattr(main, "TractorSafetyMonitor", monitor_cls)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main, "_fail_closed", _noop)
    monkeypatch.setattr(main, "start_thor_link", thor_link)
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", _noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", _noop)
    try:
        async with main.lifespan(main.app):
            pass
    finally:
        await tractor.stop_supervision()
        await tractor.shutdown()
    return seen


async def test_the_thor_link_is_told_the_safety_monitor_is_running_only_once_it_started(
    monkeypatch, tmp_path
):
    seen = await run_lifespan(monkeypatch, tmp_path, _Stub)
    assert seen == {"safety_monitor_running": True}


async def test_a_monitor_object_that_failed_to_start_is_not_a_running_monitor(
    monkeypatch, tmp_path
):
    seen = await run_lifespan(monkeypatch, tmp_path, _MonitorThatCannotStart)
    assert seen == {"safety_monitor_running": False}
