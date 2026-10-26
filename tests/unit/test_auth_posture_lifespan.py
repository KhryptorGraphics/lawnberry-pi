"""The start-up posture log really runs when the backend starts.

``log_auth_posture()`` is unit-tested in ``test_auth_posture.py``; here the application's own
``lifespan`` is run, with the rest of the backend stubbed out the same way
``test_thor_link_wiring.py::_stub_the_rest_of_the_backend`` does, and the supervision tasks the
tractor started are stopped before the test exits.
"""

from __future__ import annotations

import logging

import pytest

from backend.src.services.tractor_service import TractorControlService

LOGGER = "backend.src.core.auth_posture"


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


async def _noop(*_a, **_k):
    return None


def _stub_the_rest_of_the_backend(monkeypatch, main, tractor) -> None:
    async def park(source="shutdown"):
        return {"faults": [], "complete": True}

    # ``main.app`` is one object for the whole test session: forget what an earlier run set.
    for name in ("tractor_safety_monitor", "tractor_watchdog", "thor_link", "thor_link_error"):
        main.app.state._state.pop(name, None)
    monkeypatch.setattr(main, "get_tractor_service", lambda: tractor)
    monkeypatch.setattr(main, "initialize_robohat_service", _noop)
    monkeypatch.setattr(main, "shutdown_robohat_service", _noop)
    monkeypatch.setattr(main, "park_actuators", park)
    monkeypatch.setattr(main, "GPSDegradationMonitor", _Stub)
    monkeypatch.setattr(main, "Watchdog", _Stub)
    monkeypatch.setattr(main, "TractorSafetyMonitor", _Stub)
    monkeypatch.setattr(main, "camera_service", _Stub())
    monkeypatch.setattr(main.websocket_hub, "start_telemetry_loop", _noop)
    monkeypatch.setattr(main.websocket_hub, "stop_telemetry_loop", _noop)


@pytest.fixture
def backend(monkeypatch, tmp_path):
    import backend.src.main as main

    monkeypatch.setenv("SIM_MODE", "1")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    tractor = TractorControlService(config={"enabled": True})
    _stub_the_rest_of_the_backend(monkeypatch, main, tractor)
    return main, tractor


async def _run_lifespan(main, tractor, inside=None) -> None:
    try:
        async with main.lifespan(main.app):
            if inside is not None:
                inside()
    finally:
        await tractor.stop_supervision()  # the supervision task the lifespan started
        await tractor.shutdown()


async def test_lifespan_calls_log_auth_posture_once_before_it_loads_anything(backend, monkeypatch):
    main, tractor = backend
    order: list[str] = []
    monkeypatch.setattr(main, "log_auth_posture", lambda: order.append("posture"))
    real_loader = main.ConfigLoader

    class RecordingLoader(real_loader):
        def __init__(self, *args, **kwargs):
            order.append("config_loader")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(main, "ConfigLoader", RecordingLoader)

    seen: list[list[str]] = []
    await _run_lifespan(main, tractor, inside=lambda: seen.append(list(order)))

    assert seen == [["posture", "config_loader"]], "the posture is logged first, at startup"
    assert order.count("posture") == 1


@pytest.mark.parametrize(
    ("value", "fragment"),
    [("0", "operator authentication is DISABLED"), ("maybe", "not a recognised boolean")],
)
async def test_lifespan_reports_a_disabled_or_ambiguous_flag(
    backend, monkeypatch, caplog, value, fragment
):
    main, tractor = backend
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", value)
    reported: list[bool] = []

    def already_reported() -> None:
        messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
        reported.append(any(fragment in message for message in messages))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await _run_lifespan(main, tractor, inside=already_reported)

    assert reported == [True], "the message is logged before the application serves a request"


async def test_lifespan_is_silent_for_an_enforced_configuration(backend, monkeypatch, caplog):
    main, tractor = backend
    monkeypatch.setenv("OPERATOR_AUTH_REQUIRED", "1")
    monkeypatch.delenv("LAWNBERRY_ENABLE_DEBUG_ROUTES", raising=False)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await _run_lifespan(main, tractor)
    assert [r for r in caplog.records if r.name == LOGGER] == []
