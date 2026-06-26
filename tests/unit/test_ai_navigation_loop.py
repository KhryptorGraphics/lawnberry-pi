"""Tests for the autonomous AI inference loop in NavigationService.

Drives the loop over a mocked recorder + inference service and asserts it runs,
applies predictions to motors, grows the coverage map causally, and stops
cleanly.
"""

import asyncio

from backend.src.models import NavigationMode, Position
from backend.src.models.action_prediction import ActionPrediction
from backend.src.models.mower_data_frame import GPSData, MowerDataFrame, RTKFixType
from backend.src.nav.location_features import CoverageMap
from backend.src.services.navigation_service import NavigationService


class _FakeRecorder:
    def __init__(self):
        self.i = 0

    async def capture_frame(self):
        self.i += 1
        frame = MowerDataFrame()
        frame.gps = GPSData(
            latitude=40.0 + self.i * 1e-5,
            longitude=-83.0 + self.i * 1e-5,
            rtk_fix_type=RTKFixType.FIXED,
            num_satellites=20,
        )
        return frame


class _FakeAI:
    def __init__(self):
        self.calls = 0
        self.last_coverage = None

    async def infer(self, frame, coverage_grid=None, datum=None):
        self.calls += 1
        self.last_coverage = coverage_grid
        return ActionPrediction(steering=0.0, throttle=0.5, blade=True, confidence=0.9)

    async def disable(self):
        pass


async def test_ai_loop_applies_and_grows_coverage(monkeypatch):
    nav = NavigationService()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav.navigation_state.target_velocity = 0.6
    nav.set_home_position(Position(latitude=40.0, longitude=-83.0))
    nav._ai_datum = nav._select_ai_datum()
    nav._coverage = CoverageMap(nav._ai_datum)
    nav._ai_loop_hz = 100.0  # fast ticks for the test

    fake_ai = _FakeAI()
    fake_rec = _FakeRecorder()
    monkeypatch.setattr(
        "backend.src.services.ai_inference_service.get_ai_inference_service",
        lambda: fake_ai,
    )
    monkeypatch.setattr(
        "backend.src.services.perimeter_recorder.get_perimeter_recorder",
        lambda: fake_rec,
    )
    speeds: list[tuple[float, float]] = []

    async def _fake_set_speed(left, right):
        speeds.append((left, right))

    monkeypatch.setattr(nav, "set_speed", _fake_set_speed)

    task = asyncio.create_task(nav._ai_inference_loop())
    await asyncio.sleep(0.15)
    nav.navigation_state.navigation_mode = NavigationMode.MANUAL  # signal stop
    await asyncio.wait_for(task, timeout=2.0)

    assert fake_ai.calls >= 1  # inference ran
    assert len(speeds) >= 1  # predictions applied to motors
    assert nav._coverage.snapshot().sum() > 0.0  # blade on + RTK -> coverage grew


async def test_stop_ai_navigation_cancels_loop(monkeypatch):
    nav = NavigationService()
    nav.navigation_state.navigation_mode = NavigationMode.AI
    nav._ai_datum = (40.0, -83.0)
    nav._coverage = CoverageMap(nav._ai_datum)
    nav._ai_loop_hz = 100.0

    fake_ai = _FakeAI()
    monkeypatch.setattr(
        "backend.src.services.ai_inference_service.get_ai_inference_service",
        lambda: fake_ai,
    )
    monkeypatch.setattr(
        "backend.src.services.perimeter_recorder.get_perimeter_recorder",
        lambda: _FakeRecorder(),
    )

    async def _fake_set_speed(left, right):
        pass

    monkeypatch.setattr(nav, "set_speed", _fake_set_speed)

    nav._ai_task = asyncio.create_task(nav._ai_inference_loop())
    await asyncio.sleep(0.05)
    assert await nav.stop_ai_navigation() is True
    assert nav._ai_task is None
    assert nav.navigation_state.navigation_mode == NavigationMode.MANUAL


async def test_start_ai_navigation_refuses_without_model_on_hardware(monkeypatch):
    """On hardware (not SIM), autonomy must not start from placeholder inference."""
    monkeypatch.setenv("SIM_MODE", "0")
    nav = NavigationService()
    nav.navigation_state.navigation_mode = NavigationMode.IDLE

    class _NoModelAI:
        initialized = True
        _model_loaded = False

        async def enable(self):
            return True

    monkeypatch.setattr(
        "backend.src.services.ai_inference_service.get_ai_inference_service",
        lambda: _NoModelAI(),
    )

    assert await nav.start_ai_navigation() is False
    assert nav.navigation_state.navigation_mode != NavigationMode.AI
    assert nav._ai_task is None
