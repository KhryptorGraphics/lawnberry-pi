"""Unit tests for camera AI frame processing (analyzer hook + Hailo path)."""

import pytest

from backend.src.models.camera_stream import CameraFrame, FrameMetadata
from backend.src.services.camera_stream_service import CameraStreamService


def _frame() -> CameraFrame:
    return CameraFrame(metadata=FrameMetadata(frame_id="f1", width=640, height=640))


@pytest.mark.asyncio
async def test_registered_analyzer_takes_precedence():
    service = CameraStreamService(sim_mode=True)
    calls = {}

    def analyzer(frame):
        calls["seen"] = True
        return [{"type": "custom", "objects": [{"class": "obstacle", "confidence": 0.7}]}]

    service.register_ai_analyzer(analyzer)
    frame = _frame()
    await service._process_frame_for_ai(frame)

    assert calls.get("seen") is True
    assert frame.processed_for_ai is True
    assert frame.ai_annotations[0]["type"] == "custom"


@pytest.mark.asyncio
async def test_async_analyzer_supported():
    service = CameraStreamService(sim_mode=True)

    async def analyzer(frame):
        return [{"type": "async-result"}]

    service.register_ai_analyzer(analyzer)
    frame = _frame()
    await service._process_frame_for_ai(frame)
    assert frame.ai_annotations[0]["type"] == "async-result"


@pytest.mark.asyncio
async def test_onboard_detector_runs_in_sim():
    service = CameraStreamService(sim_mode=True)
    frame = _frame()
    await service._process_frame_for_ai(frame)

    assert frame.processed_for_ai is True
    assert frame.ai_annotations, "Expected detector annotations in sim mode"
    annotation = frame.ai_annotations[0]
    assert annotation["type"] == "object_detection"
    # Real inference timing recorded by the (simulated) Hailo driver.
    assert annotation["processing_time_ms"] >= 0.0
    assert annotation["source"] == "simulation"
