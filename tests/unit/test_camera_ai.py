"""Unit tests for camera AI frame processing (analyzer hook + Hailo path)."""

import pytest

from backend.src.drivers.ai.hailo_driver import HailoDriver
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


STATUS_KEYS = {
    "state",
    "hef_path",
    "tag",
    "classes",
    "conf_threshold",
    "hardware_available",
    "error",
    "health",
}


async def test_reload_with_missing_hef_fails_cleanly(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "detector.hef"
    monkeypatch.setenv("LAWNBERRY_DETECTOR_HEF", str(missing))
    service = CameraStreamService(sim_mode=True)
    service.sim_mode = False

    resp = await service._handle_client_message({"command": "reload_detector"})

    assert resp["status"] == "success"
    assert resp["data"]["state"] == "failed"
    assert str(missing.with_suffix(".json")) in resp["data"]["error"]
    frame = _frame()
    await service._process_frame_for_ai(frame)
    assert frame.processed_for_ai is True
    assert frame.ai_annotations == []


class _StubDriver:
    hardware_available = True

    def __init__(self):
        self.stopped = False
        self.infers = 0

    async def infer(self, tensor):
        self.infers += 1
        raise AssertionError("released detector must not infer")

    async def stop(self):
        self.stopped = True

    async def health_check(self):
        return {"driver": "stub"}


async def test_release_stops_driver_and_stays_down(monkeypatch):
    service = CameraStreamService(sim_mode=True)
    stub = _StubDriver()
    service._ai_detector = stub
    service._ai_detector_state = "active"

    def _no_init(*args, **kwargs):
        raise AssertionError("released detector re-initialized")

    monkeypatch.setattr(HailoDriver, "__init__", _no_init)
    resp = await service._handle_client_message({"command": "release_detector"})

    assert resp["data"]["state"] == "released"
    assert stub.stopped is True
    frame = _frame()
    await service._process_frame_for_ai(frame)
    assert frame.ai_annotations == []
    assert stub.infers == 0


async def test_detector_status_keys():
    service = CameraStreamService(sim_mode=True)
    service._ai_detector = _StubDriver()
    service._ai_detector_state = "active"

    resp = await service._handle_client_message({"command": "detector_status"})

    assert resp["status"] == "success"
    assert set(resp["data"]) == STATUS_KEYS
    assert resp["data"]["health"] == {"driver": "stub"}
    assert resp["data"]["hardware_available"] is True


async def test_release_waits_for_in_flight_hardware_inference():
    """release_detector must not stop the driver while a frame is inside infer_blocking."""
    import asyncio
    import threading

    from backend.src.drivers.ai.yolo26 import DetectorSpec, letterbox_params

    events: list[str] = []
    entered = threading.Event()

    class BlockingDriver(_StubDriver):
        def infer_blocking(self, tensor):
            entered.set()
            threading.Event().wait(0.2)
            events.append("infer_done")
            return "result"

        async def stop(self):
            events.append("stop")

    service = CameraStreamService(sim_mode=True)
    service._ai_detector = BlockingDriver()
    service._ai_detector_state = "active"
    service._ai_spec = DetectorSpec("yolo26", 640, ("a",), 0.25, 100, "t")
    lb = letterbox_params(640, 640, 640)
    service._frame_to_tensor = lambda frame: ("tensor", lb)
    service._decode_detections = lambda result, lb: [{"result": result}]

    frame = _frame()
    task = asyncio.create_task(service._process_frame_for_ai(frame))
    await asyncio.get_running_loop().run_in_executor(None, entered.wait, 2)
    resp = await service._handle_client_message({"command": "release_detector"})
    await task

    assert events == ["infer_done", "stop"]
    assert resp["data"]["state"] == "released"
    assert frame.ai_annotations == [{"result": "result"}]


async def test_hardware_inference_failure_is_reported_then_cleared():
    from backend.src.drivers.ai.yolo26 import DetectorSpec, letterbox_params

    class FlakyDriver(_StubDriver):
        fail = True

        def infer_blocking(self, tensor):
            if self.fail:
                raise RuntimeError("HAILO_TIMEOUT")
            return "ok"

    service = CameraStreamService(sim_mode=True)
    driver = FlakyDriver()
    service._ai_detector = driver
    service._ai_detector_state = "active"
    service._ai_spec = DetectorSpec("yolo26", 640, ("a",), 0.25, 100, "t")
    lb = letterbox_params(640, 640, 640)
    service._frame_to_tensor = lambda frame: ("tensor", lb)
    service._decode_detections = lambda result, lb: [{"result": result}]

    for _ in range(2):
        await service._process_frame_for_ai(_frame())
    status = (await service._handle_client_message({"command": "detector_status"}))["data"]
    assert status["state"] == "active"
    assert status["error"] == "inference failed (2x): HAILO_TIMEOUT"

    driver.fail = False
    frame = _frame()
    await service._process_frame_for_ai(frame)
    status = (await service._handle_client_message({"command": "detector_status"}))["data"]
    assert frame.ai_annotations == [{"result": "ok"}]
    assert status["error"] is None
