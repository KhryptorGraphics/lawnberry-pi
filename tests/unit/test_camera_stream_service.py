import asyncio
import io

import numpy as np
import pytest

import backend.src.services.camera_stream_service as camera_module
from backend.src.services.camera_stream_service import CameraStreamService

# Pillow is an on-device image dependency, not part of the base/CI env; skip the
# whole module (rather than fail collection) when it is unavailable.
Image = pytest.importorskip("PIL.Image")


@pytest.mark.asyncio
async def test_slow_camera_client_is_dropped():
    service = CameraStreamService(sim_mode=True)
    service.running = True
    service.stream.statistics.frames_captured = 1
    service._client_drain_timeout = 0.05

    class SlowWriter:
        def __init__(self):
            self.buffer = b""

        def write(self, data):
            self.buffer += data

        async def drain(self):
            await asyncio.sleep(0.2)

    slow_client = SlowWriter()
    service.clients.add(slow_client)

    frame = service._create_frame_object(b"data", (10, 10))

    await service._broadcast_frame_to_clients(frame)

    assert slow_client not in service.clients
    assert service.stream.statistics.transmission_errors > 0


@pytest.mark.parametrize(
    "color_space,pixel_value",
    [
        ("RGB", [10, 40, 180]),
        ("BGR", [180, 40, 10]),
    ],
)
def test_encode_numpy_frame_preserves_expected_rgb(monkeypatch, color_space, pixel_value):
    service = CameraStreamService(sim_mode=True)
    monkeypatch.setattr(camera_module, "OPENCV_AVAILABLE", False)

    frame = np.array([[pixel_value]], dtype=np.uint8)
    encoded = service._encode_numpy_frame_to_jpeg(frame, color_space=color_space)
    assert encoded is not None

    image = Image.open(io.BytesIO(encoded)).convert("RGB")
    rgb_pixel = list(image.getpixel((0, 0)))

    if color_space == "RGB":
        expected = pixel_value
    else:
        # BGR input should round-trip to RGB by swapping channels
        expected = [pixel_value[2], pixel_value[1], pixel_value[0]]

    assert len(rgb_pixel) == len(expected)
    for observed, exp in zip(rgb_pixel, expected, strict=False):
        assert abs(observed - exp) <= 2


def test_picamera_rgb888_frames_encode_with_true_colours(monkeypatch):
    """Picamera2 "RGB888" arrays are [B, G, R]; a red object must stay red in the JPEG."""

    class FakePicamera2:
        def capture_array(self, stream):
            # A red pixel (R=200, G=30, B=20) as Picamera2 RGB888 lays it out.
            return np.full((8, 8, 3), [20, 30, 200], dtype=np.uint8)

    monkeypatch.setattr(camera_module, "PICAMERA_AVAILABLE", True)
    monkeypatch.setattr(camera_module, "Picamera2", FakePicamera2, raising=False)
    monkeypatch.setattr(camera_module, "OPENCV_AVAILABLE", False)
    service = CameraStreamService(sim_mode=True)
    service.camera = FakePicamera2()

    encoded, size = service._capture_real_frame()

    assert size == (8, 8)
    r, g, b = Image.open(io.BytesIO(encoded)).convert("RGB").getpixel((4, 4))
    assert abs(r - 200) <= 3 and abs(g - 30) <= 3 and abs(b - 20) <= 3


@pytest.mark.asyncio
async def test_second_instance_cannot_steal_a_live_socket(tmp_path):
    """A CLI or test that starts a second service must not unlink the live socket."""
    path = tmp_path / "camera.sock"

    owner = CameraStreamService(sim_mode=True)
    owner.socket_path = str(path)
    await owner._setup_ipc_server()
    assert path.exists()

    intruder = CameraStreamService(sim_mode=True)
    intruder.socket_path = str(path)
    with pytest.raises(RuntimeError, match="another live camera service"):
        await intruder._setup_ipc_server()

    # The intruder shutting down must leave the owner's socket alone.
    await intruder.shutdown()
    assert path.exists()

    await owner.shutdown()
    assert not path.exists()


@pytest.mark.asyncio
async def test_stale_socket_is_replaced(tmp_path):
    """A leftover socket file with no listener is stale and may be replaced."""
    path = tmp_path / "stale.sock"
    path.write_text("")

    service = CameraStreamService(sim_mode=True)
    service.socket_path = str(path)
    await service._setup_ipc_server()

    assert service._socket_owned is True
    assert service._socket_is_live()

    await service.shutdown()
    assert not path.exists()
