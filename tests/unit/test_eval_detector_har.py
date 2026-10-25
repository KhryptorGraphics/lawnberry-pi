"""``scripts/eval_detector.py --backend har``: the DFC emulation path, with a faked DFC.

The real ``hailo_sdk_client`` only exists in the workshop DFC env. The fake records what the
backend feeds the runner and returns six raw YOLO26 head tensors containing one known detection,
so the letterbox input, the output pairing and the decode are checked end to end.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest

from backend.src.drivers.ai.yolo26 import letterbox_params
from scripts import eval_detector as ev
from workshop.classes import NAMES

SPEC = ev.DetectorSpec(
    arch="yolo26",
    input_size=640,
    class_names=tuple(NAMES),
    conf_threshold=0.25,
    max_det=100,
    tag="t",
)
HOSE = NAMES.index("hose")


def _heads(as_dict: bool):
    """Six head tensors (NHWC): one 'hose' at stride-8 cell (gx=10, gy=20), l,t,r,b = 2,3,4,5."""
    nc = len(NAMES)
    out = {}
    for stride_cells in (80, 40, 20):
        out[f"box{stride_cells}"] = np.zeros((1, stride_cells, stride_cells, 4), np.float32)
        out[f"cls{stride_cells}"] = np.full((1, stride_cells, stride_cells, nc), -10.0, np.float32)
    out["box80"][0, 20, 10] = (2.0, 3.0, 4.0, 5.0)
    out["cls80"][0, 20, 10, HOSE] = 5.0
    return out if as_dict else list(out.values())


def _fake_dfc(monkeypatch, heads, fed=None):
    """A stand-in ``hailo_sdk_client``; every array handed to ``infer`` is appended to ``fed``."""
    calls = []

    class InferenceContext(enum.Enum):
        SDK_QUANTIZED = "q"
        SDK_BIT_EXACT = "b"

    class ClientRunner:
        def __init__(self, har):
            calls.append(("har", har))

        @contextmanager
        def infer_context(self, ctx):
            calls.append(("context", ctx.name))
            yield "ctx-object"
            calls.append(("closed", None))

        def infer(self, ctx, data):
            calls.append(("infer", ctx, data.dtype, data.shape))
            if fed is not None:
                fed.append(data)
            return heads

    module = types.ModuleType("hailo_sdk_client")
    module.ClientRunner = ClientRunner
    module.InferenceContext = InferenceContext
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", module)
    return calls


@pytest.fixture(autouse=True)
def _identity_letterbox(monkeypatch):
    """The test venv has no cv2; a 640x640 source needs no resize, so the identity is faithful."""

    def identity(rgb, size):
        assert rgb.shape[:2] == (size, size)
        return rgb[np.newaxis], letterbox_params(size, size, size)

    monkeypatch.setattr(ev, "letterbox", identity)


@pytest.mark.parametrize("as_dict", [False, True])
def test_har_backend_decodes_the_emulated_heads(monkeypatch, tmp_path, as_dict):
    calls = _fake_dfc(monkeypatch, _heads(as_dict))
    image = np.full((640, 640, 3), 7, dtype=np.uint8)  # 640x640: letterbox is the identity

    async def go():
        async with ev.HarBackend(tmp_path / "m.har", SPEC, "bit-exact") as be:
            return await be.predict(image)

    dets, ms = asyncio.run(go())

    assert calls[0] == ("har", str(tmp_path / "m.har"))
    assert calls[1] == ("context", "SDK_BIT_EXACT")
    assert calls[2] == ("infer", "ctx-object", np.dtype("float32"), (1, 640, 640, 3))
    assert calls[-1] == ("closed", None)  # the inference context is closed on exit
    assert len(dets) == 1 and dets[0]["class"] == "hose" and dets[0]["class_id"] == HOSE
    # x1,y1 = (cell centre - l,t) * 8; x2,y2 = (cell centre + r,b) * 8
    assert dets[0]["bbox"] == pytest.approx([68.0, 140.0, 116.0, 204.0])
    assert dets[0]["confidence"] == pytest.approx(1 / (1 + np.exp(-5.0)), rel=1e-5)
    assert ms >= 0


def test_default_context_is_quantized(monkeypatch, tmp_path):
    calls = _fake_dfc(monkeypatch, _heads(False))

    async def go():
        async with ev.HarBackend(tmp_path / "m.har", SPEC):
            pass

    asyncio.run(go())

    assert ("context", "SDK_QUANTIZED") in calls


def test_har_backend_without_the_dfc_is_a_setup_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", None)  # import raises ImportError

    async def go():
        async with ev.HarBackend(tmp_path / "m.har", SPEC):
            pass

    with pytest.raises(ev.SetupError, match="hailo_sdk_client"):
        asyncio.run(go())


def test_har_backend_needs_an_explicit_spec(tmp_path, capsys):
    har = tmp_path / "m.har"
    har.write_bytes(b"har")

    rc = ev.main(["--backend", "har", "--model", str(har), "--predict", str(har)])

    assert rc == 2
    assert "--backend har needs --spec" in capsys.readouterr().err


def test_backend_factory_passes_the_context(tmp_path):
    har = tmp_path / "m.har"
    har.write_bytes(b"har")
    args = types.SimpleNamespace(backend="har", model=str(har), har_context="bit-exact")

    backend = ev._backend(args, SPEC)

    assert isinstance(backend, ev.HarBackend) and backend.context == "bit-exact"


def _gradient_image() -> np.ndarray:
    """640x640 RGB with a different value in every pixel/channel position (catches any reorder)."""
    return (np.arange(640 * 640 * 3) % 251).astype(np.uint8).reshape(640, 640, 3)


def test_har_backend_feeds_raw_0_to_255_pixels_in_nhwc(monkeypatch, tmp_path):
    # The runner gets the letterbox output as float32, NOT divided by 255: the zoo's model script
    # normalizes inside the network (the HEF's uint8 input path relies on the same layer), so a
    # /255 here would feed the emulated network a dark image and every recall would drop.
    # [UNVERIFIED against a real HAR: needs the DFC env; see the audit, TOOL-06.]
    fed: list[np.ndarray] = []
    _fake_dfc(monkeypatch, _heads(False), fed)
    image = _gradient_image()

    async def go():
        async with ev.HarBackend(tmp_path / "m.har", SPEC) as be:
            await be.predict(image)

    asyncio.run(go())

    [data] = fed
    assert data.dtype == np.float32 and data.shape == (1, 640, 640, 3)
    np.testing.assert_array_equal(data, image[np.newaxis].astype(np.float32))
    assert data.max() > 200  # i.e. still 0..255 pixels, not 0..1


def test_onnx_backend_feeds_nchw_pixels_scaled_to_0_1(monkeypatch, tmp_path):
    # The other side of "directly comparable": ONNX takes NCHW floats in 0..1 while the HAR and
    # the HEF take 0..255 NHWC. Pinned so a change to either side is deliberate.
    fed: dict[str, np.ndarray] = {}

    class Session:
        def __init__(self, path, providers):
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def run(self, outputs, feed):
            fed.update(feed)
            return [np.zeros((1, 3, 6), np.float32)]

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    image = _gradient_image()

    async def go():
        async with ev.OnnxBackend(tmp_path / "m.onnx", SPEC) as be:
            return await be.predict(image)

    dets, _ = asyncio.run(go())

    x = fed["images"]
    assert dets == [] and x.dtype == np.float32 and x.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(x, image.transpose(2, 0, 1)[np.newaxis] / 255.0, rtol=1e-6)


def test_onnx_output_with_a_class_id_the_spec_does_not_have_is_a_setup_error(monkeypatch, tmp_path):
    # A stale spec with fewer names than the model has classes: an IndexError traceback exits 1,
    # which a script reads as "the model failed the gate".
    class Session:
        def __init__(self, path, providers):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def run(self, outputs, feed):
            return [np.array([[[10, 10, 50, 50, 0.9, 30]]], np.float32)]  # class 30 of 24

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    async def go():
        async with ev.OnnxBackend(tmp_path / "m.onnx", SPEC) as be:
            await be.predict(_gradient_image())

    with pytest.raises(ev.SetupError, match="class id 30 but the spec has 24 classes"):
        asyncio.run(go())


def test_a_model_whose_heads_do_not_fit_the_spec_exits_2_not_1(monkeypatch, tmp_path, capsys):
    # decode_yolo26 raises ValueError for head tensors that cannot be paired with the class count.
    _fake_dfc(monkeypatch, _heads(False))
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    (labels / "a.txt").write_text("")
    (tmp_path / "m.har").write_bytes(b"har")
    short = dataclasses.replace(SPEC, class_names=tuple(NAMES[:10]))  # 10 names, 24-class heads
    monkeypatch.setattr(ev, "_load_spec", lambda a: short)
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((640, 640, 3), dtype=np.uint8))

    rc = ev.main(
        ["--backend", "har", "--model", str(tmp_path / "m.har"), "--spec", "s.json"]
        + ["--images", str(images), "--labels", str(labels), "--out", str(tmp_path / "r.json")]
    )

    assert rc == 2
    assert "the model output does not fit the spec" in capsys.readouterr().err
