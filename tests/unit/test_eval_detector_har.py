"""``scripts/eval_detector.py --backend har``: the DFC emulation path, with a faked DFC.

The real ``hailo_sdk_client`` only exists in the workshop DFC env. The fake records what the
backend feeds the runner and returns six raw YOLO26 head tensors containing one known detection,
so the letterbox input, the output pairing and the decode are checked end to end.
"""

from __future__ import annotations

import asyncio
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


def _fake_dfc(monkeypatch, heads):
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
