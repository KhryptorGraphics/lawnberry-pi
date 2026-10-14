"""YOLO26 head decoding: geometry, letterbox inversion, top-k and spec loading."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from backend.src.drivers.ai.yolo26 import (
    SPEC_SCHEMA,
    DetectorSpec,
    decode_yolo26,
    letterbox_params,
    load_detector_spec,
    spec_path_for,
)

NAMES = tuple(f"c{i}" for i in range(6))
NEG = -20.0  # sigmoid ≈ 2e-9: below any threshold


def _spec(**kw) -> DetectorSpec:
    base = dict(
        arch="yolo26", input_size=640, class_names=NAMES, conf_threshold=0.25, max_det=100, tag="t"
    )
    base.update(kw)
    return DetectorSpec(**base)


def _outputs(nc: int = len(NAMES)) -> dict[str, np.ndarray]:
    out = {}
    for i, g in enumerate((80, 40, 20)):
        out[f"box{i}"] = np.zeros((1, g, g, 4), np.float32)
        out[f"cls{i}"] = np.full((1, g, g, nc), NEG, np.float32)
    return out


IDENTITY = letterbox_params(640, 640, 640)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_single_anchor_geometry():
    out = _outputs()
    out["box0"][0, 20, 10] = 2.0
    out["cls0"][0, 20, 10, 3] = 4.0

    dets = decode_yolo26(out, _spec(), IDENTITY)

    assert len(dets) == 1
    det = dets[0]
    assert det["class"] == NAMES[3]
    assert det["class_id"] == 3
    assert det["confidence"] == pytest.approx(_sigmoid(4.0))
    assert det["bbox"] == pytest.approx([68, 148, 100, 180], abs=1e-4)


def test_letterbox_mapping_1280x720():
    lb = letterbox_params(1280, 720, 640)
    assert lb.scale == 0.5
    assert lb.pad_x == 0
    assert lb.pad_y == 140

    # stride-32 cell (x=4, y=8): centre (144, 272); distances give (100,240)-(200,340).
    out = _outputs()
    out["box2"][0, 8, 4] = [44 / 32, 32 / 32, 56 / 32, 68 / 32]
    out["cls2"][0, 8, 4, 0] = 3.0

    dets = decode_yolo26(out, _spec(), lb)

    assert len(dets) == 1
    assert dets[0]["bbox"] == pytest.approx([200, 200, 400, 400], abs=1e-4)


def test_threshold_boundary_is_inclusive():
    out = _outputs()
    out["box1"][0, 5, 5] = 1.0
    out["cls1"][0, 5, 5, 1] = 0.0  # sigmoid 0.5

    assert len(decode_yolo26(out, _spec(conf_threshold=0.5), IDENTITY)) == 1
    assert len(decode_yolo26(out, _spec(conf_threshold=0.25), IDENTITY)) == 1
    assert decode_yolo26(out, _spec(conf_threshold=0.6), IDENTITY) == []


def test_two_classes_at_one_anchor():
    out = _outputs()
    out["box0"][0, 30, 30] = 1.0
    out["cls0"][0, 30, 30, 0] = 2.0
    out["cls0"][0, 30, 30, 5] = 1.0

    dets = decode_yolo26(out, _spec(), IDENTITY)

    assert [d["class_id"] for d in dets] == [0, 5]
    assert dets[0]["bbox"] == dets[1]["bbox"]


def test_max_det_caps_output():
    out = _outputs()
    for x in range(10):
        out["box0"][0, 0, x] = 1.0
        out["cls0"][0, 0, x, 2] = 1.0 + x * 0.1

    dets = decode_yolo26(out, _spec(max_det=3), IDENTITY)

    assert len(dets) == 3
    expected = [_sigmoid(1.9), _sigmoid(1.8), _sigmoid(1.7)]
    assert [d["confidence"] for d in dets] == pytest.approx(expected)


def test_wrong_tensor_count_or_class_count_raises():
    out = _outputs()
    del out["box2"]
    with pytest.raises(ValueError):
        decode_yolo26(out, _spec(), IDENTITY)

    with pytest.raises(ValueError):
        decode_yolo26(_outputs(nc=7), _spec(), IDENTITY)

    four = tuple("abcd")
    with pytest.raises(ValueError):
        decode_yolo26(_outputs(nc=4), _spec(class_names=four), IDENTITY)


def test_dict_order_does_not_matter():
    out = _outputs()
    rng = np.random.default_rng(0)
    for i in range(3):
        out[f"box{i}"] = rng.uniform(0.5, 3.0, out[f"box{i}"].shape).astype(np.float32)
        out[f"cls{i}"] = rng.normal(-3.0, 2.0, out[f"cls{i}"].shape).astype(np.float32)
    items = list(out.items())
    random.Random(1).shuffle(items)

    assert decode_yolo26(dict(items), _spec(), IDENTITY) == decode_yolo26(out, _spec(), IDENTITY)


def _write_spec(path, **overrides):
    data = {
        "schema": SPEC_SCHEMA,
        "arch": "yolo26",
        "input_size": 640,
        "class_names": list(NAMES),
        "conf_threshold": 0.3,
        "max_det": 50,
        "tag": "x",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_spec_loading(tmp_path):
    hef = tmp_path / "yolo26s_x.hef"
    hef.write_bytes(b"")
    _write_spec(tmp_path / "yolo26s_x.json")
    link = tmp_path / "detector.hef"
    link.symlink_to("yolo26s_x.hef")

    assert spec_path_for(link) == (tmp_path / "yolo26s_x.json").resolve()
    spec = load_detector_spec(link)
    assert spec.class_names == NAMES
    assert spec.conf_threshold == 0.3
    assert spec.max_det == 50
    assert spec.tag == "x"


def test_spec_missing_or_invalid(tmp_path):
    hef = tmp_path / "m.hef"
    with pytest.raises(FileNotFoundError):
        load_detector_spec(hef)

    _write_spec(tmp_path / "m.json", schema="other/1")
    with pytest.raises(ValueError):
        load_detector_spec(hef)

    _write_spec(tmp_path / "m.json", conf_threshold=1.0)
    with pytest.raises(ValueError):
        load_detector_spec(hef)


def test_letterbox_pads_with_gray():
    pytest.importorskip("cv2")
    from backend.src.drivers.ai.yolo26 import letterbox

    img = np.zeros((720, 1280, 3), np.uint8)
    tensor, lb = letterbox(img, 640)

    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.uint8
    assert (tensor[0, :140] == 114).all()
    assert (tensor[0, 140:500] == 0).all()
    assert (tensor[0, 500:] == 114).all()
    assert lb.pad_y == 140
