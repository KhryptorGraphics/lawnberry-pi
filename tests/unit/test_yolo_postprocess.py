"""Unit tests for the pure-numpy YOLOv8 output decoder.

These tests build synthetic ``(1, 84, 8400)`` tensors deterministically (no
randomness, numpy only) and assert the decoder filters, converts, and applies
NMS as expected.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.src.drivers.ai.yolo_postprocess import COCO_CLASSES, decode_yolov8, nms

NUM_ATTRS = 84  # 4 box coords + 80 COCO classes
NUM_ANCHORS = 8400


def _empty_tensor() -> np.ndarray:
    """A zero-filled YOLOv8 output tensor (all anchors below threshold)."""
    return np.zeros((1, NUM_ATTRS, NUM_ANCHORS), dtype=np.float32)


def _set_anchor(
    tensor: np.ndarray,
    anchor: int,
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    class_id: int,
    score: float,
) -> None:
    """Write a box + single high class score into one anchor column."""
    tensor[0, 0, anchor] = cx
    tensor[0, 1, anchor] = cy
    tensor[0, 2, anchor] = w
    tensor[0, 3, anchor] = h
    tensor[0, 4 + class_id, anchor] = score


def test_single_high_confidence_detection():
    tensor = _empty_tensor()
    # class_id 0 == "person"
    _set_anchor(tensor, 100, cx=320.0, cy=240.0, w=50.0, h=80.0, class_id=0, score=0.9)

    detections = decode_yolov8(tensor, class_names=COCO_CLASSES)

    assert len(detections) == 1
    det = detections[0]
    assert det["class"] == "person"
    assert det["confidence"] == pytest.approx(0.9, abs=1e-5)

    x1, y1, x2, y2 = det["bbox"]
    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    assert x1 == pytest.approx(295.0, abs=1e-3)
    assert y1 == pytest.approx(200.0, abs=1e-3)
    assert x2 == pytest.approx(345.0, abs=1e-3)
    assert y2 == pytest.approx(280.0, abs=1e-3)
    # Plausible, ordered, in-frame box.
    assert x2 > x1 and y2 > y1


def test_overlapping_same_class_boxes_collapse_after_nms():
    tensor = _empty_tensor()
    # Two heavily overlapping boxes of the same class (2 == "car").
    _set_anchor(tensor, 10, cx=300.0, cy=300.0, w=100.0, h=100.0, class_id=2, score=0.9)
    _set_anchor(tensor, 20, cx=305.0, cy=305.0, w=100.0, h=100.0, class_id=2, score=0.8)

    detections = decode_yolov8(tensor, class_names=COCO_CLASSES)

    assert len(detections) == 1
    # The higher-confidence box survives.
    assert detections[0]["class"] == "car"
    assert detections[0]["confidence"] == pytest.approx(0.9, abs=1e-5)


def test_overlapping_different_class_boxes_both_kept():
    tensor = _empty_tensor()
    # Same location, different classes -> NMS is class-aware, both survive.
    _set_anchor(tensor, 10, cx=300.0, cy=300.0, w=100.0, h=100.0, class_id=2, score=0.9)
    _set_anchor(tensor, 20, cx=302.0, cy=302.0, w=100.0, h=100.0, class_id=5, score=0.85)

    detections = decode_yolov8(tensor, class_names=COCO_CLASSES)

    assert len(detections) == 2
    classes = {d["class"] for d in detections}
    assert classes == {"car", "bus"}


def test_all_below_threshold_returns_empty():
    tensor = _empty_tensor()
    # Every class score is 0.1, well under the 0.25 default threshold.
    tensor[0, 4:, :] = 0.1

    detections = decode_yolov8(tensor)

    assert detections == []


def test_conf_threshold_boundary():
    tensor = _empty_tensor()
    _set_anchor(tensor, 0, cx=100.0, cy=100.0, w=20.0, h=20.0, class_id=1, score=0.30)

    # 0.30 passes the default 0.25 threshold...
    assert len(decode_yolov8(tensor)) == 1
    # ...but not a stricter 0.5 threshold.
    assert decode_yolov8(tensor, conf_threshold=0.5) == []
    # Exactly at the threshold (>=) the detection is kept.
    assert len(decode_yolov8(tensor, conf_threshold=0.30)) == 1


def test_default_class_names_use_index_labels():
    tensor = _empty_tensor()
    _set_anchor(tensor, 5, cx=50.0, cy=50.0, w=10.0, h=10.0, class_id=7, score=0.6)

    detections = decode_yolov8(tensor)  # no class_names provided

    assert len(detections) == 1
    assert detections[0]["class"] == "class_7"


def test_accepts_2d_attribute_major_tensor():
    tensor = _empty_tensor()[0]  # shape (84, 8400)
    tensor[0, 0] = 320.0
    tensor[1, 0] = 240.0
    tensor[2, 0] = 40.0
    tensor[3, 0] = 60.0
    tensor[4 + 0, 0] = 0.7

    detections = decode_yolov8(tensor, class_names=COCO_CLASSES)

    assert len(detections) == 1
    assert detections[0]["class"] == "person"


def test_nms_helper_is_class_aware():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # overlaps box 0
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8], dtype=np.float32)

    same_class = np.array([3, 3])
    assert nms(boxes, scores, same_class, iou_threshold=0.45) == [0]

    diff_class = np.array([3, 4])
    assert sorted(nms(boxes, scores, diff_class, iou_threshold=0.45)) == [0, 1]
