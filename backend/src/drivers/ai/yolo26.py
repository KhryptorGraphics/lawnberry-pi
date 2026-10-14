"""YOLO26 detector support for Hailo HEFs compiled with model zoo v2.19.1.

A YOLO26 HEF (``hailomz compile yolo26s``) ends at the six ``one2one`` head
convolutions, so the device returns raw per-level tensors:

* three box tensors ``(H, W, 4)``: distances ``l, t, r, b`` from the cell centre,
  in grid units;
* three class tensors ``(H, W, nc)``: class logits.

There is no on-chip NMS; YOLO26's one-to-one head is NMS-free, so decoding is a
sigmoid, a top-k and a threshold (zoo ``_yolo6_decode`` + ``yolo26_filter``).

Each HEF ships with a JSON sidecar (``<hef stem>.json`` next to the resolved HEF)
holding class names and thresholds; see :func:`load_detector_spec`.

numpy only; ``cv2`` is imported lazily inside :func:`letterbox`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPEC_SCHEMA = "lawnberry-detector/1"
LETTERBOX_FILL = 114


@dataclass(frozen=True)
class DetectorSpec:
    arch: str
    input_size: int
    class_names: tuple[str, ...]
    conf_threshold: float
    max_det: int
    tag: str


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int


def spec_path_for(hef_path: str | Path) -> Path:
    """Sidecar path: next to the symlink *target*, so swapping the link swaps the spec."""
    return Path(hef_path).resolve().with_suffix(".json")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def load_detector_spec(hef_path: str | Path) -> DetectorSpec:
    """Load the sidecar spec of ``hef_path``; ``tag`` defaults to the HEF stem."""
    return load_spec_file(spec_path_for(hef_path))


def load_spec_file(spec_path: str | Path) -> DetectorSpec:
    path = Path(spec_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"detector spec not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec must be a JSON object")
    if raw.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r} != {SPEC_SCHEMA!r}")
    if raw.get("arch") != "yolo26":
        raise ValueError(f"{path}: unsupported arch {raw.get('arch')!r}")
    input_size = raw.get("input_size")
    if not _positive_int(input_size):
        raise ValueError(f"{path}: input_size must be a positive int")
    names = raw.get("class_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{path}: class_names must be a non-empty list of str")
    conf = raw.get("conf_threshold", 0.25)
    if isinstance(conf, bool) or not isinstance(conf, int | float) or not 0.0 < conf < 1.0:
        raise ValueError(f"{path}: conf_threshold must be in (0, 1)")
    max_det = raw.get("max_det", 100)
    if not _positive_int(max_det):
        raise ValueError(f"{path}: max_det must be a positive int")
    tag = raw.get("tag") or path.stem
    return DetectorSpec(
        arch="yolo26",
        input_size=input_size,
        class_names=tuple(names),
        conf_threshold=float(conf),
        max_det=max_det,
        tag=str(tag),
    )


def letterbox_params(src_w: int, src_h: int, size: int) -> Letterbox:
    scale = min(size / src_w, size / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    return Letterbox(
        scale=scale,
        pad_x=(size - new_w) / 2,
        pad_y=(size - new_h) / 2,
        src_w=src_w,
        src_h=src_h,
    )


def letterbox(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize ``image_rgb`` (uint8 HxWx3) into a gray-padded ``(1, size, size, 3)`` tensor."""
    import cv2

    h, w = image_rgb.shape[:2]
    lb = letterbox_params(w, h, size)
    new_w = round(w * lb.scale)
    new_h = round(h * lb.scale)
    canvas = np.full((size, size, 3), LETTERBOX_FILL, dtype=np.uint8)
    resized = (
        image_rgb
        if (new_w, new_h) == (w, h)
        else cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    )
    # Ultralytics LetterBox convention: round(pad - 0.1) puts the odd pixel bottom/right.
    top = int(round(lb.pad_y - 0.1))
    left = int(round(lb.pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas[np.newaxis], lb


def _pair_levels(
    outputs: Mapping[str, np.ndarray], num_classes: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if num_classes == 4:
        raise ValueError("4-class detectors are ambiguous with box tensors; unsupported")
    tensors = []
    for name, arr in outputs.items():
        a = np.asarray(arr)
        while a.ndim > 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3:
            raise ValueError(f"output {name!r} has shape {np.shape(arr)}, expected (H,W,C)")
        tensors.append(a)
    if len(tensors) != 6:
        raise ValueError(f"expected 6 output tensors, got {len(tensors)}")
    boxes = {t.shape[:2]: t for t in tensors if t.shape[2] == 4}
    scores = {t.shape[:2]: t for t in tensors if t.shape[2] == num_classes}
    if len(boxes) != 3 or len(scores) != 3 or set(boxes) != set(scores):
        shapes = sorted(t.shape for t in tensors)
        raise ValueError(f"cannot pair box/class tensors for {num_classes} classes: {shapes}")
    # Fine-to-coarse order keeps anchor indexing deterministic regardless of dict order.
    return [(boxes[hw], scores[hw]) for hw in sorted(boxes, key=lambda s: (-s[0], -s[1]))]


def decode_yolo26(
    outputs: Mapping[str, np.ndarray], spec: DetectorSpec, lb: Letterbox
) -> list[dict[str, Any]]:
    """Decode raw YOLO26 head tensors into detections in source-image pixels."""
    nc = len(spec.class_names)
    all_boxes = []
    all_scores = []
    for box_t, cls_t in _pair_levels(outputs, nc):
        h, w = box_t.shape[:2]
        stride = spec.input_size / h
        gy, gx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        d = box_t.astype(np.float32)
        xyxy = (
            np.stack((gx - d[..., 0], gy - d[..., 1], gx + d[..., 2], gy + d[..., 3]), axis=-1)
            * stride
        )
        all_boxes.append(xyxy.reshape(-1, 4))
        logits = cls_t.astype(np.float32).reshape(-1, nc)
        all_scores.append(1.0 / (1.0 + np.exp(-logits)))
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)

    n = boxes.shape[0]
    k = min(spec.max_det, n)
    # Top-k anchors by best class score, then top-k (anchor, class) pairs among them.
    anchor_idx = np.argsort(-scores.max(axis=1), kind="stable")[:k]
    cand = scores[anchor_idx].reshape(-1)
    pair_idx = np.argsort(-cand, kind="stable")[:k]
    pair_scores = cand[pair_idx]
    keep = pair_scores >= spec.conf_threshold
    pair_idx = pair_idx[keep]
    pair_scores = pair_scores[keep]
    sel_anchor = anchor_idx[pair_idx // nc]
    sel_class = pair_idx % nc

    out = boxes[sel_anchor].copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - lb.pad_x) / lb.scale, 0, lb.src_w)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - lb.pad_y) / lb.scale, 0, lb.src_h)

    detections = []
    for box, cls, score in zip(out, sel_class, pair_scores, strict=True):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(
            {
                "class": spec.class_names[int(cls)],
                "class_id": int(cls),
                "confidence": float(score),
                "bbox": [float(v) for v in box],
            }
        )
    detections.sort(key=lambda det: det["confidence"], reverse=True)
    return detections


__all__ = [
    "SPEC_SCHEMA",
    "DetectorSpec",
    "Letterbox",
    "decode_yolo26",
    "letterbox",
    "letterbox_params",
    "load_detector_spec",
    "load_spec_file",
    "spec_path_for",
]
