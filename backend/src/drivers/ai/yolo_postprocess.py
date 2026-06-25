"""Pure-numpy YOLOv8 detection post-processing for the Hailo 8L pipeline.

The Hailo ``yolov8m`` HEF emits a single output tensor (``output0``) shaped
``(1, 84, 8400)``:

* ``84`` attributes per anchor = 4 box coordinates ``(cx, cy, w, h)`` followed
  by ``80`` COCO class scores (already activated to ``[0, 1]`` by the model
  head — no extra sigmoid is applied here).
* ``8400`` anchors across the three detection strides for a ``640x640`` input.

:func:`decode_yolov8` turns that raw tensor into a list of detection dicts,
applying confidence filtering and class-aware Non-Max Suppression with nothing
but numpy (no torch, no OpenCV) so it stays importable and testable off-device.
"""

from __future__ import annotations

import numpy as np

# Standard 80-class COCO label set, ordered to match the yolov8m class indices.
COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _to_anchor_major(output: np.ndarray) -> np.ndarray:
    """Normalize a YOLOv8 output tensor to ``(num_anchors, num_attrs)``.

    Accepts ``(1, A, N)`` / ``(A, N)`` (attribute-major, as Hailo emits) as well
    as ``(1, N, A)`` / ``(N, A)`` (anchor-major) tensors.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D YOLOv8 tensor, got shape {np.asarray(output).shape}")

    # The attribute axis (4 + num_classes) is far smaller than the anchor axis.
    # Hailo emits attribute-major (84, 8400); transpose it to anchor-major.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.ascontiguousarray(arr, dtype=np.float32)


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Class-aware greedy Non-Max Suppression (pure numpy).

    Args:
        boxes: ``(N, 4)`` array of ``(x1, y1, x2, y2)`` boxes.
        scores: ``(N,)`` confidence scores.
        class_ids: ``(N,)`` integer class ids; boxes of different classes never
            suppress one another.
        iou_threshold: boxes overlapping above this IoU are suppressed.

    Returns:
        Indices into the input arrays to keep, ordered by descending score.
    """
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        inter_w = np.clip(xx2 - xx1, 0.0, None)
        inter_h = np.clip(yy2 - yy1, 0.0, None)
        inter = inter_w * inter_h
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        same_class = class_ids[rest] == class_ids[current]
        suppress = same_class & (iou > iou_threshold)
        order = rest[~suppress]

    return keep


def decode_yolov8(
    output: np.ndarray,
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Decode a raw YOLOv8 output tensor into detection dicts.

    Args:
        output: Raw model output, typically shaped ``(1, 84, 8400)``.
        conf_threshold: Minimum per-anchor class confidence to keep a detection.
        iou_threshold: IoU above which overlapping same-class boxes are merged.
        class_names: Optional class label lookup (defaults to ``class_<id>``).

    Returns:
        A list of ``{"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}``
        detections, ordered by descending confidence.
    """
    preds = _to_anchor_major(output)
    num_attrs = preds.shape[1]
    num_classes = num_attrs - 4
    if num_classes <= 0:
        raise ValueError(f"Expected >4 attributes (box + classes), got {num_attrs}")

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = confidences >= conf_threshold
    if not np.any(keep_mask):
        return []

    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh[keep_mask])
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]

    kept = nms(boxes_xyxy, confidences, class_ids, iou_threshold)

    detections: list[dict] = []
    for idx in kept:
        cid = int(class_ids[idx])
        if class_names is not None and 0 <= cid < len(class_names):
            label = class_names[cid]
        else:
            label = f"class_{cid}"
        detections.append(
            {
                "class": label,
                "confidence": float(confidences[idx]),
                "bbox": [float(coord) for coord in boxes_xyxy[idx]],
            }
        )
    return detections


__all__ = ["COCO_CLASSES", "decode_yolov8", "nms"]
