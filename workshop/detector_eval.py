"""Per-class detection matching for compiled-model gates (pure Python, no numpy).

Predictions are detection dicts (``class_id``, ``confidence``, ``bbox`` = xyxy in
source pixels); ground truth is ``(class_id, xyxy)`` from YOLO label files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

Box = Sequence[float]
Counts = dict[int, dict[str, int]]


def load_yolo_labels(txt: str | Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    """YOLO ``cls cx cy bw bh`` (normalized) rows -> ``(cls, [x1, y1, x2, y2])`` in pixels."""
    path = Path(txt)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append(
            (cls, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        )
    return out


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bucket(counts: Counts, cls: int) -> dict[str, int]:
    return counts.setdefault(cls, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})


def match(preds: Iterable[Mapping], gts: Iterable[tuple[int, Box]], iou_thr: float = 0.5) -> Counts:
    """Greedy per-class matching by descending confidence.

    Each GT box matches at most one prediction (IoU >= ``iou_thr``); unmatched or
    duplicate predictions are FP, unmatched GT are FN.
    """
    counts: Counts = {}
    gt_by_cls: dict[int, list[Box]] = {}
    for cls, box in gts:
        gt_by_cls.setdefault(int(cls), []).append(box)
        _bucket(counts, int(cls))["gt"] += 1
    used: dict[int, set[int]] = {}
    for p in sorted(preds, key=lambda d: d["confidence"], reverse=True):
        cls = int(p["class_id"])
        cands = gt_by_cls.get(cls, [])
        taken = used.setdefault(cls, set())
        best_i, best_iou = -1, iou_thr
        for i, g in enumerate(cands):
            if i in taken:
                continue
            v = iou(p["bbox"], g)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            taken.add(best_i)
            _bucket(counts, cls)["tp"] += 1
        else:
            _bucket(counts, cls)["fp"] += 1
    for cls, boxes in gt_by_cls.items():
        _bucket(counts, cls)["fn"] += len(boxes) - len(used.get(cls, ()))
    return counts


def merge(total: Counts, counts: Counts) -> Counts:
    for cls, c in counts.items():
        b = _bucket(total, cls)
        for k in ("gt", "tp", "fp", "fn"):
            b[k] += c[k]
    return total


def summarize(counts: Counts, names: Sequence[str]) -> dict[str, dict]:
    out = {}
    for cls, name in enumerate(names):
        c = counts.get(cls, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out[name] = {
            **c,
            "recall": tp / (tp + fn) if c["gt"] > 0 else None,
            "precision": tp / (tp + fp) if tp + fp > 0 else None,
        }
    return out
