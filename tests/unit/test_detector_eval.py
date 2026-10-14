"""Per-class TP/FP/FN accounting behind the compiled-model recall gates."""

from __future__ import annotations

from workshop.detector_eval import load_yolo_labels, match, merge, summarize


def _p(cls: int, conf: float, box):
    return {"class_id": cls, "confidence": conf, "bbox": list(box)}


BOX = (0.0, 0.0, 10.0, 10.0)


def test_single_match():
    assert match([_p(0, 0.9, BOX)], [(0, BOX)]) == {0: {"gt": 1, "tp": 1, "fp": 0, "fn": 0}}


def test_duplicate_prediction_is_fp():
    counts = match([_p(0, 0.9, BOX), _p(0, 0.8, BOX)], [(0, BOX)])
    assert counts == {0: {"gt": 1, "tp": 1, "fp": 1, "fn": 0}}


def test_class_mismatch_is_fp_and_fn():
    counts = match([_p(1, 0.9, BOX)], [(0, BOX)])
    assert counts[0] == {"gt": 1, "tp": 0, "fp": 0, "fn": 1}
    assert counts[1] == {"gt": 0, "tp": 0, "fp": 1, "fn": 0}


def test_iou_threshold_boundary():
    # Same height; x-overlap chosen so IoU = inter / (200 - inter).
    def shifted(iou_target: float):
        inter = 200 * iou_target / (1 + iou_target)  # in units of height 10
        return (10.0 - inter / 10.0, 0.0, 20.0 - inter / 10.0, 10.0)

    hit = match([_p(0, 0.9, shifted(0.51))], [(0, BOX)], iou_thr=0.5)
    miss = match([_p(0, 0.9, shifted(0.49))], [(0, BOX)], iou_thr=0.5)
    assert hit[0]["tp"] == 1
    assert miss[0] == {"gt": 1, "tp": 0, "fp": 1, "fn": 1}


def test_image_without_ground_truth():
    assert match([_p(2, 0.5, BOX)], []) == {2: {"gt": 0, "tp": 0, "fp": 1, "fn": 0}}
    assert match([], []) == {}


def test_summarize_and_merge():
    total = merge({}, match([_p(0, 0.9, BOX)], [(0, BOX)]))
    total = merge(total, match([], [(0, BOX)]))
    total = merge(total, match([_p(1, 0.9, BOX)], []))

    s = summarize(total, ["a", "b", "c"])

    assert s["a"]["recall"] == 0.5
    assert s["a"]["precision"] == 1.0
    assert s["b"]["recall"] is None
    assert s["b"]["precision"] == 0.0
    assert s["c"] == {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "recall": None, "precision": None}


def test_load_yolo_labels(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("3 0.5 0.5 0.25 0.5\n\n", encoding="utf-8")
    assert load_yolo_labels(txt, 200, 100) == [(3, [75.0, 25.0, 125.0, 75.0])]
    assert load_yolo_labels(tmp_path / "missing.txt", 200, 100) == []
