"""``scripts/eval_detector.py``: the gate and baseline verdicts the HEF decision rests on.

``run_eval`` is exercised with a scripted backend and in-memory images, so the matching, the
per-tier recall gates, the baseline recall-drop check, the report and the exit code are real; only
the model is faked.
"""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts import eval_detector as ev
from workshop.classes import NAMES

W, H = 100, 100
SPEC = ev.DetectorSpec(
    arch="yolo26",
    input_size=640,
    class_names=tuple(NAMES),
    conf_threshold=0.25,
    max_det=100,
    tag="t",
)


def _box_row(cls: int) -> str:
    return f"{cls} 0.5 0.5 0.4 0.4\n"  # one 40x40 box centred in the 100x100 image


class ScriptedBackend:
    """Detects an image's box iff ``found(image_index, class_id)``."""

    def __init__(self, found):
        self.found = found
        self.i = -1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def predict(self, rgb):
        self.i += 1
        dets = [
            {
                "class": NAMES[c],
                "class_id": c,
                "confidence": 0.9,
                "bbox": [30.0, 30.0, 70.0, 70.0],
            }
            for c in self.classes_of[self.i]
            if self.found(self.i, c)
        ]
        return dets, 1.0


def _dataset(tmp_path: Path, classes_per_image: list[list[int]]):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i, classes in enumerate(classes_per_image):
        (images / f"{i:04d}.jpg").write_bytes(b"x")
        (labels / f"{i:04d}.txt").write_text("".join(_box_row(c) for c in classes))
    return images, labels


def _run(tmp_path, monkeypatch, classes_per_image, found, **over):
    images, labels = _dataset(tmp_path, classes_per_image)
    backend = ScriptedBackend(found)
    backend.classes_of = classes_per_image
    monkeypatch.setattr(ev, "_backend", lambda a, spec: backend)
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))
    args = Namespace(
        backend="onnx",
        model=str(tmp_path / "m.onnx"),
        images=str(images),
        labels=str(labels),
        iou=0.5,
        gate=over.pop("gate", False),
        baseline=over.pop("baseline", None),
        max_recall_drop=0.10,
        min_gt=20,
        out=str(tmp_path / "report.json"),
    )
    assert not over
    (tmp_path / "m.onnx").write_bytes(b"onnx")
    rc = asyncio.run(ev.run_eval(args, SPEC))
    return rc, json.loads((tmp_path / "report.json").read_text())


PERSON, ROCK, SHED = NAMES.index("person"), NAMES.index("rock"), NAMES.index("shed")


def test_perfect_detector_passes_the_gate(tmp_path, monkeypatch):
    rc, report = _run(tmp_path, monkeypatch, [[PERSON, ROCK]] * 10, lambda i, c: True, gate=True)

    assert rc == 0
    assert report["per_class"]["person"]["recall"] == 1.0
    assert report["gate"] == {"applied": True, "failures": []}
    assert report["model_sha256"]


def test_tier1_recall_below_98_percent_fails_the_gate(tmp_path, monkeypatch):
    # person (tier 1, gate 0.98): miss 1 of 10 -> 0.90; rock (tier 2, gate 0.85): all found.
    rc, report = _run(
        tmp_path,
        monkeypatch,
        [[PERSON, ROCK]] * 10,
        lambda i, c: not (c == PERSON and i == 0),
        gate=True,
    )

    assert rc == 1
    assert report["gate"]["failures"] == [{"class": "person", "recall": 0.9, "gate": 0.98}]


def test_gate_is_not_applied_unless_asked_and_missing_classes_are_skipped(tmp_path, monkeypatch):
    # Recall is 0 but no gate was requested, so nothing fails.
    rc, report = _run(tmp_path, monkeypatch, [[PERSON]] * 5, lambda i, c: False)

    assert rc == 0 and report["gate"] == {"applied": False, "failures": []}
    assert report["per_class"]["shed"]["recall"] is None  # no ground truth -> no verdict


def test_baseline_drop_over_ten_points_fails_only_for_classes_with_enough_gt(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "per_class": {
                    "person": {"recall": 1.0},
                    "rock": {"recall": 1.0},
                    "shed": {"recall": 1.0},
                }
            }
        )
    )
    # 30 images with person+rock, 5 with shed. person recall 25/30 = 0.833 (drop 0.167 -> fail),
    # rock 28/30 = 0.933 (drop 0.067 -> ok), shed 0/5 (drop 1.0 but gt 5 < min_gt 20 -> ignored).
    per_image = [[PERSON, ROCK]] * 30 + [[SHED]] * 5

    def found(i, c):
        if c == PERSON:
            return i >= 5
        if c == ROCK:
            return i >= 2
        return False

    rc, report = _run(tmp_path, monkeypatch, per_image, found, baseline=str(baseline))

    assert rc == 1
    assert [f["class"] for f in report["baseline"]["failures"]] == ["person"]
    fail = report["baseline"]["failures"][0]
    assert fail["baseline"] == 1.0 and round(fail["drop"], 3) == 0.167


def test_baseline_within_tolerance_passes(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"per_class": {"person": {"recall": 0.95}}}))

    rc, report = _run(
        tmp_path,
        monkeypatch,
        [[PERSON]] * 40,
        lambda i, c: i >= 3,  # 37/40 = 0.925: drop 0.025
        baseline=str(baseline),
    )

    assert rc == 0 and report["baseline"]["failures"] == []
