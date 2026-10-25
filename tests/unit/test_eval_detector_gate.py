"""``scripts/eval_detector.py``: the gate and baseline verdicts the HEF decision rests on.

``main`` is exercised with a scripted backend and in-memory images, so the argument checks, the
label and baseline validation, the matching, the per-tier recall gates, the baseline recall-drop
check, the report and the exit code are real; only the model is faked.

The gate decides whether a compiled detector may replace the live one, so every way it used to
exit 0 without having measured anything is pinned here (audit review TOOL-01, bypasses V1-V5):
a labels directory with no files for the images, a tier-1 class decided on one sample,
``--iou 0``, a spec that is not the class set, a baseline that is not a report. Exit codes:
0 passed, 1 the model failed, 2 usage/setup, 3 the images and labels cannot support a verdict,
4 the baseline cannot be compared.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import eval_detector as ev
from workshop.classes import NAMES
from workshop.detector_eval import GATE_MIN_GT, MIN_IMAGES

W, H = 100, 100
SPEC = ev.DetectorSpec(
    arch="yolo26",
    input_size=640,
    class_names=tuple(NAMES),
    conf_threshold=0.25,
    max_det=100,
    tag="t",
)
PERSON, PET, WILDLIFE = NAMES.index("person"), NAMES.index("pet"), NAMES.index("wildlife")
ROCK, SHED, HOSE = NAMES.index("rock"), NAMES.index("shed"), NAMES.index("hose")
TIER1 = [PERSON, PET, WILDLIFE]
GOOD = (30.0, 30.0, 70.0, 70.0)  # the box every label draws (40x40, centred in 100x100)


class ScriptedBackend:
    """Detects an image's box iff ``found(image_index, class_id)``, at ``bbox``."""

    def __init__(self, found, classes_of, bbox=GOOD):
        self.found, self.classes_of, self.bbox, self.i = found, classes_of, list(bbox), -1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def predict(self, rgb):
        self.i += 1
        dets = [
            {"class": NAMES[c], "class_id": c, "confidence": 0.9, "bbox": self.bbox}
            for c in self.classes_of[self.i]
            if self.found(self.i, c)
        ]
        return dets, 1.0


def _box_row(cls: int) -> str:
    return f"{cls} 0.5 0.5 0.4 0.4\n"


def _dataset(root: Path, classes_per_image: list[list[int]], *, write_labels: bool = True):
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for i, classes in enumerate(classes_per_image):
        (images / f"{i:04d}.jpg").write_bytes(b"x")
        if write_labels:
            (labels / f"{i:04d}.txt").write_text("".join(_box_row(c) for c in classes))
    return images, labels


def run_cli(
    tmp_path,
    monkeypatch,
    classes_per_image,
    found=lambda i, c: True,
    *flags,
    name="run",
    spec=SPEC,
    bbox=GOOD,
    write_labels=True,
    backend="onnx",
):
    """``eval_detector.main`` on a crafted dataset -> ``(exit code, report or None)``."""
    root = tmp_path / name
    images, labels = _dataset(root, classes_per_image, write_labels=write_labels)
    (root / "m.onnx").write_bytes(b"onnx")
    scripted = ScriptedBackend(found, classes_per_image, bbox)
    monkeypatch.setattr(ev, "_backend", lambda a, s: scripted)
    monkeypatch.setattr(ev, "_load_spec", lambda a: spec)
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))
    out = root / "report.json"
    argv = [
        *("--backend", backend),
        *("--model", str(root / "m.onnx")),
        *("--images", str(images)),
        *("--labels", str(labels)),
        *("--out", str(out)),
        *flags,
    ]
    rc = ev.main(argv)
    return rc, (json.loads(out.read_text()) if out.exists() else None)


def covering(n: int, extra: list[list[int]] | None = None) -> list[list[int]]:
    """``n`` images, each with one box of every tier-1 class and a rock (plus ``extra``)."""
    return [[*TIER1, ROCK]] * n + (extra or [])


SMALL = ("--gate-min-gt", "10", "--min-images", "10")  # a small but sufficient crafted dataset


# ---- the legitimate cases keep working -------------------------------------------------------


def test_perfect_detector_passes_the_gate(tmp_path, monkeypatch):
    rc, report = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL)

    assert rc == 0
    assert report["per_class"]["person"]["recall"] == 1.0
    assert report["verdict"] == "passed" and report["status"] == "ok"
    gate = report["gate"]
    assert gate["applied"] is True and gate["status"] == "passed" and gate["failures"] == []
    assert gate["min_gt"] == 10 and gate["tier1_gt"] == {"person": 12, "pet": 12, "wildlife": 12}
    assert report["model_sha256"] and report["images_dir"].endswith("images")
    assert report["labels_dir"].endswith("labels") and report["label_files"] == 12


def test_tier1_recall_below_98_percent_fails_the_gate(tmp_path, monkeypatch):
    # person (tier 1, gate 0.98): miss 1 of 12 -> 0.917; rock (tier 2, gate 0.85): all found.
    rc, report = run_cli(
        tmp_path,
        monkeypatch,
        covering(12),
        lambda i, c: not (c == PERSON and i == 0),
        "--gate",
        *SMALL,
    )

    assert rc == 1
    assert report["verdict"] == "failed" and report["gate"]["status"] == "failed"
    [failure] = report["gate"]["failures"]
    assert failure["class"] == "person" and failure["gate"] == 0.98
    assert failure["recall"] == pytest.approx(11 / 12)


def test_gate_is_not_applied_unless_asked_and_classes_without_gt_are_not_a_verdict(
    tmp_path, monkeypatch
):
    # Recall is 0 but no gate was requested: a measurement, exit 0, and the report says so.
    rc, report = run_cli(tmp_path, monkeypatch, [[PERSON]] * 5, lambda i, c: False)

    assert rc == 0 and report["verdict"] == "not_applied"
    assert report["gate"] == {"applied": False, "status": "not_applied", "failures": []}
    assert report["baseline"]["applied"] is False
    assert report["per_class"]["shed"]["recall"] is None  # no ground truth -> no verdict


def test_classes_without_ground_truth_are_listed_not_silently_skipped(tmp_path, monkeypatch):
    rc, report = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL)

    assert rc == 0
    unevaluated = {c["class"]: c for c in report["gate"]["not_evaluated"]}
    assert unevaluated["shed"] == {"class": "shed", "tier": 3, "reason": "no ground truth"}
    assert "person" not in unevaluated and "rock" not in unevaluated


def _baseline_from(tmp_path, monkeypatch, classes_per_image, found, name="base", **edit):
    """A baseline made the way the workflow makes it: this script's own report on the same split."""
    rc, report = run_cli(tmp_path, monkeypatch, classes_per_image, found, name=name)
    assert rc == 0
    report = {**report, **edit}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report))
    return path, report


def test_baseline_drop_over_ten_points_fails_only_for_classes_with_enough_gt(tmp_path, monkeypatch):
    # 30 images with person+rock, 5 with shed. Baseline finds everything. This run: person recall
    # 25/30 = 0.833 (drop 0.167 -> fail), rock 28/30 = 0.933 (drop 0.067 -> ok), shed 0/5
    # (drop 1.0, but gt 5 < min_gt 20 -> listed as skipped, not compared).
    per_image = [[PERSON, ROCK]] * 30 + [[SHED]] * 5
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True)

    def found(i, c):
        if c == PERSON:
            return i >= 5
        if c == ROCK:
            return i >= 2
        return False

    rc, report = run_cli(
        tmp_path, monkeypatch, per_image, found, "--baseline", str(base), "--min-images", "10"
    )

    assert rc == 1 and report["verdict"] == "failed"
    block = report["baseline"]
    assert block["status"] == "failed" and block["compared"] == ["person", "rock"]
    assert [f["class"] for f in block["failures"]] == ["person"]
    assert block["failures"][0]["baseline"] == 1.0
    assert round(block["failures"][0]["drop"], 3) == 0.167
    assert {"class": "shed", "gt": 5, "reason": "fewer than 20 ground-truth boxes"} in block[
        "skipped"
    ]
    assert block["min_gt"] == 20 and block["max_recall_drop"] == 0.10


def test_baseline_within_tolerance_passes(tmp_path, monkeypatch):
    per_image = [[PERSON]] * 40
    # baseline finds 38 of 40 (0.95); this run 37 of 40 (0.925): drop 0.025
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: i >= 2)

    rc, report = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: i >= 3, "--baseline", str(base),
        "--min-images", "10",
    )  # fmt: skip

    assert rc == 0 and report["verdict"] == "passed"
    assert report["baseline"]["status"] == "passed" and report["baseline"]["failures"] == []
    assert report["baseline"]["compared"] == ["person"]


def test_gate_and_baseline_together_both_have_to_pass(tmp_path, monkeypatch):
    per_image = covering(12)
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True)

    rc, report = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--gate", "--baseline", str(base),
        "--gate-min-gt", "10", "--min-images", "10", "--min-gt", "10",
    )  # fmt: skip

    assert rc == 0 and report["verdict"] == "passed"
    assert report["baseline"]["compared"] == ["person", "pet", "wildlife", "rock"]


# ---- TOOL-01: the bypasses the review reproduced (each one used to exit 0) --------------------


def test_V1_a_labels_directory_without_the_files_is_not_a_pass(tmp_path, monkeypatch, capsys):
    # The detector finds NOTHING and the --labels directory holds no files for these images
    # (wrong directory or split): the old loader read "no label file" as "no objects", every class
    # had gt 0, the gate had nothing to fail and exited 0.
    rc, report = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 20, lambda i, c: False, "--gate", write_labels=False
    )

    assert rc == 3
    err = capsys.readouterr().err
    assert "INVALID, no verdict" in err and "20 of 20 images have no label file" in err
    assert report["verdict"] == "invalid" and report["error"] == "dataset"
    assert report["gate"]["status"] == "invalid" and "per_class" not in report


def test_V1_even_without_a_gate_a_wrong_labels_directory_is_refused(tmp_path, monkeypatch):
    # A report from the wrong directory would later be taken as a baseline.
    rc, report = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 5, lambda i, c: True, write_labels=False
    )

    assert rc == 3 and report["verdict"] == "invalid"


def test_a_labels_directory_that_does_not_exist_is_refused(tmp_path, monkeypatch, capsys):
    root = tmp_path / "run"
    images, _ = _dataset(root, [[PERSON]] * 3)
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)

    rc = ev.main(
        ["--backend", "onnx", "--model", "m", "--images", str(images)]
        + ["--labels", str(root / "nowhere"), "--out", str(root / "r.json"), "--gate"]
    )

    assert rc == 3 and "labels directory not found" in capsys.readouterr().err


def test_a_single_missing_label_file_is_not_read_as_an_empty_one(tmp_path, monkeypatch):
    per_image = covering(12)
    root = tmp_path / "run"
    images, labels = _dataset(root, per_image)
    (labels / "0005.txt").unlink()
    (root / "m.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    monkeypatch.setattr(ev, "_backend", lambda a, s: ScriptedBackend(lambda i, c: True, per_image))
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))

    rc = ev.main(
        ["--backend", "onnx", "--model", str(root / "m.onnx"), "--images", str(images)]
        + ["--labels", str(labels), "--out", str(root / "r.json"), "--gate", *SMALL]
    )

    report = json.loads((root / "r.json").read_text())
    assert rc == 3 and "1 of 12 images have no label file" in report["problems"][0]
    assert "0005.jpg" in report["problems"][0]


def test_V2_a_tier1_class_decided_on_one_sample_is_refused(tmp_path, monkeypatch, capsys):
    # person has ONE ground-truth box (found); 119 other images hold rocks. The 98 % gate was
    # evaluated on a single sample and passed. pet and wildlife had no ground truth at all.
    per_image = [[PERSON]] + [[ROCK]] * (MIN_IMAGES + 19)

    rc, report = run_cli(tmp_path, monkeypatch, per_image, lambda i, c: True, "--gate")

    assert rc == 3
    [problem] = report["problems"]
    assert f"at least {GATE_MIN_GT} ground-truth boxes" in problem
    assert "person 1" in problem and "pet 0" in problem and "wildlife 0" in problem
    assert report["verdict"] == "invalid"
    assert "person 1" in capsys.readouterr().err


def test_the_default_thresholds_accept_exactly_enough_evidence_and_refuse_one_box_less(
    tmp_path, monkeypatch
):
    # 150 boxes per tier-1 class and 150 images: the smallest set the defaults accept.
    enough, report = run_cli(
        tmp_path, monkeypatch, covering(GATE_MIN_GT), lambda i, c: True, "--gate", name="enough"
    )
    short, refused = run_cli(
        tmp_path, monkeypatch, covering(GATE_MIN_GT - 1), lambda i, c: True, "--gate", name="short"
    )

    assert enough == 0 and report["gate"]["min_gt"] == GATE_MIN_GT == 150
    assert short == 3 and f"at least {GATE_MIN_GT} ground-truth boxes" in refused["problems"][0]
    assert "person 149" in refused["problems"][0]


def test_the_default_image_floor_is_exactly_min_images(tmp_path, monkeypatch):
    # 100 images, each with 2 boxes per tier-1 class: the tier-1 minimum is met either way.
    two = [[*TIER1, *TIER1]]
    at_floor, _ = run_cli(
        tmp_path, monkeypatch, two * MIN_IMAGES, lambda i, c: True, "--gate", "--gate-min-gt",
        "200", name="at",
    )  # fmt: skip
    below, report = run_cli(
        tmp_path, monkeypatch, two * (MIN_IMAGES - 1), lambda i, c: True, "--gate",
        "--gate-min-gt", "198", name="below",
    )  # fmt: skip

    assert at_floor == 0 and below == 3
    assert f"{MIN_IMAGES - 1} images; a verdict needs at least {MIN_IMAGES}" in report["problems"]


def test_the_tier1_minimum_is_configurable_and_reported(tmp_path, monkeypatch):
    per_image = covering(30)

    refused, report = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--gate", "--min-images", "10",
        "--gate-min-gt", "31", name="a",
    )  # fmt: skip
    accepted, ok = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--gate", "--min-images", "10",
        "--gate-min-gt", "30", name="b",
    )  # fmt: skip

    assert refused == 3 and "at least 31 ground-truth boxes" in report["problems"][0]
    assert accepted == 0 and ok["gate"]["min_gt"] == 30


def test_V3_iou_zero_cannot_pass_boxes_in_the_wrong_place(tmp_path, monkeypatch, capsys):
    # Every detection has the right class but sits in a corner (IoU 0 with the label). With
    # --iou 0.0 every one of them "matched": recall 1.0 for a detector that locates nothing.
    wrong = (0.0, 0.0, 5.0, 5.0)
    control, _ = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL,
        bbox=wrong, name="control",
    )  # fmt: skip
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL, "--iou", "0.0",
        bbox=wrong,
    )  # fmt: skip

    assert control == 1  # at the default IoU the same detector fails the gate
    assert rc == 2 and report is None
    assert "--iou must be in (0, 1], got 0.0" in capsys.readouterr().err


@pytest.mark.parametrize("iou", ["0", "-0.5", "1.5", "2", "nan", "inf", "-inf"])
def test_iou_outside_zero_to_one_is_a_usage_error(tmp_path, monkeypatch, capsys, iou):
    # `--iou=-inf`: argparse would read a separate "-inf" as an option name
    rc, _ = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, f"--iou={iou}")

    assert rc == 2 and "--iou must be in (0, 1]" in capsys.readouterr().err


def test_iou_of_exactly_one_is_allowed(tmp_path, monkeypatch):
    rc, report = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, "--iou", "1")

    assert rc == 0 and report["iou"] == 1.0


@pytest.mark.parametrize(
    "flags, message",
    [
        (("--min-iou", "0"), "--min-iou"),
        (("--conf", "0"), "--conf"),
        (("--conf", "1"), "--conf"),
        (("--conf", "nan"), "--conf"),
        (("--max-recall-drop", "1"), "--max-recall-drop"),
        (("--max-recall-drop", "nan"), "--max-recall-drop"),
        (("--max-recall-drop", "-0.1"), "--max-recall-drop"),
        (("--min-gt", "0"), "--min-gt"),
        (("--gate-min-gt", "0"), "--gate-min-gt"),
        (("--min-images", "0"), "--min-images"),
    ],
)
def test_thresholds_that_would_disable_a_check_are_usage_errors(
    tmp_path, monkeypatch, capsys, flags, message
):
    rc, _ = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, *flags)

    assert rc == 2 and message in capsys.readouterr().err


def test_V4_a_spec_that_is_not_the_class_set_cannot_hide_classes(tmp_path, monkeypatch, capsys):
    # A stale 10-name spec against labels that use dropoff (id 22, tier 4): the class vanished
    # from the report and --gate had nothing to check.
    short = dataclasses.replace(SPEC, class_names=tuple(NAMES[:10]))
    dropoff = NAMES.index("dropoff")

    rc, report = run_cli(
        tmp_path, monkeypatch, [[dropoff]] * 20, lambda i, c: False, "--gate", spec=short
    )

    assert rc == 3
    assert "class id 22 but the spec has 10 classes" in capsys.readouterr().err
    assert report["verdict"] == "invalid"


def test_the_gate_needs_the_real_class_set_in_order(tmp_path, monkeypatch):
    # Same number of classes, other order: ids would be credited to the wrong tiers.
    swapped = dataclasses.replace(SPEC, class_names=tuple(reversed(NAMES)))

    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL, spec=swapped
    )

    assert rc == 3 and "workshop.classes.NAMES" in " ".join(report["problems"])


def test_V5_a_baseline_that_is_not_a_detector_eval_report_is_refused(tmp_path, monkeypatch, capsys):
    # A train manifest (no 'per_class') used as --baseline: nothing was compared, exit 0.
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"schema": "train-v1", "per_class_val": {"person": {}}}))

    rc, report = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: False, "--baseline", str(manifest),
        "--min-images", "10",
    )  # fmt: skip

    assert rc == 4
    assert "not a detector-eval report (schema 'train-v1'" in capsys.readouterr().err
    assert report["verdict"] == "invalid" and report["error"] == "baseline"
    assert report["baseline"]["status"] == "invalid"


def test_a_baseline_without_a_per_class_table_is_refused(tmp_path, monkeypatch, capsys):
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"schema": "detector-eval/1", "images": 40}))

    rc, _ = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10",
    )  # fmt: skip

    assert rc == 4 and "no 'per_class' table" in capsys.readouterr().err


def test_a_baseline_with_other_classes_is_refused(tmp_path, monkeypatch, capsys):
    # The hand-made three-class baseline the old tests used: not what this script writes.
    base = tmp_path / "b.json"
    base.write_text(
        json.dumps({"schema": "detector-eval/1", "per_class": {"person": {"recall": 1.0}}})
    )

    rc, _ = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10",
    )  # fmt: skip

    assert rc == 4 and "other classes" in capsys.readouterr().err


def test_an_unparsable_baseline_is_refused_with_its_own_exit_code(tmp_path, monkeypatch, capsys):
    base = tmp_path / "b.json"
    base.write_text("{this is not json")

    rc, report = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10",
    )  # fmt: skip

    assert rc == 4 and "cannot read" in capsys.readouterr().err
    assert report["verdict"] == "invalid"


def test_a_missing_baseline_file_is_refused(tmp_path, monkeypatch, capsys):
    rc, _ = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True, "--baseline",
        str(tmp_path / "nothing.json"), "--min-images", "10",
    )  # fmt: skip

    assert rc == 4 and "cannot read" in capsys.readouterr().err


def test_a_nan_recall_in_the_baseline_is_refused_not_compared(tmp_path, monkeypatch, capsys):
    # NaN - recall is NaN and "NaN > max_drop" is False: the class passed whatever the model did.
    per_image = [[PERSON]] * 40
    base, report = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True)
    base.write_text(json.dumps(report).replace('"recall": 1.0', '"recall": NaN', 1))

    rc, _ = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: False, "--baseline", str(base),
        "--min-images", "10", name="second",
    )  # fmt: skip

    assert rc == 4 and "non-finite number literal NaN" in capsys.readouterr().err


@pytest.mark.parametrize(
    "edit, message",
    [
        ({"iou": 0.3}, "measured at iou 0.3"),
        ({"conf": 0.01}, "measured at conf 0.01"),
        ({"images": 39}, "covers 39 images"),
        ({"status": "invalid"}, "not a valid evaluation"),
        ({"schema": "detector-eval/9"}, "not a detector-eval report"),
    ],
)
def test_a_baseline_of_another_evaluation_is_refused(tmp_path, monkeypatch, capsys, edit, message):
    per_image = [[PERSON]] * 40
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True, **edit)

    rc, _ = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: False, "--baseline", str(base),
        "--min-images", "10", name="second",
    )  # fmt: skip

    assert rc == 4 and message in capsys.readouterr().err


def test_a_baseline_measured_on_other_ground_truth_is_refused(tmp_path, monkeypatch, capsys):
    base, _ = _baseline_from(tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True)

    rc, _ = run_cli(
        tmp_path, monkeypatch, [[PERSON, PERSON]] * 40, lambda i, c: False, "--baseline", str(base),
        "--min-images", "10", name="second",
    )  # fmt: skip

    assert rc == 4 and "other ground truth" in capsys.readouterr().err


def test_a_baseline_comparison_with_no_comparable_class_is_refused(tmp_path, monkeypatch, capsys):
    # 12 boxes per class, --min-gt 20: every class would be skipped and the check pass empty.
    per_image = [[PERSON]] * 12
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True)

    rc, report = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: False, "--baseline", str(base),
        "--min-images", "10", name="second",
    )  # fmt: skip

    assert rc == 4 and "nothing to compare" in capsys.readouterr().err
    assert report["verdict"] == "invalid"


def test_a_report_of_an_invalid_run_cannot_serve_as_a_baseline(tmp_path, monkeypatch, capsys):
    rc, invalid = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True, write_labels=False, name="bad"
    )
    assert rc == 3
    base = tmp_path / "bad.json"
    base.write_text(json.dumps(invalid))

    rc, _ = run_cli(
        tmp_path, monkeypatch, [[PERSON]] * 40, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10", name="second",
    )  # fmt: skip

    assert rc == 4 and "not a valid evaluation" in capsys.readouterr().err


def test_a_tiny_dataset_cannot_decide_a_gate_or_a_baseline(tmp_path, monkeypatch):
    refused, report = run_cli(
        tmp_path, monkeypatch, covering(MIN_IMAGES - 1), lambda i, c: True, "--gate",
        "--gate-min-gt", "10", name="tiny",
    )  # fmt: skip
    plain, _ = run_cli(
        tmp_path, monkeypatch, covering(MIN_IMAGES - 1), lambda i, c: True, name="plain"
    )

    assert (
        refused == 3
        and f"{MIN_IMAGES - 1} images; a verdict needs at least {MIN_IMAGES}"
        in " ".join(report["problems"])
    )
    assert plain == 0  # measuring a small set is fine; only a verdict needs evidence


def test_a_baseline_check_alone_also_needs_the_image_floor(tmp_path, monkeypatch):
    # No --gate: the same 99 images are no better a basis for a recall-drop verdict.
    per_image = covering(MIN_IMAGES - 1)
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True)

    rc, report = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--baseline", str(base),
        name="second",
    )  # fmt: skip

    assert rc == 3
    assert f"{MIN_IMAGES - 1} images; a verdict needs at least {MIN_IMAGES}" in report["problems"]


def test_the_baseline_is_compared_at_this_runs_confidence_not_a_constant(tmp_path, monkeypatch):
    # The spec's threshold is 0.4 here; a baseline measured at 0.25 is another operating point.
    at_04 = dataclasses.replace(SPEC, conf_threshold=0.4)
    per_image = [[PERSON]] * 40
    rc, base_report = run_cli(tmp_path, monkeypatch, per_image, spec=at_04, name="base")
    assert rc == 0 and base_report["conf"] == 0.4
    base = tmp_path / "base.json"
    base.write_text(json.dumps(base_report))

    same, _ = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10", spec=at_04, name="same",
    )  # fmt: skip
    other, _ = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10", name="other",
    )  # fmt: skip

    assert (same, other) == (0, 4)


def test_an_empty_images_directory_is_refused(tmp_path, monkeypatch, capsys):
    rc, report = run_cli(tmp_path, monkeypatch, [], lambda i, c: True, "--gate")

    assert rc == 3 and "no images in" in capsys.readouterr().err
    assert report["verdict"] == "invalid"


def test_a_missing_images_directory_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)

    rc = ev.main(
        ["--backend", "onnx", "--model", "m", "--images", str(tmp_path / "nope")]
        + ["--labels", str(tmp_path), "--out", str(tmp_path / "r.json"), "--gate"]
    )

    assert rc == 3 and "images directory not found" in capsys.readouterr().err


def test_labels_without_an_image_are_a_mismatch(tmp_path, monkeypatch):
    # The labels directory holds the whole split; --images was pointed at a partial copy.
    per_image = covering(12)
    root = tmp_path / "run"
    images, labels = _dataset(root, per_image)
    for i in range(6, 12):
        (images / f"{i:04d}.jpg").unlink()
    (root / "m.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    monkeypatch.setattr(ev, "_backend", lambda a, s: ScriptedBackend(lambda i, c: True, per_image))
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))

    rc = ev.main(
        ["--backend", "onnx", "--model", str(root / "m.onnx"), "--images", str(images)]
        + ["--labels", str(labels), "--out", str(root / "r.json"), "--gate", *SMALL]
    )

    report = json.loads((root / "r.json").read_text())
    assert rc == 3 and "6 label files have no image" in report["problems"][0]


def test_a_malformed_label_row_is_an_error_not_a_skipped_box(tmp_path, monkeypatch, capsys):
    per_image = covering(12)
    root = tmp_path / "run"
    images, labels = _dataset(root, per_image)
    (labels / "0003.txt").write_text("0 0.5 0.5 0.4\n")  # four fields
    (root / "m.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    monkeypatch.setattr(ev, "_backend", lambda a, s: ScriptedBackend(lambda i, c: True, per_image))
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))

    rc = ev.main(
        ["--backend", "onnx", "--model", str(root / "m.onnx"), "--images", str(images)]
        + ["--labels", str(labels), "--out", str(root / "r.json")]
    )

    assert (
        rc == 3 and "0003.txt:1: expected 'cls cx cy w h', got 4 fields" in capsys.readouterr().err
    )


def test_an_invalid_run_replaces_a_stale_passing_report(tmp_path, monkeypatch):
    rc, first = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL)
    assert rc == 0 and first["verdict"] == "passed"
    root = tmp_path / "run"
    for label in (root / "labels").glob("*.txt"):
        label.unlink()

    rc = ev.main(
        ["--backend", "onnx", "--model", str(root / "m.onnx"), "--images", str(root / "images")]
        + ["--labels", str(root / "labels"), "--out", str(root / "report.json"), "--gate", *SMALL]
    )

    assert rc == 3
    assert json.loads((root / "report.json").read_text())["verdict"] == "invalid"


def test_a_crash_of_the_backend_is_a_setup_error_never_a_failed_gate(tmp_path, monkeypatch, capsys):
    # An uncaught exception exits 1, which is what "the model failed its gate" exits with.
    class Crashing(ScriptedBackend):
        async def predict(self, rgb):
            raise RuntimeError("the Hailo device vanished")

    per_image = covering(12)
    root = tmp_path / "run"
    images, labels = _dataset(root, per_image)
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    monkeypatch.setattr(ev, "_backend", lambda a, s: Crashing(lambda i, c: True, per_image))
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))
    (root / "m.onnx").write_bytes(b"onnx")

    rc = ev.main(
        ["--backend", "onnx", "--model", str(root / "m.onnx"), "--images", str(images)]
        + ["--labels", str(labels), "--out", str(root / "r.json"), "--gate", *SMALL]
    )

    err = capsys.readouterr().err
    assert rc == 2 and "unexpected RuntimeError: the Hailo device vanished" in err
    assert "Traceback" in err  # the cause is not hidden
    assert not (root / "r.json").exists()  # no report: nothing was measured


def test_exit_codes_tell_a_failing_model_from_an_unusable_evaluation(tmp_path, monkeypatch):
    failed, _ = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: False, "--gate", *SMALL, name="a"
    )
    no_labels, _ = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: False, "--gate", *SMALL, name="b",
        write_labels=False,
    )  # fmt: skip
    bad_iou, _ = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: False, "--iou", "0", name="c"
    )

    assert (failed, no_labels, bad_iou) == (1, 3, 2)


# ---- modes and arguments that used to be silently ignored -------------------------------------


def test_gate_and_baseline_are_refused_in_predict_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    for flag in (["--gate"], ["--baseline", str(tmp_path / "b.json")]):
        rc = ev.main(["--backend", "onnx", "--model", "m", "--predict", "x.jpg", *flag])

        assert rc == 2
        assert "apply to --images" in capsys.readouterr().err


def test_expect_is_refused_in_images_mode(tmp_path, monkeypatch, capsys):
    rc, _ = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, "--expect", "ref.json")

    assert rc == 2 and "--expect applies to --predict" in capsys.readouterr().err


def _predict(tmp_path, monkeypatch, reference, detections=None):
    """``--predict x.jpg --expect ref``; the detector finds ``detections`` (default: the box)."""
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))
    found = [{"class": "person", "class_id": PERSON, "confidence": 0.9, "bbox": list(GOOD)}]

    class OneImage:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def predict(self, rgb):
            return (found if detections is None else detections), 1.0

    monkeypatch.setattr(ev, "_backend", lambda a, s: OneImage())
    ref = tmp_path / "ref.json"
    ref.write_text(reference if isinstance(reference, str) else json.dumps(reference))
    return ev.main(
        ["--backend", "onnx", "--model", "m", "--predict", "x.jpg", "--expect", str(ref)]
    )


def test_expect_with_reference_boxes_still_checks_them(tmp_path, monkeypatch):
    ref = [{"class": "person", "confidence": 0.9, "bbox": list(GOOD)}]
    far = [{"class": "person", "confidence": 0.9, "bbox": [0.0, 0.0, 5.0, 5.0]}]

    assert _predict(tmp_path, monkeypatch, ref) == 0
    assert _predict(tmp_path, monkeypatch, far) == 1
    assert _predict(tmp_path, monkeypatch, ref, detections=[]) == 1  # nothing found: missing


@pytest.mark.parametrize(
    "reference",
    ["[]", "{}", "not json", '[{"class": "person"}]', '[{"class": "person", "bbox": [1, 2, 3]}]',
     '[{"class": "person", "bbox": [1, 2, 3, NaN]}]'],
)  # fmt: skip
def test_an_empty_or_malformed_expect_file_is_not_a_pass(tmp_path, monkeypatch, capsys, reference):
    # `--expect ref.json` with no boxes in it matched 0 of 0 and exited 0.
    rc = _predict(tmp_path, monkeypatch, reference)

    assert rc == 3
    assert "INVALID, no verdict" in capsys.readouterr().err


# ---- TOOL-06: the report says which DFC context produced the numbers --------------------------


def test_the_report_records_the_har_context_and_that_it_is_emulation(tmp_path, monkeypatch):
    rc, quantized = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, backend="har", name="q"
    )
    rc2, exact = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--har-context", "bit-exact",
        backend="har", name="b",
    )  # fmt: skip
    rc3, onnx = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, name="o")

    assert (rc, rc2, rc3) == (0, 0, 0)
    assert quantized["kind"] == exact["kind"] == "dfc-emulation"
    assert (quantized["har_context"], exact["har_context"]) == ("quantized", "bit-exact")
    assert onnx["har_context"] is None and onnx["kind"] == "float-cpu"


def test_an_invalid_report_also_names_what_was_to_be_evaluated(tmp_path, monkeypatch):
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", "--har-context",
        "bit-exact", backend="har", write_labels=False,
    )  # fmt: skip

    assert rc == 3
    assert report["har_context"] == "bit-exact" and report["backend"] == "har"
    assert report["model_sha256"] and report["conf"] == 0.25 and report["iou"] == 0.5


def test_the_baseline_block_names_the_report_it_was_compared_with(tmp_path, monkeypatch):
    per_image = [[PERSON]] * 40
    base, _ = _baseline_from(tmp_path, monkeypatch, per_image, lambda i, c: True)

    _, report = run_cli(
        tmp_path, monkeypatch, per_image, lambda i, c: True, "--baseline", str(base),
        "--min-images", "10", name="second",
    )  # fmt: skip

    assert report["baseline"]["path"] == str(base)
    assert copy.deepcopy(report["baseline"]["failures"]) == []


# ---- TOOL3-09: the gate says which confidence it was measured at -------------------------------

DEPLOYED = 0.25  # SPEC's own threshold: what the live detector reports at


def test_without_conf_the_report_says_it_was_measured_at_the_deployed_confidence(
    tmp_path, monkeypatch
):
    rc, report = run_cli(tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL)

    assert rc == 0
    assert (report["conf"], report["spec_conf"], report["conf_overridden"]) == (
        DEPLOYED,
        DEPLOYED,
        False,
    )
    assert report["gate"]["conf_below_deployed"] is False


def test_a_gate_at_a_confidence_below_the_deployed_one_is_refused_and_writes_no_verdict(
    tmp_path, monkeypatch, capsys
):
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL, "--conf", "0.05"
    )

    assert rc == 2, "a passing gate at 0.05 says nothing about a detector that reports at 0.25"
    assert report is None, "no report that could be mistaken for a verdict"
    err = capsys.readouterr().err
    assert "--conf 0.05 is below the deployed confidence 0.25" in err
    assert "--allow-conf-below-deployed" in err


def test_the_lower_confidence_can_be_chosen_on_purpose_and_the_report_says_so(
    tmp_path, monkeypatch, capsys
):
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True,
        "--gate", *SMALL, "--conf", "0.05", "--allow-conf-below-deployed",
    )  # fmt: skip

    assert rc == 0
    assert (report["conf"], report["spec_conf"], report["conf_overridden"]) == (0.05, 0.25, True)
    assert report["gate"]["conf_below_deployed"] is True
    assert "WARNING gate measured at conf 0.05, below the deployed 0.25" in capsys.readouterr().out


def test_a_higher_confidence_only_makes_the_gate_stricter_and_is_recorded(tmp_path, monkeypatch):
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL, "--conf", "0.6"
    )

    assert rc == 0
    assert (report["conf"], report["spec_conf"], report["conf_overridden"]) == (0.6, 0.25, True)
    assert report["gate"]["conf_below_deployed"] is False


def test_a_conf_equal_to_the_deployed_one_is_not_an_override(tmp_path, monkeypatch):
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", *SMALL, "--conf", "0.25"
    )

    assert rc == 0 and report["conf_overridden"] is False and report["spec_conf"] == 0.25


def test_a_measurement_without_a_gate_may_use_any_confidence_and_records_it(tmp_path, monkeypatch):
    rc, report = run_cli(
        tmp_path,
        monkeypatch,
        covering(12),
        lambda i, c: True,
        "--conf",
        "0.05",
        "--min-images",
        "10",
    )

    assert rc == 0 and report["verdict"] == "not_applied"
    assert (report["conf"], report["spec_conf"], report["conf_overridden"]) == (0.05, 0.25, True)
    assert "conf_below_deployed" not in report["gate"], "no gate was applied: nothing to qualify"


def test_an_invalid_report_carries_the_deployed_confidence_too(tmp_path, monkeypatch):
    rc, report = run_cli(
        tmp_path, monkeypatch, covering(12), lambda i, c: True, "--gate", "--conf", "0.6",
        write_labels=False,
    )  # fmt: skip

    assert rc == 3 and report["verdict"] == "invalid"
    assert (report["conf"], report["spec_conf"], report["conf_overridden"]) == (0.6, 0.25, True)


def test_conf_reaches_the_backend_the_spec_it_builds_carries_the_overridden_threshold(
    tmp_path, monkeypatch
):
    """No test passed ``--conf`` (override dropped went unnoticed): the backend is built on it."""
    handed: list[float] = []
    root = tmp_path / "wired"
    images, labels = _dataset(root, covering(12))
    (root / "m.onnx").write_bytes(b"onnx")
    scripted = ScriptedBackend(lambda i, c: True, covering(12))

    def backend(a, s):
        handed.append(s.conf_threshold)
        return scripted

    monkeypatch.setattr(ev, "_backend", backend)
    monkeypatch.setattr(ev, "_load_spec", lambda a: SPEC)
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))
    argv = [*("--backend", "onnx"), *("--model", str(root / "m.onnx")), *("--conf", "0.6")]
    argv += [*("--images", str(images)), *("--labels", str(labels)), *("--min-images", "10")]
    argv += ["--out", str(root / "r.json")]

    assert ev.main(argv) == 0

    assert handed == [0.6]


def test_the_report_of_a_real_spec_file_records_the_sidecars_own_confidence(tmp_path, monkeypatch):
    """Through the real ``_load_spec``: the 0.5 of the sidecar, not a default, is ``spec_conf``."""
    root = tmp_path / "real"
    images, labels = _dataset(root, covering(12))
    (root / "m.onnx").write_bytes(b"onnx")
    spec_file = root / "spec.json"
    spec_file.write_text(
        json.dumps(
            {
                "schema": "lawnberry-detector/1",
                "arch": "yolo26",
                "input_size": 640,
                "class_names": list(NAMES),
                "conf_threshold": 0.5,
            }
        )
    )
    scripted = ScriptedBackend(lambda i, c: True, covering(12))
    monkeypatch.setattr(ev, "_backend", lambda a, s: scripted)
    monkeypatch.setattr(ev, "_read_rgb", lambda p: np.zeros((H, W, 3), dtype=np.uint8))
    common = [*("--backend", "onnx"), *("--model", str(root / "m.onnx")), "--spec", str(spec_file)]
    common += [*("--images", str(images)), *("--labels", str(labels)), *("--min-images", "10")]

    assert ev.main([*common, "--out", str(root / "a.json"), "--conf", "0.05", "--gate"]) == 2
    assert not (root / "a.json").exists()
    assert ev.main([*common, "--out", str(root / "b.json"), "--gate", *SMALL[:2]]) == 0
    report = json.loads((root / "b.json").read_text())

    assert (report["conf"], report["spec_conf"], report["conf_overridden"]) == (0.5, 0.5, False)
