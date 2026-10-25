"""Per-class TP/FP/FN accounting behind the compiled-model recall gates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from workshop.classes import CLASSES, NAMES
from workshop.detector_eval import (
    BASELINE_SCHEMAS,
    GATE_MIN_GT,
    MIN_IMAGES,
    BaselineError,
    DatasetError,
    baseline_status,
    count_ground_truth,
    evidence_problems,
    finite_number,
    gate_status,
    load_baseline,
    load_yolo_labels,
    match,
    merge,
    pair_labels,
    parse_label_file,
    read_json,
    summarize,
    to_pixels,
    validate_baseline,
)


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
    assert load_yolo_labels(txt, 200, 100, 24) == [(3, [75.0, 25.0, 125.0, 75.0])]


def test_a_missing_label_file_is_an_error_not_an_empty_list(tmp_path):
    # It used to read as "no objects", so a wrong labels directory left every class at gt 0.
    with pytest.raises(DatasetError, match="label file missing"):
        load_yolo_labels(tmp_path / "missing.txt", 200, 100, 24)


def test_an_empty_label_file_is_an_image_without_objects(tmp_path):
    empty = tmp_path / "a.txt"
    empty.write_text("\n  \n", encoding="utf-8")

    assert parse_label_file(empty, 24) == []


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 0.5 0.5 0.1", "got 4 fields"),
        ("0 0.5 0.5 0.1 0.1 0.9", "got 6 fields"),
        ("0 0.5 0.5 0.1 0.1 0.1 0.1 0.1 0.1", "got 9 fields"),
        ("person 0.5 0.5 0.1 0.1", "non-numeric"),
        ("0 0.5 abc 0.1 0.1", "non-numeric"),
        ("0 nan 0.5 0.1 0.1", "non-finite"),
        ("0 0.5 0.5 inf 0.1", "non-finite"),
        ("-1 0.5 0.5 0.1 0.1", "not a non-negative integer"),
        ("1.5 0.5 0.5 0.1 0.1", "not a non-negative integer"),
        ("24 0.5 0.5 0.1 0.1", "class id 24 but the spec has 24 classes"),
        ("0 0.5 0.5 0 0.1", "no area"),
        ("0 0.5 0.5 0.1 -0.1", "no area"),
    ],
)
def test_a_row_that_cannot_be_used_is_an_error_never_skipped(tmp_path, row, message):
    # A skipped row is a ground-truth box that silently stops counting (recall goes up).
    txt = tmp_path / "a.txt"
    txt.write_text(f"1 0.5 0.5 0.1 0.1\n{row}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=message) as err:
        parse_label_file(txt, len(NAMES))
    assert f"{txt}:2:" in str(err.value)


def test_integral_float_class_ids_are_accepted(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("3.0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    assert parse_label_file(txt, 24) == [(3, 0.5, 0.5, 0.1, 0.1)]


def test_a_label_file_that_is_not_utf8_is_a_dataset_error(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_bytes(b"0 0.5 0.5 0.1 \xff\n")

    with pytest.raises(DatasetError, match="cannot read label file"):
        parse_label_file(txt, 24)


def test_to_pixels_scales_by_the_image_size():
    assert to_pixels([(3, 0.5, 0.5, 0.25, 0.5)], 200, 100) == [(3, [75.0, 25.0, 125.0, 75.0])]


def _images(root: Path, stems: list[str], ext: str = ".jpg") -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for stem in stems:
        (root / f"{stem}{ext}").write_bytes(b"x")
        out.append(root / f"{stem}{ext}")
    return out


def _labels(root: Path, stems: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return root


def test_pair_labels_maps_every_image_to_its_label(tmp_path):
    images = _images(tmp_path / "images", ["a", "b"])
    labels = _labels(tmp_path / "labels", ["a", "b"])

    assert pair_labels(images, labels) == {images[0]: labels / "a.txt", images[1]: labels / "b.txt"}


def test_pair_labels_needs_the_labels_directory(tmp_path):
    images = _images(tmp_path / "images", ["a"])

    with pytest.raises(DatasetError, match="labels directory not found"):
        pair_labels(images, tmp_path / "nowhere")


def test_pair_labels_names_the_images_without_a_label_file(tmp_path):
    images = _images(tmp_path / "images", [f"img{i}" for i in range(8)])
    labels = _labels(tmp_path / "labels", ["img0", "img1"])

    with pytest.raises(DatasetError) as err:
        pair_labels(images, labels)

    [problem] = err.value.problems
    assert problem.startswith("6 of 8 images have no label file")
    assert "img2.jpg" in problem and "... (6 in all)" in problem  # five quoted, then the count
    assert "empty label file" in problem


def test_pair_labels_rejects_labels_without_an_image(tmp_path):
    images = _images(tmp_path / "images", ["a"])
    labels = _labels(tmp_path / "labels", ["a", "b", "c"])

    with pytest.raises(DatasetError, match="2 label files have no image.*b.txt, c.txt"):
        pair_labels(images, labels)


def test_pair_labels_reports_both_problems_at_once(tmp_path):
    images = _images(tmp_path / "images", ["a", "b"])
    labels = _labels(tmp_path / "labels", ["a", "z"])

    with pytest.raises(DatasetError) as err:
        pair_labels(images, labels)

    assert len(err.value.problems) == 2


def test_pair_labels_ignores_the_classes_file_of_labelling_tools(tmp_path):
    images = _images(tmp_path / "images", ["a"])
    labels = _labels(tmp_path / "labels", ["a", "classes"])

    assert list(pair_labels(images, labels)) == images


def test_pair_labels_refuses_two_images_that_share_a_label(tmp_path):
    images = _images(tmp_path / "images", ["a"]) + _images(tmp_path / "images", ["a"], ".png")
    labels = _labels(tmp_path / "labels", ["a"])

    with pytest.raises(DatasetError, match=r"share a file stem.*: a$"):
        pair_labels(images, labels)


def test_count_ground_truth_lists_every_class_zero_included(tmp_path):
    rows = {
        Path("a"): [(0, 0.5, 0.5, 0.1, 0.1), (3, 0.5, 0.5, 0.1, 0.1)],
        Path("b"): [(0, 1, 1, 1, 1)],
    }

    got = count_ground_truth(rows, NAMES)

    assert list(got) == NAMES
    assert got["person"] == 2 and got["hose"] == 1 and got["shed"] == 0


TIER1 = [c.name for c in CLASSES if c.tier == 1]


def _gt(**over: int) -> dict[str, int]:
    return {**dict.fromkeys(NAMES, 0), **{n: GATE_MIN_GT for n in TIER1}, **over}


def test_the_defaults_are_the_documented_ones():
    assert (GATE_MIN_GT, MIN_IMAGES) == (150, 100)
    assert 0.98**149 < 0.05 < 0.98**148  # why 150: zero misses in 149+ proves 98 % at 95 %


def test_enough_evidence_has_no_problems():
    assert evidence_problems(NAMES, _gt(), MIN_IMAGES, gate=True) == []


def test_a_verdict_needs_the_image_floor_and_the_gate_needs_more():
    gate = evidence_problems(NAMES, _gt(), MIN_IMAGES - 1, gate=True)
    baseline_only = evidence_problems(NAMES[:3], {}, MIN_IMAGES - 1, gate=False)

    assert gate == [f"{MIN_IMAGES - 1} images; a verdict needs at least {MIN_IMAGES}"]
    assert baseline_only == gate  # the class set and tier-1 rules are the gate's alone


@pytest.mark.parametrize("short", TIER1)
def test_each_tier1_class_needs_its_own_minimum(short):
    gt = _gt(**{short: GATE_MIN_GT - 1})

    [problem] = evidence_problems(NAMES, gt, MIN_IMAGES, gate=True)

    assert f"{short} {GATE_MIN_GT - 1}" in problem
    assert sum(f"{n} {GATE_MIN_GT}" in problem for n in TIER1) == 0


def test_lower_tiers_have_no_minimum_for_the_gate():
    assert evidence_problems(NAMES, _gt(hose=0, dropoff=1), MIN_IMAGES, gate=True) == []


def test_the_gate_needs_the_class_set_in_order():
    other = [*NAMES[1:], NAMES[0]]

    [problem] = evidence_problems(other, _gt(), MIN_IMAGES, gate=True)

    assert "workshop.classes.NAMES" in problem


def test_gate_status_fails_a_class_below_its_tier_gate_and_lists_the_unevaluated():
    counts = merge({}, match([], [(NAMES.index("person"), BOX)] * 50))  # person: 0 of 50
    counts = merge(counts, match([_p(NAMES.index("rock"), 0.9, BOX)], [(NAMES.index("rock"), BOX)]))

    verdict = gate_status(summarize(counts, NAMES))

    assert verdict["failures"] == [{"class": "person", "recall": 0.0, "gate": 0.98}]
    unevaluated = {c["class"] for c in verdict["not_evaluated"]}
    assert "shed" in unevaluated and "person" not in unevaluated and "rock" not in unevaluated
    assert {"class": "pet", "tier": 1, "reason": "no ground truth"} in verdict["not_evaluated"]


def test_gate_status_uses_each_tiers_own_threshold():
    # recall 0.8 passes tier 3 (0.80), tier 4 (0.70) and tier 5 (0.80); fails tier 1 and 2 (0.85)
    per_class = {
        name: {"gt": 10, "tp": 8, "fp": 0, "fn": 2, "recall": 0.8, "precision": 1.0}
        for name in ("person", "rock", "shed", "path", "unknown_obstacle")
    }

    failed = {f["class"] for f in gate_status(per_class)["failures"]}

    assert failed == {"person", "rock"}  # shed (3) 0.80, path (4) 0.70 and tier 5 (0.80) pass


@pytest.mark.parametrize(
    "value, expected",
    [(0, True), (1.5, True), (-2, True), (True, False), (None, False), ("1", False),
     (math.nan, False), (math.inf, False), (-math.inf, False)],
)  # fmt: skip
def test_finite_number(value, expected):
    assert finite_number(value) is expected


def test_read_json_refuses_nan_and_infinity_literals(tmp_path):
    for text in ('{"recall": NaN}', '{"recall": Infinity}', '{"recall": -Infinity}'):
        path = tmp_path / "a.json"
        path.write_text(text)

        with pytest.raises(BaselineError, match="non-finite number literal"):
            read_json(path, BaselineError)


def test_read_json_reports_unreadable_unparsable_and_binary_files_as_the_given_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    binary = tmp_path / "bin.json"
    binary.write_bytes(b"\xff\xfe{}")

    for path in (bad, binary, tmp_path / "missing.json"):
        with pytest.raises(DatasetError, match="cannot read"):
            read_json(path, DatasetError)
    assert load_baseline(_write(tmp_path / "ok.json", {"a": 1})) == {"a": 1}


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


def _report(per_class=None, **over):
    """What ``eval_detector`` writes for a valid run (the parts a baseline is checked on)."""
    table = (
        per_class
        if per_class is not None
        else {
            n: {"gt": 40 if n == "person" else 0, "recall": 0.95 if n == "person" else None}
            for n in NAMES
        }
    )
    return {
        "schema": "detector-eval/2", "status": "ok", "images": 50, "iou": 0.5, "conf": 0.25,
        "per_class": table, **over,
    }  # fmt: skip


def _validate(baseline, **over):
    gt = over.pop("gt", {n: 40 if n == "person" else 0 for n in NAMES})
    args = {"class_names": NAMES, "gt": gt, "images": 50, "iou": 0.5, "conf": 0.25, "min_gt": 20}
    return validate_baseline(baseline, **{**args, **over})


def test_a_report_of_the_same_evaluation_validates():
    table = _validate(_report())

    assert table["person"] == {"gt": 40, "recall": 0.95}


@pytest.mark.parametrize("schema", BASELINE_SCHEMAS)
def test_reports_of_both_schema_versions_are_comparable(schema):
    assert _validate(_report(schema=schema))["person"]["recall"] == 0.95


def test_a_v1_report_without_a_status_is_accepted():
    old = _report()
    del old["status"]
    old["schema"] = "detector-eval/1"

    assert _validate(old)["person"]["gt"] == 40


@pytest.mark.parametrize(
    "baseline, message",
    [
        ([], "not a JSON object"),
        ("report", "not a JSON object"),
        ({"schema": "train-v1"}, "not a detector-eval report \\(schema 'train-v1'"),
        ({}, "not a detector-eval report"),
        (_report(status="invalid"), "not a valid evaluation"),
        (_report(per_class=[]), "no 'per_class' table"),
        ({**_report(), "per_class": None}, "no 'per_class' table"),
        (_report(images=49), "covers 49 images"),
        (_report(images=None), "covers None images"),
        (_report(iou=0.3), "measured at iou 0.3"),
        (_report(iou=None), "measured at iou None"),
        (_report(iou=math.nan), "measured at iou nan"),
        (_report(conf=0.5), "measured at conf 0.5"),
        (_report(conf="0.25"), "measured at conf '0.25'"),
    ],
)
def test_a_baseline_that_is_not_this_evaluation_is_refused(baseline, message):
    with pytest.raises(BaselineError, match=message):
        _validate(baseline)


def test_a_baseline_with_other_classes_names_the_difference():
    table = _report()["per_class"]
    del table["pet"]
    table["zebra"] = {"gt": 0, "recall": None}

    with pytest.raises(BaselineError, match=r"missing: pet; unexpected: zebra"):
        _validate(_report(per_class=table))


@pytest.mark.parametrize(
    "entry",
    [
        {"gt": 40, "recall": math.nan},
        {"gt": 40, "recall": math.inf},
        {"gt": 40, "recall": -0.1},
        {"gt": 40, "recall": 1.1},
        {"gt": 40, "recall": None},  # ground truth but no recall
        {"gt": 40, "recall": "0.9"},
        {"gt": 40, "recall": True},
        {"gt": 0, "recall": 0.5},  # a recall without ground truth
        {"gt": -1, "recall": None},
        {"gt": 4.5, "recall": 0.5},
        {"gt": True, "recall": 0.5},
        {"recall": 0.5},
        None,
        "person",
    ],
)
def test_a_class_entry_that_is_not_a_count_and_a_recall_is_refused(entry):
    table = _report()["per_class"]
    table["person"] = entry

    with pytest.raises(BaselineError, match="baseline class 'person'"):
        _validate(_report(per_class=table))


def test_a_baseline_measured_on_other_ground_truth_names_the_classes():
    table = _report()["per_class"]
    table["person"] = {"gt": 41, "recall": 0.9}

    with pytest.raises(BaselineError, match="other ground truth: person 41 vs 40"):
        _validate(_report(per_class=table))


def test_a_baseline_with_nothing_to_compare_is_refused():
    with pytest.raises(BaselineError, match="no class has 50 ground-truth boxes"):
        _validate(_report(), min_gt=50)


def test_the_comparison_floats_tolerate_json_round_trips():
    assert _validate(_report(iou=0.5 + 1e-12, conf=0.25 - 1e-13))["person"]["gt"] == 40


def test_baseline_status_applies_the_drop_and_lists_what_it_compared_and_skipped():
    per_class = summarize(
        merge(merge({}, match([], [(0, BOX)] * 40)), match([], [(1, BOX)] * 5)), NAMES
    )  # person recall 0.0 on 40 boxes, pet 0.0 on 5
    table = {"person": {"recall": 0.95}, "pet": {"recall": 1.0}}

    got = baseline_status(
        {k: per_class[k] for k in ("person", "pet")}, table, max_drop=0.10, min_gt=20
    )

    assert [f["class"] for f in got["failures"]] == ["person"]
    assert got["failures"][0]["baseline"] == 0.95 and got["failures"][0]["drop"] == 0.95
    assert got["compared"] == ["person"]
    assert got["skipped"] == [
        {"class": "pet", "gt": 5, "reason": "fewer than 20 ground-truth boxes"}
    ]


def test_baseline_status_drop_boundary_is_strict():
    per_class = {"person": {"gt": 20, "tp": 18, "fp": 0, "fn": 2, "recall": 0.9, "precision": 1.0}}

    at_limit = baseline_status(per_class, {"person": {"recall": 1.0}}, max_drop=0.10, min_gt=20)
    over = baseline_status(per_class, {"person": {"recall": 1.0}}, max_drop=0.09, min_gt=20)

    assert at_limit["failures"] == [] and len(over["failures"]) == 1


def test_the_gt_floor_of_the_baseline_check_is_inclusive():
    per_class = {"person": {"gt": 20, "tp": 0, "fp": 0, "fn": 20, "recall": 0.0, "precision": None}}

    got = baseline_status(per_class, {"person": {"recall": 1.0}}, max_drop=0.10, min_gt=20)

    assert got["compared"] == ["person"] and len(got["failures"]) == 1
