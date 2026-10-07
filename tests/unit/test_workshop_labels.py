"""W2 labeling helpers: class order, prompt mapping, YOLO conversion, review gate."""

import json

from workshop.autolabel import prompt_lookup, to_yolo
from workshop.classes import CLASSES, HARD_STOP, NAMES
from workshop.review import release_check


def test_hard_stop_classes_first_and_unique():
    assert NAMES[:3] == ["person", "pet", "wildlife"] == HARD_STOP
    assert len(set(NAMES)) == len(NAMES)
    assert CLASSES[-1].name == "unknown_obstacle"


def test_every_prompt_phrase_maps_to_its_class():
    lk = prompt_lookup()
    assert NAMES[lk["dog"]] == "pet" and NAMES[lk["garden hose"]] == "hose"
    assert NAMES[lk["deer"]] == "wildlife"


def test_yolo_conversion_clips_to_image():
    cx, cy, w, h = to_yolo([-10, 0, 50, 100], 100, 200)
    assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)


def _ds(tmp_path, reviewed=True, checks=None):
    (tmp_path / "images").mkdir()
    for i in range(20):
        (tmp_path / "images" / f"{i}.jpg").write_bytes(b"x")
    (tmp_path / "review.jsonl").write_text(
        "\n".join(
            json.dumps({"image": f"{i}.jpg", "cls": 0, "reviewed": reviewed}) for i in range(20)
        )
    )
    if checks is not None:
        (tmp_path / "spot_checks.jsonl").write_text("\n".join(json.dumps(c) for c in checks))
    return tmp_path


def test_release_blocks_unreviewed_and_missing_spot_checks(tmp_path):
    p = release_check(_ds(tmp_path, reviewed=False))
    assert any("not human-reviewed" in x for x in p) and any("spot checks" in x for x in p)


def test_release_blocks_high_spot_error_rate(tmp_path):
    p = release_check(_ds(tmp_path, checks=[{"image": "0.jpg", "boxes": 10, "errors": 1}]))
    assert p == ["spot-check error rate 10.0% > 2%"]


def test_release_passes_when_reviewed_and_audited(tmp_path):
    assert release_check(_ds(tmp_path, checks=[{"image": "0.jpg", "boxes": 50, "errors": 0}])) == []
