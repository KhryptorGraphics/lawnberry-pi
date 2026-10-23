"""The gate must refuse an inconsistent dataset, not just an unreviewed one."""

from __future__ import annotations

import json

from workshop.review import release_check


def _dataset(tmp_path, boxes: dict[str, int], reviewed: bool = True, manifest: bool = True):
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir()
    rows = []
    for name, count in boxes.items():
        (ds / "images" / name).write_bytes(b"")
        lines = "\n".join("0 0.5 0.5 0.1 0.1" for _ in range(count))
        (ds / "labels" / f"{name.rsplit('.', 1)[0]}.txt").write_text(
            lines + ("\n" if count else "")
        )
        rows += [
            {
                "image": name,
                "cls": 0,
                "score": 0.5,
                "yolo": [0.5, 0.5, 0.1, 0.1],
                "reviewed": reviewed,
            }
            for _ in range(count)
        ]
    (ds / "review.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if manifest:
        (ds / "manifest.json").write_text("{}")
    return ds, rows


def _spot(ds, images: int, boxes: int = 2, errors: int = 0):
    (ds / "spot_checks.jsonl").write_text(
        "\n".join(
            json.dumps({"image": f"i{n}.jpg", "boxes": boxes, "errors": errors})
            for n in range(images)
        )
        + "\n"
    )


def test_consistent_reviewed_dataset_is_releasable(tmp_path):
    ds, _ = _dataset(tmp_path, {"a.jpg": 2, "b.jpg": 1})

    assert release_check(ds) == ["0 spot checks < 1 (5% of 2 images)"]

    _spot(ds, images=1)
    assert release_check(ds) == []


def test_unreviewed_boxes_block_release(tmp_path):
    ds, _ = _dataset(tmp_path, {"a.jpg": 1}, reviewed=False)
    _spot(ds, 1)

    problems = release_check(ds)

    assert any("not human-reviewed" in p for p in problems)


def test_deleting_a_label_file_blocks_release(tmp_path):
    ds, _ = _dataset(tmp_path, {"a.jpg": 1})
    _spot(ds, 1)
    (ds / "labels" / "a.txt").unlink()

    problems = release_check(ds)

    assert any("no label file" in p for p in problems)


def test_label_and_review_disagreement_blocks_release(tmp_path):
    ds, _ = _dataset(tmp_path, {"a.jpg": 2})
    _spot(ds, 1)
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")  # one box now

    problems = release_check(ds)

    assert any("disagree on box count" in p for p in problems)


def test_deleted_verdict_is_excluded_from_the_count(tmp_path):
    ds, rows = _dataset(tmp_path, {"a.jpg": 2})
    _spot(ds, 1)
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    rows[0]["verdict"] = "deleted"
    (ds / "review.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert release_check(ds) == []


def test_missing_manifest_blocks_release(tmp_path):
    ds, _ = _dataset(tmp_path, {"a.jpg": 1}, manifest=False)
    _spot(ds, 1)

    problems = release_check(ds)

    assert any("manifest.json" in p for p in problems)


def test_spot_check_floor_rounds_up(tmp_path):
    # 21 images * 5% = 1.05 -> two checks required, not one.
    boxes = {f"i{n}.jpg": 1 for n in range(21)}
    ds, _ = _dataset(tmp_path, boxes)
    _spot(ds, images=1)

    problems = release_check(ds)

    assert any("spot checks < 2" in p for p in problems)

    _spot(ds, images=2)
    assert release_check(ds) == []
