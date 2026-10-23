"""Dataset audit must count what a training run would actually see."""

from __future__ import annotations

import json

from PIL import Image

from workshop.classes import NAMES
from workshop.dataset_stats import main, scan


def _img(path, size=(32, 24), shade=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (shade, shade, shade)).save(path)


def test_flat_layout_counts_labels_and_gaps(tmp_path):
    _img(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.1 0.1\n")
    _img(tmp_path / "b.png")
    (tmp_path / "b.txt").write_text("")  # present but empty
    _img(tmp_path / "c.png")  # no label file at all

    report = scan(tmp_path, NAMES)

    assert report["images"] == 3
    assert report["boxes"] == 2
    assert report["images_with_zero_boxes"] == 1
    assert report["images_without_label_file"] == 1
    assert report["malformed_lines"] == 0
    by_name = {c["name"]: c for c in report["classes"]}
    assert by_name[NAMES[0]]["instances"] == 1
    assert by_name[NAMES[3]]["instances"] == 1
    assert by_name[NAMES[1]]["instances"] == 0


def test_yolo_split_layout_resolves_labels_under_labels_dir(tmp_path):
    _img(tmp_path / "images" / "train" / "x.png")
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "x.txt").write_text("1 0.5 0.5 0.4 0.4\n")

    report = scan(tmp_path, NAMES)

    assert report["boxes"] == 1
    assert report["images_without_label_file"] == 0


def test_malformed_and_out_of_range_lines_are_counted_not_trusted(tmp_path):
    _img(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "nonsense\n"
        f"{len(NAMES)} 0.5 0.5 0.2 0.2\n"  # class id outside the name table
        "1 0.5 0.5\n"  # truncated
    )

    report = scan(tmp_path, NAMES)

    assert report["boxes"] == 1
    assert report["malformed_lines"] == 3


def test_exact_duplicates_are_reported(tmp_path):
    _img(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    _img(tmp_path / "copy_of_a.png")
    (tmp_path / "copy_of_a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    report = scan(tmp_path, NAMES)

    assert report["exact_duplicate_count"] == 1
    assert sorted(report["exact_duplicate_groups"][0]) == ["a.png", "copy_of_a.png"]


def test_near_duplicates_find_a_gradient_variant(tmp_path):
    base = tmp_path / "base.png"
    variant = tmp_path / "variant.png"
    tmp_path.mkdir(exist_ok=True)
    im = Image.new("L", (64, 64))
    for x in range(64):
        for y in range(64):
            im.putpixel((x, y), (x * 4) % 256)
    im.save(base)
    im.point(lambda v: min(255, v + 3)).save(variant)

    report = scan(tmp_path, NAMES, near_dupes=True, max_near=5)

    assert report["near_duplicate_count"] >= 1


def test_json_report_is_written(tmp_path, capsys):
    _img(tmp_path / "ds" / "a.png")
    (tmp_path / "ds" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "report.json"

    assert main([str(tmp_path / "ds"), "--json", str(out)]) == 0

    assert json.loads(out.read_text())["images"] == 1
    assert "boxes 1" in capsys.readouterr().out
