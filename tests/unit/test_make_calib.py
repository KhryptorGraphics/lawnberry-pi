"""``workshop.make_calib``: the calibration folder the zoo reads must hold every source.

The zoo lists the folder sorted and keeps the first ``calibset_size`` files, so a folder with more
files than that silently drops the later sources (here: every v2 frame behind v1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from workshop.make_calib import IMG_EXTS, build, main


def _src(root: Path, name: str, n: int) -> Path:
    d = root / name / "images"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i:06d}.jpg").write_bytes(f"{name}-{i}".encode())
    (d / "notes.txt").write_text("not an image")
    return d


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def test_exactly_count_images_with_an_equal_share_from_each_source(tmp_path):
    a, b = _src(tmp_path, "v1", 50), _src(tmp_path, "v2", 50)

    m = build(tmp_path / "calib", [a, b], 20, seed=1)

    names = [p.name for p in _images(tmp_path / "calib")]
    assert len(names) == 20 == m["count"]
    assert sum(n.startswith("c0_") for n in names) == 10
    assert sum(n.startswith("c1_") for n in names) == 10
    assert m["per_source"] == [10, 10]


def test_the_zoo_sorted_prefix_sees_every_source(tmp_path):
    a, b = _src(tmp_path, "v1", 40), _src(tmp_path, "v2", 40)
    build(tmp_path / "calib", [a, b], 24, seed=3)

    first_24_sorted = [p.name for p in _images(tmp_path / "calib")][:24]  # the zoo's slice

    assert {n[:2] for n in first_24_sorted} == {"c0", "c1"}


def test_odd_count_gives_the_remainder_to_the_first_sources(tmp_path):
    a, b, c = (_src(tmp_path, n, 20) for n in ("a", "b", "c"))

    m = build(tmp_path / "calib", [a, b, c], 10, seed=1)

    assert m["per_source"] == [4, 3, 3]


def test_same_seed_same_files_different_seed_different_files(tmp_path):
    a = _src(tmp_path, "v1", 100)

    one = build(tmp_path / "c1", [a], 30, seed=7)
    two = build(tmp_path / "c2", [a], 30, seed=7)
    other = build(tmp_path / "c3", [a], 30, seed=8)

    assert one["list_sha256"] == two["list_sha256"]
    assert one["list_sha256"] != other["list_sha256"]


def test_manifest_records_source_digests(tmp_path):
    a = _src(tmp_path, "v1", 5)

    build(tmp_path / "calib", [a], 3, seed=1)

    manifest = json.loads((tmp_path / "calib" / "calib_manifest.json").read_text())
    for f in manifest["files"]:
        assert (
            f["sha256"] == hashlib.sha256((tmp_path / "calib" / f["name"]).read_bytes()).hexdigest()
        )
        assert Path(f["source"]).read_bytes() == (tmp_path / "calib" / f["name"]).read_bytes()


def test_too_few_images_or_dirty_output_are_refused(tmp_path):
    a = _src(tmp_path, "v1", 5)
    with pytest.raises(SystemExit, match="need 10"):
        build(tmp_path / "calib", [a], 10, seed=1)
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        build(tmp_path / "used", [a], 2, seed=1)


def test_cli(tmp_path, capsys):
    a = _src(tmp_path, "v1", 8)

    assert main(["--out", str(tmp_path / "calib"), "--from", str(a), "--count", "4", "--copy"]) == 0

    assert "4 calibration images" in capsys.readouterr().out
