"""``workshop.train_detector.stage_dataset``: several sources must all reach the dataset.

Every SDG run numbers its frames ``000000.jpg`` upward, so the v1 + v2 training set the detector
pipeline needs had colliding file names. The old staging skipped an existing destination yet still
counted the file, so training silently saw only the first source's frames for every shared name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workshop.train_detector import stage_dataset


def _sdg(root: Path, name: str, n: int, cls: int) -> Path:
    """Flat SDG-style dir: ``images/NNNNNN.jpg`` + ``labels/NNNNNN.txt`` whose class is ``cls``."""
    src = root / name
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    for i in range(n):
        (src / "images" / f"{i:06d}.jpg").write_bytes(f"{name}-{i}".encode())
        (src / "labels" / f"{i:06d}.txt").write_text(f"{cls} 0.5 0.5 0.1 0.1\n")
    return src


def _staged(out: Path, kind: str, split: str) -> list[Path]:
    return sorted((out / kind / split).glob("*"))


def test_two_sdg_sources_with_the_same_frame_names_both_reach_training(tmp_path):
    v1 = _sdg(tmp_path, "v1", 30, cls=3)
    v2 = _sdg(tmp_path, "v2", 30, cls=7)
    out = tmp_path / "ds"

    stats = stage_dataset(out, [(v1, False), (v2, False)], link=False)

    train_imgs = _staged(out, "images", "train")
    assert stats["train"] == 60 and stats["val"] == 0
    assert len(train_imgs) == 60  # not 30: the second source must not be dropped
    classes = {
        p.read_text().split()[0] for p in _staged(out, "labels", "train")
    }  # both sources' labels survive
    assert classes == {"3", "7"}
    assert stats["per_source"][str(v1)]["prefix"] != stats["per_source"][str(v2)]["prefix"]


def test_every_staged_image_has_a_label_with_the_same_stem_and_content_survives(tmp_path):
    v1 = _sdg(tmp_path, "v1", 12, cls=0)
    v2 = _sdg(tmp_path, "v2", 12, cls=1)
    out = tmp_path / "ds"

    stage_dataset(out, [(v1, True), (v2, True)], link=False)  # md5 train/val split of both

    for split in ("train", "val"):
        imgs = {p.stem for p in _staged(out, "images", split)}
        labels = {p.stem for p in _staged(out, "labels", split)}
        assert imgs == labels
    contents = {p.read_bytes() for s in ("train", "val") for p in _staged(out, "images", s)}
    assert contents == {f"{n}-{i}".encode() for n in ("v1", "v2") for i in range(12)}


def test_unlabeled_images_are_not_staged(tmp_path):
    src = _sdg(tmp_path, "v1", 5, cls=2)
    (src / "labels" / "000003.txt").unlink()
    out = tmp_path / "ds"

    stats = stage_dataset(out, [(src, False)], link=False)

    assert stats["train"] == 4
    assert len(_staged(out, "images", "train")) == 4


def test_a_name_repeated_inside_one_source_fails_loudly(tmp_path):
    # images/train/x.jpg and images/val/x.jpg: both map to the same staged name.
    src = tmp_path / "real"
    for split in ("train", "val"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        (src / "images" / split / "x.jpg").write_bytes(split.encode())
        (src / "labels" / split / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    with pytest.raises(SystemExit, match="occurs twice"):
        stage_dataset(tmp_path / "ds", [(src, True)], link=False)


def test_hardlink_mode_stages_the_same_files(tmp_path):
    v1 = _sdg(tmp_path, "v1", 6, cls=4)
    v2 = _sdg(tmp_path, "v2", 6, cls=5)

    stats = stage_dataset(tmp_path / "ds", [(v1, False), (v2, False)], link=True)

    assert stats["train"] == 12
    assert len(_staged(tmp_path / "ds", "images", "train")) == 12
