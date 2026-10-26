"""``workshop.merge_sdg``: concurrent SDG shards become one honest dataset.

The shards all number their frames from ``000000``. A merge that keeps the names loses frames,
and one that trusts the manifests loses the fact that a killed shard wrote fewer labels than it
declared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workshop.classes import NAMES
from workshop.merge_sdg import main, merge


def _shard(root: Path, name: str, frames: list[list[int]], **over) -> Path:
    """One finished shard: ``frames[i]`` lists the class ids boxed in frame ``i``."""
    shard = root / name
    for sub in ("images", "labels", "rejected"):
        (shard / sub).mkdir(parents=True)
    counts = dict.fromkeys(NAMES, 0)
    for i, classes in enumerate(frames):
        (shard / "images" / f"{i:06d}.jpg").write_bytes(f"{name}-{i}".encode())
        lines = [f"{c} 0.5 0.5 0.2 0.2" for c in classes]
        (shard / "labels" / f"{i:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        for c in classes:
            counts[NAMES[c]] += 1
    (shard / "rejected" / "000099_exposure.jpg").write_bytes(b"black")
    manifest = {
        "stage": "isaac_sdg",
        "isaac_sim": "6.0.0",
        "gpu": "Tesla PG500-216",
        "seed": 100 + len(list(root.iterdir())),
        "frames": len(frames) + 1,
        "width": 1280,
        "height": 720,
        "accum": 16,
        "renderer": "RealTimePathTracing",
        "spp": None,
        "labels": "idpass",
        "assets_root": "/data/isaac-assets/6.0",
        "asset_sources": {"local": 2, "s3": 3},
        "assets_sha256": "ab" * 32,
        "classes": list(NAMES),
        "instance_counts": counts,
        "rejected_frames": 0,
        "rejected_exposure": 1,
        "seconds": 100.0 + len(frames),
    }
    manifest.update(over)
    (shard / "manifest.json").write_text(json.dumps(manifest))
    return shard


def test_merge_keeps_every_frame_of_every_shard(tmp_path):
    a = _shard(tmp_path, "a", [[0], [3, 3], []])
    b = _shard(tmp_path, "b", [[5], [23]])
    out = tmp_path / "merged"

    merged = merge(out, [a, b])

    names = sorted(p.name for p in (out / "images").iterdir())
    assert names == [f"sh0_{i:06d}.jpg" for i in range(3)] + [f"sh1_{i:06d}.jpg" for i in range(2)]
    assert (out / "images" / "sh1_000001.jpg").read_bytes() == b"b-1"  # content, not just names
    assert (out / "labels" / "sh0_000001.txt").read_text().count("\n") == 2
    assert merged["images"] == 5
    assert sorted(p.name for p in (out / "rejected").iterdir()) == [
        "sh0_000099_exposure.jpg",
        "sh1_000099_exposure.jpg",
    ]


def test_merged_manifest_sums_counters_and_keeps_shards(tmp_path):
    a = _shard(tmp_path, "a", [[0], [3, 3]])
    b = _shard(tmp_path, "b", [[3], [23]])

    merged = merge(tmp_path / "merged", [a, b])

    assert merged["instance_counts"]["hose"] == 3
    assert merged["instance_counts"]["person"] == 1
    assert merged["instance_counts"]["unknown_obstacle"] == 1
    assert merged["frames"] == 6 and merged["rejected_exposure"] == 2
    assert merged["seeds"] == [r["seed"] for r in merged["shards"]]
    assert merged["seconds"] == max(r["seconds"] for r in merged["shards"])
    on_disk = json.loads((tmp_path / "merged" / "manifest.json").read_text())
    assert on_disk["instance_counts"] == merged["instance_counts"]
    assert [s["images"] for s in on_disk["shards"]] == [2, 2]


def test_shards_rendered_differently_are_refused(tmp_path):
    a = _shard(tmp_path, "a", [[0]])
    b = _shard(tmp_path, "b", [[1]], accum=2)

    with pytest.raises(SystemExit, match="accum"):
        merge(tmp_path / "merged", [a, b])


def test_a_shard_killed_mid_run_is_refused(tmp_path):
    # The manifest was written for 3 boxes but a label file lost one (e.g. a truncated write).
    a = _shard(tmp_path, "a", [[0], [3, 3]])
    (a / "labels" / "000001.txt").write_text("3 0.5 0.5 0.2 0.2\n")

    with pytest.raises(SystemExit, match="instance_counts"):
        merge(tmp_path / "merged", [a])


def test_an_image_without_a_label_is_refused(tmp_path):
    a = _shard(tmp_path, "a", [[0], [1]])
    (a / "labels" / "000001.txt").unlink()

    with pytest.raises(SystemExit, match="no label file"):
        merge(tmp_path / "merged", [a])


def test_a_shard_without_a_manifest_did_not_finish(tmp_path):
    a = _shard(tmp_path, "a", [[0]])
    (a / "manifest.json").unlink()

    with pytest.raises(SystemExit, match="did not finish"):
        merge(tmp_path / "merged", [a])


def test_class_order_must_match_the_workshop_classes(tmp_path):
    reordered = [NAMES[1], NAMES[0], *NAMES[2:]]
    a = _shard(tmp_path, "a", [[0]], classes=reordered)

    with pytest.raises(SystemExit, match="class order"):
        merge(tmp_path / "merged", [a])


def test_existing_output_is_never_overwritten(tmp_path):
    a = _shard(tmp_path, "a", [[0]])
    out = tmp_path / "merged"
    out.mkdir()
    (out / "keep.txt").write_text("x")

    with pytest.raises(SystemExit, match="not empty"):
        merge(out, [a])
    assert (out / "keep.txt").read_text() == "x"


def test_cli_reports_missing_classes(tmp_path, capsys):
    a = _shard(tmp_path, "a", [[0]])

    assert main(["--out", str(tmp_path / "merged"), "--copy", str(a)]) == 0

    captured = capsys.readouterr()
    assert "1 images" in captured.out and "1/24 classes present" in captured.out
    assert "zero instances of" in captured.err and "pet" in captured.err


# ---- audit review TOOL-05: duplicate shards and the second label set ---------------------------


def test_the_same_shard_twice_is_refused_not_doubled(tmp_path):
    # `merge(out, [a, a])` used to give 6 images from 3 frames and seeds [101, 101].
    a = _shard(tmp_path, "a", [[0], [3, 3], []])
    out = tmp_path / "merged"

    with pytest.raises(SystemExit, match="the same shard is given more than once"):
        merge(out, [a, a])

    assert not out.exists()


def test_the_same_shard_spelled_two_ways_is_still_the_same_shard(tmp_path):
    a = _shard(tmp_path, "a", [[0]])
    link = tmp_path / "link-to-a"
    link.symlink_to(a, target_is_directory=True)

    for other in (tmp_path / "a" / ".." / "a", link, Path(str(a) + "/")):
        with pytest.raises(SystemExit, match="more than once"):
            merge(tmp_path / "merged", [a, other])


def test_two_shards_rendered_with_one_seed_are_refused(tmp_path):
    a = _shard(tmp_path, "a", [[0]], seed=7)
    b = _shard(tmp_path, "b", [[0]], seed=7)

    with pytest.raises(SystemExit, match=r"share a seed \(7\)"):
        merge(tmp_path / "merged", [a, b])


def test_a_manifest_without_a_seed_is_a_clean_refusal(tmp_path):
    a = _shard(tmp_path, "a", [[0]])
    manifest = json.loads((a / "manifest.json").read_text())
    del manifest["seed"]
    (a / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match="manifest has no seed"):
        merge(tmp_path / "merged", [a])


def _both(shard: Path, frames: list[list[int]]) -> None:
    """Turn a shard into a ``--labels both`` one: a second label set per image."""
    manifest = json.loads((shard / "manifest.json").read_text())
    manifest["labels"] = "both"
    (shard / "manifest.json").write_text(json.dumps(manifest))
    (shard / "labels_idpass").mkdir()
    for i, classes in enumerate(frames):
        (shard / "labels_idpass" / f"{i:06d}.txt").write_text(
            "".join(f"{c} 0.4 0.4 0.1 0.1\n" for c in classes)
        )


def test_the_second_label_set_of_a_both_shard_is_merged_not_dropped(tmp_path):
    frames_a, frames_b = [[0], [3, 3]], [[5]]
    a, b = _shard(tmp_path, "a", frames_a), _shard(tmp_path, "b", frames_b)
    _both(a, frames_a)
    _both(b, frames_b)

    merged = merge(tmp_path / "merged", [a, b])

    out = tmp_path / "merged" / "labels_idpass"
    assert sorted(p.name for p in out.iterdir()) == [
        "sh0_000000.txt",
        "sh0_000001.txt",
        "sh1_000000.txt",
    ]
    assert (out / "sh0_000001.txt").read_text() == "3 0.4 0.4 0.1 0.1\n3 0.4 0.4 0.1 0.1\n"
    assert merged["labels"] == "both" and "labels_idpass" in merged["subdirs"]
    # every merged image has both its label files
    for image in (tmp_path / "merged" / "images").iterdir():
        assert (tmp_path / "merged" / "labels" / f"{image.stem}.txt").is_file()
        assert (out / f"{image.stem}.txt").is_file()


def test_a_both_shard_without_its_second_label_set_is_refused(tmp_path):
    a = _shard(tmp_path, "a", [[0]], labels="both")  # manifest says both, no labels_idpass/

    with pytest.raises(SystemExit, match="labels is 'both' but labels_idpass/ is missing"):
        merge(tmp_path / "merged", [a])


def test_a_second_label_set_the_manifest_does_not_declare_is_refused(tmp_path):
    a = _shard(tmp_path, "a", [[0]])  # labels: idpass
    (a / "labels_idpass").mkdir()

    with pytest.raises(SystemExit, match="labels is 'idpass' but labels_idpass/ exists"):
        merge(tmp_path / "merged", [a])


def test_second_labels_without_an_image_are_refused(tmp_path):
    a = _shard(tmp_path, "a", [[0]])
    _both(a, [[0]])
    (a / "labels_idpass" / "000042.txt").write_text("0 0.4 0.4 0.1 0.1\n")

    with pytest.raises(
        SystemExit, match=r"labels_idpass/ has labels without an image: \['000042.txt'\]"
    ):
        merge(tmp_path / "merged", [a])


def test_the_idpass_images_of_save_id_shards_are_carried_along(tmp_path):
    a, b = _shard(tmp_path, "a", [[0]]), _shard(tmp_path, "b", [[1]])
    for shard in (a, b):
        (shard / "idpass").mkdir()
        (shard / "idpass" / "000000.png").write_bytes(f"id-{shard.name}".encode())
        (shard / "idpass" / "000099.png").write_bytes(b"of a rejected frame")  # no image: kept

    merged = merge(tmp_path / "merged", [a, b])

    out = tmp_path / "merged" / "idpass"
    assert (out / "sh1_000000.png").read_bytes() == b"id-b"
    assert sorted(p.name for p in out.iterdir()) == [
        "sh0_000000.png",
        "sh0_000099.png",
        "sh1_000000.png",
        "sh1_000099.png",
    ]
    assert "idpass" in merged["subdirs"]


def test_an_unknown_entry_in_a_shard_is_named_not_silently_dropped(tmp_path, capsys):
    a = _shard(tmp_path, "a", [[0]])
    (a / "sdg.log").write_text("frame 1/1")
    (a / "notes").mkdir()
    b = _shard(tmp_path, "b", [[1]])

    merged = merge(tmp_path / "merged", [a, b])

    assert merged["not_merged"] == {str(a): ["notes", "sdg.log"]}
    assert not (tmp_path / "merged" / "sdg.log").exists()
    assert json.loads((tmp_path / "merged" / "manifest.json").read_text())["not_merged"]
    assert main(["--out", str(tmp_path / "again"), "--copy", str(a), str(b)]) == 0
    assert (
        f"{a}: not merged (not written by isaac_sdg.py): notes, sdg.log" in capsys.readouterr().err
    )


def test_a_clean_merge_reports_nothing_unmerged(tmp_path):
    a = _shard(tmp_path, "a", [[0]])

    merged = merge(tmp_path / "merged", [a])

    assert merged["not_merged"] == {} and merged["subdirs"] == ["images", "labels", "rejected"]
