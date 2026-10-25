"""``workshop.make_calib``: the calibration folder the zoo reads must hold every source.

The zoo lists the folder sorted and keeps the first ``calibset_size`` (1024) files, so the *sorted
prefix* of the folder is what the compiled model is calibrated on. Audit review TOOL-04: files
named ``c<source>_...`` made the first source sort first, so ``--count 2048`` calibrated on source
0 alone. The names now interleave the sources and ``--verify`` checks the rule.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from workshop.make_calib import (
    CALIBSET_SIZE,
    IMG_EXTS,
    build,
    calib_name,
    interleave,
    main,
    verify,
)


def _src(root: Path, name: str, n: int) -> Path:
    d = root / name / "images"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i:06d}.jpg").write_bytes(f"{name}-{i}".encode())
    (d / "notes.txt").write_text("not an image")
    return d


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def source_of(name: str) -> int:
    return int(re.search(r"(?:^|_)c(\d+)_", name).group(1))


def shares(names: list[str], n_sources: int) -> list[int]:
    count = Counter(source_of(n) for n in names)
    return [count[k] for k in range(n_sources)]


def test_exactly_count_images_with_an_equal_share_from_each_source(tmp_path):
    a, b = _src(tmp_path, "v1", 50), _src(tmp_path, "v2", 50)

    m = build(tmp_path / "calib", [a, b], 20, seed=1)

    names = [p.name for p in _images(tmp_path / "calib")]
    assert len(names) == 20 == m["count"]
    assert shares(names, 2) == [10, 10]
    assert m["per_source"] == [10, 10]


def test_every_sorted_prefix_holds_every_source_in_equal_share(tmp_path):
    a, b, c = (_src(tmp_path, n, 40) for n in ("v1", "v2", "v3"))
    build(tmp_path / "calib", [a, b, c], 40, seed=3)
    names = [p.name for p in _images(tmp_path / "calib")]

    for length in range(1, len(names) + 1):
        got = shares(names[:length], 3)
        assert max(got) - min(got) <= 1, (length, got)  # not just the one prefix the zoo uses
    assert shares(names[:24], 3) == [8, 8, 8]


def test_a_count_above_the_zoo_prefix_cannot_starve_a_source(tmp_path):
    # The review's repro: --count 2048 from two sources. The zoo reads the first 1024 sorted files
    # only; named c0_/c1_ those were all source 0 and source 1 was never calibrated on.
    a, b = _src(tmp_path, "v1", 1100), _src(tmp_path, "v2", 1100)

    m = build(tmp_path / "calib", [a, b], 2 * CALIBSET_SIZE, seed=1)

    names = [p.name for p in _images(tmp_path / "calib")]
    assert len(names) == 2048 and m["per_source"] == [1024, 1024]
    assert shares(names[:CALIBSET_SIZE], 2) == [512, 512]
    assert m["zoo_prefix"] == {"files": 1024, "per_source": [512, 512]}
    assert verify(tmp_path / "calib")["per_source"] == [512, 512]


def test_a_count_below_the_zoo_prefix_is_read_whole(tmp_path):
    a, b = _src(tmp_path, "v1", 30), _src(tmp_path, "v2", 30)

    m = build(tmp_path / "calib", [a, b], 20, seed=1)

    assert m["zoo_prefix"] == {"files": 20, "per_source": [10, 10]}


def test_names_start_with_a_zero_padded_position(tmp_path):
    a = _src(tmp_path, "v1", 3)

    build(tmp_path / "calib", [a], 3, seed=1)

    assert [p.name[:7] for p in _images(tmp_path / "calib")] == ["00000_c", "00001_c", "00002_c"]


def test_positions_stay_in_sort_order_however_large_the_count():
    # Padded to the width of the largest position (never fewer than 5 digits), so the zoo's
    # sorted listing is the position order.
    assert calib_name(7, 1, "x.jpg", 1024) == "00007_c1_x.jpg"
    assert calib_name(99999, 0, "x.jpg", 100000) == "99999_c0_x.jpg"
    assert calib_name(99999, 0, "x.jpg", 100001) == "099999_c0_x.jpg"
    names = [calib_name(p, 0, "x.jpg", 120000) for p in (9, 10, 99999, 100000, 119999)]
    assert names == sorted(names)


@pytest.mark.parametrize(
    "shares_in, expected",
    [
        ([4, 3, 3], [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
        ([2, 2], [0, 1, 0, 1]),
        ([3, 1], [0, 1, 0, 0]),
        ([0, 2], [1, 1]),
        ([5], [0, 0, 0, 0, 0]),
        ([], []),
    ],
)
def test_interleave_goes_round_robin_and_drops_exhausted_sources(shares_in, expected):
    assert interleave(shares_in) == expected


def test_odd_count_gives_the_remainder_to_the_first_sources(tmp_path):
    a, b, c = (_src(tmp_path, n, 20) for n in ("a", "b", "c"))

    m = build(tmp_path / "calib", [a, b, c], 10, seed=1)

    assert m["per_source"] == [4, 3, 3]
    assert shares([p.name for p in _images(tmp_path / "calib")], 3) == [4, 3, 3]


def test_same_seed_same_files_different_seed_different_files(tmp_path):
    a = _src(tmp_path, "v1", 100)

    one = build(tmp_path / "c1", [a], 30, seed=7)
    two = build(tmp_path / "c2", [a], 30, seed=7)
    other = build(tmp_path / "c3", [a], 30, seed=8)

    assert one["list_sha256"] == two["list_sha256"]
    assert [f["source"] for f in one["files"]] == [f["source"] for f in two["files"]]
    assert [f["source"] for f in one["files"]] != [f["source"] for f in other["files"]]


def test_manifest_records_source_digests_and_the_rule_it_was_built_for(tmp_path):
    a, b = _src(tmp_path, "v1", 5), _src(tmp_path, "v2", 5)

    build(tmp_path / "calib", [a, b], 6, seed=1)

    manifest = json.loads((tmp_path / "calib" / "calib_manifest.json").read_text())
    assert manifest["schema"] == "calib-set/2" and manifest["calibset_size"] == 1024
    for f in manifest["files"]:
        assert (
            f["sha256"] == hashlib.sha256((tmp_path / "calib" / f["name"]).read_bytes()).hexdigest()
        )
        assert Path(f["source"]).read_bytes() == (tmp_path / "calib" / f["name"]).read_bytes()
        assert f["source_index"] == source_of(f["name"])


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

    captured = capsys.readouterr()
    assert "4 calibration images" in captured.out
    assert "hailo_compile.sh needs at least 1024" in captured.err  # 4 is not a usable count


def test_cli_says_when_the_zoo_will_ignore_files(tmp_path, capsys):
    a, b = _src(tmp_path, "v1", 1100), _src(tmp_path, "v2", 1100)

    main(["--out", str(tmp_path / "calib"), "--from", str(a), "--from", str(b), "--count", "2048"])

    err = capsys.readouterr().err
    assert "the zoo reads the first 1024 sorted files ([512, 512] per source)" in err
    assert "the rest are never used" in err


def test_cli_needs_out_unless_verifying(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--from", str(_src(tmp_path, "v1", 3))])

    assert exc.value.code == 2 and "--out is required" in capsys.readouterr().err


# ---- verify -----------------------------------------------------------------------------------


def test_verify_accepts_what_build_made(tmp_path, capsys):
    a, b = _src(tmp_path, "v1", 40), _src(tmp_path, "v2", 40)
    build(tmp_path / "calib", [a, b], 30, seed=1)

    found = verify(tmp_path / "calib", calibset_size=24)

    assert found == {"files": 30, "zoo_prefix": 24, "per_source": [12, 12]}
    assert main(["--verify", str(tmp_path / "calib"), "--calibset-size", "24"]) == 0
    assert "calibration folder ok: 30 images" in capsys.readouterr().out


def test_verify_refuses_a_folder_whose_prefix_misses_a_source(tmp_path):
    # The old naming: source 0 first. 60 files, the zoo (here) reads 24: all source 0.
    folder = tmp_path / "calib"
    folder.mkdir()
    files = []
    for k in (0, 1):
        for i in range(30):
            name = f"c{k}_{i:06d}.jpg"
            (folder / name).write_bytes(b"x")
            files.append({"name": name})
    (folder / "calib_manifest.json").write_text(
        json.dumps({"schema": "calib-set/1", "sources": ["a", "b"], "files": files})
    )

    with pytest.raises(SystemExit, match=r"first 24 of 60 sorted files .* \[24, 0\]"):
        verify(folder, calibset_size=24)
    assert verify(folder, calibset_size=60)["per_source"] == [30, 30]  # read whole: balanced


def test_verify_accepts_an_old_folder_that_the_zoo_reads_whole(tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    files = []
    for k in (0, 1):
        for i in range(12):
            name = f"c{k}_{i:06d}.jpg"
            (folder / name).write_bytes(b"x")
            files.append({"name": name})
    (folder / "calib_manifest.json").write_text(
        json.dumps({"schema": "calib-set/1", "sources": ["a", "b"], "files": files})
    )

    assert verify(folder, calibset_size=24)["per_source"] == [12, 12]


def test_verify_notices_files_added_or_removed_after_the_build(tmp_path):
    a = _src(tmp_path, "v1", 20)
    build(tmp_path / "calib", [a], 10, seed=1)
    (tmp_path / "calib" / "99999_c0_added.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit, match=r"1 not in the manifest"):
        verify(tmp_path / "calib")
    (tmp_path / "calib" / "99999_c0_added.jpg").unlink()
    _images(tmp_path / "calib")[0].unlink()
    with pytest.raises(SystemExit, match=r"1 missing"):
        verify(tmp_path / "calib")


def test_verify_needs_a_readable_manifest(tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    with pytest.raises(SystemExit, match="no calib_manifest.json"):
        verify(folder)
    (folder / "calib_manifest.json").write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable manifest"):
        verify(folder)
    (folder / "calib_manifest.json").write_text(json.dumps({"schema": "other/9"}))
    with pytest.raises(SystemExit, match="unreadable manifest"):  # no 'files'
        verify(folder)
    (folder / "calib_manifest.json").write_text(
        json.dumps({"schema": "other/9", "files": [], "sources": ["a"]})
    )
    with pytest.raises(SystemExit, match="schema 'other/9'"):
        verify(folder)


def test_verify_refuses_a_name_that_does_not_say_which_source(tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    (folder / "mystery.jpg").write_bytes(b"x")
    (folder / "calib_manifest.json").write_text(
        json.dumps({"schema": "calib-set/2", "sources": ["a"], "files": [{"name": "mystery.jpg"}]})
    )

    with pytest.raises(SystemExit, match="does not say which of the 1 sources"):
        verify(folder)
