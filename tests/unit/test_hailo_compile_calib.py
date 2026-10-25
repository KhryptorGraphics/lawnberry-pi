"""``scripts/hailo_compile.sh`` checks the calibration folder for what it relies on.

The zoo reads the first 1024 files of the folder in sorted order and nothing else; the script used
to check only that there were at least 1024 files (audit review TOOL-04), so a folder whose first
1024 files all came from one dataset compiled without a word. The script is run for real (bash),
with the DFC wheel's checksum faked and the workshop tools it would call next absent, so what is
observed is the calibration step and whether the script stops there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from workshop.make_calib import build

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "hailo_compile.sh"
TAG = "t"


@pytest.fixture
def box(tmp_path):
    """A fake workshop: $LB and $HOME in tmp, the wheel and ONNX present, sha256sum always ok."""
    lb, home, fake = tmp_path / "lb", tmp_path / "home", tmp_path / "fakebin"
    (home / "Downloads").mkdir(parents=True)
    (home / "Downloads" / "hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl").write_bytes(
        b""
    )
    onnx = lb / "data" / "models" / f"yolo26s_{TAG}" / "weights" / "best.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"")
    fake.mkdir()
    # Symlinks to existing programs, never a script written and run at once: exec of a file that
    # was just written fails with "Text file busy" when another thread of this process forks in
    # between (seen once in a shuffled full run, where it read as a sha256 mismatch).
    (fake / "sha256sum").symlink_to(shutil.which("true"))  # `sha256sum -c` always succeeds
    (fake / "python3").symlink_to(sys.executable)
    return {"lb": lb, "home": home, "fake": fake}


def run_script(box, calib: Path, *flags: str):
    env = {
        **os.environ,
        "LB": str(box["lb"]),
        "HOME": str(box["home"]),
        "PATH": f"{box['fake']}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        ["bash", str(SCRIPT), "--tag", TAG, "--calib", str(calib), *flags],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _sources(tmp_path: Path, per_source: int) -> list[Path]:
    out = []
    for name in ("v1", "v2"):
        d = tmp_path / name / "images"
        d.mkdir(parents=True)
        for i in range(per_source):
            (d / f"{i:06d}.jpg").write_bytes(f"{name}-{i}".encode())
        out.append(d)
    return out


def _legacy_folder(folder: Path, per_source: int) -> Path:
    """What make_calib used to write: c0_ first, then c1_, plus a manifest."""
    folder.mkdir()
    files = []
    for k in (0, 1):
        for i in range(per_source):
            name = f"c{k}_{i:06d}.jpg"
            (folder / name).write_bytes(b"x")
            files.append({"name": name})
    (folder / "calib_manifest.json").write_text(
        json.dumps({"schema": "calib-set/1", "sources": ["a", "b"], "files": files})
    )
    return folder


def stopped_at_the_calibration_step(done) -> bool:
    return "FATAL: calibration folder" in done.stderr and "export_detector check" not in done.stdout


def test_a_folder_whose_first_1024_files_are_one_dataset_stops_the_compile(box, tmp_path):
    folder = _legacy_folder(tmp_path / "calib", 600)  # 1200 files, the zoo reads 600 + 424

    done = run_script(box, folder)

    assert done.returncode == 1
    assert stopped_at_the_calibration_step(done), done.stdout + done.stderr
    assert "[600, 424]" in done.stderr and "rebuild the folder with make_calib" in done.stderr


def test_a_balanced_folder_past_the_zoo_prefix_passes_the_calibration_step(box, tmp_path):
    a, b = _sources(tmp_path, 560)
    build(tmp_path / "calib", [a, b], 1100, seed=1)  # 1100 files: the zoo reads 1024 of them

    done = run_script(box, tmp_path / "calib")

    assert "calibration images: 1100" in done.stdout
    assert "calibration folder ok: 1100 images, the zoo reads the first 1024 sorted" in done.stdout
    assert "per source [512, 512]" in done.stdout
    assert "FATAL: calibration" not in done.stderr


def test_a_file_added_after_the_manifest_stops_the_compile(box, tmp_path):
    a, b = _sources(tmp_path, 560)
    build(tmp_path / "calib", [a, b], 1100, seed=1)
    (tmp_path / "calib" / "00000_c0_extra.jpg").write_bytes(b"x")

    done = run_script(box, tmp_path / "calib")

    assert done.returncode == 1 and stopped_at_the_calibration_step(done)
    assert "1 not in the manifest" in done.stderr


def test_too_few_images_still_stop_the_compile(box, tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    for i in range(10):
        (folder / f"{i}.jpg").write_bytes(b"x")

    done = run_script(box, folder)

    assert done.returncode == 1
    assert "need >= 1024 calibration images, found 10" in done.stderr


def test_a_folder_without_a_manifest_past_the_prefix_is_warned_about(box, tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    for i in range(1100):
        (folder / f"{i:06d}.jpg").write_bytes(b"x")

    done = run_script(box, folder)

    assert "WARN: the zoo calibrates on the first 1024 of the 1100 sorted files" in done.stderr
    assert "FATAL: calibration" not in done.stderr  # a single dataset folder is allowed


def test_exactly_1024_files_need_no_warning(box, tmp_path):
    folder = tmp_path / "calib"
    folder.mkdir()
    for i in range(1024):
        (folder / f"{i:06d}.jpg").write_bytes(b"x")

    done = run_script(box, folder)

    assert "calibration images: 1024" in done.stdout
    assert "WARN: the zoo calibrates" not in done.stderr and "FATAL: calibration" not in done.stderr


def test_a_dry_run_reports_a_bad_folder_as_a_warning_and_still_exits_0(box, tmp_path):
    folder = _legacy_folder(tmp_path / "calib", 600)

    done = run_script(box, folder, "--dry-run")

    assert done.returncode == 0, done.stderr
    assert "WARN: calibration folder:" in done.stderr and "[600, 424]" in done.stderr
