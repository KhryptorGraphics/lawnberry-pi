"""check_docs_drift.sh must really diff (the old `fetch --depth=0` error left it vacuous)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docs_drift.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _setup(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(origin), str(work))
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        _git(work, "config", k, v)
    (work / "backend").mkdir()
    (work / "backend" / "app.py").write_text("x = 1\n")
    (work / "docs").mkdir()
    (work / "docs" / "a.md").write_text("a\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "base")
    _git(work, "branch", "-M", "main")
    _git(work, "push", "-u", "origin", "main")
    _git(work, "checkout", "-b", "feature")
    return work


def _run(work: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "BASE_REF": "origin/main", "HEAD_REF": "HEAD"}
    return subprocess.run(["bash", str(SCRIPT)], cwd=work, env=env, capture_output=True, text=True)


def test_script_has_no_depth_zero_fetch():
    assert "--depth=0" not in SCRIPT.read_text()


def test_code_change_without_docs_fails(tmp_path):
    work = _setup(tmp_path)
    (work / "backend" / "app.py").write_text("x = 2\n")
    _git(work, "commit", "-am", "code only")
    result = _run(work)
    assert result.returncode == 1, result.stderr
    assert "Docs drift detected" in result.stderr
    assert "depth 0" not in result.stderr


def test_code_and_docs_change_passes(tmp_path):
    work = _setup(tmp_path)
    (work / "backend" / "app.py").write_text("x = 2\n")
    (work / "docs" / "a.md").write_text("b\n")
    _git(work, "commit", "-am", "code and docs")
    result = _run(work)
    assert result.returncode == 0, result.stderr
    assert "code_changed=true, docs_changed=true" in result.stderr


def test_shallow_clone_is_deepened_not_vacuous(tmp_path):
    work = _setup(tmp_path)
    (work / "backend" / "app.py").write_text("x = 3\n")
    _git(work, "commit", "-am", "code only")
    _git(work, "push", "origin", "feature")
    shallow = tmp_path / "shallow"
    _git(
        tmp_path,
        "clone",
        "--depth=1",
        "--branch",
        "feature",
        f"file://{tmp_path / 'origin.git'}",
        str(shallow),
    )
    _git(shallow, "fetch", "--depth=1", "origin", "main:refs/remotes/origin/main")
    result = _run(shallow)
    assert result.returncode == 1, result.stderr
    assert "Docs drift detected" in result.stderr


def test_script_is_executable_bash():
    assert shutil.which("bash")
    assert SCRIPT.read_text().startswith("#!/usr/bin/env bash")
