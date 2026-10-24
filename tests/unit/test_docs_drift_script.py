"""check_docs_drift.sh must really diff, and must never pass when it cannot.

Every scenario builds throw-away git repositories. The script's exit codes under test:
0 = diff computed and clean/empty, 1 = docs drift, 2 = the diff could not be computed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_docs_drift.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _git_out(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _identity(work: Path) -> None:
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        _git(work, "config", k, v)


def _setup(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(origin), str(work))
    _identity(work)
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


def _run(
    work: Path, *, base_ref: str | None = "origin/main", **extra_env: str
) -> subprocess.CompletedProcess[str]:
    # Scrub the CI variables the script reads: the suite itself may run inside GitHub Actions.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GITHUB_") and k not in ("BASE_REF", "HEAD_REF")
    }
    env["HEAD_REF"] = "HEAD"
    if base_ref is not None:
        env["BASE_REF"] = base_ref
    env.update(extra_env)
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
    """A depth-1 checkout (what actions/checkout makes) has neither origin/main nor the history
    that links it to HEAD. Main also moved on after the branch forked (a docs-only commit), so a
    tip-to-tip diff would see that docs file as "changed" and wrongly pass. Only deepening the
    clone yields the merge-base and the right verdict."""
    work = _setup(tmp_path)
    # main moves on after the fork: a docs-only commit the feature branch does not have
    _git(work, "checkout", "main")
    (work / "docs" / "main_only.md").write_text("main moved on\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "main: docs only")
    _git(work, "push", "origin", "main")
    # the feature branch forked earlier and only changes code
    _git(work, "checkout", "feature")
    (work / "backend" / "app.py").write_text("x = 3\n")
    _git(work, "commit", "-am", "feature: code only")
    _git(work, "push", "origin", "feature")

    shallow = tmp_path / "shallow"
    shallow.mkdir()
    _git(shallow, "init", "-b", "feature")
    _git(shallow, "remote", "add", "origin", f"file://{tmp_path / 'origin.git'}")
    _git(shallow, "fetch", "--depth=1", "origin", "feature")
    _git(shallow, "checkout", "FETCH_HEAD")
    assert _git_out(shallow, "rev-parse", "--is-shallow-repository") == "true"
    assert subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main"], cwd=shallow
    ).returncode, "the scenario must start without origin/main"

    result = _run(shallow)
    assert result.returncode == 1, result.stderr
    assert "Docs drift detected" in result.stderr
    assert "backend/app.py" in result.stderr
    assert "docs/main_only.md" not in result.stderr  # the merge-base diff, not tip-to-tip
    assert "Shallow repository detected" in result.stderr  # the scenario took the deepen path


def test_unresolvable_base_is_a_hard_failure(tmp_path):
    """No origin/main anywhere: the old script printed "No changed files" and exited 0."""
    work = tmp_path / "lonely"
    work.mkdir()
    _git(work, "init", "-b", "feature")
    _identity(work)
    (work / "backend").mkdir()
    (work / "backend" / "app.py").write_text("x = 1\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "base")
    (work / "backend" / "app.py").write_text("x = 2\n")
    _git(work, "commit", "-am", "code only")
    result = _run(work)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "does not resolve" in result.stderr
    assert "No changed files" not in result.stdout + result.stderr


def test_a_base_that_only_exists_on_origin_is_fetched(tmp_path):
    """origin/main is not in the local clone (single-branch or shallow checkout) but is on the
    remote: the script must fetch it by an explicit refspec. The old `git fetch origin origin/main`
    named a ref the remote does not have and always failed silently."""
    work = _setup(tmp_path)
    (work / "backend" / "app.py").write_text("x = 2\n")
    _git(work, "commit", "-am", "code only")
    _git(work, "update-ref", "-d", "refs/remotes/origin/main")
    assert subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main"], cwd=work
    ).returncode
    result = _run(work)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Docs drift detected" in result.stderr


def _push_code_only_commit_to_main(work: Path) -> str:
    """A code-only commit pushed straight to main; returns the pre-push commit."""
    _git(work, "checkout", "main")
    before = _git_out(work, "rev-parse", "HEAD")
    (work / "backend" / "app.py").write_text("x = 2\n")
    _git(work, "commit", "-am", "code only, straight to main")
    _git(work, "push", "origin", "main")
    return before


def test_push_to_main_compares_the_pushed_commits(tmp_path):
    work = _setup(tmp_path)
    before = _push_code_only_commit_to_main(work)
    # On a push to main the checkout has origin/main == HEAD: the pre-push commit is the base.
    result = _run(work, base_ref=before, GITHUB_EVENT_NAME="push")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Docs drift detected" in result.stderr


def test_push_event_without_a_pre_push_base_is_an_error_not_a_pass(tmp_path):
    """The default base on a push to main is origin/main == HEAD: an empty range that used to
    exit 0. A push event with nothing to compare must fail, naming the fix."""
    work = _setup(tmp_path)
    _push_code_only_commit_to_main(work)
    result = _run(work, base_ref=None, GITHUB_EVENT_NAME="push")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "github.event.before" in result.stderr
    assert "No changed files" not in result.stdout + result.stderr


def test_head_equal_to_base_outside_a_push_is_a_clean_no_op(tmp_path):
    work = _setup(tmp_path)
    result = _run(work, base_ref="origin/main")
    assert result.returncode == 0, result.stderr
    assert "No changed files detected" in result.stdout


def test_non_ascii_paths_are_classified(tmp_path):
    """git quotes non-ASCII paths ("backend/\\303\\251.py") unless -z is used, so they matched
    neither regex and a code-only change passed."""
    work = _setup(tmp_path)
    (work / "backend" / "é.py").write_text("x = 1\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "non-ascii code only")
    result = _run(work)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "code_changed=true, docs_changed=false" in result.stderr

    (work / "docs" / "é.md").write_text("é\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "non-ascii docs")
    result = _run(work)
    assert result.returncode == 0, result.stderr
    assert "code_changed=true, docs_changed=true" in result.stderr


def test_moving_code_out_of_backend_is_a_code_change(tmp_path):
    """With rename detection `--name-only` prints only the destination, so a move out of backend/
    to an unwatched path looked like no change at all."""
    work = _setup(tmp_path)
    (work / "elsewhere").mkdir()
    _git(work, "mv", "backend/app.py", "elsewhere/app.py")
    _git(work, "commit", "-m", "move code out of backend")
    result = _run(work)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "backend/app.py" in result.stderr


def test_unrelated_histories_are_diffed_tip_to_tip_with_a_warning(tmp_path):
    work = _setup(tmp_path)
    _git(work, "checkout", "--orphan", "unrelated")
    (work / "backend" / "app.py").write_text("x = 9\n")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "orphan root, code differs, docs identical")
    result = _run(work)
    assert "no merge-base" in result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    assert "backend/app.py" in result.stderr
    assert "docs/a.md" not in result.stderr


def test_script_is_executable_bash():
    assert shutil.which("bash")
    assert SCRIPT.read_text().startswith("#!/usr/bin/env bash")
