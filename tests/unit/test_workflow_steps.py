"""CI workflow steps that must really fail (audit review T-02, TOOL-07, TOOL-12).

GitHub runs a step's ``run:`` body with ``bash -e {0}``. Under ``-e`` a negated command
(``! grep ...``) cannot fail the script unless it is the very last command, so a ban written that
way gates only if it happens to come last. ``hardware-guard.yml`` had that defect (T-02, fixed and
tested in ``test_hardware_guard_workflow.py``); ``deps-lock.yml`` had it too (TOOL-07): a
``pywin32`` in ``pyproject.toml`` passed. The steps are run here exactly as GitHub runs them.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
PYPROJECT = (REPO / "pyproject.toml").read_text()

WINDOWS_STEP = "Ban Windows/x86-only deps"
CORAL_STEP = "Ban Coral pkgs in main env"


def _steps(workflow: str) -> dict[str, str]:
    """Step name -> ``run`` body, for every step of every job of ``workflow``."""
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return {
        step["name"]: step["run"]
        for job in doc["jobs"].values()
        for step in job["steps"]
        if "run" in step and "name" in step
    }


def run_step(tmp_path: Path, name: str, pyproject: str) -> subprocess.CompletedProcess:
    """One deps-lock step on a crafted ``pyproject.toml``, as ``bash -e {0}`` runs it."""
    (tmp_path / "pyproject.toml").write_text(pyproject)
    script = tmp_path / "step.sh"
    script.write_text(_steps("deps-lock.yml")[name])
    return subprocess.run(
        ["bash", "-e", str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )


CLEAN = '[project]\nname = "x"\ndependencies = ["fastapi>=0.110"]\n'


def test_the_shipped_pyproject_passes_both_bans(tmp_path):
    for step in (WINDOWS_STEP, CORAL_STEP):
        assert run_step(tmp_path, step, PYPROJECT).returncode == 0, step


@pytest.mark.parametrize(
    "offender",
    [
        '"pywin32>=300"',
        "# win32 only",
        "# built with msvc",
        '"PyWin32"',
        "\"colorama; sys_platform == 'WIN32'\"",
    ],
)
def test_a_windows_dependency_fails_the_first_pattern(tmp_path, offender):
    # `! grep '(pywin32|win32|msvc)'` was the first line of its step: under bash -e it never
    # failed, and only the second line (win_amd64|x86_64) gated.
    done = run_step(tmp_path, WINDOWS_STEP, CLEAN + f"extra = [{offender}]\n")

    assert done.returncode == 1
    assert "banned pattern 'pywin32|win32|msvc'" in done.stderr


@pytest.mark.parametrize(
    "offender", ['"torch-2.0-cp311-win_amd64.whl"', "# x86_64 wheel", "X86_64"]
)
def test_an_x86_only_reference_fails_the_second_pattern(tmp_path, offender):
    done = run_step(tmp_path, WINDOWS_STEP, CLEAN + f"extra = [{offender}]\n")

    assert done.returncode == 1
    assert "banned pattern 'win_amd64|x86_64'" in done.stderr


def test_the_first_pattern_is_not_masked_by_a_clean_second_one(tmp_path):
    # The shape of the old defect: only the first pattern matches, the last line is clean.
    done = run_step(tmp_path, WINDOWS_STEP, CLEAN + "# pywin32 would break the Pi\n")

    assert done.returncode == 1


def test_a_clean_pyproject_passes(tmp_path):
    assert run_step(tmp_path, WINDOWS_STEP, CLEAN).returncode == 0


@pytest.mark.parametrize(
    "dependency", ["pycoral>=2.0", "edgetpu-runtime", "EdgeTPU", "pycoral", "tflite; pycoral"]
)
def test_a_coral_package_fails_its_step(tmp_path, dependency):
    done = run_step(tmp_path, CORAL_STEP, CLEAN + f'extra = ["{dependency}"]\n')

    assert done.returncode == 1
    assert "pycoral/edgetpu must not be in the main environment" in done.stderr


def test_words_that_merely_contain_coral_are_not_banned(tmp_path):
    assert run_step(tmp_path, CORAL_STEP, CLEAN + "# mypycoralish\n").returncode == 0


# ---- the same defect anywhere else ---------------------------------------------------------------


def _commands(script: str) -> list[str]:
    """The non-blank, non-comment lines of a step script."""
    return [
        ln.strip() for ln in script.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]


def test_no_step_of_any_workflow_hides_a_failure_behind_a_bare_negation():
    # `! cmd` as a line of its own only gates when it is the last command of the step.
    offenders = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        for name, script in _steps(workflow.name).items():
            lines = _commands(script)
            offenders += [
                f"{workflow.name}: {name}: {ln}" for ln in lines[:-1] if re.match(r"!\s", ln)
            ]
    assert offenders == []


def test_the_detector_of_that_defect_sees_the_old_deps_lock_step():
    old = (
        "! grep -Ei '(pywin32|win32|msvc)' pyproject.toml\n"
        "! grep -Ei '(win_amd64|x86_64)'   pyproject.toml\n"
    )

    assert [ln for ln in _commands(old)[:-1] if re.match(r"!\s", ln)] == [_commands(old)[0]]


# ---- TOOL-12: the frontend unit tests run in CI --------------------------------------------------


def test_the_webui_workflow_runs_the_frontend_unit_tests_before_building():
    doc = yaml.safe_load((WORKFLOWS / "webui-build.yml").read_text())
    steps = doc["jobs"]["build-ui"]["steps"]
    runs = [s.get("run") for s in steps]

    assert "npm test" in runs
    unit = steps[runs.index("npm test")]
    assert unit["working-directory"] == "frontend"
    assert "continue-on-error" not in unit  # a failing spec must fail the job
    assert runs.index("npm ci") < runs.index("npm test") < runs.index("npm run build")
    # `npm test` is vitest in one pass, not a watcher that would hang CI
    scripts = json.loads((REPO / "frontend" / "package.json").read_text())["scripts"]
    assert scripts["test"].split()[:2] == ["vitest", "run"]
