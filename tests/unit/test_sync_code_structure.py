"""``scripts/sync_code_structure.py`` keeps the structure overview in step with the source.

Every case runs the real CLI against a throwaway git repository: a tiny overview, a few sources,
one commit on ``main``, then the edit under test in the working tree.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_code_structure as scs  # noqa: E402

HEADER = "| Path | Purpose | Subsystem | Callable interfaces |\n|---|---|---|---|\n"
DOC = (
    "# Overview\n\n"
    "## Backend services (FastAPI/domain services)\n\n" + HEADER + "{backend}\n"
    "## Navigation and planning algorithms\n\n" + HEADER + "{nav}\n"
    "## Operations and maintenance scripts\n\n" + HEADER + "{ops}\n"
    "## Notes and scope\n\nthe end\n"
)
SERVICE = '''"""Runs the service."""


class Service:
    def go(self, x, y=1, *, z=None):
        return x

    @property
    def ready(self):
        return True

    def _private(self):
        return 0


def main(argv=None):
    return 0
'''
SERVICE_CELL = "Class `Service`: `go(x, y=1, *, z=None)`, `ready`; `main(argv=None)`"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text) if not text.startswith('"""') else text)


def _doc(repo: Path, *, backend: str = "", nav: str = "", ops: str = "") -> None:
    _write(repo, "docs/code_structure_overview.md", DOC.format(backend=backend, nav=nav, ops=ops))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, "backend/src/services/svc.py", SERVICE)
    _doc(
        tmp_path,
        backend=f"| `backend/src/services/svc.py` | Runs the service. | Backend | {SERVICE_CELL} |",
    )
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def run(repo: Path, *args: str, base: str = "main") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_code_structure.py"), "--root", str(repo)]
        + ["--base", base, *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def doc_text(repo: Path) -> str:
    return (repo / "docs" / "code_structure_overview.md").read_text()


# ------------------------------------------------------------------- the generated cell


def test_the_interface_cell_lists_public_callables_with_real_arguments_and_defaults():
    cell = scs.interfaces(ast.parse(SERVICE))

    assert cell == SERVICE_CELL


def test_a_default_with_a_pipe_is_escaped_so_the_table_row_keeps_four_cells():
    src = "def pick(mode=1 | 2, flag=None):\n    return mode\n"

    cell = scs.interfaces(ast.parse(src))

    assert cell == "`pick(mode=1 \\| 2, flag=None)`"
    assert len(scs.read_rows([f"| `a.py` | p | s | {cell} |"])[0].cells) == 4


def test_a_file_with_nothing_public_has_no_cell():
    assert scs.interfaces(ast.parse("def _a():\n    pass\n\nclass _B:\n    pass\n")) == ""


# ------------------------------------------------------------------------ check and fix


def test_a_current_overview_passes(repo):
    done = run(repo, "--check")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "current" in done.stdout


def test_a_changed_signature_is_drift_and_fix_regenerates_only_the_interface_cell(repo):
    _write(
        repo,
        "backend/src/services/svc.py",
        SERVICE.replace("def go(self, x,", "def go(self, w, x,"),
    )

    checked = run(repo, "--check")
    assert checked.returncode == 1
    assert "backend/src/services/svc.py: the interface list differs" in checked.stdout

    fixed = run(repo, "--fix")
    assert fixed.returncode == 0, fixed.stderr
    text = doc_text(repo)
    assert "`go(w, x, y=1, *, z=None)`" in text
    assert "| Runs the service. | Backend |" in text, "purpose and subsystem are kept"
    assert run(repo, "--check").returncode == 0


def test_a_new_file_gets_a_row_in_its_section_with_a_two_sentence_docstring_purpose(repo):
    _write(
        repo,
        "backend/src/nav/sweep.py",
        '"""Plans a sweep. It covers the lawn. A third sentence is not kept."""\n\n\n'
        "def plan(boundary, *, step=0.3):\n    return boundary\n",
    )

    assert run(repo, "--check").returncode == 1
    assert run(repo, "--fix").returncode == 0

    lines = doc_text(repo).splitlines()
    row = next(i for i, ln in enumerate(lines) if "`backend/src/nav/sweep.py`" in ln)
    nav = lines.index("## Navigation and planning algorithms")
    notes = lines.index("## Operations and maintenance scripts")
    assert nav < row < notes, "the row is in the navigation table"
    assert (
        "| Plans a sweep. It covers the lawn. | Navigation | `plan(boundary, *, step=0.3)` |"
        in (lines[row])
    )
    assert "third sentence" not in lines[row]
    assert run(repo, "--check").returncode == 0


def test_a_new_file_without_a_docstring_cannot_be_given_a_row(repo):
    _write(repo, "scripts/tool.py", "def run(arg):\n    return arg\n")

    fixed = run(repo, "--fix")

    assert fixed.returncode == 2
    assert "scripts/tool.py: add a module docstring" in fixed.stderr
    assert "scripts/tool.py" not in doc_text(repo), "nothing half-written"


def test_an_untouched_row_whose_signature_is_stale_is_caught_and_repaired(repo):
    _write(repo, "scripts/old.py", '"""Old."""\n\n\ndef keep(a):\n    return a\n')
    _doc(
        repo,
        backend=f"| `backend/src/services/svc.py` | Runs the service. | Backend | {SERVICE_CELL} |",
        ops="| `scripts/old.py` | Old. | Ops | `keep(a, b)` |",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")  # nothing is touched relative to HEAD ... base moves

    checked = run(repo, "--check", base="HEAD")
    assert checked.returncode == 1
    assert "scripts/old.py: `keep(a, b)` does not match the source" in checked.stdout

    assert run(repo, "--fix", base="HEAD").returncode == 0
    assert "| Old. | Ops | `keep(a)` |" in doc_text(repo)
    assert run(repo, "--check", base="HEAD").returncode == 0


def test_a_summary_with_a_bare_call_is_not_called_stale(repo):
    _write(repo, "scripts/old.py", '"""Old."""\n\n\ndef keep(a, b):\n    return a\n')
    _doc(
        repo,
        backend=f"| `backend/src/services/svc.py` | Runs the service. | Backend | {SERVICE_CELL} |",
        ops="| `scripts/old.py` | Old. | Ops | `keep(...)`, `keep()` |",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")

    assert run(repo, "--check", base="HEAD").returncode == 0


def test_a_row_of_a_deleted_file_is_dropped_and_duplicate_rows_are_merged(repo):
    _write(repo, "scripts/gone.py", '"""Gone."""\n\n\ndef x():\n    return 1\n')
    _doc(
        repo,
        backend=(
            f"| `backend/src/services/svc.py` | Runs the service. | Backend | {SERVICE_CELL} |\n"
            f"| `backend/src/services/svc.py` | Also keeps a latch. | Backend | {SERVICE_CELL} |"
        ),
        ops="| `scripts/gone.py` | Gone. | Ops | `x()` |",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    (repo / "scripts" / "gone.py").unlink()

    checked = run(repo, "--check", base="HEAD")
    assert checked.returncode == 1
    assert "scripts/gone.py: the file is gone" in checked.stdout
    assert "backend/src/services/svc.py: 2 rows" in checked.stdout

    assert run(repo, "--fix", base="HEAD").returncode == 0
    text = doc_text(repo)
    assert "scripts/gone.py" not in text
    assert text.count("`backend/src/services/svc.py`") == 1
    assert "Runs the service." in text and "Also keeps a latch." in text
    assert run(repo, "--check", base="HEAD").returncode == 0


def test_all_takes_untouched_files_too(repo):
    _write(repo, "scripts/quiet.py", '"""Quiet."""\n\n\ndef hush():\n    return 1\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")

    assert run(repo, "--check", base="HEAD").returncode == 0, "untouched: not looked at"
    every = run(repo, "--check", "--all", base="HEAD")
    assert every.returncode == 1 and "scripts/quiet.py: new file" in every.stdout


def test_an_unresolvable_base_is_an_unknown_comparison_not_a_pass(repo):
    done = run(repo, "--check", base="no-such-ref")

    assert done.returncode == 2
    assert "cannot compare" in done.stderr
