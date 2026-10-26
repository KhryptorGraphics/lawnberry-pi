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


# ---- the cell the tool generated is never drift (audit TOOL3-06) ---------------------------------


def _doc_row(
    path: str, cell: str, *, purpose: str = "Does a thing.", subsystem: str = "Ops"
) -> str:
    return f"| `{path}` | {purpose} | {subsystem} | {cell} |"


def _service_row() -> str:
    return f"| `backend/src/services/svc.py` | Runs the service. | Backend | {SERVICE_CELL} |"


@pytest.mark.parametrize(
    "signature",
    [
        'pick(sep=",", n=1)',
        'pick(open="(", n=1)',
        'pick(close=")", n=1)',
        "pick(key=lambda a, b: a, n=1)",
        "pick(a=(1, 2), b=[3, 4], c={5: 6})",
        'pick(label="a|b", n=1)',
        'pick(table={"x": 1, "y": 2}, n=1)',
        "pick(x: dict[str, int] = {}, *, y=lambda: (1, 2))",
        'pick(*args, sep=", ", **kwargs)',
    ],
)
def test_a_row_the_tool_generated_for_a_string_or_lambda_default_is_not_drift(repo, signature):
    """The comma splitter cut ``sep=','`` and ``lambda a, b: a`` into pieces no source has."""
    _write(repo, "scripts/pick.py", f'"""Picks."""\n\n\ndef {signature}:\n    return 1\n')
    cell = scs.interfaces(ast.parse((repo / "scripts" / "pick.py").read_text()))
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/pick.py", cell))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a row for pick.py, which is then left alone")

    checked = run(repo, "--check", base="HEAD")

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert run(repo, "--fix", base="HEAD").returncode == 0
    assert run(repo, "--check", base="HEAD").returncode == 0, "--fix cannot clear what is not drift"


def test_a_real_change_next_to_a_string_default_is_still_drift_and_fix_clears_it(repo):
    _write(repo, "scripts/pick.py", '"""Picks."""\n\n\ndef pick(sep=",", count=1):\n    return 1\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/pick.py", "`pick(sep=',', n=1)`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "n was renamed to count")

    checked = run(repo, "--check", base="HEAD")

    assert checked.returncode == 1
    assert "scripts/pick.py: `pick(sep=',', n=1)` does not match the source" in checked.stdout
    assert run(repo, "--fix", base="HEAD").returncode == 0
    assert "`pick(sep=',', count=1)`" in doc_text(repo)
    assert run(repo, "--check", base="HEAD").returncode == 0


@pytest.mark.parametrize(
    ("listed", "source_args", "stale"),
    [
        ("a, b=(1, 2), ...", "a, b=(1, 2)", False),
        ("a, b={1: 2, 3: 4}, …", "a, b={1: 2, 3: 4}", False),
        ("a, ...", "a, c", True),
        ("a, b=(1, 2), ...", "a, c", True),
    ],
)
def test_a_summary_that_is_not_valid_python_is_still_compared_by_argument_name(
    repo, listed, source_args, stale
):
    """``f(a, ...)`` is a hand-written summary: the names it does list must be the source's."""
    _write(repo, "scripts/keep.py", f'"""Keeps."""\n\n\ndef keep({source_args}):\n    return 1\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/keep.py", f"`keep({listed})`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "summary row")

    checked = run(repo, "--check", base="HEAD")

    assert (checked.returncode == 1) is stale, checked.stdout + checked.stderr


def test_a_class_name_used_like_a_call_is_not_a_missing_function(repo):
    _write(
        repo,
        "scripts/widget.py",
        '"""W."""\n\n\nclass Widget:\n    def __init__(self, a):\n        pass\n',
    )
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/widget.py", "Class `Widget(a, b)`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")

    assert run(repo, "--check", base="HEAD").returncode == 0


def test_a_row_that_names_a_function_the_file_does_not_define_is_drift(repo):
    _write(repo, "scripts/real.py", '"""R."""\n\n\ndef real(a):\n    return a\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/real.py", "`real(a)`, `ghost(x)`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")

    checked = run(repo, "--check", base="HEAD")

    assert checked.returncode == 1 and "scripts/real.py: `ghost` is not defined" in checked.stdout


# ---- --all is a report, and what the check does not cover (audit TOOL3-07) ----------------------


def test_all_reports_a_hand_written_summary_that_the_default_check_accepts(repo):
    _write(repo, "scripts/old.py", '"""Old."""\n\n\ndef keep(a, b):\n    return a\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/old.py", "`keep(...)`, `keep()`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "summary row")

    assert run(repo, "--check", base="HEAD").returncode == 0
    every = run(repo, "--check", "--all", base="HEAD")
    assert every.returncode == 1
    assert "scripts/old.py: the interface list differs from the source" in every.stdout


def test_all_takes_an_untracked_file_too(repo):
    _write(repo, "scripts/fresh.py", '"""Fresh."""\n\n\ndef hush():\n    return 1\n')

    every = run(repo, "--check", "--all", base="HEAD")

    assert every.returncode == 1 and "scripts/fresh.py: new file" in every.stdout


def test_fix_all_names_every_file_without_a_docstring_and_writes_nothing(repo):
    _write(repo, "scripts/a_bare.py", "def one():\n    return 1\n")
    _write(repo, "backend/src/b_bare.py", "def two():\n    return 2\n")
    _write(repo, "scripts/c_documented.py", '"""Has one."""\n\n\ndef three():\n    return 3\n')
    before = doc_text(repo)

    fixed = run(repo, "--fix", "--all", base="HEAD")

    assert fixed.returncode == 2
    assert "scripts/a_bare.py: add a module docstring" in fixed.stderr
    assert "backend/src/b_bare.py: add a module docstring" in fixed.stderr
    assert "c_documented" not in fixed.stderr
    assert doc_text(repo) == before, "nothing half-written"


def test_a_row_of_a_file_that_is_not_python_is_checked_for_existing_and_nothing_else(repo):
    _write(repo, "frontend/src/api.ts", "export function login(user: string) {}\n")
    _write(repo, "scripts/run.sh", "run() { :; }\n")
    _doc(
        repo,
        backend=_service_row()
        + "\n"
        + _doc_row("frontend/src/api.ts", "Exported functions from file (see source).")
        + "\n"
        + _doc_row("frontend/src/composables/useGone.ts", "`useGone()`")
        + "\n"
        + _doc_row("frontend/src/composables/useGone2.ts", "`useGone2()`"),
        ops=_doc_row("scripts/run.sh", "Shell functions: `stale_name`."),
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rows of other languages")
    (repo / "frontend" / "src" / "api.ts").write_text("export function logout() {}\n")

    checked = run(repo, "--check", base="HEAD")

    assert checked.returncode == 1
    assert "frontend/src/composables/useGone.ts: the file is gone" in checked.stdout
    assert "frontend/src/composables/useGone2.ts: the file is gone" in checked.stdout
    assert "api.ts" not in checked.stdout and "run.sh" not in checked.stdout, "unchecked by design"
    assert run(repo, "--fix", base="HEAD").returncode == 0
    text = doc_text(repo)
    assert "useGone" not in text, "the rows of the files that are gone are dropped"
    assert "`frontend/src/api.ts`" in text and "Shell functions: `stale_name`." in text
    assert run(repo, "--check", base="HEAD").returncode == 0


# ---- the generated cell: what a reader of the overview is told ---------------------------------

DETAILED = '''"""Detailed."""


class Gate:
    @property
    def ready(self):
        return True

    @ready.setter
    def ready(self, value):
        pass

    @ready.deleter
    def ready(self):
        pass

    @classmethod
    def make(cls, x, y=2):
        return cls()

    @staticmethod
    def helper(a):
        return a

    async def poll(self, timeout=1.0):
        return 1

    def collect(self, first, *rest, flag=False, **options):
        return first

    def __init__(self, config):
        pass


async def fetch(url, /, retries=3):
    return url


def plain(a, b, /, c):
    return a
'''


def test_the_cell_says_what_a_caller_passes_and_lists_a_property_once():
    cell = scs.interfaces(ast.parse(DETAILED))

    assert cell == (
        "Class `Gate`: `ready`, `make(x, y=2)`, `helper(a)`, `poll(timeout=1.0)`, "
        "`collect(first, *rest, flag=False, **options)`; "
        "`fetch(url, retries=3)`, `plain(a, b, c)`"
    )


def test_a_definition_name_that_is_private_is_not_listed_but_dunder_methods_are_not_public_either():
    assert scs.interfaces(ast.parse("class _P:\n    def m(self):\n        pass\n")) == ""
    assert scs.interfaces(ast.parse("class P:\n    def __repr__(self):\n        return ''\n")) == (
        "Class `P`"
    )


def test_the_purpose_is_the_first_two_sentences_and_is_cut_at_300_characters():
    long_first = "A" * 250 + ". " + "B" * 100 + ". Never shown."
    purpose = scs.purpose_of(ast.parse(f'"""{long_first}"""\n'), "x.py")

    assert len(purpose) <= 300 and purpose.endswith("...")
    assert purpose.startswith("A" * 250)
    assert "Never shown" not in purpose
    assert scs.purpose_of(ast.parse('"""One. Two. Three."""\n'), "x.py") == "One. Two."
    assert scs.purpose_of(ast.parse('"""A | B."""\n'), "x.py") == "A \\| B."
    with pytest.raises(scs.SyncError, match="x.py: add a module docstring"):
        scs.purpose_of(ast.parse("x = 1\n"), "x.py")


def test_duplicate_purposes_merge_every_distinct_clause_longest_row_first():
    merged = scs.merge_purposes(
        ["Runs the service; keeps a latch", "Runs the service", "Also logs the stop; and more"]
    )

    assert merged == "Runs the service; keeps a latch; Also logs the stop; and more"


def test_a_row_with_too_few_cells_cannot_be_read_and_one_with_a_stray_pipe_is_merged():
    with pytest.raises(scs.SyncError, match="has 3 cells, not 4"):
        scs.read_rows(["| `a.py` | purpose | subsystem |"])

    rows = scs.read_rows(["| `a.py` | purpose | subsystem | `f(a | b)` |"])

    assert len(rows[0].cells) == 4 and rows[0].cells[3] == "`f(a | b)`"


@pytest.mark.parametrize(
    ("path", "in_scope"),
    [
        ("backend/src/services/x.py", True),
        ("backend/src/nav/x.py", True),
        ("scripts/x.py", True),
        ("workshop/x.py", True),
        ("firmware/x.py", True),
        (".specify/scripts/x.py", True),
        ("backend/src/tests/x.py", False),
        ("backend/src/compat_stubs/x.py", False),
        ("scripts/__pycache__/x.py", False),
        ("tests/unit/x.py", False),
        ("docs/x.py", False),
        ("frontend/src/x.py", False),
        ("scripts/x.sh", False),
        ("backend/src/x.pyi", False),
    ],
)
def test_which_files_the_check_looks_at(path, in_scope):
    assert scs.in_scope(path) is in_scope


def test_a_section_heading_that_appears_twice_takes_the_new_row_in_the_first(tmp_path):
    lines = ["## A", "", "| p |", "| `x.py` |", "", "## A", "", "| p |", "| `y.py` |"]

    assert scs.table_end(lines, "A") == 4
    with pytest.raises(scs.SyncError, match="no section '## Z'"):
        scs.table_end(lines, "Z")
    with pytest.raises(scs.SyncError, match="section 'B' has no table"):
        scs.table_end(["## B", "text", "## C"], "B")


# ---- which files a change touches (the claims of --base and --all) ------------------------------


def test_a_renamed_file_is_a_touched_file(repo):
    _write(repo, "scripts/old_name.py", '"""Moves."""\n\n\ndef go(a):\n    return a\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/old_name.py", "`go(a)`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a file with a row")
    _git(repo, "mv", "scripts/old_name.py", "scripts/new_name.py")

    checked = run(repo, "--check", base="HEAD")

    assert checked.returncode == 1
    assert "scripts/old_name.py: the file is gone" in checked.stdout
    assert "scripts/new_name.py: new file with public callables and no row" in checked.stdout


def test_files_the_base_branch_changed_after_the_branch_point_are_not_this_branchs(repo):
    """The comparison point is the MERGE-BASE: a change only ``main`` made is not drift here."""
    _write(repo, "scripts/old.py", '"""Old."""\n\n\ndef keep(a, b):\n    return a\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/old.py", "`keep(...)`"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a hand-written summary row")
    _git(repo, "checkout", "-q", "-b", "feat")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "scripts/old.py", '"""Old."""\n\n\ndef keep(a, b):\n    return a + b\n')
    _git(repo, "commit", "-qam", "main edits the body after feat branched")
    _git(repo, "checkout", "-q", "feat")

    checked = run(repo, "--check", base="main")

    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_touched_file_without_public_callables_and_its_placeholder_row_are_current(repo):
    _write(repo, "scripts/quiet.py", '"""Quiet."""\n\n\ndef _inner():\n    return 1\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/quiet.py", "(no public callables)"))

    checked = run(repo, "--check", base="HEAD")

    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_new_file_without_public_callables_needs_no_row(repo):
    _write(repo, "scripts/private.py", '"""P."""\n\n\ndef _x():\n    return 1\n')

    assert run(repo, "--check", base="HEAD").returncode == 0
    assert run(repo, "--fix", base="HEAD").returncode == 0
    assert "scripts/private.py" not in doc_text(repo)


def test_a_file_that_lost_its_public_callables_gets_the_placeholder_and_then_stays_current(repo):
    _write(repo, "scripts/shrunk.py", '"""S."""\n\n\ndef _x():\n    return 1\n')
    _doc(repo, backend=_service_row(), ops=_doc_row("scripts/shrunk.py", "`was_public(a)`"))

    checked = run(repo, "--check", base="HEAD")
    assert checked.returncode == 1 and "`was_public` is not defined" in checked.stdout
    assert run(repo, "--fix", base="HEAD").returncode == 0
    assert "(no public callables)" in doc_text(repo)
    assert run(repo, "--check", base="HEAD").returncode == 0


def test_a_file_that_cannot_be_parsed_is_an_unknown_comparison_not_a_pass(repo):
    _write(repo, "scripts/broken.py", "def broken(:\n    pass\n")

    done = run(repo, "--check", base="HEAD")

    assert done.returncode == 2
    assert "cannot parse" in done.stderr and "broken.py" in done.stderr


def test_the_default_base_is_origin_main_and_a_repository_without_one_cannot_compare(repo):
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_code_structure.py")]
        + ["--root", str(repo), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 2
    assert "'origin/main'" in done.stderr


def test_the_dunder_and_the_bound_arguments_stay_out_of_the_cell():
    assert scs.interfaces(ast.parse("def f(self, cls, a):\n    pass\n")) == "`f(a)`"
