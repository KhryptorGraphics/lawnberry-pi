"""The CI workflows behave under the events that matter, replayed the way GitHub runs them.

``test_workflow_steps.py`` replays the dependency and hardware guards. This file replays the three
workflows that judge every pull request, with a small evaluator for the ``${{ }}`` expressions they
use (comparisons, ``&&``, ``||``, ``github.*`` lookups), so what is tested is what GitHub would do:

* ``secret-scan.yml`` runs the BASE branch's scanner and allow-list on a pull request (SEC3-12),
  and a push is never cancelled or replaced, so no pushed range goes unscanned (TOOL3-04);
* ``docs-drift.yml`` never puts a branch name into shell text (TOOL3-13);
* no ``run:`` body of any workflow expands attacker-chosen text (the class of TOOL3-13);
* the job that runs the whole suite has a time limit (TOOL3-01).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
FAKE = "Zq" + "7" * 4 + "kLm9"  # assembled so this file holds no credential-looking literal


# ------------------------------------------------------------------ the ${{ }} expressions

_TOKEN = re.compile(r"\s*(?:([A-Za-z_][\w.]*)|'([^']*)'|(==|!=|&&|\|\||\(|\)))")


def evaluate(expression: str, context: dict):
    """One GitHub expression: ``a == 'x' && b.c || ''``. ``&&`` and ``||`` return an operand."""
    tokens: list[tuple[str, str]] = []
    position = 0
    expression = expression.strip()
    while position < len(expression):
        m = _TOKEN.match(expression, position)
        assert m, f"cannot read {expression!r} at {position}"
        position = m.end()
        if m[1]:
            tokens.append(("name", m[1]))
        elif m[2] is not None:
            tokens.append(("str", m[2]))
        else:
            tokens.append(("op", m[3]))
    cursor = 0

    def peek():
        return tokens[cursor] if cursor < len(tokens) else ("end", "")

    def take():
        nonlocal cursor
        cursor += 1
        return tokens[cursor - 1]

    def atom():
        kind, text = take()
        if (kind, text) == ("op", "("):
            value = disjunction()
            assert take() == ("op", ")")
            return value
        if kind == "str":
            return text
        value = context
        for part in text.split("."):
            value = value.get(part, "") if isinstance(value, dict) else ""
        return value

    def comparison():
        left = atom()
        if peek() in (("op", "=="), ("op", "!=")):
            op = take()[1]
            right = atom()
            same = str(left).lower() == str(right).lower()
            return same if op == "==" else not same
        return left

    def conjunction():
        value = comparison()
        while peek() == ("op", "&&"):
            take()
            right = comparison()
            value = right if value else value
        return value

    def disjunction():
        value = conjunction()
        while peek() == ("op", "||"):
            take()
            right = conjunction()
            value = value if value else right
        return value

    result = disjunction()
    assert peek()[0] == "end", f"trailing tokens in {expression!r}"
    return result


def render(text: str, context: dict) -> str:
    """What GitHub does to a ``run:`` body before the shell sees it."""

    def one(m: re.Match[str]) -> str:
        value = evaluate(m[1], context)
        return {True: "true", False: "false"}.get(value, str(value))

    return re.sub(r"\$\{\{(.*?)\}\}", one, text)


def workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def step(name: str, title: str) -> dict:
    return next(
        s for job in workflow(name)["jobs"].values() for s in job["steps"] if s.get("name") == title
    )


def replay(name: str, title: str, context: dict, cwd: Path, *, path: str | None = None):
    """Run one step as GitHub would: its ``env:`` evaluated, its ``run:`` rendered, ``bash -e``."""
    s = step(name, title)
    env = {"PATH": path or "/usr/bin:/bin:" + str(Path(sys.executable).parent), **GIT_ENV}
    env |= {key: render(str(value), context) for key, value in s.get("env", {}).items()}
    return subprocess.run(
        ["bash", "-e", "-c", render(s["run"], context)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def github(**values) -> dict:
    event = values.pop("event", {})
    return {"github": {**values, "event": event}}


# --------------------------------------------------------- the pull request is judged by the base

PULL_REQUEST_STEP = "Scan the lines this pull request adds"
LEAK = f"ssh with sudo `{FAKE}`\n"


def sha12(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:12]


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={**GIT_ENV, "PATH": "/usr/bin:/bin"},
    )
    return done.stdout.strip()


def commit(repo: Path, files: dict[str, str | None], message: str) -> str:
    for name, text in files.items():
        path = repo / name
        if text is None:
            git(repo, "rm", "-q", "--", name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        git(repo, "add", "--", name)
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A repository whose ``main`` has the real scanner and an allow-list that excuses nothing."""
    repo = tmp_path / "project"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    scanner = (REPO / "scripts" / "check_no_secrets.py").read_text()
    commit(
        repo,
        {
            "scripts/check_no_secrets.py": scanner,
            "spec/secret_scan_allow.txt": "# nothing is excused\n",
            "README.md": "clean\n",
        },
        "base",
    )
    return repo


def pull_request(repo: Path, base_sha: str):
    context = github(event_name="pull_request", event={"pull_request": {"base": {"sha": base_sha}}})
    return replay("secret-scan.yml", PULL_REQUEST_STEP, context, repo)


STUB = "import sys\nprint('check_no_secrets: OK')\nsys.exit(0)\n"


def test_a_pull_request_that_replaces_the_scanner_is_still_judged_by_the_base_scanner(project):
    base = git(project, "rev-parse", "HEAD")
    git(project, "checkout", "-qb", "feat")
    commit(
        project, {"docs/leak.md": LEAK, "scripts/check_no_secrets.py": STUB}, "leak, scanner off"
    )

    # what the workflow did before: the pull request's own scanner, which says all is well
    own = subprocess.run(
        [sys.executable, "scripts/check_no_secrets.py", "--base", base],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert own.returncode == 0, "the premise: the pull request's own scanner waves it through"

    done = pull_request(project, base)

    assert done.returncode == 1, done.stdout + done.stderr
    assert "docs/leak.md:1: sudo-literal" in done.stdout
    assert FAKE not in done.stdout + done.stderr


def test_a_pull_request_cannot_excuse_its_own_literal_in_the_allow_list(project):
    base = git(project, "rev-parse", "HEAD")
    git(project, "checkout", "-qb", "feat")
    commit(
        project,
        {
            "docs/leak.md": LEAK,
            "spec/secret_scan_allow.txt": f"docs/leak.md:{sha12(FAKE)}  # trust me\n",
        },
        "leak and its own excuse",
    )
    own = subprocess.run(
        [sys.executable, "scripts/check_no_secrets.py", "--base", base],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert own.returncode == 0, "the premise: its own allow-list excuses it"

    done = pull_request(project, base)

    assert done.returncode == 1 and "docs/leak.md:1: sudo-literal" in done.stdout


def test_a_pull_request_that_deletes_the_allow_list_or_the_scanner_changes_nothing(project):
    base = git(project, "rev-parse", "HEAD")
    git(project, "checkout", "-qb", "feat")
    commit(
        project,
        {
            "docs/leak.md": LEAK,
            "scripts/check_no_secrets.py": None,
            "spec/secret_scan_allow.txt": None,
        },
        "x",
    )

    done = pull_request(project, base)

    assert done.returncode == 1 and "docs/leak.md:1: sudo-literal" in done.stdout


def test_a_clean_pull_request_passes_whatever_it_did_to_the_scanner(project):
    base = git(project, "rev-parse", "HEAD")
    git(project, "checkout", "-qb", "feat")
    commit(
        project,
        {"docs/ok.md": "fine\n", "scripts/check_no_secrets.py": STUB},
        "clean, scanner edited",
    )

    done = pull_request(project, base)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "check_no_secrets: OK" in done.stdout


def test_the_base_allow_list_still_excuses_what_it_excuses(project):
    """The reviewed entry of the base applies to a pull request that adds the same literal."""
    git(project, "checkout", "-q", "main")
    base = commit(
        project,
        {"spec/secret_scan_allow.txt": f"docs/known.md:{sha12(FAKE)}  # reviewed\n"},
        "allow",
    )
    git(project, "checkout", "-qb", "feat")
    commit(project, {"docs/known.md": LEAK}, "the same literal in the same file")

    done = pull_request(project, base)

    assert done.returncode == 0, done.stdout + done.stderr


def test_the_pull_request_that_introduces_the_scanner_is_judged_by_its_own_and_says_so(tmp_path):
    repo = tmp_path / "bootstrap"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    base = commit(repo, {"README.md": "clean\n"}, "a base with no scanner")
    git(repo, "checkout", "-qb", "feat")
    scanner = (REPO / "scripts" / "check_no_secrets.py").read_text()
    commit(
        repo,
        {"scripts/check_no_secrets.py": scanner, "docs/leak.md": LEAK},
        "adds the scanner and a leak",
    )

    done = pull_request(repo, base)

    assert "::warning::the base branch has no scripts/check_no_secrets.py yet" in done.stdout
    assert done.returncode == 1 and "docs/leak.md:1: sudo-literal" in done.stdout


def test_the_base_without_an_allow_list_leaves_the_pull_request_none(project):
    git(project, "checkout", "-q", "main")
    base = commit(project, {"spec/secret_scan_allow.txt": None}, "the base has no allow-list")
    git(project, "checkout", "-qb", "feat")
    commit(
        project,
        {
            "docs/leak.md": LEAK,
            "spec/secret_scan_allow.txt": f"docs/leak.md:{sha12(FAKE)}  # mine\n",
        },
        "x",
    )

    done = pull_request(project, base)

    assert done.returncode == 1 and "docs/leak.md:1: sudo-literal" in done.stdout


PUSH_STEP = "Scan the lines this push adds"


def push(repo: Path, before: str):
    return replay(
        "secret-scan.yml", PUSH_STEP, github(event_name="push", event={"before": before}), repo
    )


def test_a_push_scans_exactly_the_commits_it_brought(project):
    before = git(project, "rev-parse", "HEAD")
    commit(project, {"docs/leak.md": LEAK}, "pushed")

    assert push(project, before).returncode == 1
    assert push(project, git(project, "rev-parse", "HEAD")).returncode == 0


@pytest.mark.parametrize("before", ["0" * 40, "f" * 40, ""])
def test_a_push_with_no_usable_before_scans_the_whole_tree_instead_of_nothing(project, before):
    commit(project, {"docs/leak.md": LEAK}, "new branch")

    done = push(project, before)

    assert done.returncode == 1 and "docs/leak.md:1: sudo-literal" in done.stdout


# ------------------------------------------------------------------------------- concurrency


def group_of(name: str, **values) -> tuple[str, bool]:
    concurrency = workflow(name)["concurrency"]
    context = github(**values)
    group = render(concurrency["group"], context)
    cancel = concurrency["cancel-in-progress"]
    cancel = evaluate(cancel[3:-2], context) if isinstance(cancel, str) else cancel
    return group, bool(cancel)


def test_two_pushes_to_main_never_share_a_group_so_neither_is_cancelled_or_replaced():
    first = group_of("secret-scan.yml", event_name="push", ref="refs/heads/main", sha="a" * 40)
    second = group_of("secret-scan.yml", event_name="push", ref="refs/heads/main", sha="b" * 40)

    assert first[0] != second[0], "same group: GitHub keeps one pending run and drops the rest"
    assert first[1] is False and second[1] is False, "a cancelled push run is an unscanned range"


def test_a_pull_request_run_is_superseded_by_the_next_push_to_the_same_pull_request():
    one = group_of(
        "secret-scan.yml", event_name="pull_request", ref="refs/pull/20/merge", sha="a" * 40
    )
    two = group_of(
        "secret-scan.yml", event_name="pull_request", ref="refs/pull/20/merge", sha="b" * 40
    )
    other = group_of(
        "secret-scan.yml", event_name="pull_request", ref="refs/pull/21/merge", sha="c" * 40
    )

    assert one[0] == two[0] != other[0]
    assert one[1] is True, "the next run scans the whole range again"


# -------------------------------------------------------------------- branch names in the shell

DRIFT = "Code structure overview follows the changed files"
EVIL = 'a"; echo INJECTED #'


def fake_python(tmp_path: Path) -> str:
    """A ``python3`` that prints each argument on its own line: shows how the shell split them."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "python3"
    exe.write_text('#!/bin/sh\nfor a in "$@"; do printf \'ARG<%s>\\n\' "$a"; done\n')
    exe.chmod(0o755)
    return f"{bin_dir}:/usr/bin:/bin"


def test_the_harness_shows_the_old_step_could_be_made_to_run_a_command(tmp_path):
    old = 'python3 scripts/sync_code_structure.py --check --base "origin/${{ github.base_ref }}"'
    script = tmp_path / "old.sh"
    script.write_text(render(old, github(base_ref=EVIL)))

    done = subprocess.run(
        ["bash", "-e", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": fake_python(tmp_path)},
    )

    assert "\nINJECTED\n" in "\n" + done.stdout, "the premise: a branch name became a command"


def test_a_branch_name_with_shell_syntax_is_one_argument_and_runs_nothing(tmp_path):
    done = replay(
        "docs-drift.yml",
        DRIFT,
        github(event_name="pull_request", base_ref=EVIL),
        tmp_path,
        path=fake_python(tmp_path),
    )

    lines = done.stdout.splitlines()
    assert done.returncode == 0, done.stderr
    assert 'ARG<origin/a"; echo INJECTED #>' in lines
    assert "INJECTED" not in lines, "the branch name was executed"


def test_a_substitution_in_a_branch_name_is_not_expanded(tmp_path):
    marker = tmp_path / "ran"
    base = f"a$(touch {marker})b`touch {marker}`"

    replay(
        "docs-drift.yml",
        DRIFT,
        github(event_name="pull_request", base_ref=base),
        tmp_path,
        path=fake_python(tmp_path),
    )

    assert not marker.exists()


DANGEROUS = re.compile(
    r"\$\{\{[^}]*(?:head_ref|base_ref|\.title|\.body|\.head\.ref|\.head\.label|\.default_branch"
    r"|\.page_name|\.message|\.author\.(?:email|name)|\.email)\b[^}]*\}\}"
)


def test_no_run_body_expands_text_an_outsider_chooses():
    """Branch names, titles, bodies and commit messages belong in ``env:``, never in a script."""
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job in yaml.safe_load(path.read_text())["jobs"].values():
            for s in job.get("steps", []):
                for hit in DANGEROUS.findall(s.get("run", "")):
                    offenders.append(f"{path.name}: {s.get('name', s.get('run', '')[:40])}: {hit}")
    assert offenders == []


def test_the_detector_of_that_class_sees_the_old_step():
    assert DANGEROUS.search('--base "origin/${{ github.base_ref }}"')
    assert DANGEROUS.search("echo ${{ github.event.pull_request.title }}")
    assert not DANGEROUS.search("echo ${{ github.sha }} ${{ runner.os }}")


# ----------------------------------------------------------------------- the suite's job limit


@pytest.mark.parametrize("name", ["sim-tests.yml", "secret-scan.yml"])
def test_the_job_that_runs_the_checks_has_a_time_limit_far_below_githubs_default(name):
    limits = [job.get("timeout-minutes") for job in workflow(name)["jobs"].values()]

    assert all(isinstance(m, int) and 5 <= m <= 120 for m in limits), (
        f"{name}: {limits}: GitHub's default is 360 minutes with no verdict"
    )
