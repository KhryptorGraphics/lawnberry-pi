"""The secret tripwire must catch the leaks that actually happened and stay quiet on code."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_no_secrets as cns  # noqa: E402

# Assembled at run time so this file does not itself contain a credential-looking literal.
FAKE = "Zq" + "7" * 4 + "kLm9"
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "PATH": "/usr/bin:/bin",
}


def rules(line: str, path: str = "docs/x.md") -> list[str]:
    return [rule for rule, _ in cns.scan_line(line, path)]


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        (f"7. Shared machines: workshop box (sudo `{FAKE}`), GPUs", "sudo-literal"),
        (f"- Password: {FAKE} (for initial setup only)", "password-colon-literal"),
        (f'sshpass -p "{FAKE}" ssh kp@host', "sshpass-literal"),
        (f'This creates temporary password: "{FAKE}"', "password-colon-literal"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private-key"),
        ("token=ghp_" + "a" * 36, "github-token"),
        ("HF_TOKEN=hf_" + "B" * 34, "huggingface-token"),
        ("key AKIA" + "ABCDEFGHIJKLMNOP", "aws-access-key"),
    ],
)
def test_the_leaks_that_happened_are_caught(line: str, rule: str):
    assert rule in rules(line)


@pytest.mark.parametrize(
    "line",
    [
        'sudo cp /etc/nginx/nginx.conf "/etc/nginx/nginx.conf.bak.$ts"',
        "sudo systemctl restart lawnberry-backend",
        "Password: provided_after_registration",
        "password: <your password here>",
        "password: ${NTRIP_PASSWORD}",
        'sshpass -p "$PI_PASSWORD" ssh kp@host',
        "the password is hashed with bcrypt before it is compared",
        "password: changeme",
    ],
)
def test_documentation_and_placeholders_are_not_flagged(line: str):
    assert rules(line) == []


def test_source_files_are_not_flagged_for_field_names_but_still_for_real_tokens():
    # `password: str` and env reads are code, not secrets ...
    assert rules(f"password: {FAKE}", "backend/src/x.py") == []
    # ... but a real token format is a leak anywhere.
    assert "github-token" in rules("ghp_" + "c" * 36, "backend/src/x.py")


def test_an_allow_list_entry_silences_exactly_that_literal_in_that_file():
    text = f"sudo `{FAKE}`\n"
    entry = f"docs/a.md:{cns._digest(FAKE)}"
    assert cns.scan_text("docs/a.md", text, {entry}) == []
    assert cns.scan_text("docs/other.md", text, {entry}) != []


def test_added_lines_parsing_reports_the_right_file_and_line():
    diff = (
        "diff --git a/docs/a.md b/docs/a.md\n--- a/docs/a.md\n+++ b/docs/a.md\n"
        "@@ -1,0 +5,2 @@\n+first\n+second\n"
        "diff --git a/old.md b/old.md\n--- a/old.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n"
    )
    assert list(cns.added_lines(diff)) == [("docs/a.md", 5, "first"), ("docs/a.md", 6, "second")]


# --- the real CLI, inside a throwaway git repository ----------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=GIT_ENV)


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", f"add {name}")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_no_secrets.py", r / "scripts" / "check_no_secrets.py")
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _commit(r, "a.md", "clean\n")
    return r


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_no_secrets.py"), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env=GIT_ENV,
        check=False,
    )


def test_a_new_leak_fails_the_diff_scan_and_the_value_is_never_printed(repo: Path):
    _commit(repo, "b.md", f"ssh with sudo `{FAKE}`\n")
    proc = _run(repo, "--base", "HEAD~1")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert "b.md:1: sudo-literal" in out
    assert cns._digest(FAKE) in out
    assert FAKE not in out


def test_an_unchanged_tree_passes_the_diff_scan(repo: Path):
    assert _run(repo, "--base", "HEAD").returncode == 0


def test_a_secret_added_and_deleted_inside_the_same_change_still_fails(repo: Path):
    """Deleting it in a later commit does not unpublish it: the history of the PR still holds it."""
    _commit(repo, "b.md", f"ssh with sudo `{FAKE}`\n")
    _commit(repo, "b.md", "scrubbed\n")
    assert (repo / "b.md").read_text() == "scrubbed\n"
    proc = _run(repo, "--base", "HEAD~2")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "b.md:1: sudo-literal" in proc.stdout
    assert FAKE not in proc.stdout + proc.stderr


def test_the_reviewed_allow_list_holds_only_the_known_historic_leak():
    # One entry, for the goal file whose first commit carried a credential; nothing else is excused.
    entries = sorted(cns.load_allow())
    assert len(entries) == 1
    assert entries[0].startswith("docs/GOAL-2026-10-24-finish.md:")


def test_a_leak_already_in_history_does_not_block_a_later_clean_change(repo: Path):
    _commit(repo, "b.md", f"ssh with sudo `{FAKE}`\n")
    _commit(repo, "c.md", "nothing secret here\n")
    assert _run(repo, "--base", "HEAD~1").returncode == 0


def test_tree_mode_finds_a_committed_leak(repo: Path):
    _commit(repo, "b.md", f"ssh with sudo `{FAKE}`\n")
    proc = _run(repo, "--tree")
    assert proc.returncode == 1
    assert FAKE not in proc.stdout + proc.stderr


def test_an_unresolvable_base_fails_closed_not_open(repo: Path):
    proc = _run(repo, "--base", "refs/definitely/not/a/ref")
    assert proc.returncode == 2
    assert "could not scan" in proc.stderr


def test_the_repository_tree_has_no_credential_literal_outside_the_reviewed_allow_list():
    allow = cns.load_allow()
    found = cns.scan_tree(allow)
    assert [(f.path, f.line_no, f.rule) for f in found] == []
