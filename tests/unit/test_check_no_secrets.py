"""The secret tripwire must catch the leaks that actually happened and stay quiet on code.

Three layers: the patch parser (a content line can never be read as a header), the rules (the
shapes of the leaks that happened, of the credentials this product creates, and what must not be
mistaken for one), and the real CLI run inside throwaway git repositories (merges, attributes, odd
paths, allow-list).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_no_secrets as cns  # noqa: E402

# Assembled at run time so this file does not itself contain a credential-looking literal.
FAKE = "Zq" + "7" * 4 + "kLm9"
STRONG = "Zq7!" + "vK2m" + "-lawn-crow-" + "9Xp"  # what a chosen passphrase looks like
HEX64 = "9f2c4b7a" * 8  # what a generated signing secret looks like
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "PATH": "/usr/bin:/bin",
}


def rules(line: str, path: str = "docs/x.md") -> list[str]:
    return [rule for rule, _ in cns.scan_line(line, path)]


def sha12(literal: str) -> str:
    """The digest an allow-list entry carries, computed here and not by the code under test."""
    return hashlib.sha256(literal.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------- the leaks so far


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        (f"7. Shared machines: workshop box (sudo `{FAKE}`), GPUs", "sudo-literal"),
        (f"- Password: {FAKE} (for initial setup only)", "password-colon-literal"),
        (f'sshpass -p "{FAKE}" ssh kp@host', "sshpass-literal"),
        (f'This creates temporary password: "{FAKE}"', "password-colon-literal"),
        (f"root password is `{FAKE}` on the box", "labelled-password-literal"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private-key"),
        ("-----BEGIN PRIVATE KEY-----", "private-key"),
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


@pytest.mark.parametrize(
    "line",
    [
        f"password: example{FAKE}",
        f"password: redacted-{FAKE}",
        f"password: placeholder-{FAKE}",
        f"password: your-{FAKE}",
        f"password: your_{FAKE}",
        f"password: changeme{FAKE}",
        f"password: change-me-{FAKE}",
        "password: provided-by-installer-2026",
        "password: some_lower_case_name",
        "password: generate_password(12)",
        f"sudo `<{FAKE}>`",
        f"sudo `${FAKE}`",
        f"sudo `{{{FAKE}}}`",
        "password: " + "x" * 12,
        "password: " + "*" * 12,
        "password: " + "." * 12,
        "password: secret",
        "password: passwd",
        "password: password",
        "Password: provided_after_registration",
    ],
)
def test_every_placeholder_family_is_excused(line: str):
    assert rules(line) == []


@pytest.mark.parametrize("word", ["secret", "passwd", "password"])
def test_only_the_bare_placeholder_word_is_excused_not_a_value_that_starts_with_it(word: str):
    assert rules(f"password: {word}") == []
    assert rules(f"password: {word}x1") == ["password-colon-literal"]


def test_an_allow_list_entry_silences_exactly_that_literal_in_that_file():
    text = f"sudo `{FAKE}`\n"
    entry = f"docs/a.md:{sha12(FAKE)}"
    assert cns.scan_text("docs/a.md", text, {entry}) == []
    assert cns.scan_text("docs/other.md", text, {entry}) != []
    other = f"sudo `{FAKE}x`\n"
    assert cns.scan_text("docs/a.md", other, {entry}) != [], "another literal in the same file"


# ------------------------------------------------------- the credentials this product creates


def env_lines() -> list[tuple[str, str, str]]:
    """``(path, line, rule)`` for the credential shapes of ``.env.example`` and the settings."""
    return [
        ("docs/x.md", f"LAWN_BERRY_OPERATOR_CREDENTIAL={STRONG}", "secret-assignment"),
        ("scripts/x.sh", f"export LAWN_BERRY_OPERATOR_CREDENTIAL={STRONG}", "secret-assignment"),
        (".env", f"LAWN_BERRY_AUTH_SECRET={HEX64}", "secret-assignment"),
        (".env", f"NTRIP_PASSWORD={STRONG}", "secret-assignment"),
        (".env", f"MANUAL_CONTROL_PASSWORD={STRONG}", "secret-assignment"),
        (".env", f"NTRIP_PASS={STRONG}", "secret-assignment"),
        (
            ".env",
            f"NGROK_AUTHTOKEN={'2abcDEF1ghiJKL2mnoPQR3stuVWX4yzA_5bcdEFG'}",
            "secret-assignment",
        ),
        (".env", f"THOR_LINK_KEY={HEX64}", "secret-assignment"),
        ("scripts/x.sh", f"SSHPASS={STRONG} sshpass -e ssh pi@host", "secret-assignment"),
        ("config/x.json", f'{{"password": "{STRONG}"}}', "secret-assignment"),
        ("config/x.yaml", f'password: "{STRONG}"', "password-colon-literal"),
        ("config/x.yaml", f"ntrip_password: {STRONG}", "secret-assignment"),
        ("config/x.yaml", f"pw: {STRONG}", "secret-assignment"),
        ("config/x.yaml", f"passwd: {STRONG}", "secret-assignment"),
        ("docs/x.md", f"wifi psk={STRONG}", "secret-assignment"),
        ("docs/x.md", f"X-Api-Key: {STRONG}", "secret-assignment"),
        ("docs/x.md", f"--api-key={STRONG}", "secret-assignment"),
        ("scripts/x.py", f'PASSWORD = "{STRONG}"', "secret-assignment"),
        ("scripts/x.py", f'ntrip_password="{STRONG}"', "secret-assignment"),
        ("web/x.ts", f"const cfg = {{ googleApiKey: '{STRONG}' }}", "secret-assignment"),
        ("web/x.vue", f'const apiKey = "{STRONG}"', "secret-assignment"),
        ("docs/x.md", f"sudo password is {STRONG}", "password-prose-literal"),
        ("docs/x.md", f"the sudo password: {STRONG}", "password-colon-literal"),
        ("docs/x.md", f"ssh pi@192.168.1.64 # password {STRONG}", "password-prose-literal"),
        ("docs/x.md", f"Password: {STRONG}", "password-colon-literal"),
        ("docs/x.md", f"password = {STRONG}", "password-colon-literal"),
    ]


@pytest.mark.parametrize(("path", "line", "rule"), env_lines())
def test_the_credential_shapes_the_product_creates_are_caught(path: str, line: str, rule: str):
    assert rule in rules(line, path)


def test_the_literal_is_the_value_not_the_name():
    assert [lit for r, lit in cns.scan_line(f"NTRIP_PASSWORD={STRONG}", ".env")] == [STRONG]
    assert [lit for r, lit in cns.scan_line(f'{{"password": "{STRONG}"}}', "a.json")] == [STRONG]
    assert [lit for r, lit in cns.scan_line(f"ssh # password {STRONG}", "a.md")] == [STRONG]


def test_one_literal_is_reported_once_even_when_two_rules_see_it():
    # `Password: X` is the old password rule and a credential-named assignment at the same time
    assert rules(f"Password: {STRONG}") == ["password-colon-literal"]


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("AIza" + "x1Y2" * 8 + "abc", "google-api-key"),
        ("key=AIza" + "x1Y2" * 8 + "ab-", "google-api-key"),
        ("glpat-" + "abcdefghij" * 2, "gitlab-token"),
        ("ASIA" + "ABCDEFGHIJKLMNOP", "aws-access-key"),
        ("xoxb-" + "1234567890-" * 2 + "abcdefghijklmnop", "slack-token"),
        (
            "https://hooks.slack.com/services/T" + "0" * 8 + "/B" + "0" * 8 + "/" + "X" * 24,
            "slack-webhook",
        ),
        ("npm_" + "a1B2" * 9, "npm-token"),
        ("pypi-AgEIcHlwaS5vcmc" + "A" * 60, "pypi-token"),
        ("sk-" + "a" * 40, "openai-style-key"),
        ("sk-ant-api03-" + "A" * 40, "openai-style-key"),
        ("eyJ" + "a" * 20 + ".eyJ" + "b" * 20 + "." + "c" * 12, "jwt"),
        ("tunnel eyJ" + "hIjoi" * 14, "cloudflare-tunnel-token"),
        ("sudo cloudflared service install eyJ" + "hIjoi" * 30, "cloudflare-tunnel-token"),
        ("ghp_" + "a" * 36, "github-token"),
        ("github_pat_11ABCDEFG0123456789_" + "a" * 60, "github-token"),
        ("xoxp-" + "1234567890-" * 2 + "abcdefghijklmnop", "slack-token"),
    ],
)
def test_token_formats_are_caught_anywhere_and_never_excused_as_placeholders(line: str, rule: str):
    assert rule in rules(line, "docs/x.md")
    assert rule in rules(line, "backend/src/x.py"), "a token format is a leak in source too"


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        (
            f"ngrok config add-authtoken {'2abcDEF1ghiJKL2mnoPQR3stuVWX4yzA_5bcdEFGhij'}",
            "ngrok-authtoken-literal",
        ),
        ('curl -H "Authorization: Bearer ' + "3f9a8b7c" * 4 + '" http://x', "bearer-literal"),
        (f"curl -u admin:{STRONG} http://192.168.1.64:8081", "curl-user-literal"),
        (f"curl --user admin:{STRONG} http://x", "curl-user-literal"),
        (f"rtsp://camuser:{STRONG}@192.168.1.10/stream", "url-userinfo-literal"),
        (f"postgresql://lawn:{STRONG}@localhost/db", "url-userinfo-literal"),
        (f'echo "{STRONG}" | sudo -S systemctl restart x', "sudo-stdin-literal"),
        (f"printf '%s\\n' '{STRONG}' | sudo -S apt update", "sudo-stdin-literal"),
        (f"echo {FAKE} | sudo -S true", "sudo-stdin-literal"),
    ],
)
def test_command_line_shapes_with_a_literal_credential_are_caught(line: str, rule: str):
    assert rule in rules(line)


@pytest.mark.parametrize(
    ("line", "path"),
    [
        # the placeholders of this repository's own runbooks
        ("NTRIP_PASSWORD=your_password", ".env.example"),
        ("NTRIP_PASSWORD=<pass>", "docs/gps.md"),
        ("CLOUDFLARE_API_TOKEN=cf_xxx ./scripts/setup_lets_encrypt.sh", "docs/lets-encrypt.md"),
        ('LAWN_BERRY_OPERATOR_CREDENTIAL="ci-$(openssl rand -hex 16)"', ".github/workflows/x.yml"),
        ("export LAWN_BERRY_OPERATOR_CREDENTIAL='your-strong-credential'", "docs/x.md"),
        ("LAWN_BERRY_OPERATOR_CREDENTIAL=<a strong passphrase>", "docs/x.md"),
        ("THOR_LINK_KEY=$(openssl rand -hex 32)", "docs/x.md"),
        (
            'TOKEN=$(curl -s -X POST http://localhost:8081/api/v2/auth/login -d "$BODY")',
            "docs/x.md",
        ),
        ("curl -H 'Authorization: Bearer $TOKEN' http://x", "docs/x.md"),
        ('curl -H "Authorization: Bearer <token>" http://x', "docs/x.md"),
        ('curl -H "Authorization: Bearer ${TOKEN}" http://x', "docs/x.md"),
        ("curl -u admin:$PASSWORD http://x", "docs/x.md"),
        ("curl -u admin: http://x", "docs/x.md"),
        ("postgresql://user:password@localhost/db", "docs/x.md"),
        ("rtsp://user:<password>@192.168.1.10/stream", "docs/x.md"),
        ("echo $PASSWORD | sudo -S true", "docs/x.md"),
        ("echo 'x' | sudo tee /etc/x", "docs/x.md"),
        ("ngrok config add-authtoken <YOUR_TOKEN>", "docs/x.md"),
        ("sshpass -f /tmp/pw ssh x", "docs/x.md"),
        # settings ABOUT a credential are not a credential
        ("token_type: bearer-access-token", "config/x.yaml"),
        ("AUTH_TOKEN_TTL=3600", ".env"),
        ("PASSWORD_POLICY=min-12-chars-and-more", ".env"),
        ("secret_name: lawnberry-signing-secret", "config/x.yaml"),
        ("LAWN_BERRY_AUTH_SECRET_FILE=/etc/lawnberry/secret", ".env"),
        ("token_endpoint: https://example.test/oauth/token", "config/x.yaml"),
        ("TOKEN_SERVICE=backend.src.services.token_service", ".env"),
        ("EXPIRES_TOKEN=2026-10-25T01:00:00Z", ".env"),
        ("PUBLIC_KEY=" + "A" * 60, ".env"),
        ("test_get_routes_expose_no_secrets.py:43 (old: 34)", "docs/x.md"),
        ("`tests/unit/test_auth.py::test_password_authentication_success`", "docs/x.md"),
        ("--password-file /run/secrets/pw", "docs/x.md"),
        # source code: names, types, reads, constants, calls
        ("password: str = ''", "backend/src/x.py"),
        ("password: Optional[str] = None", "backend/src/x.py"),
        ("password = os.environ['NTRIP_PASSWORD']", "backend/src/x.py"),
        ("token = secrets.token_urlsafe(32)", "backend/src/x.py"),
        ("SIMULATION_TOKEN = '1'", "backend/src/x.py"),
        ("MIN_CREDENTIAL_LENGTH = 12", "backend/src/x.py"),
        ("KEY_SECRET_NAME = 'THOR_LINK_KEY'", "backend/src/x.py"),
        ("AUTH_COOKIE_TOKEN = 'lb_session_token'", "backend/src/x.py"),
        ("TOKEN_URL = '/api/v2/auth/token'", "backend/src/x.py"),
        ("token: string | null = null", "frontend/src/x.ts"),
        ("const token = localStorage.getItem('auth_token')", "frontend/src/x.ts"),
        (':password="form.password"', "frontend/src/x.vue"),
        ('headers["Authorization"] = f"Bearer {token}"', "backend/src/x.py"),
        ("Authorization: `Bearer ${token}`", "frontend/src/x.ts"),
    ],
)
def test_what_the_repository_writes_about_credentials_is_not_a_credential(line: str, path: str):
    assert rules(line, path) == []


def test_a_quoted_literal_in_source_is_data_but_an_unquoted_one_is_code():
    assert rules(f'api_key = "{STRONG}"', "backend/src/x.py") == ["secret-assignment"]
    assert rules(f"api_key = {STRONG}", "backend/src/x.py") == []
    assert rules(f"api_key = {STRONG}", "scripts/x.sh") == ["secret-assignment"]


def test_a_line_of_binary_garbage_is_not_scanned_but_a_text_line_with_a_stray_byte_is():
    noise = "\x01\x02\x03\ufffd" * 8
    assert cns.scan_text("docs/x.md", f"{noise}PW={STRONG}{noise}\n", set()) == []
    assert cns.scan_text("docs/x.md", f"PW={STRONG} \ufffd\n", set()) != []
    assert cns.scan_text("docs/x.md", f"sudo `{FAKE}`\x0c and more\n", set()) != []


@pytest.mark.parametrize(
    "line",
    [
        "a" * 200_000,
        "a-" * 100_000,
        "A1_" * 70_000,
        "eyJ" + "a" * 200_000,
        "x.y." * 50_000,
        "password " * 20_000,
        "sudo " * 40_000,
        "://" * 60_000,
        ("name=" + "a" * 90) * 2_000,
    ],
)
def test_no_rule_is_quadratic_on_a_long_line(line: str):
    started = time.monotonic()
    list(cns.scan_line(line, "docs/x.md"))
    assert time.monotonic() - started < 20, "a minified or encoded line must not stall the scan"


# ------------------------------------------------------------------------------ the patch parser


def git_diff(*sections: str) -> str:
    return "".join(sections)


def section(path: str, hunks: str, *, old: str | None = None) -> str:
    header = f"diff --git a/{old or path} b/{path}\nindex 1111111..2222222 100644\n"
    return header + f"--- a/{old or path}\n+++ b/{path}\n" + hunks


def test_added_lines_parsing_reports_the_right_file_and_line():
    diff = (
        "diff --git a/docs/a.md b/docs/a.md\n--- a/docs/a.md\n+++ b/docs/a.md\n"
        "@@ -1,0 +5,2 @@\n+first\n+second\n"
        "diff --git a/old.md b/old.md\n--- a/old.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n"
    )
    assert list(cns.added_lines(diff)) == [("docs/a.md", 5, "first"), ("docs/a.md", 6, "second")]


def test_an_added_line_that_looks_like_a_file_header_is_content_and_does_not_move_the_path():
    """``++ tests/x`` is added TEXT; read as a header it renamed the file to a skipped path."""
    diff = section("docs/plus.md", "@@ -0,0 +1,3 @@\n+ordinary\n+++ tests/not-a-header\n+after\n")

    assert list(cns.added_lines(diff)) == [
        ("docs/plus.md", 1, "ordinary"),
        ("docs/plus.md", 2, "++ tests/not-a-header"),
        ("docs/plus.md", 3, "after"),
    ]


@pytest.mark.parametrize("text", ["++ b/x", "++x", "+", "++", "+++ /dev/null", "-- a/x", "@@ x @@"])
def test_an_added_line_is_never_dropped_whatever_it_starts_with(text: str):
    diff = section("docs/a.md", f"@@ -0,0 +1,1 @@\n+{text}\n")

    assert list(cns.added_lines(diff)) == [("docs/a.md", 1, text)]


def test_a_removed_line_that_looks_like_a_header_or_a_hunk_is_just_removed():
    diff = section(
        "docs/a.md",
        "@@ -1,3 +1,1 @@\n--- a/old\n-@@ -1 +1 @@\n-diff --git x y\n+kept\n",
    )

    assert list(cns.added_lines(diff)) == [("docs/a.md", 1, "kept")]


def test_a_line_with_a_form_feed_or_a_unicode_separator_is_one_line():
    """``str.splitlines`` cuts at \\x0c and \\u2028: the secret after it lost its ``+``."""
    diff = section("docs/a.md", f"@@ -0,0 +1,2 @@\n+before\x0csudo `{FAKE}`\n+x\u2028y\n")

    assert [text for _, _, text in cns.added_lines(diff)] == [
        f"before\x0csudo `{FAKE}`",
        "x\u2028y",
    ]
    assert [r for r, _ in cns.scan_line(f"before\x0csudo `{FAKE}`", "docs/a.md")] == [
        "sudo-literal"
    ]


def test_context_lines_and_no_newline_markers_keep_the_line_numbers_right():
    diff = section(
        "docs/a.md",
        "@@ -3,3 +3,4 @@ heading\n context one\n+added\n context two\n-removed\n"
        "\\ No newline at end of file\n+last\n\\ No newline at end of file\n",
    )

    assert list(cns.added_lines(diff)) == [("docs/a.md", 4, "added"), ("docs/a.md", 6, "last")]


def test_several_hunks_and_several_files_each_get_their_own_numbers():
    diff = git_diff(
        section("a.md", "@@ -0,0 +1,1 @@\n+a1\n@@ -9,0 +10,2 @@\n+a10\n+a11\n"),
        section("b.md", "@@ -1 +1 @@\n-x\n+b1\n"),
    )

    assert list(cns.added_lines(diff)) == [
        ("a.md", 1, "a1"),
        ("a.md", 10, "a10"),
        ("a.md", 11, "a11"),
        ("b.md", 1, "b1"),
    ]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("+++ b/my notes.md\t", "my notes.md"),
        ('+++ "b/dir \\"q\\"/f.md"\t', 'dir "q"/f.md'),
        ('+++ "b/caf\\303\\251.md"', "café.md"),
        ('+++ "b/tab\\there.md"', "tab\there.md"),
        ('+++ "b/back\\\\slash.md"', "back\\slash.md"),
        ("+++ b/b/inner.md", "b/inner.md"),
        ("+++ b/plain.md", "plain.md"),
    ],
)
def test_paths_are_read_the_way_git_writes_them(header: str, expected: str):
    diff = f"diff --git a/x b/x\n--- /dev/null\n{header}\n@@ -0,0 +1 @@\n+text\n"

    assert list(cns.added_lines(diff)) == [(expected, 1, "text")]


def test_a_deleted_file_has_no_path_and_no_added_lines():
    diff = "diff --git a/x b/x\n--- a/x\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n"

    assert list(cns.added_lines(diff)) == []


def test_a_combined_diff_yields_only_the_lines_no_parent_had():
    diff = (
        "diff --cc conf.md\nindex 2560298,4f24896..f3da5dd\n--- a/conf.md\n+++ b/conf.md\n"
        "@@@ -2,1 -2,2 +2,3 @@@ line\n"
        "- line2 feat\n -line2 main\n -theirs\n"
        "++brand new\n"
        "+ only the first parent lacks this\n"
        " +only the second parent lacks this\n"
        "++second new\n"
    )

    assert list(cns.added_lines(diff)) == [
        ("conf.md", 2, "brand new"),
        ("conf.md", 5, "second new"),
    ]


def test_a_combined_diff_of_an_octopus_merge_needs_a_plus_in_every_column():
    diff = (
        "diff --cc 'a b.md'\n--- a/x\n+++ b/x\n@@@@ -1 -1 -1 +1,2 @@@@\n"
        "+++new in all three\n++ new in two\n"
    )
    # three parents, three columns: "+++" is new, "++ " is not
    assert list(cns.added_lines(diff)) == [("x", 1, "new in all three")]


# ----------------------------------------------------------------------- the real CLI in a repo


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, env=GIT_ENV)


def _commit(repo: Path, name: str, text: str | bytes, message: str | None = None) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    _git(repo, "add", "--", name)
    _git(repo, "commit", "-qm", message or f"add {name}")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_no_secrets.py", r / "scripts" / "check_no_secrets.py")
    _git(r, "init", "-q", "-b", "main")
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
    assert sha12(FAKE) in out
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


def test_the_line_numbers_are_those_of_the_file_whatever_surrounds_the_change(repo: Path):
    _commit(repo, "n.md", "one\ntwo\nthree\nfour\nfive\n")
    _commit(repo, "n.md", f"one\ntwo\nsudo `{FAKE}`\nthree\nfour\nfive\nsix\n")

    proc = _run(repo, "--base", "HEAD~1")

    assert "n.md:3: sudo-literal" in proc.stdout
    assert proc.stdout.count("n.md:") == 1, "reported once, though two layers read the change"


# ---- a line that looks like a header must not hide the rest of its file (SEC3-05 a, TOOL3-03) ----


def test_an_added_plus_plus_line_does_not_blind_the_scan_of_its_file(repo: Path):
    """One added ``++ tests/x`` line read as a header sent its file's rest to a skipped path."""
    _commit(repo, "docs/blind.md", f"ordinary\n++ tests/not-a-header\nssh with sudo `{FAKE}`\n")
    _commit(repo, "docs/plain.md", f"ordinary\nssh with sudo `{FAKE}`\n")

    proc = _run(repo, "--base", "HEAD~2")

    assert proc.returncode == 1
    assert "docs/plain.md:2: sudo-literal" in proc.stdout
    assert "docs/blind.md:3: sudo-literal" in proc.stdout, proc.stdout


def test_an_added_line_that_starts_with_plus_plus_is_scanned_itself(repo: Path):
    _commit(repo, "docs/pp.md", f"++ ssh with sudo `{FAKE}`\n")

    proc = _run(repo, "--base", "HEAD~1")

    assert proc.returncode == 1 and "docs/pp.md:1: sudo-literal" in proc.stdout


@pytest.mark.parametrize(
    "name",
    ["my notes.md", "café.md", 'q"uote.md', "tab\there.md", "back\\slash.md", "a b/c d.md"],
)
def test_a_path_with_a_space_a_quote_or_a_non_ascii_letter_is_reported_as_itself(repo, name):
    _commit(repo, f"docs/{name}", f"ssh with sudo `{FAKE}`\n")

    proc = _run(repo, "--base", "HEAD~1")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"docs/{name}:1: sudo-literal" in proc.stdout
    # so it can be allow-listed under that path, and the entry then silences exactly it
    (repo / "spec").mkdir(exist_ok=True)
    (repo / "spec" / "secret_scan_allow.txt").write_text(f"docs/{name}:{sha12(FAKE)}  # a test\n")
    assert _run(repo, "--base", "HEAD~1").returncode == 0


def test_a_path_that_ends_in_a_skipped_suffix_is_skipped_also_when_it_has_a_space(repo: Path):
    _commit(repo, "docs/my picture.svg", f"ssh with sudo `{FAKE}`\n")

    assert _run(repo, "--base", "HEAD~1").returncode == 0


def test_a_git_attribute_cannot_hide_a_files_patch_from_the_scan(repo: Path):
    """``* -diff`` makes git print "Binary files differ" for plain text: the secret vanished."""
    (repo / ".gitattributes").write_text("* -diff\n")
    _commit(repo, "docs/hidden.md", f"ssh with sudo `{FAKE}`\n")

    proc = _run(repo, "--base", "HEAD~1")

    assert proc.returncode == 1 and "docs/hidden.md:1: sudo-literal" in proc.stdout


def test_a_text_file_with_a_nul_byte_is_still_scanned_line_by_line(repo: Path):
    _commit(repo, "docs/nul.md", f"x\0y\nssh with sudo `{FAKE}`\n".encode())

    assert "docs/nul.md:2: sudo-literal" in _run(repo, "--base", "HEAD~1").stdout


def test_a_latin_1_file_is_scanned_in_both_modes(repo: Path):
    _commit(repo, "docs/latin.md", b"caf\xe9\nssh with sudo `" + FAKE.encode() + b"`\n")

    assert "docs/latin.md:2: sudo-literal" in _run(repo, "--base", "HEAD~1").stdout
    assert "docs/latin.md:2: sudo-literal" in _run(repo, "--tree").stdout


def test_binary_garbage_that_happens_to_hold_a_credential_shape_is_not_a_finding(repo: Path):
    noise = bytes(range(1, 32)) * 3 + b"\xff\xfe\xfd" * 20
    _commit(repo, "hardware/part.3mf", noise + b"PW=" + STRONG.encode() + noise + b"\n")

    assert _run(repo, "--base", "HEAD~1").returncode == 0
    assert _run(repo, "--tree").returncode == 0


def test_a_moved_file_is_added_text_at_its_new_path(repo: Path):
    """A secret that was in a skipped path (a fixture) and is moved under ``docs/`` is new there."""
    _commit(repo, "tests/fixture.md", f"ssh with sudo `{FAKE}`\n")
    assert _run(repo, "--base", "HEAD~1").returncode == 0
    _git(repo, "mv", "tests/fixture.md", "docs-fixture.md")
    _git(repo, "commit", "-qm", "move out of tests/")

    proc = _run(repo, "--base", "HEAD~1")

    assert proc.returncode == 1 and "docs-fixture.md:1: sudo-literal" in proc.stdout


# ------------------------------------------------------- which paths are scanned (N2, C15, C38)


@pytest.mark.parametrize(
    "path",
    [
        "docs/x.md",
        "scripts/x.sh",
        "backend/src/x.py",
        "config/x.yaml",
        ".env",
        "x.md",
        "src/tests/x.md",
    ],
)
def test_these_paths_are_scanned_in_both_modes(repo: Path, path: str):
    _commit(repo, path, f"ssh with sudo `{FAKE}`\n")

    assert _run(repo, "--base", "HEAD~1").returncode == 1
    assert _run(repo, "--tree").returncode == 1


@pytest.mark.parametrize(
    "path",
    [
        "tests/x.md",
        "tests/unit/x.py",
        "frontend/tests/x.ts",
        "node_modules/x.md",
        "verification_artifacts/x.json",
        "a.spec.ts",
        "a.spec.tsx",
        "a.spec.js",
        "a.test.ts",
        "a.test.tsx",
        "a.test.js",
        "uv.lock",
        "a.svg",
        "a.png",
        "a.jpg",
        "a.stl",
        "a.step",
        "a.hef",
    ],
)
def test_fixtures_and_generated_artefacts_are_skipped_in_both_modes(repo: Path, path: str):
    _commit(repo, path, f"ssh with sudo `{FAKE}`\n")

    assert _run(repo, "--base", "HEAD~1").returncode == 0, path
    assert _run(repo, "--tree").returncode == 0, path


def test_the_tree_scan_reads_every_file_not_only_the_first_few(repo: Path):
    for n in range(80):
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "docs" / f"f{n:03d}.md").write_text("clean\n")
    (repo / "docs" / "zzz-last.md").write_text(f"ssh with sudo `{FAKE}`\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "many files")

    proc = _run(repo, "--tree")

    assert proc.returncode == 1 and "docs/zzz-last.md:1: sudo-literal" in proc.stdout


def test_a_tree_without_git_is_scanned_by_walking_it(tmp_path: Path):
    """A source tarball, ``git archive`` or a Docker build context has no ``.git``."""
    tree = tmp_path / "plain"
    (tree / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_no_secrets.py", tree / "scripts" / "check_no_secrets.py")
    (tree / "docs").mkdir()
    (tree / "docs" / "ok.md").write_text("clean\n")
    (tree / "node_modules" / "p").mkdir(parents=True)
    (tree / "node_modules" / "p" / "x.md").write_text(f"ssh with sudo `{FAKE}`\n")
    (tree / ".venv").mkdir()
    (tree / ".venv" / "x.md").write_text(f"ssh with sudo `{FAKE}`\n")
    assert not (tree / ".git").exists()

    clean = _run(tree, "--tree")
    (tree / "docs" / "leak.md").write_text(f"ssh with sudo `{FAKE}`\n")
    leaked = _run(tree, "--tree")

    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert leaked.returncode == 1 and "docs/leak.md:1: sudo-literal" in leaked.stdout
    assert "node_modules" not in leaked.stdout and ".venv" not in leaked.stdout


def test_the_diff_scan_needs_a_checkout_and_says_so_instead_of_passing(tmp_path: Path):
    tree = tmp_path / "plain"
    (tree / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_no_secrets.py", tree / "scripts" / "check_no_secrets.py")

    proc = _run(tree, "--base", "origin/main")

    assert proc.returncode == 2 and "could not scan" in proc.stderr


# ---------------------------------------------------------------------- merges (TOOL3-02)


def _branches(repo: Path) -> None:
    """``main`` and ``feat`` both change line 2 of conf.md: merging one into the other conflicts."""
    _commit(repo, "conf.md", "line1\nline2\nline3\n")
    _git(repo, "checkout", "-qb", "feat")
    _commit(repo, "conf.md", "line1\nline2 feat\nline3\n", "feat")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "conf.md", "line1\nline2 main\nline3\n", "main")
    _git(repo, "checkout", "-q", "feat")
    assert _git(repo, "merge", "main", check=False).returncode != 0, "the premise: a conflict"


def _resolve(repo: Path, text: str) -> None:
    (repo / "conf.md").write_text(text)
    _git(repo, "add", "conf.md")
    _git(repo, "commit", "-qm", "merge main (resolved)")


def test_a_secret_written_while_resolving_a_merge_conflict_fails_the_scan(repo: Path):
    _branches(repo)
    _resolve(repo, f"line1\nssh with sudo `{FAKE}`\nline3\n")

    proc = _run(repo, "--base", "main")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "conf.md:2: sudo-literal" in proc.stdout
    assert FAKE not in proc.stdout + proc.stderr


def test_a_secret_a_merge_wrote_and_a_later_commit_scrubbed_still_fails(repo: Path):
    """Added in the merge, gone from the tree: only the merge's combined diff still holds it."""
    _branches(repo)
    _resolve(repo, f"line1\nssh with sudo `{FAKE}`\nline3\n")
    _commit(repo, "conf.md", "line1\nline2 resolved\nline3\n", "scrub")

    proc = _run(repo, "--base", "main")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "conf.md:2: sudo-literal" in proc.stdout
    assert _run(repo, "--tree").returncode == 0, "the tip tree is clean: the history is not"


def test_each_layer_of_the_diff_scan_reads_a_merge_on_its_own(repo: Path):
    """The net diff and the per-commit patches are two independent derivations of the same thing."""
    _branches(repo)
    _resolve(repo, f"line1\nssh with sudo `{FAKE}`\nline3\n")
    saved = cns.ROOT
    cns.ROOT = repo
    try:
        assert [f.rule for f in cns.scan_commits("main", set())] == ["sudo-literal"]
        assert [f.rule for f in cns.scan_net("main", set())] == ["sudo-literal"]
        _commit(repo, "conf.md", "line1\nclean\nline3\n", "scrub")
        assert [f.rule for f in cns.scan_commits("main", set())] == ["sudo-literal"]
        assert cns.scan_net("main", set()) == [], "the net diff only sees what is left"
    finally:
        cns.ROOT = saved


def test_the_diff_scan_asks_both_layers(repo: Path, monkeypatch):
    """Whichever layer is blind, the other still reports: neither may be left out of the scan."""
    _branches(repo)
    _resolve(repo, f"line1\nssh with sudo `{FAKE}`\nline3\n")
    monkeypatch.setattr(cns, "ROOT", repo)

    monkeypatch.setattr(cns, "scan_commits", lambda base, allow: [])
    assert [f.rule for f in cns.scan_diff("main", set())] == ["sudo-literal"]
    monkeypatch.undo()
    monkeypatch.setattr(cns, "ROOT", repo)
    monkeypatch.setattr(cns, "scan_net", lambda base, allow: [])
    assert [f.rule for f in cns.scan_diff("main", set())] == ["sudo-literal"]


def test_a_clean_merge_and_a_resolution_that_takes_one_side_are_not_findings(repo: Path):
    _branches(repo)
    _resolve(repo, "line1\nline2 main\nline3\n")  # take main's line: nothing new

    proc = _run(repo, "--base", "main")

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_leak_that_arrived_from_the_base_branch_in_a_merge_is_not_the_branchs(repo: Path):
    _git(repo, "checkout", "-qb", "feat")
    _commit(repo, "feat.md", "feature\n")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "leak.md", f"ssh with sudo `{FAKE}`\n")
    base = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()  # the base already has it
    _git(repo, "checkout", "-q", "feat")
    _git(repo, "merge", "-q", "--no-edit", "main")

    assert _run(repo, "--base", base).returncode == 0


def test_unrelated_histories_are_scanned_whole(repo: Path):
    _git(repo, "checkout", "-q", "--orphan", "other")
    _git(repo, "rm", "-rfq", ".")
    _commit(repo, "orphan.md", f"ssh with sudo `{FAKE}`\n")

    proc = _run(repo, "--base", "main")

    assert proc.returncode == 1 and "orphan.md:1: sudo-literal" in proc.stdout


# ---------------------------------------------------------------------------- the allow-list


def test_the_digest_is_the_first_12_hex_chars_of_the_literals_sha256():
    # the FIPS 180 test vector for "abc"; the allow-list entries of the repository depend on it
    assert cns._digest("abc") == "ba7816bf8f01"
    assert cns._digest(FAKE) == sha12(FAKE)


def test_an_entry_needs_a_path_and_twelve_hex_chars_or_the_scan_stops(repo: Path):
    (repo / "spec").mkdir()
    allow = repo / "spec" / "secret_scan_allow.txt"
    for bad in ("docs/a.md", "docs/a.md:abc", "docs/a.md:ABCDEF012345", "docs/a.md:0123456789abcd"):
        allow.write_text(f"# header\n{bad}  # reason\n")
        proc = _run(repo, "--base", "HEAD")
        assert proc.returncode == 2, bad
        assert "could not scan" in proc.stderr and "secret_scan_allow.txt:2" in proc.stderr


def test_a_comment_or_a_blank_line_is_not_an_entry(repo: Path):
    (repo / "spec").mkdir()
    (repo / "spec" / "secret_scan_allow.txt").write_text(
        f"# header\n\n   \ndocs/a.md:{sha12(FAKE)}   # why\n# trailing\n"
    )

    assert cns.load_allow(repo / "spec" / "secret_scan_allow.txt") == {f"docs/a.md:{sha12(FAKE)}"}


def test_every_entry_of_the_reviewed_allow_list_is_well_formed_and_carries_a_reason():
    """Adding an entry is the documented way to excuse a reviewed false positive, so the test
    does not count them: it checks that each one can be reviewed (a path, a digest, and why)."""
    entries = []
    for number, raw in enumerate(cns.ALLOW_FILE.read_text().splitlines(), start=1):
        body, _, reason = raw.partition("#")
        if body.strip():
            entries.append((number, body.strip(), reason.strip()))

    assert entries, "the historic leak below must stay excused while its commit is in the history"
    for number, entry, reason in entries:
        assert cns._ENTRY.match(entry), f"{cns.ALLOW_FILE.name}:{number}: {entry!r}"
        assert len(reason) >= 10, f"{cns.ALLOW_FILE.name}:{number}: say why {entry!r} is excused"
    assert set(cns.load_allow()) == {e for _, e, _ in entries}


def test_the_known_historic_leak_stays_excused():
    # commit d85d4fc wrote a sudo password into the goal file; it is public history: this entry
    # only keeps the scan of the branch that carries that commit from failing on it
    assert "docs/GOAL-2026-10-24-finish.md:4b9de61dce9c" in cns.load_allow()


def test_the_repository_tree_has_no_credential_literal_outside_the_reviewed_allow_list():
    allow = cns.load_allow()
    found = cns.scan_tree(allow)
    assert [(f.path, f.line_no, f.rule) for f in found] == []
