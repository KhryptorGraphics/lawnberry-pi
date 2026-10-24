"""scripts/provision_env.py: persistent signing secret, credential gate, env-file hygiene.

Everything runs the real script (CLI) against a temp directory; the pure helpers are
exercised directly. No test prints, and none may leak, the credential or the secret.
"""

from __future__ import annotations

import importlib.util
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "provision_env.py"
STRONG = "correct horse battery staple"  # spaces on purpose: a passphrase must survive

_spec = importlib.util.spec_from_file_location("provision_env_under_test", SCRIPT)
assert _spec and _spec.loader
pe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pe  # dataclasses resolves string annotations through sys.modules
_spec.loader.exec_module(pe)


def run(env_file: Path, *args: str, credential: str | None = None, cwd: Path | None = None):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LAWN_BERRY_OPERATOR_CREDENTIAL", "LAWN_BERRY_AUTH_SECRET"}
    }
    if credential is not None:
        env["LAWN_BERRY_OPERATOR_CREDENTIAL"] = credential
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--env-file", str(env_file), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=60,
    )


def values(env_file: Path) -> dict[str, str | None]:
    entries = pe.parse_entries(pe.read_lines(env_file))
    return {e.key: e.value for e in entries}


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---- parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("abc", "abc"),
        ("  abc  ", "abc"),
        ("", ""),
        ("'a b $c #d \"e\"'", 'a b $c #d "e"'),
        ('"a b"', "a b"),
        (r'"a \"b\" \\ c"', 'a "b" \\ c'),
        ("a#b", "a#b"),
    ],
)
def test_parse_value_matches_what_systemd_reads(raw, expected):
    assert pe.parse_value(raw) == (expected, None)


@pytest.mark.parametrize(
    "raw",
    [
        "'unterminated",
        "'it's'",
        '"unterminated',
        r'"bad \n escape"',
        'a"b',
        "a'b",
        r"a\b",
        "'a' # c",
    ],
)
def test_parse_value_refuses_what_it_cannot_interpret_safely(raw):
    value, problem = pe.parse_value(raw)
    assert value is None and problem


def test_upsert_replaces_in_place_drops_duplicates_and_appends_once():
    lines = [
        "# NTRIP",
        "NTRIP_HOST=caster.example",
        "LAWN_BERRY_AUTH_SECRET=old1",
        "THOR_UPLOAD_ENABLED=1",
        "LAWN_BERRY_AUTH_SECRET=old2",
        "",
    ]
    out = pe.upsert(
        lines,
        {
            "LAWN_BERRY_AUTH_SECRET": "LAWN_BERRY_AUTH_SECRET=new",
            "LAWN_BERRY_OPERATOR_CREDENTIAL": "LAWN_BERRY_OPERATOR_CREDENTIAL='x'",
        },
    )
    assert out[:4] == [
        "# NTRIP",
        "NTRIP_HOST=caster.example",
        "THOR_UPLOAD_ENABLED=1",
        "LAWN_BERRY_AUTH_SECRET=new",
    ]
    assert out.count(pe.MANAGED_HEADER) == 1
    assert out[-1] == "LAWN_BERRY_OPERATOR_CREDENTIAL='x'"
    assert "LAWN_BERRY_AUTH_SECRET=old1" not in out


# ---- the credential gate -----------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "operator123",
        "Operator123!",
        "admin",
        "password1234",
        "short-pw-1",
        "aaaaaaaaaaaaaaaa",
        "   ",
    ],
)
def test_default_and_weak_credentials_are_refused_and_nothing_is_written(tmp_path, bad):
    env_file = tmp_path / ".env"
    result = run(env_file, credential=bad)
    assert result.returncode == 3, result.stderr
    assert "REFUSED" in result.stderr
    assert not env_file.exists(), "a refusal must not create the file"
    assert [p.name for p in tmp_path.iterdir()] == [], "nor leave a temp file behind"
    if bad.strip():
        assert bad not in result.stdout + result.stderr, "the credential must never be echoed"


def test_no_credential_anywhere_is_refused_with_instructions(tmp_path):
    result = run(tmp_path / ".env")
    assert result.returncode == 3
    assert "no operator credential is configured" in result.stderr
    assert "--operator-credential-file" in result.stderr  # tells the owner what to do
    assert not (tmp_path / ".env").exists()


def test_a_default_already_in_the_env_file_is_refused_too(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LAWN_BERRY_OPERATOR_CREDENTIAL=operator123\n")
    env_file.chmod(0o600)
    before = env_file.read_bytes()
    result = run(env_file)
    assert result.returncode == 3 and "REFUSED" in result.stderr
    assert env_file.read_bytes() == before


def test_a_credential_the_env_file_cannot_store_is_refused(tmp_path):
    for bad in ("correct-horse's-battery-staple", "correct-horse\\battery-staple"):
        result = run(tmp_path / ".env", credential=bad)
        assert result.returncode == 3 and "cannot be stored" in result.stderr
    assert not (tmp_path / ".env").exists()


def test_a_good_credential_and_a_generated_secret_are_written_0600(tmp_path):
    env_file = tmp_path / ".env"
    result = run(env_file, credential=STRONG)
    assert result.returncode == 0, result.stderr
    assert mode(env_file) == 0o600
    got = values(env_file)
    assert got["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG, "round-trips through the parser"
    assert re.fullmatch(r"[0-9a-f]{64}", got["LAWN_BERRY_AUTH_SECRET"] or "")
    assert STRONG not in result.stdout + result.stderr
    assert got["LAWN_BERRY_AUTH_SECRET"] not in result.stdout + result.stderr
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


def test_secret_persists_across_runs_and_the_file_is_not_rewritten(tmp_path):
    env_file = tmp_path / ".env"
    assert run(env_file, credential=STRONG).returncode == 0
    secret = values(env_file)["LAWN_BERRY_AUTH_SECRET"]
    before = (env_file.read_bytes(), env_file.stat().st_mtime_ns, env_file.stat().st_ino)

    again = run(env_file)  # credential now comes from the file
    assert again.returncode == 0, again.stderr
    assert values(env_file)["LAWN_BERRY_AUTH_SECRET"] == secret
    assert (env_file.read_bytes(), env_file.stat().st_mtime_ns, env_file.stat().st_ino) == before

    same = run(env_file, credential=STRONG)  # identical credential offered again: still a no-op
    assert same.returncode == 0
    assert env_file.read_bytes() == before[0]


def test_rotate_replaces_only_the_secret(tmp_path):
    env_file = tmp_path / ".env"
    assert run(env_file, credential=STRONG).returncode == 0
    first = values(env_file)
    assert run(env_file, "--rotate-auth-secret").returncode == 0
    second = values(env_file)
    assert second["LAWN_BERRY_AUTH_SECRET"] != first["LAWN_BERRY_AUTH_SECRET"]
    assert second["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG


def test_a_new_credential_replaces_the_old_one_and_keeps_the_secret(tmp_path):
    env_file = tmp_path / ".env"
    assert run(env_file, credential=STRONG).returncode == 0
    secret = values(env_file)["LAWN_BERRY_AUTH_SECRET"]
    assert run(env_file, credential="another-strong-credential-7").returncode == 0
    got = values(env_file)
    assert got["LAWN_BERRY_OPERATOR_CREDENTIAL"] == "another-strong-credential-7"
    assert got["LAWN_BERRY_AUTH_SECRET"] == secret
    assert (
        sum(
            1
            for e in pe.parse_entries(pe.read_lines(env_file))
            if e.key.startswith("LAWN_BERRY_OPERATOR")
        )
        == 1
    )


def test_credential_file_source_strips_one_newline_and_warns_when_readable(tmp_path):
    cred = tmp_path / "cred"
    cred.write_text(STRONG + "\n")
    cred.chmod(0o644)
    env_file = tmp_path / ".env"
    result = run(env_file, "--credential-file", str(cred))
    assert result.returncode == 0, result.stderr
    assert values(env_file)["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG
    assert "readable by other users" in result.stderr

    cred.write_text("operator123\n")
    refused = run(tmp_path / ".env2", "--credential-file", str(cred))
    assert refused.returncode == 3 and "credential file" in refused.stderr

    missing = run(tmp_path / ".env3", "--credential-file", str(tmp_path / "nope"))
    assert missing.returncode == 3 and "cannot read the credential file" in missing.stderr

    cred.write_text(STRONG + "\nsecond line\n")
    multi = run(tmp_path / ".env4", "--credential-file", str(cred))
    assert multi.returncode == 3 and "line break" in multi.stderr


def test_the_credential_file_wins_over_the_environment(tmp_path):
    cred = tmp_path / "cred"
    cred.write_text("from-the-file-credential-1\n")
    cred.chmod(0o600)
    env_file = tmp_path / ".env"
    result = run(env_file, "--credential-file", str(cred), credential=STRONG)
    assert result.returncode == 0
    assert values(env_file)["LAWN_BERRY_OPERATOR_CREDENTIAL"] == "from-the-file-credential-1"


# ---- the signing secret ------------------------------------------------------------------


def test_a_short_existing_secret_is_refused_unless_rotated(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"LAWN_BERRY_AUTH_SECRET=short\nLAWN_BERRY_OPERATOR_CREDENTIAL='{STRONG}'\n"
    )
    env_file.chmod(0o600)
    before = env_file.read_bytes()
    refused = run(env_file)
    assert refused.returncode == 3 and "shorter than 32" in refused.stderr
    assert env_file.read_bytes() == before

    rotated = run(env_file, "--rotate-auth-secret")
    assert rotated.returncode == 0, rotated.stderr
    assert len(values(env_file)["LAWN_BERRY_AUTH_SECRET"] or "") == 64


def test_an_empty_secret_line_is_replaced(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LAWN_BERRY_AUTH_SECRET=\n")
    result = run(env_file, credential=STRONG)
    assert result.returncode == 0, result.stderr
    assert len(values(env_file)["LAWN_BERRY_AUTH_SECRET"] or "") == 64


def test_an_unparseable_secret_line_is_refused_not_guessed(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LAWN_BERRY_AUTH_SECRET='half-quoted\n")
    result = run(env_file, credential=STRONG)
    assert result.returncode == 3 and "LAWN_BERRY_AUTH_SECRET" in result.stderr


# ---- hygiene -----------------------------------------------------------------------------


def test_other_lines_survive_byte_for_byte_and_the_mode_is_tightened(tmp_path):
    env_file = tmp_path / ".env"
    original = (
        "# my notes\nNTRIP_HOST=caster.example\nNTRIP_PASSWORD='p@ss # word'\n\n"
        "THOR_UPLOAD_ENABLED=1\n"
    )
    env_file.write_text(original)
    env_file.chmod(0o644)
    result = run(env_file, credential=STRONG)
    assert result.returncode == 0, result.stderr
    assert "tightened to 0600" in result.stdout
    assert mode(env_file) == 0o600
    text = env_file.read_text()
    assert text.startswith(original), "the owner's lines are untouched and stay first"
    assert values(env_file)["NTRIP_PASSWORD"] == "p@ss # word"


def test_a_world_readable_env_file_is_tightened_even_when_nothing_else_changes(tmp_path):
    env_file = tmp_path / ".env"
    assert run(env_file, credential=STRONG).returncode == 0
    env_file.chmod(0o644)
    result = run(env_file)
    assert result.returncode == 0
    assert mode(env_file) == 0o600


def test_check_writes_nothing_and_reports_what_would_happen(tmp_path):
    env_file = tmp_path / ".env"
    result = run(env_file, "--check", credential=STRONG)
    assert result.returncode == 0
    assert "would be created" in result.stdout and "generated" in result.stdout
    assert not env_file.exists()
    assert run(env_file, "--check").returncode == 3  # and the gate applies to --check


def test_auth_disabled_in_the_env_file_is_a_loud_warning(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPERATOR_AUTH_REQUIRED=0\n")
    result = run(env_file, credential=STRONG)
    assert result.returncode == 0
    assert "operator authentication is DISABLED" in result.stderr


def test_a_symlinked_env_file_keeps_its_link_and_target_mode(tmp_path):
    real = tmp_path / "real.env"
    real.write_text("NTRIP_HOST=x\n")
    link = tmp_path / ".env"
    link.symlink_to(real)
    assert run(link, credential=STRONG).returncode == 0
    assert link.is_symlink(), "the atomic replace must write through the link"
    assert "LAWN_BERRY_AUTH_SECRET" in real.read_text()
    assert mode(real) == 0o600


def test_the_env_example_template_is_a_valid_starting_point(tmp_path):
    """Copying .env.example to .env is the documented first step: its empty credential and
    secret lines must be refused (credential) and filled in (secret), not mis-parsed."""
    template = Path(__file__).resolve().parents[2] / ".env.example"
    env_file = tmp_path / ".env"
    env_file.write_text(template.read_text())
    refused = run(env_file)
    assert refused.returncode == 3 and "is empty" in refused.stderr

    assert run(env_file, credential=STRONG).returncode == 0
    got = values(env_file)
    assert got["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG
    assert re.fullmatch(r"[0-9a-f]{64}", got["LAWN_BERRY_AUTH_SECRET"] or "")
    assert got["NTRIP_SERIAL_DEVICE"] == "/dev/ttyAMA0", "the rest of the template is untouched"
    names = [e.key for e in pe.parse_entries(pe.read_lines(env_file))]
    assert (
        names.count("LAWN_BERRY_AUTH_SECRET") == names.count("LAWN_BERRY_OPERATOR_CREDENTIAL") == 1
    )
