"""The installer's OPERATOR_AUTH_REQUIRED warning agrees with what the backend enforces.

``scripts/provision_env.py`` used to warn "DISABLED" for any value other than ``1`` (so for
``true``, which the backend now enforces) and said nothing useful about an ambiguous one. It now
reads the flag through ``credential_policy.parse_operator_auth_flag``, the function the backend's
``core/auth_posture.py`` uses, so the two cannot disagree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.src.core.auth_posture import read_operator_auth_flag

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "provision_env.py"
STRONG = "correct horse battery staple"  # spaces on purpose: a passphrase must survive
DISABLED = "operator authentication is DISABLED"
AMBIGUOUS = "not a recognised boolean"

VALUES = [
    "1", "true", "TRUE", "yes", "on", " 1 ", "True",
    "0", "false", "FALSE", "no", "off", " off ", "Off",
    "", "  ", "2", "disabled", "maybe", "y", "n",
]  # fmt: skip


def _run(env_file: Path, value: str) -> subprocess.CompletedProcess:
    env_file.write_text(f"OPERATOR_AUTH_REQUIRED={value}\n")
    env = {k: v for k, v in os.environ.items() if k != "LAWN_BERRY_AUTH_SECRET"}
    env["LAWN_BERRY_OPERATOR_CREDENTIAL"] = STRONG
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--env-file", str(env_file), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "True"])
def test_a_true_token_is_not_called_disabled(tmp_path, value):
    result = _run(tmp_path / ".env", value)
    assert result.returncode == 0, result.stderr
    assert DISABLED not in result.stderr and AMBIGUOUS not in result.stderr, result.stderr


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "Off"])
def test_an_explicit_false_token_is_a_loud_warning(tmp_path, value):
    result = _run(tmp_path / ".env", value)
    assert result.returncode == 0, result.stderr
    assert DISABLED in result.stderr, result.stderr


@pytest.mark.parametrize("value", ["2", "disabled", "maybe", "y", "n"])
def test_an_ambiguous_value_is_flagged_as_enforced_not_as_disabled(tmp_path, value):
    result = _run(tmp_path / ".env", value)
    assert result.returncode == 0, result.stderr
    assert AMBIGUOUS in result.stderr and "ENFORCED" in result.stderr, result.stderr
    assert DISABLED not in result.stderr, result.stderr


def test_the_installer_and_the_backend_classify_every_value_the_same_way(tmp_path):
    """Run the installer's own copy of the policy (loaded by path) against the backend's."""
    import importlib.util

    path = SCRIPT.parents[1] / "backend" / "src" / "core" / "credential_policy.py"
    spec = importlib.util.spec_from_file_location("installer_credential_policy", path)
    installer_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer_policy)
    for value in VALUES + [None]:
        required, recognised = installer_policy.parse_operator_auth_flag(value)
        env = {} if value is None else {"OPERATOR_AUTH_REQUIRED": value}
        backend = read_operator_auth_flag(env)
        assert (required, recognised) == (backend.required, backend.recognised), value
