"""The default/weak policy shared by the backend and the installer.

The values below are the ones that actually shipped or were documented in this repository
(``operator123`` as the code default, ``admin`` from the removed alias, ``emergency123`` from
the docs), plus their cheap variations; a regression here re-opens a public credential.
"""

from __future__ import annotations

import pytest

from backend.src.core.credential_policy import (
    MIN_CREDENTIAL_LENGTH,
    MIN_SIGNING_SECRET_LENGTH,
    assess_operator_credential,
    assess_signing_secret,
    is_known_default_credential,
    known_default_reason,
)

PUBLISHED_DEFAULTS = [
    "operator123",  # the old code default (auth_service.py, rest_v1.py)
    "Operator123",
    "OPERATOR123!",
    "operator",
    "admin",  # the removed admin/admin alias
    "admin123",
    "admin2025",
    "emergency123",  # docs/authentication-config.md
    "password",
    "Password1!",
    "changeme",
    "change-me",
    "  operator123  ",
    "lawnberry",
]

STRONG = [
    "correct-horse-battery-staple",
    "correct horse battery staple",
    "Mower-Pi-Shed-Gate-2026",
    "password-correct-horse-9",  # a default word inside a passphrase is fine
    "k7#Qp9vL2$xT",
]


@pytest.mark.parametrize("value", PUBLISHED_DEFAULTS)
def test_published_defaults_are_recognised(value):
    assert known_default_reason(value) is not None
    assert is_known_default_credential(value)
    assert any("default" in p for p in assess_operator_credential(value))


@pytest.mark.parametrize("value", STRONG)
def test_strong_credentials_are_not_defaults_and_pass(value):
    assert known_default_reason(value) is None
    assert not is_known_default_credential(value)
    assert assess_operator_credential(value) == []


@pytest.mark.parametrize("value", [None, "", "   ", "\t"])
def test_empty_credentials_are_refused(value):
    assert assess_operator_credential(value) == ["is empty"]
    assert known_default_reason(value) is None  # empty is its own problem, not a "default"


def test_short_and_repetitive_credentials_are_weak():
    short = assess_operator_credential("Xk9#mQ2$vL")  # 10 chars, otherwise fine
    assert any(f"shorter than {MIN_CREDENTIAL_LENGTH}" in p for p in short)
    repetitive = assess_operator_credential("abababababababab")  # long but 2 distinct chars
    assert any("distinct characters" in p for p in repetitive)


def test_whitespace_and_line_breaks_are_paste_errors():
    assert any("whitespace" in p for p in assess_operator_credential(" correct-horse-battery "))
    multi = assess_operator_credential("correct-horse-battery\nstaple-extra-line")
    assert any("line break" in p for p in multi)
    assert any("control characters" in p for p in assess_operator_credential("correct\x00horse-9"))


def test_every_reason_is_reported_not_just_the_first():
    reasons = assess_operator_credential("admin")
    assert any(f"shorter than {MIN_CREDENTIAL_LENGTH}" in p for p in reasons)
    assert any("default" in p for p in reasons)


def test_signing_secret_policy():
    assert assess_signing_secret("a1b2c3d4" * 8) == []  # 64 hex-ish chars
    assert any(
        f"shorter than {MIN_SIGNING_SECRET_LENGTH}" in p for p in assess_signing_secret("s3cr3t")
    )
    assert assess_signing_secret(None) == ["is empty"]
    assert any("distinct" in p for p in assess_signing_secret("a" * 64))
    assert any("default" in p for p in assess_signing_secret("change-me"))
