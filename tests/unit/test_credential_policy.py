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


# ---- SEC20-11: the gate is structural, not a list of exact values -----------------------------

# The review's reproduction (all of these used to pass) and the usual disguises of a default.
TRIVIAL_VARIANTS = [
    "123456789012",
    "qwertyuiop12",
    "abcdefghijkl",
    "passwordpassword",
    "Password-Password",
    "lawnberry-pi-2026",
    "operatoroperator",
    "1qaz2wsx3edc",
    "P@ssw0rd-P@ssw0rd",  # disguised with symbols and digits
    "Lawnberry-Pi-2026!",
    "mylawnberry2026",  # a default word plus a little filler
    "0perat0r0perat0r",
    "abcabcabcabcab",  # a block repeated
    "x7q9x7q9x7q9",
]


@pytest.mark.parametrize("value", TRIVIAL_VARIANTS)
def test_trivial_variants_of_a_default_are_refused(value):
    assert assess_operator_credential(value), value
    assert is_known_default_credential(value), value  # what the backend itself checks


@pytest.mark.parametrize("value", ["qwertyuiop12", "passwordpassword", "123456789012"])
def test_the_backend_refuses_a_variant_at_start_up_not_only_the_installer(monkeypatch, value):
    from backend.src.services.auth_service import AuthService

    monkeypatch.delenv("LAWN_BERRY_SIM_OPEN_AUTH", raising=False)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService(operator_credential=value)
    assert service.operator_credential_hash is None  # login disabled, fail closed


def test_filler_around_a_default_word_has_a_limit():
    assert is_known_default_credential("xyzlawnberry")  # three letters of filler: still a default
    assert not is_known_default_credential("xyzwlawnberry")  # four: a different credential


def test_a_passphrase_that_merely_contains_a_default_word_is_fine():
    for value in (
        "password-correct-horse-9",
        "Mower-Pi-Shed-Gate-2026",
        "test-operator-credential",
    ):
        assert assess_operator_credential(value) == [], value


def test_walk_thresholds_sit_where_documented():
    # an eight-character run is below the refused length, nine is not
    assert assess_operator_credential("abcdefgh7q2m") == []
    assert is_known_default_credential("abcdefghi7q2")
    # nine steps in twelve pairs (0.75) pass, nine in eleven (0.82) do not
    assert assess_operator_credential("qwerasdfzxcv7") == []
    assert is_known_default_credential("qwerasdfzxcv")


def test_the_walk_check_is_not_applied_to_the_signing_secret():
    """A 64-character generated secret may contain a long run by chance; refusing it would make
    the installer fail at random."""
    secret = "abcdefghijklmnop" + "7f3a9c1e5b2d8046" * 3
    assert assess_signing_secret(secret) == []
    assert any("keyboard walk" in p for p in assess_operator_credential(secret[:16]))


def test_generated_credentials_are_not_refused_by_the_new_rules():
    """The new rules look at structure only: random text must pass them. Fixed seed, so the
    sample (and the answer) never changes; the measured rate over 300 000 per generator was
    below 1 in 20 000 for the walk check and 0 for the dictionary and repetition checks."""
    import random
    import string

    rng = random.Random(20261024)
    alnum = string.ascii_letters + string.digits
    generators = {
        "hex12": lambda: "".join(rng.choice("0123456789abcdef") for _ in range(12)),
        "alnum12": lambda: "".join(rng.choice(alnum) for _ in range(12)),
        "lower12": lambda: "".join(rng.choice(string.ascii_lowercase) for _ in range(12)),
        "digits12": lambda: "".join(rng.choice(string.digits) for _ in range(12)),
        "urlsafe16": lambda: "".join(rng.choice(alnum + "-_") for _ in range(16)),
        "printable20": lambda: "".join(rng.choice(alnum + string.punctuation) for _ in range(20)),
    }
    for name, make in generators.items():
        refused = [v for v in (make() for _ in range(12000)) if known_default_reason(v)]
        assert refused == [], (name, refused[:3])
    for _ in range(3000):
        assert assess_signing_secret(f"{rng.getrandbits(256):064x}") == []
