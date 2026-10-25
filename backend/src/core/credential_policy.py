"""Operator-credential and signing-secret policy.

One definition of "default or weak", shared by the backend and the installer:

* ``backend/src/services/auth_service.py`` never falls back to a built-in credential, and
  refuses (login disabled, E-stop untouched) a published default outside ``SIM_MODE``;
* ``scripts/provision_env.py``, run by ``systemd/install_services.sh``, refuses to finish an
  install with an empty, default or weak operator credential or a short signing secret.

Standard library only and import-light on purpose: the installer loads this file by path
under the system ``python3``, before any virtualenv exists.
"""

from __future__ import annotations

import re

MIN_CREDENTIAL_LENGTH = 12
MIN_CREDENTIAL_DISTINCT_CHARS = 5
MIN_SIGNING_SECRET_LENGTH = 32
MIN_SIGNING_SECRET_DISTINCT_CHARS = 8

# Words that are published in this repository (the old ``operator123`` code default, the
# removed ``admin``/``admin`` alias, the documented "emergency123") or are the usual
# factory defaults. A credential that is one of these, optionally followed by digits or
# punctuation ("Operator123!", "admin2025"), is a default for our purposes.
_DEFAULT_BASE_WORDS = frozenset(
    {
        "operator",
        "admin",
        "administrator",
        "root",
        "password",
        "passwd",
        "changeme",
        "change-me",
        "changeit",
        "letmein",
        "welcome",
        "secret",
        "default",
        "emergency",
        "lawnberry",
        "lawnberrypi",
        "raspberry",
        "mower",
    }
)

_TRAILING_NON_LETTERS = re.compile(r"[^a-z]+$")
_NON_LETTERS = re.compile(r"[^a-z]+")

# Words a person disguises with digits and symbols ("P@ssw0rd", "0perat0r", "L4wnberry"). Applied
# before the non-letters are dropped, so the disguised and the plain spelling are one word.
_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
)

# Dictionary for the "made of default words" check: the base words above (separators removed) and
# the usual first guesses against a shared password. At least four letters each, so a short word
# cannot cover a random string by accident.
_DEFAULT_DICTIONARY = frozenset(
    {re.sub(r"[^a-z]", "", word) for word in _DEFAULT_BASE_WORDS}
    | {
        "qwerty",
        "qwertyuiop",
        "asdfgh",
        "asdfghjkl",
        "zxcvbn",
        "iloveyou",
        "monkey",
        "dragon",
        "sunshine",
        "princess",
        "football",
        "baseball",
        "trustno",
        "master",
        "login",
        "shadow",
        "superman",
        "batman",
        "mustang",
        "hello",
        "freedom",
        "whatever",
        "starwars",
        "access",
        "flower",
        "loveme",
        "tractor",
        "lawnmower",
        "robot",
    }
)
_LONGEST_WORD = max(len(word) for word in _DEFAULT_DICTIONARY)
# A credential that is default words plus at most this many other letters ("mylawnberry") is one.
_MAX_LEFTOVER_LETTERS = 3
_MIN_COVERED_LETTERS = 4

# Keyboard rows and columns, and the alphabet: two characters that sit next to each other in one of
# these (either direction) are a "step". Walks and runs ("qwertyuiop12", "1qaz2wsx3edc",
# "abcdefghijkl", "123456789012") are almost all steps; random text is not (about 1 pair in 15).
_SEQUENCES = (
    "1234567890",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "abcdefghijklmnopqrstuvwxyz",
    "1qaz",
    "2wsx",
    "3edc",
    "4rfv",
    "5tgb",
    "6yhn",
    "7ujm",
    "8ik",
    "9ol",
    "0p",
)
_LONGEST_RUN_REFUSED = 9
_STEP_FRACTION_REFUSED = 0.8


def _covered_by_default_words(letters: str) -> bool:
    """Is ``letters`` default words with at most a few letters of filler (shortest cover)?"""
    n = len(letters)
    leftover = [0] * (n + 1)  # fewest letters not covered by a word, per prefix length
    for end in range(1, n + 1):
        best = leftover[end - 1] + 1
        for size in range(_MIN_COVERED_LETTERS, min(end, _LONGEST_WORD) + 1):
            if letters[end - size : end] in _DEFAULT_DICTIONARY:
                best = min(best, leftover[end - size])
        leftover[end] = best
    return leftover[n] <= _MAX_LEFTOVER_LETTERS and n - leftover[n] >= _MIN_COVERED_LETTERS


def _is_repeated_block(text: str) -> bool:
    """Is ``text`` a block repeated (``abcabcabc``, ``passwordpassword``, ``1212121212``)?"""
    n = len(text)
    return any(text == (text[:size] * (n // size + 1))[:n] for size in range(1, n // 2 + 1))


def _is_walk(text: str) -> bool:
    """Is ``text`` mostly a keyboard walk or an alphabet or number run?"""
    chars = [c for c in text.lower() if c.isalnum()]
    if len(chars) < 8:
        return False
    steps = [
        any(a + b in row or b + a in row for row in _SEQUENCES)
        for a, b in zip(chars, chars[1:], strict=False)
    ]
    longest = run = 1
    for step in steps:
        run = run + 1 if step else 1
        longest = max(longest, run)
    return longest >= _LONGEST_RUN_REFUSED or sum(steps) / len(steps) >= _STEP_FRACTION_REFUSED


def known_default_reason(value: str | None, *, patterns: bool = True) -> str | None:
    """Why ``value`` is a published or trivially derived default; ``None`` if it is not.

    Beyond the exact published values (plus digits or punctuation) it catches the cheap variants:
    default words repeated, joined, separated or disguised with digits and symbols
    (``passwordpassword``, ``Lawnberry-Pi-2026``, ``P@ssw0rd!``, ``mylawnberry``), a block repeated
    over and over, and (``patterns``) keyboard walks and alphabet or number runs
    (``qwertyuiop12``, ``1qaz2wsx3edc``, ``123456789012``). Random text is not caught: nothing
    here looks at length or character classes, only at structure, so a generated credential
    passes (``tests/unit/test_credential_policy.py`` pins the false-rejection rate at zero over
    a large random sample). ``patterns=False`` skips the repetition and walk checks, which target
    human-chosen credentials: the signing secret is generated (64 hex characters, where a long
    step run by chance is not that rare) and existing deployments and tests set simple repeated
    ones, so only the default-word checks apply to it.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    base = _TRAILING_NON_LETTERS.sub("", text)
    if base in _DEFAULT_BASE_WORDS or text in _DEFAULT_BASE_WORDS:
        return "is a published or factory default (or one plus digits or punctuation)"
    for spelling in (text, text.translate(_LEET)):
        if _covered_by_default_words(_NON_LETTERS.sub("", spelling)):
            return "is made of published or factory default words (repeated, joined or disguised)"
    if not patterns:
        return None
    if len(text) >= 6 and _is_repeated_block(text):
        return "is a short block repeated over and over"
    if _is_walk(text):
        return "is a keyboard walk or an alphabet or number sequence"
    return None


def is_known_default_credential(value: str | None) -> bool:
    """``True`` for the values the backend must never accept as the operator credential."""
    return known_default_reason(value) is not None


def assess_operator_credential(value: str | None) -> list[str]:
    """Every reason ``value`` must not protect the mower; an empty list means acceptable."""
    if value is None or value.strip() == "":
        return ["is empty"]
    problems: list[str] = []
    if value != value.strip():
        problems.append("has leading or trailing whitespace (paste error?)")
    if "\n" in value or "\r" in value:
        problems.append("contains a line break (extra lines in the credential file?)")
    elif not value.isprintable():
        problems.append("contains control characters")
    if len(value) < MIN_CREDENTIAL_LENGTH:
        problems.append(f"is shorter than {MIN_CREDENTIAL_LENGTH} characters")
    if len(set(value)) < MIN_CREDENTIAL_DISTINCT_CHARS:
        problems.append(f"uses fewer than {MIN_CREDENTIAL_DISTINCT_CHARS} distinct characters")
    default = known_default_reason(value)
    if default:
        problems.append(default)
    return problems


def assess_signing_secret(value: str | None) -> list[str]:
    """Every reason ``value`` must not sign operator tokens; empty list means acceptable."""
    if value is None or value.strip() == "":
        return ["is empty"]
    problems: list[str] = []
    if value != value.strip():
        problems.append("has leading or trailing whitespace")
    if len(value) < MIN_SIGNING_SECRET_LENGTH:
        problems.append(f"is shorter than {MIN_SIGNING_SECRET_LENGTH} characters")
    if len(set(value)) < MIN_SIGNING_SECRET_DISTINCT_CHARS:
        problems.append(f"uses fewer than {MIN_SIGNING_SECRET_DISTINCT_CHARS} distinct characters")
    default = known_default_reason(value, patterns=False)
    if default:
        problems.append(default)
    return problems


# ---- OPERATOR_AUTH_REQUIRED: one parser for the backend and the installer ---------------------

AUTH_FLAG_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
AUTH_FLAG_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})


def parse_operator_auth_flag(value: str | None) -> tuple[bool, bool]:
    """``(required, recognised)`` for an ``OPERATOR_AUTH_REQUIRED`` value (``None`` = unset).

    Fail closed: only ``0``, ``false``, ``no`` and ``off`` (any case, surrounding blanks ignored)
    disable operator authentication. Unset, ``1``, ``true``, ``yes``, ``on`` and every other
    value (an empty ``OPERATOR_AUTH_REQUIRED=`` line, ``2``, ``disabled``, a typo) keep it
    required; ``recognised`` is ``False`` for those other values. The backend
    (``core/auth_posture.py``) and ``scripts/provision_env.py`` both read the flag through this
    function, so the installer's warning can never disagree with what the backend enforces.
    """
    if value is None:
        return True, True
    token = value.strip().lower()
    if token in AUTH_FLAG_FALSE_TOKENS:
        return False, True
    return True, token in AUTH_FLAG_TRUE_TOKENS
