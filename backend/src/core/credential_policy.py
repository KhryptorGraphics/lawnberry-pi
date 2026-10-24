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


def known_default_reason(value: str | None) -> str | None:
    """Why ``value`` is a published or trivially derived default; ``None`` if it is not."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    base = _TRAILING_NON_LETTERS.sub("", text)
    if base in _DEFAULT_BASE_WORDS or text in _DEFAULT_BASE_WORDS:
        return "is a published or factory default (or one plus digits or punctuation)"
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
    default = known_default_reason(value)
    if default:
        problems.append(default)
    return problems
