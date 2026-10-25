"""Settings that hold secrets are write-only: which names are secrets and how one is shown.

The settings stores (``api/routers/settings.py``) accept any key the operator PUTs, and the
Settings view keeps its tunnel tokens and map key in them. They used to be read back verbatim
from an open GET (SEC20-01), so anyone who could reach the port got the Cloudflare tunnel token,
the ngrok authtoken and the Google Maps key. The rule now:

* a value stored under a secret-looking name is NEVER put in a response;
* the operator-only reads show that it is set (``<name>_set``) and, for a value long enough that
  four characters give nothing away, the last four (``<name>_last4``) so a form can say which
  token is saved;
* a write keeps the old value when the secret field is empty (an untouched password input),
  replaces it for any other value and clears it for an explicit ``null`` (stored as ``null``,
  which reads as "not set");
* the derived ``<name>_set`` / ``<name>_last4`` keys of a read view that a client echoes back are
  never stored.

Classification is by NAME, at any depth: stores take arbitrary keys, so there is no fixed list
to keep up to date. A secret under a harmless name cannot be found by this module.

Standard library only.
"""

from __future__ import annotations

import re
from typing import Any

#: A tail is shown only for a secret at least this long; for a shorter one four characters
#: would give away a quarter or more of the value.
MIN_LENGTH_FOR_TAIL = 16
TAIL_LENGTH = 4

_SECRET_WORDS = frozenset(
    {
        "token",
        "tokens",
        "secret",
        "secrets",
        "password",
        "passwd",
        "passphrase",
        "passcode",
        "pwd",
        "credential",
        "credentials",
        "apikey",
        "authtoken",
        "privatekey",
        "secretkey",
        "bearer",
        "jwt",
        "psk",
        "webhook",
        "authorization",
        "cookie",
        "sid",
    }
)

# Adjacent words that are a secret together although neither is alone (``auth_level`` and
# ``session_timeout_minutes`` are not secrets).
_SECRET_PAIRS = frozenset(
    {
        ("api", "key"),
        ("private", "key"),
        ("secret", "key"),
        ("access", "key"),
        ("signing", "key"),
        ("encryption", "key"),
        ("backup", "codes"),
        ("recovery", "codes"),
        ("session", "id"),
    }
)

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_HINT_SUFFIXES = ("_set", "_last4")


def _words(name: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(name)]


def is_secret_name(name: object) -> bool:
    """Whether a settings key names a secret (``google_api_key``, ``cloudflareToken``, ...)."""
    if not isinstance(name, str):
        return False
    words = _words(name)
    if not words:
        return False
    if any(word in _SECRET_WORDS for word in words):
        return True
    if words[-1] in {"key", "keys"}:
        return True
    return any(pair in _SECRET_PAIRS for pair in zip(words, words[1:], strict=False))


def is_hint_name(name: object) -> bool:
    """``<secret>_set`` / ``<secret>_last4``: the keys a read view derives, never stored values."""
    if not isinstance(name, str):
        return False
    return any(
        name.endswith(suffix) and is_secret_name(name[: -len(suffix)]) for suffix in _HINT_SUFFIXES
    )


def secret_is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def secret_tail(value: Any) -> str | None:
    """The last :data:`TAIL_LENGTH` characters of a long string secret, else ``None``."""
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= MIN_LENGTH_FOR_TAIL:
            return text[-TAIL_LENGTH:]
    return None


def redact_secrets(value: Any, *, hints: bool = False) -> Any:
    """A copy of ``value`` with every secret-named entry removed, at any depth.

    ``hints=True`` (operator-only reads) adds ``<name>_set`` and, when one is safe to show,
    ``<name>_last4`` next to where the secret was. The default adds nothing: an open read must
    not even say that a secret exists.
    """
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if is_secret_name(key):
                if hints:
                    out[f"{key}_set"] = secret_is_set(item)
                    tail = secret_tail(item)
                    if tail is not None:
                        out[f"{key}_last4"] = tail
                continue
            out[key] = redact_secrets(item, hints=hints)
        return out
    if isinstance(value, list):
        return [redact_secrets(item, hints=hints) for item in value]
    return value


def merge_secret_update(current: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """``{**current, **payload}`` with the write-only rules for the top-level secret fields.

    * a derived ``<secret>_set`` / ``<secret>_last4`` key is dropped (a client echoing the view);
    * a secret field that is empty or blank keeps what is stored;
    * a secret field that is ``None`` clears what is stored (it is stored as ``None``, so the
      operator view reports ``<name>_set: false`` instead of the key vanishing);
    * any other value replaces it, verbatim.
    """
    merged = dict(current)
    for key, value in payload.items():
        if is_hint_name(key):
            continue
        if is_secret_name(key):
            if value is None:
                merged[key] = None
                continue
            if isinstance(value, str) and not value.strip():
                continue
        merged[key] = value
    return merged
