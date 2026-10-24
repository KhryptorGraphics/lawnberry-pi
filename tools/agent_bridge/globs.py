"""Repository path normalisation and the glob dialect shared with ``omp_guard.mjs``.

Dialect (anchored at the repository root, POSIX separators):

* ``*`` and ``?`` never match ``/``;
* ``**`` matches any number of path segments, including none (``docs/**`` also matches ``docs``);
* ``**/`` at the start or in the middle matches zero or more whole directories;
* every other character is literal.

The JavaScript guard implements the same translation; the shared test vectors in
``tests/agent_bridge/fixtures/glob_vectors.json`` are checked against both implementations.
Matching is case-sensitive, except where a caller asks for ``casefold=True``: deny lists use it
so that ``claude.md`` cannot slip past a protected ``CLAUDE.md`` on a file system that ignores case.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

__all__ = [
    "HIDDEN_CATEGORIES",
    "PathError",
    "fold_case",
    "glob_to_regex",
    "hidden_characters",
    "match_any",
    "normalize_glob",
    "normalize_repo_path",
]

_MAX_PATH = 400
_FORBIDDEN = re.compile(r"[\x00-\x1f\x7f\\]")


@lru_cache(maxsize=2048)
def fold_case(value: str) -> str:
    """Caseless canonical form of a path or glob, for comparing against deny lists.

    Case-insensitive file systems (NTFS, APFS, HFS+, ext4 with casefold) treat more than ASCII
    letters as equal: ``\u017f`` (long s) is an ``S`` to NTFS, dotless ``\u0131`` an ``I``, the
    Kelvin sign a ``K``, ``\u00df`` and capital ``\u1e9e`` are ``ss``, and ``e`` plus a combining
    acute is ``\u00e9``.  Lower case, upper case, lower case again, between two NFC
    normalisations, maps every such spelling onto one form (plain ``lower().upper()`` alone is
    not idempotent for ``\u1e9e``).  Glob syntax (``*``, ``?``, ``/``) is unaffected.
    ``foldCase`` in ``omp_guard.mjs`` takes the identical steps; the shared tests keep the two
    in lock-step.
    """
    text = unicodedata.normalize("NFC", value)
    return unicodedata.normalize("NFC", text.lower().upper().lower())


# What a file name may not contain besides the C0 controls and backslashes of ``_FORBIDDEN``: C1
# controls, format characters (zero-width, bidirectional marks and overrides, BOM, soft hyphen,
# tags), line and paragraph separators, and lone surrogates (``surrogateescape`` turns the bytes of
# a name that is not UTF-8 into those, and a name with one cannot be encoded for GitHub).
HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def hidden_characters(text: str, limit: int = 5) -> list[str]:
    """``U+XXXX`` of the first ``limit`` characters of ``text`` in :data:`HIDDEN_CATEGORIES`."""
    found: list[str] = []
    for char in text:
        if unicodedata.category(char) in HIDDEN_CATEGORIES:
            found.append(f"U+{ord(char):04X}")
            if len(found) >= limit:
                break
    return found


class PathError(ValueError):
    """A path or glob is not a safe repository-relative reference."""


def normalize_repo_path(raw: str) -> str:
    """Return a canonical repo-relative POSIX path or raise :class:`PathError`."""
    if not isinstance(raw, str):
        raise PathError("path must be a string")
    value = raw.strip()
    if not value:
        raise PathError("path is empty")
    if len(value) > _MAX_PATH:
        raise PathError("path is too long")
    if _FORBIDDEN.search(value):
        raise PathError("path contains control characters or backslashes")
    hidden = hidden_characters(value)
    if hidden:
        raise PathError(
            f"path contains control, invisible or non-UTF-8 characters ({', '.join(hidden)})"
        )
    if value.startswith(("/", "~")) or re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", value):
        raise PathError(f"path must be repository-relative: {value!r}")
    normalized = posixpath.normpath(value)
    if normalized in {".", ""}:
        raise PathError("path refers to the repository root")
    if normalized == ".." or normalized.startswith("../"):
        raise PathError(f"path escapes the repository: {value!r}")
    return normalized


def normalize_glob(raw: str) -> str:
    """Validate a glob pattern (same rules as paths, wildcards allowed) and canonicalise it.

    Empty and ``.`` segments are dropped (``a//b`` -> ``a/b``, ``./a`` -> ``a``) and a trailing
    slash means "everything below" (``scripts/`` -> ``scripts/**``), so the result is never a
    pattern that can only match non-canonical paths, i.e. nothing at all.
    """
    if not isinstance(raw, str):
        raise PathError("glob must be a string")
    value = raw.strip()
    if not value:
        raise PathError("glob is empty")
    if len(value) > _MAX_PATH:
        raise PathError("glob is too long")
    if _FORBIDDEN.search(value):
        raise PathError("glob contains control characters or backslashes")
    if value.startswith(("/", "~")):
        raise PathError(f"glob must be repository-relative: {value!r}")
    if any(segment == ".." for segment in value.split("/")):
        raise PathError(f"glob escapes the repository: {value!r}")
    segments = [segment for segment in value.split("/") if segment not in ("", ".")]
    if not segments:
        raise PathError("glob refers to the repository root")
    canonical = "/".join(segments)
    return f"{canonical}/**" if value.endswith("/") else canonical


@lru_cache(maxsize=512)
def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob in the shared dialect to an anchored regular expression."""
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if pattern.startswith("**", i):
                j = i + 2
                if j < n and pattern[j] == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                    continue
                # trailing/inner "**": swallow a preceding "/" so "docs/**" matches "docs" too
                if out and out[-1] == "/":
                    out.pop()
                    out.append("(?:/.*)?")
                else:
                    out.append(".*")
                i = j
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + r"\Z")


def match_any(path: str, patterns: Iterable[str], *, casefold: bool = False) -> bool:
    """True when ``path`` (normalised, repo-relative) matches at least one glob.

    ``casefold=True`` compares both sides through :func:`fold_case`.  Deny lists use it (a file
    system that ignores case opens ``claude.md`` for ``CLAUDE.md``); allow lists stay exact.
    """
    if casefold:
        folded = fold_case(path)
        return any(glob_to_regex(fold_case(p)).match(folded) for p in patterns)
    return any(glob_to_regex(p).match(path) for p in patterns)
