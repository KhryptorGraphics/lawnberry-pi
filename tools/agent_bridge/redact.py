"""Secret redaction and safe rendering of untrusted text.

Everything the bridge stores, logs or publishes that originates from an agent or from the
network passes through :class:`Redactor`.  Agent text is additionally rendered inside a code
fence (:func:`fenced`) so Markdown, HTML, images and ``@mentions`` stay inert on GitHub.

Three layers, all linear in the size of the text:

* registered secret values ("literals") in any case, in their URL-encoded, JSON-escaped and
  base64 forms (at any of the three alignments a base64 string can have);
* secret-shaped strings: provider tokens, HTTP ``Authorization`` headers, private keys, ...;
* a second look through line breaks and invisible characters, because a secret cut in two by a
  newline or a zero-width space is still a secret.
"""

from __future__ import annotations

import base64
import bisect
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import quote, quote_plus

__all__ = [
    "Redactor",
    "SecretFinding",
    "fenced",
    "strip_control_chars",
    "truncate",
]

# Order matters: each pattern sees the text the previous ones left, so the most specific kinds
# come first and a token inside an ``Authorization`` header is named by its own kind.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "github-token",
        # no upper bound: a cap left the tail of a long token behind the placeholder
        re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}"),
    ),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    # "sk-" also ends ordinary words ("disk-usage-...", "risk-assessment-..."): no letter or digit
    # may precede it.  The other prefixes are distinctive, so they match even when glued to text.
    ("openai-key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("aws-access-key", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    ("slack-token", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("google-oauth-token", re.compile(r"ya29\.[A-Za-z0-9_\-]{20,}")),
    (
        "gitlab-token",
        re.compile(r"gl(?:pat|dt|rt|cbt|ptt|ft|imt|agent|oas|soat)-[A-Za-z0-9_\-]{20,}"),
    ),
    ("npm-token", re.compile(r"npm_[A-Za-z0-9]{20,}")),
    ("stripe-key", re.compile(r"[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("huggingface-token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("groq-key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("nvidia-key", re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}")),
    ("xai-key", re.compile(r"xai-[A-Za-z0-9]{20,}")),
    (
        "private-key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[^\"]*?"
            r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|(?=\")|\Z)"
        ),
    ),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{20,}")),
    (
        # ``Authorization: Basic <base64>``, ``token <anything>`` and the short bearer tokens the
        # rule above leaves; the scheme word must follow the header name, so prose is untouched
        "authorization",
        re.compile(
            r"(?i)(?<![a-z])(?:proxy-)?authorization\\?[\"']?\s*[:=]\s*\\?[\"']?"
            r"(?:basic|bearer|token|negotiate|ntlm)\s+[A-Za-z0-9+/=._~\-]{8,}"
        ),
    ),
    (
        "jwt",
        # a start may not sit inside a longer token run: every run is scanned once (linear time)
        re.compile(
            r"(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]{8,}"
            r"\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
        ),
    ),
    (
        # A bare AWS secret access key: exactly 40 base64 characters standing alone (not inside a
        # longer run, not a dotted name or file name) with a capital, a small letter and a digit,
        # which rules out git object ids, digests and words.  The look-aheads scan at most 40
        # characters from a run's first character, so the cost stays linear.
        "aws-secret-key",
        re.compile(
            r"(?<![A-Za-z0-9/+_\-])(?<![A-Za-z0-9]\.)"
            r"(?=[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+_=\-]|\.[A-Za-z0-9]))"
            r"(?=[^A-Z\s]{0,39}[A-Z])(?=[^a-z\s]{0,39}[a-z])(?=[^0-9\s]{0,39}[0-9])"
            r"[A-Za-z0-9/+]{40}"
        ),
    ),
    (
        "url-credentials",
        re.compile(r"(?i)(?<![a-z0-9+.\-])(?P<scheme>[a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@"),
    ),
)
# Kinds that need their own context (a multi-line block, a URL scheme to keep) are matched on the
# text as written only; every other kind is also matched through line breaks.
_SPLITTABLE = tuple((n, p) for n, p in _PATTERNS if n not in {"private-key", "url-credentials"})

# NAME=value / NAME: value where NAME looks like a credential.  Only the value is replaced.
# The key is the credential word plus a bounded tail of identifier characters; no unbounded
# quantifier sits before it, so scanning stays linear on any input.  Values of 8+ characters
# count; shorter ones only for a password that looks like one (see ``_is_secret_assignment``).
_ASSIGNMENT = re.compile(
    r"(?i)(?P<key>(?:API[_-]?KEY|ACCESS[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"
    r"[A-Z0-9_]{0,64}+)(?P<kq>\\?['\"]?)(?P<sep>\s*[:=]\s*)(?P<quote>\\?['\"]?)"
    r"(?P<value>[^\s'\"\\]{4,})(?P=quote)"
)
_MIN_VALUE = 8
# `password: str` and `password=args.pw` are code; `password=hunter2` and `password=p@ss!` are not.
_LOOKS_LIKE_A_PASSWORD = re.compile(r"[0-9!@#$%^&*~?+]")
# Values that are obviously placeholders must not trigger the assignment rule.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:<[^>]*>|\$\{?[A-Z0-9_]+\}?|\[REDACTED[^\]]*\]|x{4,}|\*{4,}|\.{3,}|none|null)$"
)

# What can cut a secret in two without a reader noticing it: invisible characters (zero-width,
# bidirectional marks, soft hyphen, variation selectors, tags: Unicode Default_Ignorable) and line
# breaks, with the blanks around them.  Possessive quantifiers keep the scan linear.
_INVISIBLE = (
    "\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e"
    "\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\U000e0000-\U000e0fff"
)
_LINE_BREAKS = "\r\n\x0b\x0c\x85\u2028\u2029"
_BREAK = re.compile(
    r"(?<=\S)[ \t]*+["
    + _INVISIBLE
    + _LINE_BREAKS
    + r"][ \t"
    + _INVISIBLE
    + _LINE_BREAKS
    + r"]*+(?=\S)"
)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_MIN_LITERAL = 8


@dataclass(frozen=True)
class SecretFinding:
    """A secret-looking string located in a diff (the value itself is never kept)."""

    kind: str
    path: str
    line: int


def _lower(text: str) -> str:
    """``text.lower()`` with the same length, so positions found in it hold in ``text``."""
    folded = text.lower()
    if len(folded) == len(text):
        return folded
    # a few characters (U+0130, ...) grow when lower-cased: map one character at a time
    return "".join(c if len(c.lower()) != 1 else c.lower() for c in text)


def _base64_forms(data: bytes) -> set[str]:
    """The base64 spellings of ``data`` wherever it sits in a longer string.

    A secret that starts one or two bytes into a base64 group (``user:`` + password) shares its
    first characters with the bytes before it, and its last characters with the bytes after it;
    the characters in between depend on the secret alone.  The standalone forms are kept whole.
    """
    forms: set[str] = set()
    for shift in (0, 1, 2):
        total = shift + len(data)
        first = (0, 2, 3)[shift]
        last = (total // 3) * 4 + (0, 1, 2)[total % 3]
        for encode in (base64.b64encode, base64.urlsafe_b64encode):
            encoded = encode(b"\0" * shift + data).decode("ascii").rstrip("=")
            forms.add(encoded[first:last])
    for encode in (base64.b64encode, base64.urlsafe_b64encode):
        whole = encode(data).decode("ascii")
        forms.update((whole, whole.rstrip("=")))
    return forms


def _literal_forms(value: str) -> tuple[set[str], set[str]]:
    """``(case-insensitive forms, exact forms)`` of one registered secret (8+ characters each)."""
    text = value.strip()
    folded = {value, text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1]}
    folded |= {quote(text, safe=""), quote(text), quote_plus(text, safe=""), quote_plus(text)}
    exact = _base64_forms(text.encode("utf-8"))
    return (
        {_lower(form) for form in folded if len(form.strip()) >= _MIN_LITERAL},
        {form for form in exact if len(form) >= _MIN_LITERAL},
    )


def _occurrences(haystack: str, needle: str, spans: list[tuple[int, int, str]]) -> None:
    start = haystack.find(needle)
    while start != -1:
        spans.append((start, start + len(needle), "literal"))
        start = haystack.find(needle, start + len(needle))


def _splice(text: str, spans: list[tuple[int, int, str]]) -> str:
    """``text`` with every ``(start, end, kind)`` span replaced; overlapping spans become one."""
    if not spans:
        return text
    spans.sort(key=lambda span: (span[0], -span[1]))
    out: list[str] = []
    cursor = 0
    start, end, kind = spans[0]
    for next_start, next_end, next_kind in [*spans[1:], (len(text) + 1, len(text) + 1, "")]:
        if next_start < end:
            end = max(end, next_end)
            continue
        out += [text[cursor:start], f"[REDACTED:{kind}]"]
        cursor = end
        start, end, kind = next_start, next_end, next_kind
    out.append(text[cursor:])
    return "".join(out)


class Redactor:
    """Replace credentials in text with ``[REDACTED:<kind>]`` placeholders."""

    def __init__(self, literals: Iterable[str] = ()) -> None:
        self._exact: list[str] = []
        self._folded: list[str] = []
        for value in literals:
            self.add_literal(value)

    def add_literal(self, value: str | None) -> None:
        """Register one more exact secret value (ignored when too short to be one)."""
        if not isinstance(value, str) or len(value.strip()) < _MIN_LITERAL:
            return
        folded, exact = _literal_forms(value)
        self._folded = sorted(set(self._folded) | folded, key=len, reverse=True)
        self._exact = sorted(set(self._exact) | exact, key=len, reverse=True)

    # ------------------------------------------------------------------ literals
    def _literal_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for needle in self._exact:
            _occurrences(text, needle, spans)
        if self._folded:
            lowered = _lower(text)
            for needle in self._folded:
                _occurrences(lowered, needle, spans)
        return spans

    def _has_literal(self, text: str) -> bool:
        if any(needle in text for needle in self._exact):
            return True
        if not self._folded:
            return False
        lowered = _lower(text)
        return any(needle in lowered for needle in self._folded)

    # ------------------------------------------------------------ cut-in-two secrets
    def _broken_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Secrets of ``text`` that a line break or an invisible character cuts in two.

        The breaks are taken out (only between two visible characters), the literals and the
        splittable patterns are matched on that, and a match counts when it spans a removed
        break; it is returned in the coordinates of ``text``.  Matches inside one piece are left
        to the ordinary scan.
        """
        if _BREAK.search(text) is None:
            return []
        view = _BREAK.sub("", text)
        spans = self._literal_spans(view)
        for name, pattern in _SPLITTABLE:
            spans.extend((m.start(), m.end(), name) for m in pattern.finditer(view))
        if not spans:
            return []
        cuts: list[int] = []  # where each removed break sat in ``view``
        shifts: list[int] = []  # characters removed up to and including each break
        removed = 0
        for found in _BREAK.finditer(text):
            cuts.append(found.start() - removed)
            removed += found.end() - found.start()
            shifts.append(removed)
        crossing: list[tuple[int, int, str]] = []
        for start, end, kind in spans:
            before = bisect.bisect_right(cuts, start)
            through = bisect.bisect_right(cuts, end - 1)
            if through > before:
                first = start + (shifts[before - 1] if before else 0)
                crossing.append((first, end + shifts[through - 1], kind))
        return crossing

    # ---------------------------------------------------------------- redaction
    def redact(self, text: str) -> str:
        """Return ``text`` with known secret literals and secret-shaped strings removed."""
        if not text:
            return text
        text = _splice(text, self._literal_spans(text))
        for name, pattern in _PATTERNS:
            if name == "url-credentials":
                text = pattern.sub(
                    lambda m: f"{m.group('scheme')}[REDACTED:url-credentials]@", text
                )
            else:
                text = pattern.sub(f"[REDACTED:{name}]", text)
        text = _ASSIGNMENT.sub(self._assignment, text)
        return _splice(text, self._broken_spans(text))

    @staticmethod
    def _is_secret_assignment(match: re.Match[str]) -> bool:
        value = match.group("value")
        if _PLACEHOLDER.match(value):
            return False
        # `"maxOutputTokens":32000}},` is JSON structure, not a credential: after a quoted key
        # only a quoted value counts, otherwise numbers and brackets would be eaten.
        if match.group("kq") and not match.group("quote"):
            return False
        if len(value) >= _MIN_VALUE:
            return True
        # a short value is believed for a password only, and only if it looks like one
        if not match.group("key").upper().startswith(("PASSWORD", "PASSWD")):
            return False
        return bool(match.group("quote")) or _LOOKS_LIKE_A_PASSWORD.search(value) is not None

    @staticmethod
    def _assignment(match: re.Match[str]) -> str:
        if not Redactor._is_secret_assignment(match):
            return match.group(0)
        quote_mark = match.group("quote")
        head = f"{match.group('key')}{match.group('kq')}{match.group('sep')}"
        return f"{head}{quote_mark}[REDACTED:assignment]{quote_mark}"

    def find(self, text: str) -> list[str]:
        """Names of the secret kinds present in ``text`` (values are never returned)."""
        kinds: list[str] = []
        if self._has_literal(text):
            kinds.append("literal")
        for name, pattern in _PATTERNS:
            if pattern.search(text):
                kinds.append(name)
        for match in _ASSIGNMENT.finditer(text):
            if self._is_secret_assignment(match):
                kinds.append("assignment")
                break
        for _start, _end, kind in self._broken_spans(text):
            if kind not in kinds:
                kinds.append(kind)
        return kinds

    def scan_added_lines(self, diff_text: str) -> list[SecretFinding]:
        """Scan a unified diff and report secrets on added lines only.

        Hunks are followed by their declared line counts, so an added line that happens to read
        ``+++ ...`` is content, not a file header, and the ``\\ No newline at end of file`` marker
        never shifts line numbers.  The text is split on ``\\n`` only: ``str.splitlines`` also
        cuts at ``\\r``, form feeds and U+2028, which would hide the tail of an added line.
        """
        findings: list[SecretFinding] = []
        path = ""
        new_line = 0
        old_left = new_left = 0
        for raw in diff_text.split("\n"):
            if raw.startswith("\\"):  # "\ No newline at end of file"
                continue
            if old_left > 0 or new_left > 0:
                tag = raw[:1]
                if tag == "+":
                    for kind in self.find(raw[1:]):
                        findings.append(SecretFinding(kind=kind, path=path, line=new_line))
                    new_line += 1
                    new_left -= 1
                    continue
                if tag == "-":
                    old_left -= 1
                    continue
                if tag == " " or raw == "":
                    new_line += 1
                    old_left -= 1
                    new_left -= 1
                    continue
                old_left = new_left = 0  # truncated or malformed hunk: read the line as a header
            if raw.startswith("+++ "):
                target = raw[4:].strip()
                path = target[2:] if target.startswith("b/") else target
                continue
            if raw.startswith("--- ") or raw.startswith("diff --git"):
                continue
            hunk = _HUNK.match(raw)
            if hunk:
                new_line = int(hunk.group(2))
                old_left = 1 if hunk.group(1) is None else int(hunk.group(1))
                new_left = 1 if hunk.group(3) is None else int(hunk.group(3))
                continue
            if raw.startswith("+"):  # a '+' line outside any counted hunk (hand-written diff)
                for kind in self.find(raw[1:]):
                    findings.append(SecretFinding(kind=kind, path=path, line=new_line))
                new_line += 1
        return findings


def strip_control_chars(text: str) -> str:
    """Drop control characters other than newline and tab."""
    return _CONTROL.sub("", text)


def truncate(text: str, limit: int, marker: str = "\n... [truncated]") -> str:
    """Cut ``text`` to at most ``limit`` characters, appending ``marker`` when shortened."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    keep = limit - len(marker)
    if keep < 0:  # the marker does not fit: a plain cut still honours "at most limit"
        return text[:limit]
    return text[:keep] + marker


def fenced(text: str, *, language: str = "text") -> str:
    """Wrap ``text`` in a Markdown code fence that the text itself cannot close."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{language}\n{body}{fence}"
