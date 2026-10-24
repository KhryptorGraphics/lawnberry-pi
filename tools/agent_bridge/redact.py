"""Secret redaction and safe rendering of untrusted text.

Everything the bridge stores, logs or publishes that originates from an agent or from the
network passes through :class:`Redactor`.  Agent text is additionally rendered inside a code
fence (:func:`fenced`) so Markdown, HTML, images and ``@mentions`` stay inert on GitHub.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "Redactor",
    "SecretFinding",
    "fenced",
    "strip_control_chars",
    "truncate",
]

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "github-token",
        re.compile(r"gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{50,255}"),
    ),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    # "sk-" also ends ordinary words ("disk-usage-...", "risk-assessment-..."): no letter or digit
    # may precede it.  The other prefixes are distinctive, so they match even when glued to text.
    ("openai-key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("aws-access-key", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    ("slack-token", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    (
        "private-key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[^\"]*?"
            r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|(?=\")|\Z)"
        ),
    ),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{20,}")),
    (
        "jwt",
        # a start may not sit inside a longer token run: every run is scanned once (linear time)
        re.compile(
            r"(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]{8,}"
            r"\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
        ),
    ),
    (
        "url-credentials",
        re.compile(r"(?i)(?<![a-z0-9+.\-])(?P<scheme>[a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@"),
    ),
)

# NAME=value / NAME: value where NAME looks like a credential.  Only the value is replaced.
# The key is the credential word plus a bounded tail of identifier characters; no unbounded
# quantifier sits before it, so scanning stays linear on any input.
_ASSIGNMENT = re.compile(
    r"(?i)(?P<key>(?:API[_-]?KEY|ACCESS[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"
    r"[A-Z0-9_]{0,64}+)(?P<kq>\\?['\"]?)(?P<sep>\s*[:=]\s*)(?P<quote>\\?['\"]?)"
    r"(?P<value>[^\s'\"\\]{8,})(?P=quote)"
)
# Values that are obviously placeholders must not trigger the assignment rule.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:<[^>]*>|\$\{?[A-Z0-9_]+\}?|\[REDACTED[^\]]*\]|x{4,}|\*{4,}|\.{3,}|none|null)$"
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


class Redactor:
    """Replace credentials in text with ``[REDACTED:<kind>]`` placeholders."""

    def __init__(self, literals: Iterable[str] = ()) -> None:
        cleaned = {s for s in literals if isinstance(s, str) and len(s.strip()) >= _MIN_LITERAL}
        self._literals = sorted(cleaned, key=len, reverse=True)

    def add_literal(self, value: str | None) -> None:
        """Register one more exact secret value (ignored when too short to be one)."""
        if value and len(value.strip()) >= _MIN_LITERAL and value not in self._literals:
            self._literals.append(value)
            self._literals.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        """Return ``text`` with known secret literals and secret-shaped strings removed."""
        if not text:
            return text
        for literal in self._literals:
            if literal in text:
                text = text.replace(literal, "[REDACTED:literal]")
        for name, pattern in _PATTERNS:
            if name == "url-credentials":
                text = pattern.sub(
                    lambda m: f"{m.group('scheme')}[REDACTED:url-credentials]@", text
                )
            else:
                text = pattern.sub(f"[REDACTED:{name}]", text)
        return _ASSIGNMENT.sub(self._assignment, text)

    @staticmethod
    def _is_secret_assignment(match: re.Match[str]) -> bool:
        if _PLACEHOLDER.match(match.group("value")):
            return False
        # `"maxOutputTokens":32000}},` is JSON structure, not a credential: after a quoted key
        # only a quoted value counts, otherwise numbers and brackets would be eaten.
        return not (match.group("kq") and not match.group("quote"))

    @staticmethod
    def _assignment(match: re.Match[str]) -> str:
        if not Redactor._is_secret_assignment(match):
            return match.group(0)
        quote = match.group("quote")
        head = f"{match.group('key')}{match.group('kq')}{match.group('sep')}"
        return f"{head}{quote}[REDACTED:assignment]{quote}"

    def find(self, text: str) -> list[str]:
        """Names of the secret kinds present in ``text`` (values are never returned)."""
        kinds: list[str] = []
        for literal in self._literals:
            if literal in text:
                kinds.append("literal")
                break
        for name, pattern in _PATTERNS:
            if pattern.search(text):
                kinds.append(name)
        for match in _ASSIGNMENT.finditer(text):
            if self._is_secret_assignment(match):
                kinds.append("assignment")
                break
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
