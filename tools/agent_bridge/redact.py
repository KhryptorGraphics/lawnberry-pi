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
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{50,255})\b"),
    ),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
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
        re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    ),
    (
        "url-credentials",
        re.compile(r"(?i)\b(?P<scheme>[a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@"),
    ),
)

# NAME=value / NAME: value where NAME looks like a credential.  Only the value is replaced.
_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<key>[A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"
    r"[A-Z0-9_]*)(?P<sep>\s*[:=]\s*)(?P<quote>['\"]?)(?P<value>[^\s'\"\\]{8,})(?P=quote)"
)
# Values that are obviously placeholders must not trigger the assignment rule.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:<[^>]*>|\$\{?[A-Z0-9_]+\}?|\[REDACTED[^\]]*\]|x{4,}|\*{4,}|\.{3,}|none|null)$"
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
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
    def _assignment(match: re.Match[str]) -> str:
        if _PLACEHOLDER.match(match.group("value")):
            return match.group(0)
        quote = match.group("quote")
        return f"{match.group('key')}{match.group('sep')}{quote}[REDACTED:assignment]{quote}"

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
            if not _PLACEHOLDER.match(match.group("value")):
                kinds.append("assignment")
                break
        return kinds

    def scan_added_lines(self, diff_text: str) -> list[SecretFinding]:
        """Scan a unified diff and report secrets on added lines only."""
        findings: list[SecretFinding] = []
        path = ""
        new_line = 0
        for raw in diff_text.splitlines():
            if raw.startswith("+++ "):
                target = raw[4:].strip()
                path = target[2:] if target.startswith("b/") else target
                continue
            if raw.startswith("--- ") or raw.startswith("diff --git"):
                continue
            hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if hunk:
                new_line = int(hunk.group(1))
                continue
            if raw.startswith("+"):
                for kind in self.find(raw[1:]):
                    findings.append(SecretFinding(kind=kind, path=path, line=new_line))
                new_line += 1
            elif not raw.startswith("-"):
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
    keep = max(0, limit - len(marker))
    return text[:keep] + marker


def fenced(text: str, *, language: str = "text") -> str:
    """Wrap ``text`` in a Markdown code fence that the text itself cannot close."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{language}\n{body}{fence}"
