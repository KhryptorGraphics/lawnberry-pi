#!/usr/bin/env python3
"""Fail when a credential-looking literal is committed.

Two modes:

* ``--base REF``  scan the lines ADDED by every commit in ``REF..HEAD`` (what CI runs on a pull
  request): a secret added and later deleted is still public in the history, so it still fails,
  while a leak that predates ``REF`` neither hides a new one nor blocks unrelated work.
* ``--tree``      scan every tracked file at HEAD (used locally, to list what is committed).

The patterns are deliberately few and high-signal. This is a tripwire for the mistakes that have
actually happened in this repository (a sudo password written into a prompt file and a goal file),
not a replacement for a full secret scanner. A reviewed false positive is silenced by adding a
line to ``spec/secret_scan_allow.txt`` (``<path>:<sha256 of the matched literal, first 12 hex
chars>``); the literal itself is never printed.

Exit 0 = clean, 1 = findings, 2 = the scan could not run (fails closed).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW_FILE = ROOT / "spec" / "secret_scan_allow.txt"

# Paths whose content is deliberately fake credentials (fixtures) or generated artefacts.
SKIP_PREFIXES = ("tests/", "frontend/tests/", "node_modules/", "verification_artifacts/")
SKIP_SUFFIXES = (".spec.ts", ".test.ts", ".lock", ".svg", ".png", ".jpg", ".stl", ".step", ".hef")

_PLACEHOLDER = re.compile(
    r"^(<.*>|\$.*|\{.*\}|\*+|x{3,}|\.{3,}|your[-_ ]?.*|changeme.*|change[-_ ]me.*|example.*|"
    r"redacted.*|"
    r"placeholder.*|provided.*|secret|password|passwd|token|null|none|true|false|[a-z_]+\(.*|"
    r"(?-i:[a-z]+(?:_[a-z]+)+))$",  # lower-case snake_case word: a name, not a secret
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: str
    line_no: int
    rule: str
    digest: (
        str  # first 12 hex chars of sha256(literal): lets a reviewer allow-list without printing it
    )


def _digest(literal: str) -> str:
    return hashlib.sha256(literal.encode()).hexdigest()[:12]


def _is_placeholder(literal: str) -> bool:
    return bool(_PLACEHOLDER.match(literal.strip()))


# Source files where "password:" is a field name, a type annotation or an env read, never a
# literal secret.
CODE_SUFFIXES = (".py", ".ts", ".tsx", ".vue", ".js", ".mjs", ".cjs")

# (rule name, compiled regex, group holding the literal, applies only to non-source files). Each
# regex keeps the literal out of a whitespace-containing span so a sentence is not mistaken for one.
RULES: tuple[tuple[str, re.Pattern[str], int, bool], ...] = (
    ("sshpass-literal", re.compile(r"sshpass\s+-p\s*['\"]?([^\s'\"$<{]{4,})"), 1, False),
    # The credential written straight after the word sudo, quoted or in backticks.
    (
        "sudo-literal",
        re.compile(r"(?i)\bsudo\b\s*[:=(]?\s*[`'\"]([^`'\"\s]{6,})[`'\"]"),
        1,
        False,
    ),
    (
        "labelled-password-literal",
        re.compile(
            r"(?i)\b(?:sudo|root|ssh|login|admin)\s+(?:password|passwd|pw)\b"
            r"[^\n`'\"]{0,20}[`'\"]([^`'\"\s]{6,})[`'\"]"
        ),
        1,
        True,
    ),
    (
        "password-colon-literal",
        re.compile(r"(?i)\bpassword\s*[:=]\s*[`'\"]?([^\s`'\"<>$]{6,})"),
        1,
        True,
    ),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----()"), 1, False),
    (
        "github-token",
        re.compile(r"\b((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
        1,
        False,
    ),
    ("huggingface-token", re.compile(r"\b(hf_[A-Za-z0-9]{30,})\b"), 1, False),
    ("openai-style-key", re.compile(r"\b(sk-[A-Za-z0-9_-]{32,})\b"), 1, False),
    ("aws-access-key", re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), 1, False),
    ("slack-token", re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{20,})\b"), 1, False),
    (
        "jwt",
        re.compile(r"\b(eyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,})\b"),
        1,
        False,
    ),
)

# Rules that match a free-form literal, where a placeholder or a lower-case name is likely. Token
# and key formats are specific enough that a match is never excused as a placeholder.
WEAK_RULES = frozenset(
    {"sshpass-literal", "sudo-literal", "labelled-password-literal", "password-colon-literal"}
)


def scan_line(line: str, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(rule, literal)`` for each credential-looking literal on one line of ``path``."""
    is_code = path.endswith(CODE_SUFFIXES)
    for name, rx, group, docs_only in RULES:
        if docs_only and is_code:
            continue
        for m in rx.finditer(line):
            literal = m.group(group)
            if name in WEAK_RULES and (not literal or _is_placeholder(literal)):
                continue
            yield name, literal


def _skipped(path: str) -> bool:
    return path.startswith(SKIP_PREFIXES) or path.endswith(SKIP_SUFFIXES)


def load_allow(path: Path = ALLOW_FILE) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def scan_text(path: str, text: str, allow: set[str], first_line: int = 1) -> list[Finding]:
    found: list[Finding] = []
    for offset, line in enumerate(text.splitlines()):
        for rule, literal in scan_line(line, path):
            digest = _digest(literal)
            if f"{path}:{digest}" in allow:
                continue
            found.append(Finding(path, first_line + offset, rule, digest))
    return found


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False, timeout=120
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def scan_tree(allow: set[str]) -> list[Finding]:
    found: list[Finding] = []
    for rel in _git("ls-files", "-z").split("\0"):
        if not rel or _skipped(rel):
            continue
        p = ROOT / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found.extend(scan_text(rel, text, allow))
    return found


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)")


def added_lines(diff: str) -> Iterable[tuple[str, int, str]]:
    """Parse ``git diff -U0`` into ``(path, line_no, text)`` for every added line."""
    path = ""
    line_no = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].removeprefix("b/") if raw[4:] != "/dev/null" else ""
        elif raw.startswith("@@"):
            m = _HUNK.match(raw)
            line_no = int(m.group(1)) if m else 0
        elif raw.startswith("+") and not raw.startswith("+++"):
            yield path, line_no, raw[1:]
            line_no += 1


def scan_diff(base: str, allow: set[str]) -> list[Finding]:
    """Scan the patch of EVERY commit in ``base..HEAD``, not just the net change.

    A secret added in one commit and deleted in a later one is still public in the pull request's
    history, so a net diff would wave it through.
    """
    patches = _git(
        "log",
        "-p",
        "-U0",
        "--no-color",
        "--no-ext-diff",
        "--no-merges",
        "--format=",
        f"{base}..HEAD",
    )
    found: list[Finding] = []
    for path, line_no, text in added_lines(patches):
        if not path or _skipped(path):
            continue
        found.extend(scan_text(path, text, allow, first_line=line_no))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="scan lines added by each commit in REF..HEAD")
    mode.add_argument("--tree", action="store_true", help="scan every tracked file at HEAD")
    args = ap.parse_args(argv)
    try:
        allow = load_allow()
        findings = scan_tree(allow) if args.tree else scan_diff(args.base, allow)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"check_no_secrets: could not scan: {exc}", file=sys.stderr)
        return 2
    for f in findings:
        print(f"{f.path}:{f.line_no}: {f.rule} (literal sha256:{f.digest}; value not printed)")
    if findings:
        print(
            f"check_no_secrets: {len(findings)} credential-looking literal(s). Remove them and "
            "ROTATE the credential (a pushed secret is compromised). A reviewed false positive "
            "goes in spec/secret_scan_allow.txt as '<path>:<digest>'.",
            file=sys.stderr,
        )
        return 1
    print("check_no_secrets: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
