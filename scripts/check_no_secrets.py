#!/usr/bin/env python3
"""Fail when a credential-looking literal is committed.

Two modes:

* ``--base REF``  scan every line that ``REF..HEAD`` ADDS, however it got into the history: the
  patch of each commit, the combined diff of each merge commit (what a conflict resolution wrote
  that neither parent had) and the net diff from the merge-base (so a merge's result counts even if
  a later commit scrubbed it). A secret added and later deleted is still public in the history, so
  it still fails, while a leak that predates ``REF`` neither hides a new one nor blocks unrelated
  work. CI runs this on a pull request with the BASE branch's copy of this script and allow-list
  (``.github/workflows/secret-scan.yml``), so a pull request cannot loosen its own check.
* ``--tree``      scan every tracked file at HEAD (every file under the repository root when that is
  not a git checkout: a source tarball, ``git archive`` or a Docker build context).

The rules are few and aim at the mistakes that have actually happened in this repository (a sudo
password written into a prompt file and a goal file) and at the credential shapes the product
itself creates: the ``.env`` assignments of ``.env.example``, the tunnel and maps keys of the
settings, ``curl -u`` and URL userinfo in the runbooks. It is a tripwire, not a replacement for a
full secret scanner: a password of one character class that is a plain word (``sunshine``) or a
secret split across lines passes, and ``SKIP_PREFIXES`` / ``SKIP_SUFFIXES`` are never scanned
(fixtures, generated artefacts). A reviewed false positive is silenced by adding a line to
``spec/secret_scan_allow.txt``: ``<path>:<sha256 of the matched literal, first 12 hex chars>`` and a
``# reason`` (the literal itself is never printed). The file is part of the reviewed surface:
``.github/CODEOWNERS`` names its owner.

Exit 0 = clean, 1 = findings, 2 = the scan could not run (fails closed).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW_FILE = ROOT / "spec" / "secret_scan_allow.txt"

# Paths whose content is deliberately fake credentials (fixtures) or generated artefacts.
SKIP_PREFIXES = ("tests/", "frontend/tests/", "node_modules/", "verification_artifacts/")
SKIP_SUFFIXES = (
    *(f".{kind}.{ext}" for kind in ("spec", "test") for ext in ("ts", "tsx", "js")),
    *(".lock", ".svg", ".png", ".jpg", ".stl", ".step", ".hef"),
)
# Without a git checkout the tree is walked; these directories are never part of the source.
PRUNED_DIRS = frozenset(
    (".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache")
) | {".pytest_cache"}

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


# ----------------------------------------------------------------------------------- the rules

# Source files where "password:" is a field name, a type annotation or an env read, never a
# literal secret.
CODE_SUFFIXES = (".py", ".ts", ".tsx", ".vue", ".js", ".mjs", ".cjs")

# What a literal must look like, beyond "not a placeholder", to be taken for a credential (the
# rules of free-form text: a prose sentence or a settings name is not one).
SHAPED, STRONG = "shaped", "strong"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    group: int = 1  # the group holding the literal
    docs_only: bool = False  # not in source files (a field name, a type, an env read)
    weak: bool = False  # free-form literal: a placeholder or a lower-case name is excused
    shape: str = ""  # SHAPED: 8+ chars, 2+ character classes; STRONG: also a letter AND a digit


def _rule(name: str, pattern: str, **kw) -> Rule:
    return Rule(name, re.compile(pattern), **kw)


# Each regex keeps the literal out of a whitespace-containing span so a sentence is not mistaken
# for one, and starts at a boundary (never inside a long run of word characters) so that a minified
# or encoded line cannot make it quadratic.
RULES: tuple[Rule, ...] = (
    _rule("sshpass-literal", r"sshpass\s+-p\s*['\"]?([^\s'\"$<{]{4,})", weak=True),
    # The credential written straight after the word sudo, quoted or in backticks.
    _rule("sudo-literal", r"(?i)\bsudo\b\s*[:=(]?\s*[`'\"]([^`'\"\s]{6,})[`'\"]", weak=True),
    _rule(
        "labelled-password-literal",
        r"(?i)\b(?:sudo|root|ssh|login|admin)\s+(?:password|passwd|pw)\b"
        r"[^\n`'\"]{0,20}[`'\"]([^`'\"\s]{6,})[`'\"]",
        docs_only=True,
        weak=True,
    ),
    _rule(
        "password-colon-literal",
        r"(?i)\bpassword\s*[:=]\s*[`'\"]?([^\s`'\"<>$]{6,})",
        docs_only=True,
        weak=True,
    ),
    # "the sudo password is X" and "# password X" written as prose, unquoted.
    _rule(
        "password-prose-literal",
        r"(?i)\b(?:(?:sudo|root|ssh|login|admin)\s+)?(?:password|passwd|pw)\s+"
        r"(?:(?:is|was)\s+)?[`'\"]?([^\s`'\"<>$,;)]{8,})",
        docs_only=True,
        weak=True,
        shape=STRONG,
    ),
    # ``echo SECRET | sudo -S cmd`` and ``printf '%s\n' SECRET | sudo -S cmd``.
    _rule(
        "sudo-stdin-literal",
        r"(?:echo|printf)\s+(?:-[A-Za-z]+\s+)*(?:'%[^']*'\s+|\"%[^\"]*\"\s+)?['\"]?([^\s'\"|]{6,})"
        r"['\"]?\s*\|\s*sudo\s+(?:-[A-Za-z]+\s+)*-[A-Za-z]*S\b",
        weak=True,
    ),
    _rule(
        "bearer-literal",
        r"(?i)\bauthorization\b[\"']?\s*[:=]\s*[\"']?bearer\s+([A-Za-z0-9._~+/=-]{8,})",
        weak=True,
        shape=SHAPED,
    ),
    _rule(
        "curl-user-literal",
        r"(?i)\bcurl\b[^\n|;&]{0,200}?\s(?:-u|--user)[\s=]+[\"']?[^\s:\"']+:([^\s\"']{6,})",
        weak=True,
        shape=SHAPED,
    ),
    # scheme://user:secret@host
    _rule(
        "url-userinfo-literal",
        r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]{1,20}://[^\s/:@'\"<>]{1,64}:([^\s/@'\"<>]{6,})@",
        weak=True,
        shape=SHAPED,
    ),
    _rule(
        "ngrok-authtoken-literal",
        r"(?i)\b(?:add-authtoken|--authtoken)[\s=]+[\"']?([A-Za-z0-9_]{20,})",
        weak=True,
    ),
    _rule("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----()"),
    _rule(
        "github-token",
        r"\b((?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    ),
    _rule("gitlab-token", r"(?<![A-Za-z0-9_-])(glpat-[A-Za-z0-9_-]{20,})"),
    _rule("huggingface-token", r"\b(hf_[A-Za-z0-9]{30,})\b"),
    _rule("openai-style-key", r"\b(sk-[A-Za-z0-9_-]{32,})\b"),
    _rule("aws-access-key", r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b"),
    _rule("google-api-key", r"(?<![A-Za-z0-9_-])(AIza[0-9A-Za-z_-]{35})(?![A-Za-z0-9_-])"),
    _rule("slack-token", r"\b(xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    _rule(
        "slack-webhook",
        r"(hooks\.slack\.com/services/T[A-Z0-9]{6,}/B[A-Z0-9]{6,}/[A-Za-z0-9]{20,})",
    ),
    _rule("npm-token", r"\b(npm_[A-Za-z0-9]{36})\b"),
    _rule("pypi-token", r"\b(pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,})"),
    _rule(
        "jwt",
        r"\b(eyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,})\b",
    ),
    # A Cloudflare tunnel token is base64 of {"a":...,"t":...,"s":...}: one long segment.
    _rule(
        "cloudflare-tunnel-token",
        r"(?<![A-Za-z0-9+/_-])(eyJ[A-Za-z0-9+/_-]{60,}={0,2})(?![A-Za-z0-9+/_.=-])",
    ),
)


# Assignments by name: ``NTRIP_PASSWORD=x``, ``"google_api_key": "x"``, ``psk = x``, ``pw: x``. The
# name starts at a boundary and is bounded, so a long run of word characters costs nothing.
_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<name>[A-Za-z_][A-Za-z0-9_.-]{0,79})"
    r"[\"']?[ \t]{0,16}(?::=|=|:)[ \t]{0,16}"
    r"(?:\"(?P<dq>[^\"\n]*)\"|'(?P<sq>[^'\n]*)'|`(?P<bt>[^`\n]*)`|(?P<bare>[^\s\"'`,;#<>]+))"
)
_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_SECRET_WORDS = frozenset(
    {"password", "passwd", "passphrase", "passcode", "pwd", "pass", "pw", "secret", "secrets"}
    | {"token", "credential", "credentials", "apikey", "authtoken", "psk", "sshpass", "hmac"}
)
_SECRET_PAIRS = frozenset(
    {("api", "key"), ("auth", "key"), ("access", "key"), ("private", "key"), ("secret", "key")}
    | {("signing", "key"), ("encryption", "key"), ("master", "key")}
)
_TRAILING_WORDS = frozenset({"value", "string", "text", "plain", "cleartext", "raw", "literal"})
_LONG_KEY = re.compile(r"^(?:[A-Fa-f0-9]{32,}|[A-Za-z0-9+/_-]{40,}={0,2})$")
# A value that names something instead of being a secret: snake_case or SNAKE_CASE, a dotted
# module path, CamelCase words, a date.
_NAME_LIKE = re.compile(
    r"^(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+"
    r"|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|(?:[A-Z][a-z]+){2,}"
    r"|\d{4}-\d\d-\d\d.*)$"
)
_NAMED, _KEY_NAMED = 1, 2


def _secret_name(name: str) -> int:
    """``_NAMED`` for a name that ends in a credential word, ``_KEY_NAMED`` for ``*_KEY``, else 0.

    The credential word must END the name (``NTRIP_PASSWORD``, ``googleApiKey``, ``X-Auth-Token``,
    ``NTRIP_PASSWORD_VALUE``): ``token_type``, ``password_policy`` and ``secret_name`` name settings
    about a credential, and ``test_secrets.py:43`` is a file and a line.
    """
    words = [w.lower() for w in _WORDS.findall(name)]
    if len(words) > 1 and words[-1] in _TRAILING_WORDS:
        words.pop()
    if not words or "public" in words:
        return 0
    if words[-1] in _SECRET_WORDS or tuple(words[-2:]) in _SECRET_PAIRS:
        return _NAMED
    return _KEY_NAMED if words[-1] == "key" else 0


def _secret_shaped(value: str, *, strong: bool = False) -> bool:
    """Looks generated or chosen, not a word, a name, a path, a URL or a number."""
    v = value.strip()
    if len(v) < 8 or any(c.isspace() for c in v) or "://" in v:
        return False
    if v.startswith(("/", "./", "../", "~", "$", "%", "@", "{", "[", "(")) or _NAME_LIKE.match(v):
        return False
    if strong:
        return any(c.isalpha() for c in v) and any(c.isdigit() for c in v)
    classes = (
        any(c.islower() for c in v),
        any(c.isupper() for c in v),
        any(c.isdigit() for c in v),
        any(not c.isalnum() for c in v),
    )
    return sum(classes) >= 2


def _named_secrets(line: str, *, is_code: bool) -> Iterator[str]:
    """The literal of every ``<credential-name> = <literal>`` on the line."""
    for m in _ASSIGNMENT.finditer(line):
        kind = _secret_name(m["name"])
        if not kind:
            continue
        quoted = next((m[g] for g in ("dq", "sq", "bt") if m[g] is not None), None)
        value = quoted if quoted is not None else m["bare"]
        # In source "password: str" and "token = read()" are code; only a quoted literal is data.
        if value is None or (is_code and quoted is None) or _is_placeholder(value):
            continue
        if quoted is None and ("(" in value or ")" in value):
            continue  # ``token = secrets.token_urlsafe(32)``: a call, not a literal
        if kind == _KEY_NAMED:
            ok = bool(_LONG_KEY.match(value))
        else:
            ok = _secret_shaped(value)
        if ok:
            yield value


def scan_line(line: str, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(rule, literal)`` for each credential-looking literal on one line of ``path``."""
    is_code = path.endswith(CODE_SUFFIXES)
    seen: set[str] = set()
    for rule in RULES:
        if rule.docs_only and is_code:
            continue
        for m in rule.pattern.finditer(line):
            literal = m.group(rule.group)
            if rule.weak and (not literal or _is_placeholder(literal)):
                continue
            if rule.shape and not _secret_shaped(literal, strong=rule.shape == STRONG):
                continue
            if literal not in seen:
                seen.add(literal)
                yield rule.name, literal
    for literal in _named_secrets(line, is_code=is_code):
        if literal not in seen:
            seen.add(literal)
            yield "secret-assignment", literal


# ----------------------------------------------------------------------------------- the files


def _skipped(path: str) -> bool:
    return path.startswith(SKIP_PREFIXES) or path.endswith(SKIP_SUFFIXES)


_ENTRY = re.compile(r"^\S.*:[0-9a-f]{12}$")


def load_allow(path: Path = ALLOW_FILE) -> set[str]:
    """The reviewed ``<path>:<digest>`` entries; a malformed one stops the scan (fails closed)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = raw.split("#", 1)[0].strip()
        if not entry:
            continue
        if not _ENTRY.match(entry):
            raise RuntimeError(f"{path}:{n}: not '<path>:<12 hex chars of the literal's sha256>'")
        out.add(entry)
    return out


def _lines(text: str) -> list[str]:
    """The lines of ``text``, split on newlines only (``str.splitlines`` also splits on \\x0c)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln[:-1] if ln.endswith("\r") else ln for ln in lines]


_GARBAGE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ufffd]")


def _is_binary_garbage(line: str) -> bool:
    """More than a tenth control bytes and undecodable bytes: the inside of a binary file.

    ``git log -p --text`` prints a binary file as lines; matching on random bytes finds a "token"
    in every megabyte. A text line with a stray control character or two is still a text line.
    """
    return len(_GARBAGE.findall(line)) * 10 > len(line)


def scan_text(path: str, text: str, allow: set[str], first_line: int = 1) -> list[Finding]:
    found: list[Finding] = []
    for offset, line in enumerate(_lines(text)):
        if _is_binary_garbage(line):
            continue
        for rule, literal in scan_line(line, path):
            digest = _digest(literal)
            if f"{path}:{digest}" in allow:
                continue
            found.append(Finding(path, first_line + offset, rule, digest))
    return found


def _run(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False, timeout=300)


def _git(*args: str) -> str:
    proc = _run(*args)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:300]
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.decode("utf-8", "replace")


def _tracked_files() -> list[str]:
    if (ROOT / ".git").exists():
        return sorted(p for p in _git("ls-files", "-z").split("\0") if p)
    found: list[str] = []
    for here, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in PRUNED_DIRS)
        found += [str(Path(here, f).relative_to(ROOT)) for f in sorted(files)]
    return found


def scan_tree(allow: set[str]) -> list[Finding]:
    found: list[Finding] = []
    for rel in _tracked_files():
        if _skipped(rel):
            continue
        try:
            data = (ROOT / rel).read_bytes()
        except OSError:
            continue  # a dangling symlink, a submodule
        found.extend(scan_text(rel, data.decode("utf-8", "replace"), allow))
    return found


# ----------------------------------------------------------------------------------- the patches

_HUNK = re.compile(r"^(@@+) (?:-\d+(?:,\d+)? )+\+(\d+)")
_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _unquote(name: str) -> str:
    """A path as git quotes it when it holds a quote, a backslash or a control character."""
    if len(name) < 2 or not (name.startswith('"') and name.endswith('"')):
        return name
    body, out, i = name[1:-1], bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
            octal = re.match(r"[0-7]{1,3}", body[i + 1 :])
            if octal:
                out.append(int(octal.group(), 8) & 0xFF)
                i += 1 + len(octal.group())
                continue
        out += ch.encode("utf-8")
        i += 1
    return out.decode("utf-8", "replace")


def _header_path(spec: str) -> str:
    """The path of a ``+++`` header: ``""`` for a deleted file, without the ``b/`` prefix."""
    spec = spec.rstrip("\t")  # git appends a TAB to a name that holds a space
    return "" if spec == "/dev/null" else _unquote(spec).removeprefix("b/")


def added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """Parse ``git diff -U0`` or ``git log -p --cc`` into ``(path, line_no, text)`` of added lines.

    A content line starts with ``+``, ``-``, a space or a backslash, never with ``d`` or ``@``, so
    ``diff --`` and ``@@`` always open a header or a hunk. Inside a hunk every line is content,
    whatever it says: an added line ``++ tests/x`` or ``+++ b/x`` is text, not a file header. A
    combined diff (a merge) has one marker column per parent; a line is new only when every column
    is ``+``.
    """
    path, columns, line_no, in_hunk = "", 1, 0, False
    for raw in diff.split("\n"):
        if raw.startswith("diff --"):
            in_hunk = False
            path = ""
            if raw.startswith(("diff --cc ", "diff --combined ")):
                path = _unquote(raw.split(" ", 2)[2]).rstrip("\t")
        elif raw.startswith("@@"):
            m = _HUNK.match(raw)
            in_hunk = m is not None
            if m:
                columns, line_no = len(m.group(1)) - 1, int(m.group(2))
        elif not in_hunk:
            if raw.startswith("+++ "):
                path = _header_path(raw[4:])
        elif not raw.startswith("\\"):  # "\ No newline at end of file"
            marks = raw[:columns]
            if marks == "+" * columns:
                yield path, line_no, raw[columns:]
                line_no += 1
            elif "-" not in marks:  # context, or a line one parent already had
                line_no += 1


# What every diff the scan reads is asked to be, whatever the repository or the user configure:
# text (a ``-diff`` attribute or a NUL byte would otherwise hide a file's patch behind "Binary
# files differ"), unquoted non-ASCII names, no renames (a moved file's lines count as added at the
# new path), no prefix heuristics, no external diff or textconv, no signature lines.
DIFF_FLAGS = (
    "-U0",
    "--text",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-renames",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


def _diff_text(*args: str) -> str:
    return _git("-c", "core.quotepath=off", "-c", "log.showSignature=false", *args)


def _added_findings(diff: str, allow: set[str]) -> list[Finding]:
    found: list[Finding] = []
    for path, line_no, text in added_lines(diff):
        if path and not _skipped(path):
            found.extend(scan_text(path, text, allow, first_line=line_no))
    return found


def scan_commits(base: str, allow: set[str]) -> list[Finding]:
    """The lines each commit of ``base..HEAD`` adds: a patch per commit, a combined diff per merge.

    A secret added in one commit and deleted in a later one is still public in the pull request's
    history, so the net diff alone would wave it through. A merge has no patch of its own, only the
    combined diff against its parents, which holds what a conflict resolution wrote that neither
    parent had (``git log -p`` without ``--cc`` prints nothing for a merge).
    """
    return _added_findings(
        _diff_text("log", "--cc", "-p", *DIFF_FLAGS, "--format=", f"{base}..HEAD"), allow
    )


def scan_net(base: str, allow: set[str]) -> list[Finding]:
    """The lines the branch adds in all, from the merge-base to HEAD (a merge's result included).

    Unrelated histories have no merge-base: then everything at HEAD is new.
    """
    merge_base = _run("merge-base", base, "HEAD")
    if merge_base.returncode == 0:
        start = merge_base.stdout.decode().strip()
    elif merge_base.returncode == 1:
        start = _git("hash-object", "-t", "tree", os.devnull).strip()  # the empty tree
    else:
        raise RuntimeError(f"git merge-base failed: {merge_base.stderr.decode()[:300]}")
    return _added_findings(_diff_text("diff", *DIFF_FLAGS, start, "HEAD"), allow)


def scan_diff(base: str, allow: set[str]) -> list[Finding]:
    """Every line ``base..HEAD`` adds, however it got there (``scan_commits`` and ``scan_net``)."""
    _git("rev-parse", "--verify", f"{base}^{{commit}}")  # an unknown base fails closed
    found: dict[tuple[str, int, str, str], Finding] = {}
    for f in scan_commits(base, allow) + scan_net(base, allow):
        found.setdefault((f.path, f.line_no, f.rule, f.digest), f)
    return list(found.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="scan lines added by REF..HEAD (commits, merges, net diff)")
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
            "goes in spec/secret_scan_allow.txt as '<path>:<digest>  # reason'.",
            file=sys.stderr,
        )
        return 1
    print("check_no_secrets: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
