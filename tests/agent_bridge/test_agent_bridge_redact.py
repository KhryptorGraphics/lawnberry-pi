"""Tests for ``tools/agent_bridge/redact.py``: secret redaction, diff scanning, safe rendering."""

from __future__ import annotations

import random
import re
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from bridgefakes.rig import GitRepoEnv, make_git_env

from tools.agent_bridge.redact import (
    _PATTERNS,
    Redactor,
    SecretFinding,
    fenced,
    strip_control_chars,
    truncate,
)

# Secrets are assembled at runtime so no scanner mistakes this file for a credential leak.
ALNUM36 = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
GITHUB_TOKEN = "ghp_" + ALNUM36
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
SLACK_TOKEN = "xoxb-" + "123456789012-abcdefghijkl"
PEM_BODY = "MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun"
PEM = f"-----BEGIN RSA PRIVATE KEY-----\n{PEM_BODY}\nabcDEF123\n-----END RSA PRIVATE KEY-----"
JWT = (
    "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)

# kind -> secret strings that must be replaced *entirely* by ``[REDACTED:<kind>]``.
SECRET_SAMPLES: dict[str, list[str]] = {
    "github-token": [
        "ghp_" + ALNUM36,
        "gho_" + ALNUM36,
        "ghu_" + ALNUM36,
        "ghs_" + ALNUM36,
        "ghr_" + ALNUM36,
        "github_pat_" + "11ABCDEFG0" * 6,
    ],
    "anthropic-key": ["sk-ant-" + "api03-AbCdEfGhIjKlMnOpQrStUvWxYz012345"],
    "openai-key": ["sk-" + ALNUM36, "sk-proj-" + ALNUM36, "sk-svcacct-" + ALNUM36],
    "aws-access-key": [AWS_KEY, "ASIA" + "IOSFODNN7EXAMPLE"],
    "slack-token": [SLACK_TOKEN, "xoxp-" + "1234567890-ABCDEFGHIJ", "xoxa-" + "2-1234567890abcdef"],
    "google-api-key": ["AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"],
    "private-key": [PEM, PEM.replace("RSA ", ""), PEM.replace("RSA", "OPENSSH")],
    "bearer-token": ["Bearer " + "abcdefghijklmnopqrstuvwxyz0123456789", "bearer " + "A" * 24],
    "jwt": [JWT],
}
# (prefix, suffix) pairs: the text around a secret must survive untouched.
SURROUNDINGS = [
    ("", ""),
    ("the value is ", " and nothing else"),
    ("(", ")"),
    ('"', '"'),
    ("`", "`"),
    ("line one\n", "\nline three"),
    ("Authorization: ", ", next=header"),
]
ADDED_PLACEHOLDERS = [
    "TOKEN=<your-token>",
    "GITHUB_TOKEN=${GITHUB_TOKEN}",
    "GITHUB_TOKEN=$GITHUB_TOKEN",
    "API_KEY=[REDACTED:assignment]",
    "password: null",
    "secret=None",
    "TOKEN=xxxxxxxx",
    "TOKEN=********",
    "TOKEN=........",
    "token=abc",
    "token=1234567",
    "key = value",
    "TOKEN",
]


def kinds_for(text: str) -> list[str]:
    return Redactor().find(text)


# --------------------------------------------------------------------------- patterns
def test_every_pattern_has_a_sample() -> None:
    """Adding a pattern to ``_PATTERNS`` without a test sample here fails loudly."""
    known = {name for name, _ in _PATTERNS}
    assert known == set(SECRET_SAMPLES) | {"url-credentials"}


@pytest.mark.parametrize(
    ("kind", "secret"),
    [(kind, secret) for kind, items in SECRET_SAMPLES.items() for secret in items],
    ids=lambda value: value[:18] if isinstance(value, str) else value,
)
@pytest.mark.parametrize(("prefix", "suffix"), SURROUNDINGS)
def test_pattern_redacts_the_secret_and_keeps_surrounding_text(
    kind: str, secret: str, prefix: str, suffix: str
) -> None:
    text = f"{prefix}{secret}{suffix}"
    redactor = Redactor()

    assert redactor.redact(text) == f"{prefix}[REDACTED:{kind}]{suffix}"
    assert kind in redactor.find(text)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "clone https://user:p4ssw0rd@example.com/repo.git now",
            "clone https://[REDACTED:url-credentials]@example.com/repo.git now",
        ),
        ("git://bob:hunter2@host:22/x", "git://[REDACTED:url-credentials]@host:22/x"),
        (
            "x postgres://app:s3cr3t%21@db.internal:5432/app y",
            "x postgres://[REDACTED:url-credentials]@db.internal:5432/app y",
        ),
        ("see HTTPS://U:P@host/", "see HTTPS://[REDACTED:url-credentials]@host/"),
    ],
)
def test_url_credentials_keep_scheme_and_host(url: str, expected: str) -> None:
    redactor = Redactor()
    assert redactor.redact(url) == expected
    assert "url-credentials" in redactor.find(url)


@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/a:b@c",
        "https://example.com/path",
        "mailto:someone@example.com",
        "ssh://git@github.com/owner/repo.git",
        "12:30:45 meeting@desk",
    ],
)
def test_urls_without_embedded_credentials_are_untouched(text: str) -> None:
    assert Redactor().redact(text) == text
    assert kinds_for(text) == []


def test_private_key_without_end_marker_is_redacted_to_the_end_of_the_text() -> None:
    text = "before\n-----BEGIN PRIVATE KEY-----\nMIIEow\nabc\nafter"
    assert Redactor().redact(text) == "before\n[REDACTED:private-key]"


def test_text_after_a_private_key_block_survives() -> None:
    text = f"a\n{PEM}\nmiddle\n{PEM}\nz"
    assert Redactor().redact(text) == "a\n[REDACTED:private-key]\nmiddle\n[REDACTED:private-key]\nz"


@pytest.mark.parametrize(
    "text",
    [
        "ghp_" + "A" * 29,  # one character too short
        "AKIA" + "IOSFODNN7EXAMP",
        "Bearer abc",
        "sk-ant-short",
        "xoxb-123",
        "eyJhbGci.eyJzdWI.sig",
        "the word skeleton and sk- alone",
        "AIza" + "x" * 10,
    ],
)
def test_near_misses_are_not_redacted(text: str) -> None:
    assert Redactor().redact(text) == text
    assert kinds_for(text) == []


def test_several_secrets_in_one_text_are_all_redacted() -> None:
    text = f"a={GITHUB_TOKEN} b={AWS_KEY} c={SLACK_TOKEN} d={JWT}"
    out = Redactor().redact(text)
    assert out == (
        "a=[REDACTED:github-token] b=[REDACTED:aws-access-key] "
        "c=[REDACTED:slack-token] d=[REDACTED:jwt]"
    )


def test_redaction_is_idempotent() -> None:
    text = "\n".join(
        [
            GITHUB_TOKEN,
            PEM,
            "https://user:p4ssw0rd@example.com/x",
            "GITHUB_TOKEN=abcdefgh12345678",
            "Bearer " + "a" * 30,
            JWT,
        ]
    )
    once = Redactor(["literal-secret-value"]).redact(text + "\nliteral-secret-value")
    assert Redactor(["literal-secret-value"]).redact(once) == once


def test_empty_text_is_returned_unchanged() -> None:
    assert Redactor().redact("") == ""
    assert Redactor().find("") == []


# --------------------------------------------------------------------------- assignments
ASSIGNMENTS = [
    (
        "export GITHUB_TOKEN=abcdefgh12345678 # note",
        "export GITHUB_TOKEN=[REDACTED:assignment] # note",
    ),
    ("api_key: 'abcdefgh12345678'", "api_key: '[REDACTED:assignment]'"),
    ('password = "abcdefgh12345678"', 'password = "[REDACTED:assignment]"'),
    ("  client_secret: abcdefgh1234", "  client_secret: [REDACTED:assignment]"),
    ("x-api-key=abcdefgh12345678", "x-api-key=[REDACTED:assignment]"),
    ("PASSWD: abcdefgh12345678", "PASSWD: [REDACTED:assignment]"),
    ("credentials=abcdefgh12", "credentials=[REDACTED:assignment]"),
    ("access-key=abcdefgh12", "access-key=[REDACTED:assignment]"),
    ("SECRET_KEY=abcdefgh", "SECRET_KEY=[REDACTED:assignment]"),
    ("--token=abcdefgh12345678", "--token=[REDACTED:assignment]"),
    ("DB_PASSWORD='abcdefgh'", "DB_PASSWORD='[REDACTED:assignment]'"),
    ("a TOKEN=abcdefgh12345678 b", "a TOKEN=[REDACTED:assignment] b"),
    ("token=abcdefgh12345678,next=1", "token=[REDACTED:assignment]"),
]


@pytest.mark.parametrize(("text", "expected"), ASSIGNMENTS)
def test_assignment_rule_replaces_only_the_value(text: str, expected: str) -> None:
    redactor = Redactor()
    assert redactor.redact(text) == expected
    assert redactor.find(text) == ["assignment"]


@pytest.mark.parametrize(
    ("short", "long"),
    [("token=1234567", "token=12345678"), ("password: abcdefg", "password: abcdefgh")],
)
def test_assignment_values_need_at_least_eight_characters(short: str, long: str) -> None:
    redactor = Redactor()
    assert redactor.redact(short) == short
    assert redactor.redact(long) != long


@pytest.mark.parametrize("text", ADDED_PLACEHOLDERS)
def test_placeholders_are_not_redacted(text: str) -> None:
    redactor = Redactor()
    assert redactor.redact(text) == text
    assert redactor.find(text) == []


def test_a_real_token_in_an_assignment_is_reported_as_the_token_kind() -> None:
    text = f"GITHUB_TOKEN={GITHUB_TOKEN}"
    redactor = Redactor()
    assert redactor.redact(text) == "GITHUB_TOKEN=[REDACTED:github-token]"
    assert "github-token" in redactor.find(text)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ('{"token": "abcdefgh12345678"}', '"token"'),
        ('{"password":"hunter2hunter2"}', '"password"'),
        ('{"GITHUB_TOKEN": "abcdefgh12345678", "x": 1}', '"GITHUB_TOKEN"'),
        ("{'api_key': 'abcdefgh12345678'}", "'api_key'"),
    ],
)
def test_json_and_dict_style_credentials_are_redacted(text: str, key: str) -> None:
    """``"NAME": "value"`` is the most common way a credential file is printed."""
    redactor = Redactor()
    out = redactor.redact(text)
    secret = re.search(r"[A-Za-z0-9]{12,}", text.split(":", 1)[1])
    assert secret is not None
    assert secret.group(0) not in out
    assert key in out  # the key stays readable
    assert redactor.find(text) != []


# --------------------------------------------------------------------------- literals
def test_literals_are_redacted_everywhere_they_appear() -> None:
    redactor = Redactor(["supersecretvalue"])
    text = "a supersecretvalue b\nsupersecretvalue"
    assert redactor.redact(text) == "a [REDACTED:literal] b\n[REDACTED:literal]"
    assert redactor.find(text) == ["literal"]


def test_literals_shorter_than_eight_characters_are_ignored() -> None:
    redactor = Redactor(["short", "1234567"])
    redactor.add_literal("abc")
    assert redactor.redact("short 1234567 abc") == "short 1234567 abc"
    assert redactor.find("short 1234567 abc") == []


def test_a_literal_of_exactly_eight_characters_is_used() -> None:
    assert Redactor(["12345678"]).redact("x 12345678 y") == "x [REDACTED:literal] y"
    redactor = Redactor()
    redactor.add_literal("abcdefgh")
    assert redactor.redact("abcdefgh") == "[REDACTED:literal]"


@pytest.mark.parametrize("junk", [None, "", "        ", 12345678, b"abcdefghij", ["abcdefgh"]])
def test_junk_constructor_literals_are_ignored(junk: object) -> None:
    redactor = Redactor([junk, "realsecret!"])  # type: ignore[list-item]
    assert redactor.redact("text realsecret! 12345678") == "text [REDACTED:literal] 12345678"


@pytest.mark.parametrize("junk", [None, "", "        ", "\n\t \n\t \n"])
def test_empty_or_blank_literals_added_later_are_ignored(junk: str | None) -> None:
    redactor = Redactor()
    redactor.add_literal(junk)
    assert redactor.redact("text 12345678   \n") == "text 12345678   \n"


def test_overlapping_literals_are_replaced_longest_first() -> None:
    redactor = Redactor(["abcdefgh", "abcdefghijkl"])
    assert (
        redactor.redact("abcdefghijkl and abcdefgh") == "[REDACTED:literal] and [REDACTED:literal]"
    )
    late = Redactor(["abcdefgh"])
    late.add_literal("abcdefghijkl")
    assert late.redact("abcdefghijkl") == "[REDACTED:literal]"


def test_literals_are_matched_verbatim_not_as_regular_expressions() -> None:
    redactor = Redactor(["p@ss.w0rd*(1)+x"])
    assert redactor.redact("a p@ss.w0rd*(1)+x b") == "a [REDACTED:literal] b"
    assert redactor.redact("p@ssXw0rd*(1)+x") == "p@ssXw0rd*(1)+x"


def test_a_literal_wins_over_the_pattern_it_also_matches() -> None:
    assert Redactor([GITHUB_TOKEN]).redact(f"t={GITHUB_TOKEN}") == "t=[REDACTED:literal]"


def test_literal_added_later_is_honoured_and_none_is_a_noop() -> None:
    redactor = Redactor()
    redactor.add_literal(None)
    redactor.add_literal("")
    assert redactor.redact("plain") == "plain"
    redactor.add_literal("tokenvalue-1")
    assert redactor.redact("x tokenvalue-1") == "x [REDACTED:literal]"


# --------------------------------------------------------------------------- find()
def test_find_returns_kind_names_never_values() -> None:
    text = "\n".join(
        [
            GITHUB_TOKEN,
            AWS_KEY,
            PEM,
            JWT,
            "https://u:pw0rdpw0rd@h/",
            "PASSWORD=abcdefgh1234",
            "mylitvalue1",
        ]
    )
    redactor = Redactor(["mylitvalue1"])
    found = redactor.find(text)

    valid = {name for name, _ in _PATTERNS} | {"literal", "assignment"}
    assert found and set(found) <= valid
    assert {"github-token", "aws-access-key", "private-key", "jwt", "url-credentials"} <= set(found)
    assert {"assignment", "literal"} <= set(found)
    blob = repr(found)
    for value in (
        GITHUB_TOKEN,
        AWS_KEY,
        PEM_BODY,
        JWT,
        "pw0rdpw0rd",
        "abcdefgh1234",
        "mylitvalue1",
    ):
        assert value not in blob


def test_find_reports_each_kind_at_most_once_for_literals_and_assignments() -> None:
    redactor = Redactor(["aaaaaaaa", "bbbbbbbb"])
    text = "aaaaaaaa bbbbbbbb TOKEN=abcdefgh12 PASSWORD=zzzzzzzzzz"
    found = redactor.find(text)
    assert found.count("literal") == 1
    assert found.count("assignment") == 1


def test_find_on_clean_text_is_empty() -> None:
    assert Redactor(["secretvalue"]).find("nothing to see here") == []


# --------------------------------------------------------------------------- diff scanning
def _stage(git: GitRepoEnv, changes: dict[str, str | None]) -> str:
    """Apply ``changes`` (``None`` deletes) to the clone and stage everything."""
    for rel, content in changes.items():
        target = git.clone / rel
        if content is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git.git("add", "-A")
    return git.git("rev-parse", "HEAD")


def _diff(git: GitRepoEnv, base: str, context: int) -> str:
    done = git.git(
        "diff", "--cached", f"-U{context}", "--no-color", "--no-ext-diff", "--no-renames", base
    )
    return done + "\n"


def _lines(*rows: str) -> str:
    return "".join(f"{row}\n" for row in rows)


def _triples(findings: list[SecretFinding]) -> list[tuple[str, str, int]]:
    return sorted((f.kind, f.path, f.line) for f in findings)


@pytest.mark.parametrize("context", [0, 1, 3])
def test_scan_flags_only_added_lines_with_correct_paths_and_line_numbers(
    tmp_path: Path, context: int
) -> None:
    legacy_old = f"legacy token {GITHUB_TOKEN}"
    removed_line = f"removed {AWS_KEY}"
    git = make_git_env(
        tmp_path,
        {
            "docs/legacy.md": _lines("l1", "l2", legacy_old, "l4", "l5", "l6", "l7", "l8"),
            "docs/guide.md": _lines("a", "b", "c", "d", "e", "f", "g", "h"),
            "docs/removed.md": _lines("one", removed_line, "three"),
            "docs/gone.md": _lines("bye", f"gone {SLACK_TOKEN}"),
            "docs/keep.md": _lines("untouched", f"also {JWT}"),
        },
    )
    base = _stage(
        git,
        {
            # the legacy secret stays as a context line, only line 5 changes
            "docs/legacy.md": _lines("l1", "l2", legacy_old, "l4", "l5 changed", "l6", "l7", "l8"),
            # two insertions: after "b" (new line 3) and replacing "g" (new line 8)
            "docs/guide.md": _lines(
                "a", "b", f"added {GITHUB_TOKEN}", "c", "d", "e", "f", f"swap {AWS_KEY}", "h"
            ),
            "docs/removed.md": _lines("one", "three"),
            "docs/gone.md": None,
            "docs/new.md": _lines("x", "y", f"fresh {SLACK_TOKEN}"),
        },
    )

    findings = Redactor().scan_added_lines(_diff(git, base, context))

    assert _triples(findings) == [
        ("aws-access-key", "docs/guide.md", 8),
        ("github-token", "docs/guide.md", 3),
        ("slack-token", "docs/new.md", 3),
    ]
    assert all(isinstance(f, SecretFinding) for f in findings)
    blob = repr(findings)
    for value in (GITHUB_TOKEN, AWS_KEY, SLACK_TOKEN, JWT):
        assert value not in blob


def test_scan_reports_several_kinds_on_one_added_line(tmp_path: Path) -> None:
    git = make_git_env(tmp_path, {"docs/a.md": _lines("x")})
    base = _stage(git, {"docs/a.md": _lines("x", f"two {GITHUB_TOKEN} and {AWS_KEY}")})

    findings = Redactor().scan_added_lines(_diff(git, base, 0))

    assert _triples(findings) == [
        ("aws-access-key", "docs/a.md", 2),
        ("github-token", "docs/a.md", 2),
    ]


def test_scan_handles_paths_with_spaces_and_multiple_hunks(tmp_path: Path) -> None:
    old = [f"row {i}" for i in range(1, 31)]
    git = make_git_env(tmp_path, {"docs/my notes.md": _lines(*old)})
    new = list(old)
    new[1] = f"early {GITHUB_TOKEN}"  # line 2
    new.insert(10, "inserted")  # shifts everything below by one
    new[25] = f"late {AWS_KEY}"  # old line 25 -> new line 26
    base = _stage(git, {"docs/my notes.md": _lines(*new)})

    findings = Redactor().scan_added_lines(_diff(git, base, 1))

    assert _triples(findings) == [
        ("aws-access-key", "docs/my notes.md", 26),
        ("github-token", "docs/my notes.md", 2),
    ]


def test_scan_line_numbers_survive_a_missing_trailing_newline(tmp_path: Path) -> None:
    """``\\ No newline at end of file`` is not a content line and must not shift numbering."""
    git = make_git_env(tmp_path, {"docs/noeol.md": "a\nb"})  # no trailing newline
    base = _stage(git, {"docs/noeol.md": f"a\nb\nsecret {GITHUB_TOKEN}\n"})

    findings = Redactor().scan_added_lines(_diff(git, base, 3))

    assert _triples(findings) == [("github-token", "docs/noeol.md", 3)]


def test_scan_finds_a_secret_on_a_line_that_starts_with_two_plus_signs(tmp_path: Path) -> None:
    """In a diff such a line reads ``+++ ...`` and must not be mistaken for a file header."""
    git = make_git_env(tmp_path, {"docs/a.md": _lines("x", "y")})
    base = _stage(
        git,
        {"docs/a.md": _lines("x", f"++ {GITHUB_TOKEN}", f"after {AWS_KEY}", "y")},
    )

    findings = Redactor().scan_added_lines(_diff(git, base, 0))

    assert _triples(findings) == [
        ("aws-access-key", "docs/a.md", 3),
        ("github-token", "docs/a.md", 2),
    ]


@pytest.mark.parametrize("diff", ["", "\n", "no diff here\n", "diff --git a/x b/x\n"])
def test_scan_of_text_without_added_lines_is_empty(diff: str) -> None:
    assert Redactor().scan_added_lines(diff) == []


def test_scan_uses_registered_literals_too(tmp_path: Path) -> None:
    git = make_git_env(tmp_path, {"docs/a.md": _lines("x")})
    base = _stage(git, {"docs/a.md": _lines("x", "the codeword is plum-tree-42")})

    redactor = Redactor(["plum-tree-42"])

    assert _triples(redactor.scan_added_lines(_diff(git, base, 0))) == [("literal", "docs/a.md", 2)]


# --------------------------------------------------------------------------- fenced()
def _closing_line(opening: str) -> re.Pattern[str]:
    return re.compile(rf" {{0,3}}`{{{len(opening)},}}[ \t]*")


def _parse_fence(block: str) -> tuple[str, str] | None:
    """Minimal CommonMark fenced-code parser: ``(info, content)`` or ``None`` if never closed."""
    lines = block.split("\n")
    match = re.fullmatch(r"( {0,3})(`{3,})([^`]*)", lines[0])
    if match is None:
        return None
    close = _closing_line(match.group(2))
    for index in range(1, len(lines)):
        if close.fullmatch(lines[index]):
            if index != len(lines) - 1:
                return None  # closed early: something follows the closing fence
            return match.group(3), "\n".join(lines[1:index])
    return None


FENCE_TEXTS = [
    "plain",
    "",
    "```",
    "```\nclosing attempt\n```",
    "`" * 7,
    "a `` b ``` c ```` d",
    "~~~\ntilde\n~~~",
    "line\n````python\ncode\n````\nafter\n```",
    "trailing newline\n",
    "  ```  \n",
    "<script>alert(1)</script> @everyone ![img](http://x/y.png)",
]


@pytest.mark.parametrize("text", FENCE_TEXTS)
def test_fenced_text_cannot_close_its_own_fence(text: str) -> None:
    block = fenced(text)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)

    parsed = _parse_fence(block)

    assert parsed is not None, block
    info, content = parsed
    assert info == "text"
    assert content == text.removesuffix("\n")
    assert len(re.match(r"`+", block).group(0)) >= max(3, longest + 1)  # type: ignore[union-attr]


def test_fenced_survives_random_backtick_soup() -> None:
    rng = random.Random(20241024)
    pieces = ["`", "``", "```", "````", "\n", " ", "a", "~~~", "\n```\n", "``` \n"]
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 25)))
        block = fenced(text, language="json")
        parsed = _parse_fence(block)
        assert parsed is not None, (text, block)
        assert parsed[0] == "json"
        assert parsed[1] == (text[:-1] if text.endswith("\n") else text)


def test_fenced_uses_three_backticks_when_the_text_has_none_and_honours_language() -> None:
    assert fenced("a") == "```text\na\n```"
    assert fenced("a\n", language="json") == "```json\na\n```"
    assert fenced("") == "```text\n\n```"
    assert fenced("x ``` y").startswith("````text\n")


# --------------------------------------------------------------------------- truncate / strip
def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("abc", 3) == "abc"
    assert truncate("abc", 10) == "abc"
    assert truncate("", 5) == ""


def test_truncate_cuts_to_exactly_the_limit_with_a_marker() -> None:
    text = "0123456789" * 20
    out = truncate(text, 100)
    assert len(out) == 100
    assert out.endswith("\n... [truncated]")
    assert out.startswith("0123456789")
    assert text.startswith(out[: -len("\n... [truncated]")])


def test_truncate_supports_a_custom_marker() -> None:
    assert truncate("abcdefghij", 8, "...") == "abcde..."
    assert truncate("abcdefghij", 10, "...") == "abcdefghij"


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_truncate_with_a_non_positive_limit_is_empty(limit: int) -> None:
    assert truncate("abc", limit) == ""


@pytest.mark.parametrize("limit", [1, 2, 5, 15, 16, 17, 40])
def test_truncate_never_exceeds_the_limit(limit: int) -> None:
    """The docstring promises *at most* ``limit`` characters, whatever the marker length."""
    assert len(truncate("x" * 500, limit)) <= limit


def test_strip_control_chars_drops_terminal_controls_but_keeps_layout() -> None:
    dirty = "a\x00b\x07c\x08d\x0be\x0cf\x1bg\x7fh\ni\tj"
    assert strip_control_chars(dirty) == "abcdefgh\ni\tj"
    assert strip_control_chars("\x1b[31mred\x1b[0m") == "[31mred[0m"


@pytest.mark.parametrize(
    "text", ["", "plain text", "caf\u00e9 \u4e2d\u6587 \U0001f600", "tab\there\nnewline"]
)
def test_strip_control_chars_keeps_ordinary_text(text: str) -> None:
    assert strip_control_chars(text) == text


# --------------------------------------------------------------------------- ReDoS
class _Deadline(Exception):
    """Raised when a block burns more CPU time than allowed."""


@contextmanager
def cpu_deadline(seconds: float) -> Iterator[None]:
    def on_signal(signum: int, frame: object) -> None:
        raise _Deadline

    previous = signal.signal(signal.SIGPROF, on_signal)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)


# ~1 MB each; every family targets one regular expression's worst case.
HOSTILE_INPUTS: dict[str, Callable[[], str]] = {
    "plain-letters": lambda: "a" * 1_000_000,
    "dotted-words": lambda: "a." * 500_000,
    "dashed-words": lambda: "a-" * 500_000,
    "jwt-like-segments": lambda: "eyJaaaaaaaa-" * 83_000,
    "anthropic-prefix-run": lambda: "sk-ant-" * 143_000,
    "openai-prefix-run": lambda: "sk-" * 333_000,
    "slack-prefix-run": lambda: "xoxb-" * 200_000,
    "credential-word-run": lambda: "TOKEN_" * 167_000,
    "api-key-word-run": lambda: "API_KEY_" * 125_000,
    "assignment-operators": lambda: "TOKEN=" * 167_000,
    "assignment-then-spaces": lambda: "token" + " " * 1_000_000,
    "bearer-run": lambda: "Bearer " * 143_000,
    "bearer-then-long-token": lambda: "Bearer " + "a" * 1_000_000,
    "pem-header-run": lambda: "-----BEGIN " * 91_000,
    "pem-without-end": lambda: "-----BEGIN PRIVATE KEY-----\n" + "A" * 1_000_000,
    "url-userinfo-run": lambda: "a://b:c " * 125_000,
    "github-prefix-run": lambda: "ghp_" * 250_000,
    "whitespace": lambda: " \n" * 500_000,
    "log-like": lambda: "INFO 2024-10-24 request id=123 path=/a/b token=<hidden>\n" * 17_000,
}


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
@pytest.mark.parametrize("name", sorted(HOSTILE_INPUTS))
def test_redaction_of_one_megabyte_of_hostile_text_is_fast(name: str) -> None:
    text = HOSTILE_INPUTS[name]()
    redactor = Redactor(["never-present-literal"])
    for label, call in (("redact", redactor.redact), ("find", redactor.find)):
        started = time.process_time()
        try:
            with cpu_deadline(2.0):
                call(text)
        except _Deadline:
            pytest.fail(f"{label}() burned more than 2 s of CPU on {len(text)} chars of {name!r}")
        elapsed = time.process_time() - started
        assert elapsed < 1.0, f"{label}() needed {elapsed:.2f} s of CPU for {name!r}"


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
def test_literal_redaction_of_a_huge_text_is_fast() -> None:
    text = "padding supersecretvalue-1 " * 40_000
    redactor = Redactor(["supersecretvalue-1", "another-secret-value"])
    started = time.process_time()
    out = redactor.redact(text)
    assert time.process_time() - started < 1.0
    assert "supersecretvalue-1" not in out
