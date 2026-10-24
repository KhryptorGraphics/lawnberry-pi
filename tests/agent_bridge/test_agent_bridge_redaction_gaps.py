"""Redaction gaps found by the PR 17 containment review (finding C-10).

The redactor missed whole families of credentials (HTTP ``Authorization`` headers, npm, GitLab,
Stripe, Hugging Face, Groq, NVIDIA, xAI and Google OAuth tokens, a bare AWS secret key), short
passwords, registered secrets in another case or encoding, the tail of a very long GitHub token,
and secrets broken up by a line break or a zero-width character.  It also never learned the
credentials an operator passes to the agent (``pass_env``).  Secrets are assembled from fragments
so that this file contains no credential-shaped literal.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import signal
import string
import time
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tools.agent_bridge import cli
from tools.agent_bridge.config import BridgeConfig
from tools.agent_bridge.redact import Redactor

ALNUM = string.ascii_letters + string.digits
URLSAFE = ALNUM + "_-"


def chars(count: int, seed: int, alphabet: str = ALNUM) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(count))


GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
AWS_SECRET = "wJalrXUtnFEMI" + "/K7MDENG/bPxRfiCYEXAMPLEKEY"
BASIC_CREDENTIALS = "dXNlcjpodW50ZXIyaHVudGVyMg=="  # base64("user:hunter2hunter2")
NON_GITHUB_TOKEN = "0123456789abcdef" * 3  # no provider prefix: only the header names it

# (name, text, fragment that must not survive, placeholder that must appear)
LEAKS: list[tuple[str, str, str, str]] = [
    (
        "basic-auth-header",
        "Authorization: Basic " + BASIC_CREDENTIALS,
        BASIC_CREDENTIALS[:16],
        "[REDACTED:authorization]",
    ),
    (
        "token-scheme-non-github",
        "Authorization: token " + NON_GITHUB_TOKEN,
        NON_GITHUB_TOKEN[:16],
        "[REDACTED:authorization]",
    ),
    (
        "bearer-scheme-short-token",
        "Authorization: Bearer " + "s3cr3tT0k3n",
        "s3cr3tT0k3n",
        "[REDACTED:authorization]",
    ),
    (
        "proxy-authorization",
        "Proxy-Authorization: Basic " + BASIC_CREDENTIALS,
        BASIC_CREDENTIALS[:16],
        "[REDACTED:authorization]",
    ),
    (
        "authorization-in-a-curl-command",
        "curl -H 'Authorization: Basic " + BASIC_CREDENTIALS + "' https://x.test",
        BASIC_CREDENTIALS[:16],
        "[REDACTED:authorization]",
    ),
    (
        "authorization-in-json",
        json.dumps({"headers": {"Authorization": "Basic " + BASIC_CREDENTIALS}}),
        BASIC_CREDENTIALS[:16],
        "[REDACTED:authorization]",
    ),
    (
        "authorization-in-escaped-json",
        json.dumps(json.dumps({"Authorization": "Bearer " + "short-token-123"})),
        "short-token-123",
        "[REDACTED:authorization]",
    ),
    ("npm-token", "npm_" + chars(36, 1), "npm_", "[REDACTED:npm-token]"),
    ("gitlab-token", "glpat-" + chars(20, 2, URLSAFE), "glpat-", "[REDACTED:gitlab-token]"),
    ("stripe-live-key", "sk_live_" + chars(24, 3), "sk_live_", "[REDACTED:stripe-key]"),
    ("stripe-test-key", "sk_test_" + chars(24, 4), "sk_test_", "[REDACTED:stripe-key]"),
    ("stripe-restricted-key", "rk_live_" + chars(24, 5), "rk_live_", "[REDACTED:stripe-key]"),
    ("huggingface-token", "hf_" + chars(34, 6), "hf_", "[REDACTED:huggingface-token]"),
    ("groq-key", "gsk_" + chars(52, 7), "gsk_", "[REDACTED:groq-key]"),
    ("nvidia-key", "nvapi-" + chars(64, 8, URLSAFE), "nvapi-", "[REDACTED:nvidia-key]"),
    ("xai-key", "xai-" + chars(80, 9), chars(80, 9)[:16], "[REDACTED:xai-key]"),
    (
        "google-oauth-token",
        "ya29." + "a0" + chars(100, 10, URLSAFE),
        "ya29.",
        "[REDACTED:google-oauth-token]",
    ),
    ("bare-aws-secret-key", AWS_SECRET, AWS_SECRET[:20], "[REDACTED:aws-secret-key]"),
    (
        "ghs-token-longer-than-255",
        "ghs_" + chars(300, 11),
        "ghs_",
        "[REDACTED:github-token]",
    ),
    (
        "fine-grained-pat-longer-than-255",
        "github_pat_" + chars(300, 12, ALNUM + "_"),
        "github_pat_",
        "[REDACTED:github-token]",
    ),
]
SURROUNDINGS = [
    ("", ""),
    ("the value is ", " and nothing else"),
    ("(", ")"),
    ('"', '"'),
    ("`", "`"),
    ("line one\n", "\nline three"),
    ("Authorization: ", ", next=header"),
]


@pytest.mark.parametrize(
    ("text", "fragment", "placeholder"),
    [pytest.param(t, f, p, id=name) for name, t, f, p in LEAKS],
)
def test_the_leak_is_redacted(text: str, fragment: str, placeholder: str) -> None:
    redactor = Redactor()

    out = redactor.redact(text)

    assert fragment not in out, out
    assert placeholder in out
    assert redactor.find(text) != []


@pytest.mark.parametrize(
    ("text", "fragment", "placeholder"),
    [
        pytest.param(t, f, p, id=name)
        for name, t, f, p in LEAKS
        if not name.startswith(("authorization", "basic", "token", "bearer", "proxy"))
    ],
)
@pytest.mark.parametrize(("prefix", "suffix"), SURROUNDINGS)
def test_a_token_is_replaced_whole_and_its_surroundings_survive(
    text: str, fragment: str, placeholder: str, prefix: str, suffix: str
) -> None:
    out = Redactor().redact(f"{prefix}{text}{suffix}")

    assert out == f"{prefix}{placeholder}{suffix}"


@pytest.mark.parametrize("length", [255, 256, 257, 300, 5_000, 100_000])
@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"])
def test_a_github_token_is_redacted_to_its_last_character(prefix: str, length: int) -> None:
    token = prefix + chars(length, length)

    assert Redactor().redact(f"x {token} y") == "x [REDACTED:github-token] y"


def test_the_authorization_header_keeps_the_text_around_it() -> None:
    header = f"Authorization: Basic {BASIC_CREDENTIALS}"
    text = f"GET / HTTP/1.1\nHost: example.test\n{header}\nAccept: */*\n"

    assert Redactor().redact(text) == (
        "GET / HTTP/1.1\nHost: example.test\n[REDACTED:authorization]\nAccept: */*\n"
    )


def test_a_bearer_token_in_a_header_is_still_reported_as_a_bearer_token() -> None:
    long_token = "Bearer " + "abcdefghijklmnopqrstuvwxyz0123456789"

    assert Redactor().redact("Authorization: " + long_token) == (
        "Authorization: [REDACTED:bearer-token]"
    )


# ---------------------------------------------------------------------------- near misses
PROSE_AND_CODE = [
    "The token is stored in the keychain and expires after one hour.",
    "Use a bearer token in the Authorization header.",
    "Basic usage of the tool: run it without arguments.",
    "Basic configuration options are described below.",
    "Authorization header carries the credentials.",
    "Authorization: Bearer <your-token>",
    "Authorization: token ${GITHUB_TOKEN}",
    "Authorization: token $TOKEN",
    "Authorization: Basic base64(user:pass)",
    "Authorization: Bearer",
    "token: str",
    "password: str",
    "password: bytes",
    "password: bool = False",
    "def login(user: str, password: str, token: str) -> None:",
    "password=None",
    "password: null",
    "password = input()",
    "password=args.pw",
    "password=abc",
    "pwd = hunter",
    "set the password to something long and random",
    "npm_config_registry and npm_package_version are environment variables",
    "export HF_HOME=/data/huggingface and use the hf_transfer extra",
    "gsk_ and nvapi- and xai- and glpat- and ya29. are bare prefixes",
    "see docs/xai-integration-and-evaluation-strategy-notes.md",
    "the nvapi-style key header",
    "git commit 356a192b7913b04c54574d18c28d46e6395428ab fixes it",
    "sha1: da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "digest sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "id 123e4567-e89b-12d3-a456-426614174000",
    "frontend/src/components/ControlView/Panel.vue",
    "backend/src/services/MotorService/handlers.py",
    "AbstractSingletonProxyFactoryBeanImplementationX",
    "ThisIsAVeryLongCamelCaseIdentifierWithoutDigits",
    "0123456789012345678901234567890123456789",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMN",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmn",
    "{'maxOutputTokens': 32000, 'password': None, 'token': null}",
]


@pytest.mark.parametrize("text", PROSE_AND_CODE)
def test_ordinary_prose_and_code_of_a_similar_shape_is_untouched(text: str) -> None:
    redactor = Redactor()

    assert redactor.redact(text) == text
    assert redactor.find(text) == []


def test_a_long_markdown_document_is_untouched() -> None:
    document = "\n".join(
        [
            "# Agent bridge notes",
            "",
            "The bridge never stores a token; it reads one from the environment.",
            "- password: str is a type hint, token: str as well",
            "- see `docs/agent-bridge.md`",
            "",
            *[
                f"Paragraph {index} wraps a long line of words\nacross two lines."
                for index in range(200)
            ],
            "Done.",
        ]
    )

    assert Redactor().redact(document) == document
    assert Redactor().find(document) == []


# --------------------------------------------------------------------------- short passwords
SHORT_PASSWORDS = [
    ("password=hunter2", "password=[REDACTED:assignment]"),
    ("PASSWORD: hunter2", "PASSWORD: [REDACTED:assignment]"),
    ("DB_PASSWORD=hunter2", "DB_PASSWORD=[REDACTED:assignment]"),
    ('password = "hunter2"', 'password = "[REDACTED:assignment]"'),
    ("password: 'hunter'", "password: '[REDACTED:assignment]'"),
    ('{"password": "hunter2"}', '{"password": "[REDACTED:assignment]"}'),
    ("passwd=p@ss!", "passwd=[REDACTED:assignment]"),
    ("password=1234", "password=[REDACTED:assignment]"),
    ("--password=hunter2 --user=bob", "--password=[REDACTED:assignment] --user=bob"),
]


@pytest.mark.parametrize(("text", "expected"), SHORT_PASSWORDS)
def test_short_passwords_are_redacted(text: str, expected: str) -> None:
    redactor = Redactor()

    assert redactor.redact(text) == expected
    assert redactor.find(text) == ["assignment"]


@pytest.mark.parametrize(
    "text",
    [
        "token=1234567",
        "secret=hunter2",
        "api_key=hunter2",
        "password: abcdefg",
        "password=secret",
    ],
)
def test_the_short_value_rule_is_limited_to_passwords_that_look_like_passwords(text: str) -> None:
    assert Redactor().redact(text) == text


# ---------------------------------------------------------------------------- literals
SECRET = "Correct-Horse-Battery-Staple-42"
SPECIAL = "p@ss w0rd/with+special&chars=1"
QUOTED = 'pa"ss\\word-9876 \u00e9'


def b64_variants(secret: str) -> list[str]:
    data = secret.encode()
    return [
        base64.b64encode(data).decode(),
        base64.b64encode(data).decode().rstrip("="),
        base64.urlsafe_b64encode(data).decode(),
        base64.urlsafe_b64encode(data).decode().rstrip("="),
    ]


@pytest.mark.parametrize("variant", ["upper", "lower", "swapcase", "title"])
def test_a_registered_literal_is_found_in_any_case(variant: str) -> None:
    text = f"leaked: {getattr(SECRET, variant)()} here"
    redactor = Redactor([SECRET])

    assert redactor.redact(text) == "leaked: [REDACTED:literal] here"
    assert redactor.find(text) == ["literal"]


def test_the_exact_literal_still_works_and_other_text_is_untouched() -> None:
    redactor = Redactor([SECRET])

    assert redactor.redact(f"a {SECRET} b {SECRET}") == "a [REDACTED:literal] b [REDACTED:literal]"
    assert redactor.redact("correct horse battery staple") == "correct horse battery staple"


@pytest.mark.parametrize("encoded", b64_variants(SECRET))
def test_the_base64_form_of_a_literal_is_redacted(encoded: str) -> None:
    redactor = Redactor([SECRET])

    assert redactor.redact(f"data: {encoded} end") == "data: [REDACTED:literal] end"
    assert redactor.find(f"data: {encoded}") == ["literal"]


@pytest.mark.parametrize("lead", range(6))
def test_a_literal_inside_a_longer_base64_string_is_redacted(lead: int) -> None:
    """The base64 of ``<lead><secret><trail>`` has a core that only the secret determines."""
    secret = SECRET.encode()
    for trail in (b"", b"!", b"?!", b":x:"):
        blob = base64.b64encode(b"u" * lead + secret + trail).decode()
        first = (lead // 3) * 4 + (0, 2, 3)[lead % 3]  # leading characters the lead bytes share
        total = lead + len(secret)
        last = (total // 3) * 4 + (0, 1, 2)[total % 3]  # trailing ones the trail bytes share

        out = Redactor([SECRET]).redact(f"blob={blob};")

        assert "[REDACTED:literal]" in out, (lead, trail, blob)
        assert blob[first:last] not in out
        assert out.startswith("blob=" + blob[:first]) and out.endswith(";")


def test_the_base64_of_user_and_password_leaves_only_the_user_prefix() -> None:
    blob = base64.b64encode(b"user:" + SECRET.encode()).decode()

    out = Redactor([SECRET]).redact(f"X-Auth: {blob}")

    assert out == f"X-Auth: {blob[:7]}[REDACTED:literal]"  # 'dXNlcjp' only encodes `user:`


@pytest.mark.parametrize(
    "form",
    [
        urllib.parse.quote(SPECIAL, safe=""),
        urllib.parse.quote(SPECIAL, safe="").lower(),
        urllib.parse.quote(SPECIAL),
        urllib.parse.quote_plus(SPECIAL),
        urllib.parse.quote_plus(SPECIAL).upper(),
    ],
)
def test_the_url_encoded_form_of_a_literal_is_redacted(form: str) -> None:
    redactor = Redactor([SPECIAL])

    assert redactor.redact(f"https://x.test/?q={form}&z=1") == (
        "https://x.test/?q=[REDACTED:literal]&z=1"
    )
    assert redactor.redact(f"plain {SPECIAL} too") == "plain [REDACTED:literal] too"


def test_the_json_escaped_form_of_a_literal_is_redacted() -> None:
    redactor = Redactor([QUOTED])

    for spelling in (json.dumps(QUOTED), json.dumps(QUOTED, ensure_ascii=False)):
        out = redactor.redact(f'{{"v": {spelling}}}')
        assert out == '{"v": "[REDACTED:literal]"}', out


def test_a_literal_with_surrounding_whitespace_is_found_without_it() -> None:
    redactor = Redactor([f"  {SECRET}\n"])

    assert redactor.redact(f"x {SECRET} y") == "x [REDACTED:literal] y"


def test_short_literals_are_still_ignored_in_every_form() -> None:
    redactor = Redactor(["abc1234", "SHORT"])

    assert redactor.redact("ABC1234 YWJjMTIzNA== SHORT short") == "ABC1234 YWJjMTIzNA== SHORT short"


def test_overlapping_forms_of_one_literal_are_replaced_once() -> None:
    encoded = base64.b64encode(SECRET.encode()).decode()

    assert Redactor([SECRET]).redact(f"{SECRET}{encoded}") == (
        "[REDACTED:literal][REDACTED:literal]"
    )
    assert Redactor([SECRET, SECRET.upper()]).redact(f"x {SECRET} y") == "x [REDACTED:literal] y"


def test_partially_overlapping_literals_leave_no_tail() -> None:
    redactor = Redactor(["abcdefgh", "efghijkl"])

    assert redactor.redact("abcdefghijkl") == "[REDACTED:literal]"


def test_literals_added_later_get_every_form_too() -> None:
    redactor = Redactor()
    redactor.add_literal(SECRET)

    assert redactor.redact(SECRET.upper()) == "[REDACTED:literal]"
    assert redactor.redact(base64.b64encode(SECRET.encode()).decode()) == "[REDACTED:literal]"


# ------------------------------------------------------------- broken-up secrets
BREAKS = [
    pytest.param("\n", id="newline"),
    pytest.param("\r\n", id="crlf"),
    pytest.param("\n    ", id="newline-and-indent"),
    pytest.param(" \n  \n ", id="blank-lines"),
    pytest.param("\u200b", id="zero-width-space"),
    pytest.param("\u200c\u200d", id="zero-width-non-joiner-and-joiner"),
    pytest.param("\u2060", id="word-joiner"),
    pytest.param("\ufeff", id="byte-order-mark"),
    pytest.param("\u00ad", id="soft-hyphen"),
    pytest.param("\u202e", id="right-to-left-override"),
    pytest.param("\u2028", id="line-separator"),
    pytest.param("\n\u200b\n", id="newline-and-zero-width"),
    pytest.param("\ufe0f", id="variation-selector"),
    pytest.param("\U000e0020", id="tag-character"),
]


def split_at(token: str, position: int, separator: str) -> str:
    return token[:position] + separator + token[position:]


@pytest.mark.parametrize("separator", BREAKS)
@pytest.mark.parametrize("position", [5, 12, 20, 27, 35])
def test_a_github_token_split_by_a_break_is_redacted(separator: str, position: int) -> None:
    text = f"token {split_at(GITHUB_TOKEN, position, separator)} end"
    redactor = Redactor()

    out = redactor.redact(text)

    assert GITHUB_TOKEN[:12] not in out and GITHUB_TOKEN[-12:] not in out, out
    assert out.startswith("token ") and out.endswith(" end")
    assert "[REDACTED:github-token]" in out
    assert "github-token" in redactor.find(text)


@pytest.mark.parametrize("separator", BREAKS)
def test_a_registered_literal_split_by_a_break_is_redacted(separator: str) -> None:
    literal = "plum-tree-42-ripe-fruit"
    text = f"the codeword is {split_at(literal, 9, separator)}."
    redactor = Redactor([literal])

    out = redactor.redact(text)

    assert out == "the codeword is [REDACTED:literal]."
    assert redactor.find(text) == ["literal"]


@pytest.mark.parametrize("separator", ["\n", "\u200b"])
def test_the_encoded_forms_of_a_literal_may_be_split_too(separator: str) -> None:
    encoded = base64.b64encode(SECRET.encode()).decode()

    out = Redactor([SECRET]).redact(f"a {split_at(encoded, 14, separator)} b")

    assert out == "a [REDACTED:literal] b"


@pytest.mark.parametrize("separator", ["\n", "\u200b"])
@pytest.mark.parametrize(
    ("secret", "placeholder"),
    [
        ("npm_" + chars(36, 21), "[REDACTED:npm-token]"),
        ("sk-ant-" + "api03-" + chars(40, 22, URLSAFE), "[REDACTED:anthropic-key]"),
        ("AKIA" + "IOSFODNN7EXAMPLE", "[REDACTED:aws-access-key]"),
        ("hf_" + chars(34, 23), "[REDACTED:huggingface-token]"),
        ("Bearer " + chars(40, 24), "[REDACTED:bearer-token]"),
    ],
    ids=["npm", "anthropic", "aws-access-key", "huggingface", "bearer"],
)
def test_other_token_shapes_split_by_a_break_are_redacted(
    secret: str, placeholder: str, separator: str
) -> None:
    out = Redactor().redact(f"x {split_at(secret, len(secret) // 2, separator)} y")

    assert out == f"x {placeholder} y"


def test_a_secret_whose_head_is_redacted_on_its_own_loses_at_least_its_head() -> None:
    head = "ghp_" + chars(40, 31)
    tail = chars(10, 32)

    out = Redactor().redact(f"{head}\n{tail}")

    assert head not in out and out.startswith("[REDACTED:github-token]")


def test_lines_that_each_hold_a_whole_token_are_each_redacted() -> None:
    lines = ["ghp_" + chars(36, index) for index in range(50)]

    out = Redactor().redact("\n".join(lines) + "\n")

    assert out == "[REDACTED:github-token]\n" * 50


def test_breaks_between_ordinary_words_change_nothing() -> None:
    text = "alpha\nbeta gamma\n\n   delta\u200bepsilon\r\nzeta\n"

    assert Redactor(["never-present-secret"]).redact(text) == text


def test_a_plain_space_is_not_a_break() -> None:
    """Only line breaks and invisible characters are looked through, never a visible gap."""
    prefix, rest = GITHUB_TOKEN[:20], GITHUB_TOKEN[20:]

    assert Redactor().redact(f"{prefix} {rest}") == f"{prefix} {rest}"
    assert Redactor().redact(f"{prefix}\t{rest}") == f"{prefix}\t{rest}"


def test_find_ignores_text_without_secrets_even_when_it_has_breaks() -> None:
    assert Redactor(["never-present-secret"]).find("one\ntwo\u200bthree\r\nfour") == []


# ----------------------------------------------------------------- redactor in the CLI
@pytest.fixture
def restored_bridge_logger() -> Iterator[None]:
    logger = logging.getLogger(cli.BRIDGE_LOGGER) if hasattr(cli, "BRIDGE_LOGGER") else None
    logger = logger or logging.getLogger("agent_bridge")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


def build_redactor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config: Any) -> Redactor:
    monkeypatch.setattr(cli, "HttpGitHubClient", _FakeClient)
    monkeypatch.setenv("LAWNBERRY_BRIDGE_GITHUB_TOKEN", GITHUB_TOKEN)
    cfg = BridgeConfig.model_validate(
        {
            "repository": "owner/repo",
            "approvers": ["alice"],
            "repo_path": str(tmp_path),
            "state_dir": str(tmp_path / "state"),
            **config,
        }
    )
    runtime = cli.build_runtime(cfg, execute=False)
    runtime.ledger.close()
    return runtime.redactor


PROVIDER_KEY = "provider-key-VALUE-1234567890"
OTHER_KEY = "another-secret-Value-0987654321"
EXTRA_KEY = "extra-secret-Value-5555555555"


def test_the_cli_redactor_knows_the_values_passed_to_every_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored_bridge_logger: None
) -> None:
    monkeypatch.setenv("MY_PROVIDER_KEY", PROVIDER_KEY)
    monkeypatch.setenv("MY_OTHER_KEY", OTHER_KEY)
    monkeypatch.setenv("MY_EXTRA_KEY", EXTRA_KEY)
    redactor = build_redactor(
        tmp_path,
        monkeypatch,
        agent={
            "omp": {"binary": "omp", "pass_env": ["MY_PROVIDER_KEY"]},
            "claude": {"binary": "claude", "pass_env": ["MY_OTHER_KEY", "MY_ABSENT_KEY"]},
        },
        extra_secret_env=["MY_EXTRA_KEY"],
    )

    text = f"a {PROVIDER_KEY} b {OTHER_KEY.upper()} c {EXTRA_KEY} d {GITHUB_TOKEN}"

    assert redactor.redact(text) == "a [REDACTED:literal] b [REDACTED:literal] c " + (
        "[REDACTED:literal] d [REDACTED:literal]"
    )


def test_the_default_claude_pass_env_is_registered_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored_bridge_logger: None
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", PROVIDER_KEY)

    redactor = build_redactor(tmp_path, monkeypatch)

    assert redactor.redact(f"key {PROVIDER_KEY}") == "key [REDACTED:literal]"


def test_unset_empty_and_short_values_are_skipped_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored_bridge_logger: None
) -> None:
    monkeypatch.setenv("MY_EMPTY", "")
    monkeypatch.setenv("MY_SHORT", "short")
    monkeypatch.delenv("MY_UNSET", raising=False)

    redactor = build_redactor(
        tmp_path,
        monkeypatch,
        agent={"omp": {"binary": "omp", "pass_env": ["MY_EMPTY", "MY_SHORT", "MY_UNSET"]}},
    )

    assert redactor.redact("short and empty stay") == "short and empty stay"


# ---------------------------------------------------------------------------- linear time
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


HOSTILE: dict[str, Callable[[], str]] = {
    "npm-prefix-run": lambda: "npm_" * 250_000,
    "hf-prefix-run": lambda: "hf_" * 333_000,
    "gsk-prefix-run": lambda: "gsk_" * 250_000,
    "nvapi-prefix-run": lambda: "nvapi-" * 166_000,
    "xai-prefix-run": lambda: "xai-" * 250_000,
    "ya29-prefix-run": lambda: "ya29." * 200_000,
    "glpat-prefix-run": lambda: "glpat-" * 166_000,
    "stripe-prefix-run": lambda: "sk_live_" * 125_000,
    "stripe-short-body": lambda: ("sk_live_" + "a" * 15 + " ") * 40_000,
    "authorization-run": lambda: "Authorization: " * 66_000,
    "authorization-then-spaces": lambda: "Authorization:" + " " * 1_000_000,
    "authorization-basic-run": lambda: "Authorization: Basic " * 47_000,
    "authorization-colon-only": lambda: "authorization" * 77_000 + ":",
    "aws-secret-sized-runs": lambda: ("aB3" * 13 + "a.") * 24_000,
    "aws-secret-exact-no-digit": lambda: ("aBcD" * 10 + " ") * 24_000,
    "aws-secret-one-long-run": lambda: "aB3/" * 250_000,
    "github-long-tail": lambda: "ghp_" + "a" * 1_000_000,
    "github-fine-grained-tail": lambda: "github_pat_" + "a_" * 500_000,
    "password-run": lambda: "password=" * 111_000,
    "password-then-spaces": lambda: "password" + " " * 1_000_000,
    "password-short-values": lambda: "password=ab1 " * 77_000,
    "newlines-between-letters": lambda: "a\n" * 500_000,
    "zero-width-between-letters": lambda: "a\u200b" * 500_000,
    "breaks-between-token-prefixes": lambda: "ghp_\n" * 200_000,
    "breaks-between-token-bodies": lambda: ("ghp_" + "a" * 12 + "\n") * 55_000,
    "bearer-split-forever": lambda: "Bearer\n" * 140_000,
    "whitespace-then-break": lambda: "a" + " " * 500_000 + "\n" + "b",
    "break-runs": lambda: "a" + "\n\u200b" * 250_000 + "b",
    "alternating-words-and-breaks": lambda: "ab \ncd\n" * 140_000,
    "case-fold-expansion": lambda: "\u0130" * 500_000,
    "base64-lookalikes": lambda: "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 28_000,
}


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_one_megabyte_of_hostile_text_redacts_in_linear_time(name: str) -> None:
    text = HOSTILE[name]()
    redactor = Redactor(["never-present-literal", SECRET, SPECIAL])
    for label, call in (("redact", redactor.redact), ("find", redactor.find)):
        started = time.process_time()
        try:
            with cpu_deadline(3.0):
                call(text)
        except _Deadline:
            pytest.fail(f"{label}() burned more than 3 s of CPU on {len(text)} chars of {name!r}")
        elapsed = time.process_time() - started
        assert elapsed < 1.5, f"{label}() needed {elapsed:.2f} s of CPU for {name!r}"


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
def test_a_hostile_text_with_a_split_secret_in_it_is_still_redacted_quickly() -> None:
    noise = "a\n" * 200_000
    text = noise + split_at(GITHUB_TOKEN, 20, "\n") + "\n" + noise
    started = time.process_time()

    out = Redactor().redact(text)

    assert time.process_time() - started < 1.5
    assert "[REDACTED:github-token]" in out and GITHUB_TOKEN[:12] not in out


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
def test_many_registered_literals_stay_cheap_on_a_large_text() -> None:
    literals = [f"registered-secret-{index:04d}-{chars(12, index)}" for index in range(30)]
    text = "ordinary log line with nothing to hide\n" * 20_000
    started = time.process_time()

    out = Redactor(literals).redact(text)

    assert out == text
    assert time.process_time() - started < 1.5
