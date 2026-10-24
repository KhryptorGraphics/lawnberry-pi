"""Tests for ``tools/agent_bridge/schema.py``: strict task parsing, hygiene, revisions."""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pydantic
import pytest
import yaml

from tools.agent_bridge.config import BridgeConfig
from tools.agent_bridge.errors import TaskValidationError
from tools.agent_bridge.schema import (
    MAX_BLOCK_CHARS,
    SCHEMA_VERSION,
    ParsedTask,
    TaskSpec,
    compute_revision,
    find_task_blocks,
    hygiene_errors,
    normalize_text,
    parse_task,
    strict_yaml_load,
    validate_against_config,
)

OBJECTIVE = "Add a short note about the agent bridge to docs/guide.md."
MINIMAL_YAML = f"version: 1\nobjective: {OBJECTIVE}"
TITLE = "Document the bridge"


def fenced_block(content: str, *, fence: str = "```", info: str = "yaml agent-task") -> str:
    return f"{fence}{info}\n{content}\n{fence}"


def issue_body(content: str = MINIMAL_YAML, *, prose: str = "Please do this.", **fence: str) -> str:
    return f"{prose}\n\n{fenced_block(content, **fence)}\n"


def task_yaml(**fields: Any) -> str:
    """YAML for a task; a field set to ``...`` is left out."""
    data: dict[str, Any] = {"version": 1, "objective": OBJECTIVE}
    for key, value in fields.items():
        if value is ...:
            data.pop(key, None)
        else:
            data[key] = value
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()


def parse(content: str = MINIMAL_YAML, *, title: str = TITLE, **fence: str) -> ParsedTask:
    return parse_task(title, issue_body(content, **fence))


def problems(title: str, body: str) -> list[str]:
    with pytest.raises(TaskValidationError) as info:
        parse_task(title, body)
    assert info.value.errors and all(isinstance(e, str) and e for e in info.value.errors)
    assert str(info.value) == "; ".join(info.value.errors)
    return info.value.errors


def reject(content: str, *, expect: str | None = None, **fence: str) -> list[str]:
    errors = problems(TITLE, issue_body(content, **fence))
    if expect is not None:
        assert any(expect in e for e in errors), errors
    return errors


def make_config(**overrides: Any) -> BridgeConfig:
    data = {"repository": "owner/repo", "approvers": ["alice"], **overrides}
    return BridgeConfig.model_validate(data)


# --------------------------------------------------------------------------- happy path
def test_minimal_task_parses_with_safe_defaults() -> None:
    task = parse()

    assert isinstance(task, ParsedTask)
    assert task.title == TITLE
    assert task.spec.version == SCHEMA_VERSION == 1
    assert task.spec.objective == OBJECTIVE
    assert task.spec.scope == "docs-only"
    assert task.spec.agent is None
    assert task.spec.allowed_paths == [] and task.spec.context_files == []
    assert task.spec.acceptance_criteria == [] and task.spec.checks == []
    assert task.spec.budget_usd is None and task.spec.timeout_minutes is None
    assert task.spec.notes is None
    assert len(task.revision) == 64 and len(task.spec_hash) == 64
    int(task.revision, 16), int(task.spec_hash, 16)


def test_fully_specified_task_is_normalised() -> None:
    content = task_yaml(
        scope="code",
        agent="claude",
        allowed_paths=["./docs/**", "tests/*.py"],
        context_files=["./docs/guide.md", "docs//api.md"],
        acceptance_criteria=["  keeps links working  ", "adds a test"],
        checks=["lint", "unit-tests"],
        budget_usd=1.5,
        timeout_minutes=30,
        notes="  remember the changelog  ",
    )

    spec = parse(content).spec

    assert spec.scope == "code" and spec.agent == "claude"
    assert spec.allowed_paths == ["docs/**", "tests/*.py"]
    assert spec.context_files == ["docs/guide.md", "docs/api.md"]
    assert spec.acceptance_criteria == ["keeps links working", "adds a test"]
    assert spec.checks == ["lint", "unit-tests"]
    assert spec.budget_usd == 1.5 and spec.timeout_minutes == 30
    assert spec.notes == "remember the changelog"


def test_title_is_normalised_and_parsed_task_is_immutable() -> None:
    task = parse(title="  Title with spaces \r\n")
    assert task.title == "Title with spaces"
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.title = "other"  # type: ignore[misc]


def test_prose_around_the_block_is_ignored() -> None:
    block = fenced_block(MINIMAL_YAML)
    body = f"## Heading\n\nSome prose, a [link](http://x) and `code`.\n\n{block}\n\nMore prose."
    assert parse_task(TITLE, body).spec.objective == OBJECTIVE


# --------------------------------------------------------------------------- fence detection
@pytest.mark.parametrize(
    "info",
    [
        "yaml agent-task",
        "agent-task",
        "yml agent-task",
        "YAML agent-task",
        "Yml agent-task",
        "yaml   agent-task",
        "yaml\tagent-task",
        "yaml agent-task  ",
    ],
)
def test_recognised_info_strings_open_a_task_block(info: str) -> None:
    body = issue_body(info=info)
    assert find_task_blocks(body) == [MINIMAL_YAML]
    assert parse_task(TITLE, body).spec.objective == OBJECTIVE


@pytest.mark.parametrize(
    "info",
    [
        "yaml",
        "json agent-task",
        "agent-task yaml",
        "yaml agent-task extra",
        "agent-tasks",
        "yaml Agent-Task",
        "yaml,agent-task",
        "{.yaml .agent-task}",
        "xagent-task",
        "",
    ],
)
def test_other_info_strings_are_not_task_blocks(info: str) -> None:
    body = issue_body(info=info)
    assert find_task_blocks(body) == []
    assert "found 0" in problems(TITLE, body)[0]


def test_tilde_fences_are_supported() -> None:
    body = issue_body(fence="~~~")
    assert find_task_blocks(body) == [MINIMAL_YAML]
    assert parse_task(TITLE, body).spec.objective == OBJECTIVE


def test_a_longer_fence_may_contain_shorter_fence_lines() -> None:
    content = task_yaml(notes="before\n```\nfence line inside the notes\n```\nafter")
    body = issue_body(content, fence="````")

    assert find_task_blocks(body) == [content]
    assert "fence line inside the notes" in parse_task(TITLE, body).spec.notes  # type: ignore[operator]


def test_a_task_block_nested_in_another_fence_is_just_text() -> None:
    body = f"````md\n{fenced_block(MINIMAL_YAML)}\n````\n"
    assert find_task_blocks(body) == []
    assert "found 0" in problems(TITLE, body)[0]


def test_three_space_indent_is_still_a_fence_but_four_spaces_is_code() -> None:
    three = "\n".join(f"   {line}" for line in fenced_block(MINIMAL_YAML).splitlines())
    four = "\n".join(f"    {line}" for line in fenced_block(MINIMAL_YAML).splitlines())

    assert len(find_task_blocks(three)) == 1
    assert find_task_blocks(four) == []
    assert "found 0" in problems(TITLE, four)[0]


def test_a_fence_inside_a_blockquote_is_ignored() -> None:
    quoted = "\n".join(f"> {line}" for line in fenced_block(MINIMAL_YAML).splitlines())
    assert find_task_blocks(quoted) == []
    assert "found 0" in problems(TITLE, quoted)[0]


def test_an_unclosed_fence_is_ignored() -> None:
    body = f"```yaml agent-task\n{MINIMAL_YAML}\n"
    assert find_task_blocks(body) == []
    assert "found 0" in problems(TITLE, body)[0]


@pytest.mark.parametrize(
    "closing",
    ["``", "~~~", "``` yaml", "``` x"],
    ids=["too-short", "wrong-character", "closing-with-info", "closing-with-text"],
)
def test_only_a_proper_closing_fence_ends_the_block(closing: str) -> None:
    body = f"```yaml agent-task\n{MINIMAL_YAML}\n{closing}\n"
    assert find_task_blocks(body) == []


def test_tilde_block_cannot_be_closed_by_backticks() -> None:
    body = f"~~~yaml agent-task\n{MINIMAL_YAML}\n```\n~~~\n"
    assert find_task_blocks(body) == [f"{MINIMAL_YAML}\n```"]
    problems(TITLE, body)  # the stray fence line is not valid YAML


def test_crlf_and_lone_cr_line_endings_are_accepted() -> None:
    for newline in ("\r\n", "\r"):
        body = issue_body().replace("\n", newline)
        assert parse_task(TITLE, body).spec.objective == OBJECTIVE


def test_inline_triple_backticks_do_not_open_a_fence() -> None:
    body = "use ```code``` here\n\n" + fenced_block(MINIMAL_YAML)
    assert find_task_blocks(body) == [MINIMAL_YAML]


def test_other_fenced_blocks_are_ignored() -> None:
    body = "```bash\nls -la\n```\n\n" + fenced_block(MINIMAL_YAML) + "\n\n```json\n{}\n```"
    assert find_task_blocks(body) == [MINIMAL_YAML]
    assert parse_task(TITLE, body).spec.objective == OBJECTIVE


def test_exactly_one_block_is_required() -> None:
    assert "found 0" in problems(TITLE, "Just prose, no block.")[0]
    assert "found 0" in problems(TITLE, "")[0]
    two = fenced_block(MINIMAL_YAML) + "\n\n" + fenced_block(MINIMAL_YAML)
    assert "found 2" in problems(TITLE, two)[0]
    other = fenced_block(task_yaml(notes="second one"), info="agent-task")
    assert "found 2" in problems(TITLE, fenced_block(MINIMAL_YAML) + "\n" + other)[0]


# --------------------------------------------------------------------------- strict YAML
def _with_base(extra: str) -> str:
    return f"{MINIMAL_YAML}\n{extra}"


STRICT_REJECTIONS = [
    ("anchor-and-alias", _with_base("x: &a 1\ny: *a"), "anchors and aliases"),
    ("anchor-only", _with_base("x: &a 1"), "anchors and aliases"),
    ("alias-only", _with_base("y: *nothing"), "anchors and aliases"),
    ("anchored-mapping", _with_base("x: &a {k: v}"), "anchors and aliases"),
    ("duplicate-key", _with_base(f"objective: {OBJECTIVE} again"), "duplicate key 'objective'"),
    ("integer-key", _with_base("1: x"), "keys must be strings"),
    ("boolean-key", _with_base("true: x"), "keys must be strings"),
    ("null-key", _with_base("null: x"), "keys must be strings"),
    ("float-key", _with_base("1.5: x"), "keys must be strings"),
    ("complex-key", _with_base("? [a, b]\n: x"), "keys must be strings"),
    ("binary-key", _with_base("!!binary YQ==: x"), "keys must be strings"),
    ("python-object-tag", _with_base("x: !!python/object:os.system {}"), "not valid YAML"),
    ("python-name-tag", _with_base("x: !!python/name:os.system ''"), "not valid YAML"),
    ("custom-tag", _with_base("x: !custom 1"), "not valid YAML"),
    ("custom-tag-on-mapping", "!!java.util.HashMap {a: 1}", "not valid YAML"),
    ("merge-key", _with_base("<<: {a: 1}"), "not valid YAML"),
    ("second-document", "---\nversion: 1\n---\nversion: 2", "not valid YAML"),
    ("tab-indentation", "version: 1\nobjective:\n\t- x", "not valid YAML"),
    ("unclosed-flow", "version: 1\nobjective: [unclosed", "not valid YAML"),
    ("unclosed-quote", 'version: 1\nobjective: "unclosed', "not valid YAML"),
    ("bad-indentation", "version: 1\n  objective: x\n y: 2", "not valid YAML"),
    ("list-top-level", "- a\n- b", "must be a YAML mapping"),
    ("scalar-top-level", "just some words", "must be a YAML mapping"),
    ("number-top-level", "42", "must be a YAML mapping"),
    ("null-top-level", "null", "must be a YAML mapping"),
    ("empty-block", "", "must be a YAML mapping"),
    ("comment-only", "# nothing here", "must be a YAML mapping"),
    ("deeply-nested-flow-sequence", "[" * 5000 + "]" * 5000, "nested too deeply"),
    ("deeply-nested-flow-mapping", "{a: " * 3000 + "1" + "}" * 3000, "nested too deeply"),
]


@pytest.mark.parametrize(
    ("content", "message"),
    [(content, message) for _, content, message in STRICT_REJECTIONS],
    ids=[name for name, _, _ in STRICT_REJECTIONS],
)
def test_strict_yaml_rejections(content: str, message: str) -> None:
    errors = reject(content, expect=message)
    assert all(len(e) < 400 for e in errors)  # useful, bounded messages


def test_block_larger_than_the_limit_is_rejected_but_the_limit_itself_is_fine() -> None:
    limit = MAX_BLOCK_CHARS
    assert limit == 20_000
    assert strict_yaml_load("a: " + "x" * (limit - 3)) == {"a": "x" * (limit - 3)}
    with pytest.raises(TaskValidationError, match="20000"):
        strict_yaml_load("a: " + "x" * (limit - 2))
    reject(f"{MINIMAL_YAML}\nnotes: {'x' * limit}", expect="longer than 20000")


def test_strict_yaml_load_accepts_plain_documents_and_reports_the_line_of_errors() -> None:
    assert strict_yaml_load("a: 1\nb: [x, y]\n") == {"a": 1, "b": ["x", "y"]}
    with pytest.raises(TaskValidationError, match=r"\(line 2\)"):
        strict_yaml_load("a: 1\n\tb: 2\n")


def test_nul_bytes_in_the_yaml_are_a_validation_error() -> None:
    with pytest.raises(TaskValidationError):
        strict_yaml_load("a: b\x00c")


UNHANDLED_SCALAR_CASES = [
    ("timestamp-invalid-month", "x: 2020-13-45"),
    ("timestamp-invalid-hour", "x: 2001-12-14 25:00:00"),
    ("explicit-int-garbage", "x: !!int abc"),
    ("explicit-int-without-value", "x: !!int"),
    ("explicit-set-with-scalar", "x: !!set 1e999"),
    ("explicit-float-garbage", "x: !!float abc"),
    ("explicit-bool-garbage", "x: !!bool maybe"),
    ("explicit-timestamp-garbage", "x: !!timestamp foo"),
    ("integer-with-too-many-digits", "x: 1" + "0" * 5000),
    ("tagged-key-garbage", "!!int abc: 1"),
]


@pytest.mark.parametrize(
    "content",
    [content for _, content in UNHANDLED_SCALAR_CASES],
    ids=[name for name, _ in UNHANDLED_SCALAR_CASES],
)
def test_scalars_that_fail_to_construct_are_validation_errors_not_crashes(content: str) -> None:
    """The poller only catches ``TaskValidationError``; anything else would abort a poll cycle."""
    reject(f"{MINIMAL_YAML}\n{content}")


def test_random_yaml_never_raises_anything_but_a_validation_error() -> None:
    rng = random.Random(20261024)
    fragments = [
        "version: 1",
        "objective: Add a short note to docs/guide.md please",
        "scope: code",
        "notes: text",
        "checks: [a, b]",
        "budget_usd: 1.5",
        "- item",
        "{",
        "}",
        "[",
        "]",
        "&a",
        "*a",
        "!!int",
        "!!str",
        "!!float",
        "!!bool",
        "!!timestamp",
        "!!binary",
        "!!set",
        "2020-13-45",
        "2001-12-14 25:00:00",
        ": ",
        "? ",
        "---",
        "...",
        "|",
        ">",
        "'",
        '"',
        "\\x",
        "0x",
        "1e999",
        ".inf",
        ".nan",
        "~",
        "null",
        "true",
        "<<:",
        "  ",
        "\t",
        "\x0b",
        "\u2028",
        "\u00e9",
        "\U0001f600",
    ]
    for _ in range(600):
        lines = [
            " ".join(rng.choice(fragments) for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(1, 6))
        ]
        text = "\n".join(lines)
        try:
            result = parse_task(TITLE, issue_body(text))
        except TaskValidationError as exc:
            assert exc.errors
        except Exception as exc:  # noqa: BLE001 - the point is to catch every other type
            pytest.fail(f"{type(exc).__name__}: {exc} for input {text!r}")
        else:
            assert isinstance(result, ParsedTask)


# --------------------------------------------------------------------------- field rules
def field_errors(**fields: Any) -> list[str]:
    return reject(task_yaml(**fields))


def has(errors: list[str], *fragments: str) -> None:
    joined = " | ".join(errors)
    for fragment in fragments:
        assert fragment in joined, (fragment, errors)


def test_extra_keys_are_forbidden() -> None:
    has(field_errors(bogus=1), "bogus", "Extra inputs are not permitted")
    has(field_errors(allowed_commands=["rm -rf /"]), "allowed_commands")


@pytest.mark.parametrize("version", [0, 2, 99, "1", "one", None, [1], 1.5])
def test_version_must_be_one(version: Any) -> None:
    has(field_errors(version=version), "version")


def test_version_is_required() -> None:
    has(field_errors(version=...), "version", "Field required")


def test_objective_is_required_and_length_bounded() -> None:
    has(field_errors(objective=...), "objective", "Field required")
    has(field_errors(objective="x" * 9), "objective", "at least 10")
    has(field_errors(objective="   short   "), "objective", "at least 10")
    has(field_errors(objective="x" * 8001), "objective", "at most 8000")
    has(field_errors(objective=12345678901234), "objective", "valid string")
    has(field_errors(objective=["a", "b"]), "objective", "valid string")
    assert parse(task_yaml(objective="x" * 10)).spec.objective == "x" * 10
    assert parse(task_yaml(objective="y" * 8000)).spec.objective == "y" * 8000
    assert parse(task_yaml(objective="   0123456789   ")).spec.objective == "0123456789"


@pytest.mark.parametrize("scope", ["docs", "CODE", "Docs-Only", "", "everything", 5, None])
def test_scope_is_an_enum(scope: Any) -> None:
    has(field_errors(scope=scope), "scope")


@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_valid_scopes(scope: str) -> None:
    assert parse(task_yaml(scope=scope)).spec.scope == scope


@pytest.mark.parametrize(
    "agent", ["Omp", "1omp", "omp!", "a" * 33, "", "-omp", "om p", "OMP", "o.mp", "omp x"]
)
def test_agent_name_must_match_the_pattern(agent: str) -> None:
    has(field_errors(agent=agent), "agent")


@pytest.mark.parametrize("agent", ["omp", "claude", "a", "a" * 32, "my_agent-2", None])
def test_valid_agent_names(agent: str | None) -> None:
    assert parse(task_yaml(agent=agent)).spec.agent == agent


@pytest.mark.parametrize(
    "path",
    [
        "docs/\u00fc.md",
        "docs/\u4e2d.md",
        "docs/../x",
        "..",
        "a/..",
        "/etc/passwd",
        "~/x",
        "docs\\x",
        "docs/{a,b}",
        "docs/a b",
        "docs/a;b",
        "",
        "docs/[ab]",
        "$HOME/x",
    ],
)
def test_allowed_paths_reject_unsafe_globs(path: str) -> None:
    has(field_errors(allowed_paths=[path]), "allowed_paths")


def test_allowed_paths_limits_and_normalisation() -> None:
    has(field_errors(allowed_paths=[f"docs/{i}.md" for i in range(31)]), "at most 30")
    spec = parse(task_yaml(allowed_paths=["./docs/**", "*.md", "docs/a?.md"])).spec
    assert spec.allowed_paths == ["docs/**", "*.md", "docs/a?.md"]
    assert (
        len(parse(task_yaml(allowed_paths=[f"d/{i}.md" for i in range(30)])).spec.allowed_paths)
        == 30
    )


@pytest.mark.parametrize(
    "path",
    [
        "docs/*.md",
        "docs/?.md",
        "/etc/passwd",
        "..",
        "../x",
        ".",
        "docs/\u00fc.md",
        "",
        "a\\b",
        "~/x",
    ],
)
def test_context_files_must_be_plain_repository_paths(path: str) -> None:
    has(field_errors(context_files=[path]), "context_files")


def test_context_files_limits_and_normalisation() -> None:
    has(field_errors(context_files=[f"docs/{i}.md" for i in range(21)]), "at most 20")
    spec = parse(task_yaml(context_files=["./docs/guide.md", "docs//api.md", "README.md"])).spec
    assert spec.context_files == ["docs/guide.md", "docs/api.md", "README.md"]


def test_acceptance_criteria_bounds() -> None:
    has(field_errors(acceptance_criteria=[""]), "acceptance_criteria")
    has(field_errors(acceptance_criteria=["   "]), "acceptance_criteria")
    has(field_errors(acceptance_criteria=["x" * 501]), "acceptance_criteria", "1-500")
    has(field_errors(acceptance_criteria=["ok"] * 31), "at most 30")
    assert parse(task_yaml(acceptance_criteria=["x" * 500])).spec.acceptance_criteria == ["x" * 500]
    assert parse(task_yaml(acceptance_criteria=["  a  "])).spec.acceptance_criteria == ["a"]


@pytest.mark.parametrize(
    "name", ["-lint", ".lint", "a b", "a;b", "", "x" * 42, "a/b", "lint!", "$(id)"]
)
def test_check_names_must_match_the_pattern(name: str) -> None:
    has(field_errors(checks=[name]), "checks")


def test_check_names_valid_unique_and_limited() -> None:
    assert parse(task_yaml(checks=["x" * 41, "unit.tests-1_a"])).spec.checks == [
        "x" * 41,
        "unit.tests-1_a",
    ]
    has(field_errors(checks=["a", "b", "a"]), "must not repeat")
    has(field_errors(checks=[f"c{i}" for i in range(11)]), "at most 10")


@pytest.mark.parametrize("budget", [0, -1, -0.01, 1000.01, "abc", ".inf", ".nan", [1], {"a": 1}])
def test_budget_must_be_positive_and_bounded(budget: Any) -> None:
    has(field_errors(budget_usd=budget), "budget_usd")


@pytest.mark.parametrize("budget", [0.01, 1, 1000, "5"])
def test_valid_budgets(budget: Any) -> None:
    assert parse(task_yaml(budget_usd=budget)).spec.budget_usd == float(budget)


@pytest.mark.parametrize("minutes", [0, -5, 1440.01, "soon", ".inf", ".nan"])
def test_timeout_must_be_positive_and_bounded(minutes: Any) -> None:
    has(field_errors(timeout_minutes=minutes), "timeout_minutes")


@pytest.mark.parametrize("minutes", [0.5, 20, 1440])
def test_valid_timeouts(minutes: float) -> None:
    assert parse(task_yaml(timeout_minutes=minutes)).spec.timeout_minutes == minutes


def test_notes_are_length_bounded() -> None:
    has(field_errors(notes="x" * 4001), "notes", "at most 4000")
    assert parse(task_yaml(notes="x" * 4000)).spec.notes == "x" * 4000
    assert parse(task_yaml(notes=None)).spec.notes is None


def test_task_spec_can_be_built_directly_and_stays_strict() -> None:
    spec = TaskSpec(version=1, objective="x" * 10)
    assert spec.scope == "docs-only"
    with pytest.raises(pydantic.ValidationError):
        TaskSpec(version=1, objective="x" * 10, surprise=True)  # type: ignore[call-arg]


def test_validation_messages_name_the_field() -> None:
    errors = reject(task_yaml(scope="nope", budget_usd=-1, objective="short"))
    assert any(e.startswith("scope: ") for e in errors)
    assert any(e.startswith("budget_usd: ") for e in errors)
    assert any(e.startswith("objective: ") for e in errors)


# --------------------------------------------------------------------------- hygiene
HIDDEN_COMMENTS = [
    "<!-- hidden instruction -->\n{block}",
    "{block}\n<!-- after the block -->",
    "text <!-- inline --> text\n{block}",
    "<!--\nmulti\nline\n-->\n{block}",
    "`<!-- in code span -->`\n{block}",
    "> ```\n> <!-- quoted fence is not a fence -->\n> ```\n{block}",
    "```text\nclosed\n```\n<!-- after a closed fence -->\n{block}",
]


@pytest.mark.parametrize("template", HIDDEN_COMMENTS)
def test_html_comments_outside_a_fence_are_rejected(template: str) -> None:
    body = template.replace("{block}", fenced_block(MINIMAL_YAML))
    errors = problems(TITLE, body)
    assert any("HTML comment" in e for e in errors), errors


def test_html_comments_inside_fences_are_allowed() -> None:
    other = "```text\n<!-- visible to the reviewer inside a fence -->\n```\n"
    assert parse_task(TITLE, other + fenced_block(MINIMAL_YAML)).spec.objective == OBJECTIVE
    inside = task_yaml(notes="mentions <!-- a comment --> literally")
    assert parse(inside).spec.notes == "mentions <!-- a comment --> literally"
    tilde = "~~~text\n<!-- tilde fenced -->\n~~~\n" + fenced_block(MINIMAL_YAML)
    assert parse_task(TITLE, tilde).spec.objective == OBJECTIVE


INVISIBLE = [
    ("right-to-left-override", "\u202e"),
    ("left-to-right-override", "\u202d"),
    ("right-to-left-embedding", "\u202b"),
    ("pop-directional", "\u202c"),
    ("left-to-right-mark", "\u200e"),
    ("right-to-left-mark", "\u200f"),
    ("bidi-isolate", "\u2066"),
    ("pop-isolate", "\u2069"),
    ("zero-width-space", "\u200b"),
    ("zero-width-non-joiner", "\u200c"),
    ("word-joiner", "\u2060"),
    ("soft-hyphen", "\u00ad"),
    ("byte-order-mark", "\ufeff"),
    ("tag-latin-a", "\U000e0041"),
    ("tag-cancel", "\U000e007f"),
    ("nul", "\x00"),
    ("escape", "\x1b"),
    ("backspace", "\x08"),
    ("delete", "\x7f"),
    ("next-line", "\u0085"),
    ("vertical-tab", "\x0b"),
    ("private-use", "\ue000"),
    ("lone-surrogate", "\ud800"),
]


@pytest.mark.parametrize("char", [c for _, c in INVISIBLE], ids=[n for n, _ in INVISIBLE])
def test_invisible_and_control_characters_are_rejected_in_the_body(char: str) -> None:
    body = issue_body(f"{MINIMAL_YAML}\nnotes: abc{char}def")  # raw character, not a YAML escape
    errors = problems(TITLE, body)
    code = f"U+{ord(char):04X}"
    assert any("invisible or control" in e and code in e and e.startswith("body") for e in errors)


@pytest.mark.parametrize("char", ["\u202e", "\u200b", "\x1b", "\U000e0041"])
def test_invisible_characters_are_rejected_in_the_title(char: str) -> None:
    errors = problems(f"Fix{char}it", issue_body())
    assert any(e.startswith("title contains invisible or control characters") for e in errors)


def test_legitimate_unicode_is_allowed() -> None:
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"  # emoji joined by zero-width joiners
    notes = (
        f"family {family} heart \u2764\ufe0f caf\u00e9 \u4e2d\u6587 \u0645\u0631\u062d\u0628\u0627"
    )
    task = parse(task_yaml(notes=notes), title=f"Title {family}")
    assert task.spec.notes == notes
    assert family in task.title


def test_tabs_carriage_returns_and_newlines_are_not_invisible() -> None:
    assert hygiene_errors("title", "line\twith\ttabs\r\nand\rcr\nand lf") == []


def test_hygiene_reports_title_and_body_separately_and_caps_the_offset_list() -> None:
    errors = hygiene_errors("a\u202eb", "c\u200bd" + "\u200b" * 20)
    assert len(errors) == 2
    assert errors[0].startswith("title contains") and "U+202E@1" in errors[0]
    assert errors[1].startswith("body contains") and "U+200B@1" in errors[1]
    assert errors[1].count("U+200B") == 5  # only the first five are listed


def test_title_and_body_size_limits() -> None:
    assert hygiene_errors("t" * 256, "b") == []
    assert any("title is longer than 256" in e for e in hygiene_errors("t" * 257, "b"))
    assert hygiene_errors("t", "b" * 60_000) == []
    assert any("body is longer than 60000" in e for e in hygiene_errors("t", "b" * 60_001))


def test_oversize_issues_are_rejected_by_parse_task() -> None:
    assert any("title is longer" in e for e in problems("t" * 257, issue_body()))
    filler = "x" * (60_001 - len(issue_body()))
    assert any("body is longer" in e for e in problems(TITLE, issue_body() + filler))
    exact = issue_body()
    padded = exact + "p" * (60_000 - len(exact))
    assert len(padded) == 60_000
    assert parse_task(TITLE, padded).spec.objective == OBJECTIVE


def test_hygiene_runs_before_yaml_parsing_and_reports_all_problems() -> None:
    body = "<!-- x -->\n" + issue_body(task_yaml(notes="a\u202eb"))
    errors = problems("t" * 300, body)
    assert len(errors) == 3
    has(errors, "title is longer", "invisible or control", "HTML comment")


# --------------------------------------------------------------------------- revisions & hashes
def test_normalize_text_handles_line_endings_and_surrounding_whitespace() -> None:
    assert normalize_text("  a\r\nb\rc\n\n ") == "a\nb\nc"
    assert normalize_text("") == ""
    assert normalize_text(" \n\t ") == ""


def test_revision_has_a_stable_known_value() -> None:
    """Revisions are persisted in the ledger; the hashing scheme must not drift silently."""
    assert (
        compute_revision("Title", "Body")
        == "da2b36b51e9d73c93dd952628f7b2f579c247afe21b7c0fc3153b9d82306132b"
    )


def test_revision_ignores_line_endings_and_surrounding_whitespace() -> None:
    base = compute_revision("Title", "line one\nline two")
    assert compute_revision("Title", "line one\r\nline two") == base
    assert compute_revision("Title", "line one\rline two") == base
    assert (
        compute_revision("  Title \n", "\n\n line one\nline two\n\n ".replace(" line", "line"))
        == base
    )
    assert compute_revision("Title\r\n", "line one\r\nline two\r\n\r\n") == base


@pytest.mark.parametrize(
    ("title", "body"),
    [
        ("Title!", "line one\nline two"),
        ("title", "line one\nline two"),
        ("Title", "line one\nline two!"),
        ("Title", "Line one\nline two"),
        ("Title", "line one\n\nline two"),
        ("Title", "line one\nline  two"),
        ("Title", "line one \nline two"),
        ("Title", "line one\n line two"),
        ("Title", "line one\tline two"),
        ("Title", "line two\nline one"),
        ("Title", "line one\nline two\u200b"),
        ("Title line one", "line two"),
        ("", "Title\nline one\nline two"),
        ("Title", ""),
        ("Cafe\u0301", "line one\nline two"),
    ],
)
def test_revision_changes_for_every_other_edit(title: str, body: str) -> None:
    assert compute_revision(title, body) != compute_revision("Title", "line one\nline two")


def test_revision_is_not_confused_by_moving_text_between_title_and_body() -> None:
    assert compute_revision("ab", "") != compute_revision("a", "b")
    assert compute_revision("a", "b") != compute_revision("b", "a")
    assert compute_revision("a\nb", "") != compute_revision("a", "b")


def test_revision_of_parsed_task_covers_title_and_prose() -> None:
    base = parse_task(TITLE, issue_body())
    assert parse_task(TITLE, issue_body()).revision == base.revision
    assert parse_task(TITLE, issue_body().replace("\n", "\r\n") + "\n\n").revision == base.revision
    assert parse_task("Another title", issue_body()).revision != base.revision
    assert parse_task(TITLE, issue_body(prose="Different prose.")).revision != base.revision
    assert parse_task(TITLE, issue_body(task_yaml(notes="x"))).revision != base.revision


def test_spec_hash_ignores_prose_title_formatting_and_key_order() -> None:
    base = parse_task(TITLE, issue_body())
    reordered = f"objective: {OBJECTIVE}\nversion: 1"
    commented = f"# leading comment\nversion: 1  # trailing\nobjective: '{OBJECTIVE}'\n"
    explicit = task_yaml(scope="docs-only", allowed_paths=[], checks=[], notes=None)
    variants = [
        parse_task("A completely different title", issue_body()),
        parse_task(TITLE, issue_body(prose="Other prose\n\n<not a comment>")),
        parse_task(TITLE, f"before\n\n```yaml agent-task\n{reordered}\n```\nafter"),
        parse_task(TITLE, issue_body(commented)),
        parse_task(TITLE, issue_body(explicit)),
        parse_task(TITLE, issue_body(info="agent-task")),
        parse_task(TITLE, issue_body(fence="~~~")),
    ]
    assert {v.spec_hash for v in variants} == {base.spec_hash}
    assert len({v.revision for v in variants}) > 1  # the revision, unlike the hash, differs


@pytest.mark.parametrize(
    "change",
    [
        {"objective": OBJECTIVE + " Also fix typos."},
        {"scope": "code"},
        {"agent": "claude"},
        {"allowed_paths": ["docs/**"]},
        {"context_files": ["docs/guide.md"]},
        {"acceptance_criteria": ["done"]},
        {"checks": ["lint"]},
        {"budget_usd": 1.0},
        {"timeout_minutes": 5},
        {"notes": "n"},
    ],
    ids=lambda change: next(iter(change)),
)
def test_spec_hash_changes_with_every_field(change: dict[str, Any]) -> None:
    assert parse(task_yaml(**change)).spec_hash != parse().spec_hash


def test_spec_hash_depends_on_list_order() -> None:
    first = parse(task_yaml(checks=["a", "b"])).spec_hash
    assert parse(task_yaml(checks=["b", "a"])).spec_hash != first


# --------------------------------------------------------------------------- against config
def spec_for(**fields: Any) -> TaskSpec:
    return TaskSpec.model_validate({"version": 1, "objective": OBJECTIVE, **fields})


def test_a_conforming_task_has_no_problems() -> None:
    cfg = make_config(checks={"lint": {"argv": ["ruff"]}})
    spec = spec_for(checks=["lint"], budget_usd=cfg.limits.max_budget_usd)
    assert validate_against_config(spec, cfg) == []
    assert validate_against_config(spec_for(), cfg) == []


def test_scope_must_be_enabled() -> None:
    problems_ = validate_against_config(spec_for(scope="code"), make_config())
    assert len(problems_) == 1 and "scope 'code' is not enabled" in problems_[0]
    assert "docs-only" in problems_[0]
    both = make_config(scopes={"allowed": ["docs-only", "code"]})
    assert validate_against_config(spec_for(scope="code"), both) == []


def test_agent_must_be_allowed() -> None:
    cfg = make_config(agent={"default": "omp", "allowed": ["omp"]})
    problems_ = validate_against_config(spec_for(agent="claude"), cfg)
    assert len(problems_) == 1 and "agent 'claude' is not enabled" in problems_[0]
    assert validate_against_config(spec_for(agent="omp"), cfg) == []
    assert validate_against_config(spec_for(), cfg) == []  # falls back to the default backend


def test_unknown_checks_are_reported_by_name() -> None:
    cfg = make_config(checks={"lint": {"argv": ["ruff"]}})
    problems_ = validate_against_config(spec_for(checks=["lint", "deploy", "rm"]), cfg)
    assert [p for p in problems_ if "'deploy'" in p] and [p for p in problems_ if "'rm'" in p]
    assert not [p for p in problems_ if "'lint'" in p]
    assert len(problems_) == 2


def test_budget_and_timeout_may_not_exceed_the_configured_maxima() -> None:
    cfg = make_config(limits={"max_budget_usd": 5, "max_timeout_minutes": 60})
    assert validate_against_config(spec_for(budget_usd=5, timeout_minutes=60), cfg) == []
    over = validate_against_config(spec_for(budget_usd=5.01, timeout_minutes=61), cfg)
    assert len(over) == 2
    assert "budget_usd 5.01 exceeds the configured maximum 5" in over[0]
    assert "timeout_minutes 61" in over[1] and "60" in over[1]


def test_all_problems_are_reported_together() -> None:
    cfg = make_config(agent={"default": "omp", "allowed": ["omp"]})
    spec = spec_for(
        scope="code", agent="claude", checks=["nope"], budget_usd=999, timeout_minutes=999
    )
    assert len(validate_against_config(spec, cfg)) == 5
