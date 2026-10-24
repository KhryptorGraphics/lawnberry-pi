"""Task schema: strict parsing of the fenced ``agent-task`` YAML block in an issue body.

Only the single validated block (plus the issue title) ever reaches an agent.  Prose, comments
and everything else in the issue are ignored.  Content that GitHub hides from a reviewer
(HTML comments, invisible Unicode) is rejected outright so the approver sees what the agent gets.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import BridgeConfig
from .errors import TaskValidationError
from .globs import PathError, normalize_glob, normalize_repo_path

__all__ = [
    "SCHEMA_VERSION",
    "TASK_FENCE_TOKEN",
    "ParsedTask",
    "TaskSpec",
    "compute_revision",
    "find_task_blocks",
    "hygiene_errors",
    "normalize_text",
    "parse_task",
    "strict_yaml_load",
    "validate_against_config",
]

SCHEMA_VERSION = 1
TASK_FENCE_TOKEN = "agent-task"
MAX_BLOCK_CHARS = 20_000
MAX_BODY_CHARS = 60_000
MAX_TITLE_CHARS = 256

_FENCE_OPEN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")
_ASCII_PATH = re.compile(r"^[A-Za-z0-9._\-/*?]+$")
_ALLOWED_CONTROL = {"\n", "\r", "\t"}
_ALLOWED_FORMAT = {"\u200d"}  # zero-width joiner is part of legitimate emoji sequences


def normalize_text(text: str) -> str:
    """Normalise line endings and surrounding whitespace for revision hashing."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def compute_revision(title: str, body: str) -> str:
    """Hash of everything an editor could change in an issue's task text."""
    payload = json.dumps(
        {"title": normalize_text(title), "body": normalize_text(body)},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fence_blocks(body: str) -> list[tuple[str, str]]:
    """Return ``(info_string, content)`` for every fenced block, honouring CommonMark rules."""
    blocks: list[tuple[str, str]] = []
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        match = _FENCE_OPEN.match(lines[i])
        if not match:
            i += 1
            continue
        fence = match.group("fence")
        info = match.group("info").strip()
        if fence[0] == "`" and "`" in info:
            i += 1
            continue  # an inline code span, not a fence
        closing = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")
        j = i + 1
        content: list[str] = []
        closed = False
        while j < len(lines):
            if closing.match(lines[j]):
                closed = True
                break
            content.append(lines[j])
            j += 1
        if closed:
            blocks.append((info, "\n".join(content)))
        i = j + 1 if closed else len(lines)
    return blocks


def _is_task_info(info: str) -> bool:
    tokens = info.split()
    if not tokens:
        return False
    if tokens == [TASK_FENCE_TOKEN]:
        return True
    return (
        len(tokens) == 2 and tokens[0].lower() in {"yaml", "yml"} and tokens[1] == TASK_FENCE_TOKEN
    )


def find_task_blocks(body: str) -> list[str]:
    """Raw YAML text of every fenced block tagged ``agent-task``."""
    return [content for info, content in _fence_blocks(body) if _is_task_info(info)]


def hygiene_errors(title: str, body: str) -> list[str]:
    """Reasons an issue cannot be trusted to show a reviewer what an agent would receive."""
    errors: list[str] = []
    if len(title) > MAX_TITLE_CHARS:
        errors.append(f"title is longer than {MAX_TITLE_CHARS} characters")
    if len(body) > MAX_BODY_CHARS:
        errors.append(f"body is longer than {MAX_BODY_CHARS} characters")
    for label, text in (("title", title), ("body", body)):
        bad: list[str] = []
        for index, ch in enumerate(text):
            cat = unicodedata.category(ch)
            if cat == "Cc" and ch not in _ALLOWED_CONTROL:
                bad.append(f"U+{ord(ch):04X}@{index}")
            elif cat == "Cf" and ch not in _ALLOWED_FORMAT:
                bad.append(f"U+{ord(ch):04X}@{index}")
            elif cat in {"Co", "Cs"}:
                bad.append(f"U+{ord(ch):04X}@{index}")
            if len(bad) >= 5:
                break
        if bad:
            errors.append(
                f"{label} contains invisible or control characters ({', '.join(bad)}); "
                "remove them so a reviewer can see the whole task"
            )
    if _has_hidden_html_comment(body):
        errors.append(
            "body contains an HTML comment outside a code fence; GitHub hides it from reviewers, "
            "so it is not allowed"
        )
    return errors


def _has_hidden_html_comment(body: str) -> bool:
    in_fence: str | None = None
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _FENCE_OPEN.match(line)
        if in_fence is None:
            if match and not (match.group("fence")[0] == "`" and "`" in match.group("info")):
                in_fence = match.group("fence")
                continue
            if "<!--" in line:
                return True
        else:
            if re.match(rf"^ {{0,3}}{re.escape(in_fence[0])}{{{len(in_fence)},}}\s*$", line):
                in_fence = None
    return False


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    None, None, "mapping keys must be strings", key_node.start_mark
                )
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def strict_yaml_load(text: str) -> Any:
    """Parse one YAML document without aliases, anchors, custom tags or duplicate keys."""
    if len(text) > MAX_BLOCK_CHARS:
        raise TaskValidationError([f"task block is longer than {MAX_BLOCK_CHARS} characters"])
    try:
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                raise TaskValidationError(["YAML anchors and aliases are not allowed"])
        return yaml.load(text, Loader=_StrictLoader)
    except TaskValidationError:
        raise
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or "invalid YAML"
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise TaskValidationError([f"task block is not valid YAML: {problem}{where}"]) from exc
    except RecursionError as exc:
        raise TaskValidationError(["task block is nested too deeply"]) from exc
    except Exception as exc:  # noqa: BLE001 - PyYAML constructors raise an open-ended set
        # (timestamps, !!int, !!bool, huge integers, !!set, ...: ValueError, KeyError,
        # IndexError, AttributeError, ...) instead of YAMLError for malformed scalars.
        detail = str(exc).replace("\n", " ")[:80]
        raise TaskValidationError(
            [f"task block is not valid YAML: invalid value ({type(exc).__name__}: {detail})"]
        ) from exc


class TaskSpec(BaseModel):
    """The machine-readable task an approver signs off on."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    version: Literal[1]
    objective: str = Field(min_length=10, max_length=8000)
    scope: Literal["docs-only", "code"] = "docs-only"
    agent: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_\-]{0,31}$")
    allowed_paths: list[str] = Field(default_factory=list, max_length=30)
    context_files: list[str] = Field(default_factory=list, max_length=20)
    acceptance_criteria: list[str] = Field(default_factory=list, max_length=30)
    checks: list[str] = Field(default_factory=list, max_length=10)
    budget_usd: float | None = Field(default=None, gt=0, le=1000)
    timeout_minutes: float | None = Field(default=None, gt=0, le=24 * 60)
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("allowed_paths")
    @classmethod
    def _globs(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            if not _ASCII_PATH.match(item.strip()):
                raise ValueError("allowed_paths may only use ASCII letters, digits and . _ - / * ?")
            try:
                out.append(normalize_glob(item))
            except PathError as exc:
                raise ValueError(str(exc)) from exc
        return out

    @field_validator("context_files")
    @classmethod
    def _paths(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            if not _ASCII_PATH.match(item.strip()) or "*" in item or "?" in item:
                raise ValueError("context_files must be plain ASCII repository paths")
            try:
                out.append(normalize_repo_path(item))
            except PathError as exc:
                raise ValueError(str(exc)) from exc
        return out

    @field_validator("acceptance_criteria")
    @classmethod
    def _criteria(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > 500 for item in cleaned):
            raise ValueError("each acceptance criterion must be 1-500 characters")
        return cleaned

    @field_validator("checks")
    @classmethod
    def _checks(cls, value: list[str]) -> list[str]:
        for name in value:
            if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,40}$", name):
                raise ValueError(f"invalid check name {name!r}")
        if len(set(value)) != len(value):
            raise ValueError("checks must not repeat")
        return value


@dataclass(frozen=True)
class ParsedTask:
    title: str
    spec: TaskSpec
    revision: str
    spec_hash: str


def _format_validation(exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "task"
        messages.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return messages


def parse_task(title: str, body: str) -> ParsedTask:
    """Validate an issue and return the task, or raise :class:`TaskValidationError`."""
    problems = hygiene_errors(title, body)
    if problems:
        raise TaskValidationError(problems)
    blocks = find_task_blocks(body)
    if len(blocks) != 1:
        raise TaskValidationError(
            [
                f"expected exactly one fenced ```yaml {TASK_FENCE_TOKEN} block, "
                f"found {len(blocks)}",
            ]
        )
    data = strict_yaml_load(blocks[0])
    if not isinstance(data, dict):
        raise TaskValidationError(["task block must be a YAML mapping"])
    try:
        spec = TaskSpec.model_validate(data)
    except ValidationError as exc:
        raise TaskValidationError(_format_validation(exc)) from exc
    canonical = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return ParsedTask(
        title=normalize_text(title),
        spec=spec,
        revision=compute_revision(title, body),
        spec_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def validate_against_config(spec: TaskSpec, cfg: BridgeConfig) -> list[str]:
    """Task requests the configuration does not permit (empty list = acceptable)."""
    problems: list[str] = []
    if spec.scope not in cfg.scopes.allowed:
        problems.append(
            f"scope {spec.scope!r} is not enabled (allowed: {', '.join(cfg.scopes.allowed)})"
        )
    backend = spec.agent or cfg.agent.default
    if backend not in cfg.agent.allowed:
        problems.append(
            f"agent {backend!r} is not enabled (allowed: {', '.join(cfg.agent.allowed)})"
        )
    for name in spec.checks:
        if name not in cfg.checks:
            problems.append(f"check {name!r} is not defined in the bridge configuration")
    if spec.budget_usd is not None and spec.budget_usd > cfg.limits.max_budget_usd:
        problems.append(
            f"budget_usd {spec.budget_usd} exceeds the configured maximum "
            f"{cfg.limits.max_budget_usd}"
        )
    if spec.timeout_minutes is not None and spec.timeout_minutes > cfg.limits.max_timeout_minutes:
        problems.append(
            f"timeout_minutes {spec.timeout_minutes} exceeds the configured maximum "
            f"{cfg.limits.max_timeout_minutes}"
        )
    return problems
