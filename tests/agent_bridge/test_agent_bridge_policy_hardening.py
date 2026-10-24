"""Hardening of the change policy, its node twin and the report parser.

Each section pins one finding of the PR 17 containment review (``C-04`` .. ``C-09``).  The
policy tests run ``evaluate_changes`` on hand-made :class:`Change` rows and, where the file
system matters, against real git; the guard tests drive the real ``omp_guard.mjs`` through node.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import subprocess
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.rig import BASE_FILES, GitRepoEnv, make_git_env
from pydantic import ValidationError

from tools.agent_bridge import globs
from tools.agent_bridge.agents.base import AgentReport, extract_report
from tools.agent_bridge.agents.omp import GUARD_PATH
from tools.agent_bridge.config import DEFAULT_PROTECTED_PATHS, BridgeConfig, GitConfig
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.globs import PathError, match_any, normalize_repo_path
from tools.agent_bridge.models import RunStatus
from tools.agent_bridge.policy import Change, WritePolicy, build_write_policy, evaluate_changes
from tools.agent_bridge.schema import TaskSpec

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

REGULAR, EXECUTABLE, ABSENT = "100644", "100755", "000000"
OBJECTIVE = "Update the documentation of the agent bridge, please."

NODE_DRIVER = """
import * as fs from "node:fs";
import { pathToFileURL } from "node:url";

const guard = await import(pathToFileURL(process.argv[2]).href);
const request = JSON.parse(fs.readFileSync(0, "utf8"));
let out;
if (request.op === "decide") {
  out = request.items.map((item) => guard.decide(item.policy, item.event, undefined));
} else if (request.op === "fold") {
  out = request.values.map((value) => guard.foldCase(value));
} else if (request.op === "match") {
  out = request.pairs.map(([pattern, path]) => guard.matchAnyFolded(path, [pattern]));
} else {
  throw new Error(`unknown op ${request.op}`);
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("hardening-driver") / "driver.mjs"
    path.write_text(NODE_DRIVER, encoding="utf-8")
    return path


def run_node(driver: Path, request: dict[str, Any]) -> Any:
    assert NODE is not None
    done = subprocess.run(
        [NODE, str(driver), str(GUARD_PATH)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def make_config(**overrides: Any) -> BridgeConfig:
    data = {
        "repository": "owner/repo",
        "approvers": ["alice"],
        "scopes": {"allowed": ["docs-only", "code"]},
        **overrides,
    }
    return BridgeConfig.model_validate(data)


def make_policy(scope: str, **config: Any) -> WritePolicy:
    spec = TaskSpec.model_validate({"version": 1, "objective": OBJECTIVE, "scope": scope})
    return build_write_policy(spec, make_config(**config))


def added(path: str, **kwargs: Any) -> Change:
    base: dict[str, Any] = {
        "status": "A",
        "old_mode": ABSENT,
        "new_mode": REGULAR,
        "size": 10,
        "added": 1,
    }
    return Change(path=path, **{**base, **kwargs})


LIMITS = make_config().limits


def guard_policy(root: Path, policy: WritePolicy, **extra: Any) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "root": str(root),
        "tools": ["read", "grep", "glob", "find", "edit", "write"],
        "read_deny": [],
        "bash_exact": [],
        **policy.describe(),
        **extra,
    }


def guard_decisions(
    driver: Path, policy: dict[str, Any], tool: str, paths: list[str]
) -> list[dict[str, Any]]:
    items = [
        {
            "policy": policy,
            "event": {"toolName": tool, "input": {"path": path, "content": "x"}},
        }
        for path in paths
    ]
    return run_node(driver, {"op": "decide", "items": items})


# =========================================================================== C-04 / C-05
# Agent-policy files are read hierarchically (omp, Claude Code, Gemini, Cursor, ...) and by git
# itself (.gitattributes can hide a diff in the review UI, .gitignore can hide a later file from
# ``git add -A``), so they are protected at every depth, in every case.  Names from the review:
REPORTED_NAMES = [
    "docs/AGENTS.md",
    "docs/CLAUDE.md",
    "CLAUDE.local.md",
    "claude.md",
    "agents.md",
    "Agents.md",
    "GEMINI.md",
    "docs/.claude/x.md",
    "docs/.omp/x.md",
    "docs/.github/workflows/x.yml",
    "docs/.git/config",
    "docs/Constitution.md",
]
# ... plus the other names the protected globs now cover, and their case variants.
EXTRA_NAMES = [
    "Claude.MD",
    "docs/.CLAUDE/x.md",
    "docs/CLAUDE.local.md",
    "docs/Claude.Local.MD",
    "docs/.gitattributes",
    "docs/.GITATTRIBUTES",
    "docs/.gitignore",
    "docs/.gitmodules",
    "docs/.cursor/rules/x.md",
    "docs/.Cursor/rules/x.md",
    "docs/.cursorrules",
    "docs/.agents/skills/x.md",
    "docs/.mcp.json",
    "docs/.OMP/x.md",
    "docs/GEMINI.md",
    "docs/gemini.md",
    "docs/deep/er/AGENTS.md",
    "docs/Constitution.MD",
]
CODE_ONLY_NAMES = [
    "tools/AGENTS.md",
    "tests/CLAUDE.md",
    "tests/.claude/settings.json",
    "tools/.omp/x.md",
    "tests/AGENTS.md",
    "tools/.cursorrules",
    "tests/.mcp.json",
    "tools/.gitattributes",
    "tests/.gitignore",
    "tools/helpers/.GITMODULES",
]
PROTECTED_NAMES = [*REPORTED_NAMES, *EXTRA_NAMES, *CODE_ONLY_NAMES]

# Ordinary files whose names merely resemble the protected ones.
ORDINARY_NAMES = [
    "docs/agents-guide.md",
    "docs/claude-notes.md",
    "docs/gemini-notes.md",
    "docs/agents/overview.md",
    "docs/.claudette/notes.md",
    "docs/.omp-notes.md",
    "docs/gitignore.md",
    "docs/cursorrules.md",
    "docs/github-actions.md",
    "docs/.github-notes.md",
    "docs/git-notes.md",
    "docs/mcp.json.md",
    "docs/constitution-draft.md",
    "README.md",
    "docs/guide.md",
]


@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_the_defaults_protect_agent_policy_files_at_every_depth(scope: str) -> None:
    """Both scopes: the protected list wins over the scope's own allow list."""
    policy = make_policy(scope)

    found = {
        name: [v.code for v in evaluate_changes([added(name)], policy, LIMITS)]
        for name in PROTECTED_NAMES
    }

    assert {name: codes for name, codes in found.items() if codes != ["protected_path"]} == {}


@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_check_path_names_the_protected_path_for_every_reported_file(scope: str) -> None:
    policy = make_policy(scope)
    for name in PROTECTED_NAMES:
        violation = policy.check_path(name)
        assert violation is not None and violation.code == "protected_path", name
        assert violation.path == name


@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_files_that_only_resemble_a_protected_name_stay_writable(scope: str) -> None:
    policy = make_policy(scope)

    for name in ORDINARY_NAMES:
        assert evaluate_changes([added(name)], policy, LIMITS) == [], name


@pytest.mark.parametrize(
    "glob",
    [
        "**/AGENTS.md",
        "**/CLAUDE.md",
        "**/CLAUDE.local.md",
        "**/GEMINI.md",
        "**/.claude/**",
        "**/.omp/**",
        "**/.cursor/**",
        "**/.cursorrules",
        "**/.agents/**",
        "**/.mcp.json",
        "**/.gitattributes",
        "**/.gitmodules",
        "**/.gitignore",
        "**/.github/**",
        "**/.git/**",
    ],
)
def test_the_default_protected_list_names_the_nested_agent_policy_globs(glob: str) -> None:
    assert glob in DEFAULT_PROTECTED_PATHS


def test_the_root_anchored_defaults_are_kept() -> None:
    for glob in (
        "AGENTS.md",
        "CLAUDE.md",
        ".claude/**",
        ".omp/**",
        ".agents/**",
        ".cursor/**",
        ".gitmodules",
        ".gitattributes",
        "docs/constitution.md",
    ):
        assert glob in DEFAULT_PROTECTED_PATHS


@needs_node
@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_the_node_guard_refuses_every_protected_name_too(
    node_driver: Path, tmp_path: Path, scope: str
) -> None:
    policy = guard_policy(tmp_path / "work", make_policy(scope))

    decisions = guard_decisions(node_driver, policy, "write", PROTECTED_NAMES)

    wrong = {
        name: decision.get("reason")
        for name, decision in zip(PROTECTED_NAMES, decisions, strict=True)
        if decision["allow"] or "protected" not in (decision.get("reason") or "")
    }
    assert wrong == {}


@needs_node
@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_the_node_guard_still_allows_ordinary_documentation(
    node_driver: Path, tmp_path: Path, scope: str
) -> None:
    policy = guard_policy(tmp_path / "work", make_policy(scope))

    decisions = guard_decisions(node_driver, policy, "write", ORDINARY_NAMES)

    assert [d["reason"] for d in decisions if not d["allow"]] == []


@needs_node
@pytest.mark.parametrize("tool", ["write", "edit"])
def test_every_write_class_tool_is_subject_to_the_case_insensitive_deny_list(
    node_driver: Path, tmp_path: Path, tool: str
) -> None:
    policy = guard_policy(tmp_path / "work", make_policy("docs-only"))

    decisions = guard_decisions(node_driver, policy, tool, ["docs/AGENTS.md", "Claude.MD"])

    assert [d["allow"] for d in decisions] == [False, False]


@needs_node
def test_the_node_guard_compares_write_deny_globs_in_any_case(
    node_driver: Path, tmp_path: Path
) -> None:
    open_scope = WritePolicy(allow_sets=(("**",),), deny=("backend/src/safety/**", "**/.env"))
    policy = guard_policy(tmp_path / "work", open_scope)
    paths = [
        "backend/src/safety/estop.py",
        "Backend/SRC/Safety/estop.py",
        "BACKEND/src/safety",
        "docs/.ENV",
        "docs/.Env",
        "backend/src/safetyx/a.py",
        "docs/env.md",
    ]

    allowed = [d["allow"] for d in guard_decisions(node_driver, policy, "write", paths)]

    assert allowed == [False, False, False, False, False, True, True]


@needs_node
def test_the_node_guard_compares_read_deny_globs_in_any_case(
    node_driver: Path, tmp_path: Path
) -> None:
    root = tmp_path / "work"
    policy = guard_policy(
        root,
        WritePolicy(allow_sets=(), deny=()),
        read_deny=[".git", ".git/**", "**/.env", "**/*.pem", "config/secrets*.json"],
    )
    names = [
        ".git/config",
        ".GIT/config",
        ".Git",
        ".ENV",
        "sub/.Env",
        "keys/Server.PEM",
        "Config/Secrets.JSON",
        "docs/guide.md",
        "docs/environment.md",
    ]
    items = [
        {"policy": policy, "event": {"toolName": "read", "input": {"path": name}}} for name in names
    ]

    decisions = run_node(node_driver, {"op": "decide", "items": items})

    assert [d["allow"] for d in decisions] == [False] * 7 + [True, True]
    assert all("not readable" in d["reason"] for d in decisions[:7])


def test_allow_sets_stay_case_sensitive_in_python() -> None:
    policy = WritePolicy(allow_sets=(("docs/**", "*.md"),), deny=())

    assert policy.check_path("docs/a.md") is None
    assert policy.check_path("README.md") is None
    for name in ("DOCS/a.md", "Docs/a.md", "README.MD", "readme.Md"):
        violation = policy.check_path(name)
        assert violation is not None and violation.code == "outside_scope", name


@needs_node
def test_allow_sets_stay_case_sensitive_in_the_node_guard(
    node_driver: Path, tmp_path: Path
) -> None:
    policy = guard_policy(
        tmp_path / "work", WritePolicy(allow_sets=(("docs/**", "*.md"),), deny=())
    )

    decisions = guard_decisions(
        node_driver, policy, "write", ["docs/a.md", "README.md", "DOCS/a.md", "README.MD"]
    )

    assert [d["allow"] for d in decisions] == [True, True, False, False]
    assert all("outside the task's allowed scope" in d["reason"] for d in decisions[2:])


# ---------------------------------------------------------------------------- match_any casefold
def test_match_any_is_case_sensitive_unless_asked() -> None:
    assert match_any("docs/CLAUDE.md", ["**/claude.md"]) is False
    assert match_any("docs/CLAUDE.md", ["**/claude.md"], casefold=False) is False
    assert match_any("docs/CLAUDE.md", ["**/claude.md"], casefold=True) is True
    assert match_any("docs/claude.md", ["**/CLAUDE.md"], casefold=True) is True  # glob folds too


def test_casefold_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        match_any("docs/CLAUDE.md", ["**/claude.md"], True)  # type: ignore[misc]


def test_casefold_keeps_the_glob_dialect() -> None:
    assert match_any("DOCS/A/B.MD", ["docs/**"], casefold=True) is True
    assert match_any("docs", ["DOCS/**"], casefold=True) is True
    assert match_any("Docs/sub/a.md", ["docs/*"], casefold=True) is False  # '*' never crosses '/'
    assert match_any("X/.ENV", ["**/.env"], casefold=True) is True
    assert match_any("X/.ENV.local", ["**/.env"], casefold=True) is False
    assert match_any("a.MD", ["?.md"], casefold=True) is True
    assert match_any("docs/a.md", [], casefold=True) is False


# Characters that case-insensitive file systems (NTFS up-case table, APFS, HFS+, ext4 casefold)
# treat as the same letter as an ASCII one, or that decompose, plus ordinary letters.
FOLD_STRINGS = [
    "AGENTS.md",
    "agents.md",
    "AGENT\u017f.md",  # LATIN SMALL LETTER LONG S: upper-cases to S
    ".G\u0131TATTRIBUTES",  # dotless i: upper-cases to I
    ".GITATTRIBUTES",
    "SERVER.\u212aEY",  # KELVIN SIGN: lower-cases to k
    "STRASSE",
    "Stra\u00dfe",
    "STRA\u1e9eE",  # capital sharp s
    "\u0130STANBUL",  # I with dot above
    "caf\u00e9",
    "cafe\u0301",
    "CAF\u00c9",
    "\u212b-unit",  # ANGSTROM SIGN: canonically equivalent to A with ring
    "\u00c5-unit",
    "\u0391\u03a3",  # final sigma context
    "\u03c3\u03c2",
    "\u00b5-meter",  # micro sign
    "\u039c-meter",
    "docs/Sub/\u00c9.md",
    "plain/ascii/path.txt",
    "",
]


def test_fold_case_is_idempotent_and_equates_the_spellings_a_file_system_would() -> None:
    for text in FOLD_STRINGS:
        assert globs.fold_case(globs.fold_case(text)) == globs.fold_case(text), text
    same = [
        ("AGENTS.md", "agents.md"),
        ("AGENTS.md", "AGENT\u017f.md"),
        (".GITATTRIBUTES", ".G\u0131TATTRIBUTES"),
        ("server.key", "SERVER.\u212aEY"),
        ("STRASSE", "Stra\u00dfe"),
        ("STRASSE", "STRA\u1e9eE"),
        ("caf\u00e9", "cafe\u0301"),
        ("caf\u00e9", "CAF\u00c9"),
        ("\u00c5-unit", "\u212b-unit"),
    ]
    for left, right in same:
        assert globs.fold_case(left) == globs.fold_case(right), (left, right)
    assert globs.fold_case("agents.md") != globs.fold_case("agents-md")


def test_fold_case_never_touches_glob_syntax() -> None:
    assert globs.fold_case("**/.Claude/**") == "**/.claude/**"
    assert globs.fold_case("a?*/B") == "a?*/b"
    for char in "*?/":
        assert char in globs.fold_case(f"A{char}B")


def test_casefold_catches_the_unicode_spellings_of_a_protected_name() -> None:
    assert match_any("docs/AGENT\u017f.md", ["**/AGENTS.md"], casefold=True)
    assert match_any("docs/.G\u0131TATTRIBUTES", ["**/.gitattributes"], casefold=True)
    assert match_any("keys/server.\u212aey", ["**/*.key"], casefold=True)
    assert match_any("docs/cafe\u0301/x.md", ["docs/caf\u00e9/**"], casefold=True)
    assert not match_any("docs/AGENT\u017f.md", ["**/AGENTS.md"])  # case-sensitive by default


@needs_node
def test_python_and_node_fold_every_string_identically(node_driver: Path) -> None:
    js = run_node(node_driver, {"op": "fold", "values": FOLD_STRINGS})

    assert js == [globs.fold_case(text) for text in FOLD_STRINGS]


def assigned_bmp_characters() -> list[str]:
    """Every assigned BMP character (Python's Unicode tables are older than node's: skip ``Cn``)."""
    return [
        chr(cp)
        for cp in range(0x10000)
        if not 0xD800 <= cp <= 0xDFFF and unicodedata.category(chr(cp)) != "Cn"
    ]


def test_fold_case_is_idempotent_for_every_assigned_bmp_character() -> None:
    unstable = [
        hex(ord(c))
        for c in assigned_bmp_characters()
        if globs.fold_case(globs.fold_case(c)) != globs.fold_case(c)
    ]
    assert unstable == []


@needs_node
def test_python_and_node_fold_every_assigned_bmp_character_identically(node_driver: Path) -> None:
    characters = assigned_bmp_characters()

    js = run_node(node_driver, {"op": "fold", "values": characters})

    different = [
        (hex(ord(c)), python, node)
        for c, python, node in zip(
            characters, (globs.fold_case(c) for c in characters), js, strict=True
        )
        if python != node
    ]
    assert different == []


CASEFOLD_PATHS = [
    *PROTECTED_NAMES,
    *ORDINARY_NAMES,
    "AGENTS.md",
    ".claude",
    ".Claude/x",
    "docs/AGENT\u017f.md",
    "docs/.G\u0131TATTRIBUTES",
    "keys/server.\u212aey",
    "keys/Server.PEM",
    "a/B/.ENV.local",
    "BACKEND/SRC/SAFETY/estop.py",
    "backend/src/safetyx/a.py",
    "Config/Secrets.Prod.JSON",
    "config/sub/secrets.json",
    "TOOLS/Agent_Bridge/worker.py",
    "tools/agent_bridge_extra/x.py",
    "docs/cafe\u0301/x.md",
    "docs/caf\u00e9/x.md",
    ".git",
    ".GIT/config",
    ".github/workflows/ci.yml",
]
CASEFOLD_PATTERNS = [
    *DEFAULT_PROTECTED_PATHS,
    "docs/caf\u00e9/**",
    "docs/**",
    "*.md",
    "**/*.MD",
    ".git",
    ".git/**",
]


@needs_node
def test_python_and_node_agree_on_every_default_protected_glob_and_path(node_driver: Path) -> None:
    pairs = [[pattern, path] for pattern in CASEFOLD_PATTERNS for path in CASEFOLD_PATHS]

    js = run_node(node_driver, {"op": "match", "pairs": pairs})

    py = [match_any(path, [pattern], casefold=True) for pattern, path in pairs]
    assert [pair for pair, a, b in zip(pairs, py, js, strict=True) if a != b] == []
    assert any(py) and not all(py)


@needs_node
@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_python_policy_and_node_guard_agree_on_every_protected_and_ordinary_name(
    node_driver: Path, tmp_path: Path, scope: str
) -> None:
    policy = make_policy(scope)
    names = [*PROTECTED_NAMES, *ORDINARY_NAMES, "tools/helper.py", "tests/test_x.py", "src/x.py"]

    decisions = guard_decisions(
        node_driver, guard_policy(tmp_path / "work", policy), "write", names
    )

    python_allows = [policy.check_path(name) is None for name in names]
    node_allows = [d["allow"] for d in decisions]
    assert [n for n, a, b in zip(names, python_allows, node_allows, strict=True) if a != b] == []


# =========================================================================== C-06
# Only a *new* executable bit is refused.  Editing a file that is already executable in the base
# keeps mode 100755 -> 100755 and is an ordinary content change.
def mode_change(path: str, old_mode: str, new_mode: str, status: str = "M") -> Change:
    return Change(path, status, old_mode, new_mode, size=10, added=1, deleted=1)


def test_a_content_edit_of_an_already_executable_file_is_allowed() -> None:
    change = mode_change("tools/run.sh", EXECUTABLE, EXECUTABLE)

    assert evaluate_changes([change], make_policy("code"), LIMITS) == []


@pytest.mark.parametrize(
    ("old_mode", "new_mode", "status"),
    [(REGULAR, EXECUTABLE, "M"), (ABSENT, EXECUTABLE, "A")],
    ids=["chmod-plus-x", "new-executable-file"],
)
def test_a_new_executable_bit_is_still_refused(old_mode: str, new_mode: str, status: str) -> None:
    change = mode_change("tools/run.sh", old_mode, new_mode, status)

    found = evaluate_changes([change], make_policy("code"), LIMITS)

    assert [(v.code, v.path) for v in found] == [("executable", "tools/run.sh")]


@pytest.mark.parametrize(
    ("old_mode", "new_mode", "status"),
    [(EXECUTABLE, REGULAR, "M"), (EXECUTABLE, ABSENT, "D"), (REGULAR, REGULAR, "M")],
    ids=["chmod-minus-x", "deleted", "plain-edit"],
)
def test_dropping_or_never_having_the_executable_bit_is_fine(
    old_mode: str, new_mode: str, status: str
) -> None:
    change = mode_change("tools/run.sh", old_mode, new_mode, status)

    assert evaluate_changes([change], make_policy("code"), LIMITS) == []


@pytest.fixture
def executable_repo(tmp_path: Path) -> tuple[GitRepoEnv, GitOps, str]:
    files = {
        **BASE_FILES,
        "tools/run.sh": "#!/bin/sh\necho one\n",
        "tools/plain.sh": "#!/bin/sh\necho plain\n",
    }
    git = make_git_env(tmp_path, files)
    (git.clone / "tools/run.sh").chmod(0o755)
    git.git("add", "-A")
    git.git("commit", "-q", "-m", "make run.sh executable")
    git.git("push", "-q", "origin", "main")
    ops = GitOps(git.clone, GitConfig(), state_dir=tmp_path / "state", environ=git.env)
    return git, ops, git.git("rev-parse", "HEAD")


def staged_violations(
    git: GitRepoEnv, ops: GitOps, base: str, scope: str = "code"
) -> list[tuple[str, str | None]]:
    ops.stage_all(git.clone)
    staged = ops.staged_changes(git.clone, base, max_diff_bytes=1_000_000)
    found = evaluate_changes(
        staged.changes, make_policy(scope), LIMITS, diff_bytes=staged.diff_bytes
    )
    return [(v.code, v.path) for v in found]


def test_real_git_editing_an_executable_file_is_not_a_violation(
    executable_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = executable_repo
    (git.clone / "tools/run.sh").write_text("#!/bin/sh\necho two\n", encoding="utf-8")

    ops.stage_all(git.clone)
    staged = ops.staged_changes(git.clone, base, max_diff_bytes=1_000_000)

    assert [(c.path, c.old_mode, c.new_mode) for c in staged.changes] == [
        ("tools/run.sh", EXECUTABLE, EXECUTABLE)
    ]
    assert staged_violations(git, ops, base) == []


def test_real_git_making_a_plain_file_executable_is_a_violation(
    executable_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = executable_repo
    (git.clone / "tools/plain.sh").chmod(0o755)

    assert staged_violations(git, ops, base) == [("executable", "tools/plain.sh")]


def test_real_git_a_new_executable_file_is_a_violation(
    executable_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = executable_repo
    (git.clone / "tools/new.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (git.clone / "tools/new.sh").chmod(0o755)

    assert staged_violations(git, ops, base) == [("executable", "tools/new.sh")]


def test_the_worker_accepts_an_agent_edit_of_an_existing_executable(
    make_rig_for: Any, tmp_path: Path
) -> None:
    git = make_git_env(tmp_path, {**BASE_FILES, "tools/run.sh": "#!/bin/sh\necho one\n"})
    (git.clone / "tools/run.sh").chmod(0o755)
    git.git("add", "-A")
    git.git("commit", "-q", "-m", "make run.sh executable")
    git.git("push", "-q", "origin", "main")
    rig = make_rig_for(git=git)
    number = rig.new_task_issue(scope="code", allowed_paths=["tools/**"])
    rig.set_scenario(
        [
            {"op": "write_file", "path": "tools/run.sh", "content": "#!/bin/sh\necho two\n"},
            {"op": "final", "text": "Done.", "report": {"status": "completed", "summary": "Edit"}},
        ]
    )

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.status, run.failure_kind, run.failure_detail)
    assert run.result is not None and run.result["files"] == ["tools/run.sh"]
    assert len(rig.fake.pulls) == 1


# =========================================================================== C-07
# ``git diff --numstat`` calls a file binary when a NUL byte sits in its first 8000 bytes, so a
# payload that starts as plain text and turns binary later passed the "no binary files" rule.
POLYGLOT = b"# note\n" + b"x" * 8100 + b"\n\x00\x7fELF" + random.Random(7).randbytes(2000)
LATIN1_DOC = b"# caf\xe9 notes\n" + b"plain text line\n" * 600  # not UTF-8, no NUL byte
UTF8_DOC = (
    "# Caf\u00e9 \u2014 na\u00efve fa\u00e7ade \u65e5\u672c\u8a9e \U0001f680\n" * 400
).encode()


@pytest.fixture
def blank_repo(tmp_path: Path) -> tuple[GitRepoEnv, GitOps, str]:
    git = make_git_env(tmp_path, {**BASE_FILES, "docs/old.md": "old\n"})
    ops = GitOps(git.clone, GitConfig(), state_dir=tmp_path / "state", environ=git.env)
    return git, ops, git.git("rev-parse", "HEAD")


def put(git: GitRepoEnv, rel: str, content: bytes | str) -> None:
    target = git.clone / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


def staged_binary(git: GitRepoEnv, ops: GitOps, base: str, **kwargs: Any) -> dict[str, bool]:
    ops.stage_all(git.clone)
    staged = ops.staged_changes(git.clone, base, max_diff_bytes=10_000_000, **kwargs)
    return {change.path: change.binary for change in staged.changes}


def test_git_alone_calls_the_polyglot_text(blank_repo: tuple[GitRepoEnv, GitOps, str]) -> None:
    """Control: this is exactly what the old numstat-only check saw."""
    git, _ops, base = blank_repo
    put(git, "docs/poly.md", POLYGLOT)
    git.git("add", "-A")

    numstat = git.git("diff", "--cached", "--numstat", base)

    assert not numstat.startswith("-\t-\t"), numstat  # git reports line counts, not "binary"


def test_a_nul_byte_after_the_first_8000_bytes_makes_the_file_binary(
    blank_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = blank_repo
    put(git, "docs/poly.md", POLYGLOT)

    assert staged_binary(git, ops, base) == {"docs/poly.md": True}


def test_invalid_utf8_without_a_nul_byte_is_binary_too(
    blank_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = blank_repo
    put(git, "docs/latin1.md", LATIN1_DOC)

    assert staged_binary(git, ops, base) == {"docs/latin1.md": True}


def test_ordinary_utf8_text_is_not_binary_however_non_ascii_it_is(
    blank_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = blank_repo
    put(git, "docs/unicode.md", UTF8_DOC)
    put(git, "docs/empty.md", b"")
    put(git, "docs/small.md", "# Small\n")
    put(git, "docs/old.md", "changed\n")  # a modification is scanned like an addition

    assert len(UTF8_DOC) > 8000
    assert staged_binary(git, ops, base) == {
        "docs/unicode.md": False,
        "docs/empty.md": False,
        "docs/small.md": False,
        "docs/old.md": False,
    }


def test_a_modified_file_that_turns_binary_is_flagged(
    blank_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = blank_repo
    put(git, "docs/old.md", POLYGLOT)

    assert staged_binary(git, ops, base) == {"docs/old.md": True}


@pytest.mark.parametrize("scope", ["docs-only", "code"])
def test_the_policy_reports_the_polyglot_as_binary(
    blank_repo: tuple[GitRepoEnv, GitOps, str], scope: str
) -> None:
    git, ops, base = blank_repo
    put(git, "docs/poly.md", POLYGLOT)
    put(git, "docs/fine.md", UTF8_DOC)

    assert staged_violations(git, ops, base, scope) == [("binary", "docs/poly.md")]


def spied_ops(
    git: GitRepoEnv, tmp_path: Path
) -> tuple[GitOps, list[tuple[list[str], bytes | None]]]:
    """A GitOps that records every git subprocess it starts (argv and stdin)."""
    calls: list[tuple[list[str], bytes | None]] = []

    def spy(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs.get("input")))
        return subprocess.run(argv, **kwargs)

    ops = GitOps(git.clone, GitConfig(), state_dir=tmp_path / "state", environ=git.env, run=spy)
    return ops, calls


def content_reads(calls: list[tuple[list[str], bytes | None]]) -> list[list[str]]:
    """The object ids requested by each ``cat-file --batch`` call."""
    return [
        stdin.decode("ascii").split()
        for argv, stdin in calls
        if "cat-file" in argv and "--batch" in argv and stdin is not None
    ]


def test_blobs_above_the_file_limit_are_never_read(
    blank_repo: tuple[GitRepoEnv, GitOps, str], tmp_path: Path
) -> None:
    """They are refused as too large anyway; reading them would only cost memory."""
    git, _ops, base = blank_repo
    ops, calls = spied_ops(git, tmp_path)
    put(git, "docs/small.md", b"\xff" * 50)
    put(git, "docs/big.md", b"\xff" * 5000)
    ops.stage_all(git.clone)

    staged = ops.staged_changes(git.clone, base, max_diff_bytes=1_000_000, max_file_bytes=1000)

    by_path = {c.path: c for c in staged.changes}
    assert by_path["docs/small.md"].binary is True
    assert by_path["docs/big.md"].binary is False and by_path["docs/big.md"].size == 5000
    assert content_reads(calls) == [[git.git("rev-parse", ":docs/small.md")]]


def test_many_blobs_are_read_in_bounded_batches(
    blank_repo: tuple[GitRepoEnv, GitOps, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.agent_bridge import gitops

    git, _ops, base = blank_repo
    ops, calls = spied_ops(git, tmp_path)
    monkeypatch.setattr(gitops, "SCAN_BATCH_BYTES", 2500)
    for index in range(12):
        put(git, f"docs/page{index:02d}.md", f"page {index} ".encode() * 100)
    put(git, "docs/page07.md", b"page 7 " * 100 + b"\x00")
    put(git, "docs/page11.md", b"\xfe" + b"page 11 " * 100)

    found = staged_binary(git, ops, base)

    assert [path for path, binary in found.items() if binary] == [
        "docs/page07.md",
        "docs/page11.md",
    ]
    assert len(found) == 12
    reads = content_reads(calls)
    assert len(reads) >= 4 and all(len(batch) <= 4 for batch in reads)
    assert sum(len(batch) for batch in reads) == 12


def test_the_worker_blocks_an_agent_that_hides_binary_data_behind_a_text_prefix(
    make_rig_for: Any,
) -> None:
    rig = make_rig_for()
    number = rig.new_task_issue()
    rig.set_scenario(
        [
            {
                "op": "write_file",
                "path": "docs/poly.md",
                "content": "# note\n" + "x" * 8100 + "\n\x00\x7fELF" + "y" * 100,
            },
            {"op": "final", "text": "Done.", "report": {"status": "completed", "summary": "Note"}},
        ]
    )

    run = rig.run_issue(number)

    assert run.status is RunStatus.BLOCKED, (run.status, run.failure_kind, run.failure_detail)
    assert run.failure_kind == "policy_violation"
    assert run.result is not None and run.result["violations"] == [
        "binary `docs/poly.md`: binary files are not allowed"
    ]
    assert rig.fake.pulls == {}
    assert rig.git_env.origin_branches() == ["main"]


# =========================================================================== C-08
# A staged path that is not valid UTF-8 (git hands such names over as lone surrogates) passed the
# policy, was committed and pushed, and then crashed publication when the PR body listing the
# files could not be encoded.  Zero-width and bidirectional characters were accepted likewise.
NON_UTF8_NAME = b"docs/\xff\xfe.md".decode("utf-8", errors="surrogateescape")
UNSAFE_NAMES = [
    pytest.param(NON_UTF8_NAME, id="not-utf8"),
    pytest.param("docs/\udcff.md", id="lone-surrogate"),
    pytest.param("docs/a\u202eb.md", id="right-to-left-override"),
    pytest.param("docs/a\u2066b.md", id="bidi-isolate"),
    pytest.param("docs/a\u200bb.md", id="zero-width-space"),
    pytest.param("docs/a\u200db.md", id="zero-width-joiner"),
    pytest.param("docs/\ufeffa.md", id="byte-order-mark"),
    pytest.param("docs/a\u00adb.md", id="soft-hyphen"),
    pytest.param("docs/a\u2060b.md", id="word-joiner"),
    pytest.param("docs/a\u2028b.md", id="line-separator"),
    pytest.param("docs/a\u2029b.md", id="paragraph-separator"),
    pytest.param("docs/a\x85b.md", id="next-line-control"),
    pytest.param("docs/a\x9bb.md", id="c1-csi"),
    pytest.param("docs/\U000e0041a.md", id="tag-character"),
]
FINE_NAMES = [
    "docs/caf\u00e9.md",
    "docs/cafe\u0301.md",
    "docs/\u65e5\u672c\u8a9e.md",
    "docs/\u00fc\u00f1\u00ee\u00e7\u00f8d\u00e9.md",
    "docs/\u0410\u0431\u0432.md",
    "docs/a b.md",
    "docs/\U0001f680.md",
]


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_normalize_repo_path_rejects_invisible_and_unencodable_characters(name: str) -> None:
    with pytest.raises(PathError, match=r"control, invisible or non-UTF-8"):
        normalize_repo_path(name)


@pytest.mark.parametrize("name", FINE_NAMES)
def test_normalize_repo_path_keeps_ordinary_unicode_names(name: str) -> None:
    assert normalize_repo_path(name) == name


@pytest.mark.parametrize("scope", ["docs-only", "code"])
@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_unsafe_staged_names_are_path_invalid(name: str, scope: str) -> None:
    found = evaluate_changes([added(name)], make_policy(scope), LIMITS)

    assert [v.code for v in found] == ["path_invalid"]
    assert "non-UTF-8" in found[0].detail


@pytest.mark.parametrize("name", FINE_NAMES)
def test_ordinary_unicode_names_are_not_violations(name: str) -> None:
    assert evaluate_changes([added(name)], make_policy("docs-only"), LIMITS) == []


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_a_rejected_name_is_reported_in_a_form_that_can_be_published(name: str) -> None:
    """The report is posted to GitHub; a lone surrogate there crashed the request encoder."""
    (violation,) = evaluate_changes([added(name)], make_policy("docs-only"), LIMITS)

    for text in (violation.path or "", violation.detail, violation.render()):
        text.encode("utf-8")  # raises UnicodeEncodeError for a surrogate
        assert not any(unicodedata.category(c) in {"Cs", "Cf", "Zl", "Zp"} for c in text)
    assert violation.path is not None and len(violation.path) <= 120
    assert violation.path.startswith("docs/")


def test_the_reported_name_spells_out_what_was_hidden() -> None:
    (surrogates,) = evaluate_changes([added(NON_UTF8_NAME)], make_policy("docs-only"), LIMITS)
    (override,) = evaluate_changes([added("docs/a\u202eb.md")], make_policy("docs-only"), LIMITS)

    assert surrogates.path == "docs/\\udcff\\udcfe.md"
    assert override.path == "docs/a\\u202eb.md"
    assert "U+DCFF" in surrogates.detail and "U+202E" in override.detail


def test_long_unsafe_names_are_still_cut_to_120_characters() -> None:
    (violation,) = evaluate_changes(
        [added("docs/" + "\udcff" * 200 + ".md")], make_policy("docs-only"), LIMITS
    )

    assert violation.path is not None and len(violation.path) <= 120


def test_an_unsafe_name_cannot_hide_a_protected_path() -> None:
    """Rejected as path_invalid before any glob is consulted: no spelling reaches the lists."""
    for name in (
        "docs/.claude\u200b/x.md",
        "docs/AGENTS\u202e.md",
        "backend/src/safety\u2028/a.py",
    ):
        assert [v.code for v in evaluate_changes([added(name)], make_policy("code"), LIMITS)] == [
            "path_invalid"
        ]


def make_odd_file(git: GitRepoEnv, raw_name: bytes) -> None:
    target = os.fsencode(str(git.clone)) + b"/" + raw_name
    try:
        with open(target, "wb") as handle:
            handle.write(b"z\n")
    except OSError:
        pytest.skip("this file system rejects the file name")


def test_real_git_a_non_utf8_file_name_is_a_path_invalid_violation(
    blank_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = blank_repo
    make_odd_file(git, b"docs/\xff\xfe.md")

    ops.stage_all(git.clone)
    staged = ops.staged_changes(git.clone, base, max_diff_bytes=1_000_000)

    assert [c.path for c in staged.changes] == [NON_UTF8_NAME]
    assert staged_violations(git, ops, base) == [("path_invalid", "docs/\\udcff\\udcfe.md")]


def test_real_git_bidirectional_and_zero_width_names_are_violations(
    blank_repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = blank_repo
    put(git, "docs/rtl\u202egnp.md", "x\n")
    put(git, "docs/zero\u200bwidth.md", "x\n")
    put(git, "docs/caf\u00e9.md", "x\n")

    assert staged_violations(git, ops, base) == [
        ("path_invalid", "docs/rtl\\u202egnp.md"),
        ("path_invalid", "docs/zero\\u200bwidth.md"),
    ]


@pytest.mark.parametrize(
    ("scenario_path", "shown"),
    [
        pytest.param("docs/\udcff\udcfe.md", "docs/\\udcff\\udcfe.md", id="not-utf8"),
        pytest.param("docs/a\u202eb.md", "docs/a\\u202eb.md", id="rtl-override"),
    ],
)
def test_the_worker_blocks_the_run_instead_of_crashing_in_publication(
    make_rig_for: Any, scenario_path: str, shown: str
) -> None:
    rig = make_rig_for()
    number = rig.new_task_issue()
    rig.set_scenario(
        [
            {"op": "write_file", "path": scenario_path, "content": "z\n"},
            {"op": "final", "text": "Done.", "report": {"status": "completed", "summary": "Note"}},
        ]
    )

    run = rig.run_issue(number)

    assert run.status is RunStatus.BLOCKED, (run.status, run.failure_kind, run.failure_detail)
    assert run.failure_kind == "policy_violation"
    assert run.result is not None
    assert [v.split(":")[0] for v in run.result["violations"]] == [f"path_invalid `{shown}`"]
    assert rig.fake.pulls == {}
    assert rig.git_env.origin_branches() == ["main"]  # nothing was pushed
    assert any("path_invalid" in body for body in rig.comments(number))


# =========================================================================== C-09
# ``extract_report`` used a lazy ``.*?`` with DOTALL tried from every ``agent-result`` opener, so
# an agent that ends its message with many unterminated openers made the single worker burn about
# a minute of CPU after the process had exited (27 s for 16000 openers, 256k characters).
REPORT = {"status": "completed", "summary": "Edited docs", "files_changed": ["docs/a.md"]}
LEGACY_REPORT_FENCE = re.compile(
    r"^ {0,3}(?P<fence>`{3,})\s*(?:json\s+)?agent-result\s*$\n(?P<body>.*?)^ {0,3}(?P=fence)\s*$",
    re.MULTILINE | re.DOTALL,
)


def legacy_extract_report(text: str) -> AgentReport | None:
    """The implementation C-09 replaced, kept as the oracle for the differential test."""
    for match in reversed(list(LEGACY_REPORT_FENCE.finditer(text))):
        try:
            return AgentReport.model_validate(json.loads(match.group("body")))
        except (ValueError, ValidationError, RecursionError):
            continue
    return None


def fenced_report(body: str | dict[str, Any] = REPORT, ticks: str = "```") -> str:
    text = body if isinstance(body, str) else json.dumps(body)
    return f"{ticks}json agent-result\n{text}\n{ticks}\n"


class Stalled(Exception):
    """The block needed more wall-clock time than allowed."""


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    def on_alarm(signum: int, frame: object) -> None:
        raise Stalled

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


needs_timers = pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs interval timers")

HOSTILE_REPORT_TEXTS: dict[str, Callable[[], str]] = {
    "the-reviews-case": lambda: "```agent-result\n" * 16_000,
    "json-openers": lambda: "```json agent-result\n" * 12_000,
    "indented-openers": lambda: "   ```agent-result\n" * 14_000,
    "openers-then-one-closer": lambda: "```agent-result\n" * 16_000 + "```\n",
    "openers-then-a-longer-closer": lambda: "```agent-result\n" * 16_000 + "````\n",
    "distinct-fence-lengths": lambda: "".join(
        "`" * (3 + index % 400) + "agent-result\n" for index in range(5_000)
    ),
    "openers-and-prose": lambda: "```agent-result\nsome prose here\n" * 8_000,
    "crlf-openers": lambda: "```agent-result\r\n" * 16_000,
    "one-huge-backtick-line": lambda: "`" * 400_000,
    "one-huge-info-line": lambda: "```" + " " * 400_000 + "x",
    "json-then-spaces": lambda: "```json" + " " * 400_000 + "x",
    "blank-lines": lambda: "\n" * 400_000,
    "closers-only": lambda: "```\n" * 100_000,
    "unterminated-final-report": lambda: (
        "noise\n" * 20_000 + "```agent-result\n" + json.dumps(REPORT)
    ),
}


@needs_timers
@pytest.mark.parametrize("name", sorted(HOSTILE_REPORT_TEXTS))
def test_extract_report_is_fast_on_hostile_text(name: str) -> None:
    text = HOSTILE_REPORT_TEXTS[name]()

    started = time.process_time()
    try:
        with time_limit(5.0):
            report = extract_report(text)
    except Stalled:
        pytest.fail(f"extract_report needed more than 5 s for {len(text)} characters of {name}")

    assert report is None
    assert time.process_time() - started < 2.0, name


@needs_timers
def test_extract_report_finds_a_report_after_a_wall_of_unterminated_openers() -> None:
    """Openers of another fence length never close, so the real block after them is found.

    (Same-length openers pair up with the report's own closing fence, exactly as in CommonMark
    and as the replaced regex did: the first opener's block then swallows the report.)
    """
    text = "````agent-result\n" * 16_000 + fenced_report({**REPORT, "summary": "the real one"})

    with time_limit(5.0):
        report = extract_report(text)

    assert report is not None and report.summary == "the real one"


def test_a_report_after_same_length_unterminated_openers_is_swallowed_as_before() -> None:
    wall = "```agent-result\n" * 3
    text = wall + fenced_report({**REPORT, "summary": "swallowed"})

    assert extract_report(text) is None
    assert extract_report(text) == legacy_extract_report(text)


@needs_timers
def test_extract_report_time_grows_linearly_not_quadratically() -> None:
    def cost(openers: int) -> float:
        text = "```agent-result\n" * openers
        started = time.process_time()
        with time_limit(10.0):
            extract_report(text)
        return time.process_time() - started

    small, large = cost(4_000), cost(32_000)  # 8x the input

    assert large < 0.5, large
    assert large < max(small, 0.005) * 40  # quadratic growth would be ~64x


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no fence at all",
        "```agent-result\n",
        "```agent-result",
        "```json agent-result\n{}\n```",
        "```agent-result\n{not json}\n```\n",
        "    ```agent-result\n" + json.dumps(REPORT) + "\n```\n",  # indented four spaces
        "```agent-results\n" + json.dumps(REPORT) + "\n```\n",
        "``agent-result\n" + json.dumps(REPORT) + "\n``\n",  # two backticks are no fence
    ],
)
def test_extract_report_returns_none_without_a_valid_block(text: str) -> None:
    assert extract_report(text) is None


@pytest.mark.parametrize(
    "opener", ["``` agent-result", "```\tjson   agent-result  ", "   ```agent-result"]
)
def test_the_opener_tolerates_whitespace_around_its_info_string(opener: str) -> None:
    report = extract_report(f"{opener}\n{json.dumps(REPORT)}\n```\n")

    assert report is not None and report.summary == "Edited docs"


def test_the_last_complete_valid_fence_wins() -> None:
    first = fenced_report({**REPORT, "summary": "first"})
    second = fenced_report({**REPORT, "summary": "second", "status": "blocked"})
    broken = fenced_report("{nope")

    assert (extract_report(f"{first}\n{second}") or fail()).summary == "second"
    assert (extract_report(f"{first}\n{broken}") or fail()).summary == "first"
    assert (extract_report(f"{broken}\n{second}\n{broken}") or fail()).summary == "second"


def test_a_fence_needs_the_same_number_of_backticks_to_close() -> None:
    inner = f"```agent-result\n{json.dumps(REPORT)}\n```\n"
    outer = "````agent-result\n" + inner + "````\n"

    assert extract_report(outer) is None  # the outer body is not JSON; the inner fence is body
    assert extract_report("````agent-result\n" + json.dumps(REPORT) + "\n```\n") is None
    assert extract_report("```agent-result\n" + json.dumps(REPORT) + "\n````\n") is None
    assert extract_report("````agent-result\n" + json.dumps(REPORT) + "\n````\n") is not None


def test_a_report_after_an_unclosed_longer_fence_is_still_found() -> None:
    text = "````agent-result\nlost\n" + fenced_report({**REPORT, "summary": "found"})

    assert (extract_report(text) or fail()).summary == "found"


def fail() -> AgentReport:
    pytest.fail("no report found")


# Differential test against the replaced regex.  Inputs are shuffled from fence lines, info
# strings, bodies and filler; the replaced regex also let the opener be split over two lines
# (a bare fence line, then a line reading ``agent-result``), which no agent output relies on and
# the line-based scanner does not accept, so those shapes are left out.
SPLIT_OPENER = re.compile(r"^ {0,3}`{3,}\s*\n\s*(?:json\s+)?agent-result\s*$", re.MULTILINE)
INFO_STRINGS = [
    "agent-result",
    "json agent-result",
    "json   agent-result  ",
    "  agent-result",
    "\tagent-result",
    "agent-result\t",
    "AGENT-RESULT",
    "python agent-result",
    "agent-results",
    "json",
    "",
    "agent-result extra",
]
BODIES = [
    json.dumps(REPORT),
    json.dumps({**REPORT, "summary": "second", "status": "blocked"}),
    json.dumps(REPORT, indent=2),
    "{not json",
    "[1, 2]",
    "{}",
    "",
    "   ",
    "prose line",
    "```python",
    "````",
    "~~~",
]
FILLER = ["", "text", "   ", "Thanks!", "\t"]


def random_report_text(rng: random.Random) -> str:
    """A few fenced blocks, each opened, filled and (usually) closed in a slightly wrong way."""
    lines: list[str] = []
    for _ in range(rng.randint(1, 5)):
        ticks = rng.choice([3, 3, 3, 4, 5])
        lines.append(
            " " * rng.choice([0, 0, 0, 1, 3, 4])
            + "`" * ticks
            + rng.choice(INFO_STRINGS[:3] * 3 + INFO_STRINGS)
        )
        lines.extend(rng.choice(BODIES[:3] * 3 + BODIES) for _ in range(rng.randint(0, 3)))
        if rng.random() < 0.85:
            closing = ticks if rng.random() < 0.8 else rng.choice([3, 4, 5, 6])
            lines.append(
                " " * rng.choice([0, 0, 0, 1, 3, 4])
                + "`" * closing
                + rng.choice(["", "", "", " ", "\t", " x"])
            )
        lines.extend(rng.choice(FILLER) for _ in range(rng.randint(0, 2)))
    eol = rng.choice(["\n", "\n", "\r\n"])
    return eol.join(lines) + rng.choice(["", eol])


def test_the_scanner_agrees_with_the_replaced_regex_on_random_texts() -> None:
    rng = random.Random(20261024)
    checked = found = 0
    for _ in range(6_000):
        text = random_report_text(rng)
        if SPLIT_OPENER.search(text):
            continue
        expected, actual = legacy_extract_report(text), extract_report(text)
        assert actual == expected, text
        checked += 1
        found += expected is not None
    assert checked > 4_000 and 500 < found < checked - 500  # both outcomes are well represented
