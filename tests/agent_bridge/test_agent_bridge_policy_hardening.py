"""Hardening of the change policy, its node twin and the report parser.

Each section pins one finding of the PR 17 containment review (``C-04`` .. ``C-09``).  The
policy tests run ``evaluate_changes`` on hand-made :class:`Change` rows and, where the file
system matters, against real git; the guard tests drive the real ``omp_guard.mjs`` through node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.rig import BASE_FILES, GitRepoEnv, make_git_env

from tools.agent_bridge import globs
from tools.agent_bridge.agents.omp import GUARD_PATH
from tools.agent_bridge.config import DEFAULT_PROTECTED_PATHS, BridgeConfig, GitConfig
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.globs import match_any
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
