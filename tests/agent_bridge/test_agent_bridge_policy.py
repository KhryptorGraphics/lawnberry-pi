"""Tests for ``tools/agent_bridge/policy.py``: what an agent may leave in its worktree."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.rig import BASE_FILES, GitRepoEnv, make_git_env

from tools.agent_bridge.config import DEFAULT_PROTECTED_PATHS, BridgeConfig, GitConfig, LimitsConfig
from tools.agent_bridge.gitops import GitOps
from tools.agent_bridge.policy import (
    Change,
    Violation,
    WritePolicy,
    build_write_policy,
    evaluate_changes,
)
from tools.agent_bridge.schema import TaskSpec

OBJECTIVE = "Add a short note about the agent bridge to docs/guide.md."
REGULAR, EXECUTABLE, SYMLINK, GITLINK, ABSENT = "100644", "100755", "120000", "160000", "000000"


def make_config(**overrides: Any) -> BridgeConfig:
    data = {"repository": "owner/repo", "approvers": ["alice"], **overrides}
    return BridgeConfig.model_validate(data)


def make_spec(**fields: Any) -> TaskSpec:
    return TaskSpec.model_validate({"version": 1, "objective": OBJECTIVE, **fields})


def docs_policy(**config: Any) -> WritePolicy:
    return build_write_policy(make_spec(), make_config(**config))


def added(path: str, **kwargs: Any) -> Change:
    base: dict[str, Any] = {
        "status": "A",
        "old_mode": ABSENT,
        "new_mode": REGULAR,
        "size": 10,
        "added": 1,
    }
    return Change(path=path, **{**base, **kwargs})


def modified(path: str, **kwargs: Any) -> Change:
    base: dict[str, Any] = {
        "status": "M",
        "old_mode": REGULAR,
        "new_mode": REGULAR,
        "size": 10,
        "added": 1,
        "deleted": 1,
    }
    return Change(path=path, **{**base, **kwargs})


def deleted(path: str, **kwargs: Any) -> Change:
    base: dict[str, Any] = {
        "status": "D",
        "old_mode": REGULAR,
        "new_mode": ABSENT,
        "size": None,
        "deleted": 3,
    }
    return Change(path=path, **{**base, **kwargs})


def codes(violations: list[Violation]) -> list[tuple[str, str | None]]:
    return [(v.code, v.path) for v in violations]


LIMITS = make_config().limits


# --------------------------------------------------------------------------- build_write_policy
def test_docs_only_policy_uses_the_docs_scope_and_the_protected_defaults() -> None:
    policy = docs_policy()

    assert policy.allow_sets == (("docs/**", "*.md"),)
    assert policy.deny == tuple(DEFAULT_PROTECTED_PATHS)


def test_code_scope_uses_the_code_paths() -> None:
    policy = build_write_policy(make_spec(scope="code"), make_config(scopes={"allowed": ["code"]}))
    assert policy.allow_sets == (("docs/**", "*.md", "tools/**", "tests/**"),)


def test_task_allowed_paths_add_a_narrowing_set() -> None:
    policy = build_write_policy(
        make_spec(allowed_paths=["docs/api/**", "docs/api.md"]), make_config()
    )

    assert policy.allow_sets == (("docs/**", "*.md"), ("docs/api/**", "docs/api.md"))
    assert policy.check_path("docs/api/ref.md") is None
    assert policy.check_path("docs/guide.md").code == "outside_scope"  # type: ignore[union-attr]


def test_a_task_cannot_widen_the_configured_scope() -> None:
    policy = build_write_policy(make_spec(allowed_paths=["src/**", "**"]), make_config())

    for path in ("src/app.py", "backend/src/services/weather.py", "frontend/App.vue"):
        violation = policy.check_path(path)
        assert violation is not None and violation.code == "outside_scope", path


def test_a_task_cannot_unprotect_anything() -> None:
    policy = build_write_policy(make_spec(allowed_paths=["**"]), make_config())
    assert policy.check_path("docs/constitution.md").code == "protected_path"  # type: ignore[union-attr]
    assert policy.check_path("backend/src/safety/estop.py").code == "protected_path"  # type: ignore[union-attr]


def test_configured_scopes_and_protected_paths_are_honoured() -> None:
    cfg = make_config(scopes={"docs_paths": ["notes/**"]}, protected_paths=["notes/private/**"])
    policy = build_write_policy(make_spec(), cfg)

    assert policy.allow_sets == (("notes/**",),)
    assert policy.deny == ("notes/private/**",)
    assert policy.check_path("notes/a.md") is None
    assert policy.check_path("notes/private/a.md").code == "protected_path"  # type: ignore[union-attr]
    assert policy.check_path("docs/a.md").code == "outside_scope"  # type: ignore[union-attr]


def test_an_empty_scope_fails_closed() -> None:
    policy = build_write_policy(make_spec(), make_config(scopes={"docs_paths": []}))
    for path in ("docs/a.md", "README.md", "anything"):
        assert policy.check_path(path).code == "outside_scope"  # type: ignore[union-attr]


def test_policy_is_immutable_and_built_from_tuples() -> None:
    policy = docs_policy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.deny = ()  # type: ignore[misc]
    assert isinstance(policy.deny, tuple)
    assert all(isinstance(s, tuple) for s in policy.allow_sets)


def test_describe_is_the_json_contract_with_the_omp_guard() -> None:
    policy = build_write_policy(make_spec(allowed_paths=["docs/api/**"]), make_config())
    described = policy.describe()

    assert set(described) == {"write_allow_sets", "write_deny"}
    assert described["write_allow_sets"] == [["docs/**", "*.md"], ["docs/api/**"]]
    assert described["write_deny"] == list(DEFAULT_PROTECTED_PATHS)
    assert json.loads(json.dumps(described)) == described
    described["write_deny"].append("mutated")  # a copy: the policy itself is unaffected
    assert "mutated" not in policy.deny


# --------------------------------------------------------------------------- check_path
@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("docs/guide.md", None),
        ("docs/deep/er/page.md", None),
        ("docs", None),
        ("README.md", None),
        ("CHANGELOG.md", None),
        ("docs/constitution.md", "protected_path"),
        ("docs/.env", "protected_path"),
        ("docs/keys/server.pem", "protected_path"),
        ("CLAUDE.md", "protected_path"),
        ("AGENTS.md", "protected_path"),
        ("backend/src/safety/estop.py", "protected_path"),  # also outside the docs scope
        ("src/app.py", "outside_scope"),
        ("frontend/src/App.vue", "outside_scope"),
        ("notes.txt", "outside_scope"),
        ("sub/README.md", "outside_scope"),
    ],
)
def test_check_path_decisions_for_the_docs_policy(path: str, code: str | None) -> None:
    violation = docs_policy().check_path(path)
    assert (violation.code if violation else None) == code
    if violation is not None:
        assert violation.path == path and violation.detail


def test_protected_wins_over_allowed_and_over_out_of_scope() -> None:
    policy = WritePolicy(allow_sets=(("docs/**",),), deny=("docs/secret/**", "other/**"))

    assert policy.check_path("docs/secret/x.md").code == "protected_path"  # type: ignore[union-attr]
    assert policy.check_path("other/x.md").code == "protected_path"  # type: ignore[union-attr]  # also out of scope
    assert policy.check_path("third/x.md").code == "outside_scope"  # type: ignore[union-attr]
    assert policy.check_path("docs/ok.md") is None


def test_every_allow_set_must_match() -> None:
    policy = WritePolicy(allow_sets=(("docs/**", "*.md"), ("docs/api/**",), ("**/*.md",)), deny=())

    assert policy.check_path("docs/api/ref.md") is None
    assert policy.check_path("docs/api/ref.txt").code == "outside_scope"  # type: ignore[union-attr]
    assert policy.check_path("docs/guide.md").code == "outside_scope"  # type: ignore[union-attr]
    assert policy.check_path("README.md").code == "outside_scope"  # type: ignore[union-attr]


def test_any_glob_within_one_set_is_enough() -> None:
    policy = WritePolicy(allow_sets=(("a/**", "b/**", "c.md"),), deny=())
    for path in ("a/x", "b/y/z", "c.md"):
        assert policy.check_path(path) is None
    assert policy.check_path("d/x").code == "outside_scope"  # type: ignore[union-attr]


def test_violation_rendering() -> None:
    assert Violation("symlink", "not allowed", "docs/a.md").render() == (
        "symlink `docs/a.md`: not allowed"
    )
    assert Violation("too_many_files", "41 files changed").render() == (
        "too_many_files: 41 files changed"
    )
    assert Violation("a", "b").path is None


def test_change_defaults_describe_an_unmodified_empty_entry() -> None:
    change = Change(path="docs/a.md", status="M")
    assert (change.old_mode, change.new_mode) == (ABSENT, ABSENT)
    assert change.size is None and change.binary is False
    assert (change.added, change.deleted) == (0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        change.path = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- evaluate_changes
def evaluate(
    changes: list[Change], policy: WritePolicy | None = None, **kwargs: Any
) -> list[Violation]:
    limits = kwargs.pop("limits", LIMITS)
    return evaluate_changes(changes, policy or docs_policy(), limits, **kwargs)


def test_a_clean_change_set_has_no_violations() -> None:
    changes = [
        added("docs/new.md"),
        modified("docs/guide.md"),
        modified("README.md"),
        deleted("docs/old.md"),
        added("docs/sub/dir/page.md", size=999_999),
    ]
    assert evaluate(changes) == []


def test_an_empty_change_set_has_no_violations() -> None:
    assert evaluate([]) == []


SINGLE_VIOLATIONS: list[tuple[str, Change, list[tuple[str, str | None]]]] = [
    (
        "protected-modified",
        modified("backend/src/safety/estop.py"),
        [("protected_path", "backend/src/safety/estop.py")],
    ),
    (
        "protected-deleted",
        deleted("config/hardware.yaml"),
        [("protected_path", "config/hardware.yaml")],
    ),
    ("protected-added", added("systemd/new.service"), [("protected_path", "systemd/new.service")]),
    (
        "protected-in-docs",
        added("docs/constitution.md"),
        [("protected_path", "docs/constitution.md")],
    ),
    ("protected-secret-name", added("docs/.env.local"), [("protected_path", "docs/.env.local")]),
    ("outside-scope", added("src/app.py"), [("outside_scope", "src/app.py")]),
    ("outside-scope-deleted", deleted("frontend/x.vue"), [("outside_scope", "frontend/x.vue")]),
    ("symlink-added", added("docs/link.md", new_mode=SYMLINK), [("symlink", "docs/link.md")]),
    (
        "symlink-deleted",
        deleted("docs/link.md", old_mode=SYMLINK),
        [("symlink", "docs/link.md")],
    ),
    (
        "symlink-type-change",
        Change("docs/a.md", "T", REGULAR, SYMLINK, size=5),
        [("symlink", "docs/a.md"), ("type_change", "docs/a.md")],
    ),
    (
        "file-replaced-by-directory-link",
        Change("docs/a.md", "T", SYMLINK, REGULAR, size=5),
        [("symlink", "docs/a.md"), ("type_change", "docs/a.md")],
    ),
    ("gitlink-added", added("docs/sub", new_mode=GITLINK, size=None), [("submodule", "docs/sub")]),
    (
        "gitlink-removed",
        deleted("docs/sub", old_mode=GITLINK),
        [("submodule", "docs/sub")],
    ),
    (
        "executable-added",
        added("docs/run.sh", new_mode=EXECUTABLE),
        [("executable", "docs/run.sh")],
    ),
    (
        "executable-bit-set",
        modified("docs/guide.md", new_mode=EXECUTABLE),
        [("executable", "docs/guide.md")],
    ),
    ("executable-bit-cleared-is-fine", modified("docs/run.sh", old_mode=EXECUTABLE), []),
    ("executable-deleted-is-fine", deleted("docs/run.sh", old_mode=EXECUTABLE), []),
    ("binary", added("docs/blob.bin", binary=True), [("binary", "docs/blob.bin")]),
    (
        "too-large",
        added("docs/big.md", size=LIMITS.max_file_bytes + 1),
        [("too_large", "docs/big.md")],
    ),
    ("exactly-at-the-size-limit", added("docs/big.md", size=LIMITS.max_file_bytes), []),
    ("unknown-size-is-not-too-large", added("docs/big.md", size=None), []),
    ("canonical-nested-path", added("docs/a/b/c/d.md"), []),
]


@pytest.mark.parametrize(
    ("change", "expected"),
    [(c, e) for _, c, e in SINGLE_VIOLATIONS],
    ids=[n for n, _, _ in SINGLE_VIOLATIONS],
)
def test_single_change_violations(change: Change, expected: list[tuple[str, str | None]]) -> None:
    assert codes(evaluate([change])) == expected


INVALID_PATHS = [
    ("", "path_invalid"),
    ("   ", "path_invalid"),
    ("/etc/passwd", "path_invalid"),
    ("~/x.md", "path_invalid"),
    ("../docs/a.md", "path_invalid"),
    ("docs/../../x", "path_invalid"),
    ("..", "path_invalid"),
    (".", "path_invalid"),
    ("docs\\a.md", "path_invalid"),
    ("docs/a\x00.md", "path_invalid"),
    ("docs/a\nb.md", "path_invalid"),
    ("docs/a\tb.md", "path_invalid"),
    ("docs/a\x1b[31m.md", "path_invalid"),
    ("http://example.com/x", "path_invalid"),
    ("docs/" + "a" * 500 + ".md", "path_invalid"),
    # accepted by normalisation but not canonical
    ("docs//a.md", "path_invalid"),
    ("./docs/a.md", "path_invalid"),
    ("docs/./a.md", "path_invalid"),
    ("docs/a.md/", "path_invalid"),
    ("docs/x/../a.md", "path_invalid"),
    (" docs/a.md", "path_invalid"),
    ("docs/a.md ", "path_invalid"),
]


@pytest.mark.parametrize("path", [p for p, _ in INVALID_PATHS], ids=lambda p: repr(p)[:30])
def test_invalid_or_non_canonical_paths_are_violations(path: str) -> None:
    violations = evaluate([added(path)])

    assert len(violations) == 1
    assert violations[0].code == "path_invalid"
    assert violations[0].detail
    assert violations[0].path is None or len(violations[0].path) <= 120


def test_overlong_paths_are_truncated_in_the_report() -> None:
    violation = evaluate([added("a" * 5000)])[0]
    assert violation.code == "path_invalid" and violation.path == "a" * 120


def test_non_canonical_paths_cannot_dodge_the_protected_list() -> None:
    for sneaky in (
        "backend/src/safety/../safety/estop.py",
        "./backend/src/safety/estop.py",
        "backend//src/safety/estop.py",
        "docs/../backend/src/safety/estop.py",
    ):
        assert [v.code for v in evaluate([modified(sneaky)])] == ["path_invalid"], sneaky


def test_one_change_can_violate_several_rules_at_once() -> None:
    change = Change(
        "backend/src/safety/estop.py",
        "T",
        REGULAR,
        SYMLINK,
        size=LIMITS.max_file_bytes * 2,
        binary=True,
    )

    assert [v.code for v in evaluate([change])] == [
        "protected_path",
        "symlink",
        "type_change",
        "binary",
        "too_large",
    ]


def test_executable_gitlink_and_symlink_rules_are_independent() -> None:
    change = Change("docs/odd", "M", GITLINK, EXECUTABLE, size=1)
    assert [v.code for v in evaluate([change])] == ["submodule", "executable"]


def test_violations_accumulate_across_changes_in_order() -> None:
    changes = [
        added("docs/ok.md"),
        added("src/outside.py"),
        modified("backend/src/drivers/motor.py"),
        added("docs/link.md", new_mode=SYMLINK),
        added("docs/blob.bin", binary=True),
    ]

    assert codes(evaluate(changes)) == [
        ("outside_scope", "src/outside.py"),
        ("protected_path", "backend/src/drivers/motor.py"),
        ("symlink", "docs/link.md"),
        ("binary", "docs/blob.bin"),
    ]


def test_too_many_files_is_reported_first_and_boundaries_are_exact() -> None:
    limit = LIMITS.max_changed_files
    at_limit = [added(f"docs/{i}.md") for i in range(limit)]
    assert evaluate(at_limit) == []

    over = [*at_limit, added("docs/one-more.md")]
    violations = evaluate(over)
    assert [v.code for v in violations] == ["too_many_files"]
    assert violations[0].path is None
    assert str(limit + 1) in violations[0].detail and str(limit) in violations[0].detail

    both = [*over[:-1], added("src/x.py"), added("src/y.py")]
    assert [v.code for v in evaluate(both)] == ["too_many_files", "outside_scope", "outside_scope"]


def test_diff_size_boundaries_are_exact() -> None:
    limit = LIMITS.max_diff_bytes
    assert evaluate([added("docs/a.md")], diff_bytes=limit) == []
    violations = evaluate([added("docs/a.md")], diff_bytes=limit + 1)
    assert [v.code for v in violations] == ["diff_too_big"]
    assert str(limit + 1) in violations[0].detail and str(limit) in violations[0].detail


def test_global_limits_come_before_per_file_findings() -> None:
    limits = LimitsConfig(max_changed_files=1, max_diff_bytes=1000)
    violations = evaluate_changes(
        [added("src/a.py"), added("src/b.py")], docs_policy(), limits, diff_bytes=5000
    )
    assert [v.code for v in violations] == [
        "too_many_files",
        "diff_too_big",
        "outside_scope",
        "outside_scope",
    ]


def test_custom_limits_are_used() -> None:
    limits = LimitsConfig(max_file_bytes=1000)
    assert evaluate_changes([added("docs/a.md", size=1000)], docs_policy(), limits) == []
    found = evaluate_changes([added("docs/a.md", size=1001)], docs_policy(), limits)
    assert [v.code for v in found] == ["too_large"]


def test_evaluation_does_not_mutate_its_inputs() -> None:
    policy = docs_policy()
    changes = [added("src/a.py"), added("docs/ok.md")]
    before = (policy, list(changes))
    evaluate(changes, policy)
    assert (policy, changes) == before


# --------------------------------------------------------------------------- real git end to end
@pytest.fixture
def repo(tmp_path: Path) -> tuple[GitRepoEnv, GitOps, str]:
    git = make_git_env(tmp_path, {**BASE_FILES, "docs/old.md": "old\n", "notes.txt": "n\n"})
    ops = GitOps(git.clone, GitConfig(), state_dir=tmp_path / "state", environ=git.env)
    return git, ops, git.git("rev-parse", "HEAD")


def staged_violations(
    git: GitRepoEnv,
    ops: GitOps,
    base: str,
    *,
    policy: WritePolicy | None = None,
    limits: LimitsConfig | None = None,
) -> tuple[list[Violation], Any]:
    ops.stage_all(git.clone)
    staged = ops.staged_changes(git.clone, base, max_diff_bytes=1_000_000)
    found = evaluate_changes(
        staged.changes, policy or docs_policy(), limits or LIMITS, diff_bytes=staged.diff_bytes
    )
    return found, staged


def write(git: GitRepoEnv, rel: str, content: str | bytes = "text\n") -> Path:
    target = git.clone / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")
    return target


def mutate_clean(git: GitRepoEnv) -> None:
    write(git, "docs/guide.md", "# Guide\n\nOriginal text.\n\nAnd a new paragraph.\n")
    write(git, "docs/new/page.md", "# New\n")
    write(git, "README.md", "# Test repo\n\nMore.\n")
    (git.clone / "docs/old.md").unlink()


def mutate_protected(git: GitRepoEnv) -> None:
    write(git, "backend/src/safety/estop.py", "ESTOP = False\n")
    (git.clone / "config/hardware.yaml").unlink()
    write(git, "src/outside.py", "x = 1\n")
    write(git, "docs/ok.md", "fine\n")


def mutate_symlink(git: GitRepoEnv) -> None:
    (git.clone / "docs").mkdir(exist_ok=True)
    os.symlink("../README.md", git.clone / "docs/link.md")


def mutate_executable(git: GitRepoEnv) -> None:
    write(git, "docs/run.sh", "#!/bin/sh\necho hi\n").chmod(0o755)


def mutate_mode_only(git: GitRepoEnv) -> None:
    (git.clone / "docs/guide.md").chmod(0o755)


def mutate_binary(git: GitRepoEnv) -> None:
    write(git, "docs/blob.bin", b"\x00\x01\x02binary\x00" * 10)


def mutate_type_change(git: GitRepoEnv) -> None:
    (git.clone / "docs/guide.md").unlink()
    os.symlink("../README.md", git.clone / "docs/guide.md")


def mutate_odd_names(git: GitRepoEnv) -> None:
    write(git, "docs/with\nnewline.md")
    write(git, "docs/back\\slash.md")
    write(git, "docs/fine name.md")


def mutate_rename_into_docs(git: GitRepoEnv) -> None:
    git.git("mv", "backend/src/safety/estop.py", "docs/estop.py")


def mutate_nested_repository(git: GitRepoEnv) -> None:
    nested = git.clone / "docs/nested"
    nested.mkdir(parents=True)
    git.git("init", "-q", cwd=nested)
    (nested / "a.md").write_text("inside\n", encoding="utf-8")
    git.git("add", "-A", cwd=nested)
    git.git("commit", "-q", "-m", "nested", cwd=nested)


REAL_GIT_SCENARIOS: list[tuple[str, Callable[[GitRepoEnv], None], list[tuple[str, str | None]]]] = [
    ("clean-documentation-change", mutate_clean, []),
    (
        "protected-and-out-of-scope",
        mutate_protected,
        [
            ("protected_path", "backend/src/safety/estop.py"),
            ("protected_path", "config/hardware.yaml"),
            ("outside_scope", "src/outside.py"),
        ],
    ),
    ("symlink", mutate_symlink, [("symlink", "docs/link.md")]),
    ("executable-file", mutate_executable, [("executable", "docs/run.sh")]),
    ("mode-change-only", mutate_mode_only, [("executable", "docs/guide.md")]),
    ("binary-file", mutate_binary, [("binary", "docs/blob.bin")]),
    (
        "file-turned-into-symlink",
        mutate_type_change,
        [("symlink", "docs/guide.md"), ("type_change", "docs/guide.md")],
    ),
    (
        "unsafe-file-names",
        mutate_odd_names,
        [("path_invalid", "docs/back\\slash.md"), ("path_invalid", "docs/with\nnewline.md")],
    ),
    (
        "protected-file-moved-into-docs",
        mutate_rename_into_docs,
        [("protected_path", "backend/src/safety/estop.py")],
    ),
    ("nested-git-repository", mutate_nested_repository, [("submodule", "docs/nested")]),
]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [(m, e) for _, m, e in REAL_GIT_SCENARIOS],
    ids=[n for n, _, _ in REAL_GIT_SCENARIOS],
)
def test_real_staged_changes_are_judged_correctly(
    repo: tuple[GitRepoEnv, GitOps, str],
    mutate: Callable[[GitRepoEnv], None],
    expected: list[tuple[str, str | None]],
) -> None:
    git, ops, base = repo
    mutate(git)

    found, _ = staged_violations(git, ops, base)

    assert sorted(codes(found), key=lambda item: (item[0], item[1] or "")) == sorted(
        expected, key=lambda item: (item[0], item[1] or "")
    )


def test_real_clean_change_reports_sane_statistics(repo: tuple[GitRepoEnv, GitOps, str]) -> None:
    git, ops, base = repo
    mutate_clean(git)

    found, staged = staged_violations(git, ops, base)

    assert found == []
    by_path = {c.path: c for c in staged.changes}
    assert set(by_path) == {"docs/guide.md", "docs/new/page.md", "README.md", "docs/old.md"}
    assert by_path["docs/new/page.md"].status == "A" and by_path["docs/new/page.md"].size == 6
    assert by_path["docs/old.md"].status == "D" and by_path["docs/old.md"].size is None
    assert by_path["docs/guide.md"].status == "M" and by_path["docs/guide.md"].added >= 1
    assert all(
        c.old_mode in (REGULAR, ABSENT) and c.new_mode in (REGULAR, ABSENT) for c in staged.changes
    )
    assert staged.diff_bytes > 0 and not staged.truncated


def test_real_oversize_file_trips_the_size_limit(repo: tuple[GitRepoEnv, GitOps, str]) -> None:
    git, ops, base = repo
    write(git, "docs/big.md", "x" * 2000)
    write(git, "docs/small.md", "x" * 1000)

    found, _ = staged_violations(git, ops, base, limits=LimitsConfig(max_file_bytes=1000))

    assert codes(found) == [("too_large", "docs/big.md")]


def test_real_change_count_and_diff_size_limits(repo: tuple[GitRepoEnv, GitOps, str]) -> None:
    git, ops, base = repo
    for index in range(5):
        write(git, f"docs/page{index}.md", "line\n" * 100)

    found, staged = staged_violations(
        git, ops, base, limits=LimitsConfig(max_changed_files=4, max_diff_bytes=1000)
    )

    assert [v.code for v in found] == ["too_many_files", "diff_too_big"]
    assert staged.diff_bytes > 1000


def test_real_scope_narrowing_by_the_task(repo: tuple[GitRepoEnv, GitOps, str]) -> None:
    git, ops, base = repo
    write(git, "docs/api/ref.md", "api\n")
    write(git, "docs/guide.md", "changed\n")
    policy = build_write_policy(make_spec(allowed_paths=["docs/api/**"]), make_config())

    found, _ = staged_violations(git, ops, base, policy=policy)

    assert codes(found) == [("outside_scope", "docs/guide.md")]


def test_real_code_scope_allows_tools_and_tests_but_not_the_bridge_itself(
    repo: tuple[GitRepoEnv, GitOps, str],
) -> None:
    git, ops, base = repo
    write(git, "tools/helper.py", "x = 1\n")
    write(git, "tests/test_helper.py", "def test_x():\n    pass\n")
    write(git, "tools/agent_bridge/worker.py", "evil = True\n")
    cfg = make_config(scopes={"allowed": ["docs-only", "code"]})
    policy = build_write_policy(make_spec(scope="code"), cfg)

    found, _ = staged_violations(git, ops, base, policy=policy)

    assert codes(found) == [("protected_path", "tools/agent_bridge/worker.py")]
