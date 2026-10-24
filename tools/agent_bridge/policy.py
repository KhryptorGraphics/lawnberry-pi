"""Authoritative change policy applied to whatever an agent leaves in its worktree.

The agents also receive tool-level restrictions (see ``omp_guard.mjs`` and the Claude Code
permission flags), but those are defence in depth.  This module decides what may be committed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import BridgeConfig, LimitsConfig
from .globs import PathError, match_any, normalize_repo_path
from .schema import TaskSpec

__all__ = [
    "Change",
    "Violation",
    "WritePolicy",
    "build_write_policy",
    "evaluate_changes",
]

MODE_SYMLINK = "120000"
MODE_GITLINK = "160000"
MODE_EXECUTABLE = "100755"


@dataclass(frozen=True)
class Change:
    """One staged path as reported by ``git diff --cached --raw`` plus ``--numstat``."""

    path: str
    status: str
    old_mode: str = "000000"
    new_mode: str = "000000"
    size: int | None = None
    binary: bool = False
    added: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str
    path: str | None = None

    def render(self) -> str:
        where = f" `{self.path}`" if self.path else ""
        return f"{self.code}{where}: {self.detail}"


@dataclass(frozen=True)
class WritePolicy:
    """Every ``allow_sets`` entry must be matched (any glob within a set) and no ``deny`` glob.

    Allow globs are case-sensitive.  ``deny`` globs are compared case-insensitively: on a checkout
    whose file system ignores case, ``claude.md`` is the protected ``CLAUDE.md``.
    """

    allow_sets: tuple[tuple[str, ...], ...]
    deny: tuple[str, ...]

    def check_path(self, path: str) -> Violation | None:
        if match_any(path, self.deny, casefold=True):
            return Violation("protected_path", "path is protected and needs a human change", path)
        for allowed in self.allow_sets:
            if not match_any(path, allowed):
                return Violation("outside_scope", "path is outside the task's allowed scope", path)
        return None

    def describe(self) -> dict[str, list[str]]:
        """Machine-readable summary (also handed to the omp guard)."""
        return {
            "write_allow_sets": [list(s) for s in self.allow_sets],
            "write_deny": list(self.deny),
        }


def build_write_policy(spec: TaskSpec, cfg: BridgeConfig) -> WritePolicy:
    """Combine the configured scope, the task's own narrowing and the protected paths."""
    scope_globs = cfg.scopes.docs_paths if spec.scope == "docs-only" else cfg.scopes.code_paths
    sets: list[tuple[str, ...]] = [tuple(scope_globs)]
    if spec.allowed_paths:
        sets.append(tuple(spec.allowed_paths))
    return WritePolicy(allow_sets=tuple(sets), deny=tuple(cfg.protected_paths))


def evaluate_changes(
    changes: Sequence[Change],
    policy: WritePolicy,
    limits: LimitsConfig,
    *,
    diff_bytes: int = 0,
) -> list[Violation]:
    """Return every reason the change set must not be published (empty = acceptable)."""
    violations: list[Violation] = []
    if len(changes) > limits.max_changed_files:
        violations.append(
            Violation(
                "too_many_files",
                f"{len(changes)} files changed; the limit is {limits.max_changed_files}",
            )
        )
    if diff_bytes > limits.max_diff_bytes:
        violations.append(
            Violation(
                "diff_too_big", f"diff is {diff_bytes} bytes; the limit is {limits.max_diff_bytes}"
            )
        )
    for change in changes:
        try:
            path = normalize_repo_path(change.path)
        except PathError as exc:
            violations.append(Violation("path_invalid", str(exc), change.path[:120]))
            continue
        if path != change.path:
            violations.append(Violation("path_invalid", "path is not in canonical form", path))
            continue
        found = policy.check_path(path)
        if found is not None:
            violations.append(found)
        modes = {change.old_mode, change.new_mode}
        if MODE_SYMLINK in modes:
            violations.append(Violation("symlink", "symbolic links are not allowed", path))
        if MODE_GITLINK in modes:
            violations.append(
                Violation("submodule", "submodules/nested repositories are not allowed", path)
            )
        if change.new_mode == MODE_EXECUTABLE:
            violations.append(Violation("executable", "executable files are not allowed", path))
        if change.status == "T":
            violations.append(Violation("type_change", "file type changes are not allowed", path))
        if change.binary:
            violations.append(Violation("binary", "binary files are not allowed", path))
        if change.size is not None and change.size > limits.max_file_bytes:
            violations.append(
                Violation(
                    "too_large",
                    f"file is {change.size} bytes; the limit is {limits.max_file_bytes}",
                    path,
                )
            )
    return violations
