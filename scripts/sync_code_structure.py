#!/usr/bin/env python3
"""Keep ``docs/code_structure_overview.md`` in step with the source.

The repository directive (``.github/copilot-instructions.md``, "Automated code structure
documentation sync") asks for the overview to follow every structural code change. This script is
that sync, made deterministic:

* ``--check`` exits 0 when the overview is current, 1 when it has drifted, and 2 when the
  comparison cannot be made (no base commit, a file that cannot be parsed). An unknown comparison
  is never a pass.
* ``--fix`` rewrites the overview in place: it regenerates the interface cell of every stale row,
  inserts a row for a new file, merges duplicate rows and drops the row of a deleted file. A row's
  purpose and subsystem are kept; a new row takes them from the module docstring and the path.

Which files count: the Python files in scope (``backend/src``, ``scripts``, ``workshop``,
``firmware``, ``.specify/scripts``) that changed between the merge-base with ``--base`` and the
working tree, plus untracked ones; ``--all`` takes every file in scope. A row of a file that did not
change is only checked for signatures that no longer match its source (and fixed when drifted).

The interface cell is generated from the AST: public module functions and the public methods of
public classes, with their real argument names and defaults (annotations are left out).
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DOC = Path("docs") / "code_structure_overview.md"
SCOPE_PREFIXES = ("backend/src/", "scripts/", "workshop/", "firmware/", ".specify/scripts/")
SKIP_PARTS = ("__pycache__", "/compat_stubs/", "/tests/")
# (path prefix, section heading in the overview, subsystem label for a new row), first match wins.
SECTIONS = (
    ("backend/src/nav/", "Navigation and planning algorithms", "Navigation"),
    ("backend/src/cli/", "Backend CLI utilities", "Backend CLI"),
    ("backend/src/", "Backend services (FastAPI/domain services)", "Backend"),
    ("firmware/", "Firmware (MCU)", "Firmware"),
    ("workshop/", "Operations and maintenance scripts", "Workshop"),
    ("scripts/", "Operations and maintenance scripts", "Ops"),
    (".specify/scripts/", "Operations and maintenance scripts", "Ops"),
)
ROW = re.compile(r"^\|\s*`(?P<path>[^`]+\.py)`\s*\|")
CALL = re.compile(r"`([A-Za-z_][\w.]*)\(([^`]*)\)`")
EXIT_CLEAN, EXIT_DRIFT, EXIT_UNKNOWN = 0, 1, 2


class SyncError(Exception):
    """The comparison cannot be made (exit 2)."""


@dataclass
class Row:
    line: int  # index into the doc's lines
    path: str
    cells: list[str]  # path, purpose, subsystem, interfaces (stripped)


# ----------------------------------------------------------------------------- the source side


def _args(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    out: list[str] = []
    positional = a.posonlyargs + a.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults))
    defaults += list(a.defaults)
    for arg, default in zip(positional, defaults, strict=True):
        if arg.arg in ("self", "cls"):
            continue
        out.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    if a.vararg:
        out.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        out.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return out


def _is_property(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = {ast.unparse(d).split(".")[0] for d in fn.decorator_list}
    return "property" in names or any(
        ast.unparse(d).endswith((".setter", ".getter", ".deleter")) for d in fn.decorator_list
    )


def _esc(text: str) -> str:
    return text.replace("|", "\\|")


def _public(name: str) -> bool:
    return not name.startswith("_")


def parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise SyncError(f"cannot parse {path}: {exc}") from exc


def interfaces(tree: ast.Module) -> str:
    """The generated "Callable interfaces" cell; ``""`` when the file has no public callable."""
    funcs: list[str] = []
    classes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(node.name):
            funcs.append(f"`{node.name}({', '.join(_args(node))})`")
        elif isinstance(node, ast.ClassDef) and _public(node.name):
            members: list[str] = []
            for m in node.body:
                if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) or not _public(
                    m.name
                ):
                    continue
                if _is_property(m):
                    token = f"`{m.name}`"
                else:
                    token = f"`{m.name}({', '.join(_args(m))})`"
                if token not in members:
                    members.append(token)
            classes.append(f"Class `{node.name}`" + (": " + ", ".join(members) if members else ""))
    return _esc("; ".join(classes + ([", ".join(funcs)] if funcs else [])))


def defs_by_name(tree: ast.Module) -> dict[str, list[list[str]]]:
    found: dict[str, list[list[str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, []).append(_args(node))
    return found


def _arg_names(parts: Iterable[str]) -> list[str]:
    return [re.split(r"[=:]", p)[0].strip() for p in parts]


def _split_listed(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return [p for p in parts if p and p not in ("...", "…")]


def stale_signatures(cell: str, tree: ast.Module) -> list[str]:
    """Calls named in ``cell`` that the source does not define with those argument names."""
    defs = defs_by_name(tree)
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    problems: list[str] = []
    for name, listed in CALL.findall(cell.replace("\\|", "|")):
        short = name.split(".")[-1]
        if short in classes and short not in defs:
            continue
        if short not in defs:
            problems.append(f"`{name}` is not defined")
            continue
        wanted = _arg_names(_split_listed(listed))
        if not any(_arg_names(real) == wanted for real in defs[short]):
            if wanted:  # a bare ``name()`` or ``name(...)`` is a deliberate summary
                problems.append(f"`{name}({listed})` does not match the source")
    return problems


def purpose_of(tree: ast.Module, path: str) -> str:
    doc = ast.get_docstring(tree)
    if not doc:
        raise SyncError(f"{path}: add a module docstring (it becomes the row's purpose)")
    text = " ".join(doc.split("\n\n")[0].split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(sentences[:2])
    return _esc(out if len(out) <= 300 else out[:297].rstrip() + "...")


# ----------------------------------------------------------------------------- the doc side


def read_rows(lines: list[str]) -> list[Row]:
    rows: list[Row] = []
    for i, line in enumerate(lines):
        m = ROW.match(line)
        if not m:
            continue
        inner = line.strip()
        inner = inner[1:-1] if inner.endswith("|") else inner[1:]
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", inner)]
        cells[0] = cells[0].strip("`")
        if len(cells) < 4:
            raise SyncError(f"{DOC}:{i + 1}: row of {m['path']} has {len(cells)} cells, not 4")
        if len(cells) > 4:  # an unescaped pipe inside a cell: keep the text, merge the tail
            cells = [*cells[:3], " | ".join(cells[3:])]
        rows.append(Row(i, m["path"], cells))
    return rows


def render(row_cells: list[str]) -> str:
    path, purpose, subsystem, cell = row_cells
    return f"| `{path}` | {purpose} | {subsystem} | {cell} |"


def merge_purposes(variants: list[str]) -> str:
    kept: list[str] = []
    for text in sorted(variants, key=len, reverse=True):
        for clause in (c.strip() for c in re.split(r";\s+", text) if c.strip()):
            key = clause[:40]
            if not any(k.startswith(key) or clause.startswith(k[:40]) for k in kept):
                kept.append(clause)
    return "; ".join(kept)


def section_for(path: str) -> tuple[str, str]:
    for prefix, heading, subsystem in SECTIONS:
        if path.startswith(prefix):
            return heading, subsystem
    raise SyncError(f"{path}: no section of the overview covers this path")


def table_end(lines: list[str], heading: str) -> int:
    """Index just past the last row of the table under ``## heading`` (the first such section)."""
    start = next((i for i, ln in enumerate(lines) if ln.strip() == f"## {heading}"), None)
    if start is None:
        raise SyncError(f"{DOC}: no section '## {heading}'")
    i = start + 1
    while i < len(lines) and not lines[i].startswith("|"):
        if lines[i].startswith("## "):
            raise SyncError(f"{DOC}: section '{heading}' has no table")
        i += 1
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    return i


# ----------------------------------------------------------------------------- git and scope


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, timeout=120, check=False
    )
    if proc.returncode != 0:
        raise SyncError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def in_scope(path: str) -> bool:
    return (
        path.endswith(".py")
        and path.startswith(SCOPE_PREFIXES)
        and not any(part in "/" + path for part in SKIP_PARTS)
    )


def touched_files(root: Path, base: str) -> set[str]:
    """Files in scope that exist now and changed since the merge-base with ``base``."""
    try:
        merge_base = _git(root, "merge-base", base, "HEAD").strip()
    except SyncError as exc:
        raise SyncError(f"no merge-base with {base!r} ({exc}); fetch it or pass --base") from exc
    changed = _git(root, "diff", "--name-only", "--diff-filter=AMR", merge_base).split()
    changed += _git(root, "ls-files", "--others", "--exclude-standard").split()
    return {p for p in changed if in_scope(p) and (root / p).is_file()}


def all_files(root: Path) -> set[str]:
    tracked = (
        _git(root, "ls-files").split()
        + _git(root, "ls-files", "--others", "--exclude-standard").split()
    )
    return {p for p in tracked if in_scope(p) and (root / p).is_file()}


# ----------------------------------------------------------------------------- the sync


@dataclass
class Plan:
    """What the sync found: problems to report and the edits that would fix them."""

    problems: list[str]
    drop: set[int]
    replace: dict[int, str]
    additions: dict[str, list[str]]  # section heading -> new rows


def review_group(root: Path, path: str, group: list[Row], targets: set[str], plan: Plan) -> None:
    """Check one file's row(s): gone, duplicated, or with an interface cell that has drifted."""
    if not (root / path).is_file():
        plan.problems.append(f"{path}: the file is gone, its row is stale")
        plan.drop.update(r.line for r in group)
        return
    tree = parse(root / path)
    cells = list(group[0].cells)
    changed = False
    if len(group) > 1:
        plan.problems.append(f"{path}: {len(group)} rows, there must be one")
        cells[1] = merge_purposes([r.cells[1] for r in group])
        plan.drop.update(r.line for r in group[1:])
        changed = True
    generated = interfaces(tree)
    if path in targets and generated:
        drifted = cells[3] != generated
        if drifted:
            plan.problems.append(f"{path}: the interface list differs from the source")
    else:
        stale = stale_signatures(cells[3], tree)
        plan.problems.extend(f"{path}: {item}" for item in stale)
        drifted = bool(stale)
    if drifted:
        cells[3] = generated or "(no public callables)"
        changed = True
    if changed:
        plan.replace[group[0].line] = render(cells)


def review_new_files(root: Path, new: Iterable[str], plan: Plan, *, fix: bool) -> None:
    """A touched file with public callables and no row gets one (``--fix``) or is a drift."""
    for path in sorted(new):
        tree = parse(root / path)
        generated = interfaces(tree)
        if not generated:
            continue  # nothing public to document
        plan.problems.append(f"{path}: new file with public callables and no row")
        if fix:
            heading, subsystem = section_for(path)
            row = render([path, purpose_of(tree, path), subsystem, generated])
            plan.additions.setdefault(heading, []).append(row)


def apply_plan(lines: list[str], plan: Plan) -> list[str]:
    out = [plan.replace.get(i, line) for i, line in enumerate(lines) if i not in plan.drop]
    for heading, new_rows in plan.additions.items():
        end = table_end(out, heading)
        out[end:end] = new_rows
    return out


def sync(root: Path, targets: set[str], *, fix: bool) -> tuple[list[str], list[str]]:
    """Return ``(problems, new_doc_lines)``; the lines are only meaningful with ``fix``."""
    try:
        lines = (root / DOC).read_text(encoding="utf-8").split("\n")
    except OSError as exc:
        raise SyncError(f"cannot read {DOC}: {exc}") from exc
    by_path: dict[str, list[Row]] = {}
    for row in read_rows(lines):
        by_path.setdefault(row.path, []).append(row)
    plan = Plan([], set(), {}, {})
    for path, group in by_path.items():
        review_group(root, path, group, targets, plan)
    review_new_files(root, targets - set(by_path), plan, fix=fix)
    return plan.problems, apply_plan(lines, plan) if fix else lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report drift (exit 1), change nothing")
    mode.add_argument("--fix", action="store_true", help="rewrite the overview in place")
    ap.add_argument("--base", default="origin/main", help="ref the changed files are taken from")
    ap.add_argument("--all", action="store_true", help="every file in scope, not only changed ones")
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        targets = all_files(root) if args.all else touched_files(root, args.base)
        problems, new_lines = sync(root, targets, fix=args.fix)
    except SyncError as exc:
        print(f"sync_code_structure: cannot compare: {exc}", file=sys.stderr)
        return EXIT_UNKNOWN
    for p in problems:
        print(("fixed: " if args.fix else "drift: ") + p)
    if args.fix:
        if problems:
            (root / DOC).write_text("\n".join(new_lines), encoding="utf-8")
        print(f"sync_code_structure: {len(problems)} change(s) written to {DOC}")
        return EXIT_CLEAN
    if problems:
        print(f"sync_code_structure: {len(problems)} drift(s); run --fix", file=sys.stderr)
        return EXIT_DRIFT
    print("sync_code_structure: current")
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
