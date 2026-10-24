"""Tests for ``tools/agent_bridge/globs.py`` and its JavaScript twin in ``omp_guard.mjs``."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.agent_bridge.agents.omp import GUARD_PATH
from tools.agent_bridge.config import DEFAULT_PROTECTED_PATHS
from tools.agent_bridge.globs import (
    PathError,
    glob_to_regex,
    match_any,
    normalize_glob,
    normalize_repo_path,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
VECTORS: list[dict[str, Any]] = json.loads((FIXTURES / "glob_vectors.json").read_text("utf-8"))
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def vector_id(vector: dict[str, Any]) -> str:
    return f"{vector['pattern']}~{vector['path']}->{vector['match']}"


# --------------------------------------------------------------------------- shared vectors
def test_vector_file_is_well_formed_and_covers_both_outcomes() -> None:
    assert len(VECTORS) >= 50
    for vector in VECTORS:
        assert set(vector) == {"pattern", "path", "match"}
        assert isinstance(vector["pattern"], str) and isinstance(vector["path"], str)
        assert isinstance(vector["match"], bool)
    assert {v["match"] for v in VECTORS} == {True, False}


@pytest.mark.parametrize("vector", VECTORS, ids=vector_id)
def test_python_matches_the_shared_vector(vector: dict[str, Any]) -> None:
    assert bool(glob_to_regex(vector["pattern"]).match(vector["path"])) is vector["match"]
    assert match_any(vector["path"], [vector["pattern"]]) is vector["match"]


# --------------------------------------------------------------------------- node parity
NODE_DRIVER = """
import { pathToFileURL } from "node:url";
import * as fs from "node:fs";

const guard = await import(pathToFileURL(process.argv[2]).href);
const request = JSON.parse(fs.readFileSync(0, "utf8"));
let out;
if (request.pairs) {
  out = request.pairs.map(([pattern, path]) => guard.globToRegExp(pattern).test(path));
} else {
  out = request.patterns.map((pattern) => {
    const re = guard.globToRegExp(pattern);
    const hits = [];
    request.paths.forEach((path, index) => { if (re.test(path)) hits.push(index); });
    return hits;
  });
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("globdriver") / "driver.mjs"
    path.write_text(NODE_DRIVER, encoding="utf-8")
    return path


def run_node(driver: Path, request: dict[str, Any]) -> Any:
    assert NODE is not None
    done = subprocess.run(
        [NODE, str(driver), str(GUARD_PATH)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_guard_source_exists_where_the_omp_backend_loads_it() -> None:
    assert GUARD_PATH.is_file()
    assert GUARD_PATH.name == "omp_guard.mjs"


@pytest.fixture(scope="module")
def js_vector_results(node_driver: Path) -> list[bool]:
    if NODE is None:
        pytest.skip("node is not installed")
    return run_node(node_driver, {"pairs": [[v["pattern"], v["path"]] for v in VECTORS]})


@needs_node
@pytest.mark.parametrize(
    ("index", "vector"), list(enumerate(VECTORS)), ids=[vector_id(v) for v in VECTORS]
)
def test_javascript_matches_the_shared_vector(
    index: int, vector: dict[str, Any], js_vector_results: list[bool]
) -> None:
    assert js_vector_results[index] is vector["match"]


def _instantiate(pattern: str, rng: random.Random) -> str:
    """A path that (usually) matches ``pattern`` by expanding each wildcard."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(rng.choice(["", "a/", "a/b/"]))
            i += 3
        elif pattern.startswith("/**", i):
            out.append(rng.choice(["", "/a", "/a/b"]))
            i += 3
        elif pattern.startswith("**", i):
            out.append(rng.choice(["", "a", "a/b"]))
            i += 2
        elif pattern[i] == "*":
            out.append(rng.choice(["", "a", "ab"]))
            i += 1
        elif pattern[i] == "?":
            out.append(rng.choice("ab.-"))
            i += 1
        else:
            out.append(pattern[i])
            i += 1
    return "".join(out)


@needs_node
def test_javascript_and_python_agree_on_random_patterns_and_paths(node_driver: Path) -> None:
    rng = random.Random(1024)
    tokens = ["a", "b", "ab", ".", "/", "*", "**", "?", "**/", "/**", ".md", "-", "+", "(", ")"]
    tokens += ["[", "]", "{", "}", "$", "^", "|", "x.y", "a/", "/b"]
    segments = ["a", "b", "ab", ".", "x.y", ".md", "a.md", "-", "$"]
    pairs: list[tuple[str, str]] = []
    for _ in range(500):
        pattern = "".join(rng.choice(tokens) for _ in range(rng.randint(1, 7)))
        for _ in range(12):
            if rng.random() < 0.5:
                path = _instantiate(pattern, rng)
            else:
                path = "/".join(rng.choice(segments) for _ in range(rng.randint(1, 4)))
            pairs.append((pattern, path))

    js = run_node(node_driver, {"pairs": [list(pair) for pair in pairs]})
    py = [bool(glob_to_regex(pattern).match(path)) for pattern, path in pairs]

    mismatches = [(pair, a, b) for pair, a, b in zip(pairs, py, js, strict=True) if a != b]
    assert not mismatches, mismatches[:5]
    assert sum(py) >= 400 and py.count(False) >= 400  # both outcomes are well exercised


def _tracked_files() -> list[str]:
    git = shutil.which("git")
    if git is None:
        return []
    done = subprocess.run(
        [git, "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=False, timeout=60
    )
    if done.returncode != 0:
        return []
    names = done.stdout.decode("utf-8", errors="replace").split("\0")
    return [name for name in names if name and "\n" not in name]


SYNTHETIC_PATHS = [
    ".env",
    "a/b/.env",
    "a/.env.local",
    "docs/.env",
    "docs/sub/key.pem",
    "id_rsa",
    "home/.ssh/id_rsa.pub",
    "config/secrets.json",
    "config/secrets.prod.json",
    "config/sub/secrets.json",
    "backend/src/safety",
    "backend/src/safety/estop.py",
    "backend/src/safetyx/estop.py",
    "backend/src/core/x/y/z.py",
    "systemd",
    "systemd/a.service",
    "tools/agent_bridge/omp_guard.mjs",
    "tools/agent_bridge_extra/x.py",
    "docs/guide.md",
    "README.md",
    ".github/workflows/ci.yml",
    ".git/config",
    ".gitmodules",
    "x/.gitmodules",
]


@needs_node
def test_javascript_and_python_agree_on_the_default_protected_paths(node_driver: Path) -> None:
    patterns = [*DEFAULT_PROTECTED_PATHS, "docs/**", "*.md", ".git", ".git/**"]
    paths = sorted({*_tracked_files(), *SYNTHETIC_PATHS})

    js = run_node(node_driver, {"patterns": patterns, "paths": paths})

    for pattern, js_hits in zip(patterns, js, strict=True):
        py_hits = [i for i, path in enumerate(paths) if glob_to_regex(pattern).match(path)]
        assert py_hits == js_hits, pattern


# --------------------------------------------------------------------------- normalize_repo_path
ACCEPTED_PATHS = [
    ("docs/a.md", "docs/a.md"),
    ("./docs/a.md", "docs/a.md"),
    ("docs//a.md", "docs/a.md"),
    ("docs/./a.md", "docs/a.md"),
    ("docs/a.md/", "docs/a.md"),
    ("docs/", "docs"),
    ("  docs/a.md  ", "docs/a.md"),
    ("\tdocs/a.md\n", "docs/a.md"),
    ("a/b/../c", "a/c"),
    ("a/./b/../../c", "c"),
    ("docs/a:b.md", "docs/a:b.md"),
    ("a b/c d.md", "a b/c d.md"),
    ("caf\u00e9/\u00e4.md", "caf\u00e9/\u00e4.md"),
    (".git/config", ".git/config"),
    ("..a/b..", "..a/b.."),
    ("a/..b/c", "a/..b/c"),
    ("...", "..."),
    ("a" * 400, "a" * 400),
]
REJECTED_PATHS = [
    ("", "empty"),
    ("   ", "empty"),
    ("\n", "empty"),
    ("/etc/passwd", "repository-relative"),
    ("//server/share", "repository-relative"),
    ("~", "repository-relative"),
    ("~/x", "repository-relative"),
    ("~root/x", "repository-relative"),
    ("C:/x", "repository-relative"),
    ("c:x", "repository-relative"),
    ("a:b", "repository-relative"),
    ("file:///etc/passwd", "repository-relative"),
    ("http://example.com/x", "repository-relative"),
    ("xd://web_fetch", "repository-relative"),
    ("ssh://host/x", "repository-relative"),
    (".", "repository root"),
    ("./", "repository root"),
    ("./.", "repository root"),
    ("a/..", "repository root"),
    ("a/b/../..", "repository root"),
    ("..", "escapes"),
    ("../x", "escapes"),
    ("a/../..", "escapes"),
    ("./../a", "escapes"),
    ("a/../../b", "escapes"),
    ("a\\b", "backslashes"),
    ("..\\x", "backslashes"),
    ("a\x00b", "control characters"),
    ("a\nb", "control characters"),
    ("a\tb", "control characters"),
    ("a\rb", "control characters"),
    ("a\x1bb", "control characters"),
    ("a\x7fb", "control characters"),
    ("a" * 401, "too long"),
    ("a/" * 201, "too long"),
]


@pytest.mark.parametrize(("raw", "expected"), ACCEPTED_PATHS, ids=lambda v: repr(v)[:40])
def test_normalize_repo_path_accepts_and_canonicalises(raw: str, expected: str) -> None:
    assert normalize_repo_path(raw) == expected
    assert normalize_repo_path(expected) == expected  # canonical form is a fixed point


@pytest.mark.parametrize(("raw", "message"), REJECTED_PATHS, ids=lambda v: repr(v)[:40])
def test_normalize_repo_path_rejects(raw: str, message: str) -> None:
    with pytest.raises(PathError, match=message):
        normalize_repo_path(raw)


@pytest.mark.parametrize("raw", [None, 5, b"docs/a.md", ["docs"], 1.5])
def test_normalize_repo_path_rejects_non_strings(raw: Any) -> None:
    with pytest.raises(PathError, match="must be a string"):
        normalize_repo_path(raw)


def test_path_error_is_a_value_error() -> None:
    assert issubclass(PathError, ValueError)


# --------------------------------------------------------------------------- normalize_glob
ACCEPTED_GLOBS = [
    ("docs/**", "docs/**"),
    ("*.md", "*.md"),
    ("**/.env", "**/.env"),
    ("**/.env.*", "**/.env.*"),
    ("config/secrets*.json", "config/secrets*.json"),
    ("**", "**"),
    ("?", "?"),
    ("a/**/b", "a/**/b"),
    ("./docs/**", "docs/**"),
    ("  docs/**  ", "docs/**"),
    ("a:b/**", "a:b/**"),
    ("docs/a..b", "docs/a..b"),
    ("..a/**", "..a/**"),
    ("docs/", "docs/**"),
    ("scripts/", "scripts/**"),
    ("a//b", "a/b"),
    ("a/./b/", "a/b/**"),
    (".//a", "a"),
    ("docs//**", "docs/**"),
    ("a" * 400, "a" * 400),
]
REJECTED_GLOBS = [
    ("", "empty"),
    ("   ", "empty"),
    ("/abs/**", "repository-relative"),
    ("~/x/**", "repository-relative"),
    ("~", "repository-relative"),
    ("..", "escapes"),
    ("../x", "escapes"),
    ("a/../b", "escapes"),
    ("a/..", "escapes"),
    ("**/../x", "escapes"),
    ("a\\b", "backslashes"),
    ("a\x00b", "control characters"),
    ("a\nb/**", "control characters"),
    ("a\tb", "control characters"),
    ("a\x7f", "control characters"),
    (".", "repository root"),
    ("./", "repository root"),
    ("./.", "repository root"),
    ("//", "repository-relative"),
    ("a" * 401, "too long"),
]


@pytest.mark.parametrize(("raw", "expected"), ACCEPTED_GLOBS, ids=lambda v: repr(v)[:40])
def test_normalize_glob_accepts_and_canonicalises(raw: str, expected: str) -> None:
    assert normalize_glob(raw) == expected


@pytest.mark.parametrize(("raw", "message"), REJECTED_GLOBS, ids=lambda v: repr(v)[:40])
def test_normalize_glob_rejects(raw: str, message: str) -> None:
    with pytest.raises(PathError, match=message):
        normalize_glob(raw)


@pytest.mark.parametrize("raw", [None, 5, b"docs/**", ["docs"]])
def test_normalize_glob_rejects_non_strings(raw: Any) -> None:
    with pytest.raises(PathError, match="must be a string"):
        normalize_glob(raw)


def test_canonical_globs_match_canonical_paths_where_the_raw_form_never_could() -> None:
    """``scripts/`` and ``a//b`` as written can never equal a normalised path; after
    canonicalisation a protected glob written that way protects what it looks like it protects."""
    assert match_any("scripts/deploy.sh", [normalize_glob("scripts/")])
    assert match_any("scripts", [normalize_glob("scripts/")])
    assert match_any("a/b", [normalize_glob("a//b")])
    assert match_any("a/b/c", [normalize_glob("./a/./b/")])
    assert not match_any("scripts-extra/x", [normalize_glob("scripts/")])


# --------------------------------------------------------------------------- match_any / regex
def test_match_any_without_patterns_matches_nothing() -> None:
    assert match_any("docs/a.md", []) is False
    assert match_any("docs/a.md", ()) is False


def test_match_any_accepts_any_iterable_and_any_position() -> None:
    assert match_any("docs/a.md", ["x", "y", "docs/*"]) is True
    assert match_any("docs/a.md", ["docs/*", "x", "y"]) is True
    assert match_any("docs/a.md", iter(["x", "*.md", "y"])) is False
    assert match_any("docs/a.md", (p for p in ["nope", "**/*.md"])) is True


def test_match_any_is_anchored_at_both_ends() -> None:
    assert match_any("xdocs/a.md", ["docs/**"]) is False
    assert match_any("docs/a.md.bak", ["docs/*.md"]) is False
    assert match_any("docs/a.md\n", ["docs/*.md"]) is False
    assert match_any("docs/a.md", ["*.md"]) is False  # '*' never crosses '/'


@pytest.mark.parametrize("literal", ["a.b", "a+b", "a(b)", "a[b]", "a{b,c}", "a^b", "a$b", "a|b"])
def test_regex_metacharacters_are_literal(literal: str) -> None:
    assert match_any(literal, [literal]) is True
    assert match_any(literal.replace(literal[1], "x"), [literal]) is False


def test_glob_to_regex_is_cached() -> None:
    assert glob_to_regex("docs/**") is glob_to_regex("docs/**")


def test_double_star_forms() -> None:
    assert match_any("docs", ["docs/**"]) is True
    assert match_any("docs/a/b/c", ["docs/**"]) is True
    assert match_any("a/b", ["a/**/b"]) is True
    assert match_any("a/x/y/b", ["a/**/b"]) is True
    assert match_any("ab", ["a/**/b"]) is False
    assert match_any(".env", ["**/.env"]) is True
    assert match_any("x/y/.env", ["**/.env"]) is True
    assert match_any("x/.env.local", ["**/.env"]) is False
