"""Tests for ``tools/agent_bridge/omp_guard.mjs``, the omp tool-call guard (node ESM).

The guard is driven through a tiny node script that imports the real module, so every assertion
exercises the file the omp backend passes to ``--trusted-extension``.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tools.agent_bridge.agents.omp import GUARD_PATH, GUARD_PREFIX, MAX_WRITE_BYTES, READ_DENY
from tools.agent_bridge.config import BridgeConfig
from tools.agent_bridge.policy import build_write_policy
from tools.agent_bridge.schema import TaskSpec

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DRIVER = """
import * as fs from "node:fs";
import { pathToFileURL } from "node:url";

const guard = await import(pathToFileURL(process.argv[2]).href);
const request = JSON.parse(fs.readFileSync(0, "utf8"));

function decide(req) {
  try {
    return { ok: true, value: guard.decide(req.policy, req.event, req.cwd ?? undefined) };
  } catch (error) {
    return { ok: false, error: String(error) };
  }
}

async function handler(req) {
  const handlers = {};
  if (req.policy_file === null) delete process.env.LAWNBERRY_BRIDGE_GUARD_POLICY;
  else process.env.LAWNBERRY_BRIDGE_GUARD_POLICY = req.policy_file;
  guard.default({ on(event, fn) { handlers[event] = fn; } });
  const results = [];
  const total = req.repeat ?? 1;
  for (let round = 0; round < total; round++) {
    for (const call of req.calls) {
      const returned = await handlers.tool_call(call.event, call.ctx);
      if (round === 0) {
        results.push(
          returned === undefined ? { kind: "undefined" } : { kind: "value", value: returned },
        );
      }
    }
  }
  return { events: Object.keys(handlers), results };
}

const batch = Array.isArray(request);
const out = [];
for (const req of batch ? request : [request]) {
  out.push(req.op === "handler" ? await handler(req) : decide(req));
}
process.stdout.write(JSON.stringify(batch ? out : out[0]));
"""

TOOLS = ["read", "grep", "glob", "find", "edit", "write"]
REASON_PREFIX = f"{GUARD_PREFIX} "


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str | None
    paths: list[str]


class Guard:
    """Runs ``decide`` / the registered handler of the real guard inside node."""

    def __init__(self, driver: Path, world: World) -> None:
        self.driver = driver
        self.world = world

    def run(self, payload: Any) -> Any:
        assert NODE is not None
        done = subprocess.run(
            [NODE, str(self.driver), str(GUARD_PATH)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    def decide_raw(
        self, policy: Any, tool: str, tool_input: Any, *, cwd: str | None = None
    ) -> dict[str, Any]:
        event = {"toolName": tool, "input": self.world.fill(tool_input)}
        return self.run({"op": "decide", "policy": policy, "event": event, "cwd": cwd})

    def decide(
        self,
        tool: str,
        tool_input: Any,
        *,
        policy: Any = None,
        cwd: str | None = None,
    ) -> Decision:
        result = self.decide_raw(policy or self.world.policy(), tool, tool_input, cwd=cwd)
        assert result["ok"], result
        value = result["value"]
        return Decision(value["allow"], value.get("reason"), value["paths"])

    def handler(
        self,
        calls: list[dict[str, Any]],
        *,
        policy_file: Path | None,
        repeat: int = 1,
    ) -> dict[str, Any]:
        request = {
            "op": "handler",
            "policy_file": None if policy_file is None else str(policy_file),
            "calls": [
                {"event": self.world.fill(call["event"]), "ctx": call.get("ctx")} for call in calls
            ],
            "repeat": repeat,
        }
        return self.run(request)


class World:
    """A task worktree with an outside neighbour and a few awkward symlinks."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "work"
        self.outside = base / "outside"
        self.alias = base / "root_alias"
        files = {
            "README.md": "readme\n",
            ".gitignore": "x\n",
            "docs/guide.md": "guide\n",
            "docs/api/ref.md": "ref\n",
            "docs/sub/deep.md": "deep\n",
            "src/code.py": "x = 1\n",
            "backend/src/safety/estop.py": "ESTOP = True\n",
            ".git/config": "[core]\n",
            ".env": "SECRET=1\n",
            "sub/.env": "SECRET=2\n",
        }
        for rel, content in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.outside.mkdir()
        (self.outside / "secret.txt").write_text("outside secret\n", encoding="utf-8")
        links = {
            "link_out": "../outside",
            "docs/escape.md": "../../outside/secret.txt",
            "docs/readme_link.md": "../README.md",
            "docs/src_link": "../src",
            "docs/safety_link": "../backend/src/safety",
            "docs/env_link": "../.env",
        }
        for rel, target in links.items():
            os.symlink(target, self.root / rel)
        os.symlink(self.root, self.alias)

    def fill(self, value: Any) -> Any:
        if isinstance(value, str):
            return (
                value.replace("{root}", str(self.root))
                .replace("{outside}", str(self.outside))
                .replace("{alias}", str(self.alias))
            )
        if isinstance(value, list):
            return [self.fill(item) for item in value]
        if isinstance(value, dict):
            return {key: self.fill(item) for key, item in value.items()}
        return value

    def policy(self, **overrides: Any) -> dict[str, Any]:
        policy: dict[str, Any] = {
            "version": 1,
            "root": str(self.root),
            "tools": list(TOOLS),
            "write_allow_sets": [["docs/**", "*.md"]],
            "write_deny": ["backend/src/safety/**", ".github/**", "**/.env"],
            "read_deny": [".git", ".git/**", ".env", "**/.env", "**/.env.*"],
            "bash_exact": [],
            "max_write_bytes": 100,
        }
        policy.update(overrides)
        return policy


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("guarddriver") / "driver.mjs"
    path.write_text(DRIVER, encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path / "world")


@pytest.fixture
def guard(driver: Path, world: World) -> Guard:
    return Guard(driver, world)


def write_policy_file(world: World, policy: Any) -> Path:
    path = world.base / "policy.json"
    path.write_text(policy if isinstance(policy, str) else json.dumps(policy), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- allowed calls
ALLOWED = [
    ("write", {"path": "docs/new.md", "content": "x"}, ["docs/new.md"]),
    ("write", {"path": "./docs/new.md"}, ["docs/new.md"]),
    ("write", {"path": "docs//new.md"}, ["docs/new.md"]),
    ("write", {"path": "{root}/docs/new.md"}, ["docs/new.md"]),
    ("write", {"path": "README.md"}, ["README.md"]),
    ("write", {"path": "docs/api/ref.md"}, ["docs/api/ref.md"]),
    ("write", {"path": "docs/new.md:3-5"}, ["docs/new.md:3-5", "docs/new.md"]),
    ("write", {"path": "docs/a.md;docs/b.md"}, ["docs/a.md", "docs/b.md"]),
    ("write", {"path": ";docs/a.md;"}, ["docs/a.md"]),
    ("write", {"paths": ["docs/a.md", "docs/b.md"]}, ["docs/a.md", "docs/b.md"]),
    ("write", {"path": ["docs/a.md"]}, ["docs/a.md"]),
    ("write", {"file_path": "docs/a.md"}, ["docs/a.md"]),
    ("write", {"filePath": "docs/a.md"}, ["docs/a.md"]),
    ("write", {"file": "docs/a.md"}, ["docs/a.md"]),
    ("write", {"path": "docs/a.md", "content": "../../etc/passwd", "note": "/etc/shadow"}, None),
    ("edit", {"path": "docs/guide.md", "edits": [{"old": "a", "new": "b"}]}, ["docs/guide.md"]),
    (
        "edit",
        {"edits": [{"path": "docs/a.md", "to": "docs/b.md"}]},
        ["docs/a.md", "docs/b.md"],
    ),
    ("edit", {"path": "docs/guide.md", "moveTo": "docs/renamed.md"}, None),
    ("read", {"path": "docs/guide.md"}, ["docs/guide.md"]),
    ("read", {"path": "docs/guide.md:10-20"}, ["docs/guide.md:10-20", "docs/guide.md"]),
    ("read", {"path": "docs/guide.md:raw:2-4"}, None),
    ("read", {"path": "."}, ["."]),
    ("read", {"path": "docs/readme_link.md"}, ["docs/readme_link.md"]),
    ("read", {"path": ".gitignore"}, [".gitignore"]),
    ("read", {"path": ".github/workflows/ci.yml"}, [".github/workflows/ci.yml"]),
    ("read", {"path": "docs/guide.md;README.md"}, ["docs/guide.md", "README.md"]),
    ("read", {"pattern": "/etc/passwd", "path": "docs/guide.md"}, ["docs/guide.md"]),
    ("read", {}, []),
    ("glob", {"pattern": "**/*.py"}, ["**/*.py"]),
    ("glob", {"pattern": "docs/**", "path": "."}, ["docs/**", "."]),
    ("find", {"pattern": "*.md"}, ["*.md"]),
    ("grep", {"pattern": "../../etc/passwd"}, []),
    ("grep", {"pattern": "foo|bar\\.baz", "path": "docs"}, ["docs"]),
]


@pytest.mark.parametrize(
    ("tool", "tool_input", "paths"),
    ALLOWED,
    ids=[f"{t}:{json.dumps(i)[:50]}" for t, i, _ in ALLOWED],
)
def test_legitimate_calls_are_allowed(
    guard: Guard, tool: str, tool_input: dict[str, Any], paths: list[str] | None
) -> None:
    decision = guard.decide(tool, tool_input)

    assert decision.allow is True, decision.reason
    assert decision.reason is None
    if paths is not None:
        assert decision.paths == paths


@pytest.mark.parametrize("tool", ["yield", "think"])
def test_yield_and_think_are_always_allowed(guard: Guard, tool: str) -> None:
    for tool_input in ({}, {"path": "/etc/passwd"}, {"command": "rm -rf /"}, None, "text"):
        policy = guard.world.policy(tools=[])
        assert guard.decide(tool, tool_input, policy=policy) == Decision(True, None, [])


# --------------------------------------------------------------------------- blocked calls
BLOCKED = [
    # outside the worktree, however it is spelled
    ("write", {"path": "/etc/passwd"}, "escapes the task worktree"),
    ("write", {"path": "{outside}/x.md"}, "escapes the task worktree"),
    ("write", {"path": "../outside/x.md"}, "escapes the task worktree"),
    ("write", {"path": "docs/../../outside/x.md"}, "escapes the task worktree"),
    ("write", {"path": ".."}, "escapes the task worktree"),
    ("read", {"path": "/etc/hostname"}, "escapes the task worktree"),
    ("read", {"path": "{alias}/README.md"}, "escapes the task worktree"),
    ("write", {"path": "~"}, "home-relative"),
    ("write", {"path": "~/x.md"}, "home-relative"),
    ("read", {"path": "~root/.ssh/id_rsa"}, "home-relative"),
    ("write", {"path": "xd://web_fetch"}, "URL or device"),
    ("write", {"path": "XD://web_fetch"}, "URL or device"),
    ("read", {"path": "https://example.com/x"}, "URL or device"),
    ("read", {"path": "file:///etc/passwd"}, "URL or device"),
    ("read", {"path": "ssh://host/x"}, "URL or device"),
    ("read", {"path": "mcp://server/resource"}, "URL or device"),
    ("write", {"path": "docs/a\u0000.md"}, "NUL"),
    ("read", {"path": "docs/a\u0000/../../x"}, "NUL"),
    ("write", {"path": ""}, "empty path argument"),
    ("write", {"path": ";"}, "empty path argument"),
    ("read", {"path": ";;"}, "empty path argument"),
    # writes without a path fail closed
    ("write", {"content": "x"}, "without a recognisable path"),
    ("write", {}, "without a recognisable path"),
    ("write", {"path": 5, "content": "x"}, "without a recognisable path"),
    ("write", {"path": None}, "without a recognisable path"),
    ("write", {"path": {"nested": "docs/a.md"}}, "without a recognisable path"),
    ("edit", {"edits": [{"old": "a", "new": "b"}]}, "without a recognisable path"),
    ("write", {"path": "."}, "cannot write the repository root"),
    ("write", {"path": "docs/.."}, "parent-directory segments"),
    # symlinks
    ("read", {"path": "link_out/secret.txt"}, "resolves outside"),
    ("read", {"path": "link_out"}, "resolves outside"),
    ("read", {"path": "docs/escape.md"}, "resolves outside"),
    ("write", {"path": "link_out/new.md"}, "resolves outside"),
    ("write", {"path": "docs/escape.md"}, "resolves outside"),
    ("write", {"path": "docs/readme_link.md"}, "symbolic link"),
    ("edit", {"path": "docs/readme_link.md"}, "symbolic link"),
    # selector tricks
    ("read", {"path": "foo:bar/../../../etc/passwd"}, "escapes the task worktree"),
    ("write", {"path": "docs/a.md:../../../../etc/passwd"}, "escapes the task worktree"),
    ("write", {"path": "/etc/passwd:1-5"}, "escapes the task worktree"),
    ("read", {"path": "link_out/secret.txt:1-2"}, "resolves outside"),
    ("read", {"path": ".env:1-2"}, "not readable"),
    ("write", {"path": "backend/src/safety/estop.py:1-3"}, "protected"),
    # several roots in one argument: every one is checked
    ("read", {"path": "docs/guide.md;/etc/passwd"}, "escapes the task worktree"),
    ("read", {"path": "docs/guide.md;.env"}, "not readable"),
    ("write", {"path": "docs/new.md;src/code.py"}, "outside the task's allowed scope"),
    ("write", {"paths": ["docs/a.md", "../x"]}, "escapes the task worktree"),
    # nested and alternative argument names
    ("edit", {"path": "docs/guide.md", "moveTo": "/etc/x"}, "escapes the task worktree"),
    ("edit", {"path": "docs/guide.md", "moveTo": "src/x.py"}, "outside the task's allowed scope"),
    ("edit", {"path": "docs/guide.md", "to": "backend/src/safety/x.py"}, "protected"),
    ("edit", {"edits": [{"path": "docs/a.md"}, {"path": "../x"}]}, "escapes the task worktree"),
    ("edit", {"from": "docs/a.md", "dest": "src/b.py"}, "outside the task's allowed scope"),
    ("edit", {"source": "docs/a.md", "destination": "src/b.py"}, "outside the task's allowed"),
    ("write", {"file_path": "/etc/passwd"}, "escapes the task worktree"),
    ("write", {"filePath": "../x"}, "escapes the task worktree"),
    ("write", {"file": "~/x"}, "home-relative"),
    ("write", {"path": "docs/a.md", "cwd": "/"}, "escapes the task worktree"),
    ("read", {"path": "docs/guide.md", "dir": "/etc"}, "escapes the task worktree"),
    ("read", {"directory": ".."}, "escapes the task worktree"),
    ("read", {"root": "/"}, "escapes the task worktree"),
    ("write", {"a": {"b": [{"c": {"path": "/etc/passwd"}}]}}, "escapes the task worktree"),
    # read_deny
    ("read", {"path": ".git"}, "not readable"),
    ("read", {"path": ".git/config"}, "not readable"),
    ("read", {"path": ".env"}, "not readable"),
    ("read", {"path": "sub/.env"}, "not readable"),
    ("read", {"path": "docs/.env.local"}, "not readable"),
    ("grep", {"pattern": "x", "path": ".git"}, "not readable"),
    ("glob", {"pattern": ".git/**"}, "not readable"),
    ("write", {"path": ".env"}, "not readable"),
    ("write", {"path": ".git/hooks/pre-commit"}, "not readable"),
    # write scope and protected paths
    ("write", {"path": "src/code.py"}, "outside the task's allowed scope"),
    ("write", {"path": "notes.txt"}, "outside the task's allowed scope"),
    ("write", {"path": "sub/README.md"}, "outside the task's allowed scope"),
    ("write", {"path": "backend/src/safety/estop.py"}, "protected"),
    ("write", {"path": "docs/../backend/src/safety/estop.py"}, "parent-directory segments"),
    ("write", {"path": ".github/workflows/ci.yml"}, "protected"),
    ("edit", {"path": "backend/src/safety/estop.py"}, "protected"),
    # path-bearing arguments of the search tools
    ("glob", {"pattern": "../../**"}, "escapes the task worktree"),
    ("glob", {"pattern": "/etc/*"}, "escapes the task worktree"),
    ("find", {"pattern": "~/*"}, "home-relative"),
    ("find", {"pattern": "docs/**", "path": "/tmp"}, "escapes the task worktree"),
    ("grep", {"pattern": "x", "path": "../outside"}, "escapes the task worktree"),
    ("grep", {"pattern": "x", "path": "link_out"}, "resolves outside"),
    # tools that are not enabled
    ("bash", {"command": "ls"}, "tool 'bash' is not enabled"),
    ("browser", {"url": "https://example.com"}, "tool 'browser' is not enabled"),
    ("web_fetch", {}, "tool 'web_fetch' is not enabled"),
    ("Write", {"path": "docs/a.md"}, "tool 'Write' is not enabled"),
    ("write ", {"path": "docs/a.md"}, "is not enabled"),
    ("", {"path": "docs/a.md"}, "tool '' is not enabled"),
]


@pytest.mark.parametrize(
    ("tool", "tool_input", "reason"),
    BLOCKED,
    ids=[f"{t}:{json.dumps(i)[:48]}" for t, i, _ in BLOCKED],
)
def test_dangerous_calls_are_blocked(guard: Guard, tool: str, tool_input: Any, reason: str) -> None:
    decision = guard.decide(tool, tool_input)

    assert decision.allow is False
    assert decision.reason is not None and reason in decision.reason, decision.reason


def test_a_missing_tool_name_or_event_is_blocked(guard: Guard) -> None:
    policy = guard.world.policy()
    for event in ({"input": {"path": "docs/a.md"}}, {}, None, {"toolName": None}):
        result = guard.run({"op": "decide", "policy": policy, "event": event})
        assert result["ok"] and result["value"]["allow"] is False
        assert "is not enabled" in result["value"]["reason"]


def test_a_non_object_input_counts_as_no_input(guard: Guard) -> None:
    policy = guard.world.policy()
    result = guard.run(
        {"op": "decide", "policy": policy, "event": {"toolName": "write", "input": "docs/a.md"}}
    )
    assert result["value"]["allow"] is False
    assert "without a recognisable path" in result["value"]["reason"]


def test_tools_outside_the_enabled_list_are_blocked_even_if_harmless(guard: Guard) -> None:
    policy = guard.world.policy(tools=["read"])
    assert guard.decide("read", {"path": "README.md"}, policy=policy).allow is True
    blocked = guard.decide("write", {"path": "docs/a.md"}, policy=policy)
    assert blocked.allow is False and "tool 'write' is not enabled" in (blocked.reason or "")


# ----------------------------------------------------------------- symlinks inside the worktree
def test_writing_or_reading_through_a_directory_symlink_cannot_reach_protected_paths(
    guard: Guard,
) -> None:
    """``docs/safety_link`` points at ``backend/src/safety``; the real target is what counts."""
    write = guard.decide("write", {"path": "docs/safety_link/estop.py", "content": "x"})
    assert write.allow is False, "write reached a protected file through a directory symlink"


def test_writing_through_a_directory_symlink_cannot_leave_the_write_scope(guard: Guard) -> None:
    """``docs/src_link`` points at ``src``, which the docs-only scope does not cover."""
    write = guard.decide("write", {"path": "docs/src_link/code.py", "content": "x"})
    assert write.allow is False, "write reached an out-of-scope file through a directory symlink"


def test_reading_a_read_denied_file_through_a_symlink_is_blocked(guard: Guard) -> None:
    """``docs/env_link`` points at the read-denied ``.env`` file."""
    assert guard.decide("read", {"path": "docs/env_link"}).allow is False


def test_the_policy_root_may_be_given_through_a_symlink(guard: Guard, world: World) -> None:
    policy = world.policy(root=str(world.alias))

    assert guard.decide("write", {"path": "docs/new.md"}, policy=policy).allow is True
    assert guard.decide("read", {"path": "docs/guide.md"}, policy=policy).allow is True
    escape = guard.decide("write", {"path": "../outside/x.md"}, policy=policy)
    assert escape.allow is False and "escapes" in (escape.reason or "")
    linked = guard.decide("read", {"path": "link_out/secret.txt"}, policy=policy)
    assert linked.allow is False and "resolves outside" in (linked.reason or "")


# ------------------------------------------------------------ paths omp rewrites before opening
# omp rewrites a path argument before it opens the file (omp tools/path-utils.ts, expandPath and
# splitDelimitedPathEntry): it strips a leading "@" or ":", drops surrounding quotes, splits one
# string into several paths on whitespace, commas and semicolons, expands "~" and braces, maps
# Unicode spaces to spaces and turns file:// into a path; an absolute path keeps its ".." for the
# kernel to follow through symlinks.  A guard that read the text literally approved a different
# file than omp opened: ``read @/etc/hostname`` returned the real /etc/hostname.
OMP_REWRITES = [
    ("at-absolute", "@/etc/passwd"),
    ("at-home", "@~/.omp/agent/agent.db"),
    ("at-proc", "@/proc/self/environ"),
    ("at-parent-escape", "@../../x.md"),
    ("at-relative", "@docs/guide.md"),
    ("at-url", "@https://example.com/x"),
    ("colon-absolute", ":/etc/passwd"),
    ("colon-parent", ":../x.md"),
    ("double-quoted", '"/etc/hostname"'),
    ("quoted-padding", '" /etc/hostname"'),
    ("single-quoted", "'/etc/hostname'"),
    ("whitespace-list", "docs/guide.md /etc/hostname"),
    ("comma-list", "docs/guide.md,/etc/hostname"),
    ("newline-list", "docs/guide.md\n/etc/hostname"),
    ("tab-list", "docs/guide.md\t/etc/hostname"),
    ("brace-expansion", "{docs,..}/guide.md"),
    ("brace-synthesised-parent", ".{.,}/outside/secret.txt"),
    ("backslash", "docs\\guide.md"),
    ("environment-variable", "$HOME/.ssh/id_rsa"),
    ("command-substitution", "$(cat /etc/passwd)"),
    ("percent-encoding", "%2e%2e/outside/secret.txt"),
    ("no-break-space", "docs/guide.md\u00a0/etc/hostname"),
    ("ideographic-space", "docs/guide.md\u3000/etc/hostname"),
    ("one-dot-leader", "\u2024\u2024/outside/secret.txt"),
    ("non-ascii-name", "docs/caf\u00e9.md"),
    ("selector-with-space", "docs/guide.md:1 -5"),
    ("empty-selector", "docs/guide.md:"),
    ("selector-then-path", "docs/guide.md:1-5 /etc/hostname"),
    ("dotdot-through-symlink", "{root}/link_out/../README.md"),
    ("dotdot-through-symlink-relative", "link_out/../README.md"),
    ("dotdot-inside", "docs/sub/../guide.md"),
]


@pytest.mark.parametrize("tool", ["read", "grep", "glob", "find", "write", "edit"])
def test_paths_omp_would_rewrite_are_never_allowed(guard: Guard, tool: str) -> None:
    policy = guard.world.policy()
    requests = [
        {
            "op": "decide",
            "policy": policy,
            "event": {"toolName": tool, "input": guard.world.fill({"path": text})},
        }
        for _, text in OMP_REWRITES
    ]

    results = guard.run(requests)

    allowed = [
        name for (name, _), r in zip(OMP_REWRITES, results, strict=True) if r["value"]["allow"]
    ]
    assert allowed == []


@pytest.mark.parametrize(
    "text",
    [
        "@/etc/passwd",
        "@~/x",
        ":/etc/passwd",
        '"/etc/hostname"',
        "docs/guide.md /etc/hostname",
        ".{.,}/outside/secret.txt",
        "docs/caf\u00e9.md",
    ],
)
def test_the_block_comes_from_the_character_rule_not_from_chance(guard: Guard, text: str) -> None:
    """Each of these reads as an in-worktree file to a literal reader (the old hole)."""
    decision = guard.decide("read", {"path": text})
    assert decision.allow is False
    assert "cannot verify" in (decision.reason or ""), decision.reason


def test_the_exact_probe_of_the_review_is_blocked(guard: Guard) -> None:
    for tool, tool_input in [
        ("read", {"path": "@/etc/passwd"}),
        ("read", {"path": "@~/.omp/agent/agent.db"}),
        ("read", {"path": "@/proc/self/environ"}),
        ("grep", {"pattern": "x", "path": "@/home"}),
        ("glob", {"path": "@/home/*/.ssh/*"}),
        ("glob", {"pattern": "@/home/*/.ssh/*"}),
        ("write", {"path": "@../../x.md"}),
        ("write", {"path": "@../../docs/x.md"}),
    ]:
        assert guard.decide(tool, tool_input).allow is False, (tool, tool_input)


PLAIN_READS = [
    "README.md",
    "docs/guide.md",
    "docs/api/ref.md",
    ".github/workflows/ci.yml",
    "docs/my-file_v2+draft.md",
    "./docs/guide.md",
    "docs//guide.md",
    "{root}/docs/guide.md",
    "docs/guide.md:10-20",
    "docs/guide.md:raw:2-4",
    "docs/guide.md:-60",
    "docs/guide.md:1-12,-15",
    "docs/guide.md;README.md",
    ".",
    "docs",
]


@pytest.mark.parametrize("text", PLAIN_READS)
def test_plain_posix_paths_are_still_readable(guard: Guard, text: str) -> None:
    decision = guard.decide("read", {"path": text})
    assert decision.allow is True, decision.reason


def _search_input(key: str, text: str) -> dict[str, str]:
    """Arguments of a search call whose path-bearing field is ``key``."""
    data = {key: text}
    if key != "pattern":
        data["pattern"] = "x"  # grep needs its (non-path) pattern; other tools ignore it
    return data


def test_glob_characters_are_only_allowed_for_read_tools(guard: Guard) -> None:
    for tool, key in [
        ("glob", "path"),
        ("glob", "pattern"),
        ("find", "pattern"),
        ("grep", "path"),
        ("read", "path"),
    ]:
        for pattern in ["docs/**/*.md", "docs/[ab].md", "docs/?.md", "**"]:
            decision = guard.decide(tool, _search_input(key, pattern))
            assert decision.allow is True, (tool, key, pattern, decision.reason)
    for tool in ("write", "edit"):
        for pattern in ["docs/*.md", "docs/[ab].md", "docs/?.md"]:
            decision = guard.decide(tool, {"path": pattern})
            assert decision.allow is False and "cannot verify" in (decision.reason or "")


@pytest.mark.parametrize("tool", ["glob", "find", "grep", "read"])
def test_brace_patterns_are_refused_because_braces_can_synthesise_parent_segments(
    guard: Guard, tool: str
) -> None:
    for pattern in ["docs/{a,b}.md", "{docs,..}/**", ".{.,}/**", "{}"]:
        key = "pattern" if tool in ("glob", "find") else "path"
        decision = guard.decide(tool, _search_input(key, pattern))
        assert decision.allow is False and "cannot verify" in (decision.reason or ""), pattern


def test_a_file_name_with_dots_inside_is_not_a_parent_segment(guard: Guard) -> None:
    assert guard.decide("read", {"path": "docs/a..b.md"}).allow is True
    assert guard.decide("read", {"path": "docs/..hidden/x.md"}).allow is True
    assert guard.decide("read", {"path": "docs/.../x.md"}).allow is True


# ------------------------------------------------------------ edit syntaxes and write payloads
def test_patch_mode_rename_destinations_are_checked_like_any_other_path(guard: Guard) -> None:
    def rename(target: str) -> Decision:
        edits = [{"op": "update", "rename": target, "diff": "@@\n x\n"}]
        return guard.decide("edit", {"path": "docs/guide.md", "edits": edits})

    escaped = rename("../../x.md")
    assert escaped.allow is False and "escapes" in (escaped.reason or "")
    assert "escapes" in (rename("/tmp/x.md").reason or "")
    assert "outside the task's allowed scope" in (rename("src/moved.py").reason or "")
    assert "protected" in (rename("backend/src/safety/estop.py").reason or "")
    assert rename("@/etc/cron.d/x").allow is False
    ok = rename("docs/renamed.md")
    assert ok.allow is True and ok.paths == ["docs/guide.md", "docs/renamed.md"]


INLINE_EDITS = [
    ("hashline-without-path", {"input": "[docs/guide.md#1A2B]\nPUT 1.=1:\n+x"}),
    (
        "hashline-with-decoy-path",
        {"path": "docs/a.md", "input": "[backend/src/safety/estop.py#1A2B]\nPUT 1.=1:\n+x"},
    ),
    ("hashline-move-out", {"path": "docs/a.md", "input": "[docs/a.md#1A2B]\nMV /etc/cron.d/x\n"}),
    ("hashline-delete", {"path": "docs/a.md", "input": "[docs/guide.md#1A2B]\nREM\n"}),
    (
        "apply-patch-outside",
        {"input": "*** Begin Patch\n*** Add File: ../../x\n+y\n*** End Patch\n"},
    ),
    ("nested-input", {"edits": [{"path": "docs/a.md", "input": "[docs/b.md#1A2B]\nREM\n"}]}),
]


@pytest.mark.parametrize("tool", ["edit", "write"])
@pytest.mark.parametrize(
    "tool_input", [i for _, i in INLINE_EDITS], ids=[name for name, _ in INLINE_EDITS]
)
def test_edit_syntaxes_that_name_files_inside_their_text_are_refused(
    guard: Guard, tool: str, tool_input: dict[str, Any]
) -> None:
    decision = guard.decide(tool, tool_input)
    assert decision.allow is False
    assert "names files inside its text" in (decision.reason or "")


def test_an_input_field_is_only_a_problem_for_write_class_tools(guard: Guard) -> None:
    assert guard.decide("read", {"path": "docs/guide.md", "input": "anything"}).allow is True


def test_replace_mode_edits_are_accepted(guard: Guard) -> None:
    decision = guard.decide(
        "edit",
        {"path": "docs/guide.md", "old_string": "guide", "new_string": "manual", "i": "Editing"},
    )
    assert decision.allow is True, decision.reason
    assert decision.paths == ["docs/guide.md"]


@pytest.mark.parametrize("field", ["new_string", "old_string", "diff", "replacement"])
def test_every_string_argument_counts_towards_the_write_limit(guard: Guard, field: str) -> None:
    at_limit = guard.decide("edit", {"path": "docs/a.md", field: "x" * 100})
    over = guard.decide("edit", {"path": "docs/a.md", field: "x" * 101})

    assert at_limit.allow is True
    assert over.allow is False and "100 byte limit" in (over.reason or "")


def test_the_write_limit_adds_up_the_whole_call_and_looks_into_nested_edits(guard: Guard) -> None:
    split = {"path": "docs/a.md", "old_string": "a" * 60, "new_string": "b" * 60}
    assert guard.decide("edit", split).allow is False
    nested = {"path": "docs/a.md", "edits": [{"new_string": "c" * 60}, {"new_string": "d" * 60}]}
    assert guard.decide("edit", nested).allow is False
    long_name = {"path": "docs/" + "n" * 150 + ".md", "content": "x" * 100}
    assert guard.decide("write", long_name).allow is True  # path text is not payload


# --------------------------------------------------------------------------- working directory
def test_relative_paths_resolve_against_the_tool_call_directory(guard: Guard, world: World) -> None:
    inside = guard.decide("write", {"path": "guide.md"}, cwd=str(world.root / "docs"))
    assert inside.allow is True and inside.paths == ["docs/guide.md"]

    parent = guard.decide("write", {"path": "../README.md"}, cwd=str(world.root / "docs"))
    assert parent.allow is False and "parent-directory segments" in (parent.reason or "")


@pytest.mark.parametrize("cwd_name", ["outside", "base", "/"])
def test_a_tool_call_directory_outside_the_worktree_blocks_relative_paths(
    guard: Guard, world: World, cwd_name: str
) -> None:
    cwd = {"outside": str(world.outside), "base": str(world.base), "/": "/"}[cwd_name]
    decision = guard.decide("write", {"path": "docs/new.md"}, cwd=cwd)
    assert decision.allow is False and "escapes the task worktree" in (decision.reason or "")


# --------------------------------------------------------------------------- write sets & deny
def test_every_write_allow_set_must_match_and_deny_wins(guard: Guard, world: World) -> None:
    policy = world.policy(
        write_allow_sets=[["docs/**", "*.md"], ["docs/api/**"]],
        read_deny=[],
        write_deny=["**/.env"],
    )

    assert guard.decide("write", {"path": "docs/api/ref.md"}, policy=policy).allow is True
    for path in ("docs/guide.md", "README.md", "docs/api/.env"):
        decision = guard.decide("write", {"path": path}, policy=policy)
        assert decision.allow is False, path
    scope = guard.decide("write", {"path": "docs/guide.md"}, policy=policy)
    assert "outside the task's allowed scope" in (scope.reason or "")
    protected = guard.decide("write", {"path": "docs/api/.env"}, policy=policy)
    assert "protected" in (protected.reason or "")


def test_any_glob_inside_one_allow_set_is_enough(guard: Guard, world: World) -> None:
    policy = world.policy(write_allow_sets=[["docs/**", "src/**"]], write_deny=[], read_deny=[])
    assert guard.decide("write", {"path": "src/new.py"}, policy=policy).allow is True
    assert guard.decide("write", {"path": "docs/x.md"}, policy=policy).allow is True
    assert guard.decide("write", {"path": "README.md"}, policy=policy).allow is False


def test_with_no_allow_sets_the_deny_list_alone_applies(guard: Guard, world: World) -> None:
    policy = world.policy(write_allow_sets=[], write_deny=["secret/**"], read_deny=[])
    assert guard.decide("write", {"path": "anything.txt"}, policy=policy).allow is True
    assert guard.decide("write", {"path": "secret/x"}, policy=policy).allow is False


def test_read_tools_ignore_the_write_rules(guard: Guard) -> None:
    assert guard.decide("read", {"path": "backend/src/safety/estop.py"}).allow is True
    assert guard.decide("read", {"path": "src/code.py"}).allow is True
    assert guard.decide("glob", {"pattern": "backend/**/*.py"}).allow is True


# --------------------------------------------------------------------------- size limit
@pytest.mark.parametrize("field", ["content", "text", "new_text"])
def test_oversize_writes_are_blocked_by_bytes(guard: Guard, field: str) -> None:
    at_limit = guard.decide("write", {"path": "docs/a.md", field: "x" * 100})
    over = guard.decide("write", {"path": "docs/a.md", field: "x" * 101})

    assert at_limit.allow is True
    assert over.allow is False and "100 byte limit" in (over.reason or "")


def test_the_limit_counts_utf8_bytes_not_characters(guard: Guard) -> None:
    assert guard.decide("write", {"path": "docs/a.md", "content": "\u00e9" * 50}).allow is True
    over = guard.decide("write", {"path": "docs/a.md", "content": "\u00e9" * 51})
    assert over.allow is False and "byte limit" in (over.reason or "")
    emoji = guard.decide("write", {"path": "docs/a.md", "content": "\U0001f600" * 26})
    assert emoji.allow is False


def test_the_limit_applies_to_edits_and_not_to_reads(guard: Guard) -> None:
    assert guard.decide("edit", {"path": "docs/a.md", "new_text": "y" * 500}).allow is False
    assert guard.decide("read", {"path": "docs/guide.md", "content": "y" * 500}).allow is True


@pytest.mark.parametrize("overrides", [{"max_write_bytes": 0}, {"max_write_bytes": None}, {}])
def test_without_a_limit_large_writes_pass(guard: Guard, world: World, overrides: dict) -> None:
    policy = world.policy(**overrides)
    if not overrides:
        del policy["max_write_bytes"]
    big = guard.decide("write", {"path": "docs/a.md", "content": "z" * 1_000_000}, policy=policy)
    assert big.allow is True


def test_the_omp_backend_limit_is_a_real_number(guard: Guard, world: World) -> None:
    assert MAX_WRITE_BYTES > 0
    policy = world.policy(max_write_bytes=MAX_WRITE_BYTES)
    ok = guard.decide(
        "write", {"path": "docs/a.md", "content": "a" * MAX_WRITE_BYTES}, policy=policy
    )
    assert ok.allow is True
    over = guard.decide(
        "write", {"path": "docs/a.md", "content": "a" * (MAX_WRITE_BYTES + 1)}, policy=policy
    )
    assert over.allow is False


# --------------------------------------------------------------------------- bash
BASH_POLICY = {"bash_exact": ["python -m pytest tests/unit -q", "git status"]}


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest tests/unit -q",
        "git status",
        "  python -m pytest tests/unit -q",
        "python -m pytest tests/unit -q\n",
        "\t git status \r\n",
    ],
)
def test_bash_runs_only_allowlisted_command_lines_after_trimming(
    guard: Guard, world: World, command: str
) -> None:
    policy = world.policy(tools=[*TOOLS, "bash"], **BASH_POLICY)
    assert guard.decide("bash", {"command": command}, policy=policy).allow is True


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest tests/unit -q; id",
        "python -m pytest tests/unit -q && id",
        "python -m pytest tests/unit -q | tee x",
        "python -m pytest tests/unit -q > out.txt",
        "python -m pytest tests/unit -q $(id)",
        "python -m pytest tests/unit -q `id`",
        "python -m pytest tests/unit -q\nid",
        "python  -m pytest tests/unit -q",
        "python -m pytest tests/unit",
        "python -m pytest tests/unit -q -x",
        "PYTHON -m pytest tests/unit -q",
        "git",
        "git status ",  # trailing space is trimmed; the extra argument below is not
        "git status --short",
        "git  status",
        "rm -rf /",
        "",
        "   ",
    ],
)
def test_bash_blocks_everything_else(guard: Guard, world: World, command: str) -> None:
    policy = world.policy(tools=[*TOOLS, "bash"], **BASH_POLICY)
    decision = guard.decide("bash", {"command": command}, policy=policy)
    if command == "git status ":
        assert decision.allow is True  # trimmed to the allowlisted line
        return
    assert decision.allow is False and "not on the allowlist" in (decision.reason or "")


@pytest.mark.parametrize(
    "tool_input", [{}, {"command": None}, {"command": 5}, {"command": ["git"]}]
)
def test_bash_without_a_usable_command_is_blocked(
    guard: Guard, world: World, tool_input: dict[str, Any]
) -> None:
    policy = world.policy(tools=[*TOOLS, "bash"], **BASH_POLICY)
    assert guard.decide("bash", tool_input, policy=policy).allow is False


@pytest.mark.parametrize("exact", [None, "git status", [], {"a": "git status"}])
def test_bash_without_a_proper_allowlist_is_blocked(guard: Guard, world: World, exact: Any) -> None:
    policy = world.policy(tools=[*TOOLS, "bash"])
    policy["bash_exact"] = exact
    assert guard.decide("bash", {"command": "git status"}, policy=policy).allow is False


def test_a_bash_command_cannot_run_outside_the_worktree(guard: Guard, world: World) -> None:
    """``cwd`` is a path argument like any other: the allowlisted command must stay inside."""
    policy = world.policy(tools=[*TOOLS, "bash"], **BASH_POLICY)

    assert guard.decide("bash", {"command": "git status", "cwd": "docs"}, policy=policy).allow
    for cwd in ("/", "/etc", "..", "{outside}"):
        decision = guard.decide("bash", {"command": "git status", "cwd": cwd}, policy=policy)
        assert decision.allow is False, cwd


# --------------------------------------------------------------------------- invalid policies
INVALID_POLICIES = [
    None,
    {},
    [],
    "policy",
    {"root": "work", "tools": [], "write_allow_sets": [], "write_deny": [], "read_deny": []},
    {"root": 5, "tools": [], "write_allow_sets": [], "write_deny": [], "read_deny": []},
    {"tools": [], "write_allow_sets": [], "write_deny": [], "read_deny": []},
    {"root": "/tmp", "write_allow_sets": [], "write_deny": [], "read_deny": []},
    {"root": "/tmp", "tools": [], "write_deny": [], "read_deny": []},
    {"root": "/tmp", "tools": [], "write_allow_sets": [], "read_deny": []},
    {"root": "/tmp", "tools": [], "write_allow_sets": [], "write_deny": []},
    {"root": "/tmp", "tools": "read", "write_allow_sets": [], "write_deny": [], "read_deny": []},
    {"root": "/tmp", "tools": [], "write_allow_sets": {}, "write_deny": [], "read_deny": []},
]


@pytest.mark.parametrize("policy", INVALID_POLICIES, ids=lambda p: json.dumps(p)[:60])
@pytest.mark.parametrize("tool", ["write", "read", "yield", "think"])
def test_an_invalid_policy_blocks_everything(guard: Guard, policy: Any, tool: str) -> None:
    result = guard.decide_raw(policy, tool, {"path": "docs/a.md"})

    assert result["ok"]
    assert result["value"] == {
        "allow": False,
        "reason": "guard policy missing or invalid",
        "paths": [],
    }


def test_a_worktree_that_does_not_exist_blocks_calls_but_not_yield(
    guard: Guard, world: World
) -> None:
    policy = world.policy(root=str(world.base / "no-such-worktree"))

    blocked = guard.decide("read", {"path": "a.md"}, policy=policy)
    assert blocked.allow is False and "worktree is not accessible" in (blocked.reason or "")
    assert guard.decide("yield", {}, policy=policy).allow is True


# --------------------------------------------------------------------------- tool_call handler
def call(tool: str, tool_input: Any, cwd: str | None = None) -> dict[str, Any]:
    return {"event": {"toolName": tool, "input": tool_input}, "ctx": {"cwd": cwd} if cwd else None}


def test_the_default_export_registers_exactly_one_tool_call_handler(
    guard: Guard, world: World
) -> None:
    policy_file = write_policy_file(world, world.policy())
    result = guard.handler([], policy_file=policy_file)
    assert result == {"events": ["tool_call"], "results": []}


def test_blocked_calls_return_a_block_object_and_allowed_calls_return_undefined(
    guard: Guard, world: World
) -> None:
    policy_file = write_policy_file(world, world.policy())
    result = guard.handler(
        [
            call("write", {"path": "docs/new.md", "content": "x"}),
            call("write", {"path": "/etc/passwd", "content": "x"}),
            call("browser", {}),
            call("yield", {}),
            call("read", {"path": "backend/src/safety/estop.py"}),
            call("write", {"path": "backend/src/safety/estop.py"}),
        ],
        policy_file=policy_file,
    )

    allowed = {"kind": "undefined"}
    assert result["results"] == [
        allowed,
        {
            "kind": "value",
            "value": {
                "block": True,
                "reason": f"{REASON_PREFIX}path escapes the task worktree: /etc/passwd",
            },
        },
        {
            "kind": "value",
            "value": {"block": True, "reason": f"{REASON_PREFIX}tool 'browser' is not enabled"},
        },
        allowed,
        allowed,
        {
            "kind": "value",
            "value": {
                "block": True,
                "reason": f"{REASON_PREFIX}path is protected and needs a human change: "
                "backend/src/safety/estop.py",
            },
        },
    ]


def test_the_handler_uses_the_context_directory_for_relative_paths(
    guard: Guard, world: World
) -> None:
    policy_file = write_policy_file(world, world.policy())
    docs = str(world.root / "docs")

    result = guard.handler(
        [call("write", {"path": "guide.md"}, docs), call("write", {"path": "guide.md"}, "/")],
        policy_file=policy_file,
    )

    assert result["results"][0] == {"kind": "undefined"}
    assert result["results"][1]["value"]["block"] is True


@pytest.mark.parametrize(
    "policy",
    [
        "missing-file",
        "not json at all",
        "[]",
        '{"root": "/tmp"}',
        '{"root": "relative", "tools": [], "write_allow_sets": [], "write_deny": [],'
        ' "read_deny": []}',
        "",
    ],
    ids=["missing-file", "not-json", "array", "missing-keys", "relative-root", "empty-file"],
)
def test_a_missing_or_invalid_policy_file_blocks_every_call(
    guard: Guard, world: World, policy: str
) -> None:
    policy_file = world.base / "absent.json" if policy == "missing-file" else None
    if policy_file is None:
        policy_file = write_policy_file(world, policy)
    calls = [call(tool, {"path": "docs/a.md"}) for tool in ("write", "read", "yield", "think")]

    result = guard.handler(calls, policy_file=policy_file)

    expected = {
        "kind": "value",
        "value": {"block": True, "reason": f"{REASON_PREFIX}guard policy missing or invalid"},
    }
    assert result["results"] == [expected] * 4


def test_an_unset_policy_variable_blocks_every_call(guard: Guard) -> None:
    result = guard.handler(
        [call("write", {"path": "docs/a.md"}), call("yield", {})], policy_file=None
    )
    assert [r["value"]["block"] for r in result["results"]] == [True, True]
    assert all("policy missing or invalid" in r["value"]["reason"] for r in result["results"])


def test_an_internal_error_fails_closed(guard: Guard, world: World) -> None:
    broken = world.policy(write_allow_sets=["docs/**"])  # a flat list instead of a list of lists
    policy_file = write_policy_file(world, broken)

    result = guard.handler([call("write", {"path": "docs/a.md"})], policy_file=policy_file)

    value = result["results"][0]["value"]
    assert value["block"] is True
    assert value["reason"].startswith(f"{REASON_PREFIX}guard error:")


# --------------------------------------------------------------------------- audit log
def read_audit(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_decisions_are_appended_to_the_audit_log_without_the_payload(
    guard: Guard, world: World
) -> None:
    log = world.base / "audit.jsonl"
    policy_file = write_policy_file(world, world.policy(audit_log=str(log)))

    guard.handler(
        [
            call("write", {"path": "docs/new.md", "content": "TOP-SECRET-PAYLOAD"}),
            call("write", {"path": "/etc/passwd", "content": "TOP-SECRET-PAYLOAD"}),
            call("browser", {"url": "https://example.com"}),
            call("yield", {}),
        ],
        policy_file=policy_file,
    )
    rows = read_audit(log)

    assert [(r["tool"], r["allow"]) for r in rows] == [
        ("write", True),
        ("write", False),
        ("browser", False),
        ("yield", True),
    ]
    assert all(set(r) == {"ts", "tool", "allow", "reason", "paths"} for r in rows)
    assert rows[0]["reason"] is None and rows[0]["paths"] == ["docs/new.md"]
    assert "escapes the task worktree" in rows[1]["reason"] and rows[1]["paths"] == []
    assert rows[2]["reason"] == "tool 'browser' is not enabled"
    assert "TOP-SECRET-PAYLOAD" not in log.read_text(encoding="utf-8")
    assert "example.com" not in log.read_text(encoding="utf-8")
    for row in rows:
        datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))


def test_the_audit_log_is_appended_to_not_overwritten(guard: Guard, world: World) -> None:
    log = world.base / "audit.jsonl"
    log.write_text('{"earlier": "run"}\n', encoding="utf-8")
    policy_file = write_policy_file(world, world.policy(audit_log=str(log)))

    guard.handler([call("read", {"path": "README.md"})], policy_file=policy_file)
    guard.handler([call("read", {"path": "README.md"})], policy_file=policy_file)

    rows = read_audit(log)
    assert rows[0] == {"earlier": "run"} and len(rows) == 3


def test_the_audit_log_is_capped_per_run(guard: Guard, world: World) -> None:
    log = world.base / "audit.jsonl"
    policy_file = write_policy_file(world, world.policy(audit_log=str(log)))

    result = guard.handler([call("browser", {})], policy_file=policy_file, repeat=5100)

    assert result["results"][0]["value"]["block"] is True  # decisions keep being enforced
    assert len(log.read_text(encoding="utf-8").splitlines()) == 5000


@pytest.mark.parametrize("target", ["missing-directory/audit.jsonl", "a-directory"])
def test_an_unwritable_audit_log_never_breaks_the_run(
    guard: Guard, world: World, target: str
) -> None:
    (world.base / "a-directory").mkdir()
    policy_file = write_policy_file(world, world.policy(audit_log=str(world.base / target)))

    result = guard.handler(
        [call("write", {"path": "docs/new.md"}), call("write", {"path": "/etc/x"})],
        policy_file=policy_file,
    )

    assert result["results"][0] == {"kind": "undefined"}
    assert result["results"][1]["value"]["block"] is True


def test_no_audit_file_is_created_without_an_audit_log_setting(guard: Guard, world: World) -> None:
    before = {p.name for p in world.base.iterdir()}
    policy_file = write_policy_file(world, world.policy())
    guard.handler([call("write", {"path": "docs/new.md"})], policy_file=policy_file)
    assert {p.name for p in world.base.iterdir()} == before | {"policy.json"}


# --------------------------------------------------------------------- python / node contract
CORPUS = [
    "docs/guide.md",
    "docs/api/ref.md",
    "docs/api/deep/er/ref.md",
    "docs/new.md",
    "docs/.env",
    "docs/keys/server.pem",
    "docs/keys/server.key",
    "docs/constitution.md",
    "docs",
    "README.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "AGENTS.md",
    "sub/README.md",
    "src/code.py",
    "notes.txt",
    "backend/src/safety/estop.py",
    "backend/src/safety",
    "backend/src/safetyx/a.py",
    "backend/src/services/motor_service.py",
    "backend/src/services/weather_service.py",
    "backend/src/core/config.py",
    "backend/src/drivers/gps/zed.py",
    "config/hardware.yaml",
    "config/secrets.json",
    "config/agent-bridge.yaml",
    "systemd/lawnberry.service",
    "scripts/deploy.sh",
    ".github/workflows/ci.yml",
    ".gitmodules",
    ".gitattributes",
    ".env",
    "a/b/.env.local",
    "tools/agent_bridge/omp_guard.mjs",
    "tools/helper.py",
    "tests/test_x.py",
    "pyproject.toml",
    "frontend/package.json",
    "frontend/src/App.vue",
    "docs/sub/.hidden.md",
    "docs/a b.md",
    "docs/caf\u00e9.md",
    # nested agent-policy files and the case spellings of the protected names (C-04, C-05)
    "docs/AGENTS.md",
    "docs/CLAUDE.md",
    "Claude.MD",
    "claude.md",
    "Agents.md",
    "docs/.CLAUDE/x.md",
    "docs/.claude/x.md",
    "docs/.omp/x.md",
    "CLAUDE.local.md",
    "docs/CLAUDE.local.md",
    "docs/.gitattributes",
    "docs/.GITIGNORE",
    "tests/AGENTS.md",
    "tools/AGENTS.md",
    "GEMINI.md",
    "docs/gemini.md",
    "docs/.cursorrules",
    "docs/.mcp.json",
    "docs/.git/config",
    "docs/.github/workflows/x.yml",
    "docs/Constitution.md",
    "docs/agents-guide.md",
    "docs/claude-notes.md",
]


def test_python_policy_and_node_guard_agree_on_every_write(guard: Guard, world: World) -> None:
    cfg = BridgeConfig.model_validate({"repository": "owner/repo", "approvers": ["alice"]})
    spec = TaskSpec.model_validate(
        {
            "version": 1,
            "objective": "Update the API reference and the guide in the docs.",
            "allowed_paths": ["docs/api/**", "docs/guide.md", "*.md"],
        }
    )
    write_policy = build_write_policy(spec, cfg)
    policy = {
        "version": 1,
        "root": str(world.root),
        "tools": TOOLS,
        "read_deny": [],
        "bash_exact": [],
        **write_policy.describe(),
    }
    requests = [
        {"op": "decide", "policy": policy, "event": {"toolName": "write", "input": {"path": p}}}
        for p in CORPUS
    ]

    results = guard.run(requests)

    node_allows = [r["value"]["allow"] for r in results]
    python_allows = [write_policy.check_path(p) is None for p in CORPUS]
    disagreements = [
        p for p, a, b in zip(CORPUS, python_allows, node_allows, strict=True) if a != b
    ]
    assert disagreements == []
    assert True in node_allows and False in node_allows


def test_the_policy_the_omp_backend_writes_is_accepted_by_the_guard(
    guard: Guard, world: World
) -> None:
    """Use the same keys, tools and read-deny list the real backend serialises."""
    cfg = BridgeConfig.model_validate({"repository": "owner/repo", "approvers": ["alice"]})
    spec = TaskSpec.model_validate({"version": 1, "objective": "Update the docs, please."})
    policy = {
        "version": 1,
        "root": str(world.root),
        "tools": ["read", "grep", "glob", "edit", "write"],
        "read_deny": list(READ_DENY),
        "bash_exact": [],
        "audit_log": str(world.base / "audit.jsonl"),
        "max_write_bytes": MAX_WRITE_BYTES,
        **build_write_policy(spec, cfg).describe(),
    }
    policy_copy = copy.deepcopy(policy)

    assert guard.decide("write", {"path": "docs/new.md", "content": "x"}, policy=policy).allow
    assert guard.decide("read", {"path": "src/code.py"}, policy=policy).allow
    for blocked in (".env", ".git/config", "sub/.env"):
        assert guard.decide("read", {"path": blocked}, policy=policy).allow is False
    assert guard.decide("write", {"path": "src/code.py"}, policy=policy).allow is False
    assert guard.decide("write", {"path": "docs/constitution.md"}, policy=policy).allow is False
    assert policy == policy_copy
