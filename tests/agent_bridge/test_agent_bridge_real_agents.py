"""Opt-in smoke tests against the REAL ``omp`` and ``claude`` CLIs.

These run the actual agent binaries through ``tools.agent_bridge.smoke.run_smoke`` in a throwaway
git repository.  They spend real model budget (a few cents for ``omp``) and need working
credentials, so they are skipped unless you ask for them::

    LAWNBERRY_BRIDGE_REAL_AGENTS=1 \\
    LAWNBERRY_BRIDGE_OMP_MODEL=anthropic/claude-sonnet-5-5 \\
        python -m pytest tests/agent_bridge/test_agent_bridge_real_agents.py -s -p no:cacheprovider

``LAWNBERRY_BRIDGE_REAL_AGENTS=1``
    Enables the real-agent tests below (each is also skipped when its binary is not on PATH).
``LAWNBERRY_BRIDGE_OMP_MODEL``
    Optional model for the ``omp`` run (``provider/model``); omp's own default when unset.

Pass ``-s`` (or ``-rP``) to see the printed smoke reports.  The ``claude`` test accepts both a
working and a non-working login: a workstation whose Claude account cannot authenticate must
produce a structured, classified failure, never an exception.

The last tests in the module are hermetic and always run: they feed reports produced by the fake
agent CLIs through the very same checks, so a typo in an assertion cannot hide until someone
finally pays for a real run.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from bridgefakes.agent_fixtures import (
    docs_edit_steps,
    install_fake_cli,
    write_scenario,
)

from tools.agent_bridge.smoke import SMOKE_BODY, run_smoke

ENABLE_ENV = "LAWNBERRY_BRIDGE_REAL_AGENTS"
OMP_MODEL_ENV = "LAWNBERRY_BRIDGE_OMP_MODEL"
SMOKE_LINE = "agent bridge smoke ok"
SMOKE_NOTE = "docs/smoke-note.md"
CLAUDE_FAILURE_KINDS = {"auth", "credit", "rate_limit", "api_error"}
GUARD_BLOCK = "[agent-bridge guard] blocked: path is outside the task worktree"

real_agents_enabled = os.environ.get(ENABLE_ENV) == "1"
needs_opt_in = pytest.mark.skipif(
    not real_agents_enabled, reason=f"real agent tests are opt-in: set {ENABLE_ENV}=1"
)
needs_omp = pytest.mark.skipif(shutil.which("omp") is None, reason="omp is not installed")
needs_claude = pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")


def show(label: str, report: dict[str, Any]) -> None:
    """Print a smoke report for the log (visible with ``-s`` or in a failure report)."""
    print(f"----- {label} -----")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


# ----------------------------------------------------------------------------------------------
# the checks shared by the real and the hermetic tests
# ----------------------------------------------------------------------------------------------
def check_omp_report(report: dict[str, Any]) -> None:
    """A successful omp smoke run with the guard probe: file written, cost recorded, guard held."""
    assert report["backend"] == "omp"
    assert report["binary"], "the omp binary was not resolved"
    assert report["version"], "omp --version printed nothing"
    result = report["result"]
    assert result["status"] == "succeeded", result.get("error")
    assert result["failure_kind"] is None and result["error"] is None
    assert result["exit_code"] == 0
    assert report["ok"] is True, report["git"]["violations"]
    assert report["note_content"] in (SMOKE_LINE, SMOKE_LINE + "\n")
    assert report["git"]["changed"] == [SMOKE_NOTE]
    assert report["git"]["violations"] == []
    assert result["cost_usd"] is not None and result["cost_usd"] > 0, "no cost was recorded"
    assert result["cost_usd"] <= result["budget_usd"]
    probe = report["guard_probe"]
    assert probe["outside_file_created"] is False, "the guard let a write escape the worktree"
    assert probe["blocked_tool_calls"] >= 1, "the guard blocked nothing"


def check_claude_report(report: dict[str, Any]) -> None:
    """A structured claude result, whether or not this workstation's login works."""
    assert report["backend"] == "claude"
    assert "result" in report, f"no structured result: {report.get('error')}"
    result = report["result"]
    assert result["status"] in {"succeeded", "failed"}, result
    if result["status"] == "failed":
        assert result["failure_kind"] in CLAUDE_FAILURE_KINDS, result
        assert result["error"], "a failed run must say why"
        assert report["ok"] is False
    else:
        assert report["ok"] is True, report["git"]["violations"]
        assert report["note_content"] in (SMOKE_LINE, SMOKE_LINE + "\n")
        assert report["git"]["violations"] == []


# ----------------------------------------------------------------------------------------------
# real agents (opt-in)
# ----------------------------------------------------------------------------------------------
@needs_opt_in
@needs_omp
def test_real_omp_writes_the_note_and_the_guard_holds() -> None:
    report = run_smoke("omp", model=os.environ.get(OMP_MODEL_ENV) or None, probe_guard=True)
    show("real omp", report)
    check_omp_report(report)


@needs_opt_in
@needs_claude
def test_real_claude_always_returns_a_structured_result() -> None:
    report = run_smoke("claude")
    show("real claude", report)
    check_claude_report(report)


# ----------------------------------------------------------------------------------------------
# hermetic self-checks (always run): the same checks against the fake CLIs
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def fake_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    scenario = tmp_path / "scenario.json"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FAKE_AGENT_SCENARIO", str(scenario))
    monkeypatch.delenv("FAKE_AGENT_LOG", raising=False)
    return {
        "scenario": scenario,
        "omp": install_fake_cli(tmp_path / "bin", "omp"),
        "claude": install_fake_cli(tmp_path / "bin", "claude"),
    }


def smoke_with_fake(fake_agents: dict[str, Any], backend: str, **kwargs: Any) -> dict[str, Any]:
    return run_smoke(
        backend,
        binary=str(fake_agents[backend]),
        pass_env=["FAKE_AGENT_SCENARIO"],
        **kwargs,
    )


def test_smoke_task_asks_for_exactly_the_checked_file() -> None:
    assert SMOKE_NOTE in SMOKE_BODY and SMOKE_LINE in SMOKE_BODY
    assert "scope: docs-only" in SMOKE_BODY


def test_omp_checks_accept_a_successful_fake_run(fake_agents: dict[str, Any]) -> None:
    blocked = {"tool": "write", "args": {"path": "/etc/x"}, "blocked": True}
    steps = docs_edit_steps(path=SMOKE_NOTE, content=SMOKE_LINE + "\n")
    write_scenario(
        fake_agents["scenario"],
        [
            *steps[:-1],
            {"op": "tool_call", **blocked, "result": GUARD_BLOCK},
            steps[-1],
        ],
    )
    report = smoke_with_fake(fake_agents, "omp", probe_guard=True)
    check_omp_report(report)


def test_omp_checks_reject_a_run_that_wrote_nothing(fake_agents: dict[str, Any]) -> None:
    write_scenario(fake_agents["scenario"], [{"op": "final", "text": "nothing"}])
    report = smoke_with_fake(fake_agents, "omp", probe_guard=True)
    with pytest.raises(AssertionError):
        check_omp_report(report)


def test_claude_checks_accept_a_working_login(fake_agents: dict[str, Any]) -> None:
    write_scenario(
        fake_agents["scenario"], docs_edit_steps(path=SMOKE_NOTE, content=SMOKE_LINE + "\n")
    )
    check_claude_report(smoke_with_fake(fake_agents, "claude"))


@pytest.mark.parametrize("kind", sorted(CLAUDE_FAILURE_KINDS))
def test_claude_checks_accept_every_classified_login_failure(
    fake_agents: dict[str, Any], kind: str
) -> None:
    write_scenario(fake_agents["scenario"], [{"op": "fail", "kind": kind}])
    report = smoke_with_fake(fake_agents, "claude")
    assert report["result"]["failure_kind"] == kind
    check_claude_report(report)


def test_claude_checks_reject_an_unclassified_failure(fake_agents: dict[str, Any]) -> None:
    write_scenario(fake_agents["scenario"], [{"op": "fail", "kind": "crash"}])
    report = smoke_with_fake(fake_agents, "claude")
    assert report["result"]["failure_kind"] == "nonzero_exit"
    with pytest.raises(AssertionError):
        check_claude_report(report)
