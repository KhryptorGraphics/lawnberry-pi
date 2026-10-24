"""Publication integrity: what the bridge writes into git history and onto GitHub.

Three properties, each pinned against a real git origin, the fake GitHub and the fake agent CLIs
(every ``rig`` test runs once per backend):

* the commit message carries no agent text: it lands on the default branch once a human merges
  the pull request, and there closing keywords, ``@mentions`` and ``Co-authored-by`` lines in it
  would take effect;
* issue-derived text that the bridge posts (refusal reasons, the pull request's task title) is
  rendered as an inline code span nothing inside it can close;
* the pull request the worker records for a run is an open draft from the run's own branch,
  never one it merely found there.
"""

from __future__ import annotations

import os
import re

import pytest
from bridgefakes.rig import REPO, Rig

from tools.agent_bridge.agents.base import AgentReport, AgentResult, AgentStatus
from tools.agent_bridge.ledger import LeaseRequest, RunRecord
from tools.agent_bridge.models import RunStatus
from tools.agent_bridge.worker import Eligible, _Ctx

DOCS_NOTE = "docs/agent-note.md"
TITLE = "Document the bridge"

# ----------------------------------------------------------------------------------------------
# F4: the commit message carries no agent text
# ----------------------------------------------------------------------------------------------
# What GitHub acts on once such a commit reaches the default branch.
CLOSING_REFERENCE = re.compile(r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#\d+")

HOSTILE_SUMMARIES = {
    "closing-keywords-and-a-forged-co-author": (
        "Updated docs.\n\nFixes #1\nCloses #2\nCo-authored-by: Mallory <m@evil.example>"
    ),
    "mention-and-forged-trailers": (
        "Done.\n\n@everyone please look at this\nApproved-By: mallory\n"
        "Signed-off-by: Mallory <m@evil.example>\nResolves other/repo#3"
    ),
    "long-text-once-copied-in-up-to-800-characters": "word " * 400,
}


def expected_commit_message(run: RunRecord, *, title: str = TITLE, keyword: str = "Refs") -> str:
    """The whole message the bridge writes: subject and its own trailers, nothing else."""
    return (
        f"agent(#{run.issue_number}): {title}\n"
        "\n"
        f"{keyword} #{run.issue_number}\n"
        f"Agent-Bridge-Run: {run.run_id}\n"
        f"Agent-Backend: {run.backend}\n"
        f"Approved-By: {run.approved_by}\n"
    )


def context_for_commit(rig: Rig, summary: str) -> tuple[_Ctx, RunRecord]:
    """A real leased run and the context the worker holds right before it commits."""
    number = rig.new_task_issue()
    decision = rig.worker._evaluate(rig.client.get_issue(number))
    assert isinstance(decision, Eligible), decision
    task, approval = decision.task, decision.approval
    lease = rig.ledger.lease(
        LeaseRequest(
            repo=REPO,
            issue_number=number,
            issue_title=task.title,
            revision=task.revision,
            spec_hash=task.spec_hash,
            approval_event_id=approval.event_id,
            approved_by=approval.actor,
            approved_at=approval.created_at,
            backend=decision.backend,
            worker_id=rig.worker.worker_id,
            worker_pid=os.getpid(),
            ttl_seconds=rig.cfg.limits.lease_ttl_seconds,
            max_attempts_per_issue=rig.cfg.limits.max_attempts_per_issue,
            dedupe_by_content=rig.cfg.dedupe_by_content,
        )
    )
    assert lease.granted and lease.run is not None
    agent = AgentResult(
        backend=decision.backend,
        status=AgentStatus.SUCCEEDED,
        report=AgentReport(status="completed", summary=summary),
    )
    ctx = _Ctx(run=lease.run, el=decision, run_dir=rig.tmp_path / "run", agent=agent)
    return ctx, lease.run


@pytest.mark.parametrize("case_id", sorted(HOSTILE_SUMMARIES))
def test_commit_message_drops_the_agent_summary_and_keeps_the_bridge_trailers(
    rig: Rig, case_id: str
) -> None:
    ctx, run = context_for_commit(rig, HOSTILE_SUMMARIES[case_id])

    message = rig.worker._commit_message(ctx)

    assert message == expected_commit_message(run)
    assert not CLOSING_REFERENCE.search(message)
    for planted in ("Co-authored-by", "@everyone", "Signed-off-by", "mallory", "word word"):
        assert planted not in message


def test_the_real_commit_carries_no_agent_text_but_the_evidence_stays_in_the_pr(rig: Rig) -> None:
    rig.docs_scenario(
        report={
            "status": "completed",
            "summary": HOSTILE_SUMMARIES["closing-keywords-and-a-forged-co-author"],
            "files_changed": [DOCS_NOTE],
        }
    )
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.branch is not None and run.pr_number is not None
    message = rig.git_env.origin_git("log", "-1", "--format=%B", run.branch)
    assert message == expected_commit_message(run).strip()
    assert "Co-authored-by" not in message and not CLOSING_REFERENCE.search(message)
    assert rig.git_env.origin_git("rev-list", "--count", f"main..{run.branch}") == "1"
    # The summary is not lost: it is still shown to the reviewer, fenced, on the PR and the issue.
    evidence = "Co-authored-by: Mallory <m@evil.example>"
    for text in (rig.fake.pulls[run.pr_number].body, rig.comments(number)[-1]):
        fences = re.findall(r"^(`{3,})text\n(.*?)^\1$", text, re.DOTALL | re.MULTILINE)
        assert [evidence in body for _fence, body in fences] == [True], text
        outside = re.sub(r"^(`{3,})text\n.*?^\1$", "", text, flags=re.DOTALL | re.MULTILINE)
        assert evidence not in outside and "Fixes #1" not in outside
