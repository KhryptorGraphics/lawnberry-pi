"""End-to-end worker flow (happy path) against fake GitHub and fake agent CLIs."""

from __future__ import annotations

from bridgefakes.rig import Rig

from tools.agent_bridge.models import RunPhase, RunStatus


def test_approved_docs_task_becomes_one_draft_pr(rig: Rig) -> None:
    number = rig.new_task_issue()
    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.phase is RunPhase.FINISHED
    assert run.pr_number is not None
    pull = rig.fake.pulls[run.pr_number]
    assert pull.draft is True
    assert pull.head_ref == run.branch
    assert run.branch in rig.git_env.origin_branches()
    labels = rig.fake.issue_labels(number)
    assert "agent-done" in labels and "agent-running" not in labels
    assert len(rig.invocations()) == 1
