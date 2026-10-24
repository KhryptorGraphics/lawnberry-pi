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
from dataclasses import replace
from typing import Any

import pytest
from bridgefakes.github_server import WEB_URL, FakeGitHub, FakePull
from bridgefakes.rig import REPO, Rig, task_body

from tools.agent_bridge.agents.base import AgentReport, AgentResult, AgentStatus
from tools.agent_bridge.ledger import LeaseRequest, RunRecord
from tools.agent_bridge.models import RunStatus
from tools.agent_bridge.publisher import render_pull_request_body, render_refusal
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


def lease_run(rig: Rig) -> tuple[Eligible, RunRecord]:
    """An approved task for a new issue and the run the worker leases for it (as ``_execute``)."""
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
    return decision, lease.run


def context_for_commit(rig: Rig, summary: str) -> tuple[_Ctx, RunRecord]:
    """The context the worker holds right before it commits a run whose agent said ``summary``."""
    decision, run = lease_run(rig)
    agent = AgentResult(
        backend=decision.backend,
        status=AgentStatus.SUCCEEDED,
        report=AgentReport(status="completed", summary=summary),
    )
    return _Ctx(run=run, el=decision, run_dir=rig.tmp_path / "run", agent=agent), run


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


# ----------------------------------------------------------------------------------------------
# F5: issue-derived text is posted as an inline code span nothing in it can close
# ----------------------------------------------------------------------------------------------
_BACKTICK_RUN = re.compile(r"`+")


def code_spans(text: str) -> list[tuple[int, int, str]]:
    """The inline code spans of ``text`` as ``(start, end, content)``, as CommonMark reads them.

    A run of N backticks opens a span that the next run of exactly N backticks closes; a run
    without such a partner is literal text.  Backslashes mean nothing inside a span.
    """
    runs = [(m.start(), m.end()) for m in _BACKTICK_RUN.finditer(text)]
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(runs):
        start, end = runs[index]
        closer = next(
            (j for j in range(index + 1, len(runs)) if runs[j][1] - runs[j][0] == end - start),
            None,
        )
        if closer is None:
            index += 1
            continue
        content = text[end : runs[closer][0]].replace("\n", " ")
        if content.startswith(" ") and content.endswith(" ") and content.strip(" "):
            content = content[1:-1]
        spans.append((start, runs[closer][1], content))
        index = closer + 1
    return spans


def outside_code(text: str) -> str:
    """``text`` with every inline code span cut out (what GitHub would render as live text)."""
    kept: list[str] = []
    for line in text.splitlines():
        cursor = 0
        for start, end, _content in code_spans(line):
            kept.append(line[cursor:start])
            cursor = end
        kept.append(line[cursor:] + "\n")
    return "".join(kept)


def gfm_cells(row: str) -> list[str]:
    """The cells of one GFM table row, split the way GitHub's cmark-gfm splits them.

    A pipe separates cells unless a backslash directly precedes it - code spans do not protect
    it - and the escaping backslash is then dropped from the cell text.
    """
    assert row.startswith("|") and row.endswith("|"), row
    cells: list[str] = []
    current: list[str] = []
    index = 1
    while index < len(row):
        char = row[index]
        if char == "\\" and row[index + 1 : index + 2] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    assert not "".join(current).strip(), f"text after the closing pipe of {row!r}"
    return cells


def refusal_bullets(body: str) -> list[str]:
    """The reason lines of a refusal comment (everything after ``- Reason:``)."""
    lines = body.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("- Reason:")) + 1
    end = lines.index("", start)
    return lines[start:end]


def task_row(body: str) -> str:
    (row,) = [line for line in body.splitlines() if line.startswith("| Task |")]
    return row


# Attacker-chosen text that reaches the bridge's own comments: YAML key names end up in the
# validation messages, the issue title ends up in the pull request.
YAML_KEY = "@everyone [click](http://evil.example)"
HOSTILE_REASONS = {
    "mention-and-link-from-a-yaml-key": (
        f"{YAML_KEY}: Extra inputs are not permitted",
        f"{YAML_KEY}: Extra inputs are not permitted",
    ),
    "backtick-runs-cannot-close-the-span": (
        "key with `ticks`, ``more`` and ``` fences ```: bad",
        "key with `ticks`, ``more`` and ``` fences ```: bad",
    ),
    "starts-and-ends-with-a-backtick": ("`weird`", "`weird`"),
    "html-and-references": (
        "<img src=x onerror=alert(1)> <!-- hidden --> Closes #1 @org/team",
        "<img src=x onerror=alert(1)> <!-- hidden --> Closes #1 @org/team",
    ),
    "lines-and-controls-collapse": ("first\nsecond\r\n\tthird\x00 tail", "first second third tail"),
    "long-text-is-cut-at-300": ("word " * 100, ("word " * 100).strip()[:297] + "..."),
}
TITLE_WITH_EVERYTHING = "Fix a|b and `c` for @x [click](http://evil.example)"
HOSTILE_TITLES = {
    "pipe-backticks-mention-and-link": TITLE_WITH_EVERYTHING,
    "a-lone-pipe": "|",
    "backslash-before-a-pipe": "path\\|name",
    "trailing-backslash": "ends with a backslash\\",
    "starts-and-ends-with-a-backtick": "`code`",
    "a-lone-backtick": "`",
    "backtick-runs": "a ``` b ```` c",
    "html-and-references": "<img src=x onerror=alert(1)> Closes #9 <!-- c --> @org/team",
}


def test_refusal_reasons_are_posted_inside_an_inline_code_span() -> None:
    body = render_refusal(
        "invalid_task", [f"{YAML_KEY}: Extra inputs are not permitted"], key="refusal-7-3-x"
    )

    assert refusal_bullets(body) == [f"- `{YAML_KEY}: Extra inputs are not permitted`"]


@pytest.mark.parametrize("case_id", sorted(HOSTILE_REASONS))
def test_every_refusal_bullet_is_exactly_one_code_span_with_the_message_inside(
    case_id: str,
) -> None:
    message, shown = HOSTILE_REASONS[case_id]

    body = render_refusal("invalid_task", ["an ordinary reason", message], key="refusal-7-3-x")

    ordinary, hostile = refusal_bullets(body)
    assert ordinary == "- `an ordinary reason`"
    assert hostile.startswith("- ")
    assert code_spans(hostile[2:]) == [(0, len(hostile) - 2, shown)]  # nothing outside the span


def test_a_hostile_yaml_key_reaches_the_issue_only_as_code(rig: Rig) -> None:
    number = rig.new_task_issue(body=task_body(**{YAML_KEY: 1}))

    rig.poll()
    rig.poll()  # the notice is posted once, however often the issue is polled

    (notice,) = rig.comments(number)
    assert "### Agent bridge cannot run this task" in notice
    assert f"- `{YAML_KEY}: Extra inputs are not permitted`" in notice.splitlines()
    live = outside_code(notice)
    for fragment in ("@everyone", "[click]", "evil.example"):
        assert fragment in notice and fragment not in live
    assert rig.ledger.list_runs() == [] and rig.invocations() == []


def pull_request_body_for(rig: Rig, title: str) -> tuple[str, int]:
    """The PR body for a leased run whose issue title reads ``title``, and its issue number."""
    _decision, run = lease_run(rig)
    body = render_pull_request_body(
        replace(run, issue_title=title),
        {},
        issue_number=run.issue_number,
        keyword="Refs",
        redactor=rig.redactor,
    )
    return body, run.issue_number


def test_pull_request_table_row_for_a_hostile_title_is_exact(rig: Rig) -> None:
    body, number = pull_request_body_for(rig, TITLE_WITH_EVERYTHING)

    assert task_row(body) == (
        f"| Task | #{number} ``Fix a\\|b and `c` for @x [click](http://evil.example)`` |"
    )


@pytest.mark.parametrize("case_id", sorted(HOSTILE_TITLES))
def test_a_hostile_title_stays_one_cell_holding_one_code_span_with_the_title_inside(
    rig: Rig, case_id: str
) -> None:
    title = HOSTILE_TITLES[case_id]

    body, number = pull_request_body_for(rig, title)

    table = [line for line in body.splitlines() if line.startswith("|")]
    assert [len(gfm_cells(row)) for row in table] == [2] * len(table)
    label, cell = gfm_cells(task_row(body))
    prefix = f"#{number} "
    assert label == "Task" and cell.startswith(prefix)
    assert code_spans(cell) == [(len(prefix), len(cell), title)]  # all of the title, only that


def test_a_long_title_is_cut_at_120_characters_inside_the_span(rig: Rig) -> None:
    body, number = pull_request_body_for(rig, "x" * 200)

    assert task_row(body) == f"| Task | #{number} `{'x' * 117}...` |"


def test_text_that_is_empty_once_cleaned_renders_as_none(rig: Rig) -> None:
    reasons = render_refusal("invalid_task", ["", "\x00 \t \n"], key="refusal-7-3-x")
    body, number = pull_request_body_for(rig, " \x00 ")

    assert refusal_bullets(reasons) == ["- _none_", "- _none_"]
    assert task_row(body) == f"| Task | #{number} _none_ |"


def test_a_hostile_title_reaches_the_real_pull_request_as_one_well_formed_row(rig: Rig) -> None:
    number = rig.new_task_issue(title=TITLE_WITH_EVERYTHING)

    run = rig.run_issue(number)

    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.pr_number is not None
    body = rig.fake.pulls[run.pr_number].body
    assert task_row(body) == (
        f"| Task | #{number} ``Fix a\\|b and `c` for @x [click](http://evil.example)`` |"
    )
    table = [line for line in body.splitlines() if line.startswith("|")]
    assert [len(gfm_cells(row)) for row in table] == [2] * len(table)
    live = outside_code(body)
    assert "@x" not in live and "evil.example" not in live


# ----------------------------------------------------------------------------------------------
# F6: the pull request recorded for a run is an open draft from the run's own branch
# ----------------------------------------------------------------------------------------------
DONE_HEADING = "Draft pull request ready for review"
BLOCKED_HEADING = "Agent run blocked"

# Pull requests that are not what the worker expects, as they stand when it finds them.
UNEXPECTED_PULLS = {
    "closed-and-not-a-draft": {"state": "closed", "draft": False},  # the reviewer's repro
    "open-and-ready-for-review": {"state": "open", "draft": False},
    "closed-draft": {"state": "closed", "draft": True},
}


def plant_pull(fake: FakeGitHub, branch: str, *, author: str, state: str, draft: bool) -> int:
    """A pull request from ``branch`` opened by ``author`` (the fake cannot create as another)."""
    title = f"Pull request by {author}"
    number = fake.add_issue(title=title, author=author, state=state, pull_request=True)
    fake.pulls[number] = FakePull(
        number=number,
        title=title,
        body="",
        head_ref=branch,
        base_ref="main",
        draft=draft,
        state=state,
        html_url=f"{WEB_URL}/{REPO}/pull/{number}",
        created_at=fake.now(),
    )
    return number


def plant_pull_after_push(rig: Rig, monkeypatch: pytest.MonkeyPatch, **pull: Any) -> list[int]:
    """Have a pull request appear for the run's branch the moment the bridge has pushed it."""
    push = rig.git.push
    planted: list[int] = []

    def push_then_plant(branch: str) -> None:
        push(branch)
        planted.append(plant_pull(rig.fake, branch, **pull))

    monkeypatch.setattr(rig.git, "push", push_then_plant)
    return planted


def pull_posts(rig: Rig) -> list[Any]:
    return [r for r in rig.fake.requests if r.method == "POST" and r.path.endswith("/pulls")]


def canned_pull(number: int, *, head: str) -> dict[str, Any]:
    """The JSON of an open draft pull request as a (misbehaving) server might answer a POST."""
    return {
        "number": number,
        "state": "open",
        "draft": True,
        "html_url": f"{WEB_URL}/{REPO}/pull/{number}",
        "title": "canned",
        "head": {"ref": head},
        "base": {"ref": "main"},
    }


def assert_pull_refused(rig: Rig, number: int, run: RunRecord, pull_number: int) -> None:
    """The run ended BLOCKED because pull request ``pull_number`` is not the draft it expects."""
    assert run.status is RunStatus.BLOCKED, (run.status, run.failure_kind, run.failure_detail)
    assert run.failure_kind == "pull_request_unexpected"
    assert f"pull request #{pull_number} " in (run.failure_detail or "")
    assert run.pr_number is None and run.pr_url is None
    labels = rig.fake.issue_labels(number)
    assert "agent-blocked" in labels and not {"agent-done", "agent-running"} & set(labels)
    labelled = [
        r
        for r in rig.fake.requests
        if r.method == "POST" and r.path.endswith(f"/issues/{pull_number}/labels")
    ]
    assert labelled == []  # the pull request was not claimed as the bridge's own
    comments = rig.comments(number)
    assert [c for c in comments if DONE_HEADING in c] == []
    (blocked,) = [c for c in comments if BLOCKED_HEADING in c]
    assert "`pull_request_unexpected`" in blocked and f"#{pull_number}" in blocked
    assert "re-approve" in blocked


@pytest.mark.parametrize("case_id", sorted(UNEXPECTED_PULLS))
def test_a_pull_request_someone_else_opened_for_the_branch_is_never_adopted(
    make_rig_for: Any, monkeypatch: pytest.MonkeyPatch, case_id: str
) -> None:
    rig: Rig = make_rig_for(pull_request={"labels": ["bridge-pr"]})
    planted = plant_pull_after_push(rig, monkeypatch, author="mallory", **UNEXPECTED_PULLS[case_id])
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    (foreign,) = planted
    assert_pull_refused(rig, number, run, foreign)
    assert rig.fake.issue_labels(foreign) == []
    assert pull_posts(rig) == []  # it found that pull request, so it never opened one of its own
    assert rig.git_env.origin_branches() == sorted(["main", run.branch or ""])  # pushed, as before


class SimulatedCrash(BaseException):
    """The bridge process dying: nothing in the worker catches ``BaseException``."""


def test_recovery_does_not_adopt_a_pull_request_someone_else_opened_either(
    make_rig_for: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig: Rig = make_rig_for(pull_request={"labels": ["bridge-pr"]})
    push = rig.git.push
    planted: list[int] = []

    def push_then_die(branch: str) -> None:
        push(branch)
        planted.append(plant_pull(rig.fake, branch, author="mallory", state="closed", draft=False))
        raise SimulatedCrash

    monkeypatch.setattr(rig.git, "push", push_then_die)
    number = rig.new_task_issue()
    with pytest.raises(SimulatedCrash):
        rig.poll()
    monkeypatch.undo()  # the replacement process has no such hook
    rig.clock.advance(rig.cfg.limits.lease_ttl_seconds + 1)  # the dead worker's lease runs out

    rig.worker.recover(execute=True)

    (foreign,) = planted
    (run,) = rig.ledger.runs_for_issue(REPO, number)
    assert_pull_refused(rig, number, run, foreign)
    assert rig.fake.issue_labels(foreign) == []


def test_a_pull_request_opened_ready_for_review_is_never_recorded_as_the_draft(
    make_rig_for: Any,
) -> None:
    rig: Rig = make_rig_for(pull_request={"labels": ["bridge-pr"]})
    rig.fake.ignore_draft = True  # a server that drops the ``draft: true`` the bridge sends
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    [(pull_number, pull)] = rig.fake.pulls.items()  # the bridge did open it, as asked
    assert pull.draft is False and pull.head_ref == run.branch
    assert_pull_refused(rig, number, run, pull_number)
    assert rig.fake.issue_labels(pull_number) == []
    assert len(pull_posts(rig)) == 1


def test_a_created_pull_request_from_another_branch_is_never_recorded(rig: Rig) -> None:
    rig.fake.inject("POST", r"/pulls$", status=201, json=canned_pull(77, head="agent/elsewhere"))
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    assert_pull_refused(rig, number, run, 77)
    assert "agent/elsewhere" in (run.failure_detail or "")


def test_an_open_draft_for_the_branch_from_an_earlier_attempt_is_still_adopted(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = plant_pull_after_push(
        rig, monkeypatch, author=rig.fake.bot_login, state="open", draft=True
    )
    number = rig.new_task_issue()

    run = rig.run_issue(number)

    (earlier,) = planted
    assert run.status is RunStatus.DONE, (run.failure_kind, run.failure_detail)
    assert run.pr_number == earlier and list(rig.fake.pulls) == [earlier]
    assert pull_posts(rig) == []  # adopted, not duplicated
    assert "agent-done" in rig.fake.issue_labels(number)
    assert len([c for c in rig.comments(number) if DONE_HEADING in c]) == 1


def test_the_fake_can_play_a_server_that_drops_the_draft_flag() -> None:
    """Guards the guard: the knob the tests above rely on must really change the answer."""
    fake = FakeGitHub(REPO)
    request = {"title": "t", "base": "main", "draft": True}
    with fake.client() as http:
        honoured = http.post(f"/repos/{REPO}/pulls", json={**request, "head": "agent/a"})
        fake.ignore_draft = True
        ignored = http.post(f"/repos/{REPO}/pulls", json={**request, "head": "agent/b"})

    assert (honoured.status_code, honoured.json()["draft"]) == (201, True)
    assert (ignored.status_code, ignored.json()["draft"]) == (201, False)
