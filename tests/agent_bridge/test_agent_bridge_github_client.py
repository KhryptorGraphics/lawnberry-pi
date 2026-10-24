"""``HttpGitHubClient`` against the in-process fake GitHub, plus the approval helpers.

Covers every client method, the retry / rate-limit / error-mapping rules, pagination safety, the
rule that the token only ever travels in the ``Authorization`` header, and the pure
``find_approval`` / ``edited_after`` decision tables the worker relies on for its approval gate.
"""

from __future__ import annotations

import inspect
import logging
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from bridgefakes.github_server import API_URL, FakeGitHub, FakeLabel, RecordedRequest, iso_z

from tools.agent_bridge.config import GitHubConfig, LabelsConfig
from tools.agent_bridge.errors import (
    GitHubConflict,
    GitHubError,
    GitHubNotFound,
    GitHubPermissionError,
    GitHubUnavailable,
)
from tools.agent_bridge.github import (
    LABEL_STYLE,
    GitHubClient,
    HttpGitHubClient,
    edited_after,
    find_approval,
)
from tools.agent_bridge.models import Approval, Comment, Issue, IssueEvent, PullRequest

REPO = "owner/repo"
TOKEN = "ghp_" + "Q7wE3rT9yU1iO5pA2sD4fG6hJ8kL0zXcVb"
USER_AGENT = "bridge-tests/9"
WALL_NOW = 1_800_000_000.0
ISSUES = "/repos/owner/repo/issues"
LABELS = "/repos/owner/repo/labels"
PULLS = "/repos/owner/repo/pulls"

ClientFactory = Callable[..., HttpGitHubClient]


# ------------------------------------------------------------------------------ fixtures
@dataclass
class Sleeper:
    """Injected ``sleep``: records the requested delays instead of waiting."""

    calls: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub(REPO, token=TOKEN)


@pytest.fixture
def sleeper() -> Sleeper:
    return Sleeper()


@pytest.fixture
def make_client(fake: FakeGitHub, sleeper: Sleeper) -> Iterator[ClientFactory]:
    """Factory for clients bound to the fake; config keywords override the test defaults."""
    created: list[HttpGitHubClient] = []

    def factory(
        *,
        token: str = TOKEN,
        repository: str = REPO,
        transport: httpx.BaseTransport | None = None,
        **config: Any,
    ) -> HttpGitHubClient:
        settings: dict[str, Any] = {
            "api_url": API_URL,
            "max_retries": 2,
            "retry_backoff_seconds": 0,
            "rate_limit_max_wait_seconds": 60,
            "user_agent": USER_AGENT,
            **config,
        }
        client = HttpGitHubClient(
            GitHubConfig(**settings),
            repository,
            token,
            transport=transport or fake.transport(),
            sleep=sleeper,
            wall_clock=lambda: WALL_NOW,
        )
        created.append(client)
        return client

    yield factory
    for client in created:
        client.close()


@pytest.fixture
def client(make_client: ClientFactory) -> HttpGitHubClient:
    return make_client()


def outcome_of(call: Callable[[], Any]) -> Any:
    """The value ``call`` returns, or the exception it raises (so facts can be asserted first)."""
    try:
        return call()
    except Exception as exc:
        return exc


def requests_to(fake: FakeGitHub, method: str, path: str) -> list[RecordedRequest]:
    return [r for r in fake.requests if r.method == method and r.path == path]


def event(
    event_id: int,
    kind: str = "labeled",
    label: str | None = "agent-approved",
    *,
    actor: str | None = "owner",
    at: str = "2026-10-24T10:00:00Z",
) -> IssueEvent:
    return IssueEvent(id=event_id, event=kind, actor=actor, created_at=at, label=label)


# -------------------------------------------------------------- list_candidate_issues
def test_list_candidate_issues_needs_every_label_and_only_lists_open_issues(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    both = fake.add_issue(title="both", labels=["agent-task", "agent-approved"])
    fake.add_issue(title="task only", labels=["agent-task"])
    fake.add_issue(title="approved only", labels=["agent-approved"])
    extra = fake.add_issue(title="extra", labels=["agent-task", "agent-approved", "bug"])
    fake.add_issue(title="unlabelled")
    closed = fake.add_issue(title="closed", labels=["agent-task", "agent-approved"])
    fake.close_issue(closed)

    found = client.list_candidate_issues(["agent-task", "agent-approved"])

    assert [i.number for i in found] == [both, extra]
    assert all(i.labels >= {"agent-task", "agent-approved"} and i.is_open for i in found)
    query = fake.requests[0].query
    assert query["labels"] == "agent-task,agent-approved"
    assert query["state"] == "open"

    fake.reopen_issue(closed)
    assert [i.number for i in client.list_candidate_issues(["agent-task", "agent-approved"])] == [
        both,
        extra,
        closed,
    ]
    assert client.list_candidate_issues(["no-such-label"]) == []


def test_list_candidate_issues_excludes_pull_requests(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    issue = fake.add_issue(title="real", labels=["agent-task"])
    seeded_pr = fake.add_issue(title="seeded pr", labels=["agent-task"], pull_request=True)
    opened = client.create_pull_request(title="via api", head="agent/x", base="main", body="b")
    client.add_labels(opened.number, ["agent-task"])
    assert fake.issues[opened.number].pull_request  # it is in the issues feed, flagged as a PR

    found = client.list_candidate_issues(["agent-task"])

    assert [i.number for i in found] == [issue]
    assert seeded_pr not in {i.number for i in found}
    assert not any(i.is_pull_request for i in found)


def test_list_candidate_issues_pages_through_link_headers_oldest_first(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    numbers = [fake.add_issue(title=f"t{i}", labels=["agent-task"]) for i in range(130)]

    found = client.list_candidate_issues(["agent-task"])

    assert [i.number for i in found] == numbers  # ascending, nothing lost or duplicated
    pages = requests_to(fake, "GET", ISSUES)
    assert [r.query.get("page") for r in pages] == [None, "2"]
    for page in pages:
        assert page.query["per_page"] == "100"
        assert page.query["labels"] == "agent-task"
        assert (page.query["sort"], page.query["direction"]) == ("created", "asc")


def test_list_candidate_issues_is_ordered_by_creation_not_by_issue_number(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    first = fake.add_issue(title="created first", labels=["agent-task"], number=50)
    second = fake.add_issue(title="created second", labels=["agent-task"], number=3)

    assert [i.number for i in client.list_candidate_issues(["agent-task"])] == [first, second]


def test_pagination_allows_exactly_thirty_pages_and_refuses_a_thirty_first(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    for _ in range(3000):
        fake.add_issue(title="t", labels=["agent-task"])
    assert len(client.list_candidate_issues(["agent-task"])) == 3000
    assert len(requests_to(fake, "GET", ISSUES)) == 30

    fake.add_issue(title="one too many", labels=["agent-task"])
    fake.reset_requests()
    with pytest.raises(GitHubError, match="too many pages"):
        client.list_candidate_issues(["agent-task"])
    assert len(requests_to(fake, "GET", ISSUES)) == 30  # it never asked for page 31


def test_endless_next_links_are_cut_off_after_thirty_requests(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    next_link = f'<{API_URL}{ISSUES}?page=2>; rel="next"'
    fake.inject("GET", r"/issues$", json=[], headers={"Link": next_link}, times=100)

    with pytest.raises(GitHubError, match="too many pages"):
        client.list_candidate_issues(["agent-task"])

    assert len(fake.requests) == 30


# ------------------------------------------------------------------------- get_issue
def test_get_issue_maps_every_field(fake: FakeGitHub, client: HttpGitHubClient) -> None:
    number = fake.add_issue(
        title="Add a note", body="Please do it", author="alice", labels=["agent-task", "bug"]
    )

    issue = client.get_issue(number)

    stored = fake.issues[number]
    assert issue == Issue(
        number=number,
        title="Add a note",
        body="Please do it",
        state="open",
        author="alice",
        labels=frozenset({"agent-task", "bug"}),
        state_reason=None,
        created_at=iso_z(stored.created_at),
        updated_at=iso_z(stored.updated_at),
        html_url=f"{fake.web_url}/owner/repo/issues/{number}",
        is_pull_request=False,
    )
    assert isinstance(issue.labels, frozenset)
    assert issue.is_open


def test_get_issue_reports_state_reason_of_closed_issues(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t")
    fake.close_issue(number, reason="not_planned")

    issue = client.get_issue(number)

    assert (issue.state, issue.state_reason, issue.is_open) == ("closed", "not_planned", False)


def test_get_issue_flags_pull_requests_and_turns_a_null_body_into_empty_text(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="a pr", pull_request=True)

    issue = client.get_issue(number)

    assert issue.is_pull_request
    assert issue.body == ""  # GitHub serialises an empty body as null
    assert issue.html_url.endswith(f"/pull/{number}")


def test_get_issue_tolerates_sparse_payloads_of_deleted_users(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    fake.inject(
        "GET",
        r"/issues/5$",
        json={
            "number": 5,
            "title": None,
            "body": None,
            "state": "closed",
            "state_reason": "completed",
            "user": None,
            "labels": ["plain", {"name": "object"}],
        },
    )

    issue = client.get_issue(5)

    assert (issue.title, issue.body, issue.author) == ("", "", "")
    assert issue.labels == frozenset({"plain", "object"})
    assert (issue.created_at, issue.updated_at, issue.html_url) == ("", "", "")
    assert issue.state_reason == "completed"


def test_get_issue_of_a_missing_issue_is_not_found(client: HttpGitHubClient) -> None:
    with pytest.raises(GitHubNotFound) as raised:
        client.get_issue(404)
    assert raised.value.status == 404


# ----------------------------------------------------------------------- issue_events
def test_issue_events_parses_labels_renames_and_state_changes_in_order(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="Old title", labels=["agent-task"], author="alice")
    fake.label_issue(number, "agent-approved", actor="owner")
    fake.edit_issue(number, title="New title", actor="alice")
    fake.unlabel_issue(number, "agent-approved", actor="owner")
    fake.close_issue(number, actor="owner", reason="not_planned")

    events = client.issue_events(number)

    assert [e.event for e in events] == ["labeled", "labeled", "renamed", "unlabeled", "closed"]
    assert [e.id for e in events] == sorted(e.id for e in events)
    assert [e.actor for e in events] == ["alice", "owner", "alice", "owner", "owner"]
    assert [e.label for e in events] == [
        "agent-task",
        "agent-approved",
        None,
        "agent-approved",
        None,
    ]
    renamed = events[2]
    assert (renamed.rename_from, renamed.rename_to) == ("Old title", "New title")
    stored = fake.events[number]
    assert [e.created_at for e in events] == [item["created_at"] for item in stored]


def test_issue_events_sorts_by_id_even_when_the_server_does_not(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    fake.add_issue(title="t")
    stamp = "2026-10-24T10:00:00Z"
    fake.inject(
        "GET",
        r"/issues/1/events$",
        json=[
            {"id": 9, "event": "closed", "actor": {"login": "a"}, "created_at": stamp},
            {"id": 3, "event": "labeled", "actor": {"login": "b"}, "created_at": stamp},
            {"id": 5, "event": "reopened", "actor": {"login": "c"}, "created_at": stamp},
        ],
    )

    assert [e.id for e in client.issue_events(1)] == [3, 5, 9]


def test_issue_events_keep_a_missing_actor_as_none(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    fake.add_issue(title="t")
    fake.inject(
        "GET",
        r"/issues/1/events$",
        json=[
            {"id": 1, "event": "labeled", "actor": None, "label": {"name": "agent-approved"}},
            {"id": 2, "event": "unlabeled", "label": {"name": "agent-approved"}},
            {"id": 3},
        ],
    )

    events = client.issue_events(1)

    assert [e.actor for e in events] == [None, None, None]
    assert events[0].label == "agent-approved" and events[0].created_at == ""
    assert events[2].event == ""


def test_issue_events_pages_through_every_event(fake: FakeGitHub, client: HttpGitHubClient) -> None:
    number = fake.add_issue(title="t")
    for _ in range(60):
        fake.label_issue(number, "flap")
        fake.unlabel_issue(number, "flap")

    events = client.issue_events(number)

    assert len(events) == 120
    assert [e.id for e in events] == sorted({e.id for e in events})
    assert len(requests_to(fake, "GET", f"{ISSUES}/{number}/events")) == 2


# ------------------------------------------------------------------ issue_last_edited_at
def test_issue_last_edited_at_is_none_until_the_text_is_edited(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", body="b")
    assert client.issue_last_edited_at(number) is None

    fake.edit_issue(number, body="b2")

    edited = fake.issues[number].last_edited_at
    assert edited is not None
    assert client.issue_last_edited_at(number) == iso_z(edited)
    sent = requests_to(fake, "POST", "/graphql")[-1].json
    assert isinstance(sent, dict)
    assert "lastEditedAt" in sent["query"]
    assert sent["variables"] == {"owner": "owner", "name": "repo", "number": number}


def test_issue_last_edited_at_raises_not_found_for_unknown_issues_and_pull_requests(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    pull = fake.add_issue(title="pr", pull_request=True)  # GraphQL repository.issue skips PRs
    for number in (999, pull):
        with pytest.raises(GitHubNotFound, match="GraphQL"):
            client.issue_last_edited_at(number)


def test_issue_last_edited_at_of_an_unresolvable_repository_is_not_found(
    fake: FakeGitHub, make_client: ClientFactory
) -> None:
    fake.add_issue(title="t")
    other = make_client(repository="other/elsewhere")

    with pytest.raises(GitHubNotFound, match="Could not resolve"):
        other.issue_last_edited_at(1)


def test_graphql_failures_are_reported_as_github_errors(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    fake.add_issue(title="t")
    fake.inject("POST", r"/graphql$", json=["not", "an", "object"])
    with pytest.raises(GitHubError, match="unexpected GraphQL response"):
        client.issue_last_edited_at(1)


@pytest.mark.parametrize(
    ("api_url", "graphql_url", "issue_url"),
    [
        (
            "https://api.github.test",
            "https://api.github.test/graphql",
            "https://api.github.test/repos/owner/repo/issues/3",
        ),
        (
            "https://ghes.example.test/api/v3",
            "https://ghes.example.test/api/graphql",
            "https://ghes.example.test/api/v3/repos/owner/repo/issues/3",
        ),
        (
            "https://ghes.example.test:8443/api/v3/",
            "https://ghes.example.test:8443/api/graphql",
            "https://ghes.example.test:8443/api/v3/repos/owner/repo/issues/3",
        ),
    ],
    ids=["github.com-style", "ghes", "ghes-with-port-and-trailing-slash"],
)
def test_graphql_endpoint_follows_the_api_flavour(
    make_client: ClientFactory, api_url: str, graphql_url: str, issue_url: str
) -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url).split("?")[0]))
        if request.method == "POST":
            data = {"data": {"repository": {"issue": {"lastEditedAt": "2026-10-24T10:00:05Z"}}}}
            return httpx.Response(200, json=data)
        return httpx.Response(200, json={"number": 3, "title": "t", "state": "open"})

    client = make_client(api_url=api_url, transport=httpx.MockTransport(handler))

    assert client.issue_last_edited_at(3) == "2026-10-24T10:00:05Z"
    assert client.get_issue(3).number == 3
    assert seen == [("POST", graphql_url), ("GET", issue_url)]


# ---------------------------------------------------------------------------- labels
def test_add_labels_posts_the_list_and_is_idempotent(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", labels=["agent-task"])

    client.add_labels(number, ["agent-running", "extra"])
    client.add_labels(number, ["agent-running"])

    assert fake.issue_labels(number) == ["agent-running", "agent-task", "extra"]
    posted = requests_to(fake, "POST", f"{ISSUES}/{number}/labels")
    assert [r.json for r in posted] == [
        {"labels": ["agent-running", "extra"]},
        {"labels": ["agent-running"]},
    ]


def test_add_labels_with_nothing_to_add_sends_no_request(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t")
    client.add_labels(number, [])
    assert fake.requests == []
    assert fake.issue_labels(number) == []


def test_add_labels_to_a_missing_issue_is_an_error(client: HttpGitHubClient) -> None:
    with pytest.raises(GitHubNotFound):
        client.add_labels(404, ["agent-running"])


def test_remove_label_removes_it_and_url_quotes_the_name(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", labels=["agent-running", "needs review/soon"])

    client.remove_label(number, "agent-running")
    client.remove_label(number, "needs review/soon")

    assert fake.issue_labels(number) == []
    deletes = [r for r in fake.requests if r.method == "DELETE"]
    assert deletes[0].raw_path.endswith("/labels/agent-running")
    assert deletes[1].raw_path.endswith("/labels/needs%20review%2Fsoon")


def test_remove_label_tolerates_a_label_that_is_not_on_the_issue(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", labels=["agent-task"])

    client.remove_label(number, "agent-running")  # GitHub answers 404; nothing to do

    assert fake.issue_labels(number) == ["agent-task"]
    assert [r.method for r in fake.requests] == ["DELETE"]  # a 404 is not retried


def test_remove_label_still_reports_permission_errors(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", labels=["agent-running"])
    fake.deny("DELETE", r"/labels/")

    with pytest.raises(GitHubPermissionError):
        client.remove_label(number, "agent-running")
    assert fake.issue_labels(number) == ["agent-running"]


def test_ensure_labels_creates_only_the_missing_ones_with_the_bridge_styles(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    fake.labels["agent-task"] = FakeLabel(name="agent-task", color="5319e7", id=1)

    created = client.ensure_labels(["agent-task", "agent-approved", "agent-running", "custom"])

    assert created == ["agent-approved", "agent-running", "custom"]
    for name in ("agent-approved", "agent-running"):
        color, description = LABEL_STYLE[name]
        assert (fake.labels[name].color, fake.labels[name].description) == (color, description)
    assert (fake.labels["custom"].color, fake.labels["custom"].description) == ("ededed", "")
    posts = requests_to(fake, "POST", LABELS)
    assert [r.json["name"] for r in posts if isinstance(r.json, dict)] == created

    fake.reset_requests()
    assert client.ensure_labels(["agent-task", "agent-approved"]) == []
    assert fake.writes() == []  # nothing missing, nothing written


def test_ensure_labels_reads_every_page_of_existing_labels(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    for i in range(120):
        fake.labels[f"a-{i:03d}"] = FakeLabel(name=f"a-{i:03d}", id=i + 1)
    fake.labels["agent-task"] = FakeLabel(name="agent-task", id=500)  # sorts onto page two

    assert client.ensure_labels(["agent-task"]) == []
    assert len(requests_to(fake, "GET", LABELS)) == 2
    assert fake.writes() == []


def test_ensure_labels_tolerates_losing_a_creation_race(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    already_exists = {
        "message": "Validation Failed",
        "errors": [{"resource": "Label", "code": "already_exists", "field": "name"}],
    }
    fake.inject("POST", r"/labels$", status=422, json=already_exists)

    created = client.ensure_labels(["agent-approved", "agent-running"])

    assert created == ["agent-running"]  # the first one "already existed" by the time we posted
    assert set(fake.labels) == {"agent-running"}


def test_ensure_labels_ignores_duplicate_names_in_the_request(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    assert client.ensure_labels(["agent-done", "agent-done"]) == ["agent-done"]
    assert len(requests_to(fake, "POST", LABELS)) == 2  # the second one drew a real 422


def test_ensure_labels_reports_permission_errors(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    fake.deny("POST", r"/labels$")
    with pytest.raises(GitHubPermissionError):
        client.ensure_labels(["agent-done"])


def test_label_styles_cover_the_configured_bridge_labels_with_valid_colours() -> None:
    assert set(LABEL_STYLE) == set(LabelsConfig().all())
    for color, description in LABEL_STYLE.values():
        assert len(color) == 6 and int(color, 16) >= 0
        assert description


# -------------------------------------------------------------------------- comments
def test_list_comments_pages_through_every_comment_in_order(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t")
    ids = [fake.add_comment(number, f"comment {i}", author=f"user{i}") for i in range(120)]

    comments = client.list_comments(number)

    assert [c.id for c in comments] == ids
    assert comments[0] == Comment(
        id=ids[0],
        body="comment 0",
        author="user0",
        created_at=iso_z(fake.comments[number][0].created_at),
    )
    assert comments[-1].body == "comment 119"
    assert len(requests_to(fake, "GET", f"{ISSUES}/{number}/comments")) == 2


def test_list_comments_of_a_missing_issue_is_not_found(client: HttpGitHubClient) -> None:
    with pytest.raises(GitHubNotFound):
        client.list_comments(404)


def test_create_comment_posts_the_body_and_returns_the_stored_comment(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t")

    comment = client.create_comment(number, "hello **world**\n<!-- marker -->")

    stored = fake.comments[number][0]
    assert comment == Comment(
        id=stored.id,
        body="hello **world**\n<!-- marker -->",
        author=fake.bot_login,
        created_at=iso_z(stored.created_at),
    )
    assert requests_to(fake, "POST", f"{ISSUES}/{number}/comments")[0].json == {
        "body": "hello **world**\n<!-- marker -->"
    }


def test_create_comment_errors_are_typed(fake: FakeGitHub, client: HttpGitHubClient) -> None:
    number = fake.add_issue(title="t")
    with pytest.raises(GitHubNotFound):
        client.create_comment(404, "x")
    with pytest.raises(GitHubConflict):
        client.create_comment(number, "x" * 70_000)  # over GitHub's 65536 character limit
    assert fake.comments[number] == []


# ----------------------------------------------------------------------- pull requests
def open_pull(
    client: HttpGitHubClient, head: str = "agent/issue-1-abc", **extra: Any
) -> PullRequest:
    params: dict[str, Any] = {"title": "[agent] t", "head": head, "base": "main", "body": "body"}
    return client.create_pull_request(**{**params, **extra})


def test_create_pull_request_is_always_a_draft(fake: FakeGitHub, client: HttpGitHubClient) -> None:
    pull = open_pull(client, title="[agent] Document it", body="Refs #1")

    assert pull == PullRequest(
        number=pull.number,
        html_url=f"{fake.web_url}/owner/repo/pull/{pull.number}",
        state="open",
        draft=True,
        head_ref="agent/issue-1-abc",
        base_ref="main",
        title="[agent] Document it",
    )
    sent = requests_to(fake, "POST", PULLS)[0].json
    assert sent == {
        "title": "[agent] Document it",
        "head": "agent/issue-1-abc",
        "base": "main",
        "body": "Refs #1",
        "draft": True,
    }
    assert fake.pulls[pull.number].draft is True


def test_create_pull_request_falls_back_to_the_existing_pull_request_on_duplicate(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    first = open_pull(client)

    again = open_pull(client, title="a retry after a lost reply")

    assert again.number == first.number
    assert again.title == first.title  # the original, not the retry's title
    assert len(fake.pulls) == 1
    assert [r.method for r in fake.requests] == ["POST", "POST", "GET"]  # create, 422, find


def test_create_pull_request_reraises_other_validation_failures(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    with pytest.raises(GitHubConflict) as raised:
        open_pull(client, title="")  # 422 missing title, and there is no PR to fall back to
    assert raised.value.status == 422
    assert fake.pulls == {}
    assert [r.method for r in fake.requests] == ["POST", "GET"]  # it did look for a PR first


def test_find_pull_request_filters_by_head_branch(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    wanted = open_pull(client, "agent/issue-1-aaa")
    open_pull(client, "agent/issue-2-bbb")

    found = client.find_pull_request("agent/issue-1-aaa")

    assert found is not None and found.number == wanted.number
    assert found.head_ref == "agent/issue-1-aaa"
    query = requests_to(fake, "GET", PULLS)[-1].query
    assert (query["head"], query["state"]) == ("owner:agent/issue-1-aaa", "all")
    assert client.find_pull_request("agent/issue-3-ccc") is None


def test_find_pull_request_prefers_an_open_pull_request_over_a_newer_closed_one(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    older_open = open_pull(client)
    newer_closed = open_pull(client, base="release")  # same head, other base: allowed
    fake.close_issue(newer_closed.number)
    assert newer_closed.number > older_open.number

    found = client.find_pull_request("agent/issue-1-abc")

    assert found is not None and found.number == older_open.number and found.state == "open"


def test_find_pull_request_returns_a_closed_one_when_that_is_all_there_is(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    pull = open_pull(client)
    fake.close_issue(pull.number)

    found = client.find_pull_request("agent/issue-1-abc")

    assert found is not None and (found.number, found.state) == (pull.number, "closed")


def test_find_pull_request_ignores_results_for_other_head_refs(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    wrong = {
        "number": 8,
        "state": "open",
        "head": {"ref": "somebody/else"},
        "base": {"ref": "main"},
    }
    fake.inject("GET", r"/pulls$", json=[wrong])

    assert client.find_pull_request("agent/issue-1-abc") is None


# -------------------------------------------------------------- the client protocol
def test_http_client_matches_the_protocol_the_worker_depends_on() -> None:
    methods = [
        name
        for name, member in vars(GitHubClient).items()
        if not name.startswith("_") and callable(member)
    ]
    assert set(methods) >= {
        "list_candidate_issues",
        "get_issue",
        "issue_events",
        "issue_last_edited_at",
        "add_labels",
        "remove_label",
        "ensure_labels",
        "list_comments",
        "create_comment",
        "find_pull_request",
        "create_pull_request",
        "close",
    }
    for name in methods:
        wanted = inspect.signature(getattr(GitHubClient, name)).parameters
        actual = inspect.signature(getattr(HttpGitHubClient, name)).parameters
        assert [(p.name, p.kind) for p in actual.values()] == [
            (p.name, p.kind) for p in wanted.values()
        ], name


# ------------------------------------------------------------ find_approval (pure)
APPROVED = "agent-approved"


@pytest.mark.parametrize(
    ("label", "events", "current", "expected"),
    [
        pytest.param(APPROVED, [], {APPROVED}, None, id="no-events"),
        pytest.param(
            APPROVED,
            [event(4, at="2026-10-24T10:00:04Z")],
            {"agent-task", APPROVED},
            Approval(4, "owner", "2026-10-24T10:00:04Z"),
            id="single-labeled-event",
        ),
        pytest.param(APPROVED, [event(4)], {"agent-task"}, None, id="label-no-longer-on-the-issue"),
        pytest.param(
            APPROVED,
            [
                event(2, actor="alice", at="2026-10-24T10:00:02Z"),
                event(3, "unlabeled"),
                event(7, actor="bob", at="2026-10-24T10:00:07Z"),
            ],
            {APPROVED},
            Approval(7, "bob", "2026-10-24T10:00:07Z"),
            id="latest-labeled-event-wins",
        ),
        pytest.param(
            APPROVED,
            [event(2), event(3, "unlabeled")],
            {APPROVED},
            None,
            id="unlabeled-after-labeled-fails-closed",
        ),
        pytest.param(
            APPROVED, [event(3, "unlabeled")], {APPROVED}, None, id="only-an-unlabeled-event"
        ),
        pytest.param(
            APPROVED,
            [
                event(7, actor="bob", at="2026-10-24T10:00:07Z"),
                event(5, "unlabeled"),
                event(2, actor="alice"),
            ],
            {APPROVED},
            Approval(7, "bob", "2026-10-24T10:00:07Z"),
            id="events-are-ordered-by-id-not-by-position",
        ),
        pytest.param(
            APPROVED,
            [
                event(1, at="2026-10-24T10:00:01Z"),
                event(2, label="agent-task"),
                event(3, "unlabeled", label="bug"),
                event(4, "renamed", label=None),
                event(5, "closed", label=None),
            ],
            {APPROVED},
            Approval(1, "owner", "2026-10-24T10:00:01Z"),
            id="other-labels-and-other-events-are-ignored",
        ),
        pytest.param(
            APPROVED,
            [event(3, actor=None, at="2026-10-24T10:00:03Z")],
            {APPROVED},
            Approval(3, "", "2026-10-24T10:00:03Z"),
            id="deleted-user-becomes-empty-actor",
        ),
        pytest.param(
            "ok-to-run",
            [event(1, label="ok-to-run", actor="carol"), event(2, label=APPROVED)],
            {"ok-to-run", APPROVED},
            Approval(1, "carol", "2026-10-24T10:00:00Z"),
            id="configured-label-name",
        ),
        pytest.param(
            APPROVED,
            [event(1), event(2, "unlabeled"), event(3, actor="dave", at="2026-10-24T10:00:03Z")],
            {APPROVED},
            Approval(3, "dave", "2026-10-24T10:00:03Z"),
            id="re-approval-after-withdrawal",
        ),
    ],
)
def test_find_approval_table(
    label: str,
    events: list[IssueEvent],
    current: set[str],
    expected: Approval | None,
) -> None:
    assert find_approval(events, label, frozenset(current)) == expected


# ------------------------------------------------------------- edited_after (pure)
APPROVAL = Approval(event_id=5, actor="owner", created_at="2026-10-24T10:00:10Z")
BEFORE, AT, AFTER = "2026-10-24T10:00:09Z", "2026-10-24T10:00:10Z", "2026-10-24T10:00:11Z"


def renamed(at: str, event_id: int = 6) -> IssueEvent:
    return event(event_id, "renamed", label=None, at=at)


@pytest.mark.parametrize(
    ("last_edited", "events", "stale"),
    [
        pytest.param(None, [], False, id="never-edited"),
        pytest.param("", [], False, id="empty-timestamp-counts-as-never-edited"),
        pytest.param(BEFORE, [], False, id="body-edited-before-approval"),
        pytest.param(AT, [], True, id="body-edited-at-the-moment-of-approval"),
        pytest.param(AFTER, [], True, id="body-edited-after-approval"),
        pytest.param("2027-01-01T00:00:00Z", [], True, id="edited-in-a-later-year"),
        pytest.param("2025-12-31T23:59:59Z", [], False, id="edited-in-an-earlier-year"),
        pytest.param(None, [renamed(BEFORE)], False, id="renamed-before-approval"),
        pytest.param(None, [renamed(AT)], True, id="renamed-at-the-moment-of-approval"),
        pytest.param(None, [renamed(AFTER)], True, id="renamed-after-approval"),
        pytest.param(
            None,
            [event(6, "labeled", at=AFTER), event(7, "closed", label=None, at=AFTER)],
            False,
            id="other-events-after-approval-are-not-edits",
        ),
        pytest.param(BEFORE, [renamed(AFTER)], True, id="old-body-edit-but-new-rename"),
        pytest.param(AFTER, [renamed(BEFORE)], True, id="old-rename-but-new-body-edit"),
        pytest.param(BEFORE, [renamed(BEFORE)], False, id="everything-predates-approval"),
    ],
)
def test_edited_after_table(last_edited: str | None, events: list[IssueEvent], stale: bool) -> None:
    assert edited_after(APPROVAL, events, last_edited) is stale


@pytest.mark.parametrize(
    ("edit_before", "edit_after", "stale"),
    [
        pytest.param(None, None, False, id="untouched"),
        pytest.param("body", None, False, id="body-edited-before-approval-only"),
        pytest.param("title", None, False, id="title-edited-before-approval-only"),
        pytest.param(None, "body", True, id="body-edited-after-approval"),
        pytest.param(None, "title", True, id="title-edited-after-approval"),
        pytest.param("body", "body", True, id="edited-both-before-and-after"),
    ],
)
def test_approval_gate_against_real_issue_history(
    fake: FakeGitHub,
    client: HttpGitHubClient,
    edit_before: str | None,
    edit_after: str | None,
    stale: bool,
) -> None:
    def edit(what: str | None, version: str) -> None:
        if what == "body":
            fake.edit_issue(number, body=f"body {version}")
        elif what == "title":
            fake.edit_issue(number, title=f"title {version}")

    number = fake.add_issue(title="title 0", body="body 0", labels=["agent-task"])
    edit(edit_before, "1")
    fake.label_issue(number, APPROVED, actor="owner")
    edit(edit_after, "2")

    issue = client.get_issue(number)
    events = client.issue_events(number)
    approval = find_approval(events, APPROVED, issue.labels)

    assert approval is not None and approval.actor == "owner"
    assert edited_after(approval, events, client.issue_last_edited_at(number)) is stale


def test_withdrawn_and_reapplied_approval_is_a_new_approval(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", labels=["agent-task"])
    first_event = fake.label_issue(number, APPROVED, actor="owner")
    fake.unlabel_issue(number, APPROVED, actor="owner")
    assert find_approval(client.issue_events(number), APPROVED, frozenset({"agent-task"})) is None

    second_event = fake.label_issue(number, APPROVED, actor="maintainer")
    current = client.get_issue(number).labels
    approval = find_approval(client.issue_events(number), APPROVED, current)

    assert second_event > first_event
    assert approval is not None
    assert (approval.event_id, approval.actor) == (second_event, "maintainer")


# ------------------------------------------------------------------------ resilience
def test_get_retries_connect_errors_and_then_succeeds(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    number = fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", exception=httpx.ConnectError("boom"), times=2)

    assert client.get_issue(number).number == number

    assert len(fake.requests) == 3
    assert sleeper.calls == []  # backoff is zero in the test configuration


def test_retry_backoff_doubles_and_is_capped_at_thirty_seconds(
    fake: FakeGitHub, make_client: ClientFactory, sleeper: Sleeper
) -> None:
    number = fake.add_issue(title="t")
    client = make_client(max_retries=4, retry_backoff_seconds=10)
    fake.inject("GET", r"/issues/1$", exception=httpx.ConnectError("boom"), times=4)

    assert client.get_issue(number).number == number

    assert sleeper.calls == [10.0, 20.0, 30.0, 30.0]


def test_retries_are_exhausted_into_github_unavailable(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", exception=httpx.ConnectError("boom"), times=10)

    with pytest.raises(GitHubUnavailable, match="cannot connect") as raised:
        client.get_issue(1)

    assert len(fake.requests) == 3  # max_retries=2 means three attempts in total
    assert isinstance(raised.value.__cause__, httpx.ConnectError)
    assert raised.value.status is None


def test_without_retries_a_single_failure_is_final(
    fake: FakeGitHub, make_client: ClientFactory
) -> None:
    client = make_client(max_retries=0)
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", exception=httpx.ConnectError("boom"))

    with pytest.raises(GitHubUnavailable):
        client.get_issue(1)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_get_retries_server_errors(fake: FakeGitHub, client: HttpGitHubClient, status: int) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", status=status)

    assert client.get_issue(1).number == 1
    assert len(fake.requests) == 2


@pytest.mark.parametrize("status", [502, 503])
def test_persistent_server_errors_surface_with_their_status(
    fake: FakeGitHub, client: HttpGitHubClient, status: int
) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", status=status, times=10)

    with pytest.raises(GitHubUnavailable) as raised:
        client.get_issue(1)

    assert raised.value.status == status
    assert len(fake.requests) == 3


@dataclass(frozen=True)
class WriteCall:
    """A client write, the POST route it uses, and a check that it took effect exactly once."""

    call: Callable[[HttpGitHubClient], Any]
    route: str
    applied_once: Callable[[FakeGitHub], bool]


WRITE_CALLS: dict[str, WriteCall] = {
    "create_comment": WriteCall(
        lambda c: c.create_comment(1, "hello"),
        r"/issues/1/comments$",
        lambda f: [c.body for c in f.comments[1]] == ["hello"],
    ),
    "add_labels": WriteCall(
        lambda c: c.add_labels(1, ["agent-running"]),
        r"/issues/1/labels$",
        lambda f: f.issue_labels(1) == ["agent-running"],
    ),
    "create_pull_request": WriteCall(
        lambda c: c.create_pull_request(title="t", head="agent/x", base="main", body="b"),
        r"/pulls$",
        lambda f: [p.head_ref for p in f.pulls.values()] == ["agent/x"],
    ),
    "ensure_labels": WriteCall(
        lambda c: c.ensure_labels(["agent-done"]),
        r"/labels$",
        lambda f: list(f.labels) == ["agent-done"],
    ),
}


@pytest.mark.parametrize("status", [500, 502, 503])
@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_posts_are_not_retried_after_a_server_error(
    fake: FakeGitHub, client: HttpGitHubClient, name: str, status: int
) -> None:
    write = WRITE_CALLS[name]
    fake.add_issue(title="t")
    fake.reset_requests()
    fake.inject("POST", write.route, status=status)

    with pytest.raises(GitHubUnavailable) as raised:
        write.call(client)

    assert raised.value.status == status
    assert len([r for r in fake.requests if r.method == "POST"]) == 1  # a single attempt
    assert not write.applied_once(fake)


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ConnectTimeout])
@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_posts_are_retried_when_the_connection_never_opened(
    fake: FakeGitHub, client: HttpGitHubClient, name: str, failure: type[httpx.TransportError]
) -> None:
    write = WRITE_CALLS[name]
    fake.add_issue(title="t")
    fake.inject("POST", write.route, exception=failure("no route to host"))

    write.call(client)  # the second attempt goes through

    assert len([r for r in fake.requests if r.method == "POST"]) == 2
    assert write.applied_once(fake)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError])
def test_an_ambiguous_network_failure_is_retried_for_reads_but_not_for_posts(
    fake: FakeGitHub, client: HttpGitHubClient, failure: type[httpx.TransportError]
) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", exception=failure("lost reply"))
    assert client.get_issue(1).number == 1  # a read is safe to repeat
    assert len(fake.requests) == 2

    fake.reset_requests()
    fake.inject("POST", r"/issues/1/comments$", exception=failure("lost reply"))
    with pytest.raises(GitHubUnavailable, match="network error"):
        client.create_comment(1, "maybe delivered")
    assert len(fake.requests) == 1  # the caller must verify before repeating a write


def test_delete_and_graphql_reads_are_repeated_after_transient_failures(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    number = fake.add_issue(title="t", labels=["agent-running"])
    fake.inject("DELETE", r"/labels/agent-running$", status=503)
    client.remove_label(number, "agent-running")
    assert fake.issue_labels(number) == []

    fake.inject("POST", r"/graphql$", status=502)
    fake.inject("POST", r"/graphql$", exception=httpx.ReadTimeout("slow"))
    assert client.issue_last_edited_at(number) is None
    assert len(requests_to(fake, "POST", "/graphql")) == 3


def test_rate_limited_requests_wait_for_retry_after_and_then_succeed(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    fake.add_issue(title="t")
    fake.inject(
        "GET",
        r"/issues/1$",
        status=403,
        json={"message": "You have exceeded a secondary rate limit."},
        headers={"Retry-After": "7"},
    )

    assert client.get_issue(1).number == 1

    assert sleeper.calls == [7.0]
    assert len(fake.requests) == 2


def test_http_429_is_a_rate_limit_too(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", status=429, headers={"Retry-After": "3"})

    assert client.get_issue(1).number == 1
    assert sleeper.calls == [3.0]


@pytest.mark.parametrize(
    ("reset_in", "waited"),
    [(30, 31.0), (0, 1.0), (-500, 1.0)],
    ids=["reset-in-30s", "reset-now", "reset-already-passed"],
)
def test_exhausted_quota_waits_until_the_reset_time(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper, reset_in: int, waited: float
) -> None:
    fake.add_issue(title="t")
    fake.inject(
        "GET",
        r"/issues/1$",
        status=403,
        json={"message": "API rate limit exceeded for user ID 1."},
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(WALL_NOW) + reset_in),
        },
    )

    assert client.get_issue(1).number == 1
    assert sleeper.calls == [waited]


@pytest.mark.parametrize(
    "message",
    [
        "You have exceeded a secondary rate limit.",
        "You have triggered an abuse detection mechanism.",
    ],
)
def test_a_rate_limit_message_without_headers_waits_a_minute(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper, message: str
) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", status=403, json={"message": message})

    assert client.get_issue(1).number == 1
    assert sleeper.calls == [60.0]  # exactly the configured maximum is still acceptable


@pytest.mark.parametrize(
    "headers",
    [
        {"Retry-After": "120"},
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(WALL_NOW) + 3600)},
    ],
    ids=["retry-after", "reset-header"],
)
def test_a_wait_above_the_configured_maximum_fails_without_sleeping(
    fake: FakeGitHub,
    client: HttpGitHubClient,
    sleeper: Sleeper,
    headers: dict[str, str],
) -> None:
    fake.add_issue(title="t")
    fake.inject(
        "GET",
        r"/issues/1$",
        status=403,
        json={"message": "API rate limit exceeded"},
        headers=headers,
    )

    with pytest.raises(GitHubUnavailable, match="rate limit") as raised:
        client.get_issue(1)

    assert sleeper.calls == []
    assert raised.value.status == 403
    assert len(fake.requests) == 1


def test_a_persistent_rate_limit_gives_up_after_the_configured_retries(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues/1$", status=429, headers={"Retry-After": "2"}, times=10)

    with pytest.raises(GitHubUnavailable, match="rate limit") as raised:
        client.get_issue(1)

    assert len(fake.requests) == 3
    assert sleeper.calls == [2.0, 2.0]  # no pointless sleep before the final failure
    assert raised.value.status == 429


def test_rate_limited_posts_are_repeated_because_github_did_not_process_them(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    number = fake.add_issue(title="t")
    fake.inject("POST", r"/comments$", status=403, headers={"Retry-After": "4"})

    client.create_comment(number, "once")

    assert sleeper.calls == [4.0]
    assert [c.body for c in fake.comments[number]] == ["once"]


def test_plain_permission_denials_are_not_retried_and_never_sleep(
    fake: FakeGitHub, client: HttpGitHubClient, sleeper: Sleeper
) -> None:
    number = fake.add_issue(title="t")
    fake.deny("POST", r"/comments$")  # 403 "Resource not accessible by personal access token"

    with pytest.raises(GitHubPermissionError, match="not accessible") as raised:
        client.create_comment(number, "x")

    assert raised.value.status == 403
    assert len(fake.requests) == 1 and sleeper.calls == []


def test_a_rejected_token_is_a_permission_error_without_retries(
    fake: FakeGitHub, make_client: ClientFactory, sleeper: Sleeper
) -> None:
    fake.add_issue(title="t")
    stranger = make_client(token="not-the-right-token")

    with pytest.raises(GitHubPermissionError, match="Bad credentials") as raised:
        stranger.get_issue(1)

    assert raised.value.status == 401
    assert len(fake.requests) == 1 and sleeper.calls == []
    assert "not-the-right-token" not in str(raised.value)


ERROR_MAPPING = [
    (401, GitHubPermissionError),
    (403, GitHubPermissionError),
    (404, GitHubNotFound),
    (409, GitHubConflict),
    (422, GitHubConflict),
    (400, GitHubError),
    (410, GitHubError),
    (418, GitHubError),
]


@pytest.mark.parametrize(("status", "expected"), ERROR_MAPPING)
@pytest.mark.parametrize("verb", ["GET", "POST"])
def test_http_errors_map_to_typed_exceptions_without_retrying(
    fake: FakeGitHub,
    client: HttpGitHubClient,
    sleeper: Sleeper,
    verb: str,
    status: int,
    expected: type[GitHubError],
) -> None:
    fake.add_issue(title="t")
    fake.inject(verb, r"/issues/1", status=status, json={"message": "server says no"})

    with pytest.raises(GitHubError) as raised:
        if verb == "GET":
            client.get_issue(1)
        else:
            client.create_comment(1, "x")

    assert type(raised.value) is expected
    assert raised.value.status == status
    assert "server says no" in str(raised.value)
    assert len(fake.requests) == 1 and sleeper.calls == []


def test_error_bodies_that_are_not_json_still_produce_a_typed_error(
    make_client: ClientFactory,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, content=b"<html>nope</html>")
    )
    client = make_client(transport=transport)

    with pytest.raises(GitHubNotFound, match="nope"):
        client.get_issue(1)


def test_invalid_json_in_a_success_response_is_a_github_error(make_client: ClientFactory) -> None:
    garbage = httpx.MockTransport(lambda request: httpx.Response(200, content=b"<html>oops</html>"))
    client = make_client(transport=garbage)
    calls: list[Callable[[], Any]] = [
        lambda: client.get_issue(1),
        lambda: client.list_candidate_issues(["agent-task"]),
        lambda: client.issue_events(1),
        lambda: client.issue_last_edited_at(1),
        lambda: client.create_comment(1, "x"),
        lambda: client.create_pull_request(title="t", head="h", base="main", body="b"),
    ]
    for call in calls:
        with pytest.raises(GitHubError, match="invalid JSON") as raised:
            call()
        assert raised.value.status == 200


def test_list_endpoints_refuse_a_json_object(fake: FakeGitHub, client: HttpGitHubClient) -> None:
    fake.add_issue(title="t")
    fake.inject("GET", r"/issues$", json={"message": "surprise"})
    with pytest.raises(GitHubError, match="expected a JSON list"):
        client.list_candidate_issues(["agent-task"])


FOREIGN_LINKS = [
    pytest.param("https://evil.example/repos/owner/repo/issues?page=2", id="other-host"),
    pytest.param("http://api.github.test/repos/owner/repo/issues?page=2", id="scheme-downgrade"),
    pytest.param(
        "https://api.github.test.evil.example/repos/owner/repo/issues?page=2", id="lookalike-suffix"
    ),
    pytest.param(
        "https://api.github.test@evil.example/repos/owner/repo/issues?page=2", id="userinfo-trick"
    ),
    pytest.param("https://api.github.test:8444/repos/owner/repo/issues?page=2", id="other-port"),
]


@pytest.mark.parametrize("link", FOREIGN_LINKS)
def test_a_pagination_link_to_another_origin_is_refused_and_never_requested(
    fake: FakeGitHub, client: HttpGitHubClient, link: str
) -> None:
    fake.inject("GET", r"/issues$", json=[], headers={"Link": f'<{link}>; rel="next"'})

    outcome = outcome_of(lambda: client.list_candidate_issues(["agent-task"]))

    hosts = [r.headers["host"] for r in fake.requests]
    assert hosts == ["api.github.test"], f"the bearer token was sent to {hosts[1:]}"
    assert isinstance(outcome, GitHubError)
    assert "another host" in str(outcome)


def test_pagination_stays_on_the_ghes_api_prefix(make_client: ClientFactory) -> None:
    pages: dict[str, httpx.Response] = {}
    base = "https://ghes.example.test/api/v3"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "page=2" in url:
            return pages["second"]
        return pages["first"]

    pages["first"] = httpx.Response(
        200,
        json=[{"number": 1}],
        headers={"Link": f'<{base}/repos/owner/repo/issues?page=2>; rel="next"'},
    )
    pages["second"] = httpx.Response(200, json=[{"number": 2}])
    client = make_client(api_url=base, transport=httpx.MockTransport(handler))

    assert [i.number for i in client.list_candidate_issues(["agent-task"])] == [1, 2]


# ----------------------------------------------------------------- headers and transport
def exercise_every_endpoint(fake: FakeGitHub, client: HttpGitHubClient) -> None:
    number = fake.add_issue(title="t", body="b", labels=["agent-task", "agent-running"])
    client.list_candidate_issues(["agent-task"])
    client.get_issue(number)
    client.issue_events(number)
    client.issue_last_edited_at(number)
    client.add_labels(number, ["agent-done"])
    client.remove_label(number, "agent-running")
    client.ensure_labels(["agent-blocked"])
    client.list_comments(number)
    client.create_comment(number, "hello")
    client.find_pull_request("agent/x")
    client.create_pull_request(title="t", head="agent/x", base="main", body="b")


def test_every_request_carries_auth_api_version_and_user_agent(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    exercise_every_endpoint(fake, client)

    assert {r.method for r in fake.requests} == {"GET", "POST", "DELETE"}
    assert len(fake.requests) >= 11
    for request in fake.requests:
        assert request.headers["authorization"] == f"Bearer {TOKEN}", request.path
        assert request.headers["x-github-api-version"] == "2022-11-28", request.path
        assert request.headers["user-agent"] == USER_AGENT, request.path
        assert request.headers["accept"] == "application/vnd.github+json", request.path
        assert request.headers["host"] == "api.github.test", request.path
        if request.method == "POST":
            assert request.headers["content-type"] == "application/json"


def test_the_default_user_agent_comes_from_the_configuration_defaults(fake: FakeGitHub) -> None:
    fake.add_issue(title="t")
    client = HttpGitHubClient(
        GitHubConfig(api_url=API_URL), REPO, TOKEN, transport=fake.transport()
    )
    try:
        client.get_issue(1)
    finally:
        client.close()

    assert fake.requests[0].headers["user-agent"] == GitHubConfig().user_agent
    assert GitHubConfig().user_agent.startswith("lawnberry-agent-bridge/")


def test_the_token_is_only_ever_sent_in_the_authorization_header(
    fake: FakeGitHub, client: HttpGitHubClient
) -> None:
    exercise_every_endpoint(fake, client)

    for request in fake.requests:
        assert TOKEN not in request.path
        assert TOKEN not in str(request.query)
        assert TOKEN not in str(request.json)
        leaked = [k for k, v in request.headers.items() if TOKEN in v and k != "authorization"]
        assert leaked == []


def test_the_request_timeout_comes_from_the_configuration(make_client: ClientFactory) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json={"number": 1})

    client = make_client(request_timeout_seconds=7.5, transport=httpx.MockTransport(handler))
    client.get_issue(1)

    assert seen == [{"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}]


def test_close_releases_the_underlying_transport(make_client: ClientFactory) -> None:
    class Spy(httpx.MockTransport):
        closed = False

        def close(self) -> None:
            self.closed = True

    spy = Spy(lambda request: httpx.Response(200, json={}))
    client = make_client(transport=spy)
    client.close()
    assert spy.closed


REDIRECT_CALLS: dict[str, tuple[Callable[[HttpGitHubClient], Any], str, str]] = {
    "get_issue": (lambda c: c.get_issue(1), "GET", r"/issues/1$"),
    "list_candidate_issues": (
        lambda c: c.list_candidate_issues(["agent-task"]),
        "GET",
        r"/issues$",
    ),
    "create_comment": (lambda c: c.create_comment(1, "x"), "POST", r"/comments$"),
    "add_labels": (lambda c: c.add_labels(1, ["agent-running"]), "POST", r"/issues/1/labels$"),
    "create_pull_request": (
        lambda c: c.create_pull_request(title="t", head="agent/x", base="main", body="b"),
        "POST",
        r"/pulls$",
    ),
    "remove_label": (lambda c: c.remove_label(1, "agent-running"), "DELETE", r"/labels/"),
    "issue_last_edited_at": (lambda c: c.issue_last_edited_at(1), "POST", r"/graphql$"),
}


@pytest.mark.parametrize("status", [301, 302, 307, 308])
@pytest.mark.parametrize("name", sorted(REDIRECT_CALLS))
def test_redirects_are_not_followed_and_surface_as_errors(
    fake: FakeGitHub, client: HttpGitHubClient, name: str, status: int
) -> None:
    call, verb, route = REDIRECT_CALLS[name]
    fake.add_issue(title="t", labels=["agent-task", "agent-running"])
    fake.reset_requests()
    fake.inject(
        verb,
        route,
        status=status,
        json={"message": "Moved Permanently"},
        headers={"Location": "https://evil.example/steal"},
    )

    outcome = outcome_of(lambda: call(client))

    hosts = [r.headers["host"] for r in fake.requests]
    assert hosts == ["api.github.test"], "a redirect must never be followed"
    assert isinstance(outcome, GitHubError), f"a {status} must be a GitHubError, got {outcome!r}"
    assert not isinstance(outcome, GitHubNotFound)  # a moved repository is not a missing label


# --------------------------------------------------------------------- token hygiene
def run_failure_battery(
    fake: FakeGitHub, make_client: ClientFactory, client: HttpGitHubClient
) -> list[BaseException]:
    """Provoke every kind of failure the client can raise and return the exceptions."""
    number = fake.add_issue(title="t", labels=["agent-task"])
    failures: list[BaseException] = []

    def attempt(call: Callable[[], Any]) -> None:
        failures.append(outcome_of(call))

    attempt(lambda: client.get_issue(404))
    fake.deny("POST", r"/comments$")
    attempt(lambda: client.create_comment(number, "x"))
    fake.allow("POST", r"/comments$")
    fake.inject("GET", rf"/issues/{number}$", status=422, json={"message": "Validation Failed"})
    attempt(lambda: client.get_issue(number))
    fake.inject("GET", rf"/issues/{number}$", status=503, times=10)
    attempt(lambda: client.get_issue(number))
    fake.inject("GET", rf"/issues/{number}$", exception=httpx.ConnectError("refused"), times=10)
    attempt(lambda: client.get_issue(number))
    fake.inject("GET", rf"/issues/{number}$", status=429, headers={"Retry-After": "999"})
    attempt(lambda: client.get_issue(number))
    fake.inject(
        "GET", r"/issues$", json=[], headers={"Link": '<https://evil.example/x>; rel="next"'}
    )
    attempt(lambda: client.list_candidate_issues(["agent-task"]))
    broken = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"{")))
    attempt(lambda: broken.get_issue(number))
    wrong_token = make_client(token="a-different-secret-value")
    attempt(lambda: wrong_token.get_issue(number))
    return failures


def test_the_token_never_appears_in_repr_exceptions_or_logs(
    fake: FakeGitHub,
    make_client: ClientFactory,
    client: HttpGitHubClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    failures = run_failure_battery(fake, make_client, client)
    exercise_every_endpoint(fake, client)

    assert repr(client) == "HttpGitHubClient(owner/repo)" == str(client)
    assert all(isinstance(f, GitHubError) for f in failures), failures
    assert len(failures) == 9
    for failure in failures:
        rendered = "".join(traceback.format_exception(failure))
        assert TOKEN not in rendered
        assert TOKEN not in repr(failure) and TOKEN not in str(failure)
        assert all(TOKEN not in str(arg) for arg in failure.args)
    assert caplog.records, "the log capture saw nothing, so it proves nothing"
    assert TOKEN not in caplog.text
    assert all(TOKEN not in str(record.args) for record in caplog.records)
