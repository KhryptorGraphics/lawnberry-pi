"""Tests for the in-process fake GitHub server used by the agent-bridge suites."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from bridgefakes.github_server import (
    MAX_BODY_LENGTH,
    FakeComment,
    FakeGitHub,
    FakeIssue,
    FakePull,
    RecordedRequest,
    iso_z,
)

REPO = "/repos/owner/repo"
API = "https://api.github.test"
LAST_EDITED_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) {"
    " repository(owner: $owner, name: $name) { issue(number: $number) { lastEditedAt } } }"
)


@pytest.fixture
def fake() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
def client(fake: FakeGitHub) -> Iterator[httpx.Client]:
    with fake.client() as http:
        yield http


def numbers(response: httpx.Response) -> list[int]:
    """Issue/PR numbers of a list response, in order."""
    return [item["number"] for item in response.json()]


def names(items: list[dict[str, Any]]) -> list[str]:
    return [item["name"] for item in items]


def walk(http: httpx.Client, first: httpx.Response) -> list[list[dict[str, Any]]]:
    """Follow ``rel="next"`` links from ``first`` and return the JSON of every page."""
    pages = [first.json()]
    response = first
    while "next" in response.links:
        response = http.get(response.links["next"]["url"])
        pages.append(response.json())
    return pages


def graphql(
    http: httpx.Client,
    number: int,
    *,
    owner: str = "owner",
    name: str = "repo",
    query: str = LAST_EDITED_QUERY,
) -> httpx.Response:
    variables = {"owner": owner, "name": name, "number": number}
    return http.post("/graphql", json={"query": query, "variables": variables})


def open_pull(http: httpx.Client, head: str = "agent/issue-1", base: str = "main", **extra: Any):
    return http.post(f"{REPO}/pulls", json={"title": "Fix it", "head": head, "base": base, **extra})


# --------------------------------------------------------------------------- clock


def test_clock_starts_at_start_and_advances_one_second_per_call(fake: FakeGitHub) -> None:
    first = fake.now()
    assert first == datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)
    assert fake.now() == first + timedelta(seconds=1)
    assert fake.now() == first + timedelta(seconds=2)
    assert first.utcoffset() == timedelta(0)


def test_clock_accepts_custom_aware_and_naive_starts_and_tick() -> None:
    fake = FakeGitHub(start=datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert iso_z(fake.now()) == "2030-01-02T03:04:05Z"
    fake.tick()  # default skips nothing
    assert iso_z(fake.now()) == "2030-01-02T03:04:06Z"
    fake.tick(60)
    assert iso_z(fake.now()) == "2030-01-02T03:05:07Z"
    with pytest.raises(ValueError, match="backwards"):
        fake.tick(-1)

    naive = FakeGitHub(start=datetime(2030, 1, 2))
    assert iso_z(naive.now()) == "2030-01-02T00:00:00Z"
    offset = FakeGitHub(start=datetime(2030, 1, 2, 2, 0, 0, tzinfo=timezone(timedelta(hours=2))))
    assert iso_z(offset.now()) == "2030-01-02T00:00:00Z"


def test_each_state_change_takes_one_tick_and_reads_take_none(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t", labels=["a"])  # created 10:00:00, labeled 10:00:01
    issue = client.get(f"{REPO}/issues/{n}").json()
    assert issue["created_at"] == "2026-10-24T10:00:00Z"
    assert issue["updated_at"] == "2026-10-24T10:00:01Z"
    assert client.get(f"{REPO}/issues/{n}/events").json()[0]["created_at"] == "2026-10-24T10:00:01Z"

    client.get(f"{REPO}/issues")
    client.get(f"{REPO}/labels")
    fake.add_comment(n, "hi")  # 10:00:02, unaffected by the reads above
    assert client.get(f"{REPO}/issues/{n}").json()["updated_at"] == "2026-10-24T10:00:02Z"
    assert iso_z(fake.now()) == "2026-10-24T10:00:03Z"


# ------------------------------------------------------------------ seeding helpers


def test_issue_numbers_auto_increment_and_accept_explicit_numbers(fake: FakeGitHub) -> None:
    assert fake.add_issue(title="a") == 1
    assert fake.add_issue(title="b") == 2
    assert fake.add_issue(title="c", number=10) == 10
    assert fake.add_issue(title="d") == 11
    assert sorted(fake.issues) == [1, 2, 10, 11]
    with pytest.raises(ValueError, match="already in use"):
        fake.add_issue(title="dup", number=10)
    with pytest.raises(ValueError, match="state"):
        fake.add_issue(title="bad", state="weird")
    assert fake.add_issue(title="e") == 12


def test_seeded_state_uses_public_dataclasses(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="t", body="b", author="alice", state="closed")
    comment_id = fake.add_comment(n, "hello", author="bob")
    open_pull(client)
    client.get(f"{REPO}/issues")
    issue = fake.issues[n]
    assert isinstance(issue, FakeIssue)
    assert (issue.title, issue.body, issue.author, issue.state) == ("t", "b", "alice", "closed")
    assert issue.state_reason == "completed"
    assert issue.last_edited_at is None
    assert issue.pull_request is False
    comment = fake.comments[n][0]
    assert isinstance(comment, FakeComment)
    assert (comment.id, comment.body, comment.author) == (comment_id, "hello", "bob")
    pull = fake.pulls[2]
    assert isinstance(pull, FakePull)
    assert (pull.head_ref, pull.base_ref, pull.state, pull.draft) == (
        "agent/issue-1",
        "main",
        "open",
        False,
    )
    assert isinstance(fake.requests[0], RecordedRequest)


def test_add_issue_labels_get_labeled_events_by_the_author(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t", author="alice", labels=["bug", "agent:approved", "bug"])
    events = client.get(f"{REPO}/issues/{n}/events").json()
    assert [(e["event"], e["label"]["name"], e["actor"]["login"]) for e in events] == [
        ("labeled", "bug", "alice"),
        ("labeled", "agent:approved", "alice"),
    ]
    assert fake.issue_labels(n) == ["agent:approved", "bug"]
    assert fake.issues[n].labels == ["agent:approved", "bug"]
    assert fake.issue_labels(fake.add_issue(title="single", labels="agent:approved")) == [
        "agent:approved"
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda f: f.label_issue(99, "x"),
        lambda f: f.unlabel_issue(99, "x"),
        lambda f: f.edit_issue(99, title="x"),
        lambda f: f.close_issue(99),
        lambda f: f.reopen_issue(99),
        lambda f: f.add_comment(99, "x"),
        lambda f: f.issue_labels(99),
    ],
)
def test_seeding_helpers_reject_unknown_issue_numbers(
    fake: FakeGitHub, call: Callable[[FakeGitHub], object]
) -> None:
    with pytest.raises(KeyError, match="no such issue #99"):
        call(fake)


def test_fake_can_target_another_repository_and_rejects_bad_names() -> None:
    fake = FakeGitHub("acme/widgets")
    n = fake.add_issue(title="t")
    with fake.client() as http:
        issue = http.get(f"/repos/acme/widgets/issues/{n}").json()
        assert issue["html_url"] == "https://github.test/acme/widgets/issues/1"
        assert issue["author_association"] == "NONE"
        assert http.get(f"{REPO}/issues").status_code == 404
    for bad in ("owner", "/repo", "owner/", "a/b/c"):
        with pytest.raises(ValueError, match="owner/name"):
            FakeGitHub(bad)


# ------------------------------------------------------------------- auth, headers


def test_missing_or_invalid_credentials_get_401_without_touching_state(
    fake: FakeGitHub,
) -> None:
    n = fake.add_issue(title="t")
    attempts = [
        fake.client(token=None),
        fake.client(token="wrong"),
        fake.client(token=None, headers={"Authorization": "Basic dGVzdC10b2tlbg=="}),
        fake.client(token=None, headers={"Authorization": "Bearer"}),
    ]
    for http in attempts:
        response = http.get(f"{REPO}/issues/{n}")
        assert response.status_code == 401
        assert response.json() == {"message": "Bad credentials"}
    response = attempts[0].post(f"{REPO}/issues/{n}/comments", json={"body": "x"})
    assert response.status_code == 401
    assert fake.comments[n] == []
    assert graphql(attempts[1], n).status_code == 401


def test_bearer_and_token_schemes_are_accepted(fake: FakeGitHub) -> None:
    for scheme in ("Bearer", "bearer", "token"):
        http = fake.client(token=None, headers={"Authorization": f"{scheme} test-token"})
        assert http.get(f"{REPO}/issues").status_code == 200


def test_manual_client_over_the_raw_transport_works(fake: FakeGitHub) -> None:
    fake.add_issue(title="t")
    with httpx.Client(
        base_url=API,
        transport=fake.transport(),
        headers={"Authorization": "token test-token"},
    ) as http:
        assert numbers(http.get(f"{REPO}/issues")) == [1]


def test_server_without_a_token_accepts_anonymous_requests() -> None:
    fake = FakeGitHub(token=None)
    with fake.client(token=None) as http:
        assert http.get(f"{REPO}/issues").status_code == 200
    assert "authorization" not in fake.requests[0].headers


@pytest.mark.parametrize(
    ("path", "token", "status"),
    [
        (f"{REPO}/issues", "test-token", 200),
        (f"{REPO}/issues/99", "test-token", 404),
        ("/nope", "test-token", 404),
        (f"{REPO}/issues", "bad-token", 401),
    ],
)
def test_every_response_carries_rate_limit_and_json_headers(
    fake: FakeGitHub, path: str, token: str, status: int
) -> None:
    response = fake.client(token=token).get(path)
    assert response.status_code == status
    assert response.headers["x-ratelimit-limit"] == "5000"
    assert response.headers["x-ratelimit-remaining"] == "4999"
    start = datetime(2026, 10, 24, 10, tzinfo=UTC)
    assert int(response.headers["x-ratelimit-reset"]) > int(start.timestamp())
    assert response.headers["content-type"] == "application/json; charset=utf-8"


def test_client_helper_targets_the_api_url_and_merges_headers(fake: FakeGitHub) -> None:
    assert fake.api_url == "https://api.github.test"
    with fake.client(headers={"X-Trace": "abc"}, timeout=3.0) as http:
        assert str(http.base_url).rstrip("/") == fake.api_url
        http.get(f"{REPO}/issues")
    headers = fake.requests[0].headers
    assert headers["authorization"] == "Bearer test-token"
    assert headers["x-trace"] == "abc"
    assert headers["accept"] == "application/vnd.github+json"


# ---------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "path",
    [
        "/nope",
        "/repos/owner/repo",
        "/repos/owner/repo/unknown",
        "/repos/other/repo/issues",
        "/repos/owner/other/issues/1",
        f"{REPO}/issues/abc",
        f"{REPO}/issues/1/unknown",
        f"{REPO}/issues/",
    ],
)
def test_unknown_routes_and_other_repositories_are_404(client: httpx.Client, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert response.json() == {"message": "Not Found"}


def test_repository_path_is_case_insensitive(fake: FakeGitHub, client: httpx.Client) -> None:
    fake.add_issue(title="t")
    assert numbers(client.get("/repos/OWNER/Repo/issues")) == [1]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("DELETE", f"{REPO}/issues"),
        ("POST", f"{REPO}/issues"),
        ("PATCH", f"{REPO}/issues/1"),
        ("POST", f"{REPO}/issues/1/events"),
        ("GET", f"{REPO}/issues/1/labels"),
        ("PUT", f"{REPO}/issues/1/labels"),
        ("GET", f"{REPO}/issues/1/labels/bug"),
        ("DELETE", f"{REPO}/pulls/1"),
        ("GET", "/graphql"),
    ],
)
def test_wrong_method_on_a_known_route_is_405(client: httpx.Client, method: str, path: str) -> None:
    response = client.request(method, path)
    assert response.status_code == 405
    assert response.headers["allow"]


# ---------------------------------------------------------------------- issues


def test_get_issue_returns_github_shape(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="Mow the lawn", body="front yard", author="alice", labels=["bug"])
    fake.add_comment(n, "first")
    data = client.get(f"{REPO}/issues/{n}").json()
    assert data["number"] == n
    assert data["title"] == "Mow the lawn"
    assert data["body"] == "front yard"
    assert data["state"] == "open"
    assert data["state_reason"] is None
    assert data["user"]["login"] == "alice"
    assert [(label["name"], label["color"]) for label in data["labels"]] == [("bug", "ededed")]
    assert data["comments"] == 1
    assert data["html_url"] == "https://github.test/owner/repo/issues/1"
    assert data["created_at"] == "2026-10-24T10:00:00Z"
    assert data["closed_at"] is None
    assert "pull_request" not in data


def test_get_issue_unknown_number_is_404_and_empty_body_is_null(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="no description")
    assert client.get(f"{REPO}/issues/{n}").json()["body"] is None
    response = client.get(f"{REPO}/issues/{n + 1}")
    assert response.status_code == 404
    assert response.json() == {"message": "Not Found"}


def test_list_issues_defaults_to_open_newest_first_and_honours_state_and_direction(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    a = fake.add_issue(title="a")
    b = fake.add_issue(title="b", state="closed")
    c = fake.add_issue(title="c")
    assert numbers(client.get(f"{REPO}/issues")) == [c, a]
    assert numbers(client.get(f"{REPO}/issues", params={"state": "open"})) == [c, a]
    assert numbers(client.get(f"{REPO}/issues", params={"state": "closed"})) == [b]
    assert numbers(client.get(f"{REPO}/issues", params={"state": "all"})) == [c, b, a]
    ascending = {"state": "all", "direction": "asc"}
    assert numbers(client.get(f"{REPO}/issues", params=ascending)) == [a, b, c]


def test_list_issues_sorts_by_updated_and_comment_count(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    a = fake.add_issue(title="a")
    b = fake.add_issue(title="b")
    c = fake.add_issue(title="c")
    fake.add_comment(a, "bump")
    assert numbers(client.get(f"{REPO}/issues", params={"sort": "updated"})) == [a, c, b]
    fake.add_comment(b, "one")
    fake.add_comment(b, "two")
    assert numbers(client.get(f"{REPO}/issues", params={"sort": "comments"})) == [b, a, c]


@pytest.mark.parametrize(
    "params",
    [{"state": "bogus"}, {"sort": "bogus"}, {"direction": "sideways"}],
)
def test_list_issues_rejects_invalid_enum_values(
    client: httpx.Client, params: dict[str, str]
) -> None:
    response = client.get(f"{REPO}/issues", params=params)
    assert response.status_code == 422
    assert response.json()["message"] == "Validation Failed"
    assert response.json()["errors"][0]["field"] == next(iter(params))


def test_label_filter_requires_all_labels(fake: FakeGitHub, client: httpx.Client) -> None:
    both = fake.add_issue(title="both", labels=["agent:approved", "bug"])
    approved = fake.add_issue(title="approved", labels=["agent:approved"])
    fake.add_issue(title="bug", labels=["bug"])
    fake.add_issue(title="none")
    all_of = client.get(f"{REPO}/issues", params={"labels": "agent:approved,bug"})
    assert numbers(all_of) == [both]
    assert numbers(client.get(f"{REPO}/issues", params={"labels": "agent:approved"})) == [
        approved,
        both,
    ]
    assert numbers(client.get(f"{REPO}/issues", params={"labels": "agent:approved,nope"})) == []
    assert len(client.get(f"{REPO}/issues", params={"labels": ""}).json()) == 4


def test_list_issues_includes_pull_requests_marked_with_pull_request_key(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    issue = fake.add_issue(title="issue")
    seeded = fake.add_issue(title="seeded pr", pull_request=True)
    opened = open_pull(client).json()["number"]
    items = {i["number"]: i for i in client.get(f"{REPO}/issues", params={"state": "all"}).json()}
    assert sorted(items) == [issue, seeded, opened]
    assert "pull_request" not in items[issue]
    assert items[seeded]["pull_request"]["url"] == f"{API}{REPO}/pulls/{seeded}"
    assert (
        items[opened]["pull_request"]["html_url"] == f"https://github.test/owner/repo/pull/{opened}"
    )
    assert items[opened]["html_url"] == f"https://github.test/owner/repo/pull/{opened}"
    assert items[opened]["title"] == "Fix it"


def test_issue_list_pagination_link_headers_and_page_walking(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    for i in range(7):
        fake.add_issue(title=f"issue {i}", labels=["x"])
    first = client.get(f"{REPO}/issues", params={"labels": "x", "per_page": 3})
    base = f"{API}{REPO}/issues?labels=x&per_page=3"
    assert numbers(first) == [7, 6, 5]
    assert set(first.links) == {"next", "last"}
    assert first.links["next"]["url"] == f"{base}&page=2"
    assert first.links["last"]["url"] == f"{base}&page=3"

    pages = walk(client, first)
    assert [[i["number"] for i in page] for page in pages] == [[7, 6, 5], [4, 3, 2], [1]]

    middle = client.get(f"{REPO}/issues", params={"labels": "x", "per_page": 3, "page": 2})
    assert numbers(middle) == [4, 3, 2]
    assert set(middle.links) == {"next", "last"}
    assert middle.links["next"]["url"] == f"{base}&page=3"

    last = client.get(first.links["last"]["url"])
    assert numbers(last) == [1]
    assert "link" not in last.headers

    single = client.get(f"{REPO}/issues", params={"labels": "x", "per_page": 50})
    assert "link" not in single.headers
    beyond = client.get(f"{REPO}/issues", params={"labels": "x", "per_page": 3, "page": 9})
    assert beyond.json() == []


def test_link_header_preserves_and_encodes_the_original_query(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    for _ in range(3):
        fake.add_issue(title="x", labels=["agent:approved", "bug"])
    fake.add_issue(title="closed", labels=["agent:approved", "bug"], state="closed")
    params = {"labels": "agent:approved,bug", "per_page": 2, "state": "all", "direction": "asc"}
    first = client.get(f"{REPO}/issues", params=params)
    expected = (
        f"{API}{REPO}/issues"
        "?labels=agent%3Aapproved%2Cbug&per_page=2&state=all&direction=asc&page=2"
    )
    assert first.links["next"]["url"] == expected
    assert numbers(first) == [1, 2]
    assert numbers(client.get(first.links["next"]["url"])) == [3, 4]


def test_per_page_defaults_to_30_is_capped_at_100_and_ignores_junk(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    for i in range(105):
        fake.add_issue(title=str(i))
    assert len(client.get(f"{REPO}/issues").json()) == 30
    big = client.get(f"{REPO}/issues", params={"per_page": 500})
    assert len(big.json()) == 100
    assert big.links["next"]["url"].endswith("per_page=500&page=2")
    assert len(client.get(f"{REPO}/issues", params={"per_page": 0}).json()) == 30
    assert len(client.get(f"{REPO}/issues", params={"per_page": "many"}).json()) == 30
    assert len(client.get(f"{REPO}/issues", params={"page": -4}).json()) == 30


# ----------------------------------------------------------------- state changes


def test_edit_issue_sets_last_edited_at_and_renames_only_on_title_change(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="old", body="b1")
    issue = fake.issues[n]
    assert issue.last_edited_at is None

    fake.edit_issue(n, body="b2")
    stamp = datetime(2026, 10, 24, 10, 0, 1, tzinfo=UTC)
    assert issue.last_edited_at == stamp
    assert issue.updated_at == stamp
    assert fake.events[n] == []

    fake.edit_issue(n, title="new", actor="alice")
    event = fake.events[n][-1]
    assert event["event"] == "renamed"
    assert event["actor"]["login"] == "alice"
    assert event["rename"] == {"from": "old", "to": "new"}
    assert event["created_at"] == iso_z(issue.last_edited_at) == "2026-10-24T10:00:02Z"
    data = client.get(f"{REPO}/issues/{n}").json()
    assert (data["title"], data["body"]) == ("new", "b2")

    stamp = issue.last_edited_at
    fake.edit_issue(n, title="new", body="b2")  # nothing really changes
    assert issue.last_edited_at == stamp
    assert len(fake.events[n]) == 1

    fake.add_comment(n, "comment")
    fake.label_issue(n, "x")
    assert issue.last_edited_at == stamp
    assert issue.updated_at > stamp


def test_close_and_reopen_update_state_reason_and_events(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    fake.close_issue(n, actor="erin")
    closed = client.get(f"{REPO}/issues/{n}").json()
    assert (closed["state"], closed["state_reason"]) == ("closed", "completed")
    assert closed["closed_at"] == fake.events[n][-1]["created_at"]
    fake.close_issue(n)  # already closed: nothing happens
    assert [e["event"] for e in fake.events[n]] == ["closed"]

    fake.reopen_issue(n, actor="frank")
    reopened = client.get(f"{REPO}/issues/{n}").json()
    assert (reopened["state"], reopened["state_reason"], reopened["closed_at"]) == (
        "open",
        None,
        None,
    )
    fake.reopen_issue(n)  # already open: nothing happens
    fake.close_issue(n, reason="not_planned")
    assert client.get(f"{REPO}/issues/{n}").json()["state_reason"] == "not_planned"
    events = client.get(f"{REPO}/issues/{n}/events").json()
    assert [e["event"] for e in events] == ["closed", "reopened", "closed"]
    assert [e["state_reason"] for e in events] == ["completed", "reopened", "not_planned"]
    assert [e["actor"]["login"] for e in events] == ["erin", "frank", "owner"]


def test_label_issue_is_idempotent_and_returns_the_previous_event_id(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    first = fake.label_issue(n, "agent:approved", actor="owner")
    again = fake.label_issue(n, "agent:approved", actor="someone-else")
    assert again == first
    events = client.get(f"{REPO}/issues/{n}/events").json()
    assert [(e["id"], e["actor"]["login"]) for e in events] == [(first, "owner")]
    assert fake.issue_labels(n) == ["agent:approved"]

    fake.unlabel_issue(n, "agent:approved")
    third = fake.label_issue(n, "agent:approved")
    assert third > first
    assert fake.issue_labels(n) == ["agent:approved"]


def test_unlabel_issue_appends_event_and_ignores_absent_labels(fake: FakeGitHub) -> None:
    n = fake.add_issue(title="t", labels=["a", "b"])
    before = len(fake.events[n])
    fake.unlabel_issue(n, "missing")
    assert len(fake.events[n]) == before
    fake.unlabel_issue(n, "a", actor="bob")
    assert fake.issue_labels(n) == ["b"]
    last = fake.events[n][-1]
    assert (last["event"], last["actor"]["login"], last["label"]["name"]) == (
        "unlabeled",
        "bob",
        "a",
    )


def test_pull_request_items_mirror_state_and_title_changes(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = open_pull(client).json()["number"]
    fake.close_issue(n)
    assert fake.pulls[n].state == "closed"
    closed = client.get(f"{REPO}/pulls/{n}").json()
    assert closed["state"] == "closed"
    assert closed["closed_at"] is not None
    fake.reopen_issue(n)
    fake.edit_issue(n, title="Better title", body="details")
    pull = client.get(f"{REPO}/pulls/{n}").json()
    assert (pull["state"], pull["title"], pull["body"]) == ("open", "Better title", "details")
    assert (fake.pulls[n].title, fake.pulls[n].body) == ("Better title", "details")


# ------------------------------------------------------------------------ events


def test_events_have_ordered_unique_ids_actors_and_payloads(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="old", author="alice", labels=["a"])
    fake.label_issue(n, "b", actor="bob")
    fake.edit_issue(n, title="new", actor="carol")
    fake.unlabel_issue(n, "a", actor="dave")
    fake.close_issue(n, actor="erin", reason="not_planned")
    fake.reopen_issue(n, actor="frank")
    events = client.get(f"{REPO}/issues/{n}/events").json()
    kinds = ["labeled", "labeled", "renamed", "unlabeled", "closed", "reopened"]
    assert [e["event"] for e in events] == kinds
    actors = ["alice", "bob", "carol", "dave", "erin", "frank"]
    assert [e["actor"]["login"] for e in events] == actors
    ids = [e["id"] for e in events]
    assert ids == sorted(set(ids))
    assert all(isinstance(i, int) for i in ids)
    created = [e["created_at"] for e in events]
    assert created == sorted(set(created))
    assert events[0]["label"] == {"name": "a", "color": "ededed"}
    assert events[2]["rename"] == {"from": "old", "to": "new"}
    assert events[4]["state_reason"] == "not_planned"
    assert "label" not in events[2]
    assert "rename" not in events[0]


def test_event_ids_are_unique_and_increasing_across_issues(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    first = fake.add_issue(title="a", labels=["x"])
    second = fake.add_issue(title="b", labels=["x"])
    later = fake.label_issue(first, "y")
    ids = [
        event["id"]
        for number in (first, second)
        for event in client.get(f"{REPO}/issues/{number}/events").json()
    ]
    assert len(set(ids)) == 3
    assert max(ids) == later
    assert fake.events[first][0]["id"] < fake.events[second][0]["id"] < later


def test_events_are_paginated_with_link_headers(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="t")
    for i in range(5):
        fake.label_issue(n, f"l{i}")
    first = client.get(f"{REPO}/issues/{n}/events", params={"per_page": 2})
    assert first.links["next"]["url"] == f"{API}{REPO}/issues/{n}/events?per_page=2&page=2"
    pages = walk(client, first)
    assert [len(page) for page in pages] == [2, 2, 1]
    assert [e["id"] for page in pages for e in page] == [e["id"] for e in fake.events[n]]


def test_events_of_unknown_issue_are_404(client: httpx.Client) -> None:
    assert client.get(f"{REPO}/issues/5/events").status_code == 404


# ----------------------------------------------------------------- label endpoints


def test_add_labels_autocreates_registry_entries_and_logs_only_new_labels(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t", labels=["existing"])
    response = client.post(
        f"{REPO}/issues/{n}/labels", json={"labels": ["agent:working", "existing"]}
    )
    assert response.status_code == 200
    assert names(response.json()) == ["agent:working", "existing"]
    assert set(fake.labels) == {"agent:working", "existing"}
    labeled = [e for e in fake.events[n] if e["event"] == "labeled"]
    assert [(e["label"]["name"], e["actor"]["login"]) for e in labeled] == [
        ("existing", "owner"),
        ("agent:working", "agent-bot"),
    ]
    assert names(client.get(f"{REPO}/labels").json()) == ["agent:working", "existing"]

    before = len(fake.events[n])
    again = client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["agent:working"]})
    assert again.status_code == 200
    assert len(fake.events[n]) == before


def test_add_labels_accepts_bare_lists_and_name_objects_and_uses_bot_login(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    fake.bot_login = "bridge-bot"
    n = fake.add_issue(title="t")
    assert names(client.post(f"{REPO}/issues/{n}/labels", json=["a"]).json()) == ["a"]
    response = client.post(f"{REPO}/issues/{n}/labels", json={"labels": [{"name": "b"}]})
    assert names(response.json()) == ["a", "b"]
    assert {e["actor"]["login"] for e in fake.events[n]} == {"bridge-bot"}


@pytest.mark.parametrize(
    "payload",
    [{}, {"labels": []}, {"labels": "bug"}, {"labels": [1]}, {"labels": ["ok", ""]}, []],
)
def test_add_labels_validation_failures_are_422_and_change_nothing(
    fake: FakeGitHub, client: httpx.Client, payload: object
) -> None:
    n = fake.add_issue(title="t")
    response = client.post(f"{REPO}/issues/{n}/labels", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "labels"
    assert fake.issue_labels(n) == []
    assert fake.labels == {}


def test_add_labels_to_unknown_issue_is_404(client: httpx.Client) -> None:
    response = client.post(f"{REPO}/issues/3/labels", json={"labels": ["x"]})
    assert response.status_code == 404


def test_remove_label_handles_url_encoded_names_and_404_when_absent(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t", labels=["agent:approved", "needs/triage", "keep"])
    first = client.delete(f"{REPO}/issues/{n}/labels/agent%3Aapproved")
    assert first.status_code == 200
    assert names(first.json()) == ["keep", "needs/triage"]
    second = client.delete(f"{REPO}/issues/{n}/labels/needs%2Ftriage")
    assert names(second.json()) == ["keep"]
    assert fake.issue_labels(n) == ["keep"]
    last = fake.events[n][-1]
    assert (last["event"], last["label"]["name"], last["actor"]["login"]) == (
        "unlabeled",
        "needs/triage",
        "agent-bot",
    )
    assert "agent:approved" in fake.labels  # removing from an issue keeps the repo label

    gone = client.delete(f"{REPO}/issues/{n}/labels/agent%3Aapproved")
    assert gone.status_code == 404
    assert gone.json() == {"message": "Label does not exist"}
    assert client.delete(f"{REPO}/issues/99/labels/keep").status_code == 404

    recorded = fake.requests[0]
    assert recorded.path.endswith("/labels/agent:approved")
    assert recorded.raw_path.endswith("/labels/agent%3Aapproved")


def test_repository_labels_can_be_created_listed_and_paginated(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    created = client.post(
        f"{REPO}/labels",
        json={"name": "agent:approved", "color": "#0E8A16", "description": "Ready to run"},
    )
    assert created.status_code == 201
    assert created.json()["color"] == "0e8a16"
    assert created.json()["description"] == "Ready to run"
    assert client.post(f"{REPO}/labels", json={"name": "plain"}).json()["color"] == "ededed"
    client.post(f"{REPO}/labels", json={"name": "zeta"})
    first = client.get(f"{REPO}/labels", params={"per_page": 2})
    assert names(first.json()) == ["agent:approved", "plain"]
    pages = walk(client, first)
    assert [names(page) for page in pages] == [["agent:approved", "plain"], ["zeta"]]
    assert set(fake.labels) == {"agent:approved", "plain", "zeta"}


def test_creating_an_existing_label_is_422_already_exists(client: httpx.Client) -> None:
    client.post(f"{REPO}/labels", json={"name": "bug"})
    duplicate = client.post(f"{REPO}/labels", json={"name": "bug", "color": "ffffff"})
    assert duplicate.status_code == 422
    assert duplicate.json() == {
        "message": "Validation Failed",
        "errors": [{"resource": "Label", "code": "already_exists", "field": "name"}],
    }
    assert client.get(f"{REPO}/labels").json()[0]["color"] == "ededed"


@pytest.mark.parametrize(
    ("payload", "field", "code"),
    [
        ({}, "name", "missing_field"),
        ({"name": "  "}, "name", "missing_field"),
        ({"name": "x", "color": "red"}, "color", "invalid"),
        ({"name": "x", "description": 5}, "description", "invalid"),
    ],
)
def test_creating_a_label_validates_its_fields(
    client: httpx.Client, payload: dict[str, object], field: str, code: str
) -> None:
    response = client.post(f"{REPO}/labels", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"] == [{"resource": "Label", "field": field, "code": code}]


# ----------------------------------------------------------------------- comments


def test_comments_can_be_created_and_listed_in_order(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    human = fake.add_comment(n, "from a human", author="alice")
    response = client.post(f"{REPO}/issues/{n}/comments", json={"body": "from the bot"})
    assert response.status_code == 201
    created = response.json()
    assert created["body"] == "from the bot"
    assert created["user"]["login"] == "agent-bot"
    assert created["id"] > human
    assert created["created_at"].endswith("Z")
    expected_url = f"https://github.test/owner/repo/issues/{n}#issuecomment-{created['id']}"
    assert created["html_url"] == expected_url
    listed = client.get(f"{REPO}/issues/{n}/comments").json()
    assert [c["body"] for c in listed] == ["from a human", "from the bot"]
    assert [c["user"]["login"] for c in listed] == ["alice", "agent-bot"]
    assert (fake.comments[n][-1].author, fake.comments[n][-1].body) == ("agent-bot", "from the bot")
    assert client.get(f"{REPO}/issues/{n}").json()["comments"] == 2

    fake.bot_login = "other-bot"
    other = client.post(f"{REPO}/issues/{n}/comments", json={"body": "again"}).json()
    assert other["user"]["login"] == "other-bot"


def test_comments_are_paginated_ascending(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="t")
    for i in range(3):
        fake.add_comment(n, f"c{i}")
    first = client.get(f"{REPO}/issues/{n}/comments", params={"per_page": 2})
    assert first.links["next"]["url"] == f"{API}{REPO}/issues/{n}/comments?per_page=2&page=2"
    pages = walk(client, first)
    assert [[c["body"] for c in page] for page in pages] == [["c0", "c1"], ["c2"]]


@pytest.mark.parametrize("payload", [{}, {"body": ""}, {"body": "   "}, {"body": None}])
def test_comment_without_text_is_422(
    fake: FakeGitHub, client: httpx.Client, payload: dict[str, object]
) -> None:
    n = fake.add_issue(title="t")
    response = client.post(f"{REPO}/issues/{n}/comments", json=payload)
    assert response.status_code == 422
    assert response.json()["message"] == "Validation Failed"
    assert response.json()["errors"][0]["field"] == "body"
    assert fake.comments[n] == []


def test_comment_length_limit_and_bad_json_and_unknown_issue(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    at_limit = client.post(f"{REPO}/issues/{n}/comments", json={"body": "x" * MAX_BODY_LENGTH})
    assert at_limit.status_code == 201
    too_long = client.post(
        f"{REPO}/issues/{n}/comments", json={"body": "x" * (MAX_BODY_LENGTH + 1)}
    )
    assert too_long.status_code == 422
    assert "too long" in too_long.json()["errors"][0]["message"]
    wrong_type = client.post(f"{REPO}/issues/{n}/comments", json={"body": 12})
    assert wrong_type.status_code == 422
    garbage = client.post(f"{REPO}/issues/{n}/comments", content=b"{not json")
    assert garbage.status_code == 400
    assert garbage.json() == {"message": "Problems parsing JSON"}
    assert fake.requests[-1].json is None
    assert client.post(f"{REPO}/issues/{n}/comments").status_code == 400
    assert client.post(f"{REPO}/issues/77/comments", json={"body": "x"}).status_code == 404
    assert len(fake.comments[n]) == 1


def test_comments_bump_updated_at_but_not_last_edited_at(fake: FakeGitHub) -> None:
    n = fake.add_issue(title="t")
    issue = fake.issues[n]
    created = issue.updated_at
    fake.add_comment(n, "hi")
    assert issue.updated_at > created
    assert issue.last_edited_at is None


# --------------------------------------------------------------------- pull requests


def test_create_pull_returns_github_shape_and_registers_an_issue_item(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    issue = fake.add_issue(title="work")
    response = open_pull(client, body="Closes #1", draft=True)
    assert response.status_code == 201
    pr = response.json()
    assert pr["number"] == issue + 1
    assert pr["html_url"] == "https://github.test/owner/repo/pull/2"
    assert (pr["state"], pr["draft"], pr["title"], pr["body"]) == (
        "open",
        True,
        "Fix it",
        "Closes #1",
    )
    assert pr["head"] == {"ref": "agent/issue-1", "label": "owner:agent/issue-1"}
    assert pr["base"]["ref"] == "main"
    assert pr["user"]["login"] == "agent-bot"
    assert pr["created_at"] == "2026-10-24T10:00:01Z"

    stored = fake.pulls[2]
    assert (stored.head_ref, stored.base_ref, stored.draft) == ("agent/issue-1", "main", True)
    assert stored.html_url == pr["html_url"]
    assert fake.issues[2].pull_request is True
    assert fake.issues[2].author == "agent-bot"
    assert client.get(f"{REPO}/pulls/2").json() == pr
    as_issue = client.get(f"{REPO}/issues/2").json()
    assert "pull_request" in as_issue
    assert as_issue["number"] == 2


def test_create_pull_defaults_to_non_draft_and_normalises_head(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    plain = open_pull(client, head="owner:agent/a").json()
    assert plain["draft"] is False
    assert plain["body"] is None
    assert plain["head"] == {"ref": "agent/a", "label": "owner:agent/a"}
    fork = open_pull(client, head="fork:agent/a").json()
    assert fork["head"] == {"ref": "agent/a", "label": "fork:agent/a"}
    assert fake.pulls[fork["number"]].head_owner == "fork"


def test_duplicate_open_pull_for_same_head_and_base_is_422_and_consumes_no_number(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    first = open_pull(client).json()["number"]
    duplicate = open_pull(client)
    assert duplicate.status_code == 422
    assert duplicate.json() == {
        "message": "Validation Failed",
        "errors": [
            {
                "resource": "PullRequest",
                "code": "custom",
                "message": "A pull request already exists for owner:agent/issue-1.",
            }
        ],
    }
    assert list(fake.pulls) == [first]
    assert fake.add_issue(title="next") == first + 1

    other_base = open_pull(client, base="develop")
    assert other_base.status_code == 201

    fake.close_issue(first)
    reopened_work = open_pull(client)
    assert reopened_work.status_code == 201
    assert reopened_work.json()["number"] == first + 3


@pytest.mark.parametrize("missing", ["title", "head", "base"])
def test_create_pull_requires_title_head_and_base(
    fake: FakeGitHub, client: httpx.Client, missing: str
) -> None:
    payload = {"title": "t", "head": "agent/x", "base": "main"}
    del payload[missing]
    response = client.post(f"{REPO}/pulls", json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "message": "Validation Failed",
        "errors": [{"resource": "PullRequest", "field": missing, "code": "missing_field"}],
    }
    blank = client.post(f"{REPO}/pulls", json={**payload, missing: " "})
    assert blank.status_code == 422
    assert fake.pulls == {}
    assert fake.issues == {}


def test_create_pull_validates_body_draft_and_head_format(client: httpx.Client) -> None:
    too_long = open_pull(client, body="x" * (MAX_BODY_LENGTH + 1))
    assert too_long.status_code == 422
    assert too_long.json()["errors"][0]["field"] == "body"
    assert open_pull(client, draft="yes").json()["errors"][0]["field"] == "draft"
    assert open_pull(client, head=":branch").json()["errors"][0]["field"] == "head"
    assert open_pull(client, head="owner:").json()["errors"][0]["field"] == "head"
    assert client.post(f"{REPO}/pulls", content=b"[]").status_code == 400


def test_list_pulls_filters_by_head_base_and_state(fake: FakeGitHub, client: httpx.Client) -> None:
    a = open_pull(client, head="agent/a").json()["number"]
    b = open_pull(client, head="agent/b", base="develop").json()["number"]
    only_a = client.get(f"{REPO}/pulls", params={"head": "owner:agent/a"})
    assert numbers(only_a) == [a]
    assert numbers(client.get(f"{REPO}/pulls")) == [b, a]
    assert numbers(client.get(f"{REPO}/pulls", params={"direction": "asc"})) == [a, b]
    assert numbers(client.get(f"{REPO}/pulls", params={"base": "develop"})) == [b]
    # GitHub only understands the <owner>:<branch> form of the head filter
    assert client.get(f"{REPO}/pulls", params={"head": "agent/a"}).json() == []
    assert client.get(f"{REPO}/pulls", params={"head": "other:agent/a"}).json() == []

    fake.close_issue(a)
    assert client.get(f"{REPO}/pulls", params={"head": "owner:agent/a"}).json() == []
    closed = client.get(f"{REPO}/pulls", params={"head": "owner:agent/a", "state": "closed"})
    assert numbers(closed) == [a]
    assert closed.json()[0]["state"] == "closed"
    assert numbers(client.get(f"{REPO}/pulls", params={"state": "all"})) == [b, a]
    assert client.get(f"{REPO}/pulls", params={"state": "weird"}).status_code == 422


def test_list_pulls_is_paginated(client: httpx.Client) -> None:
    for i in range(3):
        open_pull(client, head=f"agent/{i}")
    first = client.get(f"{REPO}/pulls", params={"per_page": 2, "state": "all"})
    assert first.links["next"]["url"] == f"{API}{REPO}/pulls?per_page=2&state=all&page=2"
    assert [len(page) for page in walk(client, first)] == [2, 1]


def test_get_pull_unknown_or_plain_issue_is_404(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="just an issue")
    assert client.get(f"{REPO}/pulls/{n}").status_code == 404
    assert client.get(f"{REPO}/pulls/999").json() == {"message": "Not Found"}
    assert client.get(f"{REPO}/pulls/abc").status_code == 404


# ------------------------------------------------------------------------ GraphQL


def test_graphql_last_edited_at_is_null_until_the_issue_is_edited(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t", body="b")
    response = graphql(client, n)
    assert response.status_code == 200
    assert response.json() == {"data": {"repository": {"issue": {"lastEditedAt": None}}}}

    fake.edit_issue(n, body="b2")
    stamp = iso_z(fake.issues[n].last_edited_at)
    assert stamp == "2026-10-24T10:00:01Z"
    assert graphql(client, n).json() == {"data": {"repository": {"issue": {"lastEditedAt": stamp}}}}
    fake.add_comment(n, "comments do not count as edits")
    assert graphql(client, n).json()["data"]["repository"]["issue"]["lastEditedAt"] == stamp
    fake.edit_issue(n, title="renamed")
    later = graphql(client, n).json()["data"]["repository"]["issue"]["lastEditedAt"]
    assert later > stamp


def test_graphql_unknown_issue_reports_not_found(fake: FakeGitHub, client: httpx.Client) -> None:
    response = graphql(client, 99)
    assert response.status_code == 200
    assert response.json() == {
        "data": {"repository": {"issue": None}},
        "errors": [{"type": "NOT_FOUND", "message": "Could not resolve to an Issue"}],
    }
    pr = open_pull(client).json()["number"]
    assert graphql(client, pr).json()["data"] == {"repository": {"issue": None}}


def test_graphql_unsupported_query_and_invalid_variables(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    unsupported = client.post("/graphql", json={"query": "{ viewer { login } }"})
    assert unsupported.status_code == 200
    assert unsupported.json() == {"errors": [{"message": "unsupported query"}]}
    assert client.post("/graphql", json={"variables": {}}).json() == {
        "errors": [{"message": "unsupported query"}]
    }
    wrong_repo = graphql(client, n, owner="someone", name="else").json()
    assert wrong_repo["data"] == {"repository": None}
    assert wrong_repo["errors"][0]["type"] == "NOT_FOUND"
    for bad_number in ("1", True, None):
        invalid = graphql(client, bad_number)  # type: ignore[arg-type]
        assert invalid.json()["errors"][0]["type"] == "INVALID_VARIABLE"
    assert client.post("/graphql", content=b"nope").status_code == 400


# ---------------------------------------------------------------- fault injection


def test_inject_returns_canned_response_for_exactly_times_requests(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    fake.add_issue(title="t")
    fake.inject(
        "GET",
        r"/issues$",
        status=503,
        json={"message": "unavailable"},
        headers={"Retry-After": "7"},
        times=2,
    )
    for _ in range(2):
        response = client.get(f"{REPO}/issues")
        assert response.status_code == 503
        assert response.json() == {"message": "unavailable"}
        assert response.headers["retry-after"] == "7"
        assert response.headers["x-ratelimit-limit"] == "5000"
        assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert client.get(f"{REPO}/issues").status_code == 200
    assert len(fake.requests) == 3


def test_inject_status_only_has_a_json_error_body_and_json_only_defaults_to_200(
    client: httpx.Client, fake: FakeGitHub
) -> None:
    fake.inject("GET", r"/labels$", status=500)
    failed = client.get(f"{REPO}/labels")
    assert failed.status_code == 500
    assert failed.json() == {"message": "Injected error"}
    fake.inject("GET", r"/labels$", json=[{"name": "canned"}])
    canned = client.get(f"{REPO}/labels")
    assert canned.status_code == 200
    assert canned.json() == [{"name": "canned"}]


def test_inject_headers_override_the_default_rate_limit_headers(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    fake.inject(
        "GET",
        r"/issues$",
        status=403,
        json={"message": "API rate limit exceeded"},
        headers={"x-ratelimit-remaining": "0", "X-RateLimit-Reset": "1792839999"},
    )
    limited = client.get(f"{REPO}/issues")
    assert limited.status_code == 403
    assert limited.headers.get_list("x-ratelimit-remaining") == ["0"]
    assert limited.headers.get_list("x-ratelimit-reset") == ["1792839999"]
    assert limited.headers["x-ratelimit-limit"] == "5000"
    assert client.get(f"{REPO}/issues").headers["x-ratelimit-remaining"] == "4999"


def test_inject_exception_is_raised_before_any_state_change(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    fake.inject("POST", r"/comments$", exception=httpx.ConnectError("boom"))
    with pytest.raises(httpx.ConnectError, match="boom"):
        client.post(f"{REPO}/issues/{n}/comments", json={"body": "lost"})
    assert fake.comments[n] == []
    assert fake.requests[-1].json == {"body": "lost"}  # recorded even though it failed

    retried = client.post(f"{REPO}/issues/{n}/comments", json={"body": "kept"})
    assert retried.status_code == 201
    assert [c.body for c in fake.comments[n]] == ["kept"]


def test_inject_exception_times_and_failed_writes_leave_state_untouched(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    fake.inject("POST", r"/labels$", exception=httpx.ReadTimeout("slow"), times=2)
    for _ in range(2):
        with pytest.raises(httpx.ReadTimeout):
            client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["x"]})
    assert fake.issue_labels(n) == []
    assert fake.labels == {}
    assert client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["x"]}).status_code == 200
    assert fake.issue_labels(n) == ["x"]


def test_inject_matches_on_method_and_path_regex_only(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    fake.inject("*", r"/issues/\d+/labels$", status=500)
    assert client.get(f"{REPO}/issues").status_code == 200  # path does not match
    assert client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["x"]}).status_code == 500
    assert fake.issue_labels(n) == []  # the failed write was not applied
    assert client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["x"]}).status_code == 200

    fake.inject("DELETE", r"/issues/\d+$", status=502)
    assert client.get(f"{REPO}/issues/{n}").status_code == 200  # method does not match
    assert client.delete(f"{REPO}/issues/{n}").status_code == 502
    assert client.delete(f"{REPO}/issues/{n}").status_code == 405  # consumed, normal handling


def test_inject_rules_are_consumed_in_registration_order(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    fake.inject("GET", r"/issues", status=500)
    fake.inject("GET", r"/issues", status=502)
    statuses = [client.get(f"{REPO}/issues").status_code for _ in range(3)]
    assert statuses == [500, 502, 200]


def test_inject_rejects_nonsensical_arguments(fake: FakeGitHub) -> None:
    with pytest.raises(ValueError, match="needs a status"):
        fake.inject("GET", r"/issues")
    with pytest.raises(ValueError, match="either an exception"):
        fake.inject("GET", r"/issues", status=500, exception=RuntimeError("x"))
    with pytest.raises(ValueError, match="times"):
        fake.inject("GET", r"/issues", status=500, times=0)
    with pytest.raises(Exception, match="unterminated|missing"):
        fake.inject("GET", r"(", status=500)


def test_deny_blocks_matching_requests_until_allowed(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    fake.deny("POST", r"/issues/\d+/comments$")
    for _ in range(2):  # permanent, not consumed
        denied = client.post(f"{REPO}/issues/{n}/comments", json={"body": "x"})
        assert denied.status_code == 403
        body = denied.json()
        assert body["message"] == "Resource not accessible by personal access token"
        assert body["documentation_url"].startswith("https://docs.github.com")
    assert fake.comments[n] == []
    assert client.get(f"{REPO}/issues/{n}/comments").status_code == 200  # other methods fine
    assert client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["x"]}).status_code == 200

    fake.allow("POST", r"/issues/\d+/comments$")
    assert client.post(f"{REPO}/issues/{n}/comments", json={"body": "x"}).status_code == 201


def test_deny_supports_custom_messages_wildcards_and_requires_the_same_args_to_allow(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    fake.deny("*", r"/pulls", "Must have admin rights to Repository.")
    denied = client.get(f"{REPO}/pulls")
    assert denied.status_code == 403
    assert denied.json()["message"] == "Must have admin rights to Repository."
    assert open_pull(client).status_code == 403
    assert fake.pulls == {}

    fake.allow("GET", r"/pulls")  # different method: still denied
    assert client.get(f"{REPO}/pulls").status_code == 403
    fake.allow("*", r"/pulls")
    assert client.get(f"{REPO}/pulls").status_code == 200
    fake.allow("*", r"/pulls")  # allowing twice is harmless


# ---------------------------------------------------------------------- recording


def test_requests_are_recorded_before_processing_with_parsed_fields(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    client.post(f"{REPO}/issues/1/comments?x=1&y=two", json={"body": "hi"})  # issue 1 is unknown
    client.get(f"{REPO}/issues", params={"labels": "a,b"})
    first, second = fake.requests
    assert first.method == "POST"
    assert first.path == "/repos/owner/repo/issues/1/comments"
    assert first.query == {"x": "1", "y": "two"}
    assert first.json == {"body": "hi"}
    assert first.headers["authorization"] == "Bearer test-token"
    assert first.headers["content-type"] == "application/json"
    assert all(key == key.lower() for key in first.headers)
    assert (second.method, second.query, second.json) == ("GET", {"labels": "a,b"}, None)
    assert first.time < second.time
    assert isinstance(first.time, int)


def test_unauthenticated_requests_are_recorded_too(fake: FakeGitHub) -> None:
    fake.client(token=None).get(f"{REPO}/issues")
    assert len(fake.requests) == 1
    assert "authorization" not in fake.requests[0].headers


def test_writes_excludes_gets_and_graphql_but_keeps_failed_writes(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    assert fake.requests == []  # seeding is not an HTTP request
    client.get(f"{REPO}/issues")
    graphql(client, n)
    client.post(f"{REPO}/issues/{n}/comments", json={"body": "ok"})
    client.delete(f"{REPO}/issues/{n}/labels/missing")  # 404, still a write attempt
    client.post(f"{REPO}/issues/{n}/comments", json={})  # 422
    writes = fake.writes()
    assert [(w.method, w.path.split("/")[-1]) for w in writes] == [
        ("POST", "comments"),
        ("DELETE", "missing"),
        ("POST", "comments"),
    ]
    assert len(fake.requests) == 5
    assert all(w.path != "/graphql" for w in writes)
    assert FakeGitHub().writes() == []


def test_reset_requests_clears_recordings_but_keeps_state_and_sequence(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="t")
    client.post(f"{REPO}/issues/{n}/comments", json={"body": "x"})
    last = fake.requests[-1].time
    recorded = fake.requests
    fake.reset_requests()
    assert fake.requests == []
    assert recorded is fake.requests  # same list object, so existing references see the reset
    assert fake.writes() == []
    assert len(fake.comments[n]) == 1
    client.get(f"{REPO}/issues")
    assert fake.requests[0].time > last


def test_request_json_is_none_for_invalid_bodies(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="t")
    client.post(f"{REPO}/issues/{n}/comments", content=b"\xff\xfe not json")
    client.post(f"{REPO}/issues/{n}/comments")
    assert [rec.json for rec in fake.requests] == [None, None]


# ------------------------------------------------------------------- bridge scenarios


def test_approval_then_edit_is_detectable_from_event_and_graphql_timestamps(
    fake: FakeGitHub, client: httpx.Client
) -> None:
    n = fake.add_issue(title="Mow", body="v1")
    fake.label_issue(n, "agent:approved", actor="owner")

    def last_approval() -> str:
        events = client.get(f"{REPO}/issues/{n}/events").json()
        return [e for e in events if e["label"]["name"] == "agent:approved"][-1]["created_at"]

    def last_edited() -> str | None:
        return graphql(client, n).json()["data"]["repository"]["issue"]["lastEditedAt"]

    assert last_edited() is None  # untouched since approval: safe to run
    fake.edit_issue(n, body="v2, changed after approval")
    assert last_edited() is not None
    assert last_edited() > last_approval()  # ISO strings order chronologically: approval is stale

    fake.unlabel_issue(n, "agent:approved")
    fake.label_issue(n, "agent:approved", actor="owner")  # approved again after the edit
    assert last_approval() > last_edited()


def test_bridge_style_claim_run_and_report_flow(fake: FakeGitHub, client: httpx.Client) -> None:
    n = fake.add_issue(title="Trim hedge", body="please", labels=["agent:approved"])
    queue = client.get(
        f"{REPO}/issues",
        params={
            "labels": "agent:approved",
            "state": "open",
            "per_page": 100,
            "sort": "created",
            "direction": "asc",
        },
    )
    assert numbers(queue) == [n]

    assert client.post(f"{REPO}/issues/{n}/labels", json={"labels": ["agent:working"]}).is_success
    assert client.delete(f"{REPO}/issues/{n}/labels/agent%3Aapproved").is_success
    pull = open_pull(client, head=f"agent/issue-{n}", body=f"Closes #{n}").json()
    comment = client.post(
        f"{REPO}/issues/{n}/comments", json={"body": f"Opened {pull['html_url']}"}
    )
    assert comment.status_code == 201

    assert fake.issue_labels(n) == ["agent:working"]
    assert client.get(f"{REPO}/issues", params={"labels": "agent:approved"}).json() == []
    assert [(w.method, w.path.rsplit("/", 1)[-1]) for w in fake.writes()] == [
        ("POST", "labels"),
        ("DELETE", "agent:approved"),
        ("POST", "pulls"),
        ("POST", "comments"),
    ]
    existing = client.get(f"{REPO}/pulls", params={"head": f"owner:agent/issue-{n}"})
    assert numbers(existing) == [pull["number"]]
