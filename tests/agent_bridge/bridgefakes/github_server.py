"""In-process fake GitHub server for agent-bridge tests.

``FakeGitHub`` emulates the slice of the GitHub REST v3 and GraphQL APIs that the bridge uses
and plugs into ``httpx`` as a ``MockTransport``::

    fake = FakeGitHub()
    with fake.client() as http:
        http.get("/repos/owner/repo/issues", params={"labels": "agent:approved"})

State lives in memory. Every request is recorded before it is processed. All timestamps come
from a deterministic clock (one second per state change). Faults can be injected, and
permissions denied, without ever touching the state of the failed call.

Fidelity notes (kept on purpose so client bugs surface here and not in production):

* empty issue/PR bodies serialise as ``null``; comment/PR bodies over 65536 characters are 422;
* ``POST .../labels`` with an empty list is 422 (never "remove all labels");
* pull requests are issues: they show up in the issue list with a ``pull_request`` key and
  share the number counter, but GraphQL ``repository.issue`` does not resolve them;
* ``GET /pulls?head=`` only matches the ``<owner>:<branch>`` form, like GitHub.
* ``GET /user`` answers with the account that authored every comment, label event and pull
  request the fake creates (``bot_login``), which is how the bridge recognises its own comments.

Fault knobs (plain attributes, off by default; ``inject`` and ``deny`` cover everything else):

* ``ignore_draft``: ``POST .../pulls`` opens ready-for-review pull requests whatever ``draft``
  says, like a server that drops a request field it does not understand.

Not implemented (404 as unknown routes): ``GET /repos/{owner}/{repo}``.
"""

import json as jsonlib
import math
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, TypeVar
from urllib.parse import unquote, urlencode

import httpx

__all__ = [
    "API_URL",
    "MAX_BODY_LENGTH",
    "WEB_URL",
    "FakeComment",
    "FakeGitHub",
    "FakeIssue",
    "FakeLabel",
    "FakePull",
    "RecordedRequest",
    "iso_z",
]

API_URL = "https://api.github.test"
WEB_URL = "https://github.test"
DOCS_URL = "https://docs.github.com/rest"
DEFAULT_START = datetime(2026, 10, 24, 10, 0, 0, tzinfo=UTC)
DEFAULT_LABEL_COLOR = "ededed"
DENY_MESSAGE = "Resource not accessible by personal access token"
MAX_BODY_LENGTH = 65536
RATE_LIMIT = 5000
DEFAULT_PER_PAGE = 30
MAX_PER_PAGE = 100

T = TypeVar("T")
R = TypeVar("R")


def iso_z(value: datetime) -> str:
    """Format ``value`` as second-precision ISO 8601 UTC with a trailing ``Z``."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FakeIssue:
    """An issue or pull-request item as stored by the fake (``pull_request`` marks PRs)."""

    number: int
    title: str
    body: str
    state: str
    state_reason: str | None
    author: str
    labels: list[str]
    created_at: datetime
    updated_at: datetime
    last_edited_at: datetime | None = None
    pull_request: bool = False


@dataclass
class FakeComment:
    """A comment on an issue or pull request."""

    id: int
    body: str
    author: str
    created_at: datetime


@dataclass
class FakePull:
    """Pull-request details; ``head_owner`` of ``""`` means the repository owner."""

    number: int
    title: str
    body: str
    head_ref: str
    base_ref: str
    draft: bool
    state: str
    html_url: str
    created_at: datetime
    head_owner: str = ""


@dataclass
class FakeLabel:
    """An entry in the repository label registry."""

    name: str
    color: str = DEFAULT_LABEL_COLOR
    description: str | None = None
    id: int = 0


@dataclass
class RecordedRequest:
    """One request seen by the fake. ``path`` is URL-decoded, ``raw_path`` is as sent."""

    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    json: object | None
    time: int
    raw_path: str = ""


@dataclass
class _Reply:
    status: int
    body: object
    headers: dict[str, str] = field(default_factory=dict)


class _Abort(Exception):
    """Raised inside handlers to short-circuit with an error reply."""

    def __init__(self, reply: _Reply) -> None:
        super().__init__(reply.status)
        self.reply = reply


@dataclass
class _Call:
    request: httpx.Request
    rec: RecordedRequest
    params: dict[str, str]


@dataclass
class _Rule:
    """A pending fault injection (``remaining`` matching requests left)."""

    method: str
    pattern: re.Pattern[str]
    remaining: int
    status: int | None
    json: object | None
    headers: dict[str, str]
    exception: Exception | None


@dataclass
class _Denial:
    method: str
    pattern: re.Pattern[str]
    message: str


def _error(status: int, message: str, **extra: object) -> _Reply:
    return _Reply(status, {"message": message, **extra})


def _not_found() -> _Reply:
    return _error(404, "Not Found")


def _validation_failed(*errors: dict[str, object]) -> _Reply:
    return _Reply(422, {"message": "Validation Failed", "errors": list(errors)})


def _locked(method: Callable[..., R]) -> Callable[..., R]:
    """Run ``method`` under the instance lock so a polling thread cannot interleave with seeding."""

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _route_matches(method: str, pattern: re.Pattern[str], rec: RecordedRequest) -> bool:
    if method not in ("*", rec.method):
        return False
    return bool(pattern.search(rec.path) or pattern.search(rec.raw_path))


def _match(pattern: tuple[str, ...], segments: list[str]) -> dict[str, str] | None:
    """Match decoded path segments against ``pattern``; ``{name}`` captures one segment."""
    if len(pattern) != len(segments):
        return None
    params: dict[str, str] = {}
    for expected, actual in zip(pattern, segments, strict=True):
        if expected.startswith("{"):
            params[expected[1:-1]] = actual
        elif expected != actual:
            return None
    return params


def _bounded_int(raw: str | None, default: int, minimum: int, maximum: int | None) -> int:
    """Parse a pagination parameter the way GitHub does: junk or too small means default."""
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        return default
    if value < minimum:
        return default
    return value if maximum is None else min(value, maximum)


def _json_body(call: _Call) -> object:
    raw = call.request.content
    if not raw:
        raise _Abort(_error(400, "Problems parsing JSON"))
    try:
        return jsonlib.loads(raw)
    except ValueError:
        raise _Abort(_error(400, "Problems parsing JSON")) from None


def _json_object(call: _Call) -> dict[str, Any]:
    body = _json_body(call)
    if not isinstance(body, dict):
        raise _Abort(_error(400, "Problems parsing JSON"))
    return body


def _check_choice(resource: str, name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise _Abort(_validation_failed({"resource": resource, "field": name, "code": "invalid"}))


def _check_text_body(resource: str, value: object, *, required: bool) -> str:
    """Validate a comment/PR body: must be a string within GitHub's length limit."""
    if value is None or value == "":
        if required:
            raise _Abort(
                _validation_failed({"resource": resource, "field": "body", "code": "missing_field"})
            )
        return ""
    if not isinstance(value, str):
        raise _Abort(_validation_failed({"resource": resource, "field": "body", "code": "invalid"}))
    if len(value) > MAX_BODY_LENGTH:
        raise _Abort(
            _validation_failed(
                {
                    "resource": resource,
                    "field": "body",
                    "code": "custom",
                    "message": f"body is too long (maximum is {MAX_BODY_LENGTH} characters)",
                }
            )
        )
    return value


class FakeGitHub:
    """In-memory GitHub for a single repository, served through ``httpx.MockTransport``."""

    api_url: str = API_URL
    web_url: str = WEB_URL

    def __init__(
        self,
        repo: str = "owner/repo",
        *,
        token: str | None = "test-token",
        start: datetime | None = None,
    ) -> None:
        owner, sep, name = repo.partition("/")
        if not sep or not owner or not name or "/" in name:
            raise ValueError(f"repo must look like 'owner/name', got {repo!r}")
        if start is None:
            start = DEFAULT_START
        elif start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        self.repo = repo
        self.owner = owner
        self.name = name
        self.token = token
        self.bot_login = "agent-bot"
        self.ignore_draft = False
        self.issues: dict[int, FakeIssue] = {}
        self.comments: dict[int, list[FakeComment]] = {}
        self.pulls: dict[int, FakePull] = {}
        self.events: dict[int, list[dict[str, Any]]] = {}
        self.labels: dict[str, FakeLabel] = {}
        self.requests: list[RecordedRequest] = []
        self._lock = threading.RLock()
        self._clock = start.astimezone(UTC)
        self._next_number = 1
        self._next_event_id = 1
        self._next_comment_id = 1
        self._next_label_id = 1
        self._sequence = 0
        self._user_ids: dict[str, int] = {}
        self._injections: list[_Rule] = []
        self._denials: dict[tuple[str, str], _Denial] = {}
        repo_prefix = ("repos", "{owner}", "{repo}")
        self._routes: list[tuple[tuple[str, ...], dict[str, Callable[[_Call], _Reply]]]] = [
            (("graphql",), {"POST": self._graphql}),
            (("user",), {"GET": self._get_user}),
            (repo_prefix + ("issues",), {"GET": self._list_issues}),
            (repo_prefix + ("issues", "{n}"), {"GET": self._get_issue}),
            (repo_prefix + ("issues", "{n}", "events"), {"GET": self._list_events}),
            (
                repo_prefix + ("issues", "{n}", "comments"),
                {"GET": self._list_comments, "POST": self._create_comment},
            ),
            (repo_prefix + ("issues", "{n}", "labels"), {"POST": self._add_labels}),
            (
                repo_prefix + ("issues", "{n}", "labels", "{name}"),
                {"DELETE": self._remove_label},
            ),
            (
                repo_prefix + ("labels",),
                {"GET": self._list_labels, "POST": self._create_label},
            ),
            (
                repo_prefix + ("pulls",),
                {"GET": self._list_pulls, "POST": self._create_pull},
            ),
            (repo_prefix + ("pulls", "{n}"), {"GET": self._get_pull}),
        ]

    # ------------------------------------------------------------------ clock

    @_locked
    def now(self) -> datetime:
        """Return the current fake time and advance the clock by exactly one second."""
        value = self._clock
        self._clock = value + timedelta(seconds=1)
        return value

    @_locked
    def tick(self, seconds: float = 0) -> None:
        """Skip the clock forward by ``seconds`` (never backwards)."""
        if seconds < 0:
            raise ValueError("the fake clock cannot go backwards")
        self._clock += timedelta(seconds=seconds)

    # ------------------------------------------------- seeding / simulating humans

    @_locked
    def add_issue(
        self,
        *,
        title: str,
        body: str = "",
        author: str = "owner",
        labels: Iterable[str] = (),
        state: str = "open",
        pull_request: bool = False,
        number: int | None = None,
    ) -> int:
        """Create an issue (or PR item); ``labels`` get 'labeled' events by ``author``."""
        if state not in ("open", "closed"):
            raise ValueError(f"state must be 'open' or 'closed', got {state!r}")
        issue_number = self._allocate_number(number)
        created = self.now()
        issue = FakeIssue(
            number=issue_number,
            title=title,
            body=body,
            state=state,
            state_reason="completed" if state == "closed" else None,
            author=author,
            labels=[],
            created_at=created,
            updated_at=created,
            pull_request=pull_request,
        )
        self._register_issue(issue)
        for name in [labels] if isinstance(labels, str) else labels:
            if name not in issue.labels:
                self._apply_label(issue, name, author)
        return issue_number

    @_locked
    def label_issue(self, number: int, label: str, actor: str = "owner") -> int:
        """Add ``label``; returns the 'labeled' event id (the previous one if already present)."""
        issue = self._issue(number)
        if label in issue.labels:
            previous = self._last_label_event(number, label)
            if previous is not None:
                return previous["id"]
        return self._apply_label(issue, label, actor)["id"]

    @_locked
    def unlabel_issue(self, number: int, label: str, actor: str = "owner") -> None:
        """Remove ``label`` and append an 'unlabeled' event (no-op if it is not present)."""
        issue = self._issue(number)
        if label in issue.labels:
            self._remove_label_from(issue, label, actor)

    @_locked
    def edit_issue(
        self,
        number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        actor: str = "owner",
    ) -> None:
        """Edit title/body; real changes set ``updated_at``/``last_edited_at`` (what GraphQL
        ``lastEditedAt`` reports) and a title change appends a 'renamed' event."""
        issue = self._issue(number)
        title_changed = title is not None and title != issue.title
        body_changed = body is not None and body != issue.body
        if not (title_changed or body_changed):
            return
        stamp = self.now()
        if title is not None and title_changed:
            self._append_event(
                issue,
                "renamed",
                actor,
                at=stamp,
                rename={"from": issue.title, "to": title},
            )
            issue.title = title
        if body is not None and body_changed:
            issue.body = body
        issue.updated_at = stamp
        issue.last_edited_at = stamp
        self._sync_pull(issue)

    @_locked
    def close_issue(self, number: int, actor: str = "owner", reason: str = "completed") -> None:
        """Close the issue (no-op if already closed) and append a 'closed' event."""
        issue = self._issue(number)
        if issue.state == "closed":
            return
        issue.state = "closed"
        issue.state_reason = reason
        self._append_event(issue, "closed", actor, state_reason=reason)
        self._sync_pull(issue)

    @_locked
    def reopen_issue(self, number: int, actor: str = "owner") -> None:
        """Reopen the issue (no-op if open) and append a 'reopened' event."""
        issue = self._issue(number)
        if issue.state == "open":
            return
        issue.state = "open"
        issue.state_reason = None
        self._append_event(issue, "reopened", actor, state_reason="reopened")
        self._sync_pull(issue)

    @_locked
    def add_comment(self, number: int, body: str, author: str = "someone") -> int:
        """Append a comment and return its id."""
        return self._append_comment(self._issue(number), body, author).id

    # -------------------------------------------------------------- inspection

    def issue_labels(self, number: int) -> list[str]:
        """Sorted label names currently on the issue."""
        return sorted(self._issue(number).labels)

    def writes(self) -> list[RecordedRequest]:
        """Recorded requests that mutate state: any non-GET except ``POST /graphql``."""
        return [
            rec
            for rec in self.requests
            if rec.method != "GET" and not (rec.method == "POST" and rec.path == "/graphql")
        ]

    @_locked
    def reset_requests(self) -> None:
        """Forget recorded requests (sequence numbers keep increasing)."""
        self.requests.clear()

    # ------------------------------------------------ fault injection / permissions

    @_locked
    def inject(
        self,
        method: str,
        path_regex: str,
        *,
        status: int | None = None,
        json: object | None = None,
        headers: dict[str, str] | None = None,
        exception: Exception | None = None,
        times: int = 1,
    ) -> None:
        """Answer the next ``times`` matching requests with a canned response or an exception.

        ``method`` may be ``'*'``; ``path_regex`` is ``re.search``-ed against the URL path.
        Injection runs before authentication and routing, so a failed write never changes
        state. Rules are consumed in registration order.
        """
        if exception is not None and (status is not None or json is not None or headers):
            raise ValueError("inject() takes either an exception or a canned response, not both")
        if exception is None and status is None and json is None and headers is None:
            raise ValueError("inject() needs a status, json, headers or an exception")
        if times < 1:
            raise ValueError("times must be at least 1")
        self._injections.append(
            _Rule(
                method=method.upper(),
                pattern=re.compile(path_regex),
                remaining=times,
                status=status,
                json=json,
                headers={key: str(value) for key, value in (headers or {}).items()},
                exception=exception,
            )
        )

    @_locked
    def deny(self, method: str, path_regex: str, message: str = DENY_MESSAGE) -> None:
        """Answer every matching request with 403 until ``allow`` is called with the same args."""
        key = (method.upper(), path_regex)
        self._denials[key] = _Denial(key[0], re.compile(path_regex), message)

    @_locked
    def allow(self, method: str, path_regex: str) -> None:
        """Lift a ``deny`` registered with the same method and regex."""
        self._denials.pop((method.upper(), path_regex), None)

    # --------------------------------------------------------------- transport

    def transport(self) -> httpx.MockTransport:
        """A transport that routes every request into this fake."""
        return httpx.MockTransport(self._handle)

    def client(self, token: str | None = "test-token", **kw: Any) -> httpx.Client:
        """An ``httpx.Client`` bound to this fake (``base_url`` is ``api_url``)."""
        headers = httpx.Headers({"Accept": "application/vnd.github+json"})
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        headers.update(kw.pop("headers", None) or {})
        kw.setdefault("base_url", self.api_url)
        kw.setdefault("transport", self.transport())
        return httpx.Client(headers=headers, **kw)

    # ------------------------------------------------------- request processing

    @_locked
    def _handle(self, request: httpx.Request) -> httpx.Response:
        rec = self._record(request)
        rule = self._take_injection(rec)
        if rule is not None:
            if rule.exception is not None:
                raise rule.exception
            status = 200 if rule.status is None else rule.status
            body = rule.json
            if body is None:
                body = {"message": "Injected error"} if status >= 400 else {}
            return self._render(_Reply(status, body, dict(rule.headers)))
        try:
            reply = self._process(request, rec)
        except _Abort as abort:
            reply = abort.reply
        return self._render(reply)

    def _record(self, request: httpx.Request) -> RecordedRequest:
        content = request.read()
        parsed: object | None = None
        if content:
            try:
                parsed = jsonlib.loads(content)
            except ValueError:
                parsed = None
        self._sequence += 1
        rec = RecordedRequest(
            method=request.method.upper(),
            path=request.url.path,
            query=dict(request.url.params.multi_items()),
            headers={key.lower(): value for key, value in request.headers.items()},
            json=parsed,
            time=self._sequence,
            raw_path=request.url.raw_path.split(b"?", 1)[0].decode("utf-8", "replace"),
        )
        self.requests.append(rec)
        return rec

    def _take_injection(self, rec: RecordedRequest) -> _Rule | None:
        for rule in self._injections:
            if _route_matches(rule.method, rule.pattern, rec):
                rule.remaining -= 1
                if rule.remaining <= 0:
                    self._injections.remove(rule)
                return rule
        return None

    def _authorized(self, rec: RecordedRequest) -> bool:
        if self.token is None:
            return True
        scheme, _, credential = rec.headers.get("authorization", "").partition(" ")
        return scheme.lower() in ("bearer", "token") and credential.strip() == self.token

    def _process(self, request: httpx.Request, rec: RecordedRequest) -> _Reply:
        if not self._authorized(rec):
            return _error(401, "Bad credentials")
        for denial in self._denials.values():
            if _route_matches(denial.method, denial.pattern, rec):
                return _error(403, denial.message, documentation_url=DOCS_URL)
        segments = [unquote(part) for part in rec.raw_path.split("/")[1:]]
        for pattern, handlers in self._routes:
            params = _match(pattern, segments)
            if params is None:
                continue
            if "owner" in params and (params["owner"].lower(), params["repo"].lower()) != (
                self.owner.lower(),
                self.name.lower(),
            ):
                return _not_found()
            handler = handlers.get(rec.method)
            if handler is None:
                return _Reply(
                    405,
                    {"message": "Method Not Allowed"},
                    {"Allow": ", ".join(sorted(handlers))},
                )
            return handler(_Call(request, rec, params))
        return _not_found()

    def _render(self, reply: _Reply) -> httpx.Response:
        headers = httpx.Headers(
            {
                "Content-Type": "application/json; charset=utf-8",
                "X-RateLimit-Limit": str(RATE_LIMIT),
                "X-RateLimit-Remaining": str(RATE_LIMIT - 1),
                "X-RateLimit-Reset": str(int(self._clock.timestamp()) + 3600),
            }
        )
        headers.update(reply.headers)
        return httpx.Response(
            reply.status, headers=headers, content=jsonlib.dumps(reply.body).encode()
        )

    def _paginate(self, call: _Call, items: list[T]) -> tuple[list[T], dict[str, str]]:
        """Slice ``items`` by ``per_page``/``page``; add ``Link`` (next and last) if more remain."""
        query = call.rec.query
        per_page = _bounded_int(query.get("per_page"), DEFAULT_PER_PAGE, 1, MAX_PER_PAGE)
        page = _bounded_int(query.get("page"), 1, 1, None)
        last = max(1, math.ceil(len(items) / per_page))
        start = (page - 1) * per_page
        headers: dict[str, str] = {}
        if page < last:
            headers["Link"] = (
                f'<{self._page_url(call, page + 1)}>; rel="next", '
                f'<{self._page_url(call, last)}>; rel="last"'
            )
        return items[start : start + per_page], headers

    def _page_url(self, call: _Call, page: int) -> str:
        pairs: list[tuple[str, str]] = []
        placed = False
        for key, value in call.request.url.params.multi_items():
            if key != "page":
                pairs.append((key, value))
            elif not placed:
                pairs.append((key, str(page)))
                placed = True
        if not placed:
            pairs.append(("page", str(page)))
        return f"{self.api_url}{call.rec.raw_path}?{urlencode(pairs)}"

    # ----------------------------------------------------------------- handlers

    def _get_user(self, call: _Call) -> _Reply:
        return _Reply(200, self._user(self.bot_login))

    def _list_issues(self, call: _Call) -> _Reply:
        query = call.rec.query
        state = query.get("state", "open")
        sort = query.get("sort", "created")
        direction = query.get("direction", "desc")
        _check_choice("Issue", "state", state, ("open", "closed", "all"))
        _check_choice("Issue", "sort", sort, ("created", "updated", "comments"))
        _check_choice("Issue", "direction", direction, ("asc", "desc"))
        wanted = {part.strip() for part in query.get("labels", "").split(",") if part.strip()}
        items = [
            issue
            for issue in self.issues.values()
            if (state == "all" or issue.state == state) and wanted.issubset(issue.labels)
        ]
        sort_keys: dict[str, Callable[[FakeIssue], tuple[Any, int]]] = {
            "created": lambda issue: (issue.created_at, issue.number),
            "updated": lambda issue: (issue.updated_at, issue.number),
            "comments": lambda issue: (len(self.comments.get(issue.number, ())), issue.number),
        }
        items.sort(key=sort_keys[sort], reverse=direction == "desc")
        page, headers = self._paginate(call, items)
        return _Reply(200, [self._issue_json(issue) for issue in page], headers)

    def _get_issue(self, call: _Call) -> _Reply:
        return _Reply(200, self._issue_json(self._issue_or_404(call)))

    def _list_events(self, call: _Call) -> _Reply:
        issue = self._issue_or_404(call)
        events = sorted(self.events.get(issue.number, ()), key=lambda event: event["id"])
        page, headers = self._paginate(call, events)
        return _Reply(200, page, headers)

    def _list_comments(self, call: _Call) -> _Reply:
        issue = self._issue_or_404(call)
        comments = sorted(self.comments.get(issue.number, ()), key=lambda comment: comment.id)
        page, headers = self._paginate(call, comments)
        return _Reply(200, [self._comment_json(issue, comment) for comment in page], headers)

    def _create_comment(self, call: _Call) -> _Reply:
        issue = self._issue_or_404(call)
        payload = _json_object(call)
        body = _check_text_body("IssueComment", payload.get("body"), required=True)
        if not body.strip():
            raise _Abort(
                _validation_failed(
                    {"resource": "IssueComment", "field": "body", "code": "missing_field"}
                )
            )
        comment = self._append_comment(issue, body, self.bot_login)
        return _Reply(201, self._comment_json(issue, comment))

    def _add_labels(self, call: _Call) -> _Reply:
        issue = self._issue_or_404(call)
        payload = _json_body(call)
        raw = payload.get("labels") if isinstance(payload, dict) else payload
        names: list[str] = []
        if isinstance(raw, list):
            for entry in raw:
                name = entry.get("name") if isinstance(entry, dict) else entry
                if not isinstance(name, str) or not name.strip():
                    names = []
                    break
                if name not in names:
                    names.append(name)
        if not names:
            raise _Abort(
                _validation_failed({"resource": "Issue", "field": "labels", "code": "invalid"})
            )
        for name in names:
            if name not in issue.labels:
                self._apply_label(issue, name, self.bot_login)
        return _Reply(200, self._issue_labels_json(issue))

    def _remove_label(self, call: _Call) -> _Reply:
        issue = self._issue_or_404(call)
        name = call.params["name"]
        if name not in issue.labels:
            return _error(404, "Label does not exist")
        self._remove_label_from(issue, name, self.bot_login)
        return _Reply(200, self._issue_labels_json(issue))

    def _list_labels(self, call: _Call) -> _Reply:
        labels = sorted(self.labels.values(), key=lambda label: label.name)
        page, headers = self._paginate(call, labels)
        return _Reply(200, [self._label_json(label) for label in page], headers)

    def _create_label(self, call: _Call) -> _Reply:
        payload = _json_object(call)
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return _validation_failed(
                {"resource": "Label", "field": "name", "code": "missing_field"}
            )
        color = payload.get("color", DEFAULT_LABEL_COLOR)
        if not isinstance(color, str) or not re.fullmatch(r"#?[0-9a-fA-F]{6}", color):
            return _validation_failed({"resource": "Label", "field": "color", "code": "invalid"})
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            return _validation_failed(
                {"resource": "Label", "field": "description", "code": "invalid"}
            )
        if name in self.labels:
            return _validation_failed(
                {"resource": "Label", "code": "already_exists", "field": "name"}
            )
        label = self._ensure_label(name)
        label.color = color.lstrip("#").lower()
        label.description = description
        return _Reply(201, self._label_json(label))

    def _list_pulls(self, call: _Call) -> _Reply:
        query = call.rec.query
        state = query.get("state", "open")
        direction = query.get("direction", "desc")
        _check_choice("PullRequest", "state", state, ("open", "closed", "all"))
        _check_choice("PullRequest", "direction", direction, ("asc", "desc"))
        head = query.get("head")
        base = query.get("base")
        items = [
            pull
            for pull in self.pulls.values()
            if (state == "all" or pull.state == state)
            and (not head or self._head_label(pull) == head)
            and (not base or pull.base_ref == base)
        ]
        items.sort(key=lambda pull: (pull.created_at, pull.number), reverse=direction == "desc")
        page, headers = self._paginate(call, items)
        return _Reply(200, [self._pull_json(pull) for pull in page], headers)

    def _create_pull(self, call: _Call) -> _Reply:
        payload = _json_object(call)
        errors: list[dict[str, object]] = [
            {"resource": "PullRequest", "field": name, "code": "missing_field"}
            for name in ("title", "head", "base")
            if not isinstance(payload.get(name), str) or not payload[name].strip()
        ]
        if errors:
            return _validation_failed(*errors)
        title: str = payload["title"]
        base: str = payload["base"]
        head_owner, sep, head_ref = payload["head"].partition(":")
        if not sep:
            head_owner, head_ref = self.owner, payload["head"]
        if not head_owner or not head_ref:
            return _validation_failed(
                {"resource": "PullRequest", "field": "head", "code": "invalid"}
            )
        body = _check_text_body("PullRequest", payload.get("body"), required=False)
        draft = payload.get("draft", False)
        if not isinstance(draft, bool):
            return _validation_failed(
                {"resource": "PullRequest", "field": "draft", "code": "invalid"}
            )
        if self.ignore_draft:
            draft = False
        for pull in self.pulls.values():
            if (
                pull.state == "open"
                and pull.head_ref == head_ref
                and (pull.head_owner or self.owner) == head_owner
                and pull.base_ref == base
            ):
                return _validation_failed(
                    {
                        "resource": "PullRequest",
                        "code": "custom",
                        "message": f"A pull request already exists for {head_owner}:{head_ref}.",
                    }
                )
        number = self._allocate_number()
        created = self.now()
        pull = FakePull(
            number=number,
            title=title,
            body=body,
            head_ref=head_ref,
            base_ref=base,
            draft=draft,
            state="open",
            html_url=f"{self.web_url}/{self.repo}/pull/{number}",
            created_at=created,
            head_owner=head_owner,
        )
        self.pulls[number] = pull
        self._register_issue(
            FakeIssue(
                number=number,
                title=title,
                body=body,
                state="open",
                state_reason=None,
                author=self.bot_login,
                labels=[],
                created_at=created,
                updated_at=created,
                pull_request=True,
            )
        )
        return _Reply(201, self._pull_json(pull))

    def _get_pull(self, call: _Call) -> _Reply:
        pull = self.pulls.get(self._number(call))
        if pull is None:
            return _not_found()
        return _Reply(200, self._pull_json(pull))

    def _graphql(self, call: _Call) -> _Reply:
        payload = _json_object(call)
        query = payload.get("query")
        if not isinstance(query, str) or "lastEditedAt" not in query:
            return _Reply(200, {"errors": [{"message": "unsupported query"}]})
        variables = payload.get("variables")
        if not isinstance(variables, dict):
            variables = {}
        owner, name, number = variables.get("owner"), variables.get("name"), variables.get("number")
        if not (isinstance(owner, str) and isinstance(name, str)) or (
            owner.lower(),
            name.lower(),
        ) != (self.owner.lower(), self.name.lower()):
            message = f"Could not resolve to a Repository with the name '{owner}/{name}'."
            return _Reply(
                200,
                {
                    "data": {"repository": None},
                    "errors": [{"type": "NOT_FOUND", "path": ["repository"], "message": message}],
                },
            )
        if not isinstance(number, int) or isinstance(number, bool):
            message = "Variable $number of type Int! was provided invalid value"
            return _Reply(200, {"errors": [{"type": "INVALID_VARIABLE", "message": message}]})
        issue = self.issues.get(number)
        if issue is None or issue.pull_request:
            return _Reply(
                200,
                {
                    "data": {"repository": {"issue": None}},
                    "errors": [{"type": "NOT_FOUND", "message": "Could not resolve to an Issue"}],
                },
            )
        edited = None if issue.last_edited_at is None else iso_z(issue.last_edited_at)
        return _Reply(200, {"data": {"repository": {"issue": {"lastEditedAt": edited}}}})

    # ---------------------------------------------------------- state internals

    def _allocate_number(self, explicit: int | None = None) -> int:
        if explicit is None:
            number = self._next_number
        else:
            if explicit < 1 or explicit in self.issues:
                raise ValueError(f"issue number {explicit} is invalid or already in use")
            number = explicit
        self._next_number = max(self._next_number, number + 1)
        return number

    def _register_issue(self, issue: FakeIssue) -> None:
        self.issues[issue.number] = issue
        self.events.setdefault(issue.number, [])
        self.comments.setdefault(issue.number, [])

    def _issue(self, number: int) -> FakeIssue:
        try:
            return self.issues[number]
        except KeyError:
            raise KeyError(f"no such issue #{number}") from None

    def _number(self, call: _Call) -> int:
        raw = call.params["n"]
        if not (raw.isascii() and raw.isdigit()):
            raise _Abort(_not_found())
        return int(raw)

    def _issue_or_404(self, call: _Call) -> FakeIssue:
        issue = self.issues.get(self._number(call))
        if issue is None:
            raise _Abort(_not_found())
        return issue

    def _ensure_label(self, name: str) -> FakeLabel:
        label = self.labels.get(name)
        if label is None:
            label = FakeLabel(name=name, id=self._next_label_id)
            self._next_label_id += 1
            self.labels[name] = label
        return label

    def _append_event(
        self,
        issue: FakeIssue,
        kind: str,
        actor: str,
        *,
        at: datetime | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        stamp = self.now() if at is None else at
        event: dict[str, Any] = {
            "id": self._next_event_id,
            "event": kind,
            "actor": self._user(actor),
            "created_at": iso_z(stamp),
            **extra,
        }
        self._next_event_id += 1
        self.events.setdefault(issue.number, []).append(event)
        issue.updated_at = stamp
        return event

    def _append_comment(self, issue: FakeIssue, body: str, author: str) -> FakeComment:
        stamp = self.now()
        comment = FakeComment(id=self._next_comment_id, body=body, author=author, created_at=stamp)
        self._next_comment_id += 1
        self.comments.setdefault(issue.number, []).append(comment)
        issue.updated_at = stamp
        return comment

    def _apply_label(self, issue: FakeIssue, name: str, actor: str) -> dict[str, Any]:
        label = self._ensure_label(name)
        if name not in issue.labels:
            issue.labels.append(name)
            issue.labels.sort()
        return self._append_event(
            issue, "labeled", actor, label={"name": label.name, "color": label.color}
        )

    def _remove_label_from(self, issue: FakeIssue, name: str, actor: str) -> dict[str, Any]:
        label = self._ensure_label(name)
        issue.labels.remove(name)
        return self._append_event(
            issue, "unlabeled", actor, label={"name": label.name, "color": label.color}
        )

    def _last_label_event(self, number: int, name: str) -> dict[str, Any] | None:
        for event in reversed(self.events.get(number, [])):
            if event["event"] == "labeled" and event["label"]["name"] == name:
                return event
        return None

    def _sync_pull(self, issue: FakeIssue) -> None:
        """Mirror title/body/state onto the FakePull behind a pull-request issue item."""
        pull = self.pulls.get(issue.number)
        if pull is not None:
            pull.title, pull.body, pull.state = issue.title, issue.body, issue.state

    def _head_label(self, pull: FakePull) -> str:
        return f"{pull.head_owner or self.owner}:{pull.head_ref}"

    # ------------------------------------------------------------ serialisation

    def _user(self, login: str) -> dict[str, Any]:
        user_id = self._user_ids.setdefault(login, 1000 + len(self._user_ids))
        return {"login": login, "id": user_id, "type": "User"}

    def _association(self, login: str) -> str:
        return "OWNER" if login.lower() == self.owner.lower() else "NONE"

    def _label_json(self, label: FakeLabel) -> dict[str, Any]:
        return {
            "id": label.id,
            "name": label.name,
            "color": label.color,
            "default": False,
            "description": label.description,
        }

    def _issue_labels_json(self, issue: FakeIssue) -> list[dict[str, Any]]:
        return [
            self._label_json(self.labels.get(name) or FakeLabel(name=name))
            for name in sorted(issue.labels)
        ]

    def _closed_at(self, issue: FakeIssue) -> str | None:
        if issue.state != "closed":
            return None
        for event in reversed(self.events.get(issue.number, [])):
            if event["event"] == "closed":
                return event["created_at"]
        return iso_z(issue.created_at)

    def _issue_json(self, issue: FakeIssue) -> dict[str, Any]:
        number = issue.number
        kind = "pull" if issue.pull_request else "issues"
        data: dict[str, Any] = {
            "id": 500_000 + number,
            "url": f"{self.api_url}/repos/{self.repo}/issues/{number}",
            "html_url": f"{self.web_url}/{self.repo}/{kind}/{number}",
            "number": number,
            "title": issue.title,
            "body": issue.body or None,
            "state": issue.state,
            "state_reason": issue.state_reason if issue.state == "closed" else None,
            "locked": False,
            "user": self._user(issue.author),
            "labels": self._issue_labels_json(issue),
            "assignees": [],
            "comments": len(self.comments.get(number, ())),
            "created_at": iso_z(issue.created_at),
            "updated_at": iso_z(issue.updated_at),
            "closed_at": self._closed_at(issue),
            "author_association": self._association(issue.author),
        }
        if issue.pull_request:
            pull_url = f"{self.api_url}/repos/{self.repo}/pulls/{number}"
            web_url = f"{self.web_url}/{self.repo}/pull/{number}"
            data["pull_request"] = {
                "url": pull_url,
                "html_url": web_url,
                "diff_url": f"{web_url}.diff",
                "patch_url": f"{web_url}.patch",
                "merged_at": None,
            }
        return data

    def _comment_json(self, issue: FakeIssue, comment: FakeComment) -> dict[str, Any]:
        kind = "pull" if issue.pull_request else "issues"
        stamp = iso_z(comment.created_at)
        return {
            "id": comment.id,
            "url": f"{self.api_url}/repos/{self.repo}/issues/comments/{comment.id}",
            "html_url": (
                f"{self.web_url}/{self.repo}/{kind}/{issue.number}#issuecomment-{comment.id}"
            ),
            "issue_url": f"{self.api_url}/repos/{self.repo}/issues/{issue.number}",
            "body": comment.body,
            "user": self._user(comment.author),
            "created_at": stamp,
            "updated_at": stamp,
            "author_association": self._association(comment.author),
        }

    def _pull_json(self, pull: FakePull) -> dict[str, Any]:
        issue = self.issues.get(pull.number)
        return {
            "id": 600_000 + pull.number,
            "url": f"{self.api_url}/repos/{self.repo}/pulls/{pull.number}",
            "html_url": pull.html_url,
            "number": pull.number,
            "state": pull.state,
            "draft": pull.draft,
            "locked": False,
            "title": pull.title,
            "body": pull.body or None,
            "user": self._user(issue.author if issue else self.bot_login),
            "head": {"ref": pull.head_ref, "label": self._head_label(pull)},
            "base": {"ref": pull.base_ref, "label": f"{self.owner}:{pull.base_ref}"},
            "created_at": iso_z(pull.created_at),
            "updated_at": iso_z(issue.updated_at if issue else pull.created_at),
            "closed_at": self._closed_at(issue) if issue else None,
            "merged_at": None,
            "merged": False,
        }
