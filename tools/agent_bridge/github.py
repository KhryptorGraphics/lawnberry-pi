"""Minimal GitHub REST/GraphQL client with bounded retries, plus approval helpers.

Only the calls the bridge needs are implemented.  The token is sent as a bearer header and is
never logged.  POST requests are retried only when GitHub provably did not process them (the
connection never opened, or the request was rejected for rate limiting); anything ambiguous
surfaces as :class:`GitHubUnavailable` and the caller verifies before repeating.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from .config import GitHubConfig
from .errors import (
    GitHubConflict,
    GitHubError,
    GitHubNotFound,
    GitHubPermissionError,
    GitHubUnavailable,
)
from .models import Approval, Comment, Issue, IssueEvent, PullRequest

__all__ = [
    "GitHubClient",
    "HttpGitHubClient",
    "edited_after",
    "find_approval",
]

LABEL_STYLE: dict[str, tuple[str, str]] = {
    "agent-task": ("5319e7", "Structured task for the local coding-agent bridge"),
    "agent-approved": ("0e8a16", "Owner approval: the bridge may execute this task"),
    "agent-running": ("fbca04", "The bridge is working on this task"),
    "agent-blocked": ("d93f0b", "The bridge could not finish; needs a human"),
    "agent-done": ("1d76db", "The bridge opened a draft PR for this task"),
}

_API_VERSION = "2022-11-28"
_MAX_PAGES = 30
_LAST_EDITED_QUERY = (
    "query($owner: String!, $name: String!, $number: Int!) {"
    " repository(owner: $owner, name: $name) { issue(number: $number) { lastEditedAt } } }"
)


class GitHubClient(Protocol):
    """What the worker needs from GitHub (the HTTP client implements it)."""

    def list_candidate_issues(self, labels: Sequence[str]) -> list[Issue]: ...

    def get_issue(self, number: int) -> Issue: ...

    def issue_events(self, number: int) -> list[IssueEvent]: ...

    def issue_last_edited_at(self, number: int) -> str | None: ...

    def add_labels(self, number: int, labels: Sequence[str]) -> None: ...

    def remove_label(self, number: int, label: str) -> None: ...

    def ensure_labels(self, names: Sequence[str]) -> list[str]: ...

    def list_comments(self, number: int) -> list[Comment]: ...

    def create_comment(self, number: int, body: str) -> Comment: ...

    def viewer_login(self) -> str: ...

    def find_pull_request(self, head_branch: str) -> PullRequest | None: ...

    def create_pull_request(
        self, *, title: str, head: str, base: str, body: str
    ) -> PullRequest: ...

    def close(self) -> None: ...


def find_approval(
    events: Sequence[IssueEvent], approved_label: str, current_labels: frozenset[str]
) -> Approval | None:
    """The event through which the approval label currently on the issue was applied."""
    if approved_label not in current_labels:
        return None
    for event in sorted(events, key=lambda e: e.id, reverse=True):
        if event.label != approved_label:
            continue
        if event.event == "unlabeled":
            return None
        if event.event == "labeled":
            return Approval(event_id=event.id, actor=event.actor or "", created_at=event.created_at)
    return None


def edited_after(
    approval: Approval, events: Sequence[IssueEvent], last_edited_at: str | None
) -> bool:
    """True when the title or body changed at or after the moment of approval (fail closed)."""
    if last_edited_at and last_edited_at >= approval.created_at:
        return True
    return any(e.event == "renamed" and e.created_at >= approval.created_at for e in events)


class HttpGitHubClient:
    def __init__(
        self,
        cfg: GitHubConfig,
        repository: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = cfg
        self._owner, self._repo = repository.split("/", 1)
        self._sleep = sleep
        self._wall = wall_clock
        self._api_url = cfg.api_url.rstrip("/")
        self._viewer: str | None = None
        self._client = httpx.Client(
            base_url=self._api_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": _API_VERSION,
                "User-Agent": cfg.user_agent,
            },
            timeout=cfg.request_timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        return f"HttpGitHubClient({self._owner}/{self._repo})"

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------------- transport
    def _graphql_url(self) -> str:
        if self._api_url.endswith("/api/v3"):
            return self._api_url[: -len("/api/v3")] + "/api/graphql"
        return self._api_url + "/graphql"

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        idempotent: bool | None = None,
    ) -> httpx.Response:
        safe_to_retry = method in {"GET", "HEAD", "DELETE"} if idempotent is None else idempotent
        attempts = self._cfg.max_retries + 1
        last_problem = "request failed"
        for attempt in range(attempts):
            final = attempt == attempts - 1
            try:
                response = self._client.request(method, url, params=params, json=json)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_problem = f"cannot connect to GitHub ({type(exc).__name__})"
                if final:
                    raise GitHubUnavailable(last_problem) from exc
                self._backoff(attempt)
                continue
            except httpx.TransportError as exc:
                last_problem = f"network error talking to GitHub ({type(exc).__name__})"
                if final or not safe_to_retry:
                    raise GitHubUnavailable(last_problem) from exc
                self._backoff(attempt)
                continue
            status = response.status_code
            if status < 300:
                return response
            if status < 400:
                raise GitHubError(
                    f"GitHub answered HTTP {status}: redirects are never followed, so the "
                    "repository may have been renamed or moved - check the `repository` setting",
                    status=status,
                )
            wait = self._rate_limit_wait(response)
            if wait is not None:
                if wait > self._cfg.rate_limit_max_wait_seconds:
                    raise GitHubUnavailable(
                        f"GitHub rate limit: retry in {wait:.0f}s exceeds the configured wait",
                        status=status,
                    )
                last_problem = f"GitHub rate limit (HTTP {status})"
                if final:
                    raise GitHubUnavailable(last_problem, status=status)
                self._sleep(wait)
                continue
            if status >= 500:
                last_problem = f"GitHub server error (HTTP {status})"
                if final or not safe_to_retry:
                    raise GitHubUnavailable(last_problem, status=status)
                self._backoff(attempt)
                continue
            self._raise_for(response)
        raise GitHubUnavailable(last_problem)  # pragma: no cover - loop always returns/raises

    def _backoff(self, attempt: int) -> None:
        delay = min(30.0, self._cfg.retry_backoff_seconds * (2**attempt))
        if delay > 0:
            self._sleep(delay)

    def _rate_limit_wait(self, response: httpx.Response) -> float | None:
        if response.status_code not in (403, 429):
            return None
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return float(retry_after)
        if response.headers.get("x-ratelimit-remaining") == "0":
            reset = response.headers.get("x-ratelimit-reset", "")
            if reset.isdigit():
                return max(0.0, float(reset) - self._wall()) + 1.0
        message = self._message(response).lower()
        if "rate limit" in message or "abuse detection" in message:
            return 60.0
        return None

    @staticmethod
    def _message(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(data, dict):
            return str(data.get("message", ""))[:300]
        return ""

    def _raise_for(self, response: httpx.Response) -> None:
        status = response.status_code
        message = self._message(response) or f"HTTP {status}"
        if status in (401, 403):
            raise GitHubPermissionError(f"GitHub denied the request: {message}", status=status)
        if status == 404:
            raise GitHubNotFound(f"GitHub object not found: {message}", status=status)
        if status in (409, 422):
            raise GitHubConflict(f"GitHub rejected the request: {message}", status=status)
        raise GitHubError(f"GitHub error HTTP {status}: {message}", status=status)

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubError("GitHub returned invalid JSON", status=response.status_code) from exc

    def _on_api(self, url: str) -> bool:
        """True when ``url`` is on the configured API origin and under its path prefix."""
        default_ports = {"https": 443, "http": 80}
        try:
            base, target = urlsplit(self._api_url), urlsplit(url)
            origins = [
                (p.scheme, p.hostname, p.port or default_ports.get(p.scheme))
                for p in (base, target)
            ]
        except ValueError:
            return False
        if target.username is not None or target.password is not None:
            return False
        prefix = base.path.rstrip("/")
        return origins[0] == origins[1] and (
            target.path == prefix or target.path.startswith(prefix + "/")
        )

    def _paginate(self, path: str, params: Mapping[str, Any] | None = None) -> list[Any]:
        items: list[Any] = []
        url: str | None = path
        query: Mapping[str, Any] | None = {"per_page": 100, **(params or {})}
        for _ in range(_MAX_PAGES):
            response = self._request("GET", url, params=query)
            data = self._json(response)
            if not isinstance(data, list):
                raise GitHubError("expected a JSON list from GitHub")
            items.extend(data)
            nxt = response.links.get("next", {}).get("url")
            if not nxt:
                return items
            if not self._on_api(str(nxt)):
                raise GitHubError("refusing to follow a pagination link to another host")
            url, query = str(nxt), None
        raise GitHubError("too many pages from GitHub")

    def _repo_path(self, suffix: str) -> str:
        return f"/repos/{self._owner}/{self._repo}{suffix}"

    # ---------------------------------------------------------------- parsing
    @staticmethod
    def _issue(data: Mapping[str, Any]) -> Issue:
        return Issue(
            number=int(data["number"]),
            title=str(data.get("title") or ""),
            body=str(data.get("body") or ""),
            state=str(data.get("state") or "open"),
            state_reason=data.get("state_reason"),
            author=str((data.get("user") or {}).get("login") or ""),
            labels=frozenset(
                str(label.get("name") if isinstance(label, dict) else label)
                for label in data.get("labels") or []
            ),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            html_url=str(data.get("html_url") or ""),
            is_pull_request="pull_request" in data,
        )

    @staticmethod
    def _pull(data: Mapping[str, Any]) -> PullRequest:
        return PullRequest(
            number=int(data["number"]),
            html_url=str(data.get("html_url") or ""),
            state=str(data.get("state") or ""),
            draft=bool(data.get("draft")),
            head_ref=str((data.get("head") or {}).get("ref") or ""),
            base_ref=str((data.get("base") or {}).get("ref") or ""),
            title=str(data.get("title") or ""),
        )

    # ---------------------------------------------------------------- issues
    def list_candidate_issues(self, labels: Sequence[str]) -> list[Issue]:
        raw = self._paginate(
            self._repo_path("/issues"),
            {"labels": ",".join(labels), "state": "open", "sort": "created", "direction": "asc"},
        )
        return [self._issue(item) for item in raw if "pull_request" not in item]

    def get_issue(self, number: int) -> Issue:
        return self._issue(self._json(self._request("GET", self._repo_path(f"/issues/{number}"))))

    def issue_events(self, number: int) -> list[IssueEvent]:
        events: list[IssueEvent] = []
        for item in self._paginate(self._repo_path(f"/issues/{number}/events")):
            label = item.get("label") or {}
            rename = item.get("rename") or {}
            events.append(
                IssueEvent(
                    id=int(item["id"]),
                    event=str(item.get("event") or ""),
                    actor=(item.get("actor") or {}).get("login"),
                    created_at=str(item.get("created_at") or ""),
                    label=label.get("name"),
                    rename_from=rename.get("from"),
                    rename_to=rename.get("to"),
                )
            )
        events.sort(key=lambda e: e.id)
        return events

    def issue_last_edited_at(self, number: int) -> str | None:
        response = self._request(
            "POST",
            self._graphql_url(),
            json={
                "query": _LAST_EDITED_QUERY,
                "variables": {"owner": self._owner, "name": self._repo, "number": number},
            },
            idempotent=True,  # a read
        )
        data = self._json(response)
        if not isinstance(data, dict):
            raise GitHubError("unexpected GraphQL response")
        issue = ((data.get("data") or {}).get("repository") or {}).get("issue")
        if issue is None:
            errors = data.get("errors") or [{"message": "issue not found"}]
            raise GitHubNotFound(f"GraphQL: {errors[0].get('message', 'issue not found')}")
        value = issue.get("lastEditedAt")
        return str(value) if value else None

    # ---------------------------------------------------------------- identity
    def viewer_login(self) -> str:
        """Lowercase login of the account the token belongs to (``GET /user``, cached).

        The bridge recognises its own comments by this login: a hidden marker is only trusted
        when the account that wrote it is this one, because anyone can copy a marker text.
        """
        if self._viewer is None:
            try:
                data = self._json(self._request("GET", "/user"))
            except GitHubPermissionError as exc:
                raise GitHubPermissionError(
                    f"{exc} (GET /user: the bridge token must be able to read its own account, "
                    "which any personal access token can, so the bridge can recognise its own "
                    "comments)",
                    status=exc.status,
                ) from exc
            login = data.get("login") if isinstance(data, dict) else None
            if not isinstance(login, str) or not login.strip():
                raise GitHubError("GitHub returned an unexpected body for GET /user")
            self._viewer = login.strip().lower()
        return self._viewer

    # ---------------------------------------------------------------- labels
    def add_labels(self, number: int, labels: Sequence[str]) -> None:
        if labels:
            self._request(
                "POST", self._repo_path(f"/issues/{number}/labels"), json={"labels": list(labels)}
            )

    def remove_label(self, number: int, label: str) -> None:
        try:
            self._request(
                "DELETE", self._repo_path(f"/issues/{number}/labels/{quote(label, safe='')}")
            )
        except GitHubNotFound:
            return

    def ensure_labels(self, names: Sequence[str]) -> list[str]:
        existing = {str(item.get("name")) for item in self._paginate(self._repo_path("/labels"))}
        created: list[str] = []
        for name in names:
            if name in existing:
                continue
            color, description = LABEL_STYLE.get(name, ("ededed", ""))
            try:
                self._request(
                    "POST",
                    self._repo_path("/labels"),
                    json={"name": name, "color": color, "description": description},
                    idempotent=False,
                )
            except GitHubConflict:
                continue
            created.append(name)
        return created

    # ---------------------------------------------------------------- comments
    def list_comments(self, number: int) -> list[Comment]:
        return [
            Comment(
                id=int(item["id"]),
                body=str(item.get("body") or ""),
                author=str((item.get("user") or {}).get("login") or ""),
                created_at=str(item.get("created_at") or ""),
            )
            for item in self._paginate(self._repo_path(f"/issues/{number}/comments"))
        ]

    def create_comment(self, number: int, body: str) -> Comment:
        data = self._json(
            self._request(
                "POST", self._repo_path(f"/issues/{number}/comments"), json={"body": body}
            )
        )
        return Comment(
            id=int(data["id"]),
            body=str(data.get("body") or ""),
            author=str((data.get("user") or {}).get("login") or ""),
            created_at=str(data.get("created_at") or ""),
        )

    # ---------------------------------------------------------------- pull requests
    def find_pull_request(self, head_branch: str) -> PullRequest | None:
        raw = self._paginate(
            self._repo_path("/pulls"),
            {"head": f"{self._owner}:{head_branch}", "state": "all"},
        )
        matches = [
            self._pull(item) for item in raw if (item.get("head") or {}).get("ref") == head_branch
        ]
        if not matches:
            return None
        open_ones = [pr for pr in matches if pr.state == "open"]
        return (open_ones or matches)[0]

    def create_pull_request(self, *, title: str, head: str, base: str, body: str) -> PullRequest:
        try:
            data = self._json(
                self._request(
                    "POST",
                    self._repo_path("/pulls"),
                    json={"title": title, "head": head, "base": base, "body": body, "draft": True},
                )
            )
        except GitHubConflict:
            existing = self.find_pull_request(head)
            if existing is not None:
                return existing
            raise
        return self._pull(data)
