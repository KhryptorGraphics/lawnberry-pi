"""Security-related FastAPI middleware and helpers.

The middleware layer enforces security headers, correlates requests,
performs lightweight rate limiting for authentication endpoints, and
configures CORS for production deployments.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from collections.abc import Iterable, Sequence

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.context import sanitize_correlation_id, set_correlation_id
from ..core.peer_address import request_peer_key

# Routes under a protected prefix that take no credential to guess (SEC20-08). Each only
# verifies a bearer token the caller already holds, so a 401 there means "that token is dead",
# never "wrong password": the UI sends ``refresh``, ``logout`` and ``profile`` from every browser
# that wakes up after a backend restart (all in-memory sessions are gone) and ``ws-ticket`` before
# every WebSocket (re)connect, so counting those 401s locked the operator out of the very login
# that renews the token; and a 200 there must not reset the failure counters of a client that is
# guessing the credential. What is limited and counted is every OTHER route under the prefix
# (today the two login routes), so a new credential check can never ship unlimited by accident:
# it stays limited until someone lists it here. Matched by exact path, a trailing slash ignored.
UNLIMITED_AUTH_PATHS = frozenset(
    {
        "/api/v2/auth/ws-ticket",
        "/api/v2/auth/refresh",
        "/api/v2/auth/logout",
        "/api/v2/auth/profile",
    }
)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Apply security headers, correlation IDs, and auth rate limiting.

    The auth limiter and lockout count per TCP peer (``core/peer_address.py``); the client's
    ``X-Client-Id`` is not consulted (rotating it used to reset both). Its tables are bounded:
    idle keys are dropped and no table holds more than ``MAX_TRACKED_KEYS``. Only credential
    checks (the login routes) are limited and counted; ``UNLIMITED_AUTH_PATHS`` lists the
    token-only routes that never are, so an expired token cannot lock the login out (SEC20-08).
    """

    MAX_TRACKED_KEYS = 4096
    FAILURE_MEMORY_SECONDS = 3600.0
    PRUNE_INTERVAL_SECONDS = 30.0

    def __init__(
        self,
        app: FastAPI,
        *,
        rate_limit_window_seconds: int = 60,
        rate_limit_max_attempts: int = 3,
        lockout_failures: int = 3,
        lockout_seconds: int = 60,
        protected_prefixes: Sequence[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._window = max(1, rate_limit_window_seconds)
        self._max_attempts = max(1, rate_limit_max_attempts)
        self._lockout_failures = max(1, lockout_failures)
        self._lockout_seconds = max(1, lockout_seconds)
        self._protected_prefixes = tuple(protected_prefixes or ("/api/v2/auth", "/api/v1/auth"))

        self._attempts: dict[str, deque[float]] = {}
        self._failures: dict[str, int] = {}
        self._lockout_until: dict[str, float] = {}
        self._failure_at: dict[str, float] = {}
        self._last_prune = time.time()
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = getattr(request.state, "correlation_id", None)
        if not correlation_id:
            correlation_id = self._extract_correlation_id(request)
            request.state.correlation_id = correlation_id
        set_correlation_id(correlation_id)

        is_protected = self._is_protected_path(request.url.path)
        client_token = self._client_identifier(request)

        if is_protected:
            limited_response = await self._preprocess_rate_limit(client_token)
            if limited_response is not None:
                limited_response.headers.setdefault("X-Correlation-ID", correlation_id)
                self._apply_security_headers(limited_response)
                return limited_response

        try:
            validation_response = None
            if is_protected and request.method in {"POST", "PUT", "PATCH"}:
                validation_response = self._validate_request_content(request)
            if validation_response is not None:
                validation_response.headers.setdefault("X-Correlation-ID", correlation_id)
                self._apply_security_headers(validation_response)
                return validation_response

            response = await call_next(request)
        except Exception:
            raise

        try:
            if is_protected:
                await self._postprocess_rate_limit(client_token, response.status_code)

            self._apply_security_headers(response)
            response.headers.setdefault("X-Correlation-ID", correlation_id)
        finally:
            pass

        return response

    def _extract_correlation_id(self, request: Request) -> str:
        for name in ("X-Correlation-ID", "X-Request-ID", "X-Amzn-Trace-Id"):
            value = sanitize_correlation_id(request.headers.get(name))
            if value:
                return value
        return uuid.uuid4().hex

    def _client_identifier(self, request: Request) -> str:
        return request_peer_key(request)

    def _is_protected_path(self, path: str) -> bool:
        if path.rstrip("/") in UNLIMITED_AUTH_PATHS:
            return False
        return any(path.startswith(prefix) for prefix in self._protected_prefixes)

    async def _preprocess_rate_limit(self, client_token: str) -> Response | None:
        now = time.time()
        async with self._lock:
            self._prune(now)
            lock_until = self._lockout_until.get(client_token)
            if lock_until and now < lock_until:
                retry_after = int(max(1, lock_until - now))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many authentication attempts"},
                    headers={"Retry-After": str(retry_after)},
                )

            attempts = self._attempts.get(client_token)
            if attempts is not None:
                cutoff = now - self._window
                while attempts and attempts[0] < cutoff:
                    attempts.popleft()

                if not attempts:
                    self._attempts.pop(client_token, None)
                elif len(attempts) >= self._max_attempts:
                    oldest = attempts[0]
                    retry_after = int(max(1, self._window - (now - oldest)))
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many authentication attempts"},
                        headers={"Retry-After": str(retry_after)},
                    )
        return None

    async def _postprocess_rate_limit(self, client_token: str, status_code: int) -> None:
        now = time.time()
        async with self._lock:
            self._prune(now)
            attempts = self._attempts.get(client_token)
            if attempts is not None:
                cutoff = now - self._window
                while attempts and attempts[0] < cutoff:
                    attempts.popleft()
                if not attempts:
                    self._attempts.pop(client_token, None)
                    attempts = None

            if status_code == 401:
                if attempts is None:
                    attempts = self._attempts.setdefault(client_token, deque())
                attempts.append(now)
                failures = self._failures.get(client_token, 0) + 1
                self._failures[client_token] = failures
                self._failure_at[client_token] = now
                if failures >= self._lockout_failures:
                    self._lockout_until[client_token] = now + self._lockout_seconds
            elif status_code < 400:
                # Successful authentication resets counters and lockout timers
                self._failures.pop(client_token, None)
                self._failure_at.pop(client_token, None)
                self._lockout_until.pop(client_token, None)
                self._attempts.pop(client_token, None)

    def _prune(self, now: float) -> None:
        """Forget idle keys and cap every table; the caller holds the lock."""
        cap = self.MAX_TRACKED_KEYS
        over = max(len(self._attempts), len(self._failures), len(self._lockout_until)) > cap
        if not over and now - self._last_prune < self.PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        cutoff = now - self._window
        for key, attempts in list(self._attempts.items()):
            while attempts and attempts[0] < cutoff:
                attempts.popleft()
            if not attempts:
                del self._attempts[key]
        for key, until in list(self._lockout_until.items()):
            if now >= until:
                del self._lockout_until[key]
        for key, at in list(self._failure_at.items()):
            if key not in self._lockout_until and now - at > self.FAILURE_MEMORY_SECONDS:
                del self._failure_at[key]
                self._failures.pop(key, None)
        keep = int(cap * 0.9)
        if len(self._attempts) > cap:
            newest_first = sorted(self._attempts, key=lambda k: self._attempts[k][-1], reverse=True)
            for key in newest_first[keep:]:
                del self._attempts[key]
        if len(self._failures) > cap:
            newest_first = sorted(
                self._failures, key=lambda k: self._failure_at.get(k, 0.0), reverse=True
            )
            for key in newest_first[keep:]:
                if key not in self._lockout_until:
                    del self._failures[key]
                    self._failure_at.pop(key, None)
        if len(self._lockout_until) > cap:
            latest_first = sorted(self._lockout_until, key=self._lockout_until.get, reverse=True)
            for key in latest_first[keep:]:
                del self._lockout_until[key]

    def _validate_request_content(self, request: Request) -> Response | None:
        """415 for a request that HAS a body and is not JSON; a body-less POST is fine.

        The UI's logout and refresh are body-less POSTs (axios drops ``Content-Type`` when there
        is no data) and used to be refused here, so a "logout" never revoked anything (SEC20-06).
        ``InputValidationMiddleware`` applies the same rule app-wide.
        """
        if not _has_body(request):
            return None
        content_type = request.headers.get("Content-Type", "").lower()
        if "application/json" not in content_type:
            return JSONResponse(
                status_code=415, content={"detail": "Content-Type must be application/json"}
            )
        return None

    def _apply_security_headers(self, response: Response) -> None:
        security_headers = {
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'",  # noqa: E501
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "X-XSS-Protection": "1; mode=block",
        }
        for header, value in security_headers.items():
            response.headers.setdefault(header, value)


def _has_body(request: Request) -> bool:
    """Whether the request declares a body (``Content-Length`` > 0 or any ``Transfer-Encoding``).

    A malformed length counts as a body, so it still has to be JSON. A body sent with neither
    header (not valid HTTP/1.1) is left to ``InputValidationMiddleware``, which reads the body.
    """
    if request.headers.get("Transfer-Encoding"):
        return True
    length = request.headers.get("Content-Length")
    if length is None:
        return False
    try:
        return int(length) > 0
    except ValueError:
        return True


def _default_allowed_origins() -> list[str]:
    """Origins allowed by CORS (and, for a WebSocket upgrade, besides the page's own origin).

    The UI is served and proxied from the mower itself (``frontend/server.mjs`` forwards ``/api``
    and ``/ws``), so a browser reaches the API from its own origin and needs no CORS at all. The
    default is therefore just the UI server's own loopback origins; the two external
    ``lawnberry.ai`` origins it used to name were credentialed cross-origin grants to hosts this
    repository does not control (SEC20-14). Set ``LAWN_BERRY_ALLOWED_ORIGINS`` (comma-separated)
    to allow others.
    """
    env_value = os.getenv("LAWN_BERRY_ALLOWED_ORIGINS")
    if env_value:
        return [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


def register_security_middleware(
    app: FastAPI, *, allowed_origins: Iterable[str] | None = None
) -> None:
    """Attach security middleware stack to the FastAPI application."""
    origins = list(allowed_origins or _default_allowed_origins())
    app.add_middleware(
        SecurityMiddleware,
        rate_limit_window_seconds=int(os.getenv("AUTH_RATE_LIMIT_WINDOW", "45")),
        rate_limit_max_attempts=int(os.getenv("AUTH_RATE_LIMIT_MAX_ATTEMPTS", "6")),
        lockout_failures=int(os.getenv("AUTH_LOCKOUT_FAILURES", "5")),
        lockout_seconds=int(os.getenv("AUTH_LOCKOUT_SECONDS", "30")),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-Client-Id",
            "X-Correlation-ID",
        ],
        expose_headers=["X-Correlation-ID"],
        max_age=86400,
    )
