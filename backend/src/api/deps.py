"""Shared FastAPI dependencies for the API layer."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import HTTPException, Request

from ..core.globals import _manual_control_sessions
from ..services.auth_service import primary_auth_service


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token or None
    return None


def _has_valid_manual_session(request: Request) -> bool:
    sid = request.headers.get("X-Session-Id") or request.headers.get("x-session-id")
    if not sid:
        return False
    now = datetime.now(UTC)
    for entry in _manual_control_sessions.values():
        if entry.get("session_id") == sid and entry.get("expires_at", now) > now:
            return True
    return False


async def require_operator_auth(request: Request) -> None:
    """Require operator authentication on config-changing write endpoints.

    Accepts EITHER a valid bearer token (issued by ``/api/v2/auth/login``) OR a
    valid manual-control ``session_id`` (``X-Session-Id`` header). Enforcement is
    gated by ``OPERATOR_AUTH_REQUIRED`` (default ``"1"`` — secure by default);
    the test suite disables it so existing contract tests keep passing, and the
    systemd unit sets it explicitly in production.
    """
    if os.getenv("OPERATOR_AUTH_REQUIRED", "1") != "1":
        return

    token = _bearer_token(request)
    if token:
        try:
            session = await primary_auth_service.verify_token(token)
        except Exception:
            session = None
        if session:
            return

    if _has_valid_manual_session(request):
        return

    raise HTTPException(status_code=401, detail="Operator authentication required")
