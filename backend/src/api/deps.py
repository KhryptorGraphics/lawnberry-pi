"""Shared FastAPI dependencies for the API layer."""

from __future__ import annotations

from fastapi import HTTPException, Request

from ..core.auth_posture import operator_auth_required
from ..services.auth_service import primary_auth_service


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token or None
    return None


async def require_operator_auth(request: Request) -> None:
    """Require operator authentication on every state-changing route that is not an E-stop.

    Accepts ONE credential: a bearer token issued by ``/api/v2/auth/login`` (or ``/api/v1``)
    that still has a live server-side session. A manual-control session id (from
    ``/control/manual-unlock``) is NOT an operator credential and is not accepted here: it is
    minted only for an operator who is already logged in and re-enters the credential, and it
    travels in request bodies and camera URLs that end up in logs, so it can only ever be an
    additional factor for ``/control/drive`` and ``/control/blade``, never a replacement for
    the login (it used to be accepted via ``X-Session-Id``, which turned any leaked or minted
    session id into a full operator login).

    Enforcement follows ``OPERATOR_AUTH_REQUIRED``, parsed fail-closed by
    :func:`backend.src.core.auth_posture.operator_auth_required` (only an explicit
    ``0``/``false``/``no``/``off`` disables it; unset keeps it on). The test suite disables it
    so contract tests can call writes without a token, and the systemd unit sets it to ``1``.
    The E-stop POSTs deliberately do not use this dependency; they must work with no
    credential at all (``tests/integration/test_operator_auth_coverage.py`` lists every route
    that is open and why).
    """
    if not operator_auth_required():
        return

    token = _bearer_token(request)
    if token:
        try:
            session = await primary_auth_service.verify_token(token)
        except Exception:
            session = None
        if session:
            return

    raise HTTPException(
        status_code=401,
        detail="Operator authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
