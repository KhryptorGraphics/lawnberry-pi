"""Operator login, token refresh/logout and the manual-control unlock.

Authority model (see ``docs/authentication-config.md``):

* ``POST /auth/login`` exchanges the one shared operator credential for a bearer token. That
  token is the ONLY credential ``api.deps.require_operator_auth`` accepts.
* ``POST /control/manual-unlock`` is a step-up: it requires that bearer token AND the credential
  again (the bcrypt ``password_hash`` of the security settings, else ``MANUAL_CONTROL_PASSWORD``,
  else the operator credential), and mints a random manual-control session id. That id is an
  additional factor the drive and blade routes check in the request body; it is not an operator
  credential. Nothing is minted for an unverified identity: Cloudflare Access assertions are
  not accepted (they were decoded but never verified, so any header value unlocked control).
"""

import functools
import hmac
import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import pyotp
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from pydantic import BaseModel

from ...core.auth_posture import operator_auth_required, sim_open_auth_enabled
from ...core.globals import _manual_control_sessions, _security_settings
from ...core.peer_address import (
    client_host,
    is_loopback_address,
    is_trusted_proxy_peer,
    peer_key,
    request_peer_key,
)
from ...core.ws_policy import (
    ANON_ANY,
    ANON_LOOPBACK,
    REFUSAL_LOG_INTERVAL_S,
    WsLimitError,
    WsSlot,
    anonymous_telemetry_mode,
    ws_registry,
)
from ...middleware.security import _default_allowed_origins
from ...models.auth_security_config import SecurityLevel
from ...models.user_session import Permission, UserRole, UserSession
from ...services.auth_service import AuthenticationError, primary_auth_service, run_password_check
from ..deps import require_operator_auth
from .settings import get_security_level

logger = logging.getLogger(__name__)
router = APIRouter()

# At most this many manual-control sessions are live at once; the soonest to expire makes room.
MANUAL_SESSION_LIMIT = 64
# What the auth-disabled unlock hands back (nothing resolves it: with auth off, any id passes).
AUTH_DISABLED_SESSION_ID = "manual-auth-disabled"
CLOUDFLARE_UNSUPPORTED = (
    "Cloudflare Access assertions are not verified by this backend, so they cannot unlock "
    "manual control; unlock with the operator password"
)


# Models
class ManualUnlockRequest(BaseModel):
    method: str | None = None
    password: str | None = None
    totp_code: str | None = None


class ManualUnlockResponse(BaseModel):
    authorized: bool
    session_id: str
    expires_at: str
    principal: str | None = None
    source: str = "manual_control"


class ManualUnlockStatusResponse(BaseModel):
    authorized: bool
    session_id: str | None = None
    expires_at: str | None = None
    principal: str | None = None
    reason: str | None = None


class AuthLoginRequest(BaseModel):
    # Support both shared-credential and username/password payloads
    credential: str | None = None
    username: str | None = None
    password: str | None = None


class UserOut(BaseModel):
    id: str
    username: str
    role: str = "admin"
    created_at: datetime = datetime.now(UTC)


class AuthResponse(BaseModel):
    # Back-compat fields
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: UserOut
    # Contract-required convenience fields
    token: str
    expires_at: datetime


class RefreshResponse(BaseModel):
    access_token: str
    token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    expires_at: datetime


# Helpers
def _manual_session_expiry(default_minutes: int | None = None) -> datetime:
    minutes = default_minutes or 60
    try:
        minutes = int(minutes)
    except Exception:
        minutes = 60
    return datetime.now(UTC) + timedelta(minutes=max(1, minutes))


def _issue_manual_session(expires_at: datetime, principal: str | None) -> dict[str, Any]:
    """Mint a manual-control session whose id is random, not derived from any secret.

    The id used to be ``sha256`` of the unlocking password (and peer address) or of the
    Cloudflare token: anyone who saw an id (the camera stream URL carries it, so it reaches
    access logs) could test password guesses against it offline.
    """
    now = datetime.now(UTC)
    for key in [k for k, v in _manual_control_sessions.items() if v["expires_at"] <= now]:
        _manual_control_sessions.pop(key, None)
    while len(_manual_control_sessions) >= MANUAL_SESSION_LIMIT:
        soonest = min(
            _manual_control_sessions, key=lambda k: _manual_control_sessions[k]["expires_at"]
        )
        _manual_control_sessions.pop(soonest, None)

    session_id = f"manual-{secrets.token_urlsafe(24)}"
    entry = {"session_id": session_id, "expires_at": expires_at, "principal": principal}
    _manual_control_sessions[session_id] = entry
    return entry


def _auth_disabled_unlock() -> tuple[str, str]:
    """Session id and expiry handed back while ``OPERATOR_AUTH_REQUIRED`` is off."""
    timeout_minutes = getattr(_security_settings, "session_timeout_minutes", 60)
    return AUTH_DISABLED_SESSION_ID, _manual_session_expiry(timeout_minutes).isoformat()


def _resolve_manual_session(session_id: str | None) -> dict[str, Any]:
    if not operator_auth_required():
        return {"principal": "operator", "session_id": "auth-disabled"}

    token = (session_id or "").strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Manual control session required")

    now = datetime.now(UTC)
    matched_entry: dict[str, Any] | None = None
    expired: list[str] = []
    wanted = token.encode("utf-8")
    for key, entry in _manual_control_sessions.items():
        expires_at: datetime = entry.get("expires_at", now)
        if expires_at <= now:
            expired.append(key)
            continue
        if secrets.compare_digest(str(entry.get("session_id", "")).encode("utf-8"), wanted):
            matched_entry = entry

    for key in expired:
        _manual_control_sessions.pop(key, None)

    if matched_entry is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Manual control session invalid or expired"
        )

    return matched_entry


def _manual_unlock_principal(request: Request, method: str) -> str:
    try:
        host = client_host(request)  # the real client behind a trusted proxy, else the peer
    except Exception:
        host = None
    host_str = str(host).strip() if host else ""
    return f"{method}:{host_str}" if host_str else method


async def _manual_password_matches(pwd: str) -> bool:
    """Compare ``pwd`` with the first usable manual-unlock secret; ``False`` if none matches.

    In order: the bcrypt ``password_hash`` of the security settings, ``MANUAL_CONTROL_PASSWORD``
    (constant-time compare), else the shared operator credential. With none usable (no
    credential configured, or only a published default) nothing matches. bcrypt runs on the
    hash pool, never on the event loop.
    """
    if _security_settings.password_hash:
        try:
            return await run_password_check(
                primary_auth_service.password_manager.matches,
                pwd,
                _security_settings.password_hash,
            )
        except ValueError as exc:
            logger.error("Invalid password hash configuration: %s", exc)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password configuration invalid"
            ) from exc
    fallback = os.getenv("MANUAL_CONTROL_PASSWORD")
    if fallback:
        return hmac.compare_digest(pwd.encode("utf-8"), fallback.encode("utf-8"))
    return await primary_auth_service.verify_operator_credential(pwd)


async def _validate_manual_password(password: str | None) -> str:
    """Check the password of a manual unlock; never fail open.

    A blank password is a 400 and a wrong one a 401. (The caller wraps this in the per-peer
    unlock throttle, so a wrong guess counts.)
    """
    pwd = (password or "").strip()
    if not pwd:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Password required for manual unlock"
        )
    if not await _manual_password_matches(pwd):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    return pwd


async def _throttled_unlock(request: Request, check: Callable[[], Awaitable[Any]]) -> Any:
    """Run ``check`` under the per-peer unlock throttle (same limiter, separate key, as login).

    The unlock is a second place to guess the operator credential, so it gets the same
    attempt-rate limit and failure lockout as the login, keyed on the TCP peer, and an IPv6
    client the same coarse /64 lockout (``AuthService.group_limiter``, also under its own
    ``unlock:`` key: a LAN host rotating its own addresses must not get a fresh guess per
    address). Only a 401 (a wrong secret) counts as a failure.
    """
    limiter = primary_auth_service.rate_limiter
    host = client_host(request)
    key = f"unlock:{peer_key(host)}"
    group_limiter = primary_auth_service.group_limiter
    group = primary_auth_service.group_key(host)
    group_key = f"unlock:{group}" if group is not None else None
    try:
        limiter.assert_not_locked(key)
        if group_key is not None:
            group_limiter.assert_not_locked(group_key)
        retry_after = limiter.record_attempt(key)
        if retry_after is not None:
            raise AuthenticationError(
                "Too many authentication attempts", status_code=429, retry_after=retry_after
            )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.detail, headers=exc.headers
        ) from exc

    try:
        result = await check()
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            started = [limiter.record_failure(key)]
            if group_key is not None:
                started.append(group_limiter.record_failure(group_key))
            lockout = max((seconds for seconds in started if seconds is not None), default=None)
            if lockout is not None:
                headers = dict(exc.headers or {})
                headers["Retry-After"] = str(lockout)
                raise HTTPException(
                    status_code=exc.status_code, detail=exc.detail, headers=headers
                ) from exc
        raise
    limiter.reset(key)
    if group_key is not None:
        group_limiter.reset(group_key)
    return result


def _verify_totp_code(code: str | None) -> tuple[bool, bool]:
    config = _security_settings.totp_config
    if not config or not getattr(config, "enabled", True):
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED, detail="TOTP authentication not configured"
        )
    normalized = (code or "").strip()
    if not normalized:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="TOTP code required")
    backup_codes = list(getattr(config, "backup_codes", []) or [])
    if normalized in backup_codes:
        try:
            backup_codes.remove(normalized)
            config.backup_codes = backup_codes  # type: ignore[attr-defined]
        except Exception:
            pass
        return True, True
    try:
        totp = pyotp.TOTP(config.secret, digits=config.digits, interval=config.period)
    except Exception as exc:  # pragma: no cover - misconfiguration
        logger.error("Invalid TOTP configuration: %s", exc)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid TOTP configuration"
        ) from exc
    if not totp.verify(normalized, valid_window=1):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid TOTP code")
    return True, False


def _client_ip(request: Request) -> str | None:
    """The client address for login throttling and the session record (proxy-aware)."""
    host = client_host(request)
    return str(host) if host else None


def _extract_bearer_token(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    scheme, _, token = auth_header.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip() or None


async def _require_session(request: Request) -> UserSession:
    token = _extract_bearer_token(request.headers.get("Authorization"))
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    session = await primary_auth_service.verify_token(token)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return session


# Single-use WebSocket tickets. A browser cannot set an Authorization header on a WebSocket
# handshake, and the 8 hour login token must not travel in a URL (uvicorn logs the query
# string), so the UI exchanges its bearer token for a 30 s, single-use ticket per connection.
WS_TICKET_TTL_SECONDS = 30
WS_TICKET_LIMIT = 256
_ws_tickets: dict[str, dict[str, Any]] = {}


class WsTicketResponse(BaseModel):
    ticket: str
    expires_in: int


def _mint_ws_ticket(session_id: str) -> str:
    now = datetime.now(UTC)
    for key in [k for k, v in _ws_tickets.items() if v["expires_at"] <= now]:
        _ws_tickets.pop(key, None)
    while len(_ws_tickets) >= WS_TICKET_LIMIT:
        _ws_tickets.pop(min(_ws_tickets, key=lambda k: _ws_tickets[k]["expires_at"]), None)
    ticket = secrets.token_urlsafe(24)
    expires_at = now + timedelta(seconds=WS_TICKET_TTL_SECONDS)
    _ws_tickets[ticket] = {"session_id": session_id, "expires_at": expires_at}
    return ticket


async def _session_for_ws_ticket(ticket: str) -> UserSession | None:
    """Redeem a ticket: the live login session it was minted for, else ``None``. Single use."""
    entry = _ws_tickets.pop(ticket, None)  # popped first: a ticket can never be presented twice
    if entry is None or entry["expires_at"] <= datetime.now(UTC):
        return None
    return await primary_auth_service.verify_session(entry["session_id"])


def _peek_ws_ticket_session_id(ticket: str) -> str | None:
    """The live session a ticket was minted for, WITHOUT redeeming it (``None``: not live).

    For the handshake budget key only: a ticket that belongs to a live login is counted as that
    login, anything else as the peer. Never consumes, never logs.
    """
    entry = _ws_tickets.get(ticket)
    if entry is None or entry["expires_at"] <= datetime.now(UTC):
        return None
    session = primary_auth_service.active_sessions.get(entry["session_id"])
    if session is None or session.is_expired():
        return None
    return str(entry["session_id"])


def _ws_ticket_of(websocket: WebSocket) -> str | None:
    """The ``?ticket=`` of a handshake, else ``None`` (never raises)."""
    try:
        getter = getattr(getattr(websocket, "query_params", None), "get", None)
        ticket = getter("ticket") if callable(getter) else None
    except Exception:
        return None
    return str(ticket) if ticket else None


def _anonymous_session(client_ip: str | None, user_agent: str | None) -> UserSession:
    """The throw-away, read-only identity of an anonymous socket (never stored anywhere)."""
    session = UserSession(client_ip=client_ip, user_agent=user_agent)
    session.security_context.role = UserRole.VIEWER
    session.security_context.permissions = [Permission.VIEW_STATUS, Permission.VIEW_TELEMETRY]
    return session


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _origin_parts(value: str) -> tuple[str, str, int | None] | None:
    """``(scheme, host, port)`` of an ``Origin`` value (default ports filled in), else ``None``."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    return scheme, parts.hostname.lower(), port or _DEFAULT_PORTS.get(scheme)


def _host_parts(value: str, scheme: str) -> tuple[str, int | None] | None:
    """``(host, port)`` of a ``Host`` / ``X-Forwarded-Host`` value, the port defaulted by scheme."""
    try:
        parts = urlsplit("//" + value.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname:
        return None
    return parts.hostname.lower(), port or _DEFAULT_PORTS.get(scheme)


def _origin_allowed(websocket: WebSocket) -> bool:
    """May a browser page at this ``Origin`` open an ANONYMOUS socket? (CORS does not cover WS.)

    No ``Origin`` header means not a browser (a script, curl): allowed here, the other rules still
    apply. A browser always sends it, and a page the operator opened in the browser that sits on
    the mower (or on the LAN) must not be able to read the live stream cross-origin. Accepted:
    an origin on the CORS allow-list (``LAWN_BERRY_ALLOWED_ORIGINS``, default the UI's own), or
    the page's own origin, i.e. its host:port is the ``Host`` of the request (or the
    ``X-Forwarded-Host`` a declared forwarder set: the UI server rewrites ``Host`` to the
    backend's). ``Origin: null`` and anything unparseable is refused.
    """
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    parsed = _origin_parts(origin)
    if parsed is None:
        return False
    if any(_origin_parts(allowed) == parsed for allowed in _default_allowed_origins()):
        return True
    scheme, host, port = parsed
    hosts = [websocket.headers.get("host")]
    if is_trusted_proxy_peer(websocket):
        forwarded = websocket.headers.get("x-forwarded-host")
        hosts.append(forwarded.split(",")[-1] if forwarded else None)
    return any(h and _host_parts(h, scheme) == (host, port) for h in hosts)


async def _admit(websocket: WebSocket, *, command_capable: bool) -> tuple[UserSession, bool]:
    """The admission decision: ``(session, anonymous)``, or raise 401. See ``admit_websocket``.

    A bad credential is checked QUIETLY (no WARNING per attempt, ``verify_token(quiet=True)``):
    the refusal is one line per peer per minute, written by ``admit_websocket``.
    """
    token = _extract_bearer_token(websocket.headers.get("Authorization"))
    if token:
        session = await primary_auth_service.verify_token(token, quiet=True)
        if not session:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return session, False

    ticket = _ws_ticket_of(websocket)
    if ticket:
        session = await _session_for_ws_ticket(ticket)
        if session is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return session, False

    # No credential from here on.
    if command_capable and operator_auth_required():
        raise HTTPException(status_code=401, detail="Unauthorized")
    client_ip = client_host(websocket)
    user_agent = websocket.headers.get("User-Agent")
    if sim_open_auth_enabled():
        # The explicit simulator opt-in (never in production): nothing is enforced anywhere, the
        # sockets follow. Switching OPERATOR_AUTH_REQUIRED off does NOT open the sockets: the
        # anonymous streams below stay a separate, explicit opt-in.
        return UserSession.create_operator_session(client_ip, user_agent), False
    mode = anonymous_telemetry_mode()
    if mode == ANON_ANY or (mode == ANON_LOOPBACK and is_loopback_address(client_ip)):
        if _origin_allowed(websocket):
            return _anonymous_session(client_ip, user_agent), True
    raise HTTPException(status_code=401, detail="Unauthorized")


async def _authorize_websocket(
    websocket: WebSocket, *, command_capable: bool = False
) -> UserSession:
    """Who may open this socket (the policy only; ``admit_websocket`` also takes a slot).

    A credential is honoured on every path: ``Authorization: Bearer <login token>`` (non-browser
    clients) or a single-use ``?ticket=`` from ``POST /auth/ws-ticket`` (browsers). A presented
    but invalid credential is always refused, whoever the peer is. The login token in a query
    string (``access_token``/``token``/``auth_token``) is not read at all.

    With no credential NO peer is admitted, the loopback included: a browser on the mower used to
    count as "loopback", but every ingress the product ships (a Cloudflare or ngrok agent on the
    Pi, nginx) reaches the backend from a loopback address, so remote visitors looked exactly like
    the local browser. The exceptions, all explicit:

    * the simulator opt-in (``SIM_MODE=1`` AND ``LAWN_BERRY_SIM_OPEN_AUTH=1``, never in
      production): nothing is enforced anywhere, so the sockets follow. Switching operator
      authentication off (``OPERATOR_AUTH_REQUIRED=0``; CRITICAL at start-up) is NOT the same
      thing and does NOT open the sockets: it only stops the HTTP routes asking for a bearer, a
      socket with no credential is refused exactly as before;
    * ``LAWN_BERRY_ANONYMOUS_TELEMETRY=loopback|any`` (``core/ws_policy.py``): the read-only
      streams, never ``/ws/control`` while auth is on, for loopback clients or everybody, and
      only for a page whose ``Origin`` is allowed (``_origin_allowed``).

    "Client" is :func:`core.peer_address.client_host`: the TCP peer, or the address a DECLARED
    forwarder recorded; never a header from an undeclared peer.

    No WebSocket handler changes mower state (``/ws/control`` only acknowledges), so an
    anonymous connection could not actuate anything even where one is admitted.

    A ``CF-Access-Jwt-Assertion`` header is deliberately NOT a credential: it was accepted
    without any verification, so any client could name itself an operator.
    """
    return (await _admit(websocket, command_capable=command_capable))[0]


@dataclass(eq=False)
class WsAdmission:
    """An admitted socket: who it is, whether it is anonymous, and its place in the budget."""

    session: UserSession
    slot: WsSlot
    anonymous: bool


async def _end_session_sockets(session_id: str, reason: str) -> None:
    """Termination listener: a login that ends takes its open WebSockets with it (SEC3-07)."""
    closed = await ws_registry.close_session(session_id)
    if closed:
        logger.info("WebSocket: %d socket(s) closed with their session (%s)", closed, reason)


def _handshake_key(websocket: WebSocket, peer: str) -> str:
    """Whose handshake budget this handshake spends: the live login session of the credential
    it presents, else its peer.

    The HTTP limiter's rule exactly (``middleware/rate_limiting.py::limiter_client_key``): junk
    from an address the operator shares (every visitor behind an undeclared forwarder is one
    address) can empty its own bucket and never the operator's, and a client cannot choose the
    key (a session id sits inside a signed token, a ticket is a 192-bit secret). The credential
    is only LOOKED AT here, silently: a ticket is not redeemed, nothing is logged and an expired
    session is not terminated (``_admit`` does all of that, after the budget).
    """
    token = _extract_bearer_token(websocket.headers.get("Authorization"))
    if token:
        session = primary_auth_service.peek_session(token)
        return f"session:{session.session_id}" if session is not None else peer
    ticket = _ws_ticket_of(websocket)
    session_id = _peek_ws_ticket_session_id(ticket) if ticket else None
    return f"session:{session_id}" if session_id else peer


def _log_refusal(peer: str, why: str) -> None:
    """One WARNING per peer per minute for the handshakes refused, saying how many it stands for
    (``ws_policy.RefusalLog``: a flood of junk handshakes must not be a flood of journal lines)."""
    stands_for = ws_registry.refusals.note(peer)
    if stands_for is not None:
        logger.warning(
            "WebSocket handshake refused for %s: %s (%d refusal(s) since the last line; at most "
            "one line per peer per %d s)",
            peer,
            why,
            stands_for,
            int(REFUSAL_LOG_INTERVAL_S),
        )


async def admit_websocket(websocket: WebSocket, *, command_capable: bool = False) -> WsAdmission:
    """Authorize the socket AND take a slot in the connection budget (``core/ws_policy.py``).

    In this order: (1) the handshake budget, BEFORE any credential is parsed or verified: 429 +
    ``Retry-After`` when the bucket of this login session (or, for a credential that is not a
    live login, this peer) is empty; (2) the policy (``_admit``): 401 for no or a bad
    credential; (3) the connection budget: 503 + ``Retry-After`` when too many sockets are open
    overall, for this login session or, for anonymous ones, for this client. Every refusal is
    logged at most once per peer per minute.

    A socket admitted with a stored login session is BOUND to it: the session's end (logout,
    expiry, credential change) closes the socket with 1008, and an expiry timer re-checks the
    session at its deadline (module docstring of ``core/ws_policy.py``). The handler MUST hand
    the slot back (``ws_registry.release(admission.slot)``) in its ``finally``.
    """
    peer = request_peer_key(websocket)
    retry_after = ws_registry.check_handshake(_handshake_key(websocket, peer))
    if retry_after is not None:
        _log_refusal(peer, "handshake rate limit")
        raise HTTPException(
            status_code=429,
            detail="Too many WebSocket handshakes",
            headers={"Retry-After": str(retry_after)},
        )
    try:
        session, anonymous = await _admit(websocket, command_capable=command_capable)
    except HTTPException as exc:
        _log_refusal(peer, f"{exc.status_code} {exc.detail}")
        raise
    # Bound = a stored login (the simulator opt-in's throw-away session is not in the store,
    # and has no login to end; an anonymous socket has none either).
    bound = (
        not anonymous and primary_auth_service.active_sessions.get(session.session_id) is session
    )
    try:
        slot = ws_registry.acquire(
            session_id=session.session_id,
            peer=peer,
            anonymous=anonymous,
            websocket=websocket if bound else None,
        )
    except WsLimitError as exc:
        _log_refusal(peer, str(exc))
        raise HTTPException(
            status_code=503, detail="Too many WebSocket connections", headers={"Retry-After": "5"}
        ) from exc
    if bound:
        # The service that admitted the socket is the one that must end it (whichever instance
        # that is: tests swap the module's service), so the listener is added here, not at import.
        primary_auth_service.add_termination_listener(_end_session_sockets)
        ws_registry.watch_expiry(
            slot, functools.partial(primary_auth_service.session_seconds_left, session.session_id)
        )
    return WsAdmission(session=session, slot=slot, anonymous=anonymous)


# Endpoints


@router.post("/auth/login", response_model=AuthResponse)
async def auth_login(payload: AuthLoginRequest, request: Request):
    credential = payload.credential
    if credential is None and payload.username is not None and payload.password is not None:
        # The login form posts username + password; the password IS the single
        # shared operator credential (the username is informational). There is no
        # hard-wired admin/admin alias: it only works if the operator credential
        # (LAWN_BERRY_OPERATOR_CREDENTIAL) is literally "admin".
        credential = payload.password

    try:
        result = await primary_auth_service.authenticate(
            credential or "",
            client_ip=_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.detail, headers=exc.headers
        ) from exc  # noqa: E501

    session = result.session
    user = UserOut(
        id=session.user_id,
        username=session.username,
        role=session.security_context.role.value,
        created_at=session.created_at,
    )
    expires_in = max(0, int((result.expires_at - datetime.now(UTC)).total_seconds()))

    return AuthResponse(
        access_token=result.token,
        token=result.token,
        expires_in=expires_in,
        expires_at=result.expires_at,
        user=user,
    )


@router.post("/auth/refresh", response_model=RefreshResponse)
async def auth_refresh(request: Request):
    session = await _require_session(request)
    result = primary_auth_service.refresh_session_token(session)
    expires_in = max(0, int((result.expires_at - datetime.now(UTC)).total_seconds()))
    return RefreshResponse(
        access_token=result.token,
        token=result.token,
        expires_in=expires_in,
        expires_at=result.expires_at,
    )


@router.post("/auth/logout")
async def auth_logout(request: Request):
    session = await _require_session(request)
    await primary_auth_service.terminate_session(session.session_id, "user_logout")
    return {"ok": True}


@router.get("/auth/profile", response_model=UserOut)
async def auth_profile(request: Request):
    session = await _require_session(request)
    return UserOut(
        id=session.user_id,
        username=session.username,
        role=session.security_context.role.value,
        created_at=session.created_at,
    )


@router.post(
    "/auth/ws-ticket",
    response_model=WsTicketResponse,
    dependencies=[Depends(require_operator_auth)],
)
async def auth_ws_ticket(request: Request):
    """A 30 s single-use ticket for one WebSocket connection (``?ticket=``).

    Verifies the bearer token itself, so it answers 401 without a login whatever
    ``OPERATOR_AUTH_REQUIRED`` says. Fetch a fresh ticket for every (re)connect.
    """
    session = await _require_session(request)
    return WsTicketResponse(
        ticket=_mint_ws_ticket(session.session_id), expires_in=WS_TICKET_TTL_SECONDS
    )


@router.get("/control/manual-unlock/status", response_model=ManualUnlockStatusResponse)
async def manual_unlock_status():
    """Cloudflare-gate probe of the Control view.

    With operator auth disabled it approves, as the unlock does. With auth enforced it never
    authorizes: the Cloudflare Access assertion it used to trust was decoded but not verified
    (any header value passed), and verifying it needs the Access issuer, audience and signing
    keys plus an RS256 implementation that this deployment does not carry.
    """
    if not operator_auth_required():
        session_id, expires_at = _auth_disabled_unlock()
        return ManualUnlockStatusResponse(
            authorized=True,
            session_id=session_id,
            expires_at=expires_at,
            principal="operator",
        )
    return ManualUnlockStatusResponse(authorized=False, reason="cloudflare_access_not_verified")


@router.post(
    "/control/manual-unlock",
    response_model=ManualUnlockResponse,
    dependencies=[Depends(require_operator_auth)],
)
async def manual_unlock(request: Request, body: ManualUnlockRequest):
    """Step-up unlock of manual control for an already logged-in operator.

    Needs the operator's bearer token (``require_operator_auth``) and, in the body, the
    credential again. Wrong credentials are throttled per TCP peer exactly like the login.
    """
    if not operator_auth_required():
        session_id, expires_at = _auth_disabled_unlock()
        return ManualUnlockResponse(
            authorized=True,
            session_id=session_id,
            expires_at=expires_at,
            principal="operator",
            source="auth_disabled",
        )

    method = (body.method or "").strip().lower()
    security_level = get_security_level()
    if not method:
        if security_level == SecurityLevel.TUNNEL_AUTH:
            method = "cloudflare"
        elif security_level == SecurityLevel.GOOGLE_OAUTH:
            method = "google"
        elif security_level == SecurityLevel.TOTP:
            method = "totp"
        else:
            method = "password"
    if method in {"cloudflare", "cloudflare_tunnel_auth", "tunnel"}:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=CLOUDFLARE_UNSUPPORTED
        )

    timeout_minutes = getattr(_security_settings, "session_timeout_minutes", 60)

    if method in {"password", "password_only"}:
        await _throttled_unlock(request, lambda: _validate_manual_password(body.password))
        principal = _manual_unlock_principal(request, "password")
        expires_at = _manual_session_expiry(timeout_minutes)
        session_entry = _issue_manual_session(expires_at, principal)
        logger.info("manual_control.unlock", extra={"method": "password", "principal": principal})
        return ManualUnlockResponse(
            authorized=True,
            session_id=session_entry["session_id"],
            expires_at=session_entry["expires_at"].isoformat(),
            principal=principal,
            source="password",
        )

    if method in {"totp", "password_totp"}:

        async def _check_totp_and_password() -> tuple[bool, bool]:
            totp_ok, backup_used = _verify_totp_code(body.totp_code)
            await _validate_manual_password(body.password)
            return totp_ok, backup_used

        totp_ok, backup_used = await _throttled_unlock(request, _check_totp_and_password)
        principal = _manual_unlock_principal(request, "totp")
        expires_at = _manual_session_expiry(timeout_minutes)
        session_entry = _issue_manual_session(expires_at, principal)
        logger.info(
            "manual_control.unlock",
            extra={
                "method": "totp",
                "principal": principal,
                "backup_used": backup_used,
                "verified": totp_ok,
            },
        )
        return ManualUnlockResponse(
            authorized=True,
            session_id=session_entry["session_id"],
            expires_at=session_entry["expires_at"].isoformat(),
            principal=principal,
            source="totp_backup" if backup_used else "totp",
        )

    if method in {"google", "google_auth"}:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google-based manual unlock is not implemented",
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unknown manual unlock method",
    )
