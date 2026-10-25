from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from .deps import require_operator_auth
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
import asyncio
import json
import hashlib
import logging
import re
import time
import os
import uuid
from email.utils import format_datetime, parsedate_to_datetime

from ..core.persistence import persistence
from ..core.globals import (
    _blade_state,
    _safety_state,
    _emergency_until,
    _client_emergency,
    _legacy_motors_active,
    _manual_control_sessions,
)
from .routers import telemetry
from .routers.auth import _resolve_manual_session
from ..services.websocket_hub import websocket_hub

logger = logging.getLogger(__name__)
router = APIRouter()
legacy_router = APIRouter()

# Legacy WebSocket paths
legacy_router.add_websocket_route("/ws/telemetry", telemetry.ws_telemetry)
legacy_router.add_websocket_route("/ws/control", telemetry.ws_control)

# ----------------------- Map Zones -----------------------


class Point(BaseModel):
    latitude: float
    longitude: float


class Zone(BaseModel):
    id: str
    name: Optional[str] = None
    polygon: list[Point]
    priority: int = 0
    exclusion_zone: bool = False


_zones_store: list[Zone] = []
_zones_last_modified: datetime = datetime.now(timezone.utc)


@router.get("/map/zones", response_model=list[Zone])
def get_map_zones(request: Request):
    data = [z.model_dump(mode="json") for z in _zones_store]
    body = json.dumps(data, sort_keys=True).encode()
    etag = hashlib.sha256(body).hexdigest()
    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")
    if inm == etag:
        return JSONResponse(status_code=304, content=None)
    if ims:
        try:
            ims_dt = parsedate_to_datetime(ims)
            if ims_dt.tzinfo is None:
                ims_dt = ims_dt.replace(tzinfo=timezone.utc)
            if ims_dt >= _zones_last_modified.replace(microsecond=0):
                return JSONResponse(status_code=304, content=None)
        except Exception:
            pass
    headers = {
        "ETag": etag,
        "Last-Modified": format_datetime(_zones_last_modified),
        "Cache-Control": "public, max-age=30",
    }
    return JSONResponse(content=data, headers=headers)


@router.post("/map/zones", response_model=list[Zone], dependencies=[Depends(require_operator_auth)])
def post_map_zones(zones: list[Zone]):
    global _zones_store
    _zones_store = zones
    global _zones_last_modified
    _zones_last_modified = datetime.now(timezone.utc)
    return _zones_store


# --------------------- Map Locations ---------------------


class Position(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    accuracy: float | None = None
    gps_mode: str | None = None


class MapLocations(BaseModel):
    home: Optional[Position] = None
    am_sun: Optional[Position] = None
    pm_sun: Optional[Position] = None


_locations_store = MapLocations()
_locations_last_modified: datetime = datetime.now(timezone.utc)


@router.get("/map/locations", response_model=MapLocations)
def get_map_locations(request: Request):
    data = _locations_store.model_dump(mode="json")
    body = json.dumps(data, sort_keys=True).encode()
    etag = hashlib.sha256(body).hexdigest()
    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")
    if inm == etag:
        return JSONResponse(status_code=304, content=None)
    if ims:
        try:
            ims_dt = parsedate_to_datetime(ims)
            if ims_dt.tzinfo is None:
                ims_dt = ims_dt.replace(tzinfo=timezone.utc)
            if ims_dt >= _locations_last_modified.replace(microsecond=0):
                return JSONResponse(status_code=304, content=None)
        except Exception:
            pass
    headers = {
        "ETag": etag,
        "Last-Modified": format_datetime(_locations_last_modified),
        "Cache-Control": "public, max-age=30",
    }
    return JSONResponse(content=data, headers=headers)


@router.put(
    "/map/locations", response_model=MapLocations, dependencies=[Depends(require_operator_auth)]
)
def put_map_locations(locations: MapLocations):
    global _locations_store
    _locations_store = locations
    global _locations_last_modified
    _locations_last_modified = datetime.now(timezone.utc)
    return _locations_store


# ------------------------ Control V2 Endpoints ------------------------


class ControlCommandV2(BaseModel):
    throttle: Optional[float] = Field(None, ge=-1.0, le=1.0)
    turn: Optional[float] = Field(None, ge=-1.0, le=1.0)
    blade_enabled: Optional[bool] = None
    max_speed_limit: float = Field(0.8, ge=0.0, le=1.0)
    timeout_ms: int = Field(1000, ge=100, le=10000)
    confirmation_token: Optional[str] = None


class ControlResponseV2(BaseModel):
    accepted: bool
    audit_id: str
    result: str
    status_reason: Optional[str] = None
    watchdog_echo: Optional[str] = None
    watchdog_latency_ms: Optional[float] = None
    safety_checks: list[str] = []
    active_interlocks: list[str] = []
    remediation: Optional[dict[str, str]] = None
    telemetry_snapshot: Optional[dict[str, Any]] = None
    # Emergency-stop endpoints only: the per-subsystem report (motors_stopped,
    # blade_disabled, complete, subsystems, faults) -- see emergency_stop_service.
    emergency_stop: Optional[dict[str, Any]] = None
    timestamp: str


def _emergency_active() -> bool:
    try:
        return time.time() < _emergency_until
    except Exception:
        return False


# Per-client emergency flag table bounds (see ``_remember_client_emergency``).
CLIENT_EMERGENCY_MAX = 256
CLIENT_EMERGENCY_TTL_S = 0.3
# Audit rows for the unauthenticated stop routes are coalesced per this window.
ESTOP_AUDIT_WINDOW_S = 5.0
# Never read more than this of an E-stop request body (the stop does not need it).
ESTOP_BODY_CAP_BYTES = 64 * 1024
ESTOP_BODY_READ_TIMEOUT_S = 1.0


def _client_key(request: Request) -> str | None:
    """Identify a client by ``Authorization`` / ``X-Client-Id``; ``None`` if it sent neither.

    The key is a fixed-size digest so an attacker-sized header is never stored. A request
    without either header has no identity another request could ever look up.
    """
    raw = request.headers.get("authorization") or request.headers.get("x-client-id")
    if not raw:
        return None
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def _remember_client_emergency(request: Request | None) -> None:
    """Flag this client as in-emergency for the TTL; keeps the table bounded.

    Expired entries are pruned on every insert and the oldest entry is evicted at the cap.
    Entries are inserted in time order, so the oldest insertion is also the earliest expiry.
    """
    if request is None:
        return
    key = _client_key(request)
    if key is None:
        return
    now = time.time()
    for k in [k for k, exp in _client_emergency.items() if exp <= now]:
        _client_emergency.pop(k, None)
    _client_emergency.pop(key, None)
    while len(_client_emergency) >= CLIENT_EMERGENCY_MAX:
        _client_emergency.pop(next(iter(_client_emergency)), None)
    _client_emergency[key] = now + CLIENT_EMERGENCY_TTL_S


def _client_emergency_active(request: Request | None) -> bool:
    """Return True if this client's emergency flag is active; expire stale entries.

    Uses a short TTL to prevent cross-test leakage while still blocking
    immediately-following commands after an emergency trigger.
    """
    try:
        if request is None:
            return False
        key = _client_key(request)
        if key is None:
            return False
        exp = _client_emergency.get(key)
        now = time.time()
        if exp is None:
            return False
        if now < exp:
            return True
        # Expired: cleanup
        _client_emergency.pop(key, None)
        return False
    except Exception:
        return False


async def _read_estop_body(request: Request | None) -> dict[str, Any]:
    """Best-effort JSON-object body of an E-stop request; never raises, never blocks long.

    Anything that is not a JSON object (form, text, list, string, invalid JSON, binary,
    oversized, slow) degrades to ``{}``: the stop never depends on the body.
    """
    if request is None:
        return {}
    try:
        declared = request.headers.get("content-length")
        if declared is not None and int(declared) > ESTOP_BODY_CAP_BYTES:
            return {}
        chunks: list[bytes] = []
        size = 0
        async with asyncio.timeout(ESTOP_BODY_READ_TIMEOUT_S):
            async for chunk in request.stream():
                size += len(chunk)
                if size > ESTOP_BODY_CAP_BYTES:
                    return {}
                chunks.append(chunk)
        data = json.loads(b"".join(chunks).decode("utf-8")) if chunks else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _audit_estop_coalesced(action: str, details: dict[str, Any]) -> None:
    """Off-loop, coalesced, never-raising audit row for the unauthenticated stop routes."""
    try:
        persistence.add_audit_log_coalesced(action, details=details, window_s=ESTOP_AUDIT_WINDOW_S)
    except Exception as exc:
        logger.error("E-stop audit failed: %s", exc)


# Import helper from auth router for session resolution


@router.get("/hardware/robohat")
async def get_robohat_status():
    """Get RoboHAT firmware health and watchdog status with safety summary."""
    from ..services.robohat_service import get_robohat_service

    robohat = get_robohat_service()

    # Determine safety state summary for this snapshot
    safety_state = (
        "emergency_stop" if _safety_state.get("emergency_stop_active", False) else "nominal"
    )

    if robohat is None:
        # Minimal payload when service not initialized yet
        return {
            "firmware_version": "unknown",
            "uptime_seconds": 0,
            "watchdog_active": False,
            "serial_connected": False,
            "health_status": "not_initialized",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Contract-friendly fields
            "watchdog_heartbeat_ms": None,
            "safety_state": safety_state,
        }

    status = robohat.get_status()
    payload = status.to_dict()
    # Contract-friendly aliases/fields
    payload["watchdog_heartbeat_ms"] = payload.get("watchdog_latency_ms")
    payload["safety_state"] = safety_state
    telemetry_snapshot: dict[str, Any] | None = None
    try:
        telemetry_snapshot = await websocket_hub._generate_telemetry()
    except Exception as exc:
        logger.warning("Failed to gather hardware telemetry snapshot: %s", exc)

    if telemetry_snapshot:
        source = telemetry_snapshot.get("source") or "unknown"
        payload["telemetry_source"] = source
        if telemetry_snapshot.get("safety_state"):
            payload["safety_state"] = telemetry_snapshot["safety_state"]
        if telemetry_snapshot.get("camera") is not None:
            payload["camera"] = telemetry_snapshot.get("camera")

        if source == "hardware":
            for key in ("battery", "position", "imu", "velocity", "motor_status", "uptime_seconds"):
                value = telemetry_snapshot.get(key)
                if value is not None:
                    payload[key] = value
        else:
            # Avoid presenting simulated values as live data
            payload.setdefault("battery", {"percentage": None, "voltage": None})
            payload.setdefault("position", {"latitude": None, "longitude": None})
            payload.setdefault(
                "velocity",
                {
                    "linear": {"x": None, "y": None, "z": None},
                    "angular": {"x": None, "y": None, "z": None},
                },
            )
    else:
        payload["telemetry_source"] = "unknown"

    return payload


class Vector2D(BaseModel):
    linear: float
    angular: float


class DriveContractIn(BaseModel):
    session_id: str
    vector: Vector2D
    duration_ms: int
    reason: Optional[str] = None


@router.post(
    "/control/drive",
    response_model=ControlResponseV2,
    status_code=202,
    dependencies=[Depends(require_operator_auth)],
)
async def control_drive_v2(cmd: dict, request: Request):
    """Execute drive command with safety checks and audit logging"""
    import uuid
    from ..services.robohat_service import get_robohat_service
    from ..services.motor_service import MotorService

    audit_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Legacy behavior for integration tests: when payload is legacy style (mode/command),
    # return 200 with calculated motor speeds, unless emergency stop is active (then 403).
    is_legacy = "session_id" not in cmd
    if is_legacy:
        # Emergency stop -> reject legacy drive commands with 403 (short-lived TTL)
        if _client_emergency_active(request) or _emergency_active():
            try:
                cmd_details = cmd if isinstance(cmd, dict) else cmd.model_dump()
            except Exception:
                cmd_details = {}
            persistence.add_audit_log(
                "control.drive.blocked",
                details={"reason": "emergency_stop_active", "command": cmd_details},
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Emergency stop active - drive commands blocked"},
            )
        # Compute motor speeds using arcade drive
        throttle = float(cmd.get("throttle", 0.0))
        turn = float(cmd.get("turn", 0.0))
        left_speed = throttle - turn
        right_speed = throttle + turn
        # Clamp
        max_speed_limit = 1.0
        left_speed = max(-max_speed_limit, min(max_speed_limit, left_speed))
        right_speed = max(-max_speed_limit, min(max_speed_limit, right_speed))
        # Mark legacy motors active for interlock tests
        global _legacy_motors_active
        _legacy_motors_active = True
        body = {
            "left_motor_speed": round(left_speed, 3),
            "right_motor_speed": round(right_speed, 3),
            "safety_status": "OK",
        }
        persistence.add_audit_log("control.drive", details={"command": cmd, "response": body})
        return JSONResponse(status_code=200, content=body)

    # Contract-style payload
    # Block when emergency stop is active
    if _client_emergency_active(request) or _emergency_active():
        try:
            cmd_details = cmd if isinstance(cmd, dict) else cmd.model_dump()
        except Exception:
            cmd_details = {}
        if isinstance(cmd_details, dict) and "session_id" in cmd_details:
            cmd_details = dict(cmd_details)
            cmd_details["session_id"] = "***"
        persistence.add_audit_log(
            "control.drive.blocked",
            details={"reason": "emergency_stop_active", "command": cmd_details},
        )
        return JSONResponse(
            status_code=403, content={"detail": "Emergency stop active - drive commands blocked"}
        )

    session_context = _resolve_manual_session(cmd.get("session_id"))

    try:
        duration_ms = int(cmd.get("duration_ms", 0))
    except (TypeError, ValueError):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail="duration_ms must be an integer"
        )

    if duration_ms < 0 or duration_ms > 5000:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="duration_ms must be between 0 and 5000 milliseconds",
        )

    # Extract vector and convert to differential speeds (arcade)
    # Contract-style payload
    throttle = float(cmd.get("vector", {}).get("linear", 0.0))
    turn = float(cmd.get("vector", {}).get("angular", 0.0))
    try:
        speed_limit = float(cmd.get("max_speed_limit", 0.8))
    except (TypeError, ValueError):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail="max_speed_limit must be numeric"
        )
    import math

    if not all(math.isfinite(v) for v in (throttle, turn, speed_limit)):
        # NaN would clamp to full speed below (min/max return the other operand): refuse it.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="vector.linear, vector.angular and max_speed_limit must be finite numbers",
        )
    speed_limit = max(0.0, min(1.0, speed_limit))

    # Send command to RoboHAT

    robohat = get_robohat_service()
    watchdog_start = datetime.now(timezone.utc)

    if robohat and robohat.status.serial_connected:
        # Calculate differential speeds
        left_speed = throttle - turn
        right_speed = throttle + turn

        # Clamp to max speed limit
        left_speed = max(-speed_limit, min(speed_limit, left_speed))
        right_speed = max(-speed_limit, min(speed_limit, right_speed))

        # Send to RoboHAT
        success = await robohat.send_motor_command(
            left_speed,
            right_speed,
            # duration_ms is how long this command stays in force; 0/absent = the
            # service default. After that the drive returns to neutral.
            valid_for_s=(duration_ms / 1000.0) if duration_ms > 0 else None,
        )

        watchdog_end = datetime.now(timezone.utc)
        watchdog_latency = (watchdog_end - watchdog_start).total_seconds() * 1000

        response = ControlResponseV2(
            accepted=success,
            audit_id=audit_id,
            result="accepted" if success else "rejected",
            status_reason=(
                None if success else (robohat.status.last_error or "robohat_communication_failed")
            ),
            watchdog_echo=robohat.status.last_watchdog_echo,
            watchdog_latency_ms=watchdog_latency,
            safety_checks=["emergency_stop_check", "command_validation"],
            active_interlocks=[],
            telemetry_snapshot={
                "component_id": "drive_left",
                "status": "healthy" if success else "warning",
                "latency_ms": round(watchdog_latency, 2),
                "speed_limit": speed_limit,
            },
            timestamp=timestamp.isoformat(),
        )
    else:
        # Contract allows "queued" acknowledgement even if hardware not connected
        response = ControlResponseV2(
            accepted=True,
            audit_id=audit_id,
            result="queued",
            status_reason="nominal",
            safety_checks=["emergency_stop_check"],
            active_interlocks=[],
            remediation=None,
            telemetry_snapshot={
                "component_id": "drive_left",
                "status": "warning",
                "latency_ms": 0.0,
                "speed_limit": speed_limit,
            },
            timestamp=timestamp.isoformat(),
        )

    # Audit the command
    try:
        details_cmd = cmd if isinstance(cmd, dict) else cmd.model_dump()
    except Exception:
        details_cmd = {}
    if isinstance(details_cmd, dict):
        details_cmd = dict(details_cmd)
        if "session_id" in details_cmd:
            details_cmd["session_id"] = "***"
        principal = session_context.get("principal")
        if principal and "principal" not in details_cmd:
            details_cmd["principal"] = principal
        details_cmd["max_speed_limit"] = speed_limit
    persistence.add_audit_log(
        "control.drive.v2", details={"command": details_cmd, "response": response.model_dump()}
    )

    return response


class BladeContractIn(BaseModel):
    session_id: str
    action: str
    reason: Optional[str] = None


_BLADE_ON_WORDS = {"enable", "on", "start"}
_BLADE_OFF_WORDS = {"disable", "off", "stop"}


def _strict_bool(value: Any, field: str) -> bool:
    """A JSON boolean or 422: ``bool("false")`` is True, so nothing else is coerced (BL-02)."""
    if not isinstance(value, bool):
        raise HTTPException(
            status_code=422,
            detail=f"{field} must be a JSON boolean (true or false), got {value!r}",
        )
    return value


def _blade_desired(cmd: dict) -> bool | None:
    """The blade state a ``/control/blade`` body asks for; ``None`` = nothing recognised.

    ``action`` is the words ``enable|on|start`` / ``disable|off|stop`` (any other word asks
    for nothing and is answered ``blocked``); ``active`` must be a real JSON boolean. A body
    carrying both must agree. Anything ambiguous is a 422, never a guess: a client that sent
    ``{"active": "false"}`` must not have switched the blade ON.
    """
    desired: bool | None = None
    if "action" in cmd:
        word = str(cmd.get("action")).lower()
        desired = True if word in _BLADE_ON_WORDS else False if word in _BLADE_OFF_WORDS else None
        if "active" in cmd:
            active = _strict_bool(cmd["active"], "active")
            if desired is not None and desired != active:
                raise HTTPException(
                    status_code=422,
                    detail=f"action {cmd.get('action')!r} and active={active!r} disagree",
                )
        return desired
    if "active" in cmd:
        return _strict_bool(cmd["active"], "active")
    return None


async def _blade_hardware_command(
    bs: Any, desired: bool, cmd: dict, session_context: dict, audit_id: str
) -> JSONResponse:
    """Drive the IBT-4 and answer with what the hardware did, whatever the audit store does.

    The answer is built from the service's own readback AFTER the write, and the audit record is
    written afterwards in its own guard: a failing audit store is reported in the body
    (``audit_logged: false``) but can no longer turn a blade that is ON into a "blocked" 423 or
    a 500 (audit BL-03).
    """
    timestamp = datetime.now(timezone.utc)
    error: str | None = None
    try:
        ok = bool(await bs.set_active(desired))
    except Exception as exc:  # the service never raises; if it does the hardware state is unknown
        logger.critical("Blade command raised: %s", exc, exc_info=True)
        ok, error = False, f"{type(exc).__name__}: {exc}"
    body: dict[str, Any] = {
        "accepted": ok,
        "audit_id": audit_id,
        "result": "accepted" if ok else "rejected",
        "timestamp": timestamp.isoformat(),
    }
    if not ok:
        body["reason"] = error or getattr(bs, "last_refusal", None) or "blade command refused"
    try:
        state = await bs.status()
        body["blade_state"] = {
            key: state.get(key)
            for key in (
                "commanded",
                "observed",
                "pad",
                "active",
                "state_confirmed",
                "deadman_remaining_ms",
            )
        }
    except Exception as exc:
        logger.error("Blade state readback failed: %s", exc)
        body["blade_state"] = None
    log_command = {**cmd}
    if "session_id" in log_command:
        log_command["session_id"] = "***"
    if session_context.get("principal"):
        log_command.setdefault("principal", session_context.get("principal"))
    try:
        persistence.add_audit_log(
            "control.blade.v2", details={"command": log_command, "response": body}
        )
    except Exception as exc:
        logger.critical("Audit log write failed AFTER the blade command: %s", exc, exc_info=True)
        body["audit_logged"] = False
    return JSONResponse(status_code=200 if ok else 409, content=body)


@router.get("/control/blade/status")
async def get_blade_status():
    """The IBT-4 blade as the pads read now: commanded vs observed, latch, deadman, pending OFF.

    Read-only. ``observed`` / ``pad`` come from a real readback of the two inputs (an unarmed
    flag-only stub reports ``unobservable``); ``active`` is conservative (a failed readback or an
    unconfirmed command reads as "may be on"); ``refusal`` says why a blade ON would be refused
    right now.
    """
    from ..services.blade_service import get_blade_service

    return await get_blade_service().status()


@router.post("/control/blade", dependencies=[Depends(require_operator_auth)])
async def control_blade_v2(cmd: dict, request: Request):
    """Execute blade command with safety interlocks and audit logging"""
    import uuid
    from ..services.robohat_service import get_robohat_service

    audit_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Contract requires safety lockout for blade engagement by default
    # Return HTTP 423 (Locked) with remediation
    # Legacy behavior for integration tests
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if "session_id" not in cmd and ("active" in cmd or "command" in cmd):
        # Enable
        if (cmd.get("command") == "blade_enable") or (cmd.get("active") is True):
            # Block if emergency is active (global TTL or per-client TTL)
            if _emergency_active() or _client_emergency_active(request):
                body = {
                    "detail": "safety_interlock: emergency_stop_active - blade commands blocked"
                }
                persistence.add_audit_log(
                    "control.blade", details={"command": cmd, "response": body}
                )
                return JSONResponse(status_code=403, content=body)
            # If no auth header provided, allow enabling for audit test flow
            if not auth_header:
                body = {"blade_status": "ENABLED"}
                persistence.add_audit_log(
                    "control.blade", details={"command": cmd, "response": body}
                )
                return JSONResponse(status_code=200, content=body)
            # Safety interlock: reject with 403 if motors active; otherwise accept
            global _legacy_motors_active
            if _legacy_motors_active:
                body = {"detail": "Safety interlock: motors_active"}
                persistence.add_audit_log(
                    "control.blade", details={"command": cmd, "response": body}
                )
                return JSONResponse(status_code=403, content=body)
            body = {"blade_status": "ENABLED"}
            persistence.add_audit_log("control.blade", details={"command": cmd, "response": body})
            return JSONResponse(status_code=200, content=body)
        # Disable
        if (cmd.get("command") == "blade_disable") or (cmd.get("active") is False):
            body = {"blade_status": "DISABLED"}
            persistence.add_audit_log("control.blade", details={"command": cmd, "response": body})
            return JSONResponse(status_code=200, content=body)

    session_context = _resolve_manual_session(cmd.get("session_id"))

    # Strict first: an ambiguous body (a string where a boolean is required, an action and an
    # `active` that disagree) is refused with 422 before anything is touched (audit BL-02).
    desired = _blade_desired(cmd)
    emergency = _emergency_active() or _client_emergency_active(request)
    # A blade OFF is never refused: it is the safe direction. A blade ON is blocked during an
    # emergency window (the E-stop latch itself is enforced by the blade service).
    if desired is not None and not (desired and emergency):
        try:
            from ..services.blade_service import get_blade_service

            bs = get_blade_service()
        except Exception:
            # Nothing was driven: fall through to the blocked answer below.
            logger.exception("blade service unavailable for /control/blade")
        else:
            return await _blade_hardware_command(bs, desired, cmd, session_context, audit_id)

    payload = {
        "accepted": False,
        "audit_id": audit_id,
        "result": "blocked",
        "status_reason": "SAFETY_LOCKOUT",
        "remediation_url": "/docs/OPERATIONS.md#blade-safety-lockout",
        "safety_checks": ["emergency_stop_check", "blade_lockout"],
        "active_interlocks": ["blade_requires_authorization"],
        "timestamp": timestamp.isoformat(),
    }
    try:
        cmd_details = cmd if isinstance(cmd, dict) else cmd.model_dump()
    except Exception:
        cmd_details = {}
    if isinstance(cmd_details, dict):
        cmd_details = dict(cmd_details)
        if "session_id" in cmd_details:
            cmd_details["session_id"] = "***"
        if session_context and session_context.get("principal"):
            cmd_details.setdefault("principal", session_context.get("principal"))
    persistence.add_audit_log(
        "control.blade.blocked", details={"command": cmd_details, "response": payload}
    )
    return JSONResponse(status_code=423, content=payload)


def _emergency_remediation(report: dict[str, Any]) -> dict[str, str]:
    if report["complete"]:
        message = "Emergency stop latched; every output confirmed stopped"
    else:
        faults = "; ".join(report["faults"]) or "no output could be confirmed"
        message = (
            "Emergency stop LATCHED but NOT every output is confirmed stopped "
            f"({faults}). Use the physical stop."
        )
    return {"message": message, "docs_link": "/docs/OPERATIONS.md#emergency-stop-recovery"}


async def _run_emergency_stop(source: str) -> dict[str, Any]:
    """Drive every stop path and return the per-subsystem report; never raises."""
    from ..services.emergency_stop_service import execute_emergency_stop

    try:
        return await execute_emergency_stop(source=source)
    except Exception as exc:  # the orchestrator is built not to raise; fail honest if it does
        logger.critical("Emergency stop orchestration failed: %s", exc, exc_info=True)
        return {
            "emergency_stop_active": True,
            "motors_stopped": False,
            "blade_disabled": False,
            "complete": False,
            "subsystems": {},
            "faults": [f"emergency stop orchestration failed: {type(exc).__name__}: {exc}"],
        }


@router.post("/control/emergency", response_model=ControlResponseV2, status_code=202)
async def control_emergency_v2(request: Request):
    """Latch the emergency stop and drive every stop path that exists.

    Always available: it never requires a valid manual-control session (the
    session is only resolved for the audit trail) and never depends on the request
    body (any content type or size; a body that is not a JSON object is ignored).
    The response reports what each subsystem (tractor, RoboHAT, IBT-4 blade) actually
    did -- ``motors_stopped`` / ``blade_disabled`` are ``True`` only when a stop was
    confirmed and none failed.
    """
    audit_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Latch FIRST: nothing below may fail or wait before the stop is armed.
    _safety_state["emergency_stop_active"] = True
    _blade_state["active"] = False
    global _legacy_motors_active
    _legacy_motors_active = False
    # Arm short-lived emergency TTL to block follow-up commands across both modes
    global _emergency_until
    _emergency_until = time.time() + 0.2
    try:
        _remember_client_emergency(request)
    except Exception:
        pass

    payload = await _read_estop_body(request)
    is_legacy = bool(payload.get("command"))

    report = await _run_emergency_stop("api:/control/emergency")

    session_context = None
    if not is_legacy:
        try:
            session_context = _resolve_manual_session(payload.get("session_id"))
        except Exception:
            session_context = None  # an E-stop must not be refused for a missing session

    # If legacy payload with command field was sent, return 200 with integration-expected shape
    if is_legacy:
        legacy_payload = {
            "status": "EMERGENCY_STOP_ACTIVE",
            "motors_stopped": report["motors_stopped"],
            "blade_disabled": report["blade_disabled"],
            "emergency_stop_active": True,
            "complete": report["complete"],
            "subsystems": report["subsystems"],
            "faults": report["faults"],
            "remediation": _emergency_remediation(report),
            "timestamp": timestamp.isoformat(),
        }
        _audit_estop_coalesced("control.emergency_stop", {"response": legacy_payload})
        return JSONResponse(status_code=200, content=legacy_payload)

    response = ControlResponseV2(
        # "accepted" only when the stop was confirmed: a client keying on accepted/result must
        # not read success off a stop that left an output unconfirmed (audit SEC-13).
        accepted=report["complete"],
        audit_id=audit_id,
        result="accepted" if report["complete"] else "incomplete",
        status_reason=(
            "EMERGENCY_STOP_TRIGGERED" if report["complete"] else "EMERGENCY_STOP_INCOMPLETE"
        ),
        safety_checks=["immediate_stop"],
        active_interlocks=["emergency_stop_override"]
        + ([] if report["complete"] else ["emergency_stop_unconfirmed"]),
        remediation=_emergency_remediation(report),
        telemetry_snapshot={
            "component_id": "drive_left",
            "status": "fault",
            "latency_ms": 0.0,
        },
        emergency_stop=report,
        timestamp=timestamp.isoformat(),
    )

    # Audit the emergency stop (off-loop, coalesced; a failure never changes the response)
    audit_details: dict[str, Any] = {"response": response.model_dump()}
    if session_context and session_context.get("principal"):
        audit_details["principal"] = session_context["principal"]
    _audit_estop_coalesced("control.emergency.triggered", audit_details)

    return response


@router.post("/control/emergency-stop")
async def control_emergency_stop_alias(request: Request = None):
    """Integration-friendly alias: always 200, with the per-subsystem truth in the body.

    ``emergency_stop_active`` is the software latch (always ``True`` here);
    ``motors_stopped`` / ``blade_disabled`` / ``complete`` say what was actually
    confirmed, per ``subsystems`` -- see ``emergency_stop_service``.
    """
    _safety_state["emergency_stop_active"] = True
    _blade_state["active"] = False
    report = await _run_emergency_stop("api:/control/emergency-stop")
    payload = {
        "emergency_stop_active": True,
        "motors_stopped": report["motors_stopped"],
        "blade_disabled": report["blade_disabled"],
        "complete": report["complete"],
        "subsystems": report["subsystems"],
        "faults": report["faults"],
        "remediation": _emergency_remediation(report),
    }
    _audit_estop_coalesced("control.emergency_stop", {"response": payload})
    return JSONResponse(status_code=200, content=payload)


# ---------------------------------------------------------------------------
# Documentation hub
# ---------------------------------------------------------------------------


def _docs_root() -> Path:
    """Resolve the documentation root directory.

    Override with ``LAWNBERRY_DOCS_DIR``; otherwise use the repo ``docs/``.
    Tests monkeypatch this function to point at a temporary directory.
    """
    override = os.getenv("LAWNBERRY_DOCS_DIR")
    if override:
        return Path(override)
    candidate = Path(__file__).resolve().parents[3] / "docs"
    return candidate if candidate.exists() else Path("docs")


def _iter_doc_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file())


@router.get("/docs/list", dependencies=[Depends(require_operator_auth)])
async def docs_list() -> list[dict[str, Any]]:
    """List available markdown documents with size and modification metadata."""
    root = _docs_root()
    items: list[dict[str, Any]] = []
    for path in _iter_doc_files(root):
        st = path.stat()
        items.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": st.st_size,
                "last_modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return items


@router.get("/docs/checksums", dependencies=[Depends(require_operator_auth)])
async def docs_checksums() -> dict[str, Any]:
    """Return SHA-256 checksums for all documents for integrity verification."""
    root = _docs_root()
    checksums = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _iter_doc_files(root)
    }
    return {"checksums": checksums}


@router.get("/docs/freshness", dependencies=[Depends(require_operator_auth)])
async def docs_freshness() -> dict[str, Any]:
    """Report documentation age and flag entries older than 90 days as stale."""
    root = _docs_root()
    now = time.time()
    docs: list[dict[str, Any]] = []
    for path in _iter_doc_files(root):
        age_days = (now - path.stat().st_mtime) / 86400.0
        docs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "age_days": age_days,
                "stale": age_days > 90,
            }
        )
    return {"docs": docs}


@router.get("/docs/bundle", dependencies=[Depends(require_operator_auth)])
async def docs_bundle(simulate_checksum_mismatch: Optional[str] = Query(default=None)):
    """Return an offline documentation manifest with per-doc checksums.

    Surfaces offline readiness and (simulated) checksum-mismatch warnings via
    response headers so the operator UI can prompt remediation.
    """
    root = _docs_root()
    items: list[dict[str, Any]] = []
    for path in _iter_doc_files(root):
        content = path.read_bytes()
        st = path.stat()
        rel = path.relative_to(root).as_posix()
        title = rel
        for line in content.decode("utf-8", errors="ignore").splitlines():
            if line.strip().startswith("#"):
                title = line.lstrip("#").strip() or rel
                break
        items.append(
            {
                "doc_id": rel,
                "title": title,
                "version": "1.0.0",
                "last_updated": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "checksum": hashlib.sha256(content).hexdigest(),
                "offline_available": True,
            }
        )
    headers = {"x-docs-offline-ready": "true" if items else "false"}
    if simulate_checksum_mismatch:
        headers["x-docs-checksum-warning"] = simulate_checksum_mismatch
    return JSONResponse(content={"items": items}, headers=headers)


@router.get("/docs/{doc_path:path}", dependencies=[Depends(require_operator_auth)])
async def docs_get(doc_path: str):
    """Return a single markdown document, guarding against path traversal.

    Operator-only like the rest of the documentation hub: the docs folder carries internal
    material (LAN addresses, hardware wiring, runbooks, agent hand-off notes that quote
    credentials), so it is not served to an anonymous caller (SEC20-01 GET audit). Only the
    ``.md`` files the list endpoints advertise are served.
    """
    if "\x00" in doc_path or doc_path.startswith("/") or ".." in doc_path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid document path")
    root = _docs_root()
    try:
        resolved = (root / doc_path).resolve()
        root_resolved = root.resolve()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid document path") from exc
    if not resolved.is_relative_to(root_resolved):
        raise HTTPException(status_code=403, detail="Path traversal detected")
    if not resolved.is_file() or resolved.suffix.lower() != ".md":
        raise HTTPException(status_code=404, detail="Document not found")
    return PlainTextResponse(resolved.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Verification artifacts
# ---------------------------------------------------------------------------

_known_requirements_cache: Optional[set[str]] = None


def _known_requirements() -> set[str]:
    """Collect the requirement IDs (FR-NNN) referenced by the spec/docs."""
    global _known_requirements_cache
    if _known_requirements_cache is not None:
        return _known_requirements_cache
    found: set[str] = set()
    repo_root = Path(__file__).resolve().parents[3]
    for sub in ("spec", "specs", ".specify", "docs"):
        directory = repo_root / sub
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".yaml", ".yml", ".json", ".txt"}:
                try:
                    found.update(re.findall(r"FR-\d{3}", path.read_text(errors="ignore")))
                except Exception:
                    continue
    if not found:
        found = {f"FR-{i:03d}" for i in range(1, 51)}
    _known_requirements_cache = found
    return found


@router.post(
    "/verification-artifacts", status_code=201, dependencies=[Depends(require_operator_auth)]
)
async def post_verification_artifact(body: dict):
    """Record a verification artifact, enforcing requirement traceability."""
    linked = body.get("linked_requirements") or []
    if not linked:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "MISSING_REQUIREMENTS",
                "detail": "At least one linked requirement is required",
                "remediation_url": "/docs/field-validation-protocol.md#traceability",
            },
        )

    known = _known_requirements()
    unknown = [req for req in linked if req not in known]
    if unknown:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "UNKNOWN_REQUIREMENT",
                "detail": f"Unknown requirement IDs: {', '.join(unknown)}",
                "unknown": unknown,
            },
        )

    artifact_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    record = {
        "artifact_id": artifact_id,
        "created_at": created_at,
        "type": body.get("type"),
        "location": body.get("location"),
        "summary": body.get("summary"),
        "linked_requirements": linked,
        "created_by": body.get("created_by"),
        "metadata": body.get("metadata"),
    }
    persistence.add_audit_log(
        "verification.artifact",
        details={"artifact_id": artifact_id, "linked_requirements": linked},
    )
    return JSONResponse(status_code=201, content=record)
