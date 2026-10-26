"""Zero-turn mower control endpoints (v2).

Operate the mower's actuators (left/right drive lever, throttle, blade PTO,
starter) with the standard safety interlocks enforced by
``TractorControlService``. Writes require operator auth; the emergency stop
and state read are always available.

Lever and throttle commands are held only for a bounded time (the optional
``duration_ms``, clamped by the service's configured maximum, else its default
hold) and then return to neutral/idle unless refreshed -- a dropped client
stops the machine. The effective ``hold_ms`` is echoed in the response.
``POST /tractor/clear-emergency`` is the explicit authenticated clear for the
E-stop latch and, when it is safe, the blade interlock latch (tilt): a clear on a
tilted machine, or one whose tilt sensor is unavailable, releases the E-stop but keeps the
blade latch and answers ``status: "partial"`` with ``still_latched``. Every clear is audited
with its optional ``reason``. A clear never authorizes: after a stop (which revokes the
authorization) nothing autonomous moves the machine until ``POST /tractor/authorize`` is called
again and a new run is started. The lever and throttle routes here are the ``operator`` source
and keep their documented behaviour (no ``authorize`` needed for an authenticated person's
own drive; the blade and starter still need it).

A platform that is not enabled in ``config/tractor.yaml`` is never driven through these
routes outside SIM_MODE: writes answer ``rejected`` and the E-stop answers ``not_enabled``,
and no GPIO is claimed (audit SF-05).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ...core.peer_address import client_host
from ...core.persistence import persistence
from ...models.tractor_control import TractorCommand
from ...services.tractor_service import TractorControlService, get_tractor_service
from ..deps import require_operator_auth

logger = logging.getLogger(__name__)

# Audit rows of the open stop routes are coalesced per this window (same value as api/rest.py).
ESTOP_AUDIT_WINDOW_S = 5.0

router = APIRouter()


class ValueBody(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    duration_ms: int | None = Field(None, ge=1, le=60_000)


class BladeBody(BaseModel):
    engaged: bool


class ClearBody(BaseModel):
    reason: str | None = Field(None, max_length=500, description="Operator note, audited")


def _platform_refusal(svc: TractorControlService) -> dict[str, Any] | None:
    """``rejected`` when the platform is disabled and this is not SIM_MODE, else ``None``."""
    why = svc.platform_refusal()
    if why is None:
        return None
    logger.warning("Tractor write refused: %s", why)
    return {"status": "rejected", "reason": why}


@router.get("/tractor/state")
async def tractor_state() -> dict[str, Any]:
    return get_tractor_service().get_state().model_dump(mode="json")


@router.post("/tractor/left-lever", dependencies=[Depends(require_operator_auth)])
async def tractor_left_lever(body: ValueBody) -> dict[str, Any]:
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.set_left_lever(body.value, source="operator", duration_ms=body.duration_ms)


@router.post("/tractor/right-lever", dependencies=[Depends(require_operator_auth)])
async def tractor_right_lever(body: ValueBody) -> dict[str, Any]:
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.set_right_lever(body.value, source="operator", duration_ms=body.duration_ms)


@router.post("/tractor/throttle", dependencies=[Depends(require_operator_auth)])
async def tractor_throttle(body: ValueBody) -> dict[str, Any]:
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.set_throttle(body.value, source="operator", duration_ms=body.duration_ms)


@router.post("/tractor/blade", dependencies=[Depends(require_operator_auth)])
async def tractor_blade(body: BladeBody) -> dict[str, Any]:
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.engage_blade(body.engaged, source="operator")


@router.post("/tractor/starter", dependencies=[Depends(require_operator_auth)])
async def tractor_starter() -> dict[str, Any]:
    """Run the engine-start sequence (interlock-gated)."""
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.start_engine()


@router.post("/tractor/stop-engine", dependencies=[Depends(require_operator_auth)])
async def tractor_stop_engine() -> dict[str, Any]:
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.stop_engine()


@router.post("/tractor/command", dependencies=[Depends(require_operator_auth)])
async def tractor_command(command: TractorCommand) -> dict[str, Any]:
    """Apply a full actuation command, honoring interlocks."""
    svc = get_tractor_service()
    if (refused := _platform_refusal(svc)) is not None:
        return refused
    return await svc.apply(command, source="operator")


@router.post("/tractor/authorize", dependencies=[Depends(require_operator_auth)])
async def tractor_authorize() -> dict[str, Any]:
    get_tractor_service().authorize()
    return {"status": "authorized"}


@router.post("/tractor/revoke", dependencies=[Depends(require_operator_auth)])
async def tractor_revoke() -> dict[str, Any]:
    """Withdraw the authorization and stand the machine down (audit SAF-01, T15).

    Not just a flag: the blade is switched off first, autonomous lever motion goes neutral and
    every mission or AI run ends, because none of them may command anything without the
    authorization. A person's own manual drive is left alone. The body lists any step that
    could not be done (``errors``).
    """
    return await get_tractor_service().withdraw_authorization(source="operator")


@router.post("/tractor/emergency-stop")
async def tractor_emergency_stop(request: Request = None) -> dict[str, Any]:
    """Latch the E-stop and drive the blade/levers/throttle to safe (always available).

    The body reports each output's actual result (``complete``, ``subsystems``,
    ``faults``): ``complete: false`` means the latch is set but at least one
    output could not be confirmed driven. ``status: "not_enabled"`` means the platform is
    disabled in ``config/tractor.yaml`` and nothing was touched. The stop is shielded from a
    cancelled request: once started it runs to completion. ``latch_persisted: false`` means
    the latch could not be saved to disk (a restart would release it).

    The route is open, so its audit row is the only record of WHO stopped the machine: it is
    written after the stop (never before, never able to delay or fail it), off the event loop and
    coalesced per 5 s window like the REST stop routes (``core/persistence.py::AuditCoalescer``),
    attributed to the connection's peer address (``client_host``), never to a client header.
    """
    result = await asyncio.shield(get_tractor_service().emergency_stop())
    try:
        persistence.add_audit_log_coalesced(
            "tractor.emergency_stop",
            client_id=client_host(request),
            resource="tractor_emergency_stop",
            details={"response": result},
            window_s=ESTOP_AUDIT_WINDOW_S,
        )
    except Exception:
        logger.exception("Audit log write failed for the tractor E-stop")
    return result


@router.post("/tractor/clear-emergency", dependencies=[Depends(require_operator_auth)])
async def tractor_clear_emergency(
    request: Request, body: ClearBody | None = None
) -> dict[str, Any]:
    """Explicit authenticated clear of the E-stop latch and, when safe, the blade interlock latch.

    ``status`` is ``ok`` when everything was released, ``partial`` when the blade latch had to
    stay (``still_latched``), ``rejected`` while a stop is still in flight or the tractor is
    locked out. The call and its outcome are written to the audit log with ``reason``, attributed
    to the connection's peer address (never to a client-written header).
    """
    result = await get_tractor_service().clear_emergency()
    try:
        persistence.add_audit_log(
            "tractor.clear_emergency",
            client_id=client_host(request),
            resource="tractor_emergency_stop",
            details={"reason": body.reason if body else None, "result": result},
        )
    except Exception:
        logger.exception("Audit log write failed for the tractor E-stop clear")
    return result
