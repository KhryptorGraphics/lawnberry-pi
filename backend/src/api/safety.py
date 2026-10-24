from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.persistence import persistence
from ..services import emergency_stop_service as ess
from .deps import require_operator_auth

# Reuse safety/blade state used by existing v2 control endpoints
from .rest import _blade_state, _safety_state  # minimal coupling to keep behavior consistent

logger = logging.getLogger(__name__)

router = APIRouter()


class EmergencyClearRequest(BaseModel):
    confirmation: bool = Field(False, description="Operator confirmation required to clear E-stop")
    reason: str | None = Field(
        default=None, max_length=500, description="Optional reason or operator note"
    )


def _tractor_still_latched() -> list[str]:
    """Tractor latches the general clear does NOT release (cleared only by
    ``POST /tractor/clear-emergency``, each an explicit authenticated act).

    Fails closed: a tractor whose latch state cannot be read is reported as latched, never as
    clear (an unreadable state used to read as an empty list, i.e. "nothing latched").
    """
    try:
        from ..services.tractor_service import get_tractor_service

        tractor = get_tractor_service()
        latched: list[str] = []
        if tractor.lockout_reason is not None:
            latched.append(f"tractor_locked_out: {tractor.lockout_reason}")
        if not tractor.enabled:
            return latched
        if tractor.state.emergency_stop_active:
            latched.append("tractor_emergency_stop")
        if tractor.state.interlock_latched:
            latched.append(f"tractor_interlock:{tractor.state.latch_reason}")
        return latched
    except Exception as exc:
        logger.exception("Tractor latch state unreadable: reporting it as latched")
        return [f"tractor_state_unreadable: {type(exc).__name__}: {exc}"]


def _audit_clear(
    request: Request, payload: EmergencyClearRequest, outcome: str, **details: Any
) -> None:
    """Record an E-stop clear (or a refused one) with its reason: triggers were audited, clears
    were not (audit SEC-11). A failing audit store must not strand the operator."""
    try:
        client = request.headers.get("X-Client-Id") or (
            request.client.host if request.client else None
        )
        persistence.add_audit_log(
            "control.emergency_clear",
            client_id=client,
            resource="emergency_stop",
            details={"reason": payload.reason, "outcome": outcome, **details},
        )
    except Exception:
        logger.exception("Audit log write failed for the E-stop clear (%s)", outcome)


@router.post("/control/emergency_clear", dependencies=[Depends(require_operator_auth)])
async def clear_emergency_stop(payload: EmergencyClearRequest, request: Request):
    """Clear emergency stop after explicit operator confirmation.

    Behavior:
    - Requires operator authentication (same dependency as the other control writes)
    - Requires payload.confirmation == True, otherwise 422
    - Refused with 409 while an emergency stop is still driving the outputs (or if one starts
      while this clear runs): a clear must never undo a stop that is in flight
    - If nothing is latched, returns 200 idempotently
    - Clears E-stop latch and keeps blade off (also releases the IBT-4 blade latch and
      the RoboHAT stop hold, even when only those were set, e.g. by the navigation stop)
    - Attempts to clear emergency on RoboHAT if connected
    - Does NOT clear a latched tractor E-stop / blade interlock: those need their own
      authenticated ``POST /tractor/clear-emergency`` and are listed in ``still_latched``
    - ``status`` is ``EMERGENCY_CLEARED`` only when nothing the tractor holds is still latched;
      otherwise ``EMERGENCY_PARTIALLY_CLEARED`` (something was cleared) or
      ``EMERGENCY_STILL_LATCHED`` (nothing was), both with ``still_latched``
    - Every call is written to the audit log with its ``reason`` and outcome
    """
    # Enforce explicit confirmation
    if not payload.confirmation:
        raise HTTPException(status_code=422, detail="Confirmation required to clear emergency stop")

    if ess.stop_in_flight():
        return _clear_refused(request, payload, "an emergency stop is still in progress")
    epoch = ess.stop_epoch()

    from ..services.blade_service import get_blade_service
    from ..services.robohat_service import get_robohat_service

    blade_service = get_blade_service()
    robohat = get_robohat_service()
    subsystem_latched = blade_service.emergency_latched or bool(
        robohat and robohat.emergency_latched
    )
    # Read before any early return: the idempotent path used to skip it and answer
    # EMERGENCY_CLEARED while the tractor was still latched (audit SEC-07).
    still_latched = _tractor_still_latched()

    # If nothing is latched, respond idempotently
    if not _safety_state.get("emergency_stop_active", False) and not subsystem_latched:
        response: dict[str, object] = {
            "status": "EMERGENCY_STILL_LATCHED" if still_latched else "EMERGENCY_CLEARED",
            "idempotent": True,
        }
        if still_latched:
            response["still_latched"] = still_latched
        _audit_clear(
            request, payload, "idempotent", status=response["status"], still_latched=still_latched
        )
        return response

    # Clear local safety state and ensure blade remains off
    _safety_state["emergency_stop_active"] = False
    _blade_state["active"] = False

    # A stop that starts while this clear runs wins: after each await the epoch is compared and
    # the remaining releases are abandoned (the stop has just set the very latches they would
    # release; its own latch of _safety_state is already back in place).
    superseded = "an emergency stop started while the clear was running"

    # Release the IBT-4 blade latch (the blade stays off until explicitly re-enabled)
    try:
        await blade_service.clear_emergency()
    except Exception:
        logger.warning("Blade service E-stop latch release failed", exc_info=True)

    if ess.stop_epoch() != epoch or ess.stop_in_flight():
        return _clear_refused(request, payload, superseded)

    # Inform RoboHAT firmware if available
    try:
        if robohat and robohat.status.serial_connected:
            await robohat.clear_emergency()
    except Exception:
        # Don't fail the clear operation if hardware isn't present in SIM/CI
        pass

    if ess.stop_epoch() != epoch or ess.stop_in_flight():
        return _clear_refused(request, payload, superseded)

    # Minimal response aligned with integration tests
    response = {
        "status": "EMERGENCY_PARTIALLY_CLEARED" if still_latched else "EMERGENCY_CLEARED",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if still_latched:
        response["still_latched"] = still_latched
    _audit_clear(
        request, payload, "cleared", status=response["status"], still_latched=still_latched
    )
    return response


def _clear_refused(request: Request, payload: EmergencyClearRequest, why: str) -> JSONResponse:
    """409: the clear did not (or no longer) happen because an emergency stop is in flight."""
    logger.warning("E-stop clear refused: %s", why)
    _audit_clear(request, payload, "refused", why=why)
    return JSONResponse(
        status_code=409,
        content={
            "status": "EMERGENCY_STOP_IN_PROGRESS",
            "detail": f"Clear refused: {why}. Retry once the stop's result is out.",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
