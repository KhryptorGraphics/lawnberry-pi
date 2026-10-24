from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .deps import require_operator_auth

# Reuse safety/blade state used by existing v2 control endpoints
from .rest import _blade_state, _safety_state  # minimal coupling to keep behavior consistent

logger = logging.getLogger(__name__)

router = APIRouter()


class EmergencyClearRequest(BaseModel):
    confirmation: bool = Field(False, description="Operator confirmation required to clear E-stop")
    reason: str | None = Field(default=None, description="Optional reason or operator note")


def _tractor_still_latched() -> list[str]:
    """Tractor latches the general clear does NOT release (cleared only by
    ``POST /tractor/clear-emergency``, each an explicit authenticated act)."""
    try:
        from ..services.tractor_service import get_tractor_service

        tractor = get_tractor_service()
        if not tractor.enabled:
            return []
        latched = []
        if tractor.state.emergency_stop_active:
            latched.append("tractor_emergency_stop")
        if tractor.state.interlock_latched:
            latched.append(f"tractor_interlock:{tractor.state.latch_reason}")
        return latched
    except Exception:
        return []


@router.post("/control/emergency_clear", dependencies=[Depends(require_operator_auth)])
async def clear_emergency_stop(payload: EmergencyClearRequest):
    """Clear emergency stop after explicit operator confirmation.

    Behavior:
    - Requires operator authentication (same dependency as the other control writes)
    - Requires payload.confirmation == True, otherwise 422
    - If nothing is latched, returns 200 idempotently
    - Clears E-stop latch and keeps blade off (also releases the IBT-4 blade latch and
      the RoboHAT stop hold, even when only those were set, e.g. by the navigation stop)
    - Attempts to clear emergency on RoboHAT if connected
    - Does NOT clear a latched tractor E-stop / blade interlock: those need their own
      authenticated ``POST /tractor/clear-emergency`` and are listed in ``still_latched``
    """
    # Enforce explicit confirmation
    if not payload.confirmation:
        raise HTTPException(status_code=422, detail="Confirmation required to clear emergency stop")

    from ..services.blade_service import get_blade_service
    from ..services.robohat_service import get_robohat_service

    blade_service = get_blade_service()
    robohat = get_robohat_service()
    subsystem_latched = blade_service.emergency_latched or bool(
        robohat and robohat.emergency_latched
    )

    # If nothing is latched, respond idempotently
    if not _safety_state.get("emergency_stop_active", False) and not subsystem_latched:
        return {"status": "EMERGENCY_CLEARED", "idempotent": True}

    # Clear local safety state and ensure blade remains off
    _safety_state["emergency_stop_active"] = False
    _blade_state["active"] = False

    # Release the IBT-4 blade latch (the blade stays off until explicitly re-enabled)
    try:
        await blade_service.clear_emergency()
    except Exception:
        logger.warning("Blade service E-stop latch release failed", exc_info=True)

    # Inform RoboHAT firmware if available
    try:
        if robohat and robohat.status.serial_connected:
            await robohat.clear_emergency()
    except Exception:
        # Don't fail the clear operation if hardware isn't present in SIM/CI
        pass

    # Minimal response aligned with integration tests
    response: dict[str, object] = {
        "status": "EMERGENCY_CLEARED",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    still_latched = _tractor_still_latched()
    if still_latched:
        response["still_latched"] = still_latched
    return response
