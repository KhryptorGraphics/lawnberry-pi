"""Ride-on lawn tractor control endpoints (v2).

Operate the tractor's discrete actuators (steering, throttle, ground-speed
pedal, clutch, gear, blade PTO, starter) with the standard safety interlocks
enforced by ``TractorControlService``. Writes require operator auth; the
emergency stop and state read are always available.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...models.tractor_control import TractorCommand, Transmission
from ...services.tractor_service import get_tractor_service
from ..deps import require_operator_auth

router = APIRouter()


class ValueBody(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)


class GearBody(BaseModel):
    gear: Transmission


class BladeBody(BaseModel):
    engaged: bool


@router.get("/tractor/state")
async def tractor_state() -> dict[str, Any]:
    return get_tractor_service().get_state().model_dump(mode="json")


@router.post("/tractor/steering", dependencies=[Depends(require_operator_auth)])
async def tractor_steering(body: ValueBody) -> dict[str, Any]:
    return await get_tractor_service().set_steering(body.value)


@router.post("/tractor/throttle", dependencies=[Depends(require_operator_auth)])
async def tractor_throttle(body: ValueBody) -> dict[str, Any]:
    return await get_tractor_service().set_throttle(body.value)


@router.post("/tractor/speed", dependencies=[Depends(require_operator_auth)])
async def tractor_speed(body: ValueBody) -> dict[str, Any]:
    """Set the ground-speed / gas pedal (0..1)."""
    return await get_tractor_service().set_ground_speed(body.value)


@router.post("/tractor/clutch", dependencies=[Depends(require_operator_auth)])
async def tractor_clutch(body: ValueBody) -> dict[str, Any]:
    return await get_tractor_service().set_clutch(body.value)


@router.post("/tractor/gear", dependencies=[Depends(require_operator_auth)])
async def tractor_gear(body: GearBody) -> dict[str, Any]:
    return await get_tractor_service().set_gear(body.gear)


@router.post("/tractor/blade", dependencies=[Depends(require_operator_auth)])
async def tractor_blade(body: BladeBody) -> dict[str, Any]:
    return await get_tractor_service().engage_blade(body.engaged)


@router.post("/tractor/starter", dependencies=[Depends(require_operator_auth)])
async def tractor_starter() -> dict[str, Any]:
    """Run the engine-start sequence (interlock-gated)."""
    return await get_tractor_service().start_engine()


@router.post("/tractor/stop-engine", dependencies=[Depends(require_operator_auth)])
async def tractor_stop_engine() -> dict[str, Any]:
    return await get_tractor_service().stop_engine()


@router.post("/tractor/command", dependencies=[Depends(require_operator_auth)])
async def tractor_command(command: TractorCommand) -> dict[str, Any]:
    """Apply a full actuation command, honoring interlocks."""
    return await get_tractor_service().apply(command)


@router.post("/tractor/authorize", dependencies=[Depends(require_operator_auth)])
async def tractor_authorize() -> dict[str, Any]:
    get_tractor_service().authorize()
    return {"status": "authorized"}


@router.post("/tractor/revoke", dependencies=[Depends(require_operator_auth)])
async def tractor_revoke() -> dict[str, Any]:
    get_tractor_service().revoke()
    return {"status": "revoked"}


@router.post("/tractor/emergency-stop")
async def tractor_emergency_stop() -> dict[str, Any]:
    """Disengage drive + blade and brake immediately (always available)."""
    return await get_tractor_service().emergency_stop()


@router.post("/tractor/clear-emergency", dependencies=[Depends(require_operator_auth)])
async def tractor_clear_emergency() -> dict[str, Any]:
    return await get_tractor_service().clear_emergency()
