"""Motor dry-run route: ``POST /motors/drive`` mixes throttle and turn into left/right PWM.

It computes the numbers only and never commands hardware; it needs the operator bearer token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from .deps import require_operator_auth

router = APIRouter()


class DriveCommand(BaseModel):
    throttle: float = Field(..., description="-1.0..1.0")
    turn: float = Field(..., description="-1.0..1.0")

    @field_validator("throttle", "turn")
    @classmethod
    def range_check(cls, v: float):
        if v < -1.0 or v > 1.0:
            raise ValueError("must be between -1.0 and 1.0")
        return float(v)


def _clamp_pwm(v: int) -> int:
    return max(-255, min(255, v))


@router.post("/motors/drive", dependencies=[Depends(require_operator_auth)])
def drive(cmd: DriveCommand) -> dict[str, dict[str, int]]:
    # Compute left/right PWM using arcade mix (dry-run, no hardware)
    left = _clamp_pwm(int((cmd.throttle + cmd.turn) * 255))
    right = _clamp_pwm(int((cmd.throttle - cmd.turn) * 255))
    return {"pwm": {"left": left, "right": right}}
