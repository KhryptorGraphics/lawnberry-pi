"""Actuation model for a 50" Toro TimeCutter zero-turn mower conversion.

The mower's gas engine and hydrostatic transaxles are kept completely
intact; two high-torque servos physically push/pull the existing twin drive
levers instead of an Ackermann steering rack. The actuation model is:

- **left_lever** / **right_lever** — continuous, -1 (full reverse) .. +1
  (full forward), 0 = neutral detent
- **throttle**    — continuous, 0 (idle) .. 1 (full engine RPM)
- **blade** (PTO) — on/off power take-off clutch
- **starter**     — momentary engine-crank relay (an action, not a sustained state)

Positional actuators (levers, throttle) are driven as PCA9685 I2C PWM
channels; starter and blade PTO are GPIO relays. See
``services/tractor_service.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

# Lever magnitude below which a lever reads as "neutral" (in its detent).
LEVER_NEUTRAL_EPS = 0.05


class EngineState(StrEnum):
    """Engine running state."""

    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"


class TractorCommand(BaseModel):
    """A complete desired actuation state for the mower.

    Each field is independent; partial updates are applied via the per-actuator
    service methods.
    """

    left_lever: float = Field(0.0, ge=-1.0, le=1.0)
    right_lever: float = Field(0.0, ge=-1.0, le=1.0)
    throttle: float = Field(0.0, ge=0.0, le=1.0)
    blade_engaged: bool = False


class TractorState(BaseModel):
    """Observed/last-commanded actuator state plus interlock status."""

    left_lever: float = 0.0
    right_lever: float = 0.0
    throttle: float = 0.0
    blade_engaged: bool = False
    engine: EngineState = EngineState.OFF
    enabled: bool = False  # platform-detection flag: is a tractor actually configured

    # Safety / interlock status
    emergency_stop_active: bool = False
    authorized: bool = False
    interlock_reason: str | None = None

    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def engine_running(self) -> bool:
        return self.engine == EngineState.RUNNING

    @computed_field  # type: ignore[prop-decorator]
    @property
    def moving(self) -> bool:
        """Whether the drivetrain is delivering motion."""
        return self.engine_running and (
            abs(self.left_lever) > LEVER_NEUTRAL_EPS or abs(self.right_lever) > LEVER_NEUTRAL_EPS
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reversing(self) -> bool:
        """True only when BOTH levers are pulled back past neutral.

        A single lever going negative while the other stays forward is a
        routine zero-radius pivot turn while mowing forward, not reverse --
        treating that as reverse would drop the blade on every pivot turn.
        Reverse is deliberately defined as both levers pulled back together.
        """
        return self.left_lever < -LEVER_NEUTRAL_EPS and self.right_lever < -LEVER_NEUTRAL_EPS


__all__ = [
    "LEVER_NEUTRAL_EPS",
    "EngineState",
    "TractorCommand",
    "TractorState",
]
