"""Actuation model for a converted ride-on lawn tractor (Craftsman class).

Unlike the differential-drive robot model in ``motor_control.py`` (two wheel
motors), a lawn tractor is an Ackermann-steered, gas-engine vehicle operated
through discrete actuators:

- **steering**   — continuous, -1 (full left) .. +1 (full right)
- **throttle**   — continuous, 0 (idle) .. 1 (full engine RPM)
- **ground_speed** (gas/drive pedal) — continuous, 0 (stop) .. 1 (full)
- **gear**       — discrete forward / neutral / reverse selector
- **clutch**     — continuous, 0 (engaged/driving) .. 1 (pressed = declutched/brake)
- **blade** (PTO)— on/off power take-off clutch
- **starter**    — momentary engine-crank relay (an action, not a sustained state)

Positional actuators are driven as RC-PWM channels on the RoboHAT RP2040;
starter and blade PTO are GPIO relays. See ``services/tractor_service.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class Transmission(StrEnum):
    """Gear selector position."""

    FORWARD = "forward"
    NEUTRAL = "neutral"
    REVERSE = "reverse"


class EngineState(StrEnum):
    """Engine running state."""

    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"


class TractorCommand(BaseModel):
    """A complete desired actuation state for the tractor.

    Each field is independent; partial updates are applied via the per-actuator
    service methods. ``ground_speed`` only produces motion when the engine is
    running, a gear is selected, and the clutch is released.
    """

    steering: float = Field(0.0, ge=-1.0, le=1.0)
    throttle: float = Field(0.0, ge=0.0, le=1.0)
    ground_speed: float = Field(0.0, ge=0.0, le=1.0)
    gear: Transmission = Transmission.NEUTRAL
    clutch: float = Field(0.0, ge=0.0, le=1.0)  # 1.0 = fully pressed (declutched)
    blade_engaged: bool = False


class TractorState(BaseModel):
    """Observed/last-commanded actuator state plus interlock status."""

    steering: float = 0.0
    throttle: float = 0.0
    ground_speed: float = 0.0
    gear: Transmission = Transmission.NEUTRAL
    clutch: float = 1.0  # safe default: clutch pressed (no drive)
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
        return (
            self.engine_running
            and self.gear != Transmission.NEUTRAL
            and self.clutch < 0.5
            and self.ground_speed > 0.0
        )


__all__ = [
    "Transmission",
    "EngineState",
    "TractorCommand",
    "TractorState",
]
