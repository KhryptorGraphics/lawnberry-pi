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
    # How long (ms) the lever/throttle values stay in force before the service's
    # deadman returns levers to neutral and throttle to idle. None = the
    # service's configured default hold. The service clamps this to its
    # configured maximum; the effective value is echoed in the response.
    duration_ms: int | None = Field(None, ge=1, le=60_000)


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

    # Actuator hardware health (PCA9685 + relays). False => lever/throttle/blade/
    # start commands are refused rather than reported as success.
    actuator_ok: bool = False
    actuator_fault: str | None = None
    simulated: bool = False

    # Blade interlock latch (tilt, ...): the blade stays off, and
    # engage_blade(True) -- including from navigation -- is refused, until an
    # authenticated clear (POST /tractor/clear-emergency).
    interlock_latched: bool = False
    latch_reason: str | None = None

    # Obstacle stop (safety/obstacle_stop.py). ``speed_cap_mps`` is the forward ground speed
    # autonomous sources may currently command (None when no limiter is attached); 0 = hold.
    # ``motion_limited`` is true while the guard is not CLEAR (an obstacle in view, a hold, or
    # a failed/stale sensor), with the reasons in ``motion_limit_reason``.
    speed_cap_mps: float | None = None
    motion_limited: bool = False
    motion_limit_reason: str | None = None

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
