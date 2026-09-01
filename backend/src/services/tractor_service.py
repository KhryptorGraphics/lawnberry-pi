"""Zero-turn mower actuation service (Toro TimeCutter, twin-lever hydrostatic drive).

Coordinates the mower's actuators through a PCA9685 I2C PWM driver (left/right
drive-lever + throttle servos) and GPIO relays (starter, blade PTO), enforcing
the standard interlocks:

- **Start sequence**: engine cranks only when authorized, both drive levers
  neutral, and the blade/PTO off.
- **Blade/PTO**: engages only with the engine running and not in reverse;
  commanding both levers back into reverse auto-disengages the blade (Reverse
  Operation System). Reverse is deliberately both levers back together -- a
  single lever back is a routine pivot turn, not reverse.
- **E-stop**: blade off first, then both levers to neutral and throttle to
  idle -- the engine keeps running (per configuration).

SIM-safe: positional commands degrade to state-tracking when no PCA9685 board
is present, and relays only touch GPIO on real hardware.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import yaml

from ..drivers.actuators import RelayActuator, ServoActuator, ServoCalibration
from ..drivers.actuators.pca9685_driver import PCA9685Driver
from ..models.tractor_control import (
    LEVER_NEUTRAL_EPS,
    EngineState,
    TractorCommand,
    TractorState,
)

logger = logging.getLogger(__name__)


def _load_tractor_config() -> dict[str, Any]:
    """Load ``config/tractor.yaml`` if present; otherwise return {} (use defaults)."""
    config_dir = os.getenv("LAWNBERRY_CONFIG_DIR", "config")
    path = os.path.join(config_dir, "tractor.yaml")
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - malformed config
        logger.warning("Failed to load %s: %s; using defaults", path, exc)
        return {}


def _servo_cal(raw: dict | None, channel: int, bidirectional: bool) -> ServoCalibration:
    raw = raw or {}
    return ServoCalibration(
        channel=int(raw.get("channel", channel)),
        us_min=int(raw.get("us_min", 1000)),
        us_center=int(raw.get("us_center", 1500)),
        us_max=int(raw.get("us_max", 2000)),
        bidirectional=bool(raw.get("bidirectional", bidirectional)),
    )


class TractorControlService:
    """Actuation + interlock coordinator for the zero-turn mower."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else _load_tractor_config()
        self.enabled = bool(cfg.get("enabled", False))
        self._pca9685 = PCA9685Driver(cfg.get("pca9685", {}) or {})

        # PCA9685 channels are 0-indexed.
        act = cfg.get("actuators", {}) or {}
        self.left_lever = ServoActuator("left_lever", _servo_cal(act.get("left_lever"), 0, True))
        self.throttle = ServoActuator("throttle", _servo_cal(act.get("throttle"), 1, False))
        self.right_lever = ServoActuator("right_lever", _servo_cal(act.get("right_lever"), 2, True))

        relays = cfg.get("relays", {}) or {}
        starter_cfg = relays.get("starter", {}) or {}
        blade_cfg = relays.get("blade_pto", {}) or {}
        self.starter = RelayActuator(
            "starter", int(starter_cfg.get("gpio", 5)), bool(starter_cfg.get("active_high", True))
        )
        self.blade_pto = RelayActuator(
            # Default moved off GPIO 6 (spec/hardware.yaml: ToF Left Interrupt) to GPIO 26.
            "blade_pto",
            int(blade_cfg.get("gpio", 26)),
            bool(blade_cfg.get("active_high", True)),
        )
        self.starter_pulse_ms = int(starter_cfg.get("pulse_ms", 800))

        il = cfg.get("interlocks", {}) or {}
        self.require_levers_neutral_to_start = bool(il.get("require_levers_neutral_to_start", True))
        self.require_blade_off_to_start = bool(il.get("require_blade_off_to_start", True))
        self.require_engine_running_for_blade = bool(
            il.get("require_engine_running_for_blade", True)
        )
        self.disengage_blade_in_reverse = bool(il.get("disengage_blade_in_reverse", True))

        # Safe initial state: both levers neutral, blade off, engine off.
        self.state = TractorState(enabled=self.enabled)
        self.initialized = False

    # ----------------------------- lifecycle -----------------------------

    async def initialize(self) -> None:
        # Bring the PCA9685 up first so the parking commands below actually reach it.
        await self._pca9685.initialize()
        # Park everything safely: levers centered, throttle idle, blade off.
        await self._apply_servo(self.left_lever, 0.0, "left_lever")
        await self._apply_servo(self.right_lever, 0.0, "right_lever")
        await self._apply_servo(self.throttle, 0.0, "throttle")
        self.blade_pto.set(False)
        self.state.blade_engaged = False
        self.initialized = True

    # ----------------------------- helpers -------------------------------

    def authorize(self) -> None:
        self.state.authorized = True

    def revoke(self) -> None:
        self.state.authorized = False

    def _reject(self, reason: str) -> dict[str, Any]:
        self.state.interlock_reason = reason
        self.state.last_updated = datetime.now(UTC)
        logger.info("Tractor command rejected: %s", reason)
        return {"status": "rejected", "reason": reason}

    def _ok(self, **extra: Any) -> dict[str, Any]:
        self.state.interlock_reason = None
        self.state.last_updated = datetime.now(UTC)
        return {"status": "ok", **extra}

    async def _send_pwm(self, channel: int, us: int) -> None:
        """Send a single-channel PWM command via the PCA9685 I2C driver.

        Hardware-absent (SIM_MODE / board not wired) degrades to a no-op
        inside the driver itself, so the commanded state is still tracked and
        SIM/tests stay deterministic. A write failure on hardware that was
        previously confirmed present is deliberately NOT caught here: it
        propagates to the caller (interlocks, ``emergency_stop``) so a real
        mid-mission transport fault is visible instead of a false "ok".
        """
        await self._pca9685.set_pwm_us(channel, us)

    async def _apply_servo(self, actuator: ServoActuator, value: float, field: str) -> None:
        us = actuator.command(value)
        await self._send_pwm(actuator.cal.channel, us)
        setattr(self.state, field, actuator.value)

    # --------------------------- actuator API ----------------------------

    async def set_levers(self, left: float, right: float) -> dict[str, Any]:
        """Command both drive levers together.

        The single chokepoint for lever commands: ``set_left_lever``,
        ``set_right_lever`` and ``apply()`` all funnel through here, so the
        e-stop guard and the reverse-entry blade auto-disengage exist in
        exactly one place instead of being duplicated per entry point.
        """
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        # Reverse Operation System: if this command would put both levers back
        # (reverse) while the blade is engaged, drop the blade first -- before
        # moving the levers. A single lever going negative is a pivot turn,
        # not reverse; see TractorState.reversing.
        would_reverse = left < -LEVER_NEUTRAL_EPS and right < -LEVER_NEUTRAL_EPS
        if would_reverse and self.state.blade_engaged and self.disengage_blade_in_reverse:
            logger.info("ROS: disengaging blade before entering reverse")
            await self.engage_blade(False)
        await self._apply_servo(self.left_lever, left, "left_lever")
        await self._apply_servo(self.right_lever, right, "right_lever")
        return self._ok(
            left_lever=self.state.left_lever,
            right_lever=self.state.right_lever,
            moving=self.state.moving,
        )

    async def set_left_lever(self, value: float) -> dict[str, Any]:
        return await self.set_levers(value, self.state.right_lever)

    async def set_right_lever(self, value: float) -> dict[str, Any]:
        return await self.set_levers(self.state.left_lever, value)

    async def set_throttle(self, value: float) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        await self._apply_servo(self.throttle, value, "throttle")
        return self._ok(throttle=self.state.throttle)

    async def engage_blade(self, on: bool) -> dict[str, Any]:
        if on:
            if self.state.emergency_stop_active:
                return self._reject("emergency stop active")
            if not self.state.authorized:
                return self._reject("motors not authorized")
            if self.require_engine_running_for_blade and not self.state.engine_running:
                return self._reject("engine must be running to engage blade")
            if self.disengage_blade_in_reverse and self.state.reversing:
                return self._reject("cannot engage blade while in reverse")
        self.blade_pto.set(on)
        self.state.blade_engaged = bool(on)
        return self._ok(blade_engaged=self.state.blade_engaged)

    async def start_engine(self) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        if not self.state.authorized:
            return self._reject("motors not authorized")
        if self.require_blade_off_to_start and self.state.blade_engaged:
            return self._reject("blade/PTO must be disengaged to start")
        if self.require_levers_neutral_to_start and (
            abs(self.state.left_lever) > LEVER_NEUTRAL_EPS
            or abs(self.state.right_lever) > LEVER_NEUTRAL_EPS
        ):
            return self._reject("drive levers must be neutral to start")

        self.state.engine = EngineState.STARTING
        await self.starter.pulse(self.starter_pulse_ms)
        # No engine-run sensor on the conversion: assume a successful crank.
        self.state.engine = EngineState.RUNNING
        return self._ok(engine=self.state.engine.value)

    async def stop_engine(self) -> dict[str, Any]:
        # Disengage blade first, then mark engine off.
        await self.engage_blade(False)
        self.state.engine = EngineState.OFF
        return self._ok(engine=self.state.engine.value)

    async def emergency_stop(self) -> dict[str, Any]:
        """Disengage the blade and center/idle the drive; leave the engine running.

        Each PWM-bearing actuation gets its own try/except: a PCA9685 bus
        fault on one channel must not abort the rest of the safing sequence.
        The blade-PTO relay cutoff is pure GPIO (unaffected by the I2C
        transport) and stays first and unconditional.
        """
        self.state.emergency_stop_active = True
        self.state.last_updated = datetime.now(UTC)
        self.revoke()
        self.blade_pto.set(False)
        self.state.blade_engaged = False

        try:
            await self._apply_servo(self.left_lever, 0.0, "left_lever")
        except Exception:
            logger.exception("Tractor e-stop: left-lever-to-neutral actuation failed")
        try:
            await self._apply_servo(self.right_lever, 0.0, "right_lever")
        except Exception:
            logger.exception("Tractor e-stop: right-lever-to-neutral actuation failed")
        try:
            await self._apply_servo(self.throttle, 0.0, "throttle")  # idle engine
        except Exception:
            logger.exception("Tractor e-stop: throttle-to-idle actuation failed")

        self.state.interlock_reason = "emergency_stop"
        logger.warning("Tractor EMERGENCY STOP: blade off, levers neutral, throttle idle")
        return {"status": "emergency_stop", "engine": self.state.engine.value}

    async def clear_emergency(self) -> dict[str, Any]:
        self.state.emergency_stop_active = False
        self.state.interlock_reason = None
        return self._ok()

    async def apply(self, command: TractorCommand) -> dict[str, Any]:
        """Apply a full command, honoring interlocks (rejections are collected)."""
        results: dict[str, Any] = {}
        results["throttle"] = await self.set_throttle(command.throttle)
        results["levers"] = await self.set_levers(command.left_lever, command.right_lever)
        # Blade last: a reverse+blade-on command surfaces as a visible
        # "rejected" here rather than engaging then immediately auto-dropping.
        results["blade"] = await self.engage_blade(command.blade_engaged)
        return {"status": "applied", "results": results, "moving": self.state.moving}

    def get_state(self) -> TractorState:
        return self.state.model_copy()


_tractor_service: TractorControlService | None = None


def get_tractor_service() -> TractorControlService:
    global _tractor_service
    if _tractor_service is None:
        _tractor_service = TractorControlService()
    return _tractor_service
