"""Ride-on lawn tractor actuation service (Craftsman-class conversion).

Coordinates the seven tractor actuators through a PCA9685 I2C PWM driver
(steering/throttle/gas-pedal/clutch/gear) and GPIO relays (starter, blade
PTO), enforcing the standard lawn-tractor safety interlocks:

- **Start sequence**: engine cranks only when authorized, in NEUTRAL, blade/PTO
  off, and the clutch/brake pedal pressed.
- **Blade/PTO**: engages only with the engine running and not in reverse;
  selecting reverse auto-disengages the blade (Reverse Operation System).
- **E-stop**: disengages the blade, shifts to neutral, presses the clutch/brake,
  and idles throttle/pedal — the engine keeps running (per configuration).

SIM-safe: positional commands degrade to state-tracking when no PCA9685 board
is present, and relays only touch GPIO on real hardware.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from ..drivers.actuators import (
    GearActuator,
    GearCalibration,
    RelayActuator,
    ServoActuator,
    ServoCalibration,
)
from ..drivers.actuators.pca9685_driver import PCA9685Driver
from ..models.tractor_control import (
    EngineState,
    TractorCommand,
    TractorState,
    Transmission,
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


def _gear_cal(raw: dict | None, channel: int) -> GearCalibration:
    raw = raw or {}
    return GearCalibration(
        channel=int(raw.get("channel", channel)),
        us_forward=int(raw.get("us_forward", 1900)),
        us_neutral=int(raw.get("us_neutral", 1500)),
        us_reverse=int(raw.get("us_reverse", 1100)),
    )


class TractorControlService:
    """Actuation + interlock coordinator for the lawn tractor."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else _load_tractor_config()
        self.enabled = bool(cfg.get("enabled", False))
        self._pca9685 = PCA9685Driver(cfg.get("pca9685", {}) or {})

        # PCA9685 channels are 0-indexed (0-4), unlike the old 1-5 RoboHAT convention.
        act = cfg.get("actuators", {}) or {}
        self.steering = ServoActuator("steering", _servo_cal(act.get("steering"), 0, True))
        self.throttle = ServoActuator("throttle", _servo_cal(act.get("throttle"), 1, False))
        self.gas_pedal = ServoActuator("gas_pedal", _servo_cal(act.get("gas_pedal"), 2, False))
        self.clutch = ServoActuator("clutch", _servo_cal(act.get("clutch"), 3, False))
        self.gear = GearActuator("gear", _gear_cal(act.get("gear"), 4))

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
        self.require_neutral_to_start = bool(il.get("require_neutral_to_start", True))
        self.require_clutch_to_start = bool(il.get("require_clutch_to_start", True))
        self.require_blade_off_to_start = bool(il.get("require_blade_off_to_start", True))
        self.require_engine_running_for_blade = bool(
            il.get("require_engine_running_for_blade", True)
        )
        self.disengage_blade_in_reverse = bool(il.get("disengage_blade_in_reverse", True))
        self.clutch_pressed_threshold = float(il.get("clutch_pressed_threshold", 0.9))

        # Safe initial state: clutch/brake pressed, neutral, blade off, engine off.
        self.state = TractorState(clutch=1.0)
        self.initialized = False

    # ----------------------------- lifecycle -----------------------------

    async def initialize(self) -> None:
        # Bring the PCA9685 up first so the parking commands below actually reach it.
        await self._pca9685.initialize()
        # Park everything safely.
        await self._apply_servo(self.clutch, 1.0, "clutch")
        await self._apply_servo(self.throttle, 0.0, "throttle")
        await self._apply_servo(self.gas_pedal, 0.0, "ground_speed")
        await self._apply_servo(self.steering, 0.0, "steering")
        await self._apply_gear(Transmission.NEUTRAL)
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
        logger.info("Tractor command rejected: %s", reason)
        return {"status": "rejected", "reason": reason}

    def _ok(self, **extra: Any) -> dict[str, Any]:
        self.state.interlock_reason = None
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

    async def _apply_gear(self, gear: Transmission) -> None:
        us = self.gear.command(gear.value)
        await self._send_pwm(self.gear.cal.channel, us)
        self.state.gear = gear

    # --------------------------- actuator API ----------------------------

    async def set_steering(self, value: float) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        await self._apply_servo(self.steering, value, "steering")
        return self._ok(steering=self.state.steering)

    async def set_throttle(self, value: float) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        await self._apply_servo(self.throttle, value, "throttle")
        return self._ok(throttle=self.state.throttle)

    async def set_ground_speed(self, value: float) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        await self._apply_servo(self.gas_pedal, value, "ground_speed")
        return self._ok(ground_speed=self.state.ground_speed, moving=self.state.moving)

    async def set_clutch(self, value: float) -> dict[str, Any]:
        await self._apply_servo(self.clutch, value, "clutch")
        return self._ok(clutch=self.state.clutch)

    async def set_gear(self, gear: Transmission) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        # Reverse Operation System: drop the blade before going into reverse.
        if (
            gear == Transmission.REVERSE
            and self.state.blade_engaged
            and self.disengage_blade_in_reverse
        ):
            logger.info("ROS: disengaging blade before selecting reverse")
            await self.engage_blade(False)
        await self._apply_gear(gear)
        return self._ok(gear=self.state.gear.value, blade_engaged=self.state.blade_engaged)

    async def engage_blade(self, on: bool) -> dict[str, Any]:
        if on:
            if self.state.emergency_stop_active:
                return self._reject("emergency stop active")
            if not self.state.authorized:
                return self._reject("motors not authorized")
            if self.require_engine_running_for_blade and not self.state.engine_running:
                return self._reject("engine must be running to engage blade")
            if self.disengage_blade_in_reverse and self.state.gear == Transmission.REVERSE:
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
        if self.require_neutral_to_start and self.state.gear != Transmission.NEUTRAL:
            return self._reject("transmission must be in neutral to start")
        if self.require_clutch_to_start and self.state.clutch < self.clutch_pressed_threshold:
            return self._reject("clutch/brake must be pressed to start")

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
        """Disengage drive + blade and brake; leave the engine running.

        Each PWM-bearing actuation gets its own try/except: a PCA9685 bus
        fault on one channel must not abort the rest of the safing sequence.
        The blade-PTO relay cutoff is pure GPIO (unaffected by the I2C
        transport) and stays first and unconditional.
        """
        self.state.emergency_stop_active = True
        self.revoke()
        self.blade_pto.set(False)
        self.state.blade_engaged = False

        try:
            await self._apply_servo(self.gas_pedal, 0.0, "ground_speed")
        except Exception:
            logger.exception("Tractor e-stop: gas-pedal-to-0 actuation failed")
        try:
            await self._apply_gear(Transmission.NEUTRAL)
        except Exception:
            logger.exception("Tractor e-stop: gear-to-neutral actuation failed")
        try:
            await self._apply_servo(self.clutch, 1.0, "clutch")  # press clutch/brake
        except Exception:
            logger.exception("Tractor e-stop: clutch/brake actuation failed")
        try:
            await self._apply_servo(self.throttle, 0.0, "throttle")  # idle engine
        except Exception:
            logger.exception("Tractor e-stop: throttle-to-idle actuation failed")

        self.state.interlock_reason = "emergency_stop"
        logger.warning("Tractor EMERGENCY STOP: drive+blade disengaged, brake set")
        return {"status": "emergency_stop", "engine": self.state.engine.value}

    async def clear_emergency(self) -> dict[str, Any]:
        self.state.emergency_stop_active = False
        self.state.interlock_reason = None
        return self._ok()

    async def apply(self, command: TractorCommand) -> dict[str, Any]:
        """Apply a full command, honoring interlocks (rejections are collected)."""
        results: dict[str, Any] = {}
        results["steering"] = await self.set_steering(command.steering)
        results["throttle"] = await self.set_throttle(command.throttle)
        results["clutch"] = await self.set_clutch(command.clutch)
        results["gear"] = await self.set_gear(command.gear)
        results["ground_speed"] = await self.set_ground_speed(command.ground_speed)
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
