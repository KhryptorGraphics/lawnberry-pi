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
  idle -- the engine keeps running (per configuration). The result reports
  every subsystem's *actual* outcome (``complete``/``subsystems``/``faults``);
  a latched E-stop with a failed output is never reported as a clean stop.
- **Interlock latch** (tilt, ...): the blade is dropped and stays off --
  ``engage_blade(True)``, including from navigation, is refused -- until an
  explicit ``clear_emergency()`` (operator-authenticated at the API).
- **Command deadman**: a lever/throttle command is only in force for a bounded
  hold (the command's ``duration_ms``, else the configured default, clamped to
  the configured maximum). Unless it is refreshed the levers return to neutral
  and the throttle to idle. The blade is deliberately *not* touched by the
  deadman; it follows the interlocks above.
- **Actuator health**: when the PCA9685 (or a relay GPIO) is absent or failed,
  commands are *refused* with the reason and the fault is exposed on
  ``TractorState`` -- never a silent no-op "ok". SIM_MODE is the exception: the
  simulated driver succeeds so the stack stays testable.

SIM-safe: positional commands degrade to state-tracking in SIM_MODE, and relays
only touch GPIO on real hardware.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import yaml

from ..core.simulation import is_simulation_mode
from ..drivers.actuators import RelayActuator, ServoActuator, ServoCalibration
from ..drivers.actuators.pca9685_driver import PCA9685Driver
from ..models.tractor_control import (
    LEVER_NEUTRAL_EPS,
    EngineState,
    TractorCommand,
    TractorState,
)

logger = logging.getLogger(__name__)

# Command-deadman defaults; ``config/tractor.yaml`` ``command_deadman`` overrides.
# Conservative on purpose: the shipped clients re-send well above 2 Hz
# (navigation 5 Hz, scripts/teleop.py 20 Hz). Field values are a bench decision.
DEFAULT_COMMAND_HOLD_MS = 500
DEFAULT_MAX_COMMAND_HOLD_MS = 2000


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


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class TractorControlService:
    """Actuation + interlock coordinator for the zero-turn mower.

    Every accepted actuation notifies an optional ``on_change`` callback
    (see ``_notify``) so the W1 capture service can log ``commands.jsonl``
    without polling; the emergency-stop path logs its safing values once,
    after the servos.
    """

    #: A single PCA9685 write must finish within this, or it counts as a fault.
    ACTUATION_TIMEOUT_S = 1.0
    #: How long an E-stop waits for an in-flight actuation to land before it
    #: writes neutral anyway (so a slow write cannot overwrite the stop).
    ESTOP_LOCK_WAIT_S = 0.25

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

        dm = cfg.get("command_deadman", {}) or {}
        self.default_hold_s = max(
            0.01, float(dm.get("default_hold_ms", DEFAULT_COMMAND_HOLD_MS)) / 1000.0
        )
        self.max_hold_s = max(
            self.default_hold_s, float(dm.get("max_hold_ms", DEFAULT_MAX_COMMAND_HOLD_MS)) / 1000.0
        )
        # Monotonic deadlines after which the levers/throttle return to neutral/idle.
        self._lever_deadline: float | None = None
        self._throttle_deadline: float | None = None
        self._deadman_task: asyncio.Task | None = None
        # Wakes the deadman watcher when a (possibly earlier) deadline is armed
        # while it sleeps towards a later one. Created with the task so it is
        # bound to the running loop.
        self._deadman_wake: asyncio.Event | None = None
        # Serialises lever/throttle writes (commands vs. deadman expiry). The
        # E-stop does not queue behind it indefinitely: see _acquire_for_estop().
        self._act_lock = asyncio.Lock()
        # Last servo write fault per PCA9685 channel (incl. timeouts), each
        # cleared by that channel's next successful write.
        self._write_faults: dict[int, str] = {}

        # Safe initial state: both levers neutral, blade off, engine off.
        self.state = TractorState(enabled=self.enabled)
        # Capture-stream hook; called on_change(kind, values, source) after every
        # accepted actuation. None when nothing subscribed.
        self.on_change: Callable[[str, dict[str, Any], str], None] | None = None
        self.initialized = False
        self._sync_hardware_state()

    def _notify(self, kind: str, source: str, **values: Any) -> None:
        """Fire the capture hook. Logging must never break actuation."""
        if self.on_change is None:
            return
        try:
            self.on_change(kind, values, source)
        except Exception:
            logger.exception("tractor on_change hook failed (kind=%s)", kind)

    # ----------------------------- lifecycle -----------------------------

    async def initialize(self) -> None:
        # Bring the PCA9685 up first so the parking commands below actually reach it.
        await self._pca9685.initialize()
        self._sync_hardware_state()
        ready, reason = self.actuators_ready()
        if not ready:
            logger.error(
                "Tractor actuators NOT ready (%s): lever/throttle/blade/start commands "
                "will be refused until the hardware is fixed and the backend restarted",
                reason,
            )
        # Park everything safely: levers centered, throttle idle, blade off. Each
        # is best effort and none may raise: a dead PCA9685 or relay must not stop
        # the watchdog / tilt monitor / E-stop wiring from coming up behind us.
        for actuator, field in (
            (self.left_lever, "left_lever"),
            (self.right_lever, "right_lever"),
            (self.throttle, "throttle"),
        ):
            try:
                await self._apply_servo(actuator, 0.0, field)
            except Exception as exc:
                logger.error("Tractor init: parking %s failed: %s", field, _err(exc))
        try:
            self.blade_pto.set(False)
            self.state.blade_engaged = False
        except Exception as exc:
            logger.error("Tractor init: blade PTO relay off failed: %s", _err(exc))
        self.initialized = True
        self._sync_hardware_state()

    async def shutdown(self) -> None:
        """Stop the deadman task (does not move any actuator; see ``park``)."""
        task = self._deadman_task
        self._cancel_deadman()
        if task is not None and task is not asyncio.current_task():
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ----------------------------- helpers -------------------------------

    def authorize(self) -> None:
        self.state.authorized = True

    def revoke(self) -> None:
        self.state.authorized = False

    def _latched_text(self) -> str | None:
        if not self.state.interlock_latched:
            return None
        return f"blade interlock latched ({self.state.latch_reason}): operator clear required"

    def _reject(self, reason: str) -> dict[str, Any]:
        self.state.interlock_reason = reason
        self.state.last_updated = datetime.now(UTC)
        logger.info("Tractor command rejected: %s", reason)
        return {"status": "rejected", "reason": reason}

    def _ok(self, **extra: Any) -> dict[str, Any]:
        # A latched interlock stays visible on every accepted command.
        self.state.interlock_reason = self._latched_text()
        self.state.last_updated = datetime.now(UTC)
        return {"status": "ok", **extra}

    # -- actuator health ---------------------------------------------------

    def actuators_ready(self) -> tuple[bool, str | None]:
        """``(ok, reason)``: can lever/throttle writes be expected to take effect?

        ``True`` in SIM_MODE (simulated by design) or once the PCA9685 answered
        ``initialize()``. ``False`` => callers must refuse to actuate.
        """
        if self._pca9685.is_ready():
            return True, None
        return False, self._pca9685.last_error or "PCA9685 not ready"

    def _current_fault(self) -> str | None:
        ready, reason = self.actuators_ready()
        if not ready:
            return reason
        return (
            "; ".join(self._write_faults.values())
            or self._pca9685.last_error
            or self.blade_pto.fault
            or self.starter.fault
        )

    def _sync_hardware_state(self) -> None:
        ready, _ = self.actuators_ready()
        fault = self._current_fault()
        self.state.actuator_ok = ready and fault is None
        self.state.actuator_fault = fault
        self.state.simulated = is_simulation_mode()

    def _guard_actuation(self) -> dict[str, Any] | None:
        """A rejection if the actuator hardware cannot take commands, else ``None``."""
        ready, reason = self.actuators_ready()
        if not ready:
            return self._reject(f"actuator hardware not ready: {reason}")
        return None

    # -- servo output --------------------------------------------------------

    async def _send_pwm(self, channel: int, us: int) -> None:
        """Send a single-channel PWM command via the PCA9685 I2C driver.

        SIM_MODE is a simulated success inside the driver. Anywhere else the
        driver either emits the pulse or raises: a missing chip
        (``PCA9685UnavailableError``) and a write failure on confirmed hardware
        both propagate to the caller (interlocks, ``emergency_stop``) so a real
        fault is visible instead of a false "ok". A write that hangs counts as
        a fault after ``ACTUATION_TIMEOUT_S`` rather than blocking forever.
        """
        try:
            await asyncio.wait_for(
                self._pca9685.set_pwm_us(channel, us), timeout=self.ACTUATION_TIMEOUT_S
            )
        except TimeoutError:
            self._write_faults[channel] = (
                f"PCA9685 write timed out after {self.ACTUATION_TIMEOUT_S}s (channel {channel})"
            )
            raise
        except Exception as exc:
            self._write_faults[channel] = f"PCA9685 write failed (channel {channel}): {_err(exc)}"
            raise
        self._write_faults.pop(channel, None)

    async def _apply_servo(self, actuator: ServoActuator, value: float, field: str) -> None:
        # State follows the output: the actuator/state only change after the
        # write went through, so a failed write never reads as a moved lever.
        us = actuator.microseconds(value)
        await self._send_pwm(actuator.cal.channel, us)
        actuator.command(value)
        setattr(self.state, field, actuator.value)

    # -- command deadman -------------------------------------------------------

    def _hold_s(self, duration_ms: int | None) -> float:
        """Effective hold for a command: its ``duration_ms``, clamped to the maximum."""
        if duration_ms is None or duration_ms <= 0:
            return self.default_hold_s
        return min(duration_ms / 1000.0, self.max_hold_s)

    def _arm_levers(self, hold_s: float) -> None:
        self._lever_deadline = time.monotonic() + hold_s
        self._ensure_deadman_task()

    def _arm_throttle(self, hold_s: float) -> None:
        self._throttle_deadline = time.monotonic() + hold_s
        self._ensure_deadman_task()

    def _ensure_deadman_task(self) -> None:
        if self._deadman_task is None or self._deadman_task.done():
            self._deadman_wake = asyncio.Event()
            self._deadman_task = asyncio.get_running_loop().create_task(self._deadman_loop())
        elif self._deadman_wake is not None:
            self._deadman_wake.set()  # re-evaluate: an earlier deadline may have just been armed

    def _cancel_deadman(self) -> None:
        self._lever_deadline = None
        self._throttle_deadline = None
        task = self._deadman_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _deadman_loop(self) -> None:
        try:
            while True:
                deadlines = [
                    d for d in (self._lever_deadline, self._throttle_deadline) if d is not None
                ]
                if not deadlines:
                    return  # nothing armed; re-created by the next non-neutral command
                wait = min(deadlines) - time.monotonic()
                if wait > 0:
                    wake = self._deadman_wake
                    assert wake is not None
                    wake.clear()
                    try:
                        await asyncio.wait_for(wake.wait(), timeout=wait)
                    except TimeoutError:
                        pass
                    continue
                await self._expire_due()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A dead deadman is a silent hazard: fail safe instead.
            logger.exception("Command deadman loop crashed -> emergency stop")
            await self.emergency_stop(source="deadman")

    async def _expire_due(self) -> None:
        """Return whichever of levers/throttle ran past its hold to neutral/idle."""
        failed: list[str] = []
        async with self._act_lock:
            now = time.monotonic()
            if self._lever_deadline is not None and now >= self._lever_deadline:
                self._lever_deadline = None  # act once, then escalate rather than retry
                logger.warning("Command deadman: lever command expired -> neutral")
                for actuator, field in (
                    (self.left_lever, "left_lever"),
                    (self.right_lever, "right_lever"),
                ):
                    try:
                        await self._apply_servo(actuator, 0.0, field)
                    except Exception as exc:
                        failed.append(f"{field}: {_err(exc)}")
                self._notify(
                    "levers", "deadman", left=self.state.left_lever, right=self.state.right_lever
                )
            if self._throttle_deadline is not None and now >= self._throttle_deadline:
                self._throttle_deadline = None
                logger.warning("Command deadman: throttle command expired -> idle")
                try:
                    await self._apply_servo(self.throttle, 0.0, "throttle")
                except Exception as exc:
                    failed.append(f"throttle: {_err(exc)}")
                self._notify("throttle", "deadman", throttle=self.state.throttle)
            self.state.last_updated = datetime.now(UTC)
        if failed:
            # The lock is released: the E-stop may take it.
            logger.critical("Command deadman could not neutralise %s -> emergency stop", failed)
            await self.emergency_stop(source="deadman")

    # --------------------------- actuator API ----------------------------

    async def set_levers(
        self,
        left: float,
        right: float,
        source: str = "operator",
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Command both drive levers together, for a bounded hold.

        The single chokepoint for lever commands: ``set_left_lever``,
        ``set_right_lever`` and ``apply()`` all funnel through here, so the
        e-stop guard, the actuator-health guard, the deadman and the
        reverse-entry blade auto-disengage exist in exactly one place instead
        of being duplicated per entry point.

        A non-neutral command is held for ``duration_ms`` (clamped to the
        configured maximum; default: the configured default hold), after which
        the levers return to neutral unless the command is refreshed.
        """
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        if (rejection := self._guard_actuation()) is not None:
            return rejection
        # Reverse Operation System: if this command would put both levers back
        # (reverse) while the blade is engaged, drop the blade first -- before
        # moving the levers. A single lever going negative is a pivot turn,
        # not reverse; see TractorState.reversing.
        would_reverse = left < -LEVER_NEUTRAL_EPS and right < -LEVER_NEUTRAL_EPS
        if would_reverse and self.state.blade_engaged and self.disengage_blade_in_reverse:
            logger.info("ROS: disengaging blade before entering reverse")
            await self.engage_blade(False)
        neutral = abs(left) <= LEVER_NEUTRAL_EPS and abs(right) <= LEVER_NEUTRAL_EPS
        hold_s = self._hold_s(duration_ms)
        async with self._act_lock:
            if self.state.emergency_stop_active:  # E-stop landed while we waited
                return self._reject("emergency stop active")
            if not neutral:
                # Arm BEFORE writing: if the second lever write fails the first
                # may already have moved, and the deadman must still bring it back.
                self._arm_levers(hold_s)
            await self._apply_servo(self.left_lever, left, "left_lever")
            await self._apply_servo(self.right_lever, right, "right_lever")
            if neutral:
                self._lever_deadline = None
        self._notify("levers", source, left=self.state.left_lever, right=self.state.right_lever)
        return self._ok(
            left_lever=self.state.left_lever,
            right_lever=self.state.right_lever,
            moving=self.state.moving,
            hold_ms=None if neutral else round(hold_s * 1000),
        )

    async def set_left_lever(
        self, value: float, source: str = "operator", duration_ms: int | None = None
    ) -> dict[str, Any]:
        return await self.set_levers(
            value, self.state.right_lever, source=source, duration_ms=duration_ms
        )

    async def set_right_lever(
        self, value: float, source: str = "operator", duration_ms: int | None = None
    ) -> dict[str, Any]:
        return await self.set_levers(
            self.state.left_lever, value, source=source, duration_ms=duration_ms
        )

    async def set_throttle(
        self, value: float, source: str = "operator", duration_ms: int | None = None
    ) -> dict[str, Any]:
        """Command the engine throttle for a bounded hold, then back to idle."""
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        if (rejection := self._guard_actuation()) is not None:
            return rejection
        idle = value <= 0.0
        hold_s = self._hold_s(duration_ms)
        async with self._act_lock:
            if self.state.emergency_stop_active:
                return self._reject("emergency stop active")
            if not idle:
                self._arm_throttle(hold_s)
            await self._apply_servo(self.throttle, value, "throttle")
            if idle:
                self._throttle_deadline = None
        self._notify("throttle", source, throttle=self.state.throttle)
        return self._ok(
            throttle=self.state.throttle, hold_ms=None if idle else round(hold_s * 1000)
        )

    def _set_blade_relay(self, on: bool) -> None:
        """Drive the PTO relay; ``blade_engaged`` only follows a successful write."""
        self.blade_pto.set(on)
        self.state.blade_engaged = bool(on)

    async def engage_blade(self, on: bool, source: str = "operator") -> dict[str, Any]:
        if on:
            if self.state.emergency_stop_active:
                return self._reject("emergency stop active")
            if self.state.interlock_latched:
                # Navigation re-sends blade_engaged=True every tick: this is the
                # latch that stops it re-engaging after a tilt cutoff.
                return self._reject(self._latched_text() or "blade interlock latched")
            if (rejection := self._guard_actuation()) is not None:
                return rejection
            if not self.state.authorized:
                return self._reject("motors not authorized")
            if self.require_engine_running_for_blade and not self.state.engine_running:
                return self._reject("engine must be running to engage blade")
            if self.disengage_blade_in_reverse and self.state.reversing:
                return self._reject("cannot engage blade while in reverse")
        # A relay that cannot be driven raises (RelayUnavailableError) rather
        # than reporting a switch that did not happen.
        self._set_blade_relay(on)
        self._notify("blade", source, blade=self.state.blade_engaged)
        return self._ok(blade_engaged=self.state.blade_engaged)

    async def latch_interlock(self, reason: str, source: str = "safety") -> dict[str, Any]:
        """Drop the blade and latch it off until ``clear_emergency()``.

        Used by the tilt cutoff (and any future blade interlock). Idempotent and
        cheap enough to call every monitor tick while the condition persists; it
        never raises, so a failing relay cannot take the monitor loop down -- the
        latch is software state and holds regardless, and the relay fault is
        reported in the result and on ``TractorState.actuator_fault``.
        """
        was_latched = self.state.interlock_latched
        self.state.interlock_latched = True
        self.state.latch_reason = reason
        self.state.interlock_reason = self._latched_text()
        self.state.last_updated = datetime.now(UTC)
        if not was_latched:
            logger.warning("Interlock LATCHED (%s): blade off until operator clear", reason)
        error: str | None = None
        if self.state.blade_engaged or not was_latched:
            try:
                self._set_blade_relay(False)
                self._notify("blade", source, blade=False)
            except Exception as exc:
                error = _err(exc)
                logger.critical("Interlock latch (%s): blade relay off FAILED: %s", reason, error)
        self._sync_hardware_state()
        return {
            "status": "latched" if error is None else "fault",
            "reason": reason,
            "blade_engaged": self.state.blade_engaged,
            "error": error,
        }

    async def start_engine(self) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active")
        if (rejection := self._guard_actuation()) is not None:
            return rejection
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
        try:
            await self.starter.pulse(self.starter_pulse_ms)
        except Exception as exc:
            self.state.engine = EngineState.OFF
            logger.critical("Starter relay fault: %s", _err(exc))
            return self._reject(f"starter relay fault (may still be energised): {_err(exc)}")
        # No engine-run sensor on the conversion: assume a successful crank.
        self.state.engine = EngineState.RUNNING
        return self._ok(engine=self.state.engine.value)

    async def stop_engine(self) -> dict[str, Any]:
        # Disengage blade first, then mark engine off.
        await self.engage_blade(False)
        self.state.engine = EngineState.OFF
        return self._ok(engine=self.state.engine.value)

    # ------------------------------ safing ------------------------------

    async def _acquire_for_estop(self) -> bool:
        """Wait (briefly) for an in-flight actuation so it cannot land after the stop."""
        try:
            await asyncio.wait_for(self._act_lock.acquire(), timeout=self.ESTOP_LOCK_WAIT_S)
        except TimeoutError:
            logger.warning(
                "Safing: an in-flight actuation did not finish within %.2fs; stopping anyway",
                self.ESTOP_LOCK_WAIT_S,
            )
            return False
        return True

    def _relay_off(self, relay: RelayActuator) -> dict[str, Any]:
        try:
            relay.set(False)
        except Exception as exc:
            logger.critical("Safing: relay %s off FAILED: %s", relay.name, _err(exc))
            return {"ok": False, "simulated": relay.simulated, "error": _err(exc)}
        return {"ok": True, "simulated": relay.simulated, "error": None}

    async def _servo_safe(self, actuator: ServoActuator, field: str) -> dict[str, Any]:
        try:
            await self._apply_servo(actuator, 0.0, field)
        except Exception as exc:
            logger.critical("Safing: %s to neutral/idle FAILED: %s", field, _err(exc))
            return {"ok": False, "simulated": is_simulation_mode(), "error": _err(exc)}
        return {"ok": True, "simulated": is_simulation_mode(), "error": None}

    async def _drive_to_safe_state(self) -> dict[str, dict[str, Any]]:
        """Blade/starter relays off, then levers neutral and throttle idle.

        Each subsystem is attempted independently (one bad channel must not
        abort the rest) and reported with its actual outcome. State fields only
        follow outputs that really changed: a failed channel keeps its last
        known value.
        """
        subsystems: dict[str, dict[str, Any]] = {}
        # Blade PTO first and unconditionally: pure GPIO, independent of I2C.
        subsystems["blade_pto"] = self._relay_off(self.blade_pto)
        if subsystems["blade_pto"]["ok"]:
            self.state.blade_engaged = False
        subsystems["starter"] = self._relay_off(self.starter)
        subsystems["left_lever"] = await self._servo_safe(self.left_lever, "left_lever")
        subsystems["right_lever"] = await self._servo_safe(self.right_lever, "right_lever")
        subsystems["throttle"] = await self._servo_safe(self.throttle, "throttle")  # idle engine
        return subsystems

    async def emergency_stop(self, source: str = "operator") -> dict[str, Any]:
        """Disengage the blade and center/idle the drive; leave the engine running.

        The software latch is set first and unconditionally (new commands are
        refused from this instant). Each output then gets its own try/except: a
        PCA9685 bus fault on one channel must not abort the rest of the safing
        sequence, and the result says exactly which outputs were confirmed:

        ``{"status": "emergency_stop", "complete": bool, "subsystems": {...},
        "faults": [...]}`` -- ``complete`` is ``True`` only when every output
        was driven (or simulated in SIM_MODE).
        """
        self.state.emergency_stop_active = True
        self.state.last_updated = datetime.now(UTC)
        self.revoke()
        self._cancel_deadman()
        got_lock = await self._acquire_for_estop()
        try:
            subsystems = await self._drive_to_safe_state()
        finally:
            if got_lock:
                self._act_lock.release()

        faults = [f"{name}: {info['error']}" for name, info in subsystems.items() if not info["ok"]]
        complete = not faults
        self.state.interlock_reason = "emergency_stop"
        self._notify(
            "emergency_stop",
            source,
            left=self.state.left_lever,
            right=self.state.right_lever,
            throttle=self.state.throttle,
            blade=self.state.blade_engaged,
        )
        self._sync_hardware_state()
        if complete:
            logger.warning("Tractor EMERGENCY STOP: blade off, levers neutral, throttle idle")
        else:
            logger.critical("Tractor EMERGENCY STOP INCOMPLETE -- unconfirmed outputs: %s", faults)
        return {
            "status": "emergency_stop",
            "engine": self.state.engine.value,
            "complete": complete,
            "subsystems": subsystems,
            "faults": faults,
        }

    async def park(self, source: str = "shutdown") -> dict[str, Any]:
        """Bring every actuator to its safe resting state without latching an E-stop.

        Used on shutdown: authorization is revoked, the deadman is stopped, the
        blade/starter relays go off and levers/throttle go neutral/idle, with the
        same per-subsystem reporting as ``emergency_stop``. The engine is left
        alone (there is no ignition-kill actuator).
        """
        self.revoke()
        self._cancel_deadman()
        got_lock = await self._acquire_for_estop()
        try:
            subsystems = await self._drive_to_safe_state()
        finally:
            if got_lock:
                self._act_lock.release()
        faults = [f"{name}: {info['error']}" for name, info in subsystems.items() if not info["ok"]]
        self._notify(
            "park",
            source,
            left=self.state.left_lever,
            right=self.state.right_lever,
            throttle=self.state.throttle,
            blade=self.state.blade_engaged,
        )
        self._sync_hardware_state()
        if faults:
            logger.critical("Tractor park INCOMPLETE -- unconfirmed outputs: %s", faults)
        else:
            logger.info("Tractor parked: blade off, levers neutral, throttle idle")
        return {
            "status": "parked",
            "complete": not faults,
            "subsystems": subsystems,
            "faults": faults,
        }

    async def clear_emergency(self) -> dict[str, Any]:
        """Explicit operator clear of the E-stop latch *and* any blade interlock latch.

        Does not re-authorize and does not re-engage anything: the operator must
        ``authorize()`` and command the blade again. A latch whose cause persists
        (e.g. still tilted) is re-asserted by its monitor on its next tick.
        """
        self.state.emergency_stop_active = False
        self.state.interlock_latched = False
        self.state.latch_reason = None
        self.state.interlock_reason = None
        return self._ok()

    async def apply(self, command: TractorCommand, source: str = "operator") -> dict[str, Any]:
        """Apply a full command, honoring interlocks (rejections are collected)."""
        results: dict[str, Any] = {}
        results["throttle"] = await self.set_throttle(
            command.throttle, source=source, duration_ms=command.duration_ms
        )
        results["levers"] = await self.set_levers(
            command.left_lever,
            command.right_lever,
            source=source,
            duration_ms=command.duration_ms,
        )
        # Blade last: a reverse+blade-on command surfaces as a visible
        # "rejected" here rather than engaging then immediately auto-dropping.
        results["blade"] = await self.engage_blade(command.blade_engaged, source=source)
        statuses = {r.get("status") for r in results.values()}
        if statuses == {"ok"}:
            status = "applied"
        elif "ok" in statuses:
            status = "partial"
        else:
            status = "rejected"
        return {"status": status, "results": results, "moving": self.state.moving}

    def get_state(self) -> TractorState:
        self._sync_hardware_state()
        return self.state.model_copy()


_tractor_service: TractorControlService | None = None


def get_tractor_service() -> TractorControlService:
    global _tractor_service
    if _tractor_service is None:
        _tractor_service = TractorControlService()
    return _tractor_service
