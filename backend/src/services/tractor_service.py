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
- **E-stop**: the blade PTO and starter relays open FIRST (pure GPIO: independent of the
  PCA9685 and of the actuation lock), then both levers go to neutral and the throttle to
  idle -- the engine keeps running (per configuration). A stop wins durably (audit SF-01):
  it bumps a *stop generation* that every non-neutral lever/throttle command captured before
  it began and re-checks around each write, so a command that started before the stop never
  writes after it; if an in-flight write still held the actuation lock when the stop wrote
  its neutral pulses, the stop waits (bounded) for that write to release the lock and
  re-asserts neutral after it, and reports ``complete`` only when those final writes were
  confirmed. The result reports every subsystem's *actual* outcome
  (``complete``/``subsystems``/``faults``); a latched E-stop with a failed output is never
  reported as a clean stop.
- **Authorization** (audit SAF-01): the blade, the starter and every lever or throttle command from
  an *autonomous* source (anything but ``operator``/``teleop``: a mission, the AI, the Thor
  follower, an unlabeled caller) are refused with ``motors not authorized`` while the
  tractor is not authorized. An E-stop revokes the authorization and a clear never restores it:
  after a stop nothing autonomous moves the machine until a person calls ``authorize()`` again.
  The REST ``operator`` source keeps its documented behaviour: an authenticated person's
  lever and throttle commands do not need ``authorize()`` (the blade and starter still do).
  Every stop-type event also tells ``safety.stop_events`` so a running mission or AI loop ends
  with it (a clear can never re-activate a run).
- **Interlock latch** (tilt, ...): the blade is dropped and stays off --
  ``engage_blade(True)``, including from navigation, is refused -- until an
  explicit ``clear_emergency()`` (operator-authenticated at the API). A clear never releases
  a latch whose cause may still hold: the tilt monitor vetoes the release while the latest
  tilt reading is over the limit or the tilt sensor is unavailable (audit SEC-06).
- **Park** (``park``, shutdown): safes every output like the E-stop but latches no E-stop;
  instead a park latch makes the service accept only neutral / idle / blade-off commands, so
  a mission or navigation loop that is still running cannot re-command the machine while the
  process exits and the PCA9685 keeps its last pulse (audit SF-06, SEC-08). ``initialize()``
  clears it.
- **Command deadman**: a lever/throttle command is only in force for a bounded
  hold (the command's ``duration_ms``, else the configured default, clamped to
  the configured maximum). Unless it is refreshed the levers return to neutral
  and the throttle to idle. The blade is deliberately *not* touched by the
  deadman; it follows the interlocks above.
- **Motion limit** (obstacle stop, ``safety/obstacle_stop.py``): when a ``motion_limiter``
  is attached, forward lever demand from any *autonomous* source is scaled down to the
  speed the stopping-distance model allows for the range the sensors prove free, and
  forced to neutral (blade off too) while the guard holds. This sits in ``set_levers`` /
  ``engage_blade`` so every caller reaches it -- navigation, the AI, a Thor plan driven
  through this interface. Only a caller that labels itself ``operator``/``teleop`` is a
  human in control; an unlabeled caller is treated as autonomous. It only ever reduces
  motion: neutral, reverse, blade-off and the E-stop are never limited.
- **Actuator health**: when the PCA9685 (or a relay GPIO) is absent or failed,
  commands are *refused* with the reason and the fault is exposed on
  ``TractorState`` -- never a silent no-op "ok". SIM_MODE is the exception: the
  simulated driver succeeds so the stack stays testable.
- **Actuation self-check** (``supervise_once``, run every ``supervise_interval_s`` by
  ``start_supervision``): reads the servo outputs back from the PCA9685, checks that no
  write fault is outstanding, that an armed command hold has a running deadman, and that a
  latched E-stop really reached the safe state. Only a clean cycle fires the heartbeat the
  watchdog's ``actuation`` source needs, so a healthy loop cannot vouch for a broken
  command path. A *hardware* watchdog is still required (audit O-7): this one shares the
  event loop.
- **Lock-out** (``lock_out``): when a safety function failed to start, every actuation is
  refused until the process restarts; ``clear_emergency`` cannot lift it.
- **Disabled platform** (``enabled: false``): outside SIM_MODE the service never claims or
  drives a GPIO or PWM output -- E-stop and park answer ``not_enabled`` and the relays are
  built unarmed (audit SF-05; the conservative default pending owner decision O-5).
- **Strict configuration** (``tractor_config.py``): wrong types, NaN/inf and out-of-range
  values in ``config/tractor.yaml`` are explicit errors. ``get_tractor_service()`` turns them
  into a disabled, locked-out service (``misconfigured``) so the E-stop route keeps answering
  (audit SF-11).

SIM-safe: positional commands degrade to state-tracking in SIM_MODE, and relays
only touch GPIO on real hardware.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
from ..safety import latch_store, stop_events
from ..safety.obstacle_stop import MANUAL_SOURCES, limit_pair
from .tractor_config import ServoSettings, TractorConfigError, validate_tractor_config

if TYPE_CHECKING:
    from ..safety.obstacle_stop import ObstacleAssessment, ObstacleGuard

logger = logging.getLogger(__name__)

# Command-source labels. ``operator``/``teleop`` are a human in direct control; everything
# else (including this default) is autonomous and subject to the motion limit.
DEFAULT_SOURCE = "unspecified"
OBSTACLE_SOURCE = "obstacle_stop"

# Machine-readable refusal codes (``{"status": "rejected", "code": ...}``). The *standing* ones
# hold until a person acts (clear, authorize, restart): an autonomous caller that gets one has
# lost the right to drive and must end its run instead of retrying every tick.
CODE_ESTOP = "estop"
CODE_NOT_AUTHORIZED = "not_authorized"
CODE_LOCKOUT = "lockout"
CODE_PARKED = "parked"
STANDING_REFUSALS = frozenset({CODE_ESTOP, CODE_NOT_AUTHORIZED, CODE_LOCKOUT, CODE_PARKED})


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _load_tractor_config() -> dict[str, Any]:
    """Load ``config/tractor.yaml``; ``{}`` (all defaults, platform disabled) when it is absent.

    A file that exists but cannot be parsed, or is not a mapping, raises
    ``TractorConfigError``: a mower whose owner edited the file must not silently come up
    disabled (or with half of it applied).
    """
    config_dir = os.getenv("LAWNBERRY_CONFIG_DIR", "config")
    path = os.path.join(config_dir, "tractor.yaml")
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise TractorConfigError([f"{path} cannot be read or parsed: {_err(exc)}"]) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TractorConfigError(
            [f"{path}: the top level must be a mapping, got {type(data).__name__}"]
        )
    return data


def _servo_cal(settings: ServoSettings) -> ServoCalibration:
    return ServoCalibration(
        channel=settings.channel,
        us_min=settings.us_min,
        us_center=settings.us_center,
        us_max=settings.us_max,
        bidirectional=settings.bidirectional,
    )


def _first_non_finite(*values: float | None) -> float | None:
    """The first NaN/inf among ``values`` (``None`` entries are skipped), else ``None``."""
    for value in values:
        if value is not None and not math.isfinite(value):
            return value
    return None


class _StopSuperseded(Exception):
    """A stop landed while a command was being written (internal: never reaches a caller)."""


class TractorControlService:
    """Actuation + interlock coordinator for the zero-turn mower.

    Every accepted actuation notifies an optional ``on_change`` callback
    (see ``_notify``) so the W1 capture service can log ``commands.jsonl``
    without polling; the emergency-stop path logs its safing values once,
    after the servos.
    """

    #: A single PCA9685 write must finish within this, or it counts as a fault.
    ACTUATION_TIMEOUT_S = 1.0
    #: How long an E-stop waits for an in-flight actuation to release the actuation lock
    #: before it writes neutral anyway. Short on purpose (the acceptance criteria want each
    #: safe-value command issued within 100 ms of the signal): a write that still holds the
    #: lock after this is dealt with by the re-assert (``ESTOP_REASSERT_MARGIN_S``), not by
    #: waiting for it.
    ESTOP_LOCK_WAIT_S = 0.05
    #: Beyond ``ACTUATION_TIMEOUT_S`` (the longest one in-flight write can hold the lock),
    #: how much longer a stop that had to write without the lock waits for it, to re-assert
    #: neutral after the in-flight write.
    ESTOP_REASSERT_MARGIN_S = 0.25
    #: Period of the actuation self-check whose success is the watchdog's actuation heartbeat.
    SUPERVISE_INTERVAL_S = 0.25

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else _load_tractor_config()
        # Raises TractorConfigError (every problem listed) for a wrong type, NaN/inf or an
        # out-of-range value; get_tractor_service() turns that into misconfigured().
        settings = validate_tractor_config(cfg)
        self.enabled = settings.enabled
        self._pca9685 = PCA9685Driver(settings.pca9685)

        # PCA9685 channels are 0-indexed.
        self.left_lever = ServoActuator("left_lever", _servo_cal(settings.left_lever))
        self.throttle = ServoActuator("throttle", _servo_cal(settings.throttle))
        self.right_lever = ServoActuator("right_lever", _servo_cal(settings.right_lever))

        # Relays are armed only on an enabled platform: a disabled one never claims its pins
        # (outside SIM_MODE), which may belong to something else on the header (audit SF-05).
        self.starter = RelayActuator(
            "starter", settings.starter_gpio, settings.starter_active_high, armed=self.enabled
        )
        self.blade_pto = RelayActuator(
            "blade_pto", settings.blade_gpio, settings.blade_active_high, armed=self.enabled
        )
        self.starter_pulse_ms = settings.starter_pulse_ms

        self.require_levers_neutral_to_start = settings.require_levers_neutral_to_start
        self.require_blade_off_to_start = settings.require_blade_off_to_start
        self.require_engine_running_for_blade = settings.require_engine_running_for_blade
        self.disengage_blade_in_reverse = settings.disengage_blade_in_reverse

        self.default_hold_s = settings.default_hold_s
        self.max_hold_s = settings.max_hold_s
        # Monotonic deadlines after which the levers/throttle return to neutral/idle.
        self._lever_deadline: float | None = None
        self._throttle_deadline: float | None = None
        self._deadman_task: asyncio.Task | None = None
        # Wakes the deadman watcher when a (possibly earlier) deadline is armed
        # while it sleeps towards a later one. Created with the task so it is
        # bound to the running loop.
        self._deadman_wake: asyncio.Event | None = None
        # Serialises lever/throttle writes (commands vs. deadman expiry). A stop does not
        # queue behind it: see _drive_to_safe_state().
        self._act_lock = asyncio.Lock()
        # Bumped by every stop, park and lock-out. A lever/throttle command captures it before
        # it waits for the actuation lock and each non-neutral write re-checks it, so a command
        # that began before a stop can never put its pulse on the chip after the stop's neutral
        # (audit SF-01).
        self._stop_generation = 0
        # emergency_stop() calls currently running: a clear is refused while one is, so a clear
        # cannot undo a stop that is still driving the outputs (audit SEC-12).
        self._stops_in_flight = 0
        # park() latch: only neutral / idle / blade-off commands pass until initialize().
        self._parked: str | None = None
        # Optional veto on releasing the blade interlock latch; the tilt monitor attaches one
        # (audit SEC-06). Called with the latch reason, returns why it must stay, else None.
        self._interlock_clear_gate: Callable[[str | None], str | None] | None = None
        # Set by misconfigured(): the validation error that keeps this service disabled.
        self._config_error: str | None = None
        # Last servo write fault per PCA9685 channel (incl. timeouts), each
        # cleared by that channel's next successful write.
        self._write_faults: dict[int, str] = {}

        # Safe initial state: both levers neutral, blade off, engine off.
        self.state = TractorState(enabled=self.enabled)
        # Capture-stream hook; called on_change(kind, values, source) after every
        # accepted actuation. None when nothing subscribed.
        self.on_change: Callable[[str, dict[str, Any], str], None] | None = None
        self.initialized = False
        # Obstacle stop (safety/obstacle_stop.py): attached by the lifespan. None = no limit
        # (bench/unit configuration); main.py never starts the tractor without it.
        self.motion_limiter: ObstacleGuard | None = None
        # Label of the last accepted lever / blade-on command: lets the hold handler act only
        # on motion an autonomous source has in force, never on a human's manual drive.
        self._lever_source: str | None = None
        self._blade_source: str | None = None
        # Actuation self-check (watchdog heartbeat). ``config/tractor.yaml`` may set
        # ``actuation_supervision.interval_ms``; the default is far under the watchdog window.
        self.supervise_interval_s = (
            self.SUPERVISE_INTERVAL_S
            if settings.supervise_interval_s is None
            else settings.supervise_interval_s
        )
        self._heartbeat: Callable[[], None] | None = None
        self._supervise_task: asyncio.Task | None = None
        self._supervision_fault: str | None = None
        # Set by lock_out(): every actuation is refused until the process restarts.
        self._lockout: str | None = None
        self._sync_hardware_state()

    @classmethod
    def misconfigured(cls, error: BaseException | str) -> TractorControlService:
        """The service for a ``config/tractor.yaml`` that failed validation (audit SF-11).

        Disabled (all defaults) and locked out until the process restarts: every actuation is
        refused with the reason, the E-stop route keeps answering (``not_enabled``, naming the
        error) instead of returning 500, and no GPIO or PWM output is claimed or driven,
        because a pin read from a file that cannot be trusted must not be driven. The error is
        also on ``TractorState.actuator_fault``.
        """
        service = cls(config={})
        service._config_error = str(error)
        service.lock_out(f"invalid config/tractor.yaml: {error}")
        return service

    @property
    def config_error(self) -> str | None:
        """The ``config/tractor.yaml`` validation error that disabled this service, else None."""
        return self._config_error

    @property
    def lockout_reason(self) -> str | None:
        """Why every actuation is refused until restart (``lock_out``), else ``None``."""
        return self._lockout

    @property
    def parked(self) -> bool:
        """Is the park latch set (only neutral / idle / blade-off commands are accepted)?"""
        return self._parked is not None

    @property
    def stop_in_flight(self) -> bool:
        """Is an ``emergency_stop`` still driving the outputs? (A clear is refused meanwhile.)"""
        return self._stops_in_flight > 0

    def attach_interlock_clear_gate(self, gate: Callable[[str | None], str | None] | None) -> None:
        """Set the veto ``clear_emergency`` consults before releasing the blade interlock latch.

        ``gate(latch_reason)`` returns the reason the latch must stay (or ``None``). The tilt
        monitor attaches one while it runs; ``None`` detaches. A gate that raises keeps the
        latch (fail closed).
        """
        self._interlock_clear_gate = gate

    def detach_interlock_clear_gate(self, gate: Callable[[str | None], str | None]) -> None:
        """Remove ``gate`` if it is the one attached (a stopped monitor must not veto for ever)."""
        if self._interlock_clear_gate == gate:
            self._interlock_clear_gate = None

    def _hardware_off_limits(self) -> bool:
        """True when this service must not touch real hardware at all (disabled, not SIM)."""
        return not self.enabled and not is_simulation_mode()

    def platform_refusal(self) -> str | None:
        """Why the HTTP write routes must not drive this service at all, else ``None``.

        Set for a platform that is disabled (or misconfigured) outside SIM_MODE: its pins are
        not ours to claim (audit SF-05).
        """
        return self.not_enabled_text() if self._hardware_off_limits() else None

    def not_enabled_text(self) -> str:
        """What a stop/park report says when it leaves the hardware alone (``not_enabled``)."""
        if self._config_error is not None:
            return (
                f"tractor config invalid ({self._config_error}): no GPIO or PWM output was "
                "claimed or driven"
            )
        return (
            "tractor platform not enabled in config/tractor.yaml: no GPIO or PWM output was "
            "claimed or driven"
        )

    def _not_enabled_report(self) -> dict[str, Any]:
        text = self.not_enabled_text()
        logger.warning("Tractor stop/park skipped: %s", text)
        return {
            "status": "not_enabled",
            "engine": self.state.engine.value,
            "complete": False,
            "subsystems": {},
            "faults": [text],
            "detail": text,
        }

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
        # A fresh bring-up re-arms a service that was parked (the shutdown latch).
        self._parked = None
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
        """Stop the deadman and supervision tasks (moves no actuator; see ``park``)."""
        await self.stop_supervision()
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

    async def withdraw_authorization(self, source: str = "operator") -> dict[str, Any]:
        """An operator's revoke: authorization goes, and so does what depended on it.

        ``revoke()`` alone only flips the flag (the E-stop, the park and the lock-out call it
        and drive the outputs themselves). A revoke from the operator must also stand the
        machine down: the blade (which needs the authorization to run) is switched off first,
        then any lever motion an autonomous source has in force goes neutral, and every
        mission or AI run ends (``stop_events``): without authorization none of them could
        command a thing anyway, and one that kept spinning would drive off the moment someone
        authorized again. A person's own manual drive is left alone. Never raises.
        """
        self.revoke()
        stop_events.record_stop("authorization revoked", source)
        report: dict[str, Any] = {"status": "revoked", "errors": []}
        if self._hardware_off_limits():
            return report
        if self.state.blade_engaged:
            try:
                self._set_blade_relay(False)
                self._notify("blade", source, blade=False)
            except Exception as exc:
                logger.critical("Revoke: blade relay off FAILED: %s", _err(exc))
                report["errors"].append(f"blade off: {_err(exc)}")
        levers_out = (
            abs(self.state.left_lever) > LEVER_NEUTRAL_EPS
            or abs(self.state.right_lever) > LEVER_NEUTRAL_EPS
        )
        if levers_out and self._lever_source not in MANUAL_SOURCES:
            try:
                result = await self.set_levers(0.0, 0.0, source=source)
                if result.get("status") != "ok":
                    report["errors"].append(f"levers neutral: {result.get('reason')}")
            except Exception as exc:
                report["errors"].append(f"levers neutral: {_err(exc)}")
        report["blade_engaged"] = self.state.blade_engaged
        report["left_lever"] = self.state.left_lever
        report["right_lever"] = self.state.right_lever
        return report

    def restore_latches(self, *, estop: bool, interlock: str | None, why: str = "") -> None:
        """Re-latch what a previous process left latched (audit SAF-16); never authorizes.

        Called once at start-up by ``safety.latch_store.restore``, before ``initialize()``
        parks the outputs. A disabled platform has no latch to hold. Only an authenticated
        ``clear_emergency()`` releases these.
        """
        if not self.enabled:
            return
        self.revoke()
        self._stop_generation += 1
        if estop:
            self.state.emergency_stop_active = True
            self.state.interlock_reason = "emergency_stop"
        if interlock is not None:
            self.state.interlock_latched = True
            self.state.latch_reason = interlock
            if not estop:
                self.state.interlock_reason = self._latched_text()
        self.state.last_updated = datetime.now(UTC)
        logger.critical(
            "Tractor latch RESTORED after a restart (%s): E-stop=%s, blade interlock=%s",
            why or "persisted latch",
            estop,
            interlock,
        )
        self._sync_hardware_state()

    def _latched_text(self) -> str | None:
        if not self.state.interlock_latched:
            return None
        return f"blade interlock latched ({self.state.latch_reason}): operator clear required"

    def _parked_text(self) -> str | None:
        if self._parked is None:
            return None
        return f"parked ({self._parked}): only neutral/idle/blade-off commands until initialize()"

    def _standing_reason(self) -> str | None:
        """What stays visible on every accepted command: the park latch, else the interlock."""
        return self._parked_text() or self._latched_text()

    def _reject(self, reason: str, code: str | None = None) -> dict[str, Any]:
        self.state.interlock_reason = reason
        self.state.last_updated = datetime.now(UTC)
        logger.info("Tractor command rejected: %s", reason)
        if code is None:
            return {"status": "rejected", "reason": reason}
        return {"status": "rejected", "reason": reason, "code": code}

    def _ok(self, **extra: Any) -> dict[str, Any]:
        # A latched interlock (or the park latch) stays visible on every accepted command.
        self.state.interlock_reason = self._standing_reason()
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

    def _hardware_fault(self) -> str | None:
        """An outstanding actuator fault (what the hardware itself reported), else ``None``."""
        ready, reason = self.actuators_ready()
        if not ready:
            return reason
        return (
            "; ".join(self._write_faults.values())
            or self._pca9685.last_error
            or self.blade_pto.fault
            or self.starter.fault
        )

    def _current_fault(self) -> str | None:
        # The self-check's verdict is reported too, but it is deliberately not an input to
        # the self-check itself (that would latch a stale verdict). A configuration error is
        # the root cause of everything else a misconfigured service reports, so it comes first.
        return self._config_error or self._hardware_fault() or self._supervision_fault

    def _sync_hardware_state(self) -> None:
        ready, _ = self.actuators_ready()
        fault = self._current_fault()
        self.state.actuator_ok = ready and fault is None
        self.state.actuator_fault = fault
        self.state.simulated = is_simulation_mode()

    def _guard_actuation(self) -> dict[str, Any] | None:
        """A rejection if the actuator hardware cannot take commands, else ``None``."""
        if self._lockout is not None:
            return self._reject(f"locked out: {self._lockout}", CODE_LOCKOUT)
        ready, reason = self.actuators_ready()
        if not ready:
            return self._reject(f"actuator hardware not ready: {reason}")
        return None

    def lock_out(self, reason: str) -> None:
        """Refuse every actuation until the process restarts: a safety function did not start.

        Used when the watchdog or tilt monitor could not be brought up (audit N-05), so the
        actuation routes fail closed instead of staying live with nothing supervising them.
        Unlike the E-stop latch no API call clears this: ``clear_emergency`` refuses. It does
        not move an actuator; the caller drives the safe state (``emergency_stop``).
        """
        self._lockout = reason
        self._stop_generation += 1  # an in-flight command must not land after a lock-out
        self.revoke()
        self.state.interlock_reason = f"locked out: {reason}"
        self.state.last_updated = datetime.now(UTC)
        logger.critical("Tractor LOCKED OUT until restart: %s", reason)
        stop_events.record_stop("lockout", "lock_out")

    # -- actuation self-check (the watchdog's heartbeat) ------------------------

    def attach_heartbeat(self, beat: Callable[[], None] | None) -> None:
        """Set the callback ``supervise_once`` fires, only after a clean self-check.

        main.py passes ``lambda: watchdog.heartbeat(ACTUATION_SOURCE)``. ``None`` detaches.
        """
        self._heartbeat = beat

    async def start_supervision(self) -> None:
        """Start the periodic actuator-control loop (idempotent)."""
        if self._supervise_task is None or self._supervise_task.done():
            self._supervise_task = asyncio.get_running_loop().create_task(self._supervise_loop())

    async def stop_supervision(self) -> None:
        task, self._supervise_task = self._supervise_task, None
        if task is None or task.done() or task is asyncio.current_task():
            return
        # Re-cancel instead of waiting for one cancel to land: a cancellation that arrives
        # while the loop is inside a nested await can be absorbed, and ``await task`` would
        # then wait for a loop that never ends.
        for _ in range(20):
            task.cancel()
            done, _pending = await asyncio.wait({task}, timeout=0.25)
            if done:
                break
        else:
            logger.error("Actuation supervision task did not stop after 20 cancellations")
            return
        if not task.cancelled():
            task.exception()  # retrieved, so a late failure is not reported as never-awaited

    async def _supervise_loop(self) -> None:
        while True:
            # supervise_once swallows everything but cancellation: a failed cycle is a cycle
            # without a heartbeat, and the loop keeps going so the next one can recover.
            await self.supervise_once()
            await asyncio.sleep(self.supervise_interval_s)

    async def supervise_once(self) -> str | None:
        """One actuator-control cycle: check the outputs and, only if they check out, beat.

        The heartbeat is therefore evidence about the actuation path, not about a loop that
        turns: it is withheld while the PCA9685 is not ready or a write fault is outstanding,
        while a command hold is armed but the deadman is not running or is overdue, while a
        latched E-stop (or blade interlock) has not actually reached the safe state, and
        while the servo outputs read back from the PCA9685 differ from what was commanded
        (a reset chip, a dead read path, a hung bus). Returns ``None`` when it beat, else
        the reason it did not.
        """
        try:
            reason = await self._verify_actuation()
            if reason is None and self._heartbeat is not None:
                self._heartbeat()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a self-check (or a heartbeat sink) that crashes has not passed
            reason = f"actuation self-check crashed: {_err(exc)}"
        previous, self._supervision_fault = self._supervision_fault, reason
        if reason is None:
            if previous is not None:
                logger.warning("Actuation self-check recovered: heartbeat resumed")
        elif reason != previous:
            logger.error("Actuation self-check FAILED (no heartbeat): %s", reason)
        self._sync_hardware_state()
        return reason

    async def _verify_actuation(self) -> str | None:
        fault = self._hardware_fault()
        if fault:
            return f"actuator fault outstanding: {fault}"
        return self._deadman_fault() or self._safe_state_fault() or await self._read_back_outputs()

    def _deadman_fault(self) -> str | None:
        """A command hold that nothing is going to expire (the machine would keep driving)."""
        armed = [d for d in (self._lever_deadline, self._throttle_deadline) if d is not None]
        if not armed:
            return None
        task = self._deadman_task
        if task is None or task.done():
            return "a command hold is armed but the deadman task is not running"
        # A normal expiry waits for the actuation lock, which a write may hold for a while.
        grace = self.ACTUATION_TIMEOUT_S + self.supervise_interval_s
        late = time.monotonic() - min(armed)
        if late > grace:
            return f"command deadman is {late:.2f}s overdue (grace {grace:.2f}s)"
        return None

    def _safe_state_fault(self) -> str | None:
        """A latched stop whose outputs are not at the safe state (an unconfirmed E-stop)."""
        s = self.state
        if s.emergency_stop_active:
            unsafe = [
                name
                for name, value in (
                    ("left_lever", s.left_lever),
                    ("right_lever", s.right_lever),
                    ("throttle", s.throttle),
                )
                if value != 0.0
            ]
            if s.blade_engaged:
                unsafe.append("blade")
            if unsafe:
                return f"E-stop is latched but {', '.join(unsafe)} not at the safe state"
        elif s.interlock_latched and s.blade_engaged:
            return "blade interlock is latched but the blade is engaged"
        return None

    async def _read_back_outputs(self) -> str | None:
        """Read every servo channel back; ``None`` when each holds what was commanded."""
        try:
            async with asyncio.timeout(self.supervise_interval_s):
                await self._act_lock.acquire()
        except TimeoutError:
            return f"actuation lock held for more than {self.supervise_interval_s:.2f}s"
        try:
            # Under the lock so a command cannot change the expected value between reading
            # the registers and comparing them.
            async with asyncio.timeout(self.ACTUATION_TIMEOUT_S):
                await self._verify_servo_outputs()
        except TimeoutError:
            return f"PCA9685 read-back timed out after {self.ACTUATION_TIMEOUT_S}s"
        except Exception as exc:
            return f"PCA9685 read-back failed: {_err(exc)}"
        finally:
            self._act_lock.release()
        return None

    async def _verify_servo_outputs(self) -> None:
        for actuator in (self.left_lever, self.right_lever, self.throttle):
            await self._pca9685.verify_pwm_us(
                actuator.cal.channel, actuator.microseconds(actuator.value)
            )

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

    async def _apply_servo(
        self, actuator: ServoActuator, value: float, field: str, gen: int | None = None
    ) -> None:
        """Write one servo and record it; state follows the output.

        ``gen`` is the stop generation a lever/throttle *command* captured before it began (a
        stop's own neutral writes and the deadman pass none). A non-neutral write is refused
        before it starts when a stop has landed since, and, because an I2C write cannot be
        cancelled once it is on the bus, a write that was already in flight when the stop
        landed is recorded (it IS the output now) and then raises ``_StopSuperseded`` so the
        command writes nothing further; the stop re-asserts neutral once this write releases
        the actuation lock (audit SF-01).
        """
        guarded = gen is not None and value != 0.0
        if guarded and gen != self._stop_generation:
            raise _StopSuperseded(field)
        # The actuator/state only change after the write went through, so a failed write
        # never reads as a moved lever.
        us = actuator.microseconds(value)
        await self._send_pwm(actuator.cal.channel, us)
        actuator.command(value)
        setattr(self.state, field, actuator.value)
        if guarded and gen != self._stop_generation:
            raise _StopSuperseded(field)

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
        left: float | None,
        right: float | None,
        source: str = DEFAULT_SOURCE,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Command both drive levers together, for a bounded hold.

        The single chokepoint for lever commands: ``set_left_lever``,
        ``set_right_lever`` and ``apply()`` all funnel through here, so the
        e-stop guard, the actuator-health guard, the motion limit, the deadman and
        the reverse-entry blade auto-disengage exist in exactly one place instead
        of being duplicated per entry point.

        ``None`` for a lever means "leave it where it is": its *current* value is read under
        the actuation lock, so two concurrent single-lever requests cannot overwrite each other
        with stale readings (one request's write would otherwise undo the other's).

        A non-neutral command is held for ``duration_ms`` (clamped to the
        configured maximum; default: the configured default hold), after which
        the levers return to neutral unless the command is refreshed.

        ``source`` labels who is asking. ``operator``/``teleop`` are a human in
        direct control; any other label (including the default) is autonomous and,
        while a ``motion_limiter`` is attached, has its forward demand scaled down to
        the limiter's cap -- to neutral during a hold. The reply says so
        (``limited``/``speed_cap_mps``/``limit_reason``); neutral, reverse and the
        E-stop are never limited.

        A non-finite lever value (NaN/inf) is refused: it would clamp to full scale. A
        command that started before a stop never writes after it (``_stop_generation``), and
        once the service is parked only neutral commands are accepted.
        """
        if (bad := _first_non_finite(left, right)) is not None:
            return self._reject(f"non-finite lever command {bad!r} refused")
        gen = self._stop_generation  # captured before anything can wait
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active", CODE_ESTOP)
        if (rejection := self._guard_actuation()) is not None:
            return rejection
        hold_s = self._hold_s(duration_ms)
        async with self._act_lock:
            if gen != self._stop_generation or self.state.emergency_stop_active:
                # a stop landed while we waited
                return self._reject("emergency stop active", CODE_ESTOP)
            left_v = self.state.left_lever if left is None else left
            right_v = self.state.right_lever if right is None else right
            left_v, right_v, limit = self._limit_levers(left_v, right_v, source)
            neutral = abs(left_v) <= LEVER_NEUTRAL_EPS and abs(right_v) <= LEVER_NEUTRAL_EPS
            if self._parked is not None and not neutral:
                return self._reject(self._parked_text() or "parked", CODE_PARKED)
            if not neutral and source not in MANUAL_SOURCES and not self.state.authorized:
                # A person's own lever commands do not need authorize() (a hold-to-run
                # joystick); a mission, the AI or the Thor follower must be authorized, and an
                # E-stop, a revoke or a restart takes that away until a person gives it back.
                return self._reject("motors not authorized", CODE_NOT_AUTHORIZED)
            # Reverse Operation System: if this command would put both levers back
            # (reverse) while the blade is engaged, drop the blade first -- before
            # moving the levers. A single lever going negative is a pivot turn,
            # not reverse; see TractorState.reversing.
            would_reverse = left_v < -LEVER_NEUTRAL_EPS and right_v < -LEVER_NEUTRAL_EPS
            if would_reverse and self.state.blade_engaged and self.disengage_blade_in_reverse:
                logger.info("ROS: disengaging blade before entering reverse")
                await self.engage_blade(False, source=source)
            if not neutral:
                # Arm BEFORE writing: if the second lever write fails the first
                # may already have moved, and the deadman must still bring it back.
                self._arm_levers(hold_s)
            try:
                await self._apply_servo(self.left_lever, left_v, "left_lever", gen)
                await self._apply_servo(self.right_lever, right_v, "right_lever", gen)
            except _StopSuperseded:
                return self._reject("emergency stop landed during the command")
            if neutral:
                self._lever_deadline = None
        self._lever_source = source
        self._notify("levers", source, left=self.state.left_lever, right=self.state.right_lever)
        return self._ok(
            left_lever=self.state.left_lever,
            right_lever=self.state.right_lever,
            moving=self.state.moving,
            hold_ms=None if neutral else round(hold_s * 1000),
            limited=bool(limit and limit["limited"]),
            speed_cap_mps=None if limit is None else limit["speed_cap_mps"],
            limit_reason=None if limit is None else limit["reason"],
        )

    async def set_left_lever(
        self, value: float, source: str = DEFAULT_SOURCE, duration_ms: int | None = None
    ) -> dict[str, Any]:
        return await self.set_levers(value, None, source=source, duration_ms=duration_ms)

    async def set_right_lever(
        self, value: float, source: str = DEFAULT_SOURCE, duration_ms: int | None = None
    ) -> dict[str, Any]:
        return await self.set_levers(None, value, source=source, duration_ms=duration_ms)

    async def set_throttle(
        self, value: float, source: str = DEFAULT_SOURCE, duration_ms: int | None = None
    ) -> dict[str, Any]:
        """Command the engine throttle for a bounded hold, then back to idle.

        Same stop rules as ``set_levers``: NaN/inf is refused, a command that began before a
        stop never writes after it, and a parked service accepts only idle.
        """
        if (bad := _first_non_finite(value)) is not None:
            return self._reject(f"non-finite throttle command {bad!r} refused")
        gen = self._stop_generation
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active", CODE_ESTOP)
        if (rejection := self._guard_actuation()) is not None:
            return rejection
        idle = value <= 0.0
        hold_s = self._hold_s(duration_ms)
        async with self._act_lock:
            if gen != self._stop_generation or self.state.emergency_stop_active:
                return self._reject("emergency stop active", CODE_ESTOP)
            if self._parked is not None and not idle:
                return self._reject(self._parked_text() or "parked", CODE_PARKED)
            if not idle and source not in MANUAL_SOURCES and not self.state.authorized:
                return self._reject("motors not authorized", CODE_NOT_AUTHORIZED)
            if not idle:
                self._arm_throttle(hold_s)
            try:
                await self._apply_servo(self.throttle, value, "throttle", gen)
            except _StopSuperseded:
                return self._reject("emergency stop landed during the command")
            if idle:
                self._throttle_deadline = None
        self._notify("throttle", source, throttle=self.state.throttle)
        return self._ok(
            throttle=self.state.throttle, hold_ms=None if idle else round(hold_s * 1000)
        )

    # -- motion limit (obstacle stop) ------------------------------------------

    def _assess_limit(self, source: str | None) -> tuple[float, str | None, float | None] | None:
        """``(lever_cap, reason, speed_cap_mps)`` for ``source``, or None if not limited.

        A limiter that raises is a hold, never a free pass: the machine is not allowed to
        move on the strength of a safety check that could not be evaluated.
        """
        limiter = self.motion_limiter
        if limiter is None or not limiter.applies_to(source):
            return None
        try:
            a = limiter.assess()
        except Exception as exc:
            logger.exception("Motion limiter failed: holding autonomous motion")
            return 0.0, f"motion limiter failed: {exc!r}", 0.0
        reason = "; ".join(a.reasons) if a.reasons else None
        return a.lever_cap, reason, a.speed_cap_mps

    def _limit_levers(
        self, left: float, right: float, source: str
    ) -> tuple[float, float, dict[str, Any] | None]:
        """Scale a forward lever demand to the motion limit (``None`` info = not limited)."""
        assessed = self._assess_limit(source)
        if assessed is None:
            return left, right, None
        lever_cap, reason, speed_cap = assessed
        new_left, new_right = limit_pair(left, right, lever_cap)
        limited = (new_left, new_right) != (left, right)
        if limited:
            logger.debug(
                "Motion limit (%s): levers (%.2f, %.2f) -> (%.2f, %.2f), cap %.2f: %s",
                source,
                left,
                right,
                new_left,
                new_right,
                lever_cap,
                reason,
            )
        info = {
            "limited": limited,
            "speed_cap_mps": speed_cap,
            "reason": reason if limited else None,
        }
        return new_left, new_right, info

    def _motion_hold(self, source: str) -> str | None:
        """Why a command from ``source`` is held right now, or None (the blade path)."""
        assessed = self._assess_limit(source)
        if assessed is None or assessed[0] > 0.0:
            return None
        return assessed[1] or "motion held"

    def _sync_limit_state(self) -> None:
        """Publish the limiter's current verdict on ``TractorState`` (visible at /tractor/state)."""
        limiter = self.motion_limiter
        if limiter is None:
            self.state.speed_cap_mps = None
            self.state.motion_limited = False
            self.state.motion_limit_reason = None
            return
        try:
            a = limiter.assess()
        except Exception as exc:
            self.state.speed_cap_mps = 0.0
            self.state.motion_limited = True
            self.state.motion_limit_reason = f"motion limiter failed: {exc!r}"
            return
        self.state.speed_cap_mps = a.speed_cap_mps
        self.state.motion_limited = a.status.value != "clear"
        self.state.motion_limit_reason = "; ".join(a.reasons) if self.state.motion_limited else None

    async def enforce_motion_hold(self, assessment: ObstacleAssessment | None = None) -> None:
        """Stop the motion an autonomous source has in force: blade off, levers neutral.

        The obstacle monitor's hold handler. It acts the moment a hold begins instead of
        waiting for the planner's next command, and only on motion that came from a limited
        (autonomous) source: a human's manual drive is left alone. Blade first (pure GPIO,
        independent of the I2C bus), then the levers; both are always attempted, and the
        first failure is raised afterwards so the monitor reports it and retries next cycle.
        """
        limiter = self.motion_limiter
        if limiter is None:
            return
        errors: list[str] = []
        if self.state.blade_engaged and (
            self._blade_source is None or limiter.applies_to(self._blade_source)
        ):
            try:
                await self.engage_blade(False, source=OBSTACLE_SOURCE)
            except Exception as exc:
                errors.append(f"blade off: {_err(exc)}")
        levers_out = (
            abs(self.state.left_lever) > LEVER_NEUTRAL_EPS
            or abs(self.state.right_lever) > LEVER_NEUTRAL_EPS
        )
        if levers_out and (self._lever_source is None or limiter.applies_to(self._lever_source)):
            try:
                await self.set_levers(0.0, 0.0, source=OBSTACLE_SOURCE)
            except Exception as exc:
                errors.append(f"levers neutral: {_err(exc)}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def _set_blade_relay(self, on: bool) -> None:
        """Drive the PTO relay; ``blade_engaged`` only follows a successful write."""
        self.blade_pto.set(on)
        self.state.blade_engaged = bool(on)

    async def engage_blade(self, on: bool, source: str = DEFAULT_SOURCE) -> dict[str, Any]:
        if on:
            if self.state.emergency_stop_active:
                return self._reject("emergency stop active", CODE_ESTOP)
            if self._parked is not None:
                return self._reject(self._parked_text() or "parked", CODE_PARKED)
            if self.state.interlock_latched:
                # Navigation re-sends blade_engaged=True every tick: this is the
                # latch that stops it re-engaging after a tilt cutoff.
                return self._reject(self._latched_text() or "blade interlock latched")
            if (hold := self._motion_hold(source)) is not None:
                # Refuse AND drop: an obstacle can appear while the blade already turns.
                return self._refuse_blade(f"obstacle hold: {hold}", source)
            if (rejection := self._guard_actuation()) is not None:
                return rejection
            if not self.state.authorized:
                # Refuse AND drop, like a hold: no authorization means no blade, so a blade
                # that is already turning when the authorization went away is switched off.
                return self._refuse_blade("motors not authorized", source, CODE_NOT_AUTHORIZED)
            if self.require_engine_running_for_blade and not self.state.engine_running:
                return self._reject("engine must be running to engage blade")
            if self.disengage_blade_in_reverse and self.state.reversing:
                return self._reject("cannot engage blade while in reverse")
        # A relay that cannot be driven raises (RelayUnavailableError) rather
        # than reporting a switch that did not happen.
        self._set_blade_relay(on)
        if on:
            self._blade_source = source
        self._notify("blade", source, blade=self.state.blade_engaged)
        return self._ok(blade_engaged=self.state.blade_engaged)

    def _refuse_blade(self, reason: str, source: str, code: str | None = None) -> dict[str, Any]:
        """Reject a blade-on request and make sure the blade is off (never raises)."""
        if self.state.blade_engaged:
            try:
                self._set_blade_relay(False)
                self._notify("blade", source, blade=False)
            except Exception as exc:
                logger.critical("Blade refusal (%s): relay off FAILED: %s", reason, _err(exc))
                return self._reject(f"{reason}; blade relay off FAILED: {_err(exc)}", code)
        return self._reject(reason, code)

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
        persisted: bool | None = None
        if not was_latched:
            # A new latch ends every autonomous run (a mission that is still sending
            # blade-on would re-engage it the moment an operator clears the latch) and is
            # saved so that a restart does not release it (audit SAF-01, SAF-16).
            stop_events.record_stop(f"interlock:{reason}", source)
            persisted = (await latch_store.persist(tractor=self)).ok
        return {
            "status": "latched" if error is None else "fault",
            "reason": reason,
            "blade_engaged": self.state.blade_engaged,
            "error": error,
            "latch_persisted": persisted,
        }

    async def start_engine(self) -> dict[str, Any]:
        if self.state.emergency_stop_active:
            return self._reject("emergency stop active", CODE_ESTOP)
        if self._parked is not None:
            return self._reject(self._parked_text() or "parked", CODE_PARKED)
        if (rejection := self._guard_actuation()) is not None:
            return rejection
        if not self.state.authorized:
            return self._reject("motors not authorized", CODE_NOT_AUTHORIZED)
        if self.require_blade_off_to_start and self.state.blade_engaged:
            return self._reject("blade/PTO must be disengaged to start")
        if self.require_levers_neutral_to_start and (
            abs(self.state.left_lever) > LEVER_NEUTRAL_EPS
            or abs(self.state.right_lever) > LEVER_NEUTRAL_EPS
        ):
            return self._reject("drive levers must be neutral to start")

        gen = self._stop_generation
        self.state.engine = EngineState.STARTING
        try:
            await self.starter.pulse(self.starter_pulse_ms)
        except Exception as exc:
            self.state.engine = EngineState.OFF
            logger.critical("Starter relay fault: %s", _err(exc))
            return self._reject(f"starter relay fault (may still be energised): {_err(exc)}")
        if gen != self._stop_generation:
            # A stop opened the starter relay mid-crank (audit SF-09): nothing observed this
            # engine run, so it is not "running". Left OFF, the blade interlock keeps refusing
            # until a start that was not interrupted completes.
            self.state.engine = EngineState.OFF
            return self._reject(
                "stop during the start: the crank was cut short, engine not running"
            )
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
        """Take the actuation lock if an in-flight actuation releases it within a moment.

        A stop never waits longer than ``ESTOP_LOCK_WAIT_S``: its neutral commands must reach
        the bus within the 100 ms acceptance limit even while a stalled write holds the lock.
        Returns ``False`` when it could not; the caller then writes without the lock and
        re-asserts neutral once the lock frees (``_reassert_neutral``).
        """
        try:
            async with asyncio.timeout(self.ESTOP_LOCK_WAIT_S):
                await self._act_lock.acquire()
        except TimeoutError:
            logger.warning(
                "Safing: an in-flight actuation still held the actuation lock after %.2fs; "
                "writing neutral without it and re-asserting after it",
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

    _SERVO_FIELDS = ("left_lever", "right_lever", "throttle")

    async def _servos_to_neutral(self) -> dict[str, dict[str, Any]]:
        """One pass: every servo channel to neutral/idle, issued together. Never raises.

        The three writes are issued concurrently so the last channel does not wait for the
        others (a slow bus would otherwise push the throttle past the 100 ms limit).
        """
        servos = (self.left_lever, self.right_lever, self.throttle)
        results = await asyncio.gather(
            *(self._servo_safe(a, f) for a, f in zip(servos, self._SERVO_FIELDS, strict=True))
        )
        return dict(zip(self._SERVO_FIELDS, results, strict=True))

    async def _reassert_neutral(self) -> dict[str, dict[str, Any]]:
        """Neutral once more, under the lock, after the write that was in flight released it.

        The stop wrote neutral without the lock because a command's write was still in
        flight; that write cannot be cancelled and may reach the chip after the stop's neutral.
        Waiting (bounded: a write holds the lock at most ``ACTUATION_TIMEOUT_S``) and writing
        neutral again makes the stop's neutral the last word on every channel. If the lock
        never frees, nothing is confirmed: every servo is reported failed so ``complete`` stays
        False, and the supervision loop's read-back keeps the watchdog repeating the stop.
        """
        wait_s = self.ACTUATION_TIMEOUT_S + self.ESTOP_REASSERT_MARGIN_S
        try:
            async with asyncio.timeout(wait_s):
                await self._act_lock.acquire()
        except TimeoutError:
            error = (
                f"an in-flight actuation still held the actuation lock after {wait_s:.2f}s: "
                "neutral could not be re-asserted after it"
            )
            logger.critical("Safing: %s", error)
            return {
                field: {"ok": False, "simulated": is_simulation_mode(), "error": error}
                for field in self._SERVO_FIELDS
            }
        try:
            return await self._servos_to_neutral()
        finally:
            self._act_lock.release()

    async def _drive_to_safe_state(self) -> tuple[dict[str, dict[str, Any]], bool]:
        """Blade/starter relays off, then levers neutral and throttle idle.

        Each subsystem is attempted independently (one bad channel must not
        abort the rest) and reported with its actual outcome. State fields only
        follow outputs that really changed: a failed channel keeps its last
        known value.

        The relays go first, in the same event-loop turn as the call and before anything can
        wait: they are pure GPIO and need neither the PCA9685 nor the actuation lock (audit
        SF-04: waiting for the lock first delayed the PTO cut by 250 ms). The servos follow;
        if the lock was busy the neutral pulses go out without it and are re-asserted once it
        frees. Returns ``(subsystems, contended)``; ``contended`` says an in-flight write held
        the lock past the stop.
        """
        subsystems: dict[str, dict[str, Any]] = {}
        subsystems["blade_pto"] = self._relay_off(self.blade_pto)
        if subsystems["blade_pto"]["ok"]:
            self.state.blade_engaged = False
        subsystems["starter"] = self._relay_off(self.starter)
        got_lock = await self._acquire_for_estop()
        try:
            subsystems.update(await self._servos_to_neutral())
        finally:
            if got_lock:
                self._act_lock.release()
        if got_lock:
            return subsystems, False
        subsystems.update(await self._reassert_neutral())
        return subsystems, True

    async def emergency_stop(self, source: str = "operator") -> dict[str, Any]:
        """Disengage the blade and center/idle the drive; leave the engine running.

        The software latch is set first and unconditionally (new commands are
        refused from this instant). Each output then gets its own try/except: a
        PCA9685 bus fault on one channel must not abort the rest of the safing
        sequence, and the result says exactly which outputs were confirmed:

        ``{"status": "emergency_stop", "complete": bool, "inflight_write": bool,
        "subsystems": {...}, "faults": [...]}`` -- ``complete`` is ``True`` only when every
        output was driven (or simulated in SIM_MODE); ``inflight_write`` says a command's
        write was still in flight when the stop began, so the servos were driven again after
        it released the lock.

        A disabled platform outside SIM_MODE is not touched at all (``status:
        "not_enabled"``): no GPIO is claimed and nothing is driven.
        """
        if self._hardware_off_limits():
            return self._not_enabled_report()
        self._stops_in_flight += 1
        try:
            self.state.emergency_stop_active = True
            self._stop_generation += 1
            self.state.last_updated = datetime.now(UTC)
            self.revoke()
            self._cancel_deadman()
            # Every autonomous run ends with the stop (a clear must never re-activate one) and
            # the latch is queued for the disk (SAF-01, SAF-16). Both are synchronous and
            # cheap, and both happen before the first await, so the outputs below are not
            # delayed: the relays open in the same event-loop turn.
            stop_events.record_stop("emergency_stop", source)
            persist = latch_store.request_persist(tractor=self)
            subsystems, contended = await self._drive_to_safe_state()
        finally:
            self._stops_in_flight -= 1
        saved = await latch_store.wait(persist)

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
            "inflight_write": contended,
            "subsystems": subsystems,
            "faults": faults,
            # False: the latch could not be written to disk, so a restart would release it.
            "latch_persisted": saved.ok,
            "latch_error": saved.error,
        }

    async def park(self, source: str = "shutdown") -> dict[str, Any]:
        """Bring every actuator to its safe resting state without latching an E-stop.

        Used on shutdown: authorization is revoked, the deadman is stopped, the
        blade/starter relays go off and levers/throttle go neutral/idle, with the
        same per-subsystem reporting as ``emergency_stop``. The engine is left
        alone (there is no ignition-kill actuator).

        Parking *latches*: until ``initialize()`` only neutral, idle and blade-off commands are
        accepted, and a command that began before the park never writes after it. A mission or
        navigation loop still running during shutdown therefore cannot re-command the machine
        after the park, while the PCA9685 holds the last pulse past process exit (audit SF-06,
        SEC-08). A disabled platform outside SIM_MODE is not touched (``not_enabled``).
        """
        if self._hardware_off_limits():
            return self._not_enabled_report()
        self._parked = source
        self._stop_generation += 1
        self.revoke()
        self._cancel_deadman()
        stop_events.record_stop("park", source)
        subsystems, contended = await self._drive_to_safe_state()
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
            "inflight_write": contended,
            "subsystems": subsystems,
            "faults": faults,
        }

    def _interlock_clear_veto(self, reason: str | None) -> str | None:
        """Why the blade interlock latch must stay now (the attached gate), else ``None``."""
        gate = self._interlock_clear_gate
        if gate is None:
            return None
        try:
            return gate(reason)
        except Exception as exc:
            logger.exception("Blade interlock clear gate failed: keeping the latch")
            return f"clear gate failed: {_err(exc)}"

    async def clear_emergency(self) -> dict[str, Any]:
        """Explicit operator clear of the E-stop latch and, when it is safe, the blade latch.

        Never authorizes and never re-engages anything (audit SAF-01): a stop revoked the
        authorization and the clear leaves it revoked, so after a clear nothing autonomous moves
        the machine until a person calls ``authorize()`` and starts a new run (the runs a stop
        ended stay ended). Refused while the tractor is locked out (``lock_out``: that needs a
        restart) and while a stop is still driving the outputs (a clear must not undo it,
        audit SEC-12).

        The blade interlock latch is released only if the attached gate (the tilt monitor)
        agrees: while the latest tilt reading is over the limit, or the tilt sensor is
        unavailable, the latch stays and the reply says so (``status: "partial"``,
        ``still_latched``) -- the E-stop latch is still released, since the blade path stays
        closed (audit SEC-06). The park latch is not released here (only ``initialize()``).
        """
        if self._lockout is not None:
            return self._reject(f"locked out: {self._lockout}", CODE_LOCKOUT)
        if self._stops_in_flight:
            return self._reject("emergency stop in progress: clear refused until its result is out")
        cleared: list[str] = []
        still_latched: list[str] = []
        veto: str | None = None
        if self.state.emergency_stop_active:
            cleared.append("emergency_stop")
        self.state.emergency_stop_active = False
        if self.state.interlock_latched:
            latch = f"blade_interlock:{self.state.latch_reason}"
            veto = self._interlock_clear_veto(self.state.latch_reason)
            if veto is None:
                cleared.append(latch)
                self.state.interlock_latched = False
                self.state.latch_reason = None
            else:
                still_latched.append(latch)
                logger.warning("Blade interlock NOT released by the clear (%s): %s", latch, veto)
        self.state.interlock_reason = None
        # What is still latched, if anything, is what a restart must restore (audit SAF-16).
        saved = await latch_store.persist(tractor=self)
        durability = {"latch_persisted": saved.ok, "latch_error": saved.error}
        if not still_latched:
            return self._ok(cleared=cleared, **durability)
        self.state.interlock_reason = self._standing_reason()
        self.state.last_updated = datetime.now(UTC)
        return {
            "status": "partial",
            "cleared": cleared,
            "still_latched": still_latched,
            "reason": veto,
            **durability,
        }

    async def apply(self, command: TractorCommand, source: str = DEFAULT_SOURCE) -> dict[str, Any]:
        """Apply a full command, honoring interlocks (rejections are collected).

        ``source`` is the label of the caller (see ``set_levers``): the REST router passes
        ``operator``; autonomous callers pass their own name or are treated as autonomous.
        """
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
        self._sync_limit_state()
        return self.state.model_copy()


_tractor_service: TractorControlService | None = None


def get_tractor_service() -> TractorControlService:
    """The process-wide tractor service.

    A ``config/tractor.yaml`` that fails validation (or any constructor failure) must not make
    this raise: it is on the path of the unauthenticated E-stop route, which would then answer
    500 on every call (audit SF-11). The service is built ``misconfigured`` instead: disabled,
    locked out, naming the error, touching no hardware.
    """
    global _tractor_service
    if _tractor_service is None:
        try:
            _tractor_service = TractorControlService()
        except TractorConfigError as exc:
            logger.critical(
                "Tractor config invalid -> platform held disabled and locked out: %s", exc
            )
            _tractor_service = TractorControlService.misconfigured(exc)
        except Exception as exc:
            logger.critical(
                "Tractor service could not be built -> platform held disabled and locked out",
                exc_info=True,
            )
            _tractor_service = TractorControlService.misconfigured(exc)
    return _tractor_service
