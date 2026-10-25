"""
Blade service for IBT-4 control.

Provides a simple async wrapper around IBT4BladeDriver with a process-wide
singleton accessor. Safe in SIM mode; hardware access is lazy and guarded.

The singleton is configured from the ``ibt4:`` block of ``config/hardware.yaml``. Its
``gpio_enabled`` flag is the arming gate for the real lgpio adapter and ships ``false``; a
missing or unreadable block never arms anything (the driver then uses the flag-only stub, and
outside SIM_MODE a flag-only stub REFUSES a blade ON: nothing would be driven, so it must not
answer "accepted"). An unreadable block is retried (``get_blade_service``) instead of being
cached for the life of the process.

An ARMED service (``gpio_enabled``) that was never initialised, e.g. right after a backend
restart when the lines are in an unknown state, is initialised by ``startup()``,
``emergency_stop()`` and ``park()`` themselves: they claim the lines at level 0, drive both low
and read them back, so the stop is confirmed or reported as failed, never "nothing to stop". An
unarmed service is left untouched.

What keeps the blade from running when nobody is steering it (audit BL-01)
--------------------------------------------------------------------------
* ``set_active(True)`` is a DEADMAN: the blade stays on for ``ibt4.deadman_ms`` (default 2 s)
  and is driven OFF, confirmed by readback, unless the command is repeated. A dead client, a lost
  link or a stopped UI cannot leave it running.
* While the blade is on, a supervision task (``SUPERVISE_INTERVAL_S``) re-evaluates the
  standing reasons every tick and drops the blade when one holds: the process-wide E-stop latch,
  the tractor's E-stop / blade interlock (tilt latch, watchdog trip, fail-closed start-up) /
  lock-out / park, the RoboHAT stop hold, any attached permit, a pad that no longer reads ON,
  and the deadman. These are read as STATE, so no stop path has to remember to call the blade
  service. The navigation holds (obstacle, GNSS, leg end, abort) are events and call
  :meth:`request_blade_off`.
* An OFF that is not confirmed stays pending and is retried every tick; while one is pending no
  ON is accepted. An ON is also refused unless the pads READ off first (never from an unconfirmed
  OFF), while the E-stop latch is set, after the shutdown park, and while a standing reason
  holds.
* :meth:`clear_emergency` releases the E-stop latch only over a CONFIRMED OFF.
* :meth:`status` reports what the pads read now (``observed``, ``pad``), not what was commanded.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable
from typing import Any

from ..core.simulation import is_simulation_mode
from ..drivers.blade.ibt4_gpio import IBT4BladeDriver

logger = logging.getLogger(__name__)

#: Blade ON lasts this long unless repeated (``ibt4.deadman_ms`` overrides).
DEFAULT_DEADMAN_MS = 2000
#: How often an unreadable ``ibt4:`` block is re-read by ``get_blade_service``.
CONFIG_RETRY_S = 1.0
#: Minimum seconds between two repeats of the same supervision log line.
_LOG_EVERY_S = 5.0

Permit = Callable[[], str | None]


def _deadman_s(value: Any) -> float:
    """The deadman window in seconds; a missing or unusable value is the default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        value = DEFAULT_DEADMAN_MS
    return max(0.2, min(10.0, float(value) / 1000.0))


class BladeService:
    #: Period of the supervision loop that runs while the blade may be on.
    SUPERVISE_INTERVAL_S = 0.05

    def __init__(
        self, config: dict[str, Any] | None = None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        cfg = dict(config or {})
        self._driver = IBT4BladeDriver(cfg)
        self._initialized = False
        self._lock = asyncio.Lock()
        self._clock = clock
        self.deadman_s = _deadman_s(cfg.get("deadman_ms"))
        #: Set by ``get_blade_service`` while the ``ibt4:`` block could not be read.
        self.config_error: str | None = None
        #: Why the last ON was refused, why the last clear was refused, why the service last
        #: switched the blade off by itself (deadman, standing reason, pad mismatch, hold).
        self.last_refusal: str | None = None
        self.clear_refusal: str | None = None
        self.last_drop: str | None = None
        self._parked: str | None = None
        self._commanded_on = False
        self._on_deadline: float | None = None
        self._off_pending: str | None = None
        self._permits: list[tuple[str, Permit]] = []
        self._supervisor: asyncio.Task[None] | None = None
        self._logged_at: dict[str, float] = {}
        #: Bumped by every emergency_stop(): a clear that began before one never releases after it.
        self._stop_gen = 0

    # ------------------------------------------------------------------ lifecycle

    async def initialize(self) -> bool:
        async with self._lock:
            if self._initialized:
                return True
            try:
                await self._driver.initialize()
                if not self._driver.initialized:
                    # Fail closed: the driver could not claim its outputs. Stay uninitialised so
                    # every command is refused and the next call retries the setup.
                    logger.error(
                        "BladeService: IBT-4 driver did not initialise (%s); blade stays disabled",
                        self._driver.last_error,
                    )
                    return False
                await self._driver.start()
                self._initialized = True
                return True
            except Exception as e:
                logger.warning("BladeService init failed: %s", e)
                return False

    async def startup(self) -> dict[str, Any]:
        """Process start-up: lift the shutdown park and, when armed, claim both lines LOW.

        After a restart the IBT-4 inputs are in whatever state the previous process (or its
        death) left them in; nothing claimed them. An armed service claims both at level 0,
        drives OFF and reads them back, so the output is known to be off before anything can
        command it, or the failure is logged CRITICAL and ``status()`` says why. An unarmed
        service touches nothing. Never raises.
        """
        self._parked = None
        if not self._driver.drives_hardware or is_simulation_mode():
            return self._stop_report(latched=self._driver.estop_latched, off=True, error=None)
        error = await self._claim_lines_for_stop()
        off = self._drive_off("start-up") if self._driver.initialized else False
        report = self._stop_report(latched=self._driver.estop_latched, off=off, error=error)
        if off:
            logger.info("IBT-4 claimed at start-up: both inputs low and read back low")
        else:
            logger.critical(
                "IBT-4 NOT confirmed off at start-up (the blade stays refused): %s",
                error or self._driver.last_error,
            )
        return report

    # --------------------------------------------------------------- commands

    async def set_active(self, active: bool) -> bool:
        """Blade ON (a deadman: repeat it to keep the blade on) or OFF; True only if confirmed.

        A refusal leaves the reason in ``last_refusal``. ON needs: not parked, no E-stop latch,
        no OFF still pending, an armed adapter (outside SIM_MODE), no standing reason (see the
        module docstring), a pad that READS off (or an ON that is already confirmed), and then
        a write that reads back at the ON levels.
        """
        if not self._initialized:
            await self.initialize()
        try:
            if not active:
                ok = self._drive_off("commanded off")
                if not ok:
                    self.last_refusal = self._driver.last_error or "IBT-4 OFF not confirmed"
                return ok
            return self._switch_on()
        except Exception as e:
            logger.error("BladeService set_active failed: %s", e)
            return False

    def _switch_on(self) -> bool:
        """The synchronous ON path: no await between its checks and the write."""
        drv = self._driver
        self.last_refusal = None
        refusal = self._on_refusal()
        if refusal is None and not drv.initialized:
            refusal = drv.last_error or "IBT-4 driver not initialised: nothing can be driven"
        if refusal is not None:
            self.last_refusal = refusal
            logger.warning("Blade ON refused: %s", refusal)
            return False
        now = self._clock()
        obs = drv.observe()
        if self._commanded_on:
            # Already on: this is the deadman refresh. It holds only while the pads still read
            # ON; anything else means the output is not what was commanded, so drop it.
            if obs.state in ("on", "unobservable") and drv.state_confirmed:
                self._on_deadline = now + self.deadman_s
                self._ensure_supervisor()
                return True
            self.last_refusal = f"blade inputs read {obs.state} while the blade is commanded on"
            self._drive_off(self.last_refusal)
            return False
        if obs.state in ("on", "mixed", "error") or not drv.state_confirmed:
            # Never switch on over an output that was not seen off (BL-04): put it off first.
            why = f"blade inputs read {obs.state}" if obs.state != "off" else "last OFF unconfirmed"
            if not self._drive_off(f"{why} before ON"):
                self.last_refusal = f"{why} and the OFF could not be confirmed: ON refused"
                return False
        if not drv.drive(True):
            refusal = drv.last_error or "IBT-4 ON not confirmed"
            # The ON write may have landed although it was not confirmed: the readback failed and
            # the adapter's own rollback failed too ("the output state is UNKNOWN"). So an ON
            # that fails is followed by a service-level OFF, which is confirmed on the pads or
            # stays pending and is retried by the supervision loop every tick (HW-01).
            self._drive_off("ON not confirmed")
            self.last_refusal = refusal
            return False
        self._commanded_on = True
        self._on_deadline = now + self.deadman_s
        self._ensure_supervisor()
        if not self.supervising:  # no running loop: an unsupervised ON is never left on
            self._drive_off("supervision could not start")
            self.last_refusal = "blade supervision could not start: ON refused"
            return False
        return True

    def _on_refusal(self) -> str | None:
        """Why a blade ON must not be accepted right now, else ``None``."""
        drv = self._driver
        if self._parked is not None:
            return f"blade service parked ({self._parked}): restart or startup() required"
        if drv.estop_latched:
            return "blade E-stop latched: operator clear required"
        if self._off_pending is not None:
            return f"a blade OFF is still unconfirmed ({self._off_pending})"
        if not drv.hardware_backed and not is_simulation_mode():
            why = f"; {self.config_error}" if self.config_error else ""
            return (
                "IBT-4 GPIO is not armed (ibt4.gpio_enabled is false, or the adapter is not "
                f"set up): nothing would be driven{why}"
            )
        return self._standing_refusal()

    def _standing_refusal(self) -> str | None:
        """A STATE that forbids the blade (read, never subscribed), else ``None``.

        Evaluated before every ON and on every supervision tick while the blade is on.
        Unreadable state is a refusal, never a pass.
        """
        try:
            from ..core.globals import _safety_state

            if _safety_state.get("emergency_stop_active"):
                return "emergency stop latched"
        except Exception as exc:
            return f"emergency stop state unreadable: {type(exc).__name__}: {exc}"
        try:
            from . import tractor_service

            tractor = tractor_service.get_tractor_service()
            if tractor is not None and getattr(tractor, "enabled", False):
                state = tractor.state
                if tractor.lockout_reason is not None:
                    return f"tractor locked out: {tractor.lockout_reason}"
                if state.emergency_stop_active:
                    return "tractor emergency stop latched"
                if state.interlock_latched:
                    return f"tractor blade interlock latched ({state.latch_reason})"
                if tractor.parked:
                    return "tractor parked"
        except Exception as exc:
            return f"tractor state unreadable: {type(exc).__name__}: {exc}"
        try:
            from . import robohat_service

            robohat = robohat_service.get_robohat_service()
            if robohat is not None and robohat.emergency_latched:
                return "RoboHAT stop hold latched"
        except Exception as exc:
            return f"RoboHAT state unreadable: {type(exc).__name__}: {exc}"
        for name, permit in self._permits:
            try:
                reason = permit()
            except Exception as exc:
                return f"permit {name} failed: {type(exc).__name__}: {exc}"
            if reason:
                return f"{name}: {reason}"
        return None

    def attach_permit(self, name: str, permit: Permit) -> None:
        """Add a standing reason: ``permit()`` returns why the blade must be off, else ``None``.

        Read before every ON and every supervision tick; a permit that raises forbids the blade.
        """
        self._permits.append((name, permit))

    def detach_permit(self, name: str) -> None:
        self._permits = [(n, p) for n, p in self._permits if n != name]

    # ------------------------------------------------------- off and supervision

    def _drive_off(self, reason: str) -> bool:
        """Drive the blade OFF now (synchronously) and record whether it was confirmed.

        Confirmed: every pending reason and the deadman are cleared. Not confirmed: the OFF
        stays pending (``reason``), ON is refused and the supervision loop retries every tick.
        Never raises. An uninitialised driver has nothing to drive: ``False``, nothing pending.
        """
        drv = self._driver
        if not drv.initialized:
            return False
        try:
            ok = drv.drive(False)
        except Exception as exc:
            logger.error("BladeService OFF raised: %s", exc)
            ok = False
        if ok:
            if self._commanded_on or self._off_pending is not None:
                self.last_drop = reason
            self._commanded_on = False
            self._on_deadline = None
            self._off_pending = None
            return True
        self._off_pending = self._off_pending or reason
        self._log_limited("off", logging.CRITICAL, "IBT-4 OFF NOT confirmed (%s): retrying", reason)
        self._ensure_supervisor()
        return False

    def request_blade_off(self, reason: str) -> None:
        """Drive the IBT-4 OFF now, from any stop, hold, leg end or abort path. Never raises.

        Synchronous (a few lgpio calls) and idempotent: nothing is driven when the lines were
        never claimed, and a blade that is commanded off and READS off is left alone. Otherwise
        the OFF is attempted before this returns; if it cannot be confirmed it stays pending,
        ON is refused and the supervision loop retries every tick. Call it on the event loop.
        """
        try:
            drv = self._driver
            if not drv.initialized:
                return
            if not self._commanded_on and self._off_pending is None and drv.state_confirmed:
                if drv.observe().state in ("off", "unobservable"):
                    return
            self._drive_off(reason)
        except Exception:
            logger.exception("request_blade_off(%r) failed", reason)

    async def blade_off(self, reason: str) -> dict[str, Any]:
        """:meth:`request_blade_off`, but reports the outcome (``off`` True only if confirmed)."""
        error: str | None = None
        if self._armed_but_uninitialised:
            error = await self._claim_lines_for_stop()
        off = False
        try:
            off = self._drive_off(reason) if self._driver.initialized else False
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
        return self._stop_report(latched=self._driver.estop_latched, off=off, error=error)

    def _ensure_supervisor(self) -> None:
        """Start the supervision task on the running loop if it is not running; no loop, no task."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._log_limited(
                "noloop", logging.CRITICAL, "IBT-4 supervision needs a running event loop"
            )
            return
        task = self._supervisor
        if task is not None and not task.done() and task.get_loop() is loop:
            return
        self._supervisor = loop.create_task(self._supervise(), name="ibt4-supervisor")

    @property
    def supervising(self) -> bool:
        task = self._supervisor
        return task is not None and not task.done()

    async def _supervise(self) -> None:
        """Run :meth:`_tick` every ``SUPERVISE_INTERVAL_S`` until the blade is off and settled."""
        while True:
            await asyncio.sleep(self.SUPERVISE_INTERVAL_S)
            if self._tick():
                return

    def _tick(self) -> bool:
        """One supervision step; True when nothing is left to supervise. Never raises."""
        try:
            if not self._driver.initialized:
                # The lines were released (driver stopped): nothing can be driven or supervised.
                self._commanded_on = False
                self._on_deadline = None
                self._off_pending = None
                return True
            if self._commanded_on and self._off_pending is None:
                reason = self._drop_reason()
                if reason is not None:
                    self.last_drop = reason
                    logger.warning("IBT-4 blade dropped: %s", reason)
                    self._off_pending = reason
            if self._off_pending is not None:
                self._drive_off(self._off_pending)
            return not self._commanded_on and self._off_pending is None
        except Exception as exc:
            logger.exception("IBT-4 supervision step failed")
            self._off_pending = self._off_pending or f"supervision error: {exc}"
            return False

    def _drop_reason(self) -> str | None:
        """Why the blade that is commanded on must go off now, else ``None``."""
        drv = self._driver
        if self._parked is not None:
            return f"parked ({self._parked})"
        if drv.estop_latched:
            return "E-stop latched"
        standing = self._standing_refusal()
        if standing is not None:
            return standing
        if self._on_deadline is not None and self._clock() >= self._on_deadline:
            return f"deadman: the blade ON was not repeated within {self.deadman_s:.1f}s"
        obs = drv.observe()
        if obs.state in ("off", "mixed", "error"):
            return f"blade inputs read {obs.state} while commanded on" + (
                f" ({obs.error})" if obs.error else ""
            )
        return None

    def _log_limited(self, key: str, level: int, msg: str, *args: object) -> None:
        now = self._clock()
        if now - self._logged_at.get(key, -math.inf) >= _LOG_EVERY_S:
            self._logged_at[key] = now
            logger.log(level, msg, *args)

    # ------------------------------------------------------------------ stops

    @property
    def emergency_latched(self) -> bool:
        """Is the blade E-stop latch set (blade enable refused until cleared)?"""
        return self._driver.estop_latched

    def _stop_report(self, *, latched: bool, off: bool, error: str | None) -> dict[str, Any]:
        return {
            "latched": latched,
            # True only when the LAST off write was confirmed (every write succeeded and both
            # inputs read back low); an unconfirmed write is False, never "probably off".
            "off": off,
            "initialized": bool(self._driver.initialized),
            # False => the driver only tracks a flag (IBT-4 GPIO adapter is a stub):
            # "off" then has no hardware effect and must not be reported as a stop.
            "hardware_backed": self._driver.hardware_backed,
            # False => the last command was not confirmed and the blade may still be on.
            "state_confirmed": self._driver.state_confirmed,
            # The driver's last failure text (setup refused, write/readback mismatch, ...).
            "gpio_error": self._driver.last_error,
            "pins": {"in1": self._driver.in1, "in2": self._driver.in2},
            "error": error,
        }

    @property
    def _armed_but_uninitialised(self) -> bool:
        """Armed (``ibt4.gpio_enabled``) yet never initialised, e.g. right after a restart."""
        return self._driver.gpio_enabled and not self._initialized

    async def _claim_lines_for_stop(self) -> str | None:
        """Initialise an armed, never-initialised service so the stop can drive its lines.

        Nothing in a restarted process has claimed the IBT-4 inputs, so their state is unknown
        and "never initialised" must not read as "nothing to stop": claiming them (level 0)
        lets the stop drive both low and read them back. A failure leaves the driver
        uninitialised with ``gpio_error`` set, which the E-stop reports as ``failed``.
        Returns an error text only if initialising itself raised. Never raises.
        """
        try:
            await self.initialize()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("BladeService: claiming the IBT-4 lines for the stop failed: %s", error)
            return f"claiming the IBT-4 lines for the stop failed: {error}"
        return None

    async def emergency_stop(self) -> dict[str, Any]:
        """Latch the blade E-stop and switch the blade off; report what happened.

        An ARMED service (``ibt4.gpio_enabled``) that was never initialised, for example
        right after a backend restart, is initialised first: the stop claims the lines at
        level 0, drives both low and reads them back, so the report carries a confirmed
        ``off`` or a ``gpio_error``, never "nothing to stop". An unarmed service is left
        exactly as it is (never initialised stays never initialised, no GPIO is touched).

        The latch is set even if the lines could not be claimed, so a later initialise +
        enable is refused until ``clear_emergency()``. ``off`` is ``True`` only when the
        last off write was confirmed by the driver (written and read back); the write is
        retried once if the first attempt was not confirmed. An OFF that is still unconfirmed
        stays pending and is retried by the supervision loop. Never raises.
        """
        error: str | None = None
        off = False
        # The latch and the generation first, in the same loop turn as the call and before any
        # await: a cancelled or slow stop must not leave the blade enable open.
        self._stop_gen += 1
        self._driver.latch_estop()
        if self._armed_but_uninitialised:
            error = await self._claim_lines_for_stop()
        try:
            # Latches first, then drives the blade off: the latch holds even if the write fails.
            off = bool(await self._driver.set_estop(True))
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
            logger.error("BladeService E-stop latch failed: %s", exc)
        if not off:
            try:
                off = bool(await self._driver.set_active(False))
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"
                logger.error("BladeService E-stop blade-off failed: %s", exc)
        self._settle_after_stop(off, "emergency stop")
        return self._stop_report(latched=self._driver.estop_latched, off=off, error=error)

    def _settle_after_stop(self, off: bool, reason: str) -> None:
        """Mirror a stop's outcome in the service state (and keep retrying an unconfirmed OFF)."""
        if off:
            self._commanded_on = False
            self._on_deadline = None
            self._off_pending = None
        elif self._driver.initialized:
            self._off_pending = self._off_pending or reason
            self._ensure_supervisor()

    async def clear_emergency(self) -> None:
        """Release the blade E-stop latch, but only over a CONFIRMED OFF (audit BL-04).

        The inputs are driven low once more and read back first. If that cannot be confirmed
        (pad stuck high, lines not claimable, readback failing) the latch STAYS:
        ``emergency_latched`` remains True and ``clear_refusal`` says why, so a clear cannot
        reopen the blade path over an output that was never seen off. The blade stays off until
        re-enabled.
        """
        self.clear_refusal = None
        generation = self._stop_gen
        if self._armed_but_uninitialised:
            await self._claim_lines_for_stop()
        if generation != self._stop_gen:
            self.clear_refusal = "an emergency stop started while the clear was running"
            return
        released = await self._driver.set_estop(False)
        if released:
            self._settle_after_stop(True, "emergency clear")
            return
        self.clear_refusal = (
            self._driver.last_error or "IBT-4 OFF could not be confirmed: E-stop latch kept"
        )
        logger.critical("Blade E-stop latch NOT released: %s", self.clear_refusal)

    async def park(self) -> dict[str, Any]:
        """Switch the blade off and LATCH the park (shutdown). Never raises.

        Like ``emergency_stop()``, an armed service that was never initialised claims its
        lines first so the off write can be driven and confirmed (see there). Parking
        latches, like the tractor's and the RoboHAT's park: until ``startup()`` no blade ON is
        accepted, so a late request, or a loop that outlives the shutdown, cannot switch the
        blade on again after the park. It is not an E-stop latch (``latched`` stays False).
        """
        error: str | None = None
        off = False
        self._parked = "shutdown"
        if self._armed_but_uninitialised:
            error = await self._claim_lines_for_stop()
        try:
            off = bool(await self._driver.set_active(False))
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            error = error or text
            logger.error("BladeService park failed: %s", text)
        self._settle_after_stop(off, "park")
        return self._stop_report(latched=False, off=off, error=error)

    # ------------------------------------------------------------------ state

    async def status(self) -> dict[str, Any]:
        """The blade as it is NOW: pads read back, latch, deadman, pending OFF, ON refusal."""
        health = await self._driver.health_check()
        remaining = None
        if self._commanded_on and self._on_deadline is not None:
            remaining = max(0, int((self._on_deadline - self._clock()) * 1000))
        return {
            "armed": health["gpio_enabled"],
            "initialized": health["initialized"],
            "hardware_backed": health["hardware_backed"],
            "commanded": "on" if self._commanded_on else "off",
            "observed": health["observed"],
            "pad": health["pad"],
            "active": health["active"],
            "state_confirmed": health["state_confirmed"],
            "emergency_latched": health["estop_latched"],
            "parked": self._parked,
            "off_pending": self._off_pending,
            "deadman_ms": int(self.deadman_s * 1000),
            "deadman_remaining_ms": remaining,
            "supervising": self.supervising,
            "refusal": self._on_refusal(),
            "last_drop": self.last_drop,
            "last_error": health["last_error"],
            "observe_error": health["observe_error"],
            "clear_refusal": self.clear_refusal,
            "config_error": self.config_error,
            "pins": health["pins"],
        }


#: The last error of ``_load_ibt4_config`` (``None`` after a clean read).
_config_error: str | None = None


def _load_ibt4_config() -> dict[str, Any]:
    """The ``ibt4:`` block of hardware.yaml as a plain dict.

    ``{}`` (=> the flag-only stub: fail closed, never armed) on ANY problem or when the block
    is absent, with a warning: a config problem must never arm the real GPIO adapter.
    ``_config_error`` records why, or ``None`` after a clean read.
    """
    global _config_error
    _config_error = None
    try:
        from ..core.config_loader import ConfigLoader

        loader = ConfigLoader(config_dir=os.environ.get("LAWNBERRY_CONFIG_DIR") or None)
        ibt4 = loader.get()[0].ibt4_config
        if ibt4 is not None:
            return ibt4.model_dump()
        _config_error = "hardware.yaml has no ibt4 block"
        logger.warning("hardware.yaml has no ibt4 block; IBT-4 GPIO NOT armed (flag-only stub)")
    except Exception as exc:
        _config_error = f"ibt4 config unreadable ({type(exc).__name__}: {exc})"
        logger.warning(
            "ibt4 config unreadable (%s: %s); IBT-4 GPIO NOT armed (flag-only stub)",
            type(exc).__name__,
            exc,
        )
    return {}


_blade_service: BladeService | None = None
_config_retry_at = 0.0


def get_blade_service() -> BladeService:
    """The process-wide service.

    Built from the ``ibt4:`` block on first use. A service built while that block was absent
    or unreadable is not cached for good: the block is re-read at most every
    ``CONFIG_RETRY_S`` and, once readable, the service is rebuilt from it (an E-stop latch or
    park already set is carried over). A stub never held a GPIO line, so nothing else is lost.
    """
    global _blade_service, _config_retry_at
    service = _blade_service
    if service is None:
        service = BladeService(_load_ibt4_config())
        service.config_error = _config_error
        _blade_service = service
        _config_retry_at = time.monotonic() + CONFIG_RETRY_S
    elif service.config_error is not None and time.monotonic() >= _config_retry_at:
        _config_retry_at = time.monotonic() + CONFIG_RETRY_S
        config = _load_ibt4_config()
        if _config_error is None:
            fresh = BladeService(config)
            fresh._parked = service._parked
            fresh._driver._estop_latched = service._driver.estop_latched
            fresh._permits = list(service._permits)  # the navigation gate's permit is attached once
            logger.warning("ibt4 config readable again: the blade service was rebuilt from it")
            _blade_service = service = fresh
        else:
            service.config_error = _config_error
    return service
