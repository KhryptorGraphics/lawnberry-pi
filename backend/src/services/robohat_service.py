"""Text protocol bridge for the RoboHAT RP2040 control board.

The bundled CircuitPython firmware (`robohat-rp2040-code/code.py`) exposes a
human-readable command set on its USB serial interface.  Earlier iterations of
the backend spoke a JSON RPC dialect which the current firmware rejects, so the
communication layer has to emit the same text commands you would type over a
serial console (e.g. ``rc=disable``, ``pwm,1500,1600``).

This module keeps the async façade expected by the FastAPI endpoints while
performing the minimal blocking serial I/O in tiny, protected critical sections.
"""

import asyncio
import glob
import json
import logging
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import serial

try:  # Best-effort optional import for USB device enumeration.
    from serial.tools import list_ports  # type: ignore
except Exception:  # pragma: no cover - serial.tools may be unavailable in CI
    list_ports = None  # type: ignore

logger = logging.getLogger(__name__)

# How long a drive command stays in force when the caller gives no duration, and
# the ceiling for any requested duration. Once a non-neutral command outlives its
# window the service sends neutral once and stops maintaining USB control, so the
# firmware's own failsafe (USB timeout -> RC mode -> neutral on RC loss) takes
# over instead of being kept alive by a blind resend of the last PWM.
DEFAULT_COMMAND_VALIDITY_S = 0.5
MAX_COMMAND_VALIDITY_S = 10.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _clamp_validity(seconds: float) -> float:
    return max(0.05, min(float(seconds), MAX_COMMAND_VALIDITY_S))


@dataclass
class RoboHATStatus:
    """RoboHAT firmware status"""

    firmware_version: str = "unknown"
    uptime_seconds: int = 0
    watchdog_active: bool = False
    last_watchdog_echo: str | None = None
    watchdog_latency_ms: float = 0.0
    serial_connected: bool = False
    error_count: int = 0
    last_error: str | None = None
    motor_controller_ok: bool = False
    encoder_feedback_ok: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "firmware_version": self.firmware_version,
            "uptime_seconds": self.uptime_seconds,
            "watchdog_active": self.watchdog_active,
            "last_watchdog_echo": self.last_watchdog_echo,
            "watchdog_latency_ms": self.watchdog_latency_ms,
            "serial_connected": self.serial_connected,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "motor_controller_ok": self.motor_controller_ok,
            "encoder_feedback_ok": self.encoder_feedback_ok,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "health_status": self.get_health_status(),
        }

    def get_health_status(self) -> str:
        """Get overall health status"""
        if not self.serial_connected:
            return "disconnected"
        if self.error_count > 10:
            return "fault"
        if not self.watchdog_active or not self.motor_controller_ok:
            return "warning"
        return "healthy"


class RoboHATService:
    """RoboHAT RP2040 serial bridge service"""

    def __init__(self, serial_port: str = "/dev/ttyACM0", baud_rate: int = 115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial_conn: serial.Serial | None = None
        self.status = RoboHATStatus()
        self.running = False
        self.watchdog_task: asyncio.Task | None = None
        self.read_task: asyncio.Task | None = None
        self._serial_lock = asyncio.Lock()
        self._rc_enabled = True
        self._usb_control_requested = False
        self._pending_rc_state: bool | None = None
        self._pending_rc_since: float = 0.0
        self._last_status_at: float = 0.0
        # Track last PWM sent so we can refresh it to avoid firmware USB timeout
        self._last_pwm: tuple[int, int] = (1500, 1500)
        self._last_pwm_at: float = 0.0
        # Drive-command validity window (monotonic deadline; 0.0 = no non-neutral
        # command in force). See DEFAULT_COMMAND_VALIDITY_S.
        self._command_validity_s = _clamp_validity(
            _env_float("ROBOHAT_COMMAND_VALIDITY_S", DEFAULT_COMMAND_VALIDITY_S)
        )
        self._cmd_valid_until: float = 0.0
        self._cmd_expiry_task: asyncio.Task | None = None
        # Wakes the expiry task when a command with a *shorter* window arrives while it sleeps
        # towards a later deadline (audit SF-07). Created together with the task.
        self._cmd_wake: asyncio.Event | None = None
        # Software E-stop latch: hold neutral (and USB control) until cleared, so
        # the stop cannot be undone by the RC receiver or a later drive command.
        self._estop_hold = False
        # Shutdown park latch: after park() only neutral drive and blade-off commands are
        # accepted, so a loop still running during shutdown cannot re-command the platform.
        # Cleared by initialize().
        self._parked = False
        # Firmware echo counters, used to confirm a stop was actually accepted.
        self._acks: dict[str, int] = {"pwm": 0, "blade": 0}

    @property
    def emergency_latched(self) -> bool:
        """Is a software E-stop hold latched (drive/blade-on refused until cleared)?"""
        return self._estop_hold

    def restore_emergency_hold(self) -> None:
        """Re-latch the stop hold a previous process left set (audit SAF-16); never clears one."""
        self._estop_hold = True
        self._last_pwm = (1500, 1500)
        self._last_pwm_at = time.monotonic()
        self.status.last_error = "emergency_stop_latched"

    async def initialize(self) -> bool:
        """Initialize RoboHAT serial connection"""
        self._parked = False  # a fresh bring-up re-arms a parked service
        try:
            logger.info(
                f"Initializing RoboHAT service on {self.serial_port} at {self.baud_rate} baud"
            )

            # Open serial connection
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=1.0,
                write_timeout=1.0,
                # A second reader on this tty (a GPS probe, a script) steals the firmware's
                # `[USB] pwm/blade` acknowledgements: the E-stop then reads "unacked" (SAF-S1).
                exclusive=True,
            )

            # Wait for connection to stabilize
            await asyncio.sleep(2.0)

            # Flush any banner/boot messages so we begin with a clean buffer
            self._drain_serial_buffer()

            # Place the firmware in USB (backend) control mode
            await self._set_rc_enabled(False, force=True)

            self.status.serial_connected = True
            self.running = True

            # Start background tasks
            self.watchdog_task = asyncio.create_task(self._watchdog_loop())
            self.read_task = asyncio.create_task(self._read_loop())

            logger.info("RoboHAT service initialized successfully")
            return True

        except serial.SerialException as e:
            logger.error(f"Failed to initialize RoboHAT service: {e}")
            self.status.serial_connected = False
            self.status.last_error = str(e)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing RoboHAT: {e}")
            self.status.last_error = str(e)
            return False

    async def _send_line(self, line: str) -> bool:
        """Send a raw line to the RoboHAT firmware."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False

        message = (line.rstrip("\r\n") + "\n").encode("utf-8")

        try:
            async with self._serial_lock:
                await asyncio.to_thread(self.serial_conn.write, message)
                await asyncio.to_thread(self.serial_conn.flush)
            return True
        except Exception as exc:  # pragma: no cover - hardware error path
            logger.error("Failed to send RoboHAT command line '%s': %s", line, exc)
            self.status.error_count += 1
            self.status.last_error = str(exc)
            return False

    def _drain_serial_buffer(self) -> None:
        """Synchronously drain any pending bytes from the serial buffer."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return

        try:
            # Read until buffer empty; keep the last status to process quickly
            while self.serial_conn.in_waiting:
                raw = self.serial_conn.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    self._process_line(line)
        except Exception:
            # Non-fatal – buffer draining is best effort
            pass

    def _mark_rc_state(self, enabled: bool) -> None:
        """Track the latest RC enable/disable state acknowledgement from firmware."""
        self._rc_enabled = enabled
        if self._pending_rc_state is not None:
            if self._pending_rc_state == enabled:
                self._pending_rc_state = None
                self._pending_rc_since = 0.0
                if enabled:
                    # An explicit enable acknowledgement corresponds to a backend request
                    self._usb_control_requested = False
            else:
                logger.warning(
                    "RoboHAT reported RC %s while backend awaited %s",
                    "enabled" if enabled else "disabled",
                    "enabled" if self._pending_rc_state else "disabled",
                )
                self._pending_rc_state = None
                self._pending_rc_since = 0.0

    async def _set_rc_enabled(self, enabled: bool, *, force: bool = False) -> None:
        """Toggle RC passthrough on the firmware so USB commands take over."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return

        # Record our desired control mode so the watchdog maintains it.
        self._usb_control_requested = not enabled

        if not force:
            if self._pending_rc_state is not None and self._pending_rc_state == enabled:
                return
            if self._pending_rc_state is None and self._rc_enabled == enabled:
                return

        cmd = "rc=enable" if enabled else "rc=disable"
        if await self._send_line(cmd):
            self._pending_rc_state = enabled
            self._pending_rc_since = time.monotonic()
        else:
            self._pending_rc_state = None
            self._pending_rc_since = 0.0

    async def _watchdog_loop(self):
        """Periodic watchdog ping loop"""
        while self.running:
            try:
                # When USB control is active, periodically refresh the neutral
                # PWM command so the firmware keeps RC disabled.
                await self._maintain_usb_control()

                await asyncio.sleep(1.0)

                now = time.monotonic()
                if self._last_status_at:
                    delta = now - self._last_status_at
                    self.status.watchdog_latency_ms = delta * 1000.0
                    self.status.watchdog_active = delta < 3.0
                else:
                    self.status.watchdog_active = False
                    self.status.watchdog_latency_ms = 0.0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog loop error: {e}")
                self.status.watchdog_active = False
                await asyncio.sleep(1.0)

    async def _maintain_usb_control(self) -> None:
        """Keep USB control only while a fresh drive command (or an E-stop hold) needs it.

        The firmware falls back to RC mode after ~2 s of USB silence and, with no
        RC signal, to neutral -- that is the failsafe. This loop therefore never
        re-acquires USB control or re-sends a stale non-neutral PWM on its own:
        it refreshes the *current* command only while it is inside its validity
        window (long commands need that to outlive the firmware's 2 s timeout),
        and holds neutral after an E-stop until it is cleared.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return

        # A latched E-stop hold keeps USB control alive whatever the request flag says: the flag
        # is dropped by the idle check below and by an RC-enable acknowledgement, and a hold that
        # depended on it stopped re-sending neutral once it was gone (audit SF-10).
        if not (self._usb_control_requested or self._estop_hold):
            return

        now = time.monotonic()
        if not (now < self._cmd_valid_until or self._estop_hold):
            # No fresh command and no latched stop: leave it to the firmware.
            self._usb_control_requested = False
            return

        if self._rc_enabled:
            await self._set_rc_enabled(False)
            return

        if self._pending_rc_state is not None:
            if (time.monotonic() - self._pending_rc_since) > 1.0:
                logger.warning("RoboHAT RC disable acknowledgement still pending; retrying command")
                await self._set_rc_enabled(False, force=True)
            return
        # Refresh the current command so the firmware does not time out to RC
        # mode mid-command (SERIAL_TIMEOUT ≈ 2s on firmware). Under an E-stop hold the refresh is
        # the constant neutral, never ``_last_pwm``: a command that slipped past the hold check
        # must not be re-sent as the "hold" (audit SF-02).
        if (now - max(self._last_status_at, self._last_pwm_at)) >= 0.9:
            steer_us, thr_us = (1500, 1500) if self._estop_hold else self._last_pwm
            await self._send_line(f"pwm,{steer_us},{thr_us}")
            self._last_pwm_at = now

    def _ensure_command_expiry(self) -> None:
        """Have a task waiting for the *current* deadline; wake the one that is already asleep.

        A command with a shorter window than the deadline the task sleeps towards must pull
        its wake-up forward (audit SF-07): a 0.2 s command that followed a 1.5 s one used to
        stay in force until the first deadline.
        """
        if self._cmd_expiry_task is None or self._cmd_expiry_task.done():
            self._cmd_wake = asyncio.Event()
            self._cmd_expiry_task = asyncio.get_running_loop().create_task(
                self._command_expiry_loop()
            )
        elif self._cmd_wake is not None:
            self._cmd_wake.set()  # re-evaluate: an earlier deadline may have just been armed

    def _cancel_command_expiry(self) -> None:
        self._cmd_valid_until = 0.0
        task = self._cmd_expiry_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _command_expiry_loop(self) -> None:
        """Send neutral once when a non-neutral drive command outlives its window."""
        wake = self._cmd_wake
        while True:
            if self._cmd_valid_until <= 0.0:
                return  # cancelled, or replaced by a neutral command
            remaining = self._cmd_valid_until - time.monotonic()
            if remaining > 0:
                if wake is None:  # defensive: always created with the task
                    await asyncio.sleep(remaining)
                    continue
                wake.clear()
                try:
                    async with asyncio.timeout(remaining):
                        await wake.wait()
                except TimeoutError:
                    pass
                continue
            self._cmd_valid_until = 0.0
            logger.warning(
                "RoboHAT drive command expired: neutral sent, firmware failsafe in charge"
            )
            self._last_pwm = (1500, 1500)
            self._last_pwm_at = time.monotonic()
            if not await self._send_line("pwm,1500,1500"):
                # The firmware's own 2 s USB timeout still applies.
                self.status.last_error = "command_expiry_neutral_failed"
            return

    async def _wait_for_usb_control(self, timeout: float = 0.75) -> bool:
        """Block until the firmware confirms USB control or timeout expires."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._rc_enabled and self._pending_rc_state is None:
                return True
            await asyncio.sleep(0.02)

        logger.warning("Timed out waiting for RoboHAT to acknowledge USB control")
        return False

    async def _ensure_usb_control(self, *, timeout: float = 0.75, retries: int = 1) -> bool:
        """Ensure RoboHAT is in USB control mode, retrying if acknowledgements lag."""
        if not self.serial_conn or not self.serial_conn.is_open or not self.running:
            return False

        attempts = max(1, retries + 1)
        for attempt in range(attempts):
            try:
                await self._set_rc_enabled(False, force=attempt > 0)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Attempt %d/%d to disable RC control failed: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
            adjusted_timeout = timeout + 0.1 * attempt
            if await self._wait_for_usb_control(timeout=adjusted_timeout):
                self.status.last_error = None
                return True
            await asyncio.sleep(0.05)

        logger.warning("Unable to obtain RoboHAT USB control after %d attempts", attempts)
        self.status.motor_controller_ok = False
        self.status.last_error = "usb_control_unavailable"
        return False

    async def _read_loop(self):
        """Continuous read loop for RoboHAT messages"""
        while self.running:
            try:
                line = None
                if self.serial_conn and self.serial_conn.is_open and self.serial_conn.in_waiting:
                    raw = await asyncio.to_thread(self.serial_conn.readline)
                    line = raw.decode("utf-8", errors="ignore").strip()

                if line:
                    self._process_line(line)

                await asyncio.sleep(0.02)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Read loop error: {e}")
                await asyncio.sleep(0.1)

    def _process_line(self, line: str) -> None:
        """Parse human-readable firmware messages and update status fields."""
        if not line:
            return

        line_lower = line.lower()

        if line_lower.startswith("[status]"):
            try:
                brace_index = line.index("{")
                payload = json.loads(line[brace_index:])
            except Exception as exc:
                logger.warning("Failed to parse RoboHAT status payload '%s': %s", line, exc)
                return
            self._last_status_at = time.monotonic()
            if isinstance(payload.get("uptime_seconds"), (int, float)):
                self.status.uptime_seconds = int(payload["uptime_seconds"])
            rc_enabled = bool(payload.get("rc_enabled", self._rc_enabled))
            if rc_enabled != self._rc_enabled:
                self._mark_rc_state(rc_enabled)
            self.status.motor_controller_ok = not rc_enabled
            self.status.encoder_feedback_ok = payload.get("encoder") is not None
            self.status.last_watchdog_echo = "status"
            return

        if "rc enabled" in line_lower or "timeout" in line_lower and "rc mode" in line_lower:
            self._mark_rc_state(True)
            self._last_status_at = time.monotonic()
            logger.info("RoboHAT reported RC mode active")
            return

        if "rc disabled" in line_lower:
            self._mark_rc_state(False)
            self._last_status_at = time.monotonic()
            logger.info("RoboHAT reported USB control active")
            return

        if line_lower.startswith("[usb] pwm"):
            self._last_status_at = time.monotonic()
            self.status.motor_controller_ok = True
            self.status.last_watchdog_echo = "pwm_ack"
            self._acks["pwm"] += 1
            return

        if line_lower.startswith("[usb] blade"):
            self._last_status_at = time.monotonic()
            self._acks["blade"] += 1
            return

        if line_lower.startswith("[usb] invalid"):
            # The backend attempted an unsupported command earlier. Count it once.
            if "pwm" in line_lower:
                self._mark_rc_state(True)
            logger.warning("RoboHAT rejected command: %s", line)
            self.status.motor_controller_ok = False
            self.status.error_count += 1
            self.status.last_error = line
            return

        if line_lower.startswith("\u25b6") or line_lower.startswith("▶"):
            # Firmware banner – useful for firmware version detection.
            self.status.firmware_version = line.strip()
            return

        if line_lower.startswith("[rc]"):
            # Heartbeat from firmware – keep as last echo.
            self.status.last_watchdog_echo = line
            self._last_status_at = time.monotonic()
            return

        # For everything else, keep a debug breadcrumb without polluting logs.
        logger.debug("RoboHAT: %s", line)

    async def send_motor_command(
        self, left_speed: float, right_speed: float, valid_for_s: float | None = None
    ) -> bool:
        """Send a drive command that stays in force for ``valid_for_s`` seconds.

        ``valid_for_s`` defaults to the configured validity
        (``ROBOHAT_COMMAND_VALIDITY_S``, 0.5 s) and is clamped to
        ``MAX_COMMAND_VALIDITY_S``. A caller driving continuously must re-send
        inside that window; if it lapses without a refresh the service sends
        neutral once and stops maintaining USB control. A neutral command needs
        no window. Refused while an E-stop hold is latched.
        """
        if not self.serial_conn or not self.serial_conn.is_open or not self.running:
            return False

        if self._estop_hold:
            self.status.last_error = "emergency_stop_latched"
            return False

        try:
            steer_us, throttle_us = self._mix_arcade_to_pwm(left_speed, right_speed)
        except ValueError:
            # NaN/inf would have clamped to full scale (audit SF-12): refuse it.
            self.status.last_error = "non_finite_command"
            return False
        neutral = (steer_us, throttle_us) == (1500, 1500)
        if self._parked and not neutral:
            self.status.last_error = "parked"
            return False

        if not await self._ensure_usb_control(timeout=0.9, retries=2):
            return False

        # The handshake above can take ~3 s. An E-stop (or the shutdown park) that ran in the
        # meantime must not be overtaken by this command: re-check, with no await between the
        # check and the write (audit SF-02).
        if self._estop_hold or (self._parked and not neutral):
            self.status.last_error = "emergency_stop_latched" if self._estop_hold else "parked"
            return False

        validity = self._command_validity_s if valid_for_s is None else _clamp_validity(valid_for_s)
        ok = await self._send_line(f"pwm,{steer_us},{throttle_us}")

        if self._estop_hold or (self._parked and not neutral):
            # The stop latched while this write was in flight. Its neutral is queued behind this
            # write on the serial lock and lands after it, so this command must not be recorded
            # as the one in force (that would re-arm its window and let the hold re-send it).
            self.status.last_error = "emergency_stop_latched" if self._estop_hold else "parked"
            return False

        if ok:
            self.status.motor_controller_ok = True
            self.status.last_watchdog_echo = f"pwm:{steer_us}/{throttle_us}"
            self.status.last_error = None
            now = time.monotonic()
            self._last_pwm = (steer_us, throttle_us)
            self._last_pwm_at = now
            if neutral:
                self._cancel_command_expiry()  # neutral: nothing to expire
            else:
                self._cmd_valid_until = now + validity
                self._ensure_command_expiry()
        else:
            self.status.motor_controller_ok = False
            self.status.last_error = "pwm_send_failed"
        return ok

    async def send_blade_command(self, active: bool, speed: float = 1.0) -> bool:
        """Send blade motor command to RoboHAT"""
        if not self.serial_conn or not self.serial_conn.is_open or not self.running:
            return False

        blade_on = active and speed > 0
        if blade_on and (self._estop_hold or self._parked):
            self.status.last_error = "emergency_stop_latched" if self._estop_hold else "parked"
            return False

        if not await self._ensure_usb_control(timeout=0.9, retries=2):
            return False
        # Re-check after the handshake, as in send_motor_command (audit SF-02): a blade-on that
        # passed the hold check must not land after the stop's blade=off.
        if blade_on and (self._estop_hold or self._parked):
            self.status.last_error = "emergency_stop_latched" if self._estop_hold else "parked"
            return False
        command = "blade=on" if blade_on else "blade=off"
        ok = await self._send_line(command)
        if ok and not active:
            self.status.last_watchdog_echo = "blade:off"
        elif ok:
            self.status.last_watchdog_echo = "blade:on"
            self.status.last_error = None
        else:
            self.status.last_error = "blade_command_failed"
        return ok

    async def _send_confirmed(self, line: str, ack: str, timeout: float = 0.5) -> str:
        """Send ``line`` and wait for the firmware's echo.

        Returns ``"acked"`` (firmware echoed it), ``"unacked"`` (written to the
        port but no echo within ``timeout`` -- e.g. the firmware is in RC mode and
        rejected it) or ``"send_failed"`` (the write itself failed).
        """
        before = self._acks[ack]
        if not await self._send_line(line):
            return "send_failed"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._acks[ack] > before:
                return "acked"
            await asyncio.sleep(0.02)
        return "unacked"

    async def emergency_stop_detail(self) -> dict[str, Any]:
        """Latch a stop on the RoboHAT and report what the firmware actually did.

        ``drive``/``blade`` are each ``"acked"``, ``"unacked"`` or
        ``"send_failed"`` (see ``_send_confirmed``); ``usb_control`` says whether
        the firmware was in USB-control mode for the drive stop. The blade is
        switched off first (accepted in any firmware mode, no USB control
        needed), then neutral PWM. After this the service holds neutral until
        ``clear_emergency()`` and refuses new drive / blade-on commands.
        """
        logger.critical("Sending emergency stop to RoboHAT")
        detail: dict[str, Any] = {
            "drive": "send_failed",
            "blade": "send_failed",
            "usb_control": False,
            "error": None,
        }
        if not self.serial_conn or not self.serial_conn.is_open or not self.running:
            detail["error"] = "serial not connected"
            return detail

        self._estop_hold = True
        self._cancel_command_expiry()
        self._last_pwm = (1500, 1500)
        self._last_pwm_at = time.monotonic()
        detail["blade"] = await self._send_confirmed("blade=off", "blade")
        detail["drive"] = await self._send_confirmed("pwm,1500,1500", "pwm", timeout=0.3)
        detail["usb_control"] = not self._rc_enabled
        if detail["drive"] != "acked":
            # pwm is only accepted in USB-control mode: take control and retry.
            detail["usb_control"] = await self._ensure_usb_control(timeout=0.6, retries=2)
            if not detail["usb_control"]:
                logger.warning("Emergency stop could not obtain USB control of the RoboHAT")
            detail["drive"] = await self._send_confirmed("pwm,1500,1500", "pwm")
        self.status.motor_controller_ok = False
        self.status.last_watchdog_echo = "emergency_stop"
        if "send_failed" in (detail["drive"], detail["blade"]):
            self.status.last_error = "emergency_stop_failed"
            detail["error"] = "serial write failed"
        return detail

    async def emergency_stop(self) -> bool:
        """Send emergency stop to the RoboHAT; ``True`` if both commands were written.

        This is the legacy "was it sent" result. Use ``emergency_stop_detail()``
        for whether the firmware accepted the stop.
        """
        detail = await self.emergency_stop_detail()
        return "send_failed" not in (detail["drive"], detail["blade"])

    async def park(self) -> dict[str, Any]:
        """Neutral PWM and blade off without latching an E-stop (shutdown).

        Best effort and bounded: never takes USB control just to park (in RC mode
        the RC receiver / firmware failsafe governs drive, and ``blade=off`` is
        accepted in any mode). Same result shape as ``emergency_stop_detail``.
        """
        detail: dict[str, Any] = {"drive": "send_failed", "blade": "send_failed", "error": None}
        if not self.serial_conn or not self.serial_conn.is_open:
            detail["error"] = "serial not connected"
            return detail
        self._cancel_command_expiry()
        self._parked = True  # latch: only neutral drive / blade-off until initialize()
        self._estop_hold = False
        self._last_pwm = (1500, 1500)
        self._last_pwm_at = time.monotonic()
        detail["blade"] = await self._send_confirmed("blade=off", "blade")
        detail["drive"] = await self._send_confirmed("pwm,1500,1500", "pwm", timeout=0.3)
        if "send_failed" in (detail["drive"], detail["blade"]):
            detail["error"] = "serial write failed"
        return detail

    async def clear_emergency(self) -> bool:
        """Clear emergency stop on RoboHAT.

        The hold is released only AFTER the firmware is under USB control and acknowledged the
        handshake (audit SAF-15). Until then it keeps sending neutral, so neither the RC
        receiver nor a stale command can move the machine. A clear that fails leaves the hold
        set and returns ``False``; the caller must report the platform as still held.
        """
        logger.info("Clearing emergency stop on RoboHAT")
        if not self.serial_conn or not self.serial_conn.is_open or not self.running:
            return False

        if not await self._ensure_usb_control(timeout=0.9, retries=2):
            self.status.last_error = "clear_emergency_failed"
            return False
        ok = await self._send_line("rc=disable")
        if not ok:
            self.status.last_error = "clear_emergency_failed"
            return False
        self._estop_hold = False
        self.status.last_error = None
        return True

    def get_status(self) -> RoboHATStatus:
        """Get current RoboHAT status"""
        self.status.timestamp = datetime.now(UTC)
        return self.status

    @staticmethod
    def _mix_arcade_to_pwm(left_speed: float, right_speed: float) -> tuple[int, int]:
        """Convert differential wheel speeds into steer/throttle PWM microseconds.

        Raises ``ValueError`` for NaN/inf: clamping turns NaN into full scale (audit SF-12).
        """
        if not (math.isfinite(left_speed) and math.isfinite(right_speed)):
            raise ValueError(f"non-finite wheel speeds ({left_speed!r}, {right_speed!r}) refused")
        max_input = max(1.0, abs(left_speed), abs(right_speed))
        left_norm = left_speed / max_input
        right_norm = right_speed / max_input

        linear = (left_norm + right_norm) / 2.0
        angular = (right_norm - left_norm) / 2.0

        throttle_us = RoboHATService._scale_to_pwm(linear)
        steer_us = RoboHATService._scale_to_pwm(angular, span=350)
        return steer_us, throttle_us

    @staticmethod
    def _scale_to_pwm(value: float, span: int = 450, center: int = 1500) -> int:
        if not math.isfinite(value):
            raise ValueError(f"non-finite pwm value {value!r} refused")
        value = max(-1.0, min(1.0, value))
        us = int(round(center + value * span))
        return max(1000, min(2000, us))

    async def shutdown(self):
        """Shutdown RoboHAT service"""
        logger.info("Shutting down RoboHAT service")

        # Park first: neutral PWM and blade off, so a mid-command shutdown does not
        # leave the firmware holding the last command until its own timeout.
        try:
            await asyncio.wait_for(self.park(), timeout=3.0)
        except Exception as exc:
            logger.warning("RoboHAT park on shutdown failed: %s", exc)

        self.running = False

        # Try to hand control back to RC so that the mower is safe even if the
        # backend stops responding.
        try:
            await self._set_rc_enabled(True)
        except Exception:
            pass

        self._cancel_command_expiry()

        # Cancel background tasks
        if self.watchdog_task:
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass

        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass

        # Close serial connection
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

        self.status.serial_connected = False
        logger.info("RoboHAT service shutdown complete")


# Global RoboHAT service instance
robohat_service: RoboHATService | None = None


def get_robohat_service() -> RoboHATService | None:
    """Get global RoboHAT service instance"""
    return robohat_service


_ROBOHAT_PORT_ENV_VARS: tuple[str, ...] = (
    "ROBOHAT_SERIAL_PORT",
    "ROBOHAT_PORT",
    "LAWN_ROBOHAT_PORT",
)
_ROBOHAT_KEYWORDS: tuple[str, ...] = (
    "robohat",
    "rp2040",
    "circuitpython",
    "cytron",
    "lawnberry",
)
_RP2040_VENDOR_IDS: set[int] = {0x2E8A}  # Raspberry Pi RP2040 VID


def _settings_profile_paths() -> Iterable[Path]:
    """Candidate settings profile files that may contain RoboHAT overrides."""

    env_dir = os.getenv("LAWN_SETTINGS_DIR") or os.getenv("LAWN_CONFIG_DIR")
    if env_dir:
        base = Path(env_dir)
        yield base / "default.json"
        yield base / "settings.json"

    system_base = Path("./config")
    yield system_base / "default.json"
    yield system_base / "settings.json"

    project_base = Path(os.getcwd()) / "config"
    yield project_base / "default.json"
    yield project_base / "settings.json"


def _read_profile_robohat_port() -> str | None:
    """Extract RoboHAT serial port override from persisted settings if present."""

    for path in _settings_profile_paths():
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            hardware = data.get("hardware") if isinstance(data, dict) else None
            port = hardware.get("robohat_port") if isinstance(hardware, dict) else None
            if isinstance(port, str) and port.strip():
                return port.strip()
        except Exception:  # pragma: no cover - permissive best-effort parsing
            continue
    return None


def _serial_by_id_candidates() -> list[str]:
    patterns = (
        "/dev/serial/by-id/*RoboHAT*",
        "/dev/serial/by-id/*Robo_HAT*",
        "/dev/serial/by-id/*RP2040*",
        "/dev/serial/by-id/*CircuitPython*",
        "/dev/serial/by-id/*Pico*",
    )
    candidates: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.exists(path):
                candidates.append(path)
    return candidates


def _list_ports_candidates() -> list[str]:
    """Return serial devices that resemble the RoboHAT firmware."""

    if list_ports is None:  # pragma: no cover - serial.tools optional in CI
        return []

    matches: list[str] = []
    try:
        for info in list_ports.comports():  # type: ignore[attr-defined]
            desc_parts = [
                getattr(info, "device", ""),
                getattr(info, "description", ""),
                getattr(info, "manufacturer", ""),
                getattr(info, "product", ""),
            ]
            descriptor = " ".join(str(p or "") for p in desc_parts).lower()
            vid = getattr(info, "vid", None)
            if any(keyword in descriptor for keyword in _ROBOHAT_KEYWORDS):
                matches.append(info.device)
            elif isinstance(vid, int) and vid in _RP2040_VENDOR_IDS:
                matches.append(info.device)
    except Exception:  # pragma: no cover - enumeration issues should not abort startup
        return []
    return matches


def _candidate_serial_ports(explicit: str | None = None) -> list[str]:
    """Build an ordered list of serial ports to try for RoboHAT."""

    seen: set[str] = set()
    ordered: list[str] = []

    def add(port: str | None, *, require_exists: bool = False) -> None:
        if port is None:
            return
        value = str(port).strip()
        if not value or value in seen:
            return
        if require_exists and value.startswith("/dev/") and not os.path.exists(value):
            return
        seen.add(value)
        ordered.append(value)

    add(explicit)

    for env_var in _ROBOHAT_PORT_ENV_VARS:
        add(os.getenv(env_var))

    add(_read_profile_robohat_port())

    for path in _serial_by_id_candidates():
        add(path, require_exists=True)

    for path in _list_ports_candidates():
        add(path, require_exists=True)

    for path in sorted(glob.glob("/dev/ttyACM*")):
        add(path, require_exists=True)
    for path in sorted(glob.glob("/dev/ttyUSB*")):
        add(path, require_exists=True)

    # Always include the historic default as a last resort.
    add("/dev/ttyACM0")

    return ordered


async def initialize_robohat_service(
    serial_port: str | None = None, baud_rate: int = 115200
) -> bool:
    """Initialize global RoboHAT service, probing common serial ports when needed."""
    global robohat_service

    if (
        robohat_service
        and robohat_service.running
        and robohat_service.status.serial_connected
        and (serial_port is None or robohat_service.serial_port == serial_port)
    ):
        return True

    if robohat_service and robohat_service.running:
        try:
            await robohat_service.shutdown()
        except Exception:  # pragma: no cover - shutdown best-effort
            pass
        robohat_service = None

    candidates = _candidate_serial_ports(serial_port)
    if not candidates:
        candidates = [serial_port or "/dev/ttyACM0"]

    last_attempt: RoboHATService | None = None
    for candidate in candidates:
        svc = RoboHATService(candidate, baud_rate)
        ok = await svc.initialize()
        last_attempt = svc
        if ok:
            robohat_service = svc
            logger.info("RoboHAT service initialized on %s", candidate)
            return True
        logger.warning(
            "RoboHAT initialization failed on %s: %s",
            candidate,
            svc.status.last_error or "unknown_error",
        )

    robohat_service = last_attempt
    if candidates:
        logger.error(
            "Unable to initialize RoboHAT service; attempted ports: %s", ", ".join(candidates)
        )
    return False


async def shutdown_robohat_service():
    """Shutdown global RoboHAT service"""
    global robohat_service

    if robohat_service:
        await robohat_service.shutdown()
        robohat_service = None
