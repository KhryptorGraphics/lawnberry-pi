"""RoboHAT drive-command validity: the host must not defeat the firmware failsafe.

The firmware falls back to RC mode after ~2 s of USB silence and, with no RC
signal, to neutral. The backend used to re-send the *last PWM* every 0.9 s
forever (and re-take USB control whenever the firmware fell back), so a client
that vanished mid-command left the machine on that command for as long as the
process lived. Now a command is refreshed only inside its validity window; past
it the service sends neutral once and lets the firmware failsafe govern.
"""

from __future__ import annotations

import asyncio
import time

from backend.src.services.robohat_service import RoboHATService


class _Serial:
    is_open = True
    in_waiting = 0

    def close(self) -> None:
        self.is_open = False


def _svc(monkeypatch, *, echo: bool = True, write_ok: bool = True):
    """A service on a fake serial line that echoes like the real firmware."""
    svc = RoboHATService()
    svc.serial_conn = _Serial()
    svc.running = True
    svc._rc_enabled = False
    sent: list[str] = []

    async def send_line(line: str) -> bool:
        sent.append(line)
        if not write_ok:
            return False
        if echo:
            if line.startswith("pwm,"):
                svc._process_line("[USB] PWM set → steer=1500 µs throttle=1500 µs")
            elif line.startswith("blade="):
                svc._process_line("[USB] Blade disabled")
        return True

    async def ensure_usb(**_kw) -> bool:
        svc._usb_control_requested = True
        return True

    monkeypatch.setattr(svc, "_send_line", send_line)
    monkeypatch.setattr(svc, "_ensure_usb_control", ensure_usb)
    return svc, sent


async def test_a_vanished_client_gets_neutral_once_then_the_firmware_failsafe(monkeypatch):
    svc, sent = _svc(monkeypatch)
    set_rc: list[bool] = []

    async def fake_set_rc(enabled: bool, *, force: bool = False) -> None:
        set_rc.append(enabled)

    monkeypatch.setattr(svc, "_set_rc_enabled", fake_set_rc)

    assert await svc.send_motor_command(0.8, 0.8, valid_for_s=0.1) is True
    assert sent == ["pwm,1500,1860"], "the driving command went out"

    await asyncio.sleep(0.3)  # the client is gone; the window lapsed
    assert sent == ["pwm,1500,1860", "pwm,1500,1500"], "exactly one neutral, no repeats"

    # The maintain loop (1 Hz in production) must now do NOTHING: no refresh of the
    # stale command, no re-taking of USB control after the firmware falls back to RC.
    svc._rc_enabled = True  # firmware timed out to RC mode
    for _ in range(3):
        svc._last_status_at = 0.0
        svc._last_pwm_at = time.monotonic() - 5.0
        await svc._maintain_usb_control()
    assert sent == ["pwm,1500,1860", "pwm,1500,1500"]
    assert set_rc == [], "must not re-disable RC: that would defeat the firmware failsafe"
    assert svc._usb_control_requested is False
    await svc.shutdown()


async def test_the_current_command_is_refreshed_only_while_inside_its_window(monkeypatch):
    svc, sent = _svc(monkeypatch)
    svc._usb_control_requested = True
    svc._last_pwm = (1500, 1950)
    svc._last_pwm_at = time.monotonic() - 1.0
    svc._last_status_at = 0.0

    svc._cmd_valid_until = time.monotonic() + 5.0  # a 5 s command, 1 s in
    await svc._maintain_usb_control()
    assert sent == ["pwm,1500,1950"], "long commands must outlive the firmware's 2 s timeout"

    sent.clear()
    svc._last_pwm_at = time.monotonic() - 1.0
    svc._cmd_valid_until = time.monotonic() - 0.01  # window just lapsed
    await svc._maintain_usb_control()
    assert sent == []


async def test_a_refreshed_command_keeps_the_window_open(monkeypatch):
    svc, sent = _svc(monkeypatch)
    for _ in range(8):  # a caller driving at ~20 Hz with 100 ms commands
        assert await svc.send_motor_command(0.5, 0.5, valid_for_s=0.1)
        await asyncio.sleep(0.03)
    assert all(line != "pwm,1500,1500" for line in sent), "no neutral while it keeps commanding"

    await asyncio.sleep(0.25)
    assert sent[-1] == "pwm,1500,1500"
    await svc.shutdown()


async def test_default_validity_and_env_override(monkeypatch):
    assert RoboHATService()._command_validity_s == 0.5
    monkeypatch.setenv("ROBOHAT_COMMAND_VALIDITY_S", "1.5")
    assert RoboHATService()._command_validity_s == 1.5
    monkeypatch.setenv("ROBOHAT_COMMAND_VALIDITY_S", "9999")
    assert RoboHATService()._command_validity_s == 10.0, "clamped to the ceiling"


async def test_a_neutral_command_needs_no_window(monkeypatch):
    svc, sent = _svc(monkeypatch)
    await svc.send_motor_command(0.5, 0.5, valid_for_s=0.2)
    await svc.send_motor_command(0.0, 0.0)
    assert svc._cmd_valid_until == 0.0
    sent.clear()
    await asyncio.sleep(0.4)
    assert sent == [], "no expiry neutral after an explicit neutral"
    await svc.shutdown()


# ------------------------------ E-stop hold and report ------------------------------


async def test_estop_holds_neutral_and_refuses_commands_until_cleared(monkeypatch):
    svc, sent = _svc(monkeypatch)
    await svc.send_motor_command(0.8, 0.8, valid_for_s=5.0)

    detail = await svc.emergency_stop_detail()

    assert detail["drive"] == "acked" and detail["blade"] == "acked"
    assert detail["blade"] == "acked" and sent[-2:] == ["blade=off", "pwm,1500,1500"]
    assert svc.emergency_latched is True and svc._cmd_valid_until == 0.0

    # New drive / blade-on commands are refused while latched...
    assert await svc.send_motor_command(0.5, 0.5) is False
    assert svc.status.last_error == "emergency_stop_latched"
    assert await svc.send_blade_command(True) is False
    # ...and the hold keeps neutral alive so RC/anything else cannot undo the stop.
    svc._usb_control_requested = True
    svc._last_status_at = 0.0
    svc._last_pwm_at = time.monotonic() - 5.0
    sent.clear()
    await svc._maintain_usb_control()
    assert sent == ["pwm,1500,1500"]

    assert await svc.clear_emergency() is True
    assert svc.emergency_latched is False
    assert await svc.send_motor_command(0.5, 0.5) is True
    await svc.shutdown()


async def test_estop_detail_says_when_the_firmware_did_not_accept_the_stop(monkeypatch):
    svc, _ = _svc(monkeypatch, echo=False)  # firmware in RC mode: rejects/ignores everything

    detail = await svc.emergency_stop_detail()

    assert detail["blade"] == "unacked" and detail["drive"] == "unacked"
    assert detail["usb_control"] is True  # (patched) but the stop still was not echoed
    assert await svc.emergency_stop() is True, "legacy bool = 'was it written', unchanged"
    await svc.shutdown()


async def test_estop_detail_reports_a_dead_serial_port(monkeypatch):
    svc, _ = _svc(monkeypatch, write_ok=False)
    detail = await svc.emergency_stop_detail()
    assert detail["drive"] == "send_failed" and detail["blade"] == "send_failed"
    assert detail["error"] == "serial write failed"
    assert await svc.emergency_stop() is False

    svc.serial_conn = None
    assert (await svc.emergency_stop_detail())["error"] == "serial not connected"


async def test_park_sends_neutral_and_blade_off_without_latching(monkeypatch):
    svc, sent = _svc(monkeypatch)
    await svc.send_motor_command(0.8, 0.8, valid_for_s=5.0)
    sent.clear()

    detail = await svc.park()

    assert detail == {"drive": "acked", "blade": "acked", "error": None}
    assert sent == ["blade=off", "pwm,1500,1500"]
    assert svc.emergency_latched is False and svc._cmd_valid_until == 0.0
    await svc.shutdown()
