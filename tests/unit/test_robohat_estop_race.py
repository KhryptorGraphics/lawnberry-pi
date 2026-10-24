"""RoboHAT E-stop races and command windows (audit SF-02, SF-07, SF-10, SF-12, park latch).

SF-02 (S1): a drive / blade-on command that had passed the E-stop-hold check landed AFTER the stop
while it waited for the RC->USB handshake, and the hold then re-sent that forward PWM. SF-10: a
hold whose USB maintenance had already been dropped re-sent nothing. SF-07: a command window
shorter than the one the expiry task slept towards was ignored. SF-12: NaN mixed to full scale.
The firmware is a fake whose ``rc=disable`` handshake takes ``handshake_s`` to be acknowledged;
the service's own ``_ensure_usb_control`` / ``_send_confirmed`` / maintenance code runs for real.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.src.services.robohat_service import RoboHATService

NAN, INF = float("nan"), float("inf")


class _Serial:
    is_open = True
    in_waiting = 0

    def close(self) -> None:
        self.is_open = False


class FakeFirmware:
    """Records every line sent; acks pwm/blade at once and ``rc=disable`` after ``handshake_s``."""

    def __init__(self, svc: RoboHATService, handshake_s: float = 0.0) -> None:
        self.svc, self.handshake_s = svc, handshake_s
        self.t0 = time.monotonic()
        self.sent: list[tuple[float, str]] = []

    @property
    def lines(self) -> list[str]:
        return [line for _, line in self.sent]

    async def send_line(self, line: str) -> bool:
        self.sent.append((time.monotonic() - self.t0, line))
        if line == "rc=disable":
            asyncio.get_running_loop().call_later(
                self.handshake_s, self.svc._process_line, "[USB] RC disabled: USB control"
            )
        elif line.startswith("pwm,"):
            self.svc._process_line("[USB] PWM set -> steer=1500 us throttle=1500 us")
        elif line.startswith("blade="):
            self.svc._process_line("[USB] Blade disabled")
        return True


def make_service(monkeypatch, *, handshake_s: float = 0.0, rc_mode: bool = False):
    svc = RoboHATService()
    svc.serial_conn = _Serial()
    svc.running = True
    svc._rc_enabled = rc_mode  # True: the firmware fell back to RC mode and must handshake
    fw = FakeFirmware(svc, handshake_s)
    monkeypatch.setattr(svc, "_send_line", fw.send_line)
    return svc, fw


# ----------------------------------------- SF-02 -----------------------------------------


async def test_a_drive_command_waiting_for_the_usb_handshake_cannot_land_after_the_estop(
    monkeypatch,
):
    svc, fw = make_service(monkeypatch, handshake_s=0.3, rc_mode=True)
    cmd = asyncio.create_task(svc.send_motor_command(0.8, 0.8, valid_for_s=5.0))
    await asyncio.sleep(0.1)  # passed the hold check, waiting for the handshake

    detail = await svc.emergency_stop_detail()
    accepted = await asyncio.wait_for(cmd, 3)

    assert detail["drive"] == "acked" and detail["blade"] == "acked"
    assert accepted is False and svc.status.last_error == "emergency_stop_latched"
    assert "pwm,1500,1860" not in fw.lines, "the forward PWM must never reach the firmware"
    assert svc._last_pwm == (1500, 1500) and svc._cmd_valid_until == 0.0
    await svc.shutdown()


async def test_a_blade_on_waiting_for_the_usb_handshake_cannot_land_after_the_estop(monkeypatch):
    svc, fw = make_service(monkeypatch, handshake_s=0.3, rc_mode=True)
    cmd = asyncio.create_task(svc.send_blade_command(True))
    await asyncio.sleep(0.1)

    await svc.emergency_stop_detail()
    accepted = await asyncio.wait_for(cmd, 3)

    assert accepted is False and "blade=on" not in fw.lines
    assert fw.lines.count("blade=off") >= 1
    await svc.shutdown()


async def test_the_hold_resends_neutral_never_a_stale_command(monkeypatch):
    svc, fw = make_service(monkeypatch)
    await svc.emergency_stop_detail()
    svc._usb_control_requested = True
    svc._last_pwm = (1500, 1860)  # whatever overwrote the neutral
    svc._last_pwm_at = time.monotonic() - 5.0
    svc._last_status_at = 0.0
    fw.sent.clear()

    await svc._maintain_usb_control()

    assert fw.lines == ["pwm,1500,1500"], "the 'hold neutral' must be the constant neutral"
    await svc.shutdown()


# ----------------------------------------- SF-10 -----------------------------------------


async def test_the_hold_keeps_neutral_alive_even_after_usb_maintenance_was_dropped(monkeypatch):
    svc, fw = make_service(monkeypatch)
    svc._usb_control_requested = False  # the 1 Hz watchdog already dropped it (command lapsed)

    await svc.emergency_stop_detail()
    assert svc.emergency_latched is True
    svc._last_pwm_at = time.monotonic() - 5.0
    svc._last_status_at = 0.0
    fw.sent.clear()
    await svc._maintain_usb_control()

    assert fw.lines == ["pwm,1500,1500"], "the hold must not depend on the request flag"
    await svc.shutdown()


# ----------------------------------------- SF-07 -----------------------------------------


async def test_a_shorter_command_window_pulls_the_expiry_forward(monkeypatch):
    svc, fw = make_service(monkeypatch)
    await svc.send_motor_command(0.5, 0.5, valid_for_s=0.6)
    await asyncio.sleep(0.1)
    await svc.send_motor_command(0.5, 0.5, valid_for_s=0.1)  # a shorter window than the first
    second_at = time.monotonic() - fw.t0
    await asyncio.sleep(0.6)

    neutral_at = next(t for t, line in fw.sent if line == "pwm,1500,1500")

    assert neutral_at < second_at + 0.3, (
        f"the 0.1 s command must expire ~0.1 s after it was sent, not at the first deadline "
        f"(neutral at {neutral_at:.2f}s, command at {second_at:.2f}s)"
    )
    await svc.shutdown()


async def test_the_command_window_is_not_defeated_by_a_wall_clock_step_backwards(monkeypatch):
    svc, fw = make_service(monkeypatch)
    await svc.send_motor_command(0.5, 0.5, valid_for_s=0.15)
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 3600.0)

    await asyncio.sleep(0.4)

    assert "pwm,1500,1500" in fw.lines, "the window runs on the monotonic clock"
    await svc.shutdown()


# ----------------------------------------- SF-12 -----------------------------------------


@pytest.mark.parametrize("bad", [(NAN, NAN), (NAN, 0.5), (0.5, INF), (-INF, INF)])
def test_mixing_refuses_non_finite_wheel_speeds(bad):
    with pytest.raises(ValueError, match="non-finite"):
        RoboHATService._mix_arcade_to_pwm(*bad)


async def test_a_non_finite_drive_command_is_refused_and_nothing_is_sent(monkeypatch):
    svc, fw = make_service(monkeypatch)

    assert await svc.send_motor_command(NAN, NAN) is False

    assert fw.lines == [] and svc.status.last_error == "non_finite_command"
    await svc.shutdown()


# ------------------------------------- shutdown park latch -------------------------------------


async def test_a_parked_robohat_accepts_only_neutral_and_blade_off(monkeypatch):
    svc, fw = make_service(monkeypatch)
    await svc.send_motor_command(0.5, 0.5, valid_for_s=5.0)

    detail = await svc.park()
    fw.sent.clear()

    assert detail["drive"] == "acked" and svc.emergency_latched is False
    assert await svc.send_motor_command(0.5, 0.5) is False and svc.status.last_error == "parked"
    assert await svc.send_blade_command(True) is False
    assert await svc.send_motor_command(0.0, 0.0) is True, "neutral still passes"
    assert await svc.send_blade_command(False) is True, "blade-off still passes"
    assert fw.lines == ["pwm,1500,1500", "blade=off"]
    await svc.shutdown()
