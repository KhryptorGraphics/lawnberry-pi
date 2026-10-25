"""HW-04 (PR #20 review 3): ``collect_blade_active()`` reads the pads and the RoboHAT blade.

It feeds ``GET /dashboard/status`` (``blade_active``). It used to read two private cache flags of
the IBT-4 driver (a pad forced high by something else, or a failed readback, still said "off") and
did not consult the RoboHAT blade, the blade of the deployed differential platform, at all. The
IBT-4 runs on the FAKE ``lgpio`` of ``fake_lgpio.py``; the RoboHAT is the real ``RoboHATService``
over a fake serial wire.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _blade_fakes import armed, blade_world, fake_robohat, on
from fake_lgpio import FakeLgpio

from backend.src.api.routers.sensors import dashboard_status
from backend.src.core.globals import _safety_state
from backend.src.core.safety_status import collect_blade_active
from backend.src.services import blade_service
from backend.src.services.blade_service import BladeService
from backend.src.services.robohat_service import RoboHATService


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    blade_world(monkeypatch)
    monkeypatch.setattr(blade_service, "_blade_service", None)
    yield
    _safety_state["emergency_stop_active"] = False


def install(monkeypatch, svc: BladeService) -> BladeService:
    """``svc`` is the process-wide blade service the status reads."""
    monkeypatch.setattr(blade_service, "_blade_service", svc)
    return svc


def with_robohat(monkeypatch, robohat) -> None:
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: robohat)


# ---------------------------------------------------------------------------- the IBT-4 pads


async def test_an_idle_ibt4_blade_reads_inactive(monkeypatch):
    svc, _ = armed(monkeypatch)
    install(monkeypatch, svc)
    assert await svc.initialize() is True

    assert collect_blade_active() is False
    assert svc.blade_may_be_on() is False


async def test_a_pad_forced_high_by_something_else_reads_active(monkeypatch):
    """The repro of the finding: IN1 held high, no command issued: the cache said "off"."""
    svc, fake = armed(monkeypatch)
    install(monkeypatch, svc)
    assert await svc.initialize() is True
    assert collect_blade_active() is False

    fake.hold_pad(24, 1)

    status = await svc.status()
    assert status["observed"] == "on" and status["active"] is True
    assert status["state_confirmed"] is False
    assert collect_blade_active() is True
    fake.release_pad(24)
    assert collect_blade_active() is False


async def test_a_running_blade_reads_active_and_a_confirmed_off_reads_inactive(monkeypatch):
    svc, _ = armed(monkeypatch)
    install(monkeypatch, svc)
    await on(svc)
    assert collect_blade_active() is True

    assert await svc.set_active(False) is True

    assert collect_blade_active() is False


async def test_pads_that_cannot_be_read_read_active(monkeypatch):
    svc, fake = armed(monkeypatch)
    install(monkeypatch, svc)
    assert await svc.initialize() is True

    fake.fail_read(25, "read path dead")

    assert collect_blade_active() is True, "a blade that cannot be seen to be off may be on"
    fake.clear_read_fault(25)
    assert collect_blade_active() is False


async def test_an_armed_blade_whose_lines_were_never_claimed_reads_active(monkeypatch):
    """Armed but the claim failed (GPIO25 held by the HaLow HAT): the pads' state is unknown."""
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    svc, _ = armed(monkeypatch, fake=fake)
    install(monkeypatch, svc)
    assert await svc.initialize() is False

    assert collect_blade_active() is True


async def test_the_flag_only_stub_reports_what_was_commanded(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "1")
    svc = install(monkeypatch, BladeService({}))
    assert collect_blade_active() is False

    assert await svc.set_active(True) is True
    assert collect_blade_active() is True

    assert await svc.set_active(False) is True
    assert collect_blade_active() is False


async def test_the_driver_snapshot_is_the_synchronous_health_check(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)
    fake.hold_pad(25, 1)

    assert svc._driver.snapshot() == await svc._driver.health_check()


# ------------------------------------------------------------- the other blades, and failures


def test_a_tractor_pto_that_is_engaged_reads_active(monkeypatch):
    tractor = SimpleNamespace(state=SimpleNamespace(blade_engaged=True))
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    assert collect_blade_active() is True

    tractor.state.blade_engaged = False
    assert collect_blade_active() is False


def test_a_robohat_blade_that_was_switched_on_reads_active(monkeypatch):
    robohat = fake_robohat(blade_on=True)
    with_robohat(monkeypatch, robohat)
    assert collect_blade_active() is True

    robohat.blade_commanded_on = False
    assert collect_blade_active() is False


@pytest.mark.parametrize("source", ["tractor", "robohat", "ibt4"])
def test_a_source_that_cannot_be_read_counts_as_a_blade_that_may_be_on(monkeypatch, source):
    def broken(*_args, **_kwargs):
        raise RuntimeError("source unreadable")

    target = {
        "tractor": "backend.src.services.tractor_service.get_tractor_service",
        "robohat": "backend.src.services.robohat_service.get_robohat_service",
        "ibt4": "backend.src.services.blade_service.get_blade_service",
    }[source]
    assert collect_blade_active() is False
    monkeypatch.setattr(target, broken)

    assert collect_blade_active() is True


async def test_the_dashboard_route_reports_the_pads(monkeypatch):
    svc, fake = armed(monkeypatch)
    install(monkeypatch, svc)
    assert await svc.initialize() is True
    assert dashboard_status().blade_active is False

    fake.hold_pad(24, 1)

    assert dashboard_status().blade_active is True


# ------------------------------------------------------- the RoboHAT records what it was told


class _Wire:
    """A serial port whose writes can fail; the firmware echoes pwm and blade lines at once."""

    is_open = True
    in_waiting = 0

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.fail = False
        self.svc: RoboHATService | None = None

    def write(self, data: bytes) -> int:
        if self.fail:
            raise OSError("wire cut")
        line = data.decode().strip()
        self.lines.append(line)
        if self.svc is not None:  # the firmware's acknowledgement
            kind = "blade" if line.startswith("blade=") else "pwm"
            self.svc._acks[kind] += 1
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


def robohat_on_a_wire(monkeypatch) -> tuple[RoboHATService, _Wire]:
    svc = RoboHATService()
    wire = _Wire()
    wire.svc = svc
    svc.serial_conn = wire
    svc.running = True
    svc._rc_enabled = False

    async def usb_control(**_kw) -> bool:
        svc._usb_control_requested = True
        return True

    monkeypatch.setattr(svc, "_ensure_usb_control", usb_control)
    return svc, wire


async def test_the_robohat_remembers_the_last_blade_line_it_wrote(monkeypatch):
    svc, wire = robohat_on_a_wire(monkeypatch)
    with_robohat(monkeypatch, svc)
    assert svc.blade_commanded_on is False and collect_blade_active() is False

    assert await svc.send_blade_command(True) is True

    assert wire.lines == ["blade=on"]
    assert svc.blade_commanded_on is True and collect_blade_active() is True
    assert await svc.send_blade_command(False) is True
    assert svc.blade_commanded_on is False and collect_blade_active() is False


async def test_an_on_write_that_failed_may_have_landed_so_the_blade_reads_on(monkeypatch):
    svc, wire = robohat_on_a_wire(monkeypatch)
    wire.fail = True

    assert await svc.send_blade_command(True) is False

    assert svc.blade_commanded_on is True, "unknown is not off"
    wire.fail = False
    assert await svc.send_blade_command(False) is True
    assert svc.blade_commanded_on is False


async def test_an_off_write_that_failed_does_not_clear_the_blade(monkeypatch):
    svc, wire = robohat_on_a_wire(monkeypatch)
    assert await svc.send_blade_command(True) is True
    wire.fail = True

    assert await svc.send_blade_command(False) is False

    assert svc.blade_commanded_on is True, "the blade was not told to stop"


async def test_an_on_that_the_hold_refuses_never_reaches_the_wire_or_the_record(monkeypatch):
    svc, wire = robohat_on_a_wire(monkeypatch)
    svc._estop_hold = True

    assert await svc.send_blade_command(True) is False

    assert wire.lines == [] and svc.blade_commanded_on is False


@pytest.mark.parametrize("how", ["emergency_stop_detail", "park"])
async def test_a_stop_that_wrote_blade_off_clears_the_record(monkeypatch, how):
    svc, wire = robohat_on_a_wire(monkeypatch)
    assert await svc.send_blade_command(True) is True

    await getattr(svc, how)()

    assert "blade=off" in wire.lines
    assert svc.blade_commanded_on is False
