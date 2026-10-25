"""IBT-4 blade: deadman, supervised drop on every stop path, readback state, clear gate (round 2).

PR #20 review BL-01, BL-04, BL-09, BL-10, SUSPECTED S-2, mutants L22 and B09. Everything runs
against the FAKE ``lgpio`` of ``fake_lgpio.py`` (the adapter has never run on hardware):
``fake.pad`` is what a meter on GPIO24/25 would show. SIM_MODE is off so the real adapter is armed.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from types import SimpleNamespace

import pytest
from _tractor_fakes import FakeGpiozero, mowing_tractor
from fake_lgpio import FakeLgpio

from backend.src.core.globals import _safety_state
from backend.src.drivers.blade.ibt4_lgpio import IBT4GpioError, LgpioBladeAdapter
from backend.src.services import blade_service
from backend.src.services.blade_service import BladeService

OFF = {24: 0, 25: 0}
ON = {24: 1, 25: 0}


class Clock:
    """A manual monotonic clock: the deadman is proved without sleeping through it."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    _safety_state["emergency_stop_active"] = False
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: None)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)
    yield
    _safety_state["emergency_stop_active"] = False


def armed(monkeypatch, fake=None, clock=None, **cfg):
    fake = fake or FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    config = {"gpio_enabled": True, "pins": {"in1": 24, "in2": 25}, **cfg}
    return BladeService(config, clock=clock or Clock()), fake


async def eventually(cond, timeout=3.0):
    """Poll ``cond`` (sync or async) until true; the supervisor runs every 50 ms."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        result = cond()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(0.02)


async def on(svc) -> None:
    assert await svc.initialize() is True
    assert await svc.set_active(True) is True


# --------------------------------------------------------------------------- BL-01 deadman


async def test_a_blade_on_that_is_not_repeated_is_switched_off_by_the_deadman(monkeypatch):
    clock = Clock()
    svc, fake = armed(monkeypatch, clock=clock, deadman_ms=1000)
    await on(svc)
    assert fake.pad == ON

    for _ in range(4):  # a client that keeps repeating the ON keeps the blade on
        clock.t += 0.8
        assert await svc.set_active(True) is True
    await asyncio.sleep(0.15)  # several supervision ticks
    assert fake.pad == ON

    clock.t += 1.2  # the client goes silent for longer than the window
    await eventually(lambda: fake.pad == OFF)
    assert "deadman" in svc.last_drop
    status = await svc.status()
    assert status["commanded"] == "off" and status["observed"] == "off"
    assert await svc.set_active(True) is True, "a deadman drop is not a latch: ON works again"


async def test_the_deadman_window_comes_from_the_ibt4_config(monkeypatch):
    svc, _fake = armed(monkeypatch, deadman_ms=500)
    assert svc.deadman_s == 0.5
    assert (await svc.status())["deadman_ms"] == 500
    default, _ = armed(monkeypatch)
    assert default.deadman_s == 2.0, "ibt4.deadman_ms ships as 2000"


# ------------------------------------------------- BL-01 every stop path reaches the IBT-4


@pytest.mark.parametrize("how", ["emergency_stop", "tilt_latch", "park", "lock_out"])
async def test_every_tractor_stop_path_switches_the_ibt4_off(monkeypatch, how):
    FakeGpiozero(monkeypatch)
    tractor, _chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    svc, fake = armed(monkeypatch)
    await on(svc)
    assert fake.pad == ON

    if how == "emergency_stop":  # the watchdog trip, /tractor/emergency-stop, _fail_closed
        await tractor.emergency_stop(source="watchdog")
    elif how == "tilt_latch":
        await tractor.latch_interlock("tilt", source="tilt_monitor")
    elif how == "park":
        await tractor.park("shutdown")
    else:
        tractor.lock_out("safety wiring failed at start-up")

    await eventually(lambda: fake.pad == OFF)
    assert svc.last_drop
    assert await svc.set_active(True) is False, "and the standing reason keeps it off"
    assert svc.last_refusal
    await tractor.shutdown()


async def test_the_process_estop_latch_and_the_robohat_hold_drop_the_blade(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)
    _safety_state["emergency_stop_active"] = True
    await eventually(lambda: fake.pad == OFF)
    assert "emergency stop" in svc.last_drop
    _safety_state["emergency_stop_active"] = False

    robohat = SimpleNamespace(emergency_latched=False)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: robohat)
    assert await svc.set_active(True) is True
    robohat.emergency_latched = True
    await eventually(lambda: fake.pad == OFF)
    assert "RoboHAT" in svc.last_drop


async def test_an_attached_permit_that_raises_forbids_the_blade(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)

    def broken():
        raise RuntimeError("sensor feed unreadable")

    svc.attach_permit("gnss", broken)
    await eventually(lambda: fake.pad == OFF)
    assert "gnss" in svc.last_drop and await svc.set_active(True) is False
    svc.detach_permit("gnss")
    assert await svc.set_active(True) is True


async def test_a_hold_request_drops_the_blade_before_it_returns(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)

    svc.request_blade_off("obstacle hold")  # what the navigation holds call

    assert fake.pad == OFF, "synchronous: confirmed on the pads before the call returns"
    assert svc.last_drop == "obstacle hold"
    assert await svc.set_active(True) is True, "a hold is not a latch: the operator can re-enable"


async def test_an_off_that_cannot_be_confirmed_stays_pending_and_blocks_the_next_on(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)
    fake.hold_pad(24, 1)  # the pad will not go low

    svc.request_blade_off("gnss hold")

    status = await svc.status()
    assert status["off_pending"] == "gnss hold"
    assert status["active"] is True and status["state_confirmed"] is False
    assert await svc.set_active(True) is False
    assert "unconfirmed" in svc.last_refusal

    fake.release_pad(24)  # the supervision loop retries every tick
    await eventually(lambda: fake.pad == OFF)
    await eventually(lambda: svc._off_pending is None)
    assert await svc.set_active(True) is True


async def test_a_hold_request_is_a_noop_when_nothing_was_ever_claimed(monkeypatch):
    svc, fake = armed(monkeypatch)
    svc.request_blade_off("hold before the lines were claimed")
    assert fake.calls == [], "nothing is claimed or driven just because a hold fired"


async def test_a_failed_start_up_switches_the_ibt4_off_too(monkeypatch):
    from backend.src import main

    svc, fake = armed(monkeypatch)
    monkeypatch.setattr(blade_service, "_blade_service", svc)
    await on(svc)

    class Tractor:
        def lock_out(self, why):
            self.why = why

        async def stop_supervision(self):
            pass

        async def emergency_stop(self, source):
            return {"complete": True}

    app = SimpleNamespace(state=SimpleNamespace(tractor_watchdog=None))
    await main._fail_closed(app, Tractor())

    assert fake.pad == OFF


# ------------------------------------------------------------ BL-04 the clear needs a confirmed off


async def test_the_estop_clear_is_refused_while_the_off_is_unconfirmed(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)
    fake.hold_pad(24, 1)
    report = await svc.emergency_stop()
    assert report["off"] is False and svc.emergency_latched is True

    await svc.clear_emergency()

    assert svc.emergency_latched is True, "the latch stays over an output never seen off"
    assert "reads back" in svc.clear_refusal
    assert await svc.set_active(True) is False

    fake.release_pad(24)
    await svc.clear_emergency()
    assert svc.emergency_latched is False and svc.clear_refusal is None
    assert await svc.set_active(True) is True, "a confirmed clear really releases the latch"
    assert fake.pad == ON


async def test_an_on_is_never_written_over_a_pad_that_does_not_read_off(monkeypatch):
    svc, fake = armed(monkeypatch)
    await svc.initialize()
    fake.hold_pad(25, 1)  # something else holds IN2 high while we believe the blade is off

    assert await svc.set_active(True) is False

    assert "could not be confirmed" in svc.last_refusal
    assert ("gpio_write", 7, 24, 1) not in fake.calls, "no ON write went out"


# ------------------------------------------------------------- BL-10 the state is a measurement


async def test_the_status_reads_the_pads_instead_of_a_cache(monkeypatch):
    svc, fake = armed(monkeypatch)
    await on(svc)
    assert await svc.set_active(False) is True
    status = await svc.status()
    assert (status["observed"], status["pad"]) == ("off", {"in1": 0, "in2": 0})
    assert status["active"] is False and status["state_confirmed"] is True

    fake.hold_pad(24, 1)  # forced high behind our back: no command was issued

    status = await svc.status()
    assert status["observed"] == "on" and status["pad"] == {"in1": 1, "in2": 0}
    assert status["active"] is True and status["state_confirmed"] is False


async def test_the_status_of_an_armed_service_that_could_not_claim_its_lines_is_unknown(
    monkeypatch,
):
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    svc, _ = armed(monkeypatch, fake=fake)
    await svc.initialize()
    status = await svc.status()
    assert status["observed"] == "unobservable" and status["pad"] is None
    assert status["active"] is True and status["state_confirmed"] is False
    assert "GPIO25" in status["last_error"]


# ---------------------------------------------------------- BL-09 start-up and park latch


async def test_start_up_claims_both_lines_low_and_the_shutdown_park_latches(monkeypatch):
    svc, fake = armed(monkeypatch)

    report = await svc.startup()

    assert report["off"] is True and fake.claimed == {24, 25} and fake.pad == OFF
    assert ("gpio_claim_output", 7, 24, 0) in fake.calls
    assert ("gpio_claim_output", 7, 25, 0) in fake.calls
    assert await svc.set_active(True) is True

    parked = await svc.park()

    assert parked["off"] is True and parked["latched"] is False and fake.pad == OFF
    assert await svc.set_active(True) is False and "parked" in svc.last_refusal
    assert fake.pad == OFF, "a late ON after the shutdown park never reaches the pins"
    await svc.startup()  # a fresh bring-up lifts the park
    assert await svc.set_active(True) is True


async def test_start_up_of_an_unarmed_service_touches_nothing(monkeypatch):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    svc = BladeService({"gpio_enabled": False})
    await svc.startup()
    assert fake.calls == []


# ---------------------------------------- SUSPECTED S-2: no silent fall back to another gpiochip


async def test_a_chip_that_cannot_be_opened_is_not_replaced_by_another_bank(monkeypatch):
    class Chip4Dead(FakeLgpio):
        def gpiochip_open(self, chip):
            if chip == 4:
                self.calls.append(("gpiochip_open", 4))
                return self._fail("can not open gpiochip 4")
            return super().gpiochip_open(chip)

    fake = Chip4Dead()
    svc, _ = armed(monkeypatch, fake=fake)

    report = await svc.emergency_stop()

    assert report["off"] is False and report["hardware_backed"] is False
    assert "gpiochip" in report["gpio_error"]
    assert ("gpiochip_open", 0) not in fake.calls, "the readback of chip 0 would prove nothing"


# ------------------------------------------------------------------ mutant L22 (adapter confirm)


def test_an_on_write_that_never_reaches_the_pad_is_not_confirmed():
    fake = FakeLgpio()
    adapter = LgpioBladeAdapter(24, 25, (1, 0), lgpio_module=fake)
    adapter.setup()
    fake.hold_pad(24, 0)  # IN1 cannot rise: the write returns success, the pad stays low

    with pytest.raises(IBT4GpioError, match="reads back 0, expected 1"):
        adapter.set_active(True)

    assert adapter.is_active() is False
    assert fake.pad == OFF, "the half-applied ON was rolled back"
