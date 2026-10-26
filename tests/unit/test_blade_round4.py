"""IBT-4 blade service, review round 4: the surviving mutants of the service (B01..B45).

PR #20 review 3 (hardware), HW-07. The mutation run found these gaps in the tests of
``blade_service.py`` (the mutant id is in each test's docstring; HW-01 has its own file,
``test_blade_failed_on.py``). Everything runs against the FAKE ``lgpio`` of ``fake_lgpio.py``:
``fake.pad`` is the last level written to each pin and ``hold_pad`` makes a pin ignore writes and
read a fixed level, like a pin held by something else.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import pytest
from _blade_fakes import OFF, ON, Clock, armed, blade_world, eventually, on
from fake_lgpio import FakeLgpio

from backend.src.core.globals import _safety_state
from backend.src.services import blade_service
from backend.src.services.blade_service import BladeService


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    blade_world(monkeypatch)
    yield
    _safety_state["emergency_stop_active"] = False


def _off_writes(fake: FakeLgpio) -> int:
    """How many times an OFF level went out to IN1."""
    return sum(1 for write in fake.writes() if write == (24, 0))


async def _pending_cleared(svc: BladeService) -> bool:
    return (await svc.status())["off_pending"] is None


# -------------------------------------------------------------------------- B33: E-stop OFF


async def test_an_estop_whose_off_cannot_be_confirmed_stays_pending_and_is_retried(monkeypatch):
    """B33: an unconfirmed E-stop OFF was forgotten instead of pending (nobody retried it)."""
    svc, fake = armed(monkeypatch)
    await on(svc)
    fake.hold_pad(24, 1)  # something else holds IN1 high: the OFF cannot be confirmed

    report = await svc.emergency_stop()

    assert report["off"] is False and report["latched"] is True
    status = await svc.status()
    assert status["off_pending"] == "emergency stop"
    assert status["supervising"] is True
    assert status["active"] is True and status["state_confirmed"] is False
    seen = _off_writes(fake)
    await eventually(lambda: _off_writes(fake) >= seen + 2)  # the supervisor keeps writing OFF
    fake.release_pad(24)
    await eventually(lambda: _pending_cleared(svc))
    assert (await svc.status())["observed"] == "off"


# ---------------------------------------------------------------- B29, B28: the pads vs the command


async def test_a_blade_whose_pad_readback_starts_failing_is_dropped(monkeypatch):
    """B29: with the readback dead the blade is no longer known to be on: it is switched off."""
    svc, fake = armed(monkeypatch, deadman_ms=10_000)
    await on(svc)

    fake.fail_read(25, "read path dead")

    await eventually(lambda: svc.last_drop is not None)
    assert "read error" in svc.last_drop
    assert fake.pad == OFF, "an OFF went out although it cannot be confirmed"
    status = await svc.status()
    assert status["off_pending"] is not None and status["state_confirmed"] is False
    fake.clear_read_fault(25)
    await eventually(lambda: _pending_cleared(svc))
    assert (await svc.status())["observed"] == "off"


async def test_a_blade_whose_pads_read_off_while_commanded_on_is_dropped(monkeypatch):
    """B28: bookkeeping said ON while both pads read low: the service must not keep believing it."""
    svc, fake = armed(monkeypatch, deadman_ms=10_000)
    await on(svc)

    fake.hold_pad(24, 0)  # IN1 reads low although it was driven high

    await eventually(lambda: svc.last_drop is not None)
    assert svc.last_drop.startswith("blade inputs read off while commanded on")
    assert (await svc.status())["commanded"] == "off"


async def test_a_deadman_refresh_over_pads_that_do_not_read_on_is_refused_and_drops_the_blade(
    monkeypatch,
):
    """B04/B41: the refresh answered "accepted" for a blade whose pad reads off (the supervisor
    only dropped it up to 50 ms later). The refresh itself must say no, and why."""
    svc, fake = armed(monkeypatch, deadman_ms=10_000)
    await on(svc)
    fake.hold_pad(24, 0)

    assert await svc.set_active(True) is False  # no event-loop turn between the pad and the check

    assert svc.last_refusal == "blade inputs read off while the blade is commanded on"
    assert svc.last_drop == svc.last_refusal
    assert (await svc.status())["commanded"] == "off"


# ------------------------------------------------------------------------- B06: ON over a pad


@pytest.mark.parametrize(
    ("pin", "level", "reads"),
    [(24, 1, "on"), (25, 1, "mixed"), (None, None, "error")],
    ids=["IN1 already at the ON level", "IN2 held high", "readback failing"],
)
async def test_an_on_is_never_written_over_a_pad_that_does_not_read_off(
    monkeypatch, pin, level, reads
):
    """B06: a pad already forced to the ON levels (a short to 3.3 V) accepted an ON."""
    svc, fake = armed(monkeypatch)
    assert await svc.initialize() is True
    if pin is None:
        fake.fail_read(25, "read path dead")
    else:
        fake.hold_pad(pin, level)
    assert (await svc.status())["observed"] == reads

    assert await svc.set_active(True) is False

    assert "could not be confirmed" in svc.last_refusal
    assert ("gpio_write", 7, 24, 1) not in fake.calls, "no ON write went out"
    assert (await svc.status())["commanded"] == "off"


# ---------------------------------------------------------------------- B27, B01, B02: deadman


async def test_the_deadman_drops_the_blade_exactly_at_the_end_of_the_window(monkeypatch):
    """B27: ``>=`` against the deadline: at the deadline itself the blade goes."""
    clock = Clock()
    svc, fake = armed(monkeypatch, clock=clock, deadman_ms=1000)
    await on(svc)
    deadline = clock.t + svc.deadman_s

    clock.t = deadline - 0.001
    await asyncio.sleep(0.2)  # several supervision ticks
    assert fake.pad == ON, "inside the window the blade stays on"

    clock.t = deadline
    await eventually(lambda: fake.pad == OFF, timeout=1.5)
    assert "deadman" in svc.last_drop


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        (50, 0.2),
        (0, 0.2),
        (200, 0.2),
        (2000, 2.0),
        (10_000, 10.0),
        (100_000, 10.0),
        (1e12, 10.0),
        (None, 2.0),
        ("1000", 2.0),
        (True, 2.0),
        (float("nan"), 2.0),
        (float("inf"), 2.0),
    ],
)
def test_the_deadman_window_is_clamped_whatever_the_config_dict_says(value, seconds):
    """B01/B02: a service built straight from a dict (not through ``Ibt4Config``) can still not
    get a window below 0.2 s, above 10 s, or a garbage one."""
    assert BladeService({"deadman_ms": value}).deadman_s == seconds


# ---------------------------------------------------------------------- B37, B09: the refusal


async def test_the_status_says_why_an_on_would_be_refused(monkeypatch):
    """B37: ``status()["refusal"]`` is documented (OPERATIONS.md) and was never asserted."""
    svc, fake = armed(monkeypatch)
    assert await svc.initialize() is True
    assert (await svc.status())["refusal"] is None, "nothing stands in the way"

    await svc.emergency_stop()
    assert "E-stop latched" in (await svc.status())["refusal"]

    await svc.clear_emergency()
    assert (await svc.status())["refusal"] is None
    await svc.park()
    assert "parked" in (await svc.status())["refusal"]

    await svc.startup()
    fake.hold_pad(24, 1)
    assert await svc.set_active(False) is False  # an OFF that cannot be confirmed stays pending
    assert "still unconfirmed" in (await svc.status())["refusal"]

    _safety_state["emergency_stop_active"] = True
    fake.release_pad(24)
    await eventually(lambda: _pending_cleared(svc))
    assert "emergency stop latched" in (await svc.status())["refusal"]


async def test_an_on_over_the_estop_latch_names_the_latch(monkeypatch):
    """B09: without the explicit check the operator was told "ON not confirmed" instead."""
    svc, fake = armed(monkeypatch)
    assert await svc.initialize() is True
    await svc.emergency_stop()
    writes = len(fake.writes())

    assert await svc.set_active(True) is False

    assert "E-stop latched" in svc.last_refusal and "operator clear" in svc.last_refusal
    assert len(fake.writes()) == writes, "not a single GPIO write for a refused ON"


# ---------------------------------------------------------------- B26: a latch set on the driver


async def test_a_latch_set_on_the_driver_drops_a_running_blade(monkeypatch):
    """B26: the latch lives on the driver; whoever sets it (not only ``emergency_stop``) must
    stop a blade that is running."""
    svc, fake = armed(monkeypatch, deadman_ms=10_000)
    await on(svc)

    svc._driver.latch_estop()

    await eventually(lambda: fake.pad == OFF, timeout=1.5)
    assert svc.last_drop == "E-stop latched"


# ---------------------------------------------------------------- B44: a hold after a bad write


async def test_a_hold_request_after_an_unconfirmed_write_drives_the_off_again(monkeypatch):
    """B44: a blade that is commanded off and reads off is left alone, but not while the last
    write was unconfirmed: that hold must try the OFF (and keep trying while it fails)."""
    svc, fake = armed(monkeypatch)
    await on(svc)
    assert (await svc.emergency_stop())["off"] is True
    fake.fail_write(24, "rc")  # the write returns an error code, the pad stays low
    await svc.clear_emergency()
    assert svc.emergency_latched is True and svc._driver.state_confirmed is False
    assert (await svc.status())["off_pending"] is None

    svc.request_blade_off("obstacle hold")

    status = await svc.status()
    assert status["off_pending"] == "obstacle hold", "the hold tried an OFF and could not confirm"
    fake.clear_write_faults()
    await eventually(lambda: _pending_cleared(svc))
    assert svc._driver.state_confirmed is True


# ----------------------------------------------------------- B21: nothing claimed, nothing pending


async def test_an_off_over_lines_that_were_never_claimed_is_not_a_pending_off(monkeypatch):
    """B21: with nothing claimed there is nothing to retry: no pending OFF, no supervisor."""
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    svc, _ = armed(monkeypatch, fake=fake)

    assert await svc.set_active(False) is False

    status = await svc.status()
    assert status["initialized"] is False and "GPIO25" in status["last_error"]
    assert status["off_pending"] is None and status["supervising"] is False


# --------------------------------------------------------------------- B45: start-up reporting


async def test_start_up_that_could_not_claim_the_lines_says_so(monkeypatch, caplog):
    """B45: ``startup()`` reported ``off: True`` although the claim failed."""
    fake = FakeLgpio()
    fake.fail_claim(25, "GPIO busy")
    svc, _ = armed(monkeypatch, fake=fake)

    with caplog.at_level(logging.INFO, logger=blade_service.logger.name):
        report = await svc.startup()

    assert report["off"] is False and report["initialized"] is False
    assert "GPIO25" in report["gpio_error"]
    critical = [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]
    assert any("NOT confirmed off at start-up" in text for text in critical)
    assert not any("claimed at start-up" in r.getMessage() for r in caplog.records)


# ------------------------------------------------ B07: an ON nobody can supervise is not left on


def test_an_on_that_cannot_be_supervised_is_never_left_on(monkeypatch):
    """B07: the synchronous ON path has no running loop here: no supervisor can start, so the
    blade must be switched straight back off and the ON refused."""
    svc, fake = armed(monkeypatch)
    assert asyncio.run(svc.initialize()) is True

    assert svc._switch_on() is False

    assert fake.pad == OFF
    assert svc.last_refusal == "blade supervision could not start: ON refused"
    assert svc._commanded_on is False


# ------------------------------------------------------------- B31: the latch before any await


async def test_the_estop_latch_is_set_before_the_stop_awaits_anything(monkeypatch):
    """B31: a stop that is slow (the chip claim) or cancelled has already latched the blade."""
    svc, _ = armed(monkeypatch)  # armed and never initialised: the stop claims the lines first
    gate = asyncio.Event()
    claim = svc._claim_lines_for_stop

    async def slow_claim():
        await gate.wait()
        return await claim()

    monkeypatch.setattr(svc, "_claim_lines_for_stop", slow_claim)
    stop = asyncio.create_task(svc.emergency_stop())
    await asyncio.sleep(0)  # the stop has run up to its first suspension

    assert svc.emergency_latched is True, "latched while the claim is still pending"
    stop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop
    assert svc.emergency_latched is True, "and a cancelled stop leaves the latch set"


# ---------------------------------------------------------- B30, B34: a stale clear vs an E-stop


async def test_a_clear_that_began_before_an_estop_never_releases_after_it(monkeypatch):
    """B30/B34: a clear suspended in the line claim must not release the latch of an E-stop that
    started (and finished) while it waited."""
    svc, fake = armed(monkeypatch)  # never initialised: the clear first claims the lines
    gate = asyncio.Event()
    claim = svc._claim_lines_for_stop

    async def slow_claim():
        await gate.wait()
        return await claim()

    monkeypatch.setattr(svc, "_claim_lines_for_stop", slow_claim)
    clear = asyncio.create_task(svc.clear_emergency())
    await asyncio.sleep(0)  # the clear is suspended inside its claim
    monkeypatch.setattr(svc, "_claim_lines_for_stop", claim)

    report = await svc.emergency_stop()  # a full E-stop while the clear waits
    assert report["latched"] is True and svc.emergency_latched is True

    gate.set()
    await clear

    assert svc.emergency_latched is True, "the stale clear did not release the new latch"
    assert svc.clear_refusal == "an emergency stop started while the clear was running"
    assert fake.pad == OFF


# -------------------------------------------------- B42: a supervisor bound to a dead event loop


def test_a_supervisor_task_of_a_dead_event_loop_is_replaced(monkeypatch):
    """B42: a supervisor task left pending on a loop that no longer runs looked "running" and
    no new one was started on the live loop: the deadman then never fired."""
    clock = Clock()
    svc, fake = armed(monkeypatch, clock=clock, deadman_ms=1000)
    dead = asyncio.new_event_loop()
    dead.set_exception_handler(lambda *_: None)  # the task is destroyed pending: stay quiet
    dead.run_until_complete(on(svc))
    stale = svc._supervisor
    assert stale is not None and not stale.done() and stale.get_loop() is dead
    dead.close()

    async def live() -> None:
        assert await svc.set_active(True) is True  # a repeat of the ON on the new loop
        assert svc._supervisor is not stale and svc._supervisor.get_loop() is not dead
        clock.t += 5.0  # the client goes silent
        await eventually(lambda: fake.pad == OFF, timeout=2.0)
        assert "deadman" in svc.last_drop

    asyncio.run(live())


# --------------------------------------------------------------- B38: the park survives a rebuild

BROKEN = "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 99}\n"
REPAIRED = "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 25}\n"


async def test_the_shutdown_park_survives_a_config_repair_rebuild(monkeypatch, tmp_path):
    """B38: only the E-stop latch was tested to be carried over; the park latch must be too."""
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(BROKEN, encoding="utf-8")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(blade_service, "_blade_service", None)
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)
    monkeypatch.setitem(sys.modules, "lgpio", FakeLgpio())
    stub = blade_service.get_blade_service()
    assert stub.config_error
    await stub.park()
    assert (await stub.status())["parked"] == "shutdown"

    hardware.write_text(REPAIRED, encoding="utf-8")
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)
    fresh = blade_service.get_blade_service()

    assert fresh is not stub and fresh.config_error is None
    assert (await fresh.status())["parked"] == "shutdown", "the park latch is carried over"
    assert await fresh.set_active(True) is False and "parked" in fresh.last_refusal
