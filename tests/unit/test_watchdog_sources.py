"""Multi-source, continuous watchdog (T031; audit N-04, WP-P1-08, D.4).

The baseline watchdog was fed by one loop's liveness and exited its loop after the first
timeout (``break``): a second hang was unsupervised, and a stop whose neutral write failed was
never retried. These tests drive the new ``Watchdog`` on a fake clock (deterministic) and, for
the loop itself, in real time with small windows.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from backend.src.safety.estop_handler import EstopHandler
from backend.src.safety.motor_authorization import MotorAuthorization
from backend.src.safety.watchdog import (
    ACTUATION_SOURCE,
    DEFAULT_SOURCE,
    SAFETY_MONITOR_SOURCE,
    Watchdog,
)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Rig:
    """A watchdog on a fake clock whose stop callback counts (and can misbehave)."""

    def __init__(self, *, sources=(DEFAULT_SOURCE,), result=None, **kw) -> None:
        self.clock = Clock()
        self.auth = MotorAuthorization()
        self.auth.authorize()
        self.stops = 0
        self.result = result

        async def on_timeout():
            self.stops += 1
            if isinstance(self.result, BaseException):
                raise self.result
            return self.result

        self.wd = Watchdog(
            EstopHandler(self.auth),
            timeout_ms=kw.pop("timeout_ms", 1000),
            on_timeout=kw.pop("on_timeout", on_timeout),
            required_sources=sources,
            clock=self.clock,
            **kw,
        )


# ------------------------------------------------------------------ the window


async def test_a_silent_source_trips_after_the_window_and_not_before():
    rig = Rig()
    rig.clock.advance(0.9)
    assert await rig.wd.evaluate() is False
    assert rig.auth.is_enabled() and rig.stops == 0

    rig.clock.advance(0.2)  # 1.1 s of silence against a 1.0 s window
    assert await rig.wd.evaluate() is True
    assert not rig.auth.is_enabled(), "a trip revokes authorization"
    assert rig.stops == 1 and rig.wd.tripped and rig.wd.trips == 1


async def test_a_beat_restarts_the_window():
    rig = Rig()
    for _ in range(5):
        rig.clock.advance(0.8)
        rig.wd.heartbeat()
        assert await rig.wd.evaluate() is False
    assert rig.stops == 0 and not rig.wd.tripped


async def test_every_required_source_must_beat_and_the_silent_one_is_named():
    rig = Rig(sources=(ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE))
    rig.clock.advance(0.7)
    rig.wd.heartbeat(ACTUATION_SOURCE)
    rig.wd.heartbeat(SAFETY_MONITOR_SOURCE)
    rig.clock.advance(0.7)
    rig.wd.heartbeat(SAFETY_MONITOR_SOURCE)  # the monitor keeps beating; actuation does not
    assert await rig.wd.evaluate() is False
    rig.clock.advance(0.5)
    rig.wd.heartbeat(SAFETY_MONITOR_SOURCE)
    assert await rig.wd.evaluate() is True

    reason = rig.wd.snapshot()["last_trip_reason"]
    assert ACTUATION_SOURCE in reason and SAFETY_MONITOR_SOURCE not in reason
    assert rig.wd.stale_sources().keys() == {ACTUATION_SOURCE}


def test_unknown_sources_and_empty_configuration_are_refused():
    rig = Rig(sources=(ACTUATION_SOURCE,))
    with pytest.raises(KeyError, match="typo"):
        rig.wd.heartbeat("typo")
    with pytest.raises(KeyError):
        rig.wd.heartbeat()  # no default source on this watchdog: be explicit
    with pytest.raises(ValueError):
        Watchdog(EstopHandler(MotorAuthorization()), timeout_ms=100, required_sources=())
    with pytest.raises(ValueError):
        Watchdog(EstopHandler(MotorAuthorization()), timeout_ms=100, retrip_interval_s=0)
    assert rig.wd.required_sources == (ACTUATION_SOURCE,)


async def test_start_gives_every_source_one_full_window():
    rig = Rig()
    rig.clock.advance(60.0)  # constructed long ago, never started
    await rig.wd.start()
    try:
        assert await rig.wd.evaluate() is False
    finally:
        await rig.wd.stop()


# ----------------------------------------------- continuous, not one-shot


async def test_it_repeats_the_stop_at_a_bounded_rate_while_a_source_stays_silent():
    rig = Rig(retrip_interval_s=1.0)
    rig.clock.advance(1.1)
    assert await rig.wd.evaluate() is True  # t = 1.1: first stop
    rig.clock.advance(0.5)
    assert await rig.wd.evaluate() is False, "bounded: no second stop within the interval"
    assert rig.stops == 1
    rig.clock.advance(0.6)
    assert await rig.wd.evaluate() is True  # t = 2.2
    rig.clock.advance(1.0)
    assert await rig.wd.evaluate() is True  # t = 3.2
    assert rig.stops == 3 and rig.wd.trips == 3 and rig.wd.tripped


async def test_it_rearms_when_every_source_is_fresh_and_trips_again_on_a_later_hang():
    """The baseline broke out of its loop after one trip, so a later hang was unsupervised."""
    rig = Rig()
    rig.clock.advance(1.1)
    await rig.wd.evaluate()
    assert rig.wd.tripped

    rig.wd.heartbeat()  # the feeder is back
    assert await rig.wd.evaluate() is False
    assert not rig.wd.tripped, "re-armed by fresh beats"

    rig.clock.advance(1.1)  # a second, later hang
    assert await rig.wd.evaluate() is True
    assert rig.stops == 2


async def test_recovery_needs_every_source_not_just_one():
    rig = Rig(sources=(ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE))
    rig.clock.advance(1.1)
    await rig.wd.evaluate()
    rig.wd.heartbeat(SAFETY_MONITOR_SOURCE)
    rig.clock.advance(1.1)  # the stop repeats; actuation never came back
    rig.wd.heartbeat(SAFETY_MONITOR_SOURCE)
    assert await rig.wd.evaluate() is True
    assert rig.wd.tripped

    rig.wd.heartbeat(ACTUATION_SOURCE)
    assert await rig.wd.evaluate() is False and not rig.wd.tripped


# ------------------------------------------------------------- the stop itself


async def test_a_raising_stop_callback_is_logged_and_retried_never_fatal(caplog):
    rig = Rig(result=RuntimeError("boom"))
    rig.clock.advance(1.1)
    with caplog.at_level(logging.ERROR):
        assert await rig.wd.evaluate() is True  # must not raise
    assert not rig.auth.is_enabled(), "authorization is revoked before the callback runs"
    assert any("on_timeout callback failed" in r.getMessage() for r in caplog.records)

    rig.clock.advance(1.1)
    assert await rig.wd.evaluate() is True  # still supervising: the stop is retried
    assert rig.stops == 2


async def test_an_incomplete_stop_is_reported_critical_and_repeated(caplog):
    rig = Rig(result={"complete": False, "faults": ["left_lever: OSError: bus"]})
    rig.clock.advance(1.1)
    with caplog.at_level(logging.CRITICAL):
        await rig.wd.evaluate()
    assert any(
        "INCOMPLETE" in r.getMessage() and "left_lever" in r.getMessage() for r in caplog.records
    )
    rig.clock.advance(1.1)
    assert await rig.wd.evaluate() is True


async def test_a_hung_stop_callback_is_abandoned_at_the_deadline_and_retried(caplog):
    calls = 0

    async def hung():
        nonlocal calls
        calls += 1
        await asyncio.sleep(3600)

    rig = Rig(on_timeout=hung, on_timeout_deadline_s=0.05)
    rig.clock.advance(1.1)
    started = time.monotonic()
    with caplog.at_level(logging.CRITICAL):
        assert await rig.wd.evaluate() is True
    assert time.monotonic() - started < 1.0, "the loop must not wait for a hung stop"
    assert any("UNCONFIRMED" in r.getMessage() for r in caplog.records)

    rig.clock.advance(1.1)
    assert await rig.wd.evaluate() is True
    assert calls == 2


def test_snapshot_reports_per_source_age_and_staleness():
    rig = Rig(sources=(ACTUATION_SOURCE, SAFETY_MONITOR_SOURCE))
    rig.clock.advance(0.4)
    rig.wd.heartbeat(ACTUATION_SOURCE)
    rig.clock.advance(0.7)
    snap = rig.wd.snapshot()
    assert snap["timeout_ms"] == 1000 and snap["tripped"] is False and snap["trips"] == 0
    assert snap["sources"][ACTUATION_SOURCE] == {"age_ms": 700, "stale": False}
    assert snap["sources"][SAFETY_MONITOR_SOURCE] == {"age_ms": 1100, "stale": True}


# ------------------------------------------------------------ the real loop


async def test_the_running_loop_keeps_stopping_the_machine_while_a_source_is_silent():
    """Real time. The old loop ``break``-ed after the first trip, so this would stay at 1."""
    auth = MotorAuthorization()
    auth.authorize()
    stops = 0

    async def on_timeout():
        nonlocal stops
        stops += 1

    wd = Watchdog(
        EstopHandler(auth),
        timeout_ms=30,
        on_timeout=on_timeout,
        retrip_interval_s=0.03,
        poll_interval_s=0.005,
    )
    await wd.start()
    await asyncio.sleep(0.4)
    started = time.monotonic()
    await wd.stop()
    assert time.monotonic() - started < 0.5, "stop() is prompt"
    assert stops >= 3 and wd.trips == stops
    assert not auth.is_enabled()


async def test_the_running_loop_rearms_by_itself_when_the_feeder_returns():
    stops = 0

    async def on_timeout():
        nonlocal stops
        stops += 1

    wd = Watchdog(
        EstopHandler(MotorAuthorization()),
        timeout_ms=40,
        on_timeout=on_timeout,
        retrip_interval_s=0.02,
        poll_interval_s=0.005,
    )
    await wd.start()
    await asyncio.sleep(0.15)
    assert wd.tripped
    for _ in range(10):  # the feeder comes back
        wd.heartbeat()
        await asyncio.sleep(0.01)
    assert not wd.tripped
    settled = stops
    for _ in range(10):
        wd.heartbeat()
        await asyncio.sleep(0.01)
    assert stops == settled, "no further stops while fed"
    await wd.stop()


async def test_the_loop_survives_an_internal_error():
    """An exception inside one cycle must not leave the machine unsupervised."""
    stops = 0

    async def on_timeout():
        nonlocal stops
        stops += 1

    wd = Watchdog(
        EstopHandler(MotorAuthorization()),
        timeout_ms=30,
        on_timeout=on_timeout,
        retrip_interval_s=0.03,
        poll_interval_s=0.005,
    )
    real = wd.evaluate
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise RuntimeError("clock glitch")
        return await real()

    wd.evaluate = flaky  # type: ignore[method-assign]
    await wd.start()
    await asyncio.sleep(0.25)
    await wd.stop()
    assert calls > 3 and stops >= 1


async def test_stop_is_idempotent_and_restart_works():
    wd = Watchdog(EstopHandler(MotorAuthorization()), timeout_ms=1000)
    await wd.stop()  # never started
    await wd.start()
    assert wd.running
    await wd.stop()
    await wd.stop()
    assert not wd.running
    await wd.start()
    assert wd.running
    await wd.stop()
