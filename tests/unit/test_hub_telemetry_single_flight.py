"""HW-06 (PR #20 review 3): the telemetry loop and the ``/ws/status`` readers never overlap.

``WebSocketHub.current_telemetry`` serialised its readers on ``_telemetry_lock``, but the telemetry
loop called ``_generate_telemetry()`` without it, so a reader arriving while the loop was inside a
sensor read started a second, overlapping ``read_all_sensors()``: the I2C and serial drivers saw
overlapping callers every cycle. The sensor read is replaced by a slow fake that counts how many
reads run at the same time.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.src.services.websocket_hub import WebSocketHub


class SlowSensors:
    """What ``_generate_telemetry`` does to the hardware: a read that takes ``read_s``."""

    def __init__(self, read_s: float = 0.15) -> None:
        self.read_s = read_s
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.fail_next = False

    async def __call__(self) -> dict:
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.read_s)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("sensor bus glitch")
            return {"read": self.calls, "t": time.monotonic()}
        finally:
            self.active -= 1


@pytest.fixture
def hub():
    hub = WebSocketHub()
    hub.telemetry_cadence_hz = 10.0
    return hub


async def test_a_reader_that_arrives_during_the_loops_read_waits_for_it_and_shares_it(hub):
    sensors = SlowSensors(0.15)
    hub._generate_telemetry = sensors
    await hub.start_telemetry_loop()
    try:
        await asyncio.sleep(0.05)  # the loop is inside its read
        assert sensors.active == 1

        samples = await asyncio.gather(*(hub.current_telemetry(max_age_s=0.5) for _ in range(3)))
    finally:
        await hub.stop_telemetry_loop()

    assert sensors.peak == 1, "two sensor reads overlapped"
    assert sensors.calls == 1, "the readers took the loop's sample instead of reading again"
    assert [sample["read"] for sample in samples] == [1, 1, 1]


async def test_readers_among_themselves_still_make_one_read(hub):
    sensors = SlowSensors(0.1)
    hub._generate_telemetry = sensors

    samples = await asyncio.gather(*(hub.current_telemetry(max_age_s=0.5) for _ in range(5)))

    assert sensors.peak == 1 and sensors.calls == 1
    assert len({sample["read"] for sample in samples}) == 1


async def test_a_reader_never_overlaps_a_loop_read_that_starts_during_its_own(hub):
    """The other order: the reader is inside its read when the loop wakes up."""
    sensors = SlowSensors(0.15)
    hub._generate_telemetry = sensors
    hub.telemetry_cadence_hz = 200.0  # the loop is ready the moment the reader holds the lock

    reader = asyncio.create_task(hub.current_telemetry(max_age_s=0.0))
    await asyncio.sleep(0.03)
    await hub.start_telemetry_loop()
    try:
        await asyncio.sleep(0.4)
        await reader
    finally:
        await hub.stop_telemetry_loop()

    assert sensors.peak == 1, "two sensor reads overlapped"
    assert sensors.calls >= 2, "the loop went on sampling after the reader"


async def test_stopping_the_loop_inside_a_read_frees_the_readers(hub):
    sensors = SlowSensors(1.0)
    hub._generate_telemetry = sensors
    await hub.start_telemetry_loop()
    await asyncio.sleep(0.05)  # inside the loop's read, holding the lock

    await hub.stop_telemetry_loop()
    sensors.read_s = 0.0

    sample = await asyncio.wait_for(hub.current_telemetry(max_age_s=0.0), timeout=2.0)
    assert sample["read"] == 2


async def test_a_failing_loop_read_does_not_leave_the_lock_held(hub):
    sensors = SlowSensors(0.02)
    sensors.fail_next = True
    hub._generate_telemetry = sensors
    await hub.start_telemetry_loop()
    try:
        await asyncio.sleep(0.1)  # the first read raised; the loop backs off for a second
        sample = await asyncio.wait_for(hub.current_telemetry(max_age_s=0.0), timeout=2.0)
    finally:
        await hub.stop_telemetry_loop()

    assert sample["read"] >= 2 and sensors.peak == 1
