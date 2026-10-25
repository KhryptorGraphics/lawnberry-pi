"""``POST /control/emergency`` starts the stop before it reads the request body (review 3, NAV-04).

The stop does not use the body (it only picks the reply shape and the audit principal) but the
route read it first, waiting up to ``ESTOP_BODY_READ_TIMEOUT_S`` for a client whose body trails its
headers (a lossy HaLow/Wi-Fi link that splits the segments, a streaming client, ``Expect:
100-continue``). Only the software latch was armed meanwhile: the PTO relay, the levers and the
RoboHAT waited for the body, and a client that gave up during that wait never got a stop at all.
The acceptance limit for the stop is 100 ms.

The tractor is the real service over the stallable PCA9685; the body is an async generator that
stalls for as long as the test says. The orderings are asserted against that stall, not against
absolute times, so a loaded host cannot turn them into timing tests.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from _nav_rig import wait_until
from _tractor_fakes import NEUTRAL, mowing_tractor

from backend.src.api import rest as rest_module

STALL_S = 2.0


@pytest.fixture(autouse=True)
def _quiet_audit(monkeypatch):
    monkeypatch.setattr(rest_module.persistence, "add_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(
        rest_module.persistence, "add_audit_log_coalesced", lambda *a, **k: True, raising=False
    )
    # The read must be able to wait out the whole stall: any early stop is the route's doing.
    monkeypatch.setattr(rest_module, "ESTOP_BODY_READ_TIMEOUT_S", 30.0)


@pytest.fixture
async def mowing(monkeypatch):
    tractor, chip = mowing_tractor()
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: tractor)
    assert (await tractor.engage_blade(True, source="operator"))["status"] == "ok"
    moved = await tractor.set_levers(0.6, 0.6, source="operator", duration_ms=8000)
    assert moved["status"] == "ok" and tractor.state.blade_engaged
    yield tractor, chip
    await tractor.shutdown()


def _client() -> httpx.AsyncClient:
    from backend.src.main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=30.0
    )


def _is_stopped(tractor) -> bool:
    return (
        tractor.state.emergency_stop_active
        and tractor.blade_pto.state is False
        and tractor.state.left_lever == 0.0
    )


async def test_a_body_that_trails_its_headers_does_not_delay_the_stop(mowing):
    tractor, chip = mowing
    marks: dict[str, float] = {}

    async def slow_body():
        yield b'{"reason": '
        await asyncio.sleep(STALL_S)
        marks["body_resumed"] = time.monotonic()
        yield b'"operator"}'

    async def watch():
        while not _is_stopped(tractor):
            await asyncio.sleep(0.002)
        marks["stopped"] = time.monotonic()

    watcher = asyncio.create_task(watch())
    try:
        async with _client() as client:
            response = await client.post(
                "/api/v2/control/emergency",
                content=slow_body(),
                headers={"content-type": "application/json"},
            )
        await asyncio.wait_for(watcher, 5)
    finally:
        watcher.cancel()

    assert response.status_code == 202, response.text
    assert "stopped" in marks and "body_resumed" in marks
    assert marks["stopped"] < marks["body_resumed"], (
        "the PTO relay and the levers waited for the request body"
    )
    assert chip.regs == NEUTRAL


async def test_a_client_that_gives_up_while_its_body_is_stalled_still_gets_its_stop(mowing):
    tractor, chip = mowing

    async def never_finishing_body():
        yield b'{"reason": '
        await asyncio.sleep(60)
        yield b'"x"}'

    async with _client() as client:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                client.post(
                    "/api/v2/control/emergency",
                    content=never_finishing_body(),
                    headers={"content-type": "application/json"},
                ),
                timeout=0.5,
            )

    # the request is gone (cancelled); a stop that had not started would never start now
    await wait_until(lambda: _is_stopped(tractor), timeout=5.0)
    assert chip.regs == NEUTRAL
    assert tractor.state.authorized is False


async def test_the_body_still_picks_the_legacy_reply_shape(mowing):
    tractor, _chip = mowing

    async def two_chunks():
        yield b'{"command": "emergency_'
        await asyncio.sleep(0.2)
        yield b'stop", "reason": "operator_request"}'

    async with _client() as client:
        response = await client.post(
            "/api/v2/control/emergency",
            content=two_chunks(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "EMERGENCY_STOP_ACTIVE" and body["emergency_stop_active"] is True
    assert _is_stopped(tractor)
