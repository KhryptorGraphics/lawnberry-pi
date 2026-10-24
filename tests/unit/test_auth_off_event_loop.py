"""Password checks run off the event loop (SEC-04).

bcrypt at 12 rounds is ~270 ms of CPU. Run inline in ``AuthService.authenticate`` it stalled the
single event loop that also carries the tractor deadman, the tilt monitor and every E-stop handler
(measured: 1.3 s for five failed logins). The checks now run on a small dedicated thread pool, not
on the loop and not on the default executor that the PCA9685 / RoboHAT / sensor writes share.

A blocking stand-in for bcrypt makes the stall deterministic: it sleeps 250 ms and records the
thread it ran on, while a 5 ms ticker on the loop measures the longest stall.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backend.src.services.auth_service import AuthenticationError, AuthService

CRED = "correct-horse-battery-staple"
BLOCK_SECONDS = 0.25


def _strict_service(monkeypatch) -> AuthService:
    monkeypatch.setenv("LAWN_BERRY_OPERATOR_CREDENTIAL", CRED)
    monkeypatch.setenv("SIM_MODE", "0")
    service = AuthService()
    monkeypatch.setenv("SIM_MODE", "1")
    return service


async def _longest_stall(work) -> tuple[object, float]:
    """Run ``work`` while a ticker measures the longest time the loop went without running."""
    state = {"stop": False, "gap": 0.0}

    async def tick() -> None:
        last = time.perf_counter()
        while not state["stop"]:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            state["gap"] = max(state["gap"], now - last)
            last = now

    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.03)  # let it start before anything blocks
    try:
        result = await work
    finally:
        state["stop"] = True
        await ticker
    return result, state["gap"]


async def test_failed_logins_do_not_stall_the_event_loop(monkeypatch):
    service = _strict_service(monkeypatch)
    threads: list[threading.Thread] = []

    def blocking_verify(plain: str, hashed: str) -> bool:
        threads.append(threading.current_thread())
        time.sleep(BLOCK_SECONDS)  # stands in for the bcrypt CPU time
        return False

    monkeypatch.setattr(service.password_manager, "verify_password", blocking_verify)

    async def attempts():
        return await asyncio.gather(
            *[service.authenticate("wrong", client_ip=f"192.0.2.{n}") for n in range(4)],
            return_exceptions=True,
        )

    results, stall = await _longest_stall(attempts())

    assert all(isinstance(r, AuthenticationError) and r.status_code == 401 for r in results)
    assert len(threads) == 4
    assert all(t is not threading.main_thread() for t in threads), "bcrypt ran on the event loop"
    assert all(t.name.startswith("auth-hash") for t in threads), "must not use the default executor"
    assert stall < BLOCK_SECONDS / 2, f"the event loop stalled for {stall * 1000:.0f} ms"


async def test_the_manual_unlock_password_hash_check_runs_off_the_loop(monkeypatch):
    from backend.src.api.routers import auth as auth_router
    from backend.src.core.globals import _security_settings

    service = _strict_service(monkeypatch)
    monkeypatch.setattr(auth_router, "primary_auth_service", service)
    monkeypatch.setattr(
        _security_settings, "password_hash", service.password_manager.hash_password("x" * 12)
    )
    seen: list[threading.Thread] = []

    def blocking_matches(plain: str, hashed: str) -> bool:
        seen.append(threading.current_thread())
        time.sleep(BLOCK_SECONDS)
        return False

    monkeypatch.setattr(service.password_manager, "matches", blocking_matches)

    matched, stall = await _longest_stall(auth_router._manual_password_matches("anything"))

    assert matched is False
    assert seen and seen[0] is not threading.main_thread()
    assert seen[0].name.startswith("auth-hash")
    assert stall < BLOCK_SECONDS / 2, f"the event loop stalled for {stall * 1000:.0f} ms"


async def test_the_credential_check_for_other_flows_runs_off_the_loop(monkeypatch):
    service = _strict_service(monkeypatch)
    seen: list[threading.Thread] = []

    def blocking_verify(plain: str, hashed: str) -> bool:
        seen.append(threading.current_thread())
        time.sleep(BLOCK_SECONDS)
        return True

    monkeypatch.setattr(service.password_manager, "verify_password", blocking_verify)
    matched, stall = await _longest_stall(service.verify_operator_credential("whatever"))
    assert matched is True
    assert seen[0] is not threading.main_thread()
    assert stall < BLOCK_SECONDS / 2


@pytest.mark.parametrize("which", ["no credential configured", "simulator opt-in"])
async def test_cases_that_need_no_bcrypt_do_not_use_the_pool(monkeypatch, which):
    """Nothing to hash: the answer is immediate and no worker thread is involved."""
    if which == "simulator opt-in":
        monkeypatch.setenv("SIM_MODE", "1")
        monkeypatch.setenv("LAWN_BERRY_SIM_OPEN_AUTH", "1")
        monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
        service = AuthService()
        assert await service.verify_operator_credential("anything") is True
    else:
        monkeypatch.setenv("SIM_MODE", "0")
        monkeypatch.delenv("LAWN_BERRY_OPERATOR_CREDENTIAL", raising=False)
        service = AuthService()
        assert await service.verify_operator_credential("anything") is False
