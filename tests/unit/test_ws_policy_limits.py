"""The WebSocket registry binds sockets to their login, and the handshake budget (SEC3-07, SEC3-09).

``core/ws_policy.py`` is standard-library only, so everything here runs against fakes with an
injected clock: a socket that records how it was closed, one whose close never completes, one whose
close raises, a session that is asked again and again. The end-to-end behaviour (a real app, real
tickets, a real uvicorn) is in ``tests/integration/test_ws_session_binding.py``,
``test_ws_handshake_budget.py`` and ``test_ws_live_uvicorn.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time

from backend.src.core import ws_policy
from backend.src.core.ws_policy import (
    HANDSHAKE_BURST,
    HandshakeLimiter,
    RefusalLog,
    WsRegistry,
)


class FakeSocket:
    """What the registry needs of a socket: an awaitable ``close`` that records its arguments."""

    def __init__(self) -> None:
        self.closed: list[tuple[int, str | None]] = []

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))


class HangingSocket(FakeSocket):
    """A peer that never answers the closing handshake."""

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))
        await asyncio.Event().wait()


class GatedSocket(FakeSocket):
    """A socket whose close parks until the test lets it go (Starlette's TestClient queues the
    frame after an anyio checkpoint, a real transport after a write: both are a point where the
    task that is closing the socket can be cancelled)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.entered.set()
        await self.gate.wait()
        self.closed.append((code, reason))


class BrokenSocket(FakeSocket):
    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---- binding and closing from outside


def _bind(registry: WsRegistry, session_id: str, socket=None):
    socket = socket if socket is not None else FakeSocket()
    return socket, registry.acquire(
        session_id=session_id, peer="ip:10.0.0.1", anonymous=False, websocket=socket
    )


async def test_closing_a_session_closes_exactly_its_sockets_with_1008():
    registry = WsRegistry()
    mine = [_bind(registry, "mine") for _ in range(3)]
    theirs, _ = _bind(registry, "theirs")
    unbound = FakeSocket()
    registry.acquire(session_id="mine", peer="ip:10.0.0.1", anonymous=False)  # no websocket given
    assert registry.bound_count("mine") == 3

    assert await registry.close_session("mine") == 3

    for socket, _slot in mine:
        assert socket.closed == [(1008, "session ended")]
    assert theirs.closed == [] and unbound.closed == []
    assert registry.bound_count("theirs") == 1


async def test_a_second_close_of_the_same_session_does_not_close_a_socket_twice():
    registry = WsRegistry()
    socket, _ = _bind(registry, "s")
    assert await registry.close_session("s") == 1
    assert await registry.close_session("s") == 0
    assert len(socket.closed) == 1


async def test_an_anonymous_socket_is_never_bound():
    registry = WsRegistry()
    socket = FakeSocket()
    registry.acquire(session_id=None, peer="ip:10.0.0.9", anonymous=True, websocket=socket)
    registry.acquire(session_id="sim", peer="ip:10.0.0.9", anonymous=True, websocket=socket)
    assert registry.bound_count("sim") == 0
    assert await registry.close_session("sim") == 0
    assert socket.closed == []


async def test_a_released_socket_is_no_longer_bound_or_closed():
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")
    registry.release(slot)
    registry.release(slot)  # idempotent
    assert registry.bound_count("s") == 0
    assert await registry.close_session("s") == 0
    assert socket.closed == []
    assert registry._bound == {}, "no empty set is left behind for a finished session"


async def test_a_peer_that_never_answers_the_close_cannot_hold_up_whoever_ends_the_session():
    registry = WsRegistry()
    hung, _ = _bind(registry, "s", HangingSocket())
    fine, _ = _bind(registry, "s")
    started = time.monotonic()
    assert await registry.close_session("s", timeout=0.15) == 2
    assert time.monotonic() - started < 1.0, "the bounded wait did not bound the wait"
    assert fine.closed == [(1008, "session ended")], "one hung peer must not starve the others"
    assert hung.closed == [(1008, "session ended")]


async def test_a_close_that_raises_is_swallowed_and_logged_not_propagated(caplog):
    registry = WsRegistry()
    _bind(registry, "s", BrokenSocket())
    fine, _ = _bind(registry, "s")
    with caplog.at_level(logging.DEBUG, logger=ws_policy.logger.name):
        assert await registry.close_session("s") == 2
    assert fine.closed == [(1008, "session ended")]
    assert any("raised RuntimeError" in r.getMessage() for r in caplog.records)


async def test_the_shipped_close_bound_is_about_a_second():
    """The finding says "within ~1 s": a bound that is not small would hold up every logout."""
    registry = WsRegistry()
    _bind(registry, "s", HangingSocket())
    started = time.monotonic()
    await registry.close_session("s")
    assert 0.9 <= time.monotonic() - started < 2.5


# ---- the expiry timer --------------------------------------------------------------------------


class Session:
    """A session as the timer sees it: ``seconds_left()`` is what the timer asks."""

    def __init__(self, *answers: float | None) -> None:
        self.answers = list(answers)
        self.asked = 0

    async def seconds_left(self) -> float | None:
        self.asked += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


async def _eventually(predicate, *, within=2.0, every=0.01) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        if predicate():
            return True
        await asyncio.sleep(every)
    return predicate()


async def test_the_timer_closes_the_socket_when_the_session_is_over(monkeypatch):
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")
    session = Session(0.05, 0.03, None)  # two re-arms, then "gone"
    registry.watch_expiry(slot, session.seconds_left)
    assert await _eventually(lambda: socket.closed)
    assert socket.closed == [(1008, "session ended")]
    assert session.asked == 3


async def test_the_timer_keeps_a_socket_whose_session_was_extended():
    """A refresh moves the session's expiry: the socket asks again at the old deadline, is told
    there is time left, and stays."""
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")
    session = Session(0.05, 3600.0)  # the second answer is the refreshed expiry
    registry.watch_expiry(slot, session.seconds_left)
    assert await _eventually(lambda: session.asked >= 2)
    await asyncio.sleep(0.2)
    assert socket.closed == [], "a session that was refreshed must keep its sockets"
    registry.release(slot)


async def test_the_timer_asks_again_at_least_every_recheck_interval(monkeypatch):
    """A far-away expiry is not slept through: a wall-clock step or a session removed behind the
    service's back is noticed within ``EXPIRY_RECHECK_MAX_S``."""
    monkeypatch.setattr(ws_policy, "EXPIRY_RECHECK_MAX_S", 0.05)
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")
    session = Session(8 * 3600.0, 8 * 3600.0, None)
    registry.watch_expiry(slot, session.seconds_left)
    assert await _eventually(lambda: socket.closed)
    assert session.asked == 3


async def test_a_check_that_raises_closes_the_socket(caplog):
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")

    async def broken() -> float | None:
        raise ValueError("the session store is broken")

    with caplog.at_level(logging.ERROR, logger=ws_policy.logger.name):
        registry.watch_expiry(slot, broken)
        assert await _eventually(lambda: socket.closed)
    assert socket.closed == [(1008, "session ended")]


async def test_release_stops_the_timer():
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")
    session = Session(0.05, None)
    registry.watch_expiry(slot, session.seconds_left)
    watcher = slot.watcher
    assert watcher is not None
    registry.release(slot)
    await asyncio.sleep(0.2)
    assert watcher.cancelled() or watcher.done()
    assert socket.closed == [], "a socket that was already given back must not be closed"
    assert slot.watcher is None


async def test_a_close_the_timer_has_started_survives_the_handlers_release():
    """The review-3 race, found by the full suite hanging: Starlette marks a socket closed BEFORE
    it queues the close frame, so the handler's next send fails at once, its ``finally`` hands the
    slot back and ``release`` cancels the timer that is in the middle of the close. The close must
    still complete, or the peer is never told 1008 (the TestClient then waits for ever; a real
    client sees an abnormal closure instead of the policy code)."""
    registry = WsRegistry()
    socket, slot = _bind(registry, "s", GatedSocket())
    registry.watch_expiry(
        slot, Session(None).seconds_left
    )  # the session is gone at the first check
    await asyncio.wait_for(socket.entered.wait(), timeout=2)  # the timer is inside the close
    assert socket.closed == [] and slot.closing

    registry.release(slot)  # what the handler's ``finally`` does
    socket.gate.set()  # the transport now takes the frame
    assert await _eventually(lambda: socket.closed)
    assert socket.closed == [(1008, "session ended")]
    assert slot.watcher is None


async def test_an_unbound_slot_gets_no_timer():
    registry = WsRegistry()
    slot = registry.acquire(session_id="sim", peer="ip:10.0.0.1", anonymous=False)
    registry.watch_expiry(slot, Session(None).seconds_left)
    assert slot.watcher is None


async def test_the_timer_and_the_session_end_closing_the_same_socket_close_it_once():
    registry = WsRegistry()
    socket, slot = _bind(registry, "s")
    registry.watch_expiry(slot, Session(0.05, None).seconds_left)
    await registry.close_session("s")
    await asyncio.sleep(0.2)
    assert socket.closed == [(1008, "session ended")]


# ---- the handshake bucket


def _limiter(clock: Clock, **kwargs) -> HandshakeLimiter:
    return HandshakeLimiter(clock=clock, **kwargs)


def test_a_new_key_may_burst_then_is_refused_with_a_retry_after():
    clock = Clock()
    limiter = _limiter(clock)
    allowed = [limiter.allow("ip:1.2.3.4") for _ in range(HANDSHAKE_BURST)]
    assert allowed == [None] * 20, "the shipped burst is 20"
    retry = limiter.allow("ip:1.2.3.4")
    assert retry == 1, "an empty bucket refills a token in half a second: wait the whole second"


def test_the_bucket_refills_two_tokens_a_second_and_no_more_than_the_burst():
    clock = Clock()
    limiter = _limiter(clock)
    for _ in range(20):
        assert limiter.allow("k") is None
    assert limiter.allow("k") is not None
    clock.now += 1.0  # two tokens
    assert [limiter.allow("k") for _ in range(3)] == [None, None, 1]
    clock.now += 3600.0  # never more than the burst, however long it was idle
    assert [limiter.allow("k") for _ in range(22)].count(None) == 20


def test_retry_after_is_the_wait_for_one_token():
    clock = Clock()
    limiter = _limiter(clock, rate_hz=0.25, burst=2)
    assert limiter.allow("k") is None and limiter.allow("k") is None
    assert limiter.allow("k") == 4, "0.25 tokens a second: four seconds for one token"
    clock.now += 3.0
    assert limiter.allow("k") == 1


def test_keys_do_not_share_a_bucket():
    clock = Clock()
    limiter = _limiter(clock)
    for _ in range(25):
        limiter.allow("session:junk-sender")
    assert limiter.allow("session:operator") is None
    assert limiter.allow("ip:other-peer") is None


def test_the_table_never_holds_more_than_the_cap_whatever_the_keys():
    clock = Clock()
    limiter = _limiter(clock, max_keys=50)
    for n in range(2000):
        limiter.allow(f"ip:10.9.{n // 250}.{n % 250}")
        clock.now += 0.001
        assert len(limiter) <= 50
    assert len(limiter) > 0


def test_a_flood_of_keys_does_not_unlock_a_key_that_is_being_limited():
    """Eviction is least-recently-used: the key that keeps asking is the freshest one, so a flood
    of other keys (at one instant: no token refills meanwhile) cannot flush it to a full bucket."""
    clock = Clock()
    limiter = _limiter(clock, max_keys=50)
    for _ in range(20):
        limiter.allow("ip:attacker")
    for n in range(500):
        limiter.allow(f"ip:10.9.{n // 250}.{n % 250}")
        assert limiter.allow("ip:attacker") is not None, n


def test_a_bucket_that_is_full_again_is_forgotten():
    clock = Clock()
    limiter = _limiter(clock)
    for n in range(100):
        limiter.allow(f"ip:10.0.0.{n}")
    assert len(limiter) == 100
    clock.now += 600.0  # every bucket has refilled to its burst: indistinguishable from new
    limiter.allow("ip:late")
    assert len(limiter) == 1


def test_the_shipped_handshake_budget_is_two_a_second_burst_twenty():
    assert (ws_policy.HANDSHAKE_RATE_HZ, ws_policy.HANDSHAKE_BURST) == (2.0, 20)
    assert ws_policy.MAX_HANDSHAKE_KEYS <= 8192


def test_the_registry_counts_refused_handshakes_and_owns_its_own_buckets():
    clock = Clock()
    registry = WsRegistry(handshakes=HandshakeLimiter(clock=clock, burst=2))
    assert registry.check_handshake("k") is None and registry.check_handshake("k") is None
    assert registry.check_handshake("k") is not None
    assert registry.snapshot()["refused"] == {"handshake": 1}
    assert WsRegistry().check_handshake("k") is None, "another registry starts with full buckets"


# ---- the refusal log


def test_one_line_per_peer_per_interval_saying_how_many_refusals_it_stands_for():
    clock = Clock()
    log = RefusalLog(clock=clock, interval_s=60.0)
    assert log.note("ip:1.2.3.4") == 1, "the first refusal of a peer is logged at once"
    assert [log.note("ip:1.2.3.4") for _ in range(299)] == [None] * 299
    clock.now += 59.0
    assert log.note("ip:1.2.3.4") is None
    clock.now += 2.0
    assert log.note("ip:1.2.3.4") == 301, "300 hidden + this one: nothing is lost, only summed"
    assert log.note("ip:1.2.3.4") is None


def test_a_second_peer_has_its_own_line():
    clock = Clock()
    log = RefusalLog(clock=clock)
    assert log.note("ip:1.2.3.4") == 1
    assert log.note("ip:5.6.7.8") == 1


def test_a_flood_of_distinct_peers_is_capped_in_lines_per_interval():
    """A client that rotates its addresses must not be able to turn per-peer limiting into one
    line per attempt."""
    clock = Clock()
    log = RefusalLog(clock=clock, interval_s=60.0, max_lines=30)
    lines = [log.note(f"ip:10.0.{n // 250}.{n % 250}") for n in range(1000)]
    assert sum(1 for stands_for in lines if stands_for is not None) == 30
    clock.now += 61.0
    assert log.note("ip:10.0.0.0") is not None, "a new interval has lines again"


def test_the_refusal_log_table_is_bounded():
    clock = Clock()
    log = RefusalLog(clock=clock, max_keys=40, max_lines=10_000)
    for n in range(500):
        log.note(f"ip:10.0.{n // 250}.{n % 250}")
        assert len(log) <= 40


def test_a_suppressed_count_survives_the_global_cap():
    """A line held back by the per-interval ceiling is not forgotten: the count rides on the next
    line that is allowed."""
    clock = Clock()
    log = RefusalLog(clock=clock, interval_s=60.0, max_lines=1)
    assert log.note("a") == 1
    assert log.note("b") is None  # over the ceiling, but counted
    assert log.note("b") is None
    clock.now += 61.0
    assert log.note("b") == 3
