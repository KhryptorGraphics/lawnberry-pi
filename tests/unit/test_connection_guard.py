"""SEC20-04: idle TCP connections must not starve the E-stop of descriptors.

``backend/src/core/connection_guard.py`` is the policy (pure logic: fakes and a fake clock, no
sockets); ``backend/src/core/uvicorn_protocol.py`` wires it to uvicorn's connection life cycle
(driven here with a fake transport on a real event loop, for both httptools and h11). The real
thing, a real uvicorn under a lowered descriptor limit and a real flood, is in
``tests/integration/test_connection_guard_e2e.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import resource

import pytest
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
from uvicorn.server import ServerState

from backend.src.core import connection_guard as cg
from backend.src.core import uvicorn_protocol as up
from backend.src.core.connection_guard import ConnectionGuard

ESTOP_PATH = "/api/v2/control/emergency"


class Conn:
    """A hashable stand-in for one HTTP connection that remembers how often it was closed."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def __repr__(self) -> str:
        return f"Conn({self.name})"


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def guard_of(max_connections: int = 4, reserve: int = 1, deadline: float = 10.0, **kw):
    return ConnectionGuard(max_connections, reserve, deadline, **kw)


def admit(guard: ConnectionGuard, name: str) -> Conn:
    conn = Conn(name)
    assert guard.register(conn, conn.close) is True
    return conn


@pytest.fixture(autouse=True)
def _clean_process_state():
    cg.reset_process_guard(None)
    up._warned.clear()
    up._budget_checked = False
    yield
    cg.reset_process_guard(None)


# --------------------------------------------------------------------------------------
# the table: admission, eviction, refusal
# --------------------------------------------------------------------------------------


def test_below_the_cap_everything_is_admitted_and_nothing_is_closed():
    guard = guard_of(4, 1)
    conns = [admit(guard, str(i)) for i in range(4)]
    assert len(guard) == 4
    assert [c.closed for c in conns] == [0, 0, 0, 0]
    assert guard.accepted == 4 and guard.evicted == 0 and guard.refused == 0


def test_at_the_cap_the_oldest_unproductive_connection_is_evicted_first():
    guard = guard_of(3, 1)
    a, b, c = (admit(guard, n) for n in "abc")
    d = admit(guard, "d")
    assert [x.closed for x in (a, b, c, d)] == [1, 0, 0, 0]
    e = admit(guard, "e")
    assert [x.closed for x in (a, b, c, d, e)] == [1, 1, 0, 0, 0]
    assert len(guard) == 3  # the cap holds
    assert guard.evicted == 2 and guard.accepted == 5 and guard.refused == 0


def test_a_productive_connection_is_skipped_and_the_next_oldest_junk_goes():
    guard = guard_of(3, 1)
    a, b, c = (admit(guard, n) for n in "abc")
    assert guard.mark_productive(a) is True  # the oldest, but it has delivered a request
    admit(guard, "d")
    assert (a.closed, b.closed, c.closed) == (0, 1, 0)


def test_when_every_connection_is_productive_the_newcomer_is_refused_and_nobody_is_closed():
    guard = guard_of(3, 1)  # soft cap 2: the third one is an E-stop request
    a, b, c = (admit(guard, n) for n in "abc")
    assert guard.mark_productive(a) and guard.mark_productive(b)
    assert guard.mark_productive(c, estop=True)
    newcomer = Conn("late")
    assert guard.register(newcomer, newcomer.close) is False
    assert guard.refused == 1
    assert (a.closed, b.closed, c.closed, newcomer.closed) == (0, 0, 0, 0)
    assert len(guard) == 3
    # the refused connection is not tracked: it cannot be promoted to productive either
    assert guard.mark_productive(newcomer, estop=True) is False


def test_a_connection_outlives_as_many_newcomers_as_the_junk_pool_is_deep_minus_one():
    """The documented limit of oldest-first eviction, and why the reserve is a quarter of the cap.

    A connection that has just arrived and has not spoken yet is evictable like any other. It is
    the LAST one the guard takes (oldest first): it survives as many newcomers as there were
    older evictable connections when it arrived, and the next newcomer takes it. That junk pool
    is never smaller than the reserve, because the quota keeps productive connections below
    ``soft_cap``; it is why the default reserve is a quarter of the cap, not an eighth.
    """
    guard = guard_of(6, 3)  # soft cap 3
    for name in "PQR":
        assert guard.mark_productive(admit(guard, name))
    for name in ("j1", "j2", "j3"):
        admit(guard, name)  # fills the table: 3 productive + 3 junk
    victim = admit(guard, "estop-client")  # evicts j1
    admit(guard, "n1")
    admit(guard, "n2")
    assert victim.closed == 0  # two newcomers after it: still here (it can speak now)
    admit(guard, "n3")
    assert victim.closed == 1  # pool_size (3) newcomers after it: gone


def test_registering_the_same_connection_twice_is_a_no_op():
    guard = guard_of(3, 1)
    a = admit(guard, "a")
    assert guard.register(a, a.close) is True
    assert len(guard) == 1 and guard.accepted == 1
    guard.mark_productive(a)
    assert guard.register(a, a.close) is True
    assert len(guard) == 1 and guard.accepted == 1


def test_a_failing_close_does_not_block_the_newcomer_and_is_counted_and_logged_once(caplog):
    caplog.set_level(logging.ERROR, logger="backend.src.core.connection_guard")
    guard = guard_of(2, 1)

    def boom() -> None:
        raise RuntimeError("transport exploded")

    assert guard.register("old-1", boom) and guard.register("old-2", boom)
    assert guard.register("new-1", lambda: None) is True  # evicts old-1; its close raises
    assert guard.register("new-2", lambda: None) is True  # evicts old-2; raises again
    assert guard.close_errors == 2 and guard.evicted == 2
    assert len(guard) == 2
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1  # a flood of failing closes cannot flood the journal


# --------------------------------------------------------------------------------------
# the soft cap and the E-stop reserve
# --------------------------------------------------------------------------------------


def test_the_quota_refuses_productive_status_to_requests_that_are_not_estops():
    guard = guard_of(6, 2)  # soft cap 4
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}")) is True
    assert guard.over_soft_cap() is True
    extra = admit(guard, "extra")
    assert guard.mark_productive(extra) is False
    assert guard.over_quota == 1
    assert guard.is_productive(extra) is False  # still evictable junk
    # ... but an E-stop request is never refused by the quota
    assert guard.mark_productive(extra, estop=True) is True
    assert guard.is_productive(extra) is True
    assert guard.over_quota == 1


def test_over_soft_cap_counts_productive_connections_only_so_junk_cannot_trip_it():
    guard = guard_of(10, 2)  # soft cap 8
    for i in range(10):
        admit(guard, f"junk{i}")
    assert guard.over_soft_cap() is False
    assert guard.snapshot()["productive"] == 0


def test_over_soft_cap_flips_exactly_at_the_quota_and_back_when_one_connection_leaves():
    guard = guard_of(5, 2)  # soft cap 3
    conns = [admit(guard, f"p{i}") for i in range(3)]
    for i, conn in enumerate(conns):
        assert guard.over_soft_cap() is False, i
        guard.mark_productive(conn)
    assert guard.over_soft_cap() is True
    guard.release(conns[1])
    assert guard.over_soft_cap() is False


def test_a_productive_connection_keeps_its_quota_slot_when_asked_again_at_full_quota():
    guard = guard_of(5, 2)
    conns = [admit(guard, f"p{i}") for i in range(3)]
    for conn in conns:
        assert guard.mark_productive(conn)
    assert guard.over_soft_cap() is True
    assert guard.mark_productive(conns[0]) is True
    assert guard.over_quota == 0


def test_mark_productive_of_a_connection_the_guard_does_not_track_is_false():
    guard = guard_of(3, 1)
    assert guard.mark_productive(Conn("stranger")) is False
    assert guard.mark_productive(Conn("stranger2"), estop=True) is False


# --------------------------------------------------------------------------------------
# release
# --------------------------------------------------------------------------------------


def test_release_frees_a_slot_so_the_next_newcomer_evicts_nobody():
    guard = guard_of(2, 1)
    a, b = admit(guard, "a"), admit(guard, "b")
    guard.release(a)
    c = admit(guard, "c")
    assert (a.closed, b.closed, c.closed) == (0, 0, 0)
    assert guard.evicted == 0 and len(guard) == 2


def test_releasing_a_productive_connection_gives_its_quota_slot_back():
    guard = guard_of(4, 2)  # soft cap 2
    a, b = admit(guard, "a"), admit(guard, "b")
    guard.mark_productive(a)
    guard.mark_productive(b)
    late = admit(guard, "late")
    assert guard.mark_productive(late) is False
    guard.release(a)
    assert guard.mark_productive(late) is True


def test_release_is_idempotent_and_ignores_connections_it_never_saw():
    guard = guard_of(3, 1)
    a = admit(guard, "a")
    guard.release(a)
    guard.release(a)
    guard.release(Conn("never-registered"))
    assert len(guard) == 0


def test_releasing_a_connection_that_was_already_evicted_changes_nothing():
    """uvicorn calls ``connection_lost`` for a connection the guard aborted: it must be harmless."""
    guard = guard_of(2, 1)
    a, b = admit(guard, "a"), admit(guard, "b")
    admit(guard, "c")  # evicts a
    guard.release(a)
    assert len(guard) == 2
    assert b.closed == 0


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------


def test_snapshot_reports_every_counter_and_gauge_with_a_fake_clock():
    clock = Clock(100.0)
    guard = guard_of(4, 1, 7.5, clock=clock)
    first = admit(guard, "first")
    clock.now = 103.0
    admit(guard, "second")
    clock.now = 104.0
    admit(guard, "third")
    guard.mark_productive(first)
    clock.now = 105.0
    guard.count_deadline_close()
    guard.count_shed()
    guard.count_shed()
    assert guard.snapshot() == {
        "max_connections": 4,
        "reserve": 1,
        "soft_cap": 3,
        "header_deadline_s": 7.5,
        "body_deadline_s": 15.0,
        "connections": 3,
        "productive": 1,
        "unproductive": 2,
        "over_soft_cap": False,
        # the OLDEST unproductive one ("second", registered at 103), not the newest (104)
        "oldest_unproductive_age_s": pytest.approx(2.0),
        "peak_connections": 3,
        "accepted": 3,
        "evicted": 0,
        "refused": 0,
        "over_quota": 0,
        "deadline_closed": 1,
        "shed": 2,
        "close_errors": 0,
    }


def test_snapshot_has_no_age_when_nothing_is_waiting_and_is_plain_json():
    guard = guard_of(4, 1)
    snapshot = guard.snapshot()
    assert snapshot["oldest_unproductive_age_s"] is None
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_peak_connections_is_the_high_water_mark_not_the_current_count():
    guard = guard_of(4, 1)
    conns = [admit(guard, str(i)) for i in range(3)]
    for conn in conns:
        guard.release(conn)
    admit(guard, "x")
    assert guard.snapshot()["connections"] == 1
    assert guard.snapshot()["peak_connections"] == 3


@pytest.mark.parametrize(
    "max_connections, reserve",
    [(1, 1), (0, 0), (-4, 1), (4, 0), (4, 4), (4, 9), (4, -1), (4.0, 1), (4, 1.0)],
)
def test_the_constructor_refuses_a_geometry_that_cannot_work(max_connections, reserve):
    with pytest.raises(ValueError):
        ConnectionGuard(max_connections, reserve)


@pytest.mark.parametrize("deadline", [0, -1.0, float("nan"), float("inf")])
def test_the_constructor_refuses_a_deadline_that_is_not_a_positive_finite_number(deadline):
    with pytest.raises(ValueError):
        ConnectionGuard(8, 2, deadline)


# --------------------------------------------------------------------------------------
# sizing and configuration
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "soft_nofile, expected",
    [
        (10, 32),
        (64, 32),
        (65, 32),
        (128, 64),
        (256, 128),
        (1024, 512),
        (4096, 1024),
        (1 << 20, 1024),
    ],
)
def test_the_default_cap_is_half_the_descriptor_limit_clamped_to_32_and_1024(soft_nofile, expected):
    assert cg.default_max_connections(soft_nofile) == expected


@pytest.mark.parametrize(
    "max_connections, expected",
    [(2, 1), (4, 3), (16, 8), (32, 8), (128, 32), (512, 128), (1024, 256), (4096, 256)],
)
def test_the_default_reserve_is_a_quarter_clamped_to_8_and_256_and_below_the_cap(
    max_connections, expected
):
    assert cg.default_reserve(max_connections) == expected
    assert cg.default_reserve(max_connections) < max_connections


def test_defaults_follow_the_descriptor_limit():
    shipped = cg.guard_from_environment({}, soft_nofile=4096)  # the unit's LimitNOFILE
    assert (shipped.max_connections, shipped.reserve, shipped.soft_cap) == (1024, 256, 768)
    assert shipped.header_deadline_s == 10.0
    small = cg.guard_from_environment({}, soft_nofile=256)  # the review's scaled-down repro
    assert (small.max_connections, small.reserve, small.soft_cap) == (128, 32, 96)


def test_valid_overrides_are_used_and_say_nothing(caplog):
    caplog.set_level(logging.DEBUG, logger="backend.src.core.connection_guard")
    env = {
        cg.MAX_CONNECTIONS_ENV: "24",
        cg.CONNECTION_RESERVE_ENV: "4",
        cg.HEADER_DEADLINE_ENV: "2.5",
    }
    guard = cg.guard_from_environment(env, soft_nofile=4096)
    assert (guard.max_connections, guard.reserve, guard.header_deadline_s) == (24, 4, 2.5)
    assert caplog.records == []


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_a_blank_value_means_unset_and_is_not_an_error(blank, caplog):
    caplog.set_level(logging.DEBUG, logger="backend.src.core.connection_guard")
    env = {name: blank for name in (cg.MAX_CONNECTIONS_ENV, cg.CONNECTION_RESERVE_ENV)}
    env[cg.HEADER_DEADLINE_ENV] = blank
    guard = cg.guard_from_environment(env, soft_nofile=4096)
    assert (guard.max_connections, guard.reserve, guard.header_deadline_s) == (1024, 256, 10.0)
    assert caplog.records == []


BAD_INTEGERS = [
    "abc", "15", "0", "-24", "+24", "24.0", "1e2", "0x20", "1_000", "٢٤", "24 24",
    "1000000000", "true", "4033",
]  # fmt: skip


@pytest.mark.parametrize("bad", BAD_INTEGERS)
def test_a_bad_maximum_logs_one_error_naming_the_variable_and_uses_the_default(bad, caplog):
    caplog.set_level(logging.DEBUG, logger="backend.src.core.connection_guard")
    guard = cg.guard_from_environment({cg.MAX_CONNECTIONS_ENV: bad}, soft_nofile=4096)
    assert guard.max_connections == 1024
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and len(caplog.records) == 1
    assert cg.MAX_CONNECTIONS_ENV in errors[0].getMessage()
    assert repr(bad) in errors[0].getMessage()


@pytest.mark.parametrize(
    "bad, max_connections",
    [("3", 24), ("abc", 24), ("-1", 24), ("4.0", 24), ("24", 24), ("100", 24), ("0x8", 24)],
)
def test_a_bad_reserve_logs_one_error_and_uses_the_default_for_the_effective_maximum(
    bad, max_connections, caplog
):
    caplog.set_level(logging.DEBUG, logger="backend.src.core.connection_guard")
    env = {cg.MAX_CONNECTIONS_ENV: str(max_connections), cg.CONNECTION_RESERVE_ENV: bad}
    guard = cg.guard_from_environment(env, soft_nofile=4096)
    assert guard.max_connections == max_connections
    assert guard.reserve == cg.default_reserve(max_connections) == 8
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and cg.CONNECTION_RESERVE_ENV in errors[0].getMessage()


def test_the_reserve_must_stay_below_the_maximum_that_is_actually_in_force(caplog):
    """A reserve that is fine for the default cap is refused when the cap is overridden lower."""
    env = {cg.MAX_CONNECTIONS_ENV: "16", cg.CONNECTION_RESERVE_ENV: "16"}
    guard = cg.guard_from_environment(env, soft_nofile=4096)
    assert (guard.max_connections, guard.reserve) == (16, 8)
    assert sum(r.levelno == logging.ERROR for r in caplog.records) == 1
    env = {cg.MAX_CONNECTIONS_ENV: "16", cg.CONNECTION_RESERVE_ENV: "15"}
    assert cg.guard_from_environment(env, soft_nofile=4096).reserve == 15


@pytest.mark.parametrize(
    "bad",
    ["0.9", "121", "120.5", "abc", "-5", "nan", "inf", "1e1", "10s", "0", ".5", "5.", "٣"],
)
def test_a_bad_deadline_logs_one_error_naming_the_variable_and_uses_ten_seconds(bad, caplog):
    caplog.set_level(logging.DEBUG, logger="backend.src.core.connection_guard")
    guard = cg.guard_from_environment({cg.HEADER_DEADLINE_ENV: bad}, soft_nofile=4096)
    assert guard.header_deadline_s == 10.0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and cg.HEADER_DEADLINE_ENV in errors[0].getMessage()


@pytest.mark.parametrize("good, expected", [("1", 1.0), ("1.0", 1.0), ("2.5", 2.5), ("120", 120.0)])
def test_the_deadline_accepts_its_whole_range_inclusive(good, expected):
    guard = cg.guard_from_environment({cg.HEADER_DEADLINE_ENV: good}, soft_nofile=4096)
    assert guard.header_deadline_s == expected


@pytest.mark.parametrize("good", ["16", "024", "4032"])
def test_the_maximum_accepts_its_whole_range_inclusive(good):
    guard = cg.guard_from_environment({cg.MAX_CONNECTIONS_ENV: good}, soft_nofile=4096)
    assert guard.max_connections == int(good)


def test_every_bad_variable_is_reported_once_each_and_the_defaults_are_used(caplog):
    env = {
        cg.MAX_CONNECTIONS_ENV: "many",
        cg.CONNECTION_RESERVE_ENV: "some",
        cg.HEADER_DEADLINE_ENV: "soon",
    }
    guard = cg.guard_from_environment(env, soft_nofile=4096)
    assert (guard.max_connections, guard.reserve, guard.header_deadline_s) == (1024, 256, 10.0)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(messages) == 3
    for name in (cg.MAX_CONNECTIONS_ENV, cg.CONNECTION_RESERVE_ENV, cg.HEADER_DEADLINE_ENV):
        assert sum(name in message for message in messages) == 1


def test_a_maximum_that_leaves_no_room_for_the_process_own_descriptors_is_refused(caplog):
    guard = cg.guard_from_environment({cg.MAX_CONNECTIONS_ENV: "250"}, soft_nofile=256)
    assert guard.max_connections == 128  # the default for 256, not the unusable 250
    assert any(cg.MAX_CONNECTIONS_ENV in r.getMessage() for r in caplog.records)
    ok = cg.guard_from_environment({cg.MAX_CONNECTIONS_ENV: "192"}, soft_nofile=256)
    assert ok.max_connections == 192


def test_a_descriptor_limit_too_low_for_even_the_default_cap_is_warned_about(caplog):
    caplog.set_level(logging.WARNING, logger="backend.src.core.connection_guard")
    guard = cg.guard_from_environment({}, soft_nofile=64)
    assert guard.max_connections == 32  # the floor; the server still starts
    assert any("RLIMIT_NOFILE" in r.getMessage() for r in caplog.records)


def test_the_descriptor_limit_is_read_from_the_process_and_unlimited_is_finite(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (4096, 4096))
    assert cg.soft_nofile_limit() == 4096
    monkeypatch.setattr(
        resource, "getrlimit", lambda _which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    assert cg.soft_nofile_limit() == 1 << 20
    assert cg.default_max_connections(cg.soft_nofile_limit()) == 1024


def test_an_unreadable_descriptor_limit_falls_back_to_1024_and_never_raises(monkeypatch, caplog):
    def broken(_which):
        raise OSError("no")

    monkeypatch.setattr(resource, "getrlimit", broken)
    assert cg.soft_nofile_limit() == 1024
    assert any("RLIMIT_NOFILE" in r.getMessage() for r in caplog.records)


def test_the_process_guard_is_built_once_from_the_environment_and_can_be_replaced(monkeypatch):
    assert cg.process_snapshot() is None  # nothing built yet
    monkeypatch.setenv(cg.MAX_CONNECTIONS_ENV, "40")
    monkeypatch.setenv(cg.CONNECTION_RESERVE_ENV, "10")
    first = cg.process_guard()
    assert first is cg.process_guard()
    assert (first.max_connections, first.reserve) == (40, 10)
    assert cg.process_snapshot()["max_connections"] == 40
    replacement = guard_of(8, 2)
    cg.reset_process_guard(replacement)
    assert cg.process_guard() is replacement


@pytest.mark.parametrize(
    "max_connections, backlog, nofile, fits",
    [
        (1024, 2048, 4096, True),  # the shipped unit with uvicorn's default backlog
        (1024, 128, 4096, True),
        (128, 2048, 256, False),  # the review's ulimit -n 256 with the default backlog
        (128, 64, 256, True),
        (512, 2048, 1024, False),  # systemd's default soft limit with the default backlog
        (1024, None, 64, True),  # no backlog known: nothing to say
    ],
)
def test_the_descriptor_budget_check_says_when_a_burst_can_outrun_the_table(
    max_connections, backlog, nofile, fits
):
    problem = cg.descriptor_budget_problem(max_connections, backlog, nofile)
    assert (problem is None) is fits
    if problem:
        assert "--backlog" in problem and str(nofile) in problem


# --------------------------------------------------------------------------------------
# the uvicorn protocol, driven with a fake transport on a real event loop
# --------------------------------------------------------------------------------------


class FakeTransport:
    """Just enough of ``asyncio.Transport`` for uvicorn's protocols."""

    def __init__(self) -> None:
        self.aborted = False
        self.closed = False
        self.written: list[bytes] = []
        self.protocol = None

    def get_extra_info(self, name, default=None):
        return {"sockname": ("127.0.0.1", 8081), "peername": ("192.0.2.7", 40000)}.get(
            name, default
        )

    def is_closing(self) -> bool:
        return self.aborted or self.closed

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def pause_reading(self) -> None:
        pass

    def resume_reading(self) -> None:
        pass

    def get_protocol(self):
        return self.protocol

    def set_protocol(self, protocol) -> None:
        self.protocol = protocol


class StubWebSocketProtocol:
    """Takes over the transport when a WebSocket upgrade is accepted (records it, nothing else)."""

    instances: list[StubWebSocketProtocol] = []

    def __init__(self, **_kw) -> None:
        self.made = False
        self.data = b""
        StubWebSocketProtocol.instances.append(self)

    def connection_made(self, transport) -> None:
        self.made = True

    def data_received(self, data: bytes) -> None:
        self.data += data


async def ok_app(scope, receive, send):
    """An application that answers 200 once it has read the whole request body."""
    if scope["type"] != "http":
        return
    while True:
        message = await receive()
        if message["type"] != "http.request" or not message.get("more_body"):
            break
    ok_app.paths.append(scope["path"])
    await send(
        {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
    )
    await send({"type": "http.response.body", "body": b"ok"})


ok_app.paths = []

BASES = [pytest.param(HttpToolsProtocol, id="httptools"), pytest.param(H11Protocol, id="h11")]


def build(base, guard: ConnectionGuard, *, ws="none", app=None):
    """One guarded protocol instance on top of ``base`` plus its (not yet connected) transport."""
    cg.reset_process_guard(guard)
    cls = type(f"Guarded{base.__name__}", (up.GuardedHttpProtocolMixin, base), {})
    config = uvicorn.Config(app or ok_app, http=cls, ws=ws, lifespan="off", log_level="critical")
    config.load()
    proto = cls(
        config=config,
        server_state=ServerState(),
        app_state={},
        _loop=asyncio.get_running_loop(),
    )
    return proto, FakeTransport()


def connect(base, guard, **kw):
    proto, transport = build(base, guard, **kw)
    proto.connection_made(transport)
    return proto, transport


def get(path="/api/v1/status", extra=b"") -> bytes:
    return b"GET " + path.encode() + b" HTTP/1.1\r\nHost: t\r\n" + extra + b"\r\n"


def post_head(path: str, length: int) -> bytes:
    return (
        b"POST " + path.encode() + b" HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
        b"Content-Length: " + str(length).encode() + b"\r\n\r\n"
    )


UPGRADE = get(
    "/ws",
    b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
    b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n",
)


@pytest.fixture
def proto_log(monkeypatch, caplog):
    """Capture the protocol module's log lines (uvicorn's own logger config would eat them)."""
    logger = logging.getLogger("test.uvicorn_protocol")
    monkeypatch.setattr(up, "logger", logger)
    caplog.set_level(logging.DEBUG, logger="test.uvicorn_protocol")
    return caplog


def test_the_exported_class_loads_from_a_real_uvicorn_config_and_extends_what_auto_picks():
    config = uvicorn.Config(ok_app, http=up.GUARDED_PROTOCOL, lifespan="off", log_level="critical")
    config.load()
    cls = config.http_protocol_class
    assert cls is up.LawnBerryHttpProtocol
    assert issubclass(cls, up.AutoHTTPProtocol)
    assert issubclass(cls, up.GuardedHttpProtocolMixin)
    assert up.GUARDED_PROTOCOL == "backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol"


def test_the_exported_class_is_constructible_the_way_uvicorn_constructs_it():
    cg.reset_process_guard(guard_of(8, 2))
    config = uvicorn.Config(ok_app, http=up.GUARDED_PROTOCOL, lifespan="off", log_level="critical")
    config.load()
    loop = asyncio.new_event_loop()
    try:
        proto = config.http_protocol_class(
            config=config, server_state=ServerState(), app_state={}, _loop=loop
        )
    finally:
        loop.close()
    assert proto._guard is cg.process_guard()
    assert isinstance(proto.app, up._GuardedApp)


@pytest.mark.parametrize("base", BASES)
async def test_a_connection_is_registered_on_connect_and_forgotten_on_loss(base):
    guard = guard_of(8, 2)
    proto, transport = connect(base, guard)
    assert len(guard) == 1 and guard.snapshot()["unproductive"] == 1
    assert proto._deadline_handle is not None  # the header deadline is armed
    proto.connection_lost(None)
    assert len(guard) == 0
    assert proto._deadline_handle is None  # and cancelled


@pytest.mark.parametrize("base", BASES)
async def test_a_complete_request_makes_the_connection_productive(base):
    guard = guard_of(8, 2)
    proto, transport = connect(base, guard)
    proto.data_received(get())
    assert guard.is_productive(proto)
    assert proto._deadline_handle is None
    await asyncio.sleep(0.01)  # the application task answers the request
    assert b"200 OK" in b"".join(transport.written)


@pytest.mark.parametrize("base", BASES)
async def test_an_incomplete_request_line_or_header_is_not_productive(base):
    guard = guard_of(8, 2)
    proto, _ = connect(base, guard)
    proto.data_received(b"GET /api/v1/status HTT")
    proto.data_received(b"P/1.1\r\nHost: t\r\nX-Slow: a")
    assert not guard.is_productive(proto)
    assert proto._deadline_handle is not None  # still racing the header deadline
    proto.data_received(b"\r\n\r\n")  # the headers finish: now it is a request
    assert guard.is_productive(proto)


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("path", ["/api/v1/jobs", ESTOP_PATH])
async def test_a_request_whose_body_never_arrives_stays_evictable_even_on_the_estop_path(
    base, path
):
    """``POST /control/emergency`` with a ``Content-Length`` and no body must not pin the table."""
    guard = guard_of(8, 2)
    proto, _ = connect(base, guard)
    proto.data_received(post_head(path, 10))
    assert not guard.is_productive(proto)
    # the headers are in, so the header deadline is over; the BODY has a deadline of its own now
    # (SEC3-02: a slow upload used to be left alone for ever)
    assert proto._deadline_kind == "body"
    proto.data_received(b"{" + b" " * 3)  # part of the body
    assert not guard.is_productive(proto)
    proto.data_received(b" " * 5 + b"}")  # the rest
    assert guard.is_productive(proto)
    assert proto._deadline_handle is None  # the body arrived in time: nothing left to race


@pytest.mark.parametrize("base", BASES)
async def test_the_header_deadline_aborts_a_connection_that_never_sends_a_request(base):
    guard = guard_of(8, 2, deadline=0.05)
    proto, transport = connect(base, guard)
    await asyncio.sleep(0.2)
    assert transport.aborted
    assert guard.deadline_closed == 1


@pytest.mark.parametrize("base", BASES)
async def test_the_header_deadline_is_total_so_dribbling_one_byte_at_a_time_does_not_extend_it(
    base,
):
    guard = guard_of(8, 2, deadline=0.1)
    proto, transport = connect(base, guard)
    payload = b"GET /x HTTP/1.1\r\nHost: t\r\nX-Slow: " + b"a" * 100
    for i in range(30):
        if transport.aborted:
            break
        proto.data_received(payload[i : i + 1])
        await asyncio.sleep(0.02)
    assert transport.aborted
    assert guard.deadline_closed == 1


@pytest.mark.parametrize("base", BASES)
async def test_a_request_cancels_the_header_deadline_so_a_slow_response_is_left_alone(base):
    guard = guard_of(8, 2, deadline=0.08)
    proto, transport = connect(base, guard)
    await asyncio.sleep(0.02)
    proto.data_received(get())
    await asyncio.sleep(0.25)  # well past the deadline; the application has answered long ago
    assert not transport.aborted
    assert guard.deadline_closed == 0
    # the response is finished: the connection waits for its next request, evictable again
    assert not guard.is_productive(proto) and len(guard) == 1
    # a timer that fires anyway (queued before the request began) must not cut the connection
    proto._on_deadline("headers")
    assert not transport.aborted and guard.deadline_closed == 0


@pytest.mark.parametrize("base", BASES)
async def test_an_accepted_websocket_upgrade_gives_the_slot_back(base):
    guard = guard_of(8, 2)
    StubWebSocketProtocol.instances.clear()
    proto, transport = connect(base, guard, ws=StubWebSocketProtocol)
    proto.data_received(UPGRADE)
    assert StubWebSocketProtocol.instances and StubWebSocketProtocol.instances[0].made
    assert len(guard) == 0  # upgraded sockets are capped by the WebSocket registry, not here
    assert proto._deadline_handle is None
    proto.connection_lost(None)  # harmless if it ever runs
    assert len(guard) == 0


@pytest.mark.parametrize("base", BASES)
async def test_a_refused_connection_is_aborted_at_once(base):
    guard = guard_of(3, 1)
    for name in "ab":
        assert guard.mark_productive(admit(guard, name))
    assert guard.mark_productive(admit(guard, "c"), estop=True)
    proto, transport = connect(base, guard)
    assert transport.aborted
    assert guard.refused == 1 and len(guard) == 3
    proto.connection_lost(None)  # what asyncio does next: nothing to release
    assert len(guard) == 3


@pytest.mark.parametrize("base", BASES)
async def test_at_the_cap_a_new_connection_aborts_the_oldest_silent_one(base):
    guard = guard_of(2, 1)
    first, first_transport = connect(base, guard)
    second, second_transport = connect(base, guard)
    third, third_transport = connect(base, guard)
    assert first_transport.aborted
    assert not second_transport.aborted and not third_transport.aborted
    assert guard.evicted == 1 and len(guard) == 2
    first.connection_lost(None)  # asyncio reports the abort: the slot is already gone
    assert len(guard) == 2


@pytest.mark.parametrize("base", BASES)
async def test_a_connection_that_has_made_a_request_is_not_evicted(base):
    guard = guard_of(2, 1)
    busy, busy_transport = connect(base, guard)
    busy.data_received(get())
    idle, idle_transport = connect(base, guard)
    newcomer, newcomer_transport = connect(base, guard)
    assert not busy_transport.aborted
    assert idle_transport.aborted
    assert not newcomer_transport.aborted


@pytest.mark.parametrize("base", BASES)
async def test_an_estop_request_is_protected_even_when_the_quota_is_full(base):
    """Admitted past the quota, the E-stop's connection must not be evictable while it is served."""
    guard = guard_of(6, 2)  # soft cap 4
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    estop, _ = connect(base, guard)
    estop.data_received(post_head(ESTOP_PATH, 2) + b"{}")
    other, _ = connect(base, guard)
    other.data_received(post_head("/api/v1/jobs", 2) + b"{}")
    assert guard.is_productive(estop)
    assert not guard.is_productive(other)  # an ordinary request stays outside the quota


# ---- the ASGI wrapper ---------------------------------------------------------------


def http_scope(path: str, **extra) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "root_path": "",
        **extra,
    }


async def run_app(proto, scope):
    """Call the guarded application of ``proto``: what it sent and which paths the real app saw."""
    sent: list[dict] = []
    ok_app.paths.clear()

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await proto.app(scope, receive, send)
    return sent, list(ok_app.paths)


@pytest.mark.parametrize("base", BASES)
async def test_while_the_quota_is_full_a_non_estop_request_gets_503_before_the_app_runs(base):
    guard = guard_of(6, 2)  # soft cap 4
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, _ = connect(base, guard)
    sent, ran = await run_app(proto, http_scope("/api/v1/jobs"))
    assert ran == []
    start, body = sent
    assert start["status"] == 503
    headers = dict(start["headers"])
    assert headers[b"connection"] == b"close"
    assert headers[b"retry-after"] == b"1"
    assert headers[b"content-length"] == str(len(body["body"])).encode()
    assert json.loads(body["body"])["detail"]
    assert guard.shed == 1


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize(
    "path",
    [
        "/api/v2/control/emergency",
        "/api/v2/control/emergency-stop",
        "/api/v2/tractor/emergency-stop",
        "/api/v2/tractor/emergency-stop/",
    ],
)
async def test_the_estop_paths_are_served_even_when_the_quota_is_full(base, path):
    guard = guard_of(6, 2)
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, _ = connect(base, guard)
    sent, ran = await run_app(proto, http_scope(path))
    assert ran == [path] and sent[0]["status"] == 200
    assert guard.shed == 0


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize(
    "path",
    [
        "/api/v2/control/emergency_clear",  # authenticated, must stay limited
        "/api/v2/tractor/clear-emergency",
        "/api/v2/control/emergency/extra",
        "/api/v2/control/emergencyx",
    ],
)
async def test_only_the_exact_estop_paths_are_exempt_not_their_prefixes(base, path):
    guard = guard_of(6, 2)
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, _ = connect(base, guard)
    sent, ran = await run_app(proto, http_scope(path))
    assert ran == [] and sent[0]["status"] == 503


@pytest.mark.parametrize("base", BASES)
async def test_an_estop_under_a_root_path_prefix_is_still_exempt(base):
    guard = guard_of(6, 2)
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, _ = connect(base, guard)
    scope = http_scope("/lawnberry" + ESTOP_PATH, root_path="/lawnberry")
    sent, ran = await run_app(proto, scope)
    assert ran == ["/lawnberry" + ESTOP_PATH] and sent[0]["status"] == 200


@pytest.mark.parametrize("base", BASES)
async def test_the_request_that_fills_the_quota_is_itself_served_not_shed(base):
    """Productive means "a complete request is in flight": the one that takes the last quota
    slot must not be answered 503 by the check that sees the quota full right after it."""
    guard = guard_of(6, 2)  # soft cap 4
    for i in range(3):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, transport = connect(base, guard)
    proto.data_received(get())  # delivers a complete request: the 4th and last quota slot
    assert guard.is_productive(proto) and guard.over_soft_cap()
    await asyncio.sleep(0.02)  # the application task answers
    written = b"".join(transport.written)
    assert b"200 OK" in written and b"503" not in written
    assert guard.shed == 0


@pytest.mark.parametrize("base", BASES)
async def test_below_the_quota_nothing_is_shed(base):
    guard = guard_of(6, 2)
    proto, _ = connect(base, guard)
    sent, ran = await run_app(proto, http_scope("/api/v1/jobs"))
    assert ran == ["/api/v1/jobs"] and sent[0]["status"] == 200


@pytest.mark.parametrize("base", BASES)
async def test_lifespan_and_websocket_scopes_pass_through_untouched(base):
    guard = guard_of(6, 2)
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, _ = connect(base, guard)
    seen = []

    async def spy(scope, receive, send):
        seen.append(scope["type"])

    proto.app._app = spy
    for scope_type in ("lifespan", "websocket"):
        await proto.app({"type": scope_type, "path": "/anything"}, None, None)
    assert seen == ["lifespan", "websocket"]


# ---- never raise into uvicorn ----------------------------------------------------------


class NoUpgradeBase(asyncio.Protocol):
    """A uvicorn that lacks ``handle_websocket_upgrade`` (a future release that renamed it)."""

    def __init__(self, config=None, server_state=None, app_state=None, _loop=None) -> None:
        self.app = object()
        self.loop = _loop
        self.transport = None
        self.cycle = None

    def connection_made(self, transport) -> None:
        self.transport = transport

    def connection_lost(self, exc) -> None:
        pass

    def data_received(self, data) -> None:
        pass


async def test_an_uvicorn_without_the_expected_internals_runs_unguarded_and_says_so_once(
    proto_log,
):
    guard = guard_of(8, 2)
    cg.reset_process_guard(guard)
    cls = type("Guarded", (up.GuardedHttpProtocolMixin, NoUpgradeBase), {})
    first, second = cls(_loop=asyncio.get_running_loop()), cls(_loop=asyncio.get_running_loop())
    transport = FakeTransport()
    first.connection_made(transport)
    first.data_received(b"anything")
    first.connection_lost(None)
    assert first._guard is None and second._guard is None
    assert len(guard) == 0
    errors = [r for r in proto_log.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and "handle_websocket_upgrade" in errors[0].getMessage()


@pytest.mark.parametrize("base", BASES)
async def test_a_guard_failure_degrades_to_no_guard_and_never_reaches_uvicorn(
    base, proto_log, monkeypatch
):
    guard = guard_of(8, 2)
    proto, transport = connect(base, guard)

    def explode(*_a, **_kw):
        raise RuntimeError("bug in the guard")

    monkeypatch.setattr(guard, "mark_productive", explode)
    proto.data_received(get())  # must not raise
    await asyncio.sleep(0.01)
    assert b"200 OK" in b"".join(transport.written)  # the request was served regardless
    assert proto._guard is None and len(guard) == 0
    assert proto._deadline_handle is None
    errors = [r for r in proto_log.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    # the wrapper no longer sheds anything on an unguarded connection
    sent, ran = await run_app(proto, http_scope("/api/v1/jobs"))
    assert ran == ["/api/v1/jobs"]


@pytest.mark.parametrize("base", BASES)
async def test_connection_lost_still_reaches_uvicorn_when_the_guard_cannot_release(
    base, proto_log, monkeypatch
):
    guard = guard_of(8, 2)
    proto, _ = connect(base, guard)
    server_state = proto.server_state
    assert proto in server_state.connections

    def explode(_conn):
        raise RuntimeError("bug in the guard")

    monkeypatch.setattr(guard, "release", explode)
    proto.connection_lost(None)  # must not raise into asyncio
    assert proto not in server_state.connections  # uvicorn's own bookkeeping ran regardless
    assert sum(r.levelno >= logging.ERROR for r in proto_log.records) == 1


@pytest.mark.parametrize("base", BASES)
async def test_a_register_failure_leaves_the_connection_served_and_unguarded(
    base, proto_log, monkeypatch
):
    guard = guard_of(8, 2)

    def explode(*_a, **_kw):
        raise RuntimeError("bug in the guard")

    monkeypatch.setattr(guard, "register", explode)
    proto, transport = connect(base, guard)
    assert not transport.aborted and proto._guard is None
    proto.data_received(get())
    await asyncio.sleep(0.01)
    assert b"200 OK" in b"".join(transport.written)


async def test_the_listener_budget_warning_is_logged_once_per_process(proto_log, monkeypatch):
    monkeypatch.setattr(cg, "soft_nofile_limit", lambda: 256)
    guard = guard_of(128, 32)  # 128 + the default backlog 2048 + 64 does not fit in 256
    for _ in range(3):
        connect(HttpToolsProtocol, guard)
    warnings = [r for r in proto_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "--backlog" in warnings[0].getMessage()


# --------------------------------------------------------------------------------------
# SEC3-02: the state follows the REQUEST, an E-stop is a POST, every request has a deadline
# --------------------------------------------------------------------------------------


def guard_with(max_connections=8, reserve=2, deadline=10.0, body_deadline=10.0, **kw):
    return ConnectionGuard(max_connections, reserve, deadline, body_deadline_s=body_deadline, **kw)


async def early_app(scope, receive, send):
    """Answers at once and never reads the request body (a stop that does not need it, a 413)."""
    if scope["type"] != "http":
        return
    await send(
        {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
    )
    await send({"type": "http.response.body", "body": b"ok"})


def slow_app(gate: asyncio.Event):
    """An application whose response only finishes once ``gate`` is set (a streaming response)."""

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x", "more_body": True})
        await gate.wait()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


SECOND_REQUEST_START = b"GET /api/v1/second HTTP/1.1\r\nHost: t\r\nX-Hold: a"  # headers unfinished


def test_a_connection_that_went_idle_is_the_newest_evictable_one():
    clock = Clock(100.0)
    guard = guard_of(3, 1, clock=clock)
    a, b, c = (admit(guard, name) for name in "abc")
    assert guard.mark_productive(a)  # a arrived first and is now being served
    clock.now = 200.0
    assert guard.mark_idle(a) is True  # its response is done
    assert not guard.is_productive(a) and len(guard) == 3
    # a is the oldest by arrival but the newest by idleness: b goes first, then c, then a
    evicted_after_each = []
    for i in range(3):
        newcomer = Conn(f"n{i}")
        assert guard.register(newcomer, newcomer.close)
        evicted_after_each.append([conn.name for conn in (a, b, c) if conn.closed])
    assert evicted_after_each == [["b"], ["b", "c"], ["a", "b", "c"]]


def test_mark_idle_ignores_connections_that_are_not_productive_or_not_tracked():
    guard = guard_of(4, 1)
    silent = admit(guard, "silent")
    assert guard.mark_idle(silent) is False and not guard.is_productive(silent)
    assert guard.mark_idle(Conn("stranger")) is False
    assert len(guard) == 1


def test_a_connection_that_went_idle_gives_its_quota_slot_back():
    guard = guard_of(4, 1)  # soft cap 3
    conns = [admit(guard, str(i)) for i in range(3)]
    for conn in conns:
        assert guard.mark_productive(conn)
    fourth = admit(guard, "fourth")
    assert guard.over_soft_cap() and guard.mark_productive(fourth) is False
    guard.mark_idle(conns[0])
    assert not guard.over_soft_cap()
    assert guard.mark_productive(fourth) is True


def test_when_every_connection_has_gone_idle_a_newcomer_evicts_one_instead_of_being_refused():
    guard = guard_of(3, 1)  # soft cap 2
    a, b, c = (admit(guard, name) for name in "abc")
    assert guard.mark_productive(a) and guard.mark_productive(b)
    assert guard.mark_productive(c, estop=True)  # a stop being served, beyond the quota
    refused = Conn("refused")
    assert guard.register(refused, refused.close) is False and guard.refused == 1
    for conn in (a, b, c):
        guard.mark_idle(conn)
    admitted = Conn("admitted")
    assert guard.register(admitted, admitted.close) is True
    assert guard.refused == 1 and guard.evicted == 1 and a.closed == 1


@pytest.mark.parametrize("base", BASES)
async def test_a_table_of_clients_that_each_started_a_second_request_still_admits_a_newcomer(base):
    """The review's scenario: every connection completed ONE request, then started a second that it
    never finishes. The first request used to make the connection productive for ever, so the
    table was full of untouchable connections and the E-stop's connection was refused."""
    guard = guard_of(4, 1)
    holders = [connect(base, guard) for _ in range(4)]
    for proto, _ in holders:
        proto.data_received(get())
    await asyncio.sleep(0.05)  # every application task has answered
    for proto, _ in holders:
        proto.data_received(SECOND_REQUEST_START)
    newcomer, newcomer_transport = connect(base, guard)
    assert not newcomer_transport.aborted
    assert guard.refused == 0 and guard.evicted == 1
    assert sum(transport.aborted for _, transport in holders) == 1


@pytest.mark.parametrize("base", BASES)
async def test_the_header_deadline_runs_again_for_every_request_of_a_keep_alive_connection(base):
    guard = guard_of(8, 2, deadline=0.08)
    proto, transport = connect(base, guard)
    proto.data_received(get())
    await asyncio.sleep(0.03)  # the first request is served
    assert not transport.aborted
    proto.data_received(SECOND_REQUEST_START)  # the first bytes of the next request, then silence
    await asyncio.sleep(0.25)
    assert transport.aborted and guard.deadline_closed == 1


@pytest.mark.parametrize("base", BASES)
async def test_the_second_requests_header_deadline_is_total_and_ends_when_its_headers_do(base):
    # dribbling: one more byte every 20 ms does not keep a request alive past its deadline
    guard = guard_of(8, 2, deadline=0.1)
    proto, transport = connect(base, guard)
    proto.data_received(get())
    await asyncio.sleep(0.03)
    proto.data_received(SECOND_REQUEST_START)
    for _ in range(30):
        if transport.aborted:
            break
        proto.data_received(b"b")
        await asyncio.sleep(0.02)
    assert transport.aborted and guard.deadline_closed == 1

    # headers that complete in time end the deadline: the same connection is then left alone
    guard = guard_of(8, 2, deadline=0.1)
    proto, transport = connect(base, guard)
    proto.data_received(get())
    await asyncio.sleep(0.03)
    proto.data_received(SECOND_REQUEST_START)
    proto.data_received(b"\r\n\r\n")
    await asyncio.sleep(0.3)
    assert not transport.aborted and guard.deadline_closed == 0


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("path", ["/api/v1/jobs", ESTOP_PATH])
async def test_a_request_body_that_does_not_arrive_in_time_closes_the_connection(base, path):
    guard = guard_with(deadline=10.0, body_deadline=0.08)
    proto, transport = connect(base, guard)
    proto.data_received(post_head(path, 100) + b"{")
    assert not transport.aborted
    await asyncio.sleep(0.25)
    assert transport.aborted and guard.deadline_closed == 1
    assert len(guard) == 1  # the abort is asyncio's to report: connection_lost releases the slot
    proto.connection_lost(None)
    assert len(guard) == 0


@pytest.mark.parametrize("base", BASES)
async def test_a_dribbled_body_does_not_extend_the_body_deadline(base):
    guard = guard_with(body_deadline=0.1)
    proto, transport = connect(base, guard)
    proto.data_received(post_head("/api/v1/jobs", 1000))
    for _ in range(30):
        if transport.aborted:
            break
        proto.data_received(b"x")
        await asyncio.sleep(0.02)
    assert transport.aborted and guard.deadline_closed == 1


@pytest.mark.parametrize("base", BASES)
async def test_a_body_that_arrives_in_time_is_left_alone(base):
    guard = guard_with(body_deadline=0.1)
    proto, transport = connect(base, guard)
    proto.data_received(post_head("/api/v1/jobs", 10) + b"{   ")
    await asyncio.sleep(0.03)
    proto.data_received(b"     }")  # 4 + 6 = the announced 10 bytes
    await asyncio.sleep(0.25)
    assert not transport.aborted and guard.deadline_closed == 0


@pytest.mark.parametrize("base", BASES)
async def test_a_response_sent_before_the_body_ended_keeps_the_body_deadline(base):
    """A 413 or a stop that ignores its body answers at once; the body may never come."""
    guard = guard_with(body_deadline=0.1)
    proto, transport = connect(base, guard, app=early_app)
    proto.data_received(post_head("/api/v1/jobs", 100) + b"{")
    await asyncio.sleep(0.03)
    assert b"200 OK" in b"".join(transport.written)  # answered
    assert not transport.aborted
    await asyncio.sleep(0.25)
    assert transport.aborted and guard.deadline_closed == 1


@pytest.mark.parametrize("base", BASES)
async def test_a_streaming_response_keeps_the_slot_until_it_ends(base):
    gate = asyncio.Event()
    guard = guard_of(6, 2)
    proto, _ = connect(base, guard, app=slow_app(gate))
    proto.data_received(get())
    await asyncio.sleep(0.03)
    assert guard.is_productive(proto)  # the response is still in flight
    gate.set()
    await asyncio.sleep(0.03)
    assert not guard.is_productive(proto) and len(guard) == 1  # finished: evictable again


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("method", [b"GET", b"HEAD", b"PUT", b"DELETE", b"OPTIONS", b"PATCH"])
async def test_only_a_post_to_an_estop_path_earns_priority_over_the_quota(base, method):
    guard = guard_of(8, 2)  # soft cap 6
    for i in range(6):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    other, _ = connect(base, guard)
    other.data_received(method + b" " + ESTOP_PATH.encode() + b" HTTP/1.1\r\nHost: t\r\n\r\n")
    assert not guard.is_productive(other)
    assert guard.over_quota == 1
    stop, _ = connect(base, guard)
    stop.data_received(post_head(ESTOP_PATH, 2) + b"{}")
    assert guard.is_productive(stop)


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "DELETE", "OPTIONS", "PATCH"])
@pytest.mark.parametrize("path", ["/api/v2/control/emergency", "/api/v2/tractor/emergency-stop"])
async def test_a_request_to_an_estop_path_that_is_not_a_post_is_shed_like_any_other(
    base, method, path
):
    guard = guard_of(6, 2)
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    proto, _ = connect(base, guard)
    sent, ran = await run_app(proto, http_scope(path, method=method))
    assert ran == [] and sent[0]["status"] == 503 and guard.shed == 1


@pytest.mark.parametrize("base", BASES)
async def test_an_estop_post_on_a_served_keep_alive_connection_is_served_at_full_quota(base):
    guard = guard_of(6, 2)
    proto, transport = connect(base, guard)
    proto.data_received(get())
    await asyncio.sleep(0.03)  # the first request is done; the connection waits for the next
    for i in range(4):
        assert guard.mark_productive(admit(guard, f"p{i}"))
    assert guard.over_soft_cap()
    transport.written.clear()
    proto.data_received(post_head(ESTOP_PATH, 2) + b"{}")
    assert guard.is_productive(proto)
    await asyncio.sleep(0.03)
    assert b"200 OK" in b"".join(transport.written) and guard.shed == 0


@pytest.mark.parametrize("bad", ["0.9", "301", "300.5", "abc", "-5", "nan", "10s", "0", ".5"])
def test_a_bad_body_deadline_logs_one_error_naming_the_variable_and_uses_fifteen_seconds(
    bad, caplog
):
    caplog.set_level(logging.DEBUG, logger="backend.src.core.connection_guard")
    guard = cg.guard_from_environment({cg.BODY_DEADLINE_ENV: bad}, soft_nofile=4096)
    assert guard.body_deadline_s == 15.0
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and cg.BODY_DEADLINE_ENV in errors[0].getMessage()


@pytest.mark.parametrize("good, expected", [("1", 1.0), ("2.5", 2.5), ("300", 300.0)])
def test_the_body_deadline_accepts_its_whole_range_inclusive(good, expected, caplog):
    guard = cg.guard_from_environment({cg.BODY_DEADLINE_ENV: good}, soft_nofile=4096)
    assert guard.body_deadline_s == expected and guard.header_deadline_s == 10.0
    assert not caplog.records


@pytest.mark.parametrize("deadline", [0, -1.0, float("nan"), float("inf")])
def test_the_constructor_refuses_a_body_deadline_that_is_not_a_positive_finite_number(deadline):
    with pytest.raises(ValueError, match="body_deadline_s"):
        ConnectionGuard(8, 2, 10.0, body_deadline_s=deadline)
