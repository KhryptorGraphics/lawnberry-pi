"""SEC20-04 end to end: a real uvicorn, a lowered descriptor limit and a real connection flood.

The finding: uvicorn never reaps a connection that does not send a request, so a LAN client that
opens enough idle sockets takes every descriptor of the backend and the E-stop POST is reset.
Here a stub ASGI app that answers the E-stop paths with 202 runs under the REAL uvicorn CLI in a
subprocess whose ``RLIMIT_NOFILE`` is lowered with ``preexec_fn`` (never the pytest process's), and
the SAME flood is pointed at

* uvicorn's stock protocol (the control: the E-stop must fail, which is the bug), and
* ``backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol`` passed to ``--http`` exactly as the
  systemd unit passes it (plus the h11 implementation, which uvicorn picks when httptools is not
  installed), on the default uvloop and on the plain asyncio loop.

The stub reads the whole request body before it answers, like FastAPI parsing a body, so a request
whose body never arrives really does park the application.
"""

from __future__ import annotations

import json
import os
import resource
import select
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="counts descriptors in /proc and lowers RLIMIT_NOFILE in the child process",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
GUARDED = "backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol"
ESTOP = "/api/v2/tractor/emergency-stop"
ESTOP_BODY_PATH = "/api/v2/control/emergency"
# The server's descriptor limit. Small enough that a few hundred idle sockets exhaust a stock
# uvicorn, large enough that max_connections (24) + the guard's 64-descriptor headroom fit.
NOFILE = 192
SMALL = {
    "LAWN_BERRY_MAX_CONNECTIONS": "24",
    "LAWN_BERRY_CONNECTION_RESERVE": "4",
    "LAWN_BERRY_HEADER_DEADLINE_S": "2",
}
FLOOD = 300

STUB_APP = textwrap.dedent(
    """
    import json

    from uvicorn.protocols.http.h11_impl import H11Protocol
    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

    from backend.src.core import connection_guard
    from backend.src.core.uvicorn_protocol import GuardedHttpProtocolMixin

    ESTOP_PATHS = {
        "/api/v2/control/emergency",
        "/api/v2/control/emergency-stop",
        "/api/v2/tractor/emergency-stop",
    }


    class GuardH11(GuardedHttpProtocolMixin, H11Protocol):
        pass


    class GuardHttptools(GuardedHttpProtocolMixin, HttpToolsProtocol):
        pass


    async def read_body(receive):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return False
            if not message.get("more_body"):
                return True


    async def reply(send, status, payload=b"ok", content_type=b"text/plain"):
        headers = [(b"content-type", content_type), (b"content-length", str(len(payload)).encode())]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})


    async def app(scope, receive, send):
        if scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.accept"})
            while (await receive())["type"] != "websocket.disconnect":
                pass
            return
        if scope["type"] != "http":
            return
        if not await read_body(receive):
            return
        path = scope["path"]
        if path in ESTOP_PATHS and scope["method"] == "POST":
            await reply(send, 202, b'{"stopped":true}', b"application/json")
        elif path == "/hold":  # a response that is still in flight: a productive connection
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"x", "more_body": True})
            while (await receive())["type"] != "http.disconnect":
                pass
        elif path == "/_guard":
            snapshot = json.dumps(connection_guard.process_snapshot()).encode()
            await reply(send, 200, snapshot, b"application/json")
        else:
            await reply(send, 200)
    """
)


@dataclass
class Reply:
    status: int | None
    elapsed: float
    error: str | None = None
    headers: dict[str, str] | None = None
    body: bytes = b""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


def exchange(port: int, raw: bytes, timeout: float = 3.0) -> Reply:
    """Send one request on a fresh connection and read to EOF (the request says ``close``)."""
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
            conn.settimeout(timeout)
            conn.sendall(raw)
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
    except OSError as exc:  # reset, refused, timed out: all of them are "no answer"
        return Reply(None, time.monotonic() - started, type(exc).__name__)
    elapsed = time.monotonic() - started
    if not data:
        return Reply(None, elapsed, "EmptyReply")
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    headers = {
        k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])
    }
    return Reply(int(lines[0].split(" ", 2)[1]), elapsed, None, headers, body)


def post(port: int, path: str = ESTOP, timeout: float = 3.0) -> Reply:
    raw = (
        f"POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
        "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
    ).encode()
    return exchange(port, raw, timeout)


def get(port: int, path: str, timeout: float = 3.0) -> Reply:
    raw = f"GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".encode()
    return exchange(port, raw, timeout)


def flood_idle(port: int, count: int, *, chunk: int = 25, pause: float = 0.01, send=b""):
    """Open ``count`` connections (``chunk`` per ``pause`` s), send ``send`` on each, hold them."""
    held = []
    for i in range(count):
        conn = socket.create_connection(("127.0.0.1", port), timeout=3)
        if send:
            conn.sendall(send)
        held.append(conn)
        if (i + 1) % chunk == 0:
            time.sleep(pause)
    return held


def closed_by_peer(conn: socket.socket) -> bool:
    """``True`` once the server closed (or reset) this connection; ``False`` while it is open."""
    conn.settimeout(0.05)
    try:
        return conn.recv(1) == b""
    except TimeoutError:
        return False
    except OSError:
        return True


def closure_times(conns: list[socket.socket], started: float, timeout: float) -> dict[int, float]:
    """Seconds after ``started`` at which the server closed each connection (index -> time)."""
    closed: dict[int, float] = {}
    while len(closed) < len(conns) and time.monotonic() - started < timeout:
        pending = [(i, c) for i, c in enumerate(conns) if i not in closed]
        readable, _, _ = select.select([c for _, c in pending], [], [], 0.05)
        for i, conn in pending:
            if conn in readable:
                try:
                    gone = conn.recv(1) == b""
                except OSError:
                    gone = True
                if gone:
                    closed[i] = time.monotonic() - started
    return closed


def wait_until(predicate, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Server:
    """The real uvicorn CLI in a subprocess with a lowered ``RLIMIT_NOFILE``."""

    def __init__(
        self,
        directory: Path,
        http: str,
        *,
        nofile: int = NOFILE,
        env: dict[str, str] | None = None,
        extra: tuple[str, ...] = (),
    ) -> None:
        self.directory = directory
        self.http = http
        self.nofile = nofile
        self.extra_env = env or {}
        self.extra = extra
        self.port = free_port()
        self.log = directory / "server.log"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> Server:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("LAWN_BERRY_", "UVICORN_")) and k != "PYTEST_ADDOPTS"
        }
        env.update(self.extra_env)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        limit = self.nofile
        command = [
            sys.executable, "-m", "uvicorn", "stub_app:app", "--app-dir", str(self.directory),
            "--host", "127.0.0.1", "--port", str(self.port), "--lifespan", "off",
            "--no-access-log", "--log-level", "warning", "--http", self.http, *self.extra,
        ]  # fmt: skip
        with self.log.open("wb") as log:
            self.proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit)),
            )
        try:
            self._wait_ready()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert self.proc is not None
            if self.proc.poll() is not None:
                raise AssertionError(f"server exited early:\n{self.log_text()}")
            if get(self.port, "/ping", timeout=1.0).ok:
                return
            time.sleep(0.1)
        raise AssertionError(f"server did not become ready:\n{self.log_text()}")

    def __exit__(self, *exc_info) -> None:
        proc = self.proc
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)

    def log_text(self) -> str:
        return self.log.read_text(errors="replace")[-3000:] if self.log.exists() else ""

    def sockets(self) -> int:
        """Sockets the server process holds right now (listener, loop pipes and connections)."""
        assert self.proc is not None
        fd_dir = f"/proc/{self.proc.pid}/fd"
        count = 0
        for name in os.listdir(fd_dir):
            try:
                count += os.readlink(f"{fd_dir}/{name}").startswith("socket:")
            except OSError:
                pass  # the descriptor closed while we were looking
        return count

    def snapshot(self) -> dict:
        reply = get(self.port, "/_guard")
        assert reply.status == 200, (reply, self.log_text())
        return json.loads(reply.body)


@pytest.fixture
def stub_dir(tmp_path: Path) -> Path:
    (tmp_path / "stub_app.py").write_text(STUB_APP)
    return tmp_path


@pytest.fixture(autouse=True)
def _enough_descriptors_in_the_test_process():
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft != resource.RLIM_INFINITY and soft < FLOOD + 200:
        pytest.skip(f"the flood needs ~{FLOOD + 200} descriptors in the test process (soft={soft})")


STOCK = [
    pytest.param("httptools", (), id="stock-httptools"),
    pytest.param("h11", ("--loop", "asyncio"), id="stock-h11-asyncio-loop"),
]
GUARDS = [
    pytest.param(GUARDED, (), id="exported-auto"),
    pytest.param("stub_app:GuardH11", ("--loop", "asyncio", "--backlog", "64"), id="h11-asyncio"),
]


# ---- the finding ------------------------------------------------------------------------


@pytest.mark.parametrize("http, extra", STOCK)
def test_control_stock_uvicorn_loses_the_estop_to_idle_connections(stub_dir, http, extra):
    """The bug itself: the same flood, uvicorn's own protocol. The E-stop POST gets no answer."""
    with Server(stub_dir, http, extra=extra) as server:
        assert post(server.port).status == 202  # fine before the flood
        flood = flood_idle(server.port, FLOOD)
        try:
            time.sleep(0.3)
            assert server.sockets() >= NOFILE - 40  # every descriptor is gone
            during = post(server.port, timeout=2.5)
            assert not during.ok, "stock uvicorn answered the E-stop under the flood"
            assert during.status is None
        finally:
            for conn in flood:
                conn.close()
        # the sockets are released: the server itself was never broken, only starved
        assert wait_until(lambda: post(server.port).status == 202, 10)


@pytest.mark.parametrize("http, extra", GUARDS)
def test_the_guarded_protocol_keeps_the_estop_reachable_under_the_same_flood(stub_dir, http, extra):
    with Server(stub_dir, http, env=SMALL, extra=extra) as server:
        baseline = server.sockets()
        before = post(server.port)
        assert before.status == 202
        flood = flood_idle(server.port, FLOOD)
        try:
            time.sleep(0.3)
            # the table is bounded: nowhere near the descriptor limit, and it is the guard's doing
            assert server.sockets() <= baseline + 24 + 2
            for path in (ESTOP, "/api/v2/control/emergency-stop", ESTOP_BODY_PATH):
                during = post(server.port, path)
                assert during.status == 202, (path, during, server.log_text())
                assert during.elapsed < 2.0
        finally:
            for conn in flood:
                conn.close()
        snapshot = server.snapshot()
        assert snapshot["peak_connections"] <= 24
        assert snapshot["evicted"] + snapshot["deadline_closed"] >= FLOOD - 24 - 5
        assert snapshot["refused"] == 0


@pytest.mark.parametrize(
    "http, extra, survives",
    [
        pytest.param("httptools", (), False, id="stock"),
        pytest.param(GUARDED, (), True, id="guarded"),
    ],
)
def test_requests_that_announce_a_body_and_never_send_it_do_not_pin_the_estop_out(
    stub_dir, http, extra, survives
):
    """``POST /control/emergency`` + ``Content-Length`` and no body, from many sockets.

    The headers are complete, so a guard that protected a connection from the headers on would
    fill its table with these and refuse the real E-stop; the stock server loses it to
    descriptor exhaustion.
    """
    held_request = (
        b"POST /api/v2/control/emergency HTTP/1.1\r\nHost: t\r\n"
        b"Content-Type: application/json\r\nContent-Length: 64\r\n\r\n"
    )
    with Server(stub_dir, http, env=SMALL, extra=extra) as server:
        flood = flood_idle(server.port, 200, send=held_request)
        try:
            time.sleep(0.3)
            during = post(server.port, ESTOP, timeout=2.5)
        finally:
            for conn in flood:
                conn.close()
        assert (during.status == 202) is survives, (during, server.log_text())


# ---- reaping ----------------------------------------------------------------------------


def test_idle_connections_are_reaped_after_the_header_deadline(stub_dir):
    env = {**SMALL, "LAWN_BERRY_HEADER_DEADLINE_S": "1"}
    with Server(stub_dir, GUARDED, env=env) as server:
        baseline = server.sockets()
        started = time.monotonic()
        idle = flood_idle(server.port, 10, chunk=10, pause=0)
        try:
            closed = closure_times(idle, started, timeout=10)
            assert len(closed) == 10, (closed, server.log_text())
            # not before the deadline (nothing else could have closed them: 10 < 24) ...
            assert min(closed.values()) >= 0.9
            # ... and promptly after it
            assert max(closed.values()) < 6
            assert wait_until(lambda: server.sockets() <= baseline, 5)
        finally:
            for conn in idle:
                conn.close()
        snapshot = server.snapshot()
        assert snapshot["deadline_closed"] >= 10
        assert snapshot["evicted"] == 0  # the deadline did it, not the cap


def test_control_stock_uvicorn_never_reaps_a_connection_that_sends_nothing(stub_dir):
    with Server(stub_dir, "httptools", extra=("--timeout-keep-alive", "1")) as server:
        baseline = server.sockets()
        idle = flood_idle(server.port, 10, chunk=10, pause=0)
        try:
            time.sleep(2.5)  # well past any keep-alive timeout
            assert server.sockets() >= baseline + 10
            assert not any(closed_by_peer(conn) for conn in idle)
        finally:
            for conn in idle:
                conn.close()


def _dribble(port: int, seconds: float) -> float | None:
    """Send an unfinished header one byte at a time; seconds until the server cut us, or None."""
    conn = socket.create_connection(("127.0.0.1", port), timeout=3)
    started = time.monotonic()
    try:
        conn.sendall(b"GET /x HTTP/1.1\r\nHost: t\r\nX-Slow: ")
        while time.monotonic() - started < seconds:
            time.sleep(0.2)
            try:
                conn.sendall(b"a")
            except OSError:
                return time.monotonic() - started
            if closed_by_peer(conn):
                return time.monotonic() - started
        return None
    finally:
        conn.close()


def test_a_header_sent_one_byte_at_a_time_is_cut_at_the_deadline_not_extended_by_it(stub_dir):
    env = {**SMALL, "LAWN_BERRY_HEADER_DEADLINE_S": "1"}
    with Server(stub_dir, GUARDED, env=env) as server:
        cut_after = _dribble(server.port, 6.0)
        assert cut_after is not None, "a dribbling connection kept the server busy"
        assert 0.8 <= cut_after < 3.0
        assert server.snapshot()["deadline_closed"] >= 1


def test_control_stock_uvicorn_lets_a_dribbled_header_hold_its_descriptor(stub_dir):
    with Server(stub_dir, "httptools") as server:
        assert _dribble(server.port, 3.0) is None


def test_a_keep_alive_connection_that_made_a_request_is_not_cut_by_the_header_deadline(stub_dir):
    env = {**SMALL, "LAWN_BERRY_HEADER_DEADLINE_S": "1"}
    with Server(stub_dir, GUARDED, env=env, extra=("--timeout-keep-alive", "30")) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as conn:
            request = b"GET /x HTTP/1.1\r\nHost: t\r\n\r\n"
            conn.sendall(request)
            assert conn.recv(4096).startswith(b"HTTP/1.1 200")
            time.sleep(2.5)  # two and a half times the header deadline
            conn.sendall(request)
            assert conn.recv(4096).startswith(b"HTTP/1.1 200")
        assert server.snapshot()["deadline_closed"] == 0


# ---- the reserve ------------------------------------------------------------------------


def test_requests_above_the_soft_cap_get_503_while_the_estop_still_succeeds(stub_dir):
    with Server(stub_dir, GUARDED, env=SMALL, extra=("--timeout-keep-alive", "30")) as server:
        holds = []
        try:
            for _ in range(20):  # soft cap = 24 - 4 = 20 productive connections
                conn = socket.create_connection(("127.0.0.1", server.port), timeout=5)
                conn.sendall(b"GET /hold HTTP/1.1\r\nHost: t\r\n\r\n")
                assert conn.recv(4096).startswith(b"HTTP/1.1 200")  # in flight, never finishes
                holds.append(conn)

            shed = get(server.port, "/api/v2/status")
            assert shed.status == 503
            assert shed.headers["retry-after"] == "1"
            assert shed.headers["connection"] == "close"
            assert json.loads(shed.body)["detail"]
            for path in (ESTOP, "/api/v2/control/emergency-stop", ESTOP_BODY_PATH):
                assert post(server.port, path).status == 202, path
            # an authenticated route next to the E-stop is NOT exempt (exact paths only)
            assert post(server.port, "/api/v2/control/emergency_clear").status == 503

            # a flood on top does not dislodge the productive connections or the E-stop
            flood = flood_idle(server.port, 100)
            try:
                time.sleep(0.3)
                assert post(server.port).status == 202
                assert not any(closed_by_peer(conn) for conn in holds)
            finally:
                for conn in flood:
                    conn.close()
        finally:
            for conn in holds:
                conn.close()

        # the quota recovers once the productive connections let go
        assert wait_until(lambda: get(server.port, "/api/v2/status").status == 200, 5)
        snapshot = server.snapshot()
        assert snapshot["shed"] >= 2
        assert snapshot["refused"] == 0


def test_websocket_upgrades_leave_the_table_and_are_not_evicted(stub_dir):
    upgrade = (
        b"GET /ws HTTP/1.1\r\nHost: t\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    with Server(stub_dir, GUARDED, env=SMALL) as server:
        sockets = []
        try:
            for i in range(40):  # far more than the 24 the table holds
                conn = socket.create_connection(("127.0.0.1", server.port), timeout=5)
                conn.sendall(upgrade)
                assert conn.recv(4096).startswith(b"HTTP/1.1 101"), i
                sockets.append(conn)
            time.sleep(0.5)
            assert not any(closed_by_peer(conn) for conn in sockets)
            snapshot = server.snapshot()
            assert snapshot["evicted"] == 0 and snapshot["connections"] <= 2
        finally:
            for conn in sockets:
                conn.close()


# ---- configuration never stops the server ---------------------------------------------------


def test_a_bad_environment_value_is_logged_once_and_the_server_still_serves_the_estop(stub_dir):
    env = {
        "LAWN_BERRY_MAX_CONNECTIONS": "banana",
        "LAWN_BERRY_CONNECTION_RESERVE": "2",
        "LAWN_BERRY_HEADER_DEADLINE_S": "-3",
    }
    with Server(stub_dir, GUARDED, env=env) as server:
        assert post(server.port).status == 202
        snapshot = server.snapshot()
        # the defaults for a 192-descriptor limit
        assert (snapshot["max_connections"], snapshot["reserve"]) == (96, 24)
        assert snapshot["header_deadline_s"] == 10.0
        log = server.log_text()
    for name in env:
        assert log.count(name) == 1, (name, log)


# ---- SEC3-02: a request to an E-stop PATH is not an E-stop; "productive" ends with the response


SECOND_REQUEST_HEADERS_UNFINISHED = b"GET /x HTTP/1.1\r\nHost: t\r\nX-Hold: a"
SECOND_REQUEST_BODY_UNFINISHED = (
    b"POST /api/v1/x HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n"
    b"Content-Length: 1000\r\n\r\n{"
)
DEADLINES = {**SMALL, "LAWN_BERRY_BODY_DEADLINE_S": "2"}


def _hold_the_table(port: int, first: bytes, second: bytes, count: int) -> list[socket.socket]:
    """``count`` connections that each complete ``first`` (answered), then start ``second``
    and fall silent: the review's way of spending the reserve that keeps the E-stop reachable."""
    held = []
    for _ in range(count):
        conn = socket.create_connection(("127.0.0.1", port), timeout=5)
        conn.sendall(first)
        assert conn.recv(4096).startswith(b"HTTP/1.1 200")
        conn.sendall(second)
        held.append(conn)
    return held


@pytest.mark.parametrize(
    "second, label",
    [
        pytest.param(SECOND_REQUEST_HEADERS_UNFINISHED, "headers", id="second-headers-unfinished"),
        pytest.param(SECOND_REQUEST_BODY_UNFINISHED, "body", id="second-body-unfinished"),
    ],
)
def test_clients_that_got_one_answer_and_then_went_silent_do_not_hold_the_estop_out(
    stub_dir, second, label
):
    """Old: 24 'E-stop' GETs (a path-only classification) filled the table with connections that
    were productive for ever, and the real E-stop POST on a new connection was reset (the reviewer
    measured ReadError at 2 s, 12 s and 30 s). Now each connection falls back to evictable when its
    response is done and its second request is judged by its own deadline."""
    first = b"GET /api/v2/tractor/emergency-stop HTTP/1.1\r\nHost: t\r\n\r\n"
    with Server(stub_dir, GUARDED, env=DEADLINES, extra=("--timeout-keep-alive", "30")) as server:
        held = _hold_the_table(server.port, first, second, 24)
        started = time.monotonic()
        try:
            for path in (ESTOP, "/api/v2/control/emergency-stop", ESTOP_BODY_PATH):
                during = post(server.port, path, timeout=3.0)
                assert during.status == 202, (label, path, during, server.log_text())
            closed = closure_times(held, started, timeout=8)
            # every holder is gone: evicted by the newcomers or closed by its own deadline
            assert len(closed) == 24, (label, len(closed), server.log_text())
            assert post(server.port).status == 202
        finally:
            for conn in held:
                conn.close()
        snapshot = server.snapshot()
        assert snapshot["evicted"] + snapshot["deadline_closed"] >= 24
        assert snapshot["refused"] == 0


def test_clients_that_keep_asking_a_stop_path_with_get_do_not_hold_the_estop_out(stub_dir):
    """The reviewer's second shape: no unfinished request, just a complete GET to an E-stop path
    every second on every connection (64 x 405 in the live run). A GET stops nothing: it is an
    ordinary request and its connection is evictable between two of them."""
    get_stop_path = b"GET /api/v2/tractor/emergency-stop HTTP/1.1\r\nHost: t\r\n\r\n"
    with Server(stub_dir, GUARDED, env=DEADLINES, extra=("--timeout-keep-alive", "30")) as server:
        conns = []
        for _ in range(24):
            conn = socket.create_connection(("127.0.0.1", server.port), timeout=5)
            conn.sendall(get_stop_path)
            assert conn.recv(4096).startswith(b"HTTP/1.1 200")
            conns.append(conn)
        try:
            for round_number in range(4):
                for conn in conns:
                    try:
                        conn.sendall(get_stop_path)
                    except OSError:
                        pass  # the server evicted it: that is the point
                during = post(server.port, timeout=3.0)
                assert during.status == 202, (round_number, during, server.log_text())
                time.sleep(0.5)
        finally:
            for conn in conns:
                conn.close()
        assert server.snapshot()["refused"] == 0


def test_a_request_body_that_stops_arriving_is_cut_at_the_body_deadline(stub_dir):
    env = {**SMALL, "LAWN_BERRY_BODY_DEADLINE_S": "1"}
    with Server(stub_dir, GUARDED, env=env, extra=("--timeout-keep-alive", "30")) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as conn:
            started = time.monotonic()
            conn.sendall(SECOND_REQUEST_BODY_UNFINISHED)
            gone = closure_times([conn], started, timeout=6)
            assert 0 in gone and 0.8 <= gone[0] < 4.0, (gone, server.log_text())
        assert server.snapshot()["deadline_closed"] >= 1


def test_a_second_request_on_a_keep_alive_connection_gets_the_header_deadline_again(stub_dir):
    env = {**SMALL, "LAWN_BERRY_HEADER_DEADLINE_S": "1"}
    with Server(stub_dir, GUARDED, env=env, extra=("--timeout-keep-alive", "30")) as server:
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as conn:
            conn.sendall(b"GET /x HTTP/1.1\r\nHost: t\r\n\r\n")
            assert conn.recv(4096).startswith(b"HTTP/1.1 200")
            started = time.monotonic()
            conn.sendall(SECOND_REQUEST_HEADERS_UNFINISHED)
            gone = closure_times([conn], started, timeout=6)
            assert 0 in gone and 0.8 <= gone[0] < 4.0, (gone, server.log_text())
        assert server.snapshot()["deadline_closed"] >= 1
