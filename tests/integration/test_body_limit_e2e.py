"""SEC3-01 end to end: one anonymous request announcing a huge body, a real uvicorn, real memory.

The finding: ``await request.body()`` collected every chunk before the size check, on a path that
needs no credential. Two anonymous ``POST`` s of 200 MiB took the backend from 164 MiB to 614 MiB
and, inside the unit's ``MemoryMax=512M`` cgroup, to a SIGKILL (exit status -9) in 0.4 s, with the
E-stop endpoint unreachable until systemd restarted it (``RestartSec=10``).

Here the REAL middleware stack (``register_input_validation_middleware``) runs under the REAL
uvicorn CLI with the unit's ``--http`` protocol, and the server's own peak resident set
(``VmHWM`` in ``/proc``) is what is compared, with a control that is the same app WITHOUT the body
bound (the old behaviour): it must grow by about the size of the body, or the test proves nothing.
The bodies are 48 MiB, not 200: enough to be unmistakable, short enough for a loaded CI box.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc for the RSS"),
]

ROOT = Path(__file__).resolve().parents[2]
GUARDED = "backend.src.core.uvicorn_protocol:LawnBerryHttpProtocol"
MIB = 1024 * 1024
BODY = 48 * MIB
# What a bounded server may grow by: a few sockets, one 16 KiB body, interpreter noise.
BOUNDED_GROWTH = 12 * MIB
# What the unbounded control must grow by (it holds the whole body and its join: > 1x the body).
UNBOUNDED_GROWTH = 32 * MIB

STUB = textwrap.dedent(
    """
    import os
    from fastapi import FastAPI, Request
    from backend.src.middleware.input_validation import (
        InputValidationMiddleware,
        register_input_validation_middleware,
    )

    app = FastAPI()
    if os.environ.get("STUB_UNBOUNDED") == "1":
        # the OLD stack: validation only, nothing in front of it
        app.add_middleware(InputValidationMiddleware, max_body_bytes=1_000_000)
    else:
        register_input_validation_middleware(app)


    @app.post("/api/v2/echo")
    async def echo(payload: dict):
        return {"keys": len(payload)}


    @app.post("/api/v2/auth/login")
    async def login(payload: dict):
        return {"ok": True}


    @app.post("/api/v2/control/emergency")
    async def estop(request: Request):
        return {"stopped": True, "body_bytes": len(await request.body())}


    @app.websocket("/ws/accept")
    async def ws_accept(websocket):
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()


    @app.websocket("/ws/refuse")
    async def ws_refuse(websocket):
        await websocket.close(code=1008)


    @app.get("/ping")
    async def ping():
        return {"ok": True}
    """
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def status_value(pid: int, key: str) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise AssertionError(key)


class Server:
    def __init__(self, directory: Path, *, unbounded: bool = False, extra_env=None) -> None:
        self.directory = directory
        self.port = free_port()
        self.unbounded = unbounded
        self.extra_env = extra_env or {}
        self.proc: subprocess.Popen | None = None
        self.log = directory / ("unbounded.log" if unbounded else "bounded.log")

    def __enter__(self) -> Server:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("LAWN_BERRY_", "UVICORN_", "INPUT_", "ANONYMOUS_"))
            and k != "PYTEST_ADDOPTS"
        }
        env.update(
            OPERATOR_AUTH_REQUIRED="1",
            LAWN_BERRY_OPERATOR_CREDENTIAL=secrets.token_urlsafe(24),
            LAWN_BERRY_AUTH_SECRET=secrets.token_hex(32),
            STUB_UNBOUNDED="1" if self.unbounded else "0",
            PYTHONDONTWRITEBYTECODE="1",
            SIM_MODE="1",
        )
        env.update(self.extra_env)
        command = [
            sys.executable, "-m", "uvicorn", "stub_app:app", "--app-dir", str(self.directory),
            "--host", "127.0.0.1", "--port", str(self.port), "--lifespan", "off",
            "--no-access-log", "--log-level", "warning", "--http", GUARDED,
            "--ws-max-size", "65536", "--timeout-keep-alive", "5",
        ]  # fmt: skip
        with self.log.open("wb") as log:
            self.proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, self.log.read_text(errors="replace")[-3000:]
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    time.sleep(0.3)
                    return self
            except OSError:
                time.sleep(0.1)
        raise AssertionError("server did not start")

    def __exit__(self, *exc_info) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_text(self) -> str:
        return self.log.read_text(errors="replace")[-3000:]


def exchange(port: int, raw: bytes, timeout: float = 5.0) -> tuple[int | None, dict, bytes]:
    """Send ``raw``, read one response to EOF (the request says ``close``)."""
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
    except OSError:
        return None, {}, b""
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}
    return (int(lines[0].split(" ", 2)[1]) if lines[0] else None), headers, body


def estop(port: int) -> tuple[int | None, dict, bytes]:
    return exchange(
        port,
        b"POST /api/v2/control/emergency HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
    )


def blast(port: int, head: bytes, total: int, *, chunked: bool, pause: float = 0.0) -> int:
    """Send ``head`` then up to ``total`` body bytes as fast as the server takes them.

    Returns how many body bytes the server accepted before it stopped reading or cut us off.
    """
    piece = b"0" * MIB
    sent = 0
    conn = socket.create_connection(("127.0.0.1", port), timeout=10)
    conn.settimeout(10)
    try:
        conn.sendall(head)
        while sent < total:
            data = (b"%x\r\n" % len(piece) + piece + b"\r\n") if chunked else piece
            try:
                conn.sendall(data)
            except OSError:
                break
            sent += len(piece)
            if pause:
                time.sleep(pause)
    finally:
        conn.close()
    return sent


def post_head(path: str, *, length: int | None, auth: str | None = None) -> bytes:
    lines = [f"POST {path} HTTP/1.1", "Host: t", "Content-Type: application/json"]
    lines.append("Transfer-Encoding: chunked" if length is None else f"Content-Length: {length}")
    if auth:
        lines.append(f"Authorization: {auth}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


@pytest.fixture
def stub_dir(tmp_path: Path) -> Path:
    (tmp_path / "stub_app.py").write_text(STUB)
    return tmp_path


def growth_after(server: Server, action) -> int:
    before = status_value(server.pid, "VmHWM")
    action()
    time.sleep(0.5)  # let the server finish what it is doing with the connection
    return status_value(server.pid, "VmHWM") - before


SHAPES = [
    pytest.param(lambda: post_head("/api/v2/echo", length=BODY), False, id="content-length"),
    pytest.param(lambda: post_head("/api/v2/echo", length=None), True, id="chunked"),
    pytest.param(lambda: post_head("/api/v2/auth/login", length=1024), False, id="lying-length"),
]


def test_control_without_the_body_bound_the_old_stack_buffers_the_whole_body(stub_dir):
    """The premise: with only the validation middleware (the old stack) the body is held in full."""
    with Server(stub_dir, unbounded=True) as server:
        grew = growth_after(
            server,
            lambda: blast(server.port, post_head("/api/v2/echo", length=BODY), BODY, chunked=False),
        )
        assert grew >= UNBOUNDED_GROWTH, (grew / MIB, server.log_text())


@pytest.mark.parametrize("make_head, chunked", SHAPES)
def test_an_anonymous_body_of_48_mib_does_not_grow_the_server_and_the_estop_still_answers(
    stub_dir, make_head, chunked
):
    with Server(stub_dir) as server:
        grew = growth_after(server, lambda: blast(server.port, make_head(), BODY, chunked=chunked))
        assert server.alive(), server.log_text()
        assert grew < BOUNDED_GROWTH, f"+{grew / MIB:.1f} MiB: {server.log_text()}"
        status, _, body = estop(server.port)
        assert status == 200 and json.loads(body)["stopped"] is True


def test_the_413_reaches_a_client_that_announces_too_much_and_waits_for_the_answer(stub_dir):
    with Server(stub_dir) as server:
        status, headers, body = exchange(
            server.port,
            post_head("/api/v2/echo", length=10 * MIB).replace(
                b"\r\n\r\n", b"\r\nConnection: close\r\n\r\n"
            ),
        )
        assert status == 413
        assert headers["connection"] == "close"
        assert json.loads(body) == {"detail": "Payload too large"}
        # just over the anonymous limit (16 KiB) is refused, just under it is not
        over = post_head("/api/v2/echo", length=16 * 1024 + 1)
        assert (
            exchange(server.port, over.replace(b"\r\n\r\n", b"\r\nConnection: close\r\n\r\n"))[0]
            == 413
        )
        under = post_head("/api/v2/echo", length=2).replace(
            b"\r\n\r\n", b"\r\nConnection: close\r\n\r\n"
        )
        assert exchange(server.port, under + b"{}")[0] == 200


def test_an_estop_with_a_huge_body_is_answered_and_the_body_is_capped(stub_dir):
    """The stop route is never refused for its body and never handed more than 64 KiB of it."""
    with Server(stub_dir) as server:
        head = (
            b"POST /api/v2/control/emergency HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
            b"Content-Type: application/json\r\nContent-Length: 1000000000\r\n\r\n"
        )
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as conn:
            conn.settimeout(10)
            conn.sendall(head + b"x" * (256 * 1024))  # 256 KiB of a body that claims to be 1 GB
            data = b""
            while b"\r\n\r\n" not in data or b"body_bytes" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        status = int(data.split(b" ", 2)[1])
        assert status == 200, (data[:300], server.log_text())
        assert json.loads(data.partition(b"\r\n\r\n")[2])["body_bytes"] == 64 * 1024


@pytest.mark.parametrize("path", ["/ws/accept", "/ws/refuse"])
def test_a_websocket_upgrade_with_an_enormous_body_costs_no_memory(stub_dir, path):
    """ "Every WebSocket upgrade path": the upgrade is handed to the WebSocket protocol before any
    HTTP body is parsed, and a frame is bounded by ``--ws-max-size``: nothing grows."""
    head = (
        f"GET {path} HTTP/1.1\r\nHost: t\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n"
        "Content-Length: 200000000\r\n\r\n"
    ).encode()
    with Server(stub_dir) as server:
        grew = growth_after(server, lambda: blast(server.port, head, BODY, chunked=False))
        assert server.alive(), server.log_text()
        assert grew < BOUNDED_GROWTH, f"+{grew / MIB:.1f} MiB: {server.log_text()}"
        assert estop(server.port)[0] == 200
