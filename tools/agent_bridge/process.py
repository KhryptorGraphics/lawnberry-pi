"""Bounded subprocess supervision for agent CLIs and checks.

Every child runs in its own session/process group, so a timeout or a cancellation takes the
whole tree down.  If the bridge itself is killed the kernel kills the direct child
(``PDEATHSIG``); recovery terminates a leftover group only while its leader is provably the
recorded process.  Output is captured with hard caps, streamed line by line to an optional
callback (budget enforcement), and redacted before it is stored or logged.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

__all__ = [
    "ProcessOutcome",
    "ProcessSpec",
    "SubprocessRunner",
    "is_same_process",
    "kill_process_group",
    "proc_start_ticks",
]

_PR_SET_PDEATHSIG = 1
_PRCTL = None
if sys.platform.startswith("linux"):  # pragma: no branch - platform specific
    try:
        _PRCTL = ctypes.CDLL("libc.so.6", use_errno=True).prctl
    except (OSError, AttributeError):  # pragma: no cover - exotic libc
        _PRCTL = None


def _set_pdeathsig() -> None:  # pragma: no cover - runs in the forked child
    """Ask the kernel to SIGKILL this child if the bridge dies (Linux only)."""
    if _PRCTL is not None:
        _PRCTL(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


def proc_start_ticks(pid: int) -> int | None:
    """Start time of ``pid`` in clock ticks since boot (``/proc/<pid>/stat`` field 22)."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        rest = raw[raw.rindex(")") + 2 :].split()
        return int(rest[19])
    except (ValueError, IndexError):
        return None


def is_same_process(pid: int, start_ticks: int | None) -> bool:
    """True when ``pid`` is alive and still the process that was recorded."""
    if start_ticks is None:
        return False
    return proc_start_ticks(pid) == start_ticks


def kill_process_group(pgid: int, *, grace_seconds: float = 5.0) -> None:
    """SIGTERM then SIGKILL a whole process group; missing groups are ignored."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


@dataclass
class ProcessSpec:
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    stdin_text: str | None = None
    timeout_s: float = 600.0
    kill_grace_s: float = 10.0
    max_capture_bytes: int = 4_000_000
    max_line_bytes: int = 8_000_000
    stdout_log: Path | None = None
    stderr_log: Path | None = None


@dataclass
class ProcessOutcome:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    kill_reason: str | None = None
    duration_s: float = 0.0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    pid: int | None = None
    start_ticks: int | None = None
    spawn_error: str | None = None
    lines: int = 0
    extra: dict[str, object] = field(default_factory=dict)


_LOG_FULL_MARKER = "\n[agent-bridge: log truncated, capture limit reached]\n"


class _Sink:
    """Capture one stream with a byte cap, an optional log file and redaction."""

    _KEY_BEGIN = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
    _KEY_END = re.compile(r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----")

    def __init__(self, cap: int, log: Path | None, redact: Callable[[str], str]) -> None:
        self.cap = cap
        self.redact = redact
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False
        self.line_cut = False  # an over-long line lost its tail; later lines are still captured
        self.lines = 0
        self._log: IO[str] | None = None
        self._closed = False
        self._log_size = 0
        self._log_full = False
        self._in_key_block = False
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
            with contextlib.suppress(OSError):
                os.chmod(log, 0o600)
            self._log_size = os.fstat(self._log.fileno()).st_size

    def _redact_line(self, text: str) -> str:
        """Redact one line; a PEM private key spanning several plain-text lines goes whole.

        Structured (JSON) lines are left to the regex redactor, which stops at the closing quote.
        """
        if self._in_key_block:
            end = self._KEY_END.search(text)
            if end is None:
                return "[REDACTED:private-key]" + ("\n" if text.endswith("\n") else "")
            self._in_key_block = False
            return "[REDACTED:private-key]" + self.redact(text[end.end() :])
        clean = self.redact(text)
        if text.endswith("\n") and not clean.endswith("\n"):
            clean += "\n"
        begin = self._KEY_BEGIN.search(text)
        if begin is not None and not text.lstrip().startswith(("{", "[")):
            self._in_key_block = self._KEY_END.search(text, begin.end()) is None
        return clean

    def add(self, text: str) -> str:
        clean = self._redact_line(text)
        if self._closed:  # a reader that outlived the run: nothing is recorded any more
            return clean
        self.lines += 1
        encoded = len(clean.encode("utf-8", errors="replace"))
        if self._log is not None and not self._log_full:
            if self._log_size + encoded > self.cap:  # the disk follows the same cap as memory
                self._log_full = True
                self._log.write(_LOG_FULL_MARKER)
            else:
                self._log.write(clean)
                self._log_size += encoded
            self._log.flush()
        if not self.truncated:
            if self.size + encoded > self.cap:
                self.truncated = True
            else:
                self.parts.append(clean)
                self.size += encoded
        return clean

    def text(self) -> str:
        return "".join(self.parts)

    def close(self) -> None:
        self._closed = True
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()


class SubprocessRunner:
    """Run one command to completion under a deadline, with kill-on-abort semantics."""

    # How long the stream readers may keep draining once the leader is gone.  A descendant that
    # escaped the process group (its own setsid) can hold the pipes open indefinitely.
    drain_seconds = 3.0

    def __init__(
        self,
        redact: Callable[[str], str] | None = None,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._redact = redact or (lambda text: text)
        self._popen = popen

    def run(
        self,
        spec: ProcessSpec,
        *,
        on_line: Callable[[str], str | None] | None = None,
        abort_check: Callable[[], str | None] | None = None,
        abort_poll_s: float = 5.0,
        on_start: Callable[[int, int | None], None] | None = None,
    ) -> ProcessOutcome:
        started = time.monotonic()
        kwargs: dict[str, object] = {}
        if _PRCTL is not None:
            kwargs["preexec_fn"] = _set_pdeathsig
        # Open the log sinks before the child exists: a log path that cannot be opened must fail
        # the run cleanly instead of leaking a running process nobody reads.
        sinks: list[_Sink] = []
        try:
            out_sink = _Sink(spec.max_capture_bytes, spec.stdout_log, self._redact)
            sinks.append(out_sink)
            err_sink = _Sink(spec.max_capture_bytes, spec.stderr_log, self._redact)
            sinks.append(err_sink)
        except OSError as exc:
            for sink in sinks:
                sink.close()
            return ProcessOutcome(returncode=None, spawn_error=f"cannot open log file: {exc}")
        try:
            proc = self._popen(
                spec.argv,
                cwd=str(spec.cwd),
                env=spec.env,
                stdin=subprocess.PIPE if spec.stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
                **kwargs,
            )
        except (OSError, ValueError) as exc:
            out_sink.close()
            err_sink.close()
            return ProcessOutcome(returncode=None, spawn_error=f"{type(exc).__name__}: {exc}")

        pid = proc.pid
        ticks = proc_start_ticks(pid)
        if on_start is not None:
            with contextlib.suppress(Exception):
                on_start(pid, ticks)

        wake = threading.Event()
        state: dict[str, str | None] = {"reason": None}
        lock = threading.Lock()

        def request_kill(reason: str) -> None:
            with lock:
                if state["reason"] is None:
                    state["reason"] = reason
            wake.set()

        def pump(
            stream: IO[bytes], sink: _Sink, callback: Callable[[str], str | None] | None
        ) -> None:
            limit = spec.max_line_bytes
            try:
                while True:
                    chunk = stream.readline(limit + 1)
                    if not chunk:
                        break
                    if len(chunk) > limit and not chunk.endswith(b"\n"):
                        # Over-long line: keep the head, discard the rest of the line.
                        sink.line_cut = True
                        chunk = chunk[:limit] + b"\n"
                        while True:
                            rest = stream.readline(limit + 1)
                            if not rest or rest.endswith(b"\n"):
                                break
                    clean = sink.add(chunk.decode("utf-8", errors="replace"))
                    if callback is not None:
                        try:
                            reason = callback(clean)
                        except Exception as exc:  # a broken hook must not wedge the run
                            reason = f"hook-error: {type(exc).__name__}"
                        if reason:
                            request_kill(reason)
            except (OSError, ValueError):
                pass

        def feed() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write((spec.stdin_text or "").encode("utf-8", errors="replace"))
            except (OSError, ValueError):
                pass
            finally:
                with contextlib.suppress(OSError, ValueError):
                    proc.stdin.close()

        assert proc.stdout is not None and proc.stderr is not None
        threads = [
            threading.Thread(target=pump, args=(proc.stdout, out_sink, on_line), daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, err_sink, None), daemon=True),
        ]
        if spec.stdin_text is not None:
            threads.append(threading.Thread(target=feed, daemon=True))
        for thread in threads:
            thread.start()

        timed_out = False
        deadline = started + spec.timeout_s
        next_abort = started + abort_poll_s
        returncode: int | None = None
        while True:
            try:
                returncode = proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if wake.is_set() and state["reason"] is not None:
                returncode = self._terminate(proc, spec.kill_grace_s)
                break
            if now >= deadline:
                timed_out = True
                returncode = self._terminate(proc, spec.kill_grace_s)
                break
            if abort_check is not None and now >= next_abort:
                next_abort = now + abort_poll_s
                try:
                    reason = abort_check()
                except Exception:
                    reason = None
                if reason:
                    request_kill(reason)
        # The leader is gone; make sure no descendants survive in its group.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        # Give the readers a bounded window to drain what is left in the pipes.  Readers still
        # blocked after that are abandoned (they are daemons) and their streams are left open:
        # closing a buffered stream another thread is reading would block until it returns.
        drain_deadline = time.monotonic() + self.drain_seconds
        for thread in threads:
            thread.join(timeout=max(0.0, drain_deadline - time.monotonic()))
        if not any(thread.is_alive() for thread in threads):
            for stream in (proc.stdout, proc.stderr):
                with contextlib.suppress(Exception):
                    stream.close()
        out_sink.close()
        err_sink.close()
        return ProcessOutcome(
            returncode=returncode,
            stdout=out_sink.text(),
            stderr=err_sink.text(),
            timed_out=timed_out,
            kill_reason=state["reason"],
            duration_s=time.monotonic() - started,
            stdout_truncated=out_sink.truncated or out_sink.line_cut,
            stderr_truncated=err_sink.truncated or err_sink.line_cut,
            pid=pid,
            start_ticks=ticks,
            lines=out_sink.lines,
        )

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes], grace: float) -> int | None:
        pgid = proc.pid
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        try:
            return proc.wait(timeout=max(0.0, grace))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            try:
                return proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable (D state)
                return None
