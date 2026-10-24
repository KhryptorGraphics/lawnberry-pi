"""Tests for bounded subprocess supervision and the verification checks built on it.

``SubprocessRunner`` is exercised against real child processes (small ``python -c`` scripts):
stream capture and caps, prompts on stdin, line hooks, abort polling, deadlines and kill
escalation, process-group clean-up, private redacted logs and spawn failures.  The process
helpers (``proc_start_ticks``, ``is_same_process``, ``kill_process_group``), the Linux
parent-death signal and ``checks.run_checks`` are covered at the end.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tools.agent_bridge import process as process_module
from tools.agent_bridge.checks import run_checks
from tools.agent_bridge.config import CheckConfig
from tools.agent_bridge.errors import CheckAborted
from tools.agent_bridge.process import (
    ProcessOutcome,
    ProcessSpec,
    SubprocessRunner,
    is_same_process,
    kill_process_group,
    proc_start_ticks,
)
from tools.agent_bridge.redact import Redactor

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_TOKEN = "ghp_" + "a1B2" * 9  # shaped like a GitHub token, assembled so scanners ignore it
CHILD_ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONCOERCECLOCALE": "0"}
PID_THEN_SLEEP = (
    "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
)
SIGTERM, SIGKILL = signal.SIGTERM, signal.SIGKILL

requires_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")


# ------------------------------------------------------------------------------------ helpers
def py(script: str, *args: str) -> list[str]:
    """argv for a hermetic, unbuffered child interpreter running ``script``."""
    return [sys.executable, "-S", "-u", "-c", textwrap.dedent(script), *args]


def make_spec(argv: list[str], cwd: Path, **overrides: Any) -> ProcessSpec:
    """A spec with a safety-net timeout so a broken runner fails the test instead of hanging."""
    fields: dict[str, Any] = {
        "argv": argv,
        "cwd": cwd,
        "env": dict(CHILD_ENV),
        "timeout_s": 30.0,
        "kill_grace_s": 2.0,
    }
    fields.update(overrides)
    return ProcessSpec(**fields)


def pid_alive(pid: int) -> bool:
    """True while ``pid`` is a live process; a zombie awaiting its reaper counts as dead."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return Path("/proc/self").exists() is False
    except OSError:
        return True
    return stat_text.rsplit(")", 1)[-1].split()[0] != "Z"


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def wait_dead(pid: int, timeout: float = 10.0) -> bool:
    return wait_until(lambda: not pid_alive(pid), timeout)


def read_pid(path: Path, timeout: float = 15.0) -> int:
    """Wait for a helper process to publish its pid in ``path``."""

    def published() -> bool:
        try:
            return path.read_text().strip().split()[0].isdigit()
        except (OSError, IndexError):
            return False

    assert wait_until(published, timeout), f"{path} was never written"
    return int(path.read_text().split()[0])


class Reaper:
    """Kills helper processes a test registered, even when the test fails halfway."""

    def __init__(self) -> None:
        self.pids: list[int] = []
        self.pid_files: list[Path] = []

    def add_pid_file(self, path: Path) -> Path:
        self.pid_files.append(path)
        return path

    def cleanup(self) -> None:
        for path in self.pid_files:
            with contextlib.suppress(OSError, ValueError, IndexError):
                self.pids.append(int(path.read_text().split()[0]))
        for pid in self.pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, SIGKILL)


@pytest.fixture
def reaper() -> Iterator[Reaper]:
    instance = Reaper()
    try:
        yield instance
    finally:
        instance.cleanup()


@pytest.fixture(autouse=True)
def no_stray_threads() -> Iterator[None]:
    """Reader/feeder threads must not outlive a test (the reaper fixture is torn down first)."""
    before = set(threading.enumerate())
    yield
    extra: list[threading.Thread] = []
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        extra = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        if not extra:
            return
        time.sleep(0.05)
    pytest.fail(f"threads left behind: {extra}")


@pytest.fixture
def runner() -> SubprocessRunner:
    return SubprocessRunner()


# ---------------------------------------------------------------------------- capture basics
def test_captures_both_streams_and_the_exit_status(runner: SubprocessRunner, tmp_path: Path):
    outcome = runner.run(
        make_spec(
            py(
                """
                import sys, time
                print("out-1"); print("out-2")
                print("err-1", file=sys.stderr)
                time.sleep(0.3)
                sys.exit(3)
                """
            ),
            tmp_path,
        )
    )

    assert outcome.returncode == 3
    assert outcome.stdout == "out-1\nout-2\n"
    assert outcome.stderr == "err-1\n"
    assert (outcome.timed_out, outcome.kill_reason, outcome.spawn_error) == (False, None, None)
    assert (outcome.stdout_truncated, outcome.stderr_truncated) == (False, False)
    assert outcome.lines == 2  # stdout lines
    assert outcome.pid is not None and outcome.pid > 0
    assert 0.3 <= outcome.duration_s < 10


@pytest.mark.parametrize("sig", [SIGKILL, SIGTERM], ids=["sigkill", "sigterm"])
def test_a_child_killed_by_a_signal_reports_a_negative_return_code(
    runner: SubprocessRunner, tmp_path: Path, sig: signal.Signals
) -> None:
    outcome = runner.run(make_spec(py(f"import os; os.kill(os.getpid(), {int(sig)})"), tmp_path))

    assert outcome.returncode == -int(sig)
    assert (outcome.timed_out, outcome.kill_reason) == (False, None)


def test_argv_is_passed_verbatim_and_the_child_runs_in_the_requested_directory(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    workdir = tmp_path / "work dir"
    workdir.mkdir()
    odd = ["a b", "$(echo pwned)", ";", "*", '--flag="q"', "é"]

    outcome = runner.run(
        make_spec(
            py("import json, os, sys; print(json.dumps([os.getcwd(), *sys.argv[1:]]))", *odd),
            workdir,
        )
    )

    cwd, *argv = json.loads(outcome.stdout)
    assert Path(cwd).resolve() == workdir.resolve()
    assert argv == odd  # no shell ever sees the arguments


@linux_only
def test_child_leads_its_own_session_and_process_group(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    outcome = runner.run(
        make_spec(
            py("import os; print(os.getpid(), os.getpgid(0), os.getsid(0))"),
            tmp_path,
        )
    )

    pid, pgid, sid = (int(x) for x in outcome.stdout.split())
    assert pid == outcome.pid
    assert pgid == sid == pid  # killing the child's group can never hit the bridge's own group
    assert pgid != os.getpgrp()


def test_env_is_passed_exactly(
    runner: SubprocessRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIDGE_TEST_CANARY", "leak")
    env = {"ALPHA": "1", "BETA": "two words", "EQ": "a=b", "UNI": "é✓", "PYTHONCOERCECLOCALE": "0"}

    outcome = runner.run(
        make_spec(py("import json, os; print(json.dumps(dict(os.environ)))"), tmp_path, env=env)
    )

    assert json.loads(outcome.stdout) == env  # nothing inherited, nothing dropped


@requires_proc
def test_child_inherits_no_extra_file_descriptors(runner: SubprocessRunner, tmp_path: Path) -> None:
    leaked = open(tmp_path / "leak.txt", "w")  # noqa: SIM115 - the descriptor is the point
    os.set_inheritable(leaked.fileno(), True)
    try:
        outcome = runner.run(
            make_spec(
                py("import os; print(sorted(int(n) for n in os.listdir('/proc/self/fd')))"),
                tmp_path,
            )
        )
    finally:
        leaked.close()

    assert set(json.loads(outcome.stdout)) <= {0, 1, 2, 3}  # 3 is the directory listing itself


def test_without_a_prompt_stdin_is_devnull(runner: SubprocessRunner, tmp_path: Path) -> None:
    outcome = runner.run(
        make_spec(
            py(
                """
                import os, sys
                print(os.path.samestat(os.fstat(0), os.stat(os.devnull)), repr(sys.stdin.read()))
                """
            ),
            tmp_path,
        )
    )

    assert outcome.stdout == "True ''\n"


@requires_proc
def test_the_finished_child_is_reaped_and_cannot_be_mistaken_for_a_live_process(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    outcome = runner.run(make_spec(py("pass"), tmp_path))

    assert outcome.pid is not None and outcome.start_ticks is not None
    assert not Path(f"/proc/{outcome.pid}").exists()  # no zombie left behind
    assert is_same_process(outcome.pid, outcome.start_ticks) is False


# --------------------------------------------------------------------------------------- stdin
@pytest.mark.parametrize(
    "prompt", ["hello\nworld", "héllo ✓ 日本語\n", ""], ids=["ascii", "unicode", "empty"]
)
def test_prompt_is_delivered_on_stdin_as_utf8_and_stdin_is_closed(
    runner: SubprocessRunner, tmp_path: Path, prompt: str
) -> None:
    outcome = runner.run(
        make_spec(
            py("import sys; print(repr(sys.stdin.buffer.read().decode('utf-8')))"),
            tmp_path,
            stdin_text=prompt,
        )
    )

    assert outcome.stdout == repr(prompt) + "\n"
    assert outcome.returncode == 0


def test_a_five_megabyte_prompt_reaches_the_child_intact(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    prompt = ("0123456789abcdef" * 63 + "0123456789abcde\n") * 5120  # exactly 5 MiB
    outcome = runner.run(
        make_spec(
            py(
                """
                import hashlib, sys
                data = sys.stdin.buffer.read()
                print(len(data), hashlib.sha256(data).hexdigest())
                """
            ),
            tmp_path,
            stdin_text=prompt,
        )
    )

    expected = hashlib.sha256(prompt.encode()).hexdigest()
    assert len(prompt) == 5 * 1024 * 1024
    assert (outcome.returncode, outcome.timed_out) == (0, False)
    assert outcome.stdout == f"{len(prompt)} {expected}\n"


def test_a_large_prompt_does_not_deadlock_with_a_child_that_talks_first(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    """The child fills both output pipes before reading; stdin must be fed concurrently."""
    prompt = ("y" * 1023 + "\n") * 5120
    outcome = runner.run(
        make_spec(
            py(
                """
                import sys
                line = "x" * 1023
                for _ in range(3000):
                    print(line)
                    print(line, file=sys.stderr)
                print("received", len(sys.stdin.buffer.read()))
                """
            ),
            tmp_path,
            stdin_text=prompt,
            max_capture_bytes=16_000_000,
            timeout_s=60,
        )
    )

    assert (outcome.returncode, outcome.timed_out) == (0, False)
    assert outcome.stdout.endswith(f"received {len(prompt)}\n")
    assert len(outcome.stderr) == 3000 * 1024


def test_a_child_that_never_reads_its_prompt_neither_hangs_nor_breaks_the_runner(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    outcome = runner.run(
        make_spec(py("print('done')"), tmp_path, stdin_text="z" * 5_000_000, timeout_s=20)
    )

    assert (outcome.returncode, outcome.stdout, outcome.timed_out) == (0, "done\n", False)


def test_a_prompt_that_cannot_be_encoded_still_reaches_eof_instead_of_wedging_the_child(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    """Lone surrogates are not valid UTF-8; the child must still see end-of-input."""
    outcome = runner.run(
        make_spec(
            py("import sys; print(len(sys.stdin.buffer.read()))"),
            tmp_path,
            stdin_text="before \ud800 after",
            timeout_s=5,
            kill_grace_s=1,
        )
    )

    assert outcome.timed_out is False
    assert outcome.returncode == 0
    assert outcome.stdout.strip().isdigit()


# -------------------------------------------------------------------------------------- on_line
def test_on_line_gets_redacted_stdout_lines_in_order_and_never_stderr(tmp_path: Path) -> None:
    runner = SubprocessRunner(redact=Redactor().redact)
    seen: list[str] = []

    outcome = runner.run(
        make_spec(
            py(
                f"""
                import sys
                print("first")
                print("token {FAKE_TOKEN} end")
                print("only on stderr", file=sys.stderr)
                print("last")
                """
            ),
            tmp_path,
        ),
        on_line=lambda line: seen.append(line),
    )

    assert seen == ["first\n", "token [REDACTED:github-token] end\n", "last\n"]
    assert outcome.stdout == "".join(seen)
    assert outcome.stderr == "only on stderr\n"
    assert FAKE_TOKEN not in outcome.stdout + outcome.stderr


def test_every_line_arrives_once_and_in_order(runner: SubprocessRunner, tmp_path: Path) -> None:
    seen: list[str] = []

    runner.run(
        make_spec(py("for i in range(2000): print(i, 'p' * (i % 97))"), tmp_path),
        on_line=lambda line: seen.append(line),
    )

    assert [line.split()[0] for line in seen] == [str(i) for i in range(2000)]


def test_a_line_hook_reason_kills_the_child_and_is_reported(
    runner: SubprocessRunner, tmp_path: Path, reaper: Reaper
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "child.pid")

    outcome = runner.run(
        make_spec(
            py(
                """
                import os, sys, time
                open(sys.argv[1], "w").write(str(os.getpid()))
                print("one"); print("budget exceeded"); print("never seen?", flush=True)
                time.sleep(60)
                """,
                str(pid_file),
            ),
            tmp_path,
        ),
        on_line=lambda line: "over budget" if line.startswith("budget") else None,
    )

    assert outcome.kill_reason == "over budget"
    assert outcome.timed_out is False
    assert outcome.returncode == -SIGTERM
    assert outcome.stdout.startswith("one\nbudget exceeded\n")
    assert outcome.duration_s < 20
    assert wait_dead(read_pid(pid_file))


def test_the_first_kill_reason_wins(runner: SubprocessRunner, tmp_path: Path) -> None:
    reasons = iter(["first", "second", "third"])

    outcome = runner.run(
        make_spec(py("import time\nfor i in range(300): print(i); time.sleep(0.01)"), tmp_path),
        on_line=lambda line: next(reasons, "later"),
    )

    assert outcome.kill_reason == "first"


@pytest.mark.parametrize("empty", [None, ""], ids=["none", "empty-string"])
def test_a_falsy_hook_result_lets_the_child_finish(
    runner: SubprocessRunner, tmp_path: Path, empty: str | None
) -> None:
    outcome = runner.run(
        make_spec(py("print('a'); print('b')"), tmp_path), on_line=lambda _l: empty
    )

    assert (outcome.returncode, outcome.kill_reason, outcome.stdout) == (0, None, "a\nb\n")


def test_a_raising_line_hook_kills_the_child_with_a_hook_error(
    runner: SubprocessRunner, tmp_path: Path, reaper: Reaper
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "child.pid")
    seen: list[str] = []

    def hook(line: str) -> None:
        seen.append(line)
        if line == "boom\n":
            raise RuntimeError("hook exploded")

    outcome = runner.run(
        make_spec(
            py(
                """
                import os, sys, time
                open(sys.argv[1], "w").write(str(os.getpid()))
                print("fine"); print("boom"); time.sleep(60)
                """,
                str(pid_file),
            ),
            tmp_path,
        ),
        on_line=hook,
    )

    assert outcome.kill_reason is not None and outcome.kill_reason.startswith("hook-error")
    assert "RuntimeError" in outcome.kill_reason
    assert seen[:2] == ["fine\n", "boom\n"]
    assert outcome.stdout == "fine\nboom\n"
    assert wait_dead(read_pid(pid_file))


def test_the_log_already_holds_a_line_when_the_hook_sees_it(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    log = tmp_path / "out.log"
    snapshots: list[tuple[str, str]] = []

    runner.run(
        make_spec(py("print('one'); print('two')"), tmp_path, stdout_log=log),
        on_line=lambda line: snapshots.append((line, log.read_text())),
    )

    assert [line for line, _ in snapshots] == ["one\n", "two\n"]
    assert [text for _, text in snapshots] == ["one\n", "one\ntwo\n"]


# ------------------------------------------------------------------------------- abort_check
def test_abort_check_is_polled_at_the_configured_interval_and_its_reason_kills(
    runner: SubprocessRunner, tmp_path: Path, reaper: Reaper
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "child.pid")
    polls: list[float] = []

    def abort() -> str | None:
        polls.append(time.monotonic())
        return "stop now" if len(polls) >= 3 else None

    begun = time.monotonic()
    outcome = runner.run(
        make_spec(
            py(
                PID_THEN_SLEEP,
                str(pid_file),
            ),
            tmp_path,
        ),
        abort_check=abort,
        abort_poll_s=0.2,
    )

    assert outcome.kill_reason == "stop now"
    assert (outcome.timed_out, outcome.returncode) == (False, -SIGTERM)
    assert len(polls) == 3
    assert polls[0] - begun >= 0.2 - 0.005  # not polled before the first interval
    assert all(b - a >= 0.2 - 0.005 for a, b in zip(polls, polls[1:], strict=False))
    assert outcome.duration_s < 5  # honours abort_poll_s, not the 5 s default
    assert wait_dead(read_pid(pid_file))


def test_a_failing_abort_check_is_ignored_and_polling_continues(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    calls: list[int] = []

    def abort() -> str | None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("github is down")
        return "now" if len(calls) == 3 else ""

    outcome = runner.run(
        make_spec(py("import time; time.sleep(60)"), tmp_path),
        abort_check=abort,
        abort_poll_s=0.1,
    )

    assert outcome.kill_reason == "now"
    assert len(calls) == 3


def test_abort_check_is_not_consulted_when_the_child_finishes_first(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    calls: list[int] = []

    outcome = runner.run(
        make_spec(py("print('quick')"), tmp_path),
        abort_check=lambda: calls.append(1),
    )

    assert calls == [] and outcome.returncode == 0  # default poll interval is seconds long


def test_an_abort_reason_escalates_to_sigkill_for_a_child_that_ignores_sigterm(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    ready = tmp_path / "ready"
    script = """
        import signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open(sys.argv[1], "w").write("x")
        time.sleep(60)
    """

    outcome = runner.run(
        make_spec(py(script, str(ready)), tmp_path, kill_grace_s=0.5),
        abort_check=lambda: "cancel" if ready.exists() else None,
        abort_poll_s=0.1,
    )

    assert outcome.kill_reason == "cancel"
    assert outcome.returncode == -SIGKILL
    assert outcome.duration_s >= 0.5


# ---------------------------------------------------------------------------- hard deadlines
def test_the_hard_timeout_terminates_the_child(
    runner: SubprocessRunner, tmp_path: Path, reaper: Reaper
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "child.pid")

    outcome = runner.run(
        make_spec(
            py(
                """
                import os, sys, time
                open(sys.argv[1], "w").write(str(os.getpid()))
                print("started")
                time.sleep(60)
                """,
                str(pid_file),
            ),
            tmp_path,
            timeout_s=2.0,
        )
    )

    assert outcome.timed_out is True and outcome.kill_reason is None
    assert outcome.returncode == -SIGTERM
    assert outcome.stdout == "started\n"  # what was produced before the deadline is kept
    assert 2.0 <= outcome.duration_s < 10
    assert wait_dead(read_pid(pid_file))


IGNORE_SIGTERM = """
    import signal, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready")
    time.sleep(60)
"""


def test_a_sigterm_ignoring_child_is_sigkilled_after_the_grace_period_on_timeout(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    outcome = runner.run(make_spec(py(IGNORE_SIGTERM), tmp_path, timeout_s=2.0, kill_grace_s=0.5))

    assert outcome.stdout == "ready\n"  # the handler was installed before the deadline
    assert outcome.timed_out is True
    assert outcome.returncode == -SIGKILL
    assert outcome.duration_s >= 2.45  # deadline plus the grace period


def test_a_sigterm_ignoring_child_is_sigkilled_after_the_grace_period_on_a_hook_kill(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    outcome = runner.run(
        make_spec(py(IGNORE_SIGTERM), tmp_path, kill_grace_s=0.5),
        on_line=lambda line: "stop" if line == "ready\n" else None,
    )

    assert outcome.kill_reason == "stop" and outcome.timed_out is False
    assert outcome.returncode == -SIGKILL
    assert outcome.duration_s >= 0.5


# ------------------------------------------------------------------------ descendants / tree
GRANDCHILD_SCRIPT = """
    import os, subprocess, sys, time
    mode, pid_file = sys.argv[1], sys.argv[2]
    sink = subprocess.DEVNULL if mode == "devnull" else None  # None: inherit our pipes
    grandchild = subprocess.Popen(
        [sys.executable, "-S", "-c", "import time; time.sleep(120)"], stdout=sink, stderr=sink
    )
    open(pid_file, "w").write(str(grandchild.pid))
    print("spawned", flush=True)
    if sys.argv[3] == "linger":
        time.sleep(60)
"""


@pytest.mark.parametrize("pipes", ["inherit", "devnull"])
@pytest.mark.parametrize("ending", ["exit", "timeout", "hook"])
def test_descendants_in_the_process_group_never_survive_the_run(
    runner: SubprocessRunner, tmp_path: Path, reaper: Reaper, ending: str, pipes: str
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "grandchild.pid")
    options: dict[str, Any] = {"timeout_s": 2.5} if ending == "timeout" else {}
    on_line = (lambda line: "stop" if line == "spawned\n" else None) if ending == "hook" else None
    argv = py(GRANDCHILD_SCRIPT, pipes, str(pid_file), "exit" if ending == "exit" else "linger")

    outcome = runner.run(make_spec(argv, tmp_path, **options), on_line=on_line)

    assert outcome.stdout == "spawned\n"
    assert outcome.timed_out is (ending == "timeout")
    assert (outcome.kill_reason == "stop") is (ending == "hook")
    assert outcome.duration_s < 15  # an inherited pipe must not keep the run waiting
    assert wait_dead(read_pid(pid_file))


ESCAPED_SCRIPT = """
    import os, subprocess, sys, time
    ready, pid_file = sys.argv[1], sys.argv[2]
    daemon = (
        "import os, sys, time\\nos.setsid()\\nopen(sys.argv[1], 'w').write('x')\\ntime.sleep(40)"
    )
    grandchild = subprocess.Popen([sys.executable, "-S", "-c", daemon, ready])
    open(pid_file, "w").write(str(grandchild.pid))
    while not os.path.exists(ready):  # wait until the daemon has left our process group
        time.sleep(0.01)
    print("leader done")
"""


def test_a_daemonised_descendant_holding_the_pipes_cannot_stall_the_run(
    tmp_path: Path, reaper: Reaper
) -> None:
    runner = SubprocessRunner()
    pid_file = reaper.add_pid_file(tmp_path / "daemon.pid")

    outcome = runner.run(
        make_spec(
            py(ESCAPED_SCRIPT, str(tmp_path / "ready"), str(pid_file)), tmp_path, timeout_s=30
        )
    )

    daemon = read_pid(pid_file)
    assert pid_alive(daemon)  # it really escaped the group kill and still owns the pipes
    assert (outcome.returncode, outcome.timed_out) == (0, False)
    assert outcome.stdout == "leader done\n"
    assert outcome.duration_s < SubprocessRunner.drain_seconds + 8  # nowhere near its 40 s


def test_the_drain_window_is_configurable(tmp_path: Path, reaper: Reaper) -> None:
    runner = SubprocessRunner()
    runner.drain_seconds = 0.3
    pid_file = reaper.add_pid_file(tmp_path / "daemon.pid")

    outcome = runner.run(
        make_spec(
            py(ESCAPED_SCRIPT, str(tmp_path / "ready"), str(pid_file)), tmp_path, timeout_s=30
        )
    )

    assert outcome.stdout == "leader done\n"
    assert outcome.duration_s < 5


# ----------------------------------------------------------------------------- output caps
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_capture_cap_truncates_to_whole_lines_but_keeps_draining(
    runner: SubprocessRunner, tmp_path: Path, stream: str
) -> None:
    other = "stderr" if stream == "stdout" else "stdout"
    script = f"""
        import sys
        line = "x" * 1023
        for _ in range(2048):  # 2 MiB: far beyond any pipe buffer
            print(line, file=sys.{stream})
        print("tail", file=sys.{other})
    """

    outcome = runner.run(make_spec(py(script), tmp_path, max_capture_bytes=4096))

    assert (outcome.returncode, outcome.timed_out) == (0, False)  # the child was never blocked
    assert getattr(outcome, stream) == ("x" * 1023 + "\n") * 4  # whole lines up to the cap
    assert getattr(outcome, f"{stream}_truncated") is True
    assert getattr(outcome, other) == "tail\n"
    assert getattr(outcome, f"{other}_truncated") is False
    assert outcome.lines == (2048 if stream == "stdout" else 1)  # dropped lines still count


@pytest.mark.parametrize(("cap_delta", "truncated"), [(0, False), (-1, True), (1, False)])
def test_capture_cap_boundary(
    runner: SubprocessRunner, tmp_path: Path, cap_delta: int, truncated: bool
) -> None:
    script = "print('a' * 99); print('b' * 99)"  # 200 bytes in two lines

    outcome = runner.run(make_spec(py(script), tmp_path, max_capture_bytes=200 + cap_delta))

    assert outcome.stdout_truncated is truncated
    expected = "a" * 99 + "\n" + ("" if truncated else "b" * 99 + "\n")
    assert outcome.stdout == expected


def test_once_truncated_later_small_lines_are_not_captured_either(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    script = "print('a' * 999); print('b' * 999); print('tiny')"

    outcome = runner.run(make_spec(py(script), tmp_path, max_capture_bytes=1500))

    assert outcome.stdout == "a" * 999 + "\n"  # a prefix of the stream, never a patchwork
    assert outcome.stdout_truncated is True


@pytest.mark.parametrize("length", [99, 100, 101, 102, 450])
def test_an_over_long_line_is_cut_safely_and_the_next_line_stays_intact(
    runner: SubprocessRunner, tmp_path: Path, length: int
) -> None:
    seen: list[str] = []

    outcome = runner.run(
        make_spec(py(f"print('L' * {length}); print('next')"), tmp_path, max_line_bytes=100),
        on_line=lambda line: seen.append(line),
    )

    kept = min(length, 100)
    assert outcome.stdout == "L" * kept + "\nnext\n"
    assert outcome.stdout_truncated is (length > 100)
    assert seen == ["L" * kept + "\n", "next\n"]  # hooks never receive an over-long chunk
    assert outcome.lines == 2


def test_an_over_long_unterminated_last_line_is_cut_and_a_short_one_is_kept(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    long_tail = runner.run(
        make_spec(py("import sys; sys.stdout.write('y' * 450)"), tmp_path, max_line_bytes=100)
    )
    short_tail = runner.run(
        make_spec(py("import sys; sys.stdout.write('abc')"), tmp_path, max_line_bytes=100)
    )

    assert long_tail.stdout == "y" * 100 + "\n" and long_tail.stdout_truncated is True
    assert short_tail.stdout == "abc" and short_tail.stdout_truncated is False


def test_over_long_lines_are_also_cut_on_stderr(runner: SubprocessRunner, tmp_path: Path) -> None:
    script = "import sys; print('E' * 5000, file=sys.stderr); print('ok', file=sys.stderr)"

    outcome = runner.run(make_spec(py(script), tmp_path, max_line_bytes=100))

    assert outcome.stderr == "E" * 100 + "\nok\n" and outcome.stderr_truncated is True


def test_a_cut_in_the_middle_of_a_multibyte_character_is_decoded_safely(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    outcome = runner.run(make_spec(py("print('é' * 5)"), tmp_path, max_line_bytes=5))

    assert outcome.stdout == "éé\ufffd\n"  # five bytes: two whole characters and half of a third
    assert outcome.stdout_truncated is True


def test_invalid_utf8_output_is_replaced_not_fatal(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    script = "import sys; sys.stdout.buffer.write(b'ok \\xff\\xfe end\\n')"

    outcome = runner.run(make_spec(py(script), tmp_path))

    assert outcome.stdout == "ok \ufffd\ufffd end\n" and outcome.returncode == 0


# --------------------------------------------------------------------------------- log files
def test_logs_are_private_redacted_and_complete(tmp_path: Path) -> None:
    runner = SubprocessRunner(redact=Redactor().redact)
    out_log, err_log = tmp_path / "logs" / "deep" / "out.log", tmp_path / "logs" / "err.log"
    script = f"""
        import sys
        print("out token {FAKE_TOKEN}")
        print("err token {FAKE_TOKEN}", file=sys.stderr)
        print("API_KEY=supersecretvalue123")
    """
    previous = os.umask(0)  # a permissive umask must not make the logs readable
    try:
        outcome = runner.run(
            make_spec(py(script), tmp_path, stdout_log=out_log, stderr_log=err_log)
        )
    finally:
        os.umask(previous)

    assert stat.S_IMODE(out_log.stat().st_mode) == 0o600
    assert stat.S_IMODE(err_log.stat().st_mode) == 0o600
    assert out_log.read_text() == outcome.stdout
    assert err_log.read_text() == outcome.stderr
    for text in (out_log.read_text(), err_log.read_text(), outcome.stdout, outcome.stderr):
        assert FAKE_TOKEN not in text and "supersecretvalue123" not in text
    assert "[REDACTED:github-token]" in out_log.read_text()
    assert "[REDACTED:github-token]" in err_log.read_text()


def test_logs_are_appended_across_runs_and_survive_a_killed_child(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    log = tmp_path / "agent.log"
    runner.run(make_spec(py("print('attempt 1')"), tmp_path, stdout_log=log))

    outcome = runner.run(
        make_spec(
            py("import time; print('attempt 2', flush=True); time.sleep(60)"),
            tmp_path,
            timeout_s=2,
            stdout_log=log,
        )
    )

    assert outcome.timed_out is True
    assert log.read_text() == "attempt 1\nattempt 2\n"


def test_on_disk_logs_follow_the_capture_cap(runner: SubprocessRunner, tmp_path: Path) -> None:
    log = tmp_path / "out.log"
    script = "line = 'x' * 1023\nfor _ in range(2048): print(line)"

    outcome = runner.run(make_spec(py(script), tmp_path, max_capture_bytes=4096, stdout_log=log))

    text = log.read_text()
    assert outcome.returncode == 0 and outcome.stdout_truncated is True
    assert text.startswith(("x" * 1023 + "\n") * 4)
    assert text.count("log truncated") == 1  # one marker, then the log stops growing
    assert len(text.encode()) < 4096 + 200


@pytest.mark.parametrize("broken", ["stdout_log", "stderr_log"])
def test_an_unusable_log_path_fails_the_run_without_starting_the_child(
    runner: SubprocessRunner, tmp_path: Path, broken: str
) -> None:
    (tmp_path / "afile").write_text("not a directory")
    marker = tmp_path / "child-ran"
    logs = {"stdout_log": tmp_path / "ok-out.log", "stderr_log": tmp_path / "ok-err.log"}
    logs[broken] = tmp_path / "afile" / "x.log"
    starts: list[int] = []

    outcome = runner.run(
        make_spec(
            py("import sys, time; open(sys.argv[1], 'w').write('x'); time.sleep(30)", str(marker)),
            tmp_path,
            **logs,
        ),
        on_start=lambda pid, _ticks: starts.append(pid),
    )

    assert outcome.spawn_error is not None and "log" in outcome.spawn_error
    assert (outcome.returncode, outcome.pid, outcome.start_ticks) == (None, None, None)
    assert starts == []
    time.sleep(0.3)
    assert not marker.exists()  # no process was left running unobserved


# ---------------------------------------------------------------------------- spawn failures
@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing-binary", "FileNotFoundError"),
        ("not-executable", "PermissionError"),
        ("missing-cwd", "FileNotFoundError"),
        ("nul-byte", "ValueError"),
    ],
)
def test_spawn_failures_are_reported_not_raised(
    runner: SubprocessRunner, tmp_path: Path, case: str, error: str
) -> None:
    script = tmp_path / "tool.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    cwd, argv = tmp_path, ["/nonexistent/binary", "--flag"]
    if case == "not-executable":
        argv = [str(script)]
    elif case == "missing-cwd":
        cwd, argv = tmp_path / "gone", [sys.executable, "-c", "pass"]
    elif case == "nul-byte":
        argv = [sys.executable, "-c", "pass\x00"]
    log = tmp_path / "logs" / "out.log"
    starts: list[int] = []

    outcome = runner.run(
        make_spec(argv, cwd, stdout_log=log), on_start=lambda pid, _t: starts.append(pid)
    )

    assert outcome.spawn_error is not None and outcome.spawn_error.startswith(error)
    assert (outcome.returncode, outcome.pid, outcome.start_ticks) == (None, None, None)
    assert (outcome.timed_out, outcome.kill_reason, outcome.stdout, outcome.stderr) == (
        False,
        None,
        "",
        "",
    )
    assert starts == []


# --------------------------------------------------------------------------------- on_start
@requires_proc
def test_on_start_reports_pid_and_start_ticks_before_any_output(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    events: list[tuple[Any, ...]] = []

    def on_start(pid: int, ticks: int | None) -> None:
        events.append(("start", pid, ticks, is_same_process(pid, ticks)))

    outcome = runner.run(
        make_spec(py("import sys; sys.stdin.read(); print('hello')"), tmp_path, stdin_text="go"),
        on_line=lambda line: events.append(("line", line)),
        on_start=on_start,
    )

    assert events[0][0] == "start" and events[1:] == [("line", "hello\n")]
    _, pid, ticks, alive_then = events[0]
    assert (pid, ticks) == (outcome.pid, outcome.start_ticks)
    assert isinstance(ticks, int) and alive_then is True  # the recorded identity matched live


def test_a_raising_on_start_does_not_break_the_run(
    runner: SubprocessRunner, tmp_path: Path
) -> None:
    def broken(_pid: int, _ticks: int | None) -> None:
        raise RuntimeError("ledger is locked")

    outcome = runner.run(make_spec(py("print('still ran')"), tmp_path), on_start=broken)

    assert (outcome.returncode, outcome.stdout) == (0, "still ran\n")


# ------------------------------------------------------------------------- process helpers
def raw_start_ticks(pid: int) -> int:
    """Independent parse of /proc/<pid>/stat: the name field may contain spaces and ')'."""
    after_name = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1]
    return int(after_name.split()[19])


@requires_proc
def test_proc_start_ticks_matches_the_kernel_value_for_this_process() -> None:
    ticks = proc_start_ticks(os.getpid())

    assert isinstance(ticks, int) and ticks > 0
    assert ticks == raw_start_ticks(os.getpid()) == proc_start_ticks(os.getpid())


@requires_proc
def test_proc_start_ticks_survives_a_hostile_process_name(tmp_path: Path) -> None:
    script = (
        "import ctypes, time\n"
        "ctypes.CDLL(None).prctl(15, b') S 1 2 (x y', 0, 0, 0)\n"  # PR_SET_NAME
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-S", "-c", script], stdout=subprocess.PIPE, text=True, env=CHILD_ENV
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == "ready\n"
        assert Path(f"/proc/{child.pid}/comm").read_text().strip() == ") S 1 2 (x y"

        assert proc_start_ticks(child.pid) == raw_start_ticks(child.pid)
    finally:
        child.kill()
        child.wait()


@requires_proc
def test_is_same_process_distinguishes_live_wrong_and_dead_processes() -> None:
    child = subprocess.Popen(
        [sys.executable, "-S", "-c", "import time; time.sleep(60)"], env=CHILD_ENV
    )
    try:
        ticks = proc_start_ticks(child.pid)
        assert ticks is not None
        assert is_same_process(child.pid, ticks) is True
        assert is_same_process(child.pid, ticks + 1) is False  # a recycled pid has other ticks
        assert is_same_process(child.pid, None) is False
    finally:
        child.kill()
        child.wait()

    assert proc_start_ticks(child.pid) is None
    assert is_same_process(child.pid, ticks) is False
    assert proc_start_ticks(4_194_305 + 11) is None  # beyond pid_max: never exists


def _reap_in_background(*procs: subprocess.Popen[bytes]) -> None:
    for proc in procs:
        threading.Thread(target=proc.wait, daemon=True).start()


def spawn_group(ignore_term: bool) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]:
    """A group leader and a sibling in that group, both direct children of the test."""
    leader_script = textwrap.dedent(
        f"""
        import signal, time
        if {ignore_term}:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(120)
        """
    )
    leader = subprocess.Popen(
        [sys.executable, "-S", "-c", leader_script],
        stdout=subprocess.PIPE,
        process_group=0,
        env=CHILD_ENV,
    )
    assert leader.stdout is not None and leader.stdout.readline() == b"ready\n"
    sibling = subprocess.Popen(
        [sys.executable, "-S", "-c", "import time; time.sleep(120)"],
        process_group=leader.pid,
        env=CHILD_ENV,
    )
    assert os.getpgid(sibling.pid) == leader.pid
    _reap_in_background(leader, sibling)
    return leader, sibling


@linux_only
def test_kill_process_group_terminates_every_member_without_waiting_out_the_grace() -> None:
    leader, sibling = spawn_group(ignore_term=False)
    try:
        begun = time.monotonic()
        kill_process_group(leader.pid, grace_seconds=10.0)
        elapsed = time.monotonic() - begun

        assert leader.wait(timeout=5) == -SIGTERM and sibling.wait(timeout=5) == -SIGTERM
        assert elapsed < 5  # it returned as soon as the group was gone
        with pytest.raises(ProcessLookupError):
            os.killpg(leader.pid, 0)
    finally:
        for proc in (leader, sibling):
            proc.kill()


@linux_only
def test_kill_process_group_escalates_to_sigkill_for_members_that_ignore_sigterm() -> None:
    leader, sibling = spawn_group(ignore_term=True)
    try:
        begun = time.monotonic()
        kill_process_group(leader.pid, grace_seconds=0.5)
        elapsed = time.monotonic() - begun

        assert sibling.wait(timeout=5) == -SIGTERM  # everyone got the polite signal first
        assert leader.wait(timeout=5) == -SIGKILL  # the stubborn member was killed after the grace
        assert 0.5 <= elapsed < 5
    finally:
        for proc in (leader, sibling):
            proc.kill()


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_kill_process_group_ignores_missing_and_foreign_groups(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    calls: list[tuple[int, int]] = []

    def unavailable(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        raise error

    monkeypatch.setattr(os, "killpg", unavailable)
    begun = time.monotonic()

    kill_process_group(4242, grace_seconds=5.0)

    assert time.monotonic() - begun < 1
    assert calls == [(4242, SIGTERM), (4242, 0)]  # asked politely, saw nobody, never escalated


def test_kill_process_group_sends_sigkill_once_when_the_group_outlives_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    kill_process_group(4242, grace_seconds=0.2)

    assert calls[0] == (4242, SIGTERM)
    assert calls[-1] == (4242, SIGKILL)
    assert [sig for _, sig in calls].count(SIGKILL) == 1
    assert {sig for _, sig in calls[1:-1]} == {0}  # only liveness probes in between


@linux_only
def test_kill_process_group_on_a_nonexistent_group_returns_at_once() -> None:
    begun = time.monotonic()

    kill_process_group(4_194_305 + 7, grace_seconds=5.0)  # above Linux's maximum pid

    assert time.monotonic() - begun < 1


# ------------------------------------------------------------------- parent-death (Linux)
PDEATH_HARNESS = """
    import sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    from tools.agent_bridge.process import ProcessSpec, SubprocessRunner

    pid_file = Path(sys.argv[2])
    spec = ProcessSpec(
        argv=[sys.executable, "-S", "-c", "import time; time.sleep(120)"],
        cwd=Path("."),
        env={},
        timeout_s=300,
    )
    SubprocessRunner().run(spec, on_start=lambda pid, ticks: pid_file.write_text(f"{pid} {ticks}"))
"""


@linux_only
def test_the_child_dies_with_the_bridge_even_when_the_bridge_is_killed(
    tmp_path: Path, reaper: Reaper
) -> None:
    if getattr(process_module, "_PRCTL", None) is None:
        pytest.skip("libc prctl is unavailable")
    pid_file = reaper.add_pid_file(tmp_path / "child.pid")
    harness = subprocess.Popen(
        [
            sys.executable,
            "-S",
            "-c",
            textwrap.dedent(PDEATH_HARNESS),
            str(REPO_ROOT),
            str(pid_file),
        ],
        env=CHILD_ENV,
    )
    try:
        child = read_pid(pid_file)
        assert pid_alive(child)

        os.kill(harness.pid, SIGKILL)  # the bridge crashes without any chance to clean up
        harness.wait(timeout=10)

        assert wait_dead(child, timeout=10), "the agent outlived a crashed bridge"
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait()


# --------------------------------------------------------------------------------- run_checks
def check(script: str, *args: str, **kwargs: Any) -> CheckConfig:
    return CheckConfig(argv=[sys.executable, "-S", "-c", textwrap.dedent(script), *args], **kwargs)


@pytest.fixture
def check_env(tmp_path: Path) -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home")}


def go(
    names: list[str],
    checks: dict[str, CheckConfig],
    tmp_path: Path,
    environ: dict[str, str],
    *,
    runner: SubprocessRunner | None = None,
    redactor: Redactor | None = None,
    **kwargs: Any,
):
    return run_checks(
        names,
        checks,
        tmp_path,
        runner=runner or SubprocessRunner(redact=(redactor or Redactor()).redact),
        redactor=redactor or Redactor(),
        log_dir=tmp_path / "logs",
        environ=environ,
        **kwargs,
    )


def test_a_passing_check_reports_success_and_keeps_private_logs(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    checks = {"lint": check("import os; print('all good'); print('cwd', os.getcwd())")}

    (result,) = go(["lint"], checks, tmp_path, check_env)

    assert result.name == "lint" and result.passed is True
    assert (result.returncode, result.timed_out) == (0, False)
    assert result.argv == checks["lint"].argv
    assert result.duration_s >= 0
    assert result.output_tail.startswith("all good\ncwd ")
    assert Path(result.output_tail.splitlines()[1].split(" ", 1)[1]).resolve() == tmp_path.resolve()
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()  # fit for result_json
    out_log, err_log = (
        tmp_path / "logs" / "check-lint.stdout.log",
        tmp_path / "logs" / "check-lint.stderr.log",
    )
    assert (
        out_log.read_text() == "all good\ncwd " + str(tmp_path) + "\n"
        or "all good" in out_log.read_text()
    )
    assert stat.S_IMODE(out_log.stat().st_mode) == 0o600
    assert stat.S_IMODE(err_log.stat().st_mode) == 0o600


def test_a_failing_check_reports_its_status_and_both_streams(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    checks = {
        "unit": check(
            "import sys; print('stdout text'); print('stderr text', file=sys.stderr); sys.exit(2)"
        )
    }

    (result,) = go(["unit"], checks, tmp_path, check_env)

    assert result.passed is False and result.returncode == 2 and result.timed_out is False
    assert result.output_tail.index("stdout text") < result.output_tail.index("stderr text")


def test_a_check_that_exceeds_its_timeout_is_killed_and_fails(
    tmp_path: Path, check_env: dict[str, str], reaper: Reaper
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "check.pid")
    script = (
        "import os, sys, time\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "print('before', flush=True)\n"
        "time.sleep(60)\n"
    )
    checks = {"slow": check(script, str(pid_file), timeout_seconds=1.5)}

    (result,) = go(["slow"], checks, tmp_path, check_env)

    assert result.passed is False and result.timed_out is True
    assert result.returncode == -SIGTERM
    assert "before" in result.output_tail
    assert result.duration_s < 15
    assert wait_dead(read_pid(pid_file))


def test_checks_run_in_the_requested_order_and_stop_at_the_first_failure(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    def marker(name: str, code: int) -> CheckConfig:
        return check(f"import sys; open({name!r} + '.marker', 'w').write('x'); sys.exit({code})")

    checks = {"c": marker("c", 0), "a": marker("a", 0), "b": marker("b", 1), "z": marker("z", 0)}

    results = go(["a", "b", "c", "z"], checks, tmp_path, check_env)

    assert [(r.name, r.passed) for r in results] == [("a", True), ("b", False)]
    assert (tmp_path / "a.marker").exists() and (tmp_path / "b.marker").exists()
    assert not (tmp_path / "c.marker").exists() and not (tmp_path / "z.marker").exists()


def test_an_unconfigured_check_name_cannot_run_anything(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    checks = {"known": check("open('known.marker', 'w').write('x')")}

    with pytest.raises(KeyError):
        go(["rm -rf /"], checks, tmp_path, check_env)

    assert not (tmp_path / "known.marker").exists()


def test_a_check_that_cannot_be_started_fails_with_the_reason(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    checks = {
        "missing": CheckConfig(argv=["/nonexistent/tool", "--x"]),
        "after": check("open('after.marker', 'w').write('x')"),
    }

    results = go(["missing", "after"], checks, tmp_path, check_env)

    assert len(results) == 1
    (result,) = results
    assert result.passed is False and result.returncode is None and result.timed_out is False
    assert result.output_tail.startswith("FileNotFoundError")
    assert not (tmp_path / "after.marker").exists()


ENV_DUMP = "import json, os; open('env.json', 'w').write(json.dumps(dict(os.environ)))"


def test_checks_see_only_the_allow_listed_environment_plus_their_own(tmp_path: Path) -> None:
    environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/agent",
        "USER": "agent",
        "LOGNAME": "agent",
        "LANG": "en_US.UTF-8",
        "TMPDIR": "/tmp/agent",
        "LC_ALL": "C",
        "GITHUB_TOKEN": FAKE_TOKEN,
        "LAWNBERRY_BRIDGE_GITHUB_TOKEN": FAKE_TOKEN,
        "ANTHROPIC_API_KEY": "sk-ant-" + "x" * 30,
        "SSH_AUTH_SOCK": "/run/agent.sock",
        "PYTHONPATH": "/evil",
    }
    cfg_env = {"CHECK_EXTRA": "from config", "LANG": "C.UTF-8", "PYTHONCOERCECLOCALE": "0"}
    checks = {"env": check(ENV_DUMP, env=cfg_env)}

    (result,) = go(["env"], checks, tmp_path, environ)

    assert result.passed, result.output_tail
    assert json.loads((tmp_path / "env.json").read_text()) == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/agent",
        "USER": "agent",
        "LOGNAME": "agent",
        "LANG": "C.UTF-8",  # the check's own setting wins over the inherited value
        "TMPDIR": "/tmp/agent",
        "CHECK_EXTRA": "from config",
        "PYTHONCOERCECLOCALE": "0",
    }


def test_allow_listed_variables_that_are_absent_are_not_invented(tmp_path: Path) -> None:
    checks = {"env": check(ENV_DUMP, env={"PYTHONCOERCECLOCALE": "0"})}

    (result,) = go(["env"], checks, tmp_path, {"PATH": os.environ.get("PATH", "")})

    assert result.passed
    assert set(json.loads((tmp_path / "env.json").read_text())) == {"PATH", "PYTHONCOERCECLOCALE"}


def test_the_process_environment_is_used_when_none_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIDGE_TEST_CANARY", "leak")
    monkeypatch.setenv("LANG", "C.UTF-8")
    checks = {"env": check(ENV_DUMP, env={"PYTHONCOERCECLOCALE": "0"})}

    result = run_checks(
        ["env"],
        checks,
        tmp_path,
        runner=SubprocessRunner(),
        redactor=Redactor(),
        log_dir=tmp_path / "logs",
    )[0]

    seen = json.loads((tmp_path / "env.json").read_text())
    allowed = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "TMPDIR")
    assert result.passed
    assert seen == {
        **{k: os.environ[k] for k in allowed if k in os.environ},
        "PYTHONCOERCECLOCALE": "0",
    }
    assert "BRIDGE_TEST_CANARY" not in seen and seen["LANG"] == "C.UTF-8"


def test_command_prefix_wraps_every_check_but_is_not_part_of_the_reported_argv(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import json, sys\nopen('invocations.jsonl', 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    checks = {
        "one": CheckConfig(argv=["--flag", "value with space"]),
        "two": CheckConfig(argv=["second"]),
    }

    results = go(
        ["one", "two"],
        checks,
        tmp_path,
        check_env,
        command_prefix=[sys.executable, "-S", str(wrapper)],
    )

    assert [r.passed for r in results] == [True, True]
    lines = (tmp_path / "invocations.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [["--flag", "value with space"], ["second"]]
    assert [r.argv for r in results] == [["--flag", "value with space"], ["second"]]


def test_the_output_tail_is_redacted_even_when_the_runner_is_not(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    script = (
        f"import sys; print('token {FAKE_TOKEN}')\n"
        "print('API_KEY=supersecretvalue123', file=sys.stderr)\n"
        "print('phrase correct horse battery staple')\n"
        "sys.exit(1)\n"
    )
    checks = {"leaky": check(script)}

    (result,) = go(
        ["leaky"],
        checks,
        tmp_path,
        check_env,
        runner=SubprocessRunner(),  # no runner-level redaction: the tail has its own
        redactor=Redactor(literals=["correct horse battery staple"]),
    )

    for secret in (FAKE_TOKEN, "supersecretvalue123", "correct horse battery staple"):
        assert secret not in result.output_tail
    assert "[REDACTED:github-token]" in result.output_tail
    assert "[REDACTED:literal]" in result.output_tail


def test_check_logs_are_redacted_when_the_runner_redacts(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    checks = {"leaky": check(f"print('token {FAKE_TOKEN}')")}

    go(["leaky"], checks, tmp_path, check_env)

    log = (tmp_path / "logs" / "check-leaky.stdout.log").read_text()
    assert FAKE_TOKEN not in log and "[REDACTED:github-token]" in log


def test_a_short_output_is_reported_whole(tmp_path: Path, check_env: dict[str, str]) -> None:
    checks = {"short": check("print('  padded line  '); print('second')")}

    (result,) = go(["short"], checks, tmp_path, check_env)

    assert result.output_tail == "padded line  \nsecond"  # surrounding whitespace is trimmed


def test_a_long_output_keeps_its_end_where_the_failure_is_reported(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    script = """
        import sys
        for i in range(400):
            print(f"line {i:03d} " + "x" * 50)
        print("FINAL-FAILURE-MARKER: 1 failed")
        sys.exit(1)
    """
    checks = {"tests": check(script)}

    (result,) = go(["tests"], checks, tmp_path, check_env)

    tail = result.output_tail
    full = (
        "\n".join(f"line {i:03d} " + "x" * 50 for i in range(400))
        + "\nFINAL-FAILURE-MARKER: 1 failed"
    )
    assert tail.endswith("FINAL-FAILURE-MARKER: 1 failed")
    assert "line 000" not in tail  # the head was dropped, and the output says so
    assert "omitted" in tail.splitlines()[0]
    assert len(tail) < 1700
    body = tail.split("\n", 1)[1]
    assert full.endswith(body)  # what is kept is a genuine suffix of the output


def test_a_secret_cut_by_the_length_limit_leaves_no_fragment_behind(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    script = f"print('a' * 9000 + ' {FAKE_TOKEN} ' + 'b' * 1478)"
    checks = {"cut": check(script)}

    (result,) = go(["cut"], checks, tmp_path, check_env)

    fragments = [FAKE_TOKEN[i : i + 12] for i in range(len(FAKE_TOKEN) - 11)]
    assert not any(fragment in result.output_tail for fragment in fragments)


def test_abort_check_kills_a_running_check_and_raises_check_aborted(
    tmp_path: Path, check_env: dict[str, str], reaper: Reaper
) -> None:
    pid_file = reaper.add_pid_file(tmp_path / "slow.pid")
    checks = {
        "slow": check(
            "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)",
            str(pid_file),
        ),
        "after": check("open('after.marker', 'w').write('x')"),
    }
    polls: list[float] = []

    def abort() -> str | None:
        polls.append(time.monotonic())
        return "cancel: the issue was closed" if len(polls) >= 2 else None

    with pytest.raises(CheckAborted) as caught:
        go(["slow", "after"], checks, tmp_path, check_env, abort_check=abort, abort_poll_s=0.2)

    assert caught.value.reason == "cancel: the issue was closed"
    assert len(polls) == 2 and polls[1] - polls[0] >= 0.2 - 0.005
    assert wait_dead(read_pid(pid_file))
    assert not (tmp_path / "after.marker").exists()  # later checks never start


def test_a_quiet_abort_check_lets_checks_complete(
    tmp_path: Path, check_env: dict[str, str]
) -> None:
    checks = {"steady": check("import time; time.sleep(0.7); print('fine')")}
    polls: list[int] = []

    (result,) = go(
        ["steady"],
        checks,
        tmp_path,
        check_env,
        abort_check=lambda: polls.append(1),
        abort_poll_s=0.1,
    )

    assert result.passed and result.output_tail == "fine"
    assert len(polls) >= 3  # polled while the check was running (used for the lease heartbeat)


def test_outcome_type_exposes_the_documented_fields() -> None:
    outcome = ProcessOutcome(returncode=None, spawn_error="x")

    assert (outcome.stdout, outcome.stderr, outcome.lines) == ("", "", 0)
    assert (outcome.timed_out, outcome.kill_reason, outcome.pid) == (False, None, None)
