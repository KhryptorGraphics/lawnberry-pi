"""Executable emulator of the OhMyPi ``omp`` CLI (18.12.6) for hermetic agent-bridge tests.

Standalone (stdlib only, no repo imports): ``python fake_omp.py [omp flags] [prompt ...]``.
Flag acceptance, error texts, exit codes and the ``--mode json`` NDJSON event stream follow real
captures (``tests/agent_bridge/fixtures/omp_real_*.ndjson`` plus probes of the installed binary).

Environment
    FAKE_AGENT_SCENARIO  JSON file ``{"steps": [...], "log_env": [...], "ignore_sigterm": false}``;
                         unset means a single ``final`` step with the text ``ok``.
    FAKE_AGENT_LOG       when set, one JSON line is appended per invocation before anything else
                         happens: ``argv`` (without argv[0]), ``cwd`` (effective, after
                         ``--cwd``), ``env_keys`` (sorted names only), ``env`` (values of the
                         scenario's ``log_env`` names), ``prompt``, ``stdin_was_tty`` and
                         ``parsed`` (``{"flags": {...}, "positional": [...]}``).

Prompt: the positional arguments joined by a space (``@file`` arguments are read, like the real
CLI) or, with no positional argument, stdin read to EOF when stdin is not a TTY.

Scenario steps run in order.  Unknown ops and missing keys are harness errors (exit 99, nothing
executed).  Steps that only touch the filesystem emit no output.
    write_file{path,content,abs?}  append_file{path,content,abs?}  delete_file{path,abs?}
    symlink{path,target,abs?}  chmod_exec{path,abs?}   relative paths must stay inside the cwd
        unless ``abs`` is true (simulates a misbehaving agent)
    tool_call{tool,args,blocked?,result?,intent?,partial?}  tool_execution events (isError when
        blocked)
    cost{usd}  a new assistant message whose usage.cost.total is ``usd``, flushed immediately
    sleep{seconds}  hang  (``--max-time N`` ends the run with an ``aborted`` message and exit 1)
    spawn_child{pid_file}  ``sleep 300`` in the same process group; its pid goes to pid_file
    echo_prompt_to{path}  notice{message,level?,source?}  garbage_stdout{text}
    final{text,report?}  the final answer; ``report`` appends a ```json agent-result`` block
    fail{kind,message?}  kind: auth | credit | rate_limit | api_error | crash (traceback, exit 3)
    exit{code?,stdout?,stderr?}  abrupt exit without any result
A run that ends without ``final``/``fail``/``exit`` finishes with the text ``ok``.

In ``--mode json`` the ``session`` header and the user message are emitted lazily, just before
the first step that produces events, so ``exit``, ``garbage_stdout`` and ``fail crash`` leave
stdout free of JSON events.  Leading ``notice`` steps are emitted right after ``session``.
"""

from __future__ import annotations

import errno
import json
import os
import re
import signal
import sys
import time
from typing import Any

PROG = "fake_omp"
VERSION = "18.12.6"
DEFAULT_MODEL = "claude-opus-5-5"
SCENARIO_ENV = "FAKE_AGENT_SCENARIO"
LOG_ENV = "FAKE_AGENT_LOG"
HARNESS_EXIT = 99  # scenario or harness misuse
CRASH_EXIT = 3
STARTED = time.monotonic()
ERROR_ID = 135168  # opaque id carried by real stopReason "error" messages

# >>> shared plumbing (kept identical in fake_omp.py and fake_claude.py: standalone scripts)

FAIL_KINDS = ("auth", "credit", "rate_limit", "api_error", "crash")
_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "write_file": ("path",),
    "append_file": ("path",),
    "delete_file": ("path",),
    "symlink": ("path", "target"),
    "chmod_exec": ("path",),
    "tool_call": ("tool",),
    "cost": ("usd",),
    "sleep": ("seconds",),
    "hang": (),
    "spawn_child": ("pid_file",),
    "echo_prompt_to": ("path",),
    "final": (),
    "fail": ("kind",),
    "exit": (),
    "garbage_stdout": ("text",),
    "notice": ("message",),
}


class HarnessError(Exception):
    """Scenario or harness misuse: reported on stderr, exit status 99."""


def now_ms() -> int:
    return int(time.time() * 1000)


def iso_now() -> str:
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    return f"{stamp}.{int(now * 1000) % 1000:03d}Z"


def new_uuid() -> str:
    raw = bytearray(os.urandom(16))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    h = raw.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def write_out(text: str) -> None:
    """Write to stdout and flush; a vanished reader ends the run quietly."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        os._exit(1)


def write_err(text: str) -> None:
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except BrokenPipeError:
        os._exit(1)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def normalize_scenario(data: Any) -> dict[str, Any]:
    """Validate a decoded scenario and fill defaults; raises ValueError on a bad scenario."""
    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError('"steps" must be a list')
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} is not an object")
        op = step.get("op")
        if op not in _REQUIRED_KEYS:
            raise ValueError(f"step {index}: unknown op {op!r}")
        for key in _REQUIRED_KEYS[op]:
            if key not in step:
                raise ValueError(f"step {index} ({op}): missing {key!r}")
        if op == "cost" and not _is_number(step["usd"]):
            raise ValueError(f"step {index} (cost): usd must be a number")
        if op == "sleep" and not _is_number(step["seconds"]):
            raise ValueError(f"step {index} (sleep): seconds must be a number")
        if op == "fail" and step["kind"] not in FAIL_KINDS:
            raise ValueError(f"step {index} (fail): kind must be one of {FAIL_KINDS}")
        if op == "exit" and not isinstance(step.get("code", 1), int):
            raise ValueError(f"step {index} (exit): code must be an integer")
        if op == "tool_call" and not isinstance(step.get("args", {}), dict):
            raise ValueError(f"step {index} (tool_call): args must be an object")
        if op == "final" and not isinstance(step.get("report", {}), dict):
            raise ValueError(f"step {index} (final): report must be an object")
    log_env = data.get("log_env", [])
    if not isinstance(log_env, list):
        raise ValueError('"log_env" must be a list of names')
    return {
        "steps": steps,
        "log_env": [str(name) for name in log_env],
        "ignore_sigterm": bool(data.get("ignore_sigterm", False)),
    }


def load_scenario() -> tuple[dict[str, Any], str | None]:
    """Load the scenario from ``FAKE_AGENT_SCENARIO``; returns (scenario, error text or None)."""
    path = os.environ.get(SCENARIO_ENV)
    if not path:
        return normalize_scenario({"steps": [{"op": "final", "text": "ok"}]}), None
    try:
        with open(path, encoding="utf-8") as handle:
            return normalize_scenario(json.load(handle)), None
    except (OSError, ValueError) as exc:
        empty = {"steps": [], "log_env": [], "ignore_sigterm": False}
        return empty, f"cannot load scenario {path!r}: {exc}"


def stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except ValueError:
        return False


def read_stdin_prompt() -> str:
    """Read stdin to EOF as the prompt (empty when stdin is closed or unreadable)."""
    try:
        if sys.stdin is None:
            return ""
        return sys.stdin.buffer.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""


def log_invocation(
    argv: list[str],
    cwd: str,
    prompt: str,
    scenario: dict[str, Any],
    stdin_was_tty: bool,
    parsed: dict[str, Any],
) -> None:
    path = os.environ.get(LOG_ENV)
    if not path:
        return
    entry = {
        "argv": list(argv),
        "cwd": cwd,
        "env_keys": sorted(os.environ),
        "env": {name: os.environ[name] for name in scenario["log_env"] if name in os.environ},
        "prompt": prompt,
        "stdin_was_tty": stdin_was_tty,
        "parsed": parsed,
    }
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as exc:
        write_err(f"{PROG}: cannot write invocation log {path!r}: {exc}\n")


def resolve_path(raw: Any, allow_abs: bool) -> str:
    """Absolute path for a scenario path; escaping the cwd needs ``"abs": true``."""
    text = str(raw)
    if os.path.isabs(text):
        if not allow_abs:
            raise HarnessError(f'absolute path {text!r} needs "abs": true')
        return text
    norm = os.path.normpath(text)
    if not allow_abs and (norm == ".." or norm.startswith(".." + os.sep)):
        raise HarnessError(f'path {text!r} leaves the working directory; needs "abs": true')
    return os.path.join(os.getcwd(), norm)


def write_text_file(path: str, content: str, *, append: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def atomic_write(path: str, content: str) -> None:
    """Write via a temp file so a polling reader never sees a half-written file."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp{os.getpid()}"
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temp, path)


def spawn_child(pid_file: str) -> None:
    """Start ``sleep 300`` in this process group, detached from stdio; record its pid."""
    import shutil
    import subprocess

    sleeper = shutil.which("sleep")
    command = (
        [sleeper, "300"] if sleeper else [sys.executable, "-c", "import time; time.sleep(300)"]
    )
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atomic_write(pid_file, f"{child.pid}\n")


def run_shared_op(step: dict[str, Any], prompt: str) -> bool:
    """Run the filesystem/process ops common to both emulators; False for any other op."""
    op = step["op"]
    allow_abs = bool(step.get("abs", False))
    if op == "write_file":
        write_text_file(resolve_path(step["path"], allow_abs), str(step.get("content", "")))
    elif op == "append_file":
        path = resolve_path(step["path"], allow_abs)
        write_text_file(path, str(step.get("content", "")), append=True)
    elif op == "delete_file":
        path = resolve_path(step["path"], allow_abs)
        if os.path.isdir(path) and not os.path.islink(path):
            raise HarnessError(f"delete_file: {step['path']!r} is a directory")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    elif op == "symlink":
        link = resolve_path(step["path"], allow_abs)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(str(step["target"]), link)
    elif op == "chmod_exec":
        path = resolve_path(step["path"], allow_abs)
        os.chmod(path, os.stat(path).st_mode | 0o111)
    elif op == "spawn_child":
        spawn_child(str(step["pid_file"]))
    elif op == "echo_prompt_to":
        write_text_file(os.path.abspath(str(step["path"])), prompt)
    elif op == "garbage_stdout":
        text = str(step["text"])
        write_out(text if text.endswith("\n") else text + "\n")
    else:
        return False
    return True


def do_exit(step: dict[str, Any]) -> int:
    """The ``exit`` op: print the raw stdout/stderr texts, then return the exit status."""
    if step.get("stdout"):
        write_out(str(step["stdout"]))
    if step.get("stderr"):
        write_err(str(step["stderr"]))
    return int(step.get("code", 1))


def do_crash(message: str) -> int:
    """Print a genuine Python traceback to stderr; the caller exits 3 without any JSON."""
    import traceback

    try:
        raise RuntimeError(message)
    except RuntimeError:
        write_err(traceback.format_exc())
    return CRASH_EXIT


def compose_final_text(step: dict[str, Any]) -> str:
    """Final text, followed by the fenced ``agent-result`` block when a report is given."""
    text = str(step.get("text", ""))
    report = step.get("report")
    if report is None:
        return text
    block = "```json agent-result\n" + json.dumps(report) + "\n```"
    head = text.rstrip("\n")
    return f"{head}\n\n{block}" if head else block


# <<< end shared plumbing

# ---------------------------------------------------------------------------------------------
# omp command line
# ---------------------------------------------------------------------------------------------

_HELP_HINT = "Run `omp --help` for available flags."
_BOOL_FLAGS = frozenset(
    {
        "--prewalk",
        "--no-prewalk",
        "--plan-yolo",
        "--allow-home",
        "--print",
        "--continue",
        "--from-claude",
        "--from-codex",
        "--no-session",
        "--no-tools",
        "--no-lsp",
        "--no-pty",
        "--hide-thinking",
        "--advisor",
        "--external-thinking",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-title",
        "--no-ui",
        "--print-thoughts",
        "--auto-approve",
    }
)
_VALUE_FLAGS = frozenset(
    {
        "--model",
        "--smol",
        "--slow",
        "--plan",
        "--goal",
        "--prewalk-into",
        "--plan-yolo-into",
        "--provider",
        "--api-key",
        "--system-prompt",
        "--system-prompt-template",
        "--append-system-prompt",
        "--profile",
        "--alias",
        "--cwd",
        "--mode",
        "--config",
        "--add-dir",
        "--restore",
        "--session-dir",
        "--models",
        "--tools",
        "--thinking",
        "--service-tier",
        "--hook",
        "--extension",
        "--skills",
        "--export",
        "--max-time",
        "--approval-mode",
        "--trusted-extension",
        "--plugin-dir",
    }
)
_OPTIONAL_VALUE_FLAGS = frozenset({"--resume"})  # a value is taken only if it is not flag-like
_REPEATABLE = frozenset({"--config", "--add-dir", "--hook", "--extension", "--plugin-dir"})
_SHORT_FLAGS = {
    "-p": "--print",
    "-c": "--continue",
    "-r": "--resume",
    "-e": "--extension",
    "-v": "--version",
    "-h": "--help",
}
_CHOICES = {
    "--mode": ("text", "json", "rpc", "rpc-ui", "acp"),
    "--thinking": ("off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"),
    "--approval-mode": ("always-ask", "always-ask-cached", "write", "yolo"),
}
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
DEFAULT_TOOLS = [
    "read",
    "bash",
    "edit",
    "write",
    "grep",
    "glob",
    "lsp",
    "notebook",
    "browser",
    "task",
    "todo",
    "web_search",
]
FAIL_PRESETS: dict[str, tuple[int, str]] = {
    "auth": (401, "401 Failed to authenticate: invalid x-api-key (type=authentication_error)"),
    "credit": (
        400,
        "400 Credit balance is too low to access the Anthropic API. Please go to Plans & "
        "Billing to upgrade or purchase credits. (type=invalid_request_error)",
    ),
    "rate_limit": (429, "429 rate limit exceeded (type=rate_limit_error)"),
    "api_error": (500, "500 Internal server error (type=api_error)"),
}


def parse_duration(text: str) -> float | None:
    """Seconds for ``600``, ``5s``, ``10m``, ``1h30m``...; None unless positive and well formed."""
    text = text.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        total = float(text)
    elif re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ms|s|m|h|d))+", text):
        pairs = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", text)
        total = sum(float(number) * _DURATION_UNITS[unit] for number, unit in pairs)
    else:
        return None
    return total if total > 0 else None


class Parsed:
    """Result of parsing argv: flags by canonical long name, positionals, error or early exit."""

    def __init__(self) -> None:
        self.flags: dict[str, Any] = {}
        self.positional: list[str] = []
        self.error: tuple[str, int] | None = None  # (stderr text, exit status)
        self.early: str | None = None  # "version" | "help"

    def fail(self, text: str, code: int = 2) -> None:
        self.error = (text, code)

    def store(self, name: str, value: Any) -> None:
        if name in _REPEATABLE:
            self.flags.setdefault(name, []).append(value)
        else:
            self.flags[name] = value


def _check_value(parsed: Parsed, name: str, value: str) -> None:
    if name in _CHOICES and value not in _CHOICES[name]:
        expected = ", ".join(_CHOICES[name])
        parsed.fail(f'Error: Invalid {name} value: "{value}". Expected one of: {expected}.')
    elif name == "--max-time" and parse_duration(value) is None:
        parsed.fail(
            f'Error: Invalid --max-time value: "{value}". Expected a positive number of seconds '
            f'or duration like "5s", "10m", "1h".\n{_HELP_HINT}'
        )
    elif name == "--trusted-extension" and not os.path.isabs(value):
        parsed.fail(f"Error: --trusted-extension requires an absolute path: {value}\n{_HELP_HINT}")


def _take_flag(parsed: Parsed, name: str, inline: str | None, args: list[str]) -> None:
    """Consume the value (if any) of the recognised flag ``name`` and store it."""
    if name == "--version":
        parsed.early = "version"
    elif name == "--help":
        parsed.early = "help"
    elif name in _BOOL_FLAGS:
        if inline is not None:
            parsed.fail(f"Error: Flag {name} does not take a value\n{_HELP_HINT}")
        else:
            parsed.store(name, True)
    elif name in _OPTIONAL_VALUE_FLAGS:
        if inline is None and args and not args[0].startswith("-"):
            inline = args.pop(0)
        parsed.store(name, True if inline is None else inline)
    elif name == "--trusted-extension":
        value = inline if inline is not None else (args.pop(0) if args else "")
        if not value or value.startswith("-"):
            parsed.fail(
                f"Error: --trusted-extension requires a non-empty, non-flag value\n{_HELP_HINT}"
            )
            return
        _check_value(parsed, name, value)
        parsed.store(name, value)
    else:  # value flag
        if inline is None:
            if not args:
                parsed.fail(f"Error: Flag {name} expects a value\n{_HELP_HINT}")
                return
            inline = args.pop(0)
        _check_value(parsed, name, inline)
        if parsed.error is None:
            parsed.store(name, inline)


def parse_args(argv: list[str]) -> Parsed:
    """Parse argv the way the real CLI accepts it; never exits, errors land in ``Parsed``."""
    parsed = Parsed()
    args = list(argv)
    known = _BOOL_FLAGS | _VALUE_FLAGS | _OPTIONAL_VALUE_FLAGS | {"--version", "--help"}
    while args and parsed.error is None and parsed.early is None:
        token = args.pop(0)
        if token == "--":
            parsed.positional.extend(args)
            break
        if token.startswith("--"):
            name, has_inline, inline = token.partition("=")
            if name not in known:
                parsed.fail(f"Error: unknown flag: {name}\n{_HELP_HINT}")
                break
            _take_flag(parsed, name, inline if has_inline else None, args)
        elif token.startswith("-") and len(token) > 1:
            cluster = token[1:]
            while cluster and parsed.error is None and parsed.early is None:
                long_name = _SHORT_FLAGS.get(f"-{cluster[0]}")
                if long_name is None:
                    parsed.fail(f"Error: unknown flag: -{cluster[0]}\n{_HELP_HINT}")
                    break
                rest = cluster[1:]
                takes_value = long_name in _VALUE_FLAGS or long_name in _OPTIONAL_VALUE_FLAGS
                _take_flag(parsed, long_name, (rest or None) if takes_value else None, args)
                cluster = "" if takes_value else rest
        else:
            parsed.positional.append(token)
    return parsed


def resolve_model(model_flag: str | None, provider_flag: str | None) -> tuple[str, str]:
    """(provider, model) for ``--model`` (``provider/model`` allowed) and ``--provider``."""
    model = model_flag or DEFAULT_MODEL
    provider = provider_flag or "anthropic"
    if model_flag and "/" in model_flag and not provider_flag:
        provider, model = model_flag.split("/", 1)
    return provider, model


def resolve_tools(flags: dict[str, Any]) -> list[str]:
    if "--no-tools" in flags:
        return []
    raw = flags.get("--tools")
    if raw is None:
        return list(DEFAULT_TOOLS)
    return [name.strip() for name in str(raw).split(",") if name.strip()]


def enter_cwd(requested: str | None) -> tuple[tuple[str, int] | None, str]:
    """Honour ``--cwd``; returns (failure or None, the cwd to record)."""
    if requested is None:
        return None, os.getcwd()
    old = os.getcwd()
    target = os.path.abspath(requested)
    try:
        os.chdir(target)
    except OSError as exc:
        code = errno.errorcode.get(exc.errno or 0, "ERROR")
        detail = f"{code}: {exc.strerror.lower() if exc.strerror else 'error'}"
        message = (
            f"Error: Cannot change working directory to {requested}: "
            f"{detail}, chdir '{old}' -> '{target}'."
        )
        return (message, 1), target
    return None, os.getcwd()


def gather_prompt(parsed: Parsed, stdin_tty: bool) -> tuple[str, tuple[str, int] | None]:
    """Prompt text from positionals (``@file`` read) or stdin; or a startup failure."""
    printing = "--print" in parsed.flags
    if not parsed.positional:
        if stdin_tty:
            if printing:
                return "", None
            return "", (f"Error: interactive mode is not emulated by {PROG}; pass -p.", 2)
        return read_stdin_prompt(), None
    parts: list[str] = []
    for token in parsed.positional:
        if token.startswith("@") and len(token) > 1:
            path = os.path.abspath(token[1:])
            if not os.path.isfile(path):
                return "", (f"Error: File not found: {path}", 1)
            with open(path, encoding="utf-8", errors="replace") as handle:
                parts.append(handle.read())
        else:
            parts.append(token)
    return " ".join(parts), None


# ---------------------------------------------------------------------------------------------
# NDJSON event stream
# ---------------------------------------------------------------------------------------------


def chunk_text(text: str, size: int = 64) -> list[str]:
    return [text[start : start + size] for start in range(0, len(text), size)]


def usage_block(output_tokens: int, costs: dict[str, float] | None = None) -> dict[str, Any]:
    """Assistant ``usage`` with the real key set; zero cost unless ``costs`` is given."""
    costs = costs or {"input": 0, "output": 0, "total": 0}
    return {
        "input": 4,
        "output": output_tokens,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 4 + output_tokens,
        "cost": {
            "input": costs["input"],
            "output": costs["output"],
            "cacheRead": 0,
            "cacheWrite": 0,
            "total": costs["total"],
        },
        "cttl": {"ephemeral5m": 0},
    }


def usage_for_cost(usd: float) -> dict[str, Any]:
    first = round(usd * 0.25, 12)
    return usage_block(10, {"input": first, "output": round(usd - first, 12), "total": usd})


def timed(message: dict[str, Any], started: float) -> dict[str, Any]:
    elapsed = (time.monotonic() - started) * 1000
    return dict(message, duration=round(elapsed, 3), ttft=round(elapsed / 2, 3))


class OmpRun:
    """One emulated omp run: executes the scenario steps and emits the event stream."""

    def __init__(self, scenario: dict[str, Any], parsed: Parsed, prompt: str) -> None:
        flags = parsed.flags
        self.steps: list[dict[str, Any]] = scenario["steps"]
        self.prompt = prompt
        self.json_mode = flags.get("--mode", "text") == "json"
        self.provider, self.model = resolve_model(flags.get("--model"), flags.get("--provider"))
        self.api = "anthropic-messages" if self.provider == "anthropic" else "openai-responses"
        self.tools = resolve_tools(flags)
        self.configured = flags.get("--thinking", "auto")
        self.thinking = "low" if self.configured == "auto" else self.configured
        limit = parse_duration(flags["--max-time"]) if "--max-time" in flags else None
        self.deadline = STARTED + limit if limit else None
        self.session_id = new_uuid()
        self.messages: list[dict[str, Any]] = []
        self.header_done = False
        self.pending_notices: list[dict[str, Any]] = []
        self.responses = 0
        self.calls = 0

    # -- plumbing ---------------------------------------------------------------------------

    def emit(self, event: dict[str, Any]) -> None:
        if self.json_mode:
            write_out(json.dumps(event, separators=(",", ":")) + "\n")

    def ensure_header(self) -> None:
        """Emit session..user message once, lazily (see the module docstring)."""
        if self.header_done:
            return
        self.header_done = True
        self.emit(
            {
                "type": "session",
                "version": 3,
                "id": self.session_id,
                "timestamp": iso_now(),
                "cwd": os.getcwd(),
            }
        )
        for notice in self.pending_notices:
            self.emit(notice)
        self.pending_notices = []
        self.emit(
            {
                "type": "thinking_level_changed",
                "thinkingLevel": self.thinking,
                "configured": self.configured,
                "resolved": self.thinking,
            }
        )
        self.emit({"type": "agent_start"})
        self.emit({"type": "turn_start"})
        user = {
            "role": "user",
            "content": [{"type": "text", "text": self.prompt}],
            "attribution": "user",
            "timestamp": now_ms(),
        }
        self.messages.append(user)
        self.emit({"type": "message_start", "message": user})
        self.emit({"type": "message_end", "message": user})

    def new_assistant(
        self, content: list[dict[str, Any]], stop_reason: str, usage: dict[str, Any]
    ) -> dict[str, Any]:
        self.responses += 1
        return {
            "role": "assistant",
            "content": content,
            "api": self.api,
            "provider": self.provider,
            "model": self.model,
            "usage": usage,
            "stopReason": stop_reason,
            "timestamp": now_ms(),
            "requestControls": {
                "messageIndex": len(self.messages),
                "tools": {
                    "declared": list(self.tools),
                    "deferred": [],
                    "active": list(self.tools),
                },
                "effort": {"topLevel": self.thinking, "tail": self.thinking},
            },
            "responseId": f"msg_fake{self.responses:012d}",
        }

    def stream_assistant(
        self,
        content: list[dict[str, Any]],
        stop_reason: str,
        usage: dict[str, Any],
        *,
        start_content: list[dict[str, Any]] | None = None,
        updates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """message_start, message_update*, message_end; returns the completed message."""
        started = time.monotonic()
        message = self.new_assistant(content, stop_reason, usage)
        # Real streams carry a placeholder "stop" in message_start until the message completes.
        shown = "stop" if stop_reason in ("stop", "toolUse") else stop_reason
        opening = dict(
            message,
            content=content if start_content is None else start_content,
            stopReason=shown,
        )
        self.emit({"type": "message_start", "message": opening})
        for update in updates or []:
            self.emit({"type": "message_update", "assistantMessageEvent": update})
        done = timed(message, started)
        self.emit({"type": "message_end", "message": dict(done, completedAt=now_ms())})
        self.messages.append(done)
        return done

    def error_turn(self, stop_reason: str, message: str, status: int | None = None) -> int:
        """An assistant message that ended in ``error``/``aborted``, then agent_end; exit 1."""
        self.ensure_header()
        failed: dict[str, Any] = {
            "role": "assistant",
            "content": [],
            "api": self.api,
            "provider": self.provider,
            "model": self.model,
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": stop_reason,
            "timestamp": now_ms(),
        }
        if status is not None:
            failed["errorStatus"] = status
            failed["errorId"] = ERROR_ID
        failed["errorMessage"] = message
        self.emit({"type": "message_start", "message": failed})
        self.emit({"type": "message_end", "message": dict(failed, completedAt=now_ms())})
        self.emit({"type": "turn_end", "message": failed, "toolResults": []})
        self.messages.append(failed)
        self.emit(
            {"type": "agent_end", "messages": self.messages, "isTerminal": True, "yielded": True}
        )
        if not self.json_mode:
            write_err(message + "\n")
        return 1

    # -- deadline (--max-time) ----------------------------------------------------------------

    def abort(self) -> int:
        return self.error_turn("aborted", "Deadline exceeded")

    def sleep(self, seconds: float) -> int | None:
        seconds = max(0.0, seconds)
        if self.deadline is not None and time.monotonic() + seconds >= self.deadline:
            time.sleep(max(0.0, self.deadline - time.monotonic()))
            return self.abort()
        time.sleep(seconds)
        return None

    def hang(self) -> int | None:
        if self.deadline is not None:
            time.sleep(max(0.0, self.deadline - time.monotonic()))
            return self.abort()
        while True:
            time.sleep(3600)

    # -- steps --------------------------------------------------------------------------------

    def run(self) -> int:
        if not self.json_mode:
            write_err("Working...\n")
        for step in self.steps:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return self.abort()
            code = self.dispatch(step)
            if code is not None:
                return code
        return self.op_final({"text": "ok"})

    def dispatch(self, step: dict[str, Any]) -> int | None:
        op = step["op"]
        if run_shared_op(step, self.prompt):
            return None
        if op == "sleep":
            return self.sleep(float(step["seconds"]))
        if op == "hang":
            return self.hang()
        if op == "exit":
            return do_exit(step)
        return getattr(self, f"op_{op}")(step)

    def op_notice(self, step: dict[str, Any]) -> int | None:
        event = {
            "type": "notice",
            "level": step.get("level", "info"),
            "message": str(step["message"]),
            "source": step.get("source", "xdev"),
        }
        if not self.json_mode:
            write_err(event["message"] + "\n")
        elif self.header_done:
            self.emit(event)
        else:
            self.pending_notices.append(event)
        return None

    def op_cost(self, step: dict[str, Any]) -> int | None:
        self.ensure_header()
        usage = usage_for_cost(float(step["usd"]))
        self.stream_assistant([{"type": "text", "text": ""}], "toolUse", usage)
        return None

    def op_tool_call(self, step: dict[str, Any]) -> int | None:
        self.ensure_header()
        self.calls += 1
        call_id = f"toolu_fake{self.calls:08d}"
        tool = str(step["tool"])
        args = dict(step.get("args") or {})
        blocked = bool(step.get("blocked", False))
        intent = str(step.get("intent") or f"Calling {tool}")
        text = str(step.get("result") or ("blocked by policy" if blocked else "ok"))
        call_args = dict(args, i=intent)
        details: dict[str, Any] = {}
        if not blocked and "path" in args:
            details["resolvedPath"] = os.path.join(os.getcwd(), str(args["path"]))
        encoded = json.dumps(args)
        updates: list[dict[str, Any]] = [{"type": "toolcall_start", "contentIndex": 0}]
        for piece in chunk_text(encoded, max(16, -(-len(encoded) // 6))):
            updates.append({"type": "toolcall_delta", "contentIndex": 0, "delta": piece})
        updates.append(
            {
                "type": "toolcall_end",
                "contentIndex": 0,
                "toolCall": {
                    "type": "toolCall",
                    "id": call_id,
                    "name": tool,
                    "arguments": call_args,
                },
            }
        )
        done = self.stream_assistant(
            [
                {
                    "type": "toolCall",
                    "id": call_id,
                    "name": tool,
                    "arguments": call_args,
                    "intent": intent,
                }
            ],
            "toolUse",
            usage_block(16),
            start_content=[{"type": "toolCall", "id": call_id, "name": tool, "arguments": {}}],
            updates=updates,
        )
        self.emit(
            {
                "type": "tool_execution_start",
                "toolCallId": call_id,
                "toolName": tool,
                "args": args,
                "intent": intent,
            }
        )
        if step.get("partial"):
            partial = {
                "content": [{"type": "text", "text": str(step["partial"])}],
                "details": details,
            }
            self.emit(
                {
                    "type": "tool_execution_update",
                    "toolCallId": call_id,
                    "toolName": tool,
                    "args": args,
                    "partialResult": partial,
                }
            )
        content = [{"type": "text", "text": text}]
        self.emit(
            {
                "type": "tool_execution_end",
                "toolCallId": call_id,
                "toolName": tool,
                "result": {"content": content, "details": details},
                "isError": blocked,
            }
        )
        result = {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": tool,
            "content": content,
            "details": details,
            "isError": blocked,
            "timestamp": now_ms(),
        }
        self.emit({"type": "message_start", "message": result})
        self.emit({"type": "message_end", "message": result})
        self.messages.append(result)
        self.emit({"type": "turn_end", "message": done, "toolResults": [result]})
        self.emit({"type": "turn_start"})
        return None

    def op_final(self, step: dict[str, Any]) -> int | None:
        text = compose_final_text(step)
        self.ensure_header()
        updates: list[dict[str, Any]] = [{"type": "text_start", "contentIndex": 0}]
        for piece in chunk_text(text):
            updates.append({"type": "text_delta", "contentIndex": 0, "delta": piece})
        updates.append({"type": "text_end", "contentIndex": 0, "content": text})
        done = self.stream_assistant(
            [{"type": "text", "text": text}],
            "stop",
            usage_block(max(1, -(-len(text) // 4))),
            updates=updates,
        )
        self.emit({"type": "turn_end", "message": done, "toolResults": []})
        self.emit(
            {"type": "agent_end", "messages": self.messages, "isTerminal": True, "yielded": True}
        )
        if not self.json_mode:
            write_out(text + "\n")
        return 0

    def op_fail(self, step: dict[str, Any]) -> int | None:
        kind = step["kind"]
        if kind == "crash":
            return do_crash(str(step.get("message") or "simulated crash"))
        status, default = FAIL_PRESETS[kind]
        return self.error_turn("error", str(step.get("message") or default), status)


# ---------------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    scenario, scenario_error = load_scenario()
    if scenario["ignore_sigterm"]:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    parsed = parse_args(argv)
    stdin_tty = stdin_is_tty()
    failure = parsed.error
    cwd_label = os.getcwd()
    prompt = ""
    running = failure is None and parsed.early is None
    if running:
        failure, cwd_label = enter_cwd(parsed.flags.get("--cwd"))
    if running and failure is None:
        mode = parsed.flags.get("--mode", "text")
        if mode not in ("text", "json"):
            failure = (f"Error: --mode {mode} is not emulated by {PROG}.", 2)
    if running and failure is None:
        extension = parsed.flags.get("--trusted-extension")
        if extension is not None and not os.path.isfile(extension):
            failure = (f"error: Trusted extension must be an existing module file: {extension}", 1)
    if running and failure is None:
        prompt, failure = gather_prompt(parsed, stdin_tty)
    log_invocation(
        argv,
        cwd_label,
        prompt,
        scenario,
        stdin_tty,
        {"flags": parsed.flags, "positional": parsed.positional},
    )
    if parsed.early == "version":
        write_out(f"omp/{VERSION}\n")
        return 0
    if parsed.early == "help":
        write_out(f"omp v{VERSION} ({PROG} stub: help is not emulated)\n")
        return 0
    if failure is not None:
        write_err(failure[0] + "\n")
        return failure[1]
    if scenario_error is not None:
        write_err(f"{PROG}: {scenario_error}\n")
        return HARNESS_EXIT
    if not prompt:
        if "--print" not in parsed.flags:
            write_err(
                "Error: interactive mode requires a terminal, but stdin is not a TTY.\n"
                'Pass a prompt (`omp -p "…"`), pipe one on stdin, or use `--mode rpc`.\n'
            )
            return 2
        write_err(f"{PROG}: no prompt received (positional or stdin); nothing to run\n")
        if parsed.flags.get("--mode") == "json":
            header = {"type": "session", "version": 3, "id": new_uuid(), "timestamp": iso_now()}
            write_out(json.dumps(dict(header, cwd=os.getcwd()), separators=(",", ":")) + "\n")
        return 0
    try:
        return OmpRun(scenario, parsed, prompt).run()
    except (HarnessError, OSError) as exc:
        write_err(f"{PROG}: {exc}\n")
        return HARNESS_EXIT


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
