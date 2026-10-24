"""Executable emulator of the Claude Code ``claude`` CLI (2.1.293) for hermetic bridge tests.

Standalone (stdlib only, no repo imports): ``python fake_claude.py -p [claude flags] [prompt]``.
The option table is the ``--help`` text of the real CLI, parsed with commander's rules (variadic
``<x...>`` options swallow following non-flag arguments until the next flag, ``--`` ends options,
``choices`` are enforced).  Output formats follow real captures (``claude_real_*.json`` fixtures
plus probes of the installed binary).

Environment
    FAKE_AGENT_SCENARIO  JSON file ``{"steps": [...], "log_env": [...], "ignore_sigterm": false}``;
                         unset means a single ``final`` step with the text ``ok``.
    FAKE_AGENT_LOG       when set, one JSON line is appended per invocation before anything else
                         happens: ``argv`` (without argv[0]), ``cwd``, ``env_keys`` (sorted names
                         only), ``env`` (values of the scenario's ``log_env`` names), ``prompt``,
                         ``stdin_was_tty`` and ``parsed`` (``{"flags": {...}, "positional": []}``).

Prompt: the positional arguments joined by a space or, with none, stdin read to EOF when stdin
is not a TTY.  An empty prompt is the real "Input must be provided ..." error (exit 1).

Output (``-p`` is required; usage errors exit 2 although the real CLI exits 1)
    text (default)       the final text
    json                 ONE result object; with ``--verbose`` the array [system/init, assistant,
                         result] exactly like the captured fixtures (the real CLI does the same)
    stream-json          the same three messages as lines; needs ``--verbose`` like the real CLI
The exit status is 1 whenever the result has ``is_error`` true, else 0.

Scenario steps run in order; unknown ops and missing keys are harness errors (exit 99, nothing
executed).  Ops are the ones of fake_omp.py: write_file, append_file, delete_file, symlink,
chmod_exec, tool_call (counts as a tool use; ``blocked`` adds a ``permission_denials`` entry),
cost (cumulative ``total_cost_usd``; ``--max-budget-usd B`` stops the run once the total reaches
B with ``error_max_budget_usd``), sleep, hang, spawn_child, echo_prompt_to, notice (stderr),
garbage_stdout, final{text,report?}, fail{kind,message?} (auth | credit | rate_limit | api_error
| crash) and exit{code?,stdout?,stderr?}.  A run that ends without final/fail/exit finishes with
the text ``ok``.  All output of a run is printed at the end (nothing for ``fail crash``/``exit``).
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from typing import Any

PROG = "fake_claude"
VERSION = "2.1.293"
DEFAULT_MODEL = "claude-opus-5-5"
SCENARIO_ENV = "FAKE_AGENT_SCENARIO"
LOG_ENV = "FAKE_AGENT_LOG"
HARNESS_EXIT = 99  # scenario or harness misuse
CRASH_EXIT = 3
PARSE_ERROR_EXIT = 2  # the real CLI (commander) exits 1 on usage errors; the harness spec says 2
RUNTIME_ERROR_EXIT = 1
STARTED = time.monotonic()

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
# claude command line (the ``--help`` of the real CLI, parsed with commander's rules)
# ---------------------------------------------------------------------------------------------

_DECLARED = (
    "--add-dir <directories...>",
    "--agent <agent>",
    "--agents <json-or-file>",
    "--allow-dangerously-skip-permissions",
    "--allowedTools, --allowed-tools <tools...>",
    "--append-system-prompt <prompt>",
    "--autocompact <auto|tokens>",
    "--ax-screen-reader",
    "--bg, --background",
    "--bare",
    "--betas <betas...>",
    "--brief",
    "--chrome",
    "--cloud [description|session_id|url]",
    "-c, --continue",
    "--dangerously-skip-permissions",
    "-d, --debug [filter]",
    "--debug-file <path>",
    "--desktop",
    "--disable-slash-commands",
    "--disallowedTools, --disallowed-tools <tools...>",
    "--effort <level>",
    "--environment <environment_id>",
    "--exclude-dynamic-system-prompt-sections",
    "--fallback-model <model>",
    "--file <specs...>",
    "--fork-session",
    "--forward-subagent-text",
    "--from-pr [value]",
    "-h, --help",
    "--ide",
    "--include-hook-events",
    "--include-partial-messages",
    "--input-format <format>",
    "--json-schema <schema>",
    "--max-budget-usd <amount>",
    "--mcp-config <configs...>",
    "--model <model>",
    "-n, --name <name>",
    "--no-chrome",
    "--no-session-persistence",
    "--output-format <format>",
    "--permission-mode <mode>",
    "--permission-prompts <target>",
    "--plugin-dir <path>",
    "--plugin-url <url>",
    "-p, --print",
    "--prompt-suggestions [value]",
    "--remote-control [name]",
    "--remote-control-session-name-prefix <prefix>",
    "--replay-user-messages",
    "--restricted",
    "-r, --resume [value]",
    "--safe-mode",
    "--session-id <uuid>",
    "--setting-sources <sources>",
    "--settings <file-or-json>",
    "--strict-mcp-config",
    "--system-prompt <prompt>",
    "--system-prompt-snapshot <on|off>",
    "--teleport [session]",
    "--tmux",
    "--tools <tools...>",
    "--verbose",
    "-v, --version",
    "-w, --worktree [name]",
)
_CHOICES = {
    "--input-format": ("text", "stream-json"),
    "--output-format": ("text", "json", "stream-json"),
    "--permission-mode": ("acceptEdits", "auto", "bypassPermissions", "manual", "dontAsk", "plan"),
    "--permission-prompts": ("host", "none"),
    "--prompt-suggestions": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "--system-prompt-snapshot": ("on", "off"),
}
_REPEATABLE = frozenset({"--plugin-dir", "--plugin-url"})
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
DEFAULT_TOOLS = [
    "Task",
    "Bash",
    "Glob",
    "Grep",
    "Read",
    "Edit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "TodoWrite",
    "WebSearch",
]


class Option:
    """One declared commander option: its names, argument kind and variadic-ness."""

    _SPEC = re.compile(r"((?:-{1,2}[\w-]+)(?:, -{1,2}[\w-]+)*)(?: ([<\[].*[>\]]))?")

    def __init__(self, declared: str) -> None:
        match = self._SPEC.fullmatch(declared)
        if match is None:
            raise ValueError(f"bad option declaration: {declared!r}")
        argument = match.group(2) or ""
        self.declared = declared
        self.names = match.group(1).split(", ")
        self.canonical = next(name for name in self.names if name.startswith("--"))
        if argument.startswith("<"):
            self.kind = "required"
        elif argument.startswith("["):
            self.kind = "optional"
        else:
            self.kind = "bool"
        self.variadic = argument.endswith("...>")


OPTIONS: dict[str, Option] = {}
for _declared in _DECLARED:
    _option = Option(_declared)
    for _name in _option.names:
        OPTIONS[_name] = _option


class Parsed:
    """Result of parsing argv: flags by canonical long name, positionals, error or early exit."""

    def __init__(self) -> None:
        self.flags: dict[str, Any] = {}
        self.positional: list[str] = []
        self.error: tuple[str, int] | None = None  # (stderr text, exit status)
        self.early: str | None = None  # "version" | "help"
        self.help = False

    def fail(self, text: str, code: int = PARSE_ERROR_EXIT) -> None:
        self.error = (text, code)

    def store(self, option: Option, value: Any) -> None:
        name = option.canonical
        if option.variadic or name in _REPEATABLE:
            self.flags.setdefault(name, []).append(value)
        else:
            self.flags[name] = value


def maybe_option(token: str) -> bool:
    return len(token) > 1 and token.startswith("-")


def positive_number(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if 0 < number < float("inf") else None


def add_value(parsed: Parsed, option: Option, value: str) -> None:
    """Validate (choices, budget) and store one option argument."""
    choices = _CHOICES.get(option.canonical)
    if choices is not None and value not in choices:
        allowed = ", ".join(choices)
        parsed.fail(
            f"error: option '{option.declared}' argument '{value}' is invalid. "
            f"Allowed choices are {allowed}."
        )
    elif option.canonical == "--max-budget-usd" and positive_number(value) is None:
        parsed.fail(
            f"error: option '{option.declared}' argument '{value}' is invalid. "
            "--max-budget-usd must be a positive number greater than 0"
        )
    else:
        parsed.store(option, value)


def handle_option(parsed: Parsed, option: Option, args: list[str]) -> Option | None:
    """Apply a recognised option; returns it when it is variadic and keeps consuming values."""
    if option.canonical == "--version":
        parsed.early = "version"
    elif option.canonical == "--help":
        parsed.help = True
    elif option.kind == "required":
        if not args:
            parsed.fail(f"error: option '{option.declared}' argument missing")
        else:  # a required argument is taken unconditionally, even when it looks like a flag
            add_value(parsed, option, args.pop(0))
            return option if option.variadic else None
    elif option.kind == "optional":
        if args and not maybe_option(args[0]):
            add_value(parsed, option, args.pop(0))
        else:
            parsed.store(option, True)
    else:
        parsed.store(option, True)
    return None


def parse_args(argv: list[str]) -> Parsed:
    """Parse argv with commander's semantics; never exits, errors land in ``Parsed``."""
    parsed = Parsed()
    args = list(argv)
    unknown: list[str] = []
    dest = parsed.positional
    active: Option | None = None
    while args and parsed.error is None and parsed.early is None:
        token = args.pop(0)
        if token == "--":
            if dest is unknown:
                dest.append(token)
            dest.extend(args)
            break
        if active is not None and not maybe_option(token):
            add_value(parsed, active, token)
            continue
        active = None
        if maybe_option(token):
            option = OPTIONS.get(token)
            if option is not None:
                active = handle_option(parsed, option, args)
                continue
            if not token.startswith("--") and len(token) > 2:  # clustered short flags: -pc
                first = OPTIONS.get(token[:2])
                if first is not None:
                    if first.kind == "bool":
                        handle_option(parsed, first, args)
                        args.insert(0, "-" + token[2:])
                    else:
                        add_value(parsed, first, token[2:])
                    continue
            if token.startswith("--") and "=" in token:
                name, _, value = token.partition("=")
                inline = OPTIONS.get(name)
                if inline is not None and (inline.kind != "bool" or name == "--tmux"):
                    add_value(parsed, inline, value)
                    continue
            dest = unknown
        dest.append(token)
    if parsed.error is None and parsed.early is None:
        if parsed.help:
            parsed.early = "help"
        elif unknown:
            parsed.fail(f"error: unknown option '{unknown[0]}'")
    return parsed


def resolve_tools(values: list[str] | None) -> list[str]:
    """The ``tools`` list of system/init for ``--tools`` ('' = none, 'default' = all)."""
    if values is None:
        return list(DEFAULT_TOOLS)
    names: list[str] = []
    for value in values:
        for part in re.split(r"[,\s]+", value):
            if part == "default":
                names.extend(DEFAULT_TOOLS)
            elif part:
                names.append(part)
    return names


def startup_failure(parsed: Parsed) -> tuple[str, int] | None:
    """Checks the real CLI makes after option parsing, before reading the prompt."""
    flags = parsed.flags
    if "--print" not in flags:
        return (f"Error: interactive mode is not emulated by {PROG}; pass -p.", PARSE_ERROR_EXIT)
    if flags.get("--output-format") == "stream-json" and "--verbose" not in flags:
        text = "Error: When using --print, --output-format=stream-json requires --verbose"
        return (text, RUNTIME_ERROR_EXIT)
    session = flags.get("--session-id")
    if session is not None and not _UUID.fullmatch(session):
        return ("Error: Invalid session ID. Must be a valid UUID.", RUNTIME_ERROR_EXIT)
    return None


# ---------------------------------------------------------------------------------------------
# result messages
# ---------------------------------------------------------------------------------------------

FAIL_PRESETS: dict[str, dict[str, Any]] = {
    "auth": {
        "text": "Failed to authenticate: OAuth session expired and could not be refreshed",
        "error": "authentication_failed",
        "status": None,
        "request_id": False,
        "key_source": "none",
    },
    "credit": {
        "text": "Credit balance is too low",
        "error": "billing_error",
        "status": 400,
        "request_id": True,
        "key_source": "ANTHROPIC_API_KEY",
    },
    "rate_limit": {
        "text": "rate limit exceeded",
        "error": "rate_limit",
        "status": 429,
        "request_id": True,
        "key_source": None,
    },
    "api_error": {
        "text": "API Error: 500 Internal server error",
        "error": "server_error",
        "status": 500,
        "request_id": True,
        "key_source": None,
    },
}


def tokens_of(text: str) -> int:
    return max(1, -(-len(text) // 4))


def message_usage(input_tokens: int, output_tokens: int, *, synthetic: bool) -> dict[str, Any]:
    """``usage`` of an assistant message (key set of the real captures)."""
    return {
        "output_tokens_details": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "service_tier": None if synthetic else "standard",
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "inference_geo": None,
        "iterations": None,
        "speed": None,
        "fallback_credit": None,
    }


def result_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """``usage`` of the result message (key set of the real captures)."""
    return {
        "output_tokens_details": {"thinking_tokens": 0},
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": output_tokens,
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        "service_tier": "standard",
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "inference_geo": "",
        "iterations": [],
        "speed": "standard",
        "fallback_credit": None,
    }


def subagent_stats() -> dict[str, Any]:
    return {
        "spawned": 0,
        "requested": {"background": 0, "foreground": 0, "unset": 0},
        "started_in_background": 0,
        "max_depth": 0,
        "spawned_by_subagents": 0,
        "completed": 0,
        "failed": 0,
        "killed": {"parent": 0, "user": 0, "system": 0},
        "refused": {"depth_limit": 0, "concurrency_limit": 0, "budget": 0},
        "by_type": {},
    }


class ClaudeRun:
    """One emulated claude run: executes the steps, then prints the result in the chosen format."""

    def __init__(self, scenario: dict[str, Any], parsed: Parsed, prompt: str) -> None:
        flags = parsed.flags
        self.steps: list[dict[str, Any]] = scenario["steps"]
        self.prompt = prompt
        self.model = flags.get("--model") or DEFAULT_MODEL
        self.output_format = flags.get("--output-format", "text")
        self.verbose = "--verbose" in flags
        default_mode = (
            "bypassPermissions" if "--dangerously-skip-permissions" in flags else "default"
        )
        self.permission_mode = flags.get("--permission-mode") or default_mode
        self.tools = resolve_tools(flags.get("--tools"))
        self.budget = float(flags["--max-budget-usd"]) if "--max-budget-usd" in flags else None
        self.session_id = flags.get("--session-id") or new_uuid()
        self.cost = 0.0
        self.tool_uses = 0
        self.denials: list[dict[str, Any]] = []

    # -- steps --------------------------------------------------------------------------------

    def run(self) -> int:
        for step in self.steps:
            code = self.dispatch(step)
            if code is not None:
                return code
        return self.op_final({"text": "ok"})

    def dispatch(self, step: dict[str, Any]) -> int | None:
        op = step["op"]
        if run_shared_op(step, self.prompt):
            return None
        if op == "sleep":
            time.sleep(max(0.0, float(step["seconds"])))
            return None
        if op == "hang":
            while True:
                time.sleep(3600)
        if op == "exit":
            return do_exit(step)
        return getattr(self, f"op_{op}")(step)

    def op_notice(self, step: dict[str, Any]) -> int | None:
        write_err(str(step["message"]) + "\n")
        return None

    def op_tool_call(self, step: dict[str, Any]) -> int | None:
        self.tool_uses += 1
        if step.get("blocked"):
            denial = {
                "tool_name": str(step["tool"]),
                "tool_use_id": f"toolu_fake{self.tool_uses:08d}",
                "tool_input": dict(step.get("args") or {}),
            }
            self.denials.append(denial)
        return None

    def op_cost(self, step: dict[str, Any]) -> int | None:
        self.cost += float(step["usd"])
        if self.budget is not None and self.cost >= self.budget:
            return self.finish(
                text=f"Reached maximum budget (${self.budget:g})",
                assistant_text="",
                subtype="error_max_budget_usd",
                is_error=True,
                terminal_reason="budget_exhausted",
                stop_reason=None,
            )
        return None

    def op_final(self, step: dict[str, Any]) -> int | None:
        return self.finish(text=compose_final_text(step))

    def op_fail(self, step: dict[str, Any]) -> int | None:
        kind = step["kind"]
        if kind == "crash":
            return do_crash(str(step.get("message") or "simulated crash"))
        preset = FAIL_PRESETS[kind]
        return self.finish(
            text=str(step.get("message") or preset["text"]),
            is_error=True,
            terminal_reason="api_error",
            stop_reason="stop_sequence",
            api_error_status=preset["status"],
            failure=preset,
        )

    # -- output -------------------------------------------------------------------------------

    def finish(
        self,
        *,
        text: str,
        assistant_text: str | None = None,
        subtype: str = "success",
        is_error: bool = False,
        terminal_reason: str = "completed",
        stop_reason: str | None = "end_turn",
        api_error_status: int | None = None,
        failure: dict[str, Any] | None = None,
    ) -> int:
        """Build [system/init, assistant, result] and print them in the requested format."""
        synthetic = failure is not None
        spoken = text if assistant_text is None else assistant_text
        input_tokens = 0 if synthetic else 3 + len(self.prompt) // 4
        output_tokens = 0 if synthetic else tokens_of(spoken)
        elapsed = int((time.monotonic() - STARTED) * 1000)
        key_source = failure["key_source"] if failure else None
        if key_source is None:
            key_source = "ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY") else "none"
        init = {
            "type": "system",
            "subtype": "init",
            "cwd": os.getcwd(),
            "session_id": self.session_id,
            "tools": self.tools,
            "mcp_servers": [],
            "model": self.model,
            "permissionMode": self.permission_mode,
            "slash_commands": [],
            "apiKeySource": key_source,
            "claude_code_version": VERSION,
            "output_style": "default",
            "skills": [],
            "analytics_disabled": False,
            "product_feedback_disabled": False,
            "uuid": new_uuid(),
            "fast_mode_state": "off",
            "fast_mode_disabled_reason": "sdk_opt_in_required",
            "per_turn_effort_active": True,
            "view_mode": "default",
        }
        assistant: dict[str, Any] = {
            "type": "assistant",
            "message": {
                "diagnostics": None,
                "id": "msg_fake" + new_uuid().replace("-", "")[:20],
                "container": None,
                "model": "<synthetic>" if synthetic else self.model,
                "role": "assistant",
                "stop_details": None,
                "stop_reason": "stop_sequence" if synthetic else stop_reason,
                "stop_sequence": "" if synthetic else None,
                "type": "message",
                "usage": message_usage(input_tokens, output_tokens, synthetic=synthetic),
                "content": [{"type": "text", "text": spoken}],
                "context_management": None,
            },
            "parent_tool_use_id": None,
            "session_id": self.session_id,
            "uuid": new_uuid(),
            "timestamp": iso_now(),
        }
        if failure is not None:
            assistant["error"] = failure["error"]
            if failure["request_id"]:
                assistant["request_id"] = "req_fake" + new_uuid().replace("-", "")[:20]
            assistant["is_api_error_message"] = True
        model_usage = {}
        if not synthetic:
            model_usage[self.model] = {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "webSearchRequests": 0,
                "costUSD": self.cost,
                "contextWindow": 200000,
                "maxOutputTokens": 32000,
            }
        result = {
            "duration_api_ms": 0 if synthetic else elapsed // 2,
            "stop_reason": "stop_sequence" if synthetic else stop_reason,
            "session_id": self.session_id,
            "total_cost_usd": self.cost,
            "usage": result_usage(input_tokens, output_tokens),
            "modelUsage": model_usage,
            "permission_denials": self.denials,
            "terminal_reason": terminal_reason,
            "fast_mode_state": "off",
            "fast_mode_disabled_reason": "sdk_opt_in_required",
            "subagent_stats": subagent_stats(),
            "safety_stops": 0,
            "is_error": is_error,
            "num_turns": 1 + self.tool_uses,
            "subtype": subtype,
            "api_error_status": api_error_status,
            "result": text,
            "type": "result",
            "duration_ms": elapsed,
            "uuid": new_uuid(),
            "queued_turn_count": 0,
            "result_index": 0,
        }
        return self.render([init, assistant, result])

    def render(self, messages: list[dict[str, Any]]) -> int:
        result = messages[-1]
        compact = (",", ":")
        if self.output_format == "json":
            payload: Any = messages if self.verbose else result
            write_out(json.dumps(payload, separators=compact) + "\n")
        elif self.output_format == "stream-json":
            for message in messages:
                write_out(json.dumps(message, separators=compact) + "\n")
        else:
            write_out(str(result["result"]) + "\n")
        return 1 if result["is_error"] else 0


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
    prompt = ""
    if failure is None and parsed.early is None:
        failure = startup_failure(parsed)
    if failure is None and parsed.early is None:
        if parsed.positional:
            prompt = " ".join(parsed.positional)
        elif not stdin_tty:
            prompt = read_stdin_prompt()
    log_invocation(
        argv,
        os.getcwd(),
        prompt,
        scenario,
        stdin_tty,
        {"flags": parsed.flags, "positional": parsed.positional},
    )
    if parsed.early == "version":
        write_out(f"{VERSION} (Claude Code)\n")
        return 0
    if parsed.early == "help":
        write_out(
            f"Usage: claude [options] [command] [prompt] ({PROG} stub: help is not emulated)\n"
        )
        return 0
    if failure is not None:
        write_err(failure[0] + "\n")
        return failure[1]
    if scenario_error is not None:
        write_err(f"{PROG}: {scenario_error}\n")
        return HARNESS_EXIT
    effort = parsed.flags.get("--effort")
    if effort is not None and effort not in _EFFORT_LEVELS:
        valid = ", ".join(_EFFORT_LEVELS)
        write_err(
            f"Warning: Unknown --effort value '{effort}' — ignoring it and using the default "
            f"effort. Valid values: {valid}.\n"
        )
    if not prompt:
        write_err(
            "Error: Input must be provided either through stdin or as a prompt argument "
            "when using --print\n"
        )
        return RUNTIME_ERROR_EXIT
    try:
        return ClaudeRun(scenario, parsed, prompt).run()
    except (HarnessError, OSError) as exc:
        write_err(f"{PROG}: {exc}\n")
        return HARNESS_EXIT


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
