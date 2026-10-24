"""Command line interface: ``python -m tools.agent_bridge <command>``.

Dry run is the default everywhere.  A real execution needs ``dry_run: false`` in the configuration
*and* ``--execute`` on the command line *and* an explicit confirmation (``--yes`` when unattended).
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .agents import BACKENDS, build_backends
from .config import BridgeConfig, load_config, resolve_github_token
from .errors import BridgeError, ConfigError, TaskValidationError
from .github import HttpGitHubClient
from .gitops import GitOps
from .ledger import Ledger
from .models import RunStatus
from .publisher import ReadOnlyGitHub
from .redact import Redactor
from .schema import parse_task, validate_against_config
from .worker import PollReport, Preview, Worker

__all__ = ["main"]

CONFIG_ENV = "LAWNBERRY_AGENT_BRIDGE_CONFIG"
DEFAULT_CONFIG = Path("config/agent-bridge.yaml")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3


@dataclass
class Runtime:
    cfg: BridgeConfig
    worker: Worker
    ledger: Ledger
    github: Any
    git: GitOps
    redactor: Redactor


class _RedactingFilter(logging.Filter):
    """Scrub secrets from the message *and* from any traceback or stack text of a record."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor
        self._plain = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redactor.redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            # Render the traceback here; the Formatter reuses exc_text instead of formatting
            # exc_info itself, so an exception message cannot carry a secret into a log.
            record.exc_text = self._redactor.redact(self._plain.formatException(record.exc_info))
        elif record.exc_text:
            record.exc_text = self._redactor.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redactor.redact(record.stack_info)
        return True


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating log whose files (current and every rotated one) are readable only by the owner."""

    def _open(self) -> Any:
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
            return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)
        except BaseException:
            os.close(fd)
            raise


def setup_logging(state_dir: Path, redactor: Redactor, *, verbose: bool) -> None:
    root = logging.getLogger("agent_bridge")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console.addFilter(_RedactingFilter(redactor))
    root.addHandler(console)
    try:
        log_dir = state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        rotating = _PrivateRotatingFileHandler(
            log_dir / "bridge.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        rotating.setFormatter(fmt)
        rotating.addFilter(_RedactingFilter(redactor))
        root.addHandler(rotating)
    except OSError:
        root.warning("file logging is unavailable in %s", state_dir)


def find_config(arg: str | None) -> Path:
    candidate = arg or os.environ.get(CONFIG_ENV) or str(DEFAULT_CONFIG)
    return Path(candidate).expanduser()


def build_runtime(cfg: BridgeConfig, *, execute: bool, verbose: bool = False) -> Runtime:
    token = resolve_github_token(cfg.github)
    extra = [os.environ[name] for name in cfg.extra_secret_env if os.environ.get(name)]
    redactor = Redactor([token, *extra])
    state_dir = cfg.resolved_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    setup_logging(state_dir, redactor, verbose=verbose)
    client = HttpGitHubClient(cfg.github, cfg.repository, token)
    github: Any = client if execute else ReadOnlyGitHub(client)
    repo_path = cfg.repo_path or Path.cwd()
    git = GitOps(
        repo_path,
        cfg.git,
        state_dir=state_dir,
        token=token if cfg.git.push_auth == "token" else None,
        redactor=redactor,
    )
    ledger = Ledger(state_dir / "ledger.sqlite")
    backends = build_backends(cfg.agent, redactor=redactor, max_log_bytes=cfg.limits.max_log_bytes)
    worker = Worker(
        cfg, github=github, ledger=ledger, git=git, backends=backends, redactor=redactor
    )
    return Runtime(cfg, worker, ledger, github, git, redactor)


def _print(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _dump(data: Any) -> None:
    _print(json.dumps(data, indent=2, sort_keys=True, default=str))


def render_preview(preview: Preview, *, show_prompt: bool) -> str:
    lines = [
        f"Issue #{preview.issue_number}: {preview.title}",
        f"  approved by {preview.approver} (label event {preview.approval_event_id})",
        f"  backend {preview.backend}, scope {preview.scope}, "
        f"budget ${preview.budget_usd:.2f}, timeout {preview.timeout_minutes:g} min",
        f"  branch {preview.branch} from {preview.base[:12]}",
        f"  may write: {' AND '.join(', '.join(s) for s in preview.write_allow_sets)}",
        f"  never writes: {', '.join(preview.write_deny[:6])} ... "
        f"({len(preview.write_deny)} patterns)",
        f"  checks: {', '.join(preview.checks) or 'none'}",
        f"  command: {' '.join(preview.command)}",
        f"  prompt: {preview.prompt_chars} characters on stdin",
    ]
    lines += [f"  note: {note}" for note in preview.notes]
    if show_prompt:
        lines += ["", "----- prompt -----", preview.prompt, "----- end prompt -----"]
    return "\n".join(lines)


def render_report(report: PollReport, *, execute: bool, show_prompt: bool = False) -> str:
    lines = [f"considered {report.considered} candidate issue(s)"]
    lines += [f"recovery: {item}" for item in report.recovered]
    for preview in report.previews:
        lines += ["", "DRY RUN - would execute:", render_preview(preview, show_prompt=show_prompt)]
    for refusal in report.refused:
        lines.append(f"skipped {refusal.render()}")
    for run in report.runs:
        lines.append(
            f"run {run.run_id[:12]} issue #{run.issue_number}: {run.status.value}/{run.phase.value}"
            + (f" PR #{run.pr_number} {run.pr_url}" if run.pr_number else "")
            + (f" ({run.failure_kind}: {run.failure_detail})" if run.failure_kind else "")
        )
    lines += [f"error: {err}" for err in report.errors]
    if not execute and not report.previews:
        lines.append("dry run: nothing would be executed")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- commands
def cmd_backends(cfg: BridgeConfig, _args: argparse.Namespace) -> int:
    redactor = Redactor()
    rows = []
    for name, backend in build_backends(cfg.agent, redactor=redactor).items():
        rows.append(
            {
                "backend": name,
                "default": name == cfg.agent.default,
                "binary": backend.resolve_binary(),
                "version": backend.version(),
            }
        )
    _dump(rows)
    return EXIT_OK


def cmd_validate_config(cfg: BridgeConfig, _args: argparse.Namespace) -> int:
    _print(f"configuration for {cfg.repository} is valid")
    _print(f"  dry_run: {cfg.dry_run}")
    _print(f"  default agent: {cfg.agent.default} (allowed: {', '.join(cfg.agent.allowed)})")
    _print(
        f"  approvers: {', '.join(cfg.approvers)}; task authors: {', '.join(cfg.trusted_authors)}"
    )
    _print(f"  scopes: {', '.join(cfg.scopes.allowed)}; state: {cfg.resolved_state_dir()}")
    return EXIT_OK


def cmd_validate_task(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    if args.file:
        try:
            body = Path(args.file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise BridgeError(f"cannot read task file {args.file}: {exc}") from exc
        title = args.title or Path(args.file).stem
    else:
        rt = build_runtime(cfg, execute=False, verbose=args.verbose)
        issue = rt.github.get_issue(args.issue)
        title, body = issue.title, issue.body
    try:
        task = parse_task(title, body)
    except TaskValidationError as exc:
        _print("INVALID task:")
        for error in exc.errors:
            _print(f"  - {error}")
        return EXIT_FAILED
    problems = validate_against_config(task.spec, cfg)
    if problems:
        _print("task is well formed but not permitted by this configuration:")
        for problem in problems:
            _print(f"  - {problem}")
        return EXIT_FAILED
    _print("task is valid")
    _print(
        f"  revision {task.revision[:16]}  scope {task.spec.scope}  "
        f"agent {task.spec.agent or cfg.agent.default}"
    )
    return EXIT_OK


def cmd_setup_labels(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    if not args.yes:
        _print("this creates the agent labels on GitHub; re-run with --yes to confirm")
        return EXIT_USAGE
    rt = build_runtime(cfg, execute=True, verbose=args.verbose)
    created = rt.github.ensure_labels(list(cfg.labels.all()))
    _print(f"created labels: {', '.join(created) if created else 'none (all present)'}")
    return EXIT_OK


def cmd_preview(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    rt = build_runtime(cfg, execute=False, verbose=args.verbose)
    report = rt.worker.poll_once(execute=False, only_issue=args.issue)
    if args.json:
        _dump(
            {
                "considered": report.considered,
                "previews": [p.to_dict(include_prompt=args.show_prompt) for p in report.previews],
                "refused": [r.render() for r in report.refused],
                "recovery": report.recovered,
                "errors": report.errors,
            }
        )
    else:
        _print(render_report(report, execute=False, show_prompt=args.show_prompt))
    return EXIT_FAILED if report.errors else EXIT_OK


def _authorize_execution(cfg: BridgeConfig, args: argparse.Namespace, what: str) -> int | None:
    """Return an exit code when execution must not proceed, else ``None``."""
    if not args.execute:
        return None
    if cfg.dry_run:
        _print("refusing to execute: the configuration has dry_run: true (set it to false first)")
        return EXIT_CONFIG
    if args.yes:
        return None
    if not sys.stdin.isatty():
        _print("refusing to execute without a human present: pass --yes to authorise explicitly")
        return EXIT_USAGE
    try:
        answer = input(f"{what} - authorise real execution? [y/N] ").strip().lower()
    except EOFError:  # Ctrl-D at the prompt is a "no"
        _print()
        answer = ""
    if answer not in {"y", "yes"}:
        _print("not authorised; nothing was executed")
        return EXIT_USAGE
    return None


def cmd_run_once(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    execute = bool(args.execute)
    denied = _authorize_execution(
        cfg, args, "run the next approved issue" if not args.issue else f"run issue #{args.issue}"
    )
    if denied is not None:
        return denied
    rt = build_runtime(cfg, execute=execute, verbose=args.verbose)
    report = rt.worker.poll_once(execute=execute, only_issue=args.issue)
    _print(render_report(report, execute=execute, show_prompt=args.show_prompt))
    failed = any(r.status in {RunStatus.BLOCKED} for r in report.runs) or bool(report.errors)
    return EXIT_FAILED if failed else EXIT_OK


def cmd_serve(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    execute = bool(args.execute)
    if execute and not args.yes:
        _print("serve --execute runs unattended: pass --yes to authorise it explicitly")
        return EXIT_USAGE
    denied = _authorize_execution(cfg, args, "serve")
    if denied is not None:
        return denied
    rt = build_runtime(cfg, execute=execute, verbose=args.verbose)
    stop = threading.Event()

    def handle(signum: int, _frame: Any) -> None:
        logging.getLogger("agent_bridge").warning("signal %s received; stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    logging.getLogger("agent_bridge").info(
        "serving %s (%s); polling every %ss",
        cfg.repository,
        "EXECUTE" if execute else "dry run",
        cfg.poll.interval_seconds,
    )
    rt.worker.serve(execute=execute, stop=stop)
    return EXIT_OK


def cmd_status(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(cfg.resolved_state_dir() / "ledger.sqlite")
    runs = ledger.list_runs(limit=args.limit)
    if args.json:
        _dump(
            [
                {
                    "run_id": r.run_id,
                    "issue": r.issue_number,
                    "status": r.status.value,
                    "phase": r.phase.value,
                    "backend": r.backend,
                    "pr": r.pr_number,
                    "failure": r.failure_kind,
                    "started": r.started_at,
                    "finished": r.finished_at,
                }
                for r in runs
            ]
        )
        return EXIT_OK
    if not runs:
        _print("no runs recorded")
    for r in runs:
        cost = ((r.result or {}).get("agent") or {}).get("cost_usd")
        cost_text = "-" if cost is None else f"${cost:.4f}"
        _print(
            f"{r.run_id[:12]}  #{r.issue_number:<5} {r.status.value:<10} {r.phase.value:<13} "
            f"{r.backend:<6} PR {r.pr_number or '-':<5} cost {cost_text:<9} "
            f"{r.failure_kind or ''}"
        )
    return EXIT_OK


def cmd_recover(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    execute = bool(args.execute)
    denied = _authorize_execution(cfg, args, "recover interrupted runs")
    if denied is not None:
        return denied
    rt = build_runtime(cfg, execute=execute, verbose=args.verbose)
    actions = rt.worker.recover(execute=execute)
    _print("\n".join(actions) if actions else "nothing to recover")
    return EXIT_OK


def resolve_run_id(ledger: Ledger, given: str) -> str:
    """Full run id for a full id or a unique prefix (``status`` and comments show 12 characters)."""
    prefix = given.strip().lower()
    if not prefix:
        raise BridgeError("--run-id must not be empty")
    matches = [r.run_id for r in ledger.list_runs(limit=1_000_000) if r.run_id.startswith(prefix)]
    if not matches:
        raise BridgeError(f"no run has an id starting with {given!r} (see `status`)")
    if len(matches) > 1:
        shown = ", ".join(run_id[:16] for run_id in matches[:5])
        raise BridgeError(f"{given!r} matches {len(matches)} runs ({shown}); give more characters")
    return matches[0]


def cmd_cancel(cfg: BridgeConfig, args: argparse.Namespace) -> int:
    ledger = Ledger(cfg.resolved_state_dir() / "ledger.sqlite")
    run_id = resolve_run_id(ledger, args.run_id) if args.run_id is not None else None
    count = ledger.request_cancel(run_id=run_id, issue_number=args.issue)
    _print(
        f"cancellation requested for {count} run(s); "
        "a running worker stops its agent within seconds"
    )
    return EXIT_OK if count else EXIT_FAILED


def cmd_smoke(cfg: BridgeConfig | None, args: argparse.Namespace) -> int:
    from .smoke import run_smoke

    report = run_smoke(
        args.backend,
        model=args.model,
        binary=args.binary,
        timeout_s=args.timeout,
        budget_usd=args.budget,
        probe_guard=args.probe_guard,
        keep=args.keep,
        pass_env=args.pass_env,
    )
    _dump(report)
    return EXIT_OK if report.get("ok") else EXIT_FAILED


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-bridge",
        description="Approval-gated GitHub-issue-driven local coding-agent bridge "
        "(dry run by default).",
    )
    parser.add_argument("--version", action="version", version=f"agent-bridge {__version__}")
    parser.add_argument(
        "--config", help=f"config file (default: ${CONFIG_ENV} or {DEFAULT_CONFIG})"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("backends", help="show which agent CLIs are installed")
    sub.add_parser("validate-config", help="check the configuration file")

    p = sub.add_parser("validate-task", help="validate an issue body against the task schema")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="issue body saved as a file (offline)")
    src.add_argument("--issue", type=int, help="issue number (reads GitHub)")
    p.add_argument("--title", help="issue title to use with --file")

    p = sub.add_parser("setup-labels", help="create the agent-* labels on GitHub")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("preview", help="dry run: show what an execution would do (default mode)")
    p.add_argument("--issue", type=int)
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--json", action="store_true")

    for name, text in (
        ("run-once", "process approved issues once"),
        ("serve", "poll continuously"),
    ):
        p = sub.add_parser(name, help=text + " (dry run unless --execute)")
        p.add_argument(
            "--execute", action="store_true", help="really run agents (needs dry_run: false)"
        )
        p.add_argument("--yes", action="store_true", help="authorise execution without a prompt")
        if name == "run-once":
            p.add_argument("--issue", type=int, help="only this issue")
            p.add_argument("--show-prompt", action="store_true")

    p = sub.add_parser("status", help="recent runs from the ledger")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("recover", help="resolve runs left behind by a stopped worker")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("cancel", help="ask a running or pending run to stop")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--issue", type=int)

    p = sub.add_parser("smoke", help="run the real agent CLI once in a throwaway repo")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="omp")
    p.add_argument("--model")
    p.add_argument("--binary", help="path of the agent CLI (default: from PATH)")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument(
        "--probe-guard", action="store_true", help="also try to step outside the worktree"
    )
    p.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    p.add_argument(
        "--pass-env",
        action="append",
        metavar="NAME",
        help="environment variable to pass to the agent (default: ANTHROPIC_API_KEY for claude)",
    )
    return parser


COMMANDS: dict[str, Callable[[Any, argparse.Namespace], int]] = {
    "backends": cmd_backends,
    "validate-config": cmd_validate_config,
    "validate-task": cmd_validate_task,
    "setup-labels": cmd_setup_labels,
    "preview": cmd_preview,
    "run-once": cmd_run_once,
    "serve": cmd_serve,
    "status": cmd_status,
    "recover": cmd_recover,
    "cancel": cmd_cancel,
    "smoke": cmd_smoke,
}
NEEDS_NO_CONFIG = {"smoke"}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = None
        if args.command not in NEEDS_NO_CONFIG:
            cfg = load_config(find_config(args.config))
        return COMMANDS[args.command](cfg, args)
    except ConfigError as exc:
        _print(f"configuration error: {exc}")
        return EXIT_CONFIG
    except BridgeError as exc:
        _print(f"error: {exc}")
        return EXIT_FAILED
    except KeyboardInterrupt:
        return 130
