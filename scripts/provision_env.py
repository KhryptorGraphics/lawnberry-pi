#!/usr/bin/env python3
"""Provision the runtime env file the LawnBerry units load (``<root>/.env``).

Run by ``systemd/install_services.sh``; usable on its own. It

* generates the persistent ``LAWN_BERRY_AUTH_SECRET`` (the JWT signing key) once and keeps it
  across re-runs, so operator sessions survive a backend restart (without it the backend
  signs with a random per-process key);
* validates the operator credential (``LAWN_BERRY_OPERATOR_CREDENTIAL``) and REFUSES, exit
  status 3 with nothing written, when none is configured or it is empty, a published default
  or weak (``backend/src/core/credential_policy.py``);
* leaves every other line of the file (NTRIP, THOR_*, comments) untouched, writes atomically
  and leaves the file mode 0600.

Where the credential comes from, first match wins: ``--credential-file FILE``, the
``LAWN_BERRY_OPERATOR_CREDENTIAL`` environment variable of this process, the value already in
the env file. A credential taken from the first two is written to the env file (rotation).
It is never accepted on the command line (``ps`` and shell history) and never printed.

Exit status: 0 ok, 1 error, 2 usage, 3 refused by the credential / secret policy.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import re
import secrets
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

CREDENTIAL_KEY = "LAWN_BERRY_OPERATOR_CREDENTIAL"
SECRET_KEY = "LAWN_BERRY_AUTH_SECRET"
AUTH_FLAG_KEY = "OPERATOR_AUTH_REQUIRED"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 3

MANAGED_HEADER = (
    "# Managed by systemd/install_services.sh (scripts/provision_env.py). Keep this file mode 0600."
)

_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _load_policy():
    """Load backend/src/core/credential_policy.py by path (no package import, no deps)."""
    path = Path(__file__).resolve().parents[1] / "backend" / "src" / "core" / "credential_policy.py"
    spec = importlib.util.spec_from_file_location("lawnberry_credential_policy", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise RuntimeError(f"cannot load the credential policy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _load_policy()


def new_secret() -> str:
    return secrets.token_hex(32)


# --------------------------------------------------------------------------------------
# env file parsing and editing
# --------------------------------------------------------------------------------------


def parse_value(raw: str) -> tuple[str | None, str | None]:
    """Value of one ``KEY=raw`` line as systemd's EnvironmentFile reads it.

    Returns ``(value, None)``, or ``(None, problem)`` for the forms this tool refuses to
    interpret (anything but a bare token, a complete ``'literal'`` or a ``"..."`` using only
    the ``\\\\`` and ``\\"`` escapes), so a secret is never validated as something other than
    what the service will see.
    """
    text = raw.strip()
    if text == "":
        return "", None
    quote = text[0]
    if quote == "'":
        if len(text) >= 2 and text.endswith("'") and "'" not in text[1:-1]:
            return text[1:-1], None
        return None, "single-quoted value is not one complete 'literal'"
    if quote == '"':
        if len(text) < 2 or not text.endswith('"'):
            return None, "unterminated double-quoted value"
        inner, out, i = text[1:-1], [], 0
        while i < len(inner):
            ch = inner[i]
            if ch == '"':
                return None, "unescaped double quote inside a double-quoted value"
            if ch == "\\":
                if i + 1 < len(inner) and inner[i + 1] in '\\"':
                    out.append(inner[i + 1])
                    i += 2
                    continue
                return None, 'escape other than \\\\ or \\" in a double-quoted value'
            out.append(ch)
            i += 1
        return "".join(out), None
    if "\\" in text or "'" in text or '"' in text:
        return None, "unquoted value contains a quote or a backslash"
    return text, None


@dataclass
class Entry:
    index: int
    key: str
    value: str | None
    problem: str | None


def parse_entries(lines: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        match = _ASSIGNMENT.match(line)
        if match is None:
            continue
        value, problem = parse_value(match.group(2))
        entries.append(Entry(index, match.group(1), value, problem))
    return entries


def last_entry(entries: list[Entry], key: str) -> Entry | None:
    found = [e for e in entries if e.key == key]
    return found[-1] if found else None


def storage_problems(value: str) -> list[str]:
    """Why ``value`` cannot be written to the env file as a ``'literal'`` unambiguously."""
    problems = []
    if "'" in value:
        problems.append("contains a single quote")
    if "\\" in value:
        problems.append("contains a backslash")
    return problems


def upsert(lines: list[str], assignments: Mapping[str, str]) -> list[str]:
    """``lines`` with each ``KEY=value`` set: last occurrence replaced in place, earlier
    duplicates dropped, missing keys appended under one managed header."""
    entries = parse_entries(lines)
    drop: set[int] = set()
    replace: dict[int, str] = {}
    appended: list[str] = []
    for key, rendered in assignments.items():
        indices = [e.index for e in entries if e.key == key]
        if indices:
            drop.update(indices[:-1])
            replace[indices[-1]] = rendered
        else:
            appended.append(rendered)
    out = [replace.get(i, line) for i, line in enumerate(lines) if i not in drop]
    if appended:
        if out and out[-1].strip():
            out.append("")
        if MANAGED_HEADER not in out:
            out.append(MANAGED_HEADER)
        out.extend(appended)
    return out


def _chown(path: str, owner: str | None, group: str | None, existing: os.stat_result | None):
    if owner is None and group is None:
        if existing is None:
            return
        uid, gid = existing.st_uid, existing.st_gid
    else:
        import grp
        import pwd

        try:
            uid = pwd.getpwnam(owner).pw_uid if owner else -1
            gid = grp.getgrnam(group).gr_gid if group else -1
        except KeyError as exc:
            raise OSError(f"unknown user or group {exc}") from exc
    try:
        os.chown(path, uid, gid)
    except PermissionError:
        # Not root and the target owner is someone else. The unit's EnvironmentFile is read
        # by systemd as root, so the file still works; say so instead of failing.
        print("provision_env: note: not root, env file ownership left unchanged", file=sys.stderr)


def read_lines(path: Path) -> list[str]:
    """Lines of ``path`` split on ``\\n`` only, so no character inside a value is a break."""
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def write_atomic(path: Path, lines: list[str], *, owner: str | None, group: str | None) -> None:
    """Replace ``path`` with ``lines`` (mode 0600, optional owner), all or nothing."""
    target = Path(os.path.realpath(path))
    existing = target.stat() if target.exists() else None
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        _chown(tmp_name, owner, group, existing)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
    with contextlib.suppress(OSError):  # best effort: make the rename itself durable
        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


# --------------------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------------------


@dataclass
class Plan:
    refusals: list[str] = field(default_factory=list)
    credential_refused: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    assignments: dict[str, str] = field(default_factory=dict)
    new_file: bool = False
    tighten_from: int | None = None
    lines: list[str] = field(default_factory=list)

    @property
    def needs_write(self) -> bool:
        return bool(self.assignments) or self.new_file or self.tighten_from is not None


def _read_credential_file(path: Path) -> tuple[str | None, list[str], list[str]]:
    """(value, refusals, warnings) for ``--credential-file``."""
    try:
        raw = path.read_bytes()
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        return None, [f"cannot read the credential file {path}: {exc.strerror or exc}"], []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, [f"the credential file {path} is not valid UTF-8"], []
    for ending in ("\r\n", "\n"):
        if text.endswith(ending):
            text = text[: -len(ending)]
            break
    warnings = []
    if mode & 0o077:
        warnings.append(
            f"the credential file {path} is readable by other users (mode {mode:04o}); "
            "delete it once the install is done"
        )
    return text, [], warnings


def make_plan(
    env_path: Path,
    *,
    credential_file: Path | None,
    environ: Mapping[str, str],
    rotate_secret: bool,
    generate: Callable[[], str] = new_secret,
) -> Plan:
    plan = Plan(new_file=not env_path.exists())
    if not plan.new_file:
        try:
            plan.lines = read_lines(env_path)
            mode = stat.S_IMODE(env_path.stat().st_mode)
        except (OSError, UnicodeDecodeError) as exc:
            plan.refusals.append(f"cannot read {env_path}: {exc}")
            return plan
        if mode != 0o600:
            plan.tighten_from = mode
    entries = parse_entries(plan.lines)

    # ---- operator credential ---------------------------------------------------------
    provided: str | None = None
    source = ""
    if credential_file is not None:
        provided, file_refusals, file_warnings = _read_credential_file(credential_file)
        if file_refusals:
            plan.credential_refused = True
        plan.refusals.extend(file_refusals)
        plan.warnings.extend(file_warnings)
        source = f"the credential file {credential_file}"
    elif environ.get(CREDENTIAL_KEY) is not None:
        provided = environ[CREDENTIAL_KEY]
        source = f"the {CREDENTIAL_KEY} environment variable"

    existing = last_entry(entries, CREDENTIAL_KEY)
    if provided is not None:
        reasons = policy.assess_operator_credential(provided) + [
            f"cannot be stored in the env file ({problem})"
            for problem in storage_problems(provided)
        ]
        if reasons:
            plan.credential_refused = True
            plan.refusals.append(f"operator credential from {source} {'; '.join(reasons)}")
        elif existing is not None and existing.problem is None and existing.value == provided:
            plan.notes.append(f"{CREDENTIAL_KEY}: kept (from {source}, identical to the file's)")
        else:
            plan.assignments[CREDENTIAL_KEY] = f"{CREDENTIAL_KEY}='{provided}'"
            action = "replacing the existing value" if existing else "adding it"
            plan.notes.append(f"{CREDENTIAL_KEY}: from {source}, {action}")
    elif credential_file is None:
        if existing is None:
            plan.credential_refused = True
            plan.refusals.append(f"no operator credential is configured ({CREDENTIAL_KEY} unset)")
        elif existing.problem:
            plan.credential_refused = True
            plan.refusals.append(f"{CREDENTIAL_KEY} in {env_path}: {existing.problem}")
        else:
            reasons = policy.assess_operator_credential(existing.value)
            if reasons:
                plan.credential_refused = True
                plan.refusals.append(f"operator credential in {env_path} {'; '.join(reasons)}")
            else:
                plan.notes.append(f"{CREDENTIAL_KEY}: kept (existing, passes the policy)")

    # ---- signing secret --------------------------------------------------------------
    secret = last_entry(entries, SECRET_KEY)
    if rotate_secret:
        plan.assignments[SECRET_KEY] = f"{SECRET_KEY}={generate()}"
        plan.notes.append(f"{SECRET_KEY}: rotated (existing sessions are invalidated)")
    elif secret is None or secret.value == "":
        plan.assignments[SECRET_KEY] = f"{SECRET_KEY}={generate()}"
        plan.notes.append(f"{SECRET_KEY}: generated (64 hex characters)")
    elif secret.problem:
        plan.refusals.append(
            f"{SECRET_KEY} in {env_path}: {secret.problem}; "
            "fix the line or pass --rotate-auth-secret"
        )
    else:
        reasons = policy.assess_signing_secret(secret.value)
        if reasons:
            plan.refusals.append(
                f"{SECRET_KEY} in {env_path} {'; '.join(reasons)}; "
                "remove the line or pass --rotate-auth-secret"
            )
        else:
            plan.notes.append(
                f"{SECRET_KEY}: kept (existing, {len(secret.value or '')} characters)"
            )

    # ---- a loud line, not a refusal --------------------------------------------------
    flag = last_entry(entries, AUTH_FLAG_KEY)
    if flag is not None and flag.value != "1":
        plan.warnings.append(
            f"{AUTH_FLAG_KEY} is not 1 in {env_path}: it overrides the unit and operator "
            "authentication is DISABLED"
        )
    return plan


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------


def _credential_guidance(env_path: Path) -> list[str]:
    return [
        "Provide an operator credential of at least "
        f"{policy.MIN_CREDENTIAL_LENGTH} characters that is not a default, then re-run:",
        '  f=$(mktemp) && ${EDITOR:-nano} "$f"      # one line: the credential, no quotes',
        '  sudo bash systemd/install_services.sh --operator-credential-file "$f"; shred -u "$f"',
        f"or put {CREDENTIAL_KEY}='...' into {env_path} yourself (mode 0600).",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provision_env.py", description=(__doc__ or "").split("\n\n", 1)[0]
    )
    parser.add_argument("--env-file", required=True, type=Path, help="path of the env file")
    parser.add_argument("--credential-file", type=Path, help="read the operator credential here")
    parser.add_argument("--rotate-auth-secret", action="store_true", help="replace the secret")
    parser.add_argument("--owner", help="chown the env file to this user (root only)")
    parser.add_argument("--group", help="chgrp the env file to this group (root only)")
    parser.add_argument("--check", action="store_true", help="validate and report; write nothing")
    return parser


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_path: Path = args.env_file
    plan = make_plan(
        env_path,
        credential_file=args.credential_file,
        environ=os.environ if environ is None else environ,
        rotate_secret=args.rotate_auth_secret,
    )

    for warning in plan.warnings:
        print(f"provision_env: WARNING: {warning}", file=sys.stderr)
    if plan.refusals:
        for refusal in plan.refusals:
            print(f"provision_env: REFUSED: {refusal}", file=sys.stderr)
        if plan.credential_refused:
            for line in _credential_guidance(env_path):
                print(f"provision_env: {line}", file=sys.stderr)
        return EXIT_REFUSED

    if not args.check and plan.needs_write:
        try:
            write_atomic(
                env_path,
                upsert(plan.lines, plan.assignments),
                owner=args.owner,
                group=args.group,
            )
        except OSError as exc:
            print(f"provision_env: ERROR: cannot write {env_path}: {exc}", file=sys.stderr)
            return EXIT_ERROR

    future = args.check
    if plan.new_file:
        print(
            f"provision_env: {env_path} {'would be created' if future else 'created'} (mode 0600)"
        )
    else:
        print(f"provision_env: {env_path} exists")
    if plan.tighten_from is not None:
        print(
            f"provision_env: mode {plan.tighten_from:04o} {'would be' if future else 'was'} "
            "tightened to 0600 (the file holds secrets)"
        )
    for note in plan.notes:
        print(f"provision_env: {note}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
