"""systemd/install_services.sh: --root validation and the bounded ownership fix (SEC20-13).

The installer used to take any absolute path as ``--root``, render it into every unit, and the
database unit then ran a recursive chown of it as root at every boot. The review's reproduction:
``--dry-run --root /usr/local`` printed ``ExecStartPre=/bin/chown -R kp:kp /usr/local`` and said
"a real run would SUCCEED".

The real script is run in every test, and never in a way that can reach the system:
``--dry-run`` / ``--destdir``; a refused ``--root`` (the script exits before its privilege check,
so even a call without either flag stops there); or, for the apply phase, inside ``fakeroot`` with
shims for ``install`` and ``systemctl``. fakeroot needs no privilege, so it cannot write /etc; the
apply tests skip themselves when fakeroot is missing or when a system directory the install would
touch is writable by this user.
"""

from __future__ import annotations

import getpass
import grp
import os
import pwd
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "systemd" / "install_services.sh"
STRONG = "correct horse battery staple"
ME = getpass.getuser()
MARKER = "backend/src/main.py"
UNTOUCHED = ["backend", "backend/src", MARKER]  # tree() of a bare checkout
STATE_DIRS = ("data", "logs", "config")

# The installer refuses these trees (and everything below them) as a root. A temp dir under one
# (TMPDIR=/dev/shm or /run/user/N) cannot hold a checkout the installer accepts.
REFUSED_TREES = (
    *("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32"),
    *("/proc", "/root", "/run", "/sbin", "/sys"),
)

# The installer refuses root as the service account (getpass.getuser() would be root), and a bug
# must never be able to run it as root against the real /etc.
pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="installer tests run as a normal user")


def run_installer(
    *args: str,
    credential: str | None = STRONG,
    script: Path = SCRIPT,
) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LAWN_BERRY_OPERATOR_CREDENTIAL", "LAWN_BERRY_AUTH_SECRET", "SUDO_USER"}
    }
    if credential is not None:
        env["LAWN_BERRY_OPERATOR_CREDENTIAL"] = credential
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=120,
    )


def make_checkout(root: Path) -> Path:
    """A directory the installer accepts as --root: it holds the marker file."""
    real = os.path.realpath(root)
    if any(real == tree or real.startswith(tree + "/") for tree in REFUSED_TREES):
        pytest.skip(f"{real} is under a tree the installer refuses as a root: set TMPDIR elsewhere")
    marker = root / MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return root


def tree(root: Path) -> list[str]:
    """Everything under ``root``, relative: proves a refused or dry run changed nothing."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def unit_dir(dest: Path) -> Path:
    return dest / "etc" / "systemd" / "system"


def service_group() -> str:
    return grp.getgrgid(pwd.getpwnam(ME).pw_gid).gr_name


MODES = ("dry-run", "destdir", "real")


def mode_args(mode: str, tmp_path: Path) -> list[str]:
    """The flags that select a mode. "real" is no flag at all: it is only ever used with a root
    the installer refuses (exit 2 before the privilege check) or as a plain user (exit 1)."""
    return {
        "dry-run": ["--dry-run"],
        "destdir": ["--destdir", str(tmp_path / "dest")],
        "real": [],
    }[mode]


# ---- roots the installer refuses ---------------------------------------------------------

SYSTEM_ROOTS = [
    # the system directories themselves
    *("/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64", "/libx32"),
    *("/media", "/mnt", "/opt", "/proc", "/root", "/run", "/sbin", "/srv", "/sys", "/tmp"),
    *("/usr", "/var"),
    # anything directly under /usr and /var
    *("/usr/local", "/usr/share", "/usr/lib", "/usr/bin", "/var/lib", "/var/log", "/var/tmp"),
    # below a tree the operating system owns
    *("/etc/systemd", "/boot/firmware", "/dev/shm", "/proc/self", "/sys/kernel", "/run/lock"),
    *("/lib/systemd", "/root/lawnberry-pi"),
    # the same directories spelled differently
    *("//usr/local", "/usr/./local", "/usr/local/", "/usr//local//", "/etc/", "/var/lib/."),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("root", SYSTEM_ROOTS)
def test_a_system_directory_is_refused_as_the_root_in_every_mode(root, mode, tmp_path):
    result = run_installer(*mode_args(mode, tmp_path), "--user", ME, "--root", root)
    assert result.returncode == 2, result.stdout + result.stderr
    err = result.stderr
    assert f"--root '{root}' is a system directory" in err
    # refused by name, not because it happens to lack the marker or to be missing
    assert "does not exist" not in err and "is not a LawnBerry checkout" not in err
    assert "backend/src/main.py" in err and "--root /apps/lawnberry-pi" in err, "say what to pass"
    assert "################" not in result.stdout, "nothing may be rendered"
    assert not (tmp_path / "dest").exists()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "root", ["/opt/my mower", "relative/dir", "/opt/$(id)", "/opt/../etc", "/tmp/a;b"]
)
def test_the_character_rule_stays_and_is_a_usage_error(root, mode, tmp_path):
    result = run_installer(*mode_args(mode, tmp_path), "--user", ME, "--root", root)
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"--root '{root}' must be an absolute path made of letters, digits and" in result.stderr
    assert "################" not in result.stdout and not (tmp_path / "dest").exists()


def test_the_reviewed_reproduction_is_refused(tmp_path):
    """The review's repro: --dry-run --root /usr/local rendered a root-run recursive chown of
    /usr/local and said "a real run would SUCCEED" (exit 0)."""
    cred = tmp_path / "cred"
    cred.write_text(STRONG + "\n")
    cred.chmod(0o600)
    result = run_installer(
        *("--dry-run", "--user", ME, "--root", "/usr/local"),
        *("--operator-credential-file", str(cred)),
        credential=None,
    )
    assert result.returncode == 2
    assert "would SUCCEED" not in result.stdout and "chown -R" not in result.stdout


LINK_TARGETS = ["/usr/local", "/etc", "/usr", "/", "/var/lib", "/home", "/opt", "/etc/systemd"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", LINK_TARGETS)
def test_a_symlink_that_resolves_to_a_system_directory_is_refused(target, mode, tmp_path):
    link = tmp_path / "checkout"
    link.symlink_to(target)
    result = run_installer(*mode_args(mode, tmp_path), "--user", ME, "--root", str(link))
    assert result.returncode == 2, result.stdout + result.stderr
    real = os.path.realpath(target)
    assert f"--root '{link}' resolves to '{real}', a system directory" in result.stderr
    assert "################" not in result.stdout


def test_chains_and_links_in_the_middle_of_the_path_are_followed(tmp_path):
    first = tmp_path / "first"
    first.symlink_to("/etc")
    second = tmp_path / "second"
    second.symlink_to(first)
    via = tmp_path / "via"
    via.symlink_to("/")
    for given, real in ((second, "/etc"), (via / "usr" / "local", "/usr/local")):
        result = run_installer("--dry-run", "--user", ME, "--root", str(given))
        assert result.returncode == 2, result.stdout + result.stderr
        assert f"resolves to '{real}', a system directory" in result.stderr


def _missing(tmp: Path) -> Path:
    return tmp / "nope"


def _a_file(tmp: Path) -> Path:
    path = tmp / "file"
    path.write_text("x")
    return path


def _without_the_marker(tmp: Path) -> Path:
    path = tmp / "empty"
    path.mkdir()
    return path


def _marker_is_a_directory(tmp: Path) -> Path:
    path = tmp / "odd"
    (path / MARKER).mkdir(parents=True)
    return path


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("build", "message"),
    [
        (_missing, "does not exist"),
        (_a_file, "is a file, not a directory"),
        (_without_the_marker, "is not a LawnBerry checkout"),
        (_marker_is_a_directory, "is not a LawnBerry checkout"),
    ],
)
def test_a_root_that_is_not_a_checkout_directory_is_refused(build, message, mode, tmp_path):
    root = build(tmp_path)
    result = run_installer(*mode_args(mode, tmp_path), "--user", ME, "--root", str(root))
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"--root '{root}' " in result.stderr and message in result.stderr
    assert "backend/src/main.py" in result.stderr, "say what a checkout looks like"
    assert "################" not in result.stdout and not (tmp_path / "dest").exists()


@pytest.mark.parametrize("name", ["a b", "a&b", "a|b", "a$b", "a;b", "a'b"])
def test_a_link_to_a_path_the_units_cannot_carry_is_refused(name, tmp_path):
    """The path that goes into the units is the resolved one, so it gets the same character
    rule as the one typed: a clean-looking link must not smuggle a space or & into sed."""
    real = make_checkout(tmp_path / name / "app")
    link = tmp_path / "clean-link"
    link.symlink_to(real)
    result = run_installer("--dry-run", "--user", ME, "--root", str(link))
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"resolves to '{real}'" in result.stderr and "characters other than" in result.stderr
    assert "################" not in result.stdout


# ---- roots the installer accepts ---------------------------------------------------------

LEGITIMATE = [
    "/apps/lawnberry-pi",
    "/opt/lawnberry-pi",
    "/srv/lawnberry",
    f"/home/{ME}/lawnberry-pi",
    "/usr/local/lawnberry-pi",
    "/var/lib/lawnberry-pi",
    "/mnt/ssd/lawnberry-pi",
    "/media/usb/lawnberry-pi",
    "/tmp/lawnberry-pi",
]


@pytest.mark.parametrize("root", LEGITIMATE)
def test_a_checkout_below_a_system_container_is_never_called_a_system_directory(root):
    """These do not exist here (or are not checkouts), so the run stops on that: the point is
    that the system-directory rules do not claim them."""
    result = run_installer("--dry-run", "--user", ME, "--root", root)
    assert "system directory" not in result.stderr
    if result.returncode != 0:
        assert result.returncode == 2
        assert "does not exist" in result.stderr or "is not a LawnBerry checkout" in result.stderr


def _given(how: str, real: Path, tmp: Path) -> str:
    if how == "plain":
        return str(real)
    if how == "trailing slashes":
        return f"{real}//"
    if how == "dot component":
        return f"{real.parent}/./{real.name}"
    if how == "symlink":
        link = tmp / "link"
        link.symlink_to(real)
        return str(link)
    if how == "symlink chain":
        first, second = tmp / "first", tmp / "second"
        first.symlink_to(real)
        second.symlink_to(first)
        return str(second)
    if how == "symlink in the middle":
        parent = tmp / "parent"
        parent.symlink_to(real.parent)
        return f"{parent}/{real.name}"
    raise AssertionError(how)


HOW = [
    "plain",
    "trailing slashes",
    "dot component",
    "symlink",
    "symlink chain",
    "symlink in the middle",
]


@pytest.mark.parametrize("mode", ["dry-run", "destdir"])
@pytest.mark.parametrize("how", HOW)
def test_a_checkout_is_accepted_and_the_units_carry_its_real_path(how, mode, tmp_path):
    real = make_checkout(tmp_path / "real" / "app").resolve()
    given = _given(how, real, tmp_path)
    result = run_installer(*mode_args(mode, tmp_path), "--user", ME, "--root", given)
    assert result.returncode == 0, result.stdout + result.stderr

    differs = given.rstrip("/") != str(real)
    if differs:
        assert f"NOTE: --root '{given}' resolves to '{real}'" in result.stderr
    else:
        assert "NOTE" not in result.stderr, "a trailing slash is not worth a note"

    if mode == "dry-run":
        assert f"application root: {real}" in result.stdout
        assert f"WorkingDirectory={real}" in result.stdout
        units = result.stdout
    else:
        units = "".join(p.read_text() for p in unit_dir(tmp_path / "dest").iterdir())
        assert (real / ".env").is_file(), "the env file belongs in the real root"
    assert f"ReadWritePaths={real}" in units and f"EnvironmentFile=-{real}/.env" in units
    if how.startswith("symlink") or how == "dot component":
        assert given not in units, "the typed path must not reach a unit"


def test_the_default_root_is_this_checkout():
    if any(str(REPO) == t or str(REPO).startswith(t + "/") for t in REFUSED_TREES):
        pytest.skip(f"this checkout is under {REPO}, a tree the installer refuses")
    result = run_installer("--dry-run", "--user", ME)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"application root: {REPO}" in result.stdout


def test_help_states_the_rules_and_exit_codes():
    out = run_installer("--help").stdout
    assert "backend/src/main.py" in out and "system directory" in out
    assert "2 usage (including a --root" in out and "3 refused" in out


# ---- no unit may recurse as root ---------------------------------------------------------

_RECURSIVE_OWNERSHIP = re.compile(
    r"\b(?:chown|chgrp|chmod)\b[^\n;&|]*?\s(?:-[A-Za-z]*R[A-Za-z]*|--recursive)\b"
)

OLD_DATABASE_UNIT_LINES = """\
ExecStartPre=/bin/mkdir -p /apps/lawnberry-pi/config
PermissionsStartOnly=true
ExecStartPre=/bin/chown -R kp:kp /apps/lawnberry-pi
"""


def exec_commands(text: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", text)  # systemd line continuations
    return [m.group(0) for m in re.finditer(r"^Exec\w+=.*$", joined, re.MULTILINE)]


def root_recursion_problems(text: str) -> list[str]:
    """What a unit must not contain: PermissionsStartOnly (it ran every pre-start command as
    root) and any recursive chown/chgrp/chmod in an Exec line."""
    problems = [
        line.strip() for line in text.splitlines() if re.match(r"\s*PermissionsStartOnly\s*=", line)
    ]
    problems += [c for c in exec_commands(text) if _RECURSIVE_OWNERSHIP.search(c)]
    return problems


@pytest.mark.parametrize(
    "text",
    [
        OLD_DATABASE_UNIT_LINES,
        "ExecStartPre=/bin/chown -R kp:kp /x\n",
        "ExecStartPre=/usr/bin/chown -hR kp:kp /x\n",
        "ExecStartPre=/bin/chown --recursive kp:kp /x\n",
        "ExecStart=/bin/bash -lc 'mkdir -p d && chown -R kp:kp d'\n",
        "ExecStartPre=/bin/chgrp -R kp /x\n",
        "ExecStartPre=/bin/chmod -R u+w /x\n",
        "ExecStartPre=+/bin/chown -vR kp:kp /x\n",
        "ExecStartPre=/bin/chown \\\n    -R kp:kp /x\n",
        "PermissionsStartOnly=true\n",
    ],
)
def test_the_detector_flags_root_recursion(text):
    assert root_recursion_problems(text)


@pytest.mark.parametrize(
    "text",
    [
        "ExecStartPre=/bin/mkdir -p /x/data\n",
        "ExecStartPre=/bin/chown kp:kp /x/file\n",
        "ExecStart=/bin/chmod 0755 /x\n",
        "# no recursive chown of /x at boot\n",
        "ExecStart=/x/.venv/bin/uvicorn backend.src.main:app --port 8081\n",
    ],
)
def test_the_detector_ignores_harmless_lines(text):
    assert root_recursion_problems(text) == []


def install_for_real_into_destdir(tmp_path: Path) -> tuple[Path, Path]:
    root = make_checkout(tmp_path / "app")
    dest = tmp_path / "dest"
    result = run_installer("--destdir", str(dest), "--root", str(root), "--user", ME)
    assert result.returncode == 0, result.stdout + result.stderr
    return root, dest


def test_no_installed_unit_runs_a_recursive_ownership_change_as_root(tmp_path):
    _, dest = install_for_real_into_destdir(tmp_path)
    files = sorted(p for p in unit_dir(dest).rglob("*") if p.is_file())
    assert len(files) >= 12, "every unit and drop-in is looked at"
    for path in files:
        text = path.read_text()
        assert root_recursion_problems(text) == [], path.name
        assert "chown -R" not in text and "PermissionsStartOnly" not in text, path.name


def test_the_database_unit_runs_every_line_as_the_service_account(tmp_path):
    root, dest = install_for_real_into_destdir(tmp_path)
    text = (unit_dir(dest) / "lawnberry-database.service").read_text()
    assert re.search(rf"^User={ME}$", text, re.MULTILINE)
    commands = exec_commands(text)
    privileged = [c for c in commands if re.match(r"Exec\w+=[-@:]*[+!]", c)]
    assert privileged == [], "a + or ! prefix would run that line outside User="
    assert "PermissionsStartOnly" not in text, "it ran every pre-start command as root"
    assert [c for c in commands if re.search(r"\bchown\b", c)] == [], "no ownership change at boot"
    # what the unit is for still works: it creates its directories and initialises the database
    for name in STATE_DIRS:
        assert f"ExecStartPre=/bin/mkdir -p {root}/{name}" in commands
    assert [c for c in commands if c.startswith("ExecStart=") and "init_database.py" in c]
    assert "ProtectSystem=strict" in text and f"ReadWritePaths={root}" in text


# ---- the dry run shows the ownership step and writes nothing -----------------------------


def test_dry_run_prints_what_the_ownership_step_would_do(tmp_path):
    root = make_checkout(tmp_path / "app")
    (root / "logs").mkdir()
    before = tree(root)
    result = run_installer("--dry-run", "--user", ME, "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert f"would create {root}/data (0755" in out and f"would create {root}/config (0755" in out
    assert f"would create {root}/logs" not in out, "it exists already"
    prefix = f"cd {root} && "
    lines = [line for line in out.splitlines() if line.startswith(prefix)]
    assert len(lines) == 3, "one bounded command per state directory, nothing above it"
    group = service_group()
    # This is the very argv the apply phase executes (one array feeds both).
    for name, line in zip(STATE_DIRS, lines, strict=True):
        argv = shlex.split(line.removeprefix(prefix))
        env_path = "PATH=/usr/sbin:/usr/bin:/sbin:/bin"
        assert argv[:5] == ["/usr/bin/env", env_path, "timeout", "600", "find"]
        assert argv[5:10] == ["-P", f"./{name}", "-xdev", "-uid", "0"]
        i = argv.index("-execdir")
        assert argv[i : i + 7] == ["-execdir", "chown", "-h", "--", f"{ME}:{group}", "{}", "+"]
        assert not {"-L", "-H", "-follow", "-R", "-exec"} & set(argv)
    assert tree(root) == before, "--dry-run must write nothing"


def test_dry_run_reports_a_symlinked_state_directory_as_a_failure(tmp_path):
    root = make_checkout(tmp_path / "app")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "config").symlink_to(elsewhere)
    before = tree(root)
    # no credential either: a real run stops on the link first, before the credential gate (3)
    result = run_installer("--dry-run", "--user", ME, "--root", str(root), credential=None)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"would REFUSE: {root}/config is a symbolic link (to {elsewhere})" in result.stdout
    assert "a real run would FAIL (exit 1)" in result.stdout
    assert tree(root) == before


def test_a_link_target_with_control_characters_is_not_printed_raw(tmp_path):
    """The target is whatever the owner of the tree made it: no escape sequence may reach the
    terminal of the administrator who runs the installer."""
    root = make_checkout(tmp_path / "app")
    (root / "config").symlink_to("\x1b[2Jtarget")
    result = run_installer("--dry-run", "--user", ME, "--root", str(root))
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"would REFUSE: {root}/config is a symbolic link" in result.stdout
    assert "\x1b" not in result.stdout + result.stderr


def test_dry_run_with_destdir_plans_no_ownership_change(tmp_path):
    root = make_checkout(tmp_path / "app")
    (root / "data").symlink_to(tmp_path)  # would be refused in a real run; --destdir ignores it
    dest = tmp_path / "dest"
    result = run_installer("--dry-run", "--destdir", str(dest), "--user", ME, "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "would not create or change ownership of data logs config" in result.stdout
    assert "find -P" not in result.stdout and "would create" not in result.stdout


def test_a_destdir_install_leaves_the_state_directories_alone(tmp_path):
    root = make_checkout(tmp_path / "app")
    (root / "data").symlink_to(tmp_path)
    dest = tmp_path / "dest"
    result = run_installer("--destdir", str(dest), "--user", ME, "--root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not creating or changing ownership of data logs config" in result.stdout
    assert not (root / "logs").exists() and not (root / "config").exists()


# ---- the apply phase, for real, in fakeroot ----------------------------------------------

FAKEROOT = shutil.which("fakeroot")
APPLY_TOUCHES = ("/etc/systemd/system", "/etc/systemd/system.conf.d", "/usr/local/sbin")


def _apply_phase_unavailable() -> str | None:
    """Why the apply phase cannot be exercised here, or None when it can."""
    if FAKEROOT is None:
        return "fakeroot is not installed"
    probe = 'f=$(mktemp) && chown 4242:4242 "$f" && stat -c %u "$f"; rm -f "$f"'
    try:
        done = subprocess.run(
            [FAKEROOT, "--", "sh", "-c", probe], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"fakeroot does not run here: {exc}"
    if done.stdout.strip() != "4242":
        return "fakeroot does not fake ownership here"
    for path in APPLY_TOUCHES:
        if not os.path.isdir(path):
            return f"{path} does not exist and a real-mode install would try to create it"
        if os.access(path, os.W_OK):
            return f"{path} is writable by this user: not running a real-mode install against it"
    if os.path.lexists("/etc/systemd/system/lawnberry-sensors.service"):
        return "a retired lawnberry unit is installed here and a real-mode run would delete it"
    return None


@pytest.fixture(scope="module")
def apply_phase():
    reason = _apply_phase_unavailable()
    if reason:
        pytest.skip(reason)


@dataclass
class Applied:
    returncodes: list[int]
    stdout: list[str]
    stderr: list[str]
    owners: dict[str, tuple[int, int, int]]  # path relative to tmp -> (uid, gid, mode)


def apply_under_fakeroot(
    tmp_path: Path,
    root: Path,
    *,
    owned_by_others: dict[Path, tuple[int, int]] | None = None,
    probes: list[Path] = (),
    runs: int = 1,
    while_provisioning: str | None = None,
) -> Applied:
    """Run the installer for REAL (no --dry-run, no --destdir) as fakeroot's uid 0.

    fakeroot fakes ownership only: every file this user owns looks root-owned inside it, and
    chown records the new owner without touching the disk, so the real find and chown run and
    the result is visible with stat in the same session. install and systemctl are shims: the
    install step would write /etc/systemd/system, which this user cannot (and must not) do.

    ``while_provisioning`` is shell that runs once, as the owner of the tree could run it, in the
    window between the installer's pre-flight check and its ownership step. The installer calls
    python3 there (the credential gate), so a python3 shim that runs it and then execs the real
    python3 stands in for a race the owner of the tree would have to win.
    """
    q = shlex.quote
    shims = tmp_path / "shims"
    shims.mkdir()
    for name in ("install", "systemctl"):
        (shims / name).write_text("#!/bin/sh\nexit 0\n")
        (shims / name).chmod(0o755)
    if while_provisioning:
        once = tmp_path / "raced"
        (shims / "python3").write_text(
            f"#!/bin/sh\nif [ ! -e {q(str(once))} ]; then\n  : > {q(str(once))}\n"
            f'  {while_provisioning}\nfi\nexec {q(shutil.which("python3"))} "$@"\n'
        )
        (shims / "python3").chmod(0o755)
    cred = tmp_path / "cred"
    cred.write_text(STRONG + "\n")
    cred.chmod(0o600)

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LAWN_BERRY_OPERATOR_CREDENTIAL", "LAWN_BERRY_AUTH_SECRET", "SUDO_USER"}
    }
    # An empty PATH entry makes find refuse -execdir; the installer has to cope with that.
    env["PATH"] = f"{shims}::{env.get('PATH', '/usr/bin:/bin')}"

    lines = ["set -u", "umask 077"]  # umask 077: a mkdir that ignores its mode would show
    for path, (uid, gid) in (owned_by_others or {}).items():
        lines.append(f"chown {uid}:{gid} {q(str(path))}")
    for n in range(runs):
        lines.append(
            f"bash {q(str(SCRIPT))} --user {ME} --root {q(str(root))} "
            f"--operator-credential-file {q(str(cred))} "
            f"> {q(str(tmp_path / f'run{n}.out'))} 2> {q(str(tmp_path / f'run{n}.err'))}"
        )
        lines.append(f"echo $? >> {q(str(tmp_path / 'rc'))}")
    paths = " ".join(q(str(p)) for p in probes)
    lines.append(f"stat -c '%n|%u|%g|%a' {paths} > {q(str(tmp_path / 'stat'))} 2>/dev/null")
    lines.append("exit 0")
    done = subprocess.run(
        [FAKEROOT, "--", "bash", "-c", "\n".join(lines)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr

    owners = {}
    stat_file = tmp_path / "stat"
    for line in stat_file.read_text().splitlines() if stat_file.exists() else []:
        name, uid, gid, mode = line.split("|")
        owners[str(Path(name).relative_to(tmp_path))] = (int(uid), int(gid), int(mode, 8))
    return Applied(
        returncodes=[int(x) for x in (tmp_path / "rc").read_text().split()],
        stdout=[(tmp_path / f"run{n}.out").read_text() for n in range(runs)],
        stderr=[(tmp_path / f"run{n}.err").read_text() for n in range(runs)],
        owners=owners,
    )


def build_scenario(tmp_path: Path) -> tuple[Path, Path]:
    """A checkout whose state directories hold what a long-lived install collects, plus traps
    that lead outside them: a symbolic link and a hard link to files that are not theirs."""
    root = make_checkout(tmp_path / "app")
    outside = tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (outside / "victim").write_text("x")
    (outside / "dir" / "inner").write_text("x")
    data = root / "data"
    (data / "sub").mkdir(parents=True)
    for rel in ("db.sqlite", "sub/deep.log", "theirs.txt"):
        (data / rel).write_text("x")
    (data / "escape").symlink_to(outside / "dir")
    os.link(outside / "victim", data / "hard")
    (root / "config").mkdir()
    (root / "config" / "app.yaml").write_text("x")
    (root / "other").mkdir()
    (root / "other" / "file").write_text("x")
    return root, outside  # logs is missing on purpose


def test_the_apply_phase_hands_back_what_root_owns_in_the_state_dirs_and_nothing_else(
    apply_phase, tmp_path
):
    root, outside = build_scenario(tmp_path)
    probes = [
        *(root, root / "other", root / "other" / "file"),
        *(root / "data", root / "data" / "db.sqlite", root / "data" / "sub"),
        *(root / "data" / "sub" / "deep.log", root / "data" / "theirs.txt"),
        *(root / "data" / "escape", root / "data" / "hard", root / "logs"),
        *(root / "config", root / "config" / "app.yaml"),
        *(outside, outside / "victim", outside / "dir", outside / "dir" / "inner"),
    ]
    applied = apply_under_fakeroot(
        tmp_path,
        root,
        owned_by_others={root / "data" / "theirs.txt": (4242, 4242)},
        probes=probes,
        runs=2,
    )
    assert applied.returncodes == [0, 0], applied.stdout[0] + applied.stderr[0]
    own = applied.owners
    mine = (pwd.getpwnam(ME).pw_uid, pwd.getpwnam(ME).pw_gid)

    handed_over = [
        *("app/data", "app/data/db.sqlite", "app/data/sub", "app/data/sub/deep.log"),
        *("app/config", "app/config/app.yaml", "app/logs"),
        "app/data/escape",  # the link itself (chown -h)
    ]
    for rel in handed_over:
        assert own[rel][:2] == mine, f"{rel} is root-owned and must go to {ME}: {own[rel]}"
    assert own["app/data/theirs.txt"][:2] == (4242, 4242), "someone else's file is left alone"
    left_alone = [
        *("app", "app/other", "app/other/file"),  # the root and its other children
        *("outside", "outside/victim", "outside/dir", "outside/dir/inner"),  # beyond the links
        "app/data/hard",  # root-owned but a second name for outside/victim: skipped
    ]
    for rel in left_alone:
        assert own[rel][:2] == (0, 0), f"{rel} must not be touched: {own[rel]}"

    assert own["app/logs"][2] == 0o755, "a missing state directory is created 0755"
    assert "Creating" in applied.stdout[0] and "app/logs" in applied.stdout[0]
    assert "Creating" not in applied.stdout[1], "the second run finds nothing to create"
    for err in applied.stderr:
        assert "left ./data/hard as it is" in err, "the skipped hard link is reported"
    assert (root / ".env").is_file()


def test_the_apply_phase_refuses_a_symlinked_state_directory_before_changing_anything(
    apply_phase, tmp_path
):
    root, outside = build_scenario(tmp_path)
    shutil.rmtree(root / "data")
    (root / "data").symlink_to(outside)
    before = tree(root)
    applied = apply_under_fakeroot(
        tmp_path, root, probes=[root, outside, outside / "victim", root / "config"]
    )
    assert applied.returncodes == [1], applied.stdout[0] + applied.stderr[0]
    assert f"ERROR: {root}/data is a symbolic link (to {outside})" in applied.stderr[0]
    assert "will not follow it" in applied.stderr[0]
    assert "Installing" not in applied.stdout[0], "refused before the install began"
    assert tree(root) == before, "no .env, no new directory: the run changed nothing"
    own = applied.owners
    assert own["outside"][:2] == (0, 0) and own["outside/victim"][:2] == (0, 0)
    assert own["app/config"][:2] == (0, 0), "the other state directories were not touched either"


def test_the_apply_phase_refuses_a_state_path_that_is_a_file(apply_phase, tmp_path):
    root, _ = build_scenario(tmp_path)
    shutil.rmtree(root / "config")
    (root / "config").write_text("not a directory")
    applied = apply_under_fakeroot(tmp_path, root, probes=[root / "data"])
    assert applied.returncodes == [1], applied.stdout[0] + applied.stderr[0]
    assert f"ERROR: {root}/config exists but is not a directory" in applied.stderr[0]
    assert not (root / ".env").exists()
    assert applied.owners["app/data"][:2] == (0, 0)


def test_a_state_directory_swapped_for_a_link_after_the_pre_flight_is_still_refused(
    apply_phase, tmp_path
):
    """The owner of the tree can win a race: the directory is real when the installer first
    looks and a link by the time it works on it."""
    root, outside = build_scenario(tmp_path)
    q = shlex.quote
    swap = f"rm -rf {q(str(root / 'data'))} && ln -s {q(str(outside))} {q(str(root / 'data'))}"
    applied = apply_under_fakeroot(
        tmp_path,
        root,
        while_provisioning=swap,
        probes=[outside, outside / "victim", outside / "dir", outside / "dir" / "inner"],
    )
    assert applied.returncodes == [1], applied.stdout[0] + applied.stderr[0]
    assert f"ERROR: {root}/data is a symbolic link (to {outside})" in applied.stderr[0]
    assert not (root / "logs").exists(), "it stopped at the first state directory"
    for rel in ("outside", "outside/victim", "outside/dir", "outside/dir/inner"):
        assert applied.owners[rel][:2] == (0, 0), f"{rel} is beyond the link and must not change"


def test_a_root_replaced_after_the_validation_is_not_followed(apply_phase, tmp_path):
    """The path the installer validated now leads to another checkout: it must notice that
    the directory it stands in is not the one it checked, and touch nothing in either."""
    root, _ = build_scenario(tmp_path)
    decoy = make_checkout(tmp_path / "decoy")
    (decoy / "data").mkdir()
    (decoy / "data" / "file").write_text("x")
    moved = tmp_path / "app-moved"
    q = shlex.quote
    swap = f"mv {q(str(root))} {q(str(moved))} && ln -s {q(str(decoy))} {q(str(root))}"
    applied = apply_under_fakeroot(
        tmp_path,
        root,
        while_provisioning=swap,
        probes=[
            decoy / "data",
            decoy / "data" / "file",
            moved / "data",
            moved / "data" / "db.sqlite",
        ],
    )
    assert applied.returncodes == [1], applied.stdout[0] + applied.stderr[0]
    assert (
        f"the application root '{root}' changed while the installer was running"
        in (applied.stderr[0])
    )
    for rel in ("decoy/data", "decoy/data/file", "app-moved/data", "app-moved/data/db.sqlite"):
        assert applied.owners[rel][:2] == (0, 0), f"{rel} must not be touched"


def test_a_hostile_file_name_in_the_hard_link_warning_is_not_printed_raw(apply_phase, tmp_path):
    root, outside = build_scenario(tmp_path)
    os.link(outside / "victim", root / "data" / "evil\x1b[2Jname")
    applied = apply_under_fakeroot(tmp_path, root)
    assert applied.returncodes == [0], applied.stdout[0] + applied.stderr[0]
    assert "left ./data/evil^[[2Jname as it is" in applied.stderr[0]
    assert "\x1b" not in applied.stderr[0]
