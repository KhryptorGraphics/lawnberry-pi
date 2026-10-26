"""systemd/install_services.sh: unit rendering, credential refusal, secret persistence.

The installer is run for real (``bash systemd/install_services.sh``) against temp
directories: ``--dry-run`` for the rendered text, ``--destdir`` for an actual install that
needs no root and never touches /etc. No test runs the installer without one of the two.
"""

from __future__ import annotations

import filecmp
import getpass
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SYSTEMD = REPO / "systemd"
STRONG = "correct horse battery staple"
ME = getpass.getuser()

# Units the installer deliberately does not install: superseded by lawnberry-cert-renewal.*
LEGACY_UNITS = {"lawnberry-acme-renew.service", "lawnberry-acme-renew.timer"}
REPO_UNIT_FILES = sorted(p.name for p in SYSTEMD.iterdir() if p.suffix in {".service", ".timer"})

# The installer refuses root as the service account (getpass.getuser() would be root), and a
# bug in --dry-run/--destdir must never be able to write to the real /etc as root.
pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="installer tests run as a normal user")


def installer(
    *args: str,
    credential: str | None = None,
    repo: Path = REPO,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LAWN_BERRY_OPERATOR_CREDENTIAL", "LAWN_BERRY_AUTH_SECRET", "SUDO_USER"}
    }
    if credential is not None:
        env["LAWN_BERRY_OPERATOR_CREDENTIAL"] = credential
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(repo / "systemd" / "install_services.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo,
        timeout=120,
    )


CHECKOUT_MARKER = "backend/src/main.py"  # the installer requires this file under --root
UNTOUCHED = ["backend", "backend/src", CHECKOUT_MARKER]  # what tree() lists for a bare checkout


def make_checkout(root: Path) -> Path:
    """An application root the installer accepts: a directory holding the marker file."""
    marker = root / CHECKOUT_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return root


def tree(root: Path) -> list[str]:
    """Everything under ``root`` (relative): proves a refused or dry run changed nothing."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.fixture
def site(tmp_path):
    """An app root (a LawnBerry checkout) and a staging root, as the installer sees them."""
    root = tmp_path / "app"
    dest = tmp_path / "dest"
    make_checkout(root)
    return root, dest


def install_args(root: Path, dest: Path, *extra: str) -> list[str]:
    return ["--destdir", str(dest), "--root", str(root), "--user", ME, *extra]


def unit_dir(dest: Path) -> Path:
    return dest / "etc" / "systemd" / "system"


def env_values(env_file: Path) -> dict[str, str]:
    out = {}
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            out[key] = value.strip("'")
    return out


def fake_repo(tmp_path: Path) -> Path:
    """A copy of the files the installer needs, so a test can add or break a unit."""
    fake = tmp_path / "repo"
    shutil.copytree(SYSTEMD, fake / "systemd")
    (fake / "scripts").mkdir()
    for name in ("provision_env.py", "halow_link_watch.sh", "renew_certificates.sh"):
        shutil.copy(REPO / "scripts" / name, fake / "scripts" / name)
    (fake / "backend/src/core").mkdir(parents=True)
    shutil.copy(REPO / "backend/src/core/credential_policy.py", fake / "backend/src/core")
    return fake


# ---- the repository units are templates --------------------------------------------------


def test_repo_units_carry_no_hard_coded_account_or_checkout():
    for name in REPO_UNIT_FILES:
        text = (SYSTEMD / name).read_text()
        for needle in ("User=pi", "Group=pi", "pi:pi", "/apps/lawnberry-pi"):
            assert needle not in text, f"{name} still hard-codes {needle!r}"
        tokens = set(re.findall(r"@[A-Z_]+@", text))
        assert tokens <= {"@LB_USER@", "@LB_GROUP@", "@LB_ROOT@"}, f"{name}: unknown {tokens}"
        for match in re.finditer(r"^(User|Group)=(.*)$", text, re.MULTILINE):
            assert match.group(2) in {"root", "video", "@LB_USER@", "@LB_GROUP@"}, (
                f"{name}: {match.group(0)} is neither root, the video group nor a token"
            )


def test_the_units_that_ran_as_pi_run_as_the_installer_account():
    for name in (
        "lawnberry-backend.service",
        "lawnberry-database.service",
        "lawnberry-frontend.service",
        "lawnberry-health.service",
        "lawnberry-remote-access.service",
        "lawnberry-backup.service",
        "lawnberry-camera.service",
    ):
        assert "User=@LB_USER@" in (SYSTEMD / name).read_text(), name


def _exec_start_commands(text: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", text)  # systemd line continuations
    return [m.group(1) for m in re.finditer(r"^ExecStart=(.*)$", joined, re.MULTILINE)]


def _is_placeholder(command: str) -> bool:
    command = command.lstrip("-@+!: ")
    if re.search(r"sleep\s+infinity", command):
        return True
    inner = re.sub(r"^(?:/usr)?/bin/(?:ba)?sh\s+-l?c\s+['\"]", "", command)
    return re.match(r"\s*(?:/?(?:usr/)?(?:bin/)?)(?:echo|true)\b", inner) is not None


def test_the_placeholder_detector_catches_the_removed_sensors_unit():
    old = "/bin/bash -c 'echo \"Sensor manager service placeholder\" && sleep infinity'"
    assert _is_placeholder(old)
    assert _is_placeholder("/bin/true")
    assert _is_placeholder("/usr/bin/echo ready")
    assert not _is_placeholder("/apps/x/.venv/bin/uvicorn backend.src.main:app --port 8081")
    assert not _is_placeholder(
        "/bin/bash -c 'while true; do curl -f http://x/health; sleep 30; done'"
    )


def test_no_unit_runs_a_placeholder_command():
    seen = 0
    for name in REPO_UNIT_FILES:
        for command in _exec_start_commands((SYSTEMD / name).read_text()):
            seen += 1
            assert not _is_placeholder(command), f"{name}: placeholder ExecStart {command!r}"
    assert seen >= 8
    assert "lawnberry-sensors.service" not in REPO_UNIT_FILES


# ---- --dry-run ---------------------------------------------------------------------------


def test_dry_run_prints_every_unit_rendered_for_the_given_account_and_root(tmp_path):
    root = make_checkout(tmp_path / "mower")
    result = installer(
        "--dry-run", "--user", "alice", "--group", "staff", "--root", str(root), credential=STRONG
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "would SUCCEED" in out
    assert f"WorkingDirectory={root}" in out
    assert f"ExecStart={root}/.venv/bin/uvicorn backend.src.main:app" in out
    assert f"EnvironmentFile=-{root}/.env" in out
    assert "chown -R" not in out and "PermissionsStartOnly" not in out
    assert out.count("User=alice") >= 7 and "Group=staff" in out
    assert "Group=video" in out, "the camera unit keeps its device group"
    for stale in ("@LB_", "User=pi", "Group=pi", "pi:pi", "/apps/lawnberry-pi"):
        assert stale not in out, f"rendered output still contains {stale!r}"
    for name in REPO_UNIT_FILES:
        if name not in LEGACY_UNITS:
            assert f"/etc/systemd/system/{name} (0644 root:root)" in out, name
    assert tree(root) == UNTOUCHED, "--dry-run must write nothing"


def test_dry_run_never_prints_the_credential(tmp_path):
    result = installer(
        "--dry-run", "--user", "alice", "--root", str(make_checkout(tmp_path)), credential=STRONG
    )
    assert STRONG not in result.stdout + result.stderr


def test_dry_run_reports_a_refusal_but_still_shows_the_units(tmp_path):
    root = make_checkout(tmp_path / "m")
    result = installer("--dry-run", "--user", "alice", "--root", str(root))
    assert result.returncode == 3
    assert "would REFUSE" in result.stdout
    assert "REFUSED: no operator credential is configured" in result.stderr
    assert "User=alice" in result.stdout, "the units are still rendered for review"


def test_the_account_defaults_to_the_owner_of_the_root(tmp_path):
    result = installer("--dry-run", "--root", str(make_checkout(tmp_path)), credential=STRONG)
    assert result.returncode == 0, result.stderr
    assert f"service account : {ME}" in result.stdout
    assert f"User={ME}" in result.stdout


def test_sudo_user_beats_the_owner_of_the_root(tmp_path):
    root = make_checkout(tmp_path)
    result = installer(
        "--dry-run", "--root", str(root), credential=STRONG, extra_env={"SUDO_USER": "bob"}
    )
    assert "service account : bob" in result.stdout


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (["--user", "root"], "never root"),
        (["--user", "bad name"], "not a usable account name"),
        (["--user", "alice", "--group", "x;y"], "not a usable group name"),
        (["--user", "alice", "--root", "/opt/my mower"], "--root"),
        (["--user", "alice", "--root", "relative/dir"], "--root"),
        (["--user", "alice", "--root", "/opt/$(id)"], "--root"),
        (["--user", "alice", "--root", "/opt/../etc"], "--root"),
        (["--user", "alice", "--root", "/"], "--root"),
        (["--user"], "needs a value"),
        (["--no-such-flag"], "unknown option"),
    ],
)
def test_unusable_substitution_values_are_rejected_before_anything_runs(args, needle):
    result = installer("--dry-run", *args, credential=STRONG)
    assert result.returncode in (1, 2), result.stdout
    assert needle in result.stderr
    assert "################" not in result.stdout


# ---- a real install into a staging root --------------------------------------------------


def test_destdir_install_renders_modes_and_installs_everything(site):
    root, dest = site
    result = installer(*install_args(root, dest), credential=STRONG)
    assert result.returncode == 0, result.stderr + result.stdout

    for path in unit_dir(dest).iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o644, path.name
        text = path.read_text()
        assert "@LB_" not in text and "/apps/lawnberry-pi" not in text, path.name
    backend = (unit_dir(dest) / "lawnberry-backend.service").read_text()
    assert f"User={ME}" in backend and f"WorkingDirectory={root}" in backend
    assert f"ReadWritePaths={root}" in backend

    script = dest / "usr/local/sbin/halow-link-watch.sh"
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    assert filecmp.cmp(script, REPO / "scripts/halow_link_watch.sh", shallow=False)
    watchdog = dest / "etc/systemd/system.conf.d/10-watchdog.conf"
    assert filecmp.cmp(watchdog, SYSTEMD / "10-watchdog.conf", shallow=False)

    env_file = root / ".env"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    values = env_values(env_file)
    assert re.fullmatch(r"[0-9a-f]{64}", values["LAWN_BERRY_AUTH_SECRET"])
    assert values["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG
    assert STRONG not in result.stdout + result.stderr
    assert values["LAWN_BERRY_AUTH_SECRET"] not in result.stdout + result.stderr
    assert "not running systemctl" in result.stdout


def test_every_unit_file_in_the_repo_is_installed_or_deliberately_legacy(site):
    """A new systemd/*.service|timer must be added to the installer lists (a conscious
    choice); a deleted one must leave them. Both directions fail here."""
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    installed = {p.name for p in unit_dir(dest).iterdir()}
    assert installed.isdisjoint(LEGACY_UNITS)
    assert installed | LEGACY_UNITS == set(REPO_UNIT_FILES)


def test_installed_units_have_valid_systemd_syntax(site):
    if shutil.which("systemd-analyze") is None:
        pytest.skip("systemd-analyze not installed")
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    paths = sorted(str(p) for p in unit_dir(dest).iterdir() if p.is_file())
    verify = subprocess.run(
        ["systemd-analyze", "verify", *paths], capture_output=True, text=True, timeout=120
    )
    # Other units on the machine may complain; only these units' lines count, and the on-device
    # paths (.venv/bin/..., /usr/local/sbin/...) legitimately do not exist here.
    mine = [
        line
        for line in (verify.stdout + verify.stderr).splitlines()
        if str(dest) in line or any(Path(p).name in line for p in paths)
    ]
    genuine = [line for line in mine if not re.search(r"is not executable|No such file", line)]
    assert genuine == []


def test_the_removed_sensors_placeholder_is_cleaned_up_on_upgrade(site):
    root, dest = site
    stale = unit_dir(dest) / "lawnberry-sensors.service"
    stale.parent.mkdir(parents=True)
    stale.write_text("[Service]\nExecStart=/bin/bash -c 'sleep infinity'\n")
    result = installer(*install_args(root, dest), credential=STRONG)
    assert result.returncode == 0, result.stderr
    assert not stale.exists()
    assert "lawnberry-sensors" not in result.stdout.split("To start them")[1]


def test_drop_ins_next_to_a_unit_are_rendered_and_installed(tmp_path):
    """The live Pi has a lawnberry-backend.service.d/thor-upload.conf drop-in; the installer
    must carry such a file once it is added to the repository."""
    fake = fake_repo(tmp_path)
    dropin = fake / "systemd/lawnberry-backend.service.d/10-test.conf"
    dropin.parent.mkdir()
    dropin.write_text("[Service]\nEnvironment=EXTRA_DIR=@LB_ROOT@/extra\nUser=@LB_USER@\n")

    root, dest = tmp_path / "app", tmp_path / "dest"
    make_checkout(root)
    result = installer(*install_args(root, dest), credential=STRONG, repo=fake)
    assert result.returncode == 0, result.stderr
    out = unit_dir(dest) / "lawnberry-backend.service.d" / "10-test.conf"
    assert out.read_text() == f"[Service]\nEnvironment=EXTRA_DIR={root}/extra\nUser={ME}\n"
    assert stat.S_IMODE(out.stat().st_mode) == 0o644


def test_an_unknown_token_in_a_unit_aborts_the_install(tmp_path):
    fake = fake_repo(tmp_path)
    unit = fake / "systemd/lawnberry-health.service"
    unit.write_text(unit.read_text().replace("@LB_GROUP@", "@LB_GRUOP@"))
    root, dest = tmp_path / "app", tmp_path / "dest"
    make_checkout(root)
    result = installer(*install_args(root, dest), credential=STRONG, repo=fake)
    assert result.returncode == 1
    assert "@LB_GRUOP@" in result.stderr
    assert not dest.exists() and tree(root) == UNTOUCHED


# ---- credential refusal ------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "operator123",  # the old code default
        "Operator123!",
        "admin",
        "emergency123",
        "password12345",
        "short-pw-1",
        "aaaaaaaaaaaaaaaa",
        "",
    ],
)
def test_default_weak_or_empty_credentials_refuse_the_install(site, bad):
    root, dest = site
    result = installer(*install_args(root, dest), credential=bad)
    assert result.returncode == 3, result.stderr + result.stdout
    assert "REFUSED" in result.stderr
    assert "nothing was installed or changed" in result.stderr
    assert not dest.exists(), "no unit may be installed"
    assert tree(root) == UNTOUCHED, "no .env may be created or modified"
    if bad:
        assert bad not in result.stdout + result.stderr


def test_a_twelve_character_non_default_credential_is_enough(site):
    root, dest = site
    assert installer(*install_args(root, dest), credential="Xk9#mQ2$vLp4").returncode == 0


def test_no_credential_at_all_refuses_and_says_how_to_fix_it(site):
    root, dest = site
    result = installer(*install_args(root, dest))
    assert result.returncode == 3
    assert "no operator credential is configured" in result.stderr
    assert "--operator-credential-file" in result.stderr
    assert not dest.exists() and tree(root) == UNTOUCHED


def test_a_default_already_sitting_in_the_env_file_refuses(site):
    root, dest = site
    env_file = root / ".env"
    env_file.write_text("NTRIP_HOST=x\nLAWN_BERRY_OPERATOR_CREDENTIAL=operator123\n")
    env_file.chmod(0o600)
    before = env_file.read_bytes()
    result = installer(*install_args(root, dest))
    assert result.returncode == 3 and "REFUSED" in result.stderr
    assert env_file.read_bytes() == before and not dest.exists()


def test_credential_file_option_installs_and_stores_it(site, tmp_path):
    root, dest = site
    cred = tmp_path / "cred"
    cred.write_text(STRONG + "\n")
    cred.chmod(0o600)
    result = installer(*install_args(root, dest, "--operator-credential-file", str(cred)))
    assert result.returncode == 0, result.stderr
    assert env_values(root / ".env")["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG


def test_a_weak_credential_file_is_refused_like_any_other(site, tmp_path):
    root, dest = site
    cred = tmp_path / "cred"
    cred.write_text("operator123\n")
    cred.chmod(0o600)
    result = installer(*install_args(root, dest, "--operator-credential-file", str(cred)))
    assert result.returncode == 3 and "REFUSED" in result.stderr
    assert not dest.exists() and tree(root) == UNTOUCHED


def test_an_existing_good_env_file_needs_no_credential_option_and_keeps_its_lines(site):
    root, dest = site
    env_file = root / ".env"
    env_file.write_text(f"NTRIP_HOST=caster.example\nLAWN_BERRY_OPERATOR_CREDENTIAL='{STRONG}'\n")
    env_file.chmod(0o644)
    result = installer(*install_args(root, dest))
    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    values = env_values(env_file)
    assert values["NTRIP_HOST"] == "caster.example"
    assert values["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG
    assert "LAWN_BERRY_AUTH_SECRET" in values


# ---- signing-secret persistence ----------------------------------------------------------


def test_the_signing_secret_is_generated_once_and_persists_across_installs(site):
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    first = env_values(root / ".env")["LAWN_BERRY_AUTH_SECRET"]
    for _ in range(2):  # a backend update re-runs the installer; sessions must survive it
        assert installer(*install_args(root, dest)).returncode == 0
        assert env_values(root / ".env")["LAWN_BERRY_AUTH_SECRET"] == first
    assert stat.S_IMODE((root / ".env").stat().st_mode) == 0o600


def test_rotate_auth_secret_replaces_it_and_keeps_the_credential(site):
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    first = env_values(root / ".env")
    assert installer(*install_args(root, dest, "--rotate-auth-secret")).returncode == 0
    second = env_values(root / ".env")
    assert second["LAWN_BERRY_AUTH_SECRET"] != first["LAWN_BERRY_AUTH_SECRET"]
    assert second["LAWN_BERRY_OPERATOR_CREDENTIAL"] == STRONG


def test_a_short_existing_secret_refuses_the_install(site):
    root, dest = site
    env_file = root / ".env"
    env_file.write_text(
        f"LAWN_BERRY_AUTH_SECRET=hunter2\nLAWN_BERRY_OPERATOR_CREDENTIAL='{STRONG}'\n"
    )
    env_file.chmod(0o600)
    result = installer(*install_args(root, dest))
    assert result.returncode == 3
    assert "LAWN_BERRY_AUTH_SECRET" in result.stderr and "--rotate-auth-secret" in result.stderr
    assert not dest.exists()


def test_auth_disabled_in_the_env_file_is_a_warning_not_a_refusal(site):
    root, dest = site
    (root / ".env").write_text("OPERATOR_AUTH_REQUIRED=0\n")
    result = installer(*install_args(root, dest), credential=STRONG)
    assert result.returncode == 0
    assert "operator authentication is DISABLED" in result.stderr


# ---- what root runs: only root-owned copies, never the checkout ------------------------------
#
# The checkout, and everything in it, belongs to the service account (the backend unit can write
# all of it). A unit that runs as root and executes or loads a file from there lets a code
# execution bug in the network-facing backend become code execution as root at the next start.
# The installer therefore copies the scripts root runs to /usr/local/sbin (root:root 0755) and the
# units name only those copies and system binaries.

RENEWAL_SCRIPT = "/usr/local/sbin/lawnberry-renew-certificates.sh"
HALOW_SCRIPT = "/usr/local/sbin/halow-link-watch.sh"
SYSTEM_BIN_DIRS = ("/usr/bin/", "/usr/sbin/", "/bin/", "/sbin/")
EXEC_KEYS = (
    *("ExecStart", "ExecStartPre", "ExecStartPost", "ExecCondition"),
    *("ExecReload", "ExecStop", "ExecStopPost"),
)


def unit_text(path: Path) -> str:
    """The unit file and its drop-ins in the order systemd applies them."""
    text = path.read_text()
    dropins = path.parent / (path.name + ".d")
    for conf in sorted(dropins.glob("*.conf")) if dropins.is_dir() else []:
        text += "\n" + conf.read_text()
    return text


def unit_settings(text: str) -> dict[tuple[str, str], list[str]]:
    """(section, key) -> every assignment in order, read the way systemd reads a unit: line
    continuations joined, comments dropped, an empty assignment resets what came before."""
    joined = re.sub(r"\\\n[ \t]*", " ", text)
    settings: dict[tuple[str, str], list[str]] = {}
    section = ""
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        bucket = settings.setdefault((section, key.strip()), [])
        if value.strip() == "":
            bucket.clear()
        else:
            bucket.append(value.strip())
    return settings


def last(settings: dict[tuple[str, str], list[str]], section: str, key: str) -> str:
    values = settings.get((section, key)) or [""]
    return values[-1]


def under(path: str, tree: str) -> bool:
    return path == tree or path.startswith(tree.rstrip("/") + "/")


def root_commands(settings: dict[tuple[str, str], list[str]]) -> list[tuple[str, str]]:
    """(key, command line) of every command that runs as root: all of them in a unit that has
    no User= (or User=root), and the ones with a "+" or "!" prefix in any other unit."""
    unit_is_root = last(settings, "Service", "User") in ("", "root", "0")
    found = []
    for key in EXEC_KEYS:
        for line in settings.get(("Service", key), []):
            prefix = re.match(r"[@\-:+!]*", line).group(0)
            if unit_is_root or "+" in prefix or "!" in prefix:
                found.append((key, line[len(prefix) :]))
    return found


def root_run_problems(text: str, checkout: str, installed: set[str]) -> list[str]:
    """Why a unit lets root execute or load something the service account can change."""
    settings = unit_settings(text)
    commands = root_commands(settings)
    if not commands:
        return []
    problems = []
    names_checkout = re.compile(re.escape(checkout) + r"(?![A-Za-z0-9_.-])")
    for key, command in commands:
        if names_checkout.search(command):
            problems.append(f"{key} runs a command that names the checkout: {command}")
        words = command.split()
        exe = words[0] if words else ""
        if exe not in installed and not exe.startswith(SYSTEM_BIN_DIRS):
            problems.append(f"{key} runs {exe!r}: not an installed copy and not a system binary")
    for key in ("WorkingDirectory", "EnvironmentFile"):
        for value in settings.get(("Service", key), []):
            if under(value.lstrip("-"), checkout):
                problems.append(f"{key}={value} is inside the checkout")
    return problems


def installed_sbin(dest: Path) -> set[str]:
    sbin = dest / "usr/local/sbin"
    return {"/usr/local/sbin/" + p.name for p in sbin.iterdir()}


def test_no_unit_that_runs_as_root_executes_or_loads_anything_from_the_checkout(site):
    """Every installed unit that runs as root (no User=, or User=root, or a "+"/"!" command in
    another unit) executes only a copy the installer put in /usr/local/sbin or a system binary,
    has no working directory or environment file in the checkout, and names the checkout in no
    command line. There are no exceptions: lawnberry-acme-renew.service, the one root unit that
    runs from the checkout, is never installed (test_the_invariant_flags_the_legacy_acme_unit)."""
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    installed = installed_sbin(dest)
    checkout = str(root.resolve())
    examined = set()
    for path in sorted(unit_dir(dest).glob("*.service")):
        text = unit_text(path)
        problems = root_run_problems(text, checkout, installed)
        assert problems == [], f"{path.name}: {problems}"
        if root_commands(unit_settings(text)):
            examined.add(path.name)
    # the check is not vacuous: both root units were looked at
    assert {"lawnberry-cert-renewal.service", "halow-link-watch.service"} <= examined
    assert RENEWAL_SCRIPT in installed and HALOW_SCRIPT in installed


BAD_ROOT_UNITS = {
    "no User=, a script of the checkout": "[Service]\nExecStart={root}/scripts/x.sh\n",
    "User=root, the checkout through a login shell": (
        "[Service]\nUser=root\nExecStart=/usr/bin/env bash -lc 'bash {root}/scripts/x.sh'\n"
    ),
    "User=root, the same line after a continuation": (
        "[Service]\nUser=root\nExecStart=/usr/bin/env bash \\\n    {root}/scripts/x.sh\n"
    ),
    "User=root, a pre-start step in the checkout": (
        f"[Service]\nUser=root\nExecStartPre={{root}}/scripts/x.sh\nExecStart={RENEWAL_SCRIPT}\n"
    ),
    "User=root, working directory in the checkout": (
        f"[Service]\nUser=root\nWorkingDirectory={{root}}\nExecStart={RENEWAL_SCRIPT}\n"
    ),
    "User=root, environment file in the checkout": (
        f"[Service]\nUser=root\nEnvironmentFile=-{{root}}/.env\nExecStart={RENEWAL_SCRIPT}\n"
    ),
    "User= reset to root": "[Service]\nUser=svc\nUser=\nExecStart={root}/x\n",
    "service account, a + pre-start step": (
        "[Service]\nUser=svc\nExecStartPre=+{root}/scripts/x.sh\nExecStart=/usr/bin/true\n"
    ),
    "service account, a ! pre-start step": (
        "[Service]\nUser=svc\nExecStartPre=!{root}/scripts/x.sh\nExecStart=/usr/bin/true\n"
    ),
    "service account, a + step that loads the checkout's env file": (
        f"[Service]\nUser=svc\nEnvironmentFile=-{{root}}/.env\n"
        f"ExecStartPre=+{RENEWAL_SCRIPT}\nExecStart=/usr/bin/true\n"
    ),
    "User=root, an executable nobody installed": "[Service]\nUser=root\nExecStart=/opt/o/run.sh\n",
    "User=root, a relative executable": "[Service]\nUser=root\nExecStart=./scripts/x.sh\n",
}
GOOD_UNITS = {
    "a service-account unit uses its checkout": (
        "[Service]\nUser=svc\nWorkingDirectory={root}\nEnvironmentFile=-{root}/.env\n"
        "ExecStart={root}/.venv/bin/uvicorn backend.src.main:app\n"
    ),
    "a root unit running an installed copy": f"[Service]\nExecStart={HALOW_SCRIPT}\n",
    "the checkout as data in Environment=": (
        f"[Service]\nUser=root\nEnvironment=LAWNBERRY_ENV_FILE={{root}}/.env\n"
        f"ExecStart={RENEWAL_SCRIPT}\n"
    ),
    "a sibling directory that shares the prefix": (
        "[Service]\nUser=root\nExecStart=/usr/bin/env bash -c 'echo {root}2/x'\n"
    ),
}


@pytest.mark.parametrize("name", BAD_ROOT_UNITS)
def test_the_root_invariant_flags_every_way_to_run_the_checkout_as_root(name):
    checkout = "/apps/lawnberry-pi"
    text = BAD_ROOT_UNITS[name].replace("{root}", checkout)
    assert root_run_problems(text, checkout, {RENEWAL_SCRIPT, HALOW_SCRIPT}), name


@pytest.mark.parametrize("name", GOOD_UNITS)
def test_the_root_invariant_accepts_what_is_legitimate(name):
    checkout = "/apps/lawnberry-pi"
    text = GOOD_UNITS[name].replace("{root}", checkout)
    assert root_run_problems(text, checkout, {RENEWAL_SCRIPT, HALOW_SCRIPT}) == [], name


def test_the_invariant_flags_the_legacy_acme_unit_which_is_why_it_is_never_installed(site):
    """lawnberry-acme-renew.service is User=root and runs ./.venv/bin/python from the checkout:
    exactly what the invariant forbids. It stays in the repository as legacy and the installer
    never installs it (and disables it on a host that has it)."""
    root, dest = site
    legacy = (SYSTEMD / "lawnberry-acme-renew.service").read_text()
    text = legacy.replace("@LB_ROOT@", str(root.resolve()))
    assert root_run_problems(text, str(root.resolve()), set())
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    assert not (unit_dir(dest) / "lawnberry-acme-renew.service").exists()


def test_the_renewal_unit_runs_only_the_root_owned_copy_the_installer_puts_in_sbin(site):
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0

    script = dest / RENEWAL_SCRIPT.lstrip("/")
    assert script.is_file() and not script.is_symlink()
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    assert filecmp.cmp(script, REPO / "scripts/renew_certificates.sh", shallow=False)

    s = unit_settings(unit_text(unit_dir(dest) / "lawnberry-cert-renewal.service"))
    assert s[("Service", "ExecStart")] == [RENEWAL_SCRIPT], "one path: no shell, no arguments"
    assert (dest / s[("Service", "ExecStart")][0].lstrip("/")).is_file()
    assert s[("Unit", "ConditionPathExists")] == [RENEWAL_SCRIPT], "skipped, not failed, before"
    assert s[("Service", "WorkingDirectory")] == ["/"], "not the checkout"
    assert last(s, "Service", "User") == "root" and last(s, "Service", "Group") == "root"
    env = dict(item.partition("=")[::2] for item in s[("Service", "Environment")])
    assert env["PATH"] == "/usr/sbin:/usr/bin:/sbin:/bin", "only root-owned directories"
    assert env["LAWNBERRY_ENV_FILE"] == f"{root.resolve()}/.env", "the .env is passed as data"
    assert not any(k[1] in ("EnvironmentFile", "ExecStartPre") for k in s), "nothing else loaded"

    # the hardening the unit always had is still there, and a hung run ends
    assert last(s, "Service", "NoNewPrivileges") == "true"
    assert last(s, "Service", "ProtectSystem") in ("full", "strict")
    assert last(s, "Service", "PrivateTmp") == "true"
    assert last(s, "Service", "PrivateDevices") == "true"
    assert "CAP_SYS_ADMIN" not in last(s, "Service", "CapabilityBoundingSet")
    assert last(s, "Service", "TimeoutStartSec") not in ("", "infinity", "0")


# What the renewal job writes, by what writes it: scripts/renew_certificates.sh (openssl for the
# self-signed fallback, sed/cp on the nginx configs, nginx -t and systemctl reload nginx), certbot
# renew (scripts/setup_lets_encrypt.sh shows its paths) and its deploy hook.
RENEWAL_WRITES = [
    "/etc/lawnberry/certs/selfsigned/privkey.pem",
    "/etc/lawnberry/certs/selfsigned/fullchain.pem",
    "/etc/nginx/nginx.conf",
    "/etc/nginx/sites-available/lawnberry-https.bak.1760000000",
    "/etc/nginx/sites-enabled/lawnberry-https",
    "/etc/letsencrypt/live/mower.example.com/fullchain.pem",
    "/etc/letsencrypt/archive/mower.example.com/cert2.pem",
    "/etc/letsencrypt/renewal/mower.example.com.conf",
    "/var/lib/letsencrypt/.certbot.lock",
    "/var/log/letsencrypt/letsencrypt.log",
    "/var/www/.well-known/acme-challenge/token",
    "/var/log/nginx/error.log",
    "/tmp/certbot-scratch",
]
# What it must NOT be able to write: the copy of itself and the rest of the system.
RENEWAL_READ_ONLY = [
    RENEWAL_SCRIPT,
    "/usr/bin/openssl",
    "/etc/passwd",
    "/etc/systemd/system/lawnberry-cert-renewal.service",
    "/home/alice/lawnberry-pi/scripts/renew_certificates.sh",
    "/home/alice/lawnberry-pi/.env",
]
PROTECT_SYSTEM = {
    **dict.fromkeys(("yes", "true", "1", "on"), ("/usr", "/boot", "/efi")),
    "full": ("/usr", "/boot", "/efi", "/etc"),
    "strict": ("/",),
}
HOME_TREES = ("/home", "/root", "/run/user")


def sandbox_access(path: str, s: dict[tuple[str, str], list[str]]) -> str:
    """What a process of the unit gets on PATH: "rw", "ro" or "none". It follows systemd.exec(5)
    for ProtectSystem= (true: /usr, /boot, /efi read-only; full: and /etc; strict: everything),
    ProtectHome= (true: /home, /root, /run/user hidden; read-only) and ReadWritePaths= (a "-"
    prefix only skips a path that is missing); PrivateTmp= keeps /tmp and /var/tmp writable."""
    access = "rw"
    if any(
        under(path, tree) for tree in PROTECT_SYSTEM.get(last(s, "Service", "ProtectSystem"), ())
    ):
        access = "ro"
    if any(under(path, tree) for tree in HOME_TREES):
        home = last(s, "Service", "ProtectHome")
        if home in ("yes", "true", "1", "on", "tmpfs"):
            return "none"
        if home == "read-only":
            access = "ro"
    for entry in s.get(("Service", "ReadWritePaths"), []):
        if any(under(path, item.lstrip("-")) for item in entry.split()):
            access = "rw"
    if last(s, "Service", "PrivateTmp") == "true" and (
        under(path, "/tmp") or under(path, "/var/tmp")
    ):
        access = "rw"
    return access


def test_the_renewal_sandbox_lets_the_job_write_what_it_writes_and_nothing_else(site):
    """As shipped, ProtectSystem=full made /etc read-only for a job that writes the self-signed
    certificate and the nginx configs under /etc, so it could not do its work. The writable
    paths must now cover everything the job writes, and still leave the rest read-only."""
    root, dest = site
    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    s = unit_settings(unit_text(unit_dir(dest) / "lawnberry-cert-renewal.service"))
    blocked = [p for p in RENEWAL_WRITES if sandbox_access(p, s) != "rw"]
    assert blocked == [], f"the job cannot write {blocked}"
    writable = [p for p in RENEWAL_READ_ONLY if sandbox_access(p, s) == "rw"]
    assert writable == [], f"the sandbox leaves {writable} writable"


def test_a_checkout_under_home_stays_readable_to_the_renewal_job(tmp_path):
    """ProtectHome=true hides /home: with the checkout there, the job could not read .env and
    would silently ignore LB_DOMAIN (and, as shipped, could not even enter the working
    directory). Rendered for a checkout under /home, everything the unit points at is readable."""
    fake_root = "/home/alice/lawnberry-pi"
    text = (SYSTEMD / "lawnberry-cert-renewal.service").read_text().replace("@LB_ROOT@", fake_root)
    s = unit_settings(text)
    env_items = [v.partition("=")[2] for v in s.get(("Service", "Environment"), [])]
    pointed_at = [value for value in env_items if fake_root in value]
    pointed_at += [
        v for key in ("WorkingDirectory", "ExecStart") for v in s.get(("Service", key), [])
    ]
    pointed_at = [p for p in pointed_at if p.startswith("/")]
    hidden = [p for p in pointed_at if sandbox_access(p, s) == "none"]
    assert hidden == [], f"ProtectHome hides {hidden} from the job"
    assert f"{fake_root}/.env" in pointed_at, "the unit tells the job where .env is"
    assert sandbox_access(f"{fake_root}/.env", s) in ("ro", "rw")


def test_the_sandbox_model_matches_systemd_on_the_cases_it_decides():
    """The model above is only as good as its reading of systemd.exec(5): pin it on the shapes
    the renewal unit relies on (and on the way the shipped unit used to fail)."""
    old = unit_settings("[Service]\nProtectSystem=full\nProtectHome=true\nPrivateTmp=true\n")
    assert sandbox_access("/etc/lawnberry/certs/selfsigned/privkey.pem", old) == "ro"
    assert sandbox_access("/home/alice/lawnberry-pi/.env", old) == "none"
    assert sandbox_access("/var/lib/letsencrypt/x", old) == "rw"
    new = unit_settings(
        "[Service]\nProtectSystem=full\nProtectHome=read-only\n"
        "ReadWritePaths=-/etc/lawnberry -/etc/nginx\nReadWritePaths=-/etc/letsencrypt\n"
    )
    assert sandbox_access("/etc/lawnberry/certs/x", new) == "rw"
    assert sandbox_access("/etc/letsencrypt/live/x", new) == "rw"
    assert sandbox_access("/etc/lawnberryX/x", new) == "ro"
    assert sandbox_access("/home/alice/x", new) == "ro"
    strict = unit_settings("[Service]\nProtectSystem=strict\nPrivateTmp=true\n")
    assert sandbox_access("/var/lib/x", strict) == "ro"
    assert sandbox_access("/tmp/x", strict) == "rw"


def test_the_dry_run_lists_the_root_owned_script_copies_and_writes_nothing(tmp_path):
    root = make_checkout(tmp_path / "mower")
    result = installer("--dry-run", "--user", "alice", "--root", str(root), credential=STRONG)
    assert result.returncode == 0, result.stderr
    for dest_name, source in (
        ("lawnberry-renew-certificates.sh", "scripts/renew_certificates.sh"),
        ("halow-link-watch.sh", "scripts/halow_link_watch.sh"),
    ):
        line = f"would install /usr/local/sbin/{dest_name} (0755 root:root) from {source}"
        assert line in result.stdout, result.stdout
    assert tree(root) == UNTOUCHED


def test_a_rerun_replaces_the_installed_script_with_the_repository_one(tmp_path):
    fake = fake_repo(tmp_path)
    root, dest = tmp_path / "app", tmp_path / "dest"
    make_checkout(root)
    assert installer(*install_args(root, dest), credential=STRONG, repo=fake).returncode == 0
    installed = dest / RENEWAL_SCRIPT.lstrip("/")
    installed.write_text("#!/bin/sh\necho an older or tampered copy\n")
    installed.chmod(0o777)
    source = fake / "scripts/renew_certificates.sh"
    source.write_text(source.read_text() + "\n# a later revision\n")

    assert installer(*install_args(root, dest), credential=STRONG, repo=fake).returncode == 0
    assert installed.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(installed.stat().st_mode) == 0o755
    # and a third run changes nothing
    before = installed.read_bytes()
    assert installer(*install_args(root, dest), credential=STRONG, repo=fake).returncode == 0
    assert installed.read_bytes() == before


def test_a_symlink_where_the_script_goes_is_replaced_not_written_through(site):
    """An operator who once linked /usr/local/sbin/<script> to the checkout's copy (so the unit
    would "run the repository version") would keep root running a file the service account owns.
    The installer replaces the link with a root-owned regular file and leaves the target alone."""
    root, dest = site
    decoy = root / "scripts" / "renew_certificates.sh"
    decoy.parent.mkdir()
    decoy.write_text("#!/bin/sh\necho the checkout's copy\n")
    installed = dest / RENEWAL_SCRIPT.lstrip("/")
    installed.parent.mkdir(parents=True)
    installed.symlink_to(decoy)

    assert installer(*install_args(root, dest), credential=STRONG).returncode == 0
    assert not installed.is_symlink() and installed.is_file()
    assert filecmp.cmp(installed, REPO / "scripts/renew_certificates.sh", shallow=False)
    assert decoy.read_text() == "#!/bin/sh\necho the checkout's copy\n"


def test_a_root_script_missing_from_the_repository_aborts_before_anything_is_installed(tmp_path):
    fake = fake_repo(tmp_path)
    (fake / "scripts/renew_certificates.sh").unlink()
    root, dest = tmp_path / "app", tmp_path / "dest"
    make_checkout(root)
    result = installer(*install_args(root, dest), credential=STRONG, repo=fake)
    assert result.returncode == 1
    assert "renew_certificates.sh" in result.stderr
    assert not dest.exists() and tree(root) == UNTOUCHED, "no unit without its script"


FAKEROOT = shutil.which("fakeroot")


@pytest.mark.skipif(FAKEROOT is None, reason="fakeroot is not installed")
def test_run_as_root_the_scripts_are_installed_root_root_0755(tmp_path):
    """A non-root --destdir run passes no -o/-g to install (it could not chown), so it cannot show
    that root's copy is root:root. Under fakeroot the installer sees EUID 0 and does pass them: an
    install shim records every call, then runs the real install (fakeroot fakes the chown)."""
    probe = subprocess.run(
        [FAKEROOT, "--", "sh", "-c", "id -u"], capture_output=True, text=True, timeout=60
    )
    if probe.stdout.strip() != "0":
        pytest.skip("fakeroot does not give this process uid 0 here")
    real_install = shutil.which("install") or "/usr/bin/install"
    shims, log = tmp_path / "shims", tmp_path / "install.log"
    shims.mkdir()
    (shims / "install").write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> "$INSTALL_LOG"\nexec {real_install} "$@"\n'
    )
    (shims / "install").chmod(0o755)
    root, dest = tmp_path / "app", tmp_path / "dest"
    make_checkout(root)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LAWN_BERRY_OPERATOR_CREDENTIAL", "LAWN_BERRY_AUTH_SECRET", "SUDO_USER"}
    }
    env.update(
        LAWN_BERRY_OPERATOR_CREDENTIAL=STRONG,
        INSTALL_LOG=str(log),
        PATH=f"{shims}:{env.get('PATH', '/usr/bin:/bin')}",
    )
    done = subprocess.run(
        [FAKEROOT, "--", "bash", str(REPO / "systemd/install_services.sh")]
        + install_args(root, dest),
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    calls = log.read_text().splitlines()
    for name in ("lawnberry-renew-certificates.sh", "halow-link-watch.sh"):
        (call,) = [c for c in calls if c.endswith(f"/usr/local/sbin/{name}")]
        assert "-m 0755" in call and "-o root -g root" in call, call
    assert stat.S_IMODE((dest / RENEWAL_SCRIPT.lstrip("/")).stat().st_mode) == 0o755
