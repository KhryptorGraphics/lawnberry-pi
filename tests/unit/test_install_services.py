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


@pytest.fixture
def site(tmp_path):
    """An app root and a staging root, as the installer sees them."""
    root = tmp_path / "app"
    dest = tmp_path / "dest"
    root.mkdir()
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
    for name in ("provision_env.py", "halow_link_watch.sh"):
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
    root = tmp_path / "mower"
    result = installer(
        "--dry-run", "--user", "alice", "--group", "staff", "--root", str(root), credential=STRONG
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "would SUCCEED" in out
    assert f"WorkingDirectory={root}" in out
    assert f"ExecStart={root}/.venv/bin/uvicorn backend.src.main:app" in out
    assert f"EnvironmentFile=-{root}/.env" in out
    assert f"ExecStartPre=/bin/chown -R alice:staff {root}" in out
    assert out.count("User=alice") >= 7 and "Group=staff" in out
    assert "Group=video" in out, "the camera unit keeps its device group"
    for stale in ("@LB_", "User=pi", "Group=pi", "pi:pi", "/apps/lawnberry-pi"):
        assert stale not in out, f"rendered output still contains {stale!r}"
    for name in REPO_UNIT_FILES:
        if name not in LEGACY_UNITS:
            assert f"/etc/systemd/system/{name} (0644 root:root)" in out, name
    assert not root.exists(), "--dry-run must write nothing"


def test_dry_run_never_prints_the_credential(tmp_path):
    result = installer("--dry-run", "--user", "alice", "--root", str(tmp_path), credential=STRONG)
    assert STRONG not in result.stdout + result.stderr


def test_dry_run_reports_a_refusal_but_still_shows_the_units(tmp_path):
    result = installer("--dry-run", "--user", "alice", "--root", str(tmp_path / "m"))
    assert result.returncode == 3
    assert "would REFUSE" in result.stdout
    assert "REFUSED: no operator credential is configured" in result.stderr
    assert "User=alice" in result.stdout, "the units are still rendered for review"


def test_the_account_defaults_to_the_owner_of_the_root(tmp_path):
    result = installer("--dry-run", "--root", str(tmp_path), credential=STRONG)
    assert result.returncode == 0, result.stderr
    assert f"service account : {ME}" in result.stdout
    assert f"User={ME}" in result.stdout


def test_sudo_user_beats_the_owner_of_the_root(tmp_path):
    result = installer(
        "--dry-run", "--root", str(tmp_path), credential=STRONG, extra_env={"SUDO_USER": "bob"}
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
    root.mkdir()
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
    root.mkdir()
    result = installer(*install_args(root, dest), credential=STRONG, repo=fake)
    assert result.returncode == 1
    assert "@LB_GRUOP@" in result.stderr
    assert not dest.exists() and list(root.iterdir()) == []


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
    assert list(root.iterdir()) == [], "no .env may be created or modified"
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
    assert not dest.exists() and list(root.iterdir()) == []


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
    assert not dest.exists() and list(root.iterdir()) == []


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
