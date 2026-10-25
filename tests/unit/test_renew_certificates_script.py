"""scripts/renew_certificates.sh: root reads ``.env`` as data, never as shell.

``lawnberry-cert-renewal.service`` runs this script as root. The checkout and the ``.env`` in it
belong to the service account (the backend, a network-facing process, can rewrite both), so the
script must not run anything that file says. These tests run the real script, through a real
bash, against hostile and valid ``.env`` files:

* ``--print-config`` prints ``domain=`` and ``env_file=`` without root, openssl, nginx, a syslog
  call or a single write, so everything the parser decides is observable here;
* the normal path is run only as far as it can go without root: with no ``openssl`` on PATH it
  stops at ``require_cmd`` (the place where the reviewed demo showed the ``.env`` already
  executed), so the same files are also checked on the way the timer takes.

A hostile line carries a payload that writes ``marker``. Every payload is first SOURCED by a
plain ``bash`` here (``test_the_payloads_are_live_when_sourced``): that proves the marker
assertions can fail, i.e. that "marker absent" means "not executed" and not "could never run".
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "renew_certificates.sh"
BASH = shutil.which("bash") or "/usr/bin/bash"

pytestmark = pytest.mark.skipif(not Path(BASH).exists(), reason="the renewal script needs bash")

# What a stub ``logger`` records: every call, and the environment it was started with. It is a
# BASH script on purpose: a bash child of the renewal script honours an exported BASH_ENV.
LOGGER_STUB = """#!/bin/bash
{ printf 'logger'; printf ' %s' "$@"; printf '\\n'; } >> "$LOGGER_LOG"
for v in $(compgen -e); do printf '%s=%s\\n' "$v" "${!v}"; done > "$LOGGER_ENV"
exit 0
"""


@dataclass
class Run:
    returncode: int
    stdout: str
    stderr: str

    @property
    def config(self) -> dict[str, str]:
        """The two ``key=value`` lines of ``--print-config`` (anything else fails the test)."""
        lines = self.stdout.splitlines()
        assert [ln.partition("=")[0] for ln in lines] == ["domain", "env_file"], self.stdout
        return dict(ln.partition("=")[::2] for ln in lines)

    @property
    def domain(self) -> str:
        return self.config["domain"]


class Harness:
    """A checkout holding a copy of the script, a PATH holding only a stub ``logger``."""

    def __init__(self, tmp: Path, script: Path = SCRIPT) -> None:
        tmp = tmp.resolve()  # the script prints physical paths
        self.tmp = tmp
        self.checkout = tmp / "checkout"
        (self.checkout / "scripts").mkdir(parents=True)
        self.script = self.checkout / "scripts" / "renew_certificates.sh"
        shutil.copy(script, self.script)
        self.bin = tmp / "bin"
        self.bin.mkdir()
        logger = self.bin / "logger"
        logger.write_text(LOGGER_STUB)
        logger.chmod(0o755)
        # The script needs no tool but logger. The one it replaced needed dirname to find its own
        # .env; without it that script would never read the file and every "not executed"
        # assertion would pass for the wrong reason.
        (self.bin / "dirname").symlink_to(shutil.which("dirname") or "/usr/bin/dirname")
        self.marker = tmp / "marker"
        self.logger_log = tmp / "logger.log"
        self.logger_env = tmp / "logger.env"
        self.env_file = self.checkout / ".env"

    def write_env(self, content: str | bytes) -> Path:
        data = content.encode() if isinstance(content, str) else content
        self.env_file.write_bytes(data)
        return self.env_file

    def run(
        self,
        *args: str,
        env: dict[str, str] | None = None,
        path: str | None = None,
        script: Path | None = None,
        timeout: int = 60,
    ) -> Run:
        full = {
            "PATH": path if path is not None else str(self.bin),
            "LOGGER_LOG": str(self.logger_log),
            "LOGGER_ENV": str(self.logger_env),
        }
        full.update(env or {})
        done = subprocess.run(
            [BASH, str(script or self.script), *args],
            capture_output=True,
            text=True,
            env=full,
            cwd=self.tmp,
            timeout=timeout,
        )
        return Run(done.returncode, done.stdout, done.stderr)

    def print_config(self, **kwargs) -> Run:
        done = self.run("--print-config", **kwargs)
        assert not self.marker.exists(), "a line of the .env was executed"
        assert done.returncode == 0, done.stdout + done.stderr
        return done

    def logger_calls(self) -> list[str]:
        return self.logger_log.read_text().splitlines() if self.logger_log.exists() else []

    def payload(self) -> str:
        """A shell fragment that writes the marker; only builtins, so it needs no PATH."""
        return f"echo pwned > {self.marker}"


@pytest.fixture
def h(tmp_path) -> Harness:
    return Harness(tmp_path)


# ---- the hostile lines ---------------------------------------------------------------------

# (id, .env text with {p} = the payload, domain the script may report). "" means: no domain.
HOSTILE = [
    ("command-substitution", "X=$({p})\n", ""),
    ("backticks", "X=`{p}`\n", ""),
    ("bare-command-line", "{p}\n", ""),
    ("domain-command-substitution", "LB_DOMAIN=$({p})\n", ""),
    ("domain-backticks", "LB_DOMAIN=`{p}`\n", ""),
    ("domain-semicolon-unquoted", "LB_DOMAIN=a.example; {p}\n", ""),
    ("domain-semicolon-single-quoted", "LB_DOMAIN='a.example; {p}'\n", ""),
    ("domain-semicolon-double-quoted", 'LB_DOMAIN="a.example; {p}"\n', ""),
    ("domain-ampersand", "LB_DOMAIN=a.example & {p}\n", ""),
    ("domain-pipe", "LB_DOMAIN=a.example | {p}\n", ""),
    ("domain-dollar-expansion", "LB_DOMAIN=${X:-$({p})}\n", ""),
    ("export-command-substitution", "export LB_DOMAIN=$({p})\n", ""),
    ("export-valid-then-payload", "export LB_DOMAIN=ok.example; {p}\n", ""),
    ("domain-path-traversal", "LB_DOMAIN=../../etc\n", ""),
    ("domain-slash", "LB_DOMAIN=a/b\n", ""),
    ("domain-leading-dot", "LB_DOMAIN=.example.com\n", ""),
    ("domain-dot-dot", "LB_DOMAIN=a..example.com\n", ""),
    ("domain-leading-hyphen", "LB_DOMAIN=-rf.example\n", ""),
    ("domain-space", "LB_DOMAIN=a b.example\n", ""),
    ("domain-inline-comment", "LB_DOMAIN=mower.example.com # comment\n", ""),
    ("domain-openssl-subject", "LB_DOMAIN=a.example/O=evil\n", ""),
    ("domain-cr-in-the-middle", "LB_DOMAIN=a.example\rb.example\n", ""),
    ("crlf-then-payload-line", "LB_DOMAIN=a.example\r\n{p}\r\n", "a.example"),
    ("crlf-payload-in-value", "LB_DOMAIN=a.example\r$({p})\r\n", ""),
    ("domain-unbalanced-quote", "LB_DOMAIN='a.example\n", ""),
    ("domain-quote-then-text", "LB_DOMAIN='a.example'b\n", ""),
    ("domain-two-quoted-words", "LB_DOMAIN='a.example' 'b.example'\n", ""),
    ("domain-too-long", "LB_DOMAIN=" + "a" * 254 + "\n", ""),
    ("domain-non-ascii", "LB_DOMAIN=m\u00e4her.example\n", ""),
    ("megabyte-line", "LB_DOMAIN=" + "a" * 1_000_000 + "\n", ""),
    ("megabyte-comment-line", "#" + "a" * 1_000_000 + "\n", ""),
    ("megabyte-line-then-valid", "X=" + "a" * 1_000_000 + "\nLB_DOMAIN=late.example\n", ""),
    ("bash-env", "export BASH_ENV={p_file}\n", ""),
    ("path", "PATH={evil}\n", ""),
    ("export-path", "export PATH={evil}\n", ""),
    ("ld-preload", "export LD_PRELOAD=/nonexistent/evil.so\nLD_PRELOAD=/x.so\n", ""),
    ("export-lowercase-key", "export lb_domain=ok.example\n", ""),
    ("export-valid-domain", "export LB_DOMAIN=ok.example\n", ""),
    ("export-valid-domain-fallback", "export DOMAIN=ok.example\n", ""),
]


def _render(text: str, h: Harness, evil: Path) -> str:
    payload_file = h.tmp / "payload.sh"
    payload_file.write_text(f"{h.payload()}\n")
    return (
        text.replace("{p_file}", str(payload_file))
        .replace("{evil}", str(evil))
        .replace("{p}", h.payload())
    )


@pytest.fixture
def evil(h) -> Path:
    """A directory of fake ``openssl`` and ``nginx`` that write the marker when they run."""
    directory = h.tmp / "evil"
    directory.mkdir()
    for name in ("openssl", "nginx"):
        stub = directory / name
        stub.write_text(f"#!/bin/sh\n{h.payload()}\nexit 0\n")
        stub.chmod(0o755)
    return directory


@pytest.mark.parametrize(("name", "text", "expected"), HOSTILE, ids=[x[0] for x in HOSTILE])
def test_a_hostile_env_file_is_never_executed_and_gives_no_domain(h, evil, name, text, expected):
    h.write_env(_render(text, h, evil))
    done = h.print_config()
    assert not h.marker.exists(), f"{name}: the .env was executed"
    assert done.domain == expected
    assert done.config["env_file"] == str(h.env_file)
    # nothing of the file is echoed back (a warning names the line, never the value)
    for leak in ("pwned", "evil"):
        assert leak not in done.stdout + done.stderr, done.stderr
    assert not h.logger_log.exists(), "--print-config must not call logger"


EXECUTABLE = [  # the subset of HOSTILE that runs its payload when the file is SOURCED
    "command-substitution",
    "backticks",
    "bare-command-line",
    "domain-command-substitution",
    "domain-backticks",
    "domain-semicolon-unquoted",
    "domain-dollar-expansion",
    "export-command-substitution",
    "export-valid-then-payload",
]


@pytest.mark.parametrize("name", EXECUTABLE)
def test_the_payloads_are_live_when_sourced(h, evil, name):
    """Control for the marker assertions: a shell that sources these files does run them."""
    text = next(t for n, t, _ in HOSTILE if n == name)
    env_file = h.write_env(_render(text, h, evil))
    subprocess.run(
        [BASH, "-c", 'source "$1"', "_", str(env_file)],
        capture_output=True,
        env={"PATH": str(h.bin)},
        timeout=30,
    )
    assert h.marker.exists(), f"{name}: sourcing it does not run the payload; the test is blind"


def test_a_hostile_env_file_does_not_change_what_the_normal_run_executes(h, evil):
    """The way the timer takes: with no openssl on PATH the run must stop at require_cmd, as a
    plain error, whatever the file says. This is the reviewed demo (a line in .env ran as the
    user running the script), plus a PATH line that would have made the stubs resolvable."""
    if os.geteuid() == 0:
        pytest.skip("a run that went wrong here would not be harmless as root")
    payload_file = h.tmp / "payload.sh"
    payload_file.write_text(f"{h.payload()}\n")
    h.write_env(
        f"X=$({h.payload()})\n"
        f"PATH={evil}\n"
        f"export BASH_ENV={payload_file}\n"
        "export LD_PRELOAD=/nonexistent/evil.so\n"
        f"LB_DOMAIN=$({h.payload()})\n"
    )
    done = h.run()
    assert done.returncode == 1, done.stdout + done.stderr
    assert "[ERR ] Missing command: openssl" in done.stdout
    assert not h.marker.exists(), "the .env (or a stub it made resolvable) ran"
    # the one child the run started (logger) got the environment the run itself was given
    seen = dict(
        line.partition("=")[::2] for line in h.logger_env.read_text().splitlines() if "=" in line
    )
    assert seen["PATH"] == str(h.bin)
    assert "LD_PRELOAD" not in seen and "BASH_ENV" not in seen
    assert any(
        "daemon.err" in call and "Missing command: openssl" in call for call in h.logger_calls()
    )


# ---- the valid files -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("LB_DOMAIN=mower.example.com\n", "mower.example.com"),
        ("LB_DOMAIN='mower.example.com'\n", "mower.example.com"),
        ('LB_DOMAIN="mower.example.com"\n', "mower.example.com"),
        ("LB_DOMAIN=mower.example.com", "mower.example.com"),  # no final newline
        ("LB_DOMAIN=mower.example.com\r\nX=1\r\n", "mower.example.com"),  # CRLF
        ("LB_DOMAIN='mower.example.com'\r\n", "mower.example.com"),
        ("  LB_DOMAIN  =  mower.example.com  \n", "mower.example.com"),
        ("\n\n# a comment\nLB_DOMAIN=a.example\n\nOTHER=1\n", "a.example"),
        ("DOMAIN=fallback.example\n", "fallback.example"),
        ("DOMAIN=fallback.example\nLB_DOMAIN=primary.example\n", "primary.example"),
        ("LB_DOMAIN=primary.example\nDOMAIN=fallback.example\n", "primary.example"),
        ("LB_DOMAIN=\nDOMAIN=fallback.example\n", "fallback.example"),
        ("LB_DOMAIN=''\nDOMAIN=fallback.example\n", "fallback.example"),
        ("# LB_DOMAIN=commented.example\n; DOMAIN=commented.example\n", ""),
        ("LB_DOMAIN_EXTRA=nope.example\nALT_DOMAINS=nope.example\nXDOMAIN=nope.example\n", ""),
        ("OTHER=1\nNTRIP_PASSWORD='quoted'\n", ""),
        ("LB_DOMAIN=a.example\nLB_DOMAIN=second.example\n", "second.example"),  # last wins
        ("LB_DOMAIN=a.example\nLB_DOMAIN=\n", ""),  # an empty last assignment unsets it
        ("LB_DOMAIN=a.example\nLB_DOMAIN=../../etc\n", ""),  # an invalid last one unsets it too
        ("DOMAIN=old.example\nDOMAIN=new.example\n", "new.example"),
        ("X=" + "a" * 1021 + "\nLB_DOMAIN=after.example\n", "after.example"),  # 1023-char line
        ("X=" + "a" * 1022 + "\nLB_DOMAIN=after.example\n", "after.example"),  # 1024
        ("X=" + "a" * 5000 + "\nLB_DOMAIN=after.example\n", "after.example"),  # skipped unread
        ("LB_DOMAIN=early.example\nX=" + "a" * 1_000_000 + "\n", "early.example"),
    ],
)
def test_valid_env_files_give_their_domain(h, text, expected):
    h.write_env(text)
    assert h.print_config().domain == expected
    assert not h.marker.exists()


def test_nothing_after_the_byte_budget_is_read(h):
    """Only the first 32 KiB of the file count: a service-owned file cannot cost root more."""
    filler = "".join(f"K{i}=v\n" for i in range(5000))  # about 50 KB of harmless lines
    h.write_env(filler + "LB_DOMAIN=too-late.example\n")
    done = h.print_config()
    assert done.domain == ""
    assert "larger than 32768 bytes" in done.stderr


def test_a_long_line_that_is_not_ours_is_skipped_without_a_word(h):
    h.write_env("A=" + "x" * 5000 + "\nLB_DOMAIN=after.example\n")
    done = h.print_config()
    assert done.domain == "after.example"
    assert done.stderr == ""


def test_a_long_value_of_ours_counts_as_invalid_and_unsets_the_key(h):
    secret = "s3cr3t-" * 300
    h.write_env("LB_DOMAIN=good.example\nLB_DOMAIN=" + secret + "\nDOMAIN=fallback.example\n")
    done = h.print_config()
    assert done.domain == "fallback.example", "the earlier valid LB_DOMAIN is not resurrected"
    assert "LB_DOMAIN on line 2" in done.stderr and "1024 characters or more" in done.stderr
    assert "s3cr3t" not in done.stdout + done.stderr


@pytest.mark.parametrize("blank", ["\n", "a\n", "LB_DOMAIN=x y\n"])
def test_a_file_of_many_tiny_lines_costs_little(h, blank):
    """A file made of 32 K lines must not take quadratic time (it is read one line at a time)."""
    h.write_env(blank * (32_768 // len(blank)))
    done = h.run("--print-config", timeout=30)  # the 30 s deadline is the assertion
    assert done.returncode == 0


# ---- what is not a regular file ------------------------------------------------------------


def test_a_symbolic_link_is_not_followed(h):
    target = h.tmp / "elsewhere.env"
    target.write_text("LB_DOMAIN=from-the-target.example\n")
    h.env_file.symlink_to(target)
    done = h.print_config()
    assert done.domain == ""
    assert "symbolic link" in done.stderr
    assert "from-the-target" not in done.stdout + done.stderr


def test_a_dangling_symbolic_link_is_refused_not_ignored_silently(h):
    h.env_file.symlink_to(h.tmp / "missing.env")
    done = h.print_config()
    assert done.domain == "" and "symbolic link" in done.stderr


def test_a_directory_is_refused(h):
    h.env_file.mkdir()
    (h.env_file / "LB_DOMAIN").write_text("x")
    done = h.print_config()
    assert done.domain == ""
    assert "not a regular file" in done.stderr


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_a_fifo_is_refused_without_waiting_for_a_writer(h):
    os.mkfifo(h.env_file)
    done = h.run("--print-config", timeout=20)  # opening a FIFO would block for ever
    assert done.returncode == 0 and done.stdout.startswith("domain=\n")
    assert "not a regular file" in done.stderr


def test_a_missing_file_is_not_an_error(h):
    done = h.print_config()
    assert done.domain == ""
    assert done.stderr == ""
    assert done.config["env_file"] == str(h.env_file)


def test_an_unreadable_file_is_reported_and_the_run_goes_on(h):
    if os.geteuid() == 0:
        pytest.skip("root reads everything")
    h.write_env("LB_DOMAIN=a.example\n").chmod(0)
    done = h.print_config()
    assert done.domain == ""
    assert "cannot be read" in done.stderr


# ---- the process environment -------------------------------------------------------------


def test_the_environment_wins_over_the_file_and_lb_domain_over_domain(h):
    h.write_env("LB_DOMAIN=file.example\nDOMAIN=file-domain.example\n")
    assert h.print_config().domain == "file.example"
    assert h.print_config(env={"LB_DOMAIN": "env.example"}).domain == "env.example"
    assert h.print_config(env={"DOMAIN": "env-domain.example"}).domain == "file.example"
    h.write_env("DOMAIN=file-domain.example\n")
    assert h.print_config(env={"DOMAIN": "env-domain.example"}).domain == "env-domain.example"
    assert h.print_config(env={"LB_DOMAIN": "env.example"}).domain == "env.example"


def test_an_invalid_environment_value_is_ignored_with_a_warning_that_hides_it(h):
    h.write_env("LB_DOMAIN=file.example\n")
    done = h.print_config(env={"LB_DOMAIN": "../../etc/shadow-secret"})
    assert done.domain == "file.example"
    assert "LB_DOMAIN in the environment is not a valid host name" in done.stderr
    assert "shadow-secret" not in done.stdout + done.stderr
    done = h.print_config(env={"DOMAIN": "a b"})
    assert done.domain == "file.example"


def test_an_environment_domain_is_used_with_no_file_at_all(h):
    assert h.print_config(env={"LB_DOMAIN": "env-only.example"}).domain == "env-only.example"
    assert h.print_config(env={"DOMAIN": "env-only.example"}).domain == "env-only.example"


SPEC = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")  # the reviewed host-name rule


def spec_accepts(value: str) -> bool:
    return SPEC.fullmatch(value) is not None and ".." not in value


DOMAIN_CANDIDATES = [
    "a",
    "A",
    "0",
    "a.b",
    "a-b",
    "a--b",
    "a.b.c.d",
    "mower.example.com",
    "MOWER.Example.COM",
    "a1",
    "1a",
    "xn--bcher-kva.example",
    "a" * 63 + "." + "b" * 63,
    "a" * 253,
    "a" * 252 + "b",
    "a" * 251 + ".b",
    "1.2.3.4",
    "localhost",
    "lawnberry.local",
    "",
    "-a",
    "a-",
    ".a",
    "a.",
    "a..b",
    "..",
    ".",
    "-",
    "a b",
    " a",
    "a ",
    "a/b",
    "/a",
    "a/",
    "../../etc",
    "a;b",
    "a&b",
    "a|b",
    "a$b",
    "a$(id)",
    "a`id`",
    "a'b",
    'a"b',
    "a\\b",
    "a*b",
    "a?b",
    "a#b",
    "a%b",
    "a:b",
    "a@b",
    "a,b",
    "a=b",
    "a+b",
    "a_b",
    "a!b",
    "a~b",
    "a<b",
    "a>b",
    "a(b)",
    "a[b]",
    "a{b}",
    "\u00e4.example",
    "a\u00e4",
    "a\tb",
    "a\nb",
    "a\rb",
    "a\x01b",
    "a" * 254,
    "a" * 253 + "b",
    "a" * 300,
    "~",
    "$HOME",
    "${HOME}",
    "*",
    "-rf",
    "--help",
    "/etc",
    "a.-b",
    "a-.b",  # label edges the reviewed rule allows
]


@pytest.mark.parametrize("value", DOMAIN_CANDIDATES, ids=[repr(v)[:40] for v in DOMAIN_CANDIDATES])
def test_the_host_name_rule_is_exactly_the_reviewed_one(h, value):
    got = h.print_config(env={"LB_DOMAIN": value}).domain
    assert got == (value if spec_accepts(value) else "")


@pytest.mark.parametrize(
    "value", [v for v in DOMAIN_CANDIDATES if "\n" not in v and "\r" not in v and v == v.strip()]
)
def test_the_same_rule_applies_to_a_value_in_the_file_in_every_quoting(h, value):
    for template in ("LB_DOMAIN={v}\n", "LB_DOMAIN='{v}'\n", 'LB_DOMAIN="{v}"\n'):
        if "'" in value and "'{v}'" in template or '"' in value and '"{v}"' in template:
            continue  # a quote inside the matching quotes is a different (unquoted) shape
        h.write_env(template.format(v=value))
        got = h.print_config().domain
        bare = template == "LB_DOMAIN={v}\n"
        # an unquoted value is trimmed; the quoted forms keep what is inside the quotes
        accepted = spec_accepts(value) and not (bare and value != value.strip())
        assert got == (value if accepted else ""), (template, value)


# ---- where the file comes from -------------------------------------------------------------


def test_the_env_file_defaults_to_the_checkout_the_script_is_in(h):
    h.write_env("LB_DOMAIN=checkout.example\n")
    done = h.print_config()
    assert done.config == {"domain": "checkout.example", "env_file": str(h.env_file)}


def test_lawnberry_env_file_names_the_file_and_beats_the_default(h):
    other = h.tmp / "elsewhere" / "service.env"
    other.parent.mkdir()
    other.write_text("LB_DOMAIN=named.example\n")
    h.write_env("LB_DOMAIN=checkout.example\n")
    done = h.print_config(env={"LAWNBERRY_ENV_FILE": str(other)})
    assert done.config == {"domain": "named.example", "env_file": str(other)}


def test_an_installed_copy_reads_only_the_file_it_is_told_about(h):
    """The unit runs /usr/local/sbin/...: no checkout around it, so no default file. A stray
    .env next to (or above) the installed script must not be picked up."""
    sbin = h.tmp / "usr" / "local" / "sbin"
    sbin.mkdir(parents=True)
    installed = sbin / "lawnberry-renew-certificates.sh"
    shutil.copy(SCRIPT, installed)
    for stray in (sbin / ".env", sbin.parent / ".env", sbin.parent.parent / ".env"):
        stray.write_text("LB_DOMAIN=stray.example\n")
    done = h.print_config(script=installed)
    assert done.config == {"domain": "", "env_file": ""}
    h.write_env("LB_DOMAIN=unit.example\n")
    done = h.print_config(script=installed, env={"LAWNBERRY_ENV_FILE": str(h.env_file)})
    assert done.config == {"domain": "unit.example", "env_file": str(h.env_file)}


def test_an_empty_lawnberry_env_file_means_no_file(h):
    h.write_env("LB_DOMAIN=checkout.example\n")
    done = h.print_config(env={"LAWNBERRY_ENV_FILE": ""})
    assert done.config == {"domain": "", "env_file": ""}
    done = h.print_config(env={"LAWNBERRY_ENV_FILE": "", "LB_DOMAIN": "pinned.example"})
    assert done.domain == "pinned.example"


@pytest.mark.parametrize("bad", ["relative/.env", "@LB_ROOT@/.env", "~/.env", ".env"])
def test_a_path_that_is_not_absolute_is_ignored_with_a_warning(h, bad):
    """An unrendered template (@LB_ROOT@, as scripts/setup.sh installs it) is not a path."""
    h.write_env("LB_DOMAIN=checkout.example\n")
    done = h.print_config(env={"LAWNBERRY_ENV_FILE": bad})
    assert "LAWNBERRY_ENV_FILE is not an absolute path" in done.stderr
    # it falls back to the file next to the script, which here is the checkout's
    assert done.config["env_file"] == str(h.env_file)


def test_a_named_file_that_does_not_exist_is_reported(h):
    gone = h.tmp / "gone.env"
    done = h.print_config(env={"LAWNBERRY_ENV_FILE": str(gone)})
    assert done.domain == "" and f"env file {gone} does not exist" in done.stderr


def test_a_named_symbolic_link_is_refused_too(h):
    target = h.tmp / "target.env"
    target.write_text("LB_DOMAIN=from-the-target.example\n")
    link = h.tmp / "link.env"
    link.symlink_to(target)
    done = h.print_config(env={"LAWNBERRY_ENV_FILE": str(link)})
    assert done.domain == "" and "symbolic link" in done.stderr


# ---- --print-config changes nothing, needs nothing ------------------------------------------


def _snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    out = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        out[str(path.relative_to(root))] = (st.st_mode, st.st_size, st.st_mtime_ns)
    return out


def test_print_config_needs_no_tool_writes_nothing_and_calls_no_syslog(h):
    h.write_env("LB_DOMAIN=quiet.example\nLB_DOMAIN=../bad\n")
    before = _snapshot(h.tmp)
    done = h.run("--print-config", path=str(h.tmp / "empty-path-dir"))  # not even logger on PATH
    assert done.returncode == 0
    assert done.config["domain"] == ""
    assert _snapshot(h.tmp) == before
    assert not h.logger_log.exists()
    # and with logger on PATH it is still not called, even while warning
    done = h.print_config()
    assert "not a valid host name" in done.stderr
    assert not h.logger_log.exists()


def test_print_config_prints_exactly_two_lines_on_stdout(h):
    h.write_env("LB_DOMAIN=a.example\nexport DOMAIN=b.example\nLB_DOMAIN=bad value\n")
    done = h.print_config()
    assert done.stdout == f"domain=\nenv_file={h.env_file}\n"
    assert done.stderr.count("[WARN]") == 2


def test_the_normal_run_logs_its_configuration_warnings_the_way_it_logs_everything(h):
    h.write_env("LB_DOMAIN=../../etc\n")
    done = h.run()  # no openssl on PATH: stops at require_cmd after the configuration
    assert done.returncode == 1
    assert "[WARN] LB_DOMAIN on line 1 of" in done.stdout and "not a valid host name" in done.stdout
    assert any(
        "daemon.warning" in call and "not a valid host name" in call for call in h.logger_calls()
    )
    assert "../../etc" not in done.stdout + done.stderr + h.logger_log.read_text()


def test_a_self_signed_directory_that_cannot_be_created_is_explained_and_nothing_else_runs(h):
    """Under the unit /etc is read-only outside three trees, so a host without /etc/lawnberry
    (scripts/setup_https.sh creates it) fails here: with a message that says so, and before the
    script touches anything. The tools it would otherwise run are stubs that record themselves."""
    ran = h.tmp / "ran"
    for name, body in (
        ("openssl", f"echo openssl >> {ran}"),
        ("nginx", f"echo nginx >> {ran}"),
        ("systemctl", f"echo systemctl >> {ran}"),
        ("mkdir", f'echo "mkdir $*" >> {ran}; exit 1'),
    ):
        stub = h.bin / name
        stub.write_text(
            f"#!/bin/sh\n{body}\nexit 0\n" if name != "mkdir" else f"#!/bin/sh\n{body}\n"
        )
        stub.chmod(0o755)
    done = h.run(env={"LAWNBERRY_ENV_FILE": ""})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "[ERR ] Cannot create /etc/lawnberry/certs/selfsigned" in done.stdout
    assert "setup_https.sh" in done.stdout
    assert any("daemon.err" in call and "Cannot create" in call for call in h.logger_calls())
    assert ran.read_text().splitlines() == ["mkdir -p /etc/lawnberry/certs/selfsigned"]


# ---- command line --------------------------------------------------------------------------


def test_help_describes_the_modes_and_changes_nothing(h):
    done = h.run("--help")
    assert done.returncode == 0
    assert "--print-config" in done.stdout and "domain=" in done.stdout
    assert not h.logger_log.exists()


@pytest.mark.parametrize(
    "args", [("--print-confg",), ("--force",), ("-x",), ("a", "b"), ("--print-config", "x")]
)
def test_an_unknown_or_extra_argument_stops_before_anything_runs(h, args):
    """A typo in --print-config must not start a real renewal run as root."""
    done = h.run(*args)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "Missing command" not in done.stdout, "it got as far as require_cmd"
    assert not h.logger_log.exists()
