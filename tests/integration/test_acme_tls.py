"""ACME / TLS integration tests: issuance, renewal, fail-closed behaviour and expiry policy.

Nothing here reaches Let's Encrypt. The only external program the service runs (``certbot``) is
replaced by a recording fake *executable* (``ACME_CERTBOT_BIN``), so the real
``ACMEService._run_certbot`` command construction and ``subprocess`` handling are exercised, and
certificates are real PEM files produced with the ``openssl`` CLI.

Not covered, because the repository has no code for it: the Let's Encrypt validation handshake,
CSR generation, chain verification, private-key/certificate matching (all done inside certbot),
and any "disable HTTPS on failure" mode (the shipped fallback is a self-signed certificate in
``scripts/renew_certificates.sh``).

Known gap, reported and deliberately not asserted: ``ACMEService.certificates`` is an in-memory
registry, so a freshly started ``acme_renew`` process starts with none and renews only a
certificate it is told about with ``--domain`` (``lawnberry-acme-renew.service`` passes none; it is
the legacy unit that the installer disables).
"""

from __future__ import annotations

import configparser
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from backend.src.cli import acme_renew
from backend.src.core import tls_status
from backend.src.core.observability import observability
from backend.src.services.acme_service import ACMEService

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

_FAKE_CERTBOT_SOURCE = r"""
import json, os, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
with open(os.environ["FAKE_CERTBOT_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\n")

rc = int(os.environ.get("FAKE_CERTBOT_RC", "0"))
if rc:
    sys.stderr.write(os.environ.get("FAKE_CERTBOT_STDERR", "certbot failed"))
    sys.exit(rc)


def opt(name):
    return argv[argv.index(name) + 1]


if "--dry-run" in argv:  # a dry run validates the flow and never touches the live certificate
    sys.exit(0)

domain = opt("-d") if argv[0] == "certonly" else opt("--cert-name")
kind = "issue" if argv[0] == "certonly" else "renew"
live = Path(opt("--config-dir")) / "live" / domain
live.mkdir(parents=True, exist_ok=True)
stage = Path(os.environ["FAKE_CERTBOT_STAGE"])
shutil.copyfile(stage / f"{domain}.{kind}.pem", live / "fullchain.pem")
"""


class FakeCertbot:
    """Recording stand-in for the ``certbot`` executable used by ``ACMEService``.

    ``stage()`` queues the certificate a successful ``certonly`` / ``renew`` run would leave under
    ``<config-dir>/live/<domain>/fullchain.pem``; ``fail()`` makes every run exit non-zero.
    """

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self._stage_dir = root / "stage"
        self._stage_dir.mkdir()
        self._log = root / "certbot-calls.jsonl"
        script = root / "fake_certbot.py"
        script.write_text(_FAKE_CERTBOT_SOURCE)
        self.executable = root / "certbot"
        self.executable.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        self.executable.chmod(0o755)
        monkeypatch.setenv("FAKE_CERTBOT_LOG", str(self._log))
        monkeypatch.setenv("FAKE_CERTBOT_STAGE", str(self._stage_dir))
        monkeypatch.setenv("ACME_CERTBOT_BIN", str(self.executable))
        monkeypatch.delenv("FAKE_CERTBOT_RC", raising=False)

    def stage(self, domain: str, *, issue: Path | None = None, renew: Path | None = None) -> None:
        if issue is not None:
            (self._stage_dir / f"{domain}.issue.pem").write_bytes(issue.read_bytes())
        if renew is not None:
            (self._stage_dir / f"{domain}.renew.pem").write_bytes(renew.read_bytes())

    def fail(self, stderr: str, returncode: int = 1) -> None:
        self._monkeypatch.setenv("FAKE_CERTBOT_RC", str(returncode))
        self._monkeypatch.setenv("FAKE_CERTBOT_STDERR", stderr)

    def succeed(self) -> None:
        self._monkeypatch.delenv("FAKE_CERTBOT_RC", raising=False)

    def calls(self) -> list[list[str]]:
        if not self._log.exists():
            return []
        return [json.loads(line) for line in self._log.read_text().splitlines()]


def make_cert(path: Path, *, days: int, cn: str) -> Path:
    """Write a real self-signed certificate valid for ``days`` days (EC key, so it is fast)."""
    subprocess.run(
        [
            "openssl", "req", "-x509",
            "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
            "-days", str(days), "-subj", f"/CN={cn}",
            "-keyout", str(path.with_suffix(".key")), "-out", str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )  # fmt: skip
    return path


def assert_expires_in(expires_at: datetime, since: datetime, days: int) -> None:
    """``expires_at`` is ``days`` after ``since`` (openssl truncates notAfter to whole seconds)."""
    assert expires_at.tzinfo is not None, "expiry must be timezone-aware"
    drift = abs((expires_at - (since + timedelta(days=days))).total_seconds())
    assert drift < 300, f"expected expiry ~{days} days after {since}, got {expires_at}"


def flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.fixture
def fake_certbot(tmp_path, monkeypatch) -> FakeCertbot:
    root = tmp_path / "bin"
    root.mkdir()
    return FakeCertbot(root, monkeypatch)


@pytest.fixture
def service(tmp_path, monkeypatch, fake_certbot) -> ACMEService:
    """An ``ACMEService`` whose every path lives under ``tmp_path`` and whose certbot is fake."""
    monkeypatch.setenv("ACME_LE_DIR", str(tmp_path / "le"))
    monkeypatch.setenv("ACME_WEBROOT", str(tmp_path / "www"))
    monkeypatch.setenv(
        "ACME_CHALLENGE_DIR", str(tmp_path / "www" / ".well-known" / "acme-challenge")
    )
    monkeypatch.setenv("LAWNBERRY_CERT_DIR", str(tmp_path / "certs"))
    monkeypatch.delenv("ACME_STAGING", raising=False)
    return ACMEService()


@pytest.fixture
def fresh_observability(monkeypatch):
    """Give the test its own error history so ACME failures never leak into other tests."""
    monkeypatch.setattr(
        observability, "_event_history", deque(maxlen=observability._event_history.maxlen)
    )


# --- a minimal nginx.conf reader (comments, quoted strings, nested blocks) -------------------

_NGINX_TOKEN = re.compile(
    r"""
    "(?:[^"\\]|\\.)*" | '(?:[^'\\]|\\.)*'   # quoted strings (may contain ; { })
    | \#[^\n]*                              # comments
    | [^\s;{}"'#]+                          # words
    | [;{}]
    """,
    re.VERBOSE,
)


def parse_nginx(text: str) -> list[dict]:
    tokens = [t for t in _NGINX_TOKEN.findall(text) if not t.startswith("#")]

    def block(i: int) -> tuple[list[dict], int]:
        items: list[dict] = []
        words: list[str] = []
        while i < len(tokens):
            tok = tokens[i]
            if tok == ";":
                if words:
                    items.append({"name": words[0], "args": words[1:], "children": None})
                words = []
            elif tok == "{":
                children, i = block(i + 1)
                items.append({"name": words[0], "args": words[1:], "children": children})
                words = []
            elif tok == "}":
                return items, i
            else:
                words.append(tok)
            i += 1
        return items, i

    return block(0)[0]


def children(node: dict, name: str) -> list[dict]:
    return [c for c in node["children"] if c["name"] == name]


# --------------------------------------------------------------------------------------------
# HTTP-01 challenge path
# --------------------------------------------------------------------------------------------


def test_acme_http01_challenge_handling(monkeypatch):
    """The HTTP-01 path is wired end to end: certbot's webroot == what port 80 serves.

    certbot writes the challenge under ``<webroot>/.well-known/acme-challenge/`` and Let's Encrypt
    fetches it over plain HTTP on port 80, *before* any HTTPS certificate exists. Three
    independently edited places must agree for that to work: the service's webroot, the challenge
    directory it uses, and the nginx port-80 server in ``config/nginx.conf``.
    """
    monkeypatch.delenv("ACME_WEBROOT", raising=False)
    monkeypatch.delenv("ACME_CHALLENGE_DIR", raising=False)
    service = ACMEService()

    http = children({"children": parse_nginx((REPO / "config" / "nginx.conf").read_text())}, "http")
    servers = children(http[0], "server")
    port80 = [s for s in servers if any(ln["args"][:1] == ["80"] for ln in children(s, "listen"))]
    assert len(port80) == 1, "expected exactly one nginx server listening on port 80"

    locations = {loc["args"][0]: loc for loc in children(port80[0], "location")}
    assert "/.well-known/acme-challenge/" in locations, (
        "port 80 must serve the ACME challenge path; without it Let's Encrypt gets a 301 to an "
        "HTTPS endpoint that has no certificate yet"
    )
    challenge = locations["/.well-known/acme-challenge/"]
    assert not children(challenge, "return"), "the challenge location must not redirect"
    assert children(challenge, "try_files")[0]["args"] == ["$uri", "=404"]
    nginx_root = Path(children(challenge, "root")[0]["args"][0])

    # Everything else on port 80 is redirected to HTTPS (the longer prefix above wins for ACME).
    redirect = children(locations["/"], "return")[0]["args"]
    assert redirect[0] == "301" and redirect[1].startswith("https://")

    # Service defaults agree with nginx: certbot -w <webroot>; files under <webroot>/.well-known/
    assert service.webroot == nginx_root
    assert service.challenge_dir == nginx_root / ".well-known" / "acme-challenge"


def test_acme_challenge_file_serving(service):
    """Challenge files land where nginx serves them and are cleaned up afterwards."""
    service.initialize()
    assert service.challenge_dir.is_dir(), "initialize() must create the challenge directory"

    token, key_auth = "evaGxfADs6pSRb2LAv9IZf17Dt3juxGJ-PCt92wr-oA", "evaGx.9jg46WB3rR_AILRVo"
    written = Path(service.create_challenge_file(token, key_auth))

    # nginx maps  GET /.well-known/acme-challenge/<token>  ->  <root>/<request path>, root = webroot
    url_path = "/.well-known/acme-challenge/" + token
    assert written == service.webroot / url_path.lstrip("/")
    assert written.read_text() == key_auth, "the body must be exactly the key authorization"
    assert service.get_challenge_content(token) == key_auth

    service.cleanup_challenge(token)
    assert not written.exists(), "a finished challenge must not stay on disk"
    assert service.get_challenge_content(token) is None
    service.cleanup_challenge(token)  # cleaning an unknown/finished token is a no-op


# --------------------------------------------------------------------------------------------
# Provisioning
# --------------------------------------------------------------------------------------------


def test_acme_certificate_provisioning(service, fake_certbot, tmp_path, monkeypatch):
    """Issuance: certbot HTTP-01/webroot against the *staging* CA; the real expiry is recorded."""
    domain = "mower.example.com"
    fake_certbot.stage(domain, issue=make_cert(tmp_path / "issued.pem", days=89, cn=domain))
    asked_at = datetime.now(UTC)

    record = service.request_certificate(domain, "ops@example.com")

    [argv] = fake_certbot.calls()
    assert argv[0] == "certonly"
    assert "--webroot" in argv and flag_value(argv, "-w") == str(service.webroot)
    assert flag_value(argv, "-d") == domain
    assert flag_value(argv, "-m") == "ops@example.com"
    assert "--agree-tos" in argv and "--non-interactive" in argv
    assert "--staging" in argv, "the staging CA must be the default (production rate limits)"
    assert "--dry-run" not in argv
    assert flag_value(argv, "--config-dir") == str(service.le_dir)
    assert flag_value(argv, "--work-dir") == str(service.le_dir / "work")
    assert flag_value(argv, "--logs-dir") == str(service.le_dir / "logs")

    live = service.le_dir / "live" / domain
    assert record["status"] == "issued" and record["staging"] is True
    assert (
        record["cert_path"] == str(live / "fullchain.pem") and Path(record["cert_path"]).is_file()
    )
    assert record["key_path"] == str(live / "privkey.pem")
    # the recorded expiry is the certificate's own notAfter (89 d), not the assumed 90 d default
    assert_expires_in(record["expires_at"], asked_at, days=89)
    assert service.get_certificate_info(domain) == record
    assert service.is_certificate_valid(domain) is True
    assert service.get_certificates_needing_renewal() == []

    # The production CA is an explicit opt-in, through the environment or the call.
    fake_certbot.stage("prod.example.com", issue=make_cert(tmp_path / "p.pem", days=89, cn="p"))
    monkeypatch.setenv("ACME_STAGING", "0")
    prod = ACMEService()
    assert prod.request_certificate("prod.example.com", "ops@example.com")["staging"] is False
    assert "--staging" not in fake_certbot.calls()[-1]
    assert prod.request_certificate("prod.example.com", "ops@example.com", staging=True)["staging"]
    assert "--staging" in fake_certbot.calls()[-1]

    # A dry run validates the flow only: it is passed to certbot and yields no usable certificate.
    dry = service.request_certificate("dry.example.com", "ops@example.com", dry_run=True)
    assert "--dry-run" in fake_certbot.calls()[-1]
    assert dry["status"] == "validated"
    assert not Path(dry["cert_path"]).exists()
    assert service.is_certificate_valid("dry.example.com") is False


# --------------------------------------------------------------------------------------------
# Renewal
# --------------------------------------------------------------------------------------------


def test_acme_automatic_renewal(service, fake_certbot, tmp_path, monkeypatch):
    """Certificates inside the 30-day window are renewed by the timer's CLI; the rest are left."""
    monkeypatch.setattr(acme_renew, "acme_service", service)
    due, fresh, dry = "due.example.com", "fresh.example.com", "dry.example.com"
    fake_certbot.stage(due, issue=make_cert(tmp_path / "due.pem", days=29, cn=due),
                       renew=make_cert(tmp_path / "due-new.pem", days=90, cn=due))  # fmt: skip
    fake_certbot.stage(fresh, issue=make_cert(tmp_path / "fresh.pem", days=31, cn=fresh))
    fake_certbot.stage(dry, issue=make_cert(tmp_path / "dry.pem", days=10, cn=dry),
                       renew=make_cert(tmp_path / "dry-new.pem", days=90, cn=dry))  # fmt: skip
    for domain in (due, fresh):
        assert service.request_certificate(domain, "ops@example.com")["status"] == "issued"
    issued_calls = len(fake_certbot.calls())

    # The renewal window is 30 days: 29 days left is due, 31 days left is not.
    assert service.get_certificates_needing_renewal() == [due]
    assert service.get_renewal_status()["renewal_needed"] == 1

    ran_at = datetime.now(UTC)
    assert acme_renew.main([]) == 0

    renew_calls = fake_certbot.calls()[issued_calls:]
    assert len(renew_calls) == 1, "only the certificate inside the window may be renewed"
    assert renew_calls[0][:3] == ["renew", "--cert-name", due]
    assert "--dry-run" not in renew_calls[0]
    renewed = service.get_certificate_info(due)
    assert renewed["status"] == "issued" and renewed["renewed_at"] >= ran_at
    assert_expires_in(renewed["expires_at"], ran_at, days=90)
    assert service.get_certificates_needing_renewal() == []
    status = service.get_renewal_status()
    assert (status["total_certificates"], status["valid_certificates"]) == (2, 2)
    assert status["renewal_needed"] == 0

    # Idempotent: the next timer run finds nothing to do and must not call certbot (rate limits).
    assert acme_renew.main([]) == 0
    assert len(fake_certbot.calls()) == issued_calls + 1

    # --dry-run is forwarded to certbot and must not replace the live certificate.
    service.request_certificate(dry, "ops@example.com")
    live_pem = service.le_dir / "live" / dry / "fullchain.pem"
    before = live_pem.read_bytes()
    expiry_before = service.get_certificate_info(dry)["expires_at"]
    assert acme_renew.main(["--dry-run"]) == 0
    assert "--dry-run" in fake_certbot.calls()[-1]
    assert fake_certbot.calls()[-1][:3] == ["renew", "--cert-name", dry]
    assert live_pem.read_bytes() == before
    assert service.get_certificate_info(dry)["expires_at"] == expiry_before

    # The systemd unit is a fresh process whose in-memory registry starts empty; a certificate
    # that exists only on disk is still renewable by naming it.
    asked_at = datetime.now(UTC)
    fresh_process = ACMEService()
    monkeypatch.setattr(acme_renew, "acme_service", fresh_process)
    assert fresh_process.list_certificates() == {}
    assert acme_renew.main(["--domain", dry]) == 0
    assert fake_certbot.calls()[-1][:3] == ["renew", "--cert-name", dry]
    assert_expires_in(fresh_process.get_certificate_info(dry)["expires_at"], asked_at, days=90)


# --------------------------------------------------------------------------------------------
# Fail closed
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acme_fail_closed_behavior(
    service, fake_certbot, fresh_observability, tmp_path, monkeypatch, caplog
):
    """Failures yield no valid certificate, keep the old one, are retried, and are visible."""
    from backend.src.main import app

    monkeypatch.setattr(acme_renew, "acme_service", service)
    due = "due.example.com"
    old_pem = make_cert(tmp_path / "due.pem", days=20, cn=due)
    fake_certbot.stage(due, issue=old_pem, renew=make_cert(tmp_path / "new.pem", days=90, cn=due))
    assert service.request_certificate(due, "ops@example.com")["status"] == "issued"
    live_pem = service.le_dir / "live" / due / "fullchain.pem"
    record_before = dict(service.get_certificate_info(due))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def acme_health() -> dict:
            resp = await client.get("/api/v2/health")
            assert resp.status_code == 200
            return resp.json()["subsystems"]["acme"]

        assert (await acme_health())["status"] == "healthy"

        # 1. A new domain whose challenge fails: no certificate is recorded or considered valid,
        #    and the operator gets certbot's own explanation.
        fake_certbot.fail("Challenge failed for domain new.example.com: 404 on port 80")
        result = service.request_certificate("new.example.com", "ops@example.com")
        assert result["status"] == "failed"
        assert "Challenge failed for domain new.example.com" in result["error"]
        assert "new.example.com" not in service.list_certificates()
        assert service.is_certificate_valid("new.example.com") is False
        assert not (service.le_dir / "live" / "new.example.com").exists()

        # 2. The renewal timer's run fails: non-zero exit (systemd marks the unit failed), the
        #    existing certificate and its record are untouched, and it is still due for the retry.
        caplog.set_level(logging.ERROR)
        assert acme_renew.main([]) == 1
        assert f"Renewal failed for {due}" in caplog.text
        assert live_pem.read_bytes() == old_pem.read_bytes()
        assert service.get_certificate_info(due) == record_before
        assert service.get_certificates_needing_renewal() == [due]

        # 3. Both failures are reported to observability and surface in /api/v2/health.
        acme = await acme_health()
        assert acme["status"] == "degraded"
        assert acme["errors_last_window"] == 2
        messages = {e["message"] for e in acme["recent_errors"]}
        assert messages == {"certbot certonly failed", "certbot renew failed"}

    # 4. The next run after the cause is fixed renews it: failure was not sticky.
    fake_certbot.succeed()
    assert acme_renew.main([]) == 0
    assert live_pem.read_bytes() != old_pem.read_bytes()
    assert service.get_certificates_needing_renewal() == []


# --------------------------------------------------------------------------------------------
# Expiry policy (what health reports about the certificate nginx is serving)
# --------------------------------------------------------------------------------------------


@pytest.fixture
def tls_env(tmp_path, monkeypatch) -> Path:
    """Hermetic ``tls_status``: no host nginx config, no host Let's Encrypt, no ./.env."""
    live = tmp_path / "letsencrypt" / "live"
    live.mkdir(parents=True)
    monkeypatch.setattr(tls_status, "NGINX_CONF_PATHS", [])
    monkeypatch.setattr(tls_status, "LE_LIVE_DIR", live)
    monkeypatch.setattr(
        tls_status, "SELF_SIGNED_FULLCHAIN", tmp_path / "selfsigned" / "fullchain.pem"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LB_DOMAIN", raising=False)
    monkeypatch.delenv("DOMAIN", raising=False)
    return live


@pytest.mark.parametrize(
    ("days_left", "level"),
    [
        (60, "healthy"),
        (31, "healthy"),
        (29, "degraded"),
        (8, "degraded"),
        (6, "critical"),
        (1, "critical"),
    ],
)  # fmt: skip
def test_acme_certificate_validation(tls_env, days_left, level):
    """Expiry level of the active certificate: >=30 d healthy, <30 d degraded, <7 d critical."""
    domain = "mower.example.com"
    (tls_env / domain).mkdir()
    pem = make_cert(tls_env / domain / "fullchain.pem", days=days_left, cn=domain)

    report = tls_status.get_tls_status()

    assert report["status"] == level
    assert report["mode"] == "letsencrypt"
    assert report["domain"] == domain, "the reported domain is the certificate's subject"
    assert report["active_cert_path"] == str(pem)
    assert report["valid_now"] is True
    assert abs(report["days_until_expiry"] - days_left) < 0.1


def test_acme_certificate_validation_expired_is_never_healthy(tls_env, monkeypatch):
    """A certificate past notAfter is critical and not valid_now (clock moved, PEM untouched)."""
    (tls_env / "mower.example.com").mkdir()
    make_cert(tls_env / "mower.example.com" / "fullchain.pem", days=60, cn="mower.example.com")

    class _Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(days=100)

    monkeypatch.setattr(tls_status, "datetime", _Later)
    report = tls_status.get_tls_status()

    assert report["status"] == "critical"
    assert report["valid_now"] is False
    assert report["days_until_expiry"] < 0


def test_acme_certificate_validation_unreadable_or_missing_is_unknown(tls_env, tmp_path):
    """No certificate / garbage certificate: status is "unknown", never "healthy"."""
    assert tls_status.get_tls_status() == {
        "status": "unknown",
        "detail": "No certificate found",
        "mode": "unknown",
    }

    (tls_env / "mower.example.com").mkdir()
    (tls_env / "mower.example.com" / "fullchain.pem").write_text(
        "-----BEGIN CERTIFICATE-----\nnope\n"
    )
    report = tls_status.get_tls_status()
    assert report["status"] == "unknown"
    assert report["valid_now"] is False
    assert report["days_until_expiry"] is None


def test_acme_certificate_validation_domain_hint_and_self_signed(tls_env, monkeypatch):
    """LB_DOMAIN selects the matching Let's Encrypt certificate; self-signed is the last resort."""
    for name in ("alpha.example.com", "mower.example.com"):
        (tls_env / name).mkdir()
        make_cert(tls_env / name / "fullchain.pem", days=60, cn=name)
    monkeypatch.setenv("LB_DOMAIN", "mower.example.com")
    assert tls_status.get_tls_status()["domain"] == "mower.example.com"

    # With no Let's Encrypt directory at all, the self-signed baseline is reported as such.
    for name in ("alpha.example.com", "mower.example.com"):
        shutil.rmtree(tls_env / name)
    selfsigned = tls_status.SELF_SIGNED_FULLCHAIN
    selfsigned.parent.mkdir()
    make_cert(selfsigned, days=365, cn="lawnberry.local")
    report = tls_status.get_tls_status()
    assert (report["mode"], report["status"], report["domain"]) == (
        "self-signed",
        "healthy",
        "lawnberry.local",
    )


# --------------------------------------------------------------------------------------------
# Renewal schedule (the maintained path: certbot + lawnberry-cert-renewal.timer)
# --------------------------------------------------------------------------------------------


def _unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # systemd keys are case-sensitive
    parser.read_string(path.read_text())
    return parser


def _runs_per_day(on_calendar: str) -> int:
    """Firings per day for the OnCalendar forms this repo uses (anything else fails loudly)."""
    spec = on_calendar.strip()
    if spec == "hourly":
        return 24
    if spec == "daily":
        return 1
    match = re.fullmatch(r"(?:\*-\*-\* )?(\d\d(?:,\d\d)*):(\d\d)(?::(\d\d))?", spec)
    if not match:
        raise AssertionError(f"OnCalendar={spec!r}: extend _runs_per_day() to understand it")
    return len(match.group(1).split(","))


def test_acme_systemd_timer_configuration():
    """The renewal timer exists, fires daily, survives power-off and runs a real script.

    ``lawnberry-acme-renew.*`` is legacy: ``systemd/install_services.sh`` and ``scripts/setup.sh``
    disable it in favour of ``lawnberry-cert-renewal.timer`` (renew <30 d, fall back to self-signed
    <7 d) plus certbot's own timers, so that is the chain checked here. A daily run gives ~30
    attempts inside the 30-day renewal window.
    """
    timer = _unit(REPO / "systemd" / "lawnberry-cert-renewal.timer")
    unit = _unit(REPO / "systemd" / "lawnberry-cert-renewal.service")

    assert _runs_per_day(timer["Timer"]["OnCalendar"]) >= 1
    # the mower may be off at the scheduled time: a missed run must be made up at next boot
    assert timer["Timer"]["Persistent"].lower() == "true"
    assert "Unit" not in timer["Timer"], "default activation: lawnberry-cert-renewal.service"
    assert timer["Install"]["WantedBy"] == "timers.target"
    assert unit["Service"]["Type"] == "oneshot"

    script_rel = re.search(r"scripts/[\w.\-]+\.sh", unit["Service"]["ExecStart"]).group(0)
    script = REPO / script_rel
    assert script.is_file(), f"the unit runs {script_rel}, which does not exist"
    if unit.has_option("Unit", "ConditionPathExists"):
        assert unit["Unit"]["ConditionPathExists"].endswith(script_rel)
    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr

    # The script's two thresholds (renew inside 30 d; give up and self-sign inside 7 d) are the
    # same ones health reports as degraded / critical and ACMEService uses as its renewal window.
    windows = re.findall(
        r'check_cert_expiry\s+"\$pem"\s+\$\(\(\s*(\d+)\*24\*3600\s*\)\)', script.read_text()
    )
    assert [int(w) for w in windows] == [30, 7]
