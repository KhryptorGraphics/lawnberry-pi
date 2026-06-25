"""Unit tests for the certbot-backed ACME service (no network/domain required)."""

import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from backend.src.services.acme_service import ACMEService


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["certbot"], returncode=returncode, stdout="", stderr=stderr
    )


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_LE_DIR", str(tmp_path / "le"))
    monkeypatch.setenv("ACME_WEBROOT", str(tmp_path / "www"))
    monkeypatch.setenv("LAWNBERRY_CERT_DIR", str(tmp_path / "certs"))
    return ACMEService()


def test_request_certificate_success(service, monkeypatch):
    future = datetime.now(UTC) + timedelta(days=89)
    monkeypatch.setattr(service, "_run_certbot", lambda args: _completed(0))
    monkeypatch.setattr(service, "_certificate_expiry", lambda domain: future)

    result = service.request_certificate("mower.example.com", "admin@example.com")

    assert result["status"] == "issued"
    assert result["expires_at"] == future
    assert service.is_certificate_valid("mower.example.com") is True


def test_request_certificate_dry_run_does_not_issue(service, monkeypatch):
    monkeypatch.setattr(service, "_run_certbot", lambda args: _completed(0))
    monkeypatch.setattr(service, "_certificate_expiry", lambda domain: None)

    result = service.request_certificate("mower.example.com", "admin@example.com", dry_run=True)

    assert result["status"] == "validated"
    # A validated (dry-run) certificate is not considered valid for serving.
    assert service.is_certificate_valid("mower.example.com") is False


def test_request_certificate_fails_closed_on_certbot_error(service, monkeypatch):
    monkeypatch.setattr(service, "_run_certbot", lambda args: _completed(1, stderr="boom"))

    result = service.request_certificate("mower.example.com", "admin@example.com")

    assert result["status"] == "failed"
    assert "boom" in result["error"]
    assert service.is_certificate_valid("mower.example.com") is False


def test_request_certificate_fails_closed_when_certbot_missing(service, monkeypatch):
    def _raise(args):
        raise FileNotFoundError("certbot")

    monkeypatch.setattr(service, "_run_certbot", _raise)
    result = service.request_certificate("mower.example.com", "admin@example.com")
    assert result["status"] == "failed"


def test_renew_unknown_certificate_returns_error(service):
    assert service.renew_certificate("unknown.example.com").get("error")


def test_revoke_unknown_certificate_returns_false(service):
    assert service.revoke_certificate("unknown.example.com") is False
