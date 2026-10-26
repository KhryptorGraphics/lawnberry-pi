"""The live-only HaLow units are kept as reference copies that the installer must never install."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HALOW = ROOT / "systemd" / "halow"
INSTALLER = (ROOT / "systemd" / "install_services.sh").read_text(encoding="utf-8")
CREDENTIAL = re.compile(
    r"(?i)(wpa_passphrase|psk\s*=|password\s*=|passphrase\s*=|secret\s*=|token\s*=)"
)


def _units() -> list[Path]:
    return sorted(
        p for p in HALOW.rglob("*") if p.is_file() and p.suffix in {".service", ".timer", ".conf"}
    )


def test_the_reference_copies_are_present() -> None:
    names = {p.name for p in _units()}
    assert {
        "halow-bringup.service",
        "halow-failover.service",
        "halow-lan-routes.timer",
        "wifi-halow-admin.service",
        "wificommand.service",
        "10-halow-ap-mode.conf",
        "thor-upload.conf",
    } <= names


def test_the_installer_does_not_install_site_specific_units() -> None:
    # They describe one site's radio and addresses; installing them elsewhere would break the host.
    assert "systemd/halow" not in INSTALLER
    for p in _units():
        assert p.name not in INSTALLER, f"install_services.sh mentions {p.name}"


def test_no_credential_is_committed_with_the_units() -> None:
    for p in _units() + sorted((HALOW / "sbin").glob("*.sh")):
        assert not CREDENTIAL.search(p.read_text(encoding="utf-8")), f"credential-like text in {p}"
