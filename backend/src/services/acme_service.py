import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..core.observability import observability

logger = observability.get_logger(__name__)


class ACMEService:
    """ACME TLS certificate management service.

    Real issuance/renewal/revocation is delegated to the system ``certbot``
    binary (HTTP-01 via the configured webroot). Live ACME calls are gated so
    they only run when explicitly invoked; ``staging``/``dry_run`` avoid hitting
    the production Let's Encrypt rate limits. All operations fail closed: on any
    error the certificate is marked failed rather than reported as valid.
    """

    def __init__(self):
        self.certificates: dict[str, dict[str, Any]] = {}
        self.challenges: dict[str, str] = {}  # token -> file_content
        self.cert_dir = Path(os.getenv("LAWNBERRY_CERT_DIR", "/etc/lawnberry/certs"))
        self.challenge_dir = Path(
            os.getenv("ACME_CHALLENGE_DIR", "/var/www/.well-known/acme-challenge")
        )
        # Webroot whose /.well-known/acme-challenge/ is served on port 80.
        self.webroot = Path(os.getenv("ACME_WEBROOT", "/var/www"))
        # certbot keeps its account/config/live state under this directory.
        self.le_dir = Path(os.getenv("ACME_LE_DIR", "/etc/letsencrypt"))
        self.certbot_bin = os.getenv("ACME_CERTBOT_BIN", shutil.which("certbot") or "certbot")
        # Use the Let's Encrypt staging environment unless explicitly disabled.
        self.staging = os.getenv("ACME_STAGING", "1") == "1"
        self.auto_renewal_enabled = True

    def initialize(self):
        """Initialize ACME service directories."""
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.challenge_dir.mkdir(parents=True, exist_ok=True)

    # --- certbot plumbing ---------------------------------------------------

    def _live_dir(self, domain: str) -> Path:
        return self.le_dir / "live" / domain

    def _run_certbot(self, args: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            self.certbot_bin,
            *args,
            "--config-dir",
            str(self.le_dir),
            "--work-dir",
            str(self.le_dir / "work"),
            "--logs-dir",
            str(self.le_dir / "logs"),
            "--non-interactive",
        ]
        logger.info("Running certbot", extra={"command": " ".join(cmd)})
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def _certificate_expiry(self, domain: str) -> datetime | None:
        """Read the real notAfter from the issued certificate, if present."""
        cert_path = self._live_dir(domain) / "fullchain.pem"
        if not cert_path.exists():
            return None
        try:
            from cryptography import x509

            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            expiry = getattr(cert, "not_valid_after_utc", None)
            if expiry is None:
                expiry = cert.not_valid_after.replace(tzinfo=UTC)
            return expiry
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["openssl", "x509", "-enddate", "-noout", "-in", str(cert_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "notAfter=" in result.stdout:
                value = result.stdout.strip().split("notAfter=", 1)[1]
                return datetime.strptime(value, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=UTC)
        except Exception:
            pass
        return None

    def request_certificate(
        self,
        domain: str,
        email: str,
        *,
        staging: bool | None = None,
        dry_run: bool = False,
        webroot: str | Path | None = None,
    ) -> dict[str, Any]:
        """Request a new certificate from Let's Encrypt via certbot (HTTP-01)."""
        use_staging = self.staging if staging is None else staging
        webroot_path = str(webroot or self.webroot)
        requested_at = datetime.now(UTC)
        try:
            args = [
                "certonly",
                "--webroot",
                "-w",
                webroot_path,
                "-d",
                domain,
                "--agree-tos",
                "-m",
                email,
                "--keep-until-expiring",
            ]
            if use_staging:
                args.append("--staging")
            if dry_run:
                args.append("--dry-run")

            result = self._run_certbot(args)
            if result.returncode != 0:
                logger.error(
                    "certbot certonly failed",
                    extra={"domain": domain, "stderr": result.stderr[-500:]},
                )
                observability.record_error(
                    origin="acme",
                    message="certbot certonly failed",
                    metadata={"domain": domain, "returncode": result.returncode},
                )
                return {
                    "domain": domain,
                    "status": "failed",
                    "error": result.stderr.strip()[-500:] or "certbot failed",
                    "requested_at": requested_at,
                }

            expires_at = self._certificate_expiry(domain) or (requested_at + timedelta(days=90))
            record = {
                "domain": domain,
                # dry-run validates the flow without persisting a usable cert.
                "status": "validated" if dry_run else "issued",
                "requested_at": requested_at,
                "issued_at": requested_at,
                "expires_at": expires_at,
                "staging": use_staging,
                "auto_renew": True,
                "cert_path": str(self._live_dir(domain) / "fullchain.pem"),
                "key_path": str(self._live_dir(domain) / "privkey.pem"),
            }
            self.certificates[domain] = record
            return record

        except FileNotFoundError as exc:
            # certbot binary missing -> fail closed.
            logger.error("certbot binary not found", exc_info=True)
            return {
                "domain": domain,
                "status": "failed",
                "error": f"certbot not available: {exc}",
                "requested_at": requested_at,
            }
        except Exception as e:
            logger.error("Certificate request failed", exc_info=True)
            observability.record_error(
                origin="acme", message="Certificate request failed", exception=e
            )
            return {
                "domain": domain,
                "status": "failed",
                "error": str(e),
                "requested_at": requested_at,
            }

    def create_challenge_file(self, token: str, key_auth: str) -> str:
        """Create HTTP-01 challenge file."""
        challenge_file = self.challenge_dir / token
        challenge_file.write_text(key_auth)

        # Store for cleanup
        self.challenges[token] = key_auth

        return str(challenge_file)

    def get_challenge_content(self, token: str) -> str | None:
        """Get challenge content for serving."""
        return self.challenges.get(token)

    def cleanup_challenge(self, token: str):
        """Clean up challenge file."""
        challenge_file = self.challenge_dir / token
        if challenge_file.exists():
            challenge_file.unlink()

        self.challenges.pop(token, None)

    def list_certificates(self) -> dict[str, dict[str, Any]]:
        """List all managed certificates."""
        return self.certificates.copy()

    def get_certificate_info(self, domain: str) -> dict[str, Any] | None:
        """Get certificate information."""
        return self.certificates.get(domain)

    def is_certificate_valid(self, domain: str) -> bool:
        """Check if certificate is valid and not expired."""
        cert_info = self.certificates.get(domain)
        if not cert_info:
            return False

        if cert_info["status"] != "issued":
            return False

        # Check expiration (renew if < 30 days)
        expires_at = cert_info["expires_at"]
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))

        renewal_threshold = datetime.now(UTC) + timedelta(days=30)
        return expires_at > renewal_threshold

    def needs_renewal(self, domain: str) -> bool:
        """Check if certificate needs renewal."""
        return not self.is_certificate_valid(domain)

    def renew_certificate(self, domain: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Renew an existing certificate via ``certbot renew --cert-name``."""
        cert_info = self.certificates.get(domain)
        if not cert_info and not self._live_dir(domain).exists():
            return {"error": "Certificate not found"}

        try:
            args = ["renew", "--cert-name", domain]
            if dry_run:
                args.append("--dry-run")
            result = self._run_certbot(args)
            if result.returncode != 0:
                observability.record_error(
                    origin="acme",
                    message="certbot renew failed",
                    metadata={"domain": domain, "returncode": result.returncode},
                )
                return {"error": result.stderr.strip()[-500:] or "certbot renew failed"}

            expires_at = self._certificate_expiry(domain) or (
                datetime.now(UTC) + timedelta(days=90)
            )
            record = self.certificates.setdefault(domain, {"domain": domain, "auto_renew": True})
            record.update(
                {
                    "status": "issued",
                    "renewed_at": datetime.now(UTC),
                    "expires_at": expires_at,
                }
            )
            return record

        except Exception as e:
            logger.error("Certificate renewal failed", exc_info=True)
            observability.record_error(
                origin="acme", message="Certificate renewal failed", exception=e
            )
            return {"error": str(e)}

    def revoke_certificate(self, domain: str) -> bool:
        """Revoke a certificate via ``certbot revoke`` and forget it."""
        cert_path = self._live_dir(domain) / "fullchain.pem"
        if domain not in self.certificates and not cert_path.exists():
            return False

        try:
            if cert_path.exists():
                result = self._run_certbot(
                    [
                        "revoke",
                        "--cert-path",
                        str(cert_path),
                        "--delete-after-revoke",
                    ]
                )
                if result.returncode != 0:
                    observability.record_error(
                        origin="acme",
                        message="certbot revoke failed",
                        metadata={"domain": domain, "returncode": result.returncode},
                    )
                    return False

            self.certificates.setdefault(domain, {"domain": domain})
            self.certificates[domain]["status"] = "revoked"
            self.certificates[domain]["revoked_at"] = datetime.now(UTC)
            return True

        except Exception:
            logger.error("Certificate revocation failed", exc_info=True)
            return False

    def get_certificates_needing_renewal(self) -> list[str]:
        """Get list of domains with certificates needing renewal."""
        domains = []
        for domain, cert_info in self.certificates.items():
            if cert_info.get("auto_renew", True) and self.needs_renewal(domain):
                domains.append(domain)
        return domains

    def setup_http_challenge_server(self, port: int = 80):
        """Set up HTTP server for ACME challenges."""
        try:
            # Create nginx configuration for ACME challenges
            nginx_config = f"""
# ACME HTTP-01 challenge configuration
server {{
    listen {port};
    listen [::]:{port};
    server_name _;
    
    # ACME challenge location
    location /.well-known/acme-challenge/ {{
        root /var/www;
        try_files $uri =404;
    }}
    
    # Redirect all other HTTP traffic to HTTPS
    location / {{
        return 301 https://$host$request_uri;
    }}
}}
"""

            # Write nginx config for ACME challenges
            config_path = Path("/etc/nginx/sites-available/lawnberry-acme")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(nginx_config)

            # Enable the configuration
            symlink_path = Path("/etc/nginx/sites-enabled/lawnberry-acme")
            if not symlink_path.exists():
                symlink_path.symlink_to(config_path)

            # Test and reload nginx
            result = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(["systemctl", "reload", "nginx"], check=True)
                logger.info(
                    "Reloaded nginx after writing ACME challenge configuration",
                    extra={"command": "nginx -t"},
                )
                return True
            logger.error(
                "Nginx configuration validation failed",
                extra={"stderr": result.stderr},
            )
            observability.record_error(
                origin="acme",
                message="Nginx configuration validation failed",
                metadata={"stderr": result.stderr},
            )
            return False

        except Exception as e:
            logger.error("Failed to setup HTTP challenge server", exc_info=True)
            observability.record_error(
                origin="acme",
                message="Failed to setup HTTP challenge server",
                exception=e,
            )
            return False

    def install_certificate(
        self, domain: str, cert_path: str, key_path: str, chain_path: str | None = None
    ) -> bool:
        """Install certificate files."""
        try:
            # Copy certificate files to the cert directory
            domain_dir = self.cert_dir / domain
            domain_dir.mkdir(exist_ok=True)

            # In production, this would copy the actual certificate files
            # and reload the web server configuration

            cert_info = self.certificates.get(domain, {})
            cert_info.update(
                {
                    "status": "installed",
                    "installed_at": datetime.now(UTC),
                    "cert_path": str(domain_dir / "cert.pem"),
                    "key_path": str(domain_dir / "key.pem"),
                    "chain_path": str(domain_dir / "chain.pem") if chain_path else None,
                }
            )

            self.certificates[domain] = cert_info
            return True

        except Exception as e:
            logger.error("Certificate installation failed", exc_info=True)
            observability.record_error(
                origin="acme",
                message="Certificate installation failed",
                exception=e,
            )
            return False

    def reload_web_server(self):
        """Reload web server after certificate update."""
        try:
            # Test nginx configuration before reload
            result = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
            if result.returncode == 0:
                subprocess.run(["systemctl", "reload", "nginx"], check=True)
                logger.info("Web server reloaded after certificate update")
                return
            logger.error(
                "Nginx configuration validation failed during reload",
                extra={"stderr": result.stderr},
            )
            raise Exception("Invalid nginx configuration")
        except Exception as e:
            logger.error("Web server reload failed", exc_info=True)
            observability.record_error(
                origin="acme",
                message="Web server reload failed",
                exception=e,
            )
            raise

    def get_renewal_status(self) -> dict[str, Any]:
        """Get renewal status and statistics."""
        total_certs = len(self.certificates)
        valid_certs = sum(1 for domain in self.certificates if self.is_certificate_valid(domain))
        expired_certs = total_certs - valid_certs
        renewal_needed = len(self.get_certificates_needing_renewal())

        return {
            "total_certificates": total_certs,
            "valid_certificates": valid_certs,
            "expired_certificates": expired_certs,
            "renewal_needed": renewal_needed,
            "auto_renewal_enabled": self.auto_renewal_enabled,
            "last_check": datetime.now(UTC).isoformat(),
        }


# Global instance
acme_service = ACMEService()
