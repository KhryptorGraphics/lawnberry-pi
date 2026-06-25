"""ACME certificate renewal entrypoint.

Invoked by ``lawnberry-acme-renew.timer``. Renews any managed certificate that
is within the renewal window via the system ``certbot``. Use ``--dry-run`` to
validate the renewal flow without contacting the ACME server.
"""

from __future__ import annotations

import argparse
import logging

from ..services.acme_service import acme_service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Renew ACME/TLS certificates")
    parser.add_argument("--dry-run", action="store_true", help="Validate renewal without issuing")
    parser.add_argument("--domain", help="Renew a single domain instead of all due domains")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[acme-renew] %(message)s")
    acme_service.initialize()

    if args.domain:
        domains = [args.domain]
    else:
        domains = acme_service.get_certificates_needing_renewal()

    if not domains:
        logging.info("No certificates require renewal.")
        return 0

    failures = 0
    for domain in domains:
        logging.info("Renewing certificate for %s", domain)
        result = acme_service.renew_certificate(domain, dry_run=args.dry_run)
        if result.get("error"):
            logging.error("Renewal failed for %s: %s", domain, result["error"])
            failures += 1
        else:
            logging.info("Renewed %s (expires %s)", domain, result.get("expires_at"))

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
