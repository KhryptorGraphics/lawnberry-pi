"""Health and liveness routes (open, for platform monitors).

``/health`` and ``/api/v2/health`` (the full report, secret-named keys redacted),
``/api/v2/health/liveness``, ``/api/v2/health/readiness`` and ``/healthz``.
"""

from fastapi import APIRouter

from ..core.auth_posture import auth_health
from ..core.health import HealthService
from ..core.secret_redaction import redact_secrets
from ..services.auth_service import primary_auth_service

router = APIRouter()


health_service = HealthService()


def _report() -> dict:
    """The health report plus the authentication state (is operator auth enforced at all).

    The key is ``operator_auth``, not ``auth``: the response sanitizer redacts any ``auth`` key.
    The report echoes whole blocks of ``config/hardware.yaml`` (``detail: <the gps block>``) to an
    anonymous caller, so a secret-named key an operator keeps in one of them (``ntrip_password``,
    ``api_token``) is dropped from the report (SEC20-01 GET audit).
    """
    report = health_service.evaluate()
    report["operator_auth"] = auth_health(login_available=primary_auth_service.login_available())
    return redact_secrets(report)


@router.get("/health")
def health_root() -> dict:
    """Return aggregated health status for platform monitoring.

    ``evaluate()`` includes the core subsystem rollup (status, message_bus,
    drivers, persistence, safety) used by platform monitors.
    """

    return _report()


@router.get("/api/v2/health")
def health_api_v2() -> dict:
    """Expose health report under the versioned API namespace."""

    return _report()


@router.get("/api/v2/health/liveness")
def health_liveness() -> dict:
    """Simple liveness probe for container orchestration."""

    return {"status": "alive"}


@router.get("/api/v2/health/readiness")
def health_readiness() -> dict:
    """Readiness probe including subsystem rollup."""

    report = health_service.evaluate()
    return redact_secrets(
        {
            "status": report.get("overall_status"),
            "timestamp": report.get("timestamp"),
            "hardware": report.get("hardware"),
            "sensor_health": report.get("sensor_health"),
            "subsystems": report.get("subsystems"),
            "dependencies": report.get("dependencies"),
        }
    )


@router.get("/healthz")
def healthz() -> dict:
    """Tiny liveness endpoint for cheap probes."""
    return {"status": "ok"}
