"""Which authentication relaxations are active, parsed strictly, and said out loud at start-up.

Switches that decide whether the backend asks for credentials, or exposes test hooks, at all:

``OPERATOR_AUTH_REQUIRED``
    Fail-closed parse, ONE definition shared with the installer
    (``credential_policy.parse_operator_auth_flag``, also used by ``scripts/provision_env.py``).
    Only the explicit false tokens ``0``, ``false``, ``no`` and ``off`` (any case, surrounding
    whitespace ignored) disable operator authentication. Unset, an explicit true token (``1``,
    ``true``, ``yes``, ``on``) and every ambiguous value (empty, ``2``, ``disabled``, a typo) keep
    it ENFORCED: this used to be ``!= "1"``, so ``true``, ``TRUE`` and an empty
    ``OPERATOR_AUTH_REQUIRED=`` line in ``.env`` (which overrides the unit) silently opened every
    write route.

``LAWN_BERRY_SIM_OPEN_AUTH``
    ``SIM_MODE=1`` alone no longer makes the login accept any non-empty credential or admits any
    WebSocket client as an operator. That needs this explicit second opt-in as well, and it is
    never honoured when ``LAWNBERRY_ENV=production``. ``tests/conftest.py`` sets it; a developer
    running a throw-away simulator may.

``LAWNBERRY_ENABLE_DEBUG_ROUTES``
    The ``/api/v2/debug/*`` injectors (GPS position, geofence, ToF distance, tilt) exist only
    while ``SIM_MODE=1`` or this is set to a true token; otherwise they answer 404, before any
    authentication check. Default off. Operator authentication is required on top, always.

Imports only the standard library and ``credential_policy`` (itself standard library only).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .credential_policy import AUTH_FLAG_TRUE_TOKENS, parse_operator_auth_flag

logger = logging.getLogger(__name__)

AUTH_FLAG = "OPERATOR_AUTH_REQUIRED"
SIM_OPEN_AUTH_FLAG = "LAWN_BERRY_SIM_OPEN_AUTH"
DEBUG_ROUTES_FLAG = "LAWNBERRY_ENABLE_DEBUG_ROUTES"

TRUE_TOKENS = AUTH_FLAG_TRUE_TOKENS


@dataclass(frozen=True)
class OperatorAuthFlag:
    """``OPERATOR_AUTH_REQUIRED`` as parsed: ``required`` is what the backend acts on."""

    required: bool
    raw: str | None
    recognised: bool  # False only for a value that is neither a true nor a false token


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def read_operator_auth_flag(environ: Mapping[str, str] | None = None) -> OperatorAuthFlag:
    raw = _env(environ).get(AUTH_FLAG)
    required, recognised = parse_operator_auth_flag(raw)
    return OperatorAuthFlag(required=required, raw=raw, recognised=recognised)


def operator_auth_required(environ: Mapping[str, str] | None = None) -> bool:
    """``False`` only when ``OPERATOR_AUTH_REQUIRED`` is an explicit false token."""
    return read_operator_auth_flag(environ).required


def sim_mode(environ: Mapping[str, str] | None = None) -> bool:
    return _env(environ).get("SIM_MODE", "0") == "1"


def in_production(environ: Mapping[str, str] | None = None) -> bool:
    return _env(environ).get("LAWNBERRY_ENV", "").strip().lower() == "production"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_TOKENS


def sim_open_auth_requested(environ: Mapping[str, str] | None = None) -> bool:
    """The opt-in variable is set to a true token (whether or not it can take effect)."""
    return _truthy(_env(environ).get(SIM_OPEN_AUTH_FLAG))


def sim_open_auth_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Simulated runs may accept any non-empty credential: needs SIM_MODE=1 AND the opt-in,
    and never in a production environment."""
    return sim_mode(environ) and sim_open_auth_requested(environ) and not in_production(environ)


def debug_routes_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True when the ``/debug/*`` injectors exist: SIM_MODE=1 or the debug-routes flag is set."""
    return sim_mode(environ) or _truthy(_env(environ).get(DEBUG_ROUTES_FLAG))


def under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def configured_bind_host(
    argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None
) -> str | None:
    """The address the server was told to bind, as far as this process can tell.

    Reads ``--host X`` / ``--host=X`` from the command line and ``UVICORN_HOST``; a uvicorn
    command line without either binds uvicorn's default, the loopback. ``None`` means the
    launcher is something else (a script calling ``uvicorn.run``, gunicorn) and the address
    cannot be known from here.
    """
    args = list(sys.argv if argv is None else argv)
    for index, arg in enumerate(args):
        if arg == "--host" and index + 1 < len(args):
            return args[index + 1].strip()
        if arg.startswith("--host="):
            return arg.split("=", 1)[1].strip()
    from_env = _env(environ).get("UVICORN_HOST")
    if from_env and from_env.strip():
        return from_env.strip()
    if any("uvicorn" in arg for arg in args[:3]):
        return "127.0.0.1"
    return None


def is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    text = host.strip().strip("[]").lower()
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False  # "0.0.0.0", "::", a hostname we cannot resolve here: treat as exposed


def log_auth_posture(
    *,
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    log: logging.Logger | None = None,
    pytest_process: bool | None = None,
) -> None:
    """Say, at start-up, every way this process is configured to skip authentication.

    CRITICAL is reserved for configurations that leave a mower reachable without credentials
    or that run simulated safety confirmations somewhere real actuators could be attached.
    ``pytest_process`` overrides the test-process detection (the unit tests of this function
    run inside pytest and still need the non-pytest branches).
    """
    out = log or logger
    flag = read_operator_auth_flag(environ)
    simulated = sim_mode(environ)

    if not flag.required:
        if simulated:
            out.warning(
                "%s=%r: operator authentication is DISABLED (SIM_MODE=1): every write route "
                "is open to anyone who can reach this port",
                AUTH_FLAG,
                flag.raw,
            )
        else:
            out.critical(
                "%s=%r DISABLES operator authentication outside SIM_MODE: every write route "
                "(tractor, drive, blade, missions, AI, navigation, E-stop clear) is open to "
                "anyone who can reach this port. The unit ships 1; <root>/.env "
                "(EnvironmentFile) overrides it, so look there.",
                AUTH_FLAG,
                flag.raw,
            )
    elif not flag.recognised:
        out.warning(
            "%s=%r is not a recognised boolean: operator authentication stays ENFORCED "
            "(only 0, false, no or off disable it)",
            AUTH_FLAG,
            flag.raw,
        )

    if _truthy(_env(environ).get(DEBUG_ROUTES_FLAG)) and not simulated:
        out.warning(
            "%s is set: the /api/v2/debug/* injectors (GPS, geofence, ToF, tilt) are reachable "
            "outside SIM_MODE (operator authentication is still required)",
            DEBUG_ROUTES_FLAG,
        )

    if under_pytest() if pytest_process is None else pytest_process:
        return

    host = configured_bind_host(argv, environ)
    exposed = not is_loopback_host(host)
    where = "an unknown address" if host is None else host

    if simulated and in_production(environ):
        out.critical(
            "SIM_MODE=1 in a LAWNBERRY_ENV=production process: hardware and safety "
            "confirmations are SIMULATED. The unit ships SIM_MODE=0; <root>/.env "
            "(EnvironmentFile) overrides it."
        )
    elif simulated and exposed:
        out.critical(
            "SIM_MODE=1 while binding %s (not loopback): hardware and safety confirmations are "
            "SIMULATED and the API is reachable from the network. Never attach real actuators "
            "to this process.",
            where,
        )

    if sim_open_auth_requested(environ):
        if in_production(environ):
            out.critical(
                "%s is set but IGNORED in a production environment: SIM_MODE never opens the "
                "login there",
                SIM_OPEN_AUTH_FLAG,
            )
        elif not simulated:
            out.warning("%s has no effect without SIM_MODE=1", SIM_OPEN_AUTH_FLAG)
        elif exposed:
            out.critical(
                "%s=1 with SIM_MODE=1 while binding %s (not loopback): ANY non-empty "
                "credential logs in and any WebSocket client is an operator.",
                SIM_OPEN_AUTH_FLAG,
                where,
            )
        else:
            out.warning(
                "%s=1: any non-empty credential logs in (simulator bound to %s)",
                SIM_OPEN_AUTH_FLAG,
                where,
            )


def auth_health(*, login_available: bool, environ: Mapping[str, str] | None = None) -> dict:
    """The authentication state ``/health`` reports (booleans only, nothing secret)."""
    flag = read_operator_auth_flag(environ)
    return {
        "operator_auth_required": flag.required,
        "operator_auth_flag_ambiguous": not flag.recognised,
        "operator_login_available": bool(login_available),
        "simulation_open_auth": sim_open_auth_enabled(environ),
        "debug_routes_enabled": debug_routes_enabled(environ),
    }
