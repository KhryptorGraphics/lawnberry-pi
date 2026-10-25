"""Start and stop the Thor link inside the backend (W7). Default OFF; fails closed.

``config/thor_link.yaml`` (``LAWNBERRY_CONFIG_DIR`` overrides the directory) holds the flag
and every setting; a missing file means *disabled*. ``start_thor_link`` builds, in order:

1. the settings (unknown keys are errors: a typo in a safety setting must not be ignored);
2. the preconditions: ``tractor.enabled``, the tractor safety monitor running (tilt cutoff
   and watchdog heartbeat), a usable ``THOR_LINK_KEY`` (outside ``SIM_MODE`` there is no
   fallback to the public sim key), and, outside ``SIM_MODE``, a geofence;
3. ``ThorPlanFollower`` (the only consumer of the plan, driving it through the navigation
   loop and the tractor interlocks) and ``ThorLinkService`` (authenticated UDP link, safe
   hold), cross-wired: the follower builds the uplink, the link supplies ``current_plan()``.

Anything missing raises ``ThorLinkStartError`` and the caller keeps the mower on local
control only. The link never starts half-configured and never runs unauthenticated.

The link key never goes in the file, and nothing the file holds goes into an error. A setting
whose name looks like secret material (``key``, ``psk``, ``password``, ``token``...) is refused
by name, and every error is built from where a value was rejected and why, never from the value:
pydantic prints rejected inputs (and, for unknown keys, a prefix and suffix of them) and
PyYAML prints the offending source lines, both of which would put a pasted key into the journal
at CRITICAL (``main.py`` logs the exception, ``app.state.thor_link_error`` keeps it).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..core.simulation import is_simulation_mode
from ..models.geofence import Geofence, LatLng
from ..models.thor_link import MAX_UPLINK_HZ, UPLINK_HZ, LinkKeyError
from ..nav.geofence_validator import GeofenceShape, build_shape
from .thor_link_service import (
    KEY_SECRET_NAME,
    ThorLinkService,
    parse_allowed_sources,
    resolve_link_key,
)
from .thor_plan_follower import FollowerConfig, GnssProvider, ThorPlanFollower

if TYPE_CHECKING:
    from .navigation_service import NavigationService
    from .tractor_service import TractorControlService

logger = logging.getLogger(__name__)

CONFIG_FILE = "thor_link.yaml"


class ThorLinkStartError(RuntimeError):
    """The Thor link was asked for but cannot be started safely: it stays OFF."""


class ThorLinkConfigError(ThorLinkStartError):
    """``thor_link.yaml`` exists but cannot be read or does not validate."""


# Words that make a setting name read as secret material, matched in the lowercased name with
# everything but letters and digits removed (``THOR-link-Key`` -> ``thorlinkkey``). No real setting
# contains one (a test holds that), so this can only refuse a name the file has no use for.
_SECRET_NAME_WORDS = ("key", "secret", "psk", "password", "passwd", "passphrase", "token", "hmac")
# A setting name worth echoing back: at most 24 characters (the longest real one has 18), made of
# what names are made of. Anything else in the name position (a pasted key, a passphrase) is text
# the file should not have had there, and is not repeated.
_SHOWN_NAME = re.compile(r"[A-Za-z0-9_.\-\[\]']{1,24}")
_NAME_HIDDEN = "<name not shown>"


def _looks_like_secret_name(name: object) -> bool:
    if not isinstance(name, str):
        return False
    squashed = re.sub(r"[^a-z0-9]", "", name.lower())
    return squashed == "pass" or any(word in squashed for word in _SECRET_NAME_WORDS)


def _shown(name: object) -> str:
    """``name`` as it may appear in a message: a short setting-like name, else a placeholder."""
    if isinstance(name, str) and _SHOWN_NAME.fullmatch(name):
        return name
    return _NAME_HIDDEN


def _refuse_secret_names(data: Any) -> Any:
    """Raise ``ValueError`` naming (only) the settings in ``data`` that look like secret material.

    Whatever the setting holds is not looked at, let alone repeated: the name is the mistake.
    """
    if isinstance(data, dict):
        flagged = sorted({_shown(name) for name in data if _looks_like_secret_name(name)})
        if flagged:
            many = len(flagged) > 1
            raise ValueError(
                f"{'settings' if many else 'setting'} {', '.join(repr(n) for n in flagged)} "
                f"{'look' if many else 'looks'} like secret material: the link key belongs in "
                f"{KEY_SECRET_NAME} (environment) or config/secrets.json, never in this file"
            )
    return data


def _where(loc: tuple[Any, ...]) -> str:
    """A pydantic error location (``geofence.boundary[0]``) with every name vetted by ``_shown``."""
    out = ""
    for item in loc:
        if isinstance(item, int) and not isinstance(item, bool):
            out += f"[{item}]" if 0 <= item < 1000 else "[?]"
        else:
            out += f"{'.' if out else ''}{_shown(item)}"
    return out


def describe_validation_error(exc: ValidationError) -> str:
    """What a ``ValidationError`` says, as ``where: why`` pairs and with no input in it.

    Only the location and pydantic's own message are used: not ``input``, not ``url`` and not
    ``ctx`` (which can carry the exception a validator raised, and so a value it formatted).
    """
    problems = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False):
        where = _where(error["loc"])
        why = error["msg"].removeprefix("Value error, ")
        problems.append(f"{where}: {why}" if where else why)
    return "; ".join(problems)


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """Where PyYAML gave up, and nothing it quotes: its message holds the offending source line."""
    mark = getattr(exc, "problem_mark", None)
    where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
    return f"not valid YAML{where}"


def _ipv4(value: str, what: str) -> str:
    try:
        ipaddress.IPv4Address(value)
        valid = True
    except ValueError:
        valid = False
    if not valid:  # raised outside the ``except``, without the value: nothing chains to it
        raise ValueError(f"{what} is not an IPv4 address literal")
    return value


class FenceSettings(BaseModel):
    """The mowing boundary the Pi enforces on every plan, in ``[lat, lon]`` vertex lists."""

    # ``hide_input_in_errors``: even a bare ``FenceSettings(...)`` must not print what it rejected.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    boundary: list[tuple[float, float]] = Field(min_length=3)
    no_go: list[list[tuple[float, float]]] = Field(default_factory=list)
    # Pulled in from the boundary for every check. UNVALIDATED: size it from the measured
    # stopping distance and the GNSS error (audit M-43, M-51).
    margin_m: float = Field(0.5, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _no_secret_material(cls, data: Any) -> Any:
        return _refuse_secret_names(data)

    @field_validator("no_go")
    @classmethod
    def _zones_are_polygons(cls, zones: list[list[tuple[float, float]]]):
        if any(len(zone) < 3 for zone in zones):
            raise ValueError("every no_go zone needs at least 3 vertices")
        return zones


class ThorLinkSettings(BaseModel):
    """Everything configurable about the link; the defaults are OFF and closed."""

    # ``hide_input_in_errors``: even a bare ``ThorLinkSettings(...)`` must not print what it
    # rejected (a link key pasted into the wrong setting).
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    enabled: bool = False
    # Where uplinks go (IPv4 literal: a hostname lookup must never block the link loop).
    thor_host: str | None = None
    thor_port: int = Field(47101, ge=1, le=65535)
    # Where the Pi listens. Loopback by default so a forgotten setting is closed, not open:
    # set it to the Pi's HaLow address (audit RS-2).
    bind_host: str = "127.0.0.1"
    bind_port: int = Field(47101, ge=0, le=65535)
    # Source addresses a datagram must come from. Unset: the Thor's address only;
    # "any" switches the allowlist off (HMAC is then the only control). Entries are parsed
    # strictly (``parse_allowed_sources``): host bits, /0 and anything wider than one LAN
    # (IPv4 shorter than /16) are errors, the last unless ``allow_wide_sources`` says it is meant.
    allowed_sources: list[str] | Literal["any"] | None = None
    allow_wide_sources: bool = False
    # Timings are finite and positive: an infinite timeout would mean "never time out".
    timeout_s: float = Field(0.5, gt=0, allow_inf_nan=False)
    plan_max_age_s: float = Field(1.0, gt=0, allow_inf_nan=False)
    uplink_hz: float = Field(UPLINK_HZ, gt=0, le=MAX_UPLINK_HZ)
    neutral_timeout_s: float = Field(1.0, gt=0, allow_inf_nan=False)
    # The Thor can only ask for the blade; the Pi grants it only if this is true.
    allow_blade: bool = False
    # UNVALIDATED: equals the mission default (share of NavigationService.max_speed).
    speed_percent: int = Field(50, ge=1, le=100)
    require_fix: list[Literal["gps", "dgps", "rtk_float", "rtk_fixed"]] = Field(
        default_factory=lambda: ["rtk_fixed"], min_length=1
    )
    max_position_age_s: float = Field(1.0, gt=0, allow_inf_nan=False)
    geofence: FenceSettings | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_secret_material(cls, data: Any) -> Any:
        return _refuse_secret_names(data)

    @field_validator("thor_host", "bind_host")
    @classmethod
    def _addresses(cls, value: str | None, info: Any):
        return value if value is None else _ipv4(value, info.field_name)

    @model_validator(mode="after")
    def _sources_say_what_they_mean(self) -> ThorLinkSettings:
        # The same strict parser the link uses: host bits, /0 and too-wide networks are errors
        # here, at config time, not a silently widened allowlist.
        if isinstance(self.allowed_sources, list):
            parse_allowed_sources(self.allowed_sources, allow_wide=self.allow_wide_sources)
        return self

    @model_validator(mode="after")
    def _enabled_needs_a_thor(self) -> ThorLinkSettings:
        if self.enabled and self.thor_host is None:
            raise ValueError("thor_host is required when enabled is true")
        return self

    def sources(self) -> list[str] | None:
        """The allowlist handed to the link: the Thor's address unless overridden."""
        if self.allowed_sources == "any" or self.thor_host is None:
            return None
        if self.allowed_sources is None:
            return [self.thor_host]
        return list(self.allowed_sources)


def load_thor_link_settings(config_dir: str | os.PathLike[str] | None = None) -> ThorLinkSettings:
    """Read ``thor_link.yaml``; no file means the defaults (disabled).

    Every error says where the file is wrong and why, and quotes nothing the file contains: see
    the module docstring. The errors are raised outside the ``except`` blocks, so no exception
    that holds the offending text (a pydantic ``ValidationError``, a PyYAML scanner error) is
    chained to them, for a traceback or an error reporter to print.
    """
    base = Path(config_dir or os.getenv("LAWNBERRY_CONFIG_DIR", "config"))
    path = base / CONFIG_FILE
    why: str
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return ThorLinkSettings()
    except yaml.YAMLError as exc:
        why = _yaml_problem(exc)
    except UnicodeDecodeError:
        why = "not valid UTF-8 text"
    except OSError as exc:
        why = f"cannot be read ({exc.strerror or type(exc).__name__})"
    else:
        try:
            return ThorLinkSettings.model_validate(raw)
        except ValidationError as exc:
            problems = describe_validation_error(exc)
        raise ThorLinkConfigError(f"invalid {path}: {problems}")
    raise ThorLinkConfigError(f"cannot read {path}: {why}")


def build_fence(fence: FenceSettings) -> GeofenceShape:
    """The enforced fence: boundary pulled in by ``margin_m``, no-go zones excluded."""
    try:
        boundary = [LatLng(latitude=lat, longitude=lon) for lat, lon in fence.boundary]
        zones = [[LatLng(latitude=lat, longitude=lon) for lat, lon in z] for z in fence.no_go]
        fence_geofence = Geofence(geofence_id="thor_link", boundary=boundary)
        return build_shape(fence_geofence, no_go=zones, inset_m=fence.margin_m)
    except Exception as exc:
        raise ThorLinkStartError(f"geofence is not usable: {exc}") from exc


@dataclass
class ThorLinkRuntime:
    """The running link and its consumer."""

    link: ThorLinkService
    follower: ThorPlanFollower
    settings: ThorLinkSettings

    def status(self) -> dict[str, Any]:
        link = self.link
        return {
            "session_id": link.session_id,
            "safe_hold": link.safe_hold,
            "hold_reason": link.hold_reason.value if link.hold_reason else None,
            "hold_fault": link.hold_fault,
            "hold_error": link.hold_error,
            "rejected": link.rejected,
            "rejects": dict(link.rejects),
            "verifications": link.verifications,
            "bytes_sent": link.bytes_sent,
            "bytes_received": link.bytes_received,
            "uplink_errors": link.uplink_errors,
            "builder_errors": link.builder_errors,
            "uplinks_degraded": link.uplinks_degraded,
            "detections_dropped": link.detections_dropped,
            "follower": self.follower.status(),
        }


async def start_thor_link(
    tractor: TractorControlService,
    *,
    safety_monitor_running: bool,
    gnss: GnssProvider,
    nav: NavigationService | None = None,
    settings: ThorLinkSettings | None = None,
) -> ThorLinkRuntime | None:
    """Start the link and its follower if, and only if, it is enabled and safe to.

    Returns None when the config says disabled. Raises ``ThorLinkStartError`` when it is
    enabled but a precondition fails or the socket cannot be bound: log it loudly and carry
    on without the link, never with a weaker one.
    """
    cfg = settings if settings is not None else load_thor_link_settings()
    if not cfg.enabled:
        logger.info("Thor link is off (%s: enabled is false)", CONFIG_FILE)
        return None
    if not tractor.enabled:
        raise ThorLinkStartError(
            "tractor.enabled is false: the Thor link drives the tractor platform only"
        )
    if not safety_monitor_running:
        raise ThorLinkStartError(
            "the tractor safety monitor (tilt cutoff, watchdog heartbeat) is not running"
        )
    try:
        key = resolve_link_key()
    except LinkKeyError as exc:
        raise ThorLinkStartError(str(exc)) from exc
    simulated = is_simulation_mode()
    if cfg.geofence is None and not simulated:
        raise ThorLinkStartError(
            "no geofence configured: outside SIM_MODE a plan is never accepted unbounded"
        )
    fence = build_fence(cfg.geofence) if cfg.geofence is not None else None
    if cfg.bind_host == "0.0.0.0" and not simulated:
        logger.warning("Thor link binds 0.0.0.0: prefer the Pi's HaLow address (bind_host)")
    if cfg.allowed_sources == "any":
        logger.warning("Thor link source allowlist is off: only the HMAC authenticates datagrams")

    if nav is None:
        from .navigation_service import NavigationService

        nav = NavigationService.get_instance()
    follower = ThorPlanFollower(
        nav,
        tractor,
        gnss,
        FollowerConfig(
            allow_blade=cfg.allow_blade,
            speed_percent=cfg.speed_percent,
            require_fix=tuple(cfg.require_fix),
            max_position_age_s=cfg.max_position_age_s,
            neutral_timeout_s=cfg.neutral_timeout_s,
        ),
        fence,
    )
    try:
        link = ThorLinkService(
            tractor,
            (str(cfg.thor_host), cfg.thor_port),
            bind_addr=(cfg.bind_host, cfg.bind_port),
            timeout_s=cfg.timeout_s,
            uplink_hz=cfg.uplink_hz,
            build_uplink=follower.build_uplink,
            key=key,
            plan_max_age_s=cfg.plan_max_age_s,
            neutral_timeout_s=cfg.neutral_timeout_s,
            allowed_sources=cfg.sources(),
            allow_wide_sources=cfg.allow_wide_sources,
            geofence=fence,
        )
    except ValueError as exc:  # includes LinkKeyError
        raise ThorLinkStartError(f"Thor link refused its configuration: {exc}") from exc
    follower.bind(link)
    try:
        await link.start()
    except OSError as exc:
        raise ThorLinkStartError(
            f"cannot bind the Thor link to {cfg.bind_host}:{cfg.bind_port}: {exc}"
        ) from exc
    try:
        await follower.start()
    except BaseException:
        await link.stop()
        raise
    logger.info(
        "Thor link on: blade %s, speed %d%%, fix %s, fence %s",
        "allowed" if cfg.allow_blade else "forbidden",
        cfg.speed_percent,
        "/".join(cfg.require_fix),
        "on" if fence is not None else "off",
    )
    return ThorLinkRuntime(link=link, follower=follower, settings=cfg)


async def stop_thor_link(runtime: ThorLinkRuntime | None) -> None:
    """Stop following, then close the link. Never raises: it runs inside shutdown."""
    if runtime is None:
        return
    for name, stop in (("follower", runtime.follower.stop), ("link", runtime.link.stop)):
        try:
            await stop()
        except Exception:
            logger.exception("Thor link: stopping the %s failed", name)


__all__ = [
    "CONFIG_FILE",
    "FenceSettings",
    "ThorLinkConfigError",
    "ThorLinkRuntime",
    "ThorLinkSettings",
    "ThorLinkStartError",
    "build_fence",
    "load_thor_link_settings",
    "start_thor_link",
    "stop_thor_link",
]
