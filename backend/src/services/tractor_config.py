"""Strict validation of ``config/tractor.yaml`` (audit SF-11).

The tractor service used to coerce whatever the YAML held: ``int(None)`` raised ``TypeError`` from
the constructor (so ``get_tractor_service()`` raised on every call and the *unauthenticated
E-stop route returned 500*), ``bool("false")`` turned ``enabled: "false"`` into ``True``, and
nothing bounded a hold, a pulse or a calibration. :func:`validate_tractor_config` accepts only
the types the file is documented to hold (a YAML ``true``/``false`` is a bool, a string is not),
rejects NaN/inf and out-of-range values, and reports **every** problem in one
:class:`TractorConfigError`.

The caller decides what an invalid file means. ``TractorControlService`` raises it from its
constructor so a programmer sees it, and ``get_tractor_service()`` converts it into a service
that is held *disabled and locked out* (``TractorControlService.misconfigured``): the E-stop
route keeps answering (with an honest ``not_enabled`` report), every actuation is refused with
the reason, and no GPIO or PWM output is touched, because a pin from a file that cannot be
trusted must not be driven.

The numeric bounds below are **software sanity limits** that catch typos (``max_hold_ms:
1e9``, ``gpio: 99``); they are not validated hardware limits and they never replace the bench
calibration (``us_min``/``us_center``/``us_max`` are checked only for being plausible servo
pulses in a sensible order, never changed).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Command-deadman defaults; ``command_deadman`` overrides. Conservative on purpose: the shipped
# clients re-send well above 2 Hz (navigation 5 Hz, scripts/teleop.py 20 Hz). Field values are a
# bench decision.
DEFAULT_COMMAND_HOLD_MS = 500
DEFAULT_MAX_COMMAND_HOLD_MS = 2000
#: No configured hold may exceed this (the RoboHAT drive validity has the same 10 s ceiling): a
#: hold is how long a command stays in force with nobody refreshing it.
MAX_CONFIGURABLE_HOLD_MS = 10_000
MIN_HOLD_MS = 10

#: BCM numbers on the 40-pin header.
GPIO_RANGE = (0, 27)
#: The starter solenoid is cranked for ``pulse_ms``; a typo must not hold it for minutes.
MAX_STARTER_PULSE_MS = 5000
#: Plausible RC servo pulse widths (the PCA9685 at 50 Hz has a 20 ms period).
SERVO_US_RANGE = (500, 2500)
PCA9685_CHANNELS = 16
I2C_ADDRESS_RANGE = (0x08, 0x77)
I2C_BUS_RANGE = (0, 31)
#: PCA9685 datasheet PWM frequency range.
PWM_FREQUENCY_RANGE = (24.0, 1526.0)
SUPERVISE_INTERVAL_MS_RANGE = (5.0, 5000.0)


class TractorConfigError(ValueError):
    """``config/tractor.yaml`` holds invalid values; ``problems`` lists every one."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("invalid config/tractor.yaml: " + "; ".join(self.problems))


@dataclass(frozen=True)
class ServoSettings:
    channel: int
    us_min: int
    us_center: int
    us_max: int
    bidirectional: bool


@dataclass(frozen=True)
class TractorSettings:
    """Validated, typed ``config/tractor.yaml`` (defaults filled in)."""

    enabled: bool
    #: Only the keys present in the file (``address``/``bus``/``frequency_hz``); the driver
    #: supplies its own default for each absent one.
    pca9685: dict[str, Any]
    left_lever: ServoSettings
    throttle: ServoSettings
    right_lever: ServoSettings
    starter_gpio: int
    starter_active_high: bool
    starter_pulse_ms: int
    blade_gpio: int
    blade_active_high: bool
    require_levers_neutral_to_start: bool
    require_blade_off_to_start: bool
    require_engine_running_for_blade: bool
    disengage_blade_in_reverse: bool
    default_hold_s: float
    max_hold_s: float
    #: ``None`` => the service's own default self-check period.
    supervise_interval_s: float | None


def _describe(value: Any) -> str:
    if value is None:
        return "None (an empty value?)"
    return f"{value!r} ({type(value).__name__})"


def _section(cfg: Mapping[str, Any], key: str, problems: list[str]) -> Mapping[str, Any]:
    value = cfg.get(key)
    if value is None:  # absent or an empty section: all defaults
        return {}
    if not isinstance(value, Mapping):
        problems.append(f"{key} must be a mapping, got {_describe(value)}")
        return {}
    return value


def _bool(sec: Mapping[str, Any], path: str, key: str, default: bool, problems: list[str]) -> bool:
    if key not in sec:
        return default
    value = sec[key]
    if isinstance(value, bool):
        return value
    problems.append(f"{path}.{key} must be true or false (a YAML boolean), got {_describe(value)}")
    return default


def _int(
    sec: Mapping[str, Any],
    path: str,
    key: str,
    default: int,
    lo: int,
    hi: int,
    problems: list[str],
) -> int:
    if key not in sec:
        return default
    value = sec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"{path}.{key} must be an integer in {lo}..{hi}, got {_describe(value)}")
        return default
    if not lo <= value <= hi:
        problems.append(f"{path}.{key} must be in {lo}..{hi}, got {value}")
        return default
    return value


def _number(
    sec: Mapping[str, Any],
    path: str,
    key: str,
    default: float,
    lo: float,
    hi: float,
    problems: list[str],
) -> float:
    if key not in sec:
        return default
    value = sec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{path}.{key} must be a number in {lo:g}..{hi:g}, got {_describe(value)}")
        return default
    if not math.isfinite(value):
        problems.append(f"{path}.{key} must be finite, got {value!r}")
        return default
    if not lo <= value <= hi:
        problems.append(f"{path}.{key} must be in {lo:g}..{hi:g}, got {value:g}")
        return default
    return float(value)


def _servo(
    actuators: Mapping[str, Any],
    name: str,
    channel: int,
    bidirectional: bool,
    problems: list[str],
) -> ServoSettings:
    path = f"actuators.{name}"
    sec = _section(actuators, name, problems)
    lo, hi = SERVO_US_RANGE
    ch = _int(sec, path, "channel", channel, 0, PCA9685_CHANNELS - 1, problems)
    us_min = _int(sec, path, "us_min", 1000, lo, hi, problems)
    us_center = _int(sec, path, "us_center", 1500, lo, hi, problems)
    us_max = _int(sec, path, "us_max", 2000, lo, hi, problems)
    bidir = _bool(sec, path, "bidirectional", bidirectional, problems)
    if us_min >= us_max:
        problems.append(f"{path}: us_min ({us_min}) must be below us_max ({us_max})")
    elif bidir and not us_min <= us_center <= us_max:
        problems.append(
            f"{path}: us_center ({us_center}) must lie between us_min ({us_min}) "
            f"and us_max ({us_max})"
        )
    return ServoSettings(ch, us_min, us_center, us_max, bidir)


def validate_tractor_config(cfg: Mapping[str, Any] | None) -> TractorSettings:
    """Validate a parsed ``config/tractor.yaml``; raise :class:`TractorConfigError` listing
    every problem, else return the typed settings (defaults filled in)."""
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise TractorConfigError([f"the top level must be a mapping, got {_describe(cfg)}"])
    problems: list[str] = []

    enabled = _bool(cfg, "tractor", "enabled", False, problems)

    # --- PCA9685 ----------------------------------------------------------------------------
    pca = _section(cfg, "pca9685", problems)
    pca_out: dict[str, Any] = {}
    if "address" in pca:
        pca_out["address"] = _int(pca, "pca9685", "address", 0x40, *I2C_ADDRESS_RANGE, problems)
    if "bus" in pca:
        pca_out["bus"] = _int(pca, "pca9685", "bus", 1, *I2C_BUS_RANGE, problems)
    if "frequency_hz" in pca:
        pca_out["frequency_hz"] = _number(
            pca, "pca9685", "frequency_hz", 50.0, *PWM_FREQUENCY_RANGE, problems
        )

    # --- servos -----------------------------------------------------------------------------
    actuators = _section(cfg, "actuators", problems)
    left = _servo(actuators, "left_lever", 0, True, problems)
    throttle = _servo(actuators, "throttle", 1, False, problems)
    right = _servo(actuators, "right_lever", 2, True, problems)
    channels = [left.channel, throttle.channel, right.channel]
    if len(set(channels)) != len(channels):
        problems.append(
            "actuators: left_lever, throttle and right_lever must use different PCA9685 "
            f"channels, got {channels}"
        )

    # --- relays -----------------------------------------------------------------------------
    relays = _section(cfg, "relays", problems)
    starter = _section(relays, "starter", problems)
    blade = _section(relays, "blade_pto", problems)
    gpio_lo, gpio_hi = GPIO_RANGE
    starter_gpio = _int(starter, "relays.starter", "gpio", 5, gpio_lo, gpio_hi, problems)
    # Default moved off GPIO 6 (spec/hardware.yaml: ToF Left Interrupt) to GPIO 26.
    blade_gpio = _int(blade, "relays.blade_pto", "gpio", 26, gpio_lo, gpio_hi, problems)
    if starter_gpio == blade_gpio:
        problems.append(
            f"relays: starter and blade_pto must use different GPIO pins, both are {starter_gpio}"
        )
    starter_active_high = _bool(starter, "relays.starter", "active_high", True, problems)
    blade_active_high = _bool(blade, "relays.blade_pto", "active_high", True, problems)
    pulse_ms = _int(starter, "relays.starter", "pulse_ms", 800, 1, MAX_STARTER_PULSE_MS, problems)

    # --- interlocks -------------------------------------------------------------------------
    il = _section(cfg, "interlocks", problems)
    levers_neutral = _bool(il, "interlocks", "require_levers_neutral_to_start", True, problems)
    blade_off = _bool(il, "interlocks", "require_blade_off_to_start", True, problems)
    engine_for_blade = _bool(il, "interlocks", "require_engine_running_for_blade", True, problems)
    reverse_drop = _bool(il, "interlocks", "disengage_blade_in_reverse", True, problems)

    # --- command deadman --------------------------------------------------------------------
    dm = _section(cfg, "command_deadman", problems)
    default_ms = _number(
        dm,
        "command_deadman",
        "default_hold_ms",
        DEFAULT_COMMAND_HOLD_MS,
        MIN_HOLD_MS,
        MAX_CONFIGURABLE_HOLD_MS,
        problems,
    )
    max_ms = _number(
        dm,
        "command_deadman",
        "max_hold_ms",
        DEFAULT_MAX_COMMAND_HOLD_MS,
        MIN_HOLD_MS,
        MAX_CONFIGURABLE_HOLD_MS,
        problems,
    )
    if max_ms < default_ms:
        problems.append(
            f"command_deadman: max_hold_ms ({max_ms:g}) must not be below "
            f"default_hold_ms ({default_ms:g})"
        )

    # --- actuation self-check ---------------------------------------------------------------
    sup = _section(cfg, "actuation_supervision", problems)
    interval_ms = _number(
        sup,
        "actuation_supervision",
        "interval_ms",
        -1.0,
        *SUPERVISE_INTERVAL_MS_RANGE,
        problems,
    )

    if problems:
        raise TractorConfigError(problems)
    return TractorSettings(
        enabled=enabled,
        pca9685=pca_out,
        left_lever=left,
        throttle=throttle,
        right_lever=right,
        starter_gpio=starter_gpio,
        starter_active_high=starter_active_high,
        starter_pulse_ms=pulse_ms,
        blade_gpio=blade_gpio,
        blade_active_high=blade_active_high,
        require_levers_neutral_to_start=levers_neutral,
        require_blade_off_to_start=blade_off,
        require_engine_running_for_blade=engine_for_blade,
        disengage_blade_in_reverse=reverse_drop,
        default_hold_s=default_ms / 1000.0,
        max_hold_s=max_ms / 1000.0,
        supervise_interval_s=None if interval_ms < 0 else interval_ms / 1000.0,
    )


__all__ = [
    "DEFAULT_COMMAND_HOLD_MS",
    "DEFAULT_MAX_COMMAND_HOLD_MS",
    "MAX_CONFIGURABLE_HOLD_MS",
    "ServoSettings",
    "TractorConfigError",
    "TractorSettings",
    "validate_tractor_config",
]
