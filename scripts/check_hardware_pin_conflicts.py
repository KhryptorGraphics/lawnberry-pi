#!/usr/bin/env python3
"""
Hardware pin/address conflict checker (Constitution Principle V).

The registry is HAND-MAINTAINED: spec/hardware.yaml records which GPIO pin and I2C address each
role uses, with evidence and status. This script checks the registry against itself, against the
config that drives the hardware and against the few places in backend/ that hard-code pins. It
cannot see the Pi: the `deployment` block is a record of a read-only inspection, not a probe.

Reads (paths relative to --root):
  spec/hardware.yaml          the registry
    pins.gpio_used            GPIO claims: a `gpio` row, or a txd/rxd row resolved through its
                              physical `pin`; a row may name an `owner` (see ARMING)
    pins.i2c_addresses        I2C claims, on the header bus unless a row names another `bus`
    pins.ultrasonic           the legacy 3x HC-SR04 Pi pins; claimed only when limits.yaml
                              ultrasonic_enabled AND config/hardware.yaml ultrasonic.backend is
                              gpio_hcsr04 (the default mcu_usb backend, four JSN-SR04T behind a
                              USB MCU, claims no Pi pin)
    pins.uart_overlays        cross-validated with the txd/rxd rows (physical pin <-> GPIO)
    deployment                what the deployed Pi exposes: I2C buses, enabled UARTs
  config/tractor.yaml         relays.*.gpio, pca9685.address/bus, actuators.*.channel; a missing key
                              means the runtime default (TractorControlService, PCA9685Driver)
  config/limits.yaml          ultrasonic_enabled
  config/hardware.yaml        ultrasonic.backend, ibt4.gpio_enabled and ibt4.pins,
                              sensors.tof_config (shutdown GPIOs, bus, addresses)
  backend/src/...             UltrasonicDriver.SENSORS, hw_selftest.ULTRASONIC_GPIO_PINS,
                              BladeController.gpio_pins, ibt4_gpio.py `pins.get("in1"/"in2", N)`:
                              compared with the registry (a value that differs is drift; a file or
                              symbol that moved is only a warning)
  spec/known_conflicts.yaml   the reviewed allow-list

Then reports any GPIO number, or I2C address on the same bus, claimed by more than one role across
the merged set; every such conflict must be in the allow-list. Besides conflicts it fails on:
  - a registry that is empty, internally inconsistent (pin <-> GPIO, txd/rxd <-> overlays,
    ultrasonic pin_trig/pin_echo <-> GPIO) or that differs from config/ and backend/;
  - tractor values that cannot be right: a relay on a GPIO that is not a usable header GPIO
    (for instance a physical pin number), two actuators on one PCA9685 channel, a channel
    outside 0-15, an address outside the I2C range;
  - an allow-listed conflict whose claimant is owned by a component that is armed
    (config/tractor.yaml `enabled: true`, config/hardware.yaml `ibt4.gpio_enabled: true`), and a
    tractor that is enabled while its PCA9685 bus does not exist on the deployed Pi (BLOCKED);
  - an allow-list owner_action that names a GPIO the registry already claims.

Exit codes: 0 = no conflicts (warnings allowed), 1 = conflict(s) found or parse error.

Usage:
  python3 scripts/check_hardware_pin_conflicts.py            # normal check
  python3 scripts/check_hardware_pin_conflicts.py --self-test  # inline unit tests
  python3 scripts/check_hardware_pin_conflicts.py --root DIR   # run against DIR (for tests)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Components with an arming switch. A claim owned by one of them makes an allow-listed conflict
# BLOCKING while the switch is on: owner -> (file, key path, what the switch means).
ARMING: dict[str, tuple[str, str]] = {
    "tractor": ("config/tractor.yaml", "enabled"),
    "ibt4": ("config/hardware.yaml", "ibt4.gpio_enabled"),
}

# Runtime defaults for keys config/tractor.yaml may omit. They mirror TractorControlService
# (relay pins, servo channels) and PCA9685Driver (address, bus); the unit tests compare them with
# the real classes so a changed default cannot go unnoticed here.
TRACTOR_RELAY_DEFAULTS = {"starter": 5, "blade_pto": 26}
ACTUATOR_CHANNEL_DEFAULTS = {"left_lever": 0, "throttle": 1, "right_lever": 2}
PCA9685_DEFAULT_ADDRESS = 0x40
PCA9685_DEFAULT_BUS = 1
PCA9685_CHANNELS = range(16)
I2C_ADDRESSES = range(0x08, 0x78)  # valid 7-bit addresses; 0x00-0x07 and 0x78-0x7F are reserved


@dataclass(frozen=True, order=True)
class Claim:
    """A single claim on a GPIO pin or I2C address."""

    kind: str  # "gpio" or "i2c"
    key: int | str  # GPIO number (int) or "bus<N>:0xNN" (I2C address on a bus)
    label: str  # human-readable claimant identifier
    source: str  # source file path relative to root
    detail: str  # role description
    status: str = "verified"  # verified | documented | unverified
    pin: int | None = None  # physical header pin (for gpio kind)
    owner: str | None = None  # component whose arming switch gates the claim (ARMING)


@dataclass
class Conflict:
    kind: str
    key: int | str
    claimants: list[Claim]
    blocked_by: list[str] = field(default_factory=list)  # armed owners among the claimants


@dataclass
class AllowEntry:
    kind: str
    pin: int | None = None
    i2c_address: str | None = None
    bus: int | None = None
    claimants: list[str] = field(default_factory=list)
    gap: list[str] = field(default_factory=list)
    reason: str = ""
    owner_action: str = ""

    @property
    def key(self) -> tuple[str, int | str | None]:
        if self.kind == "gpio":
            return ("gpio", self.pin)
        return ("i2c", i2c_key(self.bus, self.i2c_address))


@dataclass
class Report:
    conflicts: list[Conflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    known: list[Conflict] = field(default_factory=list)
    new: list[Conflict] = field(default_factory=list)
    stale: list[AllowEntry] = field(default_factory=list)
    blocked: list[Conflict] = field(default_factory=list)
    blocked_deps: list[tuple[str, str]] = field(default_factory=list)  # (owner, message)
    unclaimed: list[int] = field(default_factory=list)  # header GPIOs no registry row claims


class RegistryError(Exception):
    """Raised for malformed registry/spec/allow-list entries."""

    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise RegistryError(f"{path}: YAML root must be a mapping")
            return data
    except yaml.YAMLError as e:
        raise RegistryError(f"{path}: YAML parse error: {e}") from e
    except OSError as e:
        raise RegistryError(f"{path}: {e}") from e


def _load_hardware_config(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """config/hardware.yaml, or (None, warning) when the file is absent (a bare --root fixture).

    A file that exists but cannot be parsed is an error, not a fallback.
    """
    path = root / "config" / "hardware.yaml"
    if not path.is_file():
        return None, [
            "config/hardware.yaml not found: ultrasonic backend, ibt4, ToF and the backend "
            "sources were not cross-checked (legacy ultrasonic pins are claimed if enabled)"
        ]
    return _load_yaml(path), []


def _is_int(value: Any) -> bool:
    """An int that is not a bool (bool is an int; YAML `yes`/`true` load as bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_i2c_addr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RegistryError(f"invalid I2C address {value!r}")
    try:
        if isinstance(value, int):
            number = value
        else:
            text = str(value).strip().lower()
            number = int(text, 16) if text.startswith("0x") else int(text)
    except ValueError as e:
        raise RegistryError(f"invalid I2C address {value!r}") from e
    if number not in I2C_ADDRESSES:
        raise RegistryError(f"I2C address {value!r} is outside the valid 7-bit range 0x08-0x77")
    return hex(number)


def i2c_key(bus: int | None, address: str | None) -> str:
    """Claim key of an I2C address: addresses only conflict on the same bus."""
    return f"bus{bus}:{address}"


# Raspberry Pi 40-pin header: physical pin -> BCM GPIO (identical on the Pi 4B and the Pi 5)
HEADER_PIN_TO_GPIO = {
    1: None,
    2: None,
    3: 2,
    4: None,
    5: 3,
    6: None,
    7: 4,
    8: 14,
    9: None,
    10: 15,
    11: 17,
    12: 18,
    13: 27,
    14: None,
    15: 22,
    16: 23,
    17: None,
    18: 24,
    19: 10,
    20: None,
    21: 9,
    22: 25,
    23: 11,
    24: 8,
    25: None,
    26: 7,
    27: 0,
    28: 1,
    29: 5,
    30: None,
    31: 6,
    32: 12,
    33: 13,
    34: None,
    35: 19,
    36: 16,
    37: 26,
    38: 20,
    39: None,
    40: 21,
}
GPIO_TO_PIN = {g: p for p, g in HEADER_PIN_TO_GPIO.items() if g is not None}
HEADER_GPIOS = frozenset(GPIO_TO_PIN)
# GPIO0/1 are ID_SD/ID_SC, the HAT ID EEPROM bus: not for a relay or a signal
USABLE_GPIOS = HEADER_GPIOS - {0, 1}


def _gpio_from_pin(pin: int) -> int | None:
    return HEADER_PIN_TO_GPIO.get(pin)


def _pin_from_gpio(gpio: int) -> int | None:
    return GPIO_TO_PIN.get(gpio)


def _check_pin_gpio(where: str, pin: Any, gpio: Any) -> None:
    """A physical pin number and a BCM GPIO number that must describe the same header pin."""
    if not _is_int(pin) or not _is_int(gpio):
        raise RegistryError(
            f"{where}: pin and GPIO must both be integers (pin={pin!r}, gpio={gpio!r})"
        )
    if _gpio_from_pin(pin) != gpio:
        raise RegistryError(
            f"{where}: physical pin {pin} is GPIO{_gpio_from_pin(pin)}, "
            f"but the entry says GPIO{gpio}"
        )


def _ultrasonic_sensors(hw: dict) -> dict[str, dict[str, Any]]:
    ultrasonic = (hw.get("pins") or {}).get("ultrasonic") or {}
    sensors = ultrasonic.get("sensors") or {}
    return sensors if isinstance(sensors, dict) else {}


def _load_ultrasonic_pins(hw: dict) -> dict[str, dict[str, int]] | None:
    """Legacy HC-SR04 pin assignments from the registry (pins.ultrasonic.sensors)."""
    result = {}
    for name, pins in _ultrasonic_sensors(hw).items():
        if isinstance(pins, dict) and "trig_gpio" in pins and "echo_gpio" in pins:
            result[name] = {"trig": pins["trig_gpio"], "echo": pins["echo_gpio"]}
    return result or None


def _ultrasonic_backend(hwcfg: dict | None) -> str:
    """Backend selected by ``config/hardware.yaml`` ``ultrasonic.backend``.

    ``mcu_usb`` (the runtime default when the block or the key is absent) talks to an MCU over USB
    serial and uses no Pi header pin. Anything this checker cannot read or recognise is treated as
    ``gpio_hcsr04`` (claim the legacy pins): a conflict checker must err towards reporting.
    """
    if hwcfg is None:
        return "gpio_hcsr04"
    block = hwcfg.get("ultrasonic")
    if block is None:
        return "mcu_usb"
    backend = block.get("backend", "mcu_usb") if isinstance(block, dict) else None
    return backend if backend in ("mcu_usb", "gpio_hcsr04") else "gpio_hcsr04"


def armed_owners(tractor: dict, hwcfg: dict | None) -> set[str]:
    """Owners (ARMING) whose switch is on, with the runtime's truthiness (`"false"` is on)."""
    armed = set()
    if bool(tractor.get("enabled", False)):
        armed.add("tractor")
    ibt4 = (hwcfg or {}).get("ibt4")
    if isinstance(ibt4, dict) and bool(ibt4.get("gpio_enabled", False)):
        armed.add("ibt4")
    return armed


def _switch_text(owner: str) -> str:
    path, key = ARMING[owner]
    return f"{path} {key.split('.')[-1]}: true"


# ----------------------------------------------------------------------------------------------
# Tractor config: effective values (runtime defaults for missing keys) and their validation
# ----------------------------------------------------------------------------------------------


def _mapping(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RegistryError(f"{where}: must be a mapping")
    return value


def effective_relays(tractor: dict) -> dict[str, int]:
    """Relay name -> BCM GPIO the tractor service will claim (defaults for starter/blade_pto)."""
    relays = _mapping(tractor.get("relays"), "config/tractor.yaml relays")
    effective = dict(TRACTOR_RELAY_DEFAULTS)
    for name, relay in relays.items():
        relay = _mapping(relay, f"config/tractor.yaml relays.{name}")
        if "gpio" in relay:
            gpio = relay["gpio"]
        elif name in TRACTOR_RELAY_DEFAULTS:
            gpio = TRACTOR_RELAY_DEFAULTS[name]
        else:
            raise RegistryError(f"config/tractor.yaml relays.{name}: missing 'gpio' field")
        if not _is_int(gpio):
            raise RegistryError(f"config/tractor.yaml relays.{name}: gpio must be int")
        if gpio in (0, 1):
            raise RegistryError(
                f"config/tractor.yaml relays.{name}: GPIO{gpio} is ID_SD/ID_SC (the HAT ID EEPROM "
                "bus), not usable for a relay"
            )
        if gpio not in USABLE_GPIOS:
            hint = ""
            if _gpio_from_pin(gpio) is not None:
                hint = (
                    f" (physical pin {gpio} is GPIO{_gpio_from_pin(gpio)}: this file takes BCM "
                    "GPIO numbers, not physical pin numbers)"
                )
            raise RegistryError(
                f"config/tractor.yaml relays.{name}: gpio={gpio} is not a usable Raspberry Pi "
                f"header GPIO (BCM 2-27){hint}"
            )
        effective[name] = gpio
    return effective


def effective_channels(tractor: dict) -> dict[str, int]:
    """Actuator name -> PCA9685 channel (defaults 0, 1, 2); each channel drives one servo."""
    actuators = _mapping(tractor.get("actuators"), "config/tractor.yaml actuators")
    channels = {}
    for name, default in ACTUATOR_CHANNEL_DEFAULTS.items():
        cfg = _mapping(actuators.get(name), f"config/tractor.yaml actuators.{name}")
        channel = cfg.get("channel", default)
        if not _is_int(channel) or channel not in PCA9685_CHANNELS:
            raise RegistryError(
                f"config/tractor.yaml actuators.{name}.channel={channel!r}: PCA9685 channels are "
                "the integers 0-15"
            )
        channels[name] = channel
    for channel in sorted(set(channels.values())):
        users = [n for n, c in channels.items() if c == channel]
        if len(users) > 1:
            raise RegistryError(
                f"config/tractor.yaml actuators {', '.join(users)} are all on PCA9685 channel "
                f"{channel}: one channel drives one servo"
            )
    return channels


def effective_pca9685(tractor: dict) -> tuple[int, int, bool]:
    """(address, bus, address_was_configured) the PCA9685 driver will use."""
    cfg = _mapping(tractor.get("pca9685"), "config/tractor.yaml pca9685")
    raw = cfg.get("address")
    address = PCA9685_DEFAULT_ADDRESS if raw is None else int(_normalize_i2c_addr(raw) or "0", 16)
    bus = cfg.get("bus")
    if bus is None:
        bus = PCA9685_DEFAULT_BUS
    elif not _is_int(bus) or bus < 0:
        raise RegistryError(f"config/tractor.yaml pca9685.bus={bus!r}: must be a bus number >= 0")
    return address, bus, raw is not None


def validate_tractor(tractor: dict) -> None:
    """Raise RegistryError for values that cannot be right (see the module docstring)."""
    effective_relays(tractor)
    effective_channels(tractor)
    effective_pca9685(tractor)


# ----------------------------------------------------------------------------------------------
# Registry validation (spec/hardware.yaml against itself)
# ----------------------------------------------------------------------------------------------


def validate_registry(hw: dict) -> None:
    """Raise RegistryError when the registry is empty or contradicts itself."""
    pins = hw.get("pins")
    if not isinstance(pins, dict):
        raise RegistryError("spec/hardware.yaml: 'pins' mapping is missing")
    if not isinstance(pins.get("gpio_used"), list) or not pins["gpio_used"]:
        raise RegistryError(
            "spec/hardware.yaml: pins.gpio_used is missing or empty; refusing to pass on an "
            "empty registry"
        )
    if not isinstance(pins.get("i2c_addresses"), dict) or not pins["i2c_addresses"]:
        raise RegistryError(
            "spec/hardware.yaml: pins.i2c_addresses is missing or empty; refusing to pass on an "
            "empty registry"
        )

    overlays = _mapping(pins.get("uart_overlays"), "spec/hardware.yaml pins.uart_overlays")
    for uart, platforms in overlays.items():
        for platform, entry in _mapping(platforms, f"pins.uart_overlays.{uart}").items():
            if not isinstance(entry, dict):  # e.g. the free-text `note`
                continue
            where = f"spec/hardware.yaml pins.uart_overlays.{uart}.{platform}"
            _check_pin_gpio(f"{where} txd", entry.get("pin"), entry.get("txd_gpio"))
            _check_pin_gpio(f"{where} rxd", entry.get("pin_rxd"), entry.get("rxd_gpio"))

    for entry in pins["gpio_used"]:
        if not isinstance(entry, dict):
            raise RegistryError(f"spec/hardware.yaml: gpio_used entry {entry!r} is not a mapping")
        role = entry.get("role", "?")
        if entry.get("owner") is not None and entry["owner"] not in ARMING:
            raise RegistryError(
                f"spec/hardware.yaml: gpio_used entry role={role}: owner must be one of "
                f"{', '.join(sorted(ARMING))}"
            )
        direction = "txd" if "txd" in entry else "rxd" if "rxd" in entry else None
        if direction is None:
            continue
        uart = f"uart{entry[direction]}"
        if uart not in overlays:
            raise RegistryError(
                f"spec/hardware.yaml: gpio_used entry role={role}: {direction}: {entry[direction]} "
                f"names {uart}, which pins.uart_overlays does not define"
            )
        pin_key, gpio_key = ("pin", "txd_gpio") if direction == "txd" else ("pin_rxd", "rxd_gpio")
        pi5 = (overlays[uart].get("pi5") or {}) if isinstance(overlays[uart], dict) else {}
        if pi5.get(pin_key) != entry.get("pin"):
            raise RegistryError(
                f"spec/hardware.yaml: gpio_used entry role={role}: pin {entry.get('pin')} but "
                f"pins.uart_overlays.{uart}.pi5.{pin_key} is {pi5.get(pin_key)}"
            )
        alt = entry.get("rpi4_alt_pin")
        if alt is not None:
            pi4 = (overlays[uart].get("pi4") or {}) if isinstance(overlays[uart], dict) else {}
            if _gpio_from_pin(alt) != pi4.get(gpio_key):
                raise RegistryError(
                    f"spec/hardware.yaml: gpio_used entry role={role}: rpi4_alt_pin {alt} is "
                    f"GPIO{_gpio_from_pin(alt)} but pins.uart_overlays.{uart}.pi4.{gpio_key} is "
                    f"{pi4.get(gpio_key)}"
                )

    for name, sensor in _ultrasonic_sensors(hw).items():
        where = f"spec/hardware.yaml pins.ultrasonic.sensors.{name}"
        if not isinstance(sensor, dict):
            raise RegistryError(f"{where}: must be a mapping")
        for line in ("trig", "echo"):
            gpio = sensor.get(f"{line}_gpio")
            if not _is_int(gpio) or gpio not in HEADER_GPIOS:
                raise RegistryError(f"{where}.{line}_gpio={gpio!r}: not a Raspberry Pi header GPIO")
            if sensor.get(f"pin_{line}") is not None:
                _check_pin_gpio(f"{where} {line}", sensor[f"pin_{line}"], gpio)

    deployment = hw.get("deployment")
    if isinstance(deployment, dict):
        for uart in deployment.get("uart_overlays_enabled") or []:
            if uart not in overlays:
                raise RegistryError(
                    f"spec/hardware.yaml deployment.uart_overlays_enabled: '{uart}' is not a key "
                    f"of pins.uart_overlays ({', '.join(overlays)}); list the UART name (uart4), "
                    "not the overlay (uart4-pi5)"
                )


def _gpio_rows(hw: dict) -> dict[str, int]:
    """Role -> GPIO of the registry's plain `gpio` rows."""
    rows: dict[str, int] = {}
    for entry in (hw.get("pins") or {}).get("gpio_used") or []:
        if isinstance(entry, dict) and "gpio" in entry:
            rows.setdefault(entry.get("role", "?"), entry["gpio"])
    return rows


def _registry_i2c(hw: dict) -> list[tuple[str, int, str]]:
    """(device, bus, normalized address) for each registry I2C row; rows sit on the header bus."""
    header_bus = (hw.get("deployment") or {}).get("header_i2c_bus", 1)
    rows = []
    for addr, device in ((hw.get("pins") or {}).get("i2c_addresses") or {}).items():
        norm = _normalize_i2c_addr(addr)
        if norm is None:
            continue
        if isinstance(device, dict):
            rows.append((str(device.get("device", "?")), device.get("bus", header_bus), norm))
        else:
            rows.append((str(device), header_bus, norm))
    return rows


# ----------------------------------------------------------------------------------------------
# Claim collection
# ----------------------------------------------------------------------------------------------


def collect_gpio_claims(
    hw: dict, tractor: dict, limits: dict, hwcfg: dict | None = None
) -> tuple[list[Claim], list[str]]:
    """Collect all GPIO claims from registry, tractor and (when enabled) the legacy ultrasonics.
    Returns (claims, warnings)."""
    claims: list[Claim] = []
    warnings: list[str] = []

    # 1. spec/hardware.yaml pins.gpio_used
    for entry in hw.get("pins", {}).get("gpio_used", []) or []:
        role = entry.get("role", "?")
        status = entry.get("status", "verified")
        evidence = entry.get("evidence", [])
        pin = entry.get("pin")
        owner = entry.get("owner")

        if "gpio" in entry:
            gpio = entry["gpio"]
            if not _is_int(gpio):
                raise RegistryError(
                    f"spec/hardware.yaml: gpio_used entry role={role}: gpio must be int"
                )
            if gpio not in HEADER_GPIOS:
                raise RegistryError(
                    f"spec/hardware.yaml: gpio_used entry role={role}: GPIO{gpio} is not a "
                    "Raspberry Pi header GPIO"
                )
            if pin is not None and _gpio_from_pin(pin) != gpio:
                raise RegistryError(
                    f"spec/hardware.yaml: gpio_used entry role={role}: pin {pin} -> GPIO {_gpio_from_pin(pin)}, but gpio={gpio}"  # noqa: E501
                )
            claims.append(
                Claim(
                    kind="gpio",
                    key=gpio,
                    label=f"spec/hardware.yaml: {role}",
                    source="spec/hardware.yaml",
                    detail=role,
                    status=status,
                    pin=pin,
                    owner=owner,
                )
            )
        elif "txd" in entry or "rxd" in entry:
            # UART pin claim: derive GPIO from physical pin via header map
            if pin is None:
                raise RegistryError(
                    f"spec/hardware.yaml: gpio_used entry role={role}: txd/rxd row requires 'pin'"
                )
            gpio = _gpio_from_pin(pin)
            if gpio is None:
                raise RegistryError(
                    f"spec/hardware.yaml: gpio_used entry role={role}: pin {pin} not in header map"
                )
            claims.append(
                Claim(
                    kind="gpio",
                    key=gpio,
                    label=f"spec/hardware.yaml: {role}",
                    source="spec/hardware.yaml",
                    detail=role,
                    status=status,
                    pin=pin,
                    owner=owner,
                )
            )
        else:
            raise RegistryError(
                f"spec/hardware.yaml: gpio_used entry role={role}: must have 'gpio' or 'txd'/'rxd'"
            )

        # Validate status/evidence discipline
        if status == "unverified":
            if not entry.get("measure"):
                warnings.append(
                    f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): unverified claim missing 'measure' field"  # noqa: E501
                )
            if not evidence:
                warnings.append(
                    f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): unverified claim missing 'evidence' field"  # noqa: E501
                )
        elif status in ("verified", "documented"):
            if not evidence:
                warnings.append(
                    f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): {status} claim missing 'evidence' field"  # noqa: E501
                )
        elif status:
            warnings.append(
                f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): unknown status '{status}' (expected verified|documented|unverified)"  # noqa: E501
            )

    # 2. config/tractor.yaml relays (always claim, even if enabled: false: the service builds both
    #    relays whatever `enabled` says, and a missing key means the runtime default pin)
    for relay_name, gpio in effective_relays(tractor).items():
        claims.append(
            Claim(
                kind="gpio",
                key=gpio,
                label=f"config/tractor.yaml relays.{relay_name}",
                source="config/tractor.yaml",
                detail=f"relay {relay_name}",
                status="verified",
                pin=_pin_from_gpio(gpio),
                owner="tractor",
            )
        )

    # 3. Header I2C1 SDA/SCL (GPIO2/GPIO3) are claimed by explicit gpio_used rows in the spec;
    #    bus availability on the deployed Pi is reported from the `deployment` block.

    # 4. Legacy HC-SR04 pins (registry pins.ultrasonic, equal to UltrasonicDriver.SENSORS): only if
    #    limits.yaml ultrasonic_enabled AND the backend is the legacy GPIO driver. The mcu_usb
    #    backend (JSN-SR04T x4 behind a USB MCU) claims no Pi pin.
    ul_enabled = True  # default matches SafetyLimits default
    if limits:
        ul_enabled = bool(limits.get("ultrasonic_enabled", True))
    ultrasonic_pins = _load_ultrasonic_pins(hw)
    if ul_enabled and ultrasonic_pins and _ultrasonic_backend(hwcfg) == "gpio_hcsr04":
        for sensor_name, pins in ultrasonic_pins.items():
            for pin_type, gpio in (("trig", pins["trig"]), ("echo", pins["echo"])):
                claims.append(
                    Claim(
                        kind="gpio",
                        key=gpio,
                        label=f"spec/hardware.yaml pins.ultrasonic: {sensor_name} {pin_type}",
                        source="spec/hardware.yaml",
                        detail=f"ultrasonic {sensor_name} {pin_type}",
                        status="verified",
                        pin=_pin_from_gpio(gpio),
                    )
                )

    return claims, warnings


def collect_i2c_claims(hw: dict, tractor: dict) -> tuple[list[Claim], list[str]]:
    """Collect all I2C address claims. An address only conflicts with the same address on the
    same bus."""
    claims: list[Claim] = []
    warnings: list[str] = []

    # spec/hardware.yaml pins.i2c_addresses
    for device, bus, norm in _registry_i2c(hw):
        claims.append(
            Claim(
                kind="i2c",
                key=i2c_key(bus, norm),
                label=f"spec/hardware.yaml: {device}",
                source="spec/hardware.yaml",
                detail=device,
                status="verified",
            )
        )

    # config/tractor.yaml pca9685: the service builds the driver whatever `enabled` says, and a
    # missing address means the driver default 0x40
    address, bus, configured = effective_pca9685(tractor)
    claims.append(
        Claim(
            kind="i2c",
            key=i2c_key(bus, hex(address)),
            label="config/tractor.yaml pca9685.address",
            source="config/tractor.yaml",
            detail="PCA9685",
            status="verified",
            owner="tractor",
        )
    )
    if not configured:
        warnings.append(
            "config/tractor.yaml pca9685.address not set; the driver default 0x40 is claimed "
            "(it collides with an INA3221 at 0x40 on the same bus)"
        )

    return claims, warnings


def find_conflicts(claims: list[Claim]) -> dict[tuple[str, int | str], list[Claim]]:
    """Group claims by (kind, key); return those with >=2 distinct labels."""
    by_key: dict[tuple[str, int | str], list[Claim]] = {}
    for c in claims:
        by_key.setdefault((c.kind, c.key), []).append(c)
    return {k: v for k, v in by_key.items() if len({c.label for c in v}) > 1}


def load_allow_list(root: Path) -> list[AllowEntry]:
    """Load and validate spec/known_conflicts.yaml."""
    path = root / "spec" / "known_conflicts.yaml"
    data = _load_yaml(path)
    if data.get("version") != 1:
        raise RegistryError(f"{path}: version must be 1")
    entries = []
    seen_keys = set()
    for i, e in enumerate(data.get("conflicts", [])):
        kind = e.get("kind")
        if kind not in ("gpio", "i2c_address"):
            raise RegistryError(f"{path}: entry {i}: kind must be 'gpio' or 'i2c_address'")
        pin = e.get("pin")
        i2c_addr = e.get("i2c_address")
        bus = e.get("bus")
        norm = None
        if kind == "gpio":
            if pin is None or not _is_int(pin):
                raise RegistryError(f"{path}: entry {i}: gpio kind requires integer 'pin'")
            key = ("gpio", pin)
        else:
            if not i2c_addr:
                raise RegistryError(f"{path}: entry {i}: i2c_address kind requires 'i2c_address'")
            norm = _normalize_i2c_addr(i2c_addr)
            if norm is None:
                raise RegistryError(f"{path}: entry {i}: invalid i2c_address")
            if not _is_int(bus) or bus < 0:
                raise RegistryError(
                    f"{path}: entry {i}: i2c_address kind requires integer 'bus' (the same address "
                    "on another bus is not the same conflict)"
                )
            key = ("i2c", i2c_key(bus, norm))
        if key in seen_keys:
            raise RegistryError(f"{path}: duplicate entry for {key}")
        seen_keys.add(key)

        claimants = e.get("claimants", [])
        if not isinstance(claimants, list) or len(claimants) < 2:
            raise RegistryError(
                f"{path}: entry {i}: claimants must be list with >=2 unique strings"
            )
        if len(set(claimants)) != len(claimants):
            raise RegistryError(f"{path}: entry {i}: claimants must be unique")

        gap = e.get("gap", [])
        if not isinstance(gap, list) or not gap:
            raise RegistryError(f"{path}: entry {i}: gap must be non-empty list")
        for g in gap:
            if not isinstance(g, str) or not g.startswith("H-"):
                raise RegistryError(f"{path}: entry {i}: gap ID '{g}' must match H-XX pattern")

        reason = e.get("reason", "")
        owner_action = e.get("owner_action", "")
        if not reason or not owner_action:
            raise RegistryError(f"{path}: entry {i}: reason and owner_action required")

        # reject unknown keys
        allowed = {
            "kind",
            "pin",
            "i2c_address",
            "bus",
            "claimants",
            "gap",
            "reason",
            "owner_action",
        }
        for k in e:
            if k not in allowed:
                raise RegistryError(f"{path}: entry {i}: unknown key '{k}'")

        entries.append(
            AllowEntry(
                kind=kind,
                pin=pin,
                i2c_address=norm,
                bus=bus if kind == "i2c_address" else None,
                claimants=claimants,
                gap=gap,
                reason=reason,
                owner_action=owner_action,
            )
        )
    return entries


def reconcile(
    detected: dict[tuple[str, int | str], list[Claim]],
    allowed: list[AllowEntry],
    armed: Collection[str] = (),
) -> Report:
    """Match detected conflicts against allow-list; classify as known/new/stale/blocked.

    A known conflict is blocked while any claimant's owner is in `armed` (ARMING).
    """
    report = Report()
    report.conflicts = [Conflict(kind=k[0], key=k[1], claimants=v) for k, v in detected.items()]

    allowed_by_key = {a.key: a for a in allowed}
    matched_keys = set()

    for conflict in report.conflicts:
        key = (conflict.kind, conflict.key)
        labels = sorted({c.label for c in conflict.claimants})

        if key in allowed_by_key:
            a = allowed_by_key[key]
            if labels == sorted(a.claimants):
                matched_keys.add(key)
                owners = sorted({c.owner for c in conflict.claimants if c.owner in armed})
                if owners:
                    conflict.blocked_by = owners
                    report.blocked.append(conflict)
                else:
                    report.known.append(conflict)
            else:
                # Claimant set changed -> NEW + STALE (mark as matched to avoid double-counting)
                matched_keys.add(key)
                report.new.append(conflict)
                report.stale.append(a)
        else:
            # No allow-list entry
            report.new.append(conflict)

    # Stale allow-list entries (no matching live conflict)
    for a in allowed:
        if a.key not in matched_keys:
            report.stale.append(a)

    return report


def check_deployment(
    hw: dict, tractor: dict, hwcfg: dict | None, armed: Collection[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Warnings and blockers (owner, message) from the `deployment` block: what the deployed Pi
    exposes (I2C buses, UARTs) against what the config needs."""
    warnings: list[str] = []
    blockers: list[tuple[str, str]] = []
    dep = hw.get("deployment")
    if not dep:
        warnings.append(
            "spec/hardware.yaml has no 'deployment' block: I2C bus availability and UART "
            "overlays were not checked"
        )
        return warnings, blockers

    # I2C buses the config needs
    header_bus = dep.get("header_i2c_bus", 1)
    header_enabled = dep.get("header_i2c_bus_enabled")
    present = set(dep.get("i2c_buses_present") or [])
    _, pca_bus, _ = effective_pca9685(tractor)
    if header_enabled is False:
        affected = []
        if pca_bus == header_bus:
            affected.append(f"config/tractor.yaml pca9685 (bus={pca_bus})")
        tof = ((hwcfg or {}).get("sensors") or {}).get("tof_config")
        if isinstance(tof, dict) and tof.get("bus", 1) == header_bus:
            affected.append(f"config/hardware.yaml sensors.tof_config (bus={header_bus})")
        for device, bus, norm in _registry_i2c(hw):
            if bus == header_bus:
                affected.append(f"spec/hardware.yaml i2c_addresses: {device} ({norm})")
        if affected:
            warnings.append(
                f"DEPLOYMENT WARNING: Header I2C bus {header_bus} is disabled on deployed Pi "
                f"(i2c_arm=off, GPIO3=HaLow WAKE). The following devices are configured "
                f"for this bus and will be unreachable: {', '.join(affected)}. "
                f"Relocate to another I2C bus (e.g., RP1 I2C4, software I2C) or re-enable i2c_arm "
                f"and resolve GPIO3 WAKE conflict."
            )
        if pca_bus == header_bus and "tractor" in armed:
            blockers.append(
                (
                    "tractor",
                    f"config/tractor.yaml pca9685.bus={pca_bus} is the header I2C bus, which is "
                    "disabled on the deployed Pi (i2c_arm=off, GPIO3 is the HaLow WAKE line): the "
                    "actuator bus does not exist",
                )
            )
    if present and pca_bus != header_bus and pca_bus not in present:
        message = (
            f"config/tractor.yaml pca9685.bus={pca_bus} is not among the I2C buses present on "
            f"the deployed Pi ({', '.join(str(b) for b in sorted(present))}): the actuator bus "
            "does not exist"
        )
        warnings.append(f"DEPLOYMENT WARNING: {message}")
        if "tractor" in armed:
            blockers.append(("tractor", message))

    # UART4 overlay availability (BNO085): `uart_overlays_enabled` lists UART names (uart0, uart4)
    uart_overlays_enabled = set(dep.get("uart_overlays_enabled", []) or [])
    uart4 = hw.get("pins", {}).get("uart_overlays", {}).get("uart4", {})
    host = dep.get("host", "pi5")
    platform_entry = uart4.get(host) if isinstance(uart4, dict) else None
    if platform_entry and "uart4" not in uart_overlays_enabled:
        overlay = platform_entry.get("overlay", "uart4-pi5")
        for entry in hw.get("pins", {}).get("gpio_used", []):
            if "BNO085" in entry.get("role", "") and ("txd" in entry or "rxd" in entry):
                txd, rxd = platform_entry.get("txd_gpio"), platform_entry.get("rxd_gpio")
                warnings.append(
                    f"DEPLOYMENT WARNING: UART4 overlay '{overlay}' not enabled on deployed Pi. "
                    f"BNO085 IMU claims UART4 TX/RX (GPIO{txd}/GPIO{rxd}). "
                    f"Add 'dtoverlay={overlay}' to /boot/firmware/config.txt to enable, then "
                    "record 'uart4' in deployment.uart_overlays_enabled."
                )
                break

    return warnings, blockers


def check_ultrasonic_hard_conflicts(claims: list[Claim]) -> list[str]:
    """Warn when unverified claims overlap with verified claims on same GPIO."""
    warnings = []
    by_gpio: dict[int, list[Claim]] = {}
    for c in claims:
        if c.kind == "gpio":
            by_gpio.setdefault(c.key, []).append(c)

    for gpio, clist in by_gpio.items():
        unverified = [c for c in clist if c.status == "unverified"]
        verified = [c for c in clist if c.status != "unverified"]
        if unverified and verified:
            for u in unverified:
                for v in verified:
                    if u.label != v.label:
                        warnings.append(
                            f"WARNING: Unverified claim '{u.label}' (GPIO{gpio}) overlaps "
                            f"with verified claim '{v.label}'. Physical verification recommended "
                            f"({u.detail})."
                        )
    return warnings


# ----------------------------------------------------------------------------------------------
# Registry versus the config and backend sources that hard-code the same pins
# ----------------------------------------------------------------------------------------------

_MISSING = object()


def _parse_py(root: Path, rel: str) -> ast.Module | None:
    path = root / rel
    if not path.is_file():
        return None
    try:
        return ast.parse(path.read_text(), filename=rel)
    except SyntaxError as e:
        raise RegistryError(f"{rel}: cannot parse: {e}") from e


def _literal(node: ast.AST | None) -> Any:
    try:
        return ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, TypeError, SyntaxError):
        return _MISSING


def _assigned_to(node: ast.AST, name: str, attribute_of: str | None = None) -> ast.AST | None:
    """The value assigned by `node` to the bare name (or to `self.<name>`), else None."""
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        targets, value = [node.target], node.value
    else:
        return None
    for target in targets:
        if attribute_of is None and isinstance(target, ast.Name) and target.id == name:
            return value
        if (
            attribute_of is not None
            and isinstance(target, ast.Attribute)
            and target.attr == name
            and isinstance(target.value, ast.Name)
            and target.value.id == attribute_of
        ):
            return value
    return None


def _module_literal(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        value = _assigned_to(node, name)
        if value is not None:
            return _literal(value)
    return _MISSING


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _class_attr_literal(tree: ast.Module, cls: str, name: str) -> Any:
    node = _class_node(tree, cls)
    for stmt in node.body if node else []:
        value = _assigned_to(stmt, name)
        if value is not None:
            return _literal(value)
    return _MISSING


def _self_attr_literal(tree: ast.Module, cls: str, attr: str) -> Any:
    node = _class_node(tree, cls)
    for stmt in ast.walk(node) if node else []:
        value = _assigned_to(stmt, attr, attribute_of="self")
        if value is not None:
            return _literal(value)
    return _MISSING


def _get_defaults(tree: ast.Module, receiver: str, keys: Collection[str]) -> dict[str, Any]:
    """Defaults of `receiver.get("<key>", <literal>)` calls anywhere in the module."""
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == receiver
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in keys
        ):
            found.setdefault(node.args[0].value, _literal(node.args[1]))
    return found


def _pin_summary(pins: dict[str, dict[str, int]]) -> dict[str, tuple[int, int]]:
    return {name: (p["trig"], p["echo"]) for name, p in sorted(pins.items())}


def check_registry_sources(
    root: Path, hw: dict, tractor: dict, hwcfg: dict | None
) -> tuple[list[str], list[str]]:
    """Compare the registry with config/ and backend/. Returns (errors, warnings): a value that
    differs is an error (drift); a source file or symbol that cannot be found is a warning."""
    errors: list[str] = []
    warnings: list[str] = []
    rows = _gpio_rows(hw)

    def expect(source: str, gpio: Any, role: str) -> None:
        want = rows.get(role)
        if want is None:
            errors.append(
                f"registry drift: {source} uses GPIO{gpio} but spec/hardware.yaml has no "
                f"'{role}' row"
            )
        elif gpio != want:
            errors.append(
                f"registry drift: {source} is GPIO{gpio} but the spec/hardware.yaml '{role}' row is GPIO{want}"  # noqa: E501
            )

    def source_tree(rel: str) -> ast.Module | None:
        try:
            tree = _parse_py(root, rel)
        except RegistryError as e:
            errors.append(str(e))
            return None
        if tree is None:
            warnings.append(f"registry not cross-checked against {rel}: file not found")
        return tree

    def not_found(rel: str, what: str) -> None:
        warnings.append(f"registry not cross-checked against {rel}: {what} not found")

    # Blade driver (IBT-4): registry rows, spec cutting_system, config, the driver's own defaults
    blade_roles = {"in1": "Blade IN1", "in2": "Blade IN2"}
    blade_spec = ((hw.get("cutting_system") or {}).get("blade_driver")) or {}
    for line, role in blade_roles.items():
        if f"gpio_{line}" in blade_spec:
            expect(
                f"spec/hardware.yaml cutting_system.blade_driver.gpio_{line}",
                blade_spec[f"gpio_{line}"],
                role,
            )
    cfg_pins = (((hwcfg or {}).get("ibt4")) or {}).get("pins") or {}
    for line, role in blade_roles.items():
        if line in cfg_pins:
            expect(f"config/hardware.yaml ibt4.pins.{line}", cfg_pins[line], role)
    rel = "backend/src/drivers/blade/ibt4_gpio.py"
    if (tree := source_tree(rel)) is not None:
        defaults = _get_defaults(tree, "pins", blade_roles)
        if not defaults:
            not_found(rel, 'the pins.get("in1"/"in2", N) defaults')
        for line, value in defaults.items():
            expect(f'{rel} pins.get("{line}") default', value, blade_roles[line])
    rel = "backend/src/services/motor_service.py"
    if (tree := source_tree(rel)) is not None:
        pins = _self_attr_literal(tree, "BladeController", "gpio_pins")
        if not isinstance(pins, dict):
            not_found(rel, "BladeController.gpio_pins")
        else:
            for line, role in blade_roles.items():
                if f"blade_{line}" in pins:
                    expect(
                        f"{rel} BladeController.gpio_pins['blade_{line}']",
                        pins[f"blade_{line}"],
                        role,
                    )

    # ToF (VL53L0X): XSHUT pins, addresses and bus from config/hardware.yaml
    tof = (((hwcfg or {}).get("sensors")) or {}).get("tof_config")
    if isinstance(tof, dict):
        for side, role in (("left", "ToF Left Shutdown"), ("right", "ToF Right Shutdown")):
            if tof.get(f"{side}_shutdown_gpio") is not None:
                expect(
                    f"config/hardware.yaml sensors.tof_config.{side}_shutdown_gpio",
                    tof[f"{side}_shutdown_gpio"],
                    role,
                )
        registry = {device: (bus, norm) for device, bus, norm in _registry_i2c(hw)}
        for side, device in (("left", "VL53L0X Left"), ("right", "VL53L0X Right")):
            if f"{side}_address" in tof and device in registry:
                got = _normalize_i2c_addr(tof[f"{side}_address"])
                if got != registry[device][1]:
                    errors.append(
                        f"registry drift: config/hardware.yaml sensors.tof_config.{side}_address "
                        f"is {got} but spec/hardware.yaml i2c_addresses has {device} at "
                        f"{registry[device][1]}"
                    )
        buses = {registry[d][0] for d in ("VL53L0X Left", "VL53L0X Right") if d in registry}
        if "bus" in tof and buses and tof["bus"] not in buses:
            errors.append(
                f"registry drift: config/hardware.yaml sensors.tof_config.bus is {tof['bus']} "
                "but spec/hardware.yaml puts the VL53L0X sensors on bus "
                + ", ".join(str(b) for b in sorted(buses))
            )

    # Legacy HC-SR04 pins
    want = _load_ultrasonic_pins(hw) or {}
    for rel, name in (
        ("backend/src/drivers/sensors/ultrasonic_driver.py", "UltrasonicDriver.SENSORS"),
        ("backend/src/services/hw_selftest.py", "ULTRASONIC_GPIO_PINS"),
    ):
        tree = source_tree(rel)
        if tree is None:
            continue
        found = (
            _class_attr_literal(tree, *name.split("."))
            if "." in name
            else _module_literal(tree, name)
        )
        if not isinstance(found, dict):
            not_found(rel, name)
            continue
        try:
            got = {n: {"trig": p["trig"], "echo": p["echo"]} for n, p in found.items()}
        except (KeyError, TypeError):
            errors.append(
                f"registry drift: {rel} {name} is not a {{sensor: {{trig, echo}}}} mapping"
            )
            continue
        if got != want:
            errors.append(
                f"registry drift: {rel} {name} is {_pin_summary(got)} (trig, echo) but "
                f"spec/hardware.yaml pins.ultrasonic.sensors is {_pin_summary(want)}"
            )

    # The tractor's PCA9685 address in the registry's tractor section
    spec_pca = ((hw.get("tractor") or {}).get("pca9685")) or {}
    if spec_pca.get("address") is not None:
        address, _, _ = effective_pca9685(tractor)
        if _normalize_i2c_addr(spec_pca["address"]) != hex(address):
            errors.append(
                f"registry drift: config/tractor.yaml pca9685.address is {hex(address)} but "
                f"spec/hardware.yaml tractor.pca9685.address is "
                f"{_normalize_i2c_addr(spec_pca['address'])}"
            )

    return errors, warnings


_GPIO_MENTION = re.compile(r"GPIO\s*(\d+(?:\s*/\s*\d+)*)")


def lint_owner_actions(allowed: list[AllowEntry], claimed: Collection[int]) -> list[str]:
    """An owner_action must not point at a GPIO the registry already claims: following the advice
    would create a new conflict. Its own pin is fine; the report lists the unclaimed GPIOs."""
    errors = []
    for a in allowed:
        own = {a.pin} if a.kind == "gpio" else set()
        for mention in _GPIO_MENTION.finditer(a.owner_action):
            for gpio in (int(n) for n in re.split(r"\s*/\s*", mention.group(1))):
                if gpio not in own and gpio in claimed:
                    errors.append(
                        f"spec/known_conflicts.yaml: owner_action of {a.key[0]} {a.key[1]} names "
                        f"GPIO{gpio}, which the registry already claims; name no GPIO (the report "
                        "lists those no row claims) or one of those"
                    )
    return errors


def run_check(root: Path) -> Report:
    hw = _load_yaml(root / "spec" / "hardware.yaml")
    tractor = _load_yaml(root / "config" / "tractor.yaml")
    limits = _load_yaml(root / "config" / "limits.yaml")
    hwcfg, cfg_warnings = _load_hardware_config(root)

    validate_registry(hw)
    validate_tractor(tractor)

    # Collect all claims
    gpio_claims, gw = collect_gpio_claims(hw, tractor, limits, hwcfg)
    i2c_claims, iw = collect_i2c_claims(hw, tractor)

    all_claims = gpio_claims + i2c_claims
    warnings = cfg_warnings + gw + iw

    # Check for spec errors (duplicate pin+role in gpio_used)
    seen_gpio_role = set()
    for c in gpio_claims:
        if c.kind == "gpio" and c.source == "spec/hardware.yaml":
            key = (c.key, c.detail)
            if key in seen_gpio_role:
                warnings.append(
                    f"SPEC ERROR: Duplicate claim on GPIO{c.key} for role '{c.detail}' in spec/hardware.yaml"  # noqa: E501
                )
            seen_gpio_role.add(key)

    armed = armed_owners(tractor, hwcfg)

    # Deployment warnings and blockers
    dep_warnings, dep_blockers = check_deployment(hw, tractor, hwcfg, armed)
    warnings += dep_warnings

    # Ultrasonic unverified overlap warnings
    warnings += check_ultrasonic_hard_conflicts(gpio_claims)

    # Registry versus the config and backend sources
    drift_errors, drift_warnings = check_registry_sources(root, hw, tractor, hwcfg)
    warnings += drift_warnings

    # Find conflicts (only verified/documented claims participate)
    conflict_claims = [c for c in all_claims if c.status != "unverified"]
    detected = find_conflicts(conflict_claims)

    # Load allow-list
    try:
        allowed = load_allow_list(root)
    except RegistryError as e:
        return Report(errors=[str(e)])

    report = reconcile(detected, allowed, armed)
    report.warnings = warnings
    report.errors += drift_errors
    report.blocked_deps = dep_blockers

    # GPIOs the registry leaves unclaimed (every claim counts, unverified and legacy ultrasonic too)
    claimed = {c.key for c in gpio_claims}
    for pins in (_load_ultrasonic_pins(hw) or {}).values():
        claimed.update(pins.values())
    report.unclaimed = sorted(USABLE_GPIOS - claimed)
    report.errors += lint_owner_actions(allowed, claimed)
    return report


def _fmt_key(kind: str, key: int | str) -> str:
    return f"{kind.upper()} {key}"


def print_report(report: Report, verbose: bool = False) -> int:
    """Print report; return exit code (0 ok, 1 fail)."""
    has_fail = False

    if report.errors:
        for e in report.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        has_fail = True

    if report.blocked:
        for c in report.blocked:
            labels = ", ".join(sorted({cl.label for cl in c.claimants}))
            switches = ", ".join(_switch_text(o) for o in c.blocked_by)
            print(
                f"BLOCKED ({switches}): {_fmt_key(c.kind, c.key)} claimed by: {labels}",
                file=sys.stderr,
            )
        has_fail = True

    if report.blocked_deps:
        for owner, message in report.blocked_deps:
            print(f"BLOCKED ({_switch_text(owner)}): {message}", file=sys.stderr)
        has_fail = True

    if report.new:
        for c in report.new:
            labels = ", ".join(sorted({cl.label for cl in c.claimants}))
            print(
                f"NEW CONFLICT (not in allow-list): {_fmt_key(c.kind, c.key)} claimed by: {labels}",
                file=sys.stderr,
            )
        has_fail = True

    if report.stale:
        for a in report.stale:
            key = f"GPIO{a.pin}" if a.kind == "gpio" else f"I2C {a.key[1]}"
            print(
                f"STALE ALLOW-LIST ENTRY: {key} (claimants: {', '.join(a.claimants)}) no longer conflicts",  # noqa: E501
                file=sys.stderr,
            )
        has_fail = True

    if report.known:
        print("Known conflicts (allow-listed):")
        for c in report.known:
            labels = ", ".join(sorted({cl.label for cl in c.claimants}))
            print(f"  {_fmt_key(c.kind, c.key)}: {labels}")
        print(
            "Header GPIOs no registry row claims (candidates for a move; each still needs the "
            "audit M-02 check with the HAT fitted): "
            + (", ".join(str(g) for g in report.unclaimed) or "none")
        )

    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  {w}")

    if not has_fail and not report.known and not report.warnings:
        print("check_hardware_pin_conflicts: OK — no GPIO or I2C address conflicts found.")

    return 1 if has_fail else 0


def self_test() -> int:
    """Run inline unit tests."""
    # Test 1: basic GPIO conflict
    gpio_claims, _ = collect_gpio_claims(
        {"pins": {"gpio_used": [{"gpio": 6, "role": "A"}]}},
        {"relays": {"x": {"gpio": 6}}},
        {},
    )
    detected = find_conflicts(gpio_claims)
    assert ("gpio", 6) in detected
    assert len(detected[("gpio", 6)]) == 2

    # Test 2: rxd/txd rows map to the right GPIOs via the header map, and do NOT collide
    gpio_claims2, _ = collect_gpio_claims(
        {
            "pins": {
                "gpio_used": [
                    {"pin": 33, "rxd": 4, "role": "RX"},
                    {"pin": 32, "txd": 4, "role": "TX"},
                ]
            }
        },
        {},
        {},
    )
    spec_claims = [c for c in gpio_claims2 if c.source == "spec/hardware.yaml"]
    assert {c.detail: c.key for c in spec_claims} == {"RX": 13, "TX": 12}, spec_claims
    assert find_conflicts(spec_claims) == {}, "rxd/txd rows falsely collided"

    # Test 3: txd on pin 32 collides with gpio=12 row
    gpio_claims3, _ = collect_gpio_claims(
        {
            "pins": {
                "gpio_used": [
                    {"pin": 32, "gpio": 12, "role": "ToF IRQ"},
                    {"pin": 32, "txd": 4, "role": "UART4 TX"},
                ]
            }
        },
        {},
        {},
    )
    detected3 = find_conflicts(gpio_claims3)
    assert ("gpio", 12) in detected3
    assert len(detected3[("gpio", 12)]) == 2

    # Test 4: I2C string vs int normalization, bus-aware
    i2c_claims, _ = collect_i2c_claims(
        {"pins": {"i2c_addresses": {"0x40": "INA3221"}}},
        {"pca9685": {"address": 0x40}},
    )
    assert ("i2c", "bus1:0x40") in find_conflicts(i2c_claims)
    i2c_claims, _ = collect_i2c_claims(
        {"pins": {"i2c_addresses": {"0x40": "INA3221"}}},
        {"pca9685": {"address": 0x40, "bus": 13}},
    )
    assert find_conflicts(i2c_claims) == {}, "the same address on another bus is no conflict"

    # Test 5: allow-list reconciliation
    allowed = [
        AllowEntry(
            kind="gpio",
            pin=6,
            claimants=["A", "B"],
            gap=["H-01"],
            reason="test",
            owner_action="test",
        ),
    ]
    # matching
    detected = {("gpio", 6): [Claim("gpio", 6, "A", "s", "A"), Claim("gpio", 6, "B", "s", "B")]}
    r = reconcile(detected, allowed)
    assert len(r.known) == 1
    # changed claimants
    detected = {
        ("gpio", 6): [
            Claim("gpio", 6, "A", "s", "A"),
            Claim("gpio", 6, "B", "s", "B"),
            Claim("gpio", 6, "C", "s", "C"),
        ]
    }
    r = reconcile(detected, allowed)
    assert len(r.new) == 1 and len(r.stale) == 1
    # stale
    detected = {}
    r = reconcile(detected, allowed)
    assert len(r.stale) == 1

    # Test 6: an armed owner blocks an allow-listed conflict that one of its claims is in,
    # whatever the claim's label says
    allowed = [
        AllowEntry(
            kind="gpio",
            pin=5,
            claimants=["HaLow", "Starter"],
            gap=["H-02b"],
            reason="test",
            owner_action="test",
        )
    ]
    detected = {
        ("gpio", 5): [
            Claim("gpio", 5, "HaLow", "s", "H"),
            Claim("gpio", 5, "Starter", "s", "T", owner="tractor"),
        ]
    }
    r = reconcile(detected, allowed, {"tractor"})
    assert len(r.blocked) == 1 and r.blocked[0].blocked_by == ["tractor"]
    assert len(reconcile(detected, allowed, {"ibt4"}).known) == 1
    assert len(reconcile(detected, allowed).known) == 1

    # Test 7: tractor values that cannot be right are rejected; omitted keys mean the runtime
    # default (so a deleted address is the 0x40 that collides with the INA3221)
    for bad in (
        {"relays": {"starter": {"gpio": 31}}},
        {"relays": {"starter": {"gpio": 99}}},
        {"relays": {"starter": {"gpio": 0}}},
        {"relays": {"starter": {"gpio": True}}},
        {"actuators": {"throttle": {"channel": 0}}},
        {"actuators": {"throttle": {"channel": 16}}},
        {"pca9685": {"address": 0x400}},
    ):
        try:
            validate_tractor(bad)
        except RegistryError:
            continue
        raise AssertionError(f"accepted a bad tractor config: {bad}")
    claims, warns = collect_i2c_claims({"pins": {"i2c_addresses": {"0x40": "INA3221"}}}, {})
    assert ("i2c", "bus1:0x40") in find_conflicts(claims) and warns

    print("self-test OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware pin/address conflict checker")
    parser.add_argument("--self-test", action="store_true", help="Run inline unit tests")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    try:
        report = run_check(args.root)
        return print_report(report, args.verbose)
    except RegistryError as e:
        print(f"check_hardware_pin_conflicts: ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"check_hardware_pin_conflicts: UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
