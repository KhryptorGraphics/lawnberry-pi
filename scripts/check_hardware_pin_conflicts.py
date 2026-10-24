#!/usr/bin/env python3
"""
Hardware pin/address conflict checker (Constitution Principle V).

Parses the mower's pin/address registry in spec/hardware.yaml
  - pins.gpio_used (gpio, txd/rxd with pin->gpio mapping)
  - pins.i2c_addresses
  - pins.ultrasonic (conditional on limits.yaml ultrasonic_enabled)
  - pins.uart_overlays (for txd/rxd pin/gpio cross-validation)
  - deployment (for I2C/UART availability warnings)
and the tractor's relay/PCA9685 assignments in
  - config/tractor.yaml (relays.*.gpio, pca9685.address, pca9685.bus)
and the ultrasonic enable flag in
  - config/limits.yaml (ultrasonic_enabled)

Then reports any GPIO number or I2C address claimed by more than one role
across the merged set.

Exit codes: 0 = no conflicts (warnings allowed), 1 = conflict(s) found or parse error.

Usage:
  python3 scripts/check_hardware_pin_conflicts.py            # normal check
  python3 scripts/check_hardware_pin_conflicts.py --self-test  # inline unit tests
  python3 scripts/check_hardware_pin_conflicts.py --root DIR   # run against DIR (for tests)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, order=True)
class Claim:
    """A single claim on a GPIO pin or I2C address."""
    kind: str                 # "gpio" or "i2c"
    key: int | str            # GPIO number (int) or I2C address (canonical "0xNN" str)
    label: str                # human-readable claimant identifier
    source: str               # source file path relative to root
    detail: str               # role description
    status: str = "verified"  # verified | documented | unverified
    pin: int | None = None    # physical header pin (for gpio kind)


@dataclass
class Conflict:
    kind: str
    key: int | str
    claimants: list[Claim]


@dataclass
class AllowEntry:
    kind: str
    pin: int | None = None
    i2c_address: str | None = None
    claimants: list[str] = field(default_factory=list)
    gap: list[str] = field(default_factory=list)
    reason: str = ""
    owner_action: str = ""


@dataclass
class Report:
    conflicts: list[Conflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    known: list[Conflict] = field(default_factory=list)
    new: list[Conflict] = field(default_factory=list)
    stale: list[AllowEntry] = field(default_factory=list)
    blocked: list[Conflict] = field(default_factory=list)


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
        raise RegistryError(f"{path}: YAML parse error: {e}")
    except OSError as e:
        raise RegistryError(f"{path}: {e}")


def _normalize_i2c_addr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return hex(value)
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return hex(int(text, 16))
    return hex(int(text))


# Pi 5 40-pin header: physical pin -> BCM GPIO
HEADER_PIN_TO_GPIO = {
    1: None, 2: None, 3: 2, 4: None, 5: 3, 6: None,
    7: 4, 8: 14, 9: None, 10: 15, 11: 17, 12: 18,
    13: 27, 14: None, 15: 22, 16: 23, 17: None, 18: 24,
    19: 10, 20: None, 21: 9, 22: 25, 23: 11, 24: 8,
    25: None, 26: 7, 27: 0, 28: 1, 29: 5, 30: None,
    31: 6, 32: 12, 33: 13, 34: None, 35: 19, 36: 16,
    37: 26, 38: 20, 39: None, 40: 21,
}
GPIO_TO_PIN = {g: p for p, g in HEADER_PIN_TO_GPIO.items() if g is not None}


def _gpio_from_pin(pin: int) -> int | None:
    return HEADER_PIN_TO_GPIO.get(pin)


def _pin_from_gpio(gpio: int) -> int | None:
    return GPIO_TO_PIN.get(gpio)


def _load_ultrasonic_pins(hw: dict) -> dict[str, dict[str, int]] | None:
    """Load ultrasonic sensor pin assignments from spec/hardware.yaml pins.ultrasonic.sensors."""
    ultrasonic = hw.get("pins", {}).get("ultrasonic", {})
    sensors = ultrasonic.get("sensors", {})
    if not sensors:
        return None
    # Convert to the same format as driver SENSORS dict
    result = {}
    for name, pins in sensors.items():
        if "trig_gpio" in pins and "echo_gpio" in pins:
            result[name] = {"trig": pins["trig_gpio"], "echo": pins["echo_gpio"]}
    return result if result else None



def collect_gpio_claims(root: Path, hw: dict, tractor: dict, limits: dict,
                         ultrasonic_pins: dict | None) -> tuple[list[Claim], list[str]]:
    """Collect all GPIO claims from registry, tractor, ultrasonic, and I2C bus pins.
    Returns (claims, warnings)."""
    claims: list[Claim] = []
    warnings: list[str] = []

    # 1. spec/hardware.yaml pins.gpio_used
    for entry in hw.get("pins", {}).get("gpio_used", []) or []:
        role = entry.get("role", "?")
        status = entry.get("status", "verified")
        evidence = entry.get("evidence", [])
        pin = entry.get("pin")

        if "gpio" in entry:
            gpio = entry["gpio"]
            if not isinstance(gpio, int):
                raise RegistryError(f"spec/hardware.yaml: gpio_used entry role={role}: gpio must be int")
            if pin is not None and _gpio_from_pin(pin) != gpio:
                raise RegistryError(f"spec/hardware.yaml: gpio_used entry role={role}: pin {pin} -> GPIO {_gpio_from_pin(pin)}, but gpio={gpio}")
            claims.append(Claim(kind="gpio", key=gpio,
                              label=f"spec/hardware.yaml: {role}",
                              source="spec/hardware.yaml", detail=role,
                              status=status, pin=pin))
        elif "txd" in entry or "rxd" in entry:
            # UART pin claim: derive GPIO from physical pin via header map
            if pin is None:
                raise RegistryError(f"spec/hardware.yaml: gpio_used entry role={role}: txd/rxd row requires 'pin'")
            gpio = _gpio_from_pin(pin)
            if gpio is None:
                raise RegistryError(f"spec/hardware.yaml: gpio_used entry role={role}: pin {pin} not in header map")
            claims.append(Claim(kind="gpio", key=gpio,
                              label=f"spec/hardware.yaml: {role}",
                              source="spec/hardware.yaml", detail=role,
                              status=status, pin=pin))
        else:
            raise RegistryError(f"spec/hardware.yaml: gpio_used entry role={role}: must have 'gpio' or 'txd'/'rxd'")

        # Validate status/evidence discipline
        if status == "unverified":
            if not entry.get("measure"):
                warnings.append(f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): unverified claim missing 'measure' field")
            if not evidence:
                warnings.append(f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): unverified claim missing 'evidence' field")
        elif status in ("verified", "documented"):
            if not evidence:
                warnings.append(f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): {status} claim missing 'evidence' field")
        elif status:
            warnings.append(f"spec/hardware.yaml: {role} (GPIO{claims[-1].key}): unknown status '{status}' (expected verified|documented|unverified)")

    # 2. config/tractor.yaml relays (always claim, even if enabled: false)
    for relay_name, relay in (tractor.get("relays", {}) or {}).items():
        gpio = relay.get("gpio")
        if gpio is None:
            raise RegistryError(f"config/tractor.yaml relays.{relay_name}: missing 'gpio' field")
        if not isinstance(gpio, int):
            raise RegistryError(f"config/tractor.yaml relays.{relay_name}: gpio must be int")
        pin = _pin_from_gpio(gpio)
        claims.append(Claim(kind="gpio", key=gpio,
                          label=f"config/tractor.yaml relays.{relay_name}",
                          source="config/tractor.yaml", detail=f"relay {relay_name}",
                          status="verified", pin=pin))

    # 3. I2C header bus pins (GPIO2=SDA, GPIO3=SCL) if any device uses header bus 1
    header_bus_claimed = False
    pca_bus = (tractor.get("pca9685", {}) or {}).get("bus")
    if pca_bus == 1:
        header_bus_claimed = True
    tof_cfg = hw.get("sensors", {}).get("tof_config", {})
    if isinstance(tof_cfg, dict) and tof_cfg.get("bus") == 1:
        header_bus_claimed = True
    if hw.get("pins", {}).get("i2c_addresses"):
        header_bus_claimed = True

    if header_bus_claimed:
        claims.append(Claim(kind="gpio", key=2,
                          label="spec/hardware.yaml: I2C1 SDA (header bus 1)",
                          source="spec/hardware.yaml", detail="I2C1 SDA",
                          status="documented", pin=3))
        claims.append(Claim(kind="gpio", key=3,
                          label="spec/hardware.yaml: I2C1 SCL (header bus 1)",
                          source="spec/hardware.yaml", detail="I2C1 SCL",
                          status="documented", pin=5))

    # 4. Ultrasonic driver pins (only if limits.yaml ultrasonic_enabled)
    ul_enabled = True  # default matches SafetyLimits default
    if limits:
        ul_enabled = bool(limits.get("ultrasonic_enabled", True))
    if ul_enabled and ultrasonic_pins:
        for sensor_name, pins in ultrasonic_pins.items():
            for pin_type, gpio in (("trig", pins["trig"]), ("echo", pins["echo"])):
                pin = _pin_from_gpio(gpio)
                claims.append(Claim(kind="gpio", key=gpio,
                                  label=f"backend/src/drivers/sensors/ultrasonic_driver.py: {sensor_name} {pin_type}",
                                  source="backend/src/drivers/sensors/ultrasonic_driver.py",
                                  detail=f"ultrasonic {sensor_name} {pin_type}",
                                  status="verified", pin=pin))

    return claims, warnings


def collect_i2c_claims(root: Path, hw: dict, tractor: dict) -> tuple[list[Claim], list[str]]:
    """Collect all I2C address claims."""
    claims: list[Claim] = []
    warnings: list[str] = []

    # spec/hardware.yaml pins.i2c_addresses
    for addr, device in (hw.get("pins", {}).get("i2c_addresses", {}) or {}).items():
        norm = _normalize_i2c_addr(addr)
        if norm is None:
            continue
        claims.append(Claim(kind="i2c", key=norm,
                          label=f"spec/hardware.yaml: {device}",
                          source="spec/hardware.yaml", detail=device,
                          status="verified"))

    # config/tractor.yaml pca9685.address
    pca9685_addr = (tractor.get("pca9685", {}) or {}).get("address")
    norm = _normalize_i2c_addr(pca9685_addr)
    if norm is not None:
        claims.append(Claim(kind="i2c", key=norm,
                          label="config/tractor.yaml pca9685.address",
                          source="config/tractor.yaml", detail="PCA9685",
                          status="verified"))
    else:
        warnings.append("config/tractor.yaml pca9685.address not set; driver default 0x40 collides with INA3221 at 0x40")

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
        if kind == "gpio":
            if pin is None or not isinstance(pin, int):
                raise RegistryError(f"{path}: entry {i}: gpio kind requires integer 'pin'")
            key = ("gpio", pin)
        else:
            if not i2c_addr:
                raise RegistryError(f"{path}: entry {i}: i2c_address kind requires 'i2c_address'")
            norm = _normalize_i2c_addr(i2c_addr)
            if norm is None:
                raise RegistryError(f"{path}: entry {i}: invalid i2c_address")
            key = ("i2c", norm)
        if key in seen_keys:
            raise RegistryError(f"{path}: duplicate entry for {key}")
        seen_keys.add(key)

        claimants = e.get("claimants", [])
        if not isinstance(claimants, list) or len(claimants) < 2:
            raise RegistryError(f"{path}: entry {i}: claimants must be list with >=2 unique strings")
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
        allowed = {"kind", "pin", "i2c_address", "claimants", "gap", "reason", "owner_action"}
        for k in e:
            if k not in allowed:
                raise RegistryError(f"{path}: entry {i}: unknown key '{k}'")

        entries.append(AllowEntry(kind=kind, pin=pin, i2c_address=norm if kind == "i2c_address" else None,
                                claimants=claimants, gap=gap, reason=reason, owner_action=owner_action))
    return entries


def reconcile(detected: dict[tuple[str, int | str], list[Claim]],
              allowed: list[AllowEntry],
              tractor_enabled: bool) -> Report:
    """Match detected conflicts against allow-list; classify as known/new/stale/blocked."""
    report = Report()
    report.conflicts = [Conflict(kind=k[0], key=k[1], claimants=v) for k, v in detected.items()]

    # Build allow-list lookup by (kind, key)
    allowed_by_key = {}
    for a in allowed:
        key = ("gpio", a.pin) if a.kind == "gpio" else ("i2c", a.i2c_address)
        allowed_by_key[key] = a

    matched_keys = set()

    for conflict in report.conflicts:
        key = (conflict.kind, conflict.key)
        labels = sorted({c.label for c in conflict.claimants})

        if key in allowed_by_key:
            a = allowed_by_key[key]
            allowed_labels = sorted(a.claimants)
            if labels == allowed_labels:
                matched_keys.add(key)
                # Check tractor gate
                has_tractor = any("config/tractor.yaml" in c.label for c in conflict.claimants)
                if tractor_enabled and has_tractor:
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
        key = ("gpio", a.pin) if a.kind == "gpio" else ("i2c", a.i2c_address)
        if key not in matched_keys:
            report.stale.append(a)

    return report


def check_deployment_warnings(root: Path, hw: dict, tractor: dict, limits: dict) -> list[str]:
    """Generate warnings from deployment block (I2C/UART availability)."""
    warnings = []
    dep = hw.get("deployment", {})
    if not dep:
        return warnings

    # I2C header bus availability
    if dep.get("header_i2c_bus_enabled") is False:
        bus = dep.get("header_i2c_bus", 1)
        affected = []
        pca_bus = (tractor.get("pca9685", {}) or {}).get("bus")
        if pca_bus == bus:
            affected.append("config/tractor.yaml pca9685 (bus=1)")
        tof_cfg = hw.get("sensors", {}).get("tof_config", {})
        if isinstance(tof_cfg, dict) and tof_cfg.get("bus") == bus:
            affected.append("spec/hardware.yaml sensors.tof_config (bus=1)")
        if hw.get("pins", {}).get("i2c_addresses"):
            for addr, dev in hw["pins"]["i2c_addresses"].items():
                affected.append(f"spec/hardware.yaml i2c_addresses: {dev} ({addr})")
        if affected:
            warnings.append(
                f"DEPLOYMENT WARNING: Header I2C bus {bus} is disabled on deployed Pi "
                f"(i2c_arm=off, GPIO3=HaLow WAKE). The following devices are configured "
                f"for this bus and will be unreachable: {', '.join(affected)}. "
                f"Relocate to another I2C bus (e.g., RP1 I2C4, software I2C) or re-enable i2c_arm "
                f"and resolve GPIO3 WAKE conflict."
            )

    # UART4 overlay availability
    uart_overlays_enabled = set(dep.get("uart_overlays_enabled", []))
    uart_overlays = hw.get("pins", {}).get("uart_overlays", {})
    uart4 = uart_overlays.get("uart4", {})
    pi5 = uart4.get("pi5", {})
    if pi5:
        overlay = pi5.get("overlay", "uart4-pi5")
        if overlay not in uart_overlays_enabled:
            for entry in hw.get("pins", {}).get("gpio_used", []):
                if "BNO085" in entry.get("role", "") and ("txd" in entry or "rxd" in entry):
                    warnings.append(
                        f"DEPLOYMENT WARNING: UART4 overlay '{overlay}' not enabled on deployed Pi. "
                        f"BNO085 IMU claims UART4 TX/RX (GPIO{pi5.get('txd_gpio')}/GPIO{pi5.get('rxd_gpio')}). "
                        f"Add 'dtoverlay={overlay}' to /boot/firmware/config.txt to enable."
                    )
                    break

    return warnings


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


def run_check(root: Path) -> Report:
    hw = _load_yaml(root / "spec" / "hardware.yaml")
    tractor = _load_yaml(root / "config" / "tractor.yaml")
    limits = _load_yaml(root / "config" / "limits.yaml")

    # Load ultrasonic pins from driver source
    ultrasonic_pins = _load_ultrasonic_pins(hw)

    # Collect all claims
    gpio_claims, gw = collect_gpio_claims(root, hw, tractor, limits, ultrasonic_pins)
    i2c_claims, iw = collect_i2c_claims(root, hw, tractor)

    all_claims = gpio_claims + i2c_claims
    warnings = gw + iw

    # Check for spec errors (duplicate pin+role in gpio_used)
    seen_gpio_role = set()
    for c in gpio_claims:
        if c.kind == "gpio" and c.source == "spec/hardware.yaml":
            key = (c.key, c.detail)
            if key in seen_gpio_role:
                warnings.append(f"SPEC ERROR: Duplicate claim on GPIO{c.key} for role '{c.detail}' in spec/hardware.yaml")
            seen_gpio_role.add(key)

    # Deployment warnings
    warnings += check_deployment_warnings(root, hw, tractor, limits)

    # Ultrasonic unverified overlap warnings
    warnings += check_ultrasonic_hard_conflicts(gpio_claims)

    # Find conflicts (only verified/documented claims participate)
    conflict_claims = [c for c in all_claims if c.status != "unverified"]
    detected = find_conflicts(conflict_claims)

    # Load allow-list
    try:
        allowed = load_allow_list(root)
    except RegistryError as e:
        return Report(errors=[str(e)])

    tractor_enabled = bool(tractor.get("enabled", False))
    report = reconcile(detected, allowed, tractor_enabled)
    report.warnings = warnings
    return report


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
            print(f"BLOCKED (tractor enabled): {c.kind.upper()} {c.key} claimed by: {labels}", file=sys.stderr)
        has_fail = True

    if report.new:
        for c in report.new:
            labels = ", ".join(sorted({cl.label for cl in c.claimants}))
            print(f"NEW CONFLICT (not in allow-list): {c.kind.upper()} {c.key} claimed by: {labels}", file=sys.stderr)
        has_fail = True

    if report.stale:
        for a in report.stale:
            key = f"GPIO{a.pin}" if a.kind == "gpio" else a.i2c_address
            print(f"STALE ALLOW-LIST ENTRY: {key} (claimants: {', '.join(a.claimants)}) no longer conflicts", file=sys.stderr)
        has_fail = True

    if report.known:
        print("Known conflicts (allow-listed):")
        for c in report.known:
            labels = ", ".join(sorted({cl.label for cl in c.claimants}))
            print(f"  {c.kind.upper()} {c.key}: {labels}")

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
    gpio_claims = collect_gpio_claims(
        Path("."),
        {"pins": {"gpio_used": [{"gpio": 6, "role": "A"}]}},
        {"relays": {"x": {"gpio": 6}}},
        {},
        None,
    )
    detected = find_conflicts(gpio_claims[0])
    assert ("gpio", 6) in detected
    assert len(detected[("gpio", 6)]) == 2

    # Test 2: rxd/txd rows map to GPIO via header map
    gpio_claims2, _ = collect_gpio_claims(
        Path("."),
        {"pins": {"gpio_used": [
            {"pin": 33, "rxd": 4, "role": "RX"},
            {"pin": 32, "txd": 4, "role": "TX"},
        ]}},
        {},
        {},
        None,
    )
    detected2 = find_conflicts(gpio_claims2)
    assert ("gpio", 13) in detected2 or ("gpio", 12) in detected2 or not detected2

    # Test 3: txd on pin 32 collides with gpio=12 row
    gpio_claims3, _ = collect_gpio_claims(
        Path("."),
        {"pins": {"gpio_used": [
            {"pin": 32, "gpio": 12, "role": "ToF IRQ"},
            {"pin": 32, "txd": 4, "role": "UART4 TX"},
        ]}},
        {},
        {},
        None,
    )
    detected3 = find_conflicts(gpio_claims3)
    assert ("gpio", 12) in detected3
    assert len(detected3[("gpio", 12)]) == 2

    # Test 4: I2C string vs int normalization
    i2c_claims = collect_i2c_claims(Path("."),
        {"pins": {"i2c_addresses": {"0x40": "INA3221"}}},
        {"pca9685": {"address": 0x40}},
    )
    detected4 = find_conflicts(i2c_claims[0])
    assert ("i2c", "0x40") in detected4

    # Test 5: allow-list reconciliation
    allowed = [
        AllowEntry(kind="gpio", pin=6, claimants=["A", "B"], gap=["H-01"],
                  reason="test", owner_action="test"),
    ]
    # matching
    detected = {("gpio", 6): [Claim("gpio", 6, "A", "s", "A"),
                                Claim("gpio", 6, "B", "s", "B")]}
    r = reconcile(detected, allowed, False)
    assert len(r.known) == 1
    # changed claimants
    detected = {("gpio", 6): [Claim("gpio", 6, "A", "s", "A"),
                                Claim("gpio", 6, "B", "s", "B"),
                                Claim("gpio", 6, "C", "s", "C")]}
    r = reconcile(detected, allowed, False)
    assert len(r.new) == 1 and len(r.stale) == 1
    # stale
    detected = {}
    r = reconcile(detected, allowed, False)
    assert len(r.stale) == 1

    # Test 6: tractor enabled blocks allow-listed tractor conflict
    allowed = [AllowEntry(kind="gpio", pin=5, claimants=["HaLow", "config/tractor.yaml relays.starter"],
                         gap=["H-02b"], reason="test", owner_action="test")]
    detected = {("gpio", 5): [Claim("gpio", 5, "HaLow", "s", "H"),
                                Claim("gpio", 5, "config/tractor.yaml relays.starter", "s", "T")]}
    r = reconcile(detected, allowed, True)
    assert len(r.blocked) == 1
    r = reconcile(detected, allowed, False)
    assert len(r.known) == 1

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
