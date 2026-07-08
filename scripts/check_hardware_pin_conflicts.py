#!/usr/bin/env python3
"""
Hardware pin/address conflict checker (Constitution Principle V).

Parses the mower's pin/address registry in spec/hardware.yaml (`pins.gpio_used`,
`pins.i2c_addresses`) and the tractor's relay/PCA9685 assignments in
config/tractor.yaml (`relays.*.gpio`, `pca9685.address`), then fails if any GPIO
number or I2C address is claimed by more than one role across the merged set.

This is the enforcement half of spec/hardware.yaml's `tractor:` section: that
section documents the constraint ("must not collide"), this script is what
actually checks it. Wired into .github/workflows/hardware-guard.yml.

Exit codes: 0 = no conflicts, 1 = conflict(s) found or a file failed to parse.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
HARDWARE_YAML = ROOT / "spec" / "hardware.yaml"
TRACTOR_YAML = ROOT / "config" / "tractor.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _normalize_i2c_addr(value: Any) -> str | None:
    """Normalize an I2C address to a canonical "0xNN" lowercase string.

    spec/hardware.yaml stores addresses as quoted strings (e.g. "0x40"); a
    tractor pca9685.address may be written as a YAML int (0x40 -> 64) or a
    string. Compare on a normalized form so int vs. string never masks a
    real collision.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return hex(value)
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return hex(int(text, 16))
    return hex(int(text))  # plain decimal string, if ever used


def collect_gpio_claims(hardware: dict[str, Any], tractor: dict[str, Any]) -> dict[int, list[str]]:
    """Map GPIO number -> list of "source: role" claims on it."""
    claims: dict[int, list[str]] = {}

    for entry in hardware.get("pins", {}).get("gpio_used", []) or []:
        gpio = entry.get("gpio")  # deliberately .get(): rxd/txd rows have no "gpio" key
        if not gpio:
            continue
        claims.setdefault(gpio, []).append(f"spec/hardware.yaml: {entry.get('role', '?')}")

    for relay_name, relay in (tractor.get("relays", {}) or {}).items():
        gpio = relay.get("gpio")
        if not gpio:
            continue
        claims.setdefault(gpio, []).append(f"config/tractor.yaml relays.{relay_name}")

    return claims


def collect_i2c_claims(hardware: dict[str, Any], tractor: dict[str, Any]) -> dict[str, list[str]]:
    """Map normalized I2C address -> list of "source: role" claims on it."""
    claims: dict[str, list[str]] = {}

    for addr, device in (hardware.get("pins", {}).get("i2c_addresses", {}) or {}).items():
        norm = _normalize_i2c_addr(addr)
        if norm is None:
            continue
        claims.setdefault(norm, []).append(f"spec/hardware.yaml: {device}")

    pca9685_addr = (tractor.get("pca9685", {}) or {}).get("address")
    norm = _normalize_i2c_addr(pca9685_addr)
    if norm is not None:
        claims.setdefault(norm, []).append("config/tractor.yaml pca9685.address")

    return claims


def find_conflicts(claims: dict[Any, list[str]]) -> dict[Any, list[str]]:
    return {key: sources for key, sources in claims.items() if len(sources) > 1}


def main() -> int:
    hardware = _load_yaml(HARDWARE_YAML)
    tractor = _load_yaml(TRACTOR_YAML)

    gpio_conflicts = find_conflicts(collect_gpio_claims(hardware, tractor))
    i2c_conflicts = find_conflicts(collect_i2c_claims(hardware, tractor))

    if not gpio_conflicts and not i2c_conflicts:
        print("check_hardware_pin_conflicts: OK — no GPIO or I2C address conflicts found.")
        return 0

    print("check_hardware_pin_conflicts: CONFLICTS FOUND", file=sys.stderr)
    for gpio, sources in sorted(gpio_conflicts.items()):
        print(f"  GPIO {gpio} claimed by multiple roles:", file=sys.stderr)
        for source in sources:
            print(f"    - {source}", file=sys.stderr)
    for addr, sources in sorted(i2c_conflicts.items()):
        print(f"  I2C address {addr} claimed by multiple roles:", file=sys.stderr)
        for source in sources:
            print(f"    - {source}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        # ponytail: smallest runnable check for this script's own logic, not a test
        # framework — run with `python3 scripts/check_hardware_pin_conflicts.py --self-test`.
        gpio_claims = collect_gpio_claims(
            {"pins": {"gpio_used": [{"gpio": 6, "role": "A"}]}},
            {"relays": {"x": {"gpio": 6}}},
        )
        assert find_conflicts(gpio_claims) == {6: ["spec/hardware.yaml: A", "config/tractor.yaml relays.x"]}

        # rxd/txd rows (no "gpio" key) must not be treated as GPIO claims or crash.
        gpio_claims_bno = collect_gpio_claims(
            {"pins": {"gpio_used": [{"pin": 33, "rxd": 4, "role": "RX"}, {"pin": 32, "txd": 4, "role": "TX"}]}},
            {},
        )
        assert find_conflicts(gpio_claims_bno) == {}, "rxd/txd rows falsely collided"

        # string "0x40" (spec/hardware.yaml style) vs. int 0x40 (YAML tractor style) must collide.
        i2c_claims = collect_i2c_claims(
            {"pins": {"i2c_addresses": {"0x40": "INA3221"}}},
            {"pca9685": {"address": 0x40}},
        )
        assert find_conflicts(i2c_claims) == {
            "0x40": ["spec/hardware.yaml: INA3221", "config/tractor.yaml pca9685.address"]
        }
        print("self-test OK")
        sys.exit(0)

    sys.exit(main())
