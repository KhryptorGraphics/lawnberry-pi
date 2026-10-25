"""IBT-4 configuration, the singleton's config retry, relay claim level and gpiochip strictness.

PR #20 review BL-11 (a one-off config failure cached the flag-only stub for ever and it answered
"accepted"), BL-12 (``on_levels`` (1, 1)), BL-13 / mutants T03 and T04 (the relay claim level and
polarity were never asserted) and SUSPECTED S-2 (``open_gpiochip`` fell back to another bank).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fake_lgpio import FakeLgpio
from pydantic import ValidationError

from backend.src.core.config_loader import ConfigLoader
from backend.src.drivers.actuators.tractor_actuators import RelayActuator
from backend.src.drivers.base import open_gpiochip
from backend.src.models.hardware_config import Ibt4Config, Ibt4Levels
from backend.src.services import blade_service

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- BL-12


@pytest.mark.parametrize("levels", [(0, 0), (1, 1)])
def test_ibt4_on_levels_need_exactly_one_input_high(levels):
    with pytest.raises(ValidationError, match="exactly one input must be high"):
        Ibt4Levels(in1=levels[0], in2=levels[1])
    with pytest.raises(ValidationError):
        Ibt4Config(on_levels={"in1": levels[0], "in2": levels[1]})


def test_ibt4_defaults_and_deadman_bounds():
    cfg = Ibt4Config()
    assert (cfg.on_levels.in1, cfg.on_levels.in2) == (1, 0) and cfg.deadman_ms == 2000
    for bad in (0, 199, 10_001, -5):
        with pytest.raises(ValidationError):
            Ibt4Config(deadman_ms=bad)
    assert Ibt4Config(deadman_ms=200).deadman_ms == 200
    assert Ibt4Config(deadman_ms=10_000).deadman_ms == 10_000


def test_the_shipped_hardware_yaml_still_loads_with_the_deadman():
    hardware, _limits = ConfigLoader(config_dir=str(ROOT / "config")).get()
    assert hardware.ibt4_config.gpio_enabled is False
    assert hardware.ibt4_config.deadman_ms == 2000


# ----------------------------------------------------------------------------- BL-11


def _write_config(tmp_path, ibt4_block: str) -> None:
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    (tmp_path / "hardware.yaml").write_text(ibt4_block, encoding="utf-8")


BROKEN = "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 99}\n"  # an operator mid-edit
ARMED = "ibt4:\n  gpio_enabled: true\n  pins: {in1: 24, in2: 25}\n  deadman_ms: 1500\n"


async def test_a_one_off_config_failure_is_retried_instead_of_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(blade_service, "_blade_service", None)
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: None)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)

    _write_config(tmp_path, BROKEN)
    stub = blade_service.get_blade_service()
    assert stub.config_error and stub._driver.gpio_enabled is False
    # Outside SIM_MODE the flag-only stub drives nothing, so it must not say "accepted".
    assert await stub.set_active(True) is False and "not armed" in stub.last_refusal
    assert stub.config_error.split("(")[0] in stub.last_refusal
    await stub.emergency_stop()  # a latch set on the stub must survive the rebuild
    assert stub.emergency_latched is True

    _write_config(tmp_path, ARMED)
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)  # the 1 s retry interval elapsed
    fresh = blade_service.get_blade_service()

    assert fresh is not stub and fresh.config_error is None
    assert fresh._driver.gpio_enabled is True and fresh.deadman_s == 1.5
    assert fresh.emergency_latched is True, "the E-stop latch is carried over"
    assert blade_service.get_blade_service() is fresh, "and the readable config is cached again"
    assert fake.calls == [], "building the service touched no GPIO"


async def test_a_readable_config_is_never_rebuilt(monkeypatch, tmp_path):
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(blade_service, "_blade_service", None)
    _write_config(tmp_path, ARMED)
    first = blade_service.get_blade_service()
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)
    assert first.config_error is None and blade_service.get_blade_service() is first


# ----------------------------------------------------------------- BL-13: T03 and T04


@pytest.mark.parametrize("active_high", [True, False])
def test_a_relay_is_claimed_inactive_and_with_its_configured_polarity(monkeypatch, active_high):
    monkeypatch.setenv("SIM_MODE", "0")
    claims: list[tuple] = []

    class OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            claims.append((pin, active_high, initial_value))

        def on(self):
            pass

        def off(self):
            pass

    module = types.ModuleType("gpiozero")
    module.OutputDevice = OutputDevice
    monkeypatch.setitem(sys.modules, "gpiozero", module)

    relay = RelayActuator("blade_pto", 26, active_high, armed=True)
    relay.set(False)

    # initial_value False = the INACTIVE level (never energised at the claim) and the polarity
    # reaches gpiozero: an active-low relay claimed as active-high would be energised at rest.
    assert claims == [(26, active_high, False)]


# ----------------------------------------------------------------- S-2: gpiochip strictness


class _Chips:
    def __init__(self, openable: set[int]) -> None:
        self.openable, self.attempts = openable, []

    def gpiochip_open(self, chip):
        self.attempts.append(chip)
        if chip in self.openable:
            return 1000 + chip
        raise RuntimeError("can not open gpiochip")


def test_open_gpiochip_without_fallback_probes_only_the_first_candidate(monkeypatch):
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    chips = _Chips({0})
    with pytest.raises(RuntimeError, match="can not open gpiochip"):
        open_gpiochip(chips, fallback=False)
    assert chips.attempts == [4], "chip 0 would be a different bank on a Pi 5"
    assert open_gpiochip(_Chips({0})) == 1000, "the default probe order is unchanged"


def test_open_gpiochip_without_fallback_uses_the_first_entry_of_the_override(monkeypatch):
    monkeypatch.setenv("LAWNBERRY_GPIOCHIP", "0,4")
    chips = _Chips({0, 4})
    assert open_gpiochip(chips, fallback=False) == 1000
    assert chips.attempts == [0]
