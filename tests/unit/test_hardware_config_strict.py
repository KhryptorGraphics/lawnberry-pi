"""HW-05 / US3-07 (PR #20 review 3): ``ibt4:`` and ``ultrasonic:`` reject unknown keys.

``limits.yaml`` has been strict since round 2 (a misspelt key must not silently run the default).
The two ``hardware.yaml`` blocks ignored unknown keys, so an owner who tightened ``deadman_ms``,
``near_hold_s`` or ``alive_window_s`` with a typo got no error and the default. An unknown key is
now an error that names the key, at every level of both blocks, also in ``hardware.local.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.src.core.config_loader import ConfigLoader
from backend.src.models.hardware_config import (
    Ibt4Config,
    Ibt4Levels,
    Ibt4Pins,
    UltrasonicConfig,
)
from backend.src.services import blade_service

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def load(tmp_path: Path, hardware: dict, local: dict | None = None):
    (tmp_path / "hardware.yaml").write_text(yaml.safe_dump(hardware), encoding="utf-8")
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    if local is not None:
        (tmp_path / "hardware.local.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")
    return ConfigLoader(config_dir=str(tmp_path)).load()[0]


IBT4_TYPOS = [
    ("deadman_msec", {"deadman_msec": 500}),
    ("deadman", {"deadman": 100}),
    ("gpio_enable", {"gpio_enable": True}),
    ("in3", {"pins": {"in1": 24, "in2": 25, "in3": 7}}),
    ("in3", {"on_levels": {"in1": 1, "in2": 0, "in3": 1}}),
    ("pin", {"pin": {"in1": 24}}),
]
ULTRASONIC_TYPOS = [
    ("near_hold", {"near_hold": 5}),
    ("no_ecco_is_clear", {"near_hold_s": 5, "no_ecco_is_clear": True}),
    ("noecho_is_clear", {"noecho_is_clear": True}),
    ("alive_window", {"alive_window": 1}),
    ("stale_after", {"stale_after": 100}),
    ("port_path", {"port_path": "/dev/ttyACM9"}),
]


@pytest.mark.parametrize(("key", "block"), IBT4_TYPOS)
def test_an_unknown_ibt4_key_is_refused_by_name(key, block):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as excinfo:
        Ibt4Config(**block)
    assert key in str(excinfo.value)


@pytest.mark.parametrize(("key", "block"), ULTRASONIC_TYPOS)
def test_an_unknown_ultrasonic_key_is_refused_by_name(key, block):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as excinfo:
        UltrasonicConfig(**block)
    assert key in str(excinfo.value)


@pytest.mark.parametrize("model", [Ibt4Pins, Ibt4Levels])
def test_the_nested_ibt4_models_are_strict_on_their_own(model):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(extra_key=1)


@pytest.mark.parametrize(
    ("block_name", "block"),
    [("ibt4", {"deadman_msec": 500}), ("ultrasonic", {"near_hold": 5, "port": None})],
)
def test_the_loader_refuses_a_typo_in_hardware_yaml_and_names_it(tmp_path, block_name, block):
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        load(tmp_path, {block_name: block})
    text = str(excinfo.value)
    assert next(iter(block)) in text and "hardware.yaml" in text


def test_a_typo_in_the_local_override_is_refused_too(tmp_path):
    """``hardware.local.yaml`` is merged over ``hardware.yaml``: the place a typo is easiest."""
    base = {"ibt4": {"gpio_enabled": False, "deadman_ms": 2000}}
    assert load(tmp_path, base).ibt4_config.deadman_ms == 2000

    with pytest.raises((ValidationError, ValueError), match="deadmann_ms"):
        load(tmp_path, base, local={"ibt4": {"deadmann_ms": 1000}})
    with pytest.raises((ValidationError, ValueError), match="near_holds"):
        load(tmp_path, {}, local={"ultrasonic": {"near_holds": 60}})


def test_a_correct_override_still_loads(tmp_path):
    hardware = load(
        tmp_path,
        {"ibt4": {"deadman_ms": 2000}},
        local={
            "ibt4": {"deadman_ms": 1500, "on_levels": {"in1": 0, "in2": 1}},
            "ultrasonic": {"near_hold_s": 60, "alive_window_s": 5.0},
        },
    )
    assert hardware.ibt4_config.deadman_ms == 1500
    assert (hardware.ibt4_config.on_levels.in1, hardware.ibt4_config.on_levels.in2) == (0, 1)
    assert (hardware.ultrasonic_config.near_hold_s, hardware.ultrasonic_config.alive_window_s) == (
        60,
        5.0,
    )


def test_every_key_of_the_shipped_blocks_is_a_field_of_its_model():
    """The strict models must accept exactly what ``config/hardware.yaml`` ships."""
    raw = yaml.safe_load((REPO_CONFIG / "hardware.yaml").read_text(encoding="utf-8"))
    assert set(raw["ibt4"]) <= set(Ibt4Config.model_fields)
    assert set(raw["ibt4"]["pins"]) <= set(Ibt4Pins.model_fields)
    assert set(raw["ibt4"]["on_levels"]) <= set(Ibt4Levels.model_fields)
    assert set(raw["ultrasonic"]) <= set(UltrasonicConfig.model_fields)
    hardware = ConfigLoader(config_dir=str(REPO_CONFIG)).load()[0]
    assert hardware.ibt4_config is not None and hardware.ultrasonic_config is not None


async def test_the_blade_stays_refused_and_says_which_key_when_the_block_has_a_typo(
    monkeypatch, tmp_path
):
    """A typo in ``ibt4:`` must never arm the real adapter or fall back silently: the blade
    service is the unarmed stub, refuses an ON and names the key (and it is retried each second)."""
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text("ibt4:\n  gpio_enabled: true\n  deadman_msec: 500\n", encoding="utf-8")
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(blade_service, "_blade_service", None)
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)
    monkeypatch.setattr("backend.src.services.tractor_service.get_tractor_service", lambda: None)
    monkeypatch.setattr("backend.src.services.robohat_service.get_robohat_service", lambda: None)

    svc = blade_service.get_blade_service()

    assert svc._driver.gpio_enabled is False, "the broken block armed nothing"
    assert "deadman_msec" in svc.config_error
    assert await svc.set_active(True) is False
    assert "deadman_msec" in svc.last_refusal
    assert "deadman_msec" in (await svc.status())["config_error"]

    hardware.write_text("ibt4:\n  gpio_enabled: false\n  deadman_ms: 500\n", encoding="utf-8")
    monkeypatch.setattr(blade_service, "_config_retry_at", 0.0)
    fixed = blade_service.get_blade_service()
    assert fixed.config_error is None and fixed.deadman_s == 0.5
