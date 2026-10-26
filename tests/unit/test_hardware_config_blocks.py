"""``ibt4:`` and ``ultrasonic:`` blocks of config/hardware.yaml: defaults, validation, loading.

Both blocks arm real hardware paths (GPIO outputs, a USB sensor array), so the properties pinned
here are the arming gates: everything ships OFF / fail-closed, a typo cannot quietly relax a
policy, and an inconsistent block is refused at load time instead of misbehaving in the field.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.src.core.config_loader import ConfigLoader
from backend.src.models.hardware_config import Ibt4Config, UltrasonicConfig

REPO_CONFIG = Path("config")


def load(tmp_path: Path, hardware: dict, local: dict | None = None):
    (tmp_path / "hardware.yaml").write_text(yaml.safe_dump(hardware), encoding="utf-8")
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    if local is not None:
        (tmp_path / "hardware.local.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")
    return ConfigLoader(config_dir=str(tmp_path)).load()[0]


# ---- IBT-4 ------------------------------------------------------------------------------------


def test_ibt4_ships_disarmed_with_the_documented_pins():
    cfg = Ibt4Config()
    assert cfg.gpio_enabled is False
    assert (cfg.pins.in1, cfg.pins.in2) == (24, 25)
    assert (cfg.on_levels.in1, cfg.on_levels.in2) == (1, 0)


@pytest.mark.parametrize(
    "bad",
    [
        {"pins": {"in1": 24, "in2": 24}},  # two inputs on one GPIO
        {"pins": {"in1": 28, "in2": 25}},  # not a header GPIO
        {"pins": {"in1": -1, "in2": 25}},
        {"on_levels": {"in1": 2, "in2": 0}},  # a level is 0 or 1
    ],
)
def test_ibt4_rejects_impossible_pin_and_level_configuration(bad):
    with pytest.raises(ValidationError):
        Ibt4Config(**bad)


def test_the_repo_config_does_not_arm_the_blade_gpio():
    hardware = ConfigLoader(config_dir=str(REPO_CONFIG)).load()[0]
    assert hardware.ibt4_config is not None
    assert hardware.ibt4_config.gpio_enabled is False
    assert hardware.ibt4_config.pins.in2 == 25  # unchanged: moving it is owner action O-5


def test_a_local_override_can_arm_only_the_flag_and_keeps_the_pins(tmp_path):
    base = {"ibt4": {"gpio_enabled": False, "pins": {"in1": 24, "in2": 25}}}
    hardware = load(tmp_path, base, local={"ibt4": {"gpio_enabled": True}})
    assert hardware.ibt4_config.gpio_enabled is True
    assert (hardware.ibt4_config.pins.in1, hardware.ibt4_config.pins.in2) == (24, 25)


# ---- ultrasonic -------------------------------------------------------------------------------


def test_ultrasonic_defaults_are_fail_closed_and_assign_no_pi_pin():
    cfg = UltrasonicConfig()
    assert cfg.backend == "mcu_usb" and cfg.port is None
    assert cfg.out_of_range_is_clear is False and cfg.no_echo_is_clear is False
    assert cfg.no_echo_requires_alive is True
    assert cfg.channels == ["outer_left", "inner_left", "inner_right", "outer_right"]
    assert cfg.min_range_cm < cfg.max_range_cm
    # every pin-like field is absent: the MCU link is USB serial
    assert not [f for f in UltrasonicConfig.model_fields if "gpio" in f or "pin" in f]


@pytest.mark.parametrize(
    "bad",
    [
        {"channels": []},
        {"channels": ["a", "a"]},
        {"channels": [f"c{i}" for i in range(9)]},
        {"channels": ["bad name!"]},
        {"disabled_channels": ["not_a_channel"]},
        {"min_range_cm": 400.0, "max_range_cm": 400.0},
        {"min_range_cm": 500.0},
        {"echo_window_ms": 10},  # 10 ms cannot hold the echo of the default 400 cm range
        {"speed_of_sound_mps": 3430.0},
        {"backend": "i2c"},
        {"stale_after_ms": 1},
    ],
)
def test_inconsistent_ultrasonic_configuration_is_refused(bad):
    with pytest.raises(ValidationError):
        UltrasonicConfig(**bad)


def test_echo_window_must_cover_the_configured_range_exactly():
    # 2 * 4.0 m / 343.2 m/s = 23.3 ms: a 24 ms window is enough, 23 is not
    UltrasonicConfig(echo_window_ms=24, inter_sensor_gap_ms=30)
    with pytest.raises(ValidationError):
        UltrasonicConfig(echo_window_ms=23)
    with pytest.raises(ValidationError):
        UltrasonicConfig(echo_window_ms=30, max_range_cm=600.0)  # 600 cm needs 35 ms


def test_the_repo_config_is_the_conservative_mcu_configuration():
    hardware = ConfigLoader(config_dir=str(REPO_CONFIG)).load()[0]
    cfg = hardware.ultrasonic_config
    assert cfg is not None and cfg.backend == "mcu_usb" and cfg.port is None
    assert (cfg.out_of_range_is_clear, cfg.no_echo_is_clear, cfg.no_echo_requires_alive) == (
        False, False, True,
    )  # fmt: skip
    limits = ConfigLoader(config_dir=str(REPO_CONFIG)).load()[1]
    assert limits.ultrasonic_enabled is False  # the master switch is unchanged


def test_hardware_yaml_blocks_reach_the_typed_config(tmp_path):
    hardware = load(
        tmp_path,
        {
            "ultrasonic": {
                "port": "/dev/serial/by-id/usb-LawnBerry-if02",
                "disabled_channels": ["inner_left"],
            },
            "ibt4": {"gpio_enabled": True, "on_levels": {"in1": 0, "in2": 1}},
        },
    )
    assert hardware.ultrasonic_config.port == "/dev/serial/by-id/usb-LawnBerry-if02"
    assert hardware.ultrasonic_config.disabled_channels == ["inner_left"]
    assert hardware.ibt4_config.gpio_enabled is True
    assert (hardware.ibt4_config.on_levels.in1, hardware.ibt4_config.on_levels.in2) == (0, 1)


def test_a_broken_block_fails_the_load_instead_of_being_half_applied(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, {"ultrasonic": {"min_range_cm": 900.0}})
    with pytest.raises(ValueError):
        load(tmp_path, {"ibt4": {"pins": {"in1": 24, "in2": 24}}})


def test_absent_blocks_leave_the_typed_config_unset_so_runtime_defaults_apply(tmp_path):
    hardware = load(tmp_path, {"motor_controller": "robohat-rp2040"})
    assert hardware.ultrasonic_config is None and hardware.ibt4_config is None


# ---- US-07: no limit may switch a protection off ----------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"min_range_cm": 0.001},  # a blind zone of 0 is no blind zone
        {"min_range_cm": 9.9},
        {"min_range_cm": float("nan")},
        {"min_range_cm": float("inf")},
        {"max_range_cm": float("nan")},
        {"max_range_cm": float("inf")},
        {"speed_of_sound_mps": float("nan")},
        {"stale_after_ms": 10**9},  # staleness that never fires
        {"stale_after_ms": 5001},
        {"link_timeout_ms": 10**9},  # link supervision that never fires
        {"link_timeout_ms": 10001},
        {"alive_window_s": 1e9},  # a sensor that is "alive" for ever
        {"alive_window_s": 60.1},
        {"alive_window_s": float("inf")},
        {"alive_window_s": float("nan")},
        {"handshake_timeout_s": 1e9},
        {"handshake_timeout_s": float("inf")},
        {"near_hold_s": 4.9},
        {"near_hold_s": 601.0},
        {"near_hold_s": float("nan")},
        {"near_hold_s": float("inf")},
        {"echo_window_ms": 100, "inter_sensor_gap_ms": 500},  # 2400 ms cycle vs stale 700 ms
    ],
)
def test_ultrasonic_limits_that_would_disable_a_protection_are_refused(bad):
    with pytest.raises(ValidationError):
        UltrasonicConfig(**bad)


def test_stale_after_must_exceed_one_full_cycle_by_a_margin():
    # 4 channels * (30 + 30) ms = 240 ms per cycle; the margin is 100 ms
    UltrasonicConfig(stale_after_ms=340)
    with pytest.raises(ValidationError, match="measurement cycle"):
        UltrasonicConfig(stale_after_ms=339)
    slow = {"echo_window_ms": 100, "inter_sensor_gap_ms": 500}  # legal, but a 2400 ms cycle
    UltrasonicConfig(stale_after_ms=2500, **slow)
    with pytest.raises(ValidationError, match="measurement cycle"):
        UltrasonicConfig(stale_after_ms=2499, **slow)


def test_the_edges_of_the_new_bounds_are_accepted():
    UltrasonicConfig(
        min_range_cm=10.0, stale_after_ms=5000, link_timeout_ms=10000, alive_window_s=60.0,
        handshake_timeout_s=30.0, near_hold_s=5.0,
    )  # fmt: skip
    UltrasonicConfig(near_hold_s=600.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0, 0.0, 9.99])
def test_the_ultrasonic_stop_distance_must_be_finite_and_sensible(bad):
    from backend.src.models.safety_limits import SafetyLimits

    with pytest.raises(ValidationError):
        SafetyLimits(ultrasonic_obstacle_distance_cm=bad)


@pytest.mark.parametrize("good", [10.0, 30.0, 45.0])
def test_sensible_ultrasonic_stop_distances_are_accepted(good):
    from backend.src.models.safety_limits import SafetyLimits

    limits = SafetyLimits(ultrasonic_obstacle_distance_cm=good)
    assert limits.ultrasonic_obstacle_distance_cm == good


def test_both_shipped_config_files_still_load_and_keep_their_policy():
    hardware, limits = ConfigLoader(config_dir=str(REPO_CONFIG)).load()
    cfg = hardware.ultrasonic_config
    assert cfg is not None and UltrasonicConfig(**cfg.model_dump()) == cfg  # valid under the bounds
    assert (cfg.stale_after_ms, cfg.link_timeout_ms, cfg.handshake_timeout_s) == (700, 1500, 2.0)
    assert (cfg.alive_window_s, cfg.min_range_cm, cfg.near_hold_s) == (10.0, 25.0, 30.0)
    assert (cfg.out_of_range_is_clear, cfg.no_echo_is_clear) == (False, False)
    assert cfg.no_echo_requires_alive is True
    assert limits.ultrasonic_obstacle_distance_cm == 30.0 and limits.ultrasonic_enabled is False
