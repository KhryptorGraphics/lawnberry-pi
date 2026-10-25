"""``limits.yaml`` is validated strictly and read from where the unit says (audit SAF-08, SAF-09).

``.nan``/``.inf`` and absurd values used to load silently and could switch a protection off
(``tilt_threshold_degrees: .nan`` never fires, a 24 h watchdog is no watchdog); ``ConfigLoader()``
ignored ``LAWNBERRY_CONFIG_DIR`` and a missing ``limits.yaml`` became the built-in defaults, which
are not the shipped values (``ultrasonic_enabled`` defaults to true).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.src.core.config_loader import ConfigFileError, ConfigLoader
from backend.src.models.safety_limits import SafetyLimits

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def load(tmp_path: Path, text: str):
    (tmp_path / "limits.yaml").write_text(text, encoding="utf-8")
    return ConfigLoader(config_dir=str(tmp_path)).load()[1]


BAD_LIMITS = [
    # top level
    ("tilt_threshold_degrees", ".nan"),
    ("tilt_threshold_degrees", ".inf"),
    ("tilt_threshold_degrees", "0"),
    ("tilt_threshold_degrees", "-5"),
    ("tilt_threshold_degrees", "400"),
    ("watchdog_timeout_ms", "86400000"),
    ("watchdog_timeout_ms", "0"),
    ("battery_low_voltage", ".nan"),
    ("battery_critical_voltage", "-1"),
    ("motor_current_max_amps", ".inf"),
    ("motor_current_max_amps", "100000"),
    ("geofence_buffer_meters", "-0.5"),
    ("geofence_buffer_meters", ".nan"),
    ("high_temperature_celsius", "1000000"),
    ("tof_obstacle_distance_meters", "0"),
    ("tof_obstacle_distance_meters", ".nan"),
    ("ultrasonic_obstacle_distance_cm", ".nan"),
    ("ultrasonic_obstacle_distance_cm", "-5"),
    ("ultrasonic_obstacle_distance_cm", "0"),
    ("ultrasonic_enabled", "'no'"),
    ("tilt_treshold_degrees", "30"),  # a typo must not silently run the default
]
BAD_OBSTACLE = [
    ("max_reading_age_s", ".inf"),
    ("max_reading_age_s", "3600"),
    ("clear_dwell_s", ".nan"),
    ("clear_dwell_s", "-1"),
    ("clear_dwell_s", "100000"),
    ("deceleration_mps2", "1000"),
    ("deceleration_mps2", ".inf"),
    ("full_lever_speed_mps", "0.001"),
    ("full_lever_speed_mps", ".nan"),
    ("reaction_latency_s", ".inf"),
    ("actuation_latency_s", ".nan"),
    ("sensor_uncertainty_m", "-0.1"),
    ("margin_m", ".inf"),
    ("slow_zone_m", "-1"),
    ("detection_range_m", "500"),
    ("detection_range_m", ".nan"),
    ("min_motion_speed_mps", ".nan"),
    ("frozen_reading_count", "1"),
    ("validated", "'yes'"),
    ("apply_to_manual", "'maybe'"),
    ("ignored_sources", "[tof_lef]"),
]
BAD_GPS = [
    ("max_accuracy_m", ".inf"),
    ("max_accuracy_m", "1000000"),
    ("max_fix_age_s", ".inf"),
    ("max_fix_age_s", "-1"),
    ("resume_dwell_s", ".nan"),
    ("check_interval_s", "0"),
    ("check_interval_s", ".inf"),
    ("validated", "1.5"),
]


@pytest.mark.parametrize(("field", "value"), BAD_LIMITS)
def test_a_top_level_limit_that_is_not_a_sane_number_is_refused_by_name(tmp_path, field, value):
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        load(tmp_path, f"{field}: {value}\n")
    assert field in str(excinfo.value)


@pytest.mark.parametrize(("field", "value"), BAD_OBSTACLE)
def test_an_obstacle_stop_limit_that_is_not_sane_is_refused_by_name(tmp_path, field, value):
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        load(tmp_path, f"obstacle_stop:\n  {field}: {value}\n")
    assert field in str(excinfo.value)


@pytest.mark.parametrize(("field", "value"), BAD_GPS)
def test_a_gnss_limit_that_is_not_sane_is_refused_by_name(tmp_path, field, value):
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        load(tmp_path, f"gps_degradation:\n  {field}: {value}\n")
    assert field in str(excinfo.value)


def test_the_shipped_limits_and_the_edges_of_the_ranges_are_accepted(tmp_path):
    shipped = ConfigLoader(config_dir=str(REPO_CONFIG)).load()[1]
    assert shipped.obstacle_stop.frozen_reading_count == 50 and shipped.ultrasonic_enabled is False
    edge = load(
        tmp_path,
        "tilt_threshold_degrees: 60\nwatchdog_timeout_ms: 10000\n"
        "obstacle_stop:\n  clear_dwell_s: 0\n"
        "  ignored_sources: [tof_left, 'ultrasonic:inner_left']\n"
        "gps_degradation:\n  max_accuracy_m: 50\n",
    )
    assert edge.tilt_threshold_degrees == 60 and edge.obstacle_stop.clear_dwell_s == 0
    assert edge.obstacle_stop.ignored_sources == ["tof_left", "ultrasonic:inner_left"]
    assert SafetyLimits().tilt_threshold_degrees == 30.0


# ------------------------------------------------------------------------------ SAF-09


def test_a_bare_loader_reads_the_directory_the_unit_names(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    conf = tmp_path / "root" / "config"
    elsewhere.mkdir()
    conf.mkdir(parents=True)
    (conf / "limits.yaml").write_text("tilt_threshold_degrees: 12\nultrasonic_enabled: false\n")
    monkeypatch.chdir(elsewhere)  # not the config the working directory happens to hold
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(conf))
    loader = ConfigLoader()
    assert Path(loader.limits_path) == conf / "limits.yaml"
    assert loader.load()[1].tilt_threshold_degrees == 12
    monkeypatch.delenv("LAWNBERRY_CONFIG_DIR")
    assert ConfigLoader().config_dir == str(elsewhere / "config")


def test_a_missing_or_empty_limits_file_is_an_error_on_the_mower(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    with pytest.raises(ConfigFileError, match="does not exist"):
        ConfigLoader(config_dir=str(tmp_path)).load()
    (tmp_path / "limits.yaml").write_text("# nothing here\n")
    with pytest.raises(ConfigFileError, match="empty"):
        ConfigLoader(config_dir=str(tmp_path)).load()


@pytest.mark.parametrize("sim", ["0", "1"])
def test_a_limits_file_that_cannot_be_read_is_an_error_in_every_mode(tmp_path, monkeypatch, sim):
    monkeypatch.setenv("SIM_MODE", sim)
    (tmp_path / "limits.yaml").mkdir()  # exists, cannot be read as a file
    with pytest.raises(ConfigFileError, match="cannot read"):
        ConfigLoader(config_dir=str(tmp_path)).load()
    (tmp_path / "limits.yaml").rmdir()
    (tmp_path / "limits.yaml").write_text("estop_latency_ms: [unclosed\n")
    with pytest.raises(ConfigFileError, match="cannot read"):
        ConfigLoader(config_dir=str(tmp_path)).load()


def test_in_simulation_a_missing_limits_file_is_defaults_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("SIM_MODE", "1")
    with caplog.at_level(logging.WARNING, logger="backend.src.core.config_loader"):
        limits = ConfigLoader(config_dir=str(tmp_path)).load()[1]
    assert limits.estop_latency_ms == 100
    assert any("built-in default limits" in r.getMessage() for r in caplog.records)
    assert yaml.safe_load((REPO_CONFIG / "limits.yaml").read_text())["ultrasonic_enabled"] is False
    assert limits.ultrasonic_enabled is True  # why defaults must not stand in on the mower
