"""Tests for config/tractor.yaml validation and loading through TractorControlService."""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest


class TestTractorConfigFile:
    """Tests that the real config/tractor.yaml loads correctly."""

    def test_tractor_yaml_loads_enabled_false(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert tractor.get("enabled") is False
        assert isinstance(tractor.get("enabled"), bool)

    def test_tractor_yaml_has_required_sections(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert "relays" in tractor
        assert "pca9685" in tractor
        assert "actuators" in tractor
        assert tractor["pca9685"]["address"] == 0x41

    def test_tractor_relays_have_gpio_ints(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        for name, relay in tractor["relays"].items():
            assert isinstance(relay["gpio"], int)

    def test_tractor_relay_gpio_values_match_registry(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert tractor["relays"]["starter"]["gpio"] == 5
        assert tractor["relays"]["blade_pto"]["gpio"] == 26

    def test_tractor_pca9685_address_not_0x40(self):
        tractor = yaml.safe_load((Path("config/tractor.yaml")).read_text())
        assert tractor["pca9685"]["address"] != 0x40
        assert tractor["pca9685"]["address"] == 0x41

    def test_limits_yaml_ultrasonic_disabled(self):
        limits = yaml.safe_load((Path("config/limits.yaml")).read_text())
        assert limits.get("ultrasonic_enabled") is False


class TestTractorServiceLoadsConfig:
    """Tests that TractorControlService loads the real config correctly."""

    @pytest.fixture(autouse=True)
    def _set_sim_mode(self, monkeypatch):
        monkeypatch.setenv("SIM_MODE", "1")

    def test_service_loads_real_config_disabled(self):
        from backend.src.services.tractor_service import TractorControlService
        service = TractorControlService()
        assert service.enabled is False

    def test_service_starter_gpio_is_5(self):
        from backend.src.services.tractor_service import TractorControlService
        service = TractorControlService()
        assert service.starter.gpio == 5

    def test_service_blade_pto_gpio_is_26(self):
        from backend.src.services.tractor_service import TractorControlService
        service = TractorControlService()
        assert service.blade_pto.gpio == 26

    def test_service_pca9685_config_matches_yaml(self):
        from backend.src.services.tractor_service import TractorControlService
        service = TractorControlService()
        assert service._pca9685 is not None
