"""Tests for config/tractor.yaml validation and loading through TractorControlService."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


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
        for _name, relay in tractor["relays"].items():
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

    @pytest.mark.asyncio
    async def test_service_pca9685_config_matches_yaml(self):
        """The shipped address/bus/frequency must reach the driver (it used to assert only that a
        driver object existed, which a driver built from ``{}`` also satisfies)."""
        from backend.src.services.tractor_service import TractorControlService

        cfg = yaml.safe_load(Path("config/tractor.yaml").read_text())["pca9685"]
        health = await TractorControlService()._pca9685.health_check()

        assert health["address"] == hex(cfg["address"]) == "0x41"
        assert health["bus"] == cfg["bus"]
        assert health["frequency_hz"] == float(cfg["frequency_hz"])

    def test_service_deadman_config_matches_yaml(self):
        from backend.src.services.tractor_service import TractorControlService

        deadman = yaml.safe_load(Path("config/tractor.yaml").read_text())["command_deadman"]
        service = TractorControlService()

        assert service.default_hold_s == deadman["default_hold_ms"] / 1000.0 == 0.5
        assert service.max_hold_s == deadman["max_hold_ms"] / 1000.0 == 2.0


def test_ultrasonic_disabled_while_spec_shows_halow_on_spi0():
    """The ultrasonic driver pins (GPIO9/10/11) are HaLow SPI0; the flag must stay off."""
    limits = yaml.safe_load(Path("config/limits.yaml").read_text())
    spec = yaml.safe_load(Path("spec/hardware.yaml").read_text())
    halow_spi = {
        e["gpio"]
        for e in spec["pins"]["gpio_used"]
        if e.get("role", "").startswith("HaLow HAT SPI0") and e.get("status") == "verified"
    }
    assert {8, 9, 10, 11} <= halow_spi
    assert limits["ultrasonic_enabled"] is False
