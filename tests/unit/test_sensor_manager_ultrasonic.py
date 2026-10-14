"""The HaLow HAT owns GPIO4/9/10/11, so the ultrasonic array must stay off when disabled."""

from __future__ import annotations

from backend.src.core.config_loader import ConfigLoader
from backend.src.models import SensorStatus
from backend.src.services.sensor_manager import SensorManager, UltrasonicSensorInterface


async def test_disabled_ultrasonic_is_never_initialized(monkeypatch):
    async def _boom(self):
        raise AssertionError("ultrasonic driver touched while disabled")

    monkeypatch.setattr(UltrasonicSensorInterface, "initialize", _boom)
    manager = SensorManager(ultrasonic_enabled=False)

    assert await manager.initialize() is True
    assert manager.ultrasonic.status == SensorStatus.OFFLINE


def test_ultrasonic_flag_resolves_from_config_dir(tmp_path, monkeypatch):
    (tmp_path / "limits.yaml").write_text("ultrasonic_enabled: false\n", encoding="utf-8")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))

    assert SensorManager().ultrasonic_enabled is False


def test_repo_limits_disable_ultrasonic():
    assert ConfigLoader(config_dir="config").get()[1].ultrasonic_enabled is False
