"""Tests for I2C presence detection that keeps /health honest.

Absent hardware must report offline, not a fabricated 'online/healthy'.
"""

from __future__ import annotations

import sys
import types

import pytest

from backend.src.services import sensor_manager as sm


def _install_fake_smbus2(monkeypatch, present: set[int]):
    """Inject a fake smbus2 whose read_byte ACKs only `present` addresses."""

    class _FakeSMBus:
        def __init__(self, bus):
            self.bus = bus

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read_byte(self, addr):
            if addr in present:
                return 0
            raise OSError(121, "Remote I/O error")  # what a missing device raises

    mod = types.ModuleType("smbus2")
    mod.SMBus = _FakeSMBus
    monkeypatch.setitem(sys.modules, "smbus2", mod)


def _force_real(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setattr(sm, "is_simulation_mode", lambda: False)


def test_i2c_ack_true_in_sim(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "1")
    # No smbus2 needed — sim short-circuits to present.
    assert sm._i2c_ack(1, [0x29]) is True


def test_i2c_ack_detects_present_device(monkeypatch):
    _force_real(monkeypatch)
    _install_fake_smbus2(monkeypatch, present={0x29})
    assert sm._i2c_ack(1, [0x29]) is True
    assert sm._i2c_ack(1, [0x30]) is False
    assert sm._i2c_ack(1, [0x30, 0x29]) is True  # any address ACKing counts


def test_i2c_ack_false_when_bus_empty(monkeypatch):
    _force_real(monkeypatch)
    _install_fake_smbus2(monkeypatch, present=set())
    assert sm._i2c_ack(1, [0x4A, 0x4B]) is False  # BNO085 absent → offline


def test_i2c_ack_false_without_smbus2(monkeypatch):
    _force_real(monkeypatch)
    monkeypatch.setitem(sys.modules, "smbus2", None)  # import raises
    assert sm._i2c_ack(1, [0x40]) is False


@pytest.mark.asyncio
async def test_imu_interface_offline_when_absent(monkeypatch):
    """IMU is a sim-stub driver; presence must come from the bus, not the stub."""
    _force_real(monkeypatch)
    _install_fake_smbus2(monkeypatch, present=set())  # nothing on the bus
    imu = sm.IMUSensorInterface(sm.SensorCoordinator())
    ok = await imu.initialize()
    assert ok is False
    assert imu.status == sm.SensorStatus.OFFLINE
