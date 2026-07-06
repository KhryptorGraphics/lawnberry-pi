"""Unit tests for the gpiochip probe helper (Pi 5 chip 4 vs Pi 4 chip 0)."""

from __future__ import annotations

import pytest

from backend.src.drivers.base import open_gpiochip


class _FakeLgpio:
    """Fake lgpio: only the listed chips open; others raise like the real lib."""

    def __init__(self, openable: set[int]):
        self.openable = openable
        self.attempts: list[int] = []

    def gpiochip_open(self, chip: int) -> int:
        self.attempts.append(chip)
        if chip in self.openable:
            return 1000 + chip  # handle
        raise Exception("can not open gpiochip")


def test_probes_chip4_first_on_pi5(monkeypatch):
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    fake = _FakeLgpio(openable={4})  # Pi 5: only chip 4 accessible
    assert open_gpiochip(fake) == 1004
    assert fake.attempts == [4]  # chip 4 tried first, no need to fall back


def test_falls_back_to_chip0_on_pi4(monkeypatch):
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    fake = _FakeLgpio(openable={0})  # Pi 4: chip 4 absent, chip 0 works
    assert open_gpiochip(fake) == 1000
    assert fake.attempts == [4, 0]  # tried 4, failed, then 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("LAWNBERRY_GPIOCHIP", "2")
    fake = _FakeLgpio(openable={0, 2, 4})
    assert open_gpiochip(fake) == 1002
    assert fake.attempts == [2]  # only the override is tried


def test_raises_when_none_open(monkeypatch):
    monkeypatch.delenv("LAWNBERRY_GPIOCHIP", raising=False)
    fake = _FakeLgpio(openable=set())
    with pytest.raises(Exception, match="can not open gpiochip"):
        open_gpiochip(fake)
    assert fake.attempts == [4, 0]
