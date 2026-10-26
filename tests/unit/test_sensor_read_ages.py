"""How old a sensor's last read looks must not depend on the wall clock (audit SAF-05, US-02).

The drivers' ``health_check`` reports ``last_read_age_s`` and the ToF driver spaces its re-init
attempts. Both were measured on ``time.time()``: an NTP/chrony step backwards made the age negative
(read as "fresh" by anything that clamps) or stalled the re-init for as long as the step.
"""

from __future__ import annotations

import time

import pytest

from backend.src.drivers.sensors import vl53l0x_driver
from backend.src.drivers.sensors.bme280_driver import BME280Driver
from backend.src.drivers.sensors.bno085_driver import BNO085Driver
from backend.src.drivers.sensors.ina3221_driver import INA3221Driver
from backend.src.drivers.sensors.vl53l0x_driver import VL53L0XDriver

pytestmark = pytest.mark.asyncio

CASES = [
    pytest.param(lambda: VL53L0XDriver("left", {}), "read_distance_mm", id="vl53l0x"),
    pytest.param(lambda: BME280Driver({}), "read_environment", id="bme280"),
    pytest.param(lambda: BNO085Driver({}), "read_orientation", id="bno085"),
    pytest.param(lambda: INA3221Driver({}), "read_power", id="ina3221"),
]


@pytest.mark.parametrize(("make", "method"), CASES)
async def test_a_wall_clock_step_does_not_change_how_old_the_last_read_is(
    monkeypatch, make, method
):
    monkeypatch.setenv("SIM_MODE", "1")
    driver = make()
    await driver.initialize()
    assert (await driver.health_check())["last_read_age_s"] is None  # nothing read yet
    await getattr(driver, method)()
    wall = time.time
    monkeypatch.setattr(time, "time", lambda: wall() - 86_400.0)  # NTP steps the clock a day back
    age = (await driver.health_check())["last_read_age_s"]
    assert age is not None and 0.0 <= age < 5.0


@pytest.mark.parametrize(("make", "method"), CASES)
async def test_the_age_of_the_last_read_grows_on_the_monotonic_clock(monkeypatch, make, method):
    monkeypatch.setenv("SIM_MODE", "1")
    driver = make()
    await driver.initialize()
    await getattr(driver, method)()
    mono = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: mono() + 30.0)
    age = (await driver.health_check())["last_read_age_s"]
    assert age is not None and 29.0 <= age < 35.0


async def test_the_tof_reinit_spacing_survives_a_backward_wall_clock_step(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")
    attempts: list[int] = []

    async def fail_to_init(_driver) -> bool:
        attempts.append(1)
        return False

    monkeypatch.setattr(vl53l0x_driver, "_try_init_adafruit", fail_to_init)
    monkeypatch.setattr(vl53l0x_driver, "_try_init_pololu_like", fail_to_init)
    driver = VL53L0XDriver("left", {})
    driver._fail_count = 3  # the third failed read in a row
    await vl53l0x_driver._reinit_if_needed(driver)
    assert len(attempts) == 2  # adafruit, then the pololu-like fallback
    await vl53l0x_driver._reinit_if_needed(driver)
    assert len(attempts) == 2  # inside the 2 s spacing: not again

    wall, mono = time.time, time.monotonic
    monkeypatch.setattr(time, "time", lambda: wall() - 86_400.0)  # the clock is stepped back a day
    monkeypatch.setattr(time, "monotonic", lambda: mono() + 3.0)  # and 3 s really pass
    await vl53l0x_driver._reinit_if_needed(driver)
    assert len(attempts) == 4  # the spacing is over: tried again, not stalled for a day
