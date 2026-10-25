"""A position that is not a finite coordinate is "no position" to the legs (review 3, NAV-06).

The GNSS monitor's source fix (a NaN latitude is not a healthy fix) is the GNSS track's. The leg
controller's half is here: ``min(1.0, nan)`` is ``1.0``, so a NaN position turned the proportional
controller's heading error into a full-lever pivot with the blade on (the leg checked that the
heading was finite, never the position), and ``_update_position`` took a NaN reading for the
position because NaN is truthy: it became ``current_position`` and the dead-reckoning reference.

The tractor is the real service over the stallable PCA9685; the navigation service and the mission
loop are real.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from _mission_rig import FAR, HERE, new_nav, wire_mission
from _nav_rig import wait_until
from _tractor_fakes import LEFT, NEUTRAL, RIGHT, mowing_tractor

from backend.src.models import Position
from backend.src.models.sensor_data import GpsReading, SensorData
from backend.src.services.navigation_service import LegResult

BAD_COORDINATES = [
    pytest.param(math.nan, -83.0, id="nan-lat"),
    pytest.param(40.0, math.nan, id="nan-lon"),
    pytest.param(math.inf, -83.0, id="inf-lat"),
    pytest.param(40.0, -math.inf, id="-inf-lon"),
    pytest.param(91.0, -83.0, id="lat-off-the-globe"),
    pytest.param(40.0, 181.0, id="lon-off-the-globe"),
]


def _non_neutral(chip) -> list[tuple[int, int]]:
    return [(ch, us) for _t, ch, us in chip.log if us != NEUTRAL[ch]]


@pytest.mark.parametrize(("lat", "lon"), BAD_COORDINATES)
async def test_a_mission_leg_holds_on_a_position_that_is_not_a_coordinate(monkeypatch, lat, lon):
    tractor, chip = mowing_tractor()
    nav = new_nav()
    nav.control_period_s = 0.05
    mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    nav.navigation_state.current_position = Position(latitude=lat, longitude=lon)
    task = asyncio.create_task(nav.execute_mission(mission))
    try:
        await wait_until(lambda: nav._hold_reasons is not None, timeout=5.0)
        await asyncio.sleep(0.4)  # eight control ticks

        assert nav._hold_reasons == ("no usable position",)
        assert _non_neutral(chip) == [], "a pivot at full lever was commanded from a NaN position"
        assert chip.regs.get(LEFT, NEUTRAL[LEFT]) == NEUTRAL[LEFT]
        assert chip.regs.get(RIGHT, NEUTRAL[RIGHT]) == NEUTRAL[RIGHT]
        assert tractor.state.blade_engaged is False

        # a real fix is steered by again: the hold is level-triggered, not a latch
        nav.navigation_state.current_position = HERE
        await wait_until(lambda: tractor.state.moving, timeout=5.0)
        assert nav._hold_reasons is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tractor.shutdown()


async def test_a_plan_leg_holds_on_such_a_position_too(monkeypatch):
    """The Thor follower's gated legs have no GNSS monitor in front of them."""
    tractor, chip = mowing_tractor()
    nav = new_nav()
    nav.control_period_s = 0.02
    _mission, _service = wire_mission(monkeypatch, tractor, nav, [FAR])
    nav.navigation_state.current_position = Position(latitude=math.nan, longitude=math.nan)
    ticks = 0

    def gate() -> bool:
        nonlocal ticks
        ticks += 1
        return ticks <= 6

    leg = await asyncio.wait_for(nav.go_to_waypoint(FAR, should_continue=gate), timeout=5.0)

    assert leg is LegResult.HALTED and ticks > 6
    assert _non_neutral(chip) == [] and tractor.state.blade_engaged is False
    await tractor.shutdown()


@pytest.mark.parametrize(("lat", "lon"), BAD_COORDINATES)
async def test_a_reading_that_is_not_a_coordinate_never_becomes_the_position(lat, lon):
    nav = new_nav()
    nav.navigation_state.heading = 90.0
    before_fix = nav.navigation_state.last_gps_fix

    await nav.update_navigation_state(SensorData(gps=GpsReading(latitude=lat, longitude=lon)))

    position = nav.navigation_state.current_position
    assert position is not None
    assert math.isfinite(position.latitude) and math.isfinite(position.longitude)
    assert abs(position.latitude) <= 90.0 and abs(position.longitude) <= 180.0
    reference = nav.dead_reckoning.last_gps_position
    assert reference is None or (
        math.isfinite(reference.latitude) and math.isfinite(reference.longitude)
    )
    assert nav.navigation_state.last_gps_fix == before_fix, "a bad reading is not a fix"


async def test_a_good_reading_still_becomes_the_position():
    nav = new_nav()

    await nav.update_navigation_state(
        SensorData(gps=GpsReading(latitude=40.5, longitude=-83.5, accuracy=0.02))
    )

    position = nav.navigation_state.current_position
    assert (position.latitude, position.longitude) == (40.5, -83.5)
    assert nav.navigation_state.last_gps_fix is not None
