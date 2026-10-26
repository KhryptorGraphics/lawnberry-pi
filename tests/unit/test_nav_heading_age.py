"""The heading has an age, and the navigation feed's stamp moves only with new data (audit SAF-14).

The IMU yaw used to be copied into the navigation state with no age, and ``timestamp`` was
refreshed on every call whatever (or nothing) the call carried. A dead IMU, whose last reading the
sensor interface replays for ever, therefore looked fresh to the proportional steering and to any
feed-age check. Now a reading counts only when its own timestamp changed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from backend.src.models.sensor_data import ImuReading, SensorData
from backend.src.services.navigation_service import NavigationService


def _data(imu_at: datetime, yaw: float = 90.0) -> SensorData:
    return SensorData(imu=ImuReading(yaw=yaw, timestamp=imu_at))


async def test_a_heading_the_imu_stopped_refreshing_ages_and_holds_the_machine():
    nav = NavigationService()
    stamp = datetime.now(UTC) - timedelta(seconds=10)
    await nav.update_navigation_state(_data(stamp))
    assert nav.navigation_state.heading == 90.0
    age = nav.heading_age_s()
    assert age is not None and 9.0 < age < 12.0
    gate = nav.motion_gate()
    assert gate.hold and any("heading" in reason for reason in gate.reasons)

    # The IMU died: the sensor interface replays the SAME reading (same stamp) for ever.
    await asyncio.sleep(0.05)
    await nav.update_navigation_state(_data(stamp))
    assert nav.heading_age_s() > age, "a replayed reading must keep ageing"
    assert nav.motion_gate().hold


async def test_a_fresh_reading_releases_the_hold():
    nav = NavigationService()
    await nav.update_navigation_state(_data(datetime.now(UTC) - timedelta(seconds=10)))
    assert nav.motion_gate().hold
    await nav.update_navigation_state(_data(datetime.now(UTC)))
    assert nav.heading_age_s() < 1.0
    assert not nav.motion_gate().hold


async def test_a_clock_that_stepped_back_does_not_make_a_frozen_heading_look_fresh():
    nav = NavigationService()
    await nav.update_navigation_state(_data(datetime.now(UTC) + timedelta(seconds=60)))
    age = nav.heading_age_s()
    assert age is not None and age > nav.max_heading_age_s
    assert nav.motion_gate().hold


async def test_the_feed_stamp_moves_only_when_something_new_arrived():
    nav = NavigationService()
    stamp = datetime.now(UTC)
    await nav.update_navigation_state(_data(stamp))
    first = nav.navigation_state.timestamp
    await asyncio.sleep(0.02)
    await nav.update_navigation_state(_data(stamp))  # a replay carries nothing new
    assert nav.navigation_state.timestamp == first, "a replay must not refresh the feed stamp"
    await nav.update_navigation_state(_data(stamp + timedelta(seconds=1), yaw=91.0))
    assert nav.navigation_state.timestamp > first
