"""The ultrasonic sensor interface: four JSN-SR04T behind the MCU, the legacy GPIO backend, config.

The entry shape that ``read_ultrasonic()`` returns is a contract with the obstacle-stop guard in
the navigation stack, so the keys and their meaning are pinned here. The safety rule pinned
throughout: an entry that is not a fresh in-range echo is ``valid`` False and never "clear" unless
the owner's policy says so; a missing channel is visible, not absent.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.src.drivers.sensors.jsn_sr04t_driver import sector_distances
from backend.src.models import SensorStatus
from backend.src.models.mower_data_frame import MowerDataFrame, UltrasonicData
from backend.src.services.perimeter_recorder import PerimeterRecordingService
from backend.src.services.sensor_manager import (
    SensorCoordinator,
    SensorManager,
    UltrasonicSensorInterface,
)
from tests.unit._usonic_fakes import FakeClock, FakeMcuFactory, SensorModel

CONTRACT_KEYS = {
    "sensor_id", "index", "distance_cm", "valid", "timestamp", "status", "age_ms", "health",
    "alive", "verdict", "clear_to_cm", "echo_us", "fails",
}  # fmt: skip
NAMES = ["outer_left", "inner_left", "inner_right", "outer_right"]


def make(clock, factory, **cfg):
    """An interface wired to fake time and the fake MCU (the seam is ``driver_kwargs``)."""
    return UltrasonicSensorInterface(
        SensorCoordinator(),
        config={"port": "/dev/ttyACM-usonic", **cfg},
        obstacle_threshold_cm=30.0,
        driver_kwargs=dict(
            serial_factory=factory, clock=clock.now, wall_clock=clock.wall,
            sleep=clock.sleep, simulate=False,
        ),
    )  # fmt: skip


def run(clock, iface, seconds):
    for _ in range(round(seconds / 0.02)):
        clock.advance(0.02)
        iface._driver.poll()


async def active(sensors=None, **cfg):
    clock = FakeClock()
    factory = FakeMcuFactory(
        clock, sensors=sensors or [SensorModel(distance_cm=d) for d in (80, 120, 200, 60)]
    )
    iface = make(clock, factory, **cfg)
    assert await iface.initialize() is True
    run(clock, iface, 0.6)
    return clock, factory, iface


# ---- the contract -----------------------------------------------------------------------------


async def test_one_entry_per_channel_with_the_stable_keys_and_meanings():
    clock, _factory, iface = await active()
    entries = await iface.read_ultrasonic()
    assert [e["sensor_id"] for e in entries] == NAMES == iface.channel_ids
    for e in entries:
        assert set(e) == CONTRACT_KEYS
        assert e["valid"] is True and e["status"] == "ok" and e["verdict"] == "clear"
        assert isinstance(e["distance_cm"], float) and e["distance_cm"] > 0
        assert e["clear_to_cm"] == e["distance_cm"]
        # the timestamp is the MEASUREMENT (host receive time of that frame), ISO-8601 UTC
        ts = datetime.fromisoformat(e["timestamp"])
        assert ts.utcoffset().total_seconds() == 0
        assert 0 <= (clock.wall() - ts).total_seconds() < 0.5
        assert e["age_ms"] == pytest.approx((clock.wall() - ts).total_seconds() * 1000, abs=1.5)
    assert iface.last_readings == entries and iface.status == SensorStatus.ONLINE
    assert iface.get_minimum_distance() == pytest.approx(60.0, abs=0.1)


async def test_valid_is_true_only_for_a_fresh_in_range_echo_and_unusable_reads_have_no_distance():
    models = [
        SensorModel(distance_cm=100.0),
        SensorModel(distance_cm=12.0),  # blind zone
        SensorModel(mode="silent"),  # open air or dead
        SensorModel(mode="stuck"),  # fault
    ]
    _clock, _f, iface = await active(sensors=models)
    entries = {e["sensor_id"]: e for e in await iface.read_ultrasonic()}
    assert entries["outer_left"]["valid"] is True
    for name, status in (
        ("inner_left", "blind_zone"), ("inner_right", "no_echo"), ("outer_right", "fault"),
    ):  # fmt: skip
        e = entries[name]
        assert (e["status"], e["valid"], e["distance_cm"]) == (status, False, 0.0)
        assert e["verdict"] == "unknown" and e["clear_to_cm"] is None


async def test_stale_data_flips_valid_off_at_read_time_even_though_the_timestamp_is_old():
    clock, factory, iface = await active()
    factory.current.freeze()
    run(clock, iface, 0.9)
    entries = await iface.read_ultrasonic()
    assert all(e["status"] == "stale" and e["valid"] is False for e in entries)
    assert all(e["timestamp"] is not None and e["age_ms"] > 700 for e in entries)
    assert all(e["verdict"] == "unknown" for e in entries)


async def test_a_channel_that_never_reported_is_listed_as_no_data_not_omitted():
    clock = FakeClock()
    factory = FakeMcuFactory(clock, sensors=[SensorModel(distance_cm=100.0)] * 4)
    iface = make(clock, factory)
    assert await iface.initialize() is True
    # the very first read, before any measurement could have arrived on this fresh link
    entries = await iface.read_ultrasonic()
    assert [e["sensor_id"] for e in entries] == NAMES
    assert {e["status"] for e in entries} <= {"no_data", "ok"}
    for e in entries:
        if e["status"] == "no_data":
            assert e["timestamp"] is None and e["valid"] is False and e["verdict"] == "unknown"


async def test_when_the_mcu_is_absent_every_channel_is_listed_as_link_down():
    clock = FakeClock()

    def no_such_device(path, baud):
        raise OSError(f"could not open port {path}")

    iface = make(clock, no_such_device)
    assert await iface.initialize() is False
    assert iface.status == SensorStatus.OFFLINE
    entries = await iface.read_ultrasonic()
    assert [e["sensor_id"] for e in entries] == NAMES
    assert all(e["status"] == "link_down" and e["valid"] is False for e in entries)
    assert iface.obstacle_verdict()["state"] == "unknown"
    assert iface.get_minimum_distance() is None


async def test_status_reports_online_offline_and_error_honestly():
    clock, factory, iface = await active()
    await iface.read_ultrasonic()
    assert iface.status == SensorStatus.ONLINE
    factory.current.unplug()
    run(clock, iface, 0.1)
    await iface.read_ultrasonic()
    assert iface.status == SensorStatus.OFFLINE
    # a different protocol version is not "offline", it is an error
    clock2 = FakeClock()
    factory2 = FakeMcuFactory(clock2, n=3, sensors=[SensorModel()] * 3)
    iface2 = make(clock2, factory2)
    assert await iface2.initialize() is False
    run(clock2, iface2, 0.5)
    await iface2.read_ultrasonic()
    assert iface2.status == SensorStatus.ERROR


async def test_interface_that_was_never_initialised_returns_nothing_and_never_opens_a_port():
    opened = []
    iface = UltrasonicSensorInterface(
        SensorCoordinator(),
        config={"port": "/dev/ttyACM-usonic"},
        driver_kwargs=dict(serial_factory=lambda p, b: opened.append(p), simulate=False),
    )
    assert iface.enabled is False
    assert await iface.read_ultrasonic() == [] and opened == []
    assert iface.obstacle_verdict()["state"] == "unknown"
    await iface.shutdown()  # must be a no-op, not an error
    assert opened == []


async def test_obstacle_verdict_uses_the_configured_threshold_and_can_be_overridden():
    _clock, _f, iface = await active(
        sensors=[SensorModel(distance_cm=d) for d in (80.0, 45.0, 200.0, 60.0)]
    )
    await iface.read_ultrasonic()
    assert iface.obstacle_verdict()["state"] == "clear"
    assert iface.obstacle_verdict(threshold_cm=50.0)["state"] == "obstacle"
    assert iface.obstacle_verdict(threshold_cm=50.0)["min_cm"] == pytest.approx(45.0, abs=0.1)


async def test_shutdown_stops_the_service_loop_and_releases_the_port():
    clock, factory, iface = await active()
    port = factory.current
    await iface.shutdown()
    assert port.closed and iface.status == SensorStatus.OFFLINE


# ---- the legacy HC-SR04 GPIO backend keeps working and speaks the same contract ---------------


async def test_legacy_gpio_backend_entries_follow_the_same_contract_in_simulation():
    iface = UltrasonicSensorInterface(
        SensorCoordinator(), config={"backend": "gpio_hcsr04"}, obstacle_threshold_cm=30.0
    )
    assert iface.channel_ids == ["front_left", "front_center", "front_right"]
    assert await iface.read_ultrasonic() == []  # not initialised: the manager skipped it
    assert await iface.initialize() is True
    entries = await iface.read_ultrasonic()
    assert [e["sensor_id"] for e in entries] == iface.channel_ids
    for e in entries:
        assert set(e) == CONTRACT_KEYS
        assert e["verdict"] in ("clear", "obstacle", "unknown")
        assert (e["verdict"] == "unknown") == (not e["valid"])
        if e["valid"]:
            assert e["status"] == "ok" and datetime.fromisoformat(e["timestamp"])
    await iface.shutdown()


async def test_legacy_backend_offline_still_lists_every_channel():
    iface = UltrasonicSensorInterface(SensorCoordinator(), config={"backend": "gpio_hcsr04"})
    await iface.initialize()
    iface.status = SensorStatus.OFFLINE
    entries = await iface.read_ultrasonic()
    assert [e["sensor_id"] for e in entries] == iface.channel_ids
    assert all(e["status"] == "link_down" and not e["valid"] for e in entries)
    assert iface.obstacle_verdict()["state"] == "unknown"


# ---- SensorManager wiring ---------------------------------------------------------------------


async def test_manager_with_the_array_enabled_in_simulation_exposes_four_simulated_channels():
    manager = SensorManager(
        ultrasonic_enabled=True,
        ultrasonic_config={"port": "/dev/ttyACM-usonic"},
        ultrasonic_obstacle_cm=30.0,
    )
    assert await manager.initialize() is True
    assert manager.ultrasonic.status == SensorStatus.ONLINE
    entries = await manager.ultrasonic.read_ultrasonic()
    assert [e["sensor_id"] for e in entries] == NAMES
    await manager.shutdown()
    assert manager.ultrasonic.status == SensorStatus.OFFLINE


async def test_manager_with_the_array_disabled_never_enables_or_reads_it():
    manager = SensorManager(ultrasonic_enabled=False)
    await manager.initialize()
    assert manager.ultrasonic.enabled is False and manager.ultrasonic.status == SensorStatus.OFFLINE
    assert await manager.ultrasonic.read_ultrasonic() == []
    await manager.shutdown()


def test_manager_takes_backend_and_stop_threshold_from_the_config_files(tmp_path, monkeypatch):
    (tmp_path / "limits.yaml").write_text(
        "ultrasonic_enabled: true\nultrasonic_obstacle_distance_cm: 45.0\n", encoding="utf-8"
    )
    (tmp_path / "hardware.yaml").write_text(
        "ultrasonic:\n  backend: gpio_hcsr04\n", encoding="utf-8"
    )
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    manager = SensorManager()
    assert manager.ultrasonic_enabled is True
    assert manager.ultrasonic.backend == "gpio_hcsr04"
    assert manager.ultrasonic.obstacle_threshold_cm == 45.0


def test_shipped_config_selects_the_mcu_backend_and_keeps_the_array_off(monkeypatch):
    monkeypatch.delenv("LAWNBERRY_CONFIG_DIR", raising=False)
    manager = SensorManager()
    assert manager.ultrasonic_enabled is False  # master switch ships off
    assert manager.ultrasonic.backend == "mcu_usb"
    assert manager.ultrasonic.config.no_echo_is_clear is False
    assert manager.ultrasonic.config.out_of_range_is_clear is False
    assert manager.ultrasonic.obstacle_threshold_cm == 30.0


# ---- sectors for the AI model input and the recorded datasets ---------------------------------


def entry(sensor_id, cm, valid=True):
    return {"sensor_id": sensor_id, "distance_cm": cm if valid else 0.0, "valid": valid}


def test_four_sensors_map_to_the_three_model_sectors_with_the_nearer_inner_in_the_centre():
    out = sector_distances(
        [
            entry("outer_left", 90.0), entry("inner_left", 140.0),
            entry("inner_right", 70.0), entry("outer_right", 110.0),
        ]
    )  # fmt: skip
    assert out == {
        "front_left_cm": 90.0, "front_center_cm": 70.0, "front_right_cm": 110.0,
        "min_distance_cm": 70.0, "channels_cm": [90.0, 140.0, 70.0, 110.0],
    }  # fmt: skip


def test_a_sector_without_a_valid_echo_reads_zero_which_the_ai_path_treats_as_an_obstacle():
    out = sector_distances(
        [
            entry("outer_left", 0.0, valid=False), entry("inner_left", 140.0),
            entry("inner_right", 0.0, valid=False), entry("outer_right", 0.0, valid=False),
        ]
    )  # fmt: skip
    assert (out["front_left_cm"], out["front_center_cm"], out["front_right_cm"]) == (
        0.0,
        140.0,
        0.0,
    )
    assert out["min_distance_cm"] == 140.0
    assert out["channels_cm"] == [0.0, 140.0, 0.0, 0.0]
    none_valid = sector_distances([entry(n, 0.0, valid=False) for n in NAMES])
    assert none_valid["min_distance_cm"] == 0.0 and none_valid["front_center_cm"] == 0.0


def test_legacy_three_sensor_ids_keep_their_one_to_one_mapping():
    out = sector_distances(
        [entry("front_left", 50.0), entry("front_center", 60.0), entry("front_right", 70.0)]
    )
    assert (out["front_left_cm"], out["front_center_cm"], out["front_right_cm"]) == (
        50.0,
        60.0,
        70.0,
    )
    assert out["min_distance_cm"] == 50.0 and out["channels_cm"] == [50.0, 60.0, 70.0]
    assert sector_distances([])["min_distance_cm"] == 0.0


def test_sectors_survive_the_dataset_round_trip_and_old_recordings_still_load():
    frame = MowerDataFrame()
    frame.ultrasonic = UltrasonicData(
        **sector_distances([entry(n, 50.0 + 10 * i) for i, n in enumerate(NAMES)])
    )
    restored = MowerDataFrame.from_dict(frame.to_dict())
    assert restored.ultrasonic.channels_cm == [50.0, 60.0, 70.0, 80.0]
    legacy = frame.to_dict()
    legacy["ultrasonic"].pop("channels_cm")  # recorded before the four-sensor array existed
    assert MowerDataFrame.from_dict(legacy).ultrasonic.channels_cm == []


async def test_recorder_fills_the_frame_from_the_four_channel_entries():
    clock, _factory, iface = await active(
        sensors=[SensorModel(distance_cm=d) for d in (90.0, 140.0, 70.0, 110.0)]
    )

    class Manager:
        ultrasonic = iface

        async def read_all_sensors(self):
            from backend.src.models import SensorData

            return SensorData()

    recorder = PerimeterRecordingService(sensor_manager=Manager(), robohat_service=None)
    frame = MowerDataFrame()
    await recorder._capture_sensor_data(frame)
    u = frame.ultrasonic
    assert (u.front_left_cm, u.front_center_cm, u.front_right_cm) == pytest.approx(
        (90.0, 70.0, 110.0), abs=0.1
    )
    assert u.min_distance_cm == pytest.approx(70.0, abs=0.1)
    assert len(u.channels_cm) == 4
