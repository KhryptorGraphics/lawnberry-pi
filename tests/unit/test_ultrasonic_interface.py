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
    "alive", "verdict", "clear_to_cm", "echo_us", "fails", "measured_mono",
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


def test_a_sector_fed_by_a_channel_without_a_valid_echo_reads_zero_fail_closed():
    out = sector_distances(
        [
            entry("outer_left", 0.0, valid=False), entry("inner_left", 140.0),
            entry("inner_right", 0.0, valid=False), entry("outer_right", 0.0, valid=False),
        ]
    )  # fmt: skip
    # the centre needs BOTH inner sensors: the valid one must not hide the dead one (US-09)
    assert (out["front_left_cm"], out["front_center_cm"], out["front_right_cm"]) == (0.0, 0.0, 0.0)
    assert out["min_distance_cm"] == 0.0
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


# ---- US-03: a channel that is not fitted is not a channel the guard waits for -----------------


async def test_channel_ids_list_only_the_fitted_channels_while_the_entries_still_show_all():
    _clock, _f, iface = await active(disabled_channels=["inner_right"])
    assert iface.channel_ids == ["outer_left", "inner_left", "outer_right"]
    entries = {e["sensor_id"]: e for e in await iface.read_ultrasonic()}
    assert list(entries) == NAMES  # the disabled channel stays visible, as "disabled"
    assert entries["inner_right"]["verdict"] == "disabled"
    assert iface.channel_ids == [n for n, e in entries.items() if e["verdict"] != "disabled"]
    _c, _f, nothing = await active(disabled_channels=NAMES)
    assert nothing.channel_ids == []


# ---- US-04: the entries the guard reads do not depend on who calls read_ultrasonic() ----------


async def test_last_readings_follow_the_driver_without_anyone_calling_read_ultrasonic():
    clock, _f, iface = await active()  # read_ultrasonic() is never called in this test
    assert [e["sensor_id"] for e in iface.last_readings] == NAMES
    assert all(e["status"] == "ok" and e["valid"] for e in iface.last_readings)
    before = {e["sensor_id"]: datetime.fromisoformat(e["timestamp"]) for e in iface.last_readings}
    run(clock, iface, 0.5)  # the driver's service loop keeps polling, the telemetry loop is away
    after = {e["sensor_id"]: datetime.fromisoformat(e["timestamp"]) for e in iface.last_readings}
    assert all(after[name] > before[name] for name in NAMES)


async def test_a_dead_service_loop_ages_the_entries_out_on_their_own():
    clock, _f, iface = await active()
    assert all(e["valid"] for e in iface.last_readings)
    clock.advance(1.0)  # nothing polls the driver and nobody reads the interface
    entries = iface.last_readings
    assert all(e["status"] == "stale" and not e["valid"] for e in entries)
    assert all(e["verdict"] == "unknown" and e["age_ms"] >= 1000 for e in entries)


async def test_last_readings_is_the_cached_list_when_the_interface_is_not_serving_the_mcu():
    iface = UltrasonicSensorInterface(SensorCoordinator(), config={"port": "/dev/ttyACM-usonic"})
    assert iface.enabled is False and iface.last_readings == []
    iface.last_readings = [{"sensor_id": "x"}]  # assignment stays possible (legacy backend, tests)
    assert iface.last_readings == [{"sensor_id": "x"}]


# ---- US-11: shutdown really shuts the interface down ------------------------------------------


async def test_shutdown_disables_the_interface_so_a_late_tick_cannot_reopen_the_port():
    clock, factory, iface = await active()
    await iface.shutdown()
    assert iface.enabled is False and factory.current.closed
    clock.advance(1.0)
    assert await iface.read_ultrasonic() == []  # the telemetry loop's next tick during teardown
    assert len(factory.opens) == 1 and factory.current.closed
    assert iface.last_readings == [] and iface.status == SensorStatus.OFFLINE
    assert iface.obstacle_verdict()["state"] == "unknown"
    assert await iface.initialize() is True  # an explicit initialize() brings it back
    assert len(factory.opens) == 2 and iface.enabled is True


# ---- I01 / I03: read_ultrasonic() polls the driver, and a failure is an ERROR ----------------


async def test_read_ultrasonic_polls_the_driver_itself_when_nothing_else_does():
    clock = FakeClock()
    sensors = [SensorModel(distance_cm=d) for d in (80, 120, 200, 60)]
    factory = FakeMcuFactory(clock, sensors=sensors)
    iface = make(clock, factory)
    iface.enabled = True  # what initialize() does, WITHOUT starting the driver's service loop
    assert iface._driver.running is False
    entries: list = []
    for _ in range(60):  # 1.2 s in which only the telemetry read ever touches the driver
        clock.advance(0.02)
        entries = await iface.read_ultrasonic()
    assert len(factory.opens) == 1
    assert all(e["status"] == "ok" and e["valid"] for e in entries)


async def test_an_exception_while_reading_leaves_status_error_and_returns_nothing(monkeypatch):
    _clock, _f, iface = await active()
    await iface.read_ultrasonic()
    assert iface.status == SensorStatus.ONLINE

    def boom() -> None:
        raise RuntimeError("serial driver blew up")

    monkeypatch.setattr(iface._driver, "poll", boom)
    assert await iface.read_ultrasonic() == []
    assert iface.status == SensorStatus.ERROR


# ---- I04 / I05 / I06: the legacy HC-SR04 verdicts ---------------------------------------------


class StubLegacyDriver:
    """Stands in for the GPIO HC-SR04 driver: it just returns the readings it is given."""

    def __init__(self, readings):
        self.readings = readings

    async def read_all(self):
        return list(self.readings)


def legacy_iface(*cms, threshold=30.0):
    """A legacy interface whose three sensors read ``cms`` (None = no valid echo)."""
    from backend.src.drivers.sensors.ultrasonic_driver import UltrasonicReading

    iface = UltrasonicSensorInterface(
        SensorCoordinator(), config={"backend": "gpio_hcsr04"}, obstacle_threshold_cm=threshold
    )
    iface.enabled, iface.status = True, SensorStatus.ONLINE
    iface._driver = StubLegacyDriver(
        [
            UltrasonicReading(sensor_id=sid, distance_cm=cm or 0.0, valid=cm is not None)
            for sid, cm in zip(iface.channel_ids, cms, strict=True)
        ]
    )
    return iface


async def test_legacy_reading_without_a_valid_echo_is_unknown_never_clear():
    iface = legacy_iface(None, 200.0, 150.0)
    entries = {e["sensor_id"]: e for e in await iface.read_ultrasonic()}
    bad = entries["front_left"]
    assert (bad["valid"], bad["distance_cm"], bad["verdict"]) == (False, 0.0, "unknown")
    assert bad["clear_to_cm"] is None and bad["status"] == "no_echo"
    assert entries["front_center"]["verdict"] == "clear"


@pytest.mark.parametrize(
    ("cm", "verdict"), [(29.9, "obstacle"), (30.0, "obstacle"), (30.1, "clear")]
)
async def test_legacy_threshold_boundary_is_inclusive(cm, verdict):
    iface = legacy_iface(cm, 200.0, 200.0)
    entries = await iface.read_ultrasonic()
    assert entries[0]["verdict"] == verdict
    assert iface.obstacle_verdict()["state"] == ("obstacle" if verdict == "obstacle" else "clear")


async def test_legacy_aggregate_needs_every_fitted_channel_valid_to_be_clear():
    clear = legacy_iface(200.0, 180.0, 150.0)
    await clear.read_ultrasonic()
    assert clear.obstacle_verdict()["state"] == "clear"
    one_dead = legacy_iface(200.0, None, 150.0)  # two valid channels do not vouch for the third
    await one_dead.read_ultrasonic()
    verdict = one_dead.obstacle_verdict()
    assert verdict["state"] == "unknown" and verdict["reasons"] == ["front_center:no_echo"]
    near = legacy_iface(20.0, None, 150.0)  # an obstacle beats an unknown
    await near.read_ultrasonic()
    assert near.obstacle_verdict()["state"] == "obstacle"


# ---- US-08: ONLINE only while every fitted channel is serving ---------------------------------


async def test_status_is_online_only_while_every_fitted_channel_is_serving():
    models = [SensorModel(mode=m) for m in ("silent", "stuck", "error", "silent")]
    _c, _f, dead = await active(sensors=models)  # reviewer exp e10: the link is fine, nothing else
    await dead.read_ultrasonic()
    assert dead.status == SensorStatus.ERROR
    models = [SensorModel(distance_cm=100.0)] * 3 + [SensorModel(mode="silent")]
    _c, _f, degraded = await active(sensors=models)  # one dead channel of four stays visible
    await degraded.read_ultrasonic()
    assert degraded.status == SensorStatus.ERROR
    _c, _f, fitted = await active(sensors=models, disabled_channels=["outer_right"])
    await fitted.read_ultrasonic()  # the silent one is not fitted: it is not a dead channel
    assert fitted.status == SensorStatus.ONLINE
    models = [SensorModel(distance_cm=d) for d in (12.0, 450.0, 100.0, 100.0)]
    _c, _f, working = await active(sensors=models, echo_window_ms=40)
    await working.read_ultrasonic()  # blind-zone and out-of-range channels did measure something
    assert working.status == SensorStatus.ONLINE
    _c, _f, nothing = await active(disabled_channels=NAMES)
    await nothing.read_ultrasonic()
    assert nothing.status == SensorStatus.ERROR  # nothing fitted can never be online


# ---- US-09: a dead or blind channel is not hidden by its healthy neighbour --------------------


def test_the_centre_sector_needs_both_inner_sensors_and_min_distance_needs_every_channel():
    # reviewer exp e13: inner_left blind or dead, inner_right sees 200 cm
    out = sector_distances(
        [
            entry("outer_left", 150.0), entry("inner_left", 0.0, valid=False),
            entry("inner_right", 200.0), entry("outer_right", 180.0),
        ]
    )  # fmt: skip
    assert (out["front_left_cm"], out["front_center_cm"], out["front_right_cm"]) == (
        150.0, 0.0, 180.0,
    )  # fmt: skip
    assert out["min_distance_cm"] == 0.0


def test_a_channel_that_is_not_fitted_does_not_count_against_the_sectors():
    def disabled(sensor_id):
        row = entry(sensor_id, 0.0, valid=False)
        return {**row, "status": "disabled", "verdict": "disabled"}

    out = sector_distances(
        [
            entry("outer_left", 150.0), entry("inner_left", 140.0),
            disabled("inner_right"), entry("outer_right", 180.0),
        ]
    )  # fmt: skip
    assert (out["front_left_cm"], out["front_center_cm"], out["front_right_cm"]) == (
        150.0, 140.0, 180.0,
    )  # fmt: skip
    assert out["min_distance_cm"] == 140.0
    nothing = sector_distances([disabled(n) for n in NAMES])
    assert nothing["min_distance_cm"] == 0.0 and nothing["front_center_cm"] == 0.0
    both_inner_off = sector_distances(
        [entry("outer_left", 150.0), disabled("inner_left"), disabled("inner_right"),
         entry("outer_right", 180.0)]
    )  # fmt: skip
    assert both_inner_off["front_center_cm"] == 0.0  # no sensor looks ahead: not "clear"


def test_a_legacy_sector_or_channel_without_a_valid_echo_is_zero_too():
    out = sector_distances(
        [
            entry("front_left", 50.0),
            entry("front_center", 0.0, valid=False),
            entry("front_right", 70.0),
        ]
    )
    sectors = (out["front_left_cm"], out["front_center_cm"], out["front_right_cm"])
    assert sectors == (50.0, 0.0, 70.0)
    assert out["min_distance_cm"] == 0.0
