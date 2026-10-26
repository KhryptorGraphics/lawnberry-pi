"""What the telemetry hub tells the operator about the GNSS fix (audit GPS-05, SAF-04, SAF-S1).

GPS-05: the ``telemetry.navigation`` topic filled every field a reading did NOT have from the last
reading that did: a receiver that now said "plain fix, accuracy unknown" was shown with the last
RTK 3 cm next to its new status, and a driver with nothing to report was shown as the old
coordinates, satellites and speed, with no age. The mutants h01-h04 of the same area: the hub
overriding an operator's ``GPS_DEVICE``, naming a hard-coded port, and HDOP shown as an accuracy.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime, timedelta

import pytest

from backend.src.models.sensor_data import GpsMode, GpsReading, ImuReading, SensorData
from backend.src.services import websocket_hub as hub_module
from backend.src.services.websocket_hub import WebSocketHub


class Sensors:
    """The hub's sensor source: whatever reading a test sets."""

    initialized = True
    ultrasonic = None
    stereo_camera = None

    def __init__(self, reading: GpsReading | None = None) -> None:
        self.reading = reading
        self.imu = ImuReading(yaw=90.0)

    async def read_all_sensors(self) -> SensorData:
        return SensorData(gps=self.reading, imu=self.imu)


def make_hub(sensors: Sensors) -> WebSocketHub:
    hub = WebSocketHub()
    hub._sensor_manager = sensors
    hub._gps_warm_done = True
    return hub


def rtk(**kw) -> GpsReading:
    fields = {
        "latitude": 48.1,
        "longitude": 11.5,
        "altitude": 545.0,
        "accuracy": 0.03,
        "speed": 0.4,
        "satellites": 22,
        "hdop": 0.6,
        "rtk_status": "RTK_FIXED",
        "mode": GpsMode.LC29H_UART,
    }
    fields.update(kw)
    return GpsReading(**fields)


async def navigation_topics(hub: WebSocketHub, telemetry: dict) -> list[dict]:
    sent: list[tuple[str, dict]] = []

    async def record(topic: str, data: dict) -> None:
        sent.append((topic, data))

    hub.broadcast_to_topic = record  # type: ignore[method-assign]
    await hub._broadcast_telemetry_topics(telemetry)
    return [data for topic, data in sent if topic == "telemetry.navigation"]


async def navigation_of(hub: WebSocketHub, sensors: Sensors) -> dict:
    telemetry = await hub._generate_telemetry()
    (payload,) = await navigation_topics(hub, telemetry)
    return payload


@pytest.fixture(autouse=True)
def hardware_path(monkeypatch):
    monkeypatch.setenv("SIM_MODE", "0")


# ------------------------------------------------------------------------------------------ GPS-05


async def test_an_unknown_accuracy_is_not_shown_as_the_last_rtk_figure():
    sensors = Sensors(rtk())
    hub = make_hub(sensors)
    first = await navigation_of(hub, sensors)
    assert first["position"]["accuracy"] == 0.03 and first["accuracy_m"] == 0.03
    assert first["position"]["rtk_status"] == "RTK_FIXED"
    sensors.reading = rtk(accuracy=None, rtk_status="GPS_FIX")  # a plain fix, accuracy unknown
    second = await navigation_of(hub, sensors)
    assert second["position"]["rtk_status"] == "GPS_FIX"
    assert second["position"]["accuracy"] is None
    assert "accuracy_m" not in second


async def test_a_driver_with_nothing_to_report_shows_no_old_position_as_live():
    sensors = Sensors(rtk())
    hub = make_hub(sensors)
    await navigation_of(hub, sensors)
    sensors.reading = None  # the receiver is gone
    gone = await navigation_of(hub, sensors)
    position = gone["position"]
    for key in (
        "latitude",
        "longitude",
        "altitude",
        "accuracy",
        "satellites",
        "speed",
        "rtk_status",
        "hdop",
        "gps_mode",
        "fix_age_s",
    ):
        assert position[key] is None, key
    for key in ("speed_mps", "accuracy_m", "hdop", "fix_age_s"):
        assert key not in gone, key


async def test_a_no_fix_reading_shows_its_status_and_none_of_the_old_fix():
    sensors = Sensors(rtk())
    hub = make_hub(sensors)
    await navigation_of(hub, sensors)
    sensors.reading = GpsReading(rtk_status="NO_FIX", satellites=3, mode=GpsMode.LC29H_UART)
    no_fix = (await navigation_of(hub, sensors))["position"]
    assert no_fix["rtk_status"] == "NO_FIX" and no_fix["satellites"] == 3
    assert no_fix["latitude"] is None and no_fix["longitude"] is None
    assert no_fix["accuracy"] is None and no_fix["altitude"] is None


async def test_the_navigation_topic_carries_what_the_reading_says_and_its_age():
    stamp = datetime.now(UTC) - timedelta(seconds=3.0)
    sensors = Sensors(rtk(timestamp=stamp))
    payload = await navigation_of(make_hub(sensors), sensors)
    assert payload["position"]["latitude"] == 48.1 and payload["position"]["satellites"] == 22
    assert payload["speed_mps"] == 0.4 and payload["hdop"] == 0.6
    assert payload["fix_age_s"] == pytest.approx(3.0, abs=0.5)
    assert payload["position"]["fix_age_s"] == payload["fix_age_s"]


async def test_a_replayed_reading_shows_the_age_it_has_not_a_fresh_one():
    old = datetime.now(UTC) - timedelta(seconds=40.0)
    sensors = Sensors(rtk(timestamp=old))
    payload = await navigation_of(make_hub(sensors), sensors)
    assert payload["fix_age_s"] == pytest.approx(40.0, abs=1.0)


@pytest.mark.parametrize("offset", [timedelta(minutes=10), None])
async def test_a_stamp_from_the_future_or_none_at_all_is_an_unknown_age_not_a_zero(offset):
    reading = rtk(timestamp=datetime.now(UTC) + offset) if offset else rtk()
    if offset is None:
        object.__setattr__(reading, "timestamp", None)
    sensors = Sensors(reading)
    payload = await navigation_of(make_hub(sensors), sensors)
    assert payload["position"]["fix_age_s"] is None and "fix_age_s" not in payload


async def test_the_dashboard_route_carries_the_age_of_the_fix(monkeypatch):
    from backend.src.api.routers import telemetry as telemetry_router

    stamp = datetime.now(UTC) - timedelta(seconds=7.0)
    sensors = Sensors(rtk(timestamp=stamp))
    monkeypatch.setattr(telemetry_router, "websocket_hub", make_hub(sensors))
    body = await telemetry_router.dashboard_telemetry()
    assert body["position"]["fix_age_s"] == pytest.approx(7.0, abs=1.0)
    assert body["position"]["latitude"] == 48.1 and body["position"]["accuracy"] == 0.03


# ------------------------------------------------------------------------------- SAF-04 (h03, h04)


async def test_an_unknown_accuracy_is_never_replaced_by_the_hdop():
    """HDOP is a dimensionless geometry factor: 0.6 shown as '0.6 m accurate' is how a plain fix
    of 3-4 m passed the health gate."""
    sensors = Sensors(rtk(accuracy=None, rtk_status="GPS_FIX", hdop=0.6))
    hub = make_hub(sensors)
    telemetry = await hub._generate_telemetry()
    assert telemetry["position"]["accuracy"] is None and telemetry["position"]["hdop"] == 0.6
    (payload,) = await navigation_topics(hub, telemetry)
    assert payload["position"]["accuracy"] is None and "accuracy_m" not in payload
    assert payload["hdop"] == 0.6 and payload["position"]["hdop"] == 0.6


# ---------------------------------------------------------------------------------------- h01, h02


@pytest.fixture
def hinted(monkeypatch):
    """A hub with no app state, a recorded ``preferred_gps_device`` and a clean GPS_DEVICE."""
    monkeypatch.setenv("GPS_DEVICE", "placeholder")
    monkeypatch.delenv("GPS_DEVICE")  # leaves the variable registered for the teardown restore
    calls: list[GpsMode] = []
    answer: dict[str, str | None] = {"device": "/dev/ttyAMA0"}

    def preferred(mode, ports=None):
        calls.append(mode)
        return answer["device"]

    monkeypatch.setattr(hub_module, "preferred_gps_device", preferred)
    monkeypatch.setenv("SIM_MODE", "1")
    return calls, answer


async def test_the_hub_points_the_driver_at_the_device_the_configuration_implies(hinted):
    calls, answer = hinted
    hub = WebSocketHub()
    await hub._ensure_sensor_manager()
    assert os.environ.get("GPS_DEVICE") == "/dev/ttyAMA0"
    assert calls == [GpsMode.LC29H_UART]  # asked for the configured receiver (hardware.yaml)
    await hub._sensor_manager.shutdown()


async def test_the_hub_gives_no_hint_when_there_is_no_sign_of_the_receiver(hinted):
    _calls, answer = hinted
    answer["device"] = None
    hub = WebSocketHub()
    await hub._ensure_sensor_manager()
    assert "GPS_DEVICE" not in os.environ
    await hub._sensor_manager.shutdown()


async def test_the_hub_never_overrides_the_device_the_operator_named(hinted, monkeypatch):
    calls, _answer = hinted
    monkeypatch.setenv("GPS_DEVICE", "/dev/ttyUSB5")
    hub = WebSocketHub()
    await hub._ensure_sensor_manager()
    assert os.environ["GPS_DEVICE"] == "/dev/ttyUSB5" and calls == []
    await hub._sensor_manager.shutdown()


def test_the_fix_age_helper_is_not_fooled_by_nan_or_a_wrong_type():
    from backend.src.drivers.sensors.gps_driver import reading_age_s

    now = datetime(2026, 10, 25, 12, 0, 10, tzinfo=UTC)
    stamped = lambda stamp: type("R", (), {"timestamp": stamp})()  # noqa: E731
    assert reading_age_s(stamped(now - timedelta(seconds=4)), now) == pytest.approx(4.0)
    assert reading_age_s(stamped(now), now) == 0.0  # exactly now is age 0, not unknown
    assert reading_age_s(stamped(now + timedelta(seconds=1)), now) is None
    assert reading_age_s(stamped(datetime(2026, 10, 25, 12, 0, 6)), now) == pytest.approx(4.0)
    assert reading_age_s(stamped("2026-10-25T12:00:06+00:00"), now) == pytest.approx(4.0)
    assert reading_age_s(stamped(None), now) is None
    assert reading_age_s(stamped(math.nan), now) is None
    assert reading_age_s(object(), now) is None
    assert reading_age_s(None, now) is None
