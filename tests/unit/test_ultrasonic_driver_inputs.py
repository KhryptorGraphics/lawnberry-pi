"""Host driver: what it accepts from its caller (thresholds), and what it does with every frame.

US3-06: a NaN, zero or negative stop threshold made a 27 cm echo read CLEAR (NaN compares false
with everything), so the constructor and the per-call overrides refuse it.
US3-10: ``reading_hook`` is the raw frame log of ``scripts/ultrasonic_mcu_probe.py`` (audit M-40):
it sees EVERY CRC-valid reading frame with what the driver did with it, not only the accepted ones.
The rest pins link behaviour the mutation run of the review found unguarded.
"""

from __future__ import annotations

import math
import struct

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto
from backend.src.drivers.sensors.jsn_sr04t_driver import (
    DEFAULT_QUIET_AFTER,
    RECONNECT_BACKOFF_S,
    ChannelStatus,
    JsnSr04tMcuDriver,
    LinkState,
    Verdict,
)
from backend.src.models import SensorStatus
from backend.src.models.hardware_config import UltrasonicConfig
from backend.src.services.sensor_manager import SensorCoordinator, UltrasonicSensorInterface
from tests.unit._usonic_fakes import FakeClock, FakeMcuFactory, SensorModel
from tests.unit._usonic_fakes import P as fw
from tests.unit.test_jsn_sr04t_driver import (
    ScriptedPort,
    mcu_ms,
    scripted_active,
    scripted_rig,
    steady,
)

# ---- US3-06: the stop threshold ---------------------------------------------------------------


def build(**kw) -> JsnSr04tMcuDriver:
    clock = FakeClock()
    return JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-usonic"),
        serial_factory=lambda _p, _b: ScriptedPort(),
        clock=clock.now,
        wall_clock=clock.wall,
        sleep=clock.sleep,
        simulate=False,
        **kw,
    )


@pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf, 0.0, -0.0, -1.0, -30.0, None, "high", [30.0]],
    ids=repr,
)  # fmt: skip
def test_a_stop_threshold_that_is_not_a_finite_positive_number_is_refused(bad):
    with pytest.raises(ValueError, match="obstacle_threshold_cm"):
        build(obstacle_threshold_cm=bad)


@pytest.mark.parametrize("good", [0.5, 10, 30.0, 45, "35.5", 1e6])
def test_a_finite_positive_stop_threshold_is_taken_as_a_float(good):
    driver = build(obstacle_threshold_cm=good)
    assert driver.obstacle_threshold_cm == float(good) and isinstance(
        driver.obstacle_threshold_cm, float
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0, "x"], ids=repr)
def test_a_per_call_threshold_override_is_refused_the_same_way(bad):
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)  # four channels at 100 cm
    with pytest.raises(ValueError, match="threshold_cm"):
        driver.snapshot(threshold_cm=bad)
    with pytest.raises(ValueError, match="threshold_cm"):
        driver.obstacle_verdict(threshold_cm=bad)


def test_the_override_replaces_the_configured_threshold_and_none_keeps_it():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    assert driver.obstacle_verdict()["state"] == "clear"
    assert driver.obstacle_verdict(None)["threshold_cm"] == 30.0
    assert driver.obstacle_verdict(150.0)["state"] == "obstacle"  # 100 cm echoes, 150 cm limit
    assert {c.verdict for c in driver.snapshot(150.0)} == {Verdict.OBSTACLE}
    assert {c.verdict for c in driver.snapshot()} == {Verdict.CLEAR}


@pytest.mark.parametrize("bad", [math.nan, 0.0, -1.0])
async def test_the_interface_fails_closed_when_it_is_given_a_bad_threshold(bad):
    # UltrasonicSensorInterface builds its driver inside a try/except: a refused threshold leaves
    # the array offline, never a driver that reports 27 cm as clear
    iface = UltrasonicSensorInterface(
        SensorCoordinator(), config={"port": "/dev/ttyACM-usonic"}, obstacle_threshold_cm=bad
    )
    assert iface._driver is None
    assert await iface.initialize() is False
    assert iface.status != SensorStatus.ONLINE
    assert iface.obstacle_verdict()["state"] == "unknown"


# ---- US3-10: the raw frame hook ------------------------------------------------------------------


def record(driver) -> list[tuple[int, str]]:
    seen: list[tuple[int, str]] = []
    driver.reading_hook = lambda r, wall, fate: seen.append((r.sensor, fate))
    return seen


def test_every_reading_frame_reaches_the_hook_with_what_the_driver_did_with_it():
    from backend.src.drivers.sensors.jsn_sr04t_driver import Disposition

    clock, port, driver, _ = scripted_rig(disabled_channels=["inner_right"])
    seen = record(driver)
    driver.poll()
    port.info()
    port.reading(sensor=0)  # during the handshake: not trusted
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert seen == [(0, Disposition.IGNORED_LINK)]
    steady(clock, port, driver, 0.3, sensors=(0,))  # accepted
    port.reading(sensor=2, status=proto.EchoStatus.DISABLED, echo_us=0, t_ms=mcu_ms(clock))
    port.reading(sensor=4, t_ms=mcu_ms(clock))  # no such channel
    port.reading(sensor=0, t_ms=mcu_ms(clock, behind_s=5.0))  # a replay from 5 s ago
    driver.poll()
    assert [fate for _s, fate in seen[1:]] == [
        Disposition.ACCEPTED, Disposition.ACCEPTED, Disposition.ACCEPTED,
        Disposition.IGNORED_CHANNEL, Disposition.BAD_SENSOR, Disposition.REFUSED_CLOCK,
    ]  # fmt: skip
    assert [s for s, _f in seen[-3:]] == [2, 4, 0]


def test_the_hook_sees_the_raw_frame_not_the_drivers_interpretation():
    from backend.src.drivers.sensors.jsn_sr04t_driver import Disposition

    clock, port, driver, _ = scripted_active()
    got = []
    driver.reading_hook = lambda r, wall, fate: got.append((r, fate))
    port.reading(
        sensor=1, status=proto.EchoStatus.NO_FALL, echo_us=29_999, health=1, fails=9, t_ms=777
    )
    driver.poll()
    ((r, fate),) = got
    assert (r.sensor, r.status, r.echo_us, r.health, r.fails, r.t_ms) == (1, 2, 29_999, 1, 9, 777)
    assert fate is Disposition.ACCEPTED


def test_a_raising_hook_is_logged_and_never_disturbs_ingestion(caplog):
    clock, port, driver, _ = scripted_active()
    driver.reading_hook = lambda r, wall, fate: 1 / 0
    steady(clock, port, driver, 0.5)
    assert all(c.valid for c in driver.snapshot()) and driver.stats.readings == 20
    assert "reading hook failed" in caplog.text


def test_a_corrupt_frame_never_reaches_the_hook():
    clock, port, driver, _ = scripted_active()
    seen = record(driver)
    frame = bytearray(
        proto.encode_frame(
            proto.FrameType.READING, 9, fw.pack_reading(0, 0, 0, 0, 5828, 1, 600_000)
        )
    )
    frame[-1] ^= 0x10
    port.inbox += frame
    driver.poll()
    assert seen == [] and driver.stats.crc_errors == 1


# ---- reading intake limits ---------------------------------------------------------------------


@pytest.mark.parametrize("sensor", [4, 5, 255])
def test_a_reading_for_a_channel_beyond_the_configured_ones_is_counted_and_harmless(sensor):
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 0.5)
    port.reading(sensor=sensor, t_ms=mcu_ms(clock))
    driver.poll()  # must not raise (the service loop would log and back off for 0.5 s)
    assert driver.stats.bad_readings == 1 and all(c.valid for c in driver.snapshot())


@pytest.mark.parametrize("count", [3, 5, 8, 255])
def test_an_mcu_with_more_or_fewer_sensors_than_configured_is_refused(count):
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info(sensor_count=count)
    driver.poll()
    assert driver.link_state is LinkState.FAULT
    assert driver.link_fault.startswith("channel_count_mismatch")
    assert port.types() == [], "nothing is configured on an MCU that cannot be the one expected"


def test_the_configuration_the_host_asks_for_is_what_the_firmware_documents():
    clock, port, driver, _ = scripted_rig()
    assert driver.expected_mcu_config() == proto.McuConfig(
        enable_mask=0b1111, echo_window_ms=30, gap_ms=30, quiet_after=8
    )
    assert DEFAULT_QUIET_AFTER == 8
    off = scripted_rig(disabled_channels=["inner_left"])[2]
    assert off.expected_mcu_config().enable_mask == 0b1101


def test_lost_frames_are_counted_across_the_sequence_wrap():
    clock, port, driver, _ = scripted_active()
    port.push(proto.FrameType.PONG, struct.pack("<II", 1, 2), seq=250)
    driver.poll()
    base = driver.stats.frames_lost
    port.push(proto.FrameType.PONG, struct.pack("<II", 1, 2), seq=3)  # 251..2 were lost: 8 frames
    driver.poll()
    assert driver.stats.frames_lost - base == 8


def test_a_good_frame_behind_a_corrupted_length_is_released_once_the_line_is_quiet():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 0.5)
    n = driver.stats.readings
    clock.advance(0.1)
    port.inbox += bytes([proto.SYNC0, proto.SYNC1, 1, 0x83, 0, 48]) + b"\x00" * 6  # claims 48 bytes
    port.reading(sensor=0, t_ms=mcu_ms(clock))  # a good frame, now held hostage behind it
    driver.poll()
    assert driver.stats.readings == n, (
        "the good frame cannot be seen while the bogus one is pending"
    )
    clock.advance(0.15)  # quiet for longer than the partial-frame timeout (0.1 s)
    driver.poll()
    assert driver.stats.readings == n + 1 and driver.stats.partial_timeouts == 1


def test_a_silent_identified_link_that_loses_its_handshake_state_is_not_probed_with_hello():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info()
    driver.poll()  # CONFIG sent, no ACK ever
    for _ in range(
        80
    ):  # 1.6 s: longer than link_timeout_ms (1.5 s), shorter than the handshake window
        clock.advance(0.02)
        driver.poll()
    assert driver.link_state is LinkState.HANDSHAKE
    assert port.types() == [proto.FrameType.CONFIG], "the handshake has its own timeout, no probes"


def test_a_configuration_drift_late_in_a_long_link_does_not_trip_the_old_handshake_deadline():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 6.0)  # far beyond the 2 s handshake window of the connection
    port.info(mask=0b0111)  # somebody masked a channel off: back to the handshake
    clock.advance(0.02)
    driver.poll()
    assert driver.link_state is LinkState.HANDSHAKE
    for _ in range(25):  # 0.5 s: a fresh window has been granted, the connect-time one is long gone
        clock.advance(0.02)
        driver.poll()
    assert driver.stats.link_losses == 0 and driver.link_state is LinkState.HANDSHAKE
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE


# ---- reconnecting ------------------------------------------------------------------------------


def attempts_over(clock, factory_kwargs, seconds, *, fail_first):
    factory = FakeMcuFactory(clock, **factory_kwargs)
    factory.fail_opens = fail_first
    times: list[float] = []

    def timed(path, baud):
        times.append(clock.now() - 1000.0)
        return factory(path, baud)

    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-usonic"),
        serial_factory=timed, clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=False,
    )  # fmt: skip
    for _ in range(round(seconds / 0.01)):
        clock.advance(0.01)
        driver.poll()
    return driver, times, factory


def test_open_failures_back_off_through_the_whole_schedule_and_never_run_off_its_end():
    clock = FakeClock()
    driver, times, _ = attempts_over(clock, {"sensors": [SensorModel()] * 4}, 40.0, fail_first=9)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    want = [*RECONNECT_BACKOFF_S, *[RECONNECT_BACKOFF_S[-1]] * 3]  # 0.25 .. 5, then 5 s for good
    assert gaps[:8] == pytest.approx(want, abs=0.03)
    assert driver.stats.connect_failures == 9 and driver.link_state is LinkState.ACTIVE


def test_the_backoff_starts_over_after_a_successful_handshake():
    clock = FakeClock()
    driver, times, factory = attempts_over(
        clock, {"sensors": [SensorModel(distance_cm=120.0)] * 4}, 10.0, fail_first=4
    )
    assert driver.link_state is LinkState.ACTIVE and len(times) == 5  # four failures, then success
    factory.current.unplug()
    factory.fail_opens = 2
    n, unplugged = len(times), clock.now() - 1000.0
    for _ in range(round(20.0 / 0.01)):
        clock.advance(0.01)
        driver.poll()
    after = [t - unplugged for t in times[n:]]
    assert after[:3] == pytest.approx([0.25, 0.75, 1.75], abs=0.05), (
        f"the schedule did not start over (attempts at {after})"
    )


def test_a_failed_open_does_not_leave_its_reason_on_the_next_connection():
    clock = FakeClock()
    factory = FakeMcuFactory(clock, sensors=[SensorModel()] * 4)
    factory.fail_opens = 1
    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-usonic"),
        serial_factory=factory, clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=False,
    )  # fmt: skip
    driver.poll()
    assert driver.link_fault.startswith("open_failed")
    clock.advance(0.3)
    driver.poll()  # the port opens: the handshake has not even started yet
    assert driver.link_state is LinkState.HANDSHAKE and driver.link_fault is None


async def test_initialize_waits_for_the_handshake_window_and_not_longer():
    clock, port, driver, _ = scripted_rig(handshake_timeout_s=2.0)
    await driver.initialize()  # an MCU that never answers
    assert 2.0 <= clock.now() - 1000.0 <= 2.1
    assert driver.initialized and driver.link_state is not LinkState.ACTIVE


# ---- what the channel shows ----------------------------------------------------------------------


def test_a_stuck_fault_status_is_a_fault_not_a_clear_sensor():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 0.5)
    port.reading(sensor=0, status=proto.EchoStatus.STUCK, echo_us=0, t_ms=mcu_ms(clock) + 10)
    driver.poll()
    c = driver.snapshot()[0]
    assert c.status is ChannelStatus.FAULT and c.verdict is Verdict.UNKNOWN and not c.valid


def test_a_lost_link_leaves_nothing_of_the_old_readings_to_look_at():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    assert all(c.measured_mono is not None and c.echo_us == 5828 for c in driver.snapshot())
    port.broken = OSError("device reports readiness to read but returned no data")
    clock.advance(0.02)
    driver.poll()
    for c in driver.snapshot():
        assert c.status is ChannelStatus.LINK_DOWN and not c.valid
        assert c.measured_mono is None and c.timestamp is None and c.echo_us is None
        assert c.distance_cm == 0.0 and c.clear_to_cm is None


def test_the_handshake_counts_what_it_heard():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info()
    driver.poll()
    assert driver.stats.infos == 1 and driver.stats.frames_rx == 1
    assert driver.info is not None and driver.info.sensor_count == 4


def test_silence_while_the_driver_waits_for_the_ack_is_never_clear():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info()
    driver.poll()
    assert all(c.verdict is Verdict.UNKNOWN for c in driver.snapshot())
    assert driver.obstacle_verdict()["state"] == "unknown"


def test_a_no_echo_is_clear_policy_says_so_once_and_the_shipped_one_stays_quiet(caplog):
    import asyncio

    def initialise(**cfg):
        clock, _port, driver, _ = scripted_rig(**cfg)
        asyncio.run(driver.initialize())

    with caplog.at_level("WARNING", logger="backend.src.drivers.sensors.jsn_sr04t_driver"):
        initialise()
    assert "no_echo_is_clear" not in caplog.text
    with caplog.at_level("WARNING", logger="backend.src.drivers.sensors.jsn_sr04t_driver"):
        initialise(no_echo_is_clear=True, alive_window_s=7.5)
    assert "no_echo_is_clear is ON" in caplog.text and "alive_window_s=7.5" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="backend.src.drivers.sensors.jsn_sr04t_driver"):
        initialise(no_echo_is_clear=True, no_echo_requires_alive=False)
    assert "no_echo_requires_alive is OFF" in caplog.text
