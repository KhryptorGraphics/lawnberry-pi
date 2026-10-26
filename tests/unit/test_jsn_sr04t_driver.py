"""Host driver for four JSN-SR04T behind the USB MCU, against the REAL firmware core.

Every test talks to ``firmware/ultrasonic-mcu`` (through ``FakeMcuSerial``) or to a hand-scripted
port; nothing touches hardware. Each test names the fault it would catch. The safety property
under test throughout: anything the host cannot positively establish is UNKNOWN, never CLEAR.
"""

from __future__ import annotations

import asyncio
import json
import random
import struct
from collections.abc import Callable

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto
from backend.src.drivers.sensors.jsn_sr04t_driver import (
    ChannelStatus,
    Disposition,
    Health,
    JsnSr04tMcuDriver,
    LinkState,
    Verdict,
    cm_to_echo_us,
    echo_to_cm,
)
from backend.src.models.hardware_config import UltrasonicConfig
from tests.unit._usonic_fakes import (
    FakeClock,
    FakeMcuFactory,
    SensorModel,
    approach_then_silence,
    corrupt_length,
    drop,
    flip_crc_bit,
    flip_payload_bit,
    scripted,
    truncate,
    us,
    with_garbage,
)
from tests.unit._usonic_fakes import P as fw

NAMES = ["outer_left", "inner_left", "inner_right", "outer_right"]
DISTANCES = {"outer_left": 80.0, "inner_left": 120.0, "inner_right": 200.0, "outer_right": 60.0}


def echo(cm: float) -> SensorModel:
    return SensorModel(distance_cm=cm)


def four_clear() -> list[SensorModel]:
    return [echo(DISTANCES[n]) for n in NAMES]


class Rig:
    """Fake clock + fake MCU factory + the driver under test (real-mode, never simulated)."""

    def __init__(self, sensors=None, cfg=None, factory_kwargs=None, **driver_kw):
        self.clock = FakeClock()
        port_kwargs = dict(factory_kwargs or {})
        port_kwargs.setdefault("sensors", sensors if sensors is not None else four_clear())
        self.factory = FakeMcuFactory(self.clock, **port_kwargs)
        self.cfg = UltrasonicConfig(port="/dev/ttyACM-usonic", **(cfg or {}))
        self.driver = JsnSr04tMcuDriver(
            self.cfg,
            serial_factory=self.factory,
            clock=self.clock.now,
            wall_clock=self.clock.wall,
            sleep=self.clock.sleep,
            simulate=False,
            **driver_kw,
        )

    async def boot(self) -> Rig:
        await self.driver.initialize()
        return self

    def step(self, seconds: float, dt: float = 0.02, each: Callable[[], None] | None = None):
        for _ in range(round(seconds / dt)):
            self.clock.advance(dt)
            self.driver.poll()
            if each:
                each()

    def ch(self, name: str):
        return {c.sensor_id: c for c in self.driver.snapshot()}[name]

    @property
    def port(self):
        return self.factory.current


async def booted(**kw) -> Rig:
    rig = await Rig(**kw).boot()
    rig.step(0.6)  # one full measurement cycle, with margin
    return rig


# ---- the arithmetic ---------------------------------------------------------------------------


def test_echo_to_distance_uses_round_trip_time_and_the_configured_speed_of_sound():
    assert echo_to_cm(5828, 343.2) == pytest.approx(100.0, abs=0.01)
    assert echo_to_cm(5828, 331.3) == pytest.approx(96.54, abs=0.01)  # colder air, shorter path
    for cm in (25.0, 30.0, 99.9, 399.0):
        assert echo_to_cm(cm_to_echo_us(cm, 343.2), 343.2) == pytest.approx(cm, abs=0.02)


# ---- healthy operation ------------------------------------------------------------------------


async def test_four_channels_report_the_distance_each_sensor_really_sees():
    rig = await booted()
    assert rig.driver.link_state is LinkState.ACTIVE and rig.driver.link_ok()
    for name, cm in DISTANCES.items():
        c = rig.ch(name)
        assert c.status is ChannelStatus.OK and c.valid
        assert c.distance_cm == pytest.approx(cm, abs=0.1)
        assert c.verdict is Verdict.CLEAR and c.clear_to_cm == c.distance_cm
        assert c.health is Health.OK and c.alive
    assert rig.driver.get_minimum_distance() == pytest.approx(60.0, abs=0.1)
    assert rig.driver.channel_ids == NAMES


async def test_every_channel_is_refreshed_every_cycle_and_ages_stay_bounded():
    rig = await booted()
    worst = 0

    def watch():
        nonlocal worst
        worst = max(worst, max(c.age_ms for c in rig.driver.snapshot()))

    rig.step(10.0, each=watch)
    assert worst < 400, f"a channel went {worst} ms without a fresh measurement"


async def test_timestamp_is_the_measurement_time_not_the_time_of_the_read():
    rig = await booted()
    before = rig.ch("inner_left")
    rig.clock.advance(0.3)  # no poll: nothing new arrived
    after = rig.ch("inner_left")
    assert after.timestamp == before.timestamp
    assert after.age_ms >= before.age_ms + 299
    rig.step(0.4)
    assert rig.ch("inner_left").timestamp > before.timestamp


async def test_obstacle_threshold_boundaries():
    models = [echo(30.0), echo(30.1), echo(29.9), echo(120.0)]
    rig = await booted(sensors=models)
    assert rig.ch("outer_left").verdict is Verdict.OBSTACLE  # exactly at the limit stops
    assert rig.ch("inner_left").verdict is Verdict.CLEAR  # a hair beyond does not
    assert rig.ch("inner_right").verdict is Verdict.OBSTACLE
    agg = rig.driver.obstacle_verdict()
    assert agg["state"] == "obstacle" and agg["min_cm"] == pytest.approx(29.9, abs=0.1)
    assert rig.driver.obstacle_verdict(threshold_cm=10.0)["state"] == "clear"


async def test_blind_zone_is_unknown_never_a_distance_and_never_clear():
    rig = await booted(sensors=[echo(15.0), echo(24.9), echo(100.0), echo(120.0)])
    for name in ("outer_left", "inner_left"):
        c = rig.ch(name)
        assert c.status is ChannelStatus.BLIND_ZONE
        assert not c.valid and c.distance_cm == 0.0 and c.verdict is Verdict.UNKNOWN
        assert c.clear_to_cm is None
    agg = rig.driver.obstacle_verdict()
    assert agg["state"] == "unknown"  # two clear channels do not vouch for the blind ones
    assert agg["reasons"] == ["outer_left:blind_zone", "inner_left:blind_zone"]
    assert rig.driver.get_minimum_distance() == pytest.approx(100.0, abs=0.1)


async def test_min_range_is_inclusive_a_reading_exactly_at_the_blind_zone_edge_counts():
    rig = await booted(sensors=[echo(25.0), echo(100.0), echo(100.0), echo(100.0)])
    c = rig.ch("outer_left")
    assert c.status is ChannelStatus.OK and c.valid and c.verdict is Verdict.OBSTACLE


async def test_echo_beyond_max_range_is_out_of_range_and_unknown_by_default():
    rig = await booted(sensors=[echo(450.0)] * 4, cfg={"echo_window_ms": 40})
    c = rig.ch("outer_left")
    assert (c.status, c.valid, c.verdict) == (ChannelStatus.OUT_OF_RANGE, False, Verdict.UNKNOWN)
    assert rig.driver.obstacle_verdict()["state"] == "unknown"


# ---- the owner's policy flags (all fail closed by default) ------------------------------------


async def test_out_of_range_reads_clear_only_when_the_owner_says_so():
    cfg = {"echo_window_ms": 40, "out_of_range_is_clear": True}
    rig = await booted(sensors=[echo(450.0)] * 4, cfg=cfg)
    c = rig.ch("outer_left")
    assert c.status is ChannelStatus.OUT_OF_RANGE and not c.valid
    assert c.verdict is Verdict.CLEAR and c.clear_to_cm == 400.0
    assert rig.driver.obstacle_verdict()["state"] == "clear"


async def test_a_silent_channel_is_unknown_under_the_shipped_policy():
    models = four_clear()
    models[1] = SensorModel(mode="silent")
    rig = await booted(sensors=models)
    c = rig.ch("inner_left")
    assert (c.status, c.valid, c.verdict) == (ChannelStatus.NO_ECHO, False, Verdict.UNKNOWN)
    assert rig.driver.obstacle_verdict()["state"] == "unknown"  # three clear channels do not vouch


async def test_no_echo_clear_policy_never_clears_a_sensor_that_was_never_alive():
    models = four_clear()
    models[2] = SensorModel(mode="silent")  # dead from the first measurement
    rig = await booted(sensors=models, cfg={"no_echo_is_clear": True})
    c = rig.ch("inner_right")
    assert c.status is ChannelStatus.NO_ECHO and not c.alive
    assert c.verdict is Verdict.UNKNOWN, "a dead sensor must not read clear"
    assert rig.driver.obstacle_verdict()["state"] == "unknown"


async def test_no_echo_clear_policy_expires_when_a_live_sensor_dies():
    rig = await booted(
        sensors=[echo(80.0), echo(120.0), echo(200.0), echo(60.0)],
        cfg={"no_echo_is_clear": True, "alive_window_s": 5.0},
    )
    rig.port.set_sensor(2, SensorModel(mode="silent"))  # the sensor dies (or the path opens up)
    rig.step(1.0)
    c = rig.ch("inner_right")
    assert c.status is ChannelStatus.NO_ECHO and c.alive and c.verdict is Verdict.CLEAR
    assert c.clear_to_cm == 400.0
    rig.step(5.0)  # alive window (5 s) elapsed with no pulse at all
    c = rig.ch("inner_right")
    assert c.status is ChannelStatus.NO_ECHO and not c.alive
    assert c.verdict is Verdict.UNKNOWN and c.health is Health.SILENT


async def test_no_echo_clear_without_the_liveness_requirement_is_the_owners_explicit_risk():
    models = four_clear()
    models[2] = SensorModel(mode="silent")
    cfg = {"no_echo_is_clear": True, "no_echo_requires_alive": False}
    rig = await booted(sensors=models, cfg=cfg)
    assert rig.ch("inner_right").verdict is Verdict.CLEAR
    assert rig.driver.obstacle_verdict()["state"] == "clear"


async def test_a_fault_is_never_cleared_by_any_policy_flag():
    models = four_clear()
    models[0] = SensorModel(mode="stuck")
    cfg = {"no_echo_is_clear": True, "no_echo_requires_alive": False, "out_of_range_is_clear": True}
    rig = await booted(sensors=models, cfg=cfg)
    c = rig.ch("outer_left")
    assert (c.status, c.health, c.verdict) == (ChannelStatus.FAULT, Health.FAULT, Verdict.UNKNOWN)
    assert rig.driver.obstacle_verdict()["state"] == "unknown"


async def test_no_fall_pulse_alone_is_out_of_range_and_unknown_even_if_out_of_range_is_trusted():
    # US-10: a pulse that never ends is what a shorted ECHO line looks like. Nothing else
    # vouches for this sensor, so even the owner's out_of_range_is_clear flag cannot clear it
    # (the case where it can is test_a_long_echo_from_a_channel_that_also_hears_in_range_echoes...).
    models = four_clear()
    models[3] = SensorModel(mode="no_fall")
    rig = await booted(sensors=models)
    assert rig.ch("outer_right").status is ChannelStatus.OUT_OF_RANGE
    assert rig.ch("outer_right").verdict is Verdict.UNKNOWN
    rig2 = await booted(sensors=models, cfg={"out_of_range_is_clear": True})
    assert rig2.ch("outer_right").status is ChannelStatus.OUT_OF_RANGE
    assert rig2.ch("outer_right").verdict is Verdict.UNKNOWN


# ---- a dead sensor among live ones ------------------------------------------------------------


async def test_a_dead_sensor_is_visible_and_does_not_disturb_the_others():
    models = four_clear()
    models[2] = SensorModel(mode="silent")
    rig = await booted(sensors=models)
    ages = []
    rig.step(8.0, each=lambda: ages.append(rig.ch("outer_left").age_ms))
    dead = rig.ch("inner_right")
    assert dead.status is ChannelStatus.NO_ECHO and dead.fails >= 8
    assert dead.health is Health.SILENT and not dead.alive and not dead.valid
    assert dead.distance_cm == 0.0 and dead.verdict is Verdict.UNKNOWN
    for name in ("outer_left", "inner_left", "outer_right"):
        c = rig.ch(name)
        assert c.valid and c.distance_cm == pytest.approx(DISTANCES[name], abs=0.1)
    assert max(ages) < 400  # live sensors keep their cadence
    assert rig.driver.obstacle_verdict()["state"] == "unknown"
    assert rig.driver.obstacle_verdict()["reasons"] == ["inner_right:no_echo"]


async def test_a_sensor_that_recovers_reads_valid_again_within_a_cycle():
    models = four_clear()
    models[2] = SensorModel(mode="silent")
    rig = await booted(sensors=models)
    rig.step(3.0)
    assert not rig.ch("inner_right").valid
    rig.port.set_sensor(2, echo(90.0))
    rig.step(0.5)
    c = rig.ch("inner_right")
    assert c.valid and c.distance_cm == pytest.approx(90.0, abs=0.1) and c.fails == 0
    assert rig.driver.obstacle_verdict()["state"] == "clear"


async def test_disabled_channels_are_not_fitted_and_excluded_from_the_aggregate():
    cfg = {"disabled_channels": ["inner_right"]}
    models = four_clear()
    models[2] = SensorModel(mode="silent")  # nothing is plugged in there
    rig = await booted(sensors=models, cfg=cfg)
    assert rig.driver.expected_mcu_config().enable_mask == 0b1011
    c = rig.ch("inner_right")
    assert (c.status, c.verdict, c.health) == (
        ChannelStatus.DISABLED, Verdict.DISABLED, Health.DISABLED,
    )  # fmt: skip
    agg = rig.driver.obstacle_verdict()
    assert agg["state"] == "clear" and agg["reasons"] == []
    assert [x["sensor_id"] for x in agg["channels"]] == NAMES  # still listed


async def test_a_channel_the_mcu_disabled_but_the_host_expects_is_a_fault_until_repaired():
    rig = await booted()
    rig.port.core.enable_mask = 0b1011  # the MCU masks a sensor the host believes is fitted
    seen_fault = False
    for _ in range(100):  # the next INFO heartbeat (at most 1 s away) reveals the drift
        rig.step(0.02)
        if rig.driver.stats.config_drifts:
            break
        c = rig.ch("inner_right")
        if c.status is ChannelStatus.FAULT:  # the MCU reports it disabled: never "excluded"
            seen_fault = True
            assert c.verdict is Verdict.UNKNOWN
            assert rig.driver.obstacle_verdict()["state"] == "unknown"
    assert seen_fault and rig.driver.stats.config_drifts == 1
    rig.step(1.0)  # the host re-sent CONFIG: the MCU runs the configuration that was verified
    assert rig.port.core.enable_mask == 0b1111 and rig.driver.link_state is LinkState.ACTIVE
    assert rig.ch("inner_right").valid


# ---- handshake and refusal --------------------------------------------------------------------


class ScriptedPort:
    """A port driven by the test: push exact MCU->host frames, inspect host->MCU frames."""

    def __init__(self):
        self.inbox = bytearray()
        self.decoder = proto.FrameDecoder()
        self.sent: list[proto.Frame] = []
        self.closed = False
        self.mcu_ms = 100  # the MCU clock of auto-stamped readings: every one is newer
        self.broken: Exception | None = None  # set to make the port fail like an unplugged cable

    @property
    def in_waiting(self):
        if self.broken is not None:
            raise self.broken
        return len(self.inbox)

    def read(self, size=1):
        data = bytes(self.inbox[:size])
        del self.inbox[:size]
        return data

    def write(self, data):
        self.sent += self.decoder.feed(data)
        return len(data)

    def close(self):
        self.closed = True

    def push(self, ftype, payload=b"", seq=0, version=proto.PROTO_VERSION):
        self.inbox += proto.encode_frame(ftype, seq, payload, version)

    def info(self, **kw):
        args = dict(
            fw_major=1, fw_minor=0, sensor_count=4, boot_id=7, uptime_ms=10, reset=1, mask=15
        )
        args.update(kw)
        proto_version = args.pop("proto", proto.PROTO_VERSION)
        payload = bytearray(
            fw.pack_info(args["fw_major"], args["fw_minor"], args["sensor_count"], args["boot_id"],
                         args["uptime_ms"], args["reset"], args["mask"])
        )  # fmt: skip
        payload[2] = proto_version
        self.push(proto.FrameType.INFO, bytes(payload))

    def reading(self, sensor=0, status=0, echo_us=5828, health=0, fails=0, t_ms=None):
        if t_ms is None:
            self.mcu_ms += 10
            t_ms = self.mcu_ms
        self.push(
            proto.FrameType.READING,
            fw.pack_reading(sensor, status, health, fails, echo_us, 1, t_ms),
        )

    def ack(self, cfg: proto.McuConfig):
        self.push(proto.FrameType.CONFIG_ACK, proto.build_config(cfg))

    def types(self):
        return [f.ftype for f in self.sent]


def scripted_rig(**cfg):
    clock = FakeClock()
    port = ScriptedPort()
    opens = []

    def factory(path, baud):
        opens.append((path, baud))
        return port

    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-usonic", **cfg),
        serial_factory=factory, clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=False,
    )  # fmt: skip
    return clock, port, driver, opens


def scripted_active(**cfg):
    """A scripted rig whose handshake is complete (INFO, CONFIG, CONFIG_ACK): readings count."""
    clock, port, driver, opens = scripted_rig(**cfg)
    driver.poll()
    port.info()
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE
    return clock, port, driver, opens


MCU_BASE_MS = 500_000  # the scripted MCU booted 500 s before the fake host clock's origin


def mcu_ms(clock: FakeClock, behind_s: float = 0.0) -> int:
    """The MCU clock (ms) ``behind_s`` seconds before the host's 'now', with no transport delay."""
    return MCU_BASE_MS + round((clock.now() - 1000.0 - behind_s) * 1000)


def steady(clock, port, driver, seconds=1.0, sensors=(0, 1, 2, 3)):
    """A healthy stream: every 100 ms each of ``sensors`` reports, stamped with the MCU clock."""
    for _ in range(round(seconds / 0.1)):
        clock.advance(0.1)
        for sensor in sensors:
            port.reading(sensor=sensor, t_ms=mcu_ms(clock))
        driver.poll()


def test_handshake_order_and_readings_before_the_config_echo_are_not_trusted():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    # listen first: the MCU announces itself, so the host writes nothing to a port it has not heard
    assert port.types() == [] and driver.link_state is LinkState.HANDSHAKE
    port.info()
    port.reading(sensor=0)  # arrives before the CONFIG echo: the MCU may still be on old settings
    driver.poll()
    assert port.types() == [proto.FrameType.CONFIG]
    assert driver.link_state is LinkState.HANDSHAKE
    assert driver.stats.readings_ignored == 1
    assert driver.snapshot()[0].status is ChannelStatus.LINK_DOWN
    wanted = driver.expected_mcu_config()
    assert proto.McuConfig(*struct.unpack("<BHHB", port.sent[0].payload)) == wanted
    port.ack(wanted)
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE
    port.reading(sensor=0)
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[0].valid and driver.stats.readings == 1


def test_config_echo_that_differs_from_what_was_asked_is_a_fault():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info()
    driver.poll()
    wanted = driver.expected_mcu_config()
    port.ack(proto.McuConfig(wanted.enable_mask, wanted.echo_window_ms + 1, wanted.gap_ms, 8))
    driver.poll()
    assert driver.link_state is LinkState.FAULT and "config_mismatch" in driver.link_fault
    port.reading()
    driver.poll()
    assert all(c.status is ChannelStatus.LINK_DOWN for c in driver.snapshot())
    assert not driver.link_ok()


def test_a_nak_to_the_config_is_a_fault_with_the_reason():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info()
    driver.poll()
    port.push(
        proto.FrameType.NAK, fw.pack_nak(int(proto.FrameType.CONFIG), proto.NakReason.BAD_VALUE)
    )
    driver.poll()
    assert driver.link_state is LinkState.FAULT
    assert driver.link_fault == "config_rejected: bad_value"
    assert driver.stats.naks == 1


@pytest.mark.parametrize("how", ["info", "frame"])
def test_a_different_protocol_version_is_refused_and_no_data_is_accepted(how):
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    if how == "info":
        port.info(proto=2)
    else:
        port.push(proto.FrameType.INFO, fw.pack_info(1, 0, 4, 7, 10, 1, 15), version=2)
    port.reading()
    driver.poll()
    assert driver.link_state is LinkState.FAULT
    assert "protocol_version_mismatch" in driver.link_fault
    assert all(not c.valid and c.verdict is Verdict.UNKNOWN for c in driver.snapshot())


async def test_mcu_with_the_wrong_number_of_sensors_is_refused():
    for n in (3, 0):
        rig = await Rig(factory_kwargs={"n": n, "sensors": [echo(100.0)] * n}).boot()
        rig.step(1.0)
        assert rig.driver.link_state is LinkState.FAULT
        assert rig.driver.link_fault.startswith("channel_count_mismatch")
        assert not rig.driver.link_ok()
        assert {c.status for c in rig.driver.snapshot()} == {ChannelStatus.LINK_DOWN}
        assert rig.driver.obstacle_verdict()["state"] == "unknown"


async def test_no_port_configured_is_a_reported_fault_and_never_opens_anything():
    clock = FakeClock()
    calls = []
    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(),  # port defaults to None: the owner has not said where the MCU is
        serial_factory=lambda p, b: calls.append((p, b)),
        clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep, simulate=False,
    )  # fmt: skip
    await driver.initialize()
    assert calls == []
    assert driver.link_state is LinkState.FAULT and driver.link_fault == "no_port_configured"
    assert driver.obstacle_verdict()["state"] == "unknown"


def test_open_failures_are_retried_with_growing_backoff_and_then_succeed():
    clock = FakeClock()
    factory = FakeMcuFactory(clock, sensors=four_clear())
    factory.fail_opens = 3
    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-usonic"),
        serial_factory=factory, clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=False,
    )  # fmt: skip
    t0 = clock.now()
    attempts: list[float] = []
    for _ in range(400):
        seen = len(factory.opens)
        driver.poll()
        if len(factory.opens) > seen:
            attempts.append(round(clock.now() - t0, 2))
        if driver.link_state is LinkState.ACTIVE:
            break
        clock.advance(0.02)
    assert driver.link_state is LinkState.ACTIVE
    assert len(attempts) == 4 and driver.stats.connect_failures == 3
    gaps = [round(b - a, 2) for a, b in zip(attempts, attempts[1:], strict=False)]
    assert gaps[0] < gaps[1] < gaps[2], f"backoff must grow, got attempts at {attempts}"
    assert driver.link_fault is None  # the failure reason is cleared once connected


def test_an_mcu_that_never_answers_times_out_and_the_port_is_reopened():
    clock, port, driver, opens = scripted_rig(handshake_timeout_s=2.0)
    driver.poll()
    for _ in range(80):
        clock.advance(0.02)
        driver.poll()
    assert not port.closed, "1.6 s is inside the 2.0 s handshake window"
    for _ in range(45):
        clock.advance(0.02)
        driver.poll()
    assert port.closed and driver.stats.link_losses == 1
    assert driver.last_error == "handshake_timeout"
    assert len(opens) == 2  # tried again after the 0.25 s backoff


# ---- corruption on a live link ----------------------------------------------------------------


async def test_corrupted_frames_are_rejected_counted_and_never_applied():
    rig = await booted()
    lost0, crc0 = rig.driver.stats.frames_lost, rig.driver.stats.crc_errors
    rig.port.corrupt_next(
        3, flip_payload_bit
    )  # flips the sensor-index bit: would hit a wrong channel
    rig.step(3.0)
    s = rig.driver.stats
    assert s.crc_errors - crc0 == 3 and s.frames_lost - lost0 == 3
    assert rig.driver.link_state is LinkState.ACTIVE
    for name, cm in DISTANCES.items():
        assert rig.ch(name).distance_cm == pytest.approx(cm, abs=0.1)  # nothing was misattributed


@pytest.mark.parametrize(
    "fault",
    [flip_crc_bit, truncate, with_garbage, corrupt_length, drop],
    ids=["crc_bit", "truncated", "garbage_prefix", "bogus_length", "dropped"],
)
async def test_every_kind_of_frame_damage_costs_one_frame_and_the_stream_recovers(fault):
    rig = await booted()
    rig.port.corrupt_next(2, fault)
    rig.step(3.0)
    assert rig.driver.link_state is LinkState.ACTIVE
    assert all(c.valid for c in rig.driver.snapshot())
    assert max(c.age_ms for c in rig.driver.snapshot()) < 400


async def test_random_corruption_never_turns_into_a_wrong_accepted_reading():
    rig = await booted()
    rng = random.Random(7)

    def damage(frame: bytes) -> bytes | None:
        if rng.random() >= 0.2:
            return frame
        op = rng.choice(["flip", "flip", "truncate", "drop", "garbage", "length"])
        b = bytearray(frame)
        if op == "flip":
            for _ in range(rng.randint(1, 3)):
                b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
            return bytes(b)
        if op == "truncate":
            return bytes(b[: rng.randrange(1, len(b))])
        if op == "drop":
            return None
        if op == "garbage":
            return bytes(rng.randbytes(rng.randint(1, 6))) + frame
        b[5] = rng.randrange(0, 200)
        return bytes(b)

    rig.port.mutators.append(damage)
    rig.port.max_read = 7  # also arrive in awkward pieces
    wrong = []

    def check():
        for c in rig.driver.snapshot():
            if c.valid and abs(c.distance_cm - DISTANCES[c.sensor_id]) > 0.1:
                wrong.append((c.sensor_id, c.distance_cm))

    rig.step(40.0, each=check)
    s = rig.driver.stats
    assert s.crc_errors > 20 and s.frames_lost > 20, "the injection must really have happened"
    assert wrong == [], f"corruption leaked into accepted readings: {wrong[:5]}"
    assert rig.driver.link_state is LinkState.ACTIVE
    assert all(c.valid for c in rig.driver.snapshot())


async def test_garbage_between_frames_is_counted_and_ignored():
    rig = await booted()
    g0 = rig.driver.stats.garbage_bytes
    for _ in range(10):
        rig.port.inject(b"\x00\xff\xa5\x13")
        rig.step(0.1)
    assert rig.driver.stats.garbage_bytes > g0
    assert all(c.valid for c in rig.driver.snapshot())


async def test_unparseable_payloads_and_impossible_readings_do_not_poison_channels():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    port.info()
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    port.push(proto.FrameType.READING, b"\x00\x01\x02")  # wrong payload length
    port.reading(sensor=9)  # a sensor the array does not have
    port.reading(sensor=1, status=77)  # unknown status code
    port.reading(sensor=2, status=proto.EchoStatus.OK, echo_us=0)  # "OK" with no echo
    port.reading(sensor=3, status=proto.EchoStatus.OK, echo_us=4_000_000_000)  # not physical
    clock.advance(0.02)
    driver.poll()
    s = driver.stats
    assert s.bad_payload == 1 and s.bad_readings == 2
    snap = {c.sensor_id: c for c in driver.snapshot()}
    assert snap["outer_left"].status is ChannelStatus.NO_DATA
    for name in ("inner_left", "inner_right", "outer_right"):
        assert snap[name].status is ChannelStatus.FAULT and not snap[name].valid
    assert all(c.verdict is Verdict.UNKNOWN for c in snap.values())


# ---- silence, loss and recovery ---------------------------------------------------------------


async def test_a_hung_mcu_goes_stale_then_link_down_and_is_never_clear_in_between():
    rig = await booted()
    rig.port.freeze()
    t0 = rig.clock.now()
    seen = {}

    def watch():
        dt = rig.clock.now() - t0
        agg = rig.driver.obstacle_verdict()
        if agg["state"] == "clear":
            seen["clear_until"] = dt  # keeps the LAST moment the aggregate still claimed clear
        statuses = {c["status"] for c in agg["channels"]}
        if statuses == {"stale"}:
            seen.setdefault("stale_at", dt)
        if statuses == {"link_down"}:
            seen.setdefault("link_down_at", dt)

    rig.step(2.0, each=watch)
    assert 0.66 <= seen["stale_at"] < 1.1 and 1.45 <= seen["link_down_at"] < 1.8
    assert seen["clear_until"] < 0.75, "stayed clear on data older than the stale limit"
    assert rig.driver.obstacle_verdict()["state"] == "unknown"
    assert all(not c.valid for c in rig.driver.snapshot())


async def test_a_silent_link_is_probed_with_hello_and_reopened_when_it_stays_dead():
    def hellos(port):
        return sum(
            proto.FrameDecoder().feed(w)[0].ftype == proto.FrameType.HELLO for w in port.written
        )

    rig = await booted()
    first = rig.port
    first.freeze()
    before = hellos(first)
    rig.step(5.0)
    assert hellos(first) - before >= 2, "no HELLO probes while the link was silent"
    assert first.closed and rig.driver.stats.link_losses == 1
    assert rig.driver.last_error == "silent"
    rig.step(2.0)
    assert len(rig.factory.ports) == 2 and rig.driver.link_state is LinkState.ACTIVE
    assert all(c.valid for c in rig.driver.snapshot())


async def test_unplugging_the_cable_is_detected_at_once_and_service_resumes_on_replug():
    rig = await booted()
    rig.port.unplug()
    rig.clock.advance(0.02)
    rig.driver.poll()
    assert rig.driver.link_state is LinkState.DISCONNECTED
    assert rig.driver.link_fault.startswith("read_error")
    assert all(c.status is ChannelStatus.LINK_DOWN and not c.valid for c in rig.driver.snapshot())
    assert rig.driver.obstacle_verdict()["state"] == "unknown"
    rig.step(2.0)  # the replugged MCU is a fresh port with a fresh boot
    assert len(rig.factory.ports) == 2 and rig.driver.link_state is LinkState.ACTIVE
    assert rig.driver.stats.link_losses == 1 and rig.driver.stats.mcu_resets == 0
    assert all(c.valid for c in rig.driver.snapshot())


async def test_replug_that_fails_to_open_keeps_retrying_and_stays_unknown():
    rig = await booted()
    rig.port.unplug()
    rig.factory.fail_opens = 4
    rig.step(1.0)
    assert rig.driver.link_state is LinkState.DISCONNECTED
    assert rig.driver.obstacle_verdict()["state"] == "unknown"
    rig.step(8.0)
    assert rig.driver.link_state is LinkState.ACTIVE


async def test_a_write_failure_is_a_link_loss_too():
    rig = await booted()
    first = rig.port
    first.freeze()  # the MCU stops talking, so the next thing the host does is a probe write

    def broken_write(_data):
        raise OSError("Input/output error")

    first.write = broken_write
    rig.step(2.0)  # silent for 1.5 s -> HELLO probe -> the write fails
    assert rig.driver.stats.link_losses == 1
    assert rig.driver.last_error == "write_error: Input/output error"
    assert first.closed
    rig.step(2.0)
    assert len(rig.factory.ports) == 2 and rig.driver.link_state is LinkState.ACTIVE


async def test_an_mcu_watchdog_reset_is_noticed_old_data_is_dropped_and_config_is_resent():
    rig = await booted()
    assert rig.driver.stats.mcu_resets == 0
    rig.port.reboot()
    rig.clock.advance(0.05)
    rig.driver.poll()  # sees the new boot_id, drops everything and re-sends CONFIG
    assert rig.driver.stats.mcu_resets == 1
    assert rig.driver.link_state is LinkState.HANDSHAKE
    assert all(not c.valid and c.status is ChannelStatus.LINK_DOWN for c in rig.driver.snapshot())
    rig.step(1.0)
    assert rig.driver.link_state is LinkState.ACTIVE
    assert rig.driver.info.reset_reason == fw.RESET_WATCHDOG
    assert all(c.valid for c in rig.driver.snapshot())
    assert rig.port.core.window_ms == rig.cfg.echo_window_ms  # the new boot got our settings


async def test_repeated_mcu_resets_are_counted_and_each_one_blanks_the_data():
    rig = await booted()
    for n in range(1, 4):
        rig.port.reboot()
        rig.step(0.05)
        assert rig.driver.stats.mcu_resets == n
        rig.step(0.8)
        assert rig.driver.link_state is LinkState.ACTIVE


# ---- simulation, lifecycle, reporting ---------------------------------------------------------


async def test_simulation_mode_produces_four_plausible_channels_without_any_port():
    clock = FakeClock()
    calls = []
    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(), serial_factory=lambda p, b: calls.append(p),
        clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=True, rng=random.Random(3),
    )  # fmt: skip
    await driver.initialize()
    for _ in range(100):
        clock.advance(0.02)
        driver.poll()
    snap = driver.snapshot()
    assert calls == [] and driver.link_ok()
    assert [c.sensor_id for c in snap] == NAMES
    assert all(c.valid and 20.0 < c.distance_cm < 260.0 and c.timestamp for c in snap)
    await driver.stop()
    clock.advance(2.0)  # nothing refreshes any more: it must age out rather than stay clear
    assert driver.obstacle_verdict()["state"] == "unknown"


async def test_background_service_loop_keeps_data_fresh_and_stop_releases_the_port():
    rig = await Rig().boot()

    async def yielding_sleep(dt):
        rig.clock.advance(dt)
        await asyncio.sleep(0)

    rig.driver._sleep = yielding_sleep
    await rig.driver.start()
    for _ in range(80):
        await asyncio.sleep(0)
    assert rig.driver.running and all(c.valid for c in rig.driver.snapshot())
    task = rig.driver._task
    await rig.driver.stop()
    assert task.done() and not rig.driver.running
    assert rig.port.closed and rig.driver.link_state is LinkState.DISCONNECTED


async def test_health_report_is_json_and_carries_the_diagnostics_an_operator_needs():
    rig = await booted()
    rig.port.corrupt_next(2, flip_crc_bit)
    rig.step(1.0)
    report = await rig.driver.health_check()
    json.dumps(report)
    assert report["link_state"] == "active" and report["link_ok"]
    assert report["mcu"]["sensor_count"] == 4 and report["mcu"]["proto"] == proto.PROTO_VERSION
    assert report["stats"]["crc_errors"] == 2
    assert set(report["channels"]) == set(NAMES)
    assert report["channels"]["outer_left"]["status"] == "ok"


# ---- US-01: an approached obstacle that goes silent is the dead zone, not open air ------------

OK, NO_EDGE, NO_FALL = fw.ST_OK, fw.ST_NO_EDGE, fw.ST_NO_FALL
WINDOW_US = 30_000  # echo window of the default configuration


def trace(rig: Rig, seconds: float, name: str = "outer_left"):
    """``(seconds since the call, channel reading, aggregate state)`` after every 20 ms poll."""
    t0, rows = rig.clock.now(), []
    rig.step(
        seconds,
        each=lambda: rows.append(
            (rig.clock.now() - t0, rig.ch(name), rig.driver.obstacle_verdict()["state"])
        ),
    )
    return rows


POLICIES = {
    "shipped": {},
    "silent_is_clear_if_alive": {"no_echo_is_clear": True},
    "silent_is_clear_always": {"no_echo_is_clear": True, "no_echo_requires_alive": False},
    "everything_trusted": {
        "no_echo_is_clear": True,
        "no_echo_requires_alive": False,
        "out_of_range_is_clear": True,
    },
}


@pytest.mark.parametrize("policy", list(POLICIES.values()), ids=list(POLICIES))
async def test_silence_right_after_a_near_echo_is_the_dead_zone_never_clear_whatever_the_policy(
    policy,
):
    cfg = {**policy, "near_hold_s": 6.0}
    sensors = [approach_then_silence(), echo(200), echo(200), echo(200)]
    rig = await Rig(sensors=sensors, cfg=cfg).boot()
    rig.step(2.2)  # the wall is approached; the last real echo is at 28 cm, then the line is silent
    assert rig.ch("outer_left").echo_us == 0, "scenario broken: the MCU should report silence now"
    for t, c, aggregate in trace(rig, 5.0):  # still inside the 6 s hold
        got = (c.status, c.verdict, c.valid, c.clear_to_cm)
        want = (ChannelStatus.BLIND_ZONE, Verdict.UNKNOWN, False, None)
        assert got == want, f"t={t:.2f}s: silence after a 28 cm echo read {got}"
        assert aggregate != "clear"


async def test_after_the_hold_silence_reads_clear_only_if_the_owner_trusts_silence():
    base = {"no_echo_is_clear": True, "near_hold_s": 6.0}

    def sensors():
        return [approach_then_silence(), echo(200), echo(200), echo(200)]

    trusting = await Rig(sensors=sensors(), cfg={**base, "no_echo_requires_alive": False}).boot()
    gated = await Rig(sensors=sensors(), cfg=base).boot()
    for rig in (trusting, gated):
        rig.step(2.2 + 6.0 + 0.5)  # the hold (6 s after the last echo) is over
    c = trusting.ch("outer_left")
    # the owner's explicit risk, but no_echo_is_clear is not useless once the hold has passed
    assert (c.status, c.verdict) == (ChannelStatus.NO_ECHO, Verdict.CLEAR)
    c = gated.ch("outer_left")
    # pulses of a NEAR echo are no proof that the sensor looks at open air: not alive, not clear
    assert (c.status, c.alive, c.verdict) == (ChannelStatus.NO_ECHO, False, Verdict.UNKNOWN)
    assert c.health is Health.SILENT


async def test_a_real_far_echo_ends_the_hold_at_once_and_open_air_can_read_clear_again():
    script = scripted(
        *[(OK, us(100))] * 2, *[(OK, us(28))] * 2, *[(NO_EDGE, 0)] * 3, (OK, us(150)), (NO_EDGE, 0)
    )
    sensors = [script, echo(200), echo(200), echo(200)]
    rig = await Rig(sensors=sensors, cfg={"no_echo_is_clear": True}).boot()
    rows = trace(rig, 4.0)
    silent = next(i for i, (_, c, _) in enumerate(rows) if c.echo_us == 0)
    far = next(i for i, (_, c, _) in enumerate(rows) if abs(c.distance_cm - 150.0) < 0.2)
    assert silent < far
    for t, c, aggregate in rows[silent:far]:  # latched until the far echo itself arrives
        assert (c.status, c.verdict) == (ChannelStatus.BLIND_ZONE, Verdict.UNKNOWN), f"t={t:.2f}"
        assert aggregate != "clear"
    c = rows[far][1]
    assert (c.status, c.valid, c.verdict) == (ChannelStatus.OK, True, Verdict.CLEAR)
    c = rows[-1][1]  # silent again, but the far echo was proof of life: the owner's policy applies
    assert (c.status, c.alive, c.verdict) == (ChannelStatus.NO_ECHO, True, Verdict.CLEAR)


# ---- US-10: only a real echo is proof of life --------------------------------------------------


async def test_a_line_stuck_high_is_neither_proof_of_life_nor_clear_if_out_of_range_is_trusted():
    models = four_clear()
    models[3] = SensorModel(mode="no_fall")  # ECHO shorted high == "pulse longer than the window"
    rig = await booted(sensors=models, cfg={"out_of_range_is_clear": True})
    rig.step(2.0)
    c = rig.ch("outer_right")
    assert (c.status, c.alive, c.health) == (ChannelStatus.OUT_OF_RANGE, False, Health.SILENT)
    assert c.verdict is Verdict.UNKNOWN and c.clear_to_cm is None
    assert rig.driver.obstacle_verdict()["state"] == "unknown"


async def test_no_fall_pulses_never_make_a_silent_channel_alive():
    models = four_clear()
    models[2] = scripted(*[(NO_FALL, WINDOW_US)] * 4, (NO_EDGE, 0))
    rig = await booted(sensors=models, cfg={"no_echo_is_clear": True})
    rig.step(2.0)
    c = rig.ch("inner_right")
    assert (c.status, c.alive, c.verdict) == (ChannelStatus.NO_ECHO, False, Verdict.UNKNOWN)


async def test_a_long_echo_from_a_channel_that_also_hears_in_range_echoes_follows_the_policy():
    models = four_clear()
    models[1] = scripted(*[(OK, us(200))] * 3, (NO_FALL, WINDOW_US))
    rig = await booted(sensors=models, cfg={"out_of_range_is_clear": True})
    rig.step(1.0)  # the three in-range echoes are played first, then the line never falls
    c = rig.ch("inner_left")
    assert (c.status, c.alive, c.verdict) == (ChannelStatus.OUT_OF_RANGE, True, Verdict.CLEAR)
    assert c.clear_to_cm == 400.0


async def test_a_zero_width_ok_pulse_is_a_fault_and_never_proof_of_life():
    models = four_clear()
    models[0] = scripted(*[(OK, 0)] * 4, (NO_EDGE, 0))
    rig = await Rig(sensors=models, cfg={"no_echo_is_clear": True}).boot()
    rows = trace(rig, 2.5)
    assert any(c.status is ChannelStatus.FAULT for _, c, _ in rows), "the 0 us pulses never arrived"
    assert not any(c.alive for _, c, _ in rows), "a 0 us pulse is not an echo"
    c = rows[-1][1]
    assert (c.status, c.alive, c.verdict) == (ChannelStatus.NO_ECHO, False, Verdict.UNKNOWN)


async def test_the_max_range_boundary_is_inclusive_and_a_hair_beyond_is_out_of_range():
    models = [scripted((OK, us(400.0))), scripted((OK, us(400.1))), echo(200), echo(200)]
    rig = await booted(sensors=models)
    at, beyond = rig.ch("outer_left"), rig.ch("inner_left")
    assert (at.status, at.valid, at.distance_cm, at.verdict) == (
        ChannelStatus.OK, True, 400.0, Verdict.CLEAR,
    )  # fmt: skip
    assert at.clear_to_cm == 400.0
    assert (beyond.status, beyond.valid, beyond.distance_cm, beyond.verdict) == (
        ChannelStatus.OUT_OF_RANGE, False, 0.0, Verdict.UNKNOWN,
    )  # fmt: skip


# ---- D25: nothing fitted is never a clear path -------------------------------------------------


async def test_with_no_channel_fitted_the_aggregate_is_unknown_never_clear():
    rig = await booted(cfg={"disabled_channels": NAMES})
    agg = rig.driver.obstacle_verdict()
    assert rig.driver.link_state is LinkState.ACTIVE
    assert agg["state"] == "unknown" and agg["reasons"] == ["no fitted channels"]
    assert agg["min_cm"] is None
    assert all(c["verdict"] == "disabled" for c in agg["channels"])


# ---- US-11: a stopped or closed driver never reopens the port by itself ------------------------


async def test_a_stopped_driver_never_reopens_the_port_until_it_is_initialised_again():
    rig = await booted()
    await rig.driver.stop()
    assert rig.port.closed
    rig.step(2.0)  # a late caller keeps polling; this used to reopen a port nobody closes
    assert len(rig.factory.opens) == 1 and rig.driver.link_state is LinkState.DISCONNECTED
    await rig.driver.initialize()
    assert len(rig.factory.opens) == 2 and rig.driver.link_state is LinkState.ACTIVE


async def test_a_closed_driver_does_not_reopen_the_port_either():
    rig = await booted()
    rig.driver.close()  # the script path: poll-only users have no async lifecycle
    rig.step(2.0)
    assert len(rig.factory.opens) == 1 and rig.driver.link_state is LinkState.DISCONNECTED
    assert all(c.status is ChannelStatus.LINK_DOWN for c in rig.driver.snapshot())


# ---- US-05: a reading is as old as the MCU's own clock says, not "just received" ---------------


def stall(rig: Rig, seconds: float, *, target_at: float | None = None, target=None) -> None:
    """The host's loop does not run for ``seconds``; the MCU keeps measuring and talking."""
    for i in range(round(seconds / 0.02)):
        rig.clock.advance(0.02)
        if target_at is not None and i == round(target_at / 0.02):
            rig.port.set_sensor(0, target)  # the wall closes in on outer_left at +target_at
        rig.port.sync()


async def test_a_backlog_is_drained_in_one_poll_and_the_newest_frame_decides():
    # reviewer exp e4: 40 s loop stall, a tty that hands out 4096 bytes per read(), the target
    # steps from 200 to 28 cm at +20 s. Two polls used to report 200 cm "clear" with age 0.
    rig = await Rig(sensors=[echo(200)] * 4, factory_kwargs={"max_read": 4096}).boot()
    rig.step(0.6)
    stall(rig, 40.0, target_at=20.0, target=echo(28))
    assert len(rig.port._deliverable()) > 20_000  # ~22 kB queued, five reads' worth
    rig.driver.poll()
    c = rig.ch("outer_left")
    assert (c.status, c.distance_cm, c.verdict) == (ChannelStatus.OK, 28.0, Verdict.OBSTACLE)
    assert all(x.age_ms < 400 for x in rig.driver.snapshot())
    assert rig.driver.stats.readings > 500 and rig.driver.stats.readings_stale_mcu == 0
    assert rig.driver.obstacle_verdict()["state"] == "obstacle"


async def test_a_backlog_that_one_poll_cannot_drain_is_still_aged_by_the_mcu_clock():
    # 64-byte reads and at most MAX_DRAIN_READS reads per poll: ~4 kB per poll, ~6 polls to drain
    rig = await Rig(sensors=[echo(200)] * 4, factory_kwargs={"max_read": 64}).boot()
    rig.step(0.6)
    stall(rig, 40.0, target_at=20.0, target=echo(28))
    seen = []
    for _ in range(12):
        rig.driver.poll()
        seen.append(rig.ch("outer_left"))
    assert seen[0].status is ChannelStatus.STALE and not seen[0].valid
    assert seen[0].age_ms > 20_000, "the first poll holds data from ~29 s ago"
    # a valid reading is only ever the CURRENT state, never the old 200 cm
    assert all(c.distance_cm == 28.0 for c in seen if c.valid)
    assert seen[-1].valid and seen[-1].verdict is Verdict.OBSTACLE and seen[-1].age_ms < 400


def test_frames_drained_in_one_poll_are_dated_by_their_own_mcu_timestamps():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    assert driver.snapshot()[0].valid
    clock.advance(40.0)  # the event loop stalls; the MCU's next three frames queue up
    for behind in (39.9, 39.8, 39.7):
        port.reading(sensor=0, t_ms=mcu_ms(clock, behind_s=behind))
    driver.poll()
    c = driver.snapshot()[0]
    assert c.status is ChannelStatus.STALE and not c.valid and c.verdict is Verdict.UNKNOWN
    assert 39_500 < c.age_ms < 39_900
    assert 39.0 < (clock.wall() - c.timestamp).total_seconds() < 40.0
    assert c.as_dict()["measured_mono"] == pytest.approx(clock.now() - c.age_ms / 1000, abs=0.01)
    port.reading(sensor=0, t_ms=mcu_ms(clock))  # the newest frame of the backlog is fresh
    driver.poll()
    c = driver.snapshot()[0]
    assert c.valid and c.age_ms < 20
    assert c.as_dict()["measured_mono"] == pytest.approx(clock.now(), abs=0.01)


def test_a_reading_older_than_the_newest_by_more_than_a_cycle_is_rejected_and_counted():
    clock, port, driver, _ = scripted_active()  # default cycle: 4 * (30 + 30) = 240 ms
    steady(clock, port, driver, 1.0, sensors=(0,))  # only outer_left has reported
    n = driver.stats.readings
    port.reading(sensor=1, t_ms=mcu_ms(clock, behind_s=0.3))  # 60 ms beyond a cycle
    port.reading(sensor=1, t_ms=mcu_ms(clock, behind_s=5.0))
    driver.poll()
    assert driver.stats.readings_stale_mcu == 2 and driver.stats.readings == n
    assert driver.snapshot()[1].status is ChannelStatus.NO_DATA  # neither was applied
    port.reading(sensor=1, t_ms=mcu_ms(clock, behind_s=0.2))  # inside a cycle: slightly late
    driver.poll()
    c = driver.snapshot()[1]
    assert c.valid and 190 <= c.age_ms <= 215  # accepted, and dated by its own timestamp
    assert driver.stats.readings_stale_mcu == 2 and driver.stats.readings == n + 1


def test_a_duplicated_or_replayed_frame_on_a_channel_is_rejected_and_counted():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    n, rejected = driver.stats.readings, driver.stats.readings_stale_mcu
    port.reading(sensor=2, t_ms=mcu_ms(clock))  # the very same measurement again
    driver.poll()
    assert (driver.stats.readings, driver.stats.readings_stale_mcu) == (n, rejected + 1)
    port.reading(sensor=2, t_ms=mcu_ms(clock, behind_s=0.05))  # older than its last, within a cycle
    driver.poll()
    assert (driver.stats.readings, driver.stats.readings_stale_mcu) == (n, rejected + 2)
    seen = []
    driver.reading_hook = lambda r, wall, fate: seen.append((r.sensor, fate))
    port.reading(sensor=2, t_ms=mcu_ms(clock))
    driver.poll()
    assert seen == [(2, Disposition.REFUSED_CLOCK)], "the raw log still holds the refused frame"


def test_the_uint32_mcu_clock_wraps_without_breaking_freshness_or_ordering():
    clock, port, driver, _ = scripted_active()
    base = 2**32 - 3000  # the MCU is 3 s away from wrapping to 0

    def mcu(behind_s: float = 0.0) -> int:
        return (base + round((clock.now() - 1000.0 - behind_s) * 1000)) & 0xFFFFFFFF

    for _ in range(80):  # 3 s before the wrap and 5 s after it
        clock.advance(0.1)
        port.reading(sensor=0, t_ms=mcu())
        driver.poll()
        c = driver.snapshot()[0]
        assert c.valid and c.age_ms < 50, f"t={clock.now() - 1000:.1f}s"
    assert driver.stats.readings == 80 and driver.stats.readings_stale_mcu == 0
    port.reading(sensor=1, t_ms=mcu(behind_s=9.0))  # from before the wrap: old, not "future"
    driver.poll()
    assert driver.stats.readings_stale_mcu == 1
    assert driver.snapshot()[1].status is ChannelStatus.NO_DATA


def test_a_new_boot_id_resets_the_mcu_clock_baseline():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)  # the MCU clock reads ~501 s
    port.info(boot_id=8)  # the MCU rebooted: its clock restarts near zero
    clock.advance(0.1)
    driver.poll()
    assert driver.link_state is LinkState.HANDSHAKE and driver.stats.mcu_resets == 1
    port.ack(driver.expected_mcu_config())
    clock.advance(0.1)
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE
    for k in range(5):
        clock.advance(0.1)
        port.reading(sensor=0, t_ms=200 + 100 * k)
        driver.poll()
    c = driver.snapshot()[0]
    assert c.valid and c.age_ms < 50 and driver.stats.readings_stale_mcu == 0


def test_a_reconnect_resets_the_mcu_clock_baseline():
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 1.0)
    port.broken = OSError("device reports readiness to read but returned no data")
    clock.advance(0.1)
    driver.poll()  # unplugged
    assert driver.link_state is LinkState.DISCONNECTED
    port.broken = None
    port.inbox.clear()
    clock.advance(0.3)
    driver.poll()  # reconnects to a freshly booted MCU
    assert len(opens) == 2 and driver.link_state is LinkState.HANDSHAKE
    port.info(boot_id=9)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE
    clock.advance(0.1)
    port.reading(sensor=0, t_ms=150)  # a restarted clock
    driver.poll()
    assert driver.snapshot()[0].valid and driver.stats.readings_stale_mcu == 0


async def test_initialising_again_after_a_stop_accepts_a_restarted_mcu_clock():
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 1.0)  # the MCU clock reads ~501 s
    await driver.stop()
    port.inbox.clear()
    port.info(boot_id=9)  # a freshly booted MCU answers the new handshake at once
    port.ack(driver.expected_mcu_config())
    await driver.initialize()
    assert len(opens) == 2 and driver.link_state is LinkState.ACTIVE
    for k in range(3):
        clock.advance(0.1)
        for sensor in range(4):
            port.reading(sensor=sensor, t_ms=200 + 100 * k)  # its clock started over
        driver.poll()
    assert all(c.valid and c.age_ms < 50 for c in driver.snapshot())
    assert driver.stats.readings_stale_mcu == 0


def test_a_frame_split_across_two_polls_is_reassembled():
    clock, port, driver, _ = scripted_active()
    payload = fw.pack_reading(0, 0, 0, 0, 5828, 1, 5000)
    frame = proto.encode_frame(proto.FrameType.READING, 9, payload)
    port.inbox += frame[:9]
    clock.advance(0.02)
    driver.poll()
    assert driver.stats.readings == 0
    port.inbox += frame[9:]
    clock.advance(0.02)
    driver.poll()
    assert driver.stats.readings == 1 and driver.snapshot()[0].valid


@pytest.mark.parametrize("ppm", [-400, 400], ids=["mcu_clock_slow", "mcu_clock_fast"])
def test_clock_drift_within_the_allowance_never_makes_fresh_data_look_old(ppm):
    clock, port, driver, _ = scripted_active()
    mcu, worst = float(MCU_BASE_MS), 0
    for _ in range(3000):  # ten minutes of 200 ms polls; the two clocks disagree by 400 ppm
        clock.advance(0.2)
        mcu += 200.0 * (1 + ppm * 1e-6)
        port.reading(sensor=0, t_ms=round(mcu))
        driver.poll()
        c = driver.snapshot()[0]
        assert c.valid
        worst = max(worst, c.age_ms)
    assert worst < 50


# ---- US-06: the running MCU configuration is verified continuously -----------------------------


def test_an_ordinary_heartbeat_is_not_drift_but_another_enable_mask_drops_to_the_handshake():
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 0.5)
    port.info(mask=15)  # an ordinary heartbeat
    clock.advance(0.02)
    driver.poll()
    assert driver.stats.config_drifts == 0 and port.types().count(proto.FrameType.CONFIG) == 1
    assert driver.link_state is LinkState.ACTIVE
    port.info(mask=0b0111)  # someone masked outer_right off
    clock.advance(0.02)
    driver.poll()
    assert driver.stats.config_drifts == 1 and driver.link_state is LinkState.HANDSHAKE
    assert port.types().count(proto.FrameType.CONFIG) == 2  # CONFIG re-sent on the same port
    assert driver.link_fault.startswith("config_drift")
    assert all(c.status is ChannelStatus.LINK_DOWN for c in driver.snapshot())
    port.reading(sensor=0, t_ms=mcu_ms(clock))
    clock.advance(0.02)
    driver.poll()
    assert driver.stats.readings_ignored == 1  # not trusted until re-verified
    port.ack(driver.expected_mcu_config())
    clock.advance(0.02)
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE and driver.link_fault is None
    assert all(c.status is ChannelStatus.NO_DATA for c in driver.snapshot())  # nothing survived
    assert driver.stats.link_losses == 0 and len(opens) == 1  # and no reconnect storm


def test_a_config_ack_nobody_asked_for_is_drift_and_one_that_matches_is_ignored():
    clock, port, driver, opens = scripted_active()
    wanted = driver.expected_mcu_config()
    port.ack(wanted)  # a duplicate of our own acknowledgement
    clock.advance(0.02)
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE and driver.stats.config_drifts == 0
    port.ack(proto.McuConfig(wanted.enable_mask, 10, 5, wanted.quiet_after))  # another writer
    clock.advance(0.02)
    driver.poll()
    assert driver.stats.config_drifts == 1 and driver.link_state is LinkState.HANDSHAKE
    assert port.types().count(proto.FrameType.CONFIG) == 2
    resent = proto.McuConfig(*struct.unpack("<BHHB", port.sent[-1].payload))
    assert resent == wanted  # the host asks for ITS configuration again
    assert driver.stats.link_losses == 0 and len(opens) == 1


async def test_a_second_writer_that_reconfigures_the_mcu_is_noticed_and_undone():
    # reviewer exp e9: a probe / second SensorManager writes CONFIG(10 ms window, 5 ms gap)
    rig = await booted()
    other = proto.McuConfig(enable_mask=0b1111, echo_window_ms=10, gap_ms=5, quiet_after=8)
    rig.port.write(proto.encode_frame(proto.FrameType.CONFIG, 9, proto.build_config(other)))
    rig.step(1.5)
    assert rig.driver.stats.config_drifts == 1 and rig.driver.stats.link_losses == 0
    assert len(rig.factory.opens) == 1
    assert (rig.port.core.window_ms, rig.port.core.gap_ms) == (30, 30)  # restored
    assert rig.driver.link_state is LinkState.ACTIVE and all(c.valid for c in rig.driver.snapshot())


def test_the_default_transport_is_opened_exclusively(monkeypatch):
    import sys
    from types import SimpleNamespace

    from backend.src.drivers.sensors.jsn_sr04t_driver import open_pyserial

    seen: list[dict] = []

    class FakeSerial:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=FakeSerial))
    open_pyserial("/dev/ttyACM9", 115200)
    assert seen == [
        dict(port="/dev/ttyACM9", baudrate=115200, timeout=0, write_timeout=0.2, exclusive=True)
    ]


def test_a_second_opener_is_refused_and_reported_not_allowed_to_split_the_stream(monkeypatch):
    import sys
    from types import SimpleNamespace

    class SerialException(OSError):
        pass

    class Locked:
        def __init__(self, **kwargs):
            assert kwargs.get("exclusive") is True
            raise SerialException(11, "Could not exclusively lock port /dev/ttyACM9: [Errno 11]")

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=Locked))
    clock = FakeClock()
    driver = JsnSr04tMcuDriver(  # no serial_factory: the real default transport
        UltrasonicConfig(port="/dev/ttyACM9"),
        clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep, simulate=False,
    )  # fmt: skip
    driver.poll()
    assert driver.link_state is LinkState.DISCONNECTED and driver.stats.connect_failures == 1
    assert driver.last_error.startswith("open_failed:") and "exclusively lock" in driver.last_error
    assert driver.link_fault == driver.last_error
    assert all(c.status is ChannelStatus.LINK_DOWN for c in driver.snapshot())


def test_a_real_tty_cannot_be_opened_twice_by_the_default_transport():
    import os
    import pty

    serial = pytest.importorskip("serial")
    from backend.src.drivers.sensors.jsn_sr04t_driver import open_pyserial

    master, slave = pty.openpty()
    try:
        path = os.ttyname(slave)
        first = open_pyserial(path, 115200)
        with pytest.raises(serial.SerialException):
            open_pyserial(path, 115200)
        first.close()
        open_pyserial(path, 115200).close()  # released: the next owner can open it
    finally:
        os.close(master)
        os.close(slave)


# ---- D17 / D26 / D37: blanking on reboot and link loss, and the handshake deadline -------------


def test_an_mcu_reboot_blanks_every_channel_until_fresh_readings_arrive():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 0.5)
    assert all(c.valid for c in driver.snapshot())
    port.info(boot_id=8)  # the MCU rebooted
    clock.advance(0.02)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    clock.advance(0.02)
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE
    assert all(c.status is ChannelStatus.NO_DATA and c.timestamp is None for c in driver.snapshot())


def test_a_lost_link_blanks_every_channel_until_fresh_readings_arrive():
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 0.5)
    port.broken = OSError("device reports readiness to read but returned no data")
    clock.advance(0.1)
    driver.poll()
    port.broken = None
    port.inbox.clear()
    clock.advance(0.3)
    driver.poll()  # reconnect after the 0.25 s backoff, well inside the 700 ms stale limit
    port.info(boot_id=7)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE and len(opens) == 2
    assert all(c.status is ChannelStatus.NO_DATA for c in driver.snapshot())


def test_info_heartbeats_do_not_extend_the_handshake_deadline():
    clock, port, driver, _ = scripted_rig(handshake_timeout_s=2.0)
    driver.poll()  # port open, nothing written yet
    lost_at = None
    for i in range(1, 200):  # the MCU heartbeats every 0.4 s but never acknowledges CONFIG
        clock.advance(0.02)
        if i % 20 == 0:
            port.info()
        driver.poll()
        if lost_at is None and driver.stats.link_losses:
            lost_at = i * 0.02
    assert lost_at is not None and 2.0 <= lost_at <= 2.1
    assert driver.last_error == "handshake_timeout"
