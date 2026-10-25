"""The ultrasonic MCU firmware core (``firmware/ultrasonic-mcu``) against a simulated HAL.

The firmware's own hardware-independent logic is exercised here with the host codec reading its
output, so these tests also prove that the two ends speak the same protocol. What cannot be tested
here (real GPIO, PulseIn timing, USB enumeration) is listed in the firmware README.
"""

from __future__ import annotations

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as host
from tests.unit._usonic_fakes import (
    CORE,
    FakeClock,
    FakeMcuSerial,
    SensorModel,
    load_module,
)
from tests.unit._usonic_fakes import P as fw
from tests.unit._usonic_fw_fakes import Rebooted, RecordingHal

OK, NO_EDGE, NO_FALL, STUCK = fw.ST_OK, fw.ST_NO_EDGE, fw.ST_NO_FALL, fw.ST_STUCK


class Rig:
    """A firmware core behind a fake serial port, driven from the host side of the protocol."""

    def __init__(self, sensors=None, n=4, **kw):
        self.clock = FakeClock()
        self.port = FakeMcuSerial(self.clock, sensors=sensors, n=n, **kw)
        self.decoder = host.FrameDecoder()
        self.frames: list[host.Frame] = []
        self._tx = 0

    def run(self, seconds: float, step: float = 0.005) -> None:
        for _ in range(round(seconds / step)):
            self.clock.advance(step)
            self.port.sync()
            data = self.port.read(4096)
            if data:
                self.frames += self.decoder.feed(data, self.clock.now())

    def send(self, ftype, payload=b"", version=host.PROTO_VERSION) -> None:
        self.port.write(host.encode_frame(ftype, self._tx, payload, version))
        self._tx = (self._tx + 1) & 0xFF

    def of(self, ftype):
        return [host.parse_payload(f) for f in self.frames if f.ftype == ftype]

    def readings(self, sensor=None):
        out = self.of(host.FrameType.READING)
        return out if sensor is None else [r for r in out if r.sensor == sensor]


def echo(cm):
    return SensorModel(distance_cm=cm)


# A frame header whose LEN byte claims 40 payload bytes that never arrive (a corrupted length).
_BOGUS_HEADER = bytes((0xA5, 0x5A, host.PROTO_VERSION, host.FrameType.PING, 1, 40))


def _ping(nonce):
    return host.encode_frame(host.FrameType.PING, nonce & 0xFF, host.build_ping(nonce))


def _poll_until(hal, core, t_ms):
    """One ``poll()`` per simulated millisecond until the HAL clock reaches ``t_ms``."""
    while hal.t_ms < t_ms:
        core.poll()


# ---- protocol handling ------------------------------------------------------------------------


def test_boot_announces_itself_with_the_identity_the_host_needs():
    rig = Rig(boot_id=0xABCD1234)
    rig.run(0.05)
    info = rig.of(host.FrameType.INFO)[0]
    assert (info.fw_major, info.fw_minor) == (1, 0)
    assert info.proto_version == host.PROTO_VERSION
    assert info.sensor_count == 4
    assert info.boot_id == 0xABCD1234
    assert info.enable_mask == 0b1111
    assert info.reset_reason == fw.RESET_POWER_ON


def test_start_announces_the_boot_at_once_before_any_poll():
    # M02: without this, the first poll() would still send an INFO, so a test through a whole rig
    # cannot tell. The core itself must announce the boot (new boot_id, reset reason, sensor
    # count, enable mask) the moment it is started.
    hal = RecordingHal(n=4, start_ms=7000, boot_id=0xA1B2C3D4, reset_reason=fw.RESET_WATCHDOG)
    core = CORE.Core(hal, 4, fw_version=(2, 7), info_period_ms=1000)
    assert hal.tx == []  # building the core says nothing
    core.start()
    sent = hal.messages()
    assert len(sent) == 1, "start() must announce the boot itself, before the first poll()"
    ((_t, info),) = sent
    assert isinstance(info, host.Info)
    assert (info.boot_id, info.reset_reason) == (0xA1B2C3D4, fw.RESET_WATCHDOG)
    assert (info.sensor_count, info.enable_mask) == (4, 0b1111)
    assert (info.fw_major, info.fw_minor, info.proto_version) == (2, 7, host.PROTO_VERSION)
    assert info.uptime_ms == 0
    core.poll()  # and the heartbeat does not repeat it straight away
    assert len(hal.messages(host.FrameType.INFO)) == 1


@pytest.mark.parametrize("period_ms", [200, 700])
def test_info_heartbeat_period_is_honoured_not_more_often_and_not_less(period_ms):
    hal = RecordingHal(n=0, start_ms=5000)
    core = CORE.Core(hal, 0, info_period_ms=period_ms)
    core.start()
    for _ in range(period_ms * 5 + 10):  # one pass per millisecond
        core.poll()
    uptimes = [m.uptime_ms for _t, m in hal.messages(host.FrameType.INFO)]
    assert uptimes == [k * period_ms for k in range(6)]


def test_an_info_sent_for_a_hello_restarts_the_heartbeat_period():
    hal = RecordingHal(n=0, start_ms=5000)
    core = CORE.Core(hal, 0, info_period_ms=200)
    core.start()
    for _ in range(100):
        core.poll()
    hal.queue_rx(host.encode_frame(host.FrameType.HELLO, 0, host.build_hello()))
    for _ in range(500):
        core.poll()
    uptimes = [m.uptime_ms for _t, m in hal.messages(host.FrameType.INFO)]
    # boot, the answer to HELLO at 100, then a full period after THAT (not after the boot INFO)
    assert uptimes == [0, 100, 300, 500]


def test_hello_is_answered_with_info_and_ping_with_pong():
    rig = Rig()
    rig.run(0.05)
    before = len(rig.of(host.FrameType.INFO))
    rig.send(host.FrameType.HELLO, host.build_hello())
    rig.send(host.FrameType.PING, host.build_ping(0xFEEDF00D))
    rig.run(0.1)
    assert len(rig.of(host.FrameType.INFO)) == before + 1
    (pong,) = rig.of(host.FrameType.PONG)
    assert pong.nonce == 0xFEEDF00D


def test_pong_reports_the_time_since_boot_not_the_absolute_clock():
    hal = RecordingHal(n=0, start_ms=5000)
    core = CORE.Core(hal, 0, info_period_ms=100_000)
    core.start()
    _poll_until(hal, core, 5300)
    hal.queue_rx(_ping(0x0102))
    _poll_until(hal, core, 5320)
    ((_t, pong),) = hal.messages(host.FrameType.PONG)
    assert pong.nonce == 0x0102
    assert 300 <= pong.uptime_ms <= 305, f"uptime {pong.uptime_ms} ms: it is not the 5300 ms clock"


def test_a_core_polled_before_it_is_started_still_announces_itself():
    hal = RecordingHal(n=0, start_ms=500)
    core = CORE.Core(hal, 0)
    core.poll()
    assert len(hal.messages(host.FrameType.INFO)) == 1


def test_info_heartbeat_continues_even_with_every_sensor_disabled():
    rig = Rig(info_period_ms=1000)
    rig.run(0.05)
    rig.send(host.FrameType.CONFIG, host.build_config(host.McuConfig(0, 30, 30, 8)))
    rig.run(3.2)
    # boot + at least one per second, regardless of measurements
    assert len(rig.of(host.FrameType.INFO)) >= 4
    assert all(r.status == fw.ST_DISABLED for r in rig.readings()[-8:])


def test_zero_sensor_mcu_still_reports_and_never_fires():
    rig = Rig(sensors=[], n=0)
    rig.run(2.5)
    infos = rig.of(host.FrameType.INFO)
    assert infos and infos[0].sensor_count == 0 and infos[0].enable_mask == 0
    assert rig.readings() == []
    assert rig.port.hal.triggers == []


def test_config_is_applied_and_echoed_exactly():
    rig = Rig()
    rig.run(0.05)
    wanted = host.McuConfig(enable_mask=0b1101, echo_window_ms=40, gap_ms=55, quiet_after=3)
    rig.send(host.FrameType.CONFIG, host.build_config(wanted))
    rig.run(0.5)
    (ack,) = rig.of(host.FrameType.CONFIG_ACK)
    assert ack == wanted
    # the new window is really used by the measurements that follow
    assert set(rig.port.hal.windows[-6:]) == {40_000}


@pytest.mark.parametrize(
    "cfg",
    [
        host.McuConfig(0b1111, 9, 30, 8),  # window below the minimum
        host.McuConfig(0b1111, 101, 30, 8),  # window above the maximum
        host.McuConfig(0b1111, 30, 4, 8),  # gap below the minimum
        host.McuConfig(0b1111, 30, 501, 8),  # gap above the maximum
        host.McuConfig(0b1111, 30, 30, 0),  # quiet_after zero
        host.McuConfig(0b10000, 30, 30, 8),  # enables a sensor that does not exist
    ],
)
def test_out_of_range_config_is_refused_with_a_nak_and_changes_nothing(cfg):
    rig = Rig()
    rig.run(0.05)
    rig.send(host.FrameType.CONFIG, host.build_config(cfg))
    rig.run(0.3)
    (nak,) = rig.of(host.FrameType.NAK)
    assert (nak.ref_type, nak.reason) == (host.FrameType.CONFIG, host.NakReason.BAD_VALUE)
    assert rig.of(host.FrameType.CONFIG_ACK) == []
    assert rig.port.core.window_ms == 30 and rig.port.core.enable_mask == 0b1111


def test_wrong_version_unknown_type_and_wrong_length_are_each_refused():
    rig = Rig()
    rig.run(0.05)
    rig.send(host.FrameType.HELLO, host.build_hello(), version=2)
    rig.send(0x55, b"")
    rig.send(host.FrameType.CONFIG, b"\x01\x02")
    rig.run(0.2)
    reasons = [(n.ref_type, n.reason) for n in rig.of(host.FrameType.NAK)]
    assert (host.FrameType.HELLO, host.NakReason.BAD_VERSION) in reasons
    assert (0x55, host.NakReason.UNKNOWN_TYPE) in reasons
    assert (host.FrameType.CONFIG, host.NakReason.BAD_LENGTH) in reasons


def test_every_refusal_is_counted_and_answered_once():
    hal = RecordingHal(n=4, start_ms=1000)
    core = CORE.Core(hal, 4, info_period_ms=100_000)
    core.start()
    hal.queue_rx(host.encode_frame(host.FrameType.HELLO, 0, host.build_hello(), version=2))
    hal.queue_rx(host.encode_frame(0x7E, 1, b""))
    hal.queue_rx(host.encode_frame(host.FrameType.PING, 2, b"\x00"))  # a PING with a short payload
    for _ in range(20):
        core.poll()
    assert core.naks_sent == 3 and len(hal.messages(host.FrameType.NAK)) == 3


def test_a_looped_back_line_does_not_start_a_nak_storm():
    rig = Rig()
    rig.run(0.05)
    frames_before = rig.port.core.frames_tx
    echoed = rig.frames[-6:]
    assert len(echoed) >= 2
    # feed the MCU its own output (a loopback adapter / crossed wires), several times over
    for f in echoed * 3:
        rig.port.write(host.encode_frame(f.ftype, f.seq, f.payload))
    rig.run(0.2)
    assert rig.port.core.naks_sent == 0
    assert rig.port.core.ignored == len(echoed) * 3
    # it kept doing its job: only measurement/heartbeat frames were added
    assert rig.port.core.frames_tx > frames_before


def test_corrupt_and_truncated_host_frames_are_dropped_silently_and_the_next_one_works():
    rig = Rig()
    rig.run(0.05)
    good = host.encode_frame(host.FrameType.PING, 1, host.build_ping(7))
    flipped = bytearray(good)
    flipped[-1] ^= 0x01
    rig.port.write(bytes(flipped))
    rig.port.write(good[:5])  # truncated, then silence
    rig.run(0.2)  # longer than the partial-frame timeout
    rig.send(host.FrameType.PING, host.build_ping(8))
    rig.run(0.1)
    assert [p.nonce for p in rig.of(host.FrameType.PONG)] == [8]
    assert rig.port.core.naks_sent == 0


def test_a_half_received_host_frame_is_abandoned_after_the_partial_timeout():
    # C17: a corrupt LEN byte makes the core wait for payload that never comes. Once the line has
    # been quiet for `partial_timeout_ms` that remnant must go, so the next good frame is served
    # when it arrives instead of being held hostage behind it.
    hal = RecordingHal(n=0)
    core = CORE.Core(hal, 0, partial_timeout_ms=50)
    core.start()
    t0 = hal.t_ms
    hal.queue_rx(_BOGUS_HEADER, at_ms=t0)
    hal.queue_rx(_ping(7), at_ms=t0 + 100)
    _poll_until(hal, core, t0 + 110)
    pongs = hal.messages(host.FrameType.PONG)
    assert [p.nonce for _t, p in pongs] == [7]
    assert pongs[0][0] <= t0 + 102  # answered when it arrived
    assert core.dec.partial_timeouts == 1


@pytest.mark.parametrize("timeout_ms", [20, 50, 200])
def test_a_good_frame_hidden_behind_a_bogus_length_is_released_after_the_configured_timeout(
    timeout_ms,
):
    hal = RecordingHal(n=0)
    core = CORE.Core(hal, 0, partial_timeout_ms=timeout_ms)
    core.start()
    t0 = hal.t_ms
    hal.queue_rx(_BOGUS_HEADER + _ping(9), at_ms=t0)  # swallowed by the claimed 40-byte payload
    _poll_until(hal, core, t0 + timeout_ms + 20)
    pongs = hal.messages(host.FrameType.PONG)
    assert [p.nonce for _t, p in pongs] == [9]
    # released once the line has been quiet for MORE than the configured time, and not before
    assert t0 + timeout_ms < pongs[0][0] <= t0 + timeout_ms + 3


def test_a_frame_split_across_reads_within_the_timeout_is_still_assembled():
    hal = RecordingHal(n=0)
    core = CORE.Core(hal, 0, partial_timeout_ms=50)
    core.start()
    t0 = hal.t_ms
    frame = _ping(5)
    hal.queue_rx(frame[:5], at_ms=t0)
    hal.queue_rx(frame[5:], at_ms=t0 + 40)  # the line was quiet for less than the timeout
    _poll_until(hal, core, t0 + 60)
    assert [p.nonce for _t, p in hal.messages(host.FrameType.PONG)] == [5]
    assert core.dec.partial_timeouts == 0


def test_two_host_frames_in_one_write_are_both_served():
    rig = Rig()
    rig.run(0.05)
    blob = host.encode_frame(host.FrameType.PING, 1, host.build_ping(1)) + host.encode_frame(
        host.FrameType.PING, 2, host.build_ping(2)
    )
    rig.port.write(blob)
    rig.run(0.1)
    assert [p.nonce for p in rig.of(host.FrameType.PONG)] == [1, 2]


def test_each_boot_gets_its_own_boot_id_and_says_why_it_reset():
    rig = Rig(boot_id=100)
    rig.run(0.05)
    rig.port.reboot()
    rig.run(0.05)
    infos = rig.of(host.FrameType.INFO)
    assert infos[0].boot_id == 100 and infos[-1].boot_id == 101
    assert infos[-1].reset_reason == fw.RESET_WATCHDOG


# ---- time-division firing ---------------------------------------------------------------------


def _assert_time_division(triggers, window_ms, gap_ms):
    for (_, s0, e0), (_, s1, _e1) in zip(triggers, triggers[1:], strict=False):
        assert s1 >= e0, "two sensors measuring at once"
        assert s1 >= s0 + window_ms, "next trigger inside the previous sensor's echo window"
        assert s1 >= e0 + gap_ms, "next trigger before the quiet gap elapsed"


def test_sensors_fire_one_at_a_time_with_the_window_and_gap_between_triggers():
    mixed = [
        echo(35.0),
        SensorModel(mode="silent"),
        echo(399.0),
        SensorModel(mode="script", script=[(OK, 1500), (NO_FALL, 30_000), (NO_EDGE, 0)]),
    ]
    rig = Rig(sensors=mixed)
    rig.run(8.0)
    triggers = rig.port.hal.triggers
    assert len(triggers) > 100
    _assert_time_division(triggers, window_ms=30, gap_ms=30)


def test_time_division_also_holds_after_the_host_widens_window_and_gap():
    rig = Rig(sensors=[echo(100.0)] * 4)
    rig.run(0.05)
    rig.send(host.FrameType.CONFIG, host.build_config(host.McuConfig(0b1111, 60, 120, 8)))
    rig.run(0.1)
    start = len(rig.port.hal.triggers)
    rig.run(6.0)
    _assert_time_division(rig.port.hal.triggers[start:], window_ms=60, gap_ms=120)


@pytest.mark.parametrize(
    "window_ms,gap_ms,cm",
    [(30, 5, 35.0), (60, 5, 35.0), (100, 10, 35.0), (30, 5, 400.0)],
)
def test_a_gap_shorter_than_the_window_never_pulls_a_trigger_inside_the_previous_window(
    window_ms, gap_ms, cm
):
    # C04: the next trigger is bounded by BOTH `previous trigger + window` and `measurement end +
    # gap`. With the gap below the window, a close target ends its measurement so early that `end +
    # gap` falls INSIDE the window; only the window bound keeps one sensor from hearing the
    # burst of the next. (400 cm: `end + gap` lands just 1 ms short of the window end.)
    rig = Rig(sensors=[echo(cm)] * 4)
    rig.run(0.05)
    wanted = host.McuConfig(0b1111, window_ms, gap_ms, 8)
    rig.send(host.FrameType.CONFIG, host.build_config(wanted))
    rig.run(0.2)
    assert rig.of(host.FrameType.CONFIG_ACK) == [wanted]
    start = len(rig.port.hal.triggers)
    rig.run(4.0)
    triggers = rig.port.hal.triggers[start:]
    assert len(triggers) >= 30
    starts = [s for _i, s, _e in triggers]
    assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= window_ms
    _assert_time_division(triggers, window_ms, gap_ms)


def test_firing_follows_the_configured_order_every_cycle():
    rig = Rig(sensors=[echo(100.0)] * 4, firing_order=(1, 3, 0, 2))
    rig.run(3.0)
    fired = [t[0] for t in rig.port.hal.triggers]
    assert fired[:8] == [1, 3, 0, 2, 1, 3, 0, 2]


def test_firing_order_must_be_a_permutation():
    with pytest.raises(ValueError):
        CORE.Core(Rig().port.hal, 4, firing_order=(0, 1, 1, 3))
    with pytest.raises(ValueError):
        CORE.Core(Rig().port.hal, 4, firing_order=(0, 1, 2, 4))


def test_cycle_counter_groups_a_full_round_and_wraps_at_16_bits():
    rig = Rig(sensors=[echo(100.0)] * 4)
    rig.port.core.cycle = 0xFFFE
    rig.run(2.0)
    cycles = [r.cycle for r in rig.readings()]
    assert cycles[:4] == [0xFFFE] * 4
    assert cycles[4:8] == [0xFFFF] * 4
    assert cycles[8:12] == [0] * 4
    assert {r.sensor for r in rig.readings() if r.cycle == 0xFFFE} == {0, 1, 2, 3}


def test_reading_timestamp_is_the_end_of_the_measurement():
    rig = Rig(sensors=[echo(100.0)] * 4)
    rig.run(0.5)
    first_end = rig.port.hal.triggers[0][2]
    first = rig.readings()[0]
    assert first.t_ms == first_end
    assert first.echo_us == round(100.0 * 20000 / 343.2)


# ---- per-sensor health ------------------------------------------------------------------------


def test_a_disabled_channel_takes_its_slot_and_is_announced_once_per_gap_not_once_per_poll():
    # A masked-off channel is reported DISABLED once per cycle, but its slot lasts one gap like any
    # other: without that the loop would write a frame per pass (about 1000 a second) and starve
    # the USB link the real readings need.
    hal = RecordingHal(n=4, start_ms=1000)
    core = CORE.Core(hal, 4, gap_ms=30, echo_window_ms=30, info_period_ms=100_000)
    core.start()
    hal.queue_rx(
        host.encode_frame(host.FrameType.CONFIG, 0, host.build_config(host.McuConfig(0, 30, 30, 8)))
    )
    _poll_until(hal, core, 1100)
    before = len(hal.messages(host.FrameType.READING))
    _poll_until(hal, core, 3100)  # two seconds with every channel disabled
    readings = [m for _t, m in hal.messages(host.FrameType.READING)][before:]
    assert {r.status for r in readings} == {fw.ST_DISABLED}
    assert 2000 // 30 - 2 <= len(readings) <= 2000 // 30 + 2, f"{len(readings)} frames in 2 s"


def test_a_dead_sensor_never_slows_or_silences_the_others_and_is_reported_quiet():
    models = [echo(80.0), echo(120.0), SensorModel(mode="silent"), echo(60.0)]
    rig = Rig(sensors=models)
    rig.run(10.0)
    dead = rig.readings(2)
    assert all(r.status == NO_EDGE and r.echo_us == 0 for r in dead)
    assert [r.fails for r in dead[:10]] == list(range(1, 11))
    assert dead[0].health == host.McuHealth.ACTIVE
    assert dead[7].health == host.McuHealth.QUIET  # 8th consecutive silent measurement
    # the dead sensor is still fired every cycle (a quiet sensor is not skipped) ...
    counts = {s: len(rig.readings(s)) for s in range(4)}
    assert len(set(counts.values())) == 1 or max(counts.values()) - min(counts.values()) <= 1
    # ... and the live ones keep reporting every cycle with their real distances
    for s, cm in ((0, 80.0), (1, 120.0), (3, 60.0)):
        rs = rig.readings(s)
        assert len(rs) > 20 and all(r.status == OK and r.fails == 0 for r in rs)
        assert rs[-1].echo_us == round(cm * 20000 / 343.2)
    # worst-case cycle with one dead sensor stays well under a second
    t = [r.t_ms for r in rig.readings(0)]
    assert max(b - a for a, b in zip(t, t[1:], strict=False)) < 400


def test_a_sensor_that_comes_back_clears_its_failure_count_immediately():
    rig = Rig(sensors=[echo(100.0), SensorModel(mode="silent"), echo(100.0), echo(100.0)])
    rig.run(4.0)
    assert rig.readings(1)[-1].fails >= 8
    rig.port.set_sensor(1, echo(55.0))
    rig.run(1.0)
    last = rig.readings(1)[-1]
    assert (last.status, last.fails, last.health) == (OK, 0, host.McuHealth.ACTIVE)


def test_stuck_line_is_a_fault_not_a_quiet_sensor_and_keeps_being_probed():
    rig = Rig(sensors=[echo(100.0), SensorModel(mode="stuck"), echo(100.0), echo(100.0)])
    rig.run(4.0)
    stuck = rig.readings(1)
    assert all(r.status == STUCK and r.health == host.McuHealth.FAULT for r in stuck)
    assert len(stuck) == len(rig.readings(0))  # probed every cycle, never skipped


def test_a_raising_hal_is_reported_as_a_fault_and_never_escapes_the_main_loop():
    rig = Rig(sensors=[SensorModel(mode="error"), echo(100.0), echo(100.0), echo(100.0)])
    rig.run(2.0)  # would raise out of poll() if the core did not contain it
    assert all(r.status == STUCK and r.health == host.McuHealth.FAULT for r in rig.readings(0))
    assert rig.readings(1)[-1].status == OK


def test_a_disabled_channel_is_never_triggered_and_reenabling_resets_its_health():
    rig = Rig(sensors=[echo(100.0), SensorModel(mode="silent"), echo(100.0), echo(100.0)])
    rig.run(0.05)
    rig.send(host.FrameType.CONFIG, host.build_config(host.McuConfig(0b1101, 30, 30, 2)))
    rig.run(2.0)
    assert 1 not in {t[0] for t in rig.port.hal.triggers[-20:]}
    last = rig.readings(1)[-1]
    assert (last.status, last.health) == (fw.ST_DISABLED, host.McuHealth.DISABLED)
    seen = len(rig.readings(1))
    rig.send(host.FrameType.CONFIG, host.build_config(host.McuConfig(0b1111, 30, 30, 2)))
    rig.run(0.5)
    first_live = next(r for r in rig.readings(1)[seen:] if r.status != fw.ST_DISABLED)
    # counting restarted from zero: one silent measurement, not the history from before the mask
    assert (first_live.status, first_live.fails) == (NO_EDGE, 1)
    assert first_live.health == host.McuHealth.ACTIVE  # quiet_after is 2


def test_no_fall_counts_as_a_pulse_not_as_a_failure():
    rig = Rig(sensors=[SensorModel(mode="no_fall")] * 4)
    rig.run(1.0)
    r = rig.readings(0)[-1]
    assert (r.status, r.fails, r.health) == (NO_FALL, 0, host.McuHealth.ACTIVE)


# ---- entry point ------------------------------------------------------------------------------


def test_main_build_without_pins_makes_a_zero_sensor_core(monkeypatch):
    cfg = load_module("usonic_config")
    usonic_main = load_module("usonic_main")

    monkeypatch.setattr(cfg, "TRIG_PINS", None)
    monkeypatch.setattr(cfg, "ECHO_PINS", None)
    seen = {}

    class Hal:
        sensor_count = 0

        def __init__(self, trig, echo_pins, **kw):
            seen.update(trig=trig, echo=echo_pins, **kw)

        def now_ms(self):
            return 0

        def boot_id(self):
            return 1

        def reset_reason(self):
            return 0

        def read(self):
            return b""

        def write(self, data):
            pass

        def feed_watchdog(self):
            seen["fed"] = seen.get("fed", 0) + 1

    hal, core = usonic_main.build(Hal)
    assert core.n == 0 and seen["trig"] == () and seen["echo"] == ()
    usonic_main.run(Hal, iterations=3)
    assert seen["fed"] == 3


class _Hal:
    """A HAL that remembers what ``usonic_main.build`` handed it."""

    sensor_count = 0

    def __init__(self, trig, echo_pins, **kw):
        self.args = (trig, echo_pins, kw)

    def now_ms(self):
        return 0

    def boot_id(self):
        return 1

    def reset_reason(self):
        return 0


@pytest.mark.parametrize("which", ["TRIG_PINS", "ECHO_PINS"])
def test_main_build_with_only_one_of_the_two_pin_lists_set_makes_a_zero_sensor_core(
    monkeypatch, which
):
    # half a wiring decision is no wiring decision: the MCU must come up with no sensors and keep
    # its link, not crash while handing a None to the HAL
    cfg = load_module("usonic_config")
    usonic_main = load_module("usonic_main")
    monkeypatch.setattr(cfg, "TRIG_PINS", None)
    monkeypatch.setattr(cfg, "ECHO_PINS", None)
    monkeypatch.setattr(cfg, which, ("GP2", "GP4"))
    hal, core = usonic_main.build(_Hal)
    assert hal.args[:2] == ((), ()) and core.n == 0


def test_main_build_hands_the_owners_timing_constants_to_the_core(monkeypatch):
    cfg = load_module("usonic_config")
    usonic_main = load_module("usonic_main")
    monkeypatch.setattr(cfg, "INFO_PERIOD_MS", 333)
    monkeypatch.setattr(cfg, "PARTIAL_FRAME_TIMEOUT_MS", 17)
    monkeypatch.setattr(cfg, "DEFAULT_ECHO_WINDOW_MS", 45)
    monkeypatch.setattr(cfg, "DEFAULT_GAP_MS", 55)
    monkeypatch.setattr(cfg, "DEFAULT_QUIET_AFTER", 6)
    monkeypatch.setattr(cfg, "TRIGGER_PULSE_US", 12)
    monkeypatch.setattr(cfg, "WATCHDOG_TIMEOUT_S", 3)
    hal, core = usonic_main.build(_Hal)
    assert (core.info_period_ms, core.dec.partial_timeout) == (333, 17)
    assert (core.window_ms, core.gap_ms, core.quiet_after) == (45, 55, 6)
    assert hal.args[2] == {"trigger_pulse_us": 12, "watchdog_s": 3}


def test_the_heartbeat_and_the_partial_frame_timeout_the_owner_sets_are_the_ones_that_run(
    monkeypatch,
):
    cfg = load_module("usonic_config")
    usonic_main = load_module("usonic_main")
    monkeypatch.setattr(cfg, "INFO_PERIOD_MS", 250)
    monkeypatch.setattr(cfg, "PARTIAL_FRAME_TIMEOUT_MS", 20)
    hal = RecordingHal(n=0, start_ms=1000)
    hal.queue_rx(_BOGUS_HEADER)  # a corrupted length: a good frame behind it is held hostage
    hal.queue_rx(_ping(0x42), at_ms=1005)
    usonic_main.run(lambda *a, **k: hal, iterations=1100)
    uptimes = [m.uptime_ms for _t, m in hal.messages(host.FrameType.INFO)]
    assert uptimes[:5] == [0, 250, 500, 750, 1000], "INFO_PERIOD_MS is the heartbeat"
    ((t, pong),) = hal.messages(host.FrameType.PONG)
    assert pong.nonce == 0x42 and 1020 <= t <= 1040, "released by PARTIAL_FRAME_TIMEOUT_MS"


def test_main_build_survives_a_firing_order_that_does_not_match_the_sensor_count(monkeypatch):
    cfg = load_module("usonic_config")
    usonic_main = load_module("usonic_main")

    monkeypatch.setattr(cfg, "TRIG_PINS", ("a", "b", "c"))
    monkeypatch.setattr(cfg, "ECHO_PINS", ("d", "e", "f"))
    monkeypatch.setattr(cfg, "FIRING_ORDER", (0, 1, 2, 3))  # four entries for three sensors

    class Hal:
        sensor_count = 3

        def __init__(self, *a, **k):
            pass

        def now_ms(self):
            return 0

        def boot_id(self):
            return 1

        def reset_reason(self):
            return 0

    _hal, core = usonic_main.build(Hal)
    assert core.order == (0, 1, 2)


# ---- the main loop must never leave a silent MCU ----------------------------------------------


class _ReloadException(BaseException):
    """Stand-in for CircuitPython's reload exception (``supervisor.reload()``, Ctrl-D)."""


def _run(hal, iterations):
    load_module("usonic_config")
    return load_module("usonic_main").run(lambda *a, **k: hal, iterations=iterations)


def test_run_makes_exactly_the_requested_passes_when_nothing_fails():
    hal = RecordingHal(n=0)
    assert _run(hal, 25) == 25
    assert (hal.fed, hal.calls["read"], hal.reboots) == (25, 25, 0)


def test_an_exception_out_of_poll_forces_a_reset_instead_of_ending_code_py(capsys):
    # US-12a: an uncaught exception ends code.py, and whether the hardware watchdog is still armed
    # then depends on the CircuitPython release (adafruit/circuitpython#7722). The loop must reset
    # the MCU itself rather than leave it enumerated and silent.
    hal = RecordingHal(n=0)
    hal.arm("read", 5, RuntimeError("poll blew up"))  # core.poll() reads the host link first
    with pytest.raises(Rebooted):  # a real reset never returns
        _run(hal, 1000)
    assert hal.reboots == 1
    assert hal.fed == 5  # fed on every pass up to and including the one that crashed
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "poll blew up" in out  # the console says why


def test_an_exception_out_of_feed_watchdog_also_forces_a_reset():
    hal = RecordingHal(n=0)
    hal.arm("feed_watchdog", 3, RuntimeError("watchdog object is gone"))
    with pytest.raises(Rebooted):
        _run(hal, 1000)
    assert hal.reboots == 1
    assert (hal.fed, hal.calls["read"]) == (2, 2)  # nothing ran after the failing feed


def test_a_failure_while_announcing_the_boot_also_forces_a_reset():
    hal = RecordingHal(n=0)
    hal.arm("write", 1, OSError("usb write failed"))  # the boot INFO is the first write
    with pytest.raises(Rebooted):
        _run(hal, 1000)
    assert hal.reboots == 1
    assert hal.fed == 0


@pytest.mark.parametrize("mode", ["returns", "raises", "absent"])
def test_when_the_reset_itself_fails_the_loop_stops_feeding_and_keeps_spinning(mode):
    # No reset available (it raised, returned, or the HAL has no reboot at all): the loop must not
    # end (code.py ending is exactly what may disarm the watchdog), must not feed the watchdog
    # again, and must not poll a core whose state is unknown: the armed watchdog resets the MCU.
    hal = RecordingHal(n=0, reboot_mode=mode)
    hal.arm("read", 5, RuntimeError("poll blew up"))
    assert _run(hal, 50) == 50  # still spinning when the test's pass budget ran out
    assert hal.fed == 5
    assert hal.calls["read"] == 5
    assert hal.reboots == (0 if mode == "absent" else 1)


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), _ReloadException()])
@pytest.mark.parametrize("where", ["read", "feed_watchdog"])
def test_keyboard_interrupt_system_exit_and_reload_are_never_swallowed(exc, where):
    # Ctrl-C into the REPL, Ctrl-D and an autoreload must still stop the program: they are not
    # crashes, and resetting the MCU there would make the board impossible to work on.
    hal = RecordingHal(n=0, reboot_mode="returns")
    hal.arm(where, 3, exc)
    with pytest.raises(type(exc)):
        _run(hal, 1000)
    assert hal.reboots == 0
