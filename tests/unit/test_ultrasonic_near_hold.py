"""US3-03 / US3-05 / US3-09: the NEAR line, the dead-zone hold and proof of life, pinned exactly.

A sensor whose last real echo was NEAR (closer than ``NEAR_ECHO_FACTOR`` times the larger of the
blind-zone line and the stop threshold) and that then goes silent may have an object in its dead
zone: the channel reads BLIND_ZONE / UNKNOWN for ``near_hold_s`` whatever the ``*_is_clear``
flags say. That band (30 to 60 cm with the shipped numbers) is what covers an object that crosses
from 40 cm into the dead zone inside one measurement cycle, and no test used to sit inside it.

The timing tests run on dyadic clock values and a speed of sound of 350 m/s, so every comparison
is an exact one: 4000 us at 350 m/s is 70.0 cm to the last bit.
"""

from __future__ import annotations

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto
from backend.src.drivers.sensors.jsn_sr04t_driver import (
    ChannelStatus,
    Health,
    JsnSr04tMcuDriver,
    LinkState,
    Verdict,
    echo_to_cm,
)
from backend.src.models.hardware_config import UltrasonicConfig
from tests.unit._usonic_fakes import FakeClock
from tests.unit.test_jsn_sr04t_driver import ScriptedPort, mcu_ms

OK, NO_EDGE, NO_FALL, STUCK = (
    proto.EchoStatus.OK,
    proto.EchoStatus.NO_EDGE,
    proto.EchoStatus.NO_FALL,
    proto.EchoStatus.STUCK,
)
C = 350.0  # m/s: with it 1 us is 0.0175 cm, so the numbers below are exact


def us_for(cm: float) -> int:
    return round(cm * 20000.0 / C)


class Rig:
    """One scripted MCU, a driver with a chosen stop threshold, and a clock that moves in 0.25 s."""

    def __init__(self, *, threshold_cm: float = 30.0, **cfg) -> None:
        cfg.setdefault("speed_of_sound_mps", C)
        self.clock = FakeClock()
        self.port = ScriptedPort()
        self.driver = JsnSr04tMcuDriver(
            UltrasonicConfig(port="/dev/ttyACM-usonic", **cfg),
            serial_factory=lambda _path, _baud: self.port,
            clock=self.clock.now,
            wall_clock=self.clock.wall,
            sleep=self.clock.sleep,
            obstacle_threshold_cm=threshold_cm,
            simulate=False,
        )
        self.connect()

    def connect(self) -> None:
        self.driver.poll()
        self.port.info()
        self.driver.poll()
        self.port.ack(self.driver.expected_mcu_config())
        self.driver.poll()
        assert self.driver.link_state is LinkState.ACTIVE

    def say(self, status, echo_us=0, *, dt=0.25, sensor=0, behind=0.0) -> None:
        self.clock.advance(dt)
        self.port.reading(
            sensor=sensor, status=status, echo_us=echo_us, t_ms=mcu_ms(self.clock, behind)
        )
        self.driver.poll()

    def silence(self, seconds: float, *, dt=0.25) -> None:
        for _ in range(round(seconds / dt)):
            self.say(NO_EDGE, dt=dt)

    def warm(self) -> Rig:
        """Two ordinary frames, so the clock baseline exists before a backlog arrives."""
        self.silence(0.5)
        return self

    @property
    def ch(self):
        return self.driver.snapshot()[0]


# ---- the near line -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "threshold", "echo_us", "near"),
    [
        # shipped numbers: min_range 25, stop threshold 30 -> the NEAR line is 2 x 30 = 60 cm
        ({}, 30.0, us_for(31.0), True),  # just beyond the stop threshold
        ({}, 30.0, us_for(45.0), True),
        ({}, 30.0, 3428, True),  # 59.99 cm
        ({}, 30.0, 3429, False),  # 60.0075 cm
        ({}, 30.0, us_for(80.0), False),
        # a larger stop threshold moves the line with it: 2 x 35 = 70 cm, exactly 4000 us
        ({}, 35.0, 3999, True),
        ({}, 35.0, 4000, False),  # exactly ON the line is not near (strict)
        ({}, 35.0, 4001, False),
        # the blind-zone line wins when it is the larger one: 2 x 40 = 80 cm
        ({"min_range_cm": 40.0}, 30.0, 4571, True),  # 79.99 cm
        ({"min_range_cm": 40.0}, 30.0, 4572, False),  # 80.01 cm
        # and the threshold wins when IT is larger: min_range 25 must not set the line (2 x 45)
        ({}, 45.0, 5142, True),  # 89.99 cm
        ({}, 45.0, 5143, False),  # 90.0025 cm
    ],
    ids=lambda v: str(v),
)
def test_silence_after_an_echo_is_the_dead_zone_exactly_inside_the_near_line(
    cfg, threshold, echo_us, near
):
    rig = Rig(threshold_cm=threshold, **cfg)
    rig.say(OK, echo_us)
    assert rig.ch.valid, "the echo itself is a normal reading"
    rig.silence(1.0)
    got = (rig.ch.status, rig.ch.verdict)
    if near:
        assert got == (ChannelStatus.BLIND_ZONE, Verdict.UNKNOWN), (
            f"{echo_to_cm(echo_us, C):.2f} cm is NEAR: its silence is the dead zone"
        )
    else:
        assert rig.ch.status is ChannelStatus.NO_ECHO, (
            f"{echo_to_cm(echo_us, C):.2f} cm is not near: its silence is open air or a dead sensor"
        )


def test_the_near_line_arithmetic_is_exact_at_350_mps():
    assert echo_to_cm(4000, C) == 70.0
    assert echo_to_cm(3428, C) < 60.0 <= echo_to_cm(3429, C)


# ---- how long the hold lasts and how long proof of life lasts ---------------------------------


def test_the_hold_lasts_exactly_near_hold_s_after_the_near_echo():
    rig = Rig(near_hold_s=5.0)
    rig.say(OK, us_for(28.0))  # 28 cm, measured at exactly t = 1000.25
    for _ in range(19):  # 0.25 ... 4.75 s of silence after it
        rig.say(NO_EDGE)
        assert rig.ch.status is ChannelStatus.BLIND_ZONE
    rig.say(NO_EDGE)  # exactly 5.0 s after the near echo: the hold is still on
    assert rig.ch.status is ChannelStatus.BLIND_ZONE and rig.ch.age_ms == 0
    rig.say(NO_EDGE)  # 5.25 s: over
    assert rig.ch.status is ChannelStatus.NO_ECHO


def test_proof_of_life_lasts_exactly_alive_window_s_after_a_far_echo():
    rig = Rig(no_echo_is_clear=True, alive_window_s=5.0)
    rig.say(OK, us_for(100.0))
    for _ in range(19):
        rig.say(NO_EDGE)
        assert rig.ch.alive and rig.ch.verdict is Verdict.CLEAR
    rig.say(NO_EDGE)  # exactly 5.0 s after the far echo: still alive
    assert rig.ch.alive and rig.ch.verdict is Verdict.CLEAR and rig.ch.health is Health.OK
    rig.say(NO_EDGE)
    assert not rig.ch.alive and rig.ch.verdict is Verdict.UNKNOWN
    assert rig.ch.status is ChannelStatus.NO_ECHO and rig.ch.health is Health.SILENT


def test_a_reading_is_stale_only_beyond_stale_after_ms():
    rig = Rig(stale_after_ms=750)
    rig.say(OK, us_for(100.0))
    rig.clock.advance(0.75)  # exactly the limit: still fresh
    assert rig.ch.status is ChannelStatus.OK and rig.ch.age_ms == 750
    rig.clock.advance(0.25)
    assert rig.ch.status is ChannelStatus.STALE and rig.ch.verdict is Verdict.UNKNOWN


# ---- the hold and the proof of life are dated by the MEASUREMENT --------------------------------


def backlog(rig: Rig, frames) -> None:
    """Deliver ``frames`` (status, echo_us, seconds before now) in ONE poll, oldest first."""
    for status, echo_us, behind in frames:
        rig.port.reading(status=status, echo_us=echo_us, t_ms=mcu_ms(rig.clock, behind))
    rig.driver.poll()


def silent_since(rig: Rig, seconds: float):
    return [(NO_EDGE, 0, seconds - 0.05 * k) for k in range(1, round(seconds / 0.05) + 1)]


def test_proof_of_life_counts_from_when_the_pulse_was_measured_not_from_when_it_was_read():
    rig = Rig(no_echo_is_clear=True, alive_window_s=5.0).warm()
    rig.clock.advance(10.0)  # the host loop was stalled; the whole backlog arrives at once
    backlog(rig, [(OK, us_for(100.0), 8.0), *silent_since(rig, 8.0)])
    assert rig.ch.status is ChannelStatus.NO_ECHO and rig.ch.age_ms < 100
    assert not rig.ch.alive, "a far echo measured 8 s ago is no proof the sensor lives now"
    assert rig.ch.verdict is Verdict.UNKNOWN


def test_the_hold_counts_from_the_measurement_of_the_near_echo_not_from_when_it_was_read():
    rig = Rig(near_hold_s=5.0).warm()
    rig.clock.advance(10.0)
    backlog(rig, [(OK, us_for(28.0), 6.0), *silent_since(rig, 6.0)])
    assert rig.ch.status is ChannelStatus.NO_ECHO, "6 s of silence since the near echo: hold over"
    rig2 = Rig(near_hold_s=5.0).warm()
    rig2.clock.advance(10.0)
    backlog(rig2, [(OK, us_for(28.0), 4.0), *silent_since(rig2, 4.0)])
    assert rig2.ch.status is ChannelStatus.BLIND_ZONE, "4 s since it: still inside the hold"


# ---- what must NOT release the hold -------------------------------------------------------------


@pytest.mark.parametrize("fault", [STUCK, NO_FALL], ids=["stuck", "no_fall"])
def test_a_fault_or_an_endless_pulse_does_not_release_the_dead_zone_hold(fault):
    rig = Rig(near_hold_s=30.0, no_echo_is_clear=True, no_echo_requires_alive=False)
    rig.say(OK, us_for(28.0))
    for _ in range(3):
        rig.say(fault, 30_000)
    status_during = rig.ch.status
    rig.silence(1.0)
    assert rig.ch.status is ChannelStatus.BLIND_ZONE and rig.ch.verdict is Verdict.UNKNOWN
    if fault is NO_FALL:  # not "out of range": something may still be right in front
        assert status_during is ChannelStatus.BLIND_ZONE
    else:
        assert status_during is ChannelStatus.FAULT


def test_a_far_echo_on_the_same_channel_releases_the_hold_at_once():
    rig = Rig(near_hold_s=30.0, no_echo_is_clear=True)
    rig.say(OK, us_for(28.0))
    rig.silence(0.5)
    assert rig.ch.status is ChannelStatus.BLIND_ZONE
    rig.say(OK, us_for(150.0))
    rig.silence(0.5)
    assert (rig.ch.status, rig.ch.alive, rig.ch.verdict) == (
        ChannelStatus.NO_ECHO, True, Verdict.CLEAR,
    )  # fmt: skip


def test_the_hold_survives_a_link_blip_but_proof_of_life_does_not():
    # an object that was right in front before the blip is still right in front after it; and a
    # far echo heard before it says nothing about whether the sensor still lives afterwards
    rig = Rig(near_hold_s=30.0, no_echo_is_clear=True)
    rig.say(OK, us_for(28.0))
    rig.say(NO_EDGE)
    assert rig.ch.status is ChannelStatus.BLIND_ZONE
    rig.port.broken = OSError("device reports readiness to read but returned no data")
    rig.clock.advance(0.02)
    rig.driver.poll()
    assert rig.driver.link_state is LinkState.DISCONNECTED
    rig.port.broken = None
    rig.port.inbox.clear()
    rig.clock.advance(0.5)
    rig.connect()
    rig.silence(1.0)
    assert rig.ch.status is ChannelStatus.BLIND_ZONE, "the latch is still there after the blip"

    far = Rig(no_echo_is_clear=True, alive_window_s=60.0)
    far.say(OK, us_for(150.0))
    far.silence(0.5)
    assert far.ch.alive and far.ch.verdict is Verdict.CLEAR
    far.port.broken = OSError("device reports readiness to read but returned no data")
    far.clock.advance(0.02)
    far.driver.poll()
    far.port.broken = None
    far.port.inbox.clear()
    far.clock.advance(0.5)
    far.connect()
    far.silence(1.0)
    assert not far.ch.alive and far.ch.verdict is Verdict.UNKNOWN, "proof of life did not survive"


def test_alive_needs_a_live_link_even_while_the_channel_still_holds_a_recent_pulse():
    rig = Rig(no_echo_is_clear=True, alive_window_s=60.0)
    rig.say(OK, us_for(150.0))
    rig.silence(0.5)
    assert rig.ch.alive
    rig.clock.advance(2.0)  # the MCU goes quiet: past link_timeout_ms, not yet declared lost
    c = rig.ch
    assert c.status is ChannelStatus.LINK_DOWN and not c.alive and c.verdict is Verdict.UNKNOWN


def test_a_channel_that_never_reported_has_no_echo_width_not_a_zero():
    rig = Rig()
    rig.say(OK, us_for(100.0), sensor=0)
    never = rig.driver.snapshot()[3]
    assert never.status is ChannelStatus.NO_DATA and never.echo_us is None


# ---- US3-09: an OK pulse wider than the echo window never happened ------------------------------


def test_an_ok_pulse_wider_than_the_echo_window_is_a_fault_and_cannot_release_the_hold():
    # reviewer e5_near: after a near echo and one silent cycle an OK frame with echo_us 900000 read
    # out_of_range / clear / alive and the next silent cycle was no_echo: the hold was gone
    rig = Rig(near_hold_s=30.0, no_echo_is_clear=True, out_of_range_is_clear=True)
    rig.say(OK, us_for(28.0))
    rig.silence(0.5)
    rig.say(OK, 900_000)
    c = rig.ch
    assert c.status is ChannelStatus.FAULT and c.verdict is Verdict.UNKNOWN and not c.alive
    rig.silence(1.0)
    assert rig.ch.status is ChannelStatus.BLIND_ZONE, "the lie did not release the dead-zone hold"


def test_an_ok_pulse_is_a_measurement_up_to_the_echo_window_and_a_fault_one_microsecond_beyond():
    rig = Rig(echo_window_ms=30)
    rig.say(OK, 30_000)  # as wide as the window: 525 cm, beyond max range, but possible
    assert rig.ch.status is ChannelStatus.OUT_OF_RANGE
    rig.say(OK, 30_001)
    assert rig.ch.status is ChannelStatus.FAULT and rig.ch.verdict is Verdict.UNKNOWN


def test_max_echo_us_is_the_echo_window_capped_at_the_sane_limit():
    from backend.src.drivers.sensors.jsn_sr04t_driver import max_echo_us

    cfg = UltrasonicConfig(echo_window_ms=100, inter_sensor_gap_ms=50, stale_after_ms=1000)
    assert max_echo_us(cfg) == 100_000

    class Wide:  # a window no valid configuration allows today: the sanity cap still holds
        echo_window_ms = 5000

    assert max_echo_us(Wide()) == 1_000_000
