"""US3-01 / US3-11: dating readings by the MCU's own clock, poisoned for good by one bad stamp.

The host dates every reading ``t_ms`` after one baseline per link (the lowest ``receive time -
t_ms`` it has seen). Before this round one CRC-valid reading whose ``t_ms`` ran ahead of the truth
moved the baseline's "newest timestamp" forward, and every later genuine frame of every channel
was then "older than the newest" and refused, for good: the link stayed ACTIVE, only
``stats.readings_stale_mcu`` grew (fail closed, but a permanent outage with no self-repair). The
same followed a frame of a channel the host does not use and an MCU clock that restarts under an
unchanged ``boot_id``. What must hold now:

* a timestamp ahead of the settled baseline by more than the longest spacing of two readings is
  refused and moves NOTHING; one inside the limit can only make later data look a little older;
* frames of a channel the host did not enable never touch the baseline;
* readings refused for 2 s while their own MCU clock kept pace with the host's are a live stream
  whose baseline is wrong: the baseline is rebuilt from them, and until then nothing is fresh;
* replays, duplicates, loops and fast-forwarded backlogs never qualify for that rebuild.
"""

from __future__ import annotations

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto
from backend.src.drivers.sensors.jsn_sr04t_driver import ChannelStatus, LinkState, mcu_delta
from tests.unit.test_jsn_sr04t_driver import mcu_ms, scripted_active, scripted_rig, steady

HOUR_MS = 3_600_000


def valid_flags(driver) -> list[bool]:
    return [c.valid for c in driver.snapshot() if c.sensor_id in driver.channel_ids]


def feed_live(clock, port, driver, seconds, *, restart_at=None, sensors=(0, 1, 2, 3)):
    """100 ms steps; every sensor reports each step, stamped on a clock that restarted at
    ``restart_at`` (host time) or, with None, on the original MCU clock."""
    rows = []
    for _ in range(round(seconds / 0.1)):
        clock.advance(0.1)
        stamp = (
            mcu_ms(clock) if restart_at is None else 100 + round((clock.now() - restart_at) * 1000)
        )
        for sensor in sensors:
            port.reading(sensor=sensor, t_ms=stamp)
        driver.poll()
        rows.append((clock.now() - (restart_at or 0.0), [c.status for c in driver.snapshot()]))
    return rows


# ---- one wild timestamp -------------------------------------------------------------------------


def test_one_wild_timestamp_an_hour_ahead_is_refused_and_never_applied():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    assert all(valid_flags(driver))
    port.reading(sensor=0, echo_us=1000, t_ms=mcu_ms(clock) + HOUR_MS)  # 17 cm, an hour ahead
    clock.advance(0.02)
    driver.poll()
    c = driver.snapshot()[0]
    assert c.valid and c.echo_us == 5828, "the wild reading must not have been applied"
    assert driver.stats.readings_stale_mcu == 1


def test_one_wild_timestamp_cannot_silence_every_channel_for_good():
    # reviewer e1_poison: a healthy 4-channel stream, one CRC-valid reading with t_ms + 1 h, then
    # 30 s of healthy traffic. The old driver: every channel stale for good, the link "active".
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    port.reading(sensor=0, echo_us=1000, t_ms=mcu_ms(clock) + HOUR_MS)
    clock.advance(0.02)
    driver.poll()
    steady(clock, port, driver, 30.0)
    assert all(valid_flags(driver)) and all(c.age_ms < 150 for c in driver.snapshot())
    assert driver.snapshot()[0].echo_us == 5828
    assert driver.link_state is LinkState.ACTIVE and driver.link_ok()
    assert driver.stats.mcu_rebases == 0, "nothing needed repairing: the wild reading moved nothing"


def test_a_reading_just_inside_the_slack_is_accepted_and_only_makes_later_data_look_older():
    clock, port, driver, _ = scripted_active()  # default timing: slack = twice 60 ms = 0.12 s
    steady(clock, port, driver, 1.0)
    port.reading(sensor=0, echo_us=2000, t_ms=mcu_ms(clock) + 110)  # 0.11 s ahead: inside
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[0].echo_us == 2000 and driver.stats.readings_stale_mcu == 0
    steady(clock, port, driver, 2.0)
    assert all(valid_flags(driver)), "a bias under the slack never costs a channel its freshness"
    assert all(c.age_ms < 150 for c in driver.snapshot())


def test_a_reading_beyond_the_slack_moves_nothing_and_leaves_no_blackout():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    port.reading(sensor=1, echo_us=2000, t_ms=mcu_ms(clock) + 300)  # 0.3 s ahead: beyond 0.12 s
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[1].echo_us == 5828 and driver.stats.readings_stale_mcu == 1
    clock.advance(0.08)
    for sensor in range(4):  # the very next genuine frames, channel 1 included
        port.reading(sensor=sensor, t_ms=mcu_ms(clock))
    driver.poll()
    assert all(c.valid and c.age_ms < 20 for c in driver.snapshot())
    assert driver.stats.readings_stale_mcu == 1


def test_the_slack_grows_with_the_longest_spacing_of_two_readings():
    slow = dict(echo_window_ms=100, inter_sensor_gap_ms=500, stale_after_ms=3000)
    clock, port, driver, _ = scripted_active(**slow)  # 0.6 s between readings: slack 1.2 s
    steady(clock, port, driver, 1.0)
    port.reading(sensor=0, echo_us=2000, t_ms=mcu_ms(clock) + 1000)
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[0].echo_us == 2000 and driver.stats.readings_stale_mcu == 0
    port.reading(sensor=1, echo_us=2000, t_ms=mcu_ms(clock) + 3000)  # beyond even that
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[1].echo_us == 5828 and driver.stats.readings_stale_mcu == 1
    # ... and the same one second would be wild at the default timing
    clock2, port2, driver2, _ = scripted_active()
    steady(clock2, port2, driver2, 1.0)
    port2.reading(sensor=0, echo_us=2000, t_ms=mcu_ms(clock2) + 1000)
    driver2.poll()
    assert driver2.snapshot()[0].echo_us == 5828 and driver2.stats.readings_stale_mcu == 1


def test_a_backlog_drained_at_link_up_may_lower_the_offset_with_every_frame():
    # 50 readings 60 ms apart, 3 s of backlog, all delivered in the first poll after the handshake
    clock, port, driver, _ = scripted_active()
    clock.advance(3.0)
    for k in range(50):
        port.reading(sensor=k % 4, t_ms=mcu_ms(clock, behind_s=2.94 - 0.06 * k))
    driver.poll()
    assert driver.stats.readings == 50 and driver.stats.readings_stale_mcu == 0
    assert all(c.valid and c.age_ms < 120 for c in driver.snapshot())


# ---- a channel the host does not use --------------------------------------------------------


def wild_frame_on_the_disabled_channel():
    clock, port, driver, _ = scripted_active(disabled_channels=["inner_right"])
    steady(clock, port, driver, 1.0, sensors=(0, 1, 3))
    n = driver.stats.readings
    port.reading(
        sensor=2, status=proto.EchoStatus.DISABLED, echo_us=0, t_ms=mcu_ms(clock) + 600_000
    )
    clock.advance(0.02)
    driver.poll()
    return clock, port, driver, n


def test_a_reading_of_a_channel_the_host_did_not_enable_is_ignored():
    _clock, _port, driver, n = wild_frame_on_the_disabled_channel()
    assert driver.stats.readings == n and driver.stats.readings_ignored == 1
    assert driver.stats.readings_stale_mcu == 0


def test_a_wild_timestamp_on_a_channel_the_host_did_not_enable_cannot_silence_the_others():
    # reviewer e3_replay E4c: a DISABLED reading of the host-disabled inner_right, t_ms + 10 min
    clock, port, driver, _ = wild_frame_on_the_disabled_channel()
    steady(clock, port, driver, 5.0, sensors=(0, 1, 3))
    fitted = [c for c in driver.snapshot() if c.sensor_id != "inner_right"]
    assert all(c.valid and c.age_ms < 150 for c in fitted)
    assert driver.stats.readings_stale_mcu == 0


# ---- a wrong baseline repairs itself -------------------------------------------------------------


def test_an_mcu_clock_that_restarts_under_the_same_boot_id_is_rebuilt_from_the_live_stream():
    # reviewer e2_reset case C: t_ms restarts at ~0, the boot_id repeats (os.urandom has a
    # fallback id that is not unique). The old driver refused every frame for ever.
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    assert all(valid_flags(driver))
    restart = clock.now()
    rows = feed_live(clock, port, driver, 6.0, restart_at=restart)
    first_stale = next(i for i, (_, st) in enumerate(rows) if ChannelStatus.STALE in st)
    back = next(
        (i for i, (_, st) in enumerate(rows) if i > first_stale and set(st) == {ChannelStatus.OK}),
        None,
    )
    assert back is not None, "the array never recovered: " + str(rows[-1])
    assert 0.7 <= rows[first_stale][0] <= 1.0, "the pre-restart readings age out first"
    assert 2.0 <= rows[back][0] <= 3.0, f"recovered after {rows[back][0]:.1f} s"
    for t, st in rows[first_stale:back]:  # in between nothing is fresh
        assert set(st) == {ChannelStatus.STALE}, f"t={t:.1f}s {st}"
    assert set(rows[-1][1]) == {ChannelStatus.OK} and driver.stats.mcu_rebases == 1
    assert driver.link_state is LinkState.ACTIVE


def test_a_wild_first_timestamp_of_a_link_is_repaired_the_same_way():
    clock, port, driver, _ = scripted_active()
    clock.advance(0.1)
    port.reading(sensor=0, t_ms=mcu_ms(clock) + HOUR_MS)  # the very first reading is corrupted
    driver.poll()
    rows = feed_live(clock, port, driver, 6.0)
    back = next(
        (i for i, (_, st) in enumerate(rows) if set(st) == {ChannelStatus.OK} and i > 5), None
    )
    assert back is not None, "the array never recovered: " + str(rows[-1])
    assert 2.0 <= rows[back][0] - rows[0][0] + 0.1 <= 3.0
    assert driver.stats.mcu_rebases == 1
    assert all(set(st) == {ChannelStatus.OK} for _, st in rows[back:])


def test_a_stream_that_stays_ahead_of_the_baseline_is_adopted_only_after_it_proved_itself():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    shifted = clock.now()
    ahead = []
    for _ in range(50):  # from now on every frame is stamped 1 s ahead of what the baseline says
        clock.advance(0.1)
        for sensor in range(4):
            port.reading(sensor=sensor, t_ms=mcu_ms(clock) + 1000)
        driver.poll()
        ahead.append((clock.now() - shifted, [c.status for c in driver.snapshot()]))
    assert all(s is not ChannelStatus.OK for t, st in ahead if 0.9 <= t < 2.0 for s in st), (
        "a stream nobody has confirmed yet is never fresh"
    )
    assert all(s is ChannelStatus.OK for _, st in ahead[-5:] for s in st)
    assert driver.stats.mcu_rebases == 1


# ---- what must NEVER rebuild the baseline --------------------------------------------------------


def run_replay(frames_per_step, seconds=30.0):
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    old = mcu_ms(clock, behind_s=5.0)
    seen_valid = False
    for step in range(round(seconds / 0.1)):
        clock.advance(0.1)
        for frame in frames_per_step(step, old):
            port.reading(sensor=frame[0], t_ms=frame[1])
        driver.poll()
        if step >= 10:
            seen_valid |= any(valid_flags(driver))
    return driver, seen_valid


@pytest.mark.parametrize(
    "replay",
    [
        lambda step, old: [(0, old)],  # the very same frame, again and again
        lambda step, old: [(step % 4, old + 100 * (step % 4))],  # a loop of four old frames
        lambda step, old: [(s, old + 1000 * step + s) for s in range(4)],  # old, 10x real time
    ],
    ids=["duplicate", "loop", "fast_forward"],
)
def test_replays_never_rebuild_the_baseline_and_never_read_fresh(replay):
    driver, seen_valid = run_replay(replay)
    assert driver.stats.mcu_rebases == 0
    assert not seen_valid, "old data was served as fresh"
    assert all(c.status is ChannelStatus.STALE for c in driver.snapshot())


def test_a_healthy_stream_with_an_occasional_duplicate_never_rebuilds_its_baseline():
    # a refused frame in the middle of a stream that is otherwise accepted must not add up: any
    # accepted reading ends the run of refusals
    clock, port, driver, _ = scripted_active()
    for _ in range(40):
        steady(clock, port, driver, 1.0)
        port.reading(sensor=0, t_ms=mcu_ms(clock))  # the very same reading again
        driver.poll()
    assert driver.stats.readings_stale_mcu == 40 and driver.stats.mcu_rebases == 0
    assert all(valid_flags(driver))


def test_too_few_refused_readings_never_rebuild_the_baseline_however_consistent():
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    restart = clock.now()
    for k in range(5):  # five readings 0.6 s apart, on a restarted clock that keeps pace: 2.4 s
        port.reading(sensor=0, t_ms=100 + 600 * k)
        driver.poll()
        clock.advance(0.6)
    assert driver.stats.mcu_rebases == 0 and clock.now() - restart > 2.0


def test_the_rebuild_window_grows_with_the_measurement_cycle():
    slow = dict(echo_window_ms=100, inter_sensor_gap_ms=500, stale_after_ms=3000)  # cycle 2.4 s
    clock, port, driver, _ = scripted_active(**slow)
    steady(clock, port, driver, 1.0)
    restart = clock.now()
    feed_live(clock, port, driver, 4.0, restart_at=restart)
    assert driver.stats.mcu_rebases == 0, "4 s is less than two cycles: still collecting evidence"
    feed_live(clock, port, driver, 2.0, restart_at=restart)
    assert driver.stats.mcu_rebases == 1


def test_the_rebuilt_baseline_dates_readings_by_the_best_latency_of_the_stream_that_proved_it():
    # the stream that rebuilds the baseline arrives 0.3 s late, except for one reading with no
    # delay: the first accepted reading must be dated 0.3 s old, not "just now"
    clock, port, driver, _ = scripted_active()
    steady(clock, port, driver, 1.0)
    restart = clock.now()
    late = 0.3
    for k in range(40):
        clock.advance(0.1)
        port.reading(
            sensor=0, t_ms=100 + round((clock.now() - restart - (0.0 if k == 3 else late)) * 1000)
        )
        driver.poll()
        if driver.stats.mcu_rebases:
            break
    assert driver.stats.mcu_rebases == 1
    c = driver.snapshot()[0]
    assert c.valid and 0.25 <= c.age_ms / 1000 <= 0.35, f"age {c.age_ms} ms"


def test_a_backlog_after_a_reconnect_may_again_lower_the_offset_with_every_frame():
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 2.0)  # a long settled link
    port.broken = OSError("device reports readiness to read but returned no data")
    clock.advance(0.02)
    driver.poll()
    port.broken = None
    port.inbox.clear()
    clock.advance(0.5)
    driver.poll()
    port.info(boot_id=8)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE and len(opens) == 2
    clock.advance(3.0)
    for k in range(50):  # 3 s of backlog on the new link, all delivered in one poll
        port.reading(sensor=k % 4, t_ms=mcu_ms(clock, behind_s=2.94 - 0.06 * k))
    driver.poll()
    assert driver.stats.readings_stale_mcu == 0
    assert all(c.valid and c.age_ms < 120 for c in driver.snapshot())


def test_a_link_lost_before_the_rebuild_does_not_carry_its_refusals_into_the_next_link():
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 1.0)
    restart = clock.now()
    feed_live(clock, port, driver, 1.5, restart_at=restart)  # 1.5 s of refusals: not yet enough
    port.broken = OSError("device reports readiness to read but returned no data")
    clock.advance(0.02)
    driver.poll()
    port.broken = None
    port.inbox.clear()
    clock.advance(0.5)
    driver.poll()
    port.info(boot_id=8)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert len(opens) == 2 and driver.link_state is LinkState.ACTIVE
    steady(clock, port, driver, 3.0)  # the new link's stream runs on the original clock again
    assert driver.stats.mcu_rebases == 0 and all(valid_flags(driver))


def wedge(**cfg):
    """A settled link whose MCU clock then restarts: from here on every frame is refused."""
    clock, port, driver, _ = scripted_active(**cfg)
    steady(clock, port, driver, 1.0)
    return clock, port, driver, clock.now()


def restarted(clock, restart_at, extra_ms=0):
    return 100 + extra_ms + round((clock.now() - restart_at) * 1000)


def test_a_stream_whose_clock_runs_far_slower_than_the_hosts_never_qualifies():
    clock, port, driver, restart = wedge()
    for _ in range(80):  # 8 s: the MCU clock advances 10 ms for every 100 ms of the host's
        clock.advance(0.1)
        port.reading(sensor=0, t_ms=100 + round((clock.now() - restart) * 100))
        driver.poll()
    assert driver.stats.mcu_rebases == 0
    assert driver.snapshot()[0].status is ChannelStatus.STALE


def test_a_fast_forwarded_burst_before_a_live_stream_does_not_prevent_the_rebuild():
    clock, port, driver, restart = wedge()
    for k in range(10):  # ten seconds of an old recording, delivered in one poll
        port.reading(sensor=0, t_ms=100 + 1000 * k)
    driver.poll()
    start = clock.now()
    while clock.now() - start < 8.0 and not driver.stats.mcu_rebases:
        clock.advance(0.1)
        for sensor in range(4):
            port.reading(sensor=sensor, t_ms=10_100 + round((clock.now() - start) * 1000))
        driver.poll()
    assert driver.stats.mcu_rebases == 1, "the burst's mismatch must not outlive the next window"
    assert 3.5 <= clock.now() - start <= 6.0


def test_refused_readings_separated_by_a_long_silence_are_not_one_run_of_evidence():
    clock, port, driver, restart = wedge()
    for _ in range(9):  # 0.9 s of refused live frames: not enough yet
        clock.advance(0.1)
        port.reading(sensor=0, t_ms=restarted(clock, restart))
        driver.poll()
    for _ in range(4):  # then 4 s in which only the old reading is repeated (the link stays up)
        clock.advance(1.0)
        port.reading(sensor=1, t_ms=restarted(clock, restart) - 5000)
        driver.poll()
    clock.advance(0.1)
    port.reading(sensor=0, t_ms=restarted(clock, restart))  # one more that fits the old clock
    driver.poll()
    assert driver.stats.mcu_rebases == 0


def test_a_reconnect_starts_a_new_baseline_that_may_lower_its_offset_at_once():
    # The age of the very first reading of a baseline is unknowable (the MCU clock has no absolute
    # reference), so it is dated "now". The next reading of the stream corrects that at once: the
    # new baseline is NOT settled, whatever the link that came before it was.
    clock, port, driver, opens = scripted_active()
    steady(clock, port, driver, 2.0)
    port.broken = OSError("device reports readiness to read but returned no data")
    clock.advance(0.02)
    driver.poll()
    port.broken = None
    port.inbox.clear()
    clock.advance(0.5)
    driver.poll()
    port.info(boot_id=8)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert len(opens) == 2 and driver.link_state is LinkState.ACTIVE
    clock.advance(0.1)
    port.reading(sensor=0, t_ms=mcu_ms(clock, behind_s=2.0))  # alone in its poll: 2 s old
    driver.poll()
    assert driver.snapshot()[0].valid and driver.snapshot()[0].age_ms == 0, "dated now: the limit"
    clock.advance(0.1)
    for sensor in range(4):
        port.reading(sensor=sensor, t_ms=mcu_ms(clock))
    driver.poll()
    assert all(c.valid and c.age_ms < 20 for c in driver.snapshot())
    assert driver.stats.readings_stale_mcu == 0


def test_the_slack_has_a_floor_for_fast_timings():
    fast = dict(echo_window_ms=10, inter_sensor_gap_ms=5, max_range_cm=100.0)
    clock, port, driver, _ = scripted_active(**fast)  # 15 ms between readings: twice is 30 ms
    steady(clock, port, driver, 1.0)
    port.reading(sensor=0, echo_us=2000, t_ms=mcu_ms(clock) + 80)  # 0.08 s ahead
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[0].echo_us == 2000 and driver.stats.readings_stale_mcu == 0
    port.reading(sensor=1, echo_us=2000, t_ms=mcu_ms(clock) + 200)
    clock.advance(0.02)
    driver.poll()
    assert driver.snapshot()[1].echo_us == 5828 and driver.stats.readings_stale_mcu == 1


# ---- the arithmetic of the clock -----------------------------------------------------------------


def test_mcu_delta_is_signed_on_the_wrapping_uint32_clock():
    assert mcu_delta(100, 100) == 0
    assert mcu_delta(2**31 - 1, 0) == 2**31 - 1
    assert mcu_delta(2**31, 0) == -(2**31), "exactly half the range is read as the PAST"
    assert mcu_delta(2**31 + 1, 0) == -(2**31 - 1)
    assert mcu_delta(5, 2**32 - 5) == 10, "across the wrap"
    assert mcu_delta(2**32 - 5, 5) == -10


def test_the_newest_timestamp_never_moves_back_when_a_slightly_older_frame_is_accepted():
    clock, port, driver, _ = scripted_active()  # cycle 240 ms
    steady(clock, port, driver, 1.0, sensors=(0,))
    top = mcu_ms(clock)
    port.reading(sensor=1, t_ms=top - 200)  # older than the newest but inside one cycle: accepted
    driver.poll()
    n, refused = driver.stats.readings, driver.stats.readings_stale_mcu
    port.reading(sensor=2, t_ms=top - 400)  # 400 ms behind the NEWEST: refused
    driver.poll()
    assert (driver.stats.readings, driver.stats.readings_stale_mcu) == (n, refused + 1)
    assert driver.snapshot()[2].status is ChannelStatus.NO_DATA


def test_a_reading_exactly_one_cycle_behind_the_newest_is_accepted_and_one_ms_more_is_not():
    clock, port, driver, _ = scripted_active()  # cycle 4 * (30 + 30) = 240 ms
    steady(clock, port, driver, 1.0, sensors=(0,))
    top = mcu_ms(clock)
    port.reading(sensor=1, t_ms=top - 240)
    driver.poll()
    assert (
        driver.stats.readings_stale_mcu == 0
        and driver.snapshot()[1].status is not ChannelStatus.NO_DATA
    )
    port.reading(sensor=2, t_ms=top - 241)
    driver.poll()
    assert (
        driver.stats.readings_stale_mcu == 1
        and driver.snapshot()[2].status is ChannelStatus.NO_DATA
    )


# ---- US3-11: the age of the very first frame is not knowable, so nothing before the ACK counts


def test_frames_buffered_before_the_config_echo_are_ignored_and_never_dated_now():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    clock.advance(5.0)  # the line held 5 s of backlog
    port.info()
    for k in range(4):
        port.reading(sensor=k, t_ms=mcu_ms(clock, behind_s=5.0 - 0.1 * k))
    port.ack(driver.expected_mcu_config())  # the ACK was not even requested yet: it is ignored too
    driver.poll()
    port.ack(driver.expected_mcu_config())
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE
    assert driver.stats.readings == 0 and driver.stats.readings_ignored == 4
    assert all(c.status is ChannelStatus.NO_DATA for c in driver.snapshot())


def test_the_newest_frame_of_a_channel_in_the_first_batch_after_the_handshake_dates_it():
    # The age of the first frame is unknowable (the MCU clock has no absolute reference), so the
    # first accepted frame is dated "now". It is bounded by what was emitted AFTER the CONFIG
    # echo, and a later frame of the same channel in the same batch corrects it at once.
    clock, port, driver, _ = scripted_active()
    clock.advance(2.0)
    for behind in (1.9, 1.0, 0.0):  # one channel, three frames of a stalled loop's backlog
        port.reading(sensor=0, t_ms=mcu_ms(clock, behind_s=behind))
    driver.poll()
    c = driver.snapshot()[0]
    assert c.valid and c.age_ms < 20
    assert driver.stats.readings == 3
