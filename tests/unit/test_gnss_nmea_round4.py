"""NMEA parsing, line framing, the epoch clock and the accuracy floors (audit GPS-01 .. GPS-14).

Pure functions and small state machines; no port, no thread. The driver tests that put them
together are ``test_gnss_driver_round4.py``.
"""

from __future__ import annotations

import math

import pytest
from _gnss_fakes import gga, gst, nmea, rmc, ubx

from backend.src.drivers.sensors.gps_driver import (
    EpochClock,
    NmeaLineReader,
    classify,
    combine_accuracy,
    fix_status_from_quality,
    fix_status_from_rmc,
    nmea_checksum_ok,
    parse_gga,
    parse_nmea_coordinate,
    parse_nmea_utc,
    parse_rmc,
)
from backend.src.drivers.sensors.gps_driver import _Epoch as Epoch


def text(line: bytes) -> str:
    return line.decode().strip()


# --------------------------------------------------------------- coordinates (GPS-10)


@pytest.mark.parametrize(
    ("value", "hemi", "axis", "expected"),
    [
        ("4807.038", "N", "lat", 48.1173),
        ("4807.038", "S", "lat", -48.1173),
        ("01131.000", "E", "lon", 11.516666666666667),
        ("01131.000", "W", "lon", -11.516666666666667),  # a western longitude is negative
        ("4859.9999", "N", "lat", 48.99999833333333),  # 59.9999' is not rounded up to 49
        ("9000.000", "N", "lat", 90.0),  # the pole itself
        ("9000.000", "S", "lat", -90.0),
        ("18000.000", "E", "lon", 180.0),
        ("00000.000", "W", "lon", 0.0),
        ("0000.000", "N", "lat", 0.0),
    ],
)
def test_a_good_coordinate_field_is_converted_with_the_sign_of_its_hemisphere(
    value, hemi, axis, expected
):
    assert parse_nmea_coordinate(value, hemi, axis) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("value", "hemi", "axis"),
    [
        ("9959.999", "N", "lat"),  # latitude 99.99998
        ("9000.001", "N", "lat"),  # just past the pole
        ("4860.000", "N", "lat"),  # minutes 60: that is the next degree, never a coordinate
        ("4899.000", "N", "lat"),
        ("19959.999", "E", "lon"),  # longitude 199.99998
        ("18000.001", "E", "lon"),
        ("01160.000", "E", "lon"),
        ("4807.038", "E", "lat"),  # the longitude's hemisphere letter on the latitude
        ("01131.000", "N", "lon"),  # and the other way round
        ("4807.038", "n", "lat"),  # NMEA hemisphere letters are upper case
        ("4807.038", "x", "lat"),
        ("4807.038", "", "lat"),
        ("-4807.038", "N", "lat"),  # a sign is not NMEA
        ("+4807.038", "N", "lat"),
        ("4807.0e1", "N", "lat"),  # nor is an exponent (it read as 49.17 degrees)
        ("nan", "N", "lat"),
        ("inf", "N", "lat"),
        ("", "N", "lat"),
        ("4807", "N", "lat"),  # no decimal point
        ("480.7038", "N", "lat"),  # three digits before the point: the leading digit was lost
        ("48070.38", "N", "lat"),  # five digits on a latitude
        ("1131.000", "E", "lon"),  # four digits on a longitude
        ("4807.0380000000001", "N", "lat"),  # more digits than any receiver sends
    ],
)
def test_a_coordinate_field_that_is_not_a_coordinate_is_no_coordinate(value, hemi, axis):
    assert parse_nmea_coordinate(value, hemi, axis) is None


# ------------------------------------------------------------------------------ UTC field


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("120005.00", 43205.0),
        ("120005.0", 43205.0),
        ("120005", 43205.0),
        ("120005.25", 43205.25),
        ("000000.00", 0.0),
        ("235959.99", 86399.99),
        ("235960.00", 86400.0),  # a leap second is the 60th
    ],
)
def test_the_utc_field_is_seconds_of_the_day(value, seconds):
    assert parse_nmea_utc(value) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "12",
        "12:00:05",
        "240000.00",
        "126000.00",
        "120061.00",
        "-20005.0",
        "12000a.00",
        "120005.",
    ],
)
def test_a_utc_field_that_is_not_a_time_is_no_time(value):
    assert parse_nmea_utc(value) is None


# ------------------------------------------------------------------------------ GGA / RMC


def test_a_gga_is_read_field_by_field():
    g = parse_gga(
        text(gga(7, fix=4, alt="-12.5", age="1.5", station="0004", sats="17", hdop="0.6"))
    )
    assert g is not None
    assert g.utc_s == 43207.0 and g.quality == 4
    assert g.lat == pytest.approx(48.1173) and g.lon == pytest.approx(11.5166667)
    assert g.altitude_m == -12.5  # an altitude below sea level is a number
    assert g.satellites == 17 and g.hdop == 0.6 and g.correction_age_s == 1.5


def test_a_gga_too_short_to_carry_a_fix_quality_is_nothing():
    assert parse_gga("$GNGGA,120005.00,4807.038,N,01131.000,E*00") is None
    assert parse_gga("$GNGGA*00") is None


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("altitude_m", {"alt": "nan"}),
        ("altitude_m", {"alt": "inf"}),
        ("altitude_m", {"alt": ""}),
        ("hdop", {"hdop": "nan"}),
        ("hdop", {"hdop": "-1"}),
        ("hdop", {"hdop": "inf"}),
        ("satellites", {"sats": "x"}),
        ("satellites", {"sats": "-3"}),
        ("correction_age_s", {"age": "nan"}),
        ("correction_age_s", {"age": "-1"}),
        ("correction_age_s", {"age": ""}),
    ],
)
def test_a_numeric_gga_field_that_is_not_a_plain_number_is_unknown(field, kwargs):
    g = parse_gga(text(gga(1, **kwargs)))
    assert g is not None and getattr(g, field) is None
    assert g.lat is not None  # the rest of the sentence is still read


def test_a_latitude_without_its_longitude_is_no_position_at_all():
    g = parse_gga(text(gga(1, lon="01131.000", ew="N")))  # longitude with a latitude's hemisphere
    assert g is not None and g.lat is None and g.lon is None


@pytest.mark.parametrize("quality", ["x", "", "-1", "1.5", "²"])
def test_a_garbled_fix_quality_digit_is_unknown(quality):
    g = parse_gga(text(gga(1, fix=quality)))
    assert g is not None and g.quality is None


def test_an_rmc_is_read_field_by_field_and_knots_become_metres_per_second():
    r = parse_rmc(text(rmc(3, speed="10.0", course="271.5", mode="R")))
    assert r is not None and r.valid and r.mode == "R"
    assert r.speed_mps == pytest.approx(5.144444, rel=1e-6)  # 10 knots, not 19 m/s
    assert r.course_deg == 271.5 and r.utc_s == 43203.0
    assert r.lat == pytest.approx(48.1173)


@pytest.mark.parametrize(
    ("kwargs", "speed", "course"),
    [
        ({"speed": "-1"}, None, 84.4),
        ({"speed": "nan"}, None, 84.4),
        ({"speed": ""}, None, 84.4),
        ({"course": "361"}, 0.257222, None),
        ({"course": "-3"}, 0.257222, None),
        ({"course": "nan"}, 0.257222, None),
        ({"course": "360"}, 0.257222, 360.0),
        ({"course": "0"}, 0.257222, 0.0),
    ],
)
def test_an_rmc_speed_or_course_that_is_not_a_real_one_is_unknown(kwargs, speed, course):
    r = parse_rmc(text(rmc(1, **kwargs)))
    assert r is not None
    assert r.speed_mps == (pytest.approx(speed, rel=1e-5) if speed is not None else None)
    assert r.course_deg == course


def test_an_rmc_without_a_mode_indicator_has_none_and_a_void_one_is_not_valid():
    assert parse_rmc(text(rmc(1, mode=None))).mode is None
    assert not parse_rmc(text(rmc(1, status="V"))).valid
    assert parse_rmc("$GNRMC,120005.00*00") is None


# ------------------------------------------------------------------------------ fix status (GPS-07)


@pytest.mark.parametrize(
    ("quality", "status"),
    [
        (0, "NO_FIX"),
        (1, "GPS_FIX"),
        (2, "DGPS"),
        (4, "RTK_FIXED"),
        (5, "RTK_FLOAT"),
        (6, "DEAD_RECKONING"),
        (7, "MANUAL"),
        (8, "SIMULATION"),
        (3, "NO_FIX"),  # PPS: not a fix this machine can use
        (9, "NO_FIX"),
        (44, "NO_FIX"),
        (-1, "NO_FIX"),
        (None, "NO_FIX"),
    ],
)
def test_every_fix_quality_has_a_status_and_the_unknown_ones_are_no_fix(quality, status):
    assert fix_status_from_quality(quality) == status


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({}, "GPS_FIX"),
        ({"mode": None}, "GPS_FIX"),  # NMEA < 2.3: only the A of its status field
        ({"mode": "A"}, "GPS_FIX"),
        ({"mode": "P"}, "GPS_FIX"),
        ({"mode": "D"}, "DGPS"),
        ({"mode": "F"}, "RTK_FLOAT"),
        ({"mode": "R"}, "RTK_FIXED"),
        ({"mode": "E"}, "DEAD_RECKONING"),
        ({"mode": "M"}, "MANUAL"),
        ({"mode": "S"}, "SIMULATION"),
        ({"mode": "N"}, "NO_FIX"),
        ({"mode": "Z"}, "NO_FIX"),  # a letter nobody defined is not a fix
        ({"status": "V"}, "NO_FIX"),
        ({"status": "V", "mode": "A"}, "NO_FIX"),
        ({"status": "x", "mode": "A"}, "NO_FIX"),
    ],
)
def test_an_rmc_claims_the_status_of_its_mode_indicator_and_void_is_no_fix(kwargs, status):
    assert fix_status_from_rmc(parse_rmc(text(rmc(1, **kwargs)))) == status


# ------------------------------------------------------------------------------ classify


def epoch(*lines: bytes) -> Epoch:
    """An epoch built from sentences (the assembler does this on the driver's read path)."""
    e = Epoch(key=0.0, t_pos=0.0)
    for line in lines:
        s = text(line)
        if s[3:6] == "GGA":
            e.gga = parse_gga(s)
        elif s[3:6] == "RMC":
            e.rmc = parse_rmc(s)
    return e


def test_a_gga_with_no_fix_carries_no_position_even_if_it_has_the_last_known_one():
    got = classify(epoch(gga(1, fix=0), rmc(1)), 30.0)
    assert got.status == "NO_FIX" and not got.is_fix
    assert got.lat is None and got.lon is None and got.altitude_m is None


@pytest.mark.parametrize("quality", [0, 3, 6, 7, 8, 9, 44, "x", ""])
def test_no_quality_but_a_fix_quality_yields_a_position(quality):
    got = classify(epoch(gga(1, fix=quality)), 30.0)
    assert not got.is_fix and got.lat is None


@pytest.mark.parametrize("quality", [1, 2, 4, 5])
def test_a_fix_quality_with_a_valid_position_yields_it(quality):
    got = classify(epoch(gga(1, fix=quality, alt="12.5")), 30.0)
    assert got.is_fix and got.lat == pytest.approx(48.1173) and got.altitude_m == 12.5


@pytest.mark.parametrize(
    "position",
    [
        {"lat": "9959.999"},
        {"lat": "4860.000"},
        {"lon": "19959.999"},
        {"ew": "N"},
        {"lat": ""},
        {"lat": "", "lon": ""},
    ],
)
def test_a_fix_whose_coordinates_are_not_a_position_is_no_fix(position):
    got = classify(epoch(gga(1, fix=1, **position)), 30.0)
    assert got.status == "NO_FIX" and got.lat is None and got.lon is None


def test_an_rmc_alone_is_judged_by_its_own_status_and_mode():
    plain = classify(epoch(rmc(1, mode="A")), 30.0)
    assert plain.status == "GPS_FIX" and plain.lat == pytest.approx(48.1173)
    void = classify(epoch(rmc(1, status="V", mode="A")), 30.0)  # a last-known position with V
    assert void.status == "NO_FIX" and void.lat is None
    estimated = classify(epoch(rmc(1, mode="E")), 30.0)
    assert estimated.status == "DEAD_RECKONING" and estimated.lat is None
    assert classify(epoch(), 30.0).status == "NO_FIX"


def test_an_rmc_that_says_the_fix_is_not_valid_vetoes_a_gga_that_claims_one():
    assert classify(epoch(gga(1, fix=1), rmc(1, status="V")), 30.0).status == "NO_FIX"
    assert classify(epoch(gga(1, fix=4, age="1.0"), rmc(1, mode="S")), 30.0).status == "SIMULATION"
    # a plain RMC agrees with it; a lower RMC mode is no veto (the GGA is authoritative)
    assert classify(epoch(gga(1, fix=4, age="1.0"), rmc(1, mode="A")), 30.0).status == "RTK_FIXED"


def test_the_gga_is_authoritative_for_the_position_when_both_are_there():
    other = rmc(1, lat="4808.000")
    got = classify(epoch(gga(1, fix=1), other), 30.0)
    assert got.lat == pytest.approx(48.1173)  # not the RMC's 48.1333


@pytest.mark.parametrize(
    ("age", "typical"),
    [
        ("1.0", True),
        ("0.0", True),
        ("30.0", True),  # the limit itself still counts
        ("30.1", False),
        ("999.0", False),
        ("", False),  # RTK claimed by one digit and nothing behind it
        ("nan", False),
    ],
)
def test_the_typical_rtk_figure_needs_corrections_that_are_there_and_recent(age, typical):
    assert classify(epoch(gga(1, fix=4, age=age)), 30.0).typical_ok is typical
    assert classify(epoch(gga(1, fix=5, age=age)), 30.0).typical_ok is typical


def test_only_an_rtk_fix_gets_the_typical_figure_and_never_from_an_rmc():
    assert not classify(epoch(gga(1, fix=1, age="1.0")), 30.0).typical_ok
    assert not classify(epoch(gga(1, fix=2, age="1.0")), 30.0).typical_ok
    assert not classify(epoch(rmc(1, mode="R")), 30.0).typical_ok


def test_the_correction_age_limit_is_the_callers():
    assert classify(epoch(gga(1, fix=4, age="10.0")), 5.0).typical_ok is False
    assert classify(epoch(gga(1, fix=4, age="10.0")), 10.0).typical_ok is True


# ------------------------------------------------------------------------------ accuracy floors


@pytest.mark.parametrize(
    ("sigma", "status", "expected"),
    [
        (1e-9, "GPS_FIX", 0.5),  # a lying sigma on a plain fix
        (0.4, "GPS_FIX", 0.5),
        (2.0, "GPS_FIX", 2.0),
        (0.05, "DGPS", 0.3),
        (0.5, "DGPS", 0.5),
        (0.05, "RTK_FLOAT", 0.2),
        (0.3, "RTK_FLOAT", 0.3),
        (0.001, "RTK_FIXED", 0.03),
        (0.5, "RTK_FIXED", 0.5),
        (0.0001, None, 0.005),
        (0.0001, "SOMETHING", 0.005),
    ],
)
def test_no_fix_level_is_reported_more_accurate_than_its_floor(sigma, status, expected):
    assert combine_accuracy(sigma, status) == pytest.approx(expected)


def test_the_typical_rtk_figure_is_used_only_when_allowed_and_never_for_a_plain_fix():
    assert combine_accuracy(None, "RTK_FIXED") == 0.03
    assert combine_accuracy(None, "RTK_FLOAT") == 0.20
    assert combine_accuracy(None, "RTK_FIXED", assume_typical=False) is None
    assert combine_accuracy(None, "RTK_FLOAT", assume_typical=False) is None
    assert combine_accuracy(None, "GPS_FIX") is None and combine_accuracy(None, "DGPS") is None
    # a real sigma still counts with the typical figure switched off, floored by the level
    assert combine_accuracy(0.01, "RTK_FIXED", assume_typical=False) == 0.03
    assert combine_accuracy(0.4, "RTK_FIXED", assume_typical=False) == 0.4


def test_the_checksum_is_checked_on_the_text_after_the_last_dollar():
    good = text(gga(1))
    assert nmea_checksum_ok(good)
    assert not nmea_checksum_ok("junk" + good)  # the framing, not the checksum, removes junk


# --------------------------------------------------------- line framing (GPS-01, GPS-13)


def sentences(reader: NmeaLineReader, *chunks: bytes) -> list[str]:
    out: list[str] = []
    for chunk in chunks:
        out += reader.feed(chunk)
    return out


def test_crlf_lf_and_cr_alone_each_end_a_line():
    a, b, c = text(gga(1)), text(gst(1)), text(rmc(1))
    crlf = f"{a}\r\n{b}\r\n{c}\r\n".encode()
    assert sentences(NmeaLineReader(), crlf) == [a, b, c]
    assert sentences(NmeaLineReader(), f"{a}\n{b}\n{c}\n".encode()) == [a, b, c]
    assert sentences(NmeaLineReader(), f"{a}\r{b}\r{c}\r".encode()) == [a, b, c]


def test_a_sentence_split_across_reads_is_whole_when_its_end_arrives():
    line = gga(1)
    body = line[:-2]  # without its CR LF
    reader = NmeaLineReader()
    got: list[str] = []
    for i in range(0, len(body), 7):  # seven bytes at a time, as a slow UART delivers them
        got += reader.feed(body[i : i + 7])
    assert got == []  # the line end has not arrived
    got += reader.feed(line[-2:])
    assert got == [text(line)]


def test_junk_and_binary_frames_in_front_of_a_sentence_cost_it_nothing():
    frame = ubx(0x01, 0x07, bytes(range(92)))  # its payload holds 0x0A, 0x0D and 0x24 ('$')
    assert b"\n" in frame and b"\r" in frame and b"$" in frame
    good = gga(2)
    got = sentences(NmeaLineReader(), frame + good)
    assert got[-1] == text(good)
    assert all(nmea_checksum_ok(s) is False for s in got[:-1])  # the fragments are not sentences
    # a corrupted start of the same line and a stray dollar before the real one
    assert sentences(NmeaLineReader(), b"$GNG\x00\x01$" + good)[-1] == text(good)
    assert sentences(NmeaLineReader(), b"junk without a sentence start\r\n") == []


def test_a_stream_that_never_ends_a_line_is_bounded_and_the_next_sentence_is_found():
    reader = NmeaLineReader()
    for _ in range(2000):
        reader.feed(b"A" * 512)  # a megabyte with no CR and no LF
    assert len(reader._open) <= NmeaLineReader.MAX_LINE
    good = gga(3)
    assert sentences(reader, good) == [text(good)] or sentences(reader, good)[-1:] == [text(good)]


def test_a_sentence_that_starts_inside_a_runaway_line_is_recovered():
    reader = NmeaLineReader()
    reader.feed(b"B" * 1000)  # an open line far past the bound: nothing of it is kept
    assert len(reader._open) == 0
    good = gga(4)
    assert reader.feed(b"B" * 300 + good[:30]) == []  # the real sentence starts here ...
    assert 0 < len(reader._open) <= NmeaLineReader.MAX_LINE  # ... and its beginning is kept
    assert reader.feed(good[30:]) == [text(good)]


def test_a_line_longer_than_the_bound_is_not_a_sentence_even_with_a_line_end():
    reader = NmeaLineReader()
    long_line = b"$GNGGA," + b"9" * (NmeaLineReader.MAX_LINE + 50) + b"\r\n"
    assert reader.feed(long_line) == []
    good = gga(5)
    assert reader.feed(good) == [text(good)]


def test_a_non_ascii_byte_inside_a_line_does_not_break_the_framing():
    good = gga(6)
    mangled = good[:20] + b"\xff\xfe" + good[20:]
    got = NmeaLineReader().feed(mangled)
    assert len(got) == 1 and got[0].startswith("$GNGGA")  # decoded without them
    assert nmea_checksum_ok(got[0])  # the checksum covers the original text, not the noise


# --------------------------------------------------------- the epoch clock (GPS-02, 03)


def run(clock: EpochClock, steps):
    """Feed ``(u, t)`` pairs; the list of ``(accepted, round(delay, 3), reason)``."""
    out = []
    for u, t in steps:
        v = clock.accept(u, t)
        out.append((v.accepted, round(v.delay_s, 3), v.reason))
    return out


def test_a_receiver_that_keeps_time_is_accepted_without_delay():
    clock = EpochClock()
    got = run(clock, [(43200.0 + k, 100.0 + k) for k in range(20)])
    assert got[0] == (True, 0.0, "first")
    assert all(g == (True, 0.0, "ok") for g in got[1:])


def test_the_same_time_again_is_not_a_new_fix_however_long_it_keeps_coming():
    clock = EpochClock()
    assert clock.accept(43205.0, 100.0).accepted
    for k in range(1, 200):  # a frozen receiver repeats its last epoch for ever
        v = clock.accept(43205.0, 100.0 + k)
        assert (v.accepted, v.reason) == (False, "stale")
    assert clock.accept(43206.0, 300.0).accepted  # and the real next epoch is still taken


def test_an_older_time_after_a_newer_one_is_not_a_new_fix_and_moves_nothing():
    clock = EpochClock()
    run(clock, [(43200.0 + k, 100.0 + k) for k in range(3)])  # 43200 .. 43202
    assert not clock.accept(43140.0, 103.0).accepted  # a replayed minute-old epoch
    assert not clock.accept(43201.0, 104.0).accepted
    ok = clock.accept(43203.0, 104.0)  # the genuine stream goes on as if nothing happened
    assert (ok.accepted, ok.reason) == (True, "ok")


def test_the_midnight_wrap_is_forward_progress_and_a_wrap_backwards_is_not():
    clock = EpochClock()
    assert clock.accept(86399.8, 100.0).accepted
    v = clock.accept(0.0, 100.2)  # 00:00:00.0 of the next day
    assert (v.accepted, v.reason) == (True, "ok")
    assert clock.accept(0.2, 100.4).accepted
    back = EpochClock()
    assert back.accept(100.0, 50.0).accepted
    assert not back.accept(86399.0, 51.0).accepted  # 23:59:59 of the day before


def test_one_wild_epoch_is_held_back_and_cannot_move_the_reference():
    clock = EpochClock()
    run(clock, [(43200.0 + k, 100.0 + k) for k in range(3)])
    wild = clock.accept(43200.0 + 7200.0, 103.0)  # two hours ahead of real time
    assert (wild.accepted, wild.reason) == (False, "jump")
    # the genuine stream is judged against the OLD reference and goes straight on
    for k in range(3, 10):
        v = clock.accept(43200.0 + k, 100.0 + k)
        assert (v.accepted, v.reason) == (True, "ok"), k
    # and a second wild epoch much later is again only held back
    assert clock.accept(43200.0 + 7200.0, 110.0).reason == "jump"


def test_a_receiver_whose_clock_really_moved_forward_is_followed_after_one_epoch():
    clock = EpochClock()
    run(clock, [(43200.0 + k, 100.0 + k) for k in range(3)])
    first = clock.accept(50000.0, 103.0)
    assert (first.accepted, first.reason) == (False, "jump")
    second = clock.accept(50001.0, 104.0)  # continues from the jump at the host's rate
    assert (second.accepted, second.reason, second.delay_s) == (True, "rebased", 0.0)
    assert clock.accept(50002.0, 105.0).accepted  # and from then on it is the reference
    assert not clock.accept(43210.0, 106.0).accepted  # the old time line is now the stale one


def test_two_wild_epochs_that_do_not_continue_each_other_are_not_a_new_time_line():
    clock = EpochClock()
    run(clock, [(43200.0 + k, 100.0 + k) for k in range(3)])
    assert clock.accept(60000.0, 103.0).reason == "jump"
    assert clock.accept(70000.0, 104.0).reason == "jump"  # not 60000 + 1: a different wild value
    assert clock.accept(43203.0, 104.0).accepted  # the real stream is still there


def test_a_gap_of_real_silence_is_not_a_jump():
    clock = EpochClock()
    run(clock, [(43200.0 + k, 100.0 + k) for k in range(3)])
    v = clock.accept(43202.0 + 1800.0, 102.0 + 1800.0)  # the receiver was in a tunnel for 30 min
    assert (v.accepted, v.reason) == (True, "ok")


def test_an_epoch_read_late_is_dated_by_how_late_it_was_read():
    clock = EpochClock()
    got = run(
        clock,
        [
            (43200.0, 100.0),
            (43201.0, 101.0),
            (43202.0, 102.0),
            (43203.0, 103.9),  # the host was busy: this epoch was read 0.9 s after it arrived
            (43204.0, 104.0),  # and the next one on time again
        ],
    )
    assert [g[1] for g in got] == [0.0, 0.0, 0.0, 0.9, 0.0]
    assert all(g[0] for g in got)


def test_a_stream_that_runs_slower_than_real_time_shows_the_delay_it_builds_up():
    clock = EpochClock()
    delays = []
    for k in range(40):  # one receiver second per two host seconds
        v = clock.accept(43200.0 + k, 100.0 + 2.0 * k)
        assert v.accepted
        delays.append(v.delay_s)
    assert delays[0] == 0.0 and delays[-1] == pytest.approx(39.0, abs=0.1)
    assert delays == sorted(delays)


def test_crystal_drift_between_the_host_and_the_receiver_is_not_delay():
    clock = EpochClock()
    worst = 0.0
    for k in range(0, 3 * 3600, 10):  # three hours, an epoch every 10 s
        t = 100.0 + k * (1.0 + 100e-6)  # the host clock runs 100 ppm fast
        worst = max(worst, clock.accept(43200.0 + k, t).delay_s)
    assert worst < 0.05  # unchecked it would have grown to 1.08 s


def test_a_stream_kept_back_by_a_step_stays_late_for_hours_not_minutes():
    clock = EpochClock()
    for k in range(60):
        clock.accept(43200.0 + k, 100.0 + k)
    late = [clock.accept(43260.0 + k, 160.0 + k + 5.0).delay_s for k in range(0, 3600, 10)]
    assert late[0] == pytest.approx(5.0, abs=0.01) and late[-1] > 4.0  # not forgotten in an hour


def test_the_epoch_clock_of_absolute_times_has_no_midnight():
    clock = EpochClock(period_s=None)
    base = 1_761_393_600.0
    assert clock.accept(base, 10.0).accepted
    assert clock.accept(base + 1.0, 11.0).accepted
    assert not clock.accept(base - 86_400.0 + 1.0, 12.0).accepted  # yesterday is not a wrap
    assert not clock.accept(base + 1.0, 13.0).accepted


def test_nothing_finite_is_required_of_a_time_but_the_clock_never_returns_nan():
    clock = EpochClock()
    clock.accept(43200.0, 100.0)
    v = clock.accept(43201.0, 101.0)
    assert math.isfinite(v.delay_s) and v.delay_s >= 0.0


def test_a_gst_is_not_a_position_sentence():
    assert text(gst(1)).startswith("$GNGST") and nmea_checksum_ok(text(gst(1)))
    assert nmea("GNGGA,1") != b""
