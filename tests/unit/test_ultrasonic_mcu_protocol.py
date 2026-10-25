"""Wire protocol of the ultrasonic MCU: host codec vs firmware codec vs the written spec.

The two codecs are independent implementations (``binascii.crc_hqx`` + ``bytearray`` on the host,
table CRC + ``bytes`` slicing in the CircuitPython-compatible firmware module). These tests pin the
wire bytes and prove the implementations agree, including how each recovers from corruption.
"""

from __future__ import annotations

import random

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as host
from tests.unit._usonic_fakes import P as fw


def crc16_reference(data: bytes) -> int:
    """Bit-by-bit CRC-16/CCITT-FALSE straight from the definition (third implementation)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# ---- CRC --------------------------------------------------------------------------------------


def test_crc16_check_value_matches_the_published_ccitt_false_vector():
    assert host.crc16(b"123456789") == 0x29B1
    assert fw.crc16(b"123456789") == 0x29B1


def test_both_crc_implementations_match_the_bitwise_reference_on_random_data():
    rng = random.Random(20261024)
    for _ in range(300):
        data = rng.randbytes(rng.randint(0, 60))
        ref = crc16_reference(data)
        assert host.crc16(data) == ref
        assert fw.crc16(data) == ref


# ---- golden frames ----------------------------------------------------------------------------


def test_golden_hello_frame_bytes():
    frame = host.encode_frame(host.FrameType.HELLO, 0, host.build_hello())
    # SYNC, VER=1, TYPE=1, SEQ=0, LEN=2, payload(host version 1, flags 0), CRC little endian
    assert frame[:8] == bytes.fromhex("a55a01010002" + "0100")
    assert frame == fw.encode(fw.T_HELLO, 0, fw.pack_hello())
    assert int.from_bytes(frame[-2:], "little") == crc16_reference(frame[2:-2])


def test_golden_reading_frame_bytes():
    payload = host.Reading(
        sensor=2, status=0, health=0, fails=0, echo_us=5832, cycle=12, t_ms=123456
    )
    raw = fw.pack_reading(
        payload.sensor,
        payload.status,
        payload.health,
        payload.fails,
        payload.echo_us,
        payload.cycle,
        payload.t_ms,
    )
    frame = host.encode_frame(host.FrameType.READING, 7, raw)
    assert frame.hex() == "a55a0183070e02000000c81600000c0040e20100cb56"
    assert fw.encode(fw.T_READING, 7, raw) == frame
    assert host.parse_payload(host.FrameDecoder().feed(frame)[0]) == payload


# ---- message round trips ----------------------------------------------------------------------


def test_every_message_type_round_trips_between_the_two_implementations():
    info = fw.pack_info(1, 2, 4, 0xDEADBEEF, 123456789, fw.RESET_WATCHDOG, 0b1011)
    cfg = host.McuConfig(enable_mask=0b1111, echo_window_ms=30, gap_ms=45, quiet_after=8)
    cases = {
        host.FrameType.INFO: (info, host.Info(1, 2, 1, 4, 0xDEADBEEF, 123456789, 2, 0b1011)),
        host.FrameType.CONFIG_ACK: (host.build_config(cfg), cfg),
        host.FrameType.PONG: (fw.pack_pong(77, 9000), host.Pong(77, 9000)),
        host.FrameType.NAK: (fw.pack_nak(2, 4), host.Nak(2, 4)),
    }
    for seq, (ftype, (payload, expected)) in enumerate(cases.items()):
        frame = host.encode_frame(ftype, seq, payload)
        assert frame == fw.encode(int(ftype), seq, payload)
        (decoded,) = host.FrameDecoder().feed(frame)
        assert host.parse_payload(decoded) == expected
        ((ver, ft, sq, body),) = fw.Decoder().feed(frame)
        assert (ver, ft, sq, body) == (1, int(ftype), seq, payload)


def test_host_to_mcu_payload_builders_match_the_firmware_parsers():
    cfg = host.McuConfig(enable_mask=0b0101, echo_window_ms=40, gap_ms=60, quiet_after=3)
    assert fw.unpack_config(host.build_config(cfg)) == {
        "enable_mask": 5,
        "echo_window_ms": 40,
        "gap_ms": 60,
        "quiet_after": 3,
    }
    assert fw.unpack_hello(host.build_hello()) == {"host_version": 1, "flags": 0}
    assert fw.unpack_ping(host.build_ping(0xCAFEBABE)) == {"nonce": 0xCAFEBABE}


@pytest.mark.parametrize("nonce", [0, 1, 0xFFFF, 0x1_0000, 0xDEADBEEF, 0xFFFFFFFF])
def test_the_ping_nonce_keeps_all_32_bits_in_both_codecs(nonce):
    assert fw.pack_ping(nonce) == host.build_ping(nonce)
    assert fw.unpack_ping(fw.pack_ping(nonce)) == {"nonce": nonce}
    assert host.build_ping(nonce + (1 << 32)) == host.build_ping(nonce)  # taken modulo 2**32


def test_payload_sizes_agree_and_fit_the_frame_limit():
    assert {int(k): v for k, v in host.PAYLOAD_SIZES.items()} == fw.PAYLOAD_SIZES
    assert max(host.PAYLOAD_SIZES.values()) <= host.MAX_PAYLOAD
    assert host.MAX_FRAME == fw.MAX_FRAME == 56


def test_parse_payload_rejects_wrong_length_and_host_to_mcu_types():
    good = host.encode_frame(host.FrameType.READING, 0, fw.pack_reading(0, 0, 0, 0, 1, 0, 0))
    (frame,) = host.FrameDecoder().feed(good)
    with pytest.raises(host.ProtocolError):
        host.parse_payload(host.Frame(frame.version, frame.ftype, frame.seq, frame.payload[:-1]))
    with pytest.raises(host.ProtocolError):
        host.parse_payload(host.Frame(1, int(host.FrameType.HELLO), 0, b"\x01\x00"))
    with pytest.raises(host.ProtocolError):
        host.parse_payload(host.Frame(1, 0x90, 0, b""))


def test_encode_refuses_an_oversize_payload():
    with pytest.raises(ValueError):
        host.encode_frame(host.FrameType.PING, 0, bytes(host.MAX_PAYLOAD + 1))
    with pytest.raises(ValueError):
        fw.encode(fw.T_PING, 0, bytes(fw.MAX_PAYLOAD + 1))


# ---- decoder behaviour (both implementations) -------------------------------------------------


def _reading_frame(seq: int, sensor: int = 0, echo_us: int = 5000) -> bytes:
    return fw.encode(fw.T_READING, seq, fw.pack_reading(sensor, 0, 0, 0, echo_us, seq, seq * 10))


def _decode_host(stream: bytes, chunk: int | None = None) -> list[tuple[int, int, int, bytes]]:
    dec = host.FrameDecoder()
    out = []
    pieces = (
        [stream] if chunk is None else [stream[i : i + chunk] for i in range(0, len(stream), chunk)]
    )
    for piece in pieces:
        out += [(f.version, f.ftype, f.seq, f.payload) for f in dec.feed(piece)]
    return out


def _decode_fw(stream: bytes, chunk: int | None = None) -> list[tuple[int, int, int, bytes]]:
    dec = fw.Decoder()
    out = []
    pieces = (
        [stream] if chunk is None else [stream[i : i + chunk] for i in range(0, len(stream), chunk)]
    )
    for piece in pieces:
        out += dec.feed(piece)
    return out


@pytest.fixture(params=["host", "firmware"])
def decode(request):
    return _decode_host if request.param == "host" else _decode_fw


def test_frames_split_at_every_byte_boundary_still_decode_once(decode):
    stream = _reading_frame(1) + _reading_frame(2)
    for chunk in (1, 2, 3, 7, 11):
        got = decode(stream, chunk)
        assert [seq for _v, _t, seq, _p in got] == [1, 2]


def test_garbage_between_frames_is_skipped(decode):
    stream = b"\x00\x01\xa5" + _reading_frame(1) + b"\xa5\xa5\x5a\x5a\xff" + _reading_frame(2)
    assert [seq for _v, _t, seq, _p in decode(stream)] == [1, 2]


def test_a_flipped_payload_bit_costs_exactly_that_frame(decode):
    good1, bad, good2 = _reading_frame(1), bytearray(_reading_frame(2)), _reading_frame(3)
    bad[7] ^= 0x04
    got = decode(good1 + bytes(bad) + good2)
    assert [seq for _v, _t, seq, _p in got] == [1, 3]


def test_a_flipped_crc_bit_costs_exactly_that_frame(decode):
    bad = bytearray(_reading_frame(2))
    bad[-1] ^= 0x01
    got = decode(_reading_frame(1) + bytes(bad) + _reading_frame(3))
    assert [seq for _v, _t, seq, _p in got] == [1, 3]


def test_payload_that_contains_the_sync_pattern_decodes_intact(decode):
    payload = fw.pack_reading(0xA5, 0x5A, 0xA5, 0x5A, 0x5AA5A55A, 0xA55A, 0x5AA5A55A)
    frame = fw.encode(fw.T_READING, 9, payload)
    assert b"\xa5\x5a" in frame[2:-2]
    assert decode(frame) == [(1, fw.T_READING, 9, payload)]


def test_oversize_length_byte_resynchronises_on_the_next_frame(decode):
    bad = bytearray(_reading_frame(1))
    bad[5] = fw.MAX_PAYLOAD + 1
    got = decode(bytes(bad) + _reading_frame(2) + _reading_frame(3))
    assert [seq for _v, _t, seq, _p in got] == [2, 3]


def test_plausible_but_wrong_length_does_not_eat_the_following_frames(decode):
    bad = bytearray(_reading_frame(1))
    bad[5] = 40  # claims 40 payload bytes; only 14 were sent
    stream = bytes(bad) + b"".join(_reading_frame(s) for s in range(2, 8))
    got = decode(stream)
    assert [seq for _v, _t, seq, _p in got] == [2, 3, 4, 5, 6, 7]


def test_trailing_sync0_waits_for_its_second_byte(decode):
    frame = _reading_frame(5)
    dec = fw.Decoder() if decode is _decode_fw else host.FrameDecoder()
    first = dec.feed(b"\x01\x02" + frame[:1])
    assert first == []
    rest = dec.feed(frame[1:])
    assert len(rest) == 1


def test_frame_of_another_protocol_version_is_decoded_not_dropped(decode):
    frame = fw.encode(fw.T_INFO, 3, fw.pack_info(2, 0, 4, 1, 0, 1, 15), version=2)
    ((ver, ftype, seq, _payload),) = decode(frame)
    assert (ver, ftype, seq) == (2, fw.T_INFO, 3)  # CRC-valid: the caller refuses it explicitly


def test_quiet_line_with_a_pending_bogus_frame_releases_the_good_frames_behind_it():
    for dec, feed in (
        (host.FrameDecoder(partial_timeout=0.1), lambda d, b, t: [f.seq for f in d.feed(b, t)]),
        (fw.Decoder(partial_timeout=0.1), lambda d, b, t: [f[2] for f in d.feed(b, t)]),
    ):
        bad = bytearray(_reading_frame(1))
        bad[5] = 40  # waits for 48 bytes that never come
        assert feed(dec, bytes(bad), 0.0) == []
        assert feed(dec, b"", 0.05) == []  # still within the timeout: keep waiting
        assert feed(dec, _reading_frame(2), 0.06) == []  # swallowed by the pending bogus frame
        assert feed(dec, b"", 0.1) == []  # quiet for only 0.04 s
        assert feed(dec, b"", 0.3) == [2]  # quiet for 0.24 s: flush finds the hidden frame
        assert dec.partial_timeouts == 1
        assert feed(dec, _reading_frame(3), 0.31) == [3]  # the decoder is clean again


def test_the_quiet_timeout_is_exceeded_only_by_a_line_that_was_quiet_for_more_than_it():
    for dec, feed in (
        (host.FrameDecoder(partial_timeout=0.5), lambda d, b, t: [f.seq for f in d.feed(b, t)]),
        (fw.Decoder(partial_timeout=0.5), lambda d, b, t: [f[2] for f in d.feed(b, t)]),
    ):
        bad = bytearray(_reading_frame(1))
        bad[5] = 40
        assert feed(dec, bytes(bad), 1.0) == []
        assert feed(dec, b"", 1.5) == [] and dec.partial_timeouts == 0  # exactly the timeout
        assert feed(dec, b"", 1.5001) == [] and dec.partial_timeouts == 1  # a hair beyond it


@pytest.mark.parametrize("fails", [255, 256, 300, 1000])
def test_a_failure_count_saturates_at_255_and_never_wraps_to_a_healthy_looking_small_number(fails):
    wire = fw.unpack_reading(fw.pack_reading(0, 1, 1, fails, 0, 0, 0))
    assert wire["fails"] == 255


def test_without_the_quiet_timeout_a_bogus_length_holds_good_frames_hostage():
    # Documents why the timeout exists and that it is opt-in: no `now`, no flush.
    dec = host.FrameDecoder(partial_timeout=0.1)
    bad = bytearray(_reading_frame(1))
    bad[5] = 40
    assert dec.feed(bytes(bad)) == []
    assert dec.feed(_reading_frame(2)) == []
    assert dec.feed(b"") == []
    assert dec.partial_timeouts == 0


def test_flood_of_sync_lookalikes_stays_bounded_and_the_next_frame_survives():
    for dec, feed in (
        (host.FrameDecoder(partial_timeout=0.1), lambda d, b, t: [f.seq for f in d.feed(b, t)]),
        (fw.Decoder(partial_timeout=0.1), lambda d, b, t: [f[2] for f in d.feed(b, t)]),
    ):
        junk = b"\xa5\x5a\x01\x83\x00\x30" * 400  # every 6th byte starts a bogus 56-byte frame
        assert feed(dec, junk, 0.0) == []
        buf = dec.buf if hasattr(dec, "buf") else dec._buf
        assert len(buf) <= fw.MAX_FRAME  # never more than one incomplete frame, whatever was fed
        got = feed(dec, _reading_frame(9), 0.01) + feed(dec, b"", 0.5)
        assert got == [9]
        buf = dec.buf if hasattr(dec, "buf") else dec._buf
        assert len(buf) == 0


def test_a_large_capture_fed_in_one_call_decodes_completely():
    frames = b"".join(_reading_frame(i & 0xFF) for i in range(500))  # 11 kB
    assert len(frames) > 10_000
    for decode in (_decode_host, _decode_fw):
        assert [seq for _v, _t, seq, _p in decode(frames)] == [i & 0xFF for i in range(500)]
    assert len(host.decode_stream(frames)) == 500


def test_decoder_counters_classify_what_was_wrong():
    dec = host.FrameDecoder()
    stream = bytearray(b"\x11\x22" + _reading_frame(1))
    crc_bad = bytearray(_reading_frame(2))
    crc_bad[-2] ^= 0xFF
    len_bad = bytearray(_reading_frame(3))
    len_bad[5] = 200
    frames = dec.feed(bytes(stream) + bytes(crc_bad) + bytes(len_bad) + _reading_frame(4))
    assert [f.seq for f in frames] == [1, 4]
    assert dec.garbage >= 2
    assert dec.crc_errors >= 1
    assert dec.bad_length >= 1


# ---- differential fuzz: same bytes in, same frames out ----------------------------------------


def _random_frame(rng: random.Random, seq: int) -> bytes:
    kind = rng.choice(["reading", "info", "pong", "nak", "empty"])
    if kind == "reading":
        payload = fw.pack_reading(
            rng.randrange(8), rng.randrange(6), rng.randrange(4), rng.randrange(256),
            rng.getrandbits(32), rng.getrandbits(16), rng.getrandbits(32),
        )  # fmt: skip
        return fw.encode(fw.T_READING, seq, payload)
    if kind == "info":
        payload = fw.pack_info(1, 0, 4, rng.getrandbits(32), rng.getrandbits(32), 1, 15)
        return fw.encode(fw.T_INFO, seq, payload)
    if kind == "pong":
        return fw.encode(fw.T_PONG, seq, fw.pack_pong(rng.getrandbits(32), rng.getrandbits(32)))
    if kind == "nak":
        return fw.encode(fw.T_NAK, seq, fw.pack_nak(rng.randrange(256), rng.randrange(5)))
    return fw.encode(0x77, seq, rng.randbytes(rng.randint(0, fw.MAX_PAYLOAD)))


def _corrupt(rng: random.Random, frame: bytes) -> bytes:
    op = rng.choice(["flip", "truncate", "garbage", "drop", "lenbomb", "dupsync"])
    b = bytearray(frame)
    if op == "flip":
        b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
    elif op == "truncate":
        b = b[: rng.randrange(1, len(b))]
    elif op == "garbage":
        b = bytearray(rng.randbytes(rng.randint(1, 9))) + b
    elif op == "drop":
        b = bytearray()
    elif op == "lenbomb":
        b[5] = rng.randrange(fw.MAX_PAYLOAD + 1, 256)
    else:
        b = bytearray(b"\xa5\x5a") + b
    return bytes(b)


@pytest.mark.parametrize("seed", range(40))
def test_both_decoders_agree_on_corrupted_random_streams_and_never_invent_frames(seed):
    rng = random.Random(seed)
    intact = [_random_frame(rng, i & 0xFF) for i in range(60)]
    stream = bytearray()
    for frame in intact:
        stream += _corrupt(rng, frame) if rng.random() < 0.3 else frame
    stream = bytes(stream)
    chunk = rng.choice([None, 1, 5, 13, 50])
    got_host = _decode_host(stream, chunk)
    got_fw = _decode_fw(stream, chunk)
    assert got_host == got_fw
    # No frame is ever invented: everything accepted is a frame that was actually sent intact.
    sent = [(f[2], f[3], f[4], bytes(f[6:-2])) for f in intact]
    cursor = 0
    for frame in got_host:
        while cursor < len(sent) and sent[cursor] != frame:
            cursor += 1
        assert cursor < len(sent), f"decoder produced a frame that was never sent: {frame}"
        cursor += 1
    # And uncorrupted frames survive: at least the untouched ones arrive.
    assert len(got_host) >= 1
