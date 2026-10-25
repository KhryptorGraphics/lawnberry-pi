"""US3-02: the host driver never writes to a port that has not first sent a valid MCU frame.

``ultrasonic.port`` is a free string. A renumbered ``/dev/ttyACMn`` can be the RoboHAT's
CircuitPython port or this MCU's own REPL console, and every binary frame carries the console's
control bytes (the protocol version 0x01, the lengths 0x01/0x02 and, for some sequence numbers or
CRCs, 0x03 Ctrl-C and 0x04 Ctrl-D): one write to the wrong port stops the program on the other
end. The MCU announces itself unsolicited (INFO at boot and every second), so the host has no need
to speak first: it listens, and writes only once a valid frame has proved what is on the line.
"""

from __future__ import annotations

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as proto
from backend.src.drivers.sensors.jsn_sr04t_driver import JsnSr04tMcuDriver, LinkState
from backend.src.models.hardware_config import UltrasonicConfig
from tests.unit._usonic_fakes import FakeClock, FakeMcuFactory, SensorModel
from tests.unit.test_jsn_sr04t_driver import scripted_active, scripted_rig


class RecordingPort:
    """Whatever sits behind the configured path: it records every byte written to it."""

    def __init__(self, talks: bytes = b"") -> None:
        self.inbox = bytearray()
        self.talks = talks  # what this device volunteers on every read (a console prompt, say)
        self.written = bytearray()
        self.closed = False

    @property
    def in_waiting(self) -> int:
        self.inbox += self.talks
        return len(self.inbox)

    def read(self, size: int = 1) -> bytes:
        data = bytes(self.inbox[:size])
        del self.inbox[:size]
        return data

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def close(self) -> None:
        self.closed = True


def run_against(device_factory, seconds: float, **cfg):
    clock = FakeClock()
    ports: list[RecordingPort] = []

    def factory(path, baud):
        ports.append(device_factory())
        return ports[-1]

    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-renumbered", **cfg),
        serial_factory=factory, clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=False,
    )  # fmt: skip
    states = set()
    for _ in range(round(seconds / 0.02)):
        clock.advance(0.02)
        driver.poll()
        states.add(driver.link_state)
    return driver, ports, states


def test_a_port_that_never_answers_is_never_written_to_across_every_reconnect():
    # reviewer w1_wrongport: the old driver wrote HELLO at every open, and the 4th open's sequence
    # number made it carry 0x03 (Ctrl-C), the 2nd and 5th 0x04 (Ctrl-D)
    driver, ports, _ = run_against(RecordingPort, 30.0)
    assert len(ports) >= 5, "the driver should keep trying the port"
    assert [bytes(p.written) for p in ports] == [b""] * len(ports)
    assert driver.link_state is not LinkState.ACTIVE


CONSOLE = (
    b"Adafruit CircuitPython 9.2.1 on 2025-01-01; Raspberry Pi Pico with rp2040\r\n"
    b"code.py output:\r\nusonic: crashed, resetting: RuntimeError('x')\r\n>>> "
    b"\xa5\x5a\x01"  # even a stray sync word and a header start is not a frame
)


def test_a_repl_console_that_talks_is_read_but_never_written_to():
    driver, ports, states = run_against(lambda: RecordingPort(talks=CONSOLE), 12.0)
    assert len(ports) >= 3
    assert [bytes(p.written) for p in ports] == [b""] * len(ports)
    assert states <= {LinkState.HANDSHAKE, LinkState.DISCONNECTED}
    assert driver.stats.garbage_bytes > 0, "the text was read and counted as noise"
    assert driver.stats.frames_rx == 0


def test_corrupted_and_truncated_frames_do_not_identify_the_device():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    good = proto.encode_frame(proto.FrameType.INFO, 1, bytes(10 * [0]) + b"\x00\x00")  # CRC valid
    bad_crc = bytearray(proto.encode_frame(proto.FrameType.INFO, 2, bytes(12)))
    bad_crc[-1] ^= 0x40
    port.inbox += bytes(bad_crc) + good[: len(good) // 2]  # one corrupted, one cut short
    for _ in range(10):
        clock.advance(0.02)
        driver.poll()
    assert port.types() == []
    assert driver.stats.crc_errors >= 1 and driver.stats.frames_rx == 0


def test_the_first_frame_ever_written_is_the_config_and_only_after_a_valid_info():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    clock.advance(0.5)
    driver.poll()
    assert port.types() == []
    port.info()
    clock.advance(0.02)
    driver.poll()
    assert port.types() == [proto.FrameType.CONFIG]


def test_a_ping_is_refused_until_the_link_has_been_heard():
    clock, port, driver, _ = scripted_rig()
    driver.poll()
    assert driver.ping(7) is False and port.types() == []
    port.info()
    clock.advance(0.02)
    driver.poll()
    assert driver.ping(7) is True
    assert port.types() == [proto.FrameType.CONFIG, proto.FrameType.PING]


def test_the_transmit_sequence_restarts_with_every_connection():
    clock, port, driver, opens = scripted_rig()
    for connection in range(3):
        driver.poll()
        assert len(opens) == connection + 1
        port.info(boot_id=10 + connection)
        clock.advance(0.02)
        driver.poll()
        assert port.sent[-1].ftype == proto.FrameType.CONFIG
        assert port.sent[-1].seq == 0, f"connection {connection}: the sequence did not restart"
        port.broken = OSError("device reports readiness to read but returned no data")
        clock.advance(0.02)
        driver.poll()  # unplugged
        port.broken = None
        port.inbox.clear()
        clock.advance(6.0)  # longer than any reconnect backoff


def test_a_silent_identified_link_is_probed_once_a_second_not_once_a_poll():
    clock, port, driver, _ = scripted_active()
    port.sent.clear()
    for _ in range(150):  # the MCU stops talking: 3 s of silence
        clock.advance(0.02)
        driver.poll()
    probes = [f for f in port.sent if f.ftype == proto.FrameType.HELLO]
    assert 1 <= len(probes) <= 2, "one HELLO per second of silence, not one per poll"


@pytest.mark.parametrize("configured", [0.3, 1.0])
def test_a_short_handshake_timeout_still_waits_for_the_mcus_next_announcement(configured):
    # the host speaks only after hearing the MCU, which repeats INFO every second: a handshake
    # window under that would never complete against a link the MCU opened a moment too early
    clock, port, driver, opens = scripted_rig(handshake_timeout_s=configured)
    driver.poll()
    for _ in range(65):  # 1.3 s of silence: the heartbeat is one second plus its own jitter
        clock.advance(0.02)
        driver.poll()
    assert driver.link_state is LinkState.HANDSHAKE and len(opens) == 1
    port.info()
    clock.advance(0.02)
    driver.poll()
    port.ack(driver.expected_mcu_config())
    clock.advance(0.02)
    driver.poll()
    assert driver.link_state is LinkState.ACTIVE and driver.stats.link_losses == 0


class LateOpenFactory(FakeMcuFactory):
    """The MCU has been running for a while: the boot INFO went out before anyone listened."""

    def __call__(self, port, baudrate):
        mcu = super().__call__(port, baudrate)
        mcu.hal.tx.clear()
        return mcu


async def test_against_the_real_firmware_the_link_comes_up_on_the_next_heartbeat():
    clock = FakeClock()
    factory = LateOpenFactory(clock, sensors=[SensorModel(distance_cm=120.0)] * 4)
    driver = JsnSr04tMcuDriver(
        UltrasonicConfig(port="/dev/ttyACM-usonic"),
        serial_factory=factory, clock=clock.now, wall_clock=clock.wall, sleep=clock.sleep,
        simulate=False,
    )  # fmt: skip
    await driver.initialize()
    assert driver.link_state is LinkState.ACTIVE
    assert clock.now() - 1000.0 <= 1.3, "the heartbeat comes every second"
    written = proto.FrameDecoder().feed(b"".join(factory.current.written))
    assert [f.ftype for f in written] == [proto.FrameType.CONFIG]
    assert written[0].seq == 0
