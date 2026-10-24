"""The CircuitPython hardware layer of the ultrasonic MCU against FAKE CircuitPython modules.

This is NOT a hardware test. It runs ``usonic_hal_cp.CpHal`` and ``usonic_main.run`` under
hand-written stand-ins for ``board``, ``digitalio``, ``pulseio``, ``usb_cdc``, ``microcontroller``,
``watchdog`` and ``time``. It catches typos, wrong call order (TRIG pulse before the echo capture
is armed), unbounded waits, a console/data port mix-up and protocol wiring errors that would
otherwise only show up on the bench. The fakes encode the CircuitPython API as documented for 9.x
(``PulseIn(pin, maxlen, idle_state)``, ``pause/resume/clear``, ``usb_cdc.data``); anything the real
RP2040 build does differently (PulseIn timing, USB behaviour) is exactly what remains UNVERIFIED.
"""

from __future__ import annotations

import sys
import types

import pytest

from backend.src.drivers.sensors import ultrasonic_mcu_protocol as host
from tests.unit._usonic_fakes import FW_DIR, SPEED_OF_SOUND, load_firmware, load_module

TRIG = ("GP2", "GP4", "GP6", "GP8")
ECHO = ("GP3", "GP5", "GP7", "GP9")
STEP_NS = 50_000  # every time.monotonic_ns() call advances the fake clock by 50 us


class World:
    """Fake time plus the physics tying a TRIG pin to its ECHO PulseIn."""

    def __init__(self) -> None:
        self.now_ns = 0
        self.distance_cm: dict[int, float | None] = {0: 80.0, 1: 120.0, 2: 200.0, 3: 60.0}
        self.trig_log: list[tuple[str, int, bool]] = []  # (pin, time_ns, value) edges only
        self.echo_by_pin: dict[str, FakePulseIn] = {}
        self.pulsein_calls: list[tuple[str, str]] = []  # (pin, "resume"/"pause"/"clear") in order
        self.events: list[tuple[str, str, object]] = []  # everything above, in one global order

    def monotonic_ns(self) -> int:
        self.now_ns += STEP_NS
        return self.now_ns

    def on_trig(self, pin: str, value: bool) -> None:
        self.trig_log.append((pin, self.now_ns, value))
        self.events.append(("trig", pin, value))
        if value:
            return
        idx = TRIG.index(pin)
        cm = self.distance_cm.get(idx)
        if cm is not None:
            width_us = round(cm * 20000.0 / SPEED_OF_SOUND)
            self.echo_by_pin[ECHO[idx]].arrives(self.now_ns + 200_000, width_us)


class FakePin:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeBoard:
    def __getattr__(self, name: str) -> FakePin:
        if not name.startswith("GP"):
            raise AttributeError(name)
        return FakePin(name)


class FakeDigitalInOut:
    def __init__(self, pin: FakePin, world: World) -> None:
        self.pin = pin
        self.world = world
        self.direction = None
        self._value = False

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, v: bool) -> None:
        v = bool(v)
        changed = v != self._value
        self._value = v
        if changed:  # writing the level a pin already has is not an edge
            self.world.on_trig(self.pin.name, v)


class FakePulseIn:
    def __init__(self, pin: FakePin, maxlen: int, idle_state: bool, world: World) -> None:
        self.pin, self.maxlen, self.idle_state, self.world = pin, maxlen, idle_state, world
        self.paused = False
        self._pending: tuple[int, int] | None = None
        self._pulses: list[int] = []
        world.echo_by_pin[pin.name] = self

    def arrives(self, rise_ns: int, width_us: int) -> None:
        self._pending = (rise_ns, width_us)

    def _settle(self) -> None:
        if self._pending and not self.paused:
            rise_ns, width_us = self._pending
            if self.world.now_ns >= rise_ns + width_us * 1000:
                self._pulses.append(width_us)
                self._pending = None

    def pause(self) -> None:
        self.paused = True
        self.world.pulsein_calls.append((self.pin.name, "pause"))
        self.world.events.append(("echo", self.pin.name, "pause"))

    def resume(self, trigger_duration: int = 0) -> None:
        self.paused = False
        self.world.pulsein_calls.append((self.pin.name, "resume"))
        self.world.events.append(("echo", self.pin.name, "resume"))

    def clear(self) -> None:
        self._pulses.clear()
        self._pending = None
        self.world.pulsein_calls.append((self.pin.name, "clear"))
        self.world.events.append(("echo", self.pin.name, "clear"))

    def __len__(self) -> int:
        self._settle()
        return len(self._pulses)

    def __getitem__(self, i: int) -> int:
        return self._pulses[i]


class FakeUsbData:
    def __init__(self) -> None:
        self.rx = bytearray()
        self.written = bytearray()
        self.connected = True
        self.timeout = None
        self.write_timeout = None

    @property
    def in_waiting(self) -> int:
        return len(self.rx)

    def read(self, n: int) -> bytes:
        data = bytes(self.rx[:n])
        del self.rx[:n]
        return data

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)


class FakeWatchdog:
    def __init__(self) -> None:
        self.timeout = None
        self.mode = None
        self.fed = 0

    def feed(self) -> None:
        self.fed += 1


class FakeResetReason:
    POWER_ON = "power_on"
    WATCHDOG = "watchdog"
    SOFTWARE = "software"
    RESET_PIN = "reset_pin"
    BROWNOUT = "brownout"


@pytest.fixture
def cp(monkeypatch):
    """Inject fake CircuitPython modules, import the CP HAL, and clean up afterwards."""
    world = World()
    usb_data = FakeUsbData()
    wdt = FakeWatchdog()
    cpu = types.SimpleNamespace(reset_reason=FakeResetReason.POWER_ON)

    fakes = {
        "board": FakeBoard(),
        "digitalio": types.SimpleNamespace(
            DigitalInOut=lambda pin: FakeDigitalInOut(pin, world),
            Direction=types.SimpleNamespace(OUTPUT="output", INPUT="input"),
        ),
        "pulseio": types.SimpleNamespace(
            PulseIn=lambda pin, maxlen=2, idle_state=False: FakePulseIn(
                pin, maxlen, idle_state, world
            )
        ),
        "usb_cdc": types.SimpleNamespace(data=usb_data),
        "microcontroller": types.SimpleNamespace(
            watchdog=wdt, cpu=cpu, ResetReason=FakeResetReason
        ),
        "watchdog": types.SimpleNamespace(
            WatchDogMode=types.SimpleNamespace(RESET="reset", RAISE="raise")
        ),
    }
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)
    load_firmware()
    for name in ("usonic_hal_cp", "usonic_main", "usonic_config"):
        sys.modules.pop(name, None)
    load_module("usonic_config")
    hal_mod = load_module("usonic_hal_cp")
    load_module("usonic_main")
    # Replace the module's own reference only: the real ``time`` module must stay untouched.
    monkeypatch.setattr(hal_mod, "time", types.SimpleNamespace(monotonic_ns=world.monotonic_ns))
    ns = types.SimpleNamespace(
        world=world, usb=usb_data, wdt=wdt, cpu=cpu, hal_mod=hal_mod, fakes=fakes
    )
    yield ns
    for name in ("usonic_hal_cp", "usonic_main", "usonic_config"):
        sys.modules.pop(name, None)


def make_hal(cp, **kw):
    return cp.hal_mod.CpHal(TRIG, ECHO, **kw)


# ---- construction -----------------------------------------------------------------------------


def test_construction_sets_up_pins_link_and_watchdog(cp):
    hal = make_hal(cp, trigger_pulse_us=10, watchdog_s=4)
    assert hal.sensor_count == 4
    # TRIG: outputs, driven low; ECHO: PulseIn idle-low, capture paused and cleared
    assert [t.direction for t in hal._trig] == ["output"] * 4
    assert [t.value for t in hal._trig] == [False] * 4
    assert sorted(cp.world.echo_by_pin) == sorted(ECHO)
    assert all(p.idle_state is False and p.maxlen == 2 and p.paused for p in hal._echo)
    # the data CDC (not the console) carries the protocol, non-blocking
    assert hal._serial is cp.usb
    assert cp.usb.timeout == 0 and cp.usb.write_timeout == 0.05
    # the hardware watchdog resets a hung MCU
    assert cp.wdt.timeout == 4 and cp.wdt.mode == "reset"


def test_refuses_to_run_without_the_data_serial_port(cp):
    cp.fakes["usb_cdc"].data = None
    with pytest.raises(RuntimeError, match="boot.py"):
        make_hal(cp)


def test_mismatched_pin_lists_are_refused(cp):
    with pytest.raises(ValueError):
        cp.hal_mod.CpHal(TRIG, ECHO[:3])


@pytest.mark.parametrize(
    "reason,code",
    [
        (FakeResetReason.POWER_ON, 1),
        (FakeResetReason.WATCHDOG, 2),
        (FakeResetReason.SOFTWARE, 3),
        (FakeResetReason.RESET_PIN, 4),
        (FakeResetReason.BROWNOUT, 0),
    ],
)
def test_reset_reason_is_mapped_to_the_protocol_codes(cp, reason, code):
    hal = make_hal(cp)
    cp.cpu.reset_reason = reason
    assert hal.reset_reason() == code


# ---- measuring --------------------------------------------------------------------------------


def test_measure_returns_the_echo_width_and_arms_capture_only_after_the_trigger(cp):
    hal = make_hal(cp, trigger_pulse_us=10)
    status, width = hal.measure(1, 30_000)
    assert status == 0  # ST_OK
    assert width == round(120.0 * 20000 / SPEED_OF_SOUND)
    pin, echo_pin = TRIG[1], ECHO[1]
    edges = [e for e in cp.world.trig_log if e[0] == pin]
    assert [e[2] for e in edges] == [True, False]
    assert edges[1][1] - edges[0][1] >= 10_000  # TRIG held at least 10 us
    # Global order for this sensor: clean the capture, pulse TRIG, only THEN arm the capture,
    # then stop and clear it. (Arming before the pulse would let a stale edge in.)
    mine = [
        (kind, what)
        for kind, name, what in cp.world.events
        if name in (pin, echo_pin) and not (kind == "trig" and what is False and False)
    ][-7:]
    assert mine == [
        ("echo", "pause"),
        ("echo", "clear"),
        ("trig", True),
        ("trig", False),
        ("echo", "resume"),
        ("echo", "pause"),
        ("echo", "clear"),
    ]
    assert cp.world.echo_by_pin[echo_pin].paused  # not left capturing between measurements
    assert len(cp.world.echo_by_pin[echo_pin]) == 0  # and nothing stale left behind


def test_other_sensors_are_never_triggered_while_one_is_measured(cp):
    hal = make_hal(cp)
    hal.measure(2, 30_000)
    touched = {p for p, _t, _v in cp.world.trig_log}
    assert touched == {TRIG[2]}


def test_silent_sensor_times_out_within_the_window_not_forever(cp):
    cp.world.distance_cm[0] = None  # no echo ever
    hal = make_hal(cp)
    t0 = cp.world.now_ns
    status, width = hal.measure(0, 30_000)
    elapsed_us = (cp.world.now_ns - t0) / 1000
    assert (status, width) == (host.EchoStatus.NO_EDGE, 0)
    assert 30_000 <= elapsed_us <= 30_000 + 1_000  # window plus a few polling steps
    assert cp.world.echo_by_pin[ECHO[0]].paused


def test_an_echo_longer_than_the_window_is_not_reported_as_a_distance(cp):
    cp.world.distance_cm[3] = 600.0  # 35 ms round trip, window is 30 ms
    hal = make_hal(cp)
    assert hal.measure(3, 30_000) == (host.EchoStatus.NO_EDGE, 0)


def test_a_late_pulse_from_the_previous_measurement_is_discarded_by_the_next(cp):
    hal = make_hal(cp)
    cp.world.distance_cm[0] = 600.0
    assert hal.measure(0, 30_000)[0] == host.EchoStatus.NO_EDGE  # pulse still in flight
    cp.world.distance_cm[0] = 50.0
    status, width = hal.measure(0, 30_000)
    assert (status, width) == (host.EchoStatus.OK, round(50.0 * 20000 / SPEED_OF_SOUND))


# ---- host link --------------------------------------------------------------------------------


def test_read_is_non_blocking_and_write_never_blocks_on_a_detached_host(cp):
    hal = make_hal(cp)
    assert hal.read() == b""
    cp.usb.rx.extend(b"abc")
    assert hal.read() == b"abc" and hal.read() == b""
    hal.write(b"hello")
    assert bytes(cp.usb.written) == b"hello"
    cp.usb.connected = False
    hal.write(b"dropped")
    assert bytes(cp.usb.written) == b"hello"  # nothing was queued for a host that is not there


def test_watchdog_is_fed_through_the_hal(cp):
    hal = make_hal(cp)
    hal.feed_watchdog()
    hal.feed_watchdog()
    assert cp.wdt.fed == 2


# ---- the whole firmware loop on the fake board ------------------------------------------------


def _decode(cp):
    frames = host.FrameDecoder().feed(bytes(cp.usb.written))
    return [host.parse_payload(f) for f in frames]


def test_main_loop_end_to_end_reports_all_four_sensors_and_feeds_the_watchdog(cp, monkeypatch):
    usonic_config = sys.modules["usonic_config"]
    usonic_main = sys.modules["usonic_main"]

    monkeypatch.setattr(usonic_config, "TRIG_PINS", TRIG)
    monkeypatch.setattr(usonic_config, "ECHO_PINS", ECHO)
    usonic_main.run(iterations=12000)
    msgs = _decode(cp)
    infos = [m for m in msgs if isinstance(m, host.Info)]
    readings = [m for m in msgs if isinstance(m, host.Reading)]
    assert infos and infos[0].sensor_count == 4 and infos[0].proto_version == host.PROTO_VERSION
    assert {r.sensor for r in readings} == {0, 1, 2, 3}
    by_sensor = {r.sensor: r for r in readings}
    for idx, cm in {0: 80.0, 1: 120.0, 2: 200.0, 3: 60.0}.items():
        assert by_sensor[idx].status == host.EchoStatus.OK
        assert by_sensor[idx].echo_us == round(cm * 20000 / SPEED_OF_SOUND)
    assert cp.wdt.fed == 12000  # one feed per loop iteration


def test_main_loop_serves_a_host_config_and_a_silent_sensor_is_reported_silent(cp, monkeypatch):
    usonic_config = sys.modules["usonic_config"]
    usonic_main = sys.modules["usonic_main"]

    monkeypatch.setattr(usonic_config, "TRIG_PINS", TRIG)
    monkeypatch.setattr(usonic_config, "ECHO_PINS", ECHO)
    cp.world.distance_cm[2] = None
    cfg = host.McuConfig(enable_mask=0b1111, echo_window_ms=40, gap_ms=35, quiet_after=2)
    cp.usb.rx.extend(host.encode_frame(host.FrameType.CONFIG, 0, host.build_config(cfg)))
    usonic_main.run(iterations=30000)
    msgs = _decode(cp)
    assert cfg in [m for m in msgs if isinstance(m, host.McuConfig)]
    silent = [m for m in msgs if isinstance(m, host.Reading) and m.sensor == 2]
    assert silent and all(m.status == host.EchoStatus.NO_EDGE for m in silent)
    assert silent[-1].fails >= 2 and silent[-1].health == host.McuHealth.QUIET


def test_main_loop_without_pins_still_talks_and_fires_nothing(cp):
    usonic_main = sys.modules["usonic_main"]
    usonic_main.run(iterations=300)  # usonic_config ships TRIG_PINS = None
    msgs = _decode(cp)
    assert [m.sensor_count for m in msgs if isinstance(m, host.Info)][0] == 0
    assert not [m for m in msgs if isinstance(m, host.Reading)]
    assert cp.world.trig_log == []


def test_firmware_directory_is_a_plain_copy_to_circuitpy_layout():
    names = {p.name for p in FW_DIR.glob("*.py")}
    assert {"boot.py", "code.py", "usonic_proto.py", "usonic_core.py", "usonic_hal_cp.py"} <= names
    subdirs = [p for p in FW_DIR.iterdir() if p.is_dir() and p.name != "__pycache__"]
    assert not subdirs, "the CIRCUITPY copy must stay flat"
