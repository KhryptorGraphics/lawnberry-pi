"""Four JSN-SR04T ultrasonic sensors behind a dedicated USB microcontroller: host driver.

Architecture (audit B.1.4 option B, C.7, WP-P2-02): the MCU (``firmware/ultrasonic-mcu/``) owns the
TRIG/ECHO timing and fires the four sensors one at a time. The Pi talks to it over USB serial with
a framed, versioned, CRC-protected protocol (``ultrasonic_mcu_protocol``), so **no Raspberry Pi
header pin is used**. Master switch: ``ultrasonic_enabled`` in ``config/limits.yaml`` (ships
``false``). Nothing here has been run on hardware.

This module decides what a reading MEANS. The MCU only reports what the echo line did.

Fail-closed rules (all tested with a fake serial port and the real firmware core):

* A channel is CLEAR only when it has a fresh in-range echo beyond the stop threshold, or when the
  owner's policy explicitly says that a silent / out-of-range channel is clear (and, for silence,
  the channel was demonstrably alive). The shipped policy says neither.
* Stale data, a blind-zone echo, no echo, a fault, a link that is down, a version or channel-count
  mismatch, a rejected configuration and a channel never heard from are all UNKNOWN, never clear.
* An echo closer than ``min_range_cm`` is the blind zone: UNKNOWN, never a distance and never clear.
* Disabled channels (not fitted) are excluded from the aggregate; a channel the host expects but
  the MCU reports disabled is a FAULT, not "excluded".
* Silence right after a NEAR echo (closer than ``NEAR_ECHO_FACTOR`` times the stop threshold) is
  the dead zone, not open air: BLIND_ZONE / UNKNOWN for ``near_hold_s`` or until a far echo is
  heard, whatever the ``*_is_clear`` policy flags say. Only a complete, non-near echo pulse is
  proof of life (never NO_FALL, never a 0 us width).

Every number in :class:`~backend.src.models.hardware_config.UltrasonicConfig` is an unvalidated
placeholder until audit measurement M-40.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from ...core.simulation import is_simulation_mode
from ...models.hardware_config import UltrasonicConfig
from ..base import HardwareDriver
from . import ultrasonic_mcu_protocol as proto

logger = logging.getLogger(__name__)

#: Silent measurements before the MCU labels a sensor QUIET (the MCU still fires it every cycle).
DEFAULT_QUIET_AFTER = 8
#: Interval of the background service loop.
POLL_INTERVAL_S = 0.02
#: Waits between reconnect attempts after consecutive failures.
RECONNECT_BACKOFF_S = (0.25, 0.5, 1.0, 2.0, 5.0)
#: How often a silent link is probed with HELLO.
HELLO_RETRY_S = 1.0
#: A half-received frame older than this is abandoned (resync).
PARTIAL_FRAME_TIMEOUT_S = 0.1
#: An echo longer than this cannot be physical (1 s of flight is about 170 m).
MAX_SANE_ECHO_US = 1_000_000
#: An echo closer than this many stop thresholds (or blind-zone lines, whichever is larger) is
#: NEAR: if the channel then goes silent, the object may have entered the sensor's dead zone.
NEAR_ECHO_FACTOR = 2.0
#: Seconds between simulated measurement cycles.
SIM_CYCLE_S = 0.12
_SIM_BASE_CM = (150.0, 200.0, 200.0, 180.0)


class SerialTransport(Protocol):
    """The slice of ``serial.Serial`` the driver uses (non-blocking: timeout 0)."""

    @property
    def in_waiting(self) -> int: ...

    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int | None: ...

    def close(self) -> None: ...


SerialFactory = Callable[[str, int], SerialTransport]


def open_pyserial(port: str, baudrate: int) -> SerialTransport:
    """Default transport. ``pyserial`` is imported here, never at module import."""
    import serial  # type: ignore

    return serial.Serial(port=port, baudrate=baudrate, timeout=0, write_timeout=0.2)


class LinkState(StrEnum):
    DISCONNECTED = "disconnected"  # no open port (not tried yet, failed, or lost)
    HANDSHAKE = "handshake"  # port open, waiting for INFO / the CONFIG echo
    ACTIVE = "active"  # protocol and configuration verified; readings are accepted
    FAULT = "fault"  # the MCU answers but is refused: version, channel count, configuration


class ChannelStatus(StrEnum):
    OK = "ok"  # fresh echo inside [min_range, max_range]
    STALE = "stale"
    #: an echo closer than min_range, or silence right after a NEAR echo (the dead zone): either
    #: way something may be right in front
    BLIND_ZONE = "blind_zone"
    NO_ECHO = "no_echo"  # silent line
    OUT_OF_RANGE = "out_of_range"  # pulse beyond max range (or longer than the echo window)
    FAULT = "fault"
    DISABLED = "disabled"  # configured not fitted
    LINK_DOWN = "link_down"
    NO_DATA = "no_data"  # nothing heard yet on this channel


class Verdict(StrEnum):
    CLEAR = "clear"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class Health(StrEnum):
    OK = "ok"  # newest real echo pulse is a far one, not older than alive_window_s: proof of life
    SILENT = "silent"  # no pulse inside alive_window_s: open air or dead, indistinguishable
    STALE = "stale"
    FAULT = "fault"
    LINK_DOWN = "link_down"
    NO_DATA = "no_data"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ChannelReading:
    """One channel as seen at one instant. ``as_dict`` is the stable external contract."""

    sensor_id: str
    index: int
    status: ChannelStatus
    valid: bool  # True only for a fresh in-range echo
    distance_cm: float  # 0.0 unless valid
    echo_us: int | None
    timestamp: datetime | None  # host receive time of the measurement
    age_ms: int | None
    health: Health
    alive: bool
    verdict: Verdict
    clear_to_cm: float | None
    fails: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "index": self.index,
            "distance_cm": self.distance_cm,
            "valid": self.valid,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "status": self.status.value,
            "age_ms": self.age_ms,
            "health": self.health.value,
            "alive": self.alive,
            "verdict": self.verdict.value,
            "clear_to_cm": self.clear_to_cm,
            "echo_us": self.echo_us,
            "fails": self.fails,
        }


@dataclass(slots=True)
class _Channel:
    index: int
    sensor_id: str
    enabled: bool
    status: proto.EchoStatus | None = None
    echo_us: int = 0
    mcu_health: int = 0
    fails: int = 0
    cycle: int = 0
    t_ms: int = 0
    measured_at: float | None = None  # host monotonic time the last real measurement arrived
    measured_wall: datetime | None = None
    pulse_at: float | None = None  # host monotonic time of the last real echo pulse
    #: Host monotonic time of the last real echo that was NEAR (closer than twice the stop
    #: threshold), None once a far echo has been heard. Deliberately NOT cleared by ``clear()``:
    #: an object that was right in front before a link blip is still right in front after it.
    near_at: float | None = None

    def clear(self) -> None:
        self.status = None
        self.echo_us = 0
        self.mcu_health = 0
        self.fails = 0
        self.measured_at = None
        self.measured_wall = None
        self.pulse_at = None


@dataclass(slots=True)
class LinkStats:
    frames_rx: int = 0
    readings: int = 0
    readings_ignored: int = 0
    infos: int = 0
    naks: int = 0
    pongs: int = 0
    frames_lost: int = 0  # from SEQ gaps
    bad_payload: int = 0
    bad_readings: int = 0
    connect_failures: int = 0
    link_losses: int = 0
    mcu_resets: int = 0
    crc_errors: int = 0
    bad_length: int = 0
    garbage_bytes: int = 0
    partial_timeouts: int = 0


def echo_to_cm(echo_us: float, speed_of_sound_mps: float) -> float:
    """Round-trip echo width to one-way distance: ``echo_s * c / 2`` in centimetres."""
    return echo_us * speed_of_sound_mps / 20000.0


def cm_to_echo_us(distance_cm: float, speed_of_sound_mps: float) -> int:
    return round(distance_cm * 20000.0 / speed_of_sound_mps)


def evaluate_channel(
    cfg: UltrasonicConfig,
    ch: _Channel,
    now: float,
    link_ok: bool,
    threshold_cm: float,
) -> ChannelReading:
    """Turn what the MCU last said about one channel into status, validity and verdict."""
    age_s = None if ch.measured_at is None else max(0.0, now - ch.measured_at)
    age_ms = None if age_s is None else int(age_s * 1000)
    # Proof of life: the newest real echo pulse is recent and was NOT a near one (a pulse from
    # something right in front says nothing about open air). NO_FALL and 0 us are no echo at all.
    alive = (
        ch.pulse_at is not None
        and ch.near_at is None
        and (now - ch.pulse_at) <= cfg.alive_window_s
    )
    # Silence right after a NEAR echo is the dead zone talking, not open air (US-01).
    near_hold = ch.near_at is not None and (now - ch.near_at) <= cfg.near_hold_s

    def make(status, health, verdict, *, valid=False, dist=0.0, clear_to=None):
        return ChannelReading(
            sensor_id=ch.sensor_id,
            index=ch.index,
            status=status,
            valid=valid,
            distance_cm=dist if valid else 0.0,
            echo_us=ch.echo_us if ch.measured_at is not None else None,
            timestamp=ch.measured_wall,
            age_ms=age_ms,
            health=health,
            alive=alive and link_ok,
            verdict=verdict,
            clear_to_cm=clear_to,
            fails=ch.fails,
        )

    if not ch.enabled:
        return make(ChannelStatus.DISABLED, Health.DISABLED, Verdict.DISABLED)
    if not link_ok:
        return make(ChannelStatus.LINK_DOWN, Health.LINK_DOWN, Verdict.UNKNOWN)
    if ch.measured_at is None or ch.status is None:
        return make(ChannelStatus.NO_DATA, Health.NO_DATA, Verdict.UNKNOWN)
    if age_ms is not None and age_ms > cfg.stale_after_ms:
        return make(ChannelStatus.STALE, Health.STALE, Verdict.UNKNOWN)

    s = ch.status
    if s is proto.EchoStatus.OK:
        if not 0 < ch.echo_us <= MAX_SANE_ECHO_US:
            return make(ChannelStatus.FAULT, Health.FAULT, Verdict.UNKNOWN)
        dist = round(echo_to_cm(ch.echo_us, cfg.speed_of_sound_mps), 1)
        if dist < cfg.min_range_cm:
            return make(ChannelStatus.BLIND_ZONE, Health.OK, Verdict.UNKNOWN)
        if dist > cfg.max_range_cm:
            return _out_of_range(cfg, make, alive, near_hold)
        if dist <= threshold_cm:
            return make(ChannelStatus.OK, Health.OK, Verdict.OBSTACLE, valid=True, dist=dist)
        return make(
            ChannelStatus.OK, Health.OK, Verdict.CLEAR, valid=True, dist=dist, clear_to=dist
        )
    if s is proto.EchoStatus.NO_FALL:
        return _out_of_range(cfg, make, alive, near_hold)
    if s is proto.EchoStatus.NO_EDGE:
        if near_hold:
            return make(ChannelStatus.BLIND_ZONE, Health.SILENT, Verdict.UNKNOWN)
        health = Health.OK if alive else Health.SILENT
        if cfg.no_echo_is_clear and (alive or not cfg.no_echo_requires_alive):
            return make(ChannelStatus.NO_ECHO, health, Verdict.CLEAR, clear_to=cfg.max_range_cm)
        return make(ChannelStatus.NO_ECHO, health, Verdict.UNKNOWN)
    # STUCK, a channel the host enabled but the MCU reports disabled, anything unknown.
    return make(ChannelStatus.FAULT, Health.FAULT, Verdict.UNKNOWN)


def _out_of_range(
    cfg: UltrasonicConfig, make: Callable[..., ChannelReading], alive: bool, near_hold: bool
):
    if near_hold:
        return make(ChannelStatus.BLIND_ZONE, Health.SILENT, Verdict.UNKNOWN)
    if cfg.out_of_range_is_clear and alive:
        return make(ChannelStatus.OUT_OF_RANGE, Health.OK, Verdict.CLEAR, clear_to=cfg.max_range_cm)
    return make(ChannelStatus.OUT_OF_RANGE, Health.OK if alive else Health.SILENT, Verdict.UNKNOWN)


def sector_distances(readings: list[dict[str, Any]]) -> dict[str, Any]:
    """Map per-channel entries onto the three sectors the AI model input and datasets use.

    Four JSN-SR04T (MCU order ``outer_left, inner_left, inner_right, outer_right``): left is
    ``outer_left``, right is ``outer_right`` and centre is the nearer valid one of the two inner
    sensors. The legacy HC-SR04 ids ``front_left/front_center/front_right`` map one to one. A
    sector without a fresh valid echo is 0.0: the AI path reads 0.0 as an obstacle at 0 cm, which
    keeps it fail-closed (audit N-25). ``min_distance_cm`` is the nearest valid echo, 0.0 if none.
    The 3-sector input contract is the existing model's; it is not validated for four sensors.
    """
    valid = {
        r["sensor_id"]: float(r["distance_cm"])
        for r in readings
        if r.get("valid") and r.get("sensor_id")
    }

    def pick(*ids: str) -> float:
        found = [valid[i] for i in ids if i in valid]
        return min(found) if found else 0.0

    if any(
        r.get("sensor_id") in ("outer_left", "inner_left", "inner_right", "outer_right")
        for r in readings
    ):
        left, centre, right = (
            pick("outer_left"),
            pick("inner_left", "inner_right"),
            pick("outer_right"),
        )
    else:
        left, centre, right = pick("front_left"), pick("front_center"), pick("front_right")
    return {
        "front_left_cm": left,
        "front_center_cm": centre,
        "front_right_cm": right,
        "min_distance_cm": min(valid.values()) if valid else 0.0,
        "channels_cm": [float(r["distance_cm"]) if r.get("valid") else 0.0 for r in readings],
    }


class JsnSr04tMcuDriver(HardwareDriver):
    """Host driver for the four-sensor ultrasonic MCU."""

    def __init__(
        self,
        config: UltrasonicConfig | dict[str, Any] | None = None,
        *,
        serial_factory: SerialFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        obstacle_threshold_cm: float = 30.0,
        simulate: bool | None = None,
        rng: random.Random | None = None,
    ) -> None:
        cfg = config if isinstance(config, UltrasonicConfig) else UltrasonicConfig(**(config or {}))
        super().__init__(config=cfg.model_dump())
        self.cfg = cfg
        self.channel_ids: list[str] = list(cfg.channels)
        self.obstacle_threshold_cm = float(obstacle_threshold_cm)
        self._serial_factory = serial_factory or open_pyserial
        self._clock = clock
        self._wall = wall_clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._simulate_arg = simulate
        self._rng = rng or random.Random()
        self._channels = [
            _Channel(index=i, sensor_id=name, enabled=name not in cfg.disabled_channels)
            for i, name in enumerate(cfg.channels)
        ]
        self.link_state = LinkState.DISCONNECTED
        self.link_fault: str | None = None
        self.last_error: str | None = None
        self.stats = LinkStats()
        self.info: proto.Info | None = None
        self.simulated = False
        self._transport: SerialTransport | None = None
        self._decoder = proto.FrameDecoder(partial_timeout=PARTIAL_FRAME_TIMEOUT_S)
        self._tx_seq = 0
        self._rx_seq: int | None = None
        self._boot_id: int | None = None
        self._last_rx: float | None = None
        self._last_hello = -1e9
        self._hs_deadline = 0.0
        self._requested: proto.McuConfig | None = None
        self._next_connect_at = 0.0
        self._backoff_idx = 0
        self._task: asyncio.Task[None] | None = None
        self._sim_last = -1e9
        self._sim_cycle = 0
        self._last_ping_nonce: int | None = None
        #: True after ``stop()`` / ``close()``: ``poll()`` then never (re)opens the port, so a late
        #: caller cannot leave a port open that nobody will close. ``initialize()`` clears it.
        self._closed = False
        #: Optional observer called with ``(proto.Reading, wall datetime)`` for every accepted
        #: reading frame (used by ``scripts/ultrasonic_mcu_probe.py`` to log raw echo widths for
        #: audit M-40). A raising hook is logged and ignored: it must never disturb ingestion.
        self.reading_hook: Callable[[proto.Reading, datetime], None] | None = None

    # ---- lifecycle -------------------------------------------------------------------------

    async def initialize(self) -> None:
        self._closed = False
        self.simulated = (
            self._simulate_arg if self._simulate_arg is not None else is_simulation_mode()
        )
        if self.simulated:
            self.link_state = LinkState.ACTIVE
            self.initialized = True
            return
        # Bounded wait for the handshake; a missing MCU is a reported state, not an exception.
        deadline = self._clock() + self.cfg.handshake_timeout_s
        while self._clock() < deadline:
            self.poll()
            if self.link_state in (LinkState.ACTIVE, LinkState.FAULT):
                break
            await self._sleep(POLL_INTERVAL_S)
        self.initialized = True

    async def start(self) -> None:
        self.running = True
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        self.running = False
        self._closed = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._close_transport()
        if not self.simulated:
            self.link_state = LinkState.DISCONNECTED

    async def _run(self) -> None:
        while self.running:
            try:
                self.poll()
            except Exception:  # never let the service loop die, but do not spin on a bug
                logger.exception("ultrasonic MCU service loop error")
                await self._sleep(0.5)
            await self._sleep(POLL_INTERVAL_S)

    # ---- service ---------------------------------------------------------------------------

    def poll(self) -> None:
        """One non-blocking service step: connect if needed, read, dispatch, supervise."""
        if self._closed:
            return
        if self.simulated:
            self._poll_sim()
            return
        now = self._clock()
        if self._transport is None:
            self._maybe_connect(now)
            if self._transport is None:
                return
        self._read_and_dispatch(now)
        if self._transport is not None:
            self._supervise(now)

    def _maybe_connect(self, now: float) -> None:
        if now < self._next_connect_at:
            return
        if not self.cfg.port:
            self._set_fault("no_port_configured")
            self._next_connect_at = now + RECONNECT_BACKOFF_S[-1]
            return
        try:
            self._transport = self._serial_factory(self.cfg.port, self.cfg.baudrate)
        except Exception as exc:
            self.stats.connect_failures += 1
            self.last_error = f"open_failed: {exc}"
            self._schedule_reconnect(now)
            if self.link_fault != self.last_error:
                logger.warning("ultrasonic MCU %s: %s", self.cfg.port, self.last_error)
            self.link_fault = self.last_error
            return
        self._decoder.reset()
        self._rx_seq = None
        self._boot_id = None
        self._last_rx = None
        self.info = None
        self.link_state = LinkState.HANDSHAKE
        self.link_fault = None
        self._hs_deadline = now + self.cfg.handshake_timeout_s
        self._send(proto.FrameType.HELLO, proto.build_hello(), now)

    def _read_and_dispatch(self, now: float) -> None:
        transport = self._transport
        assert transport is not None
        try:
            waiting = transport.in_waiting
            data = transport.read(waiting) if waiting else b""
        except Exception as exc:  # SerialException / OSError: the device went away
            self._link_lost(now, f"read_error: {exc}")
            return
        frames = self._decoder.feed(data, now)
        self._sync_decoder_stats()
        wall = self._wall()
        for frame in frames:
            if self._transport is None:
                break
            self._on_frame(frame, now, wall)

    def _sync_decoder_stats(self) -> None:
        d = self._decoder
        self.stats.crc_errors = d.crc_errors
        self.stats.bad_length = d.bad_length
        self.stats.garbage_bytes = d.garbage
        self.stats.partial_timeouts = d.partial_timeouts

    def _supervise(self, now: float) -> None:
        if self.link_state is LinkState.HANDSHAKE and now > self._hs_deadline:
            self._link_lost(now, "handshake_timeout")
            return
        if self._last_rx is None or self.link_state is LinkState.HANDSHAKE:
            return
        silent = now - self._last_rx
        limit = self.cfg.link_timeout_ms / 1000.0
        if silent > limit:
            if now - self._last_hello >= HELLO_RETRY_S:
                self._send(proto.FrameType.HELLO, proto.build_hello(), now)
            if self._transport is not None and silent > 3 * limit:
                self._link_lost(now, "silent")

    # ---- frames ----------------------------------------------------------------------------

    def _on_frame(self, frame: proto.Frame, now: float, wall: datetime) -> None:
        self.stats.frames_rx += 1
        self._last_rx = now
        if self._rx_seq is not None:
            self.stats.frames_lost += (frame.seq - (self._rx_seq + 1)) & 0xFF
        self._rx_seq = frame.seq
        if frame.version != proto.PROTO_VERSION:
            self._set_fault(f"protocol_version_mismatch: mcu frame version {frame.version}")
            return
        try:
            msg = proto.parse_payload(frame)
        except proto.ProtocolError as exc:
            self.stats.bad_payload += 1
            logger.warning("ultrasonic MCU: bad payload: %s", exc)
            return
        if isinstance(msg, proto.Reading):
            self._on_reading(msg, now, wall)
        elif isinstance(msg, proto.Info):
            self._on_info(msg, now)
        elif isinstance(msg, proto.McuConfig):
            self._on_config_ack(msg, now)
        elif isinstance(msg, proto.Nak):
            self._on_nak(msg)
        elif isinstance(msg, proto.Pong):
            self.stats.pongs += 1

    def _on_info(self, info: proto.Info, now: float) -> None:
        self.stats.infos += 1
        self.info = info
        if info.proto_version != proto.PROTO_VERSION:
            self._set_fault(f"protocol_version_mismatch: mcu speaks v{info.proto_version}")
            return
        if info.sensor_count != len(self._channels):
            self._set_fault(
                f"channel_count_mismatch: mcu has {info.sensor_count}, "
                f"configured {len(self._channels)}"
            )
            return
        rebooted = self._boot_id is not None and info.boot_id != self._boot_id
        self._boot_id = info.boot_id
        if rebooted:
            self.stats.mcu_resets += 1
            logger.warning("ultrasonic MCU rebooted (reset reason %d)", info.reset_reason)
            for ch in self._channels:
                ch.clear()
        if self.link_state is LinkState.ACTIVE and not rebooted:
            return  # heartbeat
        self._send_config(now)

    def _send_config(self, now: float) -> None:
        self._requested = self.expected_mcu_config()
        if self.link_state is not LinkState.HANDSHAKE:
            # Entering the handshake. A re-send while already waiting (the next INFO heartbeat)
            # must not extend the deadline, or an MCU that never acknowledges would wait forever.
            self._hs_deadline = now + self.cfg.handshake_timeout_s
        self.link_state = LinkState.HANDSHAKE
        self._send(proto.FrameType.CONFIG, proto.build_config(self._requested), now)

    def expected_mcu_config(self) -> proto.McuConfig:
        mask = 0
        for ch in self._channels:
            if ch.enabled:
                mask |= 1 << ch.index
        return proto.McuConfig(
            enable_mask=mask,
            echo_window_ms=self.cfg.echo_window_ms,
            gap_ms=self.cfg.inter_sensor_gap_ms,
            quiet_after=DEFAULT_QUIET_AFTER,
        )

    def _on_config_ack(self, ack: proto.McuConfig, now: float) -> None:
        if self.link_state is not LinkState.HANDSHAKE or self._requested is None:
            return
        if ack != self._requested:
            self._set_fault(f"config_mismatch: asked {self._requested}, mcu applied {ack}")
            return
        self.link_state = LinkState.ACTIVE
        self.link_fault = None
        self._backoff_idx = 0
        logger.info("ultrasonic MCU link active (%s)", self.info)

    def _on_nak(self, nak: proto.Nak) -> None:
        self.stats.naks += 1
        try:
            reason = proto.NakReason(nak.reason).name.lower()
        except ValueError:
            reason = f"code_{nak.reason}"
        if nak.ref_type == proto.FrameType.CONFIG:
            self._set_fault(f"config_rejected: {reason}")
        else:
            logger.warning("ultrasonic MCU NAK for type 0x%02x: %s", nak.ref_type, reason)

    def _on_reading(self, r: proto.Reading, now: float, wall: datetime) -> None:
        if self.link_state is not LinkState.ACTIVE:
            self.stats.readings_ignored += 1
            return
        self._ingest(r, now, wall)

    def _ingest(self, r: proto.Reading, now: float, wall: datetime) -> None:
        if r.sensor >= len(self._channels):
            self.stats.bad_readings += 1
            return
        self.stats.readings += 1
        ch = self._channels[r.sensor]
        try:
            status = proto.EchoStatus(r.status)
        except ValueError:
            self.stats.bad_readings += 1
            status = proto.EchoStatus.STUCK  # an unknown code is never a quiet sensor
        ch.status = status
        ch.echo_us = r.echo_us
        ch.mcu_health = r.health
        ch.fails = r.fails
        ch.cycle = r.cycle
        ch.t_ms = r.t_ms
        ch.measured_at = now
        ch.measured_wall = wall
        if status is proto.EchoStatus.OK and 0 < r.echo_us <= MAX_SANE_ECHO_US:
            # A complete echo pulse is the only proof of life (NO_FALL is no echo: a line stuck
            # high looks the same). Remember whether it came from something right in front.
            ch.pulse_at = now
            near_line_cm = NEAR_ECHO_FACTOR * max(self.cfg.min_range_cm, self.obstacle_threshold_cm)
            near = echo_to_cm(r.echo_us, self.cfg.speed_of_sound_mps) < near_line_cm
            ch.near_at = now if near else None
        if self.reading_hook is not None:
            try:
                self.reading_hook(r, wall)
            except Exception:
                logger.exception("ultrasonic reading hook failed")

    # ---- link bookkeeping ------------------------------------------------------------------

    def _send(self, ftype: proto.FrameType, payload: bytes, now: float) -> bool:
        transport = self._transport
        if transport is None:
            return False
        frame = proto.encode_frame(ftype, self._tx_seq, payload)
        self._tx_seq = (self._tx_seq + 1) & 0xFF
        if ftype is proto.FrameType.HELLO:
            self._last_hello = now
        try:
            transport.write(frame)
        except Exception as exc:
            self._link_lost(now, f"write_error: {exc}")
            return False
        return True

    def ping(self, nonce: int = 0) -> bool:
        """Diagnostic round trip request; the answer is counted in ``stats.pongs``."""
        self._last_ping_nonce = nonce
        return self._send(proto.FrameType.PING, proto.build_ping(nonce), self._clock())

    def _set_fault(self, why: str) -> None:
        if self.link_fault != why:
            logger.error("ultrasonic MCU link fault: %s", why)
        self.link_state = LinkState.FAULT
        self.link_fault = why
        self.last_error = why

    def _link_lost(self, now: float, why: str) -> None:
        logger.warning("ultrasonic MCU link lost: %s", why)
        self.stats.link_losses += 1
        self.last_error = why
        self.link_fault = why
        self.link_state = LinkState.DISCONNECTED
        self._close_transport()
        self._decoder.reset()
        for ch in self._channels:
            ch.clear()
        self._schedule_reconnect(now)

    def _schedule_reconnect(self, now: float) -> None:
        delay = RECONNECT_BACKOFF_S[min(self._backoff_idx, len(RECONNECT_BACKOFF_S) - 1)]
        self._backoff_idx += 1
        self._next_connect_at = now + delay

    def close(self) -> None:
        """Release the port without the async lifecycle (scripts that only ``poll()``)."""
        self._closed = True
        self._close_transport()
        if not self.simulated:
            self.link_state = LinkState.DISCONNECTED

    def _close_transport(self) -> None:
        transport, self._transport = self._transport, None
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()

    # ---- simulation ------------------------------------------------------------------------

    def _poll_sim(self) -> None:
        now = self._clock()
        if now - self._sim_last < SIM_CYCLE_S:
            return
        self._sim_last = now
        self._sim_cycle += 1
        self._last_rx = now
        wall = self._wall()
        for ch in self._channels:
            if not ch.enabled:
                continue
            base = _SIM_BASE_CM[ch.index % len(_SIM_BASE_CM)]
            dist = base + self._rng.gauss(0.0, 2.0)
            if self._sim_cycle % 100 == 50 + 7 * ch.index:
                dist = 27.0 + self._rng.uniform(0.0, 3.0)  # a close obstacle
            if self._rng.random() < 0.01:
                status, echo = proto.EchoStatus.NO_EDGE, 0
            else:
                status, echo = proto.EchoStatus.OK, cm_to_echo_us(dist, self.cfg.speed_of_sound_mps)
            self._ingest(
                proto.Reading(ch.index, status, 0, 0, echo, self._sim_cycle & 0xFFFF, 0),
                now,
                wall,
            )

    # ---- views -----------------------------------------------------------------------------

    def link_ok(self, now: float | None = None) -> bool:
        """True while the link is verified and frames keep arriving."""
        now = self._clock() if now is None else now
        if self.link_state is not LinkState.ACTIVE:
            return False
        if self.simulated:
            return True
        return (
            self._last_rx is not None and (now - self._last_rx) <= self.cfg.link_timeout_ms / 1000
        )

    def snapshot(self, threshold_cm: float | None = None) -> list[ChannelReading]:
        now = self._clock()
        thr = self.obstacle_threshold_cm if threshold_cm is None else float(threshold_cm)
        link = self.link_ok(now)
        return [evaluate_channel(self.cfg, ch, now, link, thr) for ch in self._channels]

    def obstacle_verdict(self, threshold_cm: float | None = None) -> dict[str, Any]:
        """Aggregate over the fitted channels. OBSTACLE beats UNKNOWN beats CLEAR."""
        thr = self.obstacle_threshold_cm if threshold_cm is None else float(threshold_cm)
        channels = self.snapshot(thr)
        fitted = [c for c in channels if c.verdict is not Verdict.DISABLED]
        reasons = [
            f"{c.sensor_id}:{c.status.value}" for c in fitted if c.verdict is not Verdict.CLEAR
        ]
        if not fitted:
            state = Verdict.UNKNOWN
            reasons = ["no fitted channels"]
        elif any(c.verdict is Verdict.OBSTACLE for c in fitted):
            state = Verdict.OBSTACLE
        elif any(c.verdict is Verdict.UNKNOWN for c in fitted):
            state = Verdict.UNKNOWN
        else:
            state = Verdict.CLEAR
        return {
            "state": state.value,
            "min_cm": self.get_minimum_distance(channels),
            "threshold_cm": thr,
            "channels": [c.as_dict() for c in channels],
            "reasons": reasons,
            "link_state": self.link_state.value,
            "link_fault": self.link_fault,
        }

    def get_minimum_distance(self, channels: list[ChannelReading] | None = None) -> float | None:
        channels = self.snapshot() if channels is None else channels
        valid = [c.distance_cm for c in channels if c.valid]
        return min(valid) if valid else None

    async def health_check(self) -> dict[str, Any]:
        now = self._clock()
        return {
            "sensor": "ultrasonic_array_jsn_sr04t_mcu",
            "backend": "mcu_usb",
            "initialized": self.initialized,
            "running": self.running,
            "simulation": self.simulated,
            "port": self.cfg.port,
            "link_state": self.link_state.value,
            "link_ok": self.link_ok(now),
            "link_fault": self.link_fault,
            "last_error": self.last_error,
            "mcu": None
            if self.info is None
            else {
                "fw": f"{self.info.fw_major}.{self.info.fw_minor}",
                "proto": self.info.proto_version,
                "sensor_count": self.info.sensor_count,
                "boot_id": self.info.boot_id,
                "uptime_ms": self.info.uptime_ms,
                "reset_reason": self.info.reset_reason,
            },
            "channels": {c.sensor_id: c.as_dict() for c in self.snapshot()},
            "stats": {f: getattr(self.stats, f) for f in LinkStats.__slots__},
        }


__all__ = [
    "ChannelReading",
    "ChannelStatus",
    "Health",
    "JsnSr04tMcuDriver",
    "LinkState",
    "LinkStats",
    "Verdict",
    "cm_to_echo_us",
    "echo_to_cm",
    "evaluate_channel",
    "open_pyserial",
    "sector_distances",
]
