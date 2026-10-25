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
* The host never writes to a port that has not first sent a valid MCU frame. The MCU announces
  itself (INFO at boot and every second), so the host listens first: a ``ultrasonic.port`` that
  points at something else (the MCU's own REPL console, the RoboHAT's port, a renumbered ttyACM)
  is only read, never written to. A binary frame contains the console control bytes 0x01..0x04
  (Ctrl-A..D), so one write to the wrong port can stop the program on the other end.
* A reading is dated by the MCU's own clock against one baseline per link. A reading that is out
  of step with that baseline (a wild timestamp, a replay) is refused and counted, and a baseline
  that turns out to be wrong is rebuilt from the live stream that kept being refused, so one bad
  timestamp can neither make data look fresher nor silence every channel for good.

Every number in :class:`~backend.src.models.hardware_config.UltrasonicConfig` is an unvalidated
placeholder until audit measurement M-40.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
#: Most read() calls one poll() makes: it drains what the port holds so no backlog is left behind,
#: but a babbling port can never keep it spinning.
MAX_DRAIN_READS = 64
#: Relative drift between the host's and the MCU's clock that the age estimate tolerates (NTP
#: slew plus two crystals); a backlog that grows slower than this is not detectable, nor harmful.
MCU_CLOCK_DRIFT = 5e-4
#: How often the MCU repeats its INFO frame (``INFO_PERIOD_MS`` in ``firmware/ultrasonic-mcu``).
#: The host never speaks first, it waits for this announcement, so a handshake gets at least this
#: long (plus ``HANDSHAKE_MARGIN_S``) whatever ``handshake_timeout_s`` says.
MCU_INFO_PERIOD_S = 1.0
#: What a handshake gets on top of one INFO period (the CONFIG round trip, scheduling jitter).
HANDSHAKE_MARGIN_S = 0.5
#: Accepted readings a clock baseline needs before a reading can be refused for running AHEAD of
#: it. The first readings of a baseline may be a backlog that is still being drained, and every
#: newer frame of such a backlog legitimately lowers the offset until the stream has caught up.
MCU_SETTLE_FRAMES = 8
#: A reading whose MCU timestamp puts it more than this far ahead of a settled baseline (the MCU
#: cannot have outrun the host's clock) is refused without touching the baseline, so one wild
#: timestamp moves nothing. Never below twice the longest spacing of two readings (see
#: ``_future_slack_s``). The bias a reading just inside the limit can add to every later age stays
#: under the 100 ms margin ``UltrasonicConfig`` keeps between ``stale_after_ms`` and one cycle.
MCU_FUTURE_SLACK_S = 0.1
#: Refused readings that kept pace with the host clock for this long (never less than twice a
#: measurement cycle) are a live stream whose BASELINE is wrong, not noise: it is rebuilt from them.
MCU_REBASE_S = 2.0
#: The run needs at least this many refused readings that advanced the MCU clock.
MCU_REBASE_FRAMES = 8
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
    """Default transport. ``pyserial`` is imported here, never at module import.

    Opened exclusively (advisory ``flock``, pyserial >= 3.3): a second opener that asks for
    exclusivity too (the bench probe, another ``SensorManager``) fails with a reported
    ``open_failed`` instead of splitting the byte stream. A process that ignores the lock is not
    stopped by it.
    """
    import serial  # type: ignore

    return serial.Serial(port=port, baudrate=baudrate, timeout=0, write_timeout=0.2, exclusive=True)


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


class Disposition(StrEnum):
    """What the driver did with one CRC-valid READING frame (for ``reading_hook``, the raw log)."""

    ACCEPTED = "accepted"  # became the channel's newest reading
    IGNORED_LINK = "ignored_link"  # the link is not ACTIVE: handshake, or the configuration drifted
    IGNORED_CHANNEL = "ignored_channel"  # a channel the host did not enable (not fitted)
    BAD_SENSOR = "bad_sensor"  # a sensor index beyond the configured channels
    REFUSED_CLOCK = "refused_clock"  # out of step with the MCU clock baseline (replay, wild stamp)


@dataclass(frozen=True, slots=True)
class ChannelReading:
    """One channel as seen at one instant. ``as_dict`` is the stable external contract."""

    sensor_id: str
    index: int
    status: ChannelStatus
    valid: bool  # True only for a fresh in-range echo
    distance_cm: float  # 0.0 unless valid
    echo_us: int | None
    timestamp: datetime | None  # when the MEASUREMENT was taken, on the host's wall clock
    age_ms: int | None
    health: Health
    alive: bool
    verdict: Verdict
    clear_to_cm: float | None
    fails: int
    #: the same instant on the driver's monotonic clock (None if the channel never reported); a
    #: consumer that must not trust the wall clock ages an entry as ``clock() - measured_mono``
    measured_mono: float | None = None

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
            "measured_mono": self.measured_mono,
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
    #: host monotonic time of the last real measurement, dated by the MCU's own clock
    measured_at: float | None = None
    measured_wall: datetime | None = None
    pulse_at: float | None = None  # host monotonic time of the last real echo pulse
    mcu_ms: int | None = None  # unwrapped MCU t_ms of the last accepted reading (replay guard)
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
        self.mcu_ms = None


@dataclass(slots=True)
class _RefusedRun:
    """The readings refused since the last accepted one, read as one candidate MCU clock.

    ``span_ms`` is how far the MCU clock advanced from the first refused reading to the newest
    one that advanced it; ``off`` the lowest ``receive time - mcu time`` over the run (the best
    case latency of that stream). Duplicates and replays do not advance it.
    """

    first_at: float  # host monotonic time of the first refused reading
    first_raw: int  # its MCU timestamp (uint32), the origin of ``span_ms``
    last_at: float = 0.0  # host monotonic time of the newest reading that advanced the run
    last_raw: int = 0
    span_ms: int = 0
    frames: int = 1
    off: float = 0.0

    def __post_init__(self) -> None:
        self.last_at = self.first_at
        self.last_raw = self.first_raw
        self.off = self.first_at - self.first_raw / 1000.0

    def advance(self, raw: int, now: float) -> None:
        step = mcu_delta(raw, self.last_raw)
        if step <= 0:
            return
        self.span_ms += step
        self.last_raw = raw
        self.last_at = now
        self.frames += 1
        self.off = min(self.off, now - (self.first_raw + self.span_ms) / 1000.0)


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
    readings_stale_mcu: int = 0  # readings refused: older than the MCU's newest, or a replay
    config_drifts: int = 0  # the MCU ran a configuration other than the verified one (US-06)
    mcu_rebases: int = 0  # clock baseline rebuilt from a live stream that kept being refused


def echo_to_cm(echo_us: float, speed_of_sound_mps: float) -> float:
    """Round-trip echo width to one-way distance: ``echo_s * c / 2`` in centimetres."""
    return echo_us * speed_of_sound_mps / 20000.0


def cm_to_echo_us(distance_cm: float, speed_of_sound_mps: float) -> int:
    return round(distance_cm * 20000.0 / speed_of_sound_mps)


def mcu_delta(new: int, ref: int) -> int:
    """Signed milliseconds from ``ref`` to ``new`` on the MCU's wrapping uint32 clock."""
    d = (new - ref) & 0xFFFFFFFF
    return d - (1 << 32) if d >= 1 << 31 else d


def max_echo_us(cfg: UltrasonicConfig) -> int:
    """The longest echo pulse a healthy sensor can report: the echo window the MCU was set to.

    The host verified that window through the CONFIG echo, so an OK pulse wider than it never
    happened (the shipped firmware refuses to send one); one that arrives anyway is a fault.
    """
    return min(MAX_SANE_ECHO_US, cfg.echo_window_ms * 1000)


def _checked_threshold(value: Any, what: str = "obstacle_threshold_cm") -> float:
    """A stop threshold that is a finite number above 0 cm, or ``ValueError``.

    NaN compares false with everything, so a NaN threshold never reports an obstacle; 0 and
    negatives never do either (a 27 cm echo read CLEAR), and infinity makes everything an obstacle.
    """
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"{what} must be finite and above 0 cm, got {value!r}")
    return threshold


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
        ch.pulse_at is not None and ch.near_at is None and (now - ch.pulse_at) <= cfg.alive_window_s
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
            measured_mono=ch.measured_at,
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
        if not 0 < ch.echo_us <= max_echo_us(cfg):
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
    ``outer_left``, right is ``outer_right`` and centre is the nearer of the two inner sensors. The
    legacy HC-SR04 ids ``front_left/front_center/front_right`` map one to one. Fail closed (audit
    N-25: the AI path reads 0.0 as an obstacle at 0 cm): a sector reads 0.0 unless EVERY fitted
    channel feeding it has a fresh valid echo, so the centre needs BOTH inner sensors and a dead
    or blind channel is never hidden by its healthy neighbour; a sector with no fitted channel
    reads 0.0. ``min_distance_cm`` is the nearest valid echo, but 0.0 as soon as any fitted
    channel is invalid (and when none is fitted). Disabled (not fitted) channels do not count.
    The 3-sector input contract is the existing model's; it is not validated for four sensors.
    """
    rows = {r["sensor_id"]: r for r in readings if r.get("sensor_id")}
    disabled = {s for s, r in rows.items() if "disabled" in (r.get("verdict"), r.get("status"))}
    fitted = [s for s in rows if s not in disabled]
    valid = {s: float(rows[s]["distance_cm"]) for s in fitted if rows[s].get("valid")}

    def sector(*ids: str) -> float:
        feeding = [i for i in ids if i not in disabled]
        if not feeding or any(i not in valid for i in feeding):
            return 0.0
        return min(valid[i] for i in feeding)

    if any(s in ("outer_left", "inner_left", "inner_right", "outer_right") for s in rows):
        left, centre, right = (
            sector("outer_left"),
            sector("inner_left", "inner_right"),
            sector("outer_right"),
        )
    else:
        left, centre, right = sector("front_left"), sector("front_center"), sector("front_right")
    return {
        "front_left_cm": left,
        "front_center_cm": centre,
        "front_right_cm": right,
        "min_distance_cm": min(valid.values()) if fitted and len(valid) == len(fitted) else 0.0,
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
        self.obstacle_threshold_cm = _checked_threshold(obstacle_threshold_cm)
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
        # MCU clock bookkeeping (US-05). READING.t_ms is uint32 ms on the MCU's clock and wraps.
        # ``_mcu_top`` is the newest unwrapped t_ms accepted on this link, ``_clock_off`` the host
        # monotonic time of MCU time zero (the lowest ``receive time - t_ms`` seen, see _place).
        self._cycle_ms = (cfg.echo_window_ms + cfg.inter_sensor_gap_ms) * len(cfg.channels)
        self._mcu_top: int | None = None
        self._clock_off: float | None = None
        self._clock_off_at = 0.0
        # Longest spacing of two readings (one window plus one gap): the offset can only fall by
        # about that much per reading while a backlog is drained, so more than twice it is wild.
        frame_gap_s = (cfg.echo_window_ms + cfg.inter_sensor_gap_ms) / 1000.0
        self._future_slack_s = max(MCU_FUTURE_SLACK_S, 2 * frame_gap_s)
        self._rebase_s = max(MCU_REBASE_S, 2 * self._cycle_ms / 1000.0)
        self._settled = 0  # readings the current baseline has accepted
        self._refused: _RefusedRun | None = None  # readings refused since the last accepted one
        # The MCU repeats INFO every MCU_INFO_PERIOD_S and the host speaks only after hearing it.
        self._handshake_s = max(cfg.handshake_timeout_s, MCU_INFO_PERIOD_S + HANDSHAKE_MARGIN_S)
        #: True after ``stop()`` / ``close()``: ``poll()`` then never (re)opens the port, so a late
        #: caller cannot leave a port open that nobody will close. ``initialize()`` clears it.
        self._closed = False
        #: Optional observer called with ``(proto.Reading, wall datetime, Disposition)`` for EVERY
        #: CRC-valid READING frame, also the ones the driver ignores (handshake, a channel that is
        #: not fitted, an unknown sensor) or refuses (replay, wild timestamp): it is the raw frame
        #: log of ``scripts/ultrasonic_mcu_probe.py`` for audit M-40. A raising hook is logged and
        #: ignored: it must never disturb ingestion.
        self.reading_hook: Callable[[proto.Reading, datetime, Disposition], None] | None = None

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
        if self.cfg.no_echo_is_clear:
            logger.warning(
                "ultrasonic no_echo_is_clear is ON: a silent channel reads CLEAR, and a sensor "
                "that dies or sticks keeps reading CLEAR for up to alive_window_s=%.1f s after "
                "its last far echo%s. Keep alive_window_s short and independent obstacle "
                "sensing in place (audit M-40).",
                self.cfg.alive_window_s,
                ""
                if self.cfg.no_echo_requires_alive
                else " (no_echo_requires_alive is OFF: always)",
            )
        # Bounded wait for the handshake; a missing MCU is a reported state, not an exception.
        deadline = self._clock() + self._handshake_s
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
        self._tx_seq = 0
        self._reset_mcu_clock()
        self._boot_id = None
        self._last_rx = None
        self.info = None
        self.link_state = LinkState.HANDSHAKE
        self.link_fault = None
        # Listen first: the MCU announces itself, and until it has (see ``_send``) nothing is
        # written to a port that might be a console, another service's port or a wrong device.
        self._hs_deadline = now + self._handshake_s

    def _read_and_dispatch(self, now: float) -> None:
        transport = self._transport
        assert transport is not None
        try:
            data = self._drain(transport)
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

    @staticmethod
    def _drain(transport: SerialTransport) -> bytes:
        """Everything the port holds right now, so no backlog is left to be read (stale) later."""
        chunks: list[bytes] = []
        for _ in range(MAX_DRAIN_READS):
            waiting = transport.in_waiting
            if not waiting:
                break
            chunk = transport.read(waiting)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

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
            self._reset_mcu_clock()
        if self.link_state is LinkState.ACTIVE and not rebooted:
            wanted = self.expected_mcu_config().enable_mask
            if info.enable_mask == wanted:
                return  # heartbeat
            self._config_drift(f"INFO enable_mask 0x{info.enable_mask:02x}, want 0x{wanted:02x}")
        self._send_config(now)

    def _send_config(self, now: float) -> None:
        self._requested = self.expected_mcu_config()
        if self.link_state is not LinkState.HANDSHAKE:
            # Entering the handshake. A re-send while already waiting (the next INFO heartbeat)
            # must not extend the deadline, or an MCU that never acknowledges would wait forever.
            self._hs_deadline = now + self._handshake_s
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

    def _config_drift(self, why: str) -> None:
        """The MCU is not running the configuration the host verified (a second opener wrote it).

        The caller drops back to the handshake (CONFIG is re-sent on the same port, no reconnect);
        readings are ignored until it is acknowledged again, and nothing from before survives.
        """
        self.stats.config_drifts += 1
        self.last_error = self.link_fault = f"config_drift: {why}"
        logger.warning("ultrasonic MCU configuration drifted (%s): re-sending CONFIG", why)
        for ch in self._channels:
            ch.clear()

    def _on_config_ack(self, ack: proto.McuConfig, now: float) -> None:
        if self._requested is None:
            return
        if self.link_state is LinkState.ACTIVE:
            # A CONFIG_ACK nobody here asked for: someone else wrote the MCU's configuration.
            if ack != self._requested:
                self._config_drift(f"CONFIG_ACK {ack}, requested {self._requested}")
                self._send_config(now)
            return
        if self.link_state is not LinkState.HANDSHAKE:
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
            fate = Disposition.IGNORED_LINK
        else:
            fate = self._ingest(r, now, wall)
        if self.reading_hook is not None:
            try:
                self.reading_hook(r, wall, fate)
            except Exception:
                logger.exception("ultrasonic reading hook failed")

    def _reset_mcu_clock(self) -> None:
        """Forget the MCU clock baseline (new connection or MCU boot: its clock starts over)."""
        self._mcu_top = None
        self._clock_off = None
        self._settled = 0
        self._refused = None
        for ch in self._channels:
            ch.mcu_ms = None  # the same coordinate system: the last timestamps mean nothing now

    def _place(self, t_ms: int, ch: _Channel, now: float) -> float | None:
        """Date a reading on the host's monotonic timeline using the MCU's own clock (US-05).

        ``t_ms`` is the MCU clock at the end of the measurement. The host keeps the lowest
        ``receive time - t_ms`` it has seen (its best-case latency; it creeps up by the worst
        clock drift so a slow MCU clock cannot make fresh data look old) and dates every reading
        ``t_ms`` after that: a backlog of old frames drained in one poll is as old as the MCU says,
        not "just received". Returns None, and changes nothing, for a frame that is out of order
        by more than one measurement cycle, not newer than its own channel's last one (replayed,
        duplicated), or, once the baseline has settled, ahead of it by more than the longest
        spacing of two readings: the MCU cannot have outrun the host's clock, so such a timestamp
        is wild (US3-01) and must not move the baseline.

        A baseline that is itself wrong refuses a healthy stream for good (a wild first
        timestamp, an MCU clock that restarted under the same ``boot_id``); ``_ingest`` rebuilds
        it from the refused stream once that has kept pace with the host's clock for a while.

        Known limit (US3-11): the MCU clock has no absolute reference, so the very first reading
        of a baseline is dated "now" whatever its age. What bounds it: nothing sent before the
        CONFIG echo is ever used (``readings_ignored``), the first readings of a baseline are
        accepted without the "ahead of the baseline" check, so a later reading of the same stream
        lowers the offset and re-dates the channel at once, and a backlog drained in one poll ends
        with each channel's NEWEST frame. What is left is a stall of the host's own loop between
        its CONFIG and its first poll, and then at most one measurement cycle per channel.
        """
        top = self._mcu_top
        u = t_ms if top is None else top + mcu_delta(t_ms, top)
        if top is not None and u < top - self._cycle_ms:
            return None
        if ch.mcu_ms is not None and u <= ch.mcu_ms:
            return None
        sample = now - u / 1000.0
        if self._clock_off is None:
            off = sample
        else:
            held = self._clock_off + MCU_CLOCK_DRIFT * max(0.0, now - self._clock_off_at)
            if self._settled >= MCU_SETTLE_FRAMES and sample < held - self._future_slack_s:
                return None
            off = min(held, sample)
        self._mcu_top = u if top is None else max(top, u)
        self._clock_off, self._clock_off_at = off, now
        self._settled += 1
        ch.mcu_ms = u
        return min(now, off + u / 1000.0)

    def _note_refused(self, t_ms: int, now: float) -> bool:
        """Count one refused reading; True once the refused stream proves the baseline wrong.

        A wrong baseline refuses every reading of a healthy stream, for good. What tells that from
        noise is that the refused readings are themselves a live stream: for ``_rebase_s`` they
        kept pace with the host's clock. Duplicates and replays stand still, so they never
        qualify, and any accepted reading ends the run.
        """
        run = self._refused
        if run is None or now - run.last_at > self._rebase_s:
            self._refused = _RefusedRun(first_at=now, first_raw=t_ms)
            return False
        run.advance(t_ms, now)
        host_s = run.last_at - run.first_at
        if host_s < self._rebase_s:
            return False
        mcu_s = run.span_ms / 1000.0
        if run.frames >= MCU_REBASE_FRAMES and abs(mcu_s - host_s) <= 0.25 + 0.1 * host_s:
            return True
        self._refused = _RefusedRun(first_at=now, first_raw=t_ms)  # not a live stream: start over
        return False

    def _rebase_clock(self, now: float) -> None:
        """Rebuild the clock baseline from the live stream that kept being refused."""
        run = self._refused
        assert run is not None
        self.stats.mcu_rebases += 1
        logger.warning(
            "ultrasonic MCU clock baseline rebuilt: %d readings over %.1f s were refused although "
            "the MCU clock kept pace with the host's (wild timestamp, or the MCU clock restarted)",
            run.frames,
            run.last_at - run.first_at,
        )
        self._reset_mcu_clock()
        self._mcu_top = run.first_raw + run.span_ms
        self._clock_off = run.off
        self._clock_off_at = now

    def _ingest(self, r: proto.Reading, now: float, wall: datetime) -> Disposition:
        if r.sensor >= len(self._channels):
            self.stats.bad_readings += 1
            return Disposition.BAD_SENSOR
        ch = self._channels[r.sensor]
        if not ch.enabled:
            # Not fitted: nothing about this channel is ever used, so its timestamps must not be
            # able to move the clock baseline the fitted ones are dated by.
            self.stats.readings_ignored += 1
            return Disposition.IGNORED_CHANNEL
        measured_at = self._place(r.t_ms, ch, now)
        if measured_at is None:
            self.stats.readings_stale_mcu += 1
            if not self._note_refused(r.t_ms, now):
                return Disposition.REFUSED_CLOCK
            self._rebase_clock(now)
            measured_at = self._place(r.t_ms, ch, now)
            if measured_at is None:  # not newer than the run's own newest: still out of step
                return Disposition.REFUSED_CLOCK
        self._refused = None
        self.stats.readings += 1
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
        ch.measured_at = measured_at
        ch.measured_wall = wall - timedelta(seconds=now - measured_at)
        if status is proto.EchoStatus.OK and 0 < r.echo_us <= max_echo_us(self.cfg):
            # A complete echo pulse is the only proof of life (NO_FALL is no echo: a line stuck
            # high looks the same; a pulse wider than the echo window never happened). Remember
            # whether it came from something right in front.
            ch.pulse_at = measured_at
            near_line_cm = NEAR_ECHO_FACTOR * max(self.cfg.min_range_cm, self.obstacle_threshold_cm)
            near = echo_to_cm(r.echo_us, self.cfg.speed_of_sound_mps) < near_line_cm
            ch.near_at = measured_at if near else None
        return Disposition.ACCEPTED

    # ---- link bookkeeping ------------------------------------------------------------------

    def _send(self, ftype: proto.FrameType, payload: bytes, now: float) -> bool:
        transport = self._transport
        if transport is None:
            return False
        if self._last_rx is None:
            # Nothing valid has been heard on this link, so it is not known to be the MCU's data
            # port. A binary frame contains the console control bytes Ctrl-A..D (0x01..0x04): on
            # the MCU's REPL or the RoboHAT's port one write stops the program there (US3-02).
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
        """Diagnostic round trip request; the answer is counted in ``stats.pongs``.

        False, and nothing written, until a valid MCU frame has been heard on this link.
        """
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
                proto.Reading(
                    ch.index,
                    status,
                    0,
                    0,
                    echo,
                    self._sim_cycle & 0xFFFF,
                    round(now * 1000) & 0xFFFFFFFF,
                ),
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
        thr = (
            self.obstacle_threshold_cm
            if threshold_cm is None
            else _checked_threshold(threshold_cm, "threshold_cm")
        )
        link = self.link_ok(now)
        return [evaluate_channel(self.cfg, ch, now, link, thr) for ch in self._channels]

    def obstacle_verdict(self, threshold_cm: float | None = None) -> dict[str, Any]:
        """Aggregate over the fitted channels. OBSTACLE beats UNKNOWN beats CLEAR."""
        thr = (
            self.obstacle_threshold_cm
            if threshold_cm is None
            else _checked_threshold(threshold_cm, "threshold_cm")
        )
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
