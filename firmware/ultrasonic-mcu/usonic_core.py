"""Measurement scheduling and protocol handling for the ultrasonic MCU (hardware independent).

``Core`` owns everything that is not a pin: sequential (time-division) triggering, per-sensor
health, the framed protocol and the INFO heartbeat. It talks to the hardware only through a
``hal`` object, so the same code runs on the RP2040 (``usonic_hal_cp.CpHal``) and in the host tests
against a simulated HAL.

HAL contract (duck typed):

* ``now_ms() -> int``                  monotonic milliseconds, never wrapping inside a run
* ``measure(index, window_us)``        trigger sensor ``index`` and wait at most ``window_us``
                                       for its echo; returns ``(status, echo_us)`` with ``status``
                                       one of ``ST_OK/ST_NO_EDGE/ST_NO_FALL/ST_STUCK`` and
                                       ``echo_us`` the echo pulse width in microseconds (0 if none)
* ``read() -> bytes``                  bytes received from the host so far (non-blocking)
* ``write(data)``                      send bytes to the host (must not block for long)
* ``boot_id() -> int``                 random 32-bit id, new on every boot
* ``reset_reason() -> int``            one of the ``RESET_*`` codes
* ``feed_watchdog()``                  feed the hardware watchdog (``usonic_main.run``: each pass)
* ``reboot()``                         reset the MCU now; never returns on hardware. Only
                                       ``usonic_main.run`` calls it, after an unexpected exception

Safety posture: the MCU never decides what a reading means and never actuates anything. It only
reports. Every enabled sensor is fired once per cycle, always: a silent or faulted sensor is never
skipped or slowed down (a healthy sensor over open ground is silent too, and skipping it would age
its data). Time-division invariant (tested): the next trigger is never earlier than
``previous trigger + echo window`` and never earlier than ``previous measurement end + gap``, so one
sensor cannot hear another's burst.
"""

import usonic_proto as P

WINDOW_MIN_MS = 10
WINDOW_MAX_MS = 100
GAP_MIN_MS = 5
GAP_MAX_MS = 500


class Core:
    def __init__(
        self,
        hal,
        sensor_count,
        firing_order=None,
        fw_version=(1, 0),
        echo_window_ms=30,
        gap_ms=30,
        quiet_after=8,
        info_period_ms=1000,
        partial_timeout_ms=50,
    ):
        self.hal = hal
        self.n = sensor_count
        order = tuple(firing_order) if firing_order is not None else tuple(range(sensor_count))
        if sorted(order) != list(range(sensor_count)):
            raise ValueError("firing_order must be a permutation of the sensor indices")
        self.order = order
        self.fw_version = fw_version
        self.info_period_ms = info_period_ms
        self.window_ms = echo_window_ms
        self.gap_ms = gap_ms
        self.quiet_after = quiet_after
        self.enable_mask = (1 << sensor_count) - 1

        self.dec = P.Decoder(partial_timeout=partial_timeout_ms)
        self.tx_seq = 0
        self.boot_id = hal.boot_id() & 0xFFFFFFFF
        self.reset_reason = hal.reset_reason()
        self._t0 = hal.now_ms()
        self._last_info_ms = None
        self._next_fire_ms = self._t0
        self.pos = 0
        self.cycle = 0

        self.fails = [0] * sensor_count  # consecutive measurements without an echo pulse
        self.health = [P.HEALTH_ACTIVE] * sensor_count

        self.frames_rx = 0
        self.frames_tx = 0
        self.measurements = 0
        self.naks_sent = 0
        self.ignored = 0

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self):
        """Announce the boot (INFO) so the host learns a new ``boot_id`` immediately."""
        self._send_info(self.hal.now_ms())

    def poll(self):
        """One non-blocking step: serve the host, heartbeat, and at most one measurement."""
        now = self.hal.now_ms()
        data = self.hal.read()
        for ver, ftype, _seq, payload in self.dec.feed(data, now):
            self.frames_rx += 1
            self._handle(ver, ftype, payload, now)
        if self._last_info_ms is None or now - self._last_info_ms >= self.info_period_ms:
            self._send_info(now)
        if self.n and now >= self._next_fire_ms:
            self._fire_next(now)

    # ---- host frames -----------------------------------------------------------------------

    def _handle(self, ver, ftype, payload, now):
        if ftype & 0x80:
            # A device-to-host frame type arriving at the device means a looped-back or crossed
            # line. Never answer it: a NAK would loop forever.
            self.ignored += 1
            return
        if ver != P.PROTO_VERSION:
            self._nak(ftype, P.NAK_BAD_VERSION)
            return
        if ftype not in (P.T_HELLO, P.T_CONFIG, P.T_PING):
            self._nak(ftype, P.NAK_UNKNOWN_TYPE)
            return
        if len(payload) != P.PAYLOAD_SIZES[ftype]:
            self._nak(ftype, P.NAK_BAD_LENGTH)
            return
        if ftype == P.T_HELLO:
            self._send_info(now)
        elif ftype == P.T_PING:
            nonce = P.unpack_ping(payload)["nonce"]
            self._send(P.T_PONG, P.pack_pong(nonce, now - self._t0))
        else:
            self._apply_config(P.unpack_config(payload))

    def _apply_config(self, cfg):
        full = (1 << self.n) - 1
        if (
            cfg["enable_mask"] & ~full
            or not WINDOW_MIN_MS <= cfg["echo_window_ms"] <= WINDOW_MAX_MS
            or not GAP_MIN_MS <= cfg["gap_ms"] <= GAP_MAX_MS
            or not 1 <= cfg["quiet_after"] <= 255
        ):
            self._nak(P.T_CONFIG, P.NAK_BAD_VALUE)
            return
        newly_enabled = cfg["enable_mask"] & ~self.enable_mask
        for i in range(self.n):
            if (newly_enabled >> i) & 1:
                self.fails[i] = 0
                self.health[i] = P.HEALTH_ACTIVE
        self.enable_mask = cfg["enable_mask"]
        self.window_ms = cfg["echo_window_ms"]
        self.gap_ms = cfg["gap_ms"]
        self.quiet_after = cfg["quiet_after"]
        # Echo the values that are now in force: the host verifies them (a confirmed write).
        self._send(
            P.T_CONFIG_ACK,
            P.pack_config(self.enable_mask, self.window_ms, self.gap_ms, self.quiet_after),
        )

    # ---- measuring -------------------------------------------------------------------------

    def _fire_next(self, now):
        idx = self.order[self.pos]
        self.pos += 1
        wrapped = self.pos >= len(self.order)
        if wrapped:
            self.pos = 0

        if not (self.enable_mask >> idx) & 1:
            self.health[idx] = P.HEALTH_DISABLED
            self._emit(idx, P.ST_DISABLED, 0, now)
            self._next_fire_ms = now + self.gap_ms
        else:
            try:
                status, echo_us = self.hal.measure(idx, self.window_ms * 1000)
            except Exception:
                # A HAL error is a hardware fault, never a quiet sensor.
                status, echo_us = P.ST_STUCK, 0
            end = self.hal.now_ms()
            self.measurements += 1
            self._update_health(idx, status)
            self._emit(idx, status, echo_us, end)
            # Time-division: no trigger before this sensor's window has closed, nor before the
            # gap after its measurement ended.
            self._next_fire_ms = max(now + self.window_ms, end + self.gap_ms)

        if wrapped:
            self.cycle = (self.cycle + 1) & 0xFFFF

    def _update_health(self, idx, status):
        if status == P.ST_OK or status == P.ST_NO_FALL:
            self.fails[idx] = 0
            self.health[idx] = P.HEALTH_ACTIVE
            return
        if self.fails[idx] < 255:
            self.fails[idx] += 1
        if status == P.ST_STUCK:
            self.health[idx] = P.HEALTH_FAULT
        elif self.fails[idx] >= self.quiet_after:
            self.health[idx] = P.HEALTH_QUIET
        else:
            self.health[idx] = P.HEALTH_ACTIVE

    def _emit(self, idx, status, echo_us, t_ms):
        self._send(
            P.T_READING,
            P.pack_reading(
                idx, status, self.health[idx], self.fails[idx], echo_us, self.cycle, t_ms
            ),
        )

    # ---- tx --------------------------------------------------------------------------------

    def _send(self, ftype, payload):
        frame = P.encode(ftype, self.tx_seq, payload)
        self.tx_seq = (self.tx_seq + 1) & 0xFF
        self.frames_tx += 1
        self.hal.write(frame)

    def _send_info(self, now):
        self._last_info_ms = now
        self._send(
            P.T_INFO,
            P.pack_info(
                self.fw_version[0],
                self.fw_version[1],
                self.n,
                self.boot_id,
                now - self._t0,
                self.reset_reason,
                self.enable_mask,
            ),
        )

    def _nak(self, ref_type, reason):
        self.naks_sent += 1
        self._send(P.T_NAK, P.pack_nak(ref_type, reason))
