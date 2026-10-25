# Ultrasonic MCU firmware (four JSN-SR04T over USB serial)

Status: **software only, UNVERIFIED on hardware.** Nothing in this directory has run on an RP2040, a
JSN-SR04T, or the mower. It was exercised against a simulated HAL, against fake CircuitPython
modules and against the real host driver (see [Testing](#testing)). No wiring is decided; every pin
and every electrical value below is a TBD with the measurement needed.

Why this exists: the Pi header cannot carry four trigger/echo ultrasonics plus the safety I/O, and
the legacy HC-SR04 pins collide with the HaLow HAT (audit B.1.4 option B, C.7, WP-P2-02, H-10/H-19).
A dedicated RP2040 owns the TRIG/ECHO timing and talks to the Pi over **USB serial only: no Raspberry
Pi header pin is used.** This MCU senses; it never actuates and is **never a stop path**. The
hardwired E-stop, the Pi's local safety supervisor and the other sensors stay in force.

```
JSN-SR04T x4 --TRIG/ECHO--> RP2040 (CircuitPython)  --USB CDC "data" port-->  Pi: JsnSr04tMcuDriver
                            time-division firing                             UltrasonicSensorInterface
                            per-sensor health                                 obstacle_verdict(), telemetry
                            framed, CRC-protected protocol                    config/hardware.yaml `ultrasonic:`
```

## Files (copy them FLAT to the CIRCUITPY drive)

| File | Role |
|---|---|
| `boot.py` | enables the second USB serial port (`usb_cdc.data`) and sets the USB product string. Needs a power cycle |
| `code.py` | CircuitPython entry point: `usonic_main.run()` |
| `usonic_main.py` | builds the HAL and the core, runs the loop, feeds the hardware watchdog, resets the MCU after an unexpected exception |
| `usonic_core.py` | sequential firing, health, protocol handling (hardware independent, fully unit tested) |
| `usonic_proto.py` | frame codec and payload packing (hardware independent) |
| `usonic_hal_cp.py` | the only file that touches pins: `digitalio` TRIG, `pulseio.PulseIn` ECHO, `usb_cdc`, watchdog |
| `usonic_config.py` | **owner-editable**: MCU pin names (TBD), firing order, scheduling defaults |

Never add this directory to `PYTHONPATH`: `code.py` would shadow the standard library's `code`
module that `pdb` imports and would then start the firmware loop.

The protocol runs on the **data** port, not the REPL console. A frame can contain `0x03`/`0x04`, which
the console treats as Ctrl-C/Ctrl-D and uses to interrupt or reset the program, so the host must be
pointed at the data port (`ultrasonic.port`). The data port enumerates as the second CDC interface of
the MCU; which `/dev/ttyACM*` or `/dev/serial/by-id/*` name that is on the Pi is **TBD (owner)**.

## TBD for the owner (nothing here is assumed)

| Item | Where it is decided | Measurement / action |
|---|---|---|
| MCU GPIO for 4x TRIG and 4x ECHO | `TRIG_PINS`, `ECHO_PINS` in `usonic_config.py` (ship `None`: the MCU then reports zero sensors, fires nothing, and the host refuses the link) | wire the board, then fill in |
| JSN-SR04T revision and mode (trigger/echo vs serial) | audit M-40 | read the fitted boards; this firmware implements trigger/echo only |
| ECHO high level at the real supply; whether the RP2040 (3.3 V, not 5 V tolerant) needs a level shifter | audit M-40 | meter/scope with the MCU disconnected. Do not reuse the legacy 1k/2k divider (H-18) |
| Supply voltage and current | audit M-40, M-13 | |
| What a healthy sensor reports in open air (no pulse? a maximum-length pulse?) | `ultrasonic.no_echo_is_clear`, `out_of_range_is_clear` | `scripts/ultrasonic_mcu_probe.py --csv` with a clear field of view |
| Blind zone, beam pattern, range on grass/people/pets/low objects, rain | `ultrasonic.min_range_cm`, `max_range_cm`, mounts | same probe, flat targets at 10 to 40 cm and field targets |
| Echo window, quiet gap, firing order | `ultrasonic.echo_window_ms`, `inter_sensor_gap_ms`, `FIRING_ORDER` | M-40 timing; order depends on the physical layout (mount design, M-01) |
| Sensor positions, yaw, pitch | not in the repository | mount survey (B.5) |
| Host serial path of the data port | `ultrasonic.port` in `config/hardware.yaml` | `ls -l /dev/serial/by-id/` on the Pi with the MCU attached |
| CircuitPython release on the board (not pinned) and what the watchdog does after an exception there | `CIRCUITPY/boot_out.txt`; behaviour of `microcontroller.watchdog` | Read the first line of `boot_out.txt` and record the release. On a spare board run a `code.py` that arms the watchdog (`timeout = 2`, `mode = WatchDogMode.RESET`) and then raises `RuntimeError`: a reset about 2 s after the traceback means it stays armed (the 10.0 behaviour, adafruit/circuitpython#7722); a board that sits at "Code done running" means it is disarmed, and only an exception that ends `code.py` outside the loop is exposed (see Firing, health and the safety posture). Also confirm the `reset_reason` the host logs: SOFTWARE after a forced reset, WATCHDOG after a hang |
| How often `usb_cdc.data.write()` is short on the real board, and the TX FIFO size | `CpHal.short_writes`; `_WRITE_RETRIES`, `_WRITE_RETRY_WAIT_S` in `usonic_hal_cp.py` | From the REPL (Ctrl-C the running firmware): `import usonic_main; hal, core = usonic_main.build(); core.start()`, then call `core.poll()` in a loop for 60 s and print `hal.short_writes, core.frames_tx`. Do it once with the backend (or `cat` on the data port) reading, once with the data port opened but never read. Reading host: `short_writes` must stay 0. Stalled host: it grows, and the loop must keep running (the watchdog must not bite). A reading host with `short_writes` above 0 means the retry budget is too small for this board |
| `PulseIn` when ECHO is already high at `resume()`, and glitch widths | `CpHal.measure` in `usonic_hal_cp.py` | In CircuitPython `main` the RP2040 `PulseIn` samples the pin at 1 MHz in a PIO state machine and, with `idle_state=False`, starts counting at the first high sample, so a pulse already high at `resume()` is reported with its remaining high time only (read from the source, not from a board). Connect a signal generator (or a second board) to ONE ECHO pin with the sensor disconnected. From the REPL call `hal.measure(i, 30000)` while the generator holds the line high for 5 ms starting just before the call. `ST_OK` with a width of about the time it stayed high after `resume()` confirms that a late-falling line reads as a plausible short width, which the HAL cannot refuse (the host already turns widths below the blind zone into UNKNOWN, never clear). Also drive 1 us, 3 us and 100 us pulses and record the narrowest width `PulseIn` returns (0 us is refused as STUCK) |
| `os.urandom` on the target build | `CpHal.boot_id` | From the REPL: `import os; [os.urandom(4).hex() for _ in range(5)]`, and once right after each of three resets. Different random-looking values every time: fine. `NotImplementedError` or a repeated value after a reset: the weaker fallback id (clock mixed with the chip id) is in use, so check that the host still notices a reset (watchdog reset and a USB replug) |
| Actual TRIG pulse width | `TRIGGER_PULSE_US` in `usonic_config.py` (10 us) | Logic analyser or scope on one TRIG pin, read the high time of one measurement. The busy-wait polls `time.monotonic_ns()`, whose resolution on the RP2040 port is 1/32768 s, about 30.5 us (CircuitPython `main` source, not measured on a board), so expect roughly 30 to 100 us, not 10 us. The tests pin the configured width on a 1 us fake clock, not the board's. At least 10 us is needed to fire the sensor; check the fitted JSN-SR04T revision's datasheet for a maximum and use a hardware-timed pulse if it has one |

## Wire protocol, version 1

Little-endian. One frame:

```
0xA5 0x5A | VER | TYPE | SEQ | LEN | PAYLOAD (LEN <= 48) | CRC16 (2 bytes)
```

* `VER` protocol major version (1). Inside the CRC, so a frame from another version is recognised and
  refused rather than mistaken for noise.
* `SEQ` per-sender counter (wraps at 256); the receiver counts gaps as lost frames.
* `CRC16` CRC-16/CCITT-FALSE (poly `0x1021`, init `0xFFFF`, no reflection, no final xor) over
  `VER..PAYLOAD`, low byte first. `crc16(b"123456789") == 0x29B1`.
* Resynchronisation: a bad length or CRC drops exactly **one** byte and rescans for the next sync
  word, so a corrupted frame costs only itself. If the line goes quiet with an incomplete frame
  pending (a corrupted LEN), the rest is scanned once for hidden good frames and then discarded.

| Type | Dir | Payload | Meaning |
|---|---|---|---|
| `0x01 HELLO` | host to MCU | `host_version u8, flags u8` | answered with INFO |
| `0x02 CONFIG` | host to MCU | `enable_mask u8, echo_window_ms u16, gap_ms u16, quiet_after u8` | applied, then echoed in CONFIG_ACK. Out-of-range values: NAK, nothing changes |
| `0x03 PING` | host to MCU | `nonce u32` | answered with PONG |
| `0x81 INFO` | MCU to host | `fw_major u8, fw_minor u8, proto u8, sensor_count u8, boot_id u32, uptime_ms u32, reset_reason u8, enable_mask u8` | at boot, on HELLO, and every second as a heartbeat |
| `0x82 CONFIG_ACK` | MCU to host | same as CONFIG, the values now in force | the host verifies it equals what it asked (confirmed write) |
| `0x83 READING` | MCU to host | `sensor u8, status u8, health u8, fails u8, echo_us u32, cycle u16, t_ms u32` | one per sensor per cycle |
| `0x84 PONG` | MCU to host | `nonce u32, uptime_ms u32` | |
| `0x8F NAK` | MCU to host | `ref_type u8, reason u8` | reasons: 1 bad version, 2 bad length, 3 unknown type, 4 bad value |

`status`: 0 OK (complete pulse, `echo_us` is its width), 1 NO_EDGE (no pulse in the window: open air,
a dead sensor and a broken wire look the same), 2 NO_FALL (rising edge, no falling edge within the
window), 3 STUCK (line high before the trigger, a HAL fault, or an impossible pulse width: 0 us,
longer than the window or 65535; `echo_us` keeps the raw width), 4 DISABLED (masked off).
`health`: 0 ACTIVE, 1 QUIET (8 or more consecutive silent measurements; still fired every cycle),
2 FAULT, 3 DISABLED. A device-to-host frame type arriving at the MCU (a looped-back line) is ignored,
never answered, so it cannot start a NAK storm.

## Firing, health and the safety posture

* Strictly one sensor at a time. Invariant (tested over hundreds of triggers with mixed sensor behaviours):
  the next trigger is never earlier than the previous trigger plus the echo window, nor than the end
  of the previous measurement plus the gap, so a sensor cannot hear another's burst.
* Every enabled sensor is fired **every cycle**, including a silent or faulted one. A healthy sensor
  over open ground is silent too; skipping it would age its data. A dead sensor costs one window plus
  one gap per cycle (about 60 ms with the defaults).
* The MCU reports raw echo widths and electrical outcomes. **It does not decide what a reading
  means.** The host applies the blind zone, range, staleness, link supervision and the owner's
  open-air policy (`JsnSr04tMcuDriver`): anything not positively established is UNKNOWN, never clear.
* Hang and crash handling (`WATCHDOG_TIMEOUT_S` = 4 s, reset mode, fed once per loop pass):
  * A **hung** loop is reset by the hardware watchdog. It is armed for as long as `usonic_main.run()`
    has not returned, so this does not depend on the CircuitPython release.
  * A **crash** (any `Exception` out of the boot INFO, the watchdog feed or `core.poll()`) is caught in
    `run()`, which calls `CpHal.reboot()` (`microcontroller.reset()`): USB re-enumerates and INFO
    carries a new `boot_id` and reset reason SOFTWARE. The console shows
    `usonic: crashed, resetting: <exception>` when it can (the reset may cut it off). If the reset
    call itself fails, `run()` stops feeding the watchdog and spins, so the watchdog resets the MCU
    within 4 s (reset reason WATCHDOG).
  * After either reset the host drops all earlier data, re-handshakes and re-sends the configuration.
    Ctrl-C, Ctrl-D and autoreload (`KeyboardInterrupt`, `SystemExit`, the reload exception) are not
    caught and stop the program as usual.
  * **Not guaranteed:** an exception that ends `code.py` before the loop, for example a pin name that
    is not on the board or a missing `boot.py` in `build()`. These are configuration errors and keep
    their traceback on the console. Whether the watchdog then stays armed depends on the CircuitPython
    release (adafruit/circuitpython#7722: staying armed after an exception other than Ctrl-C or
    reload is the 10.0 behaviour); on an older release the MCU stays enumerated and silent until a
    power cycle. The host sees every channel UNKNOWN (fail closed) either way. **The CircuitPython
    release is not pinned and not verified on the board** (see the TBD table).
  * A reset is as abrupt as a power cycle. The firmware never writes to the CIRCUITPY drive, so the
    only exposure is a host that has that drive mounted read-write and is writing to it.
* USB writes: `usb_cdc.data.write()` returns how many bytes it took, fewer than offered when the TX
  FIFO is full (the host is not draining it). `CpHal.write()` completes a frame with up to 3
  non-blocking retries 1 ms apart (about 3 ms in all, so the loop keeps feeding the watchdog; the first
  attempt may still wait up to the 50 ms write timeout for FIFO room). A frame that still does not fit
  is abandoned where it is and counted in `CpHal.short_writes` (not reported on the wire; no frame
  bytes change). Its tail is never sent later: the next frame is written whole, right behind the
  stub, and the host's decoder resynchronises on its sync word. The truncated frame is lost, the next
  one decodes. UNVERIFIED on the board: the TX FIFO size and how often the host leaves it full (see
  the TBD table).
* Limits of this HAL, reported honestly: `pulseio.PulseIn` only records a pulse that has ended, so a
  line stuck high or a pulse longer than the window both read as NO_EDGE; NO_FALL and the pre-trigger
  STUCK check are in the protocol for HALs that can tell them apart.
* Impossible widths: `CpHal.measure` never returns ST_OK for a `PulseIn` width of 0 us, longer than the
  window, or 65535 (the 16 bit counter's saturation value): those are STUCK (FAULT) with the raw
  width in `echo_us`. A pulse already in progress when capture is armed that ends inside the window
  reads as a shorter, plausible width: the HAL has no level read of the ECHO pin and cannot tell
  (UNVERIFIED, see the TBD table).
* `boot_id` is four bytes of `os.urandom`. If the build has none (an exception), the boot goes on with
  a weaker id: the clock mixed with the chip's unique id. It differs between chips but is NOT
  guaranteed new on every boot, and the host detects an in-place reboot only through a changed
  `boot_id` (a reset normally also drops the USB port, which it sees as a lost link). UNVERIFIED on
  the board (see the TBD table).

## Host side

`backend/src/drivers/sensors/jsn_sr04t_driver.py` (link, handshake, reconnect, evaluation) and
`ultrasonic_mcu_protocol.py` (independent host codec). Configuration: the `ultrasonic:` block of
`config/hardware.yaml`; master switch `ultrasonic_enabled` in `config/limits.yaml` (**ships false**).
The handshake is HELLO, INFO (version and sensor count checked), CONFIG, CONFIG_ACK (compared with
what was asked); readings are ignored until it completes. A version or channel-count mismatch, a
rejected or mismatching configuration, a silent link and a vanished port each leave every channel
UNKNOWN and are reported in `health_check()`.

## Testing

```bash
SIM_MODE=1 pytest tests/unit/test_ultrasonic_mcu_protocol.py tests/unit/test_ultrasonic_mcu_firmware.py \
  tests/unit/test_ultrasonic_mcu_cp_hal.py tests/unit/test_jsn_sr04t_driver.py \
  tests/unit/test_ultrasonic_interface.py tests/unit/test_ultrasonic_mcu_probe.py -o addopts=""
```

Covered: wire bytes and CRC against a third bit-by-bit implementation; host and firmware codecs
byte-identical and decoding identically under random corruption; the firmware core against a
simulated HAL (time-division invariant, including a gap shorter than the window; the boot INFO and
the exact heartbeat period; the partial-frame timeout; dead/stuck/raising sensors, config accept/NAK,
loopback, reboot); the main loop's crash handling (forced reset, starved watchdog when the reset
fails, Ctrl-C/reload never swallowed); the CircuitPython HAL under fake
`board/digitalio/pulseio/usb_cdc/microcontroller` modules (call order, bounded waits, the TRIG width
on a 1 us clock, short USB writes, impossible `PulseIn` widths, a missing `os.urandom`, data port,
watchdog); the host driver against the real firmware core with corrupt, truncated, dropped and
garbage frames, a hung MCU, unplug and reconnect, an MCU reset, version and channel-count mismatch,
a dead sensor, and the fail-closed policy matrix.

**Not covered (needs the bench):** real GPIO and `PulseIn` timing on the RP2040, USB enumeration and
the by-id name, the JSN-SR04T's electrical behaviour, the cycle time actually achieved, coverage and
detection distance. Until M-40 is done the shipped policy keeps a silent channel UNKNOWN.

Bench helper for M-40: `python scripts/ultrasonic_mcu_probe.py --port <data port> --seconds 60 --csv out.csv`
logs every reading with the raw `echo_us`. The host driver opens the data port exclusively, so stop
the backend (and any other probe) first; otherwise the probe prints `open_failed` with a plain-words
hint, keeps retrying until its time is up and exits 2.
