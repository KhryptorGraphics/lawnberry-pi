"""CircuitPython (RP2040) hardware layer for the ultrasonic MCU.

UNVERIFIED ON HARDWARE. This file has only been exercised against fake CircuitPython modules
(``tests/unit/test_ultrasonic_mcu_firmware.py``). It mirrors the established
``adafruit_hcsr04`` pattern: TRIG pulsed from a ``DigitalInOut``, the echo width captured by
``pulseio.PulseIn`` (hardware timed, so a garbage-collection pause or a USB packet cannot stretch
the measurement the way a polling loop would). What a JSN-SR04T ECHO line really does (idle level,
pulse when nothing is in range, level at the fitted supply) is audit item M-40: nothing here
assumes it.

Limits of this HAL, reported honestly in the protocol:

* ``PulseIn`` only records a pulse that has *ended*. A line stuck high or a pulse longer than the
  window therefore both look like ``ST_NO_EDGE`` (silent). The protocol keeps ``ST_NO_FALL`` and
  ``ST_STUCK`` for HALs that can tell them apart; the host never treats a silent line as proof of
  a clear path unless the owner's configuration says so.
* A raised exception inside ``measure`` is reported by the core as ``ST_STUCK`` (fault).
* ``PulseIn`` can hand back widths no echo can have produced: 0 us (a glitch), a width longer than
  the window (the pulse began before ``resume()``) or the 16 bit counter's saturation value.
  ``measure`` reports those as ``ST_STUCK`` (a fault; the raw width stays in ``echo_us``), never
  as ``ST_OK``. A pulse already in progress at ``resume()`` that ends inside the window reads as a
  shorter, plausible width: the HAL cannot tell (UNVERIFIED, see the README).
* ``boot_id`` uses ``os.urandom``; a build without one still boots, with a weaker id.
* ``usb_cdc.Serial.write`` returns how many bytes it took, fewer than offered when the TX FIFO is
  full (the host is not draining it). ``write`` completes the frame with a few non-blocking
  retries (about 3 ms in all, so the main loop keeps feeding the watchdog). A frame that still
  does not fit is abandoned where it is and counted in ``short_writes``: the host resynchronises
  on the stub (it costs that frame) and the next frame is written whole, never glued to the
  missing tail of this one.
"""

import os
import time

import board
import digitalio
import microcontroller
import pulseio
import usb_cdc
import usonic_proto as P

try:  # CircuitPython >= 9
    from watchdog import WatchDogMode
except ImportError:  # CircuitPython <= 8
    from microcontroller import WatchDogMode

_WRITE_TIMEOUT_S = 0.05  # the first attempt may wait this long for room in the USB TX FIFO
_WRITE_RETRIES = 3  # then up to this many non-blocking attempts for the rest of the frame ...
_WRITE_RETRY_WAIT_S = 0.001  # ... one USB frame (1 ms) apart: about 3 ms at most
_PULSEIN_SATURATED_US = 65535  # PulseIn stores 16 bit widths: a longer pulse reads exactly this


class CpHal:
    def __init__(self, trig_pins, echo_pins, trigger_pulse_us=10, watchdog_s=4):
        if usb_cdc.data is None:
            raise RuntimeError("usb_cdc.data is disabled: copy boot.py and power-cycle the MCU")
        if len(trig_pins) != len(echo_pins):
            raise ValueError("TRIG_PINS and ECHO_PINS must have the same length")
        self._trig = []
        self._echo = []
        for name in trig_pins:
            io = digitalio.DigitalInOut(getattr(board, name))
            io.direction = digitalio.Direction.OUTPUT
            io.value = False
            self._trig.append(io)
        for name in echo_pins:
            pin = pulseio.PulseIn(getattr(board, name), maxlen=2, idle_state=False)
            pin.pause()
            pin.clear()
            self._echo.append(pin)
        self.sensor_count = len(self._trig)
        self.short_writes = 0  # frames abandoned because the host did not take all of them
        self._trigger_ns = int(trigger_pulse_us) * 1000
        self._serial = usb_cdc.data
        self._serial.timeout = 0
        self._serial.write_timeout = _WRITE_TIMEOUT_S
        self._wdt = microcontroller.watchdog
        self._wdt.timeout = watchdog_s
        self._wdt.mode = WatchDogMode.RESET

    # ---- time / identity -------------------------------------------------------------------

    def now_ms(self):
        return time.monotonic_ns() // 1000000

    def boot_id(self):
        try:
            return int.from_bytes(os.urandom(4), "little")
        except Exception:
            # No usable os.urandom on this build must not stop the boot. The fallback mixes the
            # clock with the chip's unique id: chips differ, but one chip's boots are nearly alike
            # (the time since boot is almost the same each time), so it is NOT guaranteed to be
            # new on every boot. UNVERIFIED on the board (README).
            value = time.monotonic_ns() & 0xFFFFFFFF
            try:
                for byte in microcontroller.cpu.uid:
                    value = (value * 33 + byte) & 0xFFFFFFFF
            except Exception:
                pass
            return value

    def reset_reason(self):
        reasons = microcontroller.ResetReason
        reason = microcontroller.cpu.reset_reason
        if reason == reasons.POWER_ON:
            return P.RESET_POWER_ON
        if reason == reasons.WATCHDOG:
            return P.RESET_WATCHDOG
        if reason == reasons.SOFTWARE:
            return P.RESET_SOFTWARE
        if reason == reasons.RESET_PIN:
            return P.RESET_PIN
        return P.RESET_UNKNOWN

    def feed_watchdog(self):
        self._wdt.feed()

    def reboot(self):
        """Reset the RP2040 now (``microcontroller.reset()``); on the MCU this never returns.

        ``usonic_main.run`` calls it after an unexpected exception. USB re-enumerates, the host sees
        a new ``boot_id`` and reset reason SOFTWARE, and the handshake starts again. Like a watchdog
        reset it is abrupt: nothing here writes to the CIRCUITPY drive.
        """
        microcontroller.reset()

    # ---- measuring -------------------------------------------------------------------------

    def measure(self, index, window_us):
        echo = self._echo[index]
        trig = self._trig[index]
        echo.pause()
        echo.clear()
        trig.value = True
        stop = time.monotonic_ns() + self._trigger_ns
        while time.monotonic_ns() < stop:
            pass
        trig.value = False
        echo.resume()
        limit = time.monotonic_ns() + window_us * 1000
        while len(echo) == 0:
            if time.monotonic_ns() > limit:
                echo.pause()
                echo.clear()
                return P.ST_NO_EDGE, 0
        echo.pause()
        width_us = echo[0]
        echo.clear()
        if width_us <= 0 or width_us >= _PULSEIN_SATURATED_US or width_us > window_us:
            # Not a measurement: no echo is 0 us wide, wider than the window it was captured in, or
            # exactly the counter's saturation value. A fault, never a distance.
            return P.ST_STUCK, int(width_us)
        return P.ST_OK, int(width_us)

    # ---- host link -------------------------------------------------------------------------

    def read(self):
        waiting = self._serial.in_waiting
        if waiting:
            return self._serial.read(waiting)
        return b""

    def write(self, data):
        """Send one frame without ever blocking for long (see the module docstring).

        ``usb_cdc.Serial.write`` returns how many bytes it took: fewer than ``len(data)`` when the
        TX FIFO is full. The rest is retried at most ``_WRITE_RETRIES`` times, non-blocking and
        ``_WRITE_RETRY_WAIT_S`` apart. A frame that still does not fit is abandoned where it is
        and counted in ``short_writes``; its tail is never sent later.
        """
        serial = self._serial
        if not serial.connected:  # never block on a host that is not listening (cable, closed port)
            return
        total = len(data)
        sent = serial.write(data)
        if sent >= total:
            return
        serial.write_timeout = 0
        try:
            for _ in range(_WRITE_RETRIES):
                time.sleep(_WRITE_RETRY_WAIT_S)
                sent += serial.write(data[sent:])
                if sent >= total:
                    return
        finally:
            serial.write_timeout = _WRITE_TIMEOUT_S
        self.short_writes += 1
