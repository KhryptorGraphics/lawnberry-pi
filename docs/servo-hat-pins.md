# Which HAT pins carry the drive-servo signals

Planning note, 2026-10-23. Nothing here has been measured on the mower; the Pi 5 is offline, so every
claim is either from this repo, from a datasheet, or explicitly marked as needing a bench check.

**Scope assumed:** the mower platform — RoboHAT MM1 (RP2040-Zero on a custom carrier) driving a Cytron
MDDRC10. If you meant the Toro lever servos, jump to the last section; that path is the PCA9685, not the HAT.

**Platform tag (H-11):** §1–§6 are the **differential-drive mower** (RoboHAT → MDDRC10). §7 is the **Toro TimeCutter
tractor variant**, which is a different wiring harness. Neither section is an as-built record.

## 1. What the motor controller actually needs

The MDDRC10 is not a PWM/DIR driver — it takes **RC servo pulses** on RC1/RC2 (Cytron datasheet Rev 1.0):

| Parameter | Value | Consequence |
|---|---|---|
| Input signal | RC PWM, 10–100 Hz | The RoboHAT's 50 Hz output is valid |
| High level | 1.5–15 V (max) | 3.3 V logic is valid; no level shifter needed |
| Deadband / full scale | ±35 µs / ±435 µs around the captured neutral | 1000/1500/2000 µs spans full range with margin |
| Mixed mode | RC1 = forward/reverse, RC2 = left/right | Matches the firmware's steer/throttle model |
| Independent mode | RC1 = left motor, RC2 = right motor | Does **not** match the firmware's model |
| `+5V` pin | 500 mA **output** for an RC receiver | Never feed 5 V into it |
| Neutral | Auto-calibrated from the signal at power-up | Power-up order decides the neutral reference |

## 2. What the firmware already assigns

`robohat-rp2040-code/code.py` (RP2040-Zero, CircuitPython):

| GPIO | Role | Notes |
|---|---|---|
| GP2–GP7 | RC receiver inputs (`PulseIn`) | ch6, ch5, ch4, ch3, ch2, ch1 respectively |
| GP8, GP9 | Wheel encoder A/B (`rotaryio`) | |
| **GP10** | **Servo out — steer** | 50 Hz, 1000–2000 µs, driven by `pwm,<steer>,<throttle>` |
| **GP11** | **Servo out — throttle** | same |
| GP12 | Blade PWM | separate from the IBT-4 path |
| GP13 | Mode switch, digital out | |
| GP16 | NeoPixel (WS2812 on the Zero) | PIO, not PWM |

The backend sends the same two numbers over USB serial at `/dev/ttyACM0`, 115200 baud
(`RoboHATService`), as `pwm,<steer>,<throttle>` while RC is disabled.

## 3. The wiring that follows

Mixed mode is what the software's steer/throttle semantics assume, and the firmware's pin order is
`steer, throttle` — so the cable is **crossed**:

```
RoboHAT GP11 (throttle) ──► MDDRC10 RC1   (forward / reverse)
RoboHAT GP10 (steer)    ──► MDDRC10 RC2   (left / right)
RoboHAT GND             ──► MDDRC10 GND
(MDDRC10 +5V: leave unconnected — it is an output)
```

If the driver is instead used in independent mode, GP10/GP11 would be left/right motors, and the
steer/throttle semantics in the backend become wrong. Pick mixed.

## 4. Finding those two pins on this carrier (no Pi required)

The carrier's schematic is not in this repo, so the RP2040 pin → header pin mapping has to be established
on the bench. The firmware itself is the signal generator, which makes this a two-minute job:

1. Unplug the Pi, disconnect the traction battery, and leave the MDDRC10 unpowered. Power the RoboHAT
   from USB only, so a wrong probe cannot move a motor.
2. Open the USB serial console to the RP2040 (115200) and send:
   - `rc=disable` — hands control to the USB commands
   - `pwm,1000,2000` — GP10 = 1000 µs, GP11 = 2000 µs, renewed every loop
3. With a DMM on DC volts (black lead on a known HAT GND), probe the header pins. A 50 Hz pulse train on
   a 3.3 V pin reads as its average:

   | Pulse | Duty | DMM reading |
   |---|---|---|
   | 1000 µs | 5 % | ≈ 0.165 V |
   | 1500 µs | 7.5 % | ≈ 0.248 V |
   | 2000 µs | 10 % | ≈ 0.330 V |

   Exactly one pin reads ≈0.17 V and exactly one reads ≈0.33 V. Send `pwm,2000,1000` and watch them swap —
   that proves the pair and which is which, without trusting a silkscreen.
4. If a scope or logic analyser is available, skip step 3's arithmetic and read the pulse widths directly.
5. Power off and confirm continuity from those header pins to the RP2040-Zero's GP10/GP11 castellations
   if the board has to be documented. `pwm,1500,1500` should leave both at ≈0.248 V and the drive at rest.

Note the naming trap: **Pi GPIO10/11** on the 40-pin header are owned by the HaLow HAT
(`config/limits.yaml`), while **RP2040 GP10/11** are these servo outputs. Two different chips, same numbers.

## 5. More servo channels

The RP2040-Zero exposes GP0–GP15 and GP26–GP29 on its edge (GP16 is the onboard LED; GP17–GP25 are pads
only). Free on the edge today: **GP0, GP1, GP14, GP15, GP26, GP27, GP28, GP29**. Assignment constraints:

- PWM slice = `(gpio >> 1) & 7`; every servo runs 50 Hz, so sharing a slice is harmless — only the
  frequency must match, and it does.
- GP10/GP11 → slice 5; GP12/GP13 → slice 6; GP16/GP17 → slice 0; GP26/GP27 → slice 5; GP28/GP29 → slice 6.
- GP0/GP1 are UART0 by default, but this build talks to the Pi over USB CDC, so they are usable if the
  carrier does not route them elsewhere.
- Practical picks for two more channels: **GP14 and GP15** (slice 7, adjacent, no other peripheral).

Adding channels is firmware + backend work, not wiring alone: a third/fourth `pwmio.PWMOut`, a command
that carries the extra values, a neutral-on-timeout path, and a matching field in `RoboHATService`. Keep
every new channel's fail-safe at 1500 µs, as the two existing ones are.

## 6. Hazards this plan has to respect

- **Power-up order vs. MDDRC10 auto-calibration.** The driver captures neutral from whatever it sees when
  it powers up. The RP2040 only reaches neutral after CircuitPython boots and `main()` runs. Power the
  HAT first (Pi/USB 5 V), let it output 1500 µs, then close the battery path to the MDDRC10 (E-stop relay),
  or the driver may calibrate a wrong neutral. Verify on the bench: power-cycled in the wrong order, the
  drive will not sit still at `pwm,1500,1500`.
- **RC inputs are 3.3 V only.** GP2–GP7 read an RC receiver directly. RP2040 I/O is not 5 V tolerant; a
  5 V receiver needs a divider or a 3.3 V receiver. The MDDRC10 inputs are the reverse case — they accept
  up to 15 V, so 3.3 V into RC1/RC2 is fine.
- **The E-stop must cut motor power, not command neutral.** Software `pwm,1500,1500` depends on the USB
  link, the RP2040's watchdog and the firmware loop; none of those is a safety device.
- **Grounds.** RoboHAT, MDDRC10 and the battery negative must share one ground; the servo/RC cable is a
  single-ended signal and floats without it.
- **Keep the RC/servo pair away from the HaLow HAT's SPI lines.** Those share the 40-pin header region,
  and this project already has one documented case of radio damage from adjacent pin traffic.

## 7. If the target is the Toro lever servos instead

> **Platform tag: Toro TimeCutter variant (H-11).** Corrected 2026-10-24 by the integration audit; the earlier
> wording is kept (struck through) with the corrections listed under it.

That platform has no motor controller — `hardware.md` and `docs/hardware-overview.md` place the lever
servos on a **PCA9685 at I2C address 0x41 on bus 1** (channels 0/1/2), and the relays on GPIO 5 (starter)
and GPIO 26 (blade PTO). Its "HAT pins" are therefore the Pi header's I2C pins: SDA1/SCL1 (pins 3/5),
plus 5 V/GND — ~~and the 24 V servo rail must stay off the PCA9685's `V+` terminal~~.

**Corrections (2026-10-24):**

- **Servo rail (H-05, H-20).** The 24 V statement is superseded. The servos are 12 V-class (`hardware.md`
  Actuation; owner-reported nominal 10–12.6 V, possibly 8 A or more stall current, purchased label pending).
  Do not route a servo supply through the PCA9685 `V+` terminal unless that breakout's `V+` path is proven for
  the measured stall current; the default is direct wiring from a separately fused branch to each servo, with
  only the PWM signal on the PCA9685 header and a common ground.
- **Pins 3/5 are not usable as configured on the deployed Pi (H-17, H-25).** Live Pi, read-only audit
  2026-10-24: `/boot/firmware/config.txt` has `dtparam=i2c_arm=off` and GPIO3 (pin 5) is the HaLow HAT's WAKE
  line, so `/dev/i2c-1` does not exist (only `i2c-11`, `i2c-13`, `i2c-14`, none on the header). The Toro control
  path needs a different I2C route, which is undecided (`2026-10-24-integration-audit.md`, Part B.1 and B.4).
- **GPIO5 (starter relay, pin 29) is the HaLow radio's RESET line and GPIO25 its IRQ on the deployed Pi
  (H-02b).** Do not connect a relay input to GPIO5 until that is resolved.
