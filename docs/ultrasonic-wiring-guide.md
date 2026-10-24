# HC-SR04 Ultrasonic Sensor Wiring Guide (LEGACY — DO NOT WIRE)

> **LEGACY DESIGN — DO NOT WIRE FROM THIS GUIDE.** Marked legacy on 2026-10-24 by the integration audit
> ([`2026-10-24-integration-audit.md`](2026-10-24-integration-audit.md)). It is kept for history and as the
> record of the collisions below. It is **not** an approved wiring design. Platform scope: not specific to the
> differential-drive mower or the Toro tractor variant (H-11); it predates both requirements.
>
> 1. **Wrong part (H-19, H-10).** This guide covers **three HC-SR04** boards. The owner's hardware is **four
>    JSN-SR04T** waterproof modules: two inner sensors forward and two outer sensors angled outward on
>    adjustable mounts across the 50-inch Toro deck. The JSN-SR04T board revision, mode (trigger/echo vs
>    serial), supply voltage, ECHO logic level and near-range blind zone are **unverified**: read them from
>    the purchased boards and measure them before choosing any wiring. No JSN-SR04T driver exists;
>    `backend/src/drivers/sensors/ultrasonic_driver.py:80-84` is hard-coded to three HC-SR04 sensors.
> 2. **These pins collide with the HaLow radio on the deployed Pi (H-02).** Live Pi, read-only audit
>    2026-10-24: `/boot/firmware/config.txt` loads `dtoverlay=halow-dw` and `dtoverlay=spi0-1cs`; GPIO9/10/11
>    are SPI0 MISO/MOSI/SCLK (alt0, claimed by the SPI0 controller), GPIO8 is its chip-select, GPIO25 its
>    IRQ, GPIO5 its RESET, GPIO3 its WAKE and GPIO7 its BUSY. This guide puts ECHO on GPIO10 and GPIO9 and
>    TRIG on GPIO11, three of the radio's SPI lines. Driving them disrupts the HaLow link to the Thor
>    (`backend/src/services/sensor_manager.py:878-881` records that doing so "glitches the radio and has
>    taken it down on a backend restart"). `config/limits.yaml:17` also says the HAT owns GPIO4; the live
>    `config.txt`/overlay do **not** claim GPIO4 (it was unclaimed), so GPIO4 must be confirmed with the HAT
>    physically fitted before it is used or ruled out.
> 3. **Pin 17 is 3.3 V, not 5 V (H-01).** The VCC column below said "5V (Pin 17)" for sensor 3. This
>    guide's own pin diagram shows `3V3 (17)`. The 5 V header pins are 2 and 4 only.
> 4. **The divider schematic was ambiguous and its safety claim does not hold (H-18).** See "Voltage
>    Divider Circuit" below.
> 5. **Software still targets these pins.** `ultrasonic_driver.py:80-84` and
>    `backend/src/services/hw_selftest.py:36-41` hard-code them. The driver is held off by
>    `ultrasonic_enabled: false` (`config/limits.yaml:17`, fail-closed in
>    `sensor_manager.py:871-886`), but `GET /api/v2/system/selftest` (`backend/src/api/routers/maintenance.py:62-69`
>    → `hw_selftest.ultrasonic_probe()`, `hw_selftest.py:192-255`) ignores that flag. On a kernel where the
>    header GPIOs are `gpiochip4` it claims TRIG on GPIO4/27/11 as outputs and drives them low (GPIO11 is the
>    radio's SPI0 SCLK). On the deployed kernel the header is `gpiochip0`, `lgpio.gpiochip_open(4)` fails, and
>    only a gpiozero pin factory is created.
>
> Replacement requirements (four JSN-SR04T, GPIO allocation, coverage geometry, blind-zone and stopping
> distance measurements) are in `2026-10-24-integration-audit.md`, Part B.
>
> **Software for the replacement now exists (unverified on hardware):** four JSN-SR04T behind a dedicated
> RP2040 over USB serial, no Pi header pin: [`firmware/ultrasonic-mcu/README.md`](../firmware/ultrasonic-mcu/README.md)
> (protocol, TBD list for the owner), host driver `backend/src/drivers/sensors/jsn_sr04t_driver.py`, bench tool
> `scripts/ultrasonic_mcu_probe.py` for measurement M-40. No wiring is decided by this repository.

## Overview

This guide covers the wiring and installation of three HC-SR04 ultrasonic sensors for obstacle detection on the LawnBerry Pi autonomous mower. The sensors provide close-range detection (2-400cm) for safety and navigation.

## Components Required

- 3x HC-SR04 ultrasonic sensors
- 3x 1k ohm resistors (voltage dividers for ECHO pins; one per divider) — *the original list said "6x 1k", which
  is wrong: three dividers need three 1k and three 2k*
- 3x 2k ohm resistors (voltage dividers for ECHO pins)
- Jumper wires (male-to-female recommended)
- Breadboard or custom PCB for voltage dividers
- Cable ties for wire management
- Heat shrink tubing (optional, for weatherproofing)

## GPIO Pin Assignments

| Sensor | Position | VCC | TRIG GPIO | ECHO GPIO* | GND | Collision on the deployed Pi (2026-10-24 audit) |
|--------|----------|-----|-----------|------------|-----|------|
| 1 | Front-Left | 5V (Pin 2) | GPIO4 (Pin 7) | GPIO17 (Pin 11) | GND (Pin 6) | GPIO4: `limits.yaml:17` claims HaLow, live overlay does not (UNVERIFIED); GPIO17 unclaimed |
| 2 | Front-Center | 5V (Pin 4) | GPIO27 (Pin 13) | GPIO10 (Pin 19) | GND (Pin 14) | **GPIO10 = SPI0 MOSI (HaLow)**; GPIO27 unclaimed |
| 3 | Front-Right | ~~5V (Pin 17)~~ **invalid: Pin 17 is 3V3** (H-01); 5 V pins are 2 and 4 | GPIO11 (Pin 23) | GPIO9 (Pin 21) | GND (Pin 20) | **GPIO11 = SPI0 SCLK, GPIO9 = SPI0 MISO (both HaLow)** |

*ECHO pins REQUIRE a 5V-to-3.3V conversion with margin - see below. Do not use the 1k/2k divider as written.

## Voltage Divider Circuit (REQUIRED for ECHO pins)

The HC-SR04 operates at 5V but the Raspberry Pi GPIO pins are 3.3V only. Connecting 5V directly to a GPIO pin will damage the Pi. A voltage divider is required for each ECHO pin.

### Circuit Diagram (corrected 2026-10-24)
```
HC-SR04 ECHO ---[ 1k ohm ]---+--- Pi GPIO (ECHO input)
                             |
                          [ 2k ohm ]
                             |
                            GND
```
The original diagram drew the 2k resistor beside the 1k (both from the ECHO node, the 2k ending at a node
tied to `GND`), which can be read as the 2k being in parallel with the 1k or the GPIO node being tied to
ground. The divider is: ECHO → 1k in series → GPIO node → 2k to GND. The "Breadboard Layout" below was
already correct.

### Calculation and margin (H-18)
- Nominal: 5.00 V x (2k / (1k + 2k)) = 3.33 V. That is **already above** the nominal 3.3 V GPIO rail; the
  original text called it "safe ... (max 3.3V)", which it is not.
- If the 5 V rail sits at 5.25 V (+5 %): 5.25 V x 2/3 = **3.50 V**. With 5 % resistors at the unlucky
  extremes (1k at -5 %, 2k at +5 %): 5.25 V x 2.1k/(0.95k + 2.1k) = **3.61 V**.
- Raspberry Pi GPIO is 3.3 V logic and not 5 V tolerant. Use a purpose-built level shifter, or a divider
  chosen with margin, after the JSN-SR04T ECHO high level is **measured** with the module powered from the
  real supply and the Pi side disconnected. Do not reuse 1k/2k on the strength of this guide.

### Breadboard Layout (per sensor)
```
  HC-SR04 ECHO
       |
       +--[1k]--+--[GPIO]
                |
              [2k]
                |
              [GND]
```

## Installation Steps

### 1. Prepare Voltage Dividers
1. For each sensor, create a voltage divider on a breadboard or solder to a small PCB
2. Test each divider with a multimeter: input 5V should output ~3.3V
3. Label each divider for its sensor position

### 2. Mount Sensors on Mower
1. Position sensors on front bumper with 15-20cm spacing
2. Sensors should face forward with slight downward angle (10-15 degrees)
3. Use sensor mounting brackets or 3D-printed housings
4. Ensure sensors have clear line of sight (no obstructions)

### 3. Wire Power
1. Connect VCC pins to 5V rail (multiple sensors can share)
2. Connect GND pins to ground rail (can share with voltage dividers)
3. Use sufficient gauge wire for power (22-24 AWG recommended)

### 4. Wire TRIG Pins
1. Connect TRIG pins directly to GPIO pins (no voltage divider needed)
2. TRIG pins are inputs to the sensor (3.3V is sufficient to trigger)

### 5. Wire ECHO Pins
1. Connect each ECHO pin to its voltage divider input
2. Connect voltage divider output to corresponding GPIO pin
3. Double-check all connections before powering on

### 6. Secure Wiring
1. Use cable ties to bundle and route wires cleanly
2. Apply heat shrink to exposed solder joints
3. Consider a weatherproof enclosure for outdoor operation

## Testing

### Software Test (Simulation Mode)
```bash
# Test driver in simulation mode first
cd /apps/lawnberry-pi
source .venv/bin/activate
SIM_MODE=true python -c "
import asyncio
from backend.src.drivers.sensors.ultrasonic_driver import UltrasonicDriver
driver = UltrasonicDriver()
asyncio.run(driver.initialize())
asyncio.run(driver.start())
print(asyncio.run(driver.read_all()))
"
```

### Hardware Test
> **Do not run this on the deployed Pi.** It claims GPIO4/17/27/10/11/9, including the radio's SPI0 pins
> (see item 2 in the banner).
```bash
# Test with actual hardware (requires wiring complete)
python -c "
import asyncio
from backend.src.drivers.sensors.ultrasonic_driver import UltrasonicDriver
driver = UltrasonicDriver()
asyncio.run(driver.initialize())
asyncio.run(driver.start())
for i in range(10):
    print(asyncio.run(driver.read_all()))
    asyncio.sleep(0.5)
"
```

## Troubleshooting

### No readings / Always max distance
- Check VCC and GND connections
- Verify TRIG pin connections
- Ensure voltage divider is working (measure with multimeter)

### Erratic readings
- Check for electrical noise (keep wires away from motors)
- Add decoupling capacitors (100nF) near sensor VCC pins
- Ensure sensors have clear line of sight

### All sensors show same value
- Check for wiring shorts between sensors
- Verify each sensor has unique GPIO assignments

### Pi not booting or GPIO errors
- IMMEDIATELY disconnect all sensors
- Check for short circuits
- Verify voltage dividers are correct (5V to 3.3V)
- Test voltage divider outputs with multimeter before reconnecting

## Pin Reference

### Physical Pin Numbers (40-pin header)
```
           3V3  (1)  (2)  5V  [VCC Sensor 1]
         GPIO2  (3)  (4)  5V  [VCC Sensor 2]
         GPIO3  (5)  (6)  GND [GND Sensor 1]
  [TRIG1] GPIO4  (7)  (8)  GPIO14
               GND  (9) (10)  GPIO15
 [ECHO1] GPIO17 (11) (12)  GPIO18
  [TRIG2] GPIO27 (13) (14)  GND [GND Sensor 2]
        GPIO22 (15) (16)  GPIO23
           3V3 (17) (18)  GPIO24
 [ECHO2] GPIO10 (19) (20)  GND [GND Sensor 3]
  [ECHO3] GPIO9 (21) (22)  GPIO25
  [TRIG3] GPIO11 (23) (24)  GPIO8
```

## Safety Notes

1. **Never connect 5V directly to GPIO** - Always use voltage dividers for ECHO pins
2. **Power off before wiring** - Disconnect Pi power when making connections
3. **Double-check before power-on** - Verify all connections match this guide
4. **Test voltage dividers first** - Use multimeter to confirm 3.3V output

## Audit corrections log (2026-10-24)

| Item | What the guide said | Correction | Evidence |
|---|---|---|---|
| Sensor 3 VCC | 5V (Pin 17) | Pin 17 is 3V3; 5 V pins are 2 and 4 | this file's pin diagram; H-01 |
| Resistor BOM | 6x 1k, 3x 2k | 3x 1k, 3x 2k for three dividers | arithmetic; H-18 |
| Divider diagram | 2k drawn beside the 1k | 1k in series to the GPIO node, 2k from that node to GND | H-18 |
| Divider safety | 3.33 V "safe (max 3.3V)" | 3.33 V nominal exceeds 3.3 V; 3.50-3.61 V at +5 % supply / 5 % parts | arithmetic above |
| Sensors | 3x HC-SR04 | owner hardware is 4x JSN-SR04T (unverified variant) | H-19, H-10 |
| GPIO9/10/11 | free ultrasonic pins | SPI0 lines owned by the HaLow radio | live Pi audit; H-02 |
