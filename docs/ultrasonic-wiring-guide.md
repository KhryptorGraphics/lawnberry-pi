# HC-SR04 Ultrasonic Sensor Wiring Guide

## Overview

This guide covers the wiring and installation of three HC-SR04 ultrasonic sensors for obstacle detection on the LawnBerry Pi autonomous mower. The sensors provide close-range detection (2-400cm) for safety and navigation.

## Components Required

- 3x HC-SR04 ultrasonic sensors
- 6x 1k ohm resistors (voltage dividers for ECHO pins)
- 3x 2k ohm resistors (voltage dividers for ECHO pins)
- Jumper wires (male-to-female recommended)
- Breadboard or custom PCB for voltage dividers
- Cable ties for wire management
- Heat shrink tubing (optional, for weatherproofing)

## GPIO Pin Assignments

| Sensor | Position | VCC | TRIG GPIO | ECHO GPIO* | GND |
|--------|----------|-----|-----------|------------|-----|
| 1 | Front-Left | 5V (Pin 2) | GPIO4 (Pin 7) | GPIO17 (Pin 11) | GND (Pin 6) |
| 2 | Front-Center | 5V (Pin 4) | GPIO27 (Pin 13) | GPIO10 (Pin 19) | GND (Pin 14) |
| 3 | Front-Right | 5V (Pin 17) | GPIO11 (Pin 23) | GPIO9 (Pin 21) | GND (Pin 20) |

*ECHO pins REQUIRE voltage dividers (5V to 3.3V) - see below

## Voltage Divider Circuit (REQUIRED for ECHO pins)

The HC-SR04 operates at 5V but the Raspberry Pi GPIO pins are 3.3V only. Connecting 5V directly to a GPIO pin will damage the Pi. A voltage divider is required for each ECHO pin.

### Circuit Diagram
```
HC-SR04 ECHO ----+---- 1k ohm ----+---- Pi GPIO (ECHO)
                 |                |
                 +---- 2k ohm ----+---- GND
```

### Calculation
- Output Voltage = 5V x (2k / (1k + 2k)) = 5V x (2/3) = 3.33V
- This is safe for Raspberry Pi GPIO input (max 3.3V)

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
cd /home/kp/repos/lawnberry_pi
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
