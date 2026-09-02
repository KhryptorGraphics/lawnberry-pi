# LawnBerry Pi — Hardware Shopping List

Bill of materials for the autonomous zero-turn mower build, with buy links.

> Prices are **live from Amazon** via the `amazon` MCP scraper, rounded to the
> dollar. Listings change often — **re-check before ordering.**

## Already have
- Raspberry Pi 5
- RTK GPS (u-blox ZED-F9P), BNO085 IMU, ToF sensors, camera, INA3221
- Hailo-8L AI accelerator

## Need to buy

### 1. Heltec WiFi LoRa 32 — ⚠️ V2 is effectively gone; **V3 is the successor**
A live Amazon search for the **V2 (SX1276)** returns **only V3/V4 (SX1262)** —
the V2 has been discontinued/superseded and the one V2 listing I had is now a
dead link (404). The **V3 is pin-compatible** with the V2 (same form factor,
WiFi/BLE/LoRa/OLED; SX1262 radio, USB-C). For the **US choose 915 MHz
(902–928 MHz).** If you specifically need the SX1276 V2, you'll have to source it
used or from AliExpress — it's not on Amazon anymore.

Cheapest current options (all V3/V4, SX1262):

| Option | Notes | Price | Link |
|--------|-------|-------|------|
| **Cheapest/unit — AITRIP V3 2-pack + case** | ~$13 each; spare = ground-station radio | ~$26 / 2 | [amazon.com/dp/B0CWNCKXSL](https://www.amazon.com/AITRIP-Development-Dual-core-Protective-Compatible/dp/B0CWNCKXSL) |
| V3 module **+ 915 MHz antenna** | Cheapest single that includes an antenna | ~$22 | [amazon.com/dp/B07HD1CRPD](https://www.amazon.com/Display-ESP-32S-Bluetooth-Development-Transceiver/dp/B07HD1CRPD) |
| **Genuine Heltec V4** (latest) | 27 dBm, ESP32-S3 + SX1262 | ~$23 | [amazon.com/dp/B0FS1R4HXH](https://www.amazon.com/Heltec-Development-Display-Meshtastic-Communication/dp/B0FS1R4HXH) |
| Genuine Heltec **V3** 902–928 MHz | Authentic V3, 5.0★ | ~$27 | [amazon.com/dp/B0D1H1FN9Y](https://www.amazon.com/Heltec-Development-863-870MHz-ESP32-S3FN8-902-928MHz/dp/B0D1H1FN9Y) |

**Pick:** the **AITRIP V3 2-pack (~$13 each)** for lowest cost + a spare, or the
**genuine Heltec V4 (~$23)** for the authentic latest board.

---

## 2. Zero-turn conversion BOM

> **Supersedes the Craftsman/Ackermann parts list.** The platform is now a
> **50" Toro TimeCutter gas-engine zero-turn** with twin-lever hydrostatic
> drive — see `docs/tractor-platform.md`. The old list called for five RC-PWM
> channels (steering, throttle, gas pedal, clutch, gear); the zero-turn needs
> **three** (left lever, right lever, throttle) plus two relays.

The mower's engine and hydrostatic transaxles stay **stock**. Three servos
physically actuate the existing operator controls. There are **no drive motors
and no motor-driver/H-bridge stage** — these servos accept RC PWM (0.5–2.5 ms)
directly, so the PCA9685 *is* the layer between the Pi and the actuators.

### Actuation

| Qty | Part | Price | Link |
|---|---|---|---|
| 3 | **DSSERVO RDS51150SG** 150 kg 12 V servo — left lever, right lever, throttle. 165 kg·cm @ 12 V, 0.21 s/60°, mounting brackets included | $36 ea | [B0C69W2QP7](https://www.amazon.com/ANNIMOS-Voltage-Digital-Steering-Brackets/dp/B0C69W2QP7/) |
| 1 | **Adafruit PCA9685** 16-ch PWM driver — I²C at 0x41 (0x40 collides with the INA3221) | $14.95 | [adafruit.com/product/815](https://www.adafruit.com/product/815) |

Servo *speed* is why this part was chosen over the Wingxine ASMC-04B.
Constitution Principle VI requires positional actuators to *physically reach*
their safe position within 500 ms of an E-stop. At 0.21 s/60° that budget buys
~140° of rotation at 12 V; the ASMC-04B's 1.0 s/60° at 12 V buys only ~30°, and
would need a 24 V boost converter to comply.

### Power

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | **Pololu D24V50F5** 5 V/5 A regulator — Pi 5 supply. Bare 5 V output, which sidesteps the Pi 5 USB-C power-negotiation quirk | $32.95 | [pololu.com/product/2851](https://www.pololu.com/product/2851) |
| 1 | Inline fuse holders + assorted fuses, 12 AWG | $6 | [B0FDJYRGB7](https://www.amazon.com/Cooclensportey-Inline-Holder-Waterproof-Standard/dp/B0FDJYRGB7/) |
| — | 12–14 AWG wire, ring lugs, heat shrink | ~$15 | local |

### Engine & implement

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | Opto-isolated 30 A relay module — GPIO interface stage | $9 | [B0CHFJSNP6](https://www.amazon.com/HiLetgo-Channel-Optocoupler-Isolation-Trigger/dp/B0CHFJSNP6/) |
| 1 | Bosch-style 40 A relay + sealed harness, 2-pack — load stage (starter, blade PTO) | $8 | [B093GMF6M1](https://www.amazon.com/Hamolar-Pack-Relay-SPDT-Harness/dp/B093GMF6M1/) |

A Pi GPIO cannot drive a 40 A automotive relay coil directly. The chain is
GPIO → opto module → Bosch relay coil → load.

### Linkage — buy it, don't print it

| Qty | Part | Link |
|---|---|---|
| 4 | M6 rod ends / heim joints — 2 male, 2 female | [SA6TK male](https://www.amazon.com/uxcell-Bearing-M6x1-0-Joint-Thread/dp/B07WM39B6R) · [PHS6 female](https://www.amazon.com/uxcell-Bearing-Joint-Female-Thread/dp/B0BDG6S9YP) |
| — | M6 threaded rod, clevis pins + R-clips | local |
| 2 | **Square U-bolts** for the 3" × 1.5" frame rail, 5/16" leg — servo mount + e-stop bracket | hardware store |
| 2 | **Round U-bolts** for the lap-bar tube, 1/4" leg — size to your measured OD | hardware store |

Ball joints at both rod ends are not optional — they give the angular freedom
the lever needs, and the clevis pin lets you unpin the linkage in seconds to
restore full manual PARK travel.

U-bolts do all the clamping. Every mount is a saddle: the printed part locates
and spreads load, steel takes the tension. A printed clamp ear carrying bolt
preload on a vibrating petrol machine is the weakest possible arrangement, and
U-bolts also absorb the fact that Toro does not publish the lap-bar tube OD.

### Safety & enclosure

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | E-stop mushroom button, 1NC/1NO, 2-pack | $11 | [B07R9QTBG7](https://www.amazon.com/mxuteuk-Mushroom-Emergency-Warranty-HB2-ES545/dp/B07R9QTBG7/) |
| 1 | IP65 enclosure, 220×170×110 mm | $19 | [B08KWFYQQR](https://www.amazon.com/Zulkit-Dustproof-Waterproof-Electrical-220x170x110/dp/B08KWFYQQR/) |
| — | **Bus-fault watchdog** — 555 + relay + passives. No suitable off-the-shelf module found; Amazon "watchdog" results are cycle timers, which are the wrong part | ~$10 | electronics supplier |

**≈ $240 total** for the conversion hardware.

### Wiring rules

- **Never run servo power through the PCA9685's `V+` terminal.** These servos
  stall at 8 A each; that board's trace and terminal block will not carry 16 A.
  Battery → 12–14 AWG → servos direct, on a 20 A fuse. Signal wire only to the
  PCA9685 header, common ground.
- Three separately fused taps off the 12 V battery: Pi regulator, servo rail,
  relay module.

### Mounts

Parametric OpenSCAD sources, rendered STLs and preview images live in
`hardware/mounts/`. The **frame rail dimension is researched and real**
(3" × 1.5" × 0.120" wall, Toro's published carrier-frame spec); the **lap-bar
tube OD is not published by Toro** and must be measured — edit `common.scad`
and re-render. Print the saddles in PETG or ASA; buy the U-bolts and linkage.

---
### Sources (live Amazon via the `amazon` MCP scraper)
- LoRa: [AITRIP V3 2-pack](https://www.amazon.com/AITRIP-Development-Dual-core-Protective-Compatible/dp/B0CWNCKXSL) ·
  [V3 + antenna](https://www.amazon.com/Display-ESP-32S-Bluetooth-Development-Transceiver/dp/B07HD1CRPD) ·
  [Heltec V4](https://www.amazon.com/Heltec-Development-Display-Meshtastic-Communication/dp/B0FS1R4HXH) ·
  [Heltec V3 902–928](https://www.amazon.com/Heltec-Development-863-870MHz-ESP32-S3FN8-902-928MHz/dp/B0D1H1FN9Y)
- Prices and stock change; re-check before ordering.
