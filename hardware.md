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
| — | **Enclosure — printed**, `hardware/mounts/enclosure_body.scad` + `enclosure_lid.scad`. Nothing to buy | — | see `hardware/mounts/README.md` |
| 1 | **3 mm silicone O-ring cord**, ~1.2 m — the enclosure gasket. Cord stock, cut and butt-joined in the groove | ~$9 | hardware/industrial supplier |
| 12 | M4×16 screws + M4 heat-set inserts — lid fixing | ~$12 | hardware store |
| 1 | **IP68 cable glands, PG9 10-pack** (4–8 mm cable) — every cable entry | $8 | [B0FC2XJ4CW](https://www.amazon.com/Anyinn-PG9-Waterproof-Connectors-Locknut/dp/B0FC2XJ4CW/) |
| 1 | **IP68 breather vent, M12×1.5, 2-pack** — pressure equalisation | $6 | [B0F4NM8NT5](https://www.amazon.com/2-Pack-IP68-Industrial-Breather-Vent/dp/B0F4NM8NT5/) |
| — | **Bus-fault watchdog** — 555 + relay + passives. No suitable off-the-shelf module found; Amazon "watchdog" results are cycle timers, which are the wrong part | ~$10 | electronics supplier |

**≈ $256 total** for the conversion hardware (enclosure now printed, not bought).

### Weatherproofing — the part that actually matters

The enclosure is **printed** (`hardware/mounts/enclosure_body.scad` +
`enclosure_lid.scad`), not bought. A printed box can be made to seal well, but
only because it does not rely on the plastic to seal: a 3 mm silicone O-ring
cord sits in a groove in the body's flange and the lid compresses it, with the
lid screws outboard of the groove so tightening squeezes the cord rather than
bowing the lid off it.

**The walls still have to be watertight.** Printed walls leak along layer lines
if under-extruded. Print hot and slow enough that layers fuse properly, use 5+
perimeters, and if in doubt wipe the inside with epoxy or acrylic conformal
coat. A gasket cannot rescue a porous wall.

Beyond the box itself, outdoor enclosures rarely fail by bulk ingress through
the seal; they fail two other ways, and neither is fixed by a better gasket:

1. **Water tracking in along a cable.** Every wire entering the box is a leak
   path unless it goes through a gland. Hence the PG9 glands — one per entry,
   no exceptions, and no drilling a hole and stuffing wire through it.
2. **Condensation.** A sealed box heated by the engine and afternoon sun, then
   cooled overnight, pumps moist air in and out and condenses it on the
   coldest surface inside. A *perfectly* sealed box is actually worse, because
   the water that forms has no way to leave. Hence the breather vent — it
   passes water vapour and equalises pressure while blocking liquid.

Two rules when you mount it:

- **Glands face DOWN.** Mount the enclosure so every cable entry is on the
  bottom face, and leave a drip loop in each cable below its gland so water
  runs off the low point instead of tracking up into the fitting.
- **Vent on the bottom or a side face**, never the top, and never where it can
  be sprayed directly by the deck discharge.

The printed tray (`hardware/mounts/electronics_tray.scad`) assumes water gets
in eventually: it stands on feet so anything liquid pools on the enclosure
floor *below* the boards, drains its own surface through perimeter and corner
slots, carries the boards on 12 mm standoffs clear of the floor, and has
cable tie-down slots so cable movement never works a gland seal or a connector
loose.

Heat is the other half. A sealed box holding a Pi 5 and a Hailo accelerator
will run hot with no airflow — mount the enclosure out of direct sun, in
whatever airflow the machine has, and check the Pi's thermal throttling in the
first hot-weather run before trusting it to a long mow.

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
(3" × 1.5" × 0.120" wall, Toro's published carrier-frame spec). The **lap-bar
tube OD is not published by Toro anywhere** — so the lap-bar saddle uses a
self-centring 90° V-groove that seats 12.8–34 mm (0.50"–1.34") instead of a
guessed radius. Only the linkage standoff geometry needs measuring. Print the
saddles in PETG or ASA; buy the U-bolts and linkage.

---
### Sources (live Amazon via the `amazon` MCP scraper)
- LoRa: [AITRIP V3 2-pack](https://www.amazon.com/AITRIP-Development-Dual-core-Protective-Compatible/dp/B0CWNCKXSL) ·
  [V3 + antenna](https://www.amazon.com/Display-ESP-32S-Bluetooth-Development-Transceiver/dp/B07HD1CRPD) ·
  [Heltec V4](https://www.amazon.com/Heltec-Development-Display-Meshtastic-Communication/dp/B0FS1R4HXH) ·
  [Heltec V3 902–928](https://www.amazon.com/Heltec-Development-863-870MHz-ESP32-S3FN8-902-928MHz/dp/B0D1H1FN9Y)
- Prices and stock change; re-check before ordering.
