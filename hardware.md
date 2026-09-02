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

## 2. Zero-turn conversion BOM — all Amazon, cheapest sourcing

> **Supersedes the Craftsman/Ackermann parts list.** The platform is a **50"
> Toro TimeCutter gas-engine zero-turn** with twin-lever hydrostatic drive —
> see `docs/tractor-platform.md`. The old list called for five RC-PWM channels
> (steering, throttle, gas pedal, clutch, gear); the zero-turn needs **three**
> (left lever, right lever, throttle) plus two relays.

The mower's engine and hydrostatic transaxles stay **stock**. Servos actuate
the existing operator controls. There are **no drive motors and no
motor-driver/H-bridge stage** — these servos take RC PWM (0.5–2.5 ms) directly,
so the PCA9685 *is* the layer between the Pi and the actuators.

Every line below is an Amazon link with a price. Nothing is left as "hardware
store" or "electronics supplier".

### Actuation

| Qty | Part | Price | Link |
|---|---|---|---|
| 2 | **DSSERVO RDS51150SG** 150 kg 12 V servo — left + right drive levers. 165 kg·cm @ 12 V, 0.21 s/60°, brackets included | $36 ea = **$72** | [B0C69W2QP7](https://www.amazon.com/ANNIMOS-Voltage-Digital-Steering-Brackets/dp/B0C69W2QP7/) |
| 1 | **PCA9685** 16-ch PWM driver, generic | $5 | [B07RMTN4NZ](https://www.amazon.com/Dorhea-PCA9685-Interface-Controller-Raspberry/dp/B07RMTN4NZ/) |
| *(1)* | *Third servo for throttle — **optional**, see below* | *+$36* | *same as above* |

**Skip the throttle servo to save $36.** A ZTR throttle is set-and-forget: you
push it to FAST before mowing and leave it. The software holds it at a fixed
value anyway (`tractor_engine_throttle`, default 0.75) rather than modulating
it. Set it by hand, leave PCA9685 channel 1 unused, and nothing else changes.
Add it later if you want remote throttle.

Servo *speed* is why this part beats the Wingxine ASMC-04B. Constitution
Principle VI requires positional actuators to *physically reach* their safe
position within 500 ms of an E-stop. At 0.21 s/60° that budget buys ~140° of
rotation at 12 V; the ASMC-04B's 1.0 s/60° buys only ~30°, and would need a
24 V boost converter to comply.

### Power

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | **DROK buck converter** 9–36 V → 5.2 V, 3.5–6 A — Pi 5 supply | $8 | [B01NALDSJ0](https://www.amazon.com/Converter-DROK-Regulator-Inverter-Transformer/dp/B01NALDSJ0/) |
| 1 | Inline fuse holders + assorted fuses, 12 AWG, 4-pack | $6 | [B0FDJYRGB7](https://www.amazon.com/Cooclensportey-Inline-Holder-Waterproof-Standard/dp/B0FDJYRGB7/) |
| 1 | 14 AWG primary wire, red + black, 50 ft total | $8 | [B07D74Z4R1](https://www.amazon.com/American-Gauge-Copper-Wire/dp/B07D74Z4R1/) |

⚠️ **This is the one line where cheapest carries real risk.** The Pololu
D24V50F5 ($32.95, pololu.com) has published dropout and thermal curves; the
DROK is a generic module whose 6 A claim is a marketing number. A Pi that
browns out mid-mow is a safety event on a 644 lb machine, not an
inconvenience. If you buy the DROK, **load-test it before trusting it**: run
the Pi + Hailo at full tilt and confirm 5 V holds and the module isn't
scorching. If it sags, spend the extra $25.

### Engine & implement

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | Opto-isolated 30 A relay module, 2-pack — GPIO interface stage | $9 | [B0CHFJSNP6](https://www.amazon.com/HiLetgo-Channel-Optocoupler-Isolation-Trigger/dp/B0CHFJSNP6/) |
| 1 | Bosch-style 40 A relay + sealed harness, 2-pack — load stage | $8 | [B093GMF6M1](https://www.amazon.com/Hamolar-Pack-Relay-SPDT-Harness/dp/B093GMF6M1/) |

A Pi GPIO cannot drive a 40 A automotive relay coil. The chain is
GPIO → opto module → Bosch relay coil → starter / blade PTO.

### Linkage and mounting

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | **M6 rod ends**, 4 pcs with jam nuts — ball joints for both pushrod ends | $9 | [B0C7N2N5MN](https://www.amazon.com/uxcell-Female-Bearing-Thread-Self-Lubricating/dp/B0C7N2N5MN/) |
| 1 | **M6 threaded rod**, 300 mm, 2 pcs — cut to length for the pushrods | $7 | [B0CW6769K7](https://www.amazon.com/M6-1-0-300mm-Threaded-Threads-Stainless/dp/B0CW6769K7/) |
| 1 | **Square U-bolts M8**, 40 mm (1.5") wide, 4 sets — servo mounts + e-stop bracket onto the 3"×1.5" frame rail | $11 | [B0DT98CFW8](https://www.amazon.com/Square-Length-Plated-Carbon-Washers/dp/B0DT98CFW8/) |
| 1 | **Round U-bolts**, 1/4" × 1" wide, 8 sets — lap-bar saddles | $9 | [B0G4LXR4KD](https://www.amazon.com/SVLING-u-Bolts-Stainless-Washers-Trailer/dp/B0G4LXR4KD/) |

Ball joints at **both** rod ends are not optional — they give the angular
freedom the lever needs, and a clevis pin at the lever end lets you unpin the
linkage in seconds to restore full manual PARK travel.

U-bolts do all the clamping. Every printed mount is a saddle: the plastic
locates and spreads load, steel takes the tension. Square U-bolts wrap the
rail's 1.5" dimension (the saddle sits on the 3" face); round ones pull the lap
bar into the printed V-groove.

### Safety

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | E-stop mushroom button 1NC/1NO, 2-pack | $11 | [B07R9QTBG7](https://www.amazon.com/mxuteuk-Mushroom-Emergency-Warranty-HB2-ES545/dp/B07R9QTBG7/) |
| 1 | **NE555 timer**, 10-pack — bus-fault watchdog | $6 | [B00K243MIQ](https://www.amazon.com/Texas-Instruments-NE555P-Single-Precision/dp/B00K243MIQ/) |
| 1 | IP68 cable glands PG9, 10-pack (4–8 mm cable) | $8 | [B0FC2XJ4CW](https://www.amazon.com/Anyinn-PG9-Waterproof-Connectors-Locknut/dp/B0FC2XJ4CW/) |
| 1 | IP68 breather vent M12×1.5, 2-pack | $6 | [B0F4NM8NT5](https://www.amazon.com/2-Pack-IP68-Industrial-Breather-Vent/dp/B0F4NM8NT5/) |

The watchdog is a retriggerable monostable: the Pi emits a continuous pulse
train, and if the pulses stop — hang, crash, power loss, dead I²C — the timer
expires and drops a relay carrying servo power. The servos go limp and the lap
bars' own return springs pull the levers to neutral. That spring return is
manufacturer-documented (see `docs/tractor-platform.md`), which is what makes
this failsafe viable. Use a spare channel on the relay module above; the
555 plus a few passives is the only part you assemble yourself.

### Enclosure — printed, not bought

| Qty | Part | Price | Link |
|---|---|---|---|
| — | `hardware/mounts/enclosure_body.scad` + `enclosure_lid.scad` + `electronics_tray.scad` | — | print them |
| 1 | **3 mm silicone O-ring cord**, 10 ft — the gasket (needs ~1.2 m) | $17 | [B096N67R2D](https://www.amazon.com/118-Silicone-Durometer-Ring-Stock/dp/B096N67R2D/) |
| 1 | **M4 heat-set inserts + screws**, 261 pcs with insert tips | $13 | [B0G8X7GGBJ](https://www.amazon.com/Ktehloy-261Pcs-M4-Threaded-Inserts/dp/B0G8X7GGBJ/) |

---

## Totals

| Build | Cost |
|---|---|
| **Two servos** (manual throttle) | **$213** |
| Three servos (remote throttle) | **$249** |

Already owned, nothing to buy: Raspberry Pi 5, Hailo-8L, ZED-F9P RTK GPS,
BNO085 IMU, camera, ToF sensors, INA3221.

### Where the savings came from

| Change | Saved |
|---|---|
| Generic PCA9685 instead of Adafruit | $10 |
| DROK buck instead of Pololu D24V50F5 (**read the warning above**) | $25 |
| Printed enclosure instead of the Zulkit IP65 box | $19 |
| Throttle servo dropped (optional) | $36 |

Everything previously listed as "hardware store", "local" or "electronics
supplier" — U-bolts, rod ends, threaded rod, wire, silicone cord, inserts,
the 555 — is now a priced Amazon line. The list is complete as written.

### What I would not cheap out on

The buck converter, for the reason above. Everything else on this list is a
commodity where the generic and the name-brand do the same job; a power supply
feeding the compute that steers a 644 lb machine is not.

---

## Weatherproofing

The enclosure is **printed** (`hardware/mounts/enclosure_body.scad` +
`enclosure_lid.scad`). A printed box seals fine, but only because it does not
rely on the plastic to seal: a 3 mm silicone O-ring cord sits in a groove in
the body's flange and the lid compresses it, with the lid screws outboard of
the groove so tightening squeezes the cord rather than bowing the lid off it.

**The walls still have to be watertight.** Printed walls leak along layer lines
if under-extruded. Print hot and slow enough that layers fuse, use 5+
perimeters, and if in doubt wipe the inside with epoxy or acrylic conformal
coat. A gasket cannot rescue a porous wall.

Beyond the box itself, outdoor enclosures rarely fail by bulk ingress through
the seal. They fail two other ways, and neither is fixed by a better gasket:

1. **Water tracking in along a cable.** Every wire entering the box is a leak
   path unless it goes through a gland. Hence the PG9 glands — one per entry,
   no exceptions.
2. **Condensation.** A sealed box heated by the engine and afternoon sun, then
   cooled overnight, pumps moist air in and out and condenses it on the coldest
   surface inside. A *perfectly* sealed box is worse, because the water that
   forms has no way to leave. Hence the breather vent.

Two rules when you mount it:

- **Glands face DOWN.** Every cable entry on the bottom face, with a drip loop
  below each gland so water runs off the low point instead of tracking up.
- **Vent on the bottom or a side face**, never the top, and never where the
  deck discharge can spray it.

The printed tray assumes water gets in eventually: it stands on feet so liquid
pools on the enclosure floor *below* the boards, drains its own surface through
perimeter and corner slots, carries boards on 12 mm standoffs, and has cable
tie-down slots so cable movement never works a gland seal or connector loose.

Heat is the other half. A sealed box holding a Pi 5 and a Hailo runs hot with
no airflow — mount out of direct sun and check thermal throttling on the first
hot-weather run before trusting it to a long mow.

## Mounts

Parametric OpenSCAD sources, STLs and preview renders are in
`hardware/mounts/`. The **frame rail dimension is researched and real**
(3" × 1.5" × 0.120" wall, Toro's published carrier-frame spec). The **lap-bar
tube OD is not published by Toro anywhere**, so the lap-bar saddle uses a
self-centring 90° V-groove seating 12.8–34 mm (0.50"–1.34") instead of a
guessed radius. Only the linkage standoff geometry needs measuring. Print the
saddles in PETG or ASA.

---
### Sources
All parts above are live Amazon listings scraped via the `amazon` MCP tool.
Prices and stock change — re-check before ordering.
