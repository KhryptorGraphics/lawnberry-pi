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
| 2 | Opto-isolated 30 A relay module, 2-pack — GPIO interface stage (4 channels: starter, blade PTO, watchdog servo-cutoff, 1 spare) | $9 ea = $18 | [B0CHFJSNP6](https://www.amazon.com/HiLetgo-Channel-Optocoupler-Isolation-Trigger/dp/B0CHFJSNP6/) |
| 2 | Bosch-style 40 A relay + sealed harness, 2-pack — load stage (4 relays: same allocation) | $8 ea = $16 | [B093GMF6M1](https://www.amazon.com/Hamolar-Pack-Relay-SPDT-Harness/dp/B093GMF6M1/) |

A Pi GPIO cannot drive a 40 A automotive relay coil. The chain is
GPIO → opto module → Bosch relay coil → starter / blade PTO. Bumped from one
2-pack each to two: the bus-fault watchdog below needs its own channel to cut
servo power, and reusing this exact opto-module → relay pattern for that
channel — rather than a bespoke transistor-and-flyback-diode driver stage —
keeps every relay-switched load in the build opto-isolated the same way.

### Linkage and mounting

| Qty | Part | Price | Link |
|---|---|---|---|
| 8 | **M6 × 35 bolts, washers and steel nuts** — four per solid arm root, inserted from under the free 8 mm layer ears into side-entry nut slots at z=16–22.5; verify washer land and actual nut grip before ordering | Unpriced | Hardware store |
| 8 | **M6 × 35 bolts + locking nuts** — four per arm joining the flanged/spigoted lower-arm/head lap; check real washer stack, engagement and spigot clearance | Unpriced | Hardware store |
| 12 + 12 | **12 × M5 × 25 pad bolts/nuts and 12 × M4 × 10–12 short jaw screws/nuts** — two M5 per pad into side-access steel nuts and two M4 per handed half-jaw into top-loaded nuts. Print three left/right pairs; fit from +Y. M4 tips bear outside the tower tube: never drill its USB cable wall. Verify actual bolt lengths at fit-up | Unpriced | Hardware store |
| 14 + 2 | **14 × M6 × 55 and 2 × M6 × 25 bolts + washers/locking nuts** — per rod, four spigot-splice bolts, two positive-lock bolts and one clamp-yoke pin at 55 mm; one crank pin through the servo's metal arm and rod eye at 25 mm. Verify grip length/engagement at fit-up | Unpriced | Hardware store |
| 8 | **M5 × 50 bolts, broad washers and locking nuts** — four through-bolts per fully printed lap-bar clamp; starting length for the current 45.8 mm clamp stack, verify with actual washers and nuts | Unpriced | Hardware store |
| 8 | **M2.5 screws + nuts** — four per RDS51150 stationary-holder pattern on the **lateral** servo face at each arm head; choose length for the actual holder/washer stack | Unpriced | Hardware store |
| 1 set | **Square U-bolts M8**, nominal 40 mm inside width — the E-stop fixture only; verify rail and ≥102.6 mm free leg reach | Reprice; formerly $11 per 4-pack | [B0DT98CFW8](https://www.amazon.com/Square-Length-Plated-Carbon-Washers/dp/B0DT98CFW8/) |

Each 940 mm rod is three bolted 40 mm tube pieces plus a sliding 29.1 mm inner bar; two steel M6 bolts
form its positive length lock. Each rod uses one M6 crank pin through the servo's metal arm and one at
the clamp's paired yoke/eye joint. Four M5 clamp bolts per steering bar grip a user-measured tube.

The original fit-gauge rail envelope **76.2 × 38.1 mm is not confirmed** for this Toro. The E-stop
U-bolt and bracket are not part of the new servo-arm load path.

### Safety

| Qty | Part | Price | Link |
|---|---|---|---|
| 1 | E-stop mushroom button 1NC/1NO, 2-pack | $11 | [B07R9QTBG7](https://www.amazon.com/mxuteuk-Mushroom-Emergency-Warranty-HB2-ES545/dp/B07R9QTBG7/) |
| 1 | **74HC123 dual retriggerable monostable**, DIP-16, 2-pack — bus-fault watchdog | $6 | [B09KJJRWKC](https://www.amazon.com/74HC123-HD74HC123AP-SN74HC123N-MM74HC123AN-DIP-16/dp/B09KJJRWKC/) |
| 1 | IP68 cable glands PG9, 10-pack (4–8 mm cable) | $8 | [B0FC2XJ4CW](https://www.amazon.com/Anyinn-PG9-Waterproof-Connectors-Locknut/dp/B0FC2XJ4CW/) |
| 1 | IP68 breather vent M12×1.5, 2-pack | $6 | [B0F4NM8NT5](https://www.amazon.com/2-Pack-IP68-Industrial-Breather-Vent/dp/B0F4NM8NT5/) |

**Correction from an earlier pass**: this was originally specced as an NE555
described as "a retriggerable monostable" — that description is wrong. A
plain 555 does not retrigger cleanly off a repeating pulse train; the
textbook circuit that does ("missing pulse detector") is a specific,
fussier wiring of it, not a stock monostable. The 74HC123 genuinely *is* a
retriggerable monostable — it's the correct part for this job, not a
workaround. The more purpose-built alternative, a dedicated supervisor IC
(Analog Devices' MAX6369 family — pin-selectable timeout, built for exactly
this), was checked and isn't sold on Amazon (eBay/Newark only), so it didn't
make this list.

**Circuit**: feed the Pi's continuous heartbeat pulse train (any free GPIO,
software-side this is `backend/src/safety/tractor_safety_monitor.py`'s
20 Hz tick — see the constitution follow-up in that module) into the
74HC123's A trigger input with /CLR tied high; each pulse retriggers the RC
timing network before it can time out. Size R/C for a timeout comfortably
above the 20 Hz (50 ms) tick but short enough to matter — a 50 kΩ/1 µF pair
lands near a 200 ms window per the datasheet's `t = 0.45×R×C` — then take
the output to a spare channel on the opto-relay module above, cutting the
servo power rail on timeout. Loss of pulses — hang, crash, power loss, dead
I²C — must remove servo power through the independent hardware path. A geared servo
is not guaranteed to go limp or let the fitted lap bars return to neutral when unpowered.
The exact mower and complete fitted linkage must demonstrate neutral return; until then
this is an **unproven failsafe**, not manufacturer-confirmed system behaviour.
Preserve OEM interlocks. See `docs/tractor-acceptance-criteria.md` item 10.

### Enclosure — printed, not bought

| Qty | Part | Price | Link |
|---|---|---|---|
| — | Body (integral hitch tongue), **295 × 285 × 8 mm sandwich-layer center panel with two separate arms**, lid, tray, **4 relay sleds + 2 utility sleds**, camera tower (base + N segments + cap + backing template) | — | [Printable inventory and assembly](hardware/mounts/README.md) |
| 1 | **3 mm silicone O-ring cord**, 10 ft — approximately 634 mm perimeter plus trimming/splice allowance | $17 historical | [B096N67R2D](https://www.amazon.com/118-Silicone-Durometer-Ring-Stock/dp/B096N67R2D/) |
| 12 each | **M4 screws, ordinary hex nuts and flat washers** — lid; nominal M4 ×12 starting length, verify actual stack | Unpriced | Heat-set inserts are no longer used |
| 4 sets | M4 tray fasteners + sealing washers/sealant — add the 8 mm sandwich-layer thickness to the measured grip length; seal floor penetrations | Unpriced | Do not reuse short bolts blindly |
| 1 | OEM-rated bolt/thread for the measured ball-hole stack, including the adapter layer | Unpriced | Hole/thread and engagement unverified; fit `hitch_gauge` first |
| 0 or 2 | M6 auxiliary through-bolts only if the actual plate has matching holes; otherwise another positively keyed frame restraint must be designed | Unpriced | A single center bolt is not an approved powered-steering torque path |
| 6 + 2 | M5 wall bolts (6) and longer foot bolts (2) through the tongue + 8 mm layer; measure grip length and nut engagement | Unpriced | Hardware store |
| 1 strip | ≥2 mm metal, ~110 × 130 mm — backing inside the box wall behind the tower flange, from the printed template | Unpriced | Own source |
| 4 per joint | M4 × 16 + nuts — tower segment flanges (N+1 joints); plus 2 × M4 × 20 for the camera foot on the cap | Unpriced | Hardware store |
| 1 | **USB camera** + cable long enough for the tower height plus the run inside the box | Unpriced | The owned CSI Pi camera cannot make this run; a USB camera is required for the tower |

The enclosure reserves **four individual single-channel relay boards**, one per deck at 40 mm pitch.
Two utility decks sit above the Pi/Hailo reservation. PCB mounting patterns remain unknown: transfer
them to removable drilling carriers, using insulating spacers. Validate actual board/cable envelopes
before printing the body. Interior is 169 ×128 ×180 mm; lid footprint 219 ×178 mm. This is not a rated
enclosure. The center panel matches the box-footprint flange and continues to the hitch tongue; two
separate arms carry outboard servos. Plate dimensions, anti-rotation, bar OD, and linkage locations
remain physical measurement/qualification gates. The camera tower routes USB through the enclosure;
see [§ Camera tower](hardware/mounts/README.md#camera-tower).

---

## Totals

| Build | Cost |
|---|---|
| **Two servos** (manual throttle) | **$236** |
| Three servos (remote throttle) | **$272** |

These are **historical electronics totals**, not a complete current build quote. The
new printed adapter, arms, clamps and rods, their metal fasteners, and any needed
hitch anti-rotation hardware are unpriced. Do not treat these totals as an order-ready BOM.

Already owned, nothing to buy: Raspberry Pi 5, Hailo-8L, ZED-F9P RTK GPS,
BNO085 IMU, camera, ToF sensors, INA3221.

### Where the savings came from

| Change | Saved |
|---|---|
| Generic PCA9685 instead of Adafruit | $10 |
| DROK buck instead of Pololu D24V50F5 (**read the warning above**) | $25 |
| Printed enclosure instead of the Zulkit IP65 box | $19 |
| Throttle servo dropped (optional) | $36 |

The linked electronics listings remain sourcing references, not a complete purchase order.
All steering/hitch interfaces still require measurement and qualification; a positive hitch
torque restraint may require additional metal hardware. Prices/variants must be rechecked.

### What I would not cheap out on

The buck converter, for the reason above. Everything else on this list is a
commodity where the generic and the name-brand do the same job; a power supply
feeding the compute that steers a 644 lb machine is not.

---

## Weatherproofing

The printed enclosure has a 3 mm silicone cord in a constant-width 3.8 ×2.4 mm
groove, with screws/nut pockets outboard of the seal. It is **not IP-rated**.
Printing quality, cord splice/compression, gland and vent compatibility, and all
fastener penetrations require unpowered ingress testing. Conformal coating is not
a substitute for a demonstrated enclosure seal; use a rated commercial enclosure
where a documented ingress rating is required.

Water can track along cables or enter through porous layers and fasteners.
Temperature cycling can produce condensation. Proper glands, downward drip loops
and a suitable breather help manage these risks; a breather does not guarantee
dryness or provide sufficient cooling for a Pi/Hailo stack.

Two rules when you mount it:

- **Glands face DOWN.** Every cable entry on the bottom face, with a drip loop
  below each gland so water runs off the low point instead of tracking up.
- **Vent on the bottom or a side face**, never the top, and never where the
  deck discharge can spray it.

The tray sits 10 mm above the floor on body supports, with through-bores and cable
pass-throughs. The four body/tray fasteners penetrate the floor and need sealing
washers/sealant. Component boards use insulating spacers on the removable sleds.
Strain-relieve cables without blocking terminal access, ventilation or drain paths.

Heat is the other half. A sealed box holding a Pi 5 and a Hailo runs hot with
no airflow — mount out of direct sun and check thermal throttling on the first
hot-weather run before trusting it to a long mow.

## Mounts

Parametric OpenSCAD sources, **42 STL variants**, fit coupons and assembly previews are in
[`hardware/mounts/`](hardware/mounts/README.md); machine measurement and installation guidance is in
[`hardware/mounts/INSTALL.md`](hardware/mounts/INSTALL.md). The target is the Toro TimeCutter MAX
50 in MyRIDE, model 77502. The prototypes include a box-footprint
hitch sandwich panel, splayed servo arms with printed sway bars back to the camera base mount,
four-bolt printed lap-bar clamps and positively locked three-piece rods. Their 940 mm length is an
estimate in [`TORO_77502_LINKAGE.md`](hardware/mounts/TORO_77502_LINKAGE.md). Each clamp's paired
yoke ears support the rod eye in double shear with an M6 through-bolt. The single-axis pins stay
parallel to the mower's left-right axis; a built skew carries the lap bar's outboard offset.
Mower/plate dimensions, bar OD, servo crank, rod endpoints and full
travel remain measurement gates. **A positive anti-rotation hitch attachment has not been established;
powered steering is not approved by these files.** The user-owned `estop_bracket.3mf` remains untouched
and must be resliced against its current STL.
All parts above are live Amazon listings scraped via the `amazon` MCP tool.
Prices and stock change — re-check before ordering.
