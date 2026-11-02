# control_77603: electronics chassis, compute tower and interface carriers

PROTOTYPE - FIT / LOAD / ELECTRICAL VALIDATION REQUIRED. Geometry only: no retention force,
stiffness, vibration, thermal or ingress claim is made beyond what `control_checks.scad` shows
with real solids.

Everything here lives inside the UNCHANGED legacy enclosure body (`../enclosure_body.scad`:
interior 169 x 128 x 180, floor top z 4.5, lid ribs down to z 180.5). Body coordinates are used
throughout. No purchased fastener is added; the four existing box-to-arm-layer M4 bolts are reused.

## Files

| File | Purpose |
|---|---|
| `parameters.scad` | Every dimension (published value + URL, DESIGN value, or ESTIMATE / TO MEASURE), layout tables, strain checks |
| `cleat_carrier.scad` | Carrier side of the `../electronics_rail.scad` French cleat: hook, key ribs, latch tongue, feet; shared helpers |
| `chassis.scad` | `chassis_base()` (U-frame under the bolt heads) and `chassis()` (plate, panels, strips, board clips, tower seat) |
| `compute_tower.scad` | `tower_rack()` and `tower_clamp()` holding the Alta stack under the Pi 5 stack |
| `board_carriers.scad` | `strap_bay()`, `harness_clamp()`, and the printable dispatch (`carrier_kind`, `carrier_part = "body"`) |
| `board_refs.scad` | Reference solids of the boards (published outlines, ESTIMATE heights) for scenes and probes |
| `assemblies.scad` | Scenes: `electronics_packaging`, `electronics_contents`, `electronics_exploded`, `electronics_lid_lifted` |
| `control_checks.scad` | CSG probes (`-D 'check="name"' [-D idx=N]`) |

## Printed pieces (6 designs, 11 prints)

| Piece | Qty | Bodies | Print pose |
|---|---|---|---|
| `chassis_base` | 1 | 1 | flat, hooks up |
| `chassis` | 1 | 1 | plate on the bed, panels vertical |
| `tower_rack` | 1 | 1 | shelf on the bed |
| `tower_clamp` | 1 | 1 | upside down, frame top on the bed |
| `strap_bay` | 6 | 2 (PIP strap bar) | upright as installed |
| `harness_clamp` | 1 | 2 (PIP strap bar) | upright as installed |

## Stack order (compute tower, bottom up)

Libre Computer Alta -> its HaLow HAT -> [kit standoffs] -> Raspberry Pi 5 + Active Cooler -> AI HAT+
on its supplied 16 mm stacking header -> Waveshare LC29H(DA) on ONE purchased 2x20 extra-tall stacking
header (it replaces the LC29H's female-only header so GPIO passes up) -> HaLow HAT on top.
The owner's kit M-F standoffs screw to each other through every board at the Pi hole pattern; they
are spacers, never anchors. Spacings are ESTIMATES (`s_*` in `parameters.scad`). Alternative (owner
decision, NOT the default): move the LC29H off the stack on USB; that changes the GNSS port in software.

The rack's four bosses sit on flexure tabs (the stack's spring); spigots (2.3 x 1.2, shear only)
locate the Alta's holes. The clamp is a ladder frame with four pads (3.2 bores take the top standoff
threads) and four ratchet pawls (two on the front posts, two on the back wall). Push it down onto the
stack; 1.5 mm ratchet steps; travel 12 mm shorter / 3 mm taller than nominal without a reprint.

## Layout (seated)

- Base: rails x +-54..70, y -58.5..60 plus a +Y crossbar, z 14.5..18 on the boss tops, clamped by the
  existing bolts (3.5 mm, as the legacy tray). Four hooks at (+-60, -40), (+-60, 20); detent nub at (62, -9).
- Chassis plate z 18..22; -Y panel (cleat face y -60, to z 160), side panels (faces x +-80.5, y -12.5..56.6).
- Tower: board u -> +X (RJ45/USB face +X, plug room x 42.5..80), v -> +Y (USB-C/HDMI edge at y -6.6,
  plug room x -42.5..-27, y -37..-7), shelf top z 29, clamp top z 124.7 nominal / 127.7 max.
- -Y panel, tier 1 (seated ledge bottom z 31.03): B1, B2 = HiLetgo 30A relays (capacity 90 / 75 tall).
- -Y panel, upper bays (each stays under the lid antenna stations, z <= 161.5):
  B3 = steering converter L, ledge bottom z 114.03 over the Pololu, capacity 44;
  B4 = steering converter R, ledge bottom z 128.03 over the tower clamp's finger band, capacity 47.5;
  B5 = battery-to-USB-C (Pi), ledge bottom z 109.53 over B2's relay, capacity 48.5.
- -X side panel, above the tower's -X end: B6 = battery-to-USB-C (Alta), capacity 47.5 tall.
- -Y panel, x -83..-43: harness clamp (z 31..67), Pololu #2806 clipped above it (z 85..110).
- +X side panel (top z 169, under the lid post socket): PCA9685 (Pi) and PCA9685 (Alta) clipped
  upright, z 102..164, servo plugs facing -X.
- Kept free: floor-gland keep-outs (d 26 to z 30.5), camera-port mouth (x +-15, y 56..64, z 85..135),
  camera-tower M5 nuts (x +-40, y >= 55, z 40/90/140, ESTIMATE), breather nut (+X wall y 28, z 30),
  lid ribs, and under the lid the two antenna stations (30 mm round (+-64.5, -48), z >= 162) and
  the lid post-socket bosses (x +-[31.5, 82.5], y -18..4, z >= 170), copied from AntennaMount's
  antenna_lid_keepout().

## Pi 5 <-> Alta Ethernet patch

The two RJ45 jacks face +X one above the other (29 mm apart, jack faces at x 42 / 42.6). A STANDARD
straight patch cable does NOT fit: its boot ends about 32 mm past the jack face (x 74), 6.5 mm short of
the +X panel, and a 6 mm round Cat6 cable needs about a 25 mm bend radius. Buy instead:
- one FLAT Cat6 patch (about 1.5 x 6.5 mm) or a SLIM one (<= 4 mm OD), **0.15 m**;
- with **90-degree side-angle plugs at both ends, cable leaving toward -Y** (toward the back panel).
The cable leaves both plugs into the free channel x 46..79.5, y -12.4..-5.2, z 37..79 (between the
jacks and bays B2/B5) and its slack coils there; bend radius assumed >= 12 mm. Plug body 13.5 x 11.7,
10 mm inside the jack and 20 mm housing past it are ESTIMATES: check them on the cable you buy against
`patch_clear` / `patch_plug_seats`. The Alta's RJ45 position is ASSUMED from its Pi 3B+-style layout.

## HaLow pigtails (antennas on the lid)

Each radio has its own Rokland antenna on an N bulkhead in the lid near its back corners, at
(-64.5, -48) for the Alta's HaLow HAT and (+64.5, -48) for the Pi's. Each pigtail (N bulkhead to the
HAT's RF connector, U.FL or SMA as fitted, TO CONFIRM) leaves its station toward +Y at z about 164.5,
runs over bay B3 / B5 to y -12, along the bays' front (y -15..-9, z 162..169.5, under the lid post
sockets) to x -20, forward over the tower clamp to the +Y wall and down the gap y 56.5..62.5
(x -21.5..-15.5) to the HAT RF ends (Alta HAT z about 50..55, Pi HAT z about 110..115). Every corridor
junction leaves room for a 10 mm bend radius (ASSUMED for 1.13 mm micro-coax).
**Buy two pigtails of 400 mm** (route about 270 mm for the Alta, 250 mm for the Pi, plus the
bulkhead drop and service slack).
**The lid lifts with both antennas: unplug both pigtails at the HATs before lifting the lid.**

## Strap bay capacity (one print, x6)

Box width along the panel 18..43.6, depth 16.5..26.5 (from the rib tips), height >= 20 up to each
slot's capacity. Ratchet 1.5 mm steps; the spring leaf takes up the step with 1.0..2.5 mm preload.
Fit: stand the box on the ledge against the ribs, push the strap bar on. Release: squeeze both pawl
finger tops toward the middle and pull the bar off. Harness clamp: bundle 8..26 mm, runs vertically.

## Assembly order

1. Clamp `chassis_base` on the four boss tops under the existing M4 bolt heads + washers.
2. Lower the chassis 6 mm +Y of its seat (the hooks rise through its windows) and push it -Y until
   the base detent nub pops into the plate hole at (62, -9).
3. Clip the two PCA9685 and the Pololu into their pockets (bottom corners first, top clips snap over).
4. Hang the tier-1 bays (B1, B2) and the harness clamp: drop each hook into its strip's V, press the
   plate flat until the latch tooth clicks under the catch land. Load the relays; clamp the bundle.
5. Build the board stack on the kit standoffs; drop it onto the rack spigots; push the clamp on.
6. Lower the tower 6 mm -X of its seat (keyhole entries over the two mushroom studs, the toe presses
   the chassis lock finger down), slide it +X until the lock nub springs up behind the toe.
7. Plug the Ethernet patch (side-angle plugs, slack coiled in the channel at the jacks' -Y side).
8. Hang the upper bays (B3..B5) and B6 above the tower, then load them.
9. Hold the lid (both antennas fitted) just above the box, plug each pigtail into its HaLow HAT (the
   -X one to the Alta's), dress the cables into their corridors, then lower and screw down the lid.
Removal is the reverse: unplug both HaLow pigtails at the HATs before lifting the lid; unhook B4 and
B6 first (they sit over the tower) and lift the Ethernet patch's plugs out, press the tower lock nub
down and slide the tower -X, lift. To release a carrier, pull its latch lip 2 mm off the panel and lift.
Chassis: press the base detent nub through its hole (pen tip), push +Y, lift.

## Bolts kept

Only the four existing box-to-arm-layer M4 bolts (they carry the box). The base replaces the legacy
3.5 mm tray under their heads, so their grip length is unchanged from HEAD.

## Not established / to measure

- HiLetgo 30A relay outline and height (board-dims §9: about 75 x 42 x 26 ESTIMATE); PCA9685 clone
  outline (62.2 x 22.4 Adafruit fab print; clones may be 62.5 x 25.4) and pin-tail positions.
- Converter sizes (4 boxes): unknown; the bays state a capacity, not a fit.
- Alta hole pattern (Pi pattern ASSUMED), heatsink height (9 ESTIMATE), USB-C position (assumed in the
  same plug room as the Pi's); every stack spacing and PCB thickness; LC29H header gender and standoff.
- HaLow carrier owned (HT-HR01 HAT vs other) and its RF connector: an edge SMA + right-angle plug gets
  only 14 mm to the +Y wall; a U.FL pigtail is the safe end. The LC29H GNSS antenna needs a sky-view
  port that is NOT in this enclosure: owner decision.
- The purchased lid bulkheads, nuts and pigtail exits under the lid: modelled from estimates, and
  the control probes pass against the final lid's `antenna_lid_keepout()`; RJ45 side-angle plug
  sizes; 1.13 mm coax bend radius.
- M4 bolt head + washer envelope (9 x 4.8 ESTIMATE); breather and gland locknut envelopes; camera-tower
  M5 nut and bolt-tail protrusion inside the +Y wall.
- PETG snap strains are computed with the 1.5 % DESIGN limit (no datasheet): cleat latch 1.44 %, bay pawl
  1.44 %, tower boss flexure 1.49 %, signal clip 0.99 %, base detent 0.91 %, tower lock finger 0.83 %,
  tower clamp pawl 0.36 %. Holding force, creep and vibration retention are not qualified.
