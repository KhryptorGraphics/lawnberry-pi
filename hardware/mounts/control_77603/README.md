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
| `ina_holder.scad` | `ina_cradle()`: the INA3221 capacity cradle snapped onto the tower clamp frame (frame + print-in-place ratchet jaw bar) |
| `sma_clip.scad` | `sma_clip()`: the SMA service clip snapped onto the tower clamp's front corner, holding the three tower-side SMA jacks the lid's plugs screw onto |
| `board_carriers.scad` | `strap_bay()`, `harness_clamp()`, and the printable dispatch (`carrier_kind`, `carrier_part = "body"`) |
| `board_refs.scad` | Reference solids of the boards (published outlines, ESTIMATE heights) for scenes and probes |
| `assemblies.scad` | Scenes: `electronics_packaging`, `electronics_contents`, `electronics_exploded`, `electronics_lid_lifted` (the clip, its jacks and plugs, the tower-side leads and the lid cables included) |
| `control_checks.scad` | CSG probes (`-D 'check="name"' [-D idx=N]`) |

## Printed pieces (8 designs, 13 prints)

| Piece | Qty | Bodies | Print pose |
|---|---|---|---|
| `chassis_base` | 1 | 1 | flat, hooks up |
| `chassis` | 1 | 1 | plate on the bed, panels vertical |
| `tower_rack` | 1 | 1 | shelf on the bed |
| `tower_clamp` | 1 | 1 | upside down, frame top on the bed |
| `strap_bay` | 6 | 2 (PIP strap bar) | upright as installed |
| `harness_clamp` | 1 | 2 (PIP strap bar) | upright as installed |
| `ina_cradle` | 1 | 2 (PIP jaw bar) | upside down, jaw lips and bar top on the bed |
| `sma_clip` | 1 | 1 | upside down, plate top on the bed |

## Stack order (compute tower, bottom up)

Libre Computer Alta -> its HaLow HAT (Seeed WM1302 Pi HAT carrying the Wio-WM6108 card) -> [kit
standoffs] -> Raspberry Pi 5 + Active Cooler -> AI HAT+ on its supplied 16 mm stacking header ->
Waveshare LC29H(DA) on ONE purchased 2x20 extra-tall stacking header (it replaces the LC29H's
female-only header so GPIO passes up) -> the Pi's HaLow on top (Heltec HT-HC01P mini-PCIe card on
a Pi-size carrier board, 85 x 56 per the owner, SPI; carrier model TO CONFIRM). The SMA service clip
snaps onto the clamp's front
corner (§ RF service joints).
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
- INA3221 cradle: snapped on the tower clamp frame and riding with it; rails on the clamp's long
  bars, x -32..15.1 (one saddle tab on the -Y rail to x 26.2), y -8.8..52.1, z 117.7..132.7 at
  nominal clamp travel. Board flat, parts up, x -29.5 + 22..32 across, y 21.65 +- 15..19.3,
  PCB underside z 127.7; nothing above it but its own I2C lead and, with the lid on, the lid-borne
  Alta pigtail and its slack (z >= 168.5) (`ina_access_clear`).
- SMA service clip: on the tower clamp's front crossbar and finger bands, riding with it; x 15.6..42.6,
  y -12.5..55.6, z 116.3..158.3 nominal (leaf tab to plate top); its three jacks face up at
  (25, 3.2), (33.2, 27.6), (37, 44.5), mated plugs up to z 180.8 (183.8 at clamp travel +3).
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
  lid ribs, and under the lid the two antenna stations (30 mm round (+-64.5, -48), z >= 162), the
  lid post-socket bosses (x +-[31.5, 82.5], y -18..2, z >= 170) and the GNSS SMA jack (16 mm round
  (51, 41.6) from z 160): all AntennaMount's antenna_lid_keepout().

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

## RF service joints: the SMA clip (antennas and GNSS port on the lid)

Owner decisions 2026-11-02. Each radio has its own Rokland antenna on an N bulkhead near the lid's
back corners, (-64.5, -48) for the Alta's HaLow and (+64.5, -48) for the Pi's; the GNSS antenna
stays outside and screws onto an SMA-female bulkhead jack in the lid at **(51, 41.6)**
(../README.md § GNSS lid port). The lid separates from the boards at **three SMA joints on a
printed clip at the top of the tower** (`sma_clip.scad`): a short U.FL -> SMA-female-bulkhead lead
per radio ends in a jack held face-up in the clip and stays mated to its board; the lid's cables end
in SMA-male right-angle plugs screwed onto those jacks. To open the lid: lift it about 100 mm,
unscrew the three plugs with an 8 mm spanner, set the lid aside. The U.FLs are never unplugged in
service.

**Radios.**
- **Pi 5 (top of the stack): Heltec HT-HC01P mini-PCIe card (MM6108, SPI) on a Pi-size carrier**
  (owner; `config/limits.yaml` line 17: "Heltec HaLow HAT owns GPIO4/9/10/11 on this Pi"). Card
  50.95 x 30 x 3.75 with its IPEX (U.FL) 2.61 mm from the far end and 10.46 mm from the long edge
  ([drawing](https://heltec.org/wp-content/uploads/2025/05/HC01P_DEMENSION-976x1024.png),
  [datasheet](https://resource.heltec.cn/download/HT-HC01P/Datasheet/HT-HC01P_Rev.1.0.0.pdf) p.17);
  the owner states the carrier is the same size as the Pi 5, so it is modelled 85 x 56 on the Pi
  outline and holes (Heltec's own HT-HC01P Raspberry Pi HAT is 65 x 56 in its photos: carrier model
  and length TO MEASURE); card placement ESTIMATE: U.FL at about (10.0, 16.9, z 115).
- **Alta: Seeed Wio-WM6108 mini-PCIe card (Quectel FGH100M-H) on Seeed's WM1302 Pi HAT.** Antenna
  connector CN1 "CN_IPEX" = Hirose U.FL-R-SMT-1
  ([schematic](https://files.seeedstudio.com/wiki/wifi_halow/res/Wio-WM6108_V30_SCH_20241107.pdf)),
  at about (-37.2, 16.7, z 55.7) under the Pi 5 (from Seeed's CAD, inferred).
- **GNSS:** the LC29H(DA)'s IPEX socket (`lc_ufl_uv`, ESTIMATE about (11, 37.9, z 96.3)).

**Clip** (one print, 1 body, no fasteners; 27.0 x 68.1 x 42.0 in print pose, plate top on the bed).
It sits on the clamp frame's front crossbar (x 15.5..24) and on the two post-finger bands
(x 17.5..22) beyond the long bars: a north foot and column at x 15.6..24, y 38..55.6, a south
bridge on the -Y band at y -12.5..-6.8 with a web (x 29.6..32.6) up to the plate. Two jaw pairs,
4.5 mm deep, straddle the bands 0.2 mm off their sides and off the frame's +Y / -Y faces; a snap
leaf (2.0 thick, 21.4 long, strain 0.92 %) hangs down the crossbar's +X face (x 24.4..26.4,
y 39..45) and hooks a 1.0 mm lip under the crossbar. The 5 mm plate (x 15.6..42.6, y -2..50,
z Zc+28.6..33.6; Zc = clamp-frame top, 124.7 nominal, travel -12..+3) carries the three jacks,
each in the lid port's D hole (Ø6.5, flat 6.0) on a 3.0 mm land over a Ø14 counterbore for the
body hex (published SMA-F D-flat bulkhead data: Taoglas CAB.011, Linx CSH-SGFB, MIL-STD-348A;
maximum panel 3.175 mm).

| Joint | Jack axis (x, y) | Plug barrel | Spanner swing | Tower-side lead | Lid-side cable |
|---|---|---|---|---|---|
| JA Pi HaLow | (25, 3.2) | +Y | toward 180° | HC01P U.FL | from the +X N bulkhead |
| JB GNSS | (33.2, 27.6) | +X | toward 200° | LC29H U.FL | jumper from the lid GNSS jack |
| JC Alta HaLow | (37, 44.5) | -X | toward 210° | WM6108 U.FL | from the -X N bulkhead |

Heights above Zc: land top 33.6, jack (mating) face 42.0, plug coupling nut 38.57..46.97 (5/16 in
hex), plug body to 56.07 (183.8 at travel +3, 0.7 under the lid), cable axis 51.67. Right-angle
plug envelope: L-com LCCN45958 / Wellshow C01BA31174008A drawings (17.5 nut end to far face,
13.1 to the cable axis, body 7.92, barrel 10.8). The SMA plug's mating torque (7-10 in-lb, Amphenol
RF) goes from the jack's D-flat into the clip and from the clip's jaws into the clamp frame; the
jaws' play limits the clip's turn to under 1° (`sc_clip_keyed`).

Spanner room per joint (lid lifted 100 mm or off): an 8 mm open-end spanner (heads 19-20.5 wide,
3.8-5.5 thick, IS 2028 / DIN 3110) swings ±15° toward the joint's swing direction in the slab
Zc+40.5..46 on the plug nut, within 56 mm of the axis, and comes down from above over that sector
beyond 12 mm. 56 mm is all the box gives (bays B4 / B6, the INA board): published 8 mm spanners
are 90-141 long, so hold one short of its end or use a stubby spanner (TO CONFIRM by trial).

Kept clear (agreed with the INA3221 cradle, 2026-11-02): the cradle's cross member and inner tab
(x <= 15.1), its lift path over rail A (x 14.7..26.6, y -6.75..-2.45 above Zc+8.4: the plate starts
at y -2.0 and JA's washer at y -1.9), its driver column and finger rooms, and its I2C lead's
envelope (`ina_i2c_run`: up x -12..-6, then +X along y 8..16 at Zc+35..40 over the clip plate
between JA and JB, then down x 46..59, y -10..25). The printed clip stays below Zc+33.6; only the
purchased jacks and plugs rise above Zc+40, as agreed.

**Tower-side leads** (U.FL (IPEX MHF I) plug -> SMA-female D-flat bulkhead, 1.13 mm; tower-borne,
mated once at stack build; length = closed route + 20 mm dress slack):
- **Pi:** +X under the clamp crossbar and the INA cradle (z 114..117.4), up outboard of the
  crossbar (x 24.5..29.2) to JA. Route **36 mm** (R10); Taoglas CAB.011 (95 mm per its drawing) fits.
- **GNSS:** +X under the Pi's HaLow carrier (z 95.2..105; the carrier is Pi-size, so it reaches
  x 39), up past its +X edge (x 39.6..45.6), then -X over it (z >= 117.2, above its 9 mm parts
  envelope) to JB. Route **61 mm** (R10), needs >= 81 mm. **The Waveshare kit's "IPEX 1 to SMA cable
  ~17cm" can be used** for this lead: its ~109 mm spare coils vertically (about 1.7 turns of Ø20,
  in the y-z plane) in the column x 39.6..45.6, y 21.5..43.5, z 95.2..120 beside the carrier's +X
  edge. Conditions (TO CONFIRM on the kit cable): its SMA-female bulkhead
  has the D-flat (the clip's D hole keys it), a thread long enough for the 3.0 mm land + washer +
  nut with 5.54 mm left for the plug's nut (11.4 mm hex face to mating face per the drawings), and
  an 8 mm hex/nut. If it has no D-flat or less thread, replace it with the same U.FL -> SMA-F
  D-flat bulkhead 1.13 mm lead as the Pi's (any length >= 81 mm; spare coils in the same column).
- **Alta:** +Y under the Pi from the card's U.FL, +X behind the stack, round the tower's +Y post,
  up the +Y gap (x 26.5..33) between the camera nuts' z bands, -Y to JC. Route **181 mm** (R8: its 6 mm corridors
  take no larger radius), needs >= 201 mm: **buy 250 mm**.

**Lid-side cables** (lid-borne; length = shortest closed route + 100 mm lid-lift slack, the spare
stowed closed in each cable's slack zone; every corner filleted at R7: each HaLow pigtail leaves its
bulkhead straight down and must be level above the bays, which allows R7.2 at most, and the GNSS
jumper's U-turn is R7 too; all above the 6.8 mm published minimum for 1.13 mm cable,
[Taoglas](https://www.taoglas.com/specifications/30021F020000A_Taoglas.pdf)):
- **+X station -> JA (Pi): N-female bulkhead -> SMA-male RIGHT-ANGLE plug, 1.13 mm.** Out of the
  station +Y over B5, -X along the bays' front (y -16.4..-10.4, under the +X post-socket boss and
  -Y of the INA lead's descent), up beside the boss (x 39..45.6, y 2.4..7.6), +Y over the INA lead
  at z >= 168.5, -X between JA and JB and -Y into JA's barrel. Route **105 mm**, needs >= 205 mm:
  **buy 250 mm**; its ~145 mm spare loops over the INA descent (x 45.6..72, y 2.4..25.2,
  z 168.5..180).
- **-X station -> JC (Alta): N-female bulkhead -> SMA-male RIGHT-ANGLE plug, 1.13 mm.** Out of
  the station +Y over B3, +X along the bays' front under the -X boss, up beside B4's front, +Y
  over the tower top (x -24..-16, z >= 168.5: above the INA lead's Zc+40 at every travel) and +X
  along y 41..47 into JC's barrel. Route **176 mm**, needs >= 277 mm: **buy 300 mm**; its ~124 mm
  spare loops over the tower top (x -31..10, y -9..40.6, z 168.5..180).
- **Lid GNSS jack -> JB: SMA-female D-flat bulkhead (the lid jack) -> SMA-male RIGHT-ANGLE plug,
  1.13 mm.** The jumper leaves the bulkhead straight down (heat-shrink end z 161.1), under JB's
  barrel height, so it U-turns (R7) in the loft right of the clip (x 49.5..52.5, y 26.1..44.6,
  floor z 98) and rises into JB's barrel from +X. Shortest route **41 mm**, needs >= 141 mm:
  **buy 150 mm**; the spare hangs in the U (floor reached at 150 mm, asserted).

Purchase list (RF):

| Cable | Connectors | Cable | Route (closed) | Rule | Buy |
|---|---|---|---|---|---|
| Pi tower lead | U.FL (IPEX MHF I) plug -> SMA-F D-flat bulkhead | 1.13 mm | 36 mm | + 20 | Taoglas CAB.011, 95 mm |
| GNSS tower lead | U.FL plug -> SMA-F bulkhead | 1.13 mm | 61 mm | + 20 | kit cable ~170 mm (conditions above) |
| Alta tower lead | U.FL plug -> SMA-F D-flat bulkhead | 1.13 mm | 181 mm | + 20 | 250 mm |
| Pi lid pigtail | N-F bulkhead -> SMA-M right-angle plug | 1.13 mm | 105 mm | + 100 | 250 mm |
| Alta lid pigtail | N-F bulkhead -> SMA-M right-angle plug | 1.13 mm | 176 mm | + 100 | 300 mm |
| GNSS lid jumper | SMA-F D-flat bulkhead (panel >= 3.0 mm) -> SMA-M right-angle plug | 1.13 mm | 41 mm | + 100 | 150 mm |

Sources: SMA bulkhead data Taoglas CAB.011 p.7 / Linx CSH-SGFB / MIL-STD-348A; right-angle plug
L-com LCCN45958 / Wellshow C01BA31174008A; N bulkhead Amphenol Type N catalog (82-369 panel
5.3 mm); routes from `tl_way` / `ll_way` (`parameters.scad`, asserted against these lengths).

Each route is a centreline whose corners are filleted at its bend radius (tower leads: Pi and GNSS
R10, Alta R8; lid cables R7; all above the published 6.8 mm). `parameters.scad` asserts that every
fillet fits its legs, that the 1.13 mm cable's surface stays inside its corridors (the HaLow
pigtails' corridors run back into their station envelope, where each leaves its bulkhead) and that
each cable bought covers its filleted route plus its slack. Lengths above include the fillets.

## Strap bay capacity (one print, x6)

Box width along the panel 18..43.6, depth 16.5..26.5 (from the rib tips), height >= 20 up to each
slot's capacity. Ratchet 1.5 mm steps; the spring leaf takes up the step with 1.0..2.5 mm preload.
Fit: stand the box on the ledge against the ribs, push the strap bar on. Release: squeeze both pawl
finger tops toward the middle and pull the bar off. Harness clamp: bundle 8..26 mm, runs vertically.

## INA3221 cradle (power monitor)

The owner's INA3221 module is not known, so `ina_cradle` is a capacity holder: any board 22.0..32.0
across x 30..38.6 along, PCB 1.0..1.6 thick (DESIGN ranges). Common outlines inside it: Adafruit
#6062 38.354 x 22.86 (its EAGLE board file), purple/black CJMCU-3221 38.3 x 31.7 (vendor), the
yesyes INA3221-R010 30 x 26; the SwitchDoc Labs board is TO MEASURE (sources in `parameters.scad`).
The board lies flat, parts up, its two LONG edges gripped by plastic ledges and lips: the fixed jaw
along the whole near (-X) edge, 1.2 mm reach, and the bar's shoe over the middle 8 mm of the far
(+X) edge, 0.8 mm reach. The jaw bar ratchets in 1.5 mm steps (strap-bay tooth form) and a
cantilever leaf behind the shoe takes up the step with 1.0..2.5 mm preload; along the edges the
board is held by that preload only (the pawl fingers beside its short ends leave a 38.6 board
6 mm of end play, a shorter one more).

**Part clearance (every gripping feature keeps >= 0.4 mm from the board's parts).** Parts on either
face must stay >= 1.6 mm from the jaw edge and >= 1.2 mm from the shoe edge; put the edge whose
parts come closest under the shoe. **Adafruit #6062, REQUIRED orientation: terminal-block edge on
the fixed jaw (-X), header edge under the shoe (+X).** Its I2C goes through the 0.1 in header,
soldered, with a female Dupont housing plugged on it: its STEMMA QT sockets face the cradle's pawl
fingers and rails 3..5 mm away and cannot take a plug. Clearances at the seated board (shoe
pressed on the edge), from its published positions: jaw lip/ledge to the terminal blocks 0.915
(2.115 - 1.2), shoe lip to the header body and Dupont housing 0.47 (1.27 - 0.8), shoe ledge to the
header pads 0.89 (1.69 - 0.8, pads ESTIMATE), pawl fingers to the STEMMA QT sockets 3.57.
`ina_adafruit_clear` proves >= 0.4 against that board with every part inflated by 0.45; bracketing
the inflation gives 0.95..0.99 at the jaw and 0.50..0.54 at the shoe at the probe's 0.05 offsets,
i.e. 0.90..0.94 and 0.45..0.49 when seated. In the other orientation the 1.2 mm jaw lip would
reach to 0.07 mm of the header: do not fit it that way. Other boards: TO MEASURE against the
1.6 / 1.2 mm rule.
The board sits under the lid: with the lid off its terminals and plugs are reached straight from
above. It is on the Pi's I2C bus at 0x40 (`config/hardware.yaml`); the Pi's PCA9685 is configured
for 0x41 (`config/tractor.yaml`): check its A0 strap with `i2cdetect` before wiring both.

**Leads (routes and lengths ESTIMATE, not modelled or probed).**
- Sense leads (solar ch1 and battery ch3 pairs from the external shunts, `config/hardware.yaml`):
  through the -X floor gland (-40, -49), up the free slot between bay B1 and the harness clamp
  (x -43..-30, along the -Y panel) to z ~78 above the USB-C plug room, across the pocket in front
  of the Pololu (x -75..-45, y -38..-15) to the box's -X/-Y corner, up the -X alley (x -80..-73,
  y -12.5..-1: outside the tower's slide-and-lift path, which starts at x -72) to z ~Zc + 9 (Zc = clamp-frame top, 124.7 nominal), then +X over
  the tower's -X end (y -5..-1, just -Y of bay B6's post) onto the cradle, along the top of the
  fixed jaw and into the terminal blocks from the -X side, over the jaw top (3.35 mm above the PCB
  top; terminal entry window TO MEASURE). Seated, about 245 mm from the gland's inner face to the
  farthest terminal; leave 120 mm more as service slack, coiled flat in the pocket in front of the
  Pololu: cut each lead at least 370 mm long inside the box.
- I2C lead (VIN, GND, SCL, SDA): a female-female Dupont lead from the board's header, up the
  column x -12..-6, y 2..41 kept free above the Dupont housing to Zc + 40 (no coax there at any
  clamp travel), +X over the tower top along y 8..16 at z Zc + 35..40 (over the SMA clip's plate
  between its Pi and GNSS jacks, under their spanner rooms) to x 46, down into the free loft right of
  the tower (x 46..59, y -10..25: -Y of the GNSS jumper's U) to the Pi PCA9685's free 6-pin chain header (+X panel, y 28.4..50.8; the end
  not wired to the Pi, TO CONFIRM). Seated about 120..225 mm ESTIMATE (which of the PCA9685's two
  header rows is free, and where the I2C pins sit on the board's header row: TO CONFIRM); leave
  150 mm more as slack, coiled in the loft: buy 400 mm leads.
- With that slack nothing is unplugged for tower service: after B4 and B6 are unhooked, unclip the
  cradle and lift it, board and leads attached, over the -Y wall onto the outside of the box (about
  +90 mm on the sense leads, which climb the -X alley and pass over B3; about +110 mm on the I2C
  lead, which climbs the loft and passes over B5). Both leads then run outside the tower's lift
  path, and the tower slides and lifts out.
- If the board itself must come off its leads: power down, disconnect the sense leads at the
  SHUNT end first (so no live conductor hangs loose at the cradle), then unscrew them at the board;
  then unplug the I2C lead at the PCA9685 end, then at the board. Reconnect in the reverse order
  (I2C first, sense leads last).

## Assembly order

1. Clamp `chassis_base` on the four boss tops under the existing M4 bolt heads + washers.
2. Lower the chassis 6 mm +Y of its seat (the hooks rise through its windows) and push it -Y until
   the base detent nub pops into the plate hole at (62, -9).
3. Clip the two PCA9685 and the Pololu into their pockets (bottom corners first, top clips snap over).
4. Hang the tier-1 bays (B1, B2) and the harness clamp: drop each hook into its strip's V, press the
   plate flat until the latch tooth clicks under the catch land. Load the relays; clamp the bundle.
5. Build the board stack on the kit standoffs, mating each tower-side RF lead's U.FL as its board
   goes on (they are never unplugged again in service): the Alta lead onto the WM6108 card before
   the Pi 5 goes on, the GNSS lead onto the LC29H before the Pi's HaLow carrier goes on, the Pi lead
   onto the HC01P card last. Drop the stack onto the rack spigots; push the clamp on.
6. SMA clip: push each lead's SMA jack up through its clip hole from below (Pi -> JA, GNSS -> JB,
   Alta -> JC; the D-flat keys it, the body hex sits in the counterbore), fit the lock washer and
   nut on the plate top and tighten with an 8 mm spanner (bulkhead nut 15 in-lb, Cinch). Push the
   clip straight down on the clamp's front corner: the jaws drop over both finger bands, the foot
   lands on the crossbar and the leaf clicks under it. Coil the GNSS lead's spare in its zone
   beside the clip.
7. Lower the tower 6 mm -X of its seat (keyhole entries over the two mushroom studs, the toe presses
   the chassis lock finger down), slide it +X until the lock nub springs up behind the toe.
8. Plug the Ethernet patch (side-angle plugs, slack coiled in the channel at the jacks' -Y side).
9. INA3221: wire it on the bench (Adafruit #6062: header soldered, terminal-block edge to the jaw),
   lay it in the cradle (near long edge under the jaw lip), hold it level and push the jaw bar on
   until the shoe grips the far edge (ratchet clicks). Push the cradle straight down onto the tower
   clamp: the saddle drops over the front (+X) crossbar and both snap leaves click under the long
   bars. Plug the I2C lead on the header and lay it over the clip plate between JA and JB (see
   above); lay both leads' slack in their pockets.
10. Hang the upper bays (B3..B5) and B6 above the tower, then load them.
11. Hold the lid (both antennas' bulkheads, the posts and the GNSS jack fitted) about 100 mm above
   the box. Screw the three lid-cable plugs onto the clip by hand, then 8 mm spanner: the +X
   pigtail onto JA (barrel +Y), the GNSS jumper onto JB (barrel +X), the -X pigtail onto JC
   (barrel -X); dress the cables into their corridors and slack zones (the GNSS jumper's spare
   hangs in its U in the loft), then lower and screw down the lid.
Lid opening: lift the lid about 100 mm, unscrew the three SMA plugs at the clip with an 8 mm
spanner (each jack is keyed in the clip; nothing else is unplugged), set the lid aside.
Removal is the reverse: open the lid as above; unhook B4 and B6 first (they sit over the tower),
lift the INA3221 cradle off (pull both snap leaves outward, lift its -X end, lift it off the
crossbar; it stays on its leads, which lift off the clip plate with it) and lift the Ethernet
patch's plugs out, press the tower lock nub down and slide the tower -X, lift: the clip and the
three tower-side leads come out with the tower. To take the clip off, pull its leaf's release tab
+X about 1.5 mm and lift it.
To free the board, push both jaw pawl tips toward it (pen tip) and pull the bar back. To release a
carrier, pull its latch lip 2 mm off the panel and lift.
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
- HaLow boards (owner 2026-11-02: Heltec HT-HC01P card on the Pi 5, Seeed Wio-WM6108 card on the
  Alta). Heltec: the Pi carrier is Pi-size per the owner (modelled 85 x 56; Heltec's own HAT is
  65 x 56 in its photos): model and length TO MEASURE; card placement, card height in its socket
  and the carrier's parts ESTIMATE; the HC01P card and its U.FL position are published. Seeed: card U.FL position from Seeed's CAD (inferred), card height in its socket
  ESTIMATE. Both U.FLs are mated at stack build and stay mated (the SMA clip is the service joint).
- RISK, Pi 5 radio electrical fit (owner to confirm; pin assignment is an electrical/software
  decision outside the CAD): the HT-HC01P carrier runs SPI on GPIO4/9/10/11
  (`config/limits.yaml` line 17, `morse_spi` in `docs/autonomy-toolchain-matrix.md`); its chip-select
  pin is TO CONFIRM against the repo's pin plan. The HC-SR04 ultrasonics are already disabled for it
  (`ultrasonic_enabled: false` on that line).
- SMA clip and RF cables: the purchased jacks against the published drawings (D-flat, thread from
  the hex face to the mating face, panel range covering the clip's and the lid's 3.0 mm lands, hex,
  heat-shrink length); the Waveshare kit cable's bulkhead (D-flat and thread: it is the GNSS tower
  lead only if both hold); the right-angle plugs' envelope (L-com / Wellshow drawings); the lid
  bulkheads, nuts and pigtail exits under the lid (estimates; the control probes pass against the
  final lid's `antenna_lid_keepout()`); the 8 mm spanner's swing in the 56 mm the box gives (by
  trial); the clip's snap and jaws' hold under the plugs' mating torque (unqualified); RJ45
  side-angle plug sizes; LC29H U.FL position ESTIMATE (`lc_ufl_uv`).
- M4 bolt head + washer envelope (9 x 4.8 ESTIMATE); breather and gland locknut envelopes; camera-tower
  M5 nut and bolt-tail protrusion inside the +Y wall.
- INA3221 module owned: its outline (the cradle takes 22..32 x 30..38.6), PCB thickness (1.0..1.6
  accepted), parts on either face within 1.6 mm of the jaw edge or 1.2 mm of the shoe edge (lip
  reach + 0.4), parts height (10.5 ESTIMATE) and pin tails (2.0 ESTIMATE); on the Adafruit the
  plugged Dupont housing size (2.54 wide x 14 ESTIMATE), terminal-block wire-entry height and the
  header-pad diameter (1.7 ESTIMATE); the sense and I2C lead routes and lengths, and which end of
  the Pi PCA9685's chain headers is free.
- PETG snap strains are computed with the 1.5 % DESIGN limit (no datasheet): cleat latch 1.44 %, bay pawl
  1.44 %, tower boss flexure 1.49 %, signal clip 0.99 %, base detent 0.91 %, tower lock finger 0.83 %,
  tower clamp pawl 0.36 %, INA3221 jaw pawl 1.40 %, INA3221 shoe leaf 1.39 %, INA3221 cradle snap
  1.08 %, SMA clip leaf 0.92 %. Holding force, creep and vibration retention are not qualified.
