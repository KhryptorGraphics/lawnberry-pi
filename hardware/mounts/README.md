# Toro conversion — printable mounts and fit fixtures

Parametric OpenSCAD sources, binary STL exports, PNG previews and reproducible geometry checks.
Dimensions are millimetres. **These files are not a machine-fit, structural or ingress certification.**
The mower's engine and hydrostatic transaxles stay stock; the servos move the existing controls.
There are no replacement drive-motor mounts in this package.

**Where things go on the machine, with rendered views: [`INSTALL.md`](INSTALL.md).**

## Start here

1. The target is the **Toro TimeCutter MAX 50 in MyRIDE, model 77502**. Confirm the serial plate and
   measure the hitch plate's hole, width, depth and thickness. Plate dimensions remain unverified.
2. Print `hitch_gauge`, `arm_layer_gauge`, `fit_gauge_2`, all three `lapbar_clamp_gauge_*`
   coupons, and all three `pushrod_gauge_*` coupons. Measure the steering tubes and servo-holder revision.
3. The rods are generated from an **estimate** (`TORO_77502_LINKAGE.md`): 936.9 mm pin-to-pin.
   Measure the lap-bar clamp point, crank radius and travel; set `lapbar_pin_*`/`servo_crank_*`
   in `enclosure_common.scad` and rebuild before printing rods.
4. The printed center layer, arms, clamps and rods are **fit/load-test prototypes**, not qualified
   steering parts. A single hitch bolt is not an approved anti-rotation restraint. Do not power the
   steering until a positive chassis torque path, full hand-travel clearance, clamp retention and
   unpowered neutral return are physically verified.
5. The user-stated printer is 420 × 420 × 500 mm. All separate part variants are built against it;
   inspect real usable bed volume, orientation, material and slicer preview before printing.

The enclosure body prints bed-down at 211 × 285 × 185 mm including its tongue;
the separate sandwich-layer center panel is 295 × 285 × 8 mm. The longest
arm and rod members fit a 420 mm bed at the selected print orientations,
but mower clearance and load capacity remain unverified.

## Dimensional evidence

| Interface | Basis and current treatment |
|---|---|
| RDS51150 stationary-holder **side** | Confirmed 2026-10-07: the holder's **65 × 30 mm** mounting flange — the case face square to the output shaft — carries four M2.5 holes on **24 × 24 mm** centres, and only those four fasten; print bores Ø2.9. **The pattern is not concentric with the output**: on the dimensioned drawing the disc sits **~19 mm off the pattern's centre** along the case length (the holder view suggests more), so `servo_shaft_offset_l` — 0 in CAD — must be measured with the part in hand and its sign set to the servo's orientation before any rod is printed. Replaces the unsupported 50 × 20 M5 pattern. Drawing says RDS51150; verify the purchased RDS51150SG revision with the coupon. |
| Hitch sandwich + arm layer | The enclosure flange is **211 × 170 mm**; an 8 mm center layer continues to the modeled hitch hole and has 42 mm side ears (STL envelope **295 × 285 × 8 mm**). Hole Ø20.8 and auxiliary M6 centres ±40 mm mirror the unverified enclosure tongue. The **160 × 120 × 8 mm** mower-plate envelope is schematic; measure it. |
| Splayed servo arm | **MEASURED** station: servo face **209.55 mm** left/right of the ball-hole centre, **457.2 mm** from its plate-top datum, splayed **12.58°** from vertical. The axis extrapolates to x=±110, while the solid 60 × 70 × 50 mm root seats on the center layer **top** at z=0 and y=123. Four M6 per root enter upward into side-access steel nuts; the hollow 50 × 70 mm column has a through-channel and 6 mm walls/web. Lower arm joint at local t=355 mm; printed head retains a four-M6 flanged/spigoted lap and lateral servo face. Neither fit nor strength on the real mower is established. |
| Arm sway bars | Three handed first-article braces per side land at z=110/190/250 mm. Each left/right half-jaw occupies its own side of the tower; the pair forms a rear-open U with a 0.4 mm centre gap. Fit from **+Y toward the box**, never from -Y or over a flange. Two top-loaded M4 nuts per half retain short screws bearing **outside** the USB-cable tube. Two M5 bores meet each arm pad's side-entry nuts, with 0.2 mm face clearance. These are **not steering-load parts**; test slip, creep, nut pullout and pad damage unpowered. |
| Lap-bar connector | Fully printed two-piece split clamp, four M5 bolts per side. Paired clamp yoke ears capture one 32 × 20 × 8 mm rod eye in double shear with an M6 through-pin. Default tube OD **25.4 mm is coupon-only**; clamp preload, coating, slip and print strength need physical testing. |
| Pushrods | Toro 77502 **estimate**: 936.9 mm pin-to-pin, 934 mm fore-aft span, lap-bar pin 74.2 mm outboard of the crank (4.54° built skew — it fell from 103 mm/6.3° when the crank moved out to the drawing's 61.4 mm datum). 40 mm square tube, 5 mm walls, in three bolted pieces plus a 29.1 mm solid inner bar; three lock settings at 15 mm (±15 mm trim). Both ends are 32 × 20 × 8 mm eyes on Ø6.6 bores. The servo crank points up at neutral; its inner face lands on the kit disc at the drawing's **61.4 mm** overall axial envelope, and `rod_servo_sweep` proves the eye and rod transition clear the case and disc over ±35°. See [`TORO_77502_LINKAGE.md`](TORO_77502_LINKAGE.md). |
| E-stop rail U-bolts | Selected BOM 40 mm inside + 8 mm leg = 48 mm centre pitch, printed Ø9. Used only by the E-stop fixture; the 76.2 × 38.1 mm rail remains unverified. |
| E-stop HB2-ES545 | Listing drawing: Ø22 panel opening, Ø40 mushroom, 42 mm rear projection, 30 × 29 mm contact envelope. Print bore Ø22.5; 3 mm panel is a design choice, not a vendor maximum thickness. No guessed anti-rotation notch. |
| Pi 5 | Official reference drawing: **85 × 56 mm PCB**, **58 × 49 mm** mounting pitch, 3.5 mm edge offsets, Ø2.7 PCB holes. Tray uses M2.5 with Ø2.9 through-bores. Pattern centre is offset 10 mm from board centre. |
| Camera Module v2.1 | Official drawing: **23.862 × 25 mm**, Ø2.2 PCB holes at (2,2), (14.5,2), (2,23), (14.5,23). Carrier uses M2/Ø2.4, **12.5 × 21 mm** pitch, open ribbon-cable edge. Not an ELP stereo/Camera v3 mount. |
| Opto relays | The linked product images show **one relay/channel per PCB**. Two 2-packs provide four boards, of which starter/PTO/watchdog use three. Four separate stack levels are retained; no channel is deleted. Board outline/hole pitch remains unverified. |
| Other electronics/sensors | Carrier SKU, outline, terminals and mounting patterns are not established for the bought PCA9685, buck, watchdog perfboard, ZED-F9P, BNO085, VL53L0X/BME280/INA3221 carriers or antenna. Sleds and sensor blanks are explicitly drill-to-fit. |
| Glands, vent, nuts | Ø15.5 PG9 and Ø12.5 M12 cutouts are provisional clearances. Verify actual thread diameter, shoulder, panel-thickness range and locknut access. M4 nut pockets are accessible hex pockets; no unknown heat-set insert geometry. |

Vendor drawings lack manufacturing tolerances. The 0.2–0.5 mm diametral allowances are FDM starting
points, not worst-case tolerance proofs. Print coupons and ream/adjust for the chosen machine/material.

### Sources

- [Toro 2026 zero-turn catalogue](https://cdn2.toro.com/en/-/media/Files/Toro/Homeowner/zero-turn-mowers/2026/Toro-2026-Zero-Turn-Mowers-Product-Brochure.ashx):
  TimeCutter C-channel versus MAX tubular families.
- [Toro model 77502 page](https://www.toro.com/en/product/77502),
  [operator's manual 3483-555](https://www.toro.com/getpub/281676) and
  [setup instructions 3464-688](https://www.toro.com/getpub/246581): TimeCutter MAX 50 in MyRIDE
  dimensions, lever/PARK behaviour and hitch bracket. The rod estimate is in `TORO_77502_LINKAGE.md`.
- [ANNIMOS listing, B0C69W2QP7](https://www.amazon.com/dp/B0C69W2QP7/),
  [dimensioned RDS51150 image](https://m.media-amazon.com/images/I/61mhu1gYWZL._AC_SL1500_.jpg),
  [electrical ratings image](https://m.media-amazon.com/images/I/61pgKyNsMYL._AC_SL1001_.jpg).
- [Prusament ASA technical data sheet](https://storage.googleapis.com/prusa3d-content-prod-14e8-wordpress-prusament-prod/2024/05/bc8551ef-tds_prusament-asa_2024_en.pdf): in-plane/interlayer properties are specimen results, not printed-mower allowables.
- [HB2-ES545 listing, B07R9QTBG7](https://www.amazon.com/dp/B07R9QTBG7/),
  [dimensioned switch image](https://m.media-amazon.com/images/I/71UK8ymjIWL._SL1500_.jpg).
- [Pi 5 mechanical reference drawing, RP-008347-DS-1](https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008347-DS-1-raspberry-pi-5-mechanical-drawing.pdf).
- [Camera Module 2 mechanical drawing, RPI-CAM-V2_1](https://datasheets.raspberrypi.com/camera/camera-module-2-mechanical-drawing.pdf).
- [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/): stacking hardware is not a total-height guarantee.
- [HiLetgo relay listing, B0CHFJSNP6](https://www.amazon.com/dp/B0CHFJSNP6/),
  [single-channel PCB photograph](https://m.media-amazon.com/images/I/61rEA2LX5gL._SL1050_.jpg).
- Selected [square U-bolt](https://www.amazon.com/dp/B0DT98CFW8/) listing for the E-stop mount only;
  confirm supplied leg length and dimensions.
- CAD workflow reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
  arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065.

## Printable inventory

All names below have a matching `prev_<name>.png`. Quantities are installed quantities, not separate
CAD files. Never print an assembly view as one fused part.

| STL | Qty | Use / hardware |
|---|---:|---|
| `arm_layer_center` | 1 | Sandwich panel matching the box flange and tongue, with hitch/auxiliary holes, four M4 tray holes, three provisional gland passages, camera-tower foot holes and side-arm bolt ears. Measure the actual hitch plate; current dimensions are schematic. |
| `arm_layer_left`, `arm_layer_right` | 1 each | 50 × 70 mm hollow columns with three inboard pads, a 60 × 70 mm solid root with four side-loaded M6 steel nuts, and a flanged/spigoted head joint at local t=355 mm. Print lying flat; inspect nut access. |
| `arm_head_left`, `arm_head_right` | 1 each | Heads carrying the **lateral** RDS51150 holder face (4-M2.5 on 24 mm) on a rib; four-M6 flanged joint. Print mating-face down. |
| `arm_brace_bar_left_0`–`_2`, `arm_brace_bar_right_0`–`_2` | 1 each | Six handed half-jaw braces at z=110/190/250. Print the labelled left/right variants; do not substitute two identical copies. Fit each pair from +Y after tower assembly. Each half has two top-loaded M4 nuts, outside-bearing screws and two M5 bores for its arm pad. Prints flat; sway only, not a qualified steering load path. |
| `arm_layer_gauge` | 1 first | Coupon for the enclosure tongue's ball-hole/auxiliary-hole pattern. It is not a Toro hitch gauge. |
| `lapbar_clamp_anchor`, `lapbar_clamp_cap` | 1 each per side | Fully printed two-piece clamp with paired yoke ears; four M5 through-bolts per side. The clamp pin passes through both ears and the single rod eye in double shear. Default Ø25.4 tube is sample-only. |
| `lapbar_clamp_gauge_0`–`_2` | 1 each | Three diametral fit coupons; select from the measured tube OD and fit the actual print/material first. |
| `pushrod_servo_end`, `pushrod_middle`, `pushrod_sleeve`, `pushrod_inner` | 1 each per side (2 sets) | One 936.9 mm rod per side; the same set serves both sides, turned 180° about its own axis on the left. Two M6 bolts per spigot splice, two M6 lock bolts, one M6 crank pin through the metal servo arm and one M6 pin through the clamp yoke. The servo end has a 5 mm vent; keep it clear. Generated from the estimate: measure first. |
| `pushrod_gauge_0`–`_2` | 1 each | Sliding-fit coupons (tight/nominal/loose) for the 29.1 mm inner bar in the 40 mm sleeve. |
| `throttle_servo_mount` | 0 or 1 | Optional stationary-side fixture; four M6 slots at nominal 76 × 40 pitch with ±4 mm X travel. Installer-selected support, not a Toro panel pattern. |
| `estop_bracket` | 1 | Ø22 mushroom panel with external gussets and two square U-bolts. |
| `estop_contact_cover` | 1 | Removable rear guard; four M3 ×12 screws/nuts/washers; downward wire opening and tie slots. Not a sealed switch housing. |
| `enclosure_body`, `enclosure_lid`, `electronics_tray` | 1 each | Tall serviceable electronics box with an integral rear hitch tongue; common dimensions in `enclosure_common.scad`. |
| `hitch_gauge` | 1 first | Small coupon for the enclosure tongue's hitch hole. The 20 mm hole in that tongue remains unverified; measure before printing the body. |
| `camera_tower_base`, `camera_tower_segment` × N, `camera_tower_cap`, `camera_tower_backing` | see "Camera tower" | Hollow USB-cable column from the tongue to camera height; N and length derive from `tower_height`. Base prints flange-down, segments lying flat. |
| `relay_board_sled` | 4 | One single-channel PCB per deck; four M3 stack columns. |
| `utility_board_sled` | 2 | PCA9685/buck/watchdog and small internal carriers; four columns outside Pi footprint. |
| `camera_carrier`, `camera_foot` | 1 each | Camera2 tilt mount; four M2 screws, one M4 ×30 pivot with washers/locking nut, two M4 foot fixings. |
| `sensor_carrier` | As needed | Rigid drill-to-fit backplate; use another `camera_foot` for tilt. |
| `antenna_deck` | 1 if needed | 120 mm deck with edge straps/M4 mounting; optional metal ground plane. |
| `equipment_saddle` | As needed | Two-strap platform for housed automotive relays/inline fuse holders. |
| `harness_saddle` | As needed | Raised tie tunnel and two M4 fixing holes for cable strain relief. |
| `fit_gauge_1`, `fit_gauge_2` | 1 each initially | E-stop rail/U-bolt cross-section and RDS51150 stationary-side M2.5 pattern. |

The generic sensor carrier is a finished drilling blank. Enter measured `sensor_pcb_holes` and the
correct `sensor_pcb_bore` in `sensor_carrier.scad` to regenerate an exact variant, or transfer the actual
PCB pattern onto the blank. Use insulating spacers; do not strap bare boards/components. Locate IMUs
rigidly on the same chassis reference used for calibration, away from servo magnets and high-current
wiring. ToF apertures must remain unobstructed; an arbitrary transparent cover can cause crosstalk.
An exposed Camera2, sensor PCB or antenna connector still needs appropriate environmental protection.

## Enclosure and stack assembly

**Interior: 169 × 128 × 180.** Body flange 211 × 170; with the hitch tongue the print is
**211 × 285 × 184.5** (tongue extends 132 mm beyond the +Y wall, in the flange's own width).
Lid: 219 ×178 ×12 in print orientation. Closed height: 189.5.
The extra height is deliberate: it retains four relay channels and the other boards instead of claiming
that all of them fit a small two-board bay.

Assembly Z is measured from the enclosure underside:

| Item | Z / reserved capacity |
|---|---|
| Floor top | 4.5 |
| Tray underside / top | 14.5 / 18; one 10 mm support lift, no duplicate feet |
| Pi PCB underside | 26; long board axis along Y; published hole offsets retained |
| Pi + cooler + Hailo reservation | Up to Z80; actual headers/cables must be checked |
| Utility sled undersides | Z90 and Z130 |
| Relay sled undersides | Z18, Z58, Z98, Z138 — four decks, 40 mm pitch |
| Every sled | 3 mm thick; PCB insulating standoff starting point 8 mm |
| Relay usable PCB/component reservation | **46 × 76 × 25** above its standoffs, one board per level; terminal/wire overhang must also remain clear |
| Each utility reservation | **60 × 74 × 26** above its standoffs; split PCA/buck/watchdog between two levels only after actual fit check |
| Lowest lid ribs | Z180.5; top relay reserved payload ends Z174, leaving 6.5 mm |
| Hitch tongue | Z0 to 16, +Y side, gussets to Z106; tower foot sits on it at Z16 |

The conservative payload envelopes are design capacities checked against the printed assembly, **not
measured sizes of the purchased electronics**. Deck outlines are larger (relay 62 ×100, utility 77 ×100),
but stack columns, screw heads and wire routes consume space. If actual boards exceed these capacities,
change the shared layout before printing; do not omit the watchdog or stack components against the lid.

1. Fit PG9 glands and side breather to the body, checking locknuts and seal compression. The 4.5 mm floor
   is not assumed compatible with every gland. Three ports accommodate three round cable jackets/bundles
   only if a suitable gland actually seals them; use proper multi-hole inserts for multiple separate wires.
2. Fit M4 tray fasteners through the four floor/support bores and center-layer clearance holes.
   Add the 8 mm adapter thickness when selecting screw length; seal every exterior penetration and
   verify full nut engagement.
3. Install four M2.5 Pi fasteners through the tray and 8 mm printed posts; do not force M3 through Pi holes.
   Select screw length for tray + post + PCB + washers/nut, including the real HAT mounting arrangement.
4. Install the two four-column M3 stacks. Use straight threaded rods or appropriate spacers/locknuts;
   nominal deck separation is 40 mm, so the clear spacer gap between 3 mm decks is **37 mm**.
   Utility column centres are outside the Pi outline. Metal must not touch PCB traces or the HAT.
5. Drill component fixing holes in the removable sleds against the **actual** boards, preserving outer
   stack anchors and continuous webs. Existing parallel slots can help only when the real hole pattern
   matches. Do not assume slots magically fit every PCB. Use eight-millimetre insulating spacers initially.
6. Leave terminal screw/driver access, connector extraction and cable bend radii. Remove upper decks for
   service; route wires beside columns, not across the Pi fan or hot regulator. Separate high-current loads
   from logic; fuse holders and automotive relay housings can use external equipment saddles.
7. Fit the 3 mm cord to the constant-width 3.8 ×2.4 groove. Nominal compression is 20%, nominal area fill
   about 77.5%; centreline perimeter is about **634 mm**, plus trim allowance. Bond the cord ends properly.
8. Fit twelve M4 hex nuts from the flange underside and twelve M4 screws with flat washers through the lid
   (M4 ×12 is a starting point; verify actual nut/washer dimensions and projection). Tighten evenly without
   deforming the flange. Underside nut access remains clear of the seal. No heat-set inserts are required.
9. Fit `arm_layer_center` beneath the box flange/tongue and above the actual hitch plate; align the
   ball hole and auxiliary pattern only if both match. Fit each bolted arm root, then attach the
   outboard servos. The hitch plate in the assembly view is schematic: no positive anti-rotation
   interface has been verified on this mower, so do not power these arms until one is installed.
   Fit the camera tower to the +Y wall and tongue (§ Camera tower).

The body prints bed-down at 211 × 285 × 185 on the stated 420 × 420 × 500 printer.
A commercial rated enclosure remains the safer option where a demonstrated ingress rating is required.
Conformal coating does not make porous print walls certified waterproof. Test ingress unpowered and
verify Pi/Hailo/regulator temperatures under worst ambient/load; a breather is not a cooling system and
does not guarantee freedom from condensation.

## Hitch sandwich layer and drive arms

The `arm_layer_center` part adds an 8 mm panel beneath the enclosure flange and continues
along its hitch tongue. Two 42 mm side ears carry the detachable arms; its envelope is
**295 × 285 × 8 mm**. Each 60 × 70 mm solid root sits **above** the layer, with four M6
bolts inserted from beneath its free side ear into accessible side-loaded steel nuts.
The root's 50 × 70 mm hollow column has 6 mm walls and central web. Two
Ø8 side vents near its root open both channel lobes instead of trapping sealed
voids; keep them unobstructed. Its servo holder face has four M2.5 on 24 × 24 mm —
the holder's 65 × 30 flange, and only those four screw.
Print each lower arm lying flat, long axis in the bed plane. None of these
prints is qualified for powered steering.

**The mower and the electronics-box hitch interface are still unmeasured.** The body uses a
20 mm generic hole; the sandwich layer uses Ø20.8 mm. Print `hitch_gauge` and `arm_layer_gauge`
and measure the actual ball/bolt, plate width, depth and thickness before printing the center layer.
`hitch_plate_width_mm=160`, depth 120 and thickness 8 are only assembly-view placeholders.
The extrapolated arm axis is x=±110, but the root bolt centres follow that
axis at the layer seat (z=0); the arm column centre is at y=123, ahead of the
enclosure flange. Verify washer land on the side ears and clearance from the
actual mower plate and tower before fastening.

**Only the servo station is measured** — 209.55 mm out, on a 457.2 mm arm, splayed 12.58° from
vertical. The crank, lap-bar clamp point and 936.9 mm rod are a Toro 77502 **estimate**
(`TORO_77502_LINKAGE.md`); every plate dimension is schematic. None are measured mower coordinates.

The tongue/adapter also has two optional M6 auxiliary holes at x=±40, y=`tongue_bolt_y()`.
They match the enclosure's modeled tongue pattern but do **not** prove the mower hitch plate has
matching holes. A central hitch bolt by itself is not an approved powered-steering torque path.
Before powered operation, provide and verify a positive anti-rotation connection into the actual
hitch plate/chassis. No attachment strength or steering load rating is asserted here.

1. Confirm the mower model/serial and use its service manual. With engine off, key removed,
   battery isolated and mower secured, measure the hitch bolt/hole, plate outline/thickness,
   surrounding features and the bars.
2. Fit both coupons to the actual hardware; change `hitch_hole_d`, hitch-plate parameters,
   tongue reach or arm offset as needed. Rebuild and verify the revised plate/arm clearance.
3. Measure each lap-bar OD; adjust `clamp_tube_od_mm`, then choose among the three clamp coupons.
   The clamp halves use four M5 bolts per bar around the tube; nothing here is intended to drill
   the OEM bar. Test clamp slip and polymer creep without powering the mower.
4. Measure the lap-bar clamp point, crank radius and lever travel as listed in
   `TORO_77502_LINKAGE.md`. Set `lapbar_pin_fwd_mm`, `lapbar_pin_x_mm`, `lapbar_pin_dz_mm`
   and `servo_crank_*`, rebuild, and print rods only when the rod asserts and fit checks pass.
   Hand-check articulation at neutral, both travel limits and outboard PARK with pins installed and removed.
5. Four M6 per root thread into accessible **steel** M6 nuts slid into the
   root side slots, four M6 join each head, and four M2.5 secure each servo
   holder. Each sway pad takes two side-loaded M5 nuts/bolts and each half-jaw
   takes two top-loaded M4 nuts/short set screws. Verify head and washer land,
   nut retention and unpowered horn sweep against actual mower hardware.

**Unresolved and safety-critical:** mower hitch-plate anti-rotation, exact hitch shape, bar OD,
horn and drive-handle coordinates, dynamic steering forces, clamp retention, arm sway-bar and
head-lap integrity, ASA creep/fatigue, articulation, neutral return and PARK clearance. Do not run
powered steering until these are physically tested and the remaining force path is independently
qualified. A valid mesh or successful hand fit is not a strength certification.

## Camera tower

A hollow 50 × 50 mm ASA column stands on the tongue against the +Y wall and carries `camera_foot` at
`tower_height` (**900 mm placeholder** — measure from the hitch plate to the line of sight you need
over the seat back). Its bore routes a **USB** camera cable down and through a 24 mm wall port into the
box between the two electronics stacks. A CSI ribbon will not survive this run; use a USB camera.

**Parts** (`camera_tower.scad`): `camera_tower_base` (L-foot + wall flange + first 160 mm), N ×
`camera_tower_segment` (identical, flanged, spigot on top — N and length are derived from
`tower_height`, default 2 × 355 mm), `camera_tower_cap` (socket, side cable exit under a solid roof,
camera-foot bolts with side-entry nut traps), `camera_tower_backing` (template for a metal strip inside
box wall). Hardware: 6 × M5 wall bolts + nuts, 2 × M5 foot bolts through the foot, tongue and 8 mm
sandwich layer (measure grip length; longer than the old 30 mm placeholder), 4 × M4 per tower joint,
2 × M4 ×20 for the camera foot, plus `camera_foot`/`camera_carrier`.

**Print orientation is structural, not cosmetic.** Tube bending stress is 3.7 MPa at the base: FOS 11
along the filament, **2.9 across layers**. So: base prints **flange-down** (wall plate on the bed,
column horizontal, supports under the column and its joint flange); segments print **lying flat** on
one tube face; cap and backing print flat. Never print a segment standing up. Segment length is capped
at 400 mm to lie on the 420 bed with its spigot.

Sequence: bolt the base to the wall (backing strip inside) and down to the tongue; feed the USB cable
up through the base, each segment and the cap *before* fitting the cap's camera foot; drop each
segment's bore over the spigot below it and fit the 4 M4 flange bolts; cap last; a small drain at the
bottom of the base column lets any water that runs down the bore out below the port. If the column
hums at engine speed, brace its first joint flange to the box lid flange — the flange bolt holes are
the tie points.

## Actuation and E-stop installation

- The two stationary servo holders bolt to the **lateral** 24 × 24 mm M2.5 pattern on each arm head.
  The output shaft runs across the machine. The kit's metal output arm or holder is the crank: it
  points **up** at neutral, square to the level rod, and the rod eye sits outboard of it on an
  M6 pin. The case and its disc are the drawing's 65 × 30 × 48 mm and 61.4 mm axial
  envelope; the disc is about Ø30 and is not dimensioned on the drawing, so a caliper
  reading replaces `servo_disc_d` before any clearance claim. A crank radius below
  55 mm lets the rod transition hit the case and disc at ±35°; `rod_servo_sweep` is the
  check that decides, not a fixed minimum.
  Keep the crank sweep clear of the camera tower column and its sway-bar collars.
- Each tubular steering bar receives a two-piece printed clamp with **four M5 through-bolts per
  side** and broad washers. The OEM bar is not drilled. Tube OD, coating, clamp slip and long-term
  ASA creep are not known from CAD; fit the OD coupon, tighten only to a tested method and bench-load
  the clamp without a person near the mower.
- Each rod is a straight member between parallel global-X pins. The lap bar is farther outboard
  than the servo, so the eye fittings carry the measured lateral offset as a built skew. Changing the
  lock setting moves the servo eye 1.6 mm along its pin; take that up with the 2.5 mm washer stack.
  Larger changes need measured values and a rebuild. The skew adds about 0.079 × rod force as axial
  thrust on both pins. Remove a pin to check manual PARK; the linkage cannot follow the outboard swing.
- Each arm root uses four M6 through-bolts and washers/nuts, and each head joins its lower arm with
  four more M6 through a flanged, spigoted lap. The 50 × 70 × 6 mm-walled column and its 6 mm central
  web are sized as a first prototype shape, **not** a strength rating. All layer orientation, hole
  tolerances, fastening preload, ASA creep, fatigue, shock and mower-frame attachment require
  physical qualification.
- The six sway bars are position keepers, not load-path members. They tie each arm section to the
  camera base mount so the arm cannot sweep, and their print geometry assumes that interface stays
  bolted. They add no margin to the steering load path and must not be treated as a brace against
  servo reaction or a mower tip-over. Inspect for pad separation, nut-pocket tear-out and collar slip
  before and after every test.
- The listing's 12 V image gives up to 165 kg·cm stall torque (~16.2 N·m), corresponding to about
  540 N at a 30 mm horn radius. This is a stall-force warning, not a recommended duty load or proof
  that any printed arm, clamp, rod, hitch tongue, fastener or mower structure is safe.
- **A single ball-hole bolt does not establish positive torque restraint.** The arm layer has
  optional auxiliary holes matching the enclosure tongue, but this mower's hitch-plate holes are
  unverified. Add/verify a keyed restraint or matching fasteners into the actual plate/chassis
  before powered testing. No powered-use approval is made here.
- Check the OEM platform/seat relative motion, wheels, belts, hydro controls, brake/fuel wiring,
  deck travel, full lap-bar sweep and manual release. With the engine disabled, confirm the OEM
  levers return to neutral unaided when servos are disconnected/unpowered; a geared servo power cut
  alone does not prove failsafe return.
- The E-stop bracket is still the separate square-U-bolt fit fixture. Keep mushroom travel free,
  its rear cover clear and wires strain-relieved. Preserve OEM seat/PARK/PTO/start interlocks.
  Do not connect the RDS51150 servos to 24 V.

## Printing and verification

Use a well-bonded outdoor-capable material/process (ASA preferred; assess PETG temperature and creep).
No PLA for loaded/heat-exposed mounts. Start around 0.2 mm layers, at least 5–6 perimeters and appropriate
solid fastener lands. These are starting settings, not a qualified process. Use metal load spreaders;
never tighten by an invented universal torque.
The layer panel prints flat. The lower arms print **lying flat** along bed X;
the heads print **mating-face down**, and the handed half-jaw sway bars print **lying flat**
with the bar along bed Y. Use the exported left/right variants without extra mirroring. Inspect all nut slots
in the slicer. Rod pieces print lying flat with the rod axis along bed X, so lock and splice
bolts are vertical and both pin bores are horizontal. Clamp halves and fit coupons print
flat; the build refuses parts that are not bed-zero or exceed 420 × 420 × 500 mm.
For the half-jaw braces, "lying flat" means the web plane is **parallel** to the bed,
not that the whole web touches it. The jaw sets bed-zero while the web and end land
are raised. Add slicer supports beneath those raised sections and the jaw return;
inspect their first supported layers and clear every nut slot after printing.


```bash
# Rebuild ALL variants, check manifoldness/winding/connectedness/bounds,
# exercise negative-volume regression probes, then publish STLs and previews.
python3 hardware/mounts/build_printables.py

# Single variants when adjusting a measured interface:
openscad -o /tmp/arm-left.stl -D 'arm_part="left"' hardware/mounts/hitch_arm_layer.scad
openscad -o /tmp/clamp-cap.stl -D 'clamp_part="cap"' hardware/mounts/lapbar_four_bolt_clamp.scad
openscad -o /tmp/rod-servo-end.stl -D 'rod_part="servo_end"' hardware/mounts/pushrod.scad
openscad -o /tmp/arm-head.stl -D 'arm_part="head_left"' hardware/mounts/hitch_arm_layer.scad
openscad -o /tmp/sway-left.stl -D 'arm_brace_part="bar_1"' -D 'arm_brace_side="left"' hardware/mounts/arm_support.scad
```

Every part is published twice, from the same mesh: `<part>.stl` and `3mf/<part>.3mf` (the 3MF is
repackaged from the just-published STL, so the two always agree; it is shifted into the positive
octant and is byte-stable across runs). Feed a slicer either file.

Slicer projects you save into this directory (`*.3mf` at the top level) are **audited, never
rewritten** — they carry your print settings. Each build compares each project's embedded mesh
against the published part of the same name and reports `current`, `stale`, `no_source` or
`unreadable`, printing a `WARN` line and recording the verdict in `validation.json` under
`print_projects`. Re-slice from `3mf/<part>.3mf` whenever one is not `current`: a project can
silently hold a superseded part, which is how the old 939.7 mm rod survived in one of them.
OpenSCAD's STL export is not byte-reproducible between runs, so compare the recorded **dimensions**
rather than STL hashes; the 3MFs are stable.

`validation.json` records source/STL hashes, dimensions, watertight mesh topology,
and named fit probes. The build checks the **real** tower base plus segment flange,
continuous front insertion, left/right jaw separation, each of the three bar-to-pad contacts,
both root-to-layer seats, M5 bolt axes, root bores, hitch clearance, clamp/rod interfaces
and other enclosure/accessory paths. These are geometry checks, not mower fit, load,
fatigue, creep, ingress or powered-steering certification.

`assemblies.scad` includes the steering-arm, lap-bar/clamp, adjustable-rod, mounted-enclosure, E-stop,
tower and cutaway scenes used by `INSTALL.md`. Grey hitch/bar geometry and amber lap bars are
schematic; their dimensions are placeholders until measured on the mower.
The arm-containing and clamp-detail illustrations use explicit cameras because large CSG clipping solids
make OpenSCAD's automatic fit shrink the visible hardware. These cameras frame the default
dimensions; adjust the `VIEWS` camera entries after substantially changing the assembly envelope.
The clamp detail crops the illustrative tube and shows the eye and bolt at the actual modeled pivot.
