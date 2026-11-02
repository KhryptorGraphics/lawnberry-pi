# Toro conversion — printable mounts and fit fixtures

Parametric OpenSCAD sources, binary STL exports, PNG previews and reproducible geometry checks.
Dimensions are millimetres. **These files are not a machine-fit, structural or ingress certification.**
The mower's engine and hydrostatic transaxles stay stock; the servos move the existing controls.
There are no replacement drive-motor mounts in this package.

**Where things go on the machine, with rendered views: [`INSTALL.md`](INSTALL.md).**

## Start here

1. The target is the **Toro TimeCutter MAX 50 in MyRIDE, model 77502**. Confirm the serial plate and
   measure the hitch plate's hole, width, depth and thickness. Plate dimensions remain unverified.
2. Print `hitch_gauge`, `arm_layer_gauge`, `fit_gauge_2`, `servo_adapter_gauge`, the
   `lapbar_clamp_insert_0`–`_4` sleeve set, and all three `pushrod_gauge_*` coupons. Try the sleeves on
   the steering tubes, lay `fit_gauge_2` on the stationary rear bracket and `servo_adapter_gauge` on the
   moving crossplate of each purchased servo.
3. The rods are generated from an **estimate** (`TORO_77502_LINKAGE.md`): 941.9 mm pin-to-pin.
   Measure the lap-bar clamp point and lever travel; set `lapbar_pin_*` in `enclosure_common.scad`
   and rebuild before printing rods.
4. The printed center layer, arms, clamps and rods are **fit/load-test prototypes**, not qualified
   steering parts. A single hitch bolt is not an approved anti-rotation restraint. Do not power the
   steering until a positive chassis torque path, full hand-travel clearance, clamp retention and
   unpowered neutral return are physically verified.
5. The user-stated printer is 420 × 420 × 500 mm. All separate part variants are built against it;
   inspect real usable bed volume, orientation, material and slicer preview before printing.

The enclosure body prints bed-down at 211 × 285 × 185 mm including its tongue;
the separate sandwich-layer center panel is 295 × 285 × 8 mm. The longest
arm members fit a 420 mm bed and the rod pieces the 400 × 400 × 490 mm usable volume (the two
print-in-place rod pieces stand 446 and 406 mm tall) at the selected print orientations,
but mower clearance and load capacity remain unverified.

## Dimensional evidence

| Interface | Basis and current treatment |
|---|---|
| RDS51150 stationary rear bracket | **Photo-derived, not a factory drawing** (photo aa8b43). The narrow stationary rear bracket carries **six** holes; centres relative to its rear opening (Y across the bracket, Z up) are (±10, 18.5), (±8, 8) and (±8, −7.5) mm (`servo_rear_hole_points_mm()`). Each prints as a Ø2.9 rounded slot with **±0.4 mm** cutter-centre trim in Y and Z (an M2.5 shank centre can reach about ±0.6 mm), which keeps the web beside the opening but does **not** span the full photo uncertainty. The **Ø13.5** rear opening is a capacity for the photo estimate of 12 ± 1.5 mm. Selected M2.5 steel through-bolts with OD6 × 1 washers and AF5 × 2.5 nuts; nothing threads into the print. Verify the purchased RDS51150SG with `fit_gauge_2` before printing heads. |
| RDS51150 moving crossplate | The U crossplate is the MOVING member; the adapter does not bolt to the disc. By user direction (2026-10-12) the adapter uses the **same six passages, ±0.4 mm slot trim and Ø13.5 opening as the arm heads** (`servo_rear_hole_points_mm()`). On the plate, bracket Y maps to the tangential direction and bracket Z to the shaft axis, the (±10, 18.5) row outboard toward the clevis (`servo_adapter_hole_points_mm()`); that is the only orientation that fits the 52 × 30 mm photo envelope, so it is an **assumption**: check it with `servo_adapter_gauge`, whose arrow marks the 18.5 row. The 25.4 mm plate-seat radius (`servo_moving_face_r_mm`) is a capacity. Plate face **45° = neutral** (levers centred) and **90° = full reverse**, as the user stated. The 0° face is only an analysis bound, not a confirmed forward stop or PWM calibration. |
| Hitch sandwich + arm layer | The enclosure flange is **211 × 170 mm**; an 8 mm center layer continues to the modeled hitch hole and has 42 mm side ears (STL envelope **295 × 285 × 8 mm**). Hole Ø20.8 and auxiliary M6 centres ±40 mm mirror the unverified enclosure tongue. The **160 × 120 × 8 mm** mower-plate envelope is schematic; measure it. |
| Splayed servo arm | **MEASURED** station: servo face **209.55 mm** left/right of the ball-hole centre, **457.2 mm** from its plate-top datum, splayed **12.58°** from vertical. Both heads now carry the servo **101.6 mm (4 in) higher** than that datum (`servo_station_raise_mm`, 2026-11-01) so the throttle MG995 fits below the right one; the station X is unchanged. The axis extrapolates to x=±110, while the solid 60 × 70 × 50 mm root seats on the center layer **top** at z=0 and y=123. Four M6 per root enter upward into side-access steel nuts; the hollow 50 × 70 mm column has a through-channel and 6 mm walls/web. Lower arm joint at local t=355 mm; printed head retains a four-M6 flanged/spigoted lap and the lateral six-hole rear-bracket land. Neither fit nor strength on the real mower is established. |
| Arm sway bars | Three handed first-article braces per side land at z=110/190/250 mm. Each left/right half-jaw occupies its own side of the tower; the pair forms a rear-open U with a 0.4 mm centre gap. Fit from **+Y toward the box**, never from -Y or over a flange. Two top-loaded M4 nuts per half retain short screws bearing **outside** the USB-cable tube. Two M5 bores meet each arm pad's side-entry nuts, with 0.2 mm face clearance. These are **not steering-load parts**; test slip, creep, nut pullout and pad damage unpowered. The column arrives drilled **right through** at each station for the collars' M4 screws — one bolt per station passes both walls, so the collar pair and the column are a single bolted joint; the base carries station 0, the shared segment the rest. |
| Lap-bar connector | Fully printed two-piece split clamp with a **set of interchangeable split sleeves that set the bar size**: the body, bolts, yoke and pin serve every bar in the set, and the size is chosen by *trying the sleeves on the bar* — no calipers. The bore is fixed (`clamp_body_bore_mm`, Ø36.4); the set spans **20–30 mm** (`clamp_bar_od_mm`: 20 / 22.2 / 25.4 / 28.6 / 30), and a bar between entries needs one more sleeve printed, never a new clamp. A sleeve-wall assert refuses an entry too near the bore. Paired clamp yoke ears capture one 32 × 20 × 8 mm rod eye in double shear with an M6 through-pin. Clamp preload, coating, slip and print strength need physical testing. |
| Pushrods | Toro 77502 **estimate**: 941.9 mm pin-to-pin, 939.5 mm span in the fore-and-aft plane (the rod falls 101.6 mm from the raised servo crank to the lap-bar clamp estimate), lap-bar pin 67.3 mm outboard of the adapter's M6 pin (4.10° built skew). **No purchased fasteners in the rod.** 40 mm square tube, 5 mm walls, in **two** pieces plus a 29.1 mm solid inner bar: three prints per rod. The servo end's mouth is the socket for the sleeve's 29.6 mm square spigot (70 mm deep, keyed by the square); a **print-in-place joint lever** clamps the spigot and drops a tooth into a pocket in it. A second print-in-place lever on the sleeve's mouth drops its tooth into one of **19 pockets on the inner bar: 216 mm of length in 12 mm steps** (833.9–1049.9 mm), so the clamp point stays a constrained choice along the lever instead of a tape reading. Teeth, windows and pockets are arcs about each lever's pivot, so a push or pull cannot open a lever: both locks are **positive** and the lobe's clamp is friction on top, never the lock. Holding ≥ 1080 N each way (2 × the ~540 N stall at a 30 mm horn) is **unqualified** until the `pushrod_*_coupon` pairs pass a bench test. Euler estimate (E = 1.5 GPa, stepped section) 2.2 kN at 941.9 mm, 1.6 kN at the longest setting, at or above the bolted rod at the same lengths. Both ends are 32 × 20 × 8 mm eyes on Ø6.6 bores carried by the mating parts' M6 pins (servo adapter clevis, clamp yoke). The servo-end eye sits in the printed adapter's 12 mm clevis gap, 78.75 mm outboard of the measured servo face; at the 45° neutral face the pin is straight above the shaft. `rod_servo_sweep` and `rod_servo_sweep_band` check the eye and rod transition against the servo body, moving crossplate and adapter over the 0–90° face range. See [`TORO_77502_LINKAGE.md`](TORO_77502_LINKAGE.md). |
| E-stop rail U-bolts | Selected BOM 40 mm inside + 8 mm leg = 48 mm centre pitch, printed Ø9. Used only by the E-stop fixture; the 76.2 × 38.1 mm rail remains unverified. |
| E-stop HB2-ES545 | Listing drawing: Ø22 panel opening, Ø40 mushroom, 42 mm rear projection, 30 × 29 mm contact envelope. Print bore Ø22.5; 3 mm panel is a design choice, not a vendor maximum thickness. No guessed anti-rotation notch. |
| Pi 5 | Official reference drawing: **85 × 56 mm PCB**, **58 × 49 mm** mounting pitch, 3.5 mm edge offsets, Ø2.7 PCB holes. Held without screws in the `control_77603` compute tower (`tower_rack` + `tower_clamp`; kit standoffs act as spacers only). |
| Camera Module v2.1 | Official drawing: **23.862 × 25 mm** PCB with R2 corners, Ø2.2 holes at (2,2), (14.5,2), (2,23), (14.5,23), **8.5 mm square** lens housing. The PCB is held by its edges in a groove cassette, not by those holes; the open right (CSI) edge is never bent over a lip. PCB thickness, barrel diameter and lens projection are **unknown**: the cassette carries 0.6–3.0 mm PCB-thickness and 2.2–24 mm projection **capacities**, not measurements. |
| SVPRO stereo board | Dimensional listing image: **80 × 16.5 mm** PCB, two **14 × 13 mm** lens bases and a 9 mm USB-C connector width. Its 16 mm height has an ambiguous datum; barrel diameter, PCB thickness, projection and lens baseline are **unknown**. Each lens has its own independently trimmed insert, so no stereo baseline is assumed. |
| Opto relays | The linked product images show **one relay/channel per PCB**. The owner confirmed **two** HiLetgo 5 V 30 A single-channel modules on hand (`docs/2026-10-30-user-confirmed-owned-hardware.md`); `control_77603` holds them in strap bays B1/B2. Board outline/hole pitch remains unverified, so the bays grip any box within their range. |
| Other electronics/sensors | Carrier SKU, outline, terminals and mounting patterns are not established for the bought PCA9685, buck, watchdog perfboard, ZED-F9P, BNO085, VL53L0X/BME280/INA3221 carriers or antenna. Sensor blanks are explicitly drill-to-fit; the enclosure internals are the bolt-free `control_77603` chassis and carriers (`control_77603/README.md`). |
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
| `arm_layer_center` | 1 | Sandwich panel matching the box flange and tongue, with hitch/auxiliary holes, four M4 box-mount holes (those bolts also clamp the `control_77603` chassis base), three provisional gland passages, camera-tower foot holes and side-arm bolt ears. Measure the actual hitch plate; current dimensions are schematic. |
| `arm_layer_left`, `arm_layer_right` | 1 each | 50 × 70 mm hollow columns with three inboard pads, a 60 × 70 mm solid root with four side-loaded M6 steel nuts, and a flanged/spigoted head joint at local t=355 mm. Print lying flat; inspect nut access. |
| `arm_head_left`, `arm_head_right` | 1 each | Heads carrying the **lateral** six-hole land for the RDS51150 stationary rear bracket on a rib raised 101.6 mm above the measured station; four-M6 flanged joint. All six photo-derived Ø2.9 slots (±0.4 mm trim) and the Ø13.5 rear opening pass through the **whole** rib and 6 mm land, so each M2.5 through-bolt's washer and nut sit on the exposed flat **inboard X=0 face**. Print mating-face down; keep every slot open. Heads printed before 2026-11-01 sit 4 in lower: do not mix them with the raised ones. |
| `throttle_arm_head` | 1 | **Replaces `arm_head_right` for the throttle build; do not print both.** The raised right arm head — joint flange, upright rib, servo-face land, six photo-derived Ø2.9 slots (±0.4 mm trim), Ø13.5 rear opening and spigot pocket kept — plus a **shelf under the steering servo** carried by a 45° corbel and 45° wings, with the MG995 **cradle walls printed on the shelf**. Print **mating-face down** like the other arm heads, without support, and keep every slot open. The MG995 drops into the cradle and a one-piece snap keeper clips into windows in the cradle walls; **no new purchased fastener** is specified for this mount. Fit, joint strength and the load path into the mower are **not qualified** — treat the throttle station as an unproven fit prototype. Source: `throttle_77603/throttle_arm_head.scad`. |
| `arm_brace_bar_left_0`–`_2`, `arm_brace_bar_right_0`–`_2` | 1 each | Six handed half-jaw braces at z=110/190/250. Print the labelled left/right variants; do not substitute two identical copies. Fit each pair from +Y after tower assembly. Each half has two top-loaded M4 nuts, outside-bearing screws and two M5 bores for its arm pad. Prints flat; sway only, not a qualified steering load path. |
| `arm_layer_gauge` | 1 first | Coupon for the enclosure tongue's ball-hole/auxiliary-hole pattern. It is not a Toro hitch gauge. |
| `lapbar_clamp_anchor`, `lapbar_clamp_cap` | 1 each per side | The size-independent half: its Ø36.4 bore takes any sleeve, so print it once per side whatever the bar measures. Paired yoke ears; four M5 through-bolts per side. The clamp pin passes through both ears and the single rod eye in double shear. |
| `lapbar_clamp_insert_0`–`_4` | 2 per side of the chosen size | Split sleeve that sets the bar size. The set spans 20 / 22.2 / 25.4 / 28.6 / 30 mm bar OD (`clamp_bar_od_mm`), and each bore is a slip fit over its own bar, so a sleeve doubles as its own coupon: try them on the bar, then print two more per side of whichever fits. One printed half serves both clamp halves (turn it 180° about the bar axis). End flanges trap it against the clamp's end faces; they are retention, not structure. |
| `pushrod_servo_end`, `pushrod_sleeve`, `pushrod_inner` | 1 each per side (2 sets) | One 941.9 mm nominal rod per side with **216 mm of adjustment in 12 mm rack steps**; the same set serves both sides, turned 180° about its own axis on the left (levers outboard on both). It falls 101.6 mm from the raised servo crank to the lap-bar clamp estimate. `pushrod_servo_end` (446 mm) and `pushrod_sleeve` (406 mm) are **print-in-place, 2 bodies each**: they print upright, lever mouth on the bed, the lever printed open beside it; work each lever free through its 90° swing before assembly. `pushrod_inner` prints lying flat. No bolts in the rod. |
| `pushrod_gauge_0`–`_2` | 1 each | Sliding-fit coupons (tight/nominal/loose) for the 29.1 mm inner bar in the 40 mm sleeve. |
| `pushrod_joint_coupon` + `pushrod_joint_coupon_spigot`, `pushrod_rack_coupon` + `pushrod_rack_coupon_bar` | 1 each | Holding-test coupons cut from the rod's own lock geometry, each with a Ø6.6 test-pin bore in a solid grip (test-rig pins only). Lock each pair and load it axially to ≥ 1080 N in tension and compression before trusting a rod on the mower. The two `*_coupon` housings are print-in-place (2 bodies). |
| `servo_adapter_right`, `servo_adapter_left` | 1 each | Printed adapter from the RDS51150 **moving crossplate** to the rod eye, with the arm heads' **six-hole** M2.5 pattern and Ø13.5 opening. 10 mm pad, two 9 mm full-length ear webs with Ø24 pivot bosses, a clevis floor and an inboard gusset; six M2.5 steel through-bolts, OD6 washers and AF5 nuts on the flat pad top. The 12 mm clevis carries the rod eye on an M6 pin in **double shear**. Printed handed; do not mirror one copy in the slicer. Fit/load-test prototype only. |
| `servo_adapter_gauge` | 1 first | 3 mm coupon with the adapter's six slotted passages, Ø13.5 opening and orientation arrow; lay it on each purchased moving crossplate before printing adapters. |
| `throttle_servo_mount` | 0 or 1 | Optional stationary-side fixture; four M6 base slots at nominal 76 × 40 pitch with ±4 mm X travel, and the same six-slot photo-derived rear-bracket pattern and Ø13.5 opening as the arm heads, with OD6 washers/AF5 nuts on its free back face. Installer-selected support, not a Toro panel pattern. |
| `estop_bracket` | 1 | Ø22 mushroom panel with external gussets and two square U-bolts. |
| `estop_contact_cover` | 1 | Removable rear guard; four M3 ×12 screws/nuts/washers; downward wire opening and tie slots. Not a sealed switch housing. |
| `enclosure_body`, `enclosure_lid` | 1 each | Tall serviceable electronics box with an integral rear hitch tongue; common dimensions in `enclosure_common.scad`. The body is unchanged; the lid is reprinted with the two HaLow antenna stations (§ HaLow antenna port and mount). The electronics inside sit on the removable `control_77603` chassis. |
| `hitch_gauge` | 1 first | Small coupon for the enclosure tongue's hitch hole. The 20 mm hole in that tongue remains unverified; measure before printing the body. |
| `camera_tower_base`, `camera_tower_segment`, `camera_tower_backing` | 1 base, N segments, 1 backing | Hollow column below the lower camera; N and length derive from the lower lens datum `tower_height`. Backing is a drilling template for a metal strip. |
| `camera_tower_extension`, `camera_tower_cap` | 1 each | Exact 304.8 mm flange-to-flange tube above the Pi housing; closed terminal cap above the stereo housing. No cap cable slit or camera-foot mount. |
| `camera_pi_housing`, `camera_stereo_housing` | 1 each | Lower Pi / upper stereo shell with a bottom-open broad PCB/lens key and narrow upper-nut keys; four M3 tray fixings into side-access steel nuts. Prints on its rear flange (exported orientation). |
| `camera_pi_bottom`, `camera_stereo_bottom` | 1 each | Removable tray with the **opaque closing face integral** to it, so camera and face travel together. Face windows carry the inserts; two M3 cassette bolts and the floor-gasket recess. |
| `camera_pi_carrier`, `camera_stereo_carrier` | 1 each | Groove cassette: shallow top/bottom edge grooves (3.2 mm maximum PCB + pad stack, 0.8 mm edge engagement) with a fixed left stop. Two fore/aft M3 slots give Pi −11.8…+10 / stereo −8…+10 mm board travel. Pi CSI edge and stereo 24 mm upper-centre (USB-C) gap stay open. |
| `camera_pi_retainer`, `camera_stereo_retainer` | 1 each | Removable right-edge retainer; two M2 through-bolts/nuts. The Pi retainer bridges behind the CSI edge with a long through-passage (21 mm cut capacity): choose the actual M2 length on the bench. Support the corner bridges; do not print on the small PCB-edge tabs. |
| `camera_pi_lens_insert`, `camera_stereo_lens_insert_left`, `camera_stereo_lens_insert_right` | 1 each | Opaque **lens-only** insert, one round flared aperture per lens; fitted from the **front** along +Y with two M3 bolts in slots (Pi ±2 mm X, stereo ±1 mm X, both ±3 mm Z fine trim). Seal each slot with a **20 mm OD** metal plus compliant washer; 10 mm washers expose the extreme trim. Prototype bores Pi Ø8 / stereo Ø14; set `camera_insert_bore_mm` and the coarse offsets from the gauges. |
| `camera_pi_face_gasket`, `camera_pi_floor_gasket`, `camera_pi_insert_gasket`, `camera_pi_rim_gasket` | 1 each | TPU solids or cutting templates for closed-cell foam: shell-to-face, floor recess, face-to-insert and lens-rim seals. The 0.4 mm installed gaps need compliant stock; no IP/weather rating. |
| `camera_stereo_face_gasket`, `camera_stereo_floor_gasket`, `camera_stereo_insert_gasket_left/_right`, `camera_stereo_rim_gasket_left/_right` | 1 each | Stereo set of the same seals, one insert and rim gasket per lens. |
| `camera_pi_lens_gauge`, `camera_stereo_lens_gauge` | 1 first | Labelled bore try-fit strips (Pi Ø5–12, stereo Ø8–20). Offer the real lens's non-optical rim, never the glass; pick the smallest loose bore. |
| `camera_pi_tile_fit_kit`, `camera_stereo_tile_fit_kit` | 1 first | Complete opaque inserts at indexed coarse offsets (X −6…+6, Z −4…+4 in 2 mm steps, filtered to the selected bore's capacity), labelled with their offset. Multi-part plates by design; the unfiltered layout can reach about 400 × 394 mm. |
| `camera_pi_groove_pad_kit`, `camera_stereo_groove_pad_kit` | 1 first | Groove shims 0.4–2.4 mm thick for top and bottom edges, slid in from the open +X end; thin insulating compliant tape may replace them. Multi-part plate by design. |
| `camera_foot`, `sensor_carrier` | As needed | Sensor-only tilt foot and drill-to-fit backplate; M4 pivot and two M4 foot fixings. Neither camera uses this foot. |
| `antenna_deck` | 1 if needed | 120 mm deck with edge straps/M4 mounting; optional metal ground plane. |
| `equipment_saddle` | As needed | Two-strap platform for housed automotive relays/inline fuse holders. |
| `harness_saddle` | As needed | Raised tie tunnel and two M4 fixing holes for cable strain relief. |
| `fit_gauge_1`, `fit_gauge_2` | 1 each initially | E-stop rail/U-bolt cross-section, and the RDS51150 stationary rear bracket's six photo-derived slots plus Ø13.5 opening. |

The generic sensor carrier is a finished drilling blank. Enter measured `sensor_pcb_holes` and the
correct `sensor_pcb_bore` in `sensor_carrier.scad` to regenerate an exact variant, or transfer the actual
PCB pattern onto the blank. Use insulating spacers; do not strap bare boards/components. Locate IMUs
rigidly on the same chassis reference used for calibration, away from servo magnets and high-current
wiring. ToF apertures must remain unobstructed; an arbitrary transparent cover can cause crosstalk.
An exposed sensor PCB or antenna connector still needs appropriate environmental protection.
Both camera fronts are **opaque**: no transparent pane or glazing is used or supplied. Each lens looks
through its own round flared aperture in a printed insert. Printed housings and gaskets alone do not
establish an ingress rating.

## Enclosure and stack assembly

**Interior: 169 × 128 × 180.** Body flange 211 × 170; with the hitch tongue the print is
**211 × 285 × 184.5** (tongue extends 132 mm beyond the +Y wall, in the flange's own width).
Lid: 219 × 178 × 19.5 in print orientation (the two antenna post-socket bosses hang 14.5 mm into
the box). Closed height: 189.5.

**Inside the box.** The Pi 5 + Libre Alta tower and the interface electronics sit on the removable
chassis in `control_77603/` (see `control_77603/README.md`). Its one-piece `chassis_base` (3.5 mm,
the legacy tray's thickness) is clamped onto the four floor-boss tops under the existing
box-to-arm-layer M4 bolt heads and washers, exactly where the tray was, so the bolt grip is
unchanged; the chassis hooks under the base and is held by a detent, with no added fasteners. The
former electronics tray, relay sleds, utility sleds and their M3 stack columns are removed. The
two HaLow antennas' N-bulkhead ports and grip posts are on the reprinted lid's back (−Y) edge
(§ HaLow antenna port and mount).

Assembly Z is measured from the enclosure underside:

| Item | Z |
|---|---|
| Floor top | 4.5 |
| Floor boss top: M4 washer land; chassis base clamped here | 14.5 |
| Chassis plate underside (on the base) | 18 |
| Lowest lid ribs | Z180.5 |
| Hitch tongue | Z0 to 16, +Y side, gussets to Z106; tower foot sits on it at Z16 |

1. Fit PG9 glands and side breather to the body, checking locknuts and seal compression. The 4.5 mm floor
   is not assumed compatible with every gland. Three ports accommodate three round cable jackets/bundles
   only if a suitable gland actually seals them; use proper multi-hole inserts for multiple separate wires.
2. Fit the four M4 box-to-arm-layer bolts through the floor bosses and center-layer clearance holes, head
   and washer on each boss inside the box, clamping the `control_77603` `chassis_base` (3.5 mm, the
   legacy tray's thickness). Add the 8 mm adapter thickness when selecting screw length; seal every
   exterior penetration and verify full nut engagement.
3. Lower the `control_77603` chassis 6 mm +Y of its seat, push it −Y until the base detent clicks, and
   load it as `control_77603/README.md` describes. Leave terminal screw/driver access, connector
   extraction and cable bend radii; separate high-current loads from logic. To remove it, press the
   detent nub through the plate hole (x 62, y −9), push the chassis +Y and lift; bolt service means
   lifting the chassis out, then unbolting the base.
4. Fit the 3 mm cord to the constant-width 3.8 ×2.4 groove. Nominal compression is 20%, nominal area fill
   about 77.5%; centreline perimeter is about **634 mm**, plus trim allowance. Bond the cord ends properly.
5. Fit twelve M4 hex nuts from the flange underside and twelve M4 screws with flat washers through the lid
   (M4 ×12 is a starting point; verify actual nut/washer dimensions and projection). Tighten evenly without
   deforming the flange. Underside nut access remains clear of the seal. No heat-set inserts are required.
6. Fit `arm_layer_center` beneath the box flange/tongue and above the actual hitch plate; align the
   ball hole and auxiliary pattern only if both match. Fit each bolted arm root, then attach the
   outboard servos. The hitch plate in the assembly view is schematic: no positive anti-rotation
   interface has been verified on this mower, so do not power these arms until one is installed.
   Fit the camera tower to the +Y wall and tongue (§ Camera tower).

The body prints bed-down at 211 × 285 × 185 on the stated 420 × 420 × 500 printer.
A commercial rated enclosure remains the safer option where a demonstrated ingress rating is required.
Conformal coating does not make porous print walls certified waterproof. Test ingress unpowered and
verify Pi/Hailo/Alta/regulator temperatures under worst ambient/load; a breather is not a cooling system
and does not guarantee freedom from condensation.

## HaLow antenna port and mount

The owner bought two purchased Rokland 10 dBi "Backcountry" 915 MHz omnis, one for each HaLow
radio ([vendor page](https://store.rokland.com/products/10-dbi-backcountry-n-male-omni-outdoor-helium-915-mhz-antenna-48-for-rak-miner-2-nebra-indoor-bobcat-hotspots):
N-male base, 45 in = 1143 mm tall, fibreglass radome, 60 m/s wind rating, two stainless U-bolts
and a bracket supplied). The body stays exactly as committed: both ports and mounts are on the
**lid**, which is reprinted. `antenna_mount.scad` holds the stations and `enclosure_lid.scad`
carries them. The axes are at (±64.5, −48) in body coordinates, 129 mm apart, near the lid's back (−Y)
corners. The −X antenna feeds the Alta's HaLow and the +X antenna feeds the Pi 5's.

- **Port.** The 5 mm lid plate is the bulkhead panel; the Amphenol 82-369 jack accepts 5.3 mm
  ([bulkhead page](https://www.farnell.com/datasheets/269482.pdf)). Each N-female bulkhead passes
  a D-flat hole, Ø16.3 with 13.9 flats ([Type N catalog](https://www.farnell.com/datasheets/1931349.pdf)).
  Its pigtail's far end must match the radio it feeds (U.FL on a bare HaLow card, SMA on an
  HT-HR01 carrier); confirm each before ordering.
  The hole is inside the cord-seal line and over the box interior, so the nut, jack and pigtail
  hang straight into the box.
  - On the outside, the flange (or nut and washer) and the O-ring seat on the lid's flat top. That
    face prints on the bed and is the smoothest surface on the part.
  - On the inside, the nut or hex shoulder seats on the flat inner face. The HEAD lid ribs are cut
    back within 20 mm of each hole for spanner room.
  - Water: the lid top is flat and level, so water runs off its edges; it is not shed by a slope.
    No printed feature stands within 10 mm of either flange, so nothing dams a ring of water round
    the joint. The O-ring is the seal; tape over the N joint is the weather cover.
- **Mount.** A printed post stands inboard (+Y) of each antenna.
  - Its 42.6 × 12 × 16 mm spigot drops into a blind socket in a boss under the lid. The boss hangs
    to z 170 and has a 3 mm floor, so the socket has no path into the box.
  - A flexure snap on the spigot clicks under a recess in the socket wall. To release it, press
    the finger tab at the post's inboard foot into the post and lift.
  - Two band grips per antenna, centres 66 mm apart, slide down the post's dovetail rails onto
    ledges. They hold the base tube concentric with the bulkhead axis, so the connector carries
    only the antenna's weight.
  - Load path: antenna → grips → post rails → post column → the column foot bearing on the lid
    top, plus the spigot bearing in the socket boss → lid plate → 12 × M4 → body flange.
  - No fasteners are added. The snap only keeps the post seated; it is not the bending path.
- **Grip size.** Rokland does not publish the base-tube diameter, length or mass. Measure the tube,
  then print all four grips with `-D ant_tube_d_mm=<measured OD>`. The design range is 25-40 mm,
  and the probes check contact at both 25 and 40. Print the posts with `-D post_side=-1` (−X, Alta)
  and `-D post_side=1` (+X, Pi); the two posts are mirror images. All four grips are the same part.

Install order. Steps 1-2 are bench work with the lid off the box; steps 4-7 are done once the lid
is on the box:

1. Lid top up: push each post's spigot into its socket until the snap clicks.
2. Feed each pigtail's U.FL end down through its D-hole. Seat the bulkhead's O-ring and flange on
   the lid top, key the D-flat, and fit the nut and lock washer on the inner face. Tighten to the
   connector maker's torque. If the bulkhead has its hex shoulder at the cable end, it goes in
   from the inside and its nut sits on the lid top instead; both faces are lands.
3. With the boards loaded (INSTALL step 3), hold the lid top up about 100 mm above the seal line.
   Route each pigtail along the `control_77603` corridors to its HaLow card and plug it in (U.FL,
   or the SMA on an HT-HR01 carrier); `antenna_lid_keepout()` is the under-lid envelope, and the
   1.13 mm cable's minimum bend radius is not published (~6 mm is an estimate). Keep at least
   100 mm of slack, then lower the lid onto the seal cord and fit the 12 M4.
4. Slide each lower grip into the gap between its post's rail segments, then down onto the lower
   ledge. Slide each upper grip onto its upper rail from the top.
5. Lower each antenna through its two band loops onto its bulkhead and screw the N-male coupling
   nut on by hand.
6. Pull each grip's tab until the band draws the tube onto the crush ribs and the pawl holds. To
   release, press the pawl finger outward and push the tab back.
7. Wrap each N joint, from the coupling nut down onto the bulkhead flange, with self-fusing
   silicone tape.

**Opening the box.** The lid lifts with both antennas, posts and bulkheads on it, and the pigtails
tie it to the radios.

1. Cut the tape, unscrew both coupling nuts, release the four bands and lift both antennas out.
   A 1.1 m whip on a lid that is being lifted is unmanageable.
2. Remove the 12 M4 and lift the lid straight up.
3. Each pigtail needs at least 100 mm of slack beyond its closed route so the lid can rise about
   100 mm. In that gap, unplug both pigtails at their HaLow cards (U.FL, or the SMA on an HT-HR01
   carrier), then set the lid aside.
4. `control_77603` routes 400 mm pigtails. U.FL connectors are rated for few mating cycles, so
   open the box no more often than needed.
5. Reverse the steps to close.

**Fallback: the supplied U-bolts.** If a band slips or creeps, use the horizontal oblong slots
(9 mm wide, legs 18-25 mm either side of the axis) that run through each post and grip at the grip's
mid-height.
- Cut the band off and keep the grip as a concentric saddle.
- Pass the U-bolt round the tube and through the grip and post, and fit the nuts on the post's
  back face.
- The U-bolt leg diameter and spacing are not published: check them against the slots first.
- This costs nothing and changes nothing in normal use.

**RF.**
- **FCC.** Rokland's notice says conducted output with this 10 dBi antenna should not exceed
  26 dBm under Part 15.247 unless a qualified installer confirms another limit for the specific
  FCC-authorised system. Keep each radio at its 24 dBm cap (34 dBm EIRP) and do not raise either.
- **Coupling between the two antennas.** They stand 129 mm apart, about 0.4 wavelength at 915 MHz.
  Put the two HaLow links on well-separated channels and expect some coupling and receiver desense
  between them. Check both links' throughput with both radios transmitting.

**Unqualified.** Nothing here is rated for ingress, wind load, vibration, UV or fatigue.
- The band, pawl, ribs and post snap are untested on a mower.
- Each 1143 mm whip's sway loads its post, the socket boss and the 5 mm lid plate. Printed
  ASA/PETG creeps under a sustained clamp.
- After the first mowing hours, inspect the grips, the posts and the lid round the bosses; then
  retension, or fall back to the U-bolts.

**To measure:**
- the antenna's base-tube OD and length, coupling-nut height and mass;
- the purchased bulkhead's hole, flange, O-ring, nut thickness, body length and which end its hex
  shoulder is on;
- which RF connector each HaLow card or carrier has (U.FL or SMA), and the pigtail's length and
  bend radius;
- the U-bolt size.

## Hitch sandwich layer and drive arms

The `arm_layer_center` part adds an 8 mm panel beneath the enclosure flange and continues
along its hitch tongue. Two 42 mm side ears carry the detachable arms; its envelope is
**295 × 285 × 8 mm**. Each 60 × 70 mm solid root sits **above** the layer, with four M6
bolts inserted from beneath its free side ear into accessible side-loaded steel nuts.
The root's 50 × 70 mm hollow column has 6 mm walls and central web. Two
Ø8 side vents near its root open both channel lobes instead of trapping sealed
voids; keep them unobstructed. Its lateral servo land carries the six photo-derived rear-bracket
slots and the Ø13.5 rear opening, through the whole rib (see § Actuation).
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
vertical; both servos now sit 101.6 mm above it. The adapter pin radius, lap-bar clamp point and
941.9 mm rod are a Toro 77502 **estimate** (`TORO_77502_LINKAGE.md`); every plate dimension is
schematic. None are measured mower coordinates.

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
3. No calipers needed for the bar: try the sleeve set on it and keep the size that slips on
   (`lapbar_clamp_insert_0`–`_4`), then print two more per side of that size. The clamp body,
   bolts, yoke and pin are printed once and never change with the bar; a bar between entries needs
   one more sleeve printed from `clamp_bar_od_mm`. Nothing here is intended to drill the OEM bar.
   Test clamp slip and polymer creep without powering the mower.
4. Measure the lap-bar clamp point and lever travel as listed in `TORO_77502_LINKAGE.md`. Set
   `lapbar_pin_fwd_mm`, `lapbar_pin_x_mm` and `lapbar_pin_dz_mm`, rebuild, and print rods only when
   the rod asserts and fit checks pass. Hand-check articulation at the 45° neutral face, the 90°
   full-reverse face and outboard PARK with pins installed and removed.
5. Four M6 per root thread into accessible **steel** M6 nuts slid into the
   root side slots, four M6 join each head, six M2.5 through-bolts secure each stationary
   rear bracket and six M2.5 through-bolts secure each adapter to its moving crossplate.
   Each sway pad takes two side-loaded M5 nuts/bolts and each half-jaw takes two top-loaded
   M4 nuts/short set screws. Verify head and washer land, nut retention and the unpowered
   crossplate/adapter sweep against actual mower hardware.

**Unresolved and safety-critical:** mower hitch-plate anti-rotation, exact hitch shape, bar OD,
adapter and drive-handle coordinates, dynamic steering forces, clamp retention, arm sway-bar and
head-lap integrity, ASA creep/fatigue, articulation, neutral return and PARK clearance. Do not run
powered steering until these are physically tested and the remaining force path is independently
qualified. A valid mesh or successful hand fit is not a strength certification.

## Camera tower

The stack order is **base → lower mast segments → enclosed Pi Camera Module 2 → 304.8 mm
(12 in) extension → enclosed SVPRO stereo camera → closed terminal mast cap**. Both optical axes
point toward −Y. Each housing is 80 mm tall, so the optical centres are **384.8 mm apart**;
the extension dimension is not the lens-to-lens distance.

### Height and structural limits

`tower_height` is the lower Pi lens-centre datum, **900 mm by default**, not the cap or a tilt-foot
surface. Measure the required sightline over the seat before setting it in `enclosure_common.scad`.
Segment count and length derive from that datum. The added upper housing and extension invalidate
the old 3.7 MPa / FOS 11 / cross-layer FOS 2.9 estimates: **no stress, fatigue or vibration margin is
qualified for this taller, heavier stack**. Validate wall/backing strength, joint retention and
engine-speed vibration physically before use.

Print the base flange-down, mast segments and extension lying on a tube face, and cap/backing in
their exported orientation. Never stand a mast segment up to improve appearance at the cost of
layer strength. Use six M5 wall bolts/nuts with a measured metal backing strip, two M5 foot bolts
through foot/tongue/8 mm sandwich layer, and four M4 bolts/nuts at every flange joint. Select lengths
from actual grip and nut engagement; neither camera is attached by cap-roof foot bolts.

### Bottom loading and service

Each station is one shell fixed in the mast plus a removable service set: the tray with its
**integral opaque closing face**, the groove cassette, the right-edge retainer, the lens-only
inserts and the gaskets. A Pi station has **9** physical printed parts and a stereo station **12**
(shell, tray, cassette, retainer, floor gasket and face gasket, plus an insert, insert gasket and
rim gasket per lens); gauges, shim kits and purchased fasteners are extra.

1. **Bench loading.** Slide the PCB from +X into the cassette's shallow top/bottom edge grooves
   toward the fixed left stop; never bend a populated board over a lip. The grooves hold a
   **3.2 mm maximum PCB + pad stack** with **0.8 mm edge engagement**; fill the gap for the real
   board from the groove-pad kit (0.4–2.4 mm) or thin insulating compliant tape, then screw on the
   removable right-edge retainer with two M2 through-bolts and nuts. Pi contact spans x −13.6…3 mm;
   the middle of its CSI edge stays open behind a rear-bridged retainer with 12 mm chosen connector
   clearance. Stereo's 24 mm upper-centre gap clears its 9 mm USB-C connector. Confirm on the
   actual board that those narrow lands are free of components.
2. Bolt the cassette to the tray through its two fore/aft M3 slots. The slots give **Pi −11.8…+10 mm**
   and **stereo −8…+10 mm** board travel, so the board front can sit at y −107.8…−86 (Pi) or
   −104…−86 (stereo). Set it from the actual lens projection; the 2.2–24 mm (Pi) and 7–25 mm (stereo)
   projection ranges are **capacities**, not measurements.
3. Lay the floor gasket in the tray recess and lift the tray + cassette straight **up (+Z)** through
   the shell's bottom-open broad PCB/lens key and narrow upper-nut keys. Retain it with the four
   original M3 bottom fixings into side-access steel nuts.
4. **Fit the lens inserts from the front**, along +Y, each over its face gasket and insert gasket,
   with its rim gasket on the lens's non-optical rim. Never sweep a closed round hole vertically
   across a protruding lens, and never press on the glass. Two M3 bolts per insert, each with a
   **20 mm OD** metal plus compliant sealing washer. Inserts may stay fitted for later bottom
   servicing, because the camera, closing face and inserts travel together.

**Choosing a bore and centre.** Offer the real lens to `camera_<kind>_lens_gauge` and choose the smallest
loose bore (prototype default Pi Ø8, stereo Ø14). Set `camera_insert_bore_mm`, then try the opaque
tiles in `camera_<kind>_tile_fit_kit` with the lens behind them, using the actual camera as the
jig. Enter the winning coarse offset (`camera_insert_offset_x_mm`/`_z_mm` for Pi;
`camera_stereo_lens_offsets_mm = [[left_dx,left_dz],[right_dx,right_dz]]` for stereo) and rebuild;
fine slot travel closes the 2 mm gaps between tiles. An assert refuses any bore/offset whose aperture,
front flare and full fine travel would leave the opaque port window (centre capacity =
window edge + bore/2 + thickness × tan(FOV/2) + 0.5 mm). For the defaults that is Pi x −15.2…15.2,
z 26.8…53.2 mm and stereo x ±(15.3…33.7), z 33.3…46.7 mm: **capacities, not a measured stereo
baseline or lens position**.

For service, release cable strain, remove the four bottom M3 fixings and lower the whole service set
straight down **without removing the mast or shell**. Gaskets are TPU solids or foam cutting templates
and need compliant stock, preload or adhesive as appropriate.

### Cable route, electrical and weather limits

Thread cables before inserting carriers: each housing's rear **30 × 50 mm** port leads into the
continuous **32.2 mm square minimum throat**, then down through the base/backing/electronics-box
**30 × 50 mm rectangular** entry. The design connector gauge is **24 × 14 × 45 mm**, including
90-degree port bends **before camera carriers/windows and deck payloads are fitted**; the route
does not pass through installed PCBs. The closed base duct has a flared relief for an elevated
connector bend above the brace shafts. Installed M4 brace bolts are included in its geometric gate;
the integrated CAD gate passed. Restrain the flexible base cable near z=122 mm inside the duct
and centre the upper cable between the brace bolts (§7 of INSTALL). This design capacity does not
guarantee every USB overmould fits. Measure clearance and bend radii; do not force plugs or fold ribbons.

The stereo camera uses **USB**; Camera Module 2 uses **CSI ribbon, not USB**. Pi 5 requires a compatible
22-to-15-pin camera cable/extension. A passive approximately 1 m CSI run is **not proven reliable**:
prove signal integrity in the installed system or select a suitable active CSI extender and verify
its power, compatibility and environmental protection.

After threading, seal the rectangular entry with a split gasket or compatible sealant; keep the
base sump drain and separate lower web-pocket weep vent open. Enclosed bays do not confer an IP rating or
fog resistance. Qualify ingress, condensation, optical reflections and vibration physically.
Each lens has its own opaque insert, so **no stereo baseline is assumed**. The 110° (Pi) and 85°
(stereo) per-lens cones are design capacities for the insert flare, not camera calibration.

`build_printables.py` publishes every STL/PNG pair with a matching 3MF and runs every clearance and
support/contact gate; the current counts and verdicts are in `validation.json`. These are CAD/mesh
results, not a physical camera-fit, electrical, ingress or structural qualification.
See [installation steps and bottom-exploded views](INSTALL.md#7-camera-tower).

## Actuation and E-stop installation

- **Stationary side: six-hole rear bracket.** Each servo's narrow stationary rear bracket bolts to the
  lateral land on its arm head with **six M2.5 steel through-bolts**. Insert each bolt from the
  bracket (outboard) side; put an **OD6 × 1 washer and an AF5 × 2.5 nut on the exposed flat inboard
  X=0 face** of the head and turn them with a slim (≤ Ø8) socket from −X. No nut is buried in the
  rib. The slots allow **±0.4 mm** cutter-centre trim in Y and Z; the **Ø13.5** opening clears the
  rear boss. The pattern is **photo-derived** (photo aa8b43): prove it with `fit_gauge_2` on the
  purchased servo before printing heads, and never enlarge slots blindly, because the photo
  uncertainty is wider than the web around the opening allows. Each bolt must span the head's
  22.3 mm reach plus bracket, washer and nut; choose the length from the real stack.
- **Moving side: six-hole crossplate adapter.** `servo_adapter_right`/`_left` bolts to the U
  crossplate that is the **moving** member, not to the disc, using the **same six passages and
  Ø13.5 opening as the arm heads**. Insert six M2.5 steel through-bolts from inside the U and put an
  **OD6 × 1 washer and AF5 × 2.5 nut** on the flat 10 mm pad top; a ≤ Ø8 socket reaches every nut.
  Each bolt spans the plate (2 mm capacity), the 10 mm pad, washer and nut: start at M2.5 × 18 and
  choose the length from the real stack. The arrow on the pad points to the (±10, 18.5) row and the
  clevis; if the gauge only fits the real plate turned 90°, stop and re-model rather than drilling.
  The adapter's two 9 mm ears form a 12 mm clevis around the 8 mm rod eye (2 mm shim/trim space each
  side). An **M6 pin passes ear-eye-ear in double shear**; never thread a printed ear.
- **Face angle and clocking.** The crossplate face is at **45° in neutral** (levers centred) and
  **90° at full reverse**. The clevis is clocked +45° to the plate, so at the 45° neutral face the
  M6 pin is **straight above the shaft** at the 55 mm design radius, with a level rod; at 90° it
  swings rearward and down 16 mm, clear of dead centre. The 0° face is only an analysis bound, not a
  confirmed forward stop. **Clocking:** the bolt pattern fixes the adapter to the plate, so clocking
  is set at the spline. With the levers hand-set to neutral and the rod pin removed, rotate the output
  by hand until the crossplate face is at 45° and the pin bore sits vertically above the shaft, then
  bolt the adapter. Mark the spline/plate and the adapter together,
  hand-move the lever to full reverse and confirm the face reaches 90° without the adapter, rod or
  pin touching the body or arm. Re-check the marks after every disassembly. Electrical command
  direction is calibrated separately for each motor; these angles are **not PWM values**.
  View `view_servo_adapter.png` (neutral solid, full reverse ghosted).
- Fit the servo with its connector end inboard (toward the tower) so the lead runs back to the box
  rather than out across the lap-bar sweep, and keep the adapter sweep clear of the tower column and
  its sway-bar collars. `rod_servo_sweep` and `rod_servo_sweep_band` re-check the rod's servo end
  against the body, crossplate and adapter on every rebuild.
- Each tubular steering bar receives a two-piece printed clamp with **four M5 through-bolts per
  side** and broad washers. The OEM bar is not drilled. Tube OD, coating, clamp slip and long-term
  ASA creep are not known from CAD; try the sleeve set on the bar, tighten only to a tested method and bench-load
  the clamp without a person near the mower.
- Each rod is a straight member between parallel global-X pins. The lap bar is farther outboard
  than the servo, so the eye fittings carry the estimated lateral offset as a 4.12° built skew. Changing
  the lock setting moves the servo eye about 1.1 mm along its pin; take that up with the shims in the
  adapter's 12 mm clevis gap (2 mm each side). Larger changes need measured values and a rebuild. The
  skew adds about 0.072 × rod force as axial thrust on both pins. Remove a pin to check manual PARK;
  the linkage cannot follow the outboard swing.
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
- **Load warning.** The listing's 12 V image gives up to **165 kg·cm stall torque (≈16 N·m)**: about
  295 N at the 55 mm adapter pin radius, applied to the adapter, its six M2.5 bolts, the six-bolt rear
  bracket joint and every downstream part. This is a stall-force warning, not a duty load. **No part of
  the servo mounting or adapter is qualified for powered steering**; a valid mesh and a clean hand fit
  prove geometry only.
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
in the slicer. The print-in-place rod pieces (`pushrod_servo_end`, `pushrod_sleeve`) print upright,
lever mouth on the bed and lever open beside it; `pushrod_inner` prints lying flat with the rod axis
along bed X. Both pin bores are horizontal. Clamp halves and sleeves print
flat; each sleeve prints on its split face with the flanges up and needs no support.
The build refuses parts that are not bed-zero or exceed 420 × 420 × 500 mm.
For the half-jaw braces, "lying flat" means the web plane is **parallel** to the bed,
not that the whole web touches it. The jaw sets bed-zero while the web and end land
are raised. Add slicer supports beneath those raised sections and the jaw return;
inspect their first supported layers and clear every nut slot after printing.

Retired exports have been removed: the bolted rod's `pushrod_middle`, `pushrod_middle_a` and
`pushrod_middle_b` (the print-in-place rod has no middle piece); `fit_gauge_3`/`fit_gauge_4` are obsolete round-U-bolt/V-seat steering gauges.
Their old previews and the old pushrod 3MF are removed too. Keep `fit_gauge_1`, `fit_gauge_2`
and `arm_layer_gauge`: the builder now regenerates all three active fit coupons and
records their STL/3MF outputs in the validated print manifest.

The arm-layer coupon now has one central hole and two auxiliary holes wholly inside its
centred plate. Replace older edge-cutout versions of `arm_layer_gauge`; its new regression
gate requires both the three through-bores and material at all four plate corners.


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
and inspect geometry rather than relying on a raw STL hash; the 3MFs are stable for a given STL mesh.

`validation.json` records source hashes and **two per-part mesh hashes**: `stl_sha256` fingerprints
the published file bytes; `mesh_sha256` sorts rounded triangles to ignore vertex/face ordering.
Re-tessellating the same surface can change either hash, so neither proves geometric equivalence.
The manifest also records dimensions, watertight mesh topology and named fit probes.
The build checks the tower joints and cable path, bottom-loaded cameras and their lens-only
apertures, photographed servo-bracket passages, adapter/rod sweep, brace contacts and the
remaining enclosure/accessory interfaces. These are geometry checks, not mower fit,
load, fatigue, creep, ingress or powered-steering certification.

The current full CAD build published **78 STL/preview pairs, 78 matching 3MFs and 17 views**;
**78 clearance gates and 89 indexed material/contact gates passed**. This is geometry evidence,
not confirmation that the purchased servo bracket, lens barrels, PCB component edges, mower
mounting or outdoor sealing fit physically. Trial-fit the coupons before load testing.

Head-joint validation requires material at both flange seats, the locating spigot and
the socket roof, plus eight independently indexed flange-contact checks. These use
the actual head assembly placement, not a reconstructed nominal head pose. A 1 mm
floating-joint fault passes the non-overlap check but fails the new seating gates.

`assemblies.scad` includes the steering-arm, lap-bar/clamp, adjustable-rod, mounted-enclosure, E-stop,
tower and cutaway scenes used by `INSTALL.md`. Grey hitch/bar geometry and amber lap bars are
schematic; their dimensions are placeholders until measured on the mower.
The arm-containing and clamp-detail illustrations use explicit cameras because large CSG clipping solids
make OpenSCAD's automatic fit shrink the visible hardware. These cameras frame the default
dimensions; adjust the `VIEWS` camera entries after substantially changing the assembly envelope.
The clamp detail crops the illustrative tube and shows the eye and bolt at the actual modeled pivot.
