# Toro TimeCutter MAX 50 in MyRIDE — pushrod length estimate

Machine: **Toro 50 in (127 cm) TimeCutter® MAX MyRIDE®, model 77502**, Kawasaki FR691V,
twin-lever HG-EZT2200 hydrostatic drive. Operator's manual **3483-555** covers both the TimeCutter
MAX and MyRIDE variants. The serial plate selects the parts catalogue.

**This is an estimate, not a fit.** Toro publishes overall dimensions and assembly drawings
without lap-bar or hitch coordinates. Measure the lap-bar values below before printing rods.

## Result

| Quantity | Estimate | Range carried by the estimate |
|---|---|---|
| Hitch-hole centre → lap-bar clamp pin, fore-aft | **35 in (889 mm)** | 31–39 in |
| Lap-bar clamp pin from mower centreline | **14 in (356 mm)** | 13–15.5 in |
| Clamp pin height | level with the adapter's M6 pin | set by choosing the clamp height on the lever |
| Rod pin span in the fore-and-aft plane | **934 mm (36.8 in)** | ≈ 830–1040 mm |
| Rod pin-to-pin length | **936.4 mm (36.9 in)** | ≈ 33–41 in |
| Lateral offset carried by the rod | **67.3 mm** outboard at the lap bar | follows the `lapbar_pin_x_mm` estimate; the clamp point is not measured |
| Built rod skew | **4.12°** | follows the offset |

The CAD default uses the estimate. The rod adjusts **210 mm in 15 mm steps** across its two
multi-position joints, so the clamp point stays a choice inside the estimated band rather than a
reading; re-enter measured values only if the band itself turns out to be wrong.

## Basis

- **Measured station:** the servo face is 8¼ in (209.55 mm) left/right of the ball-hole centre and
  1 ft 6 in (457.2 mm) above the plate-top datum (`enclosure_common.scad`). The adapter's M6 pin
  is 55 mm above the shaft at neutral face45; the rod eye plane is 78.75 mm outboard of the servo
  face (the bracket half-span plus the printed adapter's pin local X).
- **Toro data:** [model 77502 page](https://www.toro.com/en/product/77502): assembled depth
  80 in, width 61.5 in, height 47 in, 19 in seat, 20 in rear tyres, 3 × 1.5 in tubular frame,
  standard hitch bracket. [Operator's manual 3483-555](https://www.toro.com/getpub/281676): two-axis
  motion-control levers that swing **outward to PARK**, lever height/tilt adjustment by bolts,
  and seat-only MyRIDE suspension adjustment. [Setup instructions 3464-688](https://www.toro.com/getpub/246581):
  the rear hitch bracket bolts beneath the rear frame plate.
- **Photo recognition:** in Toro's straight-front studio image of the TimeCutter MAX, scaled by the
  19 in seat, the lever lower arms meet the fender pods at about ±15–16 in in PARK and run inboard
  toward the seat-front corners. In the 3/4 views, the levers mount at the front of the fender pods,
  ahead of the seat and roughly above the front half of the 20 in rear tyres. The fore-aft
  estimate combines the hitch bracket's rearmost position, the Kawasaki engine bay behind the
  seat, and the seat depth. The 3/4 photographs alone cannot resolve better than about ±4 in.
- **Toro 2026 brochure:** the
  [zero-turn mowers product brochure](https://cdn2.toro.com/en/-/media/Files/Toro/Homeowner/zero-turn-mowers/2026/Toro-2026-Zero-Turn-Mowers-Product-Brochure.ashx)
  lists **77502** in its TimeCutter accessory table, and confirms from Toro's own text that the seat is
  isolated from the machine: *"Patented MyRIDE Suspension System isolates you from the mower frame"*.
  That is the relative-motion hazard below, stated by the manufacturer: the lever clamp is mounted on
  the moving platform while the servo sits on the frame, so fore-aft platform travel becomes a direct
  lever command. The same pages give 19 in seat, 20 × 10 in rear tyres, 5 gal fuel, a tubular frame and
  18,445 ft/min blade tip speed — each matching the datums used here.
- **Documents, verified reachable 2026-10-08.** The unit's own **operator's manual, with the
  illustrated parts lists**, is form **3477-630**: <https://www.toro.com/getpub/282292>, 44 pp,
  covering 77502 from serial 419002102, with Frame, Motion Control and Seat Pan exploded views
  (p5, p23, p37). The mechanics' guide is Toro's **TimeCutter Service Manual, form 3433-938 Rev A**
  (215 pp, 2020 CV generation), reachable as
  <https://images.webfronts.com/cache/frwfhnmtikys.pdf> — vendor-hosted, so treat that URL as
  perishable. Its **Steering Control Box** removal, disassembly and installation (p100, p105, p106,
  p110) is the lever mechanism this linkage drives; chassis/frame views are p55–60, the neutral
  adjustment p165, and torque tables span 79 pages. Those torques are for the **machine's steel
  fasteners** — not a licence for the printed parts, which carry no qualified torque. Toro indexes
  its own service manuals at
  <https://www.toro.com/en-ca/customer-support/commercial-education/service-manuals> (served as
  `media.toro.com/servicemanuals/<id>sl.pdf`); a third-party mirror of the operator's manual is
  manualspro.net/313111. Documents are linked, never committed: they are Toro's copyright.
- **Servo:** the supplied photo matches the listing used in `hardware.md`,
  [ANNIMOS B0C69W2QP7](https://www.amazon.com/dp/B0C69W2QP7/): RDS51150SG, 18-tooth spline,
  165 kg·cm at 12 V, two U-shaped aluminium holders and two aluminium discs. The listing's
  dimensioned drawing gives the 65 × 30 × 48 mm case with the output shaft along the
  48 mm dimension, a 61.4 mm overall axial envelope (case plus the kit's disc), a disc of
about Ø30 mm that the drawing does not dimension, and the **photo-derived six-hole rear bracket** (photo aa8b43) with slots and Ø13.5 opening.

## Basis

- **Measured station:** the servo face is 8¼ in (209.55 mm) left/right of the ball-hole centre and
  1 ft 6 in (457.2 mm) above the plate-top datum (`enclosure_common.scad`). The crank pin is
  55 mm above the shaft at neutral; the rod eye plane is 71.9 mm outboard of the servo face
  (the drawing's 61.4 mm axial envelope plus the crank and eye stack).
- **Toro data:** [model 77502 page](https://www.toro.com/en/product/77502): assembled depth
  80 in, width 61.5 in, height 47 in, 19 in seat, 20 in rear tyres, 3 × 1.5 in tubular frame,
  standard hitch bracket. [Operator's manual 3483-555](https://www.toro.com/getpub/281676): two-axis
  motion-control levers that swing **outward to PARK**, lever height/tilt adjustment by bolts,
  and seat-only MyRIDE suspension adjustment. [Setup instructions 3464-688](https://www.toro.com/getpub/246581):
  the rear hitch bracket bolts beneath the rear frame plate.
- **Photo recognition:** in Toro's straight-front studio image of the TimeCutter MAX, scaled by the
  19 in seat, the lever lower arms meet the fender pods at about ±15–16 in in PARK and run inboard
  toward the seat-front corners. In the 3/4 views, the levers mount at the front of the fender pods,
  ahead of the seat and roughly above the front half of the 20 in rear tyres. The fore-aft
  estimate combines the hitch bracket's rearmost position, the Kawasaki engine bay behind the
  seat, and the seat depth. The 3/4 photographs alone cannot resolve better than about ±4 in.
- **Toro 2026 brochure:** the
  [zero-turn mowers product brochure](https://cdn2.toro.com/en/-/media/Files/Toro/Homeowner/zero-turn-mowers/2026/Toro-2026-Zero-Turn-Mowers-Product-Brochure.ashx)
  lists **77502** in its TimeCutter accessory table, and confirms from Toro's own text that the seat is
  isolated from the machine: *"Patented MyRIDE Suspension System isolates you from the mower frame"*.
  That is the relative-motion hazard below, stated by the manufacturer: the lever clamp is mounted on
  the moving platform while the servo sits on the frame, so fore-aft platform travel becomes a direct
  lever command. The same pages give 19 in seat, 20 × 10 in rear tyres, 5 gal fuel, a tubular frame and
  18,445 ft/min blade tip speed — each matching the datums used here.
- **Documents, verified reachable 2026-10-08.** The unit's own **operator's manual, with the
  illustrated parts lists**, is form **3477-630**: <https://www.toro.com/getpub/282292>, 44 pp,
  covering 77502 from serial 419002102, with Frame, Motion Control and Seat Pan exploded views
  (p5, p23, p37). The mechanics' guide is Toro's **TimeCutter Service Manual, form 3433-938 Rev A**
  (215 pp, 2020 CV generation), reachable as
  <https://images.webfronts.com/cache/frwfhnmtikys.pdf> — vendor-hosted, so treat that URL as
  perishable. Its **Steering Control Box** removal, disassembly and installation (p100, p105, p106,
  p110) is the lever mechanism this linkage drives; chassis/frame views are p55–60, the neutral
  adjustment p165, and torque tables span 79 pages. Those torques are for the **machine's steel
  fasteners** — not a licence for the printed parts, which carry no qualified torque. Toro indexes
  its own service manuals at
  <https://www.toro.com/en-ca/customer-support/commercial-education/service-manuals> (served as
  `media.toro.com/servicemanuals/<id>sl.pdf`); a third-party mirror of the operator's manual is
  manualspro.net/313111. Documents are linked, never committed: they are Toro's copyright.
- **Servo:** the supplied photo matches the listing used in `hardware.md`,
  [ANNIMOS B0C69W2QP7](https://www.amazon.com/dp/B0C69W2QP7): RDS51150SG, 18-tooth spline,
  165 kg·cm at 12 V, two U-shaped aluminium holders and two aluminium discs. The listing's
  dimensioned drawing gives the 65 × 30 × 48 mm case with the output shaft along the
  48 mm dimension, a 61.4 mm overall axial envelope (case plus the kit's disc), a disc of
about Ø30 mm that the drawing does not dimension, and the **photo-derived six-hole rear bracket** (photo aa8b43) with slots and Ø13.5 opening.

## Why the rod changed

| Problem in the 500–650 mm rod | Change |
|---|---|
| About 300 mm too short for this machine | 936.4 mm default pin length from the estimate above |
| Stretched to ~937 mm, the 32 mm tube / 21.1 mm inner bar buckles below the 540 N stall force at a 30 mm horn (margin 0.77) | 40 mm × 5 mm square tube in **four** bolted pieces plus a 29.1 mm inner bar: Euler load ≈ 1.96 kN at E = 1.5 GPa |
| Horn pointed along the rod at neutral, so the servo started near dead centre | Printed adapter's clevis holds the M6 pin **up** at neutral face45, square to a level rod |
| Servo fork prongs would collide with the disc and the case | Single eye in the printed adapter's 12 mm clevis gap on an M6 pin in **double shear**; `rod_servo_sweep` sweeps the eye against case, disc and adapter over 0–90° face |
| Planar YZ placement required the lap bar directly in front of the servo | Both pins remain global X; the eye fittings carry a 4.12° built skew |

At the 55 mm **printed adapter design radius**, the stall force is
16.2 N·m / 0.055 m ≈ **295 N**, giving an Euler margin of about **6.7**. The buckling check
treats the splices as continuous and uses a modulus assumption; it is not a strength rating.
The skew adds an axial thrust of about 0.072 × rod force on each pin. The clevis ears and
shims carry that thrust.

## Measure before printing
1. Neutral: both levers centred (not PARK), seat occupied or loaded to normal riding weight. Set each
   lever's bolt-adjusted height and tilt first; changing either later invalidates these values.
2. Choose the clamp point on each lever where a level rod from the **adapter's M6 pin height**
   reaches it and the clamp clears grip, PARK swing, seat, fender and deck lift. Record from the
   **hitch-hole centre**:
   - forward distance to the clamp pin → `lapbar_pin_fwd_mm`
   - distance from the mower centreline → `lapbar_pin_x_mm`
   - height relative to the adapter's M6 pin → `lapbar_pin_dz_mm` (0 = level rod)
3. On the servo's metal arm/holder, measure the pin hole radius from the output shaft →
   `servo_crank_r`; caliper the case and the disc and set `servo_disc_d` (the drawing does not
   dimension it). 55 mm is a placeholder radius. The eye and rod transition must clear the case,
   disc and adapter over the 0–90° face range; `rod_servo_sweep` and `rod_servo_sweep_band` check
   this for the full radius band. If the kit part is shorter or the sweep fails, use a steel crank
   plate bolted to the disc or reduce `servo_crank_r`, then rebuild.
4. Move each lever fully forward and reverse and record the clamp-pin travel. With a 55 mm crank,
   rotating from face45 to face90 moves the adapter pin about 38.9 mm rearward and 16 mm down;
   the forward bound is symmetric. More travel needs a longer crank and recalculated force.
5. **MyRIDE check:** while the operator rides, measure fore-aft movement of each clamp pin relative
   to the hitch plate. Any fore-aft platform motion becomes a direct lever command because the
   servos mount on the frame. The level rod makes vertical-only motion negligible (25 mm vertical
   ≈ 0.3 mm span change), but fore-aft motion is not. Stop if it is more than a few millimetres.
6. Enter the values, run `python3 hardware/mounts/build_printables.py`, and print only if every
   rod assert and fit check passes.

Routing past the engine, muffler, belts and seat platform cannot be checked from Toro's published
data. Keep ASA well away from the muffler and exhaust heat.
