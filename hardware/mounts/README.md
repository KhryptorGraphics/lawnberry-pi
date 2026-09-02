# Actuation mounts — Toro TimeCutter 50" zero-turn conversion

Parametric OpenSCAD sources plus rendered STLs and preview images for mounting
the LawnBerry Pi drive-by-wire hardware to a 50" Toro TimeCutter zero-turn.

## What is researched vs. what you must measure

| Dimension | Status |
|---|---|
| Frame rail: **3" × 1.5" × 0.120" wall rectangular steel** | **Published by Toro** for the TimeCutter carrier frame. Used directly. |
| Lap-bar tube OD | **Not published, and no longer needed.** The saddle uses a self-centring V-groove that seats **12.8–34 mm (0.50"–1.34")**. |
| Servo body 65 × 30 × 48 mm | Manufacturer spec (DSSERVO RDS51150SG). |
| Servo U-bracket hole spacing | Approximate — mounting holes are slotted to absorb error. |

Everything machine-facing lives in `common.scad`. Edit it, re-render.

### On the lap-bar diameter

Toro does not publish it. It is absent from the operator's manual (form
3465-589), the setup instructions (form 3363-807), the TimeCutter Z service
manual, the product page, the parts catalogues, and every grip and accessory
listing checked — aftermarket ZTR brackets only ever quote a broad 0.8"–2"
range. The setup instructions do establish that the lever is a tube pinned to
a "control arm shaft" with two 3/8" bolts, which implies roughly 1" or larger,
but that is an inference, not a specification.

Rather than ship a guessed radius, the saddle uses a **90° V-groove**. Any tube
in range seats concentrically and self-centres; the U-bolt pulls it into the V.
No measurement, no reprint. The tube centre sits 0.707 × D above the V apex, so
its height varies slightly with diameter — irrelevant for a ball-jointed link
with an adjustable rod length.

## Why U-bolts instead of printed split clamps

Every mount here is a **saddle held by U-bolts**, not a printed clamp with bolt
ears. Two reasons:

1. A printed ear carrying bolt preload is the weakest possible arrangement on a
   vibrating petrol machine. U-bolts put the clamping load through steel; the
   printed part only locates and spreads it.
2. The lap-bar OD is unpublished. A U-bolt saddle tolerates whatever the tube
   actually measures; a fixed-diameter split clamp would be a guess.

This replaced an earlier split-clamp design, which is why there are no longer
any `*_cap.scad` parts.

## What the manuals established

From the operator's manual (form 3465-589) and the TimeCutter Z service manual:

| Fact | Consequence |
|---|---|
| Levers move on **two axes**: fore/aft (Fast–Slow–Neutral–Reverse) and **outboard to PARK** | Servos drive fore/aft only. Nothing here actuates the outboard swing. |
| **PARK is the parking brake** | A positive brake exists; it just isn't reachable by this linkage, so engaging it stays manual. |
| Engine cranks only with blade off **and levers in PARK** | Autonomous start is **physically impossible** here. The OEM interlock wants levers *outboard*; software's `require_levers_neutral_to_start` checks *centred*. Different positions. |
| Engine stops **within 1 s** if levers leave PARK with the seat unoccupied | Empty-seat autonomy needs the seat circuit deliberately addressed. |
| **"Let the handles go. They should return to the neutral position."** (service manual, Control Handle — Return To Neutral Adjustment) | **The levers self-centre.** This is the assumption the whole bus-fault failsafe rests on, and it is now manufacturer-documented rather than hoped for. Lap-bar return springs are the mechanism. Still verify with the linkage *fitted* — added friction could defeat it. |
| Throttle continuously variable; choke **manual** | Throttle is servo-driven; cold start is always human. |
| 292 kg (644 lb), max slope **15°** | Use 15° as the field-validation ceiling. |

### The operating sequence this forces

Human starts the engine with levers in PARK → moves them to centre → hands over
to autonomous control. The starter relay does not change this; it cannot
satisfy an interlock no actuator can reach.

## Files

| File | Qty | Notes |
|---|---|---|
| `common.scad` | — | Shared parameters + saddle/U-bolt helpers. **Edit this first.** |
| `servo_frame_mount.scad` | 2 | Drive servo → frame rail. `-D side=1` right, `-D side=-1` left. |
| `lapbar_pushrod_clamp.scad` | 2 | Lap-bar saddle with ball-joint eye. |
| `throttle_servo_mount.scad` | 1 | Throttle servo, control-panel foot. |
| `estop_bracket.scad` | 1 | 22 mm mushroom button on the frame rail. |
| `electronics_tray.scad` | 1 | Pi 5 + PCA9685 + regulator + relay module, for the IP65 box. |
| `prev_*.png` | — | Rendered previews, for sanity-checking geometry before printing. |

```bash
openscad -o servo_frame_mount_right.stl -D side=1  servo_frame_mount.scad
openscad -o servo_frame_mount_left.stl  -D side=-1 servo_frame_mount.scad
openscad -o estop_bracket.stl estop_bracket.scad
# preview (headless):
xvfb-run -a openscad -o prev_estop_bracket.png --imgsize=900,675 \
  --camera=0,0,0,62,0,38,0 --autocenter --viewall estop_bracket.scad
```

OpenSCAD errors on non-manifold geometry, so a clean render is your first check
that an edit didn't break the model. **Look at the preview too** — manifold and
*correct* are different things; several construction bugs in these parts passed
the manifold check and were only visible in a render.

## Hardware you need beyond the prints

- **U-bolts, 4 total**: 2 square/rectangular for the 3" × 1.5" frame rail
  (5/16" leg), 2 round for the lap bar (1/4" leg — buy to suit the tube once
  you can see it; the V-groove handles the diameter, the U-bolt just needs to
  reach around).
- **M6 rod ends** — 2 male, 2 female — plus M6 threaded rod and clevis pins
  with R-clips. Ball joints at *both* rod ends are not optional: they give the
  angular freedom the lever needs, and the clevis pin lets you unpin in seconds
  to restore full manual PARK travel.
- Use the **metal servo arms supplied with the servo**. Never a printed horn.

## Material and print settings

These parts hold a 165 kg·cm servo against a lever on a vibrating petrol
machine, outdoors.

- **PETG or ASA.** Not PLA — it creeps under sustained load and softens in a
  machine parked in the sun.
- **6+ perimeters**, 40–60% infill on the saddles.
- Print saddles with the U-bolt holes vertical so layer lines don't split
  along the clamping force.
- The **electronics tray** has no structural role — print it however you like.

If a saddle shows any sign of creep, have it cut from aluminium. Printed
plastic is fine for locating a servo; it is not fine as the only thing between
a 644 lb machine and an uncommanded lever movement.

## What still needs measuring

Only the servo standoff/linkage geometry — `standoff` in
`servo_frame_mount.scad` and `anchor_reach` in `lapbar_pushrod_clamp.scad` —
which set how far the servo sits off the rail and where the rod anchors. Work
these out with the lever in neutral and check the full fore/aft sweep, and the
outboard PARK swing, before committing. The frame rail is a known dimension and
the lap-bar diameter is handled by the V-groove, so nothing else is a guess.

## Remaining bench check

Confirm return-to-neutral **with the linkage fitted and unpowered**. The
service manual confirms the bare machine self-centres; what it can't tell you
is whether your ball joints, pushrod and servo cogging torque add enough
friction to prevent it. That is acceptance item 10, and item 1 (the bus-fault
failsafe) is gated on it.

## Model caveat

The manuals studied were **TimeCutter Max 50 in (form 3465-589)** and the
**TimeCutter Z service manual**. You described the machine as "TimeCutter 2",
which is not a name Toro publishes — the 50" line includes TimeCutter,
TimeCutter MAX and TimeCutter MyRIDE. Control architecture is consistent across
the line, and the 3" × 1.5" frame spec is quoted for TimeCutter MAX models, but
check your serial plate and pull the matching manual before fabricating.
