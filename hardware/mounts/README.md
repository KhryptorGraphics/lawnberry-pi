# Actuation mounts — Toro TimeCutter 50" zero-turn conversion

Parametric OpenSCAD sources plus rendered STLs for mounting the LawnBerry Pi
drive-by-wire hardware to a 50" Toro TimeCutter zero-turn mower.

**The STLs are rendered from placeholder dimensions and will not fit your
machine as-is.** The Toro operator's manual (form 3465-589) contains no
dimensioned drawings, so every machine-facing dimension here is a guess.
Measure your mower, edit the parameters, re-render. That is the intended
workflow — these are not drop-in parts.

## What the manual established

Read from the TimeCutter Max 50" operator's manual, form 3465-589:

| Fact | Consequence for this build |
|---|---|
| Motion-control levers move on **two axes**: fore/aft (Fast–Slow–Neutral–Reverse) and **outboard to PARK** | The servos drive fore/aft only. Nothing here actuates the outboard swing. |
| **PARK (levers outboard) IS the parking brake** | The machine has a positive parking brake, but it is not reachable by this linkage. Engaging it stays a manual action. |
| Engine cranks only with **blade switch off AND levers in PARK** | Autonomous engine start is **physically impossible** with fore/aft-only servos. The OEM interlock wants levers *outboard*; software's `require_levers_neutral_to_start` checks *centred*. Different positions — do not confuse them. |
| Engine shuts off **within 1 second** if levers are out of PARK and the operator rises from the seat | Empty-seat autonomy cannot work until the seat switch is deliberately addressed. |
| Throttle is a **continuous-variable lever**; choke is **separate and manual** | Throttle is servo-driven. Cold starting stays human in every phase. |
| Machine weight **292 kg (644 lb)**, max slope **15°** | Use 15° as the slope limit in field validation — this closes the "open input" the validation protocol previously flagged. |
| **Bypass levers** on both sides of the engine disengage the hydro drive for pushing | Useful for recovery; keep them accessible and locked rearward in operation. |

### The consequence worth repeating

A human starts the engine with the levers in PARK, moves them to centre, and
*then* hands over to autonomous control. The starter relay in the parts list
does not change this — it cannot satisfy an interlock that requires a lever
position no actuator here can reach.

## Files

| File | Qty | Notes |
|---|---|---|
| `common.scad` | — | Shared parameters and the split-clamp module. **Edit this first.** |
| `servo_frame_mount.scad` | 2 | Lap-bar drive servo → frame tube. `-D side=1` right, `-D side=-1` left. |
| `servo_frame_mount_cap.scad` | 2 | Closing half of the above clamp. Captive M6 nuts. |
| `lapbar_pushrod_clamp.scad` | 2 | Grips the lap bar, carries the ball-joint eye. |
| `lapbar_pushrod_clamp_cap.scad` | 2 | Closing half. Captive M5 nuts. |
| `throttle_servo_mount.scad` | 1 | Third servo, control-panel side. |
| `electronics_tray.scad` | 1 | Pi 5 + PCA9685 + Pololu reg + relay module, for the IP65 box. |
| `estop_bracket.scad` | 1 | 22 mm mushroom button on a frame tube. |

Render:

```bash
openscad -o servo_frame_mount_right.stl -D side=1  servo_frame_mount.scad
openscad -o servo_frame_mount_left.stl  -D side=-1 servo_frame_mount.scad
openscad -o estop_bracket.stl estop_bracket.scad
# ...etc
```

OpenSCAD errors on non-manifold geometry, so a clean render is your first
check that an edit didn't break the model.

## Measure these before printing

Edit `common.scad`:

- `frame_tube_od` — outside diameter of the tube each servo mount clamps to.
  Round tube assumed; square tube needs the clamp module reworked.
- `lapbar_tube_od` — outside diameter of the lap-bar tube at the clamp point.
- `servo_bracket_hole_spacing_x` / `_y` — hole pattern on the U-brackets
  supplied with the servo. Mounting holes are slotted, so small errors are
  absorbed, but get within a few mm.

Then per-file: `standoff` in `servo_frame_mount.scad` (tube centre to servo
face) and `anchor_offset` in `lapbar_pushrod_clamp.scad` set your linkage
geometry. Work these out with the lever in neutral and check the full
fore/aft sweep before committing.

## Do not print the linkage

Print the **clamps and brackets**. Buy the **load path**:

- M6 threaded rod, cut to length
- M6 ball joints / heim ends, one per rod end (4 total)
- M6 clevis pins with R-clips at the lap-bar end
- Use the **metal servo arms supplied with the servo**. Never a printed horn.

Ball joints at both ends are not optional. They give the angular freedom that
lets the lever move without binding, and the clevis pin means you can unpin
the linkage in seconds to restore full manual PARK travel.

## Material and print settings

These parts hold a 165 kg·cm servo against a lever on a vibrating petrol
machine, outdoors.

- **PETG or ASA.** Not PLA — it creeps under sustained load and softens in a
  parked machine in the sun.
- **6+ perimeters**, 40–60% infill on clamps and the pushrod clamp.
- Print clamps with the **bore axis vertical** so layer lines don't split
  along the clamping force.
- The **electronics tray** is the one part with no structural role — print it
  in whatever you like, any orientation.

If a clamp shows any sign of creep or the bolt ears flex under tightening,
stop and have the equivalent cut from aluminium. Printed plastic is fine for
holding a servo in place; it is not fine as the thing standing between a
644 lb machine and an uncommanded lever movement.

## The bench check that gates everything

**Do the lap bars spring back to neutral when released?**

The whole bus-fault failsafe design assumes they do: cut servo power, servos
go limp, lever centering springs return the levers to neutral. The manual does
not state whether these levers self-centre.

If they don't, cutting servo power leaves the levers wherever they sat — a
runaway, not a failsafe — and the failsafe has to be redesigned around a
spring-return actuator or a fail-safe brake instead.

Check this by hand, on the machine, before building anything that depends on
it.

## Model caveat

The manual studied was **TimeCutter Max 50 in, form 3465-589**. You described
the machine as "TimeCutter 2", which is not a name Toro publishes — the 50"
line includes TimeCutter, TimeCutter MAX, and TimeCutter MyRIDE. Control
architecture is consistent across the line, so the findings above hold, but
frame and lever dimensions may not. Check the serial plate and pull the
matching manual before you cut metal.
