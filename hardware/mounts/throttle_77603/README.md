# 77603 throttle station — printed parts, assembly and adjustment

**PROTOTYPE — FIT / LOAD / ELECTRICAL VALIDATION REQUIRED.**
Nothing in this directory is a measured fit for the Toro 77603. The steering geometry it references
is the repository's existing first-article datum, inherited from the 77502 baseline and **not**
verified against a 77603. See `parameters.scad` for every value's evidence class.

## What this directory is

Additive CAD for the right-hand MG995 throttle station and its electronics sidecar. It does **not**
modify, re-export or re-print any legacy steering part: `hardware/mounts/{common,enclosure_common,
hitch_arm_layer,arm_support,pushrod,servo_pushrod_adapter,lapbar_four_bolt_clamp,throttle_servo_mount}.scad`
and every `arm_*`, `servo_adapter_*`, `pushrod_*`, `lapbar_clamp_*` export stay byte-identical.

Legacy `throttle_servo_mount.scad` is a six-hole large-steering-servo fixture and is **not** the
MG995 interface; it is retained unchanged for that older purpose.

## Include paths

The builder passes no `-I`, so:

- legacy sources are reached as `include <../common.scad>` / `include <../enclosure_common.scad>`;
- sibling files in this directory use a plain `include <name.scad>`.

Never copy a legacy `.scad` in here; that would silently fork the dimensional contract.

## Five linkage variants (no winner selected)

| Variant | Mechanism | Printed pieces |
| --- | --- | --- |
| A | Direct joint through an existing X-horn arm hole | flanged eye, captive pin, keeper |
| B | Printed X reinforcement shell spreading reaction over all four horn arms | two shell halves, lug, pin |
| C | Continuously variable effective horn radius on a captive radial carriage | carriage, shoe, wedge, pin |
| D | Printed ball with a replaceable split wear socket in a structural cage | ball stem, wear socket, cage, closure |
| E | Double-shear clevis with replaceable bushings and a captive printed pin | clevis, bushings, pin, keeper |

All five share the same mount, the same adjustable rod and the same throttle-end fitting, so their
comparison is meaningful. **Do not select one before review.**

## Continuous length adjustment

Coarse stage: a keyed slider in a square sleeve with **no index holes and no detents**. Three
interchangeable printed locks are modelled and exported:

1. `wedge_cam` — captured tapered wedge, over-centre cam, separate keeper.
2. `split_collet` — replaceable split shoe, large printed threaded collar, anti-unwind keeper.
3. `lead_screw` — printed fine-pitch screw/travelling nut with a smooth anti-backdrive lip.

Fine stage: a printed screw/nut gives fractional-turn trim. Printed thread accuracy is **not**
claimed. Every configuration has an internal end-stop collar so the members cannot separate even
with the lock fully released. That prevents separation, **not** unwanted throttle motion — report that
distinction honestly.

## Throttle-end fitting

Open-bottom inverted box. Internal opening is the owner-selected **12.70 mm front/back × 31.75 mm
left/right**, with refinement clearances defaulting to zero so the allowance is not doubled. The whole
cavity is oriented by `throttle_lean_right_deg` / yaw / pitch — the lean angle is **unmeasured**, so the
default is a demonstration pose, not a claim that the lever is vertical. A captured printed two-axis
neck joint absorbs lateral drift; the fitting is a clearance capture, never a clamp.

Print `throttle_u_gauge` first to confirm the opening before committing a full linkage.

## Print envelope

Machine 420 × 420 × 500 mm; the build enforces a **400 × 400 × 490 mm** usable envelope (10 mm
reserve per XY edge, 10 mm Z). Long members are segmented with keyed spigots and printed cross-wedges
so no piece exceeds 380 mm. Each piece prints bed-zero in its stated orientation.

Material: PETG for fit coupons and unpowered geometry. ASA is preferred for the enclosure **only if**
the printer's nozzle/bed/enclosure are qualified; the stated 260 °C / 95 °C limits do not by themselves
qualify ASA. PLA is excluded for engine-bay service. Wall count, infill and layer height are
**untested print parameters** — record the actual slicer settings per piece. No G-code is produced here.

## Assembly order

1. Print the saddle halves, flange hook, both riser segments, cradle and keeper.
2. Fit the saddle halves around the right stationary column, key the dovetails, and seat the ledge on
   the root-top shoulder with the flange hook under the outboard flange overhang.
3. Stack and key the riser segments; drop the cradle on the riser top.
4. Seat the MG995 in the cradle recess, close the keeper crossbar, and confirm the horn clears the
   horn window through its full sweep.
5. Choose a horn interface (A–E), fit it to the retained X horn without enlarging any horn hole.
6. Assemble the rod segments with keyed spigots and splice wedges; insert the slider.
7. Fit the chosen lock, then the fine-trim screw and jam collar, then the end-stop collar.
8. Fit the U fitting to the retained throttle lever and connect the neck joint.
9. Adjust length continuously, verify return-to-idle by hand, and only then consider powered testing.

## What is NOT established

- No lever cross-section, lean angle, engagement depth or travel is measured.
- The 40/44 in separations and 38–41 / 43–46 in bands have undefined datums; no 4 in stroke is assumed.
- No MG995 shaft XY, horn radius, lug thickness or hole diameter is measured.
- No load, creep, fatigue, heat, UV, water or vibration qualification exists.
- A valid mesh and a clean hand fit prove geometry only.
- The printed throttle actuator is a **mechanism proposal**. Safe service needs the independent engine
  kill, the OEM interlocks and a mechanical return-to-idle path that does not depend on printed-linkage
  integrity. None of those is established by this CAD.

## Regenerating artifacts

```sh
PYTHONPATH="$PWD" python hardware/mounts/build_printables.py --profile hardware_77603
PYTHONPATH="$PWD" python hardware/mounts/preflight_77603.py      # diagnose all pieces in one pass
```

The profile writes only `hardware/mounts/generated_77603/`; the default (no-argument) build still
regenerates every legacy export in place and must be left unchanged.

**STEP export is unavailable**: OpenSCAD 2021.01 has no STEP writer and no FreeCAD/CadQuery is
installed. SCAD is the editable CAD deliverable; STL/3MF/PNG/SVG are generated. No STEP will be faked
by renaming a tessellated file.

## Revision note: where the throttle servo is carried (2026-10-30)

Two structures were built and rejected during implementation, both by geometry probes rather
than by opinion. This is recorded because the change is structural, not cosmetic.

**Rejected: column-mounted saddle.** The first revision wrapped the lower right arm column
(`arm_root_x`, world Z −66…286) and rose 96–240 mm on a riser. It failed `mount_vs_steering`:
the throttle cradle landed inside the right steering servo's own case envelope
(X 211.6…273.0, Z 378.2…445.2). Lengthening the riser made it **worse**, because the column
rakes 12.58° outboard with height, driving the cradle further into the servo case in X.

**Adopted: extension of the existing right steering servo bracket.** This is what the
requirement actually asks for ("vertically extend and structurally reinforce existing right
steering servo bracket"). The printed bracket now originates **on the servo land at
`arm_tip_z`**, reaches inboard over the arm-head flange band in a Y range clear of the servo
case, and rises outboard to the MG995 cradle.

Clearances that drove the final numbers (all DERIVED from the legacy datum, not measured):

| Constraint | Value | Consequence |
| --- | --- | --- |
| Steering servo case | X 211.6…273.0, Y 108…138, Z 378.2…445.2 | tower must avoid this Y band and X band |
| Steering horn sweep | radius 55 mm about (Y 123, Z 430) | X beyond ~297 mm |
| Steering pushrod sweep | X 273.8…370.1, Z 465.2…505.2 | X beyond ~395 mm |
| Arm-head flange band | Y 88…158 (70 mm wide) | foot attaches at Y 141…157 |

Resulting placement: tower axis ≈186 mm outboard of the servo station, cradle ≈180 mm above
`arm_tip_z`, so the throttle servo sits clear of the case, the horn and the pushrod.

**Attachment datum, verified by solid intersection rather than arithmetic.** The foot's
inboard end was checked against the *actual* legacy `arm_head_assembly()` solid, not against a
computed coordinate, and the intersection is non-empty:

| Foot ∩ arm head | Range |
| --- | --- |
| X | 187.3 … 209.6 |
| Y | 141.0 … 157.0 (clear of the servo case at Y 108…138) |
| Z | 430.2 … 438.2 |

So the foot really rests on arm-head material at the servo-face height, in the Y band that
avoids the servo case. An earlier attempt landed on nothing: the arm-head *joint flange* is a
vertical plate at Z ≈ 280, so a horizontal foot at Z 430 could never touch it — a lesson worth
keeping, since the arithmetic looked plausible.

**This is a long cantilever and is not load-qualified.** The foot is gusseted and bolts through
the arm-head flange band, but the torque path into the mower chassis is still the unresolved
hitch/single-bolt question from the legacy design. Treat the bracket as a fit prototype.

**Unverified by design:** whether the real MG995 horn and the real mower throttle lever can be
connected over this span without the pushrod fouling anything. The lever pivot, its arc, its
actual cross-section and its lean angle remain TO MEASURE; the linkage is modelled against an
illustrative lever and an illustrative horn angle sweep.

## Probe discipline

`hardware/mounts/probe_77603.py` treats **any openscad WARNING as a probe failure**. This exists
because three separate defects in this build (a 1 mm³ keeper from an undefined variable, every
rod piece rendering the whole 1 m assembly from a wrong `-D` key, and a vacuum-clean
`mount_vs_steering` "pass" with 255 undefined-variable warnings) all produced plausible-looking
but meaningless output. A probe that renders empty *because variables are undefined* is a false
pass, not evidence.
