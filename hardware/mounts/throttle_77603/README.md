# 77603 throttle station — printed parts, assembly and adjustment

**PROTOTYPE — FIT / LOAD / ELECTRICAL VALIDATION REQUIRED.**
**Geometry generated, not physically qualified.** The published STL/3MF files, previews, views,
drawings and `generated_77603/validation.json` are regenerated from the current sources: all 69
catalogued pieces pass the mesh and print-envelope gate, and all 116 CSG probe runs (50 named
checks; each linkage probe runs once per variant A-E) match their expected result. A probe proves
geometric clearance or material presence at the demonstration pose only. Keeper retention,
joint strength and wear, assembly alignment on a real mower, and the linkage's mechanical
behaviour over the lever's real arc remain unverified.

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

| Variant | Mechanism | Printed pieces (catalog selectors) |
| --- | --- | --- |
| A | Direct pivot post through an existing X-horn arm hole | `post`, `cap`, `keeper` |
| B | Printed X reinforcement shell spreading reaction over all four horn arms | `shell_a`, `shell_b`, `lug`, `cap`, `pin`, `keeper` |
| C | Continuous-radius shoe in a captured, 45-degree dovetail track | `pip` (carrier + shoe), `wedge_key`, `wedge_keeper`, `cap`, `pin`, `keeper` |
| D | Truncated ball/post captive in a carrier-integral spherical socket | `pip` (carrier socket + ball/post), `cap`, `pin`, `keeper` |
| E | Double-shear clevis with replaceable bushings and a captive printed pin | `carrier`, `clevis`, `bushing`, `eye_bushing`, `pin`, `keeper`, `index_pin`, `index_keeper` |

All five share the same mount, the adjustable rod and the throttle-end fitting, so their
comparison is meaningful. **Do not select one before review.**

## Continuous length adjustment

**The rod length is an illustrative assembly span, not a measured mower dimension.** The modelled
range is **940–1200 mm** with a default of **1066.8 mm**, expressed as the servo-side pin centre
to the throttle-side pin centre. The owner's 40 in / 44 in separations and the 38–41 in /
43–46 in adjustment bands have **undefined datums**: they are not a rod length, a straight-line
stroke or a servo travel. The source rejects lengths outside the declared range.

The fixed member is segmented so every printed segment stays inside the bed. Keyed spigots join
the sections; a separately printed transverse bar and keeper retain each splice.

The continuously adjustable slider and receiver sleeve are one **print-in-place part**:
`rod_capture_assembly`. Its two bodies print upright together from z = 0. The 45-degree
integral sleeve lug rides in the closed-ended slider groove; the bodies have a mesh-checked gap.
The groove spans the declared 940–1200 mm settings and provides secondary anti-separation stops
after overtravel, not hard limits at the selected spans; the chosen coarse lock must hold length.
No capture pin or keeper is installed afterward. The profile retains 240 mm of coarse travel plus
the separate 20 mm fine stage; load capacity and wear life remain unqualified.

Three interchangeable printed coarse locks remain available:

1. `wedge_cam` — tapered wedge and over-centre cam with a separate keeper.
2. `split_collet` — slotted shoes, compression collar and separate keeper.
3. `lead_screw` — printed screw/nut with a smooth anti-backdrive lip.

**These three locks are alternatives: fit one family, not all three.** The fine stage uses a
printed screw/nut and jam collar. Printed thread accuracy and anti-backdrive life are not claimed.
The length path has no indexed holes, notches or detents.

The catalog exports the segmented rod members, `rod_capture_assembly`, the splice wedges and
keepers, the fine-stage parts, the throttle fork, the fit gauge, and each alternative lock's
components. The slider and sleeve are deliberately not exported as independent pieces: the
capture lug and groove must be printed together.

Both rod ends are horizontal eye plates with vertical bores. The servo eye is integral to the
lower rod section; variants A–D use a printed post and a separate cap, while variant E holds the
eye in double shear. The throttle eye is integral to `rod_throttle_fork` and fits over the U
fitting's yaw post, retained by a separate cap. The `linkage_*` probes check clearance and
1 mm eye lift/pull at each variant's demonstration pose.

## Throttle-end fitting

Open-bottom inverted box with the owner's **12.70 mm front/back × 31.75 mm left/right** inside
opening. The whole cavity is oriented by lean/yaw/pitch; the lean angle is unmeasured, so the
default is a demonstration pose, not a claim that the lever is vertical.

The body and pitch yoke are a two-body print-in-place pair: the yoke legs stand on the bed, and
its V-profile stub pins run in blind teardrop bores. It pitches through the tested ±30° CAD range.
The yaw post, rod eye and cap are assembled separately; the cap must be fitted after the eye is
placed on the post. The U is a clearance capture, never a clamp.

Print `throttle_u_gauge` first to check the opening. Coupon testing is still required to confirm
the printed joints release and move.

## Print envelope

Machine 420 × 420 × 500 mm; the build enforces a **400 × 400 × 490 mm** usable envelope (10 mm
reserve per XY edge, 10 mm Z). Long members are segmented with keyed spigots and printed cross-wedges
so no piece exceeds 380 mm. Each piece prints bed-zero in its stated orientation.

Material: PETG for fit coupons and unpowered geometry. ASA is preferred for the enclosure **only if**
the printer's nozzle/bed/enclosure are qualified; the stated 260 °C / 95 °C limits do not by themselves
qualify ASA. PLA is excluded for engine-bay service. Wall count, infill and layer height are
**untested print parameters** — record the actual slicer settings per piece. No G-code is produced here.

### Manufacturing notes

- All PIP parts are single exports with multiple closed bodies. `pip_gate.py` checks body count,
  bed contact, a measured minimum gap and droop-risk overhangs; those mesh checks do not prove a
  particular printer will release the joint.
- The rod slider/sleeve, C carrier/shoe and D carrier socket/ball are print-in-place pairs. Do not
  split them into separate prints: their moving clearance and retention depend on co-printing.
- Automated probes check U pitch through +/-30 degrees and six +/-1 mm translations; rod slider
  clearance at 940, 1066.8 and 1200 mm with overtravel contacts; C sliding/capture at 12/18/30
  mm radii plus both rail ends; and D articulation at +/-10 degrees with +/-3 mm translations
  on X, Y and Z. These prove modeled clearance/contact only, not load or service life.
- The printed pins, keepers, wedge slots and as-printed fits still need coupons. PETG/ASA and
  slicer settings are assumptions, not qualified production tolerances.
- Printed threads are coarse trapezoidal approximations for continuous trim only. They are not
  load-bearing fastener threads or substitutes for moulded/machined threads.
- Captive printed surfaces can wear, creep and crack. Service life remains unmeasured.
- Print orientation per piece is recorded in `hardware/mounts/hardware_77603_catalog.py`.
  The derivative head prints mating-face down; other pieces follow their catalogued orientation.

### Joints intentionally assembled after printing

Only four interfaces are exported as PIP pairs: the U body/yoke, rod slider/sleeve, C
carrier/shoe, and D carrier/socket/ball. The following joints remain separate by design because
they fit around retained mower/servo parts, require a post-print retainer, or must remain
adjustable and serviceable:

- The U yaw joint: place the rod eye over the yoke post, then fit `throttle_u_cap`. The PIP yoke
  pivot is independent of this eye retainer.
- Horn interfaces A–D: place each post or carrier on the retained X horn; fit the catalogued
  index pins/keepers and rod-eye cap where that variant uses them. C also uses a separate
  wedge key and keeper to lock the continuous shoe after setting its radius.
- Variant E: install the carrier and bushings around the horn/rod eye, then insert the hinge pin
  and keeper. These are separate service parts in the double-shear joint.
- Rod section splices: butt the keyed sections together, insert each transverse bar, then fit its
  keeper. The coarse lock parts and fine screw, nut and jam collar also remain separate so the
  span can be adjusted and serviced.
- The cradle, MG995, keeper crossbar and locating/retaining pins are assembled around the
  retained servo and derivative head; they are not part of a co-printed interface.

These are assembly choices for the current design, not a claim that no alternative PIP geometry
could exist. Do not merge these separate pieces into the four PIP exports.


## Assembly order

1. Print the derivative head, cradle, keeper crossbar and the catalogued printed pins/keepers.
   The column carrying the cradle is integral to the derivative head; there is no separate foot
   or riser.
2. Fit the derivative head in place of `arm_head_right`, keeping the legacy joint, servo land,
   six slots and rear opening unchanged. Do not print both heads.
3. Drop the cradle onto the head's column seat. The dowels locate it but do not lock it against
   lift-off; that remains unverified.
4. Seat the MG995 and close the keeper crossbar. Check the horn clears the window through its
   full sweep.
5. Choose one horn interface (A–E) and fit the matching carrier/retainers to the retained X horn.
6. Print `rod_capture_assembly` as one PIP pair. Join the fixed rod sections with their keyed
   spigots, transverse bars and keepers; install one coarse lock family and the fine-stage parts.
7. Fit the U fitting over the retained throttle lever. Place the rod eye over its yaw post and
   install the separate cap.
8. Set the continuous length, check the selected lock, and verify return-to-idle by hand before
   considering powered testing.

## What is NOT established

- C's shoe is retained by the dovetail flanks and closed rail ends. The horn arm width, sliding
  contact and clearance below the carrier remain provisional and need measurement.
- D's ball/socket clearance and ±10° CAD sweep are unqualified. The truncated ball/post is
  captive geometrically; friction, wear and print release still need a coupon and physical test.
- Rod groove ends are secondary overtravel catches, not stops at the declared length limits:
  the current dimensions leave about 37 mm of further extension at 1200 mm and 53 mm of
  reverse travel at 940 mm before shoulder contact. The selected coarse lock must hold length;
  its force capacity and retention under vibration are unqualified.
- No load, creep, fatigue, heat, UV, water or vibration qualification exists.
- A valid mesh and a clean hand fit prove geometry only.
- The printed throttle actuator is a mechanism proposal. Safe service needs the independent
  engine kill, OEM interlocks and a mechanical return-to-idle path that does not depend on printed
  linkage integrity. None is established by this CAD.

## Regenerating artifacts

```sh
PYTHONPATH="$PWD" python hardware/mounts/build_printables.py --profile hardware_77603
PYTHONPATH="$PWD" python hardware/mounts/preflight_77603.py      # diagnose all pieces in one pass
```

The profile publishes `<part>.stl`, `3mf/<part>.3mf`, `prev_<part>.png` and `view_<scene>.png`
into `hardware/mounts/` beside the legacy exports, so one slicer sees one part list; only
`generated_77603/drawings/` and `generated_77603/validation.json` stay in the generated tree, and
obsolete profile-only exports are deleted at publish time. The default (no-argument) build still
regenerates every legacy export in place and must be left unchanged.

**STEP export is unavailable**: OpenSCAD 2021.01 has no STEP writer and no FreeCAD/CadQuery is
installed. SCAD is the editable CAD deliverable; STL/3MF/PNG/SVG are generated. No STEP will be faked
by renaming a tessellated file.

## Revision note: where the throttle servo is carried (2026-10-30)

**The throttle MG995 is carried by the derivative right arm head, not by a bracket beside it.**
`throttle_arm_head.scad` derives the legacy right head: the joint flange, upright rib, servo-face
land, six photo-derived slots, rear opening and spigot pocket are preserved, and **one integral
printed support** is added in the free Y band beside the steering-servo case. The separate
`bracket_foot` is gone; there is no duplicate head and no second foot.

`mg995_throttle_mount.scad` now supplies only the independently printable stack that sits on that
support: the cradle and the keeper crossbar, together with the printed pins and printed secondary
keepers. There is **no riser and no outboard cantilever**: the earlier riser-stack revision is
superseded, and the cradle sits directly on the column's seat face, located by two printed dowels
in blind holes (a locating fit, **not** a lock against vertical lift-off, which stays UNVERIFIED
and is a bench-test item). The stack is joined by printed hardware
only — **no new purchased fastener** is specified, and no steering fastener, hole pattern or
datum is changed.

The installed head is placed in the world with the legacy arm frame:

```scad
translate(arm_point(arm_joint_t)) throttle_arm_head();
```

The cradle/keeper stack and the MG995 reference body are placed from the mount's published world
datums (`throttle_mount_stack_origin_mm()` / `throttle_servo_base_mm()` / `throttle_horn_base_mm()`),
and the horn interfaces from `throttle_horn_variants.scad`'s frame contract (horn frame on the
shaft axis, z = 0 at the horn underside), so the servo, its shaft, the retained horn and the
printed cradle agree in every scene.

**Structural path.** The load from the throttle servo runs through the printed cradle into the
integral column, through the derivative head's rib and joint flange, and finally into
the mower chassis through the legacy arm joint. **None of that path is load-qualified.** The
head's added section is geometry-driven, not derived from a measured throttle load; no load,
fatigue, creep or vibration figure exists for the printed head, the printed pins or the printed
keepers. Treat the whole station as a fit prototype.

**Unverified by design:** whether the real MG995 horn and the real mower throttle lever can be
connected over the illustrative span without the linkage fouling anything. The lever pivot, its
arc, its actual cross-section and its lean angle remain TO MEASURE; the linkage is modelled
against an illustrative lever and an illustrative horn angle sweep. The assembly scenes derive
that illustrative pose rather than measure it: the throttle-side clevis pin sits at the declared
default span (`rods_nominal_span_mm`) from the station axis at the owner's idle-pose X, the horn
is phased so each variant's rod eye lies on that run, and each rod is drawn at the pin-to-pin span
its own eye requires (inside the declared 940–1200 mm band). The resulting lever Y and height are
DERIVED from those illustrative numbers, not measured.

## Probe discipline

`hardware/mounts/probe_77603.py` treats **any openscad WARNING as a probe failure**. This exists
because three separate defects in this build (a 1 mm³ keeper from an undefined variable, every
rod piece rendering the whole 1 m assembly from a wrong `-D` key, and a vacuum-clean
`mount_vs_steering` "pass" with 255 undefined-variable warnings) all produced plausible-looking
but meaningless output. A probe that renders empty *because variables are undefined* is a false
pass, not evidence.
