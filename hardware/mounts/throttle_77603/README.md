# 77603 throttle station — printed parts, assembly and adjustment

**PROTOTYPE — FIT / LOAD / ELECTRICAL VALIDATION REQUIRED.**
**Geometry generated, not physically qualified.** The published STL/3MF files, previews, views,
drawings and `generated_77603/validation.json` are regenerated from the current sources: all 99
catalogued pieces pass the mesh and print-envelope gate, and all 65 CSG probe runs (37 probes;
each linkage probe runs once per variant A-E) match their expected result. A probe proves
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
| C | Continuously variable effective horn radius on a captive radial carriage | `carrier`, `shoe`, `wedge_key`, `wedge_keeper`, `cap`, `pin`, `keeper` |
| D | Printed ball with a replaceable split wear socket in a structural cage | `carrier`, `ball_stem`, `cage`, `wear_socket`, `closure`, `cap`, `pin`, `keeper` |
| E | Double-shear clevis with replaceable bushings and a captive printed pin | `carrier`, `clevis`, `bushing`, `eye_bushing`, `pin`, `keeper`, `index_pin`, `index_keeper` |

All five share the same mount, the same adjustable rod and the same throttle-end fitting, so their
comparison is meaningful. **Do not select one before review.**

## Continuous length adjustment

**The rod length is an illustrative assembly span, not a measured mower dimension.** The modelled
range is **940–1200 mm** with a default of **1066.8 mm**, expressed as the servo-side clevis pin
centre to the throttle-side clevis pin centre. The owner's 40 in / 44 in separations and the
38–41 in / 43–46 in adjustment bands have **undefined datums**: they are not a rod length, not a
straight-line stroke and not a servo travel. The source rejects lengths outside the declared
range rather than silently clamping them.

The member is **segmented** so every printed segment stays inside the bed: keyed spigots join
fixed-length tube sections, and a printed splice wedge plus a printed secondary keeper retains
each splice. The adjustable portion is a continuously sliding coarse carriage plus a separate
fine-trim stage. There are **no holes, notches, detents or indexed lengths** anywhere in the
adjustment path.

Coarse stage: a keyed slider in a square sleeve with **no index holes and no detents**. Three
interchangeable printed locks are modelled and exported:

1. `wedge_cam` — captured tapered wedge, over-centre cam, separate keeper.
2. `split_collet` — replaceable split shoe, large printed threaded collar, anti-unwind keeper.
3. `lead_screw` — printed fine-pitch screw/travelling nut with a smooth anti-backdrive lip.

**These three are ALTERNATIVES: fit ONE family, not all three.** The catalog exports all of them so
each concept is independently manufacturable and so the builder can choose; the build notes must
record which family was fitted.

Fine stage: a printed screw/nut gives fractional-turn trim. Printed thread accuracy is **not**
claimed. Every configuration has an internal end-stop collar so the members cannot separate even
with the lock fully released. That prevents separation, **not** unwanted throttle motion — report that
distinction honestly.

Every distinct printed part the rod composes is exported by name in
`hardware/mounts/hardware_77603_catalog.py`: the three chain sections (`rod_segment_lower`,
`rod_segment_middle`, `rod_segment_upper`), `rod_slider`, `rod_sleeve`, the capture hardware
(`rod_capture_pin`, `rod_capture_keeper`), `rod_spigot`, the splice wedge and keeper, the fine
stage (`rod_fine_screw`, `rod_fine_carriage_nut`, `rod_fine_jam_collar`), the throttle end
(`rod_throttle_fork`), the anti-separation end stop (`rod_endstop`, `rod_overstop_lug`,
`rod_overstop_stop`) and the rod's own clevis joint (`rod_clevis`, `rod_clevis_bushing`,
`rod_clevis_pin`). The rod's `endstop` and the shared `endstop_collar` are **different parts** and
both are exported. The three coarse locks each export with their own keeper
(`lock_wedge_keeper`, `lock_collet_keeper`, `lock_anti_backdrive`).

Both rod ends are **horizontal eye plates** (4 mm thick, 6.6 mm cut bore, vertical axis) in one
plane; the 30 mm tube is cranked `rod_eye_crank_mm` above that plane so it clears the servo,
keeper and horn hardware. The servo eye is integral to `rod_segment_lower`; variants A-D drop it
over a printed pivot post and trap it with a printed cap, and variant E holds it in double shear.
The throttle eye is integral to `rod_throttle_fork` and drops over the U neck's pivot post,
retained by `throttle_u_cap`. The `linkage_*` probes in `fit_checks.scad` check clearance and
1 mm lift/pull capture for every variant at its demonstration pose only.

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

### Manufacturing notes

- Every new linkage part is intended to be **printed as modelled**. Nothing in this directory is
  meant to be drilled, reamed, tapped, heat-staked or otherwise machined, and no print is meant to
  be finished to a press fit.
- The printed pins, keepers, wedge slots and capture lands are **as-printed fits**. Print the
  fit coupons (`mg995_lug_gauge`, `throttle_u_gauge`, `rod_fit_gauge`) before committing to the
  full-size stack; the nominal FDM allowances in `parameters.scad` are placeholders, not
  qualified production tolerances.
- Printed threads are a coarse trapezoidal approximation for continuous trim only. They are **not**
  a load-bearing fastener thread and are not a substitute for a moulded or machined thread.
- Captive printed features can wear, creep and crack. The retainer parts are designed to be
  replaced, but their service life is **unmeasured**.
- Print orientation per piece is recorded in `hardware/mounts/hardware_77603_catalog.py`. The
  derivative head prints mating-face down like the legacy head; the cradle, keeper,
  pins and keepers print in their catalogued orientations so the load paths stay away from
  layer-separation planes.

## Assembly order

1. Print `throttle_arm_head` (the derivative head), the cradle, the keeper
   crossbar and the printed pins / secondary keepers. There is **no separate bracket foot**,
   **no riser** and **no new purchased fastener**: the column that seats the cradle is integral to
   the derivative head, in the free Y band beside the steering-servo case.
2. Fit the derivative head in place of `arm_head_right`, keeping the legacy joint, servo land,
   six slots and rear opening unchanged. Do not print both heads.
3. Drop the cradle onto the head's column seat: its floor rests on the seat face and its two
   printed dowels plug into the blind holes in that face. The dowels locate the cradle; they do
   **not** lock it, so vertical lift-off under vibration is **UNVERIFIED** and is a bench-test
   item.
4. Seat the MG995 in the cradle recess, close the keeper crossbar (its legs bear on the servo's
   mounting ears) and confirm the horn clears the horn window through its full sweep.
5. Choose a horn interface (A–E), fit it to the retained X horn without enlarging any horn hole.
6. Assemble the rod segments with keyed spigots, printed splice wedges and printed splice keepers;
   insert the slider.
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
