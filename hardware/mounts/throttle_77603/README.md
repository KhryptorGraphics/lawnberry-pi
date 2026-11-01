# 77603 throttle station — printed parts, assembly and adjustment

**PROTOTYPE — FIT / LOAD / ELECTRICAL VALIDATION REQUIRED.**
**Geometry generated, not physically qualified.** The published STL/3MF files, previews, views,
drawings and `generated_77603/validation.json` are regenerated from the current sources: all 57
catalogued pieces pass the mesh and print-envelope gate, and all 197 CSG probe runs (65 named
checks; linkage probes run for each variant A-E) match their expected result. A probe proves
geometric clearance or material presence at the demonstration pose only. Keeper retention,
joint strength and wear, assembly alignment on a real mower, and the linkage's mechanical
behaviour over the lever's real arc remain unverified.

Nothing in this directory is a measured fit for the Toro 77603. The steering geometry it references
is the repository's existing first-article datum, inherited from the 77502 baseline and **not**
verified against a 77603. See `parameters.scad` for every value's evidence class.

## What this directory is

CAD for the right-hand MG995 throttle station and its electronics sidecar. It changes ONE shared
steering datum: `servo_station_raise_mm` in `../enclosure_common.scad` raises **both** steering
servo stations by 101.6 mm (4 in), so the throttle servo fits below the right one. The legacy
heads keep their joint flange, rib, servo-face land, six slots and rear opening; only the rib's
height changes. The left and right steering pushrods are rebuilt from the same pin estimates
(the lap-bar clamp estimate stays at the measured datum, so each rod now falls about 6 degrees
from servo to lap bar). `hitch_arm_layer.scad`'s head rise reads the raised station; no other
legacy source is edited. The default (legacy) build therefore regenerates the `arm_*`,
`pushrod_*` and dependent legacy exports.

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
| C | Continuous-radius shoe in a captured, 45-degree dovetail track, clamped by a through-wedge that lifts it onto the undercut | `pip` (carrier + shoe), `wedge_key`, `wedge_keeper`, `cap`, `pin`, `keeper` |
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

The continuously adjustable slider, the receiver sleeve and its threaded collet nut are one
**print-in-place part**: `rod_capture_assembly`, three captive bodies printed **collet end down**
(finger tips, nut front face and slider tip all on the bed; 380 mm tall). The 45-degree
integral sleeve lug rides in the closed-ended slider groove. The groove ends are secondary
overtravel catches, not length stops: the **collet** holds the length.

**Print-in-place threaded collet (no bolts, no tools).** The sleeve's throttle end is a round
nose: an unslotted band with a 4 mm-pitch, 45-degree-flank external thread (root radius 18.6 mm,
leaving 3.04 mm of wall over the 22 mm bore's corners), then four 30 mm fingers slotted at the
bore's corners, whose outside flares 18 degrees toward the tip. The nut's inner cone sits over
the flare with a 0.45 mm gap. Turned by hand (scalloped 53 mm grip; right-hand thread, clockwise
viewed from the servo end tightens), the nut advances toward the tip: its cone meets the flare
after 1.46 mm and closes each finger's flat face onto a slider flat (0.4 mm clearance plus
0.3 mm preload) at 3.61 mm, i.e. 0.9 turn. The layout reserves 6 mm of advance (1.48 mm of
radial closure) with three thread pitches still engaged. The nut is captive: a 45-degree
shoulder stops it backing off, and it cannot pass the flare while the slider is inside.
**Unwinding resistance is only thread self-locking (1.9-degree lead) and collet preload
friction; there is no detent or ratchet, and vibration resistance is unqualified.** Clamp force,
slip load, finger fatigue and thread wear are likewise unmeasured.

**Repeatable setting.** V ticks 0.4 mm deep are engraved on the slider's +X flat for every 5 mm
of coarse setting from 940 to 1180 mm (9 mm long every 10 mm, 14 mm long every 50 mm). With the
nut loose (as printed, its face flush with the finger tips), read the tick at the index notch in
the +X finger's tip face, tighten, then trim. The slider is 380 mm long and keeps at least
65 mm inside the sleeve at the longest setting.

**Fine trim.** A separately printed throttle-end body (`rod_throttle_body`) sockets over the
slider tip behind the fork and is retained by the same transverse bar and keeper used at the
splices, through a window in the slider. Its boss journals the printed fine screw (flange
behind the boss, jam collar in front), and the eye fork (`rod_throttle_fork`) rides the screw on
its captive printed nut: the final 20 mm of the range, 2 mm per turn. Printed thread accuracy
and the fine stage's holding are not claimed. The coarse length path has no holes, notches or
detents; the ticks are reading marks only.

### Lock concepts compared

Three separately printed coarse locks were developed and compared (commit `0cd22aa`) and
**retired on 2026-11-01**, when the owner asked for a print-in-place, hand-tightened lock. The
printed lead-screw micro-fine stage is retained, moved from the slider onto the throttle-end
body; repeatable coarse setting now comes from the graduations.

| Concept | Continuous? | Print-in-place? | Separate parts | Holds by | Fine adjustment | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Wedge / over-centre cam + keeper | yes | no | body, wedge, cam, keeper | wedge friction on one slider flat | separate lead-screw stage | retired 2026-11-01 |
| Collar-tightened split collet | yes | no | body, collar, keeper | slotted shoes under a press-on collar | separate lead-screw stage | retired 2026-11-01 |
| Printed lead screw + anti-backdrive lip | yes | no | body, screw, nut, lip, jam collar | screw shoe on one slider flat; lip friction | separate lead-screw stage | retired 2026-11-01 |
| Threaded split collet (`rod_capture_assembly`) | yes | yes (sleeve, slider, nut) | none for the lock | four fingers squeezed by the nut's cone; thread self-locking | lead-screw stage on `rod_throttle_body` | **current** |

The catalog exports the segmented rod members, `rod_capture_assembly`, four splice bars and
keepers (three section joints plus the throttle-end body), the throttle-end body, the fine
screw, nut and jam collar, the throttle fork and the fit gauge. The slider, sleeve and nut are
deliberately not exported as independent pieces: the lug, groove and collet thread must be
printed together.

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
its V-profile stub pins run in blind teardrop bores. It pitches through the modelled ±30° CAD
range (probed, not physically tested). The yaw post, rod eye and cap are assembled separately;
the cap must be fitted after the eye is placed on the post. The U is a clearance capture, never
a clamp.

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
- The rod slider/sleeve/collet nut, C carrier/shoe and D carrier socket/ball are print-in-place
  sets. Do not split them into separate prints: their moving clearance and retention depend on
  co-printing. The rod set prints collet end down; its threads have 45-degree flanks and every
  gap is 0.45 mm perpendicular to the faces.
- Automated probes check U pitch through +/-30 degrees and six +/-1 mm translations; rod slider
  clearance at 940, 1066.8 and 1200 mm with overtravel contacts; the collet nut's as-printed
  clearance, +/-1 mm axial capture, free running along its thread and finger drive; the
  throttle-end body's fit, bar retention and clearance to the fully advanced nut; C
  sliding/capture at 12/18/30 mm radii plus both rail ends; and D articulation at +/-10 degrees
  with +/-3 mm translations on X, Y and Z. These prove modeled clearance/contact only, not
  load, clamp force or service life.
- The printed pins, keepers, wedge slots and as-printed fits still need coupons. PETG/ASA and
  slicer settings are assumptions, not qualified production tolerances.
- Printed threads (the collet's 4 mm-pitch, 45-degree thread and the fine stage's trapezoidal
  screw) are approximations for clamping and continuous trim only. They are not load-bearing
  fastener threads or substitutes for moulded/machined threads.
- Captive printed surfaces can wear, creep and crack. Service life remains unmeasured.
- Print orientation per piece is recorded in `hardware/mounts/hardware_77603_catalog.py`.
  The derivative head prints mating-face down; other pieces follow their catalogued orientation.

### Joints intentionally assembled after printing

Only four interfaces are exported as PIP sets: the U body/yoke, rod slider/sleeve/collet nut, C
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
  keeper. The throttle-end body (on the same bar and keeper), the fine screw, nut and jam collar
  and the fork also remain separate so the fine trim can be set and serviced.
- The MG995, keeper crossbar and keeper pins are assembled into the cradle walls that print as
  part of the derivative head; they are not part of a co-printed interface.

These are assembly choices for the current design, not a claim that no alternative PIP geometry
could exist. Do not merge these separate pieces into the four PIP exports.


## Assembly order

1. Print the derivative right head (`throttle_arm_head`: shelf, corbel and cradle walls are part
   of it), the regenerated `arm_head_left` (raised by the same 101.6 mm — a left head printed
   before this revision sits 4 in lower and must not be used), the keeper crossbar and the
   catalogued keeper pins/rings. Reprint both steering pushrods from the regenerated exports.
2. Fit the derivative head in place of `arm_head_right`, keeping the legacy joint, servo land,
   six slots and rear opening. Do not print both right heads. Tighten the joint's four M6
   bolts; the shelf leaves socket room above them.
3. Drop the MG995 into the cradle walls on the shelf, close the keeper crossbar over its ears and
   push both keeper pins through, then their split rings.
4. Bolt the steering servo to the raised land above it as before. The throttle servo, with its
   horn removed, and its keeper lift out under the steering servo, so neither servo has to come
   off to service the other (`servo_lifts_out`, `keeper_lifts_out`).
5. Choose one horn interface (A–E) and fit the matching carrier/retainers to the retained X horn.
6. Print `rod_capture_assembly` as one PIP set, collet end down. Join the fixed rod sections with
   their keyed spigots, transverse bars and keepers. Fit `rod_throttle_body` over the slider tip
   with its bar and keeper, fit the fine screw through the boss with its jam collar, and run the
   fork's nut onto the screw.
7. Fit the U fitting over the retained throttle lever. Place the rod eye over its yaw post and
   install the separate cap.
8. With the collet nut loose, set the length at the slider's ticks, tighten the nut by hand,
   trim with the fine screw, and verify return-to-idle by hand before considering powered
   testing.

## What is NOT established

- C's shoe is retained by the dovetail flanks and closed rail ends. The horn arm width, sliding
  contact and clearance below the carrier remain provisional and need measurement.
- C's lock is a printed through-wedge (1:6 taper, flat underside, 45-degree gable) driven
  across the rail through a matched slot in the shoe's neck. What the CSG probes prove at
  12/18/30 mm, with 0.05 mm modelled interference per face: driven home, the key bears on the
  rail's top face (`c_key_bears_rail`) and on the shoe slot's roof (`c_key_bears_shoe`), and
  the lifted shoe's ramps bear on the rail's 45-degree undercut (`c_shoe_clamped`); at mid-drive
  and home no overlap lies outside those three faces (`c_lock_bearing_only`); short of first
  contact the key touches only the seat it rests on (`c_key_entry_clear`); the keeper ring
  stops withdrawal after about 2.4 mm of slack (`c_keeper_*`). What is NOT proven: the wedge
  only creates normal force, so the radial hold is friction at those faces. Holding force and
  torque, preload retention under vibration (the 1:6 taper self-locks only if friction exceeds
  it), the keeper's 0.2 mm snap, and wear or creep of the printed bearing faces are
  unqualified. The keeper's slack relaxes the clamp before it stops the key.
- D's ball/socket clearance and ±10° CAD sweep are unqualified. The truncated ball/post is
  captive geometrically; friction, wear and print release still need a coupon and physical test.
- Rod groove ends are secondary overtravel catches, not stops at the declared length limits:
  the current dimensions leave about 5 mm of further extension at the 1200 mm setting, about
  245 mm of further extension at the 940 mm setting, and about 83 mm of reverse travel at the
  940 mm setting before shoulder contact. The collet must hold length; its clamp force, slip
  load, finger fatigue and retention under vibration are unqualified, and the nut's only
  unwinding resistance is thread self-locking plus preload friction (no detent).
- The exploded views separate the stack only. The C and D print-in-place pairs are rendered
  assembled, so their joint fit is evidenced by the CSG probes (`c_shoe_*`, `c_key_*`,
  `c_lock_*`, `d_ball_*`) and the `pip_gate.py` mesh report — not by the images.
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
obsolete profile-only exports are deleted at publish time. The default (no-argument) build
regenerates every legacy export in place; after the station raise it changes the `arm_*` heads
and both steering pushrods and must be rerun whenever `servo_station_raise_mm` changes.

**STEP export is unavailable**: OpenSCAD 2021.01 has no STEP writer and no FreeCAD/CadQuery is
installed. SCAD is the editable CAD deliverable; STL/3MF/PNG/SVG are generated. No STEP will be faked
by renaming a tessellated file.

## Revision note: where the throttle servo is carried (2026-11-01)

**The throttle MG995 stands BELOW the right steering servo; both steering servos are raised.**
The earlier revision carried the throttle servo on a column above the right steering servo.
Now `servo_station_raise_mm` (101.6 mm, `../enclosure_common.scad`) lifts BOTH steering
stations by the same amount, and the freed space under the right one holds the throttle stack.

`throttle_arm_head.scad` is the raised legacy right head — joint flange, upright rib, servo-face
land, six photo-derived slots, rear opening and spigot pocket preserved — plus:

- a **shelf** under the steering servo, reaching from the rib's outboard face to just past the
  cradle, whose top face is the servo's floor;
- a **45° corbel** from the rib to the shelf's outboard edge, narrower than the shelf, and 45°
  **wings** out to the cradle's length, so the head still prints mating-face down without support
  and a socket still reaches the joint's four M6 bolts (`joint_bolts_have_tool_room`);
- the **cradle walls**, printed as part of the head: there is no separate cradle, no seat dowels
  and no seat pin.

`mg995_throttle_mount.scad` supplies the cradle walls to the head and the separately printed
keeper crossbar with its two printed keeper pins and split rings. No purchased fastener is added.

The installed head is placed in the world with the legacy arm frame:

```scad
translate(arm_point(arm_joint_t)) throttle_arm_head();
```

The keeper stack and the MG995 reference body are placed from the mount's published world datums
(`throttle_servo_base_mm()` / `throttle_horn_base_mm()`), and the horn interfaces from
`throttle_horn_variants.scad`'s frame contract (horn frame on the shaft axis, z = 0 at the horn
underside), so the servo, its shaft, the retained horn and the cradle agree in every scene.

**Clearances proven on the built solids** (CSG probes, not arithmetic): the head, keeper stack and
servo body against the raised steering servo's case, moving bracket and printed adapter at
0/45/90° and its pushrod sweep (`mount_vs_steering`); every horn interface and rod servo section,
swept ±30°, against the same steering envelope (`linkage_vs_steering`) and against the head
(`linkage_vs_head`); each variant's whole throttle rod, servo eye to throttle eye, against that
envelope where the falling steering pushrod crosses above it (`rod_vs_steering`); and the servo
(horn off) and keeper lifting out from under the steering servo (`servo_lifts_out`,
`keeper_lifts_out`).

The steering envelope **samples three poses** (full forward, neutral, full reverse). At each one
the steering pushrod is moved rigidly by the crank pin's displacement, because the lap-bar clamp's
own arc is unmeasured; that drops the rod's far end by the crank's full 16 mm, which is meant to
be the cautious case for the parts below it. The poses are samples, not a bound on every real
position. Variant D's servo eye used to sit on the other side of the shaft; at full reverse its
collet nut then touched the steering pushrod about 90 mm short of the lap bar (0.03 mm gap), so
every variant's eye now sits on the same side.

**Structural path.** The load from the throttle servo runs through the cradle walls and shelf into
the corbel, through the derivative head's rib and joint flange, and finally into the mower
chassis through the legacy arm joint. **None of that path is load-qualified.** The shelf and
corbel are geometry-driven, not derived from a measured throttle load; no load, fatigue, creep or
vibration figure exists for the printed head, the printed pins or the printed keepers. Raising
the steering servos lengthens the heads' lever arm on the arm joint by 101.6 mm; that joint's
capacity is likewise unqualified. Treat the whole station as a fit prototype.

**Unverified by design:** whether the real MG995 horn and the real mower throttle lever can be
connected over the illustrative span without the linkage fouling anything. The lever pivot, its
arc, its actual cross-section and its lean angle remain TO MEASURE; the linkage is modelled
against an illustrative lever and an illustrative horn angle sweep. The assembly scenes derive
that illustrative pose rather than measure it: the throttle-side clevis pin sits at the declared
default span (`rods_nominal_span_mm`) from the station axis at the owner's idle-pose X, the horn
is phased so each variant's rod eye lies on that run (on the same side of the shaft for every
variant), and each rod is drawn at the pin-to-pin span its own eye requires (inside the declared
940–1200 mm band). The resulting lever Y and height are DERIVED from those illustrative numbers,
not measured.

## Probe discipline

`hardware/mounts/probe_77603.py` treats **any openscad WARNING as a probe failure**. This exists
because three separate defects in this build (a 1 mm³ keeper from an undefined variable, every
rod piece rendering the whole 1 m assembly from a wrong `-D` key, and a vacuum-clean
`mount_vs_steering` "pass" with 255 undefined-variable warnings) all produced plausible-looking
but meaningless output. A probe that renders empty *because variables are undefined* is a false
pass, not evidence.

The U fitting's earlier pin-path probes (`body_bore`, `neck_lower`, `mating`, `lower_pin`) were
removed with the printed neck and pin they measured: the U is now one print-in-place piece, so
its yoke-to-body fit is covered by the `u_yoke_fit`, `u_yoke_pitch_range` and `u_yoke_captive`
probes plus the `pip_gate.py` mesh gap. The rod's `rod_capture_pin`/`rod_capture_keeper` probes
were replaced the same way, by `rod_capture_pip_fit` and the `rod_overlap_capture*` pair. The
retired coarse locks' `lock_wedge_grip`, `lock_collet_grip` and `lock_screw_engagement` probes
went with their parts; the print-in-place collet is covered by `rod_collet_nut_clear`,
`rod_collet_nut_captive`, `rod_collet_nut_runs_free`, `rod_collet_nut_drives_fingers` and the
mesh gate, and the throttle-end body by the `rod_body_*` and `rod_fine_screw_engages` probes.

Each probe run is capped (300 s in `probe_77603.py`, 600 s in the build) so a hung render fails
instead of stalling. `rod_vs_steering` is the exception: it builds the whole threaded rod and
three full steering pushrods, about 11 minutes per variant here (643–655 s with all five running
at once), so its catalog entry sets its own `timeout_s`. A convex hull of the rod was tried as a
cheaper stand-in and rejected: it cost as much, and its corners met a steering pushrod that the
real rod clears. The build runs probe runs concurrently and keeps their results in catalog order.
