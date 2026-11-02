# 77603 throttle station — printed parts, assembly and adjustment

**PROTOTYPE — FIT / LOAD / ELECTRICAL VALIDATION REQUIRED.**
**Geometry generated, not physically qualified.** The published STL/3MF files, previews, views,
drawings and `generated_77603/validation.json` are regenerated from the current sources: all 52
catalogued pieces pass the mesh and print-envelope gate, and all 444 CSG probe runs (187 named
checks across the station, rod, electronics and antenna probe files; linkage probes run for each
variant A-E) match their expected result. A probe proves
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

**No purchased fasteners and no loose parts.** The rod is four prints with nine bodies, every
moving part printed in place. The lever and thumbwheel are shared primitives in
`../pip_rod_joints.scad`, also used by the steering pushrods.

| Print (catalog piece) | Bodies | Height as printed | Contents |
| --- | --- | --- | --- |
| `rod_segment_lower` | 2 | 336 mm, socket mouth down | servo eye, 30 mm tube, socket boss, joint lever |
| `rod_segment_middle` | 2 | 303.8 mm, socket mouth down | spigot, tube, socket boss, joint lever |
| `rod_capture_assembly` | 3 | 360 mm, sleeve tip down | sleeve (spigot, tube, cam housing, overtravel lug), slider, cam lever |
| `rod_fork_assembly` | 2 | 48 mm, wheel down | throttle-eye fork, thumbwheel |

**The lever (used three times).** A rigid lever 12 mm wide turns 90° on two D-section pins
(Ø6, flat top and bottom, 45° chamfers) that stand inward from two cheeks on the housing, in a
round Ø6.8 bore through the lever: every gap is 0.4 mm and no overhang lies within 1 mm of the
other body, so the pair prints apart. As printed the arm sticks out radially and lies on the bed.
Closing it, the lobe reaches through a window in the housing wall: its corner passes the clamped
member about 15° before closed, then a flat lands on it. The flat takes up the member's side play
and the pivot's 0.4 mm radial play and then overlaps the member by **0.3 mm (preload)**; the
corner reaches **0.3 mm further (the over-centre bump)**, so the closed lever sits in a valley
with its arm on the housing face and needs force to open. The pins leave the lever about 1 mm of
play along the rod; the clamp reaction is radial and bears on the pins' round sides.

**Coarse lock (friction, continuous).** The cam lever on the sleeve's tip (pivot 3 mm outside
the 30 mm tube, 8.8 mm behind the tip face; 32 mm arm) clamps the 21.2 mm square slider anywhere
in its 240 mm travel. Graduations: V ticks 0.4 mm deep on the slider's +X flat every 5 mm of
setting from 940 to 1180 mm (9 mm long every 10 mm, 14 mm every 50 mm), read at the index notch
in the sleeve's +X tip face. Open the cam, slide to the tick, close the cam. The slider is 360 mm
(325 mm square, 35 mm threaded tip) and keeps 84 mm inside the sleeve at 1200 mm. An integral
45° lug in the slider's closed -X groove is an **overtravel catch only**: about 8 mm of further
extension at 1200 mm, about 248 mm at 940 mm, and 38 mm of retraction at 940 mm before contact.

**Section joints (positive).** Each socket boss (42 mm square, 0.6 mm side play around the 34 mm
square spigot, 45° seat pyramid) carries a joint lever. Closed, its lobe clamps the spigot and a
4.5 mm tooth on its arm, an arc on a 20 mm radius about the pivot, drops through an arc window in
the boss wall into a 2.8 mm pocket in the spigot. The square keys the joint against rotation.
Because the tooth's flanks and the pocket's walls are arcs about the pivot, the tooth swings in
without sweeping the pocket, and a withdrawal load is radial to the pivot: it cannot cam the
lever open. The load path is spigot → tooth → the window's wall in the boss; withdrawal is
blocked after about 0.85 mm (two 0.4 mm gaps), probed, not friction.

**Fine trim (thumbwheel, detented).** The slider's tip is a 4 mm-pitch, 45°-flank thread (crest
Ø21.2, 0.65 mm flats). The fork's Ø32 × 12 mm thumbwheel is printed captive between two posts
whose V ridges run in its rim groove, and screws along that thread carrying the fork and its
throttle eye. Two flexure pawls on the posts ride in 16 rim notches: **0.25 mm of eye travel per
click** over the **20 mm fine range (80 clicks)**; a pawl flexes about 0.63 mm per click. The
slider's thread is phased so that every 0.25 mm of fine extension is a detent seat. With its eye
on the U post, the fork cannot turn with the wheel.

**Holding target, unqualified.** Every lock should hold **185 N** axially: twice the ~92 N an
MG995 stalling at ~1.1 N·m (assignment estimate; datasheet value TO VERIFY) pushes through the
12 mm minimum horn radius. Nothing has been measured. Bench test before mower use:

1. `rod_cam_coupon` (3 bodies: 50 mm of the real cam housing, its lever and a plain square
   stub): close the cam on the stub and pull the stub with a spring scale to 185 N for 60 s;
   record slip with calipers. Repeat 10 open/close cycles, then pull to slip.
2. `rod_joint_coupon` (socket boss + lever) with `rod_joint_coupon_spigot`: insert, close, pull
   to 185 N and then to failure; record axial play and the failure mode (tooth shear, pocket
   wall, boss window).
3. On the full rod: count clicks over a caliper-measured 5 mm of fine travel (expect 20), and
   check the closed levers stay shut under engine vibration.

### Lock concepts compared

Every earlier lock and fine stage is retired. The current set (2026-11-01) answers the owner's
request for print-in-place locks with no bolts.

| Concept | Continuous? | Print-in-place? | Loose parts | Holds by | Status |
| --- | --- | --- | --- | --- | --- |
| Wedge / over-centre cam + keeper (commit `0cd22aa`) | yes | no | body, wedge, cam, keeper | wedge friction on one slider flat | retired |
| Collar-tightened split collet (commit `0cd22aa`) | yes | no | body, collar, keeper | slotted shoes under a press-on collar | retired |
| Printed lead screw + anti-backdrive lip (commit `0cd22aa`) | yes | no | body, screw, nut, lip, jam collar | screw shoe on one slider flat; lip friction | retired |
| Threaded split collet, `rod_capture_assembly` with captive nut (commit `d542562`) | yes | yes | none for the lock | fingers squeezed by the nut's cone; thread self-locking, no detent | retired 2026-11-01 |
| Splice bar + keeper at each section joint (commit `d542562`) | n/a | no | 4 bars, 4 keepers | bar through slots, clip keeper | retired 2026-11-01 |
| Lead-screw fine stage: throttle-end body, fine screw, carriage nut, jam collar, fork (commit `d542562`) | yes | no | 5 parts + bar/keeper | printed screw friction | retired 2026-11-01 |
| **PIP cam lever** on the sleeve (coarse) | yes | yes | none | over-centre flat on the slider (friction) | **current** |
| **PIP joint lever + arc tooth** at each joint | n/a | yes | none | tooth in a spigot pocket (positive) + lobe clamp | **current** |
| **PIP detented thumbwheel** on the fork (fine) | 0.25 mm steps | yes | none | thread + 16-notch pawl detent | **current** |

Both rod ends are horizontal eye plates with vertical 6.6 mm cut bores, 4 mm thick, in the
rod's x = 0 plane, the tube axis cranked 16 mm above it: the interfaces the horn variants and the
U fitting were built to. The servo eye is integral to `rod_segment_lower`; variants A–D use a
printed post and a separate cap, while variant E holds the eye in double shear. The throttle eye
is integral to the fork in `rod_fork_assembly`. The `linkage_*` probes check clearance and 1 mm
eye lift/pull at each variant's demonstration pose. The rod's probes live in `rod_checks.scad`.

## Throttle-end fitting

Open-bottom inverted box with the owner's **12.70 mm front/back × 31.75 mm left/right** inside
opening. The whole cavity is oriented by lean/yaw/pitch; the lean angle is unmeasured, so the
default is a demonstration pose, not a claim that the lever is vertical.

The body and pitch yoke are a two-body print-in-place pair: the yoke legs stand on the bed, and
its V-profile stub pins run in blind teardrop bores. It pitches through the modelled ±30° CAD
range (probed, not physically tested). The yaw post, rod eye and cap are assembled separately;
the cap must be fitted after the eye is placed on the post. The U is a clearance capture, never
a clamp.

The rod's throttle eye is the plate of the fork in `rod_fork_assembly`. It stands 8 mm plus the
fine extension ahead of the slider's tip and 42 mm ahead of the thumbwheel's servo face. Nothing
of the rod reaches above the eye (toward the cap) within 6 mm of the bore: the fork body ends
7 mm short of the bore's axis. The fork (Ø40.8) is cut flat at the eye plate's underside, so
nothing of the rod hangs below the eye toward the U. Fit the wheel and fork by screwing the
wheel onto the slider's thread, drop the eye over the post, fit the cap, then trim with the
wheel: the post stops the fork turning with it.

Print `throttle_u_gauge` first to check the opening. Coupon testing is still required to confirm
the printed joints release and move.

## Print envelope

Machine 420 × 420 × 500 mm; the build enforces a **400 × 400 × 490 mm** usable envelope (10 mm
reserve per XY edge, 10 mm Z). Long members are segmented with keyed spigots and print-in-place
joint levers so no throttle piece exceeds 380 mm. Each piece prints bed-zero in its stated orientation.

Material: PETG for fit coupons and unpowered geometry. ASA is preferred for the enclosure **only if**
the printer's nozzle/bed/enclosure are qualified; the stated 260 °C / 95 °C limits do not by themselves
qualify ASA. PLA is excluded for engine-bay service. Wall count, infill and layer height are
**untested print parameters** — record the actual slicer settings per piece. No G-code is produced here.

### Manufacturing notes

- All PIP parts are single exports with multiple closed bodies. `pip_gate.py` checks body count,
  bed contact, a measured minimum gap and droop-risk overhangs; those mesh checks do not prove a
  particular printer will release the joint.
- The rod's four sets (`rod_segment_lower` and `rod_segment_middle`, each tube + joint lever;
  `rod_capture_assembly`, sleeve + slider + over-centre cam lever; `rod_fork_assembly`, fork +
  detented thumbwheel), C carrier/shoe and D carrier socket/ball are print-in-place sets. Do not
  split them into separate prints: their moving clearance and retention depend on co-printing.
  Every rod set prints with its levers or wheel on the bed, and every gap is 0.4 mm. The
  levers print open; close them by hand.
- Automated probes check U pitch through +/-30 degrees and six +/-1 mm translations; rod slider
  clearance at 940, 1066.8 and 1200 mm with overtravel contacts; the cam lever's as-printed
  clearance, capture in six directions, swing, over-centre valley and clamp at all three
  settings; the thumbwheel's capture, detent click and seat, free running and thread drive at
  both fine limits; each section joint's capture, insertion clearance, clamp, tooth engagement,
  withdrawal and rotation stops, and the tooth bearing on the boss wall rather than the pins; C
  sliding/capture at 12/18/30 mm radii plus both rail ends; and D articulation at +/-10 degrees
  with +/-3 mm translations on X, Y and Z. These prove modeled clearance/contact only, not
  load, clamp force or service life.
- The printed pins, keepers, wedge slots and as-printed fits still need coupons;
  `rod_cam_coupon` and `rod_joint_coupon` (+ `rod_joint_coupon_spigot`) are the rod's bench
  coupons, pulled to its 185 N holding target. PETG/ASA and slicer settings are assumptions, not
  qualified production tolerances.
- The printed thread (the slider tip's 4 mm-pitch thread that the thumbwheel drives) is an
  approximation for continuous trim only. It is not a load-bearing fastener thread or a
  substitute for a moulded/machined thread.
- Captive printed surfaces can wear, creep and crack. Service life remains unmeasured.
- Print orientation per piece is recorded in `hardware/mounts/hardware_77603_catalog.py`.
  The derivative head prints mating-face down; other pieces follow their catalogued orientation.

### Joints intentionally assembled after printing

The PIP exports are the U body/yoke, the rod's four sets (two section levers, the capture set,
the fork and its wheel), C carrier/shoe, and D carrier/socket/ball. The following joints remain separate by design because
they fit around retained mower/servo parts, require a post-print retainer, or must remain
adjustable and serviceable:

- The U yaw joint: place the rod eye over the yoke post, then fit `throttle_u_cap`. The PIP yoke
  pivot is independent of this eye retainer.
- Horn interfaces A–D: place each post or carrier on the retained X horn; fit the catalogued
  index pins/keepers and rod-eye cap where that variant uses them. C also uses a separate
  wedge key and keeper to lock the continuous shoe after setting its radius.
- Variant E: install the carrier and bushings around the horn/rod eye, then insert the hinge pin
  and keeper. These are separate service parts in the double-shear joint.
- Rod section joints: push each spigot into the next socket with that socket's lever open, then
  close the lever. The fork screws onto the slider tip by its captive wheel. No bars, keepers or
  loose fine-stage parts remain.
- The MG995 and the one-piece snap keeper are assembled into the cradle walls that print as
  part of the derivative head; the keeper's four fingers snap into the walls' windows.

These are assembly choices for the current design, not a claim that no alternative PIP geometry
could exist. Do not merge these separate pieces into the PIP exports.


## Assembly order

1. Print the derivative right head (`throttle_arm_head`: shelf, corbel and cradle walls are part
   of it), the regenerated `arm_head_left` (raised by the same 101.6 mm — a left head printed
   before this revision sits 4 in lower and must not be used) and the snap keeper. Reprint both
   steering pushrods from the regenerated exports: they are print-in-place too
   (`pushrod_servo_end`, `pushrod_sleeve`, `pushrod_inner` per side).
2. Fit the derivative head in place of `arm_head_right`, keeping the legacy joint, servo land,
   six slots and rear opening. Do not print both right heads. Tighten the joint's four M6
   bolts; the shelf leaves socket room above them.
3. Drop the MG995 into the cradle walls on the shelf and press the snap keeper down over its ears
   until all four fingers click into the wall windows (pry the finger tails outward to remove
   it).
4. Bolt the steering servo to the raised land above it as before. The throttle servo, with its
   horn removed, and its keeper lift out under the steering servo, so neither servo has to come
   off to service the other (`servo_lifts_out`, `keeper_lifts_out`).
5. Choose one horn interface (A–E) and fit the matching carrier/retainers to the retained X horn.
6. Print the four rod sets in their catalogued orientations. Open each joint lever, push the
   spigot home and close the lever until its arm lies on the boss. Screw the fork's wheel onto
   the slider tip.
7. Fit the U fitting over the retained throttle lever. Place the rod eye over its yaw post and
   install the separate cap.
8. Open the cam, slide to the tick at the index notch and close the cam over centre. Trim with
   the fork's wheel at 0.25 mm per click (the eye on the U post keeps the fork from turning; off
   the post, hold the fork). Verify return-to-idle by hand before considering powered testing.

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
  the current dimensions leave about 8 mm of further extension at the 1200 mm setting, about
  248 mm of further extension at the 940 mm setting, and about 38 mm of reverse travel at the
  940 mm setting before the catch. The cam lock must hold length: lever clamp force, joint and
  cam holding (185 N design target per lock), detent torque and retention under engine
  vibration are unqualified until the coupons are pulled.
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

**STEP export is not produced** (re-checked 2026-11-02). OpenSCAD 2021.01 has no STEP writer.
FreeCAD 26.3 (conda-forge `freecad 2026.09.16`, aarch64) was installed and tried as the converter:
its OpenSCAD importer builds every `hull()` and `minkowski()` through OpenSCAD/CGAL meshes
(`importCSG.py`: `p_hull_action` and `p_minkowski_action` return `CGALFeatureObj`), so a part that
contains a hull comes back as a faceted solid, not editable B-rep. A CSG survey of all 127 published
parts found 51 with at least one `hull()` (the shared `plate()` and `slot()` in `common.scad` are
hulls); only 72 use nothing beyond primitives, booleans and `linear_extrude`. A STEP set for 72 of
127 parts behind a 3.5 GB FreeCAD dependency was not adopted. SCAD is the editable CAD deliverable;
STL/3MF/PNG/SVG are generated. No STEP will be faked by renaming a tessellated file.

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
- the **cradle walls**, printed as part of the head, each side wall with two snap **windows** in
  its outer face: there is no separate cradle, no seat dowels and no seat pin.

`mg995_throttle_mount.scad` supplies the cradle walls to the head and **one** separately printed
snap keeper (`servo_keeper`). Its frame bears on the MG995's ears; four 2 mm spring fingers drop
beside the side walls and their barbs snap into the windows under flat roofs (0.2 mm catch play
when the frame sits on the ears); two 0.4 mm crush ribs per ear row stand proud of the frame's
underside, so the keeper has to be pressed past them to engage and then keeps the ears pressed
onto the wall tops instead of letting the servo rattle in the play. To service it, pry the four
finger tails outward (a fingernail or small flat blade between tail and wall) and lift it off;
the MG995 then lifts straight out. It replaces the keeper crossbar, its two transverse printed
pins and their two split rings (five loose parts, now one). No purchased fastener is added.

**Bolts kept, and why.** The four M6 at the legacy arm joint (they carry the raised steering
servo's and the throttle station's loads into the arm, and the shelf leaves socket room for
them: `joint_bolts_have_tool_room`) and the purchased RDS51150 steering-servo bracket bolts
through the unchanged servo-face land (the steering servo is retained hardware). The throttle
station itself adds none.

The installed head is placed in the world with the legacy arm frame:

```scad
translate(arm_point(arm_joint_t)) throttle_arm_head();
```

The keeper and the MG995 reference body are placed from the mount's published world datums
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
`keeper_lifts_out`). The keeper's snap is probed on the head itself: seated it clears the walls
(`keeper_fits_cradle`), its ribs overlap the provisional ears (`keeper_ribs_preload_lugs`),
lifted 0.5 mm its barbs meet the window roofs (`keeper_snap_blocks_liftoff`), and with its
fingers opened 0.7 mm it slides down the walls at three stations without touching them
(`keeper_snap_fingers_pass`).

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
vibration figure exists for the printed head, the snap keeper or the variants' printed pins
and keepers. Raising the steering servos lengthens the heads' lever arm on the arm joint by
101.6 mm; that joint's capacity is likewise unqualified. Treat the whole station as a fit
prototype.

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
probes plus the `pip_gate.py` mesh gap. The rod and linkage probes live in
`rod_checks.scad` (each catalog `Probe` names its own source file). The collet-era
`rod_collet_nut_*`, `rod_body_*` and `rod_fine_screw_engages` probes and the splice-bar probes
went with their parts; the print-in-place rod is covered by `rod_cam_*`, `rod_wheel_*`,
`rod_joint_*`, `rod_span_*`, `rod_capture_pip_fit`, the `rod_overlap_capture*` pair and the mesh
gate.

Each probe run is capped (300 s in `probe_77603.py`, 600 s in the build) so a hung render fails
instead of stalling. `rod_vs_steering` and `linkage_vs_steering` are the exceptions: they build
the whole rod and three full steering pushrods per variant (about 1.5 and up to 6 minutes here
under load), so their catalog entries set their own `timeout_s`. A convex hull of the rod was tried as a
cheaper stand-in and rejected: it cost as much, and its corners met a steering pushrod that the
real rod clears. The build runs probe runs concurrently and keeps their results in catalog order.
