# Claude Code CAD implementation addendum — MG995 throttle actuator and segmented 3D printing
Date: 2026-10-30
Status: Design instructions; physical dimensions and fit require verification.
This document supplements the existing servo-throttle linkage specification on this branch. Do not redesign the mower or alter the existing steering drive and pushrod except for the right-side mount extension needed to carry the new throttle servo.

## Required servo envelope (user measured)
Orient servo on its base, output shaft up on the far side:
- Body: 1.50 in high × 0.50 in thick (left–right) × 1.50 in long (front–back).
- Output shaft protrudes approximately another 0.50 in above the housing; allow horn, fastener, linkage and dynamic clearance beyond this envelope.
- Mounting-hole centers along front–back direction: 2.00 in apart.
- Mounting-hole centerline height: 1.125 in above bottom of housing.
- Actual shaft XY location, hole diameter, mounting lug thickness and horn exact hole radii remain **TO MEASURE**; do not infer them from nominal MG995 specifications.
- Servo horn: existing four-arm X horn, attach pushrod with a secure, removable joint.

## New mandatory constraint: fully 3D-printed throttle linkage (2026-10-30)
The user explicitly requires **a complete durable, long-service-life 3D-printable solution**, rather than purchased metal ball joints or clevises. Supersedes all prior recommendations for purchased ball links, metal rods, clevis pins, bolts, and commercial joints as required linkage parts.
- The **existing MG995 servo, original X-shaped horn, its existing factory shaft/retainer, and the mower's existing throttle lever** are retained, not printed. Every **newly designed linkage component** must be modeled for 3D printing: horn adapter (if used), pushrod attachment, bearing/pivot, captive pin and retainer, replaceable bushing, segmented pushrod sections, sliding fine-adjustment sleeve and its clamp/lock, bracket, and connecting/alignment elements.
- The primary solution's bill of materials must not depend on purchased ball joints, threaded rods, screws, pins, bearings, fasteners or metal linkages. Explore printed captive hinge pins, clips, latches, split-clamp screws/nuts or printed locking wedges as appropriate. If an all-printed part cannot reasonably meet durability/safety requirements, **explicitly report that engineering limitation** and do not falsely declare the design safe, durable, or completed; request a user design decision rather than silently substitute purchased hardware.
- Prefer load-distributing, **captured** printed adapter interfacing with the existing X-shaped horn over relying on a fragile single-arm snap fit. It must avoid damaging the horn and maintain clearance to the shaft-retaining screw; physical horn hole dimensions and geometry must be verified before final fit.
- For wear-prone pivots, consider a **double-shear printed clevis with a replaceable printed bushing and captive printed hinge pin**, a captured printed ball-and-socket with positive anti-release cage (only if justified), and a split-shell printed horn adapter that uses the available horn geometry. Compare fatigue, abrasion, backlash, creep, retention, maintenance, and ease of printing. Never claim printed ball sockets are intrinsically more durable than captured pivots.
- Include replaceable sacrificial wear inserts/bushings, generous fillets and radii, appropriate wall thickness, avoidance of layer-separating stresses, captured geometry, printed anti-vibration locks, and access for inspection/replacement. Structural joints must not depend on glue-only adhesion or a snap feature with a single-point brittle failure. Design to resist UV/weather, engine heat, water, vibration, and repeated load cycles.
- Explicitly evaluate candidate filaments and actual machine capabilities before choosing one. PLA is not the default for an outdoor engine-bay component; evaluate PETG and ASA and any other printable engineering filament against heat, UV, creep, interlayer strength, dimensional tolerance and exposure. Do not specify nylon/CF as printable unless the nozzle temperature, bed temperature, nozzle wear resistance, enclosure and drying requirements are actually met. Record material/print orientation and layer-strength assumptions for each part.
- Require cycle and load testing before service: printed-joint retention and wear after repeated throttle sweeps, vibration resistance, elevated-temperature creep, sustained-load deformation, loss of connection, return-to-idle on loss of power, and manual override/engine stop. Define measurable pass/fail criteria from observed throttle loads and user safety requirements; **physical service life is unverified until tested**.
- The final mower must retain a separately validated independent engine-kill/emergency shutdown path, hard mechanical stops, and a throttle return-to-safe/idle mechanism that does not depend solely on printed-linkage integrity. If any failure mode can leave the throttle open, do not authorize live mower operation.

## Design alternatives — build all five, do not automatically select one
Build editable, parametric CAD variants with the same servo envelope and common mounting datum:
A. Direct connection to an existing X-horn arm.
B. Printed adapter attached to the existing X horn.
C. Continuously adjustable effective-radius horn adapter (slot/clamp, **not discrete indexing holes**).
D. **Fully printed, mechanically captured** ball-joint variant, with replaceable wear socket and positively retained ball (not a purchased ball link); reject if strength or retention fails validation.
E. **Fully printed, double-shear clevis** with captive printed pivot pin and replaceable printed bearing insert (no purchased clevis or metal pin).
Each variant must be independently manufacturable and show complete assembled and exploded views, material/hardware list, collision/kinematic checks, and tradeoffs. If two connection types can combine, keep comparisons meaningful and preserve separate model variants.

## Fine adjustment — non-negotiable
- Throttle pushrod usable length adjustment MUST be **continuous and micro-fine**, not selectable through rows of holes or detents.
- Use a fully printed, continuously sliding telescoping sleeve/rod with positive printed split-clamp/wedge/locking collar; optionally include a fully printed fine-pitch screw/nut that gives controlled continuous movement. Printed joints must have adequate capture, strength, and anti-vibration retention. No discrete-length adjustment holes.
- Provide anti-rotation or positive orientation control where required, a visible calibration mark / datum, retention against separation, and locking against vibration. Do not rely solely on friction for a safety-critical throttle without mechanical retention.
- Previously recorded positional measurements: lowest throttle 40 in, full-forward throttle 44 in, with provisional adjustment bands 38–41 in and 43–46 in. These are **unverified measurements of unspecified datums**, not a proven 4-in pushrod stroke. Document datum definitions and validate actual travel before selecting servo-horn radius and usable servo angle.
- The separate parts of a telescoping mechanism must have sufficient overlap and strength throughout the required adjustment range; verify load, buckling and joint retention rather than assume one long 3D-printed rod is acceptable. Use fully printable segmented pushrod members and printable structural couplers. Analyze slender-member buckling, bending, interlayer strength, segment-joint pullout, long-term creep and thermal distortion; if fully printed members cannot meet the measured loads and safety margin, report the limitation transparently.

## Explicit print-only telescopic pushrod lock (2026-10-30 revision)
This section supersedes any ambiguous earlier description of a clamp secured by a bolt. **The telescoping/sliding pushrod MUST lock in any continuously selected position using parts fabricated on the 3D printer. No purchased bolts, screws, nuts, threaded inserts, pins, or other hardware may secure or adjust the sliding lock.**

- Design and compare at least three *fully printable* continuous-length locking concepts, such as (1) a captive tapered wedge driven by a printed lever/cam and backed by a positive secondary retainer, (2) a printed split-collet/split-clamp with a large printed threaded collar and captive printed locking feature, and (3) a printed fine-pitch lead screw/nut combined with a sliding sleeve and a printed anti-backdrive lock. Alternatives are examples, not presumed safe designs.
- The pushrod must slide continuously for coarse adjustment and permit fine, repeatable micro-adjustment. **Adjustment is not based on a series of holes, notches, detents, or fixed-index increments.** The locking device must be operable without tools where practical and designed for serviceable, replaceable wearing surfaces.
- Ensure a printed end-stop/secondary restraint mechanically captures telescoping members, so even if the adjustment clamp loses grip the rod cannot pull apart. Friction alone is insufficient as separation protection. Include tactile/visual confirmation of the engaged lock and require clearance for servicing.
- Parametrize minimum telescoping overlap and dimensional tolerances for the actual selected filament and print orientation. Evaluate loss of clamp force due to filament creep, vibration, temperature, humidity, repeated operation and layer-direction loading. Test for slip at measured throttle forces before any field operation.
- If no fully printable scheme meets retention, repeatability, fatigue and return-to-idle safety criteria, flag this as an **unresolved engineering blocker**, not as a reason to quietly replace parts with metal hardware.

## Inverted-U / upside-down-box throttle lever attachment — dimensions not yet measured
The pushrod's throttle end must engage the existing mower's throttle control via an upside-down U-shaped or open-bottom box-shaped printed interface. **Neither its clear inside side-to-side width nor its clear inside front-to-back depth has been measured or established in the existing supplied photos or specifications. Do not invent either number or mark a CAD fit as verified.**

- Define separately parameterized **internal clear left-to-right width** (`throttle_tip_width_mm` plus twice `fit_clearance_side_mm`) and **internal clear front-to-back depth** (`throttle_tip_depth_mm` plus twice `fit_clearance_front_mm`). Also parameterize engagement height, wall thickness, fillets, motion clearance and removal/installation path.
- Obtain measured **actual throttle lever/knob outside width and front-to-back thickness/depth at the point of engagement**, including any taper, grip, protrusions and movement. The right steering bar offset ranges are *not* dimensions of the throttle lever or inverted-U tip.
- Use configurable print clearance chosen by test coupon and material; no unmeasured inside width/depth is to be presented as fixed or final.
- Model full lever sweep and contact loads, preventing binding, snagging and accidental throttle pull-open while allowing independent steering and emergency-stop operation. Print a small fit-check gauge for the selected opening before fabricating a full-length pushrod.
- Explicitly report these two currently missing measurements to the user in the final implementation summary.

## 3D printer manufacturing envelope — mandatory
Printer nominal build volume: **420 mm X × 420 mm Y × 500 mm Z** (millimeters).
- Treat those as absolute machine limits, not target part dimensions; provide configurable XY/Z clearance margins for skirt/brim, bed clamps, dimensional error, toolhead reach and print orientation. Validate each STL's **oriented bounding box** against the *usable* envelope with margin, not just nominal CAD dimensions.
- Split EVERY assembly that cannot be printed in the available orientation into smaller individual printable parts. This includes extended right-side servo mount, bracket gussets, long covers, horn adapters and other oversized structures as applicable. Split designs into practical manageable segments even when length fits but support/warping/strength makes a monolithic print undesirable.
- For every split, provide fully printable alignment and **positive mechanical capture**: keyed/dowel/tongue-and-groove interfaces, printed captive pins/wedges or locking collar, suitable reinforcement, accessible assembly clearances and printed anti-loosening features. Place joints away from high-bending/high-load areas wherever feasible.
- Do **not** use glue-only joints in structural or safety-relevant throttle linkages. Do not introduce purchased metal hardware, rods, bearings, clevises or ball joints into any proposed primary variant. Identify loads or dimensions that make an all-printed assembly unacceptably weak rather than pretending it passes.
- Design for printer nozzle/bed material capabilities only after confirming chosen filament and print settings; mark them unknown until measured. Print orientation, layers, wall count, infill, tolerances and supports are to be recorded per piece, not invented as verified settings.
- Generate a clearly numbered, labeled STL per printable piece, native parametric CAD per component, plus STEP where applicable. Include a printable-part manifest listing dimensions in mm, orientation, machine fit check, quantity, hardware, joins and assembly order.
- Provide a complete reassembled model in CAD, assembled linkage animation/range validation where toolchain supports it, exploded views, and assembly drawings/installation guide. Verify no moving part hits seams, screws, brackets, horn or steering mechanism.
- Ensure all printed pieces can be assembled without removing/defeating throttle stops or emergency shutdown. Document safe throttle behavior during power loss and mechanical jamming; physical bench testing is still required.

## Execution for Claude Code
1. Inspect existing mower CAD, current servo-throttle plans and repository docs before editing. Preserve source files, steering, and unrelated geometry.
2. Implement the five actual CAD options and continuous sliding pushrod fine adjustment, parameterize unknown physical measurements, and explicitly mark provisional assumptions.
3. Partition and export all printed parts to respect **420 × 420 × 500 mm** maximum build envelope with print margins; include orientation and bounding-box checks for each STL.
4. Provide technical drawings, **all-printable** BOM and print orientation for every new linkage component (including pivots, bushings, retainers and sliding adjustment locks), assembly guide, printer build-plate allocation and verification report.
5. Check geometry exports reopen, evaluate linkage kinematics and interferences, identify material-specific fatigue/creep and heat failure modes, document emergency shutdown and fail-safe return, and record physical testing still required.
6. Commit all changes on a dedicated design branch when implementing CAD. Do not claim prints or physical tests occurred. Do not select a winning option until user reviews variants.
