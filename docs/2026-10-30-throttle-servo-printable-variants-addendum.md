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

## Design alternatives — build all five, do not automatically select one
Build editable, parametric CAD variants with the same servo envelope and common mounting datum:
A. Direct connection to an existing X-horn arm.
B. Printed adapter attached to the existing X horn.
C. Continuously adjustable effective-radius horn adapter (slot/clamp, **not discrete indexing holes**).
D. Ball-link pushrod joint, accommodating operating angular misalignment.
E. Clevis-type removable pushrod joint.
Each variant must be independently manufacturable and show complete assembled and exploded views, material/hardware list, collision/kinematic checks, and tradeoffs. If two connection types can combine, keep comparisons meaningful and preserve separate model variants.

## Fine adjustment — non-negotiable
- Throttle pushrod usable length adjustment MUST be **continuous and micro-fine**, not selectable through rows of holes or detents.
- Use a sliding telescoping sleeve/rod with positive split-clamp or opposing clamping screws; optionally include a threaded fine-adjustment mechanism that provides measurable continuous motion and a locknut.
- Provide anti-rotation or positive orientation control where required, a visible calibration mark / datum, retention against separation, and locking against vibration. Do not rely solely on friction for a safety-critical throttle without mechanical retention.
- Previously recorded positional measurements: lowest throttle 40 in, full-forward throttle 44 in, with provisional adjustment bands 38–41 in and 43–46 in. These are **unverified measurements of unspecified datums**, not a proven 4-in pushrod stroke. Document datum definitions and validate actual travel before selecting servo-horn radius and usable servo angle.
- The separate parts of a telescoping mechanism must have sufficient overlap and strength throughout the required adjustment range; verify load, buckling and joint retention rather than assume one long 3D-printed rod is acceptable. A metal rod with printed mounts/couplers is allowed where prudent.

## 3D printer manufacturing envelope — mandatory
Printer nominal build volume: **420 mm X × 420 mm Y × 500 mm Z** (millimeters).
- Treat those as absolute machine limits, not target part dimensions; provide configurable XY/Z clearance margins for skirt/brim, bed clamps, dimensional error, toolhead reach and print orientation. Validate each STL's **oriented bounding box** against the *usable* envelope with margin, not just nominal CAD dimensions.
- Split EVERY assembly that cannot be printed in the available orientation into smaller individual printable parts. This includes extended right-side servo mount, bracket gussets, long covers, horn adapters and other oversized structures as applicable. Split designs into practical manageable segments even when length fits but support/warping/strength makes a monolithic print undesirable.
- For every split, provide alignment features and positive attachment: keyed/dowel/tongue-and-groove interfaces as appropriate, through-bolts or inserts, washers, suitable reinforcement, accessible assembly clearances and anti-loosening features. Place joints away from high-bending/high-load areas wherever feasible.
- Do **not** use glue-only joints in structural or safety-relevant throttle linkages. Prefer metal hardware and metal rods for highly loaded or long slender members.
- Design for printer nozzle/bed material capabilities only after confirming chosen filament and print settings; mark them unknown until measured. Print orientation, layers, wall count, infill, tolerances and supports are to be recorded per piece, not invented as verified settings.
- Generate a clearly numbered, labeled STL per printable piece, native parametric CAD per component, plus STEP where applicable. Include a printable-part manifest listing dimensions in mm, orientation, machine fit check, quantity, hardware, joins and assembly order.
- Provide a complete reassembled model in CAD, assembled linkage animation/range validation where toolchain supports it, exploded views, and assembly drawings/installation guide. Verify no moving part hits seams, screws, brackets, horn or steering mechanism.
- Ensure all printed pieces can be assembled without removing/defeating throttle stops or emergency shutdown. Document safe throttle behavior during power loss and mechanical jamming; physical bench testing is still required.

## Execution for Claude Code
1. Inspect existing mower CAD, current servo-throttle plans and repository docs before editing. Preserve source files, steering, and unrelated geometry.
2. Implement the five actual CAD options and continuous sliding pushrod fine adjustment, parameterize unknown physical measurements, and explicitly mark provisional assumptions.
3. Partition and export all printed parts to respect **420 × 420 × 500 mm** maximum build envelope with print margins; include orientation and bounding-box checks for each STL.
4. Provide technical drawings, BOM, assembly guide, printer build-plate allocation and verification report.
5. Check geometry exports reopen, evaluate linkage kinematics and interferences, and record anything needing a physical measurement or test.
6. Commit all changes on a dedicated design branch when implementing CAD. Do not claim prints or physical tests occurred. Do not select a winning option until user reviews variants.
