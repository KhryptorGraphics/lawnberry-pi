// Five complete throttle linkage variants for the RETAINED MG995 X-horn.
//
// =============================================================================
// FRAME CONTRACT (this file is the authority for the horn-side interface)
// =============================================================================
// Origin : the MG995 output-shaft axis.
// +Z     : up, along the factory shaft.  z = 0 is the horn UNDERSIDE plane; the
//          retained horn occupies z = 0 .. x_horn_thickness_mm (0 .. 2).
//          The scene places this frame at the mount's published horn datum
//          (throttle_horn_base_mm() in mg995_throttle_mount.scad).  Nothing here
//          re-derives the servo station: the mount owns that.
// Plan   : the retained horn's arms lie at 45 + 90*k degrees (the legacy horn
//          reference in assemblies.scad uses the same 45 degree phase).  Each arm
//          carries ONE provisional hole of x_horn_hole_d_mm at radius
//          x_horn_arm_radius_mm.
// Parts  : every printable part is modelled in the ARM-LOCAL frame (+X along the
//          first horn arm, +Y towards the second arm) with its lowest point at
//          z = 0, so it exports bed-zero.  The assembly rotates the arm-local
//          frame by 45 degrees to land on the real horn's arm phase.
//
// The rod eye
// -----------
// RodGeometry publishes the rod's servo eye as a PLATE 4 mm thick along rod-local
// X, 21.2 mm wide along rod-local Y, 12 mm tall along rod-local Z, bored along
// rod-local X with rod_eye_bore_d (5.8 nominal; that file cuts it at 6.6 with the
// published joint clearance).  The scene must therefore place the rod with
// rotate([0, 90, 0]), which maps
//     rod-local X -> world -Z   (the 4 mm plate thickness becomes vertical)
//     rod-local Y -> world  Y
//     rod-local Z -> world +X   (the rod runs toward the throttle lever)
// With that mapping the eye is a plate 4 mm thick VERTICALLY, 21.2 mm wide along
// world Y and 12 mm long along world X, with a VERTICAL bore; its neck leaves
// along +X.  Every part below respects that footprint and keeps the +X side
// clear for the rod's neck.
//
//   * Variants A, B, C and D press the eye onto a printed pivot post's Ø6.7
//     spigot and trap it with a printed cap.  The eye is trapped between the
//     post's flange and the cap, and the load runs in bearing:
//     horn -> carrier -> post flange -> eye -> cap -> rod.
//   * Variant E puts the eye between two clevis ears stacked VERTICALLY at
//     RodGeometry's published fork gap (rod_eye_fork_gap = 4.8 mm = 4 mm plate +
//     0.4 mm clearance), so the eye is loaded in DOUBLE SHEAR.  This file
//     consumes that published gap; it does not derive its own ear gap.
//
// The horn hole and the index pins
// -------------------------------
// The retained horn's hole is PROVISIONAL 3 mm, so no 5 mm hinge pin is ever
// placed in it.  Horn keying uses a small 2.4 mm printed index pin
// (keeper_post_pin_d_mm, live in parameters.scad): its Ø4.8 head sits in the
// carrier's counterbore on the carrier's top face, its shank passes down through
// the horn's own hole, and a printed split keeper ring snaps into a groove at the
// pin's lower end, underneath the horn.  The keeper's bore is smaller than the
// pin's 2.4 mm shank, so the pin cannot back out of the horn; the keeper's
// outside is wider than the horn's hole, so it cannot fall off either.
//
// =============================================================================
// WHAT IS STILL UNVERIFIED (stated, not hidden)
// =============================================================================
//   * Horn arm width, arm radius, hole diameter and thickness are TO MEASURE.
//     horn_arm_width_provisional_mm below is a DESIGN CAPACITY only.
//   * These parts assume a FOUR-ARM X horn with an open hub.  The legacy horn
//     reference in assemblies.scad draws a solid disc; that reference is a
//     drawing placeholder and does not match a real MG995 X horn.
//   * The MG995 body outline, its output-shaft boss and the clearance between
//     the horn's underside and the servo body are TO MEASURE.  Variants B, C, D
//     and E need about 4 mm of clearance under the horn for the keeper ring.
//   * TORQUE PATH: the only thing resisting the horn's rotation about the shaft
//     axis is the shear of the 2.4 mm index pins in the horn's provisional
//     holes.  That capacity is NOT calculated and is the weakest element of
//     every variant here.  A hub clamp is deliberately NOT modelled, because the
//     horn hub diameter is unmeasured and a collar of a guessed diameter would
//     foul the arms.  This is an open engineering item, not a solved joint.
//   * The horn's mounting PHASE must be chosen so the joint's yaw sweep misses
//     the servo body and the servo boss.  Not verifiable here.
//   * LATERAL THROW: the owner's reported 63.5-114.3 mm lateral band needs about
//     48.5 mm of equivalent radius at a 60 degree sweep.  The retained horn's
//     provisional arm radius is 18 mm and variant C's continuous range is capped
//     at horn_radius_max_mm = 30 mm.  NO VARIANT IN THIS FILE COVERS THAT BAND.
//     That is a stated limitation, not a satisfied requirement; it needs the
//     measured horn radius and the measured lever pivot.
//   * The rod eye's own cut bore is 6.6 mm, so a 5 mm pin in it has 0.8 mm of
//     diametral slack.  RodGeometry owns that file; the slack is reported here,
//     not absorbed.  Variant E's eye bushing is therefore a 0.1 mm LOCATING slip
//     fit in that bore, not a press fit; a press fit would need the eye's bore
//     reamed to 6.5, which is a change to RodGeometry's file, not to this one.
//   * The rod eye RUNS on the pivot post's 6.2 mm spigot in its 6.6 mm bore, with
//     0.2 mm axial play. The cap on the post's 7.6 mm seat, the bushing in the clevis
//     ear and the ball boss on the carrier's post carry about 0.1 mm interference.
//     All are geometry to be coupon-tested, not verified joints.
//   * D's truncated Ø10 ball is captive in a carrier-integral spherical socket whose
//     top mouth is Ø9.2 mm and whose bottom mouth forms naturally at Ø9.46 mm. The ball
//     and its output post print as a second bed-zero body; both openings are narrower
//     than the ball, so retention is geometric rather than friction-dependent. The
//     modeled ±10 degree sweep and as-printed clearance are prototype capacities, not
//     measured articulation or wear life.
//   * C's continuous-radius shoe slides in a 45-degree dovetail and is locked by a
//     separate printed wedge key. The carrier's bed-through track is backed by the
//     retained horn arm after installation; the horn width and backing fit are TO MEASURE.
//     The key's blind bore in the shoe is straight and larger than the key's widest
//     section, so its FLANK BEARING IS UNPROVEN and the shoe can still slide radially
//     until the key is fully home. The key's holding torque is also unqualified.
//   * No retention, load, wear life or servo-fit claim is made from overlap alone.
//
// =============================================================================
// PART FRAME CONTRACT
// =============================================================================
// Every printable part is modelled with its lowest point at z = 0 in its own local
// frame, so it exports bed-zero.  All new hardware is printed: no bought pin,
// screw, bushing, insert or ball.
//
// =============================================================================
// PRINTABLE SELECTORS (consumed by the catalog)
// =============================================================================
//   A: post, cap, keeper
//   B: shell_a, shell_b, lug, cap, pin, keeper
//   C: pip (carrier + shoe), wedge_key, wedge_keeper, cap, pin, keeper
//   D: pip (carrier socket + truncated ball/post), cap, pin, keeper
//   E: carrier, clevis, bushing, eye_bushing, pin, keeper, index_pin,
//      index_keeper
//
// =============================================================================
// PUBLIC PLACEMENT CONTRACT (consumed by assemblies.scad)
// =============================================================================
//   variant_rod_eye_center(variant, radius_mm) -> [x, y, z] of the rod eye's bore
//                                                centre, in the frame above
//   variant_rod_eye_axis()                     -> [0, 0, 1]  (vertical bore)
//   variant_joint_radius(variant, radius_mm)    -> effective joint radius
// The scene places the rod with rotate([0, -90, 0]) at that centre (rod-local X up).

include <parameters.scad>
use <printed_joints.scad>

// ---- Provisional horn-derived capacities (ALL TO MEASURE) --------------------------
horn_r = x_horn_arm_radius_mm;               // 18, provisional
horn_t = x_horn_thickness_mm;                 // 2, provisional
horn_hole = x_horn_hole_d_mm;                 // 3, provisional
shaft_clear = x_horn_hub_clear_d_mm;          // 12, clearance around the retainer

horn_arm_width_provisional_mm = 10;           // DESIGN CAPACITY, arm width TO MEASURE
horn_arm_angle_deg = 45;                      // first arm; matches the horn reference

// ---- Published joint hardware ------------------------------------------------------
small_pin_d = keeper_post_pin_d_mm;           // 2.4 horn index pin
small_pin_bore_d = keeper_post_pin_bore_d_mm; // 2.8 running bore
small_pin_head_d = 4.8;
joint_pin_d = pin_d_mm;                       // 5 common printed hinge pin
joint_pin_head_d = pin_head_d_mm;             // 8
bushing_od = bushing_od_mm;                   // 9
bushing_id = bushing_id_mm;                   // 5.4
keeper_t = keeper_t_mm;                       // 4

// The published fork gap for the rod eye.  Main is publishing RodGeometry's
// rod_eye_fork_gap in parameters.scad; the literal fallback keeps this file
// buildable until then and the assert keeps the two values in step.
function eye_fork_gap() = is_undef(rod_eye_fork_gap) ? 4.8 : rod_eye_fork_gap;
assert(abs(eye_fork_gap() - 4.8) < 1e-6,
       "Rod eye fork gap must stay 4.8 mm (4 mm plate + 0.4 mm clearance)");

// ---- Shared local dimensions -------------------------------------------------------
carrier_t = 3;            // horn-side carrier thickness
carrier_top = horn_t + carrier_t;   // 5: the carrier's top face above the horn
index_pin_len = 7;        // shank 0..7, head 5.8..7 (in the carrier's counterbore)
index_groove_len = 2;     // the keeper's band at the pin's lower end
index_groove_d = 1.8;
keeper_t2 = 2;            // keeper ring thickness
keeper_bore_d = 1.9;      // rides on the 1.8 mm band with clearance per side,
                           // and is still smaller than the 2.4 mm shank, so the
                           // keeper cannot be pushed back over it
keeper_od = 8;
post_flange_d = pivot_flange_d_mm;        // pivot post flange
post_flange_t = pivot_flange_t_mm;
post_spigot_d = pivot_spigot_d_mm;        // running fit in the rod eye's 6.6 mm bore
post_spigot_len = pivot_spigot_len_mm;    // 4 mm eye plus 0.2 mm axial play per side
post_cap_seat_d = pivot_cap_seat_d_mm;    // the printed cap sits on this
post_cap_seat_len = pivot_cap_seat_len_mm;
post_lip_d = pivot_lip_d_mm;              // traps the cap
post_lip_t = pivot_lip_t_mm;
cap_od = pivot_cap_od_mm;                 // wider than the eye's bore, so the cap traps it
cap_id = pivot_cap_id_mm;                 // clearance on the post's seat; the lip traps it
cap_h = 4;
cheek_t = 2.5;            // printed clevis ear thickness

// =============================================================================
// Shared printed primitives
// =============================================================================

// Horn index pin: a Ø2.4 shank with a keeper BAND at its lower end and a Ø4.8 head at
// its top (for the carrier's counterbore).
// Frame: band 0..index_groove_len, shank 0..len, head len-1.2..len.
//
// An annular subtraction leaves the solid reduced-diameter keeper band.
module index_pin(len = index_pin_len) {
    difference() {
        union() {
            cylinder(d = small_pin_d, h = len);
            translate([0, 0, len - 1.2]) cylinder(d = small_pin_head_d, h = 1.2);
        }
        // The keeper's band at the lower end.
        translate([0, 0, 0])
            pin_band_relief(d = small_pin_d, band_d = index_groove_d,
                            band_t = index_groove_len);
    }
}

// Split keeper ring that snaps into a groove, underneath the horn.  Its bore is
// smaller than the member's shank, so the member cannot back out; its outside is
// wider than the horn's hole, so it cannot fall off.
// `bore` must sit between the groove's diameter and the shank's diameter.
module index_keeper(bore = keeper_bore_d, od = keeper_od) {
    difference() {
        cylinder(d = od, h = keeper_t2);
        translate([0, 0, -1]) cylinder(d = bore, h = keeper_t2 + 2);
        // Radial split so the ring can be opened onto the groove.
        translate([0, -1.0, -1]) cube([od, 2.0, keeper_t2 + 2]);
    }
}

// X-shaped horn carrier: a hub ring around the factory shaft and four arm beams
// that lie on the horn's arms.  Index pins are placed at the arms listed in
// pin_arms, so a variant that needs the first arm's run clear can key off the
// other arms instead.
// Frame: z = 0 is the carrier's underside, which sits on the horn's top face.
module horn_carrier(t = carrier_t, arm_w = horn_arm_width_provisional_mm,
                    hub_bore = shaft_clear, pin_arms = [0, 1]) {
    difference() {
        union() {
            // Hub ring around the factory shaft retainer.
            difference() {
                cylinder(d = hub_bore + 8, h = t);
                translate([0, 0, -1]) cylinder(d = hub_bore, h = t + 2);
            }
            // Four arm beams with widened pads at the arm holes.
            for (k = [0:3]) rotate([0, 0, 90 * k]) {
                translate([4, -arm_w / 2, 0]) cube([horn_r + 2, arm_w, t]);
                translate([horn_r - 7, -arm_w / 2 - 2, 0])
                    cube([14, arm_w + 4, t]);
            }
        }
        // Index-pin bores and their head counterbores at the chosen arms.
        for (k = pin_arms) rotate([0, 0, 90 * k]) {
            translate([horn_r, 0, -1]) cylinder(d = small_pin_bore_d, h = t + 2);
            translate([horn_r, 0, t - 1.21])
                cylinder(d = small_pin_head_d + 0.1, h = 1.22);
        }
    }
}

// Pivot post: the common rod-eye attachment.
//     flange   -> bears on the horn-side carrier
//     spigot   -> Ø6.7 x 4: the rod eye is pressed onto this
//     cap seat -> Ø7.6 x 4: the printed cap sits on this
//     lip      -> Ø8.6 x 1.5: traps the cap
// Frame: z = 0 is the flange's bearing face.  The eye occupies 1.5..5.5, the cap
// sits on the eye at 5.5..9.5 and the lip traps the cap.
module pivot_post(flange_d = post_flange_d, flange_t = post_flange_t) {
    union() {
        cylinder(d = flange_d, h = flange_t);
        translate([0, 0, flange_t]) cylinder(d = post_spigot_d, h = post_spigot_len);
        translate([0, 0, flange_t + post_spigot_len])
            cylinder(d = post_cap_seat_d, h = post_cap_seat_len);
        translate([0, 0, flange_t + post_spigot_len + post_cap_seat_len])
            cylinder(d = post_lip_d, h = post_lip_t);
    }
}

// Printed cap that traps the rod eye on the post.  It is pressed onto the post's
// Ø7.6 seat and its Ø10 skirt is wider than the eye's bore, so the eye cannot
// pass over it; the post's lip traps the cap.
// Frame: z = 0 is the face that lands on the rod eye.
module pivot_cap() {
    difference() {
        cylinder(d = cap_od, h = cap_h);
        translate([0, 0, -0.01]) cylinder(d = cap_id, h = cap_h + 0.02);
    }
}

// =============================================================================
// A - direct joint through the existing X-horn arm hole
// =============================================================================
// Mechanism: a printed pivot post stands in the horn's own provisional arm hole.
// Its Ø2.8 shank passes through the hole and down below the horn, where a keeper
// ring snaps into its groove; its Ø7 flange bears directly on the horn's top
// face and carries the rod eye, which is pressed onto the post's spigot and
// trapped by the printed cap.  This is the tightest joint to the horn a printed
// part can be: the rod eye sits directly over the arm's own hole.
//
// World stack above the horn underside plane: post flange 2..3.5, eye 3.5..7.5,
// cap 7.5..11.5, post lip 11.5..13; keeper ring -2..0.
//
// LIMITATION: the 2.8 mm shank in a provisional 3 mm hole carries the whole horn
// torque by shear; that capacity is NOT calculated and is the weakest element of
// the design.  A second post on another arm would halve the load.
module a_post() {
    // Frame: the shank's lower end at z = 0, so the part exports bed-zero.
    difference() {
        union() {
            // Locating shank down through the horn's own hole.
            cylinder(d = small_pin_bore_d, h = 4);
            // Flange bearing on the horn's top face, then the pivot stack.
            translate([0, 0, 4]) cylinder(d = post_flange_d, h = post_flange_t);
            translate([0, 0, 4 + post_flange_t]) cylinder(d = post_spigot_d, h = post_spigot_len);
            translate([0, 0, 4 + post_flange_t + post_spigot_len])
                cylinder(d = post_cap_seat_d, h = post_cap_seat_len);
            translate([0, 0, 4 + post_flange_t + post_spigot_len + post_cap_seat_len])
                cylinder(d = post_lip_d, h = post_lip_t);
        }
        // An annular subtraction leaves a solid, reduced-diameter keeper band.
        translate([0, 0, 0])
            pin_band_relief(d = small_pin_bore_d, band_d = index_groove_d,
                            band_t = index_groove_len);
    }
}

module a_cap() {
    pivot_cap();
}

module a_keeper() {
    index_keeper();
}

module variant_a(part = "assembly") {
    if (part == "assembly") {
        rotate([0, 0, horn_arm_angle_deg]) {
            // Post: flange bearing on the horn's top face at the arm hole.
            translate([horn_r, 0, horn_t - 4]) a_post();
            // Keeper ring under the horn, in the post's groove.
            // The keeper is CENTRED in the pin's band, so it has axial clearance at
                // both ends and the band's shoulders trap it.
                translate([horn_r, 0, -keeper_t2]) a_keeper();
            // Cap over the rod eye.
            translate([horn_r, 0, horn_t + post_flange_t + post_spigot_len]) a_cap();
        }
    } else if (part == "post") {
        a_post();
    } else if (part == "cap") {
        a_cap();
    } else if (part == "keeper") {
        a_keeper();
    } else {
        assert(false, str("A: unknown part ", part));
    }
}

// =============================================================================
// B - two-piece X-shaped reinforcement shell over the retained horn
// =============================================================================
// Mechanism: an X-shaped lower tray hugs the horn's four arms from below and an
// X-shaped upper cap covers them from above.  The cap bears on the horn's top
// face over all four arms, so the reaction is distributed across the X instead
// of being concentrated at one hole, and the tray captures the arms from below
// so the shell cannot walk sideways off them.  Four index pins pass through the
// horn's own provisional holes, joining the cap to the tray, with keeper rings
// under the horn.  A printed lug on a side tab carries the pivot post, so the
// rod eye is not directly over the horn's arm and the shell can rotate with the
// horn without the eye fouling the horn material.
//
// World stack above the horn underside plane: tray -4..0, cap 2..5 with arm rails to 7,
// index pin head 5..6.2, keeper -2..0, lug flange 5..8, eye 8.2..12.2, cap 12.4..16.4.
//
// LIMITATION: the tray's channel width is a design capacity, not a measured arm
// width; the shell assumes a four-arm X horn, not the solid disc used as a legacy
// drawing reference; the tray must clear the servo body below the horn, and the
// cap's side tab is a cantilever over open space.
b_lug_offset = 12;   // lateral offset of the lug from the first arm centreline
// Extra lug flange height: the cap's arm rails rise to 7 above the horn underside, so a
// standard 1.5 mm flange put the eye plate's underside at 6.7, into the rails. With this
// rise the eye clears the rail tops by 1.2 mm.
b_lug_rise = 1.5;

module b_shell_a() {
    // Lower X tray: hub ring plus four arm channels under the horn's arms.
    arm_w = horn_arm_width_provisional_mm;
    hub_d = shaft_clear + 8;
    difference() {
        union() {
            difference() {
                cylinder(d = hub_d, h = 4);
                translate([0, 0, -1]) cylinder(d = shaft_clear, h = 6);
            }
            for (k = [0:3]) rotate([0, 0, 90 * k]) {
                translate([0, -arm_w / 2, 0]) cube([horn_r + 2, arm_w, 2]);
                translate([0, -arm_w / 2, 0]) cube([horn_r + 2, 2, 4]);
                translate([0, arm_w / 2 - 2, 0]) cube([horn_r + 2, 2, 4]);
            }
        }
        // Clearance for the index pin's keeper ring.
        for (k = [0:3]) rotate([0, 0, 90 * k])
            translate([horn_r, 0, -1]) cylinder(d = keeper_od + 0.4, h = 3);
    }
}

module b_shell_b() {
    // Upper X cap: hub ring, four arm covers, a side tab to the lug station and
    // the index-pin bores.
    arm_w = horn_arm_width_provisional_mm;
    hub_d = shaft_clear + 8;
    difference() {
        union() {
            difference() {
                cylinder(d = hub_d, h = carrier_t);
                translate([0, 0, -1]) cylinder(d = shaft_clear, h = carrier_t + 2);
            }
            for (k = [0:3]) rotate([0, 0, 90 * k]) {
                translate([0, -arm_w / 2, 0]) cube([horn_r + 2, arm_w, carrier_t]);
                translate([0, -arm_w / 2, 0]) cube([horn_r + 2, 2, carrier_t + 2]);
                translate([0, arm_w / 2 - 2, 0]) cube([horn_r + 2, 2, carrier_t + 2]);
            }
            // Side tab bridging to the lug station, clear of the horn.
            translate([horn_r - 8, -b_lug_offset - 8, 0])
                cube([16, b_lug_offset + 8, carrier_t]);
        }
        // Index-pin bores and their head counterbores at each arm.
        for (k = [0:3]) rotate([0, 0, 90 * k]) {
            translate([horn_r, 0, -1]) cylinder(d = small_pin_bore_d, h = carrier_t + 2);
            translate([horn_r, 0, carrier_t - 1.21])
                cylinder(d = small_pin_head_d + 0.1, h = 1.22);
        }
        // Lug locating bore in the side tab.
        translate([horn_r, -b_lug_offset, -1]) cylinder(d = small_pin_bore_d,
                                                        h = carrier_t + 2);
    }
}

module b_lug() {
    // Printed lug: a locating spigot that plugs down into the cap's tab, a
    // flange that bears on the cap's top face, and the pivot post for the rod eye.
    // Frame: the spigot's lower end at z = 0, so the part exports bed-zero.
    union() {
        cylinder(d = small_pin_bore_d, h = carrier_t);          // spigot into the tab
        translate([0, 0, carrier_t])
            cylinder(d = post_flange_d + 4, h = post_flange_t + b_lug_rise);
        translate([0, 0, carrier_t + post_flange_t + b_lug_rise])
            cylinder(d = post_spigot_d, h = post_spigot_len);
        translate([0, 0, carrier_t + post_flange_t + b_lug_rise + post_spigot_len])
            cylinder(d = post_cap_seat_d, h = post_cap_seat_len);
        translate([0, 0, carrier_t + post_flange_t + b_lug_rise + post_spigot_len
                   + post_cap_seat_len])
            cylinder(d = post_lip_d, h = post_lip_t);
    }
}

module b_cap() {
    pivot_cap();
}

module b_pin() {
    index_pin(len = index_pin_len);
}

module b_keeper() {
    index_keeper();
}

module variant_b(part = "assembly") {
    if (part == "assembly") {
        rotate([0, 0, horn_arm_angle_deg]) {
            translate([0, 0, -4]) b_shell_a();              // tray under the horn
            translate([0, 0, horn_t]) b_shell_b();           // cap over the horn
            for (k = [0:3]) rotate([0, 0, 90 * k]) {
                // Index pin: head in the cap's counterbore, shank through the
                // horn's own hole, keeper ring in its groove under the horn.
                translate([horn_r, 0, carrier_top - index_pin_len]) b_pin();
                translate([horn_r, 0, -keeper_t2]) b_keeper();
            }
            // Lug on the cap's side tab, with its pivot post.
            translate([horn_r, -b_lug_offset, horn_t]) b_lug();
            translate([horn_r, -b_lug_offset,
                       horn_t + carrier_t + post_flange_t + b_lug_rise + post_spigot_len])
                b_cap();
        }
    } else if (part == "shell_a") {
        b_shell_a();
    } else if (part == "shell_b") {
        b_shell_b();
    } else if (part == "lug") {
        b_lug();
    } else if (part == "cap") {
        b_cap();
    } else if (part == "pin") {
        b_pin();
    } else if (part == "keeper") {
        b_keeper();
    } else {
        assert(false, str("B: unknown part ", part));
    }
}

// =============================================================================
// C - continuously variable effective horn radius on a captive carriage
// =============================================================================
// Mechanism: an X-shaped carrier holds a radial rail. The shoe is a PIP slider in a
// continuous 45-degree dovetail, locked at its chosen radius by a separate printed wedge.
// A narrow bed throat and a wider internal foot channel capture the tongue from below;
// the upper dovetail prevents lift-out. The continuous rail has closed end stops.
//
// The carrier's arm is widened around the through-track. The horn's arm width, the
// backing/contact at the shoe foot, and the rail's outboard reach are provisional and
// require measurement. The wedge key's holding torque is also unqualified.
c_tongue_len = 16;
c_rail_x0 = horn_radius_min_mm - c_tongue_len / 2 - 0.5;
c_rail_x1 = horn_radius_max_mm + c_tongue_len / 2 + 0.5;
c_track_x0 = 10.5;                    // beyond the hub; forms an inboard travel stop
c_track_x1 = c_rail_x1 - 1;          // closed outboard end stop
c_rail_w = 12;
c_rail_h = 8;
 c_neck_w = 5.6;                       // dovetail mouth and straight neck
 c_bottom_slot_w = 2.4;                // narrow throat below the captured shoe foot
 c_dovetail_base_w = c_bottom_slot_w + 2 * carrier_t; // 45-degree lower flare: 8.4
 c_dovetail_depth = (c_dovetail_base_w - c_neck_w) / 2; // 45-degree upper undercut: 1.4
 c_pip_gap_mm = sliding_clearance_mm + 0.2; // facet reserve above the shared 0.4 mm gap
 c_dovetail_fit_gap = c_pip_gap_mm * sqrt(2); // normal clearance on 45-degree faces
 c_tongue_bottom_w = c_bottom_slot_w - 2 * c_pip_gap_mm;
 c_tongue_base_w = c_dovetail_base_w - 2 * c_dovetail_fit_gap;
 c_tongue_ramp_top_w = c_neck_w - 2 * c_dovetail_fit_gap;
 c_tongue_neck_w = c_neck_w - 2 * c_pip_gap_mm;
 c_tongue_neck_top_z = carrier_t + c_rail_h + c_pip_gap_mm;
c_shoe_platform_w = 15;
c_shoe_platform_x_len = 10;
c_shoe_transition_h = (c_shoe_platform_w - c_tongue_neck_w) / 2;
c_shoe_platform_z0 = c_tongue_neck_top_z + c_shoe_transition_h;
c_shoe_platform_t = 0.5;
c_shoe_post_z0 = c_shoe_platform_z0 + c_shoe_platform_t;
c_key_tip_d = 2.4;                    // printed wedge key's leading taper
c_key_shaft_d = 3.0;
c_key_taper_len = 8;
c_key_groove_d = 1.8;
c_key_len = 10;
c_key_x_offset_mm = 1.5;             // keeps the key bore clear of the inboard end stop
c_key_axis_z = 10.5;                 // world height of the key's axis
c_key_tip_y = -0.7;                  // world Y of the key's tip
c_slot_h = 3.1;

module c_carrier() {
    difference() {
        union() {
            // Widen the first arm around the bed-through carriage track.
            horn_carrier(t = carrier_t, arm_w = 16, pin_arms = [1, 2]);
            translate([c_rail_x0, -c_rail_w / 2, carrier_t])
                cube([c_rail_x1 - c_rail_x0, c_rail_w, c_rail_h]);
        }
        // Bed-through base channel, 45-degree dovetail undercut, and narrow neck.
        // The track endpoints remain solid to stop the shoe leaving the radial rail.
        translate([c_track_x0, 0, 0]) rotate([90, 0, 90])
            linear_extrude(height = c_track_x1 - c_track_x0)
                polygon([
                    [-c_bottom_slot_w / 2, -0.01],
                    [ c_bottom_slot_w / 2, -0.01],
                    [ c_bottom_slot_w / 2, 0],
                    [ c_dovetail_base_w / 2, carrier_t],
                    [ c_neck_w / 2, carrier_t + c_dovetail_depth],
                    [ c_neck_w / 2, carrier_t + c_rail_h + 0.01],
                    [-c_neck_w / 2, carrier_t + c_rail_h + 0.01],
                    [-c_neck_w / 2, carrier_t + c_dovetail_depth],
                    [-c_dovetail_base_w / 2, carrier_t],
                ]);
        // Continuous key slot through the +Y neck wall; the wedge clamps at any radius.
        translate([c_track_x0, c_neck_w / 2 - 0.4,
                   c_key_axis_z - horn_t - c_slot_h / 2])
            cube([c_track_x1 - c_track_x0, 4.4, c_slot_h]);
    }
}

module c_shoe(radius_mm = horn_radius_default_mm) {
    // Bed-zero slider: the narrow foot is captured in the carrier's flared lower
    // channel; the broad tongue is trapped by the upper 45-degree dovetail.
    tongue_x0 = max(radius_mm - c_tongue_len / 2,
                    c_track_x0 + c_pip_gap_mm);
    tongue_x1 = min(radius_mm + c_tongue_len / 2,
                    c_track_x1 - c_pip_gap_mm);
    assert(tongue_x1 > tongue_x0, "C shoe tongue must overlap its captive rail");
    difference() {
        union() {
            translate([tongue_x0, 0, 0]) rotate([90, 0, 90])
                linear_extrude(height = tongue_x1 - tongue_x0)
                    polygon([
                        [-c_tongue_bottom_w / 2, 0],
                        [ c_tongue_bottom_w / 2, 0],
                        [ c_tongue_base_w / 2, carrier_t],
                        [ c_tongue_ramp_top_w / 2, carrier_t + c_dovetail_depth],
                        [ c_tongue_neck_w / 2, carrier_t + c_dovetail_depth],
                        [ c_tongue_neck_w / 2, c_tongue_neck_top_z],
                        [-c_tongue_neck_w / 2, c_tongue_neck_top_z],
                        [-c_tongue_neck_w / 2, carrier_t + c_dovetail_depth],
                        [-c_tongue_ramp_top_w / 2, carrier_t + c_dovetail_depth],
                        [-c_tongue_base_w / 2, carrier_t],
                    ]);
            // Self-supporting 45-degree transition from the narrow rail neck to the
            // post platform. The platform carries the rod eye's vertical load.
            hull() {
                translate([tongue_x0, -c_tongue_neck_w / 2,
                           c_tongue_neck_top_z - sliding_clearance_mm])
                    cube([tongue_x1 - tongue_x0, c_tongue_neck_w,
                          sliding_clearance_mm + 0.01]);
                translate([radius_mm - c_shoe_platform_x_len / 2,
                           -c_shoe_platform_w / 2, c_shoe_platform_z0])
                    cube([c_shoe_platform_x_len, c_shoe_platform_w, 0.01]);
            }
            translate([radius_mm - c_shoe_platform_x_len / 2,
                       -c_shoe_platform_w / 2, c_shoe_platform_z0])
                cube([c_shoe_platform_x_len, c_shoe_platform_w, c_shoe_platform_t]);
            translate([radius_mm, 0, c_shoe_post_z0])
                cylinder(d = post_flange_d, h = post_flange_t);
            translate([radius_mm, 0, c_shoe_post_z0 + post_flange_t])
                cylinder(d = post_spigot_d, h = post_spigot_len);
            translate([radius_mm, 0, c_shoe_post_z0 + post_flange_t + post_spigot_len])
                cylinder(d = post_cap_seat_d, h = post_cap_seat_len);
            translate([radius_mm, 0, c_shoe_post_z0 + post_flange_t + post_spigot_len
                                     + post_cap_seat_len])
                cylinder(d = post_lip_d, h = post_lip_t);
        }
        // The separate wedge key enters through the rail's +Y side. The blind bore is
        // straight and slightly larger than the key's widest section, so the key's
        // FLANK BEARING IS UNPROVEN: the shoe can still slide radially until the
        // key is driven home. See the C note in README.md.
        translate([radius_mm + c_key_x_offset_mm, 0.9, c_key_axis_z - horn_t])
            rotate([90, 0, 0]) cylinder(d = c_key_shaft_d + 0.1, h = 3.2,
                                         center = true);
    }
}

module c_wedge_key() {
    // Printed wedge key: a self-locking tapered pin with a keeper BAND near its outer
    // end for the keeper ring.
    // Frame: the tip at z = 0, the shaft above it, the band at the top.
    //
    // An annular subtraction leaves a solid, reduced-diameter keeper band.
    difference() {
        union() {
            cylinder(d1 = c_key_tip_d, d2 = c_key_shaft_d, h = c_key_taper_len);
            translate([0, 0, c_key_taper_len])
                cylinder(d = c_key_shaft_d, h = c_key_len - c_key_taper_len);
        }
        // The keeper ring's band.
        translate([0, 0, c_key_len - keeper_t2])
            pin_band_relief(d = c_key_shaft_d, band_d = c_key_groove_d,
                            band_t = keeper_t2);
    }
}

module c_wedge_keeper() {
    // Keeper ring for the wedge key: its bore is smaller than the key's shaft, so
    // it cannot pass over it.
    difference() {
        cylinder(d = keeper_od, h = keeper_t2);
        translate([0, 0, -1]) cylinder(d = c_key_groove_d + 0.1, h = keeper_t2 + 2);
        translate([0, -1.0, -1]) cube([keeper_od, 2.0, keeper_t2 + 2]);
    }
}

module c_cap() {
    pivot_cap();
}

module c_pin() {
    index_pin(len = index_pin_len);
}

module c_keeper() {
    index_keeper();
}

module variant_c(part = "assembly", radius_mm = horn_radius_default_mm) {
    assert(is_num(radius_mm)
           && radius_mm >= horn_radius_min_mm && radius_mm <= horn_radius_max_mm,
           str("C: radius must lie inside [", horn_radius_min_mm, ", ",
               horn_radius_max_mm, "] mm"));
    if (part == "assembly") {
        rotate([0, 0, horn_arm_angle_deg]) {
            translate([0, 0, horn_t]) c_carrier();
            translate([0, 0, horn_t]) c_shoe(radius_mm);
            // Wedge key locks the continuously adjustable shoe after print/assembly.
            translate([radius_mm + c_key_x_offset_mm, c_key_tip_y, c_key_axis_z])
                rotate([-90, 0, 0]) c_wedge_key();
            translate([radius_mm + c_key_x_offset_mm,
                       c_key_tip_y + c_key_len - keeper_t2, c_key_axis_z])
                rotate([-90, 0, 0]) c_wedge_keeper();
            for (k = [1, 2]) rotate([0, 0, 90 * k]) {
                translate([horn_r, 0, carrier_top - index_pin_len]) c_pin();
                translate([horn_r, 0, -keeper_t2]) c_keeper();
            }
            translate([radius_mm, 0, horn_t + c_shoe_post_z0 + post_flange_t
                                      + post_spigot_len]) c_cap();
        }
    } else if (part == "pip") {
        c_carrier();
        c_shoe(radius_mm);
    } else if (part == "wedge_key") {
        c_wedge_key();
    } else if (part == "wedge_keeper") {
        c_wedge_keeper();
    } else if (part == "cap") {
        c_cap();
    } else if (part == "pin") {
        c_pin();
    } else if (part == "keeper") {
        c_keeper();
    } else {
        assert(false, str("C: unknown part ", part));
    }
}

// =============================================================================
// D - truncated ball printed captive in a carrier-integral socket
// =============================================================================
// The carrier includes a spherical socket around a bed-zero truncated Ø10 ball.
// Its naturally formed bottom mouth and top mouth are narrower than the ball, so
// the ball and its output post print together as two captive bodies. The spherical
// running clearance is 0.6 mm per side (the shared 0.4 mm sliding allowance plus
// 0.2 mm facet/fit reserve); coupon fit, wear and load life remain unqualified.
// The mouth/neck geometry permits a declared +/-10 degree CAD sweep, not a
// measured mower articulation.
d_ball_center_z = 3;
d_ball_d = ball_d_mm;
d_socket_clearance = sliding_clearance_mm + 0.2;
d_socket_inner_d = d_ball_d + 2 * d_socket_clearance;
d_socket_outer_d = 18;
d_socket_top_mouth_d = 9.2;
d_socket_height = d_ball_center_z
                  + sqrt(d_socket_inner_d * d_socket_inner_d / 4
                         - d_socket_top_mouth_d * d_socket_top_mouth_d / 4);
d_socket_arm_w = horn_arm_width_provisional_mm + 6;
d_ball_post_start_z = d_ball_center_z + d_ball_d / 2 - 0.2;
d_post_flange_z = 16;
d_ball_articulation_deg = 10;

module d_carrier() {
    difference() {
        union() {
            // Widen the unpinned first arm around the socket passage.
            horn_carrier(t = carrier_t, arm_w = d_socket_arm_w, pin_arms = [1, 2]);
            translate([horn_r, 0, 0])
                cylinder(d = d_socket_outer_d, h = d_socket_height, $fn = 96);
        }
        // The spherical cavity itself opens through the carrier-integral socket's
        // top and bottom mouths. Its continuous surface avoids a faceted cutter seam.
        translate([horn_r, 0, d_ball_center_z])
            sphere(d = d_socket_inner_d, $fn = 96);
    }
}

module d_ball_stem() {
    union() {
        // Truncate the bottom of the sphere at z=0. This flat face prints on the
        // bed and grows at a self-supporting slope into the spherical surface.
        intersection() {
            translate([0, 0, d_ball_center_z]) sphere(d = d_ball_d, $fn = 96);
            translate([-d_ball_d, -d_ball_d, 0])
                cube([2 * d_ball_d, 2 * d_ball_d, d_ball_center_z + d_ball_d]);
        }
        // The output stem starts inside the sphere to avoid a coincident-face joint.
        translate([0, 0, d_ball_post_start_z])
            cylinder(d = post_spigot_d,
                     h = d_post_flange_z - d_ball_post_start_z + 0.2);
        translate([0, 0, d_post_flange_z])
            cylinder(d = post_flange_d, h = post_flange_t);
        translate([0, 0, d_post_flange_z + post_flange_t - 0.1])
            cylinder(d = post_spigot_d, h = post_spigot_len + 0.1);
        translate([0, 0, d_post_flange_z + post_flange_t + post_spigot_len - 0.1])
            cylinder(d = post_cap_seat_d, h = post_cap_seat_len + 0.1);
        translate([0, 0, d_post_flange_z + post_flange_t + post_spigot_len
                                  + post_cap_seat_len - 0.1])
            cylinder(d = post_lip_d, h = post_lip_t + 0.1);
    }
}

module d_ball_stem_pitched(angle) {
    translate([0, 0, d_ball_center_z])
        rotate([0, angle, 0])
            translate([0, 0, -d_ball_center_z]) d_ball_stem();
}

module d_cap() {
    pivot_cap();
}

module d_pin() {
    index_pin(len = index_pin_len);
}

module d_keeper() {
    index_keeper();
}

module variant_d(part = "assembly") {
    if (part == "assembly") {
        rotate([0, 0, horn_arm_angle_deg]) {
            translate([0, 0, horn_t]) d_carrier();
            translate([horn_r, 0, horn_t]) d_ball_stem();
            for (k = [1, 2]) rotate([0, 0, 90 * k]) {
                translate([horn_r, 0, carrier_top - index_pin_len]) d_pin();
                translate([horn_r, 0, carrier_top - index_pin_len + index_groove_len / 2])
                    d_keeper();
            }
            // Cap over the rod eye on the ball's output post.
            translate([horn_r, 0, horn_t + d_post_flange_z + post_flange_t
                                  + post_spigot_len]) d_cap();
        }
    } else if (part == "pip") {
        // The print frame keeps both socket and truncated ball on the bed.
        d_carrier();
        translate([horn_r, 0, 0]) d_ball_stem();
    } else if (part == "cap") {
        d_cap();
    } else if (part == "pin") {
        d_pin();
    } else if (part == "keeper") {
        d_keeper();
    } else {
        assert(false, str("D: unknown part ", part));
    }
}

// =============================================================================
// E - double-shear clevis with replaceable printed bushings
// =============================================================================
// Mechanism: the carrier's first arm continues outboard past the horn's arm tip
// as a pair of printed rails, so the clevis stands clear of the horn and the
// 5 mm hinge pin never passes through the 3 mm horn hole.  The index pins stay
// on the other two arms, so the first arm's run is clear.  The clevis's two ears
// are stacked VERTICALLY at RodGeometry's published fork gap.  Replaceable
// printed bushings are pressed into the ears, and a replaceable printed bushing
// is a locating slip fit in the rod eye's 6.6 mm bore.  The eye sits between the
// ears, so it is loaded in DOUBLE SHEAR.  The printed hinge pin is inserted from
// above; its head lands on the upper ear and a printed keeper ring snaps into a
// groove at its lower end, underneath the clevis's base plate, so the pin cannot
// back out.
//
// The joint's radius is 30 mm, the limit of the continuous range the retained
// horn's provisional radius allows.
//
// World stack above the horn underside plane: carrier 2..5, rails 5..8, clevis base plate
// 8..12 resting on the rails, lower ear 12..14.5, eye 14.9..18.9, upper ear 19.3..21.8,
// hinge pin 6..23.8 (head on the upper ear), keeper ring 6..8 under the base plate.
// An earlier revision fixed the plate's underside at 4, inside the rails, and put the
// hinge pin's head at the bore centre, inside the clevis gap where the rod eye sits.
//
// LIMITATION: the bushing seats are press fits that must be reamed; the eye's
// own 6.6 mm bore means the printed eye bushing is a 0.1 mm slip fit rather than
// a press fit, and a true press fit would need RodGeometry to ream the eye's
// bore; the carrier's rails are 20 mm cantilevers and their stiffness is TO
// MEASURE; the joint must clear the horn through the whole yaw sweep, which
// depends on the real horn's shape.
e_radius = 30;         // the clevis's radius from the shaft axis
e_ear_x = 9;           // ear half width along X
e_ear_y = 14;          // ear half depth along Y
e_rail_y = 16;         // the rails' centreline, clear of the horn's arms
e_rail_w = 8;
e_plate_t = 4;         // the clevis's base plate thickness
e_rail_top = horn_t + 2 * carrier_t;   // 8: the rails' top face, under the base plate
// The bore centre: base plate on the rails, then the lower ear, then half the gap.
e_bore_z = e_rail_top + e_plate_t + eye_fork_gap() / 2 + cheek_t;   // 16.9
e_pin_len = 17.8;      // hinge pin length, from its lower end to its head's top
// The hinge pin's lower end: its head lands on the upper ear's top face.
e_pin_z0 = e_bore_z + eye_fork_gap() / 2 + cheek_t + 2 - e_pin_len;  // 6
// For fit_checks, which `use`s this file and so cannot see its variables.
function e_ear_t_mm() = cheek_t;
e_keeper_bore_d = 4.95; // rides on the hinge pin's 4.9 mm band with 0.025 mm of
                          // clearance per side, and is still smaller than the 5 mm
                          // shaft, so the keeper cannot be pushed back over it
e_keeper_groove_d = 4.9; // the band's own diameter
e_keeper_od = 9;

module e_carrier() {
    difference() {
        union() {
            horn_carrier(t = carrier_t, pin_arms = [1, 2]);
            // Rails continuing the first arm past the horn's tip.
            for (s = [-1, 1])
                translate([horn_r - 4, s * e_rail_y - e_rail_w / 2, carrier_t])
                    cube([e_radius + e_ear_x - horn_r + 4, e_rail_w, carrier_t]);
            // Join each outboard rail to the carrier's first arm.
            for (s = [-1, 1])
                translate([horn_r - 4, s > 0 ? 0 : -e_rail_y, carrier_t - 1])
                    cube([8, e_rail_y, 2]);
        }
        // Blind sockets for the clevis's two locating dowels.
        for (s = [-1, 1])
            translate([e_radius, s * e_rail_y, carrier_t - 0.01])
                cylinder(d = 2.9, h = 3.02);
    }
}

module e_clevis() {
    // Two ears stacked vertically with the published gap between them, joined by
    // a web on the -Y side (the rod's neck leaves along +X, so the +X side is
    // left clear), and carried on a base plate that rests on the carrier's rails.
    // Frame: z = 0 is the bore centre; the ears are at +/- gap/2.
    gap = eye_fork_gap();
    plate_z = e_bore_z - e_rail_top;   // bore centre to the plate's underside on the rails
    difference() {
        union() {
            translate([-e_ear_x, -e_ear_y, gap / 2])
                cube([2 * e_ear_x, 2 * e_ear_y, cheek_t]);
            translate([-e_ear_x, -e_ear_y, -gap / 2 - cheek_t])
                cube([2 * e_ear_x, 2 * e_ear_y, cheek_t]);
            translate([-e_ear_x, -e_ear_y, -gap / 2 - cheek_t])
                cube([2 * e_ear_x, 2, gap + 2 * cheek_t]);
            // Base plate, resting on the carrier's rails.
            translate([-e_ear_x - 4, -e_rail_y - e_rail_w / 2, -plate_z])
                cube([2 * e_ear_x + 8, 2 * e_rail_y + e_rail_w, e_plate_t]);
            // Bridge the base to the lower ear on the same side as the ear web.
            translate([-e_ear_x, -e_ear_y, -plate_z + e_plate_t - 1])
                cube([2 * e_ear_x, 2, plate_z - e_plate_t - gap / 2 - cheek_t + 2]);
            // Locating dowels into the carrier's sockets.
            for (s = [-1, 1])
                translate([0, s * e_rail_y, -plate_z - 3])
                    cylinder(d = 2.6, h = 3.2);
        }
        // Bushing seats through both ears.
        translate([0, 0, -gap / 2 - cheek_t - 1])
            cylinder(d = bushing_od - 0.2, h = gap + 2 * cheek_t + 2);
        // The hinge pin's clearance through the base plate.
        translate([0, 0, -plate_z - 1]) cylinder(d = joint_pin_d + 0.8, h = e_plate_t + 2);
    }
}

module e_bushing() {
    // Replaceable bushing pressed into the clevis ear.
    printed_bushing(od = bushing_od, id = bushing_id, len = cheek_t);
}

module e_eye_bushing() {
    // Replaceable bushing, a locating slip fit in the rod eye's 6.6 mm bore.
    difference() {
        cylinder(d = 6.5, h = 4);
        translate([0, 0, -1]) cylinder(d = 5.4, h = 6);
    }
}

module e_hinge_pin() {
    // Printed hinge pin: a Ø5 shaft with a Ø4.9 keeper BAND at its lower end and a Ø8
    // head at its top, which lands on the upper ear.
    // Frame: the pin's lower end at z = 0.
    //
    // An annular subtraction leaves a solid, reduced-diameter keeper band.
    difference() {
        union() {
            cylinder(d = joint_pin_d, h = e_pin_len);
            translate([0, 0, e_pin_len - 2]) cylinder(d = joint_pin_head_d, h = 2);
        }
        // The keeper ring's band.
        translate([0, 0, 0])
            pin_band_relief(d = joint_pin_d, band_d = e_keeper_groove_d,
                            band_t = keeper_t2);
    }
}

module e_keeper() {
    // Printed keeper ring for the hinge pin: its bore is smaller than the pin's
    // shaft, so it cannot pass over it, and it bears on the clevis's base plate.
    // Frame: z = 0 is the ring's bottom face.
    difference() {
        cylinder(d = e_keeper_od, h = keeper_t2);
        translate([0, 0, -1]) cylinder(d = e_keeper_bore_d, h = keeper_t2 + 2);
        translate([0, -1.2, -1]) cube([e_keeper_od, 2.4, keeper_t2 + 2]);
    }
}

module e_pin() {
    index_pin(len = index_pin_len);
}

module e_keeper_ring() {
    index_keeper();
}

module variant_e(part = "assembly") {
    if (part == "assembly") {
        rotate([0, 0, horn_arm_angle_deg]) {
            translate([0, 0, horn_t]) e_carrier();
            // Clevis: bore centre at e_bore_z above the horn underside.
            translate([e_radius, 0, e_bore_z]) e_clevis();
            // Hinge pin: its lower end reaches below the base plate, where the
            // keeper ring snaps onto its groove.
            translate([e_radius, 0, e_pin_z0]) e_hinge_pin();
            translate([e_radius, 0, e_pin_z0]) e_keeper();
            // Bushings pressed into both ears, flush with the ears' faces. The pin's
            // Ø8 head lands on the upper bushing; the ears' seats are wider than the head.
            translate([e_radius, 0, e_bore_z - eye_fork_gap() / 2 - cheek_t]) e_bushing();
            translate([e_radius, 0, e_bore_z + eye_fork_gap() / 2]) e_bushing();
            // Eye bushing: a locating slip fit in the rod eye's bore, centred on the eye.
            translate([e_radius, 0, e_bore_z - 2]) e_eye_bushing();
            for (k = [1, 2]) rotate([0, 0, 90 * k]) {
                translate([horn_r, 0, carrier_top - index_pin_len]) e_pin();
                translate([horn_r, 0, -keeper_t2]) e_keeper_ring();
            }
        }
    } else if (part == "carrier") {
        e_carrier();
    } else if (part == "clevis") {
        translate([0, 0, e_bore_z - e_rail_top + 3]) e_clevis();
    } else if (part == "bushing") {
        e_bushing();
    } else if (part == "eye_bushing") {
        e_eye_bushing();
    } else if (part == "pin") {
        e_hinge_pin();
    } else if (part == "keeper") {
        e_keeper();
    } else if (part == "index_pin") {
        e_pin();
    } else if (part == "index_keeper") {
        e_keeper_ring();
    } else {
        assert(false, str("E: unknown part ", part));
    }
}

// =============================================================================
// Public placement contract
// =============================================================================
// Arm-local station of the rod eye's bore centre.
function variant_rod_eye_local(variant = horn_variant,
                               radius_mm = horn_radius_default_mm) =
    variant == "C" ? [radius_mm, 0] :
    variant == "B" ? [horn_r, -b_lug_offset] :
    variant == "D" ? [horn_r, 0] :
    variant == "E" ? [e_radius, 0] :
    [horn_r, 0];

// Height of the rod eye's bore centre above the horn underside plane.
function variant_rod_eye_z(variant = horn_variant) =
    variant == "A" ? horn_t + post_flange_t + post_spigot_len / 2 :
    variant == "B" ? horn_t + carrier_t + post_flange_t + b_lug_rise + post_spigot_len / 2 :
    variant == "C" ? horn_t + c_shoe_post_z0 + post_flange_t
                    + post_spigot_len / 2 :
    variant == "D" ? horn_t + d_post_flange_z + post_flange_t + post_spigot_len / 2 :
    e_bore_z;

// World plan position of the rod eye's bore centre.
function variant_rod_eye_xy(variant = horn_variant, radius_mm = horn_radius_default_mm) =
    let(p = variant_rod_eye_local(variant, radius_mm))
    [p[0] * cos(horn_arm_angle_deg) - p[1] * sin(horn_arm_angle_deg),
     p[0] * sin(horn_arm_angle_deg) + p[1] * cos(horn_arm_angle_deg)];

// Bore centre of the rod eye in the frame described at the top of this file.
function variant_rod_eye_center(variant = horn_variant,
                                radius_mm = horn_radius_default_mm) =
    [variant_rod_eye_xy(variant, radius_mm)[0],
     variant_rod_eye_xy(variant, radius_mm)[1],
     variant_rod_eye_z(variant)];

// The rod eye bore axis in world coordinates: vertical.
function variant_rod_eye_axis(variant = horn_variant) = [0, 0, 1];

// Effective plan radius of the joint (the distance from the shaft axis that sets
// the available lateral throw).  See the LATERAL THROW note at the top.
function variant_joint_radius(variant = horn_variant,
                              radius_mm = horn_radius_default_mm) =
    variant == "C" ? radius_mm :
    variant == "E" ? e_radius :
    variant == "B" ? sqrt(horn_r * horn_r + b_lug_offset * b_lug_offset) :
    horn_r;

// =============================================================================
// Dispatcher
// =============================================================================
// The selector names are DISTINCT from the module parameter names, so
// `-D horn_variant="B"` cannot be shadowed by a module signature default.
horn_variant = "A";
horn_part = "assembly";

module throttle_horn_variant(variant = horn_variant, part = horn_part,
                             radius_mm = horn_radius_default_mm) {
    if (variant == "A") variant_a(part);
    else if (variant == "B") variant_b(part);
    else if (variant == "C") variant_c(part, radius_mm = radius_mm);
    else if (variant == "D") variant_d(part);
    else if (variant == "E") variant_e(part);
    else assert(false, str("unknown variant: ", variant));
}

throttle_horn_variant(horn_variant, horn_part);
