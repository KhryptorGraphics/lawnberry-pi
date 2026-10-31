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
//   * D's ball is retained in BOTH directions by GEOMETRY, not by preload: the
//     cage's seat bore is 8.6 mm across, narrower than the 10 mm ball, so the
//     ball cannot pass downwards; and the independent closure's roof hole is
//     8 mm across, also narrower than the ball, so it cannot pass upwards.  The
//     wear socket between them is a separate split shell that carries the wear
//     surface and can lose its grip without the ball escaping.  The articulation
//     is limited by the socket's split and the printed ball's elasticity, both
//     TO MEASURE.
//   * C's wedge key is a self-locking tapered pin in the shoe's tongue; the shoe
//     is held against the groove wall by friction, and the slot that admits the
//     key runs the whole length of the rail, so the radius is continuous and not
//     indexed.  The holding torque is TO MEASURE; a printed wedge in plastic is
//     weak in shear.
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
//   C: carrier, shoe, wedge_key, wedge_keeper, cap, pin, keeper
//   D: carrier, ball_stem, cage, wear_socket, closure, cap, pin, keeper
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
// Mechanism: an X-shaped carrier lies on the horn's arms and is held down by two
// index pins in two of the horn's provisional holes.  It carries a radial rail
// running outboard along the first arm.  The rail has a T groove: a wide chamber
// at its base and a narrower neck above it, so the shoe's tongue is captive
// vertically and can only slide ALONG the rail.  The adjustment is CONTINUOUS:
// there is no hole, notch or detent anywhere in it.
//
// The shoe is locked by a printed wedge key, a self-locking tapered pin driven
// transversely through a CONTINUOUS longitudinal slot in the rail's neck wall
// and into a blind hole in the shoe's tongue.  The key's taper presses the
// tongue against the opposite groove wall, so the shoe is held by friction and
// cannot creep along the rail.  Because the slot runs the whole length of the
// rail, the radius is not indexed: the key is simply driven at the radius wanted.
//
// World stack above the horn underside plane: carrier 2..5, rail 5..13, groove
// chamber 5..8.5, groove neck 8.5..13, shoe tongue 5..13, shoe body 13..17,
// post flange 17..18.5, eye 18.5..22.5, cap 22.5..26.5, post lip 26.5..28.
//
// LIMITATION: the rail's outboard end is a capacity; it needs a real horn arm
// that reaches at least horn_radius_max_mm from the axis.  The carrier must clear
// the servo body below the horn.  The wedge key's holding torque is TO MEASURE.
c_tongue_len = 16;
c_rail_x0 = horn_radius_min_mm - c_tongue_len / 2 - 0.5;
c_rail_x1 = horn_radius_max_mm + c_tongue_len / 2 + 0.5;
c_rail_w = 12;                      // rail width across
c_rail_h = 8;                       // rail height above the carrier
c_chamber_w = 8.4;                  // the groove's wide chamber
c_chamber_h = 3.5;                  // chamber height
c_neck_w = 5.6;                     // the groove's narrow neck
c_tongue_base_w = 8.0;              // shoe tongue in the chamber (0.2 per side)
c_tongue_neck_w = 5.0;              // shoe tongue in the neck (0.3 per side)
c_key_tip_d = 2.4;                  // the wedge key's leading taper
c_key_shaft_d = 3.0;                // its shaft
c_key_taper_len = 8;                // taper length, from the tip to the shaft
c_key_groove_d = 1.8;               // its groove, for the keeper ring
c_key_len = 10;                     // total length
c_key_axis_z = 10.5;                // world height of the key's axis
c_key_tip_y = -0.7;                 // world Y of the key's tip
c_slot_h = 3.1;                     // the rail slot's height

module c_carrier() {
    difference() {
        union() {
            horn_carrier(t = carrier_t, pin_arms = [1, 2]);
            // Radial rail running outboard along the first arm.
            translate([c_rail_x0, -c_rail_w / 2, carrier_t])
                cube([c_rail_x1 - c_rail_x0, c_rail_w, c_rail_h]);
        }
        // The T groove: a wide chamber at the rail's base and a narrow neck above
        // it.  The shoe's tongue fills both, so it cannot lift out of the rail.
        translate([c_rail_x0 - 1, -c_chamber_w / 2, carrier_t])
            cube([c_rail_x1 - c_rail_x0 + 2, c_chamber_w, c_chamber_h]);
        translate([c_rail_x0 - 1, -c_neck_w / 2, carrier_t + c_chamber_h])
            cube([c_rail_x1 - c_rail_x0 + 2, c_neck_w, c_rail_h - c_chamber_h]);
        // Continuous longitudinal slot through the rail's +Y neck wall, through
        // which the wedge key is driven.  The slot runs the whole length of the
        // rail, so the radius is not indexed.
        translate([c_rail_x0 + 1, c_neck_w / 2 - 0.4,
                   c_key_axis_z - horn_t - c_slot_h / 2])
            cube([c_rail_x1 - c_rail_x0 - 2, 4.4, c_slot_h]);
    }
}

module c_shoe(radius_mm = horn_radius_default_mm) {
    // Sliding shoe: a stepped tongue in the T groove, a body above the rail, the
    // wedge key's blind hole through the tongue, and the pivot post for the rod
    // eye.
    // Frame: z = 0 is the groove's chamber floor.
    difference() {
        union() {
            // Tongue: wide base in the chamber, narrow neck above it.
            translate([radius_mm - c_tongue_len / 2, -c_tongue_base_w / 2, 0])
                cube([c_tongue_len, c_tongue_base_w, c_chamber_h]);
            translate([radius_mm - c_tongue_len / 2, -c_tongue_neck_w / 2, c_chamber_h])
                cube([c_tongue_len, c_tongue_neck_w, c_rail_h - c_chamber_h]);
            // Shoe body above the rail.
            translate([radius_mm - 9, -9, c_rail_h]) cube([18, 18, 4]);
            // Pivot post on the shoe's top face.
            translate([radius_mm, 0, c_rail_h + 4]) cylinder(d = post_flange_d,
                                                           h = post_flange_t);
            translate([radius_mm, 0, c_rail_h + 4 + post_flange_t])
                cylinder(d = post_spigot_d, h = post_spigot_len);
            translate([radius_mm, 0, c_rail_h + 4 + post_flange_t + post_spigot_len])
                cylinder(d = post_cap_seat_d, h = post_cap_seat_len);
            translate([radius_mm, 0, c_rail_h + 4 + post_flange_t + post_spigot_len
                                     + post_cap_seat_len])
                cylinder(d = post_lip_d, h = post_lip_t);
        }
        // The wedge key's blind hole, entered from the tongue's +Y side.
        translate([radius_mm, 0.9, c_key_axis_z - horn_t - carrier_t])
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
            // c_shoe() already builds its post at x = radius_mm in its own frame; an extra
            // radius translation here put the post at twice the radius, off the eye datum.
            translate([0, 0, horn_t + carrier_t]) c_shoe(radius_mm);
            // Wedge key, driven from the rail's +Y side into the shoe's tongue.
            translate([radius_mm, c_key_tip_y, c_key_axis_z])
                rotate([-90, 0, 0]) c_wedge_key();
            translate([radius_mm, c_key_tip_y + c_key_len - keeper_t2, c_key_axis_z])
                rotate([-90, 0, 0]) c_wedge_keeper();
            for (k = [1, 2]) rotate([0, 0, 90 * k]) {
                translate([horn_r, 0, carrier_top - index_pin_len]) c_pin();
                translate([horn_r, 0, -keeper_t2]) c_keeper();
            }
            translate([radius_mm, 0,
                       horn_t + carrier_t + c_rail_h + 4 + post_flange_t
                       + post_spigot_len]) c_cap();
        }
    } else if (part == "carrier" || part == "carriage") {
        // "carriage" is accepted as an alias: fit_checks.scad probes the carriage by
        // that name, and an unmatched selector would render nothing and report a
        // false pass.
        c_carrier();
    } else if (part == "shoe") {
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
// D - horn-supported printed ball in a structural cage with a replaceable liner
// =============================================================================
// Mechanism: the carrier's first arm carries a post that rises above the horn
// to carry the printed ball, whose flat boss stands on the post's top.  The CAGE
// is the structural socket: a tube with a seat at its bottom whose bore is
// 8.6 mm across, NARROWER than the 10 mm ball, so the ball cannot pass
// downwards; a side bridge off the cage carries the pivot post for the rod eye,
// clear of the ball's swing.  The WEAR SOCKET is a separate split shell pressed
// over the ball's upper flank inside the cage's bore: it carries the wear and
// can be replaced.  An INDEPENDENT printed closure caps the cage and its own roof
// hole is 8 mm across, also narrower than the ball, so the ball cannot pass
// upwards.  The ball is therefore retained in both directions by GEOMETRY alone:
// losing the wear socket's grip cannot let it escape.
//
// World stack above the horn underside plane: carrier 2..5, post 2..17.5,
// ball boss 17.5..22.5, ball 17.5..27.5, cage 14..27 (seat top 17.5),
// wear socket 17.5..24.5, closure 27..31, pivot bridge 14..18,
// pivot post flange 18..19.5, eye 19.5..23.5, cap 23.5..27.5.
//
// LIMITATION: the ball is captured between an 8.6 mm seat and an 8 mm roof
// hole, so the articulation is limited to the wear socket's split and the
// printed ball's own elasticity.  The ball's stiffness, the socket's wear life
// and the closure's retention are all TO MEASURE.
d_ball_z = 22.5;      // ball centre above the horn underside plane
d_ball_d = ball_d_mm; // 10
d_boss_d = 8;         // the ball's flat boss, which stands on the carrier's post.
                      // It is NARROWER than the cage's seat bore (8.6), so the boss
                      // passes through the seat's bore with real clearance instead of
                      // sharing a coincident cylindrical face with it.
d_post_h = 15.5;      // the carrier's post height above the carrier's underside
d_cage_bottom = 14;   // the cage's bottom face
d_cage_seat_z = 17.5;  // the seat's top face, where the ball rests
d_cage_seat_bore_d = 8.6;   // narrower than the ball: it cannot pass downwards
d_cage_top = 27;      // the cage's top face
d_cage_od = 18;
d_cage_bore_d = 13.2; // the cage's upper bore, which takes the wear socket
d_socket_od = 13.0;  // the wear socket's outside, nominally a slip fit in that bore
d_socket_id = 10.1;   // the wear socket's bore, 0.1 clearance on the ball
d_socket_h = 7;
d_closure_od = 20;
d_closure_cavity_d = 14;
d_closure_cavity_h = 2;
d_closure_roof_t = 2;
d_closure_hole_d = 8; // narrower than the ball: it cannot pass upwards
// The pivot station, offset from the ball station. The rod eye plate (z 19.7..23.7) lies
// inside the cage's height band, and the plate's 6 mm tail behind the bore reaches 8.49 mm
// from the bore at its corners. At 14 that tail swung into the 18 mm cage; at 19 the tail
// clears the cage at any swing angle by 1.5 mm. The plate's forward run toward the tube is
// NOT covered by that bound: linkage_servo_end_fit checks it at the demonstration pose.
d_pivot_offset = 19;

module d_carrier() {
    difference() {
     union() {
        horn_carrier(t = carrier_t, pin_arms = [1, 2]);
        // Post at the first arm carrying the ball.  It is offset by half the boss's
        // diameter in Y so that the post's top circle and the boss's bottom circle
        // OVERLAP substantially rather than meeting on a shared plane: two solids
        // that merely touch on a coincident face do not union into a valid
        // 2-manifold, and this variant rendered as "not a valid 2-manifold" with the
        // post coaxial with the boss.
        translate([horn_r, -d_boss_d / 2, 0])
            cylinder(d = d_boss_d, h = d_post_h);
     }
    }
}

module d_ball_stem() {
    // Printed ball with a flat boss that stands on the carrier's post.
    // Frame: the boss's lower end at z = 0, so the part exports bed-zero.
    //
    // The boss meets the carrier's post with a 0.1 mm press interference, because the
    // post is 0.1 mm taller than the gap to the ball's equator (see d_post_h).  Two
    // solids that merely touch on a shared plane do not union into a valid 2-manifold:
    // the coincident faces produced a "not a valid 2-manifold" mesh for this variant.
    union() {
        cylinder(d = d_boss_d, h = d_ball_z - d_cage_seat_z);
        translate([0, 0, d_ball_z - d_cage_seat_z + 0.25])
            rotate([0, 0, 3.1]) sphere(d = d_ball_d);
    }
}

module d_cage() {
    // Structural socket: the tube, its bottom seat whose bore is narrower than the
    // ball, and the side bridge that carries the pivot post.
    // Frame: z = 0 is the cage's bottom face.
    seat = d_cage_seat_z - d_cage_bottom;   // the seat's height above the bottom
    difference() {
        union() {
            cylinder(d = d_cage_od, h = d_cage_top - d_cage_bottom);
            // Side bridge out to the pivot station, clear of the ball's swing.
            translate([-5, -d_pivot_offset - 5, 0]) cube([10, d_pivot_offset + 5, 4]);
            // Pivot post for the rod eye, standing on the bridge.
            translate([0, -d_pivot_offset, 4]) cylinder(d = post_flange_d,
                                                        h = post_flange_t);
            translate([0, -d_pivot_offset, 4 + post_flange_t])
                cylinder(d = post_spigot_d, h = post_spigot_len);
            translate([0, -d_pivot_offset, 4 + post_flange_t + post_spigot_len])
                cylinder(d = post_cap_seat_d, h = post_cap_seat_len);
            translate([0, -d_pivot_offset, 4 + post_flange_t + post_spigot_len
                                     + post_cap_seat_len])
                cylinder(d = post_lip_d, h = post_lip_t);
        }
        // The seat's bore.  It is narrower than the ball (8.6 against 10), so the ball
        // cannot pass downwards, and it is wider than the ball's Ø6 boss, so the boss
        // passes through it with 1.3 mm of clearance per side: the two do not share a
        // cylindrical face, which would make the union non-manifold.
        translate([0, 0, -1]) cylinder(d = d_cage_seat_bore_d, h = seat + 1);
        // The upper bore, which takes the wear socket.
        translate([0, 0, seat - 0.01]) cylinder(d = d_cage_bore_d,
                                                h = d_cage_top - d_cage_bottom - seat + 1);
    }
}

module d_wear_socket() {
    // Replaceable split wear socket: a shell pressed over the ball's upper flank
    // and into the cage's bore.  It is the sacrificial wear part: it is replaced
    // when the clearance grows.
    // Frame: z = 0 is the socket's bottom face, which rests on the cage's seat.
    difference() {
        cylinder(d = d_socket_od, h = d_socket_h);
        // The ball's clearance bore.
        translate([0, 0, -1]) cylinder(d = d_socket_id, h = d_socket_h + 2);
        // Split slot, cut part way through the wall only, so the socket can be
        // pressed sideways over the ball.
        translate([-1.2, -8, 0]) cube([2.4, 16, 4]);
    }
}

module d_closure() {
    // Independent captive closure: it caps the cage's bore and its own roof hole
    // is 8 mm across, narrower than the 10 mm ball, so the ball cannot escape
    // upwards even if the wear socket lost its grip entirely.
    // Frame: z = 0 is the closure's bottom face, which lands on the cage's top.
    difference() {
        union() {
            cylinder(d = d_closure_od, h = d_closure_cavity_h + d_closure_roof_t);
            translate([0, 0, d_closure_cavity_h]) cylinder(d = d_closure_od,
                                                            h = d_closure_roof_t);
        }
        // The cavity the ball sits in.
        translate([0, 0, -1]) cylinder(d = d_closure_cavity_d,
                                       h = d_closure_cavity_h + 1);
        // The roof's hole, narrower than the ball.
        translate([0, 0, d_closure_cavity_h - 0.01])
            cylinder(d = d_closure_hole_d, h = d_closure_roof_t + 0.02);
    }
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
            // Ball standing on the carrier's post.
            translate([horn_r, 0, d_cage_seat_z]) d_ball_stem();
            // Cage around the ball, with its own pivot post.
            translate([horn_r, 0, d_cage_bottom]) d_cage();
            // Wear socket pressed over the ball, inside the cage's bore.
            translate([horn_r, 0, d_cage_seat_z]) d_wear_socket();
            // Closure capping the cage.
            translate([horn_r, 0, d_cage_top]) d_closure();
            for (k = [1, 2]) rotate([0, 0, 90 * k]) {
                translate([horn_r, 0, carrier_top - index_pin_len]) d_pin();
                translate([horn_r, 0, carrier_top - index_pin_len + index_groove_len / 2])
                    d_keeper();
            }
            // Cap over the rod eye, on the cage's pivot post.
            translate([horn_r, -d_pivot_offset, d_cage_bottom + 4 + post_flange_t
                       + post_spigot_len]) d_cap();
        }
    } else if (part == "carrier") {
        d_carrier();
    } else if (part == "ball_stem") {
        d_ball_stem();
    } else if (part == "cage") {
        d_cage();
    } else if (part == "wear_socket") {
        d_wear_socket();
    } else if (part == "closure") {
        d_closure();
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
    variant == "D" ? [horn_r, -d_pivot_offset] :
    variant == "E" ? [e_radius, 0] :
    [horn_r, 0];

// Height of the rod eye's bore centre above the horn underside plane.
function variant_rod_eye_z(variant = horn_variant) =
    variant == "A" ? horn_t + post_flange_t + post_spigot_len / 2 :
    variant == "B" ? horn_t + carrier_t + post_flange_t + b_lug_rise + post_spigot_len / 2 :
    variant == "C" ? horn_t + carrier_t + c_rail_h + 4 + post_flange_t
                    + post_spigot_len / 2 :
    variant == "D" ? d_cage_bottom + 4 + post_flange_t + post_spigot_len / 2 :
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
    variant == "D" ? sqrt(horn_r * horn_r + d_pivot_offset * d_pivot_offset) :
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
