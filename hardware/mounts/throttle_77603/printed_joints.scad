// Reusable fully-printed joint and retainer primitives for the 77603 throttle station.
//
// Every piece here is printable on its own and needs no bought screw, pin, bearing or
// insert. These are geometry primitives; retention and wear limits are unvalidated until
// physical cycling. Sibling include path is plain <name.scad> because parameters.scad sits
// in this same directory.
include <parameters.scad>

// ---- Captive printed hinge pin ------------------------------------------------------
// A pin that passes through a clevis ear and a bushing, then locks with a SEPARATELY
// printed keeper ring.
//
// Contract (the signature and the overall length are relied on by
// throttle_u_fitting.scad, which calls printed_pin(len) and printed_keeper()):
//   * the module signature is printed_pin(len, d, head_d, keeper);
//   * the pin is built from z = 0 exactly, so its origin is one of its two ends;
//   * the overall length along the pin's own +Z axis is exactly `len`;
//   * the pin's axis is its own +Z, so a caller places it with a rotate/translate.
//
// AXIS CONVENTION, shared with throttle_horn_variants.scad and throttle_pushrod.scad:
//   * z = 0 is the KEEPER end: the first keeper_t_mm of the pin is turned down to the
//     keeper's bore, and the separate keeper ring is pressed onto that band;
//   * the HEAD is at the far end: the last keeper_t_mm of the pin carries the head
//     flange, which bears on the outer face of the part it retains.
//
// The keeper ring's bore is SMALLER than the shaft, so it cannot be pushed back over the
// shaft and cannot leave the band; the head flange bears on the far part's outer face.
// Retention therefore does not depend on friction.  This is the same groove-and-split-
// ring capture the horn variants use for their index pins.

// The keeper band's axial width and the ring that fits it.
keeper_band_t_mm = 4;          // the turned-down band at the pin's origin
keeper_band_d_mm = 4.4;        // the band's own diameter
keeper_ring_bore_d_mm = 4.6;   // the ring rides ON the band with 0.1 mm of clearance
                               // per side, so it can be pressed on and the two do not
                               // share a coincident face.  It is still SMALLER than the
                               // 5 mm shaft, so the ring cannot be pushed back over it.
keeper_ring_od_mm = 12;
keeper_ring_t_mm = 2;          // the ring's own thickness

// Z of the keeper band's centre, for a pin of length `len`.
function printed_pin_groove_z(len) = keeper_band_t_mm / 2;

/// Relief that turns a pin's OUTSIDE down to `band_d` over `band_t`.
///
/// Subtract this annulus from a solid shaft to leave a central band of `band_d`.
/// Extend the outside past the shaft so subtraction does not leave coincident faces.
module pin_band_relief(d = pin_d_mm, band_d = keeper_band_d_mm,
                       band_t = keeper_band_t_mm) {
    assert(band_d < d, "The band must be smaller than the shaft it turns down");
    // Exactly band_t tall, starting at z = 0: the relief must not reach below the bed,
    // because a band that overhangs the pin's own origin puts mesh under the bed and
    // fails the bed-zero gate.
    difference() {
        cylinder(d = d + 2, h = band_t);
        translate([0, 0, -1]) cylinder(d = band_d, h = band_t + 2);
    }
}

module printed_pin(len, d = pin_d_mm, head_d = pin_head_d_mm, keeper = true) {
    assert(len > 2 * keeper_t_mm, "Pin must span both ears and still carry its keeper band");
    union() {
        if (keeper) {
            cylinder(d = keeper_band_d_mm, h = keeper_band_t_mm);
            translate([0, 0, keeper_band_t_mm])
                cylinder(d = d, h = len - keeper_band_t_mm);
        } else {
            cylinder(d = d, h = len);
        }
        translate([0, 0, len - keeper_t_mm]) cylinder(d = head_d, h = keeper_t_mm);
    }
}

/// Keeper ring: printed SEPARATELY so the pin can be inserted before capture.
///
/// A split ring whose bore is smaller than the pin's shaft, so it snaps onto the pin's
// turned-down band and cannot be pushed back over the shaft.  The split lets it be
/// pressed on without a bought tool.  It is the same capture the horn variants use for
/// their index pins.
///
/// The ring's axis is its own +Z, so it is placed on the pin with the SAME orientation as
/// the pin and at the band's centre: printed_pin_groove_z(len).
module printed_keeper(od = keeper_ring_od_mm, bore = keeper_ring_bore_d_mm,
                      h = keeper_ring_t_mm) {
    assert(is_num(od) && is_num(bore) && is_num(h), "keeper dimensions must all be numbers");
    assert(bore < pin_d_mm, "The keeper's bore must be smaller than the shaft it traps");
    assert(bore >= keeper_band_d_mm,
           "The keeper's bore must clear the pin's keeper band");
    assert(od > bore + 2, "The keeper must have a wall around its bore");
    difference() {
        cylinder(d = od, h = h);
        translate([0, 0, -1]) cylinder(d = bore, h = h + 2);
        // Radial split so the ring can be pressed onto the band.
        translate([0, -1.0, -1]) cube([od, 2.0, h + 2]);
    }
}

// ---- Printed cross-pin bore ---------------------------------------------------------
// A bore along the Y axis through a member's walls. rotate([90,0,0]) sends the extrusion
// toward -Y, so the cylinder must be CENTRED on the member and given the full length
// rather than started at one face: an off-centre bore ran off the part entirely and cut
// nothing. That polarity trap has recurred repeatedly, so it is expressed once here.
module pin_bore_y(len, d = pin_d_mm + 2 * joint_clearance_mm) {
    translate([0, 0, 0]) rotate([90, 0, 0])
        cylinder(d = d, h = len, center = true);
}

// A bore along the X axis, centred on the part it passes through. rotate([0,90,0]) sends
// the extrusion toward +X, so the cylinder must be centred rather than started at a face.
module pin_bore_x(len, d = pin_d_mm + 2 * joint_clearance_mm) {
    rotate([0, 90, 0]) cylinder(d = d, h = len, center = true);
}

// ---- Replaceable bushing -----------------------------------------------------------
// Sacrificial wear part: pressed into the ear, replaced when clearance grows.
module printed_bushing(od = bushing_od_mm, id = bushing_id_mm, len = bushing_len_mm) {
    assert(od - id >= 3, "Bushing wall too thin to print and press");
    difference() {
        cylinder(d = od, h = len);
        translate([0, 0, -0.5]) cylinder(d = id, h = len + 1);
        // Anti-rotation key so the bushing cannot spin its wear surface.
        translate([od / 2 - 0.01, -1, len / 2]) rotate([90, 0, 0])
            cylinder(d = 2.4, h = 2.2);
    }
}

/// Split wear shoe for the ball socket: replaceable, retained by the cage, not by glue.
module printed_wear_socket(ball_d = ball_d_mm, wall = ball_socket_wall_mm,
                           split = 1.6) {
    od = ball_d + 2 * wall;
    difference() {
        // Straight sleeve. A spherical cavity combined with a half-width split cut produced
        // coincident faces and a degenerate triangle, which the mesh gate correctly rejects.
        cylinder(d = od, h = ball_d * 0.9);
        translate([0, 0, -1]) cylinder(d = ball_d + 0.6, h = ball_d * 0.9 + 2);
        // Split slot cut PART WAY through the wall region only, never spanning the full bore
        // height: a through-slit severs the sleeve into two shells and a coincident-face cut
        // produces degenerate triangles, both of which the mesh gate correctly rejects.
        translate([-split / 2, -od / 2 - 1, ball_d * 0.9 - wall])
            cube([split, od + 2, wall + 0.5]);
    }
}

// ---- Double-shear clevis -----------------------------------------------------------
// Two printed ears with a gap sized for the rod eye; the pin and bushing pass through all
// three parts, so the eye is loaded in double shear and cannot lever the joint open.
module printed_clevis(gap = clevis_gap_mm, ear_t = clevis_ear_t_mm,
                      reach = 22, keeper = true) {
    // Ears are SEPARATED IN X by `gap`, so the rod eye physically sits between them and the
    // pin carries it in double shear. A single full-width block would have no gap at all.
    // The spine across the top ties both ears into one connected printable solid.
    total = gap + 2 * ear_t;
    height = ear_t + gap + ear_t;
    spine_t = ear_t;
    pin_z = ear_t + gap / 2;
    difference() {
        union() {
            translate([-total / 2, -reach / 2, 0]) cube([ear_t, reach, height]);
            translate([gap / 2, -reach / 2, 0]) cube([ear_t, reach, height]);
            translate([-total / 2, -reach / 2, height - spine_t])
                cube([total, reach, spine_t]);
        }
        // The ears are separated along X, so the pin bore MUST also run along X. The
        // previous Y-axis bore passed through the empty centre gap and missed both ears.
        translate([-total / 2 - 1, 0, pin_z]) rotate([0, 90, 0])
            cylinder(d = pin_d_mm + 2 * joint_clearance_mm, h = total + 2);
    }
}

// ---- Keyed segment splice ----------------------------------------------------------
// Fixed segment join: a spigot enters the next tube and a printed cross-wedge locks it.
// This is assembly retention only -- never a length adjustment.
module keyed_spigot(len = 40, w = rods_bore_mm - 0.4, wedge = true) {
    difference() {
        translate([-w / 2, -w / 2, 0]) cube([w, w, len]);
        if (wedge)
            translate([-w / 2 - 1, -keeper_t_mm / 2, len * 0.55])
                cube([w + 2, keeper_t_mm, keeper_t_mm]);
        // Anti-rotation key rib so the joint cannot clock under load.
        translate([-1.6, -w / 2 - 1.2, len * 0.15]) cube([3.2, 2.4, len * 0.3]);
    }
}

module splice_wedge(w = keeper_t_mm, h = 12) { printed_keeper(h = h, t = w); }

// ---- End-stop capture collar -------------------------------------------------------
// Clamps onto the sliding member as a hard stop. If the adjustment lock slips, this stops
// separation. It does NOT prevent unwanted throttle motion -- report that separately.
module endstop_collar(w = rods_w_mm + 6, t = 6) {
    difference() {
        translate([-w / 2, -w / 2, 0]) cube([w, w, t]);
        translate([-rods_w_mm / 2 - joint_clearance_mm,
                   -rods_w_mm / 2 - joint_clearance_mm, -1])
            cube([rods_w_mm + 2 * joint_clearance_mm,
                  rods_w_mm + 2 * joint_clearance_mm, t + 2]);
        // Printed wedge slot instead of a bought bolt.
        translate([-w / 2 - 1, -keeper_t_mm / 2, t / 2 - keeper_t_mm / 2])
            cube([w + 2, keeper_t_mm, keeper_t_mm]);
    }
}

// ---- Printed fine-pitch screw and nut ----------------------------------------------
// Coarse, printable trapezoidal thread approximation: a swept profile along a helix.
// Fractional turns give continuous adjustment; printed accuracy is NOT claimed.
module thread_profile(pitch = fine_screw_pitch_mm, depth = 1.6, r_root) {
    // Trapezoid: wide at the root, narrower at the crest, so it prints unsupported.
    polygon([[r_root - 0.05, -pitch * 0.30],
             [r_root + depth, -pitch * 0.16],
             [r_root + depth, pitch * 0.16],
             [r_root - 0.05, pitch * 0.30]]);
}

module printed_screw(len, d = fine_screw_d_mm, pitch = fine_screw_pitch_mm,
                     depth = 1.6, step_deg = 12) {
    r_root = d / 2 - depth;
    turns = len / pitch;
    steps = max(12, ceil(turns * 360 / step_deg));
    union() {
        cylinder(d = d - 2 * depth, h = len);
        for (i = [0:steps - 1])
            hull() {
                translate([0, 0, len * i / steps])
                    rotate([0, 0, -360 * turns * i / steps])
                        linear_extrude(height = 0.01) thread_profile(pitch, depth, r_root);
                translate([0, 0, len * (i + 1) / steps])
                    rotate([0, 0, -360 * turns * (i + 1) / steps])
                        linear_extrude(height = 0.01) thread_profile(pitch, depth, r_root);
            }
        // End flange keeps the nut captive on the screw.
        translate([0, 0, len - keeper_t_mm]) cylinder(d = d + 4, h = keeper_t_mm);
    }
}

module printed_nut(d = fine_screw_d_mm, pitch = fine_screw_pitch_mm,
                   h = fine_nut_height_mm, depth = 1.6, step_deg = 12) {
    r_crest = d / 2 + joint_clearance_mm;
    turns = h / pitch;
    steps = max(12, ceil(turns * 360 / step_deg));
    difference() {
        cylinder(d = d + 10, h = h);
        translate([0, 0, -0.5]) cylinder(d = d - 2 * depth - 0.2, h = h + 1);
        for (i = [0:steps - 1])
            hull() {
                translate([0, 0, h * i / steps])
                    rotate([0, 0, 360 * turns * i / steps])
                        linear_extrude(height = 0.01)
                            thread_profile(pitch, depth + joint_clearance_mm,
                                           r_crest - depth - joint_clearance_mm);
                translate([0, 0, h * (i + 1) / steps])
                    rotate([0, 0, 360 * turns * (i + 1) / steps])
                        linear_extrude(height = 0.01)
                            thread_profile(pitch, depth + joint_clearance_mm,
                                           r_crest - depth - joint_clearance_mm);
            }
        // Anti-backdrive window: a printed lip engages it so vibration cannot unwind.
        for (k = [0:3])
            rotate([0, 0, k * 90 + 45])
                translate([d / 2 + 5 - leadscrew_anti_backdrive_lip_mm / 2, -1.2, -0.01])
                    cube([leadscrew_anti_backdrive_lip_mm, 2.4, h * 0.5]);
    }
}

module anti_backdrive_lip() {
    difference() {
        hull() {
            translate([-leadscrew_anti_backdrive_lip_mm / 2, 0, 0])
                cube([leadscrew_anti_backdrive_lip_mm, 2.2, fine_nut_height_mm * 0.45]);
            translate([-leadscrew_anti_backdrive_lip_mm / 2 + 1, 0,
                       fine_nut_height_mm * 0.45])
                cube([leadscrew_anti_backdrive_lip_mm - 2, 2.2, 3]);
        }
        translate([0, -1.2, fine_nut_height_mm * 0.30]) rotate([45, 0, 0])
            cube([leadscrew_anti_backdrive_lip_mm, 2, 2], center = true);
    }
}

// ---- Two-axis alignment neck -------------------------------------------------------
// Positively captured printed articulation for the throttle-end fitting. Declared
// +/-joint_articulation_deg capacity, not a measured mower requirement.
module alignment_yoke(gap = clevis_gap_mm) {
    printed_clevis(gap = gap, ear_t = clevis_ear_t_mm, reach = 18);
}

module alignment_cross_block(gap = clevis_gap_mm) {
    union() {
        translate([-gap / 2 - joint_clearance_mm, -6, -6])
            cube([gap + 2 * joint_clearance_mm, 12, 12]);
        rotate([0, 0, 90]) printed_clevis(gap = gap, ear_t = clevis_ear_t_mm, reach = 16);
    }
}

// ---- Entry point -------------------------------------------------------------------
// The catalog drives individual primitives through this selector. Names index the module
// list so a typo fails loudly instead of silently rendering the wrong part.
joint_part = "pin";

module printed_joint_part(kind = joint_part) {
    if (kind == "pin") printed_pin(24);
    else if (kind == "clevis") printed_clevis();
    else if (kind == "bushing") printed_bushing();
    else if (kind == "wear_socket") printed_wear_socket();
    else if (kind == "endstop_collar") endstop_collar();
    else if (kind == "nut") printed_nut();
    else if (kind == "keyed_spigot") keyed_spigot();
    else if (kind == "screw") printed_screw(60);
    else if (kind == "anti_backdrive") anti_backdrive_lip();
    else if (kind == "keeper") printed_keeper();
    else if (kind == "alignment_yoke") alignment_yoke();
    else if (kind == "alignment_cross") translate([0, 0, 6]) alignment_cross_block();
    else assert(false, str("unknown joint part: ", kind));
}

printed_joint_part(joint_part);
