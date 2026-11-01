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
// Contract (the signature and the overall length are relied on by the callers):
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
/// the pin and at the band's centre.
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
