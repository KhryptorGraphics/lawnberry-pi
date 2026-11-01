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
//   * z = 0 is the KEEPER end, the pin's FREE end: a full-diameter LIP occupies
//     z = 0 .. keeper_lip_t_mm, then the keeper GROOVE (turned down to keeper_band_d_mm)
//     runs keeper_band_t_mm, then the full shaft;
//   * the HEAD is at the far end: the last keeper_t_mm of the pin carries the head
//     flange, which bears on the outer face of the part it retains.
//
// The split keeper ring's bore is SMALLER than both the shaft and the lip, so once it is
// snapped into the groove it is trapped axially between the lip and the shaft shoulder:
// it can leave neither way without being spread open again. The head flange bears on the
// far part's outer face. Snap force, spring-back and fatigue of the printed split ring are
// unqualified.

// The keeper groove's axial width and the ring that fits it.
keeper_lip_t_mm = 1;           // full-diameter lip at the pin's free end
keeper_ring_t_mm = 2;          // the ring's own thickness
keeper_band_t_mm = keeper_ring_t_mm + 0.2; // the groove: the ring plus 0.1 mm per shoulder
keeper_band_d_mm = 4.4;        // the groove's core diameter
keeper_ring_bore_d_mm = 4.6;   // the ring rides IN the groove with 0.1 mm of clearance
                               // per side, so the two do not share a coincident face.
                               // It is SMALLER than the 5 mm shaft and lip, so the ring
                               // cannot pass over either.
keeper_ring_od_mm = 12;

// For callers that `use` this file and so cannot see its variables.
function keeper_lip_t() = keeper_lip_t_mm;
// Height of the installed ring's lower face on printed_pin: centred in the groove.
function keeper_ring_seat_z() = keeper_lip_t_mm + (keeper_band_t_mm - keeper_ring_t_mm) / 2;

/// Relief that cuts a keeper GROOVE into a pin's outside.
///
/// Subtract this annulus from a solid shaft whose free end is at z = 0: it leaves a
/// full-diameter lip over z = 0 .. lip_t and a groove of `band_d` over lip_t .. lip_t +
/// band_t. The outside extends past the shaft so subtraction leaves no coincident faces,
/// and nothing reaches below z = 0, so the pin still exports bed-zero.
module pin_band_relief(d = pin_d_mm, band_d = keeper_band_d_mm,
                       band_t = keeper_band_t_mm, lip_t = keeper_lip_t_mm) {
    assert(band_d < d, "The groove must be smaller than the shaft it turns down");
    assert(lip_t > 0, "The groove needs a full-diameter lip at the pin's free end");
    translate([0, 0, lip_t])
        difference() {
            cylinder(d = d + 2, h = band_t);
            translate([0, 0, -1]) cylinder(d = band_d, h = band_t + 2, $fn = 48);
        }
}

module printed_pin(len, d = pin_d_mm, head_d = pin_head_d_mm, keeper = true) {
    assert(len > keeper_lip_t_mm + keeper_band_t_mm + keeper_t_mm,
           "Pin must span both ears and still carry its keeper lip and groove");
    difference() {
        union() {
            cylinder(d = d, h = len, $fn = 48);
            translate([0, 0, len - keeper_t_mm]) cylinder(d = head_d, h = keeper_t_mm);
        }
        if (keeper) pin_band_relief(d = d);
    }
}

/// Keeper ring: printed SEPARATELY so the pin can be inserted before capture.
///
/// A split ring whose bore is smaller than the pin's shaft and lip: it is spread over the
/// lip (or opened onto the groove) and snaps into the groove, where the lip and the shaft
/// shoulder trap it. The split lets it be fitted without a bought tool. It is the same
/// capture the horn variants use for their index pins.
///
/// The ring's axis is its own +Z, so it is placed on the pin with the SAME orientation as
/// the pin, at keeper_ring_seat_z().
module printed_keeper(od = keeper_ring_od_mm, bore = keeper_ring_bore_d_mm,
                      h = keeper_ring_t_mm) {
    assert(is_num(od) && is_num(bore) && is_num(h), "keeper dimensions must all be numbers");
    assert(bore < pin_d_mm, "The keeper's bore must be smaller than the shaft and lip");
    assert(bore >= keeper_band_d_mm,
           "The keeper's bore must clear the pin's keeper groove");
    assert(h <= keeper_band_t_mm, "The keeper must fit inside the groove's width");
    assert(od > bore + 2, "The keeper must have a wall around its bore");
    difference() {
        cylinder(d = od, h = h);
        translate([0, 0, -1]) cylinder(d = bore, h = h + 2, $fn = 48);
        // Radial split so the ring can be spread over the lip into the groove.
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
        // Relief windows on the nut's lower half (print/clean-out relief only; there is no
        // separate anti-backdrive lip).
        for (k = [0:3])
            rotate([0, 0, k * 90 + 45])
                translate([d / 2 + 5 - fine_nut_window_mm / 2, -1.2, -0.01])
                    cube([fine_nut_window_mm, 2.4, h * 0.5]);
    }
}

// ---- Helical sweep for print-in-place threads --------------------------------------
// Sweeps a convex `profile` of [r, z] points (counter-clockwise in the r-z plane, z about
// the helix centre) along a right-hand helix of `pitch`, from angle a0 to a1 (degrees,
// counter-clockwise about +Z; the centre rises pitch * a / 360). Built as one polyhedron,
// not a union of hulls: every side triangle contains a profile edge, so no facet is
// steeper than the profile flank it approximates (45-degree flanks stay 45 degrees for
// the print-in-place droop gate). The end caps lie in vertical profile planes.
module helical_sweep(profile, pitch, a0, a1, step_deg = 5) {
    n = max(1, ceil((a1 - a0) / step_deg));
    m = len(profile);
    pts = [for (i = [0:n])
               let(a = a0 + (a1 - a0) * i / n, rise = pitch * a / 360)
               for (p = profile) [p[0] * cos(a), p[0] * sin(a), p[1] + rise]];
    sides = [for (i = [0:n - 1]) for (k = [0:m - 1])
                 let(k1 = (k + 1) % m, A = i * m + k, B = i * m + k1,
                     C = (i + 1) * m + k1, D = (i + 1) * m + k)
                 each [[A, B, C], [A, C, D]]];
    caps = [[for (k = [m - 1:-1:0]) k], [for (k = [0:m - 1]) n * m + k]];
    polyhedron(points = pts, faces = concat(sides, caps), convexity = 4);
}
