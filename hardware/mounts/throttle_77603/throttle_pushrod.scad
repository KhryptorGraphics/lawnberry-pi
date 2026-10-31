// Segmented, completely printable throttle pushrod with continuous fine adjustment.
//
// Deliberately NOT a copy of ../pushrod.scad: that part uses indexed lock holes and steel M6
// fasteners, both forbidden for the throttle linkage. Only the square-tube proportions and
// the eye/clevis idea are reused. Every piece here is printable; no bought bolt, pin,
// insert, bearing or threaded rod is required.
include <parameters.scad>
use <printed_joints.scad>

// ---- Section -----------------------------------------------------------------------
rod_w = rods_w_mm;
rod_wall = rods_wall_mm;
rod_bore = rods_bore_mm;
rod_slider = rods_slider_mm;
rod_overlap = rods_min_overlap_mm;
rod_travel = rods_coarse_travel_mm;

// ---- Segment plan ------------------------------------------------------------------
// Straight lengths are computed from the requested assembly span so no piece exceeds the
// usable bed. Segment count is derived, never hand-chosen per length.
function segment_count(total_len) =
    max(3, ceil((total_len - rod_overlap) / (print_part_max_len_mm - 20)) + 1);
function segment_len(total_len, i, n) =
    (total_len - (n - 1) * rod_splice_len_here()) / n;
function rod_splice_len_here() = 40;   // keyed spigot engagement

module square_tube(len, w = rod_w, bore = rod_bore) {
    // Extruded 2D ring. A 3D cube-minus-cube boolean left coincident faces that exported as
    // degenerate triangles; a single profile extrusion is exact and has no such faces.
    assert(bore < w, "Bore must be smaller than the outside section");
    linear_extrude(height = len)
        difference() {
            square([w, w], center = true);
            square([bore, bore], center = true);
        }
}

module solid_slider(len, w = rod_slider) {
    // Key RIB added to the outside rather than a groove cut into it: an added rib cannot
    // create coincident faces, and it still prevents rotation without thinning the section.
    union() {
        translate([-w / 2, -w / 2, 0]) cube([w, w, len]);
        translate([-1.2, -w / 2 - 2.4, len * 0.1]) cube([2.4, 2.5, len * 0.8]);
    }
}

// ---- Lock mechanisms ---------------------------------------------------------------
// All three mount on the same sleeve/inner-bar skeleton and are interchangeable.

// 1. Captured tapered wedge driven by an over-centre cam, plus a separate keeper.
module lock_wedge_cam(part = "assembly") {
    body_l = wedge_len_mm;
    body_w = rod_w + 14;
    if (part == "assembly") {
        lock_wedge_cam("body");
        translate([0, 0, body_l * 0.55]) lock_wedge_cam("cam");
        translate([0, body_w / 2 + keeper_t_mm, body_l * 0.55]) lock_wedge_cam("keeper");
    } else if (part == "body") {
        // 2D cross-section with the rod passage and wedge channel, extruded once.
        linear_extrude(height = body_l)
            difference() {
                translate([-body_w / 2, -body_w / 2]) square([body_w, body_w]);
                translate([-rod_w / 2 - joint_clearance_mm,
                           -rod_w / 2 - joint_clearance_mm])
                    square([rod_w + 2 * joint_clearance_mm,
                            rod_w + 2 * joint_clearance_mm]);
                // Tapered wedge channel, inset from both ends of the X extent so it never
                // reaches a wall (a channel spanning the full width severed the body).
                polygon([[-body_w / 2 + 4, -keeper_t_mm / 2 - 1],
                         [body_w / 2 - 4, -keeper_t_mm / 2 - 1],
                         [body_w / 2 - 4, keeper_t_mm / 2 + 1],
                         [-body_w / 2 + 4, keeper_t_mm / 2 + 1
                          + body_l * tan(wedge_taper_deg)]]);
            }
    } else if (part == "cam") {
        // Over-centre lever with a visible locked position, laid flat on the bed.
        translate([0, 0, 7]) difference() {
            union() {
                rotate([90, 0, 0]) cylinder(d = 14, h = body_w - 4, center = true);
                rotate([90, 0, 0])
                    linear_extrude(height = 4, center = true)
                        polygon([[0, -3], [34, -1], [34, 6], [0, 5]]);
            }
            rotate([90, 0, 0]) cylinder(d = pin_d_mm, h = body_w, center = true);
        }
    } else if (part == "keeper") {
        printed_keeper();
    } else {
        assert(false, str("unknown wedge part: ", part));
    }
}

// 2. Split collet compressed by a large printed threaded collar, captive anti-unwind lug.
module lock_split_collet(part = "assembly") {
    body_l = 72;
    if (part == "assembly") {
        lock_split_collet("body");
        translate([0, 0, body_l * 0.72]) lock_split_collet("collar");
        translate([0, rod_w / 2 + 12, body_l * 0.72]) lock_split_collet("lug");
    } else if (part == "body") {
        // Ring cross-section with outward slit notches, extruded once.
        linear_extrude(height = body_l)
            difference() {
                circle(d = rod_w + 16);
                circle(d = rod_w + 2 * joint_clearance_mm);
                // Flexure slits strictly INSIDE the ring wall: touching the bore edge made
                // the profile self-intersecting (nonmanifold export).
                for (k = [0:3])
                    rotate([0, 0, k * 90 + 22.5])
                        translate([rod_w / 2 + joint_clearance_mm + 1.2,
                                   -collet_slit_mm / 2])
                            square([3.5, collet_slit_mm]);
            }
    } else if (part == "collar") {
        // Clamp ring: 2D ring with inward grip teeth, extruded once.
        linear_extrude(height = 20)
            difference() {
                circle(d = rod_w + 26);
                circle(d = rod_w + 16.4);
                for (k = [0:5])
                    rotate([0, 0, k * 60])
                        translate([rod_w / 2 + 11, -1.2]) square([4, 2.4]);
            }
    } else if (part == "lug") {
        printed_keeper();
    } else {
        assert(false, str("unknown collet part: ", part));
    }
}

// 3. Printed fine-pitch lead screw with travelling nut and anti-backdrive lip.
module lock_lead_screw(part = "assembly") {
    screw_len = 44;
    if (part == "assembly") {
        lock_lead_screw("screw");
        translate([0, 0, fine_nut_height_mm]) lock_lead_screw("nut");
        translate([fine_screw_d_mm / 2 + 6, 0, fine_nut_height_mm * 0.5])
            lock_lead_screw("lip");
        translate([0, 0, screw_len - fine_jam_collar_h_mm]) lock_lead_screw("jam");
    } else if (part == "screw") {
        printed_screw(44);
    } else if (part == "nut") {
        printed_nut();
    } else if (part == "lip") {
        anti_backdrive_lip();
    } else if (part == "jam") {
        linear_extrude(height = fine_jam_collar_h_mm)
            difference() {
                circle(d = fine_screw_d_mm + 12);
                circle(d = fine_screw_d_mm - 1.5);
            }
    } else {
        assert(false, str("unknown lead-screw part: ", part));
    }
}

module throttle_lock(kind = "wedge_cam", part = "assembly") {
    if (kind == "wedge_cam") lock_wedge_cam(part);
    else if (kind == "split_collet") lock_split_collet(part);
    else if (kind == "lead_screw") lock_lead_screw(part);
    else assert(false, str("unknown lock kind: ", kind));
}

// ---- Shared sleeve assembly --------------------------------------------------------
// Coarse stage: the outer sleeve slides continuously on the keyed slider and is locked by
// the selected mechanism. Fine stage: the printed lead screw gives sub-millimetre trim.
// A hard end-stop collar makes separation impossible even with every lock released.
module rod_sleeve(len = 200, lock_kind = "wedge_cam") {
    // The sleeve is ONE printed solid: a tube with a BLIND lock pocket. An open window plus a
    // separately placed lock assembly produced detached shells, and the lock is a separate
    // printable part in its own right (see throttle_lock), not part of the sleeve.
    difference() {
        square_tube(len);
        // Blind pocket on one face so the lock acts on the slider; never through-both-faces.
        translate([-rod_w / 2 - 1, -rod_w / 2 + rod_wall * 0.6, len * 0.35])
            cube([rod_w * 0.55, rod_w * 0.4, len * 0.45]);
        // Keeper land for the lock's secondary retainer.
        translate([-keeper_t_mm / 2, -rod_w / 2 - 1, len * 0.78])
            cube([keeper_t_mm, rod_wall, keeper_t_mm]);
    }
}

module rod_endstop() { endstop_collar(); }

// ---- Public assembly ---------------------------------------------------------------
rod_part = "assembly";
rod_len = rods_nominal_span_mm;
rod_lock_kind = "lead_screw";

module throttle_pushrod(part = rod_part, length_mm = rod_len,
                        lock_kind = rod_lock_kind) {
    n = segment_count(length_mm);
    seg = segment_len(length_mm, 0, n);
    sleeve_len = 220;
    if (part == "assembly") {
        // Servo-end segment with a clevis eye.
        translate([0, 0, 0]) square_tube(seg);
        // Intermediate keyed segments.
        for (i = [1:n - 2])
            translate([0, 0, i * seg]) square_tube(seg);
        // Sleeve carrying the selected lock, then the fine trim.
        translate([0, 0, (n - 1) * seg - 30]) rod_sleeve(min(sleeve_len, 200), lock_kind);
        // End-stop collar and the fine-trim stack.
        translate([0, 0, (n - 1) * seg + 20]) rod_endstop();
        translate([0, 0, (n - 1) * seg + 40])
            throttle_lock(kind = "lead_screw", part = "assembly");
    } else if (part == "segment_lower") {
        square_tube(min(seg, print_part_max_len_mm));
    } else if (part == "segment_middle") {
        square_tube(min(seg, print_part_max_len_mm));
    } else if (part == "segment_upper") {
        square_tube(min(seg, print_part_max_len_mm));
    } else if (part == "slider") {
        solid_slider(min(rod_travel + rod_overlap * 2, print_part_max_len_mm));
    } else if (part == "sleeve") {
        rod_sleeve(min(220, print_part_max_len_mm), lock_kind);
    } else if (part == "spigot") {
        keyed_spigot();
    } else if (part == "splice_wedge") {
        splice_wedge();
    } else if (part == "endstop") {
        rod_endstop();
    } else if (part == "lock_wedge_body") {
        lock_wedge_cam("body");
    } else if (part == "lock_wedge_cam") {
        lock_wedge_cam("cam");
    } else if (part == "lock_collet_body") {
        lock_split_collet("body");
    } else if (part == "lock_collet_collar") {
        lock_split_collet("collar");
    } else if (part == "lock_screw") {
        lock_lead_screw("screw");
    } else if (part == "lock_nut") {
        lock_lead_screw("nut");
    } else if (part == "lock_anti_backdrive") {
        lock_lead_screw("lip");
    } else if (part == "lock_jam_collar") {
        lock_lead_screw("jam");
    } else if (part == "clevis") {
        printed_clevis();
    } else if (part == "clevis_bushing") {
        printed_bushing();
    } else if (part == "clevis_pin") {
        printed_pin(2 * clevis_ear_t_mm + clevis_gap_mm + 2 * keeper_t_mm);
    } else if (part == "gauge") {
        // Sliding-fit coupon: a plate with a closed square passage, extruded from a 2D ring
        // so it is one clean solid with no coincident faces.
        linear_extrude(height = 14)
            difference() {
                translate([-20, -12]) square([40, 24]);
                translate([-rod_slider / 2, -rod_slider / 2])
                    square([rod_slider, rod_slider]);
            }
    } else {
        assert(false, str("unknown pushrod part: ", part));
    }
}

throttle_pushrod(rod_part);
