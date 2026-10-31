// Throttle-end inverted-U capture and its printable opening gauge.
//
// The U is a CLEARANCE CAPTURE over the retained mower throttle lever, never a clamp.
//
// Axes matter here and were wrong in an earlier revision:
//   frame: +X right (left-right), +Y forward (front-back), +Z up
//   owner opening: 12.70 mm FRONT-BACK x 31.75 mm LEFT-RIGHT
//   => the void is 31.75 mm in X and 12.70 mm in Y
//   the retained lever is VERTICAL, so the U drops over it: mouth open at -Z, closed top at +Z
// An earlier version put the mouth on -Y, which can only be engaged by sliding in
// horizontally and can never drop over a vertical lever.
//
// Built from two extrusions (outer prism minus inner void). A 2D profile with folded-in
// chamfers and a post-extrusion blind bore both produced non-closed/nonmanifold meshes.
include <parameters.scad>
include <printed_joints.scad>

u_depth = throttle_u_inner_depth_mm + fit_clearance_front_mm;  // Y, front-back
u_width = throttle_u_inner_width_mm + fit_clearance_side_mm;   // X, left-right
u_wall = throttle_u_wall_mm;
u_height = throttle_u_engagement_mm;
u_top = u_wall;                                               // closed top thickness
u_lean = throttle_lean_right_deg;
u_yaw = throttle_yaw_deg;
u_pitch = throttle_pitch_deg;

module u_body() {
    // Outer prism from the bed to the top face; inner void from below the bed to just under
    // the top, so the mouth is open at the bottom and the top is closed.
    difference() {
        translate([-(u_width / 2 + u_wall), -(u_depth / 2 + u_wall), 0])
            cube([u_width + 2 * u_wall, u_depth + 2 * u_wall, u_height + u_top]);
        translate([-u_width / 2, -u_depth / 2, -1])
            cube([u_width, u_depth, u_height + 1]);
    }
}

// The whole cavity rotates with the lever: orientation is applied to the body, not only to a
// neck, so the opening's walls stay parallel to the measured lever faces.
module u_oriented() {
    rotate([u_pitch, 0, 0]) rotate([0, 0, u_yaw]) rotate([0, u_lean, 0]) children();
}

module u_neck() {
    // Positively captured neck: a printed yoke joint between the U and the rod eye. Printed
    // on its own, bed-zero, with no placement offset (an assembly offset belongs in the
    // assembly scene, and put the printed piece below the bed).
    rotate([0, 0, 90]) printed_clevis(gap = clevis_gap_mm, ear_t = clevis_ear_t_mm,
                                      reach = 16);
}

module u_joint_pin() { printed_pin(2 * clevis_ear_t_mm + clevis_gap_mm + 2 * keeper_t_mm); }

module u_retainer() { printed_keeper(); }

// ---- Public entry points -----------------------------------------------------------
u_part = "body";

module throttle_u_fitting(part = u_part) {
    if (part == "body") {
        u_oriented() u_body();
    } else if (part == "neck") {
        u_neck();
    } else if (part == "pin") {
        u_joint_pin();
    } else if (part == "retainer") {
        u_retainer();
    } else if (part == "gauge") {
        // Opening gauge: the void is exactly the specified INSIDE opening, through the full
        // height so the check is a genuine pass/fail slip test.
        difference() {
            translate([-(u_width / 2 + 6), -(u_depth / 2 + 6), 0])
                cube([u_width + 12, u_depth + 12, 12]);
            translate([-u_width / 2, -u_depth / 2, -1])
                cube([u_width, u_depth, 14]);
        }
    } else if (part == "probe_lever") {
        // Reference lever section, used by the probes and the assembly view.
        translate([-u_width / 2, -u_depth / 2, -20]) cube([u_width, u_depth, 60]);
    } else if (part == "probe_oversize") {
        translate([-u_width / 2 - 1.5, -u_depth / 2 - 1.5, -20])
            cube([u_width + 3, u_depth + 3, 60]);
    } else {
        assert(false, str("unknown U part: ", part));
    }
}

throttle_u_fitting(u_part);
