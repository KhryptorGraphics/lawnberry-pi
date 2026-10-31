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
//
// Joints: a printed neck pivots on the body's roof eye about a Y-axis pin (double shear
// between the neck's ears), and carries a vertical pivot post on its top bridge. The rod's
// throttle eye drops over that post and a printed cap retains it, so the two axes allow
// pitch and yaw. Fit, retention and wear are unverified.
include <parameters.scad>
use <printed_joints.scad>

u_depth = throttle_u_inner_depth_mm + fit_clearance_front_mm;  // Y, front-back
u_width = throttle_u_inner_width_mm + fit_clearance_side_mm;   // X, left-right
u_wall = throttle_u_wall_mm;
u_height = throttle_u_engagement_mm;
u_top = u_wall;                                               // closed top thickness
u_lean = throttle_lean_right_deg;
u_yaw = throttle_yaw_deg;
u_pitch = throttle_pitch_deg;

// The first hinge axis is parallel to Y. Its eye fits between the printed neck's ears.
function u_eye_thickness() = clevis_gap_mm - 2 * joint_clearance_mm;
function u_neck_base_z() = throttle_u_engagement_mm + throttle_u_wall_mm + joint_clearance_mm;
function u_first_pin_z() = u_neck_base_z() + clevis_ear_t_mm + clevis_gap_mm / 2;
function u_neck_height_mm() = 2 * clevis_ear_t_mm + clevis_gap_mm + 6;
// The rod eye's centre on the neck's pivot post, in the unoriented U frame.
function u_post_eye_z_mm() =
    u_neck_base_z() + u_neck_height_mm() + pivot_flange_t_mm + pivot_spigot_len_mm / 2;
// The same point after the lean/yaw/pitch transforms of u_oriented().
function u_post_offset_mm() =
    let(z = u_post_eye_z_mm(),
        x1 = z * sin(throttle_lean_right_deg),
        z1 = z * cos(throttle_lean_right_deg),
        x2 = x1 * cos(throttle_yaw_deg),
        y2 = x1 * sin(throttle_yaw_deg))
    [x2,
     y2 * cos(throttle_pitch_deg) - z1 * sin(throttle_pitch_deg),
     y2 * sin(throttle_pitch_deg) + z1 * cos(throttle_pitch_deg)];

module u_body() {
    difference() {
        union() {
            difference() {
                translate([-(u_width / 2 + u_wall), -(u_depth / 2 + u_wall), 0])
                    cube([u_width + 2 * u_wall, u_depth + 2 * u_wall, u_height + u_top]);
                translate([-u_width / 2, -u_depth / 2, -1])
                    cube([u_width, u_depth, u_height + 1]);
            }
            // Single eye grown into the closed roof; the lever cavity stays unobstructed.
            translate([-7, -u_eye_thickness() / 2, u_height + u_top - 0.5])
                cube([14, u_eye_thickness(), u_first_pin_z() + 6 - u_height - u_top + 0.5]);
        }
        translate([0, u_eye_thickness() / 2 + 1, u_first_pin_z()])
            rotate([90, 0, 0])
                cylinder(d = pin_d_mm + 2 * joint_clearance_mm, h = u_eye_thickness() + 2);
    }
}

// The whole cavity rotates with the lever: orientation is applied to the body, not only to a
// neck, so the opening's walls stay parallel to the measured lever faces.
module u_oriented() {
    rotate([u_pitch, 0, 0]) rotate([0, 0, u_yaw]) rotate([0, u_lean, 0]) children();
}

module u_neck() {
    // Lower fork accepts the U body's eye; the top bridge carries the vertical pivot post
    // for the rod's throttle eye. The two axes are perpendicular.
    gap = clevis_gap_mm;
    ear = clevis_ear_t_mm;
    width = gap + 2 * ear;
    height = u_neck_height_mm();
    difference() {
        union() {
            translate([-8, -width / 2, 0]) cube([16, ear, height]);
            translate([-8, gap / 2, 0]) cube([16, ear, height]);
            translate([-8, -width / 2, height - ear]) cube([16, width, ear]);
            // Pivot post: flange, running spigot, cap seat and a lip that traps the cap.
            translate([0, 0, height - 0.5])
                cylinder(d = pivot_flange_d_mm, h = pivot_flange_t_mm + 0.5);
            translate([0, 0, height + pivot_flange_t_mm])
                cylinder(d = pivot_spigot_d_mm, h = pivot_spigot_len_mm);
            translate([0, 0, height + pivot_flange_t_mm + pivot_spigot_len_mm])
                cylinder(d = pivot_cap_seat_d_mm, h = pivot_cap_seat_len_mm);
            translate([0, 0, height + pivot_flange_t_mm + pivot_spigot_len_mm
                       + pivot_cap_seat_len_mm])
                cylinder(d = pivot_lip_d_mm, h = pivot_lip_t_mm);
        }
        // Lower axis along Y.
        translate([0, width / 2 + 1, ear + gap / 2])
            rotate([90, 0, 0])
                cylinder(d = pin_d_mm + 2 * joint_clearance_mm, h = width + 2);
    }
}

// Printed cap: pressed over the lip onto the post's seat, it traps the rod's throttle eye.
module u_cap() {
    difference() {
        cylinder(d = pivot_cap_od_mm, h = pivot_cap_h_mm);
        translate([0, 0, -0.01]) cylinder(d = pivot_cap_id_mm, h = pivot_cap_h_mm + 0.02);
    }
}

module u_joint_pin() { printed_pin(2 * clevis_ear_t_mm + clevis_gap_mm + 2 * keeper_t_mm); }

module u_retainer() { printed_keeper(); }

module u_assembly() {
    u_oriented() {
        u_body();
        translate([0, 0, u_neck_base_z()]) u_neck();
        translate([0, (clevis_gap_mm + 2 * clevis_ear_t_mm) / 2 + keeper_t_mm, u_first_pin_z()])
            rotate([90, 0, 0]) {
                u_joint_pin();
                u_retainer();
            }
        translate([0, 0, u_neck_base_z() + u_neck_height_mm() + pivot_flange_t_mm
                   + pivot_spigot_len_mm])
            u_cap();
    }
}

// ---- Public entry points -----------------------------------------------------------
u_part = "body";

module throttle_u_fitting(part = u_part) {
    if (part == "assembly") {
        u_assembly();
    } else if (part == "body") {
        // Print flat regardless of lever lean; the complete assembly applies the tilt.
        u_body();
    } else if (part == "neck") {
        u_neck();
    } else if (part == "pin") {
        u_joint_pin();
    } else if (part == "retainer") {
        u_retainer();
    } else if (part == "cap") {
        u_cap();
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
