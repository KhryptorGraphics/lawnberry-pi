// Throttle-end inverted-U capture, printed in place with its pitch yoke, plus the opening gauge.
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
// Joints, printed in place as ONE print of two captive bodies (no pin, no keeper):
//   * pitch, about Y: a yoke straddles the body. Its two legs run down the body's front and
//     back faces to the bed, and a stub pin on each leg sits in a blind bore in a hinge boss
//     on the body's roof. Both bodies stand on the bed, so neither starts in mid-air.
//   * yaw, about Z: the yoke's bridge carries a vertical pivot post. The rod's throttle eye
//     drops over it at installation and a separately printed cap retains it; that joint
//     must come apart to fit the rod, so it is not printed in place.
// Print-in-place rules followed (checked by pip_gate.py on the exported mesh):
//   * every gap is joint_clearance_mm (0.4) per side;
//   * the stub pins have a 45-degree V underside and the bores a 45-degree teardrop roof,
//     so neither prints a flat overhang over the other;
//   * the boss top is a half-cylinder about the hinge axis and the bridge clears it by
//     u_bridge_gap_mm, so the yoke pitches through +/-joint_articulation_deg without contact.
// Fit, wear and how far a given printer keeps the joint free are coupon-test items.
include <parameters.scad>

u_depth = throttle_u_inner_depth_mm + fit_clearance_front_mm;  // Y, front-back
u_width = throttle_u_inner_width_mm + fit_clearance_side_mm;   // X, left-right
u_wall = throttle_u_wall_mm;
u_height = throttle_u_engagement_mm;
u_top = u_wall;                                               // closed top thickness
u_lean = throttle_lean_right_deg;
u_yaw = throttle_yaw_deg;
u_pitch = throttle_pitch_deg;

u_body_y = u_depth / 2 + u_wall;          // body's front and back faces at +/- this
u_gap = joint_clearance_mm;                // every print-in-place gap
u_pin_r = pin_d_mm / 2;                    // stub pin radius
u_bore_r = u_pin_r + u_gap;                // bore radius below the hinge axis
u_boss_half_x = 7;                         // hinge boss half width, along X
u_bore_depth = 6;                          // blind bore depth from each face
u_leg_t = 5;                               // yoke leg thickness, along Y
u_bridge_t = 4;                            // yoke bridge thickness
u_bridge_gap_mm = 1.5;                     // bridge underside over the boss's round top
// Hinge axis: the bore's lowest point stays 1.6 mm above the roof.
u_axis_z = u_height + u_top + u_bore_r + 1.6;
// The boss top is a half-cylinder about the axis, 2 mm outside the teardrop apex.
u_boss_r = u_bore_r * sqrt(2) + 2;
u_bridge_z0 = u_axis_z + u_boss_r + u_bridge_gap_mm;
u_bridge_z1 = u_bridge_z0 + u_bridge_t;
u_leg_y0 = u_body_y + u_gap;               // leg inner face
u_leg_y1 = u_leg_y0 + u_leg_t;             // leg outer face

// The rod eye's centre on the yoke's pivot post, in the unoriented U frame.
function u_post_eye_z_mm() = u_bridge_z1 + pivot_flange_t_mm + pivot_spigot_len_mm / 2;
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
// The hinge axis height, for the probes and the assembly view.
function u_axis_z_mm() = u_axis_z;

// Stub pin cross-section in the XZ plane, centred on the axis: a semicircle on top and a
// 90-degree V below, so its underside is 45 degrees and never a flat overhang.
module u_pin_profile() {
    union() {
        intersection() {
            circle(r = u_pin_r);
            translate([-u_pin_r - 1, 0]) square([2 * u_pin_r + 2, u_pin_r + 1]);
        }
        polygon([[-u_pin_r, 0], [u_pin_r, 0], [0, -u_pin_r]]);
    }
}

// Bore cross-section: the clearance circle with a 45-degree teardrop roof, so the material
// above the bore is self-supporting.
module u_bore_profile() {
    hull() {
        circle(r = u_bore_r);
        translate([0, u_bore_r * sqrt(2) - 0.01]) square(0.02, center = true);
    }
}

module u_body() {
    difference() {
        union() {
            difference() {
                translate([-(u_width / 2 + u_wall), -u_body_y, 0])
                    cube([u_width + 2 * u_wall, 2 * u_body_y, u_height + u_top]);
                translate([-u_width / 2, -u_depth / 2, -1])
                    cube([u_width, u_depth, u_height + 1]);
            }
            // Hinge boss across the full depth, rounded about the axis on top.
            intersection() {
                translate([-u_boss_half_x, -u_body_y, u_height + u_top - 0.5])
                    cube([2 * u_boss_half_x, 2 * u_body_y, u_axis_z - u_height - u_top + 0.5
                          + u_boss_r]);
                union() {
                    translate([-u_boss_half_x, -u_body_y, u_height + u_top - 0.5])
                        cube([2 * u_boss_half_x, 2 * u_body_y,
                              u_axis_z - u_height - u_top + 0.5]);
                    translate([0, u_body_y + 1, u_axis_z]) rotate([90, 0, 0])
                        cylinder(r = u_boss_r, h = 2 * u_body_y + 2);
                }
            }
        }
        // A blind teardrop bore from each face; the web between them stays solid.
        for (s = [-1, 1])
            // mirror(), not a -90 rotation, on -Y: a rotation would turn the teardrop over.
            mirror([0, s < 0 ? 1 : 0, 0])
                translate([0, u_body_y + 1, u_axis_z]) rotate([90, 0, 0])
                    linear_extrude(height = u_bore_depth + 1) u_bore_profile();
    }
}

// The pitch yoke: two legs standing on the bed, a stub pin on each leg's inner face, and a
// bridge over the boss carrying the vertical pivot post for the rod's throttle eye.
module u_yoke() {
    union() {
        for (s = [-1, 1]) {
            // Leg, from the bed to the bridge.
            translate([-u_boss_half_x, s > 0 ? u_leg_y0 : -u_leg_y1, 0])
                cube([2 * u_boss_half_x, u_leg_t, u_bridge_z1]);
            // Stub pin, from inside the leg to 0.4 mm short of the bore's blind end.
            mirror([0, s < 0 ? 1 : 0, 0])
                translate([0, u_leg_y0 + 1, u_axis_z]) rotate([90, 0, 0])
                    linear_extrude(height = 1 + u_bore_depth) u_pin_profile();
        }
        // Bridge across the legs, over the boss.
        translate([-u_boss_half_x, -u_leg_y1, u_bridge_z0])
            cube([2 * u_boss_half_x, 2 * u_leg_y1, u_bridge_t]);
        // Pivot post: flange, running spigot, cap seat and a lip that traps the cap.
        translate([0, 0, u_bridge_z1 - 0.5])
            cylinder(d = pivot_flange_d_mm, h = pivot_flange_t_mm + 0.5);
        translate([0, 0, u_bridge_z1 + pivot_flange_t_mm])
            cylinder(d = pivot_spigot_d_mm, h = pivot_spigot_len_mm);
        translate([0, 0, u_bridge_z1 + pivot_flange_t_mm + pivot_spigot_len_mm])
            cylinder(d = pivot_cap_seat_d_mm, h = pivot_cap_seat_len_mm);
        translate([0, 0, u_bridge_z1 + pivot_flange_t_mm + pivot_spigot_len_mm
                   + pivot_cap_seat_len_mm])
            cylinder(d = pivot_lip_d_mm, h = pivot_lip_t_mm);
    }
}

// The yoke pitched about the hinge axis, for the range probes.
module u_yoke_pitched(angle) {
    translate([0, 0, u_axis_z]) rotate([0, angle, 0]) translate([0, 0, -u_axis_z]) u_yoke();
}

// Printed cap: pressed over the lip onto the post's seat, it traps the rod's throttle eye.
module u_cap() {
    difference() {
        cylinder(d = pivot_cap_od_mm, h = pivot_cap_h_mm);
        translate([0, 0, -0.01]) cylinder(d = pivot_cap_id_mm, h = pivot_cap_h_mm + 0.02);
    }
}

// The whole cavity rotates with the lever: orientation is applied to the body, not only to a
// yoke, so the opening's walls stay parallel to the measured lever faces.
module u_oriented() {
    rotate([u_pitch, 0, 0]) rotate([0, 0, u_yaw]) rotate([0, u_lean, 0]) children();
}

module u_assembly() {
    u_oriented() {
        u_body();
        u_yoke();
        translate([0, 0, u_bridge_z1 + pivot_flange_t_mm + pivot_spigot_len_mm]) u_cap();
    }
}

// ---- Public entry points -----------------------------------------------------------
u_part = "fitting";

module throttle_u_fitting(part = u_part) {
    if (part == "assembly") {
        u_assembly();
    } else if (part == "fitting") {
        // The print: body and yoke as printed, both on the bed. Print flat regardless of
        // lever lean; the complete assembly applies the tilt.
        u_body();
        u_yoke();
    } else if (part == "body") {
        u_body();
    } else if (part == "yoke") {
        u_yoke();
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
