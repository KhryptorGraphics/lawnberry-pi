// Elevated, removable MG995 throttle mount for the Toro 77603 RIGHT-hand steering assembly.
//
// STRUCTURE (this is the correction that matters — see README "revision note"):
// The owner requirement is to *vertically extend and structurally reinforce the existing
// RIGHT STEERING SERVO BRACKET* to carry an independent throttle MG995. An earlier revision
// hung the throttle off the lower arm COLUMN (arm_root_x, world Z ~ -66..286), which put the
// throttle cradle inside the right steering servo's own case envelope and could not be fixed
// by lengthening the riser: the column rakes outboard as Z rises, so a taller riser drove the
// cradle further into the servo case in X.
//
// This revision anchors to the existing steering servo FACE instead (world X ~ servo_station_x,
// Z ~ arm_tip_z) and extends UPWARD and OUTBOARD, so the throttle servo sits above the
// steering servo's case top and outboard of its horn sweep.
//
// Nothing here modifies the legacy steering arm, its six-hole servo fixture, its pushrods or
// its steel hardware. The saddle is still a printed split wrap, but it wraps the arm HEAD
// band around the servo face rather than the lower column.
//
// All dimensions derive from the legacy datum in ../enclosure_common.scad. No approximate
// world coordinate is hard-coded except the derived clearance offsets declared in
// parameters.scad, all of which are marked as DERIVED rather than measured.
include <parameters.scad>

// ---- Datum -------------------------------------------------------------------------
// Steering servo face, from the legacy measured station.
function steering_face_x() = servo_station_x;
function steering_face_y() = servo_shaft_y();
function steering_face_z() = arm_tip_z;

// Throttle servo base: above the steering servo case, outboard in X.
function throttle_servo_base_mm() =
    [steering_face_x() + mg995_body_width_mm / 2 + throttle_outboard_mm,
     steering_face_y(),
     steering_face_z() + throttle_above_servo_mm];

// ---- Saddle: wraps the arm head band at the servo face ------------------------------
// The head is arm_col_d x arm_col_w at this station, so the wrap uses the same section
// logic as before but referenced to the head, not the column.
module saddle_profile_2d(half = 1) {
    x_in = saddle_x_in();
    y_out = saddle_y_out();
    outer_x0 = -x_in;
    outer_x1 = x_in + saddle_back_t_mm;
    outer_y0 = -y_out - saddle_side_w_mm;
    outer_y1 = y_out + saddle_side_w_mm;
    difference() {
        union() {
            translate([outer_x0, outer_y0])
                square([outer_x1 - outer_x0, outer_y1 - outer_y0]);
            translate([x_in - saddle_ledge_w_mm, outer_y0])
                square([saddle_ledge_w_mm, outer_y1 - outer_y0]);
        }
        square([2 * x_in, 2 * y_out], center = true);
        translate([-x_in - 1, half > 0 ? -y_out - saddle_side_w_mm - 1 : y_out])
            square([2 * x_in + 2, y_out + saddle_side_w_mm + 1]);
        translate([-x_in - 1, -y_out])
            square([2 * x_in + 2, 2 * y_out]);
    }
}

function saddle_x_in() = arm_col_d / 2 + saddle_skin_clear_mm;
function saddle_y_out() = arm_col_w / 2 + saddle_side_clear_mm;

module saddle_segment(half = 1, seg = 0, h_seg = 50) {
    // Printed piece: always built up from z=0 so it is bed-zero on its own.
    difference() {
        linear_extrude(height = h_seg - saddle_split_gap_mm)
            saddle_profile_2d(half = half);
        translate([0, 0, h_seg * 0.42])
            rotate([90, 0, 0]) cylinder(d = saddle_vent_relief_d_mm,
                                        h = arm_col_d + 2 * saddle_back_t_mm + 4,
                                        center = true);
        translate([arm_col_d / 2 + saddle_back_t_mm - saddle_dovetail_h_mm / 2,
                   -saddle_dovetail_w_mm / 2, h_seg * 0.2])
            cube([saddle_dovetail_h_mm, saddle_dovetail_w_mm, h_seg * 0.6]);
    }
}

module saddle_flange_hook() {
    // Printed flat on the bed.
    cube([saddle_flange_overhang_mm + saddle_back_t_mm, saddle_dovetail_w_mm,
          saddle_dovetail_h_mm]);
}

// ---- Riser and cradle, anchored above the steering servo ---------------------------
module riser_segment(k = 0) {
    h = riser_z_mm / 2;
    difference() {
        union() {
            difference() {
                translate([-riser_w_mm / 2, -riser_d_mm / 2, 0])
                    cube([riser_w_mm, riser_d_mm, h - 0.2]);
                translate([-riser_w_mm / 2 + riser_wall_mm,
                           -riser_d_mm / 2 + riser_wall_mm, -1])
                    cube([riser_w_mm - 2 * riser_wall_mm, riser_d_mm - 2 * riser_wall_mm,
                          h + 2]);
            }
            for (s = [-1, 1])
                translate([0, s * riser_d_mm / 2, 0]) rotate([90, 0, 0])
                    linear_extrude(height = 4)
                        polygon([[-riser_w_mm / 2, 0], [riser_w_mm / 2, 0], [0, h * 0.8]]);
        }
        translate([-riser_w_mm / 2 - 1, 0, h * 0.75]) rotate([0, 90, 0])
            cylinder(d = keeper_t_mm, h = riser_w_mm + 2, center = true);
    }
}

function mg995_dovetail_w() = 8;

module cradle() {
    wall_h = mg995_body_height_mm * 0.55;
    difference() {
        union() {
            translate([-cradle_w_mm / 2, -cradle_len_mm / 2, 0])
                cube([cradle_w_mm, cradle_len_mm, cradle_t_mm]);
            for (s = [-1, 1]) {
                translate([s * (cradle_w_mm / 2 - cradle_wall_mm), -cradle_len_mm / 2,
                           cradle_t_mm])
                    cube([cradle_wall_mm, cradle_len_mm, wall_h]);
                translate([-cradle_w_mm / 2, s * cradle_len_mm / 2
                           - (s > 0 ? cradle_wall_mm : 0), cradle_t_mm])
                    cube([cradle_w_mm, cradle_wall_mm, wall_h]);
            }
        }
        translate([-mg995_body_width_mm / 2 - joint_clearance_mm,
                   -mg995_body_length_mm / 2 - joint_clearance_mm, cradle_t_mm - 0.01])
            cube([mg995_body_width_mm + 2 * joint_clearance_mm,
                  mg995_body_length_mm + 2 * joint_clearance_mm, wall_h + 1]);
        translate([0, -cradle_len_mm / 2 - 1, cradle_t_mm + wall_h * 0.5])
            rotate([90, 0, 0]) cylinder(d = 2 * x_horn_arm_radius_mm + 6,
                                        h = cradle_wall_mm + 2, center = true);
        for (s = [-1, 1])
            translate([s * (cradle_w_mm / 2 - cradle_wall_mm) - 0.01,
                       -cradle_len_mm / 2, cradle_t_mm + wall_h - keeper_t_mm])
                cube([cradle_wall_mm + 0.02, cradle_len_mm, keeper_t_mm + 0.4]);
    }
}

module keeper_crossbar() {
    difference() {
        translate([-cradle_w_mm / 2, -cradle_len_mm / 2, 0])
            cube([cradle_w_mm, cradle_len_mm, keeper_t_mm]);
        for (s = [-1, 1])
            translate([-cradle_w_mm / 2 - 1, s * (cradle_len_mm / 2 - 10) - 3,
                       keeper_t_mm * 0.45])
                cube([cradle_w_mm + 2, 6, keeper_t_mm + 1]);
    }
}

module mg995_body_envelope() {
    // Reference solid only: the purchased servo, never printed.
    translate([-mg995_body_width_mm / 2, -mg995_body_length_mm / 2, 0])
        cube([mg995_body_width_mm, mg995_body_length_mm, mg995_body_height_mm]);
    translate([mg995_shaft_xy_mm[0], mg995_shaft_xy_mm[1], mg995_body_height_mm])
        cylinder(d = 6, h = mg995_shaft_projection_mm);
}

// ---- Placement ---------------------------------------------------------------------
// Installed (assembly) transforms vs printed (bed-zero) transforms are kept entirely
// separate: reusing a print transform in the assembly silently moved parts metres.
module throttle_saddle_installed(half = 1, segment = 0) {
    h_seg = riser_z_mm / 2;
    translate([0, 0, segment * h_seg]) saddle_segment(half = half, seg = segment,
                                                      h_seg = h_seg);
}

// Printed foot: bridges from the outboard tower back to the steering-bracket servo land.
module bracket_foot() {
    // Foot: reaches inboard onto the head land and outboard to the tower axis, in the Y band
    // that clears the steering servo case. A foot centred on the shaft centre would pass
    // straight through that case (Y 108..138).
    difference() {
        // ONE narrow arm in a Y band clear of the servo case, from the arm-head flange
        // outboard to the tower axis. A wider pad would overlap the case: the servo land and
        // the case are both centred on the shaft, so the clear attachment band is the head
        // flange edge (Y 138..158), not the shaft centreline.
        translate([-foot_reach_mm, tower_offset_y - post_w_mm / 2, 0])
            cube([foot_reach_mm + tower_offset_x + riser_w_mm / 2, post_w_mm, foot_t_mm]);
        // Bolt passages through the arm-head flange band (photo-derived pattern family).
        for (dx = [14, 40])
            translate([-foot_reach_mm + dx, tower_offset_y, -1])
                cylinder(d = servo_mount_clear_d_mm, h = foot_t_mm + 2);
    }
}

module throttle_stack_installed() {
    // Local origin = the servo land. The foot sits ON the land, the riser rises from it, and
    // the cradle/tower is offset outboard so the MG995 clears the steering case and horn.
    // (An earlier revision put the foot at the BOTTOM of a 240 mm stack, i.e. 120 mm below
    // the servo face it was supposed to bolt to -- physically unbuildable.)
    bracket_foot();
    // Tower post sits in a Y band clear of the steering servo case (see tower_offset_y).
    translate([tower_offset_x, tower_offset_y, 0]) {
        translate([0, 0, foot_t_mm]) riser_segment(0);
        translate([0, 0, foot_t_mm + riser_z_mm / 2]) riser_segment(1);
        translate([0, 0, foot_t_mm + riser_z_mm]) cradle();
        translate([0, 0, foot_t_mm + riser_z_mm + mg995_body_height_mm
                   + mg995_lug_thickness_mm]) keeper_crossbar();
    }
}

// ---- Entry points ------------------------------------------------------------------
mount_part = "assembly";
riser_index = 0;

module mg995_throttle_mount(part = mount_part) {
    if (part == "assembly") {
        // Anchored ON the steering servo land, extending up and outboard from it.
        translate([servo_station_x, servo_shaft_y(), arm_tip_z])
            throttle_stack_installed();
    } else if (part == "saddle_left") {
        saddle_segment(half = -1, seg = 0, h_seg = riser_z_mm / 2);
    } else if (part == "saddle_right") {
        saddle_segment(half = 1, seg = 0, h_seg = riser_z_mm / 2);
    } else if (part == "saddle_left_upper") {
        saddle_segment(half = -1, seg = 1, h_seg = riser_z_mm / 2);
    } else if (part == "saddle_right_upper") {
        saddle_segment(half = 1, seg = 1, h_seg = riser_z_mm / 2);
    } else if (part == "bracket_foot") {
        bracket_foot();
    } else if (part == "flange_hook") {
        saddle_flange_hook();
    } else if (part == "riser_lower") {
        riser_segment(riser_index);
    } else if (part == "riser_upper") {
        riser_segment(riser_index);
    } else if (part == "cradle") {
        cradle();
    } else if (part == "keeper") {
        keeper_crossbar();
    } else if (part == "gauge") {
        difference() {
            translate([-30, -mg995_mount_pitch_fb_mm / 2 - 8, 0]) cube([60, 30, 3]);
            for (s = [-1, 1])
                translate([0, s * mg995_mount_pitch_fb_mm / 2, -1])
                    cylinder(d = mg995_mount_hole_d_mm, h = 5);
        }
    } else {
        assert(false, str("unknown mount part: ", part));
    }
}

mg995_throttle_mount(mount_part);
