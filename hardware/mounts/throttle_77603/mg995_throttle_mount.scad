// Elevated, removable MG995 throttle mount for the Toro 77603 RIGHT-hand steering assembly.
//
// STRUCTURE
//   The head (throttle_arm_head.scad) carries ONE integral support that rises from the
//   servo-face land and reaches outboard above the retained steering linkage. THIS file
//   supplies only what stands on that support: the cradle that holds the MG995 and the keeper
//   that retains it. There is no separate foot, no riser and no second bracket: the earlier
//   column-mounted, duplicate-foot and cantilever revisions were all rejected by geometry
//   probes or by the print envelope.
//
//   Nothing here modifies the legacy steering arm, its six-hole servo fixture, its pushrods
//   or its steel hardware, and no purchased fastener is used: every joint below is a printed
//   cross-pin with a separate printed keeper.
//
// All dimensions derive from the legacy datum in ../enclosure_common.scad via parameters.scad.
// MG995 body, lug and horn dimensions remain TO MEASURE, so the cradle and keeper are
// capacities, not verified fits, and nothing here is load-qualified.
include <parameters.scad>
use <printed_joints.scad>

// ---- Datum ---------------------------------------------------------------------------
// The cradle floor rests on the column seat in world coordinates.
function support_top_world_z() = arm_joint_z + cradle_seat_local_mm;
// The world functions CONVERT the head-local datum. They must not re-derive the station
// from servo_station_x: that path crosses the land-face offset, while the seat datum does
// not, so the socket and the tongue ended up 22.25 mm apart -- the whole seat joint
// misaligned by exactly the land offset.
function tower_world_x() = arm_joint_x + seat_x_local_mm;
function tower_world_y() = assembly_arm_y + seat_y_local_mm;

// ---- Cradle --------------------------------------------------------------------------
// Holds the provisional MG995 body. Lug size and shape remain TO MEASURE.
// The underside dowels are horizontal in the bed-oriented export.
module cradle() {
    difference() {
        union() {
            translate([-cradle_w_mm / 2, -cradle_len_mm / 2, 0])
                cube([cradle_w_mm, cradle_len_mm, cradle_t_mm]);
            // Both locating dowels have a common transverse pin bore.
            for (dx = [-seat_dowel_x_offset_mm, seat_dowel_x_offset_mm])
                translate([dx, seat_dowel_y_offset_mm, -seat_dowel_depth_mm])
                    cylinder(d = seat_dowel_d_mm, h = seat_dowel_depth_mm + 1);
            for (s = [-1, 1]) {
                translate([s < 0 ? -cradle_w_mm / 2 : cradle_w_mm / 2 - cradle_wall_mm,
                           -cradle_len_mm / 2, cradle_t_mm])
                    cube([cradle_wall_mm, cradle_len_mm, cradle_wall_h_mm]);
                translate([-cradle_w_mm / 2,
                           s < 0 ? -cradle_len_mm / 2 : cradle_len_mm / 2 - cradle_wall_mm,
                           cradle_t_mm])
                    cube([cradle_w_mm, cradle_wall_mm, cradle_wall_h_mm]);
            }
        }
        // Body recess: the servo drops into it from above.
        translate([-mg995_body_width_mm / 2 - joint_clearance_mm,
                   -mg995_body_length_mm / 2 - joint_clearance_mm, cradle_t_mm - 0.01])
            cube([mg995_body_width_mm + 2 * joint_clearance_mm,
                  mg995_body_length_mm + 2 * joint_clearance_mm,
                  cradle_wall_h_mm + 1]);
        translate([0, seat_dowel_y_offset_mm, -seat_dowel_depth_mm / 2])
            rotate([0, 90, 0])
                cylinder(d = seat_cross_hole_d_mm, h = 2 * seat_cross_pin_len_mm,
                         center = true);
        // Transverse keeper-pin bores through both side walls, beyond the servo's ends.
        for (s = [-1, 1])
            translate([0, s * keeper_cross_pin_y_mm, keeper_cross_pin_z_mm])
                rotate([0, 90, 0])
                    cylinder(d = seat_cross_hole_d_mm, h = cradle_w_mm + 2,
                             center = true);
        // Blind holes in the side-wall tops for the keeper's vertical locating pins.
        for (dx = [-keeper_pin_x_offset_mm, keeper_pin_x_offset_mm])
            translate([dx, keeper_pin_y_offset_mm,
                       cradle_t_mm + cradle_wall_h_mm - keeper_pin_depth_mm])
                cylinder(d = keeper_pin_d_mm + 2 * keeper_pin_clearance_mm,
                         h = keeper_pin_depth_mm + 1);
    }
}

// ---- Keeper --------------------------------------------------------------------------
// An open frame bears on the provisional lugs from above. Its full-length legs drop
// outside the side walls and are pinned to them by two transverse printed pins.
module keeper_crossbar() {
    difference() {
        union() {
            translate([-cradle_w_mm / 2 - joint_clearance_mm - keeper_leg_t_mm,
                       -cradle_len_mm / 2, 0])
                cube([cradle_w_mm + 2 * (joint_clearance_mm + keeper_leg_t_mm),
                      cradle_len_mm, keeper_t_mm]);
            for (dx = [-keeper_pin_x_offset_mm, keeper_pin_x_offset_mm])
                translate([dx, keeper_pin_y_offset_mm, keeper_t_mm])
                    cylinder(d = keeper_pin_d_mm, h = keeper_pin_depth_mm);
            for (s = [-1, 1])
                translate([s < 0 ? -cradle_w_mm / 2 - joint_clearance_mm - keeper_leg_t_mm
                                 : cradle_w_mm / 2 + joint_clearance_mm,
                           -keeper_leg_drop_mm / 2, keeper_t_mm])
                    cube([keeper_leg_t_mm, keeper_leg_drop_mm, cradle_wall_h_mm]);
        }
        translate([-mg995_body_width_mm / 2 - joint_clearance_mm,
                   -mg995_body_length_mm / 2 - joint_clearance_mm, -1])
            cube([mg995_body_width_mm + 2 * joint_clearance_mm,
                  mg995_body_length_mm + 2 * joint_clearance_mm, keeper_t_mm + 2]);
        // Transverse pin bores through both legs, in the keeper's printed frame.
        for (s = [-1, 1])
            translate([0, s * keeper_cross_pin_y_mm,
                       cradle_flange_z1_mm + keeper_t_mm - keeper_cross_pin_z_mm])
                rotate([0, 90, 0])
                    cylinder(d = seat_cross_hole_d_mm, h = 2 * cradle_w_mm, center = true);
    }
}

// Printed form of the cradle. The locating dowels point DOWN in the assembly frame, so the
// exported part must be laid on its side: the rotation puts the dowels horizontal, which
// prints cleanly, and the floor plate vertical. The assembly uses cradle() directly.
module cradle_print() {
    translate([0, 0, cradle_len_mm / 2]) rotate([90, 0, 0]) cradle();
}

// A printed cross pin and split keeper ring. The ring's bore is smaller than the shaft, so it
// cannot pass back over it. Fit, clip retention and fatigue life are unverified.
module cross_pin(len) {
    union() {
        cylinder(d = seat_cross_band_d_mm, h = seat_cross_ring_t_mm + 0.5);
        translate([0, 0, seat_cross_ring_t_mm])
            cylinder(d = seat_cross_pin_d_mm, h = len - seat_cross_ring_t_mm);
        translate([0, 0, len - 2])
            cylinder(d = seat_cross_ring_d_mm, h = 2);
    }
}

module cross_keeper() {
    difference() {
        cylinder(d = seat_cross_ring_d_mm, h = seat_cross_ring_t_mm);
        translate([0, 0, -1])
            cylinder(d = seat_cross_ring_bore_d_mm, h = seat_cross_ring_t_mm + 2);
        translate([0, -0.5, -1])
            cube([seat_cross_ring_d_mm, 1, seat_cross_ring_t_mm + 2]);
    }
}

// Installed transverse pins and rings, in the cradle frame.
module seat_cross_pin_installed() {
    translate([seat_cross_pin_x0_mm, seat_dowel_y_offset_mm, -seat_dowel_depth_mm / 2])
        rotate([0, 90, 0]) {
            cross_pin(seat_cross_pin_len_mm);
            cross_keeper();
        }
}

module keeper_cross_pins_installed() {
    for (s = [-1, 1])
        translate([keeper_cross_pin_x0_mm, s * keeper_cross_pin_y_mm, keeper_cross_pin_z_mm])
            rotate([0, 90, 0]) {
                cross_pin(keeper_cross_pin_len_mm);
                cross_keeper();
            }
}

// The keeper as installed, in the cradle frame.
module keeper_installed() {
    translate([0, 0, cradle_flange_z1_mm + keeper_t_mm])
        rotate([180, 0, 0]) keeper_crossbar();
}

// ---- Public datums (used by the assembly scene) -------------------------------------
// The printed stack's origin: the cradle floor, on the head column's seat.
function throttle_mount_stack_origin_mm() =
    [tower_world_x(), tower_world_y(), support_top_world_z()];
// The MG995 body's base, sitting on the cradle floor.
function throttle_servo_base_mm() =
    throttle_mount_stack_origin_mm() + [0, 0, cradle_t_mm];
// The horn's underside: the top of the servo's body.
function throttle_horn_base_mm() =
    throttle_servo_base_mm() + [0, 0, mg995_body_height_mm];

// ---- Placement ----------------------------------------------------------------------
module throttle_stack_installed() {
    // Self-placed on the head's integral column seat, in world coordinates. The cradle sits
    // directly on the seat; there is no riser between them.
    translate([tower_world_x(), tower_world_y(), support_top_world_z()]) {
        cradle();
        seat_cross_pin_installed();
        keeper_installed();
        keeper_cross_pins_installed();
    }
}

// ---- Entry points --------------------------------------------------------------------
mount_part = "assembly";

module mg995_throttle_mount(part = mount_part) {
    if (part == "assembly") {
        throttle_stack_installed();
    } else if (part == "cradle") {
        cradle_print();
    } else if (part == "cradle_assembly") {
        cradle();
    } else if (part == "keeper") {
        keeper_crossbar();
    } else if (part == "keeper_installed") {
        keeper_installed();
    } else if (part == "seat_pin") {
        cross_pin(seat_cross_pin_len_mm);
    } else if (part == "seat_pin_installed") {
        seat_cross_pin_installed();
    } else if (part == "keeper_pin") {
        cross_pin(keeper_cross_pin_len_mm);
    } else if (part == "keeper_pins_installed") {
        keeper_cross_pins_installed();
    } else if (part == "pin_keeper") {
        cross_keeper();
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
