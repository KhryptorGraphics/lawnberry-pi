// MG995 throttle servo mount for the Toro 77603 RIGHT arm head: cradle walls, keeper, pins.
//
// STRUCTURE
// The throttle servo stands BELOW the raised right steering servo, on a shelf grown from the
// rib of the derivative head (throttle_arm_head.scad). The head prints the shelf AND the
// cradle walls as one piece: the shelf's top face is the cradle's floor, and the servo's base
// rests on it. THIS file supplies the cradle walls (built into the head), the keeper that
// bears on the servo's ears, its two transverse printed pins with split rings, the lug gauge,
// and the world datums the scenes and probes place everything with.
//
// Nothing here modifies the steering servo, its six-hole fixture, its pushrods or its steel
// hardware, and no purchased fastener is used: the keeper is retained by printed pins.
//
// All dimensions derive from the legacy datum in ../enclosure_common.scad via parameters.scad.
// MG995 body, lug and horn dimensions remain TO MEASURE, so the cradle and keeper are
// capacities, not verified fits, and nothing here is load-qualified.
include <parameters.scad>
use <printed_joints.scad>   // pin_band_relief: the cross pins' lip and keeper groove

// ---- Datums ----------------------------------------------------------------------------
// The cradle frame's origin in WORLD coordinates: the servo's base plane at the cradle's
// centre. It CONVERTS the head-local seat. It must not re-derive the station from
// servo_station_x: that path crosses the land-face offset, which once put the seat and the
// part it carries 22.25 mm apart.
function throttle_servo_base_mm() =
    [arm_joint_x + throttle_seat_x_local_mm,
     assembly_arm_y + throttle_seat_y_local_mm,
     arm_joint_z + throttle_seat_local_mm];
// The horn's underside: the top of the servo's body.
function throttle_horn_base_mm() =
    throttle_servo_base_mm() + [0, 0, mg995_body_height_mm];

// ---- Cradle walls ----------------------------------------------------------------------
// Side and end walls in the cradle frame. They reach `root` below the base plane so the head
// fuses them into its shelf. The body recess between them is the servo's drop-in pocket;
// lug size and shape remain TO MEASURE.
module cradle_walls(root = 1) {
    difference() {
        union() {
            for (s = [-1, 1]) {
                translate([s < 0 ? -cradle_w_mm / 2 : cradle_w_mm / 2 - cradle_wall_mm,
                           -cradle_len_mm / 2, -root])
                    cube([cradle_wall_mm, cradle_len_mm, cradle_wall_h_mm + root]);
                translate([-cradle_w_mm / 2,
                           s < 0 ? -cradle_len_mm / 2 : cradle_len_mm / 2 - cradle_wall_mm,
                           -root])
                    cube([cradle_w_mm, cradle_wall_mm, cradle_wall_h_mm + root]);
            }
        }
        // Transverse keeper-pin bores through both side walls, beyond the servo's ends.
        for (s = [-1, 1])
            translate([0, s * keeper_cross_pin_y_mm, keeper_cross_pin_z_mm])
                rotate([0, 90, 0])
                    cylinder(d = cross_hole_d_mm, h = cradle_w_mm + 2, center = true);
        // Blind holes in the side-wall tops for the keeper's vertical locating pins.
        for (dx = [-keeper_pin_x_offset_mm, keeper_pin_x_offset_mm])
            translate([dx, keeper_pin_y_offset_mm, cradle_wall_h_mm - keeper_pin_depth_mm])
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
                    cylinder(d = cross_hole_d_mm, h = 2 * cradle_w_mm, center = true);
    }
}

// A printed cross pin and split keeper ring. The pin's free end (z = 0) carries a full-
// diameter lip, then the keeper groove; the ring's bore is smaller than both the shaft and
// the lip, so once snapped into the groove it cannot leave either way. Fit, snap force,
// clip retention and fatigue life are unverified.
module cross_pin(len) {
    difference() {
        union() {
            cylinder(d = cross_pin_d_mm, h = len);
            translate([0, 0, len - 2])
                cylinder(d = cross_ring_d_mm, h = 2);
        }
        pin_band_relief(d = cross_pin_d_mm, band_d = cross_band_d_mm,
                        band_t = cross_groove_t_mm, lip_t = cross_lip_t_mm);
    }
}

module cross_keeper() {
    difference() {
        cylinder(d = cross_ring_d_mm, h = cross_ring_t_mm);
        translate([0, 0, -1])
            cylinder(d = cross_ring_bore_d_mm, h = cross_ring_t_mm + 2, $fn = 48);
        translate([0, -0.5, -1])
            cube([cross_ring_d_mm, 1, cross_ring_t_mm + 2]);
    }
}

// Installed keeper pins and rings, in the cradle frame.
module keeper_cross_pins_installed() {
    for (s = [-1, 1])
        translate([keeper_cross_pin_x0_mm, s * keeper_cross_pin_y_mm, keeper_cross_pin_z_mm])
            rotate([0, 90, 0]) {
                cross_pin(keeper_cross_pin_len_mm);
                translate([0, 0, cross_ring_seat_mm]) cross_keeper();
            }
}

// The keeper as installed, in the cradle frame.
module keeper_installed() {
    translate([0, 0, cradle_flange_z1_mm + keeper_t_mm])
        rotate([180, 0, 0]) keeper_crossbar();
}

// ---- Placement ----------------------------------------------------------------------
// The keeper and its pins, in world coordinates. The cradle walls belong to the head.
module throttle_stack_installed() {
    translate(throttle_servo_base_mm()) {
        keeper_installed();
        keeper_cross_pins_installed();
    }
}

// ---- Entry points --------------------------------------------------------------------
mount_part = "assembly";

module mg995_throttle_mount(part = mount_part) {
    if (part == "assembly") {
        throttle_stack_installed();
    } else if (part == "cradle_walls") {
        cradle_walls();
    } else if (part == "keeper") {
        keeper_crossbar();
    } else if (part == "keeper_installed") {
        keeper_installed();
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
