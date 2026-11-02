// MG995 throttle servo mount for the Toro 77603 RIGHT arm head: cradle walls and ONE snap
// keeper.
//
// STRUCTURE
// The throttle servo stands BELOW the raised right steering servo, on a shelf grown from the
// rib of the derivative head (throttle_arm_head.scad). The head prints the shelf AND the
// cradle walls as one piece: the shelf's top face is the cradle's floor, and the servo's base
// rests on it. THIS file supplies the cradle walls (built into the head), the keeper that
// holds the servo's ears down, the lug gauge, and the world datums the scenes and probes
// place everything with.
//
// THE KEEPER IS ONE PART. A frame bears on the MG995's ears from above; four spring fingers
// drop beside the cradle's side walls, and each finger's barb snaps into a window in the
// wall's outer face, under the window's flat roof. Two crush ribs per ear row stand 0.4 mm
// proud of the frame's underside: the keeper must be pressed down past them for the barbs to
// pass the roofs, so once seated the ribs keep the ears pressed onto the wall tops instead of
// leaving the servo free to rattle in the catch play. Service: pry the four finger tails
// outward (fingernail or a small flat blade between tail and wall) and lift the keeper off;
// then the MG995 lifts straight out (servo_lifts_out). The previous keeper needed a crossbar,
// two transverse printed pins and two split rings: five loose parts, now one.
//
// Nothing here modifies the steering servo, its six-hole fixture, its pushrods or its steel
// hardware, and no purchased fastener is used.
//
// All dimensions derive from the legacy datum in ../enclosure_common.scad via parameters.scad.
// MG995 body, lug and horn dimensions remain TO MEASURE, so the cradle and keeper are
// capacities, not verified fits; snap force, rib crush, hold-down against vibration and
// fatigue are unqualified.
include <parameters.scad>

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
// lug size and shape remain TO MEASURE. Each side wall carries two snap WINDOWS in its outer
// face: a pocket keeper_window_depth_mm deep whose flat roof is the barb's catch.
module keeper_window(sx, sy) {
    translate([sx > 0 ? cradle_w_mm / 2 - keeper_window_depth_mm : -cradle_w_mm / 2 - 1,
               sy * keeper_finger_y_mm - keeper_finger_w_mm / 2 - joint_clearance_mm,
               keeper_barb_z0_mm - joint_clearance_mm])
        cube([keeper_window_depth_mm + 1, keeper_finger_w_mm + 2 * joint_clearance_mm,
              keeper_window_roof_mm - keeper_barb_z0_mm + joint_clearance_mm]);
}

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
        for (sx = [-1, 1], sy = [-1, 1]) keeper_window(sx, sy);
    }
}

// ---- The MG995's provisional flange --------------------------------------------------
// Four ears at the owner's 50.80 mm front-back pitch, reaching mg995_flange_ear_out_mm to
// each side and resting on the side-wall tops. DESIGN CAPACITY for the keeper probes: the
// real flange outline is TO MEASURE. Cradle frame.
module mg995_flange_envelope() {
    for (sy = [-1, 1])
        translate([-mg995_flange_ear_out_mm, sy * mg995_flange_ear_offset_mm
                   - mg995_flange_ear_w_mm / 2, cradle_flange_z0_mm])
            cube([2 * mg995_flange_ear_out_mm, mg995_flange_ear_w_mm, mg995_lug_thickness_mm]);
}

// ---- Keeper --------------------------------------------------------------------------
// Built in the INSTALLED pose, cradle frame: the frame's underside lies on the provisional
// ear tops (cradle_flange_z1_mm) and the fingers hang down beside the side walls.
// `spread` opens every finger outward by that much at its barb, rotating it about its root
// at the frame's underside: the pose the snap probes slide down the walls.
keeper_frame_x_mm = keeper_finger_x_mm + keeper_finger_t_mm;   // the frame's half width
keeper_finger_z1_mm = cradle_flange_z1_mm;                       // the fingers' root
keeper_tail_z_mm = keeper_barb_z0_mm - keeper_finger_tail_mm;   // the fingers' tips

module keeper_frame() {
    difference() {
        translate([-keeper_frame_x_mm, -cradle_len_mm / 2, cradle_flange_z1_mm])
            cube([2 * keeper_frame_x_mm, cradle_len_mm, keeper_t_mm]);
        // The servo body passes up through the frame.
        translate([-mg995_body_width_mm / 2 - joint_clearance_mm,
                   -mg995_body_length_mm / 2 - joint_clearance_mm, cradle_flange_z1_mm - 1])
            cube([mg995_body_width_mm + 2 * joint_clearance_mm,
                  mg995_body_length_mm + 2 * joint_clearance_mm, keeper_t_mm + 2]);
    }
    // Crush ribs over each ear row: a 45-degree ridge rib_h tall along X on each side of the
    // body, centred on the ear row.
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx > 0 ? keeper_rib_x0_mm : -mg995_flange_ear_out_mm + 0.2,
                   sy * mg995_flange_ear_offset_mm, cradle_flange_z1_mm + 0.01])
            rotate([0, 90, 0])
                linear_extrude(height = mg995_flange_ear_out_mm - 0.2 - keeper_rib_x0_mm)
                    polygon([[0, -keeper_rib_h_mm], [keeper_rib_h_mm + 0.01, 0],
                             [0, keeper_rib_h_mm]]);
}

// One finger with its barb, on side sx (+1 / -1 in X), at Y centre sy * finger_y.
module keeper_finger(sx, sy) {
    y0 = sy * keeper_finger_y_mm - keeper_finger_w_mm / 2;
    // The finger plate, from its tail up into the frame.
    translate([sx > 0 ? keeper_finger_x_mm : -keeper_frame_x_mm, y0, keeper_tail_z_mm])
        cube([keeper_finger_t_mm, keeper_finger_w_mm,
              keeper_finger_z1_mm - keeper_tail_z_mm + 0.5]);
    // The barb: a 45-degree lead-in from its lowest point to the tip, a land, then the flat
    // catch face at keeper_catch_z_mm (facing up, under the window's roof).
    translate([0, y0, 0]) rotate([90, 0, 0]) mirror([0, 0, 1])
        linear_extrude(height = keeper_finger_w_mm)
            polygon([for (p = [[keeper_finger_x_mm + 0.01, keeper_barb_z0_mm],
                               [keeper_finger_x_mm - keeper_barb_d_mm,
                                keeper_barb_z0_mm + keeper_barb_d_mm],
                               [keeper_finger_x_mm - keeper_barb_d_mm, keeper_catch_z_mm],
                               [keeper_finger_x_mm + 0.01, keeper_catch_z_mm]])
                         [sx * p[0], p[1]]]);
}

module keeper_installed(spread = 0) {
    arm = keeper_finger_z1_mm - (keeper_barb_z0_mm + keeper_catch_z_mm) / 2;
    angle = spread == 0 ? 0 : asin(spread / arm);
    keeper_frame();
    for (sx = [-1, 1], sy = [-1, 1])
        // Each finger opens about its root line at the frame's underside, inner face.
        translate([sx * keeper_finger_x_mm, 0, keeper_finger_z1_mm])
            rotate([0, -sx * angle, 0])
                translate([-sx * keeper_finger_x_mm, 0, -keeper_finger_z1_mm])
                    keeper_finger(sx, sy);
}

// The keeper's print pose: frame on the bed, fingers up. The barbs' 1 mm catch faces are the
// only downward faces off the bed.
module keeper_print() {
    translate([0, 0, cradle_flange_z1_mm + keeper_t_mm]) rotate([180, 0, 0])
        keeper_installed();
}

// ---- Placement ----------------------------------------------------------------------
// The keeper in world coordinates. The cradle walls belong to the head.
module throttle_stack_installed() {
    translate(throttle_servo_base_mm()) keeper_installed();
}

// ---- Entry points --------------------------------------------------------------------
mount_part = "assembly";

module mg995_throttle_mount(part = mount_part) {
    if (part == "assembly" || part == "stack_installed") {
        throttle_stack_installed();
    } else if (part == "cradle_walls") {
        cradle_walls();
    } else if (part == "keeper") {
        keeper_print();
    } else if (part == "keeper_installed") {
        keeper_installed();
    } else if (part == "flange_envelope") {
        mg995_flange_envelope();
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
