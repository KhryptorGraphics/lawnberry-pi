// Lap-bar drive servo -> frame mount.
//
// Clamps to a round frame tube beside the operator station and presents a
// flat, slotted face that the servo's supplied U-bracket bolts to. Print
// TWO of these: side = 1 (right) and side = -1 (left, mirrored).
//
// The servo drives the lap bar FORE/AFT ONLY. It deliberately does not
// reach the outboard PARK position -- per the Toro manual, PARK (levers
// swung outboard) is both the parking brake and the engine-crank
// interlock, and it stays a manual, human action. Keep the pushrod
// geometry clear of the outboard swing; see lapbar_pushrod_clamp.scad.
//
//   openscad -o servo_frame_mount_right.stl -D side=1  servo_frame_mount.scad
//   openscad -o servo_frame_mount_left.stl  -D side=-1 servo_frame_mount.scad

include <common.scad>

side          = 1;    // 1 = right, -1 = left (mirrored)
clamp_width   = 40;   // along the frame tube
plate_l       = 78;   // servo mounting face length
plate_w       = 46;   // servo mounting face width
plate_t       = 8;    // face thickness
standoff      = 26;   // tube centre -> servo face; MEASURE and adjust
slot_travel   = 8;    // adjustment range in the slots

module servo_face() {
    difference() {
        translate([0, 0, -plate_t / 2])
            hull() {
                for (x = [-1, 1], y = [-1, 1])
                    translate([x * (plate_l / 2 - 6), y * (plate_w / 2 - 6), 0])
                        cylinder(h = plate_t, r = 6);
            }
        // Slotted bolt holes for the servo's U-bracket
        for (x = [-1, 1], y = [-1, 1])
            translate([x * servo_bracket_hole_spacing_x / 2,
                       y * servo_bracket_hole_spacing_y / 2,
                       -plate_t / 2 - 1])
                slot(m5_clear, slot_travel, plate_t + 2);
        // Weight/heat relief under the servo body
        translate([0, 0, -plate_t / 2 - 1])
            slot(16, plate_l * 0.35, plate_t + 2);
    }
}

module gusset() {
    // Triangular rib tying the face back to the clamp barrel.
    r_out = frame_tube_od / 2 + clamp_wall;
    translate([0, 0, 0])
        rotate([90, 0, 0])
            linear_extrude(height = wall, center = true)
                polygon([[r_out - 1, 0],
                         [standoff + plate_t / 2, 0],
                         [r_out - 1, plate_l / 2 - 4]]);
}

module servo_frame_mount() {
    // Barrel axis runs along Z; frame tube passes through it.
    union() {
        tube_clamp_half(frame_tube_od, clamp_width, half = 0);
        // Arm out to the servo face
        translate([standoff, 0, 0])
            rotate([0, 90, 0]) servo_face();
        // Two ribs, above and below centre
        for (z = [-1, 1])
            translate([0, 0, z * (clamp_width / 2 - wall / 2)])
                gusset();
        // Spine joining the ribs
        translate([(frame_tube_od / 2 + clamp_wall + standoff) / 2, 0, 0])
            cube([standoff - frame_tube_od / 2, wall * 1.6, clamp_width],
                 center = true);
    }
}

if (side > 0) servo_frame_mount();
else mirror([0, 1, 0]) servo_frame_mount();
