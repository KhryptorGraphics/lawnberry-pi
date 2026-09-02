// Lap-bar drive servo -> frame mount.
//
// Seats against a face of the 3" x 1.5" rectangular frame rail and is held by
// two U-bolts wrapping the rail. A vertical face carries the servo's supplied
// U-bracket, braced by gussets. Print TWO: side = 1 (right), side = -1 (left).
//
// The servo drives the lap bar FORE/AFT ONLY. It deliberately cannot reach the
// outboard PARK position -- per the Toro manual, PARK is both the parking
// brake and a condition of the engine-crank interlock, and stays a manual
// human action. Keep the pushrod clear of the outboard swing.
//
//   openscad -o servo_frame_mount_right.stl -D side=1  servo_frame_mount.scad
//   openscad -o servo_frame_mount_left.stl  -D side=-1 servo_frame_mount.scad

include <common.scad>

side          = 1;     // 1 = right, -1 = left (mirrored)
saddle_l      = 130;   // along the frame rail
saddle_t      = 10;
ubolt_span    = 92;    // between U-bolt centres, along the rail
face_l        = 76;    // servo face, along the rail
face_h        = 62;    // servo face, standing up
face_t        = 9;
gusset_reach  = 30;    // how far the gussets brace back

// Vertical plate standing in the XZ plane, thickness along Y, rising from Z=0.
module servo_face() {
    difference() {
        translate([0, 0, face_h / 2])
            cube([face_l, face_t, face_h], center = true);
        // Servo U-bracket bolt slots, through Y, slotted along X
        for (x = [-1, 1], z = [-1, 1])
            translate([x * servo_bracket_hole_spacing_x / 2,
                       face_t / 2 + 1,
                       face_h / 2 + z * servo_bracket_hole_spacing_y / 2])
                rotate([90, 0, 0])
                    slot(m5_clear, 8, face_t + 2);
        // Heat / weight relief behind the servo body
        translate([0, face_t / 2 + 1, face_h / 2])
            rotate([90, 0, 0])
                slot(15, face_l * 0.28, face_t + 2);
    }
}

// Triangular gusset in the YZ plane, extruded `wall` thick along X.
module gusset() {
    rotate([0, -90, 0])
        linear_extrude(height = wall)
            polygon([[0, 0], [face_h * 0.72, 0], [0, gusset_reach]]);
}

module servo_frame_mount() {
    difference() {
        union() {
            frame_saddle(saddle_l, saddle_t, ubolt_span);
            translate([0, 0, saddle_t]) servo_face();
            // Gussets behind the face, inboard of its ends
            for (x = [-1, 1])
                translate([x * (face_l / 2 - 8) + wall / 2,
                           face_t / 2, saddle_t])
                    gusset();
        }
        frame_ubolt_slots(saddle_t, ubolt_span);
    }
}

if (side > 0) servo_frame_mount();
else mirror([1, 0, 0]) servo_frame_mount();
